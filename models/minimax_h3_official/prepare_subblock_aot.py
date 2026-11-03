"""Export and validate the unmodified public FlashInfer SubBlock AOT kernel.

    python models/minimax_h3_official/prepare_subblock_aot.py export /path/to/new-export
    python models/minimax_h3_official/prepare_subblock_aot.py check /path/to/export

`export` runs once on a B200 with torch (CUDA), flashinfer-python 0.7.0.post1 and
nvidia-cutlass-dsl 4.8.0. It compiles FlashInfer's SM100 blk64 block-sparse attention
kernel ahead of time without modifying it. `build.py` links the result after `check`
confirms the sources, configuration, header, runtime library and the object's GPU payload
sections (`.lrodata`, `.ldata`) match subblock-v07-source.json.
Inference uses only the exported object and runtime library, never PyTorch.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct

PIN = Path(__file__).with_name('subblock-v07-source.json')
NAME = 'h3_subblock_v07'
FILES = (NAME + '.o', NAME + '.h', 'libcute_dsl_runtime.so')


def elf_sections(path, names):
    """{name: sha256} of named sections in a little-endian ELF64 relocatable object."""
    raw = Path(path).read_bytes()
    if raw[:4] != b'\x7fELF' or raw[4] != 2 or raw[5] != 1:
        raise ValueError('SubBlock object is not a little-endian ELF64 file')
    shoff, = struct.unpack_from('<Q', raw, 0x28)
    entry, count, names_index = struct.unpack_from('<HHH', raw, 0x3A)
    headers = [struct.unpack_from('<IIQQQQ', raw, shoff + i * entry) for i in range(count)]
    table = headers[names_index]
    strings = raw[table[4]:table[4] + table[5]]
    found = {}
    for name, kind, _, _, offset, size in headers:
        label = strings[name:strings.index(b'\0', name)].decode()
        if label in names:
            if label in found or kind == 8:  # duplicate, or SHT_NOBITS with no file bytes
                raise ValueError('SubBlock object has an unexpected section ' + label)
            found[label] = hashlib.sha256(raw[offset:offset + size]).hexdigest()
    return found


def validate_export(root, arch):
    root = Path(root).resolve(strict=True)
    report = json.loads((root / 'export.json').read_bytes())
    pin = json.loads(PIN.read_bytes())
    for key in ('schema', 'flashinfer', 'dsl_version', 'arch', 'vendor_sources_sha256', 'kernel_modified',
                'variable_block_counts', 'has_block_sizes', 'kv_splits', 'clc'):
        if report.get(key) != pin[key]:
            raise ValueError('public SubBlock export differs in ' + key)
    if arch != '100a' or report['arch'] != 'sm_100a':
        raise ValueError('this public SubBlock export requires B200')
    # Whole files: the header and the runtime library. Older exports also listed the object.
    whole = {name: sha for name, sha in report.get('sha256', {}).items() if name != NAME + '.o'}
    if whole != pin['sha256']:
        raise ValueError('public SubBlock export differs in sha256')
    for name, expected in whole.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError('public SubBlock AOT file changed: ' + name)
    # Host launcher code varies with the build CPU; the GPU payload is the pinned kernel.
    payload = elf_sections(root / (NAME + '.o'), set(pin['gpu_payload_sha256']))
    if payload != pin['gpu_payload_sha256'] or report.get('gpu_payload_sha256', payload) != payload:
        raise ValueError('public SubBlock GPU payload changed: ' + NAME + '.o')
    return report


def _sources(base):
    return {str(p.relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (base / 'cute_dsl/sparse').rglob('*.py') if '__pycache__' not in p.parts}


def export(root):
    """Compile the pinned kernel on the current B200 and write it under a new `root`."""
    import importlib.metadata
    import torch
    import flashinfer
    import cutlass.cute as cute
    import cuda.bindings.driver as cuda
    from flashinfer.cute_dsl.sparse.sm100_blk64.bsa_fwd_sm100 import BlockSparseAttnForwardSm100Blk64
    from flashinfer.cute_dsl.sparse.sm100_blk64.dispatch_helpers import (
        make_sm100_blk64_cute_args, workaround_cutlass_hash_import_bug)
    pin = json.loads(PIN.read_bytes())
    if (flashinfer.__version__ != pin['flashinfer']
            or importlib.metadata.version('nvidia-cutlass-dsl') != pin['dsl_version']):
        raise ValueError('SubBlock export requires flashinfer-python 0.7.0.post1 and nvidia-cutlass-dsl 4.8.0')
    base = Path(flashinfer.__file__).parent
    if _sources(base) != pin['vendor_sources_sha256']:
        raise ValueError('FlashInfer source differs from the pinned reference')
    if torch.cuda.get_device_capability() != (10, 0):
        raise ValueError('SubBlock export requires a B200 (sm_100)')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    # Placeholder tensors fix the compiled signature; their values are never read.
    S, H, D = 21743, 56, 128
    G = (S + 63) // 64
    q = torch.empty((1, S, H, D), device='cuda', dtype=torch.bfloat16)
    k, v = torch.empty_like(q), torch.empty_like(q)
    o = torch.empty((1, H, S, D), device='cuda', dtype=torch.bfloat16)
    lse = torch.empty((1, H, S), device='cuda', dtype=torch.float32)
    index = torch.empty((1, H, G, G), device='cuda', dtype=torch.int32)
    sizes = torch.full((G,), 64, device='cuda', dtype=torch.int32)
    counts = torch.full((1, H, G), 88, device='cuda', dtype=torch.int32)
    args = make_sm100_blk64_cute_args(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), o, lse, None, None, None, D ** -.5,
        index, sizes, 0, counts, None, cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        enable_tvm_ffi=False)
    workaround_cutlass_hash_import_bug()
    kernel = BlockSparseAttnForwardSm100Blk64(
        D, D, qhead_per_kvhead=1, pack_gqa=False, m_block_size=64, n_block_size=256, sparse_block_size=64,
        is_persistent=True, use_clc_scheduler=True, allow_empty_block_nums=False, has_block_sizes=True,
        num_splits=1, use_int64_kv_strides=False)
    compiled = cute.compile(kernel, *args, options='--gpu-arch sm_100a')
    compiled.export_to_c(str(root), NAME)
    libraries = cute.runtime.find_runtime_libraries(enable_tvm_ffi=False)
    if len(libraries) != 1:
        raise ValueError('expected one CuTe DSL runtime library')
    shutil.copy2(libraries[0], root / 'libcute_dsl_runtime.so')
    report = dict(schema=pin['schema'], flashinfer=flashinfer.__version__, dsl_version=pin['dsl_version'],
                  gpu=torch.cuda.get_device_name(), arch='sm_100a', vendor_sources_sha256=_sources(base),
                  kernel_modified=False, variable_block_counts=True, has_block_sizes=True, kv_splits=1, clc=True,
                  sha256={n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in FILES if n != NAME + '.o'},
                  gpu_payload_sha256=elf_sections(root / (NAME + '.o'), set(pin['gpu_payload_sha256'])),
                  object_sha256_informational={NAME + '.o': hashlib.sha256((root / (NAME + '.o')).read_bytes()).hexdigest()},
                  header=(root / (NAME + '.h')).read_text(), gpu_accepted=False)
    (root / 'export.json').write_text(json.dumps(report, indent=2) + '\n')
    return validate_export(root, '100a')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('export', 'check'))
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    report = export(args.root) if args.command == 'export' else validate_export(args.root, '100a')
    print(json.dumps({k: report.get(k) for k in ('schema', 'flashinfer', 'arch', 'sha256', 'gpu_payload_sha256')}, indent=2))
