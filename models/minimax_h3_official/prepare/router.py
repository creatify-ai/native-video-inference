"""Ahead-of-time Triton compile of the SubBlock attention router kernels.

The kernels are SGLang's unmodified `subblock_sparse/kernels.py` (Apache-2.0) at a pinned
commit, fetched and checked by SHA-256; nothing is vendored or edited. Each launch scalar keeps
the specialization class Triton's JIT would give it (==1 becomes a constant, a multiple of
16 gets a divisibility hint); every other value stays a runtime argument. One CUBIN is compiled
per class reachable for sequences up to MAX_SEQUENCE. Launch options match SGLang's own calls.
Output: launch.tsv (H3_DYNAMIC_ROUTER_V1, read by transformer/dynamic_router.rs), the CUBINs
and PTX, and manifest.json with every CUBIN's SHA-256. Compilation needs no GPU.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import urllib.request

SGLANG_COMMIT = 'ad7a9b53ccad9d53599cb872233e2cedee1fe154'
KERNELS_PATH = 'python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse/kernels.py'
KERNELS_URL = f'https://raw.githubusercontent.com/sgl-project/sglang/{SGLANG_COMMIT}/{KERNELS_PATH}'
KERNELS_SHA256 = '06cc4f445476b4467e0b043751c15a5068aae7cb16a7717d6b9cec986ccc292e'
TRITON = '3.8.0'
MAX_SEQUENCE = 262144
HEADS, DIM, CELLS = 56, 128, 4
SCALARS = {
    'pool': ['stride_xb', 'stride_xt', 'stride_xh', 'stride_yl', 'stride_yn', 'S', 'H'],
    'scores': ['stride_qm', 'stride_ql', 'stride_kn', 'stride_kl', 'stride_om', 'stride_ol',
               'M', 'M_VALID', 'N_VALID', 'NOUT', 'MOUT'],
    'topk': ['G', 'K'],
}
POINTERS = {'pool_q': {'X': 'q', 'Y': 'qp'}, 'pool_k': {'X': 'k', 'Y': 'kp'},
            'scores': {'Q': 'qp', 'K': 'kp', 'O': 'scores'}, 'topk': {'S': 'scores', 'OUT': 'indices'}}


def plan_values(sequence):
    """Launch scalars of one [1, sequence, 56, 128] request at sparsity 0.75."""
    g = (sequence + 63) // 64
    kept = min(g, ((g + 3) // 4 + 7) // 8 * 8)
    return {
        'pool': dict(stride_xb=sequence * HEADS * DIM, stride_xt=HEADS * DIM, stride_xh=DIM,
                     stride_yl=g * CELLS * DIM, stride_yn=DIM, S=sequence, H=HEADS),
        'scores': dict(stride_qm=DIM, stride_ql=g * CELLS * DIM, stride_kn=DIM, stride_kl=g * CELLS * DIM,
                       stride_om=g, stride_ol=g * g, M=g * CELLS, M_VALID=(sequence + 15) // 16,
                       N_VALID=(sequence + 15) // 16, NOUT=g, MOUT=g),
        'topk': dict(G=g, K=kept),
    }


def specialization(values, keys):
    return ''.join('o' if values[k] == 1 else 'a' if values[k] % 16 == 0 else 'x' for k in keys)


def topk_launch(g):
    """SGLang's fused_topk: BLK=next_power_of_2(G), 4 warps from G>=1024, ITERS=16 at sparsity 0.75."""
    return 1 << (g - 1).bit_length(), 4 if g >= 1024 else 2


def variant_inventory(maximum=MAX_SEQUENCE):
    variants = {}
    for sequence in range(1, maximum + 1):
        values = plan_values(sequence)
        for family in ('pool', 'scores', 'topk'):
            key = specialization(values[family], SCALARS[family])
            if family == 'topk':
                block, warps = topk_launch(values['topk']['G'])
                key = f'b{block}-w{warps}-{key}'
            variants.setdefault((family, key), sequence)
    return variants


def fetch_kernels(destination):
    destination = Path(destination)
    if not destination.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(KERNELS_URL, timeout=60) as response:
            destination.write_bytes(response.read())
    if hashlib.sha256(destination.read_bytes()).hexdigest() != KERNELS_SHA256:
        raise ValueError('SGLang router kernels differ from the pinned source')
    return destination


def load_kernels(path):
    spec = importlib.util.spec_from_file_location('h3_router_reference_kernels', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parameters(ptx, symbol):
    entry = re.search(r'\.entry\s+' + re.escape(symbol) + r'\s*\((.*?)\)\s*', ptx, re.S)
    if not entry:
        raise ValueError('compiled PTX has no entry ' + symbol)
    result = []
    for declaration in entry.group(1).split(','):
        kind = re.match(r'\s*\.param\s+(?:\.align\s+\d+\s+)?\.(\w+)\b', declaration)
        name = re.search(r'\b([A-Za-z_]\w*)(?:\[(\d+)\])?\s*$', declaration)
        if not kind or not name or name.group(2):
            raise ValueError('unsupported router kernel parameter: ' + declaration)
        result.append({'kind': kind.group(1), 'name': name.group(1)})
    return result


def launch_arguments(family, arg_names, signature, constants, params):
    args = []
    for name in arg_names:
        if name in constants:
            continue
        kind = signature[name]
        if kind.startswith('*'):
            args.append(('ptr', POINTERS[family][name]))
        elif kind in ('i32', 'fp32'):
            args.append(('dynamic', name))
        else:
            raise ValueError(f'unsupported router argument {name}: {kind}')
    extra = len(params) - len(args)
    if not 0 <= extra <= 2:
        raise ValueError('unexpected compiled router ABI for ' + family)
    args.extend(('bits', '0') for _ in range(extra))  # Triton's null scratch pointers
    for (encoding, _), param in zip(args, params):
        if encoding == 'ptr' and param['kind'] not in ('u64', 'b64'):
            raise ValueError('router pointer argument is not 64-bit')
    return args


def compile_router(output, kernels_source, maximum=MAX_SEQUENCE):
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    if triton.__version__ != TRITON:
        raise ValueError(f'router export requires triton=={TRITON}')
    root = Path(output) / 'router'
    root.mkdir(parents=True, exist_ok=False)
    kernels = load_kernels(fetch_kernels(kernels_source))
    records, lines = [], ['H3_DYNAMIC_ROUTER_V1', f'max_sequence\t{maximum}']

    def emit(label, jit, signature, constants, warps, stages, scalar_values):
        signature, constants = dict(signature), dict(constants)
        for name, value in scalar_values.items():
            if value == 1:
                constants[name] = 1
                signature.pop(name)
        attrs = {(i,): [['tt.divisibility', 16]] for i, name in enumerate(jit.arg_names)
                 if signature.get(name, '').startswith('*')
                 or (name in scalar_values and name not in constants and scalar_values[name] % 16 == 0)}
        kernel = triton.compile(ASTSource(jit, signature=signature, constexprs=constants, attrs=attrs),
                                target=GPUTarget('cuda', 100, 32), options={'num_warps': warps, 'num_stages': stages})
        meta = kernel.metadata._asdict()
        if meta['num_warps'] != warps or meta.get('num_ctas', 1) != 1 or any(
                meta.get(key, 0) for key in ('global_scratch_size', 'profile_scratch_size')) or meta.get(
                'launch_cooperative_grid') or meta.get('launch_pdl'):
            raise ValueError('router kernel needs an unsupported launch: ' + label)
        cubin, ptx = root / (label + '.cubin'), root / (label + '.ptx')
        cubin.write_bytes(kernel.asm['cubin'])
        ptx.write_text(kernel.asm['ptx'])
        params = parameters(kernel.asm['ptx'], meta['name'])
        args = launch_arguments(label.split('-')[0], jit.arg_names, signature, constants, params)
        lines.append('\t'.join(map(str, ('kernel', label, cubin.name, meta['name'], warps * 32, meta['shared']))))
        lines.extend(f'arg\t{encoding}\t{value}' for encoding, value in args)
        records.append({'label': label, 'signature': signature, 'constants': constants, 'attrs': str(attrs),
                        'arguments': args, 'num_warps': warps, 'num_stages': stages, 'shared': meta['shared'],
                        'cubin_sha256': hashlib.sha256(cubin.read_bytes()).hexdigest(),
                        'ptx_sha256': hashlib.sha256(ptx.read_bytes()).hexdigest()})

    pool_signature = {name: '*bf16' if name in ('X', 'Y') else 'fp32' if name == 'SCALE' else 'i32'
                      for name in kernels._pool_kernel.arg_names if name not in ('SUB', 'D')}
    score_constants = {'stride_on': 1, 'BLK_M': 128, 'BLK_N': 128, 'NK': CELLS, 'NQR': CELLS, 'D': DIM}
    score_signature = {name: '*bf16' if name in ('Q', 'K') else '*fp32' if name == 'O' else 'i32'
                       for name in kernels._score_kernel.arg_names if name not in score_constants}
    inventory = variant_inventory(maximum)
    for (family, key), sequence in inventory.items():
        values = plan_values(sequence)[family]
        if family == 'pool':
            for label in ('pool_q', 'pool_k'):
                emit(f'{label}-{key}', kernels._pool_kernel, pool_signature, {'SUB': 16, 'D': DIM}, 1, 3, values)
        elif family == 'scores':
            emit(f'scores-{key}', kernels._score_kernel, score_signature, score_constants, 4, 3, values)
        else:
            block, warps = topk_launch(values['G'])
            emit(f'topk-{key}', kernels._topk_kernel, {'S': '*fp32', 'OUT': '*i32', 'G': 'i32', 'K': 'i32'},
                 {'BLK': block, 'ITERS': 16}, warps, 3, values)
    (root / 'launch.tsv').write_text('\n'.join(lines) + '\n')
    manifest = {'schema': 'h3-dynamic-router-aot-v1', 'max_sequence': maximum, 'triton': triton.__version__,
                'kernels_source': {'repository': 'sgl-project/sglang', 'commit': SGLANG_COMMIT,
                                   'path': KERNELS_PATH, 'sha256': KERNELS_SHA256},
                'variants': [{'family': f, 'key': k, 'representative_sequence': s} for (f, k), s in inventory.items()],
                'records': records,
                'launch_sha256': hashlib.sha256((root / 'launch.tsv').read_bytes()).hexdigest(),
                'contract': {'batch': 1, 'heads': HEADS, 'dim': DIM, 'n_q': CELLS, 'n_k': CELLS, 'sparsity': 0.75}}
    sha = (json.dumps(manifest, indent=2) + '\n').encode()
    (root / 'manifest.json').write_bytes(sha)
    return {'component': 'router', 'root': str(root), 'kernels': len(records),
            'manifest_sha256': hashlib.sha256(sha).hexdigest(), 'launch_sha256': manifest['launch_sha256']}
