"""Checkpoint manifests for in-place native loading (schema ``native-checkpoint-v1``).

A model's public checkpoint is loaded as published: nothing is rewritten. Preparation reads
every file once and records, for each tensor, its byte range in the file, dtype, shape,
SHA-256 and first/last 1 KiB SHA-256 (the trusted light guard), plus each file's size and
SHA-256. ``core_infra::checkpoint`` turns the ranges into bounded reader jobs.

Supported files: safetensors, and PyTorch zip checkpoints (``torch.save``) whose storages are
stored uncompressed, which is torch's default. Tensors must be contiguous. A ``.pth`` is parsed
with a restricted unpickler (no code runs, no tensor data is materialized, torch is not needed).

Preparation may also write a runtime-dtype copy of tensors stored wider than the model runs them
(``convert``: fp32 -> bf16 with torch's round-to-nearest-even, so the copy equals
``tensor.to(torch.bfloat16)`` bit for bit). The copy is sharded safetensors with a
``<prefix>.safetensors.index.json``, the layout ``from_pretrained`` loaders read, so every framework
in a cold-start comparison can load the same bytes.

    python -m core_infra.runtime.checkpoint convert --root <checkpoint dir> --file '*.safetensors' \\
        --out <prepared>/dit-bf16 [--copy config.json]
    python -m core_infra.runtime.checkpoint manifest --root <checkpoint dir> --out <prepared>/manifest.json \\
        --file 'models_t5_*.pth@t5.' [--extra-root prepared=<dir> --extra-file 'prepared:dit-bf16/*.safetensors@dit.']
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import fnmatch
import hashlib
import json
import os
import pickle
import struct
import shutil
import zipfile
from pathlib import Path

SCHEMA = 'native-checkpoint-v1'
SEGMENT = 1024
READ_BYTES = 64 << 20
ITEMSIZE = {'F64': 8, 'F32': 4, 'F16': 2, 'BF16': 2, 'I64': 8, 'I32': 4, 'I16': 2, 'I8': 1, 'U8': 1, 'BOOL': 1}
STORAGE_DTYPE = {'DoubleStorage': 'F64', 'FloatStorage': 'F32', 'HalfStorage': 'F16', 'BFloat16Storage': 'BF16',
                 'LongStorage': 'I64', 'IntStorage': 'I32', 'ShortStorage': 'I16', 'CharStorage': 'I8',
                 'ByteStorage': 'U8', 'BoolStorage': 'BOOL'}


def safetensors_tensors(path):
    """{name: (offset, bytes, dtype, shape)} from a safetensors header."""
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n))
    base = 8 + n
    out = {}
    for name, v in header.items():
        if name == '__metadata__':
            continue
        start, end = v['data_offsets']
        if v['dtype'] not in ITEMSIZE:
            raise ValueError(f'{path}: {name}: unsupported dtype {v["dtype"]}')
        out[name] = (base + start, end - start, v['dtype'], list(v['shape']))
    return out


class _Storage:
    def __init__(self, dtype, key, numel):
        self.dtype, self.key, self.numel = dtype, key, numel


class _Tensor:
    def __init__(self, storage, offset, shape, stride):
        self.storage, self.offset, self.shape, self.stride = storage, offset, list(shape), list(stride)


def _rebuild_tensor(storage, storage_offset, size, stride, *_):
    return _Tensor(storage, storage_offset, size, stride)


def _rebuild_parameter(tensor, *_):
    return tensor


class _Marker:
    def __init__(self, name):
        self.name = name


class _Unpickler(pickle.Unpickler):
    """torch.save's pickle with every class resolved to an inert record."""

    def find_class(self, module, name):
        if (module, name) == ('collections', 'OrderedDict'):
            return collections.OrderedDict
        if (module, name) == ('torch._utils', '_rebuild_tensor_v2'):
            return _rebuild_tensor
        if (module, name) == ('torch._utils', '_rebuild_parameter'):
            return _rebuild_parameter
        if module == 'torch' and name in STORAGE_DTYPE:
            return _Marker(name)
        raise pickle.UnpicklingError(f'unsupported pickle global {module}.{name}')

    def persistent_load(self, pid):
        kind, storage_type, key, _location, numel = pid
        if kind != 'storage':
            raise pickle.UnpicklingError(f'unsupported persistent id {kind}')
        name = storage_type.name if isinstance(storage_type, _Marker) else str(storage_type)
        return _Storage(STORAGE_DTYPE[name], str(key), numel)


def _flatten(obj, prefix=''):
    if isinstance(obj, _Tensor):
        yield prefix, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flatten(v, f'{prefix}.{k}' if prefix else str(k))
    else:
        raise ValueError(f'unsupported checkpoint value at {prefix!r}: {type(obj).__name__}')


def _data_offset(f, info):
    """File offset of a stored zip entry's payload (local header + name + extra)."""
    f.seek(info.header_offset)
    local = f.read(30)
    if local[:4] != b'PK\x03\x04':
        raise ValueError('bad zip local header')
    name_len, extra_len = struct.unpack('<HH', local[26:30])
    return info.header_offset + 30 + name_len + extra_len


def pth_tensors(path):
    """{name: (offset, bytes, dtype, shape)} of a torch.save zip checkpoint."""
    with zipfile.ZipFile(path) as z, open(path, 'rb') as f:
        pkl = [n for n in z.namelist() if n.endswith('/data.pkl')]
        if len(pkl) != 1:
            raise ValueError(f'{path}: expected one data.pkl')
        prefix = pkl[0][:-len('data.pkl')]
        state = _Unpickler(z.open(pkl[0])).load()
        infos = {i.filename: i for i in z.infolist()}
        out = {}
        for name, t in _flatten(state):
            info = infos[f'{prefix}data/{t.storage.key}']
            if info.compress_type != zipfile.ZIP_STORED:
                raise ValueError(f'{path}: {name}: compressed storage')
            size = ITEMSIZE[t.storage.dtype]
            expected, step = [], 1
            for dim in reversed(t.shape):
                expected.insert(0, step)
                step *= dim
            if any(d != 1 and s != e for d, s, e in zip(t.shape, t.stride, expected)):
                raise ValueError(f'{path}: {name}: non-contiguous tensor')
            numel = step
            if (t.offset + numel) > t.storage.numel or info.file_size != t.storage.numel * size:
                raise ValueError(f'{path}: {name}: tensor exceeds its storage')
            out[name] = (_data_offset(f, info) + t.offset * size, numel * size, t.storage.dtype, t.shape)
        return out


def tensors_of(path):
    return safetensors_tensors(path) if path.suffix == '.safetensors' else pth_tensors(path)


def hash_file(path, tensors):
    """Whole-file SHA-256 and, per tensor, SHA-256 of its range and of its first/last 1 KiB, in one pass."""
    file_hash = hashlib.sha256()
    ranges = sorted((off, off + n, name) for name, (off, n, _, _) in tensors.items())
    hashes = {name: hashlib.sha256() for name in tensors}
    heads = {name: bytearray() for name in tensors}
    tails = {name: bytearray() for name in tensors}
    active, i, pos = [], 0, 0
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(READ_BYTES)
            if not chunk:
                break
            end = pos + len(chunk)
            file_hash.update(chunk)
            while i < len(ranges) and ranges[i][0] < end:
                active.append(ranges[i])
                i += 1
            still = []
            for start, stop, name in active:
                lo, hi = max(start, pos), min(stop, end)
                if lo < hi:
                    part = chunk[lo - pos:hi - pos]
                    hashes[name].update(part)
                    if lo - start < SEGMENT:
                        heads[name] += part[:SEGMENT - (lo - start)]
                    if hi > stop - SEGMENT:
                        tails[name] += part[max(0, stop - SEGMENT - lo):]
                if stop > end:
                    still.append((start, stop, name))
            active = still
            pos = end
    return file_hash.hexdigest(), {
        name: (hashes[name].hexdigest(), hashlib.sha256(bytes(heads[name])).hexdigest(),
               hashlib.sha256(bytes(tails[name][-SEGMENT:])).hexdigest()) for name in tensors}


def build(roots, files, workers=16):
    """roots: {root name: dir}; files: [(root name, relative path[, tensor name prefix])]. Returns the
    manifest dict. Tensor names (after their file's prefix) must be unique across files."""
    def one(item):
        root, rel = item[0], item[1]
        prefix = item[2] if len(item) > 2 else ''
        path = Path(roots[root]) / rel
        tensors = tensors_of(path)
        digest, per = hash_file(path, tensors)
        real = Path(os.path.realpath(path))
        if len(real.name) == 64 and real.parent.name == 'blobs' and real.name != digest:
            raise ValueError(f'{path}: content SHA-256 differs from its Hugging Face blob name')
        return (root, rel, prefix), path.stat().st_size, digest, tensors, per
    manifest = {'schema': SCHEMA, 'segment_bytes': SEGMENT, 'files': {}, 'tensors': {}}
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for (root, rel, prefix), size, digest, tensors, per in pool.map(one, files):
            key = f'{root}:{rel}'
            manifest['files'][key] = {'root': root, 'path': rel, 'bytes': size, 'sha256': digest}
            for name, (offset, nbytes, dtype, shape) in tensors.items():
                if prefix + name in manifest['tensors']:
                    raise ValueError(f'tensor {prefix + name} appears in more than one file')
                sha, head, tail = per[name]
                manifest['tensors'][prefix + name] = {'file': key, 'offset': offset, 'bytes': nbytes, 'dtype': dtype,
                                                      'shape': shape, 'sha256': sha, 'head_sha256': head,
                                                      'tail_sha256': tail}
    manifest['tensors'] = dict(sorted(manifest['tensors'].items()))
    return manifest


def write(manifest, out):
    """Writes the manifest; returns its SHA-256 (callers pin it)."""
    data = (json.dumps(manifest, indent=1, sort_keys=True) + '\n').encode()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp')
    tmp.write_bytes(data)
    os.replace(tmp, out)
    return hashlib.sha256(data).hexdigest()


def bf16_bits(raw, name=''):
    """fp32 little-endian bytes -> bf16 bytes as torch's .to(torch.bfloat16): round to nearest even.
    NaN is refused: torch's vectorized and scalar CPU paths encode it differently (0xFFFF / 0x7FC0)."""
    import numpy as np
    u = np.frombuffer(raw, dtype='<u4').astype(np.uint64)
    if ((u & 0x7FFFFFFF) > 0x7F800000).any():
        raise ValueError(f'{name}: NaN in a weight to convert')
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype('<u2').tobytes()


def convert(root, files, out, prefix='diffusion_pytorch_model', shard_bytes=5 << 30, copy=(), to_bf16=True):
    """Every F32 tensor of `files` (under `root`) as BF16, other dtypes unchanged, written as sharded
    safetensors in `out` with `<prefix>.safetensors.index.json`; `copy` files are copied alongside.
    With `to_bf16=False` every tensor keeps its stored bytes (a re-sharding only)."""
    tensors = {}
    for rel in files:
        for name, (offset, nbytes, dtype, shape) in tensors_of(Path(root) / rel).items():
            if name in tensors:
                raise ValueError(f'tensor {name} appears in more than one file')
            tensors[name] = (rel, offset, nbytes, dtype, shape)
    shards, current, size = [], [], 0
    for name in sorted(tensors):
        rel, offset, nbytes, dtype, shape = tensors[name]
        out_bytes = nbytes // 2 if dtype == 'F32' and to_bf16 else nbytes
        if current and size + out_bytes > shard_bytes:
            shards.append(current)
            current, size = [], 0
        current.append(name)
        size += out_bytes
    if current:
        shards.append(current)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    weight_map, total = {}, 0
    for i, names in enumerate(shards):
        shard = f'{prefix}-{i + 1:05d}-of-{len(shards):05d}.safetensors'
        header, pos = {}, 0
        for name in names:
            rel, offset, nbytes, dtype, shape = tensors[name]
            narrow = dtype == 'F32' and to_bf16
            n = nbytes // 2 if narrow else nbytes
            header[name] = {'dtype': 'BF16' if narrow else dtype, 'shape': shape, 'data_offsets': [pos, pos + n]}
            pos += n
        blob = json.dumps(header, separators=(',', ':')).encode()
        blob += b' ' * (-len(blob) % 8)
        tmp = out / (shard + '.tmp')
        with open(tmp, 'wb') as f:
            f.write(struct.pack('<Q', len(blob)))
            f.write(blob)
            handles = {}
            for name in names:
                rel, offset, nbytes, dtype, shape = tensors[name]
                src = handles.setdefault(rel, open(Path(root) / rel, 'rb'))
                src.seek(offset)
                raw = src.read(nbytes)
                f.write(bf16_bits(raw, name) if dtype == 'F32' and to_bf16 else raw)
            for h in handles.values():
                h.close()
        os.replace(tmp, out / shard)
        weight_map.update({name: shard for name in names})
        total += pos
    (out / f'{prefix}.safetensors.index.json').write_text(
        json.dumps({'metadata': {'total_size': total}, 'weight_map': weight_map}, indent=2) + '\n')
    for name in copy:
        shutil.copyfile(Path(root) / name, out / name)
    return {'out': str(out), 'shards': len(shards), 'bytes': total, 'tensors': len(tensors)}


def layout(manifest, roots, out, shard_bytes=64 << 20, workers=16):
    """A storage layout of `manifest` for ``NATIVE_WEIGHT_LAYOUT_*``: every tensor's stored bytes, checked
    against its SHA-256 on the way, copied in file order into safetensors files of at most `shard_bytes`
    (a larger tensor gets a file of its own). Names, bytes, dtypes, shapes, SHA-256s and light guards stay
    the same; only files and offsets change. Writes ``<out>/manifest.json`` and, last, the declaration
    ``<out>/layout.json`` ({'manifest', 'sha256', 'roots'}), which it returns."""
    out = Path(out)
    by_file = collections.defaultdict(list)
    for name, t in manifest['tensors'].items():
        by_file[t['file']].append(name)
    jobs = []
    for key in sorted(by_file):
        source = manifest['files'][key]
        stem = Path(source['root'], source['path']).with_suffix('')
        current, size = [], 0
        for name in sorted(by_file[key], key=lambda n: manifest['tensors'][n]['offset']) + [None]:
            nbytes = manifest['tensors'][name]['bytes'] if name else 0
            if current and (name is None or size + nbytes > shard_bytes):
                jobs.append((key, f'{stem}/{len(jobs):05d}.safetensors', current))
                current, size = [], 0
            if name:
                current.append(name)
                size += nbytes

    def write_shard(job):
        key, rel, names = job
        source = manifest['files'][key]
        header, pos = {}, 0
        for name in names:
            t = manifest['tensors'][name]
            header[name] = {'dtype': t['dtype'], 'shape': t['shape'], 'data_offsets': [pos, pos + t['bytes']]}
            pos += t['bytes']
        blob = json.dumps(header, separators=(',', ':')).encode()
        blob += b' ' * (-len(blob) % 8)
        blob = struct.pack('<Q', len(blob)) + blob
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(blob)
        with open(Path(roots[source['root']]) / source['path'], 'rb') as f, open(path.with_suffix('.tmp'), 'wb') as w:
            w.write(blob)
            for name in names:
                t = manifest['tensors'][name]
                f.seek(t['offset'])
                tensor, left = hashlib.sha256(), t['bytes']
                while left:
                    chunk = f.read(min(left, READ_BYTES))
                    if not chunk:
                        raise ValueError(f'{name}: {source["path"]} is shorter than its manifest')
                    tensor.update(chunk)
                    digest.update(chunk)
                    w.write(chunk)
                    left -= len(chunk)
                if tensor.hexdigest() != t['sha256']:
                    raise ValueError(f'{name}: bytes differ from the manifest')
        os.replace(path.with_suffix('.tmp'), path)
        return rel, len(blob) + pos, digest.hexdigest(), {n: len(blob) + header[n]['data_offsets'][0] for n in names}

    physical = {'schema': SCHEMA, 'segment_bytes': SEGMENT, 'files': {}, 'tensors': {}}
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for rel, size, digest, offsets in pool.map(write_shard, jobs):
            physical['files'][f'layout:{rel}'] = {'root': 'layout', 'path': rel, 'bytes': size, 'sha256': digest}
            for name, offset in offsets.items():
                physical['tensors'][name] = {**manifest['tensors'][name], 'file': f'layout:{rel}', 'offset': offset}
    physical['tensors'] = dict(sorted(physical['tensors'].items()))
    declaration = {'manifest': str(out / 'manifest.json'), 'sha256': write(physical, out / 'manifest.json'),
                   'roots': {'layout': str(out)}}
    (out / 'layout.json').write_text(json.dumps(declaration, indent=1) + '\n')
    return declaration


def _select(root, globs):
    names = sorted(p.name for p in Path(root).iterdir() if p.is_file())
    return [n for n in names if any(fnmatch.fnmatch(n, g) for g in globs)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    m = sub.add_parser('manifest', help='hash a checkpoint in place')
    m.add_argument('--root', required=True, help='checkpoint directory (root "checkpoint")')
    m.add_argument('--file', action='append', default=[], help='GLOB[@PREFIX] under --root (repeatable)')
    m.add_argument('--extra-root', action='append', default=[], help='NAME=DIR (repeatable)')
    m.add_argument('--extra-file', action='append', default=[],
                   help='NAME:GLOB[@PREFIX] relative to that root (repeatable)')
    m.add_argument('--out', required=True)
    m.add_argument('--workers', type=int, default=16)
    c = sub.add_parser('convert', help='write a BF16 copy of F32 tensors')
    c.add_argument('--root', required=True)
    c.add_argument('--file', action='append', required=True, help='glob under --root (repeatable)')
    c.add_argument('--out', required=True)
    c.add_argument('--prefix', default='diffusion_pytorch_model')
    c.add_argument('--copy', action='append', default=[], help='file under --root to copy alongside')
    args = parser.parse_args(argv)
    if args.command == 'convert':
        files = _select(args.root, args.file)
        if not files:
            parser.error('no files selected')
        print(json.dumps(convert(args.root, files, args.out, args.prefix, copy=args.copy)))
        return
    roots = {'checkpoint': args.root, **dict(r.split('=', 1) for r in args.extra_root)}
    files = []
    for spec in [f'checkpoint:{g}' for g in args.file] + args.extra_file:
        root, pattern = spec.split(':', 1)
        pattern, _, prefix = pattern.partition('@')
        base = Path(roots[root])
        files += [(root, str(p.relative_to(base)), prefix) for p in sorted(base.glob(pattern)) if p.is_file()]
    if not files:
        parser.error('no files selected')
    manifest = build(roots, files, args.workers)
    digest = write(manifest, args.out)
    print(json.dumps({'manifest': args.out, 'sha256': digest, 'files': len(manifest['files']),
                      'tensors': len(manifest['tensors'])}))


if __name__ == '__main__':
    main()
