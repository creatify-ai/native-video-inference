"""Pack verified raw tensor exports into one aligned immutable checkpoint.

Tensor bytes and dtypes are preserved. This is an optional serving layout for
exports containing many small files; published safetensors/PyTorch checkpoints
can continue to use the in-place checkpoint loader.
"""
import hashlib
import json
import math
from pathlib import Path
import tempfile

from .checkpoint import ITEMSIZE, SCHEMA, SEGMENT

DTYPES = {'torch.float64': 'F64', 'torch.float32': 'F32', 'torch.float16': 'F16',
          'torch.bfloat16': 'BF16', 'torch.int64': 'I64', 'torch.int32': 'I32',
          'torch.int16': 'I16', 'torch.int8': 'I8', 'torch.uint8': 'U8', 'torch.bool': 'BOOL'}


def pack(root, tensors, destination, *, source_manifest_sha256):
    root, destination = Path(root), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    if not tensors or len(source_manifest_sha256) != 64 or any(c not in '0123456789abcdef' for c in source_manifest_sha256):
        raise ValueError('tensor inventory and immutable source manifest SHA required')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.tensor-archive-', dir=destination.parent) as temporary:
        staging = Path(temporary)
        payload = staging / 'payload.bin'
        file_hash = hashlib.sha256()
        entries = {}
        offset = 0
        with payload.open('wb') as output:
            for name, tensor in sorted(tensors.items()):
                relative = Path(tensor.get('file') or '')
                if not tensor.get('file') or relative.is_absolute() or any(p in ('.', '..') for p in relative.parts):
                    raise ValueError('tensor path escapes its admitted root: ' + name)
                expected = tensor['sha256']
                dtype = DTYPES.get(tensor['dtype'], tensor['dtype'])
                shape, count = tensor['shape'], tensor['bytes']
                if dtype not in ITEMSIZE or any(type(d) is not int or d < 0 for d in shape) or count <= 0 or math.prod(shape) * ITEMSIZE[dtype] != count:
                    raise ValueError('tensor shape/dtype extent differs: ' + name)
                if len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
                    raise ValueError('full tensor SHA required: ' + name)
                padding = bytes((-offset) % 256)
                output.write(padding)
                file_hash.update(padding)
                offset += len(padding)
                digest, head, tail = hashlib.sha256(), bytearray(), bytearray()
                remaining = count
                with (root / relative).open('rb') as source:
                    if source.seek(0, 2) != count:
                        raise ValueError('tensor file extent differs: ' + name)
                    source.seek(0)
                    while remaining:
                        chunk = source.read(min(16 << 20, remaining))
                        if not chunk:
                            raise ValueError('tensor file truncated: ' + name)
                        output.write(chunk)
                        digest.update(chunk)
                        file_hash.update(chunk)
                        head.extend(chunk[:max(0, SEGMENT - len(head))])
                        tail = (tail + chunk)[-SEGMENT:]
                        remaining -= len(chunk)
                    if source.read(1):
                        raise ValueError('tensor file grew: ' + name)
                if digest.hexdigest() != expected:
                    raise ValueError('tensor SHA differs: ' + name)
                entries[name] = dict(file='archive:payload.bin', offset=offset, bytes=count,
                    dtype=dtype, shape=shape, sha256=expected,
                    head_sha256=hashlib.sha256(head).hexdigest(), tail_sha256=hashlib.sha256(tail).hexdigest())
                offset += count
        manifest = dict(schema=SCHEMA, segment_bytes=SEGMENT, source_manifest_sha256=source_manifest_sha256,
            files={'archive:payload.bin': dict(root='archive', path='payload.bin', bytes=offset, sha256=file_hash.hexdigest())},
            tensors=entries)
        raw = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
        (staging / 'manifest.json').write_bytes(raw)
        # Publication happens only after every source byte has passed its SHA.
        # Never replace an already admitted archive.
        destination.mkdir()
        try:
            payload.rename(destination / payload.name)
            (staging / 'manifest.json').rename(destination / 'manifest.json')
        except BaseException:
            import shutil
            shutil.rmtree(destination)
            raise
    return dict(path=str(destination / 'manifest.json'), sha256=hashlib.sha256(raw).hexdigest(),
                payload_bytes=offset, tensors=len(entries), source_manifest_sha256=source_manifest_sha256)
