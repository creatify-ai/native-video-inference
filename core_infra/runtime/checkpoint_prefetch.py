"""Optional OS cache warming from a model's immutable checkpoint declaration.

Only page-aligned file ranges are returned. Native readers retain responsibility
for tensor guards, full audits, conversion, and GPU upload. No tensor is decoded
or retained in private Python buffers here.
"""
import hashlib
import json
from pathlib import Path


def jobs(manifest_path, roots, prefixes=(), expected_sha256=None):
    raw = Path(manifest_path).read_bytes()
    if expected_sha256 and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('checkpoint manifest digest mismatch')
    manifest = json.loads(raw)
    if manifest['schema'] != 'native-checkpoint-v1' or manifest['segment_bytes'] != 1024:
        raise ValueError('unsupported checkpoint manifest')
    ranges = {}
    for name, tensor in manifest['tensors'].items():
        if prefixes and not name.startswith(tuple(prefixes)):
            continue
        file = manifest['files'][tensor['file']]
        offset, count, extent = tensor['offset'], tensor['bytes'], file['bytes']
        if (any(type(n) is not int for n in (offset, count, extent))
                or offset < 0 or count <= 0 or offset + count > extent):
            raise ValueError('invalid checkpoint tensor extent')
        ranges.setdefault(tensor['file'], []).append((offset // 4096 * 4096, offset + count))
    result = []
    for key, parts in ranges.items():
        file = manifest['files'][key]
        relative = Path(file['path'])
        if not file['path'] or relative.is_absolute() or '..' in relative.parts:
            raise ValueError('checkpoint path escapes its declared root')
        path = Path(roots[file['root']]) / relative
        if not path.is_absolute():
            raise ValueError('checkpoint root must be absolute')
        merged = []
        for lo, hi in sorted(parts):
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(hi, merged[-1][1])
            else:
                merged.append([lo, hi])
        result.extend(dict(path=str(path), offset=lo, bytes=hi - lo, file_bytes=file['bytes'])
                      for lo, hi in merged)
    return result
