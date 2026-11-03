"""First and last KiB of every prepared payload, written after a full SHA-256 check.

Serving reads the weights once, on its way to the GPU. Checking these guards there leaves the full
hash of about 95 GB out of the cold start; this step has already checked every byte on the Volume.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

from .common import CHUNK, write_json

SEGMENT = 1024
FILENAME = 'light-sha.json'


def payloads(value, root):
    """Every {file, bytes, sha256} record of a package manifest that names a file on disk, by content
    digest. Serving rejects a payload with no guard, so a record skipped here cannot go unchecked."""
    found = {}
    if isinstance(value, dict):
        if (all(isinstance(value.get(key), str) for key in ('file', 'sha256')) and isinstance(value.get('bytes'), int)
                and (root / value['file']).is_file()):
            found[value['sha256']] = (root / value['file'], value.get('offset', 0), value['bytes'])
        for item in value.values():
            found.update(payloads(item, root))
    elif isinstance(value, list):
        for item in value:
            found.update(payloads(item, root))
    return found


def guard(digest, path, offset, size):
    full, head, tail, remaining = hashlib.sha256(), b'', b'', size
    with open(path, 'rb', buffering=0) as stream:
        stream.seek(offset)
        while remaining:
            data = stream.read(min(CHUNK, remaining))
            if not data:
                raise ValueError(f'prepared payload is short: {path}')
            full.update(data)
            head = head or data[:SEGMENT]
            tail = (tail + data)[-SEGMENT:]
            remaining -= len(data)
    if full.hexdigest() != digest:
        raise ValueError(f'prepared payload digest differs: {path}')
    return digest, {'bytes': size, 'head_sha256': hashlib.sha256(head).hexdigest(),
                    'tail_sha256': hashlib.sha256(tail).hexdigest()}


def write_light_guards(prepared, packages, workers=16):
    """Check every payload of `packages` and write their guards; returns the table's SHA-256."""
    prepared, found = Path(prepared), {}
    for name in packages:
        filename = 'model-manifest.json' if name == 'conditioner' else 'manifest.json'
        manifest = prepared / name / filename
        if manifest.is_file():
            found.update(payloads(json.loads(manifest.read_bytes()), prepared / name))
    with ThreadPoolExecutor(workers) as pool:
        entries = dict(pool.map(lambda item: guard(item[0], *item[1]), found.items()))
    return write_json(prepared / FILENAME, {'schema': 'h3-light-sha-v1', 'segment_bytes': SEGMENT,
                                            'full_sha256_verified': True, 'entries': entries})
