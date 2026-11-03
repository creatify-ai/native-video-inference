"""Full SHA-256 audit behind a light-verified streaming load (core_infra::model_load with
NATIVE_LOAD_VERIFY=light checks each tensor's first and last KiB before use). Re-reads every
selected tensor range of a native-checkpoint-v1 manifest and checks it against the manifest at low
priority; an owner pauses it (SIGSTOP / SIGCONT) while requests run.

    python -m core_infra.runtime.audit --manifest <prepared>/manifest.json --root checkpoint=<dir> \\
        --root prepared=<dir> [--prefix dit. --prefix t5.] [--sha256 <manifest digest>] --out report.json
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time

CHUNK = 16 << 20


def _check(path, offset, nbytes, want):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        f.seek(offset)
        left = nbytes
        while left:
            data = f.read(min(CHUNK, left))
            if not data:
                return False
            digest.update(data)
            left -= len(data)
    return digest.hexdigest() == want


def audit(manifest_path, roots, prefixes=(), expected_sha256=None, workers=4):
    """{'checked', 'bytes', 'mismatched', 'seconds'} for the tensors whose names start with a prefix
    (all when none); raises if the manifest itself does not match `expected_sha256`."""
    started = time.time()
    raw = Path(manifest_path).read_bytes()
    if expected_sha256 and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('checkpoint manifest digest mismatch')
    manifest = json.loads(raw)
    jobs = []
    for name, t in manifest['tensors'].items():
        if prefixes and not name.startswith(tuple(prefixes)):
            continue
        f = manifest['files'][t['file']]
        jobs.append((name, Path(roots[f['root']]) / f['path'], t['offset'], t['bytes'], t['sha256']))
    mismatched = []
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for (name, *_), ok in zip(jobs, pool.map(lambda j: _check(*j[1:]), jobs)):
            if not ok:
                mismatched.append(name)
    return {'checked': len(jobs), 'bytes': sum(j[3] for j in jobs), 'mismatched': mismatched,
            'seconds': time.time() - started}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--root', action='append', default=[], help='NAME=DIR (repeatable)')
    parser.add_argument('--prefix', action='append', default=[])
    parser.add_argument('--sha256')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    os.nice(10)
    report = audit(args.manifest, dict(r.split('=', 1) for r in args.root), args.prefix, args.sha256, args.workers)
    Path(args.out).write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({k: report[k] for k in ('checked', 'bytes', 'seconds')} | {'mismatched': len(report['mismatched'])}))
    raise SystemExit(1 if report['mismatched'] else 0)


if __name__ == '__main__':
    main()
