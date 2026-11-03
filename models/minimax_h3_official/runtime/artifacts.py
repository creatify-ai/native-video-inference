"""H3 metadata schedules and optional, caller-pinned backend plans."""
import hashlib
import json
from pathlib import Path
import subprocess

from core_infra.runtime.files import publish_bundle


def prefetch_jobs(config, component):
    root = Path(config['models'][component])
    filename = 'model-manifest.json' if component == 'conditioner' else 'manifest.json'
    manifest = json.loads((root / filename).read_bytes())
    selected = {}
    def add(base, spec):
        if 'file' not in spec or not spec.get('bytes'):
            return
        path = str(Path(base) / spec['file'])
        offset = spec.get('offset', 0)
        selected[path, offset, spec['bytes']] = {'path': path, 'offset': offset, 'bytes': spec['bytes']}
    for kind in ('weights', 'buffers'):
        base = Path(manifest['shared_weights']['root']) if kind == 'weights' and 'shared_weights' in manifest else root
        for spec in manifest.get(kind, {}).values():
            add(base, spec)
    return list(selected.values())


def prepare_backend(config, root):
    pin = config.get('runtime', {}).get('backend_plans')
    if not pin:
        return {'state': 'native_builders', 'forward_executed': False}
    raw = Path(pin['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin['sha256']:
        raise ValueError('backend plan manifest SHA256 mismatch')
    manifest = json.loads(raw)
    drivers = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                      text=True, timeout=5).splitlines()
    if not drivers or any(driver.strip() not in manifest['drivers'] for driver in drivers):
        return {'state': 'native_builders', 'reason': 'unmatched_driver', 'forward_executed': False}
    staging = root / 'verified-plans'
    publish_bundle(pin['path'], pin['sha256'], staging)
    shapes = set()
    for name in manifest['files']:
        key = (staging / name).read_bytes().split(b'\n', 1)[0].decode().split(':')
        if len(key) >= 5 and key[:2] == ['h3-exact-plan-v1', 'qwen']:
            kind, rows = int(key[3]), int(key[4])
            if kind not in (0, 1) or not 0 < rows <= 262144:
                raise ValueError('invalid Qwen metadata shape')
            shapes.add((kind, rows))
    (staging / 'qwen-shapes.json').write_text(json.dumps(sorted(shapes)))
    staging.rename(root / 'backend-plans')
    return {'state': 'configured', 'qwen_shapes': sorted(shapes), 'forward_executed': False,
            'manifest_sha256': pin['sha256']}
