"""Model-independent publication of pinned, portable cuDNN plan artifacts.

No forward pass or CUDA context is created here. The shared operator validates
its full shape, layout, GPU, driver, runtime and cuDNN key on every cache lookup;
a missing or incompatible entry falls back to the unchanged native builder.
"""
from pathlib import Path
from .files import publish_bundle


def prepare_backend_plans(pin, destination):
    if not pin:
        return {'state': 'native_builders', 'environment': {}, 'forward_executed': False}
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = publish_bundle(pin['path'], pin['sha256'], destination)
    return {'state': 'configured', 'manifest_sha256': pin['sha256'],
            'files': len(manifest['files']), 'forward_executed': False,
            'environment': {'NATIVE_CUDNN_PLAN_CACHE_MODE': 'read',
                            'NATIVE_CUDNN_PLAN_CACHE_DIR': str(destination)}}
