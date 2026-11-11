"""Download the pinned DeepSeek-V4-Flash NVFP4 checkpoint into a Modal Volume (once).

    modal run models/deepseek_v4_flash/reference/checkpoint.py

The checkpoint is nvidia/DeepSeek-V4-Flash-NVFP4 at REVISION (168.3 GB, 80 files), stored in the
Hugging Face cache layout under /dsv4/huggingface/hub. A second run only re-hashes nothing and
returns the recorded report.
"""
import json
import os
from pathlib import Path

import modal

REPOSITORY = 'nvidia/DeepSeek-V4-Flash-NVFP4'
REVISION = 'e3cd60e7de98e9867116860d522499a728de1cf9'
MOUNT = '/dsv4'
CHECKPOINT = f"{MOUNT}/huggingface/hub/models--{REPOSITORY.replace('/', '--')}/snapshots/{REVISION}"
VOLUME_NAME = os.environ.get('DSV4_VOLUME', 'dsv4-flash-native')

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App('dsv4-flash-checkpoint')
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('huggingface-hub[hf_xet]==0.36.0')
         .env({'HF_XET_HIGH_PERFORMANCE': '1'}))


@app.function(image=image, cpu=16, memory=32768, timeout=6 * 3600, volumes={MOUNT: volume})
def download() -> dict:
    import hashlib
    from huggingface_hub import snapshot_download
    report = Path(MOUNT) / f'checkpoint-{REVISION[:12]}.json'
    if report.exists():
        return {**json.loads(report.read_text()), 'fresh': False}
    path = snapshot_download(REPOSITORY, revision=REVISION, cache_dir=f'{MOUNT}/huggingface/hub', max_workers=16)
    volume.commit()
    if Path(path) != Path(CHECKPOINT):
        raise RuntimeError(f'unexpected snapshot path {path}')
    files = {}
    for f in sorted(Path(path).rglob('*')):
        if f.is_file():
            h = hashlib.sha256()
            with open(f, 'rb') as fh:
                while chunk := fh.read(64 << 20):
                    h.update(chunk)
            files[str(f.relative_to(path))] = {'bytes': f.stat().st_size, 'sha256': h.hexdigest()}
    out = {'repository': REPOSITORY, 'revision': REVISION, 'path': CHECKPOINT,
           'bytes': sum(v['bytes'] for v in files.values()), 'files': files}
    report.write_text(json.dumps(out, indent=1))
    volume.commit()
    return {**out, 'fresh': True}


@app.local_entrypoint()
def main():
    out = download.remote()
    print(json.dumps({k: v for k, v in out.items() if k != 'files'}, indent=1))
    print(len(out['files']), 'files')
