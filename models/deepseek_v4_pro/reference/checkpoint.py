"""Download the pinned DeepSeek-V4-Pro NVFP4 checkpoint into a Modal Volume (once).

    modal run models/deepseek_v4_pro/reference/checkpoint.py

The checkpoint is nvidia/DeepSeek-V4-Pro-NVFP4 at REVISION (913.1 GB, 93 files), stored as plain
files under CHECKPOINT on the v2 Volume `dsv4-pro-native`. A second run returns the recorded
report without downloading or hashing again.
"""
import json
from pathlib import Path

import modal

REPOSITORY = 'nvidia/DeepSeek-V4-Pro-NVFP4'
REVISION = '1449d1e641023406daf6b432361486c768aad740'
MOUNT = '/dsv4pro'
CHECKPOINT = f'{MOUNT}/checkpoint/{REVISION}'
VOLUME_NAME = 'dsv4-pro-native'

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)
app = modal.App('dsv4-pro-checkpoint')
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('huggingface-hub[hf_xet]==0.36.0')
         .env({'HF_XET_HIGH_PERFORMANCE': '1'}))


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        while chunk := fh.read(64 << 20):
            h.update(chunk)
    return h.hexdigest()


@app.function(image=image, cpu=32, memory=65536, timeout=12 * 3600, volumes={MOUNT: volume})
def download() -> dict:
    import concurrent.futures
    import time
    from huggingface_hub import snapshot_download
    report = Path(MOUNT) / f'checkpoint-{REVISION[:12]}.json'
    if report.exists():
        return {**json.loads(report.read_text()), 'fresh': False}
    t = time.time()
    snapshot_download(REPOSITORY, revision=REVISION, local_dir=CHECKPOINT, max_workers=16)
    download_s = time.time() - t
    paths = sorted(f for f in Path(CHECKPOINT).rglob('*') if f.is_file() and '.cache' not in f.parts)
    t = time.time()
    with concurrent.futures.ThreadPoolExecutor(24) as pool:  # hashlib releases the GIL
        sums = list(pool.map(_sha256, paths))
    files = {str(f.relative_to(CHECKPOINT)): {'bytes': f.stat().st_size, 'sha256': s} for f, s in zip(paths, sums)}
    out = {'repository': REPOSITORY, 'revision': REVISION, 'path': CHECKPOINT,
           'bytes': sum(v['bytes'] for v in files.values()), 'download_s': round(download_s),
           'hash_s': round(time.time() - t), 'files': files}
    report.write_text(json.dumps(out, indent=1))
    return {**out, 'fresh': True}


@app.local_entrypoint()
def main():
    out = download.remote()
    print(json.dumps({k: v for k, v in out.items() if k != 'files'}, indent=1))
    print(len(out['files']), 'files')
