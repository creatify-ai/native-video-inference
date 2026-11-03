"""Official MiniMax-H3 (FL2VA and Ref2VA) on Modal, run by the native Rust/CUDA pipeline.

    python -m pip install -r examples/requirements-modal.txt
    modal setup
    modal run examples/modal/h3.py

With no arguments this generates one video with audio from the default prompt and prints the cold
start. Pass your own with --prompt "..." --out sailboat.mp4. FL2VA takes optional --first-frame and
--last-frame PNGs. --workflow ref2va takes one --reference image (normalized here to raw RGB), or
--reference-descriptor, a normalized descriptor already in the Volume (image, video and audio
references). --config runs any native.json in the Volume.

The first run does the one-time work:

Image. CUDA 13.0.2 devel, the pinned cuBLAS/cuDNN/NCCL/NVRTC wheels, cuDNN frontend headers,
PyTorch 2.13.0, FlashInfer, Diffusers and Transformers at their pins, Rust 1.90.0 and this
checkout. One image-build step on a B200 exports FlashInfer's SubBlock attention kernel; the
build checks it against models/minimax_h3_official/subblock-v07-source.json (set
H3_SUBBLOCK_EXPORT to a local copy of an accepted export to skip that step). The next step
runs models/minimax_h3_official/build.py on CPU.

Checkpoint. The root-level Diffusers folders of MiniMaxAI/MiniMax-H3 at the pinned revision that
the workflow needs (about 135 GB) go to <volume>/hf/<revision>/.

Packages. `python -m models.minimax_h3_official.prepare run` runs on one B200 and writes
<volume>/prepared/<workflow>/ (about 130 GB): conditioner/, dit/, router/, video_vae/, audio_vae/
and, for Ref2VA, video_encoder/ and audio_encoder/, plus preparation.json with the SHA-256 of
every manifest and native.json. Every tensor file is listed with its SHA-256 in its manifest. The
Qwen conditioner (52 GB) is the same for both workflows: it is written once to
<volume>/prepared/shared/conditioner/ and linked from each workflow. The step is resumable;
completed components are kept. The new packages are then read through once.

Later runs go straight to one B200. Outputs go to <volume>/outputs/<run-id>/output/, including
the raw dit/final_video.raw, dit/final_audio.raw, video_decode/frames.raw and audio.raw, with
their SHA-256 in report.json. The MP4 is downloaded to --out.

    modal run examples/modal/h3.py --cold-start 30

measures the cold start of the request on 30 fresh containers, at most ten at a time, each stopped
at its first DiT step, and prints the P50 and P90.

--gpus 2 runs the DiT as two context-parallel ranks on a B200:2 container (Qwen, the encoders and
the decoders stay on the first GPU). Each rank keeps half of every layer's
weights and reads that half straight from dit/'s files; the first --gpus 2 run writes only the
small dit-cp2/ index and light-sha-cp2.json beside the packages (CPU, a few minutes).

Settings: H3_VOLUME (default minimax-h3-native), H3_MOUNT (Volume mount path, default /models;
package manifests hold absolute paths, so keep it fixed for a Volume), H3_APP_NAME (default
minimax-h3-native), H3_HF_SECRET (a Modal Secret with HF_TOKEN), H3_SUBBLOCK_EXPORT,
COLD_START_REGION (a Modal region for cold_start, such as us-west; region pinning costs extra).
"""
import json
import os
from pathlib import Path
import time

STARTED_NS = [time.monotonic_ns()]  # container entry, consumed by its first request

import modal

REPOSITORY = 'MiniMaxAI/MiniMax-H3'
REVISION = '42ed227ee7df40d41602854ae760620d6eb651fe'
WORKFLOWS = ('fl2va', 'ref2va')
CHECKPOINT_FOLDERS = ('text_encoder', 'vae', 'audio_vae', 'processor', 'tokenizer', 'scheduler', 'audio_scheduler')
TRANSFORMERS = {'fl2va': 'transformer', 'ref2va': 'transformer_ref'}
SOURCE = '/opt/native-video-inference'
SUBBLOCK = '/opt/h3-subblock'
ROUTER_TRITON = '/opt/router-triton'
SITE = '/usr/local/lib/python3.12/site-packages'
RELEASE = f'{SOURCE}/target/release'
DENSE_LIBRARY = f'{SOURCE}/target/native/minimax_h3_official/libh3_dense.so'
GPU = 'B200'
GPUS = (1, 2)
PROMPT = 'A sailboat crosses a quiet lake at sunrise while gulls call overhead and water laps against the hull.'

# The SDK set of the validated native build (see models/minimax_h3_official/README.md).
CUDA_IMAGE = 'nvidia/cuda:13.0.2-devel-ubuntu24.04'
RUST = '1.90.0'
CUDA_WHEELS = ('nvidia-cublas==13.1.1.3', 'nvidia-cuda-runtime==13.0.96', 'nvidia-cuda-nvrtc==13.0.88',
               'nvidia-cudnn-cu13==9.24.0.43', 'nvidia-nccl-cu13==2.29.7')
TORCH = ('torch==2.13.0', 'torchvision==0.28.0')  # git cf30153c: build headers and preparation
TORCH_INDEX = 'https://download.pytorch.org/whl/cu130'
PREPARATION = ('flashinfer-python==0.7.0.post1', 'nvidia-cutlass-dsl==4.8.0', 'transformers==5.15.0', 'accelerate==1.15.0',
               'diffusers @ git+https://github.com/huggingface/diffusers.git@9284607295a09f759aadd65ed08f48b35feea6d9',
               'safetensors==0.8.0', 'numpy==2.5.3', 'pillow==12.3.0', 'huggingface-hub[hf_xet]==1.31.0')
ROUTER_TRITON_PIN = 'triton==3.8.0'  # the router CUBINs' compiler, kept apart from torch's own
HEADERS = 'flashinfer-python==0.6.15.post1'  # its bundled CUTLASS headers only
CUDNN_FRONTEND = 'c4a97621eca52fa0c3a1862a411a16be580b25c6'
CUDNN_FRONTEND_HEADER_SHA256 = '38aabf33dee54fa59eea46666a15c78775025b83c54be196260e4fd9ca697a06'

VOLUME_NAME = os.environ.get('H3_VOLUME', 'minimax-h3-native')
MOUNT = os.environ.get('H3_MOUNT', '/models')
if not MOUNT.startswith('/') or MOUNT == '/':
    raise SystemExit('H3_MOUNT must be an absolute directory')
APP_NAME = os.environ.get('H3_APP_NAME', 'minimax-h3-native')
COLD_START_REGION = os.environ.get('COLD_START_REGION') or None  # only read where the app is launched
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)
# Containers import this file again; carry the local settings into them.
HF_SECRET = os.environ.get('H3_HF_SECRET', '')
settings = {'H3_VOLUME': VOLUME_NAME, 'H3_MOUNT': MOUNT, 'H3_APP_NAME': APP_NAME, 'H3_HF_SECRET': HF_SECRET}
secrets = [modal.Secret.from_name(HF_SECRET)] if HF_SECRET else []
repository = Path(__file__).resolve().parents[2] if modal.is_local() else None
CHECKPOINT = f'{MOUNT}/hf/{REVISION}'


def prepared(workflow):
    return f'{MOUNT}/prepared/{workflow}'


def build_environment():
    nvidia = Path(SITE) / 'nvidia'
    return {'CUDA_HOME': '/usr/local/cuda', 'H3_CUDA_ARCH': '100a', 'H3_SUBBLOCK_AOT_ROOT': SUBBLOCK,
            'H3_CUBLAS_ROOT': str(nvidia / 'cu13'), 'H3_CUDNN_ROOT': str(nvidia / 'cudnn'),
            'H3_NCCL_ROOT': str(nvidia / 'nccl'), 'H3_CUDNN_FRONTEND': '/opt/cudnn-frontend',
            'H3_TORCH_INCLUDE': f'{SITE}/torch/include',
            'H3_CUTLASS_ROOT': '/opt/flashinfer-headers/flashinfer/data/cutlass',
            'H3_VENDOR_INCLUDES': ':'.join(str(p) for p in sorted(nvidia.glob('*/include'))),
            'PATH': f"/root/.cargo/bin:/usr/local/cuda/bin:{os.environ['PATH']}"}


def export_subblock():
    """Image build step on a B200: export the pinned FlashInfer SubBlock kernel."""
    import subprocess
    import sys
    if not Path(SUBBLOCK, 'export.json').exists():
        subprocess.run([sys.executable, f'{SOURCE}/models/minimax_h3_official/prepare_subblock_aot.py',
                        'export', SUBBLOCK], check=True)


def build_native():
    """Image build step (CPU): header-only SDKs, then the native H3 programs from SOURCE."""
    import hashlib
    import subprocess
    import sys
    import urllib.request
    archive = Path('/tmp/cudnn-frontend.tar.gz')
    urllib.request.urlretrieve(f'https://github.com/NVIDIA/cudnn-frontend/archive/{CUDNN_FRONTEND}.tar.gz', archive)
    Path('/opt/cudnn-frontend').mkdir(parents=True, exist_ok=True)
    subprocess.run(['tar', '-xzf', str(archive), '-C', '/opt/cudnn-frontend', '--strip-components=1'], check=True)
    header = Path('/opt/cudnn-frontend/include/cudnn_frontend/graph_interface.h')
    if hashlib.sha256(header.read_bytes()).hexdigest() != CUDNN_FRONTEND_HEADER_SHA256:
        raise RuntimeError('the cuDNN frontend checkout differs from its pin')
    for target, package in (('/opt/flashinfer-headers', HEADERS), (ROUTER_TRITON, ROUTER_TRITON_PIN)):
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--target', target, package], check=True)
    # The wheels ship versioned sonames only; the linker needs the plain names.
    nvidia = Path(SITE) / 'nvidia'
    for directory, library, soname in (('cu13', 'cublas', 'libcublas.so.13'), ('cu13', 'cublasLt', 'libcublasLt.so.13'),
                                       ('cu13', 'nvrtc', 'libnvrtc.so.13'), ('cu13', 'cudart', 'libcudart.so.13'),
                                       ('cudnn', 'cudnn', 'libcudnn.so.9'), ('nccl', 'nccl', 'libnccl.so.2')):
        link = nvidia / directory / 'lib' / f'lib{library}.so'
        if not link.exists():
            link.symlink_to(soname)
    env = dict(os.environ, **build_environment())
    subprocess.run([sys.executable, 'models/minimax_h3_official/build.py'], cwd=SOURCE, env=env, check=True)
    Path('/etc/ld.so.conf.d/h3.conf').write_text('\n'.join(
        [str(nvidia / 'cu13/lib'), str(nvidia / 'cudnn/lib'), str(nvidia / 'nccl/lib'), SUBBLOCK,
         f'{SOURCE}/target/native/minimax_h3_official']) + '\n')
    subprocess.run(['ldconfig'], check=True)


image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python='3.12')
    .env({'DEBIAN_FRONTEND': 'noninteractive'})
    .apt_install('build-essential', 'ca-certificates', 'curl', 'ffmpeg', 'git')
    .run_commands("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | "
                  f'sh -s -- -y --profile minimal --default-toolchain {RUST}')
    .pip_install(*TORCH, index_url=TORCH_INDEX, extra_index_url='https://pypi.org/simple')
    .pip_install(*PREPARATION)
    # Reinstall the validated SDK wheels last: torch and FlashInfer pin their own versions.
    .pip_install(*CUDA_WHEELS, extra_options='--no-deps --force-reinstall')
)
if modal.is_local():
    image = image.add_local_dir(repository, SOURCE, copy=True, ignore=[
        '.git', '.git/**', 'target', 'target/**', '.venv', '.venv/**', '**/__pycache__/**', '**/*.pyc',
        '**/.DS_Store', '.github/**', 'docs/**', 'tests/**', 'outputs/**', 'artifacts/**'])
    if os.environ.get('H3_SUBBLOCK_EXPORT'):
        image = image.add_local_dir(os.environ['H3_SUBBLOCK_EXPORT'], SUBBLOCK, copy=True)
image = (image.run_function(export_subblock, gpu=GPU, timeout=3600)
         .run_function(build_native, cpu=32, memory=65536, timeout=3 * 3600)
         .env({**settings, 'PYTHONPATH': SOURCE, 'HF_XET_HIGH_PERFORMANCE': '1'}))


@app.function(image=image, cpu=16, memory=32768, timeout=6 * 3600, volumes={MOUNT: volume}, secrets=secrets)
def download(workflow: str) -> str:
    from huggingface_hub import snapshot_download
    folders = (*CHECKPOINT_FOLDERS, TRANSFORMERS[workflow])
    snapshot_download(REPOSITORY, revision=REVISION, local_dir=CHECKPOINT,
                      allow_patterns=[f'{folder}/*' for folder in folders] + ['model_index.json'])
    volume.commit()
    return CHECKPOINT


@app.function(image=image, gpu=GPU, cpu=16, memory=262144, timeout=12 * 3600, volumes={MOUNT: volume})
def prepare_packages(workflow: str, components: str = '', decoder_kernel_configs: str = '') -> dict:
    import subprocess
    import sys
    command = [sys.executable, '-m', 'models.minimax_h3_official.prepare', 'run', '--checkpoint', CHECKPOINT,
               '--workflow', workflow, '--out', prepared(workflow), '--shared', f'{MOUNT}/prepared/shared',
               '--verify-checkpoint',
               '--router-python-path', ROUTER_TRITON, '--programs', RELEASE, '--dense-library', DENSE_LIBRARY]
    if components:
        command += ['--components', components]
    if decoder_kernel_configs:
        command += ['--decoder-kernel-configs', decoder_kernel_configs]
    try:
        subprocess.run(command, cwd=SOURCE, check=True)
    finally:
        volume.commit()
    return json.loads(Path(prepared(workflow), 'preparation.json').read_bytes())


@app.function(image=image, cpu=16, memory=65536, timeout=3600, volumes={MOUNT: volume})
def prepare_cp(workflow: str, world: int) -> dict:
    """CPU: the context-parallel DiT index of a prepared workflow (no weight bytes are copied)."""
    import subprocess
    import sys
    try:
        subprocess.run([sys.executable, '-m', 'models.minimax_h3_official.prepare', 'cp', '--prepared', prepared(workflow),
                        '--world', str(world)], cwd=SOURCE, check=True)
    finally:
        volume.commit()
    return json.loads(Path(prepared(workflow), 'preparation.json').read_bytes())['context_parallel'][str(world)]


@app.function(image=image, cpu=1, memory=1024, volumes={MOUNT: volume})
def is_prepared(workflow: str, world: int = 1) -> bool:
    volume.reload()
    report = Path(prepared(workflow), 'preparation.json')
    if not report.is_file():
        return False
    value = json.loads(report.read_bytes())
    return 'light_sha256' in value and (world == 1 or str(world) in value.get('context_parallel', {}))


@app.function(image=image, cpu=16, memory=32768, timeout=3600, volumes={MOUNT: volume.read_only()})
def warm_volume(workflow: str):
    """Read the new packages once. A new Volume's first read is far slower than later ones, and
    this keeps that cost out of the first GPU cold start."""
    from concurrent.futures import ThreadPoolExecutor
    from models.minimax_h3_official.prepare.__main__ import PACKAGES

    def drain(path):
        with open(path, 'rb', buffering=0) as stream:
            while stream.read(1 << 26):
                pass
    files = [path for name in PACKAGES[workflow] for path in Path(prepared(workflow), name).rglob('*') if path.is_file()]
    with ThreadPoolExecutor(32) as pool:
        list(pool.map(drain, files))


@app.function(image=image, cpu=4, memory=8192, timeout=1800, volumes={MOUNT: volume})
def write_config(workflow: str, attention: str, world: int = 1) -> dict:
    """Validate the prepared packages and write their native.json (CPU only)."""
    from models.minimax_h3_official.prepare.__main__ import native_config
    from models.minimax_h3_official.prepare.common import write_json
    from models.minimax_h3_official.public_variant import validate_config
    config = native_config(prepared(workflow), workflow, RELEASE, DENSE_LIBRARY, attention,
                           {'format': 'mp4', 'capture_raw': True, 'hash_outputs': True}, world)
    manifest = validate_config(config)
    path = Path(prepared(workflow), 'native.json' if world == 1 else f'native-cp{world}.json')
    sha = write_json(path, config)
    volume.commit()
    return {'config': str(path), 'sha256': sha, 'nfe': manifest['recipe']['nfe']}


def reference_descriptor(image_path, directory, longest_edge=1536):
    """Normalize one image reference to interleaved RGB bytes at a multiple of 32 pixels."""
    from PIL import Image
    with Image.open(image_path) as source:
        source = source.convert('RGB')
        scale = min(1.0, longest_edge / max(source.size))
        width, height = (max(32, int(side * scale) // 32 * 32) for side in source.size)
        raw = directory / 'reference.rgb'
        raw.write_bytes(source.resize((width, height), Image.Resampling.BICUBIC).tobytes())
    descriptor = directory / 'reference-image.json'
    descriptor.write_text(json.dumps({'schema': 'h3-ref2va-normalized-rgb-v1', 'references': [
        {'type': 'image', 'file': str(raw), 'width': width, 'height': height, 'has_audio': False}]}))
    return descriptor


def on_gpus(function, gpus):
    """`function` on a container with `gpus` B200s."""
    return function if gpus == 1 else function.with_options(gpu=f'{GPU}:{gpus}', memory=262144)


def with_inputs(request, inputs, directory):
    """`request` with each input file written to `directory` and a reference image normalized."""
    request = dict(request)
    for field, (name, data) in inputs.items():
        path = directory / Path(name).name
        path.write_bytes(data)
        request[field] = str(path)
    if 'reference' in request:
        request['reference_descriptor'] = str(reference_descriptor(Path(request.pop('reference')), directory))
    return request


@app.function(image=image, gpu=GPU, cpu=16, memory=196608, timeout=2 * 3600, volumes={MOUNT: volume})
def infer(config: str, request: dict, inputs: dict, run_id: str) -> dict:
    """Run one request; `inputs` maps a request field to (file name, bytes) from the caller."""
    from models.minimax_h3_official.native_pipeline import MiniMaxH3Pipeline
    entered = STARTED_NS.pop() if STARTED_NS else time.monotonic_ns()
    marker = Path('/tmp/h3-first-dit')  # the DiT process appends (pid, CLOCK_MONOTONIC ns) at its first operation
    marker.unlink(missing_ok=True)
    os.environ.update(NATIVE_FIRST_DIT_TRACE='1', NATIVE_FIRST_DIT_NOTIFY_FILE=str(marker))
    run = Path(MOUNT) / 'outputs' / run_id
    (run / 'inputs').mkdir(parents=True, exist_ok=False)
    request = with_inputs(request, inputs, run / 'inputs')
    try:
        report = MiniMaxH3Pipeline.from_prepared(config).run(request, run / 'output')
    finally:
        volume.commit()
    video = run / 'output/output.mp4'
    stamp = marker.read_bytes()[8:16] if marker.exists() else b''
    return {'report': report, 'output': str(run / 'output'), 'mp4': video.read_bytes() if video.exists() else None,
            'cold_start': (int.from_bytes(stamp, 'little') - entered) / 1e9 if stamp else None}


@app.local_entrypoint()
def main(prompt: str = PROMPT, workflow: str = 'fl2va', first_frame: str = '', last_frame: str = '', reference: str = '',
         reference_descriptor: str = '', width: int = 768, height: int = 768, frames: int = 124, seed: int = 0,
         attention: str = 'subblock_sparse_attn', config: str = '', out: str = '', run_id: str = '',
         cold_start: int = 0, gpus: int = 1):
    """One request. The first run for a workflow downloads the checkpoint and builds its packages.
    --cold-start N instead measures the cold start of the request on N fresh containers (run once first).
    --gpus 2 runs the DiT on two GPUs."""
    request, inputs = request_spec(prompt, workflow, first_frame, last_frame, reference, reference_descriptor,
                                   width, height, frames, seed, attention)
    if gpus not in GPUS:
        raise SystemExit('--gpus must be 1 or 2')
    if cold_start:
        import sys
        sys.path.insert(0, str(repository))
        from core_infra.runtime import cold_start as measure
        if not config and not is_prepared.remote(workflow, gpus):
            raise SystemExit(f'Not prepared yet: run `modal run examples/modal/h3.py --workflow {workflow} '
                             f'--gpus {gpus}` once first')
        config = config or write_config.remote(workflow, attention, gpus)['config']
        measure.collect(on_gpus(cold_start_sample, gpus), cold_start, config, request, inputs)
        return
    if not config:
        if not is_prepared.remote(workflow):
            print(f'First run for {workflow}: downloading the checkpoint and building the native packages (once).')
            download.remote(workflow)
            prepare_packages.remote(workflow)
            warm_volume.remote(workflow)
        if not is_prepared.remote(workflow, gpus):
            print(f'First {gpus}-GPU run for {workflow}: indexing the DiT weights for {gpus} ranks (once).')
            prepare_cp.remote(workflow, gpus)
        config = write_config.remote(workflow, attention, gpus)['config']
    run_id = run_id or time.strftime(f'{workflow}-%Y%m%dT%H%M%S')
    result = on_gpus(infer, gpus).remote(config, request, inputs, run_id)
    report = result['report']
    if result['cold_start'] is not None:
        print(f"\ncold start  {result['cold_start']:.1f} s (container start to first DiT step)")
    print(f"generation  {report['seconds']:.1f} s ({report['nfe']} model evaluations)")
    if result['mp4'] is not None:
        target = Path(out or 'h3.mp4')
        target.write_bytes(result['mp4'])
        print(f"video       {target} ({width}x{height}, {report['num_frames']} frames)")
    print(f"outputs     Volume {VOLUME_NAME} at {result['output']}")


def request_spec(prompt, workflow, first_frame, last_frame, reference, reference_descriptor, width, height, frames,
                 seed, attention):
    """The request and its input files (field -> (name, bytes)) from the command-line options."""
    if workflow not in WORKFLOWS:
        raise SystemExit('--workflow must be fl2va or ref2va')
    references = bool(reference) + bool(reference_descriptor)
    if workflow == 'ref2va' and (references != 1 or first_frame or last_frame):
        raise SystemExit('ref2va takes exactly one of --reference or --reference-descriptor')
    if workflow == 'fl2va' and references:
        raise SystemExit('fl2va takes optional --first-frame/--last-frame only')
    if last_frame and not first_frame:
        raise SystemExit('--last-frame requires --first-frame')
    request = dict(prompt=prompt, width=width, height=height, num_frames=frames, seed=seed,
                   attention_backend=attention)
    if reference_descriptor:
        request['reference_descriptor'] = reference_descriptor
    inputs = {field: (Path(path).name, Path(path).read_bytes()) for field, path in
              (('first_png', first_frame), ('last_png', last_frame), ('reference', reference)) if path}
    return request, inputs


@app.function(image=image, gpu=GPU, cpu=16, memory=196608, timeout=1800, volumes={MOUNT: volume.read_only()},
              single_use_containers=True, max_containers=10, region=COLD_START_REGION)
def cold_start_sample(config: str, request: dict, inputs: dict) -> dict:
    """One fresh container: one request up to its first DiT step, then stop."""
    import tempfile
    import threading
    from core_infra.runtime import cold_start as measure
    from models.minimax_h3_official.native_pipeline import MiniMaxH3Pipeline
    work = Path(tempfile.mkdtemp(prefix='h3-cold-start-'))
    os.environ.update(measure.environment(work / 'first-dit'))
    request = with_inputs(request, inputs, work)
    errors = []

    def run():
        try:
            MiniMaxH3Pipeline.from_prepared(config).run(request, work / 'output')
        except BaseException as error:  # noqa: BLE001 - reported as this sample's failure
            errors.append(repr(error))
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return measure.sample(STARTED_NS[0], measure.wait(work / 'first-dit', thread.is_alive), ''.join(errors))
