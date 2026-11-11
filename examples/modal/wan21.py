"""Official Wan 2.1 I2V-14B-720P on Modal, run by the native Rust/CUDA pipeline.

    python -m pip install -r examples/requirements-modal.txt
    modal setup
    modal run examples/modal/wan21.py

With no arguments this runs the official example image and prompt, and prints the cold start. Pass
your own with --image input.png --prompt "..." (a PNG without EXIF); --out names the MP4.

The first run does the one-time work: it builds the images (CUDA 13, the pinned cuBLAS and cuDNN
wheels, Rust, and wan21-server compiled from this checkout), downloads the pinned checkpoint (about
82 GB) into a Modal Volume, writes the prepared directory beside it (models/wan2.1/prepare.py: every
weight serving reads in 64 MiB shards, with a BF16 DiT, about 48 GB) and reads it through once, and
records the request's cuDNN plans on a B200 (again once for each new image size, area or frame
count). Later runs go straight to one B200.

    modal run examples/modal/wan21.py --cold-start 30

measures the cold start of the request on 30 fresh containers, at most ten at a time (--parallel K
changes that), each stopped at its first DiT step, and prints the P50 and P90.

--gpus 2 runs on a B200:2 container: the second GPU runs every unconditional DiT pass of
classifier-free guidance. It copies the DiT weights from the first GPU over the peer link, so the
Volume is read once, as with one GPU. Works with --cold-start too.

Serving pins the prepared manifest's SHA-256 offline. The default is the reference preparation's
digest, which the pinned packages reproduce. Settings: WAN21_VOLUME (default wan21-native),
WAN21_APP_NAME (default wan21-native), WAN21_MANIFEST_SHA256 and WAN21_SERVER_SHA256 (optional
SHA-256 pin of the server executable), COLD_START_REGION (a Modal region for cold_start, such as
us-west; region pinning costs extra).
"""
import json
import os
from pathlib import Path
import tempfile
import time

STARTED_NS = time.monotonic_ns()  # container entry, the origin of cold_start's clock

import modal

REPOSITORY = 'Wan-AI/Wan2.1-I2V-14B-720P'
REVISION = '8823af45fcc58a8aa999a54b04be9abc7d2aac98'
# manifest.json SHA-256 of the reference preparation of REVISION (models/wan2.1/prepare.py).
REFERENCE_MANIFEST_SHA256 = '0261458d7d20ca90ad744df2adddf89afa888566528e3db4d290573758bb5721'
MOUNT = '/wan21'
CHECKPOINT = f"{MOUNT}/huggingface/hub/models--{REPOSITORY.replace('/', '--')}/snapshots/{REVISION}"
PREPARED = f'{MOUNT}/prepared/wan21-i2v-720p-{REVISION[:7]}-s64'  # 64 MiB shards (models/wan2.1/prepare.py)
PLANS = f'{PREPARED}/backend-plans'  # recorded cuDNN plans (record_plans)
ROOT = '/opt/wan21'  # the parts of this repository the containers use
SOURCE = '/opt/wan21-src'  # the build's copy of this repository
SERVER = f'{ROOT}/bin/wan21-server'
# The official I2V example (generate.py and examples/ of Wan-Video/Wan2.1 at the reference commit).
EXAMPLE_IMAGE = ('https://raw.githubusercontent.com/Wan-Video/Wan2.1/9737cba9c1c3c4d04b33fcad41c111989865d315/'
                 'examples/i2v_input.JPG')
EXAMPLE_IMAGE_SHA256 = '077e3d965090c9028c69c00931675f42e1acc815c6eb450ab291b3b72d211a8e'
EXAMPLE_PROMPT = (
    'Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred '
    'feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the '
    'background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white '
    'clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm '
    "sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of "
    'the seaside.')

# The SDK set of the published bitwise results (see models/wan2.1/README.md).
CUDA_IMAGE = 'nvidia/cuda:13.0.2-devel-ubuntu24.04'
RUST = '1.90.0'
CUDA_WHEELS = ('nvidia-cublas==13.1.1.3', 'nvidia-cuda-runtime==13.0.96', 'nvidia-cuda-nvrtc==13.0.88',
               'nvidia-cudnn-cu13==9.20.0.48')
TORCH = 'torch==2.13.0'  # git cf30153c; compile-time headers in the CUDA image, `freqs` in preparation
TORCH_INDEX = 'https://download.pytorch.org/whl/cu130'
FLASHINFER = 'flashinfer-python==0.6.15.post1'  # its bundled CUTLASS headers
CUDNN_FRONTEND = 'c4a97621eca52fa0c3a1862a411a16be580b25c6'
CUDNN_FRONTEND_HEADER_SHA256 = '38aabf33dee54fa59eea46666a15c78775025b83c54be196260e4fd9ca697a06'
# (sample_steps, shift) pairs with recorded UniPC scalar tables (models/wan2.1/schedules.rs).
SCHEDULES = {(40, 5.0), (40, 3.0)}
GPU = 'B200'
GPUS = (1, 2)


def sha256_pin(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(f'{name} must be the offline lowercase SHA-256')
    return value


VOLUME_NAME = os.environ.get('WAN21_VOLUME', 'wan21-native')
MANIFEST_SHA256 = sha256_pin(os.environ.get('WAN21_MANIFEST_SHA256', REFERENCE_MANIFEST_SHA256), 'WAN21_MANIFEST_SHA256')
APP_NAME = os.environ.get('WAN21_APP_NAME', 'wan21-native')
COLD_START_REGION = os.environ.get('COLD_START_REGION') or None  # only read where the app is launched
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)
# Carry the local configuration into the containers, which import this file again.
settings = {'WAN21_VOLUME': VOLUME_NAME, 'WAN21_MANIFEST_SHA256': MANIFEST_SHA256, 'WAN21_APP_NAME': APP_NAME}
if os.environ.get('WAN21_SERVER_SHA256'):
    settings['WAN21_SERVER_SHA256'] = sha256_pin(os.environ['WAN21_SERVER_SHA256'], 'WAN21_SERVER_SHA256')
here = Path(__file__).resolve()
repository = here.parents[2] if modal.is_local() else None


def sdk_environment(site_packages, frontend):
    """build.rs inputs: the toolkit, the pinned wheels' headers and libraries, and header-only SDKs."""
    site_packages = Path(site_packages)
    nvidia = site_packages / 'nvidia'
    return {'CUDA_HOME': '/usr/local/cuda', 'WAN_CUDA_ARCH': '100a',
            'H3_CUBLAS_ROOT': str(nvidia / 'cu13'), 'H3_CUDNN_ROOT': str(nvidia / 'cudnn'),
            'H3_CUDNN_FRONTEND': str(frontend), 'H3_TORCH_INCLUDE': str(site_packages / 'torch/include'),
            'H3_CUTLASS_ROOT': str(site_packages / 'flashinfer/data/cutlass'),
            'H3_VENDOR_INCLUDES': ':'.join(str(p) for p in sorted(nvidia.glob('*/include')))}


def build_native():
    """Image build step (CPU): compile wan21-server and wan21-generate from SOURCE into ROOT/bin."""
    import hashlib
    import shutil
    import site
    import subprocess
    import urllib.request
    site_packages = Path(site.getsitepackages()[0])
    nvidia = site_packages / 'nvidia'
    frontend = Path('/opt/cudnn-frontend')
    archive = Path('/tmp/cudnn-frontend.tar.gz')
    urllib.request.urlretrieve(f'https://github.com/NVIDIA/cudnn-frontend/archive/{CUDNN_FRONTEND}.tar.gz', archive)
    frontend.mkdir(parents=True, exist_ok=True)
    subprocess.run(['tar', '-xzf', str(archive), '-C', str(frontend), '--strip-components=1'], check=True)
    header = frontend / 'include/cudnn_frontend/graph_interface.h'
    if hashlib.sha256(header.read_bytes()).hexdigest() != CUDNN_FRONTEND_HEADER_SHA256:
        raise RuntimeError('the cuDNN frontend checkout differs from its pin')
    # The wheels ship versioned sonames only; the linker needs the plain names.
    for directory, library, soname in (('cu13', 'cublas', 'libcublas.so.13'), ('cu13', 'cublasLt', 'libcublasLt.so.13'),
                                       ('cu13', 'nvrtc', 'libnvrtc.so.13'), ('cu13', 'cudart', 'libcudart.so.13'),
                                       ('cudnn', 'cudnn', 'libcudnn.so.9')):
        link = nvidia / directory / 'lib' / f'lib{library}.so'
        if not link.exists():
            link.symlink_to(soname)
    env = dict(os.environ, **sdk_environment(site_packages, frontend),
               PATH=f"/root/.cargo/bin:/usr/local/cuda/bin:{os.environ['PATH']}", CARGO_TARGET_DIR='/tmp/wan21-target')
    subprocess.run(['nvcc', '--version'], env=env, check=True)
    subprocess.run(['cargo', 'build', '--locked', '--release', '-p', 'wan21-native', '--features', 'cuda',
                    '--bin', 'wan21-server', '--bin', 'wan21-generate'], cwd=SOURCE, env=env, check=True)
    target = Path(ROOT) / 'bin'
    target.mkdir(parents=True, exist_ok=True)
    for program in ('wan21-server', 'wan21-generate'):
        shutil.copy2(f'/tmp/wan21-target/release/{program}', target / program)
    # The programs carry RUNPATHs to these wheels; register them for every other loader lookup too.
    Path('/etc/ld.so.conf.d/wan21.conf').write_text(f"{nvidia / 'cu13/lib'}\n{nvidia / 'cudnn/lib'}\n")
    subprocess.run(['ldconfig'], check=True)
    shutil.rmtree('/tmp/wan21-target')


gpu_image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python='3.12')
    .env({'DEBIAN_FRONTEND': 'noninteractive'})
    .apt_install('build-essential', 'ca-certificates', 'curl', 'ffmpeg')
    .run_commands("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | "
                  f'sh -s -- -y --profile minimal --default-toolchain {RUST}')
    .pip_install(*CUDA_WHEELS)
    .pip_install(TORCH, index_url=TORCH_INDEX, extra_options='--no-deps')
    .pip_install(FLASHINFER, extra_options='--no-deps')
)
# Same Ubuntu 24.04 base as the CUDA image: the CPU `freqs` table uses glibc's cos/sin, and only
# glibc 2.39 reproduces the reference bytes (Debian's 2.36 differs by 1 ULP in 83 entries).
cpu_image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python='3.12')
    .pip_install(TORCH, index_url=TORCH_INDEX)
    .pip_install('numpy==2.2.6', 'safetensors==0.5.3', 'huggingface-hub[hf_xet]==0.36.0', 'pillow==12.0.0')
)
if modal.is_local():
    gpu_image = gpu_image.add_local_dir(repository, SOURCE, copy=True, ignore=[
        '.git', '.git/**', 'target', 'target/**', '.venv', '.venv/**', '**/__pycache__/**', '**/*.pyc',
        '**/.DS_Store', '.github/**', 'docs/**', 'examples/**', 'tests/**', 'tools/**', 'outputs/**', 'artifacts/**',
        # Outputs and inputs saved next to the checkout would otherwise change the image and rebuild it.
        '**/*.mp4', '**/*.wav', '**/*.png', '**/*.jpg', '**/*.jpeg', 'traces/**'])
gpu_image = gpu_image.run_function(build_native, cpu=16, memory=131072, timeout=3 * 3600).env(settings)
cpu_image = cpu_image.env({**settings, 'HF_XET_HIGH_PERFORMANCE': '1'})
if modal.is_local():
    runtime = dict(remote_path=f'{ROOT}/core_infra/runtime', ignore=['**/__pycache__/**'])
    gpu_image = gpu_image.add_local_dir(repository / 'core_infra/runtime', **runtime)
    cpu_image = cpu_image.add_local_dir(repository / 'core_infra/runtime', **runtime)
    cpu_image = cpu_image.add_local_file(repository / 'models/wan2.1/prepare.py', f'{ROOT}/models/wan2.1/prepare.py')


@app.function(image=cpu_image, cpu=16, memory=65536, timeout=4 * 3600, volumes={MOUNT: volume})
def prepare_checkpoint() -> dict:
    """Once per Volume: download the pinned checkpoint (Hugging Face cache layout) and prepare it."""
    import subprocess
    import sys
    from huggingface_hub import snapshot_download
    report = Path(PREPARED) / 'prepare-report.json'
    if report.exists():
        return {**json.loads(report.read_text()), 'fresh': False}
    print('First run: downloading and preparing the checkpoint (once per Volume).', flush=True)
    path = snapshot_download(REPOSITORY, revision=REVISION, cache_dir=f'{MOUNT}/huggingface/hub')
    volume.commit()
    if Path(path) != Path(CHECKPOINT):
        raise RuntimeError(f'unexpected snapshot path {path}')
    subprocess.run([sys.executable, f'{ROOT}/models/wan2.1/prepare.py', '--checkpoint', CHECKPOINT,
                    '--out', PREPARED], check=True)
    volume.commit()
    return {**json.loads(report.read_text()), 'fresh': True}


@app.function(image=cpu_image, cpu=16, memory=32768, timeout=3600, volumes={MOUNT: volume.read_only()})
def warm_volume():
    """Read a new Volume's files once. Its first read is far slower than later ones (the DiT takes
    about 70 s against 10 s), and this keeps that cost out of the first GPU cold start."""
    from concurrent.futures import ThreadPoolExecutor

    def drain(path):
        with open(path, 'rb', buffering=0) as stream:
            while stream.read(1 << 26):
                pass
    roots = {'checkpoint': CHECKPOINT, 'prepared': PREPARED}
    files = json.loads(Path(PREPARED, 'manifest.json').read_text())['files'].values()
    with ThreadPoolExecutor(len(files)) as pool:
        list(pool.map(drain, (f"{roots[entry['root']]}/{entry['path']}" for entry in files)))


@app.function(image=cpu_image, cpu=2, memory=4096, timeout=300)
def example_image() -> bytes:
    """The official example image as a PNG without EXIF."""
    import hashlib
    import io
    import urllib.request
    from PIL import Image
    with urllib.request.urlopen(EXAMPLE_IMAGE) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != EXAMPLE_IMAGE_SHA256:
        raise RuntimeError('the official example image differs from its pin')
    png = io.BytesIO()
    Image.open(io.BytesIO(data)).convert('RGB').save(png, 'PNG')
    return png.getvalue()


def checkpoint_declaration(prepared=PREPARED, checkpoint=CHECKPOINT, pin=None):
    """The trusted checkpoint declaration: offline manifest pin, roots and the tensors Wan uses."""
    pin = sha256_pin(pin if pin is not None else os.environ.get('WAN21_MANIFEST_SHA256', MANIFEST_SHA256),
                     'WAN21_MANIFEST_SHA256')
    return {'manifest': f'{prepared}/manifest.json', 'sha256': pin,
            'roots': {'checkpoint': checkpoint, 'prepared': prepared},
            'prefixes': ['dit.', 't5.', 'vae.', 'clip.visual.']}


def backend_plans():
    """The recorded cuDNN plan bundle on the Volume, or None before `record_plans` has run."""
    import hashlib
    manifest = Path(PLANS, 'manifest.json')
    if not manifest.is_file():
        return None
    return {'path': str(manifest), 'sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()}


def native_service(work_dir='/tmp/wan21', environment_overrides=None, *, server_sha256=None, plans=None, gpus=1):
    """The resident wan21-server on `gpus` GPUs under core_infra's NativeService (light guards, then
    a full audit).

    Weights are read with O_DIRECT: on a Modal Volume buffered reads stop near 5 GB/s, and the
    OS-cache helper would only add buffered reads, so it is off. `plans` is a recorded cuDNN plan
    bundle; a plan it lacks (another driver or request shape) is built as usual."""
    import sys
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from core_infra.runtime.service import NativeService
    environment = {'WAN21_PREPARED': PREPARED, 'WAN21_CHECKPOINT': CHECKPOINT, 'WAN21_REPORT_SHA': '1',
                   'WAN21_WORLD_SIZE': str(gpus), 'NATIVE_LOAD_FILE_IO': 'direct', **(environment_overrides or {})}
    if 'WAN_NO_PREWARM' in os.environ or 'WAN_NO_PREWARM' in environment:
        raise ValueError('Wan declared startup requires backend preparation; remove WAN_NO_PREWARM')
    pin = server_sha256 if server_sha256 is not None else os.environ.get('WAN21_SERVER_SHA256')
    if pin is not None:
        sha256_pin(pin, 'WAN21_SERVER_SHA256')
    return NativeService(SERVER, work_dir, environment, sha256=pin, audit=checkpoint_declaration(), backend_plans=plans,
                         startup_read_ahead_gib=int(os.environ.get('NATIVE_LOAD_AHEAD_GIB', '0')),
                         startup_file_cache=os.environ.get('NATIVE_STARTUP_FILE_CACHE', '0') != '0')


def first_dit_seconds(details):
    """Latency to the first DiT forward from the server's CLOCK_MONOTONIC marker.

    UNIX phase marks in `details['marks']` are diagnostics on another clock and are not used.
    """
    marker = details.get('first_dit_start')
    if details.get('timing_schema') != 'native-first-dit-v1' or not marker or marker.get('clock') != 'CLOCK_MONOTONIC':
        raise ValueError('native-first-dit-v1 CLOCK_MONOTONIC marker is required')
    stamp = marker['monotonic_ns']
    received = details['request_received_monotonic_ns']
    entered = details['container_entered_monotonic_ns']
    if not 0 < entered <= received <= stamp:
        raise ValueError('first DiT timing boundaries are out of order')
    return {'first_dit_start_since_enter': (stamp - entered) / 1e9 if details['cold'] else None,
            'first_dit_start_since_request': (stamp - received) / 1e9}


def plan_key(request, image):
    """What fixes a request's attention shapes: the PNG's size, the area budget and the frame count."""
    width, height = int.from_bytes(image[16:20], 'big'), int.from_bytes(image[20:24], 'big')  # IHDR
    return f"{width}x{height}:{request['max_area']}:{request['frame_num']}"


@app.function(image=gpu_image, gpu=GPU, cpu=16, memory=196608, timeout=1800, volumes={MOUNT: volume})
def record_plans(request: dict, image: bytes) -> int:
    """Once per request shape: run `request` up to its first DiT step with cuDNN recording its attention
    plans, and add them to the Volume's bundle for later cold starts (they build in several seconds; a
    recorded plan loads at once)."""
    import hashlib
    import shutil
    import threading
    recorded = Path(tempfile.mkdtemp(prefix='wan21-plans-'))
    marker = recorded / 'first-dit'
    work = Path(tempfile.mkdtemp(prefix='wan21-request-'))
    (work / 'image.png').write_bytes(image)
    service = native_service(str(work / 'owner'), {
        'NATIVE_CUDNN_PLAN_CACHE_MODE': 'record', 'NATIVE_CUDNN_PLAN_CACHE_DIR': str(recorded),
        'NATIVE_FIRST_DIT_NOTIFY_FILE': str(marker), 'NATIVE_FIRST_DIT_HOLD': '1'}).start()
    spec = {**request, 'image': str(work / 'image.png'), 'out': str(work / 'out.mp4')}
    threading.Thread(target=lambda: service.invoke('generate', json.dumps(spec)), daemon=True).start()
    deadline = time.monotonic() + 1200
    while not (marker.exists() and marker.stat().st_size >= 16):  # the server holds at its first DiT step
        if time.monotonic() > deadline:
            raise TimeoutError('no first DiT step while recording plans')
        time.sleep(0.05)
    volume.reload()
    manifest = Path(PLANS, 'manifest.json')
    bundle = json.loads(manifest.read_bytes()) if manifest.is_file() else {'files': {}, 'requests': []}
    Path(PLANS).mkdir(parents=True, exist_ok=True)
    for plan in sorted(recorded.glob('*.plan')):  # a plan's file name is its full shape key
        shutil.copyfile(plan, Path(PLANS, plan.name))
        bundle['files'][plan.name] = hashlib.sha256(plan.read_bytes()).hexdigest()
    bundle['requests'] = sorted({*bundle.get('requests', []), plan_key(request, image)})
    manifest.write_text(json.dumps(bundle, indent=1))
    volume.commit()
    return len(bundle['files'])


@app.function(image=cpu_image, cpu=1, memory=1024, volumes={MOUNT: volume})
def has_plans(key: str) -> bool:
    volume.reload()
    manifest = Path(PLANS, 'manifest.json')
    return manifest.is_file() and key in json.loads(manifest.read_bytes()).get('requests', [])


@app.cls(image=gpu_image, gpu=GPU, cpu=16, memory=196608, timeout=3600, scaledown_window=30,
         volumes={MOUNT: volume.read_only()})
class Wan21I2V:
    gpus: int = modal.parameter(default=1)

    @modal.enter()
    def start(self):
        # CPU work only: the first request's weight streaming overlaps CUDA initialization.
        self.entered_monotonic_ns = time.monotonic_ns()
        self.service = native_service(plans=backend_plans(), gpus=self.gpus).start()

    @modal.method()
    def generate(self, request: dict, image: bytes) -> dict:
        """`request`: prompt, seed and optional sampling parameters; `image`: PNG bytes."""
        work = Path(tempfile.mkdtemp(prefix='wan21-request-'))
        (work / 'image.png').write_bytes(image)
        spec = {**request, 'image': str(work / 'image.png'), 'out': str(work / 'out.mp4')}
        details = self.service.invoke('generate', json.dumps(spec))
        details['container_entered_monotonic_ns'] = self.entered_monotonic_ns
        try:
            details['first_dit'] = first_dit_seconds(details)
        except ValueError as error:
            details['first_dit'] = str(error)
        details['audit'] = self.service.audit_status()
        details['mp4'] = (work / 'out.mp4').read_bytes()
        return details

    @modal.exit()
    def stop(self):
        if getattr(self, 'service', None) is not None:
            self.service.close()


def on_gpus(function, gpus, **options):
    """`function` (a Modal Function or Cls) on a container with `gpus` B200s, with any other `options`."""
    if gpus != 1:
        options['gpu'] = f'{GPU}:{gpus}'
    return function.with_options(**options) if options else function


@app.local_entrypoint()
def main(image: str = '', prompt: str = '', out: str = 'wan21.mp4', seed: int = 42, steps: int = 40, frames: int = 81,
         max_area: int = 1280 * 720, shift: float = 5.0, guide_scale: float = 5.0, negative_prompt: str = '',
         cold_start: int = 0, gpus: int = 1, trace: str = '', parallel: int = 10):
    """One I2V request. Defaults are the official 720P settings; an empty negative prompt means the official one.
    --cold-start N instead measures the cold start of the request on N fresh containers (run once first).
    --gpus 2 runs the unconditional DiT pass on a second GPU.
    --trace DIR with --cold-start also saves each container's startup trace (NATIVE_STARTUP_TRACE_DIR) under DIR."""
    png = Path(image).read_bytes() if image else None
    if png is not None and not png.startswith(b'\x89PNG\r\n\x1a\n'):
        raise SystemExit('--image must be a PNG (convert other formats first)')
    if image and not prompt:
        raise SystemExit('--image needs a --prompt')
    if frames % 4 != 1 or seed < 0:
        raise SystemExit('--frames must be 4n+1 and --seed non-negative')
    if (steps, shift) not in SCHEDULES:
        raise SystemExit(f'--steps/--shift must be one of {sorted(SCHEDULES)} (recorded UniPC tables)')
    if gpus not in GPUS:
        raise SystemExit('--gpus must be 1 or 2')
    request = request_spec(prompt, seed, steps, frames, max_area, shift, guide_scale, negative_prompt)
    if cold_start:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from core_infra.runtime import cold_start as measure
        png = png or example_image.remote()
        if not has_plans.remote(plan_key(request, png)):
            raise SystemExit('Not prepared for this request yet: run it once without --cold-start first')
        sampler = on_gpus(cold_start_sample, gpus, max_containers=measure.parallel(parallel))
        measure.collect(sampler, cold_start, request, png, gpus, bool(trace),
                        save=trace or None)
        return
    report = prepare_checkpoint.remote()
    if report['manifest_sha256'] != MANIFEST_SHA256:
        raise SystemExit(f"The prepared manifest {report['manifest_sha256']} differs from the pinned "
                         f'{MANIFEST_SHA256}. After checking it, set WAN21_MANIFEST_SHA256 to the new value.')
    if report['fresh']:
        warm_volume.remote()
    png = png or example_image.remote()
    if not has_plans.remote(plan_key(request, png)):
        print('First run of this request shape: recording its cuDNN plans (once).', flush=True)
        record_plans.remote(request, png)
    details = on_gpus(Wan21I2V, gpus)(gpus=gpus).generate.remote(request, png)
    Path(out).write_bytes(details.pop('mp4'))
    first_dit = details['first_dit']
    print(f"\ncold start  {first_dit['first_dit_start_since_enter'] or first_dit['first_dit_start_since_request']:.1f} s"
          ' (container start to first DiT step)' if isinstance(first_dit, dict) else f'\ncold start  {first_dit}')
    print(f"generation  {details['timings']['total']:.1f} s")
    print(f"video       {out} ({details['width']}x{details['height']}, {details['frames']} frames"
          f"{', 2 GPUs' if details.get('world_size') == 2 else ''})")
    print(f"sha256      {details['video_sha256']} (decoded video)")


def request_spec(prompt='', seed=42, steps=40, frames=81, max_area=1280 * 720, shift=5.0, guide_scale=5.0,
                 negative_prompt=''):
    return {'prompt': prompt or EXAMPLE_PROMPT, 'negative_prompt': negative_prompt, 'seed': seed,
            'sample_steps': steps, 'frame_num': frames, 'max_area': max_area, 'shift': shift,
            'guide_scale': guide_scale}


@app.function(image=gpu_image, gpu=GPU, cpu=16, memory=196608, timeout=1800, volumes={MOUNT: volume.read_only()},
              single_use_containers=True, max_containers=10, region=COLD_START_REGION)
def cold_start_sample(request: dict, image: bytes, gpus: int = 1, trace: bool = False) -> dict:
    """One fresh container: `request` up to its first DiT step, then stop."""
    import sys
    import threading
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from core_infra.runtime import cold_start as measure
    work = Path(tempfile.mkdtemp(prefix='wan21-cold-start-'))
    (work / 'image.png').write_bytes(image)
    traces = work / 'trace' if trace else None
    service = native_service(str(work / 'owner'), measure.environment(work / 'first-dit', traces), plans=backend_plans(),
                             gpus=gpus).start()
    spec = {**request, 'image': str(work / 'image.png'), 'out': str(work / 'out.mp4')}
    errors = []

    def run():
        try:
            service.invoke('generate', json.dumps(spec))
        except BaseException as error:  # noqa: BLE001 - reported as this sample's failure
            errors.append(repr(error))
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return measure.sample(STARTED_NS, measure.wait(work / 'first-dit', thread.is_alive), ''.join(errors), traces)
