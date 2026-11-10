"""Native LTX 2.5 on Modal, from the official checkpoint to an MP4.

    python -m pip install -r examples/requirements-modal.txt
    modal setup
    modal run examples/modal/ltx25.py

With no arguments this runs 1472x832, 49 frames, compares the native latents, frames and waveform
with the PyTorch reference, and prints the cold start. Pass your own request with
--prompt "..." --seed 42 --out sea.mp4, and --image first.jpg for image-to-video.

The first run does the one-time work: it builds the images, downloads the pinned weights into a
Modal Volume, copies their tensors into 64 MiB files (a storage layout of the same checkpoint that
reads several times faster from a Volume) and, once per geometry (--width, --height, --frames,
--fps) and branch, traces the pinned PyTorch reference on one B200 to export a native profile.
Later runs go straight to native inference: the `ltx25` executable alone, without Python or
PyTorch. Prompt, seed and image change per request.

    modal run examples/modal/ltx25.py --cold-start 30

measures the cold start of a prepared profile's own request on 30 fresh containers, at most ten at a
time, each stopped at its first DiT step, and prints the P50 and P90.

--gpus 2 runs on a B200:2 container: the DiT blocks run sequence parallel, each GPU taking half of
the video tokens. The second GPU copies the blocks from the first over the peer link, so the Volume
is read once, as with one GPU. Works with --cold-start too.

Settings: LTX25_VOLUME (default ltx25-native), LTX25_MOUNT (default /vol; profiles record absolute
paths under it, so keep it fixed for a Volume), LTX25_HF_SECRET (a Modal Secret with HF_TOKEN),
COLD_START_REGION (a Modal region for cold_start, such as us-west; region pinning costs extra).
"""
import json
import os
from pathlib import Path
import time

STARTED_NS = [time.monotonic_ns()]  # container entry, consumed by its first request

import modal

LTX_SOURCE = ("https://github.com/Lightricks/LTX-2.git", "be6d6d0a69d11a0a399c462ed92606692e5f8697")
LTX_TREE_SHA256 = "36e5fdc6cedaec12b8d0a76130e1b85f79ab4c1fda9e06115b1ba99d95efffcf"  # source.py over git archive
MODEL_REPO, MODEL_REVISION = "Lightricks/LTX-2.5", "49b093a39c19169a8a81721dcc541e07e945232f"
WEIGHTS = {  # kind -> (file in MODEL_REPO, native checkpoint prefix; None: read whole by the runtime)
    "transformer": ("diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors", "dit."),
    "text_encoder": ("text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors", "text."),  # Gemma 4
    "video_vae": ("vae/ltx-2.5-video-vae-bf16.safetensors", "vae."),
    "spatial_upsampler": ("latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors", "upscale."),
    "audio_vae": ("vae/ltx-2.5-audio-vae-bf16.safetensors", None),
}
# Header-only build inputs: (repository, commit, file, SHA-256 of that file or None).
HEADERS = {
    "cudnn-frontend": ("NVIDIA/cudnn-frontend", "c4a97621eca52fa0c3a1862a411a16be580b25c6",
                       "include/cudnn_frontend/graph_interface.h",
                       "38aabf33dee54fa59eea46666a15c78775025b83c54be196260e4fd9ca697a06"),
    "cutlass": ("NVIDIA/cutlass", "da5e086dab31d63815acafdac9a9c5893b1c69e2", "include/cutlass/cutlass.h", None),
    "natten": ("SHI-Labs/NATTEN", "39375ea1d2f3b1bb284315fa5f79f6ecbebb6382",
               "csrc/include/natten/cuda/fna/kernel_forward.h",
               "8c52e3f90607e3d8362f0a562ba78e71c0c3e1490d51f0675b41d21d991eca42"),
    "natten-cutlass": ("NVIDIA/cutlass", "6b3e607b852f1543dc21323155a2ad71473c8642",
                       "include/cutlass/cutlass.h", None),
}
FFMPEG_SDK = ("https://github.com/PyAV-Org/pyav-ffmpeg/releases/download/9.0.2-1/ffmpeg-manylinux-x86_64.tar.gz",
              "1d9839c5880448e4aa50a6ee6710ab67e44a05c9700c09cd655d7ae182d361dd")
PROMPT = "A lighthouse beam sweeps across a stormy sea at night, waves crashing on the rocks."

VOLUME_NAME = os.environ.get("LTX25_VOLUME", "ltx25-native")
MOUNT = os.environ.get("LTX25_MOUNT", "/vol")
HF_SECRET = os.environ.get("LTX25_HF_SECRET", "")
if not MOUNT.startswith("/") or MOUNT == "/" or ".." in Path(MOUNT).parts:
    raise SystemExit("LTX25_MOUNT must be an absolute subdirectory")
WEIGHTS_DIR = f"{MOUNT}/weights/{MODEL_REVISION}"
DECLARATION = f"{WEIGHTS_DIR}/native-checkpoint.declaration.json"
LAYOUT = f"{WEIGHTS_DIR}/layout-64m"  # the same tensors in 64 MiB files; declaration in layout.json
SOURCE_IDENTITY = "/opt/ltx-ref.identity.json"
SITE = "/usr/local/lib/python3.12/site-packages"
CUDA, SDK = f"{SITE}/nvidia/cu13", "/opt/sdk"
COLD_START_REGION = os.environ.get("COLD_START_REGION") or None  # only read where the app is launched
GPU = "B200"
GPUS = (1, 2)
SETTINGS = {"LTX25_VOLUME": VOLUME_NAME, "LTX25_MOUNT": MOUNT, "LTX25_HF_SECRET": HF_SECRET}
REPO = Path(__file__).resolve().parents[2] if modal.is_local() else None

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
secrets = [modal.Secret.from_name(HF_SECRET)] if HF_SECRET else []
app = modal.App("ltx25-native")

# The pinned PyTorch reference, built from the public LTX source. It runs `prepare`, and its torch
# headers, CUDA/cuDNN wheels and PyAV/Pillow codec libraries are the native build inputs.
reference = (
    modal.Image.from_registry("python:3.12-slim-bookworm")
    .apt_install("git", "curl", "build-essential", "pkg-config", "ffmpeg", "libgl1", "libopengl0", "libglib2.0-0")
    .uv_pip_install("uv==0.12.1")
)
if REPO:
    reference = reference.add_local_file(REPO / "models/ltx25/prepare/source.py", "/opt/ltx25-source.py", copy=True)
reference = reference.run_commands(
    f"git init -q /opt/ltx-git && git -C /opt/ltx-git fetch -q --depth 1 {LTX_SOURCE[0]} {LTX_SOURCE[1]}",
    f"python /opt/ltx25-source.py --repo /opt/ltx-git --commit {LTX_SOURCE[1]} --dest /opt/ltx-ref"
    f" --identity {SOURCE_IDENTITY} --expect {LTX_TREE_SHA256} && rm -rf /opt/ltx-git",
    "cd /opt/ltx-ref && UV_PROJECT_ENVIRONMENT=/usr/local uv sync --package ltx-core --extra natten --no-dev --inexact"
    " && UV_PROJECT_ENVIRONMENT=/usr/local uv sync --package ltx-pipelines --no-dev --inexact",
    "uv pip install --system --no-deps av==19.0.1 pillow==13.0.0",
    "python -c 'import torch, av, PIL; assert (torch.__version__, av.__version__, PIL.__version__)"
    " == (\"2.13.0+cu132\", \"19.0.1\", \"13.0.0\"), (torch.__version__, av.__version__, PIL.__version__)'",
)
capture = reference.env({
    "PYTHONPATH": "/opt/ltx25:/opt/ltx-ref/packages/ltx-core/src:/opt/ltx-ref/packages/ltx-pipelines/src",
    "HF_HOME": "/tmp/huggingface", "PYTORCH_ALLOC_CONF": "expandable_segments:True", **SETTINGS})
if REPO:  # mounted at run time: editing the preparation code does not rebuild the image
    capture = capture.add_local_dir(REPO / "models/ltx25/prepare", "/opt/ltx25/prepare", ignore=["**/__pycache__/**"])


def fetch_headers():
    import hashlib
    import subprocess
    for name, (repository, commit, probe, digest) in HEADERS.items():
        root = Path(SDK) / name
        root.mkdir(parents=True)
        subprocess.run(f"curl -fsSL https://github.com/{repository}/archive/{commit}.tar.gz"
                       f" | tar -xz -C {root} --strip-components=1", shell=True, check=True)
        if digest and hashlib.sha256((root / probe).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"{name} header identity differs")
    archive = Path(SDK) / "ffmpeg.tar.gz"
    subprocess.run(["curl", "-fsSL", "-o", str(archive), FFMPEG_SDK[0]], check=True)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != FFMPEG_SDK[1]:
        raise RuntimeError("FFmpeg SDK archive identity differs")
    (Path(SDK) / "ffmpeg").mkdir()
    subprocess.run(["tar", "-xzf", str(archive), "-C", f"{SDK}/ffmpeg"], check=True)
    archive.unlink()


def build_native():
    import subprocess
    cargo = "cargo build --locked --release -p ltx25-native --features cuda --bin ltx25"
    subprocess.run(f"cd /opt/native-video-inference && {cargo} && install -m 0755 target/release/ltx25"
                   " /usr/local/bin/ltx25 && rm -rf target && ltx25 media-smoke /tmp/smoke.mp4",
                   shell=True, check=True)


LIBRARIES = [f"{CUDA}/lib", f"{SITE}/nvidia/cudnn/lib", f"{SITE}/av.libs", f"{SITE}/pillow.libs"]
native = (
    reference
    .apt_install("libjpeg-dev")  # jpeglib.h for the native image decoder
    .uv_pip_install("cuda-toolkit[nvcc,cccl]==13.2.*", "nvidia-cuda-nvcc==13.2.*", "nvidia-cuda-cccl==13.2.*")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs"
        " | sh -s -- -y --profile minimal --default-toolchain 1.90.0",
        f"cd {CUDA} && test -x bin/nvcc && (test -e lib64 || ln -s lib lib64)"
        " && for l in cublas cublasLt cudart nvrtc; do ln -sf lib$l.so.13 lib/lib$l.so; done"
        f" && ln -sf libcudnn.so.9 {SITE}/nvidia/cudnn/lib/libcudnn.so",
    )
    .run_function(fetch_headers)
    .env({
        "PATH": "/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "CUDA_HOME": CUDA, "LTX25_CUBLAS_ROOT": CUDA, "LTX25_CUDNN_ROOT": f"{SITE}/nvidia/cudnn",
        "LTX25_CUDNN_FRONTEND": f"{SDK}/cudnn-frontend", "LTX25_TORCH_INCLUDE": f"{SITE}/torch/include",
        "LTX25_CUTLASS_ROOT": f"{SDK}/cutlass", "LTX25_NATTEN_ROOT": f"{SDK}/natten",
        "LTX25_NATTEN_CUTLASS": f"{SDK}/natten-cutlass", "LTX25_FFMPEG_INCLUDE": f"{SDK}/ffmpeg/include",
        "LTX25_MEDIA_LIBS": f"{SITE}/av.libs", "LTX25_PILLOW_LIBS": f"{SITE}/pillow.libs",
        "LD_LIBRARY_PATH": ":".join(LIBRARIES), "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    })
)
downloader = modal.Image.debian_slim(python_version="3.12").uv_pip_install("huggingface_hub")
if REPO:  # this repository: the whole workspace for Cargo, the core runtime for checkpoint hashing
    native = native.add_local_dir(REPO, "/opt/native-video-inference", copy=True, ignore=[
        ".git", ".git/**", "target", "target/**", ".venv", ".venv/**", "**/__pycache__/**"])
    downloader = downloader.add_local_dir(REPO / "core_infra/runtime",
                                          "/opt/native-video-inference/core_infra/runtime",
                                          copy=True, ignore=["**/__pycache__/**"])
native = native.run_function(build_native, cpu=16, memory=32768, timeout=3600).env(SETTINGS)
downloader = downloader.env({"PYTHONPATH": "/opt/native-video-inference", **SETTINGS})


@app.function(image=downloader, cpu=16, memory=65536, timeout=4 * 3600, volumes={MOUNT: volume}, secrets=secrets)
def fetch_weights():
    """Download the pinned files, then hash them once: the trusted checkpoint manifest that native
    loading requires, and the whole-file identities that profiles record. Then copy the tensors
    into the 64 MiB layout, checking each against the manifest."""
    import hashlib
    from core_infra.runtime import checkpoint
    volume.reload()
    if not (Path(DECLARATION).exists() and Path(WEIGHTS_DIR, "weights.json").exists()):
        download_weights()
    declaration = json.loads(Path(DECLARATION).read_text())
    if not Path(LAYOUT, "layout.json").exists():
        raw = Path(declaration["manifest"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != declaration["sha256"]:
            raise RuntimeError("checkpoint manifest differs from its declaration")
        checkpoint.layout(json.loads(raw), declaration["roots"], LAYOUT)
        volume.commit()
        warm_layout.remote()
    return {"weights": WEIGHTS_DIR, "declaration": declaration}


@app.function(image=downloader, cpu=16, memory=32768, timeout=3600, volumes={MOUNT: volume.read_only()})
def warm_layout():
    """Read the new layout once. A new Volume's first read is far slower than later ones, and this
    keeps that cost out of the first GPU cold start."""
    from concurrent.futures import ThreadPoolExecutor

    def drain(path):
        with open(path, "rb", buffering=0) as stream:
            while stream.read(1 << 26):
                pass
    with ThreadPoolExecutor(32) as pool:
        list(pool.map(drain, [path for path in Path(LAYOUT).rglob("*.safetensors")]))


def download_weights():
    import hashlib
    from huggingface_hub import hf_hub_download
    from core_infra.runtime import checkpoint
    identities = Path(WEIGHTS_DIR) / "weights.json"
    for relative, _ in WEIGHTS.values():
        print(f"downloading {relative}", flush=True)
        hf_hub_download(MODEL_REPO, relative, revision=MODEL_REVISION, local_dir=WEIGHTS_DIR)
    volume.commit()
    roots = {"weights": WEIGHTS_DIR}
    declared = [("weights", relative, prefix) for relative, prefix in WEIGHTS.values() if prefix]
    manifest = checkpoint.build(roots, declared, workers=4)
    path = f"{WEIGHTS_DIR}/native-checkpoint.json"
    declaration = {"manifest": path, "roots": roots, "sha256": checkpoint.write(manifest, path)}
    Path(DECLARATION).write_text(json.dumps(declaration, indent=1) + "\n")
    files = {}
    for kind, (relative, prefix) in WEIGHTS.items():
        if prefix:
            digest = manifest["files"]["weights:" + relative]["sha256"]
        else:
            digest = hashlib.sha256()
            with open(f"{WEIGHTS_DIR}/{relative}", "rb") as stream:
                while block := stream.read(16 << 20):
                    digest.update(block)
            digest = digest.hexdigest()
        files[kind] = {"file": relative, "bytes": Path(WEIGHTS_DIR, relative).stat().st_size, "sha256": digest}
    identities.write_text(json.dumps(files, indent=1, sort_keys=True) + "\n")
    volume.commit()


@app.function(image=downloader, cpu=1, memory=1024, volumes={MOUNT: volume})
def prepared(name: str) -> dict:
    volume.reload()
    return {"profile": Path(MOUNT, "profiles", name, "profile.json").exists(),
            "layout": Path(LAYOUT, "layout.json").exists(),
            "plans": Path(MOUNT, "profiles", name, "backend-plans", "manifest.json").exists()}


@app.function(image=capture, gpu=GPU, cpu=16, memory=196608, timeout=8 * 3600, volumes={MOUNT: volume})
def prepare_profile(name: str, request: dict, image: bytes | None):
    """Trace the PyTorch reference once and export a native profile to /profiles/NAME."""
    import subprocess
    volume.reload()
    profile, work = Path(MOUNT) / "profiles" / name, Path(MOUNT) / "prepare" / name
    inputs = Path("/tmp/ltx25-inputs")
    inputs.mkdir(parents=True, exist_ok=True)
    (inputs / "request.json").write_text(json.dumps(request))
    command = ["python", "-m", "prepare", "run", "--work", str(work), "--profile", str(profile),
               "--weights", WEIGHTS_DIR, "--source-identity", SOURCE_IDENTITY,
               "--request", str(inputs / "request.json")]
    if image is not None:
        (inputs / "image.jpg").write_bytes(image)
        command += ["--image", str(inputs / "image.jpg")]
    result = subprocess.run(command, cwd="/opt/ltx25")
    volume.commit()
    if result.returncode:
        raise RuntimeError(f"profile preparation failed with exit code {result.returncode}")
    return json.loads((profile / "prepare.json").read_text())


def backend_plans(profile_dir):
    """cuDNN attention plans for this profile: the recorded bundle, or (first run) a directory to
    record into. A plan builds in seconds; a recorded one loads at once. A plan the bundle lacks
    (another driver or cuDNN) is built as usual."""
    import hashlib
    import sys
    import tempfile
    bundle = Path(profile_dir, "backend-plans", "manifest.json")
    if bundle.is_file():
        sys.path.insert(0, "/opt/native-video-inference")
        from core_infra.runtime.plans import prepare_backend_plans
        pin = {"path": str(bundle), "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest()}
        return prepare_backend_plans(pin, tempfile.mkdtemp() + "/plans")["environment"], None
    recorded = Path(tempfile.mkdtemp(prefix="ltx25-plans-"))
    return {"NATIVE_CUDNN_PLAN_CACHE_MODE": "record", "NATIVE_CUDNN_PLAN_CACHE_DIR": str(recorded)}, recorded


def save_plans(recorded, profile_dir):
    import hashlib
    import shutil
    plans = sorted(recorded.glob("*.plan"))
    target = Path(profile_dir, "backend-plans")
    target.mkdir(exist_ok=True)
    for plan in plans:
        shutil.copyfile(plan, target / plan.name)
    files = {plan.name: hashlib.sha256(plan.read_bytes()).hexdigest() for plan in plans}
    (target / "manifest.json").write_text(json.dumps({"files": files}, indent=1))


def native_environment(plans, gpus=1):
    """`ltx25 infer`'s environment on `gpus` GPUs. The model admits the published checkpoint; its
    readers take the same tensors from the layout, with O_DIRECT reads (a Volume serves buffered reads
    at about 5 GB/s per process) and 48 readers."""
    declaration = json.loads(Path(DECLARATION).read_text())
    layout = json.loads(Path(LAYOUT, "layout.json").read_text())
    kernels = Path(MOUNT, "kernel-cache")  # compiled fused elementwise kernels, shared by containers
    try:
        kernels.mkdir(exist_ok=True)
    except OSError:
        pass  # a read-only mount still serves the kernels already there
    return dict(os.environ, NATIVE_FIRST_DIT_TRACE="1", NATIVE_CHECKPOINT_MANIFEST=declaration["manifest"],
                NATIVE_CHECKPOINT_SHA256=declaration["sha256"],
                NATIVE_CHECKPOINT_ROOTS=json.dumps(declaration["roots"], sort_keys=True),
                NATIVE_WEIGHT_LAYOUT_MANIFEST=layout["manifest"], NATIVE_WEIGHT_LAYOUT_SHA256=layout["sha256"],
                NATIVE_WEIGHT_LAYOUT_ROOTS=json.dumps(layout["roots"], sort_keys=True), NATIVE_LOAD_FILE_IO="direct",
                NATIVE_LOAD_WORKERS="48", LTX25_WORLD_SIZE=str(gpus), LTX25_KERNEL_CACHE=str(kernels), **plans)


@app.function(image=native, gpu=GPU, cpu=16, memory=196608, timeout=3600, volumes={MOUNT: volume})
def infer(profile: str, request: dict | None, image: bytes | None, run: str, gpus: int = 1):
    """Run `ltx25 infer`. Without a request, replay the profile's own capture request and compare
    the native latents, frames and waveform with the PyTorch reference's hashes."""
    import hashlib
    import subprocess
    entered = STARTED_NS.pop() if STARTED_NS else time.monotonic_ns()
    volume.reload()
    config = json.loads(Path(profile).read_text())
    if request is None:
        example = json.loads((Path(profile).parent / "request.json").read_text())
        request = {"prompt": example["prompt"], "seed": example["seed"]}
        image = (Path(profile).parent / example["image"]).read_bytes() if "image" in example else None
    if not str(config["gold"]).startswith(MOUNT + "/"):
        raise ValueError(f"{profile} was exported for another mount than {MOUNT}; prepare it again")
    output = Path(MOUNT) / "outputs" / run
    output.mkdir(parents=True)
    if image is not None:
        (output / "image.jpg").write_bytes(image)
        request["image"] = "image.jpg"
    (output / "request.json").write_text(json.dumps(request))
    config["weights"] = WEIGHTS_DIR  # weights always come from this volume
    (output / "profile.json").write_text(json.dumps(config, indent=1))
    plans, recorded = backend_plans(Path(profile).parent)
    environment = native_environment(plans, gpus)
    started = time.monotonic()
    # Progress goes to stderr (container logs); the final JSON report goes to stdout.
    result = subprocess.run(["ltx25", "infer", str(output / "profile.json"), str(output / "request.json"),
                             str(output)], env=environment, stdout=subprocess.PIPE, text=True)
    if recorded and not result.returncode:
        save_plans(recorded, Path(profile).parent)
    volume.commit()
    if result.returncode:
        raise RuntimeError(f"native inference failed with exit code {result.returncode}")
    report = json.loads(result.stdout)
    marker = report.get("first_dit_start") or {}
    summary = {"seconds": round(time.monotonic() - started, 1), "output": str(output),
               "cold_start": (marker["monotonic_ns"] - entered) / 1e9 if marker else None,
               "mp4": report["decoded"].get("mp4"), "latents": report["outputs"],
               "world_size": report["parallel"]["world_size"],
               "sampling_steps_per_stage": report["sampling_steps_per_stage"]}
    manifests = {key: json.loads((Path(config[key]) / "manifest.json").read_text())
                 for key in ("gold", "audio", "video")}
    captured = manifests["gold"]["identity"]["request"]
    if (request["prompt"], request["seed"]) == (captured["prompt"], captured["seed"]) and (
            image is None if "image_sha256" not in captured
            else image is not None and hashlib.sha256(image).hexdigest() == captured["image_sha256"]):
        expected = {f"{label}.latent.raw": manifests["gold"]["tensors"][f"stage.1.out.{label}.latent"]["sha256"]
                    for label in ("video", "audio")}
        expected["audio.f32.raw"] = manifests["audio"]["tensors"]["audio.waveform"]["sha256"]
        expected.update({f"video.frames.{name.rsplit('.', 1)[-1]}.raw": entry["sha256"]
                         for name, entry in manifests["video"]["tensors"].items() if name.startswith("video.pixels.")})
        produced = report["outputs"] + [report["decoded"]["audio"]["output"]] + report["decoded"]["video"]["outputs"]
        actual = {Path(item["path"]).name: item["sha256"] for item in produced}
        summary["matches_pytorch_reference"] = {name: actual.get(name) == digest for name, digest in expected.items()}
    return summary, (output / "output.mp4").read_bytes()


@app.local_entrypoint()
def main(prompt: str = "", image: str = "", seed: int = 42, width: int = 1472, height: int = 832, frames: int = 49,
         fps: int = 24, out: str = "ltx25.mp4", cold_start: int = 0, gpus: int = 1):
    """One request. Without --prompt, replay the profile's capture request and check it against the reference.
    --cold-start N instead measures the cold start of the profile's own request on N fresh containers.
    --gpus 2 runs the DiT blocks sequence parallel on two GPUs."""
    if gpus not in GPUS:
        raise SystemExit("--gpus must be 1 or 2")
    mode = "i2v" if image else "t2v"
    name = profile_name(image, width, height, frames, fps)
    image_bytes = Path(image).read_bytes() if image else None
    state = prepared.remote(name)
    if cold_start:
        import sys
        sys.path.insert(0, str(REPO))
        from core_infra.runtime import cold_start as measure
        if not all(state.values()):
            raise SystemExit(f"{name} is not prepared yet: run `modal run examples/modal/ltx25.py` with the same options")
        profile = f"{MOUNT}/profiles/{name}/profile.json"
        measure.collect(on_gpus(cold_start_sample, gpus), cold_start, profile, gpus)
        return
    if not state["layout"]:
        print("First run: downloading the weights and writing their 64 MiB layout (once).")
        fetch_weights.remote()
    if not state["profile"]:
        print(f"First run for {name}: tracing the reference on a B200 (once).")
        prepare_profile.remote(name, {"mode": mode, "width": width, "height": height, "num_frames": frames,
                                      "frame_rate": fps, "prompt": prompt or PROMPT, "seed": seed}, image_bytes)
    request = {"prompt": prompt, "seed": seed} if prompt else None
    summary, video = on_gpus(infer, gpus).remote(f"{MOUNT}/profiles/{name}/profile.json", request,
                                                 image_bytes if prompt else None, time.strftime("%Y%m%d-%H%M%S"), gpus)
    Path(out).write_bytes(video)
    if summary["cold_start"] is not None:
        print(f"\ncold start  {summary['cold_start']:.1f} s (container start to first DiT step)")
    print(f"generation  {summary['seconds']:.1f} s")
    print(f"video       {out} ({width}x{height}, {frames} frames{', 2 GPUs' if summary['world_size'] == 2 else ''})")
    matches = summary.get("matches_pytorch_reference")
    if matches:
        print("PyTorch     " + ("byte-identical" if all(matches.values()) else f"DIFFERS: {matches}"))
    print(f"outputs     Volume {VOLUME_NAME} at {summary['output']}")


def profile_name(image, width, height, frames, fps):
    return f"{'i2v' if image else 't2v'}-{width}x{height}x{frames}" + ("" if fps == 24 else f"-{fps}fps")


def on_gpus(function, gpus):
    """`function` on a container with `gpus` B200s."""
    return function if gpus == 1 else function.with_options(gpu=f"{GPU}:{gpus}")


@app.function(image=native, gpu=GPU, cpu=16, memory=196608, timeout=1800, volumes={MOUNT: volume.read_only()},
              single_use_containers=True, max_containers=10, region=COLD_START_REGION)
def cold_start_sample(profile: str, gpus: int = 1) -> dict:
    """One fresh container: the profile's own request up to its first DiT step, then stop."""
    import subprocess
    import sys
    import tempfile
    sys.path.insert(0, "/opt/native-video-inference")
    from core_infra.runtime import cold_start as measure
    work = Path(tempfile.mkdtemp(prefix="ltx25-cold-start-"))
    example = json.loads((Path(profile).parent / "request.json").read_text())
    request = {"prompt": example["prompt"], "seed": example["seed"]}
    if "image" in example:
        (work / "image.jpg").write_bytes((Path(profile).parent / example["image"]).read_bytes())
        request["image"] = "image.jpg"
    config = {**json.loads(Path(profile).read_text()), "weights": WEIGHTS_DIR}
    (work / "request.json").write_text(json.dumps(request))
    (work / "profile.json").write_text(json.dumps(config))
    (work / "output").mkdir()
    plans, recorded = backend_plans(Path(profile).parent)
    if recorded:
        raise RuntimeError("this profile has no recorded plans; run `modal run examples/modal/ltx25.py` first")
    environment = {**native_environment(plans, gpus), **measure.environment(work / "first-dit")}
    with open(work / "stderr", "w") as stderr:
        process = subprocess.Popen(["ltx25", "infer", str(work / "profile.json"), str(work / "request.json"),
                                    str(work / "output")], env=environment, stdout=subprocess.DEVNULL, stderr=stderr)
        stamp = measure.wait(work / "first-dit", lambda: process.poll() is None)
        process.kill()
    return measure.sample(STARTED_NS[0], stamp, (work / "stderr").read_text())
