#!/usr/bin/env python3
"""Build the native H3 components from models/minimax_h3_official/ (Linux, CPU build).

Uses installed, pinned SDKs. No model export, source overlays, downloads or GPU
allocation. See README.md and docs/build.md for the SDK paths.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

from prepare_subblock_aot import validate_export as validate_subblock

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "target/native/minimax_h3_official"
PACKAGES = {
    "conditioner": "h3-official-conditioner", "dit": "h3-official-native-run",
    "video_vae": "h3-official-video-vae", "audio_vae": "h3-official-audio-vae",
    "pipeline": "h3-official-model-pipeline",
}


def run(command, env):
    subprocess.run([str(x) for x in command], cwd=ROOT, env=env, check=True)


def build(components):
    env = dict(os.environ, CARGO_TARGET_DIR=str(ROOT / "target"))
    arch = env.get("H3_CUDA_ARCH", "103a")
    if arch not in ("100a", "103a"):
        raise ValueError("H3_CUDA_ARCH must be 100a (B200) or 103a (B300)")
    env["H3_CUDA_ARCH"] = arch
    env["NATIVE_CORE_TORCH_INCLUDE"] = env["H3_TORCH_INCLUDE"]
    if "dit" in components and env.get("H3_BSA_EXACT_ROOT"):
        raise ValueError("public H3 links only the FlashInfer SubBlock export; unset H3_BSA_EXACT_ROOT")
    # These are explicit build inputs, never inferred by importing torch.
    cuda = Path(env.get("CUDA_HOME", "/usr/local/cuda"))
    blas = Path(env["H3_CUBLAS_ROOT"])
    cudnn = Path(env["H3_CUDNN_ROOT"])
    frontend = Path(env["H3_CUDNN_FRONTEND"])
    vendor = env.get("H3_VENDOR_INCLUDES", "").split(":")
    includes = [cuda / "include", blas / "include", cudnn / "include",
                frontend / "include", ROOT / "models/minimax_h3_official/kernels/common"]
    includes += [Path(p) for p in vendor if p]
    env["H3_VENDOR_INCLUDES"] = ":".join(str(p) for p in includes)
    library_dirs = [cuda / "lib64", blas / "lib", cudnn / "lib"]
    ldflags = [*[f"-L{p}" for p in library_dirs], f"-L{cuda}/lib64/stubs",
               *[f"-Wl,-rpath,{p}" for p in library_dirs]]
    incflags = [f"-I{p}" for p in includes]
    OUT.mkdir(parents=True, exist_ok=True)
    run(["cargo", "build", "--release", "--locked", "-p", "native-core-infra"], env)
    shutil.copy2(ROOT / "target/release/libnative_core_infra.so", OUT / "libnative_core_infra.so")

    def cpp(name, sources, libraries, opt="-O2", extra=()):
        run(["g++", opt, "-std=c++17", "-shared", "-fPIC", *incflags,
             *sources, *ldflags, *libraries, *extra, "-o", OUT / name], env)

    def nvcc(name, source, flags=()):
        run([cuda / "bin/nvcc", "-O3", "-std=c++17", f"-arch=sm_{arch}",
             "-Xcompiler=-fPIC", *incflags, *flags, source, "-o", OUT / name], env)

    def cargo(component):
        command = ["cargo", "build", "--release", "--locked", "-p", PACKAGES[component]]
        if component in ("conditioner", "dit"):
            command += ["--features", "cuda"]
        if component == "pipeline":
            command += ["--features", "avx-noise"]
        run(command, env)

    if "conditioner" in components:
        nccl = Path(env["H3_NCCL_ROOT"])
        nvcc("qwen_ops.o", ROOT / "models/minimax_h3_official/kernels/text_encoder/ops.cu", ["-c"])
        nvcc("qwen_linear.o", ROOT / "models/minimax_h3_official/kernels/common/linear.cu", ["-c"])
        nvcc("qwen_parallel.o", ROOT / "models/minimax_h3_official/kernels/text_encoder/parallel.cu", ["-c"])
        # Sequence-parallel ranks reuse the DiT's NCCL communicator wrapper.
        cpp("libh3_qwen.so", [ROOT / "models/minimax_h3_official/kernels/text_encoder/attention.cpp",
            ROOT / "models/minimax_h3_official/kernels/text_encoder/conv.cpp", ROOT / "models/minimax_h3_official/kernels/text_encoder/parallel_attention.cpp",
            ROOT / "models/minimax_h3_official/kernels/transformer/h3_comm.cpp", OUT / "qwen_ops.o", OUT / "qwen_linear.o", OUT / "qwen_parallel.o"],
            ["-lcudart", "-lcuda", "-l:libcublas.so.13", "-l:libcublasLt.so.13",
             "-l:libcudnn.so.9", "-l:libnvrtc.so.13", f"-L{nccl}/lib", "-l:libnccl.so.2"],
            extra=[f"-I{nccl}/include", f"-Wl,-rpath,{nccl}/lib"])
        cargo("conditioner")
    if "dit" in components:
        validate_subblock(env["H3_SUBBLOCK_AOT_ROOT"], arch)
        cpp("libh3_dense.so", [ROOT / "models/minimax_h3_official/kernels/transformer/dense.cpp"],
            ["-lcudart", "-lcuda", "-l:libcudnn.so.9", "-l:libnvrtc.so.13"])
        cargo("dit")
    if "video_vae" in components:
        cargo("video_vae")
    if "audio_vae" in components:
        # Audio's original scalar contraction/rounding contract is preserved.
        run([cuda / "bin/nvcc", "-O3", "-std=c++17", "-arch=sm_100",
             "--shared", "-Xcompiler", "-fPIC", ROOT / "models/minimax_h3_official/kernels/audio_vae/kernels.cu",
             "-o", OUT / "libh3_audio_kernels.so"], env)
        cpp("libh3_conv.so", [ROOT / "models/minimax_h3_official/kernels/audio_vae/conv.cpp"],
            ["-l:libcudnn.so.9", "-lcudart"])
        nvcc("audio_encoder_ops.o", ROOT / "models/minimax_h3_official/kernels/audio_vae/encoder_ops.cu", ["-c"])
        nvcc("audio_encoder_linear.o", ROOT / "models/minimax_h3_official/kernels/common/linear.cu", ["-c"])
        nvcc("audio_encoder_attention.o", ROOT / "models/minimax_h3_official/kernels/audio_vae/encoder_attention.cu",
             ["-c", "--expt-relaxed-constexpr", f"-I{env['H3_TORCH_INCLUDE']}", f"-I{env['H3_CUTLASS_ROOT']}/include"])
        cpp("libh3_audio_encoder.so", [OUT / "audio_encoder_ops.o", OUT / "audio_encoder_linear.o",
            OUT / "audio_encoder_attention.o", ROOT / "models/minimax_h3_official/kernels/audio_vae/encoder_linear.cpp"],
            ["-lcudart", "-l:libcublas.so.13", "-l:libcublasLt.so.13"])
        cargo("audio_vae")
    for component in ("pipeline",):
        if component in components:
            cargo(component)
    # A successful CPU build is not an inference/numerical acceptance result.
    print(f"Built {', '.join(components)}; Rust executables: {ROOT / 'target/release'}; native libraries/media: {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("components", nargs="*", metavar="COMPONENT",
                        help="Default: all; choices: " + ", ".join(list(PACKAGES)))
    args = parser.parse_args()
    if set(args.components) - set(PACKAGES):
        parser.error("Unknown component; see --help")
    if not shutil.which("cargo"):
        parser.error("Install the pinned Rust toolchain described in docs/build.md")
    build(args.components or list(PACKAGES))
