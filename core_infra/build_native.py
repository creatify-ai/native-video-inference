"""Build model-independent native math; no GPU allocation or framework import."""
import argparse
import os
from pathlib import Path
import subprocess

def build(output, arch, with_conv=False):
    root = Path(__file__).resolve().parent
    cuda = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    blas = Path(os.environ["NATIVE_CORE_CUBLAS_ROOT"])
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    command = [str(cuda / "bin/nvcc"), "--shared", "-O3", "-std=c++17",
               "-arch=sm_" + arch, "-Xcompiler=-fPIC", "-Wno-deprecated-declarations",
               "-I" + str(blas / "include"), str(root / "kernels/ops.cu"),
               str(root / "kernels/linear.cu"), "-L" + str(blas / "lib"),
               "-lcublas", "-lcublasLt", "-lcudart", "-o", str(output / "libnative_core.so")]
    if with_conv:
        cudnn = Path(os.environ["NATIVE_CORE_CUDNN_ROOT"])
        frontend = Path(os.environ["NATIVE_CORE_CUDNN_FRONTEND"])
        command += [str(root / "kernels/conv.cpp"), "-I" + str(cudnn / "include"),
                    "-I" + str(frontend / "include"), "-L" + str(cudnn / "lib"), "-lcudnn"]
    subprocess.run(command, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arch", choices=["100a", "103a"], default="100a")
    parser.add_argument("--with-conv", action="store_true")
    args = parser.parse_args(); build(args.output, args.arch, args.with_conv)
