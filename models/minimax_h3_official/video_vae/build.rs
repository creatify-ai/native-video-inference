use std::{env,path::PathBuf,process::Command};
fn main(){
    println!("cargo:rerun-if-env-changed=H3_CHECK_ONLY");
    // Type checking only: never produces a linked inference executable.
    if std::env::var_os("H3_CHECK_ONLY").is_some() { return; }
    println!("cargo:rerun-if-changed=../kernels/video_vae");
    println!("cargo:rerun-if-changed=../kernels/video_vae/pixels.cuh");
    let cuda=env::var("CUDA_HOME").unwrap_or("/usr/local/cuda".into());
    let blas=env::var("H3_CUBLAS_ROOT").expect("H3_CUBLAS_ROOT");
    let cudnn=env::var("H3_CUDNN_ROOT").expect("H3_CUDNN_ROOT");
    let frontend=env::var("H3_CUDNN_FRONTEND").expect("H3_CUDNN_FRONTEND");
    let out=PathBuf::from(env::var_os("OUT_DIR").unwrap());
    // Identical shared ABI headers; model-specific kernels retain their original bytes.
    let status=Command::new(format!("{cuda}/bin/nvcc"))
        .args(["--lib","-O3","-std=c++17","-arch=sm_100","-Xcompiler","-fPIC","-Wno-deprecated-declarations","-I../kernels/common"])
        .arg(format!("-I{blas}/include")).args(["../kernels/video_vae/linear.cu","-o"])
        .arg(out.join("libh3_video_linear.a")).status().unwrap();
    assert!(status.success());
    assert!(Command::new(format!("{cuda}/bin/nvcc"))
        .args(["--lib","-O3","-std=c++17","-arch=sm_100","-Xcompiler","-fPIC"])
        .args(["../kernels/video_vae/encoder_ops.cu","../kernels/video_vae/outer.cu","../kernels/video_vae/outer_dynamic.cu","../kernels/video_vae/outer_small.cu","../kernels/video_vae/decoder_generic.cu","../kernels/video_vae/stream.cu","-o"]).arg(out.join("libh3_video_encoder.a"))
        .status().unwrap().success());
    let mut cc=Command::new("g++");
    cc.args(["-c","-O2","-std=c++17","-fPIC","-I../kernels/common"])
      .args([format!("-I{frontend}/include"),format!("-I{cudnn}/include"),format!("-I{cuda}/include")]);
    for root in env::var("H3_VENDOR_INCLUDES").unwrap().split(':'){cc.arg(format!("-I{root}"));}
    assert!(cc.args(["../kernels/video_vae/dense.cpp","-o"]).arg(out.join("dense.o")).status().unwrap().success());
    assert!(Command::new("ar").args(["crs"]).arg(out.join("libh3_video_dense.a")).arg(out.join("dense.o")).status().unwrap().success());
    let mut conv=Command::new("g++");
    conv.args(["-c","-O2","-std=c++17","-fPIC","-I../kernels/common"])
      .args([format!("-I{frontend}/include"),format!("-I{cudnn}/include"),format!("-I{cuda}/include")]);
    for root in env::var("H3_VENDOR_INCLUDES").unwrap().split(':'){conv.arg(format!("-I{root}"));}
    assert!(conv.args(["../kernels/video_vae/conv.cpp","-o"]).arg(out.join("conv.o")).status().unwrap().success());
    assert!(Command::new("ar").args(["crs"]).arg(out.join("libh3_video_conv.a")).arg(out.join("conv.o")).status().unwrap().success());
    for path in ["../kernels/video_vae/linear.cu","../kernels/common/linear.h","../kernels/video_vae/dense.cpp","../kernels/video_vae/encoder_ops.cu","../kernels/video_vae/outer.cu","../kernels/video_vae/outer.h","../kernels/video_vae/outer_dynamic.cu","../kernels/video_vae/outer_dynamic.h","../kernels/video_vae/conv.cpp","../kernels/common/conv.h"]{println!("cargo:rerun-if-changed={path}");}
    println!("cargo:rustc-link-search=native={}",out.display());
    println!("cargo:rustc-link-lib=static=h3_video_linear");
    println!("cargo:rustc-link-lib=static=h3_video_dense");
    println!("cargo:rustc-link-lib=static=h3_video_encoder");
    println!("cargo:rustc-link-lib=static=h3_video_conv");
    for root in [format!("{cuda}/lib64"),format!("{blas}/lib"),format!("{cudnn}/lib")]{
        println!("cargo:rustc-link-search=native={root}");println!("cargo:rustc-link-arg=-Wl,-rpath,{root}");
    }
    println!("cargo:rustc-link-search=native={cuda}/lib64/stubs");
    for lib in ["cuda","cudart","cublas","cublasLt","cudnn","nvrtc","stdc++"]{println!("cargo:rustc-link-lib=dylib={lib}");}
}
