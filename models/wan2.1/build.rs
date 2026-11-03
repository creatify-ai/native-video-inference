use std::{env, path::PathBuf, process::Command};

fn main() {
    for key in ["WAN_CHECK_ONLY", "WAN_CUDA_ARCH", "CUDA_HOME", "H3_CUBLAS_ROOT", "H3_CUDNN_ROOT",
                "H3_CUDNN_FRONTEND", "H3_VENDOR_INCLUDES", "H3_TORCH_INCLUDE", "H3_CUTLASS_ROOT"] {
        println!("cargo:rerun-if-env-changed={key}");
    }
    for path in ["kernels/linear.cu", "kernels/sdpa.cpp", "kernels/eager.cu", "kernels/core_ops.cu", "kernels/dit.cu", "kernels/sched.cu", "kernels/vae.cu", "kernels/rng.cu", "kernels/mem_eff.cu", "kernels/post.cu", "kernels/encoders.cu", "kernels/softmax.cu",
                 "../../core_infra/kernels/linear.cu", "../../core_infra/kernels/linear.h",
                 "../../core_infra/kernels/ops.cu", "../../core_infra/kernels/ops.h", "../../core_infra/kernels/convert.cu"] {
        println!("cargo:rerun-if-changed={path}");
    }
    for path in ["../../core_infra/kernels/sdpa.cpp", "../../core_infra/kernels/plan_cache.h"] {
        println!("cargo:rerun-if-changed={path}");
    }
    // Type checking only: never produces a linked inference executable.
    if env::var_os("WAN_CHECK_ONLY").is_some() || env::var_os("CARGO_FEATURE_CUDA").is_none() { return; }
    assert_eq!(env::var("CARGO_CFG_TARGET_OS").unwrap(), "linux");
    let cuda = env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
    let blas = env::var("H3_CUBLAS_ROOT").expect("H3_CUBLAS_ROOT (CUDA 13 cuBLAS package)");
    let cudnn = env::var("H3_CUDNN_ROOT").expect("H3_CUDNN_ROOT (cuDNN 9.20 package)");
    let frontend = env::var("H3_CUDNN_FRONTEND").expect("H3_CUDNN_FRONTEND (pinned cudnn-frontend checkout)");
    let arch = env::var("WAN_CUDA_ARCH").unwrap_or_else(|_| "100a".into());
    assert_eq!(arch, "100a", "Wan 2.1's bitwise parity is a B200 (sm_100a) contract");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let gencode = format!("-gencode=arch=compute_{arch},code=sm_{arch}");
    for (name, source) in [("wan_linear", "kernels/linear.cu"), ("wan_eager", "kernels/eager.cu"),
                           ("wan_core_ops", "kernels/core_ops.cu"), ("w21_dit", "kernels/dit.cu"), ("w21_sched", "kernels/sched.cu"), ("w21_vae", "kernels/vae.cu"),
                           ("wan_rng", "kernels/rng.cu"), ("infra_convert", "../../core_infra/kernels/convert.cu")] {
        let status = Command::new(format!("{cuda}/bin/nvcc"))
            .args(["--lib", "-O3", "-std=c++17", &gencode, "-Xcompiler", "-fPIC",
                   "-Wno-deprecated-declarations"])
            .arg(format!("-I{blas}/include"))
            .arg(source).arg("-o").arg(out.join(format!("lib{name}.a")))
            .status().expect("nvcc launch failed");
        assert!(status.success(), "nvcc failed for {name}");
        println!("cargo:rustc-link-lib=static={name}");
    }
    // cudnn-frontend is header-only C++17; the host graph code needs no device compilation.
    for (name, source) in [("wan_conv", "kernels/conv.cpp")] {
        let object = out.join(format!("{name}.o"));
        let mut compiler = Command::new("c++");
        compiler.args(["-std=c++17", "-O2", "-fPIC", "-c", source, "-o"]).arg(&object)
            .arg(format!("-I{cuda}/include")).arg(format!("-I{cudnn}/include")).arg(format!("-I{frontend}/include"));
        for include in env::var("H3_VENDOR_INCLUDES").unwrap_or_default().split(':').filter(|p| !p.is_empty()) {
            compiler.arg(format!("-I{include}"));
        }
        assert!(compiler.status().expect("c++ launch failed").success(), "c++ failed for {name}");
        assert!(Command::new("ar").arg("crs").arg(out.join(format!("lib{name}.a"))).arg(&object).status().unwrap().success());
        println!("cargo:rustc-link-lib=static={name}");
    }
    println!("cargo:rerun-if-changed=kernels/conv.cpp");
    println!("cargo:rerun-if-changed=../../core_infra/kernels/conv.cpp");
    let object = out.join("wan_sdpa.o");
    let mut compiler = Command::new("c++");
    compiler.args(["-std=c++17", "-O2", "-fPIC", "-c", "kernels/sdpa.cpp", "-o"]).arg(&object)
        .arg(format!("-I{cuda}/include")).arg(format!("-I{cudnn}/include")).arg(format!("-I{frontend}/include"));
    // The frontend's NVRTC shim needs nvrtc.h from the CUDA 13 wheels (as H3_VENDOR_INCLUDES in H3).
    for include in env::var("H3_VENDOR_INCLUDES").unwrap_or_default().split(':').filter(|p| !p.is_empty()) {
        compiler.arg(format!("-I{include}"));
    }
    let status = compiler.status().expect("c++ launch failed");
    assert!(status.success(), "c++ failed for wan_sdpa");
    assert!(Command::new("ar").arg("crs").arg(out.join("libwan_sdpa.a")).arg(&object).status().unwrap().success());
    println!("cargo:rustc-link-lib=static=wan_sdpa");
    println!("cargo:rustc-link-search=native={cudnn}/lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{cudnn}/lib");
    println!("cargo:rustc-link-lib=dylib=cudnn");
    // The VAE attention kernel: torch's mem-efficient CUTLASS kernel from the pinned wheel's
    // headers with the flags H3 validated for its fp32 instantiation.
    let torch_include = env::var("H3_TORCH_INCLUDE").expect("H3_TORCH_INCLUDE (pinned torch headers)");
    let cutlass = env::var("H3_CUTLASS_ROOT").expect("H3_CUTLASS_ROOT (pinned CUTLASS checkout)");
    let mut compiler = Command::new(format!("{cuda}/bin/nvcc"));
    compiler.args(["--lib", "-O3", "-std=c++17", &gencode, "-Xcompiler", "-fPIC", "--expt-relaxed-constexpr"])
        .arg(format!("-I{torch_include}")).arg(format!("-I{cutlass}/include")).arg(format!("-I{blas}/include"));
    for include in env::var("H3_VENDOR_INCLUDES").unwrap_or_default().split(':').filter(|p| !p.is_empty()) {
        compiler.arg(format!("-I{include}"));
    }
    let status = compiler.arg("kernels/mem_eff.cu").arg("-o").arg(out.join("libwan_mem_eff.a"))
        .status().expect("nvcc launch failed");
    assert!(status.success(), "nvcc failed for wan_mem_eff");
    println!("cargo:rustc-link-lib=static=wan_mem_eff");
    // Reductions: torch's Reduce.cuh kernels from the same pinned headers.
    let mut compiler = Command::new(format!("{cuda}/bin/nvcc"));
    compiler.args(["--lib", "-O3", "-std=c++17", &gencode, "-Xcompiler", "-fPIC", "--expt-relaxed-constexpr",
                   "--expt-extended-lambda", "-Wno-deprecated-declarations"])
        .arg(format!("-I{torch_include}")).arg(format!("-I{blas}/include"));
    for include in env::var("H3_VENDOR_INCLUDES").unwrap_or_default().split(':').filter(|p| !p.is_empty()) {
        compiler.arg(format!("-I{include}"));
    }
    let status = compiler.arg("kernels/post.cu").arg("-o").arg(out.join("libwan_post.a"))
        .status().expect("nvcc launch failed");
    assert!(status.success(), "nvcc failed for wan_post");
    // Request encoders: torch's softmax / reduction device code from the same headers.
    let mut compiler = Command::new(format!("{cuda}/bin/nvcc"));
    compiler.args(["--lib", "-O3", "-std=c++17", &gencode, "-Xcompiler", "-fPIC", "--expt-relaxed-constexpr",
                   "--expt-extended-lambda", "-Wno-deprecated-declarations"])
        .arg(format!("-I{torch_include}")).arg(format!("-I{blas}/include"));
    for include in env::var("H3_VENDOR_INCLUDES").unwrap_or_default().split(':').filter(|p| !p.is_empty()) {
        compiler.arg(format!("-I{include}"));
    }
    let status = compiler.arg("kernels/encoders.cu").arg("kernels/softmax.cu").arg("-o").arg(out.join("libwan_encoders.a"))
        .status().expect("nvcc launch failed");
    assert!(status.success(), "nvcc failed for wan_encoders");
    println!("cargo:rustc-link-lib=static=wan_encoders");
    println!("cargo:rustc-link-lib=static=wan_post");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-search=native={cuda}/lib64");
    // The driver library is injected at run time; link against the toolkit stub.
    println!("cargo:rustc-link-search=native={cuda}/lib64/stubs");
    println!("cargo:rustc-link-search=native={blas}/lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{blas}/lib");
    for lib in ["cublasLt", "cublas", "nvrtc", "cudart", "cuda"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
    println!("cargo:rustc-link-lib=stdc++");
}
