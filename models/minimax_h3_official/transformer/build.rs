use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-env-changed=H3_CHECK_ONLY");
    println!("cargo:rerun-if-env-changed=H3_CUDA_ARCH");
    // Type checking only: never produces a linked inference executable.
    if std::env::var_os("H3_CHECK_ONLY").is_some() { return; }
    for path in ["../kernels/transformer/flashinfer_bridge.cu", "../kernels/transformer/ops.cu", "../kernels/common/linear.cu", "../kernels/common/linear.h"] {
        println!("cargo:rerun-if-changed={path}");
    }
    println!("cargo:rerun-if-changed=../kernels/transformer/swiglu_opt.cu");
    println!("cargo:rerun-if-changed=../kernels/transformer/production_fusions.cu");
    println!("cargo:rerun-if-changed=../kernels/transformer/long_video.cu");
    println!("cargo:rerun-if-changed=../kernels/transformer/h3_pw_norm_mod.cu");
    println!("cargo:rerun-if-changed=../kernels/transformer/qk_rope_opt.cu");
    println!("cargo:rerun-if-changed=../kernels/transformer/pointwise_policy.cu");
    println!("cargo:rerun-if-changed=../kernels/transformer/swiglu_opt.cuh");
    println!("cargo:rerun-if-changed=../kernels/transformer/bsa_pack_opt.cuh");
    println!("cargo:rerun-if-changed=../kernels/transformer/bsa_exact.cpp");
    for name in ["CUDA_HOME", "H3_BLK64_ROOT", "H3_CUTLASS_ROOT", "H3_CUBLAS_ROOT", "H3_BSA_EXACT_ROOT"] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    if env::var_os("CARGO_FEATURE_CUDA").is_none() { return; }
    // `cargo check` can type-check the Rust FFI on a non-CUDA development host.
    // A linked executable still requires every native symbol and cannot use this.
    if env::var_os("H3_RUNTIME_SKIP_NATIVE_BUILD_FOR_CHECK").is_some() { return; }
    assert_eq!(env::var("CARGO_CFG_TARGET_OS").unwrap(), "linux");
    let cuda = env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
    let subblock = env::var("H3_SUBBLOCK_AOT_ROOT").expect("H3_SUBBLOCK_AOT_ROOT from models/minimax_h3_official/build.py");
    
    let blas = env::var("H3_CUBLAS_ROOT").expect("H3_CUBLAS_ROOT from models/minimax_h3_official/build.py");
    let nccl=env::var("H3_NCCL_ROOT").expect("H3_NCCL_ROOT from build helper");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let arch = env::var("H3_CUDA_ARCH").unwrap_or_else(|_| "103a".into());
    assert!(matches!(arch.as_str(), "100a" | "103a"), "H3_CUDA_ARCH must be 100a or 103a");
    // Link only the unmodified public FlashInfer SM100 blk64 AOT export.
    assert!(env::var_os("H3_BSA_EXACT_ROOT").is_none(), "public H3 uses the unmodified FlashInfer SubBlock export");
    assert_eq!(arch,"100a","this public SubBlock export targets B200");
    for name in ["export.json","h3_subblock_v07.h","h3_subblock_v07.o","libcute_dsl_runtime.so"] {
        assert!(PathBuf::from(&subblock).join(name).is_file(),"missing SubBlock AOT file {name}");
        println!("cargo:rerun-if-changed={subblock}/{name}");
    }
    println!("cargo:rerun-if-env-changed=H3_SUBBLOCK_AOT_ROOT");
    let gencode = format!("-gencode=arch=compute_{arch},code=sm_{arch}");
    for (name, source) in [
        ("h3_bsa", "../kernels/transformer/flashinfer_bridge.cu"),
        ("h3_ops", "../kernels/transformer/ops.cu"),
        ("h3_cp_layout", "../kernels/transformer/cp_layout.cu"),
        ("h3_cp_comm", "../kernels/transformer/h3_comm.cpp"),
        ("h3_linear", "../kernels/common/linear.cu"),
        ("h3_pointwise_0", "../kernels/transformer/swiglu_opt.cu"),
        ("h3_pointwise_1", "../kernels/transformer/h3_pw_norm_mod.cu"),
        ("h3_pointwise_2", "../kernels/transformer/qk_rope_opt.cu"),
        ("h3_pointwise_3", "../kernels/transformer/pointwise_policy.cu"),
        ("h3_production_fusions", "../kernels/transformer/production_fusions.cu"),
        ("h3_long_video", "../kernels/transformer/long_video.cu"),
    ] {
        let mut compiler = Command::new(format!("{cuda}/bin/nvcc"));
        compiler.arg("-I../kernels/common");
        // Preserve each library's independently validated compiler contract.
        // FlashInfer's fast-math flags must not leak into the eager small ops.
        match name {
            "h3_bsa" => {
                compiler.arg(format!("-DH3_CUDA_TARGET_MINOR={}", if arch == "100a" { 0 } else { 3 }));
                compiler.args(["--lib", "-O3", "-std=c++20", "--expt-relaxed-constexpr",
                    "--expt-extended-lambda", "--use_fast_math", "--threads=4",
                    "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-Wno-deprecated-declarations", "-DCUDA_CTA_RECONFIG_ACTIVATED=1",
                    &gencode, "-Xcompiler", "-fPIC"])
                    .arg(format!("-I{subblock}"));
            },
            "h3_cp_layout" | "h3_ops" | "h3_pointwise_0" | "h3_pointwise_1" | "h3_pointwise_2" | "h3_pointwise_3" | "h3_production_fusions" | "h3_long_video" => {
                compiler.args(["--lib", "-O3", "-std=c++17", &gencode,
                    "-Xcompiler", "-fPIC"]);
            },
            "h3_linear" => {
                compiler.args(["--lib", "-O3", "-std=c++17", &gencode,
                    "-Xcompiler", "-fPIC", "-Wno-deprecated-declarations"])
                    .arg(format!("-I{blas}/include"));
            },
            "h3_cp_comm" => {
                compiler.args(["--lib","-O2","-std=c++17","-Xcompiler","-fPIC"])
                    .arg(format!("-I{nccl}/include"));
            },
            _ => unreachable!(),
        }
        let status = compiler.arg(source).arg("-o").arg(out.join(format!("lib{name}.a")))
            .status().expect("nvcc launch failed");
        assert!(status.success(), "nvcc failed for {name}");
        println!("cargo:rustc-link-lib=static={name}");
    }
    let mut compiler = Command::new(format!("{cuda}/bin/nvcc"));
    compiler.args(["--lib", "-O2", "-std=c++17", "-Xcompiler", "-fPIC"]);
    compiler.arg("-DH3_BSA_EXACT_UNAVAILABLE");
    let status = compiler.arg("../kernels/transformer/bsa_exact.cpp").arg("-o").arg(out.join("libh3_bsa_exact.a"))
        .status().expect("nvcc launch failed");
    assert!(status.success(), "nvcc failed for h3_bsa_exact");
    println!("cargo:rustc-link-lib=static=h3_bsa_exact");
    assert!(Command::new("ar").arg("crs").arg(out.join("libh3_subblock_aot.a"))
        .arg(format!("{subblock}/h3_subblock_v07.o")).status().expect("SubBlock archive launch failed").success());
    println!("cargo:rustc-link-lib=static=h3_subblock_aot");
    println!("cargo:rustc-link-search=native={subblock}");
    println!("cargo:rustc-link-lib=dylib=cute_dsl_runtime");
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{subblock}");
    println!("cargo:rustc-link-search=native={nccl}/lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{nccl}/lib");
    println!("cargo:rustc-link-lib=dylib=nccl");
    println!("cargo:rerun-if-env-changed=H3_NCCL_ROOT");
    for path in ["../kernels/transformer/h3_comm.cpp","../kernels/transformer/h3_comm.h","../kernels/transformer/contract.hpp","../kernels/transformer/cp_layout.cu"] {
        println!("cargo:rerun-if-changed={path}");
    }
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-search=native={cuda}/lib64");
    println!("cargo:rustc-link-search=native={cuda}/lib64/stubs");
    println!("cargo:rustc-link-search=native={blas}/lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{blas}/lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{cuda}/lib64");
    for lib in ["cudart", "cuda", "cublas", "cublasLt", "stdc++", "dl"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
}
