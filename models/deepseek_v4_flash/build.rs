use std::{collections::hash_map::DefaultHasher, env, hash::{Hash, Hasher}, path::{Path, PathBuf}, process::Command};

// Host-side code compiled against the reference libraries' own sources (see kernels/*.cu).
// DSV4_CHECK_ONLY=1 (or no `cuda` feature) skips native compilation for type checks.
//
//   DSV4_FLASHMLA_ROOT   sgl-project/FlashMLA checkout (pinned in reference/native_dev.py)
//   DSV4_FLASHINFER_ROOT the reference image's flashinfer data/{include,csrc} (reference/sglang_b300.py
//                        --export-sources-, copied to /dsv4/deps/flashinfer)
//   DSV4_MOE_BUILD       flashinfer's JIT build of its trtllm-gen MoE module on B300
//                        (--export-moe-build-: generated include root + vendored include trees)
fn main() {
    for key in ["DSV4_CHECK_ONLY", "CUDA_HOME", "DSV4_FLASHMLA_ROOT", "DSV4_FLASHINFER_ROOT", "DSV4_MOE_BUILD", "DSV4_BUILD_JOBS"] {
        println!("cargo:rerun-if-env-changed={key}");
    }
    for f in ["kernels/flashmla_host.cu", "kernels/redirect.cpp", "kernels/moe_host.cu", "kernels/cublaslt.cpp", "kernels/flashmla_decode.cu",
              "kernels/cublas.cpp", "kernels/exact_ops.cu"] {
        println!("cargo:rerun-if-changed={f}");
    }
    if env::var_os("CARGO_FEATURE_CUDA").is_none() { return; }
    if env::var_os("DSV4_CHECK_ONLY").is_some() {
        // exact.rs embeds the compiled cubin: a type check gets an empty placeholder.
        std::fs::write(PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("exact_ops.cubin"), b"").expect("write placeholder");
        return;
    }
    let cuda = env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let nvcc = format!("{cuda}/bin/nvcc");

    // core_infra's streaming loader converts dtypes on the device (unused here: the prepared
    // package stores every tensor in the layout it runs in, but model_load links the entry point).
    println!("cargo:rerun-if-changed=../../core_infra/kernels/convert.cu");
    let convert = out.join("libinfra_convert.a");
    assert!(Command::new(&nvcc).args(["--lib", "-O3", "-std=c++17", "-gencode=arch=compute_103a,code=sm_103a", "-Xcompiler", "-fPIC"])
        .arg("../../core_infra/kernels/convert.cu").arg("-o").arg(&convert).status().expect("launch nvcc").success(), "compile convert.cu");
    println!("cargo:rustc-link-lib=static=infra_convert");

    // FlashMLA TmaParams shim.
    let flashmla = PathBuf::from(env::var("DSV4_FLASHMLA_ROOT").unwrap_or_else(|_| "/opt/deps/flashmla".into()));
    let lib = out.join("libdsv4_flashmla_host.a");
    let status = Command::new(&nvcc)
        .args(["--lib", "-O2", "-std=c++20", "-arch=sm_100a", "-Xcompiler=-fPIC", "--expt-relaxed-constexpr",
               "--expt-extended-lambda", "-DNDEBUG", "-DFLASH_MLA_IS_BUILD_ON_CUDA", "-DKERUTILS_IS_BUILD_ON_CUDA",
               "-Wno-deprecated-declarations", "--diag-suppress=445,177,550"])
        .arg(format!("-I{}", flashmla.join("csrc_legacy").display()))
        .arg(format!("-I{}", flashmla.join("csrc/3rdparty/cutlass/include").display()))
        .arg(format!("-I{}", flashmla.join("csrc/3rdparty/cutlass/tools/util/include").display()))
        .arg(format!("-I{}", flashmla.join("csrc/3rdparty/kerutils/include").display()))
        .arg("kernels/flashmla_host.cu").arg("-o").arg(&lib)
        .status().expect("launch nvcc");
    assert!(status.success(), "compile kernels/flashmla_host.cu");

    // flashinfer's trtllm-gen MoE host code (same sources and flags as its JIT module
    // fused_moe_trtllm_sm100, minus the TVM-FFI launcher, which kernels/moe_host.cu replaces).
    let fi = PathBuf::from(env::var("DSV4_FLASHINFER_ROOT").unwrap_or_else(|_| "/dsv4/deps/flashinfer".into()));
    let mb = PathBuf::from(env::var("DSV4_MOE_BUILD").unwrap_or_else(|_| "/dsv4/deps/moe_build".into()));
    let csrc = fi.join("csrc");
    let mut common: Vec<String> = vec!["-D_GLIBCXX_USE_CXX11_ABI=1".into()];
    for d in [mb.join("gen_root_fp4"), csrc.join("nv_internal"), csrc.join("nv_internal/include"), mb.join("data/cccl/cub"),
              mb.join("data/cccl/libcudacxx/include"), mb.join("data/cccl/thrust")] {
        common.push(format!("-I{}", d.display()));
    }
    for d in [PathBuf::from(format!("{cuda}/include")), mb.join("tvm_ffi/include"), fi.join("include"), csrc.clone(), mb.join("data/cutlass/include"),
              mb.join("data/cutlass/tools/util/include"), mb.join("data/spdlog/include")] {
        common.push("-isystem".into());
        common.push(d.display().to_string());
    }
    let cuda_flags: Vec<String> = [
        "--compiler-options=-fPIC", "--expt-relaxed-constexpr", "-static-global-template-stub=false",
        "-DFLASHINFER_ENABLE_FP8_E8M0", "-DFLASHINFER_ENABLE_FP4_E2M1", "-std=c++17", "--threads=1", "-use_fast_math",
        "-Xfatbin=-compress-all", "--compress-mode=size", "-DFLASHINFER_ENABLE_F16", "-DFLASHINFER_ENABLE_BF16",
        "-DFLASHINFER_ENABLE_FP8_E4M3", "-DFLASHINFER_ENABLE_FP8_E5M2", "-DNDEBUG", "-O3", "-DTLLM_GEN_EXPORT_INTERFACE",
        "-DTLLM_GEN_EXPORT_FLASHINFER", "-DTLLM_ENABLE_CUDA", "-DENABLE_BF16", "-DENABLE_FP8", "-DENABLE_FP4",
        "-DCUTLASS_ENABLE_GDC_FOR_SM100=1",
        "-DTLLM_GEN_GEMM_CUBIN_PATH=\"35cb99413a3ebce2e87a03fc04196e528e22c8d5/batched_gemm-fdba669-7e3d0a6/\"",
        "-gencode=arch=compute_103a,code=sm_103a", "-diag-suppress=177,550,2361",
    ].iter().map(|s| s.to_string()).collect();
    let cxx_flags: Vec<String> = ["-fPIC", "-std=c++17", "-Wno-switch-bool", "-DNDEBUG", "-O3", "-w"].iter().map(|s| s.to_string()).collect();
    let mut sources: Vec<PathBuf> = ["nv_internal/cpp/kernels/quantization.cu", "nv_internal/cpp/common/envUtils.cpp",
        "nv_internal/cpp/common/logger.cpp", "nv_internal/cpp/common/stringUtils.cpp", "nv_internal/cpp/common/tllmException.cpp",
        "nv_internal/cpp/common/memoryUtils.cu", "trtllm_fused_moe_runner.cu",
        "fused_moe/trtllm_backend/trtllm_fused_moe_routing_runner.cu", "fused_moe/trtllm_backend/trtllm_fused_moe_routing_deepseek.cu",
        "fused_moe/trtllm_backend/trtllm_fused_moe_routing_llama4.cu", "fused_moe/trtllm_backend/trtllm_fused_moe_routing_custom_block.cu",
        "fused_moe/trtllm_backend/trtllm_fused_moe_routing_custom_cluster.cu",
        "fused_moe/trtllm_backend/trtllm_fused_moe_routing_custom_cluster_large.cu",
        "fused_moe/trtllm_backend/trtllm_fused_moe_routing_custom_entry.cu", "fused_moe/trtllm_backend/trtllm_fused_moe_routing_common.cu",
        "fused_moe/trtllm_backend/trtllm_fused_moe_dev_kernel.cu", "trtllm_batched_gemm_runner.cu"]
        .iter().map(|s| csrc.join(s)).collect();
    sources.push(PathBuf::from("kernels/moe_host.cu").canonicalize().unwrap());
    // Objects are cached by (source, size, mtime, flags, the generated kernel-config header) so
    // editing one shim does not rebuild flashinfer, while a new config list (filter_metainfo.py)
    // rebuilds the TUs that embed it.
    let cache = out.join("moe_objects");
    std::fs::create_dir_all(&cache).unwrap();
    let metainfo = std::fs::metadata(mb.join("gen_root_fp4/flashinferMetaInfo.h")).map(|m| (m.len(), m.modified().ok())).ok();
    let jobs: Vec<(PathBuf, PathBuf, Vec<String>, String)> = sources.iter().map(|src| {
        let is_cuda = src.extension().map(|e| e == "cu").unwrap_or(false);
        let mut flags = common.clone();
        flags.extend(if is_cuda { cuda_flags.clone() } else { cxx_flags.clone() });
        let meta = std::fs::metadata(src).unwrap_or_else(|e| panic!("{}: {e}", src.display()));
        let mut h = DefaultHasher::new();
        (src, meta.len(), meta.modified().ok(), &flags, &metainfo).hash(&mut h);
        let obj = cache.join(format!("{}-{:016x}.o", src.file_stem().unwrap().to_string_lossy(), h.finish()));
        (src.clone(), obj, flags, if is_cuda { nvcc.clone() } else { "c++".into() })
    }).collect();
    // A few at a time: the TUs that include flashinferMetaInfo.h (6966 kernel configs) take GBs each.
    let pending: Vec<_> = jobs.iter().filter(|(_, obj, _, _)| !obj.exists()).cloned().collect();
    let parallel: usize = env::var("DSV4_BUILD_JOBS").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
    for chunk in pending.chunks(parallel.max(1)) {
        let handles: Vec<_> = chunk.iter().cloned().map(|(src, obj, flags, cc)| {
            std::thread::spawn(move || {
                let tmp = obj.with_extension("tmp.o");
                // Unlimited stack: gcc recurses deeply through flashinferMetaInfo.h's initializer list.
                let o = Command::new("sh").arg("-c").arg("ulimit -s unlimited 2>/dev/null; exec \"$0\" \"$@\"").arg(&cc)
                    .args(&flags).arg("-c").arg(&src).arg("-o").arg(&tmp).output().expect("launch compiler");
                if !o.status.success() {
                    panic!("compile {}:\n{}{}", src.display(), String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr));
                }
                std::fs::rename(&tmp, &obj).unwrap();
            })
        }).collect();
        for h in handles { h.join().expect("MoE host compile"); }
    }
    let moe_lib = out.join("libdsv4_moe.a");
    let _ = std::fs::remove_file(&moe_lib);
    let mut ar = Command::new("ar");
    ar.arg("crs").arg(&moe_lib);
    for (_, obj, _, _) in &jobs { ar.arg(obj); }
    assert!(ar.status().unwrap().success());

    // FlashMLA's sparse-decode launchers (+ scheduling metadata and combine) for this GPU (sm_103a),
    // with the shim that drives them like sparse_attn_decode_interface.
    let fm_flags: Vec<String> = ["-O3", "-std=c++20", "-gencode=arch=compute_103a,code=sm_103a", "-Xcompiler=-fPIC", "--expt-relaxed-constexpr",
        "--expt-extended-lambda", "-DNDEBUG", "-DFLASH_MLA_IS_BUILD_ON_CUDA", "-DKERUTILS_IS_BUILD_ON_CUDA", "-Wno-deprecated-declarations",
        "--diag-suppress=445,177,550", "-w"].iter().map(|s| s.to_string()).chain([
        format!("-I{}", flashmla.join("csrc_legacy").display()),
        format!("-I{}", flashmla.join("csrc/3rdparty/cutlass/include").display()),
        format!("-I{}", flashmla.join("csrc/3rdparty/cutlass/tools/util/include").display()),
        format!("-I{}", flashmla.join("csrc/3rdparty/kerutils/include").display())]).collect();
    let legacy = flashmla.join("csrc_legacy/kernels");
    let fm_sources = [legacy.join("sm100/decode/sparse/head64/instantiations/v4_h64.cu"),
                      legacy.join("sm100/decode/sparse/head64/instantiations/v4_h64_no_split.cu"),
                      legacy.join("smxx/decode/combine/combine.cu"),
                      legacy.join("smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu"),
                      PathBuf::from("kernels/flashmla_decode.cu").canonicalize().unwrap()];
    let fm_jobs: Vec<(PathBuf, PathBuf, Vec<String>, String)> = fm_sources.iter().map(|src| {
        let meta = std::fs::metadata(src).unwrap_or_else(|e| panic!("{}: {e}", src.display()));
        let mut h = DefaultHasher::new();
        (src, meta.len(), meta.modified().ok(), &fm_flags).hash(&mut h);
        (src.clone(), cache.join(format!("fm-{}-{:016x}.o", src.file_stem().unwrap().to_string_lossy(), h.finish())), fm_flags.clone(), nvcc.clone())
    }).collect();
    let pending: Vec<_> = fm_jobs.iter().filter(|(_, obj, _, _)| !obj.exists()).cloned().collect();
    let handles: Vec<_> = pending.into_iter().map(|(src, obj, flags, cc)| std::thread::spawn(move || {
        let tmp = obj.with_extension("tmp.o");
        let o = Command::new(&cc).args(&flags).arg("-c").arg(&src).arg("-o").arg(&tmp).output().expect("launch nvcc");
        if !o.status.success() { panic!("compile {}:\n{}{}", src.display(), String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr)); }
        std::fs::rename(&tmp, &obj).unwrap();
    })).collect();
    for h in handles { h.join().expect("FlashMLA decode compile"); }
    let fm_lib = out.join("libdsv4_flashmla_decode.a");
    let _ = std::fs::remove_file(&fm_lib);
    let mut ar = Command::new("ar");
    ar.arg("crs").arg(&fm_lib);
    for (_, obj, _, _) in &fm_jobs { ar.arg(obj); }
    assert!(ar.status().unwrap().success());
    println!("cargo:rustc-link-lib=static=dsv4_flashmla_decode");

    // The launch redirection (linked whole so its runtime-entry definitions win over libcudart's).
    let redirect = out.join("redirect.o");
    assert!(Command::new("c++").args(["-std=c++17", "-O2", "-fPIC", "-c", "kernels/redirect.cpp"])
        .arg(format!("-I{cuda}/include")).arg("-o").arg(&redirect).status().unwrap().success());
    assert!(Command::new("ar").arg("crs").arg(out.join("libdsv4_redirect.a")).arg(&redirect).status().unwrap().success());
    // torch's cuBLASLt call path (loads the reference libcublasLt at run time).
    let lt = out.join("cublaslt.o");
    assert!(Command::new("c++").args(["-std=c++17", "-O2", "-fPIC", "-c", "kernels/cublaslt.cpp"])
        .arg(format!("-I{cuda}/include")).arg("-o").arg(&lt).status().unwrap().success());
    assert!(Command::new("ar").arg("crs").arg(out.join("libdsv4_cublaslt.a")).arg(&lt).status().unwrap().success());
    println!("cargo:rustc-link-lib=static=dsv4_cublaslt");
    // torch's fp32 cuBLAS GEMM path (loads the reference libcublas at run time).
    let blas = out.join("cublas.o");
    assert!(Command::new("c++").args(["-std=c++17", "-O2", "-fPIC", "-c", "kernels/cublas.cpp"])
        .arg(format!("-I{cuda}/include")).arg("-o").arg(&blas).status().unwrap().success());
    assert!(Command::new("ar").arg("crs").arg(out.join("libdsv4_cublas.a")).arg(&blas).status().unwrap().success());
    println!("cargo:rustc-link-lib=static=dsv4_cublas");
    // The exact element-wise ops (crate::exact), one cubin loaded at run time; no FMA contraction.
    let status = Command::new(&nvcc).args(["-cubin", "-O3", "-fmad=false", "-arch=sm_103a", "kernels/exact_ops.cu", "-o"])
        .arg(out.join("exact_ops.cubin")).status().expect("launch nvcc");
    assert!(status.success(), "compile kernels/exact_ops.cu");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static:+whole-archive=dsv4_redirect");
    println!("cargo:rustc-link-lib=static=dsv4_flashmla_host");
    println!("cargo:rustc-link-lib=static=dsv4_moe");
    println!("cargo:rustc-link-search=native={cuda}/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=dl");
    println!("cargo:rustc-link-lib=dylib=cuda");
    let _ = Path::new("");
}
