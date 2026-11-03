use std::{env, path::PathBuf, process::Command};

fn main() {
    for key in ["LTX25_CHECK_ONLY", "CUDA_HOME", "LTX25_CUBLAS_ROOT", "LTX25_CUDNN_ROOT", "LTX25_CUDNN_FRONTEND", "LTX25_TORCH_INCLUDE", "LTX25_CUTLASS_ROOT"] {
        println!("cargo:rerun-if-env-changed={key}");
    }
    for file in ["kernels/ops.cu", "kernels/rng.cu", "kernels/tensor.cu", "kernels/group_norm.cu", "kernels/reduction.cu", "kernels/mem_eff.cu", "kernels/sdpa.cpp", "kernels/linear.cu", "kernels/media.cpp", "kernels/media_encode.cpp", "kernels/resize.cu", "../../core_infra/kernels/linear.cu", "../../core_infra/kernels/conv.cpp", "../../core_infra/kernels/conv.h", "../../core_infra/kernels/plan_cache.h"] {
        println!("cargo:rerun-if-changed={file}");
    }
    if env::var_os("LTX25_CHECK_ONLY").is_some() || env::var_os("CARGO_FEATURE_CUDA").is_none() { return; }
    assert_eq!(env::var("CARGO_CFG_TARGET_OS").unwrap(), "linux", "native LTX builds target Linux (sm_100a)");
    let cuda = env::var("CUDA_HOME").expect("CUDA_HOME");
    let blas = env::var("LTX25_CUBLAS_ROOT").expect("LTX25_CUBLAS_ROOT");
    let cudnn = env::var("LTX25_CUDNN_ROOT").expect("LTX25_CUDNN_ROOT");
    let frontend = env::var("LTX25_CUDNN_FRONTEND").expect("LTX25_CUDNN_FRONTEND");
    let torch = env::var("LTX25_TORCH_INCLUDE").expect("LTX25_TORCH_INCLUDE");
    let cutlass = env::var("LTX25_CUTLASS_ROOT").expect("LTX25_CUTLASS_ROOT");
    let natten = env::var("LTX25_NATTEN_ROOT").expect("LTX25_NATTEN_ROOT");
    let natten_cutlass = env::var("LTX25_NATTEN_CUTLASS").expect("LTX25_NATTEN_CUTLASS");
    for key in ["LTX25_NATTEN_ROOT","LTX25_NATTEN_CUTLASS"]{println!("cargo:rerun-if-env-changed={key}");}
    println!("cargo:rerun-if-changed=kernels/natten.cu");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    for (name, source) in [("ltx25_ops", "kernels/ops.cu"), ("ltx25_rng", "kernels/rng.cu"),
                           ("ltx25_linear", "kernels/linear.cu"), ("ltx25_tensor", "kernels/tensor.cu"),
                           ("ltx25_group_norm","kernels/group_norm.cu"),
                           ("ltx25_reduction","kernels/reduction.cu"), ("ltx25_mem_eff","kernels/mem_eff.cu"),
                           ("ltx25_resize","kernels/resize.cu"), ("infra_convert","../../core_infra/kernels/convert.cu")] {
        println!("cargo:rerun-if-changed={source}");
        assert!(Command::new(format!("{cuda}/bin/nvcc"))
            .args(["--lib", "-O3", "-std=c++17", "-arch=sm_100a", "-Xcompiler=-fPIC", "-Wno-deprecated-declarations", "--expt-relaxed-constexpr", "--expt-extended-lambda"])
            .arg(format!("-I{torch}")).arg(format!("-I{cutlass}/include"))
            .arg(format!("-I{blas}/include")).arg(source).arg("-o").arg(out.join(format!("lib{name}.a")))
            .status().expect("launch nvcc").success(), "compile {source}");
        println!("cargo:rustc-link-lib=static={name}");
    }
    assert!(Command::new(format!("{cuda}/bin/nvcc"))
        .args(["--lib","-O3","-std=c++17","-arch=sm_100a","-Xcompiler=-fPIC","--use_fast_math","--expt-relaxed-constexpr"])
        .arg(format!("-I{natten}/csrc/include")).arg(format!("-I{natten_cutlass}/include"))
        .arg("kernels/natten.cu").arg("-o").arg(out.join("libltx25_natten.a")).status().unwrap().success());
    println!("cargo:rustc-link-lib=static=ltx25_natten");
    let object = out.join("sdpa.o");
    assert!(Command::new("c++").args(["-std=c++17", "-O2", "-fPIC", "-c", "kernels/sdpa.cpp", "-I../../core_infra/kernels"])
        .arg(format!("-I{cuda}/include")).arg(format!("-I{cudnn}/include")).arg(format!("-I{frontend}/include"))
        .arg(format!("-I{blas}/include")).arg("-o").arg(&object).status().unwrap().success());
    assert!(Command::new("ar").arg("crs").arg(out.join("libltx25_sdpa.a")).arg(object).status().unwrap().success());
    println!("cargo:rustc-link-lib=static=ltx25_sdpa");
    println!("cargo:rerun-if-changed=kernels/driver.cpp");
    let driver=out.join("driver.o");
    assert!(Command::new("c++").args(["-std=c++17","-O2","-fPIC","-c","kernels/driver.cpp"])
        .arg("-o").arg(&driver).status().unwrap().success());
    assert!(Command::new("ar").arg("crs").arg(out.join("libltx25_driver.a")).arg(driver).status().unwrap().success());
    println!("cargo:rustc-link-lib=static=ltx25_driver");
    println!("cargo:rustc-link-lib=dylib=dl");
    let conv=out.join("conv.o");
    assert!(Command::new("c++").args(["-std=c++17","-O2","-fPIC","-c","../../core_infra/kernels/conv.cpp"])
        .arg(format!("-I{cuda}/include")).arg(format!("-I{cudnn}/include")).arg(format!("-I{frontend}/include"))
        .arg("-o").arg(&conv).status().unwrap().success());
    assert!(Command::new("ar").arg("crs").arg(out.join("libltx25_conv.a")).arg(conv).status().unwrap().success());
    println!("cargo:rustc-link-lib=static=ltx25_conv");
    let media_include=env::var("LTX25_FFMPEG_INCLUDE").expect("FFmpeg SDK headers");
    let media=out.join("media.o");
    assert!(Command::new("c++").args(["-std=c++17","-O2","-fPIC","-c","kernels/media.cpp"])
        .arg(format!("-I{media_include}")).arg("-o").arg(&media).status().unwrap().success());
    let delivery=out.join("media_encode.o");
    assert!(Command::new("c++").args(["-std=c++17","-O2","-fPIC","-c","kernels/media_encode.cpp"])
        .arg(format!("-I{media_include}")).arg("-o").arg(&delivery).status().unwrap().success());
    assert!(Command::new("ar").arg("crs").arg(out.join("libltx25_media.a")).arg(media).arg(delivery).status().unwrap().success());
    println!("cargo:rustc-link-lib=static=ltx25_media");
    for (key,names) in [("LTX25_MEDIA_LIBS",vec!["libavcodec-","libavutil-","libswscale-","libavformat-"]),
        ("LTX25_PILLOW_LIBS",vec!["libjpeg-"])]{
        println!("cargo:rerun-if-env-changed={key}");
        let root=PathBuf::from(env::var(key).expect("PyAV/Pillow shared library directory"));
        for name in names{
            let path=std::fs::read_dir(&root).unwrap().map(|e|e.unwrap().path()).find(|p|
                p.file_name().unwrap().to_str().unwrap().starts_with(name)).expect("reference codec library");
            println!("cargo:rustc-link-arg={}",path.display());
        }
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}",root.display());
        println!("cargo:rustc-link-arg=-Wl,-rpath-link,{}",root.display());
    }
    println!("cargo:rustc-link-search=native={}", out.display());
    for root in [&cuda, &blas, &cudnn] {
        let lib = if root == &cuda { "lib64" } else { "lib" };
        println!("cargo:rustc-link-search=native={root}/{lib}");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{root}/{lib}");
    }
    println!("cargo:rustc-link-search=native={cuda}/lib64/stubs");
    for lib in ["cudart", "cublas", "cublasLt", "cudnn", "nvrtc", "stdc++"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
}
