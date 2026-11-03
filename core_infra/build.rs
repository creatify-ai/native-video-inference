use std::{env,path::PathBuf,process::Command};
fn main() {
    for key in ["NATIVE_CORE_CHECK_ONLY","NATIVE_CORE_TORCH_INCLUDE"] {println!("cargo:rerun-if-env-changed={key}");}
    println!("cargo:rerun-if-changed=kernels/normal.cpp");
    if env::var_os("CARGO_FEATURE_AVX_NOISE").is_none() || env::var_os("NATIVE_CORE_CHECK_ONLY").is_some(){return;}
    assert_eq!(env::var("CARGO_CFG_TARGET_ARCH").unwrap(),"x86_64","native noise requires the pinned x86-64 backend");
    let include=env::var("NATIVE_CORE_TORCH_INCLUDE").expect("set NATIVE_CORE_TORCH_INCLUDE to the pinned offline vector headers");
    let out=PathBuf::from(env::var_os("OUT_DIR").unwrap());let object=out.join("normal.o");
    assert!(Command::new("c++").args(["-std=c++17","-O2","-fPIC","-mavx2","-mfma","-DCPU_CAPABILITY_AVX2",
        "-fno-math-errno","-fno-trapping-math","-I"]).arg(include).args(["-c","kernels/normal.cpp","-o"]).arg(&object).status().unwrap().success());
    assert!(Command::new("ar").arg("rcs").arg(out.join("libnative_normal.a")).arg(object).status().unwrap().success());
    println!("cargo:rustc-link-search=native={}",out.display());
    println!("cargo:rustc-link-lib=static=native_normal");println!("cargo:rustc-link-lib=stdc++");
}
