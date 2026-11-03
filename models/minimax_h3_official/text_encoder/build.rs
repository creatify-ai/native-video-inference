fn main() {
    println!("cargo:rerun-if-env-changed=H3_CHECK_ONLY");
    // Type checking only: never produces a linked inference executable.
    if std::env::var_os("H3_CHECK_ONLY").is_some() { return; }
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() { return; }
    let cuda = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
    let root = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let native = root.join("../../../target/native/minimax_h3_official");
    println!("cargo:rustc-link-search=native={}", native.display());
    println!("cargo:rustc-link-lib=dylib=h3_qwen");
    println!("cargo:rustc-link-search=native={cuda}/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", native.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{cuda}/lib64");
    println!("cargo:rerun-if-changed=../../../target/native/minimax_h3_official/libh3_qwen.so");
}
