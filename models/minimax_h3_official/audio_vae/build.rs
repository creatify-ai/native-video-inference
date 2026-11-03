fn main() {
    println!("cargo:rerun-if-env-changed=H3_CHECK_ONLY");
    // Type checking only: never produces a linked inference executable.
    if std::env::var_os("H3_CHECK_ONLY").is_some() { return; }
    let cuda = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
    let root = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    println!("cargo:rustc-link-search=native={root}/../../../target/native/minimax_h3_official");
    println!("cargo:rustc-link-search=native={cuda}/lib64");
    println!("cargo:rustc-link-lib=dylib=h3_audio_kernels");
    println!("cargo:rustc-link-lib=dylib=h3_audio_encoder");
    println!("cargo:rustc-link-lib=dylib=h3_conv");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{root}/../../../target/native/minimax_h3_official");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{cuda}/lib64");
}
