fn main() {
    println!("cargo:rerun-if-env-changed=H3_CHECK_ONLY");
    // Type checking only: never produces a linked inference executable.
    if std::env::var_os("H3_CHECK_ONLY").is_some() { return; }
    let cuda = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
    println!("cargo:rustc-link-search=native={cuda}/lib64/stubs");
    println!("cargo:rustc-link-lib=dylib=cuda");
}
