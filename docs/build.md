# Build from source

Run commands from the repository root with Python 3.12 and Rust 1.90.0 (`rust-toolchain.toml`).
`Cargo.lock` pins registry dependencies; `target/` holds build output.

The Modal examples in [`examples/modal/`](../examples/modal/) are the reference build recipes:
each one installs the exact SDK set its model was validated with and compiles the native programs.
Read the image definition at the top of each example when building on your own Linux machine.

## CPU checks

No GPU, CUDA SDK or model weights needed:

```sh
cargo test --locked -p native-core-infra --lib
python -m pip install -r requirements-dev.txt
python tools/check_public.py
```

Type-check every crate and target without compiling native code (Linux):

```sh
H3_CHECK_ONLY=1 WAN_CHECK_ONLY=1 LTX25_CHECK_ONLY=1 NATIVE_CORE_CHECK_ONLY=1 \
  cargo check --locked --workspace --all-targets --all-features
```

The check-only variables suppress CUDA compilation and linking; they never produce a runnable binary.

## GPU build

The validated target is Linux x86-64 with an NVIDIA B200. Compilation itself allocates no GPU and
reads no weights. CPU portability of core does not imply GPU support on other platforms.

Shared inputs: CUDA 13, cuDNN 9.20.0.48, cuDNN frontend commit
`c4a97621eca52fa0c3a1862a411a16be580b25c6`, and a C++17 compiler. PyTorch headers at commit
`cf30153c4c131c8164ee7798e5022d810682e2cb` (v2.13.0) and CUTLASS are compile-time inputs only;
no PyTorch runtime is linked. Library directories need their linker names (`libcublas.so`,
`libcudnn.so`).

| Model | Build | SDK variables |
| --- | --- | --- |
| MiniMax H3 | `H3_CUDA_ARCH=100a python models/minimax_h3_official/build.py [component ...]` | `CUDA_HOME`, `H3_CUBLAS_ROOT`, `H3_CUDNN_ROOT`, `H3_CUDNN_FRONTEND`, `H3_NCCL_ROOT`, `H3_CUTLASS_ROOT`, `H3_TORCH_INCLUDE`, `H3_VENDOR_INCLUDES`, plus the prepared kernel exports in its README |
| Wan 2.1 | `WAN_CUDA_ARCH=100a cargo build --locked --release -p wan21-native --features cuda --bin wan21-server --bin wan21-generate` | `CUDA_HOME`, `H3_CUBLAS_ROOT`, `H3_CUDNN_ROOT`, `H3_CUDNN_FRONTEND`, `H3_CUTLASS_ROOT`, `H3_TORCH_INCLUDE`, `H3_VENDOR_INCLUDES` (Wan reuses these names) |
| LTX 2.5 | `cargo build --locked --release -p ltx25-native --features cuda --bin ltx25` | `CUDA_HOME`, `LTX25_CUBLAS_ROOT`, `LTX25_CUDNN_ROOT`, `LTX25_CUDNN_FRONTEND`, `LTX25_CUTLASS_ROOT`, `LTX25_TORCH_INCLUDE`, `LTX25_NATTEN_ROOT`, `LTX25_NATTEN_CUTLASS`, `LTX25_FFMPEG_INCLUDE` |

Rust executables appear in `target/release/`; H3's shared libraries appear in
`target/native/minimax_h3_official/`. A build is not a model package: each model also needs its
prepared artifacts, produced once from the official checkpoint by the steps in its README.

For core arithmetic alone:

```sh
NATIVE_CORE_CUBLAS_ROOT=/path/to/cublas \
  python core_infra/build_native.py --arch 100a --output /tmp/native-core
```

## Common failures

| Failure | Check |
| --- | --- |
| Missing SDK variable | Set every input the selected model's build reads |
| Missing CUDA or cuDNN symbol | Use the pinned library versions and make them visible to the loader |
| Missing model or tokenizer file | Run the model's preparation step; the repository ships no weights or vocabularies |
| Existing output directory | Use a fresh output path for each request |
| Check-only build cannot run | Unset the check-only variables and build with the SDKs |
| Backend cache key mismatch | Use the matching plan bundle or let the native builder create one |
