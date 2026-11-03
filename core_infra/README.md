# Core infrastructure (`native-core-infra`)

Model-independent mechanisms for native inference. Core has no model names, shapes or
workflow selection; models depend on core, never the reverse. The
[architecture guide](../docs/architecture.md) defines the model protocol and the
[cold-start guide](../docs/fast-cold-start.md) defines loader policy.

## What core provides

| Area | Modules | Purpose |
| --- | --- | --- |
| Tensors and checkpoints | `tensor`, `checkpoint`, `weight_loader`, `model_load` | Raw tensor descriptors, in-place checkpoint manifests, bounded full-integrity reads, streaming device loads and their peer copies to another GPU (`model-load` feature) |
| Integrity | `integrity`, `guard_table`, `immutable_assets`, `backend_plans` | SHA-pinned light guards, content-addressed host assets, verified backend plan bundles |
| Host and device memory | `host_buffer`, `pinned_weights`, `resident_weights`, `resident_assets`, `reuse_pool`, `allocation_ledger`, `phase_workspace` | Immutable host weights, pinned H2D sources, residency and eviction, bounded allocation reuse |
| Startup executor | `declared_startup`, `startup`, `startup_policy`, `startup_native`, `startup_owner`, `startup_gate`, `startup_input`, `startup_ffi` | One startup policy: models declare components, preparation and real dependencies; core overlaps reads, CPU work and CUDA initialization and gates consumers |
| Serving | `resident`, `backend_service`, `backend_abi`, `native_backend`, `device_context` | Resident native owner process, versioned backend ABI, thread-owned CUDA contexts and peer copies between them |
| IO scheduling | `archive_reader`, `background_io`, `cpu_priority`, `startup_cohort` | Bounded reader groups, cooperative background IO that foreground work can take over |
| Tracing | `startup_trace`, `execution_start` | Opt-in timing spans and a stream-timestamped first-execution marker; no added synchronization |
| Numerics | `rng`, `kernels/` | Pinned MT19937/normal arithmetic; generic BF16/FP32 GEMM, convolution, SDPA and conversion kernels with backend plan caching |

`runtime/` is the Python side: subprocess ownership (`ProcessGroup`), bounded native RPC,
`NativeService` and `resident_service` for serving, `declared_startup.DeclaredStartup` for the
complete startup plan, checkpoint preparation and SHA-verified file publication. Closing a
process group cancels its RPCs, terminates and reaps its processes, and joins preparation work.
The Python runtime needs no ML framework; offline checkpoint preparation uses NumPy only.

## Build and test

```sh
cargo test --locked -p native-core-infra --lib           # CPU only
NATIVE_CORE_CUBLAS_ROOT=/path/to/cublas \
  python core_infra/build_native.py --arch 100a --output /tmp/native-core   # CUDA arithmetic
```

Features: `cuda` links the CUDA kernels, `model-load` adds streaming in-place checkpoint loads,
`avx-noise` enables the AVX normal transform. `NATIVE_CORE_CHECK_ONLY=1` type-checks without
compiling native code.

## Using core from a model

A model crate depends on `native-core-infra` by path, keeps its own tensor wrappers and kernels,
and declares its components to the startup executor. See [porting a model](../docs/porting-a-model.md);
`models/wan2.1` is the worked example.
