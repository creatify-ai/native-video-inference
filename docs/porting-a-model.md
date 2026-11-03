# Port a model

This guide moves a PyTorch inference pipeline into rust-inference. A finished port:

- produces the same latents as the PyTorch reference, byte for byte, on the target GPU;
- loads the model's published checkpoint and gets the framework's [fast cold start](fast-cold-start.md);
- ships a one-shot command-line program and a resident server.

The worked example is the Wan 2.1 I2V-14B-720P port (`models/wan2.1`). On all ten of its validation requests, every recorded tensor and the decoded video
match the official PyTorch pipeline. It runs about 14–15% faster than PyTorch's fastest configuration
without any speed tuning.

## Acceptance

Given the same inputs, every latent must have the same SHA-256 as the reference: the initial noise,
every model forward's input and output, every sampling step's latent, and the final latent.
Decoded pixels are reported. They are also required when they feed later latents, for example
motion frames. There is no tolerance. A mismatch is a bug in one rung of the ladder below.

For the current startup consolidation, preserve each model's previously measured
single-GPU cold start on the same maintained route and fixed input. Keep its
hardware, CPU/RAM, checkpoint bytes, dtype and reader limits. Compare fresh
baseline/candidate distributions; retain every failure and slow sample. An
8-second limit is not the completion gate for this consolidation. Standalone,
native ABI and unified results have different setup scopes and cannot certify
one another.

The startup contract has four rules:

1. The model declares weights, components, required preparation and real dependencies.
2. The framework schedules reads, initialization, upload and dependency waits.
3. Missing required work, a failed preparation or an invalid result blocks its consumer.
4. Maintained entry points consume the same model declaration. An entry-point adapter
   may supply its existing process layout and resource envelope; it must not maintain
   another preload order or readiness policy. CPU preparation must start when
   its inputs exist; waiting for unrelated weight copies before publishing
   those inputs is also an entry-owned scheduling policy.

The complete native API is Rust `declared_startup::{Component, Preparation,
Declaration, DeclaredStartup}` with model `Bindings`. A checkpoint
`checkpoint::Declaration` admits source metadata only; it is not a startup plan.
The process-model Python binding is `core_infra.runtime.declared_startup` and
delegates scheduling and state to that Rust executor.
Declare `Component` and `Preparation` values, including each consumer's required
IDs, CPU work, weight dependencies and result validators. `DeclaredStartup`
admits the whole plan before creating owners. It starts independent work,
releases GPU initialization and checks consumers. Entry adapters bind owned or
resident transports to this same execution policy. H3 is the complete
model-internal example; see its [README](../models/minimax_h3_official/README.md).

`ComponentStartup`, `CommandSchedule`, native staged loaders and
`startup::Preparation` are lower-level mechanisms. Using them alone
does not complete this contract. A native in-process model must bind its complete
work and validators to the Rust executor. Add any missing context, owner or
real-use dependency to core before claiming migration acceptance.
Native models can bind these mechanisms through `startup_native::NativeLoadPlan`
and `startup_owner::OwnerExecutor`. Wan's `startup.rs` and LTX's `declared.rs`
contain bindings for their maintained native entries; their new GPU acceptance remains a
separate gate.

Validators must return exactly `True`. Asset admission, weight readiness and
execution readiness are distinct. Missing required input, failed/cancelled work
and rejected results must block the actual consumer. Numerical parity and each
maintained entry's original-case performance are separate completion gates.

Prepare audit and file-cache metadata from the actual declared checkpoint paths
for each device layout. Shared tensor hashes do not make
two physical paths interchangeable. Include selected DiT buffers and shared weight
roots. Perform this pinned metadata walk during image/config preparation; it reads no weight
payload. A missing audit job rejects the prefetch plan before GPU startup. Keep
this work out of the serving critical path and retain the original reader budget.

## Crate layout

Create `models/<model>/` as its own Cargo package and add it to the workspace.

| Path (Wan 2.1) | Role |
| --- | --- |
| `Cargo.toml` | Package `wan21-native`. Feature `cuda` enables `native-core-infra/model-load`; the binaries require it. |
| `build.rs` | Compiles the model's CUDA kernels and `core_infra/kernels/convert.cu` with `nvcc`. `WAN_CHECK_ONLY=1` skips native compilation for type checks. |
| `frontend.rs` | CPU front end: request parsing, image decoding and resizing, tokenization, size math |
| `t5.rs`, `clip.rs`, `vae.rs`, `dit.rs`, `sampler.rs` | One module per reference component |
| `kernels/` | The model's CUDA kernels and backend bridges |
| `pipeline.rs` | `condition`, `denoise`, `decode`, `generate` and `cold_start` |
| `bin/wan21-playground.rs` | Parity rungs. Each subcommand prints per-tensor SHA-256 comparisons as JSON. |
| `bin/wan21-generate.rs` | One-shot program, equivalent to the reference's generate script |
| `bin/wan21-server.rs` | Resident serving process (`core_infra::resident`) |
| `examples/modal/wan21.py` (repository root) | Modal class over `NativeService` |

Reuse operators from earlier ports when the new reference computes the same way. Wan 2.1 copies the
operators it needs into its own crate, so a change made for one model cannot alter another model's
accepted numerics.

## 1. Freeze the reference

- Use the deployed or official commit, extracted with `git archive`, never a working tree. Record
  its tree hash and the SHA-256 of each weight file in every report.
- Run it on the target GPU with the same CUDA, cuDNN and cuBLAS versions the native build links.
- Replace kernels that cannot be reproduced natively before any model import, and document each
  swap. Wan 2.1 replaces FlashAttention 2 with PyTorch SDPA restricted to the cuDNN backend.
- Make it deterministic. Turn on `cudnn.deterministic`, turn off benchmark mode and TunableOp, pin
  every autotuner to one configuration, and bypass caches. Set `MKL_CBWR=COMPATIBLE` when CPU
  linear algebra feeds the sampler; otherwise the result follows the host CPU.
- Disable side effects such as event logs, watermarks and remote writes.
- Accept a gold only after three runs are byte-identical.

## 2. Record golds

Hook every boundary you will port. For each tensor, record its bytes, dtype, shape and SHA-256.
Also record scalar call arguments, RNG draws, and any compiled Triton or Inductor kernels together
with their launch grids.

Store each gold as `<gold id>/inputs/` plus one directory per run. `inputs/` holds exactly what the
server receives, decoded the way production decodes it and stored losslessly, plus `request.json`.

## 3. Climb the parity ladder

Each rung takes its inputs from the gold and must be bitwise before the next one starts.

| Rung | Scope |
| --- | --- |
| Operator | Each GEMM (derive the cuBLAS arguments as PyTorch does), each recorded compiled kernel, eager elementwise ops, attention, norms and reductions |
| Layer | One transformer block from gold inputs |
| Model forward | Embeddings, all blocks, head and unpatchify |
| Step | One sampling step: every guidance pass, caches and the scheduler update |
| Loop and components | The full sampling loop with RNG, then each encoder and decoder separately |
| Request | The whole orchestration, fed from the gold's front-end tensors |
| Front end | Image decoding and resampling, tokenizer, prompt cleaning |
| End to end | From the raw request files only |

Wan 2.1's rung results were: block-0 operators 74/74, schedule and steps 201/201, encoders and VAE
exact, front end 8/8, end to end 499/499 tensors on each of the ten golds.

## 4. Numerical rules

Each of these rules once caused a mismatch.

- Eager PyTorch rounds once per op. When you fuse ops, use `__fmul_rn`, `__fadd_rn` and `__fsub_rn`
  so the compiler cannot contract them into FMAs, and keep the Python expression's op order.
- Python scalars become float32 opmath. CUDA `x / s` is `x * (1.0f / float(s))`. BF16 with BF16
  rounds to BF16; mixed dtypes promote. Read the PyTorch kernel when unsure.
- For reductions, softmax and norms, copy PyTorch's kernels or port their reduction configuration.
- Port the device RNG exactly, including Philox offsets and launch grid. Advance it even for draws
  the reference later overwrites.
- Ops on CPU tensors need CPU-exact ports, including the instruction set the wheel was compiled for.
  Wan 2.1's bicubic resize follows the compiler's FMA contraction in PyTorch's AVX2/AVX-512 build,
  fitted on 400 random sizes. Its sampler scalars come from MKL VML and ship as tables.
- Large shapes bring tensors over 2^31 elements and convolution workspace limits. Gold them too.
- Compare scalar metadata as f32, because JSON float parsing can be off by one ULP.

## 5. Coverage

- Gold every preset. For a deployed model, also gold the production request shapes that together
  cover at least 90% of traffic.
- Enumerate request-parameter branches as well as shapes. Unported branches must fail with a clear
  error. Wan 2.1 rejects random seeds and any solver other than UniPC.
- Parallel variants (CFG or sequence parallel) must equal the single-GPU gold.
- After any native change, rerun representative golds of every path the change touches.
- Keep about ten validation cases that span the branches, and record their gold ids and results.

## 6. Connect the fast cold start

Follow the [supported startup path](architecture.md#the-supported-startup-path).
The model supplies declarations and numerical callbacks; the selected shared
loader owns scheduling, completion and cancellation. Do not add another startup
implementation to the model or endpoint. Extend the shared mechanism when a
capability is missing, and verify the existing adopters as well as the new model.
The required path is identical in standalone and selected-model unified serving.

Loading weights successfully is not sufficient startup acceptance. Declare all
required encoders, DiT and VAE, their byte bounds, and which execution preparation
depends on each component. For immutable serving archives, use
`HostArchive::load_cohort` and its owner-thread readiness callback rather than
one blocking loader per component. Declare GPU preparation in the copy plan
when the callback constructs execution objects; publish its receipt only after
those dependencies complete. Weight readiness and full request scratch are
separate contracts.

Select the shared loader that matches the admitted artifacts. A serving archive
cohort, an in-place checkpoint component load and a native process schedule may
use different APIs while honoring the same identity, resource and lifetime
contracts. The archive cohort is opt-in; full verification and paused background
loads intentionally preserve their existing ownership path. Record the loader
actually selected by each maintained CLI, service, ABI and GPU-count variant.
Do not treat one model's successful startup as every route's migration result.

Declare CPU-only work, weight-independent context work and weight-dependent
constructors separately. A native `gpu_prepare_complete` receipt only completes
the work declared by `gpu_prepare_required` and the copy plan; it does not imply
that all execution objects exist when no such preparation was required. Keep
host RAM residency, host registration, VRAM copy, execution preparation, first
DiT and complete outputs as separate boundaries.

Record actual read/upload bytes and first-DiT breakdown on the target hardware.
Check that a smaller checkpoint has not acquired a serial all-components
barrier or a slower per-archive read path. Validate full uploaded-payload SHA in
a separate run, and preserve cancellation, partial-prewarm takeover and memory
reservations. Do not count a model migration complete solely because its final
output is correct.

The [cold-start guide](fast-cold-start.md) describes the schedule and the loader policy. A model
connects to it in these steps.

Treat the following as entry-point acceptance, not merely an API checklist.
Complete numerical parity and startup integration are independent requirements.
Using one core helper does not certify the whole model or its other routes.

| Declaration or evidence | Required review |
| --- | --- |
| Artifact admission | A trusted metadata pin reaches the loader actually used by the maintained entry point; a missing pin cannot silently count as migrated serving |
| Components | Required encoders, DiT, VAE/decoders and other weights; actual source/runtime byte counts and first-use dependencies |
| Preparation | CPU work, weight-independent context work, per-component constructors and any remaining lazy setup; named readiness boundaries |
| Resources | Aggregate reader bounds, retained/pinned RAM, VRAM payload, prepared objects and execution scratch; no double counting shared leases |
| Copy receipt | Completed uploads and the preparation included in the admitted plan; no execution-ready claim from a weight-only receipt |
| Ownership | Error/cancel joins before context teardown; retained cache after eviction; bounded takeover if background prewarm is supported |
| Measurement | Original stream first-DiT marker, allocation/image boot boundary stated, full component scope retained, failures/tails included |
| Route coverage | CLI, resident, ABI, unified and GPU-count variants each identified as implemented, measured or accepted |

Use the shared loader that matches the admitted artifacts. Do not rewrite source
weights or build another model-private worker schedule just to copy a reference
adapter. An ordered stage is valid when its dependency or memory constraint is
explicit; an accidental all-components barrier is an integration gap.

**Prepare the checkpoint once.** The manifest lists every tensor as a byte range of the unchanged
published files. Preserve the source layout and dtype for the baseline; the loader can convert
runtime dtypes on the GPU. An offline BF16 copy is a separate, explicitly identified experiment
and must not silently change the startup comparison. Run the commands
from the package root; `convert` needs numpy.

```sh
python -m core_infra.runtime.checkpoint convert --root <checkpoint> \
    --file 'diffusion_pytorch_model-*.safetensors' --out <prepared>/dit-bf16 --copy config.json
python -m core_infra.runtime.checkpoint manifest --root <checkpoint> --out <prepared>/manifest.json \
    --file 'models_t5_umt5-xxl-enc-bf16.pth@t5.' --file 'Wan2.1_VAE.pth@vae.' \
    --extra-root prepared=<prepared> --extra-file 'prepared:dit-bf16/*.safetensors@dit.'
```

The `@prefix` names each component's tensors. Store any tables the reference computes at load
time in a small safetensors file, and add it with another `--extra-file`. Wan 2.1 stores its
float64 RoPE `freqs` this way.

**Declare runtime dtypes.** Map manifest names to the dtype each component runs them in. The
loader converts on the device when the stored dtype differs. From Wan 2.1, abridged:

```rust
pub fn prepared_requests(manifest: &Manifest) -> Vec<Request> {
    manifest.tensors.iter().filter_map(|(name, t)| {
        let dtype = if name.starts_with("clip.visual.") { Dtype::F16 }  // published F32, run in F16
            else if ["dit.", "t5.", "vae."].iter().any(|p| name.starts_with(p)) { t.dtype }
            else { return None };
        Some(Request { name: name.clone(), dtype })
    }).collect()
}
```

**Native loader mechanisms.** `model_load::start_components` admits independent
components with a SHA-pinned `checkpoint::Source` and one bounded reader pool.
Readers can start before CUDA. Bind their tensor groups through
`startup_native::NativeLoadPlan`, supplying `WeightStage` definitions with each
consumer's required preparations. Core adds reader/device resources, advances
the original ordered or independent uploads and validates the actual tensor
names and counts. Model code supplies constructors and shape work.
`Component.load_at_use` can defer a weight loader without deferring independent
CPU work. `LoadMode::IndependentAtUse` uses that edge for native ABI initialization;
`NativeLoads::warm_host` adopts existing readers under a retained-memory budget,
and the executor uploads those retained bytes when a consumer is admitted.

**Bind the native plan.** With numerical functions and validators already bound,
the model's execution flow uses the complete executor, as abridged below.
`NativeLoadPlan::bind` is unsafe because the model's context must outlive the
executor and every transferred allocation. Close the executor before context
teardown on success and failure.

```rust
let loads = unsafe { plan.bind(&mut bindings, &mut owner_bindings, open_device)? };
// Constructors consume loads.take(component) on the owner lane. Required
// preparations declare their weight/after edges and validate real objects.
loads.bind_context_work(&mut bindings, "run", "backend", backend_prepare)?;
let declaration = Declaration {
    components: loads.components(), preparations,
    workers: 8, reader_budget_mib: 1664, timeout_seconds: 900., consumers: None,
};
let mut startup = OwnerExecutor::new(declaration, bindings, owner_bindings)?;
startup.start()?;
startup.request(scope, request_facts)?;
let conditioning = startup.invoke(Some(scope), "encoders", condition)?;
let result = startup.invoke(Some(scope), "dit", || denoise(conditioning))?;
startup.close()?;
```

**Build components from device views.** Constructors take a weight source whose `buffer(name)`
returns a view into the loaded allocation, so building a component copies nothing.

**Declare backend preparation.** Supply CPU and GPU work plus validators to the
complete model plan. `Preparation::after` waits for validated preparation
results; CPU metadata can still run early. Native context callbacks name the
device resource in `weights`. Required initialization must be present and
validated before use. `RequestInput` holds typed CPU tensors for one request;
`OwnerExecutor::enter` supports nested consumers without borrowing the owner
through their whole computation.
Use shared operators where applicable. The shared exact dense SDPA operator automatically
loads a matching serialized plan when the serving configuration supplies a pinned plan bundle.
Record bundles offline for supported shapes and ship them with the runtime image; a cache miss
uses the original native plan builder. No generation is required before a request.

**Write a resident server.** Implement `core_infra::resident::Handler`. The first request runs the
cold schedule and keeps the models; later requests reuse them.

```rust
impl Handler for Server {
    fn handle(&mut self, args: &[String]) -> Result<Value, String> {
        match args.first().map(String::as_str) {
            Some("generate") if args.len() == 2 => self.generate(&args[1]).map_err(|e| e.to_string()),
            Some("health") => Ok(json!({"loaded": self.state.is_some()})),
            _ => Err(format!("unknown command {args:?}")),
        }
    }
}
// main: serve(Path::new(&socket), &mut server)
```

**Serve it.** Call `NativeService(program, work_dir, environment, audit=...).start()` in the
container's start hook. It spawns the server and does no GPU work, so the platform admits the
request at once. Pass the manifest and roots as `audit` to run the full SHA audit after the first
request.

The same audit declaration supplies the default OS file-cache prefetch. List the
model's used tensor prefixes once; the framework coalesces their file ranges and
owns metadata/helper scheduling and cleanup. Optional metadata failures cannot
gate the native owner. `startup_file_cache=False` disables it. This file-cache
policy does not require a private-memory budget or a model-specific prefetcher.

Pass `backend_plans={'path': manifest_path, 'sha256': digest}` to `NativeService` to admit the
bundle through the same framework interface for any model. The framework defaults include
bounded pinned readers and OS-backed zero-initialized reader storage on Linux; models do not
need H3-specific environment switches. `NATIVE_STARTUP_TRACE_DIR` enables shared startup timing.
When the serving container has sufficient admitted host RAM, set
`startup_read_ahead_gib=64` on `NativeService`. The framework admits all declared stages against
this one bound, starts their CPU reads early, and gives foreground work CPU priority. Models
do not implement a second loader or a custom prefetch thread; GPU dependencies stay unchanged.

**Measure and enforce completion.** Start the clock at the first executable user-module
instructions, before model/framework imports, enter hooks, preload and initialization.
Stop it at the original stream's first-DiT execution marker, after all original
required pre-DiT work. Function-body entry is a separate diagnostic clock. Do not
insert a device-wide sync. Stop iteration owners at that marker; run full correctness separately.
Use at least ten fresh untraced containers per configuration and retain all tails/failures.
Keep hardware, original source bytes/layout, dtypes, SDK, inputs and worker counts unchanged.
Use same-host alternating original/framework owners to check software overhead, with cache
conditions stated. Compare a new model's actual loaded bytes and read/upload critical path to
an existing port; account for required conditioning separately from system setup. Follow every completion
gate in [the cold-start guide](fast-cold-start.md#checkpoint-admission-and-migration-completion).

**Wire the normal runner.** Export and pin the checkpoint declaration alongside the profile;
make the maintained serving/CLI/ABI entry point consume it. A runner with missing admission
must fail migration validation instead of silently using a full-file fallback. Keep any
retained-RAM cache construction off first-DiT readiness, join it before first-request completion,
and validate real eviction/reload with exact outputs. LTX's `profile_export.py` is an example:
it upgrades admission metadata without rewriting source weights or inference-profile bytes.

## 7. Before review

Run the [CPU checks](build.md#cpu-checks) and the check-only Cargo build from
[CONTRIBUTING.md](../CONTRIBUTING.md). Then attach the GPU evidence: the gold ids, the rung
results, the end-to-end results on the validation set, and the GPU, SDK versions and source
identity they ran with. Keep weights, tensors and media out of Git.
