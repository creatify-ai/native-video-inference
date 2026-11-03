# Fast cold start for a ported model

A model declares weights, components, required CPU/GPU preparation, real
dependencies and result validators. The framework executes that complete plan:
independent reads and preparation start early, uploads and consumers wait only
for their prerequisites, and failed or missing work blocks execution. All
maintained entries must use the same model plan and scheduling policy.

H3's process-model entries use `runtime.declared_startup.DeclaredStartup` for
this model-internal execution policy. Native Wan and LTX use the shared
loader/preparation mechanisms with some model-controlled ordering.
The native mechanics below describe available building blocks; check each
model's architecture notes for the parts it uses.

## Existing native loader mechanics

1. **Container start (CPU only).** `core_infra.runtime.service.NativeService.start()` spawns the model's
   resident process (`<program> serve <socket>`, `core_infra::resident`). That process binds its socket
   and does no GPU work, so the platform hands the waiting request over immediately. In a
   multi-model endpoint the request names the model, and only that model is read.
2. **First request.**
   - `NativeService.start()` may already have started optional OS file-cache
     reads from the audit checkpoint declaration. Range metadata is prepared
     asynchronously, so it cannot gate the native owner. State this pre-request
     work in cold-start measurements. Set `startup_file_cache=False` to disable
     the helper.
   - Readers start (`core_infra::model_load`) before the CUDA context exists, so file reads overlap
     context creation.
   - `startup::Preparation` starts the CPU front end before context creation. Its backend hook
     waits for the caller's current CUDA context, then prepares the request's shapes while
     weights stream. A matching offline plan bundle avoids rebuilding cuDNN attention graphs.
3. **Independent component load** (`model_load::start_components`). Declare the text encoder,
   connectors, DiT, VAE and other required tensors once. A single bounded reader pool serves all
   components; separate GPU upload streams start at `device_ready`. Join a component only when
   its computation needs the weights. CPU/backend preparation runs during these reads.
   `runtime.components.ComponentStartup` supplies the corresponding process-owner schedule;
   dependent components do not occupy executor threads before their prerequisites are ready.
   **Ordered staged load** (`model_load::start_staged`) remains available when stages require
   that order. Stage 0 holds the components that run first, for
   example the text and image encoders and the VAE. It fills on the request's stream, and those
   components compute while the later stages, for example the DiT, stream on a background thread
   and stream. GPU stages retain this order. If the caller declares a read-ahead RAM budget
   and the complete schedule fits, CPU readers for all stages start before CUDA creation.
   The framework assigns those workers background CPU priority and owns their cancellation.
   With bounded deferred stages, the OS-cache helper remains active through
   earlier component uploads. Its stop signal belongs to the last nonempty
   deferred stage's first successful upload. If all readers already started
   through admitted read-ahead, retain the original first-upload handoff.
   An empty stage cannot claim handoff; request failure and service teardown
   still stop and join the helper. Models must not implement this policy with
   their own stop-file writes or a second cache worker.
4. **Later requests** reuse the resident models.
5. **Background audit.** After the first request completes, a full SHA-256 audit of the
   light-verified weights runs at low priority (`core_infra.runtime.audit`) and is paused during
   requests.

## What the framework provides

| Piece | Where | Role |
| --- | --- | --- |
| Checkpoint preparation | `core_infra/runtime/checkpoint.py` | `manifest`: every tensor of an unchanged safetensors or PyTorch zip `.pth` file as a byte range with its SHA-256 and first/last-KiB guard. `convert`: a runtime-dtype copy of tensors stored wider than the model runs them (F32 to BF16 with torch's rounding), as sharded safetensors that `from_pretrained` loaders also read. |
| In-place streaming load | `core_infra::checkpoint`, `core_infra::model_load` (feature `model-load`) | Bounded readers, pinned direct upload, one 256-byte-aligned allocation, on-device dtype conversion (`kernels/convert.cu`), light or full verification, staged loads. |
| Resident process | `core_infra::resident` | A JSON-line Unix-socket command loop that binds before any CUDA work. |
| Serving glue | `core_infra/runtime/service.py`, `audit.py` | Spawn, forward requests, and run the paused background audit. |
| OS file cache | `runtime/checkpoint_prefetch.py`, `files.py`, `src/model_load.rs` | Derive aligned ranges from declared model tensors, run the shared low-priority helper, and hand off at the declared load boundary. Deferred stages retain it until the last nonempty stage starts uploading. No retained tensor payload. |
| Complete process-model plan | `runtime/declared_startup.py` | Admission of components and required preparations, CPU/IO overlap, weight-dependent GPU work, request scopes and consumer gates over owned or resident transports. Used by H3. |
| Native preparation mechanism | `core_infra::startup` | CPU/context overlap, context-bound backend work and owned worker cleanup; a native model still needs a complete framework-owned plan around it. |
| Backend plans | `core_infra/kernels/sdpa.cpp`, `plan_cache.h`, `runtime/plans.py` | Shared exact SDPA with verified offline plans, complete layout/device/version keys and fallback to native builders. |
| Modal example | `examples/modal/wan21.py` | `@modal.enter` calls `NativeService.start()`; a method forwards requests. |

## What a model provides

[Porting a model](porting-a-model.md#6-connect-the-fast-cold-start) walks through these steps with code.

1. **A tensor list with runtime dtypes.** It maps manifest names to the dtype each component runs
   them in, for example Wan's `prepared_requests`: DiT BF16, T5 BF16, CLIP visual F16 from the
   published F32, VAE F32.
2. **Real dependencies.** Which weights and preparation each component needs. Declare ordering only where the computation or resource lifetime requires it.
3. **Constructors from device weights.** Components are built from views of the loaded allocation.
   Wan's loaders take any `weights::Source` whose `buffer()` returns a device view.
4. **Required preparation.** Declare CPU work, context-dependent and weight-dependent initialization, consumer dependencies and receipt validators. Genuine optional work stays explicit; required initialization cannot be disabled.
5. **Entry adapters.** CLI, ABI and services submit the same model plan and retain their original transports and resources. The native Wan handler is an existing adapter awaiting complete-plan consolidation.
6. **Preparation inputs.** Which published files to read in place, which to convert, and any
   derived tables. Wan stores only the RoPE `freqs`, which WanModel computes on the CPU in float64.

Also declare which preparation is CPU-only, which requires a live context, and
which requires a component's uploaded weights. Publish each completion boundary
honestly. `warm` denotes host residency; `prefetch` denotes its stated copy and
preparation scope. Neither term alone certifies complete request readiness.

## Loader policy (`NATIVE_LOAD_*`)

| Variable | Default | Meaning |
| --- | --- | --- |
| `NATIVE_LOAD_WORKERS` | 24 | Reader threads |
| `NATIVE_LOAD_CHUNK_MIB` | 32 | Read/upload chunk |
| `NATIVE_LOAD_PINNED_SLOTS` | 4 | Upload slots |
| `NATIVE_LOAD_BUDGET_MIB` | exact need | Host buffer bound |
| `NATIVE_LOAD_DIRECT` | 1 | Register reader buffers with CUDA; no pinned staging copy |
| `NATIVE_LOAD_FILE_IO` | `auto` for the declared checkpoint API; `buffered` for low-level adapters | Inspect packed-file cache residency without faulting payloads. Cached ranges use buffered reads; missing ranges use aligned direct-file reads, with a compatibility fallback. Logical chunks retain the configured size; bounded alignment headroom is separate. Raw-file and retained one-use readers retain their existing path. `buffered` and `direct` are diagnostic overrides. This controls file IO, separately from `NATIVE_LOAD_DIRECT` (H2D). |
| `NATIVE_LOAD_VERIFY` | `full` (library); `light` under `NativeService` | `full` hashes every byte before use. `light` checks the first and last KiB and leaves the full audit to the background. |
| `NATIVE_CORE_READ_LAZY_ZEROED` | `1` on Linux | OS-backed initialized zero pages avoid serial buffer clearing before readers start. `0` restores eager allocation. |
| `NATIVE_LOAD_AHEAD_GIB` | `0` | One retained-read RAM budget for all declared startup stages. If the schedule does not fit, use bounded staged reads. `NativeService(startup_read_ahead_gib=64)` also selects a fixed pinned-copy ring unless the caller explicitly overrides `NATIVE_LOAD_DIRECT`. |

The optional `NativeService(..., backend_plans=pin)` configuration verifies and atomically
publishes a SHA-pinned plan bundle. Keep this small offline artifact in the runtime image to
avoid a remote-volume fetch on the request path. Cache misses and incompatible GPU/SDK versions
use the original builder. Set `NATIVE_STARTUP_TRACE_DIR` to trace CPU preparation, allocations,
readers, upload waits and the true completion of background loads without adding GPU barriers.

## Reading weights from a Modal Volume

Profiles of fresh B200 containers in `us-west` (IO traces with `NATIVE_STARTUP_TRACE_LEVEL=io`)
show what limits a cold start once the startup schedule is right:

- **Direct reads.** Buffered reads stop near 5 GB/s per process, whatever the reader or file count.
  `NATIVE_LOAD_FILE_IO=direct` reads the same files at 7–10 GB/s from a few large files and
  14–19 GB/s from a few hundred. An OS-cache helper (`startup_file_cache`) only adds buffered
  reads there, so the Wan example turns it off and forces direct reads.
- **Many files.** Each file streams at a limited rate, so a checkpoint split into many files reads
  faster than one held in a few large ones. Wan's preparation writes every weight it serves in
  64 MiB safetensors shards (the tensors are byte-identical). One tensor is still read by one
  reader, so a single very large tensor (T5's 2.1 GB embedding) sets the floor of its stage.
  A model that admits its published files can keep doing so and read from a storage layout:
  `core_infra.runtime.checkpoint.layout` copies a manifest's tensors into 64 MiB files, and
  `NATIVE_WEIGHT_LAYOUT_{MANIFEST,SHA256,ROOTS}` makes the readers take each tensor from there
  after checking that its name, bytes, dtype, shape, SHA-256 and light guard are unchanged. The
  LTX example reads its 70.7 GB this way, with `NATIVE_LOAD_WORKERS=48`.
- **Warm Volumes.** Modal caches Volume data on the hosts that read it. A new Volume, or newly
  written files, start cold, and a read from a CPU container does not warm the GPU hosts: expect
  two or three rounds of 30 slow cold starts before most hosts serve it at full speed (the README
  lists measured rounds).
- **Recorded backend plans.** cuDNN attention plans take seconds to build. A recorded bundle
  (`NativeService(backend_plans=...)`, or `NATIVE_CUDNN_PLAN_CACHE_MODE=read` with a verified
  `NATIVE_CUDNN_PLAN_CACHE_DIR`) loads them at once. Plans are keyed by their exact shapes, so a
  bundle only helps requests of the shapes it was recorded for. The Wan example adds a request
  shape's plans to its bundle the first time it sees that shape (image size, area and frame count);
  a missing shape costs about 2 s of plan building on every cold start. The LTX example records one
  bundle per profile during its first generation, which cuts its text encoder's plan setup from
  2.2 s to under 1 s.
- **Overlap the computation with the load.** Order weight stages so that the encoder that runs
  longest starts first. Wan loads the VAE (0.5 GB) first and encodes the conditioning video while
  CLIP, T5 and the DiT stream in; model setup runs on the owner thread, so a stage's tensors can be
  ready before its model object is.
- **Several GPUs.** With `--gpus 2`, H3 splits its DiT across two B200s by context parallelism.
  Each rank keeps half of every layer and reads only that half, as byte ranges of the one-GPU
  weight files (`Job::slice`, from a slice index written once at preparation). The container reads
  the same bytes as one GPU, from files that are just as warm. Each rank starts its readers before
  CUDA, and forms its NCCL communicator (about 4 s) on a helper thread during the reads and upload
  instead of before them.

  Wan's second GPU runs the unconditional pass of classifier-free guidance and needs the whole
  DiT, but reads none of it. Its CUDA context opens on its own thread at startup and builds the
  request's attention plans during the reads. When the first GPU's DiT stage is on the device, the
  second copies the 32.8 GB over the peer link (`Loaded::replicate`, about 40 ms) and builds its DiT
  while the first builds its own, so it adds about 50 ms before the first step.

  LTX's second GPU runs DiT blocks 24-47 and copies them from the first GPU's resident DiT (18.6 GB
  in about 45 ms). Keep the second GPU's own startup work off the reads: when it parsed the 16 MB
  profile manifest again during the reads, the parse took 7 s instead of 1 s and slowed the reads
  by 2.5 s. It now takes the first GPU's parsed copy (`gold::Shared`), and even its remaining small
  file checks slowed the reads by about 1 s, so it does them (about 40 ms) only after the reads,
  right before its copy. Its CUDA context also opens only once the first GPU's is ready: created
  together, both took longer and the first GPU's uploads started about 0.2 s later.

## Checkpoint admission and migration completion

Publish a `native-checkpoint-v1` manifest once, while the original files are available for a
complete offline SHA audit. Carry `{manifest, sha256, roots}` into the actual model entry point.
`checkpoint::Source::load` validates this declaration for Rust/ABI callers;
`runtime.service.checkpoint_environment` forwards the identical contract to a native process.
`Options::for_declaration` selects light serving guards unless the caller explicitly requests
full foreground verification. The service remains responsible for the full post-request audit.
Do not silently omit the declaration and fall back to whole-file loading in a migrated runner.

A background audit specification is not automatically a trusted serving
declaration. `NativeService` currently forwards it only when it contains a
`sha256`; its compatibility constructor also accepts an unpinned audit spec.
Validate the maintained caller's inputs, and confirm that the native loader
actually consumes the pin. Model-specific adapters must bridge their own pin
fields explicitly; setting `NATIVE_CHECKPOINT_*` alone does not configure a
loader that still reads a different model-specific variable.

For a serving archive adapter, use its admitted archive/guard identities and
shared owner callbacks rather than rewriting checkpoints just to select another
loader. Verify which policy it runs: foreground cold cohort, full verification,
retained-RAM reload, paused prewarm takeover and host-only warm are distinct
paths. The file-I/O defaults above describe this declared-checkpoint API, not
every archive or raw-file adapter in the repository.

The LTX profile exporter publishes this metadata without changing checkpoint bytes or
exported inference-profile bytes. Its ordinary runner verifies the profile pin and forwards only
the checkpoint declaration. The native `infer` entry point uses the same component loader as
resident requests. ABI initialization accepts `checkpoint`; it defers model GPU uploads until
the admitted prefetch operation. An explicit host-only `warm` is a different operation: it
materializes RAM residency and must not be confused with the GPU cold-start measurement.

GPU readiness must not wait for a second full retained-RAM read. LTX prepares its later RAM
cache during sampling and joins/pins it before the first request returns. Eviction/reload then
uses those already-pinned leases. Physical host-memory receipts include the shared retained
registrations exactly once, separately from logical tensor bytes.

Place the timing origin before any startup work claimed by the measurement.
In a real service, the platform's enter hook can precede method/function entry.
Report enter-hook work, including optional read-ahead and plan publication,
separately if it is outside the clock. A first-request timer after that work
cannot establish the total cold-start latency. Keep allocation and image boot
as their own boundary as well.

A migration is complete only with all of the following evidence:

- The maintained entry point consumes the declaration and uses shared reader, preload, CPU
  preparation and backend-init ownership. A benchmark-only replacement runtime is insufficient.
- The maintained entry points share one model declaration. Physical owner layouts may
  differ, but adapters cannot duplicate the preload dependency or validation policy.
  Missing required preparation and failed or invalid readiness results must stop
  execution before the consumer; diagnostic bypasses cannot become serving defaults.
- Exact outputs match the admitted reference: per-model latents first, then the composed request;
  include decoded tensor/audio checks where those outputs are part of the supported route.
- First-DiT timing uses the original stream execution marker. Record all original pre-DiT weight
  bytes and components; stop iteration owners at that marker. Do not add a device-wide sync.
- Keep GPU/CPU/RAM, original files, dtypes, SDK, inputs and worker counts fixed. Report at least
  ten fresh untraced containers per case, including failures and tails. Same-host fresh-process
  tests diagnose software overhead but do not replace cold-container measurements.
- For a newly ported model, measure its actual source bytes, the read/upload completion boundary and time from that
  boundary to first DiT. Show whether CPU/pre-init is covered by IO; parameter-count labels do
  not establish an expected latency or physical bandwidth limit.
- Verify already-pinned RAM-to-VRAM copying separately, and verify exact outputs again after
  eviction/reload. Keep file read, host registration and runtime dtype conversion out of the
  claimed pure-copy boundary.
- Enumerate CLI, resident, ABI, unified and device-count variants separately.
  Trace which entry consumes metadata, starts readers, releases uploads and
  joins preparation. A fast ordinary runner does not certify an ABI that first
  joins every upload and only then starts conditioning.
- Use a native stream marker without a timing-only device-wide sync. A successful
  first-DiT cutoff does not validate final latents/pixels or full-request latency.
  Audit uploaded weights and numerical outputs in separate complete checks.
- If a route supports multi-model prewarm, test an early foreground takeover
  before the target finishes. Preserve partial reads and do not wait for full
  target readiness. Test read permission separately from host-registration
  permission. Do not assume that all single-model adapters implement this policy.

## Historical comparison: Wan 2.1 I2V-14B-720P on one B200

This comparison predates the shared component loader and current startup defaults.
It remains evidence for those configurations, not a current cross-framework ranking.
Repeat a matched cohort before applying its differences to a newer build.

Conditions shared by every row:
- the same storage volume and the same BF16 bytes (DiT 32.8 GB);
- the same container shape (16 vCPU);
- 10 cold containers per row;
- the clock starts at the container's function entry and stops at the start of the request's
  first DiT forward (allocation and image pull are outside it).

| Runtime | p50 | p90 | min |
| --- | --- | --- | --- |
| rust-inference (staged, light verify) | 8.9 s | 18.0 s | 7.8 s |
| rust-inference (staged, full verify) | 11.3 s | 14.6 s | 8.6 s |
| vLLM-Omni 0.30.0 (official image, defaults) | 41.9 s | 73.2 s | 40.2 s |
| SGLang 0.5.21 (official image, defaults) | 75.9 s | 84.3 s | 44.7 s |

Per-container read throughput on these volumes is 1.5 to 7.7 GB/s and dominates the variance.
Full-SHA verification in the foreground costs about 2 s at p50, which is why serving defaults to
light verification plus the background audit.

The two framework rows ran their offline Python APIs (`Omni(...)`, `DiffGenerator`) in a fresh
process, which is their lightest startup path. Launching their HTTP servers (`vllm serve`,
`sglang serve`) adds server bootstrap before model loading.
vLLM-Omni's startup always includes its warmup (a 512x512, 2-step dummy generation).
