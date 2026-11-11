# FAQ for PyTorch, SGLang and vLLM engineers

[中文](faq.zh-CN.md)

Short answers to the questions engineers ask when they come from PyTorch, Diffusers, SGLang or
vLLM-Omni. [Design decisions](#design-decisions) explains why the framework is built the way it is,
and [Speed](#speed) compares it with SGLang and vLLM-Omni. Each answer links to the document with the
details. Numbers were measured on NVIDIA B200 on Modal unless stated otherwise.

- [What it is](#what-it-is)
- [Design decisions](#design-decisions)
- [Speed](#speed)
- [License](#license)
- [Your vocabulary, mapped](#your-vocabulary-mapped)

## What it is

### What is native-video-inference?

Ports of three video generation pipelines to Rust and CUDA (MiniMax H3, Wan 2.1 I2V-14B-720P and
LTX 2.5 distilled), plus a model-independent core, [`core_infra/`](../core_infra/README.md), for
checkpoint loading, startup scheduling and serving. Wan and LTX reproduce their official PyTorch
pipelines byte for byte on a B200; H3's numerics follow SGLang's H3 implementation at a pinned
commit. The core is built for cold start: about 6–9 s at P50 from container start to the first DiT
step, while reading 47–96 GB of weights.

It is not a general engine. There is no model registry that loads any Hugging Face repository, no
request scheduler or batching, and no HTTP API. A new model is a new port; see
[porting a model](porting-a-model.md).

### How does it relate to SGLang and vLLM-Omni?

They are general serving engines: many models and GPUs, batching, HTTP APIs, adapters,
quantization and larger parallel degrees. This repository makes a different trade: a few models,
ported exactly, with startup built around the storage bandwidth
([design decisions](#design-decisions)).

It builds on their work. H3's numerics follow SGLang's pinned H3 implementation, its sparse-attention
router kernels are SGLang's, unmodified, and its attention kernel is FlashInfer's
([NOTICE](../licenses/NOTICE.md)). The README and the [Speed](#speed) answers measure cold start
and warm requests against SGLang and vLLM-Omni because they are the reference points for serving
these models. Much of the cold-start work could live in them too: we have tried it on a branch of
SGLang Diffusion and proposed it upstream
([can SGLang start this fast?](#can-sglang-or-vllm-omni-start-this-fast)).

Use them when you need a model, GPU, feature or parallel degree that is not ported here. Use this
when your models are ported here and cold start or the reference's exact output matters.

## Design decisions

Each answer says what we chose, why, and what it costs.

### Why optimize cold start above all?

A video request needs a whole GPU for tens of seconds to minutes. When peak traffic arrives, we add
GPUs on demand, and the faster a new container serves its first request, the less users wait and the
less we pay for GPUs that are starting instead of generating. After the peak, the GPUs go back.
Capacity is held only while it is used: users wait less, we pay less, and GPUs are not held idle.

The design starts from the hardware's physical limits and owns every layer down to them. We measured
cold-start reads on Modal at a steady 10–20 GB/s, so reading 47–96 GB of weights should take about
5–10 s. Every other piece of startup work (CPU preparation, the CUDA context, backend plans,
conditioning) has to fit inside that window instead of adding to it. From the first DiT step on, a
request is tens of seconds of GPU compute, and owning every operator means that when something is
slow, we can go straight to the layer where the bottleneck is instead of working around a framework.

### Why native code instead of PyTorch, and why Rust?

Native code gives the end-to-end control described above:

- **Startup.** A native process starts without imports, compilation or a warm-up request, binds its
  socket before any CUDA work, and starts reading weights before the CUDA context exists. Every
  startup step is declared to one scheduler that overlaps reads, CPU work, context creation and
  backend plans ([architecture](architecture.md#the-supported-startup-path)).
- **Numerics.** Every operation is a known kernel with known arguments, so exact equality with the
  reference is reachable, and stays checkable after each optimization.
- **Ownership.** Threads, CUDA contexts, pinned buffers and child processes are owned and joined
  explicitly, including on cancellation and error.

PyTorch's arithmetic is kept: ATen's kernels, cuBLAS arguments and random generators are ported, not
replaced ([NOTICE](../licenses/NOTICE.md)). Rust is memory-safe with C-level performance, and we had
heard that much of the ML infrastructure at large companies is written in it. We started with Rust
as an experiment; the ports passed acceptance and it has been good to work in. The team had
contributed to a pure-C ML framework, [libnnc](https://libnnc.org/), before, and end-to-end
ownership matters more than the language itself.

### Why a separate port for each model instead of a general engine?

Mostly because of what we serve. Our product needs cold starts in seconds and the reference's exact
output, which existing frameworks did not provide, and our traffic is concentrated on three models:
Wan, H3 and LTX. A general engine would carry the cost of breadth we do not use.

Exactness also favors separate ports: it needs each reference's own kernels, op order, random
generators and CPU preprocessing, and a shared abstraction would impose its own numerics. Each model
therefore copies the operators it needs, so a change made for one model cannot alter another
model's accepted numerics. Core keeps only model-independent mechanisms, with no model names or
shapes ([CONTRIBUTING](../CONTRIBUTING.md)).

The cost is coverage: each model is a full port (Wan's matched 499 tensors on each of ten requests),
and adding one takes time. General engines choose the other side of this trade.

### Why exact equality instead of a tolerance?

With no tolerance, acceptance is a hash comparison, and a mismatch is a bug in one rung of the
[parity ladder](porting-a-model.md#3-climb-the-parity-ladder), not a judgment call. Every speed
change is checked the same way, so an optimization cannot quietly change the output. Parallel
variants must equal the single-GPU gold too, which is why one and two GPUs give identical bytes.

The reference is the deployed or official implementation
([freeze the reference](porting-a-model.md#1-freeze-the-reference)). The cost is that approximate
optimizations, such as quantization or a faster attention kernel with different rounding, are
outside the acceptance rule, and exactness is tied to one GPU and SDK set.

### Why no warm-up request?

A general engine serves many models and shapes and compiles or tunes kernels at run time, so a
warm-up covers that work: SGLang's server runs three warm-up requests (about 96 s), and vLLM-Omni
compiles the DiT blocks with `torch.compile` and runs one. Here the models and shapes are few, so the
same work is recorded offline. cuDNN attention plans, keyed by exact shape, load from a recorded
bundle instead of taking seconds to build. The first request then runs at warm speed: on H3, the
denoising stage of the first request in a fresh process took 151.62 s against 151.58 s for a warm
one ([measurement](#how-do-warm-request-times-compare-with-sglang-and-vllm-omni)). The cost is a
bundle per shape: a shape that was not recorded costs Wan about 2 s of plan building on every cold
start.

### Why one request per container and no batching?

Batching pays off in LLM decoding, where each step is small and memory-bound and several requests
share each weight read. A video DiT step is closer to an LLM prefill: Wan's official example has
about 75,000 tokens, and a block takes about 120 ms, so the GPU is already compute-bound and batching
adds little. Throughput scales with containers. vLLM-Omni has explored step-level
[continuous batching for diffusion](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/diffusion_continuous_batching/);
its documentation marks it experimental and recommends one request at a time for the conservative
path.

### Why at most two GPUs per request?

Peak traffic is served by on-demand GPUs. One- and two-GPU containers start on demand, but an
eight-GPU node generally has to be reserved, and waiting for one on demand can take around half an
hour, so shortening its cold start would not help during a peak.

Within two GPUs, each model uses the split that keeps the output identical to one GPU: CFG parallel
for Wan, context parallel for H3, sequence parallel for LTX ([README](../README.md#cold-start)).
Storage is the slowest part of a cold start, so the second GPU copies weights from the first over
the peer link (Wan's 32.8 GB DiT in about 40 ms) instead of reading them again; H3's ranks read
disjoint halves of the same files.

### Why no CUDA graphs?

Our focus is cold start. For a warm request the bar is to match the general engines, not to beat
them, and graphs did not earn their cost here:

- **Wan and H3** have long forwards. A Wan forward takes about 4.8 s (387 s for 80 forwards), about
  120 ms per block, so the CPU issues launches much faster than the GPU runs them. In our
  experiments neither ran slower without graphs.
- **LTX** runs 11 short forwards (2,093 video tokens in stage 1, 8,372 in stage 2), so graphs did
  help it a little. But a graph replays fixed shapes, so every shape needs its own: each sampling
  stage of each profile, and on two GPUs each GPU's share of the tokens. Run eagerly, LTX's time to
  its final latents is still comparable to SGLang's
  ([comparison](#how-do-warm-request-times-compare-with-sglang-and-vllm-omni)), so we removed the
  graphs instead of keeping one per shape.

How LTX cuts its host work instead, and its times with and without graphs, are in
[LTX's README](../models/ltx25/README.md#performance).

## Speed

### How do cold starts compare with SGLang and vLLM-Omni?

The clock runs from the first line of user code in a fresh container, before any import, to the
start of the request's first DiT forward, taken by a callback on the DiT's own CUDA stream without a
device-wide synchronization. Modal's GPU allocation and image boot are outside it; every weight read
and upload and all conditioning before the first DiT step are inside it
([architecture](architecture.md#readiness-has-several-meanings)).

The README's [cold-start table](../README.md#cold-start) has this repository's own cold start for
each model on one and two GPUs, measured on 9 and 10 November 2026: every P50 is between 6.0 and
8.7 s, LTX's 6.6 s on one B200 and 6.5 s on two. H3's rows pool two rounds of 30 containers, and
Wan's and LTX's are one round each, so their P90s include hosts that had not yet cached the Volume.
The P50 of one round moves by 1–2 s from round to round. Two earlier studies compared it with SGLang
and vLLM-Omni, with different settings.

**Wan 2.1**, 1 November 2026 ([README](../README.md#compared-with-sglang-and-vllm-omni)): each
framework ran with its documented server command and defaults on the same request, with the same
clock, in interleaved rounds of fresh containers. For SGLang and vLLM-Omni, a forward pre-hook on
`WanTransformer3DModel` marks the first DiT step and ignores their warm-up forwards. Measured that
way, the P50 was 12.1 s here, 140 s for vLLM-Omni 0.30.0 and 188 s for SGLang 0.5.22. Most of their
time is server start with warm-up ([why there is none here](#why-no-warm-up-request)), and both read
the Diffusers checkpoint, which stores the DiT and text encoder in FP32 (90 GB), while this port
reads a BF16 layout prepared once from the official checkpoint (47 GB). Their offline Python APIs
start faster than their servers: an earlier measurement, with older versions and setup, put
vLLM-Omni at 41.9 s and SGLang 0.5.21 at 75.9 s P50, against 8.9 s here
([historical comparison](fast-cold-start.md#historical-comparison-wan-21-i2v-14b-720p-on-one-b200)).

**MiniMax H3**, 21 October 2026: ten configurations, each in 50 cold containers in each of three
Modal regions (1,500 in all), on FL2VA at 768×768, 124 frames and a 28-step recipe on one B200. The
clock runs from the first instruction of the cloud handler to the entry of the first DiT forward.
Warm-up and `torch.compile` were off in every SGLang and vLLM-Omni configuration, so their time is
imports, engine start, loading and the first request's preparation:

| Configuration | P50 | P90 |
| --- | ---: | ---: |
| native-video-inference | 7.21 s | 19.88 s |
| SGLang `77ba54f`, short startup with direct GPU loading (its fastest) | 64.70 s | 119.31 s |
| SGLang, recommended configuration | 65.40 s | 121.16 s |
| vLLM-Omni `83afc05`, offline API at defaults (its fastest) | 71.50 s | 126.03 s |
| vLLM-Omni, recommended HTTP server, 4 loader threads | 73.98 s | 124.11 s |

All nine SGLang and vLLM-Omni configurations had medians between 64.70 and 75.00 s. Host CPUs were
not pinned (ap-south placed every container on slower hosts, for every engine), caches were not
cleared, and the native arm ran a build that predates this repository. The
[cold-start post](blog-cold-start.md#appendix-a-how-we-measured) has the full contract, all ten
configurations and where the time goes in each engine.

The two studies give SGLang very different numbers (188 s against 65 s) mainly because Wan's ran
the server's defaults, warm-up included, while H3's turned warm-up off. In both, the difference
reflects the [design decisions](#design-decisions) more than kernel speed. In the H3 study, the
Python engines spent 20 to 31 s importing and initializing before they read a weight and then loaded
components one after another, while a worker here starts reading within half a second, loads in
parallel and reads less (about 95 GB, where the checkpoint's DiT and text encoder are 133 GB).

### Can SGLang or vLLM-Omni start this fast?

Much of it does not depend on Rust. We tried the ideas behind this repository's cold start on a
branch of SGLang Diffusion's main, serving H3 on one B200: a worker that starts before the launcher
finishes resolving arguments, helper processes that read the checkpoint while Python imports,
direct GPU loading, and bytecode and kernel caches built into the image. On 5 November 2026, with
50 containers per arm in the same round, a fresh server reached its first DiT step at a median of
20.1 s instead of 79.0 s on fast hosts, and 31.9 s instead of 152.9 s and 42.8 s instead of 161.6 s
on two slower kinds of host. Full 28-step runs produce identical frames, and a step still takes
1.29 s. That round used a newer main and ran separately from the study above, so the two are not
comparable.

Two things are harder in one Python process: loading components in parallel, because of the GIL and
process-wide loading state, and preparing the first request while weights load, because the server
takes requests only once it is ready. The [proposal](sglang-cold-start-proposal.md) splits the
branch into 13 small pull requests and a cookbook page, with the new startup features off by default
and no change to the numerics, and lists the questions we would like SGLang's maintainers to answer
first. It is a proposal, not a finished patch set; we would be glad to contribute it the way the
SGLang team wants, and to help the vLLM-Omni team the same way.

### Will I see the same cold start outside Modal?

Nothing in core depends on Modal. Build on Linux x86-64 with a B200 as in [build.md](build.md), put
the pinned cuBLAS and cuDNN libraries on the loader path, and run `NativeService` from any Python 3.12
process; weight roots are plain directories, so local NVMe, an image layer or a network filesystem
all work. The mechanisms are not specific to Modal, but the numbers depend on storage: a cold start
in seconds needs reads at about 10 GB/s. The read-path findings, such as buffered reads stopping near
5 GB/s and many small files reading faster than a few large ones, were measured on Modal Volumes
([reading weights from a Modal Volume](fast-cold-start.md#reading-weights-from-a-modal-volume)).
On other storage, trace a cold start with `NATIVE_STARTUP_TRACE_DIR` (and
`NATIVE_STARTUP_TRACE_LEVEL=io`) and tune the `NATIVE_LOAD_*` reader settings in the same guide.

The platform's own start matters as much. Of four B200 platforms measured on 19 and 20 October
2026, Modal and fal started a new container in seconds; Together AI copied the weight volume to each
new node first, and Nebius booted a virtual machine per job, so there the platform alone took tens
of seconds to minutes ([the platform underneath](blog-cold-start.md#appendix-b-the-platform-underneath)).

### How do warm request times compare with SGLang and vLLM-Omni?

These comparisons are not a speed contest. Our focus is cold start; for warm requests, the numbers
show that inference here is on the same level as SGLang and vLLM-Omni, and as usable.

A request's time moves by about 2% between B200 hosts, so every comparison below runs the engines
in one container on the same GPU.

**MiniMax H3.** FL2VA at 1376×768, 124 frames and 49 DiT forwards, seed 42, measured on
22 October 2026. Ten B200 containers each ran every engine in balanced order: two warm-up requests,
then one measured request, timed from input processing to the saved MP4. Attention matched across
engines, and CUDA graphs, `torch.compile` and step caches were off in all of them. Means over the
ten warm requests, and over 30 fresh processes for the last column:

| Attention | Engine | Warm request | Warm denoising | Denoising in a fresh process |
| --- | --- | ---: | ---: | ---: |
| cuDNN SDPA | native-video-inference | 154.68 s | 151.58 s | 151.62 s |
| cuDNN SDPA | SGLang `ad7a9b5` | 156.98 s | 153.78 s | 154.98 s |
| cuDNN SDPA | vLLM-Omni `7cbc7e4` (vLLM 0.31.0) | 155.99 s | 152.63 s | 153.25 s |
| cuDNN SDPA | PyTorch eager, Diffusers `48cdbae` | 200.87 s | 193.83 s | 194.22 s |
| SubBlock | native-video-inference | 115.03 s | 111.94 s | 112.07 s |
| SubBlock | SGLang `ad7a9b5` | 117.73 s | 114.54 s | 118.32 s |

Paired by GPU, a warm request took 1.47% less time here than with SGLang (95% interval 1.36–1.55%),
0.84% less than with vLLM-Omni (0.72–0.94%) and 22.99% less than PyTorch eager with cuDNN attention,
and 2.30% less than SGLang with SubBlock: less in each of the ten pairs. SGLang and vLLM-Omni are
within 1.5% here; the large gap is to the Diffusers pipeline run eagerly. The native
and SGLang sampler outputs were equal bit for bit in every request. vLLM-Omni's and PyTorch eager's
were not, so those two are latency comparisons only. vLLM-Omni had no ready SubBlock-style sparse
configuration for the unmodified model at the pinned version, so it ran cuDNN attention only. This
is one FL2VA request shape on one GPU, without Ref2VA, measured on the 22 October build of this H3
port, which predates this repository; the code here has not been measured again since.

**Wan 2.1**, the official example, alternating with SGLang 0.5.22
([README](../README.md#compared-with-sglang-and-vllm-omni)):

| Request | native-video-inference | SGLang 0.5.22 |
| --- | ---: | ---: |
| One B200: whole request | 390.3–390.4 s | 398.2 s |
| One B200: the 40 steps | 387.1–387.2 s | 391.3–391.4 s |
| Two B200s: the 40 steps | 200.97–200.99 s | 200.91–201.02 s |

The two are close. SGLang also runs the unconditional CFG pass on the second GPU, and on two B200s
the 40 steps take the same time. On one GPU, the native steps are 1% faster from sharing block 0's
work between the CFG passes, and the VAE decode takes 3.2 s against 5.4–5.8 s.

**LTX 2.5**, T2V at 1472×832, against SGLang 0.5.22 run unmodified, alternating in one B200:2
container on 9 November 2026 with the same prompt, seed and size
([performance](../models/ltx25/README.md#performance)). Each warm request is timed from the request
to its final latents: for SGLang, the sum of its own logged stage times before decoding; here,
`ltx25 infer` with `LTX25_REPEAT=3`. Decoding is not counted on either side:

| Request | native-video-inference | SGLang 0.5.22 |
| --- | ---: | ---: |
| One B200, 49 frames | 1.92–2.22 s | 2.57–2.61 s |
| One B200, 57 frames | 2.19–2.23 s | 2.65–2.83 s |

There is no two-GPU comparison. On two GPUs SGLang rounds the latent frames up to a multiple of the
GPU count, so 49 frames become 57, and its two-GPU time in that container, 3.8–3.9 s, was slower
than its own one-GPU time, which suggests our two-GPU SGLang setup was not right. Natively, two
B200s took 1.6–2.2 s for 57 frames.

SGLang's output is not the official pipeline's: it samples with deterministic Euler instead of
Euler ancestral and draws different noise, and by default it decodes the latents once with the VAE
instead of running LTX 2.5's diffusion video decoder. That decoder needs NATTEN, which has no wheel
for SGLang's torch 2.14; without it, SGLang falls back to `flex_attention` and a request takes about
110 s.

## License

### Can I use it commercially?

Yes, with one condition. Core, our contributions to Wan 2.1 and the tooling are source-available
under the [community license](../LICENSE), which permits commercial and non-commercial use. If your
commercial products and services generate more than USD 10M in yearly revenue, you must obtain
separate, prior written authorization from Creatify Lab, Inc. before using the software, whatever
the purpose of the use ([Section 2.2](../LICENSE)). Contact support@creatify.ai; sending an email
does not by itself grant authorization. It is not an OSI-approved open-source license.
`models/minimax_h3_official` is under the MiniMax H3 Community License and `models/ltx25` under the
LTX-2.x Community License. Running a model also needs your own license for its weights
([README](../README.md#license)).

## Your vocabulary, mapped

| If you know | Here |
| --- | --- |
| `vllm serve`, `sglang serve` | A model's resident program, such as `wan21-server serve <socket>`, owned by `core_infra.runtime.service.NativeService`. It takes JSON-line commands over a Unix socket; there is no HTTP server. |
| `DiffusionPipeline.__call__` | An explicit operation order in Rust: `pipeline.rs` (Wan), `request.rs` (LTX), `pipeline/` (H3). |
| A model's `load_weights()` | A component declaration: tensor names and runtime dtypes, read in place from a SHA-pinned checkpoint manifest by core's bounded readers. |
| `torch.compile`, Inductor | No compilation at run time. Wan, and H3's text encoder and DiT, are written op by op. LTX and H3's VAEs replay programs recorded once from the reference: ATen dataflow graphs, or Inductor launch sequences with their Triton CUBINs. |
| Warm-up requests, CUDA graph capture | No dummy request and no CUDA graphs. cuDNN plans load from a recorded bundle, and every forward is launched eagerly ([why](#why-no-cuda-graphs)). |
| Attention backend flag | Fixed per model: the cuDNN SDPA engine the reference uses (Wan, LTX), FlashInfer's SM100 block-sparse kernel (H3's default), NATTEN (LTX's video decoder). |
| `--tp`, Ulysses and ring degrees, CFG parallel | `--gpus 2`, with one strategy per model. See [why two](#why-at-most-two-gpus-per-request). |
| `atol`, `rtol` | None. Every recorded tensor must have the reference's SHA-256. |
