# DeepSeek-V4-Flash (NVFP4) on one B300

`nvidia/DeepSeek-V4-Flash-NVFP4` served from Rust and CUDA without Python, PyTorch or SGLang. The
reference is SGLang on a single B300 (TP=1). The port reproduces it bit for bit.

## Summary

| | Native | SGLang (the reference) |
| --- | --- | --- |
| Outputs | equal to SGLang's bit for bit | |
| Throughput (TTFT, ITL, concurrent wall time, MTP decode) | matches SGLang, within 0–4% | |
| Cold start, process start → first token | **7–9 s** on a warm host; 16–58 s on a cold one | 9–27 min from launch to ready (533–1,601 s over 12 server starts) |

* **Performance matches SGLang.** The port replays SGLang's kernels, so the device work is the
  same; the tables under Long context, Concurrent serving and MTP give each comparison.
* **Cold start is at least 60 times faster** on a warm host (8.5 s against SGLang's fastest
  533 s), and about 10 times faster with native on a cold host (up to 58 s). The SGLang figure is
  its stock server on the same Modal B300s and Volume, with no tuning of its startup (weight
  load, post-load conversion, CUDA graph capture); the native one reads a prepared package.
  A warm host is one that has itself read the package; the cache is per host, not per region
  (see Cold start).
* **[DeepSeek-V4-Pro](../deepseek_v4_pro/README.md)** on eight B300s is checked for correctness
  and cold start only, with no performance work.

## Parity

On one B300, `dsv4 run` serves the four reference prompts greedily for 128 tokens each, from the
prompt text to the response JSON. Every response equals the SGLang server's: the token ids, the
decoded text, the sampled token's logprob, the top-5 logprobs (values, ids and their order,
including the 98 positions with tied values) and the input-token logprobs. Along the way, every
forward's next-token logits and sampled id have the same SHA-256 as the reference's captured
values (`rung generate`, 4 × 128 forwards), and so do all 2061 loaded parameters
(`verify-weights`). Under the reference's own check (`json.dumps(..., sort_keys=True)` per
response), the native outputs file and the server's are identical (SHA-256 `414571751e5dbef2…`).

The reference serves identical responses with and without decode CUDA graphs, so one set of
captures covers both.

The check is exact at every level: the port launches the reference's own kernels (CUBINs taken
from the reference process) with the arguments its host code builds, in the same order and with
the same shapes. Nothing is recomputed with different arithmetic.

## Pins

| Item | Value |
| --- | --- |
| Reference | `lmsysorg/sglang` v0.5.23-cu130 (`sha256:f832b5c3…`): torch 2.14.1+cu130, flashinfer 0.7.2, sgl-deep-gemm 0.2.3.post2, sgl-flash-mla 1.0.0, sglang-kernel 0.4.10, triton 3.8.0, cuBLAS 13.1.1.3 |
| Server | `--tp 1 --trust-remote-code --disable-radix-cache`; MoE `flashinfer_trtllm_routed`, attention `dsv4` (FlashMLA), KV cache fp8_e4m3, page size 256 |
| Request | `/generate` with `temperature 0`, `max_new_tokens 128`, `return_logprob`, `top_logprobs_num 5`, `logprob_start_len 0` |
| Checkpoint | [`nvidia/DeepSeek-V4-Flash-NVFP4`](https://huggingface.co/nvidia/DeepSeek-V4-Flash-NVFP4) at `e3cd60e7de98e9867116860d522499a728de1cf9` |
| Native build | CUDA 13.0.2, gcc 13.3.0 (24.04.1), Rust 1.90.0, FlashMLA `5ddc685e` sources (sgl-project), flashinfer 0.7.2 sources and JIT build tree, `nvidia-cublas==13.1.1.3` |

## Model size

Counted from the checkpoint's safetensors headers
([`reference/tools/param_count.py`](reference/tools/param_count.py); an FP4 byte holds two
weights, and scale tensors are not parameters):

| Part | Parameters | Checkpoint bytes |
| --- | --- | --- |
| Routed experts: 43 layers x 256, 6 per token | 277.0 B | 138.5 GB (NVFP4) |
| Attention, router, norms, hyper-connections | 5.2 B | 5.7 GB |
| Shared experts | 1.1 B | 1.1 GB |
| Embedding, lm_head | 0.5 B each | 1.1 GB each |
| Scales of the quantized weights | — | 17.5 GB |
| **Model** | **284.3 B** | **164.9 GB** |
| MTP layer (the speculative draft) | 3.4 B | 3.4 GB |

A token runs 13.8 B of them (6 of 256 routed experts, the rest dense). The prepared package
(165.2 GB) holds the model's bytes in the layouts the kernels read, plus the RoPE tables. FP4
experts are what make one GPU enough: in BF16 the model would take 569 GB, twice a B300's
288 GB (275,040 MiB).

## How it works

* **Kernels.** [`reference/tape`](reference/tape) records every launch of the reference server with
  CUPTI (name, grid, block, shared memory, parameter bytes, module image). `bundle.py` turns a
  census tape (all four prompts × 128 tokens) into a kernel bundle: one CUBIN per kernel variant.
  Some names map to several CUBINs: TileLang `n_splits` and tile shapes, Triton specializations,
  and the cache a dequantization reads. Tags tell them apart, and launchers ask for
  `name#tag=value`.
* **Launch arguments.** Rust launchers ([`sgl.rs`](sgl.rs), [`torch_ops.rs`](torch_ops.rs),
  [`mhc.rs`](mhc.rs), [`compress.rs`](compress.rs), [`deepgemm.rs`](deepgemm.rs)) build each
  kernel's parameter struct the way its host code does. Where host code is too large to port, its
  own sources are compiled in:
  * flashinfer's trtllm-gen MoE launcher ([`kernels/moe_host.cu`](kernels/moe_host.cu));
  * FlashMLA's sparse prefill and decode interfaces
    ([`kernels/flashmla_host.cu`](kernels/flashmla_host.cu),
    [`kernels/flashmla_decode.cu`](kernels/flashmla_decode.cu));
  * the reference's libcublasLt, with torch's descriptors and heuristics
    ([`kernels/cublaslt.cpp`](kernels/cublaslt.cpp)).

  [`kernels/redirect.cpp`](kernels/redirect.cpp) intercepts the launches those sources make,
  including CUDA 13's `__cudaLaunchKernel`, and re-issues them from the bundle's CUBINs.
* **Weights.** [`weights.rs`](weights.rs) rebuilds SGLang's post-load parameters from the
  checkpoint: UE8M0 requantization, trtllm FP4 shuffles and scale layouts.
* **Model.** [`model.rs`](model.rs) runs the forward: mHC, attention by layer kind (sliding window,
  C4 and C128 compressed, prefill and decode, FP8 paged caches), hash or learned routing, FP4 MoE,
  the head, and greedy sampling (`torch.argmax`, first maximum).
* **Logprobs.** [`logprobs.rs`](logprobs.rs) finishes on the host what the reference's kernels
  compute on the device. Output logprobs come from torch's log_softmax, then clamp and top-5; the
  top-5 order follows ATen's multi-block gather, then its unstable 32-slot bitonic sort, emulated
  step by step. Input logprobs come from SGLang's fused logsumexp/top-k kernel, then
  `(logit - max) - log_sum` in fp32.
* **Text.** [`tokenize.rs`](tokenize.rs) encodes and decodes with the checkpoint's
  `tokenizer.json`, as SGLang does (no BOS, `skip_special_tokens`).

## Run

The [Modal harness](reference/native_dev.py) builds the crate in the native image and runs it on
a B300. The checkpoint, kernel bundle and RoPE tables live on the `dsv4-flash-native` Volume.

```sh
modal run models/deepseek_v4_flash/reference/native_dev.py --cmd run --args \
  "run $CKPT /dsv4/kernels/20261108-132810-census-v2 /dsv4/runs/20261108-121947-gold/gold/tables \
   /dsv4/runs/20261108-101648-gold/outputs-a.json 128 /dsv4/runs/20261108-native/outputs-native.json"
```

`prompts.json` is a list of prompt strings, or of reference responses (as here). In the second
case, each native response is also compared with its reference response. The parity rungs
(`dsv4 rung ...`, see [`bin/dsv4.rs`](bin/dsv4.rs)) compare intermediate tensors with the
reference captures written by [`reference/gold`](reference/gold).

The reference environment is [`reference/sglang_b300.py`](reference/sglang_b300.py): serving,
capturing goldens and tapes, and exporting tables and sources.

## Cold start

From process start to the first token takes 7–9 s on a warm host: about 1 s, plus 165 GB divided
by the read bandwidth. Serving reads a prepared package, not the checkpoint, through the
framework's streaming loader (`core_infra::model_load`,
[fast cold start](../../docs/fast-cold-start.md)):

```sh
dsv4 prepare $CKPT <params dump with SHA-256> <tables dir> /dsv4/prepared/e3cd60e7-v2   # 14 min, CPU
python -m core_infra.runtime.checkpoint manifest --root /dsv4/prepared/e3cd60e7-v2/weights \
    --file '*.safetensors' --out /dsv4/prepared/e3cd60e7-v2/manifest.json            # pin: its SHA-256
CUDA_CACHE_PATH=/dsv4/prepared/e3cd60e7-v2/cuda-cache dsv4 lt-warm                   # once per driver
DSV4_MANIFEST_SHA256=<pin> dsv4 run /dsv4/prepared/e3cd60e7-v2 <bundle> - <prompts> ...   # direct, 64 x 16 MiB readers
```

* **Package** ([`prepare.rs`](prepare.rs)): all 2061 parameters, each checked against the
  reference's params dump (SHA-256), in the layouts the kernels read, plus the RoPE tables:
  165.2 GB in 2529 safetensors shards of 64 MiB. Larger tensors are split into consecutive
  pieces so readers share them; [`prepared.rs`](prepared.rs) checks the placement. Startup does
  no CPU conversion.
* **Reads**: 64 direct-I/O readers into one device allocation. Buffered reads, even from tmpfs,
  stop near 5–7 GB/s.
* **Overlap** ([`model.rs`](model.rs) `start_prepared` / `Startup::finish`): the CUDA context,
  the kernel modules and cuBLASLt load during the reads; the first token follows the last
  weight byte by 0.03 s.
* **cuBLASLt**: the reference library ships its sm_103 GEMM kernels as PTX, which the driver
  JIT-compiles on first use (4.7–8.8 s idle, minutes on a loaded host). The package carries the
  driver's compute cache (`cuda-cache/`, 189 MB, `dsv4 lt-warm`, driver 610.57.04), which warms
  every shape in about 1.5 s during the load.

**Measurements.** `native_dev.py --cmd cold` runs the prebuilt binary in a new B300 container, as a
production image would. End to end is timed on the client from the remote call, so it includes
Modal scheduling and the container start. Every run's responses equal the reference's.

| One container | Weights read | Process start → first token | End to end |
| --- | --- | --- | --- |
| Warm host | 20–23 GB/s | **7.2–8.4 s** | 11.0–11.3 s |
| Second process on a warm host | 26.7 GB/s | 6.3 s | |
| Partly warm host | 9–14 GB/s | 12–18 s | 15.5–22.4 s |
| Cold host, or package newly written | 0.5–4 GB/s | 39–346 s | 44–208 s |

| 20 containers launched together | Process start → first token, p50 (range) | Weights read | Container start, p50 | End to end, p50 |
| --- | --- | --- | --- | --- |
| No region pinned, round 1 | 17.1 s (7.8–58.5) | 2.8–21.2 GB/s | 57.6 s | 66.8 s |
| No region pinned, round 2: 12 in us-east | 9.0 s (7.7–27.5) | 6.0–21.6 GB/s | 6.2 s | 27.7 s |
| No region pinned, round 2: 8 in us-central | 56.4 s (11.8–58.8) | 2.8–3.9 GB/s (one 14.1) | 8.2 s | 64.5 s |
| Pinned to us-east | **8.5 s** (7.7–20.7) | 19.8 GB/s p50 | 380 s | 389 s |
| Pinned to us-west (AWS hosts) | 16.4 s (15.6–35.6) | 10.2 GB/s p50 | 515 s | 531 s |
| 9 h later, 6 pinned to us-east | 10.2 s (8.2–13.0) | 19.2 GB/s p50 | 86 s | 94 s |
| 9 h later, 6 pinned to us-west | 52.3 s (48.4–69.7) | 3.2 GB/s p50 | 7.3 s | 59 s |
| 9 h later, 6 pinned to us-central | 96.8 s (64.0–141.9) | 1.8 GB/s p50 | 5.8 s | 102 s |
| 15 h later, 6 pinned to us-east | 17.4 s (8.0–65.4) | 11.0 GB/s p50 (2.5–20.9) | 727 s | 744 s |
| 15 h later, 6 pinned to us-west | 42.6 s (15.1–59.4) | 7.8 GB/s p50 (2.8–11.1) | 310 s | 326 s |
| 15 h later, 6 pinned to us-central | 64.1 s (16.3–97.2) | 3.0 GB/s p50 (1.7–10.2) | 7.4 s | 76 s |

What sets the time (2026-11-10):
* **Where the Volume data is cached: on hosts, not across a region.** A host that has read the
  package serves it at about 20 GB/s (10 GB/s on the AWS hosts of us-west); the package's warm
  hosts are in us-east, where this work has run. Nine hours after 20 containers had read it in
  us-west and 8 in us-central, six new containers pinned to each region, all started within 8 s
  (so on six different hosts), read at 2.4–3.4 GB/s in us-west (48–70 s) and 1.2–2.6 GB/s in
  us-central (64–142 s), while six in us-east read at 12.8–20.2 GB/s (8.2–13.0 s). The 10 GB/s
  that 18 of the 20 us-west containers reached the day before came from containers starting one
  after another, each most likely on the host the previous one had just warmed. Six hours on
  (15 h after the first reads) the same split showed inside one round: in us-west and
  us-central the containers that started at once read at 1.7–3.9 GB/s, and those that waited
  for a GPU, so took over a host just read, at 7.8–11.1 GB/s. The six hosts per region warmed six
  hours earlier did not show up as fast first starts (not met again, or no longer warm; the
  rounds cannot tell which). In us-east two of six read at 18–21 GB/s, two at 9.5–11 and the
  two AWS hosts at 2.5 and 6.2. More readers or CPUs do not lift a cold host, so the loader is
  not the limit.
* **How long a host stays warm, and why a region can look cached.** For the video models (Warm
  Volumes in the [top-level README](../../README.md), B200, pinned to us-west), Wan's P50 held
  after about 23 idle hours, while H3 FL2VA, which reads twice as much, slowed after about 7.
  Those models had run many rounds of 30 containers in the region, enough to warm most of its
  hosts, so any new container met a warm one. DSV4 has read each of us-west and us-central for a
  round or two, which warms a few hosts and leaves the rest cold. How long one host keeps 165 GB
  is not measured: the rounds here cannot tell a host not met again from a host gone cold.
* **Free B300s.** Unpinned, 10–17 of 20 containers started within 30 s and the rest waited up to
  96 s. Pinned (`COLD_START_REGION`, which costs extra), they started about one at a time, so end
  to end was mostly queueing.

What follows for serving on Modal:
* The 7–9 s figure holds when the container lands on a host that has read the package. One or
  two rounds of reads in a region do not make later containers there fast.
* A region becomes reliably fast only once most of its hosts have read the package: several
  rounds of many-container cold starts, as for the video models (not done yet for DSV4; with
  B300s queueing for minutes when pinned, one round takes a quarter of an hour or more).
* Otherwise a fast start needs warm containers kept running (`min_containers`).

The development path (`dsv4 run <checkpoint dir> ...`) still rebuilds every parameter on first
use, and its first request takes about 6 minutes.

## Long context

Native serving matches SGLang at 16k–128k tokens, bit for bit and in throughput.

**Reference.** `sglang_b300.py --long-gold-` serves real-text prompts of 16,384, 32,768, 65,536,
102,400 and 131,072 tokens (SGLang's Python sources, tokenized) for 17 new tokens each. One server
runs with CUDA graphs; a second runs eager, with the census tape and hash-only golds, capturing
every 16k prefill chunk and 16 decode steps per prompt (run `20261109-035215-longgold`, both
servers' outputs identical). The server runs `--disable-radix-cache --dsa-topk-backend flashinfer`
with `SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC=1`. SGLang's default sparse-indexer top-k
(sgl-kernel) is not deterministic at these lengths, and its `--enable-deterministic-inference`
refuses the dsv4 attention backend. Throughput is the same under both top-k backends.

**Parity.** `dsv4 rung long-prefill` runs each prompt in SGLang's 16k chunks and compares every
layer's output (and each attention output captured) with the reference, then the next-token
logits and sample, then 16 decode steps (logits and sample). At all five lengths, every chunk is
equal through all 43 layers, and so are all decode steps. This covers chunked prefill over a cached
prefix, the 4,096-token tail of the 102,400-token prompt (SGLang's paged indexer path below 8,192
queries), and decode at up to 131k tokens of context. `dsv4 bench-long` also checks the served
outputs (tokens and top-5 logprobs) against the reference server's at every length.

**What it took.**
* The C4 sparse indexer. Its compressor writes the index-K cache (FP8 keys plus a scale per entry).
  Each query then gets MQA logits over every compressed key it may see: DeepGEMM `sm100_mqa_logits`
  (non-paged, keys gathered once) or `sm100_paged_mqa_logits` (short chunks and decode), with host
  descriptors ported from DeepGEMM 0.2.3.post2 ([`deepgemm.rs`](deepgemm.rs)). flashinfer's
  deterministic page-table top-k picks the top 512 ([`topk.rs`](topk.rs)). The paged schedule is
  computed on the host with SGLang's algorithm; it only distributes work.
* C128 compression inside a chunk, and the combined [top-k | window] indices of every layer
  kind built on the device by SGLang's `_combine_topk_swa_indices_kernel`.
* Per-chunk metadata (plans, slots, lengths) built once and uploaded in one copy, as SGLang's
  forward metadata. Layers make no host round trips.

**Throughput** on one B300. Native is measured with `dsv4 bench-long <prepared> <bundle> <run> 0,1,2,3,4 256 3`;
SGLang with `sglang_b300.py --bench-long-` (same server flags). TTFT is the uncached prefill to
the first token; ITL is (total - TTFT) / 256 over a 257-token greedy request.

| Prompt | TTFT native | TTFT SGLang | ITL native | ITL SGLang |
| --- | --- | --- | --- | --- |
| 16,384 | 0.326–0.331 s | 0.326 s | 5.79–5.80 ms | 5.85 ms |
| 32,768 | 0.678–0.689 s | 0.673 s | 5.83 ms | 5.89–5.99 ms |
| 65,536 | 1.464 s | 1.427 s | 5.94 ms | ~5.97 ms |
| 102,400 | 2.47 s | 2.405 s | 6.09 ms | 6.06 ms |
| 131,072 | 3.27 s | 3.18 s | 6.17 ms | 6.24–6.4 ms |

The device work is the same; the host-side changes that got it there:
* **Decode overlap.** As SGLang's graphs do, the KV store, the compressor and the whole indexer
  chain run on side streams beside the query path, and the shared expert runs beside the routed
  experts. Each side stream has its own cuBLASLt workspace; values are unchanged.
* **Overlapped decode loop.** `Model::generate_overlapped` samples on the device (SGLang's own ATen
  argmax kernel on the fp32 row) and writes the token into the next step's inputs. The next
  step's positions are uploaded while the current step runs, so the host never waits per token.
  Graphs and the KV cache persist across requests, as SGLang captures its graphs once.
* **Prefill.** No per-layer uploads (chunk metadata, device-built indices), and the FFN writes the
  layer output in place: a 128 MB copy per layer at 16k tokens is gone.

## Concurrent serving

Native serving of 32–256 concurrent 16k-token requests matches SGLang bit for bit, and at its
throughput, on SGLang's own schedule: past about 78 prompts it interleaves prefills with decode
steps, retracts and re-prefills requests, and packs requests into one prefill.

**Reference.** `sglang_b300.py --concurrent-gold- CxL` sends C real-text prompts of L tokens at once
(33 new tokens each, top-5 logprobs). Server A runs with CUDA graphs. Server B runs eager, with the
census tape and golds of every forward (`DSV4_GOLD_MODULES=logits_processor` keeps only logits and
samples, which the concurrency check reads, and keeps a 256-request capture under the Volume's
inode limit). Runs:
* `20261109-073208-concgold32x16384`, `-075407-concgold48x16384`, `-084910-concgold50x16384`;
* `20261109-190718-concgold128x16384`;
* `20261109-224014-concgold256x16384`.

How SGLang schedules them (from the golds):
* **Up to 50 requests.** It prefills every prompt (one 16k chunk each), then decodes all of them as
  one batch.
* **Past the SWA pool** (`swa_full_tokens_ratio` 0.1: 1,278,464 SWA tokens, about 78 prompts of
  16k, since a prompt holds its whole SWA history until it decodes), it works in waves. Each wave
  prefills as many prompts as fit, then decodes twice; the decode step frees the out-of-window SWA
  pages. On its first decode it finds the pool full and retracts the last requests (a retracted
  request is prefilled again later, keeping the tokens it had sampled). The re-prefills can pack two
  requests into one extend batch.
  * 128: `P x78, D76 x2, P x50`, 2 retractions and their re-prefills (one of 16,384 + 1 tokens),
    `D128 x30`, then the tail leaves.
  * 256: three waves (`D76`, `D150`, `D222`), 4 retractions, and two packed prefills of two
    requests each (`[1 + 16,128]` and `[256 + 16,128]` tokens), `D256 x26`, then the tail.

**Parity.** `DSV4_BATCH_CHECK=<gold dir> dsv4 bench-concurrent <prepared> <bundles> <run> 32 1`
replays the golds' schedule and compares each forward's logits and samples with server B's:

| Requests | Forwards checked | Equal |
| --- | --- | --- |
| 32 / 48 / 50 | 64 / 96 / 82 | all |
| 128 | 164 of 165 | all |
| 256 | 298 of 299 | all |

The unchecked forwards sample nothing (a re-prefill's first chunk). The 256 count includes both
packed prefills.

Server A pads a decode batch to its next graph size (1, 2, 4, 8, 12, 16, then multiples of 8), and
padding can change rows' low bits:
* At 50 rows (graph 56) the outputs are identical.
* At 76 rows (graph 80) they are not: A and B agree on 95 of 128 requests' tokens at 128, and 218 of
  256 at 256.
* Native without the check pads as A does. At 128 its tokens equal A's for all 128 requests.
* At 256 they equal A's for 239 of 256. Some of A's padded sizes (152, 224, ...) ran kernels and MoE
  tactics no tape recorded (A runs without one); a tape of server A would close this.

Kernels come from the merged bundles
`20261109-224014-conc256b:20261109-190718-conc128:20261109-084910-conc50:20261109-075407-conc48:20261109-073208-conc32:20261109-035215-long-v3`.

**Throughput.** Same workload as SGLang's `--concurrent-` baseline: 256 new tokens per request,
ignore_eos, no logprobs, the same schedule (the golds' admission, then decode to 257 tokens). Wall
time runs from the first prefill to the last token. The first configuration a SGLang server serves
carries its first-request overheads, marked *.

| Concurrency x prompt | Wall native | Wall SGLang | Output tok/s native / SGLang | Decode step native / SGLang |
| --- | --- | --- | --- | --- |
| 32 x 16,384 | 15.08–15.11 s | 15.06 s (17.06 s*) | 545 / 546 | 16.63 / 16.42 ms |
| 48 x 16,384 | 21.02–21.28 s | 20.76 s | 580–587 / 594 | 18.88–18.91 / 18.85 ms |
| 50 x 16,384 | 21.60–21.76 s | 25.75 s* | 591–595 / 499* | 18.89–19.01 / 19.59 ms |
| 64 x 16,384 | | 26.16 s | / 629 | / 20.32 ms |
| 128 x 16,384 | 50.74–51.16 s | 49.22 / 50.34 s | 643–648 / 654–668 | 25.0–25.3 / 24.7–24.9 ms |
| 256 x 16,384 | 95.29–96.97 s | 96.11 s | 679–690 / 685 | 29.6–30.3 / 30.5 ms |

The 128 and 256 SGLang decode steps are its full-batch steps. The native ones average over all
steps, including the wave steps at smaller batches.

Each native 16k prefill takes 0.337–0.342 s under sustained load, against SGLang's 0.332 s. The
profile shows the GPU continuously busy (idle 17 ms over 10.6 s), so this is device time, not host
overhead. One suspect is the MoE tactics: native uses the ones the capture's autotuner chose, while
SGLang's server picks them again at each start.

**How.** `Model::serve(prompts, max_new, schedule)` runs a list of `Step`s, the forwards SGLang's
scheduler chose: `Prefill { chunks }` (one or more requests' chunks in one forward),
`Decode { rows }` (a batch in row order) and `Retract { req, keep }`. `bench-concurrent` reads
them from the golds (request-pool rows, extend lengths, forward modes); `generate_batch` builds
the prefill-then-decode schedule.
* **Placement.** Each request owns a range of full-pool pages (C4, C128, index-K) and a
  request-pool row (the C128 state ring).
* **SWA pool.** Its SWA pages come from a separate pool, as in SGLang's hybrid SWA allocator. It
  holds a page while writing it and gives back the pages out of its window (`swa_hold`,
  `swa_evict`, `swa_release`). The C4 compressor states are keyed by SWA page, so they move with
  them.
* **Packed prefill.** A forward over several requests (`Batch::segs`) lays out the chunk metadata
  per request as SGLang's sparse-prefill cache does:
  * the window union, compress plans (ragged ids across the batch) and combine metadata
    (`_combine_topk_swa_indices_kernel`, whose `SEARCH_STEPS` constexpr follows the request count:
    the bundle's `steps` tag);
  * a paged indexer with one block-table row per query (SGLang's non-paged indexer serves batches of
    one request only);
  * logits for each request's last token, one LM-head GEMM over them.
* **Retraction.** `Retract` frees the request's SWA pages and drops the tokens SGLang did not keep.
  Its next prefill reads the kept tokens from the device token log.
* **Decode steps.** Per-row decode inputs (`DecodeLayout`) feed batched compressors, the paged
  indexer with per-request block tables, and FlashMLA with b rows.
  * Sampling is SGLang's ATen argmax over the rows (its config ported from `setReduceConfig`).
  * The samples feed the next step on the device. When the rows change, a few device copies
    gather each row's latest token.
  * There is a CUDA graph per batch size and indexer regime, and no host wait per step.
* **Row order** matters bitwise: FlashMLA's split-KV schedule partitions the whole batch. So
  `bench-concurrent` serves the requests in the order the golds recorded.
* **DeepGEMM tilings.** DeepGEMM compiles one kernel per tiling, and a shape SGLang never ran (the
  16,129-token packed prefill) needs one no tape holds. Native takes the heuristic's best tiling
  whose kernel the bundle has. Tilings only split M and N, and every output element sums its K
  blocks in the same order. The 16k prefill's `wo_a` runs 128x256 natively and 240x128 swapped in
  SGLang, bit for bit equal, and both packed prefills check equal.

Not covered yet:
* **Live traffic.** Native follows a schedule (SGLang's, or prefill-then-decode); it has no
  scheduler of its own for requests that arrive and finish at any time.
* **Short prompts.** Several of them packed into one chunk (SGLang's 32 x 2,048 baseline: 6.13 s)
  need no new machinery but have no captured schedule.
* **Serving features.** Greedy sampling with ignore_eos only.

## MTP (speculative decoding)

SGLang serves DeepSeek-V4 with its MTP layer as an EAGLE draft (`--speculative-algorithm EAGLE
--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`). Native
runs the same loop, forward for forward (`Model::generate_mtp`, `dsv4 mtp-generate`), one request
for now.

| 16k prompt, 1,024 greedy tokens | SGLang MTP | native MTP |
|---|---|---|
| tokens equal to SGLang's | — | 1,024 / 1,024 |
| mean accept length | 2.90–3.15 (logged) | 2.92 |
| decode tok/s | 338–366 (logged gen throughput) | 333 (wall, first rounds' graph capture included) |

The SGLang reference's graph server and eager server produce identical tokens (run
`20261110-042007-outputs1x16384d1023`). The forward-by-forward checks of the 33-token gold run
(`mtp-target`, `mtp-draft`) are bit-exact for all 50 forwards, logits and hidden states.

One round, with L the committed length and x the token pending at L:
1. **Two draft decodes.** d1 at L, then d2 at L + 1. Each attends only to the committed window
   (seq_lens L) and reads the draft's own previous hidden state. They give d2 and d3 (SGLang's
   Triton top-1, the first maximum).
2. **Target verify** of [x, d1, d2, d3] at L..L+3. Attention runs as decode rows. The compressors
   run as an extend of the four tokens (prefill kernels on the request's plans, with the wider
   speculative rings: C4 16, C128 256). tp is each row's ATen argmax.
3. **Accept.** a is the number of leading d_i == tp[i-1]. Commit x, d1..da; tp[a] is the next x.
   The rejected rows' C128 ring slots are reset (kv 0, score -inf), as SGLang's
   `clear_unaccepted_c128_draft_states` does.
4. **Draft extend** over the four verified rows. Ids are tp[0..=a], with 0 for rejected rows,
   and hidden states come from the verify's pre-hc_head output. The next d1 is row a's argmax.

After the prefill, the draft extends over the prompt shifted by one plus x, and its d1 is the
maximum of flashinfer's cake softmax (16-CTA cluster kernel, replayed).

The draft layer, as SGLang loads it:
* Its weights are rebuilt from the checkpoint (`load_draft_weights`, 44/44 parameters equal to
  SGLang's).
* Its experts are MXFP4 (trtllm-gen MxE4m3 x MxE2m1 cubins, swiglu clamp 10).
* `hc_head` is torch ops: ATen reductions, rsqrt and sigmoid replayed; torch's fp32 cuBLAS GEMM;
  the single IEEE ops in `exact_ops.cu`.

Performance:
* Draft tokens pass between forwards on the device. The host reads each round's eight tokens once,
  after the verify.
* Each forward kind replays as a CUDA graph, one verify graph per plan-count pattern
  (`VerifyKey`). The verify metadata lives in a fixed region (`verify_meta`).

### Concurrent MTP

`Model::serve_mtp` (`dsv4 mtp-serve`) batches the same loop over SGLang's schedule, read from a
gold run's target forwards: each EXTEND prefills requests, each TARGET_VERIFY is a round over its
request-pool rows. In a round:
* The draft decodes take one row per request, in batch order; the verify and the draft extend take
  four rows per request.
* Each request's d1 and draft hidden state live on the device indexed by request. Small copy
  kernels gather them into batch order and scatter them back.
* The C128 reset is one kernel launch for every request and layer.
* SGLang's overlap scheduler runs a finished request for one more round and drops its tokens.
  Native does the same.

| 16k prompts, 257 greedy tokens each | SGLang MTP | native MTP |
|---|---|---|
| 4 requests: forwards bit-exact | — | 60 / 60 |
| 64 requests: forwards bit-exact | — | 546 / 546 |
| 64 requests: requests equal to SGLang eager / graph server | 49 (graph vs eager) | 64 / 49 |
| 64 requests: wall | 30.87 s | 26.47–26.61 s |
| 64 requests: decode, rounds 41–80 (full batch) | 5,482 tok/s | 5,633 tok/s (graphs kept), 5,365–5,416 (first pass) |
| 128 requests: forwards bit-exact (2 retractions in the schedule) | — | 704 / 704 |
| 128 requests: requests equal to SGLang eager / graph server | 16 (graph vs eager) | 128 / 16 |
| 128 requests: wall | 54.43 s | 51.09–51.13 s |
| 128 requests: decode, rounds 41–80 (full batch) | 8,871 tok/s | 8,793–8,796 tok/s (graphs kept), 8,480–8,543 (first pass) |
| 256 requests: forwards bit-exact (4 retractions) | — | 984 / 984 |
| 256 requests: requests equal to SGLang eager / graph server | 63 (graph vs eager) | 256 / 63 |
| 256 requests: wall | 101.09 s | 99.05–99.07 s |
| 256 requests: decode, rounds 41–80 (full batch) | 13,323 tok/s | 13,092–13,129 tok/s (graphs kept), 12,779–13,004 (first pass) |
| 256 requests: decode, rounds 81 to the end (the batch drains) | 6,398 tok/s | 6,315–7,462 tok/s |

**How decode is compared.** SGLang logs its generation throughput once every 40 decode rounds
(`decode_log_interval`): the tokens of those 40 rounds over the time between the two log lines.
At these sizes its rounds 41–80 run the full batch with no prefill in between, and that is the
SGLang figure above. Native prints each round with `DSV4_MTP_TIMES=2` (step, batch, tokens
committed, time the verify finished), and the native figure is the same rounds 41–80 (three reps).
A full-batch round takes 54.5 ms at 256 (SGLang: 712 tokens at 13,323 tok/s, 53.4 ms), 40.0 ms
at 128 (SGLang 39.8 ms) and 32.0 ms at 64 (SGLang 32.7 ms). An average over every round after
the last prefill (about 11,000–11,600 tok/s at 256) also counts the draining batch and is not
comparable to SGLang's logged figure.

**Graphs.** A verify graph is keyed by its batch and its plan counts (`VerifyKey`); the C128
count changes from round to round, so a run meets dozens of keys. Each slot runs eagerly on its
first two uses and captures on its third (`DSV4_GRAPH_AT`). At 256 requests an eager verify round
takes 55.2 ms, a replay 54.3 ms and a capturing round 95 ms, so the only slow rounds are captures,
and capturing earlier makes the first pass slower (19 captures in rounds 41–80, 9,300 tok/s).
`serve_mtp` keeps its graphs, arenas and buffers between calls of the same shape (`MtpServing`),
as a server keeps them for its lifetime; the KV cache stays allocated and is zeroed at each call.
From the second call on, rounds 41–80 replay every verify, with no round above 55 ms.
`DSV4_MTP_FRESH=1` starts over at each call. Graphs live in the process (device addresses, loaded
modules): they cannot be stored with the package, and a new process captures them again, as
SGLang captures its own at every start.

**Verify inputs on the device.** At 1,024 verify rows the decode inputs are 36 MB, 35 MB of it the
C4 and C128 entry lists (8,256 entries per row) and the page table, all of them arithmetic in
the request's first page. The verify uploads only the per-row values and five parameters per row,
and `fill_decode_ids` (kernels/exact_ops.cu) writes the lists on the device. Host time building
inputs went from 3.4–9.5 ms to about 1.9 ms per round at 256, and full-batch decode (first pass)
from 11,000–11,100 to 12,800–13,000 tok/s. `DSV4_MTP_HOST_IDS=1` builds them on the host as before.

SGLang's own graph and eager servers disagree on 15 of the 64 requests, as without MTP (padded
graph batches change low bits). Native runs exact batch sizes. Each graph is captured on its third
use (`DSV4_GRAPH_AT`), so batch sizes the tail sees once or twice run eagerly.

Running 256 requests needed three things:
* **Verify metadata slots.** They hold plans for up to 2,048 verify rows.
* **MoE workspace.** It takes up to 1 GiB of the arena: the MXFP4 draft extend over 1,024 rows needs
  more than 256 MiB.
* **MXFP4 token counts no capture recorded.** These take the tactic of the nearest recorded counts
  below and above, when the two agree. The recorded MXFP4 choices are a step function of the
  count; the FP4 ones are not, so FP4 keeps its census table.

Prefills follow SGLang's chunks: several requests packed into one forward, chunks of part of a
sequence, and the re-prefill of a retracted request (its prompt plus the tokens it keeps). The
draft's extend runs over the same chunks, each shifted by one. A chunk that ends its sequence
takes the sampled x as its last id; any other chunk takes the sequence's next token.

Not covered yet: the draft weights in the prepared package (they are rebuilt from the checkpoint at
start).

## DeepSeek-V4-Pro on eight B300s

The same code built with the crate feature `pro` runs one tensor-parallel rank of
`nvidia/DeepSeek-V4-Pro-NVFP4` per GPU (SGLang `--tp 8`), up to the first token of a short prompt.
On a host that has read the 928.5 GB package, eight fresh processes reach the first token in
26.5–26.6 s (32 readers per rank; 27.7–29.9 s and one 30.8 s over 11 runs with 16), and the next
token, the logits and every sum over the ranks equal SGLang's bit for bit on the four reference
prompts. That port covers cold start and correctness only: no decode, serving, MTP or throughput
work, so none of this document's performance figures apply to it. Details, commands and limits:
[`../deepseek_v4_pro/README.md`](../deepseek_v4_pro/README.md).

The harness builds Flash by default; `native_dev.py --pro` builds with the `pro` feature into its
own target directory.

## Scope

Covered: one request at a time, greedy sampling, prompts up to 131k tokens prefilled in SGLang's 16k
chunks, decode with CUDA graphs. Outside that scope the port fails with an error, never silently.
* Kernel variants come from the captured tapes (census for the short prompts, the long-gold tape
  for 16k–128k). Triton bakes some shapes into its images, for example the combined index width,
  one image per width; a prompt length whose variant no tape caught needs a new capture.
* The MoE tactic per token count follows the reference run (`moe_tactics` in the bundle).
* `generate_overlapped` serves without logprobs or stop tokens (ignore_eos); `generate` computes
  logprobs and checks stop tokens per step.
* Concurrent serving runs a schedule SGLang chose (its waves, retractions and packed prefills
  included) or prefill-then-decode, up to 256 requests (SGLang's cap for DeepSeek-V4 and its
  largest decode graph); see above.
* A DeepGEMM shape no tape caught runs the heuristic's best tiling the bundle has (same output
  bits); other kernels with shape-specific images (Triton specializations, MoE tactics) still need a
  capture.
