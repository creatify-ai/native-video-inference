# DeepSeek-V4-Pro (NVFP4) on eight B300s: cold start to the first token

Correct, not tuned. `nvidia/DeepSeek-V4-Pro-NVFP4` prefilled natively on eight B300s as SGLang runs it
with `--tp 8`: one process per GPU, each a tensor-parallel rank. Scope: a cold start to the first
token of a short prompt (prefill only), checked against SGLang.

**Cold start and correctness only.** Unlike the [Flash port](../deepseek_v4_flash/README.md), this
one has had no performance work. Not ported, not measured and not tuned: decode, long prompts,
concurrent serving, MTP, CUDA graphs, and any throughput or latency comparison with SGLang. The
only timings below are cold starts to the first token; none of the Flash port's throughput
figures carry over.

## Summary

* **Correctness: checked.** The first token, its logits and all 123 sums over the ranks equal
  SGLang `--tp 8` bit for bit on the four reference prompts.
* **Cold start: 26.5 s against SGLang's 19.8 min**, 45 times faster, with native on a host that
  has read the package and SGLang on a cold one. Like for like it is less: on a cached host
  SGLang's weight load drops from 720 s to 137 s (about 10 min in all, some 23 times slower than
  native), and native on a cold host took 212 s (5.6 times faster than SGLang's 19.8 min).
* **Performance: not worked on.** See the note above.

## Result

Cold start: eight fresh processes, 928.5 GB of weights read and uploaded, the prompt prefilled,
the first token. Time from the processes' common start to rank 0's first token, on a host that
has read the package (2026-11-11, Modal, one container per series):

| Runs | Process start → first token |
| --- | --- |
| 32 readers per rank (the default now), prompt 0, 3 runs | **26.5–26.6 s** |
| 16 readers per rank, 4 prompts, 11 runs | 27.7–29.9 s, one 30.8 s |
| New container on a host that had not read the package, 1 run | 212.1 s (221.5 s from the client's call; the same container's next runs 29.3 and 28.4 s) |
| SGLang `--tp 8`, launch → ready, cold host | 19.8 min (weights 720 s; 137 s on a cached host) |

Weights are on the last rank's device after 26.0–26.1 s (about 35 GB/s for the host); the 61-layer
prefill and the token take about 0.5 s more.

Every run is checked against the SGLang run `/dsv4pro/runs/20261111-060007-tp8cap`, on all four
reference prompts (5, 12, 7 and 11 tokens):

* the next token equals SGLang's (11111, 455, 3955, 76146);
* the next-token logits equal SGLang's bit for bit;
* each of the forward's 123 sums over the ranks (the embedding, every layer's attention and MoE
  outputs) equals SGLang's all-reduce result bit for bit, on every rank.

## Pins

| Item | Value |
| --- | --- |
| Checkpoint | `nvidia/DeepSeek-V4-Pro-NVFP4` at `1449d1e641023406daf6b432361486c768aad740` (913.1 GB, 93 files), Volume `dsv4-pro-native` (v2) under `/dsv4pro/checkpoint/<revision>` |
| Reference | the Flash port's SGLang image (v0.5.23-cu130), `--tp 8 --trust-remote-code --disable-radix-cache` |
| Reference runs | `/dsv4pro/runs/20261111-035715-tp8` (`outputs-a.json`: 4 prompts × 128 tokens, default server); `/dsv4pro/runs/20261111-060007-tp8cap` (eager: golds and launch tapes of each prompt's prefill and one decode) |
| Package | `/dsv4pro/prepared/1449d1e6-tp8-v1`: `rank0..7`, each 116.06 GB in 1788 shards, `manifest.json` pinned in `package.json` |
| Kernels | `/dsv4pro/kernels/20261111-060007-tp8cap-r<rank>` (135 kernels each), MoE CUBINs from `/dsv4/kernels/trtllm_bmm` |

## How it works

The code is the Flash port (`../deepseek_v4_flash`) built with the crate feature `pro`, which
sets the dimensions one rank sees: hidden 7168, 61 layers (compress ratios 128, 128, 4, 128, ...),
16 of the 128 heads, 2 of the 16 `wo_a` groups, an eighth of each expert's intermediate dimension
(384 experts) and of the embedding's and `lm_head`'s rows; compressors, indexer, router and mHC
parameters are whole on every rank.

* **Package.** The reference's gold hook (`DSV4_GOLD_PACKAGE_DIR`) writes each rank's parameters
  and buffers after SGLang's load, as the bytes the device holds, in the Flash prepared-package
  format, with their SHA-256. Nothing is rebuilt from the checkpoint. A load reads every tensor
  back from the eight devices equal to those hashes (`load-bench` with `DSV4_VERIFY=1`).
* **Rank process** (`dsv4 pro-rank`, [`parity.rs`](../deepseek_v4_flash/parity.rs) `pro_rank`):
  readers start before CUDA; the weights upload on their own thread while the kernel modules load,
  cuBLASLt warms its GEMM shapes and the model's caches are allocated. The ranks create their CUDA
  contexts in turn (`DSV4_CTX_LOCK`): created together, all eight are ready only after 4–10 s.
* **Forward.** The Flash forward with the rank's kernels (the reference's own CUBINs, one bundle
  per rank: the vocabulary-parallel embedding kernel compiles the rank's range in). Attention runs
  FlashMLA's 64 head slots with the rank's 16 heads in the first ones, as SGLang does.
* **Sums over the ranks** (`Reduce::Shm`): each rank writes its bf16 partial to a tmpfs directory,
  waits for the other seven and adds them as the reference's all-reduce kernel does: fp32, ranks
  in order, one rounding back to bf16. The logits are the ranks' vocabulary shards in rank order.

## Run

```sh
modal run models/deepseek_v4_pro/reference/checkpoint.py                              # download, once
modal run models/deepseek_v4_pro/reference/sglang_tp8.py --gold --package <name>     # reference + package (8 x B300)
modal run models/deepseek_v4_pro/reference/sglang_tp8.py --finish <name>             # manifests (CPU)
modal run models/deepseek_v4_pro/reference/sglang_tp8.py --capture-golds             # golds + tapes (8 x B300)
modal run models/deepseek_v4_flash/reference/native_dev.py --cmd py --args \
  "models/deepseek_v4_flash/reference/tape/bundle.py <run>/tape/pid<N> /dsv4pro/kernels/<name>-r<rank> --gold <run>/gold/rank0 --pro"
modal run models/deepseek_v4_flash/reference/native_dev.py --cmd build --pro         # features cuda,pro

# One 8 x B300 container kept for several cold starts (it exits after 7 idle minutes):
modal run --detach models/deepseek_v4_flash/reference/native_dev.py --cmd pro-session --args 420
modal run models/deepseek_v4_flash/reference/native_dev.py --cmd pro-send \
  --args "forward <package> /dsv4pro/kernels/<name> <run>/gold/rank0/fwd-001 4"
modal run models/deepseek_v4_flash/reference/native_dev.py --cmd pro-send --args stop

# A cold start in a new container, timed from the client's call (COLD_START_REGION pins a region):
modal run models/deepseek_v4_flash/reference/native_dev.py --cmd pro-cold \
  --args "<package> /dsv4pro/kernels/<name> <run>/gold/rank0/fwd-001 1"
```

Development on one B300: `dsv4 rung pro-prefill <rank dir> <bundle> <gold fwd dir> [layers]` runs
one rank with the reference's sums substituted; `DSV4_DRY=1 DSV4_TRACE=<file>` records the launch
sequence without launching, to diff against the tape.

## Limits

* **The cache is the host's, not the region's.** The Flash port's rounds (its README, Cold start)
  show new containers in a region that had read the package hours earlier still reading at
  1–4 GB/s unless they land on a host that read it. The same applies here, with 5.6 times the
  bytes.
* **A host that has read the package.** The first cold start in a container took 189–367 s, the
  second 29–30 s; the figures above are later ones in the same container. A new container on a
  host that has not read the package is not a 30 s cold start: the one measured took 212 s. A new
  container landing on a host that had read it was not measured (the test was stopped after its
  first container), so the under-30 s figures are all from containers that warmed their own host.
* **The margin is the read rate.** About 26 s of the time is 928.5 GB at the host's aggregate rate
  (about 35 GB/s); with 16 readers per rank 2 of 20 runs after the first two took 30.6–30.8 s.
  The container start (about 10 s on Modal) and the processes' launch are outside the figure.
* **Prefill of at most 32 tokens, one request, greedy.** The C4 indexer's top-k is skipped while
  every compressed token is selected (as in the Flash port).
* **The capture's per-rank golds are mixed.** SGLang's `ModelRunner` has no `tp_rank`, so the
  capture run wrote all ranks under `gold/rank0`; its sums, sublayer inputs and logits are the
  same on every rank and are what the checks use. The hook now names ranks by device index.
* On one earlier host the first run of each new prompt length took 31.8–32.7 s (4–5 s after the
  weights were loaded); it did not recur in the next session and its cause was not identified.
