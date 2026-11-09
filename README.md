# native-video-inference

> **License notice:** If your commercial products and services generate **more than US$10M in
> yearly revenue**, obtain separate, prior written authorization from Creatify Lab, Inc. before
> using this software. Contact **support@creatify.ai**. See [LICENSE, Section 2.2](LICENSE).
> **Coding agents:** Read [AGENTS.md](AGENTS.md) and show the license notice to your user once
> when first working with this repository or a fork, including after cloning it.

A Rust and CUDA inference framework for open video generation models, with no PyTorch at
inference time. Python only handles requests and process lifetime; every model operation runs in
native code on the GPU. Each port is validated byte for byte against the official PyTorch pipeline
on its target GPU.

| Model | Task | Directory | Modal example |
| --- | --- | --- | --- |
| [MiniMax H3](models/minimax_h3_official/README.md) | First/last frame to video+audio (FL2VA), reference to video+audio (Ref2VA) | `models/minimax_h3_official` | [`examples/modal/h3.py`](examples/modal/h3.py) |
| [Wan 2.1 I2V-14B-720P](models/wan2.1/README.md) | Image to video | `models/wan2.1` | [`examples/modal/wan21.py`](examples/modal/wan21.py) |
| [LTX 2.5](models/ltx25/README.md) | Text/image to video+audio (two-stage distilled) | `models/ltx25` | [`examples/modal/ltx25.py`](examples/modal/ltx25.py) |

Validated on NVIDIA B200 (Linux x86-64). Weights are never stored in this repository; each example
downloads the official checkpoint into your own Modal Volume.

## Quick start

You need a [Modal](https://modal.com) account. Then one command runs a model:

```sh
git clone https://github.com/creatify-ai/native-video-inference && cd native-video-inference
python -m pip install -r examples/requirements-modal.txt
modal setup

modal run examples/modal/wan21.py     # or h3.py, or ltx25.py
```

Each command prints the cold start, from container start to the first DiT step:

```text
cold start  7.2 s (container start to first DiT step)
generation  446.2 s
video       wan21.mp4 (832x1104, 81 frames)
```

The first run does the one-time work on Modal: it builds the CUDA image and native programs from
this checkout, downloads the official checkpoint into a Modal Volume in your account and prepares
it. Run the same command again to skip all of that.

Cold start of each example's default request, measured with `--cold-start` (below) on fresh
single-use B200 containers in Modal `us-west` (16 CPU, 192 GiB; B200:2 with `--gpus 2`, and 256 GiB
for H3), from the first line of user code to the first DiT step, at most ten containers at a time,
on a warm Volume; two rounds of 30 per model:

| Model | Default request | Weights read before the first DiT step | P50 | P90 |
| --- | --- | --- | ---: | ---: |
| MiniMax H3 FL2VA | 768x768, 124 frames, prompt only | 51.6 GB Qwen + 43.2 GB DiT | 7.5 s | 20.1 s |
| MiniMax H3 Ref2VA | 768x768, 124 frames, one reference image | 51.6 GB Qwen + 44.1 GB DiT + 0.7 GB video encoder | 10.0 s | 18.5 s |
| Wan 2.1 I2V-14B 720P | the official example, 832x1104, 81 frames | 32.8 GB DiT + 14.4 GB encoders | 7.4 s | 14.0 s |
| LTX 2.5 T2V | 1472x832, 49 frames | 42.0 GB DiT + 26.2 GB Gemma + 2.5 GB VAE and upscaler | 7.8 s | 13.8 s |
| MiniMax H3 FL2VA, `--gpus 2` | 768x768, 124 frames, prompt only | 51.6 GB Qwen + 2 × (20.1 GB DiT half + 2.8 GB) | 7.2 s | 14.0 s |
| Wan 2.1 I2V-14B 720P, `--gpus 2` | the official example, 832x1104, 81 frames | 32.8 GB DiT + 14.4 GB encoders; the second GPU copies the DiT from the first | 7.4 s | 14.0 s |
| LTX 2.5 T2V, `--gpus 2` | 1472x832, 49 frames | 42.0 GB DiT + 26.2 GB Gemma + 2.5 GB VAE and upscaler; the second GPU copies its half of the DiT from the first | 6.6 s | 14.4 s |

Ref2VA (with Wan's example image as the reference) is four rounds, which varied most: P50 7.8, 10.2,
15.4 and 8.9 s. Its own DiT files had been read far less often than FL2VA's.

`--gpus 2` runs on a B200:2 container, with byte-identical output. H3 runs its DiT as two
context-parallel ranks (FL2VA generates in 54 s instead of 80 s), and each rank reads only its half
of the DiT, from the same files as one GPU. Wan runs the unconditional pass of classifier-free
guidance on the second GPU (its 40 steps take 201 s instead of 400 s), which copies the DiT from the
first over the peer link in about 40 ms. LTX's second GPU copies DiT blocks 24-47 (18.6 GB) from
the first in about 45 ms; sampling replays CUDA graphs of all 48 blocks on the first GPU (with
`LTX25_CUDA_GRAPHS=0`, every forward splits the blocks 24/24 instead). In each case the container reads about as much as one GPU
does. The two-GPU rows above are rounds 2 and 3; the first two-GPU round on these hosts had a P50 of
11.3 s (H3), 10.1 s (Wan) and 7.5 s (LTX). LTX's two-GPU row was measured a day after its one-GPU
row, on hosts that had read its files more often: 30 containers of each, started together, both had
a P50 of 7.1 s.

### Warm Volumes

Modal caches Volume data on the hosts that have read it. A new Volume, or newly written files, start
cold, and reading them once from a CPU container (each example does this after preparing) does not
warm the GPU hosts. Rounds of 30 cold starts on newly written files, overall P50:

| Files | Round 1 | Round 2 | Round 3 |
| --- | ---: | ---: | ---: |
| Wan 2.1, new 64 MiB layout (rounds 1–2 with an earlier stage order) | 9.5 s | 8.0 s | 7.2 s |
| LTX 2.5, new 64 MiB layout (ten containers just after writing it: 88 s each) | 22.5 s | 9.4 s | 8.8 s |

To reproduce the table, run the example once with the same options, then rounds of 30 cold starts
until the P50 stops falling, and pool the last two rounds. Even on a warm Volume, the P50 of one
round of 30 moves by 1–2 s from round to round.

```sh
modal run examples/modal/wan21.py --cold-start 30   # or h3.py, or ltx25.py
```

Each container is stopped at its first DiT step; the other options select the request as usual.
`COLD_START_REGION=us-west` pins the region the table was measured in (pinning a region costs
extra on Modal). How long warmth lasts depends on the model: Wan's P50 was unchanged after its
files sat unread for about 23 hours (8.1, 8.0 and 7.8 s after about 3, 7 and 23 idle hours), while
H3 FL2VA, which reads twice as much, went from 8.1 to 13.1 s after about 7 idle hours.
[Fast cold start](docs/fast-cold-start.md) explains the rest of the setup: direct reads, small
files, recorded plans and overlapped loading.

Pass your own input with `--prompt` (and `--image` for Wan, `--first-frame` for H3, `--image` for
LTX), and `--gpus 2` to run any of them on two B200s; each model README lists its options, pinned
versions, supported geometry and GPU requirements.

### Compared with SGLang and vLLM-Omni

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/cold-start-wan-dark.svg">
  <img alt="Wan 2.1 cold start to the first DiT step, P50: native-video-inference 12.1 s, vLLM-Omni 0.30.0 140 s, SGLang 0.5.22 188 s" src="docs/images/cold-start-wan-light.svg">
</picture>

The same Wan request (the official example: 832x1104, 81 frames, 40 steps) on one B200, each
framework run with its documented server command and defaults on the official checkpoint:

| | Command | Cold start P50 / P90 | One request, once warm |
| --- | --- | ---: | ---: |
| native-video-inference | `modal run examples/modal/wan21.py` | 12.1 / 17.1 s | 400–406 s |
| vLLM-Omni 0.30.0 | `vllm serve Wan-AI/Wan2.1-I2V-14B-720P-Diffusers --omni` | 140 / 164 s | 429–431 s |
| SGLang 0.5.22 | `sglang serve --model-path Wan-AI/Wan2.1-I2V-14B-720P-Diffusers` | 188 / 205 s | 403–410 s |

The three were measured together on 2026-11-01: three interleaved rounds of 30 fresh single-use
containers each in Modal `us-west`, P50 and P90 pooled over rounds 2 and 3. The clock is the one
above, from the first line of user code to the request's first DiT forward; for SGLang and
vLLM-Omni a forward pre-hook on `WanTransformer3DModel` marks it, ignoring the forwards of their
start-up warm-up. Most of their time is server start: SGLang runs three warm-up requests (about
96 s), and vLLM-Omni compiles the DiT blocks with `torch.compile` and runs one warm-up request. Both
read the Diffusers checkpoint, which stores the DiT and text encoder in FP32 (90 GB in all);
native-video-inference reads the BF16 layout it prepared once from the official checkpoint
(47 GB). vLLM-Omni does not list Wan 2.1 I2V as supported, but serves it. Our P50 varied that day
from 8.0 to 13.8 s over five rounds of 30, against 7.4 s in the table above, measured on 2026-10-31;
the code of both days gave the same P50 when started together.

A warm request is timed inside the container, from receiving the request to the finished video, as
the second of two back-to-back identical requests; both servers then reuse the first request's text
and image conditioning. The warm column was measured again on 2026-11-02 for native-video-inference
and SGLang (vLLM-Omni's is from 2026-11-01). A request's time moves by about 2% between B200 hosts,
more than the gap between the two, so the comparison below runs both in one container: SGLang's
server for two requests, then `wan21-generate --repeat 2` (two requests in one process), in two
alternating rounds per GPU count:

| One container, warm request | native-video-inference | SGLang 0.5.22 |
| --- | ---: | ---: |
| One B200: request | 390.3–390.4 s | 398.2 s |
| One B200: the 40 steps | 387.1–387.2 s | 391.3–391.4 s |
| Two B200s: request | 204.3 s | 209.7–210.7 s |
| Two B200s: the 40 steps | 200.97–200.99 s | 200.91–201.02 s |

The VAE decode takes 3.2 s natively and 5.4–5.8 s with SGLang. With `--num-gpus 2` SGLang also
runs the unconditional pass of classifier-free guidance on the second GPU. On one GPU the steps are
1% faster natively because the unconditional pass starts from the conditional pass's state at
block 0's text cross-attention (both read the same latent, timestep and image up to there); on two,
the GPUs split that shared part by rows instead, and a step takes as long as SGLang's. Either way a
step waits for the slower of the two GPUs, and with both busy the first B200 ran 2–3% slower than the
second in every container we traced. A first request costs natively about 0.5 s more in its steps
(work done once per process; SGLang's server runs three warm-up requests before serving).
native-video-inference reproduces the official pipeline's output byte for byte, on one GPU or two;
SGLang's output differs from it.

LTX 2.5 (T2V, 1472×832, 49 frames) is compared the same way, in one container, in its
[README](models/ltx25/README.md#performance): a warm request's final latents are ready after
1.80–1.81 s on one B200 and 1.80–1.85 s on two, against 2.75–2.83 s and 1.93–1.96 s for SGLang's
text encoding and two stages; with the official diffusion video decoder, a request takes 6.6–7.6 s
natively and 110–115 s with SGLang.

## How it is built

```mermaid
flowchart LR
    Request --> Service[Python service: requests and process lifetime]
    Service --> Model[Model crate: components, graph, kernels]
    Model --> Core[core_infra: loading, integrity, startup executor, residency]
    Core --> GPU[CUDA streams and GPU buffers]
```

- [`core_infra/`](core_infra/README.md) holds model-independent mechanisms: bounded checkpoint
  readers, SHA integrity guards, a startup executor that overlaps reads, CPU preparation and CUDA
  initialization, resident serving and pinned numerical primitives. Core has no model dependency.
- Each `models/<model>/` crate owns its graph, kernels and preparation, and depends only on core.
- [Architecture](docs/architecture.md), [fast cold start](docs/fast-cold-start.md),
  [porting a model](docs/porting-a-model.md) and [building from source](docs/build.md) document
  the protocol.

## Check a change

CPU checks need no GPU, weights or Modal account:

```sh
python -m venv .venv && . .venv/bin/activate
python -m pip install -r requirements-dev.txt
python tools/check_public.py
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Numerical changes need a GPU run against the model's
reference hashes.

## License

Core, our contributions to Wan 2.1 and the tooling are source-available under the
[community license](LICENSE). Commercial and non-commercial use are permitted, but if your
commercial products and services generate **more than USD 10M in yearly revenue**, you must obtain
separate, prior written authorization from Creatify Lab, Inc. before using the software, regardless
of the purpose of your use. Contact **support@creatify.ai**; sending an email alone does not grant
authorization. The threshold is yearly revenue, not ARR. This is not an OSI-approved open-source
license. The model directories keep their upstream terms:
`models/minimax_h3_official` is under the [MiniMax H3 Community License](models/minimax_h3_official/LICENSE),
and `models/ltx25` is under the [LTX-2.x Community License](models/ltx25/LICENSE).

**Using a model requires your own license for it.** This repository grants no rights to MiniMax H3,
Wan 2.1 or LTX-2.x weights or their derivatives. Before running H3, make sure your use is covered by
the MiniMax H3 Community License (including its territory and revenue terms) or a separate agreement
with MiniMax; the same applies to LTX-2.x with Lightricks.

Third-party attributions are in [licenses/NOTICE.md](licenses/NOTICE.md). Model weights are not
distributed here and remain under their own licenses.
