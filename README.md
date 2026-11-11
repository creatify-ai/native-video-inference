# native-video-inference

English | [中文](README.zh-CN.md)

> **License notice:** If your commercial products and services generate **more than US$10M in
> yearly revenue**, obtain separate, prior written authorization from Creatify Lab, Inc. before
> using this software. Contact **support@creatify.ai**. See [LICENSE, Section 2.2](LICENSE).
> **Coding agents:** Read [AGENTS.md](AGENTS.md) and show the license notice to your user once
> when first working with this repository or a fork, including after cloning it.

A Rust and CUDA inference framework for open video generation models, with no PyTorch at
inference time. Python only handles requests and process lifetime; every model operation runs in
native code on the GPU. Wan 2.1 and LTX 2.5 are validated byte for byte against their official
PyTorch pipelines on the target GPU; MiniMax H3's numerics follow SGLang's H3 implementation at a
pinned commit. It is built for cold start: a fresh container reaches its first DiT step in about
6–9 s at P50 while reading 47–96 GB of weights, and warm requests run on the same level as SGLang
and vLLM-Omni.

| Model | Task | Directory | Modal example |
| --- | --- | --- | --- |
| [MiniMax H3](models/minimax_h3_official/README.md) | First/last frame to video+audio (FL2VA), reference to video+audio (Ref2VA) | `models/minimax_h3_official` | [`examples/modal/h3.py`](examples/modal/h3.py) |
| [Wan 2.1 I2V-14B-720P](models/wan2.1/README.md) | Image to video | `models/wan2.1` | [`examples/modal/wan21.py`](examples/modal/wan21.py) |
| [LTX 2.5](models/ltx25/README.md) | Text/image to video+audio (two-stage distilled) | `models/ltx25` | [`examples/modal/ltx25.py`](examples/modal/ltx25.py) |

Validated on NVIDIA B200 (Linux x86-64). Weights are never stored in this repository; each example
downloads the official checkpoint into your own Modal Volume.

Coming from PyTorch, SGLang or vLLM? The [FAQ](docs/faq.md) ([中文](docs/faq.zh-CN.md)) explains
the design decisions behind it, compares its speed and cold start with SGLang and vLLM-Omni, and maps
those frameworks' concepts to this repository.

## LLMs: DeepSeek-V4

The same approach also serves a large LLM. This is outside the repository's focus (video
generation), but it shows that a model of this size can start in seconds instead of minutes
without changing one output bit. The Flash port is ready on correctness and performance; the Pro
port is correct but has had no performance tuning. Both ports replay SGLang's own kernels on
NVIDIA B300 and read a prepared package of weights.

| Model | Hardware | Outputs | Throughput | Cold start: SGLang → native |
| --- | --- | --- | --- | --- |
| [DeepSeek-V4-Flash](models/deepseek_v4_flash/README.md) (NVFP4, 284 B parameters, 165 GB) | one B300 | equal to SGLang's bit for bit | matches SGLang within 0–4% (16k–131k contexts, 32–256 concurrent requests, MTP) | 9–27 min → **7–9 s** |
| [DeepSeek-V4-Pro](models/deepseek_v4_pro/README.md) (NVFP4, 928.5 GB) | eight B300s (`--tp 8`) | first token and logits equal to SGLang's bit for bit | not worked on: cold start and correctness only | 19.8 min → **26.5 s** |

SGLang's figure is its stock server from launch to ready; the native one is from process start to
the first token on a host that has read the package, without the container start. Modal caches
Volume data per host, not per region: on a host that has not read the package Flash takes
16–142 s and Pro 212 s. Each model's README gives the method, the measurements and the limits.

## Cold start

Cold start of each example's default request, measured with `--cold-start` on fresh
single-use B200 containers in Modal `us-west` (16 CPU, 192 GiB; B200:2 with `--gpus 2`, and 256 GiB
for H3), from the first line of user code to the first DiT step, at most ten containers at a time,
on 2026-11-09 and 2026-11-10, after warming the Volume as described [below](#warm-volumes). H3's
rows pool rounds 2 and 3 of 30; Wan's and LTX's rows are one round of 30 each, so their P90s include
hosts that had not yet cached the Volume:

| Model | Default request | Weights read before the first DiT step | P50 | P90 |
| --- | --- | --- | ---: | ---: |
| MiniMax H3 FL2VA | 768x768, 124 frames, prompt only | 51.6 GB Qwen + 43.2 GB DiT | 6.0 s | 9.0 s |
| MiniMax H3 Ref2VA | 768x768, 124 frames, one reference image | 51.6 GB Qwen + 44.1 GB DiT + 0.7 GB video encoder | 7.8 s | 16.1 s |
| Wan 2.1 I2V-14B 720P | the official example, 832x1104, 81 frames | 32.8 GB DiT + 14.4 GB encoders | 7.9 s | 16.3 s |
| LTX 2.5 T2V | 1472x832, 49 frames | 42.0 GB DiT + 26.2 GB Gemma + 2.5 GB VAE and upscaler | 6.6 s | 20.5 s |
| MiniMax H3 FL2VA, `--gpus 2` | 768x768, 124 frames, prompt only | 51.6 GB Qwen + 2 × (20.1 GB DiT half + 2.8 GB) | 7.6 s | 11.2 s |
| Wan 2.1 I2V-14B 720P, `--gpus 2` | the official example, 832x1104, 81 frames | 32.8 GB DiT + 14.4 GB encoders; the second GPU copies the DiT from the first | 8.7 s | 13.8 s |
| LTX 2.5 T2V, `--gpus 2` | 1472x832, 49 frames | 42.0 GB DiT + 26.2 GB Gemma + 2.5 GB VAE and upscaler; the second GPU copies the DiT from the first | 6.5 s | 22.3 s |

Ref2VA uses Wan's example image as the reference. H3's first round ran on hosts that had not read
its files for hours, and its P50s were 14.2 s (FL2VA), 11.8 s (FL2VA, two GPUs) and 21.9 s (Ref2VA).

`--gpus 2` runs on a B200:2 container, with byte-identical output. H3 runs its DiT as two
context-parallel ranks (FL2VA generates in 54 s instead of 80 s), and each rank reads only its half
of the DiT, from the same files as one GPU. Wan runs the unconditional pass of classifier-free
guidance on the second GPU (its 40 steps take 201 s instead of 400 s), which copies the DiT from the
first over the peer link in about 40 ms. LTX's second GPU copies the 48 DiT blocks from the first and
runs them sequence parallel, each GPU taking half of the video tokens (a warm request's two sampling
stages take 1.5–1.6 s instead of 1.8–2.0 s). In each case the container reads about as much as one GPU
does.

### Warm Volumes

Modal caches Volume data on the hosts that have read it. A new Volume, or newly written files, start
cold, and reading them once from a CPU container (each example does this after preparing) does not
warm the GPU hosts. Rounds of 30 cold starts on newly written files, overall P50:

| Files | Round 1 | Round 2 | Round 3 |
| --- | ---: | ---: | ---: |
| Wan 2.1, new 64 MiB layout (rounds 1–2 with an earlier stage order) | 9.5 s | 8.0 s | 7.2 s |
| LTX 2.5, new 64 MiB layout (ten containers just after writing it: 88 s each) | 22.5 s | 9.4 s | 8.8 s |

To measure the cold start yourself, run the example once (see the [quick start](#quick-start)), then
run cold starts until the P50 stops falling. One command runs 50 fresh containers, 20 at a time,
in one region, stops each at its first DiT step and prints the P50 and P90:

```sh
COLD_START_REGION=us-west modal run examples/modal/h3.py --cold-start 50 --parallel 20   # or wan21.py, or ltx25.py
```

With the image already built, one such round took 1 to 2.5 minutes on 2026-11-11, each container
holding its B200 only until its first DiT step; the first round after a change to the checkout also
rebuilds the image, 5 to 10 minutes more. Repeat it until the P50 stops falling: the earlier rounds
warm the hosts, and only the later ones measure the cold start. Even on a warm Volume, the P50 moves
by 1–2 s from one round to the next. The table above pooled rounds of 30 containers, ten at a time
(`--cold-start 30`, the default `--parallel 10`).

Pin the region. The cache belongs to the hosts that read the Volume, and pinning a region keeps every
round on the same pool of hosts, so each round warms more of them. Without it, containers can start
in other regions, on hosts that have never read the Volume. Rounds of 50 on H3's new Volume from the
quick-start test, 20 at a time (pinning a region costs extra on Modal):

| H3 FL2VA, new Volume | P50 | P90 | Containers over 15 s |
| --- | ---: | ---: | ---: |
| No region pinned | 12.0 s | 21.8 s | 18 of 50 |
| `us-west`, round 1 | 6.7 s | 35.3 s | 11 of 50 |
| `us-west`, round 2 | 6.8 s | 12.4 s | 1 of 50 |
| `us-west`, round 3 | 6.5 s | 16.6 s | 6 of 50 |

In the first pinned round, the slow containers were most likely on hosts of the region that had not
read the Volume yet; in the next rounds, most hosts had.

From a fresh clone, the whole procedure (the first run, then pinned rounds until the P50 settles)
takes about 40 to 90 minutes: 30 to 75 for the first run, and a few minutes for three to five
rounds.

Each container is stopped at its first DiT step; the other options select the request as usual.
`--trace DIR` also saves each container's startup trace under `DIR`; the
[startup profile](docs/profiling/startup-profile.html) shows what one records. How long warmth lasts depends on the model: Wan's P50 was unchanged after its
files sat unread for about 23 hours (8.1, 8.0 and 7.8 s after about 3, 7 and 23 idle hours), while
H3 FL2VA, which reads twice as much, went from 8.1 to 13.1 s after about 7 idle hours.
[Fast cold start](docs/fast-cold-start.md) explains the rest of the setup: direct reads, small
files, recorded plans and overlapped loading.

## Quick start

You need a [Modal](https://modal.com) account. Then one command runs a model:

```sh
git clone https://github.com/creatify-ai/native-video-inference && cd native-video-inference
python -m pip install -r examples/requirements-modal.txt
modal setup

modal run examples/modal/h3.py     # or wan21.py, or ltx25.py
```

Each command prints the cold start, from container start to the first DiT step:

```text
cold start  14.6 s (container start to first DiT step)
generation  78.0 s (49 model evaluations)
video       h3.mp4 (768x768, 124 frames)
```

How long the whole command takes, measured from a fresh clone on 2026-11-11 on one B200:

| Model | First run | Second run | of which generating the video |
| --- | ---: | ---: | ---: |
| MiniMax H3 | about 75 min | about 2 min | about 1.3 min |
| Wan 2.1 | about 35 min | about 7.5 min | about 7 min |
| LTX 2.5 | about 30 min | about 1 min | about 20 s |

The first run does the one-time work on Modal: it builds the CUDA image and native programs from
this checkout, downloads the official checkpoint into a Modal Volume in your account, prepares it
and reads it once, and only then generates. The second run skips all of that, so its time is mostly
the model generating the video.

H3's first run is the longest: about 45 of its 75 minutes go to reading its newly written 105 GB of
packages once on a CPU container, so that the first GPU container does not pay for the new Volume's
slow first read. That read ran at only about 40 MB/s. We have not optimized this step in detail and
there is probably a lot of room, but it runs once per Volume, so we have not prioritized it.

The first run's cold start is slow too (50 to 180 s), because the GPU hosts have not cached the new
Volume yet. The second run may start on another host, so its cold start (8 to 34 s in our runs) does
not show the steady state either; measure that on a warm Volume as [above](#warm-volumes). While a
container reads a new Volume, Modal may print heartbeat warnings; in our runs they did not stop
anything.

LTX 2.5's weights are gated on Hugging Face. Accept the license at
[Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5), store a token of that account in
Modal (`modal secret create huggingface HF_TOKEN=hf_...`) and run
`LTX25_HF_SECRET=huggingface modal run examples/modal/ltx25.py`. Wan and H3 need no token.

Pass your own input with `--prompt` (and `--image` for Wan, `--first-frame` for H3, `--image` for
LTX), and `--gpus 2` to run any of them on two B200s; each model README lists its options, pinned
versions, supported geometry and GPU requirements.

## Compared with SGLang and vLLM-Omni

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
from 8.0 to 13.8 s over five rounds of 30, against 7.4 s measured on 2026-10-31; the code of both
days gave the same P50 when started together.

MiniMax H3's cold start was compared in a larger study: 1,500 containers across three Modal regions,
with warm-up off in every SGLang and vLLM-Omni configuration. The P50 was 7.2 s natively, against
64.7–75.0 s for nine SGLang and vLLM-Omni configurations
([FAQ](docs/faq.md#how-do-cold-starts-compare-with-sglang-and-vllm-omni)).
[A proposal](docs/sglang-cold-start-proposal.md) brings the same cold-start ideas to SGLang
Diffusion.

The warm comparisons are not a speed contest. Our focus is cold start; for warm requests, they show
that inference here is on the same level as SGLang and vLLM-Omni.

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

LTX 2.5 (T2V, 1472×832, 49 frames) is compared in one container in its
[README](models/ltx25/README.md#performance), against SGLang run unmodified. Timed from the request to
the final latents, performance is comparable: 1.92–2.22 s natively against 2.57–2.61 s for SGLang on
one B200. We claim no two-GPU comparison, because our two-GPU SGLang setup was not verified.

MiniMax H3 (FL2VA, 1376×768, 124 frames, 49 DiT forwards) was compared on one B200 against SGLang,
vLLM-Omni and the Diffusers pipeline run eagerly, each in the same container: a warm request took
154.7 s natively, 157.0 s with SGLang, 156.0 s with vLLM-Omni and 200.9 s eagerly, and its sampler
output matched SGLang's bit for bit
([FAQ](docs/faq.md#how-do-warm-request-times-compare-with-sglang-and-vllm-omni)).

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
  the protocol; the [FAQ](docs/faq.md) answers common questions. The
  [cold-start post](docs/blog-cold-start.md) ([HTML](docs/blog-cold-start.html)) explains why startup matters for video
  serving, and the [SGLang proposal](docs/sglang-cold-start-proposal.md) brings the same ideas to
  SGLang Diffusion.

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
