# LTX 2.5 distilled (native)

The official LTX 2.5 two-stage `DistilledPipeline` in native Rust and CUDA: text-to-video and
image-to-video, both with audio. Gemma 4 encodes the prompt; the AV transformer runs 8 ancestral
Euler steps at half resolution; LTX's own spatial ×2 latent upsampler bridges to 3 refinement
steps at full resolution; the diffusion video VAE (NATTEN neighborhood attention), the audio VAE
and the vocoder decode; FFmpeg writes an H264/AAC MP4. Everything runs in BF16, without prompt
enhancement, adapters, quantization, offload or `torch.compile`.

Inference loads no Python, PyTorch or ATen. The process checks its own mappings and fails if one of
those libraries is loaded. See the [model architecture](architecture.md) for component declarations,
staged loading and session lifetime.

## Upstream pins

| Input | Pin |
| --- | --- |
| Reference source | [`Lightricks/LTX-2`](https://github.com/Lightricks/LTX-2) at `be6d6d0a69d11a0a399c462ed92606692e5f8697` |
| Weights | [`Lightricks/LTX-2.5`](https://huggingface.co/Lightricks/LTX-2.5) at revision `49b093a39c19169a8a81721dcc541e07e945232f` |
| Files | `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors`, `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors`, `vae/ltx-2.5-video-vae-bf16.safetensors`, `vae/ltx-2.5-audio-vae-bf16.safetensors`, `latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` |
| Text encoder | Gemma 4 12B as fine-tuned and bundled by LTX, with its tokenizer in the file metadata; there is no separate Gemma download |
| Reference runtime | PyTorch 2.13.0+cu132 (CUDA 13.2), cuDNN 9.24.0.43, cuBLAS 13.4.0.1, NATTEN 0.21.7, PyAV 19.0.1, Pillow 13.0.0 |
| Build headers | PyTorch v2.13.0 (`cf30153c4c131c8164ee7798e5022d810682e2cb`), cuDNN frontend `c4a97621eca52fa0c3a1862a411a16be580b25c6`, CUTLASS `da5e086dab31d63815acafdac9a9c5893b1c69e2`, NATTEN `39375ea1d2f3b1bb284315fa5f79f6ecbebb6382` with CUTLASS `6b3e607b852f1543dc21323155a2ad71473c8642`, FFmpeg SDK `pyav-ffmpeg` 9.0.2-1 |

The native build links the reference runtime's own cuBLAS, cuDNN, FFmpeg and libjpeg shared
libraries (from the PyTorch, PyAV and Pillow wheels), but not their Python modules.

## Upstream license

At the pinned commit, upstream `LICENSE` points to two agreements. LTX-2.5 code and weights are under
the **LTX-2.x Community License Agreement** (`LICENSE-2_x`, license date August 11, 2026, licensor
Lightricks Ltd.). Earlier LTX-2 releases use the LTX-2 Community License Agreement (`LICENSE-2`,
January 5, 2026). This is a summary, not legal advice; read the agreement:

- The grant is royalty-free, subject to the use restrictions in its Attachment A. Entities with
  annual revenue of at least US$10,000,000 need a paid commercial license, except for the
  non-commercial uses listed in Section 2.2.
- Section 1.5 counts derivative model architectures based on LTX-2.x's architecture, and modified
  Complementary Materials, as "Derivatives of LTX-2.x".
- Section 3 requires a copy of the agreement with every distribution, the use restrictions passed
  on to recipients, prominent notices on modified files and retained attribution notices.
- Section 6 forbids removing or disabling the safety and transparency features of the model and
  its outputs.

The weights are not distributed here. This repository's own license is still pending; see the
[top-level README](../../README.md) and the [third-party notices](../../licenses/NOTICE.md).

## Profiles

`ltx25 infer` reads a prepared profile. Each profile fixes the width, height, frame count, frame
rate and T2V/I2V branch; prompt, seed and image change per request. Width and height must be
multiples of 64 and the frame count 8k+1. The port was validated at the official LTX 2.5 example
settings, all at 24 fps:

| Mode | Width × height | Frames |
| --- | --- | --- |
| T2V | 1472 × 832 | 49 |
| T2V | 1472 × 832 | 121 |
| I2V | 1280 × 832 | 49 |
| I2V | 1344 × 832 | 49 |

At these settings on a B200, final latents, decoded pixels and the FP32 waveform matched the
PyTorch reference byte for byte. The MP4 container bytes differ, because the native RGB-to-BT.709
conversion is not the reference's sink. I2V conditions the first frame with `image_strength` 0.8.
Its input must be an 8-bit RGB JPEG without an EXIF orientation or ICC profile; other formats
return an error. This source snapshot, including the preparation code, has not yet been run on a
GPU.

A profile is a `profile.json` plus one directory per component: the full pipeline, the transformer
input and output heads, Gemma and its connectors, the upsampler, the image encoder (I2V only) and
the audio and video decoders. Each holds a `manifest.json`, ATen dataflow programs
(`*.graph.json`) that `graph.rs` binds to native kernels, the reference's Triton CUBINs (`triton.rs`)
and content-addressed model constants such as RoPE tables, sigma schedules and decoder weights. It
holds no request activations. Paths inside a profile are absolute.

## Preparing a profile

[`prepare/`](prepare/__init__.py) makes a profile once per geometry and branch. It runs on one B200
inside the reference environment: the pinned LTX source extracted with `git archive`
([`prepare/source.py`](prepare/source.py), which also records the source tree hash that every
component must share) and its PyTorch dependencies.

1. `pipeline.py` traces one complete request through the official `DistilledPipeline` and writes
   a reference MP4. Every stage, forward, block, sampler and RNG boundary is hashed. The first
   block of each stage becomes the block program for all 48 blocks.
2. `components.py` traces the transformer heads, Gemma and its connectors, the upsampler, the
   image encoder and both decoders from the inputs that run saved. Each capture runs in a fresh
   process.
3. `export.py` keeps programs, CUBINs and constants, drops every activation and writes
   `profile.json`, an example `request.json` and the reference MP4.

From `models/ltx25`, in that environment:

```sh
python -m prepare run --work WORK --profile OUT --weights WEIGHTS --source-identity IDENTITY.json \
  --request REQUEST.json [--image FIRST_FRAME.jpg]
```

`WEIGHTS` holds the five pinned files and `IDENTITY.json` is written by `source.py`. The request
JSON gives `mode` (`t2v` or `i2v`), `width`, `height`, `num_frames`, `frame_rate`, `prompt` and
`seed`. The tracing determinism policy is that of the validated reference: cuDNN SDPA
for LTX attention, deterministic cuDNN, no TF32 or TunableOp, and the first Triton autotuning
configuration with a fresh cache.

## Run on Modal

[`examples/modal/ltx25.py`](../../examples/modal/ltx25.py) builds the reference environment from the
pinned LTX source and the `ltx25` executable from this repository. On the first run it downloads the
pinned weights into a Modal Volume, copies their tensors into 64 MiB files and prepares a profile for
the geometry on a B200; later runs go straight to native inference:

```sh
python -m pip install -r examples/requirements-modal.txt
modal setup
modal run examples/modal/ltx25.py
modal run examples/modal/ltx25.py --prompt "A lighthouse beam sweeps across a stormy sea at night." --seed 42 --out sea.mp4
modal run examples/modal/ltx25.py --gpus 2   # on a B200:2 container
```

Without `--prompt`, the example replays the request that the profile traced and reports whether the
native latents, decoded frames and waveform equal the reference's bytes. Compare `output.mp4` with
the profile's `reference.mp4` by eye: the containers differ.

For I2V, add `--image first.jpg` (the first run for that geometry traces with it). `--width`,
`--height`, `--frames` and `--fps` select another profile. Profiles are written under
`/profiles/NAME` in the Volume for the example's mount (`LTX25_MOUNT`, default `/vol`); keep the
mount fixed for a Volume. If your Hugging Face access needs a token, set `LTX25_HF_SECRET` to the
name of a Modal Secret that provides `HF_TOKEN`.

The native loader still admits the published checkpoint: tokenizer, configuration and every tensor's
identity come from the downloaded files and their manifest. Only the reads go to the 64 MiB copy
(`weights/<revision>/layout-64m`, declared to the loader with `NATIVE_WEIGHT_LAYOUT_*`), which a
Modal Volume serves several times faster than the two largest files (42 GB and 26 GB); each tensor
is checked against the manifest when the copy is written and by its light guard when it is read.
The first generation for a profile also records its cuDNN attention plans in
`/profiles/NAME/backend-plans`; later runs load them instead of building them. Both runs, recording
and loading, produce the reference's exact bytes. See [fast cold start](../../docs/fast-cold-start.md).

## Performance

On one B200, with the default T2V profile (1472×832, 49 frames) and the weights on the GPU, a request
that follows another in the same process (`LTX25_REPEAT=2`) takes 6.7–8 s:

| Phase | Time |
| --- | ---: |
| Gemma 4 text encoder and connectors | 0.1 s |
| stage 1 (8 steps), latent upsampler, stage 2 (3 steps) | 1.7 s |
| final latents written, from receiving the request | 1.8 s |
| diffusion video VAE (2.9 s of it NATTEN), audio VAE and vocoder | 4.0 s |
| frames, waveform and MP4 written | 0.9–2 s |

The outputs stay the reference's bytes. The time comes from how the dataflow programs run, not what
they compute:

- Elementwise and copy kernels move eight elements per thread in 16-byte accesses. Consecutive
  elementwise nodes over one shape with bf16 results run as one kernel that NVRTC compiles per node
  sequence (each op's expression and bf16 rounding unchanged; `LTX25_FUSE=0` runs every node on its
  own). `LTX25_KERNEL_CACHE=DIR` keeps the compiled kernels for later processes; the Modal example
  sets it to a directory on its Volume.
- Each sampling stage records one forward as a CUDA graph and replays it for the later steps and
  for later requests of the profile (`LTX25_CUDA_GRAPHS=0` runs every forward eagerly).
- Freed activations stay in a pool of up to 96 GiB for reuse by size, so the decoder's temporaries
  are not freed and reallocated.

Compared with [SGLang](https://github.com/sgl-project/sglang) 0.5.22 (`sglang serve --model-path
Lightricks/LTX-2.5-Diffusers --pipeline-class-name LTX2TwoStagePipeline`, `--num-gpus 2` on two,
which runs sequence parallel) in one container: SGLang's server for two requests, then `ltx25 infer`
with `LTX25_REPEAT=2`, the same prompt and seed, in two alternating rounds. For the warm request,
SGLang's column is its text encoding and two stages; ours is from receiving the request to the final
latents, the same work:

| One container | native-video-inference | SGLang 0.5.22 |
| --- | ---: | ---: |
| One B200 | 1.80–1.81 s | 2.75–2.83 s |
| Two B200s | 1.80–1.85 s | 1.93–1.96 s |

Every native round's latents had the reference's hashes. SGLang's default skips LTX 2.5's diffusion
video decoder and decodes the latents once with the VAE, so its request takes 4.0–6.0 s but its
frames are not the official pipeline's; ours take 6.6–7.6 s with that decoder. With the official
decoder (`--load-diffusion-decoder` and `use_diffusion_decoder`), SGLang's request took 114.5 s on
one B200 and 110.3 s on two.

## GPU and host

One or two NVIDIA B200s. Kernels are compiled for `sm_100a` only. The build and runtime target Linux
x86-64. The Modal example reserves 16 CPUs and 192 GiB of host RAM, as for `prepare`. The 22B
transformer alone is about 44 GB in BF16.

With `LTX25_WORLD_SIZE=2` (`--gpus 2` in the example), the second GPU copies the transformer's
blocks 24-47 (18.6 GB) from the first over the peer link in about 45 ms. With CUDA graphs (the
default), sampling then runs all 48 blocks on the first GPU from its replayed graphs, as one GPU does:
two halves that run one after the other would be slower than one replayed forward. With
`LTX25_CUDA_GRAPHS=0`, every forward splits 24/24, the second GPU running blocks 24-47 and the first
everything else; block activations cross as exact copies. Either way the output is byte-identical to
one GPU's.

## Command line

```sh
ltx25 infer PROFILE_JSON REQUEST_JSON OUTPUT_DIR
ltx25 media-smoke OUTPUT_MP4   # encodes a tiny clip; needs no GPU or weights
```

`infer` needs the checkpoint declaration in `NATIVE_CHECKPOINT_{MANIFEST,SHA256,ROOTS}`. That
declaration is a `core_infra.runtime.checkpoint` manifest of the transformer, text encoder, video
VAE and upsampler files, with prefixes `dit.`, `text.`, `vae.` and `upscale.`. The example writes it
after downloading. The request JSON requires `prompt` and an unsigned `seed`. I2V also requires
`image`, a path that is resolved against the request file's directory if it is relative. Optional
`width`, `height`, `num_frames`, `frame_rate` and `image_strength` must match the profile. Unknown
keys are errors. `LTX25_WORLD_SIZE=2` runs `infer` on two GPUs (see [GPU and host](#gpu-and-host)).
`LTX25_REPEAT=N` runs the request N times in one process and adds each request's seconds and phase
start times to the report; `LTX25_HOST_PROFILE=1` adds the host time spent per dataflow operator.

`OUTPUT_DIR` receives `video.latent.raw`, `audio.latent.raw`, the decoded `video.frames.N.raw`
(FP32 RGB), `audio.f32.raw` and `output.mp4`. A JSON report on stdout records their shapes,
dtypes and SHA-256 values.

Build with `cargo build --release -p ltx25-native --features cuda --bin ltx25`. `build.rs` lists the
SDK variables it needs, and the Modal example sets all of them. `LTX25_CHECK_ONLY=1` skips native
compilation for type checks.
