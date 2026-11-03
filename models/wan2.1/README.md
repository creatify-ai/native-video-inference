# Wan 2.1 I2V-14B-720P

The official Wan 2.1 image-to-video pipeline in native Rust and CUDA. Inference runs no Python,
PyTorch or ATen. This crate is the worked example for [porting a model](../../docs/porting-a-model.md);
its [architecture](architecture.md) describes the component declarations, loading and serving.

## Parity

On a B200 with the SDKs below, the native pipeline reproduces the reference, the official PyTorch
pipeline with the settings under [Pins](#pins), bit for bit. On ten validation requests covering seven output sizes at 81 frames, all 499 recorded tensors
have the same SHA-256 as the reference: the initial noise, every DiT forward's inputs and outputs,
every guidance and UniPC step and the final latent. The decoded video is identical too, and
repeated runs are identical. The validation inputs and the reference recorder are not part of this
repository. MP4 bytes also depend on your `ffmpeg` build; the claim covers the decoded frames.

Warm, on one B200, the official example's request (832x1104, 81 frames, 40 steps) takes 390–401 s
natively, depending on the host: 387–398 s for the 80 DiT forwards, 3.2 s for the VAE decode. On two
B200s it takes 204 s, with the unconditional pass of classifier-free guidance on the second GPU
(40 steps in 201 s). A
server reuses the last request's text, image and VAE conditioning when the next request repeats its
prompts, image and size (2–3 s otherwise). Speed comes from the access pattern, never the arithmetic:
vectorized and fused elementwise, normalization and RoPE kernels that keep every per-element
expression and reduction order (the RMSNorm of q and k sums its squares in a warp per row, in the
lanes and order of the reference's reduction, then rotates the same registers), and a bf16 GELU read
from a table of all 65536 inputs computed with
the same expression. The DiT attention runs the cuDNN engine the reference runs; cuDNN's only other
plan for that shape with the same bits is no faster. Cold-start measurements are in [fast cold
start](../../docs/fast-cold-start.md).

## Pins

| Item | Value |
| --- | --- |
| Reference | [`Wan-Video/Wan2.1`](https://github.com/Wan-Video/Wan2.1) at `9737cba9c1c3c4d04b33fcad41c111989865d315`, FlashAttention replaced by PyTorch SDPA on the cuDNN backend; `cudnn.deterministic`, `MKL_CBWR=COMPATIBLE` |
| Checkpoint | [`Wan-AI/Wan2.1-I2V-14B-720P`](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P) at `8823af45fcc58a8aa999a54b04be9abc7d2aac98` |
| Precision | DiT BF16 (the reference converts the published FP32 with `.to(torch.bfloat16)`); T5 BF16, CLIP FP16 and VAE FP32, the official defaults |
| Sampling | UniPC, 40 steps, shift 5, guidance scale 5, 81 frames, `max_area` 1280×720 |
| SDKs | CUDA 13.0 (nvcc 13.0.88), cuBLAS 13.1.1.3, cuDNN 9.20.0.48, cuDNN frontend `c4a97621`, PyTorch 2.13.0 headers (`cf30153c`), FlashInfer 0.6.15.post1's CUTLASS headers, Rust 1.90.0 |

## Build

On Linux x86-64, install the SDKs from the [build guide](../../docs/build.md) and point the
variables at them. The PyTorch and CUTLASS headers are compile-time inputs only.

```sh
export CUDA_HOME=/usr/local/cuda H3_CUBLAS_ROOT=... H3_CUDNN_ROOT=... H3_CUDNN_FRONTEND=... \
       H3_TORCH_INCLUDE=... H3_CUTLASS_ROOT=... H3_VENDOR_INCLUDES=...
cargo build --locked --release -p wan21-native --features cuda --bin wan21-generate --bin wan21-server
```

`build_native` in the [Modal example](../../examples/modal/wan21.py) does the same from pip wheels.
At run time, the cuBLAS and cuDNN libraries must be on the loader path and `ffmpeg` must be on
`PATH`. For type checks without SDKs, set `WAN_CHECK_ONLY=1`.

## Prepare the checkpoint

Download the checkpoint, then write the prepared directory once. The published files are never
modified. [`prepare.py`](prepare.py) writes every weight serving reads in 64 MiB safetensors shards
(the DiT as BF16, T5, CLIP and the VAE byte for byte), the float64 RoPE table and `manifest.json`,
which records every tensor's byte range and SHA-256. Many small files read several times faster
than a few large ones from network storage. It needs numpy and, for the RoPE table, PyTorch (CPU)
and safetensors.

```sh
hf download Wan-AI/Wan2.1-I2V-14B-720P --revision 8823af45fcc58a8aa999a54b04be9abc7d2aac98 \
    --local-dir <checkpoint>
python models/wan2.1/prepare.py --checkpoint <checkpoint> --out <prepared>
```

With torch 2.13.0 and safetensors 0.5.3, the manifest's SHA-256 should be
`0261458d7d20ca90ad744df2adddf89afa888566528e3db4d290573758bb5721`. The tensors are the ones behind
the parity results; only the file layout changed from the earlier reference preparation
(`4c9ce49d...`). `prepare-report.json` records your digest. The programs take it as an offline pin
and refuse to load weights when the manifest differs from it.

## Run

One video:

```sh
WAN21_MANIFEST_SHA256=<manifest SHA-256> wan21-generate --weights <prepared> --checkpoint <checkpoint> \
    --image in.png --prompt "A cat surfs a wave at sunset." --out out.mp4
```

`--gpus 2` runs the unconditional pass on a second GPU. `--repeat N` serves the request N times in
one process, as the server would: later requests reuse the loaded models and the first request's
conditioning.

`wan21-server serve <socket>` is the resident process that `core_infra.runtime.service.NativeService`
owns. It reads `WAN21_PREPARED` and `WAN21_CHECKPOINT`, binds its socket without touching the GPU,
streams the weights on the first `["generate", "<request json>"]` and keeps the models resident.
`WAN21_REPORT_SHA=1` adds the decoded video's SHA-256 to each reply, and `WAN21_WORLD_SIZE=2` runs it
on two GPUs (see [Limits](#limits)).

On Modal, the [example](../../examples/modal/wan21.py) builds the image, downloads and prepares the
checkpoint in a Modal Volume on the first run, and runs one request on a B200:

```sh
python -m pip install -r examples/requirements-modal.txt
modal setup
modal run examples/modal/wan21.py        # the official example image and prompt
modal run examples/modal/wan21.py --image in.png --prompt "A cat surfs a wave at sunset." --out out.mp4
modal run examples/modal/wan21.py --gpus 2   # on a B200:2 container
```

## Request parameters

| Key | Default | Notes |
| --- | --- | --- |
| `prompt` | required | |
| `negative_prompt` | official prompt | Empty means the official Chinese negative prompt |
| `seed` | required by the server, 42 in `wan21-generate` | Non-negative integer; random seeds are not supported |
| `sample_steps`, `shift` | 40, 5.0 | Only schedules with recorded UniPC tables: 40 steps at shift 5.0 or 3.0 |
| `guide_scale` | 5.0 | |
| `frame_num` | 81 | Must be 4n+1 |
| `max_area` | 921600 | Output area bound (1280×720), as in the official size math |
| `sample_solver` | `unipc` | Other solvers are rejected |

## Limits

- **GPU.** One or two B200s (`sm_100a`); the build and the device check reject other
  architectures. With two (`--gpus 2` in the Modal example, `WAN21_WORLD_SIZE=2` for
  `wan21-server`), the second GPU runs every unconditional DiT pass (`cfg_parallel.rs`) and copies
  the DiT weights from the first over the peer link instead of reading them. The official example's
  decoded video is identical to one GPU's; the ten-request parity record above is from one GPU.
- **Shared work between the two CFG passes.** Both passes of a step read the same latent, timestep,
  image and CLIP features, so they agree up to block 0's text cross-attention. On one GPU the
  unconditional pass starts from the conditional pass's state at that point (`dit::Prefix`). On
  two, the GPUs split block 0's self-attention by query rows and exchange their halves over the
  peer link (`dit::AttnSplit`); cuDNN computes every row of a range with the whole sequence's plan,
  so the bytes do not change.
  There is no attention sparsity, step caching or quantization.
- **Resolution.** The output keeps the input's aspect ratio within `max_area`, and both sides are
  multiples of 16. Parity was recorded at 1280×720, 720×1280, 1264×720, 720×1264, 960×960,
  1280×704 and 704×1280. Smaller areas run but were not validated.
- **Frames.** 4n+1 frames at 16 fps. Parity was recorded at 81 frames only.
- **Input.** PNG only. A PNG that carries EXIF is rejected, because the official
  `ImageOps.exif_transpose` step is not ported; store the image already transposed.
- **Memory.** The Modal example reserves 16 CPUs and 192 GiB of host RAM for streaming loads.

## Layout

| Path | Contents |
| --- | --- |
| `frontend.rs` | Request parsing, PNG decoding, size math, PyTorch's CPU bicubic resize, prompt cleaning and umt5 tokenization |
| `t5.rs`, `clip.rs`, `vae.rs`, `dit.rs` | The four models |
| `sampler.rs`, `schedules.rs` | `FlowUniPCMultistepScheduler` and its MKL-evaluated scalars as tables |
| `pipeline.rs` | `condition`, `denoise`, `decode`, `generate`, `cold_start` and MP4 writing |
| `startup.rs`, `admission.rs` | Declared cold startup and trusted checkpoint admission |
| `weights.rs`, `preload_contract.rs` | Weight sources and the component inventory |
| `serving.rs`, `backend.rs` | The standard native backend ABI |
| `cfg_parallel.rs` | Two GPUs: the second runs the unconditional pass with a peer copy of the DiT |
| `cuda.rs`, `linear.rs`, `conv.rs`, `sdpa.rs`, `eager.rs`, `rng.rs`, `encoders.rs`, `kernels/` | Device buffers, operator bindings and CUDA kernels |
| `gold.rs`, `bin/wan21-playground.rs` | Gold reader and parity rungs; they need golds recorded from the reference |
| `bin/wan21-generate.rs`, `bin/wan21-server.rs` | One-shot program and resident server |
| `prepare.py`, `tests/` | Checkpoint preparation and CPU tests |
