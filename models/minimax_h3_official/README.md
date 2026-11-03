# minimax_h3_official

Native Rust and CUDA inference for the official
[MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) FL2VA and Ref2VA checkpoints.
Python validates requests and owns the native processes. The text encoder, transformer,
video and audio VAEs, sampling and token layout run in native executables. Inference loads
no PyTorch. The model builds on the shared [core infrastructure](../../core_infra/README.md).

**License.** This directory is a derivative of MiniMax H3 and is distributed under the
[MiniMax H3 Community License Agreement](LICENSE) (see [NOTICE](NOTICE)). To use it you need your
own rights to MiniMax H3: either use that falls within the Community License, including its
Applicable Territory and revenue terms, or a separate license from MiniMax (api@minimax.io).
This repository does not grant any H3 rights.

## Upstream pins

| Input | Version |
| --- | --- |
| Checkpoint | `MiniMaxAI/MiniMax-H3` revision `42ed227ee7df40d41602854ae760620d6eb651fe` (`FL2VA/`, `Ref2VA/`) |
| CUDA | 13.0 toolkit; wheels `nvidia-cublas==13.1.1.3`, `nvidia-cudnn-cu13==9.20.0.48`, `nvidia-nccl-cu13==2.29.7`, `nvidia-cuda-nvrtc==13.0.88` |
| cuDNN frontend | commit `c4a97621eca52fa0c3a1862a411a16be580b25c6`, headers only |
| PyTorch | 2.13.0 (`cf30153c4c131c8164ee7798e5022d810682e2cb`): build-time headers, and one-time preparation |
| FlashInfer | 0.6.15.post1 CUTLASS headers; 0.7.0.post1 with `nvidia-cutlass-dsl==4.8.0` for the SubBlock kernel export |
| Rust | 1.90.0 |

`FL2VA/` and `Ref2VA/` keep the official component configuration files, without weights.

## Build

On Linux x86-64, set the SDK variables from the [build guide](../../docs/build.md), then:

```sh
python models/minimax_h3_official/prepare_subblock_aot.py export /opt/sdk/subblock   # once, on a B200
H3_CUDA_ARCH=100a H3_SUBBLOCK_AOT_ROOT=/opt/sdk/subblock python models/minimax_h3_official/build.py
```

The export compiles FlashInfer's unmodified SM100 block-sparse attention kernel. The build
rejects an export whose sources, header, runtime library or GPU payload sections differ from
`subblock-v07-source.json` (the object's host launcher code varies with the build CPU).
Executables (`h3-official-*`) go to `target/release/`, shared libraries to
`target/native/minimax_h3_official/`.

## Run on Modal

[examples/modal/h3.py](../../examples/modal/h3.py) builds the image (SDKs, Rust, the SubBlock
export on a B200, then `build.py`), downloads the checkpoint into a Modal Volume you name,
prepares the native packages on one B200, writes `native.json` and runs one request:

```sh
python -m pip install -r examples/requirements-modal.txt
modal setup
modal run examples/modal/h3.py
modal run examples/modal/h3.py --prompt "A sailboat crosses a quiet lake." --first-frame first.png
```

## Prepared packages

Inference reads native packages, not the safetensors checkpoint. `prepare/` builds them once
from the official Diffusers folders: the Qwen conditioner, the DiT (official parameters plus
49-step timestep tables, Q/K/V packed by `prepare_qkv.py`), the router CUBINs, and the video
and audio VAE programs (plus the reference encoders for Ref2VA). See
[prepared artifacts](../../docs/prepared-artifacts.md) for the layout and `native.json`. The Qwen
conditioner is the same for both workflows: with `prepare run --shared DIR` it is written once to
`DIR/conditioner` and linked from each workflow's directory, so a second workflow neither rebuilds
nor stores its 52 GB again, and reads files a Volume already serves warm.

The last step hashes every prepared payload in full and writes `light-sha.json`, the first and last
KiB of each one. `native.json` pins that table (`runtime.light_sha`), and serving checks those
guards as the weights stream in rather than hashing about 95 GB before the first DiT step. Qwen
builds the first request's attention plans while its weights load (`H3_QWEN_STARTUP_HINT_FILE`).

From Python, with a prepared `native.json`:

```python
from models.minimax_h3_official.native_pipeline import MiniMaxH3Pipeline

pipeline = MiniMaxH3Pipeline.from_prepared("prepared/native.json")
pipeline(prompt="A sailboat crosses a quiet lake.", first_frame="first.png",
         width=768, height=768, num_frames=124, seed=0, output_dir="outputs/example")
```

## Two GPUs

`modal run examples/modal/h3.py --gpus 2` runs the DiT as two context-parallel ranks, one per GPU,
under `h3-official-cp-coordinator`. Each rank keeps an equal byte range of every layer's packed
weights and gathers the rest from its peer layer by layer as it runs (`transformer/sharded_native.rs`).
`prepare cp --world 2` indexes those ranges once, on CPU: `dit-cp2/manifest.json` is the per-layer
layout, `dit-cp2/slices.json` says where each rank's bytes lie in `dit/`'s files, and
`light-sha-cp2.json` adds a guard for each slice. No weight bytes are copied: each rank reads its
half of the same files, which a Volume already serves warm to one-GPU runs. Qwen, the encoders and
the decoders run on the first GPU, as with one. `prepare config --world 2` writes `native-cp2.json`.

## Limits

- GPU: one or two B200s (`sm_100a`). The router kernels and the SubBlock export are B200-only.
- Canvas: width and height are multiples of 32 and at least 32. The decoded video
  (`width × height × frames × 3`) must stay below 2^31 elements by default.
- Frames: 1 to 345 at 24 fps. Requests are aligned up to the H3 grid (`17k + 5` frames).
- Schedule: 50 sigma points (49 model evaluations). The prepared timestep tables fix it.
- FL2VA: a prompt with optional first and last frames. Ref2VA: 1 to 12 ordered image, video
  or audio references, given as normalized raw media (see [prepared artifacts](../../docs/prepared-artifacts.md)).
- Attention: `subblock_sparse_attn` (default) or `torch_cudnn_sdpa`.
- `public_variant.py` admits only the official base checkpoint identity and the documented
  configuration and request fields. Unknown fields are errors.

CPU tests check the Python runtime and package contracts. They do not establish GPU
numerical correctness or startup performance.

## Layout

| Path | Contents |
| --- | --- |
| `native_pipeline.py`, `runtime/` | Request handling and startup declarations over the shared core executor |
| `pipeline/` | Canvas, sampling, token layout and latent assembly |
| `text_encoder/`, `transformer/` | Qwen3-VL conditioning and the joint video/audio DiT |
| `video_vae/`, `audio_vae/` | Encoders and decoders |
| `kernels/`, `support/` | Model-specific CUDA operators and loaders |
| `prepare/`, `prepare_*.py` | One-time package preparation; may use PyTorch, inference does not |
