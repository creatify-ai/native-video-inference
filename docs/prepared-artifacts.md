# MiniMax H3 prepared artifacts

H3 inference reads native artifact packages, not the safetensors checkpoint.
The official configuration trees in `models/minimax_h3_official/{FL2VA,Ref2VA}` describe the model
setup, not an inference cache. `models/minimax_h3_official/prepare/` builds the packages once from
the official checkpoint, and [`examples/modal/h3.py`](../examples/modal/h3.py) runs it on a B200.

## Producing the packages

Preparation reads the root-level Diffusers folders of `MiniMaxAI/MiniMax-H3` at revision
`42ed227ee7df40d41602854ae760620d6eb651fe` and runs once on a B200 with PyTorch 2.13.0,
Diffusers at commit `9284607295a09f759aadd65ed08f48b35feea6d9` and Transformers 5.15.0:

```sh
python -m models.minimax_h3_official.prepare run --checkpoint CKPT --workflow fl2va --out OUT \
    --verify-checkpoint --programs target/release \
    --dense-library target/native/minimax_h3_official/libh3_dense.so
```

| Package | Source | Method |
| --- | --- | --- |
| `conditioner/` | `text_encoder/`, `processor/`, `tokenizer/` | Parameters of the layers that reach hidden state 50, copied byte for byte; derived rotary buffers; the processor configuration serialized by Transformers |
| `dit-base/`, `dit/` | `transformer/` (FL2VA) or `transformer_ref/` (Ref2VA) | Parameters except the AdaLN projections, byte for byte; AdaLN and final-modulation tables for the 50-point schedule computed with the Diffusers modules; `dit/` packs Q/K/V rows (`prepare_qkv.py`) |
| `router/` | SGLang's SubBlock router kernels at a pinned commit | Triton 3.8.0 ahead-of-time CUBINs, one per launch specialization class |
| `video_vae/` | `vae/` | Image encoder and compiled decoder programs with their parameters |
| `audio_vae/` | `audio_vae/` | Captured audio decoder graph and constants |
| `video_encoder/`, `audio_encoder/` | `vae/`, `audio_vae/` | Ref2VA reference encoders |

The run is resumable: `preparation.json` records each finished component with the SHA-256 of its
manifest, and every manifest lists each tensor file with its byte count and SHA-256.
`--programs` also writes `native.json`. The SubBlock attention kernel is a build input, exported
by `models/minimax_h3_official/prepare_subblock_aot.py`.

## Native configuration

`MiniMaxH3Pipeline.from_prepared()` reads a JSON object with these fields.
A wrapper with the same object under `pipeline_config` is also supported.
Use absolute paths visible inside the inference process.

| Field | Meaning |
| --- | --- |
| `workflow` | `fl2va` or `ref2va` |
| `nfe` | Optional; must equal the prepared schedule (49 evaluations) |
| `programs` | Paths to native executables |
| `models` | Paths to component packages and exported programs |
| `environment` | Optional native environment values |
| `runtime` | Startup engine, reader budgets, prefetch quotas and verified backend plans |
| `media` | `format` (`mp4` or `raw`), `capture_raw`, `hash_outputs`, `ffmpeg` |

The required program keys are `conditioner`, `dit`, `video_vae`, `audio_vae`, and `pipeline`.
Audio references also need `audio_encoder`.
Their binary names are:

| Key | Binary |
| --- | --- |
| `conditioner` | `h3-official-conditioner` |
| `dit` | `h3-official-model-run` |
| `video_vae` | `h3-official-video-vae` |
| `audio_vae` | `h3-official-audio-infer` |
| `audio_encoder` | `h3-official-audio-encode` |
| `pipeline` | `h3-official-model-pipeline` |

The required model keys are `conditioner`, `dit`, `video_vae`, `audio_vae`, `router`, and `dense_library`.
Video references also need `video_encoder`; audio references need `audio_encoder`.
The conditioner package must include its tokenizer vocabulary, processor payload, and model configuration.
The router and VAEs need their matching exported tensor programs and manifests.
`dense_library` points to the matching `libh3_dense.so`.
All programs need their matching dynamic libraries on the loader path.

Prepared tensors, tables, layouts, and shards must match their manifests.
Unknown configuration fields are rejected; see `models/minimax_h3_official/public_variant.py`.

Both workflows use the shared framework startup executor (`runtime.startup_engine="framework"`,
the default). Light integrity guards are declared with `H3_MODEL_LOAD_LIGHT_SHA_MANIFEST` and
`H3_MODEL_LOAD_LIGHT_SHA_MANIFEST_SHA256` in `environment`. A missing or malformed declaration fails
before startup; it does not silently switch loaders.

## FL2VA request

[examples/fl2va-request.json](../examples/fl2va-request.json) is a small request example.
Replace the input image path with a file in your volume.
For prompt-only generation, remove `first_png`.
A last frame requires a first frame.

Both workflows require `prompt`, `width`, `height`, `num_frames`, and `seed`.
Canvas dimensions must be multiples of 32 and at least 32.
The seed must be a non-negative integer. The frame count must be between 1 and 345.
The runtime aligns frames to the H3 grid and rejects an aligned count above 345.
The frame rate is 24 fps.

## Ref2VA request

[examples/ref2va-request.json](../examples/ref2va-request.json) selects a normalized reference descriptor.
[examples/reference-image.json](../examples/reference-image.json) shows one image reference.
The image payload is interleaved RGB bytes, with exactly `width × height × 3` bytes.
It is not a PNG or JPEG file.
Prepare normalized reference media before inference. The entry does not decode or resize arbitrary media files.

Reference order affects conditioning. Keep paths absolute and geometry consistent with each payload.
A video reference has `"type": "video"`, `frames` (at most 345) and `width × height × 3 × frames`
bytes; `has_audio` with an absolute `audio_file` adds its normalized FP32 stereo audio. An audio
reference has `"type": "audio"` and `audio_file`. Video references need the `video_encoder`
package and audio needs `audio_encoder`.

## Run on Modal

See [`examples/modal/h3.py`](../examples/modal/h3.py) and the
[H3 README](../models/minimax_h3_official/README.md#run-on-modal). The example downloads the checkpoint
into your Volume (`setup`), prepares the packages under `prepared/<workflow>/` (`prepare`), writes
`native.json` (`check`) and runs one request on a B200 (`run`).

## Outputs

The output directory must not exist before the request.
It contains stage logs, intermediate files, and `report.json`.
The report records the request, aligned frame count, stage times, startup events, and output hashes.
The complete outputs are `dit/final_video.raw` and `dit/final_audio.raw` (FP32 DiT latents),
`audio.raw` (FP32 stereo PCM), `output.mp4`, and `video_decode/frames.raw` (RGB24) with `media.capture_raw`
or `media.format="raw"`. With `media.hash_outputs`, `report.json` records each file's SHA-256.
`latent_only` configuration omits VAE decoding.

## Optional backend plans

`runtime.backend_plans` accepts a manifest `path` and its `sha256`.
Keys bind graphs, shapes, GPU, driver, and SDK versions.
Missing keys use native builders. Plan preparation executes no generation warmup.
Plan bundles are generated assets and stay outside source control.
