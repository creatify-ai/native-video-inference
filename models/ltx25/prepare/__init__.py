"""One-time preparation of a native LTX 2.5 profile from the pinned PyTorch reference.

A profile serves one geometry (width, height, frame count, frame rate) and one branch (T2V or
I2V); prompt, seed and image change per request. Preparation runs on one B200 inside the
reference environment built from `Lightricks/LTX-2` at `common.REFERENCE_COMMIT`:

1. `pipeline` traces one complete request through the official two-stage `DistilledPipeline`.
2. `forward`, `text`, `upsampler`, `encoder` (I2V), `audio` and `video` trace the component
   programs from the inputs that run saved. Each capture runs in a fresh process.
3. `export` keeps programs, Triton CUBINs and model constants, drops every activation, and
   writes `profile.json` for `ltx25 infer`.

    python -m prepare run --work WORK --profile OUT --weights WEIGHTS \\
        --source-identity IDENTITY.json --request REQUEST.json [--image IMAGE.jpg]

PyTorch is used here only. Inference with the exported profile loads no Python or PyTorch.
"""
