"""One-time preparation of native H3 packages from the official MiniMax-H3 checkpoint.

Input: the root-level Diffusers layout of `MiniMaxAI/MiniMax-H3` at `common.REVISION`
(`text_encoder/`, `transformer/` or `transformer_ref/`, `vae/`, `audio_vae/`, `processor/`,
`tokenizer/`). Output, one directory per workflow:

    conditioner/    Qwen3-VL parameters, rotary buffers, tokenizer and resolved processor config
    dit-base/, dit/ transformer parameters with 49-step timestep tables; dit/ has packed Q/K/V
    router/         SubBlock router CUBINs (Triton AOT of SGLang's kernels)
    video_vae/      image encoder and Inductor decoder programs with their parameters
    audio_vae/      audio decoder graph
    video_encoder/, audio_encoder/   Ref2VA reference encoders
    native.json     the configuration `MiniMaxH3Pipeline.from_prepared()` reads
    preparation.json  per-component reports with manifest SHA-256s

    python -m models.minimax_h3_official.prepare run --checkpoint CKPT --workflow fl2va --out OUT \\
        --programs target/release --dense-library target/native/minimax_h3_official/libh3_dense.so

Every native tensor file is recorded with its byte count and SHA-256. Preparation runs once
on a B200 with PyTorch, Diffusers and Transformers at the pins in `common`; inference
loads none of them.
"""
