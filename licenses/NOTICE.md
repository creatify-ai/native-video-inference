# Third-party notices

This directory holds the license texts for adapted source and external build inputs.
One shared [Apache-2.0 text](APACHE-2.0-LICENSE) serves the Apache-licensed sources below.
Other license files keep their upstream copyright, conditions and disclaimers verbatim.
No model weights, tokenizer vocabularies or stored reference activations are distributed here;
weights keep their own terms.

## Models

- **MiniMax H3** (`models/minimax_h3_official`): MiniMax H3 is licensed under the
  [MiniMax H3 Community License Agreement](../models/minimax_h3_official/LICENSE), Copyright © 2026 MiniMax.
  All Rights Reserved. That agreement, including its territory limits and Exhibit A, governs the
  directory; see its [NOTICE](../models/minimax_h3_official/NOTICE). Model and layout code follow Diffusers commit
  `9284607295a09f759aadd65ed08f48b35feea6d9`, including `modular_pipelines/minimax_h3/before_denoise.py`
  and `decoders.py`. Copyright 2026 MiniMax and the Hugging Face team, [Apache-2.0](APACHE-2.0-LICENSE).
  The video VAE revision is `42ed227ee7df40d41602854ae760620d6eb651fe`. Qwen grid ordering,
  interpolation and multimodal preparation follow Transformers 5.15.0 (Copyright 2018- The Hugging Face
  team, Apache-2.0). `FL2VA/` and `Ref2VA/` keep the official configuration files byte for byte.
- **Wan 2.1** (`models/wan2.1`): follows the official `Wan-Video/Wan2.1` image-to-video pipeline at
  commit `9737cba` (checkpoint `Wan-AI/Wan2.1-I2V-14B-720P` revision `8823af4`), including
  `FlowUniPCMultistepScheduler`. Copyright 2024-2025 The Alibaba Wan Team Authors,
  [Apache-2.0](APACHE-2.0-LICENSE). The CPU front end ports Pillow's `Resample.c` bilinear filter
  ([Pillow license](PILLOW-LICENSE)), NumPy's float32 pairwise reductions ([BSD 3-Clause](NUMPY-LICENSE))
  and the ftfy 6.3.1 steps it reproduces (Copyright Robyn Speer, Apache-2.0). `html5_entities.rs` is
  generated from CPython 3.9.6 `html.entities.html5` ([PSF license](CPYTHON-LICENSE)). The rgb24
  conversion reproduces SLEEF's `Sleef_powf16_u10` arithmetic ([Boost 1.0](SLEEF-LICENSE)).
- **LTX 2.5** (`models/ltx25`): a derivative of LTX-2.x by Lightricks Ltd., following the
  `Lightricks/LTX-2` source at commit `be6d6d0a69d11a0a399c462ed92606692e5f8697` and model revision
  `49b093a39c19169a8a81721dcc541e07e945232f`. It is distributed under the
  [LTX-2.x Community License Agreement](../models/ltx25/LICENSE) (August 11, 2026), including its
  Attachment A use restrictions. Entities with annual revenues of at least $10,000,000 need a paid
  license from Lightricks (licensing@ltx.io) for any use other than a Non-Commercial Purpose.
  Its 3D neighborhood attention instantiates NATTEN 0.21.7 `cutlass-fna` forward headers; NATTEN and
  CUTLASS are external build inputs and keep their licenses. The image front end matches Pillow's
  RGB JPEG decode and PyAV's single-frame encode ([PyAV BSD 3-Clause](PYAV-LICENSE)).

## Shared numerical sources

- **PyTorch:** native matrix operations, cuDNN bridges, reductions, normalization, resize, softmax and
  random generation follow commit `cf30153c4c131c8164ee7798e5022d810682e2cb` (v2.13.0). Derived portions
  keep the [PyTorch BSD-style license](PYTORCH-LICENSE). Sources include
  `aten/src/ATen/native/cudnn/{MHA,Conv_v8}.cpp`, `ATen/cudnn/Descriptors.h`, `layer_norm_kernel.cu`,
  `group_norm_kernel.cu`, `SharedReduceOps.h`, `block_reduce.cuh`, `Reduce.cuh`, `PersistentSoftmax.cuh`,
  `DepthwiseConv2d.cu`, `UpSample*.cu`, `WeightNorm.cu` and the Philox `randn` path. Attention wrappers
  instantiate `ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h`. These are compile-time
  headers; no Torch or ATen runtime is linked or loaded.
- **CPU RNG:** MT19937 and sampling order follow PyTorch's `ATen/core/MT19937RNGEngine.h`,
  `TransformationHelper.h`, `DistributionsHelper.h` and `ATen/native/cpu/DistributionTemplates.h`.
  The Matsumoto/Nishimura copyright and BSD terms are in [MT19937-LICENSE](MT19937-LICENSE).
  The external `avx_mathfun.h` keeps Giovanni Garberoglio's zlib notice and its acknowledgment of
  Julien Pommier's `sse_mathfun.h`.
- **FlashInfer:** `models/minimax_h3_official/kernels/transformer/flashinfer_bridge.cu` adapts launch
  construction from version `0.6.15.post1` (`flashinfer/cute_dsl/sparse/blk64/flash_fwd_launch_template.h`).
  Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved. [Apache-2.0](APACHE-2.0-LICENSE).
  Upstream attention kernels and CUTLASS remain external build dependencies.
- **cuDNN frontend:** headers at commit `c4a97621eca52fa0c3a1862a411a16be580b25c6` keep their
  [NVIDIA MIT license](CUDNN-FRONTEND-LICENSE). CUDA, cuBLAS, cuDNN and NCCL are separately distributed.
- **JSON:** nlohmann/json keeps Niels Lohmann's [MIT copyright and license](NLOHMANN-JSON-LICENSE).
- **Tokenizers:** `vendor/native-tokenizers` is Hugging Face `tokenizers` 0.22.2 (Apache-2.0) with the
  local changes listed in its `UPSTREAM_PATCH.diff`.

Kernels exported ahead of time keep the notices of their contributing Torch, Inductor and Triton
sources and compilers. FFmpeg, libx264 and other dynamically used codecs keep their separately
distributed licenses.
