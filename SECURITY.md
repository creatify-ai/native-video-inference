# Security policy

## Reporting a vulnerability

Please report security issues privately by email to support@creatify.ai, with
"native-video-inference security" in the subject. Do not open a public issue, discussion or pull
request for them.

Include what you found, the commit you tested, how to reproduce it and the impact you expect. We
will acknowledge your report, keep you informed while we work on a fix, and credit you when it is
released if you would like.

## Scope

This policy covers the code in this repository: the core framework (`core_infra/`), the model
ports under `models/`, the tools and the Modal examples. For example: a request that makes a
resident server read or write outside its declared roots, a checkpoint whose integrity checks can
be bypassed, or memory-safety bugs reachable from a request.

Report issues in third-party components to their maintainers as well: the vendored tokenizer
library under `vendor/`, the upstream model code and weights (MiniMax H3, Wan 2.1, LTX-2.x), and
the CUDA, cuBLAS, cuDNN and FlashInfer libraries. Model outputs and content safety are outside
this policy.
