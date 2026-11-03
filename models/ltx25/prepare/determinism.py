"""The reference's determinism policy. `install` runs before torch or any LTX module is imported."""
import os


def install(cache_root):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "0"
    os.environ["TRITON_CACHE_DIR"] = str(cache_root / "triton")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_root / "inductor")
    os.environ["CUDA_CACHE_PATH"] = str(cache_root / "cuda")
    import torch
    import triton
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.tunable.enable(False)
    original_autotune = triton.autotune

    def first_config(configs, *args, **kwargs):
        kwargs.pop("cache_results", None)
        return original_autotune(configs[:1], *args, cache_results=False, **kwargs)

    triton.autotune = first_config
    return {
        "cudnn_deterministic": True, "cudnn_benchmark": False, "allow_tf32": False,
        "triton_autotune": "first-config; cache disabled; fresh directory per run",
        "ltx_attention": "cuDNN SDPA", "gemma_attention": "upstream dispatch; traced separately",
        "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }


def after_import():
    """Freeze LTX's automatic attention selection to cuDNN SDPA. Gemma keeps its own dispatch."""
    from torch.nn.attention import SDPBackend
    import ltx_core.model.transformer.attention as attention
    attention.automatic_attention.cache_clear()
    attention.automatic_masked_attention.cache_clear()
    # AttentionOps dataclass default factories keep the original function objects:
    # replace the selectors those factories call, not their names.
    attention._select_primary_attention = lambda: attention.PytorchAttention([SDPBackend.CUDNN_ATTENTION])
    attention._select_masked_attention = lambda: attention.PytorchAttention([SDPBackend.CUDNN_ATTENTION])


def begin(root):
    """Start one capture process: determinism policy, Triton recording, frozen attention."""
    policy = install(root / "kernel-cache")
    from .tracing import TritonRecorder
    kernels = TritonRecorder(root / "triton")
    kernels.install()
    import ltx_core.model.transformer.attention  # noqa: F401  (patched below)
    after_import()
    return policy, kernels
