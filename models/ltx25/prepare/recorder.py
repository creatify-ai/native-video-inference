"""Content-addressed tensor recording: every boundary is hashed, only needed bytes are written."""
import ctypes
import dataclasses
import hashlib
from pathlib import Path
import uuid

from .common import write_json


def immutable(name, weights=False):
    """Model constants a profile keeps. Decoder captures also keep their (converted) weights;
    every other weight is read from the original checkpoint at inference time."""
    return (".buffers." in name or ".constant." in name or name.startswith("gemma.rope.constants.")
            or (name.startswith("op.ancestral.") and name.endswith(".sigmas"))
            or (weights and ".weights." in name))


def byte_view(address, byte_count):
    """The caller keeps the owning allocation alive while using this view."""
    if byte_count < 0:
        raise ValueError("negative tensor byte count")
    return memoryview((ctypes.c_ubyte * byte_count).from_address(address)).cast("B")


class Recorder:
    """`keep(name)` selects the tensors whose bytes are written; all others are recorded by
    dtype, shape, strides and SHA-256 only."""

    def __init__(self, root, blobs, keep=immutable):
        self.root, self.blobs, self.keep = Path(root), Path(blobs), keep
        self.blobs.mkdir(parents=True, exist_ok=True)
        self.tensors = {}
        self.metadata = {}

    def tensor(self, name, value, save=True):
        from torch.utils._python_dispatch import _disable_current_modes
        with _disable_current_modes():
            return self._tensor(name, value, save and self.keep(name))

    def _tensor(self, name, value, save):
        if name in self.tensors:
            raise ValueError(f"duplicate tensor boundary: {name}")
        layout = {"source_stride": list(value.stride()), "source_storage_offset": value.storage_offset(),
                  "source_device": str(value.device), "source_contiguous": value.is_contiguous()}
        value = value.detach().contiguous()
        cpu = value.cpu().contiguous()
        # A buffer view keeps Py_ssize_t lengths: ctypes.string_at truncates tensors over 4 GiB.
        byte_count = cpu.numel() * cpu.element_size()
        raw = byte_view(cpu.data_ptr(), byte_count)
        if len(raw) != byte_count:
            raise RuntimeError("tensor buffer byte count differs from its shape")
        sha = hashlib.sha256(raw).hexdigest()
        path = self.blobs / (sha + ".raw")
        if save and not path.exists():
            temporary = path.with_suffix(f".{uuid.uuid4().hex}.partial")
            temporary.write_bytes(raw)
            temporary.replace(path)
        self.tensors[name] = {"dtype": str(value.dtype), "shape": list(value.shape), "stride": list(value.stride()),
                              "bytes": len(raw), "sha256": sha, "file": str(path) if save else None,
                              "storage_offset": 0, "contiguous": True, **layout}
        return self.tensors[name]

    def tree(self, name, value, save=True):
        import torch
        if isinstance(value, torch.Tensor):
            return self.tensor(name, value, save)
        if dataclasses.is_dataclass(value):
            for field in dataclasses.fields(value):
                self.tree(f"{name}.{field.name}", getattr(value, field.name), save)
        elif isinstance(value, dict):
            for key, item in value.items():
                self.tree(f"{name}.{key}", item, save)
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                self.tree(f"{name}.{index}", item, save)
        elif value is None or isinstance(value, (str, bool, int, float)):
            self.metadata[name] = value
        elif isinstance(value, slice):
            self.metadata[name] = {"slice": [value.start, value.stop, value.step]}

    def close(self, identity):
        manifest = {"schema": "ltx25-gold-v1", "identity": identity, "tensors": self.tensors, "metadata": self.metadata}
        write_json(self.root / "manifest.json", manifest)
        return manifest
