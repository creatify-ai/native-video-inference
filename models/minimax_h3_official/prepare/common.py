"""Pins and helpers shared by the H3 preparation steps. Imports no ML framework.

Preparation reads the root-level Diffusers layout of the official checkpoint
(`text_encoder/`, `transformer/`, `transformer_ref/`, `vae/`, `audio_vae/`, `processor/`,
`tokenizer/`). Every tensor file it writes is recorded with its byte count and SHA-256.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import uuid

from core_infra.runtime.checkpoint import bf16_bits, safetensors_tensors

REPO_ID = 'MiniMaxAI/MiniMax-H3'
REVISION = '42ed227ee7df40d41602854ae760620d6eb651fe'
PARTITIONS = {'fl2va': 'FL2VA', 'ref2va': 'Ref2VA'}
# The Diffusers transformer folder of each official partition.
TRANSFORMERS = {'fl2va': 'transformer', 'ref2va': 'transformer_ref'}
# Pinned preparation environment (one-time; inference never imports these).
DIFFUSERS_COMMIT = '9284607295a09f759aadd65ed08f48b35feea6d9'
TORCH = '2.13.0'
TRANSFORMERS_VERSION = '5.15.0'
SAFETENSORS_DTYPES = {'F32': ('torch.float32', 4), 'BF16': ('torch.bfloat16', 2), 'F16': ('torch.float16', 2),
                      'I64': ('torch.int64', 8), 'I32': ('torch.int32', 4)}
TORCH_WIDTHS = {name: width for name, width in SAFETENSORS_DTYPES.values()} | {'torch.uint8': 1}
CHUNK = 16 << 20


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    """Atomically write `value`; return the SHA-256 of the written bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2) + '\n').encode()
    partial = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.partial')
    partial.write_bytes(raw)
    partial.replace(path)
    return hashlib.sha256(raw).hexdigest()


def specification(path, root, dtype, shape):
    """Manifest entry of one contiguous native tensor file under `root`."""
    path, root = Path(path), Path(root)
    if dtype not in TORCH_WIDTHS or path.stat().st_size != math.prod(shape) * TORCH_WIDTHS[dtype]:
        raise ValueError('native tensor size/dtype mismatch: ' + str(path))
    return {'file': str(path.relative_to(root)), 'dtype': dtype, 'shape': list(shape),
            'stride': [math.prod(shape[i + 1:]) for i in range(len(shape))],
            'bytes': path.stat().st_size, 'sha256': digest(path), 'storage_offset': 0, 'contiguous': True}


class Checkpoint:
    """Tensor index of one Diffusers component folder, read from safetensors headers."""

    def __init__(self, root, component):
        self.root, self.component = Path(root), component
        self.tensors = {}
        for file in sorted((self.root / component).glob('*.safetensors')):
            for name, tensor in safetensors_tensors(file).items():
                if name in self.tensors:
                    raise ValueError('duplicate official tensor: ' + name)
                self.tensors[name] = (file, tensor)
        if not self.tensors:
            raise ValueError('official component missing: ' + str(self.root / component))

    def layout(self, name):
        _, (_, _, stored, shape) = self.tensors[name]
        return SAFETENSORS_DTYPES[stored][0], list(shape)

    def read(self, name):
        file, (offset, count, _, _) = self.tensors[name]
        with file.open('rb') as source:
            source.seek(offset)
            raw = source.read(count)
        if len(raw) != count:
            raise ValueError('truncated checkpoint tensor: ' + name)
        return raw

    def extract(self, name, target, dtype=None):
        """Copy one tensor's bytes; FP32 storage may be narrowed to BF16 with RNE rounding."""
        file, (offset, count, stored, shape) = self.tensors[name]
        source_dtype = SAFETENSORS_DTYPES[stored][0]
        dtype = dtype or source_dtype
        if dtype != source_dtype and (source_dtype, dtype) != ('torch.float32', 'torch.bfloat16'):
            raise ValueError(f'unsupported conversion of {name}: {source_dtype} -> {dtype}')
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        with file.open('rb') as source, target.open('xb') as output:
            source.seek(offset)
            remaining = count
            while remaining:
                raw = source.read(min(CHUNK, remaining))
                if not raw:
                    raise ValueError('truncated checkpoint tensor: ' + name)
                remaining -= len(raw)
                output.write(bf16_bits(raw, name) if dtype != source_dtype else raw)
        return dtype, list(shape)


def extract_all(checkpoint, root, names, prefix='', workers=8):
    """Extract `names` (native name -> checkpoint name) into root/weights/<native>.raw."""
    root = Path(root)

    def one(item):
        native, source = item
        target = root / 'weights' / (native + '.raw')
        dtype, shape = checkpoint.extract(prefix + source, target)
        return native, specification(target, root, dtype, shape)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(one, sorted(names.items())))


def write_tensor(root, relative, value):
    """Write a torch tensor's contiguous bytes and return its manifest entry (offline only)."""
    import torch
    value = value.detach().contiguous().cpu()
    path = Path(root) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return specification(path, root, str(value.dtype), list(value.shape))


def verify_files(root, expected, workers=8):
    """Check {relative path: sha256} against the downloaded checkpoint."""
    root = Path(root)

    def one(item):
        name, wanted = item
        if digest(root / name) != wanted:
            raise ValueError('official checkpoint file differs: ' + name)
        return name

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return sorted(pool.map(one, expected.items()))
