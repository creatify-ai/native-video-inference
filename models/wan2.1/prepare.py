"""Prepare the published Wan 2.1 I2V-14B-720P checkpoint for native loading (CPU only, once).

    python models/wan2.1/prepare.py --checkpoint <published checkpoint> --out <prepared dir>

The published files are never modified. The prepared directory holds every weight serving reads,
in 64 MiB safetensors shards: a Modal Volume reads many files in parallel far faster than a few
large ones (about 10 GB/s against 5-6 GB/s for the published layout).

- `dit/`: the DiT's FP32 shards as BF16. Each tensor equals `tensor.to(torch.bfloat16)`, the dtype
  the official pipeline runs the DiT in.
- `t5/`, `clip/`, `vae/`: the published T5, CLIP and VAE `.pth` tensors, byte for byte.
- `derived.safetensors`: `freqs`, the float64 RoPE table that `WanModel.__init__` builds on the CPU.
- `manifest.json` (`native-checkpoint-v1`): every tensor's byte range, dtype, shape and SHA-256,
  under the prefixes `dit.`, `t5.`, `clip.` and `vae.`.
- `prepare-report.json`: the manifest's SHA-256. Serving pins this digest offline.

`freqs` needs PyTorch (CPU is enough) and safetensors; the BF16 conversion needs numpy. With
torch 2.13.0, safetensors 0.5.3 and glibc 2.39 (Ubuntu 24.04) the manifest matches
`REFERENCE_MANIFEST_SHA256`, the preparation behind the published parity results. The CPU
`torch.polar` uses glibc's cos/sin, so another glibc can change a few `freqs` entries by 1 ULP.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from core_infra.runtime.checkpoint import _select, build, convert, write  # noqa: E402

REPOSITORY = 'Wan-AI/Wan2.1-I2V-14B-720P'
REVISION = '8823af45fcc58a8aa999a54b04be9abc7d2aac98'
REFERENCE_MANIFEST_SHA256 = '0261458d7d20ca90ad744df2adddf89afa888566528e3db4d290573758bb5721'
DIT_SHARDS = 'diffusion_pytorch_model-*.safetensors'
# Published encoder files, by manifest tensor prefix; re-sharded unchanged.
COMPONENT_FILES = {
    't5.': 'models_t5_umt5-xxl-enc-bf16.pth',
    'clip.': 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth',
    'vae.': 'Wan2.1_VAE.pth',
}
SHARD_BYTES = 64 << 20
DIM, HEADS = 5120, 40  # WanModel's i2v-14B configuration


def rope_freqs():
    """`WanModel.freqs` as [1024, 64, 2] float64 (real, imaginary).

    `rope_params` is copied from `wan/modules/model.py` of Wan-Video/Wan2.1 at commit
    9737cba9c1c3c4d04b33fcad41c111989865d315 (Apache-2.0, Copyright 2024-2025 The Alibaba Wan
    Team Authors), so the table is computed by the same PyTorch CPU operations.
    """
    import torch

    def rope_params(max_seq_len, dim, theta=10000):
        assert dim % 2 == 0
        freqs = torch.outer(
            torch.arange(max_seq_len),
            1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)))
        return torch.polar(torch.ones_like(freqs), freqs)

    d = DIM // HEADS
    freqs = torch.cat([rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)),
                       rope_params(1024, 2 * (d // 6))], dim=1)
    return torch.view_as_real(freqs).contiguous()


def write_derived(path):
    from safetensors.torch import save_file
    save_file({'freqs': rope_freqs()}, str(path))


def prepare(checkpoint, out, workers=16):
    """Writes the prepared directory `out` for the published checkpoint directory `checkpoint`."""
    checkpoint, out = Path(checkpoint), Path(out)
    missing = [name for name in COMPONENT_FILES.values() if not (checkpoint / name).is_file()]
    shards = _select(checkpoint, [DIT_SHARDS]) if checkpoint.is_dir() else []
    if missing or not shards:
        raise FileNotFoundError(f'{checkpoint} is not the published {REPOSITORY} checkpoint '
                                f'(missing {missing or [DIT_SHARDS]})')
    out.mkdir(parents=True, exist_ok=True)
    report = {'checkpoint': {'repository': REPOSITORY, 'revision': REVISION, 'path': str(checkpoint)},
              'prepared': str(out), 'seconds': {}}
    began = time.time()
    report['convert'] = convert(checkpoint, shards, out / 'dit', shard_bytes=SHARD_BYTES, copy=['config.json'])
    for prefix, name in COMPONENT_FILES.items():
        report['convert_' + prefix[:-1]] = convert(checkpoint, [name], out / prefix[:-1], prefix=prefix[:-1],
                                                   shard_bytes=SHARD_BYTES, to_bf16=False)
    write_derived(out / 'derived.safetensors')
    report['seconds']['convert'] = round(time.time() - began, 1)
    hashed = time.time()
    roots = {'checkpoint': str(checkpoint), 'prepared': str(out)}
    files = [('prepared', f'dit/{name}', 'dit.') for name in _select(out / 'dit', ['*.safetensors'])]
    files.append(('prepared', 'derived.safetensors', 'dit.'))
    for prefix in COMPONENT_FILES:
        files += [('prepared', f'{prefix[:-1]}/{name}', prefix) for name in _select(out / prefix[:-1], ['*.safetensors'])]
    manifest = build(roots, files, workers=workers)
    digest = write(manifest, out / 'manifest.json')
    report['seconds']['manifest'] = round(time.time() - hashed, 1)
    report.update(manifest_sha256=digest, matches_reference=digest == REFERENCE_MANIFEST_SHA256,
                  tensors=len(manifest['tensors']),
                  bytes=sum(tensor['bytes'] for tensor in manifest['tensors'].values()))
    (out / 'prepare-report.json').write_text(json.dumps(report, indent=1) + '\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--checkpoint', required=True, help=f'published {REPOSITORY} directory at {REVISION[:7]}')
    parser.add_argument('--out', required=True, help='prepared directory to write')
    parser.add_argument('--workers', type=int, default=16, help='hashing threads')
    args = parser.parse_args(argv)
    report = prepare(args.checkpoint, args.out, args.workers)
    print(json.dumps(report, indent=1))
    if not report['matches_reference']:
        print(f'note: manifest SHA-256 {report["manifest_sha256"]} differs from the reference preparation '
              f'{REFERENCE_MANIFEST_SHA256}; pin your own digest when serving', file=sys.stderr)


if __name__ == '__main__':
    main()
