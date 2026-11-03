import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / 'models/minimax_h3_official/prepare_qkv.py'
spec = importlib.util.spec_from_file_location('prepare_qkv', MODULE)
packing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packing)


def tensor(raw, filename):
    return dict(file=filename, dtype='torch.bfloat16', shape=[2, 4], stride=[4, 1],
                contiguous=True, storage_offset=0, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def test_concatenation_preserves_every_qkv_storage_bit():
    payloads = [bytes(range(start, start + 16)) for start in (0, 16, 32)]
    specs = [tensor(raw, str(i)) for i, raw in enumerate(payloads)]
    merged, raw = packing.concatenate(specs, payloads, 'qkv.raw')
    assert raw == bytes(range(48))
    assert merged['shape'] == [6, 4] and merged['stride'] == [4, 1]
    assert merged['sha256'] == hashlib.sha256(raw).hexdigest()
    assert all(spec['shape'] == [2, 4] for spec in specs)


@pytest.mark.parametrize('change', [dict(dtype='torch.float32'), dict(shape=[4, 2]),
                                   dict(stride=[5, 1]), dict(sha256='0' * 64), dict(bytes=14)])
def test_rejects_incompatible_or_corrupt_tensor(change):
    payloads = [bytes(range(16))] * 3
    specs = [tensor(raw, str(i)) for i, raw in enumerate(payloads)]
    specs[1].update(change)
    with pytest.raises(ValueError):
        packing.concatenate(specs, payloads, 'qkv.raw')


def test_package_keeps_constants_and_unmodified_weights(tmp_path):
    source, destination = tmp_path / 'source', tmp_path / 'packed'
    (source / 'weights').mkdir(parents=True)
    (source / 'schedule').mkdir()
    (source / 'schedule/sigmas.raw').write_bytes(b'unchanged constants')
    name = 'transformer_blocks.0.attn'
    weights = {}
    for i, part in enumerate(('q', 'k', 'v', 'out')):
        raw = bytes([i]) * 16
        filename = f'weights/{part}.raw'
        (source / filename).write_bytes(raw)
        weights[f'{name}.to_{part}.weight'] = tensor(raw, filename)
    model = dict(model_source=dict(repo_id='MiniMaxAI/MiniMax-H3', variant='base', revision='a' * 40),
                 numerics='sglang-bf16-cudnn-v1', modules={name: dict(fused_projections=False)}, weights=weights,
                 schedule=dict(sigmas='schedule/sigmas.raw'))
    (source / 'manifest.json').write_text(json.dumps(model))
    receipt = packing.prepare(source, destination)
    packed = json.loads((destination / 'manifest.json').read_bytes())
    assert receipt['fused_attention_modules'] == 1
    assert packed['modules'][name]['fused_projections']
    merged = packed['weights'][name + '.to_qkv.weight']
    assert (destination / merged['file']).read_bytes() == b'\0' * 16 + b'\1' * 16 + b'\2' * 16
    assert (destination / 'weights/out.raw').read_bytes() == b'\3' * 16
    assert (destination / packed['schedule']['sigmas']).read_bytes() == b'unchanged constants'
    assert json.loads((source / 'manifest.json').read_bytes()) == model
