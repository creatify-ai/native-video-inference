"""CPU checks of models/wan2.1/prepare.py on a miniature checkpoint (no weights, no GPU).

The BF16 copy needs numpy. `freqs` needs torch, which only a child process imports."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys

import pytest

WAN = Path(__file__).resolve().parents[1]
needs_numpy = pytest.mark.skipif(importlib.util.find_spec('numpy') is None, reason='numpy is not installed')
needs_torch = pytest.mark.skipif(importlib.util.find_spec('torch') is None, reason='torch is not installed')


def load():
    spec = importlib.util.spec_from_file_location('wan21_prepare', WAN / 'prepare.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_safetensors(path, tensors):
    """tensors: {name: (dtype, raw bytes, shape)}."""
    header, pos = {}, 0
    for name in sorted(tensors):
        dtype, raw, shape = tensors[name]
        header[name] = {'dtype': dtype, 'shape': list(shape), 'data_offsets': [pos, pos + len(raw)]}
        pos += len(raw)
    blob = json.dumps(header).encode()
    blob += b' ' * (-len(blob) % 8)
    path.write_bytes(struct.pack('<Q', len(blob)) + blob + b''.join(tensors[n][1] for n in sorted(tensors)))


def f32(*values):
    return struct.pack(f'<{len(values)}f', *values)


@pytest.fixture
def miniature(tmp_path, monkeypatch):
    prepare = load()
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    write_safetensors(checkpoint / 'diffusion_pytorch_model-00001-of-00002.safetensors',
                      {'blocks.0.weight': ('F32', f32(1.0, -2.0), [2])})
    write_safetensors(checkpoint / 'diffusion_pytorch_model-00002-of-00002.safetensors',
                      {'head.weight': ('F32', f32(0.5), [1]), 'patch.bias': ('BF16', b'\x80\x3f', [1])})
    (checkpoint / 'config.json').write_text('{"model_type": "i2v"}\n')
    files = {}
    for prefix in ('t5.', 'clip.', 'vae.'):
        # Stand-ins for the published .pth files; the manifest reads both formats in place.
        files[prefix] = f'{prefix[:-1]}.safetensors'
        write_safetensors(checkpoint / files[prefix], {'weight': ('F32', f32(3.0), [1])})
    monkeypatch.setattr(prepare, 'COMPONENT_FILES', files)
    monkeypatch.setattr(prepare, 'write_derived', lambda path: write_safetensors(
        path, {'freqs': ('F64', struct.pack('<4d', 1.0, 0.0, 0.5, 0.25), [2, 1, 2])}))
    return prepare, checkpoint, tmp_path / 'prepared'


@needs_numpy
def test_prepared_manifest_covers_every_weight_in_small_shards(miniature):
    prepare, checkpoint, out = miniature
    published = {p.name: p.read_bytes() for p in checkpoint.iterdir()}
    report = prepare.prepare(checkpoint, out, workers=2)
    assert {p.name: p.read_bytes() for p in checkpoint.iterdir()} == published
    manifest_bytes = (out / 'manifest.json').read_bytes()
    assert report['manifest_sha256'] == hashlib.sha256(manifest_bytes).hexdigest()
    assert report['matches_reference'] is False
    assert json.loads((out / 'prepare-report.json').read_text()) == report
    manifest = json.loads(manifest_bytes)
    tensors = manifest['tensors']
    assert set(tensors) == {'dit.blocks.0.weight', 'dit.head.weight', 'dit.patch.bias', 'dit.freqs',
                            't5.weight', 'clip.weight', 'vae.weight'}
    assert {tensors[n]['dtype'] for n in ('dit.blocks.0.weight', 'dit.head.weight', 'dit.patch.bias')} == {'BF16'}
    assert tensors['dit.freqs']['dtype'] == 'F64' and tensors['t5.weight']['dtype'] == 'F32'
    # Every weight is re-sharded under the prepared root; encoders keep their stored bytes.
    assert {entry['root'] for entry in manifest['files'].values()} == {'prepared'}
    assert {name.split('/')[0] for name in manifest['files']} == {
        'prepared:dit', 'prepared:t5', 'prepared:clip', 'prepared:vae', 'prepared:derived.safetensors'}
    entry = tensors['t5.weight']
    raw = (out / manifest['files'][entry['file']]['path']).read_bytes()[entry['offset']:entry['offset'] + entry['bytes']]
    assert raw == f32(3.0)
    # Round to nearest even, as tensor.to(torch.bfloat16).
    entry = tensors['dit.blocks.0.weight']
    file = manifest['files'][entry['file']]
    raw = (out / file['path']).read_bytes()[entry['offset']:entry['offset'] + entry['bytes']]
    assert raw == struct.pack('<2H', 0x3F80, 0xC000)
    assert (out / 'dit/config.json').read_text() == '{"model_type": "i2v"}\n'


def test_missing_published_files_fail_before_writing(miniature):
    prepare, checkpoint, out = miniature
    (checkpoint / 'vae.safetensors').unlink()
    with pytest.raises(FileNotFoundError, match='vae.safetensors'):
        prepare.prepare(checkpoint, out)
    assert not out.exists()


@needs_torch
def test_rope_table_has_wan_layout():
    script = (f'import importlib.util, json; spec = importlib.util.spec_from_file_location("p", {str(WAN / "prepare.py")!r}); '
              'p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p); f = p.rope_freqs(); '
              'print(json.dumps([list(f.shape), str(f.dtype), f[0].abs().sum().item(), f[1, 0].tolist()]))')
    shape, dtype, row0, first = json.loads(subprocess.run([sys.executable, '-c', script], check=True,
                                                          capture_output=True, text=True).stdout)
    assert shape == [1024, 64, 2] and dtype == 'torch.float64'
    assert row0 == 64.0  # position 0: cos 0 = 1, sin 0 = 0 for each of the 64 frequencies
    assert first == pytest.approx([0.5403023058681398, 0.8414709848078965])  # position 1, theta^0
