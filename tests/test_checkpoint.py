"""In-place checkpoint manifests, the BF16 preparation copy and the full audit
(core_infra.runtime.checkpoint / audit). Safetensors fixtures are written with the standard library;
the BF16 copy needs numpy, and anything that needs torch (torch.save files, torch's own BF16
rounding) runs in a child process, so this process never imports torch (other runtime tests assert
that it stays unloaded). Tests whose optional dependency is missing are skipped."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import random
import struct
import subprocess
import sys

import pytest

NATIVE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NATIVE))
from core_infra.runtime import audit, checkpoint  # noqa: E402

needs_torch = pytest.mark.skipif(importlib.util.find_spec('torch') is None, reason='torch is not installed')
needs_numpy = pytest.mark.skipif(importlib.util.find_spec('numpy') is None, reason='numpy is not installed')


def write_safetensors(path, tensors):
    """tensors: {name: (dtype, raw bytes, shape)} -> a safetensors file (sorted names, 8-byte padded header)."""
    header, pos = {}, 0
    for name in sorted(tensors):
        dtype, raw, shape = tensors[name]
        header[name] = {'dtype': dtype, 'shape': list(shape), 'data_offsets': [pos, pos + len(raw)]}
        pos += len(raw)
    blob = json.dumps(header).encode()
    blob += b' ' * (-len(blob) % 8)
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(blob)) + blob)
        for name in sorted(tensors):
            f.write(tensors[name][1])


def f32(values, shape=None):
    values = list(values)
    return ('F32', struct.pack(f'<{len(values)}f', *values), shape or (len(values),))


def normals(seed, n):
    rng = random.Random(seed)
    return [rng.gauss(0, 1) for _ in range(n)]


def torch_child(script, cwd):
    """Runs `script` with torch in a child process (cwd = the test's directory)."""
    subprocess.run([sys.executable, '-c', 'import torch\n' + script], cwd=cwd, check=True)


def read_safetensors(path):
    data = Path(path).read_bytes()
    return {name: (dtype, data[offset:offset + nbytes])
            for name, (offset, nbytes, dtype, _) in checkpoint.safetensors_tensors(path).items()}


def test_safetensors_manifest_ranges_hashes_and_light_guards(tmp_path):
    rng = random.Random(0)
    tensors = {'blocks.0.weight': f32(normals(1, 33 * 17), (33, 17)),
               'blocks.0.bias': ('BF16', bytes(rng.randrange(256) for _ in range(34)), (17,)),
               'big': f32(normals(2, 4096)), 'steps': ('I64', struct.pack('<7q', *range(7)), (7,))}
    write_safetensors(tmp_path / 'model.safetensors', tensors)
    manifest = checkpoint.build({'checkpoint': str(tmp_path)}, [('checkpoint', 'model.safetensors')], workers=2)
    data = (tmp_path / 'model.safetensors').read_bytes()
    assert manifest['files']['checkpoint:model.safetensors']['sha256'] == hashlib.sha256(data).hexdigest()
    for name, (dtype, raw, shape) in tensors.items():
        e = manifest['tensors'][name]
        payload = data[e['offset']:e['offset'] + e['bytes']]
        assert payload == raw and e['dtype'] == dtype and e['shape'] == list(shape)
        assert e['sha256'] == hashlib.sha256(payload).hexdigest()
        assert e['head_sha256'] == hashlib.sha256(payload[:1024]).hexdigest()
        assert e['tail_sha256'] == hashlib.sha256(payload[-1024:]).hexdigest()


@needs_torch
def test_pth_ranges_with_views_parameters_and_dtypes(tmp_path):
    torch_child('''
g = torch.Generator().manual_seed(0)
base = torch.randn(64, 8, generator=g)
state = {'view': base[8:16], 'param': torch.nn.Parameter(torch.randn(3, 3, generator=g), requires_grad=False),
         'bf16': torch.randn(17, generator=g).to(torch.bfloat16), 'half': torch.randn(5, generator=g).to(torch.float16),
         'ids': torch.arange(7)}
torch.save(state, 'm.pth')
for name, t in state.items():
    open(name + '.bin', 'wb').write(t.detach().contiguous().view(torch.uint8).numpy().tobytes())
''', tmp_path)
    manifest = checkpoint.build({'checkpoint': str(tmp_path)}, [('checkpoint', 'm.pth')])
    data = (tmp_path / 'm.pth').read_bytes()
    want_dtype = {'view': 'F32', 'param': 'F32', 'bf16': 'BF16', 'half': 'F16', 'ids': 'I64'}
    for name, dtype in want_dtype.items():
        e = manifest['tensors'][name]
        assert data[e['offset']:e['offset'] + e['bytes']] == (tmp_path / f'{name}.bin').read_bytes(), name
        assert e['dtype'] == dtype


@needs_torch
def test_pth_rejects_non_contiguous_tensors_and_foreign_globals(tmp_path):
    torch_child('''
import datetime
torch.save({'t': torch.randn(8, 8).t()}, 'strided.pth')
torch.save({'x': torch.zeros(2), 'obj': datetime.date(2026, 1, 1)}, 'foreign.pth')
''', tmp_path)
    with pytest.raises(ValueError, match='non-contiguous'):
        checkpoint.pth_tensors(tmp_path / 'strided.pth')
    with pytest.raises(pickle.UnpicklingError, match='unsupported pickle global'):
        checkpoint.pth_tensors(tmp_path / 'foreign.pth')


@needs_torch
@needs_numpy
def test_bf16_copy_equals_torch_rounding_including_special_values(tmp_path):
    special = [0.0, -0.0, 1.0, -1.0, float('inf'), float('-inf'), 1e-40, -1e-40, 3.4e38, 1.00390625, 1.01171875,
               65504.0, 1 + 2**-8, 1 + 3 * 2**-9]
    values = f32(special + [x * 1e3 for x in normals(3, 100000)])
    keep = struct.pack('<5q', *range(5))
    write_safetensors(tmp_path / 'diffusion_pytorch_model.safetensors', {'w': values, 'keep': ('I64', keep, (5,))})
    (tmp_path / 'values.f32').write_bytes(values[1])
    torch_child('''
import numpy as np
x = torch.from_numpy(np.fromfile('values.f32', dtype='<f4'))
open('values.bf16', 'wb').write(x.to(torch.bfloat16).view(torch.uint8).numpy().tobytes())
''', tmp_path)
    (tmp_path / 'config.json').write_text('{}')
    out = tmp_path / 'bf16'
    summary = checkpoint.convert(tmp_path, ['diffusion_pytorch_model.safetensors'], out, shard_bytes=150000,
                                 copy=['config.json'])
    assert summary['shards'] >= 2 and (out / 'config.json').exists()
    index = json.loads((out / 'diffusion_pytorch_model.safetensors.index.json').read_text())
    tensors = {}
    for shard in sorted(set(index['weight_map'].values())):
        tensors.update(read_safetensors(out / shard))
    assert tensors['w'] == ('BF16', (tmp_path / 'values.bf16').read_bytes())
    assert tensors['keep'] == ('I64', keep)
    assert index['metadata']['total_size'] == sum(len(raw) for _, raw in tensors.values())


@needs_numpy
def test_bf16_copy_refuses_nan(tmp_path):
    write_safetensors(tmp_path / 'm.safetensors', {'w': f32([1.0, float('nan')])})
    with pytest.raises(ValueError, match='NaN'):
        checkpoint.convert(tmp_path, ['m.safetensors'], tmp_path / 'out')


def test_reshard_keeps_every_stored_byte(tmp_path):
    tensors = {f'w{i}': f32(normals(i, 16)) for i in range(5)}
    tensors['nan'] = f32([float('nan'), 1.0])  # copied, never converted
    write_safetensors(tmp_path / 'm.safetensors', tensors)
    out = tmp_path / 'out'
    summary = checkpoint.convert(tmp_path, ['m.safetensors'], out, prefix='enc', shard_bytes=128, to_bf16=False)
    assert summary['shards'] > 1
    copied = {}
    for shard in sorted(out.glob('enc-*.safetensors')):
        copied.update(read_safetensors(shard))
    assert copied == read_safetensors(tmp_path / 'm.safetensors')


def test_storage_layout_moves_bytes_into_small_files_and_keeps_every_identity(tmp_path):
    write_safetensors(tmp_path / 'dit.safetensors', {f'w{i}': f32(normals(i, 16)) for i in range(5)} |
                      {'big': f32(normals(9, 600))})
    write_safetensors(tmp_path / 'text.safetensors', {'w': f32(normals(7, 16)), 'vocab': ('U8', b'abc', (3,))})
    roots = {'checkpoint': str(tmp_path)}
    logical = checkpoint.build(roots, [('checkpoint', 'dit.safetensors', 'dit.'), ('checkpoint', 'text.safetensors', 'text.')])
    out = tmp_path / 'layout'
    declaration = checkpoint.layout(logical, roots, out, shard_bytes=200, workers=2)
    raw = Path(declaration['manifest']).read_bytes()
    assert declaration['sha256'] == hashlib.sha256(raw).hexdigest()
    assert json.loads((out / 'layout.json').read_text()) == declaration
    physical = json.loads(raw)
    assert len(physical['files']) > 3 and set(physical['tensors']) == set(logical['tensors'])
    for name, t in physical['tensors'].items():
        assert {k: v for k, v in t.items() if k not in ('file', 'offset')} == \
            {k: v for k, v in logical['tensors'][name].items() if k not in ('file', 'offset')}
        data = (out / physical['files'][t['file']]['path']).read_bytes()
        assert hashlib.sha256(data[t['offset']:t['offset'] + t['bytes']]).hexdigest() == t['sha256']
    for f in physical['files'].values():
        data = (out / f['path']).read_bytes()
        assert f['bytes'] == len(data) and f['sha256'] == hashlib.sha256(data).hexdigest()
    assert len({t['file'] for n, t in physical['tensors'].items() if n == 'dit.big'}) == 1  # 2400 B: a file of its own
    logical['tensors']['dit.w0']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='bytes differ'):
        checkpoint.layout(logical, roots, tmp_path / 'bad', shard_bytes=200, workers=2)


def test_prefixes_namespace_components_and_cli_writes_a_pinned_manifest(tmp_path, capsys):
    write_safetensors(tmp_path / 'dit.safetensors', {'blocks.0.w': f32([1.0, 2.0, 3.0])})
    write_safetensors(tmp_path / 't5.safetensors', {'blocks.0.w': f32([4.0, 5.0])})
    out = tmp_path / 'manifest.json'
    checkpoint.main(['manifest', '--root', str(tmp_path), '--file', 'dit.safetensors@dit.', '--file', 't5.safetensors@t5.',
                     '--out', str(out)])
    printed = json.loads(capsys.readouterr().out)
    manifest = json.loads(out.read_text())
    assert set(manifest['tensors']) == {'dit.blocks.0.w', 't5.blocks.0.w'}
    assert printed['sha256'] == hashlib.sha256(out.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='more than one file'):
        checkpoint.build({'checkpoint': str(tmp_path)}, [('checkpoint', 'dit.safetensors'), ('checkpoint', 't5.safetensors')])


def test_audit_finds_a_corrupted_tensor_and_honors_the_manifest_pin(tmp_path):
    write_safetensors(tmp_path / 'm.safetensors', {'a': f32(normals(4, 5000)), 'b': f32(normals(5, 300))})
    manifest = checkpoint.build({'checkpoint': str(tmp_path)}, [('checkpoint', 'm.safetensors', 'x.')])
    digest = checkpoint.write(manifest, tmp_path / 'manifest.json')
    roots = {'checkpoint': str(tmp_path)}
    clean = audit.audit(tmp_path / 'manifest.json', roots, expected_sha256=digest)
    assert clean['checked'] == 2 and not clean['mismatched']
    assert audit.audit(tmp_path / 'manifest.json', roots, prefixes=['x.b'])['checked'] == 1
    data = bytearray((tmp_path / 'm.safetensors').read_bytes())
    a = manifest['tensors']['x.a']
    data[a['offset'] + a['bytes'] // 2] ^= 1  # interior byte: the light guard cannot see it, the audit must
    (tmp_path / 'm.safetensors').write_bytes(bytes(data))
    assert audit.audit(tmp_path / 'manifest.json', roots)['mismatched'] == ['x.a']
    with pytest.raises(ValueError, match='digest mismatch'):
        audit.audit(tmp_path / 'manifest.json', roots, expected_sha256='0' * 64)


def test_this_module_never_imports_torch_in_process():
    assert 'torch' not in sys.modules


def test_checkpoint_prefetch_covers_selected_bytes_without_loading_other_tensors(tmp_path):
    from core_infra.runtime.checkpoint_prefetch import jobs
    write_safetensors(tmp_path / 'm.safetensors', {
        'dit.a': f32(range(4096)), 'dit.b': f32(range(2048)), 'unused.c': f32(range(4096))})
    manifest = checkpoint.build({'checkpoint': str(tmp_path)}, [('checkpoint', 'm.safetensors')])
    pin = checkpoint.write(manifest, tmp_path / 'manifest.json')
    before = (tmp_path / 'm.safetensors').read_bytes()
    ranges = jobs(tmp_path / 'manifest.json', {'checkpoint': str(tmp_path)}, ['dit.'], pin)
    for name, tensor in manifest['tensors'].items():
        if name.startswith('dit.'):
            assert any(r['offset'] <= tensor['offset'] and
                       r['offset'] + r['bytes'] >= tensor['offset'] + tensor['bytes'] for r in ranges)
    assert all(r['offset'] % 4096 == 0 and r['offset'] + r['bytes'] <= len(before) for r in ranges)
    unused = manifest['tensors']['unused.c']
    assert max(r['offset'] + r['bytes'] for r in ranges) <= unused['offset']
    assert (tmp_path / 'm.safetensors').read_bytes() == before
    with pytest.raises(ValueError, match='digest mismatch'):
        jobs(tmp_path / 'manifest.json', {'checkpoint': str(tmp_path)}, expected_sha256='0' * 64)
    manifest['files']['checkpoint:m.safetensors']['path'] = '../outside'
    checkpoint.write(manifest, tmp_path / 'manifest.json')
    with pytest.raises(ValueError, match='escapes'):
        jobs(tmp_path / 'manifest.json', {'checkpoint': str(tmp_path)})
    manifest['files']['checkpoint:m.safetensors']['path'] = 'm.safetensors'
    manifest['tensors']['dit.a']['bytes'] = len(before) + 1
    checkpoint.write(manifest, tmp_path / 'manifest.json')
    with pytest.raises(ValueError, match='extent'):
        jobs(tmp_path / 'manifest.json', {'checkpoint': str(tmp_path)})
