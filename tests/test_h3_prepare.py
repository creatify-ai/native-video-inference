"""CPU checks of the H3 package producers on synthetic inputs (no GPU, no checkpoint)."""
import hashlib
import json
import math
from pathlib import Path
import struct

import pytest

from models.minimax_h3_official.prepare import common, conditioner, dit, router
from models.minimax_h3_official.prepare.__main__ import COMPONENTS, PACKAGES, checkpoint_files, native_config
from models.minimax_h3_official.public_variant import validate_config

OFFICIAL_CONFIG = {'num_layers': 50, 'num_refiner_layers': 2, 'norm_eps': 1e-5, 'qk_norm_eps': 1e-5,
                   'final_norm_eps': 1e-5, 'rope_theta': 10000.0, 'rope_freq_dim': 16}


def safetensors(path, tensors):
    """Write a minimal safetensors file: {name: (dtype, shape, raw bytes)}."""
    header, offset, payload = {}, 0, b''
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {'dtype': dtype, 'shape': shape, 'data_offsets': [offset, offset + len(raw)]}
        offset += len(raw)
        payload += raw
    encoded = json.dumps(header).encode()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(struct.pack('<Q', len(encoded)) + encoded + payload)


def test_checkpoint_extraction_copies_bytes_and_records_identity(tmp_path):
    bf16 = bytes(range(16))
    fp32 = struct.pack('<4f', 1.0, -2.5, 3.0e-3, 65504.0)
    safetensors(tmp_path / 'ckpt/component/model.safetensors',
                {'a.weight': ('BF16', [2, 4], bf16), 'b.bias': ('F32', [4], fp32)})
    source = common.Checkpoint(tmp_path / 'ckpt', 'component')
    assert source.layout('a.weight') == ('torch.bfloat16', [2, 4])
    specs = common.extract_all(source, tmp_path / 'out', {'a.weight': 'a.weight', 'b.bias': 'b.bias'}, workers=2)
    assert (tmp_path / 'out/weights/a.weight.raw').read_bytes() == bf16
    assert specs['a.weight'] == {'file': 'weights/a.weight.raw', 'dtype': 'torch.bfloat16', 'shape': [2, 4],
                                 'stride': [4, 1], 'bytes': 16, 'sha256': hashlib.sha256(bf16).hexdigest(),
                                 'storage_offset': 0, 'contiguous': True}
    assert specs['b.bias']['dtype'] == 'torch.float32' and specs['b.bias']['bytes'] == 16
    with pytest.raises(ValueError):
        source.extract('a.weight', tmp_path / 'widen.raw', 'torch.float32')
    with pytest.raises(FileExistsError):
        source.extract('a.weight', tmp_path / 'out/weights/a.weight.raw')


def test_fp32_storage_narrows_to_bf16_with_round_to_nearest_even(tmp_path):
    pytest.importorskip('numpy')
    fp32 = struct.pack('<4f', 1.0, -2.5, 3.0e-3, 65504.0)
    safetensors(tmp_path / 'ckpt/component/model.safetensors', {'b.bias': ('F32', [4], fp32)})
    source = common.Checkpoint(tmp_path / 'ckpt', 'component')
    narrowed = tmp_path / 'narrow.raw'
    assert source.extract('b.bias', narrowed, 'torch.bfloat16') == ('torch.bfloat16', [4])
    assert narrowed.read_bytes()[:2] == struct.pack('<f', 1.0)[2:]


def test_checkpoint_file_pins_cover_each_partition():
    for workflow, transformer in (('fl2va', 'transformer/'), ('ref2va', 'transformer_ref/')):
        files = checkpoint_files(workflow)
        shards = [name for name in files if name.endswith('.safetensors')]
        assert sum(name.startswith(transformer) for name in shards) == 14
        assert sum(name.startswith('text_encoder/') for name in shards) == 14
        assert {transformer + 'config.json', 'audio_vae/config.json', 'processor/preprocessor_config.json'} <= set(files)
        assert not any(name.startswith(('transformer_ref/' if workflow == 'fl2va' else 'transformer/')) for name in files)
        assert all(len(sha) == 64 for sha in files.values())


def test_conditioner_keeps_the_layers_that_reach_hidden_state_fifty():
    names = ['lm_head.weight', 'model.language_model.norm.weight', 'model.language_model.embed_tokens.weight',
             'model.visual.patch_embed.proj.weight']
    names += [f'model.language_model.layers.{i}.mlp.up_proj.weight' for i in range(64)]
    selected = conditioner.weight_names(names)
    assert selected['visual.patch_embed.proj.weight'] == 'model.visual.patch_embed.proj.weight'
    assert 'language_model.embed_tokens.weight' in selected
    assert sorted(int(n.split('.')[2]) for n in selected if '.layers.' in n) == list(range(50))
    assert 'lm_head.weight' not in selected and 'language_model.norm.weight' not in selected
    with pytest.raises(ValueError):
        conditioner.weight_names(['model.unknown.weight'])


def test_conditioner_rotary_buffers_match_the_official_formula():
    torch = pytest.importorskip('torch')
    config = {'text_config': {'head_dim': 128, 'rope_theta': 5000000},
              'vision_config': {'hidden_size': 1152, 'num_heads': 16}}
    buffers = conditioner.rotary_buffers(config, 'cpu')
    text = buffers['language_model.rotary_emb.inv_freq']
    assert text.shape == (64,) and buffers['visual.rotary_pos_emb.inv_freq'].shape == (18,)
    assert hashlib.sha256(text.numpy().tobytes()).hexdigest() == conditioner.TEXT_INV_FREQ_SHA256
    assert text.dtype == torch.float32


def test_dit_tables_replace_only_the_adaln_projections():
    names = ['proj_in.weight', 'transformer_blocks.0.adaln_proj.linear.weight',
             'transformer_blocks.0.attn.to_q.weight', 'norm_out.linear.weight']
    assert dit.weight_names(names) == {n: n for n in names if 'adaln_proj' not in n}
    layout = dit.modules(OFFICIAL_CONFIG)
    assert len(layout) == 262
    assert layout['transformer_blocks.49.attn'] == {'fused_projections': False}
    assert layout['token_refiner.refiner_blocks.1.attn.norm_k'] == {'eps': 1e-5}
    assert layout['norm_out.norm'] == layout['token_refiner.final_norm'] == {'eps': 1e-5}


def test_dit_schedule_and_plans_cover_each_partition_row_set():
    torch = pytest.importorskip('torch')
    sigmas, times = dit.schedule(torch)
    assert sigmas['video'].shape == (50,) and times['audio'].shape == (49,)
    assert float(sigmas['video'][0]) == 1.0 and float(sigmas['audio'][-1]) == 0.0
    assert all(float(a) <= float(b) for a, b in zip(times['video'][:-1], times['video'][1:]))
    anchor = struct.unpack('<f', struct.pack('<f', 0.999))[0]
    for workflow, extra in (('fl2va', None), ('ref2va', 1.0)):
        first, second = dit.plans(workflow, times['video'], times['audio'])
        assert len(first) == len(second) == 49
        for step, (v, a) in enumerate(zip(times['video'].tolist(), times['audio'].tolist())):
            assert first[step] == sorted({v, a, max(v, anchor)})
            assert second[step] == (sorted({v, a}) if extra is None else sorted({v, a, max(v, anchor), 1.0}))


def rust_variants(sequence, heads=56):
    """transformer/dynamic_router.rs `variants()` for the single-GPU head count."""
    g = -(-sequence // 64)
    kept = min(g, (-(-g // 4) + 7) // 8 * 8)
    spec = lambda values: ''.join('o' if v == 1 else 'a' if v % 16 == 0 else 'x' for v in values)
    pool = spec([sequence * heads * 128, heads * 128, 128, g * 512, 128, sequence, heads])
    score = spec([128, g * 512, 128, g * 512, g, g * g, g * 4, -(-sequence // 16), -(-sequence // 16), g, g])
    block, warps = 1 << (g - 1).bit_length(), 4 if g >= 1024 else 2
    return {f'pool_q-{pool}', f'pool_k-{pool}', f'scores-{score}', f'topk-b{block}-w{warps}-{spec([g, kept])}'}


def test_router_inventory_has_every_variant_the_native_loader_selects():
    inventory = router.variant_inventory(4096)
    labels = {f'{label}-{key}' for family, key in inventory for label in (
        ('pool_q', 'pool_k') if family == 'pool' else (family,))}
    for sequence in (1, 2, 63, 64, 65, 1000, 4095, 4096):
        assert rust_variants(sequence) <= labels, sequence
    assert all(1 <= s <= 4096 for s in inventory.values())


def test_router_launch_abi_and_source_pin(tmp_path):
    ptx = ('.visible .entry topk_kernel(\n .param .u64 .ptr .global .align 1 topk_kernel_param_0,\n'
           ' .param .u64 .ptr .global .align 1 topk_kernel_param_1,\n .param .u32 topk_kernel_param_2,\n'
           ' .param .u64 .ptr .global .align 1 topk_kernel_param_3\n)\n{\n}')
    params = router.parameters(ptx, 'topk_kernel')
    assert [p['kind'] for p in params] == ['u64', 'u64', 'u32', 'u64']
    args = router.launch_arguments('topk', ['S', 'OUT', 'G', 'K', 'BLK', 'ITERS'],
                                   {'S': '*fp32', 'OUT': '*i32', 'G': 'i32'}, {'K': 1, 'BLK': 4, 'ITERS': 16}, params)
    assert args == [('ptr', 'scores'), ('ptr', 'indices'), ('dynamic', 'G'), ('bits', '0')]
    copy = tmp_path / 'kernels.py'
    copy.write_text('# not the pinned source\n')
    with pytest.raises(ValueError, match='pinned source'):
        router.fetch_kernels(copy)


def prepared_tree(root, workflow):
    for name in PACKAGES[workflow]:
        marker = root / name / {'conditioner': 'model-manifest.json', 'router': 'launch.tsv', 'video_vae': 'model.json',
                                'video_encoder': 'program.json'}.get(name, 'manifest.json')
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text('{}')
    (root / 'dit/manifest.json').write_text(json.dumps({'model_source': {
        'repo_id': common.REPO_ID, 'revision': common.REVISION, 'variant': 'base',
        'partition': common.PARTITIONS[workflow]}, 'recipe': {'workflow': workflow, 'nfe': 49}}))


@pytest.mark.parametrize('workflow', ['fl2va', 'ref2va'])
def test_native_config_is_admitted_by_the_public_variant(tmp_path, workflow):
    prepared_tree(tmp_path, workflow)
    config = native_config(tmp_path, workflow, '/opt/h3/bin', '/opt/h3/lib/libh3_dense.so',
                           media={'format': 'mp4', 'capture_raw': True, 'hash_outputs': True})
    assert validate_config(config)['recipe']['nfe'] == 49
    assert config['programs']['dit'] == '/opt/h3/bin/h3-official-model-run'
    assert ('audio_encoder' in config['programs']) == (workflow == 'ref2va')
    (tmp_path / 'router/launch.tsv').unlink()
    with pytest.raises(FileNotFoundError, match='router'):
        native_config(tmp_path, workflow, '/opt/h3/bin', '/x.so')
    assert COMPONENTS == ('conditioner', 'dit', 'router', 'video_vae', 'audio_vae')


def test_written_tensor_specs_are_contiguous_and_hashed(tmp_path):
    torch = pytest.importorskip('torch')
    value = torch.arange(6, dtype=torch.float32).reshape(2, 3).t()  # non-contiguous view
    spec = common.write_tensor(tmp_path, 'x/value.raw', value)
    raw = (tmp_path / 'x/value.raw').read_bytes()
    assert spec['shape'] == [3, 2] and spec['stride'] == [2, 1] and spec['bytes'] == 24
    assert struct.unpack('<6f', raw) == (0.0, 3.0, 1.0, 4.0, 2.0, 5.0)
    assert spec['sha256'] == hashlib.sha256(raw).hexdigest() and math.prod(spec['shape']) * 4 == len(raw)


def test_dit_tables_and_qkv_packing_build_a_complete_package(tmp_path):
    torch = pytest.importorskip('torch')
    torch.manual_seed(0)

    class AdaLN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(8, 6, dtype=torch.bfloat16)

        def forward(self, x):
            return self.linear(torch.nn.functional.silu(x).to(torch.bfloat16)).chunk(2, dim=-1)

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.adaln_proj = AdaLN()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.time_embedder = torch.nn.Linear(4, 8)
            self.norm_out = torch.nn.Module()
            self.norm_out.linear = torch.nn.Linear(8, 6, dtype=torch.bfloat16)
            self.transformer_blocks = torch.nn.ModuleList([Block(), Block()])

        def time_proj(self, t):
            return torch.stack([t, t * 2, t.sin(), t.cos()], dim=-1)

    base = tmp_path / 'dit-base'
    loaded = []
    tables = dit.write_tables(Model(), base, 'ref2va', OFFICIAL_CONFIG, lambda i, b: loaded.append(i), 'cpu', nfe=3)
    assert loaded == [0, 1] and len(tables['steps']) == 3 and tables['adaln']['num_steps'] == 3
    assert [p['plan_index'] for p in tables['adaln']['plans']] == [0, 1]
    table = tables['buffers']['transformer_blocks.1.adaln_proj.table']
    assert table['shape'] == [6, 12, 6] and table['dtype'] == 'torch.bfloat16'  # 2 plans x 3 steps, 4 rows x 3
    assert tables['buffers']['rope.inv_freq']['shape'] == [16]
    assert set(tables['schedule']) == {'video_sigmas', 'audio_sigmas', 'video_timesteps', 'audio_timesteps'}
    assert tables['steps'][0]['final_modulation'] == tables['final_modulation_plans'][0]['steps'][0]
    assert tables['steps'][2]['attention'] == {'dense_prefix_steps': 10, 'sparsity': 0.75}
    # Weights for one attention module, then the packing step used for the real package.
    weights = {}
    for part in 'qkv':
        raw = bytes([ord(part)]) * 16
        path = base / f'weights/transformer_blocks.0.attn.to_{part}.weight.raw'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        weights[f'transformer_blocks.0.attn.to_{part}.weight'] = common.specification(path, base, 'torch.bfloat16', [2, 4])
    manifest = {'schema': 'h3-native-model-v1', 'status': 'complete', 'weights': weights,
                'modules': {'transformer_blocks.0.attn': {'fused_projections': False}},
                'model_source': {'repo_id': common.REPO_ID, 'revision': common.REVISION, 'variant': 'base',
                                 'partition': 'Ref2VA'},
                'recipe': {'workflow': 'ref2va', 'nfe': 3}, 'numerics': dit.NUMERICS, **tables}
    common.write_json(base / 'manifest.json', manifest)
    report = dit._qkv().prepare(base, tmp_path / 'dit')
    packed = json.loads((tmp_path / 'dit/manifest.json').read_bytes())
    fused = packed['weights']['transformer_blocks.0.attn.to_qkv.weight']
    assert (tmp_path / 'dit' / fused['file']).read_bytes() == b'q' * 16 + b'k' * 16 + b'v' * 16
    assert packed['modules']['transformer_blocks.0.attn']['fused_projections'] is True
    assert (tmp_path / 'dit/schedule').is_symlink() and len(packed['steps']) == packed['recipe']['nfe']
    assert report['manifest_sha256'] == hashlib.sha256((tmp_path / 'dit/manifest.json').read_bytes()).hexdigest()


def test_light_guards_cover_nested_payloads_and_reject_changed_bytes(tmp_path):
    from models.minimax_h3_official.prepare import light_guards
    head, tail = b'h' * 2000, b't' * 2000
    weight, small = head + tail, b'tiny'
    for path, data in (('conditioner/weights/w.raw', weight), ('dit/steps/0/t.raw', small)):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_bytes(data)
    spec = lambda file, data: {'file': file, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    (tmp_path / 'conditioner/model-manifest.json').write_text(json.dumps({'weights': {'w': spec('weights/w.raw', weight)}}))
    (tmp_path / 'dit/manifest.json').write_text(json.dumps({'steps': [{'inputs': {'t': spec('steps/0/t.raw', small)}}]}))
    digest = light_guards.write_light_guards(tmp_path, ('conditioner', 'dit', 'router'), workers=2)
    raw = (tmp_path / light_guards.FILENAME).read_bytes()
    table = json.loads(raw)
    assert hashlib.sha256(raw).hexdigest() == digest
    assert (table['schema'], table['segment_bytes'], table['full_sha256_verified']) == ('h3-light-sha-v1', 1024, True)
    assert table['entries'][hashlib.sha256(weight).hexdigest()] == {
        'bytes': len(weight), 'head_sha256': hashlib.sha256(head[:1024]).hexdigest(),
        'tail_sha256': hashlib.sha256(tail[-1024:]).hexdigest()}
    entry = table['entries'][hashlib.sha256(small).hexdigest()]
    assert entry['head_sha256'] == entry['tail_sha256'] == hashlib.sha256(small).hexdigest()
    (tmp_path / 'dit/steps/0/t.raw').write_bytes(b'TINY')
    with pytest.raises(ValueError, match='digest differs'):
        light_guards.write_light_guards(tmp_path, ('dit',), workers=1)


def test_both_workflows_link_one_shared_conditioner(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from models.minimax_h3_official.prepare import __main__ as cli
    built = []

    def fake_step(name, checkpoint, out, workflow, args):
        built.append((name, out))
        (out / name / 'weights').mkdir(parents=True)
        (out / name / 'model-manifest.json').write_text('{}')
        return {'component': name, 'root': str(out / name)}
    monkeypatch.setattr(cli, 'step', fake_step)
    (tmp_path / 'checkpoint').mkdir()
    shared = tmp_path / 'prepared/shared'
    for workflow in ('fl2va', 'ref2va'):
        (tmp_path / 'prepared' / workflow / 'conditioner').mkdir(parents=True)  # an interrupted attempt
        report = cli.run(SimpleNamespace(checkpoint=str(tmp_path / 'checkpoint'), workflow=workflow,
                                         out=str(tmp_path / 'prepared' / workflow), shared=str(shared),
                                         components=['conditioner'], verify_checkpoint=False, workers=1,
                                         programs=None))
        link = tmp_path / 'prepared' / workflow / 'conditioner'
        assert link.is_symlink() and (link / 'model-manifest.json').is_file()
        assert link.resolve() == (shared / 'conditioner').resolve()
        assert report['components']['conditioner']['shared'] == str(shared / 'conditioner')
    assert built == [('conditioner', shared)]


def test_cp_slices_rebuild_every_rank_shard_and_pin_the_two_gpu_config(tmp_path):
    from models.minimax_h3_official.prepare import light_guards
    from models.minimax_h3_official.prepare.__main__ import context_parallel
    prepared_tree(tmp_path, 'fl2va')
    sizes = {'transformer_blocks.0.attn.to_q.weight': 600, 'transformer_blocks.0.ff.bias': 70,
             'token_refiner.refiner_blocks.1.mlp.weight': 10, 'x_embedder.weight': 3000}
    weights = {}
    for index, (name, size) in enumerate(sizes.items()):
        data = bytes((index * 31 + i) % 251 + 1 for i in range(size))
        (tmp_path / 'dit/weights').mkdir(parents=True, exist_ok=True)
        (tmp_path / f'dit/weights/{name}.raw').write_bytes(data)
        weights[name] = {'file': f'weights/{name}.raw', 'dtype': 'torch.uint8', 'shape': [size], 'stride': [1],
                         'bytes': size, 'sha256': hashlib.sha256(data).hexdigest(), 'storage_offset': 0, 'contiguous': True}
    model = json.loads((tmp_path / 'dit/manifest.json').read_bytes())
    (tmp_path / 'dit/manifest.json').write_text(json.dumps({**model, 'weights': weights}))
    light_sha256 = light_guards.write_light_guards(tmp_path, ('dit',), workers=1)
    (tmp_path / 'preparation.json').write_text(json.dumps({'light_sha256': light_sha256}))
    with pytest.raises(FileNotFoundError, match='2-rank'):
        native_config(tmp_path, 'fl2va', '/opt/h3/bin', '/x.so', world=2)

    result = context_parallel(tmp_path, 2, workers=2)
    package = tmp_path / 'dit-cp2'
    manifest = json.loads((package / 'manifest.json').read_bytes())
    slices = json.loads((package / 'slices.json').read_bytes())
    table = json.loads((tmp_path / 'light-sha-cp2.json').read_bytes())
    assert result['groups'] == 3 and [g['name'] for g in manifest['groups']] == [
        'global', 'token_refiner.refiner_blocks.1', 'transformer_blocks.0']
    assert manifest['base_weights_manifest_sha256'] == hashlib.sha256((tmp_path / 'dit/manifest.json').read_bytes()).hexdigest()
    assert slices['shard_manifest_sha256'] == hashlib.sha256((package / 'manifest.json').read_bytes()).hexdigest()
    assert set(json.loads((tmp_path / 'light-sha.json').read_bytes())['entries']) < set(table['entries'])
    for index, group in enumerate(manifest['groups']):
        assert group['bytes'] % 512 == 0 and all(e['offset'] % 256 == 0 for e in group['entries'])
        shards = []
        for rank, row in enumerate(group['ranks']):
            shard = bytearray(row['bytes'])  # what a rank uploads: its slices over zeros
            for piece in (p for p in slices['ranks'][rank] if p['group'] == index):
                with open(tmp_path / 'dit' / piece['file'], 'rb') as stream:
                    stream.seek(piece['offset'])
                    data = stream.read(piece['bytes'])
                assert hashlib.sha256(data).hexdigest() == piece['sha256'] and table['entries'][piece['sha256']] == {
                    'bytes': len(data), 'head_sha256': hashlib.sha256(data[:1024]).hexdigest(),
                    'tail_sha256': hashlib.sha256(data[-1024:]).hexdigest()}
                shard[piece['dest']:piece['dest'] + piece['bytes']] = data
            assert hashlib.sha256(shard).hexdigest() == row['sha256']
            shards.append(bytes(shard))
        assert hashlib.sha256(b''.join(shards)).hexdigest() == group['sha256']
    # to_q (600 bytes) straddles the two 512-byte halves of transformer_blocks.0.
    assert [(p['offset'], p['bytes']) for p in slices['ranks'][1] if p['group'] == 2] == [(512, 88), (0, 70)]

    config = native_config(tmp_path, 'fl2va', '/opt/h3/bin', '/x.so', world=2)
    parallel = config['runtime']['context_parallel']
    assert config['programs']['cp_coordinator'] == '/opt/h3/bin/h3-official-cp-coordinator'
    assert parallel == {'world': 2, 'shards': str(package), 'manifest_sha256': slices['shard_manifest_sha256'],
                        'slices_sha256': hashlib.sha256((package / 'slices.json').read_bytes()).hexdigest()}
    assert config['runtime']['light_sha']['path'] == str(tmp_path / 'light-sha-cp2.json')
    validate_config(config)
    for broken in ({**parallel, 'world': 3}, {**parallel, 'slices_sha256': 'x'}, {**parallel, 'extra': 1}):
        with pytest.raises(ValueError):
            validate_config({**config, 'runtime': {**config['runtime'], 'context_parallel': broken}})
