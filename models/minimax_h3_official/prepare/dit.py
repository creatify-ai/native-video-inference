"""DiT package: official parameters plus the timestep tables of the 50-point schedule.

Parameters are copied byte for byte from the Diffusers transformer of the partition
(`transformer/` for FL2VA, `transformer_ref/` for Ref2VA), except the per-block AdaLN
projections. Those only ever see timestep embeddings, so their outputs for every scheduled
timestep are precomputed into tables with the reference Diffusers modules on the GPU, along
with the final-layer modulation. No request activation is read. `prepare_qkv.py` then packs
each attention's Q/K/V rows into one tensor without changing a byte.

Schedule: `linspace(1, 0, 50)` sigma points with shift 12 for video and 3 for audio,
49 model evaluations. A request's rows use the video or audio timestep, the keyframe
augmentation timestep max(t, 0.999) on conditioning video rows, and 1.0 on conditioning
audio rows, so each partition has one table plan per row-timestep set it can produce.
"""
import importlib.util
import json
from pathlib import Path
import struct

from .common import (Checkpoint, PARTITIONS, REPO_ID, REVISION, TRANSFORMERS, extract_all, write_json,
                     write_tensor)

NFE = 49
SHIFTS = {'video': 12.0, 'audio': 3.0}
KEYFRAME_TIMESTEP = 0.999
DENSE_PREFIX_STEPS, SPARSITY = 10, 0.75
NUMERICS = 'sglang-bf16-cudnn-v1'


def weight_names(names):
    """Every transformer tensor except the AdaLN projections, which become tables."""
    return {name: name for name in names if '.adaln_proj.' not in name}


def modules(config):
    eps, qk, final = config['norm_eps'], config['qk_norm_eps'], config['final_norm_eps']
    result = {}
    for prefix, count in (('transformer_blocks', config['num_layers']),
                          ('token_refiner.refiner_blocks', config['num_refiner_layers'])):
        for layer in range(count):
            block = f'{prefix}.{layer}'
            result.update({f'{block}.norm1': {'eps': eps}, f'{block}.attn': {'fused_projections': False},
                           f'{block}.attn.norm_q': {'eps': qk}, f'{block}.attn.norm_k': {'eps': qk},
                           f'{block}.norm2': {'eps': eps}})
    result.update({'token_refiner.final_norm': {'eps': final}, 'norm_out.norm': {'eps': final}})
    return result


def schedule(torch, nfe=NFE):
    base = torch.linspace(1, 0, nfe + 1, dtype=torch.float32)
    sigmas = {name: shift * base / (1 + (shift - 1) * base) for name, shift in SHIFTS.items()}
    return sigmas, {name: 1 - value[:-1] for name, value in sigmas.items()}


def plans(workflow, video, audio):
    """Sorted unique row timesteps per step for every row set the partition produces."""
    anchor = struct.unpack('<f', struct.pack('<f', KEYFRAME_TIMESTEP))[0]  # its FP32 value
    steps = [(float(v), float(a)) for v, a in zip(video, audio)]
    if workflow == 'fl2va':  # keyframe-conditioned, then text-only
        return [[sorted({v, a, max(v, anchor)}) for v, a in steps], [sorted({v, a}) for v, a in steps]]
    # Visual references, then visual and audio references.
    return [[sorted({v, a, max(v, anchor)}) for v, a in steps],
            [sorted({v, a, max(v, anchor), 1.0}) for v, a in steps]]


def prepare_dit(checkpoint, output, *, workflow, workers=8):
    import torch
    import diffusers
    from safetensors import safe_open
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
    if torch.cuda.get_device_capability() != (10, 0):
        raise ValueError('DiT table preparation runs on a B200 (sm_100)')
    checkpoint, output = Path(checkpoint), Path(output)
    folder = checkpoint / TRANSFORMERS[workflow]
    base = output / 'dit-base'
    base.mkdir(parents=True, exist_ok=False)
    source = Checkpoint(checkpoint, TRANSFORMERS[workflow])
    weights = extract_all(source, base, weight_names(source.tensors), workers=workers)
    config = json.loads((folder / 'config.json').read_bytes())

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.set_float32_matmul_precision('highest')
    with torch.device('meta'):
        model = MiniMaxH3Transformer3DModel.from_config(config)
    index = json.loads(next(folder.glob('*.safetensors.index.json')).read_bytes())['weight_map']

    def value(name, dtype):
        with safe_open(str(folder / index[name]), framework='pt', device='cpu') as handle:
            return handle.get_tensor(name).to(device='cuda', dtype=dtype)

    for name, dtype in (('time_embedder', torch.float32), ('norm_out', torch.bfloat16)):
        module = model.get_submodule(name)
        module.load_state_dict({key: value(f'{name}.{key}', dtype) for key in module.state_dict()},
                               strict=True, assign=True)
        module.eval()
    manifest = {'schema': 'h3-native-model-v1', 'status': 'complete', 'config': config,
                'modules': modules(config), 'weights': weights,
                'linear_backend': {'backend': {'allow_tf32': False, 'allow_bf16_reduced_precision_reduction': True,
                                               'torch': str(torch.__version__), 'torch_git': torch.version.git_version,
                                               'cuda': torch.version.cuda}},
                'model_source': {'repo_id': REPO_ID, 'revision': REVISION, 'variant': 'base',
                                 'partition': PARTITIONS[workflow]},
                'recipe': {'workflow': workflow, 'nfe': NFE}, 'numerics': NUMERICS}

    def load_adaln(layer, block):
        prefix = f'transformer_blocks.{layer}.adaln_proj'
        block.adaln_proj.load_state_dict({key: value(f'{prefix}.{key}', torch.bfloat16)
                                          for key in block.adaln_proj.state_dict()}, strict=True, assign=True)

    manifest.update(write_tables(model, base, workflow, config, load_adaln, 'cuda'))
    base_sha = write_json(base / 'manifest.json', manifest)
    packed = _qkv().prepare(base, output / 'dit')
    return {'component': 'dit', 'root': str(output / 'dit'), 'base_root': str(base),
            'base_manifest_sha256': base_sha, 'manifest_sha256': packed['manifest_sha256'],
            'weights': len(weights), 'tables': NFE * len(manifest['adaln']['plans']), 'diffusers': diffusers.__version__,
            'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name()}


def write_tables(model, base, workflow, config, load_adaln, device, nfe=NFE):
    """Schedule, AdaLN tables, final modulations and step records, written under `base`.

    `model` provides the reference modules time_proj, time_embedder, norm_out and
    transformer_blocks[i].adaln_proj; `load_adaln(i, block)` installs block i's projection.
    """
    import torch
    result = {'buffers': {}}
    with torch.inference_mode():
        sigmas, times = schedule(torch, nfe)
        rows = plans(workflow, times['video'], times['audio'])
        timesteps = [[torch.tensor(t, dtype=torch.float32, device=device) for t in plan] for plan in rows]
        embeddings = [[model.time_embedder(model.time_proj(t)) for t in plan] for plan in timesteps]
        maximum = max(len(t) for plan in rows for t in plan) * 3

        def padded(value):
            return torch.cat([value, value.new_zeros(maximum - len(value), value.shape[1])])

        result['adaln'] = {'num_steps': nfe, 'plans': [{'plan_index': i, 'fingerprint': plan}
                                                       for i, plan in enumerate(rows)]}
        result['buffers']['rope.inv_freq'] = write_tensor(base, 'buffers/rope.inv_freq.raw', 1.0 / (
            config['rope_theta'] ** (torch.arange(0, 2 * config['rope_freq_dim'], 2, dtype=torch.float32)
                                     / (2 * config['rope_freq_dim']))))
        for layer, block in enumerate(model.transformer_blocks):
            prefix = f'transformer_blocks.{layer}.adaln_proj'
            load_adaln(layer, block)
            table = torch.stack([padded(torch.cat(block.adaln_proj(embedding), dim=-1))
                                 for plan in embeddings for embedding in plan])
            result['buffers'][prefix + '.table'] = write_tensor(base, f'buffers/{prefix}.table.raw', table)
            block.adaln_proj.to('meta')
            del table
            print(json.dumps({'event': 'dit_table', 'layer': layer}), flush=True)
        result['schedule'] = {f'{name}_{kind}': write_tensor(base, f'schedule/{name}_{kind}.raw', data)
                              for kind, values in (('sigmas', sigmas), ('timesteps', times))
                              for name, data in values.items()}
        result['adaln']['cursor'] = write_tensor(base, 'schedule/initial_adaln_cursor.raw',
                                                 torch.tensor(0, dtype=torch.int64))
        finals = [[write_tensor(base, f'final-plans/{plan}/{step}.raw', model.norm_out.linear(
                       torch.nn.functional.silu(embedding).to(torch.bfloat16)))
                   for step, embedding in enumerate(steps)] for plan, steps in enumerate(embeddings)]
        result['final_modulation_plans'] = [{'plan_index': i, 'steps': steps} for i, steps in enumerate(finals)]
        result['steps'] = [{'index': i, 'inputs': {'timestep': write_tensor(base, f'steps/{i}/timestep.raw', t)},
                            'final_modulation': finals[0][i],
                            'attention': {'dense_prefix_steps': DENSE_PREFIX_STEPS, 'sparsity': SPARSITY}}
                           for i, t in enumerate(timesteps[0])]
    return result


def _qkv():
    path = Path(__file__).resolve().parents[1] / 'prepare_qkv.py'
    spec = importlib.util.spec_from_file_location('h3_prepare_qkv', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
