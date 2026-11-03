"""Pack official base Q/K/V weight rows without changing tensor values."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def concatenate(specs, payloads, filename):
    """Concatenate complete contiguous output rows; no floating-point work."""
    if len(specs) != 3 or len(payloads) != 3:
        raise ValueError('QKV packing requires exactly three tensors')
    first = specs[0]
    if first['dtype'] not in ('torch.bfloat16', 'torch.float32'):
        raise ValueError('unsupported QKV storage dtype')
    width = 2 if first['dtype'] == 'torch.bfloat16' else 4
    for spec, raw in zip(specs, payloads):
        if (spec['dtype'] != first['dtype'] or spec['shape'] != first['shape']
                or len(spec['shape']) not in (1, 2) or not spec['contiguous'] or spec['storage_offset'] != 0):
            raise ValueError('QKV source tensors have incompatible layouts')
        stride = 1
        for dim, actual in zip(reversed(spec['shape']), reversed(spec['stride'])):
            if actual != stride:
                raise ValueError('QKV source tensor is not contiguous')
            stride *= dim
        if spec['bytes'] != stride * width or len(raw) != spec['bytes']:
            raise ValueError('QKV source tensor byte size differs')
        if hashlib.sha256(raw).hexdigest() != spec['sha256']:
            raise ValueError('QKV source tensor digest differs')
    raw = b''.join(payloads)
    result = copy.deepcopy(first)
    result.update(file=filename, shape=[first['shape'][0] * 3, *first['shape'][1:]], bytes=len(raw),
                  sha256=hashlib.sha256(raw).hexdigest())
    return result, raw


def prepare(source, destination):
    source, destination = Path(source).resolve(), Path(destination).absolute()
    parent_raw = (source / 'manifest.json').read_bytes()
    model = json.loads(parent_raw)
    identity = model.get('model_source', {})
    if (identity.get('repo_id') != 'MiniMaxAI/MiniMax-H3' or identity.get('variant') != 'base'
            or model.get('numerics') != 'sglang-bf16-cudnn-v1'):
        raise ValueError('QKV packing requires the prepared official BF16 base variant')
    shared = model.pop('shared_weights', None)
    weight_root = source
    if shared:
        weight_root = Path(shared['root'])
        raw = (weight_root / 'manifest.json').read_bytes()
        if (hashlib.sha256(raw).hexdigest() != shared['manifest_sha256']
                or json.loads(raw)['weights'] != model['weights']):
            raise ValueError('shared source weights identity changed')
    modules = [name for name, module in model['modules'].items()
               if name.endswith('.attn') and module.get('fused_projections') is False]
    if not modules:
        raise ValueError('no separate QKV projections in source model')
    destination.mkdir(parents=True, exist_ok=False)
    # Reuse immutable timestep/rotary constants. None are reference activations.
    for path in source.iterdir():
        if path.name not in ('manifest.json', 'preparation.json', 'weights', 'fused-weights'):
            (destination / path.name).symlink_to(path)
    records = []
    for name in modules:
        for suffix in ('weight', 'bias'):
            keys = [f'{name}.to_{part}.{suffix}' for part in ('q', 'k', 'v')]
            present = [key in model['weights'] for key in keys]
            if suffix == 'bias' and not any(present):
                continue
            if not all(present):
                raise ValueError('incomplete QKV projection: ' + name)
            specs = [model['weights'][key] for key in keys]
            payloads = [(weight_root / spec['file']).read_bytes() for spec in specs]
            target = f'fused-weights/{name}.to_qkv.{suffix}.raw'
            spec, raw = concatenate(specs, payloads, target)
            path = destination / target
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            for key in keys:
                del model['weights'][key]
            model['weights'][f'{name}.to_qkv.{suffix}'] = spec
            records.append(dict(name=f'{name}.to_qkv.{suffix}', sha256=spec['sha256'],
                                source_sha256=[part['sha256'] for part in specs]))
        model['modules'][name]['fused_projections'] = True
        print(json.dumps(dict(event='packed_qkv', module=name)), flush=True)
    for spec in model['weights'].values():
        target = destination / spec['file']
        if spec['file'].startswith('fused-weights/'):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.symlink_to(weight_root / spec['file'])
    model['qkv_projection_pack'] = dict(source_manifest_sha256=hashlib.sha256(parent_raw).hexdigest(),
                                       operation='concatenate Q,K,V output rows byte-for-byte', tensors=records)
    raw = (json.dumps(model, indent=2) + '\n').encode()
    (destination / 'manifest.json').write_bytes(raw)
    report = dict(schema='h3-public-qkv-preparation-v1', passed=True, root=str(destination),
                  source_root=str(source), model_source=identity, fused_attention_modules=len(modules),
                  manifest_sha256=hashlib.sha256(raw).hexdigest(), reference_activations_used=False,
                  full_h3_latent_accepted=False, packed_tensors=records)
    (destination / 'preparation.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.destination)), flush=True)
