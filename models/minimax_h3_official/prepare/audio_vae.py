"""Audio decoder (and Ref2VA reference encoder) graphs of the official MiniMax-H3 audio VAE.

Source: the root-level Diffusers folder `audio_vae/` (`AutoencoderKLMiniMaxH3Audio`, FP32). Its
calls run once under `aten_capture.Capture`:

    audio_vae/manifest.json        h3-audio-vae-graph-v1, Diffusers' `MiniMaxH3AudioDecodeStep`:
                                   normalized FP32 [2, 32, T] -> [1, 2, 800 T], ending in the stock
                                   clamp to [-1, 1] and the stereo permute
    audio_encoder/manifest.json    Ref2VA only, h3-audio-encoder-graph-v1, the reference audio
                                   encoder `encode(x).latent_dist.mode()`: FP32 [2, 1, N] -> [2, 32, N / 800]
    */constants/<name>.raw         model-state tensors the graph reads (weight-normalized weights,
                                   resampling filters, Snake parameters, latent statistics)

One template graph serves every length: the native runtime re-derives each node's shape for the
requested T (`select_tokens`) or N (`select_encoder_samples`) from operator semantics, so one
capture at any admissible length suffices. The encoder template length is a multiple of the
800-sample hop, so it carries no padding operator (the runtime pads requests itself). Capture
inputs are synthetic; manifests carry no reference activations: input, gold and node outputs
record shape and dtype only, with blank digests.
"""
import collections
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import uuid

from .common import DIFFUSERS_COMMIT, REPO_ID, REVISION, Checkpoint, digest, write_json

COMPONENT = 'audio_vae'
MODEL_CLASS = 'AutoencoderKLMiniMaxH3Audio'
DECODER_SCHEMA, ENCODER_SCHEMA = 'h3-audio-vae-graph-v1', 'h3-audio-encoder-graph-v1'
DECODER_NODES = 1621  # the native template loader admits exactly this architecture
DECODER_TOKENS = 207  # template length; any T >= 1 records the same graph
ENCODER_SAMPLES = 32000  # template length; a multiple of HOP, at most 30 s
HOP, CHANNELS, STEREO = 800, 32, 2
UNREFERENCED = 'not-a-reference.raw'
# Operators the native decoder executes and shape-infers (lib.rs, shape_rules.rs).
DECODER_OPS = frozenset({
    'aten.convolution.default', 'aten.replication_pad1d.default', 'aten.slice.Tensor', 'aten.permute.default',
    'aten.view.default', 'aten.reshape.default', 'aten.unsqueeze.default', 'aten.squeeze.dim',
    'aten.mul.Tensor', 'aten.mul.Scalar', 'aten.add.Tensor', 'aten.add.Scalar', 'aten.div.Tensor',
    'aten.div.Scalar', 'aten.sin.default', 'aten.pow.Tensor_Scalar', 'aten.clamp.default',
    'aten.clone.default', 'aten._to_copy.default', 'aten.detach.default', 'aten.alias.default'})
# Plus the encoder's own operators (encoder.rs). A padding operator would pin the template length.
ENCODER_OPS = DECODER_OPS | frozenset({
    'aten.transpose.int', 'aten.unbind.int', 'aten.native_layer_norm.default', 'aten.addmm.default',
    'aten._scaled_dot_product_efficient_attention.default', 'aten.mean.dim', 'aten._adaptive_avg_pool2d.default',
    'aten.gelu.default'})


def _official():
    """The pinned Diffusers implementation: model class, decode step and pipeline state."""
    from diffusers import AutoencoderKLMiniMaxH3Audio
    from diffusers.modular_pipelines.minimax_h3.decoders import MiniMaxH3AudioDecodeStep
    from diffusers.modular_pipelines.modular_pipeline import PipelineState
    return AutoencoderKLMiniMaxH3Audio, MiniMaxH3AudioDecodeStep, PipelineState


def _source_sha(value):
    try:
        return digest(inspect.getsourcefile(value))
    except (TypeError, OSError):
        return None


def _diffusers():
    from importlib import metadata
    try:
        distribution = metadata.distribution('diffusers')
    except metadata.PackageNotFoundError:
        return {'version': None, 'commit': None}
    origin = json.loads(distribution.read_text('direct_url.json') or '{}')
    return {'version': distribution.version, 'commit': origin.get('vcs_info', {}).get('commit_id')}


def _environment(torch, device):
    cuda = device.type == 'cuda'
    return {
        'torch': torch.__version__, 'torch_git': torch.version.git_version, 'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version() if cuda else None, 'device': device.type,
        'gpu': torch.cuda.get_device_name(device) if cuda else None,
        'capability': list(torch.cuda.get_device_capability(device)) if cuda else None,
        # Read by the native convolution and projection plans.
        'cudnn_allow_tf32': bool(torch.backends.cudnn.allow_tf32),
        'cudnn_deterministic': bool(torch.backends.cudnn.deterministic),
        'cudnn_benchmark': bool(torch.backends.cudnn.benchmark),
        'allow_tf32': bool(torch.backends.cuda.matmul.allow_tf32)}


def _verify_state(model, checkpoint, torch):
    """Every parameter and buffer the model holds is its checkpoint tensor, byte for byte."""
    state = model.state_dict()
    missing = sorted(set(state) - set(checkpoint.tensors))
    if missing:
        raise ValueError(f'audio VAE tensors absent from the checkpoint: {missing[:4]}')
    for name, value in state.items():
        dtype, shape = checkpoint.layout(name)
        if dtype != 'torch.float32' or str(value.dtype) != dtype or list(value.shape) != shape:
            raise ValueError(f'audio VAE tensor {name} is not the FP32 checkpoint tensor')
        if value.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes() != checkpoint.read(name):
            raise ValueError(f'audio VAE tensor {name} differs from its checkpoint bytes')
    return {'verified': len(state), 'unused': sorted(set(checkpoint.tensors) - set(state))}


def _check(graph, kind, output_shape):
    """Reject what the native runtime cannot execute, before anything is published."""
    decoder = kind == 'decoder'
    ops, rank = (DECODER_OPS, 3) if decoder else (ENCODER_OPS, 5)
    for name, (value, _, _) in graph['constants'].items():
        if str(value.dtype) != 'torch.float32' or not 1 <= value.dim() <= rank or 0 in value.shape:
            raise ValueError(f'constant {name} {list(value.shape)} {value.dtype} is not native-loadable')
    for index, node in enumerate(graph['nodes']):
        op, args, out = node['op'], node['args'], node['output']
        if node['id'] != f'n{index:04}' or out['dtype'] != 'torch.float32' or not 1 <= len(out['shape']) <= rank \
                or 0 in out['shape']:
            raise ValueError(f'{node["id"]} {op}: output {out["shape"]} {out["dtype"]} is not native-executable')
        if op not in ops:
            raise ValueError(f'{node["id"]}: native {kind} has no operator {op}')
        scalar = len(args) > 1 and isinstance(args[1], (int, float)) and not isinstance(args[1], bool)
        if ((op.startswith('aten.div') and not scalar)
                or (op == 'aten.pow.Tensor_Scalar' and args[1] != 2)
                or (op == 'aten.clamp.default' and args[1:] != [-1.0, 1.0])
                or (op.startswith('aten.add') and node['kwargs'].get('alpha', 1) != 1)
                or (op == 'aten.gelu.default' and node['kwargs'].get('approximate') != 'tanh')
                or (op == 'aten._scaled_dot_product_efficient_attention.default'
                    and (args[3] is not None or (args[5:6] or [0.0])[0] != 0.0 or args[6:7] != [True]))):
            raise ValueError(f'{node["id"]} {op}: arguments {args[1:]} {node["kwargs"]} outside the native contract')
    shapes = {node['id']: node['output']['shape'] for node in graph['nodes']}
    if shapes[graph['output_id']] != list(output_shape):
        raise ValueError(f'{kind} output {shapes[graph["output_id"]]} differs from {list(output_shape)}')
    if decoder:
        if DECODER_NODES is not None and len(graph['nodes']) != DECODER_NODES:
            raise ValueError(f'decoder graph has {len(graph["nodes"])} nodes; the native template admits {DECODER_NODES}')
        clamp, last = ([None] + graph['nodes'])[-2:]
        if (clamp is None or graph['output_id'] != last['id'] or last['op'] != 'aten.permute.default'
                or last['args'] != [{'tensor': clamp['id']}, [1, 0, 2]] or clamp['op'] != 'aten.clamp.default'
                or sum(node['op'] == 'aten.clamp.default' for node in graph['nodes']) != 1):
            raise ValueError('decoder must end with the stock clamp to [-1, 1] and the stereo permute')


def _stage(directory, name, package):
    """Write one package's constants and manifest into `directory`; return its report entry."""
    from .aten_capture import write_constants
    graph = package['graph']
    blank = lambda shape: {'file': UNREFERENCED, 'dtype': 'torch.float32', 'shape': list(shape),
                           'bytes': 4 * math.prod(shape), 'sha256': ''}
    directory.mkdir(parents=True)
    constants = write_constants(directory, graph['constants'])
    nodes = [{'id': node['id'], 'op': node['op'], 'args': node['args'], 'kwargs': node['kwargs'],
              'output': blank(node['output']['shape'])} for node in graph['nodes']]
    manifest = {'schema': package['schema'], 'status': 'complete', 'input': blank(package['input_shape']),
                'output_id': graph['output_id'], 'gold': blank(package['output_shape']), 'constants': constants,
                'nodes': nodes, 'provenance': package['provenance'], **package['extra']}
    manifest_sha = write_json(directory / 'manifest.json', manifest)
    return {'root': name, 'schema': package['schema'], 'manifest_sha256': manifest_sha, 'nodes': len(nodes),
            'constants': len(constants), 'folded_constants': sum(spec['folded'] for spec in constants.values()),
            'constant_bytes': sum(spec['bytes'] for spec in constants.values()),
            'pruned_nodes': graph['pruned_nodes'], 'input_shape': list(package['input_shape']),
            'output_shape': list(package['output_shape']),
            'ops': dict(sorted(collections.Counter(node['op'] for node in nodes).items())),
            'files': {f'{name}/manifest.json': manifest_sha,
                      **{f'{name}/{spec["file"]}': spec['sha256'] for spec in constants.values()}}}


def _publish(output, packages):
    """Stage every package, then move them into place together; nothing partial is left behind."""
    staged, reports = {}, {}
    try:
        for name, package in packages.items():
            staged[name] = output / f'.{name}.{uuid.uuid4().hex}.partial'
            reports[name] = _stage(staged[name], name, package)
        for name, directory in staged.items():
            os.replace(directory, output / name)
    except BaseException:
        for directory in staged.values():
            shutil.rmtree(directory, ignore_errors=True)
        raise
    return reports


def _synthetic(torch, shape, device, scale=1.0):
    generator = torch.Generator().manual_seed(0)
    return (scale * torch.randn(shape, generator=generator, dtype=torch.float32)).to(device)


def _state(model):
    return {**dict(model.named_parameters()), **dict(model.named_buffers())}


def capture_decoder(model, decode_step, pipeline_state, config, *, tokens, provenance):
    """`MiniMaxH3AudioDecodeStep` on synthetic normalized latents, checked against the native contract."""
    import types
    import torch
    from .aten_capture import Capture
    device = next(model.parameters()).device
    latents = _synthetic(torch, (STEREO, CHANNELS, tokens), device)
    # The step builds these from the config with torch.tensor(); they are the only admitted literals.
    literals = {name: torch.tensor(config[name], dtype=torch.float32) for name in ('latents_mean', 'latents_std')}
    components = types.SimpleNamespace(audio_vae=model, audio_sampling_rate=config['sampling_rate'],
                                       _execution_device=device)
    state = pipeline_state(values={'audio_latents': latents})
    capture = Capture(latents, _state(model), literals)
    with torch.no_grad(), capture:
        _, state = decode_step()(components, state)
    graph = capture.graph(state.get('audio'))
    _check(graph, 'decoder', (1, STEREO, HOP * tokens))
    return {'schema': DECODER_SCHEMA, 'graph': graph, 'input_shape': (STEREO, CHANNELS, tokens),
            'output_shape': (STEREO, HOP * tokens), 'extra': {'config': config},
            'provenance': dict(provenance, call='MiniMaxH3AudioDecodeStep', template_tokens=tokens,
                               call_source_sha256=_source_sha(decode_step),
                               declared_literals=sorted(f'config.{name}' for name in literals))}


def capture_encoder(model, config, *, samples, provenance):
    """The reference audio encoder (posterior mean) on a synthetic waveform, checked against the native contract."""
    import torch
    from .aten_capture import Capture
    device = next(model.parameters()).device
    waveform = _synthetic(torch, (STEREO, 1, samples), device, scale=0.1)
    capture = Capture(waveform, _state(model))
    with torch.no_grad(), capture:
        posterior = model.encode(waveform, return_dict=False)[0].mode()
    graph = capture.graph(posterior)
    output_shape = (STEREO, CHANNELS, samples // HOP)
    _check(graph, 'encoder', output_shape)
    return {'schema': ENCODER_SCHEMA, 'graph': graph, 'input_shape': (STEREO, 1, samples), 'output_shape': output_shape,
            'extra': {'config': config, 'latents_mean': list(config['latents_mean']),
                      'latents_std': list(config['latents_std'])},
            'provenance': dict(provenance, call=f'{MODEL_CLASS}.encode(x).latent_dist.mode()',
                               template_samples=samples)}


def prepare_audio_vae(checkpoint, output, *, workflow, device='cuda', tokens=DECODER_TOKENS,
                      samples=ENCODER_SAMPLES):
    """Write <output>/audio_vae (and, for Ref2VA, <output>/audio_encoder); return the report."""
    import torch
    if workflow not in ('fl2va', 'ref2va'):
        raise ValueError('unknown workflow: ' + str(workflow))
    checkpoint, output = Path(checkpoint), Path(output)
    folder = checkpoint / COMPONENT
    config = json.loads((folder / 'config.json').read_bytes())
    if config.get('_class_name') != MODEL_CLASS:
        raise ValueError(f'{folder} is not the Diffusers {MODEL_CLASS} folder')
    for name in ('latents_mean', 'latents_std'):
        if len(config.get(name) or ()) != CHANNELS:
            raise ValueError(f'audio VAE config needs {CHANNELS} {name} values')
    if tokens <= 0:
        raise ValueError('decoder template length must be positive')
    if workflow == 'ref2va' and (samples <= 0 or samples % HOP or samples > 30 * config['sampling_rate']):
        raise ValueError('encoder template length must be a positive multiple of the hop, at most 30 s')
    names = [COMPONENT] + (['audio_encoder'] if workflow == 'ref2va' else [])
    existing = [str(output / name) for name in names if (output / name).exists()]
    if existing:
        raise FileExistsError('remove to prepare again: ' + ', '.join(existing))
    model_class, decode_step, pipeline_state = _official()
    model = model_class.from_pretrained(folder, torch_dtype=torch.float32, local_files_only=True).eval()
    checked = _verify_state(model, Checkpoint(checkpoint, COMPONENT), torch)
    device = torch.device(device)
    model = model.to(device)
    package = _diffusers()
    provenance = {
        'model': REPO_ID, 'model_revision': REVISION, 'component': COMPONENT, 'model_class': MODEL_CLASS,
        'checkpoint_sha256': {f'{COMPONENT}/{path.name}': digest(path) for path in sorted(folder.iterdir())
                              if path.is_file() and path.suffix in ('.json', '.safetensors')},
        'diffusers_commit': DIFFUSERS_COMMIT, 'diffusers_version': package['version'],
        'diffusers_installed_commit': package['commit'], 'model_source_sha256': _source_sha(model_class),
        **_environment(torch, device),
        'constant_rule': 'operators reading only model state run once; results the graph consumes are constants',
        'no_intermediate_initializers': True, 'capture_input': 'synthetic', 'reference_values_available': False}
    packages = {COMPONENT: capture_decoder(model, decode_step, pipeline_state, config, tokens=tokens,
                                           provenance=provenance)}
    if workflow == 'ref2va':
        packages['audio_encoder'] = capture_encoder(model, config, samples=samples, provenance=provenance)
    output.mkdir(parents=True, exist_ok=True)
    published = _publish(output, packages)
    return {'workflow': workflow, 'checkpoint_tensors': checked, 'diffusers': package,
            'torch': provenance['torch'], 'torch_git': provenance['torch_git'],
            'decoder': published[COMPONENT], **({'encoder': published['audio_encoder']} if workflow == 'ref2va' else {})}
