"""Video VAE packages from the official root-level Diffusers `vae/` folder.

    video_vae/
      weights/<name>.raw     every official `vae/` tensor, byte for byte (FP32)
      buffers/<name>.raw     derived module state absent from the checkpoint (decoder RoPE frequencies)
      manifest.json          parameter manifest and audit root of every program: per-tensor SHA-256
                             and the official `config` (latents_mean, latents_std)
      encoder/program.json   FL image encoder: the eager plan `plans/video-encoder.json` (structure and
                             official tensor hashes only) with its parameters bound to `weights/`
      decoder/program.json   Inductor decoder tile; decoder/kernels/<sha256>.cubin, decoder/wrapper.py
      model.json             h3_native_video_vae_v1: audit root and both program SHA-256s
    video_encoder/program.json   Ref2VA only: the temporal reference encoder (17-frame tile)

The decoder is `torch.compile(vae.decoder, mode='default', dynamic=False)` called on one
[1, 24, 7, 16, 16] tile under FP16 autocast, as the reference decode compiles it. Its live
wrapper is replayed through `inductor_capture` to record CUBINs, launches and GEMM/attention
calls. The temporal encoder is the eager `quant_conv(encoder(x))` of one [1, 3, 17, 256, 256]
tile recorded at the ATen dispatch level. Capture inputs are seeded synthetic tensors: the
graphs do not depend on values, programs hold structure only, and only parameters initialize
buffers (the runtime input is bound per call). The capture needs one B200 (sm_100).
"""
import copy
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from .common import (DIFFUSERS_COMMIT, PARTITIONS, REPO_ID, REVISION, TORCH, Checkpoint, digest, extract_all,
                     write_json, write_tensor)

TORCH_GIT = 'cf30153c4c131c8164ee7798e5022d810682e2cb'
ENCODER_PLAN = Path(__file__).with_name('plans') / 'video-encoder.json'
KERNEL_CONFIGS = Path(__file__).with_name('plans') / 'decoder-kernel-configs.json'  # the validated selection
ENCODER_PLAN_SHA256 = 'dfefd929d4cae53b8181193adb76fa1a3c28209f809be970587bd1ac3be98dff'
COMPILE = {'mode': 'default', 'dynamic': False, 'autocast_dtype': 'torch.float16'}
DECODER_TILE, DECODER_OUTPUT = [1, 24, 7, 16, 16], [1, 3, 28, 256, 256]
CLIP_TILE, CLIP_OUTPUT = [1, 3, 17, 256, 256], [1, 48, 5, 16, 16]
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
SEED = 0
VIEWS = ('aten.permute.default', 'aten.view.default')
PADS = ('aten.reflection_pad3d.default', 'aten.constant_pad_nd.default')
ENCODER_KINDS = (*PADS, 'aten.convolution.default', 'aten.native_group_norm.default', 'aten.silu.default',
                 'aten.add.Tensor')
ENCODER_COUNTS = dict(zip(ENCODER_KINDS, (30, 30, 34, 25, 25, 12)))
# Frame-isolated GroupNorm materializes NCTHW <-> NTCHW around each of the 25 norms when T > 1.
CLIP_COUNTS = {**ENCODER_COUNTS, 'aten.clone.default': 50}
DECODER_COUNTS = {'kernel': 690, 'addmm': 217, 'mm': 1, 'cudnn_sdpa': 36}
EAGER, INDUCTOR = 'h3_original_eager_v1', 'h3_original_inductor_v1'
STAGES = {
    'video_vae_encoder_tile': {'schema': EAGER, 'input': ([1, 3, 1, 256, 256], 'torch.float32'),
                               'output': ([1, 48, 1, 16, 16], 'torch.float32'), 'counts': ENCODER_COUNTS},
    'video_vae_encoder_clip17_tile': {'schema': EAGER, 'input': (CLIP_TILE, 'torch.float32'),
                                      'output': (CLIP_OUTPUT, 'torch.float32'), 'counts': CLIP_COUNTS},
    'video_vae_decoder_tile': {'schema': INDUCTOR, 'input': (DECODER_TILE, 'torch.float16'),
                               'output': (DECODER_OUTPUT, 'torch.float16'), 'counts': DECODER_COUNTS},
}
# The native small-tile decoder (kernels/video_vae/decoder_generic.cu) re-implements these Inductor
# kernels by the numeric suffix of their symbol: (leading pointers, parameters), and the RMS
# reductions with these R0_BLOCK layouts.
GENERIC_KERNELS = {1: (7, 11), 4: (2, 6), 5: (1, 4), 6: (8, 11), 7: (10, 13), 8: (8, 12), 11: (2, 5),
                   13: (7, 11), 14: (7, 11), 15: (5, 9), 18: (2, 5)}
GENERIC_REDUCTION_BLOCKS = {1: 512, 8: 512, 13: 1024, 14: 1024}
WIDTHS = {f'torch.{name}': width for name, width in (
    ('float64', 8), ('float32', 4), ('bfloat16', 2), ('float16', 2), ('int64', 8), ('uint64', 8), ('int32', 4),
    ('uint32', 4), ('int16', 2), ('uint16', 2), ('int8', 1), ('uint8', 1), ('bool', 1))}


def contiguous(shape):
    return [math.prod(shape[i + 1:]) for i in range(len(shape))]


def tensors(value):
    """Every buffer view referenced inside an op field."""
    if isinstance(value, dict):
        if 'buffer' in value:
            yield value
        else:
            for item in value.values():
                yield from tensors(item)
    elif isinstance(value, list):
        for item in value:
            yield from tensors(item)


def check_plan(program, stage):
    """Structural checks of the native loader (`video_vae/main.rs`, `*_manifest.rs`); returns op counts."""
    rules = STAGES[stage]

    def require(condition, message):
        if not condition:
            raise ValueError(f'{stage}: {message}')

    def view(ref, dtype=None, rank=None):
        require(isinstance(ref, dict) and 'buffer' in ref, 'expected a tensor view')
        require(dtype is None or ref['dtype'] == dtype, f'expected {dtype}, found {ref["dtype"]}')
        require(rank is None or len(ref['shape']) == rank, f'expected rank {rank}: {ref["shape"]}')
        return ref

    json.dumps(program, allow_nan=False)  # the native JSON reader rejects NaN/Infinity
    require(program.get('schema') == rules['schema'] and program.get('stage') == stage, 'schema or stage')
    require(program.get('no_intermediate_initializers') is True, 'intermediate initializers allowed')
    backend = program.get('backend', {})
    require(backend.get('allow_tf32') is False and backend.get('allow_fp16_reduced_precision_reduction') is True,
            'unvalidated GEMM backend')
    require(isinstance(program.get('kernels'), dict), 'kernel table')
    require(rules['schema'] == INDUCTOR or not program['kernels'], 'an eager program has no CUBINs')
    buffers = program['buffers']
    inputs = []
    for index, buffer in enumerate(buffers):
        require(buffer['id'] == index, 'noncanonical allocation ids')
        require(buffer['dtype'] in WIDTHS and buffer['bytes'] > 0, f'buffer {index} layout')
        role, source = buffer['role'], buffer['source']
        if role == 'parameter':
            spec = source['spec']
            require(isinstance(source['parameter'], str) and {'file', 'bytes', 'sha256'} <= spec.keys(),
                    f'parameter buffer {index} has no tensor file')
            require(spec['dtype'] == buffer['dtype'] and spec['shape'] == buffer['shape']
                    and spec['bytes'] <= buffer['bytes'], f'parameter {source["parameter"]} layout')
        elif role == 'input':
            require(isinstance(source, dict) and 'input' in source, f'input buffer {index} source')
            inputs.append(buffer)
        else:
            require(role == 'scratch' and source is None, f'buffer {index}: scratch with an initializer')
    shape, dtype = rules['input']
    require(len(inputs) == 1, 'exactly one runtime input')
    require(inputs[0]['shape'] == shape and inputs[0]['dtype'] == dtype and inputs[0]['stride'] == contiguous(shape)
            and inputs[0]['bytes'] >= math.prod(shape) * WIDTHS[dtype], f'input contract {shape} {dtype}')
    require(rules['schema'] == EAGER or inputs[0]['id'] == 0, 'the decoder input must be buffer 0')
    kinds = Counter()
    for index, op in enumerate(program['ops']):
        kind, args = op['kind'], op['args']
        kinds[kind] += 1
        for ref in tensors([args, op.get('kwargs'), op.get('outputs')]):
            require(0 <= ref['buffer'] < len(buffers) and ref['dtype'] in WIDTHS
                    and len(ref['shape']) == len(ref['stride']) and min(ref['stride'], default=0) >= 0,
                    f'op {index}: invalid view')
            span = sum((d - 1) * s for d, s in zip(ref['shape'], ref['stride'])) + 1
            end = ref['offset'] + span * WIDTHS[ref['dtype']] if math.prod(ref['shape']) else 0
            require(ref['offset'] >= 0 and end <= buffers[ref['buffer']]['bytes'], f'op {index}: view out of bounds')
            require(stage != 'video_vae_encoder_tile' or (ref['offset'] == 0 and ref['dtype'] == 'torch.float32'),
                    f'op {index}: the resizable encoder needs zero-offset FP32 views')
        if kind == 'kernel':
            require(op['kernel'] in program['kernels'] and len(op['grid']) == 3
                    and all(isinstance(x, int) and x > 0 for x in op['grid']), f'op {index}: kernel launch')
            for arg in args:
                require('buffer' in arg if arg['type'] == 'pointer' else isinstance(arg.get('bits'), int)
                        and 0 <= arg['bits'] < 1 << 64, f'op {index}: kernel argument')
        elif kind in ('mm', 'addmm'):
            first = int(kind == 'addmm')
            x, w, out = (view(ref, 'torch.float16', 2) for ref in (args[first], args[first + 1], op['kwargs']['out']))
            rows = 1797 if first else 1792
            require(x['shape'][0] == rows and x['shape'][1] == w['shape'][0] and w['stride'] == [1, w['shape'][0]]
                    and x['stride'] in ([x['shape'][1], 1], [1, rows] if not first else None)
                    and x['offset'] == 0 and out['offset'] == 0, f'op {index}: decoder GEMM view')
            require(all(op['kwargs'].get(k) in (None, 1, 1.0) for k in ('alpha', 'beta')), f'op {index}: coefficient')
        elif kind == 'cudnn_sdpa':
            for ref in (*args[:3], op['outputs'][0]):
                view(ref, 'torch.float16', 4)
                require(ref['shape'] == [1, 32, 1797, 64] and ref['stride'] == [3680256, 64, 2048, 1]
                        and ref['offset'] == 0, f'op {index}: decoder attention layout')
        elif kind in PADS:
            view(args[0], 'torch.float32', 5)
            view(op['outputs'][0], rank=5)
            require(len(args[1]) == 6 and (len(args) < 3 or args[2] == 0), f'op {index}: padding contract')
        elif kind == 'aten.clone.default':
            require(view(args[0], 'torch.float32', 5)['shape'] == view(op['outputs'][0], 'torch.float32', 5)['shape'],
                    f'op {index}: copy shape')
        elif kind == 'aten.convolution.default':
            for ref, dtype in ((args[0], 'torch.float32'), (args[1], 'torch.float32'), (op['outputs'][0], None)):
                view(ref, dtype, 5)
            require(args[6] is False and args[8] == 1 and all(len(args[i]) == 3 for i in (3, 4, 5))
                    and (args[2] is None or 'buffer' in args[2]), f'op {index}: convolution contract')
        elif kind == 'aten.native_group_norm.default':
            view(args[0], 'torch.float32')
            require(args[3] * args[4] * 8 <= 1 << 20 and args[5] > 1 and len(op['outputs']) == 3,
                    f'op {index}: group norm contract')
        elif kind in ('aten.silu.default', 'aten.add.Tensor'):
            view(args[0], 'torch.float32')
            require(op.get('kwargs', {}).get('alpha') in (None, 1), f'op {index}: alpha')
            require(kind != 'aten.add.Tensor' or args[0]['shape'] == view(args[1])['shape'], f'op {index}: broadcast')
    require(dict(kinds) == rules['counts'], f'operator counts {dict(kinds)} != {rules["counts"]}')
    shape, dtype = rules['output']
    output = view(program['outputs'][0], dtype)
    require(output['offset'] == 0 and output['shape'] == shape and output['stride'] == contiguous(shape),
            f'output contract {shape}')
    gold = program.get('gold', {})  # the native tile output descriptor: dtype, shape, bytes; never a file
    require(gold.get('shape') == shape and gold.get('dtype') == dtype and not gold.get('file')
            and gold.get('bytes') == math.prod(shape) * WIDTHS[dtype], 'output descriptor')
    return dict(kinds)


def validate_program(program, root, stage):
    """`check_plan` plus the identity of the audit manifest, parameter files and CUBINs."""
    counts = check_plan(program, stage)
    audit_root, root = Path(program['audit_root']), Path(root)
    if not audit_root.is_absolute():
        raise ValueError(f'{stage}: audit_root must be absolute')
    if digest(audit_root / 'manifest.json') != program['audit_manifest_sha256']:
        raise ValueError(f'{stage}: audit manifest identity')
    audit = json.loads((audit_root / 'manifest.json').read_bytes())
    check_audit(audit)
    for buffer in program['buffers']:
        if buffer['role'] != 'parameter':
            continue
        name, spec = buffer['source']['parameter'], buffer['source']['spec']
        if spec != audit['parameters'].get(name, audit['buffers'].get(name)):
            raise ValueError(f'{stage}: {name} is not the audited tensor')
        if (audit_root / spec['file']).stat().st_size != spec['bytes']:
            raise ValueError(f'{stage}: {name} file size')
    for name, kernel in program['kernels'].items():
        if name != kernel['sha256'] or digest(root / kernel['file']) != kernel['sha256']:
            raise ValueError(f'{stage}: changed CUBIN {name}')
        if not (isinstance(kernel['symbol'], str) and kernel['warps'] > 0 and kernel['shared'] >= 0
                and isinstance(kernel['signature'], dict)):
            raise ValueError(f'{stage}: kernel metadata {name}')
    return counts


def check_audit(audit):
    config = audit['config']
    if any(len(config.get(key, ())) != 24 for key in ('latents_mean', 'latents_std')):
        raise ValueError('the VAE config must carry 24 latents_mean and latents_std values')
    for name in ('post_quant_conv.weight', 'post_quant_conv.bias'):
        if audit['parameters'].get(name, {}).get('dtype') != 'torch.float32':
            raise ValueError('the native post-quant convolution needs FP32 ' + name)


def small_tile_contract(program):
    """Whether the kernels the native small-tile decoder re-implements keep their ABI and layout."""
    found, problems = {}, set()
    for op in program['ops']:
        if op['kind'] != 'kernel':
            continue
        kernel = program['kernels'][op['kernel']]
        suffix = kernel['symbol'].rsplit('_', 1)[-1]
        if not suffix.isdigit() or int(suffix) not in GENERIC_KERNELS:
            continue
        kind = int(suffix)
        pointers, count = GENERIC_KERNELS[kind]
        found[kind] = kernel['symbol']
        if len(op['args']) != count or any(arg['type'] != 'pointer' for arg in op['args'][:pointers]):
            problems.add(f'{kernel["symbol"]}: {len(op["args"])} parameters, native expects {count}')
        block = kernel['config']['kwargs'].get('R0_BLOCK')
        if kind in GENERIC_REDUCTION_BLOCKS and block != GENERIC_REDUCTION_BLOCKS[kind]:
            problems.add(f'{kernel["symbol"]}: R0_BLOCK {block}, native expects {GENERIC_REDUCTION_BLOCKS[kind]}')
    missing = sorted(set(GENERIC_KERNELS) - set(found))
    return {'consistent': not problems and not missing, 'kernels': {str(k): v for k, v in sorted(found.items())},
            'missing_kinds': missing, 'problems': sorted(problems)}


def output_descriptor(value):
    return {'dtype': str(value.dtype), 'shape': list(value.shape), 'bytes': value.numel() * value.element_size()}


def load_plan():
    raw = ENCODER_PLAN.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ENCODER_PLAN_SHA256:
        raise ValueError('the encoder plan differs from its pinned structure: ' + str(ENCODER_PLAN))
    plan = json.loads(raw)
    check_plan(plan, 'video_vae_encoder_tile')
    return plan


def load_kernel_configs(path):
    """{kernel: {kwargs, num_warps[, sha256]}}; a previous decoder program.json is also accepted."""
    value = json.loads(Path(path).read_bytes())
    value = value.get('kernel_configs', value)
    configs = {}
    for name, entry in value.items():
        if not name.startswith('triton_') or not isinstance(entry.get('kwargs'), dict) \
                or not isinstance(entry.get('num_warps'), int):
            raise ValueError('invalid decoder kernel configuration: ' + name)
        configs[name] = {key: entry[key] for key in ('kwargs', 'num_warps', 'sha256') if key in entry}
    return configs


def encoder_program(plan, weights, audit_root, audit_sha):
    """The FL encoder plan with every parameter bound to its extracted official tensor."""
    program = copy.deepcopy(plan)
    for buffer in program['buffers']:
        if buffer['role'] != 'parameter':
            continue
        name, expected = buffer['source']['parameter'], buffer['source']['spec']
        spec = weights.get(name)
        if spec is None or any(spec[key] != expected[key] for key in ('dtype', 'shape', 'stride', 'bytes', 'sha256')):
            raise ValueError('official tensor differs from the encoder plan: ' + name)
        buffer['source']['spec'] = copy.deepcopy(spec)
    program.update(audit_root=str(audit_root), audit_manifest_sha256=audit_sha)
    return program


def audit_manifest(checkpoint, weights, buffers):
    config_path = Path(checkpoint) / 'vae/config.json'
    audit = {'schema': 'h3-official-vae-parameters-v1', 'repo_id': REPO_ID, 'revision': REVISION,
             'diffusers_commit': DIFFUSERS_COMMIT, 'config': json.loads(config_path.read_bytes()),
             'config_sha256': digest(config_path), 'parameters': weights, 'buffers': buffers}
    check_audit(audit)
    return audit


def backend_flags():
    import torch
    matmul = torch.backends.cuda.matmul
    return {'allow_tf32': bool(matmul.allow_tf32),
            'allow_fp16_reduced_precision_reduction': bool(matmul.allow_fp16_reduced_precision_reduction),
            'allow_bf16_reduced_precision_reduction': bool(matmul.allow_bf16_reduced_precision_reduction),
            'cudnn_allow_tf32': bool(torch.backends.cudnn.allow_tf32),
            'cudnn_benchmark': bool(torch.backends.cudnn.benchmark),
            'cudnn_deterministic': bool(torch.backends.cudnn.deterministic),
            'float32_matmul_precision': torch.get_float32_matmul_precision()}


def configure_backend(backend):
    """Apply the plan's matmul and cuDNN settings (the PyTorch defaults); return what is in effect."""
    import torch
    matmul = torch.backends.cuda.matmul
    torch.set_float32_matmul_precision(backend['float32_matmul_precision'])
    matmul.allow_tf32 = backend['allow_tf32']
    matmul.allow_fp16_reduced_precision_reduction = backend['allow_fp16_reduced_precision_reduction']
    matmul.allow_bf16_reduced_precision_reduction = backend['allow_bf16_reduced_precision_reduction']
    torch.backends.cudnn.allow_tf32 = backend['cudnn_allow_tf32']
    torch.backends.cudnn.benchmark = backend['cudnn_benchmark']
    torch.backends.cudnn.deterministic = backend['cudnn_deterministic']
    if backend_flags() != backend:
        raise RuntimeError('the preparation backend could not be configured')
    return backend_flags()


def environment():
    import diffusers
    import torch
    import triton
    return {'torch': str(torch.__version__), 'torch_git': torch.version.git_version, 'cuda': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(), 'triton': triton.__version__, 'diffusers': diffusers.__version__,
            'device': torch.cuda.get_device_name(), 'capability': list(torch.cuda.get_device_capability())}


def load_vae(checkpoint):
    """The official VAE on the GPU; `_keep_in_fp32_modules` keeps every module FP32 as released."""
    import torch
    from diffusers import AutoencoderKLMiniMaxH3
    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError('video VAE preparation records sm_100 CUBINs and needs a B200')
    # The same lower-dtype request as the reference pipeline; it does not narrow any VAE module.
    vae = AutoencoderKLMiniMaxH3.from_pretrained(str(checkpoint), subfolder='vae', torch_dtype=torch.bfloat16,
                                                 local_files_only=True).eval().to('cuda')
    if {value.dtype for value in vae.parameters()} != {torch.float32}:
        raise RuntimeError('the official VAE must load in FP32')
    return vae


def tensor_sha(value):
    import torch
    return hashlib.sha256(value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()).hexdigest()


def module_state(vae, weights):
    """Check every loaded tensor against its extracted file; return the derived (unsaved) buffers."""
    buffers = dict(vae.named_buffers())
    loaded = {**dict(vae.named_parameters()), **{name: v for name, v in buffers.items() if name in weights}}
    if set(loaded) != set(weights):
        raise ValueError('loaded VAE tensors differ from the checkpoint: '
                         + ', '.join(sorted(set(loaded) ^ set(weights))[:8]))
    for name, value in sorted(loaded.items()):
        spec = weights[name]
        if (str(value.dtype), list(value.shape), tensor_sha(value)) != (spec['dtype'], spec['shape'], spec['sha256']):
            raise ValueError('loaded VAE tensor differs from the checkpoint: ' + name)
    return {name: value for name, value in buffers.items() if name not in weights}


def graph_roots(module, prefix, audit, inputs, runtime):
    """Describe each compiled graph input: a module parameter/buffer of the audit, or the runtime input."""
    known = {}
    for group, items in (('parameters', module.named_parameters()), ('buffers', module.named_buffers())):
        for name, value in items:
            known[value.data_ptr()] = {'parameter': prefix + name, 'spec': audit[group][prefix + name]}
    roots = []
    for value in inputs:
        if value.data_ptr() == runtime[1].data_ptr():
            roots.append({'input': runtime[0]})
        elif value.data_ptr() in known:
            roots.append(known[value.data_ptr()])
        else:
            raise RuntimeError(f'unknown compiled graph input: {list(value.shape)} {value.dtype}')
    if sum('input' in root for root in roots) != 1:
        raise RuntimeError('the compiled graph must take the tile exactly once')
    return roots


def decoder_program(source, namespace, inputs, roots, root, *, audit_root, audit_sha, backend, environment):
    """Record the live wrapper call into the decoder tile program; return (program, output)."""
    from .inductor_capture import config_digest, record_call, selected_kernels
    selection = selected_kernels(namespace)
    recorder, result = record_call(source, namespace, inputs, roots, root)
    output = result[0]
    (root / 'wrapper.py').write_text(source)
    program = {'schema': INDUCTOR, 'stage': 'video_vae_decoder_tile', 'audit_root': str(audit_root),
               'audit_manifest_sha256': audit_sha, 'no_intermediate_initializers': True, 'backend': backend,
               'compile': COMPILE, 'environment': environment, 'wrapper_sha256': digest(root / 'wrapper.py'),
               'kernel_configs': selection, 'kernel_configs_sha256': config_digest(selection),
               'buffers': recorder.buffers, 'kernels': recorder.kernels, 'ops': recorder.ops,
               'outputs': [recorder.reference(output)], 'gold': output_descriptor(output),
               'ignored_calls': dict(sorted(recorder.ignored.items()))}
    return program, output


def capture_decoder(vae, root, *, audit_root, audit, audit_sha, backend, environment, kernel_configs=None):
    """Compile the decoder once on a tile, optionally pin kernel configurations, and record it."""
    import torch
    from . import inductor_capture as capture
    latent = torch.randn(DECODER_TILE, generator=torch.Generator().manual_seed(SEED)).to('cuda')
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        tile = vae.post_quant_conv(latent)  # FP32 weights under FP16 autocast, as the native post-quant
    if list(tile.shape) != DECODER_TILE or tile.dtype != torch.float16 or not tile.is_contiguous():
        raise RuntimeError('unexpected decoder tile layout')
    compiled = torch.compile(vae.decoder, mode=COMPILE['mode'], dynamic=COMPILE['dynamic'])

    def run():
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            return compiled(tile)

    root.mkdir(parents=True, exist_ok=False)
    with capture.fresh_compilation():
        run()  # compile and autotune
        run()
        graph, inputs, autocast = capture.capture_graph(run)
        if autocast:
            raise RuntimeError('the compiled decoder graph runs under autocast')
        runner = graph.compiled_fn_runner
        if runner is None:
            raise RuntimeError('the compiled decoder has no Python wrapper runner')
        namespace = runner.call.__func__.__globals__  # the warmed module: its tuners hold the selection
        forced = capture.force_kernel_configs(namespace, kernel_configs) if kernel_configs else []
        roots = graph_roots(vae.decoder, 'decoder.', audit, inputs, ('decoder_tile', tile))
        with torch.no_grad(), torch._C._DisableAutocast():
            expected = runner.call(list(inputs))[0]
        program, output = decoder_program(graph.source_code, namespace, inputs, roots, root, audit_root=audit_root,
                                          audit_sha=audit_sha, backend=backend, environment=environment)
        torch.cuda.synchronize()
    if not torch.equal(output.view(torch.int16), expected.view(torch.int16)) or not torch.isfinite(output).all():
        raise RuntimeError('the recorded decoder call differs from the compiled decoder')
    mismatched = sorted(name for name, entry in (kernel_configs or {}).items()
                        if 'sha256' in entry and entry['sha256'] != program['kernel_configs'][name]['sha256'])
    return program, {'kernel_config_source': 'requested' if kernel_configs else 'inductor_autotune',
                     'forced_kernels': forced, 'cubin_sha256_differs_from_request': mismatched,
                     'capture_input': {'seed': SEED, 'shape': DECODER_TILE, 'sha256': tensor_sha(tile)},
                     'capture_output_sha256': tensor_sha(output)}


def trace_eager(table, run):
    """Run `run()` recording each encoder ATen operation against the allocation table `table`."""
    from torch.utils._python_dispatch import TorchDispatchMode
    ops = []

    class Trace(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs, name = kwargs or {}, str(func)
            if name in VIEWS:  # views are strides over recorded buffers
                return func(*args, **kwargs)
            if name not in CLIP_COUNTS:
                raise RuntimeError('unrecorded encoder operator: ' + name)
            entry = {'kind': name, 'args': table.reference(list(args)), 'kwargs': table.reference(kwargs)}
            result = func(*args, **kwargs)
            outputs = list(result) if isinstance(result, (tuple, list)) else [result]
            for value in outputs:
                table.allocate(value)
            entry['outputs'] = table.reference(outputs)
            ops.append(entry)
            return result

    with Trace():
        result = run()
    return ops, result


def capture_temporal_encoder(vae, *, audit_root, audit, audit_sha, backend, environment):
    """Record the eager 17-frame encoder tile `quant_conv(encoder(x))` (the Ref2VA video reference)."""
    import torch
    from .inductor_capture import Buffers
    table = Buffers()
    for name, value in vae.named_parameters():
        if name.startswith(('encoder.', 'quant_conv.')):
            table.declare(value, 'parameter', {'parameter': name, 'spec': audit['parameters'][name]})
    device = next(vae.encoder.parameters()).device
    pixels = torch.rand(CLIP_TILE, generator=torch.Generator().manual_seed(SEED))
    mean, std = (torch.tensor(v, dtype=torch.float32).view(1, 3, 1, 1, 1) for v in (IMAGENET_MEAN, IMAGENET_STD))
    tile = ((pixels - mean) / std).contiguous().to(device)
    table.declare(tile, 'input', {'input': 'encoder_clip17_tile'})
    with torch.no_grad():
        ops, output = trace_eager(table, lambda: vae.quant_conv(vae.encoder(tile)))
    program = {'schema': EAGER, 'stage': 'video_vae_encoder_clip17_tile', 'audit_root': str(audit_root),
               'audit_manifest_sha256': audit_sha, 'no_intermediate_initializers': True, 'backend': backend,
               'environment': environment, 'buffers': table.buffers, 'kernels': {}, 'ops': ops,
               'outputs': [table.reference(output)], 'gold': output_descriptor(output)}
    return program, {'capture_input': {'seed': SEED, 'shape': CLIP_TILE, 'sha256': tensor_sha(tile)},
                     'capture_output_sha256': tensor_sha(output)}


def publish(root, program, stage):
    counts = validate_program(program, root, stage)
    return write_json(Path(root) / 'program.json', program), counts


def prepare_video_vae(checkpoint, output, *, workflow, decoder_kernel_configs=None, workers=8):
    """Write `<output>/video_vae/` (and `<output>/video_encoder/` for Ref2VA); return a report."""
    if workflow not in PARTITIONS:
        raise ValueError('unknown workflow: ' + str(workflow))
    checkpoint, output = Path(checkpoint).resolve(strict=True), Path(output).absolute()
    root, reference = output / 'video_vae', output / 'video_encoder'
    # Autotuning times candidates, so left alone it can pick another block size, and so another
    # reduction order, from one run to the next.
    configs = load_kernel_configs(decoder_kernel_configs or KERNEL_CONFIGS)
    plan = load_plan()
    root.mkdir(parents=True, exist_ok=False)
    if workflow == 'ref2va':
        reference.mkdir(exist_ok=False)
    source = Checkpoint(checkpoint, 'vae')
    weights = extract_all(source, root, {name: name for name in source.tensors}, workers=workers)
    vae = load_vae(checkpoint)
    backend, env = configure_backend(plan['backend']), environment()
    buffers = {name: write_tensor(root, f'buffers/{name}.raw', value)
               for name, value in sorted(module_state(vae, weights).items())}
    audit = audit_manifest(checkpoint, weights, buffers)
    audit_sha = write_json(root / 'manifest.json', audit)
    shared = {'audit_root': root, 'audit_sha': audit_sha, 'backend': backend, 'environment': env}

    encoder_sha, encoder_counts = publish(root / 'encoder', encoder_program(plan, weights, root, audit_sha),
                                          'video_vae_encoder_tile')
    program, decoder = capture_decoder(vae, root / 'decoder', audit=audit, kernel_configs=configs, **shared)
    decoder_sha, decoder_counts = publish(root / 'decoder', program, 'video_vae_decoder_tile')
    decoder.update(root=str(root / 'decoder'), program_sha256=decoder_sha, op_counts=decoder_counts,
                   kernels=len(program['kernels']), kernel_configs_sha256=program['kernel_configs_sha256'],
                   small_tile_contract=small_tile_contract(program), ignored_calls=program['ignored_calls'])
    report = {'component': 'video_vae', 'root': str(root), 'audit_manifest_sha256': audit_sha,
              'tensors': len(weights), 'bytes': sum(spec['bytes'] for spec in weights.values()),
              'derived_buffers': {name: spec['sha256'] for name, spec in buffers.items()},
              'encoder': {'root': str(root / 'encoder'), 'program_sha256': encoder_sha, 'op_counts': encoder_counts,
                          'plan_sha256': ENCODER_PLAN_SHA256},
              'decoder': decoder, 'environment': env,
              'pinned_torch': env['torch'].startswith(TORCH) and env['torch_git'] == TORCH_GIT}
    if workflow == 'ref2va':
        program, details = capture_temporal_encoder(vae, audit=audit, **shared)
        sha, counts = publish(reference, program, 'video_vae_encoder_clip17_tile')
        report['video_encoder'] = {'root': str(reference), 'program_sha256': sha, 'op_counts': counts, **details}
    model = {'schema': 'h3_native_video_vae_v1', 'audit_root': str(root), 'audit_manifest_sha256': audit_sha,
             'no_reference_activations_as_runtime_inputs': True,
             'encoder': {'root': str(root / 'encoder'), 'program_sha256': encoder_sha},
             'decoder': {'root': str(root / 'decoder'), 'program_sha256': decoder_sha}}
    report['model_sha256'] = write_json(root / 'model.json', model)
    return report
