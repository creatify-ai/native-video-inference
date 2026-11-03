"""CPU checks of the video VAE package producer on synthetic inputs (no GPU, no checkpoint)."""
import copy
import dataclasses
import hashlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest

from models.minimax_h3_official.prepare import video_vae
from models.minimax_h3_official.prepare.common import TORCH, digest

FL = 'video_vae_encoder_tile'


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


def test_shipped_encoder_plan_is_pinned_structure():
    plan = video_vae.load_plan()
    assert video_vae.check_plan(plan, FL) == video_vae.ENCODER_COUNTS
    assert 'audit_root' not in plan and plan['kernels'] == {}
    roles = {}
    for buffer in plan['buffers']:
        roles[buffer['role']] = roles.get(buffer['role'], 0) + 1
        if buffer['role'] == 'parameter':
            name, spec = buffer['source']['parameter'], buffer['source']['spec']
            assert name.startswith(('encoder.', 'quant_conv.')) and spec['file'] == f'parameters/{name}.raw'
            assert len(spec['sha256']) == 64
        elif buffer['role'] == 'input':
            assert buffer['source'] == {'input': 'encoder_tile'}
        else:
            assert buffer['source'] is None
    assert roles == {'parameter': 118, 'input': 1, 'scratch': 206}


@pytest.mark.parametrize('mutate', [
    lambda p: p.update(no_intermediate_initializers=False),
    lambda p: p['backend'].update(allow_tf32=True),
    lambda p: p['buffers'][200].update(source={'parameter': 'x', 'spec': {}}),
    lambda p: p['buffers'][200].update(role='input', source={'input': 'second'}),
    lambda p: p['ops'].pop(),
    lambda p: p['ops'][2]['args'].__setitem__(8, 2),
    lambda p: p['ops'][0]['outputs'][0].update(offset=4),
    lambda p: p['outputs'][0].update(shape=[1, 48, 1, 8, 8]),
    lambda p: p['gold'].update(file='output.raw'),
])
def test_plan_check_rejects_native_contract_breaks(mutate):
    plan = copy.deepcopy(video_vae.load_plan())
    mutate(plan)
    with pytest.raises(ValueError):
        video_vae.check_plan(plan, FL)


def test_kernel_configs_accept_a_mapping_or_a_previous_program(tmp_path):
    selection = {'triton_red_fused_1': {'kwargs': {'XBLOCK': 2, 'R0_BLOCK': 512}, 'num_warps': 8, 'sha256': 'a' * 64}}
    (tmp_path / 'map.json').write_text(json.dumps(selection))
    (tmp_path / 'program.json').write_text(json.dumps({'schema': 'x', 'kernel_configs': selection}))
    assert video_vae.load_kernel_configs(tmp_path / 'map.json') == selection
    assert video_vae.load_kernel_configs(tmp_path / 'program.json') == selection
    (tmp_path / 'bad.json').write_text(json.dumps({'triton_x': {'kwargs': {}}}))
    with pytest.raises(ValueError):
        video_vae.load_kernel_configs(tmp_path / 'bad.json')


# ---- eager tracing -------------------------------------------------------------------------------

def tiny_vae():
    """A VAE stand-in with the official encoder's ATen operation counts on tiny tensors."""
    torch = pytest.importorskip('torch')
    import torch.nn.functional as F

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_in = torch.nn.Conv3d(3, 48, 1, stride=(4, 16, 16))
            self.mix = torch.nn.Conv3d(48, 48, 1, bias=False)
            self.norm = torch.nn.GroupNorm(4, 48, eps=1e-6)

        def forward(self, x):
            y = self.conv_in(x)
            for _ in range(30):
                F.pad(y, (1, 1, 1, 1, 0, 0), mode='reflect')
            for _ in range(30):
                F.pad(y, (0, 0, 0, 0, 2, 0))
            for _ in range(32):
                y = F.conv3d(y, self.mix.weight)
            for _ in range(25):  # frame-isolated GroupNorm, as the official encoder normalizes
                b, c, t, h, w = y.shape
                z = y.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, 1, h, w)
                z = F.group_norm(z, 4, self.norm.weight, self.norm.bias, 1e-6)
                y = z.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
            for _ in range(25):
                y = F.silu(y)
            for _ in range(12):
                y = y + y
            return y

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(64, 64)
            self.rope = torch.nn.Module()
            self.rope.register_buffer('inv_freq', 1.0 / 100.0 ** torch.arange(0, 1, 0.125), persistent=False)

    class VAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder, self.decoder = Encoder(), Decoder()
            self.quant_conv = torch.nn.Conv3d(48, 48, 1)
            self.post_quant_conv = torch.nn.Conv3d(24, 24, 1)

    torch.manual_seed(0)
    return VAE().eval()


def plan_spec(name, value):
    return {'dtype': str(value.dtype), 'shape': list(value.shape), 'stride': list(value.stride()),
            'bytes': value.numel() * value.element_size(), 'sha256': video_vae.tensor_sha(value),
            'file': f'parameters/{name}.raw'}


def fl_plan(vae):
    """A structure-only FL plan of `vae`, produced as the shipped plan was: an eager trace."""
    torch = pytest.importorskip('torch')
    from models.minimax_h3_official.prepare.inductor_capture import Buffers
    table = Buffers()
    for name, value in vae.named_parameters():
        if name.startswith(('encoder.', 'quant_conv.')):
            table.declare(value, 'parameter', {'parameter': name, 'spec': plan_spec(name, value)})
    tile = torch.zeros(1, 3, 1, 256, 256)
    table.declare(tile, 'input', {'input': 'encoder_tile'})
    with torch.no_grad():
        ops, output = video_vae.trace_eager(table, lambda: vae.quant_conv(vae.encoder(tile)))
    return {'schema': video_vae.EAGER, 'stage': FL, 'no_intermediate_initializers': True,
            'backend': video_vae.load_plan()['backend'], 'buffers': table.buffers, 'kernels': {}, 'ops': ops,
            'outputs': [table.reference(output)], 'gold': video_vae.output_descriptor(output)}


def test_eager_trace_records_operations_and_views_over_declared_storage():
    vae = tiny_vae()
    plan = fl_plan(vae)
    assert video_vae.check_plan(plan, FL) == video_vae.ENCODER_COUNTS  # T=1: no materializing copies
    first = plan['ops'][0]
    assert first['kind'] == 'aten.convolution.default' and first['args'][0]['buffer'] == 7  # the input
    assert first['args'][1] == {'buffer': 0, 'offset': 0, 'shape': [48, 3, 1, 1, 1], 'stride': [3, 1, 1, 1, 1],
                                'dtype': 'torch.float32'}
    assert [b['role'] for b in plan['buffers'][:8]] == ['parameter'] * 7 + ['input']
    assert all(b['source'] is None for b in plan['buffers'][8:])


def test_eager_trace_rejects_unrecorded_operators_and_undeclared_storage():
    torch = pytest.importorskip('torch')
    from models.minimax_h3_official.prepare.inductor_capture import Buffers
    table = Buffers()
    x = torch.ones(1, 3, 1, 4, 4)
    table.declare(x, 'input', {'input': 'x'})
    with pytest.raises(RuntimeError, match='unrecorded encoder operator'):
        video_vae.trace_eager(table, lambda: torch.relu(x))
    constant = torch.ones(1, 3, 1, 4, 4)
    with pytest.raises(RuntimeError, match='undeclared storage'):
        video_vae.trace_eager(table, lambda: x + constant)
    view = torch.ones(8)[2:]
    with pytest.raises(ValueError):
        Buffers().declare(view, 'parameter', {'parameter': 'view'})


# ---- Inductor wrapper recording ------------------------------------------------------------------

@dataclasses.dataclass
class Config:
    kwargs: dict
    num_warps: int


class Tuner:
    """The CachingAutotuner surface the recorder reads; launches do nothing."""

    def __init__(self, name, args, constexprs, signature, kwargs, warps=4):
        self.fn = SimpleNamespace(__name__=name, arg_names=args, constexprs=constexprs)
        self.name, self.signature, self.triton_meta = name, signature, {}
        self.compile_results = [self._precompile_config(Config(kwargs, warps))]
        self.launchers = [self.compile_results[0].make_launcher()]

    def run(self, *args, stream=None):
        return None

    def _interpret_args_grid(self, args, config):
        numel = next(a for a in args if isinstance(a, int))
        return args, (-(-numel // config.kwargs['XBLOCK']), 1, 1)

    def _ensure_kernel_loaded(self):
        pass

    def _precompile_config(self, config):
        cubin = f'{self.name}:{sorted(config.kwargs.items())}:{config.num_warps}'.encode()
        binary = SimpleNamespace(cubin_path=None, cubin_raw=cubin, name=self.name, shared=0, num_warps=config.num_warps,
                                 has_global_scratch=True, has_profile_scratch=True)
        return SimpleNamespace(config=config, kernel=binary,
                               compile_meta={'signature': self.signature, 'constants': {}},
                               make_launcher=lambda: SimpleNamespace(config=config))


def pointwise(name, source='*fp32', target='*fp16'):
    return Tuner(name, ['in_ptr0', 'out_ptr0', 'xnumel', 'XBLOCK'], [3],
                 {'in_ptr0': source, 'out_ptr0': target, 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, {'XBLOCK': 256})


def namespace(torch):
    reduction = Tuner('triton_red_fused_rms_1', ['in_ptr0', 'in_ptr1', 'out_ptr0', 'xnumel', 'r0_numel', 'XBLOCK',
                                                 'R0_BLOCK'], [5, 6],
                      {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp16', 'xnumel': 'i32',
                       'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'},
                      {'XBLOCK': 2, 'R0_BLOCK': 512}, warps=8)

    def sdpa(q, k, v, bias, dropout, scale=None):
        return (torch.empty_strided(q.shape, q.stride(), dtype=q.dtype), torch.empty(1, 32, 1797), None, None, 1797,
                1797, torch.empty(()), torch.empty(()), torch.empty(0))

    def reinterpret_tensor(base, size, stride, offset):
        return base.as_strided(size, stride, base.storage_offset() + offset)

    return {'torch': torch, 'reinterpret_tensor': reinterpret_tensor,
            'empty_strided_cuda': lambda size, stride, dtype: torch.empty_strided(size, stride, dtype=dtype),
            'assert_size_stride': lambda *args: None, 'get_raw_stream': lambda index: 0,
            'extern_kernels': SimpleNamespace(mm=lambda a, b, out: out, bmm=lambda a, b, out: out,
                                              addmm=lambda c, a, b, alpha=1, beta=1, out=None: out),
            'cudnn': SimpleNamespace(_scaled_dot_product_cudnn_attention=sdpa),
            'triton_poi_fused__to_copy_0': pointwise('triton_poi_fused__to_copy_0'),
            'triton_red_fused_rms_1': reduction,
            'triton_poi_fused_unpack_18': pointwise('triton_poi_fused_unpack_18', '*fp16', '*fp16')}


# An Inductor-style wrapper with the official decoder's operation counts and tile layouts.
DECODER_WRAPPER = '''
class Runner:
    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1 = args
        args.clear()
        assert_size_stride(arg3_1, (1, 24, 7, 16, 16), (43008, 1792, 256, 16, 1))
        stream0 = get_raw_stream(0)
        buf0 = empty_strided_cuda((64, 64), (64, 1), torch.float16)
        triton_poi_fused__to_copy_0.run(arg0_1, buf0, 4096, stream=stream0)
        buf1 = empty_strided_cuda((64, ), (1, ), torch.float16)
        triton_poi_fused__to_copy_0.run(arg1_1, buf1, 64, stream=stream0)
        buf2 = empty_strided_cuda((1797, 64), (64, 1), torch.float16)
        buf3 = empty_strided_cuda((1797, 64), (64, 1), torch.float16)
        buf4 = empty_strided_cuda((1797, 2048), (2048, 1), torch.float16)
        buf5 = empty_strided_cuda((64, 1792), (1792, 1), torch.float16)
        for _ in range(686):
            triton_red_fused_rms_1.run(arg3_1, arg2_1, buf2, 1797, 64, stream=stream0)
        for _ in range(217):
            extern_kernels.addmm(buf1, buf2, reinterpret_tensor(buf0, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf3)
        extern_kernels.mm(reinterpret_tensor(buf5, (1792, 64), (1, 1792), 0),
                          reinterpret_tensor(buf0, (64, 64), (1, 64), 0),
                          out=reinterpret_tensor(buf3, (1792, 64), (64, 1), 0))
        q = reinterpret_tensor(buf4, (1, 32, 1797, 64), (3680256, 64, 2048, 1), 0)
        for _ in range(36):
            buf6 = cudnn._scaled_dot_product_cudnn_attention(q, q, q, None, False, scale=0.125)
        del buf6
        buf7 = empty_strided_cuda((1, 3, 28, 256, 256), (5505024, 1835008, 65536, 256, 1), torch.float16)
        triton_poi_fused_unpack_18.run(buf3, buf7, 5505024, stream=stream0)
        triton_poi_fused_unpack_18.run(buf3, buf7, 5505024, stream=stream0)
        return (buf7, )
'''


def test_wrapper_recorder_declares_roots_first_and_records_the_launch_abi(tmp_path):
    torch = pytest.importorskip('torch')
    from models.minimax_h3_official.prepare import inductor_capture as capture
    weight, bias, tile = torch.ones(4, 4), torch.ones(4), torch.ones(1, 4, dtype=torch.float16)
    source = '''
def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    buf0 = empty_strided_cuda((4, 4), (4, 1), torch.float16)
    triton_poi_fused__to_copy_0.run(arg0_1, buf0, 16, stream=None)
    buf1 = empty_strided_cuda((1, 4), (4, 1), torch.float16)
    extern_kernels.addmm(arg1_1, arg2_1, reinterpret_tensor(buf0, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf1)
    return (buf1, )
'''
    scope = namespace(torch)
    roots = [{'parameter': 'w', 'spec': {}}, {'parameter': 'b', 'spec': {}}, {'input': 'tile'}]
    recorder, result = capture.record_call(source, scope, [weight, bias, tile], roots, tmp_path)
    assert [(b['role'], b['shape']) for b in recorder.buffers] == [
        ('input', [1, 4]), ('parameter', [4, 4]), ('parameter', [4]), ('scratch', [4, 4]), ('scratch', [1, 4])]
    kernel, addmm = recorder.ops
    sha = hashlib.sha256(f"triton_poi_fused__to_copy_0:{[('XBLOCK', 256)]}:4".encode()).hexdigest()
    assert kernel == {'kind': 'kernel', 'kernel': sha, 'grid': [1, 1, 1], 'args': [
        {'type': 'pointer', 'buffer': 1, 'offset': 0, 'shape': [4, 4], 'stride': [4, 1], 'dtype': 'torch.float32'},
        {'type': 'pointer', 'buffer': 3, 'offset': 0, 'shape': [4, 4], 'stride': [4, 1], 'dtype': 'torch.float16'},
        {'type': 'i32', 'bits': 16}, {'type': 'u64', 'bits': 0}, {'type': 'u64', 'bits': 0}]}
    entry = recorder.kernels[sha]
    assert (tmp_path / entry['file']).read_bytes() and digest(tmp_path / entry['file']) == sha
    assert entry['symbol'] == 'triton_poi_fused__to_copy_0' and entry['hidden_null_args'] == 2
    assert entry['config'] == {'kwargs': {'XBLOCK': 256}, 'num_warps': 4}
    assert addmm['args'][2] == {'buffer': 3, 'offset': 0, 'shape': [4, 4], 'stride': [1, 4], 'dtype': 'torch.float16'}
    assert addmm['kwargs']['out']['buffer'] == 4 and addmm['kwargs']['alpha'] == 1
    assert recorder.reference(result[0])['buffer'] == 4
    assert recorder.ignored == {'args.clear': 1, 'reinterpret_tensor': 1}


@pytest.mark.parametrize('line, error', [
    ('buf0.zero_()', 'unrecorded wrapper operation'),
    ('torch.ops.aten.mul.Tensor(buf0, buf0)', 'unrecorded wrapper operation'),
    ('extern_kernels.bmm(buf0, buf0, out=buf0)', 'unsupported extern operator'),
    ('triton_poi_fused__to_copy_0.run(frozen, buf0, 16, stream=None)', 'undeclared storage'),
])
def test_wrapper_recorder_refuses_unrecorded_computation(tmp_path, line, error):
    torch = pytest.importorskip('torch')
    from models.minimax_h3_official.prepare import inductor_capture as capture
    source = f'''
def call(args):
    arg0_1, = args
    buf0 = empty_strided_cuda((4, 4), (4, 1), torch.float16)
    {line}
    return (buf0, )
'''
    scope = {**namespace(torch), 'frozen': torch.ones(4, 4)}
    with pytest.raises(RuntimeError, match=error):
        capture.record_call(source, scope, [torch.ones(4)], [{'input': 'x'}], tmp_path)


def test_kernel_selection_is_reported_and_can_be_pinned():
    torch = pytest.importorskip('torch')
    from models.minimax_h3_official.prepare import inductor_capture as capture
    scope = namespace(torch)
    before = capture.selected_kernels(scope)
    assert sorted(before) == ['triton_poi_fused__to_copy_0', 'triton_poi_fused_unpack_18', 'triton_red_fused_rms_1']
    assert before['triton_red_fused_rms_1']['kwargs'] == {'XBLOCK': 2, 'R0_BLOCK': 512}
    wanted = {'triton_red_fused_rms_1': {'kwargs': {'XBLOCK': 2, 'R0_BLOCK': 1024}, 'num_warps': 8},
              'triton_poi_fused_unpack_18': before['triton_poi_fused_unpack_18']}
    assert capture.force_kernel_configs(scope, wanted) == ['triton_red_fused_rms_1']
    after = capture.selected_kernels(scope)
    assert after['triton_red_fused_rms_1']['kwargs']['R0_BLOCK'] == 1024
    assert after['triton_red_fused_rms_1']['sha256'] != before['triton_red_fused_rms_1']['sha256']
    assert capture.config_digest(after) != capture.config_digest(before)
    with pytest.raises(ValueError, match='absent from the graph'):
        capture.force_kernel_configs(scope, {'triton_missing_3': {'kwargs': {}, 'num_warps': 1}})


# ---- the package end to end ----------------------------------------------------------------------

@pytest.fixture
def prepared_checkpoint(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    vae = tiny_vae()
    state = {name: value for name, value in vae.state_dict().items()}
    names = sorted(state)
    for index, part in enumerate((names[::2], names[1::2])):
        safetensors(tmp_path / f'ckpt/vae/diffusion_pytorch_model-0000{index + 1}-of-00002.safetensors',
                    {name: ('F32', list(state[name].shape), state[name].contiguous().numpy().tobytes())
                     for name in part})
    config = {'_class_name': 'AutoencoderKLMiniMaxH3', 'latents_mean': [0.1] * 24, 'latents_std': [2.0] * 24}
    (tmp_path / 'ckpt/vae/config.json').write_text(json.dumps(config))
    plan = tmp_path / 'video-encoder.json'
    plan.write_text(json.dumps(fl_plan(vae), indent=2))
    monkeypatch.setattr(video_vae, 'ENCODER_PLAN', plan)
    monkeypatch.setattr(video_vae, 'ENCODER_PLAN_SHA256', digest(plan))
    monkeypatch.setattr(video_vae, 'load_vae', lambda checkpoint: vae)
    environment = {'torch': '2.13.0+cu130', 'torch_git': video_vae.TORCH_GIT, 'device': 'synthetic'}
    monkeypatch.setattr(video_vae, 'environment', lambda: environment)

    def capture_decoder(vae, root, *, audit_root, audit, audit_sha, backend, environment, kernel_configs=None):
        """The GPU capture after torch.compile: the same recording, of an Inductor-style wrapper."""
        from models.minimax_h3_official.prepare import inductor_capture as capture
        root.mkdir(parents=True)
        tile = torch.zeros(video_vae.DECODER_TILE, dtype=torch.float16)
        inputs = [vae.decoder.proj.weight, vae.decoder.proj.bias, vae.decoder.rope.inv_freq, tile]
        scope = namespace(torch)
        forced = capture.force_kernel_configs(scope, kernel_configs) if kernel_configs else []
        roots = video_vae.graph_roots(vae.decoder, 'decoder.', audit, inputs, ('decoder_tile', tile))
        program, _ = video_vae.decoder_program(DECODER_WRAPPER, scope, inputs, roots, root, audit_root=audit_root,
                                               audit_sha=audit_sha, backend=backend, environment=environment)
        return program, {'forced_kernels': forced}

    monkeypatch.setattr(video_vae, 'capture_decoder', capture_decoder)
    return tmp_path, vae


def test_prepare_writes_native_ref2va_packages(prepared_checkpoint):
    tmp_path, vae = prepared_checkpoint
    configs = tmp_path / 'selection.json'
    selection = {'triton_red_fused_rms_1': {'kwargs': {'XBLOCK': 2, 'R0_BLOCK': 1024}, 'num_warps': 8}}
    configs.write_text(json.dumps(selection))
    report = video_vae.prepare_video_vae(tmp_path / 'ckpt', tmp_path / 'out', workflow='ref2va',
                                         decoder_kernel_configs=configs, workers=2)
    json.dumps(report)
    root = tmp_path / 'out/video_vae'
    model = json.loads((root / 'model.json').read_bytes())
    assert model['schema'] == 'h3_native_video_vae_v1' and model['audit_root'] == str(root) and root.is_absolute()
    assert model['audit_manifest_sha256'] == digest(root / 'manifest.json') == report['audit_manifest_sha256']
    assert report['model_sha256'] == digest(root / 'model.json')
    for stage in ('encoder', 'decoder'):
        assert model[stage] == {'root': str(root / stage), 'program_sha256': digest(root / stage / 'program.json')}

    audit = json.loads((root / 'manifest.json').read_bytes())
    state = dict(vae.state_dict())
    assert sorted(audit['parameters']) == sorted(state) and report['tensors'] == len(state)
    for name, spec in audit['parameters'].items():
        raw = state[name].contiguous().numpy().tobytes()
        assert (root / spec['file']).read_bytes() == raw and spec['sha256'] == hashlib.sha256(raw).hexdigest()
    inv_freq = audit['buffers']['decoder.rope.inv_freq']
    assert inv_freq['file'] == 'buffers/decoder.rope.inv_freq.raw'
    assert inv_freq['sha256'] == digest(root / inv_freq['file'])
    assert audit['parameters']['post_quant_conv.weight']['dtype'] == 'torch.float32'

    encoder = json.loads((root / 'encoder/program.json').read_bytes())
    assert encoder['audit_root'] == str(root) and video_vae.validate_program(encoder, root / 'encoder', FL)
    assert all(b['source']['spec'] == audit['parameters'][b['source']['parameter']]
               for b in encoder['buffers'] if b['role'] == 'parameter')

    decoder = json.loads((root / 'decoder/program.json').read_bytes())
    stage = 'video_vae_decoder_tile'
    assert video_vae.validate_program(decoder, root / 'decoder', stage) == video_vae.DECODER_COUNTS
    assert decoder['buffers'][0]['role'] == 'input' and decoder['buffers'][0]['source'] == {'input': 'decoder_tile'}
    assert {b['source']['parameter'] for b in decoder['buffers'] if b['role'] == 'parameter'} == {
        'decoder.proj.weight', 'decoder.proj.bias', 'decoder.rope.inv_freq'}
    assert decoder['kernel_configs']['triton_red_fused_rms_1']['kwargs'] == {'XBLOCK': 2, 'R0_BLOCK': 1024}
    assert decoder['compile'] == {'mode': 'default', 'dynamic': False, 'autocast_dtype': 'torch.float16'}
    assert decoder['wrapper_sha256'] == digest(root / 'decoder/wrapper.py')
    assert decoder['gold'] == {'dtype': 'torch.float16', 'shape': video_vae.DECODER_OUTPUT, 'bytes': 11010048}
    assert report['decoder']['forced_kernels'] == ['triton_red_fused_rms_1']
    contract = report['decoder']['small_tile_contract']
    assert not contract['consistent'] and contract['kernels'] == {'1': 'triton_red_fused_rms_1',
                                                                 '18': 'triton_poi_fused_unpack_18'}
    assert any('R0_BLOCK 1024' in problem for problem in contract['problems'])

    reference = json.loads((tmp_path / 'out/video_encoder/program.json').read_bytes())
    assert reference['stage'] == 'video_vae_encoder_clip17_tile' and reference['audit_root'] == str(root)
    assert video_vae.validate_program(reference, tmp_path / 'out/video_encoder',
                                      'video_vae_encoder_clip17_tile') == video_vae.CLIP_COUNTS
    assert report['video_encoder']['program_sha256'] == digest(tmp_path / 'out/video_encoder/program.json')
    assert report['pinned_torch'] is True

    (root / 'decoder' / decoder['kernels'][decoder['ops'][0]['kernel']]['file']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed CUBIN'):
        video_vae.validate_program(decoder, root / 'decoder', stage)
    (root / 'manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='audit manifest identity'):
        video_vae.validate_program(encoder, root / 'encoder', FL)
    with pytest.raises(FileExistsError):
        video_vae.prepare_video_vae(tmp_path / 'ckpt', tmp_path / 'out', workflow='ref2va')


def test_prepare_fl2va_keeps_inductor_selection_and_writes_no_reference_encoder(prepared_checkpoint):
    import torch
    if not torch.__version__.startswith(TORCH):
        pytest.skip(f'Inductor kernel names are pinned for torch {TORCH} (preparation only)')
    tmp_path, _ = prepared_checkpoint
    report = video_vae.prepare_video_vae(tmp_path / 'ckpt', tmp_path / 'out', workflow='fl2va', workers=2)
    decoder = json.loads((tmp_path / 'out/video_vae/decoder/program.json').read_bytes())
    assert decoder['kernel_configs']['triton_red_fused_rms_1']['kwargs'] == {'XBLOCK': 2, 'R0_BLOCK': 512}
    assert report['decoder']['forced_kernels'] == [] and 'video_encoder' not in report
    assert not (tmp_path / 'out/video_encoder').exists()


def test_prepare_refuses_a_checkpoint_that_differs_from_the_encoder_plan(prepared_checkpoint, monkeypatch):
    tmp_path, _ = prepared_checkpoint
    plan = json.loads(video_vae.ENCODER_PLAN.read_bytes())
    next(b for b in plan['buffers'] if b['role'] == 'parameter')['source']['spec']['sha256'] = '0' * 64
    video_vae.ENCODER_PLAN.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='differs from its pinned structure'):
        video_vae.prepare_video_vae(tmp_path / 'ckpt', tmp_path / 'out', workflow='fl2va')
    monkeypatch.setattr(video_vae, 'ENCODER_PLAN_SHA256', digest(video_vae.ENCODER_PLAN))
    with pytest.raises(ValueError, match='differs from the encoder plan'):
        video_vae.prepare_video_vae(tmp_path / 'ckpt', tmp_path / 'other', workflow='fl2va')
