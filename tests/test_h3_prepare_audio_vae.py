"""CPU checks of the H3 audio VAE graph capture and packaging, on a tiny synthetic audio VAE.

The synthetic model mirrors the official module structure (weight-normalized convolutions,
alias-free SnakeBeta activations, transposed-convolution upsamplers, final clamp, a strided
encoder with a projection block), so the real capture, contract checks and publishing run
end to end without Diffusers, a GPU or the official checkpoint. Torch is imported only when
these tests run.
"""
import json
import math
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest

from models.minimax_h3_official.prepare import audio_vae, common

pytestmark = pytest.mark.filterwarnings('ignore:.*weight_norm.*:FutureWarning')
TOKENS, SAMPLES = 3, 2400
DECODER_NODES = 105  # step 2 + projections 2 + 3 stages x 29 + post 13 + permute 1
CONFIG = {'_class_name': 'AutoencoderKLMiniMaxH3Audio', 'sampling_rate': 32000, 'latent_channels': 32,
          'latents_mean': [0.01 * i - 0.1 for i in range(32)], 'latents_std': [1.0 + 0.05 * i for i in range(32)]}


def synthetic_models(torch):
    nn, F = torch.nn, torch.nn.functional
    weight_norm = nn.utils.weight_norm

    def lowpass(kernel=12):
        window = torch.kaiser_window(kernel, periodic=False, beta=4.0)
        return (window / window.sum()).view(1, 1, kernel)

    class SnakeBeta(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.alpha, self.beta = nn.Parameter(0.1 * torch.randn(channels)), nn.Parameter(0.1 * torch.randn(channels))

        def forward(self, x):
            alpha = torch.exp(self.alpha.unsqueeze(0).unsqueeze(-1))
            beta = torch.exp(self.beta.unsqueeze(0).unsqueeze(-1))
            return x + (beta + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)

    class Activation(nn.Module):
        """2x upsample, SnakeBeta, 2x downsample, as the decoder's anti-aliased activation."""

        def __init__(self, channels):
            super().__init__()
            self.act = SnakeBeta(channels)
            self.register_buffer('up', lowpass())
            self.register_buffer('down', lowpass())

        def forward(self, x):
            c = x.shape[1]
            x = 2 * F.conv_transpose1d(F.pad(x, (5, 5), mode='replicate'), self.up.expand(c, -1, -1), stride=2, groups=c)
            x = self.act(x[..., 15:-15])
            return F.conv1d(F.pad(x, (5, 6), mode='replicate'), self.down.expand(c, -1, -1), stride=2, groups=c)

    class Residual(nn.Module):
        def __init__(self, channels, dilation):
            super().__init__()
            self.act = Activation(channels)
            self.conv = weight_norm(nn.Conv1d(channels, channels, 3, dilation=dilation, padding=dilation))

        def forward(self, x):
            return x + self.conv(self.act(x))

    class Decoder(nn.Module):
        def __init__(self, terminal=None):
            super().__init__()
            self.terminal = terminal or (lambda x: torch.clamp(x, min=-1.0, max=1.0))
            self.conv_pre = weight_norm(nn.Conv1d(8, 8, 7, 1, padding=3))
            stages = ((8, 4, 8), (4, 4, 10), (4, 2, 10))  # 800x
            self.ups = nn.ModuleList(weight_norm(nn.ConvTranspose1d(a, b, 2 * r, r, padding=r // 2)) for a, b, r in stages)
            self.blocks = nn.ModuleList(nn.ModuleList([Residual(b, 1), Residual(b, 3)]) for _, b, _ in stages)
            self.activation_post = Activation(2)
            self.conv_post = weight_norm(nn.Conv1d(2, 1, 7, 1, padding=3, bias=False))

        def forward(self, x):
            x = self.conv_pre(x)
            for up, (first, second) in zip(self.ups, self.blocks):
                x = up(x)
                x = (first(x) + second(x)) / 2
            return self.terminal(self.conv_post(self.activation_post(x)))

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_in = weight_norm(nn.Conv1d(1, 4, 7, padding=3))
            self.alpha = nn.Parameter(torch.ones(1, 4, 1))
            self.downs = nn.Sequential(weight_norm(nn.Conv1d(4, 8, 16, 8, padding=4)),
                                       weight_norm(nn.Conv1d(8, 16, 20, 10, padding=5)),
                                       weight_norm(nn.Conv1d(16, 16, 20, 10, padding=5)))  # 800x

        def forward(self, x):
            x = self.conv_in(x)
            return self.downs(x + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * x).pow(2))

    class TinyAudioVAE(nn.Module):
        def __init__(self, config, terminal=None):
            super().__init__()
            self.config = SimpleNamespace(**config)
            self.encoder = Encoder()
            self.norm = nn.LayerNorm(16)
            self.qkv = nn.Linear(16, 48, bias=False)
            self.q_bias, self.v_bias = nn.Parameter(torch.randn(16)), nn.Parameter(torch.randn(16))
            self.register_buffer('zero_k_bias', torch.zeros(16))
            self.proj = nn.Linear(4, 16)
            self.mean_proj, self.logs_proj = nn.Conv1d(16, 32, 1), nn.Conv1d(16, 32, 1)
            self.dec_in_proj = nn.Conv1d(32, 8, 1)
            self.decoder = Decoder(terminal)

        @classmethod
        def from_pretrained(cls, folder, torch_dtype, local_files_only):
            assert torch_dtype == torch.float32 and local_files_only
            folder = Path(folder)
            model = cls(json.loads((folder / 'config.json').read_bytes()))
            source = common.Checkpoint(folder.parent, folder.name)
            model.load_state_dict({name: torch.frombuffer(bytearray(source.read(name)), dtype=torch.float32)
                                   .reshape(source.layout(name)[1]) for name in source.tensors})
            return model

        def encode(self, sample, return_dict=True):
            h = self.encoder(sample).transpose(1, 2)  # [2, frames, 16]
            batch, frames, width = h.shape
            qkv = F.linear(self.norm(h).reshape(-1, width), self.qkv.weight,
                           bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))).view(batch, frames, 3 * width)
            q, k, v = qkv.reshape(batch, frames, 3, 2, 8).permute(2, 0, 1, 3, 4).unbind(0)
            mixed = F.adaptive_avg_pool1d(torch.mean(q * k + v, dim=2), 4)
            mixed = F.linear(mixed.reshape(-1, 4), self.proj.weight, self.proj.bias).view(batch, frames, width)
            h = F.gelu(h + mixed, approximate='tanh').transpose(1, 2)
            mean, logs = self.mean_proj(h), self.logs_proj(h)
            return (SimpleNamespace(mode=lambda: mean, std=torch.exp(logs)),)

        def decode(self, latents, return_dict=True):
            return (self.decoder(self.dec_in_proj(latents)),)

    def variant(terminal):
        class Variant(TinyAudioVAE):
            def __init__(self, config):
                super().__init__(config, terminal)
        return Variant

    class State:
        def __init__(self, values):
            self.values = dict(values)

        def get(self, key):
            return self.values.get(key)

    class DecodeStep:
        """Same operations as Diffusers' MiniMaxH3AudioDecodeStep."""

        @torch.no_grad()
        def __call__(self, components, state):
            device = components._execution_device
            mean = torch.tensor(components.audio_vae.config.latents_mean, device=device).view(1, -1, 1)
            std = torch.tensor(components.audio_vae.config.latents_std, device=device).view(1, -1, 1)
            audio = components.audio_vae.decode(state.get('audio_latents') * std + mean, return_dict=False)[0]
            state.values['audio'] = audio.float().permute(1, 0, 2)
            return components, state

    return SimpleNamespace(torch=torch, nn=nn, TinyAudioVAE=TinyAudioVAE, variant=variant, State=State,
                           DecodeStep=DecodeStep)


@pytest.fixture(scope='module')
def tiny():
    return synthetic_models(pytest.importorskip('torch'))


def write_safetensors(path, tensors):
    header, payload = {}, b''
    for name, value in tensors.items():
        raw = value.detach().contiguous().numpy().tobytes()
        header[name] = {'dtype': 'F32', 'shape': list(value.shape), 'data_offsets': [len(payload), len(payload) + len(raw)]}
        payload += raw
    encoded = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack('<Q', len(encoded)) + encoded + payload)


@pytest.fixture
def checkpoint(tmp_path, monkeypatch, tiny):
    tiny.torch.manual_seed(0)
    root = tmp_path / 'checkpoint'
    write_safetensors(root / 'audio_vae/diffusion_pytorch_model.safetensors', tiny.TinyAudioVAE(CONFIG).state_dict())
    (root / 'audio_vae/config.json').write_text(json.dumps(CONFIG))
    monkeypatch.setattr(audio_vae, '_official', lambda: (tiny.TinyAudioVAE, tiny.DecodeStep, tiny.State))
    monkeypatch.setattr(audio_vae, 'DECODER_NODES', DECODER_NODES)
    return root


def tensor_id(value):
    return value.get('tensor') if isinstance(value, dict) else None


def references(value):
    if tensor_id(value):
        yield value['tensor']
    elif isinstance(value, (list, dict)):
        for item in value.values() if isinstance(value, dict) else value:
            yield from references(item)


def native_shapes(manifest, tokens):
    """Mirror of shape_rules.rs infer_graph: shapes from operator semantics for any length."""
    shapes = {name: spec['shape'] for name, spec in manifest['constants'].items()}
    shapes['input'] = [2, 32, tokens]
    for index, node in enumerate(manifest['nodes']):
        assert node['id'] == f'n{index:04}' and node['output']['dtype'] == 'torch.float32'
        op, args = node['op'], node['args']
        a = shapes[tensor_id(args[0])]
        out = list(a)
        if op == 'aten.convolution.default':
            w = shapes[tensor_id(args[1])]
            stride, padding, dilation, transposed, extra, groups = args[3][0], args[4][0], args[5][0], args[6], args[7][0], args[8]
            kernel = dilation * (w[2] - 1) + 1
            out = ([a[0], w[1] * groups, (a[2] - 1) * stride - 2 * padding + kernel + extra] if transposed
                   else [a[0], w[0], (a[2] + 2 * padding - kernel) // stride + 1])
            assert args[2] is None or shapes[tensor_id(args[2])] == [out[1]]
        elif op == 'aten.replication_pad1d.default':
            out[2] = a[2] + args[1][0] + args[1][1]
        elif op == 'aten.slice.Tensor':
            dim = args[1] % len(a)
            norm = lambda v: max(v + a[dim], 0) if v < 0 else min(v, a[dim])  # noqa: E731
            start, end, step = norm(args[2]), norm(args[3]), args[4] if len(args) > 4 else 1
            assert end > start and step > 0
            out[dim] = (end - start - 1) // step + 1
        elif op == 'aten.permute.default':
            out = [a[d] for d in args[1]]
        elif op in ('aten.mul.Tensor', 'aten.add.Tensor', 'aten.div.Tensor') and tensor_id(args[1]):
            b = shapes[tensor_id(args[1])]
            n = max(len(a), len(b))
            out = [max(x, y) for x, y in zip([1] * (n - len(a)) + a, [1] * (n - len(b)) + b)]
        else:
            assert op in ('aten.mul.Tensor', 'aten.div.Tensor', 'aten.sin.default', 'aten.pow.Tensor_Scalar',
                          'aten.clamp.default'), op
        shapes[node['id']] = out
    return shapes


def replay(torch, root, manifest, sample):
    """Execute a manifest with ATen, as the native runtime executes it, from constants only."""
    env = {name: torch.frombuffer(bytearray((root / spec['file']).read_bytes()), dtype=torch.float32)
           .reshape(spec['shape']) for name, spec in manifest['constants'].items()}
    env['input'] = sample

    def bind(value):
        if tensor_id(value):
            return env[value['tensor']]
        return [bind(item) for item in value] if isinstance(value, list) else value

    with torch.no_grad():
        for node in manifest['nodes']:
            _, name, overload = node['op'].split('.')
            op, args = getattr(getattr(torch.ops.aten, name), overload), bind(node['args'])
            kwargs = {key: bind(value) for key, value in node['kwargs'].items()}
            result = op(*args[:-1], **kwargs)[args[-1]] if node['op'] == 'aten.unbind.int' else op(*args, **kwargs)
            env[node['id']] = result[0] if isinstance(result, tuple) else result
    return env[manifest['output_id']]


def check_package(root, report, schema):
    manifest = json.loads((root / 'manifest.json').read_bytes())
    assert manifest['schema'] == schema and manifest['status'] == 'complete'
    files = {p.relative_to(root.parent).as_posix() for p in root.rglob('*') if p.is_file()}
    assert files == set(report['files']) == {f'{root.name}/manifest.json'} | {
        f'{root.name}/constants/{name}.raw' for name in manifest['constants']}
    assert all(common.digest(root.parent / path) == sha for path, sha in report['files'].items())
    assert report['manifest_sha256'] == common.digest(root / 'manifest.json')
    for name, spec in manifest['constants'].items():
        assert spec['file'] == f'constants/{name}.raw' and spec['dtype'] == 'torch.float32'
        assert spec['bytes'] == 4 * math.prod(spec['shape']) == (root / spec['file']).stat().st_size
        assert spec['sha256'] == common.digest(root / spec['file']) and spec['derived_from']
    known = set(manifest['constants']) | {'input'}
    for index, node in enumerate(manifest['nodes']):
        assert node['id'] == f'n{index:04}' and isinstance(node['args'], list) and isinstance(node['kwargs'], dict)
        assert node['output'] == {'file': audio_vae.UNREFERENCED, 'dtype': 'torch.float32',
                                  'shape': node['output']['shape'], 'bytes': 4 * math.prod(node['output']['shape']),
                                  'sha256': ''}
        assert set(references([node['args'], node['kwargs']])) <= known
        known.add(node['id'])
    for spec in (manifest['input'], manifest['gold']):
        assert spec['file'] == audio_vae.UNREFERENCED and spec['sha256'] == '' and spec['dtype'] == 'torch.float32'
        assert spec['bytes'] == 4 * math.prod(spec['shape'])
    provenance = manifest['provenance']
    for flag in ('cudnn_allow_tf32', 'cudnn_deterministic', 'allow_tf32', 'no_intermediate_initializers'):
        assert type(provenance[flag]) is bool
    assert provenance['no_intermediate_initializers'] and provenance['model_revision'] == common.REVISION
    assert provenance['diffusers_commit'] == common.DIFFUSERS_COMMIT and not provenance['reference_values_available']
    return manifest


def test_decoder_package_is_one_native_template_for_every_length(tmp_path, checkpoint, tiny):
    torch, out = tiny.torch, tmp_path / 'out'
    report = audio_vae.prepare_audio_vae(checkpoint, out, workflow='fl2va', device='cpu', tokens=TOKENS)
    assert sorted(p.name for p in out.iterdir()) == ['audio_vae']
    root = out / 'audio_vae'
    manifest = check_package(root, report['decoder'], 'h3-audio-vae-graph-v1')
    nodes = manifest['nodes']
    assert len(nodes) == report['decoder']['nodes'] == DECODER_NODES and manifest['output_id'] == nodes[-1]['id']
    assert manifest['input']['shape'] == [2, 32, TOKENS] and manifest['gold']['shape'] == [2, 800 * TOKENS]
    # The stock clamp stays, feeding the stereo permute.
    assert nodes[-2]['op'] == 'aten.clamp.default' and nodes[-2]['args'] == [{'tensor': nodes[-3]['id']}, -1.0, 1.0]
    assert nodes[-1]['op'] == 'aten.permute.default' and nodes[-1]['args'] == [{'tensor': nodes[-2]['id']}, [1, 0, 2]]
    assert set(report['decoder']['ops']) <= audio_vae.DECODER_OPS and manifest['provenance']['template_tokens'] == TOKENS
    # read_template's semantic self-check, then other lengths from the same template.
    shapes = native_shapes(manifest, TOKENS)
    assert all(shapes[node['id']] == node['output']['shape'] for node in nodes)
    for tokens in (1, 7, 41):
        assert native_shapes(manifest, tokens)[manifest['output_id']] == [1, 2, 800 * tokens]
    # Constants: declared statistics first, then model state; folded values are recomputed from the checkpoint.
    constants, source = manifest['constants'], common.Checkpoint(checkpoint, 'audio_vae')
    assert nodes[0]['args'] == [{'tensor': 'input'}, {'tensor': 'c0000'}]
    assert constants['c0000']['derived_from'] == ['latents_std'] and constants['c0001']['derived_from'] == ['latents_mean']
    assert constants['c0002']['derived_from'] == ['dec_in_proj.weight'] and not constants['c0002']['folded']
    assert (root / 'constants/c0002.raw').read_bytes() == source.read('dec_in_proj.weight')

    def load(name):
        return torch.frombuffer(bytearray(source.read(name)), dtype=torch.float32).reshape(source.layout(name)[1])
    folded = torch._weight_norm(load('decoder.conv_pre.weight_v'), load('decoder.conv_pre.weight_g'), 0)
    assert constants['c0004']['derived_from'] == ['decoder.conv_pre.weight_g', 'decoder.conv_pre.weight_v']
    assert constants['c0004']['folded'] and (root / 'constants/c0004.raw').read_bytes() == folded.numpy().tobytes()
    # The graph alone reproduces the eager decode at another length, bit for bit.
    model = tiny.TinyAudioVAE.from_pretrained(checkpoint / 'audio_vae', torch.float32, True).eval()
    latents = torch.randn(2, 32, 5, generator=torch.Generator().manual_seed(1))
    state = tiny.State({'audio_latents': latents})
    tiny.DecodeStep()(SimpleNamespace(audio_vae=model, _execution_device=torch.device('cpu')), state)
    assert torch.equal(replay(torch, root, manifest, latents), state.get('audio'))
    with pytest.raises(FileExistsError):
        audio_vae.prepare_audio_vae(checkpoint, out, workflow='fl2va', device='cpu', tokens=TOKENS)


def test_ref2va_also_writes_the_hop_aligned_reference_encoder(tmp_path, checkpoint, tiny):
    torch, out = tiny.torch, tmp_path / 'out'
    with pytest.raises(ValueError, match='multiple of the hop'):
        audio_vae.prepare_audio_vae(checkpoint, out, workflow='ref2va', device='cpu', tokens=TOKENS, samples=SAMPLES + 1)
    assert not out.exists()
    report = audio_vae.prepare_audio_vae(checkpoint, out, workflow='ref2va', device='cpu', tokens=TOKENS, samples=SAMPLES)
    assert sorted(p.name for p in out.iterdir()) == ['audio_encoder', 'audio_vae']
    root = out / 'audio_encoder'
    manifest = check_package(root, report['encoder'], 'h3-audio-encoder-graph-v1')
    assert manifest['input']['shape'] == [2, 1, SAMPLES] and manifest['gold']['shape'] == [2, 32, SAMPLES // 800]
    assert manifest['provenance']['no_intermediate_initializers'] is True
    assert manifest['latents_mean'] == CONFIG['latents_mean'] and manifest['latents_std'] == CONFIG['latents_std']
    ops = [node['op'] for node in manifest['nodes']]
    assert set(ops) <= audio_vae.ENCODER_OPS and 'aten.constant_pad_nd.default' not in ops
    # The unused log-std head and its exp are pruned; so are its parameters.
    assert report['encoder']['pruned_nodes'] == 2 and 'aten.exp.default' not in ops
    assert not any('logs_proj' in source for spec in manifest['constants'].values() for source in spec['derived_from'])
    split = [node['args'] for node in manifest['nodes'] if node['op'] == 'aten.unbind.int']
    assert [args[1:] for args in split] == [[0], [1], [2]] and len({args[0]['tensor'] for args in split}) == 1
    model = tiny.TinyAudioVAE.from_pretrained(checkpoint / 'audio_vae', torch.float32, True).eval()
    waveform = 0.1 * torch.randn(2, 1, SAMPLES, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        expected = model.encode(waveform, return_dict=False)[0].mode()
    torch.testing.assert_close(replay(torch, root, manifest, waveform), expected, rtol=1e-5, atol=1e-6)


def test_capture_admits_only_model_state_declared_literals_and_the_input(tiny):
    from models.minimax_h3_official.prepare.aten_capture import Capture
    torch = tiny.torch
    sample, weight, stray = torch.randn(2, 3, 4), tiny.nn.Parameter(torch.randn(1, 3, 1)), torch.randn(1, 3, 1)
    cases = {'neither model state': lambda x: x * stray,
             'undeclared literal': lambda x: x * torch.tensor([0.5, 1.5, 2.5]).view(1, -1, 1),
             'in-place': lambda x: (x * weight).add_(1)}
    for message, call in cases.items():
        with pytest.raises(RuntimeError, match=message), torch.no_grad(), \
                Capture(sample, {'weight': weight}, {'scale': torch.tensor([0.5, 1.5])}):
            call(sample)
    capture = Capture(sample, {'weight': weight}, {'scale': torch.tensor([0.5, 1.5, 2.5])})
    with torch.no_grad(), capture:
        result = sample * torch.exp(weight) + torch.tensor([0.5, 1.5, 2.5]).view(1, -1, 1)
    graph = capture.graph(result)
    assert [node['op'] for node in graph['nodes']] == ['aten.mul.Tensor', 'aten.add.Tensor']
    assert {name: (sources, folded) for name, (_, sources, folded) in graph['constants'].items()} == {
        'c0000': (['weight'], True), 'c0001': (['scale'], True)}
    torch.testing.assert_close(graph['constants']['c0000'][0], torch.exp(weight.detach()))


def test_decoder_outside_the_native_contract_publishes_nothing(tmp_path, checkpoint, monkeypatch, tiny):
    out = tmp_path / 'out'
    monkeypatch.setattr(audio_vae, 'DECODER_NODES', 1621)
    with pytest.raises(ValueError, match='admits 1621'):
        audio_vae.prepare_audio_vae(checkpoint, out, workflow='fl2va', device='cpu', tokens=TOKENS)
    monkeypatch.setattr(audio_vae, 'DECODER_NODES', None)
    for terminal, message in ((lambda x: x * 1.0, 'stock clamp'), (tiny.torch.tanh, 'no operator aten.tanh.default')):
        monkeypatch.setattr(audio_vae, '_official', lambda: (tiny.variant(terminal), tiny.DecodeStep, tiny.State))
        with pytest.raises(ValueError, match=message):
            audio_vae.prepare_audio_vae(checkpoint, out, workflow='fl2va', device='cpu', tokens=TOKENS)
    assert not out.exists()
