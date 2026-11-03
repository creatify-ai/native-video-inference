"""Official-model callbacks over the one shared Rust startup executor."""
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import uuid

from core_infra.runtime.declared_startup import DeclaredStartup, Preparation

from .artifacts import prepare_backend
from . import conditioning, preparation
from .startup import components


class FrameworkH3Runtime:
    def __init__(self, config):
        self.config = config
        self.options = config.get('runtime', {})
        if self.options.get('startup_engine', 'framework') != 'framework':
            raise ValueError('official H3 requires the shared framework startup executor')
        if (self.options.get('layout_preparation', True) is not True
                or self.options.get('conditioning_preparations', True) is not True):
            raise ValueError('official H3 requires layout and conditioning preparations')
        self.startup = self.group = self.directory = None
        self.owners = {}
        self.closed = False
        self.started = None
        self.requests_started = 0
        self.layout_future = None
        self.payload = None

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.close()

    def _request_components(self, request):
        visual = bool(request.get('first_png'))
        audio = has_video = False
        if self.config['workflow'] == 'ref2va':
            refs = conditioning._references(request)
            visual = any(r['type'] in ('image', 'video') for r in refs)
            audio = any(r.get('has_audio') for r in refs)
            has_video = any(r['type'] == 'video' for r in refs)
        return ['conditioner', 'dit', *(['video_encoder'] if visual else []),
                *(['audio_encoder'] if audio else [])], has_video

    def start(self, output, environment, request):
        if self.closed:
            raise RuntimeError('H3 runtime is closed')
        inputs, has_video = self._request_components(request)
        attention = request.get('attention_backend', self.config.get('attention_backend', 'subblock_sparse_attn'))
        if self.started is not None:
            if attention != self.attention_backend:
                raise ValueError('H3 session attention backend changed; start a new session')
            if any(n not in self.input_names for n in inputs) or has_video and not self.has_video_reference:
                raise ValueError('H3 session needs a new input component; start a new session')
            if environment != self.request_environment:
                raise ValueError('H3 session environment changed')
            self.output = Path(output)
            self.layout_future = None
            self.requests_started += 1
            return
        self.started = time.monotonic()
        self.requests_started = 1
        self.attention_backend, self.request_environment = attention, dict(environment)
        self.output = Path(output)
        self.directory = tempfile.TemporaryDirectory(prefix='h3-official-startup-')
        self.root = Path(self.directory.name)
        self.logs = self.output / 'startup'
        self.logs.mkdir()
        parallel = self.options.get('context_parallel')
        world = parallel['world'] if parallel else 1
        # The video decoder stays on the first GPU: splitting its tiles builds its program once
        # per GPU, one after the other, which costs more than it saves at these sizes.
        self.environment = dict(environment, H3_CP_WORLD=str(world), H3_CP_RANK='0', H3_VIDEO_VAE_WORLD_SIZE='1')
        if parallel:
            # The DiT owner is the CP coordinator: it starts one DiT rank per GPU, each reading its
            # weight shards as slices of the DiT package.
            self.environment.update(H3_CP_RANK_PROGRAM=self.config['programs']['dit'], H3_DIT_SHARDS=parallel['shards'],
                H3_DIT_SHARDS_SHA256=parallel['manifest_sha256'], H3_DIT_SHARD_SLICES_SHA256=parallel['slices_sha256'],
                H3_CP_RANK_LOG_DIR=str(self.logs))
        self.environment.setdefault('NATIVE_CORE_READ_LAZY_ZEROED', '1')
        # Qwen builds the first request's plans beside its weight load, not after it.
        self.environment['H3_QWEN_STARTUP_HINT_FILE'] = str(self.root / 'qwen-startup-shape.json')
        if self.options.get('backend_plans'):
            self.environment.update(H3_CUDNN_PLAN_CACHE_MODE='read',
                                    H3_CUDNN_PLAN_CACHE_DIR=str(self.root / 'backend-plans'))
        self.input_names, self.has_video_reference = inputs, has_video
        self.declaration = components(self.config, inputs, has_video, self.root)
        library = (self.options.get('startup_library') or os.environ.get('NATIVE_STARTUP_LIBRARY')
                   or str(Path(self.config['programs']['dit']).parent / 'libnative_core_infra.so'))
        try:
            self.startup = DeclaredStartup(self.root, self.environment, self.declaration, self._preparations(),
                library=library, library_sha256=self.options.get('startup_library_sha256'),
                owner_factory=self._owner, workers=8,
                reader_budget_mib=self.options.get('reader_budget_mib', 6144),
                timeout=self.config.get('request_timeout_s', 900))
            self.group, self.owners = self.startup.group, self.startup.owners
            self.startup.start()
            self.backend_future = self.startup.static['backend']
        except BaseException:
            self.close()
            raise

    def _owner(self, component, environment, group):
        return group.owner(component.program, self.root / (component.name + '.sock'), environment,
            self.logs / (component.name + '.log'), timeout=self.config.get('request_timeout_s', 900),
            sha256=component.program_sha256)

    def _context(self, payload):
        return SimpleNamespace(config=self.config, output=Path(payload['output']), root=self.root,
            environment=self.environment, group=self.group, owners=self.owners, request_scope=payload['scope'])

    def _needed(self, payload, name):
        request = payload['request']
        if name == 'condition-video':
            return bool(payload['canvases']) if self.config['workflow'] == 'fl2va' else any(
                r['type'] in ('image', 'video') for r in conditioning._references(request))
        return any(r.get('has_audio') for r in conditioning._references(request))

    def _condition_valid(self, receipt, name, allow_skip=False):
        return conditioning.validate(receipt, name=name, scope=self.payload['scope'],
                                     root=self.payload['output'], allow_skip=allow_skip)

    def _backend_valid(self, receipt):
        pin = self.options.get('backend_plans')
        if (not isinstance(receipt, dict) or receipt.get('forward_executed') is not False
                or receipt.get('state') not in ('configured', 'native_builders')):
            raise RuntimeError('invalid official H3 backend preparation receipt')
        if receipt['state'] == 'configured':
            if not pin or receipt.get('manifest_sha256') != pin['sha256']:
                raise RuntimeError('backend preparation differs from its pinned bundle')
        elif pin and receipt.get('reason') != 'unmatched_driver':
            raise RuntimeError('backend preparation omitted the pinned bundle')
        return True

    def _layout_valid(self, receipt):
        owner = self.owners['dit']
        if (not isinstance(receipt, dict) or receipt.get('prepared') is not True
                or receipt.get('forward_executed') is not False or receipt.get('structure_only') is not True
                or receipt.get('request_scope') != self.payload['scope']
                or receipt.get('output') != self.payload['output']
                or receipt.get('owner_pid') != owner.process.pid or owner.process.poll() is not None
                or not isinstance(receipt.get('native'), dict)
                or receipt['native'].get('prepared') is not True
                or receipt['native'].get('component') != 'dit'
                or receipt['native'].get('forward_executed') is not False):
            raise RuntimeError('required official H3 layout is invalid, stale or belongs to another owner')
        return True

    def _preparations(self):
        definitions = [Preparation('backend', tuple(c.name for c in self.declaration),
            lambda _: prepare_backend(self.config, self.root), self._backend_valid)]
        definitions.append(Preparation('condition-text', ('dit',),
            lambda data: conditioning.text(self._context(data['request']), data['request']['request'], data['request']['canvases']),
            lambda value: self._condition_valid(value, 'condition-text'), inputs=('request',),
            weights=('conditioner',), after=('backend',)))
        conditions = ['condition-text']
        if 'video_encoder' in self.input_names:
            def video_input(data):
                payload = data['request']; context = self._context(payload)
                work = conditioning.prepare_video if self._needed(payload, 'condition-video') else (
                    lambda c, r, v: conditioning.skipped(c, r, v, 'condition-video-input'))
                return work(context, payload['request'], payload['canvases'])
            def video(data):
                payload = data['request']; context = self._context(payload)
                if not self._needed(payload, 'condition-video'):
                    return conditioning.skipped(context, payload['request'], payload['canvases'], 'condition-video')
                return conditioning.video(context, payload['request'], payload['canvases'], data['prepared'])
            definitions.append(Preparation('condition-video', ('dit',), video,
                lambda value: self._condition_valid(value, 'condition-video', True), inputs=('request',),
                prepare=video_input, validate_prepared=lambda value: self._condition_valid(value, 'condition-video-input', True),
                weights=('video_encoder',), after=('backend',),
                serial_group='reference-conditioning' if self.config['workflow'] == 'ref2va' else None))
            conditions.append('condition-video')
        if 'audio_encoder' in self.input_names:
            def audio(data):
                payload = data['request']; context = self._context(payload)
                work = conditioning.audio if self._needed(payload, 'condition-audio') else (
                    lambda c, r, v: conditioning.skipped(c, r, v, 'condition-audio'))
                return work(context, payload['request'], payload['canvases'])
            definitions.append(Preparation('condition-audio', ('dit',), audio,
                lambda value: self._condition_valid(value, 'condition-audio', True), inputs=('request',),
                weights=('audio_encoder',), after=('backend',), serial_group='reference-conditioning'))
            conditions.append('condition-audio')
        def assemble(data):
            payload = data['request']
            value = conditioning.assemble(self._context(payload), payload['request'], payload['canvases'], data['preparations'])
            return value
        definitions.append(Preparation('condition-assemble', ('dit',), assemble,
            lambda value: self._condition_valid(value, 'condition-assemble'), inputs=('request',), after=tuple(conditions)))
        def structure(data):
            payload = data['request']
            return preparation.prepare_structure(self._context(payload), payload['request'], payload['canvases'])
        def shape_layout(data):
            payload = data['request']; prepared = data['prepared']
            receipt = None if prepared['actual_required'] else preparation.prepare_layout(
                self._context(payload), payload['request'], prepared)
            return {'structure': prepared, 'receipt': receipt}
        def shape_valid(value):
            preparation.validate_structure(value['structure'], self.payload)
            return value['receipt'] is None if value['structure']['actual_required'] else self._layout_valid(value['receipt'])
        definitions.append(Preparation('request-shape-layout', ('dit',), shape_layout, shape_valid,
            inputs=('request',), prepare=structure, validate_prepared=lambda value: preparation.validate_structure(value, self.payload),
            weights=('dit',), after=('backend',)))
        def actual_layout(data):
            payload = data['request']; shape = data['preparations']['request-shape-layout']
            return shape['receipt'] if shape['receipt'] is not None else preparation.prepare_layout(
                self._context(payload), payload['request'], shape['structure'], actual=True)
        definitions.append(Preparation('request-layout', ('dit',), actual_layout, self._layout_valid,
            inputs=('request',), weights=('dit',), after=('request-shape-layout', 'condition-assemble')))
        return tuple(definitions)

    def prepare_layout(self, request, canvases=()):
        previous = self.startup.scope
        if previous is not None and any(not future.done() for future in previous.values()):
            raise RuntimeError('previous official H3 request preparation is still active')
        self.payload = {'request': dict(request), 'canvases': tuple(map(str, canvases)),
                        'output': str(self.output), 'scope': uuid.uuid4().hex}
        futures = self.startup.request({'request': self.payload})
        self.conditioning_future = futures['condition-assemble']
        self.layout_future = futures['request-layout']

    def conditioning_stages(self):
        self.conditioning_future.result()
        return json.loads((self.output / 'framework-conditioning.json').read_bytes())['stages']

    def invoke(self, name, args):
        return self.startup.invoke(name, args)

    def component(self, program, args):
        programs = self.config['programs']
        if str(program) == programs['conditioner']:
            return 'conditioner'
        if str(program) == programs['video_vae']:
            return 'video_encoder' if args[0] in ('encode', 'encode-video') else 'video_decoder'
        if str(program) == programs['audio_vae']:
            return 'audio_vae'
        if str(program) == programs.get('audio_encoder'):
            return 'audio_encoder'
        return None

    def report(self):
        state = self.startup.snapshot()
        return {'startup_engine': 'framework', 'startup_executor': 'rust',
            'parallel_preload': True, 'generation_warmup': False, 'requests_started': self.requests_started,
            'started_at': self.started, 'elapsed': time.monotonic() - self.started,
            'components': dict(self.startup.reports), 'backend_plans': self.backend_future.result(),
            'layout_preparation': self.layout_future.result() if self.layout_future else None,
            'prefetch': {name: worker.report() for name, worker in self.startup.prefetch.items()},
            'prefetch_warnings': dict(self.startup.prefetch_warnings),
            'reader_buffer_bound_mib': sum(c.readers.limit_mib for c in self.declaration),
            'reader_buffer_limit_mib': self.options.get('reader_budget_mib', 6144), 'declared_startup': state}

    def close(self):
        if self.closed:
            return
        self.closed = True
        cleanup = self.startup.close() if self.startup is not None else []
        if self.directory:
            self.directory.cleanup()
        if hasattr(self, 'logs'):
            (self.logs / 'cleanup.json').write_text(json.dumps(
                {'passed': all(r['reaped'] for r in cleanup), 'processes': cleanup}, indent=2) + '\n')
