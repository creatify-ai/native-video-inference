"""Official model bindings exercised by the real shared Rust startup executor.

Native tensor operations are transport doubles here; CUDA numerical acceptance
is separate. These tests must load the built cdylib and never substitute a
Python scheduler.
"""
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest

from core_infra.runtime import ProcessGroup
from models.minimax_h3_official.runtime import H3Runtime
from models.minimax_h3_official.runtime import preparation
from models.minimax_h3_official.runtime.startup import components


def write(path, data=b'x'):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


class NativeDouble:
    def __init__(self):
        self.calls, self.owners = [], {}
        self.environments = {}
        self.reject_layout = False

    def pipeline(self, group, program, args, environment, log):
        self.calls.append(('cpu', tuple(args)))
        command = args[0]
        if command in ('prepare', 'prepare-ref2va'):
            out = Path(args[5] if command == 'prepare' else args[4])
            write(out/'text_token_tags.raw', b'\0'*24)
            return dict(grid_height=2, grid_width=2, token_count=3, grids=[(2, 2)])
        if command == 'noise':
            write(Path(args[2])/'noise0.raw')
            return {}
        if command == 'layout-ref2va':
            out = Path(args[3])
            for name in ('video', 'audio', 'text'):
                write(out/(name+'_indices.raw'), b'\0'*8)
            for name in ('position_ids', 'token_tags'):
                write(out/(name+'.raw'), b'\0'*8)
            return dict(condition_video_rows=1, condition_audio_rows=0)
        if command in ('assemble', 'assemble-structure', 'assemble-ref2va'):
            out = Path(args[6] if command != 'assemble-ref2va' else args[5])
            names = ('hidden_states', 'audio_hidden_states', 'encoder_hidden_states',
                     'position_ids', 'token_tags', 'video_indices', 'audio_indices', 'text_indices')
            manifest = dict(inputs={n: dict(file=n+'.raw', shape=[1, 3, 5120], bytes=8) for n in names},
                            structure_only=command == 'assemble-structure')
            write(out/'manifest.json', json.dumps(manifest).encode())
            return manifest
        raise AssertionError(args)

    def owner(self, runtime, component, environment, group):
        name = component.name
        self.environments[name] = environment
        state = SimpleNamespace(dead=False)
        process = SimpleNamespace(pid=1000+len(self.owners), poll=lambda: 0 if state.dead else None)
        def invoke(args):
            args = list(map(str, args))
            self.calls.append((name, tuple(args)))
            if args[0].startswith('preload'):
                value = dict(component='video_vae' if name.startswith('video_') else name,
                             model_load_count=1, requests_completed=0)
                if name.startswith('video_'):
                    value['load_counts'] = dict(encoder=int(name == 'video_encoder'),
                        decoder=int(name == 'video_decoder'),
                        reference_encoder=int(name == 'video_encoder' and runtime.has_video_reference))
                return value
            if args[0] in ('infer', 'infer-ref2va'):
                out = Path(args[5] if args[0] == 'infer' else args[4])
                write(out/'conditioning.raw')
                write(out/'text_token_tags.raw', b'\0'*24)
                if args[0] == 'infer-ref2va':
                    refs=json.loads(Path(args[3]).read_text())['references']
                    kinds=(0,1,2) if any(r['type']=='video' for r in refs) else (0,1,1)
                    write(out/'prepared/mm_token_type_ids.raw', struct.pack('<3q', *kinds))
                return {}
            if args[0] in ('encode', 'encode-video'):
                out = Path(args[5] if args[0] == 'encode' else args[4])
                for filename in ('condition0.raw', 'condition1.raw', 'condition.raw'):
                    write(out/filename)
                return {}
            if name == 'audio_encoder':
                out = Path(args[2]); write(out/'condition.raw', b'\0'*128)
                write(out/'report.json', json.dumps(dict(passed=True, condition_shape=[1,32])).encode())
                return {}
            if args[0] == 'prepare-layout':
                manifest=json.loads((Path(args[6])/'manifest.json').read_text())
                assert manifest['attention_backend'] == runtime.payload['request']['attention_backend']
                return dict(component='dit', prepared=not self.reject_layout, forward_executed=False)
            return dict(generated=name)
        owner = SimpleNamespace(invoke=invoke, process=process,
            terminate=lambda: setattr(state, 'dead', True),
            reap=lambda: dict(pid=process.pid, reaped=state.dead))
        self.owners[name] = owner
        group.owners.append(owner)
        return owner


@pytest.fixture
def native(monkeypatch):
    double = NativeDouble()
    monkeypatch.setattr(ProcessGroup, 'run_elf', lambda group, *args: double.pipeline(group, *args))
    monkeypatch.setattr(H3Runtime, '_owner', lambda runtime, *args: double.owner(runtime, *args))
    return double


def config(workflow):
    return dict(workflow=workflow, runtime={'prefetch':False},
        programs={n:'/native/'+n for n in ('conditioner','dit','video_vae','audio_vae','audio_encoder','pipeline')},
        models={n:'/weights/'+n for n in ('conditioner','dit','video_vae','video_encoder','audio_vae',
                                        'audio_encoder','router','dense_library')})


def start_request(runtime, output, workflow, first=True, attention='subblock_sparse_attn'):
    output.mkdir(); write(output/'prompt.txt', b'prompt')
    request = dict(width=32, height=32, num_frames=22, seed=42, nfe=49,
                   attention_backend=attention)
    canvases = ()
    if workflow == 'fl2va' and first:
        request['first_png'] = 'input.png'; canvases=(str(output/'canvas/first.rgb'),)
    elif workflow == 'ref2va':
        refs = [dict(type='image' if first else 'video', file='reference.rgb', width=32, height=32, frames=22,
                     has_audio=not first, audio_file='audio.raw')]
        write(output/'references.json', json.dumps(dict(schema='h3-ref2va-normalized-rgb-v1',references=refs)).encode())
        request['reference_descriptor'] = str(output/'references.json')
    runtime.start(output, {}, request)
    runtime.prepare_layout(request, canvases)
    return request


@pytest.mark.parametrize('attention', ['subblock_sparse_attn', 'torch_cudnn_sdpa'])
@pytest.mark.parametrize('workflow,first', [('fl2va',True),('fl2va',False),('ref2va',True),('ref2va',False)])
def test_framework_joins_conditioning_and_layout_then_defers_outputs(native, tmp_path, workflow, first, attention):
    with H3Runtime(config(workflow)) as runtime:
        request = start_request(runtime, tmp_path/'first', workflow, first, attention)
        assert runtime.startup.executor == 'rust'
        assert runtime.invoke('dit', ['generate']) == {'generated':'dit'}
        report = runtime.report()
        assert report['layout_preparation']['native']['forward_executed'] is False
        assert report['layout_preparation']['owner_pid'] == native.owners['dit'].process.pid
        assert 'video_decoder' not in native.owners and 'audio_vae' not in native.owners
        names = {p.name for p in runtime._preparations()}
        assert {'backend','condition-text','condition-assemble','request-layout'} <= names
        receipt = report['layout_preparation']
        runtime.invoke('video_decoder', ['decode'])
        assert 'video_decoder' in native.owners
        old_owners = dict(native.owners)
        start_request(runtime, tmp_path/'second', workflow, first, attention)
        assert runtime.invoke('dit', ['generate']) == {'generated':'dit'}
        assert native.owners == old_owners
        with pytest.raises(RuntimeError, match='stale'):
            runtime._layout_valid(receipt)
        assert sum(n == 'dit' and a[0] == 'preload' for n,a in native.calls) == 1
        manifest = json.loads((tmp_path/'second/request/manifest.json').read_bytes())
        assert manifest['attention_backend'] == request['attention_backend']
        assert manifest['text_video_token_indices'] == ([2] if workflow == 'ref2va' and not first else [])
    assert all(owner.process.poll() == 0 for owner in native.owners.values())


def test_failed_layout_blocks_native_generate(native, tmp_path):
    native.reject_layout=True
    with H3Runtime(config('fl2va')) as runtime:
        start_request(runtime, tmp_path/'first', 'fl2va')
        with pytest.raises(RuntimeError, match='layout'):
            runtime.invoke('dit', ['generate'])
        assert ('dit', ('generate',)) not in native.calls


def test_structure_receipt_cannot_refer_to_another_request(tmp_path):
    payload=dict(scope='current',output=str(tmp_path))
    with pytest.raises(RuntimeError, match='stale'):
        preparation.validate_structure(dict(prepared=True, forward_executed=False, actual_required=True,
                                            request_scope='old', output=str(tmp_path)), payload)


def test_declaration_uses_one_reader_quota_and_rejects_overbudget(tmp_path):
    from core_infra.runtime.declared_startup import DeclaredStartup
    declared=components(config('ref2va'), ['conditioner','dit','video_encoder','audio_encoder'], True, tmp_path)
    assert all(c.environment['H3_MODEL_LOAD_WORKERS'] == str(c.readers.workers) for c in declared)
    with H3Runtime(config('ref2va')) as runtime:
        runtime.declaration=declared; runtime.input_names=[c.name for c in declared]
        with pytest.raises(ValueError, match='budget'):
            DeclaredStartup(tmp_path/'never', {}, declared, runtime._preparations(), reader_budget_mib=1)
    assert not (tmp_path/'never').exists()


def test_prepared_guards_reach_every_reader_and_stay_optional(tmp_path):
    names=['conditioner','dit','video_encoder']
    plain=config('fl2va')
    assert not any('H3_MODEL_LOAD_LIGHT_SHA' in c.environment for c in components(plain, names, False, tmp_path))
    guarded=config('fl2va'); guarded['runtime']={'light_sha':{'path':'/g.json','sha256':'0'*64}}
    declared=components(guarded, names, False, tmp_path)
    assert all(c.environment['H3_MODEL_LOAD_ASYNC_SHA']=='1' and c.environment['NATIVE_LOAD_VERIFY']=='light'
               and c.environment['H3_MODEL_LOAD_LIGHT_SHA_MANIFEST']=='/g.json' for c in declared)
    guarded['runtime']={'light_sha':{'path':'/g.json'}}
    with pytest.raises(ValueError, match='light_sha'):
        components(guarded, names, False, tmp_path)


@pytest.mark.parametrize('field,value', [('model_load_count',True),('requests_completed',False),
    ('requests_completed',1),('forward_executed',True)])
def test_preload_receipt_cannot_claim_a_forward_or_boolean_counter(field, value):
    from models.minimax_h3_official.runtime.startup import weight_only
    receipt=dict(component='dit', model_load_count=1, requests_completed=0)
    receipt[field]=value
    with pytest.raises(RuntimeError, match='weights'):
        weight_only(receipt, component='dit')


@pytest.mark.parametrize('workflow,workers,override',[('fl2va',8,False),('ref2va',24,False),
    ('ref2va',8,True),('ref2va',16,True),('ref2va',24,True)])
def test_real_rust_executor_applies_declared_cache_quota_and_keeps_native_readers(native, tmp_path, monkeypatch, workflow, workers, override):
    import threading
    from core_infra.runtime import components as shared
    from models.minimax_h3_official.runtime import startup as binding
    captured={};ready=threading.Event()
    class Helper:
        def __init__(self,group,jobs,root,environment,*,workers):
            captured[Path(root).name]=workers
            if len(captured)==2:ready.set()
        def report(self):return {'workers':workers}
    monkeypatch.setattr(shared,'FileReadAhead',Helper)
    monkeypatch.setattr(binding,'prefetch_jobs',lambda *_:[{'path':'/unused','bytes':4096}])
    cfg=config(workflow);cfg['runtime'].update(prefetch=True)
    if override:cfg['runtime']['prefetch_workers']={'conditioner':workers,'dit':workers}
    with H3Runtime(cfg)as runtime:
        start_request(runtime,tmp_path/'request',workflow)
        assert runtime.invoke('dit',['generate'])=={'generated':'dit'}
        assert ready.wait(3)
        assert captured=={'prefetch-conditioner':workers,'prefetch-dit':workers}
        assert all(native.environments[name]['H3_MODEL_LOAD_WORKERS']=='24'for name in ('conditioner','dit'))
        assert runtime.startup.executor=='rust'


@pytest.mark.parametrize('value',[True,{'dit':True},{'dit':0},{'dit':65},{'unknown':24}])
def test_invalid_prefetch_quota_rejected_before_model_start(tmp_path,value):
    cfg=config('ref2va');cfg['runtime']['prefetch_workers']=value
    with pytest.raises(ValueError,match='prefetch_workers'):
        components(cfg,['conditioner','dit','video_encoder'],False,tmp_path)


def test_context_parallel_runs_the_dit_through_its_coordinator(tmp_path):
    parallel = config('ref2va')
    parallel['programs']['cp_coordinator'] = '/native/cp_coordinator'
    parallel['runtime']['context_parallel'] = {'world': 2}
    declared = {c.name: c for c in components(parallel, ['conditioner', 'dit', 'video_encoder'], False, tmp_path)}
    assert declared['dit'].program == '/native/cp_coordinator'
    assert declared['dit'].preload == ('preload', '/weights/dit', '/weights/router', '/weights/dense_library')
    assert all(c.program == '/native/' + ('video_vae' if n.startswith('video_') else n)
               for n, c in declared.items() if n != 'dit')
