"""The shared startup scheduler must preserve readiness and cancellation."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core_infra.runtime import components as module


def _loaded_receipt(report):
    """Fake native owners in these scheduler tests promise a loaded name."""
    return isinstance(report, dict) and isinstance(report.get('loaded'), str)


def test_component_dependencies_and_native_policy_are_owned_by_framework(tmp_path,monkeypatch):
    encoder_ready=threading.Event()
    release_prepare=threading.Event()
    preparation_ready=threading.Event()
    spawned=[]
    class Group:
        def __init__(self,workers):self.executor=ThreadPoolExecutor(workers)
        def submit(self,fn,*args):return self.executor.submit(fn,*args)
        def owner(self,program,address,environment,log_path,**kwargs):
            name=Path(address).stem
            if name=='decoder':assert encoder_ready.is_set()
            spawned.append((name,environment))
            class Owner:
                def invoke(self,args):
                    if args==('infer',):assert preparation_ready.is_set()
                    if name=='encoder':encoder_ready.set()
                    return {'loaded':name,'args':args}
            return Owner()
        def close(self):self.executor.shutdown(wait=True);return []
    monkeypatch.setattr(module,'ProcessGroup',Group)
    startup=module.ComponentStartup(tmp_path,{'NATIVE_LOAD_VERIFY':'light'},[
        module.Component('encoder','encoder',('preload',), required_preparations=(), validate=_loaded_receipt),
        module.Component('decoder','decoder',('preload',),after=('encoder',), required_preparations=(), validate=_loaded_receipt)])
    try:
        def prepare():
            assert release_prepare.wait(5)
            preparation_ready.set()
        startup.prepare('backend',prepare,before=('encoder',))
        startup.start()
        assert startup.ready('decoder')['loaded']=='decoder'
        assert not preparation_ready.is_set()  # pre-init never blocks weight preload
        release_prepare.set()
        assert startup.invoke('encoder',('infer',))['args']==('infer',)
        assert len(spawned)==2
        assert all(env['NATIVE_STARTUP']=='1' and env['NATIVE_LOAD_WORKERS']=='24' for _,env in spawned)
    finally:release_prepare.set();startup.close()


def test_invalid_dependencies_fail_before_any_native_owner(tmp_path):
    with pytest.raises(ValueError,match='dependencies'):
        module.ComponentStartup(tmp_path,{},[module.Component('decoder','x',(),after=('missing',), required_preparations=(), validate=_loaded_receipt)])


def test_common_policy_keeps_declared_quota_and_guard_at_actual_owner(tmp_path, monkeypatch):
    launched = []
    class Group:
        def __init__(self, workers): self.executor = ThreadPoolExecutor(workers)
        def submit(self, fn, *args): return self.executor.submit(fn, *args)
        def owner(self, program, address, environment, log_path, **kwargs):
            launched.append(dict(environment))
            class Owner:
                def invoke(self, args): return {'loaded': program}
            return Owner()
        def close(self): self.executor.shutdown(wait=True); return []
    monkeypatch.setattr(module, 'ProcessGroup', Group)
    declaration = module.ReaderBudget(4, chunk_mib=16, pinned_slots=4, limit_mib=512)
    startup = module.ComponentStartup(tmp_path, {
        'NATIVE_LOAD_DIRECT': '0', 'NATIVE_LOAD_VERIFY': 'light',
        'NATIVE_LOAD_GUARD_MANIFEST': '/trusted/guard.json', 'NATIVE_LOAD_GUARD_SHA256': 'a' * 64}, [
        module.Component('model', 'model', ('preload',), readers=declaration, required_preparations=(), validate=_loaded_receipt)])
    try:
        startup.start(); startup.ready('model')
        env = launched[0]
        assert {key: env[key] for key in ('NATIVE_LOAD_WORKERS', 'NATIVE_LOAD_CHUNK_MIB',
            'NATIVE_LOAD_PINNED_SLOTS', 'NATIVE_LOAD_BUDGET_MIB')} == dict(
                NATIVE_LOAD_WORKERS='4', NATIVE_LOAD_CHUNK_MIB='16',
                NATIVE_LOAD_PINNED_SLOTS='4', NATIVE_LOAD_BUDGET_MIB='512')
        assert env['NATIVE_LOAD_DIRECT'] == '0' and env['NATIVE_LOAD_VERIFY'] == 'light'
        assert env['NATIVE_LOAD_GUARD_SHA256'] == 'a' * 64
    finally: startup.close()


@pytest.mark.parametrize('environment', [
    {'NATIVE_LOAD_DIRECT': 'invalid'}, {'NATIVE_LOAD_VERIFY': 'invalid'},
    {'NATIVE_LOAD_VERIFY': 'light', 'NATIVE_LOAD_GUARD_MANIFEST': '/trusted/guard.json'},
    {'NATIVE_LOAD_VERIFY': 'light', 'NATIVE_LOAD_GUARD_MANIFEST': '/trusted/guard.json', 'NATIVE_LOAD_GUARD_SHA256': 'bad'},
])
def test_malformed_common_policy_never_spawns_native_owner(tmp_path, monkeypatch, environment):
    def owner(*args, **kwargs): raise AssertionError('invalid declaration reached native spawn')
    monkeypatch.setattr(module.ProcessGroup, 'owner', owner)
    startup = module.ComponentStartup(tmp_path, environment, [module.Component('model', 'unused', (), required_preparations=(), validate=_loaded_receipt)])
    try:
        startup.start()
        with pytest.raises(ValueError): startup.ready('model')
        assert not startup.reports
    finally: startup.close()


def test_full_policy_keeps_unused_guard_fields_inert():
    declaration = module.ReaderBudget(4, chunk_mib=16, pinned_slots=4, limit_mib=512)
    result = declaration.startup_environment(direct=False, verify='full',
        guard_manifest='unused-invalid-guard', guard_sha256=None)
    assert result['NATIVE_LOAD_VERIFY'] == 'full' and result['NATIVE_LOAD_DIRECT'] == '0'
    assert 'NATIVE_LOAD_GUARD_MANIFEST' not in result


def test_waiting_components_do_not_starve_request_preparation(tmp_path,monkeypatch):
    source_started=threading.Event()
    release_source=threading.Event()
    class Group:
        def __init__(self,workers):self.executor=ThreadPoolExecutor(workers)
        def submit(self,fn,*args):return self.executor.submit(fn,*args)
        def owner(self,program,address,environment,log_path,**kwargs):
            class Owner:
                def invoke(self,args):
                    if program=='source':
                        source_started.set()
                        assert release_source.wait(5)
                    return {'loaded': program}
            return Owner()
        def close(self):self.executor.shutdown(wait=True);return []
    monkeypatch.setattr(module,'ProcessGroup',Group)
    startup=module.ComponentStartup(tmp_path,{},[
        module.Component('source','source',(), required_preparations=(), validate=_loaded_receipt),
        module.Component('decoder','decoder',(),after=('source',), required_preparations=(), validate=_loaded_receipt)],workers=2)
    try:
        startup.start()
        assert source_started.wait(2)
        preparation=startup.prepare('metadata',lambda:'prepared',before=('source',))
        assert preparation.result(timeout=1)=='prepared'
        assert not startup.futures['source'].done()
        release_source.set()
        startup.ready('decoder')
    finally:release_source.set();startup.close()


def test_preload_validation_failure_cannot_become_ready(tmp_path,monkeypatch):
    class Group:
        def __init__(self,workers):self.executor=ThreadPoolExecutor(workers)
        def submit(self,fn,*args):return self.executor.submit(fn,*args)
        def owner(self,*args,**kwargs):
            class Owner:
                def invoke(self,args):return {'requests_completed':1}
            return Owner()
        def close(self):self.executor.shutdown(wait=True);return []
    def validate(report):
        if report['requests_completed']:raise RuntimeError('preload ran inference')
    monkeypatch.setattr(module,'ProcessGroup',Group)
    startup=module.ComponentStartup(tmp_path,{},[module.Component('model','x',(),validate=validate, required_preparations=())])
    try:
        startup.start()
        with pytest.raises(RuntimeError,match='ran inference'):startup.ready('model')
        assert not startup.reports
    finally:startup.close()


def test_official_h3_entry_requires_the_shared_framework_executor(tmp_path, monkeypatch):
    from models.minimax_h3_official import native_pipeline
    from models.minimax_h3_official.runtime import framework
    monkeypatch.setattr(native_pipeline, 'validate_config', lambda config: {'recipe': {'nfe': 49}})
    monkeypatch.setattr(native_pipeline.MiniMaxH3Pipeline, '_run',
        lambda self, request, output, runtime: type(runtime))
    config = {'workflow': 'fl2va',
        'programs': {k: '/program/' + k for k in ('dit', 'conditioner', 'video_vae', 'audio_vae')},
        'models': {k: '/weights/' + k for k in ('dit', 'conditioner', 'video_vae', 'audio_vae')}}
    for workflow in ('fl2va', 'ref2va'):
        config['workflow'] = workflow
        model = native_pipeline.MiniMaxH3Pipeline(config)
        assert issubclass(model.run({}, tmp_path), framework.FrameworkH3Runtime)
    for engine in ('legacy-customized', 'misspelled-framework'):
        config['runtime'] = {'startup_engine': engine}
        with pytest.raises(ValueError, match='shared framework'):
            model.run({}, tmp_path)
    for option in ('layout_preparation', 'conditioning_preparations'):
        config['runtime'] = {option: False}
        with pytest.raises(ValueError, match='layout and conditioning'):
            model.run({}, tmp_path)


@pytest.fixture
def component_group(monkeypatch):
    """Real executor/futures, with only native process creation replaced."""
    created = []
    class Group:
        def __init__(self, workers):
            self.executor = ThreadPoolExecutor(workers)
            self.spawned, self.invocations, self.submitted = [], [], []
            self.hook = None
            created.append(self)
        def submit(self, function, *args):
            self.submitted.append(function)
            return self.executor.submit(function, *args)
        def owner(self, program, address, environment, log_path, **kwargs):
            self.spawned.append((Path(address).stem, dict(environment)))
            group = self
            class Owner:
                def invoke(self, args):
                    group.invocations.append((program, args))
                    return group.hook(program, args) if group.hook else {'loaded': program}
            return Owner()
        def close(self):
            self.executor.shutdown(wait=True, cancel_futures=True)
            return []
    monkeypatch.setattr(module, 'ProcessGroup', Group)
    yield created
    for group in created:
        group.close()


def test_implicit_native_reader_quota_is_admitted_before_group_creation(tmp_path, component_group):
    components = [module.Component(str(i), 'model', ('preload',), required_preparations=(), validate=_loaded_receipt) for i in range(4)]
    with pytest.raises(ValueError, match='container budget'):
        module.ComponentStartup(tmp_path, {}, components)
    assert component_group == []
    # Disabled common policy does not silently activate/charge its default.
    legacy = module.ComponentStartup(tmp_path, {}, components, native_policy=False)
    try:
        legacy.start()
        for component in components: legacy.ready(component.name)
        assert len(legacy.group.spawned) == 4
        assert all('NATIVE_STARTUP' not in env for _, env in legacy.group.spawned)
    finally: legacy.close()


def test_admitted_default_and_explicit_quotas_match_actual_owner_environment(tmp_path, component_group):
    explicit = module.ReaderBudget(4, chunk_mib=16, pinned_slots=4, limit_mib=512)
    components = [module.Component('default', 'model', ('preload',), required_preparations=(), validate=_loaded_receipt),
                  module.Component('explicit', 'model', ('preload',), readers=explicit, required_preparations=(), validate=_loaded_receipt)]
    with pytest.raises(ValueError, match='container budget'):
        module.ComponentStartup(tmp_path, {}, components, reader_budget_mib=2559)
    assert component_group == []
    startup = module.ComponentStartup(tmp_path, {}, components, reader_budget_mib=2560)
    try:
        startup.start()
        for component in components: startup.ready(component.name)
        environments = dict(startup.group.spawned)
        assert environments['default']['NATIVE_LOAD_WORKERS'] == '24'
        assert environments['default']['NATIVE_LOAD_BUDGET_MIB'] == '2048'
        assert environments['explicit']['NATIVE_LOAD_WORKERS'] == '4'
        assert environments['explicit']['NATIVE_LOAD_BUDGET_MIB'] == '512'
    finally: startup.close()


@pytest.mark.parametrize('declared,expected', [(None,2),(8,8),(16,16),(24,24)])
def test_optional_cache_helpers_share_workers_without_reducing_native_readers(tmp_path, component_group, monkeypatch, declared, expected):
    helpers = []
    class Helper:
        def __init__(self, group, jobs, root, environment, *, workers):
            helpers.append((Path(root).name, workers))
    monkeypatch.setattr(module, 'FileReadAhead', Helper)
    components = [module.Component(name, 'model', ('preload',), required_preparations=(),
        validate=_loaded_receipt, prefetch=lambda: [{'path': '/unused', 'bytes': 1}],prefetch_workers=declared)
        for name in ('one', 'two')]
    startup = module.ComponentStartup(tmp_path, {}, components, workers=8)
    try:
        startup.start()
        for component in components:
            startup.ready(component.name)
        for future in startup.prefetch_futures:
            future.result(timeout=2)
        assert sorted(helpers) == [('prefetch-one', expected), ('prefetch-two', expected)]
        assert all(env['NATIVE_LOAD_WORKERS'] == '24' for _, env in startup.group.spawned)
    finally:
        startup.close()


@pytest.mark.parametrize('workers',[True,0,-1,65,2.5,'24'])
def test_prefetch_quota_rejected_before_owners_or_helpers_start(workers):
    with pytest.raises(ValueError,match='prefetch_workers'):
        module.Component('model','model',('preload',),prefetch=lambda:[],prefetch_workers=workers)


def test_prefetch_quota_requires_a_cache_producer():
    with pytest.raises(ValueError,match='prefetch_workers'):
        module.Component('model','model',('preload',),prefetch_workers=24)


def test_concurrent_start_only_submits_each_owner_once(tmp_path, component_group):
    barrier = threading.Barrier(2)
    startup = module.ComponentStartup(tmp_path, {}, [module.Component('model', 'model', ('preload',), required_preparations=(), validate=_loaded_receipt)])
    def start():
        barrier.wait(timeout=2)
        try: return startup.start()
        except RuntimeError as error: return error
    try:
        with ThreadPoolExecutor(2) as callers:
            results = list(callers.map(lambda _: start(), range(2)))
        assert results.count(startup) == 1
        assert sum(isinstance(result, RuntimeError) for result in results) == 1
        startup.ready('model')
        assert len(startup.group.spawned) == 1
    finally: startup.close()


def test_closed_or_unstarted_startup_cannot_admit_execution(tmp_path, component_group):
    startup = module.ComponentStartup(tmp_path, {}, [module.Component('model', 'model', ('preload',), required_preparations=(), validate=_loaded_receipt)])
    for operation in (lambda: startup.ready('model'), lambda: startup.invoke('model', ('infer',))):
        with pytest.raises(RuntimeError): operation()
    # Backend metadata is deliberately permitted before reader submission.
    assert startup.prepare('backend', lambda: 'metadata', before=('model',)).result(timeout=1) == 'metadata'
    startup.close()
    submitted = len(startup.group.submitted)
    for operation in (startup.start, lambda: startup.prepare('late', lambda: None),
                      lambda: startup.ready('model'), lambda: startup.invoke('model', ('infer',))):
        with pytest.raises(RuntimeError): operation()
    assert len(startup.group.submitted) == submitted
    assert startup.group.spawned == []


def test_preparation_registration_published_before_submission_and_joined_by_invoke(tmp_path, monkeypatch, component_group):
    startup = module.ComponentStartup(tmp_path, {}, [module.Component('model', 'model', ('preload',), required_preparations=(), validate=_loaded_receipt)])
    submission_entered, allow_submission, allow_prepare, weights_joined = [threading.Event() for _ in range(4)]
    startup.start(); startup.ready('model')
    original_submit, original_ready = startup.group.submit, startup.ready
    def submit(function, *args):
        submission_entered.set()
        assert allow_submission.wait(3)
        return original_submit(function, *args)
    def ready(name):
        result = original_ready(name)
        weights_joined.set()
        return result
    monkeypatch.setattr(startup.group, 'submit', submit)
    monkeypatch.setattr(startup, 'ready', ready)
    try:
        with ThreadPoolExecutor(2) as callers:
            registration = callers.submit(startup.prepare, 'layout', lambda: allow_prepare.wait(3), before=('model',))
            assert submission_entered.wait(1)
            inference = callers.submit(startup.invoke, 'model', ('infer',))
            assert weights_joined.wait(1)
            assert not inference.done()
            assert not any(args == ('infer',) for _, args in startup.group.invocations)
            allow_submission.set()
            registration.result(timeout=1)
            assert not inference.done()
            allow_prepare.set()
            assert inference.result(timeout=1) == {'loaded': 'model'}
    finally:
        allow_submission.set(); allow_prepare.set(); startup.close()


def test_invoke_closes_only_its_consumers_registration_before_rpc(tmp_path, component_group):
    startup = module.ComponentStartup(tmp_path, {}, [module.Component(name, name, ('preload',), required_preparations=(), validate=_loaded_receipt) for name in ('first', 'second')])
    entered, release = threading.Event(), threading.Event()
    def hook(program, args):
        if args == ('preload',): return {'loaded': program}
        if args == ('infer',) and program == 'first':
            entered.set(); assert release.wait(3)
        return {}
    startup.group.hook = hook
    try:
        startup.start()
        with ThreadPoolExecutor(1) as callers:
            inference = callers.submit(startup.invoke, 'first', ('infer',))
            assert entered.wait(1)
            submissions = len(startup.group.submitted)
            with pytest.raises(ValueError, match='late'):
                startup.prepare('too-late', lambda: None, before=('first',))
            assert len(startup.group.submitted) == submissions
            assert startup.prepare('other-layout', lambda: 'ready', before=('second',)).result(timeout=1) == 'ready'
            assert startup.invoke('second', ('infer',)) == {}
            release.set(); inference.result(timeout=1)
    finally: release.set(); startup.close()


def test_preparation_submit_failure_remains_an_execution_gate(tmp_path, monkeypatch, component_group):
    startup = module.ComponentStartup(tmp_path, {}, [module.Component('model', 'model', ('preload',), required_preparations=(), validate=_loaded_receipt)])
    error = RuntimeError('preparation submission failed')
    startup.start(); startup.ready('model')
    def fail(*args): raise error
    monkeypatch.setattr(startup.group, 'submit', fail)
    try:
        with pytest.raises(RuntimeError) as caught:
            startup.prepare('layout', lambda: None, before=('model',))
        assert caught.value is error
        with pytest.raises(RuntimeError) as caught:
            startup.invoke('model', ('infer',))
        assert caught.value is error
        assert not any(args == ('infer',) for _, args in startup.group.invocations)
    finally: startup.close()


def test_cancelled_queued_preparation_cannot_allow_execution(tmp_path, component_group):
    from concurrent.futures import CancelledError
    startup = module.ComponentStartup(tmp_path, {}, [module.Component('model', 'model', ('preload',), required_preparations=(), validate=_loaded_receipt)], workers=1)
    release = threading.Event()
    try:
        startup.start(); startup.ready('model')
        startup.group.submit(lambda: release.wait(3))
        preparation = startup.prepare('layout', lambda: 'unexpected', before=('model',))
        assert preparation.cancel()
        with pytest.raises(CancelledError): startup.invoke('model', ('infer',))
        assert not any(args == ('infer',) for _, args in startup.group.invocations)
    finally: release.set(); startup.close()


def test_partial_start_submission_failure_cannot_admit_a_preloaded_component(tmp_path, monkeypatch, component_group):
    startup = module.ComponentStartup(tmp_path, {}, [
        module.Component('first', 'first', ('preload',), required_preparations=(), validate=_loaded_receipt),
        module.Component('second', 'second', ('preload',), required_preparations=(), validate=_loaded_receipt)])
    submit = startup.group.submit
    error = RuntimeError('executor rejected remaining startup work')
    calls = 0
    def fail_second(function, *args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise error
        return submit(function, *args)
    monkeypatch.setattr(startup.group, 'submit', fail_second)
    try:
        with pytest.raises(RuntimeError) as caught:
            startup.start()
        assert caught.value is error
        # A successfully loaded first component cannot conceal an incomplete
        # declared startup. Recovery requires a new owned session.
        assert startup.futures['first'].result(timeout=1) == {'loaded': 'first'}
        for operation in (lambda: startup.ready('first'),
                          lambda: startup.invoke('first', ('infer',)),
                          lambda: startup.prepare('late', lambda: None, before=('first',)),
                          startup.start):
            with pytest.raises(RuntimeError):
                operation()
        assert not any(args == ('infer',) for _, args in startup.group.invocations)
    finally:
        startup.close()


def test_immediate_dependency_callbacks_do_not_hold_lifecycle_lock(tmp_path, monkeypatch):
    from concurrent.futures import Future
    spawned = []
    class ImmediateGroup:
        def __init__(self, workers): pass
        def submit(self, function, *args):
            future = Future()
            try: future.set_result(function(*args))
            except BaseException as error: future.set_exception(error)
            return future
        def owner(self, program, *args, **kwargs):
            spawned.append(program)
            class Owner:
                def invoke(self, args): return {'loaded': program}
            return Owner()
        def close(self): return []
    monkeypatch.setattr(module, 'ProcessGroup', ImmediateGroup)
    startup = module.ComponentStartup(tmp_path, {}, [module.Component('source', 'source', ('preload',), required_preparations=(), validate=_loaded_receipt),
        module.Component('dependent', 'dependent', ('preload',), after=('source',), required_preparations=(), validate=_loaded_receipt)])
    errors = []
    def exercise():
        try:
            startup.prepare('metadata', lambda: None, before=('dependent',))
            startup.start(); startup.invoke('dependent', ('infer',))
        except BaseException as error: errors.append(error)
    thread = threading.Thread(target=exercise, daemon=True); thread.start(); thread.join(timeout=1)
    assert not thread.is_alive(), 'synchronous dependency callback deadlocked the metadata lock'
    assert not errors
    assert spawned == ['source', 'dependent']
    startup.close()


def test_omitted_preparation_declaration_is_rejected_before_process_group(tmp_path, component_group):
    undeclared = module.Component('model', 'model', ('preload',))
    with pytest.raises(ValueError, match='must declare'):
        module.ComponentStartup(tmp_path, {}, [undeclared])
    assert component_group == []


@pytest.mark.parametrize('required', ['layout', ('',), (' layout',), ('layout', 'layout'), ('layout', 3), {}])
def test_malformed_required_preparations_fail_before_process_group(tmp_path, component_group, required):
    declaration = module.Component('model', 'model', (), required_preparations=required, validate=_loaded_receipt)
    with pytest.raises(ValueError, match='invalid required'):
        module.ComponentStartup(tmp_path, {}, [declaration])
    assert component_group == []


@pytest.mark.parametrize('contract', ['legacy', 'guess', None])
def test_preparation_contract_cannot_be_disabled_before_process_group(tmp_path, component_group, contract):
    with pytest.raises(ValueError, match='contract must be declared'):
        module.ComponentStartup(tmp_path, {}, [module.Component('model', 'model', (), required_preparations=(), validate=_loaded_receipt)],
                                preparation_contract=contract)
    assert component_group == []


def test_missing_required_hook_rejects_execution_without_waiting_for_weights(tmp_path, component_group):
    release = threading.Event()
    startup = module.ComponentStartup(tmp_path, {}, [
        module.Component('model', 'model', ('preload',), required_preparations=('layout',), validate=_loaded_receipt)])
    def hook(program, args):
        if args == ('preload',): assert release.wait(3)
        return {'loaded': program}
    startup.group.hook = hook
    try:
        startup.start()
        with pytest.raises(RuntimeError, match='preparation is missing: layout'):
            startup.invoke('model', ('infer',))
        assert not any(args == ('infer',) for _, args in startup.group.invocations)
    finally:
        release.set(); startup.close()


def test_required_registration_covers_every_consumer_and_needs_a_validator(tmp_path, component_group):
    startup = module.ComponentStartup(tmp_path, {}, [
        module.Component(name, name, ('preload',), required_preparations=('backend',), validate=_loaded_receipt)
        for name in ('first', 'second')])
    try:
        for before, validate in [(('first',), lambda _: True), (('first', 'second'), None)]:
            with pytest.raises(ValueError, match='validate and gate all'):
                startup.prepare('backend', lambda: {}, before=before, validate=validate)
        assert not startup.preparations and not startup.group.submitted
        startup.prepare('backend', lambda: {'prepared': True}, before=('first', 'second'),
                        validate=lambda receipt: receipt.get('prepared') is True).result(timeout=1)
        startup.start()
        assert startup.invoke('first', ('infer',)) == {'loaded': 'first'}
        assert startup.invoke('second', ('infer',)) == {'loaded': 'second'}
    finally:
        startup.close()


def test_optional_preparation_does_not_satisfy_an_unregistered_required_id(tmp_path, component_group):
    startup = module.ComponentStartup(tmp_path, {}, [
        module.Component('model', 'model', (), required_preparations=('layout',), validate=_loaded_receipt)])
    try:
        startup.prepare('diagnostic', lambda: None, before=('model',)).result(timeout=1)
        startup.start(); startup.ready('model')
        with pytest.raises(RuntimeError, match='preparation is missing: layout'):
            startup.invoke('model', ('infer',))
        assert not any(args == ('infer',) for _, args in startup.group.invocations)
    finally:
        startup.close()


def test_required_preparation_cannot_omit_its_validator(tmp_path, component_group):
    startup = module.ComponentStartup(tmp_path, {}, [
        module.Component('model', 'model', (), required_preparations=('layout',), validate=_loaded_receipt)])
    try:
        startup.start()
        with pytest.raises(ValueError, match='validate and gate all'):
            startup.prepare('layout', lambda: {}, before=('model',))
        with pytest.raises(RuntimeError, match='preparation is missing: layout'):
            startup.invoke('model', ('infer',))
    finally:
        startup.close()


@pytest.mark.parametrize('receipt', [
    {'prepared': False, 'owner': 42, 'shape': [1, 8]},
    {'prepared': True, 'owner': 99, 'shape': [1, 8]},
    {'prepared': True, 'owner': 42, 'shape': [1, 9]},
    {'prepared': True, 'owner': 42, 'shape': [1, 8], 'failed': True},
])
def test_model_receipt_validator_rejects_noop_wrong_owner_shape_or_failure(tmp_path, component_group, receipt):
    startup = module.ComponentStartup(tmp_path, {}, [
        module.Component('model', 'model', (), required_preparations=('layout',), validate=_loaded_receipt)])
    # These are model semantics. Core only requires an affirmative validator.
    def validate(value):
        return (value.get('prepared') is True and value.get('owner') == 42
                and value.get('shape') == [1, 8] and not value.get('failed'))
    try:
        preparation = startup.prepare('layout', lambda: receipt, before=('model',), validate=validate)
        with pytest.raises(RuntimeError, match='receipt rejected'):
            preparation.result(timeout=1)
        startup.start(); startup.ready('model')
        with pytest.raises(RuntimeError, match='receipt rejected'):
            startup.invoke('model', ('infer',))
        assert not any(e['component'] == 'layout' and e['phase'] == 'prepare_ready' for e in startup.events)
        assert not any(args == ('infer',) for _, args in startup.group.invocations)
    finally:
        startup.close()


@pytest.mark.parametrize('verdict', [False, None, 1, 'ok'])
def test_required_validator_must_return_true_exactly(tmp_path, component_group, verdict):
    startup = module.ComponentStartup(tmp_path, {}, [
        module.Component('model', 'model', (), required_preparations=('layout',), validate=_loaded_receipt)])
    try:
        future = startup.prepare('layout', lambda: {}, before=('model',), validate=lambda _: verdict)
        with pytest.raises(RuntimeError, match='receipt rejected'): future.result(timeout=1)
        startup.start()
        with pytest.raises(RuntimeError, match='receipt rejected'): startup.invoke('model', ('infer',))
        assert not any(e['phase'] == 'prepare_ready' for e in startup.events)
    finally:
        startup.close()


def test_validation_exception_preserves_primary_error_and_never_publishes_ready(tmp_path, component_group):
    startup = module.ComponentStartup(tmp_path, {}, [
        module.Component('model', 'model', (), required_preparations=('layout',), validate=_loaded_receipt)])
    error = ValueError('model receipt owner was retired')
    def validate(receipt):
        assert not startup.preparations['layout'][1].done()
        assert not any(e['phase'] == 'prepare_ready' for e in startup.events)
        raise error
    try:
        future = startup.prepare('layout', lambda: {}, before=('model',), validate=validate)
        with pytest.raises(ValueError) as caught: future.result(timeout=1)
        assert caught.value is error
        startup.start()
        with pytest.raises(ValueError) as caught: startup.invoke('model', ('infer',))
        assert caught.value is error
        assert not any(e['phase'] == 'prepare_ready' for e in startup.events)
    finally:
        startup.close()


def test_required_validation_keeps_dependent_preloads_and_other_consumers_nonblocking(tmp_path, component_group):
    release, validated = threading.Event(), threading.Event()
    startup = module.ComponentStartup(tmp_path, {}, [
        module.Component('input', 'input', ('preload',), required_preparations=(), validate=_loaded_receipt),
        module.Component('dit', 'dit', ('preload',), after=('input',), required_preparations=('layout',), validate=_loaded_receipt)], workers=2)
    def validate(receipt):
        assert release.wait(3)
        validated.set()
        return True
    try:
        startup.prepare('layout', lambda: {}, before=('dit',), validate=validate)
        startup.start()
        assert startup.ready('dit') == {'loaded': 'dit'}
        assert not validated.is_set()
        assert startup.invoke('input', ('encode',)) == {'loaded': 'input'}
        with ThreadPoolExecutor(1) as callers:
            inference = callers.submit(startup.invoke, 'dit', ('infer',))
            assert not inference.done()
            release.set()
            assert inference.result(timeout=1) == {'loaded': 'dit'}
        assert validated.is_set()
        assert any(e['component'] == 'layout' and e['phase'] == 'prepare_ready' for e in startup.events)
    finally:
        release.set(); startup.close()


def test_cancelled_required_preparation_cannot_admit_execution(tmp_path, component_group):
    from concurrent.futures import CancelledError
    release = threading.Event()
    startup = module.ComponentStartup(tmp_path, {}, [
        module.Component('model', 'model', (), required_preparations=('layout',), validate=_loaded_receipt)], workers=1)
    try:
        startup.start(); startup.ready('model')
        startup.group.submit(lambda: release.wait(3))
        future = startup.prepare('layout', lambda: {}, before=('model',), validate=lambda _: True)
        assert future.cancel()
        with pytest.raises(CancelledError): startup.invoke('model', ('infer',))
        assert not any(args == ('infer',) for _, args in startup.group.invocations)
        assert not any(e['phase'] == 'prepare_ready' for e in startup.events)
    finally:
        release.set(); startup.close()


def test_required_registration_race_waits_for_submission_and_receipt_validation(tmp_path, monkeypatch, component_group):
    submitted, release_submit, validate_started, release_validate, joined_weights = [threading.Event() for _ in range(5)]
    startup = module.ComponentStartup(tmp_path, {}, [
        module.Component('model', 'model', (), required_preparations=('layout',), validate=_loaded_receipt)])
    startup.start(); startup.ready('model')
    submit, ready = startup.group.submit, startup.ready
    def held_submit(function, *args):
        submitted.set(); assert release_submit.wait(3)
        return submit(function, *args)
    def weight_ready(name):
        result = ready(name); joined_weights.set(); return result
    def validate(receipt):
        validate_started.set(); assert release_validate.wait(3); return True
    monkeypatch.setattr(startup.group, 'submit', held_submit)
    monkeypatch.setattr(startup, 'ready', weight_ready)
    try:
        with ThreadPoolExecutor(2) as callers:
            registration = callers.submit(startup.prepare, 'layout', lambda: {}, before=('model',), validate=validate)
            assert submitted.wait(1)
            inference = callers.submit(startup.invoke, 'model', ('infer',))
            assert joined_weights.wait(1) and not inference.done()
            release_submit.set(); registration.result(timeout=1)
            assert validate_started.wait(1) and not inference.done()
            assert not any(args == ('infer',) for _, args in startup.group.invocations)
            release_validate.set(); assert inference.result(timeout=1) == {'loaded': 'model'}
    finally:
        release_submit.set(); release_validate.set(); startup.close()


def test_unknown_consumer_and_invalid_validator_fail_before_preparation_submission(tmp_path, component_group):
    startup = module.ComponentStartup(tmp_path, {}, [
        module.Component('model', 'model', (), required_preparations=(), validate=_loaded_receipt)])
    try:
        for name, before, validate in [('layout', ('unknown',), lambda _: True),
                                       ('', ('model',), lambda _: True),
                                       ('layout', ('model',), True)]:
            with pytest.raises(ValueError):
                startup.prepare(name, lambda: {}, before=before, validate=validate)
        assert not startup.preparations and not startup.group.submitted
        startup.start()
        with pytest.raises(ValueError, match='unknown'): startup.invoke('unknown', ('infer',))
    finally:
        startup.close()


@pytest.mark.parametrize('validate', [None, False, 'validate'])
def test_declared_preload_needs_callable_validator_before_group_creation(tmp_path, component_group, validate):
    declaration = module.Component('model', 'model', (), required_preparations=(), validate=validate)
    with pytest.raises(ValueError, match='must declare a preload receipt validator'):
        module.ComponentStartup(tmp_path, {}, [declaration])
    assert component_group == []


@pytest.mark.parametrize('verdict', [False, None, 1])
def test_rejected_preload_never_publishes_ready_or_starts_dependent(tmp_path, component_group, verdict):
    startup = module.ComponentStartup(tmp_path, {}, [
        module.Component('source', 'source', ('preload',), required_preparations=(), validate=lambda _: verdict),
        module.Component('dependent', 'dependent', ('preload',), after=('source',),
                         required_preparations=(), validate=_loaded_receipt)])
    try:
        startup.start()
        for name in ('source', 'dependent'):
            with pytest.raises(RuntimeError, match='preload receipt rejected: source'):
                startup.ready(name)
            with pytest.raises(RuntimeError, match='preload receipt rejected: source'):
                startup.invoke(name, ('infer',))
        assert [name for name, _ in startup.group.spawned] == ['source']
        assert not startup.reports
        assert not any(e['phase'] == 'ready' for e in startup.events)
        assert not any(args == ('infer',) for _, args in startup.group.invocations)
    finally:
        startup.close()


def test_preload_validator_exception_preserves_primary_error(tmp_path, component_group):
    error = ValueError('native preload selected a different owner')
    def validate(report): raise error
    startup = module.ComponentStartup(tmp_path, {}, [
        module.Component('model', 'model', ('preload',), required_preparations=(), validate=validate)])
    try:
        startup.start()
        for operation in (lambda: startup.ready('model'), lambda: startup.invoke('model', ('infer',))):
            with pytest.raises(ValueError) as caught: operation()
            assert caught.value is error
        assert not startup.reports and not any(e['phase'] == 'ready' for e in startup.events)
    finally:
        startup.close()


@pytest.mark.parametrize('validate', [None, lambda _: None, lambda _: False])
def test_legacy_cannot_waive_missing_or_false_preload_validation(tmp_path, component_group, validate):
    with pytest.raises(ValueError, match='contract must be declared'):
        module.ComponentStartup(tmp_path, {}, [module.Component('model', 'model', ('preload',),
            required_preparations=(), validate=validate)], preparation_contract='legacy')
    assert component_group == []
