"""The complete contract is independent of owned/resident command transport."""
from concurrent.futures import CancelledError
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import pytest

from core_infra.runtime.components import Component
from core_infra.runtime.declared_startup import DeclaredStartup, Preparation


@pytest.fixture(params=('python-reference', 'rust'), autouse=True)
def complete_executor(request, monkeypatch):
    """The native executor must satisfy the same observable safety contract."""
    if request.param == 'python-reference':
        from reference_declared_startup import DeclaredStartup as ReferenceDeclaredStartup
        monkeypatch.setitem(globals(), 'DeclaredStartup', ReferenceDeclaredStartup)


def component():
    return Component('dit', '/native/dit', ('preload',), required_preparations=('backend','layout'),
                     validate=lambda receipt: receipt == {'loaded': True})


def plan(tmp_path, *, layout=None, validate=None, weight=None, calls=None):
    calls = [] if calls is None else calls
    def owner(c, env, group):
        def invoke(args):
            calls.append(tuple(args))
            if args == ('preload',):
                if weight is not None: weight.wait(2)
                return {'loaded': True}
            return 'computed'
        return SimpleNamespace(invoke=invoke)
    preparations = (
        Preparation('backend', ('dit',), lambda _: True, lambda value: value is True),
        Preparation('layout', ('dit',), layout or (lambda inputs: inputs['request']),
                    validate or (lambda value: value is True), inputs=('request',)),
    )
    return DeclaredStartup(tmp_path, {}, (component(),), preparations, owner_factory=owner)


@pytest.mark.parametrize('mutation', [
    lambda ps: ps[:1],
    lambda ps: (*ps, ps[0]),
    lambda ps: (ps[0], replace(ps[1], before=('other',))),
    lambda ps: (ps[0], replace(ps[1], validate=None)),
])
def test_incomplete_or_invalid_plan_rejected_before_owners_or_io(tmp_path, mutation):
    preparations = (Preparation('backend', ('dit',), lambda _: True, lambda _: True),
                    Preparation('layout', ('dit',), lambda _: True, lambda _: True, ('request',)))
    with pytest.raises(ValueError):
        DeclaredStartup(tmp_path/'never-created', {}, (component(),), mutation(preparations))
    assert not (tmp_path/'never-created').exists()


def test_input_is_required_even_after_weights_are_ready(tmp_path):
    calls = []
    startup = plan(tmp_path, calls=calls).start()
    try:
        startup.ready('dit')
        with pytest.raises(RuntimeError, match='input is missing'):
            startup.invoke('dit', ('generate',))
        assert calls == [('preload',)]
        startup.request({'request': True})
        assert startup.invoke('dit', ('generate',)) == 'computed'
    finally:
        startup.close()


def test_preparation_and_loading_overlap_without_an_entry_scheduling_them(tmp_path):
    reading, preparing, finish = Event(), Event(), Event()
    def cpu(inputs):
        preparing.set()
        assert reading.wait(2)
        finish.wait(2)
        return True
    def owner(c, env, group):
        def invoke(args):
            reading.set()
            assert preparing.wait(2)
            finish.wait(2)
            return {'loaded': True}
        return SimpleNamespace(invoke=invoke)
    startup = DeclaredStartup(tmp_path, {}, (replace(component(), required_preparations=('cpu',)),),
        (Preparation('cpu', ('dit',), cpu, lambda v: v is True),), owner_factory=owner).start()
    try:
        assert reading.wait(2) and preparing.wait(2)
        assert startup.check_before_use('dit') is False
        finish.set()
        startup.before_use('dit')
    finally:
        finish.set(); startup.close()


@pytest.mark.parametrize('kind', ['failed','false','non-bool','cancelled'])
def test_required_failure_never_reaches_native_compute(tmp_path, kind):
    calls = []
    def prepare(_):
        if kind == 'failed': raise ValueError('model init failed')
        if kind == 'cancelled': raise CancelledError()
        return True
    validate = (lambda _: False) if kind == 'false' else (lambda _: 1) if kind == 'non-bool' else (lambda _: True)
    startup = plan(tmp_path, layout=prepare, validate=validate, calls=calls).start()
    try:
        startup.request({'request': True})
        with pytest.raises((ValueError, RuntimeError, CancelledError)):
            startup.invoke('dit', ('generate',))
        assert ('generate',) not in calls
    finally:
        startup.close()


def test_fresh_request_cannot_reuse_previous_request_readiness(tmp_path):
    calls = []
    startup = plan(tmp_path, calls=calls).start()
    try:
        startup.request({'request': True})
        startup.invoke('dit', ('generate',))
        startup.request({'request': False})
        with pytest.raises(RuntimeError, match='receipt rejected'):
            startup.invoke('dit', ('generate',))
        assert calls.count(('preload',)) == calls.count(('generate',)) == 1
    finally:
        startup.close()


def test_cpu_metadata_runs_during_io_and_gpu_callback_waits_for_both(tmp_path):
    metadata_ready, release_io = Event(), Event()
    events=[]
    def owner(c, env, group):
        def invoke(args):
            assert metadata_ready.wait(2)
            release_io.wait(2)
            events.append('weights-ready')
            return {'loaded':True}
        return SimpleNamespace(invoke=invoke)
    def prepare(inputs):
        events.append('cpu-ready');metadata_ready.set();return 'metadata'
    def initialize(inputs):
        assert inputs['prepared']=='metadata' and 'weights-ready'in events
        events.append('gpu-init');return True
    declaration=replace(component(), required_preparations=('layout',))
    startup=DeclaredStartup(tmp_path,{},(declaration,),(
        Preparation('layout',('dit',),initialize,lambda v:v is True,inputs=('request',),
                    prepare=prepare,validate_prepared=lambda v:v=='metadata',weights=('dit',)),
    ),owner_factory=owner).start()
    try:
        startup.request({'request':True})
        assert metadata_ready.wait(2) and 'gpu-init'not in events
        release_io.set();startup.before_use('dit')
        assert events==['cpu-ready','weights-ready','gpu-init']
    finally:
        release_io.set();startup.close()


def test_real_consumer_dependency_defers_gpu_layout_without_blocking_cpu(tmp_path):
    metadata_ready=Event();events=[]
    preparation=Preparation('layout',('dit',),lambda inputs:events.append('gpu') or True,
        lambda v:v is True,inputs=('request',),prepare=lambda inputs:metadata_ready.set() or True,
        validate_prepared=lambda v:v is True,weights=('dit',),at_use=True)
    startup=DeclaredStartup(tmp_path,{},(replace(component(),required_preparations=('layout',)),),
        (preparation,),owner_factory=lambda *args:SimpleNamespace(invoke=lambda args:{'loaded':True})).start()
    try:
        startup.request({'request':True});startup.ready('dit')
        assert metadata_ready.wait(2) and events==[]
        startup.before_use('dit')
        assert events==['gpu']
    finally:startup.close()


def test_close_cancels_preparation_waiting_for_its_real_consumer(tmp_path):
    startup=DeclaredStartup(tmp_path,{},(replace(component(),required_preparations=('layout',)),),
        (Preparation('layout',('dit',),lambda inputs:True,lambda v:v is True,
                    inputs=('request',),at_use=True),),
        owner_factory=lambda *args:SimpleNamespace(invoke=lambda args:{'loaded':True})).start()
    pending=startup.request({'request':True})['layout']
    startup.ready('dit');startup.close()
    assert pending.cancelled()
    with pytest.raises(RuntimeError,match='open started'):startup.check_before_use('dit')


def test_component_view_keeps_required_work_and_does_not_load_other_models(tmp_path):
    calls=[]
    components=(Component('encoder','/native/encoder',('preload',),required_preparations=('backend',),validate=lambda r:r is True),
                Component('dit','/native/dit',('preload',),required_preparations=('backend','layout'),validate=lambda r:r is True))
    preparations=(Preparation('backend',('encoder','dit'),lambda _:calls.append('backend')or True,lambda v:v is True),
                  Preparation('layout',('dit',),lambda _:pytest.fail('unused DiT preparation'),lambda _:True,inputs=('request',),weights=('dit',)))
    def owner(c,env,group):
        calls.append(c.name);return SimpleNamespace(invoke=lambda args:True)
    startup=DeclaredStartup(tmp_path,{},components,preparations,consumers=('encoder',),owner_factory=owner).start()
    try:
        startup.before_use('encoder')
        assert set(calls)=={'backend','encoder'}
        assert startup.required_preparations=={'encoder':('backend',)}
        with pytest.raises(ValueError,match='unknown declared'):startup.before_use('dit')
    finally:startup.close()


def test_cancelled_required_cpu_job_is_not_dispatched_later(tmp_path):
    loading,release=Event(),Event();ran=[]
    def owner(*args):
        def invoke(args):
            loading.set();assert release.wait(2);return True
        return SimpleNamespace(invoke=invoke)
    spec=Component('dit','/native/dit',('preload',),required_preparations=('layout',),validate=lambda value:value is True)
    prep=Preparation('layout',('dit',),lambda inputs:ran.append('gpu')or True,lambda value:value is True,
        inputs=('request',),prepare=lambda inputs:ran.append('cpu')or True,
        validate_prepared=lambda value:value is True,weights=('dit',))
    startup=DeclaredStartup(tmp_path,{},(spec,),(prep,),workers=1,native_policy=False,owner_factory=owner).start()
    try:
        assert loading.wait(2)
        pending=startup.request({'request':True})['layout'];assert pending.cancel()
        release.set()
        with pytest.raises(CancelledError):startup.before_use('dit')
    finally:release.set();startup.close()
    assert ran==[]


def test_preload_readiness_includes_static_init_but_not_missing_request_inputs(tmp_path):
    entered,release=Event(),Event()
    def initialize(inputs):entered.set();assert release.wait(2);return True
    startup=DeclaredStartup(tmp_path,{},(component(),),(
        Preparation('backend',('dit',),initialize,lambda value:value is True),
        Preparation('layout',('dit',),lambda inputs:True,lambda value:value is True,inputs=('request',))),
        owner_factory=lambda *args:SimpleNamespace(invoke=lambda args:{'loaded':True})).start()
    try:
        startup.ready('dit');assert entered.wait(2)
        assert startup.weights_ready and not startup.preload_ready
        release.set();startup.static['backend'].result(2)
        assert startup.preload_ready
        with pytest.raises(RuntimeError,match='request input'):startup.before_use('dit')
    finally:release.set();startup.close()
