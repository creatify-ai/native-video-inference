"""Native gates release on executor events without a Python readiness poll."""
import json
import socket
from pathlib import Path
import tempfile
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from core_infra.runtime.components import Component
from core_infra.runtime.native_startup import DeclaredStartup, Preparation, StartupGate


@pytest.fixture
def gate_root():
    # macOS's normal pytest directory exceeds sockaddr_un's pathname bound.
    with tempfile.TemporaryDirectory(prefix='native-gate-', dir='/tmp') as directory:
        yield Path(directory)


def exchange(gate, kind, *, scope='one', **fields):
    pin = json.loads(gate.environment['NATIVE_STARTUP_GATE'])
    with socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(2)
        connection.connect(pin['path'])
        connection.sendall(json.dumps(dict(pin, kind=kind, scope=scope, **fields)).encode()+b'\n')
        with connection.makefile('rb') as stream:
            return json.loads(stream.readline())


def runtime(tmp_path, release, cpu):
    def owner(*_):
        def invoke(args):
            assert release.wait(2)
            return True
        return SimpleNamespace(invoke=invoke)
    return DeclaredStartup(tmp_path/'loads', {}, (
        Component('dit', '/native/dit', ('preload',), required_preparations=('layout',), validate=lambda value:value is True),
    ), (Preparation('layout', ('dit',), lambda inputs: inputs['prepared'], lambda value:value is True,
        inputs=('request',), weights=('dit',), prepare=lambda inputs:cpu.set() or True,
        validate_prepared=lambda value:value is True),), owner_factory=owner).start()


def test_rust_gate_releases_compute_without_python_poll_or_before_use(tmp_path, gate_root, monkeypatch):
    release, cpu = Event(), Event()
    startup = runtime(tmp_path, release, cpu)
    monkeypatch.setattr(startup, 'check_before_use', lambda *_:pytest.fail('Python readiness check was called'))
    monkeypatch.setattr(startup, 'before_use', lambda *_:pytest.fail('Python readiness wait was called'))
    def publish(scope, facts):
        startup.request(facts)
    gate = StartupGate(startup, gate_root, publish)
    reply, finished = [], Event()
    try:
        assert exchange(gate, 'inputs', inputs={'request':True})['continued'] is True
        assert cpu.wait(1)
        def consume():
            reply.append(exchange(gate, 'before_use', component='dit'));finished.set()
        worker = Thread(target=consume); worker.start()
        assert not finished.wait(.02)
        release.set()
        assert finished.wait(1)
        worker.join()
        assert reply[0]['continued'] is True
        assert gate.consumers == {('one','dit')}
    finally:
        release.set();gate.close();startup.close()


def test_foreign_scope_cannot_reuse_a_ready_native_gate(tmp_path, gate_root):
    release, cpu = Event(), Event(); release.set()
    startup = runtime(tmp_path, release, cpu)
    gate = StartupGate(startup, gate_root, lambda scope,facts:startup.request(facts))
    try:
        assert exchange(gate,'inputs',inputs={'request':True})['continued'] is True
        assert exchange(gate,'before_use',component='dit')['continued'] is True
        rejected = exchange(gate,'before_use',scope='other',component='dit')
        assert rejected['continued'] is False and 'scope' in rejected['error']
        with pytest.raises(RuntimeError,match='scope'):gate.poll()
    finally:
        gate.close();startup.close()


def test_gate_close_interrupts_waiting_native_consumers(tmp_path, gate_root):
    release, cpu = Event(), Event()
    startup = runtime(tmp_path, release, cpu)
    gate = StartupGate(startup, gate_root, lambda scope,facts:startup.request(facts))
    done = Event()
    try:
        assert exchange(gate,'inputs',inputs={'request':True})['continued'] is True
        def consume():
            try:exchange(gate,'before_use',component='dit')
            except (OSError,ValueError):pass
            finally:done.set()
        worker=Thread(target=consume);worker.start()
        assert not done.wait(.02)
        gate.close()
        assert done.wait(1);worker.join()
    finally:
        release.set();gate.close();startup.close()


def test_request_results_are_bounded_and_old_handles_expire(tmp_path):
    release,cpu=Event(),Event();release.set()
    startup=runtime(tmp_path,release,cpu)
    try:
        first=startup.request({'request':True})['layout']
        startup.before_use('dit')
        count=len(startup._objects)
        for _ in range(30):
            startup.request({'request':True});startup.before_use('dit')
            assert len(startup._objects)==count
            with pytest.raises(RuntimeError,match='expired'):first.result()
    finally:startup.close()


def test_deferred_transport_opens_only_when_its_consumer_is_used(tmp_path):
    opened=[]
    def owner(component,*_):
        opened.append(component.name)
        return SimpleNamespace(invoke=lambda args:True)
    startup=DeclaredStartup(tmp_path,{},(
        Component('lazy','/native/lazy',('preload',),required_preparations=(),load_at_use=True,validate=lambda value:value is True),
        Component('eager','/native/eager',('preload',),required_preparations=(),validate=lambda value:value is True),
    ),(),workers=1,owner_factory=owner).start()
    try:
        startup.before_use('eager');assert opened==['eager']
        startup.invoke('lazy',('generate',));assert opened==['eager','lazy']
    finally:startup.close()


def test_preparation_dependency_releases_gpu_after_backend_but_cpu_runs_early(tmp_path):
    release,cpu=Event(),Event();seen=[]
    startup=DeclaredStartup(tmp_path,{},(
        Component('dit','/native/dit',('preload',),required_preparations=('backend','layout'),validate=lambda r:r is True),
    ),(
        Preparation('backend',('dit',),lambda _:release.wait(2) and {'backend':True},lambda value:value=={'backend':True}),
        Preparation('layout',('dit',),lambda data:seen.append(data['preparations']['backend']) or True,
            lambda value:value is True,inputs=('request',),weights=('dit',),after=('backend',),
            prepare=lambda data:cpu.set() or True,validate_prepared=lambda value:value is True),
    ),owner_factory=lambda *_:SimpleNamespace(invoke=lambda args:True)).start()
    try:
        startup.request({'request':True});assert cpu.wait(1)
        assert not startup.check_before_use('dit') and not seen
        release.set();startup.invoke('dit',('generate',))
        assert seen==[{'backend':True}]
    finally:release.set();startup.close()
