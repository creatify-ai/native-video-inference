"""Public process ownership: independent startup, late failure and cancellation."""
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

import pytest

NATIVE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NATIVE))
from core_infra.runtime import NativeOwner, ProcessGroup, ReaderBudget
from core_infra.runtime.files import publish_bundle, read_ahead_chunks, scan_read_ahead
import hashlib


SERVER = r'''
import json,socket,sys,time
s=socket.socket(socket.AF_UNIX);s.bind(sys.argv[1]);s.listen()
while True:
 c,_=s.accept()
 with c:
  args=json.loads(c.makefile('rb').readline())['args']
  if args[0]=='wait':time.sleep(60)
  c.sendall((json.dumps({'ok':args[0]!='fail','error':'injected failure',
                        'details':{'args':args}})+'\n').encode())
'''


@pytest.fixture
def socket_root():
    with tempfile.TemporaryDirectory(prefix='native-test-', dir='/tmp') as directory:
        yield Path(directory)


def test_native_owner_reuses_its_process_and_reaps_on_late_error(socket_root):
    address = socket_root / 'owner.sock'
    owner = NativeOwner([sys.executable, '-c', SERVER, str(address)], address, None, socket_root / 'owner.log')
    try:
        assert owner.invoke(['preload']) == {'args': ['preload']}
        pid = owner.process.pid
        assert owner.invoke(['request']) == {'args': ['request']}
        assert owner.process.pid == pid
        with pytest.raises(RuntimeError, match='injected failure'):
            owner.invoke(['fail'])
    finally:
        owner.terminate()
        assert owner.reap()['reaped']
    assert owner.process.poll() is not None


def test_group_cancels_a_blocked_rpc_and_joins_startup_threads(socket_root):
    group = ProcessGroup(workers=2)
    address = socket_root / 'owner.sock'
    owner = NativeOwner([sys.executable, '-c', SERVER, str(address)], address, None, socket_root / 'owner.log')
    group.owners.append(owner)
    pending = group.submit(owner.invoke, ['wait'])
    limit = time.monotonic() + 5
    while not address.exists():
        assert time.monotonic() < limit
        time.sleep(.005)
    started = time.monotonic()
    assert group.close()[0]['reaped']
    assert time.monotonic() - started < 5
    with pytest.raises((RuntimeError, OSError)):
        pending.result()
    assert not any(t.name.startswith('native-startup') for t in threading.enumerate())


def test_reader_budget_checks_complete_transient_bound():
    assert ReaderBudget(24, limit_mib=1664).validate().bound_mib == 1664
    with pytest.raises(ValueError):
        ReaderBudget(24, limit_mib=1600).validate()


def test_h3_module_imports_without_private_controllers():
    module = importlib.import_module('models.minimax_h3_official.native_pipeline')
    assert module.H3Runtime.__module__ == 'models.minimax_h3_official.runtime'
    # A fresh interpreter: other tests may already have imported an ML framework.
    probe = ('import sys, models.minimax_h3_official.native_pipeline; '
             "print(sorted(n for n in sys.modules if n.split('.')[0] in ('torch', 'triton', 'diffusers', 'transformers')))")
    result = subprocess.run([sys.executable, '-c', probe], cwd=NATIVE, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == '[]'


def test_resident_h3_request_reuses_owners_and_binds_the_new_output(tmp_path):
    from models.minimax_h3_official.runtime import H3Runtime
    with H3Runtime({'workflow': 'fl2va'}) as runtime:
        runtime.started = 1.0
        runtime.requests_started = 1
        runtime.request_environment = {'EXAMPLE': 'fixed'}
        runtime.attention_backend = 'subblock_sparse_attn'
        runtime.input_names = ['conditioner', 'dit', 'video_encoder']
        runtime.has_video_reference = False
        owners = runtime.owners
        runtime._spawn = lambda *args: pytest.fail('resident request spawned another owner')
        runtime.start(tmp_path / 'second', {'EXAMPLE': 'fixed'}, {'first_png': 'frame.png'})
        assert runtime.owners is owners
        assert runtime.output == tmp_path / 'second'
        assert runtime.started == 1.0 and runtime.requests_started == 2
        with pytest.raises(ValueError, match='attention backend changed'):
            runtime.start(tmp_path / 'third', {'EXAMPLE': 'fixed'}, {'attention_backend': 'torch_cudnn_sdpa'})
        with pytest.raises(ValueError, match='environment changed'):
            runtime.start(tmp_path / 'third', {}, {'first_png': 'frame.png'})
        runtime.input_names = ['conditioner', 'dit']
        with pytest.raises(ValueError, match='new input component'):
            runtime.start(tmp_path / 'third', {'EXAMPLE': 'fixed'}, {'first_png': 'frame.png'})


def test_pipeline_session_serializes_requests_and_closes_after_its_context(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from models.minimax_h3_official.native_pipeline import MiniMaxH3Pipeline
    pipeline = object.__new__(MiniMaxH3Pipeline)
    pipeline.config = {'workflow': 'fl2va'}
    active, maximum, identities = 0, 0, []
    def request(value, output, runtime):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        identities.append(id(runtime))
        time.sleep(.02)
        active -= 1
        return value
    pipeline._run = request
    with pipeline.session() as session:
        with ThreadPoolExecutor(2) as pool:
            assert list(pool.map(lambda n: session.run(n, tmp_path / str(n)), [1, 2])) == [1, 2]
    assert maximum == 1 and len(set(identities)) == 1
    with pytest.raises(RuntimeError, match='session is closed'):
        session.run(3, tmp_path / '3')


def test_pipeline_session_retires_owners_after_a_request_failure(tmp_path):
    from models.minimax_h3_official.native_pipeline import MiniMaxH3Pipeline
    pipeline = object.__new__(MiniMaxH3Pipeline)
    pipeline.config = {'workflow': 'fl2va'}
    def fail(*args):
        raise RuntimeError('injected component failure')
    pipeline._run = fail
    with pipeline.session() as session:
        with pytest.raises(RuntimeError, match='injected component failure'):
            session.run({}, tmp_path / 'bad')
        assert session.runtime.closed
        with pytest.raises(RuntimeError, match='session is closed'):
            session.run({}, tmp_path / 'later')


def test_unverified_backend_bundle_is_never_published(tmp_path):
    (tmp_path / 'one.plan').write_bytes(b'accepted plan')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'files': {'one.plan': '0' * 64}}))
    expected = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='asset SHA256'):
        publish_bundle(manifest, expected, tmp_path / 'published')
    assert not (tmp_path / 'published').exists()
    manifest.write_text(json.dumps({'files': {'one.plan': hashlib.sha256(b'accepted plan').hexdigest()}}))
    expected = hashlib.sha256(manifest.read_bytes()).hexdigest()
    publish_bundle(manifest, expected, tmp_path / 'published')
    assert (tmp_path / 'published/one.plan').read_bytes() == b'accepted plan'


def test_native_service_admits_plan_assets_before_spawning_the_model(tmp_path, monkeypatch):
    from core_infra.runtime import service
    payload = tmp_path / 'one.plan'
    payload.write_bytes(b'accepted plan')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'files': {'one.plan': hashlib.sha256(payload.read_bytes()).hexdigest()}}))
    pin = {'path': str(manifest), 'sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()}
    captured = []
    class Group:
        def __init__(self, **kwargs): pass
        def owner(self, program, socket, environment, log, **kwargs):
            captured.append(environment)
            return object()
    monkeypatch.setattr(service, 'ProcessGroup', Group)
    # A changed bundle never creates a process or publishes trusted assets.
    payload.write_bytes(b'changed plan')
    rejected = service.NativeService('/unused', tmp_path / 'bad', backend_plans=pin)
    with pytest.raises(ValueError, match='asset SHA256'):
        rejected.start()
    assert not captured and not (tmp_path / 'bad/backend-plans').exists()
    payload.write_bytes(b'accepted plan')
    accepted = service.NativeService('/unused', tmp_path / 'good', backend_plans=pin).start()
    assert accepted.backend_report['forward_executed'] is False
    assert captured[0]['NATIVE_CUDNN_PLAN_CACHE_MODE'] == 'read'
    assert Path(captured[0]['NATIVE_CUDNN_PLAN_CACHE_DIR'], 'one.plan').read_bytes() == b'accepted plan'


def test_missing_optional_plan_bundle_preserves_native_builders(tmp_path):
    from core_infra.runtime.plans import prepare_backend_plans
    report = prepare_backend_plans(None, tmp_path / 'absent')
    assert report['state'] == 'native_builders' and report['environment'] == {}
    assert not (tmp_path / 'absent').exists()


def test_startup_read_ahead_policy_preserves_explicit_transfer_choice(tmp_path, monkeypatch):
    from core_infra.runtime.service import NativeService
    monkeypatch.delenv('NATIVE_LOAD_DIRECT', raising=False)
    default = NativeService('/unused', tmp_path, startup_read_ahead_gib=64)
    assert default.environment['NATIVE_LOAD_AHEAD_GIB'] == '64'
    assert default.environment['NATIVE_LOAD_DIRECT'] == '0'
    explicit = NativeService('/unused', tmp_path, {'NATIVE_LOAD_DIRECT': '1'}, startup_read_ahead_gib=64)
    assert explicit.environment['NATIVE_LOAD_DIRECT'] == '1'
    for invalid in (-1, 193, True):
        with pytest.raises(ValueError, match='budget'):
            NativeService('/unused', tmp_path, startup_read_ahead_gib=invalid)


def test_read_ahead_respects_extent_and_stops_without_reading_another_file(tmp_path):
    path = tmp_path / 'payload'
    path.write_bytes(bytes(range(256)) * 128)
    chunks = read_ahead_chunks([{'path': str(path), 'offset': 0, 'bytes': 32768,
                                 'file_bytes': 32768}], 2, 16384)
    stop = tmp_path / 'stop'
    scan_read_ahead(chunks, stop, 16384)
    chunks[0]['file_bytes'] = 1
    with pytest.raises(ValueError, match='extent differs'):
        scan_read_ahead(chunks, stop, 16384)
    stop.touch()
    path.unlink()
    scan_read_ahead(chunks, stop, 16384)


def test_cached_read_ahead_does_not_poll_stop_for_each_page_group(tmp_path, monkeypatch):
    path = tmp_path / 'cached-payload'
    size = 32 << 20
    path.write_bytes(b'\x17' * size)
    stop = tmp_path / 'stop'
    checks = []
    original = Path.exists
    def counted_exists(self):
        if self == stop:
            checks.append(time.monotonic())
        return original(self)
    monkeypatch.setattr(Path, 'exists', counted_exists)
    scan_read_ahead([dict(path=str(path), offset=0, bytes=size, file_bytes=size)], stop, size)
    assert len(checks) <= 3, 'cached pages must not issue a stop-file stat per 256 KiB'


def test_read_ahead_preserves_a_selected_request_stop_token(tmp_path):
    class Selected:
        def exists(self):return True
    missing=tmp_path/'unrelated-model.raw'
    scan_read_ahead([dict(path=str(missing),offset=0,bytes=4096,file_bytes=4096)],Selected(),4096)
    assert not missing.exists()


def test_selection_change_stops_before_the_next_model_extent(tmp_path):
    payload=tmp_path/'selected-model.raw';payload.write_bytes(b'a'*4096)
    class Selection:
        checks=0
        def exists(self):
            self.checks+=1
            return self.checks>1
    selection=Selection()
    chunks=[dict(path=str(payload),offset=0,bytes=4096,file_bytes=4096),
            dict(path=str(tmp_path/'must-not-open'),offset=0,bytes=4096,file_bytes=4096)]
    scan_read_ahead(chunks,selection,4096)
    assert selection.checks==2 and payload.read_bytes()==b'a'*4096


def test_native_service_file_cache_metadata_never_blocks_owner_or_outlives_handoff(tmp_path, monkeypatch):
    from core_infra.runtime import checkpoint_prefetch, service
    from core_infra.runtime import files
    release, entered, owner_started = threading.Event(), threading.Event(), threading.Event()
    group = ProcessGroup(workers=2)
    class Owner:
        def invoke(self, args): return {'args': args}
    def owner(*args, **kwargs):
        owner_started.set()
        return Owner()
    group.owner = owner
    monkeypatch.setattr(service, 'ProcessGroup', lambda **kwargs: group)
    def slow(*args):
        entered.set()
        assert release.wait(5)
        return [{'path': str(tmp_path / 'weights'), 'bytes': 1, 'file_bytes': 1, 'offset': 0}]
    monkeypatch.setattr(checkpoint_prefetch, 'jobs', slow)
    def unexpected(*args): raise AssertionError('cache helper started after handoff')
    monkeypatch.setattr(files, 'FileReadAhead', unexpected)
    runtime = service.NativeService('/unused', tmp_path / 'service',
        audit={'manifest': 'unused', 'roots': {}}, startup_file_cache=True)
    assert runtime.environment['NATIVE_LOAD_PREFETCH_STOP_FILE'] == str(runtime.file_cache_stop)
    monkeypatch.setattr(runtime, '_start_audit', lambda: None)
    try:
        runtime.start()
        assert owner_started.is_set() and entered.wait(1) and not release.is_set()
        assert runtime.invoke('generate') == {'args': ['generate']}
        assert runtime.file_cache_stop.exists()
        release.set()
        runtime.file_cache_future.result(2)
        assert runtime.file_cache is None and runtime.file_cache_warning is None
    finally:
        release.set()
        runtime.close()



def test_failed_staged_load_stops_deferred_optional_cache_and_joins_metadata(tmp_path, monkeypatch):
    from core_infra.runtime import checkpoint_prefetch, files, service
    release, entered = threading.Event(), threading.Event()
    group = ProcessGroup(workers=2)
    class Owner:
        def invoke(self, args):
            if args == ['shutdown']:
                return {}
            assert not runtime.file_cache_stop.exists()
            raise RuntimeError('final staged payload guard failed before upload')
    group.owner = lambda *args, **kwargs: Owner()
    monkeypatch.setattr(service, 'ProcessGroup', lambda **kwargs: group)
    def deferred_metadata(*args):
        entered.set()
        assert release.wait(5)
        return [{'path': str(tmp_path / 'weights'), 'bytes': 1, 'file_bytes': 1, 'offset': 0}]
    monkeypatch.setattr(checkpoint_prefetch, 'jobs', deferred_metadata)
    def unexpected(*args):
        raise AssertionError('cache helper started after failed request')
    monkeypatch.setattr(files, 'FileReadAhead', unexpected)
    runtime = service.NativeService('/unused', tmp_path / 'service',
        audit={'manifest': 'unused', 'roots': {}}, startup_file_cache=True)
    try:
        runtime.start()
        assert entered.wait(1)
        with pytest.raises(RuntimeError, match='final staged payload guard failed'):
            runtime.invoke('generate')
        assert runtime.requests == 0 and runtime.file_cache_stop.exists()
        release.set()
        runtime.file_cache_future.result(2)
        assert runtime.file_cache is None and runtime.file_cache_warning is None
    finally:
        release.set()
        runtime.close()
    assert group.closed and runtime.file_cache_future.done()


def test_native_service_failed_owner_start_leaves_no_cache_worker(tmp_path, monkeypatch):
    from core_infra.runtime import service
    group = ProcessGroup(workers=2)
    def failed(*args, **kwargs): raise FileNotFoundError('missing native ELF')
    group.owner = failed
    monkeypatch.setattr(service, 'ProcessGroup', lambda **kwargs: group)
    runtime = service.NativeService('/unused', tmp_path / 'service',
        audit={'manifest': 'unused', 'roots': {}}, startup_file_cache=True)
    with pytest.raises(FileNotFoundError, match='missing native ELF'):
        runtime.start()
    assert group.closed and runtime.file_cache_future is None
    assert runtime.file_cache_stop.exists()


def test_native_service_passes_its_pinned_audit_declaration_to_the_loader(tmp_path):
    from core_infra.runtime.service import NativeService, checkpoint_environment
    spec = {'manifest': tmp_path / 'manifest.json', 'sha256': 'a' * 64,
            'roots': {'weights': tmp_path / 'weights'}}
    runtime = NativeService('/unused', tmp_path / 'service', audit=spec)
    assert runtime.environment['NATIVE_CHECKPOINT_MANIFEST'] == str(spec['manifest'])
    assert runtime.environment['NATIVE_CHECKPOINT_SHA256'] == spec['sha256']
    assert json.loads(runtime.environment['NATIVE_CHECKPOINT_ROOTS']) == {'weights': str(spec['roots']['weights'])}
    assert runtime.environment['NATIVE_LOAD_VERIFY'] == 'light'
    with pytest.raises(ValueError, match='offline SHA-256'):
        checkpoint_environment({'manifest': 'unpinned', 'roots': {}})


def test_read_ahead_interleaves_active_files_with_exact_complete_coverage(tmp_path):
    jobs = [{'path': str(tmp_path / name), 'bytes': size}
            for name, size in [('large', 3 * 4096 + 5), ('medium', 2 * 4096), ('small', 7)]]
    chunks = read_ahead_chunks(jobs, 2, 4096)
    assert [Path(c['path']).name for c in chunks[:4]] == ['large', 'medium', 'large', 'medium']
    for job in jobs:
        parts = [c for c in chunks if c['path'] == job['path']]
        assert parts[0]['offset'] == 0 and sum(c['bytes'] for c in parts) == job['bytes']
        assert all(a['offset'] + a['bytes'] == b['offset'] for a, b in zip(parts, parts[1:]))


class SimulatedOwner:
    def __init__(self, name, peers):
        self.name, self.peers = name, peers
        self.process = SimpleNamespace(pid=1000 + len(peers['envs']), poll=lambda: None)

    def invoke(self, args):
        self.peers['seen'].setdefault(self.name, []).append(args)
        if args[0].startswith('preload'):
            self.peers['gates'].setdefault(self.name, threading.Event()).wait(5)
            if self.peers.get('fail') == self.name:
                raise RuntimeError('late preload failure')
            receipt = dict(component='video_vae' if self.name.startswith('video_') else self.name,
                           model_load_count=1, requests_completed=0, forward_executed=False)
            if self.name.startswith('video_'):
                receipt['load_counts'] = dict(encoder=int(self.name == 'video_encoder'),
                    decoder=int(self.name == 'video_decoder'), reference_encoder=0)
            return receipt
        return {'generated': self.name}

    def terminate(self):
        self.peers['gates'].setdefault(self.name, threading.Event()).set()

    def reap(self):
        return {'pid': 0, 'reaped': True}


def simulated_runtime(monkeypatch, output, options=None):
    startup = importlib.import_module('models.minimax_h3_official.runtime')
    peers = {'seen': {}, 'gates': {}, 'envs': {}}
    group = ProcessGroup()
    def owner(program, address, environment, log_path, **options):
        name = address.stem
        with group.guard:
            if group.closed:
                raise RuntimeError('closed')
            value = SimulatedOwner(name, peers)
            group.owners.append(value)
            peers['envs'][name] = environment
            return value
    group.owner = owner
    import core_infra.runtime.components as common
    monkeypatch.setattr(common, 'ProcessGroup', lambda **kwargs: group)
    config = {'workflow': 'fl2va', 'runtime': {'prefetch': False, **(options or {})},
              'programs': {n: '/native/' + n for n in ('conditioner', 'dit', 'video_vae', 'audio_vae')},
              'models': {n: '/models/' + n for n in ('conditioner', 'dit', 'video_vae', 'audio_vae', 'router', 'dense_library')}}
    runtime = startup.H3Runtime(config)
    runtime.start(output, {'H3_VIDEO_VAE_WORLD_SIZE': '8'}, {'first_png': 'frame.png'})
    return runtime, group, peers


def wait_until(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(.005)


def test_public_graph_admits_conditioning_before_dit_and_output_readiness(monkeypatch, socket_root):
    runtime, group, peers = simulated_runtime(monkeypatch, socket_root)
    try:
        wait_until(lambda: all(n in peers['seen'] for n in ('conditioner', 'dit', 'video_encoder')))
        peers['gates']['conditioner'].set()
        assert runtime.invoke('conditioner', ['infer']) == {'generated': 'conditioner'}
        assert not peers['gates']['dit'].is_set()
        assert 'video_decoder' not in peers['seen']
        peers['gates']['dit'].set()
        peers['gates']['video_encoder'].set()
        runtime.startup.ready('dit')
        runtime.startup.ready('video_encoder')
        assert 'video_decoder' not in peers['seen'] and 'audio_vae' not in peers['seen']
        peers['gates']['video_decoder'] = threading.Event()
        peers['gates']['video_decoder'].set()
        assert runtime.invoke('video_decoder', ['decode']) == {'generated': 'video_decoder'}
        assert 'audio_vae' not in peers['seen']
        assert all(env['H3_VIDEO_VAE_WORLD_SIZE'] == '1' for env in peers['envs'].values())
        assert runtime.report()['reader_buffer_bound_mib'] <= 6144
    finally:
        runtime.close()
    assert group.closed


def test_public_graph_late_failure_is_reported_and_all_owners_retired(monkeypatch, socket_root):
    runtime, group, peers = simulated_runtime(monkeypatch, socket_root)
    try:
        wait_until(lambda: all(n in peers['seen'] for n in ('conditioner', 'dit', 'video_encoder')))
        peers['fail'] = 'dit'
        for gate in peers['gates'].values():
            gate.set()
        with pytest.raises(RuntimeError, match='late preload failure'):
            runtime.startup.ready('dit')
    finally:
        runtime.close()
    assert group.closed and all(row['reaped'] for row in group.cleanup)


def test_optional_cache_metadata_cannot_gate_native_input_loading(monkeypatch, socket_root):
    startup = importlib.import_module('models.minimax_h3_official.runtime')
    release = threading.Event()
    def slow_metadata(*args):
        assert release.wait(5)
        return []
    declaration = importlib.import_module('models.minimax_h3_official.runtime.startup')
    monkeypatch.setattr(declaration, 'prefetch_jobs', slow_metadata)
    runtime, group, peers = simulated_runtime(monkeypatch, socket_root, {'prefetch': True})
    try:
        wait_until(lambda: all(n in peers['seen'] for n in ('conditioner', 'dit', 'video_encoder')))
        assert not release.is_set()
        peers['gates']['conditioner'].set()
        assert runtime.invoke('conditioner', ['infer']) == {'generated': 'conditioner'}
    finally:
        release.set()
        runtime.close()
