"""CPU checks of the Wan Modal example's serving admission, build inputs and timing consumer."""
import importlib.util
import re
import sys
import types
from pathlib import Path

import pytest

WAN = Path(__file__).resolve().parents[1]
ROOT = WAN.parents[1]
EXAMPLE = ROOT / 'examples/modal/wan21.py'


class Resource:
    """Any chained Modal builder call returns the same object."""

    def __getattr__(self, name):
        return lambda *args, **kwargs: self


class App:
    def __init__(self, *args, **kwargs):
        pass

    def cls(self, **kwargs):
        return lambda value: value

    def function(self, **kwargs):
        return lambda value: value

    def local_entrypoint(self):
        return lambda value: value


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def example(monkeypatch):
    modal = types.ModuleType('modal')
    modal.Volume = types.SimpleNamespace(from_name=lambda *a, **k: Resource())
    modal.Image = types.SimpleNamespace(from_registry=lambda *a, **k: Resource(), debian_slim=lambda *a, **k: Resource())
    modal.App = App
    modal.is_local = lambda: False
    modal.enter = modal.exit = modal.method = lambda: lambda value: value
    modal.parameter = lambda default=None, **kwargs: default
    monkeypatch.setitem(sys.modules, 'modal', modal)
    monkeypatch.setenv('WAN21_VOLUME', 'test-volume')
    for name in ('WAN21_MANIFEST_SHA256', 'WAN21_SERVER_SHA256', 'WAN_NO_PREWARM',
                 'NATIVE_LOAD_AHEAD_GIB', 'NATIVE_STARTUP_FILE_CACHE',
                 'NATIVE_LOAD_WORKERS', 'NATIVE_LOAD_CHUNK_MIB', 'NATIVE_LOAD_PINNED_SLOTS',
                 'NATIVE_LOAD_BUDGET_MIB', 'NATIVE_LOAD_VERIFY'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.syspath_prepend(str(ROOT))
    module = load('wan21_modal_example', EXAMPLE)
    monkeypatch.setitem(sys.modules, 'wan21_modal_example', module)
    return module


def test_service_passes_offline_declaration_to_native_owner(example, monkeypatch):
    from core_infra.runtime.service import NativeService
    monkeypatch.setattr(NativeService, 'start', lambda self: self)
    owner = example.Wan21I2V()
    owner.start()
    assert owner.service.audit_spec['sha256'] == example.REFERENCE_MANIFEST_SHA256
    assert owner.service.environment['NATIVE_CHECKPOINT_SHA256'] == owner.service.audit_spec['sha256']
    assert owner.service.environment['NATIVE_CHECKPOINT_MANIFEST'] == f'{example.PREPARED}/manifest.json'
    assert owner.service.environment['NATIVE_LOAD_VERIFY'] == 'light'
    assert owner.service.environment['WAN21_CHECKPOINT'].endswith(f'/snapshots/{example.REVISION}')
    assert owner.entered_monotonic_ns > 0


def test_example_and_preparation_share_the_checkpoint_pins(example):
    prepare = load('wan21_prepare_pins', WAN / 'prepare.py')
    assert (example.REPOSITORY, example.REVISION) == (prepare.REPOSITORY, prepare.REVISION)
    assert example.REFERENCE_MANIFEST_SHA256 == prepare.REFERENCE_MANIFEST_SHA256


@pytest.mark.parametrize('value', ['', '0', '1', 'false'])
@pytest.mark.parametrize('source', ['container', 'override'])
def test_required_backend_preparation_cannot_be_omitted_before_owner_start(example, monkeypatch, value, source):
    from core_infra.runtime import service
    monkeypatch.setattr(service, 'ProcessGroup', lambda **kwargs: pytest.fail('created a native owner'))
    overrides = {'WAN_NO_PREWARM': value} if source == 'override' else None
    if source == 'container':
        monkeypatch.setenv('WAN_NO_PREWARM', value)
    with pytest.raises(ValueError, match='remove WAN_NO_PREWARM'):
        example.native_service(environment_overrides=overrides)


def test_factory_preserves_shared_loader_and_audit_ownership(example):
    owner = example.native_service()
    # Volume reads go straight to the native readers: O_DIRECT, no OS-cache helper.
    assert not owner.file_cache_enabled
    assert owner.environment['NATIVE_LOAD_FILE_IO'] == 'direct'
    assert 'NATIVE_LOAD_PREFETCH_STOP_FILE' not in owner.environment
    assert owner.backend_plans is None
    assert owner.audit_spec['prefixes'] == ['dit.', 't5.', 'vae.', 'clip.visual.']
    assert owner.environment['NATIVE_LOAD_VERIFY'] == 'light'
    assert owner.environment['NATIVE_LOAD_AHEAD_GIB'] == '0'
    assert owner.environment['WAN21_REPORT_SHA'] == '1'
    plans = {'path': '/plans/manifest.json', 'sha256': '0' * 64}
    assert example.native_service(plans=plans).backend_plans == plans
    # No model-local replacement of the native framework's original reader quota.
    assert not any(key in owner.environment for key in (
        'NATIVE_LOAD_WORKERS', 'NATIVE_LOAD_CHUNK_MIB',
        'NATIVE_LOAD_PINNED_SLOTS', 'NATIVE_LOAD_BUDGET_MIB'))
    assert owner.group is None and owner.owner is None and owner.audit_process is None
    assert owner.sha256 is None  # The image identity is separate from an optional executable pin.


@pytest.mark.parametrize('source', ['deployment', 'caller'])
def test_factory_forwards_the_optional_offline_native_server_pin(example, monkeypatch, source):
    pin = '3' * 64
    if source == 'deployment':
        monkeypatch.setenv('WAN21_SERVER_SHA256', pin)
        owner = example.native_service()
    else:
        owner = example.native_service(server_sha256=pin)
    assert owner.sha256 == pin
    assert owner.audit_spec['sha256'] == example.REFERENCE_MANIFEST_SHA256


@pytest.mark.parametrize('pin', ['', 'a' * 63, 'A' * 64, 'g' * 64, True])
def test_invalid_native_server_pin_fails_before_owner_start(example, pin):
    with pytest.raises(ValueError, match='offline lowercase SHA-256'):
        example.native_service(server_sha256=pin)


def test_shared_memory_bound_stays_admitted_by_native_service(example, monkeypatch):
    monkeypatch.setenv('NATIVE_LOAD_AHEAD_GIB', '193')
    with pytest.raises(ValueError, match='0..192 GiB'):
        example.native_service()


@pytest.mark.parametrize('pin', ['', 'a' * 63, 'A' * 64, 'g' * 64])
def test_invalid_deployment_pin_fails_before_owner_start(example, monkeypatch, pin):
    monkeypatch.setenv('WAN21_MANIFEST_SHA256', pin)
    with pytest.raises(ValueError, match='offline lowercase SHA-256'):
        example.checkpoint_declaration()


def test_custom_preparation_keeps_its_offline_pin_and_roots(example, monkeypatch):
    declaration = example.checkpoint_declaration('/other/prepared', '/other/checkpoint', '2' * 64)
    assert declaration['sha256'] == '2' * 64
    assert declaration['roots'] == {'checkpoint': '/other/checkpoint', 'prepared': '/other/prepared'}
    monkeypatch.setenv('WAN21_MANIFEST_SHA256', '4' * 64)
    assert example.checkpoint_declaration()['sha256'] == '4' * 64


def test_build_inputs_name_the_pinned_wheel_layout(example, tmp_path):
    for package in ('cu13', 'cudnn'):
        (tmp_path / 'nvidia' / package / 'include').mkdir(parents=True)
    env = example.sdk_environment(tmp_path, '/opt/cudnn-frontend')
    assert env['WAN_CUDA_ARCH'] == '100a'
    assert env['H3_CUBLAS_ROOT'] == str(tmp_path / 'nvidia/cu13')
    assert env['H3_CUDNN_ROOT'] == str(tmp_path / 'nvidia/cudnn')
    assert env['H3_TORCH_INCLUDE'] == str(tmp_path / 'torch/include')
    assert env['H3_CUTLASS_ROOT'] == str(tmp_path / 'flashinfer/data/cutlass')
    assert env['H3_VENDOR_INCLUDES'].split(':') == [str(tmp_path / 'nvidia/cu13/include'),
                                                    str(tmp_path / 'nvidia/cudnn/include')]
    # Every SDK input that build.rs reads is supplied.
    build = (WAN / 'build.rs').read_text()
    for key in ('CUDA_HOME', 'H3_CUBLAS_ROOT', 'H3_CUDNN_ROOT', 'H3_CUDNN_FRONTEND', 'H3_TORCH_INCLUDE',
                'H3_CUTLASS_ROOT', 'H3_VENDOR_INCLUDES', 'WAN_CUDA_ARCH'):
        assert f'"{key}"' in build and key in env


@pytest.mark.parametrize('arguments', [
    {'image_bytes': b'\xff\xd8\xff\xe0 JPEG', 'prompt': 'x'},
    {'image_bytes': b'\x89PNG\r\n\x1a\n', 'prompt': ''},
    {'image_bytes': b'\x89PNG\r\n\x1a\n', 'prompt': 'x', 'frames': 80},
    {'image_bytes': b'\x89PNG\r\n\x1a\n', 'prompt': 'x', 'seed': -1},
    {'image_bytes': b'\x89PNG\r\n\x1a\n', 'prompt': 'x', 'steps': 30},
    {'image_bytes': b'\x89PNG\r\n\x1a\n', 'prompt': 'x', 'shift': 4.0},
    {'image_bytes': b'\x89PNG\r\n\x1a\n', 'prompt': 'x', 'gpus': 4},
])
def test_request_is_checked_before_any_remote_call(example, monkeypatch, tmp_path, arguments):
    monkeypatch.setattr(example, 'prepare_checkpoint', None)  # any remote call fails differently
    image = tmp_path / 'input'
    image.write_bytes(arguments.pop('image_bytes'))
    with pytest.raises(SystemExit):
        example.main(image=str(image), out=str(tmp_path / 'out.mp4'), **arguments)


def test_two_gpus_reach_the_native_server_on_a_two_gpu_container(example):
    assert example.native_service().environment['WAN21_WORLD_SIZE'] == '1'
    assert example.native_service(gpus=2).environment['WAN21_WORLD_SIZE'] == '2'
    assert example.Wan21I2V.gpus == 1

    class Function:
        def with_options(self, **options):
            return options
    function = Function()
    assert example.on_gpus(function, 1) is function
    assert example.on_gpus(function, 2) == {'gpu': 'B200:2'}


def test_example_schedules_match_the_recorded_tables(example):
    text = (WAN / 'schedules.rs').read_text()
    assert {(steps, shift) for steps, shift in example.SCHEDULES} == {
        (int(steps), float(shift)) for steps, shift in re.findall(r'steps: (\d+), shift: ([0-9.]+)', text)}


def test_timing_consumer_uses_native_monotonic_marker_and_rejects_unix_mark(example):
    details = {'cold': True, 'timing_schema': 'native-first-dit-v1',
               'first_dit_start': {'clock': 'CLOCK_MONOTONIC', 'monotonic_ns': 9_000_000_000},
               'request_received_monotonic_ns': 3_000_000_000,
               'container_entered_monotonic_ns': 1_000_000_000,
               'marks': [['received', 1_790_000_000], ['first_dit_start', 1_790_000_070]]}
    assert example.first_dit_seconds(details) == {'first_dit_start_since_enter': 8, 'first_dit_start_since_request': 6}
    details['cold'] = False
    assert example.first_dit_seconds(details)['first_dit_start_since_enter'] is None
    details['first_dit_start']['clock'] = 'UNIX'
    with pytest.raises(ValueError, match='CLOCK_MONOTONIC'):
        example.first_dit_seconds(details)
    details['first_dit_start']['clock'] = 'CLOCK_MONOTONIC'
    details['request_received_monotonic_ns'] = 10_000_000_000
    with pytest.raises(ValueError, match='out of order'):
        example.first_dit_seconds(details)
