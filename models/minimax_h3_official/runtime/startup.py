"""Official H3 component declarations; execution policy belongs to Rust core."""
from functools import partial

from core_infra.runtime import ReaderBudget
from core_infra.runtime.components import Component
from core_infra.runtime.files import prefetch_stop_path

from .artifacts import prefetch_jobs


READERS = {'conditioner': 24, 'dit': 24, 'video_encoder': 6,
           'video_decoder': 8, 'audio_vae': 4, 'audio_encoder': 4}


def weight_only(receipt, *, component, counts=None):
    if (not isinstance(receipt, dict) or receipt.get('component') != component
            or type(receipt.get('model_load_count')) is not int
            or receipt['model_load_count'] != 1 or type(receipt.get('requests_completed')) is not int
            or receipt['requests_completed'] != 0
            or receipt.get('forward_executed', False) is not False):
        raise RuntimeError('official H3 preload must load only its declared weights')
    if counts is not None and (receipt.get('load_counts') != counts
            or any(type(v) is not int for v in receipt['load_counts'].values())):
        raise RuntimeError('official H3 preload loaded a different VAE scope')
    return True


def components(config, inputs, has_video, directory):
    """Bind the native H3 commands to the shared core Component API.

    The official native readers retain their historical ABI. Its H3 environment
    aliases below come from each declared ReaderBudget, not another scheduler.
    Core also supplies its canonical NATIVE_* reader and lifecycle environment.
    """
    options = config.get('runtime', {})
    overrides = options.get('readers', {})
    if not isinstance(overrides, dict) or overrides.keys() - READERS.keys():
        raise ValueError('unknown official H3 reader component')
    prefetch_workers = options.get('prefetch_workers', {})
    if (not isinstance(prefetch_workers, dict) or prefetch_workers.keys() - {'conditioner', 'dit'}
            or any(type(n) is not int or not 1 <= n <= 64 for n in prefetch_workers.values())):
        raise ValueError('official H3 prefetch_workers requires conditioner/dit quotas in 1..64')
    default_prefetch_workers = 24 if config['workflow'] == 'ref2va' else 8
    light = options.get('light_sha')
    if light is not None and (not isinstance(light, dict) or light.keys() != {'path', 'sha256'}):
        raise ValueError('official H3 light_sha requires a path and its SHA-256')
    names = (*inputs, 'video_decoder', 'audio_vae')
    result = []
    for name in names:
        count = overrides.get(name, READERS[name])
        if type(count) is not int or not 1 <= count <= 32:
            raise ValueError('official H3 native readers require 1..32 workers')
        minimum = 1024 if name == 'video_encoder' else 512 if name == 'audio_encoder' else 0
        budget = ReaderBudget(count, limit_mib=max((2 * count + 4) * 32, minimum)).validate()
        native = 'video_vae' if name.startswith('video_') else name
        models = config['models']
        if name == 'dit':
            args = ('preload', models['dit'], models['router'], models['dense_library'])
        elif name == 'video_encoder':
            args = ('preload-encoder', models['video_vae'], models['video_encoder'] if has_video else '-', '-')
        elif name == 'video_decoder':
            args = ('preload-decoder', models['video_vae'], '-', '-')
        else:
            args = ('preload', models[name])
        environment = dict(H3_MODEL_LOAD_PIPELINE='1', H3_MODEL_LOAD_DIRECT_PINNED='1',
            H3_MODEL_LOAD_INIT_PREFETCH='1', H3_MODEL_LOAD_BALANCED_READERS='1', H3_MODEL_LOAD_SLAB='1',
            H3_MODEL_LOAD_WORKERS=str(budget.workers), H3_MODEL_LOAD_CHUNK_MIB=str(budget.chunk_mib),
            H3_MODEL_LOAD_PINNED_SLOTS=str(budget.pinned_slots), H3_MODEL_LOAD_BUDGET_MIB=str(budget.limit_mib),
            H3_MODEL_LOAD_PREFETCH_STOP_FILE=str(prefetch_stop_path(directory, name)),
            H3_MODEL_LOAD_WIDE_READERS=str(int(budget.workers > 32)),
            H3_MODEL_LOAD_LARGE_BUDGET=str(int(budget.limit_mib > 2048)))
        if light:  # the full payload hashes ran at preparation; see prepare/light_guards.py
            environment.update(H3_MODEL_LOAD_LIGHT_SHA='1', H3_MODEL_LOAD_ASYNC_SHA='1',
                H3_MODEL_LOAD_LIGHT_SHA_MANIFEST=light['path'], H3_MODEL_LOAD_LIGHT_SHA_MANIFEST_SHA256=light['sha256'],
                NATIVE_LOAD_VERIFY='light', NATIVE_LOAD_GUARD_MANIFEST=light['path'],
                NATIVE_LOAD_GUARD_SHA256=light['sha256'])
        if name == 'conditioner':
            environment.update(H3_QWEN_LOAD_INIT_OVERLAP='1', H3_QWEN_BACKEND_PREPARE='1')
        if name == 'dit':
            environment['H3_DIT_LOAD_AUX_WITH_WEIGHTS'] = '1'
        if name in ('video_encoder', 'audio_encoder'):
            environment['H3_MODEL_LOAD_ENCODER_READ_AHEAD'] = '1'
        if name.startswith('video_') and options.get('metadata_capture', True):
            environment['H3_VAE_METADATA_CAPTURE'] = '1'
        required = ['backend']
        if name == 'dit':
            required += ['condition-text', 'condition-assemble', 'request-shape-layout', 'request-layout']
            if 'video_encoder' in inputs:
                required.append('condition-video')
            if 'audio_encoder' in inputs:
                required.append('condition-audio')
        counts = None
        if name.startswith('video_'):
            counts = dict(encoder=int(name == 'video_encoder'), decoder=int(name == 'video_decoder'),
                          reference_encoder=int(name == 'video_encoder' and has_video))
        program = 'cp_coordinator' if native == 'dit' and options.get('context_parallel') else native
        result.append(Component(name, config['programs'][program], args, environment=environment,
            after=tuple(inputs) if name in ('video_decoder', 'audio_vae') else (),
            validate=partial(weight_only, component=native, counts=counts), readers=budget,
            read_ahead=name in ('video_encoder', 'audio_encoder'),
            prefetch=partial(prefetch_jobs, config, name)
                if name in ('conditioner', 'dit') and options.get('prefetch', True) else None,
            prefetch_workers=prefetch_workers.get(name, default_prefetch_workers)
                if name in ('conditioner', 'dit') and options.get('prefetch', True) else None,
            program_sha256=config.get('program_sha256', {}).get(program),
            required_preparations=tuple(required),
            load_at_use=name in ('video_decoder', 'audio_vae')))
    return tuple(result)
