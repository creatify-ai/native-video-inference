"""Admission for the official MiniMax-H3 base checkpoint.

Only the documented configuration, request and checkpoint-identity fields are accepted.
Unknown fields are rejected instead of ignored, so a package or request that declares
anything beyond the official base model fails before any native process starts.
"""
import json
import os
from pathlib import Path
import re

DEFAULT_SIGMA_POINTS = 50
OFFICIAL_REPO = 'MiniMaxAI/MiniMax-H3'
PARTITIONS = {'fl2va': 'FL2VA', 'ref2va': 'Ref2VA'}
CONFIG_KEYS = frozenset({'workflow', 'nfe', 'num_inference_steps', 'programs', 'models', 'environment',
                         'runtime', 'media', 'latent_only', 'attention_backend', 'program_sha256',
                         'request_timeout_s'})
PROGRAM_KEYS = frozenset({'conditioner', 'dit', 'video_vae', 'audio_vae', 'pipeline', 'audio_encoder', 'cp_coordinator'})
MODEL_KEYS = frozenset({'conditioner', 'dit', 'video_vae', 'audio_vae', 'router', 'dense_library',
                        'video_encoder', 'audio_encoder'})
RUNTIME_KEYS = frozenset({'startup_engine', 'layout_preparation', 'conditioning_preparations',
                          'backend_plans', 'readers', 'prefetch', 'prefetch_workers', 'metadata_capture', 'light_sha',
                          'reader_budget_mib', 'startup_library', 'startup_library_sha256', 'context_parallel'})
CONTEXT_PARALLEL_KEYS = frozenset({'world', 'shards', 'manifest_sha256', 'slices_sha256'})
REQUEST_KEYS = frozenset({'prompt', 'width', 'height', 'num_frames', 'seed', 'fps', 'first_png', 'last_png',
                          'reference_descriptor', 'attention_backend', 'workflow', 'nfe',
                          'num_inference_steps'})
SOURCE_KEYS = frozenset({'repo_id', 'variant', 'revision', 'partition'})
# One process owns one official checkpoint; model catalogs and shared views are not public.
SINGLE_MODEL_ENVIRONMENT = ('H3_DIT_ON_DEMAND_CATALOG', 'H3_DIT_MODEL_CATALOG', 'H3_DIT_SHARED_MODEL_VIEWS')


def _only(value, allowed, what):
    if not isinstance(value, dict):
        raise ValueError(f'public H3 {what} must be a JSON object')
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f'public H3 does not accept {what} fields: ' + ', '.join(unknown))


def _identities(value, top=True):
    """Count checkpoint identities other than the top-level model_source."""
    if isinstance(value, dict):
        count = int(not top and ('repo_id' in value or 'model_source' in value))
        for key, item in value.items():
            if not (top and key == 'model_source'):
                count += _identities(item, False)
        return count
    if isinstance(value, list):
        return sum(_identities(item, False) for item in value)
    return 0


def validate_source(manifest, workflow=None):
    """Admit one pinned official base checkpoint identity and nothing else."""
    if not isinstance(manifest, dict):
        raise ValueError('public H3 model manifest must be a JSON object')
    source = manifest.get('model_source')
    _only(source, SOURCE_KEYS, 'model_source')
    if (source.get('repo_id') != OFFICIAL_REPO or source.get('variant') != 'base'
            or re.fullmatch('[0-9a-f]{40}', str(source.get('revision', ''))) is None):
        raise ValueError('public H3 requires a pinned official base checkpoint identity')
    if 'partition' in source and source['partition'] not in PARTITIONS.values():
        raise ValueError('public H3 requires an official partition')
    if workflow is not None and source.get('partition', PARTITIONS[workflow]) != PARTITIONS[workflow]:
        raise ValueError('model partition differs from the selected workflow')
    if _identities(manifest):
        raise ValueError('public H3 model packages declare exactly one checkpoint source')


def validate_config(config):
    _only(config, CONFIG_KEYS, 'configuration')
    if config.get('workflow') not in PARTITIONS:
        raise ValueError('public H3 requires an official partition')
    _only(config.get('programs'), PROGRAM_KEYS, 'program')
    _only(config.get('models'), MODEL_KEYS, 'model')
    _only(config.get('runtime', {}), RUNTIME_KEYS, 'runtime')
    parallel = config.get('runtime', {}).get('context_parallel')
    if parallel is not None:
        _only(parallel, CONTEXT_PARALLEL_KEYS, 'context_parallel')
        if (parallel.get('world') not in (2, 4, 8) or 'cp_coordinator' not in config['programs']
                or any(re.fullmatch('[0-9a-f]{64}', str(parallel.get(k, ''))) is None
                       for k in ('manifest_sha256', 'slices_sha256'))):
            raise ValueError('public H3 context parallelism needs 2, 4 or 8 ranks, the coordinator and pinned shards')
    environment = config.get('environment', {})
    if not isinstance(environment, dict) or not all(isinstance(v, str) for v in environment.values()):
        raise ValueError('public H3 environment must map names to strings')
    for key in dict(os.environ, **environment):
        if key in SINGLE_MODEL_ENVIRONMENT:
            raise ValueError('public H3 owns one official base checkpoint per process')
    manifest = json.loads((Path(config['models']['dit']) / 'manifest.json').read_bytes())
    validate_source(manifest, config['workflow'])
    return manifest


def evaluation_count(config, request):
    _only(request, REQUEST_KEYS, 'request')
    explicit_points = request.get('num_inference_steps', config.get('num_inference_steps'))
    explicit_nfe = request.get('nfe', config.get('nfe'))
    points = explicit_points if explicit_points is not None else (
        explicit_nfe + 1 if type(explicit_nfe) is int else DEFAULT_SIGMA_POINTS)
    if type(points) is not int or not 2 <= points <= 1000:
        raise ValueError('num_inference_steps must contain 2..1000 sigma points')
    if explicit_nfe is not None and (type(explicit_nfe) is not int or explicit_nfe != points - 1):
        raise ValueError('NFE must equal num_inference_steps minus one')
    return points - 1
