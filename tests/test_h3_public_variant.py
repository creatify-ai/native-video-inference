import copy
import json

import pytest

from models.minimax_h3_official.public_variant import evaluation_count, validate_config, validate_source

SOURCE = {'repo_id': 'MiniMaxAI/MiniMax-H3', 'variant': 'base', 'revision': 'a' * 40}


def test_public_schedule_uses_sigma_point_semantics():
    assert evaluation_count({}, {}) == 49
    assert evaluation_count({'num_inference_steps': 50}, {}) == 49
    assert evaluation_count({}, {'num_inference_steps': 28}) == 27
    for request in ({'nfe': 50, 'num_inference_steps': 50}, {'nfe': True},
                    {'num_inference_steps': 1}, {'num_inference_steps': 50.0}):
        with pytest.raises(ValueError):
            evaluation_count({}, request)


def test_only_documented_request_fields_are_accepted():
    request = {'prompt': 'p', 'width': 768, 'height': 768, 'num_frames': 124, 'seed': 0,
               'first_png': '/in/first.png', 'attention_backend': 'torch_cudnn_sdpa', 'workflow': 'fl2va'}
    assert evaluation_count({}, request) == 49
    for key in ('long_video', 'unknown_option'):
        with pytest.raises(ValueError, match='request fields'):
            evaluation_count({}, dict(request, **{key: True}))


def test_only_one_official_base_identity_is_admitted():
    validate_source({'model_source': SOURCE})
    validate_source({'model_source': dict(SOURCE, partition='Ref2VA')}, 'ref2va')
    with pytest.raises(ValueError, match='partition'):
        validate_source({'model_source': dict(SOURCE, partition='Ref2VA')}, 'fl2va')
    for change in ({'repo_id': 'other/model'}, {'variant': 'derived'}, {'revision': 'main'},
                   {'partition': 'other'}, {'weights': '/other'}):
        with pytest.raises(ValueError):
            validate_source({'model_source': dict(SOURCE, **change)})
    for nested in ({'repo_id': 'other/model'}, [{'model_source': {}}], {'a': {'b': {'repo_id': 'x'}}}):
        with pytest.raises(ValueError, match='exactly one checkpoint source'):
            validate_source({'model_source': SOURCE, 'extension': nested})
    validate_source({'model_source': SOURCE, 'shared_weights': {'root': '/models/base', 'manifest_sha256': 'b' * 64}})


@pytest.fixture
def config(tmp_path):
    dit = tmp_path / 'dit'
    dit.mkdir()
    (dit / 'manifest.json').write_text(json.dumps({'model_source': SOURCE, 'recipe': {'nfe': 49}}))
    return {'workflow': 'fl2va', 'programs': {'dit': '/bin/dit'},
            'models': {'dit': str(dit)}, 'runtime': {'prefetch': True}}


def test_configuration_rejects_unknown_fields_and_model_catalogs(config, monkeypatch):
    assert validate_config(config)['recipe']['nfe'] == 49
    for section, key in ((None, 'extension'), ('programs', 'extension'), ('models', 'extension'),
                         ('runtime', 'checkpoint_guards')):
        changed = copy.deepcopy(config)
        (changed if section is None else changed[section])[key] = '/other'
        with pytest.raises(ValueError, match='fields'):
            validate_config(changed)
    with pytest.raises(ValueError, match='partition'):
        validate_config(dict(config, workflow='t2v'))
    with pytest.raises(ValueError, match='one official base checkpoint'):
        validate_config(dict(config, environment={'H3_DIT_MODEL_CATALOG': '/catalog'}))
    monkeypatch.setenv('H3_DIT_SHARED_MODEL_VIEWS', '1')
    with pytest.raises(ValueError, match='one official base checkpoint'):
        validate_config(config)
