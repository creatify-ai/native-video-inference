"""Qwen3-VL conditioner package from the official `text_encoder/`, `processor/` and `tokenizer/`.

The conditioning output is hidden state 50, so decoder layers 50 and later, the final norm
and the LM head are not part of the package. Parameters are copied byte for byte. The two
rotary frequency buffers are derived (they are not learned): the text one on CPU and the
vision one on CUDA, as the reference initializes them. The processor configuration is
serialized by `transformers` so the native preprocessor reads every resolved default.
"""
import json
from pathlib import Path
import re
import shutil

from .common import Checkpoint, digest, extract_all, write_json, write_tensor

SELECTED_HIDDEN_STATE = 50
# SHA-256 of the CPU float32 text rotary frequencies of the official config.
TEXT_INV_FREQ_SHA256 = '29fc897bbb1fda6b314c1bcffa10e39b067913f538e4f07ed7cbd223b84c4cbf'


def weight_names(names, layers=SELECTED_HIDDEN_STATE):
    """{native name: checkpoint name} for every tensor that reaches hidden state `layers`."""
    result = {}
    for name in names:
        native = name.removeprefix('model.')
        layer = re.match(r'language_model\.layers\.(\d+)\.', native)
        if native in ('lm_head.weight', 'language_model.norm.weight') or (layer and int(layer.group(1)) >= layers):
            continue
        if not native.startswith(('language_model.', 'visual.')):
            raise ValueError('unexpected text encoder tensor: ' + name)
        result[native] = name
    return result


def rotary_buffers(config, device):
    import torch
    text = config['text_config']
    vision = config['vision_config']
    text_dim = text['head_dim']
    vision_dim = vision['hidden_size'] // vision['num_heads'] // 2
    text_freq = 1.0 / (text['rope_theta'] ** (torch.arange(0, text_dim, 2, dtype=torch.float32) / text_dim))
    vision_freq = 1.0 / (10000.0 ** (torch.arange(0, vision_dim, 2, dtype=torch.float32, device=device) / vision_dim))
    return {'language_model.rotary_emb.inv_freq': text_freq, 'visual.rotary_pos_emb.inv_freq': vision_freq}


def prepare_conditioner(checkpoint, output, *, device='cuda', workers=8):
    from transformers import AutoProcessor
    checkpoint, root = Path(checkpoint), Path(output) / 'conditioner'
    root.mkdir(parents=True, exist_ok=False)
    source = Checkpoint(checkpoint, 'text_encoder')
    names = weight_names(source.tensors)
    weights = extract_all(source, root, names, workers=workers)
    (root / 'model_config').mkdir()
    shutil.copyfile(checkpoint / 'text_encoder/config.json', root / 'model_config/config.json')
    config = json.loads((root / 'model_config/config.json').read_bytes())
    shutil.copytree(checkpoint / 'tokenizer', root / 'tokenizer')
    shutil.copytree(checkpoint / 'processor', root / 'processor')
    processor = AutoProcessor.from_pretrained(str(checkpoint / 'processor'), local_files_only=True)
    processor.save_pretrained(root / 'processor')
    if not (root / 'processor/processor_config.json').is_file():
        raise ValueError('the processor serializer did not write processor_config.json')
    buffers = {name: write_tensor(root, f'buffers/{name}.raw', value)
               for name, value in rotary_buffers(config, device).items()}
    if buffers['language_model.rotary_emb.inv_freq']['sha256'] != TEXT_INV_FREQ_SHA256:
        raise ValueError('text rotary frequencies differ from the official configuration')
    manifest = {'schema': 'h3-conditioner-static-model-v1', 'selected_hidden_state_index': SELECTED_HIDDEN_STATE,
                'weights': weights, 'buffers': buffers, 'activation_files_in_index': 0,
                'request_files_in_index': 0,
                'config_sha256': {str(path.relative_to(root)): digest(path)
                                  for folder in ('model_config', 'processor', 'tokenizer')
                                  for path in sorted((root / folder).rglob('*')) if path.is_file()}}
    sha = write_json(root / 'model-manifest.json', manifest)
    return {'component': 'conditioner', 'root': str(root), 'manifest_sha256': sha, 'weights': len(weights),
            'bytes': sum(spec['bytes'] for spec in weights.values()),
            'processor_class': type(processor).__name__}
