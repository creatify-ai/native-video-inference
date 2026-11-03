"""Derive real request structure during weight IO; execute no model forward."""
import json
import math
from pathlib import Path
import shutil
import time


def prepare_structure(runtime, request, canvases):
    started = time.monotonic()
    config, output = runtime.config, runtime.output
    root = output / 'layout-preparation'
    root.mkdir()
    scope = dict(request_scope=runtime.request_scope, output=str(output), forward_executed=False)
    programs, models = config['programs'], config['models']
    environment = dict(runtime.environment, H3_DEBUG_AUDIT='0')
    def native(name, program, *args):
        return runtime.group.run_elf(program, list(map(str, args)), environment, root / (name + '.log'))
    prompt, prepared, structure = output / 'prompt.txt', root / 'conditioner', root / 'structure'
    if config['workflow'] == 'ref2va':
        references = json.loads(Path(request['reference_descriptor']).read_bytes())['references']
        if any(ref['type'] != 'image' or ref.get('has_audio') for ref in references):
            return dict(scope, prepared=True, actual_required=True, cpu_prepare_seconds=time.monotonic()-started)
        preparation = native('conditioner', programs['conditioner'], 'prepare-ref2va',
                             models['conditioner'], prompt, request['reference_descriptor'], prepared)
        references = [{'kind': 'image', 'latent_frames': 1, 'latent_height': ref['height'] // 16,
                       'latent_width': ref['width'] // 16, 'audio_rows': 0} for ref in references]
        descriptor = root / 'geometry.json'
        descriptor.write_text(json.dumps({'target': request, 'references': references}))
        layout_root = root / 'layout'
        packed = native('layout', programs['pipeline'], 'layout-ref2va', prepared / 'text_token_tags.raw',
                        descriptor, layout_root)
        # The native FL structure builder supplies the common target shape
        # and tensor metadata. Replace its layout with the native Ref layout.
        manifest = native('structure', programs['pipeline'], 'assemble-structure', '-',
            prepared / 'text_token_tags.raw', '-', '-', request['seed'], structure,
            request['width'], request['height'], request['num_frames'], 0, '--nfe', request['nfe'])
        nv, na, nt = [(layout_root / (name + '_indices.raw')).stat().st_size // 8
                      for name in ('video', 'audio', 'text')]
        dims = {'hidden_states': ([1, nv, 96], 4), 'audio_hidden_states': ([1, na, 32], 4),
                'encoder_hidden_states': ([1, nt, 5120], 2), 'position_ids': ([nv + na + nt, 3], 8),
                'token_tags': ([nv + na + nt], 8), 'video_indices': ([nv], 8),
                'audio_indices': ([na], 8), 'text_indices': ([nt], 8)}
        for name, (shape, width) in dims.items():
            spec = manifest['inputs'][name]
            spec['shape'], spec['bytes'], spec['sha256'] = shape, math.prod(shape) * width, None
            spec['stride'] = [math.prod(shape[index + 1:]) for index in range(len(shape))]
            if name in ('hidden_states', 'audio_hidden_states', 'encoder_hidden_states'):
                with (structure / spec['file']).open('wb') as stream:
                    stream.truncate(spec['bytes'])
            else:
                shutil.copyfile(layout_root / (name + '.raw'), structure / spec['file'])
        manifest.update(workflow='ref2va', nfe=request['nfe'], references=references,
                        condition_video_rows=packed['condition_video_rows'],
                        condition_audio_rows=packed['condition_audio_rows'])
        (structure / 'manifest.json').write_text(json.dumps(manifest))
        grids = preparation['grids']
    else:
        preparation = native('conditioner', programs['conditioner'], 'prepare', models['conditioner'],
            prompt, canvases[0] if canvases else '-', canvases[1] if len(canvases) > 1 else '-',
            prepared, request['width'], request['height'])
        manifest = native('structure', programs['pipeline'], 'assemble-structure', '-',
            prepared / 'text_token_tags.raw', '-', '-', request['seed'], structure,
            request['width'], request['height'], request['num_frames'], len(canvases),
            '--nfe', request['nfe'])
        grids = [(preparation['grid_height'], preparation['grid_width'])] if canvases else []
    # This preparer admits FL images or image-only references. Match the
    # real request's explicit empty Qwen video provenance so its resident
    # layout can be reused. Mixed video references use the native builder.
    manifest['attention_backend'] = request['attention_backend']
    manifest['text_video_token_indices'] = []
    (structure / 'manifest.json').write_text(json.dumps(manifest))
    vision = preparation.get('vision_rows', preparation.get('grid_height', 0) * preparation.get('grid_width', 0))
    images = len(canvases) if runtime.config['workflow'] == 'fl2va' else len(grids)
    temporary = runtime.root / 'qwen-startup-shape.pending'
    temporary.write_text(json.dumps({'schema': 'h3-qwen-startup-shape-v1', 'vision_sequence': vision,
                                     'text_rows': preparation['token_count'], 'images': images}))
    temporary.rename(runtime.environment['H3_QWEN_STARTUP_HINT_FILE'])
    cpu_seconds = time.monotonic() - started
    return dict(scope, prepared=True, actual_required=False, structure=str(structure),
                cpu_prepare_seconds=cpu_seconds)


def validate_structure(value, payload):
    if (not isinstance(value, dict) or value.get('prepared') is not True
            or value.get('forward_executed') is not False
            or value.get('request_scope') != payload['scope'] or value.get('output') != payload['output']
            or type(value.get('actual_required')) is not bool):
        raise RuntimeError('invalid or stale official H3 request structure')
    if not value['actual_required']:
        structure = Path(value['structure'])
        if (not structure.resolve().is_relative_to(Path(payload['output']).resolve())
                or not (structure / 'manifest.json').is_file()):
            raise RuntimeError('official H3 structure belongs to another request')
    return True


def prepare_layout(runtime, request, structure, *, actual=False):
    """Run on the resident DiT owner after the framework weight/backend edges.

    Image-only requests use metadata built concurrently with weight IO. Mixed
    references use their assembled request because video token provenance is
    only known after Qwen preparation. Neither path executes a model forward.
    """
    started = time.monotonic()
    models, output = runtime.config['models'], runtime.output
    input_root = output / 'request' if actual else Path(structure['structure'])
    owner = runtime.owners['dit']
    reply = owner.invoke(['prepare-layout', models['dit'], models['router'], models['dense_library'],
        str(output / 'layout-preparation' / 'dit'), str(request['nfe']), str(input_root)])
    return dict(prepared=reply.get('prepared') is True, native=reply, structure_only=True,
                request_scope=runtime.request_scope, output=str(output), owner_pid=owner.process.pid,
                cpu_prepare_seconds=structure['cpu_prepare_seconds'],
                seconds=time.monotonic()-started, forward_executed=False)
