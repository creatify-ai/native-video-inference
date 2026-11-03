"""H3 request conditioning callbacks; core owns release, overlap and joins.

The native programs, commands, paths, seeds and tensor assembly are the same
as the direct pipeline. Callbacks never submit or wait for another model task.
"""
import json
from pathlib import Path
import time
import struct


def _root(context):
    return Path(context.output)


def _scope(context, request, canvases):
    return context.request_scope


def _receipt(context, request, canvases, name, *, files=(), stages=(), metadata=None):
    records = []
    for path in files:
        path = Path(path)
        size = path.stat().st_size
        if not path.is_file() or size <= 0:
            raise RuntimeError('conditioning output is absent or empty: ' + str(path))
        records.append({'path': str(path), 'bytes': size})
    return {'schema': 'h3-conditioning-v1', 'name': name, 'passed': True,
            'request_scope': _scope(context, request, canvases),
            'files': records, 'stages': list(stages), 'metadata': metadata or {}}


def validate(receipt, *, name, scope, root, allow_skip=False):
    if (not isinstance(receipt, dict) or receipt.get('schema') != 'h3-conditioning-v1'
            or receipt.get('name') != name or receipt.get('passed') is not True
            or receipt.get('request_scope') != scope or not isinstance(receipt.get('files'), list)
            or (not receipt['files'] and not (allow_skip and receipt.get('skipped') is True))
            or not isinstance(receipt.get('stages'), list) or not isinstance(receipt.get('metadata'), dict)):
        raise RuntimeError('invalid or foreign H3 conditioning receipt: ' + name)
    if receipt.get('skipped') is True and (not allow_skip or receipt['files'] or receipt['stages']):
        raise RuntimeError('conditioning skip does not match the admitted request')
    root = Path(root).resolve()
    for item in receipt['files']:
        path = Path(item['path'])
        if (not path.resolve().is_relative_to(root) or type(item.get('bytes')) is not int
                or item['bytes'] <= 0 or not path.is_file() or path.stat().st_size != item['bytes']):
            raise RuntimeError('conditioning output differs from the admitted request: ' + name)
    return True


def skipped(context, request, canvases, name):
    result=_receipt(context,request,canvases,name)
    result['skipped']=True
    return result


def _native(context, name, *args, owner=None):
    began = time.monotonic()
    root = _root(context)
    log = root / (name + '.stdout.log')
    if owner is None:
        reply = context.group.run_elf(context.config['programs']['pipeline'], list(map(str, args)),
            context.environment, log)
    else:
        # Rust releases this callback after its own weight and backend edges.
        # Using the owner here avoids re-entering this request's consumer gate.
        reply = context.owners[owner].invoke(list(map(str, args)))
        log.write_text(json.dumps(reply) + '\n')
    return reply, {'name': name, 'seconds': time.monotonic() - began}


def text(context, request, canvases):
    root, models = _root(context), context.config['models']
    if context.config['workflow'] == 'fl2va':
        args = ('infer', models['conditioner'], root/'prompt.txt',
                canvases[0] if canvases else '-', canvases[1] if len(canvases) > 1 else '-',
                root/'conditioner', request['width'], request['height'])
    else:
        args = ('infer-ref2va', models['conditioner'], root/'prompt.txt',
                request['reference_descriptor'], root/'conditioner')
    _, stage = _native(context, 'text_encoder', *args, owner='conditioner')
    return _receipt(context, request, canvases, 'condition-text',
        files=(root/'conditioner/conditioning.raw', root/'conditioner/text_token_tags.raw'), stages=(stage,))


def _references(request):
    value = json.loads(Path(request['reference_descriptor']).read_bytes())
    if value.get('schema') != 'h3-ref2va-normalized-rgb-v1' or not 1 <= len(value['references']) <= 12:
        raise ValueError('Ref2VA requires 1..12 normalized ordered references')
    if any(ref.get('type') not in ('image', 'video', 'audio') for ref in value['references']):
        raise ValueError('unsupported reference modality')
    return value['references']


def prepare_video(context, request, canvases):
    root = _root(context)
    stages, outputs = [], []
    if context.config['workflow'] == 'fl2va':
        count = 24 * (request['height']//16) * (request['width']//16)
        _, stage = _native(context, 'posterior-noise', 'noise', 42, root/'posterior_noise', count)
        stages.append(stage); outputs.append(root/'posterior_noise/noise0.raw')
    else:
        for index, ref in enumerate(_references(request)):
            if ref['type'] not in ('image', 'video'):
                continue
            frames = 1 if ref['type'] == 'image' else ref['frames']
            used = 1 if ref['type'] == 'image' else min(frames, max((frames-5)//17, 1)*17+5)
            latent_frames = 1 if ref['type'] == 'image' else (used+16)//17*5-3
            _, stage = _native(context, f'reference-{index}-noise', 'noise', 42, root/f'reference-{index}-noise',
                24 * latent_frames * (ref['height']//16) * (ref['width']//16))
            stages.append(stage); outputs.append(root/f'reference-{index}-noise/noise0.raw')
    return _receipt(context, request, canvases, 'condition-video-input', files=outputs, stages=stages)


def video(context, request, canvases, prepared):
    root, models = _root(context), context.config['models']
    stages, files, values = list(prepared['stages']), [], {}
    if context.config['workflow'] == 'fl2va':
        _, stage = _native(context, 'video_encode', 'encode', models['video_vae'], canvases[0],
            canvases[1] if len(canvases) > 1 else '-', root/'posterior_noise/noise0.raw', root/'video_encode',
            request['width'], request['height'], owner='video_encoder')
        stages.append(stage)
        files = [root/'video_encode'/f'condition{i}.raw' for i in range(len(canvases))]
    else:
        for index, ref in enumerate(_references(request)):
            kind = ref['type']
            if kind not in ('image', 'video'):
                continue
            output, noise = root/f'reference-{index}-video', root/f'reference-{index}-noise/noise0.raw'
            if kind == 'image':
                args = ('encode', models['video_vae'], ref['file'], '-', noise, output, ref['width'], ref['height'])
                filename, latent_frames = 'condition0.raw', 1
            else:
                args = ('encode-video', models['video_encoder'], ref['file'], noise, output,
                        ref['width'], ref['height'], ref['frames'])
                used = min(ref['frames'], max((ref['frames']-5)//17, 1)*17+5)
                filename, latent_frames = 'condition.raw', (used+16)//17*5-3
            _, stage = _native(context, f'reference-{index}-video', *args, owner='video_encoder')
            stages.append(stage); files.append(output/filename)
            values[str(index)] = dict(video_file=str(output/filename), latent_frames=latent_frames,
                latent_height=ref['height']//16, latent_width=ref['width']//16)
    return _receipt(context, request, canvases, 'condition-video', files=files, stages=stages, metadata=values)


def audio(context, request, canvases):
    root, stages, files, values = _root(context), [], [], {}
    for index, ref in enumerate(_references(request)):
        if not ref.get('has_audio'):
            continue
        output = root/f'reference-{index}-audio'
        _, stage = _native(context, f'reference-{index}-audio', context.config['models']['audio_encoder'],
            ref['audio_file'], output, owner='audio_encoder')
        report = json.loads((output/'report.json').read_bytes())
        if report.get('passed') is not True:
            raise ValueError('reference audio encoding failed')
        rows = report['condition_shape'][0]
        if type(rows) is not int or rows <= 0 or (output/'condition.raw').stat().st_size != rows*32*4:
            raise ValueError('reference audio encoding shape differs')
        stages.append(stage); files.append(output/'condition.raw')
        values[str(index)] = dict(audio_file=str(output/'condition.raw'), audio_rows=rows)
    return _receipt(context, request, canvases, 'condition-audio', files=files, stages=stages, metadata=values)


def assemble(context, request, canvases, dependencies):
    root = _root(context)
    if context.config['workflow'] == 'fl2va':
        args = ('assemble', root/'conditioner/conditioning.raw', root/'conditioner/text_token_tags.raw',
                root/'video_encode/condition0.raw' if canvases else '-',
                root/'video_encode/condition1.raw' if len(canvases) > 1 else '-', request['seed'], root/'request',
                request['width'], request['height'], request['num_frames'], len(canvases), '--nfe', request['nfe'])
    else:
        videos = dependencies.get('condition-video', {}).get('metadata', {})
        audios = dependencies.get('condition-audio', {}).get('metadata', {})
        assembly = {'target': dict(request), 'references': []}
        for index, ref in enumerate(_references(request)):
            item = {'geometry': {'kind': ref['type']}}
            video, audio = videos.get(str(index)), audios.get(str(index))
            if video:
                item['video_file'] = video['video_file']
                item['geometry'].update({n:video[n] for n in ('latent_frames','latent_height','latent_width')})
            if audio:
                item['audio_file'] = audio['audio_file']; item['geometry']['audio_rows'] = audio['audio_rows']
            assembly['references'].append(item)
        (root/'assembly.json').write_text(json.dumps(assembly, indent=2))
        args = ('assemble-ref2va', root/'conditioner/conditioning.raw', root/'conditioner/text_token_tags.raw',
                root/'assembly.json', request['seed'], root/'request')
    _, stage = _native(context, 'assemble', *args)
    finish_manifest(context, request)
    manifest = root/'request/manifest.json'
    metadata = json.loads(manifest.read_bytes())
    if not isinstance(metadata.get('inputs'), dict) or not metadata['inputs']:
        raise RuntimeError('conditioning assembly produced no DiT inputs')
    receipt=_receipt(context, request, canvases, 'condition-assemble', files=(manifest,), stages=(stage,))
    stages=[item for value in dependencies.values() for item in value.get('stages',[])] + [stage]
    summary=dict(schema='h3-framework-conditioning-v1',passed=True,scope=str(root),
        request_scope=receipt['request_scope'],stages=stages)
    (root/'framework-conditioning.json').write_text(json.dumps(summary)+'\n')
    return receipt


def finish_manifest(context, request):
    manifest_path = Path(context.output) / 'request/manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    manifest['attention_backend'] = request['attention_backend']
    # Modality tags label both images and videos as visual. SubBlock needs
    # the narrower video provenance: ordinary FL2VA Qwen images are dense.
    manifest['text_video_token_indices'] = []
    mm_types = Path(context.output) / 'conditioner/prepared/mm_token_type_ids.raw'
    if context.config['workflow'] == 'ref2va' and mm_types.exists():
        raw = mm_types.read_bytes()
        rows = manifest['inputs']['encoder_hidden_states']['shape'][-2]
        if len(raw) != rows * 8:
            raise ValueError('Qwen video provenance shape differs from conditioning')
        types = [value for (value,) in struct.iter_unpack('<q', raw)]
        if any(value not in (0, 1, 2) for value in types):
            raise ValueError('unsupported Qwen multimodal token type')
        manifest['text_video_token_indices'] = [i for i, value in enumerate(types) if value == 2]
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
