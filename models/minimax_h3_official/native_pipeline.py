"""H3 model orchestration using prepared artifacts and native executables.

Python transports requests and files. Tokenization, tensor assembly, sampling,
encoders and decoders execute in Rust/CUDA; this module imports no ML framework.
"""
from dataclasses import dataclass
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import threading
import tempfile

from .runtime import H3Runtime
from .public_variant import evaluation_count, validate_config
from .media import MediaOptions, MP4Stream, resolve_ffmpeg, stereo_sample_count


@dataclass(frozen=True)
class NativeComponent:
    program: Path
    artifacts: Path


class MiniMaxH3Pipeline:
    def __init__(self, config):
        manifest = validate_config(config)
        self.config = config
        self.media_options = MediaOptions.parse(config.get('media'))
        self.model_nfe = manifest['recipe']['nfe']
        self.partition = config['workflow']
        if self.partition not in ('fl2va', 'ref2va'):
            raise ValueError('H3 partition must be fl2va or ref2va')
        programs, models = config['programs'], config['models']
        self.transformer = NativeComponent(Path(programs['dit']), Path(models['dit']))
        self.text_encoder = NativeComponent(Path(programs['conditioner']), Path(models['conditioner']))
        self.video_vae = NativeComponent(Path(programs['video_vae']), Path(models['video_vae']))
        self.audio_vae = NativeComponent(Path(programs['audio_vae']), Path(models['audio_vae']))
        self.tokenizer = self.text_encoder.artifacts / 'tokenizer'
        self.processor = self.text_encoder.artifacts / 'processor'

    @classmethod
    def from_prepared(cls, config_path):
        config = json.loads(Path(config_path).read_bytes())
        # A single runtime wrapper is also accepted for the initial raw migration.
        config = config.get('pipeline_config', config)
        return cls(config)

    def run(self, request, output_dir):
        with H3Runtime(self.config) as runtime:
            return self._run(request, output_dir, runtime)

    @contextmanager
    def session(self):
        """Keep native weights resident across serialized complete requests."""
        with H3Runtime(self.config) as runtime:
            yield _Session(self, runtime)

    def _run(self, request, output_dir, runtime):
        request = dict(request)
        request.setdefault('attention_backend', self.config.get('attention_backend', 'subblock_sparse_attn'))
        if request['attention_backend'] not in ('subblock_sparse_attn', 'torch_cudnn_sdpa'):
            raise ValueError('unsupported public H3 attention_backend')
        request.setdefault('workflow', self.partition)
        if request['workflow'] != self.partition:
            raise ValueError('request partition differs from the selected H3 model')
        width, height = request['width'], request['height']
        requested_frames, seed = request['num_frames'], request['seed']
        if (type(width) is not int or type(height) is not int or min(width, height) < 32
                or width % 32 or height % 32 or type(requested_frames) is not int
                or not 1 <= requested_frames <= 345 or type(seed) is not int or seed < 0):
            raise ValueError('invalid native H3 canvas, frame count or seed')
        if request.get('fps', 24) != 24:
            raise ValueError('H3 uses 24 fps')
        frames = requested_frames + (22 - requested_frames % 17) % 17
        if frames > 345:
            raise ValueError('aligned H3 frame count exceeds 345')
        nfe = evaluation_count(self.config, request)
        if nfe != self.model_nfe:
            raise ValueError('request schedule differs from prepared public checkpoint tables')
        request.update(num_frames=frames, nfe=nfe)
        encoder = resolve_ffmpeg(self.media_options.ffmpeg) if (
            not self.config.get('latent_only', False) and self.media_options.format == 'mp4') else None
        root = Path(output_dir).absolute()
        root.mkdir(parents=True, exist_ok=False)
        (root / 'prompt.txt').write_text(request['prompt'])
        programs, models = self.config['programs'], self.config['models']
        environment = dict(os.environ, **self.config.get('environment', {}))
        library_dirs = sorted({str(Path(program).absolute().parent) for program in programs.values()})
        environment['LD_LIBRARY_PATH'] = ':'.join([*library_dirs, environment.get('LD_LIBRARY_PATH', '')])
        stages = []
        started = time.monotonic()
        runtime.start(root, environment, request)

        def run(name, program, *args):
            program = Path(program)
            with program.open('rb') as file:
                if file.read(4) != b'\x7fELF':
                    raise ValueError('native H3 stage requires an ELF executable: ' + name)
            begin = time.monotonic()
            with (root / (name + '.stdout.log')).open('wb') as stdout, (root / (name + '.stderr.log')).open('wb') as stderr:
                owner = runtime.component(program, args)
                if owner is None:
                    subprocess.run([str(program), *map(str, args)], env=environment,
                                   stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, check=True)
                else:
                    reply = runtime.invoke(owner, list(map(str, args)))
                    stdout.write((json.dumps(reply) + '\n').encode())
            stages.append({'name': name, 'seconds': time.monotonic() - begin})

        p = lambda name: root / name
        pipeline = programs['pipeline']
        if self.partition == 'fl2va':
            first, last = request.get('first_png'), request.get('last_png')
            if last and not first:
                raise ValueError('last frame requires a first frame')
            count = int(bool(first)) + int(bool(last))
            if count:
                run('canvas', pipeline, 'canvas', first, last or '-', p('canvas'), width, height, count)
            canvases = [p('canvas/first.rgb'), *([p('canvas/last.rgb')] if last else [])] if first else []
        else:
            canvases = []
        runtime.prepare_layout(request, canvases)
        stages.extend(runtime.conditioning_stages())
        begin = time.monotonic()
        args = [models['dit'], models['router'], models['dense_library'], str(p('dit')),
                'generate', str(nfe), str(p('request'))]
        reply = runtime.invoke('dit', args)
        stages.append({'name': 'transformer', 'seconds': time.monotonic() - begin, 'native': reply})
        if self.partition == 'ref2va':
            manifest = json.loads(p('request/manifest.json').read_bytes())
            descriptor = {'target': dict(request, num_frames=frames),
                          'condition_video_rows': manifest['condition_video_rows'],
                          'condition_audio_rows': manifest['condition_audio_rows']}
            p('unpack.json').write_text(json.dumps(descriptor))
            run('unpack', pipeline, 'unpack-ref2va', p('dit/final_video.raw'),
                p('dit/final_audio.raw'), p('unpack.json'), p('latents'))
        else:
            run('unpack', pipeline, 'unpack', p('dit/final_video.raw'), p('dit/final_audio.raw'),
                p('latents'), width, height, frames, count)
        media = None
        if not self.config.get('latent_only', False):
            run('audio_decode', programs['audio_vae'], models['audio_vae'],
                p('latents/audio.raw'), p('audio.raw'), p('audio-report.json'))
            if self.media_options.format == 'raw':
                run('video_decode', programs['video_vae'], 'decode', models['video_vae'],
                    p('latents/video.raw'), p('video_decode'), width, height, frames)
            else:
                audio_report = json.loads(p('audio-report.json').read_bytes())
                samples = stereo_sample_count(audio_report)
                if p('audio.raw').stat().st_size != 2 * samples * 4:
                    raise ValueError('native stereo audio byte count differs from its report')
                begin = time.monotonic()
                with tempfile.TemporaryDirectory(prefix='h3-audio-', dir='/tmp') as directory:
                    wav = Path(directory) / 'audio.wav'
                    run('audio_wav', pipeline, 'audio-wav', p('audio.raw'), wav, 2, 32000)
                    with MP4Stream(p('output.mp4'), wav, width, height, frames,
                            ffmpeg=encoder,
                            capture_path=p('video_decode/frames.raw') if self.media_options.capture_raw else None) as stream:
                        run('video_decode', programs['video_vae'], 'decode', models['video_vae'],
                            p('latents/video.raw'), p('video_decode'), width, height, frames,
                            '--rgb-stream', stream.socket_path)
                        media = stream.finish()
                # Encoding overlaps video_decode; this enclosing interval is
                # reported separately, never summed with its child stages.
                stages.append({'name': 'video_decode_and_mp4', 'seconds': time.monotonic() - begin,
                               'overlaps': ['audio_wav', 'video_decode']})
        outputs = {}
        for name in ('dit/final_video.raw', 'dit/final_audio.raw', 'video_decode/frames.raw', 'audio.raw', 'output.mp4'):
            path = p(name)
            if path.exists():
                entry = {'bytes': path.stat().st_size}
                if self.media_options.hash_outputs:
                    with path.open('rb') as file:
                        entry['sha256'] = hashlib.file_digest(file, 'sha256').hexdigest()
                outputs[name] = entry
        report = {'schema': 'h3-standalone-native-v1', 'passed': True, 'workflow': self.partition,
                  'request': request, 'nfe': nfe, 'num_frames': frames, 'outputs': outputs,
                  'stages': stages, 'startup': runtime.report(), 'seconds': time.monotonic() - started,
                  'media': media, 'output_hashes': self.media_options.hash_outputs}
        p('report.json').write_text(json.dumps(report, indent=2) + '\n')
        return report

    def __call__(self, *, prompt, output_dir, width=768, height=768,
                 num_frames=124, seed=0, first_frame=None, last_frame=None,
                 reference_descriptor=None, num_inference_steps=None):
        request = dict(prompt=prompt, width=width, height=height, num_frames=num_frames, seed=seed)
        if num_inference_steps is not None:
            request['num_inference_steps'] = num_inference_steps
        if first_frame is not None:
            request['first_png'] = str(first_frame)
        if last_frame is not None:
            request['last_png'] = str(last_frame)
        if reference_descriptor is not None:
            request['reference_descriptor'] = str(reference_descriptor)
        return self.run(request, output_dir)


class _Session:
    def __init__(self, pipeline, runtime):
        self.pipeline, self.runtime = pipeline, runtime
        self.lock = threading.Lock()

    def run(self, request, output_dir):
        with self.lock:
            if self.runtime.closed:
                raise RuntimeError('H3 session is closed')
            try:
                return self.pipeline._run(request, output_dir, self.runtime)
            except BaseException:
                self.runtime.close()
                raise
