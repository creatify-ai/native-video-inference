"""Command line: `run` prepares every package of a workflow; `cp` adds the context-parallel DiT
package for several GPUs; `config` writes native.json only."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .common import PARTITIONS, REPO_ID, REVISION, verify_files, write_json
from .light_guards import FILENAME as LIGHT_GUARDS, write_light_guards

PLANS = Path(__file__).with_name('plans')
SHARED = ('conditioner',)  # the Qwen conditioner's bytes are the same for every workflow
# Preparation steps; for Ref2VA the video_vae and audio_vae steps also write the reference encoders.
COMPONENTS = ('conditioner', 'dit', 'router', 'video_vae', 'audio_vae')
PACKAGES = {'fl2va': ('conditioner', 'dit', 'router', 'video_vae', 'audio_vae'),
            'ref2va': ('conditioner', 'dit', 'router', 'video_vae', 'audio_vae', 'video_encoder', 'audio_encoder')}
OUTPUTS = {'conditioner': ('conditioner',), 'dit': ('dit-base', 'dit'), 'router': ('router',),
           'video_vae': ('video_vae', 'video_encoder'), 'audio_vae': ('audio_vae', 'audio_encoder')}
MARKERS = {'conditioner': 'model-manifest.json', 'dit': 'manifest.json', 'router': 'launch.tsv',
           'video_vae': 'model.json', 'audio_vae': 'manifest.json', 'video_encoder': 'program.json',
           'audio_encoder': 'manifest.json'}
PROGRAMS = {'conditioner': 'h3-official-conditioner', 'dit': 'h3-official-model-run',
            'video_vae': 'h3-official-video-vae', 'audio_vae': 'h3-official-audio-infer',
            'pipeline': 'h3-official-model-pipeline', 'audio_encoder': 'h3-official-audio-encode'}
CP_COORDINATOR = 'h3-official-cp-coordinator'


def checkpoint_files(workflow):
    pins = json.loads((PLANS / 'checkpoint-files.json').read_bytes())
    if pins['repo_id'] != REPO_ID or pins['revision'] != REVISION:
        raise ValueError('checkpoint pins belong to another revision')
    transformer = 'transformer_ref/' if workflow == 'ref2va' else 'transformer/'
    return {name: sha for name, sha in pins['sha256'].items()
            if name.startswith(('text_encoder/', 'vae/', 'audio_vae/', 'processor/', 'tokenizer/', transformer))}


def step(name, checkpoint, out, workflow, args):
    if name == 'conditioner':
        from .conditioner import prepare_conditioner
        return prepare_conditioner(checkpoint, out, workers=args.workers)
    if name == 'dit':
        from .dit import prepare_dit
        return prepare_dit(checkpoint, out, workflow=workflow, workers=args.workers)
    if name == 'router':
        if args.router_python_path:  # the pinned Triton lives outside the preparation environment
            env = dict(os.environ, PYTHONPATH=os.pathsep.join([args.router_python_path, os.environ.get('PYTHONPATH', '')]))
            report = out / 'router-report.json'
            subprocess.run([sys.executable, '-m', __package__, 'router', '--out', str(out),
                            '--kernels', str(args.router_kernels or out / 'sources/subblock_kernels.py'),
                            '--report', str(report)], env=env, check=True)
            return json.loads(report.read_bytes())
        from .router import compile_router
        return compile_router(out, args.router_kernels or out / 'sources/subblock_kernels.py')
    if name == 'video_vae':
        module = importlib.import_module(__package__ + '.video_vae')
        return module.prepare_video_vae(checkpoint, out, workflow=workflow, workers=args.workers,
                                        decoder_kernel_configs=args.decoder_kernel_configs)
    if name == 'audio_vae':
        module = importlib.import_module(__package__ + '.audio_vae')
        return module.prepare_audio_vae(checkpoint, out, workflow=workflow)
    raise ValueError('unknown preparation component: ' + name)


def remove(path):
    if path.is_symlink():
        path.unlink()
    else:
        shutil.rmtree(path, ignore_errors=True)


def shared_step(name, checkpoint, out, shared, args):
    """A package whose bytes do not depend on the workflow, prepared once under `shared` and linked
    into `out`. `<shared>/<name>.json`, written last, marks it complete."""
    done = shared / f'{name}.json'
    if not done.is_file():
        for directory in OUTPUTS[name]:
            remove(shared / directory)
        shared.mkdir(parents=True, exist_ok=True)
        write_json(done, step(name, checkpoint, shared, args.workflow, args))
    for directory in OUTPUTS[name]:
        (out / directory).symlink_to(os.path.relpath(shared / directory, out), target_is_directory=True)
    return {**json.loads(done.read_bytes()), 'shared': str(shared / name)}


def file_sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def native_config(prepared, workflow, programs, dense_library, attention='subblock_sparse_attn', media=None, world=1):
    """The native.json of a complete prepared directory; checks every package marker. With
    `world` > 1 the DiT runs as that many context-parallel ranks, one per GPU."""
    prepared, programs = Path(prepared), Path(programs)
    missing = [str(prepared / name / MARKERS[name]) for name in PACKAGES[workflow]
               if not (prepared / name / MARKERS[name]).is_file()]
    if missing:
        raise FileNotFoundError('prepared H3 packages are incomplete; missing: ' + ', '.join(missing))
    config = {'workflow': workflow, 'attention_backend': attention,
              'programs': {key: str(programs / binary) for key, binary in PROGRAMS.items()
                           if key != 'audio_encoder' or workflow == 'ref2va'},
              'models': {**{name: str(prepared / name) for name in PACKAGES[workflow]},
                         'dense_library': str(dense_library)},
              'runtime': {'startup_engine': 'framework'}}
    guards = prepared / LIGHT_GUARDS
    if world > 1:
        package = prepared / f'dit-cp{world}'
        if not (package / 'slices.json').is_file():
            raise FileNotFoundError(f'the {world}-rank DiT package is missing; run `prepare cp --world {world}`')
        config['programs']['cp_coordinator'] = str(programs / CP_COORDINATOR)
        config['runtime']['context_parallel'] = {'world': world, 'shards': str(package),
            'manifest_sha256': file_sha256(package / 'manifest.json'), 'slices_sha256': file_sha256(package / 'slices.json')}
        guards = prepared / f'light-sha-cp{world}.json'
    if guards.is_file():
        config['runtime']['light_sha'] = {'path': str(guards), 'sha256': file_sha256(guards)}
    if media is not None:
        config['media'] = media
    return config


def run(args):
    checkpoint, out = Path(args.checkpoint).resolve(strict=True), Path(args.out).absolute()
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / 'preparation.json'
    report = json.loads(report_path.read_bytes()) if report_path.is_file() else {
        'schema': 'h3-official-preparation-v1', 'repo_id': REPO_ID, 'revision': REVISION,
        'partition': PARTITIONS[args.workflow], 'workflow': args.workflow, 'checkpoint': str(checkpoint),
        'components': {}}
    if report['workflow'] != args.workflow or report['revision'] != REVISION:
        raise ValueError('output directory belongs to another workflow or revision')
    if args.verify_checkpoint and 'checkpoint_files' not in report:
        report['checkpoint_files'] = verify_files(checkpoint, checkpoint_files(args.workflow), args.workers)
        write_json(report_path, report)
    for name in args.components or COMPONENTS:
        if name in report['components']:  # resumable: completed steps are kept
            continue
        for directory in OUTPUTS[name]:  # what an interrupted attempt left behind
            remove(out / directory)
        started = time.monotonic()
        if name in SHARED and getattr(args, 'shared', None):
            result = shared_step(name, checkpoint, out, Path(args.shared).absolute(), args)
        else:
            result = step(name, checkpoint, out, args.workflow, args)
        result['seconds'] = time.monotonic() - started
        report['components'][name] = result
        write_json(report_path, report)
        print(json.dumps({'event': 'prepared', 'component': name, 'seconds': result['seconds']}), flush=True)
    if args.programs:
        if 'light_sha256' not in report:
            report['light_sha256'] = write_light_guards(out, PACKAGES[args.workflow], args.workers)
            write_json(report_path, report)
        config = native_config(out, args.workflow, args.programs, args.dense_library, args.attention,
                               {'format': 'mp4', 'capture_raw': True, 'hash_outputs': True})
        report['native_config_sha256'] = write_json(out / 'native.json', config)
        write_json(report_path, report)
    return report


def context_parallel(prepared, world, workers=4):
    """Add the `world`-rank DiT package to a prepared directory and record it in preparation.json."""
    from .cp import prepare_cp
    prepared = Path(prepared).absolute()
    report_path = prepared / 'preparation.json'
    report = json.loads(report_path.read_bytes())
    if 'light_sha256' not in report:
        raise ValueError('prepare the workflow (with --programs) before its CP package')
    started = time.monotonic()
    result = prepare_cp(prepared, world, workers)
    report.setdefault('context_parallel', {})[str(world)] = {**result, 'seconds': time.monotonic() - started}
    write_json(report_path, report)
    return report['context_parallel'][str(world)]


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m models.minimax_h3_official.prepare', description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    full = commands.add_parser('run', help='prepare the packages of one workflow (resumable)')
    full.add_argument('--checkpoint', required=True, help='root of the official Diffusers snapshot')
    full.add_argument('--workflow', choices=sorted(PARTITIONS), required=True)
    full.add_argument('--out', required=True)
    full.add_argument('--shared', help='directory for the workflow-independent conditioner, linked from --out')
    full.add_argument('--components', type=lambda s: s.split(','), help='comma-separated subset')
    full.add_argument('--verify-checkpoint', action='store_true', help='check every used shard SHA-256 first')
    full.add_argument('--workers', type=int, default=8)
    full.add_argument('--router-kernels', type=Path, help='local copy of the pinned SGLang kernels.py')
    full.add_argument('--router-python-path', help='site directory holding the pinned Triton for the router')
    full.add_argument('--decoder-kernel-configs', type=Path, help='per-kernel autotune choices to reproduce')
    full.add_argument('--programs', help='directory of the h3-official-* executables; writes native.json')
    full.add_argument('--dense-library', help='path of libh3_dense.so')
    full.add_argument('--attention', default='subblock_sparse_attn', choices=('subblock_sparse_attn', 'torch_cudnn_sdpa'))
    router = commands.add_parser('router', help='compile only the router (run under the pinned Triton)')
    router.add_argument('--out', type=Path, required=True)
    router.add_argument('--kernels', type=Path, required=True)
    router.add_argument('--report', type=Path, required=True)
    cp = commands.add_parser('cp', help='add the context-parallel DiT package of a prepared directory')
    cp.add_argument('--prepared', type=Path, required=True)
    cp.add_argument('--world', type=int, choices=(2, 4, 8), required=True)
    cp.add_argument('--workers', type=int, default=4)
    config = commands.add_parser('config', help='write native.json for an already prepared directory')
    config.add_argument('--prepared', type=Path, required=True)
    config.add_argument('--workflow', choices=sorted(PARTITIONS), required=True)
    config.add_argument('--programs', required=True)
    config.add_argument('--dense-library', required=True)
    config.add_argument('--attention', default='subblock_sparse_attn', choices=('subblock_sparse_attn', 'torch_cudnn_sdpa'))
    config.add_argument('--world', type=int, default=1, choices=(1, 2, 4, 8))
    args = parser.parse_args(argv)
    if args.command == 'run':
        if not 1 <= args.workers <= 32:
            parser.error('--workers must be 1..32')
        if args.programs and not args.dense_library:
            parser.error('--programs requires --dense-library')
        print(json.dumps(run(args), indent=2))
    elif args.command == 'cp':
        print(json.dumps(context_parallel(args.prepared, args.world, args.workers), indent=2))
    elif args.command == 'router':
        from .router import compile_router
        write_json(args.report, compile_router(args.out, args.kernels))
    else:
        value = native_config(args.prepared, args.workflow, args.programs, args.dense_library, args.attention,
                              {'format': 'mp4', 'capture_raw': True, 'hash_outputs': True}, args.world)
        name = 'native.json' if args.world == 1 else f'native-cp{args.world}.json'
        print(write_json(args.prepared / name, value))


if __name__ == '__main__':
    main()
