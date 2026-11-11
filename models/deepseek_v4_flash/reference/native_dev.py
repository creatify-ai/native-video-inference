"""Build and run the native DSV4 crate on Modal during the port (development harness).

    modal run models/deepseek_v4_flash/reference/native_dev.py --cmd check      # cargo test (CPU)
    modal run models/deepseek_v4_flash/reference/native_dev.py --cmd run --args "playground ..."   # one B300

The source tree is mounted at call time (no image rebuild per edit); cargo's registry and target
directories live in the `dsv4-native-build` Volume so incremental builds survive containers. The
checkpoint/gold Volume is mounted at /dsv4 as in sglang_b300.py.
"""
import os
import shlex
import subprocess
import time
from pathlib import Path

os.environ.setdefault('MODAL_IMAGE_BUILDER_VERSION', '2025.06')
import modal  # noqa: E402

CUDA_IMAGE = 'nvidia/cuda:13.0.2-devel-ubuntu24.04'
RUST = '1.90.0'
SRC = '/opt/dsv4-src'
BUILD = '/build'
here = Path(__file__).resolve()
repository = here.parents[3] if modal.is_local() else None

data = modal.Volume.from_name(os.environ.get('DSV4_VOLUME', 'dsv4-flash-native'), create_if_missing=True)
cache = modal.Volume.from_name('dsv4-native-build', create_if_missing=True)
# The serving package on a v2 Volume (mounted at /dsv4v2), to compare read rates with v1.
data2 = modal.Volume.from_name('dsv4-flash-native-v2', create_if_missing=True, version=2)
# DeepSeek-V4-Pro: checkpoint, reference runs, per-rank packages and kernel bundles (mounted at /dsv4pro).
pro = modal.Volume.from_name('dsv4-pro-native', create_if_missing=True, version=2)
app = modal.App('dsv4-flash-native-dev')
image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python='3.12')
    .env({'DEBIAN_FRONTEND': 'noninteractive'})
    .apt_install('build-essential', 'ca-certificates', 'curl', 'git', 'pkg-config')
    # The reference image's host compiler (gcc 13.3.0-6ubuntu2~24.04.1): ~24.04 segfaults on
    # flashinfer's trtllm_batched_gemm_runner.cu (the 6966-entry flashinferMetaInfo.h initializer).
    .run_commands('apt-get update && apt-get install -y --only-upgrade gcc-13 g++-13 cpp-13 libstdc++-13-dev && gcc --version')
    .run_commands("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | "
                  f'sh -s -- -y --profile minimal --default-toolchain {RUST}')
    # Reference library sources whose host code the native shims compile against (pinned).
    .run_commands(
        'mkdir -p /opt/deps/flashmla && curl -sL https://codeload.github.com/sgl-project/FlashMLA/tar.gz/'
        '5ddc685e5ed5bee5934657ada3513678b012ef66 | tar xz -C /opt/deps/flashmla --strip-components=1',
        'mkdir -p /opt/deps/flashmla/csrc/3rdparty/cutlass && curl -sL https://codeload.github.com/NVIDIA/cutlass/tar.gz/'
        '2c85f497547eb441f0fe80bfb0fb2e6f9add1719 | tar xz -C /opt/deps/flashmla/csrc/3rdparty/cutlass --strip-components=1')
    # The reference's cuBLASLt (torch cu130's nvidia-cublas wheel): the native GEMMs torch runs
    # through cuBLASLt call the same library build (sha256-checked at run time).
    .run_commands('python3 -m pip install --no-deps --target /opt/deps/cublas nvidia-cublas==13.1.1.3 && '
                  'sha256sum /opt/deps/cublas/nvidia/cu13/lib/libcublasLt.so.13')
    .env({'PATH': '/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
          'CARGO_HOME': f'{BUILD}/cargo-home', 'CARGO_TARGET_DIR': f'{BUILD}/target', 'CUDA_HOME': '/usr/local/cuda'})
)
if modal.is_local():
    image = image.add_local_dir(repository, SRC, ignore=[
        '.git', '.git/**', 'target', 'target/**', '.venv', '.venv/**', '**/__pycache__/**', '**/*.pyc', '**/.DS_Store',
        'docs/**', 'tests/**', 'outputs/**', 'artifacts/**'])


WORK = '/tmp/src'  # writable copy of the mounted tree


def _target(features: str) -> str:
    """The cargo target directory of a feature set: Flash and Pro (`pro`) builds are kept apart."""
    return f"{BUILD}/target-pro" if 'pro' in features.split(',') else f'{BUILD}/target'


def _build(features: str, profile: str = '--release') -> dict:
    feat = f'--features {features}' if features else ''
    return _sh(f'CARGO_TARGET_DIR={_target(features)} cargo build --locked {profile} -p dsv4-flash-native {feat} --bins')


def _sh(cmd, cwd=WORK, env=None, limit=20000):
    print('+', cmd, flush=True)
    t = time.time()
    r = subprocess.run(['bash', '-lc', cmd], cwd=cwd, env=env, capture_output=True, text=True)
    out = r.stderr[-limit:] + r.stdout[-limit:]  # results (stdout) last
    return {'cmd': cmd, 'rc': r.returncode, 's': round(time.time() - t, 1), 'out': out}


@app.function(image=image, cpu=16, memory=65536, timeout=3600, volumes={BUILD: cache})
def check(extra: str = '') -> list:
    """CPU: refresh Cargo.lock for the new member, type-check and run the crate's unit tests."""
    subprocess.run(['cp', '-r', SRC, WORK], check=True)
    steps = [_sh('cp Cargo.lock /tmp/Cargo.lock.before && cargo metadata --format-version 1 >/dev/null && '
                 'diff -u /tmp/Cargo.lock.before Cargo.lock | head -200; true'),
             _sh('cargo test -p dsv4-flash-native --lib ' + extra)]
    cache.commit()
    steps.append(_sh('cat Cargo.lock', limit=10**7))
    return steps


@app.function(image=image, cpu=16, memory=131072, timeout=4 * 3600, volumes={BUILD: cache, '/dsv4': data})
def cpu(args: str = '', features: str = '') -> list:
    """CPU: build the crate's binaries (without CUDA unless `features` says so) and run `dsv4 <args>`."""
    subprocess.run(['cp', '-r', SRC, WORK], check=True)
    steps = [_build(features)]
    cache.commit()
    if steps[-1]['rc'] == 0 and args:
        steps.append(_sh(f'{_target(features)}/release/dsv4 {args}', limit=200000))
    data.commit()
    return steps


@app.function(image=image, cpu=8, memory=65536, timeout=4 * 3600, volumes={BUILD: cache, '/dsv4': data, '/dsv4v2': data2, '/dsv4pro': pro})
def pyjob(args: str = '') -> list:
    """CPU: run a Python tool from the source tree (path relative to the repository) against the Volume."""
    steps = [_sh(f'python3 {args}', cwd=SRC, limit=200000)]
    data.commit()
    data2.commit()
    return steps


@app.function(image=image, gpu='B300', cpu=16, memory=262144, timeout=4 * 3600, volumes={BUILD: cache, '/dsv4': data, '/dsv4v2': data2, '/dsv4pro': pro})
def run(args: str = '', features: str = 'cuda', release: bool = True, env: str = '') -> list:
    """B300: build the crate's binaries with CUDA and run `dsv4 <args>` (`env`: VAR=value prefixes)."""
    profile = '--release' if release else ''
    subprocess.run(['cp', '-r', SRC, WORK], check=True)
    steps = [_build(features, profile)]
    cache.commit()
    if steps[-1]['rc'] == 0 and args:
        binary = f"{_target(features)}/{'release' if release else 'debug'}/dsv4"
        steps.append(_sh(f'{env} {binary} {args}', limit=200000))
    data.commit()
    return steps


@app.function(image=image, gpu='B300', cpu=16, memory=65536, timeout=1800, volumes={'/dsv4': data})
def warm_volume(index: int = 0, root: str = '/dsv4/prepared/e3cd60e7-v2/weights') -> dict:
    """Read every package file once on a B300 host (direct I/O, 64 threads): Modal caches Volume data
    on the hosts that read it, so later cold starts there read at full speed. Returns this host's
    read rate (a cold host is storage-bound, a warm one is not)."""
    import concurrent.futures
    import mmap
    files = sorted(str(p) for p in Path(root).glob('*.safetensors'))
    chunk = 16 << 20

    def read(path):
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        buf = mmap.mmap(-1, chunk)
        n = 0
        try:
            while True:
                got = os.readv(fd, [buf])
                n += got
                if got < chunk:
                    return n
        finally:
            os.close(fd)
            buf.close()

    t = time.time()
    with concurrent.futures.ThreadPoolExecutor(64) as pool:
        total = sum(pool.map(read, files))
    s = time.time() - t
    host = subprocess.run(['bash', '-lc', 'cat /proc/sys/kernel/hostname; nvidia-smi --query-gpu=serial --format=csv,noheader'],
                          capture_output=True, text=True).stdout.split()
    return {'index': index, 'gb': total / 1e9, 's': round(s, 1), 'gb_per_s': round(total / 1e9 / s, 2), 'host': host}


@app.function(image=image, cpu=32, memory=65536, timeout=4 * 3600, volumes={'/dsv4': data, '/dsv4v2': data2})
def copy_package(src: str = '/dsv4/prepared/e3cd60e7-v2', dst: str = '/dsv4v2/prepared/e3cd60e7-v2') -> dict:
    """Copy the prepared package (identical bytes) to the v2 Volume."""
    t = time.time()
    steps = [_sh(f'mkdir -p {dst}/weights && cp -r {src}/manifest.json {src}/config.json {src}/generation_config.json '
                 f'{src}/tokenizer.json {src}/tokenizer_config.json {src}/cuda-cache {dst}/ && '
                 f'ls {src}/weights | xargs -P 64 -I{{}} cp {src}/weights/{{}} {dst}/weights/{{}} && du -sh {dst} && ls {dst}/weights | wc -l', cwd='/')]
    data2.commit()
    return {'s': time.time() - t, 'steps': steps}


@app.function(image=image, cpu=8, memory=32768, timeout=4 * 3600, volumes={'/dsv4': data, '/dsv4v2': data2})
def archive_runs(names: str) -> list:
    """Move run directories (comma-separated names under /dsv4/runs) into tar files on the v2
    Volume (/dsv4v2/archive/<name>.tar, no inode limit), each verified to list every entry before
    its directory is removed from the v1 Volume (500k inodes)."""
    import shutil
    os.makedirs('/dsv4v2/archive', exist_ok=True)
    out = []
    for name in filter(None, names.split(',')):
        src = f'/dsv4/runs/{name}'
        if not os.path.isdir(src):
            out.append(f'{name}: missing')
            continue
        entries = sum(len(ds) + len(fs) for _, ds, fs in os.walk(src)) + 1
        tar = f'/dsv4v2/archive/{name}.tar'
        r = subprocess.run(['tar', '-cf', tar, '-C', '/dsv4/runs', name], capture_output=True, text=True)
        listed = subprocess.run(['tar', '-tf', tar], capture_output=True, text=True).stdout.splitlines() if r.returncode == 0 else []
        if len(listed) != entries:
            out.append(f'{name}: not removed (tar rc {r.returncode}, {len(listed)} of {entries} entries) {r.stderr[-200:]}')
            continue
        data2.commit()
        shutil.rmtree(src)
        data.commit()
        out.append(f'{name}: archived {entries} entries, {os.path.getsize(tar) >> 20} MiB')
    return out


# COLD_START_REGION pins cold_start to a Modal region (such as us-west; pinning costs extra). Read
# where the app is launched.
COLD_START_REGION = os.environ.get('COLD_START_REGION') or None


@app.function(image=image, gpu='B300', cpu=48, memory=262144, timeout=1800, volumes={BUILD: cache, '/dsv4': data, '/dsv4v2': data2, '/dsv4pro': pro},
              region=COLD_START_REGION)
def cold_start(args: str = '', env: str = '', features: str = 'cuda') -> dict:
    """A serving-style cold start: the prebuilt binary (no cargo build, as a production image ships
    it) run once in a fresh container. Returns the container's entry time and the binary's output
    (which carries wall-clock process-start and first-token times)."""
    entered = time.time()
    environ = dict(os.environ, **dict(kv.split('=', 1) for kv in shlex.split(env)))
    prof = environ.pop('PROF', '') == '1'
    if prof:  # kernel activity of the native process (reference/tape/cupti_prof.cpp)
        lib = '/tmp/libcupti_prof.so'
        subprocess.run(['g++', '-std=c++17', '-O2', '-shared', '-fPIC', f'{SRC}/models/deepseek_v4_flash/reference/tape/cupti_prof.cpp',
                        '-I/usr/local/cuda/targets/x86_64-linux/include', '-I/usr/local/cuda/include', '-o', lib,
                        '-L/usr/local/cuda/targets/x86_64-linux/lib', '-lcupti', '-Wl,-rpath,/usr/local/cuda/targets/x86_64-linux/lib'], check=True)
        os.makedirs('/tmp/prof', exist_ok=True)
        environ.update(CUDA_INJECTION64_PATH=lib, PROF_DIR='/tmp/prof')
    r = subprocess.run([f'{_target(features)}/release/dsv4'] + shlex.split(args), env=environ, capture_output=True, text=True)
    out = {'entered': entered, 'rc': r.returncode, 'out': r.stdout[-100000:], 'err': r.stderr[-4000:],
           'region': os.environ.get('MODAL_REGION'), 'cloud': os.environ.get('MODAL_CLOUD_PROVIDER')}
    if prof:
        import sys
        sys.path.insert(0, f'{SRC}/models/deepseek_v4_flash/reference/tape')
        import prof_summary
        out['prof'] = prof_summary.summarize('/tmp/prof')
    return out



@app.function(image=image, gpu='B300:8', cpu=64, memory=262144, timeout=3600, volumes={BUILD: cache, '/dsv4': data, '/dsv4v2': data2, '/dsv4pro': pro},
              region=COLD_START_REGION)
def load_multi(package: str, sha: str, plans: str = '8x16', warm: bool = True) -> dict:
    """Eight B300s, one `dsv4 load-bench` process per GPU, all started together: the aggregate rate
    of the weight load (reads + uploads). `plans`: `<processes>x<readers per process>` items; every
    process loads the whole package onto its own GPU. `warm` reads the package once first."""
    import json
    out = {'region': os.environ.get('MODAL_REGION'), 'cloud': os.environ.get('MODAL_CLOUD_PROVIDER'), 'runs': []}
    binary = f'{BUILD}/target-pro/release/dsv4'

    def launch(procs, readers):
        at = time.time() + 3
        ps = [subprocess.Popen([binary, 'load-bench', package], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               env=dict(os.environ, DSV4_GPU=str(i), DSV4_MANIFEST_SHA256=sha, DSV4_START_EPOCH=f'{at:.3f}',
                                        NATIVE_LOAD_WORKERS=str(readers))) for i in range(procs)]
        rows, errs = [], []
        for p in ps:
            o, e = p.communicate()
            line = next((l for l in o.splitlines() if '"load_bench"' in l), None)
            if line:
                rows.append(json.loads(line)['load_bench'])
            else:
                errs.append((o + e)[-1500:])
        if not rows:
            return {'procs': procs, 'readers': readers, 'errors': errs}
        wall = max(r['epoch_done'] for r in rows) - at
        gb = sum(r['gb'] for r in rows)
        return {'procs': procs, 'readers': readers, 'gb': round(gb, 1), 'wall_s': round(wall, 2),
                'aggregate_gb_per_s': round(gb / wall, 1), 'per_proc_s': [round(r['weights_on_device_s'], 2) for r in rows],
                'context_s': [round(r['cuda_context_s'], 2) for r in rows], 'errors': errs}

    if warm:
        out['warm'] = [launch(1, 64), launch(1, 64)]
    for item in plans.split(','):
        procs, readers = (int(v) for v in item.split('x'))
        r = launch(procs, readers)
        print(r, flush=True)
        out['runs'].append(r)
    return out


def _pro_runs(package: str, readers: int, repeats: int, verify: bool, env: str) -> list:
    """`repeats` loads of the Pro package: one `dsv4 load-bench` process per TP rank, started
    together, each loading its rank's package (/dsv4pro/prepared/<package>/rank<r>, written by the
    reference's gold hook) onto its own GPU. `env`: VAR=value items; a value `a|b|c` takes one
    alternative per repeat (cycled, `-` = unset), for A/B on one host."""
    import json
    binary = f'{BUILD}/target-pro/release/dsv4'
    root = Path('/dsv4pro/prepared') / package
    ranks = json.loads((root / 'package.json').read_text())['ranks']
    sweep = dict(kv.split('=', 1) for kv in shlex.split(env))
    runs = []
    for rep in range(repeats):
        extra = {k: v.split('|')[rep % len(v.split('|'))] for k, v in sweep.items()}
        # Rank processes create their CUDA contexts in turn by default (measured on a warm host:
        # 24.9-27.9 s against 31.0-32.1 s when all eight are created together); `-` turns it off.
        extra.setdefault('DSV4_CTX_LOCK', '/tmp/dsv4-ctx.lock')
        extra = {k: v for k, v in extra.items() if v != '-'}
        if extra.get('DSV4_CTX_LOCK'):
            Path(extra['DSV4_CTX_LOCK']).unlink(missing_ok=True)
        at = time.time() + (0.2 if rep == 0 else 3)
        ps = [subprocess.Popen([binary, 'load-bench', str(root / name)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               # One visible device per rank process: cuInit then sets up one GPU, not eight.
                               env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), DSV4_GPU='0',
                                        DSV4_MANIFEST_SHA256=meta['manifest']['sha256'],
                                        DSV4_START_EPOCH=f'{at:.3f}', NATIVE_LOAD_WORKERS=str(readers),
                                        DSV4_VERIFY='1' if verify and rep == repeats - 1 else '0', **extra))
              for i, (name, meta) in enumerate(sorted(ranks.items()))]
        rows, checks, errs = [], [], []
        for p in ps:
            o, e = p.communicate()
            for l in o.splitlines():
                if '"load_bench"' in l:
                    rows.append(json.loads(l)['load_bench'])
                if '"verify"' in l:
                    checks.append(json.loads(l)['verify'])
            if p.returncode != 0:
                errs.append((o + e)[-1500:])
        r = {'env': extra, 'errors': errs, 'verify': [(c['equal'], c['total']) for c in checks]}
        if rows:
            done = max(x['epoch_done'] for x in rows)
            gb = sum(x['gb'] for x in rows)
            r.update(start_epoch=at, done_epoch=done, wall_s=round(done - at, 2), gb=round(gb, 1),
                     aggregate_gb_per_s=round(gb / (done - at), 1), per_rank_s=[round(x['weights_on_device_s'], 2) for x in rows],
                     context_s=[round(x['cuda_context_s'], 2) for x in rows],
                     ctx_create_s=[round(x.get('ctx_create_s', 0), 2) for x in rows],
                     readers_s=[round(x['readers_started_s'], 2) for x in rows])
        print(r, flush=True)
        runs.append(r)
    return runs


def _pro_forward(package: str, bundles: str, gold_fwd: str, repeats: int, env: str) -> list:
    """`repeats` cold starts of DeepSeek-V4-Pro to its first token: one `dsv4 pro-rank` process per
    TP rank (rank r: GPU r, package rank<r>, kernel bundle `<bundles>-r<r>`), started together;
    they load their weights, prefill the prompt of the reference forward `gold_fwd` and exchange
    their partial sums through a tmpfs directory. Returns, per repeat, the wall time from the
    common start to rank 0's first token and each rank's report."""
    import json
    import shutil
    binary = f'{BUILD}/target-pro/release/dsv4'
    root = Path('/dsv4pro/prepared') / package
    ranks = json.loads((root / 'package.json').read_text())['ranks']
    sweep = dict(kv.split('=', 1) for kv in shlex.split(env))
    runs = []
    for rep in range(repeats):
        extra = {k: v.split('|')[rep % len(v.split('|'))] for k, v in sweep.items()}
        extra.setdefault('DSV4_CTX_LOCK', '/tmp/dsv4-ctx.lock')
        extra = {k: v for k, v in extra.items() if v != '-'}
        if extra.get('DSV4_CTX_LOCK'):
            Path(extra['DSV4_CTX_LOCK']).unlink(missing_ok=True)
        shm = Path('/dev/shm' if Path('/dev/shm').is_dir() else '/tmp') / f'dsv4-pro-{rep}-{int(time.time())}'
        shm.mkdir(parents=True)
        at = time.time() + (0.2 if rep == 0 else 3)
        ps = [subprocess.Popen([binary, 'pro-rank', str(root / name), f'{bundles}-r{i}', gold_fwd, str(shm)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), DSV4_MANIFEST_SHA256=meta['manifest']['sha256'],
                                        DSV4_START_EPOCH=f'{at:.3f}', **extra))
              for i, (name, meta) in enumerate(sorted(ranks.items()))]
        reports, errs = [], []
        for p in ps:
            o, e = p.communicate()
            line = next((l for l in o.splitlines() if '"pro_rank"' in l), None)
            if line:
                reports.append(json.loads(line))
            if p.returncode != 0 or not line:
                errs.append((o[-1200:] + e[-1200:]))
        shutil.rmtree(shm, ignore_errors=True)
        r = {'env': extra, 'errors': errs[:3]}
        if reports:
            first = next((x for x in reports if x['pro_rank']['rank'] == 0), reports[0])
            r.update(first_token_s=round(first['epoch_first_token'] - at, 2), rank0=first['pro_rank'],
                     weights_on_device_s=[x['pro_rank']['timeline_s'].get('weights_on_device') for x in reports],
                     forward_done_s=[x['pro_rank']['timeline_s'].get('forward_done') for x in reports],
                     sums_differing=[x['pro_rank']['sums_differing_from_reference'] for x in reports],
                     in_process_first_token_s=first['pro_rank']['timeline_s'].get('first_token'),
                     slowest_rank_stages=max(reports, key=lambda x: x['pro_rank']['timeline_s'].get('weights_on_device') or 0)['pro_rank'].get('stages_s'))
        print(r, flush=True)
        runs.append(r)
    return runs


@app.function(image=image, gpu='B300:8', cpu=64, memory=262144, timeout=2 * 3600, volumes={BUILD: cache, '/dsv4pro': pro},
              region=COLD_START_REGION)
def pro_load(package: str, readers: int = 16, repeats: int = 1, verify: bool = False, build: bool = False,
             env: str = '') -> dict:
    """DeepSeek-V4-Pro on eight B300s in a fresh container: the wall time until every rank's
    weights are on its device and the aggregate rate; `verify` also reads every tensor back and
    compares SHA-256s with the reference's."""
    entered = time.time()
    out = {'entered': entered, 'region': os.environ.get('MODAL_REGION'), 'cloud': os.environ.get('MODAL_CLOUD_PROVIDER')}
    if build:
        subprocess.run(['cp', '-r', SRC, WORK], check=True)
        out['build'] = _build('cuda,pro')
        cache.commit()
    out['runs'] = _pro_runs(package, readers, repeats, verify, env)
    return out


@app.function(image=image, gpu='B300:8', cpu=64, memory=262144, timeout=2 * 3600, volumes={BUILD: cache, '/dsv4pro': pro, '/dsv4': data},
              region=COLD_START_REGION)
def pro_cold(package: str, bundles: str, gold_fwd: str, repeats: int = 1) -> dict:
    """A cold start of DeepSeek-V4-Pro in a new container: the first of `repeats` runs starts the
    eight rank processes right after the container is entered (later runs warm the host for
    containers that land on it afterwards). Returns the container's entry time, its host name
    and each run's times."""
    entered = time.time()
    host = subprocess.run(['bash', '-lc', 'cat /proc/sys/kernel/hostname; nvidia-smi --query-gpu=serial --format=csv,noheader | head -1'],
                          capture_output=True, text=True).stdout.split()
    return {'entered': entered, 'region': os.environ.get('MODAL_REGION'), 'cloud': os.environ.get('MODAL_CLOUD_PROVIDER'), 'host': host,
            'runs': _pro_forward(package, bundles, gold_fwd, repeats, '')}


PRO_COMMANDS = modal.Queue.from_name('dsv4-pro-commands', create_if_missing=True)
PRO_RESULTS = modal.Queue.from_name('dsv4-pro-results', create_if_missing=True)


@app.function(image=image, gpu='B300:8', cpu=64, memory=262144, timeout=3 * 3600, volumes={BUILD: cache, '/dsv4pro': pro, '/dsv4': data},
              region=COLD_START_REGION)
def pro_session(idle_s: int = 300) -> dict:
    """One 8xB300 container kept for several experiments, so they all run on the same (warm)
    host: runs the commands `--cmd pro-send` queues (each a `_pro_runs` call, with the build
    Volume reloaded first so a rebuilt binary is picked up) and exits after `idle_s` without one."""
    done = 0
    PRO_RESULTS.put({'session': 'ready', 'region': os.environ.get('MODAL_REGION'), 'cloud': os.environ.get('MODAL_CLOUD_PROVIDER')})
    while True:
        try:
            c = PRO_COMMANDS.get(timeout=idle_s)
        except Exception:
            c = None
        if c is None or c.get('stop'):
            break
        try:
            cache.reload()
            if c.get('kind') == 'forward':
                PRO_RESULTS.put({'id': c['id'], 'runs': _pro_forward(c['package'], c['bundles'], c['gold'], c['repeats'], c['env'])})
            else:
                PRO_RESULTS.put({'id': c['id'], 'runs': _pro_runs(c['package'], c['readers'], c['repeats'], c['verify'], c['env'])})
        except Exception as e:
            PRO_RESULTS.put({'id': c['id'], 'error': repr(e)})
        done += 1
    return {'commands': done}


def _prof_summary(prof_dir: str, top: int = 40) -> dict:
    """Decode-graph kernels from cupti_prof records: per replay of the most replayed graph, the device
    span and the kernels (name, mean duration), so two runs can be compared kernel by kernel."""
    import collections
    import glob
    import json
    best = None
    for f in glob.glob(f'{prof_dir}/prof-*.jsonl'):
        rows = [json.loads(l) for l in open(f) if l.strip()]
        graphs = collections.Counter(r['graph'] for r in rows if r['graph'])
        if graphs and (best is None or sum(graphs.values()) > best[2]):
            best = (f, rows, sum(graphs.values()))
    if best is None:
        return {'error': 'no graph kernels recorded'}
    rows = best[1]
    gid = collections.Counter(r['graph'] for r in rows if r['graph']).most_common(1)[0][0]
    ks = sorted((r for r in rows if r['graph'] == gid), key=lambda r: r['start'])
    first = ks[0]['name']
    replays, cur = [], []
    for r in ks:
        if r['name'] == first and cur:
            replays.append(cur)
            cur = []
        cur.append(r)
    if cur:
        replays.append(cur)
    n = len(replays[0])
    full = [rp for rp in replays if len(rp) == n][2:]  # skip the first replays
    spans = [(rp[-1]['end'] - rp[0]['start']) / 1e6 for rp in full]
    busy = [sum(r['end'] - r['start'] for r in rp) / 1e6 for rp in full]
    per = collections.defaultdict(list)
    for rp in full:
        for i, r in enumerate(rp):
            per[(i, r['name'])].append((r['end'] - r['start']) / 1e3)
    kernels = [(i, name[:110], round(sum(v) / len(v), 2)) for (i, name), v in sorted(per.items())]
    by_name = collections.defaultdict(float)
    for i, name, us in kernels:
        by_name[name] += us
    return {'file': best[0], 'kernels_per_replay': n, 'replays': len(full),
            'span_ms': round(sum(spans) / max(len(spans), 1), 3), 'busy_ms': round(sum(busy) / max(len(busy), 1), 3),
            'top_by_name_us': sorted(by_name.items(), key=lambda kv: -kv[1])[:top], 'sequence': kernels}


@app.local_entrypoint()
def main(cmd: str = 'check', args: str = '', extra: str = '', cpus: int = 0, pro: bool = False):
    # --pro: build and run with the `pro` feature (one DeepSeek-V4-Pro rank per GPU); default Flash.
    features = 'cuda,pro' if pro else 'cuda'
    if cmd == 'cold':  # end-to-end: from the remote call to the first token
        import json as _json
        called = time.time()
        r = cold_start.remote(args, extra, features)
        returned = time.time()
        startup = next((_json.loads(l) for l in r['out'].splitlines() if l.startswith('{"startup_s"') or '"startup_s"' in l), None)
        prompts = [_json.loads(l) for l in r['out'].splitlines() if '"equal_to_reference"' in l]
        parity = [p.get('equal_to_reference') for p in prompts]
        itl = [(p.get('decode_itl_ms'), p.get('decode_forward_ms')) for p in prompts]
        if startup is None:
            print(r['out'][-100000:], r['err'][-3000:])
            if 'prof' in r:
                print('PROF', _json.dumps(r['prof']))
            return
        s = dict(startup['startup_s'])
        print(_json.dumps({
            'end_to_end_first_token_s': round(startup['epoch_first_token'] - called, 2),
            'container_start_s': round(startup['epoch_process_start'] - called, 2),
            'process_first_token_s': s['first_token'], 'read_gb_per_s': round(startup['weights']['gb_per_s'], 1),
            'python_entry_to_process_s': round(startup['epoch_process_start'] - r['entered'], 2),
            'parity': parity, 'decode_itl_ms': itl, 'call_returned_s': round(returned - called, 1),
            'region': r.get('region'), 'cloud': r.get('cloud'),
            'graph_replay': [l for l in r['out'].splitlines() if 'graph_replay_ms' in l]}))
        if 'prof' in r:
            print('PROF', _json.dumps(r['prof']))
        return
    if cmd == 'pro-load':  # --args "<package> [readers] [repeats] [verify] [build]"; end to end from the remote call
        import json as _json
        a = args.split()
        called = time.time()
        r = pro_load.remote(a[0], int(a[1]) if len(a) > 1 else 16, int(a[2]) if len(a) > 2 else 1,
                            'verify' in a[3:], 'build' in a[3:], extra)
        first = r['runs'][0] if r['runs'] else {}
        if 'done_epoch' in first:
            r['end_to_end_s'] = round(first['done_epoch'] - called, 2)
            r['container_start_s'] = round(r['entered'] - called, 2)
        print(_json.dumps(r, indent=1))
        return
    if cmd == 'pro-cold':  # --args "<package> <bundle prefix> <gold fwd dir> [repeats]": end to end from the remote call
        import json as _json
        a = args.split()
        called = time.time()
        r = pro_cold.remote(a[0], a[1], a[2], int(a[3]) if len(a) > 3 else 1)
        first = r['runs'][0] if r['runs'] else {}
        tok = (first.get('rank0') or {})
        print(_json.dumps({'region': r['region'], 'cloud': r['cloud'], 'gpu_serial': r['host'][-1:] , 'container_start_s': round(r['entered'] - called, 2),
                           'first_run': {'spawn_to_first_token_s': first.get('first_token_s'), 'in_process_first_token_s': first.get('in_process_first_token_s'),
                                         'weights_on_device_s': sorted(first.get('weights_on_device_s') or []), 'next_token': tok.get('next_token'),
                                         'reference_next_token': tok.get('reference_next_token'), 'logits_equal_reference': tok.get('logits_equal_reference'),
                                         'sums_differing': sum(first.get('sums_differing') or [0]), 'errors': first.get('errors')},
                           'end_to_end_s': round(called and (r['entered'] - called) + (first.get('first_token_s') or 0), 2),
                           'later_runs_s': [x.get('in_process_first_token_s') for x in r['runs'][1:]]}))
        return
    if cmd == 'pro-session':  # start the kept 8xB300 container (modal run --detach); it exits when idle
        PRO_COMMANDS.clear()
        PRO_RESULTS.clear()
        print(pro_session.remote(int(args or 300)))
        return
    if cmd == 'pro-send':  # --args "<package> [readers] [repeats] [verify]" --extra "VAR=a|b ..."; `--args stop` ends the session
        import json as _json
        a = args.split()
        if a[0] == 'stop':
            PRO_COMMANDS.put({'stop': True})
            return
        ident = f'{time.time():.3f}'
        if a[0] == 'forward':  # forward <package> <bundle prefix> <gold fwd dir> [repeats]
            PRO_COMMANDS.put({'id': ident, 'kind': 'forward', 'package': a[1], 'bundles': a[2], 'gold': a[3],
                              'repeats': int(a[4]) if len(a) > 4 else 1, 'env': extra})
        else:
            PRO_COMMANDS.put({'id': ident, 'package': a[0], 'readers': int(a[1]) if len(a) > 1 else 16,
                              'repeats': int(a[2]) if len(a) > 2 else 1, 'verify': 'verify' in a[3:], 'env': extra})
        while True:
            r = PRO_RESULTS.get(timeout=3600)
            if r is None or r.get('id') == ident:
                break
            print('other result:', _json.dumps(r)[:300])
        print(_json.dumps(r, indent=1))
        return
    if cmd == 'load-multi':  # --args "<package> <manifest sha256> [plans]"
        import json as _json
        a = args.split()
        print(_json.dumps(load_multi.remote(a[0], a[1], a[2] if len(a) > 2 else '8x16'), indent=1))
        return
    if cmd == 'archive':  # --args name1,name2,...
        for line in archive_runs.remote(args):
            print(line)
        return
    if cmd == 'copy-v2':
        r = copy_package.remote()
        print(r['s'], r['steps'][0]['rc'], r['steps'][0]['out'][-500:])
        return
    if cmd == 'warm':  # --args N: N parallel host warmers
        for r in warm_volume.map(range(int(args or 8))):
            print(r)
        return
    # --cpus N: a run container with N CPUs (reads through the sandbox cost CPU; default 16).
    runner = run.with_options(cpu=cpus) if cpus else run
    steps = (check.remote(extra) if cmd == 'check' else cpu.remote(args) if cmd == 'cpu'
             else cpu.remote('', features) if cmd == 'build' else pyjob.remote(args) if cmd == 'py'
             else runner.remote(args, features=features, env=extra))
    for s in steps:
        if s['cmd'] == 'cat Cargo.lock':
            out = Path(os.environ.get('DSV4_LOCK_OUT', 'Cargo.lock.container'))
            out.write_text(s['out'])
            print(f'Cargo.lock from the container written to {out}')
            continue
        print(f"== {s['cmd']} (rc={s['rc']}, {s['s']}s)\n{s['out']}")
