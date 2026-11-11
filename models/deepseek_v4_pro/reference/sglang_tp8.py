"""SGLang reference for DeepSeek-V4-Pro NVFP4 on eight B300s (TP=8).

    modal run models/deepseek_v4_pro/reference/sglang_tp8.py                 # A: default server
    modal run models/deepseek_v4_pro/reference/sglang_tp8.py --gold          # then B: eager, gold hooks

A serves GOLD_PROMPTS greedily REPEATS times with the default server (decode CUDA graphs) and
reports the time from launch to healthy, the memory lines and whether the repeats are identical.
B restarts it with CUDA graphs off and the Flash port's gold hooks (../../deepseek_v4_flash/
reference/gold): every rank's loaded parameters with SHA-256 (params-pid<pid>.json) and the first
`forwards` forwards' logits. Equal outputs make B's captures valid for A.
Logs and captures go to the Volume under /dsv4pro/runs/<run id>/.
"""
import json
import os
import time
from pathlib import Path

# The 2025.06 builder mounts Modal's client instead of pip-installing its requirements into the
# image's Python (the 2023.12 builder downgrades the SGLang image's pydantic to 1.x).
os.environ.setdefault('MODAL_IMAGE_BUILDER_VERSION', '2025.06')
import modal  # noqa: E402

SGLANG_IMAGE = 'lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f'  # v0.5.23-cu130
REVISION = '1449d1e641023406daf6b432361486c768aad740'  # nvidia/DeepSeek-V4-Pro-NVFP4
MOUNT = '/dsv4pro'
CHECKPOINT = f'{MOUNT}/checkpoint/{REVISION}'
GPUS = 8
PORT = 30000
REPEATS = 2
GOLD_PROMPTS = [
    'The capital of France is',
    'Write a Python function that returns the n-th Fibonacci number.',
    '请用三句话介绍一下长城。',
    'In 1905, Albert Einstein published four papers that',
]

volume = modal.Volume.from_name('dsv4-pro-native', version=2)
app = modal.App('dsv4-pro-sglang')
image = modal.Image.from_registry(SGLANG_IMAGE).env({'HF_HUB_OFFLINE': '1', 'PYTHONUNBUFFERED': '1'})
TOOLS = '/opt/dsv4-reference'
if modal.is_local():
    REPO = Path(__file__).resolve().parents[3]
    image = (image.add_local_file(REPO / 'models/deepseek_v4_flash/reference/gold/sitecustomize.py', f'{TOOLS}/gold/sitecustomize.py')
             .add_local_file(REPO / 'models/deepseek_v4_flash/reference/tape/cupti_tape.cpp', f'{TOOLS}/tape/cupti_tape.cpp')
             .add_local_file(REPO / 'core_infra/runtime/checkpoint.py', f'{TOOLS}/checkpoint.py'))


def _post(path, payload, timeout=1800):
    import urllib.request
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}{path}', data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _start(cmd, environ, log, limit_s=3 * 3600):
    import subprocess
    import urllib.request
    with open(log, 'w') as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=environ)
    deadline = time.time() + limit_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f'server exited with {proc.returncode}:\n' + log.read_text()[-8000:])
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health_generate', timeout=30) as r:
                if r.status == 200:
                    return proc
        except Exception:
            pass
        time.sleep(5)
    proc.kill()
    raise TimeoutError('server did not become healthy:\n' + log.read_text()[-8000:])


def _stop(proc):
    import subprocess
    proc.terminate()
    try:
        proc.wait(timeout=180)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    subprocess.run(['pkill', '-9', '-f', 'sglang'], check=False)  # TP ranks are child processes
    time.sleep(10)


def _generate_all(max_new_tokens, arm=None, arm_count=0):
    outs = []
    for i, prompt in enumerate(GOLD_PROMPTS):
        if arm is not None and i == 0 and arm_count:
            arm.parent.mkdir(parents=True, exist_ok=True)
            arm.write_text(str(arm_count))
        out = _post('/generate', {'text': prompt, 'return_logprob': True, 'top_logprobs_num': 5, 'logprob_start_len': 0,
                                  'sampling_params': {'temperature': 0, 'max_new_tokens': max_new_tokens}})
        meta = out['meta_info']
        outs.append({'prompt': prompt, 'text': out['text'], 'output_token_logprobs': meta.get('output_token_logprobs'),
                     'input_token_logprobs': meta.get('input_token_logprobs'),
                     'output_top_logprobs': meta.get('output_top_logprobs')})
    return outs


def _same(a, b):
    return [json.dumps(x, sort_keys=True) == json.dumps(y, sort_keys=True) for x, y in zip(a, b)]


def _log_lines(log):
    keys = ('KV Cache', 'max_total_num_tokens', 'Load weight', 'avail mem', 'mem usage', 'memory pool',
            'moe_runner', 'attention_backend', 'cuda graph', 'Capture', 'WARNING', 'ERROR', 'tp_size', 'quant')
    return [l[:400] for l in log.read_text().splitlines() if any(k.lower() in l.lower() for k in keys)][-120:]


def _finish_package(pkg, run_id):
    """Pin each rank's shards with a native-checkpoint-v1 manifest and copy the text side."""
    import shutil
    import subprocess
    for f in ('config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json'):
        shutil.copyfile(Path(CHECKPOINT) / f, pkg / f)
    ranks = sorted(p for p in pkg.glob('rank*') if (p / 'params.json').exists())
    procs = [subprocess.Popen(['python3', f'{TOOLS}/checkpoint.py', 'manifest', '--root', str(r / 'weights'), '--file',
                               '*.safetensors', '--out', str(r / 'manifest.json'), '--workers', '8'],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for r in ranks]
    out = {'run': run_id, 'ranks': {}}
    for r, p in zip(ranks, procs):
        o, e = p.communicate()
        meta = json.loads((r / 'params.json').read_text())
        out['ranks'][r.name] = {'manifest': json.loads(o) if p.returncode == 0 else e[-2000:], 'gpu_id': meta['gpu_id'],
                                'stored_bytes': meta['stored_bytes'], 'files': len(meta['files']), 'params': len(meta['params'])}
    (pkg / 'package.json').write_text(json.dumps(out, indent=1))
    return out


@app.function(image=image, gpu=f'B300:{GPUS}', cpu=64, memory=524288, timeout=8 * 3600, volumes={MOUNT: volume})
def reference(max_new_tokens: int = 128, gold: bool = False, forwards: int = 6, extra_args: str = '',
              skip_default: bool = False, package: str = '') -> dict:
    import shlex
    import subprocess
    run_id = time.strftime('%Y%m%d-%H%M%S') + '-tp8'
    run_dir = Path(MOUNT) / 'runs' / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    base = ['python3', '-m', 'sglang.launch_server', '--model-path', CHECKPOINT, '--tp', str(GPUS),
            '--trust-remote-code', '--host', '127.0.0.1', '--port', str(PORT),
            # Repeats of one prompt must recompute the prefill, not reuse the first one's KV.
            '--disable-radix-cache'] + shlex.split(extra_args)
    report = {'run_id': run_id, 'region': os.environ.get('MODAL_REGION')}
    a = None
    if not skip_default:
        t = time.time()
        proc = _start(base, dict(os.environ), run_dir / 'server-a.log')
        report['a_ready_s'] = round(time.time() - t, 1)
        try:
            report['nvidia_smi'] = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader'],
                capture_output=True, text=True).stdout.splitlines()
            runs = [_generate_all(max_new_tokens) for _ in range(REPEATS)]
        finally:
            _stop(proc)
        a = runs[0]
        (run_dir / 'outputs-a.json').write_text(json.dumps(a))
        report['a_repeats_identical'] = [_same(a, r) for r in runs[1:]]
        report['a_texts'] = [o['text'][:300] for o in a]
        report['a_log'] = _log_lines(run_dir / 'server-a.log')
    if gold:
        environ = dict(os.environ, DSV4_GOLD_DIR=str(run_dir / 'gold'), DSV4_GOLD_MODULES='logits_processor',
                       PYTHONPATH=f'{TOOLS}/gold:' + os.environ.get('PYTHONPATH', ''))
        # `package`: every rank writes its loaded weights as a prepared package (and their SHA-256);
        # otherwise the params dump only hashes them.
        pkg = Path(MOUNT) / 'prepared' / package if package else None
        if pkg:
            environ['DSV4_GOLD_PACKAGE_DIR'] = str(pkg)
        else:
            environ['DSV4_GOLD_HASH_PARAMS'] = '1'
        t = time.time()
        proc = _start(base + ['--disable-cuda-graph', '--watchdog-timeout', '7200'], environ, run_dir / 'server-b.log',
                      limit_s=6 * 3600)
        report['b_ready_s'] = round(time.time() - t, 1)
        try:
            b = _generate_all(max_new_tokens, arm=run_dir / 'gold' / 'ARM', arm_count=forwards)
        finally:
            _stop(proc)
        (run_dir / 'outputs-b.json').write_text(json.dumps(b))
        report['b_texts'] = [o['text'][:300] for o in b]
        if a is not None:
            report['a_equals_b'] = _same(a, b)
        g = run_dir / 'gold'
        report['gold_root'] = sorted(p.name for p in g.iterdir())[:200] if g.exists() else []
        report['gold_errors'] = [p.read_text()[:2000] for p in g.glob('error-*')] if g.exists() else []
        report['b_log'] = _log_lines(run_dir / 'server-b.log')
        if pkg:
            report['package'] = _finish_package(pkg, run_id)
    (run_dir / 'report.json').write_text(json.dumps(report, indent=1))
    return report


def _build_tape() -> str:
    """Compile the CUPTI tape against the image's CUDA headers and CUPTI wheel; returns the .so path."""
    import glob
    import subprocess
    out = f'{TOOLS}/tape/libcupti_tape.so'
    cupti = sorted(glob.glob('/opt/sglang/lib/python3.12/site-packages/nvidia/cu13/include/cupti.h') +
                   glob.glob('/usr/local/cuda/extras/CUPTI/include/cupti.h') + glob.glob('/usr/local/cuda/include/cupti.h'))
    if not cupti:
        raise RuntimeError('no CUPTI headers in the image')
    inc = os.path.dirname(cupti[0])
    lib = os.path.join(os.path.dirname(inc), 'lib')
    libs = glob.glob(f'{lib}/libcupti.so*') or glob.glob('/usr/local/cuda/extras/CUPTI/lib64/libcupti.so*')
    cmd = ['g++', '-std=c++17', '-O2', '-shared', '-fPIC', '-fvisibility=hidden', f'{TOOLS}/tape/cupti_tape.cpp',
           f'-I{inc}', '-I/usr/local/cuda/include', '-o', out, libs[0], '-L/usr/local/cuda/lib64/stubs', '-lcuda', '-ldl',
           f'-Wl,-rpath,{os.path.dirname(libs[0])}']  # no cudart: the tape resolves the process's own
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError('tape build failed: ' + ' '.join(cmd) + '\n' + r.stderr[-6000:])
    return out


LAYERS = 61
CAPTURE_MODULES = ','.join(['model.embed_tokens', 'model.layers.*', 'model.norm', 'logits_processor'] +
                           [f'model.layers.{i}.{m}' for i in range(LAYERS) for m in ('self_attn', 'mlp')])


@app.function(image=image, gpu=f'B300:{GPUS}', cpu=64, memory=524288, timeout=4 * 3600, volumes={MOUNT: volume})
def capture(max_new_tokens: int = 2, extra_args: str = '', extra_env: str = '') -> dict:
    """The eager server with every rank's golds and launch tape: for each of GOLD_PROMPTS, the
    prefill and `max_new_tokens - 1` decode forwards. Per rank (gold/rank<r>/fwd-<n>/): batch
    inputs, each layer's output, each layer's attention and MLP outputs (this rank's partials),
    every all-reduce's input and output (ar-<k>), the logits; tape/pid<pid>/ holds the launches
    and kernel images. Outputs go to outputs-b.json."""
    import shlex
    run_id = time.strftime('%Y%m%d-%H%M%S') + '-tp8cap'
    run_dir = Path(MOUNT) / 'runs' / run_id
    (run_dir / 'tape').mkdir(parents=True, exist_ok=True)
    base = ['python3', '-m', 'sglang.launch_server', '--model-path', CHECKPOINT, '--tp', str(GPUS),
            '--trust-remote-code', '--host', '127.0.0.1', '--port', str(PORT), '--disable-radix-cache',
            '--disable-cuda-graph', '--watchdog-timeout', '7200'] + shlex.split(extra_args)
    environ = dict(os.environ, DSV4_GOLD_DIR=str(run_dir / 'gold'), DSV4_GOLD_MODULES=CAPTURE_MODULES,
                   DSV4_GOLD_PER_RANK='1', PYTHONPATH=f'{TOOLS}/gold:' + os.environ.get('PYTHONPATH', ''),
                   CUDA_INJECTION64_PATH=_build_tape(), TAPE_DIR=str(run_dir / 'tape'),
                   # Modules load when their library loads, so the tape binds every kernel to its image.
                   CUDA_MODULE_LOADING='EAGER', TAPE_ALL_MODULES='1')
    if extra_env:
        environ.update(json.loads(extra_env))
    report = {'run_id': run_id, 'region': os.environ.get('MODAL_REGION')}
    t = time.time()
    proc = _start(base, environ, run_dir / 'server-b.log', limit_s=3 * 3600)
    report['ready_s'] = round(time.time() - t, 1)
    try:
        b = _generate_all(max_new_tokens, arm=run_dir / 'gold' / 'ARM', arm_count=len(GOLD_PROMPTS) * max_new_tokens)
    finally:
        _stop(proc)
    (run_dir / 'outputs-b.json').write_text(json.dumps(b))
    report['texts'] = [o['text'][:100] for o in b]
    report['tokens'] = [[t[1] for t in o['output_token_logprobs']] for o in b]
    g = run_dir / 'gold'
    report['gold_root'] = sorted(p.name for p in g.iterdir())[:60] if g.exists() else []
    report['gold_errors'] = [p.read_text()[:2000] for p in g.glob('error-*')] if g.exists() else []
    report['rank0'] = {d.name: len(list(d.iterdir())) for d in sorted((g / 'rank0').glob('fwd-*'))} if (g / 'rank0').exists() else {}
    report['tapes'] = [{'dir': p.name, 'launches': sum(1 for _ in open(p / 'launches.jsonl'))}
                       for p in sorted((run_dir / 'tape').glob('pid*')) if (p / 'launches.jsonl').exists()]
    report['log'] = _log_lines(run_dir / 'server-b.log')[-40:]
    (run_dir / 'report.json').write_text(json.dumps(report, indent=1))
    return report


@app.function(image=image, cpu=64, memory=65536, timeout=4 * 3600, volumes={MOUNT: volume})
def finish_package(package: str, run_id: str = '') -> dict:
    """CPU: manifests for a package whose ranks were dumped (no GPU needed for the hashing)."""
    return _finish_package(Path(MOUNT) / 'prepared' / package, run_id)


@app.function(image=image, cpu=16, memory=65536, timeout=3600, volumes={MOUNT: volume})
def run_py(code: str) -> str:
    """CPU: run a Python snippet in the reference image (torch available) against the Volume."""
    import contextlib
    import io
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(code, {'__name__': 'snippet'})
    return out.getvalue()


@app.local_entrypoint()
def main(max_new_tokens: int = 128, gold: bool = False, forwards: int = 6, extra_args: str = '',
         skip_default: bool = False, package: str = '', finish: str = '', capture_golds: bool = False, py: str = ''):
    if capture_golds:  # --capture-golds: the eager server with per-rank golds and tapes
        print(json.dumps(capture.remote(max_new_tokens if max_new_tokens != 128 else 2, extra_args), indent=1))
        return
    if py:  # --py <file>: a Python snippet on CPUs in the reference image
        print(run_py.remote(Path(py).read_text()))
        return
    if finish:  # --finish <package>: only the manifests, on CPUs
        print(json.dumps(finish_package.remote(finish), indent=1))
        return
    print(json.dumps(reference.remote(max_new_tokens, gold, forwards, extra_args, skip_default, package), indent=1))
