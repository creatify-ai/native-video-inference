"""SGLang reference for DeepSeek-V4-Flash NVFP4 on one B300 (TP=1).

    modal run models/deepseek_v4_flash/reference/sglang_b300.py            # serve, generate, report
    modal run models/deepseek_v4_flash/reference/sglang_b300.py --profile  # also a torch profiler trace

Runs `sglang.launch_server` from the pinned SGLang image against the checkpoint that checkpoint.py
downloaded, waits for it, sends the same greedy request REPEATS times and reports the server's
memory lines, the outputs and whether the repeats are identical (tokens and logprobs).
Logs and traces are written to the Volume under /dsv4/runs/<run id>/.
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
REPOSITORY = 'nvidia/DeepSeek-V4-Flash-NVFP4'
REVISION = 'e3cd60e7de98e9867116860d522499a728de1cf9'
MOUNT = '/dsv4'
CHECKPOINT = f"{MOUNT}/huggingface/hub/models--{REPOSITORY.replace('/', '--')}/snapshots/{REVISION}"
VOLUME_NAME = os.environ.get('DSV4_VOLUME', 'dsv4-flash-native')
PORT = 30000
REPEATS = 3
PROMPTS = [
    'The capital of France is',
    'Write a Python function that returns the n-th Fibonacci number.',
]

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App('dsv4-flash-sglang')
image = modal.Image.from_registry(SGLANG_IMAGE).env({'HF_HUB_OFFLINE': '1', 'PYTHONUNBUFFERED': '1'})
HERE = Path(__file__).resolve().parent
TOOLS = '/opt/dsv4-reference'  # gold/sitecustomize.py and tape/cupti_tape.cpp inside the container
if modal.is_local():
    image = (image.add_local_file(HERE / 'gold/sitecustomize.py', f'{TOOLS}/gold/sitecustomize.py')
             .add_local_file(HERE / 'tape/cupti_tape.cpp', f'{TOOLS}/tape/cupti_tape.cpp')
             .add_local_file(HERE / 'tape/cupti_prof.cpp', f'{TOOLS}/tape/cupti_prof.cpp')
             .add_local_file(HERE / 'tape/prof_summary.py', f'{TOOLS}/tape/prof_summary.py'))


def _post(path, payload, timeout=600):
    import urllib.request
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}{path}', data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body


def _wait(proc, log, deadline):
    import urllib.request
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f'server exited with {proc.returncode}:\n' + log.read_text()[-8000:])
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health_generate', timeout=30) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(10)
    raise TimeoutError('server did not become healthy:\n' + log.read_text()[-8000:])


@app.function(image=image, gpu='B300', cpu=16, memory=262144, timeout=3 * 3600, volumes={MOUNT: volume})
def serve(profile: bool = False, extra_args: str = '', env: str = '', gold: bool = False, tape: bool = False,
          gold_forwards: int = 0) -> dict:
    import shlex
    import subprocess
    run_id = time.strftime('%Y%m%d-%H%M%S')
    run_dir = Path(MOUNT) / 'runs' / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log = run_dir / 'server.log'
    environ = dict(os.environ, SGLANG_TORCH_PROFILER_DIR=str(run_dir / 'profile'))
    if gold:
        environ['DSV4_GOLD_DIR'] = str(run_dir / 'gold')
        environ['PYTHONPATH'] = f'{TOOLS}/gold:' + environ.get('PYTHONPATH', '')
        if tape:
            environ['CUDA_INJECTION64_PATH'] = _build_tape()
            environ['TAPE_DIR'] = str(run_dir / 'tape')
            (run_dir / 'tape').mkdir(exist_ok=True)
    for item in filter(None, env.split(',')):
        k, v = item.split('=', 1)
        environ[k] = v
    cmd = ['python3', '-m', 'sglang.launch_server', '--model-path', CHECKPOINT, '--tp', '1',
           '--trust-remote-code', '--host', '127.0.0.1', '--port', str(PORT),
           # Repeats of one prompt must recompute the prefill, not reuse the first one's KV.
           '--disable-radix-cache'] + shlex.split(extra_args)
    versions = subprocess.run(['python3', '-c', 'import sglang, torch, flashinfer; print(sglang.__version__, torch.__version__, torch.version.cuda, flashinfer.__version__)'],
                              capture_output=True, text=True)
    started = time.time()
    with open(log, 'w') as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=environ)
    try:
        _wait(proc, log, started + 2 * 3600)
        ready_s = time.time() - started
        smi = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv'],
                             capture_output=True, text=True).stdout
        results = []
        for prompt in PROMPTS:
            runs = []
            for repeat in range(REPEATS):
                if gold and gold_forwards and repeat == 0 and prompt == PROMPTS[0]:
                    (run_dir / 'gold').mkdir(exist_ok=True)
                    (run_dir / 'gold' / 'ARM').write_text(str(gold_forwards))
                out = _post('/generate', {'text': prompt, 'return_logprob': True, 'top_logprobs_num': 5,
                                          'sampling_params': {'temperature': 0, 'max_new_tokens': 64}})
                meta = out['meta_info']
                runs.append({'text': out['text'],
                             'output_token_logprobs': meta.get('output_token_logprobs'),
                             'input_token_logprobs': meta.get('input_token_logprobs')})
            same = all(json.dumps(r, sort_keys=True) == json.dumps(runs[0], sort_keys=True) for r in runs)
            results.append({'prompt': prompt, 'text': runs[0]['text'], 'repeats_identical': same,
                            'tokens': [t[1] for t in runs[0]['output_token_logprobs']]})
        census = None
        if profile:
            _post('/start_profile', {'num_steps': 6, 'activities': ['GPU', 'CPU'], 'record_shapes': True})
            _post('/generate', {'text': PROMPTS[0], 'sampling_params': {'temperature': 0, 'max_new_tokens': 4}})
            time.sleep(30)
            census = _kernel_census(run_dir / 'profile')
            (run_dir / 'kernel-census.json').write_text(json.dumps(census, indent=1))
        (run_dir / 'results.json').write_text(json.dumps(results, indent=1))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        volume.commit()
    text = log.read_text()
    keys = ('KV Cache', 'max_total_num_tokens', 'Load weight', 'avail mem', 'mem usage', 'memory pool',
            'moe_runner', 'attention_backend', 'cuda graph', 'Capture', 'WARNING', 'ERROR')
    lines = [l for l in text.splitlines() if any(k.lower() in l.lower() for k in keys)]
    return {'run_id': run_id, 'versions': versions.stdout.strip() or versions.stderr[-2000:], 'ready_s': ready_s,
            'nvidia_smi': smi, 'results': results, 'log_lines': lines[-200:],
            'census_top': census and census['kernels'][:80], 'census_files': census and census['files'],
            'server_args': [l for l in text.splitlines() if 'server_args=' in l.lower() or 'ServerArgs(' in l][:1]}


GOLD_PROMPTS = [
    'The capital of France is',
    'Write a Python function that returns the n-th Fibonacci number.',
    '请用三句话介绍一下长城。',
    'In 1905, Albert Einstein published four papers that',
]
GOLD_MODULES = ','.join(['model.embed_tokens', 'model.layers.*', 'model.norm', 'lm_head', 'logits_processor'] +
                        [f'model.layers.{i}.{m}' for i in (0, 2, 3, 42) for m in ('self_attn', 'mlp')])


def _start(cmd, environ, log):
    import subprocess
    with open(log, 'w') as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=environ)
    _wait(proc, log, time.time() + 2 * 3600)
    return proc


def _stop(proc):
    import subprocess
    proc.terminate()
    try:
        proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


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


@app.function(image=image, gpu='B300', cpu=16, memory=262144, timeout=4 * 3600, volumes={MOUNT: volume})
def gold_run(max_new_tokens: int = 128, forwards: int = 6, hash_params: bool = True, skip_default: bool = False,
             compare_with: str = '', extra_env: str = '') -> dict:
    """A: the default server (decode CUDA graphs). B: CUDA graphs off, gold hooks and the launch tape on.
    Both serve GOLD_PROMPTS greedily; equal tokens and logprobs make B's captures valid for A."""
    run_id = time.strftime('%Y%m%d-%H%M%S') + '-gold'
    run_dir = Path(MOUNT) / 'runs' / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    base = ['python3', '-m', 'sglang.launch_server', '--model-path', CHECKPOINT, '--tp', '1', '--trust-remote-code',
            '--host', '127.0.0.1', '--port', str(PORT), '--disable-radix-cache']
    report = {'run_id': run_id}
    tape_lib = _build_tape()  # fail before the first server if the tape does not build
    a = None
    if not skip_default:
        t = time.time()
        proc = _start(base, dict(os.environ), run_dir / 'server-a.log')
        report['a_ready_s'] = time.time() - t
        try:
            a = _generate_all(max_new_tokens)
        finally:
            _stop(proc)
        (run_dir / 'outputs-a.json').write_text(json.dumps(a))
        volume.commit()
    environ = dict(os.environ, DSV4_GOLD_DIR=str(run_dir / 'gold'), DSV4_GOLD_MODULES=GOLD_MODULES,
                   PYTHONPATH=f'{TOOLS}/gold:' + os.environ.get('PYTHONPATH', ''),
                   CUDA_INJECTION64_PATH=tape_lib, TAPE_DIR=str(run_dir / 'tape'),
                   # Modules load when their library loads, so the tape binds every kernel to its image.
                   CUDA_MODULE_LOADING='EAGER', TAPE_ALL_MODULES='1')
    if hash_params:
        environ['DSV4_GOLD_HASH_PARAMS'] = '1'
    if extra_env:
        environ.update(json.loads(extra_env))
    (run_dir / 'tape').mkdir(exist_ok=True)
    t = time.time()
    # Captures synchronize every launch; keep SGLang's watchdog from killing a slow captured forward.
    proc = _start(base + ['--disable-cuda-graph', '--watchdog-timeout', '7200'], environ, run_dir / 'server-b.log')
    report['b_ready_s'] = time.time() - t
    try:
        b = _generate_all(max_new_tokens, arm=run_dir / 'gold' / 'ARM', arm_count=forwards)
    finally:
        _stop(proc)
    (run_dir / 'outputs-b.json').write_text(json.dumps(b))
    volume.commit()
    report['b_texts'] = [o['text'][:300] for o in b]
    if a is None and compare_with:
        a = json.loads((Path(MOUNT) / 'runs' / compare_with / 'outputs-a.json').read_text())
        a = a[:len(b)] if max_new_tokens == len(a[0]['output_token_logprobs']) else None
    if a is not None:
        report['a_equals_b'] = [json.dumps(x, sort_keys=True) == json.dumps(y, sort_keys=True) for x, y in zip(a, b)]
        report['first_token_diff'] = []
        for x, y in zip(a, b):
            xa, yb = x['output_token_logprobs'], y['output_token_logprobs']
            diff = next((i for i, (p, q) in enumerate(zip(xa, yb)) if p != q), None)
            report['first_token_diff'].append(None if diff is None else {'index': diff, 'a': xa[diff], 'b': yb[diff]})
    gold = run_dir / 'gold'
    report['gold_files'] = sorted(str(p.relative_to(gold)) for p in gold.rglob('meta.json')) if gold.exists() else []
    report['gold_errors'] = [p.read_text()[:2000] for p in gold.glob('error-*')] if gold.exists() else []
    report['gold_root'] = sorted(p.name for p in gold.iterdir()) if gold.exists() else []
    tapes = sorted((run_dir / 'tape').glob('pid*'))
    report['tapes'] = [{'dir': str(p), 'launches': sum(1 for _ in open(p / 'launches.jsonl'))} for p in tapes]
    report['b_log_tail'] = (run_dir / 'server-b.log').read_text()[-3000:]
    return report


def _corpus_ids(tok, n):
    """Real text for long prompts: the SGLang package's Python sources in sorted order, tokenized
    (natural token statistics, so MoE routing and attention behave as on real inputs)."""
    import glob
    import sglang
    root = os.path.dirname(sglang.__file__)
    ids = []
    for p in sorted(glob.glob(f'{root}/**/*.py', recursive=True)):
        ids += tok.encode(Path(p).read_text(errors='ignore'), add_special_tokens=False)
        if len(ids) >= n:
            break
    return ids[:n]


@app.function(image=image, gpu='B300', cpu=16, memory=262144, timeout=3 * 3600, volumes={MOUNT: volume})
def bench_long(lengths: str = '16384,32768,65536,102400,131072', decode: int = 256, repeats: int = 2,
               extra_args: str = '', extra_env: str = '') -> dict:
    """Single-request long-context baseline on one B300: for each prompt length, the uncached
    prefill latency (TTFT, max_new_tokens=1) and the decode inter-token latency over `decode`
    tokens (ignore_eos). Radix cache off so no request reuses another's prefix."""
    run_id = time.strftime('%Y%m%d-%H%M%S') + '-long'
    run_dir = Path(MOUNT) / 'runs' / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    base = ['python3', '-m', 'sglang.launch_server', '--model-path', CHECKPOINT, '--tp', '1', '--trust-remote-code',
            '--host', '127.0.0.1', '--port', str(PORT), '--disable-radix-cache'] + extra_args.split()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
    ns = [int(x) for x in lengths.split(',')]
    corpus = _corpus_ids(tok, max(ns) + 64 * repeats)
    t = time.time()
    proc = _start(base, dict(os.environ, **(json.loads(extra_env) if extra_env else {})), run_dir / 'server.log')
    report = {'run_id': run_id, 'ready_s': round(time.time() - t, 1), 'results': [], 'extra_args': extra_args, 'extra_env': extra_env}
    try:
        _post('/generate', {'input_ids': corpus[:2048], 'sampling_params': {'temperature': 0, 'max_new_tokens': 8}}, timeout=3600)
        for n in ns:
            for r in range(repeats):
                ids = corpus[64 * r:64 * r + n]
                t = time.time()
                _post('/generate', {'input_ids': ids, 'sampling_params': {'temperature': 0, 'max_new_tokens': 1}}, timeout=3600)
                ttft = time.time() - t
                t = time.time()
                out = _post('/generate', {'input_ids': ids, 'sampling_params': {'temperature': 0, 'max_new_tokens': decode + 1,
                                                                                'ignore_eos': True}}, timeout=3600)
                total = time.time() - t
                row = {'tokens': n, 'repeat': r, 'ttft_s': round(ttft, 3), 'prefill_tok_per_s': round(n / ttft),
                       'decode_itl_ms': round((total - ttft) / decode * 1e3, 2),
                       'completion_tokens': out.get('meta_info', {}).get('completion_tokens') if isinstance(out, dict) else None}
                print(json.dumps(row), flush=True)
                report['results'].append(row)
    finally:
        _stop(proc)
    log = (run_dir / 'server.log').read_text()
    report['server_lines'] = [l for l in log.splitlines() if any(k in l for k in ('max_total_num_tokens', 'chunked_prefill_size', 'context_len', 'KV Cache is allocated'))][:8]
    (run_dir / 'report.json').write_text(json.dumps(report, indent=1))
    volume.commit()
    return report


def _long_requests(ids_list, decode, arm=None, arm_count=0):
    """Greedy requests over pre-tokenized prompts (no input logprobs, top-5 output logprobs)."""
    outs = []
    for i, ids in enumerate(ids_list):
        if arm is not None and i == 0 and arm_count:
            arm.parent.mkdir(parents=True, exist_ok=True)
            arm.write_text(str(arm_count))
        out = _post('/generate', {'input_ids': ids, 'return_logprob': True, 'top_logprobs_num': 5, 'logprob_start_len': -1,
                                  'sampling_params': {'temperature': 0, 'max_new_tokens': decode + 1, 'ignore_eos': True}},
                    timeout=7200)
        meta = out['meta_info']
        outs.append({'prompt_tokens': len(ids), 'output_token_logprobs': meta.get('output_token_logprobs'),
                     'output_top_logprobs': meta.get('output_top_logprobs')})
    return outs


@app.function(image=image, gpu='B300', cpu=16, memory=262144, timeout=6 * 3600, volumes={MOUNT: volume})
def long_gold_run(lengths: str = '16384,32768,65536,102400,131072', decode: int = 16, census: int = 4,
                  max_gold_bytes: int = 4 << 20, skip_default: bool = False, extra_args: str = '',
                  extra_env: str = '') -> dict:
    """Long-context reference: A = the default server (CUDA graphs), B = eager with the census tape
    and hash-only golds (tensors above max_gold_bytes keep only their SHA-256). Both serve the same
    pre-tokenized prompts (real text, _corpus_ids) greedily for decode + 1 tokens. Every forward of
    B is captured: ceil(n / 16384) prefill chunks plus decode steps per prompt, in prompt order."""
    from transformers import AutoTokenizer
    run_id = time.strftime('%Y%m%d-%H%M%S') + '-longgold'
    run_dir = Path(MOUNT) / 'runs' / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
    ns = [int(x) for x in lengths.split(',')]
    corpus = _corpus_ids(tok, max(ns))
    ids_list = [corpus[:n] for n in ns]
    (run_dir / 'prompts.json').write_text(json.dumps(ids_list))
    chunk = 16384
    forwards = sum(-(-n // chunk) + decode for n in ns)
    base = ['python3', '-m', 'sglang.launch_server', '--model-path', CHECKPOINT, '--tp', '1', '--trust-remote-code',
            '--host', '127.0.0.1', '--port', str(PORT), '--disable-radix-cache'] + extra_args.split()
    xenv = json.loads(extra_env) if extra_env else {}
    report = {'run_id': run_id, 'lengths': ns, 'forwards': forwards, 'extra_args': extra_args, 'extra_env': extra_env}
    tape_lib = _build_tape()
    if not skip_default:
        proc = _start(base, dict(os.environ, **xenv), run_dir / 'server-a.log')
        try:
            a = _long_requests(ids_list, decode)
        finally:
            _stop(proc)
        (run_dir / 'outputs-a.json').write_text(json.dumps(a))
        volume.commit()
    environ = dict(os.environ, DSV4_GOLD_DIR=str(run_dir / 'gold'), DSV4_GOLD_MODULES=GOLD_MODULES,
                   DSV4_GOLD_MAX_BYTES=str(max_gold_bytes),
                   PYTHONPATH=f'{TOOLS}/gold:' + os.environ.get('PYTHONPATH', ''),
                   CUDA_INJECTION64_PATH=tape_lib, TAPE_DIR=str(run_dir / 'tape'), TAPE_CENSUS=str(census),
                   CUDA_MODULE_LOADING='EAGER', TAPE_ALL_MODULES='1', **xenv)
    (run_dir / 'tape').mkdir(exist_ok=True)
    proc = _start(base + ['--disable-cuda-graph', '--watchdog-timeout', '14400'], environ, run_dir / 'server-b.log')
    try:
        b = _long_requests(ids_list, decode, arm=run_dir / 'gold' / 'ARM', arm_count=forwards)
    finally:
        _stop(proc)
    (run_dir / 'outputs-b.json').write_text(json.dumps(b))
    volume.commit()
    if not skip_default:
        report['a_equals_b'] = [json.dumps(x, sort_keys=True) == json.dumps(y, sort_keys=True) for x, y in zip(a, b)]
    gold = run_dir / 'gold'
    report['gold_forwards'] = sorted(p.name for p in gold.iterdir() if p.name.startswith('fwd-'))[-3:] if gold.exists() else []
    report['tapes'] = [{'dir': str(p), 'launches': sum(1 for _ in open(p / 'launches.jsonl'))} for p in sorted((run_dir / 'tape').glob('pid*'))]
    report['b_log_tail'] = (run_dir / 'server-b.log').read_text()[-2000:]
    return report


@app.function(image=image, gpu='B300', cpu=16, memory=262144, timeout=4 * 3600, volumes={MOUNT: volume})
def determinism_long(run: str = '20261109-023619-longgold', which: str = '0,2', repeats: int = 3,
                     configs: str = 'default|--disable-cuda-graph') -> dict:
    """Is long prefill deterministic? For each server config, the same pre-tokenized prompts are sent
    `repeats` times (8 tokens, top-5 logprobs); reports whether the repeats agree within a server
    and across servers."""
    ids_all = json.loads((Path(MOUNT) / 'runs' / run / 'prompts.json').read_text())
    ids_list = [ids_all[int(i)] for i in which.split(',')]
    base = ['python3', '-m', 'sglang.launch_server', '--model-path', CHECKPOINT, '--tp', '1', '--trust-remote-code',
            '--host', '127.0.0.1', '--port', str(PORT), '--disable-radix-cache']
    run_dir = Path(MOUNT) / 'runs' / (time.strftime('%Y%m%d-%H%M%S') + '-det')
    run_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for c, cfg in enumerate(configs.split('|')):
        proc = _start(base + ([] if cfg == 'default' else cfg.split()), dict(os.environ), run_dir / f'server-{c}.log')
        try:
            reps = [_long_requests(ids_list, 7) for _ in range(repeats)]
        finally:
            _stop(proc)
        out[cfg] = reps
    report = {}
    for cfg, reps in out.items():
        report[cfg] = [all(json.dumps(r[i]) == json.dumps(reps[0][i]) for r in reps) for i in range(len(ids_list))]
        report[cfg + ' first logprobs'] = [[r[i]['output_token_logprobs'][0][0] for r in reps] for i in range(len(ids_list))]
    cfgs = list(out)
    if len(cfgs) == 2:
        report['across'] = [json.dumps(out[cfgs[0]][0][i]) == json.dumps(out[cfgs[1]][0][i]) for i in range(len(ids_list))]
    return report


@app.function(image=image, gpu='B300', cpu=16, memory=262144, timeout=4 * 3600, volumes={MOUNT: volume})
def nondet_probe(run: str = '20261109-023619-longgold', which: int = 0, repeats: int = 3, extra_env: str = '',
                 extra_args: str = '') -> dict:
    """Localize run-to-run differences in long prefill: the same prompt's prefill forward captured
    `repeats` times (hash-only golds of every layer's input, self_attn, mlp and output); reports the
    first tensors whose SHA-256 differs between repeats."""
    import re
    ids = json.loads((Path(MOUNT) / 'runs' / run / 'prompts.json').read_text())[which]
    run_dir = Path(MOUNT) / 'runs' / (time.strftime('%Y%m%d-%H%M%S') + '-nondet')
    modules = ','.join(['model.embed_tokens', 'model.layers.*', 'model.norm', 'lm_head', 'logits_processor'] +
                       [f'model.layers.{i}.{m}' for i in range(43) for m in ('self_attn', 'mlp')])
    environ = dict(os.environ, DSV4_GOLD_DIR=str(run_dir / 'gold'), DSV4_GOLD_MODULES=modules, DSV4_GOLD_MAX_BYTES='1',
                   PYTHONPATH=f'{TOOLS}/gold:' + os.environ.get('PYTHONPATH', ''))
    if extra_env:
        environ.update(json.loads(extra_env))
    base = ['python3', '-m', 'sglang.launch_server', '--model-path', CHECKPOINT, '--tp', '1', '--trust-remote-code',
            '--host', '127.0.0.1', '--port', str(PORT), '--disable-radix-cache', '--disable-cuda-graph',
            '--watchdog-timeout', '14400'] + extra_args.split()
    run_dir.mkdir(parents=True, exist_ok=True)
    proc = _start(base, environ, run_dir / 'server.log')
    try:
        (run_dir / 'gold').mkdir(parents=True, exist_ok=True)
        (run_dir / 'gold' / 'ARM').write_text(str(repeats))
        firsts = []
        for _ in range(repeats):
            out = _post('/generate', {'input_ids': ids, 'return_logprob': True, 'logprob_start_len': -1,
                                      'sampling_params': {'temperature': 0, 'max_new_tokens': 1}}, timeout=7200)
            firsts.append(out['meta_info']['output_token_logprobs'][0])
    finally:
        _stop(proc)
    volume.commit()
    gold = run_dir / 'gold'
    fwds = sorted(p for p in gold.iterdir() if p.name.startswith('fwd-'))
    hashes = [{f.name[:-5]: j['sha256'] for f in d.glob('*.json') for j in [json.loads(f.read_text())]
               if isinstance(j, dict) and 'sha256' in j} for d in fwds]
    def order(name):
        m = re.match(r'model\.layers\.(\d+)\.(.*)', name)
        if not m:
            return (-1 if 'embed' in name else 99, name)
        stage = {'in.hidden_states': 0, 'self_attn': 1, 'mlp': 3, 'out': 4}
        rest = m.group(2)
        key = next((v for k, v in stage.items() if rest.startswith(k)), 2)
        return (int(m.group(1)), key, rest)
    names = sorted(set(hashes[0]) & set.intersection(*[set(h) for h in hashes]), key=order)
    differing = [n for n in names if len({h[n] for h in hashes}) > 1]
    return {'run': run_dir.name, 'first_token_logprobs': firsts, 'forwards': [d.name for d in fwds],
            'tensors': len(names), 'first_differing': differing[:12], 'n_differing': len(differing)}


def _build_prof() -> str:
    """Compile the CUPTI kernel-activity profiler (reference/tape/cupti_prof.cpp) in this image."""
    import glob
    import subprocess
    out = f'{TOOLS}/tape/libcupti_prof.so'
    if os.path.exists(out):
        return out
    cupti = sorted(glob.glob('/opt/sglang/lib/python3.12/site-packages/nvidia/cu13/include/cupti.h') +
                   glob.glob('/usr/local/cuda/extras/CUPTI/include/cupti.h') + glob.glob('/usr/local/cuda/include/cupti.h'))
    inc = os.path.dirname(cupti[0])
    lib = os.path.join(os.path.dirname(inc), 'lib')
    libs = glob.glob(f'{lib}/libcupti.so*') or glob.glob('/usr/local/cuda/extras/CUPTI/lib64/libcupti.so*')
    cmd = ['g++', '-std=c++17', '-O2', '-shared', '-fPIC', f'{TOOLS}/tape/cupti_prof.cpp', f'-I{inc}', '-I/usr/local/cuda/include',
           '-o', out, libs[0], f'-Wl,-rpath,{os.path.dirname(libs[0])}']
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError('profiler build failed: ' + r.stderr[-4000:])
    return out


@app.function(image=image, gpu='B300', cpu=16, memory=262144, timeout=3 * 3600, volumes={MOUNT: volume})
def prof_decode(extra_args: str = '', extra_env: str = '', new_tokens: int = 96) -> dict:
    """Kernel-by-kernel device time of SGLang's decode CUDA graph (bs 1) on a short prompt."""
    run_dir = Path(MOUNT) / 'runs' / (time.strftime('%Y%m%d-%H%M%S') + '-prof')
    (run_dir / 'prof').mkdir(parents=True, exist_ok=True)
    environ = dict(os.environ, CUDA_INJECTION64_PATH=_build_prof(), PROF_DIR=str(run_dir / 'prof'),
                   **(json.loads(extra_env) if extra_env else {}))
    base = ['python3', '-m', 'sglang.launch_server', '--model-path', CHECKPOINT, '--tp', '1', '--trust-remote-code',
            '--host', '127.0.0.1', '--port', str(PORT), '--disable-radix-cache'] + extra_args.split()
    proc = _start(base, environ, run_dir / 'server.log')
    try:
        _post('/generate', {'text': GOLD_PROMPTS[0], 'sampling_params': {'temperature': 0, 'max_new_tokens': new_tokens, 'ignore_eos': True}})
        time.sleep(5)
    finally:
        _stop(proc)
    time.sleep(2)
    volume.commit()
    return _prof_summary(str(run_dir / 'prof'))


def _concurrent_requests(prompts, decode, arm=None, arm_count=0):
    """All prompts sent at once (greedy, ignore_eos, top-5 output logprobs, no input logprobs);
    responses in prompt order."""
    import concurrent.futures
    if arm is not None and arm_count:
        arm.parent.mkdir(parents=True, exist_ok=True)
        arm.write_text(str(arm_count))
    def one(ids):
        out = _post('/generate', {'input_ids': ids, 'return_logprob': True, 'top_logprobs_num': 5, 'logprob_start_len': -1,
                                  'sampling_params': {'temperature': 0, 'max_new_tokens': decode + 1, 'ignore_eos': True}}, timeout=4 * 3600)
        meta = out['meta_info']
        return {'prompt_tokens': len(ids), 'output_ids': out.get('output_ids'), 'output_token_logprobs': meta.get('output_token_logprobs'),
                'output_top_logprobs': meta.get('output_top_logprobs')}
    with concurrent.futures.ThreadPoolExecutor(len(prompts)) as ex:
        return list(ex.map(one, prompts))


@app.function(image=image, gpu='B300', cpu=16, memory=262144, timeout=6 * 3600, volumes={MOUNT: volume})
def concurrent_gold_run(concurrency: int = 32, length: int = 16384, decode: int = 32, census: int = 4,
                        max_gold_bytes: int = 4 << 20, extra_args: str = '', extra_env: str = '') -> dict:
    """Concurrent-serving reference: `concurrency` prompts of `length` tokens (the same real-text
    prompts as bench_concurrent) sent at once, decode + 1 new tokens each. A = the default server
    (CUDA graphs); B = eager with the census tape and hash-only golds of every forward (the prefill
    chunks, then the batched decode steps)."""
    from transformers import AutoTokenizer
    run_id = time.strftime('%Y%m%d-%H%M%S') + f'-concgold{concurrency}x{length}'
    run_dir = Path(MOUNT) / 'runs' / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
    corpus = _corpus_ids(tok, concurrency * 997 + length)
    prompts = [corpus[997 * i:997 * i + length] for i in range(concurrency)]
    (run_dir / 'prompts.json').write_text(json.dumps(prompts))
    # Every prefill chunk and decode step. Past the SWA pool (about 78 prompts of 16k), SGLang
    # interleaves: it decodes (freeing out-of-window SWA pages) between waves of prefills, so a few
    # more decode forwards run than `decode`; the margin covers them (an armed count left over
    # captures nothing once the requests are done).
    forwards = concurrency * -(-length // 16384) + decode + 16
    if '--speculative' in extra_args:
        # MTP: each prefill chunk runs the target and the draft; each decode round 2 draft decode
        # steps, the target verify and the draft extend, and takes at least one token.
        forwards = 2 * concurrency * -(-length // 16384) + 4 * decode + 16
    base = ['python3', '-m', 'sglang.launch_server', '--model-path', CHECKPOINT, '--tp', '1', '--trust-remote-code',
            '--host', '127.0.0.1', '--port', str(PORT), '--disable-radix-cache'] + extra_args.split()
    xenv = json.loads(extra_env) if extra_env else {}
    report = {'run_id': run_id, 'forwards': forwards, 'extra_args': extra_args, 'extra_env': extra_env}
    proc = _start(base, dict(os.environ, **xenv), run_dir / 'server-a.log')
    try:
        a = _concurrent_requests(prompts, decode)
    finally:
        _stop(proc)
    (run_dir / 'outputs-a.json').write_text(json.dumps(a))
    volume.commit()
    environ = dict(os.environ, DSV4_GOLD_DIR=str(run_dir / 'gold'), DSV4_GOLD_MODULES=GOLD_MODULES,
                   DSV4_GOLD_MAX_BYTES=str(max_gold_bytes),
                   PYTHONPATH=f'{TOOLS}/gold:' + os.environ.get('PYTHONPATH', ''),
                   CUDA_INJECTION64_PATH=_build_tape(), TAPE_DIR=str(run_dir / 'tape'), TAPE_CENSUS=str(census),
                   CUDA_MODULE_LOADING='EAGER', TAPE_ALL_MODULES='1')
    environ.update(xenv)  # e.g. DSV4_GOLD_MODULES=logits_processor: logits and samples only (concurrency checks)
    (run_dir / 'tape').mkdir(exist_ok=True)
    proc = _start(base + ['--disable-cuda-graph', '--watchdog-timeout', '21600'], environ, run_dir / 'server-b.log')
    try:
        b = _concurrent_requests(prompts, decode, arm=run_dir / 'gold' / 'ARM', arm_count=forwards)
    finally:
        _stop(proc)
    (run_dir / 'outputs-b.json').write_text(json.dumps(b))
    volume.commit()
    report['a_equals_b'] = [json.dumps(x, sort_keys=True) == json.dumps(y, sort_keys=True) for x, y in zip(a, b)]
    report['a_tokens_equal_b'] = [x['output_ids'] == y['output_ids'] for x, y in zip(a, b)]
    gold = run_dir / 'gold'
    report['gold_forwards'] = len([p for p in gold.iterdir() if p.name.startswith('fwd-')]) if gold.exists() else 0
    report['tapes'] = [{'dir': str(p), 'launches': sum(1 for _ in open(p / 'launches.jsonl'))} for p in sorted((run_dir / 'tape').glob('pid*'))]
    report['b_batches'] = [l for l in (run_dir / 'server-b.log').read_text().splitlines() if 'batch' in l][:60]
    return report


@app.function(image=image, gpu='B300', cpu=16, memory=262144, timeout=4 * 3600, volumes={MOUNT: volume})
def outputs_run(concurrency: int = 1, length: int = 16384, decode: int = 1024, extra_args: str = '', extra_env: str = '') -> dict:
    """Reference tokens only (no golds, no tape) for long decodes: the concurrent_gold_run prompts,
    decode + 1 new tokens each, from the default server (A, CUDA graphs) and an eager one (B)."""
    from transformers import AutoTokenizer
    run_id = time.strftime('%Y%m%d-%H%M%S') + f'-outputs{concurrency}x{length}d{decode}'
    run_dir = Path(MOUNT) / 'runs' / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
    corpus = _corpus_ids(tok, concurrency * 997 + length)
    prompts = [corpus[997 * i:997 * i + length] for i in range(concurrency)]
    (run_dir / 'prompts.json').write_text(json.dumps(prompts))
    base = ['python3', '-m', 'sglang.launch_server', '--model-path', CHECKPOINT, '--tp', '1', '--trust-remote-code',
            '--host', '127.0.0.1', '--port', str(PORT), '--disable-radix-cache'] + extra_args.split()
    env = dict(os.environ, **(json.loads(extra_env) if extra_env else {}))
    outs = {}
    for arm, extra in (('a', []), ('b', ['--disable-cuda-graph'])):
        proc = _start(base + extra, env, run_dir / f'server-{arm}.log')
        try:
            outs[arm] = _concurrent_requests(prompts, decode)
        finally:
            _stop(proc)
        (run_dir / f'outputs-{arm}.json').write_text(json.dumps(outs[arm]))
        volume.commit()
    accept = [l for l in (run_dir / 'server-a.log').read_text().splitlines() if 'accept len' in l][-3:]
    return {'run_id': run_id, 'a_tokens_equal_b': [x['output_ids'] == y['output_ids'] for x, y in zip(outs['a'], outs['b'])],
            'first_diff': [next((i for i, (u, v) in enumerate(zip(x['output_ids'], y['output_ids'])) if u != v), None) for x, y in zip(outs['a'], outs['b'])],
            'accept_log': accept}


@app.function(image=image, gpu='B300', cpu=16, memory=262144, timeout=4 * 3600, volumes={MOUNT: volume})
def bench_concurrent(configs: str = '32x16384,48x16384', decode: int = 256, extra_args: str = '', extra_env: str = '') -> dict:
    """Concurrent serving baseline on one B300: for each CxL config, C greedy requests (distinct
    real-text prompts of L tokens, ignore_eos, decode + 1 new tokens, no logprobs) sent at once.
    Reports the wall time from the first send to the last response, output and total token
    throughput, per-request latency, and the server's decode-batch lines (running requests and
    generation throughput per logged batch), from which the batched decode step time follows."""
    import concurrent.futures
    import re
    from transformers import AutoTokenizer
    run_dir = Path(MOUNT) / 'runs' / (time.strftime('%Y%m%d-%H%M%S') + '-concurrent')
    run_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
    cfgs = [tuple(int(v) for v in c.split('x')) for c in configs.split(',')]
    corpus = _corpus_ids(tok, max(c * 997 + l for c, l in cfgs))
    base = ['python3', '-m', 'sglang.launch_server', '--model-path', CHECKPOINT, '--tp', '1', '--trust-remote-code',
            '--host', '127.0.0.1', '--port', str(PORT), '--disable-radix-cache'] + extra_args.split()
    proc = _start(base, dict(os.environ, **(json.loads(extra_env) if extra_env else {})), run_dir / 'server.log')
    report = {'run_id': run_dir.name, 'decode': decode, 'results': []}
    try:
        _post('/generate', {'input_ids': corpus[:2048], 'sampling_params': {'temperature': 0, 'max_new_tokens': 8}}, timeout=3600)
        for c, l in cfgs:
            prompts = [corpus[997 * i:997 * i + l] for i in range(c)]
            (run_dir / f'prompts-{c}x{l}.json').write_text(json.dumps(prompts))
            log_start = (run_dir / 'server.log').stat().st_size
            def one(ids):
                t = time.time()
                out = _post('/generate', {'input_ids': ids, 'sampling_params': {'temperature': 0, 'max_new_tokens': decode + 1, 'ignore_eos': True}},
                            timeout=4 * 3600)
                return time.time() - t, out
            t0 = time.time()
            with concurrent.futures.ThreadPoolExecutor(c) as ex:
                res = list(ex.map(one, prompts))
            wall = time.time() - t0
            outs = [{'output_ids': r[1].get('output_ids'), 'latency_s': round(r[0], 3)} for r in res]
            (run_dir / f'outputs-{c}x{l}.json').write_text(json.dumps(outs))
            log = (run_dir / 'server.log').read_text()[log_start:]
            dec = [(int(m.group(1)), float(m.group(2))) for m in re.finditer(r'Decode batch.*?#running-req: (\d+).*?gen throughput \(token/s\): ([\d.]+)', log)]
            full = [g for n, g in dec if n == c]
            row = {'concurrency': c, 'prompt_tokens': l, 'wall_s': round(wall, 2),
                   'output_tok_per_s': round(c * (decode + 1) / wall, 1), 'total_tok_per_s': round(c * (l + decode + 1) / wall, 1),
                   'latency_s_mean': round(sum(r[0] for r in res) / c, 2), 'latency_s_max': round(max(r[0] for r in res), 2),
                   'decode_batches_logged': len(dec), 'full_batch_gen_tok_per_s_median': sorted(full)[len(full) // 2] if full else None}
            if full:
                row['full_batch_step_ms'] = round(1e3 * c / row['full_batch_gen_tok_per_s_median'], 2)
            # Speculative decoding (MTP): accepted tokens per step, from the decode lines.
            acc = sorted(float(m.group(1)) for m in re.finditer(r'Decode batch.*?accept len: ([\d.]+)', log))
            if acc:
                row['accept_len_median'] = acc[len(acc) // 2]
            print(json.dumps(row), flush=True)
            report['results'].append(row)
    finally:
        _stop(proc)
    log = (run_dir / 'server.log').read_text()
    report['server_lines'] = [l[:600] for l in log.splitlines() if any(k in l for k in ('max_total_num_tokens', 'chunked_prefill_size', 'cuda_graph_bs', 'Capture cuda graph bs', 'max_running_requests', 'speculative', 'Capture draft'))][:12]
    volume.commit()
    return report


@app.function(image=image, gpu='B300', cpu=16, memory=262144, timeout=3600, volumes={MOUNT: volume})
def prof_long(length: int = 16384, new_tokens: int = 64, extra_args: str = '', extra_env: str = '') -> dict:
    """Kernel activity of SGLang serving one pre-tokenized long prompt (the long-gold corpus) twice:
    the second request's prefill burst and the decode graph replays (tape/prof_summary.py)."""
    import sys
    from transformers import AutoTokenizer
    sys.path.insert(0, f'{TOOLS}/tape')
    import prof_summary
    run_dir = Path(MOUNT) / 'runs' / (time.strftime('%Y%m%d-%H%M%S') + f'-proflong{length}')
    (run_dir / 'prof').mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
    ids = _corpus_ids(tok, length)[:length]
    environ = dict(os.environ, CUDA_INJECTION64_PATH=_build_prof(), PROF_DIR=str(run_dir / 'prof'),
                   **(json.loads(extra_env) if extra_env else {}))
    base = ['python3', '-m', 'sglang.launch_server', '--model-path', CHECKPOINT, '--tp', '1', '--trust-remote-code',
            '--host', '127.0.0.1', '--port', str(PORT), '--disable-radix-cache'] + extra_args.split()
    proc = _start(base, environ, run_dir / 'server.log')
    timings = []
    try:
        for _ in range(2):
            t = time.time()
            _post('/generate', {'input_ids': ids, 'sampling_params': {'temperature': 0, 'max_new_tokens': new_tokens, 'ignore_eos': True}},
                  timeout=3600)
            timings.append(round(time.time() - t, 3))
            time.sleep(1)
        time.sleep(5)
    finally:
        _stop(proc)
    time.sleep(2)
    volume.commit()
    return {'request_s': timings, **prof_summary.summarize(str(run_dir / 'prof'))}


@app.function(image=image, gpu='B300', cpu=8, memory=65536, timeout=3600, volumes={MOUNT: volume})
def export_tables() -> dict:
    """The RoPE tables SGLang computes at load (no checkpoint tensor holds them), exported with the
    reference's own code: freqs_cis (CPU torch.polar) and the rotary embedding's cos_sin_cache (GPU)
    for the uncompressed and the compressed layers. Written to /dsv4/tables/<name>.bin."""
    import hashlib
    import torch
    from sglang.kernels.ops.attention.dsv4.rope import precompute_freqs_cis
    from sglang.srt.layers.rotary_embedding import get_rope_wrapper
    cfg = json.loads(Path(CHECKPOINT, 'config.json').read_text())
    scaling = cfg['rope_scaling']
    out_dir = Path(MOUNT) / 'tables'
    out_dir.mkdir(exist_ok=True)
    report = {}
    for kind, compressed in (('swa', False), ('compressed', True)):
        base = cfg['compress_rope_theta'] if compressed else cfg['rope_theta']
        kwargs = dict(dim=cfg['qk_rope_head_dim'], seqlen=cfg['max_position_embeddings'],
                      original_seq_len=scaling['original_max_position_embeddings'] if compressed else 0, base=base,
                      factor=scaling.get('factor', 1.0), beta_fast=scaling.get('beta_fast', 32), beta_slow=scaling.get('beta_slow', 1))
        freqs = precompute_freqs_cis(**kwargs)
        active = None
        if compressed:
            active = dict(scaling)
            active['rope_type'] = 'deepseek_yarn'
        rope = get_rope_wrapper(head_size=cfg['qk_rope_head_dim'], rotary_dim=cfg['qk_rope_head_dim'],
                                max_position=cfg['max_position_embeddings'], base=base, rope_scaling=active,
                                is_neox_style=False, device=torch.device('cuda'))
        for name, t in ((f'freqs_cis_{kind}', freqs), (f'cos_sin_cache_{kind}', rope.cos_sin_cache)):
            data = t.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()
            (out_dir / f'{name}.bin').write_bytes(data)
            report[name] = {'dtype': str(t.dtype), 'shape': list(t.shape), 'sha256': hashlib.sha256(data).hexdigest()}
    (out_dir / 'tables.json').write_text(json.dumps(report, indent=1))
    volume.commit()
    import subprocess
    report['cpu'] = subprocess.run(['bash', '-c', 'lscpu | grep -E "Model name|Flags" | cut -c1-200'], capture_output=True, text=True).stdout
    return report


@app.function(image=image, cpu=4, memory=16384, timeout=1800, volumes={MOUNT: volume})
def export_sources() -> dict:
    """Host-code sources the native port compiles against, copied from the reference image: the
    trtllm-gen batched-GEMM export headers (shipped with flashinfer-cubin, not in the sdist) and the
    installed flashinfer's own include/csrc trees. Written under /dsv4/deps/."""
    import glob
    import shutil
    import flashinfer
    import flashinfer_cubin
    out = Path(MOUNT) / 'deps'
    out.mkdir(exist_ok=True)
    report = {}
    for inc in glob.glob(os.path.join(os.path.dirname(flashinfer_cubin.__file__), 'cubins', '*', 'batched_gemm-*', 'include')):
        dst = out / 'trtllm_gen' / Path(inc).parent.name / 'include'
        shutil.copytree(inc, dst, dirs_exist_ok=True)
        report[str(dst)] = sum(1 for _ in dst.rglob('*'))
    fi_root = Path(flashinfer.__file__).parent
    for sub in ('data/include', 'data/csrc', 'include', 'csrc'):
        src = fi_root / sub
        if src.exists():
            dst = out / 'flashinfer' / sub.replace('data/', '')
            shutil.copytree(src, dst, dirs_exist_ok=True)
            report[str(dst)] = sum(1 for _ in dst.rglob('*'))
    # The Python package itself (fused_moe/core.py etc.): how SGLang's calls map onto the C++ entry points.
    dst = out / 'flashinfer' / 'python'
    shutil.copytree(fi_root, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns('data', '*.so', '__pycache__', 'cubins'))
    report[str(dst)] = sum(1 for _ in dst.rglob('*.py'))
    volume.commit()
    return report


@app.function(image=image, gpu='B300', cpu=8, memory=65536, timeout=3600, volumes={MOUNT: volume})
def export_moe_build() -> dict:
    """flashinfer's JIT build of its trtllm-gen MoE module (fused_moe_trtllm_sm100), as the reference
    would compile it on this GPU: build.ninja (sources + exact flags), the generated include root
    (export-header tree + arch-filtered flashinferMetaInfo.h) and the vendored include trees it uses.
    The native port compiles the same host sources (minus the TVM-FFI launcher) with these flags.
    Written under /dsv4/deps/moe_build/."""
    import shutil
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.fused_moe import gen_trtllm_gen_fused_moe_sm100_module
    spec = gen_trtllm_gen_fused_moe_sm100_module()
    spec.write_ninja()
    out = Path(MOUNT) / 'deps' / 'moe_build'
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copy(spec.ninja_path, out / 'build.ninja')
    gen_root = jit_env.FLASHINFER_GEN_SRC_DIR / 'trtllm_export' / 'fused_moe_trtllm_sm100'
    shutil.copytree(gen_root, out / 'gen_root', symlinks=False)
    data = Path(jit_env.FLASHINFER_DATA)
    for sub in ('cutlass/include', 'cutlass/tools/util/include', 'spdlog/include', 'cccl'):
        if (data / sub).exists():
            shutil.copytree(data / sub, out / 'data' / sub, dirs_exist_ok=True)
    import tvm_ffi  # headers only: the routing sources include tvm_ffi_utils.h
    shutil.copytree(Path(tvm_ffi.__file__).parent / 'include', out / 'tvm_ffi' / 'include', dirs_exist_ok=True)
    report = {'ninja': (out / 'build.ninja').read_text(), 'gen_root': str(gen_root),
              'cubin_dir': str(jit_env.FLASHINFER_CUBIN_DIR), 'data': str(data),
              'files': sum(1 for _ in out.rglob('*'))}
    volume.commit()
    return report


@app.function(image=image, cpu=4, memory=16384, timeout=1800, volumes={MOUNT: volume})
def run_py(code: str) -> str:
    return _run_py(code)


@app.function(image=image, gpu='B300', cpu=8, memory=65536, timeout=1800, volumes={MOUNT: volume})
def run_py_gpu(code: str) -> str:
    return _run_py(code)


def _run_py(code: str) -> str:
    """Run a Python snippet in the reference image (inspection helper); returns its stdout."""
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(code, {'__name__': '__dsv4__'})
    volume.commit()
    return buf.getvalue()


def _kernel_census(profile_dir: Path) -> dict:
    """GPU kernels of the torch profiler traces: name, launch count, total time, distinct grid/block."""
    import gzip
    files = sorted(str(p) for p in profile_dir.rglob('*.json*'))
    kernels = {}
    for path in files:
        opener = gzip.open if path.endswith('.gz') else open
        with opener(path, 'rt') as fh:
            trace = json.load(fh)
        for ev in trace.get('traceEvents', []):
            if ev.get('cat') != 'kernel':
                continue
            args = ev.get('args', {})
            k = kernels.setdefault(ev['name'], {'name': ev['name'], 'count': 0, 'us': 0.0, 'configs': set()})
            k['count'] += 1
            k['us'] += float(ev.get('dur', 0))
            k['configs'].add(json.dumps([args.get('grid'), args.get('block'), args.get('shared memory')]))
    out = sorted(kernels.values(), key=lambda k: -k['us'])
    for k in out:
        k['configs'] = sorted(k['configs'])[:6]
        k['us'] = round(k['us'], 1)
    return {'files': files, 'kernels': out}


def _build_tape() -> str:
    """Compile the CUPTI tape against the image's CUDA headers and CUPTI wheel; returns the .so path."""
    import glob
    import subprocess
    out = f'{TOOLS}/tape/libcupti_tape.so'
    if os.path.exists(out):
        return out
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


@app.function(image=image, cpu=4, memory=16384, timeout=1800, volumes={MOUNT: volume})
def versions() -> str:
    """CPU only: the image's package versions and whether the checkpoint is present."""
    import subprocess
    script = ('import sglang, torch, flashinfer, importlib.metadata as m; '
              'print("sglang", sglang.__version__, "torch", torch.__version__, "cuda", torch.version.cuda, '
              '"flashinfer", flashinfer.__version__); '
              'print([(d.metadata["Name"], d.version) for d in m.distributions() if any(k in d.metadata["Name"].lower() '
              'for k in ("flashinfer", "sgl", "deep", "flash", "triton", "cutlass", "nvidia-cu", "tilelang", "cuda"))])')
    out = subprocess.run(['python3', '-c', script], capture_output=True, text=True)
    present = Path(CHECKPOINT, 'config.json').exists()
    return out.stdout + out.stderr[-3000:] + f'\ncheckpoint config present: {present}'



def _prof_summary(prof_dir: str, top: int = 40) -> dict:
    """Decode-graph kernels from cupti_prof records: per replay of the most replayed graph, the device
    span and the kernels (name, mean duration), so two runs can be compared kernel by kernel."""
    import collections
    import glob
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
def main(profile: bool = False, extra_args: str = '', env: str = '', only_versions: bool = False,
         gold: bool = False, tape: bool = False, gold_forwards: int = 0, gold_run_: bool = False,
         max_new_tokens: int = 128, forwards: int = 6, skip_default: bool = False, compare_with: str = '',
         hash_params: bool = False, extra_env: str = '', export_tables_: bool = False, export_sources_: bool = False,
         export_moe_build_: bool = False, py: str = '', bench_long_: str = '', long_gold_: str = '', det_long_: str = '', nondet_: str = '', prof_decode_: bool = False,
         prof_long_: str = '', concurrent_: str = '', concurrent_gold_: str = '', outputs_: str = ''):
    if prof_decode_:
        print(json.dumps(prof_decode.remote(extra_args, extra_env), indent=1))
        return
    if prof_long_:
        print(json.dumps(prof_long.remote(int(prof_long_), 64, extra_args, extra_env), indent=1))
        return
    if concurrent_:
        print(json.dumps(bench_concurrent.remote(concurrent_, 256, extra_args, extra_env), indent=1))
        return
    if outputs_:  # CxLxD
        c, l, d = (int(v) for v in outputs_.split('x'))
        print(json.dumps(outputs_run.remote(c, l, d, extra_args, extra_env), indent=1))
        return
    if concurrent_gold_:  # CxL (32 decode tokens) or CxLxD
        c, l, *d = (int(v) for v in concurrent_gold_.split('x'))
        print(json.dumps(concurrent_gold_run.remote(c, l, d[0] if d else 32, 4, 4 << 20, extra_args, extra_env), indent=1))
        return
    if nondet_:
        print(json.dumps(nondet_probe.remote(which=int(nondet_), extra_env=extra_env, extra_args=extra_args), indent=1))
        return
    if det_long_:  # prompt indices; --extra-args selects the server configs (separated by |)
        print(json.dumps(determinism_long.remote(which=det_long_, configs=extra_args or 'default|--disable-cuda-graph'), indent=1))
        return
    if long_gold_:
        print(json.dumps(long_gold_run.remote(long_gold_, extra_args=extra_args, extra_env=extra_env), indent=1))
        return
    if bench_long_:  # comma-separated prompt lengths, e.g. 16384,32768,65536,102400,131072
        print(json.dumps(bench_long.remote(bench_long_, extra_args=extra_args, extra_env=extra_env), indent=1))
        return
    if py:
        code = Path(py).read_text() if Path(py).exists() else py
        print((run_py_gpu if 'cuda' in code else run_py).remote(code))
        return
    if export_moe_build_:
        r = export_moe_build.remote()
        print(r.pop('ninja'))
        print(json.dumps(r, indent=1))
        return
    if export_sources_:
        print(json.dumps(export_sources.remote(), indent=1))
        return
    if export_tables_:
        print(json.dumps(export_tables.remote(), indent=1))
        return
    if gold_run_:
        print(json.dumps(gold_run.remote(max_new_tokens, forwards, hash_params, skip_default, compare_with, extra_env), indent=1,
                         ensure_ascii=False))
        return
    if only_versions:
        print(versions.remote())
        return
    out = serve.remote(profile, extra_args, env, gold, tape, gold_forwards)
    print(json.dumps(out, indent=1, ensure_ascii=False))
