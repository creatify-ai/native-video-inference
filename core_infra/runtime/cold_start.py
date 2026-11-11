"""Cold-start measurement for the Modal examples' `cold_start` entry points.

One sample is one fresh single-use container, clocked from the first statement of its user code to
the first DiT step. With `environment(marker)`, the native process appends (pid, CLOCK_MONOTONIC ns)
to `marker` when its first DiT command starts on the GPU stream and then holds there, so the
container is stopped before denoising. Python's `time.monotonic_ns` reads the same clock on Linux.

With a `trace` directory, the native processes also write their startup spans there
(`NATIVE_STARTUP_TRACE_LEVEL=io`, on the same clock), and `sample` returns them with the container's
start and first-DiT timestamps; `collect(..., save=DIR)` writes them under DIR, one folder a sample.
"""
import gzip
import json
import statistics
import time
from pathlib import Path


def environment(marker, trace=None):
    variables = {'NATIVE_FIRST_DIT_TRACE': '1', 'NATIVE_FIRST_DIT_NOTIFY_FILE': str(marker), 'NATIVE_FIRST_DIT_HOLD': '1'}
    if trace is not None:
        Path(trace).mkdir(parents=True, exist_ok=True)
        variables.update(NATIVE_STARTUP_TRACE_DIR=str(trace), NATIVE_STARTUP_TRACE_LEVEL='io')
    return variables


def parallel(n):
    """The number of containers to run at once, checked: 1 to 50."""
    if type(n) is not int or not 1 <= n <= 50:
        raise SystemExit('--parallel must be an integer from 1 to 50')
    return n


def wait(marker, alive, limit=1200.0):
    """The first DiT step's CLOCK_MONOTONIC ns, or None once `alive()` is false or `limit` s pass."""
    marker, deadline = Path(marker), time.monotonic() + limit
    while time.monotonic() < deadline:
        try:
            data = marker.read_bytes()
        except FileNotFoundError:
            data = b''
        if len(data) >= 16:
            return int.from_bytes(data[8:16], 'little')
        if not alive():
            return None
        time.sleep(0.002)
    return None


def sample(started_ns, stamp, error='', trace=None):
    result = {'seconds': (stamp - started_ns) / 1e9 if stamp else None, 'error': '' if stamp else error[-2000:]}
    if trace is not None:
        result['trace'] = {'started_ns': started_ns, 'first_dit_ns': stamp,
                           'files': {p.name: gzip.compress(p.read_bytes()) for p in sorted(Path(trace).glob('*.jsonl'))}}
    return result


def quantile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def collect(function, n, *args, save=None):
    """Call `function` (a Modal Function returning `sample` dicts) with `args` on `n` containers,
    printing each result as it lands and then the report. With `save`, write each sample's trace there."""
    samples = []
    for result in function.map(*[[value] * n for value in args], return_exceptions=True, order_outputs=False):
        samples.append(result if isinstance(result, dict) else {'seconds': None, 'error': repr(result)})
        if save is not None:
            write_trace(Path(save) / f'{len(samples):03d}', samples[-1])
        seconds = samples[-1]['seconds']
        print(f'{len(samples):3d}/{n}  ' + (f'{seconds:.1f} s' if seconds is not None
                                            else 'failed: ' + samples[-1]['error'][-300:]), flush=True)
    print(report(samples))
    return samples


def write_trace(directory, result):
    """`result`'s trace files as `*.jsonl`, next to `sample.json` with its timestamps and outcome."""
    directory.mkdir(parents=True, exist_ok=True)
    trace = result.pop('trace', None) or {}
    for name, data in trace.pop('files', {}).items():
        (directory / Path(name).name).write_bytes(gzip.decompress(data))
    (directory / 'sample.json').write_text(json.dumps({**result, **trace}, indent=1) + '\n')


def report(samples):
    """One line: P50/P90 (linear interpolation), range and failures of `samples`."""
    seconds = [s['seconds'] for s in samples if isinstance(s, dict) and s.get('seconds') is not None]
    failed = len(samples) - len(seconds)
    if not seconds:
        return f'cold start  no sample reached the first DiT step ({failed} failed)'
    return (f'cold start  P50 {statistics.median(seconds):.1f} s, P90 {quantile(seconds, 0.9):.1f} s, '
            f'min {min(seconds):.1f} s, max {max(seconds):.1f} s ({len(seconds)} containers'
            + (f', {failed} failed)' if failed else ')'))
