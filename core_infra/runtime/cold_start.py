"""Cold-start measurement for the Modal examples' `cold_start` entry points.

One sample is one fresh single-use container, clocked from the first statement of its user code to
the first DiT step. With `environment(marker)`, the native process appends (pid, CLOCK_MONOTONIC ns)
to `marker` when its first DiT command starts on the GPU stream and then holds there, so the
container is stopped before denoising. Python's `time.monotonic_ns` reads the same clock on Linux.
"""
import statistics
import time
from pathlib import Path


def environment(marker):
    return {'NATIVE_FIRST_DIT_TRACE': '1', 'NATIVE_FIRST_DIT_NOTIFY_FILE': str(marker), 'NATIVE_FIRST_DIT_HOLD': '1'}


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


def sample(started_ns, stamp, error=''):
    return {'seconds': (stamp - started_ns) / 1e9 if stamp else None, 'error': '' if stamp else error[-2000:]}


def quantile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def collect(function, n, *args):
    """Call `function` (a Modal Function returning `sample` dicts) with `args` on `n` containers,
    printing each result as it lands and then the report."""
    samples = []
    for result in function.map(*[[value] * n for value in args], return_exceptions=True, order_outputs=False):
        samples.append(result if isinstance(result, dict) else {'seconds': None, 'error': repr(result)})
        seconds = samples[-1]['seconds']
        print(f'{len(samples):3d}/{n}  ' + (f'{seconds:.1f} s' if seconds is not None
                                            else 'failed: ' + samples[-1]['error'][-300:]), flush=True)
    print(report(samples))
    return samples


def report(samples):
    """One line: P50/P90 (linear interpolation), range and failures of `samples`."""
    seconds = [s['seconds'] for s in samples if isinstance(s, dict) and s.get('seconds') is not None]
    failed = len(samples) - len(seconds)
    if not seconds:
        return f'cold start  no sample reached the first DiT step ({failed} failed)'
    return (f'cold start  P50 {statistics.median(seconds):.1f} s, P90 {quantile(seconds, 0.9):.1f} s, '
            f'min {min(seconds):.1f} s, max {max(seconds):.1f} s ({len(seconds)} containers'
            + (f', {failed} failed)' if failed else ')'))
