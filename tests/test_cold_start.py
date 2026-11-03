"""The Modal examples' cold-start measurement (no GPU)."""
from pathlib import Path
import struct
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core_infra.runtime import cold_start  # noqa: E402


def test_wait_reads_the_native_first_dit_stamp_and_stops_with_the_run(tmp_path):
    marker = tmp_path / 'first-dit'
    threading.Timer(0.05, lambda: marker.write_bytes(struct.pack('<QQ', 1234, 98765))).start()
    assert cold_start.wait(marker, lambda: True, limit=5) == 98765
    assert cold_start.wait(tmp_path / 'never', lambda: False, limit=5) is None
    started = time.monotonic()
    assert cold_start.wait(tmp_path / 'never', lambda: True, limit=0.05) is None
    assert time.monotonic() - started < 1
    assert cold_start.environment(marker) == {'NATIVE_FIRST_DIT_TRACE': '1', 'NATIVE_FIRST_DIT_NOTIFY_FILE': str(marker),
                                              'NATIVE_FIRST_DIT_HOLD': '1'}


def test_samples_and_report():
    assert cold_start.sample(1_000_000_000, 8_500_000_000) == {'seconds': 7.5, 'error': ''}
    assert cold_start.sample(1, None, 'x' * 3000) == {'seconds': None, 'error': 'x' * 2000}
    samples = [{'seconds': s} for s in (6, 7, 8, 9, 10, 11, 12, 13, 14, 20)] + [{'seconds': None, 'error': 'e'}]
    assert abs(cold_start.quantile([s['seconds'] for s in samples[:10]], 0.9) - 14.6) < 1e-9
    assert cold_start.report(samples) == ('cold start  P50 10.5 s, P90 14.6 s, min 6.0 s, max 20.0 s '
                                          '(10 containers, 1 failed)')
    assert 'no sample' in cold_start.report([{'seconds': None}])


def test_collect_maps_one_call_per_container_and_keeps_failures(capsys):
    class Function:
        def map(self, *columns, return_exceptions, order_outputs):
            assert return_exceptions and not order_outputs and [len(c) for c in columns] == [3, 3]
            yield {'seconds': 7.0, 'error': ''}
            yield RuntimeError('container lost')
            yield {'seconds': 9.0, 'error': ''}
    samples = cold_start.collect(Function(), 3, 'config', {'prompt': 'x'})
    assert [s['seconds'] for s in samples] == [7.0, None, 9.0]
    out = capsys.readouterr().out
    assert 'failed: RuntimeError' in out and out.strip().endswith('(2 containers, 1 failed)')
