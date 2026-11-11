"""Summaries of cupti_prof.cpp kernel records (one JSON line per kernel), shared by the SGLang and
native profiling runs so the two can be compared kernel by kernel.

- decode: the most replayed CUDA graph; per replay its device span (first start to last end),
  busy time (sum of kernel durations; above the span when streams overlap) and the mean time per
  kernel name.
- prefill: bursts of non-graph kernels separated by more than `burst_gap_ms` of idle device; for
  the largest bursts their span, busy time, idle time inside the span (no kernel running: host
  bubbles) and the time per kernel name.
"""
import collections
import glob
import json


def _union_ms(rows):
    """Total time covered by at least one kernel (ms)."""
    total, cur_s, cur_e = 0, None, None
    for r in sorted(rows, key=lambda r: r['start']):
        if cur_e is None or r['start'] > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = r['start'], r['end']
        else:
            cur_e = max(cur_e, r['end'])
    if cur_e is not None:
        total += cur_e - cur_s
    return total / 1e6


def _by_name(rows, n=1, top=30):
    acc = collections.defaultdict(float)
    for r in rows:
        acc[r['name'][:100]] += (r['end'] - r['start']) / 1e3 / n
    return [(k, round(v, 1)) for k, v in sorted(acc.items(), key=lambda kv: -kv[1])[:top]]


def summarize(prof_dir, burst_gap_ms=20.0, top=30):
    rows = []
    for f in glob.glob(f'{prof_dir}/prof-*.jsonl'):
        rows.extend(json.loads(line) for line in open(f) if line.strip())
    if not rows:
        return {'error': 'no kernel records'}
    out = {}
    graphs = collections.Counter(r['graph'] for r in rows if r['graph'])
    if graphs:
        gid = graphs.most_common(1)[0][0]
        ks = sorted((r for r in rows if r['graph'] == gid), key=lambda r: r['start'])
        replays, cur = [], [ks[0]]
        for prev, r in zip(ks, ks[1:]):
            if r['start'] - prev['end'] > 100_000:  # 100 us idle between replays
                replays.append(cur)
                cur = []
            cur.append(r)
        replays.append(cur)
        size = collections.Counter(len(r) for r in replays).most_common(1)[0][0]
        full = [r for r in replays if len(r) == size][2:]
        spans = sorted((max(k['end'] for k in r) - r[0]['start']) / 1e6 for r in full)
        busy = sorted(sum(k['end'] - k['start'] for k in r) / 1e6 for r in full)
        out['decode'] = {'kernels_per_replay': size, 'replays': len(full),
                         'span_ms_median': round(spans[len(spans) // 2], 3) if spans else None,
                         'busy_ms_median': round(busy[len(busy) // 2], 3) if busy else None,
                         'streams': len({k['stream'] for r in full for k in r}),
                         'top_us_per_replay': _by_name([k for r in full for k in r], max(len(full), 1), top)}
    eager = sorted((r for r in rows if not r['graph']), key=lambda r: r['start'])
    bursts, cur = [], []
    for r in eager:
        if cur and r['start'] - max(k['end'] for k in cur[-64:]) > burst_gap_ms * 1e6:
            bursts.append(cur)
            cur = []
        cur.append(r)
    if cur:
        bursts.append(cur)
    big = sorted(bursts, key=len, reverse=True)[:4]
    out['prefill'] = []
    for b in sorted(big, key=lambda b: b[0]['start']):
        span = (max(k['end'] for k in b) - b[0]['start']) / 1e6
        covered = _union_ms(b)
        # The largest idle gaps (no kernel running) and the kernels around them.
        gaps, last_end, last_name = [], b[0]['end'], b[0]['name']
        for k in b[1:]:
            if k['start'] > last_end:
                gaps.append(((k['start'] - last_end) / 1e3, last_name[:70], k['name'][:70]))
            if k['end'] > last_end:
                last_end, last_name = k['end'], k['name']
        gaps.sort(reverse=True)
        out['prefill'].append({'kernels': len(b), 'start_s': round(b[0]['start'] / 1e9, 3), 'span_ms': round(span, 2),
                               'busy_ms': round(sum(k['end'] - k['start'] for k in b) / 1e6, 2),
                               'idle_ms': round(span - covered, 2), 'top_us': _by_name(b, 1, top),
                               'gaps_us': [(round(g, 1), a, c) for g, a, c in gaps[:25]],
                               'gaps_over_20us': sum(1 for g in gaps if g[0] > 20), 'gap_hist_us': [sum(1 for g in gaps if lo < g[0] <= hi) for lo, hi in ((0, 5), (5, 20), (20, 100), (100, 1e9))]})
    return out
