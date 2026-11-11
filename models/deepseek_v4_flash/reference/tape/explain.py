"""Explain the parameters of recorded launches: pointers resolved to parameters/allocations, TMA
descriptors resolved to their encode arguments, scalars shown as integers and floats.

    python explain.py TAPE PARAMS.json --label fwd-001 [--grep NAME] [--first N] [--seq S]
"""
import argparse
import bisect
import json
import struct
import subprocess
from pathlib import Path


def load_ranges(params_path, tape):
    ranges = []
    for r in json.loads(Path(params_path).read_text()):
        if r['bytes']:
            ranges.append((r['device_ptr'], r['device_ptr'] + r['bytes'], 'param:' + r['name']))
    allocs = []
    for i, line in enumerate((tape / 'allocations.jsonl').read_text().splitlines()):
        a = json.loads(line)
        allocs.append((a['ptr'], a['ptr'] + a['bytes'], f"alloc#{i}({a['bytes']})"))
    ranges.sort()
    allocs.sort()
    return ranges, allocs


_STARTS = {}


def lookup(v, ranges):
    starts = _STARTS.get(id(ranges))
    if starts is None:
        starts = _STARTS[id(ranges)] = [r[0] for r in ranges]
    i = bisect.bisect_right(starts, v) - 1
    if i >= 0 and ranges[i][0] <= v < ranges[i][1]:
        return f'{ranges[i][2]}+{v - ranges[i][0]}'
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tape', type=Path)
    ap.add_argument('params')
    ap.add_argument('--label', default='')
    ap.add_argument('--grep', default='')
    ap.add_argument('--first', type=int, default=1000000)
    ap.add_argument('--seq', type=int, default=-1)
    args = ap.parse_args()
    ranges, allocs = load_ranges(args.params, args.tape)
    maps = {}
    for line in (args.tape / 'tensormaps.jsonl').read_text().splitlines():
        m = json.loads(line)
        maps[m['map']] = m
    events = [json.loads(l) for l in (args.tape / 'launches.jsonl').read_text().splitlines()]
    sel = [e for e in events if (not args.label or e['label'].startswith(args.label)) and
           (args.seq < 0 or e['seq'] == args.seq) and (not args.grep or args.grep in (e.get('name') or ''))]
    names = sorted({e['name'] for e in sel if e.get('name')})
    dm = dict(zip(names, subprocess.run(['c++filt'], input='\n'.join(names), capture_output=True, text=True).stdout.splitlines()))
    for e in sel[:args.first]:
        if e.get('op'):
            print(f"{e['seq']} {e['label']} {e['op']} kind={e.get('kind')} dst={lookup(e.get('dst') or 0, ranges) or lookup(e.get('dst') or 0, allocs)} "
                  f"src={lookup(e.get('src') or 0, ranges) or lookup(e.get('src') or 0, allocs) or e.get('src')} bytes={e.get('bytes') or e.get('count')} data={(e.get('data') or '')[:96]}")
            continue
        print(f"{e['seq']} {e['label']} grid={e['grid']} block={e['block']} smem={e['smem']} attrs={[(a['id'], a['hex'][:16]) for a in e['attrs']]}")
        print(f"   {dm.get(e['name'], e['name'])[:400]}")
        for i, p in enumerate(e['params']):
            raw = bytes.fromhex(p['hex'])
            desc = ''
            if p['size'] == 8:
                v = struct.unpack('<Q', raw)[0]
                where = lookup(v, ranges) or lookup(v, allocs)
                desc = f'ptr {where}' if where else f'u64 {v} / i64 {struct.unpack("<q", raw)[0]} / f64 {struct.unpack("<d", raw)[0]:.6g}'
            elif p['size'] == 4:
                desc = f'i32 {struct.unpack("<i", raw)[0]} / f32 {struct.unpack("<f", raw)[0]:.6g}'
            elif p['size'] == 128 and p['hex'] in maps:
                m = maps[p['hex']]
                where = lookup(m['address'], ranges) or lookup(m['address'], allocs)
                desc = (f"tensormap dtype={m['dtype']} dims={m['dims']} strides={m['strides']} box={m['box']} "
                        f"estr={m['element_strides']} swz={m['swizzle']} il={m['interleave']} l2={m['l2']} oob={m['oob']} addr={where}")
            else:
                words = [struct.unpack('<Q', raw[j:j + 8])[0] for j in range(0, len(raw) - len(raw) % 8, 8)]
                notes = []
                for j, w in enumerate(words):
                    where = lookup(w, ranges) or lookup(w, allocs)
                    if where:
                        notes.append(f'@{8 * j}:{where}')
                desc = f'{p["size"]}B hex={p["hex"][:200]} ' + ' '.join(notes)
            print(f'   [{i}] off={p["offset"]} {desc}')


if __name__ == '__main__':
    main()
