"""Summarize a CUPTI launch tape (cupti_tape.cpp output).

    python summarize.py TAPE/pidNNN [--label fwd-001] [--sequence]

Prints, per label, the number of launches and each distinct kernel (demangled when c++filt is
available) with its launch count, configurations and module; --sequence prints launches in order.
"""
import argparse
import collections
import json
import shutil
import subprocess
from pathlib import Path


def demangle(names):
    if not shutil.which('c++filt'):
        return {n: n for n in names}
    out = subprocess.run(['c++filt'], input='\n'.join(names), capture_output=True, text=True).stdout.splitlines()
    return dict(zip(names, out))


def load(root):
    launches = [json.loads(l) for l in (root / 'launches.jsonl').read_text().splitlines() if l.strip()]
    return launches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tape', type=Path)
    ap.add_argument('--label')
    ap.add_argument('--sequence', action='store_true')
    ap.add_argument('--width', type=int, default=150)
    args = ap.parse_args()
    launches = [l for l in load(args.tape) if 'name' in l and (args.label is None or l['label'] == args.label)]
    names = sorted({l['name'] for l in launches})
    pretty = demangle(names)
    if args.sequence:
        for l in launches:
            print(f"{l['seq']:6d} {l['label']:10s} g={l['grid']} b={l['block']} smem={l['smem']} "
                  f"mods={l['module_ids']} {pretty[l['name']][:args.width]}")
        return
    by_label = collections.defaultdict(list)
    for l in launches:
        by_label[l['label']].append(l)
    for label, items in by_label.items():
        print(f'== {label}: {len(items)} launches, {len({l["name"] for l in items})} kernels')
        counts = collections.Counter(l['name'] for l in items)
        first = {}
        for l in items:
            first.setdefault(l['name'], l)
        for name, n in sorted(counts.items(), key=lambda kv: first[kv[0]]['seq']):
            configs = {(tuple(l['grid']), tuple(l['block']), l['smem']) for l in items if l['name'] == name}
            print(f'  {n:5d}x mods={first[name]["module_ids"]} cfgs={len(configs)} {pretty[name][:args.width]}')


if __name__ == '__main__':
    main()
