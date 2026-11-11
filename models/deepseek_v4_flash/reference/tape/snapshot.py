"""Print the snapshots of recorded launches (snapshot mode of cupti_tape.cpp).

    python snapshot.py TAPE --seq 3 [--as i32|i64|f32|bf16|u8] [--n 64]
    python snapshot.py TAPE --label fwd-001 --grep _page_table

For each launch: its parameters' snapshotted buffers, before ("in") and after ("out") the launch,
shown as the first N elements of the chosen type, plus whether the launch changed the buffer.
"""
import argparse
import json
import struct
from pathlib import Path


def decode(data, kind, n):
    size = {'i32': 4, 'i64': 8, 'f32': 4, 'bf16': 2, 'u8': 1}[kind]
    count = min(len(data) // size, n)
    if kind == 'bf16':
        vals = [struct.unpack('<f', b'\0\0' + data[2 * i:2 * i + 2])[0] for i in range(count)]
    else:
        fmt = {'i32': 'i', 'i64': 'q', 'f32': 'f', 'u8': 'B'}[kind]
        vals = list(struct.unpack('<%d%s' % (count, fmt), data[:count * size]))
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tape', type=Path)
    ap.add_argument('--seq', type=int, action='append')
    ap.add_argument('--label', default='')
    ap.add_argument('--grep', default='')
    ap.add_argument('--as', dest='kind', default='i32')
    ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--changed', action='store_true', help='only buffers the launch changed')
    args = ap.parse_args()
    snaps = {}
    for line in (args.tape / 'snapshots.jsonl').read_text().splitlines():
        s = json.loads(line)
        snaps.setdefault(s['snap'], []).append(s)
    pack = open(args.tape / 'snapshots.bin', 'rb')
    for line in (args.tape / 'launches.jsonl').read_text().splitlines():
        e = json.loads(line)
        if e.get('snap', -1) < 0 or e['snap'] not in snaps:
            continue
        if args.seq and e['seq'] not in args.seq:
            continue
        if args.label and not e['label'].startswith(args.label):
            continue
        if args.grep and args.grep not in (e.get('name') or ''):
            continue
        print(f"== seq {e['seq']} {e['label']} {e['name'][:120]} grid={e['grid']}")
        by = {}
        for s in snaps[e['snap']]:
            by.setdefault(s['param'], {})[s['phase']] = s
        for param, ph in sorted(by.items()):
            vals = {}
            for phase in ('in', 'out'):
                if phase in ph:
                    pack.seek(ph[phase]['offset'])
                    vals[phase] = pack.read(ph[phase]['bytes'])
            changed = vals.get('in') != vals.get('out')
            if args.changed and not changed:
                continue
            where = f"param {param // 1024} word {param % 1024}"
            for phase in ('in', 'out'):
                if phase in vals:
                    print(f"   [{where}] {phase:3s} ptr={ph[phase]['ptr']:#x} bytes={len(vals[phase])} "
                          f"{'CHANGED ' if changed and phase == 'out' else ''}{decode(vals[phase], args.kind, args.n)}")


if __name__ == '__main__':
    main()
