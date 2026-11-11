"""Export the kernel bundle of a launch tape: every module image a recorded launch used.

    python bundle.py TAPE/pidNNN OUT [--labels fwd-001,fwd-002]

OUT/manifest.json maps each launched kernel name to its variants: the distinct CUBINs
(OUT/cubin/<sha>.cubin) launched under that name. One name can have several variants: TileLang and
Triton bake compile-time constants (TileLang's n_splits, Triton's specializations of integer and
pointer arguments) into differently-compiled images of the same symbol. Each variant records the
declared parameter sizes, launch examples, and derived `tags` that tell the native port which
variant a launch needs:

  * mhc_pre_big_fuse_with_norm_tilelang_kernel: n_splits (the split count of the GEMM partials it
    reduces: the template argument of the preceding DeepGEMM prenorm GEMM, or the split_k argument of
    the preceding fused post/pre kernel)
  * mhc_fused_post_pre_fma_tilelang_kernel: tile (mix outputs per CTA, from grid.y = ceil(24 / tile))
  * Triton kernels (two trailing scratch pointers): spec, one letter per argument before them:
    p/P = pointer not/16-byte aligned, 1 = integer equal to 1, D = integer divisible by 16, i = other
  * _dequantize_k_cache_paged_kernel: cache (swa / c4 / c128, the pool it reads, by allocation size)
  * mhc_post_tilelang_kernel: hb (num_h_blocks = grid.y)
  * _combine_topk_swa_indices_kernel: width (its WIDTH constexpr, the combined row stride) and steps
    (its SEARCH_STEPS constexpr, bit_length of the request count)
  * every launch under a layer label: kind (swa / c4 / c128, the layer's compress ratio; meta for
    the forward's metadata), which separates variants that differ only in constexprs (page sizes)
  `other_tags` lists further tag sets the same CUBIN was launched with.

`cubin`, `param_sizes` and `example` at the top level describe the most launched variant (schema v1).
Kernels whose launch carried no module binding (images loaded through the CUDA runtime's own fatbin
registration) are resolved by searching every written module image for the symbol; names found in no
image are listed under "unresolved" (the native port supplies those kernels itself).
"""
import argparse
import bisect
import hashlib
import json
import os
import re
import struct
from pathlib import Path


# DSV4_MODEL=pro: DeepSeek-V4-Pro's layer kinds and pool sizes (default: Flash on one GPU).
PRO = os.environ.get('DSV4_MODEL') == 'pro'


def param_bytes(e, i):
    return bytes.fromhex(e['params'][i]['hex'])


class Allocations:
    def __init__(self, tape):
        rows = []
        p = tape / 'allocations.jsonl'
        if p.exists():
            for line in p.read_text().splitlines():
                a = json.loads(line)
                rows.append((a['ptr'], a['ptr'] + a['bytes']))
        rows.sort()
        self.starts = [r[0] for r in rows]
        self.rows = rows

    def find(self, v):
        i = bisect.bisect_right(self.starts, v) - 1
        return self.rows[i] if i >= 0 and self.rows[i][0] <= v < self.rows[i][1] else None

    def contains(self, v):
        return self.find(v) is not None


def kv_cache_kind(e, allocs):
    """Which paged FP8 cache a dequant launch reads (its page size is a constexpr): the reference's
    per-layer pools differ by size (C128 ~86 MB, SWA ~0.75 GB, C4 ~1.87 GB on one B300)."""
    ptr = struct.unpack('<Q', bytes.fromhex(e['params'][1]['hex']))[0]
    a = allocs.find(ptr)
    if a is None:
        return None
    size = a[1] - a[0]
    if PRO:
        # One rank of DeepSeek-V4-Pro at TP=8: C128 88 MB, SWA 754 MB, C4 1884 MB; other blocks
        # (several tensors in one allocator block) stay untagged and share their CUBIN's tags.
        mb = size >> 20
        return 'c128' if 80 <= mb <= 100 else 'swa' if 700 <= mb <= 800 else 'c4' if 1800 <= mb <= 1950 else None
    return 'c128' if size < (200 << 20) else ('swa' if size < (1200 << 20) else 'c4')


def triton_spec(e, allocs):
    """Specialization letters of a Triton launch (args before the two scratch pointers)."""
    out = []
    for p in e['params'][:-2]:
        b = bytes.fromhex(p['hex'])
        if len(b) == 8:
            v = struct.unpack('<Q', b)[0]
            if allocs.contains(v) or (v > (1 << 32) and v % 16 == 0):
                out.append('P' if v % 16 == 0 else 'p')
                continue
            v = struct.unpack('<q', b)[0]
        elif len(b) == 4:
            v = struct.unpack('<i', b)[0]
        else:
            out.append('?')
            continue
        out.append('1' if v == 1 else 'D' if v % 16 == 0 else 'i')
    return ''.join(out)


def is_triton(e):
    ps = e.get('params') or []
    return len(ps) >= 2 and all(len(p['hex']) == 16 and int(p['hex'], 16) == 0 for p in ps[-2:]) and not e['name'].startswith('_Z')


def layer_kind(label):
    m = re.search(r'model\.layers\.(\d+)', label or '')
    if not m:
        return 'meta' if label else None
    n = int(m.group(1))
    if PRO:  # compress_ratios [128, 128, 4, 128, ...]
        return 'c128' if n < 2 or n % 2 else 'c4'
    return 'swa' if n < 2 else ('c4' if n % 2 == 0 else 'c128')


def tags_for(e, prev, allocs):
    t = _tags_for(e, prev, allocs)
    # The MoE router's Triton kernel is compiled with or without programmatic dependent launch
    # (SGLang turns it off for 384 experts and at most 8 rows on sm_103): two images of one spec.
    if t is not None and e['name'] == '_router_triton_kernel':
        t = dict(t, pdl=int(any(a.get('id') == 6 for a in e.get('attrs') or [])))
    kind = layer_kind(e.get('label'))
    if kind and t is not None:
        t = dict(t, kind=kind)
    return t


def _tags_for(e, prev, allocs):
    name = e['name']
    if name == 'mhc_pre_big_fuse_with_norm_tilelang_kernel' and prev is not None:
        m = re.search(r'tf32_hc_prenorm_gemm_implILj\d+ELj\d+ELj\d+ELj\d+ELj\d+ELj(\d+)E', prev['name'] or '')
        if m:
            return {'n_splits': int(m.group(1))}
        if prev['name'] == 'mhc_fused_post_pre_fma_tilelang_kernel':
            return {'n_splits': struct.unpack('<i', param_bytes(prev, 9))[0]}
        return {}
    if name == 'mhc_fused_post_pre_fma_tilelang_kernel':
        return {'tile': {12: 2, 8: 3}.get(e['grid'][1], -e['grid'][1])}
    if name == 'mhc_post_tilelang_kernel':
        # num_h_blocks (grid.y) is compiled in: 4 for <= 256 tokens, else 1 (SGLang mhc_post).
        return {'hb': e['grid'][1]}
    if is_triton(e):
        t = {'spec': triton_spec(e, allocs)}
        if name == '_dequantize_k_cache_paged_kernel':
            t['cache'] = kv_cache_kind(e, allocs)
        if name == '_combine_topk_swa_indices_kernel':
            # WIDTH (= the output row stride) is a constexpr: one image per combined width.
            t['width'] = struct.unpack('<i', param_bytes(e, 1))[0]
            # SEARCH_STEPS = bit_length(num_reqs) too. num_reqs is the fourth parameter from the
            # end (num_tokens, num_reqs, top_k, then Triton's two scratch pointers: never
            # specialized away; earlier ones can be: the None swa_indices / swa_lengths, a topk
            # stride equal to 1).
            t['steps'] = max(struct.unpack('<i', param_bytes(e, len(e['params']) - 4))[0], 1).bit_length()
        return t
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tape', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--labels', default='')
    ap.add_argument('--examples', type=int, default=3)
    ap.add_argument('--gold', type=Path, default=None,
                    help='gold dir of the same run: maps forwards to token counts for moe_tactics')
    ap.add_argument('--pro', action='store_true', help="DeepSeek-V4-Pro's layer kinds and pool sizes (as DSV4_MODEL=pro)")
    args = ap.parse_args()
    global PRO
    PRO = PRO or args.pro
    labels = [l for l in args.labels.split(',') if l]
    allocs = Allocations(args.tape)
    launches, prev_on_stream = [], {}
    for line in (args.tape / 'launches.jsonl').read_text().splitlines():
        e = json.loads(line)
        if e.get('op') or not e.get('name') or e.get('unresolved'):
            continue
        prev = prev_on_stream.get(e.get('stream'))
        prev_on_stream[e.get('stream')] = e
        if labels and e['label'].split('|')[0] not in labels:
            continue
        e['_prev'] = prev
        launches.append(e)
    modules_dir = args.tape / 'modules'
    images, shas = {}, {}

    def image(mid):
        if mid not in images:
            p = modules_dir / f'{mid}.cubin'
            images[mid] = p.read_bytes() if p.exists() else None
        return images[mid]

    def sha_of(mid):
        if mid not in shas:
            shas[mid] = hashlib.sha256(image(mid)).hexdigest()
        return shas[mid]

    (args.out / 'cubin').mkdir(parents=True, exist_ok=True)
    all_ids = sorted(int(p.stem) for p in modules_dir.glob('*.cubin'))
    symbol_index = {}
    kernels, unresolved = {}, {}
    for e in launches:
        name = e['name']
        bound = [m for m in e.get('module_ids', []) if image(m) is not None and name.encode() + b'\0' in image(m)]
        if bound:
            mid = bound[0]
        else:
            if name not in symbol_index:
                symbol_index[name] = [m for m in all_ids if name.encode() + b'\0' in (image(m) or b'')]
            cands = symbol_index[name]
            if not cands:
                u = unresolved.setdefault(name, {'count': 0, 'example': {k: e.get(k) for k in ('grid', 'block', 'smem')}})
                u['count'] += 1
                continue
            if len({sha_of(m) for m in cands}) > 1:
                print(f'  ambiguous unbound symbol {name[:100]}: {len(cands)} images; taking the first')
            mid = cands[0]
        sha = sha_of(mid)
        out = args.out / 'cubin' / f'{sha}.cubin'
        if not out.exists():
            out.write_bytes(image(mid))
        k = kernels.setdefault(name, {'variants': {}})
        v = k['variants'].setdefault(sha, {'module_ids': [], 'count': 0, 'param_sizes': [p['size'] for p in e['params']],
                                           'tags': tags_for(e, e['_prev'], allocs), 'examples': []})
        if mid not in v['module_ids']:
            v['module_ids'].append(mid)
        v['count'] += 1
        t = tags_for(e, e['_prev'], allocs)
        if t != v['tags']:
            v.setdefault('other_tags', [])
            if t not in v['other_tags']:
                v['other_tags'].append(t)
        if len(v['examples']) < args.examples:
            v['examples'].append({'label': e['label'], 'seq': e['seq'],
                                  **{kk: e.get(kk) for kk in ('grid', 'block', 'smem', 'static_smem', 'regs', 'attrs')}})
    for name, k in kernels.items():
        top_sha, top = max(k['variants'].items(), key=lambda kv: kv[1]['count'])
        k['cubin'] = top_sha
        k['param_sizes'] = top['param_sizes']
        k['example'] = top['examples'][0]
    # The MoE tactic per token count (flashinfer's autotuner picks it per server start): the FC1 /
    # FC2 kernels each forward ran, keyed by the forward's token count from the golds. The token
    # tile is the kernel's N tile (t128x<N>x<K>).
    # NVFP4 experts (the target's) and MXFP4 experts (the MTP draft's: FC1 bmm_MxE4m3_*, FC2
    # bmm_Bfloat16_MxE2m1*) keep separate tables (moe_tactics, moe_tactics_mx).
    moe_tactics, moe_tactics_mx, conflicts = {}, {}, []
    if args.gold is not None:
        tokens = {}
        for d in args.gold.glob('fwd-*'):
            meta = d / 'batch.input_ids.json'
            if meta.exists():
                tokens[d.name] = json.loads(meta.read_text())['shape'][0]
        per_fwd = {}
        for e in launches:
            n = e['name']
            if n.startswith('bmm_'):
                mx = n.startswith('bmm_MxE4m3') or n.startswith('bmm_Bfloat16_MxE2m1')
                per_fwd.setdefault((e['label'].split('|')[0], mx), {}).setdefault('fc2' if n.startswith('bmm_Bfloat16') else 'fc1', n)
        for (fwd, mx), ks in per_fwd.items():
            t = tokens.get(fwd)
            if t is None or len(ks) != 2:
                continue
            tile = int(re.search(r'_t128x(\d+)x', ks['fc1']).group(1))
            entry = {'tile': tile, 'fc1': ks['fc1'], 'fc2': ks['fc2']}
            prev = (moe_tactics_mx if mx else moe_tactics).setdefault(str(t), entry)
            if prev != entry:
                conflicts.append({'tokens': t, 'forward': fwd, 'mx': mx})
    manifest = {'schema': 'dsv4-kernel-bundle-v2', 'tape': str(args.tape), 'labels': labels,
                'moe_tactics': moe_tactics, 'moe_tactics_mx': moe_tactics_mx, 'moe_tactic_conflicts': conflicts,
                'kernels': kernels, 'unresolved': unresolved}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    multi = {n: [(s[:12], v['count'], v['tags']) for s, v in k['variants'].items()] for n, k in kernels.items() if len(k['variants']) > 1}
    print(json.dumps({'moe_tactics': {t: v['fc1'][:60] for t, v in moe_tactics.items()},
                      'moe_tactics_mx': {t: v['fc1'][:60] for t, v in moe_tactics_mx.items()}, 'conflicts': conflicts}))
    print(json.dumps({'kernels': len(kernels), 'cubins': len({s for k in kernels.values() for s in k['variants']}),
                      'unresolved': len(unresolved), 'multi_variant': len(multi)}))
    for n, vs in multi.items():
        print(f'  {n[:90]}: {vs}')
    for n, u in sorted(unresolved.items(), key=lambda kv: -kv[1]['count']):
        print(f"  unresolved {u['count']:5d} {n[:160]}")


if __name__ == '__main__':
    main()
