import json, struct, os, re, collections
ck = '/dsv4/huggingface/hub/models--nvidia--DeepSeek-V4-Flash-NVFP4/snapshots/e3cd60e7de98e9867116860d522499a728de1cf9'
cfg = json.load(open(f'{ck}/config.json'))
print('config', json.dumps({k: v for k, v in cfg.items() if isinstance(v, (int, float, str)) and k not in ('architectures',)})[:1500])
idx = json.load(open(f'{ck}/model.safetensors.index.json'))
shards = sorted(set(idx['weight_map'].values()))
cat = collections.Counter()
nbytes = collections.Counter()
dtypes = collections.Counter()
for sh in shards:
    with open(f'{ck}/{sh}', 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        h = json.loads(f.read(n))
    for name, t in h.items():
        if name == '__metadata__': continue
        numel = 1
        for d in t['shape']: numel *= d
        b = t['data_offsets'][1] - t['data_offsets'][0]
        dt = t['dtype']
        dtypes[dt] += b
        if re.search(r'(scale|_scale_2|input_scale|global_scale)', name.split('.')[-1]) or name.endswith('scale_inv'):
            nbytes['scales'] += b; continue
        logical = numel * 2 if dt == 'U8' else numel  # NVFP4 / MXFP4: two 4-bit values per byte
        mtp = re.match(r'(model\.)?(layers\.(\d+)|mtp)', name)
        layer = None
        m = re.search(r'layers\.(\d+)\.', name)
        if m: layer = int(m.group(1))
        is_mtp = ('mtp' in name) or (layer is not None and layer >= cfg.get('num_hidden_layers', 10**9))
        if is_mtp: key = 'mtp'
        elif re.search(r'\.experts\.\d+\.', name): key = 'routed_experts'
        elif 'shared_expert' in name: key = 'shared_experts'
        elif 'embed' in name: key = 'embed'
        elif 'lm_head' in name or name.endswith('head.weight'): key = 'lm_head'
        else: key = 'other (attention, norms, router, hc)'
        cat[key] += logical
        nbytes[key] += b
tot = sum(cat.values())
print('logical params by part (B):', {k: round(v / 1e9, 2) for k, v in cat.most_common()})
print('total params (B):', round(tot / 1e9, 2), ' excluding mtp:', round((tot - cat['mtp']) / 1e9, 2))
print('bytes by part (GB):', {k: round(v / 1e9, 2) for k, v in nbytes.most_common()})
print('bytes by dtype (GB):', {k: round(v / 1e9, 2) for k, v in dtypes.most_common()})
ne, k = cfg.get('n_routed_experts'), cfg.get('num_experts_per_tok')
if ne and k:
    act = tot - cat['mtp'] - cat['routed_experts'] + cat['routed_experts'] * k / ne
    print(f'active per token (B, {k} of {ne} routed experts, excl. mtp, embed counted):', round(act / 1e9, 2))
