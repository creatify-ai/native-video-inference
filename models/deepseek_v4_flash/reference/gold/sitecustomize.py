"""Gold capture inside SGLang's scheduler process (loaded through PYTHONPATH as sitecustomize).

Inactive unless DSV4_GOLD_DIR is set. Then, in every process that imports SGLang's ModelRunner:

* after load_model: DSV4_GOLD_DIR/params.json lists every parameter and buffer of the model
  (name, device address, shape, stride, dtype, bytes) and, with DSV4_GOLD_HASH_PARAMS=1, the SHA-256
  of each one's bytes as SGLang holds them after its load-time processing;
* while DSV4_GOLD_DIR/ARM exists: each ModelRunner.forward is captured into fwd-<n>/ (inputs, the
  outputs of every hooked module, logits, sampled ids) and, when the CUPTI tape is injected, its
  kernel launches are recorded under the label fwd-<n>. ARM holds the number of forwards still to
  capture; it is removed when that reaches zero.

Tensors are written raw (<name>.bin) with a JSON sidecar (dtype, shape, stride, sha256 of the
contiguous bytes); with DSV4_GOLD_MAX_BYTES, larger tensors keep only the sidecar. Nothing here changes what SGLang computes; captures synchronize the device.
"""
import os

_DIR = os.environ.get('DSV4_GOLD_DIR')


def _install():
    import builtins
    import ctypes
    import hashlib
    import json
    import threading
    from pathlib import Path

    root = Path(_DIR)
    root.mkdir(parents=True, exist_ok=True)
    state = {'n': 0, 'active': None, 'tape': None, 'hooks_on': False}
    lock = threading.Lock()
    tape_path = os.environ.get('CUDA_INJECTION64_PATH')
    if tape_path and os.path.basename(tape_path).startswith('libcupti_tape'):
        try:
            lib = ctypes.CDLL(tape_path)
            lib.tape_start.argtypes = [ctypes.c_char_p]
            lib.tape_set_snapshot.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint64, ctypes.c_int]
            state['tape'] = lib
        except OSError:
            state['tape'] = None

    def save(directory, name, tensor):
        import torch
        if not isinstance(tensor, torch.Tensor):
            return None
        t = tensor.detach()
        meta = {'dtype': str(t.dtype).replace('torch.', ''), 'shape': list(t.shape), 'stride': list(t.stride()),
                'device_ptr': t.data_ptr() if t.is_cuda else None}
        c = t.contiguous()
        if c.dtype in (torch.float8_e4m3fn, torch.float8_e5m2, getattr(torch, 'float8_e8m0fnu', None),
                       getattr(torch, 'float4_e2m1fn_x2', None)):
            c = c.view(torch.uint8)
        # A fresh buffer: a size-1 view keeps a non-unit stride through contiguous() and clone(),
        # which view() rejects.
        if c.numel():
            flat = torch.empty(c.numel(), dtype=c.dtype, device='cpu')
            flat.copy_(c.reshape(-1))
            data = flat.view(torch.uint8).numpy().tobytes()
        else:
            data = b''
        meta['sha256'] = hashlib.sha256(data).hexdigest()
        meta['bytes'] = len(data)
        safe = name.replace('/', '_')
        # Long prompts: tensors above DSV4_GOLD_MAX_BYTES keep only their SHA-256 (comparisons need
        # nothing else); a targeted capture can fetch one in full when a mismatch needs a diff.
        cap = int(os.environ.get('DSV4_GOLD_MAX_BYTES', '0') or 0)
        if cap and len(data) > cap:
            meta['payload'] = 'omitted'
        else:
            (directory / f'{safe}.bin').write_bytes(data)
        (directory / f'{safe}.json').write_text(json.dumps(meta))
        return meta

    def flatten(prefix, value, out):
        import torch
        if isinstance(value, torch.Tensor):
            out.append((prefix, value))
        elif isinstance(value, (tuple, list)):
            for i, v in enumerate(value):
                flatten(f'{prefix}.{i}', v, out)
        elif isinstance(value, dict):
            for k, v in value.items():
                flatten(f'{prefix}.{k}', v, out)
        elif hasattr(value, '__struct_fields__'):  # msgspec structs (SGLang's unreduced / MoE outputs)
            for k in value.__struct_fields__:
                flatten(f'{prefix}.{k}', getattr(value, k, None), out)
        elif hasattr(value, 'next_token_logits'):
            flatten(f'{prefix}.next_token_logits', value.next_token_logits, out)
            if getattr(value, 'hidden_states', None) is not None:
                flatten(f'{prefix}.hidden_states', value.hidden_states, out)

    def armed():
        arm = root / 'ARM'
        if not arm.exists():
            return False
        if state.get('rank_root') is not None:
            # Several TP ranks share ARM: each captures its first N forwards and leaves the file.
            try:
                return state['n'] < int(arm.read_text().strip() or '1')
            except ValueError:
                return False
        try:
            left = int(arm.read_text().strip() or '1')
        except ValueError:
            left = 1
        if left <= 1:
            arm.unlink(missing_ok=True)
        else:
            arm.write_text(str(left - 1))
        return True

    snap_forwards = {int(x) for x in os.environ.get('DSV4_GOLD_SNAPSHOT_FORWARDS', '').split(',') if x.strip()}
    snap_modules = [m.strip() for m in os.environ.get('DSV4_GOLD_SNAPSHOT_MODULES', '').split(',') if m.strip()]

    def snapshot_wanted(active):
        if not snap_forwards or active['n'] not in snap_forwards:
            return False
        if not active['stack']:
            return ('__post__' if active.get('embedded') else '__pre__') in snap_modules
        top = active['stack'][-1]
        if top == 'model.embed_tokens':
            active['embedded'] = True
        return any(top == m or top.startswith(m + '.') for m in snap_modules)

    def snapshot_toggle(active):
        lib = state['tape']
        if lib is None or not snap_forwards:
            return
        if not state.get('excluded_set'):
            ranges = state.get('weight_ranges', [])
            arr = (ctypes.c_uint64 * (2 * len(ranges)))(*[v for r in ranges for v in r])
            lib.tape_set_snapshot(arr, len(ranges), int(os.environ.get('DSV4_GOLD_SNAPSHOT_CAP', 262144)), 0)
            state['excluded_set'] = True
        lib.tape_set_snapshot(None, 0, 0, 1 if snapshot_wanted(active) else 0)

    def hook_modules(model, draft=False):
        # Each model once: the target, and with speculative decoding the draft (its golds
        # prefixed `draft.`; DSV4_GOLD_DRAFT_MODULES picks its modules).
        hooked = state.setdefault('hooked', set())
        if id(model) in hooked:
            return
        hooked.add(id(model))
        state['hooks_on'] = True
        if draft:
            names = os.environ.get('DSV4_GOLD_DRAFT_MODULES', 'model,model.decoder,model.decoder.self_attn,model.decoder.mlp,logits_processor')
        else:
            names = os.environ.get('DSV4_GOLD_MODULES', 'model.embed_tokens,model.layers.*,model.norm,lm_head,logits_processor')
        patterns = [p.strip() for p in names.split(',') if p.strip()]
        tag = 'draft.' if draft else ''

        def wanted(name):
            for p in patterns:
                if p.endswith('.*'):
                    base = p[:-2]
                    if name.startswith(base + '.') and name[len(base) + 1:].isdigit():
                        return True
                elif name == p:
                    return True
            return False

        def save_paused(active, items):
            # Saving launches copies; keep them off the tape.
            if state['tape'] is not None:
                state['tape'].tape_stop()
            for key, tensor in items:
                active['tensors'][key] = save(active['dir'], key, tensor)

        def relabel(active):
            if state['tape'] is not None:
                label = active['dir'].name + ('|' + active['stack'][-1] if active['stack'] else '')
                snapshot_toggle(active)
                state['tape'].tape_start(label.encode())

        for name, module in model.named_modules():
            if not wanted(name):
                continue

            def pre(mod, args, kwargs, _name=tag + name):
                active = state['active']
                if active is None:
                    return
                items = []
                for k, v in kwargs.items():
                    if k in ('hidden_states', 'positions', 'input_ids', 'x', 'x_quant'):
                        flatten(f'{_name}.in.{k}', v, items)
                for i, a in enumerate(args):
                    flatten(f'{_name}.in.{i}', a, items)
                save_paused(active, items)
                active['stack'].append(_name)
                relabel(active)

            def post(mod, args, kwargs, output, _name=tag + name):
                active = state['active']
                if active is None:
                    return
                items = []
                flatten(f'{_name}.out', output, items)
                save_paused(active, items)
                if active['stack'] and active['stack'][-1] == _name:
                    active['stack'].pop()
                relabel(active)

            module.register_forward_pre_hook(pre, with_kwargs=True)
            module.register_forward_hook(post, with_kwargs=True)

    def hook_collectives():
        """Tensor parallelism: record each all-reduce's input (this rank's partial) and output (the
        sum) of a captured forward as ar-<k>.in / ar-<k>.out, in call order, and all-gathers as
        ag-<k>.in / ag-<k>.out."""
        from sglang.srt.distributed import parallel_state
        cls = parallel_state.GroupCoordinator
        if getattr(cls, '_dsv4_gold', False):
            return
        cls._dsv4_gold = True

        def wrap(method, tag):
            original = getattr(cls, method)

            def wrapped(self, input_, *a, **k):
                active = state['active']
                if active is None:
                    return original(self, input_, *a, **k)
                n = active.setdefault('collectives', 0)
                active['collectives'] = n + 1
                key = f'{tag}-{n:04d}'
                items = []
                flatten(f'{key}.in', input_, items)
                if state['tape'] is not None:
                    state['tape'].tape_stop()
                for name, tensor in items:
                    active['tensors'][name] = save(active['dir'], name, tensor)
                if state['tape'] is not None:
                    state['tape'].tape_start((active['dir'].name + '|' + key).encode())
                out = original(self, input_, *a, **k)
                items = []
                flatten(f'{key}.out', out, items)
                if state['tape'] is not None:
                    state['tape'].tape_stop()
                for name, tensor in items:
                    active['tensors'][name] = save(active['dir'], name, tensor)
                if state['tape'] is not None:
                    label = active['dir'].name + ('|' + active['stack'][-1] if active['stack'] else '')
                    state['tape'].tape_start(label.encode())
                return out
            setattr(cls, method, wrapped)

        wrap('all_reduce', 'ar')
        wrap('all_gather', 'ag')

    def dump_params(model, draft=False):
        import torch
        rows = []
        do_hash = os.environ.get('DSV4_GOLD_HASH_PARAMS') == '1'
        for kind, items in (('param', model.named_parameters()), ('buffer', model.named_buffers())):
            for name, t in items:
                row = {'kind': kind, 'name': name, 'dtype': str(t.dtype).replace('torch.', ''), 'shape': list(t.shape),
                       'stride': list(t.stride()), 'device_ptr': t.data_ptr(), 'bytes': t.numel() * t.element_size()}
                if do_hash and t.numel():
                    c = t.detach().contiguous().view(torch.uint8) if t.element_size() == 1 else t.detach().contiguous()
                    row['sha256'] = hashlib.sha256(c.cpu().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
                rows.append(row)
        state['weight_ranges'] = state.get('weight_ranges', []) + [(r['device_ptr'], r['bytes']) for r in rows if r['bytes']]
        # Load-time tables no checkpoint tensor holds (e.g. RoPE): save them for the native port.
        wanted = [b.strip() for b in os.environ.get('DSV4_GOLD_DUMP_BUFFERS', '').split(',') if b.strip()]
        if wanted:
            tables = root / 'tables'
            tables.mkdir(exist_ok=True)
            for name, t in model.named_buffers():
                if name in wanted:
                    save(tables, name, t)
        (root / f"params-pid{os.getpid()}{'-draft' if draft else ''}.json").write_text(json.dumps(rows, indent=0))

    def dump_package(model, runner):
        """DSV4_GOLD_PACKAGE_DIR: this rank's parameters and buffers, as the bytes the device holds
        (each tensor's memory image from its first element: equal to its values when contiguous),
        written to <dir>/rank<r>/weights/shard-*.safetensors in the prepared-package format
        (prepare.rs: U8 tensors, 4 KiB aligned, 64 MiB shards, larger tensors as pieces name#kkk)
        with <dir>/rank<r>/params.json (shape, stride, SHA-256 of the values and of the image)."""
        import struct
        import torch
        rank = getattr(runner, 'tp_rank', None)
        if rank is None:  # one rank per device
            rank = next(model.parameters()).device.index or 0
        rank = int(rank)
        out = Path(os.environ['DSV4_GOLD_PACKAGE_DIR']) / f'rank{rank}'
        weights = out / 'weights'
        weights.mkdir(parents=True, exist_ok=True)
        piece, align = 64 << 20, 4096
        pending, files = [], []

        def flush():
            if not pending:
                return
            header, offset, spans = {}, 0, []
            for name, data in pending:
                header[name] = {'dtype': 'U8', 'shape': [len(data)], 'data_offsets': [offset, offset + len(data)]}
                spans.append(offset)
                offset = -(-(offset + len(data)) // align) * align
            text = json.dumps(header, separators=(',', ':')).encode()
            text += b' ' * (-(8 + len(text)) % align)
            blob = bytearray(8 + len(text) + offset)
            blob[:8] = struct.pack('<Q', len(text))
            blob[8:8 + len(text)] = text
            base = 8 + len(text)
            for (name, data), span in zip(pending, spans):
                blob[base + span:base + span + len(data)] = data
            file = f'shard-{len(files):05d}.safetensors'
            (weights / file).write_bytes(blob)
            files.append({'file': file, 'bytes': len(blob), 'tensors': len(pending)})
            pending.clear()

        def push(name, data):
            aligned = -(-len(data) // align) * align
            if pending and sum(-(-len(d) // align) * align for _, d in pending) + aligned > piece:
                flush()
            pending.append((name, data))

        def raw(t):
            c = t.cpu()
            if c.is_complex():
                c = torch.view_as_real(c).reshape(-1)
            return memoryview(c.contiguous().view(torch.uint8).numpy()).cast('B')

        rows, seen, total = [], {}, 0
        for kind, items in (('param', model.named_parameters()), ('buffer', model.named_buffers())):
            for name, t in items:
                t = t.detach()
                row = {'kind': kind, 'name': name, 'dtype': str(t.dtype).replace('torch.', ''), 'shape': list(t.shape),
                       'stride': list(t.stride()), 'device_ptr': t.data_ptr(), 'bytes': t.numel() * t.element_size(),
                       'device': str(t.device)}
                rows.append(row)
                if not t.numel():
                    continue
                extent = sum((n - 1) * st for n, st in zip(t.shape, t.stride())) + 1
                key = (t.data_ptr(), extent, row['dtype'])
                if key in seen:  # the same memory under another name
                    row.update(alias_of=seen[key])
                    continue
                seen[key] = name
                image = raw(t.as_strided((extent,), (1,)))
                row['device_bytes'] = len(image)
                row['device_sha256'] = hashlib.sha256(image).hexdigest()
                row['sha256'] = row['device_sha256'] if t.is_contiguous() else hashlib.sha256(raw(t.contiguous().reshape(-1))).hexdigest()
                total += len(image)
                if len(image) > piece:
                    row['pieces'] = -(-len(image) // piece)
                    for k in range(row['pieces']):
                        push(f'{name}#{k:03d}', image[k * piece:(k + 1) * piece])
                else:
                    push(name, image)
        flush()
        (out / 'params.json').write_text(json.dumps(
            {'schema': 'dsv4-rank-package-v1', 'rank': rank, 'gpu_id': getattr(runner, 'gpu_id', None), 'pid': os.getpid(),
             'stored_bytes': total, 'files': files, 'params': rows}, indent=0))

    def patch(module):
        runner_cls = getattr(module, 'ModelRunner', None)
        if runner_cls is None or getattr(runner_cls, '_dsv4_gold', False):
            return
        runner_cls._dsv4_gold = True
        original_load = runner_cls.load_model
        original_forward = runner_cls.forward
        original_sample = runner_cls.sample

        def load_model(self, *a, **k):
            result = original_load(self, *a, **k)
            try:
                draft = bool(getattr(self, 'is_draft_worker', False))
                # DSV4_GOLD_PER_RANK=1 (tensor parallelism): each rank's forwards go to rank<r>/.
                if os.environ.get('DSV4_GOLD_PER_RANK') == '1' and not draft:
                    # SGLang's ModelRunner has no tp_rank attribute: one rank per device.
                    rank = getattr(self, 'tp_rank', None)
                    if rank is None:
                        rank = next(self.model.parameters()).device.index or 0
                    rank = int(rank)
                    state['rank_root'] = root / f'rank{rank}'
                    state['rank_root'].mkdir(parents=True, exist_ok=True)
                    (state['rank_root'] / 'rank.json').write_text(json.dumps(
                        {'rank': rank, 'pid': os.getpid(), 'gpu_id': getattr(self, 'gpu_id', None)}))
                    try:
                        hook_collectives()
                    except Exception as e:
                        (root / f'error-collectives-pid{os.getpid()}.txt').write_text(repr(e))
                dump_params(self.model, draft)
                if os.environ.get('DSV4_GOLD_PACKAGE_DIR') and not draft:
                    dump_package(self.model, self)
                hook_modules(self.model, draft)
            except Exception as e:  # never break the server
                (root / f'error-load-pid{os.getpid()}.txt').write_text(repr(e))
            return result

        def forward(self, forward_batch, *a, **k):
            import torch
            with lock:
                capture = armed()
                if capture:
                    state['n'] += 1
                    d = (state.get('rank_root') or root) / f"fwd-{state['n']:03d}"
                    d.mkdir(parents=True, exist_ok=True)
                    state['active'] = {'dir': d, 'tensors': {}, 'stack': [], 'n': state['n']}
            if not capture:
                return original_forward(self, forward_batch, *a, **k)
            active = state['active']
            fb = forward_batch
            meta = {'forward_mode': str(getattr(fb, 'forward_mode', None)), 'batch_size': getattr(fb, 'batch_size', None),
                    'runner': 'draft' if getattr(self, 'is_draft_worker', False) else 'target'}
            for key in ('input_ids', 'positions', 'seq_lens', 'out_cache_loc', 'req_pool_indices', 'extend_seq_lens',
                        'extend_prefix_lens', 'extend_start_loc'):
                v = getattr(fb, key, None)
                if isinstance(v, torch.Tensor):
                    active['tensors'][f'batch.{key}'] = save(active['dir'], f'batch.{key}', v)
            # Speculative decoding: the hidden states and draft fields the forward reads.
            spec = getattr(fb, 'spec_info', None)
            if spec is not None:
                for key in ('hidden_states', 'positions', 'draft_token', 'topk_index', 'accept_length', 'bonus_tokens', 'seq_lens_for_draft_extend'):
                    v = getattr(spec, key, None)
                    if isinstance(v, torch.Tensor):
                        active['tensors'][f'spec.{key}'] = save(active['dir'], f'spec.{key}', v)
            torch.cuda.synchronize()
            if state['tape'] is not None:
                snapshot_toggle(active)
                state['tape'].tape_start(active['dir'].name.encode())
            try:
                out = original_forward(self, forward_batch, *a, **k)
                torch.cuda.synchronize()
            finally:
                if state['tape'] is not None:
                    state['tape'].tape_stop()
                    if snap_forwards:
                        state['tape'].tape_set_snapshot(None, 0, 0, 0)
                    # Keep recording between forwards (speculative decoding's verify / accept /
                    # draft sampling kernels run outside them) until the next forward relabels.
                    state['tape'].tape_start((active['dir'].name + '|post').encode())
            items = []
            flatten('output', out, items)
            for key, tensor in items:
                active['tensors'][key] = save(active['dir'], key, tensor)
            meta['tensors'] = active['tensors']
            (active['dir'] / 'meta.json').write_text(json.dumps(meta, indent=1, default=str))
            return out

        def sample(self, logits_output, forward_batch, *a, **k):
            active = state['active']
            if active is not None and state['tape'] is not None:
                import torch
                torch.cuda.synchronize()
                active['stack'].append('sample')
                snapshot_toggle(active)
                active['stack'].pop()
                state['tape'].tape_start((active['dir'].name + '|sample').encode())
            try:
                result = original_sample(self, logits_output, forward_batch, *a, **k)
            finally:
                if active is not None and state['tape'] is not None:
                    import torch
                    torch.cuda.synchronize()
                    state['tape'].tape_stop()
                    if snap_forwards:
                        state['tape'].tape_set_snapshot(None, 0, 0, 0)
                    state['tape'].tape_start((active['dir'].name + '|post').encode())
            if active is not None:
                import torch
                torch.cuda.synchronize()
                items = []
                flatten('sample.out', result, items)
                for key, tensor in items:
                    active['tensors'][key] = save(active['dir'], key, tensor)
                (active['dir'] / 'meta.json').write_text(json.dumps({'tensors': active['tensors']}, indent=1, default=str)) \
                    if not (active['dir'] / 'meta.json').exists() else None
                state['active'] = None
            return result

        runner_cls.load_model = load_model
        runner_cls.forward = forward
        runner_cls.sample = sample

    # Patch sglang.srt.model_executor.model_runner once its module body has run: a meta-path finder
    # wraps that one module's loader. (No per-import hook: nothing else is slowed down.)
    import importlib.abc
    import importlib.machinery
    import sys

    target_name = 'sglang.srt.model_executor.model_runner'

    class _Loader(importlib.abc.Loader):
        def __init__(self, inner):
            self.inner = inner

        def create_module(self, spec):
            return self.inner.create_module(spec)

        def exec_module(self, module):
            self.inner.exec_module(module)
            try:
                patch(module)
            except Exception as e:  # never break the server; leave a trace instead
                (root / f'error-patch-pid{os.getpid()}.txt').write_text(repr(e))

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != target_name:
                return None
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            if spec is not None and spec.loader is not None:
                spec.loader = _Loader(spec.loader)
            return spec

    if target_name in sys.modules:
        patch(sys.modules[target_name])
    else:
        sys.meta_path.insert(0, _Finder())
    (root / f'installed-pid{os.getpid()}.txt').write_text(' '.join(sys.argv)[:2000])


def _chain():
    """Run the sitecustomize this file shadows, if another directory on sys.path has one."""
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    for entry in sys.path:
        candidate = os.path.join(entry or '.', 'sitecustomize.py')
        if os.path.abspath(entry or '.') != here and os.path.isfile(candidate):
            with open(candidate) as fh:
                exec(compile(fh.read(), candidate, 'exec'), {'__name__': 'sitecustomize', '__file__': candidate})
            return


if _DIR:
    _install()
_chain()
