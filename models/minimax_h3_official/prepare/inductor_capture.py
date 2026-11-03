"""Record one call of an Inductor-compiled graph as a native launch program.

`torch.compile(module, dynamic=False)` lowers a graph to a Python wrapper whose `call(args)`
allocates buffers, launches Triton kernels and calls extern GEMM and attention operators. The
live wrapper is executed once more with every call routed through `WrapperRecorder`: the
computation runs unchanged while allocations, views, CUBINs, grids and scalar arguments are
recorded. Programs index allocations by storage (`Buffers`): only declared roots (parameters
and the runtime input) exist before the call and reading any other storage raises, so an
activation can never become an initializer.

Each Triton kernel keeps the configuration Inductor's autotuning chose unless
`force_kernel_configs` pins it; `selected_kernels` reports the choice with the CUBIN SHA-256.
Offline only: the Inductor and Triton attributes used here are those of the pinned PyTorch.
"""
import ast
from collections import Counter
import contextlib
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile

SCALARS = ('i1', 'i32', 'u32', 'i64', 'u64')


def jsonable(value):
    """Informational metadata as strict JSON (the native reader rejects NaN and Infinity)."""
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    return repr(value)


def config_digest(selection):
    """SHA-256 of the {kernel: {kwargs, num_warps}} choice, independent of CUBIN bytes."""
    canonical = {name: {'kwargs': entry['kwargs'], 'num_warps': entry['num_warps']}
                 for name, entry in selection.items()}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class Buffers:
    """Allocation table of a program: one buffer per storage, views as byte offsets/strides."""

    def __init__(self):
        self.buffers, self.storage = [], {}

    def declare(self, value, role, source):
        """A root: a parameter native code uploads, or the input it binds per call."""
        if value.numel() == 0:
            return None
        key = value.untyped_storage().data_ptr()
        if key in self.storage:  # aliasing roots share one allocation
            return self.storage[key]
        if (not value.is_contiguous() or value.storage_offset()
                or value.untyped_storage().nbytes() != value.numel() * value.element_size()):
            raise ValueError(f'{role} root must own one contiguous storage: {source}')
        return self._add(value, role, source)

    def allocate(self, value):
        """A fresh scratch allocation, even when the allocator recycled an earlier address."""
        import torch
        if isinstance(value, torch.Tensor) and value.numel():
            self._add(value, 'scratch', None)

    def _add(self, value, role, source):
        key = value.untyped_storage().data_ptr()
        self.storage[key] = len(self.buffers)
        self.buffers.append({'id': len(self.buffers), 'role': role, 'bytes': value.untyped_storage().nbytes(),
                             'shape': list(value.shape), 'stride': list(value.stride()),
                             'dtype': str(value.dtype), 'source': source})
        return self.storage[key]

    def reference(self, value):
        import torch
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return {'empty': True, 'shape': list(value.shape), 'dtype': str(value.dtype)}
            key = value.untyped_storage().data_ptr()
            if key not in self.storage:
                raise RuntimeError(f'read of undeclared storage: {list(value.shape)} {value.dtype}')
            return {'buffer': self.storage[key], 'offset': value.data_ptr() - key, 'shape': list(value.shape),
                    'stride': list(value.stride()), 'dtype': str(value.dtype)}
        if isinstance(value, (tuple, list)):
            return [self.reference(item) for item in value]
        if isinstance(value, dict):
            return {key: self.reference(item) for key, item in value.items()}
        if isinstance(value, (torch.dtype, torch.memory_format, torch.layout, torch.device)):
            return str(value)
        return value


class WrapCalls(ast.NodeTransformer):
    """`f(*a, **k)` -> `_native_recorder.invoke('f', f, *a, **k)` for every call."""

    def visit_Call(self, node):
        name = ast.unparse(node.func)
        node = self.generic_visit(node)
        return ast.copy_location(ast.Call(
            func=ast.Attribute(value=ast.Name(id='_native_recorder', ctx=ast.Load()), attr='invoke', ctx=ast.Load()),
            args=[ast.Constant(name), node.func, *node.args], keywords=node.keywords), node)


def computes(name):
    """Wrapper calls that would change data outside a recorded operation."""
    last = name.rsplit('.', 1)[-1]
    return name.startswith(('torch.ops.', 'aten.')) or (last.endswith('_') and not last.endswith('__'))


def _selected(tuner):
    if len(tuner.launchers) != 1:
        raise RuntimeError(f'{tuner.fn.__name__}: {len(tuner.launchers)} launchers; run the compiled graph first')
    launcher = tuner.launchers[0]
    results = [result for result in tuner.compile_results if result.config == launcher.config]
    if len(results) != 1:
        raise RuntimeError(f'{tuner.fn.__name__}: the launcher has no unique compile result')
    return launcher, results[0]


def _binary(result):
    """(CUBIN, symbol, shared bytes, warps, hidden null launch arguments) of a compile result."""
    binary = result.kernel
    if hasattr(binary, 'cubin_path'):  # Inductor's static CUDA launcher
        if binary.cubin_raw is None and binary.cubin_path is None:
            result.reload_cubin_path()  # cache-loaded payloads drop both members
        raw = bytes(binary.cubin_raw) if binary.cubin_raw is not None else Path(binary.cubin_path).read_bytes()
        hidden = int(binary.has_global_scratch) + int(binary.has_profile_scratch)
        return raw, binary.name, binary.shared, binary.num_warps, hidden
    hidden = 0
    for field in ('global_scratch_size', 'profile_scratch_size'):
        if hasattr(binary.metadata, field):
            if getattr(binary.metadata, field) != 0:
                raise RuntimeError('nonzero Triton launch scratch is unsupported')
            hidden += 1
    return binary.asm['cubin'], binary.name, binary.metadata.shared, binary.metadata.num_warps, hidden


class WrapperRecorder(Buffers):
    def __init__(self, root):
        super().__init__()
        self.root = Path(root)
        self.ops, self.kernels, self.ignored = [], {}, Counter()

    def kernel(self, tuner, args):
        import torch
        launcher, result = _selected(tuner)
        raw, symbol, shared, warps, hidden = _binary(result)
        digest = hashlib.sha256(raw).hexdigest()
        path = self.root / 'kernels' / (digest + '.cubin')
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        actual, grid = tuner._interpret_args_grid(tuple(args), launcher.config)
        signature = result.compile_meta['signature']
        constants = result.compile_meta.get('constants', {})
        # Python passes every non-explicit-constexpr argument; implicit constants are absent from the ABI.
        names = list(tuner.fn.arg_names)
        constexprs = set(tuner.fn.constexprs)
        runtime = [name for index, name in enumerate(names) if index not in constexprs]
        if len(actual) != len(runtime):
            raise RuntimeError(f'{symbol}: {len(actual)} launch arguments for {runtime}')
        params = []
        for name, value in zip(runtime, actual):
            if name in constants or names.index(name) in constants or signature.get(name) == 'constexpr':
                continue
            kind = signature[name]
            if isinstance(value, torch.Tensor):
                if not kind.startswith('*'):
                    raise RuntimeError(f'{symbol}: tensor passed as {kind}')
                params.append({'type': 'pointer', **self.reference(value)})
            elif kind in SCALARS:
                params.append({'type': kind, 'bits': int(value) & ((1 << (64 if kind.endswith('64') else 32)) - 1)})
            elif kind == 'fp32':
                params.append({'type': kind, 'bits': struct.unpack('<I', struct.pack('<f', float(value)))[0]})
            else:
                raise RuntimeError(f'{symbol}: unsupported native Triton parameter {name}: {kind}')
        params.extend({'type': 'u64', 'bits': 0} for _ in range(hidden))
        self.kernels.setdefault(digest, {
            'file': str(path.relative_to(self.root)), 'sha256': digest, 'symbol': symbol, 'shared': shared,
            'warps': warps, 'hidden_null_args': hidden, 'signature': jsonable(signature),
            'constants': jsonable(constants),
            'config': {'kwargs': jsonable(launcher.config.kwargs), 'num_warps': launcher.config.num_warps}})
        self.ops.append({'kind': 'kernel', 'kernel': digest, 'grid': [int(x) for x in grid], 'args': params})

    def invoke(self, name, function, *args, **kwargs):
        # References are taken before an extern operator allocates its outputs.
        kind = None
        if name.startswith('extern_kernels.'):
            kind = name.rsplit('.', 1)[-1]
            if kind not in ('mm', 'addmm'):
                raise RuntimeError('unsupported extern operator: ' + name)
        elif '_scaled_dot_product_cudnn_attention' in name:
            kind = 'cudnn_sdpa'
        elif computes(name):
            raise RuntimeError('unrecorded wrapper operation: ' + name)
        entry = {'kind': kind, 'args': self.reference(args), 'kwargs': self.reference(kwargs)} if kind else None
        result = function(*args, **kwargs)
        if name.startswith('empty_strided_'):
            self.allocate(result)
        elif kind in ('mm', 'addmm'):
            self.ops.append(entry)
        elif kind == 'cudnn_sdpa':
            for value in result:
                self.allocate(value)
            entry['outputs'] = self.reference(result)
            self.ops.append(entry)
        elif name.endswith('.run') and hasattr(getattr(function, '__self__', None), 'triton_meta'):
            self.kernel(function.__self__, args)
        elif name.endswith('.run'):
            raise RuntimeError('unrecognized kernel launch: ' + name)
        else:
            self.ignored[name] += 1
        return result


def wrapper_call(source):
    """The wrapper's `call` function with every call instrumented (a method loses `self`)."""
    calls = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.FunctionDef) and node.name == 'call']
    if len(calls) != 1:
        raise RuntimeError(f'expected one wrapper call function, found {len(calls)}')
    function = WrapCalls().visit(calls[0])
    if function.args.args and function.args.args[0].arg == 'self':
        function.args.args.pop(0)
    return ast.fix_missing_locations(function)


def record_call(source, namespace, inputs, roots, root):
    """Execute the instrumented wrapper on `inputs`; return (recorder, wrapper result).

    `roots[i]` describes `inputs[i]`: {'input': name} for the runtime input, otherwise
    {'parameter': name, 'spec': tensor file spec}. The input is declared first (buffer 0).
    """
    import torch
    function = wrapper_call(source)
    recorder = WrapperRecorder(root)
    for index in sorted(range(len(inputs)), key=lambda i: 'input' not in roots[i]):
        recorder.declare(inputs[index], 'input' if 'input' in roots[index] else 'parameter', roots[index])
    scope = dict(namespace)
    scope['_native_recorder'] = recorder
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<inductor-wrapper-capture>', 'exec'), scope)
    # AOTAutograd's outer wrapper already disabled autocast; every conversion is explicit in the graph.
    with torch.no_grad(), torch._C._DisableAutocast():
        result = scope['call'](list(inputs))
    return recorder, result


def selected_kernels(namespace):
    """{kernel: {sha256, kwargs, num_warps}} of every Triton kernel in a wrapper module."""
    result = {}
    for name, tuner in sorted(namespace.items()):
        if not name.startswith('triton_') or not hasattr(tuner, 'launchers'):
            continue
        launcher, compiled = _selected(tuner)
        result[name] = {'sha256': hashlib.sha256(_binary(compiled)[0]).hexdigest(),
                        'kwargs': jsonable(launcher.config.kwargs), 'num_warps': launcher.config.num_warps}
    return result


def force_kernel_configs(namespace, configs):
    """Recompile the named kernels with the given {kwargs, num_warps}; return the changed names."""
    selected = selected_kernels(namespace)
    unknown = sorted(set(configs) - set(selected))
    if unknown:
        raise ValueError('kernel configurations name kernels absent from the graph: ' + ', '.join(unknown))
    changed = []
    for name, wanted in sorted(configs.items()):
        tuner = namespace[name]
        current = tuner.launchers[0].config
        if current.kwargs == wanted['kwargs'] and current.num_warps == wanted['num_warps']:
            continue
        config = copy.deepcopy(current)
        config.kwargs = dict(wanted['kwargs'])
        config.num_warps = wanted['num_warps']
        tuner._ensure_kernel_loaded()
        compiled = tuner._precompile_config(config)
        tuner.compile_results = [compiled]
        tuner.launchers = [compiled.make_launcher()]
        tuner._cached_launcher = None
        changed.append(name)
    after = selected_kernels(namespace)
    for name, wanted in configs.items():
        if (after[name]['kwargs'], after[name]['num_warps']) != (wanted['kwargs'], wanted['num_warps']):
            raise RuntimeError('kernel configuration was not applied: ' + name)
    return changed


def capture_graph(run):
    """Call `run()` once and return the single compiled graph it executed: (graph, inputs, autocast)."""
    import torch
    import torch._inductor.output_code as output_code
    captured = []
    original = output_code.CompiledFxGraph.__call__

    def observe(graph, inputs):
        captured.append((graph, list(inputs), torch.is_autocast_enabled('cuda')))
        return original(graph, inputs)

    output_code.CompiledFxGraph.__call__ = observe
    try:
        run()
    finally:
        output_code.CompiledFxGraph.__call__ = original
    if len(captured) != 1:
        raise RuntimeError(f'expected one compiled graph call, observed {len(captured)}')
    return captured[0]


@contextlib.contextmanager
def fresh_compilation():
    """Compile and autotune from scratch: a new cache directory holds no graph or autotuning result.

    Inductor's own cache switches are no use here: with them off, its autotuning subprocesses keep
    no CUBIN file for the selected kernel. The directory outlives the capture for the same reason.
    """
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = tempfile.mkdtemp(prefix='h3-inductor-')
    yield
