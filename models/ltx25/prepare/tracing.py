"""Record eager ATen dataflow and the exact Triton CUBINs it launches.

A traced call becomes a `*.graph.json` program: model parameters, buffers and the call's own
arguments are external tensors; every node output carries its SHA-256. An unexpected external
GPU activation raises instead of becoming a frozen answer. The native runtime binds these
programs to its own kernels and replays the recorded CUBINs (`triton.rs`).
"""
from contextvars import ContextVar
import dataclasses
import hashlib
import json
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes

from .common import write_json

ACTIVE_TRACE = ContextVar("ltx25_graph_trace", default=None)


class GraphTrace(TorchDispatchMode):
    def __init__(self, writer, name, module, args, kwargs):
        super().__init__()
        self.writer, self.name = writer, name
        self.ids, self.keepalive, self.nodes = {}, [], []
        self.scratch_storages = set()
        self.externals = {}
        for key, parameter in module.named_parameters():
            self.register(parameter, name + ".weights." + key)
        for key, buffer in getattr(module, "named_buffers", lambda: ())():
            self.register(buffer, name + ".buffers." + key)
        self.visit(name + ".args", args)
        self.visit(name + ".kwargs", kwargs)

    def __enter__(self):
        self.trace_token = ACTIVE_TRACE.set(self)
        return super().__enter__()

    def __exit__(self, *args):
        try:
            return super().__exit__(*args)
        finally:
            ACTIVE_TRACE.reset(self.trace_token)

    def register(self, tensor, name):
        if id(tensor) not in self.ids:
            self.ids[id(tensor)] = name
            self.keepalive.append(tensor)
            self.externals[name] = self.describe(tensor)
        return self.ids[id(tensor)]

    def visit(self, path, value):
        if isinstance(value, torch.Tensor):
            self.register(value, path)
        elif dataclasses.is_dataclass(value):
            for field in dataclasses.fields(value):
                self.visit(path + "." + field.name, getattr(value, field.name))
        elif isinstance(value, dict):
            for key, item in value.items():
                self.visit(path + "." + str(key), item)
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                self.visit(path + "." + str(index), item)

    def describe(self, tensor):
        return {"shape": list(tensor.shape), "stride": list(tensor.stride()), "dtype": str(tensor.dtype),
                "device": str(tensor.device), "storage_offset": tensor.storage_offset()}

    def encode(self, value):
        if isinstance(value, torch.Tensor):
            if id(value) not in self.ids:
                if value.device.type == "cuda" and value.numel() != 0:
                    raise RuntimeError(f"unregistered graph activation: {self.describe(value)}")
                key = self.name + f".constant.{len(self.externals)}"
                self.register(value, key)
                self.writer.tensor(key, value)
            return {"tensor": self.ids[id(value)]}
        if isinstance(value, (list, tuple)):
            return [self.encode(item) for item in value]
        if isinstance(value, dict):
            return {key: self.encode(item) for key, item in value.items()}
        if isinstance(value, torch.dtype):
            return {"dtype": str(value)}
        if isinstance(value, torch.device):
            return {"device": str(value)}
        if isinstance(value, torch.layout):
            return {"layout": str(value)}
        if isinstance(value, torch.memory_format):
            return {"memory_format": str(value)}
        if value is None or isinstance(value, (int, float, str, bool)):
            return value
        raise TypeError(f"unsupported graph scalar {type(value).__name__}: {value}")

    def outputs(self, value, force=False):
        if isinstance(value, torch.Tensor):
            if value.untyped_storage().data_ptr() in self.scratch_storages and not force:
                name = self.name + f".uninitialized.{len(self.keepalive)}"
                self.ids[id(value)] = name
                self.keepalive.append(value)
                return {"tensor": name, **self.describe(value), "uninitialized": True}
            # In-place/view returns may be the same Python Tensor as an input:
            # assign a fresh SSA version even when the underlying storage aliases.
            name = self.name + f".node.{len(self.writer.tensors)}"
            self.ids[id(value)] = name
            self.keepalive.append(value)
            entry = self.writer.tensor(name, value, save=False)
            return {"tensor": name, **self.describe(value), "sha256": entry["sha256"]}
        if isinstance(value, (list, tuple)):
            return [self.outputs(item, force) for item in value]
        if value is None or isinstance(value, (int, float, str, bool)):
            return value
        raise TypeError(f"unsupported graph output: {type(value).__name__}")

    def record_output(self, func, output):
        if str(func) in ("aten.empty.memory_format", "aten.empty_strided.default", "aten.empty_like.default",
                         "aten.new_empty.default"):
            if not isinstance(output, torch.Tensor):
                raise RuntimeError("unexpected scratch factory output")
            name = self.name + f".scratch.{len(self.nodes)}"
            self.ids[id(output)] = name
            self.keepalive.append(output)
            self.scratch_storages.add(output.untyped_storage().data_ptr())
            return {"tensor": name, **self.describe(output), "uninitialized": True}
        return self.outputs(output, bool(getattr(func._schema, "is_mutable", False)))

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        encoded_args, encoded_kwargs = self.encode(args), self.encode(kwargs)
        output = func(*args, **kwargs)
        self.nodes.append({"op": str(func), "args": encoded_args, "kwargs": encoded_kwargs,
                           "out": self.record_output(func, output)})
        return output

    def finish(self, output):
        graph = {"schema": "ltx25-aten-graph-v1", "externals": self.externals,
                 "nodes": self.nodes, "op_counts": {}, "outputs": [], "output_paths": {}}
        for node in self.nodes:
            graph["op_counts"][node["op"]] = graph["op_counts"].get(node["op"], 0) + 1

        def visit(value, path=""):
            if isinstance(value, torch.Tensor):
                graph["outputs"].append(self.ids[id(value)])
                graph["output_paths"][path] = self.ids[id(value)]
            elif dataclasses.is_dataclass(value):
                for field in dataclasses.fields(value):
                    visit(getattr(value, field.name), path + "." + field.name)
            elif isinstance(value, (tuple, list)):
                for index, item in enumerate(value):
                    visit(item, path + "." + str(index))
        visit(output)
        write_json(self.writer.root / (self.name + ".graph.json"), graph)
        return graph


def contains_cuda(value):
    if isinstance(value, torch.Tensor):
        return value.device.type == "cuda"
    if isinstance(value, (tuple, list)):
        return any(contains_cuda(v) for v in value)
    if isinstance(value, dict):
        return any(contains_cuda(v) for v in value.values())
    return False


class ComponentTrace(GraphTrace):
    """Fold shape/config-only CPU work (RoPE grids, arange, scalars) into constants. A CPU to GPU
    copy is recorded with its CPU operand as a constant; GPU computations are traced normally."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if not contains_cuda((args, kwargs)):
            result = func(*args, **kwargs)
            if not contains_cuda(result):
                return result
            self.nodes.append({"op": str(func), "args": self.encode(args), "kwargs": self.encode(kwargs),
                               "out": self.outputs(result)})
            return result
        return super().__torch_dispatch__(func, types, args, kwargs)


class ConstantTrace(GraphTrace):
    """Like ComponentTrace, but also admits literal empty CUDA factories and records the SDPA backend."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if (str(func) == "aten.lift_fresh.default" and isinstance(args[0], torch.Tensor)
                and args[0].device.type == "cuda" and id(args[0]) not in self.ids):
            # Transformers' empty KV cache and zero sequence counter are literal factories.
            value = args[0]
            with _disable_current_modes():
                if value.numel() > 1 or (value.numel() == 1 and value.item() != 0):
                    raise RuntimeError("unreviewed CUDA literal factory")
            key = self.name + f".constant.{len(self.externals)}"
            self.register(value, key)
            self.writer.tensor(key, value)
            self.externals[key]["factory"] = str(func)
        if not contains_cuda((args, kwargs)):
            result = func(*args, **kwargs)
            if not contains_cuda(result):
                return result
            self.nodes.append({"op": str(func), "args": self.encode(args), "kwargs": self.encode(kwargs),
                               "out": self.record_output(func, result)})
            return result
        backend = None
        if str(func) == "aten.scaled_dot_product_attention.default":
            with _disable_current_modes():
                backend = torch.ops.aten._fused_sdp_choice.default(*args, **kwargs)
        result = super().__torch_dispatch__(func, types, args, kwargs)
        if backend is not None:
            self.nodes[-1]["sdpa_backend"] = int(backend)
        return result


class CompactTrace(ConstantTrace):
    """Hold tensor identities weakly so long VAE traces keep normal liveness. A recycled Python id
    or scratch address is forgotten only after every observed view of its storage has died."""

    def __init__(self, *args, **kwargs):
        import weakref
        super().__init__(*args, **kwargs)
        owner = self

        class Identities:
            def __init__(self):
                self.count, self.refs, self.storages = 0, {}, {}

            def __len__(self):
                return self.count

            def append(self, tensor):
                self.count += 1
                key = id(tensor)
                if key in self.refs and self.refs[key]() is tensor:
                    return
                pointer = tensor.untyped_storage().data_ptr()
                self.storages[pointer] = self.storages.get(pointer, 0) + 1

                def died(reference):
                    if self.refs.get(key) is reference:
                        self.refs.pop(key)
                        owner.ids.pop(key, None)
                    self.storages[pointer] -= 1
                    if not self.storages[pointer]:
                        self.storages.pop(pointer)
                        owner.scratch_storages.discard(pointer)
                self.refs[key] = weakref.ref(tensor, died)
        identities = Identities()
        for value in self.keepalive:
            identities.append(value)
        self.keepalive = identities


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return repr(value)


class TritonRecorder:
    """Keep each compiled Triton kernel's CUBIN and metadata; inside a trace, add every launch to
    the graph as an `ltx25.triton` node. Launches themselves are unchanged."""

    def __init__(self, root):
        self.root = Path(root)
        (self.root / "cubin").mkdir(parents=True, exist_ok=True)
        self.by_function = {}
        self.launches = 0

    def install(self):
        from triton.backends.nvidia.driver import CudaLauncher
        from triton.compiler.compiler import CompiledKernel
        recorder = self
        original_init_handles = CompiledKernel._init_handles

        def _init_handles(kernel):
            original_init_handles(kernel)
            function = getattr(kernel, "function", None)
            if function is not None and function not in recorder.by_function:
                cubin = kernel.asm["cubin"]
                sha = hashlib.sha256(cubin).hexdigest()
                path = recorder.root / "cubin" / f"{sha}.cubin"
                if not path.exists():
                    path.write_bytes(cubin)
                meta = kernel.metadata._asdict() if hasattr(kernel.metadata, "_asdict") else dict(kernel.metadata)
                signature = getattr(kernel.src, "signature", {})
                recorder.by_function[function] = {
                    "sha256": sha, "name": meta.get("name"), "metadata": {k: _jsonable(v) for k, v in meta.items()},
                    "signature": {str(k): str(v) for k, v in signature.items()},
                    "signature_order": [str(v) for v in signature.values()],
                    "constants": {str(k): _jsonable(v) for k, v in getattr(kernel.src, "constants", {}).items()}}
        CompiledKernel._init_handles = _init_handles
        original_call = CudaLauncher.__call__

        def __call__(launcher, gridX, gridY, gridZ, stream, function, *args):
            recorder.launches += 1
            trace = ACTIVE_TRACE.get()
            if trace is None:
                return original_call(launcher, gridX, gridY, gridZ, stream, function, *args)
            info = recorder.by_function.get(function)
            if info is None:
                raise RuntimeError("graph launch has no recorded Triton CUBIN")
            kernel_args = args[4:]
            if len(info["signature_order"]) != len(kernel_args):
                raise RuntimeError("Triton graph argument/signature length differs")
            runtime_args = [a for a, t in zip(kernel_args, info["signature_order"]) if t != "constexpr"]
            encoded = trace.encode(runtime_args)
            result = original_call(launcher, gridX, gridY, gridZ, stream, function, *args)
            pointers = [a for a in runtime_args if isinstance(a, torch.Tensor)]
            trace.nodes.append({"op": "ltx25.triton", "args": encoded,
                                "kwargs": {"cubin_sha256": info["sha256"], "kernel_root": str(recorder.root),
                                           "grid": [gridX, gridY, gridZ]},
                                "out": trace.outputs(pointers, force=True)})
            return result
        CudaLauncher.__call__ = __call__

    def close(self):
        kernels = {v["sha256"]: {k: v[k] for k in ("name", "metadata", "signature", "signature_order", "constants")}
                   for v in self.by_function.values()}
        document = {"schema": "ltx25-triton-launches-v1", "kernels": kernels, "launch_count": self.launches}
        (self.root / "launches.json").write_text(json.dumps(document, indent=1) + "\n")
        return document
