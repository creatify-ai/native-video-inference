"""Capture the input-dependent ATen dataflow of one eager call (offline preparation only).

Every tensor reaching an operator must have a known origin: a model parameter or buffer, a
declared literal, the declared input, or the output of an earlier operator. Operators whose
operands all derive from model state (weight normalization, filter expansion, exp of an
activation parameter) run eagerly; their results become constants once an input-dependent
operator consumes them. Each input-dependent operator becomes one node. An external
activation, an undeclared literal, an in-place update or a data-dependent result raises.
"""
import math

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.weak import WeakIdKeyDictionary

from .common import write_tensor

# Only output 0 of these is part of the graph; consuming another output raises.
AUXILIARY = frozenset({'aten.native_layer_norm.default', 'aten._scaled_dot_product_efficient_attention.default'})
# One node per returned view, with the output index appended to the node's args.
SPLIT = frozenset({'aten.unbind.int'})
LITERALS = frozenset({'aten.lift_fresh.default', 'aten.lift_fresh_copy.default'})
SCALARS = (torch.dtype, torch.device, torch.layout, torch.memory_format)


def _tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        return [t for item in value for t in _tensors(item)]
    if isinstance(value, dict):
        return [t for item in value.values() for t in _tensors(item)]
    return []


def _references(value):
    if isinstance(value, dict):
        if set(value) == {'tensor'}:
            yield value['tensor']
        else:
            for item in value.values():
                yield from _references(item)
    elif isinstance(value, list):
        for item in value:
            yield from _references(item)


def _rename(value, names):
    if isinstance(value, dict):
        if set(value) == {'tensor'}:
            return {'tensor': names.get(value['tensor'], value['tensor'])}
        return {key: _rename(item, names) for key, item in value.items()}
    if isinstance(value, list):
        return [_rename(item, names) for item in value]
    return value


class Capture(TorchDispatchMode):
    """`state`: {name: parameter or buffer}; `literals`: {name: CPU tensor} admitted through
    `torch.tensor(...)` when the created value equals one exactly; `sample`: the one input."""

    def __init__(self, sample, state, literals=None):
        super().__init__()
        self.origins = WeakIdKeyDictionary()  # tensor -> ('state', sources, direct) | ('node', id) | ('aux',)
        for name, tensor in state.items():
            self.origins[tensor] = ('state', frozenset({name}), True)
        self.literals = dict(literals or {})
        self.origins[sample] = ('node', 'input')
        self.nodes, self.values, self.constant_ids = [], {}, WeakIdKeyDictionary()

    def origin(self, tensor, op):
        found = self.origins.get(tensor)
        if found is None:
            raise RuntimeError(f'{op}: operand {list(tensor.shape)} {tensor.dtype} on {tensor.device} is neither '
                               'model state, a declared literal, the input nor a traced result')
        if found[0] == 'aux':
            raise RuntimeError(f'{op}: consumes an auxiliary output that the graph does not carry')
        return found

    def literal(self, value):
        candidate = value.detach().cpu()
        for name, expected in self.literals.items():
            if candidate.dtype == expected.dtype and candidate.shape == expected.shape and torch.equal(candidate, expected):
                return name
        raise RuntimeError(f'undeclared literal tensor {list(value.shape)} {value.dtype}')

    def encode(self, value):
        if isinstance(value, torch.Tensor):
            kind, *rest = self.origins[value]
            if kind == 'node':
                return {'tensor': rest[0]}
            key = self.constant_ids.get(value)
            if key is None:
                key = f'c{len(self.values):04}'
                # Snapshot at first use: later updates cannot change a recorded constant.
                self.values[key] = (value.detach().contiguous().to('cpu', copy=True), sorted(rest[0]), not rest[1])
                self.constant_ids[value] = key
            return {'tensor': key}
        if isinstance(value, (list, tuple)):
            return [self.encode(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self.encode(item) for key, item in value.items()}
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, SCALARS):
            return str(value)
        raise TypeError(f'unsupported operator argument {type(value).__name__}')

    def node(self, op, args, kwargs, tensor):
        key = f'n{len(self.nodes):04}'
        self.origins[tensor] = ('node', key)
        self.nodes.append({'id': key, 'op': op, 'args': args, 'kwargs': kwargs,
                           'output': {'dtype': str(tensor.dtype), 'shape': list(tensor.shape),
                                      'bytes': math.prod(tensor.shape) * tensor.element_size()}})

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        op = str(func)
        operands = _tensors((args, kwargs))
        if func._schema.is_mutable:
            raise RuntimeError(f'{op}: in-place or out= operators are not captured')
        if op in LITERALS and len(operands) == 1 and operands[0] not in self.origins:
            name = self.literal(operands[0])
            result = func(*args, **kwargs)
            self.origins[result] = ('state', frozenset({name}), True)
            return result
        origins = [self.origin(tensor, op) for tensor in operands]
        if not any(found[0] == 'node' for found in origins):
            result = func(*args, **kwargs)
            if origins:  # model-state-only: folded; factories stay unregistered and raise when consumed
                sources = frozenset().union(*(found[1] for found in origins))
                for tensor in _tensors(result):
                    self.origins[tensor] = ('state', sources, False)
            return result
        encoded_args, encoded_kwargs = self.encode(list(args)), self.encode(kwargs)
        result = func(*args, **kwargs)
        if isinstance(result, torch.Tensor):
            self.node(op, encoded_args, encoded_kwargs, result)
        elif op in SPLIT:
            for index, tensor in enumerate(result):
                self.node(op, encoded_args + [index], encoded_kwargs, tensor)
        elif op in AUXILIARY:
            self.node(op, encoded_args, encoded_kwargs, result[0])
            for tensor in _tensors(result[1:]):
                self.origins[tensor] = ('aux',)
        else:
            raise RuntimeError(f'{op}: unsupported input-dependent result {type(result).__name__}')
        return result

    def graph(self, output):
        """Nodes the output depends on, renumbered n0000..; constants c0000.. in first-use order."""
        found = self.origins.get(output)
        if found is None or found[0] != 'node' or found[1] == 'input':
            raise RuntimeError('the call result is not a traced node')
        needed = {found[1]}
        for node in reversed(self.nodes):
            if node['id'] in needed:
                needed.update(_references([node['args'], node['kwargs']]))
        kept = [node for node in self.nodes if node['id'] in needed]
        names = {node['id']: f'n{index:04}' for index, node in enumerate(kept)}
        used = {}
        for node in kept:
            for key in _references([node['args'], node['kwargs']]):
                if key in self.values and key not in used:
                    used[key] = f'c{len(used):04}'
        names.update(used)
        constants = {name: self.values[key] for key, name in used.items()}
        nodes = [dict(node, id=names[node['id']], args=_rename(node['args'], names),
                      kwargs=_rename(node['kwargs'], names)) for node in kept]
        return {'nodes': nodes, 'constants': dict(sorted(constants.items())), 'output_id': names[found[1]],
                'pruned_nodes': len(self.nodes) - len(kept)}


def write_constants(root, constants):
    """constants/<name>.raw for each folded value; manifest entries with per-tensor SHA-256."""
    specs = {}
    for name, (value, sources, folded) in constants.items():
        specs[name] = dict(write_tensor(root, f'constants/{name}.raw', value), derived_from=sources, folded=folded)
    return specs
