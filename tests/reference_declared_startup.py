"""Frozen Python reference for testing the Rust executor's observable contract.

Models provide immutable work declarations and receipt validators. Inputs are
facts, never scheduling commands. The framework starts independent preparation
and loads together, and gates every consumer on its declared prerequisites.
"""
from concurrent.futures import Future, InvalidStateError
from dataclasses import dataclass, replace
import threading

from core_infra.runtime.components import ComponentStartup


@dataclass(frozen=True)
class Preparation:
    name: str
    before: tuple
    run: object
    validate: object
    inputs: tuple = ()
    prepare: object = None
    validate_prepared: object = None
    weights: tuple = ()
    at_use: object = False


class DeclaredStartup:
    def __init__(self, directory, environment, components, preparations, *, consumers=None, **load_options):
        components, preparations = tuple(components), tuple(preparations)
        names = {c.name for c in components}
        definitions = {p.name: p for p in preparations}
        if not components or len(names) != len(components) or len(definitions) != len(preparations):
            raise ValueError('startup needs unique components and preparations')
        required, seen = {}, set()
        for component in components:
            if (not isinstance(component.after, tuple) or any(name not in seen for name in component.after)):
                raise ValueError('startup dependencies must precede their component')
            seen.add(component.name)
            ids = component.required_preparations
            if (not callable(component.validate) or not isinstance(ids, tuple) or len(set(ids)) != len(ids)
                    or any(not isinstance(n, str) or not n for n in ids)):
                raise ValueError('component must explicitly declare required preparations: ' + component.name)
            required[component.name] = ids
        for preparation in preparations:
            if (not isinstance(preparation.name, str) or not preparation.name
                    or not isinstance(preparation.before, tuple) or not preparation.before
                    or len(set(preparation.before)) != len(preparation.before)
                    or any(n not in names for n in preparation.before)
                    or not callable(preparation.run) or not callable(preparation.validate)
                    or not isinstance(preparation.inputs, tuple)
                    or len(set(preparation.inputs)) != len(preparation.inputs)
                    or any(not isinstance(n, str) or not n or n == 'prepared' for n in preparation.inputs)
                    or not isinstance(preparation.weights, tuple)
                    or len(set(preparation.weights)) != len(preparation.weights)
                    or any(n not in names for n in preparation.weights)
                    or (preparation.prepare is not None and
                        (not callable(preparation.prepare) or not callable(preparation.validate_prepared)))
                    or (preparation.prepare is None and preparation.validate_prepared is not None)
                    or (type(preparation.at_use) is not bool and not callable(preparation.at_use))
                    or (not preparation.inputs and preparation.at_use is not False)):
                raise ValueError('invalid declared startup preparation')
            declared_consumers = {name for name, ids in required.items() if preparation.name in ids}
            if declared_consumers != set(preparation.before):
                raise ValueError('preparation consumers differ from the component declaration: ' + preparation.name)
        if {name for ids in required.values() for name in ids} != set(definitions):
            raise ValueError('missing or extra required startup preparation')
        # A registered shared component is a view of this same admitted plan.
        # Core computes its dependency closure; entry adapters cannot hand-code
        # another preload order or omit a required initializer.
        if consumers is not None:
            if (not isinstance(consumers, tuple) or not consumers or len(set(consumers)) != len(consumers)
                    or any(name not in names for name in consumers)):
                raise ValueError('declare existing unique startup consumers')
            selected, previous = set(consumers), set()
            by_name = {c.name: c for c in components}
            while selected != previous:
                previous = set(selected)
                selected.update(dependency for name in previous for dependency in by_name[name].after)
                ids = {p for name in selected for p in required[name]}
                selected.update(weight for name in ids for weight in definitions[name].weights)
            components = tuple(c for c in components if c.name in selected)
            ids = {p for name in selected for p in required[name]}
            definitions = {name: replace(p,before=tuple(n for n in p.before if n in selected))
                           for name,p in definitions.items() if name in ids}
            required = {name: ids for name,ids in required.items() if name in selected}
            names = selected
        # Admission is complete before creating owners, threads or doing IO.
        self.definitions, self.required_preparations = definitions, required
        self.preparation_contract = 'declared'
        self.components = {c.name: c for c in components}
        self.loads = ComponentStartup(directory, environment,
            tuple(replace(c, required_preparations=()) for c in components), **load_options)
        self.group, self.owners = self.loads.group, self.loads.owners
        self.events, self.lock = self.loads.events, threading.RLock()
        self.static, self.scope = {}, None
        self.use_signals = {}
        self.weights = {name: Future() for name in names}
        self.started = self.closed = False

    @staticmethod
    def _relay(result, gate):
        if gate.done():return
        try:
            if result.cancelled():
                gate.cancel()
                return
            try:
                value = result.result()
            except BaseException as error:
                gate.set_exception(error)
            else:
                gate.set_result(value)
        except InvalidStateError:
            # Cancellation can win between done() and the completed callback.
            if not gate.cancelled():raise

    def _submit(self, definition, inputs):
        gate = Future()
        at_use = definition.at_use(inputs) if callable(definition.at_use) else definition.at_use
        if type(at_use) is not bool:
            raise ValueError('preparation dependency must return bool: ' + definition.name)
        cpu = None
        if definition.prepare is not None:
            def prepare():
                self.loads._event(definition.name, 'cpu_prepare_start')
                value = definition.prepare(inputs)
                if definition.validate_prepared(value) is not True:
                    raise RuntimeError('startup CPU preparation receipt rejected: ' + definition.name)
                self.loads._event(definition.name, 'cpu_prepare_ready')
                return value
            cpu = self.group.submit(prepare)
            gate.add_done_callback(lambda result: cpu.cancel() if result.cancelled() else None)
        dependencies = (tuple(self.weights[n] for n in definition.weights)
            + ((cpu,) if cpu is not None else ())
            + ((self.use_signals[definition.name],) if at_use else ()))
        remaining, guard = len(dependencies), threading.Lock()
        def work():
            self.loads._event(definition.name, 'prepare_start')
            values = inputs if cpu is None else {**inputs, 'prepared': cpu.result()}
            value = definition.run(values)
            if definition.validate(value) is not True:
                raise RuntimeError('startup preparation receipt rejected: ' + definition.name)
            self.loads._event(definition.name, 'prepare_ready')
            return value
        def submit():
            if gate.done():return
            try:
                pending = self.group.submit(work)
            except BaseException as error:
                gate.set_exception(error)
            else:
                gate.add_done_callback(lambda result: pending.cancel() if result.cancelled() else None)
                pending.add_done_callback(lambda result: self._relay(result, gate))
        def ready(dependency):
            nonlocal remaining
            with guard:
                if gate.done():return
                if dependency.cancelled():
                    gate.cancel();return
                try:
                    dependency.result()
                except BaseException as error:
                    gate.set_exception(error);return
                remaining -= 1
                if remaining:return
            submit()
        if dependencies:
            for dependency in dependencies:dependency.add_done_callback(ready)
        else:
            submit()
        return gate

    def start(self):
        with self.lock:
            if self.started or self.closed:
                raise RuntimeError('declared startup can only start once while open')
            self.started = True
            for name, definition in self.definitions.items():
                if not definition.inputs:
                    self.static[name] = self._submit(definition, {})
        try:
            self.loads.start()
            for name, future in self.loads.futures.items():
                future.add_done_callback(lambda result, name=name: self._relay(result, self.weights[name]))
        except BaseException as error:
            for future in self.weights.values():
                if not future.done():future.set_exception(error)
            raise
        return self

    def request(self, inputs):
        """Publish one request's admitted facts; core schedules all required work."""
        with self.lock:
            if not self.started or self.closed:
                raise RuntimeError('request preparation requires an open started plan')
            expected = {name for p in self.definitions.values() for name in p.inputs}
            if not isinstance(inputs, dict) or set(inputs) != expected:
                raise ValueError('request inputs differ from the startup declaration')
            if self.scope is not None:
                for future in self.scope.values():
                    if not future.done():raise RuntimeError('previous request preparation is still active')
                    future.result()
            self.scope = dict(self.static)
            self.use_signals = {name: Future() for name, p in self.definitions.items() if p.inputs}
            for name, definition in self.definitions.items():
                if definition.inputs:
                    self.scope[name] = self._submit(definition, {n: inputs[n] for n in definition.inputs})
        return {name: future for name, future in self.scope.items()}

    def _gates(self, name):
        with self.lock:
            if self.closed or not self.started or not self.loads.start_complete:
                raise RuntimeError('component use requires an open started declaration')
            if name not in self.required_preparations:
                raise ValueError('unknown declared startup component: ' + str(name))
            available = self.scope if self.scope is not None else self.static
            missing = [n for n in self.required_preparations[name] if n not in available]
            if missing:
                raise RuntimeError('required startup request input is missing: ' + ', '.join(missing))
            for required in self.required_preparations[name]:
                signal = self.use_signals.get(required)
                if signal is not None and not signal.done():signal.set_result(True)
            return tuple(available[n] for n in self.required_preparations[name])

    def check_before_use(self, name):
        """Nonblocking transport adapter. Errors are never reported as waiting."""
        gates = self._gates(name)
        weight = self.loads.futures[name]
        pending = False
        for future in (weight, *gates):
            if future.done():
                future.result()
            else:
                pending = True
        return not pending

    def before_use(self, name):
        gates = self._gates(name)
        self.loads.ready(name)
        for future in gates:
            future.result()

    def invoke(self, name, args):
        self.before_use(name)
        return self.owners[name].invoke(args)

    def ready(self, name):
        return self.loads.ready(name)

    @property
    def reports(self):
        return self.loads.reports

    @property
    def prefetch(self):
        return self.loads.prefetch

    @property
    def prefetch_warnings(self):
        return self.loads.prefetch_warnings

    @property
    def weights_ready(self):
        return self.loads.start_complete and all(f.done() for f in self.loads.futures.values())

    @property
    def preload_ready(self):
        """Weights and all request-independent init; not request readiness."""
        self.poll()
        return self.weights_ready and all(f.done() for f in self.static.values())

    @property
    def states(self):
        return {name: ('ready' if future.done() else 'pending' if future.running() else 'waiting')
                for name, future in self.loads.futures.items()}

    def poll(self):
        for future in (*self.loads.futures.values(), *(self.scope or self.static).values()):
            if future.done():
                future.result()

    def close(self):
        with self.lock:
            self.closed = True
        result = self.loads.close()
        for future in (*self.use_signals.values(), *(self.scope or self.static).values(), *self.weights.values()):
            if not future.done():future.cancel()
        return result
