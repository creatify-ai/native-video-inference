"""Declared native components with owned preload, preparation and cancellation.

Models supply commands and dependencies. This module owns their scheduling;
it has no model names, tensor formats or model-specific environment switches.
"""
from dataclasses import dataclass, field
from concurrent.futures import Future
from pathlib import Path
import threading
import time

try:
    from . import ProcessGroup, ReaderBudget
    from .files import FileReadAhead, prefetch_stop_path
except ImportError:  # Exact same core sources in the flat serving image.
    from startup_ownership import ProcessGroup, ReaderBudget
    from startup_files import FileReadAhead, prefetch_stop_path


@dataclass(frozen=True)
class Component:
    name: str
    program: str
    preload: tuple
    environment: dict = field(default_factory=dict)
    # Rust DeclaredStartup resolves preceding components or static Preparations.
    after: tuple = ()
    validate: object = None
    readers: object = None
    read_ahead: bool = False
    prefetch: object = None
    program_sha256: str = None
    # None is an omitted declaration; () explicitly needs no preparation.
    required_preparations: tuple | None = None
    load_at_use: bool = False
    # Optional early native command. It starts readers/device/upload resources
    # before `after` is satisfied. The validated preload still gates readiness.
    initialize: tuple | None = None
    validate_initialize: object = None
    # Optional file-cache producer quota, independent of the native pinned
    # reader quota. Omitted values keep the shared foreground-reserving policy.
    prefetch_workers: int | None = None

    def __post_init__(self):
        if self.prefetch_workers is not None and (
                type(self.prefetch_workers) is not int or not 1 <= self.prefetch_workers <= 64
                or not callable(self.prefetch)):
            raise ValueError('component prefetch_workers requires a prefetch callback and 1..64 workers')


class ComponentStartup:
    def __init__(self, directory, environment, components, *, timeout=900, workers=8, native_policy=True,
                 reader_budget_mib=6144, preparation_contract='declared',
                 owner_factory=None, cached=None, completed=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.environment, self.timeout = dict(environment), timeout
        self.native_policy = native_policy
        for callback in (owner_factory, cached, completed):
            if callback is not None and not callable(callback):
                raise ValueError('invalid component transport callback')
        self.owner_factory, self.cached, self.completed = owner_factory, cached, completed
        self.components = {c.name: c for c in components}
        if len(self.components) != len(components):
            raise ValueError('duplicate startup component')
        # Admit the same effective quota that _preload will activate. Legacy
        # policy does not activate the common loader's implicit default.
        self.reader_budgets = {
            c.name: (c.readers or ReaderBudget(24)) if native_policy else c.readers
            for c in components}
        budgets = [b.validate() for b in self.reader_budgets.values() if b is not None]
        if sum(b.limit_mib for b in budgets) > reader_budget_mib:
            raise ValueError('component readers exceed container budget')
        # Require declaration order to be topological. No waiting worker can
        # occupy the executor before its prerequisites have been submitted.
        seen = set()
        for c in components:
            if any(n not in seen for n in c.after):
                raise ValueError('startup dependencies must precede their component')
            seen.add(c.name)
        if preparation_contract != 'declared':
            raise ValueError('component preparation contract must be declared')
        self.preparation_contract = preparation_contract
        self.required_preparations = {}
        for c in components:
            required = c.required_preparations
            if required is None:
                raise ValueError('component must declare required preparations: ' + c.name)
            if (not isinstance(required, (tuple, list))
                    or any(not isinstance(n, str) or not n or n != n.strip() for n in required)
                    or len(set(required)) != len(required)):
                raise ValueError('invalid required component preparations: ' + c.name)
            self.required_preparations[c.name] = tuple(required)
            if not callable(c.validate):
                raise ValueError('component must declare a preload receipt validator: ' + c.name)
        self.group = ProcessGroup(workers=workers)
        self.owners, self.futures, self.reports = {}, {}, {}
        self.owner_locks = {name: threading.Lock() for name in self.components}
        self.events, self.lock = [], threading.Lock()
        self.closed = False
        self.started, self.start_complete, self.start_error = False, False, None
        self.prefetch, self.prefetch_futures, self.prefetch_warnings = {}, [], {}
        # Reserve foreground capacity for device/init work. Optional cache
        # producers divide half the host-work allowance (at least one worker
        # per producer), rather than each starting a native-reader cohort.
        # Native reader quotas above remain unchanged.
        self.prefetch_workers = max(1, (workers // 2) // max(1, sum(bool(c.prefetch) for c in components)))
        self.preparations = {}
        self.invoked = set()

    def _event(self, component, phase):
        with self.lock:
            self.events.append(dict(component=component, phase=phase, monotonic=time.monotonic()))

    def prepare(self, name, function, *args, before=(), validate=None):
        """Prepare immutable/request metadata while weights load. Invocations
        of the named consumers wait for it; weight preload never waits for it.
        The shared process group owns errors, cancellation and final joining.
        Required preparation needs a model-supplied receipt validator that
        returns True. Validation precedes readiness; core knows no model fields.
        """
        consumers = tuple(before)
        if (not isinstance(name, str) or not name or name != name.strip()
                or (validate is not None and not callable(validate))):
            raise ValueError('invalid component preparation name or validator')
        gate = Future()
        with self.lock:
            if self.closed:
                raise RuntimeError('component startup is closed')
            if self.start_error is not None:
                raise RuntimeError('component startup submission failed') from self.start_error
            if (name in self.preparations or any(n not in self.components for n in consumers)
                    or any(n in self.invoked for n in consumers)):
                raise ValueError('invalid, duplicate or late component preparation')
            required_consumers = {consumer for consumer, required in self.required_preparations.items()
                                  if name in required}
            if required_consumers and (not required_consumers.issubset(consumers) or not callable(validate)):
                raise ValueError('required preparation must validate and gate all declared consumers: ' + name)
            # Publish before submission: a concurrent invoke must see either
            # this gate or the closed registration boundary, never a gap.
            self.preparations[name] = (consumers, gate, validate)
        def work():
            self._event(name, 'prepare_start')
            result = function(*args)
            if validate is not None and validate(result) is not True:
                raise RuntimeError('component preparation receipt rejected: ' + name)
            self._event(name, 'prepare_ready')
            return result
        try:
            future = self.group.submit(work)
        except BaseException as error:
            gate.set_exception(error)
            raise

        def completed(result):
            if result.cancelled():
                gate.cancel()
                return
            try:
                value = result.result()
            except BaseException as error:
                gate.set_exception(error)
            else:
                gate.set_result(value)

        future.add_done_callback(completed)
        # Preserve cancellation of queued work through the actual group future.
        return future

    def start(self):
        with self.lock:
            if self.closed or self.started:
                raise RuntimeError('component startup can only start once while open')
            self.started = True
        # Submission/callbacks must not run under the metadata lock: completed
        # dependencies can schedule their consumers synchronously here.
        try:
            for component in self.components.values():
                if component.prefetch:
                    self.prefetch_futures.append(self.group.submit(self._prefetch, component))
            for component in self.components.values():
                future = self._schedule(component)
                with self.lock:
                    self.futures[component.name] = future
        except BaseException as error:
            with self.lock:
                self.start_error = error
            raise
        with self.lock:
            self.start_complete = True
        return self

    def _schedule(self, component):
        """Waiting dependencies consume no executor slot. In particular,
        metadata preparation submitted by a request can overlap input IO even
        when all downstream components have already been declared.
        """
        if not component.after:
            return self.group.submit(self._preload, component)
        result = Future()
        remaining = len(component.after)
        guard = threading.Lock()

        def completed(work):
            if work.cancelled():
                result.cancel()
                return
            try:
                value = work.result()
            except BaseException as error:
                result.set_exception(error)
            else:
                result.set_result(value)

        def ready(dependency):
            nonlocal remaining
            with guard:
                if result.done():
                    return
                try:
                    dependency.result()
                except BaseException as error:
                    result.set_exception(error)
                    return
                remaining -= 1
                if remaining:
                    return
            try:
                work = self.group.submit(self._preload, component)
            except BaseException as error:
                result.set_exception(error)
            else:
                work.add_done_callback(completed)

        for name in component.after:
            self.futures[name].add_done_callback(ready)
        return result

    def _prefetch(self, component):
        try:
            jobs = component.prefetch()
            stop = prefetch_stop_path(self.directory,component.name)
            if jobs and not stop.exists():
                workers = component.prefetch_workers
                if workers is None:
                    workers = min(self.prefetch_workers,
                        self.reader_budgets[component.name].workers if self.reader_budgets[component.name] else self.prefetch_workers)
                self.prefetch[component.name] = FileReadAhead(self.group, jobs,
                    self.directory / ('prefetch-' + component.name), self.environment,
                    workers=workers)
        except Exception as error:
            self.prefetch_warnings[component.name] = str(error)

    def _preload(self, component):
        for dependency in component.after:
            self.ready(dependency)
        report = self._load_transport(component)
        if component.validate(report) is not True:
            raise RuntimeError('component preload receipt rejected: ' + component.name)
        self._publish_transport(component, report)
        return report

    def _owner_transport(self, component):
        with self.owner_locks[component.name]:
            with self.lock:
                if self.closed:
                    raise RuntimeError('component transport is closed')
                owner = self.owners.get(component.name)
            if owner is not None:
                return owner
            owner = self._create_transport(component)
            with self.lock:
                self.owners[component.name] = owner
            return owner

    def _create_transport(self, component):
        environment = {**self.environment, **component.environment}
        if self.native_policy:
            budget = self.reader_budgets[component.name]
            direct = environment.get('NATIVE_LOAD_DIRECT', '1')
            if direct not in ('0', '1'):
                raise ValueError('NATIVE_LOAD_DIRECT must be 0 or 1')
            environment.update(budget.startup_environment(direct=direct == '1',
                verify=environment.get('NATIVE_LOAD_VERIFY', 'full'),
                guard_manifest=environment.get('NATIVE_LOAD_GUARD_MANIFEST'),
                guard_sha256=environment.get('NATIVE_LOAD_GUARD_SHA256')))
            environment.update(
                NATIVE_COMPONENT_READ_AHEAD=str(int(component.read_ahead)),
                NATIVE_LOAD_PREFETCH_STOP_FILE=str(prefetch_stop_path(self.directory,component.name)))
        if self.owner_factory is None:
            owner = self.group.owner(component.program, self.directory / (component.name + '.sock'),
                environment, self.directory / (component.name + '.log'),
                timeout=self.timeout, sha256=component.program_sha256)
        else:
            owner = self.owner_factory(component, environment, self.group)
        return owner

    def _initialize_transport(self, component):
        self._event(component.name, 'initialize_start')
        return self._owner_transport(component).invoke(component.initialize)

    def _load_transport(self, component):
        """Finish one admitted load on its existing initialized owner, or create
        the owner for the ordinary one-phase contract. Rust owns dependencies."""
        self._event(component.name, 'start')
        owner = self._owner_transport(component)
        report = self.cached(component) if self.cached is not None else None
        if report is None:
            try:
                report = owner.invoke(component.preload)
            except BaseException as error:
                raise RuntimeError('native ' + component.name + ' preload failed: ' + str(error)) from error
        return report

    def _publish_transport(self, component, report):
        with self.lock:
            self.reports[component.name] = report
        if self.completed is not None:
            self.completed(component.name, report)
        self._event(component.name, 'ready')

    def ready(self, name):
        """Join weight preload only; execution also requires invoke's gates."""
        with self.lock:
            if self.closed or not self.started:
                raise RuntimeError('component weight readiness requires an open started load')
            if self.start_error is not None:
                raise RuntimeError('component startup submission failed') from self.start_error
            future = self.futures.get(name)
            if future is None:
                raise RuntimeError('component weight preload has not been submitted')
        return future.result()

    def invoke(self, name, args):
        with self.lock:
            if self.start_error is not None:
                raise RuntimeError('component startup submission failed') from self.start_error
            if self.closed or not self.start_complete:
                raise RuntimeError('component invocation requires an open completed startup submission')
            if name not in self.components:
                raise ValueError('unknown startup component')
            for required in self.required_preparations[name]:
                entry = self.preparations.get(required)
                if entry is None or name not in entry[0] or not callable(entry[2]):
                    raise RuntimeError('required component preparation is missing: ' + required)
            self.invoked.add(name)
            preparations = tuple(self.preparations.values())
        self.ready(name)
        for consumers, preparation, _ in preparations:
            if name in consumers:
                preparation.result()
        return self.owners[name].invoke(args)

    def close(self):
        with self.lock:
            if self.closed:
                return []
            self.closed = True
        for name in self.components:
            prefetch_stop_path(self.directory,name).touch()
        return self.group.close()
