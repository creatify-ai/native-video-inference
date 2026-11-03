"""Python bindings to core's Rust Component/Preparation/DeclaredStartup policy.

Only model functions, declarations and existing transports are bound here.
All dependency release, work queues, result state, scopes and gates live in Rust.
An absent native runtime is an error; there is no Python-executor fallback.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import uuid
from concurrent.futures import CancelledError
from dataclasses import replace

try:
    from .components import ComponentStartup, ReaderBudget
    from .startup_spec import Preparation
except ImportError:
    from startup_components import ComponentStartup, ReaderBudget
    from startup_spec import Preparation


_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p))
_LIBRARIES = {}
_LIBRARY_LOCK = threading.Lock()


def _library(path, sha256=None):
    path = Path(path or os.environ.get('NATIVE_STARTUP_LIBRARY', 'libnative_core_infra.so'))
    with _LIBRARY_LOCK:
        key = (str(path), sha256)
        if key not in _LIBRARIES:
            if sha256 is not None:
                with path.open('rb') as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() != sha256:
                        raise ValueError('native startup library SHA-256 mismatch')
            lib = ctypes.CDLL(str(path))
            lib.native_startup_create.argtypes = [ctypes.c_char_p, _CALLBACK, ctypes.c_void_p]
            lib.native_startup_create.restype = ctypes.c_void_p
            lib.native_startup_call.argtypes = [ctypes.c_uint64, ctypes.c_char_p]
            lib.native_startup_call.restype = ctypes.c_void_p
            lib.native_startup_free.argtypes = [ctypes.c_void_p]
            lib.native_startup_free.restype = None
            lib.native_startup_reply.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
            lib.native_startup_reply.restype = None
            _LIBRARIES[key] = lib
        return _LIBRARIES[key]


class NativeFuture:
    """A handle to a Rust result, with no Python future or readiness cache."""
    def __init__(self, startup, node, generation=0):
        self.startup = startup
        if isinstance(node, dict):self.node, self.generation = node['node'],node['generation']
        else:self.node, self.generation = node,generation

    def _state(self):
        nodes = self.startup._call('snapshot')['nodes']
        if self.node >= len(nodes) or nodes[self.node]['generation'] != self.generation:
            raise RuntimeError('startup result belongs to an expired request scope')
        return nodes[self.node]

    def done(self): return self._state()['done']
    def running(self): return self._state()['running']
    def cancelled(self): return self._state()['cancelled']
    def cancel(self): return self.startup._call('cancel', node=self.node, generation=self.generation)

    def result(self, timeout=None):
        return self.startup._restore(self.startup._call('result', node=self.node, generation=self.generation, wait=True, timeout=timeout))


class DeclaredStartup:
    executor = 'rust'

    def __init__(self, directory, environment, components, preparations, *, consumers=None,
                 library=None, library_sha256=None, **load_options):
        self.lock, self._guard = threading.RLock(), threading.RLock()
        self._objects, self._errors, self._publishers = {}, {}, {}
        self._request_objects = set()
        self._next_object = 0
        self._scope_local = threading.local()
        self._closed, self._handle, self._cleanup = False, None, []
        self._created = time.monotonic()
        self.library = _library(library, library_sha256)
        components, preparations = tuple(components), tuple(preparations)
        self.components = {c.name: c for c in components}
        self.definitions = {p.name: p for p in preparations}
        native_policy = load_options.get('native_policy', True)
        declarations, bindings = [], []
        def bind(operation, name, function):
            if callable(function): bindings.append(dict(operation=operation, name=name))
        for c in components:
            quota = (c.readers or ReaderBudget(24)) if native_policy else c.readers
            if quota is not None: quota.validate()
            declarations.append(dict(name=c.name, after=c.after,
                required_preparations=c.required_preparations,
                reader_limit_mib=quota.limit_mib if quota is not None else 0, prefetch=bool(c.prefetch),
                load_at_use=bool(c.load_at_use), initialize=c.initialize is not None))
            if c.initialize is not None:
                if (not isinstance(c.initialize, (tuple, list)) or not c.initialize
                        or not all(isinstance(x, str) for x in c.initialize)
                        or not c.initialize[0].strip() or c.initialize[0] != c.initialize[0].strip()
                        or not callable(c.validate_initialize)):
                    raise ValueError('component initializer requires a command and receipt validator')
                bind('component_initialize', c.name, True.__bool__)
                bind('validate_component_initialize', c.name, c.validate_initialize)
            elif c.validate_initialize is not None:
                raise ValueError('component initializer validator requires an initializer')
            bind('load', c.name, True.__bool__)
            bind('invoke', c.name, True.__bool__)
            bind('validate_load', c.name, c.validate)
            if c.prefetch: bind('prefetch', c.name, c.prefetch)
        for p in preparations:
            bind('run', p.name, p.run); bind('validate_run', p.name, p.validate)
            if p.initialize is not None:
                if not callable(p.initialize) or not callable(p.validate_initialize):
                    raise ValueError('initializer requires initialize and validate_initialize callbacks')
                bind('initialize', p.name, p.initialize);bind('validate_initialize', p.name, p.validate_initialize)
            elif p.validate_initialize is not None:
                raise ValueError('initializer validator requires an initializer')
            if p.prepare is not None:
                bind('cpu', p.name, p.prepare); bind('validate_cpu', p.name, p.validate_prepared)
            elif p.validate_prepared is not None:
                raise ValueError('CPU validator requires CPU preparation')
            if callable(p.at_use): bind('at_use', p.name, p.at_use)
            elif type(p.at_use) is not bool: raise ValueError('invalid at-use dependency')
        bindings.append(dict(operation='cancel', name=''))
        spec = dict(declaration=dict(components=declarations,
            preparations=[dict(name=p.name, before=p.before, inputs=p.inputs, weights=p.weights, after=p.after,
                cpu=p.prepare is not None, at_use='predicate' if callable(p.at_use) else 'always' if p.at_use else 'never',
                serial_group=p.serial_group, initialize=p.initialize is not None)
                for p in preparations], consumers=consumers, workers=load_options.get('workers', 8),
            reader_budget_mib=load_options.get('reader_budget_mib', 6144),
            timeout_seconds=load_options.get('timeout', 900)), bindings=bindings)
        self._callback = _CALLBACK(self._dispatch)
        # Rust admits the full plan before ComponentStartup creates even a directory.
        try:
            admitted = self._decode(self.library.native_startup_create(json.dumps(spec).encode(), self._callback, None))
        except RuntimeError as error:
            raise ValueError(str(error)) from error
        self._handle = admitted['handle']
        self._created = time.monotonic() - admitted['elapsed_seconds']
        if admitted['abi'] != 5:
            self._call('destroy');self._handle = None
            raise ValueError('incompatible Rust startup runtime ABI')
        needed=set()
        if any(not p.before and p.at_use is True for p in preparations):needed.add('preparation_release')
        if any(p.serial_group is not None for p in preparations):needed.add('preparation_serial_group')
        if any(p.initialize is not None for p in preparations):needed.add('preparation_initializer')
        if any(n not in {c.name for c in components} for c in components for n in c.after):
            needed.add('component_after_preparation')
        if any(c.initialize is not None for c in components):needed.add('component_initializer')
        if not needed <= set(admitted.get('capabilities',())):
            self._call('destroy');self._handle=None
            raise ValueError('Rust startup runtime lacks requested preparation capabilities')
        self.native_declaration = admitted['declaration']
        selected = {c['name'] for c in self.native_declaration['components']}
        self.components = {n:c for n,c in self.components.items() if n in selected}
        ids = {p['name'] for p in self.native_declaration['preparations']}
        self.definitions = {n:p for n,p in self.definitions.items() if n in ids}
        self.required_preparations = {c['name']:tuple(c['required_preparations']) for c in self.native_declaration['components']}
        self.preparation_contract = 'declared'
        try:
            self.loads = ComponentStartup(directory, environment,
                tuple(replace(c, after=(), required_preparations=()) for c in self.components.values()), **load_options)
            self.group, self.owners = self.loads.group, self.loads.owners
        except BaseException:
            self._call('destroy'); self._handle = None
            raise

    def _keep(self, value, *, permanent=False):
        with self._guard:
            self._next_object += 1
            self._objects[self._next_object] = value
            if not permanent:self._request_objects.add(self._next_object)
            return {'python_value': self._next_object}

    def _restore(self, value):
        if isinstance(value, dict) and set(value) == {'python_value'}:
            with self._guard: return self._objects[value['python_value']]
        if isinstance(value, dict):return {n:self._restore(v) for n,v in value.items()}
        if isinstance(value, list):return [self._restore(v) for v in value]
        return value

    def _dispatch(self, _context, raw, out):
        try:
            message = json.loads(raw)
            operation, name, inputs = message['operation'], message['name'], message['inputs']
            if operation == 'load':
                value = self._keep(self.loads._load_transport(self.components[name]), permanent=True)
            elif operation == 'component_initialize':
                value = self._keep(self.loads._initialize_transport(self.components[name]), permanent=True)
            elif operation == 'validate_component_initialize':
                value = self.components[name].validate_initialize(self._restore(inputs))
                if value is True:self.loads._event(name, 'initialize_ready')
            elif operation == 'invoke':
                value = self._keep(self.owners[name].invoke(inputs))
            elif operation == 'validate_load':
                receipt = self._restore(inputs)
                value = self.components[name].validate(receipt)
                if value is True: self.loads._publish_transport(self.components[name], receipt)
            elif operation == 'prefetch':
                value = self._keep(self.loads._prefetch(self.components[name]), permanent=True)
            elif operation == 'cancel':
                if hasattr(self, 'loads'): self._cleanup = self.loads.close()
                value = None
            elif operation == 'publish':
                with self._guard: receive = self._publishers[name]
                self._scope_local.value = inputs['scope']
                try: receive(inputs['scope'], inputs['inputs'])
                finally: del self._scope_local.value
                value = None
            elif operation.startswith('validate_'):
                p = self.definitions[name]
                validate = p.validate_prepared if operation == 'validate_cpu' else p.validate_initialize if operation == 'validate_initialize' else p.validate
                value = validate(self._restore(inputs))
            else:
                p = self.definitions[name]
                data = {n:self._restore(v) for n,v in inputs.items()}
                if operation == 'at_use': value = p.at_use(data)
                else: value = self._keep((p.prepare if operation == 'cpu' else p.initialize if operation == 'initialize' else p.run)(data), permanent=not p.inputs)
            response = dict(ok=True, value=value)
        except BaseException as error:
            with self._guard:
                key = 'native-callback-' + str(len(self._errors))
                self._errors[key] = error
            response = dict(ok=False, error=key + ': ' + str(error))
        self.library.native_startup_reply(out, json.dumps(response).encode())

    def _decode(self, pointer):
        if not pointer: raise RuntimeError('native startup returned no response')
        try: response = json.loads(ctypes.string_at(pointer))
        finally: self.library.native_startup_free(pointer)
        if response.get('ok') is not True:
            message = response.get('error', 'native startup failed')
            with self._guard:
                for key, error in self._errors.items():
                    if message.startswith(key + ':'): raise error
            if message.startswith(('unknown declared startup component:', 'request inputs differ')):
                raise ValueError(message)
            if message == 'startup work cancelled': raise CancelledError(message)
            raise RuntimeError(message)
        return response['value']

    def _call(self, operation, **fields):
        return self._decode(self.library.native_startup_call(self._handle, json.dumps(dict(operation=operation, **fields)).encode()))

    def start(self):
        state = self._call('start')
        self.loads.started = self.loads.start_complete = True
        self.loads.futures = {n:NativeFuture(self, node) for n,node in state['weights'].items()}
        return self

    @property
    def static(self): return {n:NativeFuture(self, node) for n,node in self._call('snapshot')['static'].items()}

    @property
    def initializers(self):
        return {n:NativeFuture(self, node) for n,node in self._call('snapshot').get('initializers', {}).items()}

    @property
    def scope(self):
        state = self._call('snapshot')
        return None if state['scope'] is None else {n:NativeFuture(self,node,state['nodes'][node]['generation'])
            for n,node in {**state['static'],**state['request']}.items()}

    def request(self, inputs):
        scope = getattr(self._scope_local, 'value', None) or uuid.uuid4().hex
        with self._guard:
            previous = self._request_objects
            self._request_objects = set()
            try:
                nodes = self._call('request', scope=scope, inputs={n:self._keep(v) for n,v in inputs.items()})
            except BaseException:
                for key in self._request_objects:self._objects.pop(key, None)
                self._request_objects = previous
                raise
            for key in previous:self._objects.pop(key, None)
        return {n:NativeFuture(self,node) for n,node in nodes.items()}

    def release_preparation(self, name):
        """Release one independent static Preparation; Rust owns its future.

        The caller must validate the external event (such as the native CUDA
        first-compute marker). This call dispatches work without waiting.
        """
        return NativeFuture(self, self._call('release_preparation', name=name))

    def check_before_use(self, name): return self._call('check_before_use', name=name)
    def before_use(self, name): self._call('before_use', name=name)

    def invoke(self, name, args):
        return self._restore(self._call('invoke', name=name, inputs=args))

    def ready(self, name): return self.loads.futures[name].result()
    def poll(self): self._call('poll')

    def snapshot(self):
        """One consistent native state read for a transport's polling cycle."""
        return self._call('snapshot')

    def weights_ready_for(self, names):
        """Observe a declared weight scope without demanding deferred outputs.

        This is not compute readiness: before_use still owns preparation gates.
        Rust owns every node, receipt validation, error and readiness fact.
        """
        names = tuple(names)
        if not names or len(names) != len(set(names)):
            raise ValueError('weight readiness requires distinct component names')
        state = self._call('snapshot')
        if any(name not in state['weights'] for name in names):
            raise ValueError('weight readiness names an undeclared component')
        self.poll()
        return all(state['nodes'][state['weights'][name]]['done']
            and not state['nodes'][state['weights'][name]]['cancelled'] for name in names)

    @property
    def weights_ready(self): return self._call('snapshot')['weights_ready']
    @property
    def closed(self): return self._call('snapshot')['closed']
    @property
    def started(self): return self._call('snapshot')['started']
    @property
    def preload_ready(self):
        self.poll()
        return self._call('snapshot')['preload_ready']
    @property
    def states(self):
        state = self._call('snapshot')
        return {name:state['nodes'][node]['state'] for name,node in state['weights'].items()}
    @property
    def events(self):
        return [dict(component=e['component'], phase=e['phase'], monotonic=self._created+e['elapsed_seconds'])
                for e in self._call('snapshot')['events']]
    @property
    def reports(self): return self.loads.reports
    @property
    def prefetch(self): return self.loads.prefetch
    @property
    def prefetch_warnings(self): return self.loads.prefetch_warnings

    def close(self):
        if not self._closed and self._handle is not None:
            self._closed = True
            self._call('close')
        return self._cleanup

    def __del__(self):
        if getattr(self, '_handle', None) is not None:
            try:
                self.close()
                self._call('destroy')
                self._handle = None
            except BaseException:
                pass


class StartupGate:
    """Native gate server: Python binds request admission once, never readiness."""
    def __init__(self, startup, directory, receive, *, timeout=900):
        self.startup, self.path = startup, Path(directory)/('startup-'+uuid.uuid4().hex[:12]+'.sock')
        self.token, self.closed = uuid.uuid4().hex, False
        with startup._guard: startup._publishers[self.token] = receive
        try: self.environment = startup._call('gate_open', path=str(self.path), token=self.token)
        except BaseException:
            with startup._guard: startup._publishers.pop(self.token, None)
            raise

    def poll(self):
        self.startup._call('gate_poll', name=self.token)

    @property
    def consumers(self):
        return {tuple(c) for c in self.startup._call('gate_status', name=self.token)['consumers']}

    def close(self):
        if not self.closed:
            self.closed = True
            try: self.startup._call('gate_close', name=self.token)
            finally:
                with self.startup._guard: self.startup._publishers.pop(self.token, None)
