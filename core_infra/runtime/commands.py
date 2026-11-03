"""Own pending commands attached to existing native residents.

Commands are sent before returning to the caller, so a later compute client
queues behind preload on a serial resident. No worker thread waits for a reply.
The model declares command order, readiness validation and when to send more
commands. The resident process owner must still retire native work on failure;
closing an RPC connection alone does not cancel a CUDA operation.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import time

try:
    from .native_control import NativeControlClient
except ImportError:  # Existing flat native controller deployments.
    from native_control import NativeControlClient


class PendingCommands:
    def __init__(self, directory, *, timeout=900, max_reply_bytes=65536):
        if timeout <= 0:
            raise ValueError('pending command timeout must be positive')
        self.directory = Path(directory)
        self.timeout, self.max_reply_bytes = timeout, max_reply_bytes
        self.pending, self.closed = {}, False

    def _check(self, name):
        if self.closed:
            raise RuntimeError('pending command group is closed')
        if not name or Path(name).name != name or name in ('.', '..'):
            raise ValueError('invalid pending command name')
        if name in self.pending:
            raise ValueError('duplicate pending command')

    def _retain(self, name, client, context, started):
        task = dict(context or {})
        task.update(client=client, connection=client.connection, buffer=client.buffer,
                    started=time.monotonic() if started is None else started)
        self.pending[name] = task
        return task

    def send(self, name, address, args, *, context=None):
        self._check(name)
        started = time.monotonic()
        client = NativeControlClient(address, args, self.directory / ('preload-' + name + '.log'),
                                     max_reply_bytes=self.max_reply_bytes)
        return self._retain(name, client, context, started)

    def adopt(self, name, connection, buffer, *, context=None, started=None):
        self._check(name)
        client = NativeControlClient.adopt(connection, buffer,
            self.directory / ('preload-' + name + '.log'), max_reply_bytes=self.max_reply_bytes)
        return self._retain(name, client, context, started)

    def poll(self, complete):
        if self.closed:
            raise RuntimeError('pending command group is closed')
        try:
            for name, task in list(self.pending.items()):
                if time.monotonic() - task['started'] > self.timeout:
                    raise TimeoutError('native ' + name + ' preload timed out')
                if task['client'].poll() is not None:
                    complete(name, task)
                    del self.pending[name]
        except BaseException:
            self.close()
            raise

    def close(self):
        if not self.closed:
            self.closed = True
            for task in self.pending.values():
                task['client'].close()
            self.pending.clear()


@dataclass(frozen=True)
class PreloadCommand:
    """One weight-only command on an existing serial native owner."""
    name: str
    component: str
    args: tuple
    validate: object
    after: tuple = ()


class CommandSchedule:
    """Execute a declared resident preload plan without waiting worker threads.

    The caller retains the native process tree and must retire it on failure.
    This owner sends/adopts each preload once, validates its receipt, and releases
    dependent commands. Conditions describe real external events, such as an
    encoder having consumed its input. They must return an actual boolean.
    """
    def __init__(self, directory, commands, *, conditions=None, timeout=900,
                 cached=None, completed=None, transport=None):
        commands = tuple(commands)
        self.directory = Path(directory)
        self.conditions = dict(conditions or {})
        self.commands = {c.name: c for c in commands}
        if not commands or len(self.commands) != len(commands):
            raise ValueError('declare unique resident preload commands')
        if (any(not isinstance(name, str) or not name or Path(name).name != name
                or not callable(check) for name, check in self.conditions.items())
                or self.conditions.keys() & self.commands.keys()):
            raise ValueError('invalid resident preload condition')
        seen = set(self.conditions)
        physical = {}
        for command in commands:
            if (not isinstance(command.name, str) or not command.name
                    or Path(command.name).name != command.name
                    or command.name in ('.', '..')
                    or not isinstance(command.component, str) or not command.component
                    or Path(command.component).name != command.component
                    or command.component in ('.', '..')
                    or not isinstance(command.args, tuple) or not command.args
                    or not callable(command.validate)
                    or not isinstance(command.after, tuple)
                    or len(set(command.after)) != len(command.after)
                    or any(name not in seen for name in command.after)):
                raise ValueError('invalid, unvalidated or unordered resident preload declaration')
            previous = physical.get(command.component)
            if previous is not None and previous not in command.after:
                raise ValueError('serial owner stages must declare their previous preload')
            physical[command.component] = command.name
            seen.add(command.name)
        if (cached is not None and not callable(cached)) or (completed is not None and not callable(completed)):
            raise ValueError('invalid resident preload receipt callback')
        self.cached, self.completed = cached, completed
        self.transport = transport if transport is not None else PendingCommands(directory, timeout=timeout)
        self.states = {name: 'waiting' for name in self.commands}
        self.receipts = {}
        self.events, self.started, self.closed = set(), False, False

    @property
    def pending(self):
        return self.transport.pending

    @property
    def ready(self):
        return all(state == 'ready' for state in self.states.values())

    def adopt(self, name, connection, buffer, *, started):
        """Admit one already-sent preload; never resend a borrowed command."""
        if self.started or self.closed or name not in self.commands or self.states[name] != 'waiting':
            raise RuntimeError('resident preload adoption must precede startup')
        command = self.commands[name]
        if command.after:
            raise ValueError('cannot adopt a preload before its declared dependencies')
        self.transport.adopt(name, connection, buffer, started=started,
            context={'component': command.component, 'mode': command.args[0]})
        self.states[name] = 'pending'

    def start(self):
        if self.started or self.closed:
            raise RuntimeError('resident preload schedule can only start once')
        self.started = True
        try:
            self._schedule()
        except BaseException:
            self.close()
            raise
        return self

    def _schedule(self):
        for name, check in self.conditions.items():
            if name not in self.events:
                result = check()
                if type(result) is not bool:
                    raise RuntimeError('resident preload condition must return bool: ' + name)
                if result:
                    self.events.add(name)
        for name, command in self.commands.items():
            if self.states[name] != 'waiting' or any(
                    dependency not in self.events and self.states.get(dependency) != 'ready'
                    for dependency in command.after):
                continue
            cached = self.cached(command) if self.cached is not None else None
            if cached is not None:
                task = {'component': command.component, 'mode': command.args[0],
                        'started': time.monotonic(), 'reused': True,
                        'buffer': json.dumps({'ok': True, 'details': cached}).encode()}
                self._complete(name, task)
            else:
                self.transport.send(name, self.directory / (command.component + '.sock'), command.args,
                    context={'component': command.component, 'mode': command.args[0]})
                self.states[name] = 'pending'

    def _complete(self, name, task):
        command = self.commands[name]
        reply = json.loads(task['buffer'])
        if reply.get('ok') is not True:
            raise RuntimeError('native ' + name + ' preload failed: ' + str(reply)[:2000])
        if command.validate(reply.get('details', {}), task) is not True:
            raise RuntimeError('resident preload receipt rejected: ' + name)
        if self.completed is not None:
            self.completed(name, task)
        self.receipts[name] = reply['details']
        self.states[name] = 'ready'

    def poll(self):
        if self.closed or not self.started:
            raise RuntimeError('resident preload polling requires an open started schedule')
        try:
            self.transport.poll(self._complete)
            self._schedule()
        except BaseException:
            self.close()
            raise

    def close(self):
        if not self.closed:
            self.closed = True
            self.transport.close()


class PendingProcessCommands:
    """The existing proxy-process transport under the same command scheduler.

    The caller retains process-tree ownership. This adapter never cancels or
    claims to retire GPU work by closing a proxy's parent log descriptor.
    """
    def __init__(self, directory, programs, environment, owner, *, timeout=900):
        self.directory, self.programs = Path(directory), dict(programs)
        self.environment, self.owner, self.timeout = environment, owner, timeout
        self.pending, self.closed = {}, False

    def send(self, name, address, args, *, context):
        if self.closed or name in self.pending:
            raise RuntimeError('proxy preload transport is closed or duplicate')
        component = context['component']
        if component not in self.programs:
            raise ValueError('preload proxy is not declared: ' + component)
        path = self.directory / ('preload-' + name + '.log')
        log = path.open('w')
        try:
            child = self.owner.spawn([self.programs[component], *args], self.environment, log)
        except BaseException:
            log.close(); raise
        self.pending[name] = dict(context, child=child, log=log, path=path, started=time.monotonic())

    def adopt(self, name, connection, buffer, *, context, started):
        if self.closed or name in self.pending:
            raise RuntimeError('proxy preload transport is closed or duplicate')
        direct=PendingCommands(self.directory,timeout=self.timeout)
        task=direct.adopt(name,connection,buffer,context=context,started=started)
        self.pending[name]=task

    def poll(self, complete):
        if self.closed:
            raise RuntimeError('proxy preload transport is closed')
        try:
            for name, task in list(self.pending.items()):
                if time.monotonic() - task['started'] > self.timeout:
                    raise TimeoutError('native ' + name + ' preload timed out')
                child = task.get('child')
                if child is None:
                    if task['client'].poll() is None:continue
                else:
                    if child.poll() is None:continue
                    self.owner.complete(child)
                    task['log'].close()
                    if child.returncode:
                        raise RuntimeError('native ' + name + ' preload failed: ' + task['path'].read_text()[-2000:])
                    task['buffer'] = task['path'].read_bytes()
                complete(name, task)
                del self.pending[name]
        except BaseException:
            self.close(); raise

    def close(self):
        if not self.closed:
            self.closed = True
            for task in self.pending.values():
                if 'client' in task:task['client'].close()
                else:task['log'].close()
            self.pending.clear()
