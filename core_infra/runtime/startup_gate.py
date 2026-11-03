"""Bounded request-owned adapter to the common startup consumer gates.

This module transports input facts and before-use checks. It cannot choose or
skip preparation: the supplied declared executor decides readiness.
"""
import json
from pathlib import Path
import socket
import time
import uuid

SCHEMA = 'native-startup-gate-v1'
ENVIRONMENT_KEY = 'NATIVE_STARTUP_GATE'
MAX_FRAME_BYTES = 65536


class StartupGate:
    def __init__(self, directory, receive, before_use, *, timeout=900, command_consumer=None):
        if not callable(receive) or not callable(before_use) or timeout <= 0:
            raise ValueError('startup gate needs input and readiness adapters')
        self.path = Path(directory) / ('startup-' + uuid.uuid4().hex[:12] + '.sock')
        if not self.path.is_absolute() or len(str(self.path).encode()) >= 104:
            raise ValueError('startup gate needs a short absolute socket path')
        self.token, self.receive, self.before_use = uuid.uuid4().hex, receive, before_use
        if command_consumer is not None and not callable(command_consumer):
            raise ValueError('startup command consumer must be callable')
        self.command_consumer = command_consumer
        self.timeout, self.peers, self.consumers = timeout, {}, set()
        self.closed = self.bound = False
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.listener.bind(str(self.path))
            self.bound = True
            self.listener.listen(16)
            self.listener.setblocking(False)
        except BaseException:
            self.close()
            raise

    @property
    def environment(self):
        pin = dict(schema=SCHEMA, path=str(self.path), token=self.token)
        return {ENVIRONMENT_KEY: json.dumps(pin, separators=(',', ':'))}

    def _reply(self, connection, message, *, error=None):
        reply = dict(schema=SCHEMA, token=self.token, kind=message.get('kind'),
                     scope=message.get('scope'), continued=error is None)
        if error is not None:
            reply['error'] = str(error)[:2000]
        connection.settimeout(1)
        connection.sendall(json.dumps(reply).encode() + b'\n')
        connection.close()
        self.peers.pop(connection, None)

    def poll(self):
        if self.closed:
            raise RuntimeError('startup gate is closed')
        for _ in range(16):
            try:
                connection, _ = self.listener.accept()
            except BlockingIOError:
                break
            connection.setblocking(False)
            self.peers[connection] = dict(buffer=bytearray(), started=time.monotonic(), message=None)
        for connection, state in list(self.peers.items()):
            message = state['message']
            try:
                if time.monotonic() - state['started'] > self.timeout:
                    raise TimeoutError('startup gate input or readiness timed out')
                if message is None:
                    try:
                        raw = connection.recv(MAX_FRAME_BYTES + 1 - len(state['buffer']))
                    except BlockingIOError:
                        continue
                    if not raw:
                        raise RuntimeError('startup gate peer closed before acknowledgement')
                    state['buffer'].extend(raw)
                    if len(state['buffer']) > MAX_FRAME_BYTES:
                        raise RuntimeError('startup gate frame exceeds its bound')
                    if b'\n' not in state['buffer']:
                        continue
                    message = json.loads(state['buffer'])
                    state['message'] = message
                    if (not isinstance(message, dict) or message.get('schema') != SCHEMA
                            or message.get('token') != self.token
                            or message.get('kind') not in ('inputs', 'before_use')
                            or not isinstance(message.get('scope'), str) or not message['scope']):
                        raise RuntimeError('startup gate frame differs from its admitted request')
                    if message['kind'] == 'inputs':
                        self.receive(message['scope'], message.get('inputs'))
                        self._reply(connection, message)
                        continue
                component = message.get('component')
                if self.command_consumer is not None:
                    command = message.get('command')
                    if not isinstance(command, str) or not command or len(command) > 4096:
                        raise RuntimeError('startup transport omitted its bounded command')
                    component = self.command_consumer(component, command)
                ready = self.before_use(message['scope'], component)
                if type(ready) is not bool:
                    raise RuntimeError('startup readiness must return bool')
                if ready:
                    self.consumers.add((message['scope'], message['component']))
                    self._reply(connection, message)
            except BaseException as error:
                try:
                    self._reply(connection, message if isinstance(message, dict) else {}, error=error)
                except OSError:
                    pass
                raise

    def close(self):
        if self.closed:
            return
        self.closed = True
        for connection in self.peers:
            connection.close()
        self.peers.clear()
        self.listener.close()
        if self.bound:
            self.path.unlink(missing_ok=True)
