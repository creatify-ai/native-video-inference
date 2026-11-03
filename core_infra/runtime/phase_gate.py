"""Request-owned barriers before a declared GPU decode phase.

The framework submits lookahead while the producer has not yet allocated its
decoder temporaries. Acknowledgement never waits for weight-copy completion.
"""
import json
from pathlib import Path
import socket
import uuid

ENVIRONMENT_KEY = 'NATIVE_DECODE_PHASE_GATE'
SCHEMA = 'native-decode-phase-gate-v1'
RETIRE_SCHEMA = 'native-decode-phase-gate-v2'
MAX_FRAME_BYTES = 1024


class DecodePhaseGate:
    def __init__(self, directory, stage, geometry, submit, *, retirement=None, on_retired=None):
        if not stage or not callable(submit) or not geometry or any(type(n) is not int or n <= 0 for n in geometry):
            raise ValueError('one declared decode phase and positive geometry required')
        self.path = Path(directory) / ('phase-' + uuid.uuid4().hex[:12] + '.sock')
        if not self.path.is_absolute() or len(str(self.path).encode()) >= 104:
            raise ValueError('phase socket requires a short absolute request-owned path')
        self.token = uuid.uuid4().hex
        self.stage, self.geometry, self.submit = stage, list(geometry), submit
        if (retirement is None) != (on_retired is None):
            raise ValueError('native retirement requires a decision and receipt consumer')
        self.schema = SCHEMA if retirement is None else RETIRE_SCHEMA
        self.retirement, self.on_retired, self.release = retirement, on_retired, None
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.bound = False
        self.connection = None
        self.buffer = bytearray()
        self.completed = self.closed = False
        try:
            self.listener.bind(str(self.path))
            self.bound = True
            self.listener.listen(1)
            self.listener.setblocking(False)
        except BaseException:
            self.close()
            raise

    @property
    def pin(self):
        return dict(schema=self.schema, path=str(self.path), token=self.token,
            stage=self.stage, geometry=list(self.geometry))

    @property
    def environment(self):
        return {ENVIRONMENT_KEY: json.dumps(self.pin, separators=(',', ':'))}

    def poll(self):
        if self.completed or self.closed:
            return False
        if self.connection is None:
            try:
                self.connection, _ = self.listener.accept()
                self.connection.setblocking(False)
            except BlockingIOError:
                return False
        try:
            raw = self.connection.recv(MAX_FRAME_BYTES + 1 - len(self.buffer))
        except BlockingIOError:
            return False
        if not raw:
            raise RuntimeError('native decode phase closed before acknowledgement')
        self.buffer.extend(raw)
        if len(self.buffer) > MAX_FRAME_BYTES:
            raise RuntimeError('native decode phase frame exceeds its bound')
        if b'\n' not in self.buffer:
            return False
        message = json.loads(self.buffer)
        self.buffer.clear()
        if self.release is None:
            if message != dict(schema=self.schema, token=self.token, stage=self.stage, geometry=self.geometry):
                raise RuntimeError('native decode phase differs from its admitted request')
            if self.retirement is not None:
                self.release = self.retirement()
                if type(self.release) is not bool:
                    raise RuntimeError('native retirement must be an explicit framework decision')
                self.connection.settimeout(1)
                self.connection.sendall((json.dumps(dict(schema=self.schema, token=self.token,
                    retire_primary=self.release)) + '\n').encode())
                self.connection.setblocking(False)
                return False
        else:
            receipt = message.get('receipt')
            if (message != dict(schema=self.schema, token=self.token, retirement_complete=True,
                    retire_primary=self.release, receipt=receipt) or type(receipt) is not dict or
                    receipt.get('retired') is not self.release):
                raise RuntimeError('native weight retirement differs from its admitted decision')
            self.on_retired(self.release, receipt)
        # Native decode is still blocked here. Submit only; DMA completion is
        # joined later by the existing copy handle at cancellation or adoption.
        self.submit()
        reply = (json.dumps(dict(schema=self.schema, token=self.token, continued=True)) + '\n').encode()
        self.connection.settimeout(1)
        self.connection.sendall(reply)
        self.completed = True
        self.close()
        return True

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.connection is not None:
            self.connection.close()
        self.listener.close()
        if self.bound:
            self.path.unlink(missing_ok=True)
