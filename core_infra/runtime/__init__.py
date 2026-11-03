"""Model-independent process ownership and bounded asynchronous native RPC."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time


@dataclass(frozen=True)
class ReaderBudget:
    workers: int
    chunk_mib: int = 32
    pinned_slots: int = 4
    limit_mib: int = 2048
    # Direct DMA leases existing reader storage; its events need no copy ring.
    direct_inflight: int | None = None

    @property
    def bound_mib(self):
        return (2 * self.workers + self.pinned_slots) * self.chunk_mib

    def validate(self):
        if not (1 <= self.workers <= 64 and 1 <= self.chunk_mib <= 128
                and 1 <= self.pinned_slots <= 16 and self.bound_mib <= self.limit_mib):
            raise ValueError('reader buffers exceed the declared resource budget')
        if self.direct_inflight is not None and (type(self.direct_inflight) is not int
                or not 1 <= self.direct_inflight <= 16):
            raise ValueError('direct upload in-flight depth must be an integer from 1 to 16')
        return self

    def startup_environment(self, *, direct=True, verify='full', guard_manifest=None, guard_sha256=None):
        """Declare the shared native policy without changing its reader quota.

        Raw-file models can bind a trusted guard table. Packed checkpoints may
        carry guards in their admitted manifest instead, so a table is optional
        but its path and SHA must be supplied together.
        """
        self.validate()
        if type(direct) is not bool or verify not in ('full', 'light'):
            raise ValueError('invalid native startup integrity or direct policy')
        if self.direct_inflight is not None and not direct:
            raise ValueError('direct upload in-flight depth requires direct uploads')
        if verify == 'light' and (guard_manifest is None) != (guard_sha256 is None):
            raise ValueError('native startup guard path and SHA must be declared together')
        environment = dict(NATIVE_STARTUP='1', NATIVE_LOAD_WORKERS=str(self.workers),
            NATIVE_LOAD_CHUNK_MIB=str(self.chunk_mib), NATIVE_LOAD_PINNED_SLOTS=str(self.pinned_slots),
            NATIVE_LOAD_BUDGET_MIB=str(self.limit_mib), NATIVE_LOAD_DIRECT=str(int(direct)),
            NATIVE_LOAD_VERIFY=verify)
        if self.direct_inflight is not None:
            environment['NATIVE_LOAD_DIRECT_INFLIGHT'] = str(self.direct_inflight)
        if verify == 'light' and guard_manifest is not None:
            if (not isinstance(guard_manifest, str) or not guard_manifest
                    or not isinstance(guard_sha256, str) or len(guard_sha256) != 64
                    or any(value not in '0123456789abcdef' for value in guard_sha256)):
                raise ValueError('invalid native startup guard identity')
            environment.update(NATIVE_LOAD_GUARD_MANIFEST=guard_manifest, NATIVE_LOAD_GUARD_SHA256=guard_sha256)
        return environment


class NativeOwner:
    """One subprocess, one serial command stream, and an owned startup future."""
    def __init__(self, command, address, environment, log_path, *, timeout=900):
        self.address = Path(address) if address is not None else None
        self.timeout = timeout
        self.lock = threading.Lock()
        self.connections = set()
        self.guard = threading.Lock()
        self.closed = False
        self.log_path = Path(log_path)
        self.log = self.log_path.open('wb')
        try:
            self.process = subprocess.Popen(list(map(str, command)), env=environment,
                stdin=subprocess.DEVNULL, stdout=self.log, stderr=self.log, start_new_session=True)
        except BaseException:
            self.log.close()
            raise

    def invoke(self, args):
        if self.address is None:
            raise RuntimeError('background process has no command socket')
        with self.lock:
            if self.closed:
                raise RuntimeError('native owner is closed')
            deadline = time.monotonic() + self.timeout
            stream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            with self.guard:
                self.connections.add(stream)
            try:
                while True:
                    if self.closed or self.process.poll() is not None:
                        raise RuntimeError('native owner exited; see ' + str(self.log_path))
                    try:
                        stream.connect(str(self.address))
                        break
                    except (FileNotFoundError, ConnectionRefusedError):
                        if time.monotonic() >= deadline:
                            raise TimeoutError('native owner startup timed out')
                        time.sleep(.005)
                stream.settimeout(max(.001, deadline - time.monotonic()))
                data = json.dumps({'args': list(map(str, args))}).encode() + b'\n'
                if len(data) > 65536:
                    raise ValueError('native command exceeds the transport bound')
                stream.sendall(data)
                with stream.makefile('rb') as reader:
                    raw = reader.readline(8 * 1024 * 1024 + 1)
                if not raw.endswith(b'\n') or len(raw) > 8 * 1024 * 1024:
                    raise RuntimeError('native owner returned an invalid bounded response')
                reply = json.loads(raw)
                if reply.get('ok') is not True:
                    raise RuntimeError(str(reply.get('error')) + '; see ' + str(self.log_path))
                return reply['details']
            finally:
                with self.guard:
                    self.connections.discard(stream)
                stream.close()

    def terminate(self):
        self.closed = True
        with self.guard:
            for stream in self.connections:
                try:
                    stream.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def reap(self):
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.process.wait(timeout=5)
        self.log.close()
        return {'pid': self.process.pid, 'returncode': self.process.returncode, 'reaped': True}


class ResidentOwner(NativeOwner):
    """A command port to an actor whose process lifetime belongs to its service.

    Closing this port interrupts its RPCs. The service must still retire/reap
    the actor on failure; this adapter never claims that closing an RPC kills it.
    """
    def __init__(self, process, address, *, timeout=900, adopted=None):
        self.process, self.address, self.timeout = process, Path(address), timeout
        self.lock, self.guard = threading.Lock(), threading.Lock()
        self.connections, self.closed = set(), False
        self.log_path = self.address.with_suffix('.log')
        self.adopted = adopted

    def invoke(self, args):
        if self.adopted is not None:
            with self.lock:
                expected, client = self.adopted
                if tuple(map(str, args)) != tuple(map(str, expected)):
                    raise RuntimeError('adopted preload differs from the declared command')
                self.adopted = None
                begun = time.monotonic()
                try:
                    while client.poll() is None:
                        if self.closed or self.process.poll() is not None:
                            raise RuntimeError('adopted native owner exited')
                        if time.monotonic() - begun > self.timeout:
                            raise TimeoutError('adopted native preload timed out')
                        time.sleep(.005)
                    reply = json.loads(client.buffer)
                    if reply.get('ok') is not True:
                        raise RuntimeError('adopted native preload failed: ' + str(reply)[:2000])
                    return reply['details']
                finally:
                    client.close()
        return super().invoke(args)

    def terminate(self):
        self.closed = True
        if self.adopted is not None:
            self.adopted[1].close()
        with self.guard:
            for stream in self.connections:
                try:
                    stream.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def reap(self):
        return {'pid': self.process.pid, 'borrowed': True, 'reaped': False}


class ConditionOwner:
    """An external readiness fact bound as a declared component transport.

    Waiting is bounded and interruptible. It neither starts a process nor
    schedules dependent work; DeclaredStartup owns those dependency edges.
    """
    def __init__(self, predicate, *, timeout=900):
        if not callable(predicate) or timeout <= 0:
            raise ValueError('condition transport requires a predicate and timeout')
        self.predicate, self.timeout = predicate, timeout
        self.cancelled = threading.Event()

    def invoke(self, args):
        if tuple(args) != ('wait',):
            raise ValueError('condition transport only accepts its declared wait')
        deadline = time.monotonic() + self.timeout
        while not self.cancelled.is_set():
            ready = self.predicate()
            if type(ready) is not bool:
                raise RuntimeError('external startup readiness must return bool')
            if ready:
                return {'ready': True}
            if time.monotonic() >= deadline:
                raise TimeoutError('external startup readiness timed out')
            self.cancelled.wait(.005)
        raise RuntimeError('external startup readiness cancelled')

    def terminate(self):
        self.cancelled.set()

    def reap(self):
        return {'external_condition': True, 'closed': self.cancelled.is_set()}


class ProcessGroup:
    """Own all asynchronous startup work; failure/cancellation retires the group."""
    def __init__(self, workers=8):
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='native-startup')
        self.owners = []
        self.closed = False
        self.cleanup = []
        self.guard = threading.Lock()

    def owner(self, program, address, environment, log_path, *, timeout=900, sha256=None):
        program = Path(program)
        with program.open('rb') as stream:
            if stream.read(4) != b'\x7fELF':
                raise ValueError('a native component must be an ELF executable')
            if sha256 is not None:
                stream.seek(0)
                if hashlib.file_digest(stream, 'sha256').hexdigest() != sha256:
                    raise ValueError('native executable SHA256 mismatch')
        with self.guard:
            if self.closed:
                raise RuntimeError('native process group is closed')
            owner = NativeOwner([program, 'serve', address], address, environment, log_path, timeout=timeout)
            self.owners.append(owner)
            return owner

    def submit(self, function, *args):
        if self.closed:
            raise RuntimeError('native process group is closed')
        return self.executor.submit(function, *args)

    def background(self, command, environment, log_path):
        with self.guard:
            if self.closed:
                raise RuntimeError('native process group is closed')
            process = NativeOwner(command, None, environment, log_path)
            self.owners.append(process)
            return process

    def run_elf(self, program, args, environment, log_path, *, timeout=900):
        """Own a finite native preparation command, including cancellation."""
        with Path(program).open('rb') as source:
            if source.read(4) != b'\x7fELF':
                raise ValueError('native preparation requires an ELF executable')
        child = self.background([program, *args], environment, log_path)
        try:
            code = child.process.wait(timeout=timeout)
        except BaseException:
            child.terminate()
            raise
        if code:
            raise RuntimeError('native preparation failed; see ' + str(log_path))
        return json.loads(Path(log_path).read_bytes())

    def close(self):
        if self.closed:
            return self.cleanup
        with self.guard:
            self.closed = True
            owners = list(self.owners)
        for owner in owners:
            owner.terminate()
        for owner in owners:
            self.cleanup.append(owner.reap())
        self.executor.shutdown(wait=True, cancel_futures=True)
        return self.cleanup
