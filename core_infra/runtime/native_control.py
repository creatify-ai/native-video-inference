"""Bounded JSON-line control frames for an owned local native process."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import time
import math

# Backend ABI v1 permits 4 MiB replies; the owner adds a small JSON envelope.
MAX_REPLY_BYTES = (4 << 20) + (64 << 10)
MAX_REQUEST_BYTES = 1 << 20


class NativeControlClient:
    direct = True

    def __init__(self, path, args, log_path: Path, *, max_reply_bytes=MAX_REPLY_BYTES,
                 connect_timeout=0, checkpoint=lambda:None):
        if type(max_reply_bytes) is not int or not 1 <= max_reply_bytes <= MAX_REPLY_BYTES:
            raise ValueError('invalid native control frame limit')
        if (not isinstance(connect_timeout,(int,float)) or not math.isfinite(connect_timeout)
                or not 0<=connect_timeout<=900 or not callable(checkpoint)):
            raise ValueError('invalid native control connection wait')
        request = json.dumps({'args': args}, separators=(',', ':')).encode() + b'\n'
        if len(request) > MAX_REQUEST_BYTES:
            raise ValueError('native control request exceeds its frame limit')
        self.limit, self.log_path = max_reply_bytes, log_path
        self.buffer = bytearray()
        self.returncode = None
        self.connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.connection.settimeout(30)
            deadline=time.monotonic()+connect_timeout
            while True:
                checkpoint()
                try:
                    self.connection.connect(str(path))
                    break
                except (FileNotFoundError,ConnectionRefusedError):
                    if connect_timeout==0:raise
                    if time.monotonic()>=deadline:
                        raise TimeoutError('native control socket did not become ready')
                    time.sleep(min(.005,max(0,deadline-time.monotonic())))
            # A request is sent exactly once after connection setup; an
            # ambiguous native execution is never replayed.
            self.connection.sendall(request)
            self.connection.setblocking(False)
        except BaseException:
            self.close()
            raise

    @classmethod
    def adopt(cls, connection, buffer, log_path: Path, *, max_reply_bytes=MAX_REPLY_BYTES):
        """Take ownership of a command already sent by another startup owner.

        The caller must stop polling the previous client. No command is sent a
        second time, and an already received partial or complete frame is kept.
        """
        if type(max_reply_bytes) is not int or not 1 <= max_reply_bytes <= MAX_REPLY_BYTES:
            raise ValueError('invalid native control frame limit')
        client = cls.__new__(cls)
        client.limit, client.log_path = max_reply_bytes, Path(log_path)
        client.buffer, client.returncode = bytearray(buffer), None
        client.connection = connection
        try:
            connection.setblocking(False)
            if len(client.buffer) > client.limit:
                raise RuntimeError('native control reply exceeds its frame limit')
        except BaseException:
            client.close()
            raise
        return client

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        try:
            while True:
                if len(self.buffer) > self.limit:
                    raise RuntimeError('native control reply exceeds its frame limit')
                if b'\n' in self.buffer:
                    if self.buffer.count(b'\n') != 1 or not self.buffer.endswith(b'\n'):
                        raise RuntimeError('native control returned multiple reply frames')
                    # Reject malformed frames before publishing the reply file.
                    value = json.loads(self.buffer)
                    if not isinstance(value, dict) or type(value.get('ok')) is not bool:
                        raise RuntimeError('native control reply lacks a status envelope')
                    self.log_path.write_bytes(self.buffer)
                    self.returncode = 0
                    self.close()
                    return 0
                block = self.connection.recv(64 << 10)
                if not block:
                    raise RuntimeError('native control closed before a complete reply')
                self.buffer.extend(block)
        except BlockingIOError:
            return None
        except BaseException:
            self.close()
            raise

    def close(self):
        self.connection.close()
