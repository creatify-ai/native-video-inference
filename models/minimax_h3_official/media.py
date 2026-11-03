"""Bounded RGB transport from the native VAE to ffmpeg; no tensor framework."""
from dataclasses import dataclass
import errno
import os
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import tempfile
import threading


@dataclass(frozen=True)
class MediaOptions:
    format: str = 'mp4'
    ffmpeg: str = 'ffmpeg'
    capture_raw: bool = False
    hash_outputs: bool = False

    @classmethod
    def parse(cls, value=None):
        value = {} if value is None else dict(value)
        if set(value) - {'format', 'ffmpeg', 'capture_raw', 'hash_outputs'}:
            raise ValueError('unsupported media option')
        result = cls(**value)
        if (result.format not in ('mp4', 'raw') or not isinstance(result.ffmpeg, str)
                or not result.ffmpeg or type(result.capture_raw) is not bool
                or type(result.hash_outputs) is not bool):
            raise ValueError('invalid media format, ffmpeg path or capture flags')
        return result


def ffmpeg_threads(height):
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = os.cpu_count() or 1
    return min(max(1, cpus * 3 // 2), max(1, ((height + 15) // 16) // 2), 128)


def resolve_ffmpeg(command):
    executable = shutil.which(command)
    if executable is None:
        raise FileNotFoundError('ffmpeg is required for MP4 output; set media.ffmpeg or use media.format=raw')
    return executable


def stereo_sample_count(report):
    shape = report.get('output_shape')
    if not isinstance(shape, list) or not all(type(n) is int and n > 0 for n in shape):
        raise ValueError('invalid native audio output shape')
    if len(shape) == 2 and shape[0] == 2:
        count = shape[1]
    elif len(shape) == 3 and shape[:2] == [2, 1]:
        count = shape[2]
    else:
        raise ValueError('MP4 output requires planar stereo audio')
    if report.get('sampling_rate') != 32000 or report.get('nonfinite') != 0:
        raise ValueError('MP4 output requires a finite native 32 kHz waveform')
    return count


class MP4Stream:
    """One serialized request owns its socket, encoder, and receiver thread."""
    def __init__(self, output, audio_wav, width, height, frames, *, ffmpeg='ffmpeg',
                 capture_path=None, timeout=120):
        self.output = Path(output)
        self.audio_wav = Path(audio_wav)
        self.width, self.height, self.frames = width, height, frames
        self.ffmpeg = ffmpeg
        self.capture_path = Path(capture_path) if capture_path is not None else None
        self.timeout = timeout
        self.stop = threading.Event()
        self.listener = self.connection = self.process = self.thread = self.log = self.directory = None
        self.error = None
        self.finished = False
        self.received_frames = self.received_bytes = 0
        self.splice = hasattr(os, 'splice') and self.capture_path is None
        self.partial = self.output.with_name('.' + self.output.stem + '.partial.mp4')

    def __enter__(self):
        if self.output.exists() or self.partial.exists():
            raise FileExistsError('MP4 output must be fresh')
        executable = resolve_ffmpeg(self.ffmpeg)
        try:
            self.output.parent.mkdir(parents=True, exist_ok=True)
            self.directory = tempfile.TemporaryDirectory(prefix='h3-rgb-', dir='/tmp')
            self.socket_path = Path(self.directory.name) / 'frames.sock'
            self.listener = socket.socket(socket.AF_UNIX)
            self.listener.bind(str(self.socket_path))
            self.listener.listen(1)
            self.listener.settimeout(.2)
            self.log = self.output.with_suffix('.ffmpeg.log').open('w+b')
            command = [executable, '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
                '-s', f'{self.width}x{self.height}', '-pix_fmt', 'rgb24', '-r', '24.00',
                '-i', 'pipe:0', '-i', str(self.audio_wav), '-vcodec', 'libx264',
                '-preset', 'fast', '-pix_fmt', 'yuv420p', '-crf', '25',
                '-threads', str(ffmpeg_threads(self.height)), '-acodec', 'aac',
                '-map', '0:v:0', '-map', '1:a:0', '-v', 'warning', str(self.partial)]
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=self.log)
            self.thread = threading.Thread(target=self._receive, name='h3-mp4-stream')
            self.thread.start()
            return self
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _read(connection, count):
        result = bytearray()
        while len(result) < count:
            data = connection.recv(count - len(result))
            if not data:
                raise EOFError('native RGB stream ended before its completion marker')
            result.extend(data)
        return result

    def _transfer(self, count, capture):
        while count:
            if self.splice:
                try:
                    transferred = os.splice(self.connection.fileno(), self.process.stdin.fileno(), count)
                except OSError as error:
                    if error.errno not in (errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP):
                        raise
                    self.splice = False
                    continue
                if not transferred:
                    raise EOFError('truncated RGB packet')
            else:
                data = self.connection.recv(min(count, 1024 * 1024))
                if not data:
                    raise EOFError('truncated RGB packet')
                if capture is not None:
                    capture.write(data)
                pending = memoryview(data)
                while pending:
                    pending = pending[os.write(self.process.stdin.fileno(), pending):]
                transferred = len(data)
            count -= transferred

    def _receive(self):
        capture = None
        try:
            while not self.stop.is_set():
                try:
                    self.connection, _ = self.listener.accept()
                    break
                except socket.timeout:
                    continue
            if self.stop.is_set():
                return
            header = self._read(self.connection, 24)
            if (header[:8] != b'H3RGB001'
                    or struct.unpack('<IIII', header[8:]) != (self.width, self.height, self.frames, 0)):
                raise ValueError('native RGB stream geometry or flags differ from the request')
            if self.capture_path is not None:
                self.capture_path.parent.mkdir(parents=True, exist_ok=True)
                capture = self.capture_path.open('xb')
            while True:
                first, count, size = struct.unpack('<IIQ', self._read(self.connection, 16))
                if count == 0:
                    if first != self.frames or size or self.received_frames != self.frames:
                        raise ValueError('incomplete native RGB frame stream')
                    break
                if (first != self.received_frames or not 1 <= count <= 22
                        or first + count > self.frames or size != count * self.width * self.height * 3):
                    raise ValueError('invalid native RGB packet ordering or byte count')
                self._transfer(size, capture)
                self.received_frames += count
                self.received_bytes += size
        except BaseException as error:
            self.error = error
        finally:
            if capture is not None:
                capture.close()
            if self.connection is not None:
                self.connection.close()
            if self.process is not None and self.process.stdin is not None:
                try:
                    self.process.stdin.close()
                except OSError as error:
                    if self.error is None:
                        self.error = error

    def finish(self):
        self.thread.join(self.timeout)
        if self.thread.is_alive():
            raise TimeoutError('native RGB stream did not complete')
        if self.error is not None:
            raise RuntimeError('native RGB streaming failed') from self.error
        if self.process.wait(timeout=self.timeout):
            self.log.seek(0)
            raise RuntimeError('ffmpeg failed: ' + self.log.read()[-4096:].decode(errors='replace'))
        self.partial.replace(self.output)
        self.finished = True
        return dict(format='mp4', file=self.output.name, bytes=self.output.stat().st_size,
            width=self.width, height=self.height, frames=self.received_frames,
            rgb_bytes=self.received_bytes, raw_capture=self.capture_path is not None,
            transport='native pinned RGB chunks through a local socket to ffmpeg')

    def close(self):
        self.stop.set()
        if self.connection is not None:
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if self.listener is not None:
            self.listener.close()
        if self.process is not None and self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=5)
        if self.thread is not None:
            self.thread.join(5)
            if self.thread.is_alive():
                raise RuntimeError('MP4 receiver thread did not stop')
        if self.log is not None:
            self.log.close()
        if self.directory is not None:
            self.directory.cleanup()
        if not self.finished:
            self.partial.unlink(missing_ok=True)

    def __exit__(self, *error):
        self.close()
