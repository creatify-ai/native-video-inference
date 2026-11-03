"""Bounded optional file-cache reads and atomic publication of verified assets."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time


_RESIDENT_BITS = bytes(flag & 1 for flag in range(256))


def prefetch_stop_path(directory, component):
    """The stop token shared by a component's native actor and cache helper."""
    return (Path(directory)/('prefetch-'+component)).with_suffix('.stop')


def publish_bundle(source, expected, destination, *, max_files=128, max_bytes=128 << 20):
    source, destination = Path(source), Path(destination)
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('asset manifest SHA256 mismatch')
    manifest = json.loads(raw)
    files = manifest['files']
    if not isinstance(files, dict) or len(files) > max_files:
        raise ValueError('asset bundle file count exceeds its bound')
    total = 0
    with tempfile.TemporaryDirectory(prefix='verified-', dir=destination.parent) as temporary:
        staging = Path(temporary) / 'assets'
        staging.mkdir()
        for name, digest in files.items():
            if Path(name).name != name or name in ('', '.', '..'):
                raise ValueError('asset name must be a flat relative filename')
            path = source.parent / name
            if path.is_symlink():
                raise ValueError('asset bundle contains a symbolic link')
            size = path.stat().st_size
            total += size
            if size > max_bytes or total > max_bytes:
                raise ValueError('asset bundle byte count exceeds its bound')
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError('asset SHA256 mismatch')
            (staging / name).write_bytes(data)
        os.rename(staging, destination)
    return manifest


def read_ahead_chunks(jobs, workers, chunk_bytes):
    """Interleave bounded file ranges in the native large-file-first order."""
    if not 1 <= workers <= 64 or not 4096 <= chunk_bytes <= 32 << 20 or chunk_bytes % 4096:
        raise ValueError('invalid read-ahead capacity')
    selected = sorted(jobs, key=lambda job: (-job['bytes'], job['path'], job.get('offset', 0)))
    remaining = iter(selected)
    active = [(next(remaining), 0) for _ in range(min(workers, len(selected)))]
    chunks = []
    while active:
        following = []
        for job, used in active:
            offset, count = job.get('offset', 0), job['bytes']
            if (not Path(job['path']).is_absolute() or type(offset) is not int or type(count) is not int
                    or offset < 0 or offset % 4096 or count <= 0):
                raise ValueError('invalid read-ahead file extent')
            size = min(chunk_bytes, count - used)
            chunks.append({'path': job['path'], 'offset': offset + used, 'bytes': size,
                           'file_bytes': job.get('file_bytes', offset + count)})
            used += size
            if used < count:
                following.append((job, used))
            elif (item := next(remaining, None)) is not None:
                following.append((item, 0))
        active = following
    return chunks


class FileReadAhead:
    """Optional helper; native consumers signal handoff without joining it."""
    def __init__(self, group, jobs, root, environment, *, workers=24, chunk_bytes=32 << 20):
        self.root = Path(root)
        self.stop_file = self.root.with_suffix('.stop')
        self.workers, self.chunk_bytes = workers, chunk_bytes
        specification = {'chunks': read_ahead_chunks(jobs, workers, chunk_bytes),
                         'workers': workers, 'chunk_bytes': chunk_bytes}
        raw = json.dumps(specification, separators=(',', ':')).encode()
        path = self.root.with_suffix('.json')
        path.write_bytes(raw)
        self.process = group.background([sys.executable, str(Path(__file__).resolve()), str(path),
            hashlib.sha256(raw).hexdigest(), str(self.stop_file)], environment, self.root.with_suffix('.log'))

    @property
    def bound_mib(self):
        # Mapped OS pages are reclaimable cache, not private payload buffers.
        return 0

    def report(self):
        return {'pid': self.process.process.pid, 'returncode': self.process.process.poll(),
                'stop_requested': self.stop_file.exists(), 'workers': self.workers,
                'private_payload_buffer_bytes': 0,
                'mapped_range_bound_bytes': self.workers * self.chunk_bytes,
                'required_for_inference': False}


def scan_read_ahead(chunks, stop_file, chunk_bytes):
    """Fault missing pages; honor a stop path or an owned selection token."""
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                         ctypes.c_int, ctypes.c_long]
    libc.mmap.restype = ctypes.c_void_p
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    if isinstance(stop_file,(str,os.PathLike)):
        stop_file = Path(stop_file)
    if not callable(getattr(stop_file,'exists',None)):
        raise TypeError('read-ahead stop requires a path or an exists() token')
    page = os.sysconf('SC_PAGE_SIZE')
    for job in chunks:
        if stop_file.exists():
            break
        path = Path(job['path'])
        offset, size = job['offset'], job['bytes']
        if not path.is_absolute() or not 0 < size <= chunk_bytes or offset < 0 or offset % page:
            raise ValueError('invalid read-ahead file extent')
        with path.open('rb', buffering=0) as source:
            stat = os.fstat(source.fileno())
            if stat.st_size < offset + size or job['file_bytes'] != stat.st_size:
                raise ValueError('read-ahead file extent differs from metadata')
            address = libc.mmap(None, size, 1, 1, source.fileno(), offset)  # PROT_READ, MAP_SHARED
            if address == ctypes.c_void_p(-1).value:
                raise OSError(ctypes.get_errno(), 'read-ahead mmap')
            try:
                pages = (size + page - 1) // page
                bitmap = (ctypes.c_ubyte * pages)()
                if libc.mincore(address, size, bitmap):
                    raise OSError(ctypes.get_errno(), 'read-ahead mincore')
                libc.madvise(address, size, 3)  # MADV_WILLNEED
                # A cached chunk needs no page walk. Inspect only mincore's
                # residency bit: the other bits are not a portable truth value.
                if b'\0' not in bytes(bitmap).translate(_RESIDENT_BITS):
                    continue
                data = (ctypes.c_ubyte * size).from_address(address)
                next_stop_check = time.monotonic() + .01
                for index in range(pages):
                    if index % 64 == 0:
                        now = time.monotonic()
                        if now >= next_stop_check:
                            next_stop_check = now + .01
                            if stop_file.exists():
                                break
                    if not bitmap[index] & 1:
                        _ = data[index * page]
                del data
            finally:
                if libc.munmap(address, size):
                    raise OSError(ctypes.get_errno(), 'read-ahead munmap')


def read_ahead(specification, stop_file):
    """Each child owns a bounded mapping; the controller owns their process group."""
    workers, chunk_bytes = specification['workers'], specification['chunk_bytes']
    if not 1 <= workers <= 64 or not 4096 <= chunk_bytes <= 32 << 20:
        raise ValueError('invalid read-ahead capacity')
    os.nice(10)
    children = []
    try:
        for index in range(workers):
            if Path(stop_file).exists():
                break
            pid = os.fork()
            if pid == 0:
                try:
                    scan_read_ahead(specification['chunks'][index::workers], stop_file, chunk_bytes)
                    os._exit(0)
                except BaseException:
                    import traceback
                    traceback.print_exc()
                    os._exit(1)
            children.append(pid)
    finally:
        codes = [os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]) for pid in children]
    print(json.dumps({'workers': len(children), 'stop_requested': Path(stop_file).exists(),
                      'returncodes': codes}), flush=True)
    if any(codes):
        raise RuntimeError('optional read-ahead helper failed; native readers retain their guards')


if __name__ == '__main__':
    source, expected, stop = sys.argv[1:]
    raw = Path(source).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('read-ahead plan SHA256 mismatch')
    read_ahead(json.loads(raw), stop)
