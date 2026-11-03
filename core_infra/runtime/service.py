"""A native model served from one container with a fast cold start (framework defaults).

`NativeService.start()` belongs in the container's start hook (e.g. Modal's @modal.enter): it does
CPU work only and returns at once, so the platform hands the waiting request over immediately. It
spawns the model's resident process (`<program> serve <socket>`, core_infra::resident), which binds
its socket and touches no GPU until a request names what it needs; the request's model loading then
starts with the CUDA context and overlaps it (core_infra::model_load). After the first request
completes, a low-priority full SHA audit of the light-verified weights (core_infra.runtime.audit)
runs and is paused while later requests run.
"""
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

from . import ProcessGroup

# The streaming loader's defaults for serving: the reader profile measured best on B200 and the
# trusted first/last-KiB guard before use, with the full audit afterwards.
LOAD_ENVIRONMENT = {'NATIVE_LOAD_VERIFY': 'light'}


def checkpoint_environment(audit):
    """Pass the same SHA-pinned checkpoint declaration to the native loader."""
    if not audit:
        return {}
    if not audit.get('sha256'):
        raise ValueError('native checkpoint declaration requires its offline SHA-256')
    return {'NATIVE_CHECKPOINT_MANIFEST': str(audit['manifest']),
            'NATIVE_CHECKPOINT_SHA256': audit['sha256'],
            'NATIVE_CHECKPOINT_ROOTS': json.dumps({k: str(v) for k, v in audit['roots'].items()}, sort_keys=True)}


class NativeService:
    def __init__(self, program, work_dir, environment=None, *, sha256=None, audit=None, backend_plans=None,
                 startup_read_ahead_gib=None, startup_file_cache=True, timeout=3600):
        """`program`: the model's resident ELF. `environment`: extra variables (model paths, NATIVE_LOAD_*).
        `audit`: {'manifest': path, 'roots': {name: dir}, 'prefixes': [...], 'sha256': digest} or None.
        `backend_plans`: a pinned offline plan manifest, ideally packaged in the runtime image.
        `startup_read_ahead_gib`: explicit host RAM admission, with a fixed pinned-copy ring.
        `startup_file_cache`: optional OS cache warming from the audit tensor declaration;
        metadata and helper failures never gate the native owner. No retained tensor copy.
        The Rust scheduler falls back to bounded staged reads if the full schedule does not fit."""
        self.program, self.sha256, self.timeout = Path(program), sha256, timeout
        self.work = Path(work_dir)
        declaration = checkpoint_environment(audit) if audit and audit.get('sha256') else {}
        self.environment = {**os.environ, **LOAD_ENVIRONMENT, **declaration, **(environment or {})}
        if startup_read_ahead_gib is not None:
            if type(startup_read_ahead_gib) is not int or not 0 <= startup_read_ahead_gib <= 192:
                raise ValueError('startup read-ahead budget must be 0..192 GiB')
            self.environment['NATIVE_LOAD_AHEAD_GIB'] = str(startup_read_ahead_gib)
            if startup_read_ahead_gib:
                # A fixed pinned ring avoids registering every retained chunk.
                self.environment.setdefault('NATIVE_LOAD_DIRECT', '0')
        self.audit_spec, self.audit_process, self.audit_report = audit, None, None
        self.backend_plans, self.backend_report = backend_plans, None
        self.file_cache_enabled = startup_file_cache and bool(audit)
        self.file_cache, self.file_cache_warning, self.file_cache_future = None, None, None
        self.file_cache_stop = self.work / 'startup-file-cache.stop'
        if self.file_cache_enabled:
            self.environment['NATIVE_LOAD_PREFETCH_STOP_FILE'] = str(self.file_cache_stop)
        self.group, self.owner = None, None
        self.lock = threading.Lock()
        self.requests = 0

    def start(self):
        """Container start: spawn the resident process (CPU only) and return."""
        self.work.mkdir(parents=True, exist_ok=True)
        from .plans import prepare_backend_plans
        self.backend_report = prepare_backend_plans(self.backend_plans, self.work / 'backend-plans')
        self.environment.update(self.backend_report['environment'])
        self.group = ProcessGroup(workers=2)
        try:
            self.owner = self.group.owner(self.program, self.work / 'model.sock', self.environment, self.work / 'model.log',
                                          timeout=self.timeout, sha256=self.sha256)
            if self.file_cache_enabled:
                self.file_cache_future = self.group.submit(self._start_file_cache)
        except BaseException:
            self.file_cache_stop.touch()
            self.group.close()
            raise
        return self

    def invoke(self, *args):
        """One request; the first one loads the model it needs. Returns the process's details."""
        with self.lock:
            self._pause_audit(True)
            try:
                details = self.owner.invoke(list(args))
            finally:
                if self.file_cache_enabled:
                    self.file_cache_stop.touch()
                self._pause_audit(False)
            self.requests += 1
            if self.requests == 1 and self.audit_spec:
                self._start_audit()
            return details

    def _start_file_cache(self):
        # Metadata is optional side work; never delay socket creation or requests.
        try:
            from .checkpoint_prefetch import jobs
            from .files import FileReadAhead
            spec = self.audit_spec
            ranges = jobs(spec['manifest'], spec['roots'], spec.get('prefixes', ()), spec.get('sha256'))
            if ranges and not self.file_cache_stop.exists():
                self.file_cache = FileReadAhead(self.group, ranges, self.work / 'startup-file-cache', self.environment)
        except Exception as error:  # native guards/audit remain authoritative
            self.file_cache_warning = str(error)

    def audit_status(self):
        """None until the audit finishes; then its report (mismatches mean: restart the container)."""
        if self.audit_process and self.audit_process.process.poll() is not None and self.audit_report is None:
            out = self.work / 'audit.json'
            self.audit_report = json.loads(out.read_text()) if out.exists() else {'error': 'audit failed; see audit.log'}
        return self.audit_report

    def _start_audit(self):
        spec = self.audit_spec
        command = [sys.executable, '-m', 'core_infra.runtime.audit', '--manifest', spec['manifest'],
                   '--out', self.work / 'audit.json']
        for name, path in spec['roots'].items():
            command += ['--root', f'{name}={path}']
        for prefix in spec.get('prefixes', ()):
            command += ['--prefix', prefix]
        if spec.get('sha256'):
            command += ['--sha256', spec['sha256']]
        runtime = str(Path(__file__).resolve().parents[2])
        environment = {**self.environment, 'PYTHONPATH': f"{runtime}:{self.environment.get('PYTHONPATH', '')}"}
        self.audit_process = self.group.background([str(c) for c in command], environment, self.work / 'audit.log')

    def _pause_audit(self, pause):
        process = self.audit_process.process if self.audit_process else None
        if process is not None and process.poll() is None:
            try:
                os.kill(process.pid, signal.SIGSTOP if pause else signal.SIGCONT)
            except ProcessLookupError:
                pass

    def close(self):
        if self.file_cache_enabled and self.work.exists():
            self.file_cache_stop.touch()
        if self.owner is not None:
            try:
                self.owner.invoke(['shutdown'])
            except Exception:  # noqa: BLE001 - the process may already be gone
                pass
        return self.group.close() if self.group else []


def wait_for(path, timeout=10.0):
    """Waits for a resident process's socket to appear (diagnostics and tests)."""
    deadline = time.monotonic() + timeout
    while not Path(path).exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f'{path} did not appear')
        time.sleep(0.005)
