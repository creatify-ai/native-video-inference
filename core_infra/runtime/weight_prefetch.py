"""Model-independent, single-owner scheduling for next-request weight copies.

Backends own memory and CUDA events. This broker never runs a model, supplies
activations, waits for an unfinished copy, or changes a backend's arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from collections import deque
import math
import threading
import time
from typing import Protocol


@dataclass(frozen=True)
class ModelKey:
    family: str
    revision: str
    recipe_sha256: str

    def __post_init__(self):
        if not self.family or not self.revision or len(self.recipe_sha256) != 64:
            raise ValueError("immutable model identity is required")
        if any(c not in "0123456789abcdef" for c in self.recipe_sha256):
            raise ValueError("invalid model recipe checksum")


@dataclass(frozen=True)
class Ticket:
    request_id: str
    model: ModelKey


@dataclass(frozen=True)
class DecodeWindow:
    request_id: str
    owner_pid: int
    device_uuid: str
    transaction_idle: bool
    free_device_bytes: int
    reclaimable_device_bytes: int
    decoder_reserve_bytes: int
    host_pinned_available_bytes: int
    source_gpu_complete: bool = False
    capacity_limit_bytes: int | None = None

    def __post_init__(self):
        if type(self.owner_pid) is not int or self.owner_pid <= 0 or not self.device_uuid or not self.request_id:
            raise ValueError("a concrete owner and active request are required")
        if type(self.transaction_idle) is not bool or type(self.source_gpu_complete) is not bool:
            raise ValueError("transaction state must be explicit")
        if any(type(value) is not int or value < 0 for value in (
                self.free_device_bytes, self.reclaimable_device_bytes,
                self.decoder_reserve_bytes, self.host_pinned_available_bytes)):
            raise ValueError("memory reservations are nonnegative integer bytes")
        if self.capacity_limit_bytes is not None and (type(self.capacity_limit_bytes) is not int or self.capacity_limit_bytes<0):
            raise ValueError('copy capacity limit must be a nonnegative byte count')

    @property
    def copy_capacity(self):
        available=max(0, self.free_device_bytes + self.reclaimable_device_bytes - self.decoder_reserve_bytes)
        return available if self.capacity_limit_bytes is None else min(available,self.capacity_limit_bytes)


@dataclass(frozen=True)
class CopyPlan:
    model: ModelKey
    device_bytes: int
    pinned_host_bytes: int
    weights_only: bool = True
    gpu_prepare_required: bool = False
    estimate_seconds: float | None = None
    deferred_auxiliary: bool = False
    deferred_primary: bool = False
    components: tuple[str, ...] | None = None

    def __post_init__(self):
        if any(type(value) is not int or value < 0 for value in (self.device_bytes, self.pinned_host_bytes)):
            raise ValueError("copy reservations are nonnegative integer bytes")
        if any(type(value) is not bool for value in (self.weights_only, self.gpu_prepare_required, self.deferred_auxiliary,self.deferred_primary)):
            raise ValueError("copy plan capabilities must be explicit")
        if self.deferred_auxiliary and self.deferred_primary:raise ValueError('partial plan must prepare at least one component')
        if self.components is not None and (type(self.components) is not tuple or not self.components or
                len(set(self.components))!=len(self.components) or any(not isinstance(n,str) or not n for n in self.components)):
            raise ValueError('partial preload requires explicit unique components')
        if self.estimate_seconds is not None and (not math.isfinite(self.estimate_seconds) or self.estimate_seconds < 0):
            raise ValueError("invalid measured copy estimate")


@dataclass(frozen=True)
class DecodeWindowPolicy:
    """A migrated model declares safe overlap phases; scheduling stays generic."""
    reserve_bytes: int
    geometries: frozenset[tuple[int, ...]] | None = None
    unavailable_families: frozenset[str] = frozenset()
    source_gpu_complete: bool = False
    reclaimed_weights_only: bool = False
    idle_retirement: bool = False

    def __post_init__(self):
        if (type(self.source_gpu_complete) is not bool or type(self.reclaimed_weights_only) is not bool or
                type(self.idle_retirement) is not bool):
            raise ValueError('source GPU completion must be explicit')
        if self.idle_retirement and (self.source_gpu_complete or not self.reclaimed_weights_only):
            raise ValueError('idle final-use retirement requires a before-decode reclaimed-only envelope')
        if type(self.reserve_bytes) is not int or self.reserve_bytes < 0:
            raise ValueError('decode reservation must be nonnegative bytes')
        if self.geometries is not None and (type(self.geometries) is not frozenset or
                any(type(shape) is not tuple or not shape or any(type(n) is not int or n <= 0 for n in shape)
                    for shape in self.geometries)):
            raise ValueError('calibrated decode geometries must be immutable positive dimensions')
        if type(self.unavailable_families) is not frozenset or any(
                type(family) is not str or not family for family in self.unavailable_families):
            raise ValueError('unavailable copy workers must be explicit model families')

    def allows(self, target_family, geometry):
        return (target_family not in self.unavailable_families and
                (self.geometries is None or geometry in self.geometries))

    def apply(self, window, target_family, geometry):
        if not self.allows(target_family,geometry):return None
        return replace(window, decoder_reserve_bytes=(max(self.reserve_bytes, window.decoder_reserve_bytes)
                       if self.reclaimed_weights_only else self.reserve_bytes),
                       source_gpu_complete=self.source_gpu_complete)


@dataclass(frozen=True)
class ReadyWeights:
    model: ModelKey
    owner_pid: int
    device_uuid: str
    generation: int
    payload_verified: bool
    copy_complete: bool
    gpu_prepare_complete: bool
    primary_weights_ready: bool = True

    def __post_init__(self):
        if any(type(value) is not bool for value in (
                self.payload_verified, self.copy_complete, self.gpu_prepare_complete,self.primary_weights_ready)):
            raise ValueError("completion flags must be verified booleans")


@dataclass(frozen=True)
class PartialWeights:
    model: ModelKey
    owner_pid: int
    device_uuid: str
    components: tuple[str, ...]
    receipt: dict


class CopyHandle(Protocol):
    def poll(self) -> ReadyWeights | None: ...
    def abandon(self) -> None:
        """Cancel its consumer; retain memory until DMA ends.

        poll() must still return the completion receipt and release abandoned
        reservations, including when abandon() follows an earlier ready poll.
        """
    def adopt(self, request_id: str, ready: ReadyWeights) -> object: ...


class Backend(Protocol):
    owner_pid: int
    device_uuid: str

    def plan(self, ticket: Ticket) -> CopyPlan: ...
    def start(self, ticket: Ticket, plan: CopyPlan, window: DecodeWindow) -> CopyHandle: ...


@dataclass
class _Pending:
    ticket: Ticket
    handle: CopyHandle
    ready: ReadyWeights | None = None
    abandoned: bool = False
    plan: CopyPlan | None = None


@dataclass(frozen=True)
class Handoff:
    status: str  # ready, auxiliary_ready, copy_pending, foreground
    payload: object = None


class WeightPrefetchBroker:
    """One GPU owner, one active request and at most one speculative weight copy.

    Requests must pass the application's CPU model/recipe admission before
    enqueue(). A decode window is explicitly granted after the final use of the
    current DiT weights; refinement/relay transactions cannot grant it early.
    """
    def __init__(self, owner_pid: int, device_uuid: str, *, allow_gpu_prepare=False):
        if type(owner_pid) is not int or owner_pid <= 0 or not device_uuid:
            raise ValueError("one concrete GPU owner is required")
        self.owner_pid, self.device_uuid = owner_pid, device_uuid
        self.allow_gpu_prepare = allow_gpu_prepare
        self.backends: dict[str, Backend] = {}
        self.windows: dict[str, DecodeWindowPolicy] = {}
        self.queue: list[Ticket] = []
        self.pending: _Pending | None = None
        self.failed = False
        self.events = deque(maxlen=512)
        self._blocked_event = None
        self._ids: set[str] = set()
        self._last_generation: dict[str, int] = {}
        self._lock = threading.RLock()

    def register(self, family: str, backend: Backend):
        with self._lock:
            if family in self.backends:
                raise ValueError("backend already registered")
            if (backend.owner_pid, backend.device_uuid) != (self.owner_pid, self.device_uuid):
                raise ValueError("prefetch backends must share the same GPU process and device")
            self.backends[family] = backend

    def register_window(self, phase: str, policy: DecodeWindowPolicy):
        with self._lock:
            if not phase or phase in self.windows or not isinstance(policy, DecodeWindowPolicy):
                raise ValueError('decode phase must have one explicit window policy')
            self.windows[phase] = policy

    def open_window(self, phase: str, window: DecodeWindow, *, geometry=None) -> bool:
        with self._lock:
            if self.failed:
                raise RuntimeError('prefetch owner failed')
            policy = self.windows[phase]
            if not self.queue:
                return False
            admitted = policy.apply(window, self.queue[0].model.family, geometry)
            return admitted is not None and self.open_decode_window(admitted)

    def is_next(self, ticket: Ticket) -> bool:
        with self._lock:
            if self.failed:
                raise RuntimeError("prefetch owner failed")
            return bool(self.queue and self.queue[0] == ticket)

    def enqueue(self, ticket: Ticket):
        with self._lock:
            if self.failed:
                raise RuntimeError("prefetch owner failed")
            if not ticket.request_id or ticket.request_id in self._ids:
                raise ValueError("live request IDs must be unique")
            if self.queue:
                raise ValueError("only one queued successor is admitted")
            if ticket.model.family not in self.backends:
                raise ValueError("model family has no admitted backend")
            self._ids.add(ticket.request_id)
            self.queue.append(ticket)

    def cancel(self, request_id: str):
        with self._lock:
            self.queue = [t for t in self.queue if t.request_id != request_id]
            self._ids.discard(request_id)
            if self.pending is not None and self.pending.ticket.request_id == request_id:
                # Keep the copy slot occupied until its CUDA event completes.
                # Cancelling a ticket cannot release a live DMA reservation.
                try:
                    self.pending.handle.abandon()
                except BaseException:
                    self.failed = True
                    raise
                self.pending.abandoned = True
                self.events.append(dict(event="cancelled", request_id=request_id))

    def open_decode_window(self, window: DecodeWindow) -> bool:
        with self._lock:
            if self.failed:
                raise RuntimeError("prefetch owner failed")
            if (window.owner_pid, window.device_uuid) != (self.owner_pid, self.device_uuid):
                raise ValueError("decode window belongs to another GPU owner")
            if any(value < 0 for value in (window.free_device_bytes, window.reclaimable_device_bytes,
                                          window.decoder_reserve_bytes, window.host_pinned_available_bytes)):
                raise ValueError("negative memory budget")
            if not window.transaction_idle or self.pending is not None or not self.queue:
                return False
            ticket = self.queue[0]
            if ticket.request_id == window.request_id:
                raise ValueError("the active request cannot prefetch itself")
            backend = self.backends[ticket.model.family]
            plan = backend.plan(ticket)
            fit = getattr(backend, 'plan_for_capacity', None)
            if fit is not None:
                plan = fit(ticket, window.copy_capacity)
            if plan.model != ticket.model or not plan.weights_only:
                raise ValueError("prefetch plan must contain only the selected immutable model weights")
            reason = None
            if plan.device_bytes > window.copy_capacity:
                reason = "decoder_memory_reserved"
            elif plan.pinned_host_bytes > window.host_pinned_available_bytes:
                reason = "host_memory_reserved"
            elif plan.gpu_prepare_required and not self.allow_gpu_prepare:
                reason = "gpu_preparation_not_admitted"
            if reason:
                state=(ticket.request_id,reason,plan.device_bytes,plan.deferred_primary,plan.deferred_auxiliary)
                if self._blocked_event is not None and self._blocked_event[0]==state:
                    event=self._blocked_event[1]
                    event['retry_count']+=1
                    event['last_checked_monotonic_ns']=time.monotonic_ns()
                    event['latest_copy_capacity_bytes']=window.copy_capacity
                else:
                    event=dict(event="skipped", request_id=ticket.request_id, reason=reason,
                        device_bytes=plan.device_bytes, copy_capacity_bytes=window.copy_capacity,
                        free_device_bytes=window.free_device_bytes, reclaimable_device_bytes=window.reclaimable_device_bytes,
                        decoder_reserve_bytes=window.decoder_reserve_bytes,retry_count=0,
                        monotonic_ns=time.monotonic_ns())
                    self.events.append(event)
                    self._blocked_event=(state,event)
                return False
            self._blocked_event=None
            try:
                handle = backend.start(ticket, plan, window)
            except BaseException:
                # start() may already have allocated/submitted CUDA work. Only
                # retiring the owner can safely recover an unknown copy state.
                self.failed = True
                raise
            self.pending = _Pending(ticket, handle, plan=plan)
            self.events.append(dict(event="started", request_id=ticket.request_id,
                                    source_request_id=window.request_id,
                                    device_bytes=plan.device_bytes, deferred_auxiliary=plan.deferred_auxiliary,
                                    deferred_primary=plan.deferred_primary,
                                    owner_pid=self.owner_pid, monotonic_ns=time.monotonic_ns()))
            return True

    def poll(self) -> ReadyWeights | None:
        with self._lock:
            if self.failed:
                raise RuntimeError("prefetch owner failed")
            pending = self.pending
            if pending is None:
                return None
            if pending.ready is not None and not pending.abandoned:
                return pending.ready
            try:
                ready = pending.handle.poll()
                if ready is None:
                    return None
                if isinstance(ready,PartialWeights):
                    if (ready.model!=pending.ticket.model or
                            (ready.owner_pid,ready.device_uuid)!=(self.owner_pid,self.device_uuid) or
                            ready.components!=pending.plan.components):
                        raise RuntimeError('partial preload receipt belongs to another admitted copy')
                    self.pending=None
                    self.events.append(dict(event='partial_ready',request_id=pending.ticket.request_id,
                        components=list(ready.components),receipt=ready.receipt,
                        abandoned=pending.abandoned,monotonic_ns=time.monotonic_ns()))
                    return None
                auxiliary_only=bool(pending.plan and pending.plan.deferred_primary)
                generation_valid=(type(ready.generation) is int and
                    ((not ready.primary_weights_ready and ready.generation==0) if auxiliary_only else
                     (ready.primary_weights_ready and ready.generation>=1 and
                      (pending.ready is not None or ready.generation>self._last_generation.get(ready.model.family,0)))))
                if (ready.model != pending.ticket.model or
                        (ready.owner_pid, ready.device_uuid) != (self.owner_pid, self.device_uuid) or
                        not generation_valid or
                        (pending.ready is not None and ready != pending.ready) or not ready.payload_verified or
                        not ready.copy_complete or not ready.gpu_prepare_complete):
                    raise RuntimeError("prefetch completion identity or CUDA event is invalid")
            except BaseException:
                self.failed = True
                pending.handle.abandon()
                raise
            pending.ready = ready
            if ready.primary_weights_ready:self._last_generation[ready.model.family] = ready.generation
            if pending.abandoned:
                self.pending = None
                self.events.append(dict(event="discarded", request_id=pending.ticket.request_id))
                return None
            self.events.append(dict(event="ready", request_id=pending.ticket.request_id,
                                    generation=ready.generation,primary_weights_ready=ready.primary_weights_ready,
                                    monotonic_ns=time.monotonic_ns()))
            return ready

    def adopt(self, ticket: Ticket) -> Handoff:
        with self._lock:
            if self.failed:
                raise RuntimeError("prefetch owner failed")
            if not self.queue or self.queue[0] != ticket:
                raise ValueError("only the next admitted request can adopt weights")
            pending = self.pending
            if pending is not None and pending.abandoned:
                self.poll()
                pending = self.pending
                if pending is not None:
                    return Handoff("copy_pending")
            if pending is None:
                self.queue.pop(0)
                self._ids.discard(ticket.request_id)
                return Handoff("foreground")
            if pending.ticket != ticket:
                raise RuntimeError("prefetch ticket changed before handoff")
            ready = self.poll()
            if ready is None:
                if self.pending is None:
                    self.queue.pop(0);self._ids.discard(ticket.request_id)
                    return Handoff('foreground')
                return Handoff("copy_pending")
            try:
                result = pending.handle.adopt(ticket.request_id, ready)
            except BaseException:
                self.failed = True
                pending.handle.abandon()
                raise
            self.pending = None
            self.queue.pop(0)
            self._ids.discard(ticket.request_id)
            self.events.append(dict(event="adopted", request_id=ticket.request_id,
                                    generation=ready.generation,primary_weights_ready=ready.primary_weights_ready,
                                    monotonic_ns=time.monotonic_ns()))
            return Handoff("ready" if ready.primary_weights_ready else "auxiliary_ready", result)
