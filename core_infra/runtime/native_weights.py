"""Default native weight residency transport for any registered model.

Registered models declare their integration contract, bytes and selection for
the standard ABI. The resident service rejects custom transport bindings.
Low-level command descriptors remain available to transport tests and tools.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
import threading
import time
import uuid

try:
    from .native_control import NativeControlClient
    from .weight_prefetch import CopyPlan, ReadyWeights, PartialWeights
except ImportError:  # Flat deployment beside the existing native controller.
    from native_control import NativeControlClient
    from weight_prefetch import CopyPlan, ReadyWeights, PartialWeights


@dataclass(frozen=True)
class WeightRequirements:
    device_bytes: int
    pinned_host_bytes: int = 0
    cold_headroom_bytes: int = 0
    warm_headroom_bytes: int = 0
    gpu_prepare_required: bool = False
    auxiliary_pinned_host_bytes: int = 0
    auxiliary_device_bytes: int = 0
    execution_device_bytes: int = 0
    cached_device_bytes: int = 0

    def __post_init__(self):
        if any(type(value) is not int or value < 0 for value in (
                self.device_bytes, self.pinned_host_bytes, self.cold_headroom_bytes,
                self.warm_headroom_bytes, self.auxiliary_pinned_host_bytes,
                self.auxiliary_device_bytes,self.execution_device_bytes,self.cached_device_bytes)) or type(self.gpu_prepare_required) is not bool:
            raise ValueError('weight requirements must declare nonnegative byte bounds')

    def plan(self, ticket, warmed):
        if type(warmed) is not bool:
            raise ValueError('host/context residency must be explicit')
        return CopyPlan(ticket.model,
            self.device_bytes + (self.warm_headroom_bytes if warmed else self.cold_headroom_bytes),
            0 if warmed else self.pinned_host_bytes, gpu_prepare_required=self.gpu_prepare_required)


@dataclass(frozen=True)
class NativePreloadComponent:
    """One immutable component declared by a migrated model, for one GPU."""
    name: str
    priority: int
    tensor_count: int
    payload_bytes: int
    manifest_sha256: str
    reader_policy: tuple[int, int, int, int]
    roles: tuple[str, ...]
    scope: str = 'before_first_dit'
    transient_tensor_count: int = 0
    transient_payload_bytes: int = 0

    def __post_init__(self):
        if (not isinstance(self.name,str) or not self.name or type(self.priority) is not int or self.priority<0 or
                any(type(n) is not int or n<=0 for n in (self.tensor_count,self.payload_bytes)) or
                not isinstance(self.manifest_sha256,str) or len(self.manifest_sha256)!=64 or
                any(c not in '0123456789abcdef' for c in self.manifest_sha256) or
                type(self.reader_policy) is not tuple or len(self.reader_policy)!=4 or
                any(type(n) is not int or n<=0 for n in self.reader_policy) or
                any(type(n) is not int or n<0 for n in (self.transient_tensor_count,self.transient_payload_bytes)) or
                bool(self.transient_tensor_count)!=bool(self.transient_payload_bytes) or
                type(self.roles) is not tuple or not self.roles or len(set(self.roles))!=len(self.roles) or
                not set(self.roles)<={'encoder','dit','decoder','stage_bridge','host_asset'} or
                self.scope not in ('before_first_dit','after_first_dit','host_only')):
            raise ValueError('preload component requires identity, priority, exact inventory and cold reader policy')


@dataclass(frozen=True)
class NativePreloadPlan:
    components: tuple[NativePreloadComponent, ...]
    peers: tuple[tuple[NativePreloadComponent, ...], ...] = ()

    def __post_init__(self):
        if (type(self.components) is not tuple or not self.components or
                any(type(c) is not NativePreloadComponent for c in self.components) or
                len({c.name for c in self.components})!=len(self.components)):
            raise ValueError('preload plan requires unique immutable components')
        before=[c for c in self.components if c.scope=='before_first_dit']
        dit=[c.priority for c in before if 'dit' in c.roles]
        if not dit:raise ValueError('preload plan must declare the first DiT dependency')
        if any(c.priority>=min(dit) for c in before if 'encoder' in c.roles):
            raise ValueError('first-DiT encoders must have priority before DiT')
        if any(c.priority<=max(dit) for c in self.components if set(c.roles)=={'decoder'}):
            raise ValueError('decoder-only components must have priority after DiT')
        if type(self.peers) is not tuple:
            raise ValueError('preload peers must declare physical rank inventories')
        for components in self.peers:
            # A peer may hold only the DiT. Never infer its inventory by dividing
            # the primary or aggregate weight quantity by the world size.
            NativePreloadPlan(components)
            if not {c.name for c in components}<={c.name for c in self.components}:
                raise ValueError('peer preload declares an unknown component')

    @property
    def ranks(self):
        return (self.components, *self.peers)


@dataclass(frozen=True)
class NativeModelProtocol:
    """Required declaration for the shared resident serving route.

    Component priorities describe first use; native loaders implement their
    layout-specific order. The framework owns every transport operation and
    checks effective reader policies through warm and copy receipts.
    """
    loader: str
    world_size: int
    components: tuple[str, ...]
    phases: frozenset[str]
    host_ready_field: tuple[str, ...]
    host_ready_value: str | bool
    component_preload: bool = False

    def __post_init__(self):
        if (not isinstance(self.loader,str) or not self.loader or
                type(self.world_size) is not int or self.world_size<1 or
                type(self.components) is not tuple or not self.components or
                any(type(c) is not str or not c for c in self.components) or
                len(set(self.components))!=len(self.components) or
                type(self.phases) is not frozenset or any(type(p) is not str or not p for p in self.phases) or
                type(self.host_ready_field) is not tuple or not self.host_ready_field or
                any(type(p) is not str or not p for p in self.host_ready_field) or
                type(self.host_ready_value) not in (str,bool) or type(self.component_preload) is not bool):
            raise ValueError('native integration requires loader, topology, ordered components and declared phases')


_STANDARD_BINDING=object()


@dataclass(frozen=True)
class NativeCopySpec:
    command: tuple[str, ...]
    expected_fields: tuple[tuple[str, object], ...] = ()

    def __post_init__(self):
        if (type(self.command) is not tuple or not self.command or
                any(type(item) is not str for item in self.command)):
            raise ValueError('native copy command must be an immutable argument tuple')
        if (type(self.expected_fields) is not tuple or
                any(type(item) is not tuple or len(item) != 2 or type(item[0]) is not str or
                    type(item[1]) not in (str, bool, int) for item in self.expected_fields)):
            raise ValueError('receipt expectations must be immutable scalar fields')
        if len({name for name, _ in self.expected_fields}) != len(self.expected_fields):
            raise ValueError('duplicate native receipt expectation')


class NativeAuxiliaryCopy:
    """Prepare the first-use conditioner without claiming a DiT generation.

    The Python worker only waits for the existing native owner's completion;
    it is not a weight-reader pool and changes no cold-start I/O policy.
    """
    def __init__(self,backend,ticket):
        self.backend,self.ticket=backend,ticket
        self.preload_plan=backend.declared_preload(ticket)
        self.components=backend._auxiliary_components
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='native-auxiliary-receipt')
        self.started_ns=time.monotonic_ns()
        self.future=self.pool.submit(backend.prefetch_auxiliary)
        self.ready=self.receipt=self.failure=None
        self.abandoned=self.reaped=False

    def poll(self):
        if self.failure is not None:raise self.failure
        if self.ready is None:
            if not self.future.done():return None
            try:
                value=self.future.result();self.backend.owner.observe()
                if self.backend._auxiliary_device_reservation()!=0:
                    raise RuntimeError('auxiliary copy completed without publishing its resident weights')
                now=time.monotonic_ns()
                self.receipt=dict(request_id=self.ticket.request_id,recipe_sha256=self.ticket.model.recipe_sha256,
                    owner_pid=self.backend.owner_pid,device_uuid=self.backend.device_uuid,generation=0,
                    primary_weights_ready=False,primary_weights_deferred=True,auxiliary_weights=value,
                    gpu_prepare_complete=False,copy_complete=True,payload_verified=True,
                    host_pinned_bytes=self.backend._observed_host_bytes,
                    preload_inventory=deepcopy(value['preload_inventory']),
                    device_bytes_uploaded=sum(r['uploaded_payload_bytes'] for r in value['preload_inventory']['components']),
                    reader_policy=value['preload_inventory']['components'][0]['reader_policy'],
                    readiness_scope='auxiliary immutable weights only; DiT preparation required at serial head',
                    copy_operation_started_monotonic_ns=self.started_ns,
                    declared_preparation_completed_monotonic_ns=now)
                if 'rank_receipts' in value:
                    self.receipt['all_ranks_completed']=value.get('all_ranks_completed')
                    self.receipt['rank_receipts']=deepcopy(value['rank_receipts'])
                    for rank in self.receipt['rank_receipts']:
                        rank['gpu_prepare_complete']=False
                    self.receipt['device_bytes_uploaded']=sum(r['device_bytes_uploaded'] for r in self.receipt['rank_receipts'])
                self.backend.observe_receipt(self.receipt,operation='prefetch',expected_plan=self.preload_plan,
                    selected_components=self.components)
                self.ready=PartialWeights(self.ticket.model,self.backend.owner_pid,self.backend.device_uuid,
                    self.components,self.receipt)
            except BaseException as error:self.failure=error;raise
            finally:self.pool.shutdown(wait=False)
        if self.abandoned and not self.reaped:
            self.backend.release_auxiliary();self.backend._resident_components.clear();self.reaped=True
        return self.ready

    def abandon(self):self.abandoned=True

    def adopt(self,request_id,ready):
        if self.abandoned or self.ready is None or ready!=self.ready or request_id!=self.ticket.request_id:
            raise RuntimeError('auxiliary adoption ticket changed')
        return self.receipt


class NativeWeightCopy:
    def __init__(self, backend, ticket, spec, *, window=None, deferred_auxiliary=False,components=None):
        self.backend, self.ticket, self.spec = backend, ticket, spec
        self.preload_plan=backend.declared_preload(ticket)
        self.components=components
        self.deferred_auxiliary = deferred_auxiliary
        self.started_ns = time.monotonic_ns()
        self.path = backend.owner.directory / ('copy-' + uuid.uuid4().hex + '.json')
        self.client = None
        self.receipt = self.ready = None
        self.native_ready = None
        self.failure = None
        self.native_complete = False
        self.pending_failure = None
        self.auxiliary_pool = self.auxiliary = self.workspace = None
        self.auxiliary_value=None
        # A still cold-loading auxiliary is joined by its model's reply, not this copy.
        self.cold_auxiliary=bool(getattr(backend,'_auxiliary_loading',lambda:False)())
        self.ordered_auxiliary=bool(getattr(backend,'_auxiliary_inventory',None) is not None and not deferred_auxiliary and not self.cold_auxiliary)
        workspace_safe=(getattr(window,'source_gpu_complete',False) and
                        getattr(backend,'requires_idle_workspace_release',False))
        copy_auxiliary = getattr(backend,'prefetch_auxiliary',None) is not None and not deferred_auxiliary and not self.cold_auxiliary
        if copy_auxiliary or workspace_safe:
            # Existing native owners execute the actual transfer. This worker
            # only waits for its receipt beside the primary model's copy.
            self.auxiliary_pool = ThreadPoolExecutor(max_workers=2,thread_name_prefix='native-preparation')
            if copy_auxiliary:
                self.auxiliary = self.auxiliary_pool.submit(backend.prefetch_auxiliary)
            if workspace_safe:
                # Only a declared GPU-complete tail can drain decoder workers
                # beside DMA. Do not block the source's CPU media work.
                self.workspace = self.auxiliary_pool.submit(backend.owner.release_idle_workspaces,lambda:None)
        # Conditioners are first-DiT prerequisites. Submit them before DiT,
        # without serializing their transfer or changing native reader budgets.
        try:
            if not self.ordered_auxiliary:
                self.client = NativeControlClient(backend.owner.socket_paths['dit'], list(spec.command), self.path)
        except BaseException:
            if self.auxiliary_pool is not None:self.auxiliary_pool.shutdown(wait=True)
            raise
        self.abandoned = self.reaped = False

    def poll(self):
        if self.failure is not None:raise self.failure
        if self.client is None and not self.native_complete:
            if self.auxiliary is None or not self.auxiliary.done():return None
            try:
                self.auxiliary_value=self.auxiliary.result()
                self.client=NativeControlClient(self.backend.owner.socket_paths['dit'],list(self.spec.command),self.path)
            except BaseException as error:
                self.pending_failure=error;self.native_complete=True
        if not self.native_complete:
            try:complete=self.client.poll()
            except BaseException as error:
                self.pending_failure=error
                complete=-1
            if complete is None:return None
            try:
                if self.pending_failure is not None:raise self.pending_failure
                response = json.loads(self.path.read_bytes())
                if response.get('ok') is not True:
                    raise RuntimeError('native weight copy failed: ' + str(response)[:2000])
                value = response['details']
                self.backend.owner.verify_identity(value)
                if value.get('request_id') != self.ticket.request_id:
                    raise RuntimeError('native copy returned another request ticket')
                for name, expected in self.spec.expected_fields:
                    if type(value.get(name)) is not type(expected) or value[name] != expected:
                        raise RuntimeError('native copy receipt differs from admitted weights: ' + name)
                self.native_ready=ReadyWeights(self.ticket.model, value['owner_pid'], value['device_uuid'],
                    value['generation'], value['payload_verified'], value['copy_complete'], value['gpu_prepare_complete'])
                self.receipt = value
                self.receipt.update(copy_operation_started_monotonic_ns=self.started_ns,
                    primary_copy_operation_completed_monotonic_ns=time.monotonic_ns())
                if self.deferred_auxiliary:
                    self.receipt['auxiliary_weights_deferred'] = True
                    self.receipt['readiness_scope'] = 'primary immutable weights; auxiliary activation required at serial head'
            except BaseException as error:
                # The complete native error frame is terminal. Cleanup may
                # poll again while retiring the owner; retain that first error
                # instead of re-reading a frame file already removed below.
                self.pending_failure=error
            finally:
                self.native_complete = True
                self.client.close()
                self.path.unlink(missing_ok=True)
        if self.auxiliary_pool is not None:
            tasks=(('auxiliary_weights',self.auxiliary),('framework_idle_workspace_release',self.workspace))
            if any(task is not None and not task.done() for _,task in tasks):return None
            for name,task in tasks:
                if task is None:continue
                try:
                    value=task.result()
                    if self.receipt is not None:
                        self.receipt[name]=value
                        if name=='framework_idle_workspace_release':
                            self.receipt['framework_idle_workspaces_released']=True
                except BaseException as error:
                    if self.pending_failure is None:self.pending_failure=error
            self.auxiliary_pool.shutdown(wait=False)
            self.auxiliary_pool = self.auxiliary = self.workspace = None
        if self.pending_failure is not None:
            self.failure=self.pending_failure
            raise self.failure
        if self.ready is None:
            try:
                plan=self.preload_plan
                if self.cold_auxiliary:
                    self.receipt['auxiliary_weights_loading']=True
                    plan=replace(plan,components=tuple(c for c in plan.components if c.name not in self.backend._auxiliary_components))
                elif getattr(self.backend,'_auxiliary_inventory',None) is not None:
                    auxiliary=self.receipt.get('auxiliary_weights') or self.auxiliary_value or self.backend._auxiliary_inventory()
                    self.backend.merge_component_receipts(self.receipt,auxiliary,operation='prefetch')
                if (not self.native_ready.payload_verified or not self.native_ready.copy_complete or
                        not self.native_ready.gpu_prepare_complete and self.components is None):
                    raise RuntimeError('native preload is incomplete outside an admitted partial component copy')
                self.backend.observe_receipt(self.receipt,operation='prefetch',expected_plan=plan,
                    selected_components=self.components)
            except BaseException as error:
                self.failure=error
                raise
            self.ready=(self.native_ready if self.native_ready.gpu_prepare_complete else
                PartialWeights(self.ticket.model,self.backend.owner_pid,self.backend.device_uuid,self.components,self.receipt))
            self.receipt['declared_preparation_completed_monotonic_ns'] = time.monotonic_ns()
        if self.abandoned and not self.reaped:
            self.backend.discard()
            self.reaped = True
        return self.ready

    def abandon(self):
        # Abandon only the consumer. The native event still owns DMA memory.
        self.abandoned = True

    def adopt(self, request_id, ready):
        if self.abandoned or self.ready is None or self.ready != ready or request_id != self.ticket.request_id:
            raise RuntimeError('copy adoption ticket changed')
        return self.receipt


class NativeWeightBackend:
    def __init__(self, owner, family, *, plan, copy, discard,
                 requires_idle_workspace_release=False, requires_idle_auxiliary_release=False,
                 on_submit=None, host_residency=lambda: 0,
                 reclaim_during_decode=False, activate=None, auxiliary_host_residency=lambda: 0,
                 auxiliary_host_limit=0, prefetch_auxiliary=None):
        if type(reclaim_during_decode) is not bool or type(auxiliary_host_limit) is not int or auxiliary_host_limit<0:
            raise ValueError('auxiliary residency and decode release must be explicitly bounded')
        self.owner, self.family = owner, family
        self.owner_pid, self.device_uuid = owner.identity['owner_pid'], owner.identity['device_uuid']
        self._plan, self._copy, self._discard = plan, copy, discard
        self.requires_idle_workspace_release = requires_idle_workspace_release
        self.requires_idle_auxiliary_release = requires_idle_auxiliary_release
        self.on_submit = on_submit
        self.prefetch_auxiliary = prefetch_auxiliary
        self._host_residency = host_residency
        self.reclaim_during_decode = reclaim_during_decode
        self._activate = activate
        self._auxiliary_host_residency = auxiliary_host_residency
        self._auxiliary_host_limit = auxiliary_host_limit
        self._observed_host_bytes = 0
        self._initialized = False
        self._host_limit = None
        self._preload = None
        self._discard_primary = self._release_auxiliary = None
        self._execution_reservation = self._resident_device_residency = None
        self._execution_limit = self._resident_device_limit = 0
        self.protocol = None
        self._standard_binding = self._sealed_operations = None
        self._reader_policy = None
        self._foreground_prepare = None
        self._preload_plan = None
        self._auxiliary_inventory=None
        self._auxiliary_components=()
        self._resident_components={}
        self._resident_rank_components={}
        self._host_selection = None
        self._host_readiness = {}
        self._host_readiness_lock = threading.RLock()

    def validate_registration(self, adapter):
        if (type(self) is not NativeWeightBackend or self._standard_binding is not _STANDARD_BINDING or
                not isinstance(self.protocol,NativeModelProtocol)):
            raise TypeError('resident models require NativeWeightBackend.standard with a NativeModelProtocol')
        if any(name in self.__dict__ for name in ('plan','start','preload','discard','discard_primary','release_auxiliary')):
            raise TypeError('model overrides a framework-owned weight operation')
        actual=(self._plan,self._copy,self._preload,self._discard,self._discard_primary,self._release_auxiliary,
                self._host_selection,self._preload_plan,self._auxiliary_inventory)
        if actual!=self._sealed_operations:
            raise TypeError('model replaced the sealed standard weight transport')
        if not callable(self._preload_plan):
            raise TypeError('migrated models must declare their preload plan')
        devices=self.owner.identity.get('devices',[self.owner.identity])
        if len(devices)!=self.protocol.world_size or len({v['device_uuid'] for v in devices})!=len(devices):
            raise ValueError('model integration topology differs from the physical native owner')
        if frozenset(adapter.windows)!=self.protocol.phases:
            raise ValueError('model windows differ from its integration protocol')
        if not callable(getattr(adapter,'startup',None)):
            raise TypeError('resident model must declare host-warm startup selections')

    def prepare_foreground(self,ticket,checkpoint):
        if self._foreground_prepare is not None:self._foreground_prepare(ticket,checkpoint)

    @property
    def host_pinned_bytes(self):
        return self._primary_host_bytes() + self.auxiliary_host_pinned_bytes

    def _primary_host_bytes(self):
        value = self._host_residency()
        if type(value) is not int or value < 0:
            raise ValueError('host residency must report nonnegative bytes')
        held = max(value, self._observed_host_bytes)
        if self._host_limit is not None and held > self._host_limit:
            raise RuntimeError('native host residency differs from its declared bound')
        return held

    @property
    def auxiliary_host_pinned_bytes(self):
        value=self._auxiliary_host_residency()
        if type(value) is not int or not 0 <= value <= self._auxiliary_host_limit:
            raise RuntimeError('auxiliary host residency differs from its declared bound')
        return value

    @property
    def retained_host_allocation_bytes(self):
        """Reserve a first immutable auxiliary snapshot before device eviction."""
        return self._auxiliary_host_limit-self.auxiliary_host_pinned_bytes

    def activate(self, checkpoint=lambda:None):
        return self._activate(checkpoint) if self._activate is not None else None

    def declared_preload(self,ticket):
        if self.protocol is None:return None
        if not callable(self._preload_plan):
            raise TypeError('migrated models must declare their preload plan')
        plan=self._preload_plan(ticket)
        if type(plan) is not NativePreloadPlan or {c.name for c in plan.components}!=set(self.protocol.components):
            raise ValueError('preload plan does not cover the registered model components')
        if len(plan.ranks)!=self.protocol.world_size:
            raise ValueError('preload plan does not cover every physical rank')
        return plan

    def verify_preload_inventory(self,value,operation,plan,selected_components=None):
        if type(plan) is not NativePreloadPlan:
            raise RuntimeError('native readiness has no admitted preload declaration')
        devices=self.owner.identity.get('devices',[self.owner.identity])
        if len(plan.ranks)!=len(devices):
            raise RuntimeError('native preload declaration has another physical topology')
        if len(devices)==1:
            return self._verify_rank_inventory(value,operation,plan,self.device_uuid,selected_components)
        ranks=value.get('rank_receipts')
        if (value.get('all_ranks_completed') is not True or type(ranks) is not list or
                len(ranks)!=len(devices) or any(type(row) is not dict for row in ranks)):
            raise RuntimeError('native preload requires a completed receipt from every rank')
        if value.get('preload_inventory')!=ranks[0].get('preload_inventory'):
            raise RuntimeError('native primary inventory differs from its rank receipt')
        for components,device,row in zip(plan.ranks,devices,ranks):
            if row.get('device_uuid')!=device['device_uuid']:
                raise RuntimeError('native preload rank has another physical GPU')
            selected=(tuple(n for n in selected_components if n in {c.name for c in components})
                      if selected_components is not None else None)
            self._verify_rank_inventory(row,operation,NativePreloadPlan(components),device['device_uuid'],selected)
        if operation=='prefetch' and (type(value.get('device_bytes_uploaded')) is not int or
                value['device_bytes_uploaded']!=sum(row['device_bytes_uploaded'] for row in ranks)):
            raise RuntimeError('native preload aggregate upload differs from physical rank receipts')

    def _verify_rank_inventory(self,value,operation,plan,device_uuid,selected_components=None):
        inventory=value.get('preload_inventory')
        if type(inventory) is not dict or inventory.get('schema')!='native-preload-inventory-v1':
            raise RuntimeError('native readiness is missing its actual preload inventory')
        actual=inventory.get('components')
        partial=operation=='prefetch' and selected_components is not None and value.get('gpu_prepare_complete') is False
        expected_names={c.name for c in plan.components}
        if (type(actual) is not list or any(type(row) is not dict for row in actual) or
                not actual or len({r.get('name') for r in actual})!=len(actual)):
            raise RuntimeError('native preload inventory omits or duplicates components')
        rows={row['name']:row for row in actual}
        if set(rows)!=expected_names and not (partial and set(rows)==set(selected_components)):
            raise RuntimeError('native preload components differ from the declaration')
        policy_fields=('workers','chunk_bytes','pinned_slots','budget_bytes')
        uploaded=[];upload_bytes=0;priorities={c.name:c.priority for c in plan.components}
        if partial and (not self.protocol.component_preload or not set(selected_components)<=set(rows)):
            raise RuntimeError('native partial preload was not admitted')
        for component in plan.components:
            if component.name not in rows:continue
            row=rows[component.name]
            required=dict(priority=component.priority,tensor_count=component.tensor_count,
                payload_bytes=component.payload_bytes,manifest_sha256=component.manifest_sha256)
            if any(type(row.get(k)) is not type(v) or row[k]!=v for k,v in required.items()):
                raise RuntimeError('native preload identity, priority or quantity differs: '+component.name)
            for suffix in ('tensor_count','payload_bytes'):
                declared=getattr(component,'transient_'+suffix)
                if row.get('transient_'+suffix,0)!=declared:
                    raise RuntimeError('native transient preload quantity differs: '+component.name)
            policy=row.get('reader_policy')
            if (type(policy) is not dict or set(policy)!=set(policy_fields) or
                    any(type(policy[k]) is not int for k in policy_fields) or
                    policy!=dict(zip(policy_fields,component.reader_policy))):
                raise RuntimeError('native preload reader policy differs: '+component.name)
            if (row.get('device_uuid')!=device_uuid or row.get('host_ready') is not True or
                    type(row.get('verified_tensor_count')) is not int or type(row.get('verified_payload_bytes')) is not int or
                    row.get('verified_tensor_count')!=component.tensor_count or
                    row.get('verified_payload_bytes')!=component.payload_bytes):
                raise RuntimeError('native preload host verification is incomplete: '+component.name)
            if operation=='prefetch' and component.scope=='before_first_dit':
                for suffix,required_count in (('tensor_count',component.tensor_count),('payload_bytes',component.payload_bytes)):
                    resident,added,reused=(row.get(prefix+'_'+suffix) for prefix in ('resident','uploaded','reused'))
                    if (any(type(n) is not int or n<0 for n in (resident,added,reused)) or
                            resident>required_count or added+reused!=resident or
                            (not partial or component.name in selected_components) and resident!=required_count):
                        raise RuntimeError('native preload upload/reuse quantity is incomplete: '+component.name)
                if (not partial or component.name in selected_components) and row.get('copy_complete') is not True:
                    raise RuntimeError('native component is not ready: '+component.name)
                if selected_components is not None and component.name not in selected_components and row['uploaded_payload_bytes']:
                    raise RuntimeError('native preload copied a component outside the admitted capacity plan')
                upload_bytes+=row['uploaded_payload_bytes']
                for suffix in ('tensor_count','payload_bytes'):
                    transient=row.get('transient_uploaded_'+suffix,0)
                    if type(transient) is not int or transient not in (0,getattr(component,'transient_'+suffix)):
                        raise RuntimeError('native transient upload escaped its declaration')
                    if selected_components is not None and component.name not in selected_components and transient:
                        raise RuntimeError('native transient upload escaped its capacity plan')
                if bool(row.get('transient_uploaded_tensor_count',0))!=bool(row.get('transient_uploaded_payload_bytes',0)):
                    raise RuntimeError('native transient tensor and byte counters disagree')
                upload_bytes+=row.get('transient_uploaded_payload_bytes',0)
                if row['uploaded_payload_bytes']:uploaded.append(component.name)
        order=inventory.get('dispatch_order')
        if (type(order) is not list or len(set(order))!=len(order) or any(n not in rows for n in order) or
                not set(uploaded)<=set(order) or
                [priorities[n] for n in order]!=sorted(priorities[n] for n in order)):
            raise RuntimeError('native preload dispatch violates the declared priority')
        if operation=='prefetch':
            total=value.get('device_bytes_uploaded')
            if type(total) is not int or total<0 or total!=upload_bytes:
                raise RuntimeError('native preload total upload quantity differs from its components')

    @staticmethod
    def merge_component_receipts(primary,auxiliary,*,operation):
        """Join actual receipts from separate native owners, never declarations."""
        if 'rank_receipts' in primary or 'rank_receipts' in (auxiliary or {}):
            left=primary.get('rank_receipts');right=(auxiliary or {}).get('rank_receipts')
            if (primary.get('all_ranks_completed') is not True or auxiliary.get('all_ranks_completed') is not True or
                    type(left) is not list or type(right) is not list or not left or len(left)!=len(right)):
                raise RuntimeError('composite native preload omits a physical rank')
            for rank,other in zip(left,right):
                if rank.get('device_uuid')!=other.get('device_uuid'):
                    raise RuntimeError('composite native preload ranks have different physical GPUs')
                NativeWeightBackend.merge_component_receipts(rank,other,operation=operation)
            primary['preload_inventory']=deepcopy(left[0]['preload_inventory'])
            if operation=='prefetch':primary['device_bytes_uploaded']=sum(r['device_bytes_uploaded'] for r in left)
            return
        left=primary.get('preload_inventory');right=(auxiliary or {}).get('preload_inventory')
        if any(type(v) is not dict or v.get('schema')!='native-preload-inventory-v1' for v in (left,right)):
            raise RuntimeError('composite native preload is missing an owner inventory')
        primary['preload_inventory']=dict(schema='native-preload-inventory-v1',
            components=deepcopy(right['components'])+deepcopy(left['components']),
            dispatch_order=list(right['dispatch_order'])+list(left['dispatch_order']))
        if operation=='prefetch':
            primary['device_bytes_uploaded']+=sum(r['uploaded_payload_bytes'] for r in right['components'])

    def observe_receipt(self, value, *, operation=None, expected_plan=None,selected_components=None):
        if self._host_limit is not None:
            pinned = value.get('host_pinned_bytes')
            used = value.get('host_resident_bytes',pinned)
            # Retained CPU bytes may await idle registration. Charge their
            # actual owned RAM once; they are not a pinned/readiness receipt.
            if (type(pinned) is not int or type(used) is not int
                    or not 0 <= pinned <= used <= self._host_limit):
                raise RuntimeError('native host residency differs from its declared bound')
            self._observed_host_bytes = used
            self._initialized = True
        if self.protocol is not None:
            if operation in ('warm','prefetch'):
                self.verify_preload_inventory(value,operation,expected_plan,selected_components)
            if operation=='warm':
                for receipt in [value,*value.get('rank_receipts',[])]:
                    actual=receipt
                    for field in self.protocol.host_ready_field:
                        actual=actual.get(field) if type(actual) is dict else None
                    if type(actual) is not type(self.protocol.host_ready_value) or actual!=self.protocol.host_ready_value:
                        raise RuntimeError('native host readiness does not prove the declared loader path on every rank')
            policies=[]
            def visit(node):
                if type(node) is dict:
                    if 'reader_policy' in node:policies.append(node['reader_policy'])
                    for key,child in node.items():
                        if key not in ('reader_policy','preload_inventory'):visit(child)
                elif type(node) is list:
                    for child in node:visit(child)
            visit(value)
            fields=('workers','chunk_bytes','pinned_slots','budget_bytes')
            if operation is None and not policies:return  # Background byte accounting is not a readiness receipt.
            if not policies or any(type(p) is not dict or set(p)!=set(fields) or
                    any(type(p[k]) is not int or p[k]<=0 for k in fields) for p in policies):
                raise RuntimeError('native loader must report its effective cold/preload reader policy')
            policy=policies[0]
            if operation in ('warm','prefetch'):
                dit_policies={c.reader_policy for c in expected_plan.components if 'dit' in c.roles}
                if tuple(policy[k] for k in fields) not in dit_policies:
                    raise RuntimeError('native primary reader policy differs from the declared cold loader')
            if any(p!=policy for p in policies) or (self._reader_policy is not None and self._reader_policy!=policy):
                raise RuntimeError('native preload reader policy differs from cold start or another rank')
            self._reader_policy=dict(policy)
            value['framework_protocol']=dict(schema='resident-native-v1',operation=operation,
                family=self.family,loader=self.protocol.loader,world_size=self.protocol.world_size,
                declared_components=list(self.protocol.components),
                component_order=list(value['preload_inventory']['dispatch_order']) if operation is not None else [],
                preload_inventory_verified=operation in ('warm','prefetch'),reader_policy=dict(policy))
            if operation=='prefetch':
                self._resident_components={r['name']:(r['manifest_sha256'],r['tensor_count'],r['payload_bytes'])
                    for r in value['preload_inventory']['components'] if r.get('copy_complete') is True}
                self._resident_rank_components={rank['device_uuid']:{r['name']:(r['manifest_sha256'],r['tensor_count'],r['payload_bytes'])
                    for r in rank['preload_inventory']['components'] if r.get('copy_complete') is True}
                    for rank in value.get('rank_receipts',[])}

    def plan(self, ticket):
        if ticket.model.family != self.family:
            raise ValueError('weight adapter received another model family')
        return self._plan(ticket)

    def plan_for_capacity(self, ticket, capacity):
        complete = self.plan(ticket)
        if (self.protocol is not None and self.protocol.component_preload and
                self._auxiliary_inventory is None and complete.device_bytes>capacity):
            declared=self.declared_preload(ticket)
            overhead=max(0,complete.device_bytes-max(sum(c.payload_bytes for c in rank) for rank in declared.ranks))
            available=max(0,capacity-overhead);selected=[];used=0
            for component in sorted(declared.components,key=lambda c:c.priority):
                identity=(component.manifest_sha256,component.tensor_count,component.payload_bytes)
                if (not declared.peers and self._resident_components.get(component.name)==identity or
                        declared.peers and all(self._resident_rank_components.get(device['device_uuid'],{}).get(c.name)==
                            (c.manifest_sha256,c.tensor_count,c.payload_bytes)
                            for rank,device in zip(declared.ranks,self.owner.identity['devices'])
                            for c in rank if c.name==component.name)):
                    continue
                size=max((c.payload_bytes for rank in declared.ranks for c in rank if c.name==component.name),default=0)
                if size<=available:
                    selected.append(component.name);used+=size;available-=size
            if selected:return replace(complete,device_bytes=used+overhead,components=tuple(selected))
        auxiliary = getattr(self, '_auxiliary_device_reservation', lambda: 0)()
        if complete.device_bytes > capacity and auxiliary and self.prefetch_auxiliary is not None and self._activate is not None:
            if auxiliary<=capacity:
                return replace(complete,device_bytes=auxiliary,pinned_host_bytes=0,gpu_prepare_required=False,deferred_primary=True,
                    components=self._auxiliary_components or None)
            if complete.device_bytes-auxiliary<=capacity:
                return replace(complete,device_bytes=complete.device_bytes-auxiliary,deferred_auxiliary=True)
        return complete

    def start(self, ticket, plan, window):
        if plan.deferred_primary:
            handle=NativeAuxiliaryCopy(self,ticket)
            if self.on_submit is not None:self.on_submit(ticket)
            return handle
        # Materialize all request-dependent fields before submitting DMA.
        # Completion never looks up a request that cancellation may remove.
        copy_window = (replace(window, free_device_bytes=min(window.copy_capacity, plan.device_bytes),
                               reclaimable_device_bytes=0, decoder_reserve_bytes=0)
                       if plan.deferred_auxiliary else window)
        spec = (self._copy(ticket,copy_window,components=plan.components) if plan.components is not None else
                self._copy(ticket,copy_window))
        if not isinstance(spec, NativeCopySpec):
            raise TypeError('native weight adapter must return a copy descriptor')
        handle = NativeWeightCopy(self, ticket, spec, window=window, deferred_auxiliary=plan.deferred_auxiliary,
            components=plan.components)
        if self.on_submit is not None:
            self.on_submit(ticket)
        return handle

    def discard(self):
        self._resident_components.clear()
        self._resident_rank_components.clear()
        return self._discard()

    def discard_primary(self):
        self._resident_components.clear()
        self._resident_rank_components.clear()
        return self._discard_primary() if self._discard_primary is not None else self.discard()

    def release_auxiliary(self):
        return self._release_auxiliary() if self._release_auxiliary is not None else None

    def execution_reservation(self,ticket):
        value=self._execution_reservation(ticket) if self._execution_reservation is not None else 0
        if type(value) is not int or not 0 <= value <= self._execution_limit:
            raise RuntimeError('execution device reservation differs from its declared bound')
        return value

    def resident_device_memory(self):
        value=(self._resident_device_residency() if self._resident_device_residency is not None else
               {uuid:sum(item[2] for name,item in components.items() if name not in self._auxiliary_components)
                for uuid,components in self._resident_rank_components.items()} if self._resident_rank_components else
               sum(item[2] for name,item in self._resident_components.items() if name not in self._auxiliary_components))
        if type(value) is dict:
            expected={row['device_uuid'] for row in self.owner.identity.get('devices',[self.owner.identity])}
            if not value or set(value)!=expected:
                raise RuntimeError('resident device payload has another physical GPU topology')
            sizes=value.values()
        else:sizes=(value,)
        if any(type(size) is not int or not 0<=size<=self._resident_device_limit for size in sizes):
            raise RuntimeError('resident device payload differs from its declared bound')
        return dict(value) if type(value) is dict else value

    def resident_device_bytes(self):
        value=self.resident_device_memory()
        return min(value.values()) if type(value) is dict else value

    def device_memory_budget(self, info, *, include_primary=False):
        """Join reusable allocations and free capacity on the same physical GPU."""
        if type(include_primary) is not bool:raise ValueError('primary reclamation must be explicit')
        resident=self.resident_device_memory()
        expected={row['device_uuid'] for row in self.owner.identity.get('devices',[self.owner.identity])}
        rows=info.get('devices',[info])
        if (not rows or len(rows)!=len(expected) or {row['device_uuid'] for row in rows}!=expected):
            raise RuntimeError('device budget has another physical GPU topology')
        devices=[]
        for row in rows:
            uuid=row['device_uuid'];free=row.get('device_free_bytes')
            primary=row.get('primary_reclaimable_bytes',0) if include_primary else 0
            reusable=resident[uuid] if type(resident) is dict else resident
            if any(type(size) is not int or size<0 for size in (free,primary)):
                raise RuntimeError('device budget requires nonnegative physical byte receipts')
            capacity=free+max(primary,reusable)
            total=row.get('device_total_bytes')
            if total is not None and (type(total) is not int or not 0<=capacity<=total):
                raise RuntimeError('reusable device budget exceeds physical GPU capacity')
            devices.append(dict(device_uuid=uuid,free_device_bytes=free,reusable_bytes=reusable,
                                primary_reclaimable_bytes=primary,capacity_bytes=capacity))
        return dict(capacity_bytes=min(row['capacity_bytes'] for row in devices),
                    resident_payload_bytes=min(row['reusable_bytes'] for row in devices),devices=devices)

    def _host_readiness_key(self, ticket):
        """Host proof survives GPU eviction, never a changed owner or selection."""
        if self.protocol is None or self._host_selection is None:return None
        identity=self.owner.identity
        rows=[identity,*identity.get('devices',[identity])]
        fields=('owner_pid','owner_instance','device_uuid')
        # Transport-only callers may have no lifetime token. Warm normally,
        # but never reuse a proof that cannot identify every participating rank.
        if (len(rows)-1!=self.protocol.world_size or any(
                type(row.get('owner_pid')) is not int or row['owner_pid']<=0 or
                any(type(row.get(k)) is not str or not row[k] for k in fields[1:]) for row in rows)):
            return None
        owner=tuple(tuple(row[k] for k in fields) for row in rows)
        selection=json.dumps(self._host_selection(ticket),sort_keys=True,separators=(',',':'),allow_nan=False)
        model=ticket.model
        return owner,(model.family,model.revision,model.recipe_sha256),selection

    def preload(self, ticket, checkpoint=lambda: None, *, force_revalidate=False):
        """Return verified immutable host readiness, with its original provenance.

        Diagnostic controls may force an actual native warm. The normal serving
        path reuses only a successful proof for this owner, recipe and selection;
        it does not manufacture a new native operation, generation or timestamp.
        """
        if ticket.model.family != self.family or self._preload is None:
            raise ValueError('this weight binding has no standard host preload operation')
        if type(force_revalidate) is not bool:raise ValueError('host revalidation must be explicitly boolean')
        begun=time.monotonic_ns();checkpoint()
        expected_plan=self.declared_preload(ticket)
        with self._host_readiness_lock:
            checkpoint();key=self._host_readiness_key(ticket)
            cached=None if force_revalidate or key is None else self._host_readiness.get(key)
            if cached is not None:
                value,proof=deepcopy(cached)
                try:
                    self.owner.observe()
                    self.owner.verify_identity(value)
                    retained=self._observed_host_bytes
                    self.observe_receipt(value,operation='warm',expected_plan=expected_plan)
                    # The receipt intentionally retains its original byte
                    # observation. Reusing an earlier selection must not undo
                    # host bytes subsequently retained by another selection.
                    self._observed_host_bytes=max(retained,self._observed_host_bytes)
                    checkpoint()
                    if self._host_readiness_key(ticket)!=key:
                        raise RuntimeError('host readiness owner or immutable selection changed')
                except BaseException:
                    self._host_readiness.pop(key,None)
                    raise
            else:
                if key is not None:self._host_readiness.pop(key,None)
                started=time.monotonic_ns()
                value=self._preload(ticket,checkpoint)
                self.owner.verify_identity(value)
                self.observe_receipt(value,operation='warm',expected_plan=expected_plan)
                checkpoint()
                if self._host_readiness_key(ticket)!=key:
                    raise RuntimeError('host readiness owner or immutable selection changed')
                ended=time.monotonic_ns()
                proof=dict(validation_id=uuid.uuid4().hex,validation_started_monotonic_ns=started,
                    validation_completed_monotonic_ns=ended,validation_seconds=(ended-started)/1e9)
                if key is not None:self._host_readiness[key]=deepcopy((value,proof))
            if self.protocol is not None:
                completed=time.monotonic_ns()
                value['framework_host_readiness']=dict(proof,cache_hit=cached is not None,
                    cache_eligible=key is not None,native_warm_called=cached is None,
                    force_revalidated=force_revalidate,lookup_started_monotonic_ns=begun,
                    lookup_completed_monotonic_ns=completed,lookup_seconds=(completed-begun)/1e9,
                    immutable_selection_sha256=hashlib.sha256(key[2].encode()).hexdigest() if key is not None else None)
            return value

    @classmethod
    def standard(cls, owner, family, *, requirements, payload, warmed=lambda: False,
                 requires_idle_workspace_release=False, requires_idle_auxiliary_release=False,
                 on_submit=None, on_evict=None,
                 reclaim_during_decode=False, on_activate=None, auxiliary_host_residency=lambda:0,
                 auxiliary_device_reservation=lambda:0, prefetch_auxiliary=None,
                 execution_reservation=None,resident_device_residency=None,on_primary_evict=None,preload_plan=None,
                 protocol=None,on_foreground_prepare=None,auxiliary_inventory=None,auxiliary_components=(),
                 auxiliary_loading=lambda:False):
        """Use warm/prefetch/evict from the standard native backend ABI.

        payload(ticket) supplies only model selection (for example a profile).
        Framework-owned operation, ticket and memory limits cannot be replaced.
        """
        def selection(ticket):
            extra = dict(payload(ticket))
            reserved = {'op', 'request_id', 'device_budget_bytes','preload_components'}
            if reserved & extra.keys():
                raise ValueError('model payload replaces framework-owned prefetch fields')
            return extra
        def copy(ticket, window, *, components=None):
            extra = selection(ticket)
            request = dict(extra, op='prefetch', request_id=ticket.request_id,
                           device_budget_bytes=min(window.copy_capacity,backend.plan(ticket).device_bytes))
            if components is not None:request['preload_components']=list(components)
            return NativeCopySpec(('backend-select', family, json.dumps(request)),
                                  (('recipe_sha256', ticket.model.recipe_sha256),))
        def discard_primary():
            value = owner.rpc(['backend', family, '{"op":"evict"}'])
            owner.verify_identity(value)
            return dict(native=value,model_state=on_primary_evict() if on_primary_evict is not None else None)
        def release_auxiliary():return on_evict() if on_evict is not None else None
        def evict():return dict(discard_primary(),auxiliary=release_auxiliary())
        def plan(ticket):
            extra=auxiliary_device_reservation()
            if type(extra) is not int or not 0 <= extra <= requirements.auxiliary_device_bytes:
                raise RuntimeError('auxiliary device reservation differs from its declared bound')
            value=requirements.plan(ticket,warmed() or backend._initialized)
            # A reusable device context does not prove every catalog selection
            # is retained in RAM. Reserve growth up to the primary host bound;
            # auxiliaries are charged separately and never reduce this reserve.
            remaining=requirements.pinned_host_bytes-backend._primary_host_bytes()
            return replace(value,device_bytes=value.device_bytes+extra,pinned_host_bytes=remaining)
        backend = cls(owner, family,
            plan=plan, copy=copy,
            discard=evict,
            requires_idle_workspace_release=requires_idle_workspace_release,
            requires_idle_auxiliary_release=requires_idle_auxiliary_release, on_submit=on_submit,
            host_residency=lambda: requirements.pinned_host_bytes if warmed() else 0,
            reclaim_during_decode=reclaim_during_decode,activate=on_activate,
            auxiliary_host_residency=auxiliary_host_residency,
            auxiliary_host_limit=requirements.auxiliary_pinned_host_bytes,prefetch_auxiliary=prefetch_auxiliary)
        backend._host_limit = requirements.pinned_host_bytes
        backend._discard_primary=discard_primary
        backend._release_auxiliary=release_auxiliary
        backend._execution_limit=requirements.execution_device_bytes
        backend._execution_reservation=execution_reservation or (lambda ticket:requirements.execution_device_bytes)
        backend._resident_device_residency=resident_device_residency
        backend._resident_device_limit=(requirements.device_bytes+requirements.cached_device_bytes+
            max(requirements.cold_headroom_bytes,requirements.warm_headroom_bytes))
        def warm(ticket,checkpoint):
            auxiliary=auxiliary_inventory() if auxiliary_inventory is not None else None
            value=owner.rpc(['backend',family,json.dumps(dict(selection(ticket),op='warm'))],checkpoint)
            if auxiliary_inventory is not None:backend.merge_component_receipts(value,auxiliary,operation='warm')
            return value
        backend._preload=warm
        backend._auxiliary_device_reservation = auxiliary_device_reservation
        if protocol is not None and not isinstance(protocol,NativeModelProtocol):
            raise TypeError('native model protocol must be an immutable NativeModelProtocol')
        backend.protocol=protocol
        backend._standard_binding=_STANDARD_BINDING
        backend._foreground_prepare=on_foreground_prepare
        backend._host_selection=selection
        backend._preload_plan=preload_plan
        backend._auxiliary_inventory=auxiliary_inventory
        backend._auxiliary_components=tuple(auxiliary_components)
        backend._auxiliary_loading=auxiliary_loading
        backend._sealed_operations=(backend._plan,backend._copy,backend._preload,backend._discard,
                                    backend._discard_primary,backend._release_auxiliary,backend._host_selection,
                                    backend._preload_plan,backend._auxiliary_inventory)
        return backend
