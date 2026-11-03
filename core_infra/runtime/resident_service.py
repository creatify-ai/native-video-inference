"""One admission queue for registered resident models and one native owner.

Platform entrypoints construct this service with model adapters. Adapters own
CPU request preparation and inference; the framework owns lookahead, tickets,
copy completion, cancellation, handoff and native process identity.
"""
import asyncio
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass
import json
import threading
import time
import uuid

try:
    from .weight_prefetch import DecodeWindow, Ticket, WeightPrefetchBroker
except ImportError:
    from weight_prefetch import DecodeWindow, Ticket, WeightPrefetchBroker


@dataclass
class Admitted:
    request: dict
    model: object
    payload: object = None


class RequestRejected(ValueError):
    """An adapter proved a late CPU-input rejection left its native owner intact.

    GPU/transport failures must never use this exception. Legacy adapters may
    perform final media admission inside their existing controller; they must
    verify that controller did not retire or leave an active transaction.
    """


class CopyAdmission:
    """One request's checked native copy, usable as a Preparation input.

    CPU/conditioning work may run while the ordinary foreground transport
    measures capacity and adopts its copy. A numerical consumer must wait for
    this receipt; submission or weight-only readiness cannot satisfy it.
    """
    def __init__(self, ticket, owner):
        self.owner = owner
        self.expected = dict(request_id=ticket.request_id,
            recipe_sha256=ticket.model.recipe_sha256,
            **{key: owner.identity[key] for key in ('owner_pid', 'owner_instance', 'device_uuid')})
        self.future = Future()

    def validate_completion(self, receipt):
        if not isinstance(receipt, dict):
            raise RuntimeError('copy admission requires a native completion receipt')
        self.owner.verify_identity(receipt)
        if any(type(receipt.get(key)) is not type(value) or receipt[key] != value
                for key, value in self.expected.items()):
            raise RuntimeError('copy admission belongs to another request or native owner')
        if (any(receipt.get(key) is not True for key in
                ('payload_verified', 'copy_complete', 'gpu_prepare_complete'))
                or type(receipt.get('generation')) is not int or receipt['generation'] < 0):
            raise RuntimeError('copy admission requires verified payload, DMA and GPU preparation')
        return dict(self.expected, generation=receipt['generation'], prepared=True)

    def complete(self, receipt):
        self.future.set_result(self.validate_completion(receipt))

    def fail(self, error):
        if not self.future.done():self.future.set_exception(error)

    def result(self):
        return self.future.result()

    def validate(self, receipt):
        return (self.future.done() and receipt == self.future.result()
                and receipt.get('prepared') is True
                and all(type(receipt.get(key)) is type(value) and receipt[key] == value
                    for key, value in self.expected.items()))


class DeclaredCopyGate:
    """Bind a checked copy to a mandatory Preparation in a real Rust graph.

    This permits CPU/conditioning dispatch while a framework-owned copy runs.
    The component's before-use gate still cannot admit numerical execution
    until the exact native copy receipt has passed both framework validators.
    """
    def __init__(self, startup, component, preparation, install, *, on_complete=None):
        try:from .native_startup import DeclaredStartup
        except ImportError:from native_startup import DeclaredStartup
        if (type(startup) is not DeclaredStartup or not callable(install)
                or on_complete is not None and not callable(on_complete)):
            raise TypeError('deferred copy requires a real Rust DeclaredStartup')
        consumer=startup.components.get(component)
        definition=startup.definitions.get(preparation)
        if (consumer is None or preparation not in (consumer.required_preparations or ())
                or definition is None or component not in definition.before
                or definition.at_use is not True or definition.prepare is not None
                or not definition.inputs):
            raise ValueError('deferred copy must be a mandatory request Preparation for the consumer')
        self.startup,self.component,self.name,self.install=startup,component,preparation,install
        self.on_complete=on_complete
        self.admission=None

    def bind(self,ticket,owner):
        from dataclasses import replace
        if self.admission is not None:raise RuntimeError('declared copy gate was already bound')
        admission=CopyAdmission(ticket,owner)
        # Native dispatch refers to this named definition. The framework
        # installs the wait and validator; a model callback cannot approve an
        # incomplete or foreign copy on the deferred path.
        previous=self.startup.definitions[self.name]
        with self.startup.lock:
            self.install(admission)
            self.startup.definitions[self.name]=replace(previous,
                run=lambda _:admission.result(),validate=admission.validate)
            self.admission=admission
        return self

    def admits_dispatch(self,startup,ticket,component):
        admission=self.admission
        return (self.startup is startup and self.component==component and admission is not None
            and admission.expected['request_id']==ticket.request_id
            and admission.expected['recipe_sha256']==ticket.model.recipe_sha256
            and not (admission.future.done() and admission.future.exception() is not None))

    def complete(self,receipt):
        self.admission.validate_completion(receipt)
        if self.on_complete is not None:self.on_complete(receipt)
        self.admission.complete(receipt)


class ResidentModelService:
    def __init__(self, base, adapters, *, owner_factory, host_budget_bytes, warm_on_foreground=True,
                 force_foreground_host_revalidation=False):
        self.base, self.adapters = base, tuple(adapters)
        self.routes = {}
        for adapter in self.adapters:
            for workflow in adapter.workflows:
                if workflow in self.routes:
                    raise ValueError('resident workflow registered twice')
                self.routes[workflow] = adapter
        self.owner_factory, self.host_budget_bytes = owner_factory, host_budget_bytes
        self.warm_on_foreground=warm_on_foreground
        if type(force_foreground_host_revalidation) is not bool:
            raise TypeError('foreground host revalidation must be an explicit diagnostic choice')
        self.force_foreground_host_revalidation=force_foreground_host_revalidation
        self.native_owner = self.broker = None
        self.backend_families = {}
        self.tickets, self.ticket_lock = {}, threading.RLock()
        self.lock = asyncio.Lock()
        self.active = self.window_phase = self.window_geometry = None
        self.failed = False
        self.loading = False
        self.extra_requests = 0
        self.calls = set()
        self._cancellation = None
        self.host_preload_ready=False
        self.startup_preparation=None
        self.preload_capacity_limit_bytes=None

    def startup_declarations(self, workflow):
        """Return model-declared, input-free selections for one registered route.

        Products choose the recipe/profile. Core never invents request media,
        shape hints, model names or a forward pass to obtain an eager target.
        """
        adapter=self.routes.get(workflow)
        if adapter is None:raise ValueError('startup workflow is not registered')
        startup=getattr(adapter,'startup',None)
        return [item for item in startup() if item.request.get('workflow')==workflow] if startup else []

    async def _start_preparation(self,adapter,admitted,checkpoint,cancellation):
        """Start owner-independent work only after this call owns the head.

        Adapters own asynchronous work in their service or admitted-call
        lifetime. Returning from this hook is dispatch, not readiness. Both
        eager and real requests dispose their admission after owned work joins.
        Cancellation joins the dispatch before owner creation can continue.
        """
        checkpoint()
        start=getattr(adapter,'start_preparation',None)
        if start is not None:
            await self.owned(lambda:start(admitted,checkpoint),cancellation)
        checkpoint()

    async def prepare_startup(self, admitted):
        """Prepare exactly one declared selection through the foreground path.

        The receipt covers the backend's declared weight/pre-init scope. It is
        not a first-DiT measurement or proof that request-specific work is done.
        Unrelated registered models do not pass through a preload barrier.
        """
        if not isinstance(admitted,Admitted):raise TypeError('startup requires a model declaration')
        workflow=admitted.request.get('workflow')
        if admitted not in self.startup_declarations(workflow):
            raise ValueError('startup target was not declared by the registered adapter')
        adapter=self.routes[workflow]
        cancellation=threading.Event();deadline=time.monotonic()+self.request_timeout
        def checkpoint():
            if cancellation.is_set() or self.closed.is_set():raise RuntimeError('resident startup cancelled')
            if time.monotonic()>=deadline:raise TimeoutError('resident startup deadline expired')
        call=asyncio.current_task();self.calls.add(call);item=None;started=False
        try:
            async with self.lock:
                checkpoint()
                if self.broken:raise RuntimeError('resident owner unavailable')
                started=True;self._cancellation=cancellation;self.loading=True
                await self._start_preparation(adapter,admitted,checkpoint,cancellation)
                await self.owned(lambda:self.bootstrap(checkpoint),cancellation)
                item=dict(ticket=Ticket(uuid.uuid4().hex,admitted.model),admitted=admitted,adapter=adapter,alive=True)
                with self.ticket_lock:
                    self._bind(item);self.tickets[item['ticket'].request_id]=item
                handoff={'status':'foreground'}
                await self.owned(lambda:self.prepare_transition(item,checkpoint,handoff),cancellation)
                prepare=getattr(adapter,'prepare_startup',None)
                if prepare is not None:
                    await self.owned(lambda:prepare(admitted,self.native_owner,checkpoint),cancellation)
                receipt=await self.owned(lambda:self.prepare_weights(item,checkpoint,handoff),cancellation)
                finish=getattr(adapter,'finish_startup',None)
                if finish is not None:
                    await self.owned(lambda:finish(admitted,self.native_owner,checkpoint),cancellation)
                remaining=getattr(adapter,'startup_remaining',lambda _:['request-specific preparation'])
                self.startup_preparation=dict(workflow=workflow,model=vars(item['ticket'].model),
                    receipt=receipt,forward_executed=False,request_preparation_complete=False,
                    remaining_request_preparation=list(remaining(admitted)))
                return self.startup_preparation
        except BaseException:
            if started:
                self.failed=True;cancellation.set()
                if self.native_owner is not None:
                    try:await asyncio.to_thread(self.native_owner.retire)
                    except BaseException as error:
                        print(json.dumps(dict(event='resident_retirement_failed',error=str(error))),flush=True)
            raise
        finally:
            if item is not None:
                with self.ticket_lock:
                    item['alive']=False;self.tickets.pop(item['ticket'].request_id,None)
            if started:self._cancellation=None;self.loading=False
            try:await self.owned(lambda:adapter.dispose(admitted),cancellation)
            finally:self.calls.discard(call)

    @staticmethod
    @contextmanager
    def preparation_span(handoff, phase):
        """Retain overlapping host intervals; these are not CUDA DMA timers."""
        span=dict(phase=phase,started_monotonic_ns=time.monotonic_ns(),passed=False)
        try:
            yield span
            span['passed']=True
        finally:
            span['completed_monotonic_ns']=time.monotonic_ns()
            handoff.setdefault('preparation_trace',[]).append(span)

    async def preload_models(self):
        """Verify registered host weights before a replica becomes ready.

        Models declare shape-independent warm selections. No model forward or
        request activation is executed, and each backend retains its host cache.
        """
        async with self.lock:
            if self.host_preload_ready:return
            cancellation=threading.Event();deadline=time.monotonic()+self.request_timeout
            def checkpoint():
                if self.closed.is_set() or cancellation.is_set() or time.monotonic()>deadline:
                    raise RuntimeError('resident host preload cancelled or timed out')
            def warm():
                self.bootstrap(checkpoint)
                for adapter in self.adapters:
                    selections=tuple(adapter.startup())
                    if not selections or len(selections)>1024:
                        raise ValueError('resident model must declare 1..1024 host-warm startup selections')
                    for admitted in selections:
                        item=dict(ticket=Ticket(uuid.uuid4().hex,admitted.model),admitted=admitted,adapter=adapter,alive=True)
                        with self.ticket_lock:
                            self._bind(item);self.tickets[item['ticket'].request_id]=item
                        try:self.broker.backends[admitted.model.family].preload(item['ticket'],checkpoint)
                        finally:
                            with self.ticket_lock:self.tickets.pop(item['ticket'].request_id,None)
                self.host_preload_ready=True
            try:await self.owned(warm,cancellation)
            except BaseException:
                self.failed=True;cancellation.set()
                if self.native_owner is not None:
                    try:self.native_owner.retire()
                    except BaseException as error:
                        print(json.dumps(dict(event='host_preload_retirement_failed',error=str(error))),flush=True)
                raise

    def __getattr__(self, name):
        return getattr(self.base, name)

    @property
    def request_timeout(self):return self.base.request_timeout
    @request_timeout.setter
    def request_timeout(self, value):self.base.request_timeout = value
    @property
    def on_resident_failure(self):return self.base.on_resident_failure
    @on_resident_failure.setter
    def on_resident_failure(self, value):self.base.on_resident_failure = value
    @property
    def factory(self):return self.base.factory
    @factory.setter
    def factory(self, value):
        self.base.factory=value
    @property
    def broken(self):return self.failed or self.base.broken or bool(self.broker and self.broker.failed)
    @property
    def awaiting_recipe(self):
        return self.extra_requests==0 and bool(getattr(self.base,'awaiting_recipe',False))

    def lookup(self, ticket):
        with self.ticket_lock:
            return self.tickets[ticket.request_id]['admitted']

    async def owned(self, action, cancellation, *, dispose=None):
        task = asyncio.create_task(asyncio.to_thread(action))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            cancellation.set()
            # Cancellation releases the consumer only. Join native work before
            # deleting CPU inputs, releasing reservations or retiring owners.
            while not task.done():
                try:await asyncio.shield(task)
                except asyncio.CancelledError:continue
                except BaseException:break
            try:
                value=task.result()
                if dispose is not None:await asyncio.to_thread(dispose,value)
            except BaseException:pass
            raise

    async def owned_parallel(self,actions,cancellation,*,disposers=None):
        """Join independent owned work before exposing results or retiring owners.

        An ordinary input error does not abort an in-flight weight copy. Caller
        cancellation joins both calls before releasing any completed input lease.
        """
        actions=tuple(actions)
        disposers=tuple(disposers) if disposers is not None else (None,)*len(actions)
        if (len(actions)!=len(disposers) or any(not callable(action) for action in actions)
                or any(dispose is not None and not callable(dispose) for dispose in disposers)):
            raise ValueError('parallel work requires one optional disposer per action')
        missing=object();completed=[missing]*len(actions)
        def run(index):
            value=actions[index]();completed[index]=value;return value
        tasks=[asyncio.create_task(self.owned(lambda index=index:run(index),cancellation))
               for index in range(len(actions))]
        joined=asyncio.gather(*tasks,return_exceptions=True)
        async def finish(task):
            while not task.done():
                try:await asyncio.shield(task)
                except asyncio.CancelledError:continue
            return task.result()
        async def dispose_completed():
            # A cancelled async waiter may hide a successfully completed thread
            # result. Retain it independently until every peer has joined.
            for value,dispose in zip(completed,disposers):
                if dispose is not None and value is not missing:
                    try:await asyncio.to_thread(dispose,value)
                    except BaseException as error:
                        print(json.dumps(dict(event='resident_parallel_disposal_failed',
                                              error_type=type(error).__name__,error=str(error))),flush=True)
        try:results=await asyncio.shield(joined)
        except asyncio.CancelledError:
            cancellation.set()
            for task in tasks:task.cancel()
            results=await finish(joined)
            await finish(asyncio.create_task(dispose_completed()))
            raise
        errors=[value for value in results if isinstance(value,BaseException)]
        if errors:
            await finish(asyncio.create_task(dispose_completed()))
            raise errors[0]
        return results

    async def owned_async_parallel(self, actions, cancellation):
        """Join owned coroutine actions, including their native/thread cleanup.

        Actions retain their normal ownership rules. Errors join every peer;
        cancellation signals the shared request and joins cancelled actions
        before the caller may dispose input files or retire the owner.
        """
        actions=tuple(actions)
        if not actions or any(not callable(action) for action in actions):
            raise ValueError('parallel async work requires coroutine factories')
        async def run(action):
            # Invoke inside the owned task: a factory can fail synchronously,
            # and peers already started still have to be joined.
            return await action()
        tasks=[asyncio.create_task(run(action)) for action in actions]
        joined=asyncio.gather(*tasks,return_exceptions=True)
        try:results=await asyncio.shield(joined)
        except asyncio.CancelledError:
            cancellation.set()
            for task in tasks:task.cancel()
            while not joined.done():
                try:await asyncio.shield(joined)
                except asyncio.CancelledError:continue
            joined.result()
            raise
        errors=[value for value in results if isinstance(value,BaseException)]
        if errors:raise errors[0]
        return results

    def bootstrap(self, checkpoint):
        if self.native_owner is None:
            owner = self.owner_factory(self.base, checkpoint)
            try:
                owner.establish_identity(checkpoint)
                broker = WeightPrefetchBroker(owner.identity['owner_pid'], owner.identity['device_uuid'],
                                              allow_gpu_prepare=True)
                for adapter in self.adapters:
                    backend = adapter.weights(owner, self)
                    self.register_backend(broker,adapter,backend)
                    for phase, policy in adapter.windows.items():
                        broker.register_window(phase, policy)
            except BaseException:
                try:owner.retire()
                except BaseException as error:
                    print(json.dumps(dict(event='registration_retirement_failed',error=str(error))),flush=True)
                raise
            self.native_owner, self.broker = owner, broker
            owner.attach(self)
        self.native_owner.observe()
        return self.native_owner

    def register_backend(self,broker,adapter,backend):
        try:from .native_weights import NativeWeightBackend
        except ImportError:from native_weights import NativeWeightBackend
        if type(backend) is not NativeWeightBackend:
            raise TypeError('resident models must use NativeWeightBackend.standard; custom loading paths are not admitted')
        backend.validate_registration(adapter)
        broker.register(backend.family,backend)
        self.backend_families[id(adapter)]=backend.family

    def prepare_dispatch(self,item,checkpoint,handoff):
        """Keep every numerical consumer behind its checked weight barrier."""
        self.prepare_transition(item,checkpoint,handoff)
        declare=getattr(item.get('adapter'),'declared_copy_gate',None)
        gate=declare(item,self) if declare is not None else None
        if gate is not None:
            if type(gate) is not DeclaredCopyGate:
                raise TypeError('deferred dispatch requires a framework DeclaredCopyGate')
            item['_declared_dispatch']=gate.bind(item['ticket'],self.native_owner)
            return
        receipt=self.prepare_weights(item,checkpoint,handoff)
        self._validate_dispatch_receipt(item,receipt)

    def _validate_dispatch_receipt(self,item,receipt):
        ticket=item['ticket']
        if (not isinstance(receipt,dict) or receipt.get('request_id')!=ticket.request_id or
                receipt.get('recipe_sha256')!=ticket.model.recipe_sha256 or
                any(receipt.get(k) is not True for k in ('copy_complete','payload_verified','gpu_prepare_complete')) or
                receipt.get('framework_protocol',{}).get('schema')!='resident-native-v1' or
                receipt['framework_protocol'].get('family')!=ticket.model.family):
            raise RuntimeError('inference requires the framework-prepared receipt for this exact ticket and recipe')

    async def generate_admitted(self,item,checkpoint,cancellation,handoff):
        gate=item.get('_declared_dispatch')
        async def generate():
            return await item['adapter'].generate(item,self,checkpoint,cancellation,handoff)
        if gate is None:return await generate()
        async def copy():
            try:
                receipt=await self.owned(lambda:self.prepare_weights(item,checkpoint,handoff),cancellation)
                self._validate_dispatch_receipt(item,receipt)
                gate.complete(receipt)
            except BaseException as error:
                gate.admission.fail(error);raise
        try:
            _,result=await self.owned_async_parallel((copy,generate),cancellation)
            return result
        finally:
            gate.admission.fail(RuntimeError('declared dispatch finished before copy admission'))

    def _successor(self):
        return next((item for item in self.tickets.values() if item is not self.active and item['alive']), None)

    def _bind(self, item):
        if item.get('bound'):return
        bind = getattr(item['adapter'], 'bind', None)
        if bind is not None:
            admitted = item['admitted']
            admitted.model = bind(admitted, self.native_owner)
            item['ticket'] = Ticket(item['ticket'].request_id, admitted.model)
        if item['ticket'].model.family!=self.backend_families[id(item['adapter'])]:
            raise ValueError('model admission does not belong to its registered backend')
        item['bound'] = True

    def window(self, phase, *, geometry=None):
        with self.ticket_lock:
            if self.active is None:return False
            if phase not in self.active['adapter'].windows:
                raise ValueError('model attempted an undeclared source window')
            self.window_phase, self.window_geometry = phase, geometry
        return self.poll()

    def decode_phase_gate(self, directory, phase, *, geometry, native_stage='decode', native_retirement=False):
        """Submit lookahead before a calibrated producer allocates its decoder.

        The native producer acknowledges final DiT use through this barrier.
        It resumes as soon as submission returns, alongside unfinished DMA.
        """
        try:
            from .phase_gate import DecodePhaseGate
        except ImportError:
            from phase_gate import DecodePhaseGate
        with self.ticket_lock:
            item = self.active
            if item is None or self.broker is None:
                return None
            if phase not in item['adapter'].windows:
                raise ValueError('producer did not declare this decode phase')
            policy = self.broker.windows[phase]
            if policy.source_gpu_complete:
                raise ValueError('a before-decode gate cannot declare completed GPU work')
            if policy.geometries is not None and geometry not in policy.geometries:
                return None
        def submit():
            with self.ticket_lock:
                if self.active is not item or not item['alive'] or self.closed.is_set() or self.broken:
                    raise RuntimeError('decode phase belongs to a retired source request')
                begun = time.monotonic_ns()
                if policy.idle_retirement:
                    self.retire_idle_final_weights(item,policy,geometry)
                started = self.window(phase, geometry=geometry)
                self.broker.events.append(dict(event='decode_phase_submitted',
                    source_request_id=item['ticket'].request_id, phase=phase, geometry=geometry,
                    submission_started_monotonic_ns=begun, submission_completed_monotonic_ns=time.monotonic_ns(),
                    copy_started=started, copy_pending=self.broker.pending is not None,
                    waited_for_dma_completion=False))
        def retirement():
            with self.ticket_lock:
                if self.active is not item or not item['alive'] or self.closed.is_set() or self.broken:
                    raise RuntimeError('weight retirement belongs to a retired source request')
                successor = self._successor()
                return bool(successor is not None and
                    successor['ticket'].model.family != item['ticket'].model.family and
                    policy.allows(successor['ticket'].model.family, geometry))
        def on_retired(released, receipt):
            with self.ticket_lock:
                if self.active is not item or not item['alive']:
                    raise RuntimeError('native retirement completed for a retired request')
                if released:item['decode_weights_released'] = True
                if released:self.broker.backends[item['ticket'].model.family]._resident_components.clear()
                if released and policy.reclaimed_weights_only:
                    devices = receipt.get('devices')
                    expected = {row['device_uuid'] for row in self.native_owner.identity['devices']}
                    if (type(devices) is not list or len(devices) != len(expected) or
                            {row.get('device_uuid') for row in devices} != expected or
                            any(type(row.get('free_before_bytes')) is not int or row['free_before_bytes'] < 0
                                for row in devices)):
                        raise RuntimeError('terminal weight retirement has another physical memory envelope')
                    item['decode_free_reserve'] = {row['device_uuid']: row['free_before_bytes'] for row in devices}
                self.broker.events.append(dict(event='native_terminal_weights',
                    source_request_id=item['ticket'].request_id, phase=phase, geometry=geometry,
                    released=released, receipt=receipt))
        return DecodePhaseGate(directory, native_stage, geometry, submit,
            **dict(retirement=retirement, on_retired=on_retired) if native_retirement else {})

    def retire_idle_final_weights(self,item,policy,geometry):
        """A final-use pipeline barrier may reclaim its already-idle native actor.

        The actor owns every eviction RPC. Preserve the decoder's physical free
        envelope; never treat the pipeline barrier as permission to evict a busy
        actor or to start another request forward.
        """
        if not policy.idle_retirement or not policy.reclaimed_weights_only:
            raise RuntimeError('idle final-use retirement lacks its memory contract')
        if self.active is not item or not item['alive'] or self.closed.is_set() or self.broken:
            raise RuntimeError('idle final-use barrier belongs to a retired request')
        successor=self._successor()
        if (successor is None or successor['ticket'].model.family==item['ticket'].model.family or
                not policy.allows(successor['ticket'].model.family,geometry)):
            return
        if item.get('decode_weights_released'):return
        if self.broker.pending is not None:
            raise RuntimeError('idle final-use retirement cannot overlap a submitted copy')
        self._bind(successor)
        expected={row['device_uuid'] for row in self.native_owner.identity['devices']}
        def physical():
            info=self.native_owner.establish_identity()
            rows=info.get('devices',[info])
            if (not info.get('transaction_idle') or len(rows)!=len(expected) or
                    {r.get('device_uuid') for r in rows}!=expected or
                    any(r.get('transaction_idle',info['transaction_idle']) is not True or
                        type(r.get('device_free_bytes')) is not int or r['device_free_bytes']<0 for r in rows)):
                raise RuntimeError('final-use retirement requires the same idle physical native ranks')
            return {r['device_uuid']:r['device_free_bytes'] for r in rows}
        before=physical()
        source=self.broker.backends[item['ticket'].model.family]
        target=self.broker.backends[successor['ticket'].model.family]
        primary=self.discard_backend(source,preserve_auxiliary=True)
        after=physical()
        auxiliary=None
        required=target.plan(successor['ticket']).device_bytes
        capacity=lambda:min(max(0,after[u]-before[u]-(256<<20)) for u in expected)
        if required>capacity():
            auxiliary=self.release_auxiliary(source)
            after=physical()
        receipt=dict(schema='framework-idle-final-use-retirement-v1',retired=True,
            primary=primary,auxiliary=auxiliary,devices=[dict(device_uuid=u,
                free_before_bytes=before[u],free_after_bytes=after[u]) for u in sorted(expected)],
            completed_monotonic_ns=time.monotonic_ns(),required_copy_bytes=required,
            reclaimed_copy_capacity_bytes=capacity(),decoder_free_envelope_preserved=True)
        item['decode_weights_released']=True
        item['decode_free_reserve']=before
        item['idle_final_use_retirement']=receipt
        self.broker.events.append(dict(event='idle_final_use_weights',
            source_request_id=item['ticket'].request_id,request_id=successor['ticket'].request_id,
            geometry=geometry,receipt=receipt))

    def poll(self):
        with self.ticket_lock:
            if self.broker is None:return False
            self.broker.poll()
            successor = self._successor()
            if self.active is None or self.window_phase is None or successor is None:
                return False
            if not self.broker.queue:
                self._bind(successor)
                # Lookahead is only for a model change. Consecutive requests
                # retain their weights; request input/layout preparation can
                # still run normally at the serial head.
                if successor['ticket'].model==self.active['ticket'].model:
                    return False
                self.broker.enqueue(successor['ticket'])
            if self.broker.pending is not None:return False
            source=self.broker.backends[self.active['ticket'].model.family]
            policy=self.broker.windows[self.window_phase]
            if policy.idle_retirement:
                receipt=self.active.get('idle_final_use_retirement')
                if receipt is None:return False
                reserves={r['device_uuid']:r['free_before_bytes'] for r in receipt.get('devices',[])}
                if (receipt.get('schema')!='framework-idle-final-use-retirement-v1' or
                        receipt.get('retired') is not True or not self.active.get('decode_weights_released') or
                        not reserves or reserves!=self.active.get('decode_free_reserve')):
                    raise RuntimeError('idle final-use window lacks a checked retirement envelope')
            info = self.native_owner.establish_identity()
            # Only the source model can declare that every future use of these
            # weights has ended. An uncalibrated window never releases them.
            if (info['transaction_idle'] and successor['ticket'].model.family!=source.family and
                    getattr(source,'reclaim_during_decode',False) and not self.active.get('decode_weights_released') and
                    policy.allows(successor['ticket'].model.family,self.window_geometry)):
                self.discard_backend(source,preserve_auxiliary=True)
                self.active['decode_weights_released']=True
                info = self.native_owner.establish_identity()
            target=self.broker.backends[successor['ticket'].model.family]
            capacity=max(0,info['device_free_bytes']+max(0,info['primary_reclaimable_bytes']-(256<<20))-
                         policy.reserve_bytes)
            required=target.plan(successor['ticket']).device_bytes
            if (required>capacity and
                    self.active.get('copy_capacity_wait_recorded')!=successor['ticket'].request_id):
                # A speculative decoder reserve is not execution pressure.
                # Wait for decoder temporaries to drain instead of evicting
                # reusable auxiliaries just to start this copy earlier. The
                # serial head checks real execution capacity and releases
                # foreign auxiliaries there only when they are required.
                self.broker.events.append(dict(event='waiting_for_copy_capacity',
                    request_id=successor['ticket'].request_id,
                    source_request_id=self.active['ticket'].request_id,
                    capacity_bytes=capacity,reserve_bytes=policy.reserve_bytes,
                    required_bytes=required))
                self.active['copy_capacity_wait_recorded']=successor['ticket'].request_id
            free = info['device_free_bytes']
            reclaimable = max(0, info['primary_reclaimable_bytes'] - (256 << 20))
            decoder_reserve = 0
            if policy.reclaimed_weights_only:
                # Replace only released weights. Keep at least as much free
                # memory on each GPU as the original resident decoder had.
                reserves = self.active.get('decode_free_reserve', {})
                devices = info.get('devices', [info])
                capacity = min((max(0, row['device_free_bytes'] - reserves.get(row['device_uuid'], row['device_free_bytes'])
                                     - (256 << 20)) for row in devices), default=0)
                reclaimable = 0
                decoder_reserve = max(0, free - capacity)
            used = sum(backend.host_pinned_bytes for backend in self.broker.backends.values())
            window = DecodeWindow(self.active['ticket'].request_id, info['owner_pid'], info['device_uuid'],
                info['transaction_idle'], free, reclaimable, decoder_reserve, max(0, self.host_budget_bytes - used),
                capacity_limit_bytes=self.preload_capacity_limit_bytes)
            return self.broker.open_window(self.window_phase, window, geometry=self.window_geometry)

    def handoff(self, item, checkpoint):
        if self.broker is None:return {'status': 'foreground'}
        while self.broker.pending is not None and self.broker.pending.abandoned:
            checkpoint();self.broker.poll();time.sleep(.005)
        if not self.broker.is_next(item['ticket']):return {'status': 'foreground'}
        while True:
            checkpoint();value = self.broker.adopt(item['ticket'])
            if value.status != 'copy_pending':return {'status': value.status, 'receipt': value.payload}
            time.sleep(.005)

    def prepare_weights(self, item, checkpoint, handoff):
        """Foreground uses the same checked copy transport as speculative work."""
        if item.get('_framework_weights_prepared') is handoff:return handoff['receipt']
        if handoff['status']=='auxiliary_ready' and 'auxiliary_prefetch' not in handoff:
            handoff['auxiliary_prefetch']=handoff['receipt']
        if handoff['status'] == 'ready':
            self.prepare_transition(item,checkpoint,handoff)
            item['_framework_weights_prepared']=handoff
            return handoff.get('receipt')
        ticket = item['ticket'];backend = self.broker.backends[ticket.model.family]
        prepare=getattr(backend,'prepare_foreground',None)
        if prepare is not None:
            with self.preparation_span(handoff,'foreground_prepare_hook'):
                prepare(ticket,checkpoint)
        # Unified request-selected serving uses the standard copy's own cold
        # loader. Other callers retain their existing explicit host warm gate.
        if self.warm_on_foreground:
            with self.preparation_span(handoff,'foreground_host_readiness') as span:
                warm=backend.preload(ticket,checkpoint,
                    force_revalidate=self.force_foreground_host_revalidation)
                handoff['host_preload']=warm.get('framework_host_readiness')
                if handoff['host_preload'] is not None:
                    span['host_readiness']=handoff['host_preload']
        self.prepare_transition(item,checkpoint,handoff)
        with self.preparation_span(handoff,'foreground_identity_and_budget'):
            info = self.native_owner.establish_identity(checkpoint)
            budget=self.device_memory_budget(backend,info,include_primary=True)
        resident=budget['resident_payload_bytes'];capacity=budget['capacity_bytes']
        free=min((row['free_device_bytes'] for row in budget['devices']),default=info['device_free_bytes'])
        reclaimable=capacity-free
        window = DecodeWindow('foreground/' + ticket.request_id, info['owner_pid'], info['device_uuid'],
            info['transaction_idle'], free, reclaimable, 0,
            self.host_budget_bytes)
        plan = backend.plan(ticket)
        if plan.device_bytes > capacity or not info['transaction_idle']:
            raise RuntimeError('native weights exceed idle device capacity: '+str(dict(
                required_bytes=plan.device_bytes,free_bytes=info['device_free_bytes'],
                reclaimable_bytes=info['primary_reclaimable_bytes'],resident_device_bytes=resident,
                copy_capacity_bytes=capacity,devices=budget['devices'],transaction_idle=info['transaction_idle'])))
        with self.preparation_span(handoff,'foreground_copy_submit'):
            handle = backend.start(ticket, plan, window)
        try:
            with self.preparation_span(handoff,'foreground_copy_join'):
                while handle.poll() is None:checkpoint();time.sleep(.005)
                receipt=handle.adopt(ticket.request_id, handle.ready)
            handoff['receipt']=receipt
            item['_framework_weights_prepared']=handoff
            return receipt
        except BaseException:
            handle.abandon()
            # Reap submitted DMA before retirement and request-file cleanup.
            while handle.poll() is None:time.sleep(.005)
            raise

    @staticmethod
    def device_memory_budget(backend,info,*,include_primary=False):
        standard=getattr(backend,'device_memory_budget',None)
        if standard is not None:return standard(info,include_primary=include_primary)
        resident=getattr(backend,'resident_device_bytes',lambda:0)()
        reclaimable=max(resident,info['primary_reclaimable_bytes'] if include_primary else 0)
        return dict(capacity_bytes=info['device_free_bytes']+reclaimable,
                    resident_payload_bytes=resident,devices=[])

    def check_auxiliary_host_capacity(self,backend):
        used=sum(getattr(other,'host_pinned_bytes',0) for other in self.broker.backends.values())
        reserve=getattr(backend,'retained_host_allocation_bytes',0)
        if used+reserve>self.host_budget_bytes:
            raise RuntimeError('immutable auxiliary snapshot exceeds the resident host budget')

    def discard_backend(self, backend, *, preserve_auxiliary=False):
        discard=getattr(backend,'discard_primary' if preserve_auxiliary else 'discard',None)
        if discard is None:discard=getattr(backend,'discard',None)
        if discard is None:return
        self.check_auxiliary_host_capacity(backend)
        return discard()

    def release_auxiliary(self,backend):
        release=getattr(backend,'release_auxiliary',None)
        if release is not None:
            self.check_auxiliary_host_capacity(backend)
            return release()

    def prepare_transition(self, item, checkpoint, handoff):
        """Release foreign weights and restore declared auxiliaries at the head.

        A ready speculative copy still needs this step: the previous model's
        decoder may have held weights until its output finished.
        """
        if handoff.get('transition_prepared'):return
        ticket=item['ticket'];backend=self.broker.backends[ticket.model.family]
        with self.preparation_span(handoff,'transition_identity'):
            prove_fresh=getattr(self.native_owner,'fresh_transition',None)
            fresh=prove_fresh(ticket.model.family,self.broker.backends,checkpoint) if prove_fresh is not None else None
            info = fresh if fresh is not None else self.native_owner.establish_identity(checkpoint)
        if not info['transaction_idle']:
            raise RuntimeError('foreground weight preparation requires an idle native transaction')
        if self.broker.pending is not None:
            raise RuntimeError('foreground weight preparation cannot evict a submitted copy')
        released={};released_auxiliaries={}
        for family, other in self.broker.backends.items():
            if family != ticket.model.family and fresh is None:
                with self.preparation_span(handoff,'transition_discard_primary/'+family):
                    checkpoint();released[family]=self.discard_backend(other,preserve_auxiliary=True)
        if fresh is not None:handoff['fresh_transition']=fresh
        with self.preparation_span(handoff,'transition_idle_workspaces'):
            self.prepare_idle_workspaces(item,checkpoint,handoff)
        if getattr(backend,'requires_idle_auxiliary_release',False):
            # Auxiliary reclamation need not depend on a static execution
            # reservation. The target may rely on native allocation pressure.
            for family,other in self.broker.backends.items():
                if family!=ticket.model.family:
                    with self.preparation_span(handoff,'transition_release_auxiliary/'+family):
                        checkpoint();released_auxiliaries[family]=self.release_auxiliary(other)
        execution=getattr(backend,'execution_reservation',None)
        required=execution(ticket) if execution is not None else 0
        if required:
            info=self.native_owner.establish_identity(checkpoint)
            budget=self.device_memory_budget(backend,info)
            for family,other in self.broker.backends.items():
                if budget['capacity_bytes']>=required:break
                if family!=ticket.model.family and family not in released_auxiliaries:
                    with self.preparation_span(handoff,'transition_release_auxiliary/'+family):
                        checkpoint();released_auxiliaries[family]=self.release_auxiliary(other)
                    info=self.native_owner.establish_identity(checkpoint)
                    budget=self.device_memory_budget(backend,info)
            if budget['capacity_bytes']<required:
                raise RuntimeError('model execution exceeds the idle physical device budget: '+str(dict(
                    required_bytes=required,**budget)))
            handoff['execution_memory']={**budget,'required_bytes':required,'free_device_bytes':info['device_free_bytes']}
        activate=getattr(backend,'activate',None)
        with self.preparation_span(handoff,'transition_activate'):
            activation=activate(checkpoint) if activate is not None else None
        handoff.update(released_weights=released,released_auxiliaries=released_auxiliaries,
                       weight_activation=activation,
                       transition_prepared=True)

    def prepare_idle_workspaces(self, item, checkpoint, handoff):
        """Drain prior scratch at the serial head, including a prefetched hit.

        A completed weight copy does not imply that the previous decoder's
        cached intermediates were released. Models declare this requirement;
        the owner supplies the workspace protocol and keeps weights/plans.
        """
        backend=self.broker.backends[item['ticket'].model.family]
        if getattr(backend,'requires_idle_workspace_release',False) and not handoff.get('idle_workspaces_released'):
            receipt=handoff.get('receipt') or {}
            if receipt.get('framework_idle_workspaces_released') is True:
                handoff['idle_workspace_release']=receipt['framework_idle_workspace_release']
            else:
                handoff['idle_workspace_release']=self.native_owner.release_idle_workspaces(checkpoint)
            handoff['idle_workspaces_released']=True

    async def generate_async(self, request):
        received_ns=time.monotonic_ns()
        if not isinstance(request, dict):raise ValueError('request must be an object')
        request = dict(request)
        workflow = request.get('workflow', 'fl2va')
        adapter = self.routes.get(workflow) if isinstance(workflow, str) else None
        if adapter is None:raise ValueError('workflow is not admitted by this unified release')
        cancellation = threading.Event();deadline = time.monotonic() + self.request_timeout
        def checkpoint():
            if cancellation.is_set() or self.closed.is_set():raise RuntimeError('resident request cancelled')
            if time.monotonic() >= deadline:raise TimeoutError('resident request deadline expired')
        call = asyncio.current_task();self.calls.add(call)
        admitted = None;item = None
        try:
            admitted = await self.owned(lambda: adapter.admit(request, checkpoint), cancellation,dispose=adapter.dispose)
            item = dict(ticket=Ticket(uuid.uuid4().hex, admitted.model), admitted=admitted,
                        adapter=adapter, alive=True, received=received_ns,admitted_ns=time.monotonic_ns())
            with self.ticket_lock:self.tickets[item['ticket'].request_id] = item
            async with self.lock:
                dequeued_ns=time.monotonic_ns()
                checkpoint()
                if self.broken:raise RuntimeError('resident owner unavailable')
                with self.ticket_lock:
                    self.active = item;self.window_phase = self.window_geometry = None
                self._cancellation = cancellation;self.loading = True
                await self._start_preparation(adapter,admitted,checkpoint,cancellation)
                await self.owned(lambda: self.bootstrap(checkpoint), cancellation)
                with self.ticket_lock:self._bind(item)
                handoff = await self.owned(lambda: self.handoff(item, checkpoint), cancellation)
                await self.owned(lambda:self.prepare_dispatch(item,checkpoint,handoff),cancellation)
                self.loading = False
                begun = time.monotonic_ns()
                result = await self.generate_admitted(item,checkpoint,cancellation,handoff)
                if adapter.counts_separately:self.extra_requests += 1
                result.setdefault('timing', {}).update(weight_handoff=handoff,
                    active_request_seconds=(time.monotonic_ns()-begun)/1e9,
                    queue_seconds=max(0,(begun-item['received'])/1e9))
                result['native_owner'] = dict(self.native_owner.identity)
                marker=result['timing'].get('first_dit_start')
                if marker is not None:
                    try:from .execution_timing import first_dit_timing
                    except ImportError:from execution_timing import first_dit_timing
                    result['timing'].update(first_dit_timing(marker,self.native_owner.identity,
                        received_ns,item['admitted_ns'],dequeued_ns,time.monotonic_ns()))
                return result
        except RequestRejected as error:
            # Keep the RPC boundary compatible with clients that do not ship
            # this internal framework exception class.
            raise ValueError(str(error)) from None
        except BaseException:
            if item is not None and self.active is item:
                self.failed = True;cancellation.set()
                if self.native_owner is not None:
                    try:self.native_owner.retire()
                    except BaseException as cleanup_error:
                        # Preserve the request's actual failure for callers and
                        # release evidence; owner retirement remains mandatory.
                        print(json.dumps(dict(event='resident_retirement_failed',error=str(cleanup_error))),flush=True)
            raise
        finally:
            if item is not None:
                with self.ticket_lock:
                    item['alive'] = False
                    if self.broker is not None:self.broker.cancel(item['ticket'].request_id)
                    self.tickets.pop(item['ticket'].request_id, None)
                    if self.active is item:
                        self.active = None;self.window_phase = self.window_geometry = None
                        self._cancellation = None;self.loading = False
            try:
                if admitted is not None:await self.owned(lambda:adapter.dispose(admitted),cancellation)
            finally:self.calls.discard(call)

    def health(self):
        value = json.loads(self.base.health())
        capabilities = dict(value.get('capabilities', {}))
        capabilities.update({w: True for w in self.routes})
        old = value.get('workflows', {})
        if isinstance(old, list):value['workflows'] = list(dict.fromkeys([*old, *self.routes]))
        else:value['workflows'] = {**old, **{w: {'registered': True} for w in self.routes if w not in old}}
        value.update(capabilities=capabilities, requests_served=value['requests_served']+self.extra_requests,
                     host_preload_ready=self.host_preload_ready,
                     startup_preparation=self.startup_preparation,
                     active_workflow=self.active['admitted'].request['workflow'] if self.active else None)
        if self.native_owner is not None:
            value['native_owner'] = dict(self.native_owner.identity)
            value['weight_prefetch'] = {'failed': self.broker.failed, 'events': list(self.broker.events)}
            value['weight_integrations']={family:dict(schema='resident-native-v1',loader=backend.protocol.loader,
                world_size=backend.protocol.world_size,component_order=list(backend.protocol.components),
                phases=sorted(backend.protocol.phases)) for family,backend in self.broker.backends.items()
                if getattr(backend,'protocol',None) is not None}
        if self.broken:value['status'] = 'failed'
        elif self.extra_requests and value['status'] in ('loading','awaiting_recipe'):value['status']='ok'
        if 'model_loading' in value and self.extra_requests:
            value['model_loading'].update(state='loading' if self.loading else 'ready',models_ready=not self.broken)
        return json.dumps(value)

    async def aclose(self):
        self.closed.set()
        if self._cancellation is not None:self._cancellation.set()
        calls = [c for c in self.calls if c is not asyncio.current_task()]
        for call in calls:call.cancel()
        await asyncio.gather(*calls, return_exceptions=True)
        await self.base.aclose()
