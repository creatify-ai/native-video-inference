"""A real Rust gate retains the mandatory barrier during framework dispatch."""
import asyncio
from dataclasses import replace
import threading
from types import SimpleNamespace

import pytest
from core_infra.runtime.components import Component
from core_infra.runtime.native_startup import DeclaredStartup
from core_infra.runtime.startup_spec import Preparation
from core_infra.runtime.resident_service import DeclaredCopyGate,ResidentModelService
from core_infra.runtime.weight_prefetch import ModelKey,Ticket


def graph(tmp_path,calls):
    class Owner:
        def invoke(self,args):calls.append(tuple(args));return {'ok':True}
        def terminate(self):pass
        def reap(self):return {'borrowed':True}
    def owner(component,environment,group):
        value=Owner()
        with group.guard:group.owners.append(value)
        return value
    # The model's validator would accept anything. The dispatch binding must
    # replace it with the framework's checked completion, not trust this stub.
    preparation=Preparation('copy-admission',('dit',),lambda _:True,lambda _:True,
        inputs=('request',),weights=('dit',),at_use=True)
    startup=DeclaredStartup(tmp_path,{},(Component('dit','/unused',('preload',),
        validate=lambda v:v['ok'],required_preparations=('copy-admission',)),),
        (preparation,),owner_factory=owner,native_policy=False).start()
    return startup


def test_arbitrary_model_object_cannot_opt_out_of_the_dispatch_barrier():
    with pytest.raises(TypeError,match='real Rust'):
        DeclaredCopyGate(SimpleNamespace(executor='rust'),'dit','copy-admission',lambda _:None)


@pytest.mark.parametrize('change',[
    dict(at_use=False),dict(inputs=()),dict(before=()),dict(prepare=lambda _:True)])
def test_gate_requires_a_mandatory_request_preparation(tmp_path,change):
    startup=graph(tmp_path,[])
    try:
        startup.definitions['copy-admission']=replace(startup.definitions['copy-admission'],**change)
        with pytest.raises(ValueError,match='mandatory request'):
            DeclaredCopyGate(startup,'dit','copy-admission',lambda _:None)
    finally:startup.close()


@pytest.mark.parametrize('failure',[None,'foreign-ticket','incomplete-copy','wrong-protocol','model-adoption'])
def test_owned_copy_overlaps_cpu_dispatch_but_never_unguards_compute(tmp_path,failure):
    calls=[];startup=graph(tmp_path,calls)
    async def exercise():
        cpu_started=threading.Event();copy_complete=threading.Event();installed=[]
        identity=dict(owner_pid=1234,owner_instance='native-1',device_uuid='GPU-1')
        owner=SimpleNamespace(identity=identity,verify_identity=lambda receipt:None)
        ticket=Ticket('request-1',ModelKey('h3','test','a'*64))
        service=ResidentModelService.__new__(ResidentModelService);service.native_owner=owner
        receipt=dict(identity,request_id=ticket.request_id,recipe_sha256=ticket.model.recipe_sha256,
            generation=1,payload_verified=True,copy_complete=True,gpu_prepare_complete=True,
            framework_protocol=dict(schema='resident-native-v1',family='h3'))
        if failure=='foreign-ticket':receipt['request_id']='another-request'
        if failure=='incomplete-copy':receipt['copy_complete']=False
        if failure=='wrong-protocol':receipt['framework_protocol']={}
        def copy(*args):
            assert cpu_started.wait(2),'CPU dispatch was serialized behind copy'
            assert ('compute',)not in calls
            copy_complete.set();return receipt
        async def generate(*args):
            startup.request({'request':'current request facts'})
            cpu_started.set()
            result=await asyncio.to_thread(startup.invoke,'dit',('compute',))
            assert copy_complete.is_set()
            return result
        def adopted(receipt):
            assert ('compute',)not in calls
            calls.append(('adopt',))
            if failure=='model-adoption':raise RuntimeError('model rejected its native handoff')
        adapter=SimpleNamespace(generate=generate,declared_copy_gate=lambda *args:
            DeclaredCopyGate(startup,'dit','copy-admission',installed.append,on_complete=adopted))
        item=dict(adapter=adapter,ticket=ticket);handoff={};cancel=threading.Event()
        service.prepare_transition=lambda *args:None;service.prepare_weights=copy
        service.prepare_dispatch(item,lambda:None,handoff)
        assert installed and not copy_complete.is_set()
        gate=item['_declared_dispatch']
        assert gate.admits_dispatch(startup,ticket,'dit')
        assert not gate.admits_dispatch(startup,Ticket('foreign',ticket.model),'dit')
        assert not gate.admits_dispatch(startup,ticket,'other')
        operation=service.generate_admitted(item,lambda:None,cancel,handoff)
        if failure:
            with pytest.raises(RuntimeError):await asyncio.wait_for(operation,5)
            assert ('compute',)not in calls
            assert not gate.admits_dispatch(startup,ticket,'dit')
        else:
            assert (await asyncio.wait_for(operation,5))['ok']
            assert calls.count(('compute',))==1
            assert calls.index(('adopt',))<calls.index(('compute',))
        assert copy_complete.is_set()
    try:asyncio.run(exercise())
    finally:startup.close()
