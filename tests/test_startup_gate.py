"""Real socket checks of the native adapter to declared startup readiness."""
import json
from pathlib import Path
import socket
import tempfile
import threading
import time

import pytest

from core_infra.runtime.startup_gate import StartupGate, SCHEMA


def peer(gate, message):
    result = {}
    def run():
        try:
            with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as stream:
                stream.settimeout(2);stream.connect(str(gate.path))
                stream.sendall(json.dumps(dict(schema=SCHEMA,token=gate.token,**message)).encode()+b'\n')
                result['reply']=json.loads(stream.makefile('rb').readline())
        except BaseException as error:
            result['error']=error
    thread=threading.Thread(target=run);thread.start()
    return result,thread


def until(gate, predicate):
    deadline=time.monotonic()+2
    while not predicate():
        gate.poll()
        assert time.monotonic()<deadline
        time.sleep(.001)


def test_independent_consumers_are_not_blocked_by_a_waiting_dit():
    ready=threading.Event();received=[]
    with tempfile.TemporaryDirectory(prefix='start-gate-',dir='/tmp') as directory:
        gate=StartupGate(directory,lambda scope,inputs:received.append(inputs),
                         lambda scope,name:name=='encoder' or ready.is_set())
        threads=[]
        try:
            inputs,t=peer(gate,dict(kind='inputs',scope='/request',inputs={'request':'facts'}));threads.append(t)
            until(gate,lambda:'reply'in inputs)
            dit,t=peer(gate,dict(kind='before_use',scope='/request',component='dit'));threads.append(t)
            encoder,t=peer(gate,dict(kind='before_use',scope='/request',component='encoder'));threads.append(t)
            until(gate,lambda:'reply'in encoder)
            assert encoder['reply']['continued'] and 'reply'not in dit
            ready.set();until(gate,lambda:'reply'in dit)
            assert dit['reply']['continued'] and received==[{'request':'facts'}]
        finally:
            gate.close()
            for thread in threads:thread.join(2)
        assert not gate.path.exists()


@pytest.mark.parametrize('result',[RuntimeError('required layout failed'),1,None])
def test_failed_or_invalid_readiness_cannot_open_native_compute(result):
    def before(scope,name):
        if isinstance(result,BaseException):raise result
        return result
    with tempfile.TemporaryDirectory(prefix='start-gate-',dir='/tmp') as directory:
        gate=StartupGate(directory,lambda *args:None,before)
        reply,thread=peer(gate,dict(kind='before_use',scope='/request',component='dit'))
        try:
            with pytest.raises(RuntimeError):until(gate,lambda:'reply'in reply)
            thread.join(2)
            assert reply['reply']['continued'] is False
            assert not gate.consumers
        finally:
            gate.close();thread.join(2)


def test_wrong_request_pin_is_rejected_before_input_callback():
    called=[]
    with tempfile.TemporaryDirectory(prefix='start-gate-',dir='/tmp') as directory:
        gate=StartupGate(directory,lambda *args:called.append(args),lambda *args:True)
        result={}
        def client():
            with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as stream:
                stream.connect(str(gate.path))
                stream.sendall(json.dumps(dict(schema=SCHEMA,token='b'*32,kind='inputs',scope='/foreign',inputs={})).encode()+b'\n')
                result['reply']=json.loads(stream.makefile('rb').readline())
        thread=threading.Thread(target=client);thread.start()
        try:
            with pytest.raises(RuntimeError,match='admitted request'):until(gate,lambda:'reply'in result)
            thread.join(2)
            assert not result['reply']['continued'] and not called
        finally:
            gate.close();thread.join(2)
