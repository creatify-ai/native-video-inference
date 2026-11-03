"""Attached command ownership must keep order, partial replies and teardown."""
import json
from pathlib import Path
import socket
import sys
import tempfile
import time

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core_infra.runtime.commands import PendingCommands


def test_command_is_sent_before_compute_without_a_waiting_thread():
    with tempfile.TemporaryDirectory(prefix='native-commands-',dir='/tmp') as root:
        root=Path(root)
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as listener:
            listener.bind(str(root/'native.sock'));listener.listen(2)
            group=PendingCommands(root)
            try:
                group.send('encoder',root/'native.sock',['preload','original-weights'],
                           context={'component':'encoder'})
                with listener.accept()[0] as peer:
                    assert json.loads(peer.makefile().readline())['args']==['preload','original-weights']
                    peer.sendall(b'{"ok":true,"details":')
                    completed=[]
                    group.poll(lambda *args:completed.append(args))
                    assert not completed and len(group.pending)==1
                    peer.sendall(b'{"loaded":true}}\n')
                    group.poll(lambda name,task:completed.append((name,json.loads(task['buffer']))))
                    assert completed==[('encoder',{'ok':True,'details':{'loaded':True}})]
                    assert not group.pending
                    assert json.loads((root/'preload-encoder.log').read_bytes())['ok']
            finally:group.close()


@pytest.mark.parametrize('complete',[False,True])
def test_adoption_keeps_bytes_without_resending_an_existing_command(tmp_path,complete):
    connection,peer=socket.socketpair()
    group=PendingCommands(tmp_path)
    raw=b'{"ok":true,"details":{"shared":true}}\n'
    prefix=raw if complete else raw[:9]
    try:
        group.adopt('shared',connection,prefix,started=time.monotonic()-.5,
                    context={'component':'conditioner'})
        peer.setblocking(False)
        with pytest.raises(BlockingIOError):peer.recv(1)  # No second preload.
        if not complete:peer.sendall(raw[len(prefix):])
        else:peer.close()  # A complete adopted frame does not require another recv.
        rows=[]
        group.poll(lambda name,task:rows.append((name,task['component'],bytes(task['buffer']))))
        assert rows==[('shared','conditioner',raw)] and connection.fileno()==-1
        assert not group.pending
    finally:group.close();peer.close()


@pytest.mark.parametrize('failure',['envelope','validation','timeout','oversize'])
def test_failure_closes_every_pending_connection_before_propagating(tmp_path,failure):
    group=PendingCommands(tmp_path,timeout=1,max_reply_bytes=64)
    pairs=[socket.socketpair() for _ in range(2)]
    try:
        for index,(connection,peer) in enumerate(pairs):
            group.adopt(str(index),connection,b'')
        if failure=='timeout':group.pending['0']['started']=time.monotonic()-2
        else:
            raw=b'{"details":{}}\n' if failure=='envelope' else b'{"ok":true}\n'
            if failure=='oversize':raw=b'x'*65
            pairs[0][1].sendall(raw)
        def validate(*args):raise ValueError('invalid model readiness')
        with pytest.raises((RuntimeError,ValueError,TimeoutError)):
            group.poll(validate)
        assert group.closed and not group.pending
        assert all(connection.fileno()==-1 for connection,_ in pairs)
        with pytest.raises(RuntimeError,match='closed'):
            group.send('next','unused',[])
        group.close()
    finally:
        group.close()
        for _,peer in pairs:peer.close()


def test_pending_close_is_idempotent_and_does_not_claim_native_cancellation(tmp_path):
    connection,peer=socket.socketpair()
    group=PendingCommands(tmp_path)
    try:
        group.adopt('weights',connection,b'partial')
        group.close();group.close()
        assert not group.pending and connection.fileno()==-1
        assert peer.recv(1)==b''
    finally:peer.close()
