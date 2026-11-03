"""Real sockets prove strict declared dependencies and borrowed-owner handoff."""
import json
from pathlib import Path
import socket
import tempfile
import time

import pytest
from core_infra.runtime.commands import CommandSchedule, PreloadCommand


def receipt(details):
    return json.dumps({'ok': True, 'details': details}).encode() + b'\n'


def test_weight_ready_does_not_release_output_until_real_input_use():
    with tempfile.TemporaryDirectory(prefix='declared-load-', dir='/tmp') as directory:
        root = Path(directory)
        connection, peer = socket.socketpair()
        consumed = {'value': False}
        plan = (
            PreloadCommand('input', 'vae', ('preload-encoder', '/weights'), lambda d, _: d.get('loaded') is True),
            PreloadCommand('output', 'vae', ('preload-output', '/weights'), lambda d, _: d.get('loaded') is True,
                           ('input', 'input-used')),
        )
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(root/'vae.sock')); listener.listen(2); listener.settimeout(.01)
            schedule = CommandSchedule(root, plan, conditions={'input-used': lambda: consumed['value']})
            try:
                schedule.adopt('input', connection, receipt({'loaded': True}), started=time.monotonic())
                schedule.start(); schedule.poll()
                assert schedule.states == {'input': 'ready', 'output': 'waiting'}
                with pytest.raises(TimeoutError): listener.accept()
                assert peer.recv(1) == b''  # Completed adoption closed without resending a command.
                consumed['value'] = True; schedule.poll()
                with listener.accept()[0] as output:
                    assert json.loads(output.makefile().readline())['args'] == ['preload-output', '/weights']
                    output.sendall(receipt({'loaded': True}))
                    schedule.poll()
                assert schedule.ready and not schedule.pending
                with pytest.raises(RuntimeError, match='only start once'): schedule.start()
            finally:
                schedule.close(); peer.close()


@pytest.mark.parametrize('result', [None, False, 1, {}, {'prepared': True}])
def test_only_affirmative_validation_can_publish_readiness(tmp_path, result):
    connection, peer = socket.socketpair()
    schedule = CommandSchedule(tmp_path, (
        PreloadCommand('dit', 'dit', ('preload', '/weights'), lambda *_: result),))
    try:
        schedule.adopt('dit', connection, receipt({'loaded': True}), started=time.monotonic())
        schedule.start()
        with pytest.raises(RuntimeError, match='receipt rejected'): schedule.poll()
        assert schedule.closed and not schedule.ready and not schedule.receipts
        assert not schedule.pending and connection.fileno() == -1
    finally:
        schedule.close(); peer.close()


def test_missing_dependency_or_serial_owner_order_fails_before_transport(tmp_path):
    validated = lambda *_: True
    with pytest.raises(ValueError, match='unordered'):
        CommandSchedule(tmp_path, (PreloadCommand('dit', 'dit', ('preload',), validated, ('missing-preparation',)),))
    with pytest.raises(ValueError, match='previous preload'):
        CommandSchedule(tmp_path, (PreloadCommand('encoder', 'vae', ('preload-encoder',), validated),
                                  PreloadCommand('decoder', 'vae', ('preload-output',), validated)))


def test_condition_cannot_silently_return_none_before_execution(tmp_path):
    schedule = CommandSchedule(tmp_path, (
        PreloadCommand('dit', 'dit', ('preload',), lambda *_: True, ('prepared',)),),
        conditions={'prepared': lambda: None})
    with pytest.raises(RuntimeError, match='return bool'): schedule.start()
    assert schedule.closed and not schedule.pending and not schedule.ready


def test_changed_cached_receipt_is_rejected_without_execution(tmp_path):
    schedule = CommandSchedule(tmp_path, (
        PreloadCommand('dit', 'dit', ('preload',), lambda d, _: d.get('identity') == 'admitted'),),
        cached=lambda _: {'identity': 'other-model'})
    with pytest.raises(RuntimeError, match='receipt rejected'): schedule.start()
    assert schedule.closed and not schedule.receipts and not schedule.ready
