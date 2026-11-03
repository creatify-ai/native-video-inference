"""Drain only an owner's already-killed process tree during slow driver teardown."""
import os
import time


def drain_killed_owner(owner, members, *, timeout=30, poll_interval=.02):
    """No signals or global waitpid: the owner already sent SIGKILL.

    CUDA unpin/context teardown can exceed the initial kill grace. Retain the
    process ledger until every known child and adopted descendant is gone.
    """
    if not 0 < timeout <= 180 or not 0 <= poll_interval <= 1:
        raise ValueError('invalid killed-owner drain bound')
    until = time.monotonic() + timeout
    direct = {child.pid for child in owner.children}
    while True:
        owner.observe()
        children = [child.pid for child in owner.children if child.poll() is None]
        descendants = set()
        for pid in owner.descendants - direct:
            if type(pid) is not int or pid <= 0:
                raise RuntimeError('invalid owned descendant identity')
            try:
                waited, _ = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                continue
            if waited != pid:
                descendants.add(pid)
        remaining_members = members(owner.groups, include_zombies=True)
        if not children and not descendants and not remaining_members:
            owner.children.clear()
            owner.groups.clear()
            owner.descendants.clear()
            return
        remaining = until - time.monotonic()
        if remaining <= 0:
            raise RuntimeError('native cleanup deadline exceeded: children=' + str(children) +
                               ' descendants=' + str(sorted(descendants)) +
                               ' group_members=' + str(sorted(remaining_members)))
        time.sleep(min(poll_interval, remaining))
