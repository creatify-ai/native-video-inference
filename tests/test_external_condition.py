from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from core_infra.runtime import ConditionOwner


def test_external_readiness_waits_for_the_fact_and_closes():
    ready=threading.Event()
    port=ConditionOwner(ready.is_set,timeout=2)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future=executor.submit(port.invoke,('wait',))
        assert not future.done()
        ready.set()
        assert future.result(timeout=1)=={'ready':True}
    port.terminate()
    assert port.reap()=={'external_condition':True,'closed':True}


def test_external_readiness_cancellation_and_invalid_receipts_fail():
    entered=threading.Event()
    def predicate():entered.set();return False
    port=ConditionOwner(predicate,timeout=2)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future=executor.submit(port.invoke,('wait',))
        assert entered.wait(1)
        port.terminate()
        with pytest.raises(RuntimeError,match='cancelled'):future.result(timeout=1)
    with pytest.raises(RuntimeError,match='return bool'):
        ConditionOwner(lambda:1).invoke(('wait',))
    with pytest.raises(ValueError,match='declared wait'):
        ConditionOwner(lambda:True).invoke(('compute',))
