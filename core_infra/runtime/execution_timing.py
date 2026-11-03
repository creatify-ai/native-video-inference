"""Validate a native stream timestamp against the request's monotonic clock."""


def first_dit_timing(marker, owner, received_ns, admitted_ns, dequeued_ns, completed_ns):
    # A context-parallel request reports the first stream operation across its
    # ranks. Either verified rank can win; its PID and lifetime must still match
    # the established native owner topology exactly.
    peers=owner.get('devices',())
    identities=[owner,*peers] if isinstance(peers,(list,tuple)) else [owner]
    def identity(value):
        if (not isinstance(value,dict) or type(value.get('owner_pid')) is not int or value['owner_pid']<=0 or
                not isinstance(value.get('owner_instance'),str) or not value['owner_instance']):return None
        return value['owner_pid'],value['owner_instance']
    actual=identity(marker)
    if (not isinstance(marker, dict) or marker.get('clock') != 'CLOCK_MONOTONIC' or
            actual is None or actual not in {pair for value in identities if (pair:=identity(value)) is not None}):
        raise RuntimeError('first DiT marker belongs to another clock or native owner')
    first_ns = marker.get('monotonic_ns')
    times = (received_ns, admitted_ns, dequeued_ns, first_ns, completed_ns)
    if any(type(value) is not int or value <= 0 for value in times) or tuple(sorted(times)) != times:
        raise RuntimeError('first DiT marker is missing or outside this request')
    queue_ns = dequeued_ns - admitted_ns
    return dict(first_dit_start=marker, request_received_monotonic_ns=received_ns,
        queue_admitted_monotonic_ns=admitted_ns, request_dequeued_monotonic_ns=dequeued_ns,
        request_completed_monotonic_ns=completed_ns,
        queue_seconds=queue_ns / 1e9, admission_seconds=(admitted_ns - received_ns) / 1e9,
        head_to_first_dit_seconds=(first_ns - dequeued_ns) / 1e9,
        first_dit_seconds=(first_ns - received_ns - queue_ns) / 1e9,
        first_dit_scope='request entry to first native DiT stream operation; queue residence excluded')
