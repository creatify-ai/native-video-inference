"""Context-parallel DiT package: each rank's weight shards as slices of the prepared DiT's files.

With CP, each rank keeps one equal byte range of every layer's packed weights and gathers the rest
from its peers layer by layer (transformer/sharded_native.rs). A layer packs its tensors in name
order, each at a 256-byte boundary, padded to a multiple of 256 x world bytes. Each rank reads its
ranges straight from dit/'s tensor files, so no weight bytes are copied. Two files are written:
manifest.json, the h3-layer-weight-shards-v1 layout, and slices.json, the file ranges of each rank.
Every shard is rebuilt from its slices and checked against the layer it was cut from.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

from .common import CHUNK, write_json
from .light_guards import FILENAME as LIGHT_GUARDS, SEGMENT

ALIGNMENT = 256
WORLDS = (2, 4, 8)
LAYERS = (('transformer_blocks.', 50), ('token_refiner.refiner_blocks.', 2))


def aligned(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def layer(name):
    """The execution layer that owns a weight, as in weight_shards.rs group_for_key."""
    for prefix, count in LAYERS:
        if name.startswith(prefix):
            index, _, tail = name[len(prefix):].partition('.')
            if not (index.isdigit() and str(int(index)) == index and int(index) < count and tail):
                raise ValueError('weight layer index outside the H3 architecture: ' + name)
            return prefix + index
    return 'global'


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_into(stream, view):
    done = 0
    while done < len(view):
        count = stream.readinto(view[done:done + CHUNK])
        if not count:
            break
        done += count
    return done


def guard(data):
    return {'bytes': len(data), 'head_sha256': sha256(data[:SEGMENT]), 'tail_sha256': sha256(data[-SEGMENT:])}


def pack(name, tensors, root, world):
    """One layer: its manifest group, each rank's slices and their light guards."""
    entries, end = [], 0
    for key, spec in tensors:
        offset = aligned(end, ALIGNMENT)
        entries.append({'names': [f'weights/{key}'], 'offset': offset, 'tensor': spec})
        end = offset + spec['bytes']
    data = bytearray(aligned(end, ALIGNMENT * world))
    sizes = {}
    for entry in entries:
        spec, offset = entry['tensor'], entry['offset']
        path = root / spec['file']
        sizes[spec['file']] = path.stat().st_size
        with path.open('rb', buffering=0) as stream:
            stream.seek(spec.get('offset', 0))
            view = memoryview(data)[offset:offset + spec['bytes']]
            if read_into(stream, view) != spec['bytes'] or sha256(view) != spec['sha256']:
                raise ValueError('prepared DiT tensor differs from its manifest: ' + spec['file'])
    width = len(data) // world
    ranks, slices, guards = [], [], {}
    for rank in range(world):
        low, high = rank * width, (rank + 1) * width
        shard = memoryview(data)[low:high]
        pieces = []
        for entry in entries:
            spec, offset = entry['tensor'], entry['offset']
            start, stop = max(offset, low), min(offset + spec['bytes'], high)
            if start >= stop:
                continue
            piece = memoryview(data)[start:stop]
            digest = sha256(piece)
            pieces.append({'dest': start - low, 'file': spec['file'], 'offset': spec.get('offset', 0) + start - offset,
                           'bytes': stop - start, 'file_bytes': sizes[spec['file']], 'sha256': digest})
            guards[digest] = guard(piece)
        # The shard a rank uploads: its slices over zeros. It must be exactly this range of the layer.
        rebuilt = bytearray(width)
        for piece in pieces:
            rebuilt[piece['dest']:piece['dest'] + piece['bytes']] = data[low + piece['dest']:low + piece['dest'] + piece['bytes']]
        if rebuilt != shard:
            raise ValueError('rank slices do not rebuild their shard: ' + name)
        ranks.append({'file': f'{name}/rank-{rank:02d}.raw', 'bytes': width, 'sha256': sha256(shard)})
        slices.append(pieces)
    group = {'name': name, 'bytes': len(data), 'sha256': sha256(data), 'entries': entries, 'ranks': ranks}
    return group, slices, guards


def prepare_cp(prepared, world, workers=4):
    """Write <prepared>/dit-cp<world>/ and light-sha-cp<world>.json; returns their SHA-256."""
    if world not in WORLDS:
        raise ValueError('CP weight shards need two, four or eight ranks')
    prepared = Path(prepared)
    dit, out = prepared / 'dit', prepared / f'dit-cp{world}'
    raw = (dit / 'manifest.json').read_bytes()
    model = json.loads(raw)
    layers = {}
    for key, spec in sorted(model['weights'].items()):
        layers.setdefault(layer(key), []).append((key, spec))
    # A layer is at most about 1 GiB of host memory while it is checked.
    with ThreadPoolExecutor(workers) as pool:
        packed = list(pool.map(lambda item: pack(*item, dit, world), sorted(layers.items())))
    manifest = {'schema': 'h3-layer-weight-shards-v1', 'world_size': world, 'workflow': model['recipe']['workflow'],
                'base_weights_manifest_sha256': sha256(raw), 'groups': [group for group, _, _ in packed]}
    manifest_sha256 = write_json(out / 'manifest.json', manifest)
    ranks = [[{'group': index, **piece} for index, (_, slices, _) in enumerate(packed) for piece in slices[rank]]
             for rank in range(world)]
    slices_sha256 = write_json(out / 'slices.json', {'schema': 'h3-dit-shard-slices-v1',
        'shard_manifest_sha256': manifest_sha256, 'full_sha256_verified': True, 'ranks': ranks})
    # The CP processes read slices as well as whole files; their guards join the checked table.
    table = json.loads((prepared / LIGHT_GUARDS).read_bytes())
    if table.get('schema') != 'h3-light-sha-v1' or table.get('full_sha256_verified') is not True:
        raise ValueError('the prepared light SHA table is not a checked h3-light-sha-v1 table')
    for _, _, guards in packed:
        table['entries'].update(guards)
    light_sha256 = write_json(prepared / f'light-sha-cp{world}.json', table)
    return {'world': world, 'manifest_sha256': manifest_sha256, 'slices_sha256': slices_sha256,
            'light_sha256': light_sha256, 'groups': len(packed),
            'rank_bytes': sum(group['bytes'] // world for group, _, _ in packed)}
