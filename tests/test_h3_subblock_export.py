"""The SubBlock export check pins the object's GPU payload sections, not its host launcher code."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct

import pytest

MODULE = Path(__file__).resolve().parents[1] / 'models/minimax_h3_official/prepare_subblock_aot.py'
spec = importlib.util.spec_from_file_location('prepare_subblock_aot', MODULE)
subblock = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subblock)


def elf64(sections):
    """A minimal little-endian ELF64 relocatable object: [(name, sh_type, payload)]."""
    names = b'\0' + b''.join(name.encode() + b'\0' for name, _, _ in sections) + b'.shstrtab\0'
    body, offsets, cursor = b'', [], 64
    for _, _, payload in sections:
        offsets.append(cursor)
        body += payload
        cursor += len(payload)
    names_offset, cursor = cursor, cursor + len(names)
    header = bytearray(64)
    header[:6] = b'\x7fELF\x02\x01'
    struct.pack_into('<Q', header, 0x28, cursor)
    struct.pack_into('<HHH', header, 0x3A, 64, len(sections) + 2, len(sections) + 1)
    table = bytes(64)  # SHN_UNDEF
    position = 1
    for (name, kind, payload), offset in zip(sections, offsets):
        table += struct.pack('<IIQQQQIIQQ', position, kind, 0, 0, offset, len(payload), 0, 0, 1, 0)
        position += len(name) + 1
    table += struct.pack('<IIQQQQIIQQ', position, 3, 0, 0, names_offset, len(names), 0, 0, 1, 0)
    return bytes(header) + body + names + table


def write_export(root, gpu=b'kernel image', data=b'launch data', host=b'host code', **report):
    pin = json.loads(subblock.PIN.read_bytes())
    root.mkdir()
    (root / 'h3_subblock_v07.o').write_bytes(elf64([('.ltext', 1, host), ('.lrodata', 1, gpu), ('.ldata', 1, data)]))
    for name in pin['sha256']:
        (root / name).write_bytes(name.encode())
    value = {key: pin[key] for key in ('schema', 'flashinfer', 'dsl_version', 'arch', 'vendor_sources_sha256',
                                       'kernel_modified', 'variable_block_counts', 'has_block_sizes', 'kv_splits', 'clc')}
    value.update(sha256={name: hashlib.sha256(name.encode()).hexdigest() for name in pin['sha256']}, **report)
    (root / 'export.json').write_text(json.dumps(value))
    return pin


def test_elf_sections_hash_only_the_named_payload(tmp_path):
    path = tmp_path / 'object.o'
    path.write_bytes(elf64([('.ltext', 1, b'abc'), ('.lrodata', 1, b'gpu'), ('.bss', 8, b'')]))
    assert subblock.elf_sections(path, {'.lrodata'}) == {'.lrodata': hashlib.sha256(b'gpu').hexdigest()}
    with pytest.raises(ValueError, match='unexpected section'):
        subblock.elf_sections(path, {'.bss'})
    path.write_bytes(b'not an object')
    with pytest.raises(ValueError, match='ELF64'):
        subblock.elf_sections(path, {'.lrodata'})


def test_export_check_accepts_any_host_launcher_but_pins_the_gpu_payload(tmp_path, monkeypatch):
    payload = {'.lrodata': hashlib.sha256(b'kernel image').hexdigest(),
               '.ldata': hashlib.sha256(b'launch data').hexdigest()}
    pin = json.loads(subblock.PIN.read_bytes())
    assert set(pin['gpu_payload_sha256']) == {'.lrodata', '.ldata'} and 'h3_subblock_v07.o' not in pin['sha256']
    pin.update(gpu_payload_sha256=payload,
               sha256={name: hashlib.sha256(name.encode()).hexdigest() for name in pin['sha256']})
    pinned = tmp_path / 'pin.json'
    pinned.write_text(json.dumps(pin))
    monkeypatch.setattr(subblock, 'PIN', pinned)
    for index, host in enumerate((b'zen launcher', b'intel launcher')):
        write_export(tmp_path / f'ok{index}', host=host)
        assert subblock.validate_export(tmp_path / f'ok{index}', '100a')['schema'] == pin['schema']
    # A legacy export that also lists the whole object is accepted on the same terms.
    legacy = tmp_path / 'legacy'
    write_export(legacy)
    report = json.loads((legacy / 'export.json').read_bytes())
    report['sha256']['h3_subblock_v07.o'] = '0' * 64
    (legacy / 'export.json').write_text(json.dumps(report))
    subblock.validate_export(legacy, '100a')
    write_export(tmp_path / 'gpu', gpu=b'other kernel')
    with pytest.raises(ValueError, match='GPU payload'):
        subblock.validate_export(tmp_path / 'gpu', '100a')
    write_export(tmp_path / 'claim', gpu_payload_sha256={'.lrodata': '0' * 64, '.ldata': '0' * 64})
    with pytest.raises(ValueError, match='GPU payload'):
        subblock.validate_export(tmp_path / 'claim', '100a')
    write_export(tmp_path / 'header')
    (tmp_path / 'header/h3_subblock_v07.h').write_bytes(b'changed')
    with pytest.raises(ValueError, match='file changed'):
        subblock.validate_export(tmp_path / 'header', '100a')
    with pytest.raises(ValueError, match='B200'):
        subblock.validate_export(tmp_path / 'ok0', '103a')
