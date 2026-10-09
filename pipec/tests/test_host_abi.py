from __future__ import annotations

import ctypes as ct

import pytest

from pipec2.host import abi


def test_installed_header_c_cpp_ctypes_layouts():
    cpp_layouts = abi.probe_layouts()
    c_layouts = abi.probe_layouts(compiler='gcc')
    assert cpp_layouts == c_layouts
    assert set(cpp_layouts) == set(abi.ABI_TYPES)
    assert {row['id'] for row in cpp_layouts.values()} == set(range(0x70000001, 0x70000001 + len(abi.ABI_TYPES)))
    assert cpp_layouts['plk_header_v2']['id'] == 0x7000000A
    for name, typ in abi.ABI_TYPES.items():
        layout = cpp_layouts[name]
        assert layout['bytes'] == ct.sizeof(typ)
        assert layout['align'] == ct.alignment(typ)
        assert set(layout['fields']) == set(layout['members']) == {field for field, _ in typ._fields_}
        for field, field_type in typ._fields_:
            assert layout['members'][field] == {'offset': getattr(typ, field).offset, 'bytes': ct.sizeof(field_type)}


def test_aligned_buffer_keeps_zero_padding_and_rejects_out_of_bounds():
    buffer = abi.aligned_buffer(17, 128)
    assert buffer.address % 128 == 0
    buffer.write(b'abc', 3)
    assert buffer.bytes == b'\0' * 3 + b'abc' + b'\0' * 11
    with pytest.raises(ValueError, match='exceeds'):
        buffer.write(b'ab', 16)
    assert abi.aligned_buffer(0, 8).address == 0
    with pytest.raises(ValueError, match='power of two'):
        abi.aligned_buffer(1, 3)
