from pipec2.tensor import expr, lower


def test_sparse_packet_worlds_preserve_full_native_column_base():
    context = {
        "packets": [{"dp": dp, "column": base + offset}
                    for dp in (0, 32) for base in (0, 128, 256, 384)
                    for offset in (0, 16)],
        "binding_domains": {"tmem_base": {
            "name": "tmem_base", "sort": "bits[32]", "values": [0, 128, 256, 384]}},
        "source_base_binding": "tmem_base", "source_base_reference": 0,
    }
    # The certified packet index is exact at the four declared bases, but it
    # cannot itself represent allocation bases between those sparse worlds.
    packet = ["add", ["mulc", 8, ["divc", ["coord", "participant"], 32]],
              ["mulc", 2, ["divc", ["toint", ["bind", "tmem_base"]], 128]]]
    domains = {"participant": expr.Domain("participant", "int", 0, 64),
               "tmem_base": expr.Domain("tmem_base", "bits[32]", 0, 512)}
    old_packet = expr.parse(packet, domains)
    assert context["packets"][expr.evaluate(old_packet, {"participant": 0, "tmem_base": 32})]["column"] == 0

    address, proof = lower._receipt_address(packet, {}, context, list(range(64)), "tmem_base")
    native = expr.parse(address, domains)
    for base in range(0, 512, 32):
        for participant in (0, 31, 32, 63):
            assert expr.evaluate(native, {"participant": participant, "tmem_base": base}) == ((participant // 32 * 32) << 16) + base
    assert proof["source_base_values"] == [0, 128, 256, 384]
    assert proof["equality_rows"] == 64 * 4


def test_chunked_receipt_relocation_fits_branch_free_native_address():
    context = {
        "packets": [{"dp": dp, "column": column}
                    for dp in range(0, 128, 32) for column in range(0, 512, 16)],
        "binding_domains": {
            "base": {"name": "base", "sort": "bits[32]", "values": [0, 256]},
            "chunk": {"name": "chunk", "sort": "bits[32]", "lo": 0, "hi": 8},
        },
        "source_base_binding": "base", "source_base_reference": 0,
    }
    packet = ["add", ["mulc", 32, ["modc", ["divc", ["coord", "participant"], 32], 4]],
              ["add", ["divc", ["toint", ["bind", "base"]], 16],
               ["add", ["mulc", 2, ["toint", ["bind", "chunk"]]],
                ["divc", ["coord", "participant"], 128]]]]
    address, proof = lower._receipt_address(packet, {}, context, list(range(256)), "source")
    domains = {
        "participant": expr.Domain("participant", "int", 0, 256),
        "chunk": expr.Domain("chunk", "bits[32]", 0, 8),
        "source": expr.Domain("source", "bits[32]", 0, 257),
    }
    native = expr.parse(address, domains)
    for base in (0, 256):
        for chunk in range(8):
            for participant in range(256):
                expected = ((participant // 32 % 4 * 32) << 16) + base + 32 * chunk + 16 * (participant // 128)
                assert expr.evaluate(native, {"participant": participant, "chunk": chunk, "source": base}) == expected
    assert proof["equality_rows"] == 256 * 8 * 2

    def operations(node):
        yield node.op
        for arg in node.args:
            if isinstance(arg, expr.Expr):
                yield from operations(arg)

    # An affine/bitfield mapping must not lower to a packet-table selector.
    assert "select" not in set(operations(native))


def test_irregular_receipt_offsets_retain_exact_selector_fallback():
    columns = [0, 16, 48, 112, 32, 144, 80]
    context = {
        "packets": [{"dp": 0, "column": base + column}
                    for base in (0, 256) for column in columns],
        "binding_domains": {
            "base": {"name": "base", "sort": "bits[32]", "values": [0, 256]},
            "chunk": {"name": "chunk", "sort": "int", "lo": 0, "hi": len(columns)},
        },
        "source_base_binding": "base", "source_base_reference": 0,
    }
    packet = ["add", ["bind", "chunk"], ["mulc", len(columns), ["divc", ["toint", ["bind", "base"]], 256]]]
    address, proof = lower._receipt_address(packet, {}, context, list(range(32)), "source")
    domains = {
        "participant": expr.Domain("participant", "int", 0, 32),
        "chunk": expr.Domain("chunk", "int", 0, len(columns)),
        "source": expr.Domain("source", "bits[32]", 0, 257),
    }
    native = expr.parse(address, domains)
    assert proof["offset_encoding"] == "packet_selection"
    for base in (0, 256):
        for chunk, column in enumerate(columns):
            for participant in range(32):
                assert expr.evaluate(native, {"participant": participant, "chunk": chunk, "source": base}) == base + column
