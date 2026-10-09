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
