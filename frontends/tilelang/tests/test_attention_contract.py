"""CPU-only contract authoring against the sm90 attention partner stages.

``fixtures/attention_partner.json`` is a verbatim copy of the work ring, corrected_o,
final_stats and o_drained endpoints (with their claims) and the declared entry
rendezvous from the attention ``tma-load``/``core`` manifests.
"""
from dataclasses import replace
import json
import subprocess
import zipfile
from pathlib import Path

import pytest
import tilelang.language as T
from tilelang import layout as tl_layout, tvm
from pipec2.pack.assemble import assemble

from pipelink_tilelang import set_context
from pipelink_tilelang.annotate import AnnotationError, Ledger
from pipelink_tilelang.capture import capture
from pipelink_tilelang.contract import Partner, consume, produce
from pipelink_tilelang.declarations import ParamField, ParamsSpec
from pipelink_tilelang.explicit import ContractStage, RoleSpec, _annotations, emit
from pipelink_tilelang.externalize import externalize_role
from pipelink_tilelang.layout import LayoutUnsupported, from_captured, plane_shape, to_tilelang

ROOT = Path(__file__).resolve().parents[3]
PACK = ROOT / "pipec/src/pipec2/pack"
FIXTURE = Path(__file__).resolve().parent / "fixtures/attention_partner.json"
SM90 = tvm.target.Target({"kind": "cuda", "arch": "sm_90a"})
CONFIG = {"tl.disable_warp_specialized": True, "tl.disable_buffer_init_check": True}
# The attention corrected_o plane: two 64-column bf16 halves, byte-unit SW128.
CORRECTED_O = {"encoding": "pl-layout-v2", "elem_bits": 16, "shape": [128, 64, 2], "stride": [64, 1, 8192],
               "swizzle": {"B": 3, "M": 4, "S": 3, "unit": "byte"}}
SMEM_BASE = "pipelink_attention_smem_base"


@pytest.fixture
def partner(tmp_path, monkeypatch):
    import pipelink_tilelang
    monkeypatch.setattr(pipelink_tilelang, "context", pipelink_tilelang.context)
    set_context("sm_90a", None)
    path = tmp_path / "attention.plink"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", FIXTURE.read_text())
    return Partner.load(path)


def _shared(shape, dtype="bfloat16"):
    return tvm.tirx.decl_buffer(shape, dtype, scope="shared")


def test_rank3_plane_canonicalizes_to_tilelang_full_bank_plane():
    assert plane_shape(CORRECTED_O, profile="sm_90a") == (128, 128)
    for shape in ((128, 128), (1, 128, 128)):
        buffer = _shared(shape)
        converted = to_tilelang(CORRECTED_O, buffer, profile="sm_90a")
        assert tvm.ir.structural_equal(converted, tl_layout.make_full_bank_swizzled_layout(buffer))
    # (r, c, h) -> (r, h*C + c): the captured 2-D plane is the same affine map.
    captured = from_captured(tl_layout.make_full_bank_swizzled_layout(_shared((128, 128))), elem_bits=16,
                             buffer=_shared((128, 128)))
    assert captured["shape"] == [128, [64, 2]] and captured["stride"] == [64, [1, 8192]]


@pytest.mark.parametrize("fields", [
    {"stride": [64, 1, 4096]},                    # halves overlap
    {"stride": [128, 1, 64]},                     # row-major 128-wide, unlike SW128 bank tiling
    {"shape": [128, 32, 4], "stride": [32, 1, 4096]},  # 64-byte leaves under a 128-byte swizzle
], ids=("overlap", "row-major", "narrow-leaf"))
def test_rank3_planes_that_do_not_concatenate_fail_closed(fields):
    with pytest.raises(LayoutUnsupported):
        to_tilelang({**CORRECTED_O, **fields}, _shared((128, 128)), profile="sm_90a")


@pytest.mark.parametrize("origin_bits,accepted", [
    (0, True), (8 * 1024, True), (8 * 160 * 1024, True),
    (8 * 512, False),   # inside a 1024-byte SW128 repeat
    (4, False),         # not byte-aligned
])
def test_swizzled_origin_aligns_to_the_swizzle_repeat(origin_bits, accepted):
    layout = {**CORRECTED_O, "origin_bits": origin_bits}
    if accepted:
        to_tilelang(layout, _shared((128, 128)), profile="sm_90a")
    else:
        with pytest.raises(LayoutUnsupported):
            to_tilelang(layout, _shared((128, 128)), profile="sm_90a")


def test_linear_origin_is_byte_and_element_aligned():
    linear = {"encoding": "pl-layout-v2", "elem_bits": 32, "shape": [13], "stride": [1]}
    to_tilelang({**linear, "origin_bits": 8 * 164968}, _shared((2, 13), "int32"), profile="sm_90a")
    with pytest.raises(LayoutUnsupported):
        to_tilelang({**linear, "origin_bits": 8 * 2}, _shared((2, 13), "int32"), profile="sm_90a")


def _kernel(work, output, stats, drained):
    @T.prim_func
    def kernel(O: T.Tensor((128, 128), "bfloat16"), L: T.Tensor((2, 128), "float32")):
        with T.Kernel(1, threads=32):
            ring = work.buffer(depth=2)
            plane = output.buffer(depth=1)
            final = stats.buffer(depth=1)
            work.wait(1, 0)
            output.wait(0, 0)
            stats.wait(0, 0)
            for i, j in T.Parallel(128, 128):
                O[i, j] = plane[0, i, j] + T.cast(ring[1, 12], "bfloat16")
            for i, j in T.Parallel(2, 128):
                L[i, j] = final[0, i, j]
            output.release(0)
            stats.release(0)
            work.release(1)
            drained.arrive(0)
    return kernel


def _channels(partner, role="epilogue"):
    return (consume(partner, "work_epilogue", role=role), consume(partner, "corrected_o", role=role),
            consume(partner, "final_stats", role=role), produce(partner, "o_drained", role=role))


def test_ring_buffers_bind_their_contract_origins(partner):
    work, output, stats, drained = channels = _channels(partner)
    result = capture(_kernel(*channels), target=SM90, pass_configs=CONFIG)
    # corrected_o: one slot of TileLang's full-bank (128, 128) plane, slot stride 32768 B.
    plane = from_captured(result.layout_objects["plane"], elem_bits=16, buffer=_shared((1, 128, 128)))
    assert plane["ring_stride_elements"] * 2 == output.endpoint.ring.stage_stride_bytes
    assert plane["swizzle"]["B"] == 3
    storage = {record.buffer: record for channel in channels for record in channel.external
               if type(record).__name__ == "ExternalStorage"}
    assert {name: (record.handle, record.offset_bytes) for name, record in storage.items()} == {
        "ring": (SMEM_BASE, 164968), "plane": (SMEM_BASE, 0), "final": (SMEM_BASE, 165280)}
    assert work.dtype == "s32"
    externals = [record for channel in channels for record in channel.external]
    body = externalize_role(result.modules["pre_codegen"], role="epilogue", warp_range=(0, 1), branch=None,
                            external=externals, prefix="t_", declared_params=("O", "L"), target=SM90).body
    bindings = {line.split("=")[0].split()[-1]: line for line in body.splitlines() if f"{SMEM_BASE}()" in line}
    assert f"{SMEM_BASE}() + 164968" in bindings["ring"] and f"{SMEM_BASE}() + 165280" in bindings["final"]
    assert "+ 0" not in bindings["plane"] and ")) +" not in bindings["plane"]


def _mirror(directory, endpoint, direction, owner_role):
    """The fixture endpoint as the opposite stage publishes it."""
    raw = json.loads(FIXTURE.read_text())
    row = next(item for item in raw["endpoints"] if item["name"] == endpoint)
    row.update(direction=direction, owner_role=owner_role)
    raw["stage"]["roles"][owner_role] = {"entry": f"test_attention_partner_role_{owner_role}", "warps": 1}
    path = directory / f"mirror-{endpoint}.plink"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(raw))
    return Partner.load(path)


def test_completion_only_endpoint_supports_wait_and_arrive_only(tmp_path, partner):
    drained = produce(partner, "o_drained", role="epilogue")
    assert drained.space == "none" and drained.dtype is None and drained.protocol.release == "external"
    with pytest.raises(ValueError):
        drained.buffer(depth=1)
    waiting = consume(_mirror(tmp_path, "o_drained", "produce", "epilogue"), "o_drained", role="load")

    @T.prim_func
    def kernel():
        with T.Kernel(1, threads=32):
            drained.arrive(0)
            waiting.wait(0, 0)

    result = capture(kernel, target=SM90, pass_configs=CONFIG)
    records = drained.external + waiting.external
    assert {(record.side, record.native.offset_bytes) for record in records} == {("full", 240)}
    body = externalize_role(result.modules["pre_codegen"], role="epilogue", warp_range=(0, 1), branch=None,
                            external=records, prefix="t_", declared_params=(), target=SM90).body
    assert "pipelink_attention_mbarrier_base(), 16, 240, 8, 0).arrive()" in body
    assert "pipelink_attention_mbarrier_base(), 16, 240, 8, 0).wait(0)" in body


def _module(partner, *, forced_warp, reg_groups=True):
    channels = _channels(partner)
    params = ParamsSpec("EpilogueParams", (ParamField("O", "__nv_bfloat16*"), ParamField("L", "float*"),
                                           ParamField("aux", "uint8_t", elements=128)), alignas=128)
    stage = ContractStage("test.attention.tilelang-epilogue", "1.0.0", 255, params,
                          {"epilogue": RoleSpec((0, 1), ("O", "L"))}, list(channels))
    composition = {"forced_bindings": {"pipelink_tilelang_warp_epilogue": forced_warp}}
    if reg_groups:
        composition["reg_groups"] = {"aux_wg": ("epilogue",)}
    module = type("Recipe", (), dict(STAGE=stage, COMPOSITION=composition))
    result = capture(_kernel(*channels), target=SM90, pass_configs=CONFIG)
    generated = {None: dict(facts=result.facts, passes=result.passes, snapshots=result.snapshots,
                            modules=result.modules, layout_objects=result.layout_objects,
                            external={"epilogue": [record for channel in channels for record in channel.external]})}
    return module, generated


def _records(directory, module, generated):
    annotations, _ = _annotations(module.STAGE, module.COMPOSITION, Ledger(), generated)
    params = module.STAGE.params
    source = directory / "annotations.cpp"
    source.write_text('#include <cstdint>\n#include <pipelink/stage.hpp>\nusing __nv_bfloat16 = unsigned short;\n'
                      f"struct alignas({params.alignas}) {params.name} {{ "
                      + " ".join(f"{field.declarator};" for field in params.fields) + " };\n"
                      + "".join(f'extern "C" __device__ void test_attention_tilelang_epilogue_role_{role}(const void*, unsigned);\n'
                                for role in module.STAGE.roles) + annotations)
    binary = directory / "annotations"
    compiled = subprocess.run(["g++", "-std=c++17", "-DPIPELINK_EMIT", "-D__device__=", f"-I{PACK / 'include'}",
                               f"-I{PACK.parent / 'host/include'}", str(source), str(PACK / "runner.cpp"),
                               "-o", str(binary)], capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr
    ran = subprocess.run([str(binary)], capture_output=True, text=True)
    assert ran.returncode == 0, ran.stderr
    return json.loads(ran.stdout)["records"]


def test_entry_rendezvous_and_reg_group_are_redeclared_from_the_partner(tmp_path, partner):
    module, generated = _module(partner, forced_warp=1)
    manifest = assemble({"records": _records(tmp_path, module, generated)}, profile="sm_90a")
    assert manifest["stage"]["entry_rendezvous"] == partner.raw["stage"]["entry_rendezvous"]
    regs = next(claim for claim in manifest["claims"] if claim["domain"] == "regs")
    partner_regs = next(claim for claim in partner.raw["claims"] if claim["domain"] == "regs")
    assert (regs["handle"], regs["reg_group"], regs["values"], regs["roles"]) == (
        partner_regs["handle"], "aux_wg", partner_regs["values"], ["epilogue"])
    assert manifest["params"]["bytes"] == 256
    ledger = Ledger()
    _annotations(module.STAGE, module.COMPOSITION, ledger, generated)
    join = ledger.sources["stage.entry_rendezvous.threads"]
    assert join["class"] == "contract" and join["source"].endswith(":/stage/entry_rendezvous/threads")
    assert ledger.sources[f"claims.{partner_regs['handle']}.values"]["class"] == "contract"


def test_slot_handle_backing_native_storage_keeps_its_partner_claim(tmp_path, partner):
    # The ring implies plain slot claims. A slot handle that also backs native barrier
    # storage stays a local claim, because native storage names a claim of the declaring stage.
    raw = json.loads(FIXTURE.read_text())
    drained = next(item for item in raw["endpoints"] if item["name"] == "o_drained")
    slot = drained["protocol"]["full"]["slots"]
    drained["protocol"]["full"]["native"] = {"base": slot, "offset_bytes": 0, "stride_bytes": 16}
    path = tmp_path / "self-backed.plink"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(raw))
    module, generated = _module(Partner.load(path), forced_warp=1)
    manifest = assemble({"records": _records(tmp_path, module, generated)}, profile="sm_90a")
    mbarriers = [claim["handle"] for claim in manifest["claims"] if claim["domain"] == "mbarrier"]
    assert slot in mbarriers
    assert not {"pipelink_attention_slots_work_epilogue_full", "pipelink_attention_slots_corrected_o_full"} & set(mbarriers)


def _entry(out, module, generated):
    projection = next(iter(emit(out, module, generated).values()))
    source = Path(projection["source"]).read_text()
    start = source.index('extern "C" __device__ void test_attention_tilelang_epilogue_role_epilogue(const void* raw_params')
    return source, source[start:]


def test_role_without_initializers_joins_at_entry(tmp_path, partner):
    module, generated = _module(partner, forced_warp=1)
    _, entry = _entry(tmp_path, module, generated)
    join = 'asm volatile("barrier.sync 0, 384;" ::: "memory");'
    assert entry.count(join) == 1 and entry.index(join) < entry.index("switch (variant)")
    assert "mbarrier.init" not in entry


def test_initializing_role_joins_after_its_kept_barrier_inits(tmp_path, partner):
    module, generated = _module(partner, forced_warp=0)
    source, entry = _entry(tmp_path, module, generated)
    join = entry.index('asm volatile("barrier.sync 0, 384;" ::: "memory");')
    fence = entry.index("fence.mbarrier_init.release.cluster")
    inits = [index for index in range(len(entry)) if entry.startswith("mbarrier.init.shared::cta.b64", index)]
    # work_epilogue full/empty, corrected_o full/empty, final_stats full/empty, o_drained full.
    assert len(inits) == 7 and max(inits) < fence < join
    assert "switch (variant)" in entry[:min(inits)]
    body = source[:source.index('extern "C" __device__ void test_attention_tilelang_epilogue_role_epilogue(const void* raw_params')]
    assert ".init(" not in body and "fence_barrier_init" not in body


def test_array_params_cannot_be_role_arguments(tmp_path, partner):
    module, generated = _module(partner, forced_warp=1)
    module.STAGE = replace(module.STAGE, roles={"epilogue": RoleSpec((0, 1), ("O", "L", "aux"))})
    with pytest.raises(AnnotationError) as caught:
        emit(tmp_path, module, generated)
    assert caught.value.field == "roles.epilogue.entry_params"


def test_partners_with_different_entry_rendezvous_are_rejected(tmp_path, partner):
    raw = json.loads(FIXTURE.read_text())
    raw["stage"]["entry_rendezvous"]["threads"] = 256
    path = tmp_path / "other.plink"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(raw))
    other = Partner.load(path)
    module, generated = _module(partner, forced_warp=1, reg_groups=False)
    module.STAGE = replace(module.STAGE, channels=[*module.STAGE.channels[:3], produce(other, "o_drained", role="epilogue")])
    with pytest.raises(AnnotationError) as caught:
        _annotations(module.STAGE, module.COMPOSITION, Ledger(), generated)
    assert caught.value.field == "stage.entry_rendezvous"
