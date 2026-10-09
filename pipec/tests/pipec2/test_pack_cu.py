import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import pytest

from pipec2 import cli, container, stubgen
from pipec2.canonjson import canonical_bytes, sha256_hex
from pipec2.decls import declaration_digest, normalize_declaration
from pipec2.model import (
    Launch, NVCC_DEFAULT_MATH, PackError, Pipeline, PipelineTopology, ProtocolExpansion, SchemaError,
    Solution, SymbolInfo, SymbolTable,
)
from pipec2.pack.assemble import assemble
from pipec2.pack.objectfacts import ObjectFacts, parse_elf
from pipec2.profile import PROFILES, get_profile
from pipec2.schema import load_stage_manifest
from pipec2.tensor import expr
from pipec2.tensor.fit import FitError


FIXTURES = Path(__file__).parent / "fixtures" / "pack"
SCRATCH = Path(os.environ.get("PIPELINK_SCRATCH_ROOT", tempfile.gettempdir())) / "annotations-v2" / "phase3" / "Assemble2"


def _domain(name, hi):
    return {"name": name, "sort": "int", "lo": 0, "hi": hi}


def _registry():
    identity = {"result": "value", "work": "work", "version": "v1", "dtype": "f32", "interpretation": "exact",
                "coords": [_domain("m", 4), _domain("n", 4)]}
    return {"records": [
        {"kind": "stage", "id": "test.assemble", "version": "1"},
        {"kind": "params", "type": "Params", "bytes": 4, "align": 4,
         "members": [{"name": "count", "offset": 0, "bytes": 4, "align": 4}]},
        {"kind": "role", "name": "x", "symbol": "plx_role_x", "warps": 1,
         "declaration": 'extern "C" __device__ void plx_role_x(const void*, uint32_t);'},
        {"kind": "export", "name": "scale", "symbol": "plx_scale",
         "declaration": 'extern "C" __device__ float plx_scale(float);'},
        {"kind": "tensor_export", "name": "value", "identity": identity, "read_capabilities": [],
         "readiness": {"after": "ready", "reuse_after": "done"}, "fixed_bindings": {}, "backing": [],
         "calls": {"produce": "produce", "release": None},
         "placement": {"space": "smem", "units": "byte", "components": ["address"], "map": "swizzle"}},
        {"kind": "call", "name": "produce", "symbol": "plx_scale", "direction": "consume", "role": None,
         "export": "scale", "declaration": 'extern "C" __device__ float plx_scale(float);'},
        {"kind": "map", "name": "swizzle", "inputs": [{**d, "role": "coord"} for d in identity["coords"]],
         "outputs": ["address"], "rows": [[m, n, m ^ n] for m in range(4) for n in range(4)]},
    ]}


def _facts(*, missing_role=False, max_threads=None, cluster_dim=None):
    strong = ("plx_scale",) if missing_role else ("plx_role_x", "plx_scale")
    table = SymbolTable(strong, (), (), {name: SymbolInfo("func", "strong") for name in strong})
    return ObjectFacts(table, max_threads, cluster_dim)


def test_assembled_gf2_placement_round_trips_and_matches_enumeration():
    registry = _registry()
    manifest = assemble(registry, facts=_facts())
    stage = load_stage_manifest(manifest)
    fitted = stage.tensor.exports[0].placements[0].components["address"]
    expected = expr.parse(["toint", ["bxor", ["tobits", 2, ["coord", "m"]], ["tobits", 2, ["coord", "n"]]]],
                          {"m": expr.Domain("m", "int", 0, 4), "n": expr.Domain("n", "int", 0, 4)})
    assert "bxor" in str(fitted.to_json_obj())
    assert expr.ground(fitted, {"m": expr.Domain("m", "int", 0, 4), "n": expr.Domain("n", "int", 0, 4)}) == expr.ground(
        expected, {"m": expr.Domain("m", "int", 0, 4), "n": expr.Domain("n", "int", 0, 4)})
    assert manifest["provenance"]["sections"]["$.tensor.exports[0].placements"] == "fitted"
    assert manifest["provenance"]["inputs"]["map:swizzle"] == sha256_hex(canonical_bytes(registry["records"][-1]))


def test_unfittable_map_names_first_inconsistent_row():
    registry = _registry()
    record = registry["records"][-1]
    record.update(inputs=[{**_domain("m", 8), "role": "coord"}], rows=[[m, m * m] for m in range(8)])
    with pytest.raises(FitError) as caught:
        assemble(registry)
    failure = caught.value.failures[0]
    assert failure.clause_id == "T-FIT"
    assert "swizzle.address" in str(caught.value)
    assert failure.detail["row_index"] == 3
    assert failure.detail["first_mismatch"]["inputs"] == [3]
    assert failure.detail["first_mismatch"]["expected"] == 9


def test_launch_bound_checks_total_role_warps():
    with pytest.raises(PackError) as caught:
        assemble(_registry(), facts=_facts(max_threads=31))
    assert caught.value.failures[0].clause_id == "P-LAUNCH"
    assert caught.value.failures[0].detail == {"threads": 32, "max_threads": 31}
    load_stage_manifest(assemble(_registry(), facts=_facts(max_threads=32)))


def test_missing_role_symbol_is_not_attested():
    with pytest.raises(PackError) as caught:
        assemble(_registry(), facts=_facts(missing_role=True))
    assert caught.value.failures[0].clause_id == "P-SYM"
    assert caught.value.failures[0].detail["symbol"] == "plx_role_x"


def test_role_signature_cannot_be_relabelled_as_fixed_entry_abi():
    registry = _registry()
    registry["records"][2]["declaration"] = 'extern "C" __device__ float plx_role_x(float);'
    with pytest.raises(PackError) as caught:
        assemble(registry, facts=_facts())
    failure = caught.value.failures[0]
    assert failure.clause_id == "P-SYM"
    assert failure.detail["signature"] == "float(float)"


def test_explicit_warp_claim_suppresses_derived_claim():
    registry = _registry()
    registry["records"].append({"kind": "claim", "handle": "pipelink_explicit_warps", "domain": "warp",
                                "extent": 1, "align": 1, "roles": ["x"], "lifetime": "kernel"})
    manifest = assemble(registry)
    assert [claim["handle"] for claim in manifest["claims"] if claim["domain"] == "warp"] == ["pipelink_explicit_warps"]


def test_bad_shared_storage_alignment_has_pack_clause():
    registry = _registry()
    registry["records"].append({"kind": "claim_smem", "handle": "pipelink_smem", "type": "Storage", "bytes": 17,
                                "align": 16, "roles": ["x"], "lifetime": "kernel"})
    with pytest.raises(PackError) as caught:
        assemble(registry)
    assert caught.value.failures[0].clause_id == "P-CLAIM"


def test_compiled_tensor_lifetime_claims_round_trip_through_pack(tmp_path):
    if not shutil.which("g++"):
        pytest.skip("g++ is unavailable")
    pack = Path(__file__).resolve().parents[2] / "src/pipec2/pack"
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    emitter = tmp_path / "emit"
    compiled = subprocess.run(
        ["g++", "-std=c++17", "-DPIPELINK_EMIT", f"-I{pack / 'include'}",
         f"-I{pack.parent / 'host/include'}",
         str(FIXTURES / "header_tensor_lifetimes.cpp"), str(pack / "runner.cpp"), "-o", str(emitter)],
        env=environment, capture_output=True, text=True, timeout=180,
    )
    assert compiled.returncode == 0, compiled.stderr
    emitted = subprocess.run([str(emitter)], env=environment, capture_output=True, text=True, timeout=60)
    assert emitted.returncode == 0, emitted.stderr
    registry = _registry()
    registry["records"].extend(json.loads(emitted.stdout)["records"])
    stage = load_stage_manifest(assemble(registry, facts=_facts()))
    claims = {claim.handle: claim for claim in stage.claims}
    for handle, start, end in (
        ("pipelink_work", "work", "ready"),
        ("pipelink_ready", "ready", "done"),
        ("pipelink_after", "done", None),
    ):
        lifetime = claims[handle].lifetime
        assert (lifetime.tensor, lifetime.from_, lifetime.to) == ("value", start, end)
        assert (claims[handle].extent.const, claims[handle].align) == (16, 16)
    assert claims["pipelink_work"].roles == ["x"]
    assert claims["pipelink_after"].roles == ["x"]
    assert claims["pipelink_kernel"].lifetime == claims["pipelink_override"].lifetime == "kernel"
    assert claims["pipelink_phase"].lifetime == "phase:compute"
    assert set(stage.tensor.sites[0].guarantees) == {"enter_once", "single_completion"}


def test_incomplete_map_is_not_fit_as_partial_function():
    registry = _registry()
    registry["records"][-1]["rows"].pop()
    with pytest.raises(PackError) as caught:
        assemble(registry)
    assert caught.value.failures[0].clause_id == "P-MAP"
    assert caught.value.failures[0].detail == {"expected": 16, "actual": 15}


def test_call_endpoint_uses_explicit_partner_export_and_type_signature():
    registry = _registry()
    registry["records"].append({"kind": "call", "name": "channel", "symbol": "pipelink_call_channel",
                                "declaration": 'extern "C" __device__ void pipelink_call_channel(const void*);',
                                "direction": "produce", "role": "x", "export": "target"})
    manifest = assemble(registry)
    stage = load_stage_manifest(manifest)
    endpoint = next(endpoint for endpoint in stage.endpoints if endpoint.name == "channel")
    assert endpoint.owner_role == "x"
    assert endpoint.protocol.call.export == "target"
    assert endpoint.protocol.call.signature == "void(const void*)"
    assert endpoint.protocol.call.decl_sha256 == declaration_digest(endpoint.protocol.call.decl)
    provided = next(endpoint for endpoint in stage.endpoints if endpoint.name == "produce")
    assert provided.owner_role is None
    assert provided.protocol.call.export == "scale"


def _with_import(kind):
    registry = _registry()
    domain = [_domain("t", 4), _domain("p", 4)]
    demand = {"kind": kind, "domain": domain, "map": "input.demand",
              "outputs": ["m", "n", "participant", "position"] if kind == "positions" else ["m", "n", "address"]}
    if kind == "memory_view":
        demand.update(space="smem", units="byte")
    registry["records"].extend([
        {"kind": "site", "name": "site", "role": "x", "participants": {"all_threads": True},
         "invocation": "once", "guarantees": ["enter_once"], "observes": ["ready"]},
        {"kind": "tensor_import", "name": "input", "symbol": "pl_import",
         "declaration": 'extern "C" __device__ void pl_import(float*);',
         "identity": registry["records"][4]["identity"], "args": [{"name": "fragment", "meaning": "dest_fragment", "ctype": "float*"}],
         "demand": demand, "sites": ["site"], "completion": "done", "lends": [], "rendezvous": []},
        {"kind": "map", "name": "input.demand", "inputs": [{**item, "role": "coord"} for item in domain],
         "outputs": demand["outputs"],
         "rows": [[t, p, t, p, *([t + 32, p] if kind == "positions" else [t * 20 + p * 4])]
                  for t in range(4) for p in range(4)]},
    ])
    return registry


def test_demand_participant_outside_site_rejects_before_pack():
    with pytest.raises(PackError) as caught:
        assemble(_with_import("positions"))
    failure = caught.value.failures[0]
    assert failure.clause_id == "P-MAP"
    assert failure.detail == {"site": "site", "participants": [32, 33, 34, 35]}


def test_finite_site_participants_fit_without_losing_selected_threads():
    registry = _with_import("positions")
    site = next(record for record in registry["records"] if record["kind"] == "site")
    site["participants"] = {"map": "site.participants"}
    for row in registry["records"][-1]["rows"]:
        row[4] = row[0] * 2
    registry["records"].append({
        "kind": "map", "name": "site.participants",
        "inputs": [{"name": "i", "sort": "int", "values": [0, 2, 4, 6], "role": "coord"}],
        "outputs": ["participant"], "rows": [[i, i] for i in (0, 2, 4, 6)],
    })
    stage = load_stage_manifest(assemble(registry))
    participants = stage.tensor.sites[0].participants
    assert participants.domain[0].values == (0, 2, 4, 6)
    assert [expr.evaluate(participants.expr, {"i": i}) for i in participants.domain[0].values] == [0, 2, 4, 6]


def test_memory_view_preserves_identity_and_fitted_byte_addresses():
    stage = load_stage_manifest(assemble(_with_import("memory_view")))
    demand = stage.tensor.imports[0].demands[0]
    assert demand.kind == "memory_view"
    assert demand.space == "smem"
    assert demand.units == "byte"
    for t in range(4):
        for p in range(4):
            env = {"t": t, "p": p}
            assert {name: expr.evaluate(value, env) for name, value in demand.identity_coords.items()} == {"m": t, "n": p}
            assert expr.evaluate(demand.components["address"], env) == t * 20 + p * 4


def test_elf_attributes_check_launch_and_cluster_without_gpu():
    text = """.nv.info.kernel
        Attribute: EIATTR_MAX_THREADS
        Format: EIFMT_SVAL
        Value: 0x00000080 0x00000001 0x00000001
        Attribute: EIATTR_CTA_PER_CLUSTER
        Format: EIFMT_SVAL
        Value: 0x00000002 0x00000001 0x00000001
    """
    assert parse_elf(text) == (128, (2, 1, 1))
    assert parse_elf("Fatbin nvvm code: no ELF image") == (None, None)
    assert parse_elf(text.replace("EIATTR_CTA_PER_CLUSTER", "EIATTR_CLUSTER_DIM")) == (128, (2, 1, 1))
    with pytest.raises(PackError) as caught:
        assemble(_registry(), facts=_facts(cluster_dim=(2, 1, 1)))
    assert caught.value.failures[0].clause_id == "P-TOPO"


def test_removed_variant_define_option_is_not_accepted():
    with pytest.raises(SystemExit) as caught:
        cli.main(["pack", "stage.cu", "-o", "unused.plink", "--variant", "M=32"])
    assert caught.value.code == 2


@pytest.mark.skipif(not os.environ.get("PIPELINK_CUDA_HOME"), reason="requires PIPELINK_CUDA_HOME nvcc toolchain")
def test_pack_cu_derives_every_architecture_spelling_from_profile(monkeypatch):
    from pipec2.pack import pack_cu
    from pipec2.symbols import cuda_home

    SCRATCH.mkdir(parents=True, exist_ok=True)
    cuobjdump = cuda_home() / "bin" / "cuobjdump"
    with tempfile.TemporaryDirectory(dir=SCRATCH, prefix="test-") as directory:
        root = Path(directory)
        monkeypatch.setenv("PIPELINK_SCRATCH_ROOT", str(root))
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        manifests, reports, fatbins = {}, {}, {}
        for profile in sorted(PROFILES):
            output = root / f"{profile}.plink"
            reports[profile] = pack_cu(FIXTURES / "stage_min.cu", output, defines=["PACK_WIDTH=2"], profile=profile)
            manifests[profile], object_bytes = container.read(output)
            (root / f"{profile}.o").write_bytes(object_bytes)
            fatbins[profile] = subprocess.run([str(cuobjdump), "--dump-elf", str(root / f"{profile}.o")],
                                              capture_output=True, text=True, check=True).stdout
    reference = manifests["sm_103a"]
    for profile, manifest in manifests.items():
        arch = get_profile(profile).arch
        assert manifest["pipelink_abi"]["target"] == {"profile": profile, "lto_arch": arch.lto, "toolkit": "cuda-13.3.0",
                                                     "math": NVCC_DEFAULT_MATH.to_json_obj()}
        # The probe mechanism and the observed symbol surface do not depend on the architecture.
        assert reports[profile]["symbol_mechanism"] == reports["sm_103a"]["symbol_mechanism"]
        assert manifest["symbols"] == reference["symbols"]
        assert manifest["provenance"]["sections"]["$.symbols"] == "observed"
        compile_record = reports[profile]["compile"]
        assert [flag for flag in compile_record["emit"] if flag.startswith("arch=")] == [f"arch={arch.compute},code={arch.sm}"]
        assert [flag for flag in compile_record["device"] if flag.startswith("arch=")] == [f"arch={arch.compute},code={arch.lto}"]
        assert f"arch = {arch.sm}" in fatbins[profile]
        # Distinct recipes attest distinct provenance (below); the shared inputs are the same bytes.
        inputs, reference_inputs = manifest["provenance"]["inputs"], reference["provenance"]["inputs"]
        assert {key: inputs[key] for key in ("tu", "header", "runner", "registry")} == \
            {key: reference_inputs[key] for key in ("tu", "header", "runner", "registry")}
    assert len({manifest["provenance"]["inputs"]["recipe"] for manifest in manifests.values()}) == len(PROFILES)


def _resource_registry(role="mma", warps=1):
    records = _registry()["records"][:2]
    records.append({"kind": "role", "name": role, "symbol": f"plx_role_{role}", "warps": warps,
                    "declaration": f'extern "C" __device__ void plx_role_{role}(const void*, uint32_t);'})
    return {"records": records}


def _accumulator_registry():
    registry = _resource_registry()
    registry["records"].extend([
        {"kind": "endpoint", "name": "accumulator", "direction": "produce", "role": "mma",
         "space": "tmem", "scope": "cta", "proxy": "async", "tile": {"dtype": "f32", "shape": [128, 512]},
         "layout": {"encoding": "pl-layout-v2", "lanes": [0, 128], "cols": 512, "elem_bits": 32},
         "ring": {"depth": {"fixed": 2}, "stage_stride_bytes": 512},
         "protocol": {"kind": "mbarrier",
                      "full": {"slots": "pipelink_gemm_evt_sync_protocol_full",
                               "arrive": {"by": "mma", "count": 1}, "tx_bytes": 0},
                      "empty": {"slots": "pipelink_gemm_evt_sync_protocol_empty",
                                "arrive": {"by": "epilogue", "count": 256}, "tx_bytes": 0},
                      "init": {"owner": "mma", "phase": 0, "materializer": "stage"},
                      "release": "once-per-acquire"},
         "binds": {"base": "pipelink_gemm_evt_tmem", "full": "pipelink_gemm_evt_sync_protocol_full",
                   "empty": "pipelink_gemm_evt_sync_protocol_empty"}},
        {"kind": "claim", "handle": "pipelink_gemm_evt_tmem", "domain": "tmem",
         "extent": {"const": 512, "per": {}}, "align": 32, "lifetime": "kernel"},
        {"kind": "claim", "handle": "pipelink_gemm_evt_smem_mainloop", "domain": "smem",
         "extent": {"const": 0, "per": {"pipelink_gemm_evt_depth": 32768}}, "align": 128, "lifetime": "kernel"},
        {"kind": "claim", "handle": "pipelink_aa_fixed_load_order", "domain": "smem",
         "extent": {"const": 16, "per": {}}, "align": 8, "lifetime": "kernel",
         "absolute_base": 32, "alias": "gemm_evt_fixed_load_order"},
        {"kind": "claim", "handle": "pipelink_gemm_evt_sync_mainloop", "domain": "mbarrier",
         "extent": {"const": 4, "per": {}}, "align": 1, "offset": 16, "lifetime": "kernel"},
        {"kind": "claim", "handle": "pipelink_gemm_evt_warp_mma", "domain": "warp",
         "extent": {"const": 1, "per": {}}, "align": 1, "lifetime": "kernel", "roles": ["mma"]},
        *[{"kind": "claim", "handle": f"pipelink_gemm_evt_sync_protocol_{side}", "domain": "mbarrier",
           "extent": {"const": 2, "per": {}}, "align": 1, "lifetime": "kernel"} for side in ("full", "empty")],
        *[{"kind": "variant", "index": index, "key": {"pipelink_gemm_evt_depth": depth}}
          for index, depth in ((2, 6), (0, 4), (1, 5))],
    ])
    return registry


def test_gemm_accumulator_tmem_ring_rich_claims_and_declared_variants():
    manifest = assemble(_accumulator_registry())
    stage = load_stage_manifest(manifest)
    endpoint = stage.endpoints[0]
    assert endpoint.owner_role == "mma"
    assert endpoint.layout.to_json_obj() == {
        "encoding": "pl-layout-v2", "lanes": [0, 128], "cols": 512, "elem_bits": 32,
    }
    assert endpoint.ring.depth.fixed == 2
    assert endpoint.ring.stage_stride_bytes == 512
    assert endpoint.protocol.full.slots == "pipelink_gemm_evt_sync_protocol_full"
    assert endpoint.protocol.empty.arrive.count == 256
    assert endpoint.binds["base"] == "pipelink_gemm_evt_tmem"
    assert manifest["stage"]["variants"] == [
        {"index": index, "key": {"pipelink_gemm_evt_depth": depth}}
        for index, depth in enumerate((4, 5, 6))
    ]
    claims = {claim.handle: claim for claim in stage.claims}
    assert claims["pipelink_gemm_evt_smem_mainloop"].extent.per == {"pipelink_gemm_evt_depth": 32768}
    assert (claims["pipelink_aa_fixed_load_order"].absolute_base,
            claims["pipelink_aa_fixed_load_order"].alias) == (32, "gemm_evt_fixed_load_order")
    assert claims["pipelink_gemm_evt_sync_mainloop"].offset == 16
    assert [claim.handle for claim in stage.claims if claim.domain == "warp"] == ["pipelink_gemm_evt_warp_mma"]


def test_attention_register_group_and_fixed_depth_barrier_claims():
    registry = _resource_registry()
    registry["records"].extend([
        {"kind": "claim_regs", "handle": "pipelink_attention_regs_aux_wg", "reg_group": "aux_wg",
         "values": [32, 40], "roles": ["mma"]},
        {"kind": "claim", "domain": "mbarrier", "handle": "pipelink_attention_slots_q_full",
         "extent": {"const": 2, "per": {}}, "align": 1, "lifetime": "kernel"},
        {"kind": "endpoint", "name": "q", "direction": "consume", "role": "mma",
         "space": "smem", "scope": "cta", "proxy": "async",
         "tile": {"dtype": "f16", "shape": [256, 128]},
         "layout": {"encoding": "pl-layout-v2", "shape": [[256], [128]], "stride": [[128], [1]], "elem_bits": 16},
         "ring": {"depth": {"fixed": 2}, "stage_stride_bytes": 65536},
         "protocol": {"kind": "mbarrier",
                      "full": {"slots": "pipelink_attention_slots_q_full",
                               "arrive": {"by": "load", "count": 1}, "tx_bytes": 16384},
                      "empty": {"slots": "pipelink_attention_slots_q_empty",
                                "arrive": {"by": "mma", "count": 1}, "tx_bytes": 0},
                      "init": {"owner": "load", "phase": 0, "materializer": "stage"},
                      "release": "once-per-acquire"},
         "binds": {"base": "pipelink_attention_smem_base", "full": "pipelink_attention_slots_q_full",
                   "empty": "pipelink_attention_slots_q_empty"}},
    ])
    manifest = assemble(registry)
    stage = load_stage_manifest(manifest)
    assert next(claim for claim in manifest["claims"] if claim["domain"] == "regs") == {
        "domain": "regs", "handle": "pipelink_attention_regs_aux_wg", "reg_group": "aux_wg",
        "values": [32, 40], "roles": ["mma"],
    }
    assert stage.endpoints[0].ring.depth.fixed == next(c for c in stage.claims if c.domain == "mbarrier").extent.const == 2


def test_instruction_named_barrier_counter_phase_descriptor_and_dynamic_cluster():
    registry = _resource_registry("instr1")
    registry["records"].extend([
        {"kind": "endpoint", "name": "page_handoff", "direction": "produce", "role": "instr1",
         "space": "smem", "scope": "cta", "proxy": "generic", "tile": {"dtype": "u32", "shape": [64]},
         "layout": {"encoding": "pl-layout-v2", "elem_bits": 32, "shape": [[64]], "stride": [[1]]},
         "protocol": {"kind": "named_barrier", "id": "pipelink_profile_consumer_page_barrier", "count": 64},
         "binds": {"base": "pipelink_profile_consumer_page_stage1"}},
        {"kind": "endpoint", "name": "dependency_counter", "direction": "produce", "role": "instr1",
         "space": "gmem", "scope": "grid", "proxy": "generic", "tile": {"dtype": "u32", "shape": [1]},
         "layout": {"encoding": "pl-layout-v2", "elem_bits": 32, "shape": [[1]], "stride": [[1]]},
         "protocol": {"kind": "counter", "slot": "pipelink_profile_consumer_counter_slot",
                      "release": "red.release.gpu", "acquire": "ld.acquire.gpu"}, "binds": {}},
        {"kind": "claim", "handle": "pipelink_profile_consumer_page_stage1", "domain": "smem_page",
         "extent": {"const": 1, "per": {}}, "align": 1,
         "lifetime": "phase:instr_org.pipelink.profile-consumer.1-instruction"},
        {"kind": "claim", "handle": "pipelink_profile_consumer_page_barrier", "domain": "named_barrier",
         "extent": {"const": 1, "per": {}}, "align": 1, "lifetime": "kernel"},
        {"kind": "claim", "handle": "pipelink_profile_consumer_counter_slot", "domain": "gmem_counter",
         "extent": {"const": 1, "per": {}}, "align": 1, "lifetime": "kernel"},
        {"kind": "claim", "handle": "pipelink_profile_consumer_warp_instr1", "domain": "warp",
         "extent": {"const": 1, "per": {}}, "align": 1, "lifetime": "kernel"},
        {"kind": "descriptor", "bytes": 32, "align": 16},
        {"kind": "topology", "cta_group": 1,
         "cluster": {"compiled": "dynamic", "launch": {"x_multiple_of": 1, "max": [2, 1, 1]}},
         "persistent": True, "scheduler": "static", "multicast": []},
        {"kind": "hint", "key": "uses_tmem", "value": False},
    ])
    manifest = assemble(registry)
    stage = load_stage_manifest(manifest)
    endpoints = {ep.name: ep for ep in stage.endpoints}
    assert endpoints["page_handoff"].protocol.named_barrier.count == 64
    assert endpoints["dependency_counter"].protocol.counter.to_json_obj() == {
        "slot_handle": "pipelink_profile_consumer_counter_slot",
        "release_op": "red.release.gpu", "acquire_op": "ld.acquire.gpu",
    }
    assert manifest["stage"]["descriptor"] == {"bytes": 32, "align": 16}
    assert stage.topology.cluster_compiled == "dynamic"
    assert stage.topology.cluster_launch.max == (2, 1, 1)
    assert next(c for c in stage.claims if c.domain == "smem_page").lifetime == "phase:instr_org.pipelink.profile-consumer.1-instruction"
    assert [c.handle for c in stage.claims if c.domain == "warp"] == ["pipelink_profile_consumer_warp_instr1"]
    assert manifest["hints"] == {"uses_tmem": False}


def test_dequant_shared_completion_affine_swizzle_multicast_and_depth_handle():
    registry = _resource_registry("quant_load")
    common = {"kind": "endpoint", "direction": "produce", "role": "quant_load", "space": "smem",
              "scope": "cta", "proxy": "async"}
    protocol = {"kind": "mbarrier",
                "empty": {"slots": "pipelink_f3_raw_empty", "arrive": {"by": "transform", "count": 256}, "tx_bytes": 0},
                "init": {"owner": "quant_load", "phase": 0, "materializer": "stage"}, "release": "once-per-acquire"}
    registry["records"].extend([
        {**common, "name": "raw_quant", "tile": {"dtype": "s4", "shape": [128, 128]},
         "layout": {"encoding": "pl-layout-v2", "elem_bits": 4, "pointer_flag_bits": 4,
                    "shape": [[128], [128]], "stride": [[128], [1]],
                    "swizzle": {"B": 2, "M": 4, "S": 3, "unit": "byte"}},
         "ring": {"depth": {"handle": "pipelink_f3_l_depth"}, "stage_stride_bytes": 8192},
         "protocol": {**protocol, "full": {"slots": "pipelink_f3_raw_full",
                                          "arrive": {"by": "quant_load", "count": 1},
                                          "tx_bytes": {"sum": ["raw_quant", "raw_scale"]}}},
         "binds": {"base": "pipelink_f3_tensor_base", "full": "pipelink_f3_raw_full", "empty": "pipelink_f3_raw_empty"}},
        {**common, "name": "raw_scale", "tile": {"dtype": "bf16", "shape": [128, 128]},
         "layout": {"encoding": "pl-layout-v2", "elem_bits": 16, "pointer_flag_bits": 16,
                    "shape": [[128], [128]], "stride": [[1], [0]]},
         "ring": {"depth": {"handle": "pipelink_f3_l_depth"}, "stage_stride_bytes": 256},
         "protocol": {**protocol, "full": {"member_of": "raw_quant", "tx_bytes": "tile"}},
         "binds": {"base": "pipelink_f3_tensor_base", "empty": "pipelink_f3_raw_empty"}},
        {"kind": "topology", "cta_group": 2,
         "cluster": {"compiled": [2, 1, 1], "launch": {"x_multiple_of": 2, "max": [2, 1, 1]}},
         "persistent": True, "scheduler": "static", "multicast": ["raw_quant", "raw_scale", "activation"]},
    ])
    stage = load_stage_manifest(assemble(registry))
    root, member = stage.endpoints
    assert root.layout.to_json_obj()["swizzle"] == {"B": 2, "M": 4, "S": 3, "unit": "byte"}
    assert root.protocol.full.tx_bytes.sum == ["raw_quant", "raw_scale"]
    assert member.protocol.full.to_json_obj() == {"member_of": "raw_quant", "tx_bytes": "tile"}
    assert member.layout.to_json_obj()["stride"] == [[1], [0]]
    assert member.ring.depth.handle == "pipelink_f3_l_depth"
    assert set(stage.topology.multicast) == {"activation", "raw_quant", "raw_scale"}


def test_affine_atom_origin_and_table_extent_preserve_authored_carriers():
    registry = _resource_registry()
    registry["records"].extend([
        {"kind": "endpoint", "name": "atom", "direction": "produce", "role": "mma",
         "space": "smem", "scope": "cta", "proxy": "generic", "tile": {"dtype": "f32", "shape": [4]},
         "layout": {"encoding": "pl-layout-v2", "atom": "example_atom", "elem_bits": 32, "origin_bits": 128,
                    "shape": [[4]], "stride": [[1]], "swizzle": {"B": 0, "M": 4, "S": 3}},
         "protocol": {"kind": "named_barrier", "id": "pipelink_named", "count": 32}, "binds": {}},
        {"kind": "claim", "domain": "smem", "handle": "pipelink_table", "align": 16, "lifetime": "phase:compute",
         "extent": {"table": {"pipelink_depth=2": 128, "pipelink_depth=3": 256}}},
    ])
    manifest = assemble(registry)
    stage = load_stage_manifest(manifest)
    assert stage.endpoints[0].layout.to_json_obj() == registry["records"][3]["layout"]
    assert next(c for c in stage.claims if c.domain == "smem").extent.table == {
        "pipelink_depth=2": 128, "pipelink_depth=3": 256,
    }


def test_variant_entries_survive_assembly_and_require_defined_abi_compatible_symbols():
    registry = _registry()
    record = {"kind": "variant", "index": 3, "key": {}, "entry": "test_assemble_variant",
              "declaration": 'extern "C" __device__ void test_assemble_variant(const void*, uint32_t);'}
    registry["records"].append(record)
    stage = load_stage_manifest(assemble(registry))
    assert stage.variants[0].entry == "test_assemble_variant"
    with pytest.raises(PackError) as caught:
        assemble(registry, facts=_facts())
    assert caught.value.failures[0].clause_id == "P-SYM"
    assert caught.value.failures[0].detail["symbol"] == "test_assemble_variant"
    record["declaration"] = 'extern "C" __device__ float test_assemble_variant(float);'
    with pytest.raises(PackError) as caught:
        assemble(registry)
    assert caught.value.failures[0].clause_id == "P-SYM"


@pytest.mark.parametrize("records", [
    [{"kind": "unsupported"}],
    [{"kind": "descriptor", "bytes": 16, "align": 16}] * 2,
    [{"kind": "variant", "index": 0, "key": {"pipelink_depth": n}} for n in (2, 3)],
])
def test_unknown_or_duplicate_registry_records_reject(records):
    registry = _resource_registry()
    registry["records"].extend(records)
    with pytest.raises(PackError) as caught:
        assemble(registry)
    assert caught.value.failures[0].clause_id == "P-MAP"


def test_duplicate_variant_key_is_not_hidden_by_distinct_indices():
    registry = _resource_registry()
    registry["records"].extend({"kind": "variant", "index": i, "key": {}} for i in (0, 1))
    with pytest.raises(SchemaError) as caught:
        load_stage_manifest(assemble(registry))
    assert caught.value.failures[0].clause_id == "L-HANDLE"


@pytest.mark.parametrize("warp, valid", [(4, True), (5, True), (3, False), (6, False)])
def test_pack_initializing_warp_must_belong_to_absolute_owner_claim(warp, valid):
    registry = _accumulator_registry()
    endpoint = next(r for r in registry["records"] if r["kind"] == "endpoint")
    endpoint["protocol"]["init"]["warp"] = warp
    claim = next(r for r in registry["records"] if r.get("domain") == "warp")
    claim.update(absolute_base=4, extent={"const": 2, "per": {}})
    if valid:
        assert load_stage_manifest(assemble(registry)).endpoints[0].protocol.init.warp == warp
    else:
        with pytest.raises(PackError) as caught:
            assemble(registry)
        failure = caught.value.failures[0]
        assert failure.clause_id == "P-INIT"
        assert failure.detail == {"owner": "mma", "warp": warp, "handle": "pipelink_gemm_evt_warp_mma",
                                  "base": 4, "extent": 2, "variant": 0}


def test_pack_defers_initializing_warp_without_absolute_owner_claim():
    registry = _accumulator_registry()
    endpoint = next(r for r in registry["records"] if r["kind"] == "endpoint")
    endpoint["protocol"]["init"]["warp"] = 7
    assert load_stage_manifest(assemble(registry)).endpoints[0].protocol.init.warp == 7


def test_pack_defers_warp_only_init_even_with_absolute_warp_claim() -> None:
    registry = _accumulator_registry()
    endpoint = next(r for r in registry["records"] if r["kind"] == "endpoint")
    init = endpoint["protocol"]["init"]
    del init["owner"]
    init["warp"] = 7
    claim = next(r for r in registry["records"] if r.get("domain") == "warp")
    claim.update(absolute_base=4, extent={"const": 2, "per": {}})
    stage = load_stage_manifest(assemble(registry))
    assert stage.endpoints[0].protocol.init.to_json_obj() == init


def test_no_params_round_trip_preserves_layout_probe_and_schema_limits():
    params = {"kind": "params", "type": "void", "bytes": 0, "align": 1, "members": []}
    registry = {"records": [_registry()["records"][0], params]}
    manifest = assemble(registry)
    assert manifest["params"] == {
        "type": "void", "bytes": 0, "align": 1, "sha256_of_layout_probe": sha256_hex(canonical_bytes(params)),
    }
    stage = load_stage_manifest(manifest)
    assert stage.params.bytes == 0 and not stage.roles
    registry["records"].append(_registry()["records"][2])
    with pytest.raises(SchemaError) as caught:
        load_stage_manifest(assemble(registry))
    assert caught.value.failures[0].clause_id == "L-PARAMS"


def test_opaque_params_hashes_layout_record_without_weakening_manifest_schema():
    registry = _registry()
    params = registry["records"][1]
    params.update(type="Opaque", bytes=8, align=4, members=[], opaque=True)
    manifest = assemble(registry)
    assert manifest["params"] == {
        "type": "Opaque", "bytes": 8, "align": 4,
        "sha256_of_layout_probe": sha256_hex(canonical_bytes(params)),
    }
    assert load_stage_manifest(manifest).params.bytes == 8


def test_registered_types_reach_manifest_and_executable_trampolines(tmp_path):
    if not shutil.which("g++"):
        pytest.skip("g++ is unavailable")
    pack = Path(__file__).resolve().parents[2] / "src/pipec2/pack"
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    emitter = tmp_path / "emit"
    compiled = subprocess.run(
        ["g++", "-std=c++17", "-DPIPELINK_EMIT", "-DTYPED_PARAMS", f"-I{pack / 'include'}",
         f"-I{pack.parent / 'host/include'}",
         str(FIXTURES / "header_authoring.cpp"), str(pack / "runner.cpp"), "-o", str(emitter)],
        env=environment, capture_output=True, text=True, timeout=180,
    )
    assert compiled.returncode == 0, compiled.stderr
    emitted = subprocess.run([str(emitter)], env=environment, capture_output=True, text=True, timeout=60)
    assert emitted.returncode == 0, emitted.stderr
    registry = json.loads(emitted.stdout)
    registry["records"] = [record for record in registry["records"] if record["kind"] != "endpoint"]
    registry["records"].extend([
        {"kind": "role", "name": "worker", "symbol": "plx_worker", "warps": 1,
         "declaration": 'extern "C" __device__ void plx_worker(const void*, uint32_t);'},
        {"kind": "site", "name": "site", "role": "worker", "participants": {"all_threads": True},
         "invocation": "once", "guarantees": ["enter_once"], "observes": ["ready"]},
    ])
    manifest = assemble(registry)
    stage = load_stage_manifest(manifest)
    params = next(record for record in registry["records"] if record["kind"] == "params")
    assert manifest["params"]["sha256_of_layout_probe"] == sha256_hex(canonical_bytes(params))
    exported = stage.exports["packet"]
    printed = next(record["declaration"] for record in registry["records"]
                   if record["kind"] == "export" and record["name"] == "packet")
    assert exported.decl == printed
    assert normalize_declaration(exported.decl) == exported.decl
    assert exported.decl_sha256 == declaration_digest(printed)
    assert stubgen.declaration_parts(exported.decl, exported.symbol) == (
        "Packet", "(const Packet&, Packet*, const Packet*, sample::nested::Token*)", "",
    )
    assert stage.endpoints[0].protocol.call.signature == "Packet(const Packet&, Packet*, const Packet*, sample::nested::Token*)"
    assert stage.tensor.imports[0].callable.declaration == stage.exports["pointer"].decl
    pipeline = Pipeline("types", [], {}, ["kernel"], [], {}, [], [], {},
                        PipelineTopology(False, "static"), stage.target)
    solution = Solution({}, {}, {}, {}, {}, [], {}, Launch(32, 32, 0, (1, 1, 1)), [], 0)
    expansions = [
        ProtocolExpansion(name, "call", None, None, None, None, None, None, None, None,
                          (export.symbol, export.decl, export.decl_sha256), None, None)
        for name, export in stage.exports.items()
    ]
    source = stubgen.generate_stub(solution, pipeline, [stage], expansions)
    definitions = """
struct alignas(16) Packet { unsigned values[4]; };
namespace sample::nested { class Token { public: unsigned value; }; }
"""
    implementation = tmp_path / "implementation.cpp"
    implementation.write_text(definitions + """
extern "C" Packet plx_packet(const Packet& lhs, Packet* out, const Packet* rhs, sample::nested::Token* token) {
    *out = lhs;
    out->values[0] += rhs->values[0] + token->value;
    return *out;
}
extern "C" void plx_pointer(Packet* packet) { ++packet->values[0]; }
extern "C" Packet plx_value(Packet packet) { ++packet.values[0]; return packet; }
""")
    main = """
#include <cstring>
int main() {
    Packet lhs{}, rhs{}, out{};
    sample::nested::Token token{};
    unsigned left = 11, right = 7, tag = 3, answer = 0;
    std::memcpy(&lhs, &left, sizeof(left));
    std::memcpy(&rhs, &right, sizeof(right));
    std::memcpy(&token, &tag, sizeof(tag));
    auto result = pipelink_call_packet(lhs, &out, &rhs, &token);
    pipelink_call_pointer(&result);
    result = pipelink_call_value(result);
    std::memcpy(&answer, &result, sizeof(answer));
    return answer == 23 ? 0 : 1;
}
"""
    for real_definition in (False, True):
        stub = tmp_path / f"stub-{real_definition}.cpp"
        prelude = "#define __device__\n#define __forceinline__ inline\n"
        if real_definition:
            prelude += "#define PACKET_DEFINED\n#define SAMPLE__NESTED__TOKEN_DEFINED\n" + definitions
        stub.write_text(prelude + source + main)
        binary = tmp_path / f"trampolines-{real_definition}"
        compiled = subprocess.run(
            ["g++", "-std=c++17", str(stub), str(implementation), "-o", str(binary)],
            env=environment, capture_output=True, text=True, timeout=180,
        )
        assert compiled.returncode == 0, compiled.stderr
        run = subprocess.run([str(binary)], env=environment, capture_output=True, text=True, timeout=60)
        assert run.returncode == 0, run.stderr


@pytest.mark.parametrize("ctype", ["uint32_t", "const float*"])
def test_pack_rejects_placement_specific_source_types(ctype):
    registry = _with_import("memory_view")
    imported = next(record for record in registry["records"] if record["kind"] == "tensor_import")
    imported["args"].insert(0, {"name": "opaque", "meaning": "source_base", "ctype": ctype})
    with pytest.raises(PackError) as caught:
        assemble(registry)
    assert [(failure.clause_id, failure.path) for failure in caught.value.failures] == [
        ("P-TENSOR-SOURCE", "$.tensor.imports.input.callable.params[0].ctype")]
