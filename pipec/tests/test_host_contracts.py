from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
import tempfile

import pytest

import pipec2.link as link_module
from pipec2.canonjson import canonical_digest
from pipec2.dispatchgen import generate_dispatch
from pipec2.host.abi import probe_layouts
from pipec2.host.contracts import (
    host_params_layout, owner_contract_digest, parse_host, resolve_host,
    role_view_declaration, role_view_fields, validate_host_assembly,
)
from pipec2.host.profile import DTYPES, HOST_ABI, dtypes_digest
from pipec2.layout import load_layout
from pipec2.model import (
    ClusterConstraints, Handle, Launch, LinkError, ParamsDecl,
    Pipeline, PipelineTopology, Role, SchemaError, Solution, StageManifest,
    StageRef, StageTopology, TargetStamp, Variant,
)
from pipec2.profile import get_profile
from pipec2.schema import load_stage_manifest


TARGET = TargetStamp("3.0", "sm_103a", "lto_103", "cuda-13.3.0", HOST_ABI)
TARGETS = {"sm_103a": TARGET, "sm_90a": TargetStamp("3.0", "sm_90a", "lto_90", "cuda-13.3.0", HOST_ABI)}



@lru_cache(maxsize=16)
def _fixture_layouts(header, names):
    with tempfile.TemporaryDirectory(prefix="host-contract-test-") as directory:
        path = Path(directory) / "fixture.h"
        path.write_text(header)
        return probe_layouts(extra_headers=(path,), extra_types=dict(names))


def make_host(owner="fixture.integration", stage_ids=None, role_names=None, *, scheduler=True, target=TARGET):
    stage_ids = tuple(stage_ids or (owner,))
    role_names = tuple(role_names or (f"compute_{index}" for index in range(len(stage_ids))))
    roles = [f"{stage}:{role}" for stage, role in zip(stage_ids, role_names)]
    prefix = owner.replace(".", "_").replace("-", "_")
    host = {
        "owner": owner, "params_group": owner + ".params",
        "provider": {**{export: f"pipelink_params_{export}_{prefix}" for export in ("abi", "query", "build", "release")},
                     "protocol": 2, "args": {"bytes": 8, "align": 8, "type_id": 40},
                     "owner_contract_digest": canonical_digest({"owner": owner}), "abi_types": []},
        "scalars": {"id": 7, "read_roles": roles,
                    "fields": [{"id": 91, "name": "fixture.M", "type": "u64", "source": {"kind": "caller", "name": "M"}}]},
        "residue": {"id": 13, "name": owner + ".residue", "bytes": 8, "align": 8, "type_id": 41, "layout_digest": "0" * 64, "read_roles": roles},
        "domains": [], "channels": [], "workspaces": [], "transfers": [], "published_facts": [],
        "scheduler_contract": {},
        "variants": {stage: [Variant({}, 0, None).to_json_obj()] for stage in stage_ids},
    }
    if scheduler:
        schema = {"required": {
            "tile_extents": {"kind": "tuple", "items": [{"kind": "static", "type": "i32"}]},
            "grid_axes": {"kind": "tuple", "items": [{"kind": "enum", "values": ["M"]}]},
            "persistent": {"kind": "static", "type": "bool"},
            "cluster": {"kind": "tuple", "items": [{"kind": "static", "type": "i32"}] * 3},
        }, "optional": {}, "pinned": {}, "runtime_i32": [], "hardware_rules": {}, "allow_no_launch": False}
        host["scheduler_contract"] = {
            "owner": owner, "strategy": "fixture.static-grid", "schema": schema,
            "input_schema_digest": canonical_digest(schema),
            "inputs": [{"name": name, "source": {"kind": "constant", "value": value}}
                       for name, value in (("tile_extents", [16]), ("grid_axes", ["M"]), ("persistent", False), ("cluster", [1, 1, 1]))],
            "runtime_inputs": [], "workspace_ids": [],
        }
    header = f"typedef struct {prefix}_args {{ unsigned long value; }} {prefix}_args;\ntypedef struct {prefix}_residue {{ unsigned long value; }} {prefix}_residue;\n"
    inventory = [(prefix + "_args", ("value",)), (prefix + "_residue", ("value",))]
    for stage, role in zip(stage_ids, role_names):
        name = "plk_role_view_" + stage.replace(".", "_").replace("-", "_") + "_" + role
        fields = role_view_fields(host, f"{stage}:{role}")
        header += role_view_declaration(name, fields) + "\n"
        inventory.append((name, tuple(field["field"] for field in fields)))
    layouts = _fixture_layouts(header, tuple(inventory))
    ids = {name: layout["id"] for name, layout in layouts.items() if "id" in layout}
    ids.update({name: index + 40 for index, (name, _) in enumerate(inventory)})
    host["provider"]["abi_types"] = [
        {"id": ids[name], "name": name, **{key: layout[key] for key in ("bytes", "align", "layout_digest", "fields")}}
        for name, layout in sorted(layouts.items(), key=lambda item: ids[item[0]])
    ]
    host["residue"]["layout_digest"] = layouts[prefix + "_residue"]["layout_digest"]
    host["target"] = {"dtypes": dtypes_digest(target.profile)}
    host["provider"]["owner_contract_digest"] = owner_contract_digest(host, {stage: [Variant({}, 0, None).to_json_obj()] for stage in stage_ids})
    return host


def make_stage(host=None, stage_id="fixture.integration", role="compute_0", target=TARGET):
    if host is None:
        host = make_host(stage_ids=(stage_id,), role_names=(role,), target=target)
    prefix = stage_id.replace(".", "_").replace("-", "_") + "_"
    return StageManifest(
        stage_id, "1", prefix, {role: Role(prefix + role, 1)} if role else {}, {},
        "void(const void* params, unsigned variant)", "v2", [Variant({}, 0, None)], [], [],
        StageTopology(1, (1, 1, 1), ClusterConstraints(1, (1, 1, 1)), [], False, "static"),
        ParamsDecl(0, 1, "NoParams", "0" * 64), None, {}, target, None,
        parse_host(host, stage_id, {role: Role(prefix + role, 1)} if role else {}, target),
    )


def make_pipeline(stages):
    return Pipeline("fixture", [StageRef(stage.id + ".pla", stage.id) for stage in stages], {}, ["steady"], [], {}, [], [], {}, PipelineTopology(False, "static"), stages[0].target)


def make_solution(stages):
    return Solution({}, {}, {}, {}, {stage.id: {} for stage in stages}, [],
                    {role: (index, index + 1) for index, role in enumerate(role for stage in stages for role in stage.roles)},
                    Launch(32 * max(1, sum(len(stage.roles) for stage in stages)), 32, 0, (1, 1, 1)), [], 0)


def make_raw_manifest(host, stage_id="fixture.integration", role="compute_0"):
    return {
        "pipelink_abi": {"schema": TARGET.schema, "target": {"profile": TARGET.profile, "lto_arch": TARGET.lto_arch, "toolkit": TARGET.toolkit, "host_abi": HOST_ABI}},
        "stage": {"id": stage_id, "version": "1", "symbol_prefix": stage_id.replace(".", "_") + "_", "entry_abi": "v2", "roles": {role: {"entry": stage_id.replace(".", "_") + "_" + role, "warps": 1}} if role else {}, "exports": {}, "entry_signature": "void(const void* params, unsigned variant)", "variants": [{"key": {}, "index": 0, "entry": None}]},
        "endpoints": [], "claims": [], "topology": {"cta_group": {"fixed": 1}, "cluster": {"compiled": [1, 1, 1], "launch": {"x_multiple_of": 1, "max": [1, 1, 1]}}, "multicast": [], "persistent": False, "scheduler": "static"},
        "params": {"bytes": 0, "align": 1, "type": "NoParams", "sha256_of_layout_probe": "0" * 64}, "host": host,
    }


def make_channel(name="fixture.tensor", access="read", channel_id=11):
    return {"id": channel_id, "name": name, "logical_dtype": "u8", "storage_dtype": "u8", "dtype_id": 1,
            "rank": 1, "access": access, "domain": None, "views": [], "source": {"kind": "caller", "name": "tensor"},
            "initializer": {"kind": "caller"}, "alias": None, "lifetime": "release"}


def make_workspace(initializer=None, reset="open"):
    return {
        "id": 5, "name": "fixture.workspace", "params_group": "fixture.integration.params", "align": 8, "space": "gmem",
        "channel": None, "bytes": "query", "initializer": initializer or {"kind": "owner", "owner": "fixture.integration"}, "reset": reset,
    }


def refresh_host(host, stage_roles=None, target=TARGET):
    stage_roles = stage_roles or [(host["owner"], "compute_0")]
    declarations, inventory = [], []
    for stage, role in stage_roles:
        name = "plk_role_view_" + stage.replace(".", "_").replace("-", "_") + "_" + role
        fields = role_view_fields(host, f"{stage}:{role}")
        declarations.append(role_view_declaration(name, fields))
        inventory.append((name, tuple(field["field"] for field in fields) or ("reserved",)))
    layouts = _fixture_layouts("\n".join(declarations), tuple(inventory))
    for typ in host["provider"]["abi_types"]:
        if typ["name"] in layouts:
            typ.update({key: layouts[typ["name"]][key] for key in ("bytes", "align", "layout_digest", "fields")})
    host["target"] = {"dtypes": dtypes_digest(target.profile)}
    host["provider"]["owner_contract_digest"] = owner_contract_digest(host, {stage: [Variant({}, 0, None).to_json_obj()] for stage, _role in stage_roles})
    return host



def _copy_accounting_stages(*, paired=False, scale_tx="tile"):
    from pipec2.model import Arrive, BarrierSide, Depth, Endpoint, Init, Protocol, Ring, SharedMember, Tile, TxSum
    stage_roles = [("fixture.loader", "load"), ("fixture.consumer", "use")]
    host = make_host(stage_ids=tuple(s for s, _ in stage_roles), role_names=("load", "use"))
    specs = ([("activation", "bf16", "bf16", [64, 64, 2], [64, 1, 4096], 16384)] if paired else [
        ("raw_quant", "s4", "u8", [128, 128], [128, 1], 8192),
        ("raw_scale", "bf16", "bf16", [128, 128], [1, 0], 256),
    ])
    device = []
    for ordinal, (name, logical_dtype, dtype, tile, stride, footprint) in enumerate(specs):
        channel = make_channel("fixture." + name, channel_id=11 + ordinal)
        channel.update(logical_dtype=logical_dtype, storage_dtype=dtype, dtype_id=DTYPES[dtype]["id"], rank=3,
                       source={"kind": "caller", "name": name})
        host["channels"].append(channel)
        host["transfers"].append({"id": 2 + ordinal, "name": channel["name"] + ".transfer", "channel": channel["name"],
                                  "device_endpoint": {"stage": "fixture.loader", "endpoint": name}, "access": "read"})
        full = (BarrierSide("pipelink_full", Arrive(1, "load"), 32768) if paired else
                BarrierSide("pipelink_full", Arrive(1, "load"), TxSum(["raw_quant", "raw_scale"]))
                if name == "raw_quant" else SharedMember("raw_quant", scale_tx))
        protocol = Protocol("mbarrier", full, BarrierSide("pipelink_empty", Arrive(1, "use"), 0),
                            Init("load", 0, "stage"), "once-per-acquire", None, None, None)
        layout = {"encoding": "pl-layout-v2", "elem_bits": DTYPES[logical_dtype]["bits"], "shape": tile, "stride": stride}
        device.append(Endpoint(name, "produce", "load", "smem", "cta", "async", Tile(logical_dtype, tuple(tile)),
                               load_layout(layout, "smem", "$.layout"), Ring(Depth(1, None), footprint), protocol, {}))
    refresh_host(host, stage_roles)
    stages = [make_stage(host, stage_id, role) for stage_id, role in stage_roles]
    stages = [replace(stage, topology=replace(stage.topology, cta_group=2, cluster_compiled=(2, 1, 1),
                      cluster_launch=ClusterConstraints(2, (2, 1, 1)))) for stage in stages]
    return [replace(stages[0], endpoints=device),
            replace(stages[1], endpoints=[replace(ep, direction="consume", owner_role="use") for ep in device])]


def test_manifest_roundtrip():
    stage = load_stage_manifest(make_raw_manifest(make_host()))
    assert StageManifest.from_json_obj(stage.to_json_obj()) == stage


def test_mixed_blob_link_publication(tmp_path, monkeypatch):
    owner = make_stage()
    blob = replace(owner, id="fixture.blob", symbol_prefix="fixture_blob_", roles={"blob": Role("fixture_blob_entry", 1)},
                   host=None, params=ParamsDecl(12, 16, "Blob", "0" * 64), target=replace(TARGET, host_abi=None))
    stages = [owner, blob]
    pipeline, solution = make_pipeline(stages), make_solution(stages)
    assert validate_host_assembly(pipeline, stages) == []
    resolved = resolve_host(stages, solution, pipeline)
    params = host_params_layout(resolved)
    assert params.entries == [("fixture.blob", 48, 12, 16), ("fixture.integration.residue", 40, 8, 8), ("pipelink.shared_scalars", 0, 40, 8)]
    assert resolved["params"] == {"bytes": 64, "align": 64, "blobs": [{"stage_id": "fixture.blob", "offset": 48, "bytes": 12, "align": 16}]}
    artifacts = [link_module._Artifact(tmp_path / (stage.id + ".pla"), stage, b"object", "a" * 64) for stage in stages]
    monkeypatch.setattr(link_module, "_load_assembly", lambda _path: (pipeline, artifacts))
    monkeypatch.setattr(link_module, "_check_symbols", lambda _artifacts: None)
    monkeypatch.setattr(link_module, "_solve_pipeline", lambda *_args: solution)
    monkeypatch.setattr(link_module, "_compile_and_measure", lambda *_args: ("pipelink_kernel", (32, 0, 0), None))
    result = link_module.link(tmp_path / "pipeline.json", tmp_path / "out")
    assert result.params == params and result.host == resolved
    source = generate_dispatch(solution, pipeline, stages, [], params, host=resolved)
    header = "#define __global__\n#define __device__\n#define __grid_constant__\n#define __launch_bounds__(...)\n#define __cluster_dims__(...)\nstruct { unsigned x; } threadIdx;\n"
    body = "\n".join([
        'extern "C" void fixture_integration_compute_0(const void* p, unsigned) { const auto* v = static_cast<const plk_role_view_fixture_integration_compute_0*>(p); if (*static_cast<const unsigned long*>(v->residue) != 73 || v->scalars == v->residue) __builtin_abort(); }',
        'extern "C" void fixture_blob_entry(const void* p, unsigned) { if (*static_cast<const unsigned*>(p) != 19) __builtin_abort(); }',
        'int main() { PipelinkParams p{}; *reinterpret_cast<unsigned long*>(reinterpret_cast<unsigned char*>(&p) + 40) = 73; *reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(&p) + 48) = 19; threadIdx.x = 0; pipelink_kernel(p); threadIdx.x = 32; pipelink_kernel(p); }',
    ])
    import subprocess
    cpp, exe = tmp_path / "view.cpp", tmp_path / "view"
    cpp.write_text(header + source + body)
    subprocess.run(["g++", "-std=c++17", str(cpp), "-o", str(exe)], check=True, capture_output=True)
    subprocess.run([str(exe)], check=True)


def test_owner_groups_share_one_scalar_image_and_order_independence():
    first = make_stage()
    second_host = make_host("fixture.other", scheduler=False)
    second_host["scheduler_contract"] = {"owner": first.host["owner"]}
    second_host["scalars"]["fields"].append({"id": 92, "name": "fixture.N", "type": "i64", "source": {"kind": "caller", "name": "N"}})
    refresh_host(second_host)
    second = make_stage(second_host, "fixture.other")
    stages = [first, second]
    pipeline, solution = make_pipeline(stages), make_solution(stages)
    assert validate_host_assembly(pipeline, stages) == []
    linked = resolve_host(stages, solution, pipeline)
    assert linked == resolve_host(list(reversed(stages)), solution, pipeline)
    assert [group["scalars"]["offset"] for group in linked["groups"]] == [0, 0]
    assert [group["residue"]["offset"] for group in linked["groups"]] == [56, 64]
    assert all([field["name"] for field in group["scalars"]["fields"]] == ["fixture.M", "fixture.N"] for group in linked["groups"])
    assert len([entry for entry in host_params_layout(linked).entries if entry[0] == "pipelink.shared_scalars"]) == 1
    second_host["scalars"]["fields"][0]["type"] = "i64"
    refresh_host(second_host)
    wrong = [first, make_stage(second_host, "fixture.other")]
    assert validate_host_assembly(pipeline, wrong)[0].clause_id == "L-HOSTCHAN"


def test_owner_group_deduplicates_members():
    host = make_host(stage_ids=("fixture.first", "fixture.second"), role_names=("first", "second"))
    stages = [make_stage(host, "fixture.first", "first"), make_stage(host, "fixture.second", "second")]
    pipeline, solution = make_pipeline(stages), make_solution(stages)
    assert validate_host_assembly(pipeline, stages) == []
    linked = resolve_host(stages, solution, pipeline)
    assert linked["groups"][0]["stage_ids"] == ["fixture.first", "fixture.second"]
    assert linked["params"]["bytes"] == 64 and len(linked["groups"]) == 1
    assert "grid" not in linked and "grid" not in linked["static_launch"]


def test_removed_grammar_keys_reject():
    mutations = [lambda h: h.update(declared_segments=[]), lambda h: h.update(normalization={}),
                 lambda h: h["scheduler_contract"].update(geometry_source="query"),
                 lambda h: h["scalars"]["fields"][0]["source"].update(conversion="checked"),
                 lambda h: h["channels"].append({**make_channel(), "extents": [16]}),
                 lambda h: h["target"].update(scheduler="0" * 64)]
    for mutate in mutations:
        host = make_host()
        mutate(host)
        with pytest.raises(SchemaError) as caught:
            make_stage(host)
        assert caught.value.failures[0].clause_id == "L-HOSTCHAN"


@pytest.mark.parametrize("mutation", [
    lambda c: c["inputs"].pop(),
    lambda c: c["inputs"].append({"name": "unknown", "source": {"kind": "constant", "value": 1}}),
    lambda c: c["inputs"][1]["source"].update(value=["N"]),
    lambda c: c.update(input_schema_digest="f" * 64),
])
def test_scheduler_rejects_missing_unknown_enum_and_digest(mutation):
    host = make_host()
    mutation(host["scheduler_contract"])
    with pytest.raises(SchemaError) as caught:
        make_stage(host)
    assert caught.value.failures[0].clause_id == "L-HOSTCHAN"


def test_scheduler_body_disagreement_across_owners_rejects():
    first = make_stage()
    host = make_host("fixture.other")
    host["scheduler_contract"]["owner"] = first.host["owner"]
    host["scheduler_contract"]["schema"]["allow_no_launch"] = True
    host["scheduler_contract"]["input_schema_digest"] = canonical_digest(host["scheduler_contract"]["schema"])
    refresh_host(host)
    stages = [first, make_stage(host, "fixture.other")]
    assert validate_host_assembly(make_pipeline(stages), stages)[0].clause_id == "L-HOSTCHAN"


def test_generic_scheduler_runtime_projection_and_pinned_fact():
    host = make_host()
    contract = host["scheduler_contract"]
    contract["strategy"] = "vendor.unregistered-strategy"
    schema = contract["schema"]
    schema["required"]["count"] = {"kind": "runtime", "type": "i32"}
    schema["runtime_i32"] = ["count"]
    schema["hardware_rules"] = {"count": {"kind": "hardware", "name": "observed_sm_count"}}
    schema["pinned"] = {"tile_extents": [16]}
    contract["inputs"].append({"name": "count", "source": {"kind": "hardware", "name": "observed_sm_count"}})
    contract["runtime_inputs"] = ["count"]
    contract["input_schema_digest"] = canonical_digest(schema)
    host["scalars"]["fields"].append({"id": 92, "name": "fixture.sm_count", "type": "i64", "source": {"kind": "hardware", "name": "observed_sm_count"}})
    host["published_facts"] = [{"name": "fixture.tile", "type": {"kind": "tuple", "items": [{"kind": "static", "type": "i32", "value": 16}]},
                               "owner": host["owner"], "stage": host["owner"], "consumers": [], "value": [16]}]
    contract["inputs"][0]["source"] = {"kind": "fact", "name": "fixture.tile"}
    refresh_host(host)
    stage = make_stage(host)
    assert validate_host_assembly(make_pipeline([stage]), [stage]) == []
    linked = resolve_host([stage], make_solution([stage]), make_pipeline([stage]))
    assert linked["scheduler_contract"]["inputs"][0]["value"] == [16]
    host["scalars"]["fields"].pop()
    refresh_host(host)
    stage = make_stage(host)
    assert validate_host_assembly(make_pipeline([stage]), [stage])[0].clause_id == "L-HOSTCHAN"


@pytest.mark.parametrize("logical,accepted", [("s4", True), ("bf16", False)])
def test_channel_bit_preserving_packing(logical, accepted):
    host = make_host()
    host["channels"] = [{**make_channel(), "logical_dtype": logical}]
    refresh_host(host)
    if accepted:
        stage = make_stage(host)
        assert validate_host_assembly(make_pipeline([stage]), [stage]) == []
    else:
        with pytest.raises(SchemaError) as caught:
            make_stage(host)
        assert caught.value.failures[0].detail["subreason"] == "dtype"


@pytest.mark.parametrize("paired,expected", [(False, [(8192, 8192), (256, 256)]), (True, [(16384, 32768)])])
def test_transfer_endpoint_physical_identity_and_expected_credit(paired, expected):
    stages = _copy_accounting_stages(paired=paired)
    assert validate_host_assembly(make_pipeline(stages), stages) == []
    linked = resolve_host(stages, make_solution(stages), make_pipeline(stages))
    transfers = linked["groups"][0]["transfers"]
    assert [(item["endpoint"]["tile_bytes"], item["expect_tx_bytes"]) for item in transfers] == expected
    for item, (tile_bytes, _) in zip(transfers, expected):
        assert item["endpoint"]["ring"] == {"depth": 1, "stage_stride_bytes": tile_bytes}
        assert item["endpoint"]["access"] == "produce"
        assert item["endpoint"]["elem_bits"] == DTYPES[item["endpoint"]["dtype"]]["bits"]


@pytest.mark.parametrize("mutation", [
    lambda h: h["transfers"][0]["device_endpoint"].update(stage="missing.stage"),
    lambda h: h["transfers"][0].update(access="write"),
    lambda h: h["channels"][0].update(logical_dtype="u8"),
    lambda h: h["residue"]["read_roles"].remove("fixture.loader:load"),
])
def test_transfer_endpoint_resolution_direction_width_and_reader_reject(mutation):
    stages = _copy_accounting_stages()
    host = deepcopy(stages[0].host)
    mutation(host)
    refresh_host(host, [("fixture.loader", "load"), ("fixture.consumer", "use")])
    wrong = [replace(stage, host=host) for stage in stages]
    assert validate_host_assembly(make_pipeline(wrong), wrong)[0].clause_id == "L-HOSTCHAN"


def test_cross_owner_residue_and_disagreeing_group_reject():
    host = make_host()
    host["residue"]["read_roles"] = ["other.owner:compute"]
    stage = make_stage(host)
    assert validate_host_assembly(make_pipeline([stage]), [stage])[0].clause_id == "L-HOSTCHAN"
    host = make_host(stage_ids=("fixture.first", "fixture.second"), role_names=("first", "second"))
    first = make_stage(host, "fixture.first", "first")
    host["provider"]["owner_contract_digest"] = "f" * 64
    second = make_stage(host, "fixture.second", "second")
    assert validate_host_assembly(make_pipeline([first, second]), [first, second])[0].clause_id == "L-HOSTCHAN"


def test_workspace_per_launch_reset_accepts_zero_initialization():
    host = make_host()
    host["workspaces"] = [make_workspace({"kind": "zero"}, "launch")]
    refresh_host(host)
    stage = make_stage(host)
    assert validate_host_assembly(make_pipeline([stage]), [stage]) == []
    assert resolve_host([stage], make_solution([stage]), make_pipeline([stage]))["groups"][0]["workspaces"][0]["reset"] == "launch"


@pytest.mark.parametrize("mutation,reason", [
    (lambda w: w.update(reset="launch"), "initializer"),
    (lambda w: w.pop("reset"), "inventory"),
])
def test_workspace_reset_rejects(mutation, reason):
    host = make_host()
    host["workspaces"] = [make_workspace()]
    mutation(host["workspaces"][0])
    refresh_host(host)
    with pytest.raises(SchemaError) as caught:
        make_stage(host)
    assert caught.value.failures[0].clause_id == "L-HOSTCHAN"
    assert caught.value.failures[0].detail["subreason"] == reason


def _roleless_member(owner="fixture.callee", *, residue=True, scalars=True):
    """A role-less member stage of its own host group exporting `<prefix>get(const void* view)`."""
    from pipec2.decls import declaration_digest
    from pipec2.model import Export
    host = make_host(owner, scheduler=False)
    host["scheduler_contract"] = {"owner": "fixture.integration"}
    for key, keep in (("residue", residue), ("scalars", scalars)):
        if keep:
            host[key]["read_roles"] = []
        else:
            host[key] = None
    refresh_host(host)
    stage = make_stage(host, owner, "")
    symbol = stage.symbol_prefix + "get"
    declaration = f'extern "C" __device__ unsigned long {symbol}(const void* view);'
    return replace(stage, exports={"get": Export(symbol, declaration, declaration_digest(declaration))})


def _view_call_pairs(caller, callees, *, role="compute_0", first="const void* view"):
    from pipec2.decls import declaration_digest
    from pipec2.layout import load_layout as layout
    from pipec2.model import CallDecl, Endpoint, Protocol, Tile
    reg = layout({"encoding": "pl-layout-v2", "elem_bits": 16, "shape": [1], "stride": [1]}, "reg", "$.layout")

    def endpoint(name, direction, symbol):
        declaration = f'extern "C" __device__ unsigned long {symbol}({first});'
        return Endpoint(name, direction, role if direction == "produce" else None, "reg", "cta", "generic",
                        Tile("bf16", (1,)), reg, None,
                        Protocol("call", None, None, None, None,
                                 CallDecl("get", "unsigned long(const void*)", declaration, declaration_digest(declaration)),
                                 None, None), {})
    return [(callee.id.replace(".", "_"),
             [link_module._EndpointRecord(caller, endpoint(callee.id.replace(".", "_"), "produce",
                                                           "pipelink_call_" + callee.id.replace(".", "_")))],
             [link_module._EndpointRecord(callee, endpoint(callee.id.replace(".", "_"), "consume",
                                                           callee.exports["get"].symbol))])
            for callee in callees]


def test_view_call_passes_each_callee_its_owner_view(tmp_path):
    from pipec2.model import ProtocolExpansion
    from pipec2.stubgen import generate_stub
    from pipec2.symbols import check_params_accessors
    # Two role-less owners, listed out of stage-ID order; one has no shared scalars.
    caller, callee, another = make_stage(), _roleless_member(), _roleless_member("fixture.another", scalars=False)
    stages = [caller, callee, another]
    pipeline = make_pipeline(stages)
    assert validate_host_assembly(pipeline, stages) == [] and check_params_accessors(stages) == []
    assert link_module._params_composition(pipeline, stages)[1] == stages
    solution = make_solution(stages)
    resolved = resolve_host(stages, solution, pipeline)
    params = host_params_layout(resolved)
    residues = {group["owner"]: group["residue"]["offset"] for group in resolved["groups"]}
    pairs = _view_call_pairs(caller, [callee, another])
    expansions = [ProtocolExpansion(channel, "call", None, None, None, None, None, None, None, None,
                                    (consumers[0].stage.exports["get"].symbol, consumers[0].stage.exports["get"].decl,
                                     consumers[0].stage.exports["get"].decl_sha256), None, None)
                  for channel, _, consumers in pairs]
    stub = generate_stub(solution, pipeline, stages, expansions, link_module._view_calls(pairs, resolved))
    source = generate_dispatch(solution, pipeline, stages, [], params, host=resolved)
    header = ("#define __global__\n#define __device__\n#define __grid_constant__\n#define __shared__\n"
              "#define __forceinline__ inline\n#define __launch_bounds__(...)\n#define __cluster_dims__(...)\n"
              "#define __syncthreads()\nstruct { unsigned x; } threadIdx;\n#include <pipelink/host_view.h>\n")
    body = "\n".join([
        # The callees read only their own owner's records, whatever view the caller holds.
        'extern "C" unsigned long fixture_callee_get(const void* view) { const auto* v = static_cast<const pipelink_host_view*>(view);'
        ' if (v->scalars == nullptr || v->scalars == v->residue) __builtin_abort(); return *static_cast<const unsigned long*>(v->residue); }',
        'extern "C" unsigned long fixture_another_get(const void* view) { const auto* v = static_cast<const pipelink_host_view*>(view);'
        ' if (v->scalars != nullptr) __builtin_abort(); return *static_cast<const unsigned long*>(v->residue); }',
        stub.replace("#include <cstdint>", ""),
        'extern "C" void fixture_integration_compute_0(const void* p, unsigned) { const auto* v = static_cast<const plk_role_view_fixture_integration_compute_0*>(p);'
        ' if (*static_cast<const unsigned long*>(v->residue) != 73 || pipelink_call_fixture_callee(p) != 91'
        ' || pipelink_call_fixture_another(p) != 37) __builtin_abort(); }',
        'int main() { PipelinkParams p{}; auto* raw = reinterpret_cast<unsigned char*>(&p);'
        f' *reinterpret_cast<unsigned long*>(raw + {residues["fixture.integration"]}) = 73;'
        f' *reinterpret_cast<unsigned long*>(raw + {residues["fixture.callee"]}) = 91;'
        f' *reinterpret_cast<unsigned long*>(raw + {residues["fixture.another"]}) = 37;'
        ' threadIdx.x = 0; pipelink_kernel(p); }',
    ])
    import subprocess
    import pipec2
    cpp, exe = tmp_path / "view.cpp", tmp_path / "view"
    cpp.write_text(header + source + body)
    include = Path(pipec2.__file__).parent / "pack/include"
    subprocess.run(["g++", "-std=c++17", "-I", str(include), str(cpp), "-o", str(exe)],
                   check=True, capture_output=True)
    subprocess.run([str(exe)], check=True)


@pytest.mark.parametrize("case", ["caller-reads-no-record", "export-without-view-parameter"])
def test_view_call_rejects(case):
    caller, callee = make_stage(), _roleless_member()
    if case == "caller-reads-no-record":
        host = deepcopy(caller.host)
        for key in ("residue", "scalars"):
            host[key]["read_roles"] = []
        caller = replace(caller, host=host)
    stages = [caller, callee]
    resolved = resolve_host(stages, make_solution(stages), make_pipeline(stages))
    pairs = _view_call_pairs(caller, [callee], first="int value" if case == "export-without-view-parameter" else "const void* view")
    if case == "export-without-view-parameter":
        from pipec2.decls import declaration_digest
        from pipec2.model import Export
        declaration = 'extern "C" __device__ unsigned long fixture_callee_get(int value);'
        pairs[0][2][0] = link_module._EndpointRecord(
            replace(callee, exports={"get": Export("fixture_callee_get", declaration, declaration_digest(declaration))}),
            pairs[0][2][0].endpoint)
    with pytest.raises(LinkError) as caught:
        link_module._view_calls(pairs, resolved)
    assert [failure.clause_id for failure in caught.value.failures] == ["L-PARAMS"]


def test_host_view_stage_rejects_params_accessor():
    from pipec2.model import SymbolInfo, SymbolTable
    from pipec2.symbols import check_params_accessors, params_accessor_symbol
    symbol = params_accessor_symbol("fixture.callee")
    stage = replace(_roleless_member(), symbols=SymbolTable((), (), (symbol,), {symbol: SymbolInfo("func", "strong")}))
    assert [failure.clause_id for failure in check_params_accessors([stage])] == ["L-PARAMS"]


def test_roleless_declaration_only_contract_links():
    declaration = make_stage({"target": {"dtypes": dtypes_digest("sm_103a")}}, "fixture.declarations", "")
    assert validate_host_assembly(make_pipeline([declaration]), [declaration]) == []
    host = resolve_host([declaration], make_solution([declaration]), make_pipeline([declaration]))
    assert host["groups"] == [] and host["role_views"] == {} and host["params"]["bytes"] == 0
    blob = replace(make_stage(), id="fixture.blob", host=None, params=ParamsDecl(8, 8, "Blob", "0" * 64))
    stages = [declaration, blob]
    assert validate_host_assembly(make_pipeline(stages), stages)[0].detail["subreason"] == "geometry"


def test_selected_variant_resolves_ring_depth_and_changes_selection_digest():
    from pipec2.model import Depth
    stages = _copy_accounting_stages(paired=True)
    host = deepcopy(stages[0].host)
    variants = [Variant({"pipelink_depth": depth}, depth - 1, None) for depth in (1, 2)]
    host["variants"] = {stage.id: [variant.to_json_obj() for variant in variants] for stage in stages}
    host["provider"]["owner_contract_digest"] = owner_contract_digest(host, host["variants"])
    stages = [replace(stage, host=host, variants=variants, endpoints=[
        replace(endpoint, ring=replace(endpoint.ring, depth=Depth(None, "pipelink_depth")))
        for endpoint in stage.endpoints
    ]) for stage in stages]
    pipeline = replace(make_pipeline(stages), handles={
        "pipelink_depth": Handle("pipelink_depth", "depth", "bind", None, [1, 2], "channel:activation"),
    })
    assert validate_host_assembly(pipeline, stages) == []
    linked = []
    for depth in (1, 2):
        solution = replace(make_solution(stages), depths={"pipelink_depth": depth},
                           variants={stage.id: {"pipelink_depth": depth} for stage in stages})
        result = resolve_host(stages, solution, pipeline)
        assert result["groups"][0]["transfers"][0]["endpoint"]["ring"]["depth"] == depth
        linked.append(result)
    assert linked[0]["selection_digest"] != linked[1]["selection_digest"]
    with pytest.raises(LinkError):
        resolve_host(stages, make_solution(stages), pipeline)


def test_channel_agreement_and_alias_cycles():
    first_host = make_host()
    first_host["channels"] = [make_channel()]
    refresh_host(first_host)
    other_host = make_host("fixture.other", scheduler=False)
    other_host["scheduler_contract"] = {"owner": first_host["owner"]}
    other_host["channels"] = [make_channel()]
    refresh_host(other_host)
    first = make_stage(first_host)
    stages = [first, make_stage(other_host, "fixture.other")]
    assert validate_host_assembly(make_pipeline(stages), stages) == []
    other_host["channels"][0]["rank"] = 2
    refresh_host(other_host)
    wrong = [first, make_stage(other_host, "fixture.other")]
    assert validate_host_assembly(make_pipeline(wrong), wrong)[0].detail["subreason"] == "rank"
    host = make_host()
    host["channels"] = [make_channel(), make_channel("fixture.alias", channel_id=12)]
    host["channels"][0]["alias"] = {"channel": "fixture.alias", "byte_offset": 0, "bit_offset": 0}
    host["channels"][1]["alias"] = {"channel": "fixture.tensor", "byte_offset": 0, "bit_offset": 0}
    refresh_host(host)
    stage = make_stage(host)
    assert validate_host_assembly(make_pipeline([stage]), [stage])[0].detail["subreason"] == "alias"


def _symmetric(host, *channels, domains=(("tp", 3),)):
    """Declare host domains and turn the named channels into (domain, views) channels."""
    host["domains"] = [{"id": i, "name": n} for n, i in domains]
    for channel, (domain, views) in channels:
        channel.update(domain=domain, views=views)
    return host


@pytest.mark.parametrize("change,path", [
    ("ok", None),
    ("undeclared-domain", "$.host"),
    ("views-without-domain", "$.host"),
    ("unsorted-views", "$.host"),
    ("workspace-source", "$.host"),
    ("duplicate-domain-name", "$.host.domains"),
])
def test_domain_channel_declarations(change, path):
    host = make_host()
    channel = make_channel()
    host["channels"] = [channel]
    domain, views = {"undeclared-domain": ("ep", ["peer"]), "views-without-domain": (None, ["peer"]),
                     "unsorted-views": ("tp", ["peer", "multicast"])}.get(change, ("tp", ["multicast", "peer"]))
    _symmetric(host, (channel, (domain, views)),
               domains=(("tp", 3), ("tp", 4)) if change == "duplicate-domain-name" else (("tp", 3),))
    if change == "workspace-source":
        channel["source"] = {"kind": "workspace", "name": "fixture.workspace"}
    refresh_host(host)
    if path is None:
        stage = make_stage(host)
        assert validate_host_assembly(make_pipeline([stage]), [stage]) == []
        return
    with pytest.raises(SchemaError) as caught:
        make_stage(host)
    failure = caught.value.failures[0]
    assert failure.clause_id == "L-HOSTCHAN" and failure.path == path
    assert failure.detail["subreason"] in ("domain", "inventory")


def test_channel_domain_agrees_across_owners_and_shared_caller_sources():
    first_host = make_host()
    first_host["channels"] = [make_channel()]
    _symmetric(first_host, (first_host["channels"][0], ("tp", ["peer"])))
    refresh_host(first_host)
    other_host = make_host("fixture.other", scheduler=False)
    other_host["scheduler_contract"] = {"owner": first_host["owner"]}
    other_host["channels"] = [make_channel()]
    _symmetric(other_host, (other_host["channels"][0], ("tp", ["multicast"])))
    refresh_host(other_host)
    stages = [make_stage(first_host), make_stage(other_host, "fixture.other")]
    [failure] = validate_host_assembly(make_pipeline(stages), stages)
    assert (failure.path, failure.detail["subreason"]) == ("host.channels.fixture.tensor.views", "domain")
    # Two channels over one caller source agree on the domain; their views may differ
    # (a multicast writer and a local mixed-view reader of the same allocation).
    for view_domain, accepted in (("tp", True), (None, False), ("ep", False)):
        host = make_host()
        host["channels"] = [make_channel(), make_channel("fixture.view", channel_id=12)]
        _symmetric(host, (host["channels"][0], ("tp", ["multicast"])), (host["channels"][1], (view_domain, [])),
                   domains=(("tp", 3), ("ep", 4)))
        refresh_host(host)
        stage = make_stage(host)
        failures = validate_host_assembly(make_pipeline([stage]), [stage])
        assert (failures == []) == accepted
        if not accepted:
            assert (failures[0].path, failures[0].detail["subreason"]) == ("host.channels.fixture.view.domain", "domain")
