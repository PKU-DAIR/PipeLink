"""Exercise the public annotation macros through compiled host emit programs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import pytest

from pipec2.host.abi import header_path


ROOT = Path(__file__).resolve().parents[3]
PACK = ROOT / "pipec/src/pipec2/pack"
FIXTURES = Path(__file__).parent / "fixtures/pack"
SCRATCH = Path(os.environ.get("PIPELINK_SCRATCH_ROOT", tempfile.gettempdir())) / "annotations-v2/phase3/StageHeader2"


@pytest.fixture(scope="module")
def workspace():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pytest-", dir=SCRATCH) as directory:
        path = Path(directory)
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", TMPDIR=str(path), TMP=str(path), TEMP=str(path))
        yield path, environment


def _compile(workspace, compiler, sources, name, *, defines=(), includes=(), emit=True, extra=()):
    path, environment = workspace
    if compiler == "nvcc":
        cuda = os.environ.get("PIPELINK_CUDA_HOME")
        if not cuda:
            pytest.skip("PIPELINK_CUDA_HOME is not set")
        command = [str(Path(cuda) / "bin/nvcc"), "-x", "cu", "--expt-relaxed-constexpr",
                   "-gencode", "arch=compute_103a,code=sm_103a"]
    else:
        if not shutil.which("g++"):
            pytest.skip("g++ is unavailable")
        command = ["g++"]
    binary = path / name
    command += ["-std=c++17", f"-I{PACK / 'include'}", f"-I{header_path().parent}"]
    if emit:
        command.append("-DPIPELINK_EMIT")
    command += [f"-D{define}" for define in defines]
    command += [f"-I{include}" for include in includes]
    command += list(extra)
    command += [str(source) for source in sources]
    command += ["-o", str(binary)]
    compiled = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=180)
    return binary, compiled


def _run(workspace, binary):
    return subprocess.run([str(binary)], env=workspace[1], capture_output=True, text=True, timeout=60)


@pytest.fixture(scope="module", params=["g++", "nvcc"])
def emitted(request, workspace):
    compiler = request.param
    sources = [FIXTURES / "header_contract.cpp", FIXTURES / "header_order.cpp", PACK / "runner.cpp"]
    binary, compiled = _compile(workspace, compiler, sources, f"contract-{compiler}")
    assert compiled.returncode == 0, compiled.stderr
    first = _run(workspace, binary)
    assert first.returncode == 0, first.stderr
    return compiler, sources, binary, first.stdout


def test_registry_is_deterministic_across_runs_and_registration_order(emitted, workspace):
    compiler, sources, binary, first = emitted
    second = _run(workspace, binary)
    assert second.returncode == 0, second.stderr
    assert second.stdout == first
    reversed_binary, compiled = _compile(workspace, compiler, list(reversed(sources)), f"reversed-{compiler}")
    assert compiled.returncode == 0, compiled.stderr
    reverse = _run(workspace, reversed_binary)
    assert reverse.returncode == 0, reverse.stderr
    assert reverse.stdout == first


@pytest.mark.parametrize("compiler", ["g++", "nvcc"])
def test_annotations_are_inert_without_emit(compiler, workspace):
    binary, compiled = _compile(workspace, compiler, [FIXTURES / "header_inert.cpp"], f"inert-{compiler}", emit=False)
    assert compiled.returncode == 0, compiled.stderr
    run = _run(workspace, binary)
    assert run.returncode == 0 and run.stdout == "" and run.stderr == ""


def test_unsupported_parameter_names_the_index(workspace):
    _, compiled = _compile(workspace, "g++", [FIXTURES / "header_unsupported.cpp", PACK / "runner.cpp"], "unsupported")
    assert compiled.returncode != 0
    assert "PipeLink unsupported parameter type" in compiled.stderr
    assert "Index = 1" in compiled.stderr
    assert "PL_TYPE" in compiled.stderr


def test_registered_type_members_must_restate_layout(workspace):
    _, compiled = _compile(
        workspace, "g++", [FIXTURES / "header_authoring.cpp", PACK / "runner.cpp"],
        "misstated-type-members", defines=["MISSTATED_TYPE_MEMBERS"],
    )
    assert compiled.returncode != 0
    assert "PL_TYPE members" in compiled.stderr


@pytest.mark.parametrize("foreign", [False, True], ids=["missing", "different-tu"])
def test_map_binding_must_be_declared_in_same_translation_unit(workspace, foreign):
    sources = [FIXTURES / "header_contract.cpp", PACK / "runner.cpp"]
    if foreign:
        sources.append(FIXTURES / "header_foreign_binding.cpp")
    binary, compiled = _compile(workspace, "g++", sources, f"missing-binding-{foreign}", defines=["OMIT_BINDING"])
    assert compiled.returncode == 0, compiled.stderr
    result = _run(workspace, binary)
    assert result.returncode == 1
    assert "P-MAP: accumulator.placement: undeclared binding 'tmem_base'" in result.stderr
    assert "PL_BINDING in the same translation unit" in result.stderr


def test_empty_participant_predicate_fails_actionably(workspace):
    binary, compiled = _compile(workspace, "g++", [FIXTURES / "header_empty_site.cpp", PACK / "runner.cpp"], "empty-site")
    assert compiled.returncode == 0, compiled.stderr
    result = _run(workspace, binary)
    assert result.returncode == 1
    assert "P-MAP: nobody.participants: participant predicate selects no threads" in result.stderr


@pytest.mark.parametrize("compiler", ["g++", "nvcc"])
def test_constexpr_maps_retain_captured_state(compiler, workspace):
    binary, compiled = _compile(workspace, compiler, [FIXTURES / "header_constexpr.cpp", PACK / "runner.cpp"], f"constexpr-{compiler}")
    assert compiled.returncode == 0, compiled.stderr
    result = _run(workspace, binary)
    assert result.returncode == 0, result.stderr
    records = json.loads(result.stdout)["records"]
    mapping = next(record for record in records if record["kind"] == "map")
    assert mapping["rows"] == [[m, base, m, base + 3] for m in range(-2, 2) for base in range(2)]
    params = next(record for record in records if record["kind"] == "params")
    assert params == {"kind": "params", "type": "Sixteen", "bytes": 64, "align": 4,
                      "members": [dict(name=name, offset=4 * index, bytes=4, align=4)
                                  for index, name in enumerate("abcdefghijklmnop")]}
    assert next(record for record in records if record["kind"] == "hint") == {
        "kind": "hint", "key": "register_budget", "value": 128}


@pytest.mark.parametrize(
    "define, diagnostic, compile_failure",
    [
        ("MISSING_PROTOCOL", "PL_ENDPOINT requires .protocol", True),
        ("EMPTY_REG_VALUES", "PL_CLAIM_REGS pipelink_regs: requires non-empty values", False),
        ("MIXED_BARRIER_SIDE", "slots/arrive and member_of are mutually exclusive; use pl::side or pl::member", False),
    ],
)
def test_invalid_extended_annotations_fail_actionably(workspace, define, diagnostic, compile_failure):
    binary, compiled = _compile(
        workspace, "g++", [FIXTURES / "header_extended_invalid.cpp", PACK / "runner.cpp"],
        f"invalid-{define}", defines=[define],
    )
    if compile_failure:
        assert compiled.returncode != 0
        assert diagnostic in compiled.stderr
    else:
        assert compiled.returncode == 0, compiled.stderr
        result = _run(workspace, binary)
        assert result.returncode == 1
        assert result.stdout == ""
        assert diagnostic in result.stderr


@pytest.mark.parametrize("defines", [
    (), ("TYPED_PARAMS",), ("WIDE_PARAMS",), ("OPAQUE_PARAMS",),
])
def test_authoring_gaps_emit_params_and_registered_types(workspace, defines):
    binary, compiled = _compile(
        workspace, "g++", [FIXTURES / "header_authoring.cpp", PACK / "runner.cpp"],
        "authoring-" + "-".join(defines), defines=defines,
    )
    assert compiled.returncode == 0, compiled.stderr
    result = _run(workspace, binary)
    assert result.returncode == 0, result.stderr
    records = json.loads(result.stdout)["records"]
    params = next(record for record in records if record["kind"] == "params")
    if "TYPED_PARAMS" in defines:
        assert params == {
            "kind": "params", "type": "Params", "bytes": 32, "align": 16,
            "members": [{"name": "first", "offset": 0, "bytes": 1, "align": 1},
                        {"name": "value", "offset": 16, "bytes": 4, "align": 16}],
        }
    elif "WIDE_PARAMS" in defines:
        names = [*"abcdefghijklmnopqrstuvwxyz", "aa", "ab", "ac", "ad", "ae", "af"]
        assert params == {"kind": "params", "type": "ThirtyTwo", "bytes": 128, "align": 4,
                          "members": [dict(name=name, offset=4 * index, bytes=4, align=4)
                                      for index, name in enumerate(names)]}
    elif "OPAQUE_PARAMS" in defines:
        assert params == {"kind": "params", "type": "Opaque", "bytes": 8, "align": 4, "members": [], "opaque": True}
    else:
        assert params == {"kind": "params", "type": "void", "bytes": 0, "align": 1, "members": []}
    assert [record for record in records if record["kind"] == "type"] == [
        {"kind": "type", "name": "Packet", "bytes": 16, "align": 16, "members": "alignas(16) unsigned values[4];"},
        {"kind": "type", "name": "sample::nested::Token", "bytes": 4, "align": 4, "members": "unsigned value;"},
    ]
    packet = '#ifndef PACKET_DEFINED\n#define PACKET_DEFINED\nstruct Packet { alignas(16) unsigned values[4]; };\n#endif\n'
    token = '#ifndef SAMPLE__NESTED__TOKEN_DEFINED\n#define SAMPLE__NESTED__TOKEN_DEFINED\nnamespace sample::nested { struct Token { unsigned value; }; }\n#endif\n'
    declaration = packet + token + (
        'extern "C" __device__ Packet plx_packet(const Packet&, Packet*, const Packet*, sample::nested::Token*);'
    )
    assert next(record for record in records if record["kind"] == "call")["declaration"] == declaration
    assert next(record for record in records if record.get("name") == "packet")["declaration"] == declaration
    imported = next(record for record in records if record["kind"] == "tensor_import")
    assert imported["declaration"] == packet + 'extern "C" __device__ void plx_pointer(Packet*);'
    assert imported["args"] == [{"name": "fragment", "meaning": "dest_fragment", "ctype": "Packet*"}]
    assert next(record for record in records if record["kind"] == "endpoint")["protocol"]["init"] == {
        "phase": 0, "materializer": "stage", "warp": 0,
    }


@pytest.mark.parametrize("defines, diagnostic", [
    (("DUPLICATE_PARAMS",), "declare exactly one"),
    (("TYPED_PARAMS", "DUPLICATE_PARAMS"), "declare exactly one"),
    (("OMIT_INIT_WARP",), "init requires owner or warp"),
])
def test_authoring_gaps_reject_conflicting_or_missing_annotations(workspace, defines, diagnostic):
    binary, compiled = _compile(
        workspace, "g++", [FIXTURES / "header_authoring.cpp", PACK / "runner.cpp"],
        "invalid-authoring-" + "-".join(defines), defines=defines,
    )
    assert compiled.returncode == 0, compiled.stderr
    result = _run(workspace, binary)
    assert result.returncode == 1
    assert diagnostic in result.stderr


@pytest.fixture(scope="module")
def host_workspace():
    scratch = Path(os.environ.get("PIPELINK_SCRATCH_ROOT", tempfile.gettempdir())) / "annotations-v2/phase3/host-annotations"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pytest-", dir=scratch) as directory:
        path = Path(directory)
        yield path, dict(os.environ, CUDA_VISIBLE_DEVICES="", TMPDIR=str(path), TMP=str(path), TEMP=str(path))


def _host_emit(workspace, compiler, defines=(), *, emit=True):
    sources = [FIXTURES / "header_host.cpp"]
    if emit:
        sources.append(PACK / "runner.cpp")
    binary, compiled = _compile(
        workspace, compiler, sources, f"host-{compiler}-{'-'.join(defines)}",
        defines=defines, emit=emit,
    )
    assert compiled.returncode == 0, compiled.stderr
    result = _run(workspace, binary)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout) if emit else result.stdout


@pytest.fixture(scope="module", params=["g++", "nvcc"])
def host_registry(request, host_workspace):
    return _host_emit(host_workspace, request.param)


def test_host_group_reflects_neutral_and_owner_layouts(host_registry):
    from pipec2.host.abi import layout_digest, probe_layouts
    from pipec2.pack.assemble import assemble
    from pipec2.schema import load_stage_manifest

    host = load_stage_manifest(assemble(host_registry)).host
    rows = host["provider"]["abi_types"]
    inventory = {row["name"]: row for row in rows}
    neutral = probe_layouts()
    assert set(inventory) == set(neutral) | {"HostArgs", "HostResidue", "HostEmpty"}
    assert len(rows) == len(inventory) == len(neutral) + 3
    assert [row["id"] for row in rows] == sorted(row["id"] for row in rows)
    for name, layout in neutral.items():
        assert inventory[name] == {"name": name, **{
            key: layout[key] for key in ("id", "bytes", "align", "layout_digest", "fields")}}

    for name, type_id, size, align, members in [
        ("HostArgs", 1, 16, 8, {"count": {"offset": 8, "bytes": 8}, "tag": {"offset": 0, "bytes": 1}}),
        ("HostResidue", 2, 8, 8, {"state": {"offset": 0, "bytes": 8}}),
        ("HostEmpty", 3, 1, 1, {}),
    ]:
        row = inventory[name]
        assert (row["id"], row["bytes"], row["align"]) == (type_id, size, align)
        assert set(row["fields"]) == set(members)
        assert row["layout_digest"] == layout_digest(name, size, align, members)
    args = host["provider"]["args"]
    residue = host["residue"]
    assert args == {"type_id": inventory["HostArgs"]["id"], "bytes": 16, "align": 8}
    assert (residue["type_id"], residue["bytes"], residue["align"], residue["layout_digest"]) == (
        inventory["HostResidue"]["id"], 8, 8, inventory["HostResidue"]["layout_digest"])


@pytest.mark.parametrize("compiler", ["g++", "nvcc"])
def test_host_args_can_reference_symbolic_neutral_type(host_workspace, compiler):
    from pipec2.host.abi import probe_layouts
    from pipec2.pack.assemble import assemble
    from pipec2.schema import load_stage_manifest

    host = load_stage_manifest(assemble(_host_emit(host_workspace, compiler, ("HOST_NEUTRAL_ARGS",)))).host
    layout = probe_layouts()["plk_header_v2"]
    args = host["provider"]["args"]
    assert args == {"type_id": layout["id"], "bytes": layout["bytes"], "align": layout["align"]}
    referenced = [row for row in host["provider"]["abi_types"] if row["id"] == args["type_id"]]
    assert len(referenced) == 1
    assert referenced[0]["name"] == "plk_header_v2"
    assert referenced[0]["layout_digest"] == layout["layout_digest"]


@pytest.mark.parametrize("define", [
    "HOST_REDECLARE_ID=99",
    "HOST_REDECLARE_ID=PLK_ABI_TYPE_ID(plk_header_v2)",
    "HOST_EMPTY_ID=0x70000000",
    "HOST_EMPTY_ID=PLK_ABI_TYPE_ID(plk_header_v2)",
    "HOST_EMPTY_ID=0x700000ff",
], ids=["explicit-neutral-local-id", "explicit-neutral-canonical-id",
        "reserved-start", "neutral-id-collision", "reserved-end"])
def test_owner_annotations_cannot_redeclare_neutral_inventory(host_workspace, define):
    from pipec2.model import PackError
    from pipec2.pack.assemble import assemble

    binary, compiled = _compile(
        host_workspace, "g++", [FIXTURES / "header_host.cpp", PACK / "runner.cpp"],
        "host-invalid-" + define.split("=")[0] + "-" + str(len(define)), defines=[define],
    )
    assert compiled.returncode == 0, compiled.stderr
    result = _run(host_workspace, binary)
    if define == "HOST_REDECLARE_ID=99":
        assert result.returncode == 0, result.stderr
        with pytest.raises(PackError) as caught:
            assemble(json.loads(result.stdout))
        assert caught.value.failures[0].clause_id == "P-MAP"
    else:
        assert result.returncode != 0


def test_host_annotations_stamp_the_packed_profile(host_registry):
    from pipec2.host.profile import dtypes_digest
    from pipec2.pack.assemble import assemble
    from pipec2.schema import load_stage_manifest
    actual = assemble(host_registry, profile="sm_90a")
    assert actual["pipelink_abi"]["target"]["profile"] == "sm_90a"
    assert actual["host"]["target"] == {"dtypes": dtypes_digest("sm_90a")}
    assert load_stage_manifest(actual).host == actual["host"]


def test_host_annotations_preserve_distinct_resources(host_workspace):
    from pipec2.pack.assemble import assemble
    from pipec2.schema import load_stage_manifest
    host = load_stage_manifest(assemble(_host_emit(host_workspace, "g++", ("HOST_MULTIPLE_RESOURCES",)))).host
    assert [channel["name"] for channel in host["channels"]] == ["fixture.tensor", "fixture.other"]
    assert [(channel["domain"], channel["views"]) for channel in host["channels"]] == [
        (None, []), ("tp", ["multicast", "peer"])]
    assert host["domains"] == [{"id": 2, "name": "ep"}, {"id": 4, "name": "tp"}]
    assert host["channels"][1]["logical_dtype"] == "u8"
    assert [workspace["name"] for workspace in host["workspaces"]] == ["fixture.workspace", "fixture.zero"]
    assert host["workspaces"][1]["reset"] == "launch"
    assert [transfer["channel"] for transfer in host["transfers"]] == ["fixture.tensor", "fixture.other"]


@pytest.mark.parametrize("define", ["HOST_HOSTLESS", "HOST_TARGET_ONLY", "HOST_DECLARATIONS_ONLY"])
def test_providerless_annotations_have_no_abi_inventory(host_workspace, define):
    from pipec2.host.profile import dtypes_digest
    from pipec2.pack.assemble import assemble

    registry = _host_emit(host_workspace, "g++", (define,))
    assert not any(row["kind"] == "host_abi_type" for row in registry["records"])
    actual = assemble(registry)
    if define == "HOST_HOSTLESS":
        assert "host" not in actual
    elif define == "HOST_TARGET_ONLY":
        assert actual["host"] == {"target": {"dtypes": dtypes_digest("sm_103a")}}
    else:
        host = actual["host"]
        assert host["provider"] is None
        assert host["params_group"] is None
        assert host["owner"] == "fixture.author"
        assert [(fact["name"], fact["value"]) for fact in host["published_facts"]] == [("fixture.width", 16)]


def test_host_annotations_zero_allocations_keep_reflected_type_identity(host_workspace):
    from pipec2.pack.assemble import assemble
    from pipec2.schema import load_stage_manifest

    host = load_stage_manifest(assemble(_host_emit(host_workspace, "g++", ("HOST_ZERO_STATE",)))).host
    inventory = {row["id"]: row for row in host["provider"]["abi_types"]}
    args, residue = host["provider"]["args"], host["residue"]
    assert args["bytes"] == residue["bytes"] == 0
    assert (args["type_id"], args["align"]) == (1, 8)
    assert (residue["type_id"], residue["align"]) == (2, 8)
    assert (inventory[args["type_id"]]["name"], inventory[args["type_id"]]["bytes"]) == ("HostArgs", 16)
    assert (inventory[residue["type_id"]]["name"], inventory[residue["type_id"]]["bytes"]) == ("HostResidue", 8)
    assert residue["layout_digest"] == inventory[residue["type_id"]]["layout_digest"]


@pytest.mark.parametrize("mutation", ["abi_size", "derived_digest", "variants", "storage_bits", "foreign_channel"])
def test_host_annotation_conflicts_are_pack_failures(host_registry, mutation):
    from copy import deepcopy
    from pipec2.model import PackError
    from pipec2.pack.assemble import assemble
    registry = deepcopy(host_registry)
    if mutation == "abi_size":
        next(row for row in registry["records"] if row["kind"] == "host_abi_type" and row["id"] == 1)["bytes"] += 8
    elif mutation == "derived_digest":
        row = next(row for row in registry["records"] if row["kind"] == "host_channel")
        row["declaration"] = json.dumps({**json.loads(row["declaration"]), "dtype_id": 1})
    elif mutation == "storage_bits":
        next(row for row in registry["records"] if row["kind"] == "host_channel")["storage_bits"] = 16
    elif mutation == "foreign_channel":
        row = next(row for row in registry["records"] if row["kind"] == "host_transfer")
        row["declaration"] = json.dumps({**json.loads(row["declaration"]), "channel": "foreign.channel"})
    else:
        registry["records"].append({"kind": "host_variants", "name": "fixture.host",
                                    "declaration": '[{"index":0,"key":{}}]'})
    with pytest.raises(PackError) as caught:
        assemble(registry)
    assert caught.value.failures[0].clause_id == "P-HOST"


@pytest.mark.parametrize("compiler", ["g++", "nvcc"])
def test_host_annotations_are_inert(host_workspace, compiler):
    assert _host_emit(host_workspace, compiler, ("HOST_INERT",), emit=False) == ""


@pytest.mark.parametrize("compiler", ["g++", "nvcc"])
def test_native_and_one_shot_scheduler_annotations_roundtrip(workspace, compiler):
    from pipec2.pack.assemble import assemble
    from pipec2.schema import load_stage_manifest
    source = workspace[0] / f"native-{compiler}.cpp"
    source.write_text(r'''
#include <pipelink/stage.hpp>
struct Params { int value; };
extern "C" void fixture_native_role_load(const void*, unsigned) {}
PL_STAGE("fixture.native", "1");
PL_PARAMS(Params, value);
PL_ROLE(load, fixture_native_role_load, 1);
PL_SCHEDULER_STRATEGY("clc-sm100-v1",
    R"(["swizzle_size","raster_order","cluster_shape_m","cluster_shape_n"])");
PL_CLAIM("pipelink_full", "mbarrier", 1, 1, "kernel");
PL_CLAIM("pipelink_empty", "mbarrier", 1, 1, "kernel");
PL_CLAIM("pipelink_storage", "smem", 16, 8, "kernel");
constexpr auto native_side = pl::side{"pipelink_full", {"load", "threads(load)"}, 0}
    .native(pl::barrier_storage{"pipelink_storage", 8, 8});
PL_ENDPOINT(signal, pl::endpoint("signal", pl::produce, "load")
    .space("smem").scope("cta").proxy("async").tile("u64", {1})
    .layout(pl::affine{{1}, {1}, 64})
    .protocol(pl::mbarrier{native_side, std::nullopt, {"load", 0, "stage"}, "never"})
    .ring(pl::fixed(1), 8).bind("full", "pipelink_full"));
PL_ENDPOINT(ring, pl::endpoint("ring", pl::produce, "load")
    .space("smem").scope("cta").proxy("async").tile("u64", {1})
    .layout(pl::affine{{1}, {1}, 64})
    .protocol(pl::mbarrier{native_side, {"pipelink_empty", {"load", 1}, 0}, {"load", 0, "stage"}})
    .ring(pl::fixed(1), 8).bind("full", "pipelink_full").bind("empty", "pipelink_empty"));
''')
    binary, compiled = _compile(workspace, compiler, [source, PACK / "runner.cpp"], f"native-{compiler}")
    assert compiled.returncode == 0, compiled.stderr
    emitted = _run(workspace, binary)
    assert emitted.returncode == 0, emitted.stderr
    manifest = assemble(json.loads(emitted.stdout))
    stage = load_stage_manifest(manifest)
    assert stage.topology.scheduler_strategy.id == "clc-sm100-v1"
    endpoint = next(item for item in stage.endpoints if item.name == "signal")
    assert endpoint.protocol.release == "never" and endpoint.protocol.empty is None
    assert endpoint.protocol.full.native.to_json_obj() == {
        "base": "pipelink_storage", "offset_bytes": 8, "stride_bytes": 8,
    }


@pytest.mark.parametrize("compiler", ["g++", "nvcc"])
def test_signal_endpoints_roundtrip_without_data_layout(workspace, compiler):
    from pipec2.pack.assemble import assemble
    from pipec2.schema import load_stage_manifest

    source = workspace[0] / f"signal-{compiler}.cpp"
    source.write_text(r'''
#include <pipelink/stage.hpp>
struct Params { int value; };
extern "C" void fixture_signal_role_work(const void*, unsigned) {}
PL_STAGE("fixture.signal", "1");
PL_PARAMS(Params, value);
PL_ROLE(work, fixture_signal_role_work, 1);
PL_CLAIM("pipelink_once_full", "mbarrier", 1, 1, "kernel");
PL_CLAIM("pipelink_reusable_full", "mbarrier", 2, 1, "kernel");
PL_ENDPOINT(once, pl::signal_endpoint("once", pl::produce, "work")
    .protocol(pl::mbarrier{pl::side{"pipelink_once_full", {"work", 1}, 0},
                          std::nullopt, {"work", 0, "dispatch"}, "never"})
    .bind("full", "pipelink_once_full"));
PL_ENDPOINT(reusable, pl::signal_endpoint("reusable", pl::consume, "work")
    .scope("cluster")
    .protocol(pl::mbarrier{pl::side{"pipelink_reusable_full", {"work", 1}, 0},
                          std::nullopt, {"work", 0, "dispatch"}, "external"})
    .ring(pl::fixed(2), 0).bind("full", "pipelink_reusable_full"));
''')
    binary, compiled = _compile(workspace, compiler, [source, PACK / "runner.cpp"], f"signal-{compiler}")
    assert compiled.returncode == 0, compiled.stderr
    emitted = _run(workspace, binary)
    assert emitted.returncode == 0, emitted.stderr
    registry = json.loads(emitted.stdout)
    records = {item["name"]: item for item in registry["records"] if item["kind"] == "endpoint"}
    manifest = assemble(registry)
    stage = load_stage_manifest(manifest)
    endpoints = {item.name: item for item in stage.endpoints}
    for name, direction, depth, scope, release in [
        ("once", "produce", 1, "cta", "never"),
        ("reusable", "consume", 2, "cluster", "external"),
    ]:
        record = records[name]
        assert "tile" not in record and "layout" not in record
        assert record["space"] == "none" and record["proxy"] == "generic"
        assert record["ring"] == {"depth": {"fixed": depth}, "stage_stride_bytes": 0}
        assert record["binds"] == {"full": f"pipelink_{name}_full"}
        authored = next(item for item in manifest["endpoints"] if item["name"] == name)
        assert "tile" not in authored and "layout" not in authored
        endpoint = endpoints[name]
        assert endpoint.space == "none"
        assert endpoint.direction == direction and endpoint.scope == scope
        assert endpoint.tile.dtype == "none" and endpoint.tile.shape == ()
        assert endpoint.layout.space == "none"
        assert endpoint.ring.depth.fixed == depth and endpoint.ring.stage_stride_bytes == 0
        assert endpoint.protocol.empty is None and endpoint.protocol.release == release
        assert type(endpoint.protocol.full.tx_bytes) is int and endpoint.protocol.full.tx_bytes == 0


@pytest.mark.parametrize("compiler", ["g++", "nvcc"])
def test_counter_builders_roundtrip_grid_and_system_scope(workspace, compiler):
    from pipec2.pack.assemble import assemble
    from pipec2.schema import load_stage_manifest

    source = workspace[0] / f"counters-{compiler}.cpp"
    source.write_text(r'''
#include <pipelink/stage.hpp>
struct Params { int value; };
extern "C" void fixture_counters_role_work(const void*, unsigned) {}
PL_STAGE("fixture.counters", "1");
PL_PARAMS(Params, value);
PL_ROLE(work, fixture_counters_role_work, 1);
PL_CLAIM("pipelink_local", "gmem_counter", 1, 1, "kernel");
PL_CLAIM("pipelink_peer", "gmem_counter", 1, 1, "kernel");
PL_ENDPOINT(local, pl::endpoint("local", pl::produce, "work")
    .space("gmem").scope("grid").proxy("generic").tile("u32", {1})
    .layout(pl::affine{{1}, {1}, 32}).protocol(pl::counter{"pipelink_local"}));
PL_ENDPOINT(peer, pl::endpoint("peer", pl::consume, "work")
    .space("gmem").scope("system").proxy("generic").tile("u32", {1})
    .layout(pl::affine{{1}, {1}, 32}).protocol(pl::counter_sys{"pipelink_peer"}));
PL_ENDPOINT(callable, pl::endpoint("callable", pl::consume)
    .space("gmem").scope("grid").proxy("generic").tile("u32", {1})
    .layout(pl::affine{{1}, {1}, 32})
    .protocol(pl::counter{"pipelink_local", "red.release.gpu", "ld.relaxed.gpu+fence.acquire.gpu"}));
''')
    binary, compiled = _compile(workspace, compiler, [source, PACK / "runner.cpp"], f"counters-{compiler}")
    assert compiled.returncode == 0, compiled.stderr
    emitted = _run(workspace, binary)
    assert emitted.returncode == 0, emitted.stderr
    stage = load_stage_manifest(assemble(json.loads(emitted.stdout)))
    endpoints = {item.name: item for item in stage.endpoints}
    assert [(endpoints[name].scope, endpoints[name].protocol.counter.to_json_obj()) for name in ("local", "peer")] == [
        ("grid", {"slot_handle": "pipelink_local", "release_op": "red.release.gpu", "acquire_op": "ld.acquire.gpu"}),
        ("system", {"slot_handle": "pipelink_peer", "release_op": "red.release.sys", "acquire_op": "ld.acquire.sys"}),
    ]
    assert (endpoints["callable"].owner_role, endpoints["callable"].protocol.counter.acquire_op) == (
        None, "ld.relaxed.gpu+fence.acquire.gpu")


@pytest.mark.parametrize("compiler", ["g++", "nvcc"])
@pytest.mark.parametrize("builder", [
    'pl::endpoint("signal", pl::produce, "work").space("smem").scope("cta").proxy("generic")'
    '.tile("u64", {1}).protocol(barrier)',
    'pl::signal_endpoint("signal", pl::produce, "work").tile("u64", {1}).protocol(barrier)',
    'pl::signal_endpoint("signal", pl::produce, "work").layout(pl::affine{{1}, {1}, 64}).protocol(barrier)',
    'pl::signal_endpoint("signal", pl::produce, "work").protocol(pl::named_barrier{"pipelink_named", 32})',
    'pl::signal_endpoint("signal", pl::produce, "work").protocol(pl::counter{"pipelink_counter"})',
    'pl::signal_endpoint("signal", pl::produce, "work").protocol(pl::counter_sys{"pipelink_counter"})',
], ids=["ordinary-missing-layout", "signal-tile", "signal-layout", "signal-named-barrier", "signal-counter",
        "signal-counter-sys"])
def test_endpoint_builders_reject_incompatible_layout_and_protocol(workspace, compiler, builder):
    source = workspace[0] / f"invalid-signal-{compiler}.cpp"
    source.write_text(r'''
#include <pipelink/stage.hpp>
constexpr auto barrier = pl::mbarrier{pl::side{"pipelink_full", {"work", 1}, 0},
                                     std::nullopt, {"work", 0, "dispatch"}, "never"};
PL_ENDPOINT(signal, ''' + builder + r'''.bind("full", "pipelink_full"));
''')
    _, compiled = _compile(workspace, compiler, [source, PACK / "runner.cpp"], f"invalid-signal-{compiler}")
    assert compiled.returncode != 0
