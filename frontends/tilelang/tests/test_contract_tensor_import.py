"""CPU-only two-provider tensor contracts and explicit TileLang lowering."""
from dataclasses import replace
import importlib.util
import json
import subprocess
import zipfile
from pathlib import Path

import pytest
from tilelang import tvm

from pipec2.decls import declaration_digest
from pipelink_tilelang import set_context
from pipec2.pack.assemble import assemble
from pipec2.profile import get_profile
from pipec2.schema import load_stage_manifest
from pipelink_tilelang.annotate import Ledger
from pipelink_tilelang.capture import capture
from pipelink_tilelang.contract import Partner, consume
from pipelink_tilelang.explicit import Export, _annotations, emit

ROOT = Path(__file__).resolve().parents[3]
RECIPE = ROOT / "examples/one-epilogue-many-producers/consumer-tilelang/epilogue_contract.py"
PACK = ROOT / "pipec/src/pipec2/pack"


def _partner(directory, label, export, coords, argument_type, members, profile="sm_103a", channel_prefix=""):
    prelude = (f"#ifndef {argument_type.upper()}_DEFINED\n#define {argument_type.upper()}_DEFINED\n"
               f"struct {argument_type} {{ {members} }};\n#endif\n")
    exports, endpoints = {}, []
    for channel, endpoint, result, parameters in (
        ("produce_channel", "acquire", "pipelink_tensor_source", f"const {argument_type}*, void*"),
        ("release_channel", "retire", "void", f"const {argument_type}*"),
        # Identical ABI, different semantic endpoint: selection must use export.calls.
        ("other_produce", "other_acquire", "pipelink_tensor_source", f"const {argument_type}*, void*"),
    ):
        symbol = f"test_{label}_{endpoint}"
        decl = prelude + f'extern "C" __device__ {result} {symbol}({parameters});'
        exports[endpoint] = dict(symbol=symbol, decl=decl, decl_sha256=declaration_digest(decl))
        endpoints.append(dict(name=channel_prefix + channel, direction="consume", owner_role=None, space="reg",
                              scope="cta", proxy="generic", tile=dict(dtype="u32", shape=[1]),
                              layout=dict(encoding="pl-layout-v2", elem_bits=32, shape=[1], stride=[1]),
                              protocol=dict(kind="call", export=endpoint, signature=f"{result}({parameters})",
                                            decl=decl, decl_sha256=declaration_digest(decl)), binds={}))
    identity = dict(result=f"{label}_result", work="tensor_work", version="completed", dtype="f32",
                    interpretation=f"{label}_numerics",
                    coords=[dict(name=name, sort="int", lo=0, hi=extent) for name, extent in coords])
    raw = dict(
        pipelink_abi=dict(schema="3.0", target=dict(profile=profile, lto_arch=get_profile(profile).lto_arch, toolkit="cuda-13.3.0")),
        stage=dict(id=f"test.{label}", version="1.0.0", symbol_prefix=f"test_{label}_", entry_abi="v2",
                   roles={}, exports=exports,
                   entry_signature="void(const void* params, unsigned variant)", variants=[dict(key={}, index=0)]),
        endpoints=endpoints, claims=[],
        topology=dict(cta_group=dict(fixed=1), cluster=dict(compiled=[1, 1, 1], launch=dict(x_multiple_of=1, max=[1, 1, 1])),
                      multicast=[], persistent=False, scheduler="static"),
        params=dict(bytes=0, align=1, type="void", sha256_of_layout_probe="0" * 64),
        payload=dict(container_owned=True),
        tensor=dict(version="tensor-v2", exports=[dict(
            name=export, identity=identity,
            placements=[dict(guard=None, space="smem", units="byte", components={"address": ["mulc", 4, ["coord", coords[0][0]]]})],
            read_capabilities=["smem_load_scalar"], readiness=dict(after=f"{label}_ready", reuse_after=f"{label}_done"),
            fixed_bindings={}, backing=[], calls=dict(produce=channel_prefix + "produce_channel", release=channel_prefix + "release_channel"))],
            imports=[], sites=[], bindings=[]),
    )
    path = directory / f"{label}.plink"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(raw))
    return Partner.load(path)


def _recipe(directory, monkeypatch, profile, collision=True):
    acc = _partner(directory, "acc", "accumulator", [("m", 128), ("n", 128)], "pl_gemm_tile_args",
                   "const void* a; const void* b; int32_t M; int32_t N; int32_t K;", profile)
    scale = _partner(directory, "scale", "row_scale", [("m", 128)], "pl_rowscale_args",
                     "const void* a; int32_t M; int32_t K; float eps;", profile, "" if collision else "scale_")
    monkeypatch.setenv("PIPELINK_ACC_PARTNER", str(acc.path))
    monkeypatch.setenv("PIPELINK_SCALE_PARTNER", str(scale.path))
    spec = importlib.util.spec_from_file_location("tensor_contract_recipe", RECIPE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _registry(directory, module, prelude=""):
    annotations, _ = _annotations(module.STAGE, {}, Ledger())
    params = module.STAGE.params
    fields = " ".join(f"{field.cpp_type} {field.name};" for field in params.fields)
    source = directory / "annotations.cpp"
    source.write_text('#include <cstdint>\n#include <pipelink/stage.hpp>\n'
                      f"struct alignas({params.alignas}) {params.name} {{ {fields} }};\n"
                      + prelude
                      + "\n".join(f'extern "C" __device__ {export.signature};' for export in module.STAGE.exports)
                      + "\n".join(f'extern "C" __device__ void tensor_consumer_tilelang_epilogue_role_{role}(const void*, unsigned);'
                                  for role in module.STAGE.roles)
                      + "\n" + annotations)
    binary = directory / "annotations"
    compiled = subprocess.run(["g++", "-std=c++17", "-DPIPELINK_EMIT", "-D__device__=", f"-I{PACK / 'include'}",
                               f"-I{PACK.parent / 'host/include'}",
                               str(source), str(PACK / "runner.cpp"), "-o", str(binary)], capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)["records"]


@pytest.mark.parametrize("profile, collision, role", (("sm_103a", True, "epilogue"), ("sm_90a", False, "local_worker"),
                                                      ("sm_120a", True, "epilogue")))
def test_contract_two_tensor_partners_lower_and_preserve_semantics(tmp_path, monkeypatch, profile, collision, role):
    import pipelink_tilelang
    monkeypatch.setattr(pipelink_tilelang, "context", pipelink_tilelang.context)
    set_context(profile, None)
    module = _recipe(tmp_path, monkeypatch, profile, collision)
    module.STAGE = replace(module.STAGE, roles={role: next(iter(module.STAGE.roles.values()))})
    for channel in module.STAGE.channels:
        channel.role = role
    captured = capture(module.program(), target=tvm.target.Target({"kind": "cuda", "arch": profile}),
                       pass_configs=module.PASS_CONFIGS)
    generation = dict(source=captured.kernel_source, facts=captured.facts, passes=captured.passes,
                      snapshots=captured.snapshots, modules=captured.modules, layout_objects=captured.layout_objects)
    emit(tmp_path, module, {None: generation})
    records = _registry(tmp_path, module)
    manifest = assemble({"records": records}, profile=profile)
    load_stage_manifest(manifest)
    imports = {item["name"]: item for item in manifest["tensor"]["imports"]}
    for channel in module.STAGE.channels:
        row = imports[channel.name]
        assert row["identity"] == channel.export["identity"]
        assert row["completion"] == channel.done
        assert {item["event"] for item in row["rendezvous"]} == {channel.ready, channel.done}
        barriers = {lend["name"] for lend in row["lends"] if lend["domain"] == "barrier"}
        assert all(item["barrier"] in barriers and item["cohort"]["set"] == list(range(128)) for item in row["rendezvous"])
        assert row["callable"]["events"] == dict(ready=channel.symbol + "_ready", done=channel.symbol + "_done")
        assert row["callable"]["params"][0]["ctype"] == "pipelink_tensor_source"
    assert imports[module.acc.name]["demands"][0]["identity_coords"] == {"m": ["coord", "t"], "n": ["coord", "p"]}
    assert imports[module.scale.name]["demands"][0]["position"] == ["const", 0]
    assert sum(lend["binding"] == "caller_frame" for lend in imports[module.acc.name]["lends"]) == 1
    calls = {item["name"]: item for item in records if item["kind"] == "call"}
    assert {item["export"] for item in calls.values()} == {"acquire", "retire"}
    assert len(calls) == 4
    for channel in module.STAGE.channels:
        for endpoint in (channel.produce_endpoint, channel.release_endpoint):
            local = endpoint.name + "_" + channel.symbol if collision else endpoint.name
            assert calls[local]["symbol"] == "pipelink_call_" + local
            assert endpoint.owner_role is None
            assert calls[local]["role"] == channel.role
            assert next(item for item in manifest["endpoints"] if item["name"] == local)["owner_role"] == channel.role
    sites = [item for item in records if item["kind"] == "site"]
    assert sites[0]["observes"] == ["acc_ready", "scale_ready"]


def test_explicit_callee_does_not_inherit_caller_role(tmp_path, monkeypatch):
    import pipelink_tilelang
    monkeypatch.setattr(pipelink_tilelang, "context", pipelink_tilelang.context)
    set_context("sm_103a", None)
    module = _recipe(tmp_path, monkeypatch, "sm_103a", collision=False)
    caller_role = "independent_caller"
    module.STAGE = replace(module.STAGE, roles={caller_role: next(iter(module.STAGE.roles.values()))})
    for channel in module.STAGE.channels:
        channel.role = caller_role
    caller_manifest = assemble({"records": _registry(tmp_path, module)}, profile="sm_103a")
    caller_path = tmp_path / "caller.plink"
    with zipfile.ZipFile(caller_path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(caller_manifest))
    caller = Partner.load(caller_path)
    endpoint_name = module.scale.release_name
    channel = consume(caller, endpoint_name, role="provider_local")
    exported = Export(channel.protocol.call.export, "provided_retire",
                      f"void provided_retire(const {module.scale.argument_type}*)", "")
    module.STAGE = replace(module.STAGE, roles={}, channels=[channel], exports=(exported,))
    provider_dir = tmp_path / "provider"
    provider_dir.mkdir()
    prelude = channel.protocol.call.decl + f"\nPL_TYPE({module.scale.argument_type}, {module.scale.type_members});\n"
    records = _registry(provider_dir, module, prelude)
    provider = assemble({"records": records}, profile="sm_103a")
    assert caller.endpoints[endpoint_name].owner_role == caller_role
    assert [(endpoint["name"], endpoint["direction"], endpoint["owner_role"], endpoint["protocol"]["export"])
            for endpoint in provider["endpoints"]] == [(endpoint_name, "consume", None, "retire")]
    assert provider["stage"]["roles"] == {}
    assert provider["claims"] == []
