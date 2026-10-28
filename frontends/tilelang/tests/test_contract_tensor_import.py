"""CPU-only scoped tensor contracts, TileLang lowering, and staging reuse."""
from dataclasses import replace
import importlib.util
import itertools
import json
import subprocess
import re
import zipfile
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from tilelang import tvm
import tilelang.language as T

import pipelink_tilelang as pl

from pipec2.decls import declaration_digest
from pipec2.tensor import expr
from pipelink_tilelang import set_context
from pipec2.pack.assemble import assemble
from pipec2.profile import get_profile
from pipec2.schema import load_stage_manifest
from pipelink_tilelang.annotate import AnnotationError, Ledger
from pipelink_tilelang.capture import capture
from pipelink_tilelang.contract import Partner, consume
from pipelink_tilelang.explicit import Export, _annotations, emit, symbol_prefix

ROOT = Path(__file__).resolve().parents[3]
RECIPE = Path(__file__).resolve().parent / "fixtures/two_import_epilogue.py"
PACK = ROOT / "pipec/src/pipec2/pack"


SMEM_ROWS = dict(guard=None, space="smem", units="byte", components={"address": ["mulc", 4, ["coord", "m"]]})
# Hopper WGMMA m64n128 accumulator registers for a 128x128 tile, as a CuTe provider publishes them.
WGMMA_REGISTERS = dict(guard=None, space="reg", units="scalar", components={
    "participant": ["add", ["add", ["modc", ["divc", ["coord", "n"], 2], 4], ["mulc", 4, ["modc", ["coord", "m"], 8]]],
                    ["mulc", 32, ["modc", ["divc", ["coord", "m"], 16], 4]]],
    "position": ["add", ["add", ["add", ["modc", ["coord", "n"], 2], ["mulc", 2, ["modc", ["divc", ["coord", "m"], 8], 2]]],
                         ["mulc", 4, ["divc", ["coord", "n"], 8]]], ["mulc", 64, ["divc", ["coord", "m"], 64]]]})
# CuTe mma.sync placement published by the sm120 register provider.
MMA_REGISTERS = dict(guard=None, space="reg", units="scalar", components={
    "participant": ["add", ["add", ["add",
        ["modc", ["divc", ["coord", "n"], 2], 4], ["mulc", 4, ["modc", ["coord", "m"], 8]]],
        ["mulc", 32, ["modc", ["divc", ["coord", "m"], 16], 2]]],
        ["mulc", 64, ["modc", ["divc", ["coord", "n"], 8], 2]]],
    "position": ["add", ["add", ["add", ["modc", ["coord", "n"], 2],
        ["mulc", 2, ["modc", ["divc", ["coord", "m"], 8], 2]]],
        ["mulc", 4, ["divc", ["coord", "m"], 32]]],
        ["mulc", 16, ["divc", ["coord", "n"], 16]]]})
ROW_REGISTERS = dict(guard=None, space="reg", units="scalar",
                     components={"participant": ["coord", "m"], "position": ["coord", "n"]})
TMEM_ROWS = dict(guard=None, space="tmem", units="dp/column",
                 components={"dp": ["coord", "m"], "column": ["add", ["coord", "n"], ["bind", "base"]]})


_SIZES = {"const void*": 8, "int32_t": 4, "float": 4}


def _type(argument_type, members):
    """The manifest ``types`` row and published prelude for natural-alignment scalar members."""
    rows, offset = [], 0
    for ctype, name in members:
        size = _SIZES[ctype]
        offset = (offset + size - 1) // size * size
        rows.append(dict(name=name, ctype=ctype, extent=0, offset=offset, bytes=size, align=size))
        offset += size
    align = max(row["align"] for row in rows)
    decl = dict(name=argument_type, bytes=(offset + align - 1) // align * align, align=align, members=rows)
    prelude = (f"#ifndef {argument_type.upper()}_DEFINED\n#define {argument_type.upper()}_DEFINED\n"
               f"struct {argument_type} {{ {' '.join(f'{ctype} {name};' for ctype, name in members)} }};\n#endif\n")
    return decl, prelude


def _partner(directory, label, export, coords, argument_type, members, profile="sm_103a", channel_prefix="",
             placement=SMEM_ROWS, mutable=False, view_call=False):
    type_decl, prelude = _type(argument_type, members)
    exports, endpoints = {}, []
    argument = f"{'' if mutable else 'const '}{argument_type}*"
    leading = "const void*, " if view_call else ""
    for channel, endpoint, result, parameters in (
        ("produce_channel", "acquire", "pipelink_tensor_source", f"{leading}{argument}, void*"),
        ("release_channel", "retire", "void", f"{leading}{argument}"),
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
        endpoints=endpoints, claims=[], types=[type_decl],
        topology=dict(cta_group=dict(fixed=1), cluster=dict(compiled=[1, 1, 1], launch=dict(x_multiple_of=1, max=[1, 1, 1])),
                      multicast=[], persistent=False, scheduler="static"),
        params=dict(bytes=0, align=1, type="void", sha256_of_layout_probe="0" * 64),
        payload=dict(container_owned=True),
        tensor=dict(version="tensor-v2", exports=[dict(
            name=export, identity=identity, placements=[placement],
            read_capabilities=[{"smem": "smem_load_scalar", "reg": "reg_copy",
                                "tmem": "tmem_packet_receipt"}[placement["space"]]],
            readiness=dict(after=f"{label}_ready", reuse_after=f"{label}_done"),
            fixed_bindings={}, backing=[], calls=dict(produce=channel_prefix + "produce_channel", release=channel_prefix + "release_channel"))],
            imports=[], sites=[], bindings=[
                dict(name="base", sort="int", domain=dict(name="base", sort="int", values=[0, 128]), units="column",
                     owner="tensor_producer", lifetime="tensor_allocated_to_tensor_reclaimed",
                     observation_scope="tensor_import_site")] if "bind" in json.dumps(placement) else []),
    )
    path = directory / f"{label}.plink"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(raw))
    return Partner.load(path)


def _recipe(directory, monkeypatch, profile, collision=True, acc_placement=SMEM_ROWS):
    acc = _partner(directory, "acc", "accumulator", [("m", 128), ("n", 128)], "pl_gemm_tile_args",
                   [("const void*", "a"), ("const void*", "b"), ("int32_t", "M"), ("int32_t", "N"), ("int32_t", "K")],
                   profile, placement=acc_placement)
    scale = _partner(directory, "scale", "row_scale", [("m", 128)], "pl_rowscale_args",
                     [("const void*", "a"), ("int32_t", "M"), ("int32_t", "K"), ("float", "eps")],
                     profile, "" if collision else "scale_")
    monkeypatch.setenv("PIPELINK_ACC_PARTNER", str(acc.path))
    monkeypatch.setenv("PIPELINK_SCALE_PARTNER", str(scale.path))
    spec = importlib.util.spec_from_file_location("tensor_contract_recipe", RECIPE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _lower(module, profile):
    from pipelink_tilelang.explicit import DEFAULT_PASS_CONFIGS
    captured = capture(module.program(), target=tvm.target.Target({"kind": "cuda", "arch": profile}),
                       pass_configs=getattr(module, "PASS_CONFIGS", DEFAULT_PASS_CONFIGS))
    return {None: dict(source=captured.kernel_source, facts=captured.facts, passes=captured.passes,
                       snapshots=captured.snapshots, modules=captured.modules, layout_objects=captured.layout_objects)}


def _demand_rows(manifest):
    """Evaluate each packed positions demand: {(logical coordinate, participant, position)}."""
    rows = {}
    for imported in manifest.tensor.imports:
        demand, = imported.demands
        names = [domain.name for domain in demand.domain]
        coords = [item.name for item in imported.identity.coords]
        rows[imported.name] = {
            (tuple(expr.evaluate(demand.identity_coords[name], env) for name in coords),
             expr.evaluate(demand.participant, env), expr.evaluate(demand.position, env))
            for env in (dict(zip(names, point)) for point in itertools.product(
                *(range(domain.lo, domain.hi) for domain in demand.domain)))}
    return rows


def _coalesced_rows(profile):
    width = 8 if profile == "sm_90a" else 16
    return {((m, n), (m % width) * (128 // width) + n // width,
             (m // width) * width + n % width) for m in range(128) for n in range(128)}


def _holders(rows):
    result = {}
    for coordinates, participant, _ in rows:
        result.setdefault(coordinates[0], set()).add(participant)
    return result


def _import_calls(generated, channels):
    symbols = {symbol for channel in channels for symbol in
               (channel.produce_symbol, channel.symbol + "_ready", channel.symbol,
                channel.symbol + "_done", channel.release_symbol)}
    calls = []

    def visit(node):
        if (isinstance(node, tvm.tirx.Call) and isinstance(node.op, tvm.ir.Op)
                and node.op.name == "tirx.call_extern" and node.args[0].value in symbols):
            calls.append(node.args[0].value)

    for function in generated[None]["modules"]["pre_codegen"].functions.values():
        tvm.tirx.stmt_functor.post_order_visit(function.body, visit)
    return calls


def _registry(directory, module, generated, prelude=""):
    annotations, _ = _annotations(module.STAGE, {}, Ledger(), generated)
    params = module.STAGE.params
    fields = " ".join(f"{field.declarator};" for field in params.fields)
    source = directory / "annotations.cpp"
    source.write_text('#include <cstdint>\n#include <pipelink/stage.hpp>\n'
                      f"struct alignas({params.alignas}) {params.name} {{ {fields} }};\n"
                      + prelude
                      + "\n".join(f'extern "C" __device__ {export.signature};' for export in module.STAGE.exports)
                      + "\n".join(f'extern "C" __device__ void {symbol_prefix(module.STAGE)}role_{role}(const void*, unsigned);'
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
    generated = _lower(module, profile)
    emit(tmp_path, module, generated)
    records = _registry(tmp_path, module, generated)
    manifest = assemble({"records": records}, profile=profile)
    loaded = load_stage_manifest(manifest)
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
    # The shared-memory partner uses the profile's coalesced fallback. Row-scale
    # replicas must follow every accumulator row holder, not one row per thread.
    demands = _demand_rows(loaded)
    assert demands[module.acc.name] == _coalesced_rows(profile)
    assert _holders(demands[module.scale.name]) == _holders(demands[module.acc.name])
    assert _import_calls(generated, module.STAGE.channels) == [
        module.acc.produce_symbol, module.acc.symbol + "_ready", module.acc.symbol,
        module.scale.produce_symbol, module.scale.symbol + "_ready", module.scale.symbol,
        module.scale.symbol + "_done", module.scale.release_symbol,
        module.acc.symbol + "_done", module.acc.release_symbol,
    ]
    for site in loaded.tensor.sites:
        assert set(site.guarantees) == {
            "enter_once", "single_completion", "converged", "full_warps", "no_early_return"}
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
    caller_manifest = assemble({"records": _registry(tmp_path, module, _lower(module, "sm_103a"))}, profile="sm_103a")
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
    prelude = channel.protocol.call.decl + f"\nPL_TYPE({module.scale.argument_type}, {', '.join(module.scale.argument_members)});\n"
    records = _registry(provider_dir, module, {}, prelude)
    provider = assemble({"records": records}, profile="sm_103a")
    assert caller.endpoints[endpoint_name].owner_role == caller_role
    assert [(endpoint["name"], endpoint["direction"], endpoint["owner_role"], endpoint["protocol"]["export"])
            for endpoint in provider["endpoints"]] == [(endpoint_name, "consume", None, "retire")]
    assert provider["stage"]["roles"] == {}
    assert provider["claims"] == []


ROTATED_ROWS = dict(guard=None, space="reg", units="scalar", components={
    # Participant is the 7-bit row index rotated left by one: the shifted-out bit must wrap, not overflow.
    "participant": ["add", ["toint", ["shlc", ["tobits", 7, ["coord", "m"]], 1]],
                    ["toint", ["shrc", ["tobits", 7, ["coord", "m"]], 6]]],
    "position": ["coord", "n"]})


@pytest.mark.parametrize("placement", (WGMMA_REGISTERS, ROTATED_ROWS), ids=("wgmma", "rotated"))
def test_register_partner_placement_becomes_the_demand(tmp_path, monkeypatch, placement):
    """The accumulator demand adopts the published register placement; row scales follow its rows."""
    import pipelink_tilelang
    monkeypatch.setattr(pipelink_tilelang, "context", pipelink_tilelang.context)
    set_context("sm_90a", None)
    module = _recipe(tmp_path, monkeypatch, "sm_90a", acc_placement=placement)
    generated = _lower(module, "sm_90a")
    loaded = load_stage_manifest(assemble({"records": _registry(tmp_path, module, generated)}, profile="sm_90a"))
    demands = _demand_rows(loaded)
    placement = module.acc.partner.manifest.tensor.exports[0].placements[0].components
    expected = {((m, n), expr.evaluate(placement["participant"], dict(m=m, n=n)),
                 expr.evaluate(placement["position"], dict(m=m, n=n))) for m in range(128) for n in range(128)}
    assert demands[module.acc.name] == expected
    # Every thread holding part of row m also demands scale m, in one of its own slots.
    holders = {}
    for (m, _), participant, _ in expected:
        holders.setdefault(m, set()).add(participant)
    scale_holders = {}
    for (m,), participant, position in demands[module.scale.name]:
        scale_holders.setdefault(m, set()).add(participant)
    assert scale_holders == holders
    destination = next(lend for lend in next(item for item in loaded.tensor.imports
                                             if item.name == module.scale.name).lends if lend.name.endswith("destination"))
    assert max(position for _, _, position in demands[module.scale.name]) < destination.extent


@pytest.mark.parametrize("participant, position", (
    (["coord", "m"], ["add", ["coord", "n"], ["bind", "base"]]),  # link-time binding
    (["toint", ["bxor", ["tobits", 7, ["coord", "m"]], ["cbits", 7, 1]]], ["coord", "n"]),  # XOR swizzle
), ids=("binding", "xor"))
def test_inexpressible_register_placement_falls_back_to_coalesced(tmp_path, monkeypatch, participant, position):
    import pipelink_tilelang
    monkeypatch.setattr(pipelink_tilelang, "context", pipelink_tilelang.context)
    set_context("sm_90a", None)
    placement = dict(guard=None, space="reg", units="scalar",
                     components={"participant": participant, "position": position})
    module = _recipe(tmp_path, monkeypatch, "sm_90a", acc_placement=placement)
    loaded = load_stage_manifest(assemble({"records": _registry(tmp_path, module, _lower(module, "sm_90a"))},
                                          profile="sm_90a"))
    assert _demand_rows(loaded)[module.acc.name] == _coalesced_rows("sm_90a")


def test_specialization_opt_out_keeps_coalesced_demand(tmp_path, monkeypatch):
    """A representable register placement is ignored when the recipe opts out of specialization."""
    import pipelink_tilelang
    monkeypatch.setattr(pipelink_tilelang, "context", pipelink_tilelang.context)
    monkeypatch.setenv("PIPELINK_ACC_SPECIALIZE", "0")
    set_context("sm_90a", None)
    module = _recipe(tmp_path, monkeypatch, "sm_90a", acc_placement=WGMMA_REGISTERS)
    loaded = load_stage_manifest(assemble({"records": _registry(tmp_path, module, _lower(module, "sm_90a"))},
                                          profile="sm_90a"))
    assert _demand_rows(loaded)[module.acc.name] == _coalesced_rows("sm_90a")


def test_captured_demand_rejects_a_role_not_at_warp_zero(tmp_path, monkeypatch):
    import pipelink_tilelang
    monkeypatch.setattr(pipelink_tilelang, "context", pipelink_tilelang.context)
    set_context("sm_103a", None)
    module = _recipe(tmp_path, monkeypatch, "sm_103a")
    generated = _lower(module, "sm_103a")
    role, spec = next(iter(module.STAGE.roles.items()))
    module.STAGE = replace(module.STAGE, roles={role: replace(spec, warp_range=(4, 8))})
    with pytest.raises(AnnotationError) as caught:
        _annotations(module.STAGE, {}, Ledger(), generated)
    assert caught.value.field == f"tensor.imports.{module.acc.name}.guarantees.full_warps"


def test_minimal_scoped_recipe_through_cli(tmp_path, monkeypatch):
    from tilelang.env import CUDA_HOME
    from pipelink_tilelang.__main__ import main

    monkeypatch.setattr(pl, "context", pl.context)
    set_context("sm_90a", None)
    module = _recipe(tmp_path, monkeypatch, "sm_90a")
    assert not any(hasattr(module, name) for name in ("SEED", "PASS_CONFIGS", "DEPTHS", "EMITTER"))
    monkeypatch.setenv("PIPELINK_CUDA_HOME", CUDA_HOME)
    output = tmp_path / "cli"
    monkeypatch.setattr(sys, "argv", [
        "pipelink_tilelang", "--kernel", str(RECIPE), "--profile", "sm_90a", "--out", str(output)])
    main()
    destination = output / "generation/tilelang-epilogue"
    projection = json.loads((destination / "projection.json").read_text())
    facts = json.loads((destination / "facts.json").read_text())
    assert projection["stage_id"] == module.STAGE.stage_id
    assert projection["variants"] == []
    assert Path(projection["source"]).is_file()
    assert Path(projection["params_header"]).is_file()
    assert facts["variants"]["None"]


def _staged_recipe(directory, profile, *, register, padded=False):
    placement = WGMMA_REGISTERS if register else dict(
        guard=None, space="smem", units="byte",
        components={"address": ["mulc", 4, ["add", ["mulc", 128, ["coord", "m"]], ["coord", "n"]]]})
    partner = _partner(directory, "staged", "accumulator", [("m", 128), ("n", 128)],
                       "staged_args", [("int32_t", "M")], profile, placement=placement)
    channel = pl.import_tensor(partner, "accumulator", symbol="local_accumulator")

    def program():
        @T.prim_func
        def epilogue(residual: T.Tensor((128, 128), "float32"),
                     output: T.Tensor((128, 128), "float32")):
            with T.Kernel(1, threads=128):
                frag = channel.fragment(specialize=True)
                with channel.imported(frag):
                    staging = channel.staging((128, 128), "float32")
                    if padded:
                        T.annotate_layout({staging: T.Layout((128, 128), lambda i, j: (i, j * 2))})
                    T.copy(residual, staging)
                    for m, n in T.Parallel(128, 128):
                        frag[m, n] = frag[m, n] + staging[m, n]
                    T.copy(frag, staging)
                    T.copy(staging, output)
        return epilogue

    stage = pl.contract_stage("tensor.consumer.tilelang-epilogue", "1.0.0", program)
    return SimpleNamespace(STAGE=stage, program=stage.program, acc=channel)


@pytest.mark.parametrize("register", (True, False), ids=("register-export", "smem-export"))
def test_event_staging_claim_solver_overlap_and_ready_fence(tmp_path, monkeypatch, register):
    from pipec2.model import (
        Claim, Extent, Handle, Pipeline, PipelineTopology, StageRef, TensorConnection, TensorLifetime)
    from pipec2.solve import solve
    from pipec2.tensor.lifetimes import resolve_lifetimes
    from pipec2.tensor.lower import _event_sources
    from pipec2.tensor.plan import ConnectionPlanner

    monkeypatch.setattr(pl, "context", pl.context)
    set_context("sm_90a", None)
    module = _staged_recipe(tmp_path, "sm_90a", register=register)
    generated = _lower(module, "sm_90a")
    emit(tmp_path, module, generated)
    consumer = load_stage_manifest(assemble(
        {"records": _registry(tmp_path, module, generated)}, profile="sm_90a"))
    staging, = (claim for claim in consumer.claims if claim.domain == "smem")
    assert staging.extent.const == 128 * 128 * 4
    assert staging.align == 128
    assert module.acc.name != module.acc.export["name"]
    assert staging.lifetime == TensorLifetime(module.acc.name, module.acc.ready, module.acc.done)

    # Run the captured frontend claim through the real allocator; only the
    # provider's arena lifetime differs between routes.
    producer = module.acc.partner.manifest
    arena = Claim("smem", "pipelink_operand_arena",
                  TensorLifetime("accumulator", "tensor_work", module.acc.ready) if register else "kernel",
                  Extent(65536, {}, {}), 128, 0, None, None, None, None, None)
    producer = replace(producer, claims=[arena])
    stages = [producer, consumer]
    handles = {claim.handle: Handle(claim.handle, "base", "bind", claim.domain, None,
                                    f"claim:{stage.id}:{claim.handle}")
               for stage in stages for claim in stage.claims}
    connection = TensorConnection("acc", f"{producer.id}:accumulator", f"{consumer.id}:{module.acc.name}")
    pipeline = Pipeline(
        "staged", [StageRef(str(module.acc.partner.path), producer.id), StageRef("consumer.plink", consumer.id)],
        handles, [], [], {}, [], [], {}, PipelineTopology(False, "static"), consumer.target, (connection,))
    resolution = resolve_lifetimes(pipeline, stages)
    assert not resolution.failures
    profile = get_profile("sm_90a")
    baseline = solve(pipeline, stages, {}, [], profile)
    selected = solve(pipeline, stages, {}, [], profile, ordered_pairs=resolution.ordered_pairs)
    overlaps = resolution.report(selected)["overlaps"]
    assert resolution.report(baseline)["overlaps"] == []
    if register:
        overlap, = overlaps
        assert set(overlap["handles"]) == {arena.handle, staging.handle}
        assert overlap["event"] == module.acc.ready
        assert overlap["helper"] == module.acc.symbol + "_ready"
        assert overlap["ending_handle"] == arena.handle
        assert selected.launch.dyn_smem_bytes < baseline.launch.dyn_smem_bytes
    else:
        assert overlaps == []
        assert selected.launch.dyn_smem_bytes == baseline.launch.dyn_smem_bytes
        intervals = {item.handle: item for item in selected.intervals["smem"]}
        a, b = intervals[arena.handle], intervals[staging.handle]
        assert a.hi <= b.lo or b.hi <= a.lo

    planned = ConnectionPlanner(pipeline, stages, connection, profile).next()
    assert planned.status == "MODEL_SAT", planned.admission
    imported, = consumer.tensor.imports
    events = frozenset(item["event"] for item in overlaps)
    helpers = _event_sources(planned, imported, selected, "reg" if register else "smem",
                             witness_events=events)
    for kind, symbol in imported.callable.events.items():
        body = helpers.split(f"void {symbol}() {{", 1)[1].split("}", 1)[0]
        operations = re.findall(r'fence\.proxy\.async\.shared::cta|__syncthreads', body)
        expected = ["fence.proxy.async.shared::cta"] if register and kind == "ready" else []
        assert operations == [*expected, "__syncthreads"]


@pytest.mark.parametrize("explicit", (False, True))
def test_produce_member_binding_and_added_params(tmp_path, explicit):
    partner = _partner(tmp_path, "binding", "tensor", [("m", 128)], "binding_args",
                       [("int32_t", "M"), ("float", "alpha")])
    channel = pl.import_tensor(partner, "tensor",
                               arguments={"M": "height", "alpha": "weight"} if explicit else None)

    def program():
        @T.prim_func
        def worker(output: T.Tensor((128,), "float32")):
            with T.Kernel(1, threads=128):
                fragment = channel.fragment()
                with channel.imported(fragment):
                    T.copy(fragment, output)
        return worker

    stage = pl.contract_stage("test.binding", "1.0.0", program)
    fields = {field.name: field.cpp_type for field in stage.params.fields}
    assert fields == ({"output": "float*", "height": "int32_t", "weight": "float"} if explicit else
                      {"output": "float*", "M": "int32_t", "alpha": "float"})
    assert channel.args_param in stage.roles["worker"].entry_params
    assert channel.args_param not in fields


def test_produce_member_binding_rejects_type_mismatch(tmp_path):
    partner = _partner(tmp_path, "binding", "tensor", [("m", 128)], "binding_args", [("int32_t", "M")])
    channel = pl.import_tensor(partner, "tensor")

    def program():
        @T.prim_func
        def worker(output: T.Tensor((128,), "float32"), M: T.float32):
            with T.Kernel(1, threads=128):
                fragment = channel.fragment()
                with channel.imported(fragment):
                    T.copy(fragment, output)
        return worker

    with pytest.raises(AnnotationError) as caught:
        pl.contract_stage("test.binding", "1.0.0", program)
    assert caught.value.field == "params.M"


def test_caller_frame_uses_published_position_extent(tmp_path):
    placement = dict(guard=None, space="reg", units="scalar",
                     components={"participant": ["coord", "m"], "position": ["mulc", 2, ["coord", "n"]]})
    partner = _partner(tmp_path, "sparse", "tensor", [("m", 128), ("n", 128)], "sparse_args",
                       [("int32_t", "M")], placement=placement)
    channel = pl.import_tensor(partner, "tensor")

    def program():
        @T.prim_func
        def worker(output: T.Tensor((128, 128), "float32")):
            with T.Kernel(1, threads=128):
                fragment = channel.fragment()
                with channel.imported(fragment):
                    T.copy(fragment, output)
        return worker

    pl.contract_stage("test.sparse", "1.0.0", program)
    assert next(lend["extent"] for lend in channel.lends if lend["kind"] == "frame") == 255
    assert next(lend["extent"] for lend in channel.lends if lend["kind"] == "metadata") == 1


@pytest.mark.parametrize("profile,placement,narrow", (
    ("sm_103a", ROW_REGISTERS, False),
    ("sm_103a", TMEM_ROWS, False),
    ("sm_90a", WGMMA_REGISTERS, True),
    ("sm_120a", MMA_REGISTERS, True),
    # Change the map without changing the profile: the result follows geometry.
    ("sm_103a", WGMMA_REGISTERS, True),
    ("sm_90a", ROW_REGISTERS, False),
))
def test_staging_selection_follows_published_contiguous_runs(tmp_path, profile, placement, narrow):
    partner = _partner(tmp_path, "contiguous", "tile", [("m", 128), ("n", 128)],
                       "contiguous_args", [("int32_t", "M")], profile, placement=placement)
    channel = pl.import_tensor(partner, "tile")
    def check_queries():
        assert channel.delivery_layout_is_narrow() is narrow
        assert channel.delivery_layout_is_narrow(access_width=2) is False
        return 0

    @T.prim_func
    def probe():
        with T.Kernel(1, threads=128):
            T.evaluate(check_queries())


def test_staging_claim_covers_lowered_physical_padding(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "context", pl.context)
    set_context("sm_90a", None)
    module = _staged_recipe(tmp_path, "sm_90a", register=True, padded=True)
    generated = _lower(module, "sm_90a")
    manifest = load_stage_manifest(assemble(
        {"records": _registry(tmp_path, module, generated)}, profile="sm_90a"))
    staging, = (claim for claim in manifest.claims if claim.domain == "smem")
    # Columns 0,2,...,254 require 255 physical slots per row, not 128.
    assert staging.extent.const == 128 * 255 * 4


@pytest.mark.parametrize("method", ("delivery_layout", "delivery_layout_is_narrow"))
def test_delivery_queries_require_active_kernel(tmp_path, method):
    partner = _partner(tmp_path, "context", "tile", [("m", 128), ("n", 128)],
                       "context_args", [("int32_t", "M")], placement=TMEM_ROWS)
    channel = pl.import_tensor(partner, "tile")
    with pytest.raises(RuntimeError):
        getattr(channel, method)()
