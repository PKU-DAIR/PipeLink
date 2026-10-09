from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess

import pytest

from pipec2 import link as link_module
from pipec2.model import NVCC_DEFAULT_MATH, Claim, Extent, Handle, Interval, TensorConnection
from pipec2.profile import get_profile
from pipec2.schema import load_pipeline, load_stage_manifest
from pipec2.symbols import cuda_home
from pipec2.tensor import expr, ir, lower, plan
from pipec2.tensor.contracts import CallableParam, Participants, Rendezvous, ResourceLocation, TensorCallable
from test_link import _solution
from test_schema import _valid_pipeline, _valid_stage
from test_tensor_machine import _instance, _rebuild
from test_tensor_synth import _fixture, _loaded

ROOT = Path(__file__).resolve().parents[3]

_driver_spec = importlib.util.spec_from_file_location(
    "pipelink_tilelang_driver", ROOT / "frontends/tilelang/src/pipelink_tilelang/driver.py",
)
tilelang_driver = importlib.util.module_from_spec(_driver_spec)
_driver_spec.loader.exec_module(tilelang_driver)


PROFILES = ["sm_103a", "sm_90a", "sm_120a"]
PARAMETERS = "uint32_t tmem_base, float* fragment32, void* import_scratch"
REG_PARAMETERS = "const float* source_base, float* fragment32, void* import_scratch"
NATIVE_BODY = """  tl::tcgen05_ld_32dp32bNx<16, false>(tmem_base, 0, fragment32);
"""
REG_BODY = """  for (int slot = 0; slot < 16; ++slot) fragment32[slot] = source_base[slot];
"""
FENCED_SYNC = "  tl::tcgen05_before_thread_sync();\n  __syncthreads();\n  tl::tcgen05_after_thread_sync();\n"
BARE_SYNC = "  __syncthreads();\n"
RELAY = ("  float* pl_scratch_2 = static_cast<float*>(import_scratch);\n"
         "  pl_scratch_2[(int)threadIdx.x] = fragment32[0];\n{sync}"
         "  fragment32[0] = pl_scratch_2[(int)threadIdx.x];\n{sync}")
STAGED_BODY = NATIVE_BODY + RELAY.format(sync=FENCED_SYNC)
# Per target: sm103 receives from TMEM and fences every rendezvous; the TMEM-free targets
# (sm90, sm120) copy the register source and their rendezvous is the bare barrier.
ENTRY_BODY = {"sm_103a": NATIVE_BODY, "sm_90a": REG_BODY, "sm_120a": REG_BODY}
STAGED_BODIES = {"sm_103a": STAGED_BODY, "sm_90a": REG_BODY + RELAY.format(sync=BARE_SYNC),
                 "sm_120a": REG_BODY + RELAY.format(sync=BARE_SYNC)}
ENTRY_PARAMETERS = {"sm_103a": PARAMETERS, "sm_90a": REG_PARAMETERS, "sm_120a": REG_PARAMETERS}
TARGET_ENTRIES = {
    "sm_103a": {"rendezvous_fences": ["tl::tcgen05_before_thread_sync", "tl::tcgen05_after_thread_sync"],
                "receipt": "tilelang.tcgen05.packet", "force_include": "tl_templates/cuda/copy_sm100.h"},
    "sm_90a": {"rendezvous_fences": [], "receipt": None, "force_include": None},
    "sm_120a": {"rendezvous_fences": [], "receipt": None, "force_include": None},
}


def _kernel(body=NATIVE_BODY, parameters=PARAMETERS, threads=32):
    return ('#include <tl_templates/cuda/copy_sm100.h>\n'
            f'extern "C" __global__ void backend({parameters});\n'
            f'extern "C" __global__ void __launch_bounds__({threads}, 1) backend({parameters}) {{\n'
            + body + '}\n')


def _backend(monkeypatch, source=None, *, returncode=0, stderr=""):
    calls = []
    lower._CACHE.clear()
    monkeypatch.setenv("PIPELINK_TILELANG_PYTHON", "/pinned/tilelang/python")

    def run(command, **kwargs):
        calls.append((command, json.loads(kwargs["input"])))
        output = {"tilelang_version": "0.1.14", "source": source or _kernel(),
                  "projection": {"inverse_structurally_equal": True,
                                 "checked_original_tir_digests": ["a" * 64], "device_tir_digest": "b" * 64}}
        return subprocess.CompletedProcess(command, returncode, json.dumps(output), stderr)

    monkeypatch.setattr(lower.subprocess, "run", run)
    return calls


def _native_plan(*, staged=False, target="sm_103a"):
    receipt = target == "sm_103a"
    profile = get_profile(target)
    instance = _instance(threads=32, width=16, tmem=receipt, ready=True, profile=profile)
    source = CallableParam("tmem_base" if receipt else "source_base", "source_base", "pipelink_tensor_source")
    callable = TensorCallable("pl_import", f'extern "C" __device__ void pl_import({source.ctype}, float*, void*);', (
        source,
        CallableParam("fragment32", "dest_fragment", "float*"),
        CallableParam("import_scratch", "scratch", "void*"),
    ), {"ready": "pl_import_ready", "done": "pl_import_done"})
    lends = tuple(replace(lend, location=ResourceLocation("fragment32", None)) if lend.name == "dest"
                  else replace(lend, binding="relocatable", location=ResourceLocation("pipelink_tensor_import_smem", None)) if lend.name == "scratch"
                  else replace(lend, location=ResourceLocation(None, 0)) if lend.name == "barrier"
                  else lend for lend in instance.imp.lends)
    imp = replace(instance.imp, callable=callable, lends=lends,
                  rendezvous=tuple(Rendezvous(event, "barrier", Participants(set=tuple(range(32))))
                                   for event in ("ready", "done")))
    instance = _rebuild(instance, imp=imp)
    if receipt:
        nodes = [ir.Issue("tmem_packet_receipt", {"packet": 0, "dest": "dest", "offset": 0, "phase": "issue", "handle": "r"}, 0),
                 ir.Issue("tmem_packet_receipt", {"phase": "wait", "handle": "r"}, 0)]
    else:
        nodes = [ir.Issue("reg_copy_round", {"width": 16, "src": "@export:reg", "dst": "dest", "offset": 0, "src_offset": 0}, 0)]
    claims, handles = (), ()
    if staged:
        register = {"resource": "dest", "index": 0}
        shared = {"resource": "scratch", "index": ["mulc", 4, ["coord", "participant"]]}
        nodes += [ir.Issue("smem_store_scalar", {"src": register, "dst": shared}, 0),
                  ir.Issue("cta_rendezvous", {"resource": "barrier"}, 0),
                  ir.Issue("smem_load_scalar", {"src": shared, "dst": register}, 0),
                  ir.Issue("cta_rendezvous", {"resource": "barrier"}, 0)]
        handles = (Handle("pipelink_tensor_import_smem", "base", "bind", "smem", None, "claim:consumer.tensor.access:pipelink_tensor_import_smem"),)
        claims = (Claim("smem", handles[0].name, "kernel", Extent(128, {}, {}), 4, 0, None, None, None, None, None),)
    nodes = [replace(node, next=index + 1) for index, node in enumerate(nodes)] + [ir.Return()]
    program = instance.program({"hook": ir.ControlGraph(tuple(nodes))})
    context = plan._lowering_context(instance)
    result = plan.TensorPlan("MODEL_SAT", plan.Admission("OK", (), {}), ir.digest(program), program,
                             claims, handles, (), None, {"lowering_context": context,
                             "resource_demands": {r.name: (128 if staged else 0) if r.domain == "smem"
                                                  else 1 if r.domain == "barrier" else r.extent
                                                  for r in program.resources}})
    solved = replace(_solution(threads=32), native_bases={"pipelink_tensor_import_smem": 128},
                     intervals={"smem": [Interval("pipelink_tensor_import_smem", 128, 256, "kernel")]})
    return result, imp, solved, profile


def _round_plan(*, width=16, offset=0, target="sm_103a"):
    planned, imp, solved, profile = _native_plan(staged=True, target=target)
    address = ["mulc", 4, ["add", ["mulc", 32, ["coord", "slot"]], ["coord", "participant"]]]
    nodes = list(planned.program.graphs["hook"].nodes)
    store = next(index for index, node in enumerate(nodes) if isinstance(node, ir.Issue) and node.primitive == "smem_store_scalar")
    nodes[store] = ir.Issue("smem_store_round", {"width": width, "offset": offset, "src": "dest", "dst": "scratch", "address": address}, store + 1)
    nodes[store + 2] = ir.Issue("smem_load_round", {"width": width, "offset": offset, "src": "scratch", "dst": "dest", "address": address}, store + 3)
    program = replace(planned.program, graphs={"hook": ir.ControlGraph(tuple(nodes))})
    demands = {**planned.provenance["resource_demands"], "scratch": width * 128}
    solved = replace(solved, intervals={"smem": [
        Interval("pipelink_tensor_import_smem", 128, 128 + width * 128, "kernel")]})
    return replace(planned, program=program, ir_digest=ir.digest(program),
                   claims=(replace(planned.claims[0], extent=Extent(width * 128, {}, {})),),
                   provenance={**planned.provenance, "resource_demands": demands}), imp, solved, profile


def test_frozen_declaration_mismatch_is_rejected(monkeypatch):
    _backend(monkeypatch)
    planned, imp, solved, profile = _native_plan()
    changed = replace(imp, callable=replace(imp.callable, declaration=imp.callable.declaration.replace("pipelink_tensor_source", "int")))
    with pytest.raises(lower.LoweringUnsupported) as caught:
        lower.emit_adapter(planned, changed, solved, profile)
    assert caught.value.reason == "declaration_mismatch"


def test_backend_cannot_change_frozen_pointer_type(monkeypatch):
    _backend(monkeypatch, _kernel(parameters=PARAMETERS.replace("void*", "float*")))
    with pytest.raises(lower.LoweringUnsupported) as caught:
        lower.emit_adapter(*_native_plan())
    assert caught.value.reason == "declaration_mismatch"


@pytest.mark.parametrize("target", PROFILES)
def test_staging_borrows_caller_scratch_and_solved_reservation(monkeypatch, target):
    calls = _backend(monkeypatch, _kernel(STAGED_BODIES[target], ENTRY_PARAMETERS[target]))
    adapter = lower.emit_adapter(*_native_plan(staged=True, target=target))
    scratch = adapter.resources["bindings"]["scratch"]
    assert (scratch["param"], scratch["base"]) == ("import_scratch", 128)
    assert (scratch["declared_extent"], scratch["required_extent"], scratch["allocated_extent"]) == (2048, 128, 128)
    assert calls[0][1]["resources"]["barrier"]["base"] == 0
    assert (calls[0][1]["target"], calls[0][1]["target_entry"]) == (target, TARGET_ENTRIES[target])


@pytest.mark.parametrize("target", PROFILES)
@pytest.mark.parametrize("named_barrier", [False, True])
@pytest.mark.parametrize("witness_events", [frozenset(), frozenset({"ready"}), frozenset({"done"})])
def test_overlap_witness_fences_before_its_outer_barrier(monkeypatch, target, named_barrier, witness_events):
    _backend(monkeypatch, _kernel(ENTRY_BODY[target], ENTRY_PARAMETERS[target]))
    planned, imp, solved, profile = _native_plan(target=target)
    if named_barrier:
        imp = replace(imp, lends=tuple(
            replace(lend, binding="relocatable", location=ResourceLocation("pipelink_outer", None))
            if lend.name == "barrier" else lend for lend in imp.lends))
        solved = replace(solved, native_bases={**solved.native_bases, "pipelink_outer": 1})
    adapter = lower.emit_adapter(planned, imp, solved, profile, witness_events=witness_events)
    for event, symbol in imp.callable.events.items():
        body = adapter.source.split(f"void {symbol}() {{", 1)[1].split("}", 1)[0]
        # Check the emitted synchronization operations, not their formatting:
        # every cohort thread fences before the rendezvous, with TMEM ordering retained.
        operations = re.findall(r'asm\s+volatile\("([^";]+)|(__syncthreads)\s*\(', body)
        actual = [instruction or intrinsic for instruction, intrinsic in operations]
        expected = []
        if target == "sm_103a":
            if event == "done":
                expected.append("tcgen05.wait::ld.sync.aligned")
            expected.append("tcgen05.fence::before_thread_sync")
        if event in witness_events:
            expected.append("fence.proxy.async.shared::cta")
        expected.append("bar.sync %0, %1" if named_barrier else "__syncthreads")
        if target == "sm_103a":
            expected.append("tcgen05.fence::after_thread_sync")
        assert actual == expected


@pytest.mark.parametrize("target", PROFILES)
def test_staged_round_borrows_the_same_reservation_without_new_sync(monkeypatch, target):
    body = STAGED_BODIES[target].replace(
        "  pl_scratch_2[(int)threadIdx.x] = fragment32[0];",
        "  for (int slot = 0; slot < 16; ++slot) pl_scratch_2[32 * slot + (int)threadIdx.x] = fragment32[slot];",
    ).replace(
        "  fragment32[0] = pl_scratch_2[(int)threadIdx.x];",
        "  for (int slot = 0; slot < 16; ++slot) fragment32[slot] = pl_scratch_2[32 * slot + (int)threadIdx.x];",
    )
    _backend(monkeypatch, _kernel(body, ENTRY_PARAMETERS[target]))
    adapter = lower.emit_adapter(*_round_plan(target=target))
    assert adapter.resources["bindings"]["scratch"]["param"] == "import_scratch"
    assert adapter.resources["bindings"]["scratch"]["base"] == 128
    assert adapter.resources["extra_storage"] == 0
    _backend(monkeypatch, _kernel(body + "  __syncthreads();\n", ENTRY_PARAMETERS[target]))
    with pytest.raises(lower.LoweringUnsupported) as caught:
        lower.emit_adapter(*_round_plan(target=target))
    assert caught.value.reason == "effect_mismatch"




@pytest.mark.parametrize("target,sync", [("sm_103a", BARE_SYNC), ("sm_90a", FENCED_SYNC), ("sm_120a", FENCED_SYNC)])
def test_rendezvous_with_the_other_targets_fence_shape_is_rejected(monkeypatch, target, sync):
    _backend(monkeypatch, _kernel(ENTRY_BODY[target] + RELAY.format(sync=sync), ENTRY_PARAMETERS[target]))
    with pytest.raises(lower.LoweringUnsupported) as caught:
        lower.emit_adapter(*_native_plan(staged=True, target=target))
    assert caught.value.reason == "effect_mismatch"


@pytest.mark.parametrize("target", ["sm_90a", "sm_120a"])
def test_unfenced_target_rejects_any_tcgen05_token_in_the_body(monkeypatch, target):
    body = STAGED_BODIES[target].replace("  fragment32[0] = pl_scratch_2", "  tl::tcgen05_after_thread_sync();\n  fragment32[0] = pl_scratch_2")
    _backend(monkeypatch, _kernel(body, REG_PARAMETERS))
    with pytest.raises(lower.LoweringUnsupported) as caught:
        lower.emit_adapter(*_native_plan(staged=True, target=target))
    assert caught.value.reason == "effect_mismatch"


def test_receipt_steps_have_no_recipe_on_a_target_without_receipts(monkeypatch):
    monkeypatch.delenv("PIPELINK_TILELANG_PYTHON", raising=False)
    planned, imp, solved, _ = _native_plan()
    with pytest.raises(lower.LoweringUnsupported) as caught:
        lower.emit_adapter(planned, imp, solved, get_profile("sm_90a"))
    assert (caught.value.reason, caught.value.details["primitive"]) == ("no_recipe", "tmem_packet_receipt")
    # The legalizer guards the same fact for a request it is handed directly.
    request = lower._prepare(planned, imp, solved, get_profile("sm_103a"))
    request["target_entry"] = TARGET_ENTRIES["sm_90a"]
    with pytest.raises(lower.LoweringUnsupported) as caught:
        lower._legalize(_kernel(), request)
    assert (caught.value.reason, caught.value.step) == ("no_recipe", "hook:0")


def test_target_without_lowering_entry_is_rejected_before_backend(monkeypatch):
    monkeypatch.delenv("PIPELINK_TILELANG_PYTHON", raising=False)
    planned, imp, solved, profile = _native_plan(target="sm_90a")
    with pytest.raises(lower.LoweringUnsupported) as caught:
        lower.emit_adapter(planned, imp, solved, replace(profile, name="sm_80"))
    assert (caught.value.step, caught.value.reason) == ("profile", "unsupported_ir")
    assert "sm_80" in caught.value.detail


def test_round_slot_coordinate_does_not_capture_published_slot_binding():
    address = ["mulc", 4, ["add", ["mulc", 32, ["coord", "slot"]], ["bind", "slot"]]]
    lowered = tilelang_driver._bind_coordinates(address, {"slot": 2})
    parsed = expr.parse(lowered, {"slot": expr.Domain("slot", "int", values=(7,))})
    assert expr.evaluate(parsed, {"slot": 7}) == (32 * 2 + 7) * 4
    # A repeated emission binds a fresh cursor, not the previous round's value.
    next_address = tilelang_driver._bind_coordinates(address, {"slot": 3})
    parsed = expr.parse(next_address, {"slot": expr.Domain("slot", "int", values=(7,))})
    assert expr.evaluate(parsed, {"slot": 7}) == (32 * 3 + 7) * 4



def test_native_receipt_admission_uses_actual_tilelang_operand_geometry():
    # Pinned 0.1.14 tl.tcgen05_ld ABI: datapaths, repetitions, pack16,
    # address, column offset, pattern bits, destination pointer.
    actual_x16 = (32, 16, False, "source", 0, 32, "destination")
    tilelang_driver._require_native_receipt(actual_x16)
    with pytest.raises(RuntimeError, match="native receipt geometry"):
        tilelang_driver._require_native_receipt((32, 32, False, "source", 0, 32, "destination"))

@pytest.mark.parametrize("width,offset", [(17, 0), (16, 1), (0, 0)])
def test_shared_round_cannot_widen_private_storage(width, offset):
    with pytest.raises(lower.LoweringUnsupported) as caught:
        lower.emit_adapter(*_round_plan(width=width, offset=offset))
    assert caught.value.reason == "effect_mismatch"


def test_recipe_effects_cannot_outgrow_generated_claim(monkeypatch):
    planned, imp, solved, profile = _native_plan(staged=True)
    planned = replace(planned, claims=(replace(planned.claims[0], extent=Extent(4, {}, {})),))
    with pytest.raises(lower.LoweringUnsupported) as caught:
        lower.emit_adapter(planned, imp, solved, profile)
    assert caught.value.reason == "effect_mismatch"


def test_unimplemented_recipe_and_missing_tooling_are_distinct(monkeypatch):
    args = _native_plan()
    catalogue = dict(lower.machine.CATALOGUE(args[-1]))
    catalogue["tmem_packet_receipt"] = replace(catalogue["tmem_packet_receipt"], lowering_recipe=None)
    monkeypatch.setattr(lower.machine, "CATALOGUE", lambda profile: catalogue)
    monkeypatch.delenv("PIPELINK_TILELANG_PYTHON", raising=False)
    with pytest.raises(lower.LoweringUnsupported) as caught:
        lower.emit_adapter(*args)
    assert caught.value.reason == "no_recipe"
    assert caught.value.to_json_obj()["status"] == "LOWERING_UNSUPPORTED"
    assert args[0].status == "MODEL_SAT"


def test_missing_interpreter_reports_actionable_tooling_failure(monkeypatch):
    monkeypatch.delenv("PIPELINK_TILELANG_PYTHON", raising=False)
    with pytest.raises(lower.LoweringUnsupported) as caught:
        lower.emit_adapter(*_native_plan())
    assert caught.value.reason == "tooling"
    assert caught.value.details["kind"] == "missing_interpreter"


def test_subprocess_failure_preserves_actual_diagnostic(monkeypatch):
    _backend(monkeypatch, returncode=7, stderr="TileLang API mismatch: missing tcgen05 lowering")
    with pytest.raises(lower.LoweringUnsupported) as caught:
        lower.emit_adapter(*_native_plan())
    assert caught.value.reason == "tooling"
    assert caught.value.details["returncode"] == 7
    assert "API mismatch" in caught.value.details["stderr"]


def test_emission_is_stable_and_records_trusted_deployment_premises(monkeypatch):
    calls = _backend(monkeypatch)
    args = _native_plan()
    first, second = lower.emit_adapter(*args), lower.emit_adapter(*args)
    assert first.source == second.source
    assert len(calls) == 1
    assert set(first.defined_symbols) == {"pl_import", "pl_import_ready", "pl_import_done"}
    assert {premise["name"] for premise in first.trusted_premises} == {"entry_linkage_rewrite", "borrowed_base_binding"}
    assert all("identity_import_differential" in premise["deployment_checks"] for premise in first.trusted_premises)
    assert first.trusted_premises[0]["body_byte_identical"] is True
    assert first.resources["native_resources_checked"] is False


@pytest.mark.parametrize("body,reason", [
    (NATIVE_BODY.replace("<16, false>", "<32, false>"), "refinement"),
    (NATIVE_BODY + "  __syncthreads();\n", "effect_mismatch"),
    ("  float undeclared[16];\n" + NATIVE_BODY, "effect_mismatch"),
    ("  ushort undeclared[16];\n" + NATIVE_BODY, "effect_mismatch"),
    ("  tl::tmem_allocate(fragment32, 32);\n" + NATIVE_BODY, "effect_mismatch"),
])
def test_backend_instruction_or_resource_widening_is_rejected(monkeypatch, body, reason):
    _backend(monkeypatch, _kernel(body))
    with pytest.raises(lower.LoweringUnsupported) as caught:
        lower.emit_adapter(*_native_plan())
    assert caught.value.reason == reason


def test_receipt_completion_cannot_move_across_another_step(monkeypatch):
    planned, imp, solved, profile = _native_plan()
    graph = planned.program.graphs["hook"]
    nodes = (graph.nodes[0], ir.Meta("seen", ["const", 1], 2), replace(graph.nodes[1], next=3), ir.Return())
    program = replace(planned.program, graphs={"hook": ir.ControlGraph(nodes)})
    planned = replace(planned, program=program, ir_digest=ir.digest(program))
    with pytest.raises(lower.LoweringUnsupported) as caught:
        lower.emit_adapter(planned, imp, solved, profile)
    assert caught.value.reason == "refinement"


@pytest.mark.parametrize("dtype,ctype,tir_dtype,elem_bytes", [
    ("f32", "float", "float32", 4),
    ("bf16", "uint16_t", "uint16", 2),
    ("i32", "int32_t", "int32", 4),
    ("u32", "uint32_t", "uint32", 4),
])
def test_direct_class_plan_lowers_through_configured_backend(monkeypatch, dtype, ctype, tir_dtype, elem_bytes):
    raw = _fixture(already_ready=True, readiness=False)
    raw["exports"][0]["identity"]["dtype"] = dtype
    raw["imports"][0]["identity"]["dtype"] = dtype
    callable = raw["imports"][0]["callable"]
    callable["params"].insert(0, {"name": "source", "meaning": "source_base", "ctype": "pipelink_tensor_source"})
    callable["params"][1]["ctype"] = f"{ctype}*"
    callable["declaration"] = f'extern "C" __device__ void pipelink_tensor_test_import(pipelink_tensor_source, {ctype}*);'
    parameters = f"const {ctype}* source, {ctype}* dest"
    tensor, _ = _loaded(raw)
    base = load_stage_manifest(_valid_stage())
    base = replace(base, target=replace(base.target, schema="3.0"))
    stages = [replace(base, id="producer", tensor=tensor), replace(base, id="consumer", tensor=tensor)]
    connection = TensorConnection("direct", "producer:source", "consumer:sink")
    planned = plan.ConnectionPlanner(load_pipeline(_valid_pipeline()), stages, connection, get_profile("sm_103a")).next()
    assert planned.status == "MODEL_SAT"
    backend_parameters = parameters.replace("uint16_t", "ushort")
    calls = _backend(monkeypatch, _kernel("  dest[0] = source[0];\n", backend_parameters, 1))
    emitted = lower.emit_adapter(planned, tensor.imports[0], _solution(threads=1), get_profile("sm_103a"))
    assert set(emitted.defined_symbols) == {"pipelink_tensor_test_import", "pipelink_tensor_test_import_ready", "pipelink_tensor_test_import_done"}
    assert (calls[0][1]["dtype"], calls[0][1]["elem_bytes"]) == (tir_dtype, elem_bytes)


def test_bf16_tmem_receipt_is_rejected_by_model_before_lowering():
    instance = _instance(threads=32, width=16, tmem=True, ready=True)
    identity = replace(instance.imp.identity, dtype="bf16")
    export = replace(instance.export, identity=identity)
    imp = replace(instance.imp, identity=identity)
    profile = get_profile("sm_103a")
    derived = plan.derive(export, imp, instance.sites, (), profile)
    admission, admitted = plan.admit(
        export, imp, derived, profile,
    )
    assert admission.status == "UNSUPPORTED_MODEL"
    assert admitted is None
    assert {issue.clause_id for issue in admission.issues} == {"T-MODEL"}
    assert "32-bit payload" in admission.issues[0].message


def test_unsupported_payload_dtype_is_rejected_before_backend(monkeypatch):
    monkeypatch.delenv("PIPELINK_TILELANG_PYTHON", raising=False)
    planned, imp, solved, profile = _native_plan()
    imp = replace(imp, identity=replace(imp.identity, dtype="f64"))
    with pytest.raises(lower.LoweringUnsupported) as caught:
        lower.emit_adapter(planned, imp, solved, profile)
    assert caught.value.reason == "unsupported_ir"


@pytest.fixture
def pinned_adapter_toolchain(monkeypatch):
    """nvcc plus the pinned TileLang interpreter and the two adapter include roots."""
    try:
        cuda = cuda_home()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    if not (cuda / "bin" / "nvcc").is_file():
        pytest.skip("configured CUDA compiler is unavailable")
    interpreter = Path(os.environ.get("PIPELINK_TILELANG_PYTHON") or ROOT / "frontends/tilelang/.venv/bin/python")
    probe = subprocess.run([str(interpreter), "-c", "import pathlib,tilelang,pipelink_tilelang; "
                            "assert tilelang.__version__ == '0.1.14'; print(pathlib.Path(tilelang.__file__).parent / 'src')"],
                           capture_output=True, text=True, check=False) if interpreter.is_file() else None
    if probe is None or probe.returncode:
        pytest.skip("pinned TileLang 0.1.14 interpreter is unavailable")
    roots = {"CUTLASS_INCLUDE_DIR": ROOT / "third_party/cutlass/include",
             "TILELANG_INCLUDE_DIR": Path(probe.stdout.strip())}
    for name, root in roots.items():
        if not root.is_dir():
            pytest.skip(f"adapter include root {name} is unavailable")
        monkeypatch.setenv(name, str(root))
    monkeypatch.setenv("PIPELINK_TILELANG_PYTHON", str(interpreter))
    lower._CACHE.clear()
    return cuda


@pytest.mark.parametrize("target", PROFILES)
def test_staged_adapter_lowers_through_the_pinned_backend_and_compiles_for_its_target(tmp_path, pinned_adapter_toolchain, target):
    adapter = lower.emit_adapter(*_native_plan(staged=True, target=target))
    source = tmp_path / f"adapter_{target}.cu"
    source.write_text(adapter.source)
    profile = get_profile(target)
    command = link_module._adapter_compile_command(pinned_adapter_toolchain / "bin" / "nvcc", profile, source, tmp_path / f"adapter_{target}.o")
    compiled = subprocess.run(command, capture_output=True, text=True, env=link_module._tool_env(pinned_adapter_toolchain))
    assert compiled.returncode == 0, compiled.stderr
    assert (tmp_path / f"adapter_{target}.o").stat().st_size > 0


@pytest.mark.parametrize("target", PROFILES)
def test_two_native_import_adapters_link_without_shared_strong_helpers(tmp_path, pinned_adapter_toolchain, target):
    planned, imported, solved, profile = _native_plan(target=target)
    adapters = {}
    declarations = []
    calls = []
    for suffix in ("a", "b"):
        symbol = f"pl_import_{suffix}"
        callable = replace(imported.callable, symbol=symbol,
                           declaration=imported.callable.declaration.replace("pl_import(", symbol + "("),
                           events={"ready": symbol + "_ready", "done": symbol + "_done"})
        adapter = lower.emit_adapter(planned, replace(imported, callable=callable), solved, profile)
        adapters[f"adapter_{suffix}.cu"] = adapter
        declarations.extend((callable.declaration,
                             f'extern "C" __device__ void {symbol}_ready();',
                             f'extern "C" __device__ void {symbol}_done();'))
        calls.extend((f"{symbol}_ready();", f"{symbol}(source, result, nullptr);",
                      f"output[threadIdx.x + {0 if suffix == 'a' else 32}] = result[0];",
                      f"{symbol}_done();"))
    source = "pl::source_of_tmem(0)" if target == "sm_103a" else "pl::source_of(frame)"
    kernel = ('#include <pipelink/tensor_source.h>\n' + "\n".join(declarations) +
              '\nextern "C" __global__ void pipelink_kernel(float* output) {\n'
              '  float frame[16], result[16];\n'
              '  for (int i = 0; i < 16; ++i) frame[i] = float(threadIdx.x + i);\n'
              f'  pipelink_tensor_source source = {source};\n' +
              "\n".join(calls) + "\n}\n")
    required = {symbol: "consumer" for adapter in adapters.values() for symbol in adapter.defined_symbols}
    symbol, observed, _ = link_module._compile_and_measure(
        tmp_path, (), {"dispatch.cu": kernel, **{name: adapter.source for name, adapter in adapters.items()}},
        profile, solved, "pipelink_kernel", NVCC_DEFAULT_MATH, adapters=adapters, tensor_required=required)
    assert symbol == "pipelink_kernel"
    assert observed[1:] == (0, 0)


def test_adapter_object_rejects_uncontracted_strong_device_symbols(tmp_path, pinned_adapter_toolchain):
    from pipec2.model import LinkError

    adapter = lower.AdapterSource("synthetic",
        'extern "C" __device__ void promised() {}\n'
        'extern "C" __device__ void accidental_global() {}\n',
        ("promised",), (), {})
    with pytest.raises(LinkError) as caught:
        link_module._compile_and_measure(
            tmp_path, (), {"adapter.cu": adapter.source}, get_profile("sm_103a"), _solution(),
            "pipelink_kernel", NVCC_DEFAULT_MATH, adapters={"adapter.cu": adapter}, tensor_required={"promised": "consumer"})
    assert [(failure.clause_id, failure.path) for failure in caught.value.failures] == [
        ("T-NATIVE-SYMBOLS", "generated.adapter.cu.symbols")]
    assert caught.value.failures[0].detail["unexpected"] == ["accidental_global"]


@pytest.mark.parametrize("cohort, accepted", [
    (tuple(range(64)), True),              # two complete warps
    (tuple(range(0, 64, 2)), False),       # 32 lanes spread across two warps: bar.sync would hang
    (tuple(range(16)) + tuple(range(32, 48)), False),
])
def test_outer_named_barrier_cohort_must_be_a_union_of_complete_warps(monkeypatch, cohort, accepted):
    _backend(monkeypatch, _kernel(REG_BODY, REG_PARAMETERS))
    planned, imp, solved, profile = _native_plan(target="sm_90a")
    lends = tuple(replace(lend, binding="relocatable", location=ResourceLocation("pipelink_outer", None))
                  if lend.name == "barrier" else lend for lend in imp.lends)
    imp = replace(imp, lends=lends,
                  rendezvous=tuple(Rendezvous(event, "barrier", Participants(set=cohort)) for event in ("ready", "done")))
    solved = replace(_solution(threads=64), native_bases={**solved.native_bases, "pipelink_outer": 1})
    if accepted:
        adapter = lower.emit_adapter(planned, imp, solved, profile)
        assert f'"r"({len(cohort)})' in adapter.source
    else:
        with pytest.raises(lower.LoweringUnsupported) as caught:
            lower.emit_adapter(planned, imp, solved, profile)
        assert caught.value.reason == "effect_mismatch"
