from __future__ import annotations

from copy import deepcopy
import ctypes
from dataclasses import replace
import shutil
from hashlib import sha256
import subprocess
from pathlib import Path

import pytest

from pipec2.profile import get_profile
from pipec2.tensor import check, ir, lower, machine, plan, synth
from test_link import _solution
from test_tensor_lower import _backend
from test_tensor_synth import _domain, _fixture, _instance, _pointer_fixture, _pointer_plan, _resource


PROFILES = ["sm_103a", "sm_90a", "sm_120a"]
VIEW_PARAMETERS = "pipelink_tensor_source, const void**, void*"
VIEW_DECLARATION = f'extern "C" __device__ void view_import({VIEW_PARAMETERS});'


def _view_fixture(*, transpose=False, ready=True, threads=2):
    """Four identities and disjoint relative source/scratch views."""
    raw = _fixture(shared=True, width=2, already_ready=ready, readiness=False)
    identity = raw["exports"][0]["identity"]
    identity["coords"] = [_domain("row", 2), _domain("slot", 2)]
    row, slot = ["coord", "row"], ["coord", "slot"]
    source_address = ["add", ["mulc", 8, row], ["mulc", 4, slot]]
    view_address = ["add", ["mulc", 4, row], ["mulc", 8, slot]] if transpose else source_address
    raw["exports"][0]["placements"] = [{
        "guard": None, "space": "smem", "units": "byte",
        "components": {"address": source_address},
    }]
    raw["exports"][0]["read_capabilities"] = ["smem_load_scalar"]
    backing = _resource("source_storage", "smem", 16)
    backing.update(binding="fixed", location={"handle": None, "base": 0})
    raw["exports"][0]["backing"] = [backing]
    lends = [_resource("temp", "reg", 2), _resource("scratch", "smem", 16),
             _resource("barrier", "barrier", 1)]
    lends[-1].update(binding="fixed", location={"handle": None, "base": 0})
    raw["sites"][0]["participants"] = {"set": list(range(threads))}
    imp = raw["imports"][0]
    imp["lends"] = lends
    imp["rendezvous"] = [{"event": event, "barrier": "barrier", "cohort": {"set": list(range(threads))}}
                         for event in ("ready", "done")]
    imp["demands"] = [{
        "kind": "memory_view", "domain": [_domain("row", 2), _domain("slot", 2)],
        "guard": None, "identity_coords": {"row": row, "slot": slot},
        "space": "smem", "units": "byte", "components": {"address": view_address},
    }]
    imp["callable"] = {
        "symbol": "view_import", "declaration": VIEW_DECLARATION,
        "events": {"ready": "view_import_ready", "done": "view_import_done"},
        "params": [
            {"name": "source_base", "meaning": "source_base", "ctype": "pipelink_tensor_source"},
            {"name": "dest_view", "meaning": "dest_view", "ctype": "const void**"},
            {"name": "import_scratch", "meaning": "scratch", "ctype": "void*"},
        ],
    }
    return raw


def _view_plan(*, transpose=False, ready=True, threads=2, profile="sm_103a"):
    raw = _view_fixture(transpose=transpose, ready=ready, threads=threads)
    profile = get_profile(profile)
    planned, imp = _pointer_plan(raw, profile)
    assert planned.status == "MODEL_SAT", planned.to_json_obj()
    solved = replace(_solution(threads=threads), native_bases={"pipelink_tensor_test_scratch": 128})
    return planned, imp, solved, profile


def _receipt_candidate(program, checked):
    return synth.SynthResult("MODEL_SAT", program, {}, {}, {
        "candidates": [{"check": checked.to_json_obj()}],
    })


@pytest.mark.parametrize("profile", PROFILES)
def test_matching_shared_view_aliases_with_exact_published_identity_evidence(profile):
    planned, imp, _, target = _view_plan(profile=profile)
    instance = _instance(_view_fixture(), target)
    assert planned.provenance["classes"][-1]["name"] == "view_alias"
    assert planned.program.view_binding == "@export:smem"
    assert not any(isinstance(node, ir.Issue)
                   for graph in planned.program.graphs.values() for node in graph.nodes)
    checked = check.check(instance, planned.program, budget=100_000)
    assert checked.status == "PASS"
    assert checked.provenance["terminal_deliveries"] == [{
        "bindings": {},
        "outputs": [[-1, 0, 0], [-1, 4, 1], [-1, 8, 2], [-1, 12, 3]],
        "view": {"resource": "@export:smem", "cells": [[0, 0, 0], [4, 1, 1], [8, 2, 2], [12, 3, 3]]},
    }]
    replay = plan._oracle(instance, imp, _receipt_candidate(planned.program, checked))
    assert replay["ok"] and replay["checked_positions"] == 4
    assert planned.provenance["admission_oracle"]["ok"]
    assert ir.parse(planned.program.to_json_obj()) == planned.program


@pytest.mark.parametrize("field", [1, 2], ids=["identity", "publication"])
def test_view_oracle_rejects_corrupted_terminal_identity_or_publication(field):
    planned, imp, _, _ = _view_plan()
    instance = _instance(_view_fixture())
    checked = check.check(instance, planned.program, budget=100_000)
    assert checked.status == "PASS"
    candidate = _receipt_candidate(planned.program, checked)
    altered = deepcopy(candidate.provenance)
    altered["candidates"][-1]["check"]["provenance"]["terminal_deliveries"][0]["view"]["cells"][1][field] = machine.EMPTY
    replay = plan._oracle(instance, imp, replace(candidate, provenance=altered))
    assert not replay["ok"]
    assert not replay["incomplete"]


@pytest.mark.parametrize("profile", PROFILES)
def test_mismatched_layout_materializes_the_demanded_view_without_loadback(profile):
    planned, imp, _, target = _view_plan(transpose=True, profile=profile)
    instance = _instance(_view_fixture(transpose=True), target)
    assert planned.provenance["classes"][-1]["name"] == "view_materialize"
    assert planned.program.view_binding == "scratch"
    checked = check.check(instance, planned.program, budget=100_000)
    assert checked.status == "PASS"
    assert checked.provenance["terminal_deliveries"] == [{
        "bindings": {},
        "outputs": [[-1, 0, 0], [-1, 4, 2], [-1, 8, 1], [-1, 12, 3]],
        "view": {"resource": "scratch", "cells": [[0, 0, 0], [4, 2, 2], [8, 1, 1], [12, 3, 3]]},
    }]
    assert plan._oracle(instance, imp, _receipt_candidate(planned.program, checked))["ok"]
    issues = [node for graph in planned.program.graphs.values()
              for node in graph.nodes if isinstance(node, ir.Issue)]
    assert any(node.primitive == "smem_store_round" for node in issues)
    assert sum(node.primitive == "cta_rendezvous" for node in issues) == 1
    assert all(node.operands["src"] == "@export:smem"
               for node in issues if node.primitive == "smem_load_round")


def test_alias_without_source_readiness_guard_is_unsafe():
    planned, _, _, _ = _view_plan(ready=False)
    instance = _instance(_view_fixture(ready=False))
    assert planned.provenance["classes"][-1]["name"] == "view_alias"
    assert check.check(instance, planned.program, budget=100_000).status == "PASS"
    unguarded = replace(planned.program, graphs={"load": ir.ControlGraph((ir.Return(),))})
    assert check.check(instance, unguarded, budget=100_000).status == "UNSAFE"


def test_materialization_without_publication_rendezvous_is_unsafe():
    planned, _, _, _ = _view_plan(transpose=True)
    instance = _instance(_view_fixture(transpose=True))
    assert check.check(instance, planned.program, budget=100_000).status == "PASS"
    graph = planned.program.graphs["load"]
    index = next(index for index, node in enumerate(graph.nodes)
                 if isinstance(node, ir.Issue) and node.primitive == "cta_rendezvous")
    assert len(graph.nodes) == index + 2 and isinstance(graph.nodes[-1], ir.Return)
    unpublished = replace(planned.program, graphs={
        "load": replace(graph, nodes=graph.nodes[:index] + (graph.nodes[-1],)),
    })
    assert check.check(instance, unpublished, budget=100_000).status == "UNSAFE"


def test_changed_view_binding_cannot_lower_under_unchanged_plan_digest(monkeypatch):
    planned, imp, solved, profile = _view_plan()
    changed = replace(planned.program, view_binding="scratch")
    assert ir.digest(changed) != planned.ir_digest

    def forbidden_backend(_request):
        pytest.fail("a program with a changed view binding reached backend lowering")

    monkeypatch.setattr(lower, "_backend", forbidden_backend)
    with pytest.raises(lower.LoweringUnsupported) as caught:
        lower.emit_adapter(replace(planned, program=changed), imp, solved, profile)
    assert caught.value.reason == "unsupported_ir"


MATERIALIZE_FILL = """  float* target = static_cast<float*>(import_scratch);
  target[0] = source_base[0];
  target[1] = source_base[2];
  target[2] = source_base[1];
  target[3] = source_base[3];
"""
FENCED_RENDEZVOUS = "  tl::tcgen05_before_thread_sync();\n  __syncthreads();\n  tl::tcgen05_after_thread_sync();\n"
BARE_RENDEZVOUS = "  __syncthreads();\n"
# Per target: the sm103 rendezvous is fenced (publication order 1-2-3); the sm90 one is the bare barrier (2).
MATERIALIZE_BODY = {"sm_103a": MATERIALIZE_FILL + FENCED_RENDEZVOUS, "sm_90a": MATERIALIZE_FILL + BARE_RENDEZVOUS,
                    "sm_120a": MATERIALIZE_FILL + BARE_RENDEZVOUS}
PUBLICATION_ORDER = {"sm_103a": 123, "sm_90a": 2, "sm_120a": 2}

HOST_RUNTIME = """
#define __device__
#include <cstdint>
static const void** watched_view;
static void* watched_scratch;
static int publication_order;
static int early_view;
static int values_at_publication;
static void publication_event(int phase) {
  publication_order = publication_order * 10 + phase;
  if (watched_view && *watched_view) early_view = 1;
}
namespace tl {
void tcgen05_before_thread_sync() { publication_event(1); }
void tcgen05_after_thread_sync() { publication_event(3); }
}
void __syncthreads() {
  publication_event(2);
  const float* values = static_cast<const float*>(watched_scratch);
  values_at_publication = values[0] == 11 && values[1] == 37 &&
                          values[2] == 23 && values[3] == 41;
}
extern "C" void watch_publication(const void** view, void* scratch) {
  watched_view = view;
  watched_scratch = scratch;
  publication_order = early_view = values_at_publication = 0;
}
extern "C" int observed_publication_order() { return publication_order; }
extern "C" int observed_early_view() { return early_view; }
extern "C" int observed_values_at_publication() { return values_at_publication; }
"""


def _materialize_kernel(profile):
    parameters = "const float* source_base, void* import_scratch"
    return (f'extern "C" __global__ void backend({parameters});\n'
            f'extern "C" __global__ void __launch_bounds__(32, 1) backend({parameters}) {{\n'
            + MATERIALIZE_BODY[profile] + '}\n')


class TensorSource(ctypes.Structure):
    _fields_ = [("word", ctypes.c_ulonglong)]


def _host_adapter(tmp_path, source):
    """Execute the emitted shell; only the external fill is a test double."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler is required to execute the emitted view shell")
    library = tmp_path / "view_adapter.so"
    compiled = subprocess.run(
        [compiler, "-std=c++17", "-shared", "-fPIC", "-x", "c++", "-",
         "-I", str(Path(__file__).resolve().parents[2] / "src/pipec2/pack/include"), "-o", str(library)],
        input=HOST_RUNTIME + source, text=True, capture_output=True,
    )
    assert compiled.returncode == 0, compiled.stderr
    loaded = ctypes.CDLL(str(library))
    loaded.view_import.argtypes = [TensorSource, ctypes.POINTER(ctypes.c_void_p),
                                  ctypes.c_void_p]
    loaded.view_import.restype = None
    loaded.watch_publication.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    loaded.watch_publication.restype = None
    return loaded




@pytest.mark.parametrize("profile", PROFILES)
def test_alias_shell_preserves_const_view_callable_without_backend(monkeypatch, tmp_path, profile):
    def forbidden_backend(_request):
        pytest.fail("zero-movement view alias must not invoke the TileLang backend")

    monkeypatch.setattr(lower, "_backend", forbidden_backend)
    adapter = lower.emit_adapter(*_view_plan(profile=profile))
    assert set(adapter.defined_symbols) == {"view_import", "view_import_ready", "view_import_done"}
    premises = {item["name"]: item for item in adapter.trusted_premises}
    shell = premises["view_shell_emission"]
    assert shell["view_binding"] == "@export:smem" and shell["fill"] is None
    assert "body_byte_identical" not in shell
    assert "entry_linkage_rewrite" not in premises and "borrowed_base_binding" not in premises
    native = _host_adapter(tmp_path, adapter.source)
    source = (ctypes.c_float * 4)(11, 23, 37, 41)
    scratch = (ctypes.c_float * 4)(-1, -1, -1, -1)
    delivered = ctypes.c_void_p()
    native.view_import(TensorSource(ctypes.addressof(source)), ctypes.byref(delivered), scratch)
    assert delivered.value == ctypes.addressof(source)
    assert list(ctypes.cast(delivered, ctypes.POINTER(ctypes.c_float))[:4]) == [11, 23, 37, 41]
    assert list(scratch) == [-1, -1, -1, -1]


@pytest.mark.parametrize("profile", PROFILES)
def test_materialize_shell_publishes_fill_before_returning_the_bound_view(monkeypatch, tmp_path, profile):
    with monkeypatch.context() as backend_patch:
        calls = _backend(backend_patch, _materialize_kernel(profile))
        adapter = lower.emit_adapter(*_view_plan(transpose=True, threads=32, profile=profile))
    assert len(calls) == 1
    assert set(adapter.defined_symbols) == {"view_import", "view_import_ready", "view_import_done"}
    premises = {item["name"]: item for item in adapter.trusted_premises}
    shell = premises["view_shell_emission"]
    fill = premises["entry_linkage_rewrite"]
    assert shell["view_binding"] == "scratch"
    assert "body_byte_identical" not in shell
    assert fill["body_byte_identical"] is True
    assert fill["scope"] == shell["fill"]["symbol"] == "view_import__fill"
    assert fill["body_digest"] == shell["fill"]["body_digest"] == sha256(
        ("{\n" + MATERIALIZE_BODY[profile] + "}").encode()).hexdigest()
    assert shell["shell_digest"] != fill["body_digest"]
    native = _host_adapter(tmp_path, adapter.source)
    source = (ctypes.c_float * 4)(11, 23, 37, 41)
    scratch = (ctypes.c_float * 4)(-1, -1, -1, -1)
    delivered = ctypes.c_void_p()
    native.watch_publication(ctypes.byref(delivered), scratch)
    native.view_import(TensorSource(ctypes.addressof(source)), ctypes.byref(delivered), scratch)
    assert native.observed_publication_order() == PUBLICATION_ORDER[profile]
    assert native.observed_early_view() == 0
    assert native.observed_values_at_publication() == 1
    assert delivered.value == ctypes.addressof(scratch)
    assert list(ctypes.cast(delivered, ctypes.POINTER(ctypes.c_float))[:4]) == [11, 37, 23, 41]
    assert list(source) == [11, 23, 37, 41]


def test_global_memory_view_remains_outside_the_model():
    raw = _view_fixture()
    raw["imports"][0]["demands"][0]["space"] = "gmem"
    planned, _ = _pointer_plan(raw)
    assert planned.status == "UNSUPPORTED_MODEL"
    assert planned.program is None


def test_materialized_view_cannot_be_rewritten_after_a_participant_returns():
    instance = _instance(_view_fixture())
    address = ["mulc", 4, ["add", ["mulc", 2, ["coord", "participant"]], ["coord", "slot"]]]
    store = {"src": "temp", "dst": "scratch", "width": 2, "address": address}
    prefix = (
        ir.Issue("smem_load_round", {
            "src": "@export:smem", "dst": "temp", "width": 2, "address": address,
        }, 1),
        ir.Issue("smem_store_round", store, 2),
        ir.Issue("cta_rendezvous", {"resource": "barrier"}, 3),
    )
    published = instance.program({"load": ir.ControlGraph(prefix + (ir.Return(),))},
                                 view_binding="scratch")
    assert check.check(instance, published, budget=100_000).status == "PASS"
    # Actor zero may expose the whole view before actor one rewrites its own
    # already-published cells. Even an identical-value write breaks that lease.
    late_write = instance.program({"load": ir.ControlGraph(prefix + (
        ir.Meta("late_writer", ["eq", ["coord", "participant"], ["const", 1]], 4),
        ir.Branch("late_writer", 5, 6),
        ir.Issue("smem_store_round", store, 6),
        ir.Return(),
    ))}, view_binding="scratch")
    assert check.check(instance, late_write, budget=100_000).status == "UNSAFE"


def test_alias_must_match_every_binding_world_or_materialize_one_shared_program():
    raw = _view_fixture()
    domain = _domain("layout", 2)
    raw["bindings"] = [{
        "name": "layout", "sort": "int", "domain": domain, "owner": "load",
        "units": "scalar", "observation_scope": "cta", "lifetime": "tile",
    }]
    scratch = next(lend for lend in raw["imports"][0]["lends"] if lend["name"] == "scratch")
    scratch["extent"] = 32
    demand = raw["imports"][0]["demands"][0]
    demand["components"]["address"] = [
        "add", demand["components"]["address"], ["mulc", 16, ["bind", "layout"]],
    ]
    callable = raw["imports"][0]["callable"]
    callable["params"].insert(1, {"name": "layout", "meaning": "binding", "ctype": "uint32_t"})
    callable["declaration"] = (
        'extern "C" __device__ void view_import(const float*, uint32_t, '
        'const void** dest_view, void*);'
    )
    planned, imp = _pointer_plan(raw)
    assert planned.status == "MODEL_SAT", planned.to_json_obj()
    alias, materialize = planned.provenance["classes"]
    assert alias["name"] == "view_alias" and alias["status"] != "MODEL_SAT"
    assert materialize["name"] == "view_materialize" and materialize["status"] == "MODEL_SAT"
    assert planned.program.view_binding == "scratch"
    instance = _instance(raw)
    checked = check.check(instance, planned.program, budget=100_000)
    assert checked.status == "PASS"
    assert checked.provenance["terminal_deliveries"] == [
        {
            "bindings": {"layout": 0},
            "outputs": [[-1, 0, 0], [-1, 4, 1], [-1, 8, 2], [-1, 12, 3]],
            "view": {"resource": "scratch", "cells": [[0, 0, 0], [4, 1, 1], [8, 2, 2], [12, 3, 3]]},
        },
        {
            "bindings": {"layout": 1},
            "outputs": [[-1, 16, 0], [-1, 20, 1], [-1, 24, 2], [-1, 28, 3]],
            "view": {"resource": "scratch", "cells": [[16, 0, 0], [20, 1, 1], [24, 2, 2], [28, 3, 3]]},
        },
    ]
    replay = plan._oracle(instance, imp, _receipt_candidate(planned.program, checked))
    assert replay["ok"] and replay["checked_positions"] == 8


@pytest.mark.parametrize("profile,atom_name", [("sm_103a", "sw128_k_bf16"), ("sm_90a", "sw128_k_bf16"), ("sm_120a", "sw128_k_bf16")])
def test_sw128_bf16_source_materializes_a_row_major_view(profile, atom_name):
    target = get_profile(profile)
    atom = target.atoms[atom_name]
    (rows,), (columns,) = atom["shape"]
    scalar_bytes = atom["elem_bits"] // 8
    swizzle = atom["swizzle"]
    mask = ((1 << swizzle["B"]) - 1) << (swizzle["M"] + swizzle["S"])
    index = ["add", ["mulc", columns, ["coord", "row"]], ["coord", "slot"]]
    bits = ["tobits", 32, index]
    source_address = ["mulc", scalar_bytes, ["toint", ["bxor", bits, [
        "shrc", ["band", bits, ["cbits", 32, mask]], swizzle["S"],
    ]]]]
    raw = _view_fixture(threads=rows)
    identity = raw["exports"][0]["identity"]
    identity.update(dtype="bf16", coords=[_domain("row", rows), _domain("slot", columns)])
    raw["exports"][0]["placements"][0]["components"]["address"] = source_address
    raw["exports"][0]["backing"][0]["extent"] = rows * columns * scalar_bytes
    for lend in raw["imports"][0]["lends"]:
        if lend["name"] == "temp":
            lend["extent"] = columns
        elif lend["name"] == "scratch":
            lend["extent"] = rows * columns * scalar_bytes
    imp = raw["imports"][0]
    imp["demands"][0]["domain"] = identity["coords"]
    imp["demands"][0]["components"]["address"] = ["mulc", scalar_bytes, index]
    imp["callable"]["params"][0]["ctype"] = "const uint16_t*"
    imp["callable"]["declaration"] = VIEW_DECLARATION.replace("float", "uint16_t")
    planned, imported = _pointer_plan(raw, target)
    assert planned.status == "MODEL_SAT", planned.to_json_obj()
    alias, materialize = planned.provenance["classes"]
    assert alias["name"] == "view_alias" and alias["status"] != "MODEL_SAT"
    assert materialize["name"] == "view_materialize" and materialize["status"] == "MODEL_SAT"
    assert planned.program.view_binding == "scratch"
    instance = _instance(raw, target)
    checked = check.check(instance, planned.program, budget=100_000)
    assert checked.status == "PASS"
    assert checked.provenance["terminal_deliveries"] == [{
        "bindings": {},
        "outputs": [[-1, 2 * identity, identity] for identity in range(512)],
        "view": {"resource": "scratch",
                 "cells": [[2 * identity, identity, identity] for identity in range(512)]},
    }]
    replay = plan._oracle(instance, imported, _receipt_candidate(planned.program, checked))
    assert replay["ok"] and replay["checked_positions"] == 512


def _register_view_fixture():
    """A register export demanded as a shared view: only a store+rendezvous route exists."""
    raw = _pointer_fixture(space="reg", threads=2, columns=2)
    imp = raw["imports"][0]
    imp["demands"] = [{
        "kind": "memory_view", "domain": [_domain("r", 2), _domain("c", 2)],
        "guard": None, "identity_coords": {"r": ["coord", "r"], "c": ["coord", "c"]},
        "space": "smem", "units": "byte",
        "components": {"address": ["mulc", 4, ["add", ["mulc", 2, ["coord", "r"]], ["coord", "c"]]]},
    }]
    imp["callable"] = {
        "symbol": "view_import", "declaration": VIEW_DECLARATION,
        "events": {"ready": "view_import_ready", "done": "view_import_done"},
        "params": [
            {"name": "source_base", "meaning": "source_base", "ctype": "pipelink_tensor_source"},
            {"name": "dest_view", "meaning": "dest_view", "ctype": "const void**"},
            {"name": "import_scratch", "meaning": "scratch", "ctype": "void*"},
        ],
    }
    imp["lends"] = [lend for lend in imp["lends"] if lend["domain"] != "smem"]
    return raw


def test_register_view_without_a_shared_lend_is_a_resource_envelope_verdict():
    raw = _register_view_fixture()
    planned, _imp = _pointer_plan(raw)
    assert planned.status == "RESOURCE_ENVELOPE"
    assert planned.provenance["classes"] == []
    (envelope,) = planned.provenance["resource_envelope"]
    assert envelope["route"]["label"] == "view_materialize"
    assert "lend:smem" in envelope["missing"]
    # Supplying exactly the missing facts admits the route (T4) and it certifies.
    raw["imports"][0]["lends"].append(_resource("scratch", "smem", 16))
    raw["sites"][0]["guarantees"] += ["converged", "no_early_return"]
    planned, _imp = _pointer_plan(raw)
    assert planned.status == "MODEL_SAT", planned.to_json_obj()
    assert planned.provenance["classes"][planned.provenance["selected"]]["name"] == "view_materialize"
    assert planned.provenance["resource_envelope"] == []
