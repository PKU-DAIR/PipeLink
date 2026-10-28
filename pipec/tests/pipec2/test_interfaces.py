"""Entry rendezvous (L-JOIN) and callable interface (L-CALL) link rules."""
from __future__ import annotations

from dataclasses import replace

import pytest

import pipec2.link as link_module
from pipec2.decls import declaration_digest
from pipec2.interfaces import check_calls, check_entry_rendezvous, launch_threads
from pipec2.layout import load_layout
from pipec2.model import (
    CallDecl, CallInterface, CallParam, CallSite, Claim, Cohort, Collectives, EntryRendezvous, Endpoint, Export,
    Extent, LinkError, Protocol, Role, Tile,
)
from pipec2.profile import get_profile

from test_link import _stage


PROFILE = get_profile("sm_90a")
REG = load_layout({"encoding": "pl-layout-v2", "elem_bits": 64, "shape": [[1]], "stride": [[1]]}, "reg", "$.layout")
SWIZZLED = {"encoding": "pl-layout-v2", "elem_bits": 16, "shape": [[8], [64]], "stride": [[64], [1]],
            "swizzle": {"B": 3, "M": 4, "S": 3, "unit": "byte"}}
LINEAR = {"encoding": "pl-layout-v2", "elem_bits": 16, "shape": [[8], [64]], "stride": [[64], [1]]}


def _decl(symbol: str) -> str:
    return f'extern "C" __device__ void {symbol}(const void*);'


def _param(space="smem", access="r", dtype="bf16", extent=512, layout=None) -> CallParam:
    return CallParam(0, space, access, dtype, extent,
                     None if layout is None else load_layout(layout, space, "$.layout"))


def _interface(threads: int, kind: str = "none", handle: str | None = None, params=None) -> CallInterface:
    return CallInterface(Cohort(threads), Collectives(kind, handle), tuple(params if params is not None else [_param()]))


def _site(threads: int, params=None) -> CallSite:
    return CallSite(Cohort(threads), tuple(params if params is not None else [_param()]))


def _call(name: str, direction: str, export: str, *, role: str | None = None, site: CallSite | None = None,
          exports: list[str] | None = None) -> Endpoint:
    symbol = f"pipelink_call_{name}" if direction == "produce" else f"impl_{export}"
    decl = _decl(symbol)
    return Endpoint(name, direction, role, "reg", "cta", "generic", Tile("u64", (1,)), REG, None,
                    Protocol("call", None, None, None, None,
                             CallDecl(export, "void(const void*)", decl, declaration_digest(decl), site), None, None),
                    {}, exports)


def _export(name: str, interface: CallInterface | None) -> Export:
    decl = _decl(f"impl_{name}")
    return Export(f"impl_{name}", decl, declaration_digest(decl), interface)


def _callee(stage_id: str, interfaces: dict[str, CallInterface | None], nested=()) -> object:
    """A role-less stage providing `interfaces`, with optional nested call endpoints."""
    return replace(_stage(stage_id), exports={name: _export(name, item) for name, item in interfaces.items()},
                   endpoints=[*(_call(f"{name}", "consume", name) for name in interfaces), *nested])


def _caller(stage_id: str, roles: dict[str, int], calls) -> object:
    return replace(_stage(stage_id), roles={name: Role(f"{stage_id.replace('.', '_')}_{name}", warps)
                                            for name, warps in roles.items()}, endpoints=list(calls))


def _check(stages, threads: int | None = None) -> None:
    check_calls(link_module._pair_channels(stages), threads or launch_threads(stages), PROFILE, stages)


def _failures(stages, threads: int | None = None) -> list[tuple[str, str]]:
    with pytest.raises(LinkError) as caught:
        _check(stages, threads)
    return [(failure.clause_id, failure.path) for failure in caught.value.failures]


# --- Entry rendezvous -------------------------------------------------------------------------


def _role_stage(stage_id: str, warps: int, join: EntryRendezvous | None):
    return replace(_stage(stage_id), roles={"r" + stage_id[-1]: Role("e", warps)}, entry_rendezvous=join)


def test_entry_rendezvous_agreeing_over_the_launch_links() -> None:
    join = EntryRendezvous(0, 384)
    stages = [_role_stage("org.a", 4, join), _role_stage("org.b", 8, join), _stage("org.lib")]
    check_entry_rendezvous(stages, launch_threads(stages))
    check_entry_rendezvous([_role_stage("org.a", 4, None), _role_stage("org.b", 8, None)], 384)


def test_entry_rendezvous_disagreement_and_absence_reject() -> None:
    stages = [_role_stage("org.a", 4, EntryRendezvous(0, 384)), _role_stage("org.b", 8, EntryRendezvous(0, 256)),
              _role_stage("org.c", 0, None)]
    with pytest.raises(LinkError) as caught:
        check_entry_rendezvous(stages, 384)
    assert [(item.clause_id, item.path) for item in caught.value.failures] == [
        ("L-JOIN", f"stages.org.{name}.stage.entry_rendezvous") for name in "abc"]


def test_entry_rendezvous_must_span_the_solved_launch() -> None:
    join = EntryRendezvous(0, 256)
    stages = [_role_stage("org.a", 4, join), _role_stage("org.b", 8, join)]
    with pytest.raises(LinkError) as caught:
        check_entry_rendezvous(stages, launch_threads(stages))
    assert {(item.clause_id, item.path) for item in caught.value.failures} == {
        ("L-JOIN", "stages.org.a.stage.entry_rendezvous.threads"),
        ("L-JOIN", "stages.org.b.stage.entry_rendezvous.threads")}


def test_role_less_stage_cannot_declare_entry_rendezvous() -> None:
    from pipec2.model import SchemaError
    from pipec2.schema import _entry_rendezvous, load_stage_manifest
    from test_schema import _valid_stage

    raw = _valid_stage()
    raw["stage"]["entry_rendezvous"] = {"barrier": 0, "threads": 128}
    assert load_stage_manifest(raw).entry_rendezvous == EntryRendezvous(0, 128)
    with pytest.raises(SchemaError) as caught:
        _entry_rendezvous({"barrier": 0, "threads": 128}, "$.stage.entry_rendezvous", {})
    assert [(item.clause_id, item.path) for item in caught.value.failures] == [
        ("L-JOIN", "$.stage.entry_rendezvous")]


# --- Callable interfaces ----------------------------------------------------------------------


def test_matching_interface_and_sub_cohort_site_link() -> None:
    caller = _caller("org.a", {"work": 4}, [_call("e1", "produce", "e1", role="work", site=_site(128))])
    middle = _callee("org.b", {"e1": _interface(128)},
                     nested=[_call("e2", "produce", "e2", site=_site(32), exports=["e1"])])
    inner = _callee("org.c", {"e2": _interface(32, "warp")})
    _check([caller, middle, inner])


def test_interface_stated_on_one_side_only_rejects() -> None:
    caller = _caller("org.a", {"work": 4}, [_call("e1", "produce", "e1", role="work")])
    assert _failures([caller, _callee("org.b", {"e1": _interface(128)})]) == [("L-CALL", "channels.e1.protocol.site")]
    caller = _caller("org.a", {"work": 4}, [_call("e1", "produce", "e1", role="work", site=_site(128))])
    assert _failures([caller, _callee("org.b", {"e1": None})]) == [
        ("L-CALL", "stages.org.b.exports.e1.interface")]


def test_cohort_mismatch_rejects() -> None:
    caller = _caller("org.a", {"work": 4}, [_call("e1", "produce", "e1", role="work", site=_site(128))])
    assert ("L-CALL", "channels.e1.protocol.site.cohort.threads") in _failures(
        [caller, _callee("org.b", {"e1": _interface(64)})])


@pytest.mark.parametrize("site, field", [
    (_param(extent=256), "extent"),
    (_param(space="gmem"), "space"),
    (_param(dtype="f16"), "dtype"),
    (_param(access="r"), "access"),
    (_param(layout=LINEAR), "layout"),
])
def test_parameter_mismatch_rejects(site, field) -> None:
    export = _param(access="rw" if field == "access" else "r", layout=SWIZZLED if field == "layout" else None)
    if field == "layout":
        site = replace(site, access="r")
    caller = _caller("org.a", {"work": 4}, [_call("e1", "produce", "e1", role="work", site=_site(128, [site]))])
    callee = _callee("org.b", {"e1": _interface(128, params=[export])})
    assert _failures([caller, callee]) == [("L-CALL", f"channels.e1.protocol.site.params[0].{field}")]


def test_equal_layouts_and_wider_caller_provision_link() -> None:
    site = _param(access="rw", extent=1024, layout=SWIZZLED)
    caller = _caller("org.a", {"work": 4}, [_call("e1", "produce", "e1", role="work", site=_site(128, [site]))])
    _check([caller, _callee("org.b", {"e1": _interface(128, params=[_param(layout=SWIZZLED)])})])


def test_warp_collective_with_sixteen_threads_rejects_at_load() -> None:
    from pipec2.model import SchemaError
    from pipec2.schema import _call_interface

    with pytest.raises(SchemaError) as caught:
        _call_interface({"cohort": {"threads": 16}, "collectives": {"kind": "warp"}, "params": []}, "$.i")
    assert [(item.clause_id, item.path) for item in caught.value.failures] == [("L-CALL", "$.i.cohort.threads")]


def test_cta_collective_requires_the_launch_cohort() -> None:
    caller = _caller("org.a", {"work": 4, "other": 4}, [_call("e1", "produce", "e1", role="work", site=_site(128))])
    assert _failures([caller, _callee("org.b", {"e1": _interface(128, "cta")})]) == [
        ("L-CALL", "stages.org.b.exports.e1.interface.cohort.threads")]


def test_nested_site_cohort_divides_enclosing_cohort() -> None:
    caller = _caller("org.a", {"work": 4}, [_call("e1", "produce", "e1", role="work", site=_site(128))])
    middle = _callee("org.b", {"e1": _interface(128)},
                     nested=[_call("e2", "produce", "e2", site=_site(96), exports=["e1"])])
    inner = _callee("org.c", {"e2": _interface(96)})
    assert ("L-CALL", "channels.e2.protocol.site.cohort.threads") in _failures([caller, middle, inner])


def test_call_cycle_rejects() -> None:
    # role -> B.e1 -> C.e2 -> B.e1 (through channel `back`).
    caller = _caller("org.a", {"work": 4}, [_call("e1", "produce", "e1", role="work", site=_site(128))])
    b = _callee("org.b", {"e1": _interface(128)}, nested=[_call("e2", "produce", "e2", site=_site(128), exports=["e1"])])
    c = _callee("org.c", {"e2": _interface(128)})
    _check([caller, b, c])
    b = replace(b, endpoints=[*b.endpoints, _call("back", "consume", "e1")])
    c = replace(c, endpoints=[*c.endpoints, _call("back", "produce", "e1", site=_site(128), exports=["e2"])])
    assert ("L-CALL", "channels.back") in _failures([caller, b, c])


# --- Barrier-bearing exclusivity: the plan's witnesses ----------------------------------------


def _named_callee(threads: int):
    stage = _callee("org.lib", {"sync": _interface(threads, "named", "pipelink_lib_bar")})
    return stage


def test_two_sub_cohort_sites_calling_a_named_barrier_callee_reject() -> None:
    caller = _caller("org.a", {"work": 4}, [
        _call("sync_lo", "produce", "sync", role="work", site=_site(64)),
        _call("sync_hi", "produce", "sync", role="work", site=_site(64)),
    ])
    callee = _named_callee(64)
    callee = replace(callee, endpoints=[_call("sync_lo", "consume", "sync"), _call("sync_hi", "consume", "sync")])
    assert set(_failures([caller, callee])) == {
        ("L-CALL", "channels.sync_lo.protocol.site.cohort"), ("L-CALL", "channels.sync_hi.protocol.site.cohort")}


def test_nested_sub_cohort_wrapper_reaching_a_named_barrier_rejects() -> None:
    # A 128-thread export invokes a 64-thread wrapper (twice, one per half); the wrapper syncs a
    # named barrier: two live instances of one barrier.
    caller = _caller("org.a", {"work": 4}, [_call("outer", "produce", "outer", role="work", site=_site(128))])
    middle = _callee("org.b", {"outer": _interface(128)},
                     nested=[_call("wrap", "produce", "wrap", site=_site(64), exports=["outer"])])
    wrapper = _callee("org.w", {"wrap": _interface(64)},
                      nested=[_call("sync", "produce", "sync", site=_site(64), exports=["wrap"])])
    callee = _named_callee(64)
    assert ("L-CALL", "channels.wrap.protocol.site.cohort") in _failures([caller, middle, wrapper, callee])


def test_two_roles_reaching_one_barrier_bearing_export_reject() -> None:
    caller = _caller("org.a", {"left": 4, "right": 4}, [
        _call("left_call", "produce", "outer", role="left", site=_site(128)),
        _call("right_call", "produce", "outer", role="right", site=_site(128)),
    ])
    middle = _callee("org.b", {"outer": _interface(128)},
                     nested=[_call("sync", "produce", "sync", site=_site(128), exports=["outer"])])
    middle = replace(middle, endpoints=[_call("left_call", "consume", "outer"), _call("right_call", "consume", "outer"),
                                        *middle.endpoints[1:]])
    assert ("L-CALL", "stages.org.b.exports.outer") in _failures([caller, middle, _named_callee(128)])


def test_two_roles_reaching_two_exports_sharing_a_named_barrier_reject() -> None:
    # Each export has one root, but both synchronize the library's single claimed barrier.
    caller = _caller("org.a", {"left": 4, "right": 4}, [
        _call("sync_a", "produce", "sync_a", role="left", site=_site(128)),
        _call("sync_b", "produce", "sync_b", role="right", site=_site(128)),
    ])
    callee = _callee("org.lib", {"sync_a": _interface(128, "named", "pipelink_lib_bar"),
                                 "sync_b": _interface(128, "named", "pipelink_lib_bar")})
    failures = _failures([caller, callee])
    assert len(failures) == 1 and failures[0][1].startswith("claims.")


def _barrier_claim(handle: str, alias: str | None = None) -> Claim:
    return Claim(domain="named_barrier", handle=handle, lifetime="kernel", extent=Extent(1, {}, {}), align=1,
                 offset=None, roles=None, reg_group=None, values=None, absolute_base=None, alias=alias)


@pytest.mark.parametrize("handles, aliases", [
    (("pipelink_bar", "pipelink_bar"), (None, None)),            # one global handle in two libraries
    (("pipelink_bar_a", "pipelink_bar_b"), ("shared", "shared")),  # two handles aliased to one barrier
    (("pipelink_bar", "pipelink_bar"), ("shared", None)),        # one shared handle, alias on one side only
])
def test_two_libraries_sharing_one_physical_barrier_reject(handles, aliases) -> None:
    caller = _caller("org.a", {"left": 4, "right": 4}, [
        _call("sync_a", "produce", "sync_a", role="left", site=_site(128)),
        _call("sync_b", "produce", "sync_b", role="right", site=_site(128)),
    ])
    libraries = [replace(_callee(stage, {export: _interface(128, "named", handle)}),
                         claims=[_barrier_claim(handle, alias)])
                 for stage, export, handle, alias in zip(("org.lib_a", "org.lib_b"), ("sync_a", "sync_b"),
                                                         handles, aliases)]
    failures = _failures([caller, *libraries])
    assert len(failures) == 1 and failures[0][1].startswith("claims.")


def test_handle_alias_bridge_joins_three_libraries_into_one_barrier() -> None:
    # lib_a and lib_b share handle h; lib_b and lib_c share alias x; so a and c share one barrier.
    caller = _caller("org.a", {"left": 4, "right": 4}, [
        _call("sync_a", "produce", "sync_a", role="left", site=_site(128)),
        _call("sync_c", "produce", "sync_c", role="right", site=_site(128)),
    ])
    lib_a = replace(_callee("org.lib_a", {"sync_a": _interface(128, "named", "h")}), claims=[_barrier_claim("h")])
    lib_b = replace(_callee("org.lib_b", {}), claims=[_barrier_claim("h", "x")])
    lib_c = replace(_callee("org.lib_c", {"sync_c": _interface(128, "named", "k")}),
                    claims=[_barrier_claim("k", "x")])
    failures = _failures([caller, lib_a, lib_b, lib_c])
    assert len(failures) == 1 and failures[0][1].startswith("claims.")


def test_distinct_named_barriers_from_two_roles_link() -> None:
    caller = _caller("org.a", {"left": 4, "right": 4}, [
        _call("sync_a", "produce", "sync_a", role="left", site=_site(128)),
        _call("sync_b", "produce", "sync_b", role="right", site=_site(128)),
    ])
    lib_a = replace(_callee("org.lib_a", {"sync_a": _interface(128, "named", "h")}), claims=[_barrier_claim("h")])
    lib_b = replace(_callee("org.lib_b", {"sync_b": _interface(128, "named", "k")}), claims=[_barrier_claim("k")])
    _check([caller, lib_a, lib_b])


def test_full_cohort_single_role_barrier_path_links() -> None:
    caller = _caller("org.a", {"work": 4}, [_call("outer", "produce", "outer", role="work", site=_site(128))])
    middle = _callee("org.b", {"outer": _interface(128)},
                     nested=[_call("sync", "produce", "sync", site=_site(128), exports=["outer"])])
    _check([caller, middle, _named_callee(128)])


def test_named_collective_handle_is_a_callee_claim() -> None:
    from pipec2.model import SchemaError
    from pipec2.schema import load_stage_manifest
    from test_schema import _valid_stage

    raw = _valid_stage()
    interface = {"cohort": {"threads": 128}, "collectives": {"kind": "named", "handle": "pipelink_named_id"},
                 "params": [{"index": 0, "space": "smem", "access": "r", "dtype": "bf16", "extent": 64,
                             "layout": LINEAR}]}
    raw["stage"]["exports"]["register_compute"]["interface"] = interface
    stage = load_stage_manifest(raw)
    assert stage.exports["register_compute"].interface.collectives == Collectives("named", "pipelink_named_id")
    interface["collectives"]["handle"] = "pipelink_unclaimed"
    with pytest.raises(SchemaError) as caught:
        load_stage_manifest(raw)
    assert [(item.clause_id, item.path) for item in caught.value.failures] == [
        ("L-CALL", "$.stage.exports.register_compute.interface.collectives.handle")]


# --- Manifest boundary ------------------------------------------------------------------------


def _raw_with_nested_call():
    from test_schema import _valid_stage

    raw = _valid_stage()
    caller = next(item for item in raw["endpoints"] if item["name"] == "register_call")
    nested = {**caller, "name": "nested_call", "owner_role": None, "exports": ["register_compute"],
              "protocol": {**caller["protocol"], "site": {"cohort": {"threads": 32}, "params": []}}}
    raw["endpoints"].append(nested)
    return raw, nested


def test_nested_call_endpoint_loads_with_exports_and_site() -> None:
    from pipec2.schema import load_stage_manifest

    raw, _ = _raw_with_nested_call()
    endpoint = next(item for item in load_stage_manifest(raw).endpoints if item.name == "nested_call")
    assert (endpoint.owner_role, endpoint.exports, endpoint.protocol.call.site) == (
        None, ["register_compute"], CallSite(Cohort(32), ()))


@pytest.mark.parametrize("mutation, path", [
    ("no_exports", "$.endpoints[{index}].exports"),
    ("role_with_exports", "$.endpoints[{index}].exports"),
    ("site_on_provided", "$.endpoints[{index}].protocol.site"),
    ("unknown_export", "$.endpoints[{index}].exports"),
])
def test_call_endpoint_boundary_rejects(mutation, path) -> None:
    from pipec2.model import SchemaError
    from pipec2.schema import load_stage_manifest

    raw, nested = _raw_with_nested_call()
    index = raw["endpoints"].index(nested)
    if mutation == "no_exports":
        del nested["exports"]
    elif mutation == "role_with_exports":
        nested["owner_role"] = "load"
    elif mutation == "site_on_provided":
        nested.update(direction="consume", owner_role=None)
        del nested["exports"]
    else:
        nested["exports"] = ["missing"]
    with pytest.raises(SchemaError) as caught:
        load_stage_manifest(raw)
    assert [item.path for item in caught.value.failures] == [path.format(index=index)]


def test_topology_is_optional_only_for_role_less_stages() -> None:
    from pipec2.model import SchemaError
    from pipec2.schema import load_stage_manifest
    from test_schema import _valid_stage

    raw = _valid_stage()
    del raw["topology"]
    with pytest.raises(SchemaError) as caught:
        load_stage_manifest(raw)
    assert [(item.clause_id, item.path) for item in caught.value.failures] == [("L-TOPO", "$.topology")]


def test_types_load_once_and_reject_overlapping_members() -> None:
    from pipec2.model import SchemaError, TypeDecl, TypeMember
    from pipec2.schema import load_stage_manifest
    from test_schema import _valid_stage

    raw = _valid_stage()
    raw["types"] = [{"name": "Pair", "bytes": 8, "align": 4, "members": [
        {"name": "a", "ctype": "int32_t", "extent": 0, "offset": 0, "bytes": 4, "align": 4},
        {"name": "b", "ctype": "float", "extent": 0, "offset": 4, "bytes": 4, "align": 4}]}]
    assert load_stage_manifest(raw).types == (TypeDecl("Pair", 8, 4, (
        TypeMember("a", "int32_t", 0, 0, 4, 4), TypeMember("b", "float", 0, 4, 4, 4))),)
    raw["types"][0]["members"][1]["offset"] = 2
    with pytest.raises(SchemaError) as caught:
        load_stage_manifest(raw)
    assert [(item.clause_id, item.path) for item in caught.value.failures] == [("L-SYM", "$.types[0].members[1]")]


def test_nested_call_sites_leave_with_their_unreferenced_exports() -> None:
    # Nobody calls B.e1, so B's nested call into C.e2 leaves, and then C.e2's channel too.
    b = _callee("org.b", {"e1": _interface(128)}, nested=[_call("e2", "produce", "e2", site=_site(32), exports=["e1"])])
    c = _callee("org.c", {"e2": _interface(32)})
    pruned, dropped = link_module._prune_unreferenced_exports([b, c])
    assert dropped == [
        {"stage": "org.b", "endpoint": "e1", "reason": "uncalled export channel"},
        {"stage": "org.b", "endpoint": "e2", "reason": "owned by unreferenced exports"},
        {"stage": "org.c", "endpoint": "e2", "reason": "uncalled export channel"},
    ]
    assert all(not stage.endpoints for stage in pruned)
