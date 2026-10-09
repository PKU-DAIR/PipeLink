from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from pipec2.decls import declaration_digest, normalize_declaration
from pipec2.layout import canonicalize, unify
from pipec2.model import (
    Arrive,
    BarrierSide,
    BarrierStorage,
    CallDecl,
    Claim,
    CanonEndpoint,
    CounterDecl,
    Export,
    Extent,
    Handle,
    Init,
    NamedBarrierDecl,
    Protocol,
    SchemaError,
    SharedMember,
    SolveError,
    TxSum,
)
from pipec2.profile import get_profile
from pipec2.protocol import check_init_warps, expand
from pipec2.schema import load_pipeline, load_stage_manifest, validate_assembly
from pipec2.solve import solve


ZERO_HASH = "0" * 64




def _abi() -> dict[str, Any]:
    return {
        "schema": "2.0-draft",
        "target": {"profile": "sm_103a", "lto_arch": "lto_103", "toolkit": "cuda-13.3.0"},
    }


def _endpoint(direction: str, role: str) -> dict[str, Any]:
    return {
        "name": "ring",
        "direction": direction,
        "owner_role": role,
        "space": "smem",
        "scope": "cta",
        "proxy": "async",
        "tile": {"dtype": "bf16", "shape": [4, 4]},
        "layout": {
            "encoding": "pl-layout-v2",
            "elem_bits": 16,
            "shape": [[4], [4]],
            "stride": [[4], [1]],
        },
        "ring": {"depth": {"handle": "pipelink_depth"}, "stage_stride_bytes": 32},
        "protocol": {
            "kind": "mbarrier",
            "full": {
                "slots": "pipelink_full",
                "arrive": {"count": 1, "by": "load"},
                "tx_bytes": "tile",
            },
            "empty": {
                "slots": "pipelink_empty",
                "arrive": {"count": "threads(mma)", "by": "mma"},
                "tx_bytes": 0,
            },
            "init": {"owner": "load", "materializer": "dispatch", "phase": 0},
            "release": "once-per-acquire",
        },
        "binds": {"full": "pipelink_full", "empty": "pipelink_empty"},
    }


def _stage(stage_id: str, role: str, endpoint: dict[str, Any], claims: list[dict[str, Any]]) -> dict[str, Any]:
    prefix = stage_id.replace(".", "_") + "_"
    return {
        "pipelink_abi": _abi(),
        "stage": {
            "id": stage_id,
            "version": "1.0.0",
            "symbol_prefix": prefix,
            "entry_abi": "v2",
            "roles": {role: {"entry": f"{prefix}role_{role}", "warps": 4}},
            "exports": {},
            "entry_signature": "void(const void* params, unsigned variant)",
            "variants": [{"key": {}, "index": 0}],
        },
        "endpoints": [endpoint],
        "claims": claims,
        "topology": {
            "cta_group": {"fixed": 1},
            "cluster": {"compiled": [1, 1, 1], "launch": {"x_multiple_of": 1, "max": [1, 1, 1]}},
            "multicast": [],
            "persistent": False,
            "scheduler": "static",
        },
        "params": {
            "bytes": 16,
            "align": 16,
            "type": f"{prefix}Params",
            "sha256_of_layout_probe": ZERO_HASH,
        },
    }


def _fixture():
    producer_raw = _stage(
        "alpha.stage",
        "load",
        _endpoint("produce", "load"),
        [
            {
                "domain": "mbarrier",
                "handle": "pipelink_full",
                "extent": {"const": 0, "per": {"pipelink_depth": 1}},
                "align": 1,
            },
            {
                "domain": "mbarrier",
                "handle": "pipelink_empty",
                "extent": {"const": 0, "per": {"pipelink_depth": 1}},
                "align": 1,
            },
        ],
    )
    consumer_raw = _stage("beta.stage", "mma", _endpoint("consume", "mma"), [])
    pipeline_raw = {
        "pipelink_abi": _abi(),
        "pipeline": {
            "name": "protocol.fixture",
            "stages": [
                {"artifact_path": "alpha.pla", "stage_id": "alpha.stage"},
                {"artifact_path": "beta.pla", "stage_id": "beta.stage"},
            ],
            "handles": {
                "pipelink_depth": {
                    "class": "depth",
                    "binding": "bind",
                    "values": [2, 4],
                    "owner": "channel:ring",
                },
                "pipelink_empty": {
                    "class": "base",
                    "binding": "bind",
                    "domain": "mbarrier",
                    "owner": "claim:empty",
                },
                "pipelink_full": {
                    "class": "base",
                    "binding": "bind",
                    "domain": "mbarrier",
                    "owner": "claim:full",
                },
            },
            "phases": ["steady"],
            "exclusive": [],
            "domains": {
                "gmem_counter": {"capacity": 16},
                "smem_page": {"capacity": 2, "page_bytes": 4096},
            },
            "depth_priority": ["pipelink_depth"],
            "regs_priority": [],
            "forced_bindings": {},
            "topology": {"persistent": False, "scheduler": "static"},
        },
    }
    stages = [load_stage_manifest(producer_raw), load_stage_manifest(consumer_raw)]
    pipeline = load_pipeline(pipeline_raw)
    assert validate_assembly(pipeline, stages) == []
    canon = {
        "alpha.stage:ring:produce": _canon("ring", "produce", 256),
        "beta.stage:ring:consume": _canon("ring", "consume", 256),
    }
    return stages, pipeline, canon


def _canon(name: str, direction: str, footprint_bits: int) -> CanonEndpoint:
    del name, direction
    return CanonEndpoint(
        ("affine",),
        16,
        None,
        8,
        footprint_bits,
        ("pipelink_depth", 32),
        "smem",
        "cta",
        "async",
        "bf16",
        (4, 4),
        None,
    )


def _replace_channel_endpoint(stages, stage_id: str, endpoint):
    return [
        replace(stage, endpoints=[endpoint]) if stage.id == stage_id else stage
        for stage in stages
    ]


def _assert_proto_failure(call) -> None:
    with pytest.raises(SolveError) as exc:
        call()
    assert {failure.clause_id for failure in exc.value.failures} == {"L-PROTO"}


def test_schema_rejects_historical_entry_abi() -> None:
    raw = _stage("alpha.stage", "load", _endpoint("produce", "load"), [])
    raw["stage"]["entry_abi"] = "v1-compat"
    with pytest.raises(SchemaError) as caught:
        load_stage_manifest(raw)
    assert [(failure.clause_id, failure.path) for failure in caught.value.failures] == [
        ("L-ENTRYABI", "$.stage.entry_abi")
    ]


def test_declaration_identity_preserves_tokens_and_literal_contents() -> None:
    text = (
        ' /* preamble */ typedef\tint abi_token;\n'
        'extern  "C" void invoke(const char* value = " // literal  /* text */ ", '
        "char marker = '\\''); // trailing comment\n"
    )
    normalized = (
        'typedef int abi_token; extern "C" void invoke(const char* value = " // literal  /* text */ ", '
        "char marker = '\\'');"
    )
    assert declaration_digest(text) == declaration_digest(normalized)
    assert declaration_digest(text) != declaration_digest(text.replace("abi_token", "abi_other"))
    assert declaration_digest(text) != declaration_digest(text.replace("literal  ", "literal "))
    raw_literal = 'R"tag(quoted " //  text /* literal */)tag"'
    assert normalize_declaration(" \n" + raw_literal + "\t// ignored") == raw_literal


@pytest.mark.parametrize("alias", [
    "alpha.stage:ring", "ring:produce", "ring:producer", "producer", "produce",
])
def test_protocol_requires_canonical_endpoint_keys(alias: str) -> None:
    stages, pipeline, canon = _fixture()
    canon[alias] = canon.pop("alpha.stage:ring:produce")
    with pytest.raises(SolveError) as caught:
        expand("ring", stages[0].endpoints[0], stages[1].endpoints[0], stages, pipeline, canon, get_profile("sm_103a"))
    assert [(failure.clause_id, failure.path) for failure in caught.value.failures] == [
        ("L-PROTO", "channels.ring.produce.layout")
    ]


def test_mbarrier_expands_tile_bytes_arrivals_and_single_init_materializer() -> None:
    stages, pipeline, canon = _fixture()
    producer, consumer = stages[0].endpoints[0], stages[1].endpoints[0]
    result = expand("ring", producer, consumer, stages, pipeline, canon, get_profile("sm_103a"))

    assert result.tx_bytes_producer == result.tx_bytes_consumer == 32
    assert result.arrive_full == 1
    assert result.arrive_empty == 128
    assert result.init_owner == "load"
    assert result.init_materializer == "dispatch"
    assert result.init_slots == ("pipelink_full", "pipelink_empty")
    assert result.shared_group is None


def _warp_fixture():
    stages, pipeline, canon = _fixture()
    for index, (role, base) in enumerate((("load", 4), ("mma", 0))):
        handle = f"pipelink_warp_{role}"
        claim = Claim("warp", handle, "kernel", Extent(4, {}, {}), 1, None, [role], None, None, base, None)
        stages[index] = replace(stages[index], claims=[*stages[index].claims, claim])
        pipeline = replace(pipeline, handles={
            **pipeline.handles, handle: Handle(handle, "base", "bind", "warp", None, f"role:{role}"),
        })
    return stages, pipeline, canon


def _with_inits(stages, *inits):
    return [
        replace(stage, endpoints=[replace(
            stage.endpoints[0], protocol=replace(stage.endpoints[0].protocol, init=init),
        )])
        for stage, init in zip(stages, inits, strict=True)
    ]


def test_initializing_warp_is_checked_against_solved_owner_binding() -> None:
    stages, pipeline, canon = _warp_fixture()
    profile = get_profile("sm_103a")
    expansion = expand("ring", stages[0].endpoints[0], stages[1].endpoints[0], stages, pipeline, canon, profile)
    solution = solve(pipeline, stages, canon, [expansion], profile)
    assert solution.warp_map["load"] == (4, 8)

    def annotated(warp):
        endpoint = stages[1].endpoints[0]
        endpoint = replace(endpoint, protocol=replace(endpoint.protocol, init=replace(endpoint.protocol.init, warp=warp)))
        return [stages[0], replace(stages[1], endpoints=[endpoint])]

    # The consumer names an init owner in its partner stage; physical membership is half-open.
    assert check_init_warps(annotated(4), solution.warp_map) == {"ring": "load"}
    assert check_init_warps(annotated(7), solution.warp_map) == {"ring": "load"}
    for warp in (3, 8):
        with pytest.raises(SolveError) as caught:
            check_init_warps(annotated(warp), solution.warp_map)
        failure = caught.value.failures[0]
        assert failure.clause_id == "L-PROTO"
        assert failure.detail["warp"] == warp
        if warp == 3:
            assert failure.detail["owner"] == "load"
            assert failure.detail["derived_owner"] == "mma"


@pytest.mark.parametrize("producer_init, consumer_init", [
    (Init(None, 0, "stage", 7), Init(None, 0, "stage", 7)),
    (Init("load", 0, "stage", 7), Init(None, 0, "stage", 7)),
    (Init(None, 0, "dispatch", 7), Init(None, 0, "dispatch", 7)),
    (Init("load", 0, "stage"), Init(None, 0, "stage", 7)),
    (Init(None, 0, "stage", 7), Init("load", 0, "stage")),
])
def test_initializing_warp_derives_channel_owner_after_solve(producer_init, consumer_init) -> None:
    stages, pipeline, canon = _warp_fixture()
    stages = _with_inits(stages, producer_init, consumer_init)
    profile = get_profile("sm_103a")
    expansion = expand("ring", stages[0].endpoints[0], stages[1].endpoints[0], stages, pipeline, canon, profile)
    solution = solve(pipeline, stages, canon, [expansion], profile)
    assert check_init_warps(stages, solution.warp_map) == {"ring": "load"}


def test_initializing_warp_rejects_missing_or_ambiguous_solved_role() -> None:
    stages, pipeline, canon = _warp_fixture()
    stages = _with_inits(stages, Init(None, 0, "stage", 8), Init(None, 0, "stage", 8))
    profile = get_profile("sm_103a")
    expansion = expand("ring", stages[0].endpoints[0], stages[1].endpoints[0], stages, pipeline, canon, profile)
    solution = solve(pipeline, stages, canon, [expansion], profile)
    with pytest.raises(SolveError) as caught:
        check_init_warps(stages, solution.warp_map)
    assert caught.value.failures[0].clause_id == "L-PROTO"
    assert caught.value.failures[0].detail == {"warp": 8}
    overlapping = {**solution.warp_map, "load": (4, 9), "mma": (0, 9)}
    with pytest.raises(SolveError) as caught:
        check_init_warps(stages, overlapping)
    assert caught.value.failures[0].clause_id == "L-PROTO"
    assert caught.value.failures[0].detail == {"warp": 8, "roles": ["load", "mma"]}


def test_same_initializing_warp_defers_declared_owner_mismatch_until_solve() -> None:
    stages, pipeline, canon = _warp_fixture()
    stages = _with_inits(stages, Init(None, 0, "stage", 7), Init("mma", 0, "stage", 7))
    profile = get_profile("sm_103a")
    expansion = expand("ring", stages[0].endpoints[0], stages[1].endpoints[0], stages, pipeline, canon, profile)
    solution = solve(pipeline, stages, canon, [expansion], profile)
    with pytest.raises(SolveError) as caught:
        check_init_warps(stages, solution.warp_map)
    failure = caught.value.failures[0]
    assert failure.clause_id == "L-PROTO"
    assert failure.detail == {"owner": "mma", "derived_owner": "load", "warp": 7}


def test_owner_only_and_warp_only_endpoints_must_resolve_to_same_role() -> None:
    stages, pipeline, canon = _warp_fixture()
    stages = _with_inits(stages, Init("mma", 0, "stage"), Init(None, 0, "stage", 7))
    profile = get_profile("sm_103a")
    expansion = expand("ring", stages[0].endpoints[0], stages[1].endpoints[0], stages, pipeline, canon, profile)
    solution = solve(pipeline, stages, canon, [expansion], profile)
    with pytest.raises(SolveError) as caught:
        check_init_warps(stages, solution.warp_map)
    failure = caught.value.failures[0]
    assert failure.clause_id == "L-PROTO"
    assert failure.detail == {"expected": "mma", "actual": "load"}


def test_ring_channels_must_agree_on_derived_init_owner() -> None:
    stages, pipeline, canon = _warp_fixture()
    stages = [
        replace(stage, endpoints=[
            replace(endpoint, protocol=replace(endpoint.protocol, init=Init(None, 0, "stage", 7))),
            replace(endpoint, name="other", protocol=replace(endpoint.protocol, init=Init("mma", 0, "stage"))),
        ])
        for stage in stages for endpoint in stage.endpoints
    ]
    profile = get_profile("sm_103a")
    expansion = expand("ring", stages[0].endpoints[0], stages[1].endpoints[0], stages, pipeline, canon, profile)
    solution = solve(pipeline, stages, canon, [expansion], profile)
    with pytest.raises(SolveError) as caught:
        check_init_warps(stages, solution.warp_map)
    failure = caught.value.failures[0]
    assert failure.clause_id == "L-PROTO"
    assert failure.detail == {"expected": "load", "actual": "mma"}


def test_mbarrier_rejects_non_byte_integral_tile_and_multiple_ring_init_owners() -> None:
    stages, pipeline, canon = _fixture()
    producer, consumer = stages[0].endpoints[0], stages[1].endpoints[0]
    bad_canon = dict(canon)
    bad_canon["alpha.stage:ring:produce"] = replace(
        bad_canon["alpha.stage:ring:produce"], footprint_bits=255
    )
    _assert_proto_failure(
        lambda: expand(
            "ring", producer, consumer, stages, pipeline, bad_canon, get_profile("sm_103a")
        )
    )

    other_p = replace(
        producer,
        name="other",
        protocol=replace(producer.protocol, init=Init("mma", 0, "dispatch")),
    )
    other_c = replace(
        consumer,
        name="other",
        protocol=replace(consumer.protocol, init=Init("mma", 0, "dispatch")),
    )
    duplicate_ring_stages = [
        replace(stages[0], endpoints=[producer, other_p]),
        replace(stages[1], endpoints=[consumer, other_c]),
    ]
    _assert_proto_failure(
        lambda: expand(
            "ring",
            producer,
            consumer,
            duplicate_ring_stages,
            pipeline,
            canon,
            get_profile("sm_103a"),
        )
    )


@pytest.mark.parametrize(
    "mismatch",
    ["tx", "empty_tx", "arrive", "slots", "init", "ring", "slot_extent", "slot_alias", "owner"],
)
def test_mbarrier_mismatch_classes_reject_under_l_proto(mismatch: str) -> None:
    stages, pipeline, canon = _fixture()
    producer, consumer = stages[0].endpoints[0], stages[1].endpoints[0]
    if mismatch == "tx":
        full = replace(consumer.protocol.full, tx_bytes=16)
        consumer = replace(consumer, protocol=replace(consumer.protocol, full=full))
        stages = _replace_channel_endpoint(stages, "beta.stage", consumer)
    elif mismatch == "empty_tx":
        empty = replace(consumer.protocol.empty, tx_bytes=8)
        consumer = replace(consumer, protocol=replace(consumer.protocol, empty=empty))
        stages = _replace_channel_endpoint(stages, "beta.stage", consumer)
    elif mismatch == "arrive":
        full = replace(consumer.protocol.full, arrive=Arrive(2, "load"))
        consumer = replace(consumer, protocol=replace(consumer.protocol, full=full))
        stages = _replace_channel_endpoint(stages, "beta.stage", consumer)
    elif mismatch == "slots":
        full = replace(consumer.protocol.full, slots="pipelink_empty")
        consumer = replace(consumer, protocol=replace(consumer.protocol, full=full))
        stages = _replace_channel_endpoint(stages, "beta.stage", consumer)
    elif mismatch == "init":
        consumer = replace(consumer, protocol=replace(consumer.protocol, init=Init("load", 0, "stage")))
        stages = _replace_channel_endpoint(stages, "beta.stage", consumer)
    elif mismatch == "ring":
        consumer = replace(consumer, ring=replace(consumer.ring, stage_stride_bytes=64))
        stages = _replace_channel_endpoint(stages, "beta.stage", consumer)
    elif mismatch == "slot_extent":
        bad_claim = replace(stages[0].claims[1], extent=replace(stages[0].claims[1].extent, const=1))
        stages = [replace(stages[0], claims=[stages[0].claims[0], bad_claim]), stages[1]]
    elif mismatch == "slot_alias":
        claims = [replace(claim, alias="same") for claim in stages[0].claims]
        stages = [replace(stages[0], claims=claims), stages[1]]
    elif mismatch == "owner":
        producer = replace(producer, protocol=replace(producer.protocol, init=Init("ghost", 0, "dispatch")))
        consumer = replace(consumer, protocol=replace(consumer.protocol, init=Init("ghost", 0, "dispatch")))
        stages = [replace(stages[0], endpoints=[producer]), replace(stages[1], endpoints=[consumer])]

    producer, consumer = stages[0].endpoints[0], stages[1].endpoints[0]
    _assert_proto_failure(
        lambda: expand("ring", producer, consumer, stages, pipeline, canon, get_profile("sm_103a"))
    )


def test_rooted_shared_completion_group_expands_contributions_and_rejects_bad_root_set() -> None:
    stages, pipeline, canon = _fixture()
    producer, consumer = stages[0].endpoints[0], stages[1].endpoints[0]
    root_full_p = replace(producer.protocol.full, tx_bytes=TxSum(["member", "root"]))
    root_full_c = replace(consumer.protocol.full, tx_bytes=TxSum(["root", "member"]))
    root_p = replace(producer, name="root", protocol=replace(producer.protocol, full=root_full_p))
    root_c = replace(consumer, name="root", protocol=replace(consumer.protocol, full=root_full_c))
    member_p = replace(
        producer,
        name="member",
        protocol=replace(producer.protocol, full=SharedMember("root", "tile")),
    )
    member_c = replace(
        consumer,
        name="member",
        protocol=replace(consumer.protocol, full=SharedMember("root", "tile")),
    )
    stages = [
        replace(stages[0], endpoints=[member_p, root_p]),
        replace(stages[1], endpoints=[root_c, member_c]),
    ]
    canon = {
        "alpha.stage:root:produce": _canon("root", "produce", 256),
        "beta.stage:root:consume": _canon("root", "consume", 256),
        "alpha.stage:member:produce": _canon("member", "produce", 64),
        "beta.stage:member:consume": _canon("member", "consume", 64),
    }

    root = expand("root", root_p, root_c, stages, pipeline, canon, get_profile("sm_103a"))
    member = expand("member", member_p, member_c, stages, pipeline, canon, get_profile("sm_103a"))
    assert root.tx_bytes_producer == root.tx_bytes_consumer == 40
    assert root.shared_group == ("root", ("member", "root"), {"member": 8, "root": 32})
    assert member.tx_bytes_producer == member.tx_bytes_consumer == 8
    assert member.init_slots is None
    assert member.init_materializer is None

    bad_root_p = replace(root_p, protocol=replace(root_p.protocol, full=replace(root_full_p, tx_bytes=TxSum(["root"]))))
    bad_stages = [replace(stages[0], endpoints=[member_p, bad_root_p]), stages[1]]
    _assert_proto_failure(
        lambda: expand("root", bad_root_p, root_c, bad_stages, pipeline, canon, get_profile("sm_103a"))
    )


def test_call_channel_pairs_caller_local_role_with_roleless_hash_bound_export() -> None:
    """The caller declares pipelink_call_<channel>; only the shape must match the export."""
    stages, pipeline, _ = _fixture()
    declaration = 'extern "C" int beta_stage_compute(int value);'
    caller_declaration = 'extern "C" int pipelink_call_invoke(int value);'
    digest = declaration_digest(declaration)
    callee = Protocol("call", None, None, None, None, CallDecl("compute", "int(int)", declaration, digest), None, None)
    caller = Protocol(
        "call", None, None, None, None,
        CallDecl("compute", "int(int)", caller_declaration, declaration_digest(caller_declaration)), None, None,
    )
    producer = replace(
        stages[0].endpoints[0], name="invoke", owner_role="load", space="reg", proxy="generic", ring=None, protocol=caller
    )
    consumer = replace(
        stages[1].endpoints[0], name="invoke", owner_role=None, space="reg", proxy="generic", ring=None, protocol=callee
    )
    stages = [
        replace(stages[0], endpoints=[producer]),
        replace(
            stages[1],
            endpoints=[consumer],
            exports={"compute": Export("beta_stage_compute", declaration, digest)},
        ),
    ]
    profile = get_profile("sm_103a")
    producer_canon = canonicalize(producer, None, profile)
    consumer_canon = canonicalize(consumer, None, profile)
    assert producer.owner_role not in stages[1].roles
    assert producer_canon.tv_role is None
    assert consumer_canon.tv_role is None
    assert unify(producer_canon, consumer_canon) == []
    canon = {"alpha.stage:invoke:produce": producer_canon, "beta.stage:invoke:consume": consumer_canon}
    result = expand("invoke", producer, consumer, stages, pipeline, canon, profile)
    assert result.callable == ("beta_stage_compute", declaration, digest)

    # The caller's manifest is independent of which stage exports the callee.
    other_declaration = 'extern "C" int gamma_stage_compute(int value);'
    other_consumer = replace(
        consumer, protocol=replace(callee, call=replace(callee.call, decl=other_declaration, decl_sha256=declaration_digest(other_declaration)))
    )
    other_stages = [
        stages[0],
        replace(stages[1], endpoints=[other_consumer], exports={"compute": Export("gamma_stage_compute", other_declaration, declaration_digest(other_declaration))}),
    ]
    assert expand("invoke", producer, other_consumer, other_stages, pipeline, {}, get_profile("sm_103a")).callable[0] == "gamma_stage_compute"

    # A caller compiled against a different parameter list rejects.
    wrong_caller_decl = 'extern "C" int pipelink_call_invoke(long value);'
    wrong_producer = replace(
        producer, protocol=replace(caller, call=replace(caller.call, decl=wrong_caller_decl, decl_sha256=declaration_digest(wrong_caller_decl)))
    )
    _assert_proto_failure(
        lambda: expand("invoke", wrong_producer, consumer, [replace(stages[0], endpoints=[wrong_producer]), stages[1]], pipeline, {}, get_profile("sm_103a"))
    )

    # A caller that names the callee's symbol is peer-coupled and rejects.
    coupled_producer = replace(producer, protocol=callee)
    _assert_proto_failure(
        lambda: expand("invoke", coupled_producer, consumer, [replace(stages[0], endpoints=[coupled_producer]), stages[1]], pipeline, {}, get_profile("sm_103a"))
    )

    # Formatting and comments do not alter declaration identity.
    spaced_decl = " /* declaration */ " + declaration + " \n"
    spaced = replace(consumer, protocol=replace(callee, call=replace(callee.call, decl=spaced_decl)))
    spaced_stages = [stages[0], replace(stages[1], endpoints=[spaced])]
    assert expand("invoke", producer, spaced, spaced_stages, pipeline, {}, get_profile("sm_103a")).callable == result.callable

    changed_decl = declaration.replace("int value", "long value")
    changed = replace(
        consumer,
        protocol=replace(callee, call=replace(callee.call, decl=changed_decl, decl_sha256=declaration_digest(changed_decl))),
    )
    changed_stages = [stages[0], replace(stages[1], endpoints=[changed])]
    _assert_proto_failure(
        lambda: expand("invoke", producer, changed, changed_stages, pipeline, {}, get_profile("sm_103a"))
    )

    missing_export_stages = [stages[0], replace(stages[1], exports={})]
    _assert_proto_failure(
        lambda: expand("invoke", producer, consumer, missing_export_stages, pipeline, {}, get_profile("sm_103a"))
    )

    # ABI-token preambles are identity-bearing: identical tokens accept,
    # divergent tokens reject even when the parameter list matches.
    token_callee = 'typedef int abi_token;\nextern "C" int beta_stage_compute(abi_token value);'
    token_caller = 'typedef int abi_token;\nextern "C" int pipelink_call_invoke(abi_token value);'
    drift_caller = 'typedef long abi_token;\nextern "C" int pipelink_call_invoke(abi_token value);'
    token_digest = declaration_digest(token_callee)
    token_export = Export("beta_stage_compute", token_callee, token_digest)
    token_consumer = replace(
        consumer,
        protocol=replace(callee, call=CallDecl("compute", "int(int)", token_callee, token_digest)),
    )
    def token_stages(caller_decl: str) -> tuple[Any, Any, list[Any]]:
        producer_side = replace(
            producer,
            protocol=replace(caller, call=CallDecl("compute", "int(int)", caller_decl, declaration_digest(caller_decl))),
        )
        return producer_side, token_consumer, [
            replace(stages[0], endpoints=[producer_side]),
            replace(stages[1], endpoints=[token_consumer], exports={"compute": token_export}),
        ]
    ok_producer, ok_consumer, ok_stages = token_stages(token_caller)
    assert expand("invoke", ok_producer, ok_consumer, ok_stages, pipeline, {}, get_profile("sm_103a")).callable[0] == "beta_stage_compute"
    bad_producer, bad_consumer, bad_token_stages = token_stages(drift_caller)
    _assert_proto_failure(
        lambda: expand("invoke", bad_producer, bad_consumer, bad_token_stages, pipeline, {}, get_profile("sm_103a"))
    )

    # A caller that smuggles a declaration of the callee symbol rejects: the
    # extra declaration is a token difference the export side does not carry.
    smuggled = (
        'extern "C" int beta_stage_compute(int value);\n'
        'extern "C" int pipelink_call_invoke(int value);'
    )
    smug_producer, smug_consumer, smug_stages = token_stages(smuggled)
    smug_consumer = consumer
    smug_stages = [smug_stages[0], stages[1]]
    _assert_proto_failure(
        lambda: expand("invoke", smug_producer, smug_consumer, smug_stages, pipeline, {}, get_profile("sm_103a"))
    )


def test_named_barrier_and_counter_agreement_and_counter_scope() -> None:
    stages, pipeline, _ = _fixture()
    producer, consumer = stages[0].endpoints[0], stages[1].endpoints[0]
    named_pipeline = replace(
        pipeline,
        handles={
            **pipeline.handles,
            "pipelink_full": replace(pipeline.handles["pipelink_full"], domain="named_barrier"),
        },
    )
    named = Protocol("named_barrier", None, None, None, None, None, NamedBarrierDecl("pipelink_full", 128), None)
    named_p = replace(producer, ring=None, protocol=named)
    named_c = replace(consumer, ring=None, protocol=named)
    named_stages = [replace(stages[0], endpoints=[named_p]), replace(stages[1], endpoints=[named_c])]
    result = expand("ring", named_p, named_c, named_stages, named_pipeline, {}, get_profile("sm_103a"))
    assert result.named_barrier == ("pipelink_full", 128)

    bad_named_c = replace(named_c, protocol=replace(named, named_barrier=NamedBarrierDecl("pipelink_full", 64)))
    _assert_proto_failure(
        lambda: expand("ring", named_p, bad_named_c, [named_stages[0], replace(stages[1], endpoints=[bad_named_c])], named_pipeline, {}, get_profile("sm_103a"))
    )

    counter = Protocol(
        "counter", None, None, None, None, None, None,
        CounterDecl("pipelink_full", "red.release.gpu", "ld.acquire.gpu"),
    )
    counter_pipeline = replace(
        pipeline,
        handles={
            **pipeline.handles,
            "pipelink_full": replace(pipeline.handles["pipelink_full"], domain="gmem_counter"),
        },
    )
    counter_p = replace(producer, ring=None, space="gmem", scope="grid", proxy="generic", protocol=counter)
    counter_c = replace(consumer, ring=None, space="gmem", scope="grid", proxy="generic", protocol=counter)
    counter_stages = [replace(stages[0], endpoints=[counter_p]), replace(stages[1], endpoints=[counter_c])]
    result = expand("ring", counter_p, counter_c, counter_stages, counter_pipeline, {}, get_profile("sm_103a"))
    assert result.counter == ("pipelink_full", "red.release.gpu", ("ld.acquire.gpu",))

    # The acquire pattern is consumer-local: a fence-acquire consumer pairs with any .gpu producer.
    fence = replace(counter, counter=CounterDecl("pipelink_full", "red.release.gpu", "ld.relaxed.gpu+fence.acquire.gpu"))
    fence_c = replace(counter_c, protocol=fence)
    fence_stages = [counter_stages[0], replace(stages[1], endpoints=[fence_c])]
    result = expand("ring", counter_p, fence_c, fence_stages, counter_pipeline, {}, get_profile("sm_103a"))
    assert result.counter == ("pipelink_full", "red.release.gpu", ("ld.relaxed.gpu+fence.acquire.gpu",))
    other_slot = replace(counter_c, protocol=replace(counter, counter=CounterDecl("pipelink_other", "red.release.gpu", "ld.acquire.gpu")))
    _assert_proto_failure(
        lambda: expand("ring", counter_p, other_slot, [counter_stages[0], replace(stages[1], endpoints=[other_slot])], counter_pipeline, {}, get_profile("sm_103a"))
    )

    bad_counter_c = replace(counter_c, scope="cta")
    _assert_proto_failure(
        lambda: expand("ring", counter_p, bad_counter_c, [counter_stages[0], replace(stages[1], endpoints=[bad_counter_c])], counter_pipeline, {}, get_profile("sm_103a"))
    )


def test_system_counter_expands_and_endpoints_must_agree() -> None:
    stages, pipeline, _ = _fixture()
    pipeline = replace(pipeline, handles={
        **pipeline.handles, "pipelink_full": replace(pipeline.handles["pipelink_full"], domain="gmem_counter"),
    })

    def endpoint(base, scope, release, acquire):
        protocol = Protocol("counter", None, None, None, None, None, None,
                            CounterDecl("pipelink_full", release, acquire))
        return replace(base, ring=None, space="gmem", scope=scope, proxy="generic", protocol=protocol)

    def run(producer, consumer):
        pair = [replace(stages[0], endpoints=[producer]), replace(stages[1], endpoints=[consumer])]
        return expand("ring", producer, consumer, pair, pipeline, {}, get_profile("sm_103a"))

    system = ("system", "red.release.sys", "ld.acquire.sys")
    grid = ("grid", "red.release.gpu", "ld.acquire.gpu")
    producer = endpoint(stages[0].endpoints[0], *system)
    consumer = endpoint(stages[1].endpoints[0], *system)
    assert run(producer, consumer).counter == ("pipelink_full", "red.release.sys", ("ld.acquire.sys",))
    # A .sys releaser cannot pair with a .gpu acquirer (or vice versa).
    _assert_proto_failure(lambda: run(producer, endpoint(stages[1].endpoints[0], *grid)))
    _assert_proto_failure(lambda: run(endpoint(stages[0].endpoints[0], *grid), consumer))


def test_arrive_sum_uses_every_launched_role_width() -> None:
    stages, pipeline, canon = _fixture()
    endpoints = []
    for stage in stages:
        endpoint = stage.endpoints[0]
        empty = replace(endpoint.protocol.empty, arrive=Arrive("threads(load)+2*threads(mma)+3*warps(load)", "mma"))
        endpoints.append(replace(endpoint, protocol=replace(endpoint.protocol, empty=empty)))
    stages = [replace(stage, endpoints=[endpoint]) for stage, endpoint in zip(stages, endpoints)]
    result = expand("ring", *endpoints, stages, pipeline, canon, get_profile("sm_103a"))
    assert result.arrive_empty == 128 + 256 + 12
    bad = replace(endpoints[0], protocol=replace(
        endpoints[0].protocol, empty=replace(endpoints[0].protocol.empty, arrive=Arrive("threads(missing)", "mma")),
    ))
    _assert_proto_failure(lambda: expand("ring", bad, endpoints[1], stages, pipeline, canon, get_profile("sm_103a")))


def test_one_shot_full_barrier_expands_without_a_release_or_dispatch_init() -> None:
    from pipec2.model import Depth
    stages, pipeline, canon = _fixture()
    changed = []
    for stage in stages:
        endpoint = stage.endpoints[0]
        protocol = replace(endpoint.protocol, empty=None, release="never", init=Init("load", 0, "stage"))
        endpoint = replace(endpoint, protocol=protocol, ring=replace(endpoint.ring, depth=Depth(1, None)))
        claims = [replace(claim, extent=Extent(1, {}, {})) for claim in stage.claims]
        changed.append(replace(stage, endpoints=[endpoint], claims=claims))
    result = expand("ring", changed[0].endpoints[0], changed[1].endpoints[0], changed, pipeline, canon, get_profile("sm_103a"))
    assert result.arrive_full == 1
    assert result.arrive_empty is None and result.init_slots is None
    assert result.init_materializer == "stage"


def _signal_fixture(release="external", materializer="dispatch", depth=1):
    from pipec2.layout import Layout
    from pipec2.model import Depth, Ring, Tile

    stages, pipeline, _ = _fixture()
    changed = []
    for stage in stages:
        endpoint = stage.endpoints[0]
        full = replace(endpoint.protocol.full, tx_bytes=0, arrive=Arrive(2, "load"))
        endpoint = replace(
            endpoint, space="none", scope="cluster", proxy="generic",
            tile=Tile("none", ()), layout=Layout(space="none"),
            ring=Ring(Depth(depth, None), 0), binds={"full": "pipelink_full"},
            protocol=replace(endpoint.protocol, full=full, empty=None, release=release,
                             init=Init("load", 0, materializer)),
        )
        claims = [replace(claim, extent=Extent(depth, {}, {}))
                  for claim in stage.claims if claim.handle == "pipelink_full"]
        changed.append(replace(stage, endpoints=[endpoint], claims=claims))
    profile = get_profile("sm_103a")
    canon = {f"{stage.id}:ring:{stage.endpoints[0].direction}":
             canonicalize(stage.endpoints[0], None, profile) for stage in changed}
    return changed, pipeline, canon


@pytest.mark.parametrize("release,materializer,depth", [
    ("external", "dispatch", 1), ("external", "stage", 3),
    ("never", "dispatch", 1), ("never", "stage", 1),
])
def test_completion_only_channel_expands_declared_arrivals_without_empty_slots(release, materializer, depth) -> None:
    stages, pipeline, canon = _signal_fixture(release, materializer, depth)
    result = expand("ring", *(stage.endpoints[0] for stage in stages),
                    stages, pipeline, canon, get_profile("sm_103a"))
    assert result.tx_bytes_producer == result.tx_bytes_consumer == 0
    assert result.arrive_full == 2 and result.arrive_empty is None
    assert result.init_slots == ("pipelink_full", None)
    assert result.init_materializer == materializer and result.init_owner == "load"
    assert result.shared_group is None
    assert all(endpoint.footprint_bits == 0 for endpoint in canon.values())
    assert unify(*canon.values()) == []


@pytest.mark.parametrize("mismatch", ["release", "scope", "depth", "space"])
def test_completion_only_channel_pair_mismatches_are_l_pair(mismatch) -> None:
    from pipec2.model import Depth

    stages, pipeline, canon = _signal_fixture()
    consumer = stages[1].endpoints[0]
    if mismatch == "release":
        consumer = replace(consumer, protocol=replace(consumer.protocol, release="never"))
    elif mismatch == "scope":
        consumer = replace(consumer, scope="cta")
    elif mismatch == "depth":
        consumer = replace(consumer, ring=replace(consumer.ring, depth=Depth(2, None)))
    else:
        consumer = replace(consumer, space="smem")
    stages[1] = replace(stages[1], endpoints=[consumer])
    with pytest.raises(SolveError) as caught:
        expand("ring", stages[0].endpoints[0], consumer, stages, pipeline, canon, get_profile("sm_103a"))
    assert caught.value.failures[0].clause_id == "L-PAIR"


@pytest.mark.parametrize("domain,unit_bytes", [("smem", 1), ("mbarrier", 16)])
def test_native_barrier_resolution_checks_solved_depth_and_extent(domain, unit_bytes) -> None:
    from pipec2.protocol import resolve_native
    stages, pipeline, canon = _warp_fixture()
    endpoint = stages[0].endpoints[0]
    native = BarrierStorage("pipelink_native", 8, 8)
    endpoint = replace(endpoint, protocol=replace(
        endpoint.protocol, full=replace(endpoint.protocol.full, native=native), init=Init("load", 0, "stage"),
    ))
    consumer = stages[1].endpoints[0]
    stages[1] = replace(stages[1], endpoints=[replace(
        consumer, protocol=replace(consumer.protocol, init=Init("load", 0, "stage")),
    )])
    claim = replace(stages[0].claims[0], handle=native.base, domain=domain,
                    extent=Extent(0, {"pipelink_depth": 16 if domain == "smem" else 1}, {}))
    stages[0] = replace(stages[0], endpoints=[endpoint], claims=[*stages[0].claims, claim])
    pipeline = replace(pipeline, handles={**pipeline.handles, native.base: Handle(
        native.base, "base", "bind", domain, None, "claim:native",
    )})
    profile = get_profile("sm_103a")
    expansion = expand("ring", endpoint, stages[1].endpoints[0], stages, pipeline, canon, profile)
    solution = solve(pipeline, stages, canon, [expansion], profile)
    result = resolve_native([expansion], stages, solution, profile)[0]
    assert result.to_json_obj()["native"]["full"] == {
        "base_solved_offset_units": solution.native_bases[native.base],
        "unit_bytes": unit_bytes, "offset_bytes": 8, "stride_bytes": 8,
    }
    overflowing = replace(endpoint, protocol=replace(
        endpoint.protocol, full=replace(endpoint.protocol.full, native=replace(
            native, stride_bytes=solution.launch.dyn_smem_bytes,
        )),
    ))
    invalid_stages = [replace(stages[0], endpoints=[overflowing]), stages[1]]
    _assert_proto_failure(lambda: resolve_native([expansion], invalid_stages, solution, profile))
