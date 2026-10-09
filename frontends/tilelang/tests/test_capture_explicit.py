"""CPU-only capture of authored warp roles and their own storage/protocol facts."""
import importlib.util
import json
from pathlib import Path

import pytest
import tilelang.language as T
from tilelang import tvm

from pipelink_tilelang.capture import CaptureError, _validate, capture


PROFILES = ("sm_103a", "sm_90a", "sm_120a")
_CONFIG = {"tl.disable_warp_specialized": True}


@pytest.fixture(scope="module", params=PROFILES)
def target(request):
    return tvm.target.Target({"kind": "cuda", "arch": request.param})


def _capture(program, target):
    return capture(program, target=target, pass_configs=_CONFIG)


@pytest.fixture(scope="module")
def explicit_ring(target):
    @T.prim_func
    def kernel(A: T.Tensor((128, 64), T.float16), C: T.Tensor((128, 64), T.float16)):
        with T.Kernel(1, threads=256):
            tx = T.get_thread_binding()
            plane = T.alloc_shared((2, 128, 64), T.float16)
            full = T.alloc_barrier([1] * 2)
            empty = T.alloc_barrier([128] * 2)
            if tx < 32:
                for k in range(4):
                    T.mbarrier_wait_parity(empty[k % 2], ((k // 2) % 2) ^ 1)
                    T.tma_copy(A, plane[k % 2, :, :], barrier=full[k % 2])
                    if T.shuffle_elect(32):
                        T.mbarrier_arrive(full[k % 2])
            elif tx >= 128:
                for k in range(4):
                    T.mbarrier_wait_parity(full[k % 2], (k // 2) % 2)
                    for i, j in T.Parallel(128, 64):
                        C[i, j] = plane[k % 2, i, j]
                    T.mbarrier_arrive(empty[k % 2])
    return _capture(kernel, target)


def test_explicit_roles_and_distinct_barrier_rings(explicit_ring):
    facts = explicit_ring.facts
    assert [(b["thread_min"], b["thread_count"], b["warp_base"], b["warps"])
            for b in facts["branches"]] == [(0, 32, 0, 1), (128, 128, 4, 4)]
    expected = [[0, 1], [1, 1], [2, 128], [3, 128]]
    for window in facts["captures"].values():
        assert window["barrier_init"] == expected
    schedule = facts["barrier_schedule"]
    waits = [s for s in schedule if s["op"].endswith("mbarrier_wait_parity")]
    arrivals = [s for s in schedule if s["op"].endswith(("ptx_arrive_barrier", "ptx_arrive_barrier_expect_tx"))]
    assert {s["barrier"] for s in waits} == {0, 1, 2, 3}
    assert {s["barrier"] for s in arrivals} == {0, 1, 2, 3}
    assert {s["transaction_bytes"] for s in schedule if s["op"].endswith("expect_tx")} == {16384}
    assert any("tma_load" in call["op"] for call in facts["branches"][0]["calls"])


def test_explicit_validation_requires_used_slot_initialization(explicit_ring):
    facts = json.loads(json.dumps(explicit_ring.facts))
    facts["captures"]["pre_codegen"]["barrier_init"] = [[0, 1], [1, 1], [2, 128]]
    with pytest.raises(CaptureError, match="barrier_schedule.barrier"):
        _validate(facts)


def test_cluster_barrier_and_unprovable_slot(target):
    @T.prim_func
    def kernel(slot: T.int32):
        with T.Kernel(1, threads=256):
            tx = T.get_thread_binding()
            done = T.alloc_cluster_barrier([128, 128])
            if tx >= 128:
                T.mbarrier_wait_parity(done[0], 0)
                T.mbarrier_arrive(done[0], cta_id=0)
                T.mbarrier_expect_tx(done[1], 64)
                T.mbarrier_wait_parity(done[slot], 0)
    facts = _capture(kernel, target).facts
    assert facts["captures"]["post_layout"]["barrier_init"] == [[0, 128], [1, 128]]
    assert [(b["warp_base"], b["warps"]) for b in facts["branches"]] == [(4, 4)]
    assert any(s["op"].endswith("ptx_arrive_cluster_barrier") and s["barrier"] == 0
               for s in facts["barrier_schedule"])
    assert any(s["op"].endswith("mbarrier_wait_parity") and s["barrier"] is None
               for s in facts["barrier_schedule"])
    assert any(s["op"].endswith("mbarrier_expect_tx") and s["barrier"] == 1
               and s["transaction_bytes"] == 64 for s in facts["barrier_schedule"])


def test_bounded_role_interval_keeps_thread_election_internal(target):
    @T.prim_func
    def kernel(C: T.Tensor((256,), T.int32)):
        with T.Kernel(1, threads=256):
            tx = T.get_thread_binding()
            if tx >= 32 and tx < 128:
                if tx % 32 == 0:
                    C[tx] = 1
    facts = _capture(kernel, target).facts
    assert [(b["thread_min"], b["thread_count"], b["warp_base"], b["warps"])
            for b in facts["branches"]] == [(32, 96, 1, 3)]


def test_tmem_allocation_is_captured():
    @T.prim_func
    def kernel(A: T.Tensor((128, 32), T.float32), C: T.Tensor((128, 32), T.float32)):
        with T.Kernel(1, threads=128):
            acc = T.alloc_tmem((128, 32), T.float32)
            frag = T.alloc_fragment((128, 32), T.float32)
            T.annotate_layout({acc: T.Layout((128, 32), lambda i, j: (i, j))})
            T.copy(A, frag)
            T.copy(frag, acc)
            T.copy(acc, frag)
            T.copy(frag, C)
    facts = _capture(kernel, tvm.target.Target({"kind": "cuda", "arch": "sm_103a"})).facts
    allocations = facts["captures"]["post_layout"]["allocations"]
    assert {"name": "acc", "scope": "shared.tmem", "shape": [128, 32],
            "dtype": "float32", "bytes": 16384} in allocations


def test_arithmetic_f1_still_captures(target):
    path = Path(__file__).resolve().parent / "fixtures/f1_epilogue.py"
    spec = importlib.util.spec_from_file_location("capture_test_f1", path)
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    facts = _capture(recipe.program(), target).facts
    assert facts["branches"] == []
    assert facts["barrier_schedule"] == []
    assert any(layout["source"] == "For.annotations.parallel_loop_layout"
               for layout in facts["layouts"].values())


def test_symbolic_grid_preserves_extent_expression(target):
    @T.prim_func
    def kernel(C: T.Tensor((128,), T.int32), blocks: T.int32):
        with T.Kernel(blocks, threads=128):
            for i in T.Parallel(128):
                C[i] = 1
    facts = _capture(kernel, target).facts
    for window in facts["captures"].values():
        assert window["attrs"]["thread_extent"]["blockIdx.x"] == "blocks"
        assert window["attrs"]["thread_extent"]["threadIdx.x"] == 128
