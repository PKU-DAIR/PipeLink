"""CPU-only checks at the pre-codegen TIR -> device CUDA seam."""
import hashlib
import importlib.util
from pathlib import Path

import pytest
import tilelang
from tilelang import language as T, tvm

from pipelink_tilelang.annotate import AnnotationError, symbol_prefix
from pipelink_tilelang.capture import _PassCapture, capture
from pipelink_tilelang.external import ExternalBarrier, ExternalStorage, ExternalThreadBase, NativeStorage
from pipelink_tilelang.externalize import externalize_role, split_roles


PROFILES = ("sm_103a", "sm_90a", "sm_120a")
SM103 = tvm.target.Target({"kind": "cuda", "arch": "sm_103a"})


@pytest.fixture(scope="module", params=PROFILES)
def target(request):
    return tvm.target.Target({"kind": "cuda", "arch": request.param})


def _lower(kernel, target, *, scheduled=False):
    trace = _PassCapture({})
    with target, tvm.transform.PassContext(opt_level=3, config={"tl.disable_warp_specialized": not scheduled}, instruments=[trace]):
        tilelang.lower(kernel, target=target, enable_host_codegen=False, enable_device_compile=False)
    assert trace.passes[-3:] == ["tl.LowerIntrin", "tirx.Simplify", "pass_fn"]
    return trace.last_mod


def _emit(mod, records, target, *, warp_range=(0, 4), params=("A", "B")):
    return externalize_role(mod, role="worker", warp_range=warp_range, branch=None,
                            external=records, prefix="test_", declared_params=params, target=target)


@pytest.fixture(scope="module")
def barriers_module(target):
    @T.prim_func
    def kernel(A: T.Tensor((256,), T.float32), B: T.Tensor((256,), T.float32)):
        with T.Kernel(1, threads=256):
            tx = T.get_thread_binding()
            full = T.alloc_barrier([1, 1])
            empty = T.alloc_barrier([128, 128])
            side = T.alloc_barrier([1, 1])
            plane = T.alloc_shared((256,), T.float32)
            scratch = T.alloc_shared((256,), T.float32)
            if tx < 128:
                T.mbarrier_wait_parity(full[tx % 2], 0)
                T.mbarrier_arrive(empty[tx % 2])
                T.mbarrier_arrive(side[0])
                T.sync_threads(6, 256)
                plane[tx] = A[tx]
                scratch[tx] = A[tx] + 1.0
                T.sync_threads()
                B[tx] = plane[tx + 1] + scratch[tx + 1]
            else:
                B[tx] = 42.0
    return _lower(kernel, target)


def _barriers(init):
    return [
        ExternalBarrier("full", "full", "full_slots", NativeStorage("native_slots", 16, 8, 8), 2, 7, init, 0),
        ExternalBarrier("empty", "empty", "empty_slots", NativeStorage("native_bytes", 1, 24, 8), 2, 9, "drop", 0),
        ExternalBarrier("side", "full", "side_slots", None, 2, 11, init, 0),
        ExternalStorage("plane", "plane_offset", "smem", None),
        ExternalThreadBase("warp_base", 4),
    ]


def test_warp_selection_native_and_sidecar_addresses(barriers_module, target):
    result = _emit(barriers_module, _barriers("drop"), target)
    source = result.body
    assert "test_native(test_smem, native_slots(), 16, 8, 8," in source or "test_native(buf_dyn_shmem, native_slots(), 16, 8, 8," in source
    assert "native_bytes(), 1, 24, 8," in source
    assert "side_slots(), 0).arrive()" in source
    assert "plane_offset()" in source and "plane =" in source
    assert "warp_base() * 32" in source
    assert "test_named_barrier_id(), 128" in source
    assert "tl::__sync_thread_partial(6, 256);" in source
    assert "< 128)" not in source and "4.200000e+01" not in source
    assert ".init(" not in source and "fence_barrier_init" not in source
    assert "__syncthreads" not in source
    for handle in ("full_slots", "empty_slots", "side_slots", "native_slots", "native_bytes", "plane_offset", "warp_base"):
        assert f'extern "C" __device__ int {handle}();' in source


def test_owned_initialization_keeps_contract_count_and_one_fence(barriers_module, target):
    source = _emit(barriers_module, _barriers("keep"), target).body
    assert source.count(".init(7)") == 2
    assert source.count(".init(11)") == 2
    assert ".init(9)" not in source and ".init(128)" not in source
    assert source.count("tl::fence_barrier_init();") == 1
    assert source.index(".init(11)") < source.index("tl::fence_barrier_init();") < source.index("test_named_barrier_id(), 128")
    assert "__syncthreads" not in source
    assert "tl_shuffle_elect<0>" not in source


def test_else_selection_drops_other_arm_and_unused_params(barriers_module, target):
    records = [record for record in _barriers("drop") if not isinstance(record, ExternalStorage)]
    source = _emit(barriers_module, records, target, warp_range=(4, 8), params=("B",)).body
    assert "4.200000e+01" in source
    assert ".wait(" not in source and ".arrive(" not in source
    assert "< 128)" not in source
    assert "const float* __restrict__ A" not in source


def test_thread_interval_chain_selects_middle_branch(target):
    @T.prim_func
    def kernel(B: T.Tensor((256,), T.int32)):
        with T.Kernel(1, threads=256):
            tx = T.get_thread_binding()
            if tx < 32:
                B[tx] = 11
            elif tx >= 32 and tx < 128:
                B[tx] = 22
            else:
                B[tx] = 33
    mod = _lower(kernel, target)
    source = _emit(mod, [ExternalThreadBase("warp_base", 3)], target, warp_range=(1, 4), params=("B",)).body
    assert "= 22;" in source and "= 11;" not in source and "= 33;" not in source
    assert "threadIdx.x) + 32" in source and "warp_base() * 32" in source
    assert "if (" not in source
    with pytest.raises(AnnotationError, match="warp_range"):
        _emit(mod, [], target, warp_range=(2, 4), params=("B",))


def test_else_role_keeps_its_internal_guard(target):
    @T.prim_func
    def kernel(B: T.Tensor((256,), T.int32)):
        with T.Kernel(1, threads=256):
            tx = T.get_thread_binding()
            if tx < 128:
                B[tx] = 11
            else:
                if tx % 32 == 0:
                    B[tx] = 22
    source = _emit(_lower(kernel, target), [], target, warp_range=(4, 8), params=("B",)).body
    assert "= 22;" in source and "= 11;" not in source
    assert "if (" in source and "% 32) == 0" in source
    assert "< 128)" not in source


def test_tmem_rebinds_published_word_and_column_offset():
    @T.prim_func
    def kernel(A: T.Tensor((128, 32), T.float32), B: T.Tensor((128, 32), T.float32)):
        with T.Kernel(1, threads=256):
            tx = T.get_thread_binding()
            acc = T.alloc_tmem((128, 32), T.float32)
            plane = T.alloc_shared((128, 32), T.float32)
            frag = T.alloc_fragment((128, 32), T.float32)
            T.annotate_layout({acc: T.Layout((128, 32), lambda i, j: (i, j))})
            if tx < 128:
                T.sync_threads(6, 256)
                T.copy(A, frag)
                T.copy(frag, acc)
                T.copy(acc, frag)
                T.copy(frag, plane)
                T.copy(plane, B)
            else:
                B[0, 0] = 42.0
    mod = _lower(kernel, SM103)
    records = [ExternalStorage("acc", "published_tmem_word", "tmem", "tmem_columns"),
               ExternalStorage("plane", "plane_offset", "smem", None), ExternalThreadBase("warp_base", 4)]
    source = _emit(mod, records, SM103).body
    assert "tmem_allocate" not in source and "tmem_deallocate" not in source
    assert "test_smem + published_tmem_word()" in source
    assert "((uint*)acc)[0]" in source and "tmem_columns()" in source
    assert "tcgen05_st_32dp32bNx" in source and "tcgen05_ld_32dp32bNx" in source
    assert "test_named_barrier_id(), 128" in source and "__syncthreads" not in source
    assert "test_smem + plane_offset()" in source
    assert source.index("tl::__sync_thread_partial(6, 256)") < source.index("((uint*)acc)[0]")


def test_tmem_mma_pointer_operand_relocates_column_not_pointer():
    @T.prim_func
    def kernel(C: T.Tensor((128, 32), T.float32)):
        with T.Kernel(1, threads=256):
            tx = T.get_thread_binding()
            lhs = T.alloc_shared((128, 64), T.float16)
            rhs = T.alloc_shared((64, 32), T.float16)
            acc = T.alloc_tmem((128, 32), T.float32)
            done = T.alloc_barrier([1])
            frag = T.alloc_fragment((128, 32), T.float32)
            if tx < 128:
                T.clear(lhs)
                T.clear(rhs)
                T.tcgen05_gemm(lhs, rhs, acc, mbar=done[0], clear_accum=True)
                T.mbarrier_wait_parity(done[0], 0)
                T.copy(acc, frag)
                T.copy(frag, C)
            else:
                C[0, 0] = 42.0
    records = [ExternalStorage("acc", "published_tmem_word", "tmem", "tmem_columns"),
               ExternalStorage("lhs", "lhs_offset", "smem", None),
               ExternalStorage("rhs", "rhs_offset", "smem", None),
               ExternalBarrier("done", "full", "done_slots", None, 1, 1, "drop", 0)]
    source = _emit(_lower(kernel, SM103), records, SM103, params=("C",)).body
    assert "tmem_allocate" not in source and "tmem_deallocate" not in source
    assert "published_tmem_word()" in source
    mma_line = next(line for line in source.splitlines() if "tcgen05mma_ss" in line)
    assert "tmem_columns()" in mma_line
    assert "test_address(test_slot(" in source


def test_scheduled_role_materialization(target):
    @T.prim_func
    def kernel(A: T.Tensor((256, 64), T.float16), B: T.Tensor((256, 64), T.float16)):
        with T.Kernel(1, threads=256):
            plane = T.alloc_shared((64, 64), T.float16)
            frag = T.alloc_fragment((64, 64), T.float16)
            T.annotate_ws_schedule(T.WSSchedule(
                num_warps=8,
                roles=[T.WSRole("load", warps_lo=0, warps_hi=1), T.WSRole("compute", warps_lo=4, warps_hi=8)],
                pipelines=[T.WSPipeline("ring", [plane], depth=2)],
                scopes=[T.WSScope("loop", {
                    "load": [T.WSSync.producer_acquire("ring"), "in", T.WSSync.producer_commit("ring")],
                    "compute": [T.WSSync.consumer_wait("ring"), "read", T.WSSync.consumer_release("ring"), "out"],
                }), T.WSScope(T.WSScope.ROOT, {"load": ["loop"], "compute": ["loop"]})],
            ))
            for i in T.Pipelined(4, num_stages=2, annotations={T.WSID: "loop"}):
                T.copy(A[i * 64, 0], plane, annotations={T.WSID: "in"})
                T.copy(plane, frag, annotations={T.WSID: "read"})
                T.copy(frag, B[i * 64, 0], annotations={T.WSID: "out"})
    records = [ExternalBarrier("ring_full", "full", "full_slots", None, 2, 1, "drop", 0),
               ExternalBarrier("ring_empty", "empty", "empty_slots", None, 2, 128, "drop", 0),
               ExternalStorage("plane", "plane_offset", "smem", None), ExternalThreadBase("warp_base", 4)]
    source = _emit(_lower(kernel, target, scheduled=True), records, target, warp_range=(4, 8), params=("B",)).body
    assert ".wait(" in source and ".arrive(" in source
    assert "tma_load" not in source and "< 128)" not in source and "< 32)" not in source
    assert "set_max_nreg" not in source and ".init(" not in source
    assert "warp_base() * 256" in source and "- 1024" not in source


def test_original_ws_arm_bodies_are_byte_identical(target):
    path = Path(__file__).resolve().parent / "fixtures/ws_gemm.py"
    spec = importlib.util.spec_from_file_location("ws_gemm_externalize_test", path)
    ws = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ws)
    captured = capture(ws.program(3), target=target, pass_configs=ws.PASS_CONFIGS)
    arch = str(target.attrs["arch"])
    # sm103: captured before the shared rewrite helpers / new entry point were added
    # (byte-identical to the sm_100a capture it was first recorded from). sm90: the
    # producer arm is byte-identical to sm103; the consumer arm is the wgmma body.
    # sm120: both arms are byte-identical to sm103 (mma_sync + ldmatrix consumer).
    baseline = {
        "sm_103a": {"producer": "e5bc7c31ecc71a6e1782ba8a6c775f92be89e122826fb4f08bd0adaee77ec55b",
                    "consumer": "50c8f8409209645b40cefa60494f459f6aae5931737f445dffd5b62e621f31a3"},
        "sm_90a": {"producer": "e5bc7c31ecc71a6e1782ba8a6c775f92be89e122826fb4f08bd0adaee77ec55b",
                   "consumer": "a632428c3422ba2ca8010b4af6ebbc11a0411f5c4cb4432648de69ab3eab971f"},
        "sm_120a": {"producer": "e5bc7c31ecc71a6e1782ba8a6c775f92be89e122826fb4f08bd0adaee77ec55b",
                    "consumer": "50c8f8409209645b40cefa60494f459f6aae5931737f445dffd5b62e621f31a3"},
    }[arch]
    for stage in ws.PIPELINK.stages:
        role, declaration = next(iter(stage.roles.items()))
        result = split_roles({3: captured.modules["pre_codegen"]}, branch=declaration.branch,
                             prefix=symbol_prefix(stage),
                             smem_handles={"A_shared": "pipelink_tilelang_ws_smem_a_ring", "B_shared": "pipelink_tilelang_ws_smem_b_ring"},
                             full="pipelink_tilelang_ws_full", empty="pipelink_tilelang_ws_empty",
                             declared_params=tuple(field.name for field in stage.params.fields), target=target)
        assert hashlib.sha256(result.body.encode()).hexdigest() == baseline[role]
        if arch == "sm_90a" and role == "consumer":
            assert "wgmma" in result.body and "tcgen05" not in result.body and "copy_sm100" not in result.body
        if arch == "sm_120a" and role == "consumer":
            assert "mma_sync" in result.body and "wgmma" not in result.body and "tcgen05" not in result.body
        # The generalized entry point must also accept the legacy WS selector.
        arm = externalize_role(captured.modules["pre_codegen"], role=role, branch=declaration.branch,
                               warp_range=None, external=[], prefix="arm_", declared_params=result.functions[0].params,
                               target=target).body
        assert "set_max_nreg" not in arm
        assert ("tma_load" in arm) == (role == "producer")
