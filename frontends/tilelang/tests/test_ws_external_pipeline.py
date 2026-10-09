"""CPU-only lowering checks for the patched TileLang scheduled pipeline ABI."""

import pytest
import tilelang
from tilelang import language as T
from tilelang import tvm


@pytest.fixture(params=["sm_103a", "sm_90a", "sm_120a"])
def target(request):
    return tvm.target.Target({"kind": "cuda", "arch": request.param})


def _kernel(*, external_producer=False):
    @T.prim_func
    def kernel(A: T.Tensor((256, 64), T.float16), B: T.Tensor((256, 64), T.float16)):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((64, 64), T.float16)
            A_frag = T.alloc_fragment((64, 64), T.float16)
            roles = [T.WSRole("Consumer", warps_lo=4, warps_hi=8)]
            bodies = {
                "Consumer": [
                    T.WSSync.consumer_wait("smem"),
                    "copy_frag",
                    T.WSSync.consumer_release("smem"),
                    "copy_out",
                ],
            }
            roots = {"Consumer": ["loop_k"]}
            if external_producer:
                pipeline = T.WSPipeline("smem", [A_shared], depth=2, full_count=1)
            else:
                roles.insert(0, T.WSRole("TMA", warps_lo=0, warps_hi=1))
                bodies["TMA"] = [
                    T.WSSync.producer_acquire("smem"),
                    "copy_in",
                    T.WSSync.producer_commit("smem"),
                ]
                roots["TMA"] = ["loop_k"]
                pipeline = T.WSPipeline(
                    "smem", [A_shared], depth=2,
                    empty_count=256, arrive_cta=0, init_phase=1,
                )
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=roles,
                    pipelines=[pipeline],
                    scopes=[T.WSScope("loop_k", bodies), T.WSScope(T.WSScope.ROOT, roots)],
                )
            )
            for i in T.Pipelined(4, num_stages=2, annotations={T.WSID: "loop_k"}):
                if not external_producer:
                    T.copy(A[i * 64, 0], A_shared, annotations={T.WSID: "copy_in"})
                T.copy(A_shared, A_frag, annotations={T.WSID: "copy_frag"})
                T.copy(A_frag, B[i * 64, 0], annotations={T.WSID: "copy_out"})

    return kernel


def _lower(kernel, target):
    with target, tvm.transform.PassContext(opt_level=3):
        return tilelang.lower(
            kernel, target=target,
            enable_host_codegen=False, enable_device_compile=False,
        )


def _calls(mod, name):
    calls = []

    def visit(node):
        if isinstance(node, tvm.tirx.Call) and getattr(node.op, "name", "").endswith(name):
            calls.append(node)

    for func in mod.functions.values():
        tvm.tirx.stmt_functor.post_order_visit(func.body, visit)
    return calls


def test_external_arrival_count_cluster_target_and_phase(target):
    artifact = _lower(_kernel(), target)
    arrivals = _calls(artifact.device_mod, "ptx_arrive_cluster_barrier")
    assert arrivals and all(int(call.args[1]) == 0 for call in arrivals)
    assert ".arrive(0u)" in artifact.kernel_source
    inits = _calls(artifact.device_mod, "ptx_init_barrier_thread_count")
    assert sorted(int(call.args[1]) for call in inits) == [32, 32, 256, 256]
    waits = _calls(artifact.device_mod, "mbarrier_wait_parity")
    consumer_wait = next(call for call in waits if "smem_full" in str(call.args[0]))
    parity = consumer_wait.args[1]
    assert parity.op.name.endswith(".bitwise_xor") and int(parity.args[1]) == 1
    producer_wait = next(call for call in waits if "smem_empty" in str(call.args[0]))
    # This pin retains both xor nodes instead of folding the empty-side
    # inversion with the phase-origin inversion.
    parity = producer_wait.args[1]
    assert parity.op.name.endswith(".bitwise_xor") and int(parity.args[1]) == 1
    assert parity.args[0].op.name.endswith(".bitwise_xor")
    assert int(parity.args[0].args[1]) == 1


def test_external_full_side_lowers_without_a_producer_role(target):
    artifact = _lower(_kernel(external_producer=True), target)
    inits = _calls(artifact.device_mod, "ptx_init_barrier_thread_count")
    assert sorted(int(call.args[1]) for call in inits) == [1, 1, 128, 128]
    waits = _calls(artifact.device_mod, "mbarrier_wait_parity")
    assert len(waits) == 1 and "smem_full" in str(waits[0].args[0])
    releases = _calls(artifact.device_mod, "ptx_arrive_barrier")
    assert len(releases) == 1 and "smem_empty" in str(releases[0].args[0])
