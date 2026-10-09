from dataclasses import replace
from pathlib import Path
import ctypes
import json
import struct

import numpy as np
import pytest

from pipec2 import container
from pipec2.canonjson import canonical_bytes
from pipec2.dispatchgen import params_layout
from pipec2.launcher import open_hostless
from pipec2.link import link, _params_composition, _check_symbols, _Artifact, _solve_pipeline
from pipec2.model import Claim, Extent, Handle, LinkError, ParamsDecl, Role, SolveError, Variant
from pipec2.pack import pack_cu
from pipec2.profile import get_profile
from pipec2.symbols import PARAMS_BASE_HANDLE, params_accessor_symbol
from test_link import _stage, _pipeline, _symbol_table, cuda_toolchain


def _provider(stage_id, size=64, align=64, symbol=None):
    return replace(_stage(stage_id), params=ParamsDecl(size, align, "Opaque", "0" * 64),
                   variants=[Variant({}, 0, None)],
                   symbols=_symbol_table(undefined=(symbol or params_accessor_symbol(stage_id),)))


def test_accessor_ownership_and_canonical_collisions():
    cases = [
        [_provider("org.provider", symbol="pipelink_params_foreign")],
        [_provider("org.provider", size=0)],
        [_provider("a.b"), _provider("a-b")],
    ]
    for stages in cases:
        with pytest.raises(LinkError, match="params accessor"):
            _check_symbols([_Artifact(Path(s.id), s, b"", "") for s in stages])


def test_accessor_cell_is_single_kernel_lifetime_claim_and_opt_in():
    pipeline = _pipeline()
    unused = replace(_stage(), symbols=_symbol_table())
    plain, stages = _params_composition(pipeline, [unused])
    assert plain == pipeline and stages == [unused]
    providers = [_provider("provider.a", 256, 128), _provider("provider.b")]
    composed, stages = _params_composition(pipeline, providers)
    claims = [c for stage in stages for c in stage.claims]
    assert len(claims) == 1
    assert (claims[0].domain, claims[0].lifetime, claims[0].extent.const, claims[0].align) == ("smem", "kernel", 8, 8)
    assert PARAMS_BASE_HANDLE in composed.handles
    layout = params_layout(stages, get_profile("sm_103a"))
    assert layout.entries == [("provider.a", 0, 256, 128), ("provider.b", 256, 64, 64)]


def test_accessor_cell_cannot_alias_scratch_and_consumes_capacity():
    profile = get_profile("sm_103a")
    handles = {
        name: Handle(name, "base", "bind", domain, None, f"claim:{name}")
        for name, domain in (("pipelink_warps", "warp"), ("pipelink_scratch", "smem"))
    }
    caller = replace(_stage("org.caller"), roles={"call": Role("caller_work", 4)},
                     variants=[Variant({}, 0, None)],
                     claims=[
                         Claim("warp", "pipelink_warps", "kernel", Extent(4, {}, {}), 1,
                               0, None, None, None, None, None),
                         Claim("smem", "pipelink_scratch", "kernel", Extent(64, {}, {}), 16,
                               0, None, None, None, 0, None),
                     ])
    pipeline, stages = _params_composition(_pipeline(handles), [caller, _provider("provider.a")])
    solution = _solve_pipeline(pipeline, stages, {}, [], profile)
    assert solution.native_bases[PARAMS_BASE_HANDLE] == 64
    assert solution.launch.dyn_smem_bytes == 72
    constrained = replace(profile, domains={
        **profile.domains, "smem": replace(profile.domains["smem"], capacity=64),
    })
    with pytest.raises(SolveError):
        _solve_pipeline(pipeline, stages, {}, [], constrained)


@pytest.fixture
def accessor_link(tmp_path, cuda_toolchain):
    for name, size, alignment in (("a", 256, 128), ("b", 64, 64)):
        source = tmp_path / f"{name}.cu"
        source.write_text(f'''#include <pipelink/stage.hpp>
struct alignas({alignment}) Params_{name} {{ int value; char padding[{size - 4}]; }};
extern "C" __device__ const void* pipelink_params_provider_{name}();
extern "C" __device__ int provider_{name}_get(unsigned index) {{
  return static_cast<const Params_{name}*>(pipelink_params_provider_{name}())->value + int(index);
}}
PL_STAGE("provider.{name}", "1.0.0");
PL_PARAMS(Params_{name}, value, padding);
PL_EXPORT(get_{name}, provider_{name}_get);
PL_CALL(get_{name}, provider_{name}_get, pl::consume, "get_{name}");
''')
        pack_cu(source, tmp_path / f"{name}.plink", profile="sm_103a")
    source = tmp_path / "caller.cu"
    source.write_text('''#include <pipelink/stage.hpp>
struct CallerParams { int* output; };
extern "C" __device__ int pipelink_call_get_a(unsigned index);
extern "C" __device__ int pipelink_call_get_b(unsigned index);
extern "C" __device__ void caller_work(const void* raw, unsigned) {
  const auto* p = static_cast<const CallerParams*>(raw);
  unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  p->output[2 * index] = pipelink_call_get_a(index);
  p->output[2 * index + 1] = pipelink_call_get_b(index);
}
PL_STAGE("org.caller", "1.0.0");
PL_PARAMS(CallerParams, output);
PL_ROLE(call, caller_work, 4);
PL_CALL(get_a, pipelink_call_get_a, pl::produce, "call");
PL_CALL(get_b, pipelink_call_get_b, pl::produce, "call");
''')
    pack_cu(source, tmp_path / "caller.plink", profile="sm_103a")
    manifest, _ = container.read(tmp_path / "caller.plink")
    pipeline = {
        "pipelink_abi": manifest["pipelink_abi"],
        "pipeline": {
            "name": "params-accessors",
            "stages": [{"stage_id": stage, "artifact_path": f"{name}.plink"}
                       for name, stage in (("a", "provider.a"), ("b", "provider.b"), ("caller", "org.caller"))],
            "handles": {"pipelink_call_warps": {"class": "base", "binding": "bind", "domain": "warp", "owner": "claim:call"}},
            "phases": ["call"], "exclusive": [], "domains": {}, "depth_priority": [],
            "regs_priority": [], "forced_bindings": {},
            "topology": {"persistent": False, "scheduler": "static"},
        },
    }
    path = tmp_path / "pipeline.json"
    path.write_bytes(canonical_bytes(pipeline))
    result = link(path, tmp_path / "out")
    return tmp_path / "out", result


def test_roleless_provider_params_link(accessor_link):
    out, result = accessor_link
    assert result.dyn_smem_bytes == 8
    entries = {name: (offset, size, align) for name, offset, size, align in result.params.entries}
    assert entries["provider.a"] == (128, 256, 128)
    assert entries["provider.b"] == (384, 64, 64)
    published = json.loads((out / "link.json").read_text())
    assert published["dyn_smem_bytes"] == 8


def test_roleless_provider_params_cuda_readback(accessor_link):
    try:
        driver = ctypes.CDLL("libcuda.so.1")
        count = ctypes.c_int()
        available = driver.cuInit(0) == 0 and driver.cuDeviceGetCount(ctypes.byref(count)) == 0 and count.value > 0
    except OSError:
        available = False
    if not available:
        pytest.skip("CUDA GPU unavailable")
    major, minor = ctypes.c_int(), ctypes.c_int()
    driver.cuDeviceComputeCapability(ctypes.byref(major), ctypes.byref(minor), 0)
    if (major.value, minor.value) != (10, 3):
        pytest.skip("requires sm_103 GPU")
    out, result = accessor_link
    elements = result.threads * 9
    for left, right in ((17, 902), (-43, 7012)):
        def params(addresses):
            return {"org.caller": struct.pack("<Q", addresses["output"]),
                    "provider.a": struct.pack("<i", left) + bytes(252),
                    "provider.b": struct.pack("<i", right) + bytes(60)}
        with open_hostless(out / "link.json", buffers={"output": ((elements, 2), np.int32)},
                           params=params, grid=(9, 1, 1)) as session:
            session.launch()
            session.synchronize()
            values = session.read("output")
        np.testing.assert_array_equal(values[:, 0], np.arange(elements) + left)
        np.testing.assert_array_equal(values[:, 1], np.arange(elements) + right)
