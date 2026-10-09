# TileLang environment

Locked frontend project at `frontends/tilelang/` (Python 3.11, CPU Torch 2.8.0,
TileLang 0.1.14). Its virtualenv is `frontends/tilelang/.venv`. TileLang is installed
from a locally built wheel, `env/wheels/tilelang-0.1.14-cp39-abi3-linux_x86_64.whl` (git-ignored), built from
the upstream `v0.1.14` tag plus `tilelang-0.1.14-device-function.patch` and
`tilelang-0.1.14-ws-external-pipeline.patch`, applied in that order.

The wheel and this rebuild recipe support **Linux x86_64 only**. Rebuilds on
other platforms are unvalidated and unsupported.

The device-function patch (11 lines, `src/cuda/codegen/codegen_cuda.{h,cc}`,
`src/transform/common/attr.h`) adds one PrimFunc attribute,
`tl.device_function`: when nonzero, `PrintFuncPrefix` emits
`extern "C" __device__` and `PrintExtraAttrs` emits no `__launch_bounds__`.
Nothing else in codegen changes; functions without the attribute print exactly
as stock 0.1.14. The adapter sets it on the structurally split WS role
functions (`../src/pipelink_tilelang/externalize.py`), so their CUDA text is stock codegen output.

The external-pipeline patch extends scheduled `T.WSPipeline` with keyword
arguments `full_count=0`, `empty_count=0` (derive when zero), `arrive_cta=-1`
(local when negative), and `init_phase=0`. A supplied count bypasses that
side's atom-derived arrival plan: a present signaling role emits per-thread
arrivals, while an absent signaling role is assumed to signal externally by
the cycle/deadlock verifier. Other, local sides retain verification.
`arrive_cta >= 0` routes per-thread arrivals to that cluster CTA; `init_phase`
is xor'd into both wait parities. Barrier initialization still carries the
counts so externalization can retain or drop it according to the contract.
The patch changes `materialize_ws_schedule.cc`, the `WSPipeline` node in
`warp_specialize.h`/`src/ir.cc`, and `tilelang/language/ws_schedule.py`; the build
resets the Python file as well as `src/` before reapplying both patches.

From the repository root, build the wheel once per machine, then sync. Alternatively,
when the release asset is available, place that wheel at
`frontends/tilelang/env/wheels/tilelang-0.1.14-cp39-abi3-linux_x86_64.whl` and skip
the build; retain the published hash as described below.

```sh
source tools/env.sh; set -a; . ./.env; set +a
frontends/tilelang/env/build-tilelang.sh
uv lock --project frontends/tilelang --refresh-package tilelang
uv sync --project frontends/tilelang --locked
```

Until the GitHub release asset exists, the committed lock pins a locally built
wheel hash; source rebuilds can differ in bytes, so refresh the TileLang
lock entry before syncing the newly built wheel. An unchanged published
release wheel must retain its published hash, not be re-locked to bypass a mismatch.

Before copying the wheel to `env/wheels/`, the build runs the stdlib-only
`add_wheel_licenses.py WHEEL SOURCE_TREE`. It preserves TileLang's MIT license,
adds bundled-code licenses/notices under
`tilelang-0.1.14.dist-info/licenses/<component>/`, and rewrites every file's
SHA-256/size entry in `RECORD` (with the required empty self-entry). Replacement
is atomic; rerunning it replaces the added entries rather than duplicating them.

The inventory follows the pinned
[TileLang build](https://github.com/tile-ai/tilelang/blob/v0.1.14/CMakeLists.txt)
and [wheel package mapping](https://github.com/tile-ai/tilelang/blob/v0.1.14/pyproject.toml):
TVM's Apache-2.0 `LICENSE` and `NOTICE` cover `libtvm_compiler.so` and
`libtvm_runtime.so`; CUTLASS/CuTe headers and tools carry BSD-3-Clause;
TVM-FFI and DLPack headers contribute to the compiled code; TVM's compiler-rt
fp16 helpers and nvbench L2-cache-flush helper are compiled into the libraries;
and vLLM-derived headers ship within TVM's packaged sources. The script includes
their upstream license texts and the two helper headers' attribution comments.
Composable Kernel is not present in this pin. Disabled optional TVM backends
are not swept into the inventory. TVM-FFI's shared library (and its libbacktrace),
Z3, and Torch are separate dependencies, not bundled shared libraries in this wheel.
