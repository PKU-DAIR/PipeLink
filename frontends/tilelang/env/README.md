# TileLang environment

The frontend project at `frontends/tilelang/` is locked to Python 3.11, CPU Torch
2.8.0 and TileLang 0.1.14, with its virtualenv in `frontends/tilelang/.venv`.
TileLang installs from a locally built, git-ignored wheel,
`env/wheels/tilelang-0.1.14-cp39-abi3-linux_x86_64.whl`, built from the upstream
`v0.1.14` tag with `tilelang-0.1.14-device-function.patch` and then
`tilelang-0.1.14-ws-external-pipeline.patch` applied. The wheel and this recipe
support **Linux x86_64 only**; rebuilds on other platforms are untested and unsupported.

The device-function patch (11 lines, `src/cuda/codegen/codegen_cuda.{h,cc}`,
`src/transform/common/attr.h`) adds one PrimFunc attribute,
`tl.device_function`: when nonzero, `PrintFuncPrefix` emits
`extern "C" __device__` and `PrintExtraAttrs` emits no `__launch_bounds__`.
That is the patch's only codegen change: functions without the attribute print
exactly as in stock 0.1.14. The adapter sets it on the structurally
split warp-specialized (WS) role functions
(`../src/pipelink_tilelang/externalize.py`), so their CUDA text is stock codegen
output apart from those two effects.

The external-pipeline patch extends scheduled `T.WSPipeline` with keyword
arguments `full_count=0`, `empty_count=0` (derive when zero), `arrive_cta=-1`
(local when negative), and `init_phase=0`. A supplied count replaces that side's
atom-derived arrival plan: if the signaling role is present it emits per-thread
arrivals; if it is absent, the cycle/deadlock verifier assumes it signals
externally. Local sides, those without a supplied count, are still verified.
`arrive_cta >= 0` routes per-thread arrivals to that cluster CTA; `init_phase` is
xor'd into both wait parities. Barrier initialization still carries the counts, so
externalization can keep or drop it as the contract requires. The patch changes
`materialize_ws_schedule.cc`, the `WSPipeline` node in
`warp_specialize.h`/`src/ir.cc`, and `tilelang/language/ws_schedule.py`; the build
resets that Python file and `src/` before reapplying both patches.

From the repository root, build the wheel once per machine, then sync. Once the
release asset is available, you can instead place it at
`frontends/tilelang/env/wheels/tilelang-0.1.14-cp39-abi3-linux_x86_64.whl` and skip
the build, keeping its published hash (see below).

```sh
source tools/env.sh; set -a; . ./.env; set +a
frontends/tilelang/env/build-tilelang.sh
uv lock --project frontends/tilelang --refresh-package tilelang
uv sync --project frontends/tilelang --locked
```

Until the GitHub release asset exists, the committed lock pins a locally built
wheel hash. Source rebuilds can differ in bytes, so refresh the TileLang lock entry
before syncing a newly built wheel. A published release wheel keeps its published
hash; if it does not match, do not re-lock to get past the mismatch.

Before copying the wheel to `env/wheels/`, the build runs the stdlib-only
`add_wheel_licenses.py WHEEL SOURCE_TREE`. It preserves TileLang's MIT license,
adds bundled-code licenses/notices under
`tilelang-0.1.14.dist-info/licenses/<component>/`, and rewrites every file's
SHA-256/size entry in `RECORD` (with the required empty self-entry). The rewrite
is atomic, and rerunning it replaces the added entries instead of duplicating them.

The inventory follows the pinned
[TileLang build](https://github.com/tile-ai/tilelang/blob/v0.1.14/CMakeLists.txt)
and [wheel package mapping](https://github.com/tile-ai/tilelang/blob/v0.1.14/pyproject.toml):
TVM's Apache-2.0 `LICENSE` and `NOTICE` cover `libtvm_compiler.so` and
`libtvm_runtime.so`; CUTLASS/CuTe headers and tools carry BSD-3-Clause;
TVM-FFI and DLPack headers contribute to the compiled code; TVM's compiler-rt
fp16 helpers and nvbench L2-cache-flush helper are compiled into the libraries;
and vLLM-derived headers ship within TVM's packaged sources. The script includes
their upstream license texts and the two helper headers' attribution comments.
The inventory leaves out Composable Kernel, which this pin does not contain, and
disabled optional TVM backends. TVM-FFI's shared library (with its libbacktrace),
Z3 and Torch are separate dependencies and are not bundled in this wheel.
