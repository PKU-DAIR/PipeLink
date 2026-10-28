# T1 — Hello, callable seam

## Goal and prerequisites

Compose two independently packed CUDA stages into
`output[i] = scale(input[i])` and launch the result with PipeLink's generic
launcher, so the example needs no host executable of its own. Along the way you
will see which endpoint is `produce`, how host parameter blobs carry device
addresses, and what evidence the acceptance runner collects before it accepts a
linked image.

Complete [T0](T0-pack-link.md)'s CUDA 13.3, `uv` and `.env` setup first. Running
needs a compatible NVIDIA driver, a Blackwell/sm103 (`sm_103a`), Hopper/sm90
(`sm_90a`) or consumer Blackwell/sm120 (`sm_120a`) GPU, and CUDA's
**Compute Sanitizer**. Run commands from the repository root. Everything below
was run on all three profiles, and each passed the runner.

An image runs only on its own profile, so pick an available GPU of that
architecture and use that profile's output paths. The runner's `--gpu` takes a
**physical** index and selects it before CUDA initializes; without it, the first
`PIPELINK_GPUS` entry in `.env` is used. Sanitizer children inherit the
selection, and clock readings record the physical index. The Python and CLI
recipes take a CUDA device ordinal instead (equal to the physical index when no
devices are hidden).

## 1. Walk through the seven annotations

[`provider.cu`](../../examples/hello-callable/src/provider.cu) implements
`hello_scale(float value)` as `value * 2.0f`. Its three annotations are:

| Annotation | What it promises |
|---|---|
| `PL_STAGE("org.hello.provider", "1.0.0")` | A separately packaged stage with this name and version. |
| `PL_NO_PARAMS()` | The provider requires no host-supplied parameter blob. |
| `PL_CALL(scale, hello_scale, pl::consume, "scale")` | The implementation is available under public callable name `scale` and consumes invocations on channel `scale`. |

[`caller.cu`](../../examples/hello-callable/src/caller.cu) declares
`HelloCallerParams` with two pointers and a count. `hello_call` casts its raw
params pointer, flattens all three grid dimensions, and uses a grid-stride loop
to call `pipelink_call_scale` for each input element. Its four annotations are:

| Annotation | What it promises |
|---|---|
| `PL_STAGE("org.hello.caller", "1.0.0")` | The caller, packaged separately from the provider. |
| `PL_PARAMS(HelloCallerParams, input, output, count)` | The public parameter type contains these fields; the compiler observes its layout. |
| `PL_ROLE(call, hello_call, 4)` | Four warps execute the `call` role through `hello_call`. |
| `PL_CALL(scale, pipelink_call_scale, pl::produce, "call")` | Role `call` produces invocations through this unresolved callable symbol. |

**The caller is `produce`; the provider is `consume`.** The terms describe call
traffic. The provider has no scheduled role: the generated trampoline runs it on
the caller's threads. The linker checks the call contract and binds
`pipelink_call_scale` to `hello_scale`, so the caller packs without any provider
source or header. No TMEM, cluster launch control or external host provider is
involved.

## 2. Build the linked and same-source control images

```sh
uv run --project pipec --locked python examples/hello-callable/build.py --profile sm_90a
uv run --project pipec --locked python examples/hello-callable/build.py --profile sm_103a
uv run --project pipec --locked python examples/hello-callable/build.py --profile sm_120a
```

[`build.py`](../../examples/hello-callable/build.py) packs each source, writes
`out/<profile>/pipeline.json` with `pipelink_abi` copied from the packed
manifest, and runs `pipec2 link`. As in T0, the pipeline gives
`pipelink_call_warps` to `claim:call`, schedules one `call` phase and uses static,
nonpersistent topology.

For comparison, the shared `onetu_control` helper compiles **the same stage
sources and generated binding/dispatch code** into two single-translation-unit
(one-TU) controls: a whole-program cubin and a backend-matched LTO cubin. The
controls run the caller exactly as written, with no handwritten baseline kernel
in its place, so any difference you measure later comes from separate packing
and linking. Packages, controls, metadata, logs and results go to the ignored `examples/hello-callable/out/<profile>/`; compiler
scratch goes under `PIPELINK_SCRATCH_ROOT/hello-callable/<profile>/`.

Both builds printed these fields (trimmed from their JSON summaries):

```text
  "threads": 128,
  "dyn_smem_bytes": 0,
    "total_bytes": 64
  "controls": {
    "one_tu": "link/one-tu.cubin",
    "one_tu_lto": "link/one-tu-lto.cubin"
  }
```

The role declaration fixes a `(128,1,1)` block; the host supplies only the
grid. Any positive grid covers the array because the loop is grid-stride; the
acceptance runner uses `(1024,1,1)`.

## 3. Bind device addresses into stage blobs

[`params.py`](../../examples/hello-callable/params.py) mirrors the caller's C++
struct as a naturally aligned `ctypes.Structure`: two pointer fields followed by
`ctypes.c_uint32`. On the supported 64-bit host ABI it is **24 bytes** with
alignment **8**, including tail padding.

The launcher allocates and initializes every named buffer, **then** calls
`build(addresses)` with `{buffer_name: device_address}`. The builder returns
`{"org.hello.caller": bytes(CallerParams(...))}`; the provider declares
`PL_NO_PARAMS` and has no blob.

`PL_PARAMS` publishes the field names, but PipeLink does not serialize from them:
the host must reproduce the published C++ layout byte for byte, padding
included. The launcher has no per-field metadata and does not look at pointer
values. It checks each blob's size and its region's alignment and bounds
against `link.params.entries`, then copies the blobs into the aggregate image
at the offsets the link chose. Here the caller blob sits at offset 0 in a
64-byte image aligned to 64; the host never chooses those offsets.

The default builder count is `1 << 24`; Python users may bind another count
with `functools.partial`. Buffer sizes and that count must agree.

## 4. Launch from Python

This demonstration uses 1,024 elements and binds the count explicitly.
`run_hostless` allocates, uploads, launches, reads back and cleans up in one
call. `open_hostless` returns a context-managed session that keeps its buffers
for repeated launches or timing.

```sh
uv run --project pipec --locked python - <<'PY'
from functools import partial
from pathlib import Path
import sys
import numpy as np
from pipec2.launcher import open_hostless, run_hostless
sys.path.insert(0, str(Path("examples/hello-callable")))
from params import build

inputs = np.arange(1024, dtype=np.float32)
expected = inputs * np.float32(2)
link = "examples/hello-callable/out/sm_103a/link/link.json"
kwargs = dict(
    buffers={"input": inputs, "output": (inputs.shape, inputs.dtype)},
    params=partial(build, count=inputs.size), grid=(4, 1, 1),
)
outputs = run_hostless(link, **kwargs, outputs=("output",))
assert np.array_equal(outputs["output"].view(np.uint32), expected.view(np.uint32))
with open_hostless(link, **kwargs) as session:
    session.launch()
    actual = session.read("output")
    assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32))
print("Python oracle: bitwise PASS (run_hostless and open_hostless)")
PY
```

Observed output:

```text
Python oracle: bitwise PASS (run_hostless and open_hostless)
```

A NumPy array uploads data; a `(shape, dtype)` pair allocates zeroed storage.
`session.read` synchronizes and returns a host copy. Session buffers are
initialized once and keep their contents across launches; leaving the context
drains work and frees resources. `session.time(n)` brackets `n` launches with
one CUDA event pair and returns total milliseconds; the acceptance harness below
times with it.

## 5. Launch directly from the CLI

The CLI follows the same buffer and params route. Keep **16,777,216** elements
here because the CLI calls the builder with its default count. The first
command saves a fixed-seed input and prints nothing.

```sh
uv run --project pipec --locked python -c 'import numpy as np; from pathlib import Path; p=Path("examples/hello-callable/out/sm_103a"); np.save(p/"input.npy", np.random.default_rng(2026).uniform(-1,1,1<<24).astype(np.float32))'
uv run --project pipec --locked pipec2 launch examples/hello-callable/out/sm_103a/link/link.json \
  --grid 1024,1,1 \
  --buffer input=examples/hello-callable/out/sm_103a/input.npy \
  --buffer output=16777216:float32 \
  --params examples/hello-callable/params.py:build \
  --output output=examples/hello-callable/out/sm_103a/output.npy
uv run --project pipec --locked python -c 'import numpy as np; from pathlib import Path; p=Path("examples/hello-callable/out/sm_103a"); expected=np.load(p/"input.npy")*np.float32(2); actual=np.load(p/"output.npy"); assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32)); print("CLI oracle: bitwise PASS")'
```

Observed output:

```text
{"geometry": {"block": [128, 1, 1], "cluster": [1, 1, 1], "dynamic_smem_bytes": 0, "grid": [1024, 1, 1]}, "outputs": {"output": "examples/hello-callable/out/sm_103a/output.npy"}}
CLI oracle: bitwise PASS
```

With `sm_90a` or `sm_120a` paths the same recipe reports the same block
**[128, 1, 1]**, grid **[1024, 1, 1]**, dynamic shared memory **0 bytes**, and
`CLI oracle: bitwise PASS`.

`--buffer NAME=FILE.npy` uploads an array; `NAME=SHAPE:DTYPE` allocates a zeroed
buffer. `--params PY_FILE:FUNCTION` runs trusted Python after allocation, and
`--output` saves selected buffers. `--device` is the CUDA device ordinal.

The Python and CLI launches show that the linked image computes the right
answer. They do not time it against a control or run it under the sanitizers;
the acceptance runner does both.

## 6. Run the acceptance evidence

```sh
uv run --project pipec --locked python examples/hello-callable/run.py --profile sm_103a
```

[`run.py`](../../examples/hello-callable/run.py) writes and prints
`out/sm_103a/results.json`, and exits nonzero if any numerical, timing-batch or
sanitizer check fails. It checks:

1. **Numerics.** Fixed-seed float32 input in `[-1,1]`, zeroed output. Linked and
   one-TU outputs must each match NumPy `input * float32(2)` **bitwise**, and
   match each other bitwise. Multiplying by two is exact for these inputs.
2. **Timing.** `time_pair` warms both images, calibrates a shared batch to at
   least **15.625 ms**, and times eight balanced ABBA/BAAB blocks with at least
   **20 launches per batch**. A timed batch shorter than **12.5 ms** gets status
   `BATCH_TOO_SHORT` and fails the runner. The count is `1 << 24` (64 MiB per
   array), so the kernel does real work and the comparison does not reduce to
   launch overhead. The reported speedup is the median
   per-block one-TU/linked time ratio (higher is faster); it is recorded, with
   no pass threshold.
3. **Clocks.** SM and max-SM clocks are read before and after. The runner never
   changes them and does not infer a lock from an environment setting; only the
   caller can attest one. `clock_locked: null` means no lock was attested, even
   if the readings equal the maximum.
4. **Sanitizers.** `racecheck` and `synccheck` always run on both the linked and
   one-TU images. Each child launches once and checks the full-sized oracle;
   `--single` keeps children from recursing.
5. **SASS.** The normalized SASS comparison is **diagnostic only**. If
   instruction counts differ, the harness also compares against the
   backend-matched LTO one-TU control to separate nvcc and nvJitLink allocation
   differences.

The [example's evidence table](../../examples/hello-callable/README.md#evidence)
lists numerical comparisons, median one-TU/linked speedups, batch sizes and clock
readings for each profile. Both images share buffers through
`HostlessSession.with_image`, and the timing helper reads clocks while the
caller attests whether they were locked.

Ratios, batch sizes and clocks are measurements, so expect different values
when you rebuild and rerun. Each profile's `out/<profile>/results.json` holds
the timing blocks and sanitizer log paths. Run the acceptance runner on the
images you actually use: a successful pack and link, or a matching SASS diff,
does not show that an image executes correctly.

## What you declared vs what was derived

| Declared by this example | Derived by PipeLink |
|---|---|
| **7 annotations total**: `PL_STAGE` ×2 name and version the provider and caller. | Independent packages with native ABI-v2 manifests and opaque LTO payloads. |
| `PL_NO_PARAMS` ×1 promises no provider blob; `PL_PARAMS` ×1 declares caller type/fields, mirrored explicitly in host `ctypes`. | Caller blob `(offset=0, bytes=24, align=8)` and aggregate `(bytes=64, align=64)`; size and alignment checks only. |
| `PL_ROLE` ×1 promises four warps for `call`; pipeline handle ownership is `claim:call`. | Warp interval `[0,4)`, **128 threads per block**, and generated dispatch. |
| `PL_CALL` ×2 declare produce/consume invocation endpoints; the consuming one publishes `scale`. | Checked call compatibility and the trampoline binding `pipelink_call_scale` to `hello_scale`. |
| One static, nonpersistent pipeline phase and the build profile. | Profile-specific cubin, entrypoint, register cap, cluster and shared-memory requirements. |
| Input/output buffers, count, params builder and grid. | Allocation before blob construction; aligned aggregate params and link-derived block/shared-memory launch dimensions. |

Previous: [T0 — Install, pack, link, read `link.json`](T0-pack-link.md).
Next: [T2 — A pipeline through an mbarrier ring](T2-ring-pipeline.md).
