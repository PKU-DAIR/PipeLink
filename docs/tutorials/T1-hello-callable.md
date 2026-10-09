# T1 — Hello, callable seam

## Goal and prerequisites

Compose two independently packed CUDA stages into
`output[i] = scale(input[i])`, then launch without an example-specific host
executable. Learn which endpoint is `produce`, how host parameter blobs bind
device addresses, and which evidence constitutes acceptance.

Complete [T0](T0-pack-link.md)'s CUDA 13.3, `uv` and `.env` setup first. Runtime
requires a compatible NVIDIA driver and a Blackwell/sm103 (`sm_103a`), Hopper/sm90
(`sm_90a`) or consumer Blackwell/sm120 (`sm_120a`) GPU, plus CUDA's
**Compute Sanitizer**. Run commands from the repository root. The commands and
runtime output below were exercised on all three profiles; each passed the runner.

Select an available GPU of the build's architecture. The runner's `--gpu` is a
**physical** index; omitting it selects the first `.env` `PIPELINK_GPUS` entry.
It selects that GPU before CUDA initializes. Sanitizer children inherit the
selection; clock observations retain the physical index. The direct Python/CLI
recipes below pass the device ordinal explicitly (the physical index on an
unrestricted machine). On another architecture, use that profile's output paths
and an available GPU of that architecture; an image runs only on its own profile.

## 1. Walk through the eight annotations

[`provider.cu`](../../examples/hello-callable/src/provider.cu) implements
`hello_scale(float value)` as `value * 2.0f`. Its four annotations are:

| Annotation | What it promises |
|---|---|
| `PL_STAGE("org.hello.provider", "1.0.0")` | This is a separately packaged, named/versioned stage. |
| `PL_NO_PARAMS()` | The provider requires no host-supplied parameter blob. |
| `PL_EXPORT(scale, hello_scale)` | The implementation is available under public callable name `scale`. |
| `PL_CALL(scale, hello_scale, pl::consume, "scale")` | That exported callable consumes invocations on channel `scale`. |

[`caller.cu`](../../examples/hello-callable/src/caller.cu) declares
`HelloCallerParams` with two pointers and a count. `hello_call` casts its raw
params pointer, flattens all three grid dimensions, and uses a grid-stride loop
to call `pipelink_call_scale` for each input element. Its four annotations are:

| Annotation | What it promises |
|---|---|
| `PL_STAGE("org.hello.caller", "1.0.0")` | This caller is independently packaged from the provider. |
| `PL_PARAMS(HelloCallerParams, input, output, count)` | The public parameter type contains these fields; the compiler observes its layout. |
| `PL_ROLE(call, hello_call, 4)` | Four warps execute the `call` role through `hello_call`. |
| `PL_CALL(scale, pipelink_call_scale, pl::produce, "call")` | Role `call` produces invocations through this unresolved callable symbol. |

**The caller is `produce`; the provider is `consume`.** These terms describe
call traffic, not which side produces array data. The provider has no scheduled
role: the generated trampoline invokes it on the caller's threads. The linker
checks the call contract and binds `pipelink_call_scale` to `hello_scale`; the
caller needs neither provider source nor a provider-private header at pack time.
No TMEM, cluster launch control or external host provider is needed.

## 2. Build the linked and same-source control images

```sh
uv run --project pipec --locked python examples/hello-callable/build.py --profile sm_90a
uv run --project pipec --locked python examples/hello-callable/build.py --profile sm_103a
uv run --project pipec --locked python examples/hello-callable/build.py --profile sm_120a
```

[`build.py`](../../examples/hello-callable/build.py) runs `pipec2 pack` per source,
generates `out/<profile>/pipeline.json`, and invokes `pipec2 link`. It hides GPUs
from compilation subprocesses and takes ABI/schema/target information from the
freshly packed manifest. The pipeline binds `pipelink_call_warps` to
`claim:call`, schedules one `call` phase and selects static, nonpersistent
topology, as in T0.

The shared `onetu_control` helper compiles **the same stage sources and generated
binding/dispatch code** as a whole-program one-TU cubin and a backend-matched LTO
one-TU cubin. No handwritten baseline kernel replaces the caller. Packages,
controls, metadata, logs and results live in the locally ignored
`examples/hello-callable/out/<profile>/`; compiler scratch lives under
`PIPELINK_SCRATCH_ROOT/hello-callable/<profile>/`.

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

The role declaration fixes a `(128,1,1)` block. The host supplies only the grid,
not a competing block dimension. Any positive legal grid covers this loop's
array; the acceptance runner uses `(1024,1,1)`.

## 3. Bind device addresses into stage blobs

[`params.py`](../../examples/hello-callable/params.py) is the host mirror of the
caller's C++ struct. It uses a naturally aligned `ctypes.Structure`: two pointer
fields followed by `ctypes.c_uint32`. On the supported 64-bit host ABI its size
is **24 bytes**, alignment **8**, including tail padding.

The launcher first allocates and initializes every named buffer, **then** calls
`build(addresses)`. The builder receives `{buffer_name: device_address}` and
returns `{"org.hello.caller": bytes(CallerParams(...))}`. There is no provider
blob because it declares `PL_NO_PARAMS`.

`PL_PARAMS` is not a serialization format inferred from field names. The host
must mirror the published C++ layout, including padding. The launcher validates
blob sizes and stage-region alignment/bounds against `link.params.entries`,
then copies the blobs into its aligned aggregate image. It has no per-field
metadata and does not inspect pointer values. Here, the caller blob starts at
offset 0 and the aggregate occupies 64 bytes aligned to 64. The host does not
choose those offsets.

The default builder count is `1 << 24`; Python users may bind another count
with `functools.partial`. Buffer sizes and that count must agree.

## 4. Launch from Python

This small demonstration uses 1,024 elements and binds the count explicitly.
`run_hostless` owns allocation, upload, launch, readback and cleanup in one call.
`open_hostless` returns a persistent context-managed session for repeated
launches or timing without reallocating its buffers.

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

A NumPy input uploads data; a `(shape, dtype)` buffer specification allocates
zeroed storage. `session.read` synchronizes and returns an independent host
array. Session buffers are initialized only once, not reset on every launch;
context-manager exit drains work and releases resources. For measurement,
`session.time(n)` brackets `n` launches with one event pair and returns total
milliseconds. The shared acceptance harness below uses that interface rather
than a Python wall-clock loop.

## 5. Launch directly from the CLI

The CLI executes the same buffer/params route, with no example-specific C++
launcher. Keep **16,777,216** elements here because the CLI calls the builder
with its default count. The first command saves the fixed-seed input and is
silent on success.

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
buffer. `--params PY_FILE:FUNCTION` executes trusted Python after allocation,
and `--output` saves selected buffers. `--device` is the CUDA device ordinal.
These launch/oracle demonstrations do **not** replace
paired timing and sanitizer acceptance.

## 6. Run the acceptance evidence

```sh
uv run --project pipec --locked python examples/hello-callable/run.py --profile sm_103a
```

[`run.py`](../../examples/hello-callable/run.py) writes
`out/sm_103a/results.json`, prints that JSON and exits nonzero if any numerical,
timing-batch-validity or sanitizer gate fails. Its evidence is:

1. Fixed-seed float32 input in `[-1,1]` and zeroed output. Linked and one-TU
   outputs must each match NumPy `input * float32(2)` **bitwise**, and must match
   each other bitwise. Multiplication by two is exact for these bounded inputs.
2. `time_pair` warms both images, calibrates a shared batch to at least
   **15.625 ms**, and scores eight balanced ABBA/BAAB blocks with at least
   **20 launches per batch**. Every scored batch must last at least **12.5 ms**;
   otherwise its status is `BATCH_TOO_SHORT`, which fails this runner. The count
   is `1 << 24` (64 MiB per array), so an empty-launch benchmark is not used as
   seam evidence. It reports median linked/one-TU and one-TU/linked ratios;
   these are observations, not an arbitrary machine-dependent ratio gate.
3. SM/max-SM clocks are observed before and after. The runner does not mutate
   clocks or infer a lock from an environment target. `clock_locked: null`
   means a lock was not attested, even when observations equal the maximum.
4. **Both `racecheck` and `synccheck` run unconditionally on both linked and
   one-TU images.** Each child launches once and checks the full-sized oracle;
   `--single` prevents recursive acceptance runs. There is no sanitizer opt-out.
5. Normalized SASS comparison is **diagnostic only**, never an equality gate.
   If instruction counts differ, the harness also reports the backend-matched
   LTO one-TU comparison to distinguish nvcc/nvJitLink allocation differences.

Measured with CUDA **13.3**; both images share buffers via `HostlessSession.with_image`.
Clocks were observed, not locked:

| Profile | Numeric comparisons | Timing validity and median linked/one-TU | Sanitizers | SASS linked / one-TU, diagnostic only | Observed SM before/after |
|---|---|---|---|---|---|
| Blackwell/sm103 | All three bitwise PASS | PASS; 216 launches/batch; **1.000259** | racecheck + synccheck, linked and one-TU: 4/4 PASS | 64 / 64, equal | 2032/2032 MHz |
| Hopper/sm90 | All three bitwise PASS | PASS; 225 launches/batch; **1.000058** | racecheck + synccheck, linked and one-TU: 4/4 PASS | 64 / 64, equal | 1980/1980 MHz |
| Consumer Blackwell/sm120 | All three bitwise PASS | PASS; 78 launches/batch; **1.000552** | racecheck + synccheck, linked and one-TU: 4/4 PASS | 56 / 56, equal | 2542/2992 MHz |

Ratios, batch calibration and clock observations are measured values, not
expected constants for a rebuild. Current timing blocks and sanitizer log paths
are in each profile's `out/<profile>/results.json`. Retain fresh acceptance evidence for
the images you actually run; a pack/link success or SASS match is insufficient.

## What you declared vs what was derived

| Declared by this example | Derived by PipeLink |
|---|---|
| **8 annotations total**: `PL_STAGE` ×2 names/version-tags the provider and caller. | Independent packages with native ABI-v2 manifests and opaque LTO payloads. |
| `PL_NO_PARAMS` ×1 promises no provider blob; `PL_PARAMS` ×1 declares caller type/fields, mirrored explicitly in host `ctypes`. | Caller blob `(offset=0, bytes=24, align=8)` and aggregate `(bytes=64, align=64)`; size/alignment checks, not field-level inference. |
| `PL_ROLE` ×1 promises four warps for `call`; pipeline handle ownership is `claim:call`. | Warp interval `[0,4)`, **128 threads per block**, and generated dispatch. |
| `PL_EXPORT` ×1 publishes `scale`; `PL_CALL` ×2 declare produce/consume invocation endpoints. | Checked call compatibility and the trampoline binding `pipelink_call_scale` to `hello_scale`. |
| One static, nonpersistent pipeline phase and the build profile. | Profile-specific cubin, entrypoint, register cap, cluster and shared-memory requirements. |
| Input/output buffers, count, params builder and grid. | Allocation before blob construction; aligned aggregate params and link-derived block/shared-memory launch dimensions. |

Previous: [T0 — Install, pack, link, read `link.json`](T0-pack-link.md).
Next: [T2 — A pipeline through an mbarrier ring](T2-ring-pipeline.md).
