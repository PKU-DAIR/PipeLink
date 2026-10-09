# T0 — Install, pack, link, read `link.json`

A **stage** is an independently packaged CUDA translation unit: an opaque LTO
object plus a manifest describing its entry roles, parameters, resources and
cross-stage contracts. A stage need not own scheduled work: this tutorial's
provider exports a callable, while its caller owns the scheduled role.

PipeLink uses **pack-attestation trust**: `pack` validates the object symbol
surface and records the attestation; `link` trusts it while checking declarations,
compatibility and resource legality. It does not independently catch a lying or
tampered manifest against its object. `inspect` explicitly checks the payload
digest, but is not required after every pack and does not turn the model into
adversarial verification. See the [compiler trust model](../reference/pack-and-link.md#trust-model).

## Goal and prerequisites

Build two `.plink` packages, inspect their contracts, compose a cubin, and read
its launch requirements **without a GPU**. T1 will launch the same source pair.

Use Linux x86_64 with:

- CUDA **13.3**, including `nvcc`, `nvJitLink` and `cuobjdump`;
- GNU `nm`, a C++17 host compiler (`g++`), Python 3 and `uv`;
- the repository checkout and a writable scratch directory.

Start from the repository's [`.env.example`](../../.env.example) when creating
your machine-local `.env`; preserve an existing configured file. Set
`PIPELINK_CUDA_HOME` to your CUDA 13.3 installation and `PIPELINK_SCRATCH_ROOT` to
a writable high-capacity directory. Values are literal `KEY=VALUE` strings, not
shell expressions: do not put `$HOME` or `~` in those paths. T0 does not need
`PIPELINK_GPUS`; T1 uses it for a default physical GPU selection.

[`tools/pipelink_env.py`](../../tools/pipelink_env.py) applies defaults, then
`.env`, then process-environment overrides. `tools/env.sh` exports those values
and the derived `CUDA_HOME`, `PATH`, `LD_LIBRARY_PATH` and scratch-backed
`UV_CACHE_DIR`. The helper's default scratch is a per-user temporary directory;
setting an explicit root keeps compiler work and caches on your chosen disk.
It does not set `TMPDIR`, so this tutorial sets that separately.

Run all commands from the repository root in the same Bash shell:

```sh
source tools/env.sh
export TMPDIR="$PIPELINK_SCRATCH_ROOT/tutorials"
mkdir -p "$TMPDIR"
uv sync --project pipec --locked
uv run --project pipec --locked pipec2 --help
```

The help output includes these lines (observed 2026-09-28):

```text
usage: pipec2 [-h] {pack,inspect,link,launch,clauses} ...

Pack, inspect, resolve, link, and launch PipeLink ABI-v2 artifacts.
```

Select a target profile. `sm_103a` targets Blackwell/sm103; `sm_90a` targets
Hopper/sm90; `sm_120a` targets consumer Blackwell/sm120. `sm_100a` and `sm_121a`
target the compute capability 10.0 and 12.1 siblings of sm103 and sm120. The pack/link recipes below
run with CUDA 13.3 on every profile; none requires a matching GPU. To repeat for another profile, change the `PROFILE` assignment and repeat the
subsequent commands.

```sh
set -o pipefail
export PROFILE=sm_103a
export T0_OUT="$TMPDIR/t0/$PROFILE"
mkdir -p "$T0_OUT"
```

## 1. Read the source contracts

The pair is [`provider.cu`](../../examples/hello-callable/src/provider.cu) and
[`caller.cu`](../../examples/hello-callable/src/caller.cu). The provider implements
`hello_scale(float)` as multiplication by two. The caller's grid-stride loop
reads `input[i]`, invokes the unresolved `pipelink_call_scale`, and writes
`output[i]`. It does not include the provider source or a provider-private header.

There are **eight annotation invocations**, four per source:

| Source / annotation | What it declares |
|---|---|
| Provider: `PL_STAGE("org.hello.provider", "1.0.0")` | The provider's stage identity and version. |
| Provider: `PL_NO_PARAMS()` | No host parameter blob is needed for this stage. |
| Provider: `PL_EXPORT(scale, hello_scale)` | Public callable `scale` is implemented by `hello_scale`. |
| Provider: `PL_CALL(scale, hello_scale, pl::consume, "scale")` | The exported callable consumes invocations on channel `scale`. |
| Caller: `PL_STAGE("org.hello.caller", "1.0.0")` | The caller's separate stage identity and version. |
| Caller: `PL_PARAMS(HelloCallerParams, input, output, count)` | The caller's public C++ parameter type and fields. |
| Caller: `PL_ROLE(call, hello_call, 4)` | Role `call` enters `hello_call` with four warps. |
| Caller: `PL_CALL(scale, pipelink_call_scale, pl::produce, "call")` | Role `call` produces invocations through the unresolved symbol. |

`produce` and `consume` describe **call traffic**, not array-data flow. The
provider has no scheduled role: its function executes on the caller's threads.

## 2. Pack each translation unit

The profile selects the target/toolkit contract and the architecture spellings
used by the CUDA compilation and symbol probes. Device visibility is explicitly
empty. The projections below print the meaningful pack summary, not provenance
digests or machine-specific compiler paths.

```sh
uv run --project pipec --locked pipec2 pack \
  examples/hello-callable/src/provider.cu --profile "$PROFILE" -o "$T0_OUT/provider.plink" \
  | python3 -c 'import json,sys; p=json.load(sys.stdin); print(p["stage_id"], p["symbol_mechanism"])'
uv run --project pipec --locked pipec2 pack \
  examples/hello-callable/src/caller.cu --profile "$PROFILE" -o "$T0_OUT/caller.plink" \
  | python3 -c 'import json,sys; p=json.load(sys.stdin); print(p["stage_id"], p["symbol_mechanism"])'
```

Observed output, for either profile:

```text
org.hello.provider nm-external-text+nvjitlink-seeded-v2
org.hello.caller nm-external-text+nvjitlink-seeded-v2
```

Pack compiles/runs the host declaration projection, compiles the device LTO
payload, and performs the `nm` plus seeded `-O0` nvJitLink symbol probe. The
`.plink` contains a manifest and `stage.o`; the stage's source is not a link input.

## 3. Inspect the manifest

`inspect --json` checks the package and emits the manifest. This projection keeps
the identity, ABI, layout and call/symbol surfaces; it does not use a digest as a
rebuild identity.

```sh
for stage in provider caller; do
  uv run --project pipec --locked pipec2 inspect "$T0_OUT/$stage.plink" --json \
    | python3 -c '
import json, sys
m = json.load(sys.stdin)
print("stage:", m["stage"]["id"], m["stage"]["version"])
print("ABI:", m["pipelink_abi"])
print("params:", {k: m["params"][k] for k in ("type", "bytes", "align")})
print("roles:", m["stage"]["roles"])
print("exports:", {k: v["symbol"] for k, v in m["stage"]["exports"].items()})
print("calls:", [(e["name"], e["direction"], e["owner_role"], e["protocol"]["signature"]) for e in m["endpoints"]])
print("symbols:", m["symbols"])
'
done
```

Observed Blackwell/sm103 output:

```text
stage: org.hello.provider 1.0.0
ABI: {'schema': '3.0', 'target': {'lto_arch': 'lto_103a', 'profile': 'sm_103a', 'toolkit': 'cuda-13.3.0'}}
params: {'type': 'void', 'bytes': 0, 'align': 1}
roles: {}
exports: {'scale': 'hello_scale'}
calls: [('scale', 'consume', None, 'float(float)')]
symbols: {'defined_strong': ['hello_scale'], 'defined_weak': [], 'mechanism': 'nm-external-text+nvjitlink-seeded-v2', 'undefined': {}}
stage: org.hello.caller 1.0.0
ABI: {'schema': '3.0', 'target': {'lto_arch': 'lto_103a', 'profile': 'sm_103a', 'toolkit': 'cuda-13.3.0'}}
params: {'type': 'HelloCallerParams', 'bytes': 24, 'align': 8}
roles: {'call': {'entry': 'hello_call', 'warps': 4}}
exports: {}
calls: [('scale', 'produce', 'call', 'float(float)')]
symbols: {'defined_strong': ['hello_call'], 'defined_weak': [], 'mechanism': 'nm-external-text+nvjitlink-seeded-v2', 'undefined': {'pipelink_call_scale': 'function'}}
```

Read it as follows:

- `stage.id`/`version` identify the stage contract; `entry_abi` is native `v2`.
  `pipelink_abi` separately carries schema and target/toolkit compatibility.
- `params` is compiler-observed size/alignment, not host-side field inference.
  Two pointers and a 32-bit count require 24 bytes including tail padding.
- `stage.roles` schedules `hello_call` on four warps. Pack also derives its warp
  claim, `pipelink_call_warps`; the provider's role map is empty.
- `stage.exports` publishes the implementation. The `endpoints` call contracts
  connect the caller's `produce` side to the provider's `consume` side and agree
  on `float(float)`.
- `symbols` attests strong/weak definitions and unresolved references.
  `pipelink_call_scale` is intentionally unresolved in the caller package;
  the linker will generate its binding. This is why a callable can be packed
  independently of its provider.

## 4. Describe the composition and link it

Write the pipeline next to the packages so its relative `artifact_path` entries
resolve there. Copy ABI/schema/target information from the freshly packed
manifest, rather than maintaining another toolkit constant.

```sh
uv run --project pipec --locked python - <<'PY'
import json, os
from pathlib import Path
from pipec2.container import read
out = Path(os.environ["T0_OUT"])
caller, _ = read(out / "caller.plink")
pipeline = {
    "pipelink_abi": caller["pipelink_abi"],
    "pipeline": {
        "name": "hello-callable",
        "stages": [
            {"stage_id": "org.hello.provider", "artifact_path": "provider.plink"},
            {"stage_id": "org.hello.caller", "artifact_path": "caller.plink"},
        ],
        "handles": {"pipelink_call_warps": {
            "class": "base", "binding": "bind", "domain": "warp", "owner": "claim:call",
        }},
        "phases": ["call"],
        "exclusive": [], "domains": {}, "depth_priority": [], "regs_priority": [],
        "forced_bindings": {},
        "topology": {"persistent": False, "scheduler": "static"},
    },
}
(out / "pipeline.json").write_text(json.dumps(pipeline, indent=2) + "\n")
print("pipeline.json:", pipeline["pipelink_abi"]["target"]["profile"], "2 stages, phase call")
PY
uv run --project pipec --locked pipec2 link "$T0_OUT/pipeline.json" --out-dir "$T0_OUT/link" \
  | python3 -c 'import json,sys; r=json.load(sys.stdin); print("linked:", r["kernel_symbol"], "threads:", r["threads"])'
```

Observed output:

```text
pipeline.json: sm_103a 2 stages, phase call
linked: pipelink_kernel threads: 128
```

The handle is owned by the pack-derived `claim:call`, but its base is not
hand-assigned. The linker solves that base and the warp interval. `phases` places
`call` in the dispatch schedule; `static`, nonpersistent topology needs no host
scheduler. The remaining empty collections introduce no extra resource domains,
exclusions, optimization priorities or forced bindings for this example.

The output is `link/pipeline.cubin` plus `link/link.json` and generated sources.
Read the metadata rather than selecting your own block size:

```sh
python3 - <<'PY'
import json, os
from pathlib import Path
r = json.loads((Path(os.environ["T0_OUT"]) / "link/link.json").read_text())
for key in ("kernel_symbol", "threads", "dyn_smem_bytes", "regs_cap", "cluster", "params", "intervals"):
    print(key + ":", r[key])
print("protocol_expansions:", [
    {"channel": p["channel"], "kind": p["kind"], "callable": p["callable"][:2]}
    for p in r["protocol_expansions"]
])
print("generated sources:", sorted(r["generated_hashes"]))
PY
```

Observed output, identical for these fields on every profile:

```text
kernel_symbol: pipelink_kernel
threads: 128
dyn_smem_bytes: 0
regs_cap: 24
cluster: [1, 1, 1]
params: {'align': 64, 'entries': [['org.hello.caller', 0, 24, 8], ['org.hello.provider', 64, 0, 1]], 'total_bytes': 64}
intervals: {'warp': [{'handle': 'pipelink_call_warps', 'hi': 4, 'lo': 0, 'phase': 'kernel'}]}
protocol_expansions: [{'channel': 'scale', 'kind': 'call', 'callable': ['hello_scale', 'extern "C" __device__ float hello_scale(float);']}]
generated sources: ['binding_stub.cu', 'dispatch.cu']
```

| Field / output | How to use it |
|---|---|
| `kernel_symbol` | CUDA entrypoint to look up in `pipeline.cubin`. |
| `threads` | Required block size `(128,1,1)`, derived from four warps. |
| `dyn_smem_bytes` | Required dynamic shared memory, zero here. |
| `regs_cap` | Link-selected register cap, not a measurement of actual register use. |
| `cluster` | Cluster dimensions selected by the link, `(1,1,1)` here. |
| `params.entries` | Each entry is `[stage_id, offset, bytes, align]`; the caller occupies bytes 0–23. The provider's zero-byte entry requires no blob. |
| `params.total_bytes` / `align` | Aggregate launch parameter image: 64 bytes aligned to 64. The host supplies stage blobs, not their aggregate offsets. |
| `protocol_expansions` | Resolved call channel and implementation declaration. There is no barrier protocol in T0. |
| `generated/binding_stub.cu` | Resolves the warp handle and binds `pipelink_call_scale` to `hello_scale`. |
| `generated/dispatch.cu` | Defines the aligned params image and `pipelink_kernel`, dispatching warps `[0,4)` into `hello_call`. |

`generated_hashes` names the generated files; the command prints only its keys.
Its values and `input_hashes` are provenance, not pinned rebuild identities.

## 5. Find diagnostic clauses

```sh
uv run --project pipec --locked pipec2 clauses
```

Selected lines from the observed catalogue:

```text
L-HANDLE: handle declarations and references are complete and consistent
L-PARAMS: stage and aggregate parameter layouts satisfy the target ABI
L-PHASE: claim phases and pairwise exclusions are finite and valid
L-PROTO: paired endpoint protocols expand consistently
L-TOPO: stage topology constraints have a legal intersection
P-IMPORT: artifact imports are declared PipeLink symbols
T-PLAN: bounded tensor synthesis records exact outcomes and limits
```

This is a diagnostic reference, not evidence that a GPU executed correctly.
[T3](T3-contract-gallery.md) deliberately violates contracts and checks failures
at their actual pack/link phases. T0's evidence is successful independent
packing, inspection and GPU-free linking for every profile; numerical and
sanitizer acceptance starts in T1.

## What you declared vs what was derived

| Declared by this example | Derived by PipeLink |
|---|---|
| **8 annotations total**: `PL_STAGE` ×2 names/version-tags the stages. | Separate manifest/object packages and stage-scoped contracts. |
| `PL_NO_PARAMS` ×1 promises no provider blob; `PL_PARAMS` ×1 publishes the caller type/fields. | Compiler-observed sizes/alignments; caller `(offset=0, bytes=24, align=8)` in a 64-byte aggregate image. |
| `PL_ROLE` ×1 promises four warps entering `hello_call`. | Warp claim, solved interval `[0,4)`, 128-thread block and generated dispatch. |
| `PL_EXPORT` ×1 publishes `scale`; `PL_CALL` ×2 promise matching produce/consume call endpoints. | Signature checks and the generated `pipelink_call_scale` → `hello_scale` binding. |
| Pipeline package paths, one phase, handle ownership and static topology; selected profile. | Legal resource binding, register cap, cluster/shared-memory requirements and target cubin. |

Next: [T1 — Hello, callable seam](T1-hello-callable.md).
