# T0 — Install, pack, link, read `link.json`

A **stage** is an independently packaged CUDA translation unit: an opaque LTO
object plus a manifest describing its entry roles, parameters, resources and
cross-stage contracts. In this tutorial the provider exports a callable and the
caller owns the scheduled role.

PipeLink uses **pack-attestation trust**. `pack` validates the object's symbol
surface and records that attestation in the manifest. `link` then trusts the
attestation while it checks declarations, compatibility and resource legality.
Link has no independent way to tell that a manifest misdescribes its object, so
a lying or tampered manifest gets through. `inspect` verifies the payload digest
whenever you run it; it is an optional integrity check and leaves the trust
model as it is. See the
[compiler trust model](../reference/pack-and-link.md#trust-model).

## Goal and prerequisites

Build two `.plink` packages, inspect their contracts, link a cubin and read its
launch requirements, all **without a GPU**. T1 launches the same source pair.

Use Linux x86_64 with:

- CUDA **13.3**, including `nvcc`, `nvJitLink` and `cuobjdump`;
- GNU `nm`, a C++17 host compiler (`g++`), Python 3 and `uv`;
- the repository checkout and a writable scratch directory.

If you have no machine-local `.env` yet, copy [`.env.example`](../../.env.example).
Set `PIPELINK_CUDA_HOME` to your CUDA 13.3 installation and
`PIPELINK_SCRATCH_ROOT` to a writable directory with plenty of space. Values are
literal `KEY=VALUE` strings, so write paths out in full (no `$HOME` or `~`).
T0 needs no GPU, so you can leave `PIPELINK_GPUS` for T1, which uses it to pick
a default device.

[`tools/pipelink_env.py`](../../tools/pipelink_env.py) layers defaults, `.env`
and the process environment, in that order, with later sources overriding
earlier ones. `tools/env.sh` exports the result plus `CUDA_HOME`, `PATH`,
`LD_LIBRARY_PATH` and a scratch-backed `UV_CACHE_DIR`. Without an explicit
`PIPELINK_SCRATCH_ROOT` the helper falls back to a per-user temporary directory;
setting one keeps compiler work and caches on the disk you chose. The helper
leaves `TMPDIR` alone, so the commands below set it.

Run all commands from the repository root in the same Bash shell:

```sh
source tools/env.sh
export TMPDIR="$PIPELINK_SCRATCH_ROOT/tutorials"
mkdir -p "$TMPDIR"
uv sync --project pipec --locked
uv run --project pipec --locked pipec2 --help
```

The help output includes these lines:

```text
usage: pipec2 [-h] {pack,inspect,link,launch,clauses} ...

Pack, inspect, resolve, link, and launch PipeLink ABI-v2 artifacts.
```

Select a target profile: `sm_103a` (Blackwell/sm103), `sm_90a` (Hopper/sm90) or
`sm_120a` (consumer Blackwell/sm120). `sm_100a` and `sm_121a` target the compute
capability 10.0 and 12.1 siblings of sm103 and sm120. Every profile packs and
links on any machine with CUDA 13.3. To try another profile, change `PROFILE`
and rerun the commands that follow.

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
reads `input[i]`, calls the unresolved `pipelink_call_scale`, and writes
`output[i]`, without including any provider source or header.

There are **seven annotations**, three in the provider and four in the caller:

| Source / annotation | What it declares |
|---|---|
| Provider: `PL_STAGE("org.hello.provider", "1.0.0")` | The provider's stage identity and version. |
| Provider: `PL_NO_PARAMS()` | No host parameter blob is needed for this stage. |
| Provider: `PL_CALL(scale, hello_scale, pl::consume, "scale")` | Public callable `scale`, implemented by `hello_scale`, consumes invocations on channel `scale`. |
| Caller: `PL_STAGE("org.hello.caller", "1.0.0")` | The caller's separate stage identity and version. |
| Caller: `PL_PARAMS(HelloCallerParams, input, output, count)` | The caller's public C++ parameter type and fields. |
| Caller: `PL_ROLE(call, hello_call, 4)` | Role `call` enters `hello_call` with four warps. |
| Caller: `PL_CALL(scale, pipelink_call_scale, pl::produce, "call")` | Role `call` produces invocations through the unresolved symbol. |

`produce` and `consume` describe **call traffic**: the caller produces
invocations and the provider consumes them. The provider has no scheduled role;
its function runs on the caller's threads.

## 2. Pack each translation unit

Each translation unit packs on its own; neither command needs the other source.
The profile sets the target, toolkit and the architecture flags used by the CUDA
compilation and symbol probes, and pack runs those steps with no GPUs visible.
The `python3` filters print the stage ID and symbol mechanism from each pack
summary and drop the provenance digests and machine-specific compiler paths,
which differ from one machine to the next.

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

Pack compiles and runs the host declaration projection, compiles the device LTO
payload, and probes its symbols with `nm` and a seeded `-O0` nvJitLink link. The
`.plink` holds a manifest and `stage.o`. That is everything link consumes; the
source itself is not a link input.

## 3. Inspect the manifest

`inspect --json` checks the package and prints its manifest. The filter keeps
the parts that describe the stage's contract: identity, ABI, parameter layout,
roles, calls and symbols.

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

Read the two manifests field by field:

- `stage.id`/`version` identify the stage contract; `entry_abi` is native `v2`.
  `pipelink_abi` separately carries the schema and target/toolkit
  compatibility.
- `params` is the size and alignment the compiler observed for the parameter
  type, so the host never has to infer the field layout. Two pointers and a
  32-bit count take 24 bytes including tail padding.
- `stage.roles` schedules `hello_call` on four warps, and pack derives the warp
  claim `pipelink_call_warps` from it. The provider has no roles.
- `stage.exports` publishes the implementation. The two `endpoints` connect the
  caller's `produce` side to the provider's `consume` side with matching
  signature `float(float)`.
- `symbols` lists strong/weak definitions and unresolved references. The
  caller leaves `pipelink_call_scale` unresolved for the linker to bind, which
  is what lets it pack without the provider.

## 4. Describe the composition and link it

Write the pipeline next to the packages so its relative `artifact_path` entries
resolve. The script copies `pipelink_abi` from the packed manifest, so no
toolkit constant is hard-coded.

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

The handle is owned by the pack-derived `claim:call`; the linker solves its
base and warp interval. `phases` places `call` in the dispatch schedule, and
static nonpersistent topology needs no host scheduler. The empty collections
add no domains, exclusions, priorities or forced bindings.

Link writes `link/pipeline.cubin`, `link/link.json` and generated sources. The
metadata states the launch configuration the cubin requires, so read the block
size and the other launch parameters from it instead of choosing your own:

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
| `regs_cap` | Register cap selected by the link (an upper bound, not measured use). |
| `cluster` | Cluster dimensions selected by the link, `(1,1,1)` here. |
| `params.entries` | Each entry is `[stage_id, offset, bytes, align]`; the caller occupies bytes 0–23. The provider's zero-byte entry requires no blob. |
| `params.total_bytes` / `align` | Aggregate launch parameter image: 64 bytes aligned to 64. The host supplies per-stage blobs; the launcher places them. |
| `protocol_expansions` | Resolved call channel and implementation declaration. T0 has no barrier protocol. |
| `generated/binding_stub.cu` | Resolves the warp handle and binds `pipelink_call_scale` to `hello_scale`. |
| `generated/dispatch.cu` | Defines the aligned params image and `pipelink_kernel`, dispatching warps `[0,4)` into `hello_call`. |

`generated_hashes` maps generated files to digests; the command prints only
the names. Those digests and `input_hashes` are provenance records.

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

Each clause ID names a check that pack or link can report, so the catalogue is
a reference for reading diagnostics; it says nothing about how a kernel behaves
on a GPU. [T3](T3-contract-gallery.md) breaks contracts on purpose and shows
each failure at the pack or link phase where it occurs.

What T0 has shown is that both stages pack independently, inspect cleanly and
link without a GPU on every profile. Numerical and sanitizer checks on a GPU
start in T1.

## What you declared vs what was derived

| Declared by this example | Derived by PipeLink |
|---|---|
| **7 annotations total**: `PL_STAGE` ×2 name and version the stages. | Separate manifest/object packages and stage-scoped contracts. |
| `PL_NO_PARAMS` ×1 promises no provider blob; `PL_PARAMS` ×1 publishes the caller type/fields. | Compiler-observed sizes/alignments; caller `(offset=0, bytes=24, align=8)` in a 64-byte aggregate image. |
| `PL_ROLE` ×1 promises four warps entering `hello_call`. | Warp claim, solved interval `[0,4)`, 128-thread block and generated dispatch. |
| `PL_CALL` ×2 promise matching produce/consume call endpoints; the consuming one publishes export `scale`. | Signature checks and the generated `pipelink_call_scale` → `hello_scale` binding. |
| Pipeline package paths, one phase, handle ownership and static topology; selected profile. | Legal resource binding, register cap, cluster/shared-memory requirements and target cubin. |

Next: [T1 — Hello, callable seam](T1-hello-callable.md).
