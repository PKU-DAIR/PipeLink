# Blackwell/sm103 tour — a whole-model megakernel from ThunderKittens and CUTLASS packages

## One composition

Pipeline `cute`, derived from [`a`](port/pipelines/a.json), links **nine packages**:
an interpreter plus NoOp, QKV/RoPE/cache append, attention partial, attention
reduction, output projection, up-gate/SiLU, down projection and LM-head ops. An
instruction stream runs the model's sixteen layers and LM head inside one persistent
kernel. The interpreter's five roles call six public entrypoints per op.
Instruction-phase-exclusive pages, semaphores and scratch let different ops reuse
the resource pool.

The partial-attention package is written with **CUTLASS/CuTe** layouts, asynchronous
copies and warp MMA. Its four compute warps process disjoint KV tiles and merge their
normalized partial results inside the stage. The interpreter and the other seven ops
come from Megakernels and use **ThunderKittens**. Swapping in this one opaque package
leaves the instruction stream, the other eight packages and the compiled host provider
intact. Pipeline `a` is the all-ThunderKittens reference. The code guide's
[cross-library attention](port/README.md#cross-library-attention) section gives the
build and numerical-verification commands.

With the instruction schedule held fixed, the
[mixed-library measurements](port/README.md#measured-composition) give **2.19×** the
all-ThunderKittens throughput at context 4096 (**1.701 ms/token**) and **1.22×** at
context 1024 with reduction, under the reference campaign's definitive protocol at a
locked clock; the linked kernel matches its same-source build within 0.02%. The CuTe
attention passes its double-precision oracle and isolated sanitizer runs, and the
composition passes all five P2 numerical receipts. The full composition still has
racecheck findings.

Start with the [port code guide](port/README.md),
[`interpreter.cu`](port/src/interpreter.cu), the
[op sources](port/src/ops/), and their shared
[`mk_abi.cuh`](port/src/mk_abi.cuh) annotation wrapper. The CuTe math lives in
[`cute_attention.cuh`](port/src/ops/cute_attention.cuh); its VM adapter is
[`attn_partial_cute.cu`](port/src/ops/attn_partial_cute.cu). The ThunderKittens ops
carry the four documented correctness patches to upstream Megakernels.

## One command

After [T0's toolkit/environment setup](../../docs/tutorials/T0-pack-link.md)
and the code guide's host-only package build, run from the repository root:

```sh
CASE=campaigns/megakernel
uv run --project pipec --locked pipec2 link \
  "$CASE/port/build/pipelines/cute.json" \
  --out-dir "$PIPELINK_SCRATCH_ROOT/t7-megakernel-cute"
```

Use the **generated** `port/build/pipelines/cute.json`, which the build derives from
the `a` template by substituting the CuTe attention package. Regenerate it after
moving the checkout. Full model inputs and hardware reproduction are in the code guide.

## ThunderKittens reference: parity is narrower than acceptance

The all-ThunderKittens [seam-parity report](evidence/RESULT.md) records
**MK-A-1024-red: 1.0001 linked/same-source-one-TU throughput**, **1.198 ms/token**
linked, eight valid pairs at **1102 MHz** on Blackwell/sm103. This cell executes
two attention partials per KV head followed by reductions in all sixteen layers.
The same report records seam parity PASS for all five cells (floor 0.97),
using current host-ABI-v3 images with fast math in both linked and control builds.
The separate P2 numerical gate **passes all 20 receipts**, and P3 timing requires
that; see the [code guide](port/README.md#current-gate-results) for its predicates.

**Do not read PASS as sanitizer-clean acceptance.** The actual sanitizer
admission in [`run_parity.py`](evidence/parity/run_parity.py) requires memcheck
success with zero errors. Racecheck **and synccheck** must have recorded integer
counts, but their zero-count/strict verdicts are not admission gates. Receipts
must cover the linked and one-TU images in the current build generation; numeric
and parity gates remain separate. The final report records memcheck 0,
synccheck 0 and **racecheck 34–47**. Logs cap reported hazards, so that range is
not an exhaustive tally.

## What you declared vs what was derived

Counts expand local author wrappers for the active nine stages, including
`MK_DEFINE_OP_EXPORTS`, the eight interpreter slot declarations and five role
views. They exclude generated neutral host-ABI inventory and annotation-library
internals. Interpreter **96**; ops **195** (NoOp 21, QKV 26, attention partial
23, reduction 23, output projection 25, up-gate 26, down projection 25, LM head
26): **291 annotation invocations** total.

| Declaration | Count | Author supplies → compiler/linker derives |
|---|---:|---|
| `PL_STAGE`; `PL_NO_PARAMS`; `PL_TOPOLOGY` | 9 each | Package identity, no ordinary stage blob, cluster contract → compatible package/launch contracts. |
| `PL_DESCRIPTOR` | 8 | Instruction type → compiler-measured descriptor size/alignment. |
| `PL_ROLE` | 5 | Interpreter entrypoints and warp counts → dispatch for 640 threads. |
| `PL_CLAIM*` | 44 | 12 interpreter + 32 op resource claims, opcode domains and instruction lifetimes → legal allocations and opcode bindings. |
| `PL_EXPORT`; `PL_CALL` | 48; 96 | Six exports per op, with 48 consuming and 48 producing endpoints → callable dispatch bindings. |
| `PL_HINT` | 52 | Configuration/advisory facts → reporting metadata, not proofs of safety. |
| `PL_HOST_GROUP`, `PL_HOST_ABI_TYPE`, `PL_HOST_ABI_EMPTY`, `PL_HOST_RESIDUE`, `PL_HOST_SCHEDULER`, `PL_HOST_TARGET` | 1 + 6 + 1 + 1 + 1 + 1 | Host owner, measured types, opaque residue and launch authority → provider contract and generated role views. |

Continue with the [full evidence](evidence/RESULT.md),
[sanitizer results](evidence/RESULT.md#sanitizers), or
[T8's case-study index](../../docs/tutorials/T8-case-studies.md).
