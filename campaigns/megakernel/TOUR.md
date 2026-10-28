# Blackwell/sm103 tour — a whole-model megakernel from ThunderKittens and CUTLASS packages

## One composition

Pipeline `cute`, derived from the [`a` template](port/README.md#host-abi-v3-packages)
(`port/pipelines/a.json`), links **nine packages**:
an interpreter plus NoOp, QKV/RoPE/cache append, attention partial, attention
reduction, output projection, up-gate/SiLU, down projection and LM-head ops. An
instruction stream runs the model's sixteen layers and LM head inside one persistent
kernel. The interpreter's five roles call six public entrypoints per op.
Pages, semaphores and scratch are exclusive per instruction phase, so different ops
reuse the same resource pool.

The partial-attention package is written with **CUTLASS/CuTe** layouts, asynchronous
copies and warp MMA. Its four compute warps process disjoint KV tiles and merge their
normalized partial results inside the stage. The interpreter and the other seven ops
come from Megakernels and use **ThunderKittens**. Swapping in this one opaque package
leaves the instruction stream, the other eight packages and the compiled host provider
intact. Pipeline `a` is the all-ThunderKittens reference. The code guide's
[cross-library attention](port/README.md#cross-library-attention) section gives the
build and numerical-verification commands.

With the instruction schedule held fixed, the
[mixed-library measurements](port/README.md#measured-composition) give **2.158×**
the all-ThunderKittens throughput at context 4096 (**1.630 ms/token**) and
**1.225×** at context 1024 with reduction. At context 16 the ratio is **0.947×**:
CuTe is slower there. These numbers describe this composition, not the libraries
in general. Each comparison uses eight `paired-reload-v2` blocks at the qualified
**1155 MHz** clock. All three CuTe linked/same-source-one-TU ratios are within
**0.07%** of one, above the 0.97 floor. The numerical receipts and attention
oracle cover these images; reload snapshots and
[paired-path sanitizers](evidence/sanitizers.md#paired-path-sanitizers) pass.

Start with the [port code guide](port/README.md),
[`interpreter.cu`](port/src/interpreter.cu), the
[op sources](port/src/ops/), and their shared
[`mk_abi.cuh`](port/src/mk_abi.cuh) annotation wrapper. The CuTe math lives in
[`cute_attention.cuh`](port/src/ops/cute_attention.cuh); its VM adapter is
[`attn_partial_cute.cu`](port/src/ops/attn_partial_cute.cu). The ThunderKittens ops
carry the five documented correctness patches to upstream Megakernels.

## One command

After `source tools/env.sh` (CUDA 13.3 at `PIPELINK_CUDA_HOME`, `uv`, and the
`third_party/Megakernels` and `third_party/cutlass` submodules) and the code guide's
host-only package build, run from the repository root:

```sh
CASE=campaigns/megakernel
uv run --project pipec --locked pipec2 link \
  "$CASE/port/build/pipelines/cute.json" \
  --out-dir "$PIPELINK_SCRATCH_ROOT/t7-megakernel-cute"
```

Use the **generated** `port/build/pipelines/cute.json`, which the build derives from
the `a` template by substituting the CuTe attention package. Regenerate it after
moving the checkout. Full model inputs and hardware reproduction are in the code guide.

## ThunderKittens reference: parity and acceptance

The all-ThunderKittens [seam-parity report](evidence/RESULT.md) records
**MK-A-1024-red: 0.999591 linked/same-source-one-TU speedup**, **1.149 ms/token**
linked, eight fixed-count blocks at **1155 MHz** on local Blackwell/sm103. This
cell executes two attention partials per KV head followed by reductions in all
sixteen layers. All five reference seam gates pass (floor 0.97): the
linked/one-TU speedup (R1) spans **0.999549–1.000178**, with healthy, stopped
clock locks and uncontended GPU windows. Timing also requires the 20 numerical
receipts; the [code guide](port/README.md#gate-results) lists their predicates.

Every timed image also has zero-count memcheck, racecheck and synccheck receipts, and
all **21 paired sanitizer legs** pass on the runner/provider path; the
[sanitizer report](evidence/sanitizers.md) lists each check. Seven reference legs
record a return code derived from the controller's PASS line because their raw
return codes were not retained. Numerical, sanitizer and seam-parity gates are
independent checks.

## What you declared vs what was derived

Counts expand local author wrappers for the active nine stages, including
`MK_DEFINE_OP_EXPORTS`, the eight interpreter slot declarations and five role
views. They exclude generated neutral host-ABI inventory and annotation-library
internals. Interpreter **80**; ops **112** (NoOp 11, QKV 15, attention partial
13, reduction 13, output projection 15, up-gate 15, down projection 15, LM head
15): **192 annotation invocations** total.

| Declaration | Count | Author supplies → compiler/linker derives |
|---|---:|---|
| `PL_STAGE`; `PL_TOPOLOGY` | 9 each | Package identity and cluster contract → compatible package/launch contracts. |
| `PL_NO_PARAMS` | 8 | Ops carry no ordinary stage blob; the interpreter's host group implies its empty params. |
| `PL_DESCRIPTOR` | 8 | Instruction type → compiler-measured descriptor size/alignment. |
| `PL_ROLE` | 5 | Interpreter entrypoints and warp counts → dispatch for 640 threads. |
| `PL_CLAIM*` | 45 | 13 interpreter (including the uniform launch-register cap) + 32 op resource claims, opcode domains and instruction lifetimes → legal allocations and opcode bindings. |
| `PL_DOMAIN_CAPACITY` | 3 | Instruction page, semaphore and scratch pool sizes the interpreter compiles → linker domain capacities. |
| `PL_CALL` | 96 | Six per op, with 48 consuming endpoints that also publish the exports and 48 producing endpoints → callable dispatch bindings. |
| `PL_HOST_GROUP`, `PL_HOST_ABI_TYPE`, `PL_HOST_RESIDUE`, `PL_HOST_SCHEDULER` | 1 + 6 + 1 + 1 | Host owner, measured types, opaque residue and launch authority → provider contract, generated role views and the implied target section. |

## Reference results

Linked and one-TU numerics are compared with a tolerance (each tensor within
max(2 × repeat floor, 2 BF16 ULP)), and bitwise equality is recorded separately. In
the authors' receipts the layer-1 fixtures are bitwise identical in every output
except `hidden_states`; at full depth only the partition-0 attention outputs are, and
on the reduction bundle none are. No configuration is bitwise identical in every
output, because two runs of the same linked image already differ there; the
[code guide](port/README.md#gate-results) has the details.

The authors' reference results are in [RESULT.md](evidence/RESULT.md),
[CUTE-RESULT.md](evidence/CUTE-RESULT.md), [sanitizers.md](evidence/sanitizers.md),
[seam-sass.md](evidence/seam-sass.md) and the code guide's
[placement table](port/README.md#placement-against-the-original-megakernel).

Continue with the [full evidence](evidence/RESULT.md) or the
[sanitizer results](evidence/sanitizers.md).
