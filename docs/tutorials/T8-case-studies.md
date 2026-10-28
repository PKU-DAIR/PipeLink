# T8 — Case studies: reading, not doing

## Goal and prerequisites

These four compositions apply the earlier mechanisms to workloads too large for a
tutorial. Reading them needs **no GPU** and introduces no new annotation API.
[T0](T0-pack-link.md) (packages and trust), [T4](T4-tensor-import.md) (declared
contracts versus generated data paths) and [T5](T5-host-abi.md) (the host-owner
boundary) are useful background for following the tours, though you do not need
them to reproduce anything.

Each tour follows one composition, gives one GPU-free link command, counts the
annotations in its stage sources and cites committed results. Running the
link command is optional; it needs T0's CUDA toolkit setup and the stage
packages, and megakernel also needs its generated pipeline input. Reproducing
the published performance needs the campaign's own runbook and compatible
hardware. The tours are reading guides, so expect to study results rather than
time anything from a new clone.

## Choose a tour

| Case study | What it shows | Where to start |
|---|---|---|
| Attention, Hopper/sm90 | A 19-line window-mask callee reused with a frozen FA3-derived bf16 core across real-model workloads; same-source parity beside absolute placement against FA3/FA4/FlashInfer/Flex. | [W2 window tour](../../campaigns/attention/TOUR-sm90.md) |
| Context-parallel attention, Hopper/sm90 | The same frozen core run across two to four GPUs: an owner load stage that walks zigzag-sharded KV, and an owner comm stage that pushes it over NVLink inside the kernel; 1.95×/3.67× over one GPU and 1.20–1.31× faster than Transformer Engine's context parallelism. Uses [T7](T7-comm-overlap.md)'s seams. | [CP section](../../campaigns/attention/README.md#context-parallel-causal-attention-cp) |
| Megakernel, Blackwell/sm103 | A CUTLASS/CuTe attention package swapped into a ThunderKittens whole-model interpreter while the other eight packages and the external schedule stay the same; instruction-phase resources, opcode binding and host-owner integration hold the nine packages together. | [Cross-library whole-model tour](../../campaigns/megakernel/TOUR.md) |
| MoE training megakernel, Blackwell/sm103 | Four owners' packages (framework, communication, GEMM, activation) composing all 24 forward/backward/recompute kernels from one op artifact set on four GPUs; cross-owner geometry as linked facts, multi-producer counter channels, and an owner swap without the other owners' sources. | [Case study](../../campaigns/mok/README.md) |

The attention and megakernel tours use different architectures and timing
conventions, so their numbers are not a head-to-head comparison between
architectures. Other research workloads are left out of this list on purpose:
the tutorials already teach the seams they use.

## Read the claims in the right order

1. **What changed?** Follow the pipeline's package paths. Reuse means the
   compositions reference the same packaged stage; separate builds need not
   produce identical container bytes.
2. **What is the control?** Seam parity compares linked code with a same-source
   one-TU control. Public implementations (FA3, FA4 and others) show where the
   result places in absolute terms; they are a reference, not that control. For
   Hopper attention, the pass/fail performance check (R1) is the
   linked/one-TU throughput ratio: it shows the link adds no cost, while gaps to
   FA3 come from how the core is written. The tour also reports FA4's faster
   W6-prefix result.
3. **What was accepted?** Numeric, sanitizer and parity checks are separate
   gates; each case study states which ones its timed images passed. The
   megakernel tour describes its acceptance protocol.

## What you declared vs what was derived

T8 authors no stage, so it adds **zero annotations**. The campaign stages do have
annotations: each tour counts their active declarations, including local author
macros, separately from generated machinery:

| Reading question | Author-declared facts | Derived facts to inspect |
|---|---|---|
| Can this callee replace another? | Export/call signatures, published types and topology. | Compatible callable binding and dispatch. |
| Can these roles share a kernel? | Warp/resource requirements, endpoints, arrival counts and lifetimes. | Legal physical assignments and launch geometry. |
| Can this host provider launch the model? | Owner types, opaque residue and scheduler authority. | Measured ABI layouts, role views and provider contract. |

The counts describe the sources as checked in, and each performance table states
its measurement conditions. ABI compatibility shows that packages can link;
on its own it does not establish how fast the result runs.

In recorded campaign evidence, machine-local paths are replaced at publication
with `$CUDA_HOME`, `$PIPELINK_SCRATCH_ROOT`, `$TMPDIR` or repo-relative paths;
the measurements and verdicts are as recorded. See
[Evidence records](../../CONTRIBUTING.md#evidence-records).

Previous: [T7 — Overlap communication with compute across GPUs](T7-comm-overlap.md).
Return to [T0](T0-pack-link.md) to build your own small composition.
