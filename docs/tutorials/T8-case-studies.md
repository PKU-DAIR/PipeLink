# T8 — Case studies: reading, not doing

## Goal and prerequisites

Read four compositions that put the earlier mechanisms under workloads too
large for a tutorial. There is **no GPU prerequisite for this reading** and no
new annotation API to learn. [T0](T0-pack-link.md) explains packages and trust;
[T4](T4-tensor-import.md) separates declared contracts from generated data paths;
[T5](T5-host-abi.md) explains the host-owner boundary. Those are useful background,
not a requirement to reproduce the campaigns.

Each tour follows one composition, gives one GPU-free link command, counts the
annotations in its actual stage sources, and cites dated committed results.
Executing the optional command needs T0's CUDA toolkit setup and available
stage packages; megakernel also needs its generated pipeline input. Reproducing
published performance needs the separate campaign runbook and compatible
hardware. These are reading guides, not fresh-clone timing exercises.

## Choose a tour

| Case study | What it shows that a small tutorial cannot | Where to start |
|---|---|---|
| Attention, Hopper/sm90 | A 28-line window-mask callee reused with a frozen FA3-derived bf16 core across real-model workloads; same-source parity beside absolute placement against FA3/FA4/FlashInfer/Flex. | [W2 window tour](../../campaigns/attention/TOUR-sm90.md) |
| Context-parallel attention, Hopper/sm90 | The same frozen core run across two to four GPUs: an owner load stage that walks zigzag-sharded KV, and an owner comm stage that pushes it over NVLink inside the kernel; 1.95×/3.66× over one GPU and 1.19–1.30× faster than Transformer Engine's context parallelism. Uses [T7](T7-comm-overlap.md)'s seams. | [CP section](../../campaigns/attention/README.md#context-parallel-causal-attention-cp) |
| Megakernel, Blackwell/sm103 | A CUTLASS/CuTe attention package swapped into a ThunderKittens whole-model interpreter while the other eight packages and the external schedule stay unchanged; instruction-phase resources, opcode binding and host-owner integration hold the nine packages together. | [Cross-library whole-model tour](../../campaigns/megakernel/TOUR.md) |
| MoE training megakernel, Blackwell/sm103 | Four owners' packages (framework, communication, GEMM, activation) composing all 24 forward/backward/recompute kernels from one op artifact set on four GPUs; cross-owner geometry as linked facts, multi-producer counter channels, and an owner swap without the other owners' sources. | [Case study](../../campaigns/mok/README.md) |

The attention and megakernel tours use different architectures and timing
conventions; their numbers are not a head-to-head architecture benchmark.
Other research workloads are intentionally not indexed as case studies: their
seam mechanisms are already taught by the tutorial ladder.

## Read the claims in the right order

1. **What changed?** Follow the pipeline's package paths. Reuse means the
   compositions reference the same packaged stage, not that separate builds
   produce identical container bytes.
2. **What is the control?** Seam parity compares linked code with a same-source
   one-TU control. A public implementation is an absolute-placement reference,
   not that control. For Hopper attention, R1 is the pass/fail performance claim.
   Absolute gaps to FA3 come from the core's authoring; R1 shows the link adds
   no cost. The tour also keeps FA4's faster W6-prefix result visible.
3. **What was actually accepted?** Numeric, sanitizer and parity predicates are
   distinct. Megakernel's memcheck-gated, racecheck/synccheck-count-recording
   admission is weaker than sanitizer-clean acceptance.
   See the megakernel tour for its recorded acceptance protocol.

## What you declared vs what was derived

T8 authors no new stage, so it has **zero new annotations**. That does not make
the campaign stages annotation-free. The tours count their active declarations,
including local author macros, and distinguish those from generated machinery:

| Reading question | Author-declared facts | Derived facts to inspect |
|---|---|---|
| Can this callee replace another? | Export/call signatures, published types and topology. | Compatible callable binding and dispatch. |
| Can these roles share a kernel? | Warp/resource requirements, endpoints, arrival counts and lifetimes. | Legal physical assignments and launch geometry. |
| Can this host provider launch the model? | Owner types, opaque residue and scheduler authority. | Measured ABI layouts, role views and provider contract. |

The counts describe the current sources; performance tables state their
measurement conditions. Package ABI compatibility alone does not establish performance.

Machine-local paths in recorded campaign evidence are replaced at publication
with `$CUDA_HOME`, `$PIPELINK_SCRATCH_ROOT`, `$TMPDIR`, or repo-relative paths.
Measurements and verdicts are unchanged; see [Evidence records](../../CONTRIBUTING.md#evidence-records).

Previous: [T7 — Overlap communication with compute across GPUs](T7-comm-overlap.md).
Return to [T0](T0-pack-link.md) to build your own small composition.
