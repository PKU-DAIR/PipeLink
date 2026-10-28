# Megakernel CuTe-attention composition — paired-reload evidence

Local Blackwell/sm103 run: **2026-10-17T03:34:49.149462+00:00**,
qualified clock **1155 MHz**.

Pipeline `cute` replaces only the attention-partial package in pipeline `a`; the
other eight packages, the provider and the instruction schedule are the same. All six
executions are admitted. The **25 numerical receipts** (20 reference + five
CuTe), the per-image [sanitizer](sanitizers.md) receipts and the paired-path checks PASS.

## Same-source seam parity: CuTe linked / CuTe one-TU

**All three seam gates PASS**, with R1 **0.999352102–0.999823674** (floor 0.97).

| case | R1 speedup B/A | CuTe one-TU ms/token | CuTe linked ms/token | blocks | seam gate |
|---|---:|---:|---:|---:|---|
| MK-CUTE-16 | 0.999759889 | 0.752648 | 0.753187 | 8 | PASS |
| MK-CUTE-1024-red | 0.999352102 | 0.937082 | 0.938292 | 8 | PASS |
| MK-CUTE-4096 | 0.999823674 | 1.629094 | 1.629874 | 8 | PASS |

## Cross-library comparison: CuTe linked / ThunderKittens linked

These admitted ratios are reported without a gate: **0.947290×** at context 16,
**1.224969×** at 1024 with reduction and **2.157828×** at 4096. At short context
the CuTe composition is slower than the all-ThunderKittens one in this fixture. The
ratios describe this composition, not the libraries in general.

| case | paired speedup B/A | TK linked ms/token | CuTe linked ms/token | blocks | seam gate |
|---|---:|---:|---:|---:|---|
| MK-CUTE-VS-TK-16 | 0.947289798 | 0.712642 | 0.752481 | 8 | NOT_GATED |
| MK-CUTE-VS-TK-1024-red | 1.224968523 | 1.150695 | 0.939302 | 8 | NOT_GATED |
| MK-CUTE-VS-TK-4096 | 2.157828412 | 3.516480 | 1.630094 | 8 | NOT_GATED |

## Machine code

Linked `cute` and its same-source one-TU control compile to identical normalized SASS
(13,712 instructions each; register names, addresses and scheduling fields
normalized, opcodes and immediates exact), as do the four all-ThunderKittens
pipelines. The [SASS report](seam-sass.md) lists each pipeline.

## Method

Each case runs once in one local CUDA context under `paired-reload-v2` at the
qualified **1155 MHz** clock. An execution is admitted when its exclusive GPU window
is uncontended and its clock lock is healthy and stopped; all of them are.

The schedule is `ABBA, BAAB, BAAB, ABBA | BAAB, ABBA, ABBA, BAAB`, where A is the
candidate and B the explicitly identified reference. Each block has four batches of
exactly ten complete fixed-position decodes, so each case has **8 blocks, 32 raw
event totals and 320 scored calls**. The common helper adds 32 discarded warmup
calls; eight single-call numerical snapshots are checked outside scored timing.
There is no calibration, duration floor, three-second slot, per-arm
telemetry-coverage gate or retry on a ratio MISS.

For block i, `s_i = sum(two B batch times) / sum(two A batch times)`. The reported
speedup is `sqrt(median(s_0..s_3) * median(s_4..s_7))`. Only same-source seam
comparisons use the **0.97** floor. The ms/token columns are the median across the
eight blocks of that arm's two summed batch times divided by 20; they are for
reference and do not enter the paired speedup.

Both modules load candidate-first, unload at the midpoint, and reload
reference-first with refreshed function attributes. The context, stream,
RuntimeBuffers/weights/KV/input/output state, immutable pinned embedding rows and
each arm's opaque params persist across the reload. Each arm runs one provider
query/build before all batches; both provider sessions survive the reload and are
released after the final drain. Each scored call's CUDA-event interval covers Bar
reset, embedding HtoD and the complete megakernel, with fixed pos_id and embedding
row `token % scalar_ntok` restarting at zero each batch.

Initial, before-reload, after-reload and final snapshots of both arms pass the
finite-nine and strict layer-zero KV predicates. These single-token snapshots are
spot checks; the full advancing-token numerical predicates (repeat floor, one-TU,
upstream) come from the image-generation receipts. The receipt behind this report
retains raw blocks, epoch windows, admission, clock operations, manifest/image
identities, numerical metrics and provider/reload provenance. Weights, bundles, binaries and tensor-dump
bytes stay in local scratch/data and are not published.

## Sanitizers

Every timed linked and one-TU image has zero memcheck, racecheck and synccheck
findings. The paired path, bound to the host runner and provider, passes **9
reference + 12 CuTe/cross-library legs**, all with zero findings. The bounded reload
probe executes 16 calls, including queued copies on both sides of the reload; the
320-call scoring protocol does not run under the sanitizer. Seven reference legs
record a return code derived from the controller's PASS line because their raw
return codes were not retained. The [sanitizer report](sanitizers.md) lists every
per-image and paired-path check.

## Publication scope

The JSON receipts behind this report stay with the authors; repository paths in
them are relative, and external locations use `$PIPELINK_SCRATCH_ROOT`,
`$MK_BUNDLE_ROOT` and `$PIPELINK_CUDA_HOME`. The upstream placement benchmark has
its own measurement protocol. These are composition and seam measurements; cross-library
comparisons describe this composition and do not rank the libraries.
