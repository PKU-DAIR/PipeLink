# Megakernel llama-1b port — paired-reload seam parity

Local Blackwell/sm103 run: **2026-10-17T03:34:27.939859+00:00**,
qualified clock **1155 MHz**.

## Verdict

**All five reference seam gates PASS**: the linked/same-source-one-TU speedup (R1)
spans **0.999549104–1.000177969** (floor 0.97), and all five executions are admitted.
Admission requires the 20 numerical receipts, which PASS, plus paired-path
snapshot and [sanitizer](sanitizers.md) checks, which also PASS.

| case | physical composition A / B | R1 speedup B/A | reference ms/token | linked ms/token | blocks | seam gate |
|---|---|---:|---:|---:|---:|---|
| MK-A-16 | a:linked / a:onetu | 0.999549104 | 0.713650 | 0.713563 | 8 | PASS |
| MK-A-1024 | a:linked / a:onetu | 0.999699572 | 1.411531 | 1.411899 | 8 | PASS |
| MK-A-4096 | a:linked / a:onetu | 1.000135447 | 3.517378 | 3.514914 | 8 | PASS |
| MK-REGS-1024 | rebind-regs:linked / rebind-regs:onetu | 1.000177969 | 1.423287 | 1.423045 | 8 | PASS |
| MK-A-1024-red | a:linked / a:onetu | 0.999591308 | 1.148988 | 1.149081 | 8 | PASS |

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
