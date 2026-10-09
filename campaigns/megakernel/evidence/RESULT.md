# Megakernel llama-1b port — seam-parity evidence

Rendered by `parity/run_parity.py` from `results.json` (campaign generated 2026-10-06T14:03:04.908960+00:00).

## Verdict

**Seam-parity gate PASS** — all 5 linked / same-source one-TU cases meet the median ratio floor of 0.97.

**P2 numerical gate PASS:** all 20 receipts (five bundles × four pipelines) pass; P3 admission requires it.

| case | linked / reference | median ratio | pair-valid count | reference median ms/token | linked median ms/token | batch tokens | clock validity | 0.97 seam gate |
|---|---|---:|---:|---:|---:|---:|---|---|
| MK-A-16 | linked / onetu | 1.0000273605241015 | 8 | 0.793798891010552 | 0.7937921890016234 | 32 | 8/8 pair-valid; original-only | PASS |
| MK-A-1024 | linked / onetu | 0.9998976414152987 | 8 | 1.5234617337890626 | 1.5236424577303687 | 32 | 8/8 pair-valid; original-only | PASS |
| MK-A-4096 | linked / onetu | 0.9999089657417459 | 8 | 3.73403075204142 | 3.7339184392948717 | 13 | 8/8 pair-valid; original-only | PASS |
| MK-REGS-1024 | linked / onetu | 1.0000765702628434 | 8 | 1.5236690804296873 | 1.5235938891015628 | 31,32 | 8/8 pair-valid; original-only | PASS |
| MK-A-1024-red | linked / onetu | 1.000125572120941 | 8 | 1.1980254942708335 | 1.1978731165747547 | 32 | 8/8 pair-valid; original-only | PASS |

Each slot calibrated a one-token probe and selected an integer batch at a 50 ms target, capped at 32 tokens. The per-case batch sizes are in the table. Across every slot, observed CUDA-event batch time ranged from 25.3385601 ms to 48.8534737 ms, below the 100 ms harness quantum. gpulockd held the selected GPU at 1102 MHz; its pre-case, per-case, and post-case status checks were healthy with zero recorded violations.

## Sanitizers

Counts on the timed images, linked / one-TU. Admission requires memcheck zero; racecheck and synccheck counts are recorded, not admission predicates. Logs cap reported hazards, so racecheck counts are not exhaustive tallies.

| case | memcheck | racecheck | synccheck |
|---|---:|---:|---:|
| MK-A-16 | 0 / 0 | 39 / 40 | 0 / 0 |
| MK-A-1024 | 0 / 0 | 46 / 37 | 0 / 0 |
| MK-A-4096 | 0 / 0 | 37 / 47 | 0 / 0 |
| MK-REGS-1024 | 0 / 0 | 38 / 46 | 0 / 0 |
| MK-A-1024-red | 0 / 0 | 34 / 36 | 0 / 0 |

## Images

`port/build.sh` builds the host-ABI-v3 linked, same-source one-TU and monolithic images with fast math (FTZ and FMA enabled, precise division and square root disabled) from upstream Megakernels `7309cec` plus the minimal correctness patches in `port/patches/` (partial-store `laneid`, partial KV stage index, tensor-allocator barrier, reduction page lifetime); without them the multi-partition attention path is dead or racy upstream. The reference is therefore patched upstream, not stock upstream.

## Interpretation

Each PipeLink-linked cubin is compared with its same-source one-TU build, isolating the ABI/link seam. MK-A-1024-red additionally executes two attention partials per KV head followed by terminal attention reductions in all sixteen layers. No upstream performance comparisons are part of this campaign.

## Method

Each case was executed by `evaluator.harness.parity.run_case` with `Protocol.definitive`: five untimed single-token check-mode warmups per image, 3 s nominal slots, 1 s leading discard, first eight chronological primary-valid pairs, and the shared ±3% clock predicate. Before every warmup request and every timed slot request, the adapter ran a separate single-token Gate P2 recheck: all nine outputs finite and layer-0 `k_row`/`v_row` within `rtol=2e-2`, `atol=2e-2*RMS(reference)`. Throughput is complete steady-window tokens divided by summed CUDA-event batch time. Pair orientation is always linked/reference; no ratios were imputed.

Running the campaign also writes machine-readable results (`results.json`) and sanitizer receipts (`sanitize.json`) next to this report; they are not published. Harness-owned requests, slot outputs, telemetry, pair decisions, and summaries are written to the git-ignored `archive/submission-2026-09-17/<case-id>/`.
