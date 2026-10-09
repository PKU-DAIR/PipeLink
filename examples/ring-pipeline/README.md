# Ring pipeline

A hand-CUDA TMA producer fills a bf16 SW128 shared-memory ring while four consumer warps compute `out = max(x * 2 + 1, 0)` in f32. Each CTA walks 16 tiles, exercising repeated full/empty mbarrier phases. One producer package and one consumer package are each reused unchanged by two links selecting depths 2 and 4; the same sources build for Hopper/sm90, Blackwell/sm103 and consumer Blackwell/sm120 without CUTLASS or TileLang.

## Commands

From the repository root, after configuring `.env` as in [T2](../../docs/tutorials/T2-ring-pipeline.md):

```sh
uv run --project pipec python examples/ring-pipeline/build.py --profile sm_90a
uv run --project pipec python examples/ring-pipeline/build.py --profile sm_103a
uv run --project pipec python examples/ring-pipeline/build.py --profile sm_120a
# Run each recipe only on a matching GPU; --gpu is the physical device index.
uv run --project pipec python examples/ring-pipeline/run.py --profile sm_90a
uv run --project pipec python examples/ring-pipeline/run.py --profile sm_103a
uv run --project pipec python examples/ring-pipeline/run.py --profile sm_120a
```

Omit `--gpu` to use the first `PIPELINK_GPUS` entry. Builds are GPU-free. Each full run writes `out/<profile>/results.json` and sanitizes both images at both depths unconditionally. No clock-lock permission is required; the runner observes clocks and never changes them.

## Evidence

Measured with CUDA **13.3** on every profile. Scored timings share buffers,
parameters, descriptors, and streams via `HostlessSession.with_image`; clocks are
observed, not locked. Each run writes `out/<profile>/results.json`.

| Evidence | Blackwell/sm103 depths 2 / 4 | Hopper/sm90 depths 2 / 4 | Consumer Blackwell/sm120 depths 2 / 4 |
|---|---|---|---|
| Build, same source | PASS | PASS | PASS |
| Workload | 32768 × 1024 bf16 → f32; 256 CTAs × 16 tiles | Same | Same |
| Solved threads / dynamic shared bytes | 160 / 32832; 160 / 65664 | Same | Same |
| Oracle and linked/one-TU | All three comparisons bitwise; 0 mismatches, max error 0 | Same | Same |
| Median launch time, linked / one-TU | 41.823 / 41.841 µs; 38.278 / 38.245 µs | 62.376 / 62.382 µs; 63.640 / 63.652 µs | 155.832 / 155.868 µs; 149.468 / 149.645 µs |
| Median paired ratio, linked / one-TU | **0.999742 / 1.000656** | **0.999935 / 0.999866** | **0.999807 / 0.999053** |
| Shared `time_pair` batch gate | PASS; 374 / 410 launches/batch | PASS; 251 / 246 launches/batch | PASS; 101 / 106 launches/batch |
| Phase-B ratio `[0.97, 1.03]` and full runner | PASS | PASS | PASS |
| racecheck + synccheck, linked and one-TU | 4/4 PASS at each depth | 4/4 PASS at each depth | 4/4 PASS at each depth |
| Normalized SASS, diagnostic only | All comparisons equal; 280/280 and 296/296 instructions | All comparisons equal; 312/312 and 328/328 instructions | All comparisons equal; 304/304 and 312/312 instructions |
| Local traffic / calls, diagnostic | None | None | One local store and one call in both images |
| Observed SM clock | 2032 MHz | 1980 MHz | 2992 MHz after the run |

Walkthrough, exact annotation count, declaration/derivation table, and neighboring lessons: [T2 — A pipeline through an mbarrier ring](../../docs/tutorials/T2-ring-pipeline.md).
