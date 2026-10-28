# Ring pipeline

A hand-written CUDA TMA producer fills a bf16 SW128 shared-memory ring while four consumer warps compute `out = max(x * 2 + 1, 0)` in f32. Each CTA walks 16 tiles, so the full/empty mbarriers cycle through their phases repeatedly. The ring depth is chosen at link time: one producer package and one consumer package serve two links, with depths 2 and 4, without being repacked. The same sources build for Hopper/sm90, Blackwell/sm103 and consumer Blackwell/sm120 without CUTLASS or TileLang.

## Commands

From the repository root, after configuring `.env` as in [T2](../../docs/tutorials/T2-ring-pipeline.md):

```sh
uv run --project pipec python examples/ring-pipeline/build.py --profile sm_90a
uv run --project pipec python examples/ring-pipeline/build.py --profile sm_103a
uv run --project pipec python examples/ring-pipeline/build.py --profile sm_120a
# Set PROFILE to match the measurement machine.
# Hold its machine-qualified clock throughout every run:
set -e
source tools/env.sh
PROFILE="${PROFILE:-sm_103a}"
GPU="${PIPELINK_GPUS%%,*}"
uv run --project pipec python tools/gpulockd.py start --gpus "$GPU" --sm-mhz "$PIPELINK_LOCK_SM_MHZ"
trap 'uv run --project pipec python tools/gpulockd.py stop --gpus "$GPU"' EXIT
uv run --project pipec python tools/gpulockd.py status --gpus "$GPU"
uv run --project pipec python examples/ring-pipeline/run.py --profile "$PROFILE" --gpu "$GPU"
uv run --project pipec python tools/gpulockd.py status --gpus "$GPU"
uv run --project pipec python tools/gpulockd.py stop --gpus "$GPU"
trap - EXIT
```

Builds need no GPU. Omit `--gpu` to use the first `PIPELINK_GPUS` entry. Each full run sanitizes both images at both depths and writes `out/<profile>/results.json`. The runner only observes clocks; the recipe locks the clock with `gpulockd` and checks the lock is healthy before and after measurement.

## Evidence

Measured with CUDA **13.3** on every profile. Timed runs share buffers,
parameters, descriptors and streams via `HostlessSession.with_image`. Each run
held its SM clock locked (1155 MHz on sm103, 1230 MHz on sm90, 1770 MHz on
sm120) and released the lock cleanly.

| Evidence | Blackwell/sm103 depths 2 / 4 | Hopper/sm90 depths 2 / 4 | Consumer Blackwell/sm120 depths 2 / 4 |
|---|---|---|---|
| Build, same source | PASS | PASS | PASS |
| Workload | 32768 × 1024 bf16 → f32; 256 CTAs × 16 tiles | Same | Same |
| Solved threads / dynamic shared bytes | 160 / 32832; 160 / 65664 | Same | Same |
| Oracle and linked/one-TU | All three comparisons bitwise; 0 mismatches, max error 0 | Same | Same |
| Median launch time, linked / one-TU | 62.846 / 62.886 µs; 62.343 / 62.416 µs | 76.171 / 76.134 µs; 80.864 / 80.844 µs | 170.957 / 170.916 µs; 171.975 / 172.006 µs |
| Median paired speedup, one-TU / linked (higher is better) | **1.000470 / 1.000590** | **0.999358 / 0.999717** | **0.999728 / 1.000256** |
| Shared `time_pair` batch gate | PASS; 249 / 251 launches/batch | PASS; 207 / 194 launches/batch | PASS; 92 / 91 launches/batch |
| Parity band `[1/1.03, 1/0.97]` on the speedup, and full runner | PASS | PASS | PASS |
| racecheck + synccheck, linked and one-TU | 4/4 PASS at each depth | 4/4 PASS at each depth | 4/4 PASS at each depth |
| Normalized SASS, diagnostic only | All comparisons equal; 280/280 and 296/296 instructions | All comparisons equal; 312/312 and 328/328 instructions | All comparisons equal; 304/304 and 312/312 instructions |
| Local traffic / calls, diagnostic | None | None | One local store and one call in both images |
| SM clock | 1155 MHz locked; healthy released lease | 1230 MHz locked; healthy released lease | 1770 MHz locked; healthy released lease |

[T2 — A pipeline through an mbarrier ring](../../docs/tutorials/T2-ring-pipeline.md) walks through the sources, counts the annotations and lists what was declared versus derived.
