# Hello, callable seam

Two separately packed CUDA stages compose into `output[i] = scale(input[i])`:
the provider implements `scale(x) = 2*x`, and the caller owns a four-warp
grid-stride loop. The same eight-annotation source pair builds for
Hopper/sm90 (`sm_90a`), Blackwell/sm103 (`sm_103a`) and consumer Blackwell/sm120
(`sm_120a`), using the host-less Python
and CLI launcher without TMEM, cluster launch control or an external host provider.

## Commands

From the repository root, configure CUDA 13.3, scratch and available GPUs in
`.env` as described in [T0](../../docs/tutorials/T0-pack-link.md). Builds are
GPU-free; `run.py --gpu` selects a physical GPU (default: first `PIPELINK_GPUS` entry).
Choose an available GPU matching the profile and run the matching `--profile`. Generated files and fresh `results.json` live in ignored `out/<profile>/`.

```sh
uv run --project pipec --locked python examples/hello-callable/build.py --profile sm_90a
uv run --project pipec --locked python examples/hello-callable/build.py --profile sm_103a
uv run --project pipec --locked python examples/hello-callable/build.py --profile sm_120a
uv run --project pipec --locked python examples/hello-callable/run.py --profile sm_103a
```

## Evidence

Measured with CUDA **13.3** on every profile. These are observations, not constants
a rebuild must reproduce. Each profile passes numeric acceptance and 4/4
racecheck/synccheck runs on linked and one-TU images; clocks are observed, not locked.

| Profile | Numeric comparisons | Paired timing | Allocation protocol | Observed SM before/after |
|---|---|---|---|---|
| Blackwell/sm103 | All three bitwise PASS | PASS; 216 launches/batch; median linked/one-TU **1.000259** | Shared buffers via `HostlessSession.with_image` | 2032/2032 MHz |
| Hopper/sm90 | All three bitwise PASS | PASS; 225 launches/batch; median linked/one-TU **1.000058** | Shared buffers via `HostlessSession.with_image` | 1980/1980 MHz |
| Consumer Blackwell/sm120 | All three bitwise PASS | PASS; 78 launches/batch; median linked/one-TU **1.000552** | Shared buffers via `HostlessSession.with_image` | 2542/2992 MHz |

Each run writes `out/<profile>/results.json`. The
[T1 §5 CLI recipe](../../docs/tutorials/T1-hello-callable.md#5-launch-directly-from-the-cli)
reports block **[128, 1, 1]**, grid **[1024, 1, 1]**, dynamic shared memory
**0 bytes**, and `CLI oracle: bitwise PASS` on every profile.

The source walkthrough, parameter ABI, exact Python/CLI launch commands and
declaration/derivation table are in
[T1 — Hello, callable seam](../../docs/tutorials/T1-hello-callable.md).
