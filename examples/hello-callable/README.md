# Hello, callable seam

Two separately packed CUDA stages compose into `output[i] = scale(input[i])`.
The provider implements `scale(x) = 2*x`; the caller owns a four-warp
grid-stride loop. The same source pair, with eight annotations, builds for
Hopper/sm90 (`sm_90a`), Blackwell/sm103 (`sm_103a`) and consumer Blackwell/sm120
(`sm_120a`). It launches through the host-less Python and CLI launcher and uses
no TMEM, cluster launch control or external host provider.

## Commands

Run from the repository root after configuring CUDA 13.3, scratch and available
GPUs in `.env` as described in [T0](../../docs/tutorials/T0-pack-link.md).
Builds need no GPU. Generated files and `results.json` go to the ignored
`out/<profile>/`.

```sh
uv run --project pipec --locked python examples/hello-callable/build.py --profile sm_90a
uv run --project pipec --locked python examples/hello-callable/build.py --profile sm_103a
uv run --project pipec --locked python examples/hello-callable/build.py --profile sm_120a
```

To run, pick an available GPU matching the profile and set `PROFILE` to that
profile. `run.py --gpu` selects a physical GPU (default: the first
`PIPELINK_GPUS` entry). The recipe holds the machine-qualified clock for the
whole run:

```sh
set -e
source tools/env.sh
PROFILE="${PROFILE:-sm_103a}"
GPU="${PIPELINK_GPUS%%,*}"
uv run --project pipec --locked python tools/gpulockd.py start --gpus "$GPU" --sm-mhz "$PIPELINK_LOCK_SM_MHZ"
trap 'uv run --project pipec --locked python tools/gpulockd.py stop --gpus "$GPU"' EXIT
uv run --project pipec --locked python tools/gpulockd.py status --gpus "$GPU"
uv run --project pipec --locked python examples/hello-callable/run.py --profile "$PROFILE" --gpu "$GPU"
uv run --project pipec --locked python tools/gpulockd.py status --gpus "$GPU"
uv run --project pipec --locked python tools/gpulockd.py stop --gpus "$GPU"
trap - EXIT
```

## Evidence

Measured with CUDA **13.3**. These are observations from one run; a rebuild need
not reproduce them exactly.
Every profile passes numeric acceptance and 4/4 racecheck/synccheck runs on the
linked and one-TU images. Each run held its SM clock locked (1155 MHz on sm103,
1230 MHz on sm90, 1770 MHz on sm120) and released the lock cleanly.

| Profile | Numeric comparisons | Paired timing | Allocation protocol | SM clock |
|---|---|---|---|---|
| Blackwell/sm103 | All three bitwise PASS | PASS; 181 launches/batch; median one-TU/linked speedup **0.999903** | Shared buffers via `HostlessSession.with_image` | 1155 MHz locked; healthy released lease |
| Hopper/sm90 | All three bitwise PASS | PASS; 181 launches/batch; median one-TU/linked speedup **1.000146** | Shared buffers via `HostlessSession.with_image` | 1230 MHz locked; healthy released lease |
| Consumer Blackwell/sm120 | All three bitwise PASS | PASS; 64 launches/batch; median one-TU/linked speedup **0.999795** | Shared buffers via `HostlessSession.with_image` | 1770 MHz locked; healthy released lease |

Speedups are medians of per-block one-TU time / linked time; higher is faster.
Each run writes `out/<profile>/results.json`.

On every profile, the
[T1 §5 CLI recipe](../../docs/tutorials/T1-hello-callable.md#5-launch-directly-from-the-cli)
reports block **[128, 1, 1]**, grid **[1024, 1, 1]**, dynamic shared memory
**0 bytes**, and `CLI oracle: bitwise PASS`.

[T1 — Hello, callable seam](../../docs/tutorials/T1-hello-callable.md) walks
through the sources, the parameter ABI, the Python/CLI launch commands and what
was declared versus derived.
