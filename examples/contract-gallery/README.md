# Contract gallery

This example needs no GPU. It links the T2 mbarrier-ring producer package with two consumers, the original affine-ReLU and a square, then feeds `pipec2` six deliberately wrong consumer contracts. Each must fail with the expected clause at the expected phase: `P-INIT` fails at pack and produces no package; the other five pack successfully and fail at link. The checker reuses the T2 pipeline builder and `.env` defaults, writes to the ignored `out/<profile>/`, and launches no kernels.

## Commands

From the repository root, with the CUDA 13.3 toolkit, nvJitLink, `g++`, `nm`, and `uv` installed:

```sh
uv run --project pipec --locked python examples/contract-gallery/check.py --profile sm_90a
uv run --project pipec --locked python examples/contract-gallery/check.py --profile sm_103a
uv run --project pipec --locked python examples/contract-gallery/check.py --profile sm_120a
```

## Evidence

| Evidence | Hopper / `sm_90a` | Blackwell / `sm_103a` | Consumer Blackwell / `sm_120a` |
|---|---:|---:|---:|
| Original consumer and square replacement linked | 2 / 2 | 2 / 2 | 2 / 2 |
| Wrong consumers rejected with exact clause and CLI phase | 6 / 6 | 6 / 6 | 6 / 6 |
| Complete checker cases / exit code | 8 / 8; 0 | 8 / 8; 0 | 8 / 8; 0 |
| `P-INIT` package produced | No | No | No |
| Producer packs per checker invocation | 1 | 1 | 1 |
| Square link: threads / dynamic shared memory | 160 / 32,832 bytes | 160 / 32,832 bytes | 160 / 32,832 bytes |
| Pack/link wall time (not GPU timing) | 132.54 s | 134.25 s | 134.04 s |
| GPU launches | 0 | 0 | 0 |
| Numeric, sanitizer, timing, SASS evidence | Not claimed by this GPU-free check | Not claimed by this GPU-free check | Not claimed by this GPU-free check |

[Walkthrough, verbatim diagnostics, and annotation burden: T3 — Swap a stage; break a contract](../../docs/tutorials/T3-contract-gallery.md).
