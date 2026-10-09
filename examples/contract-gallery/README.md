# Contract gallery

This GPU-free example reuses the T2 mbarrier-ring producer package with the original affine-ReLU consumer and a square consumer, then checks six intentionally wrong consumer contracts at their actual rejection phase. The checker invokes the `pipec2` CLI, requires `P-INIT` at pack with no package produced, and requires the other five clauses only after a successful pack. It uses the shared T2 pipeline builder, machine-local `.env` defaults, and ignored `out/<profile>/` artifacts; it never launches a kernel.

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
