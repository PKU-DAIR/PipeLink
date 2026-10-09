# Profile consumer

A three-stage interpreter composes two instruction stages using a descriptor, phase-exclusive shared-memory pages, and a grid-scoped release/acquire counter. The linked image launches through a host ABI v3 owner provider `.so`, caller-owned input fixtures, and instruction parameter blobs; a hand-written one-TU control includes the same three stage sources. Hopper/sm90, Blackwell/sm103 and consumer Blackwell/sm120 use the same source. The warp roles use unaligned `barrier.sync` because their named barriers are reached at different instruction sites.

## Commands

From the repository root, with `uv` and CUDA 13.3 installed and `PIPELINK_CUDA_HOME`, `PIPELINK_SCRATCH_ROOT`, and `PIPELINK_GPUS` configured in `.env`:

```bash
# GPU-free: pack all three stages, link, build provider.so, one-TU cubin, and runner.
uv run --project pipec --locked python examples/profile-consumer/build.py --profile sm_90a
uv run --project pipec --locked python examples/profile-consumer/build.py --profile sm_103a
uv run --project pipec --locked python examples/profile-consumer/build.py --profile sm_120a

# On a matching GPU, run the profile's recipe; the selected physical GPU is CUDA device 0.
uv run --project pipec --locked python examples/profile-consumer/run.py --profile sm_103a
uv run --project pipec --locked python examples/profile-consumer/run.py --profile sm_120a

# On Hopper/sm90, build there first and select an available physical GPU.
uv run --project pipec --locked python examples/profile-consumer/build.py --profile sm_90a
uv run --project pipec --locked python examples/profile-consumer/run.py --profile sm_90a
```

Omit `--gpu` to select the first `PIPELINK_GPUS` entry. Build and run load `.env` through `tools/pipelink_env.py`; temporary compiler files use `PIPELINK_SCRATCH_ROOT`. Artifacts, fixtures, logs, sanitizer reports, SASS diagnostics, and each new `results.json` are written under ignored `build/<profile>/`.

`run.py` retains the existing Gate A numerical/byte-comparison and structural checks, runs racecheck and synccheck unconditionally on both launch routes, and exits nonzero if either acceptance or a sanitizer fails. Its linked command is `pipec2 launch ... --provider ...provider.so --host-inputs ... --blob ... --cluster 1,1,1`. Normalized SASS comparison is diagnostic only, never an equality gate.

## Evidence

Measured with CUDA **13.3** on each profile; clocks are observed, not locked. The scope
here is Gate A and synchronization correctness, not timing parity. Each run writes
`build/<profile>/results.json`.

| Evidence | Blackwell/sm103 | Hopper/sm90 | Consumer Blackwell/sm120 |
|---|---|---|---|
| Pack, link, provider `.so`, one-TU, runner | PASS | PASS | PASS |
| Overall acceptance / `gate_a` | PASS / `true` | PASS / `true` | PASS / `true` |
| Oracle mismatches, linked / one-TU | 0 / 0 of 64 integer elements | 0 / 0 of 64 integer elements | 0 / 0 of 64 integer elements |
| Linked vs one-TU input and output bytes | Both exact | Both exact | Both exact |
| Counter, errors, linked / one-TU | 1, 0 / 1, 0 | 1, 0 / 1, 0 | 1, 0 / 1, 0 |
| Shared page bindings / counter slot | 1 and 1 / 17 | 1 and 1 / 17 | 1 and 1 / 17 |
| Racecheck, linked / one-TU | PASS / PASS | PASS / PASS | PASS / PASS |
| Synccheck, linked / one-TU | PASS / PASS | PASS / PASS | PASS / PASS |
| Registers, linked / one-TU | 28 / 28; stack and local bytes 0 / 0 | 26 / 26; stack and local bytes 0 / 0 | 28 / 28; stack and local bytes 0 / 0 |
| Normalized SASS instructions, linked / one-TU | 256 / 256; opcode/modifier sequences equal; symbols and parameter text differ | 264 / 264; opcode/modifier sequences differ; diagnostic only | 256 / 256; opcode/modifier sequences equal; symbols and parameter text differ |
| Local loads, local stores, calls | 0, 0, 0 in both images | 0, 0, 0 in both images | 0, 0, 0 in both images |
| Parity timing | Not measured: the hand one-TU control uses a different parameter ABI from the linked host ABI v3 image; this example has no shared-session timing control | Not measured | Not measured |

Start with [T1: Hello, callable seam](../../docs/tutorials/T1-hello-callable.md) for the pack/link/launch workflow; the [launch reference](../../docs/reference/launching.md#host-abi-v3-launch) describes the host ABI v3 provider route.
