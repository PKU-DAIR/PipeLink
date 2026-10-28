# Contributing to PipeLink

PipeLink is a research prototype: a linker for separately compiled, opaque partial
GPU kernels. A contribution must keep every result correct and keep the acceptance
evidence behind it intact, so never weaken a gate or swap in an easier comparison to
make a result pass. Prefer clean cutovers to compatibility shims, and derive facts
instead of adding redundant validation or authoring ceremony.

Teaching examples live in `examples/`; the attention, megakernel and MoE training
case studies live in `campaigns/`. After cloning, run
`git submodule update --init --recursive` to fetch the pinned `third_party/`
sources and their nested dependencies.

## Evidence records

Campaign tools write machine-readable receipts (`results/*.json`, `evidence/*.json`)
locally; they are not published. The committed Markdown reports next to them are
rendered from those receipts and are the published evidence.

## Checks and evidence

From the repository root, run the GPU-free tests (GPU-gated cases skip without
hardware; see [pipec/README.md](pipec/README.md) for environment and toolchain setup):

```sh
uv run --project pipec --locked python -m pytest pipec/tests evaluator
```

Every new composition needs this evidence:

- an oracle `allclose` check;
- a bitwise linked/control comparison where the computation is deterministic;
- same-source parity timing, with clocks reported;
- **racecheck and synccheck on both the linked and the one-TU executions**, with no opt-out;
- recorded normalized SASS diagnostics, with any differences explained.

**Do not hash-pin rebuildable artifacts:** no freeze markers, hash ledgers or public
content-hash literals. Check byte identity only where it is the hypothesis
(declaration or layout identity, package-invariance and source-withholding proofs,
bitwise numerical comparisons). Payload digests may be recorded as provenance but
must not gate. Dependency integrity pins are fine. Preserve third-party notices;
PipeLink's license and attribution inventory are in [LICENSE](LICENSE) and
[NOTICE](NOTICE).
