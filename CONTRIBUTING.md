# Contributing to PipeLink

PipeLink is a research prototype: a linker for separately compiled, opaque partial
GPU kernels. Contributions must preserve correctness and acceptance evidence;
never weaken a gate or substitute an easier comparison to make a result pass.
Prefer clean cutovers over compatibility shims and derive
facts rather than adding redundant validation or authoring ceremony.

Teaching examples live in `examples/`, and the attention and megakernel case
studies in `campaigns/`. After cloning, run
`git submodule update --init --recursive` to obtain the pinned `third_party/`
sources and their nested dependencies.

## Evidence records

Campaign tools write machine-readable receipts (`results/*.json`, `evidence/*.json`)
locally; they are not part of the published tree. The committed Markdown reports
next to them are rendered from those receipts and are the published evidence.

## Checks and evidence

From the repository root, run the GPU-free tests (see [pipec/README.md](pipec/README.md)
for environment/toolchain setup; GPU-gated cases skip without hardware):

```sh
uv run --project pipec --locked python -m pytest pipec/tests evaluator
```

For every new composition, provide oracle `allclose`, linked/control bitwise
comparison where deterministic, same-source parity timing with clocks reported,
and **racecheck + synccheck on both linked and one-TU executions, without an
opt-out**. Record normalized SASS diagnostics and explain differences.

**No hash-pinning of rebuildable artifacts:** no freeze markers, hash ledgers or
public content-hash literals. Byte identity belongs only where it is the
hypothesis (declaration/layout identity, package-invariance/source-withholding
proofs, numerical bitwise comparisons); payload digests may be non-gating
provenance. Dependency integrity pins remain valid. Preserve third-party notices;
PipeLink's license and attribution inventory are in [LICENSE](LICENSE) and
[NOTICE](NOTICE).
