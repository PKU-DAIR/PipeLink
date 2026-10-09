# Pack and link reference

## Pack

```sh
uv run --project pipec pipec2 pack path/to/stage.cu -o stage.plink \
  -I path/to/includes -D TILE=128 -D PIPELINE_DEPTH=2 \
  --profile sm_103a --toolkit cuda-13.3.0
```

`pack <stage.cu>` compiles one annotated translation unit into one device object
and writes a `.plink` container: a manifest plus `stage.o`. Variants are declared
in the TU, not built from separate define sets.

| Option | Meaning |
|---|---|
| `-I DIR`, `-D K=V` | Include roots and definitions, repeatable. |
| `--profile` | Target profile, default `sm_103a`; see [Profiles](profiles.md). The only place an architecture is chosen. |
| `--toolkit` | Must equal the profile's toolkit, `cuda-13.3.0`. |
| `--nvcc-flag=FLAG` | Extra nvcc flag for both compilations, repeatable. |
| `--report PATH` | Write the pack report. |

### What pack runs

1. **Emit.** Compile and run the TU's host projection:
   `nvcc -x cu -DPIPELINK_EMIT -std=c++17 --expt-relaxed-constexpr -gencode arch=<compute>,code=<sm>`
   with the packaged runner and pipec's include directories. The annotations
   register compiler-observed layouts and declarations. The program runs with
   GPUs hidden.
2. **Payload.** Compile the device object with
   `-gencode arch=<compute>,code=<lto> -dc -rdc=true -O3 -DNDEBUG -std=c++17` and the
   same user includes and definitions.
3. **Observe.** Read defined and undefined device symbols with `nm`, seed an `-O0`
   nvJitLink probe at the profile's `lto_base`/`sm` spellings, and read launch and
   cluster attributes with `cuobjdump --dump-elf`.
4. **Assemble.** Derive declaration digests and role warp claims, fit each finite
   map to a tensor-v2 expression (`T-FIT` on failure, no table fallback), reject
   cross-source inconsistencies and write the schema-3.0 manifest.

`<compute>`, `<sm>` and `<lto>` are the profile's spellings, for example
`compute_103a`/`sm_103a`/`lto_103a`. Pack uses no GPU. Scratch files go under
`PIPELINK_SCRATCH_ROOT` when set, otherwise a temporary directory; nothing is
written beside the source.

Floating-point code-generation switches among the nvcc flags (`--use_fast_math`,
`--ftz=`, `--prec-div=`, `--prec-sqrt=`, `--fmad=`, applied in order) are recorded as
the target's `math` mode. Because LTO code generation happens at the final link,
the linker passes the stages' shared mode to nvJitLink; stages packed with
different modes fail `L-LINK` at `stages.target.math`. A manifest without `math`
uses nvcc's defaults.

A pipeline may instead set the image's mode itself:
`"pipeline": {..., "math": {"ftz": true, "prec_div": false, "prec_sqrt": false, "fmad": true}}`.
The linker then uses that mode for every stage, whatever modes they were packed
with, and `link.json` records it under `math_override` (`mode`, plus the `packed`
mode of each stage that differs). It changes only these four code-generation
switches: `--use_fast_math` intrinsic substitution (`sinf` to `__sinf` and so on)
happens in the front end and is already in a stage's IR. Forcing a mode changes
the owners' numerics, so a same-source control must use the same switches.

### What the manifest records

- `symbols`: `defined_strong`, `defined_weak` and `undefined` device symbols,
  observed by mechanism `nm-external-text+nvjitlink-seeded-v2`.
- `provenance`: mechanism `pack-cu-v1`. Its `inputs` bind the TU, compiler
  recipe, annotation header, runner, registry and enumerated maps by SHA-256;
  `sections` mark each fact as declared, compiled, observed or fitted.
- The target stamp `pipelink_abi.target`: profile, LTO architecture, toolkit, and
  the optional host ABI and math mode.

Fitting certifies that an expression equals the author's enumerated map, not that
the opaque instructions implement the declaration.

## Trust model

**Pack attests the object; link trusts declarations and the packed attestation.**

- `pack` checks the declared device-symbol surface against the object and records
  the payload digest and symbol tables.
- `link` uses those tables for symbol checks instead of probing each object, and
  still checks declarations, compatibility and resource legality. A package
  without `symbols` fails closed; repack it.
- Ordinary reads do not re-hash `stage.o`. `pipec2 inspect` checks the payload
  digest explicitly; it is optional and does not strengthen the model.
- Link does not detect a manifest that lies about its object. This is not
  adversarial verification.

Opaque stages are trusted to honor their declarations. Native numerical checks
and sanitizers establish runtime correctness.

## Link

```sh
uv run --project pipec pipec2 link pipeline.json --out-dir out/link
```

The pipeline manifest lists stage packages by `artifact_path` (relative to the
manifest), pipeline-owned handles, phases, exclusive phase pairs, priorities,
forced bindings and topology; [T0](../tutorials/T0-pack-link.md#4-describe-the-composition-and-link-it)
writes one. Link takes the target from the pipeline's `pipelink_abi.target`;
every stage must carry the same stamp. It then:

1. loads the packages and checks declared symbols against the packed tables;
2. plans tensor connections, if any ([Tensor imports](tensor-imports.md#link-flow));
3. pairs endpoints, drops unreferenced exports and their associated endpoints,
   and unifies each pair strictly;
4. expands protocols and lays out the params aggregate (or the host scalar image
   and residues, [Host ABI](host-abi.md#what-the-link-publishes));
5. solves resources;
6. generates handle stubs, call trampolines and the dispatch kernel;
7. device-links everything with nvJitLink at the profile's architecture and the
   solved register cap, then audits the observed register allocation. If it does
   not fit, the link re-solves once with the observed allocation as a bound, and
   fails `L-REGS` if the second image still does not fit.

Outputs are `pipeline.cubin`, `link.json` and the generated sources. Read the
kernel symbol, `threads`, `dyn_smem_bytes`, cluster and params layout from
`link.json`; the launcher does ([Launching](launching.md)).

Failures are structured clauses with a JSON path and facts. `pipec2 clauses`
lists them; see the [clause catalogue](clauses.md). New clauses register through
`pipec2.model.register_clause(...)`.

## Resource solver

The solver uses pinned `z3-solver==4.15.4.0` with a fresh context, seed 0, one
thread and canonical constraint order. It solves native and physical capacities,
alignment, offsets, fixed and forced bindings, aliases, lifetimes, exact warp
coverage, variants and register groups jointly.

The objective is lexicographic:

1. maximize the declared depth vector;
2. maximize the register vector;
3. minimize the high-water mark of each physical domain: shared memory, then
   TMEM, then named barriers, then other physical domains in sorted order (warps
   are an exact tiling);
4. minimize variant indices in sorted stage order;
5. minimize all remaining finite values in sorted handle order.

Bases are not an objective: once every tier is fixed, the published bases are the
deterministic model Z3 returns. `dyn_smem_bytes` is the shared-memory tier's value.

Each high-water tier first computes a lower bound from handles that must be
pairwise disjoint (no alias, phase exclusion or proved lifetime order). If the
incumbent meets it, no search runs. Otherwise the solver tries first-fit packing
hints, then gallops from the incumbent, each query capped at a quarter of the
remaining allowance. The whole solve has 10,000,000 Z3 resource units; no
wall-clock timeout is used. Exhausting them before any feasible layout fails
`L-BUDGET`. When a tier stops short of its bound, the feasible incumbent is
published. `link.json.solver.tiers` reports each tier's `value`, `bound` and
`status` (`proved`, `bound_met` or `budget_gap`), and `solver.rlimit_used` the
units consumed.

An unsatisfiable link reports a canonical conflicting set (not a minimal core),
with clause precedence `L-OVERLAP`, `L-HANDLE`, `L-TOPO`, `L-REGS`, `L-ALIGN`,
`L-PHASE`, `L-OWNER`, `L-PROTO`, then path order. Core extraction has a separate
1,000,000-unit allowance.
