# Two domains, multicast and peer views

This `sm_103a` example composes two separately packed communication owners into one
persistent kernel. One Python process drives four ranks, with one execution backend
per device and a same-source one-translation-unit control image.

| Domain | Members (logical ranks) | Communication view |
|---|---|---|
| `tp` | `{0,1}`, `{2,3}` | Multicast |
| `dp` | `{0,2}`, `{1,3}` | Peer address tables |

Owner `mc` declares `tp`; owner `dp` declares both domains. Thus rank `r` receives
`tp=Domain(r % 2, 2)` and `dp=Domain(r // 2, 2)`. Equal domain sizes do not make the
names interchangeable. The two owners occupy disjoint warps and use separate named
barriers.

## Data and ordering

`mc.tp-allreduce` first performs a system-scope multicast entry barrier. All payload
lanes rendezvous, one lane publishes an arrival with `multimem.red.release.sys`,
fences the alias proxy and polls its local counter with `ld.acquire.sys`; a named
barrier hands this ordering to the role's other lanes. Each rank reduces its half
of each slice using vectorized `multimem.ld_reduce.add.f32` and multicasts the result
with `multimem.st`, then publishes completion.

`dp.gather` waits for TP completion, executes `fence.proxy.alias`, and consumes the
reduced payload through its **local unicast view**. Its `dp.red` channel declares
`.domain("tp")`, whereas `mc.red` declares `.multicast("tp")`; both borrow caller
source `red`. This mixed-view use is why the alias-proxy fence is necessary. DP's
peer entry barrier precedes stores of the slice into every DP member's output at
row block `dp_rank`; system fences and release/acquire counters complete the gather.
Every rank waits in the global order **TP, then DP**. The linker checks domain
agreement, not the device protocol's alias fences or distributed liveness.

The resulting output on every rank is

```text
out[0:elements]          = input_rank_0 + input_rank_1
out[elements:2*elements] = input_rank_2 + input_rank_3
```

The reference uses float32 addition of two finite terms and compares every output
bit, including poisoned outputs. A CTA processes slices `cta, cta + grid, ...` in
fixed order. The geometry-authority provider chooses
`grid = min(slices, observed_sm_count)` from the hardware observation; the runner
requires both images on every rank to have at least that many resident clusters.

Counters, reduced storage and each owner's per-slice epochs are caller-lent and
shared by all sessions. Entry barriers protect reuse across launches. Each
session has its own input subview and output allocation. All input generations,
shared reduced storage and TP counters are subviews of one multicast-bound backing
per TP pair: logical channel sizes need not equal the backing's binding granularity.

## Build and run

Configure the repository environment as described in the root README. Build with
CUDA 13.3; no GPU is used by the build:

```sh
uv run --project pipec --locked python examples/multimem-domains/build.py --profile sm_103a
```

The build produces two `.plink` packages, two provider DSOs, the linked image,
whole-program and backend-matched one-TU controls, and a caller-only spin helper.
The runner consumes this artifact distribution; it never imports stage source or
provider implementation details.

Select four available, mutually connected multicast-capable GPUs with
`PIPELINK_GPUS`. Run on a host supporting multicast memory binding; use the
repository GPU window wrapper and a timeout to bound a broken collective:

```sh
source tools/env.sh
tools/gpu_window.sh timeout 240 uv run --project pipec --locked python examples/multimem-domains/run.py check
tools/gpu_window.sh timeout 900 uv run --project pipec --locked python examples/multimem-domains/run.py sanitize
tools/gpu_window.sh timeout 240 uv run --project pipec --locked python examples/multimem-domains/run.py time
uv run --project pipec --locked python examples/multimem-domains/run.py sass
```

`check` covers both images, changing inputs across direct launches, three fixed
parameter graph replays, and six changing-payload generations queued without an
inter-generation cross-rank join. Rotating per-rank spin delays introduce skew;
that same sequence is captured separately on every rank and replayed three times.
Every generation is checked. The default 257 slices also exercise the persistent
loop on devices with fewer SMs. `sanitize` runs racecheck and synccheck separately
on both images with bounded child processes and preserves raw tool logs.

`time` uses the shared parity harness's eight balanced ABBA/BAAB blocks and
calibration, all-rank CUDA-event batch boundaries, and the linked/one-TU ratio gate
of **1.03**. Clocks are not changed. Any GPU-window `contended=1` run is invalid for
timing. `sass` uses the evaluator's normalized comparison, includes the
backend-matched control when useful, and is diagnostic rather than a gate.

## Domain-disagreement negative demo

Every build also repacks the DP stage with
`-D MULTIMEM_DOMAINS_BAD_DP_RED=1`: `dp.red` then declares domain `dp` while
`mc.red` still declares `tp`, though both use caller source `red`. The ordinary
public link command must reject this composition with **L-HOSTCHAN**. No GPU or
runtime rank facts are involved. The build fails if this rejection is absent.
The diagnostic and observed clause are saved in
`out/sm_103a/negative-domain/result.json` and `out/sm_103a/logs/negative-domain.link.log`.

All generated evidence is retained under ignored `out/sm_103a/`: `results-check.json`,
`results-sanitize.json`, `results-time.json`, `results-sass.json`, sanitizer logs,
and normalized SASS streams/comparisons.

## Measured results

The following measurements use four available `sm_103a` devices, CUDA 13.3,
148 observed SMs per device, 64-thread CTAs, and no dynamic shared memory.
The GPU-window wrapper reported no contention for any GPU run. Clocks were
unlocked. These are measurements of this witness, not transport bandwidth claims.

| Check | Observed result |
|---|---|
| Numerical/bitwise coverage | **PASS**; 33 generation observations per image × 4 ranks, zero mismatches; 263,168 input elements per rank |
| Direct / graph / pipelined-skew / graph-pipelined-skew | **PASS** on linked and one-TU images; six distinct input/output generations, three replays of each graph |
| Residency | Grid 148; `max_active_clusters=1184` on every rank and both images |
| Multicast subviews | 7,371,264 logical bytes per TP backing; observed recommended binding granularity 536,870,912 bytes |
| Racecheck | **PASS** on both images; zero hazards, errors or warnings |
| Synccheck | **PASS** on both images; zero errors |
| Linked / one-TU timing | **0.999886×**, passing the ≤1.03 gate; 4,194,304 input elements per rank, eight balanced blocks |
| Mean all-rank event time | Linked **2.549993 ms**, one-TU **2.550528 ms** per launch |
| Normalized SASS (diagnostic) | All comparison checks equal; **480 instructions** in each image, no local-memory traffic or calls |
| Negative domain link | **L-HOSTCHAN**, `host.channels.mc.red.domain`; rejection before GPU use |

Sanitizers ran a smaller four-slice, three-generation workload with the same
direct/graph/skew paths. Their successful execution is not a proof that the tools
detect inter-device or multicast alias races: the bitwise multi-generation tests
and explicit protocol ordering remain essential.
