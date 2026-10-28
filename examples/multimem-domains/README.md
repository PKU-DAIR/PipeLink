# Two domains, multicast and peer views

This `sm_103a` example links two separately packed communication owners into one
persistent kernel. One Python process drives four ranks, with one execution backend
per device, and checks the result against a same-source one-translation-unit control image.

| Domain | Members (logical ranks) | Communication view |
|---|---|---|
| `tp` | `{0,1}`, `{2,3}` | Multicast |
| `dp` | `{0,2}`, `{1,3}` | Peer address tables |

Owner `mc` declares `tp`; owner `dp` declares both domains. Rank `r` therefore receives
`tp=Domain(r % 2, 2)` and `dp=Domain(r // 2, 2)`. Both domains have two members, but
equal sizes do not make the names interchangeable. The two owners occupy disjoint
warps and use separate named barriers.

## Data and ordering

`mc.tp-allreduce` first performs a system-scope multicast entry barrier. All payload
lanes rendezvous, one lane publishes an arrival with `multimem.red.release.sys`,
fences the alias proxy and polls its local counter with `ld.acquire.sys`; a named
barrier passes this ordering on to the role's other lanes. Each rank reduces its half
of each slice with vectorized `multimem.ld_reduce.add.f32`, multicasts the result
with `multimem.st`, then publishes completion.

`dp.gather` waits for TP completion, executes `fence.proxy.alias`, and reads the
reduced payload through its **local unicast view**. Its `dp.red` channel declares
`.domain("tp")` while `mc.red` declares `.multicast("tp")`; both borrow caller
source `red`. Because the same memory is accessed through both views, the alias-proxy
fence is required. After DP's peer entry barrier, each rank stores its slice into
every DP member's output at row block `dp_rank`; system fences and release/acquire
counters complete the gather. Every rank waits in the same global order: **TP, then DP**.
The linker checks that the domains agree; the alias fences and distributed liveness
are the device code's responsibility.

The resulting output on every rank is

```text
out[0:elements]          = input_rank_0 + input_rank_1
out[elements:2*elements] = input_rank_2 + input_rank_3
```

The reference adds two finite float32 terms and compares every output bit,
including outputs that were poisoned before launch. A CTA processes slices
`cta, cta + grid, ...` in fixed order. The provider that owns launch geometry sets
`grid = min(slices, observed_sm_count)` from the observed hardware; the runner
requires both images on every rank to support at least that many resident clusters.

Counters, reduced storage and each owner's per-slice epochs are lent by the caller
and shared by all sessions; entry barriers make reuse across launches safe. Each
session has its own input subview and output allocation. Per TP pair, all input
generations, the shared reduced storage and the TP counters are subviews of one
multicast-bound allocation, so a logical channel's size need not match the
allocation's binding granularity.

## Build and run

Configure the repository environment as described in the root README. The build uses
CUDA 13.3 and needs no GPU:

```sh
uv run --project pipec --locked python examples/multimem-domains/build.py --profile sm_103a
```

The build produces two `.plink` packages, two provider DSOs, the linked image,
whole-program and backend-matched one-TU controls, and a spin helper used only by
the caller. The runner uses only these artifacts, without importing stage sources or
provider internals.

Select four available, mutually connected multicast-capable GPUs with
`PIPELINK_GPUS`, on a host that supports multicast memory binding. Run through the
repository GPU window wrapper, with a timeout in case a collective hangs:

```sh
source tools/env.sh
tools/gpu_window.sh timeout 240 uv run --project pipec --locked python examples/multimem-domains/run.py check
tools/gpu_window.sh timeout 900 uv run --project pipec --locked python examples/multimem-domains/run.py sanitize
tools/gpu_window.sh timeout 240 uv run --project pipec --locked python examples/multimem-domains/run.py time
uv run --project pipec --locked python examples/multimem-domains/run.py sass
```

`check` runs both images with changing inputs across direct launches, three replays
of a fixed-parameter graph, and six changing-payload generations queued without a
cross-rank join between generations. Rotating per-rank spin delays add skew; the same
skewed sequence is also captured as a graph on every rank and replayed three times.
Every generation is checked. The default 257 slices also exercise the persistent
loop on devices with fewer SMs. `sanitize` runs racecheck and synccheck separately
on both images in bounded child processes and keeps the raw tool logs.

`time` uses the shared parity harness: eight balanced ABBA/BAAB blocks with
calibration, CUDA-event batch boundaries across all ranks, and a one-TU/linked speedup
floor of **1/1.03 (≈0.970874×)**. Speedups are medians of per-block one-TU
milliseconds / linked milliseconds; higher is better. Clocks are left unchanged,
and a run where the GPU window reports `contended=1` is invalid for timing.
`sass` uses the evaluator's normalized comparison, adds the backend-matched control
when useful, and is diagnostic only.

## Domain-disagreement negative demo

Every build also repacks the DP stage with `-D MULTIMEM_DOMAINS_BAD_DP_RED=1`, so that
`dp.red` declares domain `dp` while `mc.red` still declares `tp` over the same caller
source `red`. The ordinary public link command must reject this composition with
**L-HOSTCHAN**, before any GPU or runtime rank information is involved; the build fails
if it does not. The diagnostic and observed clause are saved in
`out/sm_103a/negative-domain/result.json` and `out/sm_103a/logs/negative-domain.link.log`.

All generated evidence is kept under the ignored `out/sm_103a/`: `results-check.json`,
`results-sanitize.json`, `results-time.json`, `results-sass.json`, sanitizer logs,
and normalized SASS streams/comparisons.

## Measured results

Measured on four available `sm_103a` devices with CUDA 13.3, 148 observed SMs per
device, 64-thread CTAs and no dynamic shared memory. The GPU window wrapper reported
no contention for any run, and clocks were unlocked. These numbers characterize this
example, not transport bandwidth.

| Check | Observed result |
|---|---|
| Numerical/bitwise coverage | **PASS**; 33 generation observations per image × 4 ranks, zero mismatches; 263,168 input elements per rank |
| Direct / graph / pipelined-skew / graph-pipelined-skew | **PASS** on linked and one-TU images; six distinct input/output generations, three replays of each graph |
| Residency | Grid 148; `max_active_clusters=1184` on every rank and both images |
| Multicast subviews | 7,371,264 logical bytes per TP backing; observed recommended binding granularity 536,870,912 bytes |
| Racecheck | **PASS** on both images; zero hazards, errors or warnings |
| Synccheck | **PASS** on both images; zero errors |
| One-TU / linked speedup | **1.000114×**, passing the ≥1/1.03 floor; 4,194,304 input elements per rank, eight balanced blocks |
| Mean all-rank event time | Linked **2.549993 ms**, one-TU **2.550528 ms** per launch |
| Normalized SASS (diagnostic) | All comparison checks equal; **480 instructions** in each image, no local-memory traffic or calls |
| Negative domain link | **L-HOSTCHAN**, `host.channels.mc.red.domain`; rejection before GPU use |

The sanitizers ran a smaller workload (four slices, three generations) through the
same direct, graph and skew paths. A clean sanitizer run does not show that the tools
can detect races across devices or through multicast aliases; the bitwise
multi-generation tests and the explicit protocol ordering carry that assurance.
