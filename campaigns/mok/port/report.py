#!/usr/bin/env python3
"""Render the MoK evidence report from one receipt directory.

    python3 campaigns/mok/port/report.py EVIDENCE_DIR [--output REPORT.md]
    python3 campaigns/mok/port/report.py campaigns/mok/evidence      # re-render the reference REPORT.md

EVIDENCE_DIR layout (every producer writes into it; MOK_EVIDENCE_DIR selects the directory,
default $PIPELINK_SCRATCH_ROOT/mok/evidence):

  upstream.json                 host/collect_upstream.py: upstream MoK build, tests, sanitizers
  matrix/payload-fanout.json    port/build.py --out EVIDENCE_DIR/matrix (24 links, one payload per op)
  matrix/links/<cell>/link.json   (same build; launch geometry)
  numeric-summary.json          port/aggregate.py numeric  <- host/run.py check / sanitize receipts
  timing-summary.json           port/aggregate.py timing   <- host/run.py time receipts
  placement-summary.json        port/aggregate.py placement <- host/benchmark_upstream.sh logs
  sass.json                     port/sass.py (diagnostic)
  checked-facts/{facts-change,negative,rdc-contrast,skew}.json
                                port/checked_facts/run_checked_facts.py <mode> [--gpu]
  checked-facts/gpu-runs.json   port/aggregate.py facts-gpu <- host/run.py check receipts
  owner-swap/{result,source-absence,payload-hashes,matrix-summary}.json
                                port/owner_swap/build_owner_swap.py
  owner-swap/include-token-audit.json   port/owner_swap/audit_sources.py
  owner-swap/gpu-results.json   port/aggregate.py swap <- owner_swap/check_swap.py + host/run.py time
  owner-swap/sanitize/swap-sanitize-sanitize-summary.json
                                host/run.py sanitize --build-dir <swapped build>; per-rank receipts beside it
  cost/annotation-counts.json   port/cost/cost.py --annotations-only
  cost/cost.json                port/cost/cost.py --build-gates ... (optional: build-time measurements)
  trace-summary.json, inherited-causes.json, capacity.json
                                optional analyses recorded for the reference run only

A section whose inputs are absent is reported as not run. Only explicitly selected fields
reach the Markdown: no logs, commands, machine identities, physical GPU indices, local paths
or content digests.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from statistics import median

HERE = Path(__file__).resolve().parent
README = HERE.parent / "README.md"
BANNER = ("> Unlocked or unrecorded SM clocks: timings are valid as relative comparisons only, "
          "not comparable with the paper's locked-clock numbers.")


class Evidence:
    def __init__(self, root):
        self.root = Path(root)

    def __call__(self, name):
        return json.loads((self.root / name).read_text())

    def has(self, *names):
        return all((self.root / name).exists() for name in names)


def cell(value):
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (dict, list)):
        value = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "|" + "---|" * len(headers),
        *("| " + " | ".join(cell(v) for v in row) + " |" for row in rows),
    ])


def counts(values):
    return "; ".join(f"{value}: {n}" for value, n in sorted(Counter(values).items(), key=lambda x: str(x[0])))


def section(title, *parts):
    return "\n\n".join([f"## {title}", *parts])


def not_run(title, *names):
    return section(title, "Not run: " + ", ".join(f"`{n}`" for n in names) + " absent from the evidence directory.")


def clock_status(record):
    """True/False when the receipt says whether clocks were locked; None when it does not."""
    value = record.get("clock_locked")
    return value if isinstance(value, bool) else None


def clock_text(locked, mhz):
    return f"{mhz} MHz clock lease" if locked and mhz else "unlocked SM clocks"


def baseline(ev, claims):
    if not ev.has("upstream.json"):
        return not_run("Upstream baseline", "upstream.json")
    d = ev("upstream.json")
    tests = []
    for rank in sorted(d["tests"]["ranks"], key=lambda r: r["rank"]):
        totals = [sum(f[k] or 0 for f in rank["files"].values()) for k in ("passed", "failed", "errors", "skipped")]
        tests.append([rank["rank"], *totals])
    sanitizer_rows = []
    for run in sorted(d["sanitizers"], key=lambda r: (r["pass"], r["precision"], r["tool"])):
        sanitizer_rows.append([run["pass"], run["precision"], run["tool"],
                               counts(r["device_error_or_hazard_count"] for r in run["ranks"]),
                               counts(r["host_api_error_count"] for r in run["ranks"])])
    toolchain = d.get("toolchain") or {}
    claims["mok.upstream_build_exit"] = (d["build"]["final_attempt"] or {}).get("exit_code")
    return section("Upstream baseline",
        f"MoK `{d['upstream_revision'][:7]}`; ThunderKittens `{d['thunderkittens_revision'][:7]}`; "
        f"EP{d['ep']}; {toolchain.get('visible_device_count', 'unrecorded')} Blackwell/sm103 GPUs. "
        f"Torch {toolchain.get('torch', 'unrecorded')}; Python {toolchain.get('python', 'unrecorded')}; CUDA 13.3. "
        f"Status: **{d['status']}**. Build exit {(d['build']['final_attempt'] or {}).get('exit_code')}; "
        f"test exit {(d['tests']['run'] or {}).get('exit_code')}.",
        "Baseline patches: " + ", ".join(f"`{Path(p).name}`" for p in d["build"]["patches"]) +
        ". No port device-synchronization patches are applied.",
        table(["Logical rank", "Passed", "Failed", "Errors", "Skipped"], tests),
        "\n".join("- " + message for message in sorted({r['message'] for rank in d['tests']['ranks'] for r in rank['details']})),
        "In the sanitizer count columns, `a: b` means b logical ranks reported a. Backward includes forward setup.",
        table(["Pass", "Precision", "Tool", "Device reports", "Host API diagnostics"], sanitizer_rows),
        "Racecheck reports the two-CTA tensor allocator and the MXFP8 backward gate/up scale-word accesses, "
        "which read four-byte words and write two-byte halves in both pipeline stages. "
        "Memcheck's host API diagnostics come from torch's FABRIC probe.")


def payloads(ev, claims):
    if not ev.has("matrix/payload-fanout.json"):
        return not_run("Payload fanout", "matrix/payload-fanout.json")
    d = ev("matrix/payload-fanout.json")
    first = sorted(d["links"], key=lambda r: r["id"])[0]["id"]
    launch = ev(f"matrix/links/{first}/link.json")
    rows = [[a['stage_id'], len(a['payloads']), sum(len(p['variants']) for p in a['payloads']), len(a['links'])]
            for a in sorted(d["artifacts"], key=lambda a: a["stage_id"])]
    dropped = {}
    for link in d['links']:
        dropped.setdefault(link['id'].split('-')[1], set()).add(len(link['dropped_endpoints']['observed']))
    claims["mok.kernels"] = len(d["links"])
    claims["mok.packages"] = len(d["artifacts"])
    claims["mok.packages_in_every_link"] = all(len(a["links"]) == (8 if ".interp." in a["stage_id"] else len(d["links"]))
                                               for a in d["artifacts"])
    return section("Payload fanout",
        f"{len(d['links'])} compositions: 3 passes × 2 precisions × 2 EP choices × 2 clamp choices. "
        f"Profile `{d['profile']}`; inputs unchanged: {cell(d['inputs_unchanged'])}. "
        f"Four packages per composition (one interpreter and three op owners); {launch['threads']} worker threads, "
        f"cluster {launch['cluster']}.",
        table(["Package", "Physical payloads", "Variants", "Links"], rows),
        table(["Pass", "Dropped endpoint counts"], [[p, sorted(v)] for p, v in sorted(dropped.items())]),
        table(["Cell", "Variant indices: interpreter/comm/GEMM/act"],
              [[r['id'], '/'.join(str(p['variant']['index']) for p in [r['interpreter'], *r['ops']])]
               for r in sorted(d['links'], key=lambda r: r['id'])]))


def swap_sanitizers(ev, claims):
    name = 'owner-swap/sanitize/swap-sanitize-sanitize-summary.json'
    if not ev.has(name):
        return ['No sanitizer results are reported for this owner.']
    runs = ev(name)['runs']
    claims["mok.swap_sanitizer_device_findings"] = sum(x["device_error_or_hazard_count"] or 0
                                                      for r in runs for x in r["sanitizer_rank_counts"])
    return ['Sanitizers wrap every rank of the swapped build: EP4 forward/backward sanity cells, both precisions and '
            'clamp branches, both images, unlocked clocks. In the count columns, `a: b` means b logical ranks reported a.',
            table(["Tool", "Image", "Status", "Device errors/hazards", "Host API diagnostics"],
                  [[r["tool"], r["image"], r["status"],
                    counts(x["device_error_or_hazard_count"] for x in r["sanitizer_rank_counts"]),
                    counts(x["host_api_error_count"] for x in r["sanitizer_rank_counts"])] for r in runs])]


def parity(ev, claims):
    if not ev.has("numeric-summary.json"):
        return not_run("Numeric parity and sanitizers", "numeric-summary.json")
    d = ev("numeric-summary.json")
    rows = [[r["world"], r["rank"], r["configurations"],
             r["statuses"].get("PASS", 0), r["statuses"].get("FAIL-inherited", 0),
             r["statuses"].get("FAIL-inherited-crash", 0), r["direct_generations"],
             r["graph_image_generations"], r["bitwise_failures"], r["recompute_exact_failures"]]
            for r in d["ranks"]]
    sanitizer = [[r["tool"], r["image"], r["status"],
                  counts(x["device_error_or_hazard_count"] for x in r["sanitizer_rank_counts"]),
                  counts(x["host_api_error_count"] for x in r["sanitizer_rank_counts"])]
                 for r in d["sanitizers"]["runs"]]
    contended = sorted(world for world, w in d.get("windows", {}).items() if w["contended"])
    claims["mok.numeric_bitwise_failures"] = sum(r["bitwise_failures"] for r in d["ranks"])
    claims["mok.sanitizer_device_findings"] = sum(x["device_error_or_hazard_count"] or 0
                                                 for r in d["sanitizers"]["runs"] for x in r["sanitizer_rank_counts"])
    # Per-rank verdict counts against the reference; a world whose ranks disagree reports the distinct values.
    for world in sorted({r["world"] for r in d["ranks"]}):
        ranks = [r for r in d["ranks"] if r["world"] == world]
        for name, get in (("configurations", lambda r: r["configurations"]),
                          ("pass", lambda r: r["statuses"].get("PASS", 0)),
                          ("fail_inherited", lambda r: r["statuses"].get("FAIL-inherited", 0)),
                          ("fail_inherited_crash", lambda r: r["statuses"].get("FAIL-inherited-crash", 0))):
            values = sorted({get(r) for r in ranks})
            claims[f"mok.ep{world}_{name}_per_rank"] = values[0] if len(values) == 1 else values
    return section("Numeric parity and sanitizers",
        "Sanity cells plus the full shape × parameter matrix, both images. Numeric and sanitizer runs use unlocked "
        "clocks, as in the reproduction commands. "
        + (f"The EP{'/EP'.join(contended)} numeric run was contended, which affects timing only. " if contended else "")
        + "Outputs are NaN-poisoned before each pass. Inherited reference failures and capacity crashes count as failures.",
        table(["EP", "Logical rank", "Configurations", "PASS", "FAIL-inherited", "FAIL-inherited-crash",
               "Direct generations", "Graph-image generations", "Bitwise failures", "Recompute failures"], rows),
        "Sanitizers cover all EP4 sanity cells and both images. In the count columns, `a: b` means b logical ranks reported a.",
        table(["Tool", "Image", "Status", "Device errors/hazards", "Host API diagnostics"], sanitizer))


def timing_clock(ev, d):
    """Per-cell lock state; reference cells predate clock_locked and read it from their run summary."""
    states = []
    for r in d["cells"]:
        locked = clock_status(r)
        if locked is None:
            path = Path(r.get("records_root") or d.get("records_root", "")) / f"{r['tag']}-time-summary.json"
            locked = clock_status(ev(str(path))) if ev.has(str(path)) else None
        states.append((locked, r.get("lock_mhz") or d.get("lock_mhz")))
    return states


def timing(ev, claims, clocks):
    if not ev.has("timing-summary.json"):
        return not_run("Timing", "timing-summary.json")
    d = ev("timing-summary.json")
    states = timing_clock(ev, d)
    clocks.extend(locked for locked, _ in states)
    rows = []
    for r in d["cells"]:
        values = [("not measured" if r[k] is None else f"{r[k]:.4f}" if "speedup" in k else f"{r[k]:.3f}")
                  for k in ("speedup_vs_one_tu", "linked_ms", "upstream_ms", "speedup_vs_upstream")] \
            if r["admitted"] else ["not admitted"] * 4
        rows.append([r["cell"], r["tokens"], *values, r["status"]])
    admitted = [r for r in d["cells"] if r["admitted"]]
    one_tu = [r["speedup_vs_one_tu"] for r in admitted]
    upstream = [r["speedup_vs_upstream"] for r in admitted if r["speedup_vs_upstream"] is not None]
    claims.update({"mok.timed_kernels": len(one_tu), "mok.r1_min": min(one_tu), "mok.r1_max": max(one_tu)})
    if upstream:
        claims.update({"mok.vs_upstream_min": min(upstream), "mok.vs_upstream_max": max(upstream)})
    locks = sorted({clock_text(locked, mhz) for locked, mhz in states})
    protocols = sorted({r.get("protocol") or "unrecorded" for r in admitted})
    excluded = [[x.get("key") or ", ".join(x["cells"]), x["reason"]]
                for x in d.get("excluded", [])]
    parts = [
        "Clock: " + "; ".join(locks) + ". Admitted timings ran in an uncontended GPU window"
        + (" with a healthy lease." if all(locked for locked, _ in states) else ".")
        + " SW power-cap samples alone are informational; loaded clock deviations and hardware slowdown invalidate a lease.",
        "Eight balanced ABBA/BAAB blocks, four batches per block, 10 fixed repetitions per batch (320 scored steps). "
        "The current pass's modules reload one-TU then linked after block four; context, session ownership, buffers, "
        "parameters, streams and reset addresses are retained. Each image has eight warmup launches before each half "
        "(32 unscored launches total). CUDA events and MAX-of-rank-total reduction are used; own-image preparation is "
        "outside events. Upstream placement uses ordinary paired timing with a minimum of 10 repetitions and no module "
        "reload. Protocols in these receipts: " + ", ".join(f"`{p}`" for p in protocols) + " (`paired-reload-v1` "
        "requires a 12.5-ms minimum batch duration; `paired-reload-v2` requires positive finite elapsed times).",
        "Speedups are reference latency / candidate latency; higher is better. "
        "Parity minimum speedup 1/1.03; control band [1/1.005, 1/0.995]. "
        "The same-source speedup is the geometric mean of the two module-order median speedups; "
        "upstream placement uses the median of per-block speedups.",
        table(["Precision", "Configuration"], sorted({r["cell"].split("-")[2]: r["config"] for r in d["cells"]}.items())),
        table(["Cell", "Tokens/rank", "Speedup vs one-TU", "Linked ms", "Upstream ms", "Speedup vs upstream", "Status"], rows),
        "Linked ms comes from the same-source pair and upstream ms from the placement pair. "
        "Each speedup comes from its own paired samples, so it can differ from the quotient of the displayed medians.",
    ]
    if excluded:
        parts += ["Runs not admitted (repeated in a new window):", table(["Run", "Reason"], excluded)]
    if ev.has("trace-summary.json"):
        traces = ev("trace-summary.json")
        trace_locked = [clock_status(r) for r in traces["runs"]]
        clocks.extend(trace_locked)
        parts += [f"Backward kernel-only traces (Nsight Systems) use a {clock_text(all(trace_locked), traces['lock_mhz'])} "
                  f"and {len(traces['runs']) // max(1, len(traces['means']))} uncontended runs per precision.",
                  table(["Precision", "Tokens/rank", "Port ms", "Upstream ms", "Speedup vs upstream"],
                        [[r["precision"], 2048, f"{r['port_ms']:.3f}", f"{r['upstream_ms']:.3f}",
                          f"{r['speedup_vs_upstream']:.4f}"] for r in traces["means"]]),
                  traces["reduction"]]
    parts.append("Whole-step timing includes surrounding framework launches; backward restores forward context outside "
                 "the interval. Speedup vs upstream places the port against upstream; parity is judged against one-TU only.")
    return section("Timing", *parts)


def placement(ev, claims, clocks):
    if not ev.has("placement-summary.json"):
        return not_run("Placement against upstream", "placement-summary.json")
    d = ev("placement-summary.json")
    states = [clock_status(r) if clock_status(r) is not None else (True if d.get("lock_mhz") else None) for r in d["cells"]]
    clocks.extend(states)
    mhz = {r.get("lock_mhz") or d.get("lock_mhz") for r in d["cells"]}
    rows = [[r["implementation"], r["tokens"],
             f"{r['forward_ms']:.3f}" if r["admitted"] else "not admitted",
             f"{r['backward_ms']:.3f}" if r["admitted"] else "not admitted",
             r["status"], "not sampled" if r["power_cap_samples"] is None else r["power_cap_samples"]]
            for r in d["cells"]]
    telemetry = [[r["implementation"], r["tokens"], t["rank"], f"{t['min_clock_mhz']:.0f} / {t['median_clock_mhz']:.0f}"]
                 for r in d["cells"] for t in r["telemetry"]]
    parts = ["Clock: " + "; ".join(sorted({clock_text(s, m) for s in states for m in mhz})) + ".",
             table(["Implementation", "Tokens/rank", "Forward ms", "Backward ms", "Status", "SW power-cap samples"], rows)]
    if telemetry:
        parts.append(table(["Implementation", "Tokens/rank", "Logical rank", "Minimum / median sampled MHz"], telemetry))
    parts.append(
        "Upstream's own benchmark script with its protocol: 500 warmups, 100 event samples, median rank-max latency. "
        "It keeps upstream's precision-specific tuning: BF16 comm SMs 24/28, MXFP8 36/36. "
        "Power-cap samples are periodic GPU telemetry over the whole workload and cannot be attributed to one kernel. "
        "TorchAO MXFP8 does not support 96 local experts. DeepEP, Transformer Engine and HybridEP are not installed.")
    return section("Placement against upstream", *parts)


def sass(ev, claims):
    if not ev.has("sass.json"):
        return not_run("SASS diagnostic", "sass.json")
    d = ev('sass.json')
    rows = []
    for r in sorted(d['cells'], key=lambda r: r['id']):
        comparisons = [r['comparisons'][k] for k in ('linked_vs_control', 'linked_vs_control_lto')]
        rows.append([r['id'], '/'.join(str(r['images'][k]['instructions']) for k in ('linked', 'control', 'control_lto')),
                     '/'.join(str(c['N']) for c in comparisons),
                     '/'.join(cell(c['normalized_sass_equal']) for c in comparisons),
                     '/'.join(cell(c['seam_structure_clean']) for c in comparisons)])
    return section('SASS diagnostic',
        f"Diagnostic only: {cell(d['diagnostic_only'])}. N: {d['N_definition']}. "
        'Instruction columns: linked/plain/LTO. Comparison columns: linked versus plain/LTO.',
        table(['Cell', 'Instructions', 'Operand deltas N', 'Normalized equal', 'Seam clean'], rows))


def compatibility(ev, claims):
    parts = []
    if ev.has('checked-facts/facts-change.json'):
        f = ev('checked-facts/facts-change.json')
        changes = table(['Owner', 'Payload changed', 'Provider unchanged'],
                        [[o, f['before'][o]['payload_sha256'] != f['after'][o]['payload_sha256'],
                          f['providers_before'][o] == f['providers_after'][o]] for o in sorted(f['before'])])
        parts += [f"Fact relink status {f['status']}: {len(f['cells'])} links, "
                  f"`{f['source_change']['old']}` → `{f['source_change']['new']}`; consumers "
                  + ', '.join(f['cells'][0]['dispatch_fact']['consumers']) + '.', changes]
    else:
        parts.append("Fact relink: not run (`checked-facts/facts-change.json` absent).")
    if ev.has('checked-facts/negative.json'):
        n = ev('checked-facts/negative.json')
        rows = []
        for c in sorted(n['cases'], key=lambda c: c['case']):
            clauses = {v['clause_id'] for v in c['failures']}
            clauses.update(v['clause_id'] for failure in c['failures'] for v in failure['detail'].get('conflicting_set', []))
            rows.append([c['case'], c['change'].replace('the P2 forward cell', 'the forward cell'), c['status'], ', '.join(sorted(clauses))])
        rejected = sum(c['status'] == 'REJECTED' for c in n['cases'])
        claims.update({"mok.negative_cases": len(n['cases']), "mok.negative_rejected": rejected})
        errors = []
        if ev.has('checked-facts/gpu-runs.json'):
            errors = sorted({x for run in ev('checked-facts/gpu-runs.json')['runs'] for x in run.get('rank_errors', [])})
        parts += [f"Negative corpus: {rejected}/{len(n['cases'])} rejected; "
                  f"{sum(c['status'] != 'REJECTED' for c in n['cases'])} unsafe accepts.",
                  table(['Case', 'Incompatibility', 'Verdict', 'Observed clauses including conflicts'], rows),
                  re.sub(r' \(L\d+\)', '', n['declaration_trust']),
                  f"Provider obligation: EXPERT_PADDING={n['provider_case']['EXPERT_PADDING']}, "
                  f"MLP_Mb={n['provider_case']['MLP_Mb']}. " + ('; '.join(errors) + '. ' if errors else '')
                  + 'A successful link does not imply that the provider query or execution succeeds.']
    else:
        parts.append("Negative corpus: not run (`checked-facts/negative.json` absent).")
    out = [section('Fact propagation and negative corpus', *parts)]
    rdc = []
    if ev.has('checked-facts/rdc-contrast.json'):
        r = ev('checked-facts/rdc-contrast.json')
        rdc += [table(['Image', 'Device-link status'], [[x['case'], x['status']] for x in r['images']]),
                r['checked'], r['not_checked'],
                'The stale arm combines 256-wide comm with 512-wide host facts and descriptors. '
                'Generated binding and dispatch glue is shared with the consistent arm. '
                'GPU checks use four ranks, three input generations plus graph replay, NaN-poisoned outputs and a 300-second watchdog.']
    if ev.has('checked-facts/gpu-runs.json'):
        g = ev('checked-facts/gpu-runs.json')
        rdc += [table(['Run', 'Status', 'Rank verdicts', 'Configurations/rank', 'Parent / worker exit'],
                      [[re.sub(r'^p3-|-attempt\d+$', '', x['label']), x['status'], x['rank_statuses'], x['cells_per_rank'],
                        f"{x['returncode']} / {x.get('worker_returncode', 'not recorded')}"] for x in g['runs']]),
                'A watchdog timeout means the image completed no configuration; where it stalled was not localized.']
    out.append(section('Plain RDC contrast and GPU execution', *rdc) if rdc else
               not_run('Plain RDC contrast and GPU execution', 'checked-facts/rdc-contrast.json', 'checked-facts/gpu-runs.json'))
    return out


def skew(ev, claims):
    if not ev.has('checked-facts/skew.json'):
        return not_run('Owner skew', 'checked-facts/skew.json')
    d = ev('checked-facts/skew.json')
    rows, outcomes = [], {}
    for c in sorted(d['cases'], key=lambda c: c['case']):
        changes = '; '.join(x['old'] + ' → ' + x['new'] for x in c.get('source_change', []) if not x['old'].startswith('"smem"')) or 'unmodified control'
        arms = []
        for arm in ('rdc', 'pipelink'):
            gpu = c['arms'][arm].get('gpu')
            outcome = gpu['outcome'] if gpu else 'not run'
            if gpu and any(v == 'nan' for v in gpu.get('max_abs_error', {}).values()):
                outcome += '; outputs unwritten (NaN-poisoned)'
            arms.append(outcome)
        outcomes[c['case']] = (c['owner'], *arms)
        rows.append([c['case'], changes, *arms])
    upgrades = [v for v in outcomes.values() if v[0]]
    claims.update({
        "mok.skew_upgrades": len(upgrades),
        "mok.skew_rdc_hang": sum(v[1].startswith('HANG') for v in upgrades),
        "mok.skew_rdc_unwritten": sum('outputs unwritten' in v[1] for v in upgrades),
        "mok.skew_relinked_pass": sum(v[2] == 'PASS' for v in upgrades)})
    return section('Owner skew',
        f"Status: {d['status']}; GPU execution: {d['gpu_execution']}. " + d['definition'],
        f"EP1 MXFP8; {d['shape']['name']}; {d['shape']['parameter_family']}; "
        'three input generations, graph replay, poisoned outputs, 300-second watchdog. '
        'Five owner upgrades and two unmodified controls. Quantization layout is undeclared, so link checks do not cover it.',
        table(['Case', 'Owner geometry', 'Plain RDC', 'PipeLink'], rows))


def replacement(ev, claims, clocks):
    names = ['owner-swap/result.json', 'owner-swap/source-absence.json', 'owner-swap/include-token-audit.json',
             'owner-swap/matrix-summary.json', 'owner-swap/payload-hashes.json']
    out = []
    if ev.has(*names):
        d = ev(names[0])
        sources = ev(names[1])
        audits = {x['owner']: x for x in ev(names[2])['owners']}
        matrix = ev(names[3])
        hashes = ev(names[4])
        rows = []
        for s in sorted(sources, key=lambda s: s['owner']):
            a = audits[s['owner']]
            rows.append([s['owner'], len(s['packages']), len(s['owner_cu_files']), len(s['foreign_owner_cu_files']),
                         s['package_provider_build'], a['include_audit'], a['token_audit']])
        budgets = {}
        for c in matrix['cells']:
            budgets.setdefault(c['cell'].split('-')[1], set()).add((c['params_total_bytes'], c['params_headroom_bytes'], c['dropped_endpoint_count']))
        claims["mok.swap_linked_cells"] = d['linked_cells']
        out.append(section('Replacement activation: payloads, source separation and audits',
            f"Status {d['status']}; {d['linked_cells']} links; {d['source_absence_owners']} source-separated trees; "
            f"{sum(len(s['packages']) for s in sources)} package/provider pairs. Only changed payload: {d['only_changed_payload']}. "
            f"Host contract unchanged: {cell(d['host_contract_unchanged'])}.",
            table(['Package', 'Payload changed', 'Host contract unchanged'],
                  [[x['package'], x['changed'], x['host_contract_unchanged']] for x in sorted(hashes, key=lambda x: x['package'])]),
            table(['Owner tree', 'Pairs', 'Owner CU files', 'Foreign CU files', 'Build', 'Include audit', 'Token audit'], rows),
            'Public ABI headers, ThunderKittens and shared MXFP8 primitives are explicit dependencies; same-source controls use a separate integration tree.',
            table(['Pass', 'Parameter bytes', 'Headroom bytes', 'Limit bytes', 'Dropped endpoints'],
                  [[p, total, headroom, total + headroom, dropped] for p, values in sorted(budgets.items()) for total, headroom, dropped in sorted(values)])))
    else:
        out.append(not_run('Replacement activation: payloads, source separation and audits', *names))
    if ev.has('owner-swap/gpu-results.json'):
        gpu = ev('owner-swap/gpu-results.json')
        p = gpu['parity']
        clocks.append(clock_status(p))
        claims["mok.swap_checks_per_rank"] = min(min(r['numeric_pass'], r['bitwise_pass']) for r in gpu['numeric'])
        claims["mok.swap_cases_per_rank"] = max(r['cases'] for r in gpu['numeric'])
        claims["mok.swap_r1"] = p['ratio_onetu_over_linked']
        out.append(section('Replacement activation: GPU results',
            f"Status {gpu['status']}. {gpu['scope']}",
            table(['Logical rank', 'Status', 'Cases', 'Numeric passes', 'Bitwise passes'],
                  [[r[k] for k in ('rank', 'status', 'cases', 'numeric_pass', 'bitwise_pass')] for r in gpu['numeric']]),
            f"{p['cell']}, {p['tokens_per_rank']} tokens/rank: speedup vs one-TU **{p['ratio_onetu_over_linked']:.4f}**, "
            f"minimum speedup 1/1.03 ({p['minimum_speedup']:.6f}), status {p['status']}, {clock_text(p['clock_locked'], p['lock_mhz'])}. "
            f"Contended: {cell(p['contended'])}. " + p['placement_note'].replace('the P4 timing gate', 'the timing gate'),
            *swap_sanitizers(ev, claims)))
    else:
        out.append(not_run('Replacement activation: GPU results', 'owner-swap/gpu-results.json'))
    return out


def cost(ev, claims):
    if ev.has('cost/cost.json'):
        d = ev('cost/cost.json')
        a = d['annotations']
    elif ev.has('cost/annotation-counts.json'):
        d, a = None, ev('cost/annotation-counts.json')
    else:
        return not_run('Annotation and build cost', 'cost/annotation-counts.json')
    packages = sorted(a['packages'], key=lambda p: p['package'])
    totals = [sum(p[k] for p in packages) for k in ('device_statements', 'host_statements', 'total_statements')]
    scalar_blocks = sum(p['statements_by_kind'].get('PL_HOST_SCALARS', 0) for p in packages)
    field_builders = sum(s['nested_builder_methods_by_kind'].get('field', 0) for s in a['sources'])
    lines = sum(o['source_lines'] for o in a['owners'].values())
    claims.update({"mok.annotation_statements": totals[2], "mok.owner_source_lines": lines})
    parts = [
        f"Logical annotations: {totals[2]} total, {totals[0]} device, {totals[1]} host. "
        f"{scalar_blocks} scalar-block statements contain {field_builders} field builders. "
        f"Owner source lines: {lines} (including comments/blanks, excluding generated/vendor code).",
        table(['Package', 'Device', 'Host', 'Total'], [[p['package'], p['device_statements'], p['host_statements'], p['total_statements']] for p in packages]),
        'Top-level PL_* declarations and expanded MOK_* wrappers count once; nested builders are not counted again. '
        'Counts cover declarations only; adapting the harness and op bodies took additional work.']
    if d is None:
        parts.append('Build-time measurements: not run (`cost/cost.json` absent).')
        return section('Annotation and build cost', *parts)
    t = d['timings']
    rows = []
    for label, kind, key in [('Repack activation + relink', 'summary', 'repack_relink'), ('Repack activation', 'component_summary', 'repack_relink-{}-pack'),
                             ('Relink one cell', 'component_summary', 'repack_relink-{}-link'), ('Plain one-TU', 'component_summary', 'one_tu-{}-plain'),
                             ('Upstream extension', 'summary', 'upstream')]:
        values = [t[kind][key][s] if kind == 'summary' else t[kind][key.format(s)] for s in ('cold', 'warm')]
        rows.append([label, *(f"{v['median_seconds']:.3f}" for v in values)])
    lto = [sum(t['component_summary'][f'one_tu-{state}-{leg}']['median_seconds']
               for leg in ('lto-compile', 'lto-link')) for state in ('cold', 'warm')]
    rows.append(['LTO one-TU compile + final link', *(f'{v:.3f}' for v in lto)])
    records = {r['label']: r for r in t['records']}
    speedups = [median(
        records[f"r{trial['repetition']}-one_tu-{state}-plain"]['wall_seconds'] / trial['wall_seconds']
        for trial in t['trials'] if trial['leg'] == 'repack_relink' and trial['state'] == state)
        for state in ('cold', 'warm')]
    cpu = {x['field']: x['data'] for x in json.loads(d['environment']['lscpu'])['lscpu']}
    parts[0] = f"Cell {d['cell']}; GPU used: {cell(d['gpu_used'])}. " + parts[0]
    parts += [
        'Build times: medians of three cold/warm pairs; repack+relink uses paired-total medians, '
        'while LTO compile+final link uses the sum of component medians.',
        table(['Build work', 'Cold seconds', 'Warm seconds'], rows),
        f"Repack+relink speedup vs plain one-TU (median per-pair speedup): **{speedups[0]:.2f}× cold / {speedups[1]:.2f}× warm**. "
        'Separate packaging makes the single-cell edit loop slower. An activation edit affects 24 cells. '
        'The upstream build covers a broader workload (72 megakernel instantiations) and is listed for scale; ABI-preserving edits reuse providers.',
        f"Timed commands: {len(t['records'])}; successful: {sum(r['exit_code'] == 0 for r in t['records'])}; "
        'no excluded samples. CPU: ' + cpu['Model name:'] + '; sockets: ' + cpu['Socket(s):'] + '; CUDA 13.3.',
        d['cache_policy'],
        'The timed activation edit exists only to measure compile cost and is not a model variant.']
    return section('Annotation and build cost', *parts)


def inherited(ev, claims):
    if not ev.has('inherited-causes.json'):
        return None
    d = ev('inherited-causes.json')
    p = d['clamp_probe']
    claims['mok.inherited_cause_cases'] = len(d['rows'])
    return section('Inherited upstream causes',
        f"{len(d['rows'])} cases; classification unchanged: {cell(d['classification_unchanged'])}. "
        'Statuses: ' + counts(x['status'] for x in d['rows']) + '.',
        f"Clamp boundary magnitude: {p['inputs'][1]:.1f}.",
        table(['Probe', 'Values'], [('Inputs, clamp(-4,4)', p['inputs']), ('Torch 2.10 derivatives', p['torch_2_10']), ('Torch 2.14 derivatives', p['torch_2_14']), ('Dtypes', p['dtypes'])]),
        'MoK uses inclusive clamp boundaries. FAIL-inherited requires linked/one-TU bitwise equality, '
        'port/upstream bitwise equality on identical inputs and upstream missing the same metric; such cases keep passed=false and the standard tolerances.')


def capacity(ev, claims):
    if not ev.has('capacity.json'):
        return None
    d = ev('capacity.json')
    claims['mok.capacity_padded_rows'] = d['observed_padded_rows']
    claims['mok.capacity_rows'] = d['capacity']
    return section('Own-test and capacity evidence',
        table(['Shape', 'World', 'Parameters', 'SwiGLU', 'Result', 'Padded rows', 'Capacity'],
              [[d['shape'], d['world'], d['parameter_family'], d['swiglu'], d['result'], d['observed_padded_rows'], d['capacity']]]),
        f"Own test: `{d['source']}`. Memcheck device errors: {d['memcheck_corrected_counts']['device_error_or_hazard_count']}. "
        'scheduler::schedule_kernel traps. The port is not run on cells with insufficient capacity, so they have no port verdict.')


# README spans (start, end marker text) whose numbers must all appear in the reference report.
# Markers are content so edits elsewhere do not shift them. Commands, owner-source
# explanations and API/version labels lie outside these spans.
AUDIT_SPANS = [
    ('The source pins', '## Build and run'),
    ('Ratios here', '### Activation-owner replacement'),
    ('**Only the act payload changes.**', '## Upstream findings'),
    ('- **Torch clamp autograd', '## Known limits'),
    ('- **Parameter space is tight.**', 'Continue with the [compiler README]'),
]
NUMBER = re.compile(r'(?<![\w.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?!\w|\.\d)')


def numbers(text):
    text = re.sub(r'\]\([^)]*\)', ']', text)
    text = re.sub(r'```.*?```', '', text, flags=re.S)
    return {m.group().replace(',', '') for m in NUMBER.finditer(text)}


def crosscheck(report):
    """README measurement numbers that the rendered report does not contain."""
    readme = README.read_text()
    claims = set()
    for start, end in AUDIT_SPANS:
        if start in readme:
            claims.update(numbers(readme.split(start, 1)[1].split(end, 1)[0]))
    return sorted(claims - numbers(report), key=lambda x: (float(x), x))


def render(root):
    ev = Evidence(root)
    claims, clocks = {}, []
    body = [baseline(ev, claims), payloads(ev, claims), parity(ev, claims), timing(ev, claims, clocks),
            placement(ev, claims, clocks), sass(ev, claims), *compatibility(ev, claims), skew(ev, claims),
            *replacement(ev, claims, clocks), cost(ev, claims), inherited(ev, claims), capacity(ev, claims)]
    clock_locked = bool(clocks) and all(state is True for state in clocks)
    parts = [BANNER] if clocks and not clock_locked else []
    parts += ['Generated by `campaigns/mok/port/report.py`; do not edit.', '# MoK evidence report',
              'Selected campaign evidence, including recorded failures and scope limits. Ranks are logical and do not '
              'identify physical GPUs. Content digests are omitted; the receipts record them.',
              *(part for part in body if part)]
    return '\n\n'.join(parts) + '\n', claims, clock_locked


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("evidence", type=Path, help="receipt directory in the layout above")
    parser.add_argument("--output", type=Path, help="Markdown path (default: EVIDENCE/REPORT.md)")
    parser.add_argument("--crosscheck", action="store_true",
                        help="require every measurement number in the README result spans to appear in the report")
    args = parser.parse_args()
    report = render(args.evidence)[0]
    if args.crosscheck and (missing := crosscheck(report)):
        raise SystemExit('README numeric tokens missing from the report: ' + ', '.join(missing))
    output = args.output or args.evidence / 'REPORT.md'
    output.write_text(report)
    print(f'{output}: {len(report.encode())} bytes')


if __name__ == '__main__':
    main()
