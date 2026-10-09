"""Render the current MoK public evidence without exporting private JSON records.

Run from any directory: python3 campaigns/mok/port/report.py
Only explicitly selected fields are published; logs, commands, machine identities,
local paths, superseded measurements and raw samples are not copied to Markdown.
"""
from collections import Counter
import json
from pathlib import Path
import re

EVIDENCE = Path(__file__).resolve().parents[1] / "evidence"


def load(name):
    return json.loads((EVIDENCE / name).read_text())


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


def short(digest):
    return digest[:12]


def counts(values):
    return "; ".join(f"{value}: {n}" for value, n in sorted(Counter(values).items(), key=lambda x: str(x[0])))


def section(title, sources, *parts):
    return "\n\n".join([f"## {title}", "Records: " + ", ".join(f"`{s}`" for s in sources) + ".", *parts])


def baseline():
    d = load("p0a-upstream.json")
    tests = []
    for rank in sorted(d["tests"]["ranks"], key=lambda r: r["rank"]):
        totals = [sum(f[k] for f in rank["files"].values()) for k in ("passed", "failed", "errors", "skipped")]
        tests.append([rank["rank"], *totals])
    sanitizer_rows = []
    for run in sorted(d["sanitizers"], key=lambda r: (r["pass"], r["precision"], r["tool"])):
        sanitizer_rows.append([run["pass"], run["precision"], run["tool"],
                               counts(r["device_error_or_hazard_count"] for r in run["ranks"]),
                               counts(r["host_api_error_count"] for r in run["ranks"])])
    return section("P0a upstream baseline", ["p0a-upstream.json"],
        f"MoK `{d['upstream_revision'][:7]}`; ThunderKittens `{d['thunderkittens_revision'][:7]}`; "
        f"EP{d['ep']}; {d['toolchain']['visible_device_count']} Blackwell/sm103 GPUs. "
        f"Torch {d['toolchain']['torch']}; Python {d['toolchain']['python']}; CUDA 13.3. "
        f"Status: **{d['status']}**. Build exit {d['build']['final_attempt']['exit_code']}; "
        f"test exit {d['tests']['run']['exit_code']}.",
        "Baseline patches: " + ", ".join(f"`{Path(p).name}`" for p in d["build"]["patches"]) +
        ". No port device-synchronization patches are applied.",
        table(["Logical rank", "Passed", "Failed", "Errors", "Skipped"], tests),
        "\n".join("- " + message for message in sorted({r['message'] for rank in d['tests']['ranks'] for r in rank['details']})),
        "Sanitizer count columns are `count: number of logical ranks`, not totals. Backward includes forward setup.",
        table(["Pass", "Precision", "Tool", "Device reports", "Host API diagnostics"], sanitizer_rows),
        "Racecheck records the two-CTA tensor allocator and MXFP8 backward gate/up scale-word accesses. "
        "The latter read four-byte words and write two-byte halves across both pipeline stages. "
        "Memcheck's host diagnostics are torch FABRIC probes, not device errors. "
        "Placement measurements are presented separately below, not as a parity gate.")


def payloads():
    d = load("p2-viewcall-build/payload-fanout.json")
    launch = load("p2-viewcall-build/links/mok-fwd-bf16-ep1-c0/link.json")
    rows = []
    for a in sorted(d["artifacts"], key=lambda a: a["stage_id"]):
        rows.append([a['stage_id'], len(a['payloads']), sum(len(p['variants']) for p in a['payloads']),
                     len(a['links']), ", ".join(short(p['sha256']) for p in a['payloads']), short(a['provider']['sha256'])])
    dropped = {}
    for link in d['links']:
        dropped.setdefault(link['id'].split('-')[1], set()).add(len(link['dropped_endpoints']['observed']))
    return section("P2 payload fanout", ["p2-viewcall-build/payload-fanout.json",
                   "p2-viewcall-build/links/mok-fwd-bf16-ep1-c0/link.json"],
        f"{len(d['links'])} compositions: 3 passes × 2 precisions × 2 EP choices × 2 clamp choices. "
        f"Profile `{d['profile']}`; inputs unchanged: {cell(d['inputs_unchanged'])}. "
        f"Four packages per composition (one interpreter and three op owners); {launch['threads']} worker threads, "
        f"cluster {launch['cluster']}.",
        table(["Package", "Physical payloads", "Variants", "Links", "Payload SHA", "Provider SHA"], rows),
        table(["Pass", "Dropped endpoint counts"], [[p, sorted(v)] for p, v in sorted(dropped.items())]),
        table(["Cell", "Link SHA", "Cubin SHA", "Variant indices: interpreter/comm/GEMM/act"],
              [[r['id'], short(r['link_sha256']), short(r['cubin_sha256']),
                '/'.join(str(p['variant']['index']) for p in [r['interpreter'], *r['ops']])]
               for r in sorted(d['links'], key=lambda r: r['id'])]),
        "Source identities: " + ", ".join(f"{k} `{short(v)}`" for k, v in sorted(d['source_digests'].items())) + ".")


def parity():
    d = load("p2-viewcall-summary.json")
    n = load("p2-poison-ep4-numeric-summary.json")
    sanity = [[w['world'], r['rank'], r['status'], r['cells'], r['bitwise_or_exact_checks'], r['bitwise_or_exact_failures']]
              for w in d['correctness'] for r in w['ranks']]
    numeric = [[r[k] for k in ('rank', 'configurations')] + [r['statuses'].get('PASS', 0), r['statuses'].get('FAIL-inherited', 0)] +
               [r[k] for k in ('direct_generations', 'graph_image_generations', 'paired_bitwise_failures', 'recompute_exact_failures')]
               for r in n['ranks']]
    sanitizer = []
    # The forward-BF16 receipt is separate from the aggregate summary.
    extra = load('p2-viewcall-sanitize-fwd-bf16-sanitize-summary.json')
    runs = [(s['tag'].removeprefix('p2-viewcall-sanitize-'), r) for s in d['sanitizers'] for r in s['runs']]
    runs += [('fwd-bf16', r) for r in extra['runs']]
    for name, r in sorted(runs, key=lambda x: (x[0], x[1]['tool'])):
        sanitizer.append([name, r['tool'], r['status'], len(r['sanitizer_rank_counts']),
                          counts(x['device_error_or_hazard_count'] for x in r['sanitizer_rank_counts']),
                          counts(x['host_api_error_count'] for x in r['sanitizer_rank_counts'])])
    return section("P2 parity sanity and EP4", ['p2-viewcall-summary.json', 'p2-poison-ep4-numeric-summary.json',
                   'p2-viewcall-sanitize-fwd-bf16-sanitize-summary.json'],
        d['result'] + ". " + n['output_poisoning'] + ".",
        table(['EP', 'Logical rank', 'Status', 'Cell verdicts', 'Bitwise/exact checks', 'Failures'], sanity),
        table(['Logical rank', 'Configurations', 'PASS', 'FAIL-inherited', 'Direct generations', 'Graph-image generations', 'Bitwise failures', 'Recompute failures'], numeric),
        "Sanitizers: linked images, EP4, clamp off. Count columns are `count: number of logical ranks`. "
        "Sanitizer findings are not waived as inherited numerical errors.",
        table(['Cell family', 'Tool', 'Status', 'Ranks', 'Device errors/hazards', 'Host API diagnostics'], sanitizer))


def timing():
    d = load('p2-viewcall-summary.json')
    identities = load('p2-viewcall-image-identity.json')
    rows = [[r['cell'], r['tokens'], f"{r['linked_over_one_tu']:.4f}", f"{r['linked_ms']:.3f}",
             f"{r['upstream_ms']:.3f}", f"{r['linked_over_upstream']:.4f}"] for r in sorted(d['cells'], key=lambda r: (r['cell'], r['tokens']))]
    configs = {r['cell'].split('-')[2]: r['config'] for r in d['cells']}
    return section('P2 timing and identities', ['p2-viewcall-summary.json', 'p2-viewcall-image-identity.json'],
        f"Clock lease {d['primary_clock']['lock_mhz']} MHz; clock status {d['primary_clock']['status']}; "
        f"parity maximum {d['parity_maximum']}; {len(rows)} measured cells. "
        "Balanced ABBA/BAAB, eight block ratios, CUDA events, MAX-rank reduction. " + d['statistic'],
        table(['Cell', 'Tokens/rank', 'Linked / one-TU', 'Linked ms', 'Upstream ms', 'Linked / upstream'], rows),
        table(['Precision', 'Configuration'], sorted(configs.items())),
        table(['Timed cell', 'Linked SHA', 'One-TU SHA'], [[r['cell'], short(r['images']['linked']['sha256']), short(r['images']['one_tu']['sha256'])]
                                                        for r in sorted(identities['cells'], key=lambda r: r['cell'])]),
        'Upstream extension SHA: `' + short(d['cells'][0]['images']['upstream_extension']) + '`. ' + d['combined_intervention'],
        table(['Backward precision', 'Tokens/rank', 'Selected runs', 'Port ms', 'Upstream ms', 'Ratio'],
              [[r['precision'], r['tokens'], len(r['runs']), f"{r['port_ms']:.3f}", f"{r['upstream_ms']:.3f}", f"{r['ratio']:.4f}"] for r in d['megakernel']]),
        'Whole-step timing includes surrounding framework launches; backward restores forward context outside the interval. '
        'Only the selected uncontended runs are scored; linked/upstream is descriptive, not the parity gate.')


def placement():
    d = load('p2-placement-summary.json')
    rows = []
    # For invalid NCCL8192 measurements, retain only the final disposition, not invalid latencies.
    selected = {}
    for r in d['attempts']:
        key = r['attempt'].rsplit('-attempt', 1)[0]
        selected[key] = r
    for key, r in sorted(selected.items()):
        if r['timing_valid']:
            for t in r['timings']:
                rows.append([key, t['implementation'], f"{t['forward_ms']:.3f}", f"{t['backward_ms']:.3f}", r['timing_valid']])
        else:
            rows.append([key, 'NCCL + PyTorch BF16', 'not valid', 'not valid', r['timing_valid']])
    return section('P2 placement', ['p2-placement-summary.json'],
        table(['Workload', 'Implementation', 'Forward ms', 'Backward ms', 'Timing valid'], rows),
        'Descriptive only. Upstream benchmark protocol: 500 warmups, 100 event samples, median rank-max latency; '
        '1300 MHz requested lease. NCCL at 8192 tokens/rank is clock-invalid due to power-cap clock transients. '
        'TorchAO MXFP8 does not support 96 local experts.',
        'Configuration caveat: the placement record says "' + d['config_difference'] +
        '" Its parity-configuration note disagrees with the selected timing record; the P2 timing table renders that record directly.',
        table(['Optional backend', 'Availability'], sorted(d['optional_backends'].items())))


def sass():
    d = load('p2-sass.json')
    rows = []
    for r in sorted(d['cells'], key=lambda r: r['id']):
        comparisons = [r['comparisons'][k] for k in ('linked_vs_control', 'linked_vs_control_lto')]
        rows.append([r['id'], '/'.join(str(r['images'][k]['instructions']) for k in ('linked', 'control', 'control_lto')),
                     '/'.join(str(c['N']) for c in comparisons),
                     '/'.join(cell(c['normalized_sass_equal']) for c in comparisons),
                     '/'.join(cell(c['seam_structure_clean']) for c in comparisons)])
    return section('P2 SASS diagnostic', ['p2-sass.json'],
        f"Diagnostic only: {cell(d['diagnostic_only'])}. {d['N_definition']}. "
        'Instruction columns: linked/plain/LTO. Comparison columns: linked versus plain/LTO. Not an acceptance gate.',
        table(['Cell', 'Instructions', 'Operand deltas N', 'Normalized equal', 'Seam clean'], rows))


def compatibility():
    f = load('p3/facts-change.json')
    n = load('p3/negative.json')
    g = load('p3/gpu-runs.json')
    r = load('p3/rdc-contrast.json')
    rows = []
    for c in sorted(n['cases'], key=lambda c: c['case']):
        clauses = {v['clause_id'] for v in c['failures']}
        clauses.update(v['clause_id'] for failure in c['failures'] for v in failure['detail'].get('conflicting_set', []))
        rows.append([c['case'], c['change'], c['status'], ', '.join(sorted(clauses))])
    changes = table(['Owner', 'Original payload SHA', 'Geometry-edit payload SHA', 'Provider unchanged'],
                    [[o, short(f['before'][o]['payload_sha256']), short(f['after'][o]['payload_sha256']), f['providers_before'][o] == f['providers_after'][o]] for o in sorted(f['before'])])
    gpu = table(['Run', 'Status', 'Rank verdicts', 'Configurations/rank', 'Parent / worker exit'],
                [[x['label'].removesuffix('-attempt3'), x['status'], x['rank_statuses'], x['cells_per_rank'], f"{x['returncode']} / {x.get('worker_returncode', 'not recorded')}"] for x in g['runs']])
    return [section('P3 facts and negative corpus', ['p3/facts-change.json', 'p3/negative.json', 'p3/gpu-runs.json'],
        f"Fact relink status {f['status']}: {len(f['cells'])} links, "
        f"`{f['source_change']['old']}` → `{f['source_change']['new']}`; consumers "
        + ', '.join(f['cells'][0]['dispatch_fact']['consumers']) + '.', changes,
        f"Negative corpus: {sum(c['status'] == 'REJECTED' for c in n['cases'])}/{len(n['cases'])} rejected; "
        f"{sum(c['status'] != 'REJECTED' for c in n['cases'])} unsafe accepts.",
        table(['Case', 'Incompatibility', 'Verdict', 'Observed clauses including conflicts'], rows),
        n['declaration_trust'],
        f"Provider obligation: EXPERT_PADDING={n['provider_case']['EXPERT_PADDING']}, MLP_Mb={n['provider_case']['MLP_Mb']}. "
        + '; '.join(sorted(set(x for run in g['runs'] for x in run.get('rank_errors', [])))) +
        '. Link admission is not provider-query or execution acceptance.'),
        section('P3 RDC contrast and GPU execution', ['p3/rdc-contrast.json', 'p3/gpu-runs.json'],
            table(['Image', 'Device-link status', 'Image SHA'], [[x['case'], x['status'], short(x['cubin_sha256'])] for x in r['images']]),
            r['checked'], r['not_checked'],
            'The stale arm combines 256-wide comm with 512-wide host facts and descriptors. '
            'Generated binding and dispatch glue is shared with the consistent arm. '
            'GPU checks use four ranks, three input generations plus graph replay, NaN-poisoned outputs and a 300-second watchdog.',
            gpu, 'The stale-image timeout is observed noncompletion, not a localized fault; no completed configuration is inferred.')]


def skew():
    d = load('p3/skew.json')
    rows = []
    for c in sorted(d['cases'], key=lambda c: c['case']):
        changes = '; '.join(x['old'] + ' → ' + x['new'] for x in c.get('source_change', []) if not x['old'].startswith('"smem"')) or 'unmodified control'
        arms = []
        for arm in ('rdc', 'pipelink'):
            a = c['arms'][arm]
            gpu = a['gpu']
            outcome = gpu['outcome']
            if any(v == 'nan' for v in gpu.get('max_abs_error', {}).values()):
                outcome += '; NaN output'
            arms += [outcome, short(a['image_sha256'])]
        rows.append([c['case'], changes, *arms])
    return section('P3 owner skew', ['p3/skew.json'],
        f"Status: {d['status']}; GPU execution: {d['gpu_execution']}. " + d['definition'],
        f"EP1 MXFP8; {d['shape']['name']}; {d['shape']['parameter_family']}; "
        'three input generations, graph replay, poisoned outputs, 300-second watchdog. '
        'Five owner upgrades and two unmodified controls; undeclared quantization layout is outside the checked contract.',
        table(['Case', 'Owner geometry', 'Plain RDC', 'RDC SHA', 'PipeLink', 'PipeLink SHA'], rows))


def replacement():
    d = load('p4/result.json')
    sources = load('p4/source-absence.json')
    audits = {x['owner']: x for x in load('p4/include-token-audit.json')['owners']}
    matrix = load('p4/matrix-summary.json')
    gpu = load('p4/gpu-results.json')
    rows = []
    for s in sorted(sources, key=lambda s: s['owner']):
        a = audits[s['owner']]
        rows.append([s['owner'], len(s['packages']), len(s['owner_cu_files']), len(s['foreign_owner_cu_files']),
                     s['package_provider_build'], a['include_audit'], a['token_audit']])
    budgets = {}
    for c in matrix['cells']:
        budgets.setdefault(c['cell'].split('-')[1], set()).add((c['params_total_bytes'], c['params_headroom_bytes'], c['dropped_endpoint_count']))
    p = gpu['parity']
    return [section('P4 payload source and include audits', ['p4/result.json', 'p4/payload-hashes.json', 'p4/source-absence.json', 'p4/include-token-audit.json', 'p4/matrix-summary.json'],
        f"Status {d['status']}; {d['linked_cells']} links; {d['source_absence_owners']} source-separated trees; "
        f"{sum(len(s['packages']) for s in sources)} package/provider pairs. Only changed payload: {d['only_changed_payload']}. "
        f"Host contract unchanged: {cell(d['host_contract_unchanged'])}.",
        table(['Package', 'Original payload SHA', 'Replacement payload SHA', 'Changed', 'Host contract unchanged'],
              [[x['package'], short(x['before_payload_sha256']), short(x['after_payload_sha256']), x['changed'], x['host_contract_unchanged']] for x in sorted(load('p4/payload-hashes.json'), key=lambda x:x['package'])]),
        table(['Owner tree', 'Pairs', 'Owner CU files', 'Foreign CU files', 'Build', 'Include audit', 'Token audit'], rows),
        'Public ABI headers, ThunderKittens and shared MXFP8 primitives are explicit dependencies; same-source controls use a separate integration tree.',
        table(['Pass', 'Parameter bytes', 'Headroom bytes', 'Limit bytes', 'Dropped endpoints'],
              [[p, total, headroom, total + headroom, dropped] for p, values in sorted(budgets.items()) for total, headroom, dropped in sorted(values)])),
        section('P4 GPU results', ['p4/result.json', 'p4/gpu-results.json'],
            f"Status {gpu['status']}. {gpu['scope']}",
            table(['Logical rank', 'Status', 'Cases', 'Numeric passes', 'Bitwise passes'],
                  [[r[k] for k in ('rank', 'status', 'cases', 'numeric_pass', 'bitwise_pass')] for r in gpu['numeric']]),
            f"{p['cell']}, {p['tokens_per_rank']} tokens/rank: linked/one-TU **{p['ratio_linked_over_onetu']:.4f}**, "
            f"limit {p['maximum_ratio']}, status {p['status']}, {p['lock_mhz']} MHz clock lease. "
            f"Timed files unchanged: {cell(p['timed_files_unchanged'])}. {p['placement_note']}",
            table(['Timed file', 'SHA'], [[k, short(v)] for k, v in sorted(p['timed_files_sha256_after'].items())]),
            'No separate CUDA-graph or sanitizer acceptance is claimed for this owner.')]


def cost():
    d = load('p5/cost.json')
    a, t = d['annotations'], d['timings']
    packages = sorted(a['packages'], key=lambda p:p['package'])
    totals = [sum(p[k] for p in packages) for k in ('device_statements', 'host_statements', 'total_statements')]
    scalar_blocks = sum(p['statements_by_kind'].get('PL_HOST_SCALARS', 0) for p in packages)
    field_builders = sum(s['nested_builder_methods_by_kind'].get('field', 0) for s in a['sources'])
    rows = []
    for label, kind, key in [('Repack activation + relink', 'summary', 'repack_relink'), ('Repack activation', 'component_summary', 'repack_relink-{}-pack'),
                             ('Relink one cell', 'component_summary', 'repack_relink-{}-link'), ('Plain one-TU', 'component_summary', 'one_tu-{}-plain'),
                             ('Upstream extension', 'summary', 'upstream')]:
        values = [t[kind][key][s] if kind == 'summary' else t[kind][key.format(s)] for s in ('cold', 'warm')]
        rows.append([label, *(f"{v['median_seconds']:.3f}" for v in values)])
    lto = [sum(t['component_summary'][f'one_tu-{state}-{leg}']['median_seconds']
               for leg in ('lto-compile', 'lto-link')) for state in ('cold', 'warm')]
    rows.append(['LTO one-TU compile + final link', *(f'{v:.3f}' for v in lto)])
    ratios = [t['summary']['repack_relink'][s]['median_seconds'] / t['component_summary'][f'one_tu-{s}-plain']['median_seconds'] for s in ('cold', 'warm')]
    cpu = {x['field']: x['data'] for x in json.loads(d['environment']['lscpu'])['lscpu']}
    text = section('P5 cost inventory', ['p5/cost.json'],
        f"Cell {d['cell']}; GPU used: {cell(d['gpu_used'])}. "
        f"Logical annotations: {totals[2]} total, {totals[0]} device, {totals[1]} host. "
        f"{scalar_blocks} scalar-block statements contain {field_builders} field builders. "
        f"Owner source lines: {sum(o['source_lines'] for o in a['owners'].values())} (including comments/blanks, excluding generated/vendor code).",
        table(['Package', 'Device', 'Host', 'Total'], [[p['package'], p['device_statements'], p['host_statements'], p['total_statements']] for p in packages]),
        'Top-level PL_* declarations and expanded MOK_* wrappers count once; nested builders are not counted again. '
        'Counts cover declarations, not the entire harness/body adaptation.',
        'Build times: medians of three cold/warm pairs; repack+relink uses paired-total medians, '
        'while LTO compile+final link uses the sum of component medians.',
        table(['Build work', 'Cold seconds', 'Warm seconds'], rows),
        f"Three cold/warm pairs. Repack+relink/plain ratios: **{ratios[0]:.2f} cold / {ratios[1]:.2f} warm**. "
        'No single-cell developer-loop speedup. An activation edit affects 24 cells. '
        'Upstream is a broader 72-instantiation build, not a like-for-like baseline; ABI-preserving edits reuse providers.',
        f"Timed commands: {len(t['records'])}; successful: {sum(r['exit_code'] == 0 for r in t['records'])}; "
        'no excluded samples. CPU: ' + cpu['Model name:'] + '; sockets: ' + cpu['Socket(s):'] + '; CUDA 13.3.',
        d['cache_policy'],
        'The timed activation mutation is a compile-cost probe, not a qualified model variant.')
    return text


def inherited():
    d = load('p2-ep4-inherited-causes.json')
    p = d['clamp_probe']
    return section('Inherited upstream causes', ['p2-ep4-inherited-causes.json'],
        f"{len(d['rows'])} cases; classification unchanged: {cell(d['classification_unchanged'])}. "
        'Statuses: ' + counts(x['status'] for x in d['rows']) + '.',
        f"Clamp boundary magnitude: {p['inputs'][1]:.1f}.",
        table(['Probe', 'Values'], [('Inputs, clamp(-4,4)', p['inputs']), ('Torch 2.10 derivatives', p['torch_2_10']), ('Torch 2.14 derivatives', p['torch_2_14']), ('Dtypes', p['dtypes'])]),
        'MoK uses inclusive clamp boundaries. FAIL-inherited requires linked/one-TU bitwise equality, '
        'port/upstream bitwise equality on identical inputs and upstream missing the same metric; passed=false and tolerances are unchanged.')


def capacity():
    d = load('p2-upstream-own-deepseek-ep1.json')
    return section('Own-test and capacity evidence', ['p2-upstream-own-deepseek-ep1.json'],
        table(['Shape', 'World', 'Parameters', 'SwiGLU', 'Result', 'Padded rows', 'Capacity'],
              [[d['shape'], d['world'], d['parameter_family'], d['swiglu'], d['result'], d['observed_padded_rows'], d['capacity']]]),
        f"Own test: `{d['source']}`. Memcheck device errors: {d['memcheck_corrected_counts']['device_error_or_hazard_count']}. "
        'scheduler::schedule_kernel traps. No port verdict or execution is claimed for the insufficient-capacity cells.')


# Paragraph ranges are identified by content, not mutable line numbers. Commands,
# owner-source explanations and API/version labels are outside the evidence audit.
AUDIT_SPANS = [
    ('The source pins', '## Build and run'),
    ('Ratios here', '### Activation-owner replacement'),
    ('**Only the act payload changes.**', '## Upstream findings'),
    ('- **Torch clamp autograd', '## Known limits'),
    ('- **Parameter space is tight.**', '- **Declared compatibility'),
]
NUMBER = re.compile(r'(?<![\w.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?!\w|\.\d)')


def numbers(text):
    text = re.sub(r'\]\([^)]*\)', ']', text)
    text = re.sub(r'```.*?```', '', text, flags=re.S)
    return {m.group().replace(',', '') for m in NUMBER.finditer(text)}


def crosscheck(report):
    """README measurement numbers that the rendered report does not contain."""
    readme = (EVIDENCE.parent / 'README.md').read_text()
    claims = set()
    for start, end in AUDIT_SPANS:
        claims.update(numbers(readme.split(start, 1)[1].split(end, 1)[0]))
    return sorted(claims - numbers(report), key=lambda x: (float(x), x))


def render():
    cost_text = cost()
    parts = ['Generated by `campaigns/mok/port/report.py`; do not edit.', '# MoK evidence report',
             'Selected campaign evidence, with recorded failures and scope limits. Logical ranks are not physical GPU indices. '
             'Hashes are shortened identities. The machine-readable receipts this report renders are not published.',
             baseline(), payloads(), parity(), timing(), placement(), sass(), *compatibility(), skew(),
             *replacement(), cost_text, inherited(), capacity()]
    report = '\n\n'.join(parts) + '\n'
    missing = crosscheck(report)
    if missing:
        raise ValueError('README numeric tokens need evidence mapping: ' + ', '.join(missing))
    return report


def main():
    report = render()
    (EVIDENCE / 'REPORT.md').write_text(report)
    print(f'campaigns/mok/evidence/REPORT.md: {len(report.encode())} bytes')


if __name__ == '__main__':
    main()
