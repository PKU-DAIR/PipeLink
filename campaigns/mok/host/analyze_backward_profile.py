"""Extract numerical NCU metrics and correlated stall PCs for backward comparisons."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

from run import scrub


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cuda = Path(os.environ["PIPELINK_CUDA_HOME"])
    candidates = list(cuda.glob("nsight-compute-*/extras/python"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one NCU Python API, found {len(candidates)}")
    sys.path.insert(0, str(candidates[0]))
    import ncu_report
    report = ncu_report.load_report(str(args.report))
    actions = []
    for ri in range(report.num_ranges()):
        rng = report.range_by_idx(ri)
        for ai in range(rng.num_actions()):
            action = rng.action_by_idx(ai)
            row = {"range": ri, "action": ai, "name": action.name(), "metrics": {}, "pc_stalls": {}, "pm_sampling": {}, "source_counters": {}}
            for name in action.metric_names():
                metric = action[name]
                try:
                    value = metric.value()
                except Exception:
                    value = None
                # Numerical counters only: device names and host identifiers are
                # not necessary evidence and are deliberately not copied.
                if isinstance(value, (int, float)):
                    row["metrics"][name] = {"value": value, "unit": metric.unit()}
                if name.startswith("pmsampling:") and metric.num_instances():
                    values = []
                    for i in range(metric.num_instances()):
                        try:
                            values.append(metric.as_double(i))
                        except Exception:
                            values.append(metric.as_uint64(i))
                    row["pm_sampling"][name] = values
                if name.startswith("sass__inst_executed") and metric.num_instances() and metric.has_correlation_ids():
                    correlations = metric.correlation_ids()
                    instances = []
                    for i in range(metric.num_instances()):
                        value = metric.value(i)
                        if not value:
                            continue
                        identity = correlations.value(i)
                        entry = {"identity": identity, "value": value}
                        if isinstance(identity, int):
                            info = action.source_info(identity)
                            entry.update(file=info.file_name() if info else None,
                                         line=info.line() if info else None, sass=action.sass_by_pc(identity))
                        instances.append(entry)
                    row["source_counters"][name] = instances
                if not name.startswith("smsp__pcsamp_") or not metric.num_instances() or not metric.has_correlation_ids():
                    continue
                correlations = metric.correlation_ids()
                pcs = []
                for i in range(metric.num_instances()):
                    value = metric.as_uint64(i)
                    if not value:
                        continue
                    pc = correlations.as_uint64(i)
                    info = action.source_info(pc)
                    pcs.append({"pc": pc, "samples": value,
                                "file": info.file_name() if info is not None else None,
                                "line": info.line() if info is not None else None,
                                "sass": action.sass_by_pc(pc)})
                row["pc_stalls"][name] = sorted(pcs, key=lambda x: x["samples"], reverse=True)
            actions.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scrub({"report": str(args.report), "actions": actions}), indent=2) + "\n")
    print(json.dumps({"actions": len(actions), "metric_counts": [len(x["metrics"]) for x in actions],
                      "pc_metric_counts": [len(x["pc_stalls"]) for x in actions]}))


if __name__ == "__main__":
    main()
