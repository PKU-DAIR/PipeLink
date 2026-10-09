"""Collect completed upstream logs without rerunning any GPU work."""

import ast
import json
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from run import sanitizer_counts


HOST = Path(__file__).resolve().parent
ROOT = HOST.parents[2]
LOGS = Path(os.environ["PIPELINK_SCRATCH_ROOT"]) / "mok" / "logs"


def load_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def collect():
    tests_run = load_json(LOGS / "tests/run.json")
    tests = {"run": tests_run, "ranks": []}
    test_log = LOGS / "tests/torchrun.log"
    text = test_log.read_text() if test_log.exists() else ""
    tests["reported_collection_counts"] = [int(n) for n in re.findall(r"collected (\d+) items", text)]
    source_counts = {
        str(path.relative_to(path.parent.parent)): sum(
            isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
            for node in ast.parse(path.read_text()).body
        )
        for path in sorted((ROOT / "third_party/mixture-of-kittens/tests").glob("test_*.py"))
    }
    tests["source_declared_test_functions"] = source_counts
    for rank in range(4):
        path = LOGS / f"tests/tests-rank-{rank}.xml"
        files = {}
        details = []
        if path.exists():
            for case in ET.parse(path).iter("testcase"):
                name = case.attrib.get("classname", "").replace(".", "/") + ".py"
                counts = files.setdefault(name, {"passed": 0, "failed": 0, "errors": 0, "skipped": 0})
                outcome = "passed"
                if case.find("failure") is not None:
                    outcome = "failed"
                elif case.find("error") is not None:
                    outcome = "errors"
                elif case.find("skipped") is not None:
                    outcome = "skipped"
                counts[outcome] += 1
                for result in case:
                    if result.tag in ("failure", "error", "skipped"):
                        message = result.attrib.get("message", "")
                        message = message.replace(str(ROOT) + "/", "").replace(os.environ["PIPELINK_SCRATCH_ROOT"], "PIPELINK_SCRATCH_ROOT")
                        details.append({"test": f"{name}::{case.attrib['name']}", "outcome": result.tag, "message": message})
        else:
            files = {
                name: {"passed": None, "failed": None, "errors": None, "skipped": None, "status": "no_result"}
                for name in source_counts
            }
        tests["ranks"].append({"rank": rank, "junit_report_available": path.exists(), "files": files, "details": details})
    benchmarks = []
    for tokens in (2048, 8192):
        directory = LOGS / f"bench-{tokens}"
        run = load_json(directory / "run.json")
        attempt_directory = directory / f"attempt-{run['attempt']}" if run is not None else directory
        log = attempt_directory / "torchrun.log"
        timings = []
        if log.exists() and run is not None and run["timing_valid"]:
            for precision, fwd, fwd_tflops, bwd, bwd_tflops in re.findall(
                r"(BF16|MXFP8): forward ([0-9.]+) ms, ([0-9.]+) TFLOP/s; backward ([0-9.]+) ms, ([0-9.]+) TFLOP/s",
                log.read_text(),
            ):
                timings.append({"precision": precision, "forward_ms": float(fwd), "backward_ms": float(bwd),
                                "forward_tflops": float(fwd_tflops), "backward_tflops": float(bwd_tflops)})
        locked = bool(run and run.get("clock_status_before") == 0 and run.get("clock_status_after") == 0)
        clocks = {}
        for phase in ("before", "after"):
            path = attempt_directory / f"clocks-{phase}.txt"
            clocks[phase] = [int(value) for value in path.read_text().splitlines()] if path.exists() else None
        attempts = [
            {"attempt": int(path.name.split("-")[-1]),
             "workload": load_json(path / "run.json"), "window": load_json(path / "window.json")}
            for path in sorted(directory.glob("attempt-*"), key=lambda item: int(item.name.split("-")[-1]))
        ]
        benchmarks.append({"tokens_per_rank": tokens, "hidden_size": 7168, "intermediate_size": 3072,
                           "experts": 384, "topk": 6, "ep": 4, "clocks_locked": locked,
                           "informational": not locked, "precisions_requested": ["BF16", "MXFP8"],
                           "observed_sm_mhz_by_logical_rank": clocks,
                           "status": "recorded" if run is not None else "not_run",
                           "timing_valid": bool(run and run["timing_valid"]),
                           "run": run, "attempts": attempts, "timings": timings})
    sanitizers = []
    for tool in ("memcheck", "racecheck", "synccheck"):
        for precision in ("bf16", "mxfp8"):
            for pass_name in ("forward", "backward"):
                directory = LOGS / f"sanitizers/{tool}-{precision}-{pass_name}"
                ranks = []
                for rank in range(4):
                    log = directory / f"rank-{rank}.log"
                    text = log.read_text() if log.exists() else ""
                    ranks.append({"rank": rank, "log_available": log.exists(),
                                  **sanitizer_counts(text, tool)})
                sanitizers.append({"tool": tool, "precision": precision, "pass": pass_name,
                                   "status": "recorded" if (directory / "run.json").exists() else "not_run",
                                   "backward_includes_forward_setup": pass_name == "backward",
                                   "ep": 4, "hidden_size": 256, "intermediate_size": 256,
                                   "tokens_per_rank": 512, "experts": 4, "topk": 1,
                                   "window": load_json(directory / "window.json"),
                                   "completion": load_json(directory / "completion.json"),
                                   "run": load_json(directory / "run.json"), "ranks": ranks})
    data = {
        "schema_version": 1, "date": "2026-10-05", "ep": 4,
        "upstream_revision": "f3a8bd2432457c5a2a2ea13915f0dcd0224581e4",
        "thunderkittens_revision": "1c3920d993404dd49a6d4c7267ea11d583bd5c68",
        "toolchain": load_json(LOGS / "toolchain.json"),
        "topology": load_json(LOGS / "topology.json"),
        "build": {"d10_only_attempt": load_json(LOGS / "build-upstream-d10-failed.json"),
                  "final_attempt": load_json(LOGS / "build-upstream.json"),
                  "patches": ["campaigns/mok/host/patches/mok-f3a8bd2-nvcc-major-only.patch",
                              "campaigns/mok/host/patches/mok-f3a8bd2-qualify-kittens-detail.patch"]},
        "tests": tests, "benchmarks": benchmarks, "sanitizers": sanitizers,
        "raw_logs": "PIPELINK_SCRATCH_ROOT/mok/logs",
    }
    data["variant"] = "upstream with multicast host collectives"
    data["harness_build"] = load_json(LOGS.parent / "harness/logs/build-upstream.json")
    if tests_run is not None and tests_run["exit_code"] != 0:
        data["status"] = "upstream_ep4_tests_failed"
        data["unrun_reason"] = "Stopped after the selected baseline's EP4 tests failed."
        if tests_run["exit_code"] == 124:
            data["status"] = "upstream_ep4_tests_timeout"
            tests["timeout_seconds"] = 7200
            tests["failure"] = {"symptom": "EP4 test run exceeded its timeout; see the retained runtime log."}
    else:
        data["status"] = "recorded" if tests_run is not None else "not_run"
    sanity_directory = LOGS / "sanity-forward"
    sanity = {"run": load_json(sanity_directory / "run.json"), "ranks": [],
              "hidden_size": 256, "intermediate_size": 256, "tokens_per_rank": 512, "topk": 1,
              "selection": "Original forward MXFP8 and BF16 test bodies; shapes() restricted to the existing sanity shape."}
    for rank in range(4):
        path = sanity_directory / f"tests-rank-{rank}.xml"
        cases = []
        if path.exists():
            for case in ET.parse(path).iter("testcase"):
                outcome = next((name for name in ("failure", "error", "skipped")
                                if case.find(name) is not None), "passed")
                cases.append({"test": case.attrib["name"], "outcome": outcome,
                              "seconds": float(case.attrib["time"])})
        sanity["ranks"].append({"rank": rank, "cases": cases})
    data["sanity_forward"] = sanity
    if tests_run is None and sanity["run"] and sanity["run"]["exit_code"] == 0:
        data["status"] = "sanity_forward_passed_full_plan_not_run"
    return data


def main():
    data = collect()
    output = ROOT / "campaigns/mok/evidence/p0a-upstream.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2) + "\n")
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
