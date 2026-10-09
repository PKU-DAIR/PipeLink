"""Run each measurement in an owned shared-machine GPU window."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess


HOST = Path(__file__).resolve().parent
ROOT = HOST.parents[2]
LOGS = Path(os.environ["PIPELINK_SCRATCH_ROOT"]) / "mok/logs"
WINDOW_SUMMARY = re.compile(r"gpu_window: waited=(\d+)s ran=(\d+)s contended=([01]) rc=(\d+)")


def run_window(command, directory):
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / "window.log"
    with log.open("w") as output:
        result = subprocess.run([str(ROOT / "tools/gpu_window.sh"), *command],
                                cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
    text = log.read_text()
    summaries = WINDOW_SUMMARY.findall(text)
    if not summaries:
        raise RuntimeError(f"GPU window did not produce its completion summary; inspect {log}")
    waited, ran, contended, reported_exit = map(int, summaries[-1])
    if result.returncode != reported_exit:
        raise RuntimeError("GPU window summary and process exit codes disagree")
    report = {"waited_seconds": waited, "ran_seconds": ran,
              "contended": bool(contended), "exit_code": result.returncode}
    (directory / "window.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{directory.relative_to(LOGS)}: {report}", flush=True)
    return report


def benchmark_matrix():
    for tokens in (2048, 8192):
        directory = LOGS / f"bench-{tokens}"
        attempt = 0
        while True:
            attempt += 1
            attempt_directory = directory / f"attempt-{attempt}"
            if attempt_directory.exists():
                raise RuntimeError(f"Preserve the previous run directory before repeating: {attempt_directory}")
            window = run_window([str(HOST / "benchmark_upstream.sh"), str(tokens), str(attempt_directory)],
                                attempt_directory)
            metadata = attempt_directory / "run.json"
            report = json.loads(metadata.read_text()) if metadata.exists() else {}
            report["workload_exit_code"] = report.get("exit_code")
            report.update(window)
            report["attempt"] = attempt
            report["timing_valid"] = window["exit_code"] == 0 and not window["contended"]
            (directory / "run.json").write_text(json.dumps(report, indent=2) + "\n")
            if window["exit_code"] != 0:
                raise SystemExit(window["exit_code"])
            if window["contended"]:
                print(f"bench-{tokens}: timing invalidated by contention; repeating in a new window", flush=True)
                continue
            for line in (attempt_directory / "torchrun.log").read_text().splitlines():
                if re.match(r"(?:BF16|MXFP8): forward ", line):
                    print(line, flush=True)
            break


def sanitizer_matrix():
    for tool in ("memcheck", "racecheck", "synccheck"):
        for precision in ("bf16", "mxfp8"):
            for pass_name in ("forward", "backward"):
                directory = LOGS / f"sanitizers/{tool}-{precision}-{pass_name}"
                window = run_window([str(HOST / "sanitize_upstream.sh"), tool, precision, pass_name, str(directory)], directory)
                workload_log = directory / "torchrun.log"
                text = workload_log.read_text() if workload_log.exists() else ""
                completed = {
                    json.loads(match.group())["rank"]
                    for match in re.finditer(r'\{[^{}\n]*"completed"\s*:\s*true[^{}\n]*\}', text)
                }
                summary_ranks = []
                for rank in range(4):
                    rank_log = directory / f"rank-{rank}.log"
                    if rank_log.exists() and re.search(r"(?:ERROR|RACECHECK) SUMMARY: \d+", rank_log.read_text()):
                        summary_ranks.append(rank)
                complete = completed == set(range(4)) and len(summary_ranks) == 4
                report = {"completed_ranks": sorted(completed), "summary_ranks": summary_ranks,
                          "complete": complete, "exit_code": window["exit_code"]}
                (directory / "completion.json").write_text(json.dumps(report, indent=2) + "\n")
                if window["exit_code"] in (124, 137) or not complete:
                    raise SystemExit(window["exit_code"] or 1)
                print(f"{directory.relative_to(LOGS)}: all ranks completed and produced sanitizer counts", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("all", "benchmarks", "sanitizers"), default="all")
    args = parser.parse_args()
    if args.phase in ("all", "benchmarks"):
        benchmark_matrix()
    if args.phase in ("all", "sanitizers"):
        sanitizer_matrix()


if __name__ == "__main__":
    main()
