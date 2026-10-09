#!/usr/bin/env python3
"""bf16 SM90 attention evidence from sm90-baseline/cells.json.

The default family is W1/W2 at 8k and 32k. Causal cells use selected;
sliding-window cells use window; W3 binds score-softcap and W4 reuses selected
with chunk virtual batches. --cells all also selects supported diagnostic cells.

gate and sanitize check linked/control, plus stock where its slot image supports
the cell. smoke and definitive use the canonical parity Protocol: R1 =
linked/control is scored at >= 0.97; supported stock R3/R2 are diagnostic.
Parity throughput is iterations/s, not useful TFLOP/s or absolute placement.
definitive requires current passing smoke and sanitizer evidence.

absolute --tag TAG [--modes linked,control,stock] writes
results/sm90-absolute-TAG.json using the slot's absolute event timer.
stock-baseline uses that timer for stock on harness_equivalence_cells.
seam-cost defaults to W1-8k,W1-32k,W2a-32k and writes
results/sm90-seam-cost.json: linked production score-identity, flag off, versus
the same-source no-score-seam control, using alternating absolute repetitions.
Identity LTO may remove the branch; this does not measure retained nonidentity
branch cost and is not an R1 parity measurement or descriptor-growth comparison.
Timed runs default to unlocked clocks, recording SM clocks before/after timing.
--clock-mhz N requests a monitored gpulockd lease; violating runs are discarded.
sass writes results/sm90-sass.{json,md} for all seven linked/control pipelines
(the six single-device families plus context-parallel `cp`),
using the shared normalized SASS comparator without GPU access. Differences
are mandatory diagnostics, never an acceptance gate.
Every subcommand supports --dry-run without inputs, GPU access, or output files.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RESULTS = HERE / "results"

SLOT = HERE / "out" / "bin" / "attention-slot-sm90"
SPEC = "d128-sm90-c"
PIPELINES = {"causal": "selected", "sliding_window": "window", "chunked": "selected",
             "phi3_blocksparse": "phi", "msa_selected": "msa"}
SASS_PIPELINES = ("selected", "window", "softcap", "window-softcap", "phi", "msa", "cp")
MODES = ("linked", "control", "stock")
BASE_CELLS = ("W1-8k", "W1-32k", "W2a-8k", "W2a-32k", "W2b-8k", "W2b-32k")
SEAM_CELLS = ("W1-8k", "W1-32k", "W2a-32k")
SEAM_SCOPE = "production-score-identity-flag-off"
SEAM_ORDER = "alternating-candidate-first-reference-first"
SEAM_LIMITATIONS = [
    "Production identity only: LTO may eliminate the transform and its branch.",
    "This identity-only experiment does not measure the retained nonidentity branch or softcap cost.",
    "Both images use the same descriptor layout; this does not measure descriptor-growth cost.",
    "Oracle checks cover boundary rows on all heads, not every output row.",
    "No bitwise equality is required between these two images; ordinary R1 Gate A is unchanged.",
    "Timing alone does not establish inlining, branch retention, register usage, or absence of spills.",
]
CELLS_JSON = HERE / "sm90-baseline" / "cells.json"
CELLS: dict[str, dict[str, Any]] = {}
CELLS_DOCUMENT: dict[str, Any] = {}
HARNESS_EQUIVALENCE_CELLS: list[str] = []


class EvidenceError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_generation() -> str:
    path = HERE / "out" / "build-report.json"
    generation = load_json(path).get("build_generation") if path.is_file() else None
    if not generation:
        raise EvidenceError("missing build generation; run attention/build.py before collecting evidence")
    return generation


def run_command(command: list[str], *, env: dict[str, str] | None = None,
                stdin: str | None = None) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=REPO,
        env=env,
        input=stdin,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "command": command,
        "cwd": str(REPO),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def paths(spec: str, pipeline: str) -> tuple[Path, Path, Path]:
    link_dir = HERE / "out" / "links" / spec / pipeline
    return (
        link_dir / "link.json",
        link_dir / "pipeline.cubin",
        HERE / "out" / "controls" / spec / f"{pipeline}.cubin",
    )


def gate_request(case_id: str, image_id: str, gpu: int) -> dict[str, Any]:
    return {
        "schema_version": "m3-parity-slot-request-v1",
        "action": "warmup",
        "case_id": case_id,
        "image_id": image_id,
        "image_role": "linked",
        "image": {"image_id": image_id},
        "physical_gpu": gpu,
        "cuda_visible_device": 0,
        "case_config": {},
        "protocol": {
            "name": "gate",
            "protocol_variant": "fast-strict-v1",
            "pair_valid_target": 0,
            "attempt_cap": 0,
            "warmups_per_image": 0,
            "slot_seconds": 3.0,
            "leading_discard_seconds": 1.0,
            "steady_seconds": 2.0,
            "clock_sample_hz": 10.0,
            "ratio_floor": 0.97,
            "configured_batch_quantum_ms": 100.0,
            "selection_classifier": "strict-pair-valid-v2",
            "gap_row_allowlist": [],
        },
    }


def sass_audit(cubin: Path, cuda: Path, *, results: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    command = [str(cuda / "bin" / "cuobjdump"), "--dump-sass", str(cubin)]
    record = run_command(command)
    if record["returncode"] != 0:
        raise EvidenceError(f"cuobjdump failed for {cubin}: {record['stderr']}")
    text = record["stdout"]
    relative = cubin.parent.relative_to(HERE / "out" / "links")
    sass_path = (RESULTS if results is None else results) / "sass" / relative / "pipeline.sass"
    sass_path.parent.mkdir(parents=True, exist_ok=True)
    sass_path.write_text(text, encoding="utf-8")
    recorded_path = str(sass_path.relative_to(HERE)) if results is None else str(sass_path.resolve())
    record["stdout_path"] = recorded_path
    record["stdout"] = ""
    inc = "USETMAXREG.TRY_ALLOC.CTAPOOL" in text or "SETMAXNREG.INC" in text
    dec = "USETMAXREG.DEALLOC.CTAPOOL" in text or "SETMAXNREG.DEC" in text
    audit = {
        "image": str(cubin.resolve()),
        "sass_path": recorded_path,
        "setmaxnreg_inc_present": inc,
        "setmaxnreg_dec_present": dec,
    }
    return audit, record


def sanitizer_count(output: str, tool: str) -> int | None:
    pattern = (r"RACECHECK SUMMARY:\s*(\d+) hazards?\b" if tool == "racecheck"
               else r"ERROR SUMMARY:\s*(\d+) errors?[ \t]*$")
    matches = re.findall(pattern, output, re.MULTILINE)
    return int(matches[0]) if len(matches) == 1 else None


def gpu_status_command(gpus: list[int]) -> list[str]:
    return [
        "nvidia-smi", "-i", ",".join(str(gpu) for gpu in gpus),
        "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]


def lock_command(action: str, gpus: list[int]) -> list[str]:
    command = [sys.executable, str(REPO / "tools" / "gpulockd.py"), action,
               "--gpus", ",".join(str(gpu) for gpu in gpus)]
    if action == "start":
        mhz = os.environ.get("PIPELINK_LOCK_SM_MHZ")
        if not mhz:
            raise EvidenceError("PIPELINK_LOCK_SM_MHZ is not set; source tools/env.sh")
        command.extend(["--sm-mhz", mhz])
    return command


def load_cells(path: Path) -> None:
    global CELLS_JSON, CELLS, CELLS_DOCUMENT, HARNESS_EQUIVALENCE_CELLS
    document = load_json(path)
    if document.get("schema") != "pipelink-sm90-cells-v1":
        raise EvidenceError(f"unsupported cell schema: {path}")
    rows = document["cells"]
    CELLS = {row["id"]: row for row in rows}
    if len(CELLS) != len(rows):
        raise EvidenceError(f"duplicate cell IDs: {path}")
    CELLS_JSON = path.resolve()
    CELLS_DOCUMENT = document
    HARNESS_EQUIVALENCE_CELLS = document["harness_equivalence_cells"]


def supported(row: dict[str, Any]) -> bool:
    return (row["dtype"] == "bf16" and row["head_dim"] == 128
            and row["seqlen_q"] == row["seqlen_k"]
            and row["mask"]["kind"] in PIPELINES)

def pipeline_for(row: dict[str, Any]) -> str:
    if row["softcap"] is not None:
        return "window-softcap" if row["mask"]["kind"] == "sliding_window" else "softcap"
    return PIPELINES[row["mask"]["kind"]]


def stock_supported(row: dict[str, Any]) -> bool:
    return row["mask"]["kind"] in ("causal", "sliding_window") and row["softcap"] is None


def available_modes(cell: str, modes: tuple[str, ...] = MODES) -> tuple[str, ...]:
    return tuple(mode for mode in modes if mode != "stock" or stock_supported(CELLS[cell]))

def selection_path(cell: str) -> Path:
    return RESULTS / "sm90-selections" / f"{cell}.i32"


def prepare_selections(work: list[tuple[str, str]]) -> None:
    for cell, pipeline in work:
        if pipeline != "msa":
            continue
        path = selection_path(cell)
        if path.is_file():
            continue
        # Use exactly the same materialization and default directory as public
        # adapters. Its path joins the evidence inputs below.
        command = ["uv", "run", "--project", str(HERE / "sm90-baseline"), "python",
                   str(HERE / "sm90-baseline" / "cells.py"), "select", cell,
                   "--cells-json", str(CELLS_JSON), "--out", str(path)]
        result = run_command(command)
        if result["returncode"] != 0:
            raise EvidenceError(f"MSA selection generation failed: {result}")



def streams() -> tuple[dict[str, Any], str]:
    table = {
        cell: {
            "numeric_case": row["numeric_case"],
            "tensors": [
                {"class_id": f"ns=0x0002/case={row['numeric_case']}/role={role}",
                 "role_id": role, "role_name": name,
                 "shape": [row["batch"], row["seqlen_q"] if name == "Q" else row["seqlen_k"],
                           row["heads_q"] if name == "Q" else row["heads_kv"], row["head_dim"]],
                 "dtype": "BF16", "distribution": "U[-2,2)"}
                for role, name in enumerate(("Q", "K", "V"), 1)
            ],
        }
        for cell, row in CELLS.items() if supported(row)
    }
    digest = hashlib.sha256(json.dumps(table, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return table, digest


def case_config(cell: str, pipeline: str) -> dict[str, Any]:
    table, digest = streams()
    row = CELLS[cell]
    return {
        "stream_map_sha256": digest,
        "stream_classes": [tensor["class_id"] for tensor in table[cell]["tensors"]],
        "inputs": {"shape": [row[key] for key in
                             ("batch", "seqlen_q", "seqlen_k", "heads_q", "heads_kv", "head_dim")],
                   "input_hashes": {}, "numeric_case": row["numeric_case"]},
        "arguments": {"case_id": cell, "causal": True, "mask": row["mask"],
                      "window": row["mask"].get("window", 0), "softcap": row["softcap"],
                      "batch": row["batch"], "heads_q": row["heads_q"], "heads_kv": row["heads_kv"],
                      "head_dim": row["head_dim"], "seqlen_q": row["seqlen_q"], "seqlen_k": row["seqlen_k"],
                      "softmax_scale": row["scale"], "dtype": row["dtype"],
                      "product": "final", "pipeline": pipeline,
                      **({"adaptation": "chunk-virtual-batch"}
                         if row["mask"]["kind"] == "chunked" else {})},
    }


def image(cell: str, pipeline: str, mode: str, timing: str = "parity", *,
          seam_cost: bool = False) -> dict[str, Any]:
    """Image paths and roles; seam descriptions share one command that measures both."""
    if mode == "stock":
        if not stock_supported(CELLS[cell]):
            raise EvidenceError(f"no stock slot image implements {cell}; use the public baselines")
        mask = "local" if CELLS[cell]["mask"]["kind"] == "sliding_window" else "c"
        cubin = HERE / "out" / "stock" / "sm90" / f"fa3-hdim128-bf16-{mask}-cluster1.cubin"
        meta = cubin.with_suffix(".json")
        command = [str(SLOT), "--mode=stock", f"--stock-cubin={cubin}", f"--stock-meta={meta}"]
        inputs = {"slot": SLOT, "stock": cubin, "stock_meta": meta}
        name = "stock-cluster1"
    else:
        link, linked, control = paths(SPEC, pipeline)
        if seam_cost:
            control = control.with_name(f"{pipeline}-no-score-seam.cubin")
        cubin = control if mode == "control" else linked
        command = [str(SLOT), f"--mode={'linked' if seam_cost else mode}", f"--link-json={link}",
                   f"--linked-cubin={linked}", f"--control-cubin={control}",
                   "--product=final", f"--load={'msa' if pipeline == 'msa' else 'block' if pipeline == 'phi' else 'dense'}",
                   f"--mask={'phi' if pipeline == 'phi' else 'window' if pipeline in ('window', 'window-softcap') else 'default'}"]
        inputs = {"slot": SLOT, "link": link, "linked": linked, "control": control}
        name = mode
        if seam_cost:
            command.append("--seam-cost")
            name = "score-identity" if mode == "linked" else "no-score-seam"
    if pipeline == "msa":
        command.append(f"--selection={selection_path(cell)}")
        inputs["selection"] = selection_path(cell)
    command.extend([f"--cells-json={CELLS_JSON}", f"--cell={cell}", f"--timing={timing}"])
    inputs["cells"] = CELLS_JSON
    return {"image_id": f"attention-{cell}-{pipeline}-{name}", "command": command,
            "image_path": str(cubin), "cwd": str(REPO),
            "metadata": {"family": "attention", "spec": SPEC, "pipeline": pipeline,
                         "mode": mode, "timing": timing, "dtype": "bf16",
                         **({"adaptation": "chunk-virtual-batch"}
                            if CELLS[cell]["mask"]["kind"] == "chunked" else {}),
                         **({"seam_cost": True, "measurement_scope": SEAM_SCOPE} if seam_cost else {}),
                         **({"cluster_m": 1} if mode == "stock" else {})},
            "inputs": {role: str(path.resolve()) for role, path in inputs.items()}}


def comparisons(cell: str, pipeline: str, include_r2: bool) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    control, linked = image(cell, pipeline, "control"), image(cell, pipeline, "linked")
    result = {"R1": (control, linked)}
    if stock_supported(CELLS[cell]):
        baseline = image(cell, pipeline, "stock")
        result["R3"] = (baseline, linked)
        if include_r2:
            result["R2"] = (baseline, control)
    return result


def require_inputs(images: list[dict[str, Any]]) -> None:
    for description in images:
        for name in description["inputs"].values():
            if not Path(name).is_file():
                raise EvidenceError(f"missing input: {name} (family artifacts require Phase P3)")


def image_spec(description: dict[str, Any], role: str, ImageSpec: Any) -> Any:
    return ImageSpec.from_dict({
        **{key: value for key, value in description.items() if key != "inputs"},
        "image_id": f"{description['image_id']}-{role}",
        "image_sha256": sha256(Path(description["image_path"])),
        "metadata": {**description["metadata"], "role": role},
    })


def input_paths(images: list[dict[str, Any]]) -> dict[str, str]:
    return {role: path for description in images for role, path in description["inputs"].items()}


def write_report(name: str, records: list[dict[str, Any]], extra: dict[str, Any]) -> dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    # Preserve rows only while their input specification has the same semantics.
    previous = load_json(path) if path.is_file() else {}
    old_rows = (previous.get("cases", []) if
                previous.get("cells_document") == CELLS_DOCUMENT else [])
    merged = {(row["case_id"], row["pipeline"]): row for row in old_rows}
    merged.update({(row["case_id"], row["pipeline"]): row for row in records})
    rows = [merged[key] for key in sorted(merged)]
    table, digest = streams()
    payload = {"schema": "pipelink-attention-sm90-evidence-v2", "stage": name,
               "cells_json": str(CELLS_JSON), "cells_document": CELLS_DOCUMENT,
               "stream_map_sha256": digest, "stream_map": table, **extra,
               "cases": rows, "all_passed": all(row["status"] == "PASS" for row in rows)}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.with_suffix(".md").write_text(markdown_report(name, rows), encoding="utf-8")
    return payload


def markdown_report(name: str, rows: list[dict[str, Any]]) -> str:
    lines = [f"# {name}", "", "R1 = linked/control (scored, >= 0.97); R2 = control/stock and R3 = linked/stock (diagnostic).",
             "Ratios are medians of paired iterations/s ratios; these are not absolute-placement timings.",
             "Attempts/discards count timing pairs; pairs that violate the clock lease are discarded.", "",
             "| Cell | Pipeline | Comparison | Reference iterations/s | Candidate iterations/s | Ratio | Clock mode | Observed MHz | Valid pairs | Attempts | Discards | Verdict |",
             "|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|"]
    for row in rows:
        for label, result in row.get("ratios", {}).items():
            summary = result.get("summary", {})
            lines.append(f"| {row['case_id']} | {row['pipeline']} | {label} | "
                         f"{result['reference_iterations_per_s_median']} | {result['candidate_iterations_per_s_median']} | "
                         f"{result['ratio']} | {row.get('clock', 'locked (legacy)')} | {result['clock_mhz']} | "
                         f"{result['pair_valid_count']} | {summary.get('attempted_pairs', '')} | "
                         f"{summary.get('discarded_pairs', '')} | {result['verdict']} |")
        if "ratios" not in row:
            lines.append(f"| {row['case_id']} | {row['pipeline']} | {name} | | | | {row.get('clock', 'not timed')} | | | | | {row['status']} |")
    return "\n".join(lines) + "\n"


def require_prior(stage: str, work: list[tuple[str, str]]) -> None:
    path = RESULTS / f"sm90-{stage}.json"
    if not path.is_file():
        raise EvidenceError(f"definitive refuses: missing input: {path}")
    prior = load_json(path)
    if prior.get("cells_document") != load_json(CELLS_JSON):
        raise EvidenceError(f"definitive refuses: stale {stage} cell specification; rerun {stage}")
    rows = {(row["case_id"], row["pipeline"]): row for row in prior.get("cases", [])}
    generation = build_generation()
    for cell, pipeline in work:
        row = rows.get((cell, pipeline), {})
        images = check_images(cell, pipeline)
        current = input_paths(list(images.values()))
        if (row.get("status") == "PASS" and row.get("build_generation") == generation
                and row.get("input_paths") == current):
            continue
        # Sanitizers check the executed code (slot incl. its combine kernel, link, cubins), not
        # the length: a PASS on the same roles and paths in this build at another cell of the
        # pipeline covers long cells too slow to race-check. Cell-specific selection data is
        # excluded; an explicit record for this cell always wins.
        code = {role: path for role, path in current.items() if role != "selection"}
        if stage == "sanitize" and not row and any(
                other["status"] == "PASS" and key[1] == pipeline
                and other.get("build_generation") == generation
                and {r: p for r, p in other.get("input_paths", {}).items() if r != "selection"} == code
                for key, other in rows.items()):
            continue
        raise EvidenceError(f"definitive refuses: missing or stale passing {stage} evidence for {cell}/{pipeline}")


def check_images(cell: str, pipeline: str) -> dict[str, dict[str, Any]]:
    return {mode: image(cell, pipeline, mode) for mode in available_modes(cell)}


def read_clocks(gpus: list[int]) -> list[dict[str, Any]]:
    run = run_command([
        "nvidia-smi", "-i", ",".join(map(str, gpus)),
        "--query-gpu=index,uuid,clocks.sm,clocks.max.sm", "--format=csv,noheader,nounits",
    ])
    if run["returncode"] != 0:
        raise EvidenceError(f"cannot observe SM clocks: {run}")
    clocks = []
    for line in run["stdout"].strip().splitlines():
        index, uuid, sm, maximum = (field.strip() for field in line.split(","))
        clocks.append({"gpu": int(index), "gpu_uuid": uuid,
                       "sm_mhz": int(sm), "max_sm_mhz": int(maximum)})
    return clocks


@contextmanager
def timed_run(gpus: list[int], clock_mhz: int | None):
    """Observe clocks without mutation unless an explicit lease was requested."""
    commands: list[dict[str, Any]] = []
    clock = {"clock": "unlocked" if clock_mhz is None else "locked",
             "clock_mhz": clock_mhz, "clock_observation_scope": "timed command"}

    def checked(command: list[str]) -> None:
        run = run_command(command)
        commands.append(run)
        if run["returncode"] != 0:
            raise EvidenceError(f"timed run discarded; fix GPU/clock lock and repeat: {run}")

    def observe() -> list[dict[str, Any]]:
        samples = read_clocks(gpus)
        if clock_mhz is not None and any(abs(row["sm_mhz"] - clock_mhz) > 0.03 * clock_mhz
                                         for row in samples):
            raise EvidenceError(f"timed run discarded; SM clocks outside {clock_mhz} MHz ±3%: {samples}")
        return samples

    if clock_mhz is not None:
        os.environ["PIPELINK_LOCK_SM_MHZ"] = str(clock_mhz)
        checked(gpu_status_command(gpus))
        checked(lock_command("start", gpus))
    try:
        if clock_mhz is not None:
            checked(lock_command("status", gpus))
        clock["clocks_before"] = observe()
        yield clock, commands
        clock["clocks_after"] = observe()
        if clock_mhz is not None:
            checked(lock_command("status", gpus))
    finally:
        if clock_mhz is not None:
            stop = run_command(lock_command("stop", gpus))
            commands.append(stop)
            if stop["returncode"] != 0:
                raise EvidenceError(f"gpulockd stop failed: {stop}")


def parity(stage: str, work: list[tuple[str, str]], gpus: list[int], include_r2: bool,
           clock_mhz: int | None) -> dict[str, Any]:
    # The shared discovery module captures these settings at import time.
    os.environ["PIPELINK_GPUS"] = ",".join(map(str, gpus))
    os.environ["PIPELINK_ALLOW_UNLOCKED"] = "1" if clock_mhz is None else "0"
    os.environ["PIPELINK_LOCK_SM_MHZ"] = "" if clock_mhz is None else str(clock_mhz)
    sys.path.insert(0, str(REPO))
    from evaluator.harness.parity import (
        ImageSpec, PairCapError, Protocol, chronological_primary_pairs, discover_idle_gpus, run_case,
    )

    name = f"sm90-{stage}"
    generation = build_generation()
    plans = {(cell, pipeline): comparisons(cell, pipeline, include_r2) for cell, pipeline in work}
    for plan in plans.values():
        require_inputs([description for pair in plan.values() for description in pair])
    if stage == "definitive":
        require_prior("smoke", work)
        require_prior("sanitize", work)
    root = RESULTS / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    with timed_run(gpus, clock_mhz) as (clock, commands):
        # Explicit unlocked selection needs no lease daemon or discovery qualifier.
        # run_case still applies the canonical paired telemetry validity checks.
        discovery = (discover_idle_gpus(root / "assignment", max_devices=1)
                     if clock_mhz is not None else nullcontext())
        with discovery as lease:
            runs = {f"{cell}.{pipeline}": (cell, pipeline) for cell, pipeline in work}
            assignment = lease.assign(runs) if lease is not None else None
            for run_id, (cell, pipeline) in runs.items():
                gpu = assignment["assignments"][run_id]["physical_index"] if assignment else gpus[0]
                if gpu not in gpus:
                    raise EvidenceError(f"harness assigned unrequested GPU {gpu}")
                config = case_config(cell, pipeline)
                ratios = {}
                for label, (reference, linked) in plans[cell, pipeline].items():
                    archive = root / run_id / label
                    protocol = (Protocol.smoke(archive, config) if stage == "smoke"
                                else Protocol.definitive(archive, config))
                    reference_spec = image_spec(reference, "reference", ImageSpec)
                    linked_spec = image_spec(linked, "linked", ImageSpec)
                    pair_error = None
                    try:
                        result = run_case(cell, reference_spec, linked_spec, gpu, protocol)
                    except PairCapError as error:
                        pair_error = str(error)
                        result = {"pairs": [], "summary": load_json(archive / "summary.json")}
                    summary = result["summary"]
                    selected = chronological_primary_pairs(result["pairs"], protocol.valid_pairs)
                    complete = (pair_error is None and summary.get("status") in ("PASS", "MISS")
                                and summary.get("pair_valid_pairs_collected") == protocol.valid_pairs)
                    scored = label == "R1"
                    rates = {}
                    for role, title in (("reference", "reference"), ("linked", "candidate")):
                        outputs = [pair["outputs"][role] for pair in selected]
                        if any(output["throughput_unit"] != "iterations/s" for output in outputs):
                            raise EvidenceError("sm90 parity slot must report iterations/s")
                        rates[f"{title}_iterations_per_s_median"] = (
                            statistics.median(output["throughput"] for output in outputs) if outputs else None)
                        rates[f"{title}_latency_ms_median"] = (
                            statistics.median(output["latency_ms"] for output in outputs) if outputs else None)
                    ratios[label] = {
                        "status": "PASS" if complete and (not scored or summary["status"] == "PASS") else "FAIL",
                        "diagnostic_only": not scored, "verdict": summary.get("verdict"),
                        "ratio": summary.get("median_ratio_linked_over_reference"),
                        "throughput_unit": "iterations/s", **rates,
                        "clock_mhz": summary.get("run_median_clock_mhz"),
                        "pair_valid_count": summary.get("pair_valid_count", 0),
                        "pairs": result["pairs"], "summary": summary, "pair_cap_error": pair_error,
                        "images": {"reference": reference_spec.archive_record(),
                                   "linked": linked_spec.archive_record()},
                    }
                records.append({
                    "case_id": cell, "spec": SPEC, "pipeline": pipeline,
                    "gpu": gpu, "case_config": config, "stock_cluster": 1,
                    "status": "PASS" if all(r["status"] == "PASS" for r in ratios.values()) else "FAIL",
                    "ratios": ratios,
                    "build_generation": generation,
                    "input_paths": input_paths([item for pair in plans[cell, pipeline].values() for item in pair]),
                })
    for record in records:
        record.update(clock)
    payload = write_report(name, records, {"gpus": gpus, "lock_commands": commands, "timing": "parity", **clock})
    if any(row["status"] != "PASS" for row in records):
        raise EvidenceError(f"one or more {name} cells failed; see results/{name}.json")
    return payload


def slot_result(run: dict[str, Any]) -> dict[str, Any] | None:
    try:
        results = [json.loads(line) for line in run["stdout"].splitlines() if line.startswith("{")]
    except json.JSONDecodeError:
        return None
    return results[0] if len(results) == 1 and isinstance(results[0], dict) else None


def gates_pass(result: dict[str, Any] | None, mode: str, *, seam_cost: bool = False) -> bool:
    correctness = result.get("correctness", {}) if result else {}
    gates = ("gate_b_linked", "gate_b_control", "gate_c_lse")
    if mode != "stock" and not seam_cost:
        gates += ("gate_a_bitwise",)
    return (correctness.get("all_true") is True
            and correctness.get("checked_before_warmup") is True
            and all(correctness.get(key) is True for key in gates))


def absolute_request(cell: str, pipeline: str, description: dict[str, Any], gpu: int) -> dict[str, Any]:
    request = gate_request(cell, description["image_id"], gpu)
    request.update(action="slot", case_config=case_config(cell, pipeline),
                   image_role=description["metadata"]["mode"])
    request["protocol"].update(name="absolute", warmups_per_image=10)
    return request


def absolute(tag: str, work: list[tuple[str, str]], modes: tuple[str, ...], gpus: list[int],
             clock_mhz: int | None) -> dict[str, Any]:
    generation = build_generation()
    plans = {(cell, pipeline): {mode: image(cell, pipeline, mode, "absolute")
                                for mode in available_modes(cell, modes)}
             for cell, pipeline in work}
    for images in plans.values():
        require_inputs(list(images.values()))
    records = []
    with timed_run(gpus, clock_mhz) as (clock, commands):
        for cell, pipeline in work:
            measurements = {}
            for mode, description in plans[cell, pipeline].items():
                request = absolute_request(cell, pipeline, description, gpus[0])
                run = run_command(description["command"],
                                  env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpus[0])},
                                  stdin=json.dumps(request, separators=(",", ":")))
                result = slot_result(run)
                timing = {key: result.get(key) for key in ("absolute_median_ms", "reps", "n", "per_rep_ms")} if result else {}
                reps, n = timing.get("reps"), timing.get("n")
                intervals, latency = timing.get("per_rep_ms"), timing.get("absolute_median_ms")
                passed = (run["returncode"] == 0 and gates_pass(result, mode)
                          and type(reps) is int and reps >= 10 and type(n) is int and n > 0
                          and isinstance(intervals, list) and len(intervals) == reps
                          and all(isinstance(ms, (int, float)) and math.isfinite(ms) and ms >= 20
                                  for ms in intervals)
                          and isinstance(latency, (int, float)) and math.isfinite(latency) and latency > 0
                          and math.isclose(latency, statistics.median(intervals) / n, rel_tol=1e-5))
                measurements[mode] = {"status": "PASS" if passed else "FAIL",
                                      **(timing if passed else {}),
                                      "slot_result": result, "command": run, "image": description}
            records.append({
                "case_id": cell, "spec": SPEC, "pipeline": pipeline, "gpu": gpus[0],
                "case_config": case_config(cell, pipeline),
                "status": "PASS" if all(row["status"] == "PASS" for row in measurements.values()) else "FAIL",
                "measurements": measurements,
                "build_generation": generation,
                "input_paths": input_paths(list(plans[cell, pipeline].values())),
            })
    for record in records:
        record.update(clock)
    payload = {
        "schema": "pipelink-attention-sm90-absolute-v1", "tag": tag,
        "cells_json": str(CELLS_JSON),
        "gpus": gpus, "lock_commands": commands, **clock,
        "timing": {"mode": "absolute", "clock_mhz": clock_mhz, "cuda_graphs": False,
                   "minimum_interval_ms": 20, "minimum_reps": 10,
                   "per_rep_ms_unit": "CUDA event interval ms",
                   "latency": "median(per_rep_ms) / n",
                   "iteration": "full per-call launch sequence, including scheduler resets",
                   "inputs": "device-resident, preallocated, warm L2"},
        "cases": records, "all_passed": all(row["status"] == "PASS" for row in records),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"sm90-absolute-{tag}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not payload["all_passed"]:
        raise EvidenceError(f"one or more absolute measurements failed; see {path}")
    return payload


def seam_images(cell: str, pipeline: str) -> dict[str, dict[str, Any]]:
    return {role: image(cell, pipeline, mode, "absolute", seam_cost=True)
            for role, mode in (("candidate", "linked"), ("reference", "control"))}


def seam_cost(work: list[tuple[str, str]], gpus: list[int], clock_mhz: int | None) -> dict[str, Any]:
    generation = build_generation()
    plans = {(cell, pipeline): seam_images(cell, pipeline) for cell, pipeline in work}
    for images in plans.values():
        require_inputs(list(images.values()))
    records = []
    with timed_run(gpus, clock_mhz) as (clock, commands):
        for cell, pipeline in work:
            images = plans[cell, pipeline]
            request = absolute_request(cell, pipeline, images["candidate"], gpus[0])
            run = run_command(images["candidate"]["command"],
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpus[0])},
                              stdin=json.dumps(request, separators=(",", ":")))
            result = slot_result(run)
            correct = (run["returncode"] == 0 and gates_pass(result, "linked", seam_cost=True)
                       and result.get("correctness", {}).get("gate_a_bitwise") is None
                       and result.get("seam_cost") is True
                       and result.get("measurement_scope") == SEAM_SCOPE
                       and result.get("order") == SEAM_ORDER)
            measurements = {}
            for role, description in images.items():
                timing = result.get("measurements", {}).get(role, {}) if result else {}
                reps, n = timing.get("reps"), timing.get("n")
                intervals, latency = timing.get("per_rep_ms"), timing.get("absolute_median_ms")
                passed = (correct and type(reps) is int and reps >= 10 and reps % 2 == 0
                          and type(n) is int and n > 0
                          and isinstance(intervals, list) and len(intervals) == reps
                          and all(isinstance(ms, (int, float)) and math.isfinite(ms) and ms >= 20
                                  for ms in intervals)
                          and isinstance(latency, (int, float)) and math.isfinite(latency) and latency > 0
                          and math.isclose(latency, statistics.median(intervals) / n, rel_tol=1e-5))
                measurements[role] = {
                    "status": "PASS" if passed else "FAIL", "image": description,
                    **({**timing, "latency_us": latency * 1000} if passed else {}),
                }
            passed = (all(row["status"] == "PASS" for row in measurements.values())
                      and measurements["candidate"].get("reps") == measurements["reference"].get("reps"))
            delta = {}
            if passed:
                candidate = measurements["candidate"]["latency_us"]
                reference = measurements["reference"]["latency_us"]
                delta = {"absolute_delta_us": candidate - reference,
                         "latency_ratio": candidate / reference}
            records.append({
                "case_id": cell, "spec": SPEC, "pipeline": pipeline, "gpu": gpus[0],
                "case_config": case_config(cell, pipeline),
                "status": "PASS" if passed else "FAIL", "measurements": measurements, **delta,
                "slot_result": result, "command": run,
                "build_generation": generation,
                "input_paths": input_paths(list(images.values())),
            })
    for record in records:
        record.update(clock)
    payload = {
        "schema": "pipelink-attention-sm90-seam-cost-v1", "stage": "sm90-seam-cost",
        "measurement_scope": SEAM_SCOPE, "limitations": SEAM_LIMITATIONS,
        "candidate": "linked production score-identity with needs_score_mod clear",
        "reference": "same-source no-score-seam control with identical launch/params/layout",
        "delta_definition": "candidate latency minus reference latency",
        "latency_ratio_definition": "candidate latency / reference latency",
        "cells_json": str(CELLS_JSON),
        "gpus": gpus, "lock_commands": commands, **clock,
        "timing": {"mode": "absolute", "clock_mhz": clock_mhz, "cuda_graphs": False,
                   "minimum_interval_ms": 20, "minimum_reps": 10, "warmups_per_image": 10,
                   "order": SEAM_ORDER, "latency": "median(per_rep_ms) / n",
                   "iteration": "full per-call launch sequence, including scheduler resets",
                   "inputs": "device-resident, preallocated, warm L2",
                   "clock_violations": "discard locked runs; unlocked clocks are observations only"},
        "cases": records, "all_passed": all(row["status"] == "PASS" for row in records),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "sm90-seam-cost.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not payload["all_passed"]:
        raise EvidenceError(f"one or more seam-cost measurements failed; see {path}")
    return payload


def checks(stage: str, work: list[tuple[str, str]], gpus: list[int]) -> dict[str, Any]:
    generation = build_generation()
    plans = {(cell, pipeline): check_images(cell, pipeline) for cell, pipeline in work}
    for images in plans.values():
        require_inputs(list(images.values()))
    cuda_text = os.environ.get("PIPELINK_CUDA_HOME")
    if not cuda_text:
        raise EvidenceError("PIPELINK_CUDA_HOME is not set; source tools/env.sh")
    cuda = Path(cuda_text).resolve()
    executable = cuda / "bin" / ("compute-sanitizer" if stage == "sanitize" else "cuobjdump")
    if not executable.is_file():
        raise EvidenceError(f"missing input: {executable}")
    precheck = run_command(gpu_status_command(gpus))
    if precheck["returncode"] != 0:
        raise EvidenceError(f"GPU precheck failed: {precheck}")
    records = []
    for index, (cell, pipeline) in enumerate(work):
        gpu = gpus[index % len(gpus)]
        images = plans[cell, pipeline]
        outcomes = {}
        for mode, description in images.items():
            for tool in (("racecheck", "synccheck") if stage == "sanitize" else ("gate",)):
                request = gate_request(cell, description["image_id"], gpu)
                request.update(case_config=case_config(cell, pipeline), image_role=mode)
                if stage == "sanitize":
                    request["action"] = "sanitize"
                command = description["command"]
                if stage == "sanitize":
                    command = [str(executable), "--tool", tool, "--error-exitcode", "99", *command]
                run = run_command(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                                  stdin=json.dumps(request, separators=(",", ":")))
                result = slot_result(run)
                passed = run["returncode"] == 0 and isinstance(result, dict)
                outcome = {"command": run, "slot_result": result}
                if stage == "sanitize":
                    count = sanitizer_count(run["stdout"] + "\n" + run["stderr"], tool)
                    passed = (passed and count == 0 and result.get("action") == "sanitize"
                              and result.get("images_executed") == [mode])
                    outcome["hazards" if tool == "racecheck" else "errors"] = count
                else:
                    passed = passed and gates_pass(result, mode)
                outcomes[f"{mode}/{tool}"] = {**outcome, "passed": passed}
        if stage == "gate":
            audit, command = sass_audit(Path(images["linked"]["image_path"]), cuda, results=RESULTS)
            outcomes["linked/sass"] = {"audit": audit, "command": command,
                                       "passed": audit["setmaxnreg_inc_present"] and audit["setmaxnreg_dec_present"]}
        records.append({"case_id": cell, "spec": SPEC, "pipeline": pipeline,
                        "gpu": gpu, "checks": outcomes, "build_generation": generation,
                        "input_paths": input_paths(list(images.values())),
                        "status": "PASS" if all(check["passed"] for check in outcomes.values()) else "FAIL"})
    payload = write_report(f"sm90-{stage}", records, {"gpus": gpus, "gpu_precheck": precheck})
    if any(row["status"] != "PASS" for row in records):
        raise EvidenceError(f"one or more {stage} cells failed; see results/sm90-{stage}.json")
    return payload


def sass() -> dict[str, Any]:
    sys.path.insert(0, str(REPO))
    from evaluator.gates import comparison, normalization

    cuda_text = os.environ.get("PIPELINK_CUDA_HOME")
    if not cuda_text:
        raise EvidenceError("PIPELINK_CUDA_HOME is not set; source tools/env.sh")
    cuda = Path(cuda_text).resolve()

    def command(tool: str, *arguments: str) -> dict[str, Any]:
        result = run_command([str(cuda / "bin" / tool), *arguments])
        if result["returncode"] != 0:
            raise EvidenceError(f"{tool} failed: {result['stderr']}")
        return result

    versions = {tool: command(tool, "--version") for tool in ("nvcc", "cuobjdump")}
    records = []
    for pipeline in SASS_PIPELINES:
        _, linked, control = paths(SPEC, pipeline)
        record: dict[str, Any] = {
            "spec": SPEC, "pipeline": pipeline, "status": "error",
            "images": {}, "resource_usage": {}, "commands": [],
        }
        try:
            normalized = {}
            for role, cubin in (("control", control), ("linked", linked)):
                record["images"][role] = {"path": str(cubin)}
                dump = command("cuobjdump", "-sass", str(cubin))
                sections = normalization.sections(dump["stdout"])
                if not sections or not any(body for _, body in sections):
                    raise EvidenceError(f"no decoded instructions: {cubin}")
                normalized[role] = [(name, normalization.canonicalize(body)) for name, body in sections]
                dump["stdout"] = ""
                resources = command("cuobjdump", "--dump-resource-usage", str(cubin))
                usage = {}
                for section in resources["stdout"].split(" Function ")[1:]:
                    name, body = section.split(":\n", 1)
                    usage[name.strip()] = {
                        key: int(value) for key, value in re.findall(r"\b(REG|STACK|LOCAL):(\d+)", body)
                    }
                if not usage or any(set(row) != {"REG", "STACK", "LOCAL"} for row in usage.values()):
                    raise EvidenceError(f"incomplete REG/STACK/LOCAL resource usage: {cubin}")
                record["resource_usage"][role] = usage
                record["commands"].extend((dump, resources))
            compared = comparison.compare_streams(normalized["control"], normalized["linked"])
            deltas = compared.pop("deltas")
            record.update(compared)
            record.update({
                "status": "complete",
                "normalized_sass_equal": normalized["control"] == normalized["linked"],
                "N": len(deltas), "delta_count_total": len(deltas),
                "deltas": deltas[:200], "deltas_truncated": len(deltas) > 200,
                "delta_classes": [name for name, equal in compared["checks"].items()
                                  if not equal and name != "no_local_traffic_or_calls"],
                "operand_delta_classes": {
                    "integer_operand_encoding": sum(delta["integer_operand_encoding"] for delta in deltas),
                    "non_integer_operand": sum(not delta["integer_operand_encoding"] for delta in deltas),
                },
            })
        except (OSError, ValueError, EvidenceError) as exc:
            record["error"] = str(exc)
        records.append(record)
    payload = {
        "schema": "pipelink-attention-sm90-sass-diagnostic-v1", "diagnostic_only": True,
        "function_selection": "all device functions in cuobjdump order",
        "N_definition": "operand-delta count; structural differences are separate checks, so N=0 alone is not equality",
        "delta_limit": 200, "tool_versions": versions,
        "comparator_sources": [str(Path(module.__file__).relative_to(REPO))
                               for module in (comparison, normalization)],
        "pipelines": records,
    }
    lines = [
        "# SM90 attention SASS diagnostic", "",
        "Diagnostic only: differences never admit or reject an image. All device functions",
        "are normalized in dump order with evaluator/gates. Counts and resources below",
        "are control / linked; resource tuples are REG/STACK/LOCAL (stack/local in bytes).",
        "N counts operand deltas, not distinct instructions; N=0 alone does not imply equality.",
        "JSON retains the first 200 deltas, full delta-class counts, checks and tool output.", "",
        "| Pipeline | Instructions C / L | Equal | N | Resources C / L | Delta classes |",
        "|---|---:|:---:|---:|---|---|",
    ]
    for row in records:
        if row["status"] != "complete":
            lines.append(f"| {row['pipeline']} | — | — | — | — | error: {row['error']} |")
            continue
        counts = row["instruction_counts"]
        resources = ["; ".join(f"{name}: " + "/".join(str(usage[key]) for key in ("REG", "STACK", "LOCAL"))
                               for name, usage in row["resource_usage"][role].items())
                     for role in ("control", "linked")]
        classes = row["delta_classes"] + [
            f"{name}={count}" for name, count in row["operand_delta_classes"].items() if count
        ]
        lines.append(f"| {row['pipeline']} | {counts['control']} / {counts['linked']} | "
                     f"{row['normalized_sass_equal']} | {row['N']} | {' / '.join(resources)} | "
                     f"{', '.join(classes) or 'none'} |")
    lines.extend(["", "Delta classes describe ordered-stream differences, not proven seam attribution.",
                  "Numeric, sanitizer and performance evidence remain separate acceptance criteria.",
                  "", "## Tool versions", ""])
    for tool, version in versions.items():
        lines.extend([f"### {tool}", "", "```text", version["stdout"].strip(), "```", ""])
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "sm90-sass.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (RESULTS / "sm90-sass.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    global RESULTS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    sass_parser = subparsers.add_parser("sass", help="GPU-free linked/control SASS diagnostic")
    sass_parser.add_argument("--dry-run", action="store_true", help="print image paths without inputs or outputs")
    sass_parser.add_argument("--results", type=Path, default=RESULTS, help="output directory")
    for name in ("stock-baseline", "gate", "sanitize", "smoke", "definitive", "absolute", "seam-cost"):
        command = subparsers.add_parser(name)
        command.add_argument("--gpus", default="0", help="physical GPU IDs (timed commands require one)")
        command.add_argument("--results", type=Path, default=RESULTS, help="output and prerequisite evidence directory")
        if name not in ("gate", "sanitize"):
            command.add_argument("--clock-mhz", type=int, help="request a gpulockd lease at N MHz; default: unlocked")
        command.add_argument("--cells-json", type=Path, default=CELLS_JSON, help="shared pipelink-sm90-cells-v1 file")
        command.add_argument("--dry-run", action="store_true", help="print commands/configs without inputs or GPUs")
        if name != "stock-baseline":
            command.add_argument("--cells", help="comma-separated IDs; all = supported cells; "
                                 "default = W1-8k,W1-32k,W2a-32k for seam-cost, six W1/W2 cells otherwise")
        if name in ("smoke", "definitive"):
            command.add_argument("--include-r2", action="store_true", help="also measure diagnostic control/stock")
        if name in ("absolute", "stock-baseline"):
            command.add_argument("--tag", required=name == "absolute",
                                 default="stock-baseline" if name == "stock-baseline" else None)
        if name == "absolute":
            command.add_argument("--modes", default=",".join(MODES), help="comma-separated linked,control,stock")
    args = parser.parse_args()
    RESULTS = args.results.resolve()
    if args.command == "sass":
        if args.dry_run:
            payload = {"stage": "sass", "dry_run": True, "diagnostic_only": True,
                       "pipelines": [{"pipeline": pipeline, "linked": str(paths(SPEC, pipeline)[1]),
                                      "control": str(paths(SPEC, pipeline)[2])} for pipeline in SASS_PIPELINES]}
        else:
            result = sass()
            payload = {"stage": "sass", "diagnostic_only": True,
                       "completed": sum(row["status"] == "complete" for row in result["pipelines"]),
                       "equal": sum(row.get("normalized_sass_equal", False) for row in result["pipelines"]),
                       "pipelines": len(result["pipelines"]),
                       "outputs": [str(RESULTS / f"sm90-sass.{suffix}") for suffix in ("json", "md")]}
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    load_cells(args.cells_json)
    stage = args.command
    is_absolute = stage in ("absolute", "stock-baseline")
    modes = tuple(args.modes.split(",")) if stage == "absolute" else ("stock",)
    if is_absolute:
        if not modes or len(set(modes)) != len(modes) or set(modes) - set(MODES):
            parser.error("--modes must select unique linked,control,stock entries")
        if not args.tag or args.tag in (".", "..") or any(not (char.isalnum() or char in "-_.") for char in args.tag):
            parser.error("--tag must be a nonempty filename label containing only letters, digits, dash, underscore or dot")
    if stage == "stock-baseline":
        cells = HARNESS_EQUIVALENCE_CELLS
    elif args.cells == "all":
        cells = [cell for cell, row in CELLS.items() if supported(row)]
    else:
        defaults = SEAM_CELLS if stage == "seam-cost" else BASE_CELLS
        cells = args.cells.split(",") if args.cells is not None else list(defaults)
    if not cells or len(set(cells)) != len(cells) or set(cells) - CELLS.keys():
        parser.error("cell selection must contain unique IDs from --cells-json")
    unsupported = [cell for cell in cells if not supported(CELLS[cell])]
    if unsupported:
        parser.error(f"unsupported bf16 d128 prefill cells: {unsupported}")
    if stage == "seam-cost" and any(not stock_supported(CELLS[cell]) for cell in cells):
        parser.error("seam-cost requires identity causal/sliding_window cells")
    if is_absolute and any(not available_modes(cell, modes) for cell in cells):
        parser.error("no requested image implements one or more cells; use linked,control")
    work = [(cell, pipeline_for(CELLS[cell])) for cell in cells]
    try:
        gpus = [int(item) for item in args.gpus.split(",")]
    except ValueError:
        parser.error("--gpus must be a comma-separated list of physical GPU IDs")
    if len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        parser.error("--gpus must select unique non-negative physical GPU IDs")
    clock_mhz = getattr(args, "clock_mhz", None)
    if stage not in ("gate", "sanitize") and len(gpus) != 1:
        parser.error("timed sm90 evidence requires exactly one physical GPU")
    if clock_mhz is not None and clock_mhz <= 0:
        parser.error("--clock-mhz must be positive")
    if args.dry_run:
        plans = []
        for cell, pipeline in work:
            if stage == "seam-cost":
                images = seam_images(cell, pipeline)
                requests = {"pair": absolute_request(cell, pipeline, images["candidate"], gpus[0])}
            elif is_absolute:
                images = {mode: image(cell, pipeline, mode, "absolute")
                          for mode in available_modes(cell, modes)}
                requests = {mode: absolute_request(cell, pipeline, description, gpus[0])
                            for mode, description in images.items()}
            elif stage in ("gate", "sanitize"):
                images = check_images(cell, pipeline)
                requests = {}
            else:
                images = {label: {"reference": pair[0], "linked": pair[1]}
                          for label, pair in comparisons(cell, pipeline, args.include_r2).items()}
                requests = {}
            plans.append({"case_id": cell, "pipeline": pipeline, "case_config": case_config(cell, pipeline),
                          "images": images, **({"requests": requests} if requests else {})})
        print(json.dumps({"stage": stage, "dry_run": True, "cells_json": str(CELLS_JSON),
                          "output": str(RESULTS / f"sm90-absolute-{args.tag}.json") if is_absolute else
                                    str(RESULTS / f"sm90-{stage}.json"),
                          "cases": plans,
                          **({"clock": "unlocked" if clock_mhz is None else "locked", "clock_mhz": clock_mhz}
                             if stage not in ("gate", "sanitize") else {}),
                          **({"measurement_scope": SEAM_SCOPE, "limitations": SEAM_LIMITATIONS,
                              "order": SEAM_ORDER,
                              "minimum_interval_ms": 20, "minimum_reps": 10}
                             if stage == "seam-cost" else {})}, indent=2, sort_keys=True))
        return 0
    prepare_selections(work)
    if stage == "seam-cost":
        payload = seam_cost(work, gpus, clock_mhz)
    elif is_absolute:
        payload = absolute(args.tag, work, modes, gpus, clock_mhz)
    elif stage in ("gate", "sanitize"):
        payload = checks(stage, work, gpus)
    else:
        payload = parity(stage, work, gpus, args.include_r2, clock_mhz)
    print(json.dumps({"stage": stage, "all_passed": payload["all_passed"],
                      **({"clock": payload["clock"]} if "clock" in payload else {})}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EvidenceError, OSError) as error:
        print(f"sm90 attention evidence: {error}", file=sys.stderr)
        raise SystemExit(2)
