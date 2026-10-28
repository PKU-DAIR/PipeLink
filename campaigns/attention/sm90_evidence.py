#!/usr/bin/env python3
"""bf16 SM90 attention evidence for the cells in sm90-baseline/cells.json.

By default subcommands cover W1/W2 at 8k and 32k, definitive covers the 17
published cells and seam-cost its own three cells; --cells all selects every
supported cell, including the diagnostic ones.
Causal cells use the `selected` pipeline and sliding-window cells use `window`.
W3 binds the score softcap, and W4 reuses `selected` with chunked virtual batches.

Subcommands:
  gate         numerical gates for linked and control, plus stock where the stock
               slot image supports the cell.
  sanitize     memcheck, racecheck and synccheck on the paired reload path.
  smoke        optional diagnostic using the calibrated slot timer.
  definitive   paired-reload-v2 timing: eight balanced blocks of four batches,
               exactly ten complete calls per batch, with a module reload
               halfway. R1 must reach 0.97; R3 is reported for context. Needs
               passing gate and sanitize records from the current build.
  absolute     --tag TAG [--modes linked,control,stock] writes
               results/sm90-absolute-TAG.json with the slot's absolute event timer.
  stock-baseline  the same timer for stock on harness_equivalence_cells.
  seam-cost    times the linked production image with the score transform set to
               identity (flag off) against the same-source control without a score
               seam, alternating absolute repetitions; defaults to
               W1-8k,W1-32k,W2a-32k and writes results/sm90-seam-cost.json. LTO
               may remove the identity branch, so this bounds only that case: it
               does not measure a retained non-identity branch, R1 parity or the
               cost of a larger descriptor.
  sass         GPU-free SASS comparison of linked and control for all seven
               pipelines (the six single-device families plus context-parallel
               `cp`), written to results/sm90-sass.{json,md}. Differences are
               reported as diagnostics only.
  render       CPU-only: rewrite the gate, sanitize, definitive and SASS Markdown
               reports from the JSON records in --results.

Timed runs default to --clock-mhz $PIPELINK_LOCK_SM_MHZ; when that variable is
unset, empty or 0 (or --clock-mhz 0 is given), clocks are unlocked and only
observed before and after timing. A positive --clock-mhz N requests a monitored
gpulockd lease and discards runs that leave it. Timing receipts record
`clock_locked`. Every subcommand accepts --dry-run, which needs no inputs or GPU and
writes no files.
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

OUT = HERE / "out"
SLOT = OUT / "bin" / "attention-slot-sm90"
SPEC = "d128-sm90-c"
PIPELINES = {"causal": "selected", "sliding_window": "window", "chunked": "selected",
             "phi3_blocksparse": "phi", "msa_selected": "msa"}
SASS_PIPELINES = ("selected", "window", "softcap", "window-softcap", "phi", "msa", "cp")
MODES = ("linked", "control", "stock")
BASE_CELLS = ("W1-8k", "W1-32k", "W2a-8k", "W2a-32k", "W2b-8k", "W2b-32k")
PRIMARY_CELLS = (*BASE_CELLS, "W3a-8k", "W3a-32k", "W3b-8k",
                 "W4-32k", "W4-128k", "W5-32k", "W5-128k",
                 "W6-prefix-32k", "W6-prefix-128k", "W6-uniform-32k", "W6-uniform-128k")
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
SANITIZER_TOOLS = ("memcheck", "racecheck", "synccheck")
UNLOCKED_BANNER = ("> Unlocked or unrecorded SM clocks: timings are valid as relative comparisons only, "
                   "not comparable with the paper's locked-clock numbers.")
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


def default_clock_mhz() -> int:
    """The machine's qualified clock from PIPELINK_LOCK_SM_MHZ; 0 (unlocked) when unset or empty."""
    value = os.environ.get("PIPELINK_LOCK_SM_MHZ", "").strip()
    try:
        return int(value) if value else 0
    except ValueError:
        raise EvidenceError(f"PIPELINK_LOCK_SM_MHZ must be an integer MHz value or empty, got {value!r}")


def clock_status(record: dict[str, Any]) -> str:
    """Clock mode of a timing record: locked, unlocked or unknown (never assumed locked).

    New receipts carry top-level `clock_locked`. Older ones are read from their own
    fields: a lease record (`sm_mhz` with a healthy lease), a `clock` mode string,
    or the shared execution controller's record under `execution`.
    """
    locked = record.get("clock_locked")
    if isinstance(locked, bool):
        return "locked" if locked else "unlocked"
    clock = record.get("clock")
    if isinstance(clock, dict):
        if clock.get("locked") is False:
            return "unlocked"
        return "locked" if clock.get("sm_mhz") and clock.get("healthy") else "unknown"
    if clock in ("locked", "unlocked"):
        return clock
    execution = record.get("execution")
    return clock_status(execution) if isinstance(execution, dict) else "unknown"


def banner(statuses: list[str]) -> list[str]:
    """The unlocked-clock banner and a blank line when any timing is not known to be locked."""
    return [UNLOCKED_BANNER, ""] if any(status != "locked" for status in statuses) else []


def build_generation() -> str:
    path = OUT / "build-report.json"
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


class SanitizerCheckpoint:
    """Resumable record of completed sanitizer legs; not itself sanitizer evidence."""

    def __init__(self, path: Path, identity: dict[str, Any], *, resume: bool = False):
        self.path = path
        self.raw = path.parent / f"{path.stem}.raw"
        if resume:
            if not path.is_file():
                raise EvidenceError(f"cannot resume without checkpoint: {path}")
            self.data = load_json(path)
            if self.data.get("identity") != identity or self.data.get("status") != "INCOMPLETE":
                raise EvidenceError(f"checkpoint provenance differs: {path}")
        else:
            if path.exists():
                raise EvidenceError(f"checkpoint exists; use explicit --resume or a fresh result directory: {path}")
            self.data = {"schema": "attention-sanitizer-checkpoint-v1", "status": "INCOMPLETE",
                         "identity": identity, "completed": {}}
        self.raw.mkdir(parents=True, exist_ok=True)
        self.flush()

    def flush(self) -> None:
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.path)

    def get(self, key: str, command: list[str]) -> dict[str, Any] | None:
        outcome = self.data["completed"].get(key)
        if outcome is not None and outcome["command"]["command"] != command:
            raise EvidenceError(f"checkpoint command differs for {key}")
        return outcome

    def save(self, key: str, outcome: dict[str, Any], request: dict[str, Any] | None = None) -> None:
        # Raw files are written before the atomic manifest makes this leg resumable.
        run = outcome["command"]
        stem = self.raw / key
        stem.with_suffix(".stdout").write_text(run["stdout"])
        stem.with_suffix(".stderr").write_text(run["stderr"])
        stem.with_suffix(".json").write_text(json.dumps(
            {"request": request, "outcome": outcome}, indent=2, sort_keys=True) + "\n")
        self.data["completed"][key] = outcome
        self.flush()


def paths(spec: str, pipeline: str) -> tuple[Path, Path, Path]:
    link_dir = OUT / "links" / spec / pipeline
    return (
        link_dir / "link.json",
        link_dir / "pipeline.cubin",
        OUT / "controls" / spec / f"{pipeline}.cubin",
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
    relative = cubin.parent.relative_to(OUT / "links")
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
        cubin = OUT / "stock" / "sm90" / f"fa3-hdim128-bf16-{mask}-cluster1.cubin"
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
                raise EvidenceError(f"missing input: {name} (run build.py)")


def image_spec(description: dict[str, Any], role: str, ImageSpec: Any) -> Any:
    return ImageSpec.from_dict({
        **{key: value for key, value in description.items() if key != "inputs"},
        "image_id": f"{description['image_id']}-{role}",
        "image_sha256": sha256(Path(description["image_path"])),
        "metadata": {**description["metadata"], "role": role},
    })


def input_paths(images: list[dict[str, Any]]) -> dict[str, str]:
    return {role: path for description in images for role, path in description["inputs"].items()}


def binary_hashes(inputs: dict[str, str]) -> dict[str, str]:
    return {role: sha256(Path(path)) for role, path in inputs.items()
            if role in ("slot", "linked", "control", "stock")}


def write_report(name: str, records: list[dict[str, Any]], extra: dict[str, Any]) -> dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    # Preserve rows only while their input specification has the same semantics.
    previous = load_json(path) if path.is_file() else {}
    old_rows = (previous.get("cases", []) if
                previous.get("cells_document") == CELLS_DOCUMENT
                and previous.get("protocol") == extra.get("protocol") else [])
    merged = {(row["case_id"], row["pipeline"]): row for row in old_rows}
    merged.update({(row["case_id"], row["pipeline"]): row for row in records})
    rows = [merged[key] for key in sorted(merged)]
    table, digest = streams()
    payload = {"schema": "pipelink-attention-sm90-evidence-v2", "stage": name,
               "cells_json": str(CELLS_JSON), "cells_document": CELLS_DOCUMENT,
               "stream_map_sha256": digest, "stream_map": table, **extra,
               "cases": rows, "all_passed": all(row["status"] == "PASS" for row in rows)}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.with_suffix(".md").write_text(markdown_report(name, rows, payload), encoding="utf-8")
    return payload


def lease_released(execution: dict[str, Any]) -> bool:
    clock = execution.get("clock", {})
    return bool(clock.get("healthy") and clock.get("stopped"))


def qualification_lines(payload: dict[str, Any]) -> list[str]:
    """Describe the outer GPU window and clock lease that a correctness stage ran under."""
    execution = payload.get("qualification_execution")
    if not execution:
        return []
    clock = execution.get("clock", {})
    returned = NUMBER_WORDS.get(execution.get("command_returncode"), execution.get("command_returncode"))
    if clock_status(execution) == "locked":
        lease = (f", and its {clock.get('sm_mhz')} MHz clock lease "
                 + ("stayed healthy and was released." if lease_released(execution)
                    else "was not healthy or was not released."))
    else:
        lease = ", with unlocked SM clocks (no lease)."
    lines = ["", "## Qualification window", "", f"The qualification command returned {returned}{lease}"]
    if execution.get("window", {}).get("contended"):
        lines[-1] += (" Other work shared the GPU during the run (`admitted=false`), which leaves these "
                      "correctness results valid but means they support **no timing or performance claim**.")
    lines.append("The JSON record keeps the complete command, the native output and the numerical results, "
                 "with the outer execution record under `qualification_execution`.")
    return lines


def definitive_lines(rows: list[dict[str, Any]], payload: dict[str, Any]) -> list[str]:
    results = [(label, result) for row in rows for label, result in row.get("ratios", {}).items()]
    executions = [result["execution"] for _, result in results]
    r1 = [result for label, result in results if label == "R1"]
    r1_pass = sum(result["verdict"] == "PASS" for result in r1)
    diagnostic = sum(label != "R1" and result["status"] == "PASS" for label, result in results)
    admitted = sum(execution["admitted"] for execution in executions)
    mhz = sorted({execution.get("clock", {}).get("sm_mhz") or "unlocked" for execution in executions}, key=str)
    lines = ["", "## Admission and prerequisites", "",
             f"{admitted} of {len(results)} comparisons were admitted: {r1_pass} of {len(r1)} R1 comparisons "
             f"pass, and {NUMBER_WORDS.get(diagnostic, diagnostic)} R3 comparisons completed as context."]
    if admitted == len(results) and r1_pass == len(r1):
        lines[-1] = (f"All {len(results)} comparisons were admitted: the {r1_pass} R1 comparisons pass, and the "
                     f"{NUMBER_WORDS.get(diagnostic, diagnostic)} R3 comparisons completed as context.")
    if executions and all(not execution.get("window", {}).get("contended") for execution in executions) \
            and all(execution.get("command_returncode") == 0 for execution in executions) \
            and all(lease_released(execution) for execution in executions):
        lines[-1] += (" Every timing window was uncontended, every native command returned zero, and every "
                      + (f"{'/'.join(map(str, mhz))} MHz clock lease stayed healthy and was released."
                         if all(clock_status(execution) == "locked" for execution in executions)
                         else "run completed cleanly; clocks were not (all) locked."))
    ratios = [result["ratio"] for result in r1 if result.get("ratio") is not None]
    if ratios:
        lines.append(f"R1 ranges from **{min(ratios):.9f}** to **{max(ratios):.9f}** against the 0.97 floor; "
                     "R3 has no threshold.")
    if "sm90-gate.json" in payload.get("prerequisites", []) and (RESULTS / "sm90-gate.json").is_file():
        gate = load_json(RESULTS / "sm90-gate.json") if (RESULTS / "sm90-gate.json").is_file() else {}
        sanitize = load_json(RESULTS / "sm90-sanitize.json") if (RESULTS / "sm90-sanitize.json").is_file() else {}
        gate_cells = len(gate.get("cases", []))
        sanitize_cells = sanitize.get("cases", [])
        tools = sum(len(row["checks"]) for row in sanitize_cells)
        lines += ["", f"Timing required passing [numerical gates on all {gate_cells} cells](sm90-gate.md) and "
                  f"[sanitizer runs on {NUMBER_WORDS.get(len(sanitize_cells), len(sanitize_cells))} representative "
                  f"cells, {tools} tool invocations](sm90-sanitize.md) from the same build."]
    lines += ["", "The JSON record keeps every raw block, event duration, wall interval, numerical check and "
              "admission record; it and the raw native, window and clock logs stay local and are not published."]
    return lines


NUMBER_WORDS = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}


def markdown_report(name: str, rows: list[dict[str, Any]], payload: dict[str, Any]) -> str:
    if name in ("sm90-gate", "sm90-sanitize"):
        checks = [(row, label, result) for row in rows for label, result in row["checks"].items()]
        passed = sum(row["status"] == "PASS" for row in rows)
        if name == "sm90-gate":
            lines = [f"# {name}", "", f"{passed}/{len(rows)} cells pass their numerical gates. These are the same "
                     "cells that the definitive run times.", "",
                     "For each cell, the linked and control outputs must match bitwise over the full O and LSE "
                     "tensors (Gate A), and each must match an independent FP32 reference on the boundary rows of O "
                     "(Gate B) and LSE (Gate C). Where the stock FA3 image supports the cell, it is checked against "
                     "the same FP32 reference; it has no bitwise partner. The `linked/sass` row confirms that the "
                     "linked image's SASS keeps both setmaxnreg register-reconfiguration instructions (increase and "
                     "decrease). These are functional checks and make no timing claim.", "",
                     "| Cell | Pipeline | Check | Verdict |", "|---|---|---|---|"]
            lines += [f"| {row['case_id']} | {row['pipeline']} | {label} | {'PASS' if result['passed'] else 'FAIL'} |"
                      for row, label, result in checks]
        else:
            tools = [tool for tool in SANITIZER_TOOLS if any(label.endswith(f"/{tool}") for _, label, _ in checks)]
            findings = [f"{tool} {'hazards' if tool == 'racecheck' else 'errors'}" for tool in tools]
            findings = ", ".join(findings[:-1]) + (" or " if len(findings) > 1 else "") + findings[-1]
            lines = [f"# {name}", "", f"{passed}/{len(rows)} representative cells pass, and "
                     f"{sum(result['passed'] for _, _, result in checks)}/{len(checks)} sanitizer invocations "
                     f"report zero {findings}.", "",
                     "Each invocation runs the `paired-reload-v2-check` path under compute-sanitizer: the same "
                     "shared buffers and module reload that definitive timing uses, with two queued complete calls "
                     "per image in each half instead of the full timed sequence. Numerical checks run before the "
                     "work, after the reload and after the work. A control-reference run executes the linked and "
                     "control images; a stock-reference run executes stock as well.", "",
                     "Long cells are too slow to race-check, so the sanitizers run on shorter representative cells. "
                     "A timed cell without its own record counts as covered when a passing record exists for another "
                     "cell of the same pipeline with the same executable inputs in the same build; a recorded "
                     "failure is never covered this way. These are correctness "
                     "checks and make no timing claim.", "",
                     "| Representative | Pipeline | Paired reference / tool | Hazards or errors | Verdict |",
                     "|---|---|---|---:|---|"]
            lines += [f"| {row['case_id']} | {row['pipeline']} | {label} | "
                      f"{result.get('hazards', result.get('errors'))} | {'PASS' if result['passed'] else 'FAIL'} |"
                      for row, label, result in checks]
        return "\n".join(lines + qualification_lines(payload)) + "\n"
    statuses = [clock_status(result["execution"]) if "execution" in result else clock_status(row)
                for row in rows for result in row.get("ratios", {}).values()]
    if any(row.get("protocol") == "paired-reload-v2" for row in rows):
        lines = [f"# {name}", "", *banner(statuses),
                 "Each row compares the linked image (A) with a reference image (B) on one cell. R1 uses the "
                 "same-source control as the reference and must reach a ratio of at least 0.97. R3 uses stock FA3 "
                 "and is reported for context, with no threshold.", "",
                 "The `paired-reload-v2` protocol times eight blocks in balanced ABBA/BAAB order. Each block has "
                 "four batches of exactly ten complete calls, so a comparison scores 32 batches (320 calls) after "
                 "32 unscored warmup calls. After the fourth block both modules are unloaded and reloaded, reference "
                 "first, in the same context and with the same fixture. Each block yields a B/A time "
                 "ratio, and the reported ratio is sqrt(median of the first four × median of the last four), so "
                 "values above 1 mean the linked image is faster. The eight blocks balance order effects; they are "
                 "not independent samples.", "",
                 "Each comparison runs once; no result triggers a retry. It is admitted when the native command "
                 "completes inside an uncontended GPU window under a healthy, qualified clock lease and returns a "
                 "valid result. There is no minimum duration and no per-image telemetry requirement.", "",
                 "| Cell | Pipeline | Comparison | Ratio | Blocks | Calls/batch | Clock MHz | Admitted | Verdict |",
                 "|---|---|---|---:|---:|---:|---:|---|---|"]
        for row in rows:
            for label, result in row.get("ratios", {}).items():
                lines.append(f"| {row['case_id']} | {row['pipeline']} | {label} | "
                             f"{format(result['ratio'], '.9f') if result.get('ratio') is not None else 'unavailable'} | "
                             f"{len(result.get('pair', {}).get('blocks', []))} | "
                             f"10 | {result['execution'].get('clock', {}).get('sm_mhz') or 'unlocked'} | "
                             f"{result['execution']['admitted']} | {result['verdict']} |")
        return "\n".join(lines + definitive_lines(rows, payload)) + "\n"
    lines = [f"# {name}", "", *banner(statuses), "R1 compares linked with control and must reach 0.97. R2 (control/stock) and "
             "R3 (linked/stock) are reported for context.", "",
             "Each ratio is the median of paired iterations/s ratios from the slot timer; absolute latencies "
             "come from a separate timer. Attempts and discards count timing pairs, and a pair is discarded "
             "when its clock leaves the lease.", "",
             "| Cell | Pipeline | Comparison | Reference iterations/s | Candidate iterations/s | Ratio | Clock mode | Observed MHz | Valid pairs | Attempts | Discards | Verdict |",
             "|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|"]
    for row in rows:
        for label, result in row.get("ratios", {}).items():
            summary = result.get("summary", {})
            lines.append(f"| {row['case_id']} | {row['pipeline']} | {label} | "
                         f"{result['reference_iterations_per_s_median']} | {result['candidate_iterations_per_s_median']} | "
                         f"{result['ratio']} | {row.get('clock', 'unknown')} | {result['clock_mhz']} | "
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
                and row.get("input_paths") == current
                and (stage != "sanitize" or row.get("sanitizer_protocol") == "paired-reload-v2-check")):
            continue
        # Sanitizers check the executed code (the slot with its combine kernel, the link and the
        # cubins), which does not depend on sequence length. A PASS from this build at another
        # cell of the pipeline, with the same roles and paths, therefore covers long cells that
        # are too slow to race-check. Cell-specific selection data is ignored in the comparison,
        # and an explicit record for this cell always takes precedence.
        code = {role: path for role, path in current.items() if role != "selection"}
        if stage == "sanitize" and not row and any(
                other["status"] == "PASS" and key[1] == pipeline
                and other.get("build_generation") == generation
                and other.get("sanitizer_protocol") == "paired-reload-v2-check"
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
    clock = {"clock": "unlocked" if clock_mhz is None else "locked", "clock_locked": clock_mhz is not None,
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


def paired_request(cell: str, reference_kind: str, *, check: bool = False) -> dict[str, Any]:
    return {"schema_version": "attention-paired-request-v2", "action": "paired-check" if check else "paired",
            "case_id": cell, "reference_kind": reference_kind, "protocol": "paired-reload-v2"}


def paired_command(cell: str, pipeline: str, reference_kind: str) -> list[str]:
    command = image(cell, pipeline, "linked")["command"]
    if reference_kind == "stock":
        stock = image(cell, pipeline, "stock")
        command += [f"--stock-cubin={stock['inputs']['stock']}",
                    f"--stock-meta={stock['inputs']['stock_meta']}"]
    return command


def paired_check_pass(result: dict[str, Any] | None, cell: str, reference_kind: str) -> bool:
    if not result:
        return False
    correctness = result.get("correctness", {})
    images = ["linked", "control"] + (["stock"] if reference_kind == "stock" else [])
    return (result.get("action") == "paired-check" and result.get("protocol") == "paired-reload-v2"
            and result.get("case_id", result.get("cell")) == cell
            and result.get("reference_kind") == reference_kind and result.get("timing") is False
            and result.get("queued_calls_per_arm_per_half") == 2 and result.get("halves") == 2
            and result.get("images_executed") == images
            and all(correctness.get(key) is True for key in
                    ("all_true", "checked_before_work", "checked_after_reload", "checked_after_work",
                     "gate_a_bitwise", "gate_b_linked", "gate_b_control", "gate_c_lse"))
            and (reference_kind != "stock" or correctness.get("gate_stock") is True))


def paired_definitive(work: list[tuple[str, str]], gpus: list[int], clock_mhz: int | None) -> dict[str, Any]:
    sys.path.insert(0, str(REPO))
    from evaluator.harness.parity.fixed_count import run_native_pair
    require_prior("gate", work)
    require_prior("sanitize", work)
    generation = build_generation()
    records = []
    root = RESULTS / "sm90-definitive"
    # Earlier records are kept: a rerun needs a new results directory.
    root.mkdir(parents=True, exist_ok=False)
    for cell, pipeline in work:
        images = check_images(cell, pipeline)
        require_inputs(list(images.values()))
        ratios = {}
        for reference_kind in ("control", "stock") if stock_supported(CELLS[cell]) else ("control",):
            label = "R1" if reference_kind == "control" else "R3"
            execution = run_native_pair(
                paired_command(cell, pipeline, reference_kind), env=os.environ.copy(),
                gpus=gpus, clock_mhz=clock_mhz, out_dir=root / f"{cell}.{pipeline}" / label,
                input=json.dumps(paired_request(cell, reference_kind)), timeout=1800)
            native = execution.get("payload") or {}
            pair = native.get("pair") or {}
            correct = native.get("correctness", {})
            valid = (execution["admitted"] and native.get("case_id") == cell
                     and native.get("reference_kind") == reference_kind
                     and all(correct.get(key) is True for key in
                             ("all_true", "checked_before_warmup", "checked_after_reload",
                              "gate_a_bitwise", "gate_b_linked", "gate_b_control", "gate_c_lse"))
                     and (reference_kind != "stock" or correct.get("gate_stock") is True))
            ratio = pair.get("ratio_onetu_over_linked")
            verdict = ("INVALID" if not valid else "DIAGNOSTIC" if label == "R3"
                       else "PASS" if ratio >= 0.97 else "MISS")
            ratios[label] = {"status": "PASS" if valid and verdict != "MISS" else "FAIL",
                             "diagnostic_only": label != "R1", "verdict": verdict, "ratio": ratio,
                             "pair": pair, "execution": execution}
        records.append({"case_id": cell, "pipeline": pipeline, "spec": SPEC,
                        "protocol": "paired-reload-v2", "gpu": gpus[0],
                        "case_config": case_config(cell, pipeline), "ratios": ratios,
                        "build_generation": generation,
                        "input_paths": input_paths(list(images.values())),
                        "binary_sha256": binary_hashes(input_paths(list(images.values()))),
                        "status": "PASS" if all(r["status"] == "PASS" for r in ratios.values()) else "FAIL"})
    payload = write_report("sm90-definitive", records,
                           {"protocol": "paired-reload-v2", "gpus": gpus,
                            "clock": "locked" if clock_mhz else "unlocked",
                            "clock_locked": bool(clock_mhz), "clock_mhz": clock_mhz,
                            "prerequisites": ["sm90-gate.json", "sm90-sanitize.json"]})
    if not payload["all_passed"]:
        raise EvidenceError("one or more fixed-count comparisons failed; retained all raw results")
    return payload


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
                    protocol = Protocol.smoke(archive, config)
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
                         "speedup_over_reference": reference / candidate}
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
        "speedup_definition": "reference latency / candidate latency (higher is faster)",
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


def checks(stage: str, work: list[tuple[str, str]], gpus: list[int], *, resume: bool = False) -> dict[str, Any]:
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
    checkpoint = None
    if stage == "sanitize":
        identity = {"build_generation": generation, "gpus": gpus, "cells_document": CELLS_DOCUMENT,
                    "sanitizer_protocol": "paired-reload-v2-check",
                    "work": [{"cell": cell, "pipeline": pipeline,
                              "input_paths": input_paths(list(plans[cell, pipeline].values())),
                              "binary_sha256": binary_hashes(input_paths(list(plans[cell, pipeline].values())))}
                             for cell, pipeline in work]}
        checkpoint = SanitizerCheckpoint(RESULTS / "sm90-sanitize.partial.json", identity, resume=resume)
    records = []
    for index, (cell, pipeline) in enumerate(work):
        gpu = gpus[index % len(gpus)]
        images = plans[cell, pipeline]
        outcomes = {}
        if stage == "sanitize":
            kinds = ("control", "stock") if stock_supported(CELLS[cell]) else ("control",)
            for reference_kind in kinds:
                for tool in SANITIZER_TOOLS:
                    command = [str(executable), "--tool", tool, "--error-exitcode", "99",
                               *paired_command(cell, pipeline, reference_kind)]
                    key = f"{cell}-{pipeline}-paired-{reference_kind}-{tool}"
                    previous = checkpoint.get(key, command)
                    if previous is not None:
                        outcomes[f"paired-{reference_kind}/{tool}"] = previous
                        continue  # Failed completed legs are retained, never implicitly retried.
                    run = run_command(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                                      stdin=json.dumps(paired_request(cell, reference_kind, check=True)))
                    result = slot_result(run)
                    count = sanitizer_count(run["stdout"] + "\n" + run["stderr"], tool)
                    passed = run["returncode"] == 0 and count == 0 and paired_check_pass(result, cell, reference_kind)
                    outcomes[f"paired-{reference_kind}/{tool}"] = {
                        "command": run, "slot_result": result, "passed": passed,
                        "hazards" if tool == "racecheck" else "errors": count}
                    checkpoint.save(key, outcomes[f"paired-{reference_kind}/{tool}"],
                                    paired_request(cell, reference_kind, check=True))
        else:
            for mode, description in images.items():
                request = gate_request(cell, description["image_id"], gpu)
                request.update(case_config=case_config(cell, pipeline), image_role=mode)
                run = run_command(description["command"], env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                                  stdin=json.dumps(request, separators=(",", ":")))
                result = slot_result(run)
                outcomes[f"{mode}/gate"] = {"command": run, "slot_result": result,
                                           "passed": run["returncode"] == 0 and gates_pass(result, mode)}
        if stage == "gate":
            audit, command = sass_audit(Path(images["linked"]["image_path"]), cuda, results=RESULTS)
            outcomes["linked/sass"] = {"audit": audit, "command": command,
                                       "passed": audit["setmaxnreg_inc_present"] and audit["setmaxnreg_dec_present"]}
        records.append({"case_id": cell, "spec": SPEC, "pipeline": pipeline,
                        "gpu": gpu, "checks": outcomes, "build_generation": generation,
                        "input_paths": input_paths(list(images.values())),
                        "binary_sha256": binary_hashes(input_paths(list(images.values()))),
                        **({"sanitizer_protocol": "paired-reload-v2-check"} if stage == "sanitize" else {}),
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
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "sm90-sass.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (RESULTS / "sm90-sass.md").write_text(sass_markdown(payload), encoding="utf-8")
    return payload


def sass_markdown(payload: dict[str, Any]) -> str:
    records, versions = payload["pipelines"], payload["tool_versions"]
    lines = [
        "# SM90 attention SASS diagnostic", "",
        "This report compares the SASS of each pipeline's linked image with its same-source control.",
        "It is a diagnostic: differences inform, but never accept or reject an image.",
        "",
        "All device functions are normalized in dump order with `evaluator/gates` and compared as",
        "ordered streams. Instruction counts and resources are shown as control / linked; resources",
        "are REG/STACK/LOCAL, with stack and local sizes in bytes. N counts operand deltas, not",
        "distinct instructions, and structural differences are checked separately, so read the",
        "Equal column for equality. The JSON record keeps the first 200 deltas, the full delta-class",
        "counts, the checks and the tool output.", "",
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
    lines.extend(["", "Delta classes name where the ordered streams differ; they do not attribute a",
                  "difference to a particular seam. Acceptance rests on the numerical, sanitizer and",
                  "performance evidence.",
                  "", "## Tool versions", ""])
    for tool in sorted(versions, key=lambda name: (name != "nvcc", name)):
        version = versions[tool]
        lines.extend([f"### {tool}", "", "```text", version["stdout"].strip(), "```", ""])
    return "\n".join(lines) + "\n"


REPORT_STAGES = ("sm90-gate", "sm90-sanitize", "sm90-definitive")


def render() -> list[str]:
    """Rewrite the Markdown reports from the JSON records in RESULTS (CPU only)."""
    written = []
    for name in REPORT_STAGES:
        path = RESULTS / f"{name}.json"
        if path.is_file():
            payload = load_json(path)
            path.with_suffix(".md").write_text(markdown_report(name, payload["cases"], payload), encoding="utf-8")
            written.append(str(path.with_suffix(".md")))
    path = RESULTS / "sm90-sass.json"
    if path.is_file():
        path.with_suffix(".md").write_text(sass_markdown(load_json(path)), encoding="utf-8")
        written.append(str(path.with_suffix(".md")))
    return written


def main() -> int:
    global RESULTS, OUT, SLOT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    sass_parser = subparsers.add_parser("sass", help="GPU-free linked/control SASS diagnostic")
    sass_parser.add_argument("--dry-run", action="store_true", help="print image paths without inputs or outputs")
    sass_parser.add_argument("--results", type=Path, default=RESULTS, help="output directory")
    sass_parser.add_argument("--out-root", type=Path, default=OUT)
    render_parser = subparsers.add_parser("render", help="CPU-only: rewrite Markdown reports from --results JSON")
    render_parser.add_argument("--results", type=Path, required=True, help="directory holding the JSON records")
    for name in ("stock-baseline", "gate", "sanitize", "smoke", "definitive", "absolute", "seam-cost"):
        command = subparsers.add_parser(name)
        command.add_argument("--gpus", default=os.environ.get("PIPELINK_GPUS", "").split(",")[0],
                             help="physical GPU IDs (timed commands require one); default: first of PIPELINK_GPUS")
        command.add_argument("--results", type=Path, default=RESULTS, help="output and prerequisite evidence directory")
        command.add_argument("--out-root", type=Path, default=OUT, help="artifact tree including build-report.json")
        if name not in ("gate", "sanitize"):
            command.add_argument("--clock-mhz", type=int,
                                 help="request a gpulockd lease at N MHz; 0 = unlocked; "
                                      "default: PIPELINK_LOCK_SM_MHZ (unlocked when unset, empty or 0)")
        command.add_argument("--cells-json", type=Path, default=CELLS_JSON, help="shared pipelink-sm90-cells-v1 file")
        command.add_argument("--dry-run", action="store_true", help="print commands/configs without inputs or GPUs")
        if name == "sanitize":
            command.add_argument("--resume", action="store_true", help="reuse completed checkpoint legs with identical provenance")
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
    if args.command == "render":
        print(json.dumps({"written": render()}, indent=2))
        return 0
    OUT = args.out_root.resolve()
    SLOT = OUT / "bin" / "attention-slot-sm90"
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
        defaults = SEAM_CELLS if stage == "seam-cost" else PRIMARY_CELLS if stage == "definitive" else BASE_CELLS
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
    if not args.gpus:
        parser.error("--gpus is required when PIPELINK_GPUS is empty")
    try:
        gpus = [int(item) for item in args.gpus.split(",")]
    except ValueError:
        parser.error("--gpus must be a comma-separated list of physical GPU IDs")
    if len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        parser.error("--gpus must select unique non-negative physical GPU IDs")
    clock_mhz = None
    if stage not in ("gate", "sanitize"):
        requested = args.clock_mhz if args.clock_mhz is not None else default_clock_mhz()
        if requested < 0:
            parser.error("--clock-mhz must be a positive MHz value, or 0 for unlocked")
        clock_mhz = requested or None
    if stage not in ("gate", "sanitize") and len(gpus) != 1:
        parser.error("timed sm90 evidence requires exactly one physical GPU")
    if stage == "definitive" and args.include_r2:
        parser.error("paired definitive has no R2; R2 remains a separate smoke diagnostic")
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
            elif stage == "definitive":
                kinds = ("control", "stock") if stock_supported(CELLS[cell]) else ("control",)
                images = {kind: {"command": paired_command(cell, pipeline, kind)} for kind in kinds}
                requests = {kind: paired_request(cell, kind) for kind in kinds}
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
        payload = checks(stage, work, gpus, resume=getattr(args, "resume", False))
    elif stage == "definitive":
        payload = paired_definitive(work, gpus, clock_mhz)
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
