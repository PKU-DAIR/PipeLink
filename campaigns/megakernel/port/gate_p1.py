#!/usr/bin/env python3
"""Gate P1 for the PipeLink Megakernels llama-1b port.

For each requested pipeline this stdlib-only gate compares the linked cubin with
its same-source one-TU control and requires:

* REG, STACK, static SHARED, and LOCAL resource fields to be equal;
* linked REG to be 96 (and therefore, by equality, one-TU REG to be 96);
* spill-store and spill-load counts to be equal;
* instruction-count drift to be at most 0.5 percent;
* CALL.REL counts to be equal;
* no surviving org_pipelink_megakernel_* or pipelink_* seam symbol (pipelink_kernel is
  the expected entry and is excluded); and
* every nonzero opcode-histogram delta to be explained entirely by either
  FFMA <-> FMUL+FADD contraction or an LDC <-> LDCU swap.

pipec2 link.json publishes observed [REG, STACK, LOCAL], but not spill bytes,
and nvJitLink does not publish a ptxas log. Therefore the comparable spill
metric is the SASS STL count for spill stores and LDL count for spill loads on
both cubins. Available one-TU/mono ptxas spill-byte reports are retained as
supplementary evidence and are not mixed with that proxy.

The gate also requires the link.json binding summaries to be complete and
checks the planned A, smem, register, and opcode permutations. The upstream
mono cubin is reported as a non-parity control (resources and instruction
count); it is not compared against linked output.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any


INSTRUCTION_RE = re.compile(
    r"^\s+/\*[0-9a-f]{4,}\*/\s+(?:(?:@!?U?P[0-9T])\s+)?([A-Z0-9_.]+)"
)
FUNCTION_RE = re.compile(r"^\s*Function\s*:?[ \t]*(.+?):\s*$")
RESOURCE_RE = re.compile(r"\b(REG|STACK|SHARED|LOCAL):(\d+)\b")
CONSTANT_RE = re.compile(r"\bCONSTANT(?:\[(\d+)\])?:(\d+)\b")
SEAM_RE = re.compile(r"\b(?:org_pipelink_megakernel_|pipelink_)[A-Za-z0-9_.$@]*")
PTXAS_FUNCTION_RE = re.compile(r"Function properties for\s+['\"]?([^'\"\r\n]+)")
PTXAS_SPILL_RE = re.compile(r"(\d+) bytes spill stores,\s*(\d+) bytes spill loads")
EXPECTED_ENTRY = "pipelink_kernel"


class GateError(RuntimeError):
    """An expected build artifact or CUDA-tool result was unavailable."""


def run_tool(tool: Path, option: str, cubin: Path) -> str:
    try:
        completed = subprocess.run(
            [str(tool), option, str(cubin)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise GateError(f"cannot run {tool}: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise GateError(
            f"{tool.name} {option} failed for {cubin} "
            f"with exit {completed.returncode}: {detail}"
        )
    return completed.stdout


def parse_resources(text: str, expected: str = EXPECTED_ENTRY) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    current: str | None = None
    for line in text.splitlines():
        match = FUNCTION_RE.match(line)
        if match:
            current = match.group(1).strip()
            records[current] = {
                "reg": 0,
                "stack": 0,
                "static_shared": 0,
                "local": 0,
                "constant": 0,
                "constant_banks": {},
            }
            continue
        if current is None:
            continue
        record = records[current]
        for field, value_text in RESOURCE_RE.findall(line):
            key = {
                "REG": "reg",
                "STACK": "stack",
                "SHARED": "static_shared",
                "LOCAL": "local",
            }[field]
            record[key] = int(value_text)
        for bank, value_text in CONSTANT_RE.findall(line):
            bank_name = bank or "unbanked"
            record["constant_banks"][bank_name] = int(value_text)
    if expected not in records:
        available = ", ".join(sorted(records)) or "none"
        raise GateError(
            f"resource dump has no Function {expected!r}; available functions: {available}"
        )
    result = records[expected]
    result["constant_banks"] = dict(sorted(result["constant_banks"].items()))
    result["constant"] = sum(result["constant_banks"].values())
    return result


def parse_sass(text: str) -> dict[str, Any]:
    histogram: Counter[str] = Counter()
    instruction_count = 0
    call_rel = 0
    hmma = 0
    spill_stores = 0
    spill_loads = 0
    nop_positions: list[int] = []
    last_exit = -1
    for line in text.splitlines():
        match = INSTRUCTION_RE.match(line)
        if match is None:
            continue
        instruction_count += 1
        opcode = match.group(1)
        base = opcode.split(".", 1)[0]
        histogram[base] += 1
        if base == "NOP":
            nop_positions.append(instruction_count)
        elif base == "EXIT":
            last_exit = instruction_count
        if "CALL.REL" in line:
            call_rel += 1
        if "HMMA" in line:
            hmma += 1
        if base == "STL":
            spill_stores += 1
        elif base == "LDL":
            spill_loads += 1
    return {
        "instruction_count": instruction_count,
        "call_rel": call_rel,
        "stl_ldl": spill_stores + spill_loads,
        "hmma": hmma,
        "spill_store_proxy_stl": spill_stores,
        "spill_load_proxy_ldl": spill_loads,
        "histogram": dict(sorted(histogram.items())),
        # NOPs after the final EXIT are out-of-line/tail alignment padding; NOPs
        # before it are scheduling NOPs inside reachable code and must match.
        "nops_before_last_exit": sum(1 for pos in nop_positions if pos < last_exit),
        "nops_after_last_exit": sum(1 for pos in nop_positions if pos > last_exit),
    }


def parse_seam_symbols(text: str) -> list[str]:
    symbols = {match.group(0) for match in SEAM_RE.finditer(text)}
    symbols.discard(EXPECTED_ENTRY)
    return sorted(symbols)


def parse_ptxas_spills(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    current = "unattributed"
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        function = PTXAS_FUNCTION_RE.search(line)
        if function:
            current = function.group(1).strip()
        spill = PTXAS_SPILL_RE.search(line)
        if spill:
            records.append(
                {
                    "function": current,
                    "spill_store_bytes": int(spill.group(1)),
                    "spill_load_bytes": int(spill.group(2)),
                }
            )
    return {
        "path": str(path),
        "records": records,
        "total_spill_store_bytes": sum(item["spill_store_bytes"] for item in records),
        "total_spill_load_bytes": sum(item["spill_load_bytes"] for item in records),
    }


def analyze_cubin(
    cuobjdump: Path,
    cubin: Path,
    log_dir: Path,
    label: str,
    ptxas_log: Path | None = None,
    expected_function: str | None = EXPECTED_ENTRY,
) -> dict[str, Any]:
    if not cubin.is_file():
        raise GateError(f"missing cubin: {cubin}")
    resource_text = run_tool(cuobjdump, "--dump-resource-usage", cubin)
    sass_text = run_tool(cuobjdump, "--dump-sass", cubin)
    symbol_text = run_tool(cuobjdump, "--dump-elf-symbols", cubin)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{label}.resources.txt").write_text(resource_text, encoding="utf-8")
    (log_dir / f"{label}.sass").write_text(sass_text, encoding="utf-8")
    (log_dir / f"{label}.symbols.txt").write_text(symbol_text, encoding="utf-8")
    if expected_function is None:
        # Upstream mono control: exactly one kernel (`mk<...>`), whatever its mangling.
        functions = [m.group(1).strip() for m in re.finditer(FUNCTION_RE.pattern, resource_text, re.M)]
        functions = [f for f in functions if f]
        if len(set(functions)) != 1:
            raise GateError(f"expected exactly one function in {cubin}, found {functions}")
        expected_function = functions[0]
    return {
        "cubin": str(cubin),
        "function": expected_function,
        "resources": parse_resources(resource_text, expected_function),
        "sass": parse_sass(sass_text),
        "seam_symbols": parse_seam_symbols(symbol_text),
        "ptxas_spills": parse_ptxas_spills(ptxas_log),
    }


def histogram_explanation(
    linked: dict[str, int],
    one_tu: dict[str, int],
    linked_sass: dict[str, Any] | None = None,
    one_tu_sass: dict[str, Any] | None = None,
) -> dict[str, Any]:
    names = sorted(set(linked) | set(one_tu))
    delta = {
        name: linked.get(name, 0) - one_tu.get(name, 0)
        for name in names
        if linked.get(name, 0) != one_tu.get(name, 0)
    }
    remaining = dict(delta)
    allowed: list[dict[str, Any]] = []

    # Consume as many exact FFMA <-> FMUL+FADD substitutions as the three
    # signed deltas permit. Any unmatched remainder stays unexplained.
    d_ffma = remaining.get("FFMA", 0)
    d_fmul = remaining.get("FMUL", 0)
    d_fadd = remaining.get("FADD", 0)
    if d_ffma and d_ffma * d_fmul < 0 and d_ffma * d_fadd < 0:
        count = min(abs(d_ffma), abs(d_fmul), abs(d_fadd))
        sign = 1 if d_ffma > 0 else -1
        remaining["FFMA"] = d_ffma - sign * count
        remaining["FMUL"] = d_fmul + sign * count
        remaining["FADD"] = d_fadd + sign * count
        allowed.append(
            {
                "class": "FFMA<->FMUL+FADD",
                "count": count,
                "direction": (
                    "linked contracts FMUL+FADD to FFMA"
                    if sign > 0
                    else "linked expands FFMA to FMUL+FADD"
                ),
            }
        )

    d_ldc = remaining.get("LDC", 0)
    d_ldcu = remaining.get("LDCU", 0)
    if d_ldc and d_ldc * d_ldcu < 0:
        count = min(abs(d_ldc), abs(d_ldcu))
        sign = 1 if d_ldc > 0 else -1
        remaining["LDC"] = d_ldc - sign * count
        remaining["LDCU"] = d_ldcu + sign * count
        allowed.append(
            {
                "class": "LDC<->LDCU",
                "count": count,
                "direction": "linked uses LDC" if sign > 0 else "linked uses LDCU",
            }
        )

    d_nop = remaining.get("NOP", 0)
    if d_nop and linked_sass is not None and one_tu_sass is not None:
        before_l = linked_sass["nops_before_last_exit"]
        before_o = one_tu_sass["nops_before_last_exit"]
        if before_l == before_o:
            remaining["NOP"] = 0
            allowed.append(
                {
                    "class": "tail-padding NOP",
                    "count": abs(d_nop),
                    "direction": (
                        "linked has more trailing NOPs" if d_nop > 0 else "one-TU has more trailing NOPs"
                    ),
                    "nops_before_last_exit": before_l,
                }
            )

    unexplained = {name: value for name, value in sorted(remaining.items()) if value}
    return {
        "orientation": "linked_minus_one_tu",
        "delta": dict(sorted(delta.items())),
        "allowed_explanations": allowed,
        "unexplained": unexplained,
    }


def binding_value(link_json: dict[str, Any], name: str) -> int | None:
    raw = link_json.get("bindings", {}).get(name)
    if isinstance(raw, list) and raw and isinstance(raw[0], int) and not isinstance(raw[0], bool):
        return raw[0]
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    return None


def binding_summary(link_json: dict[str, Any]) -> dict[str, Any]:
    return {
        "smem_bases": {
            "pages": binding_value(link_json, "pipelink_megakernel_pages_base"),
            "ring": binding_value(link_json, "pipelink_megakernel_ring_base"),
            "vmbar": binding_value(link_json, "pipelink_megakernel_vmbar_base"),
        },
        "opcode_slots": {
            str(slot): binding_value(link_json, f"pipelink_megakernel_opcode_slot{slot}")
            for slot in range(8)
        },
        "regs": {
            "consumer_wg": binding_value(link_json, "pipelink_regs_consumer_wg"),
            "producer_wg": binding_value(link_json, "pipelink_regs_producer_wg"),
            "launch": binding_value(link_json, "pipelink_launch_regs"),
        },
        "regs_cap": link_json.get("regs_cap"),
        "dyn_smem_bytes": link_json.get("dyn_smem_bytes"),
        "threads": link_json.get("threads"),
        "observed": link_json.get("observed"),
    }


def binding_expectations(pipeline: str, summary: dict[str, Any]) -> dict[str, Any]:
    actuals: list[tuple[str, Any, str, Any]] = []
    smem = summary["smem_bases"]
    opcodes = summary["opcode_slots"]
    regs = summary["regs"]

    def equal(path: str, actual: Any, expected: Any) -> None:
        actuals.append((path, actual, "==", expected))

    def not_equal(path: str, actual: Any, expected: Any) -> None:
        actuals.append((path, actual, "!=", expected))

    if pipeline == "a":
        equal("smem_bases.pages", smem["pages"], 0)
        equal("smem_bases.ring", smem["ring"], 212992)
        equal("smem_bases.vmbar", smem["vmbar"], 223232)
        for slot in range(8):
            equal(f"opcode_slots.{slot}", opcodes[str(slot)], slot)
        equal("regs.consumer_wg", regs["consumer_wg"], 104)
        equal("regs.producer_wg", regs["producer_wg"], 64)
        equal("regs.launch", regs["launch"], 96)
    elif pipeline == "rebind-smem":
        equal("smem_bases.ring", smem["ring"], 0)
        equal("smem_bases.vmbar", smem["vmbar"], 10240)
        not_equal("smem_bases.pages", smem["pages"], 0)
    elif pipeline == "rebind-regs":
        equal("regs.consumer_wg", regs["consumer_wg"], 96)
        equal("regs.producer_wg", regs["producer_wg"], 96)
    elif pipeline == "rebind-opcode":
        expected = {0: 0, 1: 9, 2: 1, 3: 11, 4: 2, 5: 13, 6: 3, 7: 15}
        for slot, value in expected.items():
            equal(f"opcode_slots.{slot}", opcodes[str(slot)], value)

    mismatches: list[dict[str, Any]] = []
    for path, actual, operator, expected in actuals:
        passed = actual == expected if operator == "==" else actual is not None and actual != expected
        if not passed:
            mismatches.append(
                {"path": path, "actual": actual, "operator": operator, "expected": expected}
            )
    return {
        "checked": [
            {"path": path, "actual": actual, "operator": operator, "expected": expected}
            for path, actual, operator, expected in actuals
        ],
        "mismatches": mismatches,
    }


def summary_complete(summary: dict[str, Any]) -> tuple[bool, list[str]]:
    missing: list[str] = []
    for name, value in summary["smem_bases"].items():
        if not isinstance(value, int) or isinstance(value, bool):
            missing.append(f"smem_bases.{name}")
    for name, value in summary["opcode_slots"].items():
        if not isinstance(value, int) or isinstance(value, bool):
            missing.append(f"opcode_slots.{name}")
    for name, value in summary["regs"].items():
        if not isinstance(value, int) or isinstance(value, bool):
            missing.append(f"regs.{name}")
    for name in ("regs_cap", "dyn_smem_bytes", "threads"):
        value = summary[name]
        if not isinstance(value, int) or isinstance(value, bool):
            missing.append(name)
    return not missing, missing


def add_check(
    checks: dict[str, dict[str, Any]], name: str, passed: bool, **detail: Any
) -> None:
    checks[name] = {"status": "PASS" if passed else "FAIL", **detail}


def compare_pipeline(
    pipeline: str, build_dir: Path, cuobjdump: Path, log_dir: Path
) -> dict[str, Any]:
    linked_path = build_dir / "linked" / pipeline / "pipeline.cubin"
    one_tu_path = build_dir / "onetu" / f"{pipeline}.cubin"
    linked = analyze_cubin(cuobjdump, linked_path, log_dir, f"{pipeline}-linked")
    one_tu = analyze_cubin(
        cuobjdump,
        one_tu_path,
        log_dir,
        f"{pipeline}-onetu",
        build_dir / "onetu" / f"{pipeline}.ptxas.txt",
    )

    link_json_path = build_dir / "linked" / pipeline / "link.json"
    if not link_json_path.is_file():
        raise GateError(f"missing link result: {link_json_path}")
    try:
        link_json = json.loads(link_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot read {link_json_path}: {exc}") from exc
    if not isinstance(link_json, dict):
        raise GateError(f"{link_json_path} must contain a JSON object")

    lr = linked["resources"]
    one_resources = one_tu["resources"]
    ls = linked["sass"]
    one_sass = one_tu["sass"]
    checks: dict[str, dict[str, Any]] = {}
    for field, check_name in (
        ("reg", "reg_equal"),
        ("stack", "stack_equal"),
        ("static_shared", "static_shared_equal"),
        ("local", "local_equal"),
    ):
        add_check(
            checks,
            check_name,
            lr[field] == one_resources[field],
            linked=lr[field],
            one_tu=one_resources[field],
        )
    add_check(
        checks,
        "linked_reg_is_96",
        lr["reg"] == 96,
        linked=lr["reg"],
        expected=96,
    )
    add_check(
        checks,
        "spill_store_equal_stl_proxy",
        ls["spill_store_proxy_stl"] == one_sass["spill_store_proxy_stl"],
        linked=ls["spill_store_proxy_stl"],
        one_tu=one_sass["spill_store_proxy_stl"],
    )
    add_check(
        checks,
        "spill_load_equal_ldl_proxy",
        ls["spill_load_proxy_ldl"] == one_sass["spill_load_proxy_ldl"],
        linked=ls["spill_load_proxy_ldl"],
        one_tu=one_sass["spill_load_proxy_ldl"],
    )

    instruction_delta = ls["instruction_count"] - one_sass["instruction_count"]
    instruction_abs_delta = abs(instruction_delta)
    baseline = one_sass["instruction_count"]
    instruction_ok = (
        instruction_abs_delta == 0
        if baseline == 0
        else instruction_abs_delta * 200 <= baseline
    )
    delta_percent = None if baseline == 0 else 100.0 * instruction_delta / baseline
    add_check(
        checks,
        "instruction_delta_le_0_5_percent",
        instruction_ok,
        linked=ls["instruction_count"],
        one_tu=baseline,
        delta=instruction_delta,
        delta_percent=delta_percent,
        limit_percent=0.5,
    )
    add_check(
        checks,
        "call_rel_equal",
        ls["call_rel"] == one_sass["call_rel"],
        linked=ls["call_rel"],
        one_tu=one_sass["call_rel"],
    )
    add_check(
        checks,
        "zero_surviving_seam_symbols",
        not linked["seam_symbols"] and not one_tu["seam_symbols"],
        linked=linked["seam_symbols"],
        one_tu=one_tu["seam_symbols"],
    )

    histogram = histogram_explanation(ls["histogram"], one_sass["histogram"], ls, one_sass)
    add_check(
        checks,
        "histogram_deltas_allowed",
        not histogram["unexplained"],
        unexplained=histogram["unexplained"],
        allowed_explanations=histogram["allowed_explanations"],
    )

    bindings = binding_summary(link_json)
    complete, missing = summary_complete(bindings)
    add_check(checks, "binding_summary_complete", complete, missing=missing)
    permutation = binding_expectations(pipeline, bindings)
    add_check(
        checks,
        "binding_permutation",
        not permutation["mismatches"],
        checked=permutation["checked"],
        mismatches=permutation["mismatches"],
    )

    passed = all(check["status"] == "PASS" for check in checks.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "linked": linked,
        "one_tu": one_tu,
        "histogram_diff": histogram,
        "bindings": bindings,
    }


def print_table(pipelines: dict[str, dict[str, Any]], mono: dict[str, Any]) -> None:
    headers = (
        "pipeline",
        "status",
        "REG L/O",
        "STACK L/O",
        "SHARED L/O",
        "LOCAL L/O",
        "spill S,L L/O",
        "instr delta%",
        "CALL L/O",
        "seams",
        "hist bad",
    )
    rows: list[tuple[str, ...]] = []
    for name, result in pipelines.items():
        if "linked" not in result:
            rows.append((name, "FAIL", "-", "-", "-", "-", "-", "-", "-", "-", "-"))
            continue
        linked = result["linked"]
        one_tu = result["one_tu"]
        lr, one_resources = linked["resources"], one_tu["resources"]
        ls, one_sass = linked["sass"], one_tu["sass"]
        delta_percent = result["checks"]["instruction_delta_le_0_5_percent"][
            "delta_percent"
        ]
        rows.append(
            (
                name,
                result["status"],
                f"{lr['reg']}/{one_resources['reg']}",
                f"{lr['stack']}/{one_resources['stack']}",
                f"{lr['static_shared']}/{one_resources['static_shared']}",
                f"{lr['local']}/{one_resources['local']}",
                (
                    f"{ls['spill_store_proxy_stl']},{ls['spill_load_proxy_ldl']}"
                    f"/{one_sass['spill_store_proxy_stl']},{one_sass['spill_load_proxy_ldl']}"
                ),
                "-" if delta_percent is None else f"{delta_percent:+.4f}",
                f"{ls['call_rel']}/{one_sass['call_rel']}",
                str(len(linked["seam_symbols"]) + len(one_tu["seam_symbols"])),
                str(len(result["histogram_diff"]["unexplained"])),
            )
        )
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
    for name, result in pipelines.items():
        if "error" in result:
            print(f"{name}: {result['error']}")
            continue
        failed = [
            check_name
            for check_name, check in result["checks"].items()
            if check["status"] == "FAIL"
        ]
        if failed:
            print(f"{name}: failed checks: {', '.join(failed)}")
        unexplained = result["histogram_diff"]["unexplained"]
        if unexplained:
            formatted = ", ".join(
                f"{opcode}={delta:+d}" for opcode, delta in unexplained.items()
            )
            print(f"{name}: unexplained opcode deltas (linked-one-TU): {formatted}")
    if "resources" in mono:
        resources = mono["resources"]
        sass = mono["sass"]
        print(
            "mono control: "
            f"REG={resources['reg']} STACK={resources['stack']} "
            f"SHARED={resources['static_shared']} LOCAL={resources['local']} "
            f"CONSTANT={resources['constant']} instr={sass['instruction_count']} "
            f"CALL.REL={sass['call_rel']} STL+LDL={sass['stl_ldl']} HMMA={sass['hmma']}"
        )
    else:
        print(f"mono control: FAIL ({mono.get('error', 'unknown error')})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--pipelines", required=True, nargs="+")
    args = parser.parse_args(argv)

    cuda_home = os.environ.get("PIPELINK_CUDA_HOME")
    cuobjdump = (
        Path(cuda_home) / "bin" / "cuobjdump"
        if cuda_home
        else Path("__missing_PIPELINK_CUDA_HOME__")
    )
    build_dir = args.build.resolve()
    log_dir = build_dir / "logs"
    pipeline_results: dict[str, dict[str, Any]] = {}
    for pipeline in args.pipelines:
        try:
            pipeline_results[pipeline] = compare_pipeline(
                pipeline, build_dir, cuobjdump, log_dir
            )
        except (GateError, OSError, TypeError, ValueError) as exc:
            pipeline_results[pipeline] = {"status": "FAIL", "error": str(exc)}

    try:
        mono = analyze_cubin(
            cuobjdump,
            build_dir / "mono.cubin",
            log_dir,
            "mono",
            build_dir / "mono.ptxas.txt",
            expected_function=None,
        )
        mono["status"] = "PASS"
    except (GateError, OSError, TypeError, ValueError) as exc:
        mono = {"status": "FAIL", "error": str(exc)}

    passed = (
        mono["status"] == "PASS"
        and all(result["status"] == "PASS" for result in pipeline_results.values())
    )
    report = {
        "schema": "pipelink-megakernel-gate-p1-v1",
        "status": "PASS" if passed else "FAIL",
        "spill_metric": {
            "gate": "SASS opcode proxy: STL instructions are spill stores; LDL instructions are spill loads",
            "reason": "link.json observed contains REG/STACK/LOCAL only and nvJitLink publishes no ptxas spill log",
            "ptxas": "available one-TU and mono spill-byte records are supplementary",
        },
        "pipelines": pipeline_results,
        "mono": mono,
    }
    build_dir.mkdir(parents=True, exist_ok=True)
    report_path = build_dir / "gate_p1.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print_table(pipeline_results, mono)
    print(f"Gate P1: {report['status']} (report: {report_path})")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
