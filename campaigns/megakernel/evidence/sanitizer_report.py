"""Render the megakernel sanitizer Markdown report.

Without arguments it renders campaigns/megakernel/evidence/sanitizers.md from the
authors' sanitizer receipts in that directory. With --input-root DIR it renders
the receipts that `run_parity.py --output-root DIR` wrote (DIR/<suite>/sanitize*.json
and DIR/<suite>/paired-sanitizers/receipt.json) to --output.
CPU-only. Only selected fields reach the Markdown: pipeline, image role, bundle
name, tool, finding count, verdict and return-code basis. Commands, logs, image
paths, build identifiers, digests, GPU indices and host names stay out.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

EVIDENCE = Path(__file__).resolve().parent
OUTPUT = EVIDENCE / "sanitizers.md"
TOOLS = ("memcheck", "racecheck", "synccheck")
IMAGE_NAMES = {"linked": "linked", "onetu": "one-TU"}
SUITES = (
    ("All-ThunderKittens reference", "sanitize.json", "sanitize-paired.json"),
    ("CuTe attention composition", "sanitize-cute.json", "sanitize-paired-cute.json"),
)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "|" + "---|" * len(headers),
        *("| " + " | ".join(row) + " |" for row in rows),
    ])


def bundle_name(path: str) -> str:
    return f"`{path.rstrip('/').rsplit('/', 1)[-1]}`"


def verdict(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def image_rows(receipt: dict[str, Any]) -> list[list[str]]:
    rows = []
    for record in sorted(receipt["records"], key=lambda r: (
            r["pipeline"], r["image_role"], r["bundle"], TOOLS.index(r["tool"]))):
        rows.append([
            f"`{record['pipeline']}`",
            IMAGE_NAMES[record["image_role"]],
            bundle_name(record["bundle"]),
            record["tool"],
            str(record["count"]),
            verdict(record["passed"]),
        ])
    return rows


def image_label(image: dict[str, Any]) -> str:
    return f"`{image['pipeline']}` {IMAGE_NAMES[image['image']]}"


def returncode_basis(record: dict[str, Any]) -> str:
    evidence = record.get("returncode_evidence")
    if evidence is None:
        return f"{record['returncode']} (raw)"
    if evidence["kind"] != "derived_from_controller_PASS":
        raise ValueError(f"unknown return-code basis {evidence['kind']!r} for {record['case_id']}")
    return f"{record['returncode']} (from controller PASS line)"


def paired_rows(receipt: dict[str, Any]) -> list[list[str]]:
    rows = []
    for record in sorted(receipt["records"], key=lambda r: (r["case_id"], TOOLS.index(r["tool"]))):
        images = record["coverage"]["images"]
        rows.append([
            f"{image_label(images['candidate'])} / {image_label(images['reference'])}",
            bundle_name(record["coverage"]["bundle"]),
            record["tool"],
            str(record["count"]),
            (record.get("numeric") or {}).get("status", "FAIL"),
            f"`{record.get('status') or verdict(record['passed'])}`",
            returncode_basis(record),
        ])
    return rows


def lock_summary(provenance: dict[str, Any]) -> str:
    executions = [provenance["original_collection_timeout"]["execution"],
                  provenance["continuation_execution"]]
    mhz = {execution["clock"]["sm_mhz"] for execution in executions}
    healthy = all(e["clock"]["healthy"] and e["clock"]["stopped"] for e in executions)
    uncontended = not any(e["window"]["contended"] for e in executions)
    if len(mhz) != 1 or not healthy or not uncontended:
        raise ValueError("paired sanitizer executions do not share one healthy, stopped, uncontended clock lock")
    return (f"Every paired leg ran under a healthy, stopped **{mhz.pop()} MHz** clock lock "
            "in an uncontended GPU window.")


def receipt_paths(input_root: Path | None) -> tuple[dict[str, Path], dict[str, Path], Path | None]:
    """Per-suite image and paired receipts, plus the paired-run provenance if recorded."""
    if input_root is None:
        return ({image: EVIDENCE / image for _, image, _ in SUITES},
                {paired: EVIDENCE / paired for _, _, paired in SUITES},
                EVIDENCE / "paired-sanitizer-provenance.json")
    images, paired = {}, {}
    for (_, image, paired_name), suite in zip(SUITES, ("reference", "cute")):
        images[image] = input_root / suite / image
        paired[paired_name] = input_root / suite / "paired-sanitizers" / "receipt.json"
    return images, paired, None


def render(input_root: Path | None = None) -> str:
    image_paths, paired_paths, provenance_path = receipt_paths(input_root)
    image_receipts = {name: load(path) for name, path in image_paths.items()}
    paired_receipts = {name: load(path) for name, path in paired_paths.items()}
    provenance = load(provenance_path) if provenance_path else None

    image_total = sum(len(r["records"]) for r in image_receipts.values())
    paired_total = sum(len(r["records"]) for r in paired_receipts.values())
    if provenance and len(provenance["recovery_manifest"]["legs"]) != paired_total:
        raise ValueError("paired receipts do not cover the planned leg list")
    derived = sum(1 for r in paired_receipts.values() for rec in r["records"]
                  if rec.get("returncode_evidence") is not None)
    all_passed = all(r["passed"] for r in (*image_receipts.values(), *paired_receipts.values()))

    parts = [
        "Generated by `campaigns/megakernel/evidence/sanitizer_report.py`; do not edit.",
        "# Megakernel sanitizer evidence",
        "Results of compute-sanitizer memcheck, racecheck and synccheck for every timed image "
        "and for the paired runner/provider path. Each check requires zero findings. "
        "Bundle names refer to the local Llama-3.2-1B input bundles; image paths, logs and "
        "commands stay in the unpublished receipts.",
        f"**{verdict(all_passed)}**: {image_total} per-image checks and {paired_total} "
        "paired-path checks, all with zero findings.",
        "## Per-image sanitizers",
        "Each linked image and its same-source one-TU control runs once under each tool "
        "through the host runner on the short `p16-n2` bundle and, for pipelines timed "
        "with reduction, on the `p1024-n8-red` bundle. Timing admission requires receipts "
        "from the same build as the timed images.",
    ]
    for title, image_name, _ in SUITES:
        receipt = image_receipts[image_name]
        parts += [f"### {title}",
                  table(["Pipeline", "Image", "Bundle", "Tool", "Findings", "Verdict"],
                        image_rows(receipt))]
    parts += [
        "## Paired-path sanitizers",
        "Each leg runs the bounded `pair-check` probe of one candidate/reference pair under "
        "one tool: two queued copies and decodes per arm on each side of the module reload "
        "plus the four snapshot phases, 16 calls in total. A leg passes with zero findings, "
        "a completed tool summary and passing numerical snapshot checks "
        "(`finite-nine+strict-layer0-kv`). The 320-call scoring protocol does not run under "
        "the sanitizer." + (" " + lock_summary(provenance) if provenance else ""),
    ]
    if derived:
        parts.append(
            f"The {derived} reference legs with status `RECOVERED_COMPLETED_PASS` record a return "
            "code derived from the controller's PASS line, which the controller prints only after "
            "a zero return code, a zero tool summary and passing numerical checks; their raw "
            "return codes were not retained. The other legs record the raw return code.")
    for title, _, paired_name in SUITES:
        parts += [f"### {title}",
                  table(["Candidate / reference", "Bundle", "Tool", "Findings", "Snapshots",
                         "Status", "Return code"],
                        paired_rows(paired_receipts[paired_name]))]
    return "\n\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-root", type=Path, help="run_parity.py --output-root of a fresh run")
    parser.add_argument("--output", type=Path, help="Markdown destination (required with --input-root)")
    args = parser.parse_args()
    if args.input_root and not args.output:
        parser.error("--input-root requires --output")
    output = args.output or OUTPUT
    output.write_text(render(args.input_root), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
