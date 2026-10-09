"""Command-line entry point for the shared parity evaluator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .assign import discover_idle_gpus, write_failover_record
from .preflight import write_preflight
from .runner import ImageSpec, Protocol, run_case
from .streams import stream_catalog, write_stream_map
from .util import write_json_atomic


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _validate_sweep_options(
    name: str, sweep_arguments: dict[str, object | None]
) -> None:
    if name == "sweep":
        missing = [
            option for option, value in sweep_arguments.items() if value is None
        ]
        if missing:
            raise ValueError(
                "--protocol sweep requires all four sweep options; missing "
                + ", ".join(missing)
            )
        return
    supplied = [
        option for option, value in sweep_arguments.items() if value is not None
    ]
    if supplied:
        raise ValueError(
            "sweep options require --protocol sweep: " + ", ".join(supplied)
        )


def _protocol(
    name: str,
    archive: Path,
    case_config: dict,
    gap_row_allowlist: list[str],
    *,
    sweep_warmups: int | None = None,
    sweep_slot_seconds: float | None = None,
    sweep_valid_pairs: int | None = None,
    sweep_attempt_cap: int | None = None,
) -> Protocol:
    sweep_arguments = {
        "--sweep-warmups": sweep_warmups,
        "--sweep-slot-seconds": sweep_slot_seconds,
        "--sweep-valid-pairs": sweep_valid_pairs,
        "--sweep-attempt-cap": sweep_attempt_cap,
    }
    _validate_sweep_options(name, sweep_arguments)
    if name == "sweep":
        return Protocol.sweep(
            archive,
            case_config,
            warmups=sweep_warmups,
            slot_seconds=sweep_slot_seconds,
            valid_pairs=sweep_valid_pairs,
            attempt_cap=sweep_attempt_cap,
            gap_row_allowlist=gap_row_allowlist,
        )
    if name == "definitive":
        return Protocol.definitive(
            archive, case_config, gap_row_allowlist=gap_row_allowlist
        )
    if name == "smoke":
        return Protocol.smoke(
            archive, case_config, gap_row_allowlist=gap_row_allowlist
        )
    if name == "rehearsal":
        return Protocol.rehearsal(
            archive, case_config, gap_row_allowlist=gap_row_allowlist
        )
    if name == "original-duration":
        return Protocol.original_duration(
            archive, case_config, gap_row_allowlist=gap_row_allowlist
        )
    if name == "negative-rehearsal":
        return Protocol.negative_launch_scale(
            archive, case_config, mode="rehearsal"
        )
    if name == "negative-definitive":
        return Protocol.negative_launch_scale(
            archive, case_config, mode="definitive"
        )
    raise AssertionError(name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m evaluator.harness.parity")
    subparsers = parser.add_subparsers(dest="command", required=True)

    catalog = subparsers.add_parser("stream-catalog", help="dump frozen namespace/role/equivalence catalog")
    catalog.add_argument("--output", type=Path, required=True)

    stream_map = subparsers.add_parser("stream-map", help="validate resolved cases and write canonical stream-map.json")
    stream_map.add_argument("--input", type=Path, required=True)
    stream_map.add_argument("--output", type=Path, required=True)

    preflight = subparsers.add_parser("preflight", help="validate and write a preflight matrix")
    preflight.add_argument("--input", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)

    assignment = subparsers.add_parser(
        "discover-assign", help="discover, lock, and assign idle nonzero GPUs"
    )
    assignment.add_argument("--cases", required=True, help="comma-separated case IDs")
    assignment.add_argument("--archive", type=Path, required=True)
    assignment.add_argument("--max-devices", type=int, default=4)

    failover = subparsers.add_parser("failover-record", help="validate an objective trigger and write failover-record.json")
    failover.add_argument("--input", type=Path, required=True)
    failover.add_argument("--output", type=Path, required=True)

    run = subparsers.add_parser("run", help="run one definitive/smoke/rehearsal comparison")
    run.add_argument("--case-id", required=True)
    run.add_argument("--images", type=Path, required=True, help="JSON with reference and linked ImageSpec objects")
    run.add_argument("--case-config", type=Path, required=True)
    run.add_argument("--gpu", type=int, required=True)
    run.add_argument("--protocol", choices=("definitive", "smoke", "rehearsal", "original-duration", "negative-rehearsal", "negative-definitive", "sweep"), default="definitive")
    run.add_argument("--archive", type=Path, required=True)
    run.add_argument("--sweep-warmups", type=int)
    run.add_argument("--sweep-slot-seconds", type=float)
    run.add_argument("--sweep-valid-pairs", type=int)
    run.add_argument("--sweep-attempt-cap", type=int)
    run.add_argument(
        "--gap-row-case",
        action="append",
        default=[],
        help="closed allowlisted gap-row case ID; repeat for each permitted case",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "stream-catalog":
        write_json_atomic(args.output, stream_catalog())
        print(json.dumps({"output": str(args.output)}, sort_keys=True))
        return 0
    if args.command == "stream-map":
        result = write_stream_map(args.output, _load(args.input))
        print(json.dumps({"output": str(args.output), "sha256": result["sha256"]}, sort_keys=True))
        return 0
    if args.command == "preflight":
        result = write_preflight(args.output, _load(args.input))
        print(json.dumps({"output": str(args.output), "summary": result["summary"]}, sort_keys=True))
        return 0 if result["summary"]["all_true"] else 1
    if args.command == "discover-assign":
        cases = [value for value in args.cases.split(",") if value]
        with discover_idle_gpus(
            args.archive, max_devices=args.max_devices
        ) as lease:
            result = lease.assign(cases)
            print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "failover-record":
        result = write_failover_record(args.output, _load(args.input))
        print(json.dumps({"output": str(args.output), "objective_trigger": result["objective_trigger"]}, sort_keys=True))
        return 0
    if args.command == "run":
        _validate_sweep_options(
            args.protocol,
            {
                "--sweep-warmups": args.sweep_warmups,
                "--sweep-slot-seconds": args.sweep_slot_seconds,
                "--sweep-valid-pairs": args.sweep_valid_pairs,
                "--sweep-attempt-cap": args.sweep_attempt_cap,
            },
        )
        images = _load(args.images)
        if set(images) != {"reference", "linked"}:
            raise ValueError("images JSON must contain exactly reference and linked")
        reference = ImageSpec.from_dict(images["reference"])
        linked = ImageSpec.from_dict(images["linked"])
        protocol = _protocol(
            args.protocol,
            args.archive,
            _load(args.case_config),
            args.gap_row_case,
            sweep_warmups=args.sweep_warmups,
            sweep_slot_seconds=args.sweep_slot_seconds,
            sweep_valid_pairs=args.sweep_valid_pairs,
            sweep_attempt_cap=args.sweep_attempt_cap,
        )
        result = run_case(args.case_id, reference, linked, args.gpu, protocol)
        print(json.dumps(result["summary"], sort_keys=True))
        return 0 if result["summary"]["status"] == "PASS" else 2
    raise AssertionError(args.command)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError) as error:
        print(f"parity: {error}", file=sys.stderr)
        raise SystemExit(2)
