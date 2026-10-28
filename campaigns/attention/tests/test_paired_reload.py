"""CPU-only integration checks of the attention protocol; the native CUDA gates run separately."""
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
import sm90_evidence as ev
import cp_evidence as cp


class PairedRequests(unittest.TestCase):
    def test_request_is_not_a_legacy_slot(self):
        self.assertEqual(ev.paired_request("W1-8k", "control"), {
            "schema_version": "attention-paired-request-v2", "action": "paired",
            "case_id": "W1-8k", "reference_kind": "control", "protocol": "paired-reload-v2"})
        self.assertEqual(ev.paired_request("W1-8k", "stock", check=True)["action"], "paired-check")

    def test_sanitizer_receipt_requires_bounded_reload_and_all_images(self):
        result = {"action": "paired-check", "protocol": "paired-reload-v2", "case_id": "W1-8k",
                  "reference_kind": "stock", "timing": False, "queued_calls_per_arm_per_half": 2,
                  "halves": 2, "images_executed": ["linked", "control", "stock"],
                  "correctness": {key: True for key in
                                  ("all_true", "checked_before_work", "checked_after_reload",
                                   "checked_after_work", "gate_a_bitwise", "gate_b_linked",
                                   "gate_b_control", "gate_c_lse", "gate_stock")}}
        self.assertTrue(ev.paired_check_pass(result, "W1-8k", "stock"))
        self.assertFalse(ev.paired_check_pass({**result, "action": "sanitize"}, "W1-8k", "stock"))
        self.assertFalse(ev.paired_check_pass({**result, "images_executed": ["stock"]}, "W1-8k", "stock"))
        self.assertFalse(ev.paired_check_pass({**result, "queued_calls_per_arm_per_half": 10}, "W1-8k", "stock"))
        self.assertFalse(ev.paired_check_pass(
            {**result, "correctness": {**result["correctness"], "checked_after_reload": False}}, "W1-8k", "stock"))

    def test_cp_sanitizer_executes_bounded_pair_for_each_tool(self):
        result = {"action": "paired-check", "protocol": "paired-reload-v2", "cell": "W1-8k",
                  "world": 2, "reference_kind": "control", "timing": False,
                  "queued_calls_per_arm_per_half": 2, "halves": 2,
                  "images_executed": ["linked", "control"],
                  "correctness": {key: True for key in
                                  ("all_true", "checked_before_work", "checked_after_reload",
                                   "checked_after_work", "gate_a_bitwise", "gate_b_linked",
                                   "gate_b_control", "gate_c_lse")}}
        run = {"returncode": 0, "stdout": "", "stderr": ""}
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(ev, "RESULTS", Path(tmp)), patch.object(cp, "artifact_provenance", return_value={}), \
                patch.dict(cp.os.environ, {"PIPELINK_CUDA_HOME": "/cuda"}), \
                patch.object(cp, "run_slot", return_value=(run, result)) as execute, \
                patch.object(ev, "sanitizer_count", return_value=0):
            receipt = cp.stage_sanitize("W1-8k", [1, 3])
        self.assertEqual(receipt["status"], "PASS")
        self.assertEqual(receipt["sanitizer_protocol"], "paired-reload-v2-check")
        self.assertEqual(execute.call_count, len(ev.SANITIZER_TOOLS))
        self.assertEqual([call.args[0][2] for call in execute.call_args_list], list(ev.SANITIZER_TOOLS))
        for call in execute.call_args_list:
            self.assertIn("--action=paired-check", call.args[0])
            self.assertNotIn("--action=paired", call.args[0])
        self.assertEqual(receipt["coverage"]["measured_worlds"], [2])

    def test_checkpoint_retains_completed_failure_without_promotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sanitize.partial.json"
            identity = {"build_generation": "same"}
            command = ["compute-sanitizer", "--tool", "racecheck", "slot"]
            outcome = {"passed": False, "command": {"command": command, "returncode": 99,
                                                    "stdout": "native proof", "stderr": "hazard"}}
            checkpoint = ev.SanitizerCheckpoint(path, identity)
            checkpoint.save("racecheck", outcome)
            resumed = ev.SanitizerCheckpoint(path, identity, resume=True)
            self.assertEqual(resumed.get("racecheck", command), outcome)
            self.assertEqual(json.loads(path.read_text())["status"], "INCOMPLETE")
            self.assertEqual((Path(tmp) / "sanitize.partial.raw/racecheck.stdout").read_text(), "native proof")
            with self.assertRaises(ev.EvidenceError):
                ev.SanitizerCheckpoint(path, {"build_generation": "different"}, resume=True)
            with self.assertRaises(ev.EvidenceError):
                resumed.get("racecheck", ["different", "command"])

    def test_published_scope_and_stock_commands(self):
        ev.load_cells(HERE / "sm90-baseline/cells.json")
        self.assertEqual(len(ev.PRIMARY_CELLS), 17)
        self.assertEqual(sum(ev.stock_supported(ev.CELLS[cell]) for cell in ev.PRIMARY_CELLS), 6)
        self.assertFalse(any("dense" in cell for cell in ev.PRIMARY_CELLS))
        command = ev.paired_command("W1-8k", "selected", "stock")
        self.assertIn("--mode=linked", command)
        self.assertTrue(any(arg.startswith("--stock-cubin=") for arg in command))
        self.assertTrue(any(arg.startswith("--control-cubin=") for arg in command))

    def test_one_attempt_per_comparison_threshold_only_r1(self):
        ev.load_cells(HERE / "sm90-baseline/cells.json")
        for r1_ratio, admitted, expected in ((1.0, True, "PASS"), (0.96, True, "MISS"), (1.1, False, "INVALID")):
            with self.subTest(ratio=r1_ratio, admitted=admitted), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                root = Path(tmp)
                captured = []
                def execute(command, **kwargs):
                    request = json.loads(kwargs["input"])
                    captured.append(request)
                    kind = request["reference_kind"]
                    return {"admitted": admitted, "payload": {
                        "case_id": "W1-8k", "reference_kind": kind,
                        "correctness": {key: True for key in (
                            "all_true", "checked_before_warmup", "checked_after_reload", "gate_a_bitwise",
                            "gate_b_linked", "gate_b_control", "gate_c_lse", "gate_stock")},
                        "pair": {"protocol": "paired-reload-v2", "repetitions": 10,
                                 "ratio_onetu_over_linked": r1_ratio if kind == "control" else 0.1,
                                 "blocks": [{} for _ in range(8)]}}}
                stack.enter_context(patch.object(ev, "RESULTS", root))
                stack.enter_context(patch.object(ev, "require_prior"))
                stack.enter_context(patch.object(ev, "require_inputs"))
                stack.enter_context(patch.object(ev, "build_generation", return_value="new"))
                stack.enter_context(patch.object(ev, "sha256", return_value="hash"))
                runner = stack.enter_context(patch("evaluator.harness.parity.fixed_count.run_native_pair", side_effect=execute))
                if expected == "PASS":
                    ev.paired_definitive([("W1-8k", "selected")], [3], 1230)
                else:
                    with self.assertRaises(ev.EvidenceError):
                        ev.paired_definitive([("W1-8k", "selected")], [3], 1230)
                result = json.loads((root / "sm90-definitive.json").read_text())["cases"][0]
                self.assertEqual(runner.call_count, 2)
                self.assertEqual([r["reference_kind"] for r in captured], ["control", "stock"])
                self.assertEqual(result["ratios"]["R1"]["verdict"], expected)
                self.assertEqual(result["ratios"]["R3"]["verdict"], "DIAGNOSTIC" if admitted else "INVALID")
                text = (root / "sm90-definitive.md").read_text()
                # The paired table reports blocks and calls per batch, never legacy pair counts.
                self.assertIn("| W1-8k | selected | R1 | ", text)
                self.assertIn(" | 8 | 10 | ", text)
                for call in runner.call_args_list:
                    self.assertEqual(call.kwargs["gpus"], [3])


class Prerequisites(unittest.TestCase):
    def test_single_exact_gate_and_absent_only_sanitizer_fallback(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            cells = root / "cells.json"
            cells.write_text('{"cells": []}')
            slot = root / "slot"
            slot.write_bytes(b"current executable")
            paths = {"slot": str(slot)}
            hashes = {"slot": ev.sha256(slot)}
            stack.enter_context(patch.object(ev, "RESULTS", root))
            stack.enter_context(patch.object(ev, "CELLS_JSON", cells))
            stack.enter_context(patch.object(ev, "build_generation", return_value="new"))
            stack.enter_context(patch.object(ev, "check_images", return_value={"linked": {"inputs": paths}}))
            representative = {"case_id": "short", "pipeline": "selected", "status": "PASS",
                              "build_generation": "new", "input_paths": paths, "binary_sha256": hashes,
                              "sanitizer_protocol": "paired-reload-v2-check"}
            def save(stage, rows):
                (root / f"sm90-{stage}.json").write_text(json.dumps({"cells_document": {"cells": []}, "cases": rows}))
            save("sanitize", [representative])
            ev.require_prior("sanitize", [("long", "selected")])
            save("gate", [representative])
            with self.assertRaises(ev.EvidenceError):
                ev.require_prior("gate", [("long", "selected")])
            save("sanitize", [representative, {**representative, "case_id": "long", "status": "FAIL"}])
            with self.assertRaises(ev.EvidenceError):
                ev.require_prior("sanitize", [("long", "selected")])
            save("sanitize", [{**representative, "build_generation": "old"}])
            with self.assertRaises(ev.EvidenceError):
                ev.require_prior("sanitize", [("long", "selected")])
            save("sanitize", [{**representative, "sanitizer_protocol": "legacy"}])
            with self.assertRaises(ev.EvidenceError):
                ev.require_prior("sanitize", [("long", "selected")])

    def test_cp_requires_exact_world_and_current_provenance(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(ev, "RESULTS", Path(tmp)), \
                patch.object(cp, "artifact_provenance", return_value={"build_generation": "new", "binary_sha256": {"slot": "hash"}}):
            path = Path(tmp) / "sm90-cp-sanitize-tp2-W1-32k.json"
            receipt = {"status": "PASS", "cell": "W1-32k", "world": 2,
                       "build_generation": "new", "binary_sha256": {"slot": "hash"},
                       "sanitizer_protocol": "paired-reload-v2-check"}
            path.write_text(json.dumps(receipt))
            cp.require_prior("sanitize", "W1-32k", 2)
            with self.assertRaises(ev.EvidenceError):
                cp.require_prior("sanitize", "W1-32k", 4)
            path.write_text(json.dumps({**receipt, "build_generation": "old"}))
            with self.assertRaises(ev.EvidenceError):
                cp.require_prior("sanitize", "W1-32k", 2)
            tp2_representative = Path(tmp) / "sm90-cp-sanitize-tp2-W1-8k.json"
            tp2_representative.write_text(json.dumps({**receipt, "cell": "W1-8k", "coverage": cp.sanitizer_scope(2)}))
            # TP4 needs its own measured W1-8k record; the TP2 run does not stand in for it.
            with self.assertRaises(ev.EvidenceError):
                cp.require_prior("sanitize", "W1-32k", 4)
            representative = Path(tmp) / "sm90-cp-sanitize-tp4-W1-8k.json"
            representative.write_text(json.dumps({**receipt, "cell": "W1-8k", "world": 4,
                                                  "coverage": cp.sanitizer_scope(4)}))
            cp.require_prior("sanitize", "W1-32k", 4)
            # A failing receipt for the exact row overrides representative coverage.
            with self.assertRaises(ev.EvidenceError):
                cp.require_prior("sanitize", "W1-32k", 2)
            with self.assertRaises(ev.EvidenceError):
                cp.require_prior("gate", "W1-32k", 4)
            representative.write_text(json.dumps({**receipt, "cell": "W1-8k", "world": 4}))
            with self.assertRaises(ev.EvidenceError):
                cp.require_prior("sanitize", "W1-32k", 4)

    def test_binary_receipt_does_not_hash_semantic_metadata(self):
        with patch.object(ev, "sha256", return_value="hash") as digest:
            self.assertEqual(ev.binary_hashes({"slot": "/slot", "link": "/link.json",
                                               "cells": "/cells.json", "selection": "/selection.i32"}),
                             {"slot": "hash"})
            digest.assert_called_once_with(Path("/slot"))


if __name__ == "__main__":
    unittest.main()
