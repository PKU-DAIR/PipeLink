"""CPU-only tests of the MK paired-protocol interfaces (no CUDA execution)."""
from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_parity as parity
import launch


class PairedLaunchTests(unittest.TestCase):
    def test_two_manifest_derived_arms_and_independent_runner_override(self):
        def facts(pipeline, image, **kwargs):
            return dict(cubin=Path(f"/images/{pipeline}-{image}.cubin"), kernel=f"kernel_{pipeline}",
                        dyn=1234 if pipeline == "cute" else 5678, threads=320, cluster=[2, 1, 1],
                        params_bytes=256, params_align=128, residue_offset=128, residue_bytes=128,
                        owner_id=7, dtypes_digest="a" * 64, owner_contract="b" * 64,
                        opcode_map="0:0,1:1")

        with patch.object(launch, "launch_facts", side_effect=facts):
            command = launch.paired_command("cute", "linked", "a", "linked", "/bundle", "/out", 16,
                                            runner=Path("/isolated/mk_runner"),
                                            provider=Path("/existing/provider.so"))
        split = command.index("--reference")
        arms = [dict(zip(command[1:split:2], command[2:split:2])),
                dict(zip(command[split + 1::2], command[split + 2::2]))]
        self.assertEqual(command[0], "/isolated/mk_runner")
        self.assertEqual([arm["--dyn-smem"] for arm in arms], ["1234", "5678"])
        self.assertEqual([arm["--kernel"] for arm in arms], ["kernel_cute", "kernel_a"])
        for arm in arms:
            self.assertEqual(arm["--provider"], "/existing/provider.so")
            self.assertEqual(arm["--mode"], "pair")
            self.assertEqual(arm["--pos-id"], "16")
            self.assertEqual(arm["--threads"], "320")
            self.assertEqual(arm["--cluster"], "2,1,1")
            self.assertNotIn("--bench-seconds", arm)
            self.assertNotIn("--bench-batch-tokens", arm)

    def test_pair_rejects_exploratory_mono_and_legacy_mode(self):
        for image, mode in (("mono", "pair"), ("linked", "bench")):
            with self.subTest(image=image, mode=mode), self.assertRaises(ValueError):
                launch.paired_command("a", image, "a", "onetu", "/bundle", "/out", 16, mode=mode)

    def test_launcher_pair_owns_shared_window_instead_of_plain_subprocess(self):
        from evaluator.harness.parity import fixed_count

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "snapshots"
            argv = ["launch.py", "--pipeline", "cute", "--image", "linked",
                    "--reference-pipeline", "a", "--reference-image", "linked",
                    "--bundle", "/bundle", "--out", str(out), "--gpu", "2",
                    "--mode", "pair", "--pos-id", "16"]
            with patch.object(sys, "argv", argv), \
                    patch.dict(launch.os.environ, {"PIPELINK_LOCK_SM_MHZ": "1155"}, clear=True), \
                    patch.object(launch, "paired_command", return_value=["/native"]) as command, \
                    patch.object(fixed_count, "run_native_pair", return_value={"admitted": True}) as native, \
                    patch.object(launch.subprocess, "call") as plain, \
                    patch("builtins.print"):
                self.assertEqual(launch.main(), 0)
            plain.assert_not_called()
            command.assert_called_once()
            native.assert_called_once()
            self.assertEqual(native.call_args.kwargs["gpus"], [2])
            self.assertEqual(native.call_args.kwargs["out_dir"], out / "execution")


class PairedEvidenceTests(unittest.TestCase):
    def test_maintained_scope_and_sanitizer_representatives(self):
        self.assertEqual(len(parity.SUITES["reference"]["cases"]), 5)
        self.assertEqual(len(parity.SUITES["cute"]["cases"]), 6)
        cases = parity.SUITES["reference"]["cases"] + parity.SUITES["cute"]["cases"]
        self.assertEqual(sum(case["kind"] == "seam-parity" for case in cases), 8)
        self.assertEqual(sum(case["kind"] == "cross-library" for case in cases), 3)
        self.assertNotIn("tilelang", {case["pipeline"] for case in cases})
        for suite, expected in (("reference", 3), ("cute", 4)):
            with patch.object(parity, "CASES", parity.SUITES[suite]["cases"]), \
                    patch.object(parity, "parse_scalars", return_value={"pos_id_first": "16"}):
                representatives = parity.sanitizer_pairs()
            self.assertEqual(len(representatives), expected)
            self.assertEqual({case["bundle"] for case in representatives}, {"p16-n2", "p1024-n8-red"})

    def test_only_seam_thresholds_and_admission_before_ratio(self):
        seam = parity.SUITES["reference"]["cases"][0]
        cross = parity.SUITES["cute"]["cases"][-1]
        execution = {"admitted": True, "payload": {"pair": {"ratio_onetu_over_linked": .96}}}
        numeric = {"status": "PASS"}
        self.assertEqual(parity.collect_case(seam, execution, numeric)["gate_outcome"], "MISS")
        result = parity.collect_case(cross, execution, numeric)
        self.assertEqual(result["gate_outcome"], "NOT_GATED")
        self.assertIsNone(result["ratio_floor"])
        self.assertEqual(result["ratio_onetu_over_linked"], .96)
        execution["payload"]["pair"]["ratio_onetu_over_linked"] = .97
        self.assertEqual(parity.collect_case(seam, execution, numeric)["gate_outcome"], "PASS")
        for admitted, numeric in ((False, {"status": "PASS"}), (True, None)):
            execution["admitted"] = admitted
            result = parity.collect_case(seam, execution, numeric)
            self.assertEqual(result["status"], "EXECUTION_FAIL")
            self.assertIsNone(result["ratio_onetu_over_linked"])

    def test_current_paired_sanitizer_identity_and_all_tools_required(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(parity, "CASES", ({"case_id": "case"},)), \
                patch.object(parity, "executable_identity", return_value={"runner": "current"}), \
                patch.object(parity, "pair_coverage", return_value={"images": "exact", "bundle": "exact"}):
            path = Path(directory) / "receipt.json"
            receipt = {
                "protocol": parity.PROTOCOL, "action": "sanitize-pair", "passed": True,
                "executables": {"runner": "current"},
                "records": [{"coverage": {"images": "exact", "bundle": "exact"}, "tool": tool,
                             "passed": True, "returncode": 0, "count": 0,
                             "numeric": {"status": "PASS"}} for tool in parity.TOOLS],
            }
            path.write_text(json.dumps(receipt))
            self.assertEqual(parity.require_pair_sanitizers(path)["status"], "PASS")
            receipt["records"][0]["count"] = 1
            path.write_text(json.dumps(receipt))
            with self.assertRaises(parity.EvidenceError):
                parity.require_pair_sanitizers(path)
            receipt["records"][0]["count"] = 0
            receipt["executables"]["runner"] = "old"
            path.write_text(json.dumps(receipt))
            with self.assertRaises(parity.EvidenceError):
                parity.require_pair_sanitizers(path)
            receipt["executables"]["runner"] = "current"
            receipt["records"].pop()
            path.write_text(json.dumps(receipt))
            with self.assertRaises(parity.EvidenceError):
                parity.require_pair_sanitizers(path)

    def test_main_calls_native_once_on_miss_without_legacy_slots(self):
        case = parity.SUITES["reference"]["cases"][0]
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            stack.enter_context(patch.object(parity, "CASES", (case,)))
            stack.enter_context(patch.object(parity, "GPU", 1))
            stack.enter_context(patch.object(parity, "OUTPUT_ROOT", root, create=True))
            stack.enter_context(patch.object(parity, "RESULTS_PATH", root / "results.json", create=True))
            stack.enter_context(patch.object(parity, "RESULT_MARKDOWN_PATH", root / "RESULT.md", create=True))
            stack.enter_context(patch.dict(parity.os.environ, {"PIPELINK_LOCK_SM_MHZ": "1155"}, clear=True))
            stack.enter_context(patch.object(sys, "argv", ["run_parity.py", "--output-root", str(root)]))
            for name in ("select_suite", "case_config", "require_p2_pass", "require_pair_sanitizers", "write_results"):
                stack.enter_context(patch.object(parity, name, return_value={}))
            stack.enter_context(patch.object(parity, "require_sanitized_images", return_value={case["case_id"]: {}}))
            stack.enter_context(patch.object(parity, "physical_images", return_value={"identities": "stable"}))
            stack.enter_context(patch.object(parity, "executable_identity", return_value={"runner": "stable"}))
            stack.enter_context(patch.object(parity, "pair_command", return_value=["/native", "--mode", "pair"]))
            stack.enter_context(patch.object(parity, "check_native_payload", return_value={"status": "PASS"}))
            native = stack.enter_context(patch.object(parity, "run_native_pair", return_value={
                "admitted": True, "command_returncode": 0,
                "payload": {"pair": {"ratio_onetu_over_linked": .5}},
            }))
            self.assertEqual(parity.main(), 0)  # performance MISS is not an execution failure
            native.assert_called_once()
            self.assertEqual(native.call_args.args[0], ["/native", "--mode", "pair"])
            self.assertEqual(native.call_args.kwargs["gpus"], [1])
            self.assertEqual(native.call_args.kwargs["clock_mhz"], 1155)
            self.assertNotIn("PIPELINK_GPU_WINDOW", native.call_args.kwargs["env"])


class SnapshotPredicateTests(unittest.TestCase):
    def test_existing_compare_predicates_on_every_reload_snapshot(self):
        import numpy as np
        import compare

        case = parity.SUITES["reference"]["cases"][0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, out = root / "bundle", root / "out"
            bundle.mkdir()
            manifest = []
            for name in compare.OUTPUT_NAMES:
                shape = (2, 8, 64) if name in ("k_row", "v_row") else (4,)
                dtype = "f32" if name in ("attn_out_p0", "attn_lse_p0") else "bf16"
                value = np.full(shape, 1.0, dtype="<f4") if dtype == "f32" else np.full(shape, 0x3f80, dtype="<u2")
                filename = f"ref_tok0_{name}.bin"
                value.tofile(bundle / filename)
                manifest.append(f"ref_tok0_{name} {dtype} {len(shape)} {' '.join(map(str, shape))} {filename}")
                for phase in parity.SNAPSHOT_PHASES:
                    for arm in ("candidate", "reference"):
                        destination = out / phase / arm
                        destination.mkdir(parents=True, exist_ok=True)
                        value.tofile(destination / f"tok0_{name}.bin")
            (bundle / "manifest.txt").write_text("\n".join(manifest) + "\n")
            images = {"candidate": {"path": "/candidate"}, "reference": {"path": "/reference"}}
            payload = {"protocol": parity.PROTOCOL, "mode": "pair-check",
                       "numerical": {"status": "PASS", "snapshot_phases": list(parity.SNAPSHOT_PHASES),
                                     "root": str(out), "predicate": "finite-nine+strict-layer0-kv"},
                       "provenance": {"candidate_image": "/candidate", "reference_image": "/reference",
                                      "bundle": str(bundle), "visible_device": 0, "shared_runtime_buffers": True,
                                      "initial_load_order": ["candidate", "reference"],
                                      "reload_load_order": ["reference", "candidate"], "reload_count": 1,
                                      "reload_interval_ns": [1, 2], "pos_id": 16, "embedding_rows": 2,
                                      "queued_probe_repetitions": 2, "provider_sessions_per_arm": 1,
                                      "provider_sessions_persist_through_reload": True}}
            with patch.object(parity, "BUNDLES", root), \
                    patch.object(parity, "physical_images", return_value=images), \
                    patch.object(parity, "parse_scalars", return_value={"ntok": "2"}):
                case = {**case, "bundle": "bundle"}
                result = parity.check_native_payload(case, payload, out, "pair-check")
                self.assertEqual(len(result["snapshots"]), 8)
                corrupt = out / "after_reload" / "reference" / "tok0_k_row.bin"
                values = np.fromfile(corrupt, dtype="<u2")
                values[0] = 0x4000  # finite 2.0 violates the strict reference-1.0 predicate
                values.tofile(corrupt)
                with self.assertRaises(parity.EvidenceError):
                    parity.check_native_payload(case, payload, out, "pair-check")
                values[0] = 0x3f80
                values.tofile(corrupt)
                np.array([np.nan, 1., 1., 1.], dtype="<f4").tofile(
                    out / "final" / "candidate" / "tok0_attn_lse_p0.bin")
                with self.assertRaises(parity.EvidenceError):
                    parity.check_native_payload(case, payload, out, "pair-check")


if __name__ == "__main__":
    unittest.main()
