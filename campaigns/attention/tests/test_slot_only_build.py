"""Exercise output/generation orchestration with compilation mocked out."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("attention_build", HERE / "build.py")
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


class SlotOnlyBuild(unittest.TestCase):
    def test_external_log_location_preserves_record_and_command_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "isolated" / "compile.log"
            record = build.command_record(
                [sys.executable, "-c", "pass"], HERE, log, os.environ.copy())
            self.assertEqual((HERE / record["log"]).resolve(), log.resolve())
            self.assertEqual(record["returncode"], 0)
            with self.assertRaises(build.BuildError):
                build.command_record(
                    [sys.executable, "-c", "raise SystemExit(7)"], HERE, log, os.environ.copy())

    def test_isolated_build_preserves_images_and_invalidates_before_compile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cuda = root / "cuda"
            (cuda / "bin").mkdir(parents=True)
            (cuda / "bin/nvcc").touch()
            out = root / "staged-out"
            images = [out / "links/d128-sm90-c/cp/pipeline.cubin",
                      out / "controls/d128-sm90-c/cp.cubin",
                      out / "stock/sm90/stock.cubin"]
            for image in images:
                image.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(b"retained image")
            (out / "build-report.json").write_text(json.dumps({"build_generation": "remote-old"}))
            commands = []
            def compile_slots(items, jobs, env):
                report = json.loads((out / "build-report.json").read_text())
                self.assertNotEqual(report["build_generation"], "remote-old")
                self.assertEqual(report["parent_build_generation"], "remote-old")
                self.assertTrue(report["slot_only"])
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
                commands.extend(command for command, _, _ in items)
                return []
            with patch.dict(os.environ, {"PIPELINK_CUDA_HOME": str(cuda)}), \
                    patch.object(sys, "argv", ["build.py", "--slot-only", "--out-root", str(out), "--jobs", "2"]), \
                    patch.object(build, "pool_run", side_effect=compile_slots), \
                    patch.object(build, "build_stock") as stock, \
                    patch.object(build, "pack_stage") as pack, \
                    patch.object(build, "publish_artifacts") as publish:
                self.assertEqual(build.main(), 0)
                stock.assert_not_called()
                pack.assert_not_called()
                publish.assert_not_called()
            self.assertEqual(len(commands), 2)
            self.assertEqual({Path(command[command.index("-o") + 1]).name for command in commands},
                             {"attention-slot-sm90", "attention-slot-cp-sm90"})
            self.assertTrue(all(Path(command[command.index("-o") + 1]).parent == out / "bin" for command in commands))
            for image in images:
                self.assertEqual(image.read_bytes(), b"retained image")


if __name__ == "__main__":
    unittest.main()
