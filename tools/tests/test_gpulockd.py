import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import gpulockd


def _held(gpu: int) -> gpulockd.HeldGpu:
    return gpulockd.HeldGpu(
        identity=gpulockd.GpuIdentity(gpu, f"GPU-{gpu}", str(gpu), "580"), sm_mhz=2850,
        cuda_device=0, context=None, policy_token="0" * 64, started_at="t",
    )


class ViolationTests(unittest.TestCase):
    def sample(self, clock, reasons, utilization, tolerance=15):
        def smi(arguments, **_kwargs):
            if "--query-gpu=index,utilization.gpu" in arguments:
                return f"5, {utilization}\n"
            return f"5, {clock} MHz, {reasons:#x}\n"
        with tempfile.TemporaryDirectory() as directory:
            manager = gpulockd.LockManager(Path(directory), {}, 2850, tolerance)
            held = _held(5)
            manager.held = [held]
            with mock.patch.object(gpulockd, "_nvidia_smi", side_effect=smi):
                manager._sample()
            if reasons & gpulockd.SW_POWER_CAP_REASON:
                import json
                marker = json.loads((Path(directory) / "gpu5.json").read_text())
                self.assertEqual(marker["sw_power_cap_samples"], 1)
                self.assertEqual(marker["violations"], held.violations)
            return held

    def test_sw_cap_with_held_clock_is_informational(self):
        held = self.sample(2842, 0x4, 100)
        self.assertEqual(held.violations, 0)
        self.assertEqual(held.sw_power_cap_samples, 1)

    def test_sw_cap_with_loaded_deviation_is_violation(self):
        self.assertEqual(self.sample(2800, 0x4, 100).violations, 1)

    def test_hardware_slowdown_is_always_violation(self):
        for reason in (0x8, 0x40, 0x80):
            for utilization in (0, 100):
                with self.subTest(reason=reason, utilization=utilization):
                    self.assertEqual(self.sample(2850, reason, utilization).violations, 1)

    def test_idle_deviation_is_not_violation(self):
        self.assertEqual(self.sample(300, 0x0, 0).violations, 0)
        self.assertEqual(self.sample(300, 0x4, 0).violations, 0)

    def test_loaded_clock_is_judged_against_configured_band(self):
        self.assertEqual(self.sample(2820, 0, 100, 15).violations, 1)
        self.assertEqual(self.sample(2820, 0, 100, 45).violations, 0)
        self.assertEqual(self.sample(2800, 0, 100, 45).violations, 1)
        self.assertEqual(self.sample(2865, 0, 100).violations, 0)
        self.assertEqual(self.sample(2866, 0, 100).violations, 1)

    def test_tolerance_comes_from_environment_with_15_mhz_default(self):
        self.assertEqual(gpulockd._clock_tolerance({"PIPELINK_LOCK_TOLERANCE_MHZ": "45"}), 45)
        from tools import pipelink_env
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(pipelink_env.load(directory)["PIPELINK_LOCK_TOLERANCE_MHZ"], "15")


if __name__ == "__main__":
    unittest.main()
