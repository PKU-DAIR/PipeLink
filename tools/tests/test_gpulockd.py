import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import gpulockd


def _held(gpu: int, exempt: int) -> gpulockd.HeldGpu:
    return gpulockd.HeldGpu(
        identity=gpulockd.GpuIdentity(gpu, f"GPU-{gpu}", str(gpu), "580"), sm_mhz=2850,
        cuda_device=0, context=None, policy_token="0" * 64, started_at="t", exempt_reasons=exempt,
    )


class IdleReasonExemptionTests(unittest.TestCase):
    def test_sw_power_cap_is_exempt_only_when_asserted_idle(self):
        # GPU 6 is still settling (0%, P5, no 0x4) on the first sample and idle on the second; GPU 8 never goes idle.
        samples = iter(["5, 0, P8, 0x4\n6, 0, P5, 0x0\n7, 0, P8, 0x0\n8, 90, P1, 0x4\n", "6, 0, P8, 0x4\n8, 90, P1, 0x4\n"])
        with mock.patch.object(gpulockd, "_nvidia_smi", side_effect=lambda *_a, **_k: next(samples, "8, 90, P1, 0x4\n")), \
                mock.patch.object(gpulockd, "IDLE_WAIT_SECONDS", 0.05), \
                mock.patch.object(gpulockd, "IDLE_POLL_SECONDS", 0.01), mock.patch("sys.stderr"):
            self.assertEqual(gpulockd._idle_reason_exemptions([5, 6, 7, 8]), {5: 0x4, 6: 0x4})

    def test_exemption_never_hides_other_slowdown_reasons(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = gpulockd.LockManager(Path(directory), {}, 2850)
            manager.held = [_held(5, 0x4), _held(6, 0), _held(7, 0x4)]
            loaded = "5, 2842 MHz, 0x4\n6, 2842 MHz, 0x4\n7, 2842 MHz, 0x8\n"
            with mock.patch.object(gpulockd, "_nvidia_smi", return_value=loaded):
                manager._sample()
        self.assertEqual([held.violations for held in manager.held], [0, 1, 1])


if __name__ == "__main__":
    unittest.main()
