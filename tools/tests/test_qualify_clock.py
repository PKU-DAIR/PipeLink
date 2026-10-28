import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import qualify_clock as q


class QualificationTests(unittest.TestCase):
    def test_binary_search_finds_highest_passing_candidate(self):
        clocks = list(range(2000, 900, -10))
        for boundary in (0, 1, 50, len(clocks) - 1, len(clocks)):
            calls = []
            def trial(clock):
                calls.append(clock)
                return boundary < len(clocks) and clock <= clocks[boundary]
            self.assertEqual(q.highest_passing(clocks, trial),
                             boundary if boundary < len(clocks) else None)
            self.assertLessEqual(len(calls), 7)

    def test_supported_clocks_use_highest_memory_clock(self):
        xml = """<nvidia_smi_log><gpu><supported_clocks>
        <supported_mem_clock><value>100 MHz</value><supported_graphics_clock>2100 MHz</supported_graphics_clock></supported_mem_clock>
        <supported_mem_clock><value>200 MHz</value><supported_graphics_clock>1900 MHz</supported_graphics_clock><supported_graphics_clock>2000 MHz</supported_graphics_clock></supported_mem_clock>
        </supported_clocks></gpu></nvidia_smi_log>"""
        self.assertEqual(q.supported_clocks(xml), [2000, 1900])

    def test_summary_excludes_settling_and_separates_sw_cap(self):
        sample = {"seconds": 6, "sm_mhz": 1500, "utilization": 100, "reasons": 4,
                  "power_w": 600, "power_limit_w": 1000, "temperature_c": 60}
        samples = [{**sample, "seconds": 4, "sm_mhz": 800, "reasons": 8}, sample,
                   {**sample, "seconds": 7, "sm_mhz": 300, "utilization": 0, "reasons": 0}]
        summary = q.summarize(samples, 1500, 15)
        self.assertEqual(summary["samples"], 2)
        self.assertEqual(summary["violations"], 0)
        self.assertEqual(summary["sw_power_cap_fraction"], 0.5)
        self.assertEqual(summary["loaded_samples"], 1)
        self.assertTrue(q.sample_violation({**sample, "sm_mhz": 1400}, 1500, 15))
        for reason in (8, 0x40, 0x80):
            self.assertTrue(q.sample_violation({**sample, "reasons": reason, "utilization": 0}, 1500, 15))

    def test_env_update_preserves_all_other_bytes_and_backs_up(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            original = b"# clock\r\nPIPELINK_GPUS=1,2\r\n PIPELINK_LOCK_SM_MHZ =1665\r\n# keep\n"
            path.write_bytes(original)
            q.write_env(path, 1200, "stamp")
            self.assertEqual(path.read_bytes(), original.replace(b" PIPELINK_LOCK_SM_MHZ =1665", b"PIPELINK_LOCK_SM_MHZ=1200"))
            self.assertEqual((path.parent / ".env.backup-stamp").read_bytes(), original)

    def test_contended_trial_is_recorded_invalid_and_retried_after_gpus_free(self):
        instance = object.__new__(q.Qualification)
        instance.record = {"trials": []}
        instance.args = mock.Mock(contention_retries=1)
        outcomes = [q.InvalidTrial("foreign compute processes on selected GPUs: [7]"), None, None]
        with mock.patch.object(instance, "lock"), \
                mock.patch.object(instance, "phase", side_effect=outcomes), \
                mock.patch.object(instance, "check"), \
                mock.patch.object(instance, "unlock") as unlock, \
                mock.patch.object(instance, "stop_workers"), mock.patch.object(instance, "spawn_workers"), \
                mock.patch.object(instance, "wait_until_free") as wait, mock.patch("builtins.print"):
            self.assertTrue(instance.trial(1500, 60, "search"))
        wait.assert_called_once()
        self.assertEqual(unlock.call_count, 2)
        self.assertEqual([t["status"] for t in instance.record["trials"]], ["invalid", "passed"])

    def test_operational_failure_is_not_retried(self):
        instance = object.__new__(q.Qualification)
        instance.record = {"trials": []}
        instance.args = mock.Mock(contention_retries=5)
        with mock.patch.object(instance, "lock"), \
                mock.patch.object(instance, "phase", side_effect=q.InvalidTrial("GPU 1 worker exited or timed out")), \
                mock.patch.object(instance, "unlock"), mock.patch.object(instance, "wait_until_free") as wait, \
                mock.patch("builtins.print"):
            with self.assertRaises(q.InvalidTrial):
                instance.trial(1500, 60, "search")
        wait.assert_not_called()
        self.assertEqual(instance.record["trials"][0]["status"], "invalid")


if __name__ == "__main__":
    unittest.main()
