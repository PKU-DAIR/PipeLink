from __future__ import annotations

from contextlib import nullcontext
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from evaluator.harness.parity.example import compare, time_pair


class FakeTimer:
    def __init__(self, name, rates, events):
        self.name = name
        self.rates = iter(rates)
        self.events = events

    def time(self, repetitions):
        self.events.append((self.name, "time", repetitions))
        rate = next(self.rates)
        return rate * repetitions


class TimePairTests(unittest.TestCase):
    def test_plain_warmup_is_excluded_and_requested_minimum_is_preserved(self):
        events = []
        linked = FakeTimer("linked", [10000.0, 2.0, 2.0, *([2.0] * 16)], events)
        control = FakeTimer("one_tu", [0.001, 1.0, 1.0, *([1.0] * 16)], events)
        result = time_pair(linked, control, repetitions=23)

        self.assertEqual(len(result["blocks"]), 8)
        for block in result["blocks"]:
            self.assertEqual(block["order"].count("A"), 2)
            self.assertEqual(block["order"].count("B"), 2)
            self.assertEqual(block["ratio_onetu_over_linked"], 0.5)
            self.assertEqual(block["milliseconds"], [46.0 if label == "A" else 23.0 for label in block["order"]])
        self.assertEqual(result["ratio_onetu_over_linked"], 0.5)
        self.assertEqual(result["repetitions"], 23)
        self.assertEqual(result["status"], "PASS")
        self.assertNotIn("protocol", result)
        self.assertNotIn("statistic", result)

    def test_median_of_block_ratios_not_pooled_times_or_position_ratios(self):
        # Distinct scales make pooling all elapsed times different from taking
        # the median; unequal positions make averaging individual A/B ratios wrong.
        ratios = [8.0, 1.0, 3.0, 2.0, 7.0, 4.0, 6.0, 100.0]
        linked_rates = [10000.0, 20.0]
        control_rates = [0.001, 20.0]
        for scale, ratio in enumerate(ratios, start=1):
            linked_rates.extend([0.5 * scale * ratio, 3.5 * scale * ratio])
            control_rates.extend([1.0 * scale, 3.0 * scale])
        events = []
        result = time_pair(
            FakeTimer("linked", linked_rates, events),
            FakeTimer("one_tu", control_rates, events),
            repetitions=20,
        )

        self.assertEqual([block["ratio_onetu_over_linked"] for block in result["blocks"]], [1 / ratio for ratio in ratios])
        # An even-sample median of reciprocals need not be the reciprocal of
        # the median. Score the per-block speedups directly.
        self.assertAlmostEqual(result["ratio_onetu_over_linked"], (1 / 4 + 1 / 6) / 2)

    def test_calibration_converges_using_the_fastest_batch(self):
        events = []
        linked = FakeTimer("linked", [0.25, 0.25, 0.125, 0.125, *([0.125] * 16)], events)
        control = FakeTimer("one_tu", [0.125, 0.125, 0.1, 0.1, *([0.1] * 16)], events)
        result = time_pair(linked, control)

        # The fastest calibrated rate is 0.1 ms: ceil(15.625 / 0.1).
        self.assertEqual(result["repetitions"], 157)
        for block in result["blocks"]:
            self.assertEqual(
                block["milliseconds"],
                [157 * (0.125 if label == "A" else 0.1) for label in block["order"]],
            )
        self.assertEqual(result["status"], "PASS")

    def test_each_scored_batch_must_meet_the_inclusive_duration_floor(self):
        for last_batch, status in ((12.5, "PASS"), (np.nextafter(12.5, 0.0), "BATCH_TOO_SHORT")):
            with self.subTest(last_batch=last_batch):
                events = []
                linked_rates = [1.0, 20.0, *([0.625] * 15), last_batch / 20]
                control_rates = [1.0, 20.0, *([0.625] * 16)]
                result = time_pair(
                    FakeTimer("linked", linked_rates, events),
                    FakeTimer("one_tu", control_rates, events),
                    repetitions=20,
                )

                self.assertEqual(result["repetitions"], 20)
                self.assertEqual(result["status"], status)

    def test_calibration_aborts_above_the_repetition_cap(self):
        events = []
        with self.assertRaisesRegex(RuntimeError, "repetition cap"):
            time_pair(
                FakeTimer("linked", [1.0, 1e-9], events),
                FakeTimer("one_tu", [1.0, 1e-9], events),
            )
        self.assertLessEqual(max(event[2] for event in events), 1 << 20)

    def test_calibration_aborts_after_24_nonconverging_rounds(self):
        class NonconvergingTimer:
            def __init__(self):
                self.calls = 0

            def time(self, repetitions):
                self.calls += 1
                return 15.0

        linked, control = NonconvergingTimer(), NonconvergingTimer()
        with self.assertRaisesRegex(RuntimeError, "24 rounds"):
            time_pair(linked, control)
        self.assertEqual((linked.calls, control.calls), (25, 25))


class InstanceTimer:
    """Synthetic image cost multiplied by its current module-load slot bias."""

    def __init__(self, cost, *, first):
        self.cost = cost
        self.work = []
        self.load(first=first)

    def load(self, *, first):
        self.first = first
        self.cold = True

    def time(self, repetitions):
        self.work.append((self.cold, repetitions))
        if self.cold:
            self.cold = False
            rate = 10000.0 if self.first else 0.001
        else:
            rate = self.cost * (4 if self.first else 1)
        return rate * repetitions


class ReloadTimePairTests(unittest.TestCase):
    orders = ("ABBA", "BAAB", "BAAB", "ABBA", "BAAB", "ABBA", "ABBA", "BAAB")

    def test_reload_cancels_instance_bias_but_retains_image_cost_and_exact_work(self):
        for image_speedup in (0.5, 1.0, 2.0):
            for linked_first in (False, True):
                with self.subTest(image_speedup=image_speedup, linked_first=linked_first):
                    linked = InstanceTimer(4.0, first=linked_first)
                    control = InstanceTimer(4.0 * image_speedup, first=not linked_first)
                    before_reload = []

                    def reload():
                        before_reload.append((list(linked.work), list(control.work)))
                        linked.load(first=not linked_first)
                        control.load(first=linked_first)

                    result = time_pair(linked, control, repetitions=10, reload=reload)

                    self.assertEqual(len(before_reload), 1)
                    for work in before_reload[0]:
                        self.assertEqual(sum(n for warm, n in work if warm), 8)
                        self.assertEqual(sum(n for warm, n in work if not warm), 8 * 10)
                    work = linked.work + control.work
                    self.assertEqual([n for warm, n in work if warm], [8] * 4)
                    self.assertEqual([n for warm, n in work if not warm], [10] * (8 * 4))
                    self.assertEqual(result["repetitions"], 10)
                    self.assertEqual(tuple(block["order"] for block in result["blocks"]), self.orders)
                    first_bias = 0.25 if linked_first else 4
                    self.assertEqual(
                        [block["ratio_onetu_over_linked"] for block in result["blocks"]],
                        [image_speedup * first_bias] * 4 + [image_speedup / first_bias] * 4,
                    )
                    self.assertEqual(result["ratio_onetu_over_linked"], image_speedup)
                    self.assertEqual(result["status"], "PASS")
                    self.assertEqual(result["protocol"], "paired-reload-v2")

    def test_reload_uses_geometric_mean_of_half_medians_not_pooled_estimators(self):
        speedups = [1.0, 2.0, 3.0, 100.0, 4.0, 8.0, 12.0, 400.0]
        linked_rates, control_rates = [10000.0], [0.001]
        for index, speedup in enumerate(speedups):
            if index == 4:
                linked_rates.append(0.001)
                control_rates.append(10000.0)
            scale = index + 1
            linked_rates.extend([2.0 * scale, 14.0 * scale])
            control_rates.extend([4.0 * scale * speedup, 12.0 * scale * speedup])
        events = []

        result = time_pair(
            FakeTimer("linked", linked_rates, events),
            FakeTimer("one_tu", control_rates, events),
            repetitions=10, reload=lambda: None,
        )

        self.assertEqual([block["ratio_onetu_over_linked"] for block in result["blocks"]], speedups)
        # The half medians are 2.5 and 10; the eight-block median would be 6.
        self.assertEqual(result["ratio_onetu_over_linked"], 5.0)

    def test_fixed_count_accepts_short_positive_batches_without_calibration(self):
        events = []
        result = time_pair(
            FakeTimer("linked", [0.001] * 18, events),
            FakeTimer("one_tu", [0.002] * 18, events),
            repetitions=10, reload=lambda: None,
        )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["repetitions"], 10)
        self.assertEqual(result["ratio_onetu_over_linked"], 2.0)
        self.assertTrue(all(max(block["milliseconds"]) < 12.5 for block in result["blocks"]))

    def test_fixed_count_rejects_nonphysical_elapsed_values_in_both_halves(self):
        for bad in (0.0, -1.0, float("inf"), float("nan")):
            for bad_position in (0, 16, 31):
                with self.subTest(value=bad, position=bad_position):
                    rates = {"A": [0.001], "B": [0.001]}
                    for index, order in enumerate(self.orders):
                        if index == 4:
                            for arm_rates in rates.values():
                                arm_rates.append(0.001)
                        for position, label in enumerate(order):
                            duration = bad if index * 4 + position == bad_position else 1.0
                            rates[label].append(duration / 10)
                    events = []
                    with self.assertRaises(ValueError):
                        time_pair(FakeTimer("linked", rates["A"], events),
                                  FakeTimer("one_tu", rates["B"], events),
                                  repetitions=10, reload=lambda: None)

    def test_callback_failure_propagates_without_second_warmup_or_measurement(self):
        linked = InstanceTimer(4.0, first=True)
        control = InstanceTimer(8.0, first=False)
        failure = RuntimeError("module reload failed")
        calls = []

        def reload():
            calls.append(True)
            raise failure

        with self.assertRaises(RuntimeError) as raised:
            time_pair(linked, control, repetitions=10, reload=reload)

        self.assertIs(raised.exception, failure)
        self.assertEqual(len(calls), 1)
        work = linked.work + control.work
        self.assertEqual([n for warm, n in work if warm], [8] * 2)
        self.assertEqual([n for warm, n in work if not warm], [10] * (4 * 4))

    def test_sampled_pair_forwards_reload_through_real_timing_wrapper(self):
        linked = InstanceTimer(4.0, first=True)
        control = InstanceTimer(8.0, first=False)
        calls = []

        def reload():
            calls.append(True)
            linked.load(first=False)
            control.load(first=True)

        sampler = SimpleNamespace(samples=lambda: [])
        with (
            patch("evaluator.harness.parity.telemetry.ClockSampler", return_value=nullcontext(sampler)),
            patch("evaluator.harness.parity.example._sm_clocks", return_value={"sm": "1500", "max_sm": "1500"}),
            patch("evaluator.harness.parity.example._gpu_pids", return_value=[]),
            patch.dict("os.environ", {"PIPELINK_LOCK_STATE_DIR": ""}),
        ):
            result = time_pair(linked, control, repetitions=10, gpu=1, sample_clocks=True, reload=reload)

        self.assertEqual(len(calls), 1)
        self.assertEqual(result["ratio_onetu_over_linked"], 2.0)
        self.assertEqual(result["protocol"], "paired-reload-v2")
        self.assertEqual(result["repetitions"], 10)
        self.assertEqual(len(result["blocks"]), 8)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["foreign_pids"], [])
        self.assertEqual(set(result["telemetry"]), {"A", "B"})
        self.assertEqual([n for warm, n in linked.work + control.work if not warm], [10] * (8 * 4))


class CompareTests(unittest.TestCase):
    def test_inclusive_relative_and_absolute_tolerance_boundaries(self):
        actual = np.array([1.25, np.nextafter(1.25, np.inf), 0.125, -2.375])
        expected = np.array([1.0, 1.0, 0.0, -2.0])
        result = compare(actual, expected, rtol=0.125, atol=0.125)

        self.assertFalse(result["allclose"])
        self.assertFalse(result["bitwise"])
        self.assertEqual(result["mismatches"], 1)
        self.assertEqual(result["max_abs"], 0.375)

    def test_bitwise_distinguishes_signed_zero_despite_numeric_equality(self):
        result = compare(np.array([0.0]), np.array([-0.0]), rtol=0, atol=0)

        self.assertTrue(result["allclose"])
        self.assertFalse(result["bitwise"])
        self.assertEqual(result["mismatches"], 0)
        self.assertEqual(result["max_abs"], 0.0)

    def test_paired_nan_is_numeric_failure_even_when_bitwise_identical(self):
        actual = np.array([math.nan, 2.0])
        result = compare(actual, actual.copy(), rtol=0.125, atol=0.125)

        self.assertFalse(result["allclose"])
        self.assertTrue(result["bitwise"])
        self.assertEqual(result["mismatches"], 1)
        self.assertTrue(math.isnan(result["max_abs"]))

    def test_unpaired_nan_cannot_be_hidden_by_tolerance(self):
        result = compare(
            np.array([math.nan, 1.0, 2.0]), np.array([0.0, math.nan, 2.0]),
            rtol=1.0, atol=1.0,
        )

        self.assertFalse(result["allclose"])
        self.assertFalse(result["bitwise"])
        self.assertEqual(result["mismatches"], 2)
        self.assertTrue(math.isnan(result["max_abs"]))
