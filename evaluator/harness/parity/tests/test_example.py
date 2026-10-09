from __future__ import annotations

import math
import unittest

import numpy as np

from evaluator.harness.parity.example import compare, time_pair


class FakeTimer:
    def __init__(self, name, rates, events):
        self.name = name
        self.rates = iter(rates)
        self.events = events

    def launch(self):
        self.events.append((self.name, "launch"))

    def time(self, repetitions):
        self.events.append((self.name, "time", repetitions))
        rate = next(self.rates)
        return rate * repetitions


class TimePairTests(unittest.TestCase):
    def test_balanced_schedule_excludes_warmup_and_reports_both_ratio_directions(self):
        events = []
        linked = FakeTimer("linked", [10000.0, 2.0, 2.0, *([2.0] * 16)], events)
        control = FakeTimer("one_tu", [0.001, 1.0, 1.0, *([1.0] * 16)], events)
        result = time_pair(linked, control, repetitions=20)

        orders = ["ABBA", "BAAB", "BAAB", "ABBA", "BAAB", "ABBA", "ABBA", "BAAB"]
        expected_events = [("linked", "time", 8), ("one_tu", "time", 8)]
        expected_events += [
            (name, "time", n) for n in (1, 16) for name in ("linked", "one_tu")
        ]
        expected_events += [
            ("linked" if label == "A" else "one_tu", "time", 20)
            for order in orders for label in order
        ]
        self.assertEqual(events, expected_events)
        self.assertEqual([block["order"] for block in result["blocks"]], orders)
        for block in result["blocks"]:
            self.assertEqual(block["ratio_linked_over_onetu"], 2.0)
            self.assertEqual(block["ratio_onetu_over_linked"], 0.5)
            self.assertEqual(block["milliseconds"], [40.0 if label == "A" else 20.0 for label in block["order"]])
        self.assertEqual(result["ratio_linked_over_onetu"], 2.0)
        self.assertEqual(result["ratio_onetu_over_linked"], 0.5)
        self.assertEqual(result["repetitions"], 20)
        self.assertEqual(result["status"], "PASS")

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

        self.assertEqual([block["ratio_linked_over_onetu"] for block in result["blocks"]], ratios)
        self.assertEqual(result["ratio_linked_over_onetu"], 5.0)
        self.assertEqual([block["ratio_onetu_over_linked"] for block in result["blocks"]], [1 / ratio for ratio in ratios])
        # An even-sample median of reciprocals need not be the reciprocal of
        # the median. Preserve the runner's independently scored direction.
        self.assertAlmostEqual(result["ratio_onetu_over_linked"], (1 / 4 + 1 / 6) / 2)

    def test_calibration_converges_using_the_fastest_batch(self):
        events = []
        linked = FakeTimer("linked", [0.25, 0.25, 0.125, 0.125, *([0.125] * 16)], events)
        control = FakeTimer("one_tu", [0.125, 0.125, 0.1, 0.1, *([0.1] * 16)], events)
        result = time_pair(linked, control)

        self.assertEqual(events[2:8], [
            (name, "time", n) for n in (1, 125, 157) for name in ("linked", "one_tu")
        ])
        self.assertEqual(result["repetitions"], 157)
        self.assertTrue(all(event[2] == 157 for event in events[8:]))
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
        with self.assertRaises(RuntimeError):
            time_pair(
                FakeTimer("linked", [1.0, 1e-9], events),
                FakeTimer("one_tu", [1.0, 1e-9], events),
            )
        self.assertEqual(events, [
            ("linked", "time", 8), ("one_tu", "time", 8),
            ("linked", "time", 1), ("one_tu", "time", 1),
        ])

    def test_calibration_aborts_after_24_nonconverging_rounds(self):
        class NonconvergingTimer:
            def __init__(self):
                self.calls = 0

            def time(self, repetitions):
                self.calls += 1
                return 15.0

        linked, control = NonconvergingTimer(), NonconvergingTimer()
        with self.assertRaises(RuntimeError):
            time_pair(linked, control)
        self.assertEqual((linked.calls, control.calls), (25, 25))


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
