from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from evaluator.harness.parity.assign import (
    assign_cases,
    build_failover_record,
    discover_idle_gpus,
)
from evaluator.harness.parity.case_device import (
    FAIL as CASE_DEVICE_FAIL,
    INELIGIBLE as CASE_DEVICE_INELIGIBLE,
    PASS as CASE_DEVICE_PASS,
    F2_SCHEMA_V1,
    F2_SCHEMA_V2,
    F2_SCHEMA_V3,
    CaseDeviceFallback,
    evaluate_case_device_qualification,
    evaluate_f2_solo_anchor,
    evaluate_f2_solo_anchor_v3,
    evaluate_strict_own_clock,
    require_case_qualification_pass,
    read_case_device_qualification,
    read_f2_solo_anchor,
    read_f2_solo_anchor_v3,
    rotated_fallback_order,
)
from evaluator.harness.parity.constants import BARRIER_AUDIT_FIELDS, NONFINITE_POLICY
from evaluator.harness.parity.f2_transition import (
    evaluate_f2_fast_transition,
    read_f2_fast_transition,
)
from evaluator.harness.parity.gap_eligibility import (
    ELIGIBILITY_INPUT_SCHEMA,
    THROUGHPUT_SCHEMA,
    GapSoloError,
    evaluate_eligibility,
    freeze_policy,
    make_solo,
    score as score_gap,
)
from evaluator.harness.parity.fast_protocol import (
    decide as decide_fast,
    escalation_triggered,
    read_decision,
    seal_leg,
    seal_trigger,
)
from evaluator.harness.parity.gpu_qualifier import (
    evaluate_qualification,
    read_qualification,
)
from evaluator.harness.parity.preflight import build_preflight_record
from evaluator.harness.parity.reader import (
    chronological_primary_pairs,
    pair_is_primary_valid,
    summary_primary_valid_count,
)
from evaluator.harness.parity.records import build_correctness_record, build_resource_record
from evaluator.harness.parity.runner import (
    ImageSpec,
    PairCapError,
    Protocol,
    run_case,
)
from evaluator.harness.parity.streams import (
    build_stream_map,
    class_bytes_identical,
    philox4x32_10,
    storage_bytes,
    stream_address,
    values_match,
    word_at,
)
from evaluator.harness.parity.solo_cadence import (
    SoloCadenceError,
    SoloRefreshCadence,
)
from evaluator.harness.parity.telemetry import (
    ClockSample,
    GapPowerTelemetryError,
    evaluate_clock_predicates,
    evaluate_gap_row_predicates,
)


class StreamTests(unittest.TestCase):
    def test_random123_known_vector(self) -> None:
        self.assertEqual(
            philox4x32_10((0, 0, 0, 0), (0, 0)),
            (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8),
        )

    def test_exact_address_and_determinism(self) -> None:
        address = stream_address("F1", 12, 5, (1 << 34) + 3)
        self.assertEqual(address["key"], (0x50495045, 0x0001))
        self.assertEqual(address["counter"], (0, 1, 12, 5))
        self.assertEqual(address["lane"], 3)
        self.assertEqual(word_at("F1", 2, 1, 77), word_at("F1", 2, 1, 77))

    def test_namespace_disjoint_address_and_bytes(self) -> None:
        f1 = stream_address("F1", 0, 1, 0)
        f2 = stream_address("F2", 0, 1, 0)
        self.assertNotEqual(f1["key"], f2["key"])
        self.assertNotEqual(storage_bytes("F1", 0, 1, 32), storage_bytes("F2", 0, 1, 32))

    def test_equivalence_identity_and_f1_bias_split(self) -> None:
        self.assertTrue(class_bytes_identical("F1", 3, 4, ["v3-linked", "v3-control"], 64))
        self.assertNotEqual(storage_bytes("F1", 3, 4, 64), storage_bytes("F1", 3, 5, 64))

    def test_int4_range_and_packing(self) -> None:
        packed = storage_bytes("F3", 0, 2, 65)
        self.assertEqual(len(packed), 33)
        values = []
        for byte in packed:
            values.extend((byte & 0xF, byte >> 4))
        signed = [value if value < 8 else value - 16 for value in values[:65]]
        self.assertTrue(all(-8 <= value <= 7 for value in signed))

    def test_nonfinite_policy(self) -> None:
        self.assertTrue(values_match(math.nan, math.nan, rtol=0.0, atol=0.0))
        self.assertFalse(values_match(math.nan, 0.0, rtol=1.0, atol=1.0))
        self.assertTrue(values_match(math.inf, math.inf, rtol=0.0, atol=0.0))
        self.assertFalse(values_match(math.inf, -math.inf, rtol=1.0, atol=1.0))
        self.assertTrue(values_match(1.01, 1.0, rtol=0.01, atol=0.001))

    def test_stream_map_is_canonical_and_enforces_bias_extent(self) -> None:
        config = {
            "namespace": "F1",
            "cases": [{
                "case_id": "F1-00",
                "numeric_case": 0,
                "dimensions": {"M": 7, "N": 11, "K": 13},
                "tensors": [
                    {"role_id": 1, "shape": [7, 13], "strides": [13, 1], "consumers": ["v3-linked", "v3-control"]},
                    {"role_id": 4, "shape": [11], "strides": [1], "consumers": ["v3-linked", "v3-control"]},
                ],
            }],
        }
        first = build_stream_map(config)
        second = build_stream_map(config)
        self.assertEqual(first, second)
        self.assertRegex(first["sha256"], "^[0-9a-f]{64}$")
        config["cases"][0]["tensors"][1]["shape"] = [7]
        with self.assertRaises(ValueError):
            build_stream_map(config)


class TelemetryTests(unittest.TestCase):
    @staticmethod
    def rows(mask: int, reference_clock: int = 1000, linked_clock: int = 1000) -> tuple[list[dict[str, tuple[int, int]]], list[ClockSample]]:
        windows = [{"reference": (100, 300), "linked": (400, 600)}]
        samples = [
            ClockSample(150, True, reference_clock, 40, mask, ""),
            ClockSample(250, True, reference_clock, 40, mask, ""),
            ClockSample(450, True, linked_clock, 40, mask, ""),
            ClockSample(550, True, linked_clock, 40, mask, ""),
        ]
        return windows, samples

    @staticmethod
    def rows_with_masks(masks: list[int]) -> tuple[list[dict[str, tuple[int, int]]], list[ClockSample]]:
        if len(masks) != 4:
            raise ValueError("truth-table fixture needs four masks")
        windows, samples = TelemetryTests.rows(0x0)
        return windows, [
            ClockSample(sample.time_ns, True, 1000, 40, mask, "")
            for sample, mask in zip(samples, masks, strict=True)
        ]

    def test_pair_validity_truth_table(self) -> None:
        cases = {
            "all-0x0": ([0x0, 0x0, 0x0, 0x0], True, False, True),
            "0x0-0x1-mix": ([0x0, 0x1, 0x0, 0x1], True, False, True),
            "all-0x4": ([0x4, 0x4, 0x4, 0x4], False, True, True),
            "0x4-0x0-mix": ([0x4, 0x0, 0x4, 0x0], False, False, False),
            "0x4-0x1-mix": ([0x4, 0x1, 0x4, 0x1], False, False, False),
            "other-composite": ([0x5, 0x5, 0x5, 0x5], False, False, False),
        }
        for name, (masks, original, amended, pair) in cases.items():
            with self.subTest(name=name):
                windows, samples = self.rows_with_masks(masks)
                _, results = evaluate_clock_predicates(windows, samples)
                result = results[0]
                self.assertEqual(result["original_valid"], original)
                self.assertEqual(result["amended_valid"], amended)
                self.assertEqual(result["pair_valid"], pair)
                self.assertFalse(result["original_valid"] and result["amended_valid"])
                self.assertEqual(result["pair_valid"], result["original_valid"] or result["amended_valid"])
                self.assertEqual(result["original_discard_reasons"] == [], original)
                self.assertEqual(result["amended_discard_reasons"] == [], amended)

    def test_common_clock_failures_feed_both_branches(self) -> None:
        fixtures: dict[str, tuple[list[dict[str, tuple[int, int]]], list[ClockSample], str]] = {}
        windows, samples = self.rows(0x4)
        samples[0] = ClockSample(150, False, None, None, None, "query failed")
        fixtures["query-failure"] = (windows, samples, "query-failure")
        windows, samples = self.rows(0x4)
        samples.pop()
        fixtures["uncovered-slot"] = (windows, samples, "uncovered")
        windows, samples = self.rows(0x4)
        samples[0] = ClockSample(150, True, 1100, 40, 0x4, "")
        fixtures["clock-window"] = (windows, samples, "clock-outside-3-percent")
        windows, samples = self.rows(0x4, 1000, 1040)
        fixtures["asymmetric-medians"] = (windows, samples, "asymmetric-slot-medians-over-3-percent")
        for name, (case_windows, case_samples, reason_fragment) in fixtures.items():
            with self.subTest(name=name):
                _, results = evaluate_clock_predicates(case_windows, case_samples)
                result = results[0]
                self.assertFalse(result["original_valid"])
                self.assertFalse(result["amended_valid"])
                self.assertFalse(result["pair_valid"])
                self.assertTrue(any(reason_fragment in reason for reason in result["original_discard_reasons"]))
                self.assertTrue(any(reason_fragment in reason for reason in result["amended_discard_reasons"]))

    def test_amended_and_original_are_both_computed(self) -> None:
        windows, samples = self.rows(0x4)
        median, result = evaluate_clock_predicates(windows, samples)
        self.assertEqual(median, 1000.0)
        self.assertTrue(result[0]["amended_valid"])
        self.assertFalse(result[0]["original_valid"])
        windows, samples = self.rows(0x1)
        _, result = evaluate_clock_predicates(windows, samples)
        self.assertFalse(result[0]["amended_valid"])
        self.assertTrue(result[0]["original_valid"])

    def test_asymmetric_clock_is_discarded(self) -> None:
        windows, samples = self.rows(0x4, 1000, 1040)
        _, result = evaluate_clock_predicates(windows, samples)
        self.assertFalse(result[0]["amended_valid"])
        self.assertIn("asymmetric-slot-medians-over-3-percent", result[0]["amended_discard_reasons"])

    def test_query_failure_and_uncovered_fail_closed(self) -> None:
        windows, samples = self.rows(0x4)
        samples[0] = ClockSample(150, False, None, None, None, "query failed")
        samples.pop()
        _, result = evaluate_clock_predicates(windows, samples)
        self.assertFalse(result[0]["amended_valid"])
        self.assertTrue(any("query-failure" in reason for reason in result[0]["amended_discard_reasons"]))
        self.assertTrue(any("uncovered" in reason for reason in result[0]["amended_discard_reasons"]))


class GapRowTests(unittest.TestCase):
    @staticmethod
    def rows(
        reference_clocks: list[int],
        linked_clocks: list[int],
        reference_masks: list[int],
        linked_masks: list[int],
        *,
        query_ok: bool = True,
        include_power: bool = True,
    ) -> tuple[list[dict[str, tuple[int, int]]], list[ClockSample]]:
        windows = [{"reference": (100, 300), "linked": (400, 600)}]
        values = [
            (150, reference_clocks[0], reference_masks[0], 800.0),
            (250, reference_clocks[1], reference_masks[1], 820.0),
            (450, linked_clocks[0], linked_masks[0], 900.0),
            (550, linked_clocks[1], linked_masks[1], 940.0),
        ]
        samples = [
            ClockSample(
                timestamp,
                query_ok,
                clock if query_ok else None,
                40 if query_ok else None,
                mask if query_ok else None,
                "",
                power if include_power else None,
                1100.0 if include_power else None,
            )
            for timestamp, clock, mask, power in values
        ]
        return windows, samples

    def test_stable_unequal_clocks_same_amended_branch_accepts(self) -> None:
        windows, samples = self.rows([1000, 1000], [1500, 1500], [0x4, 0x4], [0x4, 0x4])
        _, strict = evaluate_clock_predicates(windows, samples)
        gap = evaluate_gap_row_predicates(windows, samples)[0]
        self.assertFalse(strict[0]["pair_valid"])
        self.assertTrue(gap["gap_amended_pair_valid"])
        self.assertFalse(gap["gap_original_pair_valid"])
        self.assertTrue(gap["gap_pair_valid"])
        self.assertEqual(gap["clock_median_delta_linked_minus_reference_mhz"], 500.0)
        self.assertEqual(gap["clock_median_delta_percent_linked_over_reference"], 50.0)
        self.assertEqual(gap["configured_power_limit_w"], 1100.0)
        self.assertEqual(gap["halves"]["reference"]["power_draw_w"]["median"], 810.0)

    def test_stable_unequal_clocks_both_original_accepts(self) -> None:
        windows, samples = self.rows([1000, 1000], [1500, 1500], [0x0, 0x1], [0x1, 0x0])
        gap = evaluate_gap_row_predicates(windows, samples)[0]
        self.assertTrue(gap["gap_original_pair_valid"])
        self.assertFalse(gap["gap_amended_pair_valid"])
        self.assertTrue(gap["gap_pair_valid"])

    def test_cross_branch_and_within_half_transition_reject(self) -> None:
        windows, samples = self.rows([1000, 1000], [1500, 1500], [0x0, 0x1], [0x4, 0x4])
        gap = evaluate_gap_row_predicates(windows, samples)[0]
        self.assertFalse(gap["gap_pair_valid"])
        self.assertTrue(gap["gap_same_branch_failure"])
        windows, samples = self.rows([1000, 1100], [1500, 1500], [0x4, 0x4], [0x4, 0x4])
        gap = evaluate_gap_row_predicates(windows, samples)[0]
        self.assertFalse(gap["gap_pair_valid"])
        self.assertTrue(any("own-median-clock-outside-3-percent" in reason for reason in gap["halves"]["reference"]["gap_amended_half_discard_reasons"]))

    def test_gap_fail_closed_truth_table(self) -> None:
        fixtures = {
            "mixed-mask": self.rows([1000, 1000], [1500, 1500], [0x0, 0x4], [0x4, 0x4]),
            "other-bit": self.rows([1000, 1000], [1500, 1500], [0x5, 0x5], [0x4, 0x4]),
            "query-failure": self.rows([1000, 1000], [1500, 1500], [0x4, 0x4], [0x4, 0x4], query_ok=False),
        }
        windows, samples = self.rows([1000, 1000], [1500, 1500], [0x4, 0x4], [0x4, 0x4])
        samples.pop()
        fixtures["uncovered"] = (windows, samples)
        for name, (case_windows, case_samples) in fixtures.items():
            with self.subTest(name=name):
                gap = evaluate_gap_row_predicates(case_windows, case_samples)[0]
                self.assertFalse(gap["gap_pair_valid"])

    def test_missing_power_fails_route_not_pair(self) -> None:
        windows, samples = self.rows(
            [1000, 1000], [1500, 1500], [0x4, 0x4], [0x4, 0x4],
            include_power=False,
        )
        with self.assertRaises(GapPowerTelemetryError):
            evaluate_gap_row_predicates(windows, samples)

    def test_closed_allowlist_and_dual_reader(self) -> None:
        config = {
            "stream_map_sha256": "5" * 64,
            "stream_classes": ["ns=0x0102/case=0/role=1"],
            "inputs": {"shape": [4096, 4096, 4096]},
            "arguments": {},
        }
        protocol = Protocol.smoke(
            Path("/tmp/gap-test"), config,
            gap_row_allowlist=["A-solver-D4"],
        )
        self.assertEqual(protocol.classifier_for_case("A-solver-D4"), "gap-ex-ante-solo-v2")
        with self.assertRaises(ValueError):
            protocol.classifier_for_case("A-solver-D2")
        with self.assertRaises(ValueError):
            Protocol.smoke(
                Path("/tmp/gap-test"), config,
                gap_row_allowlist=["A-authoring-0"],
            )
        strict_pair = {"schema_version": "m3-parity-pair-v2", "attempt": 0, "pair_valid": True}
        gap_invalid = {
            "schema_version": "m3-parity-pair-v6", "attempt": 1,
            "gap_classification": {"gap_pair_valid": False},
        }
        gap_valid = {
            "schema_version": "m3-parity-pair-v6", "attempt": 2,
            "gap_classification": {"gap_pair_valid": True},
        }
        self.assertTrue(pair_is_primary_valid(strict_pair))
        self.assertFalse(pair_is_primary_valid(gap_invalid))
        self.assertTrue(pair_is_primary_valid(gap_valid))
        self.assertEqual(summary_primary_valid_count({"schema_version": "m3-parity-summary-v2", "pair_valid_count": 7}), 7)
        self.assertEqual(summary_primary_valid_count({"schema_version": "m3-parity-summary-v6", "gap_pair_valid_count": 5}), 5)

    def test_gap_chronological_selection_and_cap(self) -> None:
        def gap_pair(attempt: int, valid: bool) -> dict:
            return {
                "schema_version": "m3-parity-pair-v6",
                "attempt": attempt,
                "gap_classification": {"gap_pair_valid": valid},
            }

        records = [
            gap_pair(3, True),
            gap_pair(1, True),
            gap_pair(0, False),
            gap_pair(2, True),
        ]
        selected = chronological_primary_pairs(records, 2)
        self.assertEqual([pair["attempt"] for pair in selected], [1, 2])
        capped = chronological_primary_pairs(
            [gap_pair(0, False), gap_pair(1, True), gap_pair(2, False)],
            2,
        )
        self.assertEqual([pair["attempt"] for pair in capped], [1])


class ExAnteGapTests(unittest.TestCase):
    @staticmethod
    def policy_tuple() -> dict:
        return {
            "gpu_uuid": "GPU-test",
            "pci_bus_id": "0000:00:00.0",
            "configured_power_limit_w": 1100.0,
            "enforced_power_limit_w": 1100.0,
            "application_graphics_clock_mhz": 2032,
            "application_memory_clock_mhz": 3996,
            "default_graphics_clock_mhz": 2032,
            "default_memory_clock_mhz": 3996,
            "persistence_mode": "Enabled",
            "compute_mode": "Default",
        }

    @staticmethod
    def samples(clock: int, masks: list[int]) -> list[dict]:
        start = 5_000_000_001
        return [
            {
                "time_ns": start + index * 100_000_000,
                "query_ok": True,
                "clock_mhz": clock,
                "temperature_c": 40,
                "reason_mask": mask,
                "raw": "fixture",
                "power_draw_w": 800.0,
                "power_limit_w": 1100.0,
            }
            for index, mask in enumerate(masks)
        ]

    @classmethod
    def solo(
        cls, role: str, clock: int, masks: list[int], policy: dict
    ) -> dict:
        return make_solo(
            case_id="A-solver-D2",
            role=role,
            image_id=role + "-image",
            image_sha256=(("a" if role == "reference" else "b") * 64),
            policy_record=policy,
            series_id="A-solver-D2:series",
            duration_leg="gap-original-v1",
            session_id="session-0",
            solo_block_id="block-0",
            slot_start_ns=1,
            eligibility_window_start_ns=5_000_000_001,
            eligibility_window_end_ns=30_000_000_001,
            cuda_scoring_window_start_ns=5_000_000_001,
            cuda_scoring_window_end_ns=30_000_000_001,
            slot_end_ns=30_000_000_001,
            samples=cls.samples(clock, masks),
        )

    @classmethod
    def half(
        cls, role: str, clock: int, masks: list[int], solo: dict, policy: dict
    ) -> dict:
        return {
            "role": role,
            "series_id": solo["series_id"],
            "duration_leg": solo["duration_leg"],
            "session_id": solo["session_id"],
            "solo_block_id": solo["solo_block_id"],
            "image_id": solo["image_id"],
            "image_sha256": solo["image_sha256"],
            "gpu_uuid": "GPU-test",
            "pci_bus_id": "0000:00:00.0",
            "policy_sha256": policy["policy_sha256"],
            "policy_record_sha256": policy["policy_record_sha256"],
            "solo_sha256": solo["solo_sha256"],
            "slot_start_ns": 1,
            "eligibility_window_start_ns": 5_000_000_001,
            "eligibility_window_end_ns": 30_000_000_001,
            "cuda_scoring_window_start_ns": 5_000_000_001,
            "cuda_scoring_window_end_ns": 30_000_000_001,
            "slot_end_ns": 30_000_000_001,
            "samples": cls.samples(clock, masks),
        }

    @classmethod
    def source(
        cls,
        *,
        reference_clock: int = 1000,
        linked_clock: int = 1500,
        reference_solo_masks: list[int] | None = None,
        linked_solo_masks: list[int] | None = None,
        reference_pair_masks: list[int] | None = None,
        linked_pair_masks: list[int] | None = None,
    ) -> dict:
        reference_solo_masks = reference_solo_masks or [0x0, 0x1]
        linked_solo_masks = linked_solo_masks or [0x4, 0x4]
        reference_pair_masks = reference_pair_masks or list(reference_solo_masks)
        linked_pair_masks = linked_pair_masks or list(linked_solo_masks)
        policy = freeze_policy(cls.policy_tuple())
        reference_solo = cls.solo(
            "reference", reference_clock, reference_solo_masks, policy
        )
        linked_solo = cls.solo(
            "linked", linked_clock, linked_solo_masks, policy
        )
        return {
            "schema_version": ELIGIBILITY_INPUT_SCHEMA,
            "case_id": "A-solver-D2",
            "gpu_uuid": "GPU-test",
            "pci_bus_id": "0000:00:00.0",
            "policy_before": policy,
            "policy_frozen": policy,
            "policy_after": policy,
            "target_pairs": 1,
            "attempt_cap": 3,
            "leading_discard_ns": 5_000_000_000,
            "identity_gating": True,
            "solo_blocks": {
                "block-0": {
                    "reference": reference_solo,
                    "linked": linked_solo,
                },
            },
            "attempts": [{
                "attempt": 0,
                "order": ["reference", "linked"],
                "solo_block_id": "block-0",
                "halves": {
                    "reference": cls.half(
                        "reference", reference_clock, reference_pair_masks,
                        reference_solo, policy,
                    ),
                    "linked": cls.half(
                        "linked", linked_clock, linked_pair_masks,
                        linked_solo, policy,
                    ),
                },
                "strict_classification": {
                    "original_valid": False,
                    "amended_valid": False,
                    "pair_valid": False,
                },
                "superseded_same_branch_classification": {
                    "gap_pair_valid": False,
                },
            }],
        }

    def test_fast_gap_identity_mismatch_is_diagnostic_only(self) -> None:
        source = self.source()
        source["identity_gating"] = False
        source["attempts"][0]["halves"]["reference"]["samples"] = self.samples(
            1100, [0x0, 0x1]
        )
        ledger = evaluate_eligibility(source)
        half = ledger["attempts"][0]["halves"]["reference"]
        self.assertFalse(half["identity_valid"])
        self.assertFalse(half["identity_gating"])
        self.assertNotIn("solo-clock-identity-mismatch", half["reasons"])
        self.assertTrue(ledger["attempts"][0]["gap_pair_valid"])


    def test_cross_branch_preassigned_pair_is_valid(self) -> None:
        ledger = evaluate_eligibility(self.source())
        self.assertTrue(ledger["attempts"][0]["gap_pair_valid"])
        self.assertTrue(ledger["attempts"][0]["cross_branch"])
        self.assertTrue(ledger["ratio_blind"])

    def test_within_half_branch_deviation_discards(self) -> None:
        source = self.source(linked_pair_masks=[0x0, 0x0])
        source["identity_gating"] = False
        ledger = evaluate_eligibility(source)
        self.assertFalse(ledger["attempts"][0]["gap_pair_valid"])
        self.assertIn(
            "linked:within-half-branch-deviation",
            ledger["attempts"][0]["reasons"],
        )

    def test_solo_identity_and_policy_mismatch_discard(self) -> None:
        source = self.source()
        source["attempts"][0]["halves"]["linked"]["samples"] = self.samples(
            1600, [0x4, 0x4]
        )
        ledger = evaluate_eligibility(source)
        self.assertIn(
            "linked:solo-clock-identity-mismatch",
            ledger["attempts"][0]["reasons"],
        )
        source = self.source()
        source["attempts"][0]["halves"]["linked"]["policy_sha256"] = "c" * 64
        ledger = evaluate_eligibility(source)
        self.assertIn(
            "linked:policy-hash-mismatch",
            ledger["attempts"][0]["reasons"],
        )

    def test_same_session_solo_scope_rejections(self) -> None:
        for field in (
            "series_id",
            "duration_leg",
            "session_id",
            "solo_block_id",
        ):
            with self.subTest(field=field):
                source = self.source()
                source["attempts"][0]["halves"]["linked"][field] = "reused"
                ledger = evaluate_eligibility(source)
                self.assertIn(
                    f"linked:{field}-scope-mismatch",
                    ledger["attempts"][0]["reasons"],
                )

    def test_mixed_solo_fails_route(self) -> None:
        policy = freeze_policy(self.policy_tuple())
        with self.assertRaises(GapSoloError):
            self.solo("reference", 1000, [0x0, 0x4], policy)

    def test_timestamp_coverage_proof_is_gating(self) -> None:
        source = self.source()
        source["attempts"][0]["halves"]["linked"][
            "cuda_scoring_window_start_ns"
        ] += 1
        ledger = evaluate_eligibility(source)
        half = ledger["attempts"][0]["halves"]["linked"]
        self.assertFalse(half["timestamp_coverage_identical"])
        self.assertIn(
            "linked:eligibility-cuda-scoring-window-mismatch",
            ledger["attempts"][0]["reasons"],
        )

    def test_ratio_blind_seal_then_score(self) -> None:
        source = self.source()
        contaminated = json.loads(json.dumps(source))
        contaminated["attempts"][0]["throughput"] = 999.0
        with self.assertRaises(ValueError):
            evaluate_eligibility(contaminated)
        ledger = evaluate_eligibility(source)
        scored = score_gap(
            ledger,
            {
                "schema_version": THROUGHPUT_SCHEMA,
                "case_id": "A-solver-D2",
                "eligibility_ledger_sha256": ledger[
                    "eligibility_ledger_sha256"
                ],
                "attempts": [{
                    "attempt": 0,
                    "reference_throughput": 100.0,
                    "linked_throughput": 80.0,
                }],
            },
        )
        self.assertAlmostEqual(
            scored["median_ratio_linked_over_reference"], 0.8
        )


class SoloCadenceTests(unittest.TestCase):
    @staticmethod
    def cadence() -> SoloRefreshCadence:
        return SoloRefreshCadence.begin(
            {"reference": "original", "linked": "power-cap"},
            "block-0",
            "a" * 64,
            0,
        )

    def test_refresh_fires_by_nominal_attempt_five(self) -> None:
        cadence = self.cadence()
        self.assertEqual(
            [cadence.refresh_due(attempt * 6_000_000_000) for attempt in range(1, 6)],
            [False, False, False, False, True],
        )

    def test_no_refresh_below_thirty_seconds(self) -> None:
        self.assertFalse(self.cadence().refresh_due(29_999_999_999))

    def test_refresh_branch_flip_fails_series(self) -> None:
        with self.assertRaises(SoloCadenceError):
            self.cadence().seal_refresh(
                {"reference": "power-cap", "linked": "power-cap"},
                "block-1",
                "b" * 64,
                31_000_000_000,
            )

    def test_cadence_is_not_reactive_to_attempt_outcomes(self) -> None:
        timestamps = [6, 12, 18, 24, 30]
        validities = [True, False, True, False, True]
        ratios = [0.99, 0.2, 1.1, 0.5, 0.97]
        first = self.cadence()
        second = self.cadence()
        decisions_with_outcomes = [
            first.refresh_due(timestamp * 1_000_000_000)
            for timestamp, _valid, _ratio in zip(
                timestamps, validities, ratios, strict=True
            )
        ]
        decisions_without_outcomes = [
            second.refresh_due(timestamp * 1_000_000_000)
            for timestamp in timestamps
        ]
        self.assertEqual(decisions_with_outcomes, decisions_without_outcomes)
class F2TransitionV2Tests(unittest.TestCase):
    @staticmethod
    def sample(time_ns: int, clock: int, mask: int, query: bool = True) -> dict:
        return {
            "time_ns": time_ns,
            "query_ok": query,
            "clock_mhz": clock,
            "reason_mask": mask,
        }

    @classmethod
    def source(
        cls,
        transition_clocks: list[int],
        transition_masks: list[int],
        *,
        scored_clocks: list[int] | None = None,
        scored_masks: list[int] | None = None,
        branch: str = "power-cap",
    ) -> dict:
        scored_clocks = scored_clocks or [1000, 1000]
        scored_masks = scored_masks or ([4, 4] if branch == "power-cap" else [0, 1])
        return {
            "schema_version": "f2-cross-source-fast-transition-v2",
            "preassigned_branch": branch,
            "transition_start_ns": 0,
            "transition_end_ns": 1_000_000_000,
            "scored_start_ns": 1_000_000_000,
            "scored_end_ns": 3_000_000_000,
            "transition_samples": [
                cls.sample(100_000_000 * (index + 1), clock, mask)
                for index, (clock, mask) in enumerate(
                    zip(transition_clocks, transition_masks, strict=True)
                )
            ],
            "scored_samples": [
                cls.sample(1_100_000_000 + index * 100_000_000, clock, mask)
                for index, (clock, mask) in enumerate(
                    zip(scored_clocks, scored_masks, strict=True)
                )
            ],
        }

    def test_clock_contact_truth_table(self) -> None:
        cases = [
            ([1000, 1000], [4, 4], True),
            ([1100, 1000, 1100], [0, 4, 4], True),
            ([1000, 1100], [4, 4], True),
            ([1100, 1100], [4, 4], False),
        ]
        for clocks, masks, passed in cases:
            with self.subTest(clocks=clocks):
                result = evaluate_f2_fast_transition(
                    self.source(clocks, masks)
                )
                self.assertEqual(result["valid"], passed)
                self.assertNotIn("clock_irreversible_entry_valid", result)
        oscillating = evaluate_f2_fast_transition(
            self.source([1100, 1000, 1100], [0, 4, 4])
        )
        self.assertEqual(oscillating["clock_exit_indices"], [2])
        self.assertEqual(oscillating["transition_in_band_count"], 1)

    def test_query_coverage_scored_clock_and_timing_failures(self) -> None:
        query = self.source([1100, 1000], [0, 4])
        query["transition_samples"][0]["query_ok"] = False
        self.assertFalse(evaluate_f2_fast_transition(query)["valid"])
        coverage = self.source([1000, 1000], [4, 4])
        coverage["transition_samples"].pop()
        self.assertFalse(evaluate_f2_fast_transition(coverage)["valid"])
        scored = self.source(
            [1000, 1000], [4, 4], scored_clocks=[1000, 1100]
        )
        self.assertFalse(evaluate_f2_fast_transition(scored)["valid"])
        overlap = self.source([1000, 1000], [4, 4])
        overlap["transition_samples"][-1]["time_ns"] = 1_000_000_000
        self.assertFalse(evaluate_f2_fast_transition(overlap)["valid"])

    def test_mask_one_way_and_independent_entry_indices(self) -> None:
        passed = evaluate_f2_fast_transition(
            self.source([1100, 1000, 1100], [0, 4, 4])
        )
        self.assertTrue(passed["valid"])
        self.assertEqual(passed["first_clock_contact_index"], 1)
        self.assertEqual(passed["mask_branch_entry_index"], 1)
        regression = evaluate_f2_fast_transition(
            self.source([1000, 1000, 1000], [4, 0, 4])
        )
        self.assertFalse(regression["mask_irreversible_entry_valid"])
        different = evaluate_f2_fast_transition(
            self.source([1000, 1100, 1000], [0, 0, 4])
        )
        self.assertEqual(different["first_clock_contact_index"], 0)
        self.assertEqual(different["mask_branch_entry_index"], 2)

    def test_v1_unknown_dispatch_fails(self) -> None:
        result = evaluate_f2_fast_transition(
            self.source([1000, 1000], [4, 4])
        )
        bad = dict(result)
        bad["classifier_version"] = "f2-cross-source-fast-transition-v1"
        with self.assertRaises(ValueError):
            read_f2_fast_transition(bad)



class QualifierTests(unittest.TestCase):
    @staticmethod
    def rows(clock: int, masks: list[int], start: int) -> list[dict]:
        return [
            {
                "time_ns": start + index * 100_000_000,
                "query_ok": True,
                "clock_mhz": clock,
                "reason_mask": mask,
                "power_draw_w": 900.0,
                "power_limit_w": 1100.0,
            }
            for index, mask in enumerate(masks)
        ]

    @classmethod
    def source(
        cls,
        *,
        w1_clock: int = 1000,
        w2_clock: int = 1000,
        w1_masks: list[int] | None = None,
        w2_masks: list[int] | None = None,
    ) -> dict:
        w1_masks = w1_masks or [4, 4]
        w2_masks = w2_masks or [4, 4]
        return {
            "schema_version": "fa4-f2-00-telemetry-qualification-v2",
            "epoch_id": "epoch-0",
            "physical_index": 2,
            "gpu_uuid": "GPU-2",
            "pci_bus_id": "2",
            "policy_sha256": "a" * 64,
            "baseline_dispatch_sha256": "b" * 64,
            "jit_identity_sha256": "c" * 64,
            "input_sha256": "d" * 64,
            "stream_map_sha256": "e" * 64,
            "settled_start_ns": 0,
            "w1_start_ns": 0,
            "w1_end_ns": 10_000_000_000,
            "soak_start_ns": 10_000_000_000,
            "soak_end_ns": 70_000_000_000,
            "w2_start_ns": 70_000_000_000,
            "w2_end_ns": 80_000_000_000,
            "soak_continuous": True,
            "w1_samples": cls.rows(w1_clock, w1_masks, 0),
            "w2_samples": cls.rows(w2_clock, w2_masks, 70_000_000_000),
        }

    def test_drift_boundary_and_branch_truth(self) -> None:
        self.assertEqual(
            evaluate_qualification(self.source(w2_clock=1010))["status"],
            "PASS",
        )
        over = evaluate_qualification(self.source(w2_clock=1011))
        self.assertEqual(over["status"], "INELIGIBLE_TELEMETRY_STATE")
        self.assertIn("load-soak-clock-drift-over-1-percent", over["reasons"])
        flip = evaluate_qualification(
            self.source(w1_masks=[0, 1], w2_masks=[4, 4])
        )
        self.assertIn("load-soak-branch-drift", flip["reasons"])

    def test_w2_predicate_failure_and_unknown_dispatch(self) -> None:
        source = self.source()
        source["w2_samples"][0]["clock_mhz"] = 1200
        failed = evaluate_qualification(source)
        self.assertIn("w2-clock-window-failure", failed["reasons"])
        bad = dict(failed)
        bad["schema_version"] = "m3-gpu-telemetry-qualification-result-v1"
        with self.assertRaises(ValueError):
            read_qualification(bad)

    def test_qualifier_reader_rejects_scoring_fields(self) -> None:
        source = self.source()
        source["throughput"] = 123.0
        with self.assertRaises(ValueError):
            evaluate_qualification(source)


class CaseDeviceQualificationTests(unittest.TestCase):
    @staticmethod
    def role(role_id: str, masks: list[int], *, query_ok: bool = True, clocks: list[int] | None = None) -> dict:
        clocks = clocks or [1000] * len(masks)
        return {
            "role_id": role_id,
            "image_sha256": "a" * 64,
            "input_sha256": "b" * 64,
            "stream_map_sha256": "c" * 64,
            "policy_sha256": "d" * 64,
            "samples": [
                {
                    "query_ok": query_ok,
                    "time_ns": 1_000_000_000 + index * 100_000_000,
                    "clock_mhz": clock,
                    "reason_mask": mask,
                    "policy_sha256": "d" * 64,
                }
                for index, (clock, mask) in enumerate(
                    zip(clocks, masks, strict=True)
                )
            ],
        }

    @classmethod
    def source(cls, role_masks: dict[str, list[int]], *, case_id: str = "F2-00", schema: str = "f2-case-device-branch-qualification-v1", physical_index: int = 3) -> dict:
        order = sorted(role_masks, key=lambda item: item.encode())
        return {
            "schema_version": schema,
            "phase": "fast-prepair",
            "case_id": case_id,
            "assignment_epoch_id": "epoch",
            "physical_index": physical_index,
            "gpu_uuid": f"GPU-{physical_index}",
            "pci_bus_id": str(physical_index),
            "policy_sha256": "d" * 64,
            "role_order": order,
            "roles": [cls.role(role, role_masks[role]) for role in order],
            "pair_attempt_directories": [],
        }

    def test_closed_role_and_disposition_truth_table(self) -> None:
        passed = evaluate_case_device_qualification(
            self.source({"control": [0, 1], "fa4": [4, 4], "linked": [0, 0]})
        )
        self.assertEqual(passed["disposition"], CASE_DEVICE_PASS)
        mixed = evaluate_case_device_qualification(
            self.source({"control": [0, 4], "fa4": [4, 4], "linked": [0, 0]})
        )
        self.assertEqual(mixed["disposition"], CASE_DEVICE_INELIGIBLE)
        self.assertEqual(mixed["pair_attempt_directories"], [])
        for masks in ([8, 8],):
            fatal = evaluate_case_device_qualification(
                self.source({"control": list(masks), "fa4": [4, 4], "linked": [0, 0]})
            )
            self.assertEqual(fatal["disposition"], CASE_DEVICE_FAIL)
        query = self.source({"control": [0, 0], "fa4": [4, 4], "linked": [0, 0]})
        query["roles"][0]["samples"][0]["query_ok"] = False
        self.assertEqual(
            evaluate_case_device_qualification(query)["disposition"],
            CASE_DEVICE_FAIL,
        )
        clock = self.source({"control": [0, 0], "fa4": [4, 4], "linked": [0, 0]})
        clock["roles"][0]["samples"][0]["clock_mhz"] = 1200
        self.assertEqual(
            evaluate_case_device_qualification(clock)["disposition"],
            CASE_DEVICE_FAIL,
        )

    def test_f2_v2_percentile_anchor_truth_table(self) -> None:
        strict_pass = [1000] * 20
        self.assertTrue(evaluate_strict_own_clock(strict_pass)["valid"])
        self.assertTrue(evaluate_f2_solo_anchor(strict_pass)["valid"])

        one_tail = [1000] * 89 + [1040]
        self.assertFalse(evaluate_strict_own_clock(one_tail)["valid"])
        anchor = evaluate_f2_solo_anchor(one_tail)
        self.assertTrue(anchor["valid"])
        self.assertEqual(anchor["count_outside_3_percent"], 1)
        self.assertEqual(anchor["count_outside_5_percent"], 0)

        unknown_anchor = dict(anchor)
        unknown_anchor["classifier_version"] = "unknown-f2-anchor"
        with self.assertRaises(ValueError):
            read_f2_solo_anchor(unknown_anchor)
        self.assertFalse(evaluate_f2_solo_anchor([])["valid"])
        beyond_five = [1000] * 89 + [1051]
        failed_tail = evaluate_f2_solo_anchor(beyond_five)
        self.assertFalse(failed_tail["valid"])
        self.assertIn(
            "f2-solo-anchor-tail-envelope-failure",
            failed_tail["reasons"],
        )

        percentile_tail = [969] + [1000] * 19
        failed_percentile = evaluate_f2_solo_anchor(percentile_tail)
        self.assertFalse(failed_percentile["valid"])
        self.assertIn(
            "f2-solo-anchor-percentile-envelope-failure",
            failed_percentile["reasons"],
        )
        self.assertEqual(failed_percentile["count_outside_5_percent"], 0)

    def test_exact_f2_13_fixture_dual_result(self) -> None:
        masks = {
            "control": [0] * 90,
            "fa4": [4] * 90,
            "linked": [0] * 90,
        }
        source = self.source(
            masks,
            case_id="F2-13",
            schema=F2_SCHEMA_V2,
        )
        clocks = [1282] * 9 + [1290] * 77 + [1297] * 3 + [1335]
        fa4 = next(role for role in source["roles"] if role["role_id"] == "fa4")
        for sample, clock in zip(fa4["samples"], clocks, strict=True):
            sample["clock_mhz"] = clock
        result = evaluate_case_device_qualification(source)
        self.assertEqual(result["disposition"], CASE_DEVICE_PASS)
        fa4_result = next(
            role for role in result["roles"] if role["role_id"] == "fa4"
        )
        self.assertFalse(fa4_result["strict_own_clock_valid_v1"])
        self.assertTrue(fa4_result["f2_solo_anchor_valid_v2"])
        anchor = read_f2_solo_anchor(fa4_result["f2_solo_anchor_v2"])
        self.assertEqual(
            (
                anchor["n"], anchor["minimum_mhz"], anchor["q05_mhz"],
                anchor["median_mhz"], anchor["q95_mhz"],
                anchor["maximum_mhz"],
            ),
            (90, 1282.0, 1282.0, 1290.0, 1290.0, 1335.0),
        )
        self.assertTrue(result["compatibility_only"])
        self.assertFalse(result["may_supply_pair_branch_or_identity"])

    def test_f2_v2_scope_masks_and_reader_fail_closed(self) -> None:
        mixed = evaluate_case_device_qualification(self.source(
            {"control": [0, 4], "fa4": [4, 4], "linked": [0, 0]},
            schema=F2_SCHEMA_V2,
        ))
        self.assertEqual(mixed["disposition"], CASE_DEVICE_INELIGIBLE)
        query = self.source(
            {"control": [0, 0], "fa4": [4, 4], "linked": [0, 0]},
            schema=F2_SCHEMA_V2,
        )
        query["roles"][0]["samples"][0]["query_ok"] = False
        self.assertEqual(
            evaluate_case_device_qualification(query)["disposition"],
            CASE_DEVICE_FAIL,
        )
        with self.assertRaises(ValueError):
            evaluate_case_device_qualification(self.source(
                {"a": [0, 0], "b": [4, 4], "c": [0, 0]},
                case_id="A-solver-D4",
                schema=F2_SCHEMA_V2,
            ))
        v1 = evaluate_case_device_qualification(self.source(
            {"control": [0, 0], "fa4": [4, 4], "linked": [0, 0]},
            schema=F2_SCHEMA_V1,
        ))
        self.assertEqual(
            v1["schema_version"],
            "m3-case-device-branch-qualification-result-v1",
        )
        unknown = dict(mixed)
        unknown["qualifier_version"] = "f2-case-device-branch-qualification-v3"
        with self.assertRaises(ValueError):
            read_case_device_qualification(unknown)
        aliased = dict(mixed)
        aliased["may_supply_pair_branch_or_identity"] = True
        with self.assertRaises(ValueError):
            read_case_device_qualification(aliased)

    @staticmethod
    def timestamped(clocks: list[int]) -> list[dict]:
        return [
            {
                "time_ns": 1_000_000_000 + index * 100_000_000,
                "clock_mhz": clock,
            }
            for index, clock in enumerate(clocks)
        ]

    def test_f2_v3_isolated_tail_truth_table(self) -> None:
        v2_valid = self.timestamped([1000] * 20)
        self.assertTrue(evaluate_f2_solo_anchor([1000] * 20)["valid"])
        self.assertTrue(evaluate_f2_solo_anchor_v3(v2_valid)["valid"])

        interior = self.timestamped([1000] * 90)
        interior[22]["clock_mhz"] = 1070
        accepted = evaluate_f2_solo_anchor_v3(interior)
        self.assertTrue(accepted["valid"])
        self.assertEqual(accepted["excluded_sample_count"], 1)
        self.assertEqual(
            accepted["excluded_sample"]["chronological_index_one_based"], 23
        )
        self.assertEqual(
            accepted["median_mhz"], accepted["retained_median_mhz"]
        )

        for indices in ((22, 23), (22, 60)):
            with self.subTest(indices=indices):
                samples = self.timestamped([1000] * 90)
                for index in indices:
                    samples[index]["clock_mhz"] = 1070
                rejected = evaluate_f2_solo_anchor_v3(samples)
                self.assertFalse(rejected["valid"])
                self.assertIn(
                    "f2-solo-anchor-v3-multiple-tail-outliers",
                    rejected["reasons"],
                )

        for index in (0, 89):
            with self.subTest(endpoint=index):
                samples = self.timestamped([1000] * 90)
                samples[index]["clock_mhz"] = 1070
                rejected = evaluate_f2_solo_anchor_v3(samples)
                self.assertIn(
                    "f2-solo-anchor-v3-endpoint-outlier",
                    rejected["reasons"],
                )

        bad_neighbor = self.timestamped([1000] * 90)
        bad_neighbor[21]["clock_mhz"] = 1040
        bad_neighbor[22]["clock_mhz"] = 1070
        rejected_neighbor = evaluate_f2_solo_anchor_v3(bad_neighbor)
        self.assertIn(
            "f2-solo-anchor-v3-neighbor-envelope-failure",
            rejected_neighbor["reasons"],
        )
        full_q_failure = self.timestamped([969] + [1000] * 18 + [1070])
        rejected_full_q = evaluate_f2_solo_anchor_v3(full_q_failure)
        self.assertIn(
            "f2-solo-anchor-v3-full-percentile-failure",
            rejected_full_q["reasons"],
        )

        retained_q_failure = self.timestamped([969] + [1000] * 20)
        retained_q_failure[10]["clock_mhz"] = 1070
        rejected_retained_q = evaluate_f2_solo_anchor_v3(
            retained_q_failure
        )
        self.assertNotIn(
            "f2-solo-anchor-v3-full-percentile-failure",
            rejected_retained_q["reasons"],
        )
        self.assertIn(
            "f2-solo-anchor-v3-retained-percentile-failure",
            rejected_retained_q["reasons"],
        )

        changed_median = self.timestamped([1000, 1100, 1001])
        rejected_median = evaluate_f2_solo_anchor_v3(changed_median)
        self.assertIn(
            "f2-solo-anchor-v3-retained-median-changed",
            rejected_median["reasons"],
        )

        nonmonotonic = self.timestamped([1000] * 20)
        nonmonotonic[10]["time_ns"] = nonmonotonic[9]["time_ns"]
        rejected_time = evaluate_f2_solo_anchor_v3(nonmonotonic)
        self.assertIn(
            "f2-solo-anchor-v3-timestamp-order-failure",
            rejected_time["reasons"],
        )
        self.assertFalse(
            evaluate_strict_own_clock(
                [sample["clock_mhz"] for sample in interior]
            )["valid"]
        )

    def test_scope_and_pair_launcher_guard(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_case_device_qualification(
                self.source({"a": [0, 0], "b": [4, 4]}, case_id="F1-00", schema="gap-case-device-branch-qualification-v1")
            )
        passed = evaluate_case_device_qualification(
            self.source({"control": [0, 0], "fa4": [4, 4], "linked": [0, 0]})
        )
        with tempfile.TemporaryDirectory() as directory:
            pair_root = Path(directory) / "pairs"
            require_case_qualification_pass(passed, pair_root)
            pair_root.mkdir()
            (pair_root / "attempt-00").mkdir()
            with self.assertRaises(RuntimeError):
                require_case_qualification_pass(passed, pair_root)
        self.assertNotIn("solo_sha256", passed)

    def test_rotated_fallback_is_deterministic_once_each(self) -> None:
        self.assertEqual(rotated_fallback_order([1, 3, 7], 3), [3, 7, 1])
        waited: list[int] = []
        dispatcher = CaseDeviceFallback(
            [1, 3, 7], 3, wait_for_lock=waited.append
        )
        first = dispatcher.current()
        self.assertEqual(first, 3)
        ineligible = evaluate_case_device_qualification(
            self.source(
                {"control": [0, 4], "fa4": [4, 4], "linked": [0, 0]},
                physical_index=3,
            )
        )
        self.assertIsNone(dispatcher.consume(ineligible))
        second = dispatcher.current()
        self.assertEqual(second, 7)
        passed = evaluate_case_device_qualification(
            self.source(
                {"control": [0, 0], "fa4": [4, 4], "linked": [0, 0]},
                physical_index=7,
            )
        )
        self.assertEqual(dispatcher.consume(passed), 7)
        self.assertEqual(waited, [3, 7])


class FastProtocolTests(unittest.TestCase):
    GPU = {
        "physical_index": 2,
        "uuid": "GPU-test",
        "pci_bus_id": "0000:02:00.0",
    }

    @classmethod
    def leg(
        cls, variant: str, median: float, depth: int = 0, gpu: dict | None = None
    ) -> dict:
        return seal_leg(
            protocol_variant=variant,
            case_id="case-0",
            gpu=gpu or cls.GPU,
            eligibility_ledger_sha256="a" * 64,
            median_ratio_linked_over_reference=median,
            result_path="/tmp/result.json",
            escalation_depth=depth,
        )

    def test_inclusive_trigger_boundaries(self) -> None:
        expected = {
            0.959999: False,
            0.96: True,
            0.97: True,
            0.98: True,
            0.980001: False,
        }
        for value, triggered in expected.items():
            with self.subTest(value=value):
                self.assertEqual(escalation_triggered(value), triggered)

    def test_protocol_factories_and_f2_solos(self) -> None:
        config = {
            "stream_map_sha256": "a" * 64,
            "stream_classes": ["class"],
            "inputs": {},
            "arguments": {},
        }
        fast = Protocol.definitive(Path("/tmp/fast"), config)
        self.assertEqual(
            (fast.slot_seconds, fast.discard_seconds, fast.valid_pairs, fast.attempt_cap),
            (3.0, 1.0, 8, 24),
        )
        original = Protocol.original_duration(Path("/tmp/original"), config)

        self.assertEqual(
            (original.slot_seconds, original.discard_seconds, original.valid_pairs, original.attempt_cap),
            (30.0, 5.0, 20, 60),
        )
        f2_fast = Protocol.f2_cross_source(
            Path("/tmp/f2-fast"), config, mode="fast"
        )
        f2_original = Protocol.f2_cross_source(
            Path("/tmp/f2-original"), config, mode="original"
        )
        self.assertEqual(f2_fast.protocol_variant(), "f2-fast-cross-source-v1")
        self.assertEqual(f2_original.protocol_variant(), "f2-original-cross-source-v1")
        self.assertEqual(
            (f2_original.slot_seconds, f2_original.discard_seconds, f2_original.steady_seconds),
            (30.0, 6.0, 24.0),
        )
        self.assertEqual(
            f2_fast.solo_protocol().record()["protocol_variant"],
            "f2-fast-cross-solo-v1",
        )
        self.assertEqual(
            f2_original.solo_protocol().record()["protocol_variant"],
            "f2-original-cross-solo-v1",
        )
        self.assertEqual(
            (f2_fast.solo_protocol().slot_seconds, f2_fast.solo_protocol().discard_seconds),
            (10.0, 1.0),
        )
        negative_rehearsal = Protocol.negative_launch_scale(
            Path("/tmp/negative-rehearsal"), config, mode="rehearsal"
        )
        negative_definitive = Protocol.negative_launch_scale(
            Path("/tmp/negative-definitive"), config, mode="definitive"
        )
        self.assertEqual(
            (
                negative_rehearsal.valid_pairs,
                negative_rehearsal.attempt_cap,
                negative_rehearsal.warmups_per_image,
                negative_rehearsal.protocol_variant(),
            ),
            (3, 9, 1000, "negative-launch-scale-v1"),
        )
        self.assertEqual(
            (
                negative_definitive.valid_pairs,
                negative_definitive.attempt_cap,
                negative_definitive.slot_seconds,
                negative_definitive.record()["configured_batch_quantum_ms"],
            ),
            (30, 90, 1.0, 100.0),
        )

    def test_no_trigger_uses_fast_without_pooling(self) -> None:
        trigger, decision = decide_fast(self.leg("fast-strict-v1", 0.99))
        self.assertFalse(trigger["triggered"])
        self.assertEqual(decision["decision_source"], "fast")
        self.assertEqual(decision["decision_median_ratio_linked_over_reference"], 0.99)
        self.assertFalse(decision["pairs_pooled"])
        self.assertEqual(read_decision(decision), decision)

    def test_trigger_uses_one_same_gpu_consumer_specific_long_leg(self) -> None:
        fast = self.leg("gap-fast-v1", 0.97)
        original = self.leg("gap-original-v1", 0.975, depth=1)
        evaluation = seal_trigger(fast)
        trigger, decision = decide_fast(
            fast, original, trigger_evaluation=evaluation
        )
        self.assertTrue(trigger["triggered"])
        self.assertEqual(
            decision["decision_source"], "original-duration-escalation"
        )
        self.assertEqual(decision["decision_median_ratio_linked_over_reference"], 0.975)
        self.assertEqual(
            trigger["trigger_evaluation_sha256"],
            evaluation["trigger_evaluation_sha256"],
        )
        self.assertEqual(decision["escalation_count"], 1)
        self.assertTrue(decision["same_gpu"])
        with self.assertRaises(ValueError):
            decide_fast(fast, self.leg("original-escalation-strict-v1", 0.975, depth=1))
        wrong_gpu = {**self.GPU, "physical_index": 3}
        with self.assertRaises(ValueError):
            decide_fast(fast, self.leg("gap-original-v1", 0.975, depth=1, gpu=wrong_gpu))

    def test_f2_audit_only_and_unknown_schema_fail_closed(self) -> None:
        fast = self.leg("f2-fast-cross-source-v1", 0.97)
        original = self.leg("f2-original-cross-source-v1", 0.90, depth=1)
        _trigger, decision = decide_fast(
            fast, original, direct_f2_audit_only=True
        )
        self.assertTrue(decision["direct_f2_audit_only"])
        bad = dict(decision)
        bad["schema_version"] = "unknown"
        with self.assertRaises(ValueError):
            read_decision(bad)
        with self.assertRaises(ValueError):
            self.leg("unknown-fast", 1.0)








class AssignmentPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        # The synthetic discovery snapshots below use fixed GPU indices; pin the
        # configured-GPU view and concurrency cap so the tests are machine-independent.
        import evaluator.harness.parity.assign as assign_module
        self._saved_configured_gpus = assign_module._CONFIGURED_GPUS
        assign_module._CONFIGURED_GPUS = (1, 2, 3, 4, 5, 7, 9)
        self.addCleanup(
            setattr, assign_module, "_CONFIGURED_GPUS", self._saved_configured_gpus
        )
        self.enterContext(patch.object(assign_module, "MAX_CONCURRENT_PHYSICAL_GPUS", 3))
        self.enterContext(patch.object(assign_module, "_ALLOW_UNLOCKED", True))
        self.enterContext(patch.dict(os.environ, {
            "PIPELINK_ASSIGNED_GPU": "",
            "CUDA_VISIBLE_DEVICES": "",
        }))

    def test_unlocked_discovery_without_qualifier_rejects_before_workload_execution(self) -> None:
        row = {
            "physical_index": 1, "uuid": "GPU-1", "pci_bus_id": "1",
            "memory_used_mib": 0, "utilization_percent": 0,
            "sm_clock_mhz": 1000, "reason_mask": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("subprocess.run") as execute:
                with self.assertRaises(RuntimeError):
                    discover_idle_gpus(
                        root / "discovery",
                        snapshot_provider=lambda _index: [dict(row)],
                        lock_root=root / "locks",
                        lock_state_dir=root / "markers",
                    )
                execute.assert_not_called()

    def test_locked_discovery_qualifies_policy_and_rejects_unhealthy_clock(self) -> None:
        import evaluator.harness.parity.assign as assign_module

        uuid = "GPU-00000000-0000-0000-0000-000000000001"
        clock_mhz = 1500
        driver = "test-driver"
        marker = {
            "gpu": 1, "uuid": uuid, "sm_mhz": clock_mhz,
            "pid": os.getpid(), "violations": 0,
            "started_at": "2026-01-01T00:00:00Z", "driver_version": driver,
            "policy_token": hashlib.sha256(
                f"{uuid}|{clock_mhz}|{driver}".encode("utf-8")
            ).hexdigest(),
        }
        row = {
            "physical_index": 1, "uuid": uuid, "pci_bus_id": "1",
            "memory_used_mib": 2048, "utilization_percent": 10,
            "sm_clock_mhz": clock_mhz, "reason_mask": 0,
            "temperature_c": 40, "power_draw_w": 100, "power_limit_w": 500,
        }
        self.enterContext(patch.object(assign_module, "_ALLOW_UNLOCKED", False))
        self.enterContext(patch.object(assign_module, "_LOCK_SM_MHZ", clock_mhz))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "markers"
            state.mkdir()
            (state / "gpu1.json").write_text(json.dumps(marker), encoding="utf-8")
            with discover_idle_gpus(
                root / "healthy",
                snapshot_provider=lambda _index: [dict(row)],
                lock_root=root / "locks", lock_state_dir=state,
            ) as lease:
                assignment = lease.assign(["locked-case"])
                self.assertEqual(assignment["discovery_mode"], "locked")
                self.assertEqual(
                    assignment["assignments"]["locked-case"]["physical_index"], 1
                )
            qualification = json.loads(
                (root / "healthy" / "qualification" / "gpu-1" / "qualification.json")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(qualification["status"], "PASS")
            self.assertEqual(
                qualification["qualification_satisfied_by"], "gpulockd-daemon-policy"
            )
            row["sm_clock_mhz"] = 1400
            with self.assertRaises(RuntimeError):
                discover_idle_gpus(
                    root / "unhealthy",
                    snapshot_provider=lambda _index: [dict(row)],
                    lock_root=root / "locks", lock_state_dir=state,
                )

    def test_assignment_is_lexicographic_round_robin(self) -> None:
        self.assertEqual(assign_cases(["z", "a", "m", "b"], [5, 4]), {"a": 4, "b": 5, "m": 4, "z": 5})
    def test_qualification_filters_before_numeric_order_without_retry(self) -> None:
        rows = [
            {"physical_index": index, "uuid": f"GPU-{index}", "pci_bus_id": str(index), "memory_used_mib": 0, "memory_total_mib": 1000, "utilization_percent": 0, "sm_clock_mhz": 100, "reason_mask": 0}
            for index in (1, 2)
        ]
        calls: dict[int, int] = {}

        def provider(index: int | None) -> list[dict]:
            return [dict(row) for row in rows if index is None or row["physical_index"] == index]

        def qualifier(row: dict, _epoch: str, _archive: Path) -> dict:
            index = row["physical_index"]
            calls[index] = calls.get(index, 0) + 1
            return {
                "qualifier_version": "fa4-f2-00-telemetry-qualification-v2",
                "status": "PASS" if index == 2 else "INELIGIBLE_TELEMETRY_STATE",
                "attempt_count": 1,
                "retry_permitted_same_epoch": False,
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with discover_idle_gpus(
                root / "some", snapshot_provider=provider,
                qualifier=qualifier, lock_root=root / "locks",
            ) as lease:
                self.assertEqual(
                    [row["physical_index"] for row in lease.ordered_devices],
                    [2],
                )
                assigned = lease.assign(["a"])
                self.assertEqual(
                    assigned["assignments"]["a"]["physical_index"], 2
                )
            self.assertEqual(calls, {1: 1, 2: 1})

            def fail_all(row: dict, epoch: str, archive: Path) -> dict:
                return {"status": "INELIGIBLE_TELEMETRY_STATE"}

            with self.assertRaises(RuntimeError):
                discover_idle_gpus(
                    root / "empty", snapshot_provider=provider,
                    qualifier=fail_all, lock_root=root / "locks",
                )

    def test_dynamic_discovery_lock_recheck_and_ordering(self) -> None:
        rows = [
            {"physical_index": 3, "uuid": "GPU-3", "pci_bus_id": "3", "memory_used_mib": 0, "memory_total_mib": 1000, "utilization_percent": 1, "sm_clock_mhz": 100, "reason_mask": 0},
            {"physical_index": 0, "uuid": "GPU-0", "pci_bus_id": "0", "memory_used_mib": 0, "memory_total_mib": 1000, "utilization_percent": 0, "sm_clock_mhz": 100, "reason_mask": 0},
            {"physical_index": 2, "uuid": "GPU-2", "pci_bus_id": "2", "memory_used_mib": 100, "memory_total_mib": 1000, "utilization_percent": 0, "sm_clock_mhz": 100, "reason_mask": 0},
            {"physical_index": 4, "uuid": "GPU-4", "pci_bus_id": "4", "memory_used_mib": 1024, "memory_total_mib": 2000, "utilization_percent": 0, "sm_clock_mhz": 100, "reason_mask": 0},
            {"physical_index": 1, "uuid": "GPU-1", "pci_bus_id": "1", "memory_used_mib": 1, "memory_total_mib": 1000, "utilization_percent": 0, "sm_clock_mhz": 100, "reason_mask": 0},
        ]

        def provider(index: int | None) -> list[dict]:
            return [dict(row) for row in rows if index is None or row["physical_index"] == index]
        def qualifier(row: dict, epoch: str, _archive: Path) -> dict:
            return {
                "schema_version": "m3-gpu-telemetry-qualification-result-v2",
                "qualifier_version": "fa4-f2-00-telemetry-qualification-v2",
                "epoch_id": epoch,
                "physical_index": row["physical_index"],
                "gpu_uuid": row["uuid"],
                "status": "PASS",
                "attempt_count": 1,
                "retry_permitted_same_epoch": False,
                "telemetry_only": True,
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with discover_idle_gpus(
                root / "first", snapshot_provider=provider,
                qualifier=qualifier, lock_root=root / "locks",
            ) as lease:
                self.assertEqual(
                    [row["physical_index"] for row in lease.ordered_devices],
                    [1, 2],
                )
                assignment = lease.assign(["z", "a"])
                self.assertEqual(
                    assignment["assignments"]["a"]["physical_index"], 1
                )
                self.assertEqual(
                    assignment["assignments"]["z"]["physical_index"], 2
                )
                with self.assertRaises(RuntimeError):
                    discover_idle_gpus(
                        root / "second", snapshot_provider=provider,
                        qualifier=qualifier, lock_root=root / "locks",
                    )

    def test_exact_absence_failover_record(self) -> None:
        record = {
            "case_id": "F1-00",
            "eligible_gpu_order": [2, 7, 9],
            "assigned_device": {"physical_index": 7, "uuid": "GPU-old", "pci_bus_id": "0000:01:00.0"},
            "next_device": {"physical_index": 9, "uuid": "GPU-new", "pci_bus_id": "0000:02:00.0"},
            "job_interval": {"start_ns": 1, "end_ns": 5_000_000_000},
            "trigger": {
                "class": "device-absent-two-probes",
                "probes": [
                    {"timestamp_ns": 2, "uuid_present": False, "output": "absent"},
                    {"timestamp_ns": 1_000_000_002, "uuid_present": False, "output": "absent"},
                ],
            },
            "partial_evidence_archive": "/tmp/partial",
            "raw_stderr_evidence": "",
        }
        result = build_failover_record(record)
        self.assertTrue(result["objective_trigger"]["eligible"])
        self.assertEqual(result["restart"]["attempt"], 0)
        record["trigger"]["probes"][1]["timestamp_ns"] += 1
        with self.assertRaises(ValueError):
            build_failover_record(record)

    def test_preflight_summary_is_closed_from_entries(self) -> None:
        digest = "1" * 64
        entry = {
            "entry_id": "row-0", "case_id": "F1-00", "image_id": "linked",
            "shape": [256, 256, 256], "arguments": {"alpha": 1.25},
            "eligibility": {"kind": "cutlass-can-implement", "eligible": True, "status_code": 0, "status_string": "kSuccess"},
            "hashes": {"image": digest, "config": digest},
        }
        record = {
            "completed_before_measurement_freeze": True,
            "fail_closed_no_timed_slot_ran_first": True,
            "device": {"physical_index": 5, "uuid": "GPU-test", "name": "Blackwell"},
            "environment": {"toolkit": "CUDA 13.3.0", "driver": "test", "target": "sm103"},
            "hashes": {category: {category: digest} for category in ("inputs", "images", "catalogs", "configs")},
            "entries": [entry], "summary": {"expected_count": 1},
        }
        result = build_preflight_record(record)
        self.assertEqual(result["summary"], {"expected_count": 1, "observed_count": 1, "all_true": True, "fail_closed_no_timed_slot_ran_first": True})


class RecordWriterTests(unittest.TestCase):
    @staticmethod
    def correctness(subject_id: str, rtol: float, atol: float) -> dict:
        barrier = {"passed": True, "image_derived": True}
        barrier.update({field: True for field in BARRIER_AUDIT_FIELDS})
        return {
            "subject_id": subject_id,
            "case_id": "row-0",
            "image_id": "linked",
            "stream_map_sha256": "a" * 64,
            "stream_classes": ["ns=0x0001/case=0/role=1"],
            "gate_a": {"bitwise_identical": True},
            "gate_b": {
                "passed": True,
                "tolerance": {"rtol": rtol, "atol": atol},
                "nonfinite_policy": NONFINITE_POLICY,
                "reference_outputs_finite": True,
            },
            "safety": {"racecheck_hazards": 0, "synccheck_errors": 0},
            "barrier_audit": barrier,
        }

    def test_closed_correctness_tolerance_table(self) -> None:
        f2 = build_correctness_record(self.correctness("family2", 2.0e-2, 2.0e-2))
        self.assertEqual(f2["schema_version"], "m3-correctness-record-v2")
        f1 = build_correctness_record(self.correctness("family1", 1.0e-2, 1.0e-3))
        self.assertTrue(f1["all_true"])
        with self.assertRaises(ValueError):
            build_correctness_record(self.correctness("family2", 1.0e-2, 1.0e-3))
        with self.assertRaises(ValueError):
            build_correctness_record(self.correctness("family1", 2.0e-2, 2.0e-2))

    @staticmethod
    def resource_entry(
        symbol: str, traits: dict, vector: dict, marker: str, *, stock: bool = False
    ) -> dict:
        entry = {
            "entry_symbol": symbol,
            "specialization_traits": traits,
            "cubin_sha256": marker * 64,
            "sass_sha256": marker * 64,
            "elf_sha256": marker * 64,
            "resource_vector": {**vector, "LOCAL": 0},
            "evidence": {
                "cuobjdump_resources_sha256": marker * 64,
                "direct_spill_counters_sha256": marker * 64,
                "debug_frame_sha256": marker * 64,
                "reachable_symbols_sha256": marker * 64,
                "call_local_site_inventory_sha256": marker * 64,
                "counter_source": "direct-final-link",
            },
        }
        if stock:
            entry.update({
                "source_sha256": marker * 64,
                "compile_command": ["nvcc", "-arch=sm_103a", "example77.cu"],
                "toolkit": "CUDA 13.3.0",
                "target": "sm103",
            })
        return entry

    @classmethod
    def family2_resource(cls) -> dict:
        linked_vector = {"STACK": 112, "spill_store_bytes": 92, "spill_load_bytes": 92}
        stock_vector = {"STACK": 192, "spill_store_bytes": 188, "spill_load_bytes": 284}
        traits = {
            "head_dim": 64,
            "mask": "noncausal",
            "fixed_length": True,
            "scheduler": "1cta",
        }
        return {
            "subject_id": "family2",
            "case_id": "F2-00",
            "image_id": "F2-linked",
            "image_sha256": "b" * 64,
            "sass_sha256": "b" * 64,
            "resources": {
                "REG": 128,
                "STACK": 112,
                "LOCAL": 0,
                "spill_load_bytes": 92,
                "spill_store_bytes": 92,
                "dynamic_smem_bytes": 1024,
                "tmem": {},
                "named_mbarrier_map": {},
                "warp_map": {},
                "launch_topology": {},
                "occupancy": {},
                "blocks_per_sm": 1,
                "control_blocks_per_sm": 1,
            },
            "family2_comparator": {
                "mapping_id": "F2-00:d64-nc-1cta",
                "row_id": "F2-00",
                "linked": cls.resource_entry("f2_linked_d64_nc", traits, linked_vector, "b"),
                "control": cls.resource_entry("f2_control_d64_nc", traits, linked_vector, "c"),
                "stock": cls.resource_entry(
                    "stock_example77_d64_nc", traits, stock_vector, "d", stock=True
                ),
            },
        }

    def test_family2_componentwise_equal_and_below_passes(self) -> None:
        result = build_resource_record(self.family2_resource())
        self.assertEqual(result["schema_version"], "m3-resource-record-v2")
        self.assertTrue(result["headline_predicate"]["passed"])
        self.assertTrue(result["headline_predicate"]["linked_equals_control_componentwise"])
        self.assertTrue(result["headline_predicate"]["linked_no_greater_than_stock_componentwise"])

    def test_family2_one_byte_component_regression_fails(self) -> None:
        for resource_field, vector_field in (
            ("STACK", "STACK"),
            ("spill_store_bytes", "spill_store_bytes"),
            ("spill_load_bytes", "spill_load_bytes"),
        ):
            with self.subTest(component=vector_field):
                record = self.family2_resource()
                record["resources"][resource_field] += 1
                record["family2_comparator"]["linked"]["resource_vector"][vector_field] += 1
                result = build_resource_record(record)
                self.assertFalse(result["headline_predicate"]["passed"])

    def test_family2_missing_mapping_fails_closed(self) -> None:
        record = self.family2_resource()
        del record["family2_comparator"]
        with self.assertRaises(ValueError):
            build_resource_record(record)

    def test_family1_nonzero_stack_fails(self) -> None:
        record = self.family2_resource()
        record["subject_id"] = "family1"
        del record["family2_comparator"]
        record["resources"]["STACK"] = 1
        result = build_resource_record(record)
        self.assertFalse(result["headline_predicate"]["passed"])

    @classmethod
    def family1_exception_resource(cls) -> dict:
        linked_vector = {"STACK": 120, "spill_store_bytes": 476, "spill_load_bytes": 556}
        control_vector = {"STACK": 152, "spill_store_bytes": 646, "spill_load_bytes": 676}
        stock_vector = {"STACK": 256, "spill_store_bytes": 936, "spill_load_bytes": 1192}
        traits = {
            "winner_operation": "stream-k-2sm-exact-name",
            "winner_config_sha256": "a" * 64,
            "variant": "v1",
            "entry_key": "03f6c035cb55a217",
        }

        def wrapped(symbol: str, vector: dict, marker: str, *, stock: bool = False) -> dict:
            entry = cls.resource_entry(symbol, traits, vector, marker, stock=stock)
            evidence = entry["evidence"]
            evidence["counter_source"] = "verbose-relink-byte-identical-sass"
            evidence.update({
                "measured_elf_sha256": entry["elf_sha256"],
                "verbose_relink_elf_sha256": "e" * 64,
                "measured_sass_sha256": entry["sass_sha256"],
                "verbose_relink_sass_sha256": "f" * 64,
                "decoded_sass_identity_sha256": "1" * 64,
                "non_code_only_difference_sha256": "2" * 64,
            })
            return {"register_cap": 255, "entry": entry}

        record = cls.family2_resource()
        record["subject_id"] = "family1"
        record["case_id"] = "F1-streamk-v1"
        record["image_id"] = "F1-streamk-linked-v1"
        del record["family2_comparator"]
        record["resources"].update({
            "STACK": linked_vector["STACK"],
            "spill_store_bytes": linked_vector["spill_store_bytes"],
            "spill_load_bytes": linked_vector["spill_load_bytes"],
        })
        record["family1_exception"] = {
            "mapping_id": "03f6c035cb55a217:v1",
            "row_id": "F1-streamk-v1",
            "winner_config_sha256": "a" * 64,
            "variant": "v1",
            "performance_observations_used": 0,
            "discriminator_sha256": "3" * 64,
            "waived_components": ["STACK", "spill_store_bytes", "spill_load_bytes"],
            "cap_domain": {
                "requested_cap_min": 1,
                "requested_cap_max": 255,
                "all_requested_caps_tested": True,
                "effective_cap_count": 206,
                "effective_caps_sha256": "4" * 64,
                "direct_counter_table_sha256": "5" * 64,
                "per_component": {
                    "STACK": {
                        "tested_effective_cap_count": 206,
                        "minimum": 120,
                        "zero_unattainable": True,
                    },
                    "spill_store_bytes": {
                        "tested_effective_cap_count": 206,
                        "minimum": 476,
                        "zero_unattainable": True,
                    },
                    "spill_load_bytes": {
                        "tested_effective_cap_count": 206,
                        "minimum": 556,
                        "zero_unattainable": True,
                    },
                },
            },
            "linked": wrapped("f1_linked_streamk_v1", linked_vector, "b"),
            "control": wrapped("f1_control_streamk_v1", control_vector, "c"),
            "stock": wrapped("f1_stock_streamk_exact", stock_vector, "d", stock=True),
        }
        return record

    def test_family1_complete_streamk_exception_passes(self) -> None:
        result = build_resource_record(self.family1_exception_resource())
        predicate = result["headline_predicate"]
        self.assertTrue(predicate["passed"])
        self.assertEqual(
            predicate["mode"], "family1-narrow-cap-domain-componentwise-exception"
        )
        self.assertTrue(
            all(
                component["exception_eligible"]
                for component in predicate["component_predicates"].values()
            )
        )

    def test_family1_incomplete_zero_proof_restores_absolute_rule(self) -> None:
        record = self.family1_exception_resource()
        proof = record["family1_exception"]["cap_domain"]["per_component"]["STACK"]
        proof["zero_unattainable"] = False
        proof["minimum"] = 0
        result = build_resource_record(record)
        stack = result["headline_predicate"]["component_predicates"]["STACK"]
        self.assertFalse(stack["exception_eligible"])
        self.assertTrue(stack["absolute_zero_fallback"])
        self.assertFalse(result["headline_predicate"]["passed"])

    def test_family1_exception_requires_cap255(self) -> None:
        record = self.family1_exception_resource()
        record["family1_exception"]["linked"]["register_cap"] = 254
        with self.assertRaises(ValueError):
            build_resource_record(record)


class FakeSampler:
    reason_mask = 0x4
    def __init__(self, _gpu: int, _field: str) -> None:
        self.started_ns = 0
        self.stopped_ns: int | None = None

    def start(self) -> None:
        self.started_ns = time.time_ns() - 20_000_000

    def wait_ready(self) -> None:
        pass

    def stop(self) -> None:
        self.stopped_ns = time.time_ns() + 20_000_000

    def samples(self) -> list[ClockSample]:
        end = self.stopped_ns or time.time_ns()
        return [
            ClockSample(timestamp, True, 1000, 40, self.reason_mask, "fixture")
            for timestamp in range(self.started_ns, end + 1, 5_000_000)
        ]


class TraceGapSampler(FakeSampler):
    trace_file: Path | None = None

    def samples(self) -> list[ClockSample]:
        if self.trace_file is None or not self.trace_file.exists():
            return []
        rows = []
        for line in self.trace_file.read_text(encoding="utf-8").splitlines():
            window = json.loads(line)
            reference = window["role"] == "reference"
            clock = 1000 if reference else 1500
            power = 800.0 if reference else 900.0
            rows.extend(
                ClockSample(timestamp, True, clock, 40, 0x4, "fixture", power, 1100.0)
                for timestamp in range(
                    window["window_start_ns"],
                    window["window_end_ns"] + 1,
                    5_000_000,
                )
            )
        return rows


class OriginalPairSampler(FakeSampler):
    reason_mask = 0x0


class InvalidPairSampler(FakeSampler):
    reason_mask = 0x5


def fake_gap_policy(_gpu: int) -> dict:
    return {
        "gpu_uuid": "GPU-test",
        "pci_bus_id": "0000:00:00.0",
        "configured_power_limit_w": 1100.0,
        "enforced_power_limit_w": 1100.0,
        "application_graphics_clock_mhz": 2032,
        "application_memory_clock_mhz": 3996,
        "default_graphics_clock_mhz": 2032,
        "default_memory_clock_mhz": 3996,
        "persistence_mode": "Enabled",
        "compute_mode": "Default",
    }


class RunnerTests(unittest.TestCase):
    def test_full_cpu_pair_archive_and_median(self) -> None:
        fixture = Path(__file__).with_name("mock_slot.py").resolve()
        digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
        reference = ImageSpec("ref", (sys.executable, str(fixture)), fixture, digest)
        linked = ImageSpec("linked", (sys.executable, str(fixture)), fixture, digest)
        case_config = {
            "stream_map_sha256": "2" * 64,
            "stream_classes": ["ns=0x0001/case=0/role=1"],
            "inputs": {"shape": [1, 1, 1]},
            "arguments": {"alpha": 1.25},
        }
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "case"
            protocol = Protocol._self_test(archive, case_config)
            result = run_case(
                "fixture-00", reference, linked, 5, protocol,
                sampler_factory=FakeSampler,
                device_identity_provider=lambda gpu: {
                    "physical_index": gpu, "uuid": "GPU-test", "pci_bus_id": "0000:00:00.0", "name": "Blackwell",
                },
                clock_support_provider=lambda _gpu: {"field": "clocks_event_reasons.active", "probe": {}},
            )
            summary = result["summary"]
            self.assertEqual(summary["pair_valid_pairs_collected"], 2)
            self.assertEqual(summary["original_valid_count"], 0)
            self.assertEqual(summary["amended_valid_count"], 2)
            self.assertEqual(summary["pair_valid_count"], 2)
            self.assertEqual(summary["validity_count_invariant"]["equal"], True)
            self.assertAlmostEqual(summary["median_ratio_linked_over_reference"], 0.998)
            self.assertTrue((archive / "pairs" / "pair-00.json").is_file())
            self.assertTrue((archive / "pairs" / "pair-01.json").is_file())
            self.assertEqual(result["pairs"][0]["order"], ["reference", "linked"])
            self.assertEqual(result["pairs"][1]["order"], ["linked", "reference"])
            self.assertEqual(summary["schema_version"], "m3-parity-summary-v2")
            self.assertNotIn("accepted_ratios_linked_over_reference", summary)
            self.assertEqual(len(summary["pair_valid_ratios_linked_over_reference"]), 2)
            self.assertEqual(result["pairs"][0]["schema_version"], "m3-parity-pair-v2")
            self.assertTrue(result["pairs"][0]["pair_valid"])
            self.assertEqual(result["pairs"][0]["validity_class"], "amended-valid")

    def test_original_valid_pairs_are_collected_chronologically(self) -> None:
        fixture = Path(__file__).with_name("mock_slot.py").resolve()
        digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
        reference = ImageSpec("ref", (sys.executable, str(fixture)), fixture, digest)
        linked = ImageSpec("linked", (sys.executable, str(fixture)), fixture, digest)
        case_config = {
            "stream_map_sha256": "4" * 64,
            "stream_classes": ["ns=0x0001/case=0/role=1"],
            "inputs": {"shape": [1, 1, 1]},
            "arguments": {"alpha": 1.25},
        }
        with tempfile.TemporaryDirectory() as directory:
            result = run_case(
                "fixture-original", reference, linked, 5,
                Protocol._self_test(Path(directory) / "case", case_config),
                sampler_factory=OriginalPairSampler,
                device_identity_provider=lambda gpu: {
                    "physical_index": gpu, "uuid": "GPU-test",
                    "pci_bus_id": "0000:00:00.0", "name": "Blackwell",
                },
                clock_support_provider=lambda _gpu: {
                    "field": "clocks_event_reasons.active", "probe": {},
                },
            )
            summary = result["summary"]
            self.assertEqual(summary["original_valid_count"], 2)
            self.assertEqual(summary["amended_valid_count"], 0)
            self.assertEqual(summary["pair_valid_count"], 2)
            self.assertEqual(summary["operating_regime"], "original-only")
            self.assertEqual([pair["attempt"] for pair in result["pairs"] if pair["pair_valid"]], [0, 1])

    def test_gap_runner_emits_v6_sealed_records_and_summary(self) -> None:
        fixture = Path(__file__).with_name("mock_slot.py").resolve()
        digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
        case_config = {
            "stream_map_sha256": "6" * 64,
            "stream_classes": ["ns=0x0102/case=0/role=1"],
            "inputs": {"shape": [4096, 4096, 4096]},
            "arguments": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "windows.jsonl"
            archive = Path(directory) / "case"
            command = (sys.executable, str(fixture), "--trace-file", str(trace))
            reference = ImageSpec("ref", command, fixture, digest)
            linked = ImageSpec("linked", command, fixture, digest)
            TraceGapSampler.trace_file = trace
            result = run_case(
                "A-solver-D4", reference, linked, 5,
                Protocol._self_test(
                    archive, case_config,
                    gap_row_allowlist=["A-solver-D4"],
                ),
                sampler_factory=TraceGapSampler,
                device_identity_provider=lambda gpu: {
                    "physical_index": gpu, "uuid": "GPU-test",
                    "pci_bus_id": "0000:00:00.0", "name": "Blackwell",
                },
                clock_support_provider=lambda _gpu: {
                    "field": "clocks_event_reasons.active", "probe": {},
                },
                gap_policy_provider=fake_gap_policy,
            )
            summary = result["summary"]
            self.assertEqual(summary["schema_version"], "m3-parity-summary-v6")
            self.assertEqual(summary["strict_pair_valid_count"], 0)
            self.assertEqual(summary["gap_pair_valid_count"], 2)
            self.assertEqual(summary["branch_combination_counts"], {"power-cap/power-cap": 2})
            self.assertTrue(summary["ratio_blind_eligibility"])
            self.assertTrue(summary["throughput_read_after_eligibility_seal"])
            self.assertTrue(summary["timestamp_coverage_identical"])
            self.assertEqual(summary["mid_series_solo_refresh_count"], 0)
            self.assertEqual(summary["gap_pair_valid_pairs_collected"], 2)
            self.assertEqual(summary["solo_throughput"]["reference"]["value"], 100.0)
            self.assertEqual(summary["solo_throughput"]["linked"]["value"], 99.8)
            pair = result["pairs"][0]
            self.assertEqual(
                summary["opening_solo_block_id"],
                pair["opening_solo_block_id"],
            )
            self.assertEqual(pair["schema_version"], "m3-parity-pair-v6")
            self.assertNotIn("pair_valid", pair)
            self.assertEqual(pair["mid_series_solo_refresh_count"], 0)
            self.assertEqual(
                pair["opening_solo_sha256"],
                summary["opening_solo_sha256"],
            )
            self.assertFalse((archive / "solo-refreshes").exists())
            self.assertFalse(pair["strict_classification"]["pair_valid"])
            self.assertTrue(pair["gap_classification"]["gap_pair_valid"])
            self.assertAlmostEqual(pair["ratio_linked_over_reference"], 0.998)
            self.assertTrue((archive / "eligibility-ledger.json").is_file())
            self.assertTrue((archive / "throughput.json").is_file())


    def test_pair_cap_writes_failure_summary(self) -> None:
        fixture = Path(__file__).with_name("mock_slot.py").resolve()
        digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
        reference = ImageSpec("ref", (sys.executable, str(fixture)), fixture, digest)
        linked = ImageSpec("linked", (sys.executable, str(fixture)), fixture, digest)
        case_config = {
            "stream_map_sha256": "3" * 64,
            "stream_classes": ["ns=0x0001/case=0/role=1"],
            "inputs": {"shape": [1, 1, 1]},
            "arguments": {"alpha": 1.25},
        }
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "case"
            protocol = Protocol._self_test(archive, case_config)
            with self.assertRaises(PairCapError):
                run_case(
                    "fixture-cap", reference, linked, 5, protocol,
                    sampler_factory=InvalidPairSampler,
                    device_identity_provider=lambda gpu: {
                        "physical_index": gpu, "uuid": "GPU-test",
                        "pci_bus_id": "0000:00:00.0", "name": "Blackwell",
                    },
                    clock_support_provider=lambda _gpu: {
                        "field": "clocks_event_reasons.active", "probe": {},
                    },
                )
            summary = json.loads((archive / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "EXECUTION_FAIL")
            self.assertTrue(summary["pair_cap_hit"])
            self.assertEqual(summary["attempted_pairs"], 3)
            self.assertEqual(summary["pair_valid_pairs_collected"], 0)
            self.assertEqual(summary["pair_valid_count"], 0)
            self.assertEqual(summary["original_valid_count"], 0)
            self.assertEqual(summary["amended_valid_count"], 0)


if __name__ == "__main__":
    unittest.main()
