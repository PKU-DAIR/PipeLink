"""Frozen M3 common-harness constants.

The normative numeric values below define the shared measurement policy.
Family code imports these values rather than restating measurement policy.
"""
from __future__ import annotations

from pathlib import Path

from tools.pipelink_env import load

# Philox4x32-10 random-access derivation.
PHILOX_SEED = 0x50495045
PHILOX_ROUNDS = 10
PHILOX_LANES = 4
UINT32_MASK = 0xFFFFFFFF
UNIFORM_WORD_OFFSET = 0.5
UNIFORM_SCALE = 2.0**-32
INT4_CARDINALITY = 16
INT4_SIGN_CUTOFF = 8

# Random123 Philox4x32 constants.
PHILOX_M0 = 0xD2511F53
PHILOX_M1 = 0xCD9E8D57
PHILOX_W0 = 0x9E3779B9
PHILOX_W1 = 0xBB67AE85

# Complete random-input namespace table.
NAMESPACE_IDS = {
    "F1": 0x0001,
    "F2": 0x0002,
    "F3": 0x0003,
    "A-specialization": 0x0101,
    "A-solver": 0x0102,
    "A-authoring": 0x0103,
    "A-F1-continuity": 0x0104,
    "N-register-coupling": 0x0201,
    "N-tiny-jit": 0x0202,
}

# Fixed scalar arguments and sentinel.
F1_AUX_SENTINEL = 0x7B
ALPHA = 1.25
BETA = 0.75
AUTHORING_CORRECTNESS_ALPHA = 0.5
REGISTER_OUTPUT_INITIAL = 0.0

# GEMM Gate-B predicate.
GEMM_RTOL = 1.0e-2
GEMM_ATOL = 1.0e-3
ATTENTION_RTOL = 2.0e-2
ATTENTION_ATOL = 2.0e-2

# Closed Gate-B tolerance table.
# Callers identify the family/experiment; no local override.
CORRECTNESS_TOLERANCES = {
    "family1": (GEMM_RTOL, GEMM_ATOL),
    "family2": (ATTENTION_RTOL, ATTENTION_ATOL),
    "family3": (GEMM_RTOL, GEMM_ATOL),
    "A-specialization": (GEMM_RTOL, GEMM_ATOL),
    "A-solver": (GEMM_RTOL, GEMM_ATOL),
    "A-authoring": (GEMM_RTOL, GEMM_ATOL),
    "A-F1-continuity": (GEMM_RTOL, GEMM_ATOL),
    "N-register-coupling": (GEMM_RTOL, GEMM_ATOL),
    "N-tiny-jit": (GEMM_RTOL, GEMM_ATOL),
}

# Exact Gate-A comparison targets and Gate-B independent-semantics requirement.
GATE_A_CONTROL_POLICY = {
    "F1": "exact-same-source-one-TU-control",
    "F2": "same-source-CUTLASS-control",
    "F3": "F3-one-TU-smem-only",
}
GATE_B_REFERENCE_POLICY = "specified-cuBLAS-or-FP32-attention/dequant-reference"
NONFINITE_POLICY = {
    "nan": "both-NaN-same-element-only",
    "infinity": "same-sign-infinity-only",
    "finite": "abs(actual-expected) <= atol + rtol*abs(expected)",
}

# Zero-hazard/error and zero local-storage headline predicates.
REQUIRED_RACECHECK_HAZARDS = 0
REQUIRED_SYNCCHECK_ERRORS = 0
REQUIRED_STACK_BYTES = 0
REQUIRED_LOCAL_BYTES = 0
REQUIRED_SPILL_LOAD_BYTES = 0
REQUIRED_SPILL_STORE_BYTES = 0

# Absolute-zero default, narrow pre-performance Family-1 cap-domain exception,
# and Family-2 componentwise comparator rule.
FAMILY1_RESOURCE_SUBJECT = "family1"
ABSOLUTE_ZERO_RESOURCE_SUBJECTS = frozenset({
    "family1",
    "family3",
    "A-specialization",
    "A-solver",
    "A-authoring",
    "A-F1-continuity",
    "N-register-coupling",
    "N-tiny-jit",
})
FAMILY2_RESOURCE_SUBJECT = "family2"
FAMILY2_RESOURCE_VECTOR_FIELDS = (
    "STACK",
    "spill_store_bytes",
    "spill_load_bytes",
)
FAMILY2_COMPARATOR_EVIDENCE_FIELDS = (
    "cuobjdump_resources_sha256",
    "direct_spill_counters_sha256",
    "debug_frame_sha256",
    "reachable_symbols_sha256",
    "call_local_site_inventory_sha256",
)
FAMILY2_VERBOSE_RELINK_EVIDENCE_FIELDS = (
    "measured_elf_sha256",
    "verbose_relink_elf_sha256",
    "measured_sass_sha256",
    "verbose_relink_sass_sha256",
    "decoded_sass_identity_sha256",
    "non_code_only_difference_sha256",
)
BARRIER_AUDIT_FIELDS = (
    "init",
    "producer_consumer_counts",
    "phases",
    "acquire_commit_wait_release",
    "proxy_fences",
    "tail_release",
    "named_ids",
    "expected_byte_totals",
    "fixed_aliases",
)
RESOURCE_AUDIT_FIELDS = (
    "REG",
    "STACK",
    "LOCAL",
    "spill_load_bytes",
    "spill_store_bytes",
    "SASS",
    "dynamic_smem_bytes",
    "TMEM",
    "named_mbarrier_map",
    "warp_map",
    "launch_topology",
    "occupancy",
)

# Preflight completeness/fail-closed fields.
PREFLIGHT_HASH_CATEGORIES = ("inputs", "images", "catalogs", "configs")
PREFLIGHT_COMPLETED_BEFORE_FREEZE = True
PREFLIGHT_NO_TIMED_SLOT_FIRST = True

# Fast protocol and one-level original-duration escalation.
SLOTS_PER_PAIR = 2
WARMUPS_PER_IMAGE = 5
SLOT_SECONDS = 3.0
LEADING_DISCARD_SECONDS = 1.0
STEADY_SECONDS = 2.0
CLOCK_SAMPLE_HZ = 10.0
CLOCK_SAMPLE_INTERVAL_SECONDS = 0.1
CLOCK_WINDOW_FRACTION = 0.03
ORIGINAL_ALLOWED_REASON_MASK = 0x1
SW_POWER_CAP_REASON_MASK = 0x4
VALID_PAIRS = 8
ATTEMPT_CAP = 24
RATIO_FLOOR = 0.97
MAX_BATCH_QUANTUM_MS = 100.0
ESCALATION_TRIGGER_LOW = 0.96
ESCALATION_TRIGGER_HIGH = 0.98
ORIGINAL_SLOT_SECONDS = 30.0
ORIGINAL_DISCARD_SECONDS = 5.0
ORIGINAL_STEADY_SECONDS = 25.0
ORIGINAL_VALID_PAIRS = 20
ORIGINAL_ATTEMPT_CAP = 60
FAST_SOLO_SECONDS = 10.0
FAST_SOLO_DISCARD_SECONDS = 1.0
F2_ORIGINAL_CROSS_SLOT_SECONDS = 30.0
F2_ORIGINAL_CROSS_DISCARD_SECONDS = 6.0
F2_ORIGINAL_CROSS_STEADY_SECONDS = 24.0
ORIGINAL_SOLO_SECONDS = 30.0
ORIGINAL_SOLO_DISCARD_SECONDS = 5.0
NEGATIVE_LAUNCH_SLOT_SECONDS = 1.0
NEGATIVE_LAUNCH_DISCARD_SECONDS = 0.0
NEGATIVE_LAUNCH_STEADY_SECONDS = 1.0
NEGATIVE_LAUNCH_VALID_PAIRS = 30
NEGATIVE_LAUNCH_ATTEMPT_CAP = 90
NEGATIVE_LAUNCH_WARMUPS = 1000
NEGATIVE_LAUNCH_BATCH_QUANTUM_MS = 100.0
MIN_COVERING_SAMPLES_PER_SLOT = 2

# Coordinate-ladder (sweep) protocol.  These identifiers are
# additive and deliberately excluded from every frozen protocol allowlist.
SWEEP_PROTOCOL_NAME = "sweep"
SWEEP_PROTOCOL_VARIANT = "sweep-v1"
EXPERIMENT_ONLY_PROTOCOL_VARIANTS = frozenset({SWEEP_PROTOCOL_VARIANT})

# The gap-row classifier is dispatchable only for these exact case IDs.
GAP_ROW_CLASSIFIER = "gap-ex-ante-solo-v2"
STRICT_PAIR_CLASSIFIER = "strict-pair-valid-v2"
GAP_SPECIALIZATION_CASE_IDS = (
    "A-specialization-0",
    "A-specialization-1",
    "A-specialization-2",
)
GAP_SOLVER_CASE_IDS = (
    "A-solver-D4",
    "A-solver-D2",
    "A-solver-R-id",
    "A-solver-R-rev",
)
GAP_ROW_CASE_IDS = frozenset((*GAP_SPECIALIZATION_CASE_IDS, *GAP_SOLVER_CASE_IDS))
GAP_CLOCK_WINDOW_FRACTION = 0.03
GAP_POWER_PERCENTILES = (0.10, 0.50, 0.90)
GAP_RESULT_WORDING = "realized throughput ratio under the observed operating points"
GAP_FORBIDDEN_WORDING = "clock-normalized efficiency"

# GPU split; machine concurrency follows the validated env lane.
MAX_CONCURRENT_PHYSICAL_GPUS = int(
    load(Path(__file__).resolve().parents[3])["PIPELINK_MAX_CONCURRENT_GPUS"]
)
GPU_MEMORY_ELIGIBILITY_MIB = 1024
FAILOVER_HEALTH_PROBE_COUNT = 2
FAILOVER_TRIGGER_CLASSES = (
    "device-absent-two-probes",
    "xid-in-job-interval",
    "cuda-status-reproduced",
)
FAILOVER_PROBE_INTERVAL_SECONDS = 1.0
FAILOVER_PROBE_INTERVAL_NS = 1_000_000_000
RUNTIME_FAILOVER_STATUSES = {
    "cudaErrorDevicesUnavailable": 46,
    "cudaErrorDeviceUninitialized": 201,
    "cudaErrorContextIsDestroyed": 709,
}
DRIVER_FAILOVER_STATUSES = {
    "CUDA_ERROR_DEVICE_UNAVAILABLE": 46,
    "CUDA_ERROR_NO_DEVICE": 100,
    "CUDA_ERROR_CONTEXT_IS_DESTROYED": 709,
}
ABSENT_PROBE_MARKERS = (
    "Unable to determine the device handle",
    "NVML_ERROR_GPU_IS_LOST",
)
# Expressly ineligible failover triggers.
INELIGIBLE_FAILOVER_TRIGGERS = (
    "slow-throughput",
    "changed-winner",
    "telemetry-invalid",
    "power-or-thermal-reason",
    "query-jitter-uuid-present",
    "pair-cap-exhaustion",
    "out-of-memory",
    "illegal-address",
    "assertion",
    "ordinary-launch-failure",
    "launch-timeout",
    "process-or-session-kill",
    "compilation-or-link-error",
    "user-cancellation",
)

# Work excluded from steady kernel timing.
EXCLUDED_STEADY_PHASES = (
    "compilation",
    "fa4-jit",
    "resolver-solve-validation",
    "stub-compile",
    "nvjitlink",
    "cubin-retrieval-write",
    "module-load",
    "allocation",
    "initialization",
    "one-time-prepack",
)
F3_UNFUSED_TIMED_LAUNCHES = 2

# Mandatory smoke and outer-G7 rehearsal.
SMOKE_JOB_BUDGET_THRESHOLD_HOURS = 1.0
SMOKE_VALID_PAIRS = 3
SMOKE_ATTEMPT_CAP = 9
REHEARSAL_VALID_PAIRS = 1
REHEARSAL_SLOT_SECONDS = 5.0
REHEARSAL_DISCARD_SECONDS = 1.0
REHEARSAL_STEADY_SECONDS = 4.0
REHEARSAL_ATTEMPT_CAP = 3
REHEARSAL_ROUTE_RANGES = ("G0-G6", "G8-G14", "rehearsal-closeout")

SCHEMA_VERSIONS = {
    "stream_map": "m3-stream-map-v1",
    "slot_request": "m3-parity-slot-request-v1",
    "slot_result": "m3-parity-slot-result-v1",
    "pair": "m3-parity-pair-v2",
    "summary": "m3-parity-summary-v2",
    "gap_pair": "m3-parity-pair-v6",
    "fast_pair": "m3-fast-pair-v1",
    "fast_summary": "m3-fast-summary-v1",
    "original_pair": "m3-original-duration-pair-v1",
    "original_summary": "m3-original-duration-summary-v1",
    "negative_pair": "m3-negative-launch-pair-v1",
    "negative_summary": "m3-negative-launch-summary-v1",
    "gap_summary": "m3-parity-summary-v6",
    "preflight": "m3-preflight-v1",
    "discovery": "m3-gpu-discovery-v2",
    "assignment": "m3-gpu-assignment-v2",
    "failover": "m3-failover-record-v2",
    "correctness": "m3-correctness-record-v2",
    "resource": "m3-resource-record-v2",
}
