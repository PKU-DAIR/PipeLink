from __future__ import annotations

import pytest

from pipec2.canonjson import canonical_dumps
from pipec2.profile import PROFILES, Arch, Domain, Profile, ProfileError, get_profile


def test_sm103_profile_domain_capacities() -> None:
    profile = get_profile("sm_103a")
    assert (profile.lto_arch, profile.toolkit) == ("lto_103", "cuda-13.3.0")
    assert profile.domains == {
        "smem": Domain("bytes", 232448, 0, 1, "bind", "smem", 1),
        "mbarrier": Domain("16-byte slots", 232448, 0, 1, "bind", "smem", 16),
        "tmem": Domain("columns", 512, 0, 32, "bind", "tmem", 1),
        "named_barrier": Domain("IDs", 16, 1, 1, "bind", "named_barrier", 1),
        "warp": Domain("launch warps", 32, 0, 1, "bind", "warp", 1),
        "regs": Domain("per-thread registers", 256, 24, 8, "bind", "regs", 1),
        "gmem_counter": Domain("4-byte slots", "pipeline", 0, 1, "bind", "gmem_counter", 1),
        "instr_page": Domain("pages", "pipeline", 0, 1, "bind", "instr_page", 1),
        "instr_semaphore": Domain(
            "8-byte semaphores", "pipeline", 0, 1, "bind", "instr_semaphore", 1
        ),
        "instr_scratch": Domain("bytes", "pipeline", 0, 1, "bind", "instr_scratch", 1),
        "opcode": Domain("opcodes", "pipeline", 0, 1, "bind", "opcode", 1),
        "smem_page": Domain("pages", "pipeline", 0, 1, "bind", "smem", None),
    }
    assert profile.caps.regs_min == 24
    assert profile.caps.regs_max == 256
    assert profile.caps.regs_step == 8
    assert profile.caps.warpgroup_warps == 4
    assert profile.caps.smem_capacity_dynamic == 232448
    assert profile.tmem_lanes == 128
    assert profile.caps.cta_group_legal == {1, 2}
    assert profile.adapter_compile.gencode == ("compute_103a", "lto_103a")
    assert profile.arch == Arch("sm_103a", "compute_103a", "lto_103a", "lto_103")
    assert profile.schedulers == {"clc", "static", "stream_k"}
    assert profile.scheduler_strategies == {"clc-sm100-v1"}
    assert profile.catalogue == "tcgen05-readonly-v1"


def test_sm90_profile_is_sm103_without_tmem_clc_or_cta_pairs() -> None:
    sm90 = get_profile("sm_90a")
    sm103 = get_profile("sm_103a")
    assert (sm90.lto_arch, sm90.toolkit) == ("lto_90", "cuda-13.3.0")
    assert sm90.domains == {
        name: domain for name, domain in sm103.domains.items() if name != "tmem"
    }
    assert "tmem" not in sm90.domains
    assert sm90.tmem_lanes is None
    assert sm90.caps.cta_group_legal == {1}
    assert sm90.caps.max_cluster == (8, 8, 8)
    assert sm90.caps.smem_capacity_dynamic == 232448
    assert (sm90.caps.regs_min, sm90.caps.regs_max, sm90.caps.regs_step) == (24, 256, 8)
    assert sm90.adapter_compile.gencode == ("compute_90a", "lto_90a")
    assert sm90.adapter_compile.toolkit == "cuda-13.3.0"
    assert sm90.arch == Arch("sm_90a", "compute_90a", "lto_90a", "lto_90")
    assert sm90.atoms == {
        "sw128_k_bf16": sm103.atoms["sw128_k_bf16"],
        "sw64_mn_bf16": sm103.atoms["sw64_mn_bf16"],
    }
    assert sm90.schedulers == {"static", "stream_k"}
    assert sm90.scheduler_strategies == frozenset()
    assert sm90.catalogue == "warp-readonly-v1"


def test_sm120_profile_is_sm90_with_consumer_blackwell_smem() -> None:
    sm120 = get_profile("sm_120a")
    sm90 = get_profile("sm_90a")
    assert (sm120.lto_arch, sm120.toolkit) == ("lto_120", "cuda-13.3.0")
    assert sm120.domains == {
        **sm90.domains,
        "smem": Domain("bytes", 101376, 0, 1, "bind", "smem", 1),
        "mbarrier": Domain("16-byte slots", 101376, 0, 1, "bind", "smem", 16),
    }
    assert sm120.tmem_lanes is None
    assert sm120.caps.smem_capacity_dynamic == 101376
    assert sm120.caps.cta_group_legal == {1}
    assert sm120.caps.max_cluster == (8, 8, 8)
    assert sm120.arch == Arch("sm_120a", "compute_120a", "lto_120a", "lto_120")
    assert sm120.atoms == sm90.atoms
    assert sm120.schedulers == {"static", "stream_k"}
    assert sm120.scheduler_strategies == frozenset()
    assert sm120.catalogue == "warp-readonly-v1"


@pytest.mark.parametrize("name", sorted(PROFILES))
def test_profile_round_trip(name: str) -> None:
    profile = get_profile(name)
    restored = Profile.from_json_obj(profile.to_json_obj())
    assert restored == profile
    assert canonical_dumps(restored.to_json_obj()) == canonical_dumps(profile.to_json_obj())


def test_profiles_serialize_distinctly() -> None:
    encoded = {canonical_dumps(profile.to_json_obj()) for profile in PROFILES.values()}
    assert len(encoded) == len(PROFILES)


def test_unknown_profile_is_structured_failure() -> None:
    with pytest.raises(ProfileError) as caught:
        get_profile("missing")
    assert [(item.clause_id, item.path) for item in caught.value.failures] == [
        ("L-TOPO", "target.profile")
    ]
