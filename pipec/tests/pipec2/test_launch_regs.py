from __future__ import annotations

import pytest

from pipec2.model import SchemaError
from pipec2.profile import get_profile
from pipec2.schema import load_pipeline
from pipec2.solve import _candidate_values


def _pipeline(handle_name: str, owner: str, values: list[int]) -> dict:
    return {
        "pipelink_abi": {
            "schema": "2.0-draft",
            "target": {
                "profile": "sm_103a",
                "lto_arch": "lto_103",
                "toolkit": "cuda-13.3.0",
            },
        },
        "pipeline": {
            "name": "launch-cap-validation",
            "stages": [
                {"artifact_path": "stage.plink", "stage_id": "example.stage"}
            ],
            "handles": {
                handle_name: {
                    "class": "regs",
                    "binding": "bind",
                    "domain": "regs",
                    "values": values,
                    "owner": owner,
                }
            },
            "phases": ["steady"],
            "exclusive": [],
            "domains": {},
            "depth_priority": [],
            "regs_priority": [handle_name],
            "forced_bindings": {},
            "topology": {"persistent": True, "scheduler": "static"},
        },
    }


def test_launch_cap_accepts_255_with_unit_granularity() -> None:
    pipeline = load_pipeline(
        _pipeline("pipelink_launch_regs", "reg_group:launch", [95, 160, 255])
    )

    names, values = _candidate_values(pipeline, [], get_profile("sm_103a"))

    assert names == ["pipelink_launch_regs"]
    assert values == {"pipelink_launch_regs": (95, 160, 255)}
    assert get_profile("sm_103a").caps.launch_regs_max == 255


def test_launch_cap_rejects_256_even_though_setmaxnreg_allows_it() -> None:
    with pytest.raises(SchemaError, match="launch register cap is outside"):
        load_pipeline(
            _pipeline("pipelink_launch_regs", "reg_group:launch", [256])
        )


def test_non_launch_regs_handle_still_rejects_255() -> None:
    with pytest.raises(SchemaError, match="register value is outside"):
        load_pipeline(_pipeline("pipelink_regs_worker", "reg_group:worker", [255]))


def test_launch_name_without_launch_owner_uses_setmaxnreg_rules() -> None:
    with pytest.raises(SchemaError, match="register value is outside"):
        load_pipeline(
            _pipeline("pipelink_launch_regs", "reg_group:not-launch", [255])
        )
