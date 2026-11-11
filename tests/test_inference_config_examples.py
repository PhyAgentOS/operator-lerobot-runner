from pathlib import Path

import pytest
from lerobot_inference.inference.config import PolicyNodeConfig
from lerobot_inference.inference.policies.registry import (
    normalize_policy_type,
    supported_policy_types,
)

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "inference"


@pytest.mark.parametrize(
    ("filename", "expected_type", "expected_mode"),
    [
        ("act.yaml", "act", None),
        ("pi05.yaml", "pi05", "sync"),
        ("pi05_async_rtc.yaml", "pi05", "async_rtc"),
        ("g05.yaml", "g05", None),
    ],
)
def test_inference_config_example_is_loadable(
    filename: str,
    expected_type: str,
    expected_mode: str | None,
) -> None:
    config = PolicyNodeConfig.from_yaml_path(_CONFIG_DIR / filename)

    assert normalize_policy_type(str(config.policy["type"])) == expected_type
    assert config.policy.get("inference_mode") == expected_mode
    assert config.joints
    assert config.image_input_id_to_alias


def test_every_implemented_policy_has_a_config_example() -> None:
    example_types = {
        normalize_policy_type(
            str(PolicyNodeConfig.from_yaml_path(path).policy["type"])
        )
        for path in _CONFIG_DIR.glob("*.yaml")
    }

    assert set(supported_policy_types()) <= example_types
