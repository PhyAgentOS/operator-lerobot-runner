from pathlib import Path

import pytest
from lerobot_inference.inference.config import PolicyNodeConfig

_HEADER = """\
image_inputs:
  image/top: top
joints:
  - joint1
policy:
"""


def _write(tmp_path: Path, policy_lines: str) -> Path:
    path = tmp_path / "policy.yaml"
    path.write_text(_HEADER + policy_lines, encoding="utf-8")
    return path


def test_env_vars_expand_in_model_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("VLM_DIR", "/models/SmolVLM2")
    monkeypatch.setenv("QWEN_DIR", "/models/Qwen3-VL")
    monkeypatch.setenv("WAN_DIR", "/models/Wan2.2")
    path = _write(
        tmp_path,
        "  type: smolvla\n"
        "  vlm_model_name: ${VLM_DIR}\n"
        "  qwen_model_name: ${QWEN_DIR}\n"
        "  wan_diffusers_path: ${WAN_DIR}\n",
    )

    policy = PolicyNodeConfig.from_yaml_path(path).policy

    assert policy["vlm_model_name"] == "/models/SmolVLM2"
    assert policy["qwen_model_name"] == "/models/Qwen3-VL"
    assert policy["wan_diffusers_path"] == "/models/Wan2.2"


def test_model_name_keeps_hub_repo_id(tmp_path):
    path = _write(
        tmp_path,
        "  type: smolvla\n"
        "  vlm_model_name: HuggingFaceTB/SmolVLM2-500M-Video-Instruct\n",
    )

    policy = PolicyNodeConfig.from_yaml_path(path).policy

    assert policy["vlm_model_name"] == "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"


def test_relative_wan_diffusers_path_resolves_from_yaml(tmp_path):
    path = _write(tmp_path, "  type: fastwam\n  wan_diffusers_path: wan\n")

    policy = PolicyNodeConfig.from_yaml_path(path).policy

    assert policy["wan_diffusers_path"] == str((tmp_path / "wan").resolve())


@pytest.mark.parametrize(
    "key", ["pretrained_path", "vlm_model_name", "wan_diffusers_path"]
)
def test_unset_env_var_fails_at_load(tmp_path, monkeypatch, key):
    monkeypatch.delenv("PAOS_TEST_UNSET_DIR", raising=False)
    path = _write(tmp_path, f"  type: smolvla\n  {key}: ${{PAOS_TEST_UNSET_DIR}}\n")

    with pytest.raises(ValueError, match="PAOS_TEST_UNSET_DIR"):
        PolicyNodeConfig.from_yaml_path(path)
