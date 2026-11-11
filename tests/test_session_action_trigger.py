"""action_trigger: tick vs observation（1:1 锁步 benchmark）。"""

from __future__ import annotations

import numpy as np
import pytest
from forge_msgs import Image, JointState
from lerobot_inference.inference.config import PolicyNodeConfig
from lerobot_inference.inference.session_endpoint import LeRobotServeSessionEndpoint
from lerobot_inference.inference.session_runner import LeRobotSessionPolicyRunner

JOINTS = ["j1", "j2"]
IMAGES = {"image/head": "cam_high", "image/left_wrist": "cam_left_wrist"}


class _Node:
    def __init__(self) -> None:
        self.outputs: list[str] = []

    def send_output(self, output_id, data) -> None:
        self.outputs.append(output_id)


class _Payload:
    def to_arrow(self):
        return None


class _Policy:
    def __init__(self) -> None:
        self.calls = 0

    def is_observation_needed(self) -> bool:
        return True

    def generate_action(self, observation, camera_aliases):
        assert observation["observation.state"].shape == (2,)
        self.calls += 1
        return np.zeros(2, dtype=np.float32)


def _runner(trigger: str) -> tuple[LeRobotSessionPolicyRunner, _Node, _Policy]:
    node, policy = _Node(), _Policy()
    runner = LeRobotSessionPolicyRunner(
        node,
        policy=policy,
        endpoint=LeRobotServeSessionEndpoint(policy_id="p"),
        joint_order=JOINTS,
        image_input_id_to_alias=IMAGES,
        build_action=lambda _action: _Payload(),
        policy_id="p",
        auto_start=True,
        action_trigger=trigger,
    )
    return runner, node, policy


def _inp(input_id: str, value=None) -> dict:
    return {"type": "INPUT", "id": input_id, "value": value}


def _proprio():
    return _inp("proprio_state", JointState.from_np(np.zeros(2), JOINTS, "position").to_arrow())


def _image(input_id: str):
    return _inp(input_id, Image.from_numpy(np.zeros((4, 4, 3), np.uint8), "rgb8").to_arrow())


def _actions(node: _Node) -> int:
    return node.outputs.count("action")


def _observe(runner) -> None:
    runner.process_event(_proprio())
    for input_id in IMAGES:
        runner.process_event(_image(input_id))


def test_observation_trigger_emits_one_action_per_fresh_observation() -> None:
    runner, node, policy = _runner("observation")
    runner.process_event(_inp("tick"))
    assert _actions(node) == 0

    runner.process_event(_proprio())  # benchmark 先发 proprio，图像未到
    runner.process_event(_inp("tick"))
    assert _actions(node) == 0

    for input_id in IMAGES:
        runner.process_event(_image(input_id))
    for _ in range(5):  # 同一帧观测只换一个动作，多余 tick 不出队
        runner.process_event(_inp("tick"))
    assert _actions(node) == 1 and policy.calls == 1

    _observe(runner)
    runner.process_event(_inp("tick"))
    assert _actions(node) == 2


def test_observation_trigger_requires_fresh_images_after_reset_scene() -> None:
    from forge_msgs import PolicyCommand

    runner, node, _ = _runner("observation")
    _observe(runner)
    runner.process_event(_inp("policy_command", PolicyCommand.from_inputs(
        policy_id="p", command="reset_scene", inputs={}).to_arrow()))
    runner.process_event(_proprio())  # 旧图像不能拼进新 episode 的首帧
    runner.process_event(_inp("tick"))
    assert _actions(node) == 0


def test_tick_trigger_keeps_dequeuing_every_tick() -> None:
    runner, node, policy = _runner("tick")
    _observe(runner)
    for _ in range(3):
        runner.process_event(_inp("tick"))
    assert _actions(node) == 3 and policy.calls == 3


def _config(**extra) -> dict:
    return {"joints": JOINTS, "policy": {"type": "g05"}, "image_inputs": IMAGES, **extra}


def test_config_action_trigger_defaults_and_validation() -> None:
    assert PolicyNodeConfig.from_dict(_config()).action_trigger == "tick"
    cfg = PolicyNodeConfig.from_dict(_config(mode="session_endpoint", action_trigger="observation"))
    assert cfg.action_trigger == "observation"
    with pytest.raises(ValueError, match="action_trigger"):
        PolicyNodeConfig.from_dict(_config(mode="session_endpoint", action_trigger="sometimes"))
    with pytest.raises(ValueError, match="session_endpoint"):
        PolicyNodeConfig.from_dict(_config(action_trigger="observation"))
