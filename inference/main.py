#!/usr/bin/env python3
"""Policy Dora 节点：订阅 proprio_state + 多路 image，输出 JointCommand。

对标 pick_and_place/main.py；策略后端加载 lerobot_trainer / convert 产物。
"""

from __future__ import annotations

import logging
import sys

import numpy as np
from forge_msgs import JointCommand
from forge_policy import run_dora_policy_node
from lerobot_inference.inference.config import load_config
from lerobot_inference.inference.policies.registry import create_policy_adapter


def _setup_policy(policy_config: dict):
    """与 pick_and_place._setup_policy 一致：仅依赖 policy 配置字典。"""
    return create_policy_adapter(policy_config)


def _build_joint_command(action_np, config) -> JointCommand:
    names = config.joint_order
    action = np.asarray(action_np)
    if action.ndim != 1:
        raise ValueError(f"policy action must be 1-D, got shape={action.shape}")
    if len(action) != len(names):
        raise ValueError(
            f"policy action length={len(action)} does not match action joints={len(names)}"
        )
    if not np.isfinite(action).all():
        raise ValueError("policy action must contain only finite values")

    position = [0.0] * len(names)
    velocity = [0.0] * len(names)
    effort = [0.0] * len(names)

    for i, joint in enumerate(config.joints):
        value = float(action[i])
        mode = joint.mode
        if mode == "velocity":
            velocity[i] = value
        elif mode == "effort":
            effort[i] = value
        else:
            position[i] = value

    return JointCommand(name=names, position=position, velocity=velocity, effort=effort)


def _run_session_endpoint(
    config, policy, policy_config, image_input_id_to_alias, alias_for_cameras
) -> int:
    from dora import Node
    from lerobot_inference.inference.session_endpoint import LeRobotServeSessionEndpoint
    from lerobot_inference.inference.session_runner import LeRobotSessionPolicyRunner

    node = Node()
    endpoint = LeRobotServeSessionEndpoint(
        policy_id=str(policy_config.get("policy_id", "default")),
    )
    runner = LeRobotSessionPolicyRunner(
        node,
        policy=policy,
        endpoint=endpoint,
        joint_order=config.state_joint_order,
        image_input_id_to_alias=image_input_id_to_alias,
        build_action=lambda action_np: _build_joint_command(action_np, config),
        policy_id=str(policy_config.get("policy_id", "default")),
        alias_for_cameras=alias_for_cameras,
        auto_start=bool(policy_config.get("auto_start", False)),
        call_lifecycle_hooks=True,
        action_trigger=config.action_trigger,
    )
    return runner.run(node)


def run_infer(args) -> int:
    """执行 Dora 在线推理；args 需含可选 config。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(config_path=getattr(args, "config", None))
    policy_config = config.runtime_policy_config()
    policy = _setup_policy(policy_config)
    image_inputs = config.image_inputs_for(policy.required_image_keys)
    camera_aliases = list(image_inputs.values())
    try:
        if config.mode == "session_endpoint":
            return _run_session_endpoint(
                config, policy, policy_config, image_inputs, camera_aliases
            )
        return run_dora_policy_node(
            policy,
            joint_order=config.state_joint_order,
            image_input_id_to_alias=image_inputs,
            alias_for_cameras=camera_aliases,
            build_action=lambda action_np: _build_joint_command(action_np, config),
            policy_id=str(policy_config.get("policy_id", "default")),
            auto_start=bool(policy_config.get("auto_start", False)),
            call_lifecycle_hooks=True,
        )
    finally:
        stop = getattr(policy, "stop", None)
        if callable(stop):
            stop()


def main() -> int:
    """兼容旧入口：转发到统一 CLI 的 infer 子命令。"""
    from lerobot_inference.cli import main_infer

    return main_infer()


if __name__ == "__main__":
    sys.exit(main())
