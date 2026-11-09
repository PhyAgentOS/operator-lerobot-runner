"""Policy 节点配置加载与校验（对标 pick_and_place/config.py）。"""

from __future__ import annotations

import os
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from lerobot_inference.common.paths import resolve_user_path


ACTION_TRIGGERS = ("tick", "observation")


@dataclass(frozen=True)
class JointConfig:
    name: str
    mode: str = "position"
    unit: str = "radians"
    type: str = "actuator"


@dataclass
class PolicyNodeConfig:
    """Policy 节点完整配置：joints、policy、image_inputs（input_id -> alias）。"""

    joints: list[JointConfig]
    policy: dict[str, Any]
    auto_start: bool = False
    image_input_id_to_alias: dict[str, str] = field(default_factory=dict)
    state_joints: list[str] = field(default_factory=list)
    mode: str = "dora_plain"
    # tick：每个 Dora tick 出一个动作（真机按控制频率消费）。
    # observation：每收齐一帧新观测（proprio + 全部相机）才出一个动作，
    # 供 1:1 锁步的仿真 benchmark 使用，保证 chunk 内动作逐个被执行。
    action_trigger: str = "tick"

    @property
    def joint_order(self) -> list[str]:
        """动作关节顺序（与 JointCommand / task_robot 一致）。"""
        return [j.name for j in self.joints]

    @property
    def state_joint_order(self) -> list[str]:
        """观测状态顺序；未单独配置时沿用动作关节顺序。"""
        return self.state_joints or self.joint_order

    @property
    def image_input_ids(self) -> set[str]:
        return set(self.image_input_id_to_alias.keys())

    @property
    def alias_for_cameras(self) -> list[str]:
        """相机别名顺序；由 YAML image_inputs 定义顺序决定。"""
        return list(self.image_input_id_to_alias.values())

    def image_inputs_for(
        self,
        required_image_keys: Collection[str] | None,
    ) -> dict[str, str]:
        """Return only runtime image inputs consumed by the loaded policy."""
        if required_image_keys is None:
            return dict(self.image_input_id_to_alias)
        required = set(required_image_keys)
        return {
            input_id: alias
            for input_id, alias in self.image_input_id_to_alias.items()
            if f"observation.images.{alias}" in required
        }

    def runtime_policy_config(self) -> dict[str, Any]:
        """生成策略运行时配置，补齐可由节点配置推断的字段。"""
        state_dim = len(self.state_joint_order)
        action_dim = len(self.joints)
        policy_config = {
            **self.policy,
            "auto_start": self.auto_start,
            "camera_names": self.alias_for_cameras,
            # joint_count 保留给尚未迁移的策略；ACT 使用分离后的两个维度。
            "joint_count": action_dim,
            "state_joint_count": state_dim,
            "action_joint_count": action_dim,
            "state_joint_names": list(self.state_joint_order),
            "action_joint_names": list(self.joint_order),
        }
        ptype = str(policy_config.get("type", "")).strip()
        if ptype in {"ACT", "act"}:
            policy_config["state_dim"] = _validated_dimension(
                policy_config.get("state_dim", state_dim),
                expected=state_dim,
                name="policy.state_dim",
                source="state_joints",
            )
            policy_config["action_dim"] = _validated_dimension(
                policy_config.get("action_dim", action_dim),
                expected=action_dim,
                name="policy.action_dim",
                source="joints",
            )
        return policy_config

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicyNodeConfig:
        joints_raw = data.get("joints")
        if not joints_raw:
            raise ValueError("joints 或 joint_order 不能为空")

        joints = _parse_action_joints(joints_raw)
        state_joints_raw = data.get("state_joints")
        state_joints = (
            _parse_state_joints(state_joints_raw)
            if state_joints_raw is not None
            else [joint.name for joint in joints]
        )

        policy = data.get("policy", {})
        if not policy or not policy.get("type"):
            raise ValueError("policy 必须包含 type（ACT / pi05 / smolvla 等）")

        image_inputs = data.get("image_inputs")
        if not isinstance(image_inputs, dict) or not image_inputs:
            raise ValueError("image_inputs 不能为空，且必须为 dict[str, str]")

        image_input_id_to_alias = {
            str(input_id).strip(): str(alias).strip() for input_id, alias in image_inputs.items()
        }
        if any(
            not input_id or not alias
            for input_id, alias in image_input_id_to_alias.items()
        ):
            raise ValueError("image_inputs 的 input ID 和 alias 均不能为空")
        aliases = list(image_input_id_to_alias.values())
        if len(set(aliases)) != len(aliases):
            raise ValueError("image_inputs 的 alias 必须唯一，不能让多路输入覆盖同一相机")

        mode = str(data.get("mode", "dora_plain")).strip()
        if mode not in {"dora_plain", "session_endpoint"}:
            raise ValueError("mode 必须为 dora_plain 或 session_endpoint")
        action_trigger = str(data.get("action_trigger", "tick")).strip()
        if action_trigger not in ACTION_TRIGGERS:
            raise ValueError(f"action_trigger 必须为 {ACTION_TRIGGERS} 之一，收到 {action_trigger!r}")
        if action_trigger != "tick" and mode != "session_endpoint":
            raise ValueError("action_trigger: observation 仅支持 mode: session_endpoint")

        return cls(
            joints=joints,
            policy=policy,
            state_joints=state_joints,
            auto_start=_as_bool(policy.get("auto_start", data.get("auto_start", False))),
            image_input_id_to_alias=image_input_id_to_alias,
            mode=mode,
            action_trigger=action_trigger,
        )

    @classmethod
    def from_yaml_path(cls, path: str | Path) -> PolicyNodeConfig:
        p = resolve_user_path(path)
        if not p.is_file():
            raise FileNotFoundError(f"配置文件不存在: {p}")
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        if not data:
            raise ValueError(f"配置文件为空: {p}")
        # LeRobot 产物路径：相对本 yaml 解析
        base = p.parent
        policy = data.get("policy")
        if isinstance(policy, dict):
            for key in ("run_dir", "pretrained_path", "tokenizer_path", "ckpt_path", "wan_pretrained_path"):
                value = policy.get(key)
                if not value:
                    continue
                path_value = Path(os.path.expandvars(str(value))).expanduser()
                if not path_value.is_absolute():
                    policy[key] = str((base / path_value).resolve())
                else:
                    policy[key] = str(path_value)
        return cls.from_dict(data)


# 兼容旧导入名
InferenceNodeConfig = PolicyNodeConfig


def _parse_action_joints(items: Any) -> list[JointConfig]:
    if not isinstance(items, list) or not items:
        raise ValueError("joints 必须为非空列表")
    joints: list[JointConfig] = []
    for item in items:
        if isinstance(item, str):
            joints.append(JointConfig(name=item))
        elif isinstance(item, dict) and "name" in item:
            joints.append(
                JointConfig(
                    name=str(item["name"]),
                    mode=str(item.get("mode", "position")),
                    unit=str(item.get("unit", "radians")),
                    type=str(item.get("type", item.get("joint_type", "actuator"))),
                )
            )
        else:
            raise ValueError(f"joint 项须为字符串或包含 name 的字典: {item!r}")
    return joints


def _parse_state_joints(items: Any) -> list[str]:
    if not isinstance(items, list) or not items:
        raise ValueError("state_joints 必须为非空列表")
    names: list[str] = []
    for item in items:
        if isinstance(item, str):
            name = item
        elif isinstance(item, dict) and "name" in item:
            name = str(item["name"])
        else:
            raise ValueError(f"state_joints 项须为字符串或包含 name 的字典: {item!r}")
        if not name:
            raise ValueError("state_joints 中的关节名不能为空")
        names.append(name)
    return names


def _validated_dimension(value: Any, *, expected: int, name: str, source: str) -> int:
    try:
        dimension = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须为整数: {value!r}") from exc
    if dimension != expected:
        raise ValueError(f"{name}={dimension} 与 {source} 数量={expected} 不一致")
    return dimension


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def load_config(
    config_path: str | Path | None = None,
    dataflow_config: dict[str, Any] | None = None,
) -> PolicyNodeConfig:
    """
    加载配置（优先级同 pick_and_place）：
    1. dataflow 内联 config / config_path
    2. config_path 参数
    3. 环境变量 LEROOT_INFERENCE_CONFIG
    """
    data: dict[str, Any] | None = None

    if dataflow_config:
        if "config" in dataflow_config:
            data = dataflow_config["config"]
        elif "config_path" in dataflow_config:
            return PolicyNodeConfig.from_yaml_path(dataflow_config["config_path"])

    path = config_path
    if path is None and data is None:
        path = os.environ.get("LEROOT_INFERENCE_CONFIG")

    if path:
        return PolicyNodeConfig.from_yaml_path(path)

    if data:
        return PolicyNodeConfig.from_dict(data)

    raise ValueError(
        "未找到配置。请设置 LEROOT_INFERENCE_CONFIG、"
        "或通过 --config 指定配置文件、或使用 dataflow config。"
    )
