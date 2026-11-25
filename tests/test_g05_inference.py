from __future__ import annotations

from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lerobot_inference.inference.config import PolicyNodeConfig
from lerobot_inference.inference.policies import registry
from lerobot_inference.inference.policies.g05 import G05Adapter
from lerobot_inference.inference.policies.g05 import adapter as g05_adapter

_CONFIG = Path(__file__).resolve().parents[1] / "config" / "inference" / "g05.yaml"
_CAMERAS = ("head_camera", "left_camera", "right_camera")
_IMAGE_KEYS = {f"observation.images.{name}" for name in _CAMERAS}


class FakeProcessor:
    def __init__(self) -> None:
        self.calls = 0
        self.reset_count = 0

    def __call__(self, value):
        self.calls += 1
        return value

    def reset(self) -> None:
        self.reset_count += 1


class FakeG05Policy:
    def __init__(self, *, n_action_steps: int = 3, action_dim: int = 14, cot: bool = False) -> None:
        self.config = SimpleNamespace(
            type="g05",
            device="cpu",
            n_action_steps=n_action_steps,
            input_features={"observation.state": SimpleNamespace(shape=(action_dim,))},
            output_features={"action": SimpleNamespace(shape=(action_dim,))},
            image_features={key: SimpleNamespace(shape=(3, 256, 256)) for key in _IMAGE_KEYS},
        )
        self.action_dim = action_dim
        self.cot = cot
        self.predictions = 0
        self.reset_count = 0
        self._queue: deque[torch.Tensor] = deque()

    def parameters(self):
        return iter(())

    def reset(self) -> None:
        self.reset_count += 1
        self._queue.clear()

    def select_action(self, batch):
        if not self._queue:
            assert batch, "an empty queue must be refilled from an observation"
            self.predictions += 1
            chunk = torch.arange(self.config.n_action_steps, dtype=torch.float32)[:, None]
            self._queue.extend(chunk.expand(-1, self.action_dim) + 100 * self.predictions)
        action = self._queue.popleft().unsqueeze(0)
        return {"action": action, "prediction": "subtask"} if self.cot else action


def _observation() -> dict[str, np.ndarray]:
    obs = {"observation.state": np.zeros(14, dtype=np.float32)}
    for key in _IMAGE_KEYS:
        obs[key] = np.zeros((8, 8, 3), dtype=np.uint8)
    return obs


def _adapter(**policy_kwargs):
    policy = FakeG05Policy(**policy_kwargs)
    pre, post = FakeProcessor(), FakeProcessor()
    adapter = G05Adapter(policy, pre, post, instruction="lift", expected_image_keys=set(_IMAGE_KEYS))
    return adapter, policy, pre, post


@pytest.fixture(autouse=True)
def _cpu_prepare(monkeypatch):
    def fake_prepare(observation, device, task=None, robot_type=None):
        batch = {k: torch.as_tensor(v)[None] for k, v in observation.items()}
        batch["task"] = [task]
        return batch

    monkeypatch.setattr(g05_adapter, "prepare_observation_for_inference", fake_prepare)


def test_g05_serves_full_chunk_and_postprocesses_every_step() -> None:
    adapter, policy, pre, post = _adapter(n_action_steps=3)

    assert adapter.is_observation_needed()
    actions = [adapter.generate_action(_observation())]
    while not adapter.is_observation_needed():
        actions.append(adapter.generate_action({}))

    assert len(actions) == 3
    assert [a[0] for a in actions] == [100.0, 101.0, 102.0]
    assert all(a.shape == (14,) for a in actions)
    assert policy.predictions == 1
    assert pre.calls == 1
    assert post.calls == 3

    adapter.generate_action(_observation())
    assert policy.predictions == 2


def test_g05_rejects_observation_misuse() -> None:
    adapter, *_ = _adapter(n_action_steps=2)
    with pytest.raises(ValueError, match="depleted"):
        adapter.generate_action({})
    adapter.generate_action(_observation())
    with pytest.raises(ValueError, match="queued actions"):
        adapter.generate_action(_observation())


def test_g05_reset_and_instruction_change_reset_policy_and_processors() -> None:
    adapter, policy, pre, post = _adapter(n_action_steps=3)
    adapter.generate_action(_observation())

    adapter.instruction = "lift"
    assert policy.reset_count == 0

    adapter.instruction = "stack blocks"
    assert (policy.reset_count, pre.reset_count, post.reset_count) == (1, 1, 1)
    assert adapter.is_observation_needed()


def test_g05_unwraps_policy_output_with_prediction() -> None:
    adapter, *_ = _adapter(n_action_steps=2, cot=True)
    action = adapter.generate_action(_observation())
    assert action.shape == (14,)


def test_g05_validates_observation_inputs() -> None:
    adapter, *_ = _adapter()
    obs = _observation()
    del obs["observation.images.left_camera"]
    with pytest.raises(KeyError, match="left_camera"):
        adapter.generate_action(obs)

    obs = _observation()
    obs["observation.state"][0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        adapter.generate_action(obs)


def test_g05_validates_runtime_dimensions() -> None:
    adapter, *_ = _adapter()
    adapter.validate_io_dimensions(14, 14)
    with pytest.raises(ValueError, match="do not match"):
        adapter.validate_io_dimensions(16, 16)


def test_g05_registry_rejects_n_action_steps_override() -> None:
    with pytest.raises(ValueError, match="n_action_steps"):
        registry._create_g05({"type": "g05", "n_action_steps": 4}, "/unused")


def test_g05_registry_passes_runtime_overrides_and_camera_keys(monkeypatch) -> None:
    captured = {}

    def fake_from_pretrained(path, **kwargs):
        captured.update(kwargs, path=path)
        adapter, *_ = _adapter()
        return adapter

    monkeypatch.setattr(G05Adapter, "from_pretrained", staticmethod(fake_from_pretrained))
    registry._create_g05(
        {
            "type": "g05",
            "device": "cpu",
            "instruction": "lift",
            "runtime_system": "system1",
            "camera_names": list(_CAMERAS),
            "joint_count": 14,
        },
        "/ckpt",
    )

    assert captured["path"] == "/ckpt"
    assert captured["expected_image_keys"] == _IMAGE_KEYS
    assert captured["policy_config_overrides"] == {"runtime_system": "system1"}


def test_g05_config_example_matches_robotwin_contract() -> None:
    config = PolicyNodeConfig.from_yaml_path(_CONFIG)

    assert registry.normalize_policy_type(str(config.policy["type"])) == "g05"
    assert "n_action_steps" not in config.policy
    assert len(config.joints) == 14
    assert set(config.image_input_id_to_alias.values()) == set(_CAMERAS)
