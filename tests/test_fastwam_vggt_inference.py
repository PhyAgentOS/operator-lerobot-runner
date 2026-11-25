"""Unit tests for the FastWAM VGGT-Omega adapter (no torch / no GPU needed)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lerobot_inference.inference.policies.fastwam_vggt import bundle as bundle_mod
from lerobot_inference.inference.policies.fastwam_vggt.adapter import FastWAMVGGTPolicyAdapter
from lerobot_inference.inference.policies.fastwam_vggt.bundle import (
    FastWAMBundledConfigError,
    load_bundle,
)


class FakeRunner:
    """Duck-typed stand-in for FastWAMVGGTModelRunner."""

    def __init__(self, horizon: int = 4, action_dim: int = 14) -> None:
        self.horizon = horizon
        self.action_dim = action_dim
        self.calls: list[tuple[tuple[int, ...], tuple[float, ...], str]] = []

    def infer(self, images, state, task):
        self.calls.append((tuple(images.shape), tuple(float(v) for v in state), task))
        base = np.arange(self.horizon * self.action_dim, dtype=np.float32)
        return base.reshape(self.horizon, self.action_dim) / 100.0


def _write_bundle(root: Path, *, cameras=("cam_head", "cam_left_wrist", "cam_right_wrist")):
    (root / "text").mkdir(parents=True, exist_ok=True)
    (root / "weights").mkdir(parents=True, exist_ok=True)
    (root / "model_config.json").write_text(
        json.dumps({"_target_": "fastwam.runtime.create_action_vggt_lang_recon_fastwam"}),
        encoding="utf-8",
    )
    (root / "normalization.json").write_text(
        json.dumps(
            {
                "state": {"mean": [0.0] * 14, "std": [1.0] * 14},
                "action": {"mean": [0.0] * 14, "std": [1.0] * 14},
            }
        ),
        encoding="utf-8",
    )
    (root / "weights" / "step_000100.pt").write_bytes(b"stub")
    (root / "bundle.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "checkpoint": "weights/step_000100.pt",
                "model_config": "model_config.json",
                "normalization": "normalization.json",
                "text_dir": "text",
                "camera_keys": list(cameras),
                "image_size": [256, 192],
                "action_dim": 14,
                "proprio_dim": 14,
                "action_horizon": 4,
                "num_inference_steps": 3,
                "gripper_indices": [6, 13],
            }
        ),
        encoding="utf-8",
    )
    return root


def test_load_bundle_round_trip(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path)
    cfg = load_bundle(root)
    assert cfg.camera_keys == ("cam_head", "cam_left_wrist", "cam_right_wrist")
    assert (cfg.image_width, cfg.image_height) == (256, 192)
    assert cfg.action_dim == 14 and cfg.proprio_dim == 14
    assert cfg.action_horizon == 4
    assert cfg.checkpoint is not None and cfg.checkpoint.name == "step_000100.pt"
    assert cfg.text_dir is not None and cfg.text_dir.name == "text"


def test_load_bundle_rejects_missing_field(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path)
    manifest = json.loads((root / "bundle.json").read_text())
    manifest.pop("action_dim")
    (root / "bundle.json").write_text(json.dumps(manifest))
    with pytest.raises(FastWAMBundledConfigError, match="action_dim"):
        load_bundle(root)


def test_text_embedding_path_uses_prompt_hash(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path)
    cfg = load_bundle(root)
    path = cfg.text_embedding_path("Stack the bowls")
    import hashlib

    prompt = bundle_mod.DEFAULT_PROMPT.format(task="Stack the bowls")
    assert path.name == hashlib.sha256(prompt.encode()).hexdigest() + ".pt"


def test_adapter_chunks_and_resizes(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path)
    cfg = load_bundle(root)
    runner = FakeRunner(horizon=cfg.action_horizon, action_dim=cfg.action_dim)
    adapter = FastWAMVGGTPolicyAdapter(
        runner,
        cfg,
        instruction="Stack the bowls",
        expected_image_keys={f"observation.images.{c}" for c in cfg.camera_keys},
    )

    observation = {
        "observation.images.cam_head": np.zeros((480, 640, 3), dtype=np.uint8),
        "observation.images.cam_left_wrist": np.full((480, 640, 3), 7, dtype=np.uint8),
        "observation.images.cam_right_wrist": np.full((480, 640, 3), 9, dtype=np.uint8),
        "observation.state": np.zeros(14, dtype=np.float32),
    }
    aliases = ["cam_right_wrist", "cam_head", "cam_left_wrist"]  # deliberately shuffled

    actions = [adapter.generate_action(observation, aliases) for _ in range(cfg.action_horizon)]
    assert all(action is not None for action in actions)
    assert np.allclose(actions[0], np.arange(cfg.action_dim, dtype=np.float32) / 100.0)
    assert adapter.is_observation_needed()
    assert len(runner.calls) == 1
    shape, state, task = runner.calls[0]
    assert shape == (3, 192, 256, 3)
    assert state == tuple([0.0] * 14)
    assert task == "Stack the bowls"


def test_adapter_requires_observation_per_chunk(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path)
    cfg = load_bundle(root)
    runner = FakeRunner(horizon=2, action_dim=14)
    adapter = FastWAMVGGTPolicyAdapter(runner, cfg, instruction="x")
    observation = {f"observation.images.{c}": np.zeros((8, 8, 3), np.uint8) for c in cfg.camera_keys}
    observation["observation.state"] = np.zeros(14, np.float32)
    adapter.generate_action(observation, list(cfg.camera_keys))
    assert not adapter.is_observation_needed()
    adapter.generate_action({}, list(cfg.camera_keys))
    assert adapter.is_observation_needed()


def test_adapter_rejects_unknown_camera(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path)
    cfg = load_bundle(root)
    adapter = FastWAMVGGTPolicyAdapter(FakeRunner(), cfg, instruction="x")
    observation = {
        "observation.images.cam_head": np.zeros((8, 8, 3), np.uint8),
        "observation.images.cam_left_wrist": np.zeros((8, 8, 3), np.uint8),
        "observation.images.extra": np.zeros((8, 8, 3), np.uint8),
        "observation.state": np.zeros(14, np.float32),
    }
    with pytest.raises(ValueError, match="camera aliases"):
        adapter.generate_action(
            observation, ["cam_head", "cam_left_wrist", "extra"]
        )


def test_load_bundle_from_contract_run_dir(tmp_path: Path) -> None:
    """A plain FastWAM run directory works without a pre-built bundle.json."""
    (tmp_path / "data" / "text").mkdir(parents=True)
    (tmp_path / "checkpoints" / "weights").mkdir(parents=True)
    (tmp_path / "checkpoints" / "weights" / "step_000500.pt").write_bytes(b"stub")
    (tmp_path / "data" / "text" / "dummy.pt").write_bytes(b"stub")
    (tmp_path / "contract.json").write_text(
        json.dumps(
            {
                "model_config": {
                    "_target_": "fastwam.runtime.create_action_vggt_lang_recon_fastwam",
                    "vggt_camera_keys": ["cam_head", "cam_left_wrist", "cam_right_wrist"],
                    "proprio_dim": 14,
                    "action_dit_config": {"action_dim": 14},
                },
                "objective": {"target_contract": {"horizon": 32}},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "normalization.json").write_text(
        json.dumps(
            {
                "state": {"mean": [0.0] * 14, "std": [1.0] * 14},
                "action": {"mean": [0.0] * 14, "std": [1.0] * 14},
            }
        ),
        encoding="utf-8",
    )

    cfg = load_bundle(tmp_path, image_size=[256, 192], gripper_indices=[6, 13])
    assert cfg.camera_keys == ("cam_head", "cam_left_wrist", "cam_right_wrist")
    assert (cfg.image_width, cfg.image_height) == (256, 192)
    assert cfg.checkpoint is not None and cfg.checkpoint.name == "step_000500.pt"
    assert cfg.text_dir is not None and cfg.text_dir.name == "text"
    assert cfg.gripper_indices == (6, 13)
    assert cfg.action_horizon == 32


def test_run_dir_requires_image_size(tmp_path: Path) -> None:
    (tmp_path / "contract.json").write_text(
        json.dumps(
            {
                "model_config": {
                    "_target_": "fastwam.runtime.create_action_vggt_lang_recon_fastwam",
                    "vggt_camera_keys": ["cam_head"],
                    "proprio_dim": 14,
                    "action_dit_config": {"action_dim": 14},
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FastWAMBundledConfigError, match="image size"):
        load_bundle(tmp_path)


def test_adapter_validates_dimensions(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path)
    cfg = load_bundle(root)
    adapter = FastWAMVGGTPolicyAdapter(FakeRunner(), cfg, instruction="x")
    adapter.validate_io_dimensions(14, 14)
    with pytest.raises(ValueError, match="state dimension"):
        adapter.validate_io_dimensions(8, 14)
    with pytest.raises(ValueError, match="action dimension"):
        adapter.validate_io_dimensions(14, 7)
