"""Unit tests for the FastWAM VGGT-Omega adapter (no torch / no GPU needed)."""

from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
    prompt = bundle_mod.DEFAULT_PROMPT.format(task="Stack the bowls")
    expected = root / "text" / (hashlib.sha256(prompt.encode()).hexdigest() + ".t5_len128.wan22ti2v5b.pt")
    expected.write_bytes(b"stub")
    assert cfg.text_embedding_path("Stack the bowls") == expected


class _EncoderStub:
    """Minimal stand-in for a fastwam_text_encoder node."""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:  # noqa: A003
                pass

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length) or b"{}")
                prompt = payload["prompts"][0]
                stub.prompts.append(prompt)
                digest = hashlib.sha256(prompt.encode()).hexdigest()
                path = Path(payload["cache_dir"]) / f"{digest}.t5_len128.wan22ti2v5b.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"stub")
                body = json.dumps(
                    {"results": [{"prompt": prompt, "path": str(path), "written": True}]}
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def test_text_encoder_service_is_used_on_cache_miss(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path)
    stub = _EncoderStub()
    try:
        cfg = load_bundle(root, text_encoder_url=stub.url)
        path = cfg.text_embedding_path("Stack the bowls")
        assert path.is_file()
        assert stub.prompts == [bundle_mod.DEFAULT_PROMPT.format(task="Stack the bowls")]
    finally:
        stub.close()


def test_text_encoder_url_can_come_from_env(tmp_path: Path, monkeypatch) -> None:
    root = _write_bundle(tmp_path)
    monkeypatch.setenv(bundle_mod.TEXT_ENCODER_URL_ENV, "http://127.0.0.1:19999")
    cfg = load_bundle(root)
    assert cfg.text_encoder_url == "http://127.0.0.1:19999"


def test_missing_text_without_encoder_is_actionable(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path)
    cfg = load_bundle(root)
    with pytest.raises(FastWAMBundledConfigError, match="text_encoder_url"):
        cfg.text_embedding_path("Stack the bowls")


def test_text_encoder_failure_is_reported(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path)
    cfg = load_bundle(root, text_encoder_url="http://127.0.0.1:1")
    with pytest.raises(FastWAMBundledConfigError, match="text encoder"):
        cfg.text_embedding_path("Stack the bowls")


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


def test_model_config_from_checkpoint_derives_graph(tmp_path, monkeypatch):
    """A bare checkpoint (no contract/config.yaml) is enough to describe the graph."""
    import torch

    def fake_tensor(*shape):
        return torch.zeros(*shape)

    payload = {
        "format": "action_vggt_moe_recon_fastwam",
        "version": 3,
        "metadata": {
            "bridge_config": {
                "projection_mode": "vggt_to_fastwam",
                "projection_sharing": "independent",
                "fastwam_num_heads": 24,
                "vggt_width": 1024,
                "vggt_num_heads": 16,
            },
            "vggt_camera_keys": ["image", "wrist_image"],
            "layer_pairs": [[0, 3]],
            "time_route_boundary": 0.5,
            "sharing": {"share_bridge": False},
            "reconstruction_loss_config": {"backend": "official"},
        },
        "action_expert": {
            "blocks.0.self_attn.q.weight": fake_tensor(48, 16),
            "blocks.0.ffn.0.weight": fake_tensor(32, 16),
            "time_embedding.0.weight": fake_tensor(16, 8),
            "text_embedding.0.weight": fake_tensor(16, 4096),
            "head.weight": fake_tensor(7, 16),
        },
        "proprio_encoder": {"weight": fake_tensor(16, 8)},
    }
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: payload)

    config, cameras = bundle_mod._model_config_from_checkpoint(tmp_path / "step_000001.pt")
    assert cameras == ["image", "wrist_image"]
    assert config["_target_"] == bundle_mod.MOE_MODEL_TARGET
    assert config["action_dit_config"]["action_dim"] == 7
    assert config["action_dit_config"]["hidden_dim"] == 16
    assert config["action_dit_config"]["attn_head_dim"] == 2
    assert config["action_dit_config"]["text_dim"] == 4096
    assert config["proprio_dim"] == 8
    assert config["time_route_boundary"] == 0.5
    assert config["share_bridge"] is False
