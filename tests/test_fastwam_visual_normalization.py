"""Exercise saved legacy processors through the real adapter loading path."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.fastwam.wan.modular import FastWAM
from lerobot_inference.inference.policies.fastwam import adapter as module


@pytest.mark.parametrize("visual_mode", ["MEAN_STD", "IDENTITY"])
def test_saved_processors_normalize_rgb_once_and_preserve_state_actions(
    tmp_path, monkeypatch, visual_mode
):
    checkpoint = tmp_path / "checkpoint"
    shutil.copytree(Path(__file__).parent / "fixtures/fastwam_legacy_processors", checkpoint)
    path = checkpoint / "policy_preprocessor.json"
    payload = json.loads(path.read_text())
    payload["steps"][3]["config"]["norm_map"]["VISUAL"] = visual_mode
    path.write_text(json.dumps(payload))
    original = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint.iterdir()}
    captured = {}

    class Policy(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.marker = torch.nn.Parameter(torch.zeros(1))

        def select_action(self, batch):
            captured["batch"] = batch
            return torch.tensor([[-1., -.5, 0., .5, 1., 0., 1.]])

    monkeypatch.setattr(module, "ensure_fastwam_offline_assets", lambda *args: (checkpoint, tmp_path, tmp_path))
    monkeypatch.setattr(module, "load_fastwam_policy_strict", lambda cls, path, config, **kw: Policy(config))
    adapter = module.FastWAMPolicyAdapter.from_pretrained(
        str(checkpoint), wan_diffusers_path=str(tmp_path), tokenizer_path=str(tmp_path),
        device="cpu", instruction="pick up the bowl",
    )
    pixels = np.broadcast_to(np.array([0., .5, 1.], dtype=np.float32), (224, 224, 3)).copy()
    observation = {"observation.state": np.zeros(8, dtype=np.float32),
                   "observation.images.image": pixels, "observation.images.image2": pixels.copy()}
    action = adapter.generate_action(observation)
    image = captured["batch"]["observation.images.image"]
    torch.testing.assert_close(image[0, :, 0, 0], torch.tensor([0., .5, 1.]))
    torch.testing.assert_close(captured["batch"]["observation.images.image2"], image)

    class VAE:
        def encode(self, images, **kwargs):
            captured["vae_image"] = images[0]
            return torch.zeros(1, 48, 1, 1, 1)

    FastWAM._encode_input_image_latents_tensor(
        SimpleNamespace(device="cpu", vae=VAE()), image
    )
    torch.testing.assert_close(captured["vae_image"][:, 0, 0, 0], torch.tensor([-1., 0., 1.]))

    legacy_pre, legacy_post = make_pre_post_processors(
        adapter._policy.config, pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    legacy_batch = legacy_pre({"observation.state": torch.zeros(1, 8),
                               "observation.images.image": image.clone(),
                               "observation.images.image2": image.clone()})
    torch.testing.assert_close(captured["batch"]["observation.state"], legacy_batch["observation.state"])
    expected = legacy_post(torch.tensor([[-1., -.5, 0., .5, 1., 0., 1.]])).numpy()[0]
    np.testing.assert_array_equal(action, expected)
    assert action[-1] == -1.0
    assert original == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint.iterdir()}
