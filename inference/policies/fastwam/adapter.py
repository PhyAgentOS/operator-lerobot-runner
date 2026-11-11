"""Synchronous LeRobot FastWAM policy adapter."""

from __future__ import annotations

from contextlib import nullcontext
from functools import partial
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline

from lerobot_inference.inference.observation import action_tensor_to_numpy
from lerobot_inference.inference.policies.base import LerobotPolicyAdapter
from lerobot_inference.inference.policies.loader import load_policy_bundle

from .assets import ensure_fastwam_offline_assets
from .loading import load_fastwam_policy_strict

_IMAGE_KEY_PREFIX = "observation.images."


def _fastwam_preprocessor_overrides(checkpoint: Path) -> dict[str, Any]:
    """Keep RGB in [0, 1] until LeRobot 0.6.1's Wan VAE encode boundary.

    Older converted checkpoints saved VISUAL=MEAN_STD with mean/std=0.5.
    Replaying that step would normalize twice, feeding [-3, 1] into the VAE.
    Override only VISUAL in memory, preserving checkpoint statistics and files.
    """
    payload = json.loads((checkpoint / "policy_preprocessor.json").read_text())
    normalizers = [
        step for step in payload["steps"]
        if step.get("registry_name") == "normalizer_processor"
    ]
    if len(normalizers) != 1:
        raise ValueError("FastWAM preprocessor must contain exactly one normalizer_processor")
    norm_map = dict(normalizers[0]["config"]["norm_map"])
    norm_map["VISUAL"] = "IDENTITY"
    return {"normalizer_processor": {"norm_map": norm_map}}


def _validate_fastwam_config(config: Any, runtime_image_keys: set[str] | None) -> None:
    if config.type != "fastwam":
        raise ValueError(f"FastWAMPolicyAdapter expects type fastwam, got {config.type!r}")
    model_image_keys = set(config.image_features)
    if runtime_image_keys is not None and runtime_image_keys != model_image_keys:
        raise ValueError(
            "FastWAM runtime image inputs must exactly match the checkpoint because cameras "
            "are concatenated along image width: "
            f"runtime={sorted(runtime_image_keys)}, checkpoint={sorted(model_image_keys)}"
        )


class FastWAMPolicyAdapter(LerobotPolicyAdapter):
    """FastWAM inference through LeRobot processors and native action queue."""

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        *,
        instruction: str,
        expected_image_keys: set[str],
    ) -> None:
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._instruction = str(instruction)
        self._expected_image_keys = set(expected_image_keys)
        self._queued_steps_remaining = 0
        try:
            self._device = next(policy.parameters()).device
        except StopIteration:
            self._device = torch.device(getattr(policy.config, "device", "cpu") or "cpu")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        *,
        wan_diffusers_path: str,
        tokenizer_path: str,
        device: str | None = None,
        instruction: str,
        expected_image_keys: set[str] | None = None,
        policy_config_overrides: dict[str, Any] | None = None,
    ) -> FastWAMPolicyAdapter:
        checkpoint, wan_path, tokenizer = ensure_fastwam_offline_assets(
            pretrained_path,
            wan_diffusers_path,
            tokenizer_path,
        )
        runtime_image_keys = (
            set(expected_image_keys) if expected_image_keys is not None else None
        )
        overrides = dict(policy_config_overrides or {})
        overrides.update(
            {
                "tokenizer_model_id": str(tokenizer),
                "text_encoder_model_id": str(wan_path),
            }
        )
        policy, preprocessor, postprocessor, config = load_policy_bundle(
            checkpoint,
            device=device,
            preprocessor_overrides=_fastwam_preprocessor_overrides(checkpoint),
            policy_config_overrides=overrides,
            policy_loader=partial(
                load_fastwam_policy_strict,
                wan_diffusers_path=wan_path,
            ),
            config_transform=lambda value: _validate_fastwam_config(
                value,
                runtime_image_keys,
            ),
        )
        return cls(
            policy,
            preprocessor,
            postprocessor,
            instruction=instruction,
            expected_image_keys=set(config.image_features),
        )

    @property
    def required_image_keys(self) -> frozenset[str]:
        return frozenset(self._expected_image_keys)

    @property
    def instruction(self) -> str:
        return self._instruction

    @instruction.setter
    def instruction(self, value: str) -> None:
        updated = str(value)
        if updated != self._instruction:
            self._instruction = updated
            self.reset()

    def validate_io_dimensions(self, state_dim: int, action_dim: int) -> None:
        state_feature = self._policy.config.input_features.get("observation.state")
        action_feature = self._policy.config.output_features.get("action")
        if state_feature is None or action_feature is None:
            raise ValueError("FastWAM config missing observation.state or action features")
        model_state_dim = int(state_feature.shape[0])
        model_action_dim = int(action_feature.shape[0])
        if state_dim != model_state_dim or action_dim != model_action_dim:
            raise ValueError(
                "FastWAM runtime dimensions do not match checkpoint: "
                f"state_joints={state_dim}, model_state_dim={model_state_dim}, "
                f"action_joints={action_dim}, model_action_dim={model_action_dim}"
            )

    def reset(self) -> None:
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()
        self._queued_steps_remaining = 0

    def pause(self) -> None:
        self.reset()

    def stop(self) -> None:
        self.reset()

    def is_observation_needed(self) -> bool:
        return self._queued_steps_remaining == 0

    def _copy_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        image_keys = {key for key in observation if key.startswith(_IMAGE_KEY_PREFIX)}
        if image_keys != self._expected_image_keys:
            raise ValueError(
                "FastWAM observation image keys must exactly match the checkpoint: "
                f"received={sorted(image_keys)}, expected={sorted(self._expected_image_keys)}"
            )
        if "observation.state" not in observation:
            raise KeyError("Missing observation.state")

        prepared = dict(observation)
        state = np.array(
            prepared["observation.state"],
            dtype=np.float32,
            order="C",
            copy=True,
        )
        expected_state_dim = int(
            self._policy.config.input_features["observation.state"].shape[0]
        )
        if state.ndim != 1 or state.shape[0] != expected_state_dim:
            raise ValueError(
                "FastWAM observation.state shape does not match checkpoint: "
                f"received={state.shape}, expected=({expected_state_dim},)"
            )
        if not np.isfinite(state).all():
            raise ValueError("observation.state must contain only finite values")
        prepared["observation.state"] = state

        for key in self._expected_image_keys:
            image = np.asarray(prepared[key])
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"Expected HWC RGB image for {key}, got shape={image.shape}")
            if image.dtype == np.uint8:
                pass
            elif np.issubdtype(image.dtype, np.floating):
                if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                    raise ValueError(
                        f"Floating image {key} must contain finite values in [0, 1]"
                    )
            else:
                raise ValueError(
                    f"Image {key} must use uint8 or floating [0, 1] values, got {image.dtype}"
                )
            prepared[key] = np.array(image, order="C", copy=True)
        return prepared

    def _autocast_context(self):
        if self._device.type != "cuda" or not bool(self._policy.config.use_amp):
            return nullcontext()
        dtype_name = str(getattr(self._policy.config, "torch_dtype", "bfloat16"))
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }.get(dtype_name)
        if dtype is None:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=dtype)

    def generate_action(
        self,
        observation: dict[str, Any],
        alias_for_cameras: list[str] | None = None,
    ) -> np.ndarray:
        del alias_for_cameras
        if observation and not self.is_observation_needed():
            raise ValueError("FastWAM still has queued actions; do not provide a new observation")
        if observation:
            prepared = self._copy_observation(observation)
            batch = prepare_observation_for_inference(
                prepared,
                self._device,
                task=self._instruction,
                robot_type="",
            )
        elif self.is_observation_needed():
            raise ValueError("FastWAM action queue is depleted; a fresh observation is required")
        else:
            batch = {}

        with torch.inference_mode(), self._autocast_context():
            if observation:
                batch = self._preprocessor(batch)
            action = self._policy.select_action(batch)
            if observation:
                self._queued_steps_remaining = max(
                    int(self._policy.config.n_action_steps) - 1,
                    0,
                )
            else:
                self._queued_steps_remaining -= 1
            action = self._postprocessor(action)

        output = action_tensor_to_numpy(action)
        expected_dim = int(self._policy.config.output_features["action"].shape[0])
        if output.ndim != 1 or output.shape[0] != expected_dim:
            raise ValueError(
                "FastWAM action shape does not match checkpoint: "
                f"received={output.shape}, expected=({expected_dim},)"
            )
        if not np.isfinite(output).all():
            raise ValueError("FastWAM action must contain only finite values")
        return output.astype(np.float32, copy=False)
