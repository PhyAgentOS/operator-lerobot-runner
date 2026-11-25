"""LeRobot G0.5 (OpenGalaxea) policy adapter for Dora inference."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline
from lerobot_inference.inference.observation import action_tensor_to_numpy
from lerobot_inference.inference.policies.base import LerobotPolicyAdapter
from lerobot_inference.inference.policies.loader import load_policy_bundle

logger = logging.getLogger(__name__)


class G05Adapter(LerobotPolicyAdapter):
    """Instruction-conditioned G0.5 inference via LeRobot ``select_action``.

    G0.5 predicts a ``chunk_size`` action chunk per inference and LeRobot's
    queue serves ``n_action_steps`` of it, one step per call. The checkpoint's
    postprocessor unnormalizes each queued action with that timestep's slice of
    the statistics and counts calls to know which slice applies. Two rules keep
    that counter aligned with the policy queue:

    * every ``select_action`` result goes through the postprocessor, and
      ``reset`` resets the policy and both pipelines together;
    * ``n_action_steps`` cannot be overridden at runtime, because the
      postprocessor's step count is serialized in the checkpoint.

    The embodiment projection (e.g. RoboTwin's 14 joints into the 20-slot
    policy layout) happens inside the processors, so the runtime feeds and
    receives the embodiment's raw width as declared in ``config.json``.
    """

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
        self._expected_image_keys = expected_image_keys
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
        device: str | None = None,
        instruction: str = "",
        expected_image_keys: set[str] | None = None,
        policy_config_overrides: dict[str, Any] | None = None,
    ) -> G05Adapter:
        if policy_config_overrides and "n_action_steps" in policy_config_overrides:
            raise ValueError(
                "G0.5 n_action_steps is fixed by the checkpoint's postprocessor and cannot be overridden"
            )
        policy, preprocessor, postprocessor, config = load_policy_bundle(
            pretrained_path,
            device=device,
            policy_config_overrides=policy_config_overrides,
        )
        if config.type != "g05":
            raise ValueError(f"G05Adapter expects type g05, got {config.type!r}")
        model_image_keys = set(config.image_features)
        if expected_image_keys is not None and expected_image_keys != model_image_keys:
            missing = sorted(model_image_keys - expected_image_keys)
            unexpected = sorted(expected_image_keys - model_image_keys)
            raise ValueError(
                "G0.5 camera mapping does not match checkpoint features: "
                f"missing={missing}, unexpected={unexpected}, "
                f"checkpoint={sorted(model_image_keys)}"
            )
        logger.info(
            "G0.5 embodiment=%s runtime_system=%s n_action_steps=%s",
            getattr(config, "embodiment", None),
            getattr(config, "runtime_system", None),
            config.n_action_steps,
        )
        return cls(
            policy,
            preprocessor,
            postprocessor,
            instruction=instruction,
            expected_image_keys=model_image_keys,
        )

    def validate_io_dimensions(self, state_dim: int, action_dim: int) -> None:
        config = self._policy.config
        state_feat = config.input_features.get("observation.state")
        action_feat = config.output_features.get("action")
        if state_feat is None or action_feat is None:
            raise ValueError("G0.5 config must declare observation.state and action features")
        model_state_dim = int(state_feat.shape[0])
        model_action_dim = int(action_feat.shape[0])
        if (state_dim, action_dim) != (model_state_dim, model_action_dim):
            raise ValueError(
                "G0.5 runtime dimensions do not match checkpoint: "
                f"state_joints={state_dim}, action_joints={action_dim}, "
                f"model_state_dim={model_state_dim}, model_action_dim={model_action_dim}"
            )

    @property
    def required_image_keys(self) -> frozenset[str]:
        return frozenset(self._expected_image_keys)

    @property
    def instruction(self) -> str:
        return self._instruction

    @instruction.setter
    def instruction(self, value: str) -> None:
        # The instruction is tokenized into every prediction; a changed task
        # must not reuse actions predicted for the previous one.
        new_instruction = str(value)
        if new_instruction == self._instruction:
            return
        self._instruction = new_instruction
        self.reset()

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

    def _copy_observation(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        missing = [key for key in sorted(self._expected_image_keys) if key not in observation]
        if missing:
            raise KeyError(f"Missing camera observations: {missing}")
        if "observation.state" not in observation:
            raise KeyError("Missing observation.state")
        prepared = dict(observation)
        state = np.array(prepared["observation.state"], dtype=np.float32, order="C", copy=True)
        if state.ndim != 1:
            raise ValueError(f"Expected 1-D observation.state, got shape={state.shape}")
        if not np.isfinite(state).all():
            raise ValueError("observation.state must contain only finite values")
        prepared["observation.state"] = state

        for image_key in self._expected_image_keys:
            image = np.asarray(prepared[image_key])
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"Expected HWC RGB image for {image_key}, got shape={image.shape}")
            if image.dtype == np.uint8:
                pass
            elif np.issubdtype(image.dtype, np.floating):
                if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                    raise ValueError(f"Floating image {image_key} must contain finite values in [0, 1]")
            else:
                raise ValueError(
                    f"Image {image_key} must use uint8 or floating [0, 1] values, got dtype={image.dtype}"
                )
            prepared[image_key] = np.array(image, order="C", copy=True)
        return prepared

    def generate_action(
        self,
        observation: dict[str, Any],
        alias_for_cameras: list[str] | None = None,
    ) -> np.ndarray:
        del alias_for_cameras  # keys already use observation.images.<alias>
        if observation and not self.is_observation_needed():
            raise ValueError("G0.5 still has queued actions; do not provide a new observation")
        if not observation and self.is_observation_needed():
            raise ValueError("G0.5 action queue is depleted; a fresh observation is required")

        with torch.inference_mode():
            if observation:
                batch = prepare_observation_for_inference(
                    self._copy_observation(observation),
                    self._device,
                    task=self._instruction,
                    robot_type="",
                )
                batch = self._preprocessor(batch)
            else:
                batch = {}
            action = self._policy.select_action(batch)
            if isinstance(action, dict):
                # PolicyOutput (CoT runtimes) also carries the text prediction; only the action is published.
                action = action["action"]
            # LeRobot has consumed one queued action even if postprocessing fails.
            if observation:
                self._queued_steps_remaining = max(int(self._policy.config.n_action_steps) - 1, 0)
            else:
                self._queued_steps_remaining -= 1
            action = self._postprocessor(action)
        return action_tensor_to_numpy(action)
