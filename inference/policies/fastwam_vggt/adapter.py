"""Dora adapter that serves FastWAM VGGT-Omega checkpoints as a policy.

The adapter owns the parts the official `inference.py` does inline: camera
ordering, resize to the bundle image size, state normalization-free hand-off
(the runner normalizes), and the action-chunk queue. The model itself is
created by `runner.FastWAMVGGTModelRunner`, or injected for tests.
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Any, Callable

import numpy as np

from lerobot_inference.inference.policies.base import LerobotPolicyAdapter

from .bundle import FastWAMBundledConfig, load_bundle
from .runner import FastWAMVGGTModelRunner

logger = logging.getLogger(__name__)

RunnerFactory = Callable[[FastWAMBundledConfig, str], FastWAMVGGTModelRunner]
_IMAGE_PREFIX = "observation.images."


def _resize_hwc(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Aspect-preserving resize is *not* used: training resizes to a fixed size."""
    from PIL import Image

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected HWC RGB image, got shape {array.shape}")
    if array.dtype != np.uint8:
        array = array.astype(np.uint8, copy=False)
    if array.shape[0] == height and array.shape[1] == width:
        return np.ascontiguousarray(array)
    resized = Image.fromarray(array).resize((width, height), Image.Resampling.BICUBIC)
    return np.asarray(resized, dtype=np.uint8)


class FastWAMVGGTPolicyAdapter(LerobotPolicyAdapter):
    def __init__(
        self,
        runner: FastWAMVGGTModelRunner,
        bundle: FastWAMBundledConfig,
        *,
        instruction: str = "",
        expected_image_keys: set[str] | None = None,
    ) -> None:
        self._runner = runner
        self._bundle = bundle
        self._instruction = str(instruction)
        self._expected_image_keys = (
            frozenset(expected_image_keys) if expected_image_keys is not None else None
        )
        self._queue: deque[np.ndarray] = deque()

    @classmethod
    def from_pretrained(
        cls,
        bundle_path: str | Path,
        *,
        checkpoint: str | Path | None = None,
        device: str | None = None,
        instruction: str = "",
        expected_image_keys: set[str] | None = None,
        num_inference_steps: int | None = None,
        action_horizon: int | None = None,
        image_size: Any = None,
        gripper_indices: Any = None,
        seed: int | None = None,
        dtype: Any = None,
        runner_factory: RunnerFactory | None = None,
    ) -> "FastWAMVGGTPolicyAdapter":
        bundle = load_bundle(
            bundle_path,
            checkpoint=checkpoint,
            num_inference_steps=num_inference_steps,
            action_horizon=action_horizon,
            image_size=image_size,
            gripper_indices=gripper_indices,
        )
        if bundle.checkpoint is None:
            raise ValueError(
                "FastWAM bundle has no checkpoint; pass policy.checkpoint or put a "
                "weights/step_*.pt file in the bundle"
            )
        if expected_image_keys is not None:
            expected = {str(key) for key in expected_image_keys}
            declared = {f"{_IMAGE_PREFIX}{camera}" for camera in bundle.camera_keys}
            missing = sorted(declared - expected)
            unexpected = sorted(expected - declared)
            if missing or unexpected:
                raise ValueError(
                    "FastWAM camera mapping does not match the bundle: "
                    f"missing={missing}, unexpected={unexpected}, "
                    f"bundle={sorted(declared)}"
                )

        resolved_device = device or _default_device()
        if runner_factory is not None:
            runner = runner_factory(bundle, resolved_device)
        else:
            from .model import build_and_load

            model, step = build_and_load(
                bundle.model_config,
                bundle.checkpoint,
                device=resolved_device,
                dtype=dtype,
            )
            logger.info(
                "Loaded FastWAM checkpoint %s (step %s) on %s; cameras=%s image=%dx%d",
                bundle.checkpoint,
                step,
                resolved_device,
                ",".join(bundle.camera_keys),
                bundle.image_width,
                bundle.image_height,
            )
            runner = FastWAMVGGTModelRunner(
                model, bundle, device=resolved_device, seed=seed
            )
        return cls(
            runner,
            bundle,
            instruction=instruction,
            expected_image_keys=expected_image_keys,
        )

    @property
    def bundle(self) -> FastWAMBundledConfig:
        return self._bundle

    @property
    def required_image_keys(self) -> frozenset[str]:
        if self._expected_image_keys is not None:
            return self._expected_image_keys
        return frozenset(f"{_IMAGE_PREFIX}{camera}" for camera in self._bundle.camera_keys)

    @property
    def instruction(self) -> str:
        return self._instruction

    @instruction.setter
    def instruction(self, value: str) -> None:
        updated = str(value)
        if updated == self._instruction:
            return
        self._instruction = updated
        self.reset()

    def validate_io_dimensions(self, state_dim: int, action_dim: int) -> None:
        if int(state_dim) != self._bundle.proprio_dim:
            raise ValueError(
                "FastWAM runtime state dimension does not match the bundle: "
                f"state_joints={state_dim}, bundle_proprio_dim={self._bundle.proprio_dim}"
            )
        if int(action_dim) != self._bundle.action_dim:
            raise ValueError(
                "FastWAM runtime action dimension does not match the bundle: "
                f"action_joints={action_dim}, bundle_action_dim={self._bundle.action_dim}"
            )

    def reset(self) -> None:
        self._queue.clear()

    def pause(self) -> None:
        self.reset()

    def stop(self) -> None:
        self.reset()

    def is_observation_needed(self) -> bool:
        return not self._queue

    def generate_action(
        self,
        observation: dict[str, Any],
        alias_for_cameras: list[str] | None = None,
    ) -> np.ndarray | None:
        if not self._queue:
            actions = self._predict(observation, alias_for_cameras)
            self._queue.extend(np.asarray(action, dtype=np.float32) for action in actions)
        if not self._queue:
            return None
        return self._queue.popleft()

    def _predict(
        self,
        observation: dict[str, Any],
        alias_for_cameras: list[str] | None,
    ) -> np.ndarray:
        images = self._collect_images(observation, alias_for_cameras)
        state = self._collect_state(observation)
        if not self._instruction:
            raise ValueError("FastWAM language conditioning requires a non-empty instruction")
        return self._runner.infer(images, state, self._instruction)

    def _collect_images(
        self,
        observation: dict[str, Any],
        alias_for_cameras: list[str] | None,
    ) -> np.ndarray:
        if not alias_for_cameras:
            raise ValueError(
                "FastWAM needs image_inputs/alias_for_cameras so cameras map to "
                f"{list(self._bundle.camera_keys)}"
            )
        aliases = [str(alias) for alias in alias_for_cameras]
        missing = [camera for camera in self._bundle.camera_keys if camera not in aliases]
        unexpected = [alias for alias in aliases if alias not in self._bundle.camera_keys]
        if missing or unexpected:
            raise ValueError(
                "FastWAM camera aliases do not match the bundle: "
                f"missing={missing}, unexpected={unexpected}, "
                f"bundle={list(self._bundle.camera_keys)}"
            )
        frames = []
        for camera in self._bundle.camera_keys:
            key = f"{_IMAGE_PREFIX}{camera}"
            image = observation.get(key)
            if image is None:
                raise ValueError(f"observation is missing {key}")
            frames.append(
                _resize_hwc(
                    image,
                    self._bundle.image_width,
                    self._bundle.image_height,
                )
            )
        return np.stack(frames, axis=0)

    def _collect_state(self, observation: dict[str, Any]) -> np.ndarray:
        state = observation.get("observation.state")
        if state is None:
            raise ValueError("observation is missing observation.state")
        array = np.asarray(state, dtype=np.float32).reshape(-1)
        if array.shape != (self._bundle.proprio_dim,):
            raise ValueError(
                f"observation.state must have {self._bundle.proprio_dim} values, "
                f"got {array.shape}"
            )
        return array


def _default_device() -> str:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency of the node
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


__all__ = ["FastWAMVGGTPolicyAdapter"]
