"""Thin inference runner around a built FastWAM VGGT-Omega model.

Mirrors the official `inference.py::Policy.infer` contract:

* images: uint8 RGB views already resized to the bundle image size, ordered like
  ``bundle.camera_keys`` with shape ``[V, H, W, 3]``
* state: float32 ``[proprio_dim]`` in raw units (normalized here)
* task: the instruction string; its UMT5 embedding is read from the cache
* output: float32 ``[action_horizon, action_dim]`` in raw units
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .bundle import FastWAMBundledConfig, FastWAMBundledConfigError


class FastWAMVGGTModelRunner:
    def __init__(
        self,
        model: Any,
        bundle: FastWAMBundledConfig,
        *,
        device: str = "cuda",
        seed: int | None = None,
    ) -> None:
        if bundle.checkpoint is None:
            raise FastWAMBundledConfigError("bundle.checkpoint is required")
        self.model = model
        self.bundle = bundle
        self.device = device
        self.seed = seed if seed is not None else bundle.seed
        self._text_cache: dict[str, tuple[Any, Any]] = {}

        state_norm = bundle.normalization.get("state") or {}
        action_norm = bundle.normalization.get("action") or {}
        self._state_mean = self._vector(state_norm.get("mean"), bundle.proprio_dim, "state.mean")
        self._state_std = self._vector(state_norm.get("std"), bundle.proprio_dim, "state.std")
        self._action_mean = self._vector(
            action_norm.get("mean"), bundle.action_dim, "action.mean"
        )
        self._action_std = self._vector(
            action_norm.get("std"), bundle.action_dim, "action.std"
        )
        self._steps = bundle.num_inference_steps
        self._horizon = bundle.action_horizon

    @staticmethod
    def _vector(value: Any, dim: int, name: str) -> np.ndarray | None:
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.shape != (dim,):
            raise FastWAMBundledConfigError(
                f"normalization.{name} must have {dim} values, got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise FastWAMBundledConfigError(f"normalization.{name} contains non-finite values")
        return array

    def _load_text(self, task: str) -> tuple[Any, Any]:
        import torch

        cached = self._text_cache.get(task)
        if cached is not None:
            return cached
        path = self.bundle.text_embedding_path(task)
        if not path.is_file():
            raise FastWAMBundledConfigError(
                f"missing UMT5 embedding for instruction {task!r}: {path}"
            )
        payload = torch.load(path, map_location="cpu", weights_only=True)
        prompt = str(payload.get("prompt", ""))
        expected_prompt = self.bundle.prompt_template.format(task=task)
        if prompt and prompt != expected_prompt:
            raise FastWAMBundledConfigError(
                f"text cache prompt mismatch for {path.name}: {prompt!r}"
            )
        context = payload.get("context")
        mask = payload.get("mask")
        if context is None or mask is None:
            raise FastWAMBundledConfigError(f"invalid text embedding file: {path}")
        if context.ndim != 2 or mask.ndim != 1 or context.shape[0] != mask.shape[0]:
            raise FastWAMBundledConfigError(
                f"invalid text embedding shapes in {path}: {tuple(context.shape)}, {tuple(mask.shape)}"
            )
        mask = mask.bool()
        context = context.clone()
        context[~mask] = 0
        self._text_cache[task] = (context, mask)
        return context, mask

    def infer(self, images: np.ndarray, state: np.ndarray, task: str) -> np.ndarray:
        import torch

        images = np.asarray(images)
        if images.dtype != np.uint8:
            raise ValueError(f"images must be uint8, got {images.dtype}")
        expected = (self.bundle.n_cameras, self.bundle.image_height, self.bundle.image_width, 3)
        if images.shape != expected:
            raise ValueError(f"images must have shape {expected}, got {images.shape}")

        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape != (self.bundle.proprio_dim,):
            raise ValueError(
                f"state must have {self.bundle.proprio_dim} values, got {state.shape}"
            )
        if not np.isfinite(state).all():
            raise ValueError("state contains non-finite values")

        proprio = state
        if self._state_mean is not None and self._state_std is not None:
            proprio = (state - self._state_mean) / self._state_std

        context, mask = self._load_text(task)
        with torch.inference_mode():
            result = self.model.infer_action(
                action_horizon=self._horizon,
                vggt_images=torch.from_numpy(images.copy())
                .permute(0, 3, 1, 2)
                .float()
                .div(255.0)
                .unsqueeze(0),
                context=context.unsqueeze(0),
                context_mask=torch.ones_like(mask).unsqueeze(0),
                language_context_mask=mask.unsqueeze(0),
                proprio=torch.from_numpy(proprio).unsqueeze(0),
                num_inference_steps=self._steps,
                seed=self.seed,
            )
        actions = result[0].numpy() if hasattr(result[0], "numpy") else np.asarray(result[0])
        actions = actions.astype(np.float32, copy=False)
        if self._action_mean is not None and self._action_std is not None:
            actions = actions * self._action_std + self._action_mean
        if actions.shape != (self._horizon, self.bundle.action_dim):
            raise ValueError(
                f"model returned {actions.shape}, expected "
                f"{(self._horizon, self.bundle.action_dim)}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("model returned non-finite actions")
        for index in self.bundle.gripper_indices:
            actions[:, index] = np.clip(actions[:, index], 0.0, 1.0)
        return actions


__all__ = ["FastWAMVGGTModelRunner"]
