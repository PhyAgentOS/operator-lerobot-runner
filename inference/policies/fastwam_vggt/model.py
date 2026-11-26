"""Build the FastWAM VGGT-Omega model from vendored source and load a checkpoint.

The training code ships as a normal Python package (`fastwam`) plus a vendored
`vggt_omega`. Both are copied into `third_party/fastwam/` so the node does not
depend on a separately managed checkout at runtime. `FASTWAM_VENDOR_DIR` can
override the location for local experiments.

Torch is imported lazily: bundle inspection and unit tests must work on a
machine without CUDA (or without torch at all).
"""

from __future__ import annotations

import logging
import os
import sys
import types
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MODULE_NAMES = ("action_expert", "proprio_encoder", "vggt_bridge", "vggt_backbone")

GRAPH_METADATA_KEYS = (
    "vggt_camera_keys",
    "layer_pairs",
    "bridge_config",
    "time_route_boundary",
    "route_rule",
)


class FastWAMModelError(RuntimeError):
    """Raised when the vendored model cannot be built or a checkpoint mismatches."""


def find_vendor_root(start: Path | None = None) -> Path:
    """Locate `third_party/fastwam` in a source tree or an installed package."""
    override = os.environ.get("FASTWAM_VENDOR_DIR")
    if override:
        root = Path(override).expanduser().resolve()
        if (root / "fastwam_source" / "src").is_dir():
            return root
        raise FastWAMModelError(
            f"FASTWAM_VENDOR_DIR does not contain fastwam_source/src: {root}"
        )

    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / "third_party" / "fastwam"
        if (candidate / "fastwam_source" / "src").is_dir():
            return candidate
    raise FastWAMModelError(
        "cannot find third_party/fastwam; set FASTWAM_VENDOR_DIR to the directory "
        "that contains fastwam_source/"
    )


def ensure_fastwam_importable(vendor_root: Path | None = None) -> Path:
    root = vendor_root or find_vendor_root()
    paths = (
        root / "fastwam_source" / "src",
        root / "fastwam_source" / "third_party" / "vggt-omega",
    )
    for path in paths:
        if not path.is_dir():
            raise FastWAMModelError(f"missing vendored package directory: {path}")
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return root


def _modules(model: Any) -> dict[str, Any]:
    return {
        "action_expert": model.action_expert,
        "proprio_encoder": model.proprio_encoder,
        "vggt_bridge": model.mot.vggt_bridge,
        "vggt_backbone": model.vggt_backbone,
    }


def _normalize_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_metadata(item) for item in value)
    return value


def build_model(
    model_config: dict[str, Any],
    *,
    device: str = "cpu",
    dtype: Any = None,
    vendor_root: Path | None = None,
) -> Any:
    """Instantiate the model described by `model_config` (hydra-style `_target_`)."""
    import torch

    from ._factory import (
        LANG_TARGET,
        MOE_TARGET,
        SUPPORTED_TARGETS,
        create_action_vggt_lang_recon_fastwam,
        create_action_vggt_moe_recon_fastwam,
    )

    ensure_fastwam_importable(vendor_root)
    config = dict(model_config)
    target = config.pop("_target_", None)
    if target not in SUPPORTED_TARGETS:
        raise FastWAMModelError(
            f"unsupported model _target_ {target!r}; supported: {sorted(SUPPORTED_TARGETS)}"
        )
    factory = {
        LANG_TARGET: create_action_vggt_lang_recon_fastwam,
        MOE_TARGET: create_action_vggt_moe_recon_fastwam,
    }[target]

    # The checkpoint carries the full state dict (VGGT backbone included), so the
    # constructor must never try to fetch training-time weights or a pretrained DiT.
    config["skip_vggt_load_from_pretrain"] = True
    config["vggt_weights_path"] = "unused"
    config["aggregator_checkpoint_path"] = None
    config["vggt_head_checkpoint_path"] = None
    config["skip_action_dit_load_from_pretrain"] = True

    model_dtype = dtype if dtype is not None else torch.bfloat16
    return factory(**config, model_dtype=model_dtype, device=device)


def enable_inference_mode(model: Any) -> Any:
    """Freeze the geometry heads; they are never executed for action inference."""

    if getattr(model, "_downstream_action_only", False):
        return model
    model.lambda_vggt_recon = 0.0
    for name in ("camera_head", "dense_head"):
        head = getattr(model.vggt_backbone, name, None)
        if head is None:
            raise FastWAMModelError(f"vggt_backbone is missing {name!r}")
        head.requires_grad_(False)
        head.eval()
    model._downstream_action_only = True
    return model


def load_checkpoint(model: Any, checkpoint: str | Path) -> int:
    """Load a FastWAM checkpoint into `model` and return its step."""
    import torch

    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FastWAMModelError(f"checkpoint does not exist: {path}")

    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(payload, dict):
        raise FastWAMModelError(f"unexpected checkpoint payload type: {type(payload)!r}")

    format_name = payload.get("format")
    if format_name != model.CHECKPOINT_FORMAT:
        raise FastWAMModelError(
            f"checkpoint format {format_name!r} does not match model "
            f"{model.CHECKPOINT_FORMAT!r}"
        )
    try:
        payload_version = int(payload.get("version"))
    except (TypeError, ValueError) as exc:
        raise FastWAMModelError(f"invalid checkpoint version: {payload.get('version')!r}") from exc
    model_version = int(model.CHECKPOINT_VERSION)
    if payload_version < model_version:
        raise FastWAMModelError(
            f"checkpoint version {payload_version} is older than model {model_version}"
        )
    if payload_version != model_version:
        # Newer checkpoints can carry the same graph with extra metadata; the
        # per-tensor key/shape validation below decides whether it is loadable.
        logger.info(
            "checkpoint version %s is newer than model %s; validating tensors",
            payload_version,
            model_version,
        )
    target = payload.get("target")
    if target is not None and target != "future":
        raise FastWAMModelError(f"unexpected checkpoint target: {target!r}")

    expected = _normalize_metadata(model._checkpoint_metadata())
    actual = dict(payload.get("metadata") or {})

    # Only fields that define the inference graph are compared. Everything else
    # in the metadata annotates how the weights were *trained* (reconstruction
    # targets, lambda weights, objective names); `infer_action` never runs those
    # heads, so a difference is informative, not a load error.
    required_keys = tuple(key for key in GRAPH_METADATA_KEYS if key in expected)
    missing_required = sorted(key for key in required_keys if key not in actual)
    if missing_required:
        raise FastWAMModelError(
            f"checkpoint metadata is missing graph fields: {missing_required}"
        )
    mismatched = {
        key: (expected[key], actual.get(key))
        for key in GRAPH_METADATA_KEYS
        if key in expected and _normalize_metadata(actual[key]) != expected[key]
    }
    if mismatched:
        raise FastWAMModelError(
            "checkpoint metadata contradicts the configured graph: "
            f"mismatched={sorted(mismatched)}"
        )
    training_only = sorted(
        key
        for key in set(expected) | set(actual)
        if key not in GRAPH_METADATA_KEYS and expected.get(key) != actual.get(key)
    )
    if training_only:
        logger.info(
            "checkpoint differs from this model only in training-time annotations: %s",
            training_only,
        )

    for name, module in _modules(model).items():
        source = payload.get(name)
        if source is None:
            raise FastWAMModelError(f"checkpoint is missing module {name!r}")
        target = module.state_dict()
        if set(source) != set(target):
            raise FastWAMModelError(f"{name}: checkpoint keys do not match the model")
        for key, value in source.items():
            if tuple(value.shape) != tuple(target[key].shape):
                raise FastWAMModelError(
                    f"{name}.{key}: shape {tuple(value.shape)} != {tuple(target[key].shape)}"
                )
        module.load_state_dict(source, strict=True, assign=True)

    step = payload.get("step")
    return int(step) if step is not None else -1


def build_and_load(
    model_config: dict[str, Any],
    checkpoint: str | Path,
    *,
    device: str = "cuda",
    dtype: Any = None,
    vendor_root: Path | None = None,
) -> tuple[Any, int]:
    """Build on CPU, load the checkpoint, then move the model to `device`."""
    import torch

    model = build_model(model_config, device="cpu", dtype=dtype, vendor_root=vendor_root)
    step = load_checkpoint(model, checkpoint)
    enable_inference_mode(model)
    model.to(device)
    model.eval()
    if dtype is not None:
        model.to(dtype=dtype)
    return model, step


__all__ = [
    "FastWAMModelError",
    "GRAPH_METADATA_KEYS",
    "MODULE_NAMES",
    "build_and_load",
    "build_model",
    "enable_inference_mode",
    "ensure_fastwam_importable",
    "find_vendor_root",
    "load_checkpoint",
]
