"""Configuration discovery for FastWAM VGGT-Omega checkpoints.

The node can point at either:

1. a *bundle directory* with an explicit ``bundle.json`` manifest, or
2. a FastWAM training **run directory** (``contract.json`` or ``config.yaml``),
   from which a manifest is synthesized in place -- no packaging step needed.

Expected run-directory layouts:

    contract style (RoboDojo)          hydra style (LIBERO / RoboTwin)
      contract.json                      config.yaml
      normalization.json                 dataset_stats.json
      data/text/*.pt                     data/text_embeds_cache/<bench>/*.pt
      checkpoints/weights/step_*.pt      checkpoints/weights/step_*.pt
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following "
    "instruction: {task}"
)
BUNDLE_SCHEMA_VERSION = 1
MODEL_TARGET = "fastwam.runtime.create_action_vggt_lang_recon_fastwam"
MOE_MODEL_TARGET = "fastwam.runtime.create_action_vggt_moe_recon_fastwam"

_REQUIRED_FIELDS = (
    "model_config",
    "camera_keys",
    "image_size",
    "action_dim",
    "proprio_dim",
    "action_horizon",
)


class FastWAMBundledConfigError(ValueError):
    """Raised when a FastWAM bundle is missing or internally inconsistent."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FastWAMBundledConfigError(f"missing JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FastWAMBundledConfigError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FastWAMBundledConfigError(f"{path} must contain a JSON object")
    return value


def _resolve_payload(root: Path, value: Any, *, name: str) -> Any:
    """Return an inline object or load the relative/absolute path it names."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        path = Path(value)
        return _read_json(path if path.is_absolute() else root / path)
    raise FastWAMBundledConfigError(f"bundle.{name} must be an object or a path")


def _require(manifest: dict[str, Any], name: str) -> Any:
    if name not in manifest:
        raise FastWAMBundledConfigError(f"bundle.json is missing required field {name!r}")
    return manifest[name]


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FastWAMBundledConfigError(f"bundle.{name} must be a positive integer")
    return value


def _latest_checkpoint(root: Path) -> Path | None:
    for candidate in (root / "checkpoints" / "weights", root / "weights"):
        files = sorted(candidate.glob("step_*.pt"))
        if files:
            return files[-1]
    return None


def _normalization_from(stats_path: Path, proprio_dim: int, action_dim: int) -> dict[str, Any]:
    stats = _read_json(stats_path)
    result: dict[str, Any] = {}
    for name, dim in (("state", proprio_dim), ("action", action_dim)):
        section = (stats.get(name) or {}).get("default") or {}
        mean, std = section.get("global_mean"), section.get("global_std")
        if not mean or not std or len(mean) != dim or len(std) != dim:
            raise FastWAMBundledConfigError(
                f"{stats_path} lacks usable {name}.default.global_mean/std for dim {dim}"
            )
        result[name] = {"mean": list(mean), "std": list(std)}
    return result


def _find_text_dir(root: Path, configured: str | None = None) -> Path | None:
    candidates: list[Path] = []
    if configured:
        candidate = Path(configured)
        candidates.append(candidate if candidate.is_absolute() else root / candidate)
        candidates.append(Path(configured))
    candidates += [root / "data" / "text", root / "text"]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.pt")):
            return candidate.resolve()
    return None



def _model_config_from_checkpoint(checkpoint: Path) -> tuple[dict[str, Any], list[str]]:
    """Derive the model graph from the checkpoint itself.

    Tensor shapes give action/proprio dimensions, DiT width and depth; the
    checkpoint metadata gives camera keys, bridge config and the MoE routing
    parameters. No training-side config file is needed.
    """
    import torch

    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    metadata = payload.get("metadata") or {}
    bridge = metadata.get("bridge_config") or {}
    action_expert = payload["action_expert"]
    proprio_encoder = payload["proprio_encoder"]

    heads = int(bridge.get("fastwam_num_heads") or 24)
    q_weight = action_expert["blocks.0.self_attn.q.weight"]
    hidden = int(q_weight.shape[1])
    attn_head_dim = int(q_weight.shape[0]) // heads
    action_dim = int(action_expert["head.weight"].shape[0])
    proprio_dim = int(proprio_encoder["weight"].shape[1])
    num_layers = len(
        {key.split(".")[1] for key in action_expert if key.startswith("blocks.")}
    )
    ffn_dim = int(action_expert["blocks.0.ffn.0.weight"].shape[0])
    freq_dim = int(action_expert["time_embedding.0.weight"].shape[1])
    text_dim = int(
        metadata.get("text_dim")
        or payload.get("text_dim")
        or action_expert["text_embedding.0.weight"].shape[1]
    )
    camera_keys = list(metadata.get("vggt_camera_keys") or [])
    if not camera_keys:
        raise FastWAMBundledConfigError("checkpoint metadata has no vggt_camera_keys")

    config: dict[str, Any] = {
        "layer_pairs": [list(pair) for pair in metadata.get("layer_pairs") or []],
        "projection_mode": bridge.get("projection_mode", "vggt_to_fastwam"),
        "projection_sharing": bridge.get("projection_sharing", "independent"),
        "action_dit_config": {
            "action_dim": action_dim,
            "hidden_dim": hidden,
            "ffn_dim": ffn_dim,
            "num_heads": heads,
            "attn_head_dim": attn_head_dim,
            "num_layers": num_layers,
            "text_dim": text_dim,
            "freq_dim": freq_dim,
            "eps": 1e-6,
            "use_gradient_checkpointing": False,
        },
        "action_scheduler": {
            "train_shift": 5.0,
            "infer_shift": 5.0,
            "num_train_timesteps": int(metadata.get("num_train_timesteps") or 1000),
        },
        "proprio_dim": proprio_dim,
        "mot_checkpoint_mixed_attn": False,
        "vggt_width": int(bridge.get("vggt_width") or 1024),
        "vggt_num_heads": int(bridge.get("vggt_num_heads") or 16),
        "vggt_camera_keys": camera_keys,
        "train_vggt": bool(metadata.get("train_vggt", True)),
        "lambda_vggt_recon": float(metadata.get("lambda_vggt_recon", 0.01)),
        "action_dit_pretrained_path": None,
        "reconstruction_loss_config": metadata.get("reconstruction_loss_config")
        or {"backend": "official", "official": {}, "lambda_camera": 1.0,
            "lambda_depth": 1.0, "lambda_point": 1.0, "use_conf_loss": False,
            "conf_alpha": 0.2, "gradient_weight": 0.5},
    }
    if "moe" in str(payload.get("format", "")):
        sharing = metadata.get("sharing") or {}
        config["_target_"] = MOE_MODEL_TARGET
        config.update(
            {
                "time_route_boundary": float(
                    metadata.get("time_route_boundary", 0.8333333333333334)
                ),
                "low_noise_use_language_condition": bool(
                    metadata.get("low_noise_use_language_condition", True)
                ),
                "share_patch_embedding": bool(sharing.get("share_patch_embedding", True)),
                "share_camera_register_tokens": bool(
                    sharing.get("share_camera_register_tokens", True)
                ),
                "share_geometry_heads": bool(sharing.get("share_geometry_heads", True)),
                "share_bridge": bool(sharing.get("share_bridge", True)),
            }
        )
    else:
        config["_target_"] = MODEL_TARGET
    return config, camera_keys


def _synthesize_manifest(
    root: Path,
    *,
    image_size: tuple[int, int] | None,
    gripper_indices: list[int] | None,
) -> dict[str, Any]:
    contract_path = root / "contract.json"
    config_path = root / "config.yaml"
    configured_text: str | None = None
    dataset_image_size: tuple[int, int] | None = None

    if contract_path.is_file():
        contract = _read_json(contract_path)
        model_config = dict(contract["model_config"])
        target = (contract.get("objective") or {}).get("target_contract") or {}
        horizon = int(target.get("horizon") or 32)
        camera_keys = model_config.get("vggt_camera_keys") or target.get("camera_order")
    elif config_path.is_file():
        import yaml

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        model_config = dict(raw.get("model") or {})
        moe_markers = (
            "time_route_boundary",
            "checkpoint_sharing_policy",
            "share_bridge",
            "share_patch_embedding",
        )
        default_target = (
            MOE_MODEL_TARGET
            if any(marker in model_config for marker in moe_markers)
            else MODEL_TARGET
        )
        model_config.setdefault("_target_", default_target)
        data = (raw.get("data") or {}).get("train") or {}
        horizon = int(data.get("action_horizon") or 32)
        camera_keys = model_config.get("vggt_camera_keys")
        configured_text = data.get("text_embedding_cache_dir")
        images = ((data.get("shape_meta") or {}).get("images")) or []
        shape = images[0].get("shape") if images else None
        if shape and len(shape) == 3:
            dataset_image_size = (int(shape[2]), int(shape[1]))
    else:
        checkpoint = _latest_checkpoint(root)
        if checkpoint is None:
            raise FastWAMBundledConfigError(
                f"{root} has no bundle.json, contract.json, config.yaml or weights/step_*.pt"
            )
        model_config, camera_keys = _model_config_from_checkpoint(checkpoint)
        horizon = 32

    if not camera_keys:
        raise FastWAMBundledConfigError("model config declares no vggt_camera_keys")
    action_dim = int((model_config.get("action_dit_config") or {}).get("action_dim") or 0)
    proprio_dim = int(model_config.get("proprio_dim") or 0)
    if not action_dim or not proprio_dim:
        raise FastWAMBundledConfigError(
            "model config is missing action_dit_config.action_dim or proprio_dim"
        )

    resolved_image_size = image_size or dataset_image_size
    if resolved_image_size is None:
        raise FastWAMBundledConfigError(
            "image size is not recorded in this run; pass policy.image_size [width, height] "
            "(RoboDojo uses 256x192)"
        )

    norm_path = root / "normalization.json"
    stats_path = root / "dataset_stats.json"
    if norm_path.is_file():
        normalization: Any = "normalization.json"
    elif stats_path.is_file():
        normalization = _normalization_from(stats_path, proprio_dim, action_dim)
    else:
        normalization = {}

    text_dir = _find_text_dir(root, configured_text)
    checkpoint = _latest_checkpoint(root)

    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "model_config": model_config,
        "normalization": normalization,
        "text_dir": str(text_dir) if text_dir else None,
        "camera_keys": list(camera_keys),
        "image_size": [int(resolved_image_size[0]), int(resolved_image_size[1])],
        "action_dim": action_dim,
        "proprio_dim": proprio_dim,
        "action_horizon": horizon,
        "num_inference_steps": 10,
        "gripper_indices": list(gripper_indices or []),
        "prompt_template": DEFAULT_PROMPT,
    }


@dataclass(frozen=True)
class FastWAMBundledConfig:
    root: Path
    model_config: dict[str, Any]
    normalization: dict[str, Any]
    camera_keys: tuple[str, ...]
    image_width: int
    image_height: int
    action_dim: int
    proprio_dim: int
    action_horizon: int
    num_inference_steps: int = 10
    gripper_indices: tuple[int, ...] = ()
    prompt_template: str = DEFAULT_PROMPT
    text_dir: Path | None = None
    checkpoint: Path | None = None
    seed: int | None = None

    @property
    def n_cameras(self) -> int:
        return len(self.camera_keys)

    def text_embedding_path(self, task: str) -> Path:
        if self.text_dir is None:
            raise FastWAMBundledConfigError(
                "no text embedding cache found; language-conditioned inference needs "
                "the UMT5 cache (data/text or text_embeddings)"
            )
        prompt = self.prompt_template.format(task=task)
        return self.text_dir / (hashlib.sha256(prompt.encode("utf-8")).hexdigest() + ".pt")


def load_bundle(
    path: str | Path,
    *,
    checkpoint: str | Path | None = None,
    num_inference_steps: int | None = None,
    action_horizon: int | None = None,
    image_size: Any = None,
    gripper_indices: Any = None,
) -> FastWAMBundledConfig:
    """Load ``bundle.json`` or synthesize one from a FastWAM run directory."""
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FastWAMBundledConfigError(f"bundle path is not a directory: {root}")
    if (root / "bundle.json").is_file():
        manifest = _read_json(root / "bundle.json")
    else:
        requested_size = None
        if image_size is not None:
            values = list(image_size)
            if len(values) != 2:
                raise FastWAMBundledConfigError("policy.image_size must be [width, height]")
            requested_size = (int(values[0]), int(values[1]))
        manifest = _synthesize_manifest(
            root,
            image_size=requested_size,
            gripper_indices=list(gripper_indices) if gripper_indices else None,
        )

    version = manifest.get("schema_version", BUNDLE_SCHEMA_VERSION)
    if version != BUNDLE_SCHEMA_VERSION:
        raise FastWAMBundledConfigError(
            f"unsupported bundle schema_version {version!r} (expected {BUNDLE_SCHEMA_VERSION})"
        )
    missing = [name for name in _REQUIRED_FIELDS if name not in manifest]
    if missing:
        raise FastWAMBundledConfigError(f"bundle manifest is missing required fields: {missing}")

    camera_raw = _require(manifest, "camera_keys")
    if not isinstance(camera_raw, (list, tuple)) or not camera_raw:
        raise FastWAMBundledConfigError("camera_keys must be a non-empty list")
    camera_keys = tuple(str(key) for key in camera_raw)
    if len(set(camera_keys)) != len(camera_keys):
        raise FastWAMBundledConfigError("camera_keys must be unique")

    image_raw = manifest["image_size"]
    if (
        not isinstance(image_raw, (list, tuple))
        or len(image_raw) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) for v in image_raw)
    ):
        raise FastWAMBundledConfigError("image_size must be [width, height]")
    image_width, image_height = int(image_raw[0]), int(image_raw[1])
    if image_width <= 0 or image_height <= 0:
        raise FastWAMBundledConfigError("image_size must be positive")

    action_dim = _positive_int(manifest["action_dim"], "action_dim")
    proprio_dim = _positive_int(manifest["proprio_dim"], "proprio_dim")
    horizon = _positive_int(
        action_horizon if action_horizon is not None else manifest["action_horizon"],
        "action_horizon",
    )
    steps = _positive_int(
        num_inference_steps
        if num_inference_steps is not None
        else manifest.get("num_inference_steps", 10),
        "num_inference_steps",
    )

    gripper_raw = (
        gripper_indices
        if gripper_indices is not None
        else manifest.get("gripper_indices", [])
    )
    if not isinstance(gripper_raw, (list, tuple)):
        raise FastWAMBundledConfigError("gripper_indices must be a list")
    gripper_tuple = tuple(int(index) for index in gripper_raw)
    if any(index < 0 or index >= action_dim for index in gripper_tuple):
        raise FastWAMBundledConfigError("gripper_indices must be within [0, action_dim)")

    text_dir_raw = manifest.get("text_dir")
    text_dir = None
    if text_dir_raw:
        candidate = Path(str(text_dir_raw))
        text_dir = candidate if candidate.is_absolute() else root / candidate
        if not text_dir.is_dir():
            raise FastWAMBundledConfigError(f"text_dir does not exist: {text_dir}")

    checkpoint_raw = checkpoint if checkpoint is not None else manifest.get("checkpoint")
    checkpoint_path = None
    if checkpoint_raw:
        candidate = Path(str(checkpoint_raw)).expanduser()
        checkpoint_path = candidate if candidate.is_absolute() else root / candidate
        if not checkpoint_path.is_file():
            raise FastWAMBundledConfigError(f"checkpoint does not exist: {checkpoint_path}")

    seed_raw = manifest.get("seed")
    seed = None if seed_raw is None else int(seed_raw)

    return FastWAMBundledConfig(
        root=root,
        model_config=_resolve_payload(root, manifest["model_config"], name="model_config"),
        normalization=(
            _resolve_payload(root, manifest["normalization"], name="normalization")
            if manifest.get("normalization")
            else {}
        ),
        camera_keys=camera_keys,
        image_width=image_width,
        image_height=image_height,
        action_dim=action_dim,
        proprio_dim=proprio_dim,
        action_horizon=horizon,
        num_inference_steps=steps,
        gripper_indices=gripper_tuple,
        prompt_template=str(manifest.get("prompt_template", DEFAULT_PROMPT)),
        text_dir=text_dir,
        checkpoint=checkpoint_path,
        seed=seed,
    )


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "DEFAULT_PROMPT",
    "FastWAMBundledConfig",
    "FastWAMBundledConfigError",
    "load_bundle",
]
