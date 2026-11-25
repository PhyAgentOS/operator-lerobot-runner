"""Validate local FastWAM checkpoints and frozen Wan runtime assets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _required_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing FastWAM {label}: {path}")
    return path


def _required_dir(value: str | Path | None, label: str) -> Path:
    if not value:
        raise ValueError(f"{label} is required for offline FastWAM inference")
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{label} must be an existing local directory: {path}")
    return path


def _validate_processor_assets(checkpoint: Path, filename: str) -> None:
    path = _required_file(checkpoint / filename, filename)
    try:
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid FastWAM processor config: {path}") from exc
    steps = payload.get("steps")
    if not isinstance(steps, list):
        raise ValueError(f"FastWAM processor config has no steps list: {path}")
    for step in steps:
        if not isinstance(step, dict):
            raise ValueError(f"Invalid processor step in {path}: {step!r}")
        state_file = step.get("state_file")
        if state_file:
            _required_file(checkpoint / str(state_file), f"processor state {state_file}")


def ensure_fastwam_offline_assets(
    pretrained_path: str | Path,
    wan_diffusers_path: str | Path | None,
    tokenizer_path: str | Path | None,
) -> tuple[Path, Path, Path]:
    """Resolve and validate all assets needed without accessing Hugging Face Hub."""
    checkpoint = _required_dir(pretrained_path, "policy.pretrained_path")
    wan_path = _required_dir(wan_diffusers_path, "policy.wan_diffusers_path")
    tokenizer = _required_dir(tokenizer_path, "policy.tokenizer_path")

    _required_file(checkpoint / "config.json", "config.json")
    _required_file(checkpoint / "model.safetensors", "model.safetensors")
    _validate_processor_assets(checkpoint, "policy_preprocessor.json")
    _validate_processor_assets(checkpoint, "policy_postprocessor.json")

    _required_file(wan_path / "vae" / "config.json", "Wan VAE config")
    _required_file(
        wan_path / "vae" / "diffusion_pytorch_model.safetensors",
        "Wan VAE weights",
    )
    _required_file(wan_path / "text_encoder" / "config.json", "UMT5 encoder config")
    _required_file(
        wan_path / "text_encoder" / "model.safetensors.index.json",
        "UMT5 encoder index",
    )
    if not list((wan_path / "text_encoder").glob("model-*.safetensors")):
        raise FileNotFoundError(
            f"Missing FastWAM UMT5 encoder weight shards: {wan_path / 'text_encoder'}"
        )

    _required_file(tokenizer / "tokenizer_config.json", "UMT5 tokenizer config")
    _required_file(tokenizer / "special_tokens_map.json", "UMT5 special tokens")
    if not (tokenizer / "tokenizer.json").is_file() and not (tokenizer / "spiece.model").is_file():
        raise FileNotFoundError(
            f"Missing FastWAM tokenizer.json or spiece.model: {tokenizer}"
        )
    return checkpoint, wan_path, tokenizer
