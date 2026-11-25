#!/usr/bin/env python3
"""Assemble a FastWAM inference bundle from a training run directory.

Supported run layouts:

* contract-style (RoboDojo): ``contract.json`` with ``model_config`` and a
  sibling ``normalization.json`` + ``data/text`` cache
* hydra-config style (LIBERO / RoboTwin): ``config.yaml`` with a ``model:``
  section and ``dataset_stats.json``

The bundle produced here is what ``lerobot_infer`` consumes:

    <output>/
      bundle.json
      model_config.json
      normalization.json
      weights/<checkpoint>.pt   (symlink or copy)
      text/                     (symlink or copy, optional)

Usage:
  python scripts/build_fastwam_bundle.py --run-dir RUN --output OUT \
      --image-size 256 192 --text-dir PATH --gripper-indices 6 13
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import yaml

TARGET = "fastwam.runtime.create_action_vggt_lang_recon_fastwam"
DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following "
    "instruction: {task}"
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _latest_weight(run_dir: Path) -> Path:
    weights = run_dir / "checkpoints" / "weights"
    files = sorted(weights.glob("step_*.pt"))
    if not files:
        raise SystemExit(f"no checkpoints under {weights}")
    return files[-1]


def _model_config(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    contract_path = run_dir / "contract.json"
    if contract_path.is_file():
        contract = _read_json(contract_path)
        return dict(contract["model_config"]), contract

    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        raise SystemExit(f"neither contract.json nor config.yaml found in {run_dir}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = raw.get("model")
    if not isinstance(model, dict):
        raise SystemExit("config.yaml has no `model:` section")
    config = dict(model)
    config["_target_"] = TARGET
    config.setdefault("skip_vggt_load_from_pretrain", True)
    config.setdefault("aggregator_checkpoint_path", None)
    config.setdefault("vggt_head_checkpoint_path", None)
    return config, raw


def _normalization(run_dir: Path, model_config: dict[str, Any]) -> dict[str, Any]:
    norm_path = run_dir / "normalization.json"
    if norm_path.is_file():
        return _read_json(norm_path)

    stats_path = run_dir / "dataset_stats.json"
    if not stats_path.is_file():
        raise SystemExit(
            "no normalization.json or dataset_stats.json; pass a bundle with normalization"
        )
    stats = _read_json(stats_path)
    proprio_dim = int(model_config.get("proprio_dim") or 0)
    action_dim = int((model_config.get("action_dit_config") or {}).get("action_dim") or 0)
    result: dict[str, Any] = {}
    for name, dim in (("state", proprio_dim), ("action", action_dim)):
        section = (stats.get(name) or {}).get("default") or {}
        mean = section.get("global_mean")
        std = section.get("global_std")
        if not mean or not std or len(mean) != dim:
            raise SystemExit(
                f"dataset_stats.json lacks usable {name}.default.global_mean/std "
                f"for dim {dim}"
            )
        result[name] = {"mean": list(mean), "std": list(std)}
    return result


def _action_horizon(raw: dict[str, Any], contract: dict[str, Any]) -> int:
    if contract:
        target = (contract.get("objective") or {}).get("target_contract") or {}
        if target.get("horizon"):
            return int(target["horizon"])
    data = (raw.get("data") or {}).get("train") or {}
    if data.get("action_horizon"):
        return int(data["action_horizon"])
    return 32


def _dataset_image_size(raw: dict[str, Any]) -> tuple[int, int] | None:
    data = (raw.get("data") or {}).get("train") or {}
    images = ((data.get("shape_meta") or {}).get("images")) or []
    if not images:
        return None
    shape = images[0].get("shape")
    if not shape or len(shape) != 3:
        return None
    return int(shape[2]), int(shape[1])


def _link_or_copy(source: Path, target: Path, copy: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    if copy:
        shutil.copy2(source, target)
    else:
        os.symlink(source.resolve(), target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--text-dir", type=Path, default=None)
    parser.add_argument("--image-size", type=int, nargs=2, default=None,
                        metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--gripper-indices", type=int, nargs="*", default=[])
    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT)
    parser.add_argument("--copy", action="store_true",
                        help="copy weights/text instead of symlinking")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    model_config, raw = _model_config(run_dir)
    contract = raw if "objective" in raw else {}

    camera_keys = list(model_config.get("vggt_camera_keys") or [])
    if not camera_keys:
        raise SystemExit("model config has no vggt_camera_keys")
    action_dim = int((model_config.get("action_dit_config") or {}).get("action_dim") or 0)
    proprio_dim = int(model_config.get("proprio_dim") or 0)
    if not action_dim or not proprio_dim:
        raise SystemExit("model config is missing action_dim/proprio_dim")

    image_size = tuple(args.image_size) if args.image_size else _dataset_image_size(raw)
    if image_size is None:
        raise SystemExit("cannot infer image size; pass --image-size WIDTH HEIGHT")

    horizon = args.action_horizon or _action_horizon(raw, contract)
    checkpoint = (args.checkpoint or _latest_weight(run_dir)).expanduser().resolve()

    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "model_config.json", model_config)
    _write_json(output / "normalization.json", _normalization(run_dir, model_config))
    _link_or_copy(checkpoint, output / "weights" / checkpoint.name, args.copy)

    text_dir = args.text_dir.expanduser().resolve() if args.text_dir else None
    if text_dir is None:
        candidate = (run_dir / "data" / "text")
        if candidate.is_dir():
            text_dir = candidate
    if text_dir is not None:
        if not text_dir.is_dir():
            raise SystemExit(f"text dir does not exist: {text_dir}")
        target = output / "text"
        if target.exists() or target.is_symlink():
            target.unlink()
        if args.copy:
            shutil.copytree(text_dir, target)
        else:
            os.symlink(text_dir, target)

    manifest = {
        "schema_version": 1,
        "checkpoint": f"weights/{checkpoint.name}",
        "model_config": "model_config.json",
        "normalization": "normalization.json",
        "text_dir": "text" if text_dir is not None else None,
        "prompt_template": args.prompt_template,
        "camera_keys": camera_keys,
        "image_size": [int(image_size[0]), int(image_size[1])],
        "action_dim": action_dim,
        "proprio_dim": proprio_dim,
        "action_horizon": int(horizon),
        "num_inference_steps": int(args.num_inference_steps),
        "gripper_indices": list(args.gripper_indices),
        "seed": None,
    }
    _write_json(output / "bundle.json", manifest)
    print(json.dumps({"bundle": str(output), **manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
