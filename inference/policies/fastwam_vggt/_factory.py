"""Pinned copy of FastWAM's `create_action_vggt_lang_recon_fastwam` factory.

The upstream factory lives in `fastwam.runtime`, which imports the training
stack (`trainer` -> imageio, accelerate, video writers). A serving node only
needs the model graph, so this module re-implements the same constructor calls
against `fastwam.models.*` and keeps the training-only imports out of the
frozen binary.

Keep this file in sync with `runtime.py` of the vendored FastWAM revision
(`third_party/fastwam/fastwam_source/src/fastwam/runtime.py`).
"""

from __future__ import annotations

from typing import Any

SUPPORTED_TARGETS = frozenset(
    {
        "fastwam.runtime.create_action_vggt_lang_recon_fastwam",
    }
)


def create_action_vggt_lang_recon_fastwam(
    *,
    vggt_weights_path: str,
    layer_pairs: Any,
    projection_mode: str,
    action_dit_config: Any,
    action_scheduler: Any,
    projection_sharing: str = "shared",
    aggregator_checkpoint_path: str | None = None,
    action_dit_pretrained_path: str | None = None,
    skip_action_dit_load_from_pretrain: bool = False,
    skip_vggt_load_from_pretrain: bool = False,
    proprio_dim: int | None = None,
    mot_checkpoint_mixed_attn: bool = True,
    vggt_width: int = 1024,
    vggt_num_heads: int = 16,
    vggt_camera_keys: Any = (),
    model_dtype: Any = None,
    device: str = "cuda",
    train_vggt: bool = True,
    reconstruction_loss_config: Any = None,
    lambda_vggt_recon: float = 0.05,
    vggt_head_checkpoint_path: str | None = None,
    **_ignored: Any,
):
    import torch

    from fastwam.models.wan22.action_dit import ActionDiT
    from fastwam.models.wan22.action_vggt_lang_recon_fastwam import (
        ActionVGGTLangReconFastWAM,
    )
    from fastwam.models.wan22.mot import MoT
    from fastwam.models.wan22.vggt_bridge import VGGTSharedAttentionBridge
    from fastwam.models.wan22.vggt_lang_recon_backbone import (
        build_vggt_language_recon_backbone,
    )

    if model_dtype is None:
        model_dtype = torch.bfloat16
    if reconstruction_loss_config is None:
        raise ValueError(
            "ActionVGGTLangReconFastWAM requires reconstruction_loss_config."
        )
    if not isinstance(action_dit_config, dict) or not isinstance(action_scheduler, dict):
        raise ValueError("action_dit_config and action_scheduler must be dicts.")
    if "text_dim" not in action_dit_config:
        raise ValueError("`action_dit_config.text_dim` is required.")
    required_scheduler = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_scheduler = sorted(required_scheduler - set(action_scheduler))
    if missing_scheduler:
        raise ValueError(
            f"action_scheduler is missing required keys: {missing_scheduler}."
        )

    pairs = [tuple(int(value) for value in pair) for pair in layer_pairs]
    action = ActionDiT.from_pretrained(
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_action_dit_load_from_pretrain),
        device=device,
        torch_dtype=model_dtype,
    )
    bridge = VGGTSharedAttentionBridge(
        fastwam_width=int(action.num_heads * action.attn_head_dim),
        vggt_width=int(vggt_width),
        fastwam_num_heads=int(action.num_heads),
        vggt_num_heads=int(vggt_num_heads),
        vggt_num_layers=24,
        fastwam_num_layers=len(action.blocks),
        layer_pairs=pairs,
        target_experts=("action",),
        projection_mode=str(projection_mode),
        projection_sharing=str(projection_sharing),
        detach_vggt_memory=not bool(train_vggt),
    ).to(device=device, dtype=model_dtype)
    mot = MoT(
        mixtures={"action": action},
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        vggt_bridge=bridge,
    )
    backbone = build_vggt_language_recon_backbone(
        weights_path=vggt_weights_path,
        aggregator_checkpoint_path=aggregator_checkpoint_path,
        head_checkpoint_path=vggt_head_checkpoint_path,
        callback_layer_indices=tuple(pair[0] for pair in pairs),
        text_dim=int(action_dit_config["text_dim"]),
        trainable=bool(train_vggt),
        skip_vggt_load_from_pretrain=bool(skip_vggt_load_from_pretrain),
    )
    return ActionVGGTLangReconFastWAM(
        action_expert=action,
        mot=mot,
        vggt_backbone=backbone,
        text_dim=int(action_dit_config["text_dim"]),
        vggt_camera_keys=tuple(vggt_camera_keys),
        proprio_dim=None if proprio_dim is None else int(proprio_dim),
        device=device,
        torch_dtype=model_dtype,
        train_vggt=bool(train_vggt),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        reconstruction_loss_config=reconstruction_loss_config,
        lambda_vggt_recon=float(lambda_vggt_recon),
    )


__all__ = ["SUPPORTED_TARGETS", "create_action_vggt_lang_recon_fastwam"]
