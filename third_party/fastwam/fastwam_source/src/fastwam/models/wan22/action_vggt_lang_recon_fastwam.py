from __future__ import annotations

"""ActionDiT with a language-conditioned, future-geometry VGGT memory."""

from pathlib import Path

import torch
import torch.nn.functional as F

from .action_vggt_fastwam import ActionVGGTFastWAM
from .vggt_reconstruction_loss import compute_vggt_reconstruction_loss, canonical_reconstruction_config


class ActionVGGTLangReconFastWAM(ActionVGGTFastWAM):
    """Single-stream ActionVGGT model for language-conditioned future geometry.

    The action branch still receives the normal ActionDiT context (including
    the optional proprio token).  The VGGT branch receives only the original
    text context and its padding mask, so proprio is never accidentally
    treated as a language token.
    """

    CHECKPOINT_FORMAT = "action_vggt_lang_recon_fastwam"
    CHECKPOINT_VERSION = 1
    evaluation_mode = "action_only"
    supports_language_context_mask = True
    requires_vggt_future_images = False

    def __init__(self, *args, vggt_camera_keys=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.vggt_camera_keys = tuple(vggt_camera_keys)
        if self.reconstruction_loss_config is None:
            raise ValueError(
                "ActionVGGTLangReconFastWAM requires reconstruction_loss_config."
            )
        if not hasattr(self.vggt_backbone, "language_projector"):
            raise TypeError(
                "ActionVGGTLangReconFastWAM requires a language-conditioned VGGT backbone."
            )

    def _future_recon_loss(self, prediction, sample, valid_mask=None):
        """Compute future depth/camera/point loss, excluding padded futures."""
        # Empty reductions are finite zero even if excluded rows contain NaN.
        # Keep every head in the graph on every rank, including all-pad ranks
        # and the confidence branch when confidence loss is disabled. Otherwise
        # ZeRO may launch different gradient collectives across ranks.
        graph_zero = sum(
            prediction[key].reshape(-1)[:0].float().sum()
            for key in ("future_depth", "future_depth_conf", "future_pose")
        )
        if valid_mask is not None:
            keep = valid_mask.to(device=self.device, dtype=torch.bool)
            if keep.shape != prediction["future_depth"].shape[:1]:
                raise ValueError("`vggt_recon_future_is_pad` must have shape [B].")
            if not bool(keep.any()):
                zero = graph_zero
                return zero, {
                    "loss_vggt_future_camera": zero.detach(),
                    "loss_vggt_future_depth": zero.detach(),
                    "loss_vggt_future_point": zero.detach(),
                    "vggt_future_valid_depth_ratio": zero.detach(),
                    "vggt_future_scene_scale": zero.detach(),
                }
            prediction = {
                key: prediction[key][keep]
                for key in ("future_depth", "future_depth_conf", "future_pose")
            }
            sample = {
                key: value[keep.to(value.device)]
                if isinstance(value, torch.Tensor)
                and value.ndim > 0
                and value.shape[0] == keep.shape[0]
                else value
                for key, value in sample.items()
            }

        loss, metrics = compute_vggt_reconstruction_loss(
            depth=prediction["future_depth"],
            depth_conf=prediction["future_depth_conf"],
            pose_enc=prediction["future_pose"],
            gt_depth=sample["vggt_recon_future_depth"].to(self.device),
            gt_valid=sample["vggt_recon_future_depth_valid"].to(self.device),
            gt_intrinsics=sample["vggt_recon_future_intrinsics"].to(self.device),
            gt_extrinsics=sample["vggt_recon_future_extrinsics"].to(self.device),
            **self._reconstruction_loss_kwargs(sample),
        )
        renamed = {}
        for key, value in metrics.items():
            if key.startswith("loss_vggt_"):
                key = key.replace("loss_vggt_", "loss_vggt_future_", 1)
            elif key.startswith("vggt_"):
                key = key.replace("vggt_", "vggt_future_", 1)
            renamed[key] = value
        return loss + graph_zero, renamed

    def training_loss(self, sample, vggt_kv_by_layer=None):
        """Train action flow and future reconstruction from current RGB.

        Tensor flow is ``current RGB + text -> future VGGT features``.  The
        future RGB/depth is never passed through the model; only its cached
        geometry labels are consumed by ``_future_recon_loss``.
        """
        if vggt_kv_by_layer is not None:
            raise ValueError(
                "Language-conditioned reconstruction requires live VGGT capture."
            )
        required = {
            "action",
            "context",
            "context_mask",
            "language_context_mask",
            "vggt_images",
            "vggt_recon_future_depth",
            "vggt_recon_future_depth_valid",
            "vggt_recon_future_intrinsics",
            "vggt_recon_future_extrinsics",
        }
        missing = sorted(required - set(sample))
        if missing:
            raise ValueError(
                f"ActionVGGTLangReconFastWAM sample is missing {missing}."
            )

        action = sample["action"].to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        if action.ndim != 3 or action.shape[2] != self.action_expert.action_dim:
            raise ValueError(
                "`action` must be [B,T,action_dim], got "
                f"{tuple(action.shape)} with action_dim={self.action_expert.action_dim}."
            )
        batch_size, action_horizon, _ = action.shape
        context, context_mask = self._prepare_context(
            sample["context"],
            sample["context_mask"],
            sample.get("proprio"),
            batch_size=batch_size,
            training=True,
        )
        language_context = sample["context"].to(self.device)
        language_context_mask = sample["language_context_mask"].to(
            self.device, dtype=torch.bool
        )

        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None:
            if action_is_pad.shape != (batch_size, action_horizon):
                raise ValueError("`action_is_pad` must have shape [B,T].")
            action_is_pad = action_is_pad.to(self.device, dtype=torch.bool)

        reconstruction = self.vggt_backbone.forward_with_reconstruction(
            sample["vggt_images"].to(self.device, self.torch_dtype),
            context=language_context,
            language_context_mask=language_context_mask,
        )
        noise = torch.randn_like(action)
        timestep = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise, timestep)
        target_action = self.train_action_scheduler.training_target(action, noise, timestep)
        pred_action = self._predict_action(
            noisy_action,
            timestep,
            context,
            context_mask,
            reconstruction["kv_by_layer"],
        )

        token_loss = F.mse_loss(
            pred_action.float(), target_action.float(), reduction="none"
        ).mean(dim=2)
        if action_is_pad is None:
            per_sample = token_loss.mean(dim=1)
        else:
            valid_action = (~action_is_pad).to(token_loss.dtype)
            per_sample = (token_loss * valid_action).sum(dim=1) / valid_action.sum(
                dim=1
            ).clamp_min(1.0)
        weight = self.train_action_scheduler.training_weight(timestep).to(per_sample)
        loss_action = (per_sample * weight).mean()

        future_is_pad = sample.get(
            "vggt_recon_future_is_pad",
            torch.zeros(batch_size, dtype=torch.bool),
        ).to(self.device, dtype=torch.bool)
        loss_future_raw, future_metrics = self._future_recon_loss(
            reconstruction,
            sample,
            valid_mask=~future_is_pad,
        )
        loss_future = self.lambda_vggt_recon * loss_future_raw
        total = loss_action + loss_future
        return total, {
            "loss_action": loss_action.detach(),
            "loss_vggt_recon": loss_future.detach(),
            **future_metrics,
            "vggt_future_valid_ratio": (~future_is_pad).float().mean().detach(),
        }

    @torch.no_grad()
    def infer_action(
        self,
        *,
        action_horizon,
        vggt_images,
        context,
        context_mask,
        language_context_mask=None,
        proprio=None,
        num_inference_steps=20,
        sigma_shift=None,
        seed=None,
        rand_device="cpu",
        return_reconstruction=False,
    ):
        """Run action inference using language-conditioned current observations."""
        self.eval()
        if action_horizon <= 0:
            raise ValueError("`action_horizon` must be positive.")
        # Preserve the caller's text padding before _prepare_context appends a
        # proprio token. Deployment always supplies the original language mask.
        if language_context_mask is None:
            language_context_mask = context_mask
        batch_size = 1 if context.ndim == 2 else int(context.shape[0])
        context, context_mask = self._prepare_context(
            context,
            context_mask,
            proprio,
            batch_size=batch_size,
            training=False,
        )
        if language_context_mask.ndim == 1:
            language_context_mask = language_context_mask.unsqueeze(0)
        language_context_mask = language_context_mask.to(self.device, dtype=torch.bool)
        language_length = int(language_context_mask.shape[1])
        if language_length > int(context.shape[1]):
            raise ValueError("`language_context_mask` is longer than `context`.")
        language_context = context[:, :language_length]

        images = vggt_images.to(self.device, self.torch_dtype)
        if return_reconstruction:
            reconstruction = self.vggt_backbone.forward_with_reconstruction(
                images,
                context=language_context,
                language_context_mask=language_context_mask,
            )
        else:
            reconstruction = None
            memory = self.vggt_backbone(
                images,
                context=language_context,
                language_context_mask=language_context_mask,
            )
        memory = reconstruction["kv_by_layer"] if reconstruction is not None else memory

        generator = (
            None
            if seed is None
            else torch.Generator(device=rand_device).manual_seed(seed)
        )
        action = torch.randn(
            (batch_size, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(self.device, self.torch_dtype)
        timesteps, deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=action.dtype,
            shift_override=sigma_shift,
        )
        for timestep, delta in zip(timesteps, deltas):
            action = self.infer_action_scheduler.step(
                self._predict_action(
                    action,
                    timestep.expand(batch_size),
                    context,
                    context_mask,
                    memory,
                ),
                delta,
                action,
            )
        action = action.detach().cpu().float()
        if reconstruction is None:
            return action
        return {
            "action": action,
            "reconstruction": {
                "future": {
                    **self._reconstruction_output_metadata(),
                    "depth": reconstruction["future_depth"].detach().cpu().float(),
                    "confidence": reconstruction["future_depth_conf"].detach().cpu().float(),
                    "pose": reconstruction["future_pose"].detach().cpu().float(),
                }
            },
        }

    def _checkpoint_metadata(self):
        bridge = self.mot.vggt_bridge
        return {
            # These fields make it impossible to silently load a future-only
            # language checkpoint into a differently configured model.
            "future_reconstruction": True,
            "requires_vggt_future_images": False,
            "supports_language_context_mask": True,
            "text_dim": int(self.text_dim),
            "vggt_camera_keys": self.vggt_camera_keys,
            "layer_pairs": tuple(
                (bridge.vggt_layer_for(i), i)
                for i in range(len(self.action_expert.blocks))
                if bridge.vggt_layer_for(i) is not None
            ),
            "train_vggt": bool(self.train_vggt),
            "lambda_vggt_recon": float(self.lambda_vggt_recon),
            "reconstruction_loss_config": dict(self.reconstruction_loss_config),
            **({"reconstruction_loss": self.reconstruction_loss_contract()}
               if self.reconstruction_loss_contract() is not None else {}),
            "bridge_config": {
                "projection_mode": bridge.projection_mode,
                "projection_sharing": bridge.projection_sharing,
                "fastwam_width": int(bridge.fastwam_width),
                "vggt_width": int(bridge.vggt_width),
                "fastwam_num_heads": int(bridge.fastwam_num_heads),
                "vggt_num_heads": int(bridge.vggt_num_heads),
            },
        }

    def save_checkpoint(self, path, optimizer=None, step=None):
        if optimizer is not None:
            raise ValueError("Weight checkpoints do not contain optimizer state.")
        payload = {
            "format": self.CHECKPOINT_FORMAT,
            "version": self.CHECKPOINT_VERSION,
            "target": "future",
            "text_dim": int(self.text_dim),
            "action_expert": self.action_expert.state_dict(),
            "vggt_bridge": self.mot.vggt_bridge.state_dict(),
            "vggt_backbone": self.vggt_backbone.state_dict(),
            "metadata": self._checkpoint_metadata(),
            "proprio_encoder": (
                None
                if self.proprio_encoder is None
                else self.proprio_encoder.state_dict()
            ),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        if optimizer is not None:
            raise ValueError("Weight checkpoints do not contain optimizer state.")
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        expected = {
            "format",
            "version",
            "target",
            "text_dim",
            "action_expert",
            "vggt_bridge",
            "vggt_backbone",
            "metadata",
            "proprio_encoder",
            "step",
            "torch_dtype",
        }
        missing = sorted(expected - set(payload))
        unexpected = sorted(set(payload) - expected)
        if missing or unexpected:
            raise ValueError(
                "ActionVGGTLangReconFastWAM checkpoint key mismatch: "
                f"missing={missing}, unexpected={unexpected}."
            )
        if (
            payload["format"] != self.CHECKPOINT_FORMAT
            or int(payload["version"]) != self.CHECKPOINT_VERSION
            or payload["target"] != "future"
            or int(payload["text_dim"]) != int(self.text_dim)
        ):
            raise ValueError("Unsupported or incompatible language reconstruction checkpoint.")
        if payload["metadata"] != self._checkpoint_metadata():
            raise ValueError(
                "ActionVGGTLangReconFastWAM checkpoint metadata mismatch."
            )
        self.action_expert.load_state_dict(payload["action_expert"], strict=True)
        self.mot.vggt_bridge.load_state_dict(payload["vggt_bridge"], strict=True)
        self.vggt_backbone.load_state_dict(payload["vggt_backbone"], strict=True)
        if self.proprio_encoder is None:
            if payload["proprio_encoder"] is not None:
                raise ValueError("Checkpoint contains unexpected proprio weights.")
        else:
            if payload["proprio_encoder"] is None:
                raise ValueError("Checkpoint is missing required proprio weights.")
            self.proprio_encoder.load_state_dict(
                payload["proprio_encoder"], strict=True
            )
        return payload


__all__ = ["ActionVGGTLangReconFastWAM"]
