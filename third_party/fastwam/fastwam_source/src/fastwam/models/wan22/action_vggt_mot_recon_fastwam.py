from __future__ import annotations

from pathlib import Path
import torch
import torch.nn.functional as F

from .action_vggt_mot_fastwam import ActionVGGTMoTFastWAM
from .vggt_reconstruction_loss import compute_vggt_reconstruction_loss, canonical_reconstruction_config


class ActionVGGTMoTReconFastWAM(ActionVGGTMoTFastWAM):
    """Action MoT with dual-time geometry, configurable sharing and va2 language."""

    CHECKPOINT_FORMAT = "action_vggt_mot_recon_fastwam"
    CHECKPOINT_VERSION = 2
    COMPUTATION_GRAPH = "frame_then_inter_frame_v2"
    evaluation_mode = "action_only"
    # Deployment uses this capability flag before adding model-specific kwargs
    # to infer_action(); legacy action-only models must keep their old call ABI.
    supports_language_context_mask = True

    def __init__(self, *args, reconstruction_loss_config=None, lambda_vggt_recon=0.05,
                 share_bridge=True, **kwargs):
        self.share_bridge = bool(share_bridge)
        if not self.share_bridge:
            from .vggt_stream_bridge import VGGTStreamBridge
            kwargs["mot"].vggt_bridge = VGGTStreamBridge(kwargs["mot"].vggt_bridge)
        super().__init__(*args, **kwargs)
        self.reconstruction_loss_config = canonical_reconstruction_config(reconstruction_loss_config or {})
        self.lambda_vggt_recon = float(lambda_vggt_recon)
        # VGGT geometry heads consume native FP32 aggregated features.  The
        # ActionDiT/backbone path may be BF16, so keep both shared heads FP32.
        self.vggt_backbone.camera_head.float()
        self.vggt_backbone.dense_head.float()

    def _recon_loss(self, prediction, sample, prefix, valid_mask=None):
        """Align branch views and valid batch rows before geometry supervision.

        Predictions [B,Vi,H,W,(1)] use routed cameras, whereas dataset labels
        [B,V,...] use input_camera_keys. Select V -> Vi, then discard padded
        batch rows for all depth/camera/point terms. Empty predictions retain
        a finite zero-gradient connection to every decoder (including confidence).
        """
        target = f"vggt_recon_{prefix}_"
        pred = {suffix: prediction[f"{prefix}_{suffix}"] for suffix in ("depth", "depth_conf", "pose")}
        graph_zero = sum(value.reshape(-1)[:0].float().sum() for value in pred.values())
        if valid_mask is not None and not bool(valid_mask.any()):
            return graph_zero, {f"loss_vggt_{prefix}_{term}": graph_zero.detach()
                                for term in ("camera", "depth", "point")} | {
                                    f"vggt_{prefix}_valid_depth_ratio": graph_zero.detach(),
                                    f"vggt_{prefix}_scene_scale": graph_zero.detach()}
        indices = (self.vggt_backbone.va1_view_indices if prefix == "current"
                   else self.vggt_backbone.va2_view_indices)
        labels = {suffix: sample[f"{target}{suffix}"].to(self.device)[:, list(indices)]
                  for suffix in ("depth", "depth_valid", "intrinsics", "extrinsics")}
        if valid_mask is not None:
            keep = valid_mask.to(self.device, dtype=torch.bool)
            pred = {key: value[keep] for key, value in pred.items()}
            labels = {key: value[keep] for key, value in labels.items()}
        loss, metrics = compute_vggt_reconstruction_loss(
            depth=pred["depth"], depth_conf=pred["depth_conf"], pose_enc=pred["pose"],
            gt_depth=labels["depth"], gt_valid=labels["depth_valid"],
            gt_intrinsics=labels["intrinsics"], gt_extrinsics=labels["extrinsics"],
            **self._reconstruction_loss_kwargs(sample, valid_mask))
        renamed = {}
        for key, value in metrics.items():
            if key.startswith("loss_vggt_"):
                key = key.replace("loss_vggt_", f"loss_vggt_{prefix}_", 1)
            elif key.startswith("vggt_"):
                key = key.replace("vggt_", f"vggt_{prefix}_", 1)
            renamed[key] = value
        return loss + graph_zero, renamed

    def training_loss(self, sample, vggt_kv_by_layer=None):
        if vggt_kv_by_layer is not None:
            raise ValueError("Reconstruction training requires live language-conditioned VGGT capture.")
        required = {"action", "context", "context_mask", "vggt_images"}
        if self.vggt_backbone.va2_use_language_condition:
            required.add("language_context_mask")
        required.update({f"vggt_recon_{p}_{s}" for p in ("current", "future") for s in ("depth", "depth_valid", "intrinsics", "extrinsics")})
        missing = sorted(required - set(sample))
        if missing:
            raise ValueError(f"ActionVGGTMoTReconFastWAM sample is missing {missing}.")
        action = sample["action"].to(self.device, self.torch_dtype)
        batch_size, action_horizon, _ = action.shape
        context, context_mask = self._prepare_context(sample["context"], sample["context_mask"], sample.get("proprio"), batch_size=batch_size, training=True)
        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(self.device, torch.bool)
        language_mask = sample.get("language_context_mask")
        reconstruction = self.vggt_backbone.forward_with_reconstruction(
            sample["vggt_images"].to(self.device, self.torch_dtype),
            context=sample["context"].to(self.device),
            language_context_mask=None if language_mask is None else language_mask.to(self.device),
        )
        noise = torch.randn_like(action)
        timestep = self.train_action_scheduler.sample_training_t(batch_size=batch_size, device=self.device, dtype=action.dtype)
        noisy_action = self.train_action_scheduler.add_noise(action, noise, timestep)
        target_action = self.train_action_scheduler.training_target(action, noise, timestep)
        pred_action = self._predict_action(noisy_action, timestep, context, context_mask, reconstruction["kv_by_layer"])
        token_loss = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2)
        if action_is_pad is None:
            per_sample = token_loss.mean(dim=1)
        else:
            valid = (~action_is_pad).to(token_loss.dtype)
            per_sample = (token_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        weight = self.train_action_scheduler.training_weight(timestep).to(per_sample)
        loss_action = (per_sample * weight).mean()
        loss_current, current_metrics = self._recon_loss(reconstruction, sample, "current")
        future_valid = ~sample.get("vggt_recon_future_is_pad", torch.zeros(batch_size, dtype=torch.bool)).to(self.device)
        loss_future, future_metrics = self._recon_loss(reconstruction, sample, "future", future_valid)
        recon = self.lambda_vggt_recon * 0.5 * (loss_current + loss_future)
        return loss_action + recon, {"loss_action": loss_action.detach(), "loss_vggt_recon": recon.detach(), **current_metrics, **future_metrics, "vggt_future_valid_ratio": future_valid.float().mean().detach()}

    @torch.no_grad()
    def infer_action(self, *, action_horizon, vggt_images, context, context_mask, language_context_mask=None, proprio=None, num_inference_steps=20, sigma_shift=None, seed=None, rand_device="cpu", return_reconstruction=False):
        self.eval()
        batch_size = 1 if context.ndim == 2 else int(context.shape[0])
        context, context_mask = self._prepare_context(context, context_mask, proprio, batch_size=batch_size, training=False)
        if language_context_mask is None:
            language_context_mask = torch.ones((batch_size, context.shape[1] - (1 if self.proprio_encoder is not None else 0)), dtype=torch.bool)
        elif language_context_mask.ndim == 1:
            language_context_mask = language_context_mask.unsqueeze(0)
        if language_context_mask.shape[0] != batch_size:
            raise ValueError("`language_context_mask` batch dimension must match context.")
        reconstruction = self.vggt_backbone.forward_with_reconstruction(
            vggt_images.to(self.device, self.torch_dtype),
            context=context[:, :language_context_mask.shape[1]],
            language_context_mask=language_context_mask.to(self.device),
        )
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action = torch.randn((batch_size, action_horizon, self.action_expert.action_dim), generator=generator, device=rand_device, dtype=torch.float32).to(self.device, self.torch_dtype)
        timesteps, deltas = self.infer_action_scheduler.build_inference_schedule(num_inference_steps=num_inference_steps, device=self.device, dtype=action.dtype, shift_override=sigma_shift)
        for timestep, delta in zip(timesteps, deltas):
            action = self.infer_action_scheduler.step(self._predict_action(action, timestep.expand(batch_size), context, context_mask, reconstruction["kv_by_layer"]), delta, action)
        action = action.detach().cpu().float()
        if not return_reconstruction:
            return action
        return {
            "action": action,
            "reconstruction": {
                # Public deploy names are stable even though the backbone uses
                # explicit prediction-head suffixes internally.
                "current": {
                    **self._reconstruction_output_metadata(),
                    "depth": reconstruction["current_depth"].detach().cpu().float(),
                    "confidence": reconstruction["current_depth_conf"].detach().cpu().float(),
                    "pose": reconstruction["current_pose"].detach().cpu().float(),
                },
                "future": {
                    **self._reconstruction_output_metadata(),
                    "depth": reconstruction["future_depth"].detach().cpu().float(),
                    "confidence": reconstruction["future_depth_conf"].detach().cpu().float(),
                    "pose": reconstruction["future_pose"].detach().cpu().float(),
                },
            },
        }

    def sharing_config(self):
        return {
            "share_patch_embedding": self.vggt_backbone.share_patch_embedding,
            "share_camera_register_tokens": self.vggt_backbone.share_camera_register_tokens,
            "share_geometry_heads": self.vggt_backbone.share_geometry_heads,
            "share_bridge": self.share_bridge,
        }

    def checkpoint_metadata(self):
        """JSON-compatible architecture contract for weights and trainer state."""
        metadata = self._architecture_metadata()
        metadata["layer_pairs"] = [list(pair) for pair in metadata["layer_pairs"]]
        for key in ("bridge_target_experts", "vggt_camera_keys", "va1_camera_keys", "va2_camera_keys"):
            metadata[key] = list(metadata[key])
        bridge = self.mot.vggt_bridge
        metadata.update(
            format=self.CHECKPOINT_FORMAT, version=self.CHECKPOINT_VERSION,
            computation_graph=self.COMPUTATION_GRAPH, sharing=self.sharing_config(),
            va2_use_language_condition=self.vggt_backbone.va2_use_language_condition,
            action_vggt_source=self.vggt_backbone.action_vggt_source,
            action_dim=self.action_expert.action_dim, text_dim=self.text_dim,
            proprio_dim=self.proprio_dim,
            bridge_widths=[bridge.fastwam_width, bridge.vggt_width],
            bridge_heads=[bridge.fastwam_num_heads, bridge.vggt_num_heads],
        )
        if self.reconstruction_loss_contract() is not None:
            metadata["reconstruction_loss"] = self.reconstruction_loss_contract()
        return metadata

    def validate_training_state_metadata(self, metadata, world_size):
        if metadata.get("model_architecture") != self.checkpoint_metadata():
            raise ValueError("MoTRecon training state must match corrected version 2 architecture; "
                             "old states used an incorrect graph and require retraining.")
        if metadata.get("world_size") != world_size:
            raise ValueError("MoTRecon full-state resume requires identical world size.")

    def save_checkpoint(self, path, optimizer=None, step=None):
        if optimizer is not None:
            raise ValueError("Weight checkpoints do not contain optimizer state.")
        torch.save({
            "format": self.CHECKPOINT_FORMAT, "version": self.CHECKPOINT_VERSION,
            "metadata": self.checkpoint_metadata(),
            "action_expert": self.action_expert.state_dict(),
            "vggt_bridge": self.mot.vggt_bridge.state_dict(),
            "proprio_encoder": None if self.proprio_encoder is None else self.proprio_encoder.state_dict(),
            "vggt_backbone": self.vggt_backbone.state_dict(), "step": step,
            "torch_dtype": str(self.torch_dtype),
        }, path)

    def load_checkpoint(self, path, optimizer=None):
        if optimizer is not None:
            raise ValueError("Weight checkpoints do not contain optimizer state.")
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if (not isinstance(payload, dict) or payload.get("format") != self.CHECKPOINT_FORMAT
                or payload.get("version") != self.CHECKPOINT_VERSION):
            raise ValueError("MoTRecon requires version 2 checkpoints. Version 1 used an incorrect "
                             "frame/inter-frame graph; retrain from original VGGT pretrained weights.")
        if payload.get("metadata") != self.checkpoint_metadata():
            raise ValueError("MoTRecon checkpoint architecture/sharing metadata mismatch.")
        if (payload.get("proprio_encoder") is None) != (self.proprio_encoder is None):
            raise ValueError("MoTRecon checkpoint proprio configuration mismatch.")
        self.action_expert.load_state_dict(payload["action_expert"], strict=True)
        self.mot.vggt_bridge.load_state_dict(payload["vggt_bridge"], strict=True)
        self.vggt_backbone.load_state_dict(payload["vggt_backbone"], strict=True)
        if self.proprio_encoder is not None:
            self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
        return payload


__all__ = ["ActionVGGTMoTReconFastWAM"]
