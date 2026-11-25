"""Indexed raw metric RGB-D/calibration targets for VGGT reconstruction training.

Each entry is ``<root>/<global_frame:08d>.pt`` containing depth ``[V,H,W]`` in
meters, validity, intrinsics ``[V,3,3]`` and OpenCV camera-from-world
extrinsics ``[V,3,4]``.  Keeping this separate from the legacy patchified depth
cache prevents accidental log-depth normalization before geometry loss.
"""
from pathlib import Path
import torch


class VGGTReconstructionTargetCache:
    def __init__(self, root, camera_keys=None, allow_missing=False):
        self.root = Path(root).expanduser() if root else None
        self.camera_keys = tuple(camera_keys or ())
        self.allow_missing = bool(allow_missing)

    def get(self, index: int):
        path = None if self.root is None else self.root / f"{int(index):08d}.pt"
        if path is not None and path.is_file():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            required = {"depth", "intrinsics", "extrinsics"}
            if not required.issubset(payload):
                raise ValueError(f"Reconstruction target {path} lacks {required - set(payload)}")
            depth = payload["depth"].float()
            valid = payload.get("valid", depth > 0)
            return {"depth": depth, "valid": valid.bool(), "intrinsics": payload["intrinsics"].float(), "extrinsics": payload["extrinsics"].float()}
        if not self.allow_missing:
            raise FileNotFoundError(f"Missing reconstruction target: {path}")
        # Debug-only fallback keeps model construction/smoke tests runnable.
        v = max(len(self.camera_keys), 1)
        depth = torch.ones(v, 240, 320)
        k = torch.eye(3).repeat(v, 1, 1)
        k[:, 0, 0] = k[:, 1, 1] = 256.0
        k[:, 0, 2], k[:, 1, 2] = 160.0, 120.0
        e = torch.zeros(v, 3, 4); e[:, :, :3] = torch.eye(3)
        return {"depth": depth, "valid": torch.ones_like(depth, dtype=torch.bool), "intrinsics": k, "extrinsics": e}

    def get_pair(self, index: int, *, future_index: int, future_is_pad: bool = False):
        """Read current and endpoint targets without mixing episode boundaries.

        ``index`` and ``future_index`` are global dataset-frame indices.  The
        dataset computes ``future_is_pad`` from its episode index, so this
        cache remains a deliberately simple indexed file store.
        """
        current = self.get(int(index))
        future = self.get(int(future_index))
        return {
            "current": current,
            "future": future,
            "future_is_pad": bool(future_is_pad),
        }
