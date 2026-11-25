"""FastWAM synchronous inference adapter."""

from .adapter import FastWAMPolicyAdapter
from .assets import ensure_fastwam_offline_assets
from .loading import load_fastwam_policy_strict

__all__ = [
    "FastWAMPolicyAdapter",
    "ensure_fastwam_offline_assets",
    "load_fastwam_policy_strict",
]
