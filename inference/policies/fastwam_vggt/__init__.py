"""FastWAM VGGT-Omega policy adapter (E05 family).

FastWAM here is the research codebase (`fastwam` package) that builds a
VGGT-Omega vision backbone plus a flow-matching Action DiT. It is *not* a
LeRobot `PreTrainedPolicy`, so this adapter talks to it directly and keeps the
LeRobot processor pipeline out of the loop.

Everything benchmark-specific (camera names, image size, action/proprio
dimensions, horizon, normalization, text cache) lives in the bundle manifest,
so the same node serves RoboDojo, RoboTwin and LIBERO checkpoints.
"""

from .adapter import FastWAMVGGTPolicyAdapter
from .bundle import FastWAMBundledConfig, FastWAMBundledConfigError, load_bundle

__all__ = [
    "FastWAMBundledConfig",
    "FastWAMBundledConfigError",
    "FastWAMVGGTPolicyAdapter",
    "load_bundle",
]
