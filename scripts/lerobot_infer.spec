# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec：lerobot_infer onedir，启用 torch.compile（对齐 act-local-trainer）。"""

from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

_spec_dir = Path(os.path.dirname(os.path.abspath(SPEC)))
_root_dir = _spec_dir.parent

datas: list = []
binaries: list = []
hiddenimports: list = []

for pkg in (
    "torch",
    "torchvision",
    "torchcodec",
    "triton",
    "transformers",
    # FastWAM / LingBot-VA build their Wan VAE via diffusers.AutoencoderKLWan;
    # lerobot gates that import on find_spec + dist metadata, so collect both.
    "diffusers",
    "lerobot",
    "safetensors",
    "einops",
    "cv2",
    "PIL",
    "yaml",
    "dora",
    "pyarrow",
):
    try:
        tmp = collect_all(pkg)
        datas += tmp[0]
        binaries += tmp[1]
        hiddenimports += tmp[2]
    except Exception:
        pass

# diffusers checks these dists via importlib.metadata at import time
# (dependency_versions_check.py); requests has no dist-info otherwise.
for pkg in ("requests", "filelock", "numpy"):
    datas += copy_metadata(pkg)

hiddenimports += collect_submodules("lerobot_inference")
hiddenimports += collect_submodules("forge_msgs")
hiddenimports += collect_submodules("forge_common")
hiddenimports += collect_submodules("forge_policy")
hiddenimports += collect_submodules("forge_tool")
hiddenimports += [
    "lerobot_inference.cli",
    "lerobot_inference.convert.cli",
    "lerobot_inference.inference.main",
    "lerobot_inference.inference.run_once",
    "lerobot_inference.inference.compile_utils",
    "lerobot_inference.inference.policies.act",
    "lerobot_inference.inference.policies.fastwam",
    "lerobot.policies.fastwam.modeling_fastwam",
    "diffusers.models.autoencoders.autoencoder_kl_wan",
    "lerobot_inference.inference.policies.pi05",
    "lerobot_inference.inference.policies.registry",
    "torch._dynamo",
    "torch._dynamo.backends.inductor",
    "torch._functorch",
    "torch._higher_order_ops",
    "torch._inductor",
    "torch._inductor.compile_fx",
    "torch._inductor.codecache",
    "torch.fx",
    "triton",
]

# Triton JIT needs Python C headers under include/pythonX.Y
_py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
_include = sysconfig.get_path("include") or ""
_platinclude = sysconfig.get_path("platinclude") or ""
_include_dest = f"include/{_py_ver}"
if _include and os.path.isdir(_include):
    datas.append((_include, _include_dest))
if _platinclude and os.path.isdir(_platinclude) and _platinclude != _include:
    datas.append((_platinclude, _include_dest))

runtime_hooks = [
    str(_spec_dir / "pyi_rth_nvidia_preload.py"),
    str(_spec_dir / "pyi_rth_torch_compile.py"),
]

a = Analysis(
    [str(_spec_dir / "_pyi_entry_infer.py")],
    pathex=[str(_root_dir)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=runtime_hooks,
    excludes=[
        "rerun_sdk",
        "IPython",
        "jupyter",
        "pytest",
    ],
    noarchive=False,
    optimize=0,
)
# CUDA libraries live once under nvidia/*/lib (torch RPATH and pyi_rth_nvidia_preload load them
# from there). Drop any top-level copy the dependency analysis adds for the same file name.
_nvidia_names = {
    os.path.basename(dest)
    for dest, _src, _kind in a.binaries
    if dest.replace(os.sep, "/").startswith("nvidia/")
}
a.binaries = [
    entry
    for entry in a.binaries
    if not (os.path.dirname(entry[0]) == "" and os.path.basename(entry[0]) in _nvidia_names)
]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="lerobot_infer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="lerobot_infer",
)
