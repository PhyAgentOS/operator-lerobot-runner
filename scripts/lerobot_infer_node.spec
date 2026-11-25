# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec：lerobot_infer 单文件（onefile）节点，启用 torch.compile（对齐 act-local-trainer）。

与 scripts/lerobot_infer.spec（默认 onedir 流程）的差异：
- 额外打包 requests / filelock / numpy 的 dist-info 元数据；
- collect_all 增加 diffusers；
- 增加 lerobot_inference.inference.policies.lingbot_va 隐藏导入；
- EXE 内联 binaries / datas 并省略 COLLECT，产出单文件可执行。

用途：需要以 PAOS executable_tar_gz 契约安装节点时，归档根目录只能包含一个名为
entrypoint（lerobot_infer）的普通文件，onedir 目录不满足该契约。
"""

from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

_spec_dir = Path(os.path.dirname(os.path.abspath(SPEC)))
_root_dir = _spec_dir.parent

_fastwam_vendor = _root_dir / "third_party" / "fastwam"
if _fastwam_vendor.is_dir():
    sys.path.insert(0, str(_fastwam_vendor / "fastwam_source" / "src"))
    sys.path.insert(
        0,
        str(
            _fastwam_vendor
            / "fastwam_source"
            / "third_party"
            / "vggt-omega"
        ),
    )

datas: list = []
binaries: list = []
hiddenimports: list = []
if _fastwam_vendor.is_dir():
    datas.append((str(_fastwam_vendor), "lerobot_inference/third_party/fastwam"))

for dist in ("requests", "filelock", "numpy"):
    try:
        datas += copy_metadata(dist)
    except Exception:
        pass


for pkg in (
    "torch",
    "torchvision",
    "triton",
    "transformers",
    "diffusers",
    "lerobot",
    "safetensors",
    "einops",
    "cv2",
    "PIL",
    "yaml",
    "dora",
    "pyarrow",
    "omegaconf",
    "imageio",
):
    try:
        tmp = collect_all(pkg)
        datas += tmp[0]
        binaries += tmp[1]
        hiddenimports += tmp[2]
    except Exception:
        pass

hiddenimports += collect_submodules("lerobot_inference")
hiddenimports += [
    "fastwam",
    "fastwam.models",
    "fastwam.models.wan22",
    "fastwam.models.wan22.action_dit",
    "fastwam.models.wan22.action_vggt_lang_recon_fastwam",
    "fastwam.models.wan22.mot",
    "fastwam.models.wan22.vggt_bridge",
    "fastwam.models.wan22.vggt_lang_recon_backbone",
    "fastwam.utils",
    "fastwam.utils.logging_config",
    "fastwam.utils.misc",
    "vggt_omega",
]
hiddenimports += [
    "lerobot_inference.inference.policies.fastwam_vggt",
    "lerobot_inference.inference.policies.fastwam_vggt.adapter",
    "lerobot_inference.inference.policies.fastwam_vggt.bundle",
    "lerobot_inference.inference.policies.fastwam_vggt.model",
    "lerobot_inference.inference.policies.fastwam_vggt.runner",
    "lerobot_inference.inference.policies.fastwam_vggt._factory",
]
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
    "lerobot_inference.inference.policies.pi05",
    "lerobot_inference.inference.policies.registry",
    "lerobot_inference.inference.policies.lingbot_va",
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

# Bundle NVIDIA wheel .so into root (skip libcuda — from host driver)
_site = Path(sysconfig.get_paths()["purelib"])
_nvidia = _site / "nvidia"
if _nvidia.is_dir():
    seen: set[str] = set()
    for so in _nvidia.glob("**/lib/*.so*"):
        if not so.is_file():
            continue
        name = so.name
        if name == "libcuda.so" or name.startswith("libcuda.so."):
            continue
        real = str(so.resolve())
        if real in seen:
            continue
        seen.add(real)
        binaries.append((str(so), "."))

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
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
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
