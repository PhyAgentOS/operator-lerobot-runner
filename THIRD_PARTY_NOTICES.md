# Third-party notices

The repository's original Python code, configuration, static resources, and
documentation are licensed under Apache License 2.0 unless a file states
otherwise.

## Runtime dependencies

No third-party runtime component is vendored into this source repository; all
dependencies are installed from package indexes and retain their respective
licenses:

- `torch`, `torchvision`: BSD-3-Clause
- `triton`: MIT
- `lerobot`: Apache-2.0
- `transformers`: Apache-2.0
- `numpy`: BSD-3-Clause
- `einops`: MIT
- `safetensors`: Apache-2.0
- `PyYAML`: MIT
- `pillow`: HPND
- `opencv-python-headless`: Apache-2.0
- `huggingface-hub`: Apache-2.0
- `packaging`: Apache-2.0 / BSD-2-Clause
- `tqdm`: MPL-2.0
- `forge-msgs`, `forge-common`, `forge-policy`, `forge-tool`: Apache-2.0
- `dora-rs`: MIT

## Vendored source (not from a package index)

`third_party/fastwam/` contains the FastWAM research sources
(`fastwam_source/src/fastwam`) and the vendored VGGT-Omega package
(`fastwam_source/third_party/vggt-omega`) used by `type: fastwam_vggt`.
These files are copied verbatim from the FastWAM training repository; the
upstream LICENSE files were not part of the delivered tree, so redistribution
of a source or binary release must confirm the FastWAM and VGGT-Omega licenses
with their authors before publication. The optional FastWAM adapter can be
excluded from a distribution without affecting the other policies.

Transitive dependencies retain the licenses declared by their distributions.
Consult the locked environment and installed package metadata for the complete
dependency graph.

## Build dependencies and binary releases

PyInstaller is used only to produce optional standalone executables and is not
vendored into this repository; it is a build-only dependency licensed under
GPL-2.0-or-later with its special exception. A bundled executable contains
runtime dependencies (including NVIDIA user-space libraries collected from the
Torch wheels at build time) and may create license, notice, source-offer,
codec, or platform obligations beyond source distribution. Review the complete
bundled artifact and all dependency licenses before publishing a binary
release.
