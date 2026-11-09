#!/usr/bin/env bash
# Build unified lerobot CLI (convert / infer / infer-once) into a PyInstaller onedir binary.
# 对齐 act-local-trainer/scripts/build_pyinstaller.sh。
#
# Usage:
#   bash scripts/setup.sh
#   bash scripts/build.sh
#   bash scripts/build.sh --clean
#
# Optional:
#   EXTRA_PYINSTALLER_ARGS='--log-level DEBUG' bash scripts/build.sh
#
# 产物：
#   dist/lerobot_infer/lerobot_infer
#   bin/lerobot_infer/                 # 拷贝，供 dataflow path 引用
# 用法示例：
#   bin/lerobot_infer/lerobot_infer --help
#   bin/lerobot_infer/lerobot_infer convert --config …
#   bin/lerobot_infer/lerobot_infer infer --config …

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

CLEAN=0
for arg in "$@"; do
  case "${arg}" in
    --clean) CLEAN=1 ;;
    -h|--help)
      sed -n '2,18p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: ${arg}" >&2
      exit 1
      ;;
  esac
done

PYTHON="${ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: 未找到 .venv，请先执行: bash scripts/setup.sh" >&2
  exit 1
fi

export PATH="${ROOT}/.venv/bin:${PATH}"

echo "==> project: ${ROOT}"
echo "==> python:  ${PYTHON}"
"${PYTHON}" -V

echo "==> Reinstalling current project source into .venv ..."
# Hatch 的 force-include 会将根目录模块安装为 ``lerobot_inference``。
# PyInstaller 解析 import 时读取这个已安装包，故必须在构建前重装，
# 避免把上次 uv sync 的旧源码打入新二进制。
uv pip install --python "${PYTHON}" --no-deps --reinstall "${ROOT}"

echo "==> Checking torch / triton ..."
"${PYTHON}" - <<'PY'
import importlib.util
import sys

missing = [name for name in ("torch", "torchvision", "triton") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("missing packages: " + ", ".join(missing) + "\n请先 bash scripts/setup.sh")

import torch
print(f"==> torch: {torch.__version__}")
PY

echo "==> Ensuring PyInstaller ..."
uv pip install --python "${PYTHON}" "pyinstaller>=6.0.0"
echo "==> PyInstaller: $("${PYTHON}" -m PyInstaller --version)"

RTHOOK_COMPILE="${SCRIPT_DIR}/pyi_rth_torch_compile.py"
RTHOOK_NVIDIA="${SCRIPT_DIR}/pyi_rth_nvidia_preload.py"
SPEC="${SCRIPT_DIR}/lerobot_infer.spec"
for need in "${RTHOOK_COMPILE}" "${RTHOOK_NVIDIA}" "${SPEC}" "${SCRIPT_DIR}/_pyi_entry_infer.py"; do
  if [[ ! -f "${need}" ]]; then
    echo "error: missing file: ${need}" >&2
    exit 1
  fi
done

# Prefer venv libraries while PyInstaller resolves shared-object dependencies.
VENV_LIB="$("${PYTHON}" -c "import os, sys; print(os.path.join(sys.prefix, 'lib'))")"
export LD_LIBRARY_PATH="${VENV_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
echo "==> build LD_LIBRARY_PATH: ${VENV_LIB}:..."

_npp_so="$(find "${VENV_LIB}" -name "libnppicc.so.*" 2>/dev/null | head -1)"
if [[ -z "${_npp_so}" ]]; then
  _npp_so="$(find "${ROOT}/.venv" -name "libnppicc.so.*" 2>/dev/null | head -1)"
fi
if [[ -n "${_npp_so}" ]]; then
  export LD_PRELOAD="${_npp_so}${LD_PRELOAD:+:${LD_PRELOAD}}"
  echo "==> LD_PRELOAD (libnppicc): ${_npp_so}"
fi

if [[ "${CLEAN}" -eq 1 ]]; then
  echo "==> Cleaning previous build artifacts"
  rm -rf "${ROOT}/dist/lerobot_infer" "${ROOT}/build/pyinstaller/lerobot_infer"
fi

mkdir -p "${ROOT}/dist" "${ROOT}/build/pyinstaller/lerobot_infer" "${ROOT}/bin"

PYINSTALLER_ARGS=(
  --noconfirm
  --clean
  --distpath "${ROOT}/dist"
  --workpath "${ROOT}/build/pyinstaller/lerobot_infer"
)

# Force known conflict-prone shared libraries from the active venv into bundle root.
_CONFLICT_LIBS=(
  libexpat.so.1
  libcrypto.so.3
  libssl.so.3
  libz.so.1
)
for _lib in "${_CONFLICT_LIBS[@]}"; do
  if [[ -e "${VENV_LIB}/${_lib}" ]]; then
    echo "==> bundling ${_lib}: $(readlink -f "${VENV_LIB}/${_lib}")"
    PYINSTALLER_ARGS+=(--add-binary "${VENV_LIB}/${_lib}:.")
  fi
done

if [[ -n "${EXTRA_PYINSTALLER_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  _extra_args=(${EXTRA_PYINSTALLER_ARGS})
  PYINSTALLER_ARGS+=("${_extra_args[@]}")
fi

echo "==> Building lerobot_infer ..."
"${PYTHON}" -m PyInstaller "${PYINSTALLER_ARGS[@]}" "${SPEC}"

DIST_DIR="${ROOT}/dist/lerobot_infer"
DIST_BIN="${DIST_DIR}/lerobot_infer"

# PyInstaller can leave links to libraries it did not collect (e.g. _internal/libcudart.so ->
# libcudart.so.12, whose real file lives under nvidia/cuda_runtime/lib). Nothing loads them,
# but the PAOS node installer rejects any dangling link in a directory_tar_gz archive.
while IFS= read -r -d '' _link; do
  echo "==> removing dangling symlink: ${_link#"${DIST_DIR}/"} -> $(readlink "${_link}")"
  rm -f "${_link}"
done < <(find "${DIST_DIR}" -xtype l -print0)

if [[ ! -x "${DIST_BIN}" ]]; then
  echo "ERROR: 构建失败，未找到 ${DIST_BIN}" >&2
  exit 1
fi

BIN_DIR="${ROOT}/bin/lerobot_infer"
rm -rf "${BIN_DIR}"
cp -a "${DIST_DIR}" "${BIN_DIR}"
BIN_BIN="${BIN_DIR}/lerobot_infer"
if [[ ! -x "${BIN_BIN}" ]]; then
  echo "ERROR: 复制失败，未找到 ${BIN_BIN}" >&2
  exit 1
fi

echo
echo "Done."
echo "  dist: ${DIST_BIN}"
echo "  bin:  ${BIN_BIN}"
echo
echo "统一入口用法："
echo "  ${DIST_BIN} --help"
echo "  ${DIST_BIN} convert --config ./convert.yaml"
echo "  ${DIST_BIN} infer --config ./policy_act.yaml"
echo "  ${DIST_BIN} infer-once --config ./policy_act.yaml"
echo
echo "dataflow.yaml 示例："
echo "  path: ../../bin/lerobot_infer/lerobot_infer"
echo "  args: infer --config ./policy_act.yaml"
