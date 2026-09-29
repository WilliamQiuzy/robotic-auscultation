#!/usr/bin/env bash
# Run through guarded_run.py; isolated user install, no sudo or shared-env writes.
set -euo pipefail
ROOT=/home/william/wam-sim/eval_curobo
CUROBO_COMMIT=2fbffc35225398cf9d5f382804faa9de2608753b
mkdir -p "$ROOT" "$ROOT/logs" "$ROOT/cache" "$ROOT/bin" "$ROOT/downloads"
export UV_CACHE_DIR="$ROOT/cache/uv"
export PIP_CACHE_DIR="$ROOT/cache/pip"
export MAMBA_ROOT_PREFIX="$ROOT/mamba"
export XDG_CACHE_HOME="$ROOT/cache"
export TMPDIR="$ROOT/tmp"
mkdir -p "$TMPDIR"
UV=/home/william/.local/bin/uv
if [ ! -x "$ROOT/bin/micromamba" ]; then
  curl -fsSL https://micro.mamba.pm/api/micromamba/linux-64/2.3.2 -o "$ROOT/micromamba-2.3.2.tar.bz2"
  tar -xjf "$ROOT/micromamba-2.3.2.tar.bz2" -C "$ROOT" bin/micromamba
fi
if [ ! -x "$ROOT/cuda/bin/nvcc" ]; then
  "$ROOT/bin/micromamba" create -y --no-rc -p "$ROOT/cuda" --override-channels \
    -c nvidia/label/cuda-12.4.1 cuda-nvcc=12.4.131 cuda-cudart-dev=12.4.127 cuda-cccl=12.4.127
fi
if [ ! -x "$ROOT/venv/bin/python" ]; then
  "$UV" venv --python /home/william/.local/bin/python3.10 --seed "$ROOT/venv"
fi
PY="$ROOT/venv/bin/python"
"$UV" pip install --python "$PY" torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
"$UV" pip install --python "$PY" numpy==1.26.4 warp-lang==1.8.1 setuptools==75.6.0 \
  setuptools-scm==8.1.0 wheel==0.45.1 ninja==1.11.1.3 pybind11==2.13.6 \
  scipy==1.14.1 networkx==3.4.2 numpy-quaternion==2024.0.7 pyyaml==6.0.2 \
  trimesh==4.5.3 yourdfpy==0.0.56 tqdm==4.67.1 importlib_resources==6.4.5 \
  scikit-image==0.24.0 pytest==8.3.4
if [ ! -d "$ROOT/curobo/.git" ]; then
  git clone --depth 1 --branch v0.7.6 https://github.com/NVlabs/curobo.git "$ROOT/curobo"
fi
test "$(git -C "$ROOT/curobo" rev-parse HEAD)" = "$CUROBO_COMMIT"
if [ ! -x "$ROOT/bin/git-lfs" ]; then
  curl -fsSL https://github.com/git-lfs/git-lfs/releases/download/v3.6.1/git-lfs-linux-amd64-v3.6.1.tar.gz \
    -o "$ROOT/downloads/git-lfs-3.6.1.tar.gz"
  tar -xzf "$ROOT/downloads/git-lfs-3.6.1.tar.gz" -C "$ROOT/downloads"
  cp "$ROOT/downloads/git-lfs-3.6.1/git-lfs" "$ROOT/bin/git-lfs"
fi
export PATH="$ROOT/bin:$PATH"
git -C "$ROOT/curobo" lfs install --local
git -C "$ROOT/curobo" lfs pull --include="src/curobo/content/assets/robot/franka_description/**"
export CUDA_HOME="$ROOT/cuda"
export PATH="$ROOT/cuda/bin:$ROOT/venv/bin:$PATH"
export LD_LIBRARY_PATH="$ROOT/cuda/lib:${LD_LIBRARY_PATH:-}"
export MAX_JOBS=1
export TORCH_CUDA_ARCH_LIST=8.9
export TORCH_EXTENSIONS_DIR="$ROOT/cache/torch_extensions"
if ! "$PY" -c 'import torch; from curobo.curobolib import lbfgs_step_cu, kinematics_fused_cu, line_search_cu, tensor_step_cu, geom_cu' > /dev/null 2>&1; then
  "$PY" -m pip install --no-deps --no-build-isolation -e "$ROOT/curobo"
fi
"$PY" -m pip freeze > "$ROOT/requirements.freeze.txt"
git -C "$ROOT/curobo" diff --binary > "$ROOT/curobo_build.patch"
"$ROOT/cuda/bin/nvcc" --version > "$ROOT/nvcc_version.txt"
sha256sum "$ROOT/micromamba-2.3.2.tar.bz2" > "$ROOT/installer.sha256"
