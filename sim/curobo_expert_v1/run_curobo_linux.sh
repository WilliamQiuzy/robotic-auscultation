#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/william/wam-sim/eval_curobo
PROJECT=/home/william/wam-sim/project
unset PYTHONHOME PYTHONPATH LD_PRELOAD
export PYTHONPATH="$PROJECT"
export PYTHONNOUSERSITE=1
export CUDA_HOME="$ROOT/cuda"
export PATH="$ROOT/bin:$ROOT/cuda/bin:$ROOT/venv/bin:$PATH"
export LD_LIBRARY_PATH="$ROOT/cuda/lib"
export XDG_CACHE_HOME="$ROOT/cache"
export TORCH_EXTENSIONS_DIR="$ROOT/cache/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$ROOT/cache/torchinductor"
export WARP_CACHE_PATH="$ROOT/cache/warp"
export MAX_JOBS=1
export OMP_NUM_THREADS=1
export TORCH_CUDA_ARCH_LIST=8.9
cd "$PROJECT"
if [ "${1:-}" = "--request" ]; then
  set -- -m sim.curobo_expert_v1.curobo_backend "$@"
fi
exec "$ROOT/venv/bin/python" "$@"
