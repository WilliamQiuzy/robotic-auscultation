#!/usr/bin/env bash
set -euo pipefail
WAM_SIM_ROOT="${WAM_SIM_ROOT:-$HOME/wam-sim}"
export LD_LIBRARY_PATH="$WAM_SIM_ROOT/sysroot/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 OMNI_KIT_ACCEPT_EULA=YES
cd "$WAM_SIM_ROOT/project"
exec "$WAM_SIM_ROOT/env_isaacsim51/bin/python" -u -B "$@"
