#!/usr/bin/env bash
set -euo pipefail
wam_project=/home/robot/wam-sim-media/hall-modality-v1/project
wam_source=/home/robot/wam-sim-media/skill_v4/project
wam_prefix=/home/robot/miniconda3/envs/env_isaacsim
cd "$wam_project"
export PYTHONPATH="$wam_project:$wam_source:/home/robot/wam-sim-media/pi05-v4-deps:/home/robot/Desktop/stethoscope/sim/auscultation_training/python_deps"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH="$wam_prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GSETTINGS_SCHEMA_DIR="$wam_prefix/share/glib-2.0/schemas"
exec 9>/home/robot/wam-sim-media/skill_v4/isaac.lock
flock -n 9
exec "$wam_prefix/bin/python" -u -B -m sim.auscultation_pi05_v6.collect_demo \
 --source-project "$wam_source" --annotation sim/auscultation_vla_v2/sensor_pilot_annotation.json "$@"
