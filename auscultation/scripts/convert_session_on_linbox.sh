#!/usr/bin/env bash
# Thin fail-closed entry point for the linbox-only canonical LeRobot conversion.
set -euo pipefail

actual_home="$(/usr/bin/python3 -c 'import os,pwd; print(pwd.getpwuid(os.geteuid()).pw_dir)')"
if [[ "${WAM_HOST_ROLE:-}" != "linbox" || "$actual_home" != "/home/robot" || "$(/bin/hostname)" != "robot-G914" ]]; then
    echo "conversion requires the audited linbox host identity" >&2
    exit 2
fi

session="${1:?usage: convert_session_on_linbox.sh SESSION REPO_ID INACTIVE_ENV CODE_RELEASE_ID}"
repo_id="${2:?repo id is required}"
environment="${3:?inactive immutable converter environment is required}"
code_release_id="${4:?converter code release id is required}"

scripts_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
manager="$scripts_dir/deployment/linbox_converter_env.py"
control_python="${WAM_CONTROL_PYTHON:-$HOME/envs/ausc/bin/python}"

[[ -f "$manager" ]] || { echo "converter environment manager is missing" >&2; exit 2; }
[[ -x "$control_python" ]] || { echo "converter control Python is missing: $control_python" >&2; exit 2; }

exec "$control_python" "$manager" convert \
    --session "$session" \
    --repo-id "$repo_id" \
    --environment "$environment" \
    --code-release-id "$code_release_id"
