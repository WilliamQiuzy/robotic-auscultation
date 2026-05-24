#!/usr/bin/env bash
# 在任意机器(本地/服务器)一键 clone 全部 baseline(浅克隆)。
# 第三方代码不进我们的 git 仓库,两端各自 clone。
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

declare -A VLA=(
  [openpi]="https://github.com/Physical-Intelligence/openpi.git"
  [openvla]="https://github.com/openvla/openvla.git"
  [openvla-oft]="https://github.com/moojink/openvla-oft.git"
  [RoboticsDiffusionTransformer]="https://github.com/thu-ml/RoboticsDiffusionTransformer.git"
)
declare -A WAM=(
  [lingbot-va]="https://github.com/robbyant/lingbot-va"
  [FastWAM]="https://github.com/yuantianyuan01/FastWAM"
  [Motus]="https://github.com/thu-ml/Motus"
  [cosmos-policy]="https://github.com/nvlabs/cosmos-policy"
  [dreamzero]="https://github.com/dreamzero0/dreamzero"
  [giga-world-policy]="https://github.com/open-gigaai/giga-world-policy"
)

clone_into() {  # $1=目标目录  $2=name  $3=url
  local dst="$1/$2"
  if [ -d "$dst/.git" ]; then echo "  已存在: $2"; else
    echo "  clone: $2"; git clone --depth 1 "$3" "$dst"
  fi
}

mkdir -p "$ROOT/VLA-baselines" "$ROOT/WAM-baselines"
echo "=== VLA baselines ==="; for n in "${!VLA[@]}"; do clone_into "$ROOT/VLA-baselines" "$n" "${VLA[$n]}"; done
echo "=== WAM baselines ==="; for n in "${!WAM[@]}"; do clone_into "$ROOT/WAM-baselines" "$n" "${WAM[$n]}"; done
echo "✅ baselines 就绪。注意:lingbot-va 的 model.py flash-attn 补丁由 WAM-baselines/lingbot-va-eval/apply_patches.sh 打。"
