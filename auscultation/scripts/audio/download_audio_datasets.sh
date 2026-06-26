#!/usr/bin/env bash
# ============================================================================
# 听诊声学质量模块 — 开源【肺音】数据集下载脚本
# ----------------------------------------------------------------------------
# 范围: 只听后背肺音 → 只下肺音/呼吸音, 不下心音。权威清单见 datasets_registry.yaml。
# 只下载【可直接获取】的集; Mendeley(需浏览器)/认证集会打印手动指引。
#
# 用法:
#   bash scripts/audio/download_audio_datasets.sh <out_dir> [name ...]
#   bash scripts/audio/download_audio_datasets.sh data/audio all_open   # 全部开放肺音集
#   bash scripts/audio/download_audio_datasets.sh data/audio sprsound icbhi
#   bash scripts/audio/download_audio_datasets.sh data/audio list
#
# 依赖(按需): git(+lfs), curl, unzip; SSL 集可选 huggingface-cli。
# ⚠️ 数据较大, 建议在【训练服务器】上跑, 不要塞本地。
# ============================================================================
set -euo pipefail

OUT="${1:-data/audio}"
shift || true
TARGETS=("$@")
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=("list")

mkdir -p "$OUT"
say() { printf "\n\033[1;36m[ausc-audio]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[manual]\033[0m %s\n" "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

# ── 开放肺音集: 直接下载 ─────────────────────────────────────────────────────
dl_sprsound() {           # ★肺音质量主锚: 含 'Poor Quality' 类 (CC-BY-4.0)
  say "SPRSound (GitHub, CC-BY-4.0) → $OUT/sprsound"
  git lfs install 2>/dev/null || true
  git clone --depth 1 https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound.git "$OUT/sprsound"
}
dl_icbhi() {              # 肺音设备多样性锚 (官方站 TLS 过期 → -k; 或 Kaggle 镜像)
  say "ICBHI 2017 → $OUT/icbhi"
  if curl -k -sL -o "$OUT/icbhi.zip" "https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip"; then
    mkdir -p "$OUT/icbhi" && (cd "$OUT/icbhi" && unzip -q -o ../icbhi.zip)
  else
    warn "官方 archive 失败; 用 Kaggle 镜像:"
    warn "  kaggle datasets download -d vbookshelf/respiratory-sound-database -p $OUT/icbhi --unzip"
  fi
}
dl_hf_lung_v1() {         # ICU/呼吸机高噪声硬例 (CC-BY-4.0)
  say "HF_Lung_V1 (GitLab, CC-BY-4.0) → $OUT/hf_lung_v1"
  git clone --depth 1 https://gitlab.com/techsupportHF/HF_Lung_V1.git "$OUT/hf_lung_v1"
}
dl_coswara() {            # SSL 预训练(呼吸/咳嗽, 非听诊器) (CC-BY-4.0)
  say "Coswara (SSL 预训练, GitHub) → $OUT/coswara"
  git clone --depth 1 https://github.com/iiscleap/Coswara-Data.git "$OUT/coswara"
}

# ── 需浏览器/认证: 只打印指引 ───────────────────────────────────────────────
manual_kauh() {           # Littmann 3200 干净参考 (CC-BY-4.0, Mendeley)
  warn "KAUH lung sounds (Mendeley, CC-BY-4.0): 浏览器下载 DOI 10.17632/jwyy9np4gv.3"
  warn "  https://data.mendeley.com/datasets/jwyy9np4gv/3  → 解压到 $OUT/kauh"
}
manual_respiratory_tr() { # 多通道肺 (CC-BY-4.0, Mendeley)
  warn "RespiratoryDatabase@TR (Mendeley, CC-BY-4.0): 浏览器下载"
  warn "  https://data.mendeley.com/datasets/p9z4h98s6j/1  → 解压到 $OUT/respiratory_tr"
}

OPEN=(sprsound icbhi hf_lung_v1 coswara)
MANUAL=(kauh respiratory_tr)

run_one() {
  case "$1" in
    sprsound) dl_sprsound ;;
    icbhi) dl_icbhi ;;
    hf_lung_v1|hflung) dl_hf_lung_v1 ;;
    coswara) dl_coswara ;;
    kauh) manual_kauh ;;
    respiratory_tr|tr) manual_respiratory_tr ;;
    all_open) for d in "${OPEN[@]}"; do run_one "$d"; done ;;
    all_manual) for d in "${MANUAL[@]}"; do run_one "$d"; done ;;
    list)
      echo "开放肺音集(可直接下载): ${OPEN[*]}"
      echo "需浏览器(打印指引): ${MANUAL[*]}"
      echo "便捷批量: all_open | all_manual"
      ;;
    *) echo "未知项: $1 (用 'list' 查看)" >&2; return 1 ;;
  esac
}

for t in "${TARGETS[@]}"; do run_one "$t"; done
say "完成。肺音质量金标主锚: SPRSound('Poor Quality' 类)。只做肺音, 不下心音。"
