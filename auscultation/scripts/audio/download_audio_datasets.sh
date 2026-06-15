#!/usr/bin/env bash
# ============================================================================
# 听诊声学质量模块 — 开源音频数据集下载脚本
# ----------------------------------------------------------------------------
# 权威清单见 scripts/audio/datasets_registry.yaml。本脚本只下载【可直接获取】的集;
# 需实名认证(PhysioNet)/接受条款(HeAR)/向作者申请(ZCHSound)的会打印手动指引。
#
# 用法:
#   bash scripts/audio/download_audio_datasets.sh <out_dir> [name ...]
#   bash scripts/audio/download_audio_datasets.sh data/audio all_open   # 全部开放集
#   bash scripts/audio/download_audio_datasets.sh data/audio sprsound icbhi
#   bash scripts/audio/download_audio_datasets.sh data/audio list       # 列出可选项
#
# 依赖(按需): git, curl, unzip; 可选 huggingface-cli / kaggle (各自需登录)。
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

# ── 开放集: 直接下载 ────────────────────────────────────────────────────────
dl_sprsound() {           # 肺音质量主锚: 含 'Poor Quality' 类 (CC-BY-4.0)
  say "SPRSound (GitHub, CC-BY-4.0) → $OUT/sprsound"
  git clone --depth 1 https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound.git "$OUT/sprsound"
}
dl_icbhi() {              # 肺音设备多样性锚 (Kaggle 镜像; 官方站 TLS 过期)
  say "ICBHI 2017 → $OUT/icbhi  (官方站证书过期, 用 Kaggle 镜像)"
  if have kaggle; then
    kaggle datasets download -d vbookshelf/respiratory-sound-database -p "$OUT/icbhi" --unzip
  else
    warn "需要 kaggle CLI: pip install kaggle + 配置 ~/.kaggle/kaggle.json"
    warn "  kaggle datasets download -d vbookshelf/respiratory-sound-database -p $OUT/icbhi --unzip"
  fi
}
dl_hf_lung_v1() {         # ICU/呼吸机高噪声硬例 (CC-BY-4.0)
  say "HF_Lung_V1 (GitLab, CC-BY-4.0) → $OUT/hf_lung_v1"
  git clone --depth 1 https://gitlab.com/techsupportHF/HF_Lung_V1.git "$OUT/hf_lung_v1"
}
dl_kauh() {               # Littmann 3200 干净参考 (CC-BY-4.0, Mendeley)
  say "KAUH lung sounds (Mendeley, CC-BY-4.0) → $OUT/kauh"
  warn "Mendeley 需浏览器下载或核对直链(DOI 10.17632/jwyy9np4gv.3):"
  warn "  https://data.mendeley.com/datasets/jwyy9np4gv/3  → 下载 zip 解压到 $OUT/kauh"
}
dl_respiratory_tr() {     # 多通道心+肺 (CC-BY-4.0, Mendeley)
  say "RespiratoryDatabase@TR (Mendeley, CC-BY-4.0) → $OUT/respiratory_tr"
  warn "  https://data.mendeley.com/datasets/p9z4h98s6j/1  → 下载解压到 $OUT/respiratory_tr"
}
dl_yaseen() {             # 干净心音正样本 (GitHub)
  say "Yaseen 2018 heart sounds (GitHub) → $OUT/yaseen2018"
  git clone --depth 1 https://github.com/yaseen21khan/Classification-of-Heart-Sound-Signal-Using-Multiple-Features-.git "$OUT/yaseen2018"
}
dl_circor_hf() {          # CirCor 便利镜像(HF); 完整版优先 PhysioNet(见 manual)
  say "CirCor DigiScope (HF 镜像) → $OUT/circor_hf"
  if have huggingface-cli; then
    huggingface-cli download luyangliuable/circor-heart-sound --repo-type dataset --local-dir "$OUT/circor_hf"
  else
    warn "需要 huggingface-cli: pip install -U huggingface_hub"
    warn "  huggingface-cli download luyangliuable/circor-heart-sound --repo-type dataset --local-dir $OUT/circor_hf"
  fi
}
dl_coswara() {            # SSL 预训练(呼吸/咳嗽, 非听诊器) (CC-BY-4.0)
  say "Coswara (SSL 预训练, GitHub) → $OUT/coswara"
  git clone --depth 1 https://github.com/iiscleap/Coswara-Data.git "$OUT/coswara"
}

# ── 需认证/条款/申请: 只打印指引 ────────────────────────────────────────────
manual_cinc2016() {       # ★心音质量金标(good/poor + unsure) — PhysioNet 实名认证
  warn "CinC 2016 (心音质量金标, ODC-BY): 需 PhysioNet 认证账号"
  warn "  https://physionet.org/content/challenge-2016/1.0.0/"
  warn "  wget -r -N -c -np --user <USER> --ask-password https://physionet.org/files/challenge-2016/1.0.0/"
}
manual_circor() {         # 完整 CirCor (PhysioNet)
  warn "CirCor 完整版 (PhysioNet, ODC-BY):"
  warn "  https://physionet.org/content/circor-heart-sound/1.0.3/"
}
manual_ephnogram() {      # 含 low-quality 标 + 环境噪声参考通道
  warn "EPHNOGRAM (PhysioNet, ODC-BY, 含 low-quality 标):"
  warn "  https://physionet.org/content/ephnogram/1.0.0/"
}
manual_pascal() {         # Artifact/noisy 负样本; 官方站不稳, 用镜像
  warn "PASCAL Heart Sounds (Artifact/noisy 负样本): 官方站常挂, 找 Kaggle 镜像并核对 A=176 B=656"
  warn "  https://www.peterjbentley.com/heartchallenge/index.html"
}
manual_zchsound() {       # 现成 clean/noisy 划分, 但访问未证实
  warn "ZCHSound (clean/noisy 现成划分, 访问未证实): 经 IEEE JBHI 论文/作者获取"
  warn "  https://ieeexplore.ieee.org/document/10384868/"
}

OPEN=(sprsound icbhi hf_lung_v1 kauh respiratory_tr yaseen circor_hf coswara)
MANUAL=(cinc2016 circor ephnogram pascal zchsound)

run_one() {
  case "$1" in
    sprsound) dl_sprsound ;;
    icbhi) dl_icbhi ;;
    hf_lung_v1|hflung) dl_hf_lung_v1 ;;
    kauh) dl_kauh ;;
    respiratory_tr|tr) dl_respiratory_tr ;;
    yaseen) dl_yaseen ;;
    circor_hf) dl_circor_hf ;;
    coswara) dl_coswara ;;
    cinc2016) manual_cinc2016 ;;
    circor) manual_circor ;;
    ephnogram) manual_ephnogram ;;
    pascal) manual_pascal ;;
    zchsound) manual_zchsound ;;
    all_open) for d in "${OPEN[@]}"; do run_one "$d"; done ;;
    all_manual) for d in "${MANUAL[@]}"; do run_one "$d"; done ;;
    list)
      echo "开放集(可直接下载): ${OPEN[*]}"
      echo "需认证/申请(打印指引): ${MANUAL[*]}"
      echo "便捷批量: all_open | all_manual"
      ;;
    *) echo "未知项: $1 (用 'list' 查看)" >&2; return 1 ;;
  esac
}

for t in "${TARGETS[@]}"; do run_one "$t"; done
say "完成。质量金标优先级: SPRSound(肺) + CinC2016(心) + ZCHSound(心,若可得)。"
