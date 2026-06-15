# 听诊感知 — 权威入口 / 模型映射(防歧义, 2026-06-14)

> 范围: 本期只做**后背肺/呼吸听诊**(10 点 V6A)。心脏/前胸 = future work(数据受限, 见 docs/PERCEPTION_STATUS.md)。

## 生产(用这些)
| 用途 | 脚本 | 模型 |
|---|---|---|
| **部署推理**(出 10 点 + IUV + 复核门) | `deploy_landmark.py` | `outputs/perc_lmh_v4/best.pt` |
| **金标评估**(逐点误差/PCK/结构) | `eval_landmark_gold.py` | 同上 |
| **训练 landmark 头**(纯人手标 + 自训练) | `train_landmark_human.py` | 骨干暖启动 `outputs/perc_v7_multiview/best.pt`(冻结) |

- **生产模型 = `perc_lmh_v4`**: v7 IUV 骨干(冻结)+ landmark 热图头(人手标种子 + 自训练放量), 读出结构化(脊柱中线+对称宽)。金标 PCK@10≈96% / PCK@5≈71%。
- `perc_v7_multiview` 只作**冻结骨干**(提供特征 + IUV 可视化), 不直接读点。
- HF 备份: `WilliamQiu123/wam-auscultation-perception` → models/perc_lmh_v4/。

## 已弃用 / 历史(勿当生产)
| 脚本 | 状态 |
|---|---|
| `viz_report_iuv.py` | ⚠️ DEPRECATED — 旧 IUV-地址法管线(57%), 已被 deploy_landmark 取代 |
| `eval_gold.py` | LEGACY — IUV-地址法金标评估(复现 v7=57% 基线用); 评生产模型用 eval_landmark_gold.py |
| `eval_geomaug_ab.py` / `reselect_v10.py` / `rebuild_val_targets.py` | 历史实验(几何增强 v10 / 评估基建修复), 非生产 |
| `readout_fallback.py` / `anat_plausibility.py` / `flip_consistency.py` | 旧 IUV-地址法的兜底/评估工具 |
| `annotate_gold.py` / `build_review_montage.py` | 标注/可视化辅助 |

## 一句话
**部署只跑 `deploy_landmark.py`(默认 perc_lmh_v4)。其余引用 v6/v7/v10 的脚本都不是生产入口。**
