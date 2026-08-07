# 感知精修实现 + 验证门(2026-08-07)

> 落地 [perception_improvement_roadmap_2026-08-07.md](perception_improvement_roadmap_2026-08-07.md) 的 1–5 全部条目。
> **铁律:所有改动 config-gated、默认 OFF。** live detector(`perc_lmh_v9_occ` 上的 `NormalsDetector.detect`)在所有 flag 默认下**逐字节不变** —— 明天要录模仿学习数据,任何静默回归都会污染数据。**没有离线 PCK 数字 + 你签字之前,不翻任何 flag、不部署、不动 8801、绝不在录制前盲改 detector。**

## 已实现(本次提交,已单测 + 对抗复核)

| # | 条目 | 落点 | flag(默认 OFF) | 类型 |
|---|---|---|---|---|
| C | **离线评估台**(门) | `ausc/perception/eval/landmark_offline_eval.py` | — | 新库,安全 |
| 1 | 裁剪稳定化 | `geometry.CropStabilizer` + `back_square_crop(pct=, stabilizer=)`;`detector._normal_image` | `stabilize_crop`(+`crop_pct=2.0`) | 推理期,gated |
| 4 | Flip-TTA 融合 | `detector._flip_pass`/`_fuse_flip_tta`/`detect` | `flip_tta`(+`flip_tta_guard_px=8`) | 推理期,gated |
| 5a | 稳健 structure_points | `train_landmark_human.structure_points_robust` | `robust_structure` | 后处理,gated |
| 5b | soft-argmax(DSNT)解码 | `train_landmark_human.heatmap_xy_soft` | `soft_decode` | 解码,gated |
| 3 | 真实逐帧焦距法向量 | `detector._normal_image`(f_eff=fx·res/side) | `real_focal_normals` | **重训耦合**,gated |
| 2 | 逐像素深度噪声增强 | `ausc.augment.augment_normal_channel` | — | 训练期,已提交(8530527) |

### 关键设计点
- **Flip-TTA 是"免费"的**:置信度那趟翻转前传本来就在跑,现在**同一趟**同时产出置信度 + 复原的翻转坐标(过去被丢弃)。融合规则:两趟一致(逐点距离 ≤ guard 8px)→ 平均(√2 方差缩减);不一致(遮挡/坏帧)→ 保留原估计。置信度仍从**原始**结构化点算(不被融合结果虚高)。默认 `flip_tta=False` → detect 逐字节等于旧路径(已单测 `test_flip_tta_off_is_byte_identical`)。
- **裁剪稳定化**:2/98 分位 bbox(去边界闪烁)+ (cx,cy,side) 死区-EMA(死区内=闪烁→保持;>25% side=真运动→直接 snap 无滞后;之间=EMA)。有状态,**仅推理**;body 丢失→reset。训练脚本处理独立单帧,**不传 stabilizer**。
- **稳健 structure_points**:旧版层间距 `linspace(t[0],t[4],5)` 只由最噪的两端点(R1/R5)决定;新版用 **Theil-Sen**(10 对斜率取中位 + 中位截距)拟合 5 层轴向位置,全 5 层参与,单个坏端点被压制(单测:R5/L5 各偏 40px 时中层 R3 误差 <0.9×)。
- **soft-argmax 解码**:保留硬 argmax 找峰(抗背景稀释),仅把窗口内亚像素细化从"sigmoid 概率质心"换成"logit 温度-softmax 期望"→ 平滑、无翻格跳变(单测:亚像素滑动响应单调无跳变)。
- **#3 焦距法向量(高危)**:模型是在固定 `K=560` 法向量上训练的;推理端**单独**改真实 f_eff 会造成 train/infer 分布错配→**变差**。因此 `real_focal_normals` 默认 OFF 且**只能与重训后的 checkpoint 一起开**。代码路径已就绪,重训未做(需 GPU + 你签字)。
- **被我否掉的路线图条目**:INTER_NEAREST→INTER_LINEAR(深度 resize)。线性插值会在人体轮廓/空洞处跨深度不连续混合→制造中间深度的幻影表面,而地标恰在轮廓附近。**深度必须 NEAREST**,故不改;此项从路线图移除并附理由。

## 验证门(尚未跑 —— 阻塞在资产定位)

离线 A/B 一条命令(需 GPU + checkpoint + 带深度的黄衣 clip):
```bash
python -m ausc.perception.eval.landmark_offline_eval \
  --ckpt <.../perc_lmh_v9_occ/best.pt> \
  --clip <dir: rgb.mp4 + depth/*.npy + intrinsics.json> \
  --labels <.../clothed_labels_yellow.json> \
  --variants baseline,crop_stab,flip_tta,robust_struct,soft_decode,all_free \
  --out runs/refine_ab.json
```
输出对比表列:**2D 抖动中位 px / 3D 总 mm / 3D 共模 mm / 3D 独立 mm / PCK@5 / 中位 px err**,每列带 Δvs baseline + ✓/✗。

**判据(deploy candidate):PCK 不掉 + 至少一项抖动/3D 列改善。** 共模 mm = #1 杠杆;2D 抖动 px = #4/#5。

### 资产实际位置(2026-08-07 遍历核实 —— 更正此前"不可达"的错误判断)
此前误判"checkpoint 不可达"是因为查错了目录(`~/wam_ausc/data/outputs` 是过时非-git 副本)。真实情况:
- **checkpoint 完全可达**:生产模型 = `perc_lmh_v9_occ`(sha `68037e69…f6a14`),打在**每个** sealed release 里。
  - **live 部署**:linbox `/home/robot/wam_data/perception_runtime/current` → `perception_runtime-86b152f6bd983cf33a6d`(`current` 软链,08-07 09:44),内含 `model/perc_lmh_v9_occ/best.pt`(28MB)+ `model/posterior_gate/yolo11x-pose.pt`(YOLO11x-pose 后验门,118MB)+ DINOv2 backbone。
  - H200 也有一份同 sha 的 eval 副本(`data/perception_eval_models/releases/…-33641f649a…`)。
  - **全网仅此一个 perc_lmh 版本**(无 v4/v6/v7/v8/v10;训练脚本里的 v6_normals/v7_adapt 是历史中间产物,已不存在)。"新旧"差异在**运行时代码 release**(40+ 个 hash,live=86b152f6),**模型本身稳定不变**。
  - 本地 repo `outputs/perc_v0..v5` 是 6 月的**旧 IUV 模型**,与听诊地标模型无关,别混。
- **rgb+depth clip 可达(可跑 jitter/3D A/B)**:linbox `/home/robot/wam_data/sessions/diagnostic_perception_*/`(多段,每段 ~8000 帧 depth + rgb_segments)。**jitter + 共模 3D mm(主杠杆)不需要标注**即可测。
- **仍缺(仅 PCK 需要)**:带标注的深度 clip = H200 raw release,经 `AUSC_PERCEPTION_LABELED_RELEASE`(需 `clothed_labels_a.json`)/`AUSC_PERCEPTION_EVAL_RELEASE` 解析,`require_h200_raw_release()`。本地只有 `data/lab_back_video_test/labeling_yellow/clothed_labels_yellow.json`(2D-GT)+ `_session3_rgb.mp4`,**无深度**。

**我的改动不在 live release**:`86b152f6` 早于本次编辑(release 内 `code/` grep 无 flip_tta/CropStabilizer 标记)→ 运行中的系统完全不受影响;要生效需重建 release(未做)。

**可立即做**:在 linbox 用 live checkpoint + 一段 diagnostic session 跑 jitter/3D A/B(read-only、不碰 8801)。PCK 完整门需再定位 H200 labeled release。

## 上线顺序(建议,全部在门后)
1. 建好上面这张表(定位资产即可跑)。
2. `flip_tta` + `crop_stab` 先过离线 PCK(免训练、最高 ROI)→ 通过 + 你签字 → 才在部署配置里翻 flag。
3. `robust_structure` / `soft_decode` 同法门控。
4. `real_focal_normals` + 深度域增强补全 → **一次打包重训** stage1/2(GPU)→ 出新 checkpoint → 门控 → 上线。
5. **录制当天:detector 保持当前 live(全 flag OFF)。** 精修上线在录制之后。

## 对抗复核(5 维 workflow,10 agents)+ 已修
byte-identity 与 flip-TTA 两维**零 finding**(关键安全性成立)。确认并已修 3 条(均在评估台,即"门"本身):
1. **幸存者偏差(HIGH)**:各变体按置信度门丢不同帧,指标只在幸存帧上算→丢掉最难(最抖)帧的变体反而"更好"。**修**:`run_variant` 返回 coverage;`compare_markdown` 加 cover% 列,coverage 低于 baseline→✗ 判定该对比无效(非 deploy candidate)。
2. **PCK 每变体各自 crop 尺度(MEDIUM)**:`sc=518/side` 用各变体自己的 side→裁剪变小的变体被更严阈值不公平打分。**修**:误差改**全帧 px**(全变体同一尺度)。
3. **decompose 抛异常中止全 A/B(MEDIUM)**:某变体无连续完整帧→`common_removed_jitter_mm` 抛错、整轮结果丢失。**修**:try/except 降级为 NaN(与兄弟指标一致)。

被复核**证伪**2 条(经我复核同意):CropStabilizer"棘轮"(实为有界滞后——锚点冻结,漂移对锚点累积会越过死区,不会卡死;已加注释说明);soft-decode over-peaked(顺手**加固**:改在 sigmoid 概率上做温度 soft-argmax,`temp=1` 逐字节等于现产 `heatmap_xy`——现产解码本就是窗口内 soft-argmax,故 #5b 大部分已在产)。

## 测试
- 新增 4 个测试文件:`test_landmark_offline_eval.py`(harness+coverage 门)、`test_crop_stabilizer.py`(#1 + byte-identity)、`test_structure_and_decode.py`(#5)、`test_detector_refinements.py`(#4 融合 + byte-identity)。
- 复核修复后:上述 4 文件 + `test_production_hardening.py` + eval 基元回归 **62 passed**;更早全包回归 223 passed。`getattr` 默认使旧式/部分构造实例仍走 OFF 路径。
