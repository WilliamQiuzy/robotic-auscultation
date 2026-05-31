# 已废弃脚本（DEPRECATED — 勿用）

这些是项目探索过程中**已放弃的路线**，保留仅为历史记录，**不属于当前管线**。

| 脚本 | 属于哪条废弃路线 | 被什么取代 |
|---|---|---|
| `inspect_densepose.py` `batch_label.py` `filter_torso.py` | **COCO/DensePose-COCO 实验**（开放世界图、稀疏GT最近邻→点位塌缩，域差大） | HuMMan SMPL 管线（`humman_label`+`render_iuv`+`build_dataset`） |
| `label_scoliosis.py` `densepose_predictor.py` | **现成 DensePose 给真实图打标**（裸背上点位塌缩/不准） | HMR2.0 单目 SMPL 拟合（`label_scoliosis_hmr2_gpu.py`，GPU） |
| `ausc_points.py` | 旧听诊点定义（5心/6肺 的 (I,U,V)），**与现行 4心/8肺 冲突** | `humman_label.define_ausc_vertices`（4心/8肺 SMPL顶点）；`infer_side` 已移入 `humman_label` |

⚠️ 当前正确的点位定义只有一处：**`humman_label.define_ausc_vertices`（前胸4 / 后背8）**。
