# 肺音标注员说明（Annotation Guide）

> 配套:[`annotation_schema.json`](./annotation_schema.json)(字段定义)、[`prelabel.py`](./prelabel.py)(预填脚本)、
> [`docs/acoustic_annotation_and_training.md`](../../docs/acoustic_annotation_and_training.md)(为什么这么标)。
> **范围:只标后背肺音。** 每个文件 = 机械臂一次落点的 ~10s 录音。

## 一句话原则:质量 ≠ 病理
你标两件**互不相干**的事:
- **质量**:这段录音**清不清楚、能不能用来诊断**——和听到的是正常还是异常呼吸音**无关**。
- **内容**(可选):听到的是正常 / 湿啰音(crackle)/ 哮鸣(wheeze)…——这是诊断,**单独标**。

⚠️ 干净的正常呼吸音 = **高质量**;干净的带 crackle = **也高质量**;嘈杂的 = **低质量**(不管有没有异常音)。

## 流程:确认/纠正,不从零标
1. 先跑预填:`python scripts/audio/prelabel.py --glob "session/*.wav" --point_from_name --out_dir annots/`
   每条生成一个 `.annot.json`,里面 `prelabel` 已用 Q_proxy 自动填了**建议质量分 + 建议失败原因**。
2. 你打开音频 + 频谱图 + 同步视频,**确认或改** `quality` / `respiratory` 字段。`prelabel` 只是起点。

## 必填字段(`quality` 轨)
| 字段 | 怎么判 |
|---|---|
| `quality.overall` | **0 不可用**(几乎听不到呼吸/全是噪声)/ **1 勉强**(能听到但费劲)/ **2 好**(清楚)/ **3 优**(非常清晰干净) |
| `quality.usable_for_diagnosis` | 这段够不够让医生下判断(true/false) |
| `quality.failure_reasons` | 若差,为什么(多选):`motion_artifact` 动/摩擦、`ambient_noise` 环境噪声/说话、`poor_contact` 没贴住/闷、`clipping` 过载削波、`heart_dominant` 心音盖过、`dropout` 间歇断 |
| `respiratory.clean_cycles` | 这段里**干净呼吸周期数**(一次吸+一次呼=1)。优先看 Hall 位移信号给的周期边界 |
| `respiratory.air_entry` | 听不听得到气流:`present` / `weak` / `absent` |

> **`absent`(无呼吸音)要小心**:若当时**力够、确在肺点**(看 `context`)仍无呼吸音 → 可能是真发现,标 `air_entry=absent` 但 `failure_reasons` 不要填 `poor_contact`;若力太轻/位姿不对 → 是 `poor_contact`。

## 校准提示(Q_proxy 能/不能帮你的)
预填的 `suggested_failure_reasons` 对 **环境噪声 / 运动伪影 / 削波** 比较准(已校准 AUC≈0.85–0.99),**直接信但要扫一眼**。
对 **闷/没贴住、间歇断、心音污染** Q_proxy 偏弱——**这几类必须人工听确认**(机械臂的力信号也会帮判没贴住)。

## 内容轨(可选,有医生时再标)
`content.per_cycle`:逐呼吸周期标 `normal/crackle_fine/crackle_coarse/wheeze/rhonchi/stridor/absent` + 起止秒。对齐 ICBHI 制式。**这条不影响质量分。**

## 配对采集提醒(给采集者)
尽量在**同一个后背点**录"好 vs 坏"对照(好接触 vs 太轻、静止 vs 故意动、安静 vs 说话)——这种同点 good/bad 对训 Q 最值钱,标注时在 `notes` 里注明配对组 id。
