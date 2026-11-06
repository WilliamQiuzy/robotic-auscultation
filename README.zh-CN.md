# 听诊机器人：从仿真专家到学习策略

**[在线项目演示](https://williamqiuzy.github.io/robotic-auscultation/)** · [English](README.md) · [训练方法](docs/TRAINING.md) · [评测协议](docs/BENCHMARK.md) · [Astra 接入](docs/ASTRA.md)

我们希望让 OpenArm 在 Isaac Sim 中学会对坐姿人体背部的十个听诊点，完成接近、贴合、保持、卸力、撤离和返回。这个 GitHub 项目公开仿真与控制代码、训练入口、实验视频和可检查的量化结果。

## 重建后的同场景比较 · 2026 年 11 月 6 日 · 尚未完成

新批次为 **2 个留出身份 × 2 个已见坐姿 × R3/L3**，每个系统保留 8 个独立 home-start 场景。原始独立资格已关闭：**解析专家 8/8，选定 state-only ACT 0/8；Astra 的 8 个物理结论仍全部待完成**。ACT 的 6 例触发原姿态包络门，2 例触发原 mesh-clearance 门；8 例均实际调用了选定模型。尚不能报告完整三系统结论，也未验证连续十点流程。

新 ACT 完成 **20,000 次更新**，选中 **第 4,500 步**，TRAIN/选模 VAL 的 H20 目标误差为 **5.664018/26.522571 mrad**。训练数据为 20 个计划槽中的 18 个合格专家回合，涵盖 4 个 TRAIN 身份和 1 个选模身份；离线目标误差不等于实测跟踪或接触成功。ACT 输入为 q7 + 给定目标相对 pose9 + confidence1，H2 共 34 维；没有 RGB-D/Hall 模型输入。Astra 在三个固定技能边界监督继续/中止，实际局部运动与接触由 GivenContactExpert 执行。各系统使用给定 3D 目标，并非学习出的视觉定位。传感器因果作用未验证；3/3/1 的并行资源也不是受控速度测试。

[同场景专家/ACT 实际录像与实验文章](docs/drafts/rebuild-20261105.html) · [部分闭合结果证据](docs/drafts/evidence/rebuilt-matched-20261106-v3.partial.json) · [硬件与真实世界差距](docs/real_world_gap_20261105.md)。Astra 完成前不生成最终 canonical 结论。以下 11 月 5 日历史发布的原始数字、分母和媒体保持独立。

目前**专家接触技能已有物理资格，学习模型仍在开发**。项目演示页可切换中文；专家与 ACT 视频分别标注，不把专家录像当作模型成功。

| 已公开实验 | 结果 | 范围 |
|---|---|---|
| 解析专家实际 Isaac 资格 | **8/8** | 固定八个诊断回合，完成接触、至少 3 秒保持、卸力与回 home |
| 官方 ACT 实际训练 | 两轮各 **5,000 步** | 7 个 TRAIN 身份、1 个选模 VAL 身份 |
| 选定 ACT 权重实际 Isaac 闭环 | **0/2** | 训练、验证场景各一例；均在接触前被姿态门停止 |
| 独立留出人体 benchmark | 尚无已公开结果 | 不能据此宣称已证明泛化 |
| Astra + 本地技能 | 已提供接口原型 | 尚无实时 API 或 Isaac 执行成绩 |

这些是不同的诊断批次，不能作为同协议排行榜。实际结果与视频来自 2026 年 10 月 28–29 日；本次发布于 11 月 5 日。任务定义有十个听诊点，本页的小规模实验尚未验证学习模型完成全部十点。

## 如何训练

1. 在批准的工具几何、相机与人体场景上，用解析专家采集实际物理回合。
2. 只把通过接触、保持、卸力与返回资格的专家回合用于正样本；失败保留在完整尝试分母中。
3. 筛选阶段、动作来源和历史一致的训练窗口，先按人体身份分开 TRAIN / VAL，再只用 TRAIN 拟合归一化。
4. 训练官方 LeRobot ACT：两帧 state17，输出 H20，执行 E10 后更新观测。接触与力控由本地解析控制器负责。
5. 把真实选定权重放回 Isaac，按实际接触和完整动作评估；离线误差和进程正常退出不代表任务成功。

[`demo/train_act.py`](demo/train_act.py) 提供独立的数值参考训练入口。训练还需自行准备合格的 NPZ 数组、LeRobot/PyTorch 环境与有许可证要求的人体资产。当前公开包不包含原始训练数据、权重和完整历史仿真环境，不能直接一条命令复现原实验。具体数组格式和限制见[训练说明](docs/TRAINING.md)。

当前 ACT 输入是状态和目标几何，并未直接使用 RGB、深度图或 Hall。Hall 用于本地接触控制；声学未参与仿真训练。传感器被记录，不等于策略已有效利用它。

## 查看结果和接口

```sh
pip install numpy
python -m demo.benchmark --verify-assets --output /tmp/ausc-results.json
python -m unittest discover -s demo_tests -v
python -m demo.astra_policy \
  --observation examples/astra_observation.json \
  --output /tmp/astra-request.json
```

最后一条默认是 dry run，不调用模型、不执行动作。Astra 先作为基于观测的技能决策器，选择目标、继续本地控制、撤离或中止；后续接上仿真技能桥后，需要用相同场景、执行器和物理门评测，并同时报告调用、干预、延迟和成本。它尚不是完成的多模态听诊策略。

公开视频均为实际仿真录像，不含真实患者视频。模型和人体数据的许可证要求仍由各自所有者规定；服务器地址、API 密钥和内部原始资料不在公开仓库中。
