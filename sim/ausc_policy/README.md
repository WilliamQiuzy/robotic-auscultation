# 听诊策略实验

以 `docs/superpowers/plans/2026-09-29-ausc-diverse-data-policy-redesign.md` 为当前方向。
M0 尚未通过。旧 41 维、只执行首步、单身体 VLA 全量训练已停止。

这里集中放新实验代码；变体由 `configs/ausc_policy/` 控制，历史目录只读。
Mac 用于开发、纯软件检查和查看网页。GPU 推理、训练与 Isaac 均在 Linux；
每次只有一个 GPU owner。没有真实机械臂入口。

## 入口

```sh
python -m sim.ausc_policy.diagnostics --config configs/ausc_policy/m0_diagnostics.json
python -m sim.ausc_policy.web --port 8774
python -m sim.ausc_policy.run_index --runs /home/william/wam-vla-v2/runs/ausc_policy
# 仅 Linux，由指定 Molmo Python 环境执行；每次唯一 run-id。
python -m sim.ausc_policy.run --config configs/ausc_policy/m0_seed_probe.json --run-id m0-seed512-20260929
```

GPU owner 复用既有进程组/资源保护器与共享 GPU 锁。每个 run 保存配置、
代码 commit、来源 shard、种子、进度、结果和进程退出证据；没有自动重试、
自动续训或隐含下游 GPU 作业。旧 checkpoint 的诊断使用原始适配器读取，
不改写历史结果。32 条拟合和 20×16 采样都只是训练样本诊断。
`run_index` 从已有收据生成一张 `summary.csv`；作业完成与 M0/闭环通过是不同列。

## 工程边界

`sim/` 第一方 Python、shell、必要配置进入 Git；人体数据、模型、生成 USD、
实验数组与录像留在数据/结果目录，不进 Git。代码仓库不等于完整资产备份：
Linux 已有资产与结果保留，资产定位/哈希从历史 manifest 读取。
不得上传 SMPL/HuMMan 原始数据或绕过它们的使用许可。

网页实验博客在 `docs/experiment_blog/`。正文保留问题、方法、数值、局限与
下一步；视频只使用真实 Isaac 录像。服务只绑定回环地址并开放固定清单，
不会把项目根目录、凭证、患者资产或模型暴露成网站。
