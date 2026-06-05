# 感知模型「拉满」自主迭代日志

> 用户授权(2026-06-05): 长时自主任务, 把感知模型拉满(尤其真实后背肺点 PCK@10)。反复迭代、用尽手段(更多数据/训练方法/调数据集)。用户几小时不盯, 我自主决策+全程记录+产物下载本地+提交 GitHub。

## 基线(v3 LoRA, 同 val_split held-out, soft-argmax)
| 域 | PCK@10 | 中位误差 |
|---|---|---|
| 真实裸背肺点(scoliosis) | **0.67** | 6.3% |
| 真实正面心点(HuMMan) | 0.91 | 4.9% |
| 合成同域后背 | 0.64 | 5.0% |

**关键洞察**: 真实后背(0.67) > 合成后背(0.64) → sim-to-real 在后背已非主瓶颈; 真实数据量(held-out仅92点/78图)+标签质量+读出才是。

## 迭代计划(ROI 序, 边做边调)
- [ ] **A. 免费 eval 增益**: TTA(多尺度+翻转换LR) + τ 扫描 + ckpt 集成(eval-only, 分钟级)
- [ ] **B. 训练手段扫描**: 后背过采样↑ / EMA / LoRA rank(8/16/32) / 更久 / warmup
- [ ] **C. 挖更多 HuMMan 后向 Kinect 帧**(真实+GT精确标签, 直击瓶颈)
- [ ] **D. 合成后背增产**(更多 betas + 更强 DR), 仅当 C 测出有用
- [ ] **E. scoliosis 曲率重拟合**(清纵向标签噪声)
- [ ] **F. Sapiens 编码器 A/B**(泛化天花板)
- [ ] **最终**: 合并最优发现 → 终版模型, 报告

## 进度记录
(每轮追加: 做了什么 / 结果数字 / 决策)

### 2026-06-05 启动
- 恢复 4090(instance 39462559)。读 refit_scoliosis / synthetic DR 旋钮 / data_gen 配置。
- 数据集搜索: Scoliosis1K=剪影脱敏无RGB(不可用); Roboflow nude-back=已有; THuman/3DBodyTex=gated。结论=自主可得的真实裸背增量有限。
- **box 数据现状**: 处理好的训练集在(humman4282/syn19200/scoliosis1118), 但原始数据(HuMMan Kinect/scoliosis原图/bedlam皮肤)已清。**本地有 scoliosis 86原图+labels_hmr2.pkl+UV+SMPL**(可重增强)。

### A. 免费 eval 增益(完成)
- **τ 扫描**(v3 best scoliosis): PCK@10 在 τ=0.01 最优(0.67); τ=0.02 中位更小(5.5%)但@10略降。**保持 0.01**。
- **多尺度 TTA**(已实现 eval._forward_maps + --tta): scales (1.0,1.14,1.27) → 后背 **PCK@10 0.67→0.68 / PCK@5 0.42→0.44 / 中位6.1%**, 前胸 0.91 不掉, 无 L/R 风险。**采纳为标准评测/部署协议(+0.01 免费)**。

### B. 训练手段短扫描(完成, 4配置×4epoch, 92点训练侧held-out, 单尺度)
| 配置 | best real-back | 备注 |
|---|---|---|
| A_base r16 | 0.475 @ep3 | 基线 |
| **B_over 后背过采样0.45** | **0.518 @ep0** | **最优 +0.04** |
| C_ema EMA0.999 | 0.471 @ep1 | ≈基线 |
| D_r32 rank32 | 0.427 @ep1 | 更差(容量大→过拟合) |
- 结论: **后背过采样=最佳训练杠杆**; EMA≈基线; rank32更差。绝对值低于v3训练侧0.604=短调度混淆 → 用完整15ep复现。

### C. v4 完整候选(进行中, 15ep, 均带过采样后背0.45)
- **v4a = ViT-B + r16 + 过采样**(隔离过采样 vs v3)
- **v4b = ViT-L/14 + r16 + 过采样, bs6**(编码器天花板; ViT-L预下载成功304M)
- 跑完各域 TTA 评测对比 v3(0.68), 选最优=终版。

### 编码器天花板 A/B(队列, sweep后)
- **决策=DINOv2 ViT-L/14**(embed1024 vs B的768): 同 torch.hub loader, build_encoder 已支持 dinov2_*, apply_lora 模块名一致 → **零改码, 仅 --encoder dinov2_vitl14**。无 gating。
- DINOv3(facebook/dinov3-vitb16): **gated**(需网页接受Meta许可)→ 无法自主, 延后。Sapiens: 集成重 + 可能gated, 延后。

### ★ 关键转向: 真实部署域曝光(2026-06-05, 用户给实验室照片)
- `data/lab_sample_data/back/` 3张实验室实拍(用户: 未来真实数据+action训练数据基本长这样, 仿真要对齐)。
- **域特征**: ①**穿衣**(灰色polo, 衣褶/领/红条纹) ②坐姿背面头部可见 ③**蓝色手术铺巾背景**(挂墙角+铺座面)+米色墙 ④竖构图3:4躯干居中第三人称略高 ⑤均匀室内光。
- **v3实测**(infer_overlay.py CPU+TTA, 见 docs/runs/lab_domain/): part分割连贯+正确判后背(泛化好); 肺点正背那张合理, 转身姿势会聚到一侧。**结论=模型能用但 穿衣/坐姿/铺巾 是off-distribution**。
- **新增工具**: `ausc/training/infer_overlay.py`(无GT推理+叠加, FORCE_CPU可CPU跑); `eval._forward_maps` autocast兼容CPU。
- **素材**: `assets/clinical_bg/` 9张铺巾/墙背景板(从lab图裁), 铺巾主色RGB~(184,189,184)。
- **合成对齐计划(下一大迭代, 需GPU+pyrender env重建)**: ①torso穿衣纹理(图像空间叠灰针织+衣褶+领, IUV标签不变=听诊点在衣下=符合临床) ②蓝铺巾背景合成 ③坐姿背面取景匹配。可低成本: 把86张真实scoliosis裸背叠合成衬衫→穿衣真实后背训练样本。

### 队列
1. [跑] B 训练扫描 → 选最优配置
2. [待] DINOv2 ViT-L/14 LoRA 编码器 A/B
3. [待] 合并 最优配置+最优编码器 → 终版 v4(配 TTA 评测)
4. [备] scoliosis 重增强(本地有原图) / Sapiens(若 ViT-L 有效再考虑更大)
