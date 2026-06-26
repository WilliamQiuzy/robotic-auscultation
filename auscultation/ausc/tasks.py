#!/usr/bin/env python3
"""Perception → Action 的【任务路由层】(对应架构图里的 Target Hand-off)。

把"听什么任务"映射到"读 IUV 上的哪一组听诊点、按什么顺序":
  医生 / 语言指令给一个 task(heart / lung / ...)
  → 本模块返回该任务的【有序听诊点集】+ 所在区域(前胸/后背)
  → 感知在预测的稠密 IUV 上反查这些点的固定 (I,U,V) 地址 → 2D 点
  → 目标热图 + 3D 目标点 交给 VLA(设计B 接口)。

职责分工(单一权威):
  - 点的【解剖定义】(SMPL 顶点索引) = ausc/humman_label.define_ausc_vertices
  - 点的【任务归属 + 顺序】(听哪几个、什么次序)= 本模块 TASKS
点集可自由增删: 标签全由 SMPL 顶点合成(无人工标注), 改 TASKS + define_ausc_vertices 即可。

医学标准(Bates' Guide to Physical Examination / 诊断学):
  ● heart 听心: 前胸 5 区, 助记 **APETM**;
    听诊顺序 Aortic → Pulmonic → Erb → Tricuspid → Mitral(心底 → 心尖)。
  ● lung  听肺: 后背 8 点(4 层双侧, Lopes);
    顺序 = 自肺尖向下、左右对比的阶梯(R/L 成对逐级)。

将来扩展: 加一个 `task` 参数(医生选 heart / lung / 其它), 在此注册新任务即可,
例如 lung_full(后+前+侧)、abdomen(腹部四象限肠鸣音)等 —— 不同任务调不同区域/点位。
"""

TASKS = {
    "heart": {
        "region": "front",                                              # 前胸
        "landmarks": ["aortic", "pulmonary", "erb", "tricuspid", "mitral"],  # APETM 听诊顺序
        "desc": "Cardiac auscultation — anterior chest, 5 areas (APETM, Bates')",
    },
    "lung": {
        "region": "back",                                               # 后背
        "landmarks": ["R1", "L1", "R2", "L2", "R3", "L3", "R4", "L4", "R5", "L5"],  # 肺尖→肺底, 左右对比阶梯
        "desc": "Pulmonary auscultation — posterior, 10 points (5 levels bilateral, V6A 标准, 围肩胛骨)",
    },
    # 将来可注册: "lung_full"(后+前+侧, 含右中叶/左舌叶), "abdomen"(四象限) ...
}

DEFAULT_TASK = "heart"

# 直接监督的 landmark 头【固定通道顺序】(后背肺10 + 前胸心5 = 15)。
# 训练时把样本里出现的点按名映射到通道; 后背样本只激活 0..9, 前胸样本只激活 10..14(其余 mask=0)。
# 改动需与 humman_label.define_ausc_vertices / TASKS 的点名保持一致。
LANDMARK_ORDER = [
    "R1", "L1", "R2", "L2", "R3", "L3", "R4", "L4", "R5", "L5",   # lung 后背 (channels 0..9)
    "aortic", "pulmonary", "erb", "tricuspid", "mitral",          # heart 前胸 (channels 10..14)
]
LANDMARK_INDEX = {nm: i for i, nm in enumerate(LANDMARK_ORDER)}
N_LANDMARKS = len(LANDMARK_ORDER)


def list_tasks() -> list:
    """所有已注册任务名。"""
    return list(TASKS)


def _check(task: str) -> str:
    if task not in TASKS:
        raise KeyError(f"unknown task '{task}'; available: {list(TASKS)}")
    return task


def task_region(task: str) -> str:
    """该任务用哪个区域/视角: 'front'(前胸) 或 'back'(后背)。"""
    return TASKS[_check(task)]["region"]


def task_landmarks(task: str, ausc_vertices: dict = None):
    """该任务的【有序听诊点】(顺序 = 临床听诊顺序)。
    - ausc_vertices=None → 返回点名列表 ['aortic', ...];
    - 传入 humman_label.define_ausc_vertices(model) 的结果 →
      返回 [(name, vertex_idx), ...](只取该任务区域里存在的点)。"""
    t = TASKS[_check(task)]
    names = list(t["landmarks"])
    if ausc_vertices is None:
        return names
    region_pts = ausc_vertices.get(t["region"], {})
    return [(nm, region_pts[nm]) for nm in names if nm in region_pts]
