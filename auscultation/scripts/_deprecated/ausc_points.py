#!/usr/bin/env python3
"""
Task 1: 听诊点的 DensePose (I,U,V) 解剖映射。

【UV 坐标系（由 minival 数据反推，见 docs/engineering.md）】
  正面躯干 I=2, 背面躯干 I=1，两者一致：
    U: 垂直轴。U≈0 上(肩/锁骨) → U≈1 下(腹/腰)。   corr(U,ny)≈0.60
    V: 水平轴。V 高 = 患者【右】侧, V 低 = 患者【左】侧, V≈0.5 = 中线(胸骨/脊柱)。
       (正面: V高=图像左; 背面: V高=图像右; 但都对应患者右侧 → 患者相对、解剖一致)

⚠️ 下列 (I,U,V) 为【医学草稿】，依据三篇论文解剖定义 + 上述 UV 标定推得，
   仍需医学/解剖合作者校验后再用于正式标注。
"""
import numpy as np

FRONT_PART, BACK_PART = 2, 1

# 心脏听诊点（胸面 I=2）。患者相对：V>0.5 患者右, V<0.5 患者左; U 小=上。
CARDIAC = {
    # name        (I, U=上下, V=左右)         解剖位置
    "aortic":    (FRONT_PART, 0.35, 0.57),   # 主动脉瓣: 第2肋间 右胸骨旁(患者右,上)
    "pulmonary": (FRONT_PART, 0.35, 0.43),   # 肺动脉瓣: 第2肋间 左胸骨旁(患者左,上)
    "erb":       (FRONT_PART, 0.46, 0.46),   # Erb点: 第3肋间 左胸骨旁
    "tricuspid": (FRONT_PART, 0.58, 0.47),   # 三尖瓣: 第4-5肋间 左下胸骨缘(近中线)
    "mitral":    (FRONT_PART, 0.62, 0.35),   # 二尖瓣/心尖: 第5肋间 左锁骨中线(患者左,偏外,下)
}

# 肺部听诊点（背面 I=1），双侧网格，避开肩胛骨。V>0.5 患者右, V<0.5 患者左。
LUNG = {
    "R_upper": (BACK_PART, 0.30, 0.65), "L_upper": (BACK_PART, 0.30, 0.35),
    "R_mid":   (BACK_PART, 0.50, 0.65), "L_mid":   (BACK_PART, 0.50, 0.35),
    "R_lower": (BACK_PART, 0.68, 0.65), "L_lower": (BACK_PART, 0.68, 0.35),
}

REGION_POINTS = {"front": CARDIAC, "back": LUNG}


def infer_side(part_map):
    """前胸/后背判别: 稠密IUV的躯干块 part2(前胸) vs part1(后背), 谁多算谁。
    用于推理时自动决定读 4 个心脏点(front) 还是 8 个肺点(back)。
    实测极悬殊(前胸 87209:770, 后背 544:45278)。"""
    import numpy as np
    front = int((np.asarray(part_map) == FRONT_PART).sum())
    back = int((np.asarray(part_map) == BACK_PART).sum())
    return "front" if front >= back else "back"


def bbox_to_px(dp_x, dp_y, bbox):
    x, y, w, h = bbox
    return x + np.asarray(dp_x) / 255.0 * w, y + np.asarray(dp_y) / 255.0 * h


def nn_lookup_gt(ann, target, k=3):
    """在一个 DensePose-COCO 标注的稀疏 GT 点里，按 (I,U,V) 最近邻求 2D 像素。
    返回 (px, py) 或 None。k 个最近点取中位，稳一些。"""
    ti, tu, tv = target
    I = np.asarray(ann["dp_I"], int)
    m = I == ti
    if not m.any():
        return None
    U = np.asarray(ann["dp_U"])[m]; V = np.asarray(ann["dp_V"])[m]
    px, py = bbox_to_px(ann["dp_x"], ann["dp_y"], ann["bbox"])
    px, py = px[m], py[m]
    dist = (U - tu) ** 2 + (V - tv) ** 2
    j = np.argsort(dist)[:max(1, k)]
    return float(np.median(px[j])), float(np.median(py[j]))


def label_person(ann, region):
    """对一个人(已知 region=front/back)，返回 {点名: (px,py)}。缺的点跳过。"""
    out = {}
    for name, target in REGION_POINTS[region].items():
        r = nn_lookup_gt(ann, target)
        if r is not None:
            out[name] = r
    return out


if __name__ == "__main__":
    print("CARDIAC (front, I=2):")
    for k, v in CARDIAC.items():
        print(f"  {k:10s} I={v[0]} U={v[1]} V={v[2]}")
    print("LUNG (back, I=1):")
    for k, v in LUNG.items():
        print(f"  {k:10s} I={v[0]} U={v[1]} V={v[2]}")
