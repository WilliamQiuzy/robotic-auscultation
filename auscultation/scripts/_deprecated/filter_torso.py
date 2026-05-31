#!/usr/bin/env python3
"""
正面/背面躯干过滤器 (frontal/back torso filter)。

把 DensePose-COCO 的人按「能否用于听诊定位」分类：
  - front : 正面躯干清晰 → 心脏/胸部听诊点
  - back  : 背面躯干清晰 → 肺部听诊点
  - reject: 侧身/遮挡/太小/躯干点太少

判据 (可调常数):
  1. 有 DensePose 标注 (dp_x)
  2. bbox 面积 >= MIN_AREA_FRAC * 图面积   (人够大≈近景)
  3. 躯干点数 >= MIN_TORSO_PTS            (I∈{1前,2后}够多)  *见注
  4. 朝向: front_ratio = n(前)/(n(前)+n(后))；>FRONT 判正面, <BACK 判背面, 之间=侧身丢
  5. 关键点佐证: 双肩可见且左右肩水平次序与朝向一致

注: DensePose 块 I: 1=躯干(一面), 2=躯干(另一面)。文献常用 1=back,2=front；
    实际以数据为准，可用 --swap 翻转。

用法:
  python filter_torso.py --ann ../data/densepose_coco/annotations/densepose_coco_2014_minival.json \
                         --out ../data/filtered_minival.json [--img ... --viz ../data/_viz_filter --n 4]
"""
import argparse, json, os
from collections import Counter, defaultdict
import numpy as np

TORSO_A, TORSO_B = 1, 2          # 两个躯干块
FRONT_PART = 2                    # 默认 2=正面(可 --swap)
MIN_AREA_FRAC = 0.04
MIN_TORSO_PTS = 8
FRONT_TH, BACK_TH = 0.65, 0.35
# COCO keypoints
L_SHO, R_SHO, L_HIP, R_HIP, NOSE = 5, 6, 11, 12, 0


def kp(a, i):
    k = a.get("keypoints")
    if not k:
        return None
    x, y, v = k[3*i], k[3*i+1], k[3*i+2]
    return (x, y, v)


def classify(a, img_w, img_h, front_part=FRONT_PART):
    reasons = []
    if "dp_x" not in a:
        return "reject", ["no_densepose"]
    x, y, w, h = a["bbox"]
    if w*h < MIN_AREA_FRAC*img_w*img_h:
        return "reject", ["too_small"]
    I = np.asarray(a["dp_I"], dtype=int)
    n_front = int((I == front_part).sum())
    n_back = int((I == (TORSO_A if front_part == TORSO_B else TORSO_B)).sum())
    n_torso = n_front + n_back
    if n_torso < MIN_TORSO_PTS:
        return "reject", ["few_torso_pts"]
    ratio = n_front / n_torso
    # 关键点朝向佐证
    ls, rs = kp(a, L_SHO), kp(a, R_SHO)
    kp_face = None
    if ls and rs and ls[2] == 2 and rs[2] == 2:
        # 面向镜头: 人的左肩(ls)在图像右侧 → ls.x > rs.x
        kp_face = "front" if ls[0] > rs[0] else "back"
    if ratio >= FRONT_TH:
        label = "front"
    elif ratio <= BACK_TH:
        label = "back"
    else:
        return "reject", [f"side(ratio={ratio:.2f})"]
    if kp_face and kp_face != label:
        return "reject", [f"kp_conflict(dp={label},kp={kp_face})"]
    return label, [f"ratio={ratio:.2f}", f"area={w*h/(img_w*img_h):.2f}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--img"); ap.add_argument("--viz"); ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--swap", action="store_true", help="翻转前后块定义 (front=1)")
    args = ap.parse_args()
    fp = TORSO_A if args.swap else TORSO_B

    d = json.load(open(args.ann))
    imgs = {i["id"]: i for i in d["images"]}
    kept = []
    counts = Counter(); rej = Counter()
    for a in d["annotations"]:
        info = imgs[a["image_id"]]
        label, why = classify(a, info["width"], info["height"], fp)
        counts[label] += 1
        if label == "reject":
            rej[why[0].split("(")[0]] += 1
        else:
            kept.append({"image_id": a["image_id"], "ann_id": a["id"],
                         "file_name": info["file_name"], "label": label, "bbox": a["bbox"]})
    json.dump(kept, open(args.out, "w"))
    print("=== classification ===")
    for k, v in counts.most_common():
        print(f"  {k:8s} {v}")
    print("=== reject reasons ===")
    for k, v in rej.most_common():
        print(f"  {k:18s} {v}")
    print(f"kept {len(kept)} -> {args.out}")

    if args.img and args.viz:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from PIL import Image
        os.makedirs(args.viz, exist_ok=True)
        for lab in ("front", "back"):
            sel = [k for k in kept if k["label"] == lab][:args.n]
            for k in sel:
                p = os.path.join(args.img, k["file_name"])
                if not os.path.exists(p):
                    continue
                im = np.array(Image.open(p).convert("RGB")); H, W = im.shape[:2]
                x, y, w, h = k["bbox"]; mx, my = w*.15, h*.15
                crop = im[max(0, int(y-my)):min(H, int(y+h+my)), max(0, int(x-mx)):min(W, int(x+w+mx))]
                plt.figure(figsize=(5, 7)); plt.imshow(crop); plt.axis("off")
                plt.title(f"{lab.upper()}  {k['file_name']}", fontsize=9)
                out = os.path.join(args.viz, f"{lab}_{k['image_id']}_{k['ann_id']}.png")
                plt.savefig(out, dpi=90, bbox_inches="tight"); plt.close()
                print("viz", out)


if __name__ == "__main__":
    main()
