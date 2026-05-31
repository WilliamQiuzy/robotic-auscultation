#!/usr/bin/env python3
"""
检查 DensePose-COCO 标注效果 + 演示听诊点自动标注（NN 查找）。

用法:
  python inspect_densepose.py --ann ../data/densepose_coco/annotations/densepose_coco_2014_minival.json \
                              --img ../data/densepose_coco/images/val2014 \
                              --out ../data/_viz --n 6

产出: 每张样图叠加 (1) DensePose 标注点(按身体块上色) (2) 临时听诊点(NN 查找, 待医学校验)。
说明详见 ../docs/engineering.md §Phase 1A。
"""
import argparse, json, os
from collections import defaultdict
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# DensePose 24 身体块: 1=躯干背, 2=躯干前, 23/24=头, 其余四肢
TORSO_FRONT, TORSO_BACK = 2, 1

# 临时听诊点定义 (I, U, V) —— ⚠️ 待医学/解剖校验，仅用于演示自动标注流程
# 胸面用躯干前(I=2)。UV 布局需进一步核实，这里取粗略值。
PROVISIONAL_POINTS = {
    "aortic?":   (TORSO_FRONT, 0.55, 0.80),
    "pulmonary?":(TORSO_FRONT, 0.45, 0.80),
    "mitral?":   (TORSO_FRONT, 0.40, 0.65),
    "tricuspid?":(TORSO_FRONT, 0.55, 0.65),
}


def bbox_to_px(dp_x, dp_y, bbox):
    x, y, w, h = bbox
    return x + np.asarray(dp_x) / 255.0 * w, y + np.asarray(dp_y) / 255.0 * h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--img", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = json.load(open(args.ann))
    imgs = {i["id"]: i for i in d["images"]}
    by_img = defaultdict(list)
    for a in d["annotations"]:
        if "dp_x" in a:
            by_img[a["image_id"]].append(a)

    # 选「人最大 + 躯干前可见」的图，最接近我们的近景躯干视角
    def score(anns):
        best = 0
        for a in anns:
            if TORSO_FRONT in set(map(int, a["dp_I"])):
                best = max(best, a["bbox"][2] * a["bbox"][3])
        return best
    ranked = sorted(by_img.items(), key=lambda kv: score(kv[1]), reverse=True)

    # 展平成 (img_id, person_ann)，按躯干前点数*面积排序，选最像"近景躯干"的人
    persons = []
    for img_id, anns in by_img.items():
        for a in anns:
            I = set(map(int, a["dp_I"]))
            if TORSO_FRONT in I:
                n_torso = sum(1 for i in a["dp_I"] if int(i)==TORSO_FRONT)
                persons.append((n_torso*a["bbox"][2]*a["bbox"][3], img_id, a))
    persons.sort(key=lambda t: t[0], reverse=True)

    done = 0
    for _, img_id, a in persons:
        if done >= args.n:
            break
        info = imgs[img_id]
        path = os.path.join(args.img, info["file_name"])
        if not os.path.exists(path):
            continue
        im = np.array(Image.open(path).convert("RGB"))
        H, W = im.shape[:2]
        # 裁到 bbox(留 15% margin) —— 模拟临床近景躯干视角
        x,y,w,h = a["bbox"]; mx,my = w*0.15, h*0.15
        x0,y0 = max(0,int(x-mx)), max(0,int(y-my))
        x1,y1 = min(W,int(x+w+mx)), min(H,int(y+h+my))
        crop = im[y0:y1, x0:x1]

        fig, ax = plt.subplots(1, 2, figsize=(14, 9))
        for axi in ax:
            axi.imshow(crop); axi.axis("off")
        ax[0].set_title("DensePose points (colored by body part, torso-front=red)")
        ax[1].set_title("Provisional auscultation points (NN lookup, NEEDS medical review)")

        I = np.asarray(a["dp_I"]); U=np.asarray(a["dp_U"]); V=np.asarray(a["dp_V"])
        px, py = bbox_to_px(a["dp_x"], a["dp_y"], a["bbox"])
        px, py = px-x0, py-y0  # 平移到裁剪坐标
        # 左: 所有点按块上色，躯干前高亮红
        torso = I==TORSO_FRONT
        ax[0].scatter(px[~torso], py[~torso], c=I[~torso], cmap="tab20", s=18, vmin=1, vmax=24, alpha=.7)
        ax[0].scatter(px[torso], py[torso], c="red", s=22, edgecolors="white", linewidths=.3)
        # 右: 临时听诊点(躯干前点里 NN)
        for name,(ti,tu,tv) in PROVISIONAL_POINTS.items():
            m = I==ti
            if not m.any(): continue
            j = np.argmin((U[m]-tu)**2 + (V[m]-tv)**2)
            cx, cy = px[m][j], py[m][j]
            ax[1].scatter([cx],[cy], c="cyan", s=200, marker="x", linewidths=3)
            ax[1].annotate(name, (cx,cy), color="yellow", fontsize=11, weight="bold",
                           xytext=(6,6), textcoords="offset points")
        out = os.path.join(args.out, f"viz_{img_id}_{a['id']}.png")
        plt.tight_layout(); plt.savefig(out, dpi=95, bbox_inches="tight"); plt.close()
        print("saved", out)
        done += 1
    print(f"done: {done} crops -> {args.out}")


if __name__ == "__main__":
    main()
