#!/usr/bin/env python3
"""
Task 3: 全量批标注流水线 + 导出。

流程: 遍历标注 → filter_torso 分 front/back/reject → ausc_points 查听诊点 → 导出 + 预览。
后端可插拔: backend=gt 用 DensePose-COCO 稀疏 GT 最近邻(本数据集);
            backend=predictor 用 Detectron2 逐像素 IUV(任意图/机器人自采, 见 densepose_predictor.py)。

用法:
  python batch_label.py --ann ../data/densepose_coco/annotations/densepose_coco_2014_minival.json \
      --img ../data/densepose_coco/images/val2014 \
      --out ../data/labels_minival.json --viz ../data/_viz_label --n 6
"""
import argparse, json, os
from collections import Counter
import numpy as np
from filter_torso import classify
import ausc_points as AP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--img")
    ap.add_argument("--out", required=True)
    ap.add_argument("--viz"); ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--swap", action="store_true")
    args = ap.parse_args()
    fp = AP.BACK_PART if args.swap else AP.FRONT_PART

    d = json.load(open(args.ann))
    imgs = {i["id"]: i for i in d["images"]}
    samples = []
    cnt = Counter(); npts = Counter()
    for a in d["annotations"]:
        info = imgs[a["image_id"]]
        label, _ = classify(a, info["width"], info["height"], fp)
        if label == "reject":
            cnt["reject"] += 1; continue
        pts = AP.label_person(a, label)          # {name:(px,py)}
        if not pts:
            cnt["no_points"] += 1; continue
        cnt[label] += 1; npts[label] += len(pts)
        samples.append({
            "image_id": a["image_id"], "file_name": info["file_name"], "ann_id": a["id"],
            "region": label, "bbox": a["bbox"],
            "points": {k: [round(v[0], 1), round(v[1], 1)] for k, v in pts.items()},
        })

    export = {
        "info": {"source": os.path.basename(args.ann), "backend": "gt-nn",
                 "note": "auscultation point labels; (I,U,V) defs are DRAFT pending medical review"},
        "categories": {"front": list(AP.CARDIAC), "back": list(AP.LUNG)},
        "samples": samples,
    }
    json.dump(export, open(args.out, "w"))
    print("=== batch label stats ===")
    print(f"  front  samples={cnt['front']}  avg pts/person={npts['front']/max(1,cnt['front']):.1f}")
    print(f"  back   samples={cnt['back']}   avg pts/person={npts['back']/max(1,cnt['back']):.1f}")
    print(f"  reject={cnt['reject']}  no_points={cnt['no_points']}")
    print(f"  TOTAL labeled samples = {len(samples)} -> {args.out}")

    if args.img and args.viz:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from PIL import Image
        os.makedirs(args.viz, exist_ok=True)
        done = Counter()
        for s in sorted(samples, key=lambda s: -s["bbox"][2]*s["bbox"][3]):
            if done[s["region"]] >= args.n:
                continue
            p = os.path.join(args.img, s["file_name"])
            if not os.path.exists(p):
                continue
            im = np.array(Image.open(p).convert("RGB")); H, W = im.shape[:2]
            x, y, w, h = s["bbox"]; mx, my = w*.15, h*.15
            x0, y0 = max(0, int(x-mx)), max(0, int(y-my))
            crop = im[y0:min(H, int(y+h+my)), x0:min(W, int(x+w+mx))]
            plt.figure(figsize=(5.5, 7.5)); plt.imshow(crop); plt.axis("off")
            plt.title(f"{s['region'].upper()}  {s['file_name']}", fontsize=9)
            col = "cyan" if s["region"] == "front" else "orange"
            for name, (px, py) in s["points"].items():
                plt.scatter([px-x0], [py-y0], c=col, s=160, marker="x", linewidths=2.5)
                plt.annotate(name, (px-x0, py-y0), color="yellow", fontsize=9, weight="bold",
                             xytext=(5, 4), textcoords="offset points")
            out = os.path.join(args.viz, f"{s['region']}_{s['image_id']}_{s['ann_id']}.png")
            plt.savefig(out, dpi=92, bbox_inches="tight"); plt.close()
            done[s["region"]] += 1
            print("viz", out)


if __name__ == "__main__":
    main()
