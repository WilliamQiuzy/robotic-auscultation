#!/usr/bin/env python3
"""Gold-point annotation tool for deployment-domain lung auscultation eval.

Click the 10 posterior lung points IN ORDER on each canvas image:
    R1, L1, R2, L2, R3, L3, R4, L4, R5, L5
(当前约定 = 看图者视角: R = 图像【右】侧(红点), L = 图像【左】侧(绿点)。
 每层先点图右(R)再点图左(L)。Level 1 = 上肺尖, Level 5 = 下肺底。)

Controls:
    left-click  : place the next point (in the R1,L1,...,R5,L5 order shown in the title)
    u / backspace: undo last point
    n           : next image (saves current)
    r           : reset current image
    q           : save all & quit

The canvas images are the *model-input space* (center_square, 518x518), so the saved
gold coordinates are directly comparable to model predictions (see scripts/eval_gold.py).

For a clinically trustworthy gold, count ribs / use palpation landmarks rather than eyeballing —
a casual eyeball estimate is biased (tends to sit too high and too clustered).

Usage:
    python scripts/annotate_gold.py \
        --canvas data/lab_sample_data/_gold \
        --out data/lab_sample_data/_gold/gold_points.json \
        --annotator "your_name"
"""
import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ORDER = [f"{s}{i}" for i in range(1, 6) for s in ("R", "L")]


def annotate_image(path, existing):
    img = mpimg.imread(path)
    pts = dict(existing) if existing else {}
    state = {"done": False, "next_to_next": False}

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)

    def redraw():
        ax.clear()
        ax.imshow(img)
        for nm, (x, y) in pts.items():
            c = "red" if nm.startswith("R") else "lime"
            ax.plot(x, y, "o", color=c, ms=7)
            ax.annotate(nm, (x + 4, y), color=c, fontsize=8)
        placed = [n for n in ORDER if n in pts]
        nxt = next((n for n in ORDER if n not in pts), "ALL DONE")
        side = "图右(R)" if nxt.startswith("R") else ("图左(L)" if nxt.startswith("L") else "")
        ax.set_title(f"{os.path.basename(path)}  |  next: {nxt} {side}   "
                     f"({len(placed)}/10)  [n=next img, u=undo, r=reset, q=quit]")
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or event.xdata is None:
            return
        nxt = next((n for n in ORDER if n not in pts), None)
        if nxt is None:
            return
        pts[nxt] = [float(event.xdata), float(event.ydata)]
        redraw()

    def on_key(event):
        if event.key in ("u", "backspace"):
            placed = [n for n in ORDER if n in pts]
            if placed:
                del pts[placed[-1]]
                redraw()
        elif event.key == "r":
            pts.clear()
            redraw()
        elif event.key == "n":
            state["next_to_next"] = True
            plt.close(fig)
        elif event.key == "q":
            state["done"] = True
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    plt.show()
    return pts, state["done"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canvas", required=True, help="dir of *_canvas.png images")
    ap.add_argument("--out", required=True, help="gold_points.json path")
    ap.add_argument("--annotator", default="user")
    args = ap.parse_args()

    gold = {}
    if os.path.exists(args.out):
        gold = json.load(open(args.out))

    canvases = sorted(glob.glob(os.path.join(args.canvas, "*_canvas.png")))
    if not canvases:                                    # 回退: 普通 *.png 画布(本 testset 用)
        canvases = sorted(f for f in glob.glob(os.path.join(args.canvas, "*.png"))
                          if "_grid" not in f)
    if not canvases:
        raise SystemExit(f"no canvas png in {args.canvas} "
                         f"(run scripts/prep_gold_canvas.py first)")

    gold.setdefault("_meta", {})
    gold["_meta"].update({
        "space": "center_square res=518 canvas (== model input space)",
        "schema": "posterior_lung_standard_v6a (R1,L1..R5,L5)",
        "annotator": args.annotator,
    })

    for path in canvases:
        stem = os.path.basename(path).replace("_canvas.png", "").replace(".png", "")
        existing = gold.get(stem, {}).get("points")
        pts, quit_now = annotate_image(path, existing)
        if pts:
            gold[stem] = {"conf": f"{args.annotator}", "points": pts}
            json.dump(gold, open(args.out, "w"), indent=2)
            print(f"saved {stem}: {len(pts)}/10 points")
        if quit_now:
            break

    print(f"gold written -> {args.out}")


if __name__ == "__main__":
    main()
