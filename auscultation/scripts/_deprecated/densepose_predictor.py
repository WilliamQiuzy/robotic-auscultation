#!/usr/bin/env python3
"""
Task 2: Detectron2 DensePose 预测器封装。

作用: 对【任意图】(含机器人自采临床图) 输出逐像素 (I,U,V)，从而把听诊点
      (ausc_points 里的固定 (I,U,V)) 映射到该图，无需 GT 标注。
      → batch_label.py 的 backend=predictor 用它。

⚠️ 跑在 GPU 服务器(项目的 RTX/L40 Docker)，不要在 macOS/CPU 上硬编译。
   本机仅持有代码与接口；导入被 guard，缺 detectron2 时给出安装指引。

【安装(GPU 服务器)】
  pip install 'git+https://github.com/facebookresearch/detectron2.git'
  pip install 'git+https://github.com/facebookresearch/detectron2.git#subdirectory=projects/DensePose'
  # 配置: detectron2 projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml
  # 权重: https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl

用法(服务器):
  python densepose_predictor.py --image robot_capture.jpg --region front --out preview.png \
      --config .../densepose_rcnn_R_50_FPN_s1x.yaml --weights .../model_final_162be9.pkl
"""
import argparse
import numpy as np

_DEFAULT_WEIGHTS = ("https://dl.fbaipublicfiles.com/densepose/"
                    "densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl")


class DensePosePredictor:
    def __init__(self, config, weights=_DEFAULT_WEIGHTS, device="cuda", score_thresh=0.8):
        try:
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
            from densepose import add_densepose_config
            from densepose.vis.extractor import DensePoseResultExtractor
        except ImportError as e:
            raise ImportError(
                f"detectron2 / densepose 未安装 ({e}). 见本文件头部【安装】，"
                "请在 GPU 服务器上安装后再用。") from e
        cfg = get_cfg(); add_densepose_config(cfg); cfg.merge_from_file(config)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
        cfg.MODEL.WEIGHTS = weights
        cfg.MODEL.DEVICE = device
        self._predictor = DefaultPredictor(cfg)
        self._extract = DensePoseResultExtractor()

    def predict_largest(self, img_bgr):
        """返回最大检测人的 dict: {box:(x,y,w,h), labels:HxW uint8(0..24), u:HxW, v:HxW}。"""
        inst = self._predictor(img_bgr)["instances"]
        if len(inst) == 0:
            return None
        results, boxes_xywh = self._extract(inst)   # boxes_xywh: Nx4 tensor
        boxes = boxes_xywh.cpu().numpy()
        i = int(np.argmax(boxes[:, 2] * boxes[:, 3]))   # 面积最大
        r = results[i]
        return {"box": tuple(boxes[i].tolist()),
                "labels": r.labels.cpu().numpy(),
                "u": r.uv[0].cpu().numpy(), "v": r.uv[1].cpu().numpy()}


def map_point(dp, target, k=15):
    """逐像素 IUV 结果里，按 (I,U,V) 找最近像素 → 图像坐标 (px,py)。"""
    ti, tu, tv = target
    labels, u, v = dp["labels"], dp["u"], dp["v"]
    bx, by, _, _ = dp["box"]
    mask = labels == ti
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    dist = (u[ys, xs] - tu) ** 2 + (v[ys, xs] - tv) ** 2
    j = np.argsort(dist)[:max(1, k)]
    return float(bx + np.median(xs[j])), float(by + np.median(ys[j]))


def label_image(dp, region):
    """对一张图的 IUV 结果，按 region 映射所有听诊点 → {name:(px,py)}。"""
    import ausc_points as AP
    out = {}
    for name, target in AP.REGION_POINTS[region].items():
        r = map_point(dp, target)
        if r:
            out[name] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--region", choices=["front", "back"], default="front")
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", default=_DEFAULT_WEIGHTS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="preview.png")
    args = ap.parse_args()
    import cv2
    pred = DensePosePredictor(args.config, args.weights, args.device)
    img = cv2.imread(args.image)
    dp = pred.predict_largest(img)
    if dp is None:
        print("未检测到人"); return
    pts = label_image(dp, args.region)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.imshow(img[:, :, ::-1]); plt.axis("off")
    for name, (px, py) in pts.items():
        plt.scatter([px], [py], c="cyan", s=120, marker="x")
        plt.annotate(name, (px, py), color="yellow", fontsize=8)
    plt.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"{len(pts)} points -> {args.out}")


if __name__ == "__main__":
    main()
