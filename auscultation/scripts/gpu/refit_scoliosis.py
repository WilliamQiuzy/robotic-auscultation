#!/usr/bin/env python3
"""脊柱侧弯裸背 — 干净重拟合(GPU, 4D-Humans/HMR2)。

修复旧版痛点: 旧脚本用"整图当bbox"(无检测器)→ 拟合飘/错位。
本版用 **YOLO(ultralytics) 检测人体得正确 bbox** → HMR2 拟合 → 保存**相机系 3D 顶点 verts_cam**
(供 ausc/datasets/scoliosis.py 光栅化稠密 IUV) + verts2d + wh + focal。

GPU 环境(vast.ai, 见项目记忆):
  conda env `hmr`: torch==2.0.1+cu117, pytorch-lightning==2.0.9, numpy<2, setuptools<70,
  4D-Humans(pip install -e . --no-build-isolation), ultralytics(替代 detectron2 当检测器),
  omegaconf/hydra-core/yacs/einops/smplx/timm/webdataset/pyrootutils。
  SMPL: 清洗版 SMPL_NEUTRAL.pkl 放 {CACHE_DIR_4DHUMANS}/data/smpl/SMPL_NEUTRAL.pkl。

用法: python refit_scoliosis.py <N张, 默认6预览; 传86跑全部>
输出: /root/scol_refit.pkl {name:{verts_cam:(6890,3),verts2d:(6890,2),wh:(W,H),focal,bbox}}
      + /root/refit_check.png(前6张 YOLO-bbox + HMR2 网格 叠加验证)
"""
import os, glob, sys, pickle, functools
import numpy as np, cv2, torch
torch.load = functools.partial(torch.load, weights_only=False)
from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset
from ultralytics import YOLO


def cam_crop_to_full(cb, bc, bs2, isz, fl):
    iw, ih = isz[:, 0], isz[:, 1]; cx, cy = bc[:, 0], bc[:, 1]; w2, h2 = iw / 2., ih / 2.
    bs = bs2 * cb[:, 0] + 1e-9
    return torch.stack([(2 * (cx - w2) / bs) + cb[:, 1], (2 * (cy - h2) / bs) + cb[:, 2], 2 * fl / bs], dim=-1)


dev = "cuda"
download_models(CACHE_DIR_4DHUMANS)
model, cfg = load_hmr2(DEFAULT_CHECKPOINT); model = model.to(dev).eval()
yolo = YOLO("yolov8x.pt")


def person_bbox(img, H, W):
    r = yolo(img[:, :, ::-1], classes=[0], verbose=False)[0]      # class0=person
    if len(r.boxes) == 0:
        return np.array([[W * 0.05, H * 0.05, W * 0.95, H * 0.95]])
    b = r.boxes.xyxy.cpu().numpy(); bb = b[((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])).argmax()]
    cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2; bw, bh = (bb[2] - bb[0]) * 1.15, (bb[3] - bb[1]) * 1.15
    return np.array([[cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]])


def fit(path):
    img = cv2.imread(path); H, W = img.shape[:2]
    box = person_bbox(img, H, W)
    ds = ViTDetDataset(cfg, img, box)
    batch = recursive_to(next(iter(torch.utils.data.DataLoader(ds, batch_size=1))), dev)
    with torch.no_grad():
        out = model(batch)
    verts = out["pred_vertices"][0].cpu().numpy()
    isz = batch["img_size"].float(); sfl = float(cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * isz.max())
    camt = cam_crop_to_full(out["pred_cam"], batch["box_center"].float(), batch["box_size"].float(), isz, sfl).cpu().numpy()[0]
    vc = verts + camt
    u = sfl * vc[:, 0] / vc[:, 2] + W / 2; v = sfl * vc[:, 1] / vc[:, 2] + H / 2
    return dict(verts_cam=vc.astype(np.float32), verts2d=np.stack([u, v], 1).astype(np.float32),
                wh=(W, H), focal=sfl, bbox=box[0].astype(np.float32))


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    imgs = sorted(glob.glob("/root/scol86/*"))[:limit]
    out = {}; pairs = []
    for p in imgs:
        nm = os.path.basename(p); out[nm] = fit(p)
        if len(pairs) < 6:
            img = cv2.imread(p)[:, :, ::-1].copy(); H, W = img.shape[:2]
            for (x, y) in out[nm]["verts2d"][::8]:
                if 0 <= x < W and 0 <= y < H:
                    cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), -1)
            pairs.append((img, nm[:14]))
        print("fit", nm)
    pickle.dump(out, open("/root/scol_refit.pkl", "wb"))
    print("SAVED", len(out))
