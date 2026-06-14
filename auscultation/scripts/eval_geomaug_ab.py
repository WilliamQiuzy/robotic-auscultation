#!/usr/bin/env python3
"""增强留出 PCK A/B: 在【无泄漏的真后背留出集】上,对每张图分 正常/上背裁切/小远缩放 三档,
目标点随增强几何同步变换, 比较多个 ckpt 的 PCK@10 + 中位像素误差。

- 正常档 = 护栏(重训不该退步)。裁切/缩小档 = 重训要修的塌缩(v10 应 ≫ v7)。
- 每图增强用【固定种子(由 id 派生)】→ 各 ckpt 见到完全相同的增强 → A/B 公平。
- 目标点 = 留出集存的 HMR2-IUV V6A 点(= train 时同源 held-out 真值; 测的是"场在该取景下还原得准不准")。
- 读出两路: plain(纯 readout_soft, 隔离模型本身) + fallback(部署带兜底, 看用户实得)。

用法(GPU, 训练结束后跑):
  python scripts/eval_geomaug_ab.py --ckpts v7:outputs/perc_v7_multiview/best.pt,v10:outputs/perc_v10_geomaug/best.pt \
      --val outputs/fixed_val_v7.json --task lung --n 200 --tta 1.0,1.14,1.27
"""
import argparse, json, os, sys, zlib
import numpy as np, cv2, torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for p in ("ausc", "ausc/models", "ausc/training"):
    sys.path.insert(0, os.path.join(ROOT, p))
from dataset import IMAGENET_MEAN, IMAGENET_STD, _upper_crop, _shrink  # noqa: E402
from eval import readout_soft, _forward_maps                          # noqa: E402
from infer_overlay import load_model, landmark_addresses              # noqa: E402
from tasks import task_landmarks                                      # noqa: E402
from readout_fallback import readout_with_fallback                    # noqa: E402


def _seed(sid: str) -> int:
    return zlib.crc32(sid.encode()) & 0xFFFFFFFF


def _load(row, res):
    """读 img(RGB)+iuv, resize 到 res, 点缩放到 res 空间。返回 img,iuv,pts{name:(x,y,vis)}。"""
    root = row["_root"]
    img = cv2.imread(os.path.join(root, row["image"]))[:, :, ::-1]
    iuv = cv2.imread(os.path.join(root, row["iuv"]))
    h0, w0 = img.shape[:2]
    img = cv2.resize(img, (res, res), interpolation=cv2.INTER_AREA)
    iuv = cv2.resize(iuv, (res, res), interpolation=cv2.INTER_NEAREST)
    sx, sy = res / float(w0), res / float(h0)
    pts = {nm: (float(p["xy"][0]) * sx, float(p["xy"][1]) * sy, bool(p["visible"]))
           for nm, p in (row.get("points") or {}).items()}
    return img, iuv, pts


def _normalize(img):
    x = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1)[None].copy())


def _pck(pred, tgt, res, thr=0.10, fallback=False, part_prob=None, pred_uv=None,
         addr=None, names=None, gx=None, gy=None, tau=0.01):
    """pred: {name:(x,y)} 或 None(用 fallback 现读). 对每个可见目标点判 dist<thr*res。"""
    T = thr * res
    hits = 0; tot = 0; errs = []
    if fallback:
        fb, _flags, _meta = readout_with_fallback(part_prob, pred_uv, addr, names, gx, gy, tau, back_part=1)
        pred = {nm: (fb[nm][0], fb[nm][1]) for nm in fb}
    for nm, (tx, ty, vis) in tgt.items():
        if not vis:
            continue
        tot += 1
        if nm not in pred:
            continue
        d = float(np.hypot(pred[nm][0] - tx, pred[nm][1] - ty))
        errs.append(d)
        if d < T:
            hits += 1
    return hits, tot, errs


def _readout_plain(part_prob, pred_uv, addr, names, gx, gy, tau):
    out = {}
    for nm in names:
        p = readout_soft(part_prob[addr[nm][0]], pred_uv, addr[nm], gx, gy, tau)
        if p is not None:
            out[nm] = (float(p[0]), float(p[1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True, help="逗号分隔 name:path")
    ap.add_argument("--val", default="outputs/fixed_val_v10.json",
                    help="须用锚当前 LUNG_ADDR_V6A 的 val(rebuild_val_targets.py 生成); 旧 v6/v7/v8/v9 因 R/L swap 失效")
    ap.add_argument("--task", default="lung")
    ap.add_argument("--encoder", default="dinov2_vitb14")
    ap.add_argument("--res", type=int, default=518)
    ap.add_argument("--tau", type=float, default=0.01)
    ap.add_argument("--tta", default="1.0,1.14,1.27")
    ap.add_argument("--n", type=int, default=200, help="留出真后背子集大小(0=全部)")
    ap.add_argument("--smpl", default="data/smpl/SMPL_NEUTRAL.pkl")
    ap.add_argument("--uv", default="data/densepose_coco/uv_data/UV_Processed.mat")
    ap.add_argument("--out", default="outputs/geomaug_ab.json")
    args = ap.parse_args()
    device = "cpu" if os.environ.get("FORCE_CPU") else "cuda"
    res = args.res
    gy, gx = np.mgrid[0:res, 0:res].astype(np.float32)
    tta = tuple(float(x) for x in args.tta.split(",")) if args.tta else None

    rows = json.load(open(args.val))
    backs = [r for r in rows if r.get("region") == "back" and not str(r.get("id", "")).startswith("syn")]
    backs.sort(key=lambda r: r["id"])
    if args.n and args.n < len(backs):
        idx = np.linspace(0, len(backs) - 1, args.n).astype(int)   # 均匀取子集(确定性)
        backs = [backs[i] for i in idx]
    addr, _ = landmark_addresses(args.uv, args.smpl)
    names = [n for n in task_landmarks(args.task) if n in addr]
    print(f"[eval] 真后背留出 {len(backs)} 张 × 3档 × {len(args.ckpts.split(','))}模型 | points={names}")

    CONDS = ["normal", "crop", "shrink"]
    results = {}
    for spec in args.ckpts.split(","):
        name, path = spec.split(":", 1)
        m = load_model(path, args.encoder, device)
        agg = {c: {"plain": [0, 0, []], "fb": [0, 0, []]} for c in CONDS}
        for r in backs:
            img0, iuv0, pts0 = _load(r, res)
            rng_seed = _seed(r["id"])
            for cond in CONDS:
                if cond == "normal":
                    img, iuv, pts = img0, iuv0, pts0
                elif cond == "crop":
                    img, iuv, pts = _upper_crop(img0.copy(), iuv0.copy(), dict(pts0), res,
                                                np.random.RandomState(rng_seed))
                else:
                    img, iuv, pts = _shrink(img0.copy(), iuv0.copy(), dict(pts0), res,
                                            np.random.RandomState(rng_seed + 1))
                with torch.no_grad():
                    part_prob, pred_uv = _forward_maps(m, _normalize(img).to(device), res, tta)
                pred = _readout_plain(part_prob, pred_uv, addr, names, gx, gy, args.tau)
                h, t, e = _pck(pred, pts, res)
                agg[cond]["plain"][0] += h; agg[cond]["plain"][1] += t; agg[cond]["plain"][2] += e
                h2, t2, e2 = _pck(None, pts, res, fallback=True, part_prob=part_prob, pred_uv=pred_uv,
                                  addr=addr, names=names, gx=gx, gy=gy, tau=args.tau)
                agg[cond]["fb"][0] += h2; agg[cond]["fb"][1] += t2; agg[cond]["fb"][2] += e2
        results[name] = {c: {k: {"pck10": (v[0] / v[1] if v[1] else 0.0),
                                 "medpx": float(np.median(v[2])) if v[2] else None,
                                 "n": v[1]} for k, v in agg[c].items()} for c in CONDS}
        del m
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"  done {name}")

    json.dump(results, open(args.out, "w"), indent=2)
    # 打印对比表
    print("\n================ 增强留出 PCK@10 A/B ================")
    hdr = f"{'model':>6} {'readout':>7} | " + " | ".join(f"{c:>14}" for c in CONDS)
    print(hdr); print("-" * len(hdr))
    for name, res_c in results.items():
        for k in ("plain", "fb"):
            cells = []
            for c in CONDS:
                d = res_c[c][k]
                cells.append(f"{d['pck10']*100:5.1f}% / {('%4.1f'%d['medpx']) if d['medpx'] is not None else '  - '}px")
            print(f"{name:>6} {k:>7} | " + " | ".join(f"{x:>14}" for x in cells))
    print("\n列=正常(护栏)/裁切/缩小; 每格 PCK@10 / 中位px误差。plain=纯模型, fb=带部署兜底。")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
