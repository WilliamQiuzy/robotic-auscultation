#!/usr/bin/env python3
"""重建 held-out val 的听诊点目标, 锚到【当前 LUNG_ADDR_V6A】(本会话 R/L swap + L5=0.82 之后)。

为什么: 旧 val(fixed_val_v6/v7/v8/v9)的点目标是用【旧 V6A】在 HMR2 IUV 上读出的。本会话改了
LUNG_ADDR_V6A(R/L 交换 + L5 底点 0.82), 旧目标与当前读出系统性错位(尤其 R/L 互换)→ realback PCK
塌到 ~0.10, 训练选 ckpt 的依据失效。

做法(= 原始目标生成法, 纯 CPU 无需模型): 对每张 val 后背, 读它存的 HMR2 IUV png(part,U,V),
对每个 V6A 地址 (part,ut,vt) 做 soft-argmax(part 一热 × exp(-‖UV-地址‖²/τ))→ 该点在【原生图像坐标】
的位置, 写回 points。这样目标与当前标准一致, PCK 恢复有意义。

用法: python scripts/rebuild_val_targets.py --in outputs/fixed_val_v7.json --out outputs/fixed_val_v10.json
"""
import argparse, json, os, sys
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for p in ("ausc", "ausc/training"):
    sys.path.insert(0, os.path.join(ROOT, p))
from humman_label import LUNG_ADDR_V6A  # noqa: E402  当前唯一标准源


def readout_native(part, U, V, addr, gx, gy, tau=0.01, min_px=20):
    """在原生分辨率 IUV 上对单个地址 soft-argmax。part:HxW int; U,V:HxW float∈[0,1]。"""
    p, ut, vt = addr
    mask = (part == int(p)).astype(np.float32)
    if mask.sum() < min_px:
        return None
    A = mask * np.exp(-((U - ut) ** 2 + (V - vt) ** 2) / tau)
    s = float(A.sum())
    if s < 1e-8:
        return None
    return float((A * gx).sum() / s), float((A * gy).sum() / s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="outputs/fixed_val_v7.json")
    ap.add_argument("--out", default="outputs/fixed_val_v10.json")
    ap.add_argument("--tau", type=float, default=0.01)
    args = ap.parse_args()
    rows = json.load(open(args.inp))
    names = list(LUNG_ADDR_V6A.keys())
    print(f"[rebuild] {len(rows)} 行 | 地址源=humman_label.LUNG_ADDR_V6A | 点={names}")
    print("  示例地址 R1", LUNG_ADDR_V6A["R1"], "L1", LUNG_ADDR_V6A["L1"], "R5", LUNG_ADDR_V6A["R5"])

    out_rows = []
    n_back = 0; n_pts = 0
    for r in rows:
        r = dict(r)
        # 只重建后背点(肺); 前胸/其它保持原样
        if r.get("region") != "back":
            out_rows.append(r); continue
        n_back += 1
        iuv = cv2.imread(os.path.join(r["_root"], r["iuv"]))           # BGR: 0=part,1=U255,2=V255
        H, W = iuv.shape[:2]
        part = iuv[:, :, 0].astype(np.int32)
        U = iuv[:, :, 1].astype(np.float32) / 255.0
        V = iuv[:, :, 2].astype(np.float32) / 255.0
        gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)
        pts = {}
        for nm in names:
            xy = readout_native(part, U, V, LUNG_ADDR_V6A[nm], gx, gy, args.tau)
            if xy is None:
                pts[nm] = {"xy": [0.0, 0.0], "visible": False}
            else:
                pts[nm] = {"xy": [round(xy[0], 2), round(xy[1], 2)], "visible": True}
                n_pts += 1
        r["points"] = pts
        r["point_schema"] = "V6A_current"
        out_rows.append(r)
    json.dump(out_rows, open(args.out, "w"))
    print(f"[done] 重建 {n_back} 张后背 × 10点 ({n_pts} 可见) -> {args.out}")


if __name__ == "__main__":
    main()
