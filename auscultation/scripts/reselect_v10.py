#!/usr/bin/env python3
"""事后重选 v10 最佳 epoch: 训练期选 ckpt 用的是过期 val(fixed_val_v7, R/L swap 前)→ best.pt 是噪声。
这里用【锚当前 LUNG_ADDR_V6A 的 fixed_val_v10】对每个保存的 epXX.pt 重算 realback PCK@10, 选真正最佳。

用法: python scripts/reselect_v10.py --dir outputs/perc_v10_geomaug --val outputs/fixed_val_v10.json
"""
import argparse, glob, json, os, sys
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for p in ("ausc", "ausc/models", "ausc/training"):
    sys.path.insert(0, os.path.join(ROOT, p))
from eval import evaluate                                       # noqa: E402
from infer_overlay import load_model, landmark_addresses        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--val", default="outputs/fixed_val_v10.json")
    ap.add_argument("--encoder", default="dinov2_vitb14")
    ap.add_argument("--res", type=int, default=518)
    ap.add_argument("--tta", default="")                        # 选 ckpt 与训练一致: 无 TTA
    ap.add_argument("--n", type=int, default=300, help="子集大小(排名足够稳, 提速); 0=全部")
    ap.add_argument("--smpl", default="data/smpl/SMPL_NEUTRAL.pkl")
    ap.add_argument("--uv", default="data/densepose_coco/uv_data/UV_Processed.mat")
    args = ap.parse_args()
    device = "cpu" if os.environ.get("FORCE_CPU") else "cuda"
    tta = tuple(float(x) for x in args.tta.split(",")) if args.tta else None
    rows = json.load(open(args.val))
    backs = [r for r in rows if r.get("region") == "back" and not str(r.get("id", "")).startswith("syn")]
    backs.sort(key=lambda r: r["id"])
    if args.n and args.n < len(backs):
        idx = np.linspace(0, len(backs) - 1, args.n).astype(int)
        backs = [backs[i] for i in idx]
    addr, _ = landmark_addresses(args.uv, args.smpl)
    print(f"[reselect] {len(backs)} 真后背 held-out (fixed_val_v10) | TTA={tta}")

    ckpts = sorted(glob.glob(os.path.join(args.dir, "ep*.pt")))
    best = (-1, None)
    rows_out = []
    for ck in ckpts:
        m = load_model(ck, args.encoder, device)
        rd = evaluate(m, "", backs, addr, args.res, device, readout_mode="soft", tta_scales=tta)
        pck = rd.get("back", {}).get("pck@10%")
        med = rd.get("back", {}).get("med_px")
        ep = os.path.basename(ck)
        rows_out.append((ep, pck, med))
        print(f"  {ep}: realback PCK@10={pck:.4f}  med_px={med:.1f}")
        if pck is not None and pck > best[0]:
            best = (pck, ck)
        del m
        if device == "cuda":
            torch.cuda.empty_cache()
    if best[1]:
        import shutil
        dst = os.path.join(args.dir, "best_reselected.pt")
        shutil.copy(best[1], dst)
        print(f"\n[best] {os.path.basename(best[1])} PCK@10={best[0]:.4f} -> {dst}")
    json.dump(rows_out, open(os.path.join(args.dir, "reselect_v10.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
