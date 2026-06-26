#!/usr/bin/env python3
"""生产模型(landmark 头)金标评估。读出 = landmark 热图 argmax+窗口质心 → 结构化(脊柱中线+对称宽)。
对人工金标算 逐点误差% / PCK@10 / PCK@5 + 结构指标(R/L 不平行% / 层间距不均%)。
这是评估【生产管线 deploy_landmark.py / perc_lmh_v4】的正确脚本(eval_gold.py 是旧 IUV-地址法, 勿混)。

用法:
  python scripts/eval_landmark_gold.py --ckpt outputs/perc_lmh_v4/best.pt \
      --gold gold_test/gold_points.json --canvas_root gold_test/canvas [--struct 1]
canvas_root 下需有 <stem>.png (center_square 518); gold = {stem:{points:{R1:[x,y],...}}}。
"""
import argparse, glob, json, os, sys
import numpy as np, cv2, torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
for p in ("ausc", "ausc/models", "ausc/training"):
    sys.path.insert(0, os.path.join(ROOT, p))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from iuv_model import DenseIUVModel, apply_lora                          # noqa: E402
from dataset import IMAGENET_MEAN, IMAGENET_STD                          # noqa: E402
from train_landmark_human import heatmap_xy, structure_points, ORDER, IDX  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/perc_lmh_v4/best.pt")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--canvas_root", required=True)
    ap.add_argument("--encoder", default="dinov2_vitb14")
    ap.add_argument("--res", type=int, default=518)
    ap.add_argument("--struct", type=int, default=1, help="1=结构化读出(生产默认)")
    ap.add_argument("--dump_pred", default=None)
    args = ap.parse_args()
    device = "cpu" if os.environ.get("FORCE_CPU") else "cuda"
    hm_res = args.res // 4
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    m = DenseIUVModel(encoder=args.encoder, frozen_encoder=False, n_landmarks=len(ORDER)).to(device)
    apply_lora(m, r=ck["lora_rank"], alpha=ck["lora_alpha"])
    m.load_state_dict(ck["lora_head"], strict=False); m.to(device).eval()
    gold = json.load(open(args.gold))
    errs_all = []; par = []; unev = []; dump = {}
    print(f"{'image':22} {'err%':>6} {'PCK10':>6} {'PCK5':>6}")
    for stem in [k for k in gold if k != "_meta"]:
        f = glob.glob(os.path.join(args.canvas_root, f"{stem}.png"))
        if not f:
            continue
        im = cv2.resize(cv2.imread(f[0])[:, :, ::-1], (args.res, args.res), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(((im / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None].astype(np.float32)).to(device)
        with torch.no_grad():
            hl = m.predict_landmark_heatmap(x, (hm_res, hm_res))
        xy = heatmap_xy(hl.float(), args.res)[0].cpu().numpy()
        if args.struct:
            xy = structure_points(xy)
        g = gold[stem]["points"]
        e = np.array([np.hypot(xy[IDX[nm], 0] - g[nm][0], xy[IDX[nm], 1] - g[nm][1]) for nm in ORDER if nm in g])
        errs_all += list(e)
        dump[stem] = {nm: [round(float(xy[IDX[nm], 0]), 1), round(float(xy[IDX[nm], 1]), 1)] for nm in ORDER}
        ysR = [xy[IDX[f"R{i}"]][1] for i in range(1, 6)]; ysL = [xy[IDX[f"L{i}"]][1] for i in range(1, 6)]
        span = (max(ysR + ysL) - min(ysR + ysL)) or 1
        par.append(np.mean([abs(ysR[i] - ysL[i]) for i in range(5)]) / span * 100)
        gg = np.mean([[ys[i + 1] - ys[i] for i in range(4)] for ys in (ysR, ysL)], axis=0)
        unev.append(np.std(gg) / np.mean(gg) * 100)
        print(f"{stem.replace('scol_IMG_','IMG').replace('20260605_','lab'):22} "
              f"{e.mean()/args.res*100:6.2f} {100*(e<0.1*args.res).mean():6.0f} {100*(e<0.05*args.res).mean():6.0f}")
    e = np.array(errs_all)
    print(f"\nMEAN err={e.mean()/args.res*100:.2f}%  PCK@10={100*(e<0.1*args.res).mean():.0f}  "
          f"PCK@5={100*(e<0.05*args.res).mean():.0f}  | R/L不平行={np.mean(par):.1f}%  层间距不均={np.mean(unev):.1f}%")
    if args.dump_pred:
        json.dump(dump, open(args.dump_pred, "w"), indent=2); print(f"-> {args.dump_pred}")


if __name__ == "__main__":
    main()
