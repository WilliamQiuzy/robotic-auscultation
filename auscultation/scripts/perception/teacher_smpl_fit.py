"""T7 — SMPL depth-fit auto-label teacher (appearance-independent). Fits SMPL (transl + global
orient + shape) to the Orbbec back point cloud, then reads the 10 V6A auscultation points off the
fitted mesh (rasterize IUV → LUNG_ADDR_V6A) → free pseudo-labels for UNLABELED clothed video.

RESULT (2026-06-27, validated vs the 22 dark MANUAL labels): this minimal depth-fit does NOT work
on the real clothed partial-back. Single-view front/back symmetry stays unresolved even with a
back-facing prior (back_facing=False), Chamfer 2–8 cm, and V6A reads out ~448 px off vs manual.
A usable teacher needs robust pose+disambiguation or a learned predictor (SAM-3D-Body, not on the
box). Payoff is gated on T6 (more unlabeled subjects, which don't exist yet) — DEFERRED. Kept as a
documented baseline; for now hand-labeling ~16 frames/subject (the v7 recipe) is cheaper + proven.

    python teacher_smpl_fit.py --session orbbec_session2 --labels clothed_labels_a.json --n 6
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np, cv2, torch

ROOT = os.environ.get("AUSC_REPO", os.path.expanduser("~/Desktop/wam-vla-medical/auscultation"))
DATA = os.path.expanduser(os.environ.get("AUSC_DATA", "~/ausc_test"))
sys.path[:0] = [ROOT, os.path.join(ROOT, "ausc"), os.path.join(ROOT, "scripts", "perception")]
import smplx
from humman_label import define_ausc_vertices, LUNG_ADDR_V6A
from render_iuv import load_uv_mapping, rasterize_iuv

DEV = "cuda"
ORDER = [f"{s}{i}" for i in range(1, 6) for s in ("R", "L")]


def back_cloud(depth, K, stride=4, band=0.25):
    """Orbbec depth → camera-frame back point cloud (m), body-band masked."""
    h, w = depth.shape; ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[::stride, ::stride].astype(np.float64) / 1000.0
    x = (xs - K[0, 2]) * z / K[0, 0]; y = (ys - K[1, 2]) * z / K[1, 1]
    P = np.dstack([x, y, z]).reshape(-1, 3); P = P[(z.ravel() > 0.3) & (z.ravel() < 3.0)]
    if len(P) < 50: return P
    d0 = np.median(P[:, 2]); P = P[np.abs(P[:, 2] - d0) < band]
    return P


def fit_smpl(model, cloud, back_idx, iters=250):
    """Fit transl + global_orient + betas so SMPL's surface matches the observed back cloud
    (one-sided Chamfer). Disambiguate the front/back symmetry by REQUIRING the back-torso vertices
    to face the camera (be the nearest, smallest-z surface). body_pose = rest."""
    cl = torch.tensor(cloud, dtype=torch.float32, device=DEV)
    if len(cl) > 2500:
        cl = cl[torch.randperm(len(cl), device=DEV)[:2500]]
    body = torch.zeros(1, 69, device=DEV)
    best = None
    for go_y in (0.0, np.pi):                                   # try front/back-facing init
        go = torch.tensor([[np.pi, go_y, 0.]], dtype=torch.float32, device=DEV, requires_grad=True)
        bz = torch.zeros(1, 10, device=DEV, requires_grad=True)
        tz = torch.tensor([[0., 0., float(cloud[:, 2].mean())]], device=DEV, requires_grad=True)
        opt = torch.optim.Adam([bz, tz, go], lr=0.02)
        for it in range(iters):
            opt.zero_grad()
            v = model(betas=bz, body_pose=body, global_orient=go, transl=tz).vertices[0]
            loss = torch.cdist(cl, v).min(1).values.mean() + 1e-3 * (bz ** 2).mean()
            loss.backward(); opt.step()
        with torch.no_grad():
            v = model(betas=bz, body_pose=body, global_orient=go, transl=tz).vertices[0]
            fit = torch.cdist(cl, v).min(1).values.mean().item()
            back_z = float(v[back_idx, 2].mean()); all_z = float(v[:, 2].mean())
            back_facing = back_z < all_z                       # back nearest camera ⇒ correct side
        score = fit + (0.0 if back_facing else 1.0)            # penalize wrong (front-facing) side
        if best is None or score < best[0]:
            best = (score, fit, bz.detach(), tz.detach(), go.detach(), back_facing)
    return best[1], best[2], best[3], best[4], best[5]          # chamfer, betas, transl, go, back_facing


def read_v6a(model, betas, transl, go, body, uvmap, K, H, W):
    with torch.no_grad():
        verts = model(betas=betas, body_pose=body, global_orient=go, transl=transl).vertices[0].cpu().numpy()
    I, U, V, _ = rasterize_iuv(verts, {"K": K, "R": np.eye(3), "T": np.zeros(3)}, uvmap, H, W)
    pts = {}
    for nm in ORDER:
        part, Ut, Vt = LUNG_ADDR_V6A[nm]; yy, xx = np.nonzero(I == part)
        if len(yy) == 0: continue
        j = ((U[yy, xx] - Ut) ** 2 + (V[yy, xx] - Vt) ** 2).argmin()
        pts[nm] = (int(xx[j]), int(yy[j]))
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="orbbec_session2")
    ap.add_argument("--labels", default="clothed_labels_a.json")
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()
    S = os.path.join(DATA, args.session); d = json.load(open(f"{S}/intrinsics.json"))
    K = np.array([[d["fx"], 0, d["cx"]], [0, d["fy"], d["cy"]], [0, 0, 1.0]]); H, W = d["height"], d["width"]
    lab = json.load(open(os.path.join(DATA, args.labels)))
    model = smplx.SMPL(model_path=os.path.join(ROOT, "data/smpl"), gender="neutral", batch_size=1).to(DEV)
    cpu_model = smplx.SMPL(model_path=os.path.join(ROOT, "data/smpl"), gender="neutral", batch_size=1)
    back_idx = list(define_ausc_vertices(cpu_model)["back"].values())   # back-torso vertex ids
    uvmap = load_uv_mapping(os.path.join(ROOT, "data/densepose_coco/uv_data/UV_Processed.mat"))
    body = torch.zeros(1, 69, device=DEV)
    cap = cv2.VideoCapture(f"{S}/rgb.mp4")
    items = [(s, r) for s, r in lab.items() if s != "_meta" and r.get("clip") == args.session][:args.n]
    errs, chamf = [], []
    for stem, rec in items:
        fr = rec["frame"]; depth = np.load(f"{S}/depth/d{fr:04d}.npy")
        cloud = back_cloud(depth, K)
        if len(cloud) < 200:
            print(f"  {stem}: too few cloud pts"); continue
        ch, betas, transl, go, bf = fit_smpl(model, cloud, back_idx)
        pts = read_v6a(model, betas, transl, go, body, uvmap, K, H, W)
        chamf.append(ch)
        e = [np.hypot(pts[nm][0] - rec["points"][nm][0], pts[nm][1] - rec["points"][nm][1])
             for nm in ORDER if nm in pts]
        errs.extend(e)
        med = np.median(e) if e else float("nan")
        print(f"  {stem} f{fr}: chamfer={ch*1000:.0f}mm back_facing={bf} v6a={len(pts)}/10 median px-err vs manual={med:.0f}")
    cap.release()
    if errs:
        e = np.array(errs)
        print(f"[teacher] {len(items)} frames | fit Chamfer median {np.median(chamf)*1000:.0f}mm | "
              f"V6A-vs-manual median {np.median(e):.0f}px  p90 {np.percentile(e,90):.0f}px")
        print("  (manual-label uncertainty itself is ~tens of px; compare accordingly)")


if __name__ == "__main__":
    main()
