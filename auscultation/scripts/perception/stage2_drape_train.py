"""Stage-2 (5090): drape-hardening. Warm from v7, co-train on synthetic depth with PROCEDURAL
GARMENT DRAPE (global standoff + lower-back hem flare + fold noise) -> normals, plus the 16 real
dark frames. Labels stay at the true body location (V6A), so the model learns to find the landmark
UNDER loose-garment geometry. Goal: shrink the yellow low-back lateral over-spread. -> v8_drape."""
import os, sys, json, random, numpy as np, cv2, torch
ROOT = os.environ.get("AUSC_REPO", os.path.expanduser("~/Desktop/wam-vla-medical/auscultation")); os.chdir(ROOT)
sys.path[:0] = ["ausc/perception", "ausc", "ausc/models", "ausc/training", "scripts"]
from torch.utils.data import Dataset, DataLoader
from tracking.geometry import back_square_crop, _pad_extract, Intrinsics
from iuv_model import DenseIUVModel, apply_lora
from dataset import IMAGENET_MEAN, IMAGENET_STD
from train_landmark_human import make_heatmaps, weighted_hm_loss, heatmap_xy, structure_points, ORDER, IDX

RES, HM, DEV = 518, 129, "cuda"
S0 = os.path.expanduser(os.environ.get("AUSC_DATA","~/ausc_test")+"/stage0")
OUT = "outputs/perc_lmh_v8_drape"; os.makedirs(OUT, exist_ok=True)
rng = random.Random(0)
np_rng = np.random.RandomState(0)


def normals_k560(D):
    h, w = D.shape; ys, xs = np.mgrid[0:h, 0:w]; z = D
    x = (xs - 259) * z / 560.0; y = (ys - 259) * z / 560.0
    P = np.dstack([x, y, z]); n = np.zeros_like(P)
    dx = P[1:-1, 2:] - P[1:-1, :-2]; dy = P[2:, 1:-1] - P[:-2, 1:-1]
    cr = np.cross(dx, dy); cr /= np.clip(np.linalg.norm(cr, axis=-1, keepdims=True), 1e-9, None)
    cr[cr[..., 2] > 0] *= -1; n[1:-1, 1:-1] = cr; n[D == 0] = 0; return n


def drape(D):
    """Add procedural loose-garment displacement to a metric body-depth map (closer to camera)."""
    h, w = D.shape; body = D > 0
    standoff = np_rng.uniform(0.003, 0.015)                       # whole-shirt standoff
    yy = np.linspace(0, 1, h)[:, None].astype(np.float32)          # 0 top .. 1 bottom
    flare = np_rng.uniform(0.0, 0.05) * np.clip((yy - 0.45) / 0.55, 0, 1) ** 1.5  # hem flares below waist
    fs = np_rng.randint(6, 18)                                     # fold coarseness
    folds = cv2.resize(np_rng.standard_normal((fs, fs)).astype(np.float32), (w, h),
                       interpolation=cv2.INTER_CUBIC) * np_rng.uniform(0.002, 0.011)
    disp = (standoff + flare + folds).astype(np.float32)
    out = D.astype(np.float32).copy(); out[body] = D[body] - disp[body]
    out[out < 0.05] = 0
    return out


def drape_normal_img(stem):
    D = np.load(f"{S0}/depth/{stem}.npy").astype(np.float32)       # metres (float16 saved)
    if rng.random() < 0.85:                                        # 85% draped, 15% bare (keep both)
        D = drape(D)
    return np.clip(normals_k560(D) * 127 + 127, 0, 255).astype(np.uint8)


# ---- real dark frames (real drape already present) -> crop normals + crop pts ----
lab = json.load(open(os.path.expanduser(os.environ.get("AUSC_DATA","~/ausc_test")+"/clothed_labels_a.json")))
S = os.path.expanduser(os.environ.get("AUSC_DATA","~/ausc_test")+"/orbbec_session2"); dd = json.load(open(f"{S}/intrinsics.json"))
intr = Intrinsics(dd["fx"], dd["fy"], dd["cx"], dd["cy"], dd["width"], dd["height"], 1.0)
cap = cv2.VideoCapture(f"{S}/rgb.mp4"); real = []
items = sorted([(s, r) for s, r in lab.items() if s != "_meta" and r.get("clip") == "orbbec_session2"],
               key=lambda kv: kv[1]["frame"])
for stem, rec in items:
    fr = rec["frame"]; cap.set(1, fr); ok, bgr = cap.read()
    if not ok:
        continue
    depth = np.load(f"{S}/depth/d{fr:04d}.npy")
    im, crop = back_square_crop(bgr[:, :, ::-1], depth, intr)
    dc = cv2.resize(_pad_extract(depth, crop.x0, crop.y0, crop.side), (RES, RES),
                    interpolation=cv2.INTER_NEAREST).astype(np.float32) / 1000.
    nrm = np.clip(normals_k560(dc) * 127 + 127, 0, 255).astype(np.uint8)
    pts = {nm: crop.full_to_det(np.array(rec["points"][nm], np.float32)[None])[0] for nm in ORDER}
    real.append({"nrm": nrm, "pts": {nm: (float(pts[nm][0]), float(pts[nm][1])) for nm in ORDER}})
cap.release()
te_idx = set(range(0, len(real), 4))
te_real = [real[i] for i in range(len(real)) if i in te_idx]
tr_real = [real[i] for i in range(len(real)) if i not in te_idx]
print(f"[real] train={len(tr_real)} test={len(te_real)}")

recs = [json.loads(l) for l in open(f"{S0}/manifest.jsonl")]


def aug(nm_img, pts):
    if rng.random() < 0.85:
        s = rng.uniform(0.8, 1.2); tx = rng.uniform(-.07, .07) * RES; ty = rng.uniform(-.07, .07) * RES
        rot = rng.uniform(-8, 8); M = cv2.getRotationMatrix2D((RES / 2, RES / 2), rot, s)
        M[0, 2] += tx; M[1, 2] += ty
        nm_img = cv2.warpAffine(nm_img, M, (RES, RES), borderValue=(127, 127, 127), flags=cv2.INTER_NEAREST)
        pts = (M @ np.c_[pts, np.ones(len(pts))].T).T
    if rng.random() < 0.5:
        nm_img = nm_img[:, ::-1].copy(); nm_img[..., 0] = 254 - nm_img[..., 0]; pts[:, 0] = RES - 1 - pts[:, 0]
        pts = pts[[IDX["L" + k[1:]] if k[0] == "R" else IDX["R" + k[1:]] for k in ORDER]]
    nm_img = nm_img.astype(np.float32) + rng.gauss(0, 4)
    return nm_img, pts


class DS(Dataset):
    def __init__(self, syn, realtr, kreal=60):
        self.syn = syn; self.real = realtr * kreal

    def __len__(self): return len(self.syn) + len(self.real)

    def __getitem__(self, i):
        if i < len(self.syn):
            r = self.syn[i]; nm_img = drape_normal_img(r["stem"]).astype(np.float32)
            pts = np.array([r["points"][k] for k in ORDER], np.float32)
        else:
            r = self.real[i - len(self.syn)]; nm_img = r["nrm"].astype(np.float32)
            pts = np.array([r["pts"][k] for k in ORDER], np.float32)
        nm_img, pts = aug(nm_img, pts)
        p = {k: (float(pts[IDX[k], 0]), float(pts[IDX[k], 1])) for k in ORDER}
        x = ((np.clip(nm_img, 0, 255) / 255. - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).astype(np.float32)
        hm, mask = make_heatmaps(p, HM, RES, sigma=2.5)
        return torch.from_numpy(x.copy()), torch.from_numpy(hm), torch.from_numpy(mask)


ck = torch.load("outputs/perc_lmh_v7_adapt/best.pt", map_location="cpu")
m = DenseIUVModel(encoder="dinov2_vitb14", frozen_encoder=False, n_landmarks=10)
apply_lora(m, r=ck["lora_rank"], alpha=ck["lora_alpha"]); m.load_state_dict(ck["lora_head"], strict=False)
m = m.to(DEV)
for n, p in m.named_parameters():
    p.requires_grad_(n.startswith("landmark_head.") or ".A." in n or ".B." in n)
opt = torch.optim.AdamW([{"params": [p for n, p in m.named_parameters() if n.startswith("landmark_head.")], "lr": 5e-4},
                         {"params": [p for n, p in m.named_parameters() if ".A." in n or ".B." in n], "lr": 1.5e-4}], weight_decay=1e-4)
loader = DataLoader(DS(recs, tr_real), batch_size=16, shuffle=True, num_workers=8, drop_last=True)


@torch.no_grad()
def evalr(items):
    m.eval(); e = []
    for r in items:
        x = torch.from_numpy(((r["nrm"] / 255. - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None].astype(np.float32)).to(DEV)
        xy = structure_points(heatmap_xy(m.predict_landmark_heatmap(x, (HM, HM)).float(), RES)[0].cpu().numpy())
        for k in ORDER:
            e.append(np.hypot(xy[IDX[k], 0] - r["pts"][k][0], xy[IDX[k], 1] - r["pts"][k][1]))
    e = np.array(e); return np.mean(e < .05 * RES) * 100, np.mean(e < .10 * RES) * 100, np.median(e)


print("v7 (pre-drape) on 6 held-out dark: PCK@5=%.0f PCK@10=%.0f med=%.1f" % evalr(te_real))
EP = 8
for ep in range(EP):
    m.train(); run = 0
    for x, hm, mask in loader:
        x, hm, mask = x.to(DEV), hm.to(DEV), mask.to(DEV); opt.zero_grad()
        _, _, hl = m.forward_with_landmarks(x, (HM, HM)); loss, _ = weighted_hm_loss(hl, hm, mask)
        loss.backward(); opt.step(); run += loss.item()
    p5, p10, md = evalr(te_real)
    print(f"[ep{ep}] loss {run/len(loader):.5f} | held-out dark PCK@5={p5:.0f} PCK@10={p10:.0f} med={md:.1f}", flush=True)
torch.save({"kind": "lora", "lora_rank": ck["lora_rank"], "lora_alpha": ck["lora_alpha"],
            "lora_head": {k: v.detach().cpu() for k, v in m.state_dict().items()
                          if ".A." in k or ".B." in k or k.startswith("head.") or k.startswith("landmark_head.")},
            "order": ORDER, "note": "stage2 drape-hardened: v7 + procedural-drape synthetic + 16 real dark"},
           f"{OUT}/best.pt")
print("saved v8_drape")
