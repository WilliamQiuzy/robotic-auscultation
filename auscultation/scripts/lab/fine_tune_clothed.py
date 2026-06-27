"""Phase-2a clothed fine-tune (5090 torch venv). Warm-start perc_lmh_v4; train landmark head +
LoRA on 22 real clothed Orbbec labels (heavy aug: affine crop-jitter + flip + color + graphic
overlays) mixed with the bare gold set. Saves perc_lmh_v5_clothed."""
import os, sys, json, glob, random, numpy as np, cv2, torch
ROOT = os.path.expanduser("~/Desktop/wam-vla-medical/auscultation")
os.chdir(ROOT); sys.path[:0] = ["ausc", "ausc/models", "ausc/training", "scripts"]
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from iuv_model import DenseIUVModel, apply_lora
from dataset import IMAGENET_MEAN, IMAGENET_STD
from train_landmark_human import make_heatmaps, weighted_hm_loss, heatmap_xy, ORDER, IDX

RES, HM, DEV = 518, 518 // 4, "cuda"
CLOTHED = os.path.expanduser("~/ausc_test/clothed_train")
BARE_DIR = "data/lab_sample_data/_GOLD_ALL"
V4 = "outputs/perc_lmh_v4/best.pt"
OUT = "outputs/perc_lmh_v5_clothed"; os.makedirs(OUT, exist_ok=True)
rng = random.Random(0)


def color_jitter(im):
    im = cv2.convertScaleAbs(im, alpha=rng.uniform(0.7, 1.3), beta=rng.uniform(-25, 25))
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + rng.randint(-12, 12)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.7, 1.3), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def graphic_overlay(im):
    for _ in range(rng.randint(1, 3)):
        c = tuple(int(x) for x in np.random.randint(0, 255, 3))
        cx, cy = rng.randint(80, 438), rng.randint(80, 438)
        ax, ay = rng.randint(20, 120), rng.randint(20, 120)
        ov = im.copy()
        cv2.ellipse(ov, (cx, cy), (ax, ay), rng.randint(0, 180), 0, 360, c, -1)
        im = cv2.addWeighted(ov, rng.uniform(0.4, 0.85), im, 1 - rng.uniform(0.4, 0.85), 0)
    return im


def affine(im, pts):
    ang = rng.uniform(-9, 9); sc = rng.uniform(0.82, 1.18)
    tx, ty = rng.uniform(-0.07, 0.07) * RES, rng.uniform(-0.07, 0.07) * RES
    M = cv2.getRotationMatrix2D((RES / 2, RES / 2), ang, sc); M[0, 2] += tx; M[1, 2] += ty
    im2 = cv2.warpAffine(im, M, (RES, RES), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    p2 = (M @ np.c_[pts, np.ones(len(pts))].T).T
    return im2, p2


class PointDS(Dataset):
    def __init__(self, items, clothed):
        self.items, self.clothed = items, clothed

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, pdict = self.items[i]
        im = cv2.imread(path)[:, :, ::-1]
        im = cv2.resize(im, (RES, RES), interpolation=cv2.INTER_AREA)
        pts = np.array([pdict[nm] for nm in ORDER], np.float32)
        im, pts = affine(im, pts)                          # crop-jitter
        if rng.random() < 0.5:                             # hflip + L/R swap
            im = im[:, ::-1].copy(); pts[:, 0] = RES - 1 - pts[:, 0]
            pts = pts[[IDX["L" + nm[1:]] if nm[0] == "R" else IDX["R" + nm[1:]] for nm in ORDER]]
        im = color_jitter(im)
        if self.clothed and rng.random() < 0.6:
            im = graphic_overlay(im)
        p = {nm: (float(pts[IDX[nm], 0]), float(pts[IDX[nm], 1])) for nm in ORDER}
        x = ((im / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).astype(np.float32)
        hm, mask = make_heatmaps(p, HM, RES, sigma=2.5)
        return torch.from_numpy(x.copy()), torch.from_numpy(hm), torch.from_numpy(mask)


def load_clothed():
    d = json.load(open(f"{CLOTHED}/train_clothed.json"))
    return [(f"{CLOTHED}/canvas/{k}.png", v["points"]) for k, v in d.items() if k != "_meta"]


def load_bare():
    d = json.load(open(f"{BARE_DIR}/gold_all.json")); out = []
    for stem, rec in d.items():
        if stem == "_meta":
            continue
        pts = rec.get("points", rec)
        pp = {nm: (v["xy"] if isinstance(v, dict) else v) for nm, v in pts.items()}
        cand = glob.glob(f"{BARE_DIR}/canvas/{stem}.png")
        if cand and len(pp) >= 10:
            out.append((cand[0], {nm: pp[nm] for nm in ORDER}))
    return out


clothed = load_clothed(); bare = load_bare()
te = clothed[::6]; tr_clothed = [c for c in clothed if c not in te]   # hold out ~4 clothed
print(f"[data] clothed train={len(tr_clothed)} held-out={len(te)} bare={len(bare)}")
# oversample clothed x6 so it dominates despite small count
tr = ConcatDataset([PointDS(tr_clothed, True)] * 6 + [PointDS(bare, False)])

ck = torch.load(V4, map_location="cpu")
m = DenseIUVModel(encoder="dinov2_vitb14", frozen_encoder=False, n_landmarks=10)
apply_lora(m, r=ck["lora_rank"], alpha=ck["lora_alpha"]); m.load_state_dict(ck["lora_head"], strict=False)
m = m.to(DEV)
for n, p in m.named_parameters():
    p.requires_grad_(n.startswith("landmark_head.") or ".A." in n or ".B." in n)   # head + LoRA
head_p = [p for n, p in m.named_parameters() if n.startswith("landmark_head.") and p.requires_grad]
lora_p = [p for n, p in m.named_parameters() if (".A." in n or ".B." in n) and p.requires_grad]
print(f"[model] trainable head={sum(p.numel() for p in head_p)/1e6:.2f}M lora={sum(p.numel() for p in lora_p)/1e6:.2f}M")
opt = torch.optim.AdamW([{"params": head_p, "lr": 1e-3}, {"params": lora_p, "lr": 2e-4}], weight_decay=1e-4)
loader = DataLoader(tr, batch_size=8, shuffle=True, num_workers=6, drop_last=True)


@torch.no_grad()
def pck(items):
    m.eval(); errs = []
    for path, pdict in items:
        im = cv2.resize(cv2.imread(path)[:, :, ::-1], (RES, RES), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(((im / 255. - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None].astype(np.float32)).to(DEV)
        hl = m.predict_landmark_heatmap(x, (HM, HM)); xy = heatmap_xy(hl.float(), RES)[0].cpu().numpy()
        for nm in ORDER:
            errs.append(np.hypot(xy[IDX[nm], 0] - pdict[nm][0], xy[IDX[nm], 1] - pdict[nm][1]))
    e = np.array(errs); return (e < 0.05 * RES).mean(), np.median(e)


EP = 60
for ep in range(EP):
    m.train()
    for n, p in m.named_parameters():
        if n.startswith("encoder.") and not (".A." in n or ".B." in n):
            pass
    run = 0
    for x, hm, mask in loader:
        x, hm, mask = x.to(DEV), hm.to(DEV), mask.to(DEV)
        opt.zero_grad(); _pl, _uv, hl = m.forward_with_landmarks(x, (HM, HM))
        loss, _ = weighted_hm_loss(hl, hm, mask); loss.backward(); opt.step(); run += loss.item()
    if ep % 10 == 9 or ep == EP - 1:
        p5, med = pck(te); ptr5, _ = pck(tr_clothed)
        print(f"[ep{ep:02d}] loss {run/len(loader):.5f} | heldout PCK@5 {p5*100:.0f} med {med:.1f}px | train PCK@5 {ptr5*100:.0f}", flush=True)
torch.save({"kind": "lora", "lora_rank": ck["lora_rank"], "lora_alpha": ck["lora_alpha"],
            "lora_head": {k: v.detach().cpu() for k, v in m.state_dict().items()
                          if ".A." in k or ".B." in k or k.startswith("head.") or k.startswith("landmark_head.")},
            "order": ORDER, "note": "clothed fine-tune v5 (22 orbbec + bare, aug)"},
           f"{OUT}/best.pt")
print(f"[done] saved {OUT}/best.pt")
