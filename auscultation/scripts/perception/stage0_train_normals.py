"""Stage-0 (H200): train a GEOMETRY-ONLY detector = DINOv2 on the surface-NORMAL map -> 10 points,
on 3628 synthetic HuMMan samples. Normals are garment-invariant -> should generalize to any shirt.
Saves perc_lmh_v6_normals."""
import json, os, random, sys
from runtime_paths import activate_runtime

RUNTIME_ROLE = "h200_training"
PATHS = activate_runtime(RUNTIME_ROLE)
ROOT, DATA, OUTPUTS = str(PATHS.repo_root), str(PATHS.data_root), str(PATHS.output_root)

import numpy as np, cv2, torch
from torch.utils.data import Dataset, DataLoader
from iuv_model import DenseIUVModel, apply_lora
from dataset import IMAGENET_MEAN, IMAGENET_STD
from train_landmark_human import make_heatmaps, weighted_hm_loss, heatmap_xy, ORDER, IDX

RES, HM, DEV = 518, 518 // 4, "cuda"
S0 = os.path.join(OUTPUTS, "stage0")
OUT = os.path.join(OUTPUTS, "perc_lmh_v6_normals"); os.makedirs(OUT, exist_ok=True)
rng = random.Random(0)
recs = [json.loads(l) for l in open(f"{S0}/manifest.jsonl")]
# Subject-grouped split: hold out whole subjects (sid), so heldout PCK measures
# cross-body-shape generalization, not view-interpolation on already-seen bodies.
heldout_sids = set(sorted({r["sid"] for r in recs})[::12])
te = [r for r in recs if r["sid"] in heldout_sids]
tr = [r for r in recs if r["sid"] not in heldout_sids]
print(f"[data] train={len(tr)} heldout={len(te)} (heldout subjects={len(heldout_sids)}, unseen body shapes)")


def load_norm(stem):
    return np.load(f"{S0}/normal/{stem}.npy")          # HxWx3 uint8 (n*127+127)


class NormDS(Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        r = self.items[i]; nm_img = load_norm(r["stem"]).astype(np.float32)
        pts = np.array([r["points"][k] for k in ORDER], np.float32)
        # scale/translate jitter (normal directions unchanged)
        if rng.random() < 0.8:
            s = rng.uniform(0.85, 1.15); tx = rng.uniform(-.06, .06) * RES; ty = rng.uniform(-.06, .06) * RES
            M = cv2.getRotationMatrix2D((RES/2, RES/2), 0, s); M[0, 2] += tx; M[1, 2] += ty
            nm_img = cv2.warpAffine(nm_img, M, (RES, RES), borderValue=(127, 127, 127))
            pts = (M @ np.c_[pts, np.ones(len(pts))].T).T
        if rng.random() < 0.5:                           # hflip: x-flip + L/R swap + flip nx channel
            nm_img = nm_img[:, ::-1].copy(); nm_img[..., 0] = 254 - nm_img[..., 0]
            pts[:, 0] = RES - 1 - pts[:, 0]
            pts = pts[[IDX["L"+k[1:]] if k[0] == "R" else IDX["R"+k[1:]] for k in ORDER]]
        nm_img += rng.gauss(0, 4)                         # sensor-noise sim
        p = {k: (float(pts[IDX[k], 0]), float(pts[IDX[k], 1])) for k in ORDER}
        x = ((np.clip(nm_img, 0, 255) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).astype(np.float32)
        hm, mask = make_heatmaps(p, HM, RES, sigma=2.5)
        return torch.from_numpy(x.copy()), torch.from_numpy(hm), torch.from_numpy(mask)


ck = torch.load(os.path.join(OUTPUTS, "perc_lmh_v4/best.pt"), map_location="cpu")
m = DenseIUVModel(encoder="dinov2_vitb14", frozen_encoder=False, n_landmarks=10)
apply_lora(m, r=ck["lora_rank"], alpha=ck["lora_alpha"]); m.load_state_dict(ck["lora_head"], strict=False)
m = m.to(DEV)
for n, p in m.named_parameters():
    p.requires_grad_(n.startswith("landmark_head.") or ".A." in n or ".B." in n)
opt = torch.optim.AdamW([{"params": [p for n, p in m.named_parameters() if n.startswith("landmark_head.")], "lr": 1e-3},
                         {"params": [p for n, p in m.named_parameters() if ".A." in n or ".B." in n], "lr": 3e-4}], weight_decay=1e-4)
loader = DataLoader(NormDS(tr), batch_size=16, shuffle=True, num_workers=8, drop_last=True)


@torch.no_grad()
def pck(items):
    m.eval(); e = []
    for r in items:
        nm_img = load_norm(r["stem"]).astype(np.float32)
        x = torch.from_numpy(((nm_img/255.-IMAGENET_MEAN)/IMAGENET_STD).transpose(2, 0, 1)[None].astype(np.float32)).to(DEV)
        xy = heatmap_xy(m.predict_landmark_heatmap(x, (HM, HM)).float(), RES)[0].cpu().numpy()
        for k in ORDER:
            e.append(np.hypot(xy[IDX[k], 0]-r["points"][k][0], xy[IDX[k], 1]-r["points"][k][1]))
    e = np.array(e); return (e < .05*RES).mean()*100, (e < .025*RES).mean()*100, np.median(e)


EP = 25
for ep in range(EP):
    m.train(); run = 0
    for x, hm, mask in loader:
        x, hm, mask = x.to(DEV), hm.to(DEV), mask.to(DEV)
        opt.zero_grad(); _, _, hl = m.forward_with_landmarks(x, (HM, HM))
        loss, _ = weighted_hm_loss(hl, hm, mask); loss.backward(); opt.step(); run += loss.item()
    if ep % 5 == 4 or ep == EP-1:
        p5, p25, med = pck(te)
        print(f"[ep{ep:02d}] loss {run/len(loader):.5f} | heldout PCK@5 {p5:.0f} PCK@2.5 {p25:.0f} med {med:.1f}px", flush=True)
torch.save({"kind": "lora", "lora_rank": ck["lora_rank"], "lora_alpha": ck["lora_alpha"],
            "lora_head": {k: v.detach().cpu() for k, v in m.state_dict().items()
                          if ".A." in k or ".B." in k or k.startswith("head.") or k.startswith("landmark_head.")},
            "order": ORDER, "note": "stage0 geometry/normals-input, 3628 synthetic HuMMan"}, f"{OUT}/best.pt")
print(f"[done] {OUT}/best.pt")
