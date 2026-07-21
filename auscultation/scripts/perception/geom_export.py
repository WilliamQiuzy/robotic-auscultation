"""Scaled geometry engine (H200): HuMMan GT-SMPL -> multi back-view -> depth + normals + IUV +
10 V6A points + (bare-shaded RGB). Produces the Stage-0 generalization dataset, zero labels.
Run: python geom_export.py <n_seqs> <n_views>"""
import glob, json, os, sys
from runtime_paths import activate_runtime

RUNTIME_ROLE = "h200_training"
PATHS = activate_runtime(RUNTIME_ROLE)
ROOT, DATA, OUTPUTS = str(PATHS.repo_root), str(PATHS.data_root), str(PATHS.output_root)

import numpy as np, cv2, trimesh, torch, smplx
from humman_label import define_ausc_vertices, LUNG_ADDR_V6A  # noqa
from render_iuv import load_uv_mapping, project               # noqa

NSEQ = int(sys.argv[1]) if len(sys.argv) > 1 else 40
NVIEW = int(sys.argv[2]) if len(sys.argv) > 2 else 4
RES, F = 518, 560.0
OUT = os.path.join(OUTPUTS, "stage0")
for d in ("depth", "normal", "rgb", "iuv"):
    os.makedirs(f"{OUT}/{d}", exist_ok=True)
model = smplx.SMPL(model_path=os.path.join(DATA, "smpl"), gender="neutral", batch_size=1)
uvmap = load_uv_mapping(os.path.join(DATA, "densepose_coco/uv_data/UV_Processed.mat"))
v2s, U, V, faces, fpart = uvmap
back8 = list(define_ausc_vertices(model)["back"].values())
ORDER = [f"{s}{i}" for i in range(1, 6) for s in ("R", "L")]
rng = np.random.RandomState(0)


def raster(verts, cam, H, W):
    """painter raster -> I,U,V,depth(m)."""
    dp = verts[v2s]; uv2d, dep = project(dp, cam)
    order = np.argsort(-dep[faces].mean(1))
    I = np.zeros((H, W), np.uint8); Um = np.zeros((H, W), np.float32)
    Vm = np.zeros((H, W), np.float32); D = np.zeros((H, W), np.float32)
    for fi in order:
        tri = uv2d[faces[fi]]
        if not np.isfinite(tri).all():
            continue
        (ax, ay), (bx, by), (cx, cy) = tri
        x0 = max(int(min(ax, bx, cx)), 0); x1 = min(int(np.ceil(max(ax, bx, cx))), W - 1)
        y0 = max(int(min(ay, by, cy)), 0); y1 = min(int(np.ceil(max(ay, by, cy))), H - 1)
        if x1 < x0 or y1 < y0:
            continue
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-9:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        l0 = ((by - cy) * (gx - cx) + (cx - bx) * (gy - cy)) / den
        l1 = ((cy - ay) * (gx - cx) + (ax - cx) * (gy - cy)) / den
        l2 = 1 - l0 - l1
        ins = (l0 >= -1e-4) & (l1 >= -1e-4) & (l2 >= -1e-4)
        if not ins.any():
            continue
        vi = faces[fi]; yy = gy[ins]; xx = gx[ins]
        I[yy, xx] = int(fpart[fi])
        Um[yy, xx] = l0[ins] * U[vi[0]] + l1[ins] * U[vi[1]] + l2[ins] * U[vi[2]]
        Vm[yy, xx] = l0[ins] * V[vi[0]] + l1[ins] * V[vi[1]] + l2[ins] * V[vi[2]]
        D[yy, xx] = l0[ins] * dep[faces[fi][0]] + l1[ins] * dep[faces[fi][1]] + l2[ins] * dep[faces[fi][2]]
    return I, Um, Vm, D


def normals_from_depth(D, K):
    H, W = D.shape; ys, xs = np.mgrid[0:H, 0:W]
    z = D; x = (xs - K[0, 2]) * z / K[0, 0]; y = (ys - K[1, 2]) * z / K[1, 1]
    P = np.dstack([x, y, z]); n = np.zeros_like(P)
    dx = P[1:-1, 2:] - P[1:-1, :-2]; dy = P[2:, 1:-1] - P[:-2, 1:-1]
    cr = np.cross(dx, dy); nn = np.linalg.norm(cr, axis=-1, keepdims=True)
    cr = cr / np.clip(nn, 1e-9, None); cr[cr[..., 2] > 0] *= -1
    n[1:-1, 1:-1] = cr; n[D == 0] = 0
    return n


def cv_lookat(C, t, up=np.array([0, 1.0, 0])):
    z = t - C; z /= np.linalg.norm(z); x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x); R = np.stack([x, y, z]); return R, -R @ C


def v6a(I, Um, Vm):
    out = {}
    for nm in ORDER:
        part, Ut, Vt = LUNG_ADDR_V6A[nm]; ys, xs = np.nonzero(I == part)
        if len(ys) == 0:
            continue
        j = ((Um[ys, xs] - Ut) ** 2 + (Vm[ys, xs] - Vt) ** 2).argmin()
        out[nm] = [int(xs[j]), int(ys[j])]
    return out


SEQS = sorted(glob.glob(os.path.join(DATA, "humman/smpl_params/*")))[:NSEQ]
K = np.array([[F, 0, RES / 2], [0, F, RES / 2], [0, 0, 1.0]])
man = open(f"{OUT}/manifest.jsonl", "w"); n = 0
for sq in SEQS:
    p = dict(np.load(os.path.join(sq, "smpl_params.npz"))); nf = p["betas"].shape[0]
    sid = os.path.basename(sq)
    for vi in range(NVIEW):
        fr = rng.randint(nf)
        with torch.no_grad():
            o = model(betas=torch.tensor(p["betas"][fr:fr+1], dtype=torch.float32),
                      body_pose=torch.tensor(p["body_pose"][fr:fr+1], dtype=torch.float32),
                      global_orient=torch.tensor(p["global_orient"][fr:fr+1], dtype=torch.float32),
                      transl=torch.tensor(p["transl"][fr:fr+1], dtype=torch.float32))
        verts = o.vertices[0].numpy(); mesh = trimesh.Trimesh(verts, model.faces, process=False)
        bn = mesh.vertex_normals[back8].mean(0); bn /= np.linalg.norm(bn); ctr = verts[back8].mean(0)
        az = np.radians(rng.uniform(-28, 28)); ca, sa = np.cos(az), np.sin(az)   # vary azimuth
        bn2 = np.array([bn[0] * ca + bn[2] * sa, bn[1], -bn[0] * sa + bn[2] * ca])
        dist = rng.uniform(0.95, 1.35)
        R, T = cv_lookat(ctr + bn2 * dist, ctr)
        I, Um, Vm, D = raster(verts, {"K": K, "R": R, "T": T}, RES, RES)
        pts = v6a(I, Um, Vm)
        if len(pts) < 10 or (I == 1).sum() < 3000:        # need full back visible
            continue
        nrm = normals_from_depth(D, K)
        stem = f"{sid}_v{vi}"
        np.save(f"{OUT}/depth/{stem}.npy", D.astype(np.float16))
        np.save(f"{OUT}/normal/{stem}.npy", (nrm * 127 + 127).astype(np.uint8))
        sh = np.clip((nrm @ np.array([0.3, 0.3, 0.9])) * 180 + 60, 0, 255).astype(np.uint8)  # shaded
        cv2.imwrite(f"{OUT}/rgb/{stem}.png", cv2.cvtColor(sh, cv2.COLOR_GRAY2BGR))
        man.write(json.dumps({"stem": stem, "points": pts, "sid": sid}) + "\n"); n += 1
    if n and n % 40 == 0:
        print(f"  {n} samples...", flush=True)
man.close()
print(f"WROTE {n} stage0 samples ({len(SEQS)} seqs x {NVIEW} views) -> {OUT}")
