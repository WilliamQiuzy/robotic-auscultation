"""Stage-1 (H200): co-train v6_normals on synthetic(3628) + 16 real DARK normal-maps (garment-blind),
test on 6 held-out real dark frames. Proves real-adapt reduces sim2real bias+scatter WITHOUT new labels.
v6 baseline on the same 6 held-out is printed for comparison."""
import json, os, random, re, sys
from runtime_paths import activate_runtime, require_h200_raw_release

RUNTIME_ROLE = "h200_training"
PATHS = activate_runtime(RUNTIME_ROLE)
ROOT, DATA, OUTPUTS = str(PATHS.repo_root), str(PATHS.data_root), str(PATHS.output_root)
REAL_DATA = str(require_h200_raw_release(
    "AUSC_PERCEPTION_LABELED_RELEASE", required_files=("clothed_labels_a.json",),
))

def _checkpoint_dir(outputs, environ):
    """AUSC_OUT names the checkpoint directory under AUSC_OUTPUT (default perc_lmh_v7_adapt), so
    AUSC_OCCLUSION=1 AUSC_OUT=perc_lmh_v9_occ writes v9 instead of overwriting v7_adapt, which
    stage2 warm-starts from. Bare name only: the output root itself is validated by runtime_paths."""
    name = environ.get("AUSC_OUT") or "perc_lmh_v7_adapt"
    if name in (".", "..") or not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise SystemExit("AUSC_OUT must be a bare checkpoint directory name under AUSC_OUTPUT, got %r" % name)
    return os.path.join(outputs, name)
OUT = _checkpoint_dir(OUTPUTS, os.environ)   # resolved before training so a bad value fails fast

import numpy as np, cv2, torch
from torch.utils.data import Dataset,DataLoader
from tracking.geometry import back_square_crop,_pad_extract,Intrinsics
from iuv_model import DenseIUVModel, apply_lora
from dataset import IMAGENET_MEAN, IMAGENET_STD
from train_landmark_human import make_heatmaps, weighted_hm_loss, heatmap_xy, structure_points, ORDER, IDX
RES,HM,DEV="",129,"cuda"; RES=518
rng=random.Random(0)
np_rng=np.random.RandomState(0)
from ausc.augment import augment_normal_channel
def normals(D,fx=560,fy=560,cx=259,cy=259):
    H,W=D.shape; ys,xs=np.mgrid[0:H,0:W]; z=D; x=(xs-cx)*z/fx; y=(ys-cy)*z/fy
    P=np.dstack([x,y,z]); n=np.zeros_like(P); dx=P[1:-1,2:]-P[1:-1,:-2]; dy=P[2:,1:-1]-P[:-2,1:-1]
    cr=np.cross(dx,dy); cr/=np.clip(np.linalg.norm(cr,axis=-1,keepdims=True),1e-9,None); cr[cr[...,2]>0]*=-1
    n[1:-1,1:-1]=cr; n[D==0]=0; return n
# ---- real dark frames -> crop normal maps + crop-space pts
lab=json.load(open(os.path.join(REAL_DATA, "clothed_labels_a.json")))
S=REAL_DATA; d=json.load(open(f"{S}/intrinsics.json"))
intr=Intrinsics(d["fx"],d["fy"],d["cx"],d["cy"],d["width"],d["height"],1.0)
cap=cv2.VideoCapture(f"{S}/rgb.mp4"); real=[]
items=[(s,r) for s,r in lab.items() if s!="_meta"]
items.sort(key=lambda kv: kv[1]["frame"])
for stem,rec in items:
    fr=rec["frame"]; cap.set(1,fr); ok,bgr=cap.read()
    if not ok: continue
    depth=np.load(f"{S}/depth/d{fr:04d}.npy")
    im,crop=back_square_crop(bgr[:,:,::-1],depth,intr)
    dc=cv2.resize(_pad_extract(depth,crop.x0,crop.y0,crop.side),(RES,RES),interpolation=cv2.INTER_NEAREST).astype(np.float32)/1000.
    nrm=np.clip(normals(dc)*127+127,0,255).astype(np.uint8)
    pts={nm:crop.full_to_det(np.array(rec["points"][nm],np.float32)[None])[0] for nm in ORDER}
    real.append({"nrm":nrm,"pts":{nm:(float(pts[nm][0]),float(pts[nm][1])) for nm in ORDER}})
cap.release()
# fixed split: every 4th -> test (6), rest train (16)
te_idx=set(range(0,len(real),4)); te_real=[real[i] for i in range(len(real)) if i in te_idx]; tr_real=[real[i] for i in range(len(real)) if i not in te_idx]
print(f"[real] train={len(tr_real)} test={len(te_real)}")
# ---- synthetic
S0=os.path.join(OUTPUTS, "stage0"); recs=[json.loads(l) for l in open(f"{S0}/manifest.jsonl")]
def load_norm(stem): return np.load(f"{S0}/normal/{stem}.npy")
_OCC = os.environ.get("AUSC_OCCLUSION") == "1"   # T2: arm-occlusion domain randomization
def aug(nm_img,pts):
    if rng.random()<0.85:
        s=rng.uniform(0.8,1.2); tx=rng.uniform(-.07,.07)*RES; ty=rng.uniform(-.07,.07)*RES; rot=rng.uniform(-8,8)
        M=cv2.getRotationMatrix2D((RES/2,RES/2),rot,s); M[0,2]+=tx; M[1,2]+=ty
        nm_img=cv2.warpAffine(nm_img,M,(RES,RES),borderValue=(127,127,127),flags=cv2.INTER_NEAREST)
        pts=(M@np.c_[pts,np.ones(len(pts))].T).T
    if rng.random()<0.5:
        nm_img=nm_img[:,::-1].copy(); nm_img[...,0]=254-nm_img[...,0]; pts[:,0]=RES-1-pts[:,0]
        pts=pts[[IDX["L"+k[1:]] if k[0]=="R" else IDX["R"+k[1:]] for k in ORDER]]
    if _OCC and rng.random()<0.6:                # gray occluder rectangle (simulated arm)
        rw=int(rng.uniform(0.15,0.45)*RES); rh=int(rng.uniform(0.15,0.45)*RES)
        x0=rng.randint(0,RES-rw); y0=rng.randint(0,RES-rh); nm_img[y0:y0+rh,x0:x0+rw]=127
    nm_img=augment_normal_channel(nm_img.astype(np.float32), np_rng)   # per-pixel depth-normal noise (audit fix)
    return nm_img,pts
class DS(Dataset):
    def __init__(self,syn,realtr,kreal=60):
        self.syn=syn; self.real=realtr*kreal
    def __len__(self): return len(self.syn)+len(self.real)
    def __getitem__(self,i):
        if i<len(self.syn):
            r=self.syn[i]; nm_img=load_norm(r["stem"]).astype(np.float32); pts=np.array([r["points"][k] for k in ORDER],np.float32)
        else:
            r=self.real[i-len(self.syn)]; nm_img=r["nrm"].astype(np.float32); pts=np.array([r["pts"][k] for k in ORDER],np.float32)
        nm_img,pts=aug(nm_img,pts)
        p={k:(float(pts[IDX[k],0]),float(pts[IDX[k],1])) for k in ORDER}
        x=((np.clip(nm_img,0,255)/255.-IMAGENET_MEAN)/IMAGENET_STD).transpose(2,0,1).astype(np.float32)
        hm,mask=make_heatmaps(p,HM,RES,sigma=2.5)
        return torch.from_numpy(x.copy()),torch.from_numpy(hm),torch.from_numpy(mask)
ck=torch.load(os.path.join(OUTPUTS, "perc_lmh_v6_normals/best.pt"),map_location="cpu")
def fresh():
    m=DenseIUVModel(encoder="dinov2_vitb14",frozen_encoder=False,n_landmarks=10)
    apply_lora(m,r=ck["lora_rank"],alpha=ck["lora_alpha"]); m.load_state_dict(ck["lora_head"],strict=False); return m.to(DEV)
@torch.no_grad()
def evalr(m,items):
    m.eval(); e=[]
    for r in items:
        x=torch.from_numpy(((r["nrm"]/255.-IMAGENET_MEAN)/IMAGENET_STD).transpose(2,0,1)[None].astype(np.float32)).to(DEV)
        xy=structure_points(heatmap_xy(m.predict_landmark_heatmap(x,(HM,HM)).float(),RES)[0].cpu().numpy())
        for k in ORDER: e.append(np.hypot(xy[IDX[k],0]-r["pts"][k][0],xy[IDX[k],1]-r["pts"][k][1]))
    e=np.array(e); return np.mean(e<.05*RES)*100,np.mean(e<.10*RES)*100,np.median(e)
m=fresh(); print("v6 baseline on 6 held-out dark: PCK@5=%.0f PCK@10=%.0f med=%.1f"%evalr(m,te_real))
for n,p in m.named_parameters(): p.requires_grad_(n.startswith("landmark_head.") or ".A." in n or ".B." in n)
opt=torch.optim.AdamW([{"params":[p for n,p in m.named_parameters() if n.startswith("landmark_head.")],"lr":5e-4},
                       {"params":[p for n,p in m.named_parameters() if ".A." in n or ".B." in n],"lr":1.5e-4}],weight_decay=1e-4)
loader=DataLoader(DS(recs,tr_real),batch_size=16,shuffle=True,num_workers=8,drop_last=True)
EP=8
for ep in range(EP):
    m.train(); run=0
    for x,hm,mask in loader:
        x,hm,mask=x.to(DEV),hm.to(DEV),mask.to(DEV); opt.zero_grad()
        _,_,hl=m.forward_with_landmarks(x,(HM,HM)); loss,_=weighted_hm_loss(hl,hm,mask); loss.backward(); opt.step(); run+=loss.item()
    p5,p10,md=evalr(m,te_real)
    print(f"[ep{ep}] loss {run/len(loader):.5f} | held-out dark PCK@5={p5:.0f} PCK@10={p10:.0f} med={md:.1f}",flush=True)
os.makedirs(OUT, exist_ok=True)
torch.save({"kind":"lora","lora_rank":ck["lora_rank"],"lora_alpha":ck["lora_alpha"],
            "lora_head":{k:v.detach().cpu() for k,v in m.state_dict().items() if ".A." in k or ".B." in k or k.startswith("head.") or k.startswith("landmark_head.")},
            "order":ORDER,"note":"stage1 cotrain synthetic+16 real dark normals; occ=%s"%_OCC},
           os.path.join(OUT, "best.pt"))
print("saved", OUT)
