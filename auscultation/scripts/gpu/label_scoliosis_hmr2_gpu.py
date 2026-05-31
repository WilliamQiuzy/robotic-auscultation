import os,glob,numpy as np,cv2,torch,functools
torch.load=functools.partial(torch.load,weights_only=False)   # 信任HMR2.0权重
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset
def cam_crop_to_full(cb,bc,bs2,isz,fl):
    iw,ih=isz[:,0],isz[:,1]; cx,cy=bc[:,0],bc[:,1]; w2,h2=iw/2.,ih/2.; bs=bs2*cb[:,0]+1e-9
    return torch.stack([(2*(cx-w2)/bs)+cb[:,1],(2*(cy-h2)/bs)+cb[:,2],2*fl/bs],dim=-1)
BACK={"R1":6457,"L1":3006,"R2":6310,"L2":2850,"R3":4732,"L3":1251,"R4":6543,"L4":3097}; IDX=list(BACK.values())
dev="cuda" if torch.cuda.is_available() else "cpu"; download_models(CACHE_DIR_4DHUMANS)
model,cfg=load_hmr2(DEFAULT_CHECKPOINT); model=model.to(dev).eval()
imgs=sorted(glob.glob("/workspace/train/*.JPG")+glob.glob("/workspace/train/*.jpg")); pairs=[]
for p in imgs:
    if len(pairs)>=6: break
    img=cv2.imread(p); H,W=img.shape[:2]
    if W>H*0.95: continue
    box=np.array([[W*0.05,H*0.05,W*0.95,H*0.95]])      # 整图(略内缩)当bbox: 单人居中
    ds=ViTDetDataset(cfg,img,box); batch=recursive_to(next(iter(torch.utils.data.DataLoader(ds,batch_size=1))),dev)
    with torch.no_grad(): out=model(batch)
    verts=out["pred_vertices"][0].cpu().numpy()
    isz=batch["img_size"].float(); sfl=float(cfg.EXTRA.FOCAL_LENGTH/cfg.MODEL.IMAGE_SIZE*isz.max())
    camt=cam_crop_to_full(out["pred_cam"],batch["box_center"].float(),batch["box_size"].float(),isz,sfl).cpu().numpy()[0]
    P=verts[IDX]+camt; u=sfl*P[:,0]/P[:,2]+W/2; w=sfl*P[:,1]/P[:,2]+H/2
    ov=img[:,:,::-1].copy()
    for x,y in zip(u,w): cv2.drawMarker(ov,(int(x),int(y)),(255,150,0),cv2.MARKER_TILTED_CROSS,max(16,H//28),3)
    pairs.append((img[:,:,::-1],ov,os.path.basename(p)[:14])); print("done",len(pairs))
n=len(pairs); fig,ax=plt.subplots(n,2,figsize=(9,5*n))
if n==1: ax=ax[None,:]
for i,(o,g,nm) in enumerate(pairs):
    ax[i,0].imshow(o);ax[i,0].set_title(nm,fontsize=9);ax[i,0].axis("off"); ax[i,1].imshow(g);ax[i,1].set_title("HMR2.0 SMPL fit -> 8 pts",fontsize=9);ax[i,1].axis("off")
plt.suptitle("scoliosis: SMPL-fit(HMR2.0,no-detector) -> 8 back pts",fontsize=12)
plt.tight_layout();plt.savefig("/workspace/scoliosis_hmr2.png",dpi=90,bbox_inches="tight");plt.close(); print("SAVED",n)
