import os,sys,json
import numpy as np, cv2
sys.path.insert(0,"ausc"); sys.path.insert(0,"ausc/training")
import humman_label as h
from infer_overlay import center_square
ADDR=h.LUNG_ADDR_V6A; res=518
ORDER=["R1","L1","R2","L2","R3","L3","R4","L4","R5","L5"]
mp=json.load(open("/tmp/v6a_full86_map.json"))
ROOT="data/dataset/scoliosis_canon_iuv"
mani={json.loads(l)["id"]:json.loads(l) for l in open(ROOT+"/manifest.jsonl")}
def v6a_read(iuvr):
    I=iuvr[:,:,0].astype(np.int32); U=iuvr[:,:,1]/255.; V=iuvr[:,:,2]/255.; pts={}
    for nm,(part,ut,vt) in ADDR.items():
        m=(I==part)
        if m.sum()<5: continue
        ys,xs=np.where(m); d=(U[ys,xs]-ut)**2+(V[ys,xs]-vt)**2; j=int(np.argmin(d))
        pts[nm]=(float(xs[j]),float(ys[j]))
    return pts
os.makedirs("/workspace/v6a_gold_FINAL/overlays",exist_ok=True)
gold={"_meta":{"schema":"posterior_lung_v6a_iuv_readout","standard":"THE auscultation gold; 10pt V6A read from HMR2-IUV via LUNG_ADDR_V6A","space":"center_square res=518; xy_norm=xy/518 (resolution-independent)","source":"86 scoliosis bare-back; tier from calibD_v8 review","gen":"regen_v6a_gold_full.py","date":"2026-06-11"}}
n_ok=0
for tid,info in mp.items():
    r=mani.get(info["scol"])
    if r is None: continue
    img=cv2.imread(os.path.join(ROOT,r["image"]))
    iuv=cv2.imread(os.path.join(ROOT,r["iuv"]))
    if img is None or iuv is None: continue
    ci=center_square(img[:,:,::-1],0.12); cu=center_square(iuv,0.12)
    iuvr=cv2.resize(cu,(res,res),interpolation=cv2.INTER_NEAREST)
    imr=cv2.resize(ci,(res,res),interpolation=cv2.INTER_AREA)
    pts=v6a_read(iuvr)
    gold[tid]={"scol":info["scol"],"tier":info["tier"],
               "points":{nm:{"xy_res":[round(pts[nm][0],1),round(pts[nm][1],1)],"xy_norm":[round(pts[nm][0]/res,4),round(pts[nm][1]/res,4)],"visible":True} for nm in ORDER if nm in pts}}
    ov=imr[:,:,::-1].copy()
    for nm in ORDER:
        if nm in pts:
            col=(40,40,230) if nm[0]=="R" else (230,90,40)
            cv2.circle(ov,(int(pts[nm][0]),int(pts[nm][1])),6,col,-1); cv2.putText(ov,nm,(int(pts[nm][0])+5,int(pts[nm][1])),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,255),1)
    tg=info["tier"][:4]
    cv2.imwrite("/workspace/v6a_gold_FINAL/overlays/%s_%s.png"%(tid,tg),ov); n_ok+=1
json.dump(gold,open("/workspace/v6a_gold_FINAL/v6a_gold_labels.json","w"),indent=1)
uc=sum(1 for k,v in gold.items() if isinstance(v,dict) and v.get("tier")=="manual_candidate")
print("regenerated %d/86 | usable(manual_candidate)=%d"%(n_ok,uc))
print("DONE")
