#!/usr/bin/env python3
"""解剖结构合理性评估(自动, 无需金标): 真实后背点位配置是否合规。
4 指标(对每张图, 越好=越合规):
  on_body%   : 点落在后背前景(IUV part1)内的比例 (高好)
  mono%      : R1..R5 / L1..L5 纵向严格递降的相邻对比例 (高好)
  pair_err%  : |y_Ri - y_Li| / H 均值, 每层左右应同高 (低好)
  sym_err%   : |(spine-Rx)-(Lx-spine)| / W 均值, 左右应关于脊柱等距 (低好); spine=该y后背mask中线
用法: python scripts/anat_plausibility.py --ckpt outputs/perc_v7_multiview/best.pt --imgs /workspace/lab_back
"""
import argparse, glob, os, sys
import numpy as np, cv2
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.abspath(os.path.join(HERE,".."))
for p in ("ausc","ausc/models","ausc/training"): sys.path.insert(0,os.path.join(ROOT,p))
from dataset import IMAGENET_MEAN, IMAGENET_STD
from eval import readout_soft, _forward_maps
from infer_overlay import load_model, center_square, landmark_addresses
from tasks import task_landmarks
import torch

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt",required=True); ap.add_argument("--imgs",required=True)
    ap.add_argument("--task",default="lung"); ap.add_argument("--encoder",default="dinov2_vitb14")
    ap.add_argument("--res",type=int,default=518); ap.add_argument("--tau",type=float,default=0.01)
    ap.add_argument("--tta",default="1.0,1.14,1.27"); ap.add_argument("--up_bias",type=float,default=0.12)
    ap.add_argument("--smpl",default="data/smpl/SMPL_NEUTRAL.pkl")
    ap.add_argument("--uv",default="data/densepose_coco/uv_data/UV_Processed.mat")
    args=ap.parse_args()
    device="cpu" if os.environ.get("FORCE_CPU") else "cuda"
    tta=tuple(float(x) for x in args.tta.split(",")) if args.tta else None
    res=args.res; gy,gx=np.mgrid[0:res,0:res].astype(np.float32)
    m=load_model(args.ckpt,args.encoder,device)
    addr,_=landmark_addresses(args.uv,args.smpl)
    names=[n for n in task_landmarks(args.task) if n in addr]
    files=sorted(glob.glob(os.path.join(args.imgs,"*")))
    files=[f for f in files if f.lower().endswith((".jpg",".jpeg",".png"))]
    agg={"on_body":[],"mono":[],"pair":[],"sym":[]}
    for f in files:
        raw=cv2.imread(f)
        if raw is None: continue
        crop=center_square(raw[:,:,::-1],args.up_bias)
        im=cv2.resize(crop,(res,res),interpolation=cv2.INTER_AREA)
        x=torch.from_numpy(((im/255.0-IMAGENET_MEAN)/IMAGENET_STD).transpose(2,0,1)[None]).float().to(device)
        part_prob,pred_uv=_forward_maps(m,x,res,tta)
        part=part_prob.argmax(0)
        backmask=(part==1)
        P={}
        for nm in names:
            p=readout_soft(part_prob[addr[nm][0]],pred_uv,addr[nm],gx,gy,args.tau)
            if p is not None: P[nm]=(float(p[0]),float(p[1]))
        # on_body
        ob=[]
        for nm,(x0,y0) in P.items():
            xi,yi=int(round(x0)),int(round(y0))
            ob.append(1.0 if (0<=yi<res and 0<=xi<res and backmask[yi,xi]) else 0.0)
        on_body=100*np.mean(ob) if ob else 0
        # mono per side
        def mono(side):
            ys=[P[f"{side}{i}"][1] for i in range(1,6) if f"{side}{i}" in P]
            if len(ys)<2: return None
            return np.mean([1.0 if ys[i+1]>ys[i] else 0.0 for i in range(len(ys)-1)])
        ms=[v for v in (mono("R"),mono("L")) if v is not None]
        mono_pct=100*np.mean(ms) if ms else 0
        # pair err
        pe=[]
        for i in range(1,6):
            r,l=f"R{i}",f"L{i}"
            if r in P and l in P: pe.append(abs(P[r][1]-P[l][1])/res)
        pair=100*np.mean(pe) if pe else 0
        # sym err: spine = per-row median x of backmask
        se=[]
        for i in range(1,6):
            r,l=f"R{i}",f"L{i}"
            if r in P and l in P:
                yy=int(round((P[r][1]+P[l][1])/2))
                row=np.where(backmask[min(max(yy,0),res-1)])[0]
                if row.size:
                    spine=float(np.median(row))
                    se.append(abs((spine-P[r][0])-(P[l][0]-spine))/res)
        sym=100*np.mean(se) if se else 0
        agg["on_body"].append(on_body); agg["mono"].append(mono_pct); agg["pair"].append(pair); agg["sym"].append(sym)
        print(f"{os.path.basename(f)} on_body={on_body:.0f}% mono={mono_pct:.0f}% pair_err={pair:.2f}% sym_err={sym:.2f}%")
    if agg["on_body"]:
        mo=np.mean(agg["on_body"]); mm=np.mean(agg["mono"]); mp=np.mean(agg["pair"]); msy=np.mean(agg["sym"])
        print("MEAN on_body=%.0f%% mono=%.0f%% pair_err=%.2f%% sym_err=%.2f%%  (on_body/mono high=good, errs low=good)"%(mo,mm,mp,msy))
if __name__=="__main__": main()
