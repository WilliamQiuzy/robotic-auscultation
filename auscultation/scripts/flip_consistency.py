#!/usr/bin/env python3
"""翻转一致性指标: 量化感知模型的左右对称鲁棒性(直接测 hflip-bug 是否修复)。
对每张真实后背图: 预测点 P_orig; 水平翻转图预测 P_flip; un-flip(x->res-1-x) 后,
对称模型应满足 P_flip[R_i] ~= unflip(P_orig[L_i]) (L/R 标签互换)。
误差 = 该距离均值 / res * 100 (%图宽)。越小越对称。V5崩塌->大, V6/V7->小。
用法: python scripts/flip_consistency.py --ckpt outputs/perc_v6_textbook/best.pt --imgs /workspace/lab_back
"""
import argparse, glob, os, sys
import numpy as np, cv2
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.abspath(os.path.join(HERE,".."))
for p in ("ausc","ausc/models","ausc/training"): sys.path.insert(0,os.path.join(ROOT,p))
from dataset import IMAGENET_MEAN, IMAGENET_STD
from eval import readout_soft, _forward_maps
from infer_overlay import load_model, center_square, landmark_addresses
from tasks import task_region, task_landmarks
import torch

def predict_points(m, im_rgb, res, tta, addr, names, gx, gy, tau, device):
    im=cv2.resize(im_rgb,(res,res),interpolation=cv2.INTER_AREA)
    x=torch.from_numpy(((im/255.0-IMAGENET_MEAN)/IMAGENET_STD).transpose(2,0,1)[None]).float().to(device)
    part_prob,pred_uv=_forward_maps(m,x,res,tta)
    pts={}
    for nm in names:
        p=readout_soft(part_prob[addr[nm][0]],pred_uv,addr[nm],gx,gy,tau)
        if p is not None: pts[nm]=(float(p[0]),float(p[1]))
    return pts

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
    def swap(nm): return ("L"+nm[1:]) if nm[0]=="R" else (("R"+nm[1:]) if nm[0]=="L" else nm)
    files=sorted(glob.glob(os.path.join(args.imgs,"*")))
    files=[f for f in files if f.lower().endswith((".jpg",".jpeg",".png"))]
    all_err=[]
    for f in files:
        raw=cv2.imread(f)
        if raw is None: continue
        crop=center_square(raw[:,:,::-1],args.up_bias)
        P_orig=predict_points(m,crop,res,tta,addr,names,gx,gy,args.tau,device)
        P_flip=predict_points(m,crop[:,::-1,:].copy(),res,tta,addr,names,gx,gy,args.tau,device)
        errs=[]
        for nm in names:
            sw=swap(nm)
            if nm in P_flip and sw in P_orig:
                ux=res-1-P_orig[sw][0]; uy=P_orig[sw][1]   # unflip(P_orig[swap])
                d=np.hypot(P_flip[nm][0]-ux,P_flip[nm][1]-uy)
                errs.append(d)
        if errs:
            e=float(np.mean(errs))/res*100
            all_err.append(e)
            print(f"{os.path.basename(f)}  flip_err={e:.2f}% (n={len(errs)})")
    if all_err:
        print(f"MEAN_FLIP_ERR {np.mean(all_err):.2f}% over {len(all_err)} imgs  (lower=better symmetry)")
if __name__=="__main__": main()
