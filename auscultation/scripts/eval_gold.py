#!/usr/bin/env python3
"""部署域金标准评估: 模型预测点 vs 人工金标点(解剖正确性, 独立于IUV自一致)。
gold 与预测同在 center_square res=518 画布空间。
指标(每图, 对有金标的点): mean_err%(/res), PCK@5%, PCK@10%(阈值=0.05/0.10*res)。
用法: python scripts/eval_gold.py --ckpt outputs/perc_v7_multiview/best.pt --imgs /workspace/lab_back --gold data/lab_sample_data/_gold/gold_points.json"""
import argparse, glob, os, sys, json
import numpy as np, cv2
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.abspath(os.path.join(HERE,".."))
for p in ("ausc","ausc/models","ausc/training"): sys.path.insert(0,os.path.join(ROOT,p))
from dataset import IMAGENET_MEAN, IMAGENET_STD
from eval import readout_soft, _forward_maps
from infer_overlay import load_model, center_square, landmark_addresses
from tasks import task_landmarks
from readout_fallback import readout_with_fallback
import torch

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt",required=True); ap.add_argument("--imgs",required=True); ap.add_argument("--gold",required=True)
    ap.add_argument("--task",default="lung"); ap.add_argument("--encoder",default="dinov2_vitb14")
    ap.add_argument("--res",type=int,default=518); ap.add_argument("--tau",type=float,default=0.01)
    ap.add_argument("--tta",default="1.0,1.14,1.27"); ap.add_argument("--up_bias",type=float,default=0.12)
    ap.add_argument("--smpl",default="data/smpl/SMPL_NEUTRAL.pkl")
    ap.add_argument("--uv",default="data/densepose_coco/uv_data/UV_Processed.mat")
    ap.add_argument("--fallback",type=int,default=0,help="1=部署兜底读出(夹回/V重展/lowfov), 0=纯readout_soft")
    ap.add_argument("--dump_pred",default=None,help="存预测点json(供 v7 vs v10 vs gold 叠图对比)")
    args=ap.parse_args()
    device="cpu" if os.environ.get("FORCE_CPU") else "cuda"
    tta=tuple(float(x) for x in args.tta.split(",")) if args.tta else None
    res=args.res; gy,gx=np.mgrid[0:res,0:res].astype(np.float32)
    gold=json.load(open(args.gold))
    m=load_model(args.ckpt,args.encoder,device)
    addr,_=landmark_addresses(args.uv,args.smpl)
    names=[n for n in task_landmarks(args.task) if n in addr]
    files=sorted(glob.glob(os.path.join(args.imgs,"*")))
    files=[f for f in files if f.lower().endswith((".jpg",".jpeg",".png"))]
    allerr=[]; pck5=[]; pck10=[]; dump={}
    for f in files:
        stem=os.path.splitext(os.path.basename(f))[0]
        if stem not in gold: continue
        gpts=gold[stem]["points"]; conf=gold[stem].get("conf","")
        raw=cv2.imread(f)
        if raw is None: continue
        crop=center_square(raw[:,:,::-1],args.up_bias)
        im=cv2.resize(crop,(res,res),interpolation=cv2.INTER_AREA)
        x=torch.from_numpy(((im/255.0-IMAGENET_MEAN)/IMAGENET_STD).transpose(2,0,1)[None]).float().to(device)
        part_prob,pred_uv=_forward_maps(m,x,res,tta)
        if args.fallback:
            fb,_fl,_mt=readout_with_fallback(part_prob,pred_uv,addr,names,gx,gy,args.tau,back_part=1)
            pred={nm:(fb[nm][0],fb[nm][1]) for nm in fb}
        else:
            pred={nm:readout_soft(part_prob[addr[nm][0]],pred_uv,addr[nm],gx,gy,args.tau) for nm in names}
        dump[stem]={nm:[round(pred[nm][0],1),round(pred[nm][1],1)] for nm in names if pred.get(nm) is not None}
        errs=[]
        for nm in names:
            if nm not in gpts or pred.get(nm) is None: continue
            d=np.hypot(pred[nm][0]-gpts[nm][0],pred[nm][1]-gpts[nm][1]); errs.append(d)
        if errs:
            errs=np.array(errs)
            me=errs.mean()/res*100; p5=100*(errs<0.05*res).mean(); p10=100*(errs<0.10*res).mean()
            allerr.append(me); pck5.append(p5); pck10.append(p10)
            print(f"{stem} [{conf}] mean_err={me:.2f}% PCK@5%={p5:.0f} PCK@10%={p10:.0f} (n={len(errs)})")
    if allerr:
        print(f"MEAN mean_err={np.mean(allerr):.2f}% PCK@5%={np.mean(pck5):.0f} PCK@10%={np.mean(pck10):.0f}  (err low=good, PCK high=good)")
    if args.dump_pred:
        json.dump(dump,open(args.dump_pred,"w"),indent=2); print(f"[dump] preds -> {args.dump_pred}")
if __name__=="__main__": main()
