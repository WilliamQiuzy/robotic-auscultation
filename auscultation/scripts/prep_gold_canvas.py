#!/usr/bin/env python3
"""为金标注备评估画布: 每张评估图按模型输入(center_square+resize to res)处理,
出 ① 干净画布(标注用) ② 叠坐标网格版(供肉眼读像素坐标做初版金标)。
gold 坐标即定义在这个 res-space 画布上 = 模型预测空间, 评估直接可比。"""
import argparse, glob, os, sys
import numpy as np, cv2
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.abspath(os.path.join(HERE,".."))
for p in ("ausc","ausc/models","ausc/training"): sys.path.insert(0,os.path.join(ROOT,p))
from infer_overlay import center_square

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--imgs",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--res",type=int,default=518); ap.add_argument("--up_bias",type=float,default=0.12)
    ap.add_argument("--grid",type=int,default=50)
    args=ap.parse_args()
    os.makedirs(args.out,exist_ok=True)
    files=sorted(glob.glob(os.path.join(args.imgs,"*")))
    files=[f for f in files if f.lower().endswith((".jpg",".jpeg",".png"))]
    for f in files:
        raw=cv2.imread(f)
        if raw is None: continue
        crop=center_square(raw[:,:,::-1],args.up_bias)
        im=cv2.resize(crop,(args.res,args.res),interpolation=cv2.INTER_AREA)[:,:,::-1].copy()  # BGR
        stem=os.path.splitext(os.path.basename(f))[0]
        cv2.imwrite(os.path.join(args.out,stem+"_canvas.png"),im)
        g=im.copy()
        for x in range(0,args.res,args.grid):
            cv2.line(g,(x,0),(x,args.res),(0,255,0),1)
            cv2.putText(g,str(x),(x+1,12),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,255,0),1)
        for y in range(0,args.res,args.grid):
            cv2.line(g,(0,y),(args.res,y),(0,255,0),1)
            cv2.putText(g,str(y),(1,y+12),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,255,0),1)
        cv2.imwrite(os.path.join(args.out,stem+"_grid.png"),g)
        print("canvas+grid:",stem, im.shape)
    print("PREP_DONE")
if __name__=="__main__": main()
