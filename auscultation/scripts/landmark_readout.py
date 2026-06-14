#!/usr/bin/env python3
"""landmark 头读出库: 载入带 landmark 头的 ckpt, 从热图 soft-argmax 出听诊点。
绕开有偏 V 场——点位直接从专用热图通道读。供 lab 验证/翻转一致性/anat 复用。"""
import os, sys
import numpy as np, cv2, torch
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.abspath(os.path.join(HERE,".."))
for p in ("ausc","ausc/models","ausc/training"): sys.path.insert(0,os.path.join(ROOT,p))
from iuv_model import DenseIUVModel, apply_lora
from dataset import IMAGENET_MEAN, IMAGENET_STD
from infer_overlay import center_square
from tasks import LANDMARK_INDEX

BACK_NAMES=["R1","L1","R2","L2","R3","L3","R4","L4","R5","L5"]
FRONT_NAMES=["aortic","pulmonary","erb","tricuspid","mitral"]

def load_landmark_model(ckpt_path, encoder="dinov2_vitb14", device="cuda"):
    ck=torch.load(ckpt_path,map_location=device)
    n_lm=ck.get("n_landmarks",15); hm_res=ck.get("hm_res",129)
    m=DenseIUVModel(encoder=encoder,frozen_encoder=False,n_landmarks=n_lm).to(device)
    apply_lora(m,r=ck.get("lora_rank",16),alpha=ck.get("lora_alpha",32)); m.to(device)
    sd=ck.get("lora_head")
    m.load_state_dict(sd,strict=False); m.eval()
    return m, hm_res

@torch.no_grad()
def readout_landmarks(model, im_rgb_crop, res, hm_res, device, names=BACK_NAMES, win=8, temp=0.1):
    """im_rgb_crop: 已 center_square 的 RGB。返回 {name:(x,y)} in res 空间(窗口 soft-argmax 亚像素)。
    win=8/temp=0.1 经 lab 后背翻转一致性扫描选优(1.218% vs win4/0.05 的 1.279%)。"""
    im=cv2.resize(im_rgb_crop,(res,res),interpolation=cv2.INTER_AREA)
    x=((im/255.0-IMAGENET_MEAN)/IMAGENET_STD).transpose(2,0,1)[None].astype(np.float32)
    x=torch.from_numpy(x).to(device)
    with torch.autocast("cuda",dtype=torch.float16):
        hm=model.predict_landmark_heatmap(x,(hm_res,hm_res))[0].float().cpu().numpy()  # (K,h,w)
    pts={}
    sc=res/hm_res
    for nm in names:
        c=LANDMARK_INDEX[nm]; h=hm[c]
        iy,ix=np.unravel_index(int(h.argmax()),h.shape)
        y0,y1=max(0,iy-win),min(hm_res,iy+win+1); x0,x1=max(0,ix-win),min(hm_res,ix+win+1)
        patch=h[y0:y1,x0:x1].astype(np.float64)
        patch=np.exp((patch-patch.max())/temp); patch/=patch.sum()+1e-9
        gy=np.arange(y0,y1)[:,None]; gx=np.arange(x0,x1)[None,:]
        cy=(patch*gy).sum(); cx=(patch*gx).sum()
        pts[nm]=(float(cx*sc),float(cy*sc))
    return pts
