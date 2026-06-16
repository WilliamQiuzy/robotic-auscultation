import json, os, sys, glob, re, wave
import numpy as np
from scipy.signal import butter, filtfilt
sys.path.insert(0,"/workspace/code/auscultation/ausc/audio"); sys.path.insert(0,"/workspace/OPERA")
import proxies as P
from src.benchmark.model_util import extract_opera_feature
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
CACHE="/workspace/data/audio/_q_v1"; TMP="/tmp/_q2win"; SR=16000
gold=json.load(open("/workspace/data/audio/_doctor/gold_merged.json"))
wavs=sorted(glob.glob(f"{TMP}/c*_w*.wav"), key=lambda p:(int(re.search(r"c(\d+)_",p).group(1)),int(re.search(r"_w(\d+)",p).group(1))))
owner=np.array([int(re.search(r"c(\d+)_",p).group(1)) for p in wavs])
cons=[g["consensus"] for g in gold]; idx=[i for i in range(len(gold)) if cons[i] in("usable","unusable")]
y=np.array([1 if cons[i]=="usable" else 0 for i in idx])
sig=[P.resample(*P.read_wav(w),SR)[0] for w in wavs]  # cache raw signals once
def feat_for_band(lo,hi):
    b,a=butter(4,[lo/(SR/2),hi/(SR/2)],btype="band"); bd=f"/tmp/_bp_{lo}_{hi}"; os.makedirs(bd,exist_ok=True)
    paths=[]
    for w,x in zip(wavs,sig):
        yy=filtfilt(b,a,x).astype(np.float32); pcm=(np.clip(yy/(np.max(np.abs(yy))+1e-9)*0.9,-1,1)*32767).astype("<i2")
        p=f"{bd}/{os.path.basename(w)}"; 
        with wave.open(p,"wb") as f: f.setnchannels(1);f.setsampwidth(2);f.setframerate(SR);f.writeframes(pcm.tobytes())
        paths.append(p)
    F=np.array(extract_opera_feature(paths,pretrain="operaCT",input_sec=8,dim=768))
    return np.array([F[owner==ci].mean(0) for ci in range(len(gold))])
def cvauc(X):
    return float(np.mean([roc_auc_score(y,cross_val_predict(make_pipeline(StandardScaler(),LogisticRegression(C=0.05,max_iter=5000)),X[idx],y,cv=StratifiedKFold(5,shuffle=True,random_state=sd),method="predict_proba")[:,1]) for sd in range(8)]))
res={"100-2000(cur)":0.794}
for lo,hi in [(50,2000),(100,1000),(150,2500),(80,1200)]:
    res[f"{lo}-{hi}"]=round(cvauc(feat_for_band(lo,hi)),3); print(f"band {lo}-{hi}: {res[f'{lo}-{hi}']:.3f}",flush=True)
print(f"[bandsweep done] {json.dumps(res)}",flush=True)
