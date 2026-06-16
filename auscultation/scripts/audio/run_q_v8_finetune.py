import json, os, sys, glob, re, wave
import numpy as np
from scipy.signal import butter, filtfilt
sys.path.insert(0,"/workspace/code/auscultation/ausc/audio"); sys.path.insert(0,"/workspace/OPERA")
import proxies as P
from src.benchmark.model_util import extract_opera_feature
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, cross_val_predict, KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import spearmanr
import joblib
TMP="/tmp/_q2win"; SR=16000; OUT="/workspace/data/audio/_q_v2"; os.makedirs(OUT,exist_ok=True)
gold=json.load(open("/workspace/data/audio/_doctor/gold_merged.json"))
wavs=sorted(glob.glob(f"{TMP}/c*_w*.wav"), key=lambda p:(int(re.search(r"c(\d+)_",p).group(1)),int(re.search(r"_w(\d+)",p).group(1))))
owner=np.array([int(re.search(r"c(\d+)_",p).group(1)) for p in wavs])
sig=[P.resample(*P.read_wav(w),SR)[0] for w in wavs]
s=np.array([g["sungeun"] for g in gold]); j=np.array([g["jason"] for g in gold]); mean_ov=(s+j)/2
cons=[g["consensus"] for g in gold]; idx=[i for i in range(len(gold)) if cons[i] in("usable","unusable")]
y=np.array([1 if cons[i]=="usable" else 0 for i in idx])
strong=[i for i in range(len(gold)) if (s[i]>=2 and j[i]>=2) or (s[i]<=1 and j[i]<=1)]
ys=np.array([1 if (s[i]>=2 and j[i]>=2) else 0 for i in strong])
def feat(lo,hi):
    b,a=butter(4,[lo/(SR/2),hi/(SR/2)],btype="band"); bd=f"/tmp/_bp_{lo}_{hi}"; os.makedirs(bd,exist_ok=True); paths=[]
    for w,x in zip(wavs,sig):
        yy=filtfilt(b,a,x).astype(np.float32); pcm=(np.clip(yy/(np.max(np.abs(yy))+1e-9)*0.9,-1,1)*32767).astype("<i2")
        p=f"{bd}/{os.path.basename(w)}"
        with wave.open(p,"wb") as f: f.setnchannels(1);f.setsampwidth(2);f.setframerate(SR);f.writeframes(pcm.tobytes())
        paths.append(p)
    F=np.array(extract_opera_feature(paths,pretrain="operaCT",input_sec=8,dim=768))
    return np.array([F[owner==ci].mean(0) for ci in range(len(gold))])
LR=lambda: make_pipeline(StandardScaler(),LogisticRegression(C=0.05,max_iter=5000))
def cvauc(X,ix,yy): return float(np.mean([roc_auc_score(yy,cross_val_predict(LR(),X[ix],yy,cv=StratifiedKFold(5,shuffle=True,random_state=sd),method="predict_proba")[:,1]) for sd in range(8)]))
cache={}; res={}
for lo,hi in [(80,1200),(60,1000),(70,1100),(80,1000),(90,1100),(70,1200)]:
    X=feat(lo,hi); cache[(lo,hi)]=X; res[f"{lo}-{hi}"]=round(cvauc(X,idx,y),3); print(f"band {lo}-{hi}: {res[f'{lo}-{hi}']}",flush=True)
best=max(res,key=res.get); lo,hi=map(int,best.split("-")); X=cache[(lo,hi)]
sa=cvauc(X,strong,ys)
pred=np.mean([cross_val_predict(make_pipeline(StandardScaler(),Ridge(alpha=50.0)),X,mean_ov,cv=KFold(5,shuffle=True,random_state=sd)) for sd in range(8)],axis=0)
bu=np.array([1 if (s[i]>=2 and j[i]>=2) else 0 for i in range(len(gold))])
final=make_pipeline(StandardScaler(),CalibratedClassifierCV(LogisticRegression(C=0.05,max_iter=5000),method="isotonic",cv=5)).fit(X[idx],y)
joblib.dump({"model":final,"encoder":"operaCT","preprocess":f"bandpass_{lo}_{hi}","agg":"window_mean","win_s":8,"hop_s":4},f"{OUT}/q_v2.joblib")
rep={"sweep":res,"best_band":best,"clear_consensus_auc":res[best],"strong_agree_auc":round(sa,3),"soft_spearman":round(float(spearmanr(pred,mean_ov).correlation),3),"soft_auc":round(float(roc_auc_score(bu,pred)),3),"q_v1":0.72}
json.dump(rep,open(f"{OUT}/q_v2_report.json","w"),indent=2)
print(f"[Q v2 final @ {best}] {json.dumps(rep)}",flush=True)
