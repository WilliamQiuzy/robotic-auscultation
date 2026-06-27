"""Honest real-effect eval of ModelBasedTracker on yellow: is the smoothness REAL tracking or
over-smoothing/lag? Measures lag vs detector, deviation from the zero-phase (non-causal ideal)
smoother, and error split by motion level. Low jitter alone can be gamed by freezing."""
import os,sys,json,numpy as np,cv2
ROOT=os.path.expanduser("~/Desktop/wam-vla-medical/auscultation"); os.chdir(ROOT)
sys.path[:0]=["ausc/perception","ausc","ausc/models","ausc/training","scripts"]
from tracking.detector import NormalsDetector, LUNG_ORDER, LUNG_INDEX
from tracking.geometry import Intrinsics, sample_depth_m
from tracking.model_tracker import ModelBasedTracker, ModelConfig
try:
    from scipy.signal import butter, filtfilt; HAVE_SCIPY=True
except Exception: HAVE_SCIPY=False
S=os.path.expanduser("~/ausc_test/orbbec_session3"); d=json.load(open(f"{S}/intrinsics.json"))
intr=Intrinsics(d["fx"],d["fy"],d["cx"],d["cy"],d["width"],d["height"],1.0)
det=NormalsDetector("outputs/perc_lmh_v7_adapt/best.pt")
tr=ModelBasedTracker(det,intr,ModelConfig())
cap=cv2.VideoCapture(f"{S}/rgb.mp4"); N=int(cap.get(7)); fps=30.0
DET=[]; MOD=[]; fits=[]; stat=[]; resid=[]; frames_keep={}
for fi in range(N):
    ok,bgr=cap.read()
    if not ok: break
    depth=np.load(f"{S}/depth/d{fi:04d}.npy")
    # raw detector 3D (for reference truth-ish, unbiased per-frame)
    r=det.detect(bgr[:,:,::-1],depth,intr); full=r.crop.det_to_full(r.xy_det)
    z=sample_depth_m(depth,full,1.0); z=np.where(np.isfinite(z),z,np.nan)
    DET.append(intr.unproject(full,z))
    out=tr.process(bgr[:,:,::-1],depth,t=fi/fps,depth_scale=1.0)
    MOD.append(np.array([p.xyz_camera for p in out.points]) if out.initialized else np.full((10,3),np.nan))
    fits.append(out.pose_fit_mm); stat.append([p.status for p in out.points])
    resid.append(np.linalg.norm(tr.residual,axis=1).mean()*1000 if out.initialized else np.nan)
    if fi in (int(N*f) for f in (0.03,0.15,0.3,0.45,0.6,0.75,0.9,0.97)):
        frames_keep[fi]=(bgr.copy(),out)
cap.release()
DET=np.array(DET); MOD=np.array(MOD)              # (T,10,3)
T=len(DET)
def fill(a):
    a=a.copy()
    for k in range(a.shape[1]):
        for c in range(a.shape[2]):
            v=a[:,k,c]; m=np.isfinite(v)
            if m.sum()>10: v[~m]=np.interp(np.flatnonzero(~m),np.flatnonzero(m),v[m]); a[:,k,c]=v
    return a
D=fill(DET); M=fill(MOD)
# jitter (common-mode removed)
def jit(X):
    dX=np.diff(X,0); cm=np.median(dX,1,keepdims=True); j=np.linalg.norm(dX-cm,axis=2)
    return np.median(j)*1000, np.percentile(j,90)*1000
print("=== HONEST EFFECT EVAL (real yellow, %d frames) ==="%T)
print("jitter  detector: med %.1f p90 %.1f mm | model: med %.1f p90 %.1f mm"%(jit(D)+jit(M)))
# LAG: cross-correlate model vs detector centroid x (most gross motion)
cD=D.mean(1); cM=M.mean(1)
def lag(a,b):
    a=a-a.mean(); b=b-b.mean(); c=np.correlate(b,a,"full"); return np.argmax(c)-(len(a)-1)
lags=[lag(cD[:,i],cM[:,i]) for i in range(3)]
print("LAG model-behind-detector (frames, +=lag): x=%d y=%d z=%d -> ~%d ms"%(lags[0],lags[1],lags[2],int(np.median(lags)/fps*1000)))
# ACCURACY vs non-causal ideal (zero-phase low-pass of detector @1.5Hz) = best smooth w/o lag
if HAVE_SCIPY and T>30:
    b,a=butter(2,1.5/(fps/2)); ideal=filtfilt(b,a,D,axis=0)
    dev=np.linalg.norm(M-ideal,axis=2)*1000
    print("model vs zero-phase ideal: med %.1f p90 %.1f mm (low=accurate+smooth; high=lag/bias)"%(np.median(dev),np.percentile(dev,90)))
# motion-split: model-vs-detector residual in high vs low motion frames
spd=np.r_[0,np.linalg.norm(np.diff(cD,axis=0),axis=1)]  # gross speed per frame
hi=spd>np.percentile(spd,75); lo=spd<np.percentile(spd,25)
err=np.linalg.norm(M-D,axis=2)*1000  # model deviates from detector by...
print("model-vs-detector dev: still frames %.1fmm | fast-motion frames %.1fmm (big gap=lag)"%(np.nanmedian(err[lo]),np.nanmedian(err[hi])))
# robustness
allst=np.array(stat); 
print("status: TRACKING %.0f%% COASTING %.0f%% LOST %.0f%%"%((allst=="TRACKING").mean()*100,(allst=="COASTING").mean()*100,(allst=="LOST").mean()*100))
print("pose-fit residual: med %.1f mm | per-point non-rigid residual: med %.1f mm"%(np.nanmedian(fits),np.nanmedian(resid)))
# temporal strip of model output (incl. turns)
tiles=[]
for fi in sorted(frames_keep):
    bgr,out=frames_keep[fi]; ov=bgr.copy()
    if out.initialized:
        proj={}
        for k,p in enumerate(out.points):
            if np.isfinite(p.pixel).all():
                proj[k]=p.pixel; x0,y0=int(p.pixel[0]),int(p.pixel[1]); c=(40,40,235) if p.name[0]=="R" else (40,220,40)
                cv2.circle(ov,(x0,y0),9,(255,255,255),-1); cv2.circle(ov,(x0,y0),6,c,-1)
        for side in ("R","L"):
            pl=[proj[LUNG_INDEX[f"{side}{i}"]] for i in range(1,6) if LUNG_INDEX[f"{side}{i}"] in proj]
            if len(pl)>1: cv2.polylines(ov,[np.array(pl,np.int32)],False,(40,40,235) if side=="R" else (40,220,40),2)
    cv2.putText(ov,f"f{fi} ({fi/N*100:.0f}%)",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,255),2)
    tiles.append(cv2.resize(ov,(320,180)))
rows=[np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)]
cv2.imwrite(os.path.expanduser("~/ausc_test/model_temporal_strip.png"),np.vstack(rows))
print("WROTE model_temporal_strip.png")
