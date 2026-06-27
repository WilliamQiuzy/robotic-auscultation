"""Separate NOISE from real motion: per-point output depth, high-pass >2Hz = noise (real motion
<1Hz). Compares detector / model(snap OFF) / model(snap ON). Answers: is there a real depth-NOISE
defect, and does surface-snap reduce it?"""
import os,sys,json,numpy as np,cv2
ROOT=os.path.expanduser("~/Desktop/wam-vla-medical/auscultation"); os.chdir(ROOT)
sys.path[:0]=["ausc/perception","ausc","ausc/models","ausc/training","scripts"]
from tracking.detector import NormalsDetector
from tracking.geometry import Intrinsics, sample_depth_m
from tracking.model_tracker import ModelBasedTracker, ModelConfig
S=os.path.expanduser("~/ausc_test/orbbec_session3"); d=json.load(open(f"{S}/intrinsics.json"))
intr=Intrinsics(d["fx"],d["fy"],d["cx"],d["cy"],d["width"],d["height"],1.0)
det=NormalsDetector("outputs/perc_lmh_v7_adapt/best.pt"); fps=30.0
def hf_noise(Z):  # Z (T,10) depth mm -> per-point RMS of >2Hz component (noise), median over pts
    T=len(Z); F=np.fft.rfftfreq(T,1/fps); out=[]
    for k in range(Z.shape[1]):
        z=Z[:,k]; m=np.isfinite(z)
        if m.sum()<T*0.6: continue
        z=np.interp(np.arange(T),np.flatnonzero(m),z[m])
        sp=np.fft.rfft((z-z.mean())*np.hanning(T)); hp=sp.copy(); hp[F<=2.0]=0
        out.append(np.sqrt(np.mean(np.abs(np.fft.irfft(hp,T))**2)))
    return np.median(out)
def run(mode):
    cap=cv2.VideoCapture(f"{S}/rgb.mp4"); N=int(cap.get(7)); Z=[]
    tr=ModelBasedTracker(det,intr,ModelConfig(surface_snap=(mode=="on"))) if mode!="det" else None
    for fi in range(N):
        ok,bgr=cap.read()
        if not ok: break
        depth=np.load(f"{S}/depth/d{fi:04d}.npy")
        if mode=="det":
            r=det.detect(bgr[:,:,::-1],depth,intr); full=r.crop.det_to_full(r.xy_det)
            z=sample_depth_m(depth,full,1.0); Z.append(intr.unproject(full,np.where(np.isfinite(z),z,np.nan))[:,2]*1000)
        else:
            o=tr.process(bgr[:,:,::-1],depth,t=fi/fps,depth_scale=1.0)
            Z.append([p.xyz_camera[2]*1000 for p in o.points] if o.initialized else [np.nan]*10)
    cap.release(); return hf_noise(np.array(Z))
for mode in ("det","off","on"):
    print(f"{mode:3s}: >2Hz depth NOISE (RMS, median over points) = {run(mode):.1f} mm")
