"""Auscultation perception deployment — ONE entry point for the validated pipeline.

  v9 NormalsDetector  ->  3D lift  ->  ModelBasedTracker (shared-rigid de-jitter + safety flags)

Emits, per frame, for all 10 lung points (R1-R5/L1-L5): BOTH the 2D pixel AND the 3D camera-frame
point (and 3D base-frame `xyz_world` if a hand-eye extrinsic is given), plus status/measured/std.

Sources:
  --source clip   --session orbbec_session3      (reads $AUSC_DATA/<session>/{rgb.mp4,depth,intrinsics.json})
  --source clip   --clip /path/to/clipdir
  --source orbbec                                 (live Femto Bolt via CameraStream; needs ausc_live env)

Run in the unified env that has BOTH torch and pyorbbecsdk:
  ~/miniconda3/envs/ausc_live/bin/python scripts/perception/run_pipeline.py --source orbbec --show
  ~/miniconda3/envs/ausc_live/bin/python scripts/perception/run_pipeline.py --source clip --session orbbec_session3 \
        --out /tmp/pts.jsonl --video /tmp/pipeline.mp4
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import cv2

ROOT = os.environ.get("AUSC_REPO", os.path.expanduser("~/Desktop/wam-vla-medical/auscultation"))
DATA = os.path.expanduser(os.environ.get("AUSC_DATA", "~/ausc_test"))
os.chdir(ROOT); sys.path[:0] = ["ausc/perception", "ausc", "ausc/models", "ausc/training", "scripts"]
from tracking.detector import NormalsDetector                       # noqa: E402
from tracking.geometry import Intrinsics                            # noqa: E402
from tracking.model_tracker import ModelBasedTracker, ModelConfig   # noqa: E402

STATUS_COL = {"TRACKING": (60, 230, 60), "COASTING": (40, 200, 235),
              "LOST": (120, 120, 120), "INIT": (180, 180, 180)}


def load_intr(path):
    d = json.load(open(path))
    return Intrinsics(d["fx"], d["fy"], d["cx"], d["cy"], d["width"], d["height"], 1.0)


def draw(bgr, out, scale=1.0):
    """Overlay the 10 points (2D) + status colour + label on a BGR frame."""
    img = bgr.copy()
    if out.initialized:
        for p in out.points:
            if not np.isfinite(p.pixel).all():
                continue
            u, v = int(p.pixel[0] * scale), int(p.pixel[1] * scale)
            cv2.circle(img, (u, v), 9, STATUS_COL.get(p.status, (180, 180, 180)), -1)
            cv2.circle(img, (u, v), 5, (60, 60, 235) if p.name[0] == "R" else (60, 220, 60), -1)
            cv2.putText(img, p.name, (u + 8, v + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (25, 25, 25), -1)
    cv2.putText(img, f"v9 + ModelBasedTracker | measured {out.n_measured}/10  tracking {out.n_tracking}/10",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 255, 255), 1)
    return img


def record(out):
    """One JSON-serialisable record with 2D + 3D for all points."""
    pts = []
    for p in out.points:
        r = {"name": p.name, "status": p.status, "measured": bool(p.measured),
             "std_mm": round(float(p.position_std) * 1000, 1),
             "uv": [round(float(p.pixel[0]), 1), round(float(p.pixel[1]), 1)] if np.isfinite(p.pixel).all() else None,
             "xyz_cam_mm": [round(float(c) * 1000, 1) for c in p.xyz_camera] if np.isfinite(p.xyz_camera).all() else None}
        if p.xyz_world is not None and np.isfinite(p.xyz_world).all():
            r["xyz_base_mm"] = [round(float(c) * 1000, 1) for c in p.xyz_world]
        pts.append(r)
    return {"frame": out_frame_idx[0], "t": round(out.t, 3), "initialized": bool(out.initialized),
            "n_measured": out.n_measured, "n_tracking": out.n_tracking, "points": pts}


out_frame_idx = [0]


def clip_frames(session, clip, max_frames):
    base = clip if clip else os.path.join(DATA, session)
    intr = load_intr(os.path.join(base, "intrinsics.json"))
    cap = cv2.VideoCapture(os.path.join(base, "rgb.mp4")); n = int(cap.get(7))
    n = min(n, max_frames) if max_frames else n
    for fi in range(n):
        ok, bgr = cap.read()
        if not ok:
            break
        depth = np.load(os.path.join(base, "depth", f"d{fi:04d}.npy"))
        yield bgr, depth, fi / 30.0, 1.0, intr
    cap.release()


def orbbec_frames(intr_path, max_frames):
    from camera_stream import CameraStream, STATUS_OK
    intr = load_intr(intr_path)
    with CameraStream(backend="orbbec", rgb=True) as cam:
        t0 = time.monotonic(); i = 0
        while not max_frames or i < max_frames:
            f = cam.latest_frame()
            if f.status != STATUS_OK or f.frame is None or f.depth is None:
                time.sleep(0.005); continue
            bgr = f.frame[:, :, ::-1]                       # RGB->BGR for draw/IO; process wants RGB
            yield bgr, f.depth, time.monotonic() - t0, f.depth_scale, intr
            i += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["clip", "orbbec"], default="clip")
    ap.add_argument("--session", default="orbbec_session3")
    ap.add_argument("--clip", default=None)
    ap.add_argument("--intrinsics", default=os.path.join(DATA, "orbbec_session2", "intrinsics.json"),
                    help="intrinsics json for live orbbec (same Femto Bolt @1920x1080)")
    ap.add_argument("--extrinsics", default=None, help="cam2base.json (4x4) -> also emit xyz_base_mm")
    ap.add_argument("--weight", default="outputs/perc_lmh_v9_occ/best.pt")
    ap.add_argument("--out", default=None, help="JSONL of per-frame 2D+3D points")
    ap.add_argument("--video", default=None, help="overlay mp4 (clip or recorded live)")
    ap.add_argument("--show", action="store_true", help="live cv2 window")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    E = None
    if args.extrinsics:
        E = np.array(json.load(open(args.extrinsics))["T_cam_base"], float)

    det = NormalsDetector(args.weight, half=True)

    if args.source == "clip":
        gen = clip_frames(args.session, args.clip, args.max_frames)
        first = next(gen)
        intr = first[4]
    else:
        gen = orbbec_frames(args.intrinsics, args.max_frames)
        first = next(gen)
        intr = first[4]
    tr = ModelBasedTracker(det, intr, ModelConfig(), extrinsics=E)

    fout = open(args.out, "w") if args.out else None
    vw = None
    t_start = time.monotonic(); nfr = 0
    import itertools
    for bgr, depth, t, dscale, _ in itertools.chain([first], gen):
        out = tr.process(bgr[:, :, ::-1], depth, t=t, depth_scale=dscale)
        out_frame_idx[0] = nfr
        if fout:
            fout.write(json.dumps(record(out)) + "\n")
        if args.video or args.show:
            vis = draw(bgr, out)
            if args.video:
                if vw is None:
                    h, w = vis.shape[:2]
                    vw = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
                vw.write(vis)
            if args.show:
                cv2.imshow("auscultation pipeline", cv2.resize(vis, (960, 540)))
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
        if nfr % 30 == 0:
            samp = next((p for p in out.points if p.name == "R3"), None) if out.initialized else None
            s = ""
            if samp is not None and samp.xyz_camera is not None and np.isfinite(samp.xyz_camera).all():
                s = f" R3=({samp.xyz_camera[0]*1000:.0f},{samp.xyz_camera[1]*1000:.0f},{samp.xyz_camera[2]*1000:.0f})mm/{samp.status}"
            print(f"[{nfr:4d}] init={out.initialized} measured={out.n_measured}/10{s}", flush=True)
        nfr += 1
    dt = time.monotonic() - t_start
    if fout:
        fout.close()
    if vw:
        vw.release()
    if args.show:
        cv2.destroyAllWindows()
    print(f"\nDONE {nfr} frames in {dt:.1f}s ({nfr/max(dt,1e-6):.1f} Hz). "
          f"2D+3D per-point records -> {args.out or '(none)'}; overlay -> {args.video or '(none)'}")


if __name__ == "__main__":
    main()
