"""Auscultation target points — the goal the demo is conditioned on (point-conditioned teleop).

Our collection is point-conditioned: the screen shows the doctor the CURRENT auscultation target
(a landmark on the back, highlighted on the live camera), the doctor moves the stethoscope there, and
we record WHICH target was active so the policy learns "given this point → do this action". The
target's 2D pixel + 3D position come from the perception landmarks (or a click); this module is the
schema + the doctor-facing display + the aligned logger.

  - `TargetPoint`: id, name (LUL/LLL/RUL/RLL/…), uv (normalized image), xyz (3D, if localized).
  - `TargetLog`: writes `targets.jsonl` (active target over time, `time.monotonic()`-stamped → aligns
    with record_session + the teleop robot.csv, so the converter can attach the target per frame).
  - `overlay()`: draw the targets + highlight the active one for the doctor.
  - CLI: a live display loop that logs the active target while the doctor teleops.

    python ausc/perception/ausc_targets.py --session-dir <dir> --camera-backend orbbec
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TargetPoint:
    id: int
    name: str
    uv: tuple                                    # normalized (u, v) in [0,1] on the camera image
    xyz: Optional[tuple] = None                  # (x, y, z) camera/robot frame, if localized (depth/perception)


# Posterior-lung auscultation grid (placeholder normalized positions; perception/click overrides at
# runtime). Camera sees the back → patient-left is image-right.
DEFAULT_TARGETS = [
    TargetPoint(0, "RUL", (0.58, 0.34)), TargetPoint(1, "LUL", (0.42, 0.34)),
    TargetPoint(2, "RLL", (0.58, 0.62)), TargetPoint(3, "LLL", (0.42, 0.62)),
]


def load_targets(path: Optional[str]) -> list[TargetPoint]:
    """Load targets from a config JSON (`[{id,name,uv,[xyz]}]`) or a perception landmarks JSONL
    (`{name, xy_pixel|uv, xyz_camera|xyz}`, one landmark per line); else the default grid."""
    if not path or not os.path.exists(path):
        return list(DEFAULT_TARGETS)
    txt = open(path).read().strip()
    if txt.startswith("["):
        return [TargetPoint(d.get("id", i), d["name"], tuple(d["uv"]), tuple(d["xyz"]) if d.get("xyz") else None)
                for i, d in enumerate(json.loads(txt))]
    out = []
    for i, line in enumerate(txt.splitlines()):
        if not line.strip():
            continue
        d = json.loads(line)
        uv = tuple(d["uv"]) if "uv" in d else tuple(d["xy_pixel"])   # perception may give pixels
        xyz = tuple(d.get("xyz") or d.get("xyz_camera") or d.get("xyz_world") or ()) or None
        out.append(TargetPoint(d.get("id", i), d.get("name", f"p{i}"), uv, xyz))
    return out


class TargetLog:
    """Append the active target to `targets.jsonl` (monotonic-timestamped) as the doctor works."""

    def __init__(self, session_dir: str):
        self.f = open(os.path.join(session_dir, "targets.jsonl"), "w")

    def log(self, t: float, tp: TargetPoint, w: int, h: int) -> None:
        rec = {"t": round(t, 4), "id": tp.id, "name": tp.name,
               "u_px": round(tp.uv[0] * w, 1), "v_px": round(tp.uv[1] * h, 1),
               "uv": [round(tp.uv[0], 4), round(tp.uv[1], 4)],
               "xyz": [round(v, 4) for v in tp.xyz] if tp.xyz else None}
        self.f.write(json.dumps(rec) + "\n"); self.f.flush()

    def close(self) -> None:
        self.f.close()


def _dashed_circle(frame, center, r, color, n=16):
    import cv2
    import math
    for k in range(0, n, 2):                                    # every other arc segment = dashes
        a0, a1 = k * 360 / n, (k + 1) * 360 / n
        cv2.ellipse(frame, center, (r, r), 0, a0, a1, color, 3)


def overlay(frame, targets: list[TargetPoint], active_idx: int, occluded: bool = False,
            status: str = "", completed=()):
    """Draw the targets; highlight the active one. If `occluded` (the arm covers the point), the
    active target is drawn DASHED at its held 3D→2D position so the doctor still sees where it is."""
    import cv2
    h, w = frame.shape[:2]
    for i, tp in enumerate(targets):
        px, py = int(tp.uv[0] * w), int(tp.uv[1] * h)
        if i in completed:
            cv2.circle(frame, (px, py), 9, (0, 200, 0), 2)
            cv2.putText(frame, "OK", (px - 10, py - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
        elif i == active_idx:
            col = (0, 220, 255)
            if occluded:
                _dashed_circle(frame, (px, py), 22, col)
                cv2.putText(frame, f"{tp.name} (occluded — held)", (px - 70, py - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            else:
                cv2.circle(frame, (px, py), 22, col, 3)
                cv2.putText(frame, f"GO HERE: {tp.name}", (px - 60, py - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
            cv2.circle(frame, (px, py), 4, col, -1)
        else:
            cv2.circle(frame, (px, py), 8, (120, 120, 120), 2)
            cv2.putText(frame, tp.name, (px - 12, py - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    if status:                                                  # feedback bar (bottom)
        cv2.rectangle(frame, (0, h - 26), (w, h), (30, 30, 30), -1)
        cv2.putText(frame, status, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
    return frame


def run(session_dir: str, targets: list[TargetPoint], camera_backend: str = "orbbec",
        camera_index: int = 0, start_idx: int = 0) -> None:
    """Doctor display loop: show camera + highlighted target, log active target, SPACE/n = next, q = quit."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import cv2
    from camera_stream import CameraStream

    os.makedirs(session_dir, exist_ok=True)
    log = TargetLog(session_dir)
    active = start_idx
    print(f"[targets] display: {[t.name for t in targets]} | SPACE/n=next  b=back  q=quit  -> "
          f"{os.path.join(session_dir, 'targets.jsonl')}")
    with CameraStream(backend=camera_backend, index=camera_index, rgb=True) as cam:
        while True:
            f = cam.latest_frame()
            if f is None or f.frame is None:
                time.sleep(0.01); continue
            img = overlay(f.frame[:, :, ::-1].copy(), targets, active)      # RGB->BGR for cv2
            cv2.imshow("auscultation target (doctor view)", img)
            log.log(time.monotonic(), targets[active], f.frame.shape[1], f.frame.shape[0])
            k = cv2.waitKey(1) & 0xFF
            if k in (ord(" "), ord("n")):
                active = (active + 1) % len(targets)
            elif k == ord("b"):
                active = (active - 1) % len(targets)
            elif k == ord("q"):
                break
    log.close(); cv2.destroyAllWindows()
    print(f"[targets] done -> {os.path.join(session_dir, 'targets.jsonl')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-dir", required=True)
    ap.add_argument("--targets", default=None, help="targets config JSON or perception landmarks JSONL")
    ap.add_argument("--camera-backend", default="orbbec", choices=["orbbec", "avfoundation", "fake"])
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--start-idx", type=int, default=0)
    a = ap.parse_args()
    run(a.session_dir, load_targets(a.targets), a.camera_backend, a.camera_index, a.start_idx)
