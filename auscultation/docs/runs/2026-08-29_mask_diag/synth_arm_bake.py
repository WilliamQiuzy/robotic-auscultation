"""Bake a synthetic arm-in-view session from a real recorded session.

Rewrites the follower trajectory so the arm sweeps through the camera
frustum during the soak's recording window, recomputes the frame ledger's
action/trajectory hashes and nominal observations with the PRODUCTION
functions, then composites the arm the telemetry describes into every
RGB frame and depth map (same FK/vertices/hand-eye the mask provider
uses, so downstream mask coverage is self-consistent ground truth).

Output session is a full drop-in for record_soak.py's SOAK_SESSION.
Run from a sealed code release's auscultation/ directory on linbox.
"""
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
import numpy as np
import cv2

from ausc.perception.production_runtime_activation import (
    load_production_runtime_assembly,
)
from ausc.perception.robot.robot_state_stream import (
    BracketedExactControlTrajectory,
    InterpolatedFollowerObservation,
    RobotStateContract,
)
from ausc.perception.depth_codec import encode_depth, load_depth
import ausc.perception.robot.robot_session_capture as rsc

SRC = Path(os.environ.get(
    "BAKE_SRC",
    "/home/robot/wam_data/sessions/"
    "diagnostic_perception_20260814T024117326614491Z_f165ea6a78f4",
))
DST = Path(os.environ.get(
    "BAKE_DST", "/home/robot/wam_data/scratch/t65_e2e/synth_arm_baked",
))
RIG = os.environ.get(
    "BAKE_RIG", "/home/robot/wam_data/scratch/t65_e2e/session_bundle_replay.py",
)
# In-frustum target pose found by joint-space scan WITHIN activated limits.
# Override with BAKE_QA="q1,q2,...,q7" (e.g. the heavy-occlusion pose).
QA = [0.8607, -1.4136, -1.0278, 1.3414, 1.369, -0.5428, 0.5484]
if os.environ.get("BAKE_QA"):
    QA = [float(v) for v in os.environ["BAKE_QA"].split(",")]
    assert len(QA) == 7
T_ENTER = float(os.environ.get("BAKE_T_ENTER", "100.0"))
# normal: arm depth composited at FK z. invalid: RGB painted but the depth
# in the arm region is ZEROED (sensor blind on the arm surface) — the veto
# must keep the full geometric mask there (invalid depth never vetoes).
DEPTH_MODE = os.environ.get("BAKE_DEPTH_MODE", "normal")
ALWAYS_IN = os.environ.get("BAKE_ALWAYS_IN", "0") == "1"
# realism: per-link plane-fit depth gradient + sigma-4mm sensor noise + 2px
# invalid edge rim (ToF edge artifacts) + depth-shaded RGB with per-link
# tint. flat: the original constant-min-z fill.
REALISM = os.environ.get("BAKE_REALISM", "1") == "1"
# sweep: single-pose wobble sweep (original). auscult: point-to-point
# dwell trajectory over four scanned back positions with global breathing.
TRAJ = os.environ.get("BAKE_TRAJ", "sweep")
POSES = {
    "lower-center": [0.8607, -1.4136, -1.0278, 1.3414, 1.369, -0.5428, 0.5484],
    "left-half": [-0.3402, -2.0356, -0.907, -0.35, -0.2397, 1.08, 0.9261],
    "upper-right": [-2.4007, -0.9311, -1.4191, 0.1891, 0.0746, 0.9111, -0.1448],
    "lower-right": [1.2617, -1.9477, 0.7852, 0.5879, -1.0449, -0.5268, -0.6497],
}
T_SWEEP_END = T_ENTER + 8.0 + 48.0          # 3 full wobble periods of 16 s
T_EXIT_END = T_SWEEP_END + 8.0

spec = importlib.util.spec_from_file_location("session_bundle_replay", RIG)
rig = importlib.util.module_from_spec(spec)
sys.modules["session_bundle_replay"] = rig
spec.loader.exec_module(rig)

ACT = ("/home/robot/wam_canary/reissue_releases/"
       "activation-416a3f1f0d564c8b3a42/activation.json")
SHA = "87fa96b3480f08c79b1a8e1960dd26980572a6a8d30d4257317ebcf66bfe53ee"
cam_profile = json.load(open(
    "/home/robot/wam_canary/reissue_releases/activation-416a3f1f0d564c8b3a42/"
    "artifacts/camera_profile.json"
))
assembly = load_production_runtime_assembly(
    ACT, expected_activation_sha256=SHA,
    point_names=("R1", "L1", "R2", "L2", "R3", "L3", "R4", "L4", "R5", "L5"),
    expected_camera_id="CL8326300A0",
    expected_capture_mode=cam_profile["capture_mode"],
)
contract = None
stack = [assembly.robot_capture_coordinator]
seen = set()
while stack and contract is None:
    obj = stack.pop()
    if id(obj) in seen:
        continue
    seen.add(id(obj))
    for name in dir(obj):
        if name.startswith("__"):
            continue
        try:
            val = getattr(obj, name)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(val, RobotStateContract):
            contract = val
            break
        if hasattr(val, "__dict__") and not callable(val) and len(seen) < 200:
            stack.append(val)
assert contract is not None
prov = assembly.robot_mask_provider
P = prov.projection


def ease(x):
    return 0.5 - 0.5 * math.cos(math.pi * min(max(x, 0.0), 1.0))


_AUSCULT_PLAN = None


def _auscult_plan(q0):
    # parked -> P1..P4 dwells (7 s each) with 5 s eased transits -> parked,
    # all inside the soak recording window (~[12, 85] s of session time).
    keys = [
        (0.0, 15.0, q0),
        (20.0, 27.0, POSES["lower-center"]),
        (32.0, 39.0, POSES["left-half"]),
        (44.0, 51.0, POSES["upper-right"]),
        (56.0, 63.0, POSES["lower-right"]),
        (70.0, 1e9, q0),
    ]
    return keys


def _q_auscult(t_rel, q0):
    global _AUSCULT_PLAN
    if _AUSCULT_PLAN is None:
        _AUSCULT_PLAN = _auscult_plan(q0)
    keys = _AUSCULT_PLAN
    q = None
    for idx, (t_in, t_out, pose) in enumerate(keys):
        if t_rel < t_in:
            prev_out, prev_pose = keys[idx - 1][1], keys[idx - 1][2]
            x = ease((t_rel - prev_out) / max(t_in - prev_out, 1e-9))
            q = [prev_pose[i] + x * (pose[i] - prev_pose[i]) for i in range(7)]
            break
        if t_rel < t_out:
            q = list(pose)
            break
    if q is None:
        q = list(keys[-1][2])
    # global breathing micro-motion, continuous across all segments
    q[4] += 0.02 * math.sin(2 * math.pi * t_rel / 8.0)
    q[5] += 0.015 * math.sin(2 * math.pi * t_rel / 6.0)
    return q


def q_of(t_rel, q0):
    if TRAJ == "auscult":
        return _q_auscult(t_rel, q0)
    if ALWAYS_IN:
        # Arm fully in view for the ENTIRE session (live-day scenario:
        # Start pressed with the arm already over the back), with the same
        # integer-period wobble so the ledger velocity gate stays green.
        wob = math.sin(2 * math.pi * t_rel / 16.0)
        q = list(QA)
        for j, amp in ((3, 0.12), (4, 0.10), (5, 0.15)):
            q[j] += amp * wob
        return q
    if t_rel < T_ENTER:
        alpha, wob = 0.0, 0.0
    elif t_rel < T_ENTER + 8.0:
        alpha, wob = ease((t_rel - T_ENTER) / 8.0), 0.0
    elif t_rel < T_SWEEP_END:
        alpha = 1.0
        wob = math.sin(2 * math.pi * (t_rel - (T_ENTER + 8.0)) / 16.0)
    elif t_rel < T_EXIT_END:
        alpha, wob = 1.0 - ease((t_rel - T_SWEEP_END) / 8.0), 0.0
    else:
        alpha, wob = 0.0, 0.0
    q = [q0[i] + alpha * (QA[i] - q0[i]) for i in range(7)]
    if wob:
        for j, amp in ((3, 0.12), (4, 0.10), (5, 0.15)):
            q[j] += alpha * amp * wob
    return q


# --- 1. rewrite telemetry ---------------------------------------------------
DST.mkdir(exist_ok=True)
for name in os.listdir(SRC):
    if name in ("robot_control_envelopes.jsonl", "robot_frame_index.jsonl",
                "rgb_segments", "depth"):
        continue
    target = DST / name
    if not target.exists():
        os.symlink(SRC / name, target)

rows_out = []
t0_ns = None
q0_ref = None
peak = 0.0
with open(SRC / "robot_control_envelopes.jsonl") as fh:
    for line in fh:
        row = json.loads(line)
        for rec in row["records"]:
            obs = rec["observation_monotonic_ns"]
            if t0_ns is None:
                t0_ns = obs
            if q0_ref is None:
                q0_ref = list(rec["follower_q_rad"])
            t_rel = (obs - t0_ns) / 1e9
            q = q_of(t_rel, q0_ref)
            qd = [
                (q_of(t_rel + 0.0005, q0_ref)[i]
                 - q_of(t_rel - 0.0005, q0_ref)[i]) / 0.001
                for i in range(7)
            ]
            peak = max(peak, max(abs(v) for v in qd))
            rec["follower_q_rad"] = [round(v, 9) for v in q]
            rec["follower_qd_rad_s"] = [round(v, 9) for v in qd]
            rec["follower_arm_mit_q"] = [round(v + 0.002, 9) for v in q]
            rec["follower_arm_mit_dq"] = [round(v, 9) for v in qd]
        rows_out.append(json.dumps(row))
with open(DST / "robot_control_envelopes.jsonl", "w") as fh:
    fh.write("\n".join(rows_out) + "\n")
print("[bake] telemetry rewritten; peak |qd| = %.3f rad/s" % peak)

# --- 2. rewrite ledger with production hash/observation functions ----------
by_tick = {}
for env in rig.iter_control_envelopes(DST, contract):
    for rec in env.records:
        by_tick[rec.control_tick_seq] = rec
ledger_rows = [json.loads(l) for l in open(SRC / "robot_frame_index.jsonl")]
with open(DST / "robot_frame_index.jsonl", "w") as out:
    for row in ledger_rows:
        first = int(row["observation_tick_first"])
        last = int(row["observation_tick_last"])
        bracket = tuple(by_tick[t] for t in range(first, last + 1))
        trajectory = BracketedExactControlTrajectory(
            exposure_lower_ns=int(row["exposure_lower_ns"]),
            exposure_nominal_ns=int(row["exposure_nominal_ns"]),
            exposure_upper_ns=int(row["exposure_upper_ns"]),
            records=bracket,
        )
        row["exact_action_sha256"] = rsc._exact_action_sha256(bracket)
        row["exact_control_trajectory_sha256"] = (
            rsc._exact_trajectory_sha256(bracket)
        )
        row["observation_at_nominal"] = rsc._observation_payload(
            trajectory.sample_observation_at(int(row["exposure_nominal_ns"]))
        )
        out.write(json.dumps(row) + "\n")
print("[bake] ledger rewritten:", len(ledger_rows), "rows")

# --- 3. composite arm into RGB segments + depth ----------------------------
q_times = np.asarray(sorted(by_tick))
q_lookup = {t: by_tick[t].follower_q_rad for t in by_tick}
obs_times = np.asarray(
    [by_tick[t].observation_monotonic_ns for t in sorted(by_tick)]
)
q_list = [by_tick[t].follower_q_rad for t in sorted(by_tick)]


def q_at_host(host_t_s):
    import bisect
    pos = bisect.bisect_left(obs_times.tolist(), int(host_t_s * 1e9))
    pos = min(max(pos, 0), len(q_list) - 1)
    return q_list[pos]


_paint_rng = np.random.default_rng(97)


def paint(rgb, depth_mm, host_t_s):
    q = q_at_host(host_t_s)
    state = InterpolatedFollowerObservation(
        monotonic_ns=1, follower_q_rad=tuple(q), follower_qd_rad_s=(0.0,) * 7,
        follower_gripper_q_rad=0.0, follower_gripper_qd_rad_s=0.0,
        follower_gripper_motor_q_rad=0.0, follower_gripper_motor_qd_rad_s=0.0,
        source_sequence_before=1, source_sequence_after=2,
    )
    transforms, _, _ = prov._forward_kinematics(state)
    painted = False
    for li, link in enumerate(prov.geometry.links):
        vb = prov._transform_vertices(transforms[li], link.vertices_link_m)
        vc = prov._transform_vertices(
            prov.handeye.transform_camera_from_follower_base, vb)
        z = vc[:, 2]
        keep = z > max(0.05, P.min_projection_depth_m)
        if not keep.any():
            continue
        u = P.fx * vc[keep, 0] / z[keep] + P.cx
        v = P.fy * vc[keep, 1] / z[keep] + P.cy
        inside = (u >= -50) & (u < P.width + 50) & (v >= -50) & (v < P.height + 50)
        if inside.sum() < 3:
            continue
        pts = np.stack([u[inside], v[inside]], axis=1).astype(np.int32)
        hull = cv2.convexHull(pts)
        poly = np.zeros(depth_mm.shape, dtype=np.uint8)
        cv2.fillConvexPoly(poly, hull, 1)
        region = poly.astype(bool)
        if not region.any():
            continue
        zi = z[keep][inside]
        if DEPTH_MODE == "invalid":
            depth_mm[region] = 0
            rgb[region] = (
                0.25 * rgb[region] + 0.75 * np.array([70, 72, 78])
            ).astype(rgb.dtype)
        elif REALISM:
            ui = u[inside]
            vi = v[inside]
            ys, xs = np.nonzero(region)
            # per-link plane-fit depth gradient across the hull
            A = np.stack([ui, vi, np.ones_like(ui)], axis=1)
            coef, *_ = np.linalg.lstsq(A, zi, rcond=None)
            zf = coef[0] * xs + coef[1] * ys + coef[2]
            zf = np.clip(zf, zi.min() - 0.02, zi.max() + 0.02)
            noise = _paint_rng.normal(0.0, 0.004, size=zf.shape)
            vals = np.maximum(50, ((zf + noise) * 1000.0)).astype(depth_mm.dtype)
            current = depth_mm[ys, xs]
            closer = (current == 0) | (current > vals)
            depth_mm[ys[closer], xs[closer]] = vals[closer]
            # ToF edge artifact: 2px invalid rim around the link silhouette
            eroded = cv2.erode(
                region.astype(np.uint8), np.ones((3, 3), np.uint8),
                iterations=2,
            ).astype(bool)
            rim = region & ~eroded
            depth_mm[rim] = 0
            # depth-shaded RGB with slight per-link tint + pixel noise
            span = max(zi.max() - zi.min(), 1e-3)
            shade = 0.55 + 0.45 * np.clip(
                (zi.max() - zf) / span, 0.0, 1.0)
            tint = np.array([70, 72, 78]) * (0.92 + 0.06 * (li % 3))
            base = tint[None, :] * shade[:, None]
            base += _paint_rng.normal(0.0, 3.0, size=base.shape)
            rgb[ys, xs] = np.clip(
                0.15 * rgb[ys, xs] + 0.85 * base, 0, 255,
            ).astype(rgb.dtype)
        else:
            fill_mm = int(float(zi.min()) * 1000.0)
            closer = region & ((depth_mm == 0) | (depth_mm > fill_mm))
            depth_mm[closer] = fill_mm
            rgb[region] = (
                0.25 * rgb[region] + 0.75 * np.array([70, 72, 78])
            ).astype(rgb.dtype)
        painted = True
    return painted


times = []  # position-ordered (frame_index, host_t)
with open(SRC / "rgb_times.csv") as fh:
    next(fh)
    for line in fh:
        parts = line.strip().split(",")
        if len(parts) >= 3:
            times.append((int(parts[0]), float(parts[1])))

seg_dir = DST / "rgb_segments"
seg_dir.mkdir(exist_ok=True)
depth_dir = DST / "depth"
depth_dir.mkdir(exist_ok=True)
segments = sorted((SRC / "rgb_segments").glob("segment_*.mp4"))
pos = 0
painted_frames = 0
for seg in segments:
    cap = cv2.VideoCapture(str(seg))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(seg_dir / seg.name), cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height),
    )
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frame_index, host_t = times[pos]
        depth_path_src = SRC / "depth" / ("depth_%06d.npy.zlib" % frame_index)
        depth_path_dst = depth_dir / depth_path_src.name
        depth = load_depth(str(depth_path_src)).copy()
        rgb = bgr[:, :, ::-1].copy()
        if paint(rgb, depth, host_t):
            painted_frames += 1
            with open(depth_path_dst, "wb") as dh:
                dh.write(encode_depth(depth))
            writer.write(np.ascontiguousarray(rgb[:, :, ::-1]))
        else:
            if not depth_path_dst.exists():
                os.link(depth_path_src, depth_path_dst)
            writer.write(bgr)
        pos += 1
    cap.release()
    writer.release()
print("[bake] frames total=%d painted=%d" % (pos, painted_frames))
print("[bake] DONE ->", DST)
