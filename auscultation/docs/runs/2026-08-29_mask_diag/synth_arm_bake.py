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


def q_of(t_rel, q0):
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
        if DEPTH_MODE == "invalid":
            depth_mm[region] = 0
        else:
            fill_mm = int(float(z[keep][inside].min()) * 1000.0)
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
