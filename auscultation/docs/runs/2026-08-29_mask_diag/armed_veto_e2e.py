"""Armed-session depth-veto E2E: FK-vertex oracle over a real moving-arm session.

For every replayed frame (anchors 1-4 enforced by the rig):
  ORACLE: project the nominal FK collision vertices of every link+tool into
  the image; wherever the depth sensor MEASURES the arm (|depth - vertex z|
  < 0.15 m), the veto mask MUST still cover that pixel. Any violation means
  the veto cleared a pixel the arm demonstrably occupies -> unsound.
Also reports veto reduction vs pure-geometric mask during motion.
"""
import importlib.util, json, sys, os
sys.path.insert(0, os.getcwd())
import numpy as np, cv2
from pathlib import Path
from ausc.perception.production_runtime_activation import load_production_runtime_assembly
from ausc.perception.robot.robot_state_stream import RobotStateContract
from ausc.perception.camera_stream import CameraFrame
from ausc.perception.depth_codec import load_depth

spec = importlib.util.spec_from_file_location(
    "session_bundle_replay", "/home/robot/wam_data/scratch/t65_e2e/session_bundle_replay.py")
rig = importlib.util.module_from_spec(spec)
sys.modules["session_bundle_replay"] = rig
spec.loader.exec_module(rig)

SESSION = Path(sys.argv[1])
STRIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 15
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 200
OUT = sys.argv[4] if len(sys.argv) > 4 else None

ACT = "/home/robot/wam_canary/reissue_releases/activation-416a3f1f0d564c8b3a42/activation.json"
SHA = "87fa96b3480f08c79b1a8e1960dd26980572a6a8d30d4257317ebcf66bfe53ee"
cam_profile = json.load(open("/home/robot/wam_canary/reissue_releases/activation-416a3f1f0d564c8b3a42/artifacts/camera_profile.json"))
assembly = load_production_runtime_assembly(
    ACT, expected_activation_sha256=SHA,
    point_names=("R1","L1","R2","L2","R3","L3","R4","L4","R5","L5"),
    expected_camera_id="CL8326300A0",
    expected_capture_mode=cam_profile["capture_mode"],
)
contract = None
stack = [assembly.robot_capture_coordinator]; seen = set()
while stack and contract is None:
    obj = stack.pop()
    if id(obj) in seen: continue
    seen.add(id(obj))
    for name in dir(obj):
        if name.startswith("__"): continue
        try: val = getattr(obj, name)
        except Exception: continue
        if isinstance(val, RobotStateContract): contract = val; break
        if hasattr(val, "__dict__") and not callable(val) and len(seen) < 200:
            stack.append(val)
prov = assembly.robot_mask_provider
P = prov.projection

times = {}
with open(rig.session_artifact_path(SESSION, "rgb_times.csv")) as fh:
    next(fh)
    for pos, line in enumerate(fh):
        parts = line.strip().split(",")
        if len(parts) >= 3:
            times[int(parts[0])] = (pos, float(parts[1]), float(parts[2]))

segments = sorted((SESSION / "rgb_segments").glob("segment_*.mp4"))
class SequentialRGB:
    def __init__(self):
        self.segments = list(segments); self.cap = None; self.pos = -1
    def frame_at(self, pos):
        while True:
            if self.cap is None:
                if not self.segments: raise RuntimeError("rgb exhausted")
                self.cap = cv2.VideoCapture(str(self.segments.pop(0)))
            ok, frame = self.cap.read()
            if not ok:
                self.cap.release(); self.cap = None; continue
            self.pos += 1
            if self.pos == pos: return frame[:, :, ::-1].copy()

rgb = SequentialRGB()
def frame_loader(row):
    idx = int(row["frame_index"])
    pos, host_t, dev_ms = times[idx]
    return CameraFrame(
        frame=rgb.frame_at(pos), t=host_t, index=idx, status="OK",
        depth=load_depth(str(SESSION / "depth" / ("depth_%06d.npy.zlib" % idx))),
        depth_scale=1.0, age_s=0.0, device_t_ms=dev_ms,
        time_source="orbbec_device_and_host_arrival",
    )

# ---------------------------------------------------------------------------
# COMPOSITE mode (arm-in-view synthesis): render the arm the session's OWN
# telemetry describes into every frame's depth+RGB. The composite and the
# mask provider consume the same FK/vertices/hand-eye, so mask coverage of
# the composited pixels is a self-consistent, assertable invariant. Used
# with a derived session whose telemetry q was rewritten to sweep the arm
# through the camera frustum (real sessions never had it in view).
# ---------------------------------------------------------------------------
COMPOSITE = os.environ.get("COMPOSITE", "0") == "1"
# PHANTOM mode: the telemetry says the arm is in view but the pixels are the
# ORIGINAL scene (arm absent). The geometric mask paints huge phantom hulls;
# the depth veto's whole purpose is to strip them wherever the scene depth
# PROVES no arm — this measures that positive case end to end. Hull regions
# are computed exactly like COMPOSITE but nothing is painted into the frames.
PHANTOM = os.environ.get("PHANTOM", "0") == "1"
composite_masks = {}
if COMPOSITE or PHANTOM:
    from ausc.perception.robot.robot_state_stream import (
        InterpolatedFollowerObservation,
    )
    _q_times = []
    _q_values = []
    for _env in rig.iter_control_envelopes(SESSION, contract):
        for _rec in _env.records:
            _q_times.append(int(_rec.observation_monotonic_ns))
            _q_values.append(tuple(_rec.follower_q_rad))
    _order = np.argsort(np.asarray(_q_times))
    _q_times = np.asarray(_q_times)[_order]
    _q_values = [_q_values[i] for i in _order]
    print("[composite] telemetry q samples:", len(_q_times))

    def _q_at(host_t_s):
        import bisect as _bisect
        ns = int(host_t_s * 1e9)
        pos = _bisect.bisect_left(_q_times.tolist(), ns)
        pos = min(max(pos, 0), len(_q_values) - 1)
        return _q_values[pos]

    def _composite_arm(frame_rgb, depth_mm, host_t_s, frame_index):
        q = _q_at(host_t_s)
        state = InterpolatedFollowerObservation(
            monotonic_ns=1, follower_q_rad=tuple(q),
            follower_qd_rad_s=(0.0,) * 7,
            follower_gripper_q_rad=0.0, follower_gripper_qd_rad_s=0.0,
            follower_gripper_motor_q_rad=0.0,
            follower_gripper_motor_qd_rad_s=0.0,
            source_sequence_before=1, source_sequence_after=2,
        )
        transforms, _, _ = prov._forward_kinematics(state)
        painted = np.zeros(depth_mm.shape, dtype=bool)
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
            if not PHANTOM:
                fill_mm = int(float(z[keep][inside].min()) * 1000.0)
                closer = region & ((depth_mm == 0) | (depth_mm > fill_mm))
                depth_mm[closer] = fill_mm
                frame_rgb[region] = (
                    0.25 * frame_rgb[region] + 0.75 * np.array([70, 72, 78])
                ).astype(frame_rgb.dtype)
            painted |= region
        composite_masks[int(frame_index)] = painted
        return frame_rgb, depth_mm

    _plain_loader = frame_loader

    def frame_loader(row):  # noqa: F811 - deliberate composite wrapper
        frame = _plain_loader(row)
        rgb2 = np.ascontiguousarray(frame.frame).copy()
        depth2 = np.asarray(frame.depth).copy()
        rgb2, depth2 = _composite_arm(rgb2, depth2, frame.t, frame.index)
        return CameraFrame(
            frame=rgb2, t=frame.t, index=frame.index, status=frame.status,
            depth=depth2, depth_scale=frame.depth_scale, age_s=frame.age_s,
            device_t_ms=frame.device_t_ms, time_source=frame.time_source,
        )

all_rows = [r for r in rig.iter_frame_index_rows(SESSION) if int(r["frame_index"]) in times]
rows = all_rows[::STRIDE][:LIMIT]
print("session:", SESSION.name)
print("frames total=%d replaying=%d (stride %d)" % (len(all_rows), len(rows), STRIDE))

vw = None
if OUT:
    vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (1920, 1080))

oracle_checked = 0
oracle_violations = 0
arm_in_view_frames = 0
reductions = []
coverage_rows = []
phantom_rows = []
count = 0
for result in rig.reconstruct_session_bundles(
    frame_index_rows=rows,
    envelopes=rig.iter_control_envelopes(SESSION, contract),
    contract=contract,
    exposure_contract=assembly.camera_exposure_contract,
    activated_identity=assembly.activated_identity,
    mask_provider=prov,
    frame_loader=frame_loader,
    depth_veto=True,
):
    count += 1
    bundle = result.bundle
    veto_mask = bundle.robot_mask_packet.mask
    traj = bundle.robot_trajectory
    frame = bundle.camera_frame
    depth = np.asarray(frame.depth)
    if COMPOSITE or PHANTOM:
        painted = composite_masks.get(int(result.frame_index))
        if painted is not None and painted.any():
            covered = int((painted & veto_mask).sum())
            total = int(painted.sum())
            coverage_rows.append((result.frame_index, covered, total))
    # pure-geometric comparison mask
    geo = prov.build(exposure=bundle.exposure, trajectory=traj).mask
    reductions.append((int(geo.sum()), int(veto_mask.sum())))
    if PHANTOM:
        painted = composite_masks.get(int(result.frame_index))
        if painted is not None and painted.any():
            geo_hull = int((painted & geo).sum())
            freed = int((painted & geo & ~veto_mask).sum())
            phantom_rows.append((result.frame_index, geo_hull, freed))
    # FK oracle at the nominal state
    state = traj.sample_observation_at(traj.exposure_nominal_ns)
    transforms, _, _ = prov._forward_kinematics(state)
    vis = None
    for link_index, link in enumerate(prov.geometry.links):
        vb = prov._transform_vertices(transforms[link_index], link.vertices_link_m)
        vc = prov._transform_vertices(prov.handeye.transform_camera_from_follower_base, vb)
        z = vc[:, 2]
        keep = z > P.min_projection_depth_m
        if not keep.any(): continue
        u = P.fx * vc[keep, 0] / z[keep] + P.cx
        v = P.fy * vc[keep, 1] / z[keep] + P.cy
        ui = np.rint(u).astype(int); vi = np.rint(v).astype(int)
        inside = (ui >= 0) & (ui < P.width) & (vi >= 0) & (vi < P.height)
        if not inside.any(): continue
        arm_seen = False
        for uu, vv, zz in zip(ui[inside], vi[inside], z[keep][inside]):
            measured = depth[vv, uu] / 1000.0
            if measured > 0 and abs(measured - zz) < 0.15:
                arm_seen = True
                oracle_checked += 1
                if not veto_mask[vv, uu]:
                    oracle_violations += 1
                    print("VIOLATION frame %d link %d px (%d,%d) z=%.3f measured=%.3f"
                          % (result.frame_index, link_index, uu, vv, zz, measured))
        if arm_seen and vis is None and vw is not None:
            vis = np.ascontiguousarray(frame.frame[:, :, ::-1])
            edges = cv2.Canny(veto_mask.astype(np.uint8) * 255, 50, 150)
            vis[edges > 0] = (0, 0, 255)
    if vis is not None:
        for link_index, link in enumerate(prov.geometry.links):
            vb = prov._transform_vertices(transforms[link_index], link.vertices_link_m)
            vc = prov._transform_vertices(prov.handeye.transform_camera_from_follower_base, vb)
            z = vc[:, 2]
            keep = z > P.min_projection_depth_m
            if not keep.any(): continue
            u = (P.fx * vc[keep, 0] / z[keep] + P.cx).astype(int)
            v = (P.fy * vc[keep, 1] / z[keep] + P.cy).astype(int)
            for uu, vv in zip(u, v):
                if 0 <= uu < P.width and 0 <= vv < P.height:
                    cv2.circle(vis, (uu, vv), 4, (0, 255, 0), -1)
        cv2.putText(vis, "frame %d ARM-IN-VIEW veto-mask edge=red FK verts=green"
                    % result.frame_index, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        vw.write(vis)
        arm_in_view_frames += 1

if vw is not None: vw.release()
geo_px = np.array([g for g, v in reductions]); veto_px = np.array([v for g, v in reductions])
print("frames replayed:", count, "(anchors 1-4 all enforced)")
print("arm-visible-to-depth oracle: checked=%d violations=%d" % (oracle_checked, oracle_violations))
print("arm-in-view rendered frames:", arm_in_view_frames)
print("geometric mask px median=%d max=%d" % (int(np.median(geo_px)), int(geo_px.max())))
print("veto mask px      median=%d max=%d" % (int(np.median(veto_px)), int(veto_px.max())))
print("median veto reduction: %.1f%%" % (100.0 * (1 - np.median(veto_px / np.maximum(geo_px, 1)))))
if PHANTOM and phantom_rows:
    freed_fracs = [f / max(g, 1) for _, g, f in phantom_rows]
    total_hull = sum(g for _, g, _ in phantom_rows)
    total_freed = sum(f for _, _, f in phantom_rows)
    print("[phantom] frames with phantom hull: %d" % len(phantom_rows))
    print("[phantom] hull-geo px total=%d freed-by-veto=%d (%.1f%%)"
          % (total_hull, total_freed, 100.0 * total_freed / max(total_hull, 1)))
    print("[phantom] per-frame freed fraction: min=%.3f median=%.3f max=%.3f"
          % (min(freed_fracs), float(np.median(freed_fracs)), max(freed_fracs)))
elif PHANTOM:
    print("[phantom] NO PHANTOM HULL IN ANY REPLAYED FRAME")
if COMPOSITE and coverage_rows:
    fractions = [c / max(t, 1) for _, c, t in coverage_rows]
    worst = min(range(len(fractions)), key=lambda i: fractions[i])
    print("[composite] frames with arm painted: %d" % len(coverage_rows))
    print("[composite] mask coverage of painted arm px: min=%.4f median=%.4f"
          % (min(fractions), float(np.median(fractions))))
    print("[composite] worst frame %d: %d/%d covered"
          % coverage_rows[worst])
    coverage_ok = min(fractions) >= 0.995
    print("[composite] COVERAGE:", "PASS" if coverage_ok else "FAIL")
elif COMPOSITE:
    print("[composite] NO ARM PAINTED IN ANY REPLAYED FRAME")
print("VERDICT:", "SOUND" if oracle_violations == 0 else "UNSOUND")
