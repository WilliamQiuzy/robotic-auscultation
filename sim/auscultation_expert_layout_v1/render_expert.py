"""Render recorded contact poses, with physics stopped and input layers immutable.

Run on the Isaac Sim host, for example:
  python -m sim.auscultation_expert_layout_v1.render_expert --point R1 --preview
  python -m sim.auscultation_expert_layout_v1.render_expert --point R1
  python -m sim.auscultation_expert_layout_v1.render_expert --episode /path/to/episode --output /path/to/video

Choose --robotcam-view for the original RGB-D camera RGB view or --posture-view
for an independent report camera. All robot/module
transforms come from the NPZ recording; measurement labels come from JSONL rows.
At 15 fps each displayed sample represents the end of its playback interval.
The preceding recorded pose is used (at most one 30 Hz sample old), without
interpolation, IK, contact solving, or synthetic compression.
"""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess


def validate_replay_timing(rows, indices):
    """Reject missing or stale samples against fixed acquisition rates."""
    import numpy as np
    times = np.asarray([row['time_s'] for row in rows], dtype=float)
    indices = np.asarray(indices)
    if len(times) < 2 or not np.isfinite(times).all():
        raise ValueError('Invalid measurement timestamps')
    if not np.allclose(np.diff(times), 1/240, atol=1e-7, rtol=0):
        raise ValueError('Measurement timestamps violate 240 Hz continuity')
    if (indices.ndim != 1 or len(indices) < 2 or indices.dtype.kind not in 'iu'
            or indices[0] < 0 or indices[-1] >= len(times)
            or not np.all(np.diff(indices) > 0)):
        raise ValueError('Invalid trace-to-pose mapping')
    if np.max(np.diff(times[indices])) > 1/30 + 1e-7:
        raise ValueError('Pose recording violates 30 Hz continuity')
    return times, 1/240, 1/30


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_hashed_json(path, *, lines=False):
    """Parse and fingerprint the same captured bytes, without a second read."""
    data = path.read_bytes()
    checksum = hashlib.sha256(data).hexdigest()
    parsed = ([json.loads(line) for line in data.splitlines() if line.strip()]
              if lines else json.loads(data))
    return parsed, checksum


def verify_replay_inputs(input_files, input_hashes):
    if {path.name: sha256(path) for path in input_files} != input_hashes:
        raise RuntimeError("Replay inputs changed during rendering")


def validate_scene_binding(outcome, artifact_hashes):
    """Require the saved scene to be the exact scene bound by the audited outcome."""
    expected = outcome.get("training_scene_sha256")
    if (not isinstance(expected, str) or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
            or artifact_hashes.get("training_scene.usdc") != expected):
        raise ValueError("Training scene does not match the audited outcome training_scene_sha256")


def front_force(row):
    return sum(float(row.get(k + "_N", 0.)) for k in ("diaphragm", "lip", "contact_rim"))


def replay_presentation(outcome, rows):
    """Describe recorded controller identity; filenames never determine its role."""
    controller = outcome.get("controller_type", "algorithmic_expert")
    role = outcome.get("dataset_role", "expert_demonstration")
    demonstration = outcome.get("expert_demonstration", True)
    if controller == "learned_bc":
        mode = outcome.get("contact_mode")
        if role != "student_evaluation" or demonstration is not False or mode not in ("pure_bc", "hybrid_normal"):
            raise ValueError("Inconsistent recorded controller identity")
        parameterization = outcome.get("motion_parameterization")
        if parameterization == "checked_route_learned_progress_v3":
            if (outcome.get("policy_version") != "v3" or outcome.get("route_planning_learned") is not False
                    or outcome.get("route_direction_source") != "checked planner waypoint minus previous command"
                    or outcome.get("route_progress_min_fraction") != 0.):
                raise ValueError("Inconsistent recorded route/progress identity")
            banner = "Planner route; learned progress; learned contact actions"
            if mode == "hybrid_normal":
                banner += "; analytic normal-force assistance"
        else:
            if parameterization is not None or outcome.get("policy_version") == "v3":
                raise ValueError("Unsupported recorded motion parameterization")
            banner = ("Learned joint actions; analytic normal-force assistance"
                      if mode == "hybrid_normal" else "Learned policy; pure BC joint actions")
        counts = outcome.get("action_source_counts", {})
        if counts.get("supervisor_stationary_hold", 0) > 0 or any(
                row.get("subphase") in ("settle", "align", "withdraw_settle", "home_settle") for row in rows):
            banner += "; stationary holds supervised"
        if counts.get("analytic_fault_recovery", 0) > 0 or any(
                row.get("recovery") or row.get("phase") == "RECOVERY" for row in rows):
            banner += "; analytic fault recovery recorded"
        return {"replay_role": "learned_policy", "controller_type": controller,
                "dataset_role": role, "contact_mode": mode, "expert_demonstration": False,
                "motion_parameterization": parameterization,
                "route_planning_learned": outcome.get("route_planning_learned"),
                "title": "LEARNED POLICY", "control_banner": banner, "filename_prefix": "learned_bc"}
    if controller not in ("algorithmic_expert", "analytic_expert", "expert", "algorithmic_expert_low_elbow") or role != "expert_demonstration" or demonstration is not True:
        raise ValueError("Inconsistent recorded controller identity")
    if controller == "algorithmic_expert_low_elbow":
        return {"replay_role": "expert", "controller_type": controller, "dataset_role": role,
                "contact_mode": outcome.get("contact_mode"), "expert_demonstration": True,
                "title": "LOW-ELBOW EXPERT", "filename_prefix": "low_elbow_expert",
                "control_banner": "APPROACH FROM BELOW | Algorithmic expert; recorded actions; independently checked elbow posture"}
    return {"replay_role": "expert", "controller_type": controller, "dataset_role": role,
            "contact_mode": outcome.get("contact_mode"), "expert_demonstration": True,
            "title": "ALGORITHMIC EXPERT", "control_banner": "Algorithmic expert demonstration; recorded actions only",
            "filename_prefix": "expert"}


def validate_student_export(outcome, rows, preview, audit, summary, artifact_hashes):
    """Full learned demos need bound acceptance; rejected previews are diagnostic."""
    if replay_presentation(outcome, rows)["replay_role"] != "learned_policy":
        return outcome.get("training_accepted") is True
    audit, summary = audit or {}, summary or {}
    points = outcome.get("points", [])
    accepted = (outcome.get("success") is True and outcome.get("training_accepted") is True
        and isinstance(points, list) and bool(points)
        and all(p.get("success") is True and p.get("training_accepted") is True for p in points)
        and bool(rows) and all(r.get("training_accepted") is True and not r.get("recovery")
                              and r.get("phase") not in ("RECOVERY", "ABORT", "FAILED", "FAIL") for r in rows)
        and audit.get("accepted") is True
        and all(name in artifact_hashes and audit.get("artifact_sha256", {}).get(name) == artifact_hashes[name]
                for name in ("expert_trace.jsonl", "control_commands.json", "outcome.json"))
        and all(summary.get(key) is True for key in ("closed_loop_accepted", "controller_success", "independent_audit_accepted")))
    if not accepted and not preview:
        raise ValueError("Recorded student recording is not accepted; use --preview for an explicitly rejected diagnostic")
    return bool(accepted)


def validate_low_elbow_export(outcome, rows, preview, evidence, artifact_hashes):
    """Require the independent posture evidence as well as contact mechanics."""
    if replay_presentation(outcome, rows)["controller_type"] != "algorithmic_expert_low_elbow":
        raise ValueError("Low-elbow export requires the recorded low-elbow controller")
    evidence = evidence if isinstance(evidence, dict) else {}
    reports = [evidence.get(name) for name in ("independent_audit.json", "posture_audit.json",
                                              "verification_summary.json", "runner_summary.json")]
    mechanics, posture, verification, runner = [value if isinstance(value, dict) else {} for value in reports]
    required_mechanics = ("expert_trace.jsonl", "control_commands.json", "outcome.json")
    required_posture = ("expert_trace.jsonl", "expert_replay.npz", "outcome.json")

    def bound(report, key, names):
        fingerprints = report.get(key)
        return (isinstance(fingerprints, dict) and all(
            name in artifact_hashes and fingerprints.get(name) == artifact_hashes[name] for name in names))

    points = outcome.get("points")
    accepted = (outcome.get("success") is True and outcome.get("training_accepted") is True
        and isinstance(points, list) and bool(points)
        and all(isinstance(point, dict) and point.get("success") is True
                and point.get("training_accepted") is True for point in points)
        and [point.get("target") for point in points] == outcome.get("requested_points")
        and bool(rows) and all(row.get("training_accepted") is True and not row.get("recovery")
            and not row.get("posture_violations") and row.get("phase") not in ("RECOVERY", "ABORT", "FAILED", "FAIL")
            for row in rows)
        and all(report.get("accepted") is True for report in (mechanics, posture, verification, runner))
        and bound(mechanics, "artifact_sha256", required_mechanics)
        and bound(posture, "artifact_sha256", required_posture)
        and verification.get("audits") == {"mechanics": mechanics, "posture": posture}
        and runner.get("child_returncode") == 0
        and bound(runner, "evidence_sha256", (*required_mechanics, "expert_replay.npz", "verification_summary.json")))
    if not accepted and not preview:
        raise ValueError("Recorded low-elbow recording is not accepted; use --preview for an explicitly rejected diagnostic")
    return bool(accepted)


def point_records(rows, outcome):
    """Normalize old single-point and continuous outcomes without inferring success."""
    import numpy as np
    records = outcome.get("points") if "points" in outcome else [outcome]
    if not isinstance(records, list) or not records:
        raise ValueError("No per-point camera metadata in outcome")
    points = {}
    for record in records:
        name = record.get("target", "")
        if not isinstance(name, str) or not name.isalnum() or name in points:
            raise ValueError("Invalid or duplicate outcome target")
        for key in ("world_target_m", "command_normal_world"):
            value = np.asarray(record.get(key), dtype=float)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError("Invalid camera metadata for " + name + ": " + key)
        if np.linalg.norm(record["command_normal_world"]) < 1e-9:
            raise ValueError("Zero camera normal for " + name)
        points[name] = dict(record)
        points[name]["training_accepted"] = (outcome.get("training_accepted") is True
                                               and record.get("training_accepted") is True)
    for row in rows:
        if row.get("target") not in points:
            raise ValueError("Trace target has no outcome metadata: " + str(row.get("target")))
    return points


def camera_pose(shot, point):
    """Return a separate report camera pose; never modify sensor-camera state."""
    import numpy as np
    if shot == "overview":
        return (2.7, 3.6, 2.2), (0., .68, .86), 21.
    if shot == "posture":
        # Back/side report view keeps shoulder, elbow and wrist visible.
        # This does not move the RGB-D observation camera.
        return (1.45, -1.0, 1.90), (-.12, .50, 1.24), 22.
    target = np.asarray(point["world_target_m"], dtype=float)
    normal = np.asarray(point["command_normal_world"], dtype=float).copy()
    normal /= np.linalg.norm(normal)
    lateral = np.cross(normal, [0., 0., 1.])
    if np.linalg.norm(lateral) < 1e-6:
        lateral = np.cross(normal, [1., 0., 0.])
    lateral /= np.linalg.norm(lateral)
    # In the recorded ten-point scene, the L3 elbow obscures this side of the
    # module. A report-only reverse angle exposes the contact assembly.
    side = 1. if point.get("target") == "L3" else -1.
    eye = target + .35 * normal + side * .36 * lateral + np.array([0., 0., .05])
    aim = target + .025 * normal + .075 * lateral
    return eye, aim, 28.


def accepted_frame(row, point, outcome):
    """A rejected continuous episode never contributes an accepted frame."""
    return (outcome.get("training_accepted") is True
            and point.get("training_accepted") is True
            and row.get("training_accepted", True) is True
            and not row.get("recovery", False))


def playback_frames(rows, first_pose_row, fps=15, max_motion_speed=2.):
    """Choose existing physics rows; keep each target's HOLD at normal speed.

    Transfer/home and approach motion are capped at 2x until 0.3 seconds before the first
    physical contact. Contact loading, HOLD, and withdrawal within 12 mm are 1x.
    Failure/recovery remains 1x. Nothing is interpolated or repeated to claim a
    longer physical dwell; optional title/end stills are added by the renderer.
    """
    if not math.isfinite(max_motion_speed) or not 1 <= max_motion_speed <= 2:
        raise ValueError("motion playback cap must be between 1x and 2x")
    dt = 1 / 240
    first_contacts = {}
    for i, row in enumerate(rows):
        if front_force(row) > .15:
            first_contacts.setdefault(row["target"], i)
    transit = {"settle", "transfer", "initial_home", "return_bridge", "return_home", "home_settle"}

    def category(i):
        row = rows[i]
        target, phase, subphase = row["target"], row["phase"], row.get("subphase", "")
        if phase == "HOLD":
            return target, 1., "contact", "HOLD"
        if row.get("recovery") or phase in ("RECOVERY", "FAILED", "FAIL"):
            return target, 1., "contact", "RECOVERY"
        if subphase in transit:
            return target, min(8., max_motion_speed), "overview", subphase.upper()
        if phase in ("RETRACT", "DONE") or subphase.startswith("withdraw"):
            near = front_force(row) > .15 or abs(float(row.get("gap_m", 0.))) <= .012
            return target, 1. if near else min(8., max_motion_speed), "contact" if near else "overview", "RETRACT"
        close_start = first_contacts.get(target, len(rows)) - round(.3 / dt)
        if i >= close_start:
            return target, 1., "contact", phase
        return target, min(4., max_motion_speed), "overview", "PRECONTACT"

    frames = []
    start = first_pose_row
    while start < len(rows):
        target, speed, shot, label = category(start)
        end = start + 1
        while end < len(rows) and category(end) == (target, speed, shot, label):
            end += 1
        count = max(1, int(math.ceil((end - start) * dt * fps / speed - 1e-7)))
        for n in range(count):
            index = min(end - 1, start + max(0, round((n + 1) * speed / (fps * dt)) - 1))
            frames.append({"measurement_row": index, "speed": speed, "shot": shot,
                           "target": target, "segment": label})
        start = end
    return frames


def point_dwell_manifest(rows, frames, points, fps=15):
    """Describe the actual trace dwell and rendered dwell, including failed attempts."""
    summary = {}
    for target, record in points.items():
        target_rows = [i for i, row in enumerate(rows) if row["target"] == target]
        hold_rows = [i for i in target_rows if rows[i]["phase"] == "HOLD"]
        hold_frames = [f for f in frames if f["target"] == target and f["phase"] == "HOLD"
                       and f["playback_speed"] == 1]
        intervals = []
        for i in hold_rows:
            if not intervals or intervals[-1][1] != i:
                intervals.append([i, i + 1])
            else:
                intervals[-1][1] = i + 1
        summary[target] = {
            "success": record.get("success", False),
            "training_accepted": record.get("training_accepted", False),
            "verified_front_contact_hold_s": record.get("verified_front_contact_hold_s", 0.),
            "observed_hold_s": record.get("observed_hold_s", 0.),
            "world_target_m": record["world_target_m"], "command_normal_world": record["command_normal_world"],
            "first_physics_row": target_rows[0] if target_rows else None,
            "last_physics_row": target_rows[-1] if target_rows else None,
            "start_simulation_time_s": rows[target_rows[0]]["time_s"] if target_rows else None,
            "end_simulation_time_s": rows[target_rows[-1]]["time_s"] if target_rows else None,
            "hold_physics_steps": len(hold_rows), "hold_trace_duration_s": len(hold_rows) / 240,
            "hold_frames_at_1x": len(hold_frames), "hold_video_duration_s": len(hold_frames) / fps,
            "hold_video_frames": [f["video_frame"] for f in hold_frames],
            "hold_measurement_rows": [f["measurement_row"] for f in hold_frames],
            "hold_simulation_timestamps_s": [f["simulation_time_s"] for f in hold_frames],
            "hold_elapsed_s": [f["hold_elapsed_s"] for f in hold_frames],
            "hold_intervals": [{"start_physics_row": start, "end_physics_row_exclusive": end,
                                "first_sample_time_s": rows[start]["time_s"],
                                "last_sample_time_s": rows[end - 1]["time_s"],
                                "duration_s": (end - start) / 240} for start, end in intervals]}
    return summary


def recorded_pose_digest(poses, sample_indices):
    """Hash the exact selected recorded arrays, including dtype and sample indices."""
    import numpy as np
    selected = np.asarray(sorted(set(sample_indices)), dtype="<i8")
    values = np.ascontiguousarray(poses[selected])
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(json.dumps(list(values.shape)).encode("ascii"))
    digest.update(selected.tobytes())
    digest.update(values.tobytes())
    return {"recorded_pose_indices": selected.tolist(),
            "recorded_pose_samples_sha256": digest.hexdigest()}


def recorded_schedule_digest(frames):
    """Fingerprint the complete displayed-frame evidence in stable JSON form."""
    return hashlib.sha256(json.dumps(frames, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def annotation_font(image_font, size=28):
    """Use an installed scalable font, then Pillow's default, without installing fonts."""
    for path in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                 '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
                 '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
                 'DejaVuSans.ttf'):
        try:
            return image_font.truetype(path, size), path
        except OSError:
            pass
    try:
        return image_font.load_default(size=size), 'Pillow bundled default'
    except TypeError:
        return image_font.load_default(), 'Pillow bundled bitmap default'


def read_camera_reference(episode, outcome):
    """Read camera metadata from the immutable, outcome-bound physics snapshot."""
    snapshot_path = episode / 'input_snapshot/manifest.json'
    snapshot, checksum = read_hashed_json(snapshot_path)
    if outcome.get('input_snapshot_manifest_sha256') != checksum:
        raise ValueError('Camera reference snapshot does not match the accepted outcome')
    entry = snapshot['files']['source/scene/layout_inputs']
    checksum_object = entry['sha256']
    if (len(checksum_object) != 64
            or any(character not in '0123456789abcdef' for character in checksum_object)):
        raise ValueError('Invalid camera reference object hash')
    object_name = 'input_snapshot/objects/' + checksum_object
    reference, checksum_captured = read_hashed_json(episode / object_name)
    if checksum_captured != checksum_object:
        raise ValueError('Camera reference object changed')
    camera = reference['camera']
    for key in ('width', 'height'):
        if not isinstance(camera[key], int) or camera[key] <= 0 or camera[key] % 2:
            raise ValueError('Camera resolution must contain positive even integers')
    return camera, {'input_snapshot/manifest.json': checksum, object_name: checksum_captured}


def verify_reference_inputs(episode, references):
    if any(sha256(episode / name) != checksum for name, checksum in references.items()):
        raise RuntimeError('Camera reference inputs changed during rendering')


def sensor_camera_state(stage, width, height):
    """Capture actual sensor pose and optics without authoring any camera attributes."""
    import numpy as np
    from pxr import UsdGeom
    camera = UsdGeom.Camera(stage.GetPrimAtPath('/World/RobotCam'))
    if not camera:
        raise ValueError('Original /World/RobotCam is missing')
    world = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(camera.GetPrim()), dtype=float).T
    focal, horizontal, vertical = [float(attr.Get()) for attr in (
        camera.GetFocalLengthAttr(), camera.GetHorizontalApertureAttr(), camera.GetVerticalApertureAttr())]
    ho, vo = float(camera.GetHorizontalApertureOffsetAttr().Get()), float(camera.GetVerticalApertureOffsetAttr().Get())
    return {'world_from_optical': (world @ np.diag([1., -1., -1., 1.])).tolist(),
            'focal_length': focal, 'horizontal_aperture': horizontal, 'vertical_aperture': vertical,
            'horizontal_aperture_offset': ho, 'vertical_aperture_offset': vo,
            'clipping_range': list(camera.GetClippingRangeAttr().Get()),
            'fx': focal / horizontal * width, 'fy': focal / vertical * height,
            'cx': width / 2 - ho / horizontal * width, 'cy': height / 2 + vo / vertical * height,
            'width': width, 'height': height}


def check_camera_reference(actual, expected):
    import numpy as np
    if set(actual) != set(expected):
        raise ValueError('Camera reference fields changed')
    for key in expected:
        if not np.allclose(actual[key], expected[key], rtol=0, atol=1e-10):
            raise ValueError('Original sensor camera changed: ' + key)


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point", default="R1")
    parser.add_argument("--episode", type=Path, help="absolute directory containing a recorded episode")
    parser.add_argument("--output", type=Path, help="output directory outside the immutable episode")
    parser.add_argument("--preview", action="store_true")
    views = parser.add_mutually_exclusive_group()
    views.add_argument("--robotcam-view", action="store_true",
                       help="render the unchanged original /World/RobotCam at its native resolution")
    views.add_argument("--posture-view", action="store_true",
                        help="keep a back/side view of the whole arm throughout playback")
    parser.add_argument("--motion-speed", type=int, choices=(1, 2), default=2,
                        help="maximum motion playback multiplier; contact HOLD always stays 1x")
    return parser


def main():
    parser = argument_parser()
    args = parser.parse_args()
    if not args.point.isalnum():
        parser.error("point must contain only letters and digits")
    root = Path(__file__).resolve().parent
    if args.episode and not args.episode.is_absolute():
        parser.error("--episode must be an absolute directory")
    inputs = (args.episode or root / "results" / ("expert_" + args.point)).resolve()
    name = inputs.name if args.episode else args.point
    view_mode = "robotcam" if args.robotcam_view else "posture" if args.posture_view else "report"
    out = (args.output or root / "results" / ("video_" + name + "_" + view_mode)).resolve()
    if out == inputs or inputs in out.parents:
        parser.error("--output must be outside the immutable episode directory")
    source = inputs / "training_scene.usdc"
    trace = inputs / "expert_trace.jsonl"
    replay = inputs / "expert_replay.npz"
    outcome_path = inputs / "outcome.json"
    input_files = [source, trace, replay, outcome_path]
    # The NPZ consumed below is exactly the byte sequence whose fingerprint is
    # matched to the posture auditor, not a second read of a mutable pathname.
    replay_bytes = replay.read_bytes()
    input_hashes = {source.name: sha256(source), replay.name: hashlib.sha256(replay_bytes).hexdigest()}
    rows, input_hashes[trace.name] = read_hashed_json(trace, lines=True)
    outcome, input_hashes[outcome_path.name] = read_hashed_json(outcome_path)
    presentation = replay_presentation(outcome, rows)
    if presentation["controller_type"] != "algorithmic_expert_low_elbow":
        raise ValueError("This isolated layout renderer requires the independently audited algorithmic low-elbow expert")
    if (presentation["replay_role"] == "learned_policy"
            or presentation["controller_type"] == "algorithmic_expert_low_elbow"):
        evidence = {}
        evidence_names = (("control_commands.json", "independent_audit.json", "posture_audit.json",
                           "verification_summary.json", "runner_summary.json")
                          if presentation["controller_type"] == "algorithmic_expert_low_elbow" else
                          ("control_commands.json", "independent_audit.json", "evaluation_summary.json"))
        for filename in evidence_names:
            path = inputs / filename
            if path.is_file():
                input_files.append(path)
                if filename == "control_commands.json":
                    input_hashes[filename] = sha256(path)
                else:
                    evidence[filename], input_hashes[filename] = read_hashed_json(path)
        if presentation["controller_type"] == "algorithmic_expert_low_elbow":
            accepted = validate_low_elbow_export(outcome, rows, args.preview, evidence, input_hashes)
        else:
            accepted = validate_student_export(outcome, rows, args.preview,
                evidence.get("independent_audit.json"), evidence.get("evaluation_summary.json"), input_hashes)
        # Keep source outcome untouched. Diagnostic overlays and manifests must
        # not inherit an acceptance flag contradicted by the independent audit.
        render_outcome = dict(outcome, training_accepted=accepted)
        presentation["rejected_diagnostic"] = not accepted
        if not accepted:
            presentation["title"] = "REJECTED DIAGNOSTIC"
            if presentation["controller_type"] == "algorithmic_expert_low_elbow":
                presentation["control_banner"] = "APPROACH FROM BELOW | Algorithmic expert diagnostic; posture/contact acceptance not established"
    else:
        render_outcome = outcome
        presentation["rejected_diagnostic"] = False
    validate_scene_binding(outcome, input_hashes)
    camera_reference, reference_hashes = read_camera_reference(inputs, outcome)
    width, height = ((camera_reference["width"], camera_reference["height"])
                     if args.robotcam_view else (1280, 720))
    points = point_records(rows, render_outcome)
    verify_replay_inputs(input_files, input_hashes)
    out.mkdir(parents=True, exist_ok=False)
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "width": width, "height": height,
                         "renderer": "RayTracedLighting", "anti_aliasing": 3,
                         "multi_gpu": False})
    encoder = None
    try:
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        from pxr import Usd, UsdGeom, UsdPhysics, Gf
        import omni.usd
        import omni.timeline
        import omni.replicator.core as rep

        record = np.load(io.BytesIO(replay_bytes), allow_pickle=False)
        poses, indices, paths = record["poses"], record["row_indices"], record["paths"]
        if poses.shape != (len(indices), len(paths), 7) or not np.isfinite(poses).all():
            raise ValueError("Invalid pose recording")
        times, dt, pose_period = validate_replay_timing(rows, indices)
        fps = 15

        # Flatten to an anonymous layer before edits; source files remain read-only.
        original = Usd.Stage.Open(str(source))
        original_camera = sensor_camera_state(original, camera_reference["width"], camera_reference["height"])
        check_camera_reference(original_camera, camera_reference)
        stage = Usd.Stage.Open(original.Flatten())
        del original
        for path in ("/Render", "/Replicator", "/Orchestrator"):
            if stage.GetPrimAtPath(path):
                stage.RemovePrim(path)
        disabled_bodies = disabled_joints = 0
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
                disabled_bodies += 1
            if prim.IsA(UsdPhysics.Joint):
                UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
                disabled_joints += 1
        disposable = out / "replay_render.usdc"
        stage.GetRootLayer().Export(str(disposable))
        del stage
        omni.usd.get_context().open_stage(str(disposable))
        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        for _ in range(45):
            app.update()
        stage = omni.usd.get_context().get_stage()
        cache = UsdGeom.XformCache()
        path_set = set(map(str, paths))
        transforms = []
        for path in paths:
            prim = stage.GetPrimAtPath(str(path))
            if not prim:
                raise ValueError("Recorded body missing from stage: " + str(path))
            ancestor = prim.GetParent()
            while ancestor and not ancestor.IsPseudoRoot():
                if str(ancestor.GetPath()) in path_set:
                    raise ValueError("Expected static parent for replay body: " + str(path))
                ancestor = ancestor.GetParent()
            parent_inv = cache.GetLocalToWorldTransform(prim.GetParent()).GetInverse()
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            transforms.append((xf.AddTransformOp(opSuffix="recordedReplay"), parent_inv))

        if args.robotcam_view:
            camera = UsdGeom.Camera(stage.GetPrimAtPath('/World/RobotCam'))
            camera_op = None
        else:
            camera = UsdGeom.Camera.Define(stage, "/World/RecordedReplayReportCamera")
            camera.CreateClippingRangeAttr(Gf.Vec2f(.001, 100.))
            camera.CreateHorizontalApertureAttr(20.955)
            camera_xf = UsdGeom.Xformable(camera)
            camera_xf.ClearXformOpOrder()
            camera_op = camera_xf.AddTransformOp()
        product = rep.create.render_product(str(camera.GetPath()), (width, height))
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach([product])
        rep.orchestrator.set_capture_on_play(False)
        for _ in range(30):
            app.update()
        fonts = {n: annotation_font(ImageFont, n) for n in (16, 18, 20, 23, 29)}
        regular = {n: value[0] for n, value in fonts.items()}
        font_sources = {str(n): value[1] for n, value in fonts.items()}
        static_time = timeline.get_current_time()
        manifest = []

        def set_camera(shot, target_name):
            if args.robotcam_view:
                return
            eye, aim, focal = camera_pose(shot, points[target_name])
            camera_op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*aim),
                                                  Gf.Vec3d(0., 0., 1.)).GetInverse())
            camera.GetFocalLengthAttr().Set(focal)

        def render(row_index, speed, shot, freeze=False, save=None):
            if args.robotcam_view:
                shot = "robotcam"
            elif args.posture_view:
                shot = "posture"
            row = rows[row_index]
            target_name = row["target"]
            accepted = accepted_frame(row, points[target_name], render_outcome)
            k = int(np.searchsorted(indices, row_index, side="right") - 1)
            if k < 0:
                raise ValueError("No recorded pose precedes measurement")
            age = row["time_s"] - rows[int(indices[k])]["time_s"]
            if age < -1e-8 or age > pose_period + dt + 1e-7:
                raise ValueError("Pose and measurement timestamps are too far apart")
            for (op, inv), pose in zip(transforms, poses[k]):
                matrix = Gf.Matrix4d(1.)
                matrix.SetRotate(Gf.Quatd(float(pose[3]), Gf.Vec3d(*map(float, pose[4:]))))
                matrix.SetTranslateOnly(Gf.Vec3d(*map(float, pose[:3])))
                op.Set(matrix * inv)
            set_camera(shot, target_name)
            rep.orchestrator.step(rt_subframes=3, delta_time=0., pause_timeline=True)
            if timeline.is_playing() or timeline.get_current_time() != static_time:
                raise RuntimeError("Physics timeline must remain stopped during replay")
            check_camera_reference(sensor_camera_state(stage, camera_reference["width"],
                                                       camera_reference["height"]), original_camera)
            frame = annotator.get_data()
            if frame.ndim != 3 or frame.shape[:2] != (height, width):
                raise RuntimeError("Invalid RGB capture")
            im = Image.fromarray(frame[:, :, :3]).convert("RGBA")
            overlay = Image.new("RGBA", (1280, 720), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay, "RGBA")
            draw.rectangle((0, 0, 1280, 94), fill=(13, 21, 31, 245))
            title = presentation["title"] + " / " + target_name
            draw.text((25, 12), title, font=regular[29], fill="white")
            playback = "PAUSED RECORDED FRAME" if freeze else f"PLAYBACK {speed:g}x"
            draw.text((27, 57), f"Target {target_name}  |  {row['phase']} / {row.get('subphase', '')}  |  {playback}",
                      font=regular[18], fill=(151, 214, 227))
            draw.text((817, 18), f"Simulation time  {row['time_s']:.3f} s", font=regular[23], fill="white")
            draw.text((819, 55), f"Recorded pose time  {rows[int(indices[k])]['time_s']:.3f} s",
                      font=regular[18], fill=(174, 195, 205))
            draw.rounded_rectangle((20, 112, 434, 349), radius=12, fill=(13, 21, 31, 238))
            items = [("Nominal Hall estimate", f"{row['hall_N']:.3f} N"),
                     ("Physical front force", f"{front_force(row):.3f} N"),
                     ("Membrane force", f"{row['diaphragm_N']:.3f} N"),
                     ("Compression", f"{row['compression_m'] * 1000:.3f} mm"),
                     ("Recorded hold elapsed", f"{row['hold_elapsed_s']:.3f} s")]
            for i, (label, value) in enumerate(items):
                y = 128 + i * 40
                draw.text((36, y), label, font=regular[18], fill=(189, 205, 217))
                draw.text((419, y - 2), value, font=regular[20], fill=(104, 225, 214), anchor="ra")
            draw.text((35, 330), "Front = membrane + retaining lip + contact rim", font=regular[16], fill=(169, 193, 204))
            if front_force(row) > .15:
                draw.rounded_rectangle((20, 367, 434, 430), radius=10, fill=(13, 21, 31, 238))
                detail = f"Lip {row['lip_N']:.3f} N   |   Rim {row['contact_rim_N']:.3f} N"
                draw.text((36, 379), detail, font=regular[18], fill="white")
                draw.text((36, 405), "Front-ring contact; no membrane load observed" if row['diaphragm_N'] < .001
                          else "Measured contact components shown separately", font=regular[16], fill=(239, 199, 128))
            draw.rectangle((0, 619, 1280, 720), fill=(13, 21, 31, 242))
            view_label = "ORIGINAL ROBOTCAM RGB VIEW" if args.robotcam_view else "REPORT CAMERA / " + shot.upper()
            draw.text((25, 628), view_label + " | Static physics timeline; recorded poses | Motion <=2x; HOLD 1x",
                      font=regular[16], fill=(237, 210, 157))
            acceptance = "ACCEPTED" if accepted else "NOT ACCEPTED"
            if freeze and row_index == len(rows) - 1:
                dwell = points[target_name].get("verified_front_contact_hold_s", 0.)
                status = f"Recorded result: {acceptance}  |  {target_name} front-contact hold: {dwell:.3f} s"
            else:
                status = f"Algorithmic expert | Recorded result: {acceptance}  |  240 Hz measurements / 30 Hz poses"
            draw.text((25, 660), status, font=regular[18], fill=(178, 209, 223))
            draw.text((25, 690), "Simulation contact mechanics only. No acoustic or clinical validation. Target is model-proposed.",
                      font=regular[16], fill=(178, 192, 203))
            im = Image.alpha_composite(im, overlay.resize(im.size, Image.Resampling.LANCZOS)).convert("RGB")
            if save:
                im.save(out / save)
            if encoder:
                encoder.stdin.write(np.ascontiguousarray(np.asarray(im)).tobytes())
            manifest.append({"video_frame": len(manifest), "measurement_row": row_index,
                             "target": target_name, "training_accepted": accepted,
                             "pose_row": int(indices[k]), "pose_index": k,
                             "simulation_time_s": row["time_s"], "pose_age_s": age,
                             "phase": row["phase"], "hold_elapsed_s": row["hold_elapsed_s"],
                             "physical_front_N": front_force(row), "membrane_N": row["diaphragm_N"],
                             "playback_speed": 0 if freeze else speed, "shot": shot})

        first_pose_row = int(indices[0])
        if args.preview:
            render(first_pose_row, 0, "overview", freeze=True, save="preview_overview.png")
            for target_name in points:
                target_rows = [i for i, r in enumerate(rows) if r["target"] == target_name and i >= first_pose_row]
                if not target_rows:
                    continue
                hold_rows = [i for i in target_rows if rows[i]["phase"] == "HOLD"]
                representative = hold_rows[-1] if hold_rows else max(target_rows, key=lambda i: front_force(rows[i]))
                suffix = "" if len(points) == 1 else "_" + target_name
                render(representative, 0, "overview", freeze=True, save="preview_contact_overview" + suffix + ".png")
                render(representative, 0, "contact", freeze=True, save="preview_contact" + suffix + ".png")
        else:
            import imageio_ffmpeg
            video = out / (presentation["filename_prefix"] + "_" + name + "_recorded.mp4")
            encoder = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                                        "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
                                        "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                                        str(video)], stdin=subprocess.PIPE)
            for i in range(2 * fps):
                render(first_pose_row, 0, "overview", freeze=True,
                       save="preview_overview.png" if i == 0 else None)
            previous_target = None
            for spec in playback_frames(rows, first_pose_row, fps, max_motion_speed=args.motion_speed):
                if spec["target"] != previous_target:
                    print("RENDER_POINT", spec["target"], flush=True)
                    previous_target = spec["target"]
                render(spec["measurement_row"], spec["speed"], spec["shot"])
            for _ in range(2 * fps):
                render(len(rows) - 1, 0, "overview", freeze=True)
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError("ffmpeg encoder failed")
            encoder = None
        verify_replay_inputs(input_files, input_hashes)
        verify_reference_inputs(inputs, reference_hashes)
        dwell_summary = point_dwell_manifest(rows, manifest, points, fps)
        for target_name, point_summary in dwell_summary.items():
            point_summary.update(recorded_pose_digest(poses, [f["pose_index"] for f in manifest
                                                               if f["target"] == target_name]))
            point_summary["source_trace_sha256"] = input_hashes[trace.name]
            point_summary["source_replay_sha256"] = input_hashes[replay.name]
        summary = {"point": next(iter(points)) if len(points) == 1 else None,
                   "episode_directory": str(inputs), "points": dwell_summary,
                   "preview": args.preview, "resolution": [width, height],
                   "fps": fps, "frame_count": len(manifest), "inputs_sha256": input_hashes,
                   "source_layers_unchanged": True, "physics_timeline_stopped": True,
                   "disabled_rigid_bodies": disabled_bodies, "disabled_joints": disabled_joints,
                   "original_rgbd_camera_unchanged": True,
                   "camera_reference": camera_reference, "camera_reference_inputs_sha256": reference_hashes,
                   "render_camera_path": str(camera.GetPath()),
                   "view_mode": "robotcam" if args.robotcam_view else "posture" if args.posture_view else "report",
                   "robotcam_view": args.robotcam_view, "camera_pose_optics_verified_every_frame": True,
                   "physics_timeline_mode": "stopped_static_pose_replay", "physics_timeline_time_s": static_time,
                   "pose_application": "recorded_raw_rigid_body_transforms_no_interpolation",
                   "annotation_fonts": font_sources,
                   "posture_view": args.posture_view, "motion_playback_speed_cap": args.motion_speed,
                   "recorded_pose_count": len(indices), "physics_trace_rows": len(rows),
                   "hold_frames_at_1x": sum(f["phase"] == "HOLD" and f["playback_speed"] == 1 for f in manifest),
                   "training_accepted": render_outcome.get("training_accepted", False),
                   "controller_presentation": presentation,
                   "verified_front_contact_hold_s": outcome.get("verified_front_contact_hold_s"),
                   "scope": outcome["scope"], "frames": manifest,
                   "playback_schedule_sha256": recorded_schedule_digest(manifest),
                   "recorded_poses": recorded_pose_digest(poses, [frame["pose_index"] for frame in manifest]),
                   "video_sha256": None if args.preview else sha256(video)}
        (out / ("preview_manifest.json" if args.preview else "video_manifest.json")).write_text(json.dumps(summary, indent=2))
        print("RENDER_COMPLETE", len(manifest), str(out), flush=True)
    finally:
        if encoder is not None:
            if encoder.stdin and not encoder.stdin.closed:
                encoder.stdin.close()
            encoder.wait()
        app.close()


if __name__ == "__main__":
    main()
