"""Run the robot-occlusion strategy study and render a report-ready Isaac Sim video.

The experiment uses a deterministic moving-back constellation and a kinematic robot-arm proxy in
Isaac Sim. Occlusion is computed from the Orbbec line of sight. The proxy intentionally sweeps a
range of arm silhouette radii; it is a perception/observability experiment, not a contact-dynamics
or OpenArm-controller validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Iterable

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--render-scenario", default="moving_patient")
    parser.add_argument("--render-seed", type=int, default=7)
    parser.add_argument("--render-max-frames", type=int, default=0)
    parser.add_argument("--rt-subframes", type=int, default=2)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--experiment-code-id", default="unbound")
    return parser.parse_args()


def default_scenarios():
    from ausc.perception.tracking.occlusion_experiment import Scenario

    return [
        Scenario(
            name="short_low_motion",
            duration_s=7.0,
            patient_sway_m=0.012,
            occluder_radius_m=0.045,
            contact_start_s=3.2,
            contact_end_s=4.2,
            depart_end_s=6.0,
        ),
        Scenario(
            name="moving_patient",
            duration_s=9.0,
            patient_sway_m=0.045,
            occluder_radius_m=0.055,
        ),
        Scenario(
            name="wide_arm",
            duration_s=9.0,
            patient_sway_m=0.045,
            occluder_radius_m=0.090,
        ),
        Scenario(
            name="severe_single_view_loss",
            duration_s=9.0,
            patient_sway_m=0.045,
            occluder_radius_m=0.140,
            max_freeze_frames=8,
        ),
    ]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _finite_error_mm(estimate: np.ndarray, truth: np.ndarray) -> np.ndarray:
    error = np.linalg.norm(estimate - truth, axis=1) * 1000.0
    return error


def _rgb_has_scene_content(frame: np.ndarray) -> bool:
    """Reject technically decodable but effectively black Isaac camera frames."""
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] < 3 or image.size == 0:
        return False
    luminance = image[..., :3].astype(np.float32).mean(axis=2)
    return bool(np.mean(luminance > 8.0) >= 0.01 and np.percentile(luminance, 99) > 20.0)


def _write_results(out: Path, scenarios: Iterable, seed_count: int, code_id: str):
    from ausc.perception.tracking.occlusion_experiment import simulate_trial, summarize_trials

    rows = []
    summaries = {}
    representative = {}
    for scenario in scenarios:
        trials = []
        for seed in range(seed_count):
            trial = simulate_trial(scenario, seed)
            trials.append(trial)
            one = summarize_trials([trial])
            rows.append({"scenario": scenario.name, "seed": seed, **one})
        summaries[scenario.name] = summarize_trials(trials)
        representative[scenario.name] = simulate_trial(scenario, seed=7)

    csv_path = out / "per_seed_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    freeze = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True, stderr=subprocess.STDOUT,
    )
    (out / "isaac_environment_freeze.txt").write_text(freeze, encoding="utf-8")
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "isaacsim": _package_version("isaacsim"),
        "isaaclab": _package_version("isaaclab"),
        "numpy": np.__version__,
        "pip_freeze_sha256": _sha256_bytes(freeze.encode("utf-8")),
        "experiment_code_id": code_id,
    }
    document = {
        "schema": "wam_isaac_occlusion_experiment_v1",
        "design": {
            "strategies": ["camera_lock", "patient_frame_lock", "patient_lock_then_robot_fk"],
            "seeds_per_scenario": seed_count,
            "line_of_sight": "single fixed Orbbec view with a kinematic arm capsule mask",
            "note": "Isaac kinematic observability study; not contact dynamics or full OpenArm URDF control",
        },
        "environment": environment,
        "scenarios": summaries,
    }
    (out / "metrics.json").write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8",
    )
    (out / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8",
    )
    return summaries, representative


def _package_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001
        return "unknown"


def _to_world(points: np.ndarray) -> np.ndarray:
    value = np.asarray(points, dtype=float)
    return np.stack([value[..., 0], value[..., 2], value[..., 1]], axis=-1)


def _quaternion_from_z(direction: np.ndarray) -> np.ndarray:
    vector = np.asarray(direction, dtype=float)
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    target = vector / length
    z_axis = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(z_axis @ target, -1.0, 1.0))
    if dot < -0.999999:
        return np.array([0.0, 1.0, 0.0, 0.0])
    cross = np.cross(z_axis, target)
    scalar = np.sqrt((1.0 + dot) * 0.5)
    xyz = cross / max(2.0 * scalar, 1e-12)
    return np.array([scalar, xyz[0], xyz[1], xyz[2]])


def _phase(trial: dict, frame: int) -> str:
    if bool(trial["contact"][frame]):
        return "CONTACT: patient lock -> calibrated robot FK"
    if not bool(trial["visible"][frame, int(trial["scenario"]["target_index"])]):
        return "OCCLUDED: patient-relative prediction"
    return "VISIBLE: live vision measurement"


def _dashboard(trial: dict, frame: int, width: int = 1440, height: int = 270) -> np.ndarray:
    import cv2

    canvas = np.full((height, width, 3), 22, dtype=np.uint8)
    truth = np.asarray(trial["truth"])
    colors = {
        "camera_lock": (70, 70, 235),
        "patient_lock": (40, 220, 245),
        "hybrid": (235, 210, 35),
    }
    labels = {
        "camera_lock": "Camera lock",
        "patient_lock": "Patient-frame lock",
        "hybrid": "Patient lock -> robot FK",
    }
    errors = {key: _finite_error_mm(np.asarray(trial[key]), truth) for key in colors}
    cv2.putText(canvas, _phase(trial, frame), (30, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (245, 245, 245), 2)
    support = int(trial["visible_support"][frame])
    source = str(trial["patient_source"][frame])
    cv2.putText(
        canvas,
        f"t={trial['time_s'][frame]:.2f}s   visible support={support}/10   state={source}",
        (30, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (185, 185, 185),
        1,
    )

    bar_left, bar_width = 30, 430
    for row, key in enumerate(colors):
        y = 112 + row * 46
        value = float(errors[key][frame])
        shown = min(value, 80.0) if np.isfinite(value) else 80.0
        cv2.putText(canvas, labels[key], (bar_left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.54, colors[key], 2)
        cv2.rectangle(canvas, (205, y - 17), (205 + bar_width, y + 3), (55, 55, 55), -1)
        cv2.rectangle(canvas, (205, y - 17), (205 + int(bar_width * shown / 80.0), y + 3), colors[key], -1)
        value_text = "LOST" if not np.isfinite(value) else f"{value:5.1f} mm"
        cv2.putText(canvas, value_text, (650, y), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (235, 235, 235), 1)

    chart_x0, chart_y0, chart_w, chart_h = 800, 28, 600, 205
    cv2.rectangle(canvas, (chart_x0, chart_y0), (chart_x0 + chart_w, chart_y0 + chart_h), (55, 55, 55), 1)
    cv2.putText(canvas, "Target error over time (0-80 mm)", (chart_x0 + 8, chart_y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
    total = len(truth)
    for key, color in colors.items():
        points = []
        for index in range(frame + 1):
            value = errors[key][index]
            if not np.isfinite(value):
                continue
            x = chart_x0 + int(chart_w * index / max(total - 1, 1))
            y = chart_y0 + chart_h - int((chart_h - 28) * min(value, 80.0) / 80.0)
            points.append((x, y))
        if len(points) > 1:
            cv2.polylines(canvas, [np.asarray(points, np.int32)], False, color, 2)
    return canvas


def _render_isaac_video(trial: dict, out: Path, max_frames: int, rt_subframes: int) -> None:
    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": True,
            "hide_ui": True,
            "multi_gpu": False,
            "active_gpu": 0,
            "physics_gpu": 0,
            "anti_aliasing": 0,
            "samples_per_pixel_per_frame": 1,
            "width": 1440,
            "height": 810,
            "disable_viewport_updates": False,
            "fast_shutdown": False,
        }
    )

    import cv2
    import omni.replicator.core as rep
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import FixedCuboid, VisualCuboid, VisualCylinder, VisualSphere

    world = World(stage_units_in_meters=1.0)
    world.scene.add(
        FixedCuboid(
            prim_path="/World/Backdrop",
            name="backdrop",
            position=np.array([0.0, 1.23, 0.0]),
            scale=np.array([1.3, 0.025, 0.9]),
            color=np.array([0.13, 0.16, 0.20]),
        )
    )
    back = world.scene.add(
        VisualCuboid(
            prim_path="/World/PatientBack",
            name="patient_back",
            position=np.array([0.0, 1.05, 0.0]),
            scale=np.array([0.42, 0.04, 0.50]),
            size=1.0,
            color=np.array([0.55, 0.22, 0.16]),
        )
    )
    head = world.scene.add(
        VisualSphere(
            prim_path="/World/PatientHead", name="patient_head", radius=0.105,
            position=np.array([0.0, 1.055, 0.36]), color=np.array([0.50, 0.18, 0.13]),
        )
    )
    landmarks = []
    target_index = int(trial["scenario"]["target_index"])
    for index in range(10):
        color = np.array([0.20, 0.95, 0.35]) if index == target_index else np.array([0.92, 0.92, 0.92])
        landmarks.append(
            world.scene.add(
                VisualSphere(
                    prim_path=f"/World/Landmark_{index}",
                    name=f"landmark_{index}",
                    radius=0.015 if index == target_index else 0.010,
                    color=color,
                )
            )
        )
    camera_estimate = world.scene.add(
        VisualSphere(prim_path="/World/CameraLock", name="camera_lock", radius=0.014, color=np.array([0.92, 0.12, 0.12]))
    )
    patient_estimate = world.scene.add(
        VisualSphere(prim_path="/World/PatientLock", name="patient_lock", radius=0.012, color=np.array([0.95, 0.78, 0.08]))
    )
    hybrid_estimate = world.scene.add(
        VisualSphere(prim_path="/World/Hybrid", name="hybrid", radius=0.010, color=np.array([0.10, 0.82, 0.95]))
    )
    arm_links = []
    silhouette_radius = float(trial["scenario"]["occluder_radius_m"])
    for index, radius in enumerate(
        (0.85 * silhouette_radius, 0.72 * silhouette_radius, 0.55 * silhouette_radius)
    ):
        arm_links.append(
            world.scene.add(
                VisualCylinder(
                    prim_path=f"/World/RobotArm/Link_{index}",
                    name=f"robot_link_{index}",
                    radius=radius,
                    height=1.0,
                    color=np.array([0.08, 0.30 + index * 0.08, 0.80]),
                )
            )
        )
    tool = world.scene.add(
        VisualCylinder(
            prim_path="/World/RobotArm/Stethoscope",
            name="stethoscope",
            radius=0.025,
            height=0.06,
            color=np.array([0.05, 0.05, 0.05]),
        )
    )

    front_camera = rep.create.camera(position=(0.0, 0.03, 0.0), look_at=(0.0, 1.0, 0.0), focal_length=22.0)
    side_camera = rep.create.camera(
        position=(0.78, -0.28, 0.50), look_at=(0.0, 1.0, 0.02), focal_length=24.0,
    )
    rep.create.light(light_type="dome", intensity=550.0, temperature=5600, name="FillLight")
    rep.create.light(
        light_type="distant", rotation=(315.0, 15.0, 25.0), intensity=2600.0,
        temperature=5200, name="KeyLight",
    )
    front_product = rep.create.render_product(front_camera, (720, 540))
    side_product = rep.create.render_product(side_camera, (720, 540))
    front_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    side_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    front_rgb.attach(front_product)
    side_rgb.attach(side_product)
    world.reset()
    for _ in range(5):
        rep.orchestrator.step(rt_subframes=2)

    fps = int(trial["scenario"]["fps"])
    total = len(trial["time_s"])
    if max_frames > 0:
        total = min(total, max_frames)
    video_path = out / "isaac_occlusion_comparison.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1440, 810))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open the MP4 writer")
    poster_frame = min(total - 1, int(round(5.0 * fps)))
    poster = None

    try:
        for frame in range(total):
            truth_points = np.asarray(trial["truth_points"])[frame]
            world_truth = _to_world(truth_points)
            body_center = np.array(
                [world_truth[:, 0].mean(), world_truth[:, 1].max() + 0.025, world_truth[:, 2].mean()]
            )
            back.set_world_pose(body_center)
            head.set_world_pose(body_center + np.array([0.0, 0.010, 0.36]))
            for prim, point in zip(landmarks, world_truth):
                prim.set_world_pose(point)

            arm = _to_world(np.asarray(trial["arm_keypoints"])[frame])
            for prim, first, second in zip(arm_links, arm[:-1], arm[1:]):
                delta = second - first
                prim.set_world_pose((first + second) / 2.0, _quaternion_from_z(delta))
                prim.set_local_scale(np.array([1.0, 1.0, max(float(np.linalg.norm(delta)), 1e-4)]))
            tool_delta = np.array([0.0, -0.06, 0.0])
            tool.set_world_pose(arm[-1] + np.array([0.0, -0.025, 0.0]), _quaternion_from_z(tool_delta))

            for prim, key, offset in (
                (camera_estimate, "camera_lock", -0.030),
                (patient_estimate, "patient_lock", -0.045),
                (hybrid_estimate, "hybrid", -0.060),
            ):
                estimate = np.asarray(trial[key])[frame]
                if np.isfinite(estimate).all():
                    position = _to_world(estimate)
                    position[1] += offset
                else:
                    position = np.array([5.0, 5.0, 5.0])
                prim.set_world_pose(position)

            world.step(render=False)
            rep.orchestrator.step(rt_subframes=max(rt_subframes, 1))
            front = np.asarray(front_rgb.get_data())[..., :3].astype(np.uint8)
            side = np.asarray(side_rgb.get_data())[..., :3].astype(np.uint8)
            if frame == 0 and not (_rgb_has_scene_content(front) and _rgb_has_scene_content(side)):
                raise RuntimeError("Isaac render gate rejected an effectively black camera frame")
            front = cv2.cvtColor(front, cv2.COLOR_RGB2BGR)
            side = cv2.cvtColor(side, cv2.COLOR_RGB2BGR)
            cv2.putText(front, "Orbbec view", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 245, 245), 2)
            cv2.putText(side, "3D side view", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 245, 245), 2)
            upper = np.hstack([front, side])
            dashboard = _dashboard(trial, frame)
            composed = np.vstack([upper, dashboard])
            writer.write(composed)
            if frame == poster_frame:
                poster = composed.copy()
            if (frame + 1) % 30 == 0:
                print(f"ISAAC_RENDER {frame + 1}/{total}", flush=True)

        if poster is not None:
            cv2.imwrite(str(out / "isaac_occlusion_report_frame.png"), poster)
        print(f"ISAAC_VIDEO {video_path} frames={total} fps={fps}", flush=True)
    finally:
        writer.release()
        rep.orchestrator.wait_until_complete()
        simulation_app.close()


def _write_representative_npz(trial: dict, out: Path) -> None:
    arrays = {key: value for key, value in trial.items() if isinstance(value, np.ndarray)}
    np.savez_compressed(out / "representative_trajectory.npz", **arrays)


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scenarios = default_scenarios()
    summaries, representatives = _write_results(out, scenarios, args.seeds, args.experiment_code_id)
    if args.render_scenario not in representatives:
        raise ValueError(f"unknown render scenario: {args.render_scenario}")
    from ausc.perception.tracking.occlusion_experiment import simulate_trial

    selected = next(item for item in scenarios if item.name == args.render_scenario)
    trial = simulate_trial(selected, args.render_seed)
    _write_representative_npz(trial, out)
    if not args.no_render:
        _render_isaac_video(trial, out, args.render_max_frames, args.rt_subframes)
    print(json.dumps({"out": str(out), "summaries": summaries}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
