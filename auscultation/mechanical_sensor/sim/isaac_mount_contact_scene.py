"""Isaac Sim handoff scene for the ThinkLabs passive mount contact test.

Run this on the repository's cloud Isaac Sim / Isaac Lab machine, not on macOS:

    /isaac-sim/python.sh auscultation/mechanical_sensor/sim/isaac_mount_contact_scene.py \
        --metadata auscultation/mechanical_sensor/generated/design_metadata.json \
        --out auscultation/mechanical_sensor/generated/isaac_contact

This script builds a simplified visual/kinematic scene using the final CAD
parameters and logs the same passive spring contact response in Isaac time.
It is a handoff bridge for high-fidelity testing; it is not a substitute for a
future rigid-body/contact-sensor Isaac validation pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="auscultation/mechanical_sensor/generated/design_metadata.json")
    parser.add_argument("--out", default="auscultation/mechanical_sensor/generated/isaac_contact")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=3.5)
    parser.add_argument("--dt", type=float, default=1.0 / 120.0)
    parser.add_argument("--approach-speed-mm-s", type=float, default=30.0)
    parser.add_argument("--tissue-stiffness-n-per-mm", type=float, default=0.8)
    parser.add_argument("--tissue-damping-n-s-per-m", type=float, default=12.0)
    parser.add_argument("--render-preview", action="store_true")
    parser.add_argument("--skip-motion-video", action="store_true")
    parser.add_argument("--motion-video-fps", type=int, default=30)
    parser.add_argument("--motion-video-frames", type=int, default=150)
    parser.add_argument(
        "--enable-viewport-updates",
        action="store_true",
        help="Keep viewport rendering active in headless mode. Disabled by default for cloud physics runs.",
    )
    return parser.parse_args()


def target_command_depth_mm(meta: dict) -> float:
    mech = meta["recommended_mechanics"]
    spring = float(mech["spring_rate_n_per_mm"])
    target_force = float(mech["target_contact_force_n"])
    medium_tissue = 0.8
    return target_force / spring + target_force / medium_tissue


def simulate_step_response(
    meta: dict,
    duration: float,
    dt: float,
    approach_speed_mm_s: float,
    tissue_stiffness_n_per_mm: float,
    tissue_damping_n_s_per_m: float,
) -> list[dict[str, float]]:
    mech = meta["recommended_mechanics"]
    params = meta["params"]
    k_s = float(mech["spring_rate_n_per_mm"]) * 1000.0
    c_s = float(mech["linear_damping_n_s_per_m"])
    k_t = tissue_stiffness_n_per_mm * 1000.0
    c_t = tissue_damping_n_s_per_m
    mass = 0.32
    travel_m = float(params["travel"]) / 1000.0
    command_target_m = target_command_depth_mm(meta) / 1000.0
    command_start_m = -0.003
    speed_m_s = approach_speed_mm_s / 1000.0

    rows: list[dict[str, float]] = []
    x = -0.003
    v = 0.0
    n = int(duration / dt) + 1
    for i in range(n):
        time_s = i * dt
        q = min(command_start_m + speed_m_s * time_s, command_target_m)
        qdot = speed_m_s if q < command_target_m else 0.0
        raw_compression = q - x
        compression = max(0.0, min(raw_compression, travel_m))
        stop_over = max(0.0, raw_compression - travel_m)
        spring_force = max(0.0, k_s * compression + c_s * (qdot - v) + 6000.0 * stop_over)
        tissue_force = max(0.0, k_t * x + c_t * v) if x > 0.0 else 0.0
        acc = (spring_force - tissue_force) / mass
        v += acc * dt
        x += v * dt
        if q - x > travel_m:
            x = q - travel_m
            v = min(v, qdot)
        rows.append(
            {
                "time_s": time_s,
                "robot_command_mm": q * 1000.0,
                "contact_face_z_mm": x * 1000.0,
                "spring_compression_mm": max(0.0, q - x) * 1000.0,
                "contact_force_n": tissue_force,
            }
        )
    return rows


def write_contact_plot(rows: list[dict[str, float]], meta: dict, out: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return None

    mech = meta["recommended_mechanics"]
    target_force = float(mech["target_contact_force_n"])
    max_safe_force = float(mech["max_screened_contact_force_n"])
    max_travel = float(meta["params"]["travel"])

    t = [r["time_s"] for r in rows]
    force = [r["contact_force_n"] for r in rows]
    compression = [r["spring_compression_mm"] for r in rows]
    command = [r["robot_command_mm"] for r in rows]
    face = [r["contact_face_z_mm"] for r in rows]

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.0), sharex=True)
    fig.suptitle("Isaac Contact Handoff Result", fontsize=14)

    axes[0].plot(t, force, color="#1769aa", linewidth=2.0, label="contact force")
    axes[0].axhline(target_force, color="#2e7d32", linestyle="--", linewidth=1.3, label=f"target {target_force:.1f} N")
    axes[0].axhline(max_safe_force, color="#c62828", linestyle=":", linewidth=1.3, label=f"screened max {max_safe_force:.1f} N")
    axes[0].set_ylabel("Force (N)")
    axes[0].legend(loc="lower right", fontsize=8)

    axes[1].plot(t, compression, color="#6a1b9a", linewidth=2.0, label="spring compression")
    axes[1].axhline(max_travel, color="#c62828", linestyle=":", linewidth=1.3, label=f"travel limit {max_travel:.1f} mm")
    axes[1].set_ylabel("Compression (mm)")
    axes[1].legend(loc="lower right", fontsize=8)

    axes[2].plot(t, command, color="#ef6c00", linewidth=1.8, label="robot command")
    axes[2].plot(t, face, color="#455a64", linewidth=1.8, label="contact face")
    axes[2].set_ylabel("Position (mm)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(loc="lower right", fontsize=8)

    for ax in axes:
        ax.grid(True, color="#d7d7d7", linewidth=0.7, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plot_path = out / "isaac_contact_response.png"
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(plot_path, dpi=170)
    plt.close(fig)
    return plot_path


def write_motion_video(
    rows: list[dict[str, float]],
    meta: dict,
    out: Path,
    fps: int = 30,
    max_frames: int = 150,
) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.animation as animation
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:  # noqa: BLE001
        return None

    if not rows:
        return None

    step = max(1, math.ceil(len(rows) / max_frames))
    frames = rows[::step]
    if frames[-1] is not rows[-1]:
        frames.append(rows[-1])

    target_force = float(meta["recommended_mechanics"]["target_contact_force_n"])
    max_force = float(meta["recommended_mechanics"]["max_screened_contact_force_n"])
    travel_limit = float(meta["params"]["travel"])
    times = [row["time_s"] for row in rows]
    forces = [row["contact_force_n"] for row in rows]
    compressions = [row["spring_compression_mm"] for row in rows]

    fig = plt.figure(figsize=(12.2, 7.2), dpi=130)
    gs = fig.add_gridspec(2, 2, width_ratios=(1.0, 1.35), height_ratios=(1.0, 1.0), wspace=0.24, hspace=0.24)
    ax_motion = fig.add_subplot(gs[:, 0])
    ax_force = fig.add_subplot(gs[0, 1])
    ax_comp = fig.add_subplot(gs[1, 1])

    ax_motion.set_title("Passive mount motion")
    ax_motion.set_xlim(-55, 55)
    ax_motion.set_ylim(16, -42)
    ax_motion.set_aspect("equal", adjustable="box")
    ax_motion.set_xlabel("Mount width (mm)")
    ax_motion.set_ylabel("Depth toward patient (mm)")
    ax_motion.axhline(0, color="#8d3f34", linewidth=2.0)
    ax_motion.add_patch(patches.Rectangle((-55, 0), 110, 11.5, facecolor="#efc7bf", edgecolor="none", alpha=0.75))
    ax_motion.text(-52, 4.5, "soft tissue", color="#6d352e", fontsize=9, va="center")

    robot_plate = patches.Rectangle((-38, -28), 76, 4, facecolor="#c7a96b", edgecolor="#5f4a1f", linewidth=1.0)
    cradle = patches.FancyBboxPatch(
        (-31, -14),
        62,
        12,
        boxstyle="round,pad=0.4,rounding_size=2",
        facecolor="#1f77b4",
        edgecolor="#0d3f63",
        linewidth=1.0,
    )
    thinklabs = patches.Circle((0, -19), 15.5, facecolor="#2c2c2c", edgecolor="#070707", linewidth=1.0)
    contact_disk = patches.Rectangle((-23, -1.2), 46, 2.4, facecolor="#0b0b0b", edgecolor="#000000", linewidth=0.8)
    ax_motion.add_patch(robot_plate)
    ax_motion.add_patch(cradle)
    ax_motion.add_patch(thinklabs)
    ax_motion.add_patch(contact_disk)
    guide_left, = ax_motion.plot([], [], color="#6c6c6c", linewidth=3)
    guide_right, = ax_motion.plot([], [], color="#6c6c6c", linewidth=3)
    spring_line, = ax_motion.plot([], [], color="#2e7d32", linewidth=2.2)
    status_text = ax_motion.text(-53, -39.0, "", fontsize=10, family="monospace", va="top")

    ax_force.set_title("Contact force")
    ax_force.set_xlim(times[0], times[-1])
    ax_force.set_ylim(-0.2, max(max_force * 1.08, max(forces) * 1.2, 5.0))
    ax_force.axhline(target_force, color="#2e7d32", linestyle="--", linewidth=1.2, label=f"target {target_force:.1f} N")
    ax_force.axhline(max_force, color="#c62828", linestyle=":", linewidth=1.2, label=f"max {max_force:.1f} N")
    force_line, = ax_force.plot([], [], color="#1769aa", linewidth=2.0)
    force_dot, = ax_force.plot([], [], "o", color="#1769aa", markersize=5)
    ax_force.set_ylabel("N")
    ax_force.grid(True, color="#d7d7d7", linewidth=0.7, alpha=0.75)
    ax_force.legend(loc="lower right", fontsize=8)

    ax_comp.set_title("Passive spring compression")
    ax_comp.set_xlim(times[0], times[-1])
    ax_comp.set_ylim(-0.5, max(travel_limit * 1.08, max(compressions) * 1.25, 8.0))
    ax_comp.axhline(travel_limit, color="#c62828", linestyle=":", linewidth=1.2, label=f"travel {travel_limit:.1f} mm")
    comp_line, = ax_comp.plot([], [], color="#6a1b9a", linewidth=2.0)
    comp_dot, = ax_comp.plot([], [], "o", color="#6a1b9a", markersize=5)
    ax_comp.set_xlabel("Time (s)")
    ax_comp.set_ylabel("mm")
    ax_comp.grid(True, color="#d7d7d7", linewidth=0.7, alpha=0.75)
    ax_comp.legend(loc="lower right", fontsize=8)

    def spring_points(y0: float, y1: float) -> tuple[list[float], list[float]]:
        if y1 < y0:
            y0, y1 = y1, y0
        coil_y = np.linspace(y0, y1, 80)
        span = max(y1 - y0, 1e-6)
        coil_x = 6.0 * np.sin(np.linspace(0, 9.5 * math.pi, 80)) * np.sin(np.pi * (coil_y - y0) / span)
        return coil_x.tolist(), coil_y.tolist()

    def update(frame: dict[str, float]):
        t = frame["time_s"]
        command = frame["robot_command_mm"]
        face = frame["contact_face_z_mm"]
        compression = frame["spring_compression_mm"]
        force = frame["contact_force_n"]

        contact_y = face
        spring_gap = max(7.5, 21.0 - 0.75 * compression)
        cradle_y = contact_y - 12.0
        plate_y = cradle_y - spring_gap - 4.0
        body_center_y = contact_y - 18.0

        robot_plate.set_xy((-38, plate_y))
        cradle.set_y(cradle_y)
        thinklabs.center = (0.0, body_center_y)
        contact_disk.set_xy((-23, contact_y - 1.2))
        guide_left.set_data([-36, -36], [plate_y + 3.5, cradle_y + 8.5])
        guide_right.set_data([36, 36], [plate_y + 3.5, cradle_y + 8.5])
        sx, sy = spring_points(plate_y + 4.0, cradle_y)
        spring_line.set_data(sx, sy)
        status_text.set_text(
            f"t={t:4.2f}s\nforce={force:4.2f} N\ncompression={compression:4.2f} mm\ncommand={command:4.1f} mm"
        )

        idx = max(0, min(len(rows) - 1, int(round(t / (rows[1]["time_s"] - rows[0]["time_s"])))))
        force_line.set_data(times[: idx + 1], forces[: idx + 1])
        comp_line.set_data(times[: idx + 1], compressions[: idx + 1])
        force_dot.set_data([t], [force])
        comp_dot.set_data([t], [compression])
        return (
            robot_plate,
            cradle,
            thinklabs,
            contact_disk,
            guide_left,
            guide_right,
            spring_line,
            status_text,
            force_line,
            force_dot,
            comp_line,
            comp_dot,
        )

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / max(fps, 1), blit=True)
    mp4_path = out / "isaac_contact_motion.mp4"
    gif_path = out / "isaac_contact_motion.gif"
    try:
        if animation.writers.is_available("ffmpeg"):
            ani.save(mp4_path, writer=animation.FFMpegWriter(fps=fps, bitrate=1800))
            plt.close(fig)
            return mp4_path
        if animation.writers.is_available("pillow"):
            ani.save(gif_path, writer=animation.PillowWriter(fps=min(fps, 20)))
            plt.close(fig)
            return gif_path
    except Exception:  # noqa: BLE001
        plt.close(fig)
        return None
    plt.close(fig)
    return None


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = Path(args.metadata)
    meta = json.loads(meta_path.read_text())
    rows = simulate_step_response(
        meta,
        args.duration,
        args.dt,
        args.approach_speed_mm_s,
        args.tissue_stiffness_n_per_mm,
        args.tissue_damping_n_s_per_m,
    )

    # Isaac imports are intentionally delayed so local py_compile works on macOS.
    from isaacsim import SimulationApp

    launch_config = {
        "headless": args.headless,
        "hide_ui": args.headless,
        "multi_gpu": False,
        "active_gpu": 0,
        "physics_gpu": 0,
        "anti_aliasing": 0,
        "samples_per_pixel_per_frame": 1,
        "width": 1280 if args.render_preview else 64,
        "height": 900 if args.render_preview else 64,
        "disable_viewport_updates": bool(args.headless and not args.render_preview and not args.enable_viewport_updates),
    }
    print(json.dumps({"isaac_launch_config": launch_config}, indent=2), flush=True)
    simulation_app = SimulationApp(launch_config)
    print("ISAAC_SIM_APP_READY", flush=True)

    import numpy as np
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, VisualCylinder

    try:
        import imageio.v2 as imageio
        import omni.replicator.core as rep
    except Exception:  # noqa: BLE001
        imageio = None
        rep = None

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    world.scene.add(
        FixedCuboid(
            prim_path="/World/TissuePad",
            name="tissue_pad",
            position=np.array([0.0, 0.0, -0.012]),
            scale=np.array([0.18, 0.18, 0.024]),
            color=np.array([0.83, 0.55, 0.50]),
        )
    )
    world.scene.add(
        FixedCuboid(
            prim_path="/World/RobotPlate",
            name="robot_plate",
            position=np.array([0.0, 0.0, 0.115]),
            scale=np.array([0.11, 0.07, 0.008]),
            color=np.array([0.82, 0.79, 0.68]),
        )
    )
    contact = world.scene.add(
        DynamicCuboid(
            prim_path="/World/ThinkLabsContactProxy",
            name="thinklabs_contact_proxy",
            position=np.array([0.0, 0.0, 0.03]),
            scale=np.array([0.052, 0.052, 0.006]),
            color=np.array([0.08, 0.08, 0.08]),
            mass=0.32,
        )
    )
    world.scene.add(
        DynamicCuboid(
            prim_path="/World/FloatingCradleProxy",
            name="floating_cradle_proxy",
            position=np.array([0.0, 0.0, 0.055]),
            scale=np.array([0.108, 0.085, 0.012]),
            color=np.array([0.05, 0.34, 0.75]),
            mass=0.12,
        )
    )
    for idx, x in enumerate([-0.042, 0.042]):
        VisualCylinder(
            prim_path=f"/World/GuideRod_{idx}",
            name=f"guide_rod_{idx}",
            position=np.array([x, 0.016, 0.055]),
            radius=0.003,
            height=0.09,
            color=np.array([0.55, 0.55, 0.58]),
        )
    VisualCylinder(
        prim_path="/World/SpringEnvelope",
        name="spring_envelope",
        position=np.array([0.0, 0.0, 0.055]),
        radius=0.008,
        height=0.042,
        color=np.array([0.2, 0.8, 0.25]),
    )

    camera_product = None
    rgb = None
    if args.render_preview and rep is not None:
        camera = rep.create.camera(position=(0.18, -0.22, 0.16), look_at=(0.0, 0.0, 0.03))
        camera_product = rep.create.render_product(camera, (1280, 900))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach(camera_product)

    world.reset()
    csv_path = out / "isaac_contact_trace.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            z_m = row["contact_face_z_mm"] / 1000.0 + 0.003
            contact.set_world_pose(np.array([0.0, 0.0, z_m]))
            world.step(render=args.render_preview)
            writer.writerow(row)

    if args.render_preview and rep is not None and imageio is not None and rgb is not None:
        for _ in range(6):
            rep.orchestrator.step(rt_subframes=8)
        data = rgb.get_data()
        if data is not None and getattr(data, "size", 0) > 0:
            imageio.imwrite(out / "isaac_contact_preview.png", np.asarray(data[..., :3], dtype=np.uint8))

    simulation_app.close()
    print("ISAAC_SIM_APP_CLOSED", flush=True)

    summary = {
        "metadata": str(meta_path),
        "output_csv": str(csv_path),
        "peak_force_n": max(r["contact_force_n"] for r in rows),
        "final_force_n": rows[-1]["contact_force_n"],
        "max_spring_compression_mm": max(r["spring_compression_mm"] for r in rows),
        "note": "This is an Isaac visualization/handoff scene driven by the local passive contact model; add contact sensors for final high-fidelity validation.",
    }
    plot_path = write_contact_plot(rows, meta, out)
    if plot_path is not None:
        summary["output_plot"] = str(plot_path)
    if not args.skip_motion_video:
        video_path = write_motion_video(rows, meta, out, args.motion_video_fps, args.motion_video_frames)
        if video_path is not None:
            summary["output_motion_video"] = str(video_path)
    (out / "isaac_contact_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
