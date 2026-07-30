"""MuJoCo contact smoke test for the ThinkLabs passive mount.

The analytic simulator is the design screen. This script adds a local physics
engine check using MuJoCo's contact solver: a guided contact disk is driven by
the selected passive spring/damper into flat, tilted, and rib-like tissue
surfaces. It is still lower fidelity than the repo's pending Isaac Lab scene,
but it provides engine-generated contact forces on macOS.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

tmp_root = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(tmp_root / "matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
GEN_DIR = ROOT / "auscultation" / "mechanical_sensor" / "generated"
SIM_DIR = GEN_DIR / "simulation"
ACCEPT_DIR = GEN_DIR / "acceptance"
OUT_DIR = GEN_DIR / "mujoco"
MUJOCO_SWEEP_SAMPLES = 96
MUJOCO_SWEEP_SEED = 20260609


@dataclass(frozen=True)
class Scenario:
    name: str
    tissue: str
    command_depth_mm: float
    approach_speed_mm_s: float
    tilt_deg: float
    rib_bump: bool
    solref_timeconst: float
    solimp_midpoint: float
    control_mode: str = "compression_gated"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def quat_for_x_tilt(deg: float) -> tuple[float, float, float, float]:
    half = math.radians(deg) / 2.0
    return math.cos(half), math.sin(half), 0.0, 0.0


def build_mjcf_xml(scenario: Scenario, radius_m: float, half_height_m: float, mass_kg: float) -> str:
    quat = quat_for_x_tilt(scenario.tilt_deg)
    rib_geom = ""
    if scenario.rib_bump:
        rib_geom = """
    <geom name="rib_bump" type="capsule" pos="0.000 0.000 0.0028"
          euler="0 1.57079632679 0" size="0.0055 0.050"
          rgba="0.55 0.34 0.30 1" contype="1" conaffinity="1"
          solref="0.006 1" solimp="0.90 0.98 0.002"/>"""
    return f"""
<mujoco model="thinklabs_passive_mount_contact">
  <compiler angle="radian"/>
  <option timestep="0.0005" gravity="0 0 0" integrator="implicitfast" cone="elliptic"/>
  <size njmax="100" nconmax="100"/>
  <default>
    <geom friction="0.8 0.02 0.002" condim="3"/>
  </default>
  <worldbody>
    <light pos="0 -0.4 0.35"/>
    <camera name="overview" pos="0.13 -0.20 0.10" xyaxes="0.84 0.54 0 -0.20 0.31 0.93"/>
    <geom name="tissue_plane" type="plane" pos="0 0 0"
          quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}"
          size="0.12 0.12 0.01" rgba="0.78 0.50 0.46 1"
          contype="1" conaffinity="1"
          solref="{scenario.solref_timeconst} 1" solimp="0.80 {scenario.solimp_midpoint} 0.002"/>
{rib_geom}
    <body name="carrier" pos="0 0 0">
      <joint name="contact_slide" type="slide" axis="0 0 1" limited="true" range="-0.050 0.025" damping="0"/>
      <geom name="thinklabs_contact_disk" type="cylinder"
            size="{radius_m:.6f} {half_height_m:.6f}" mass="{mass_kg:.6f}"
            rgba="0.05 0.05 0.05 1" contype="1" conaffinity="1"/>
      <geom name="floating_cradle_proxy" type="box" pos="0 0 0.022"
            size="0.055 0.042 0.006" rgba="0.05 0.34 0.75 0.45"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
""".strip()


def command_depth_at(time_s: float, start_mm: float, target_mm: float, speed_mm_s: float) -> tuple[float, float]:
    q = min(start_mm + speed_mm_s * time_s, target_mm)
    qdot = speed_mm_s if q < target_mm else 0.0
    return q, qdot


def contact_force_n(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    total = 0.0
    for idx in range(data.ncon):
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, idx, force)
        total += max(0.0, float(force[0]))
    return total


def run_scenario(
    scenario: Scenario,
    meta: dict,
    gates: dict,
    duration_s: float = 3.5,
) -> tuple[list[dict], dict]:
    params = meta["params"]
    mechanics = meta["recommended_mechanics"]
    radius_m = 0.026
    half_height_m = 0.00125
    mass_kg = 0.32
    spring_n_m = float(mechanics["spring_rate_n_per_mm"]) * 1000.0
    damping_n_s_m = float(mechanics["linear_damping_n_s_per_m"])
    travel_m = float(params["travel"]) / 1000.0
    xml = build_mjcf_xml(scenario, radius_m, half_height_m, mass_kg)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    slide_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "contact_slide")
    qpos_addr = model.jnt_qposadr[slide_id]
    qvel_addr = model.jnt_dofadr[slide_id]
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "carrier")

    rib_top_m = 0.0028 + 0.0055 if scenario.rib_bump else 0.0
    first_contact_z = half_height_m + rib_top_m
    start_depth_mm = -3.0
    data.qpos[qpos_addr] = first_contact_z - start_depth_mm / 1000.0
    mujoco.mj_forward(model, data)

    rows: list[dict] = []
    n_steps = int(duration_s / model.opt.timestep)
    command_mm = start_depth_mm
    command_speed_mm_s = scenario.approach_speed_mm_s
    hold_reason = "not_held"
    for step in range(n_steps):
        time_s = step * model.opt.timestep
        z = float(data.qpos[qpos_addr])
        zdot = float(data.qvel[qvel_addr])
        depth_m = first_contact_z - z
        depth_dot_m_s = -zdot
        prev_force = contact_force_n(model, data)
        if scenario.control_mode == "open_loop":
            command_mm, command_speed_mm_s = command_depth_at(
                time_s,
                start_depth_mm,
                scenario.command_depth_mm,
                scenario.approach_speed_mm_s,
            )
        else:
            current_compression_mm = max(0.0, command_mm - depth_m * 1000.0)
            if current_compression_mm >= gates["target_spring_compression_mm"]:
                command_speed_mm_s = 0.0
                hold_reason = "target_compression"
            elif prev_force >= gates["target_contact_force_n"] + 0.25:
                command_speed_mm_s = 0.0
                hold_reason = "contact_force_guard"
            elif command_mm >= scenario.command_depth_mm:
                command_speed_mm_s = 0.0
                hold_reason = "command_cap"
            else:
                command_speed_mm_s = scenario.approach_speed_mm_s
                command_mm = min(
                    scenario.command_depth_mm,
                    command_mm + command_speed_mm_s * model.opt.timestep,
                )
        command_m = command_mm / 1000.0
        command_speed_m_s = command_speed_mm_s / 1000.0
        raw_compression = command_m - depth_m
        compression = min(max(raw_compression, 0.0), travel_m)
        stop_over = max(0.0, raw_compression - travel_m)
        spring_force = spring_n_m * compression + damping_n_s_m * (command_speed_m_s - depth_dot_m_s)
        spring_force += 6000.0 * stop_over
        spring_force = max(0.0, spring_force)
        data.xfrc_applied[:, :] = 0.0
        data.xfrc_applied[body_id, 2] = -spring_force
        mujoco.mj_step(model, data)
        normal_force = contact_force_n(model, data)
        rows.append(
            {
                "scenario": scenario.name,
                "time_s": round(time_s, 5),
                "robot_command_depth_mm": round(command_mm, 4),
                "contact_depth_mm": round(depth_m * 1000.0, 4),
                "spring_compression_mm": round(compression * 1000.0, 4),
                "spring_force_n": round(spring_force, 4),
                "mujoco_contact_force_n": round(normal_force, 4),
                "num_contacts": int(data.ncon),
                "stop_over_mm": round(stop_over * 1000.0, 4),
                "control_mode": scenario.control_mode,
                "hold_reason": hold_reason,
            }
        )

    hold = rows[int(len(rows) * 0.82) :]
    forces = np.asarray([row["mujoco_contact_force_n"] for row in rows], dtype=float)
    hold_forces = np.asarray([row["mujoco_contact_force_n"] for row in hold], dtype=float)
    compression = np.asarray([row["spring_compression_mm"] for row in rows], dtype=float)
    contacts = np.asarray([row["num_contacts"] for row in rows], dtype=float)
    hold_contacts = np.asarray([row["num_contacts"] for row in hold], dtype=float)
    summary = {
        "scenario": scenario.name,
        "tissue": scenario.tissue,
        "tilt_deg": scenario.tilt_deg,
        "rib_bump": scenario.rib_bump,
        "approach_speed_mm_s": scenario.approach_speed_mm_s,
        "control_mode": scenario.control_mode,
        "hold_reason": hold[-1]["hold_reason"],
        "command_depth_mm": scenario.command_depth_mm,
        "final_robot_command_depth_mm": round(float(rows[-1]["robot_command_depth_mm"]), 3),
        "steady_contact_force_n": round(float(np.mean(hold_forces)), 3),
        "peak_contact_force_n": round(float(np.max(forces)), 3),
        "p95_contact_force_n": round(float(np.percentile(forces, 95)), 3),
        "max_spring_compression_mm": round(float(np.max(compression)), 3),
        "travel_used_pct": round(float(np.max(compression) / gates["travel_mm"] * 100.0), 1),
        "contact_presence_pct": round(float(100.0 * np.mean(contacts > 0)), 1),
        "hold_contact_presence_pct": round(float(100.0 * np.mean(hold_contacts > 0)), 1),
        "force_error_n": round(abs(float(np.mean(hold_forces)) - gates["target_contact_force_n"]), 3),
        "unsafe_force": bool(np.max(forces) > gates["max_safe_force_n"]),
        "travel_gate_violation": bool(np.max(compression) > gates["max_recommended_travel_mm"]),
        "stop_hit": bool(any(row["stop_over_mm"] > 0.0 for row in rows)),
    }
    return rows, summary


def plot_mujoco_profiles(rows: list[dict], summaries: list[dict]) -> None:
    by_scenario: dict[str, list[dict]] = {}
    for row in rows:
        by_scenario.setdefault(row["scenario"], []).append(row)

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.0), dpi=170, sharex=True)
    for summary in summaries:
        scenario = summary["scenario"]
        subset = by_scenario[scenario]
        time = np.asarray([row["time_s"] for row in subset], dtype=float)
        force = np.asarray([row["mujoco_contact_force_n"] for row in subset], dtype=float)
        compression = np.asarray([row["spring_compression_mm"] for row in subset], dtype=float)
        axes[0].plot(time, force, label=scenario, linewidth=1.7)
        axes[1].plot(time, compression, label=scenario, linewidth=1.7)
    axes[0].axhline(4.0, color="black", linestyle="--", linewidth=1.0, label="target 4 N")
    axes[0].axhline(8.0, color="#b83f37", linestyle=":", linewidth=1.2, label="abort 8 N")
    axes[1].axhline(20.4, color="#b83f37", linestyle=":", linewidth=1.2, label="85% travel gate")
    axes[0].set_ylabel("MuJoCo contact force (N)")
    axes[1].set_ylabel("spring compression (mm)")
    axes[1].set_xlabel("time (s)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("MuJoCo passive mount contact smoke test")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mujoco_contact_profiles.png")
    plt.close(fig)


def plot_mujoco_summary(summaries: list[dict]) -> None:
    labels = [row["scenario"] for row in summaries]
    peak = [row["peak_contact_force_n"] for row in summaries]
    steady = [row["steady_contact_force_n"] for row in summaries]
    travel = [row["travel_used_pct"] for row in summaries]
    contact = [row["hold_contact_presence_pct"] for row in summaries]
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.7), dpi=170)
    axes[0].barh(labels, steady, color="#3f6fa7")
    axes[0].axvline(4.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_title("steady force")
    axes[0].set_xlabel("N")
    axes[1].barh(labels, peak, color="#648f54")
    axes[1].axvline(8.0, color="#b83f37", linestyle=":", linewidth=1.2)
    axes[1].set_title("peak force")
    axes[1].set_xlabel("N")
    axes[2].barh(labels, travel, color="#b2794c")
    axes[2].axvline(85.0, color="#b83f37", linestyle=":", linewidth=1.2)
    axes[2].set_title("travel used")
    axes[2].set_xlabel("%")
    axes[3].barh(labels, contact, color="#7c659f")
    axes[3].set_title("hold contact presence")
    axes[3].set_xlabel("%")
    for ax in axes:
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mujoco_contact_summary.png")
    plt.close(fig)


def run_random_sweep(
    meta: dict,
    gates: dict,
    command_by_tissue: dict[str, float],
    sample_count: int = MUJOCO_SWEEP_SAMPLES,
) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(MUJOCO_SWEEP_SEED)
    tissue_params = {
        "soft_tissue": (0.022, 0.93),
        "medium_tissue": (0.014, 0.95),
        "firm_tissue": (0.010, 0.96),
        "bony_prominence": (0.008, 0.97),
    }
    tissues = list(tissue_params)
    rows: list[dict] = []
    for sample in range(sample_count):
        tissue = str(rng.choice(tissues))
        base_solref, base_solimp = tissue_params[tissue]
        tilt = float(np.clip(abs(rng.normal(3.5, 2.6)), 0.0, 10.0))
        speed = float(np.clip(rng.normal(35.0, 18.0), 12.0, 85.0))
        rib_bump = bool(rng.random() < (0.70 if tissue == "bony_prominence" else 0.18))
        command_depth = command_by_tissue[tissue] + float(np.clip(rng.normal(0.0, 1.2), -2.5, 2.5))
        scenario = Scenario(
            name=f"sweep_{sample:03d}_{tissue}",
            tissue=tissue,
            command_depth_mm=command_depth,
            approach_speed_mm_s=speed,
            tilt_deg=tilt,
            rib_bump=rib_bump,
            solref_timeconst=float(np.clip(base_solref * rng.uniform(0.75, 1.35), 0.006, 0.030)),
            solimp_midpoint=float(np.clip(base_solimp + rng.normal(0.0, 0.012), 0.90, 0.985)),
        )
        _, summary = run_scenario(scenario, meta, gates, duration_s=2.6)
        summary["sample"] = sample
        rows.append(summary)

    peak = np.asarray([row["peak_contact_force_n"] for row in rows], dtype=float)
    steady = np.asarray([row["steady_contact_force_n"] for row in rows], dtype=float)
    travel = np.asarray([row["travel_used_pct"] for row in rows], dtype=float)
    contact = np.asarray([row["contact_presence_pct"] for row in rows], dtype=float)
    hold_contact = np.asarray([row["hold_contact_presence_pct"] for row in rows], dtype=float)
    force_error = np.abs(steady - gates["target_contact_force_n"])
    aggregate = {
        "sample_count": sample_count,
        "seed": MUJOCO_SWEEP_SEED,
        "mean_steady_force_n": round(float(np.mean(steady)), 3),
        "mean_force_error_n": round(float(np.mean(force_error)), 3),
        "p95_peak_contact_force_n": round(float(np.percentile(peak, 95)), 3),
        "max_peak_contact_force_n": round(float(np.max(peak)), 3),
        "p95_travel_used_pct": round(float(np.percentile(travel, 95)), 1),
        "max_travel_used_pct": round(float(np.max(travel)), 1),
        "min_contact_presence_pct": round(float(np.min(contact)), 1),
        "min_hold_contact_presence_pct": round(float(np.min(hold_contact)), 1),
        "unsafe_force_rate": round(float(np.mean([row["unsafe_force"] for row in rows])), 4),
        "travel_gate_violation_rate": round(float(np.mean([row["travel_gate_violation"] for row in rows])), 4),
        "contact_loss_rate": round(float(np.mean([row["hold_contact_presence_pct"] < 99.0 for row in rows])), 4),
        "stop_hit_rate": round(float(np.mean([row["stop_hit"] for row in rows])), 4),
    }
    return rows, aggregate


def plot_mujoco_sweep(sweep_rows: list[dict], aggregate: dict) -> None:
    peak = np.asarray([row["peak_contact_force_n"] for row in sweep_rows], dtype=float)
    steady = np.asarray([row["steady_contact_force_n"] for row in sweep_rows], dtype=float)
    travel = np.asarray([row["travel_used_pct"] for row in sweep_rows], dtype=float)
    tilt = np.asarray([row["tilt_deg"] for row in sweep_rows], dtype=float)
    speed = np.asarray([row["approach_speed_mm_s"] for row in sweep_rows], dtype=float)
    force_error = np.abs(steady - 4.0)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), dpi=170)
    axes[0, 0].hist(peak, bins=18, color="#648f54", edgecolor="white")
    axes[0, 0].axvline(8.0, color="#b83f37", linestyle=":", linewidth=1.2)
    axes[0, 0].set_title(f"Peak force, p95={aggregate['p95_peak_contact_force_n']} N")
    axes[0, 0].set_xlabel("N")
    axes[0, 1].hist(force_error, bins=18, color="#3f6fa7", edgecolor="white")
    axes[0, 1].set_title(f"Steady force error, mean={aggregate['mean_force_error_n']} N")
    axes[0, 1].set_xlabel("N")
    axes[1, 0].hist(travel, bins=18, color="#b2794c", edgecolor="white")
    axes[1, 0].axvline(85.0, color="#b83f37", linestyle=":", linewidth=1.2)
    axes[1, 0].set_title(f"Travel use, p95={aggregate['p95_travel_used_pct']}%")
    axes[1, 0].set_xlabel("%")
    scatter = axes[1, 1].scatter(tilt, peak, c=speed, s=22, cmap="viridis", alpha=0.85)
    axes[1, 1].axhline(8.0, color="#b83f37", linestyle=":", linewidth=1.2)
    axes[1, 1].set_title("Peak force vs tilt, colored by speed")
    axes[1, 1].set_xlabel("tilt (deg)")
    axes[1, 1].set_ylabel("peak force (N)")
    cbar = fig.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label("mm/s")
    for ax in axes.flat:
        ax.grid(True, alpha=0.25)
    fig.suptitle("MuJoCo randomized compression-gated robustness sweep")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mujoco_random_sweep.png")
    plt.close(fig)


def report_lines(meta: dict, gates: dict, summaries: list[dict], sweep_aggregate: dict | None = None) -> list[str]:
    worst_peak = max(row["peak_contact_force_n"] for row in summaries)
    worst_travel = max(row["travel_used_pct"] for row in summaries)
    any_unsafe = any(row["unsafe_force"] for row in summaries)
    any_travel = any(row["travel_gate_violation"] for row in summaries)
    lines = [
        "# MuJoCo Contact Smoke Test",
        "",
        "## Scope",
        "",
        "This is a local MuJoCo physics-engine smoke test for the selected v6 passive mount. It applies the simulated spring/damper force to a guided ThinkLabs contact disk and records MuJoCo contact solver forces against flat, tilted, and rib-like tissue proxy surfaces.",
        "",
        "It is stronger evidence than the analytic-only screen because contact forces are produced by MuJoCo, but it is still not the pending Vast.ai/Isaac Lab high-fidelity validation.",
        "",
        "## Candidate",
        "",
        f"- Variant: `{meta['recommended_mechanics']['simulation_variant']}`",
        f"- Contact-face projection: `{meta['params']['contact_face_projection']} mm`",
        f"- Spring: `{gates['spring_rate_n_per_mm']} N/mm`",
        f"- Target force: `{gates['target_contact_force_n']} N`",
        f"- Abort force: `{gates['max_safe_force_n']} N`",
        "",
        "## Scenario Results",
        "",
        "| Scenario | control | final command (mm) | speed (mm/s) | tilt (deg) | steady force (N) | peak force (N) | travel (%) | hold contact (%) | unsafe | travel gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
        *[
            f"| `{row['scenario']}` | `{row['control_mode']}` | {row['final_robot_command_depth_mm']} | {row['approach_speed_mm_s']} | {row['tilt_deg']} | {row['steady_contact_force_n']} | {row['peak_contact_force_n']} | {row['travel_used_pct']} | {row['hold_contact_presence_pct']} | {row['unsafe_force']} | {row['travel_gate_violation']} |"
            for row in summaries
        ],
        "",
        "## Gate Summary",
        "",
        f"- Worst peak MuJoCo contact force: `{worst_peak} N`",
        f"- Worst travel use: `{worst_travel}%`",
        f"- Any unsafe-force gate violation: `{any_unsafe}`",
        f"- Any travel gate violation: `{any_travel}`",
        "",
    ]
    if sweep_aggregate is not None:
        lines.extend(
            [
                "## Randomized MuJoCo Sweep",
                "",
                f"- Samples: `{sweep_aggregate['sample_count']}`",
                f"- Seed: `{sweep_aggregate['seed']}`",
                f"- Mean steady force: `{sweep_aggregate['mean_steady_force_n']} N`",
                f"- Mean steady-force error: `{sweep_aggregate['mean_force_error_n']} N`",
                f"- P95 / max peak contact force: `{sweep_aggregate['p95_peak_contact_force_n']} N` / `{sweep_aggregate['max_peak_contact_force_n']} N`",
                f"- P95 / max travel use: `{sweep_aggregate['p95_travel_used_pct']}%` / `{sweep_aggregate['max_travel_used_pct']}%`",
                f"- Unsafe-force / travel-gate / contact-loss rates: `{100.0 * sweep_aggregate['unsafe_force_rate']:.1f}%` / `{100.0 * sweep_aggregate['travel_gate_violation_rate']:.1f}%` / `{100.0 * sweep_aggregate['contact_loss_rate']:.1f}%`",
                "",
            ]
        )
    lines.extend(
        [
        "## Generated Artifacts",
        "",
        "- `mujoco_contact_trace.csv`",
        "- `mujoco_contact_summary.csv`",
        "- `mujoco_contact_results.json`",
        "- `mujoco_contact_profiles.png`",
        "- `mujoco_contact_summary.png`",
        "- `mujoco_random_sweep.csv`",
        "- `mujoco_random_sweep_summary.json`",
        "- `mujoco_random_sweep.png`",
        "- `mujoco_scene_*.xml`",
        "",
        "## Limitation",
        "",
        "The body is constrained to the guide axis and the tissue is represented by MuJoCo contact parameters, not patient-specific soft tissue. The tested controller is a simple compression-gated controller that stops when spring compression reaches the 4 N calibration target. Final validation still requires importing the CAD proxy into the repo's Vast.ai/Isaac Lab workflow with rigid-body/contact sensors.",
        "",
        ]
    )
    return lines


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = json.loads((GEN_DIR / "design_metadata.json").read_text(encoding="utf-8"))
    gates = json.loads((ACCEPT_DIR / "acceptance_gates.json").read_text(encoding="utf-8"))

    command_by_tissue = {}
    with (ACCEPT_DIR / "robot_command_envelope.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            command_by_tissue[row["tissue"]] = float(row["robot_overdrive_mm"])

    scenarios = [
        Scenario("medium_flat_30mms", "medium_tissue", command_by_tissue["medium_tissue"], 30.0, 0.0, False, 0.014, 0.95),
        Scenario("medium_flat_60mms", "medium_tissue", command_by_tissue["medium_tissue"], 60.0, 0.0, False, 0.014, 0.95),
        Scenario("soft_flat_30mms", "soft_tissue", command_by_tissue["soft_tissue"], 30.0, 0.0, False, 0.022, 0.93),
        Scenario("firm_tilt_6deg", "firm_tissue", command_by_tissue["firm_tissue"], 30.0, 6.0, False, 0.010, 0.96),
        Scenario("bony_tilt_8deg_rib", "bony_prominence", command_by_tissue["bony_prominence"], 30.0, 8.0, True, 0.008, 0.97),
    ]

    all_rows: list[dict] = []
    summaries: list[dict] = []
    scene_files: list[str] = []
    for scenario in scenarios:
        xml = build_mjcf_xml(scenario, 0.026, 0.00125, 0.32)
        xml_path = OUT_DIR / f"mujoco_scene_{scenario.name}.xml"
        xml_path.write_text(xml, encoding="utf-8")
        scene_files.append(str(xml_path.relative_to(ROOT)))
        rows, summary = run_scenario(scenario, meta, gates)
        all_rows.extend(rows)
        summaries.append(summary)

    sweep_rows, sweep_aggregate = run_random_sweep(meta, gates, command_by_tissue)

    write_csv(OUT_DIR / "mujoco_contact_trace.csv", all_rows)
    write_csv(OUT_DIR / "mujoco_contact_summary.csv", summaries)
    write_csv(OUT_DIR / "mujoco_random_sweep.csv", sweep_rows)
    (OUT_DIR / "mujoco_random_sweep_summary.json").write_text(json.dumps(sweep_aggregate, indent=2), encoding="utf-8")
    plot_mujoco_profiles(all_rows, summaries)
    plot_mujoco_summary(summaries)
    plot_mujoco_sweep(sweep_rows, sweep_aggregate)
    payload = {
        "variant": meta["recommended_mechanics"]["simulation_variant"],
        "gates": gates,
        "scenarios": [scenario.__dict__ for scenario in scenarios],
        "summaries": summaries,
        "random_sweep": {
            "aggregate": sweep_aggregate,
            "summary_csv": str((OUT_DIR / "mujoco_random_sweep.csv").relative_to(ROOT)),
        },
        "scene_files": scene_files,
    }
    (OUT_DIR / "mujoco_contact_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUT_DIR / "MUJOCO_REPORT.md").write_text(
        "\n".join(report_lines(meta, gates, summaries, sweep_aggregate)),
        encoding="utf-8",
    )
    print(json.dumps({"mujoco_dir": str(OUT_DIR), "summaries": summaries, "random_sweep": sweep_aggregate}, indent=2))


if __name__ == "__main__":
    main()
