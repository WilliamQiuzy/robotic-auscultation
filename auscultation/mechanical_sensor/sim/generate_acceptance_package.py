"""Generate calibration and acceptance artifacts for the ThinkLabs mount.

This script converts the selected simulation result into practical pre-hardware
checks: spring displacement calibration, robot overdrive envelope, and explicit
go/no-go gates for the printed v6 passive mount.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

tmp_root = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(tmp_root / "matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
GEN_DIR = ROOT / "auscultation" / "mechanical_sensor" / "generated"
SIM_DIR = GEN_DIR / "simulation"
OUT_DIR = GEN_DIR / "acceptance"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def force_displacement_rows(spring_n_per_mm: float, max_force_n: float) -> list[dict]:
    forces = np.arange(0.0, max_force_n + 0.25, 0.5)
    return [
        {
            "force_n": round(float(force), 3),
            "spring_compression_mm": round(float(force / spring_n_per_mm), 3),
            "sensor_delta_mm": round(float(force / spring_n_per_mm), 3),
        }
        for force in forces
    ]


def command_envelope_rows(spring_n_per_mm: float, target_force_n: float) -> list[dict]:
    tissue_cases = [
        ("soft_tissue", 0.35),
        ("medium_tissue", 0.80),
        ("firm_tissue", 1.60),
        ("bony_prominence", 2.50),
    ]
    spring_compression = target_force_n / spring_n_per_mm
    rows = []
    for name, tissue_k in tissue_cases:
        tissue_indent = target_force_n / tissue_k
        rows.append(
            {
                "tissue": name,
                "tissue_stiffness_n_per_mm": tissue_k,
                "target_force_n": target_force_n,
                "spring_compression_mm": round(spring_compression, 3),
                "estimated_tissue_indent_mm": round(tissue_indent, 3),
                "robot_overdrive_mm": round(spring_compression + tissue_indent, 3),
            }
        )
    return rows


def plot_calibration_curve(rows: list[dict], gates: dict) -> None:
    compression = np.asarray([float(row["spring_compression_mm"]) for row in rows])
    force = np.asarray([float(row["force_n"]) for row in rows])
    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=170)
    ax.plot(compression, force, color="#2f6fa3", linewidth=2.0)
    ax.scatter(compression, force, color="#2f6fa3", s=18)
    ax.axhline(gates["target_contact_force_n"], color="black", linestyle="--", linewidth=1.0, label="target")
    ax.axhline(gates["max_safe_force_n"], color="#b83f37", linestyle=":", linewidth=1.2, label="abort line")
    ax.axvline(gates["target_spring_compression_mm"], color="#5a8f4d", linestyle="--", linewidth=1.0)
    ax.axvline(gates["max_safe_spring_compression_mm"], color="#b83f37", linestyle=":", linewidth=1.2)
    ax.set_xlabel("spring/sensor compression (mm)")
    ax.set_ylabel("estimated contact force (N)")
    ax.set_title("v6 displacement-to-force calibration target")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "force_displacement_calibration.png")
    plt.close(fig)


def plot_command_envelope(rows: list[dict]) -> None:
    labels = [row["tissue"] for row in rows]
    spring = np.asarray([float(row["spring_compression_mm"]) for row in rows])
    indent = np.asarray([float(row["estimated_tissue_indent_mm"]) for row in rows])
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=170)
    ax.barh(y, spring, color="#3f6fa7", label="spring compression")
    ax.barh(y, indent, left=spring, color="#b2794c", label="estimated tissue indentation")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("robot overdrive from first contact (mm)")
    ax.set_title("Target 4 N command envelope by tissue stiffness")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "robot_command_envelope.png")
    plt.close(fig)


def report_lines(meta: dict, sim: dict, gates: dict, command_rows: list[dict]) -> list[str]:
    final = sim["final_recommendation"]
    combined = sim["combined_recommendation"][0]
    robust = sim["robustness_summary"][0]
    patch = sim["contact_patch_summary"][0]
    return [
        "# ThinkLabs Passive Mount Acceptance Package",
        "",
        "## Selected Candidate",
        "",
        f"- Final local recommendation: `{final}`",
        f"- CAD metadata variant: `{meta['recommended_mechanics']['simulation_variant']}`",
        f"- Contact-face projection: `{meta['params']['contact_face_projection']} mm`",
        f"- Spring target: `{gates['spring_rate_n_per_mm']} N/mm`",
        f"- Damping target: `{gates['linear_damping_n_s_per_m']} N s/m`",
        "",
        "## Simulation Evidence Used",
        "",
        f"- Combined score: `{combined['combined_score']}`",
        f"- Robustness P95/P99 peak force: `{robust['p95_peak_force_n']} N` / `{robust['p99_peak_force_n']} N`",
        f"- Robustness safety/travel/binding violations: `{100.0 * robust['safety_violation_rate']:.1f}%` / `{100.0 * robust['travel_violation_rate']:.1f}%` / `{100.0 * robust['binding_risk_rate']:.1f}%`",
        f"- 2D contact patch P95 pressure: `{patch['p95_peak_pressure_kpa']} kPa`",
        f"- 2D tilted clearance violation rate: `{100.0 * patch['clearance_violation_rate']:.1f}%`",
        f"- 2D acoustic seal risk rate: `{100.0 * patch['seal_risk_rate']:.1f}%`",
        "",
        "## Hardware Calibration Targets",
        "",
        f"- Target contact force: `{gates['target_contact_force_n']} N`",
        f"- Target spring/sensor compression at 4 N: `{gates['target_spring_compression_mm']} mm`",
        f"- Abort force: `{gates['max_safe_force_n']} N`",
        f"- Abort spring/sensor compression: `{gates['max_safe_spring_compression_mm']} mm`",
        f"- Max recommended travel use before redesign: `{gates['max_recommended_travel_mm']} mm` (`85%` of travel)",
        f"- Minimum tilted printed-part clearance gate: `{gates['min_tilted_print_clearance_mm']} mm`",
        "",
        "## Robot Command Envelope",
        "",
        "| Tissue | stiffness (N/mm) | spring compression (mm) | estimated tissue indent (mm) | robot overdrive (mm) |",
        "|---|---:|---:|---:|---:|",
        *[
            f"| `{row['tissue']}` | {row['tissue_stiffness_n_per_mm']} | {row['spring_compression_mm']} | {row['estimated_tissue_indent_mm']} | {row['robot_overdrive_mm']} |"
            for row in command_rows
        ],
        "",
        "## Pre-Hardware Acceptance Procedure",
        "",
        "1. Measure the printed guide motion with the ThinkLabs proxy installed; confirm smooth travel through at least the target compression without sticking.",
        "2. Calibrate displacement-to-force using a scale or force gauge; the fitted spring rate should be within +/-15% of the target spring rate.",
        "3. Drive the robot from first contact to the medium-tissue command overdrive and confirm measured force is near 4 N and remains below the abort force.",
        "4. Repeat at 0, 5, and 8 degree tilt using a soft pad; printed parts must not touch the pad before the ThinkLabs contact face.",
        "5. Stop and redesign if measured force exceeds the abort force, travel exceeds the travel gate, or the printed cradle touches the pad under tilt.",
        "",
        "## Generated Artifacts",
        "",
        "- `force_displacement_calibration.csv`",
        "- `force_displacement_calibration.png`",
        "- `robot_command_envelope.csv`",
        "- `robot_command_envelope.png`",
        "- `acceptance_gates.json`",
        "",
        "## Limitation",
        "",
        "This package converts the current local simulation result into physical test gates. It does not replace the pending Vast.ai/Isaac Lab rigid-body/contact-sensor validation.",
        "",
    ]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = json.loads((GEN_DIR / "design_metadata.json").read_text(encoding="utf-8"))
    sim = json.loads((SIM_DIR / "simulation_results.json").read_text(encoding="utf-8"))

    mechanics = meta["recommended_mechanics"]
    params = meta["params"]
    spring = float(mechanics["spring_rate_n_per_mm"])
    target_force = float(mechanics["target_contact_force_n"])
    max_force = float(mechanics["max_screened_contact_force_n"])
    travel = float(params["travel"])
    gates = {
        "variant": sim["final_recommendation"],
        "spring_rate_n_per_mm": spring,
        "linear_damping_n_s_per_m": float(mechanics["linear_damping_n_s_per_m"]),
        "target_contact_force_n": target_force,
        "target_spring_compression_mm": round(target_force / spring, 3),
        "max_safe_force_n": max_force,
        "max_safe_spring_compression_mm": round(max_force / spring, 3),
        "travel_mm": travel,
        "max_recommended_travel_mm": round(0.85 * travel, 3),
        "contact_face_projection_mm": float(params["contact_face_projection"]),
        "min_tilted_print_clearance_mm": 5.0,
        "simulation_combined_score": sim["combined_recommendation"][0]["combined_score"],
        "simulated_p99_peak_force_n": sim["robustness_summary"][0]["p99_peak_force_n"],
        "simulated_patch_clearance_violation_rate": sim["contact_patch_summary"][0]["clearance_violation_rate"],
    }

    force_rows = force_displacement_rows(spring, max_force)
    command_rows = command_envelope_rows(spring, target_force)
    write_csv(OUT_DIR / "force_displacement_calibration.csv", force_rows)
    write_csv(OUT_DIR / "robot_command_envelope.csv", command_rows)
    (OUT_DIR / "acceptance_gates.json").write_text(json.dumps(gates, indent=2), encoding="utf-8")
    plot_calibration_curve(force_rows, gates)
    plot_command_envelope(command_rows)
    (OUT_DIR / "ACCEPTANCE_REPORT.md").write_text(
        "\n".join(report_lines(meta, sim, gates, command_rows)),
        encoding="utf-8",
    )
    print(json.dumps({"acceptance_dir": str(OUT_DIR), "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
