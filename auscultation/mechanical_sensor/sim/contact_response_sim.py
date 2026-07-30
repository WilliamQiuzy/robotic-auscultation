"""Local contact-response simulator for the ThinkLabs passive mount.

This is not a replacement for Isaac Sim.  It is a deterministic first-pass
mechanical screen: a 1-DOF spring/damper stethoscope carrier contacts a
Kelvin-Voigt tissue surface under a robot-frame approach command.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CAD_DIR = ROOT / "auscultation" / "mechanical_sensor" / "cad"
sys.path.insert(0, str(CAD_DIR))

tmp_root = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("XDG_CACHE_HOME", str(tmp_root / "cadquery-cache"))
os.environ.setdefault("MPLCONFIGDIR", str(tmp_root / "matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import thinklab_openarm_passive_mount as cad


OUT_DIR = ROOT / "auscultation" / "mechanical_sensor" / "generated" / "simulation"
TARGET_FORCE_N = 4.0
MAX_SAFE_FORCE_N = 8.0
MEDIUM_TISSUE_K_N_PER_MM = 0.8
MIN_CONTACT_CLEARANCE_MM = 5.0
ROBUSTNESS_SAMPLES_PER_VARIANT = 240
ROBUSTNESS_SEED = 20260607
CONTACT_FACE_RADIUS_MM = 26.0
CONTACT_GRID_STEP_MM = 1.25
CONTACT_PATCH_SAMPLES_PER_VARIANT = 360
CONTACT_PATCH_SEED = 20260608


@dataclass(frozen=True)
class TissueScenario:
    name: str
    stiffness_n_per_mm: float
    damping_n_s_per_m: float


@dataclass(frozen=True)
class DesignVariant:
    name: str
    spring_n_per_mm: float
    damping_n_s_per_m: float
    travel_mm: float
    moving_mass_kg: float
    contact_face_projection_mm: float
    fixed_frame_lower_z_mm: float
    guide_rod_bottom_z_mm: float
    cup_wall_mm: float
    guide_bearing_h_mm: float
    notes: str


@dataclass
class SimResult:
    variant: str
    tissue: str
    speed_mm_s: float
    command_depth_mm: float
    steady_force_n: float
    peak_force_n: float
    overshoot_n: float
    settle_time_s: float
    max_compression_mm: float
    travel_used_pct: float
    contact_dropout_count: int
    stop_hit: bool
    score_penalty: float


@dataclass
class RobustnessTrial:
    variant: str
    sample: int
    spring_n_per_mm: float
    damping_n_s_per_m: float
    moving_mass_kg: float
    tissue_stiffness_n_per_mm: float
    tissue_damping_n_s_per_m: float
    speed_mm_s: float
    command_depth_mm: float
    tilt_deg: float
    contact_offset_mm: float
    friction_coeff: float
    steady_force_n: float
    peak_force_n: float
    travel_used_pct: float
    contact_below_lowest_printed_mm: float
    side_force_at_peak_n: float
    guide_friction_loss_n: float
    safety_violation: bool
    travel_margin_violation: bool
    clearance_violation: bool
    binding_risk: bool
    acoustic_seal_risk: bool
    score_penalty: float


@dataclass
class ContactPatchTrial:
    variant: str
    sample: int
    tissue_stiffness_n_per_mm: float
    command_depth_mm: float
    tilt_deg: float
    tilt_heading_deg: float
    contact_offset_mm: float
    surface_curvature_mm: float
    rib_bump_mm: float
    center_penetration_mm: float
    spring_compression_mm: float
    contact_force_n: float
    force_error_n: float
    contact_area_pct: float
    peak_pressure_kpa: float
    mean_pressure_kpa: float
    load_centroid_offset_mm: float
    resultant_moment_nmm: float
    tilted_print_clearance_mm: float
    pressure_hotspot_risk: bool
    acoustic_seal_risk: bool
    clearance_violation: bool
    moment_risk: bool
    score_penalty: float


TISSUES = [
    TissueScenario("soft_tissue", stiffness_n_per_mm=0.35, damping_n_s_per_m=8.0),
    TissueScenario("medium_tissue", stiffness_n_per_mm=0.80, damping_n_s_per_m=12.0),
    TissueScenario("firm_tissue", stiffness_n_per_mm=1.60, damping_n_s_per_m=18.0),
    TissueScenario("bony_prominence", stiffness_n_per_mm=2.50, damping_n_s_per_m=24.0),
]


def contact_patch_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    coords = np.arange(-CONTACT_FACE_RADIUS_MM, CONTACT_FACE_RADIUS_MM + CONTACT_GRID_STEP_MM, CONTACT_GRID_STEP_MM)
    xx, yy = np.meshgrid(coords, coords)
    mask = (xx**2 + yy**2) <= CONTACT_FACE_RADIUS_MM**2
    x = xx[mask].astype(float)
    y = yy[mask].astype(float)
    weights = np.full(x.shape, 1.0 / x.size)
    node_area_mm2 = math.pi * CONTACT_FACE_RADIUS_MM**2 / x.size
    return x, y, weights, node_area_mm2

APPROACH_SPEEDS_MM_S = [15.0, 30.0, 60.0]

VARIANTS = [
    DesignVariant(
        name="v1_current_clear_contact",
        spring_n_per_mm=0.45,
        damping_n_s_per_m=2.4,
        travel_mm=18.0,
        moving_mass_kg=0.30,
        contact_face_projection_mm=9.0,
        fixed_frame_lower_z_mm=-38.0,
        guide_rod_bottom_z_mm=-74.0,
        cup_wall_mm=6.0,
        guide_bearing_h_mm=26.0,
        notes="Current v1 geometry with moderate spring and clear exposed contact face.",
    ),
    DesignVariant(
        name="v2_soft_long_travel",
        spring_n_per_mm=0.30,
        damping_n_s_per_m=3.6,
        travel_mm=26.0,
        moving_mass_kg=0.30,
        contact_face_projection_mm=12.0,
        fixed_frame_lower_z_mm=-32.0,
        guide_rod_bottom_z_mm=-84.0,
        cup_wall_mm=5.5,
        guide_bearing_h_mm=30.0,
        notes="Safer softer spring, more travel, larger skin clearance.",
    ),
    DesignVariant(
        name="v3_balanced_damped",
        spring_n_per_mm=0.38,
        damping_n_s_per_m=4.2,
        travel_mm=24.0,
        moving_mass_kg=0.32,
        contact_face_projection_mm=13.0,
        fixed_frame_lower_z_mm=-34.0,
        guide_rod_bottom_z_mm=-82.0,
        cup_wall_mm=6.0,
        guide_bearing_h_mm=30.0,
        notes="Balanced candidate: target force with improved damping and travel margin.",
    ),
    DesignVariant(
        name="v6_balanced_high_clearance",
        spring_n_per_mm=0.38,
        damping_n_s_per_m=4.2,
        travel_mm=24.0,
        moving_mass_kg=0.32,
        contact_face_projection_mm=16.0,
        fixed_frame_lower_z_mm=-34.0,
        guide_rod_bottom_z_mm=-82.0,
        cup_wall_mm=6.0,
        guide_bearing_h_mm=30.0,
        notes="v3 mechanics with 3 mm more contact-face protrusion to protect tilted skin clearance.",
    ),
    DesignVariant(
        name="v4_compact_stiff",
        spring_n_per_mm=0.65,
        damping_n_s_per_m=2.0,
        travel_mm=16.0,
        moving_mass_kg=0.28,
        contact_face_projection_mm=8.0,
        fixed_frame_lower_z_mm=-40.0,
        guide_rod_bottom_z_mm=-72.0,
        cup_wall_mm=6.5,
        guide_bearing_h_mm=24.0,
        notes="Compact but stiffer; useful as a safety-regression baseline.",
    ),
    DesignVariant(
        name="v5_very_soft_controlled",
        spring_n_per_mm=0.24,
        damping_n_s_per_m=5.5,
        travel_mm=30.0,
        moving_mass_kg=0.34,
        contact_face_projection_mm=15.0,
        fixed_frame_lower_z_mm=-30.0,
        guide_rod_bottom_z_mm=-88.0,
        cup_wall_mm=5.5,
        guide_bearing_h_mm=32.0,
        notes="Very soft long-travel candidate; tests whether passive safety costs too much force authority.",
    ),
]


def cad_params_for_variant(v: DesignVariant) -> cad.MountParams:
    return cad.MountParams(
        travel=v.travel_mm,
        contact_face_projection=v.contact_face_projection_mm,
        fixed_frame_lower_z=v.fixed_frame_lower_z_mm,
        guide_rod_bottom_z=v.guide_rod_bottom_z_mm,
        cup_wall=v.cup_wall_mm,
        guide_bearing_h=v.guide_bearing_h_mm,
    )


def geometry_clearance_mm(v: DesignVariant) -> dict[str, float]:
    params = cad_params_for_variant(v)
    fixed = cad.make_fixed_frame(params)
    carrier = cad.make_floating_carriage(params)
    thinklab = cad.make_thinklab_proxy(params)
    fixed_bb = fixed.val().BoundingBox()
    carrier_bb = carrier.val().BoundingBox()
    thinklab_bb = thinklab.val().BoundingBox()
    printed_lowest = min(fixed_bb.zmin, carrier_bb.zmin)
    return {
        "fixed_zmin_mm": round(fixed_bb.zmin, 3),
        "carrier_zmin_mm": round(carrier_bb.zmin, 3),
        "thinklab_contact_zmin_mm": round(thinklab_bb.zmin, 3),
        "contact_below_lowest_printed_mm": round(printed_lowest - thinklab_bb.zmin, 3),
        "contact_below_fixed_frame_mm": round(fixed_bb.zmin - thinklab_bb.zmin, 3),
        "contact_below_carrier_mm": round(carrier_bb.zmin - thinklab_bb.zmin, 3),
        "fixed_solids": fixed.solids().size(),
        "carrier_solids": carrier.solids().size(),
    }


def command_depth_for_variant(v: DesignVariant) -> float:
    """Robot-frame overdrive that gives TARGET_FORCE_N on medium tissue at static equilibrium."""
    return TARGET_FORCE_N / v.spring_n_per_mm + TARGET_FORCE_N / MEDIUM_TISSUE_K_N_PER_MM


def run_response(
    variant: DesignVariant,
    tissue: TissueScenario,
    speed_mm_s: float,
    command_depth_override_mm: float | None = None,
    dt_s: float = 0.0005,
    t_end_s: float = 3.5,
) -> tuple[SimResult, dict[str, np.ndarray]]:
    k_s = variant.spring_n_per_mm * 1000.0
    k_t = tissue.stiffness_n_per_mm * 1000.0
    travel_m = variant.travel_mm / 1000.0
    command_depth_mm = command_depth_override_mm if command_depth_override_mm is not None else command_depth_for_variant(variant)
    target_q_m = command_depth_mm / 1000.0
    start_q_m = -0.003
    speed_m_s = speed_mm_s / 1000.0
    ramp_time_s = (target_q_m - start_q_m) / speed_m_s

    n = int(t_end_s / dt_s) + 1
    time = np.linspace(0.0, t_end_s, n)
    q = np.minimum(start_q_m + speed_m_s * time, target_q_m)
    qdot = np.where(time < ramp_time_s, speed_m_s, 0.0)

    x = -0.003
    v = 0.0
    x_hist = np.zeros(n)
    force_hist = np.zeros(n)
    compression_hist = np.zeros(n)
    stop_hit = False

    for i in range(n):
        raw_compression = q[i] - x
        compression = max(0.0, min(raw_compression, travel_m))
        stop_over = max(0.0, raw_compression - travel_m)
        stop_hit = stop_hit or stop_over > 1e-6

        spring_force = k_s * compression + variant.damping_n_s_per_m * (qdot[i] - v)
        spring_force += 6000.0 * stop_over
        spring_force = max(0.0, spring_force)

        if x > 0.0:
            tissue_force = k_t * x + tissue.damping_n_s_per_m * v
            tissue_force = max(0.0, tissue_force)
        else:
            tissue_force = 0.0

        acc = (spring_force - tissue_force) / variant.moving_mass_kg
        v += acc * dt_s
        x += v * dt_s

        # The carrier cannot move infinitely away from the robot frame.
        if q[i] - x > travel_m:
            x = q[i] - travel_m
            v = min(v, qdot[i])

        x_hist[i] = x
        force_hist[i] = tissue_force
        compression_hist[i] = max(0.0, q[i] - x)

    hold_mask = time > (t_end_s - 0.5)
    steady_force = float(np.mean(force_hist[hold_mask]))
    peak_force = float(np.max(force_hist))
    overshoot = max(0.0, peak_force - steady_force)
    max_compression_mm = float(np.max(compression_hist) * 1000.0)
    travel_used_pct = 100.0 * max_compression_mm / variant.travel_mm

    first_contact_indices = np.where(force_hist > 0.2)[0]
    if first_contact_indices.size:
        after = force_hist[first_contact_indices[0] :]
        dropout_count = int(np.sum((after[:-1] > 0.2) & (after[1:] <= 0.2)))
    else:
        dropout_count = 1

    tolerance = 0.5
    settle_time = t_end_s
    for i in range(n):
        if time[i] < ramp_time_s:
            continue
        window = force_hist[i:]
        if window.size and np.all(np.abs(window - steady_force) <= tolerance):
            settle_time = float(time[i] - ramp_time_s)
            break

    penalty = 0.0
    penalty += abs(steady_force - TARGET_FORCE_N) * 2.0
    penalty += max(0.0, peak_force - MAX_SAFE_FORCE_N) * 20.0
    penalty += max(0.0, overshoot - 1.5) * 4.0
    penalty += max(0.0, travel_used_pct - 85.0) * 0.2
    penalty += min(settle_time, 2.0) * 0.4
    penalty += dropout_count * 5.0
    penalty += 10.0 if stop_hit else 0.0

    result = SimResult(
        variant=variant.name,
        tissue=tissue.name,
        speed_mm_s=speed_mm_s,
        command_depth_mm=command_depth_mm,
        steady_force_n=steady_force,
        peak_force_n=peak_force,
        overshoot_n=overshoot,
        settle_time_s=settle_time,
        max_compression_mm=max_compression_mm,
        travel_used_pct=travel_used_pct,
        contact_dropout_count=dropout_count,
        stop_hit=stop_hit,
        score_penalty=penalty,
    )
    series = {
        "time_s": time,
        "force_n": force_hist,
        "compression_mm": compression_hist * 1000.0,
        "contact_position_mm": x_hist * 1000.0,
        "command_mm": q * 1000.0,
    }
    return result, series


def sample_robustness_trial(
    nominal: DesignVariant,
    sample_idx: int,
    rng: np.random.Generator,
    nominal_clearance_mm: float,
) -> RobustnessTrial:
    spring_factor = float(np.clip(rng.normal(1.0, 0.15), 0.65, 1.40))
    damping_factor = float(np.clip(rng.normal(1.0, 0.25), 0.45, 1.75))
    mass_factor = float(np.clip(rng.normal(1.0, 0.08), 0.78, 1.25))
    projection_error_mm = float(np.clip(rng.normal(0.0, 0.8), -2.0, 2.0))
    travel_error_mm = float(np.clip(rng.normal(0.0, 0.7), -1.8, 1.8))
    print_low_error_mm = float(np.clip(rng.normal(0.0, 0.5), -1.2, 1.2))

    tissue_k = float(np.exp(rng.normal(math.log(0.85), 0.48)))
    tissue_k = float(np.clip(tissue_k, 0.25, 3.2))
    tissue_damping = float(8.0 + 8.5 * tissue_k + rng.normal(0.0, 2.0))
    tissue_damping = float(np.clip(tissue_damping, 5.0, 32.0))
    speed = float(rng.choice(APPROACH_SPEEDS_MM_S))
    command_error_mm = float(np.clip(rng.normal(0.0, 1.0), -2.0, 2.0))
    tilt_deg = float(np.clip(abs(rng.normal(3.0, 2.2)), 0.0, 11.0))
    contact_offset_mm = float(np.clip(rng.rayleigh(3.5), 0.0, 14.0))
    friction_coeff = float(np.clip(rng.normal(0.12, 0.035), 0.05, 0.22))

    perturbed = replace(
        nominal,
        spring_n_per_mm=nominal.spring_n_per_mm * spring_factor,
        damping_n_s_per_m=nominal.damping_n_s_per_m * damping_factor,
        moving_mass_kg=nominal.moving_mass_kg * mass_factor,
        travel_mm=max(8.0, nominal.travel_mm + travel_error_mm),
        contact_face_projection_mm=nominal.contact_face_projection_mm + projection_error_mm,
    )
    command_depth_mm = command_depth_for_variant(nominal) + command_error_mm
    tissue = TissueScenario("robustness_sample", tissue_k, tissue_damping)
    result, _ = run_response(perturbed, tissue, speed, command_depth_override_mm=command_depth_mm)

    tilt_rad = math.radians(tilt_deg)
    side_force = result.peak_force_n * math.tan(tilt_rad)
    guide_span_m = max(1.0, 2.0 * nominal.guide_x if hasattr(nominal, "guide_x") else 84.0) / 1000.0
    # The contact offset creates a moment that guide bushings must react. This
    # is a screen for stick-slip/binding tendency, not a detailed bearing model.
    moment_nm = result.peak_force_n * (contact_offset_mm / 1000.0)
    moment_reaction_n = moment_nm / guide_span_m
    guide_friction_loss = friction_coeff * (abs(side_force) + 2.0 * abs(moment_reaction_n))
    clearance_mm = nominal_clearance_mm + projection_error_mm + print_low_error_mm

    safety_violation = result.peak_force_n > MAX_SAFE_FORCE_N
    travel_violation = result.travel_used_pct > 85.0 or result.stop_hit
    clearance_violation = clearance_mm < MIN_CONTACT_CLEARANCE_MM
    binding_risk = guide_friction_loss > max(0.75, 0.18 * max(result.peak_force_n, 0.1))
    acoustic_seal_risk = tilt_deg > 8.0 or contact_offset_mm > 11.0 or side_force > 0.75

    penalty = result.score_penalty
    penalty += 40.0 if safety_violation else 0.0
    penalty += 12.0 if travel_violation else 0.0
    penalty += 20.0 if clearance_violation else 0.0
    penalty += 10.0 if binding_risk else 0.0
    penalty += 4.0 if acoustic_seal_risk else 0.0
    penalty += max(0.0, guide_friction_loss - 0.5) * 1.5

    return RobustnessTrial(
        variant=nominal.name,
        sample=sample_idx,
        spring_n_per_mm=perturbed.spring_n_per_mm,
        damping_n_s_per_m=perturbed.damping_n_s_per_m,
        moving_mass_kg=perturbed.moving_mass_kg,
        tissue_stiffness_n_per_mm=tissue_k,
        tissue_damping_n_s_per_m=tissue_damping,
        speed_mm_s=speed,
        command_depth_mm=command_depth_mm,
        tilt_deg=tilt_deg,
        contact_offset_mm=contact_offset_mm,
        friction_coeff=friction_coeff,
        steady_force_n=result.steady_force_n,
        peak_force_n=result.peak_force_n,
        travel_used_pct=result.travel_used_pct,
        contact_below_lowest_printed_mm=clearance_mm,
        side_force_at_peak_n=side_force,
        guide_friction_loss_n=guide_friction_loss,
        safety_violation=safety_violation,
        travel_margin_violation=travel_violation,
        clearance_violation=clearance_violation,
        binding_risk=binding_risk,
        acoustic_seal_risk=acoustic_seal_risk,
        score_penalty=penalty,
    )


def run_robustness_sweep(
    geometry: dict[str, dict[str, float]],
    samples_per_variant: int = ROBUSTNESS_SAMPLES_PER_VARIANT,
) -> tuple[list[RobustnessTrial], list[dict[str, float | str]]]:
    rng = np.random.default_rng(ROBUSTNESS_SEED)
    trials: list[RobustnessTrial] = []
    for variant in VARIANTS:
        clearance = geometry[variant.name]["contact_below_lowest_printed_mm"]
        for idx in range(samples_per_variant):
            trials.append(sample_robustness_trial(variant, idx, rng, clearance))

    summary = []
    for variant in sorted({t.variant for t in trials}):
        subset = [t for t in trials if t.variant == variant]
        peak = np.asarray([t.peak_force_n for t in subset])
        steady_error = np.asarray([abs(t.steady_force_n - TARGET_FORCE_N) for t in subset])
        travel = np.asarray([t.travel_used_pct for t in subset])
        penalties = np.asarray([t.score_penalty for t in subset])
        geometry_penalty = 0.0
        if geometry[variant]["fixed_solids"] != 1:
            geometry_penalty += 8.0
        if geometry[variant]["carrier_solids"] != 1:
            geometry_penalty += 8.0
        nominal_clearance = geometry[variant]["contact_below_lowest_printed_mm"]
        if nominal_clearance < MIN_CONTACT_CLEARANCE_MM:
            geometry_penalty += (MIN_CONTACT_CLEARANCE_MM - nominal_clearance) * 10.0
        row = {
            "variant": variant,
            "robust_score": round(float(np.mean(penalties)) + geometry_penalty, 3),
            "mean_dynamic_penalty": round(float(np.mean(penalties)), 3),
            "geometry_penalty": round(float(geometry_penalty), 3),
            "p95_peak_force_n": round(float(np.percentile(peak, 95)), 3),
            "p99_peak_force_n": round(float(np.percentile(peak, 99)), 3),
            "mean_force_error_n": round(float(np.mean(steady_error)), 3),
            "p95_travel_used_pct": round(float(np.percentile(travel, 95)), 1),
            "safety_violation_rate": round(float(np.mean([t.safety_violation for t in subset])), 4),
            "travel_violation_rate": round(float(np.mean([t.travel_margin_violation for t in subset])), 4),
            "clearance_violation_rate": round(float(np.mean([t.clearance_violation for t in subset])), 4),
            "binding_risk_rate": round(float(np.mean([t.binding_risk for t in subset])), 4),
            "acoustic_seal_risk_rate": round(float(np.mean([t.acoustic_seal_risk for t in subset])), 4),
        }
        summary.append(row)
    return trials, sorted(summary, key=lambda x: float(x["robust_score"]))


def solve_contact_patch(
    variant: DesignVariant,
    tissue_stiffness_n_per_mm: float,
    command_depth_mm: float,
    tilt_deg: float,
    tilt_heading_deg: float,
    contact_offset_mm: float,
    surface_curvature_mm: float,
    rib_bump_mm: float,
    nominal_clearance_mm: float,
    print_low_error_mm: float = 0.0,
    sample_idx: int = 0,
) -> tuple[ContactPatchTrial, dict[str, np.ndarray]]:
    """Static 2D contact patch screen for tilted stethoscope-to-tissue contact."""
    x, y, weights, node_area_mm2 = contact_patch_grid()
    heading = math.radians(tilt_heading_deg)
    tilt = math.radians(tilt_deg)
    slope_x = math.tan(tilt) * math.cos(heading)
    slope_y = math.tan(tilt) * math.sin(heading)

    radial = np.sqrt(x**2 + y**2) / CONTACT_FACE_RADIUS_MM
    curvature = surface_curvature_mm * (1.0 - np.clip(radial, 0.0, 1.0) ** 2)
    rib_x = contact_offset_mm * math.cos(heading + math.pi / 2.0)
    rib_y = contact_offset_mm * math.sin(heading + math.pi / 2.0)
    rib_sigma_mm = 6.5
    rib = rib_bump_mm * np.exp(-((x - rib_x) ** 2 + (y - rib_y) ** 2) / (2.0 * rib_sigma_mm**2))
    tissue_surface_mm = curvature + rib

    k_s = variant.spring_n_per_mm
    q = command_depth_mm

    def tissue_force(center_z_mm: float) -> tuple[float, np.ndarray]:
        indentation = center_z_mm + slope_x * x + slope_y * y + tissue_surface_mm
        indentation = np.maximum(indentation, 0.0)
        node_forces = tissue_stiffness_n_per_mm * indentation * weights
        return float(np.sum(node_forces)), node_forces

    def spring_force(center_z_mm: float) -> float:
        compression = min(max(q - center_z_mm, 0.0), variant.travel_mm)
        return k_s * compression

    low = -CONTACT_FACE_RADIUS_MM * abs(math.tan(tilt)) - max(surface_curvature_mm, rib_bump_mm, 0.0) - 6.0
    high = q + 6.0
    for _ in range(90):
        mid = 0.5 * (low + high)
        tissue_f, _ = tissue_force(mid)
        diff = spring_force(mid) - tissue_f
        if diff > 0.0:
            low = mid
        else:
            high = mid
    center_z = 0.5 * (low + high)
    force, node_forces = tissue_force(center_z)
    compression = min(max(q - center_z, 0.0), variant.travel_mm)

    contacted = node_forces > 1e-6
    contact_area_pct = 100.0 * float(np.mean(contacted))
    pressure_kpa = node_forces / (node_area_mm2 * 1e-6) / 1000.0
    peak_pressure = float(np.max(pressure_kpa)) if pressure_kpa.size else 0.0
    mean_pressure = float(np.mean(pressure_kpa[contacted])) if np.any(contacted) else 0.0
    if force > 1e-9:
        centroid_x = float(np.sum(x * node_forces) / force)
        centroid_y = float(np.sum(y * node_forces) / force)
    else:
        centroid_x = 0.0
        centroid_y = 0.0
    centroid_offset = math.hypot(centroid_x, centroid_y)
    guide_axis_offset_x = contact_offset_mm * math.cos(heading)
    guide_axis_offset_y = contact_offset_mm * math.sin(heading)
    moment_arm = math.hypot(centroid_x + guide_axis_offset_x, centroid_y + guide_axis_offset_y)
    resultant_moment = force * moment_arm

    holder_outer_radius = cad.MountParams().thinklab_body_d / 2.0 + cad.MountParams().cup_clearance + variant.cup_wall_mm
    tilted_clearance = nominal_clearance_mm + print_low_error_mm - holder_outer_radius * math.sin(tilt)

    pressure_hotspot = peak_pressure > 12.0
    acoustic_seal = contact_area_pct < 52.0 or centroid_offset > 9.0 or force < 2.5
    clearance_violation = tilted_clearance < MIN_CONTACT_CLEARANCE_MM
    moment_risk = resultant_moment > 70.0

    penalty = 0.0
    penalty += abs(force - TARGET_FORCE_N) * 1.8
    penalty += max(0.0, peak_pressure - 10.0) * 0.35
    penalty += max(0.0, 62.0 - contact_area_pct) * 0.06
    penalty += max(0.0, centroid_offset - 6.0) * 0.25
    penalty += max(0.0, resultant_moment - 45.0) * 0.035
    penalty += 25.0 if force > MAX_SAFE_FORCE_N else 0.0
    penalty += 8.0 if pressure_hotspot else 0.0
    penalty += 6.0 if acoustic_seal else 0.0
    penalty += 12.0 if clearance_violation else 0.0
    penalty += 6.0 if moment_risk else 0.0

    trial = ContactPatchTrial(
        variant=variant.name,
        sample=sample_idx,
        tissue_stiffness_n_per_mm=tissue_stiffness_n_per_mm,
        command_depth_mm=command_depth_mm,
        tilt_deg=tilt_deg,
        tilt_heading_deg=tilt_heading_deg,
        contact_offset_mm=contact_offset_mm,
        surface_curvature_mm=surface_curvature_mm,
        rib_bump_mm=rib_bump_mm,
        center_penetration_mm=center_z,
        spring_compression_mm=compression,
        contact_force_n=force,
        force_error_n=abs(force - TARGET_FORCE_N),
        contact_area_pct=contact_area_pct,
        peak_pressure_kpa=peak_pressure,
        mean_pressure_kpa=mean_pressure,
        load_centroid_offset_mm=centroid_offset,
        resultant_moment_nmm=resultant_moment,
        tilted_print_clearance_mm=tilted_clearance,
        pressure_hotspot_risk=pressure_hotspot,
        acoustic_seal_risk=acoustic_seal,
        clearance_violation=clearance_violation,
        moment_risk=moment_risk,
        score_penalty=penalty,
    )
    fields = {
        "x_mm": x,
        "y_mm": y,
        "pressure_kpa": pressure_kpa,
        "node_forces_n": node_forces,
        "tissue_surface_mm": tissue_surface_mm,
    }
    return trial, fields


def sample_contact_patch_trial(
    nominal: DesignVariant,
    sample_idx: int,
    rng: np.random.Generator,
    nominal_clearance_mm: float,
) -> ContactPatchTrial:
    tissue_k = float(np.exp(rng.normal(math.log(0.85), 0.50)))
    tissue_k = float(np.clip(tissue_k, 0.25, 3.2))
    tilt_deg = float(np.clip(abs(rng.normal(3.2, 2.4)), 0.0, 11.5))
    heading_deg = float(rng.uniform(0.0, 360.0))
    contact_offset_mm = float(np.clip(rng.rayleigh(3.8), 0.0, 14.0))
    curvature_mm = float(np.clip(rng.normal(0.7, 0.35), -0.2, 1.8))
    rib_bump_mm = float(np.clip(rng.gamma(1.3, 0.38), 0.0, 1.9))
    command_error_mm = float(np.clip(rng.normal(0.0, 1.0), -2.0, 2.0))
    print_low_error_mm = float(np.clip(rng.normal(0.0, 0.45), -1.1, 1.1))
    projection_error_mm = float(np.clip(rng.normal(0.0, 0.75), -1.8, 1.8))
    travel_error_mm = float(np.clip(rng.normal(0.0, 0.65), -1.6, 1.6))
    spring_factor = float(np.clip(rng.normal(1.0, 0.14), 0.68, 1.35))

    perturbed = replace(
        nominal,
        spring_n_per_mm=nominal.spring_n_per_mm * spring_factor,
        travel_mm=max(8.0, nominal.travel_mm + travel_error_mm),
        contact_face_projection_mm=nominal.contact_face_projection_mm + projection_error_mm,
    )
    command_depth = command_depth_for_variant(nominal) + command_error_mm
    trial, _ = solve_contact_patch(
        perturbed,
        tissue_stiffness_n_per_mm=tissue_k,
        command_depth_mm=command_depth,
        tilt_deg=tilt_deg,
        tilt_heading_deg=heading_deg,
        contact_offset_mm=contact_offset_mm,
        surface_curvature_mm=curvature_mm,
        rib_bump_mm=rib_bump_mm,
        nominal_clearance_mm=nominal_clearance_mm + projection_error_mm,
        print_low_error_mm=print_low_error_mm,
        sample_idx=sample_idx,
    )
    return replace(trial, variant=nominal.name)


def run_contact_patch_sweep(
    geometry: dict[str, dict[str, float]],
    samples_per_variant: int = CONTACT_PATCH_SAMPLES_PER_VARIANT,
) -> tuple[list[ContactPatchTrial], list[dict[str, float | str]]]:
    rng = np.random.default_rng(CONTACT_PATCH_SEED)
    trials: list[ContactPatchTrial] = []
    for variant in VARIANTS:
        clearance = geometry[variant.name]["contact_below_lowest_printed_mm"]
        for idx in range(samples_per_variant):
            trials.append(sample_contact_patch_trial(variant, idx, rng, clearance))

    summary = []
    for variant in sorted({t.variant for t in trials}):
        subset = [t for t in trials if t.variant == variant]
        penalties = np.asarray([t.score_penalty for t in subset])
        pressure = np.asarray([t.peak_pressure_kpa for t in subset])
        area = np.asarray([t.contact_area_pct for t in subset])
        centroid = np.asarray([t.load_centroid_offset_mm for t in subset])
        moment = np.asarray([t.resultant_moment_nmm for t in subset])
        force_error = np.asarray([t.force_error_n for t in subset])
        geometry_penalty = 0.0
        if geometry[variant]["fixed_solids"] != 1:
            geometry_penalty += 8.0
        if geometry[variant]["carrier_solids"] != 1:
            geometry_penalty += 8.0
        row = {
            "variant": variant,
            "patch_score": round(float(np.mean(penalties)) + geometry_penalty, 3),
            "mean_patch_penalty": round(float(np.mean(penalties)), 3),
            "geometry_penalty": round(float(geometry_penalty), 3),
            "mean_force_error_n": round(float(np.mean(force_error)), 3),
            "p95_peak_pressure_kpa": round(float(np.percentile(pressure, 95)), 3),
            "p05_contact_area_pct": round(float(np.percentile(area, 5)), 1),
            "p95_centroid_offset_mm": round(float(np.percentile(centroid, 95)), 2),
            "p95_resultant_moment_nmm": round(float(np.percentile(moment, 95)), 1),
            "pressure_hotspot_rate": round(float(np.mean([t.pressure_hotspot_risk for t in subset])), 4),
            "seal_risk_rate": round(float(np.mean([t.acoustic_seal_risk for t in subset])), 4),
            "clearance_violation_rate": round(float(np.mean([t.clearance_violation for t in subset])), 4),
            "moment_risk_rate": round(float(np.mean([t.moment_risk for t in subset])), 4),
        }
        summary.append(row)
    return trials, sorted(summary, key=lambda x: float(x["patch_score"]))


def summarize_results(results: list[SimResult], geometry: dict[str, dict[str, float]]) -> list[dict[str, float | str]]:
    rows = []
    for variant in sorted({r.variant for r in results}):
        subset = [r for r in results if r.variant == variant]
        peak = max(r.peak_force_n for r in subset)
        mean_steady_error = float(np.mean([abs(r.steady_force_n - TARGET_FORCE_N) for r in subset]))
        mean_penalty = float(np.mean([r.score_penalty for r in subset]))
        worst_travel = max(r.travel_used_pct for r in subset)
        worst_settle = max(r.settle_time_s for r in subset)
        stop_hits = sum(1 for r in subset if r.stop_hit)
        clearance = geometry[variant]["contact_below_lowest_printed_mm"]
        geometry_penalty = 0.0 if clearance >= 5.0 else (5.0 - clearance) * 10.0
        geometry_penalty += 8.0 if geometry[variant]["fixed_solids"] != 1 else 0.0
        geometry_penalty += 8.0 if geometry[variant]["carrier_solids"] != 1 else 0.0
        total_score = mean_penalty + geometry_penalty
        rows.append(
            {
                "variant": variant,
                "score": round(total_score, 3),
                "mean_force_error_n": round(mean_steady_error, 3),
                "peak_force_n": round(peak, 3),
                "worst_travel_used_pct": round(worst_travel, 1),
                "worst_settle_time_s": round(worst_settle, 3),
                "stop_hits": stop_hits,
                "contact_below_lowest_printed_mm": clearance,
            }
        )
    return sorted(rows, key=lambda x: float(x["score"]))


def combined_recommendation(
    summary_rows: list[dict],
    robust_rows: list[dict],
    patch_rows: list[dict],
) -> list[dict[str, float | str]]:
    summary_by_variant = {str(r["variant"]): r for r in summary_rows}
    robust_by_variant = {str(r["variant"]): r for r in robust_rows}
    patch_by_variant = {str(r["variant"]): r for r in patch_rows}
    rows = []
    for variant in sorted(summary_by_variant):
        deterministic = float(summary_by_variant[variant]["score"])
        robust = float(robust_by_variant[variant]["robust_score"])
        patch = float(patch_by_variant[variant]["patch_score"])
        safety_rate = float(robust_by_variant[variant]["safety_violation_rate"])
        travel_rate = float(robust_by_variant[variant]["travel_violation_rate"])
        clearance_rate = max(
            float(robust_by_variant[variant]["clearance_violation_rate"]),
            float(patch_by_variant[variant]["clearance_violation_rate"]),
        )
        combined = deterministic + robust + patch
        combined += 100.0 * safety_rate + 30.0 * travel_rate + 10.0 * clearance_rate
        rows.append(
            {
                "variant": variant,
                "combined_score": round(combined, 3),
                "deterministic_score": deterministic,
                "robust_score": robust,
                "patch_score": patch,
                "safety_violation_rate": safety_rate,
                "travel_violation_rate": travel_rate,
                "max_clearance_violation_rate": clearance_rate,
            }
        )
    return sorted(rows, key=lambda x: float(x["combined_score"]))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_force_profiles(series_by_variant: dict[str, dict[str, np.ndarray]]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), dpi=160, sharex=True)
    for variant, series in series_by_variant.items():
        axes[0].plot(series["time_s"], series["force_n"], label=variant, linewidth=1.8)
        axes[1].plot(series["time_s"], series["compression_mm"], label=variant, linewidth=1.8)
    axes[0].axhline(TARGET_FORCE_N, color="black", linestyle="--", linewidth=1.0, label="target 4 N")
    axes[0].axhline(MAX_SAFE_FORCE_N, color="red", linestyle=":", linewidth=1.0, label="8 N safety line")
    axes[0].set_ylabel("contact force (N)")
    axes[1].set_ylabel("spring travel used (mm)")
    axes[1].set_xlabel("time (s)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Medium-tissue 30 mm/s approach: force feedback response")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "force_profiles_medium_30mms.png")
    plt.close(fig)


def plot_summary(summary_rows: list[dict]) -> None:
    variants = [str(r["variant"]) for r in summary_rows]
    scores = [float(r["score"]) for r in summary_rows]
    peak_forces = [float(r["peak_force_n"]) for r in summary_rows]
    travel = [float(r["worst_travel_used_pct"]) for r in summary_rows]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=160)
    axes[0].barh(variants, scores, color="#3f6fa7")
    axes[0].invert_yaxis()
    axes[0].set_title("Lower score is better")
    axes[0].set_xlabel("score")
    axes[1].barh(variants, peak_forces, color="#648f54")
    axes[1].axvline(MAX_SAFE_FORCE_N, color="red", linestyle=":", linewidth=1.0)
    axes[1].invert_yaxis()
    axes[1].set_title("Worst peak force")
    axes[1].set_xlabel("N")
    axes[2].barh(variants, travel, color="#b2794c")
    axes[2].axvline(85.0, color="red", linestyle=":", linewidth=1.0)
    axes[2].invert_yaxis()
    axes[2].set_title("Worst travel used")
    axes[2].set_xlabel("%")
    for ax in axes:
        ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "variant_score_summary.png")
    plt.close(fig)


def plot_robustness_summary(robust_rows: list[dict]) -> None:
    variants = [str(r["variant"]) for r in robust_rows]
    scores = [float(r["robust_score"]) for r in robust_rows]
    peak = [float(r["p95_peak_force_n"]) for r in robust_rows]
    bind = [100.0 * float(r["binding_risk_rate"]) for r in robust_rows]
    seal = [100.0 * float(r["acoustic_seal_risk_rate"]) for r in robust_rows]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.8), dpi=160)
    axes[0].barh(variants, scores, color="#3f6fa7")
    axes[0].invert_yaxis()
    axes[0].set_title("Robust score")
    axes[0].set_xlabel("lower is better")
    axes[1].barh(variants, peak, color="#648f54")
    axes[1].axvline(MAX_SAFE_FORCE_N, color="red", linestyle=":", linewidth=1.0)
    axes[1].invert_yaxis()
    axes[1].set_title("P95 peak force")
    axes[1].set_xlabel("N")
    axes[2].barh(variants, bind, color="#b2794c")
    axes[2].invert_yaxis()
    axes[2].set_title("Binding-risk samples")
    axes[2].set_xlabel("%")
    if max(bind) == 0.0:
        axes[2].set_xlim(0.0, 1.0)
    axes[3].barh(variants, seal, color="#7c659f")
    axes[3].invert_yaxis()
    axes[3].set_title("Seal-risk samples")
    axes[3].set_xlabel("%")
    for ax in axes:
        ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "robustness_summary.png")
    plt.close(fig)


def plot_contact_patch_summary(patch_rows: list[dict]) -> None:
    variants = [str(r["variant"]) for r in patch_rows]
    scores = [float(r["patch_score"]) for r in patch_rows]
    pressure = [float(r["p95_peak_pressure_kpa"]) for r in patch_rows]
    area = [float(r["p05_contact_area_pct"]) for r in patch_rows]
    clearance = [100.0 * float(r["clearance_violation_rate"]) for r in patch_rows]
    seal = [100.0 * float(r["seal_risk_rate"]) for r in patch_rows]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.8), dpi=160)
    axes[0].barh(variants, scores, color="#3f6fa7")
    axes[0].invert_yaxis()
    axes[0].set_title("2D patch score")
    axes[0].set_xlabel("lower is better")
    axes[1].barh(variants, pressure, color="#b85d4b")
    axes[1].axvline(12.0, color="red", linestyle=":", linewidth=1.0)
    axes[1].invert_yaxis()
    axes[1].set_title("P95 peak pressure")
    axes[1].set_xlabel("kPa")
    axes[2].barh(variants, area, color="#648f54")
    axes[2].axvline(52.0, color="red", linestyle=":", linewidth=1.0)
    axes[2].invert_yaxis()
    axes[2].set_title("P05 contact area")
    axes[2].set_xlabel("% face in contact")
    axes[3].barh(variants, clearance, color="#b2794c")
    axes[3].invert_yaxis()
    axes[3].set_title("Tilted clearance risk")
    axes[3].set_xlabel("%")
    axes[4].barh(variants, seal, color="#7c659f")
    axes[4].invert_yaxis()
    axes[4].set_title("Acoustic seal risk")
    axes[4].set_xlabel("%")
    for ax in axes:
        ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "contact_patch_summary.png")
    plt.close(fig)


def plot_final_pressure_maps(geometry: dict[str, dict[str, float]], variant_name: str) -> None:
    variant = next(v for v in VARIANTS if v.name == variant_name)
    clearance = geometry[variant.name]["contact_below_lowest_printed_mm"]
    command = command_depth_for_variant(variant)
    scenarios = [
        ("Flat medium tissue", 0.0, 0.0, 0.0, 0.0, 0.8),
        ("6 deg tilted chest", 6.0, 35.0, 4.0, 0.7, 0.8),
        ("8 deg + rib bump", 8.0, 70.0, 8.0, 0.9, 1.4),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5), dpi=170)
    for ax, (title, tilt, heading, offset, curvature, rib_k) in zip(axes, scenarios):
        _, fields = solve_contact_patch(
            variant,
            tissue_stiffness_n_per_mm=rib_k,
            command_depth_mm=command,
            tilt_deg=tilt,
            tilt_heading_deg=heading,
            contact_offset_mm=offset,
            surface_curvature_mm=curvature,
            rib_bump_mm=1.2 if "rib" in title.lower() else 0.2,
            nominal_clearance_mm=clearance,
        )
        scatter = ax.scatter(
            fields["x_mm"],
            fields["y_mm"],
            c=fields["pressure_kpa"],
            s=10,
            cmap="inferno",
            vmin=0.0,
            vmax=max(12.0, float(np.max(fields["pressure_kpa"]))),
            linewidths=0.0,
        )
        circle = plt.Circle((0, 0), CONTACT_FACE_RADIUS_MM, color="#202326", fill=False, linewidth=1.0)
        ax.add_patch(circle)
        ax.set_aspect("equal")
        ax.set_xlim(-CONTACT_FACE_RADIUS_MM - 2, CONTACT_FACE_RADIUS_MM + 2)
        ax.set_ylim(-CONTACT_FACE_RADIUS_MM - 2, CONTACT_FACE_RADIUS_MM + 2)
        ax.set_title(title)
        ax.set_xlabel("x (mm)")
        ax.grid(True, alpha=0.18)
        if ax is axes[0]:
            ax.set_ylabel("y (mm)")
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("kPa", fontsize=8)
    fig.suptitle(f"{variant.name} contact-face pressure maps under tilt/curvature screens", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "contact_patch_pressure_maps_final.png")
    plt.close(fig)


def write_report(
    summary_rows: list[dict],
    geometry: dict[str, dict[str, float]],
    robust_rows: list[dict],
    patch_rows: list[dict],
    combined_rows: list[dict],
) -> None:
    winner = summary_rows[0]
    robust_winner = robust_rows[0]
    patch_winner = patch_rows[0]
    final_winner = combined_rows[0]
    lines = [
        "# ThinkLabs Passive Mount Simulation Report",
        "",
        "## What Was Simulated",
        "",
        "This local screen models the mount as a 1-DOF passive spring-damper carrier pressing a ThinkLabs One contact face into Kelvin-Voigt tissue, plus a 2D static contact-patch screen for tilted/off-center skin contact. It is intended to reject bad mechanical choices before cloud Isaac Sim testing.",
        "",
        "Scenarios: soft, medium, firm, and bony-prominence tissue stiffness at 15, 30, and 60 mm/s approach speeds. Target steady contact force is 4 N; forces above 8 N are treated as unsafe.",
        "",
        "## Variant Ranking",
        "",
        "| Rank | Variant | Score | Mean force error (N) | Peak force (N) | Worst travel (%) | Contact below printed parts (mm) |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(summary_rows, start=1):
        lines.append(
            f"| {idx} | `{row['variant']}` | {row['score']} | {row['mean_force_error_n']} | "
            f"{row['peak_force_n']} | {row['worst_travel_used_pct']} | {row['contact_below_lowest_printed_mm']} |"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"The deterministic screen winner is `{winner['variant']}`. The robustness sweep winner is `{robust_winner['variant']}`. The 2D contact-patch winner is `{patch_winner['variant']}`. Use `{final_winner['variant']}` as the next printable CAD candidate because it has the best combined score across deterministic force response, tolerance/off-axis robustness, and tilted contact-patch behavior unless Isaac Sim later contradicts it.",
            "",
            "## Combined Recommendation Score",
            "",
            "| Rank | Variant | Combined score | Deterministic | Robust | 2D patch | Safety violation | Travel violation | Max clearance violation |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for idx, row in enumerate(combined_rows, start=1):
        lines.append(
            f"| {idx} | `{row['variant']}` | {row['combined_score']} | {row['deterministic_score']} | "
            f"{row['robust_score']} | {row['patch_score']} | "
            f"{100.0 * float(row['safety_violation_rate']):.1f}% | "
            f"{100.0 * float(row['travel_violation_rate']):.1f}% | "
            f"{100.0 * float(row['max_clearance_violation_rate']):.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Robustness Sweep",
            "",
            f"Each variant was sampled {ROBUSTNESS_SAMPLES_PER_VARIANT} times with spring tolerance, damping tolerance, mass variation, tissue stiffness variation, robot overdrive error, tilt, off-center contact, guide friction, and print-height error.",
            "",
            "| Rank | Variant | Robust score | P95 peak force (N) | P99 peak force (N) | Mean force error (N) | P95 travel (%) | Safety violation | Binding risk | Seal risk |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for idx, row in enumerate(robust_rows, start=1):
        lines.append(
            f"| {idx} | `{row['variant']}` | {row['robust_score']} | {row['p95_peak_force_n']} | "
            f"{row['p99_peak_force_n']} | {row['mean_force_error_n']} | {row['p95_travel_used_pct']} | "
            f"{100.0 * float(row['safety_violation_rate']):.1f}% | "
            f"{100.0 * float(row['binding_risk_rate']):.1f}% | "
            f"{100.0 * float(row['acoustic_seal_risk_rate']):.1f}% |"
        )
    lines.extend(
        [
            "",
            "## 2D Contact Patch Sweep",
            "",
            f"Each variant was sampled {CONTACT_PATCH_SAMPLES_PER_VARIANT} times over tilted contact, off-axis load, chest curvature, rib-like protrusions, tissue stiffness variation, spring tolerance, command error, and print-height error. The circular ThinkLabs contact face is discretized at {CONTACT_GRID_STEP_MM} mm spacing.",
            "",
            "| Rank | Variant | Patch score | Mean force error (N) | P95 pressure (kPa) | P05 contact area (%) | P95 centroid offset (mm) | P95 moment (Nmm) | Clearance violation | Seal risk |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for idx, row in enumerate(patch_rows, start=1):
        lines.append(
            f"| {idx} | `{row['variant']}` | {row['patch_score']} | {row['mean_force_error_n']} | "
            f"{row['p95_peak_pressure_kpa']} | {row['p05_contact_area_pct']} | "
            f"{row['p95_centroid_offset_mm']} | {row['p95_resultant_moment_nmm']} | "
            f"{100.0 * float(row['clearance_violation_rate']):.1f}% | "
            f"{100.0 * float(row['seal_risk_rate']):.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Important Limitations",
            "",
            "- This is a local numerical mechanics screen, not a full Isaac Sim rigid-body/contact run.",
            "- The robustness and 2D patch sweeps add off-axis contact, guide-friction, tilt, pressure hot-spot, and tolerance screens, but still do not model detailed lateral rubbing, acoustic transfer quality, cable drag, soft-body deformation, or the exact OpenArm flange.",
            "- The repo's Isaac Sim stack is cloud/GPU based. A reproducible handoff scene is available at `auscultation/mechanical_sensor/sim/isaac_mount_contact_scene.py`; run it on the Isaac machine with `/isaac-sim/python.sh` and the generated `design_metadata.json`.",
            "- The handoff scene is an Isaac visualization/logging bridge driven by this passive contact model. The later high-fidelity step is still to import the final STEP/STL proxy into `sim/` and attach rigid-body/contact sensors on the configured Isaac instance.",
            "- Current cloud-simulator status from this workstation: local macOS has no `isaacsim` Python package. The repo's `sim/README.md` points to a Vast.ai Isaac Lab workflow, and the required Vast SSH/public-IP/UDP-port details are not available in this workspace. Do not treat this report as completed Isaac Sim validation.",
            "",
            "## Generated Evidence",
            "",
            "- `auscultation/mechanical_sensor/sim/isaac_mount_contact_scene.py`",
            "- `summary_scores.csv`",
            "- `all_trials.csv`",
            "- `force_profiles_medium_30mms.png`",
            "- `variant_score_summary.png`",
            "- `robustness_trials.csv`",
            "- `robustness_summary.csv`",
            "- `robustness_summary.png`",
            "- `contact_patch_trials.csv`",
            "- `contact_patch_summary.csv`",
            "- `contact_patch_summary.png`",
            "- `contact_patch_pressure_maps_final.png`",
            "- `combined_recommendation.csv`",
            "- `simulation_results.json`",
            "",
            "## Geometry Clearance Checks",
            "",
        ]
    )
    for variant, values in geometry.items():
        lines.append(f"- `{variant}`: {values}")
    lines.append("")
    (OUT_DIR / "SIM_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: list[SimResult] = []
    medium_series: dict[str, dict[str, np.ndarray]] = {}
    geometry = {variant.name: geometry_clearance_mm(variant) for variant in VARIANTS}

    for variant in VARIANTS:
        for tissue in TISSUES:
            for speed in APPROACH_SPEEDS_MM_S:
                result, series = run_response(variant, tissue, speed)
                all_results.append(result)
                if tissue.name == "medium_tissue" and math.isclose(speed, 30.0):
                    medium_series[variant.name] = series

    trial_rows = [asdict(r) for r in all_results]
    summary_rows = summarize_results(all_results, geometry)
    robustness_trials, robustness_summary = run_robustness_sweep(geometry)
    contact_patch_trials, contact_patch_summary = run_contact_patch_sweep(geometry)
    combined_rows = combined_recommendation(summary_rows, robustness_summary, contact_patch_summary)
    winner_name = str(summary_rows[0]["variant"])
    robust_winner_name = str(robustness_summary[0]["variant"])
    patch_winner_name = str(contact_patch_summary[0]["variant"])
    final_winner_name = str(combined_rows[0]["variant"])
    winner = next(v for v in VARIANTS if v.name == winner_name)
    robust_winner = next(v for v in VARIANTS if v.name == robust_winner_name)
    patch_winner = next(v for v in VARIANTS if v.name == patch_winner_name)
    final_winner = next(v for v in VARIANTS if v.name == final_winner_name)

    write_csv(OUT_DIR / "all_trials.csv", trial_rows)
    write_csv(OUT_DIR / "summary_scores.csv", summary_rows)
    write_csv(OUT_DIR / "robustness_trials.csv", [asdict(r) for r in robustness_trials])
    write_csv(OUT_DIR / "robustness_summary.csv", robustness_summary)
    write_csv(OUT_DIR / "contact_patch_trials.csv", [asdict(r) for r in contact_patch_trials])
    write_csv(OUT_DIR / "contact_patch_summary.csv", contact_patch_summary)
    write_csv(OUT_DIR / "combined_recommendation.csv", combined_rows)
    plot_force_profiles(medium_series)
    plot_summary(summary_rows)
    plot_robustness_summary(robustness_summary)
    plot_contact_patch_summary(contact_patch_summary)
    plot_final_pressure_maps(geometry, final_winner_name)
    write_report(summary_rows, geometry, robustness_summary, contact_patch_summary, combined_rows)

    payload = {
        "target_force_n": TARGET_FORCE_N,
        "max_safe_force_n": MAX_SAFE_FORCE_N,
        "winner": winner_name,
        "winner_params": asdict(winner),
        "robustness_winner": robust_winner_name,
        "robustness_winner_params": asdict(robust_winner),
        "contact_patch_winner": patch_winner_name,
        "contact_patch_winner_params": asdict(patch_winner),
        "final_recommendation": final_winner_name,
        "final_recommendation_params": asdict(final_winner),
        "summary": summary_rows,
        "robustness_summary": robustness_summary,
        "contact_patch_summary": contact_patch_summary,
        "combined_recommendation": combined_rows,
        "geometry": geometry,
        "tissues": [asdict(t) for t in TISSUES],
        "approach_speeds_mm_s": APPROACH_SPEEDS_MM_S,
        "robustness_samples_per_variant": ROBUSTNESS_SAMPLES_PER_VARIANT,
        "robustness_seed": ROBUSTNESS_SEED,
        "contact_patch_samples_per_variant": CONTACT_PATCH_SAMPLES_PER_VARIANT,
        "contact_patch_seed": CONTACT_PATCH_SEED,
        "contact_face_radius_mm": CONTACT_FACE_RADIUS_MM,
    }
    (OUT_DIR / "simulation_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "winner": winner_name,
                "robustness_winner": robust_winner_name,
                "contact_patch_winner": patch_winner_name,
                "final_recommendation": final_winner_name,
                "summary": summary_rows[:5],
                "robustness_summary": robustness_summary[:5],
                "contact_patch_summary": contact_patch_summary[:5],
                "combined_recommendation": combined_rows[:5],
            },
            indent=2,
        )
    )
    print(f"Wrote simulation outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
