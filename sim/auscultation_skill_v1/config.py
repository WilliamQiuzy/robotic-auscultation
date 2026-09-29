"""Frozen V1 simulation engineering goals; no physical/clinical acceptance."""
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
TARGETS = tuple(side + str(level) for level in range(1, 6) for side in ('R', 'L'))
PHYSICS_DT_S = 1 / 240.
CONTROL_DT_S = .05
SELECTED_PLANS = PROJECT / 'sim/upper_pair_posture_20260924/results/candidate_b_plans'
BASELINE = PROJECT / 'sim/upper_pair_posture_20260924/results/physics_selected_all10'
RUNTIME = PROJECT / 'sim/auscultation_expert_layout_v1'
CAMERA = PROJECT / 'sim/camera_framing_20260923/recommended_camera.json'


@dataclass(frozen=True)
class ExecutorConfig:
    normal_speed_limit_m_s: float = .004
    normal_inward_speed_limit_m_s: float = .002
    normal_acceleration_limit_m_s2: float = .02
    normal_jerk_limit_m_s3: float = .2
    joint_velocity_limit_rad_s: float = .25
    joint_acceleration_limit_rad_s2: float = .5
    joint_jerk_limit_rad_s3: float = 8.
    curve_spacing_m: float = .00025
    curve_position_tolerance_m: float = 2e-5
    curve_angle_tolerance_rad: float = 2e-4
    contact_clearance_m: float = .0003
    max_inward_braking_m: float = 1e-6
    # Geometry operating allowance above static Hall-target compression (4 N).
    # At the nominal 1000 N/m stiffness this reaches the existing 4.5 N band.
    # It is not a guarantee for delayed/noisy feedback or every actor action.
    nominal_compression_reserve_m: float = .0005


DEFAULT_EXECUTOR_CONFIG = ExecutorConfig()
ENGINEERING_GATES = MappingProxyType({
    'joint_step_rad': 0.08726646259971647,
    'joint_velocity_rad_s': .5,
    'joint_acceleration_rad_s2': 2.,
    'joint_jerk_rad_s3': 20.,
    'free_velocity_rad_s': .25,
    'free_acceleration_rad_s2': 1.,
    'free_jerk_rad_s3': 1.5,
    'loaded_force_N': .2,
    'loaded_drag_max_event_m': .0005,
    'loaded_drag_max_speed_m_s': .005,
    'loaded_point_change_max_events': 0,
    'loaded_free_transfer_max_events': 0,
    'measured_raw_jerk_target_rad_s3': 20.,
    'filter_window_s': .1,
    'minimum_nominal_repetitions': 3,
})


def frozen_configuration():
    """A new serialization object; callers cannot alter the default by mutation."""
    return {'schema': 'auscultation_skill_configuration_v1',
            'action_scaling': 'inward_scaled_outward_fixed_v1',
            'physics_dt_s': PHYSICS_DT_S, 'control_dt_s': CONTROL_DT_S,
            'executor': asdict(DEFAULT_EXECUTOR_CONFIG),
            'engineering_gates': dict(ENGINEERING_GATES),
            'scope': 'simulation control imitation research only; external acceptance remains pending',
            'measured_raw_target_policy': 'visible research limitation; never relabeled as achieved'}


def validate_config(configuration):
    if configuration != frozen_configuration():
        raise ValueError('frozen V1 configuration changed; create a new version and requalify')
    return configuration
