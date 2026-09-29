import importlib.util
from types import SimpleNamespace
import numpy as np
import pytest


def controller_class():
    assert importlib.util.find_spec('sim.auscultation_skill_v1.controller') is not None, 'new controller is missing'
    from sim.auscultation_skill_v1.controller import ExpertController
    return ExpertController


def test_zero_policy_is_not_replaced_by_expert_force_control():
    cls = controller_class()
    controller = object.__new__(cls)
    controller.action_provider = lambda controller, state: 0.
    controller.speed_scale = 1.
    controller.executor_config = {'normal_speed_limit_m_s': .004, 'normal_inward_speed_limit_m_s': .002}
    controller.manager = SimpleNamespace(state=SimpleNamespace(phase=SimpleNamespace(value='CONTACT')))
    controller.config = SimpleNamespace(target_force_N=4.)
    assert controller.desired_normal_velocity({'hall_N': 0., 'compression_m': 0.}) == 0.


def test_teacher_approach_force_correction_and_unload_use_asymmetric_bounds():
    cls = controller_class()
    controller = object.__new__(cls)
    controller.action_provider = None
    controller.speed_scale = .75
    controller.executor_config = {'normal_speed_limit_m_s': .004, 'normal_inward_speed_limit_m_s': .002}
    controller.manager = SimpleNamespace(state=SimpleNamespace(phase=SimpleNamespace(value='CONTACT')))
    controller.config = SimpleNamespace(target_force_N=4.)
    assert controller.desired_normal_velocity({'hall_N': 0., 'compression_m': 0.}) == -.0015
    assert controller.desired_normal_velocity({'hall_N': 1., 'compression_m': .0025}) == -.0015
    assert controller.desired_normal_velocity({'hall_N': 6., 'compression_m': .0025}) == .0015
    controller.manager.state.phase.value = 'RETRACT'
    assert controller.desired_normal_velocity({'hall_N': 4., 'compression_m': .0025}) == .004


def test_actor_observation_uses_conditioned_measurement_but_current_executor():
    cls = controller_class()
    controller = object.__new__(cls)
    controller.sensor_condition = SimpleNamespace(apply=lambda state, target: ({**state, 'hall_N': 3.}, {'observation_time_s': 1.}))
    controller.target = SimpleNamespace(name='L4')
    state = {'hall_N': 4., 'time_s': 1.05}
    observed, meta = controller.observe_actor(state)
    assert observed['hall_N'] == 3. and state['hall_N'] == 4.
    assert meta['observation_time_s'] == 1.


def test_contact_preflight_records_fixed_domain_and_failure_without_retry(monkeypatch):
    from sim.auscultation_skill_v1.curve import FixedNormalCurve
    cls = controller_class(); controller = object.__new__(cls)
    controller._point_plan = {'loaded_q': [0.] * 7, 'loaded_compression_m': .002}
    pose = np.eye(4); pose[2, 3] = .003
    controller.kin = SimpleNamespace(tcp=lambda q: pose)
    controller.target = SimpleNamespace(name='test', position_base=np.zeros(3))
    controller.normal_base = controller.normal = np.array([0., 0., 1.])
    controller.rotation_base = np.eye(3)
    controller.config = SimpleNamespace(target_force_N=4.)
    controller.executor_config = dict(nominal_compression_reserve_m=.0005, curve_spacing_m=.00025,
        curve_position_tolerance_m=2e-5, curve_angle_tolerance_rad=2e-4, contact_clearance_m=.0003)
    controller.env = SimpleNamespace(base=np.eye(4), rows=[])
    controller.q = np.zeros(7); controller.pre_gap = .025
    controller.posture_limits = None; controller.contact_curves = []
    calls = []
    def fail(*args, **kwargs):
        calls.append(args[7])
        raise RuntimeError('deliberate preflight failure')
    monkeypatch.setattr(FixedNormalCurve, 'build', fail)
    with pytest.raises(RuntimeError, match='deliberate preflight failure'):
        controller._contact_cycle()
    expected = .005 - ((4.-1.5-.067155*9.81)/1000+.0005)
    assert calls == [expected]
    record = controller.contact_curves[0]
    assert record['nominal_contact_domain']['minimum_gap_m'] == expected
    assert record['motion_complete'] is False
    assert record['preflight_passed'] is False
    assert record['end_physics_row_exclusive'] == 0
    assert controller._contact_active is False


def test_outcome_describes_actual_contact_policy_not_inherited_speeds(monkeypatch):
    from sim.auscultation_skill_v1.controller import ParentController
    monkeypatch.setattr(ParentController, 'run', lambda self, points: {
        'variation': {'joint_command_speed_limit_rad_s': .1875,
                      'free_approach_far_speed_m_s': .009, 'retraction_speed_m_s': .0075,
                      'fault_retraction_speed_m_s': .00375},
        'candidate_scope': 'unchanged contact controller',
        'candidate_timing': 'free route spline'})
    controller = object.__new__(controller_class())
    controller.executor_config = {'normal_speed_limit_m_s': .004, 'normal_inward_speed_limit_m_s': .002}
    controller.speed_scale = .75
    controller.contact_curves = []
    controller.condition = None
    controller.action_provider = None
    controller.command_collision_samples = controller.observed_collision_samples = 0
    outcome = controller.run([])
    assert outcome['variation']['free_approach_far_speed_m_s'] == .0015
    assert outcome['variation']['retraction_speed_m_s'] == .004
    assert outcome['variation']['joint_command_speed_limit_rad_s'] == .1875
    assert 'fault_retraction_speed_m_s' not in outcome['variation']
    assert outcome['actual_contact_policy']['expert_force_deadband_N'] == .02
    assert outcome['actual_contact_policy']['desired_normal_velocity_bounds_m_s'] == [-.0015, .004]
    assert outcome['actual_contact_policy']['teacher_force_correction_bounds_m_s'] == [-.0015, .0015]
    assert outcome['actual_contact_policy']['action_scaling'] == 'inward_scaled_outward_fixed_v1'
    assert 'unchanged contact controller' not in outcome['candidate_scope']
