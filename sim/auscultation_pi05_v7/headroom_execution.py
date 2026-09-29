"""Explicit runtime adaptation; checkpoint execution binding remains jerk 8.

Only the internal Ruckig jerk bound changes. The inherited whole-stream
comparison remains 8 + 1e-8, and all physical acceptance limits are unchanged.
"""
from pathlib import Path
from sim.auscultation_pi05_v6.execution import JointTrajectory
from sim.auscultation_pi05_v6.dynamics_contract import execution_contract, sha256_file, sha256_json

VERSION = 'ruckig_internal_jerk_headroom_v1'
RUNTIME_JERK = 7.99999
MODULES = ('headroom_execution.py', 'headroom_learned_controller.py', 'run_headroom_isaac.py',
           'audit_headroom_learned_actions.py', 'assess_headroom_trials.py')


def runtime_execution_contract():
    result = execution_contract()
    result['jerk_rad_s3'] = [RUNTIME_JERK] * 7
    return result


def runtime_adaptation_contract():
    trained = execution_contract()
    runtime = runtime_execution_contract()
    return dict(schema='wam_runtime_execution_adaptation_v1', version=VERSION,
        trained_execution_contract=trained, trained_execution_sha256=sha256_json(trained),
        runtime_execution_contract=runtime, runtime_execution_sha256=sha256_json(runtime),
        only_execution_descriptor_change='jerk_rad_s3: seven 8.0 values to seven 7.99999 values',
        source_sha256={name:sha256_file(Path(__file__).with_name(name)) for name in MODULES},
        original_whole_stream_gate=dict(joint_velocity_rad_s=.175,
            joint_acceleration_rad_s2=dict(inactive=1.,active=.5),joint_jerk_rad_s3=8.,comparison_slack=1e-8),
        action_composition='absolute raw_actions[0]; no measured-q addition or clipping',
        trajectory_state='original continuous Ruckig q/v/a; same query clock; target velocity and acceleration zero',
        physical_safety_gates_unchanged=True, checkpoint_retrained_for_runtime=False,
        original_failed_trials_reclassified=False, scope='zero-home full learned task; explicit post-training execution adaptation')


def verify_contract_bindings(outcome, binding):
    """Never substitute a runtime SHA into the unchanged service/trained slot."""
    descriptor = runtime_adaptation_contract()
    if outcome.get('runtime_adaptation_contract') != descriptor:
        raise ValueError('Runtime adaptation descriptor/source mismatch')
    if outcome.get('runtime_adaptation_contract_sha256') != sha256_json(descriptor):
        raise ValueError('Runtime adaptation descriptor hash mismatch')
    for key in ('trained_execution_sha256','runtime_execution_sha256'):
        if outcome.get(key) != descriptor[key]: raise ValueError('Wrong '+key)
    if binding.get('execution_contract_sha256') != descriptor['trained_execution_sha256']:
        raise ValueError('Service must retain original trained execution binding')
    if outcome.get('checkpoint_retrained_for_runtime') is not False:
        raise ValueError('Runtime adaptation cannot claim retrained checkpoint')


class HeadroomJointTrajectory(JointTrajectory):
    def __init__(self,q,*,dt=1/240,velocity=.175,acceleration=.5,jerk=RUNTIME_JERK):
        if (dt,velocity,acceleration,jerk) != (1/240,.175,.5,RUNTIME_JERK):
            raise ValueError('Headroom execution parameters differ from declared runtime adaptation')
        super().__init__(q,dt=dt,velocity=velocity,acceleration=acceleration,jerk=jerk)
