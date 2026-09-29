"""Bounded posture-only preview of velocity hints on independent CPU clones.

The original position reference and 50 ms interval never change. This helper
does not advance the live trajectory, physics or sensors. It does not certify
collision clearance, measured posture, contact or exact waypoint arrival.
Every actually dispatched tick must still pass the original _block/_tick chain.
"""
import numpy as np

from sim.auscultation_pi05_v6.execution import JointTrajectory

FACTORS = (1., .5, .25, .125, 0.)
LIMITS = dict(dt=1/240, velocity=.175, acceleration=.5, jerk=7.99999)
TICKS = 12


class MotionPreviewRejected(ValueError):
    """Fail-closed refusal with the complete small trial receipt."""
    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = receipt


def _vector(value, name):
    array = np.asarray(value)
    if array.shape != (7,) or array.dtype.kind not in 'fiu' or not np.isfinite(array).all():
        raise ValueError(name+' must be a finite numeric seven-vector')
    return array.astype(np.float64,copy=True)


def choose_velocity_hint(trajectory, position, velocity, inspect_command,
                         trajectory_factory=JointTrajectory, selection='first_feasible'):
    """Try fixed hint factors in order, accepting only a complete clear preview.

    The inspector receives a copy of each finite generated joint command and
    must return a dict containing a boolean ``clear``. Native validation/step
    failures reject that candidate. No candidate passing all 12 ticks raises
    MotionPreviewRejected. Rejection step 0 means no preview command was made;
    generated commands use one-based steps 1..12.
    """
    receipt = dict(selected_factor=None,selected_velocity=None,trials=[],accepted=False,
        scope='12-tick command posture preview only; actual _block/_tick checks remain required',
        factors=list(FACTORS),ticks_per_candidate=TICKS,label_offset_s=.05,limits=dict(LIMITS),
        collision_checked=False,physics_executed=False,observations_acquired=False,
        closed_loop_evaluated=False,m0_passed=False,
        native_trajectory_preview=trajectory_factory is JointTrajectory,selection=selection)
    try:
        if selection not in ('first_feasible','nearest_reference'):
            raise ValueError('Unknown fixed candidate-selection rule')
        if not callable(inspect_command) or not callable(trajectory_factory):
            raise ValueError('Explicit posture inspector and trajectory factory are required')
        target = _vector(position,'position'); hint = _vector(velocity,'velocity')
        if np.max(np.abs(hint)) > LIMITS['velocity']:
            raise ValueError('Input velocity exceeds 0.175 rad/s; no limit clipping')
        if not np.isfinite(trajectory.dt) or trajectory.dt != LIMITS['dt']:
            raise ValueError('Source trajectory must retain the exact 240 Hz clock')
        current = {name:_vector(getattr(trajectory.input,'current_'+name),'current_'+name)
                   for name in ('position','velocity','acceleration')}
        for name in ('velocity','acceleration','jerk'):
            source = _vector(getattr(trajectory.input,'max_'+name),'max_'+name)
            if not np.array_equal(source,np.full(7,LIMITS[name])):
                raise ValueError('Source trajectory '+name+' limits differ from the original contract')
        receipt.update(target_position=target.tolist(),requested_velocity=hint.tolist(),
                       source_command_state={k:v.tolist() for k,v in current.items()})
    except Exception as error:
        receipt['failure'] = dict(stage='input_validation',reason=type(error).__name__+': '+str(error))
        raise MotionPreviewRejected('Invalid posture preview input: '+str(error),receipt) from error

    for factor in FACTORS:
        candidate = hint*factor
        trial = dict(factor=factor,target_velocity=candidate.tolist(),preview_commands=[],
                     first_rejection=None,accepted=False)
        receipt['trials'].append(trial)
        stage = 'clone_construction'; step = 0; command = None
        try:
            clone = trajectory_factory(current['position'].copy(),**LIMITS)
            if (clone is trajectory or clone.input is trajectory.input
                    or (getattr(trajectory,'generator',None) is not None
                        and clone.generator is trajectory.generator)
                    or (getattr(trajectory,'output',None) is not None
                        and clone.output is trajectory.output)):
                raise ValueError('Preview factory aliased live trajectory state')
            # New native generator; preserve the live command state rather than
            # replacing it with measured q or restarting its derivatives at zero.
            for name,value in current.items():
                setattr(clone.input,'current_'+name,value.tolist())
            clone.velocity=current['velocity'].copy()
            clone.acceleration=current['acceleration'].copy()
            stage = 'target_assignment'
            clone.set_target(target.copy())
            clone.input.target_velocity = candidate.tolist()
            clone.input.target_acceleration = [0.]*7
            stage = 'native_input_validation'
            valid = clone.generator.validate_input(clone.input,
                check_current_state_within_limits=False,check_target_state_within_limits=True)
            if not isinstance(valid,(bool,np.bool_)) or not bool(valid):
                raise ValueError('Native Ruckig input validation rejected candidate')
            for step in range(1,TICKS+1):
                command = None; stage = 'native_step'
                value = clone.step()
                stage = 'command_validation'; command = _vector(value,'preview command')
                trial['preview_commands'].append(command.tolist())
                stage = 'posture_inspection'
                inspection = inspect_command(command.copy())
                if not isinstance(inspection,dict) or not isinstance(inspection.get('clear'),(bool,np.bool_)):
                    raise ValueError('Posture inspector must return an explicit boolean clear')
                if not bool(inspection['clear']):
                    reason = inspection.get('reason',inspection.get('reasons',
                        inspection.get('posture_violations','posture rejected')))
                    trial['first_rejection'] = dict(step=step,stage='posture_inspection',
                        reason=str(reason),command=command.tolist())
                    break
            else:
                trial['accepted'] = True
                trial['endpoint_joint_mse_rad2'] = float(np.mean((command-target)**2))
                if selection == 'first_feasible':
                    receipt.update(accepted=True,selected_factor=factor,selected_velocity=candidate.tolist())
                    return candidate.copy(),receipt
        except Exception as error:
            trial['first_rejection'] = dict(step=step,stage=stage,
                reason=type(error).__name__+': '+str(error),
                command=None if command is None else command.tolist())
    accepted = [trial for trial in receipt['trials'] if trial['accepted']]
    if accepted:
        chosen = min(accepted,key=lambda trial:(trial['endpoint_joint_mse_rad2'],trial['factor']))
        receipt.update(accepted=True,selected_factor=chosen['factor'],selected_velocity=chosen['target_velocity'],
            selected_endpoint_joint_mse_rad2=chosen['endpoint_joint_mse_rad2'],
            zero_hint_endpoint_joint_mse_rad2=next((trial['endpoint_joint_mse_rad2'] for trial in accepted if trial['factor']==0.),None),
            selection_scope='minimum joint endpoint MSE to unchanged model reference at 50ms among fully posture-valid candidates; lower factor breaks exact ties')
        return np.asarray(chosen['target_velocity'],float),receipt
    receipt['failure'] = dict(stage='candidate_selection',reason='No candidate passed all 12 posture preview ticks')
    raise MotionPreviewRejected(receipt['failure']['reason'],receipt)
