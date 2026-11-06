"""Real Ruckig, actual dispatchers, synthetic instantaneous world feedback.

This is a CPU software equivalence test, never contact/physics acceptance.
Linux evidence retains full q/v/a arrays; no raw recordings are copied to Mac.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_ausc_given_contact_pose import reference, metadata


def harness(tmp_path, learned, route, *, flip, stop_tick, target_cells=None, shared_reference=None):
    tmp_path.mkdir(parents=True,exist_ok=True)
    from sim.ausc_policy.m2_given_contact_controller import GivenContactExpert, GivenContactACT
    from sim.ausc_policy.m2_act_controller import LIMITS
    from sim.ausc_policy import m2_contract as contract
    from sim.auscultation_pi05_v6.execution import JointTrajectory
    cls = GivenContactACT if learned else GivenContactExpert
    c = cls.__new__(cls)
    c.reference = reference(tmp_path) if shared_reference is None else shared_reference
    c.metadata = metadata(c.reference)
    c.metadata.update(episode_id='parity',target_id='L1')
    c.episode_id='parity'; c.target=SimpleNamespace(name='L1'); c.sequence_index=0
    c.q=np.zeros(7); c.control_owner='PREPARATION'; c.actor_phase=None
    c.prefix=None; c._history=None; c._l2_started_at=None
    c._alignment_audit=None; c.l2_handoffs=[]; c.l2_queries=[]; c.l2_actions=[]
    c.trace=[]; c.raw_ticks=[]; c.roles=[]
    c.env=SimpleNamespace(rows=[],base=np.eye(4))
    c.env.state=lambda: dict(time_s=len(c.env.rows)/240.,q=c.q.copy())
    c.trajectory_factory=JointTrajectory
    c.kin=SimpleNamespace(lower_limits=np.full(7,-3.),upper_limits=np.full(7,3.))
    c.manager=SimpleNamespace(state=SimpleNamespace(phase=__import__(
        'sim.auscultation_expert_layout_v1.phase_manager',fromlist=['Phase']).Phase.APPROACH,failure_reason=None))
    c._validate_target_reference=lambda row: None
    c._canonical=lambda row: (np.asarray([np.zeros(7),*route]),0)
    def append(command, role):
        c.q=np.asarray(command).copy(); c.env.rows.append(dict(q_command=c.q.copy()))
        trajectory=getattr(c,'trajectory',None)
        v=np.zeros(7) if trajectory is None else trajectory.velocity.copy()
        a=np.zeros(7) if trajectory is None else trajectory.acceleration.copy()
        c.trace.append(np.r_[c.q,v,a]); c.roles.append(role)
        c.raw_ticks.append(deepcopy(getattr(c,'_current_intent_audit',None)))
    c._block=lambda commands,role,**kwargs: [append(q,role) for q in commands]
    c._hold=lambda seconds,role: [append(c.q,role) for _ in range(round(seconds*240))]
    def provider(controller,state):
        phase='APPROACH' if flip and len(c.env.rows)>=322 else 'FREE'
        # Both actual dispatcher paths consume this same explicit causal input.
        return dict(episode_id='parity',target_id='L1',revision=0,phase=phase,
            observed_at_s=state['time_s'],measured_q=state['q'].copy(),
            estimated_target_base=c.reference.pose_base,confidence=1.,source_id='given-contact-'+c.reference.sha256)
    c.estimate_provider=provider
    c._handoff_measurements=lambda: dict(passed=len(c.env.rows)>=stop_tick,
        physics_row=len(c.env.rows),time_s=len(c.env.rows)/240.,command_q=c.q.tolist())
    if learned:
        def predictor(q,pose,confidence,**kwargs):
            cell=len(c.env.rows)//12
            values=np.asarray([target_cells.get(cell+k,target_cells[max(target_cells)]) for k in range(20)])
            return contract.encode_joint_chunk(values,q,arm='block_relative_joint',metadata=c.metadata)
        c.predictor=predictor
    return c


@pytest.mark.parametrize('case', ['single_waypoint','multi_waypoint','prefix_context_stop'])
def test_actual_dispatchers_raw_target_parity(tmp_path,case):
    pytest.importorskip('ruckig')
    direction=np.array([1.,0.,0.,0.,0.,0.,0.])
    route=[direction*.4] if case=='single_waypoint' else [direction*.02,direction*.4]
    flip=case=='prefix_context_stop'
    teacher=harness(tmp_path/'teacher',False,route,flip=flip,stop_tick=901)
    teacher._move_to_prepose({})
    cells={event['physics_row_start']//12:np.asarray(event['target_q'])
           for event in teacher._intent_dispatcher.events if not event['offgrid_stop']}
    student=harness(tmp_path/'student',True,route,flip=flip,stop_tick=901,target_cells=cells,
                    shared_reference=teacher.reference)
    student._move_to_prepose({})
    a,b=np.asarray(teacher.trace),np.asarray(student.trace)
    assert a.shape==b.shape
    delta=float(np.max(abs(a[:,:7]-b[:,:7])))
    assert delta<1e-12
    assert np.max(abs(a[:,7:]-b[:,7:]))<1e-10
    assert teacher.roles==student.roles
    assert teacher._intent_dispatcher.events[-1]['offgrid_stop'] is True
    assert teacher._intent_dispatcher.events[-1]['physics_row_start']%12 != 0
    assert len(student.l2_queries)>=2  # real E10 prefix replacement
    waits=[e for e in teacher._intent_dispatcher.events if e['execution_role']=='given_l2_history_wait']
    assert len(waits)>=(2 if flip else 1)
    applied={cell:a[min((cell+1)*12-1,len(a)-1),:7] for cell in cells}
    negative=harness(tmp_path/'negative',True,route,flip=flip,stop_tick=901,target_cells=applied,
                     shared_reference=teacher.reference)
    negative._move_to_prepose({})
    negative_trace=np.asarray(negative.trace)
    n=min(len(a),len(negative_trace))
    negative_delta=float(np.max(abs(a[:n,:7]-negative_trace[:n,:7])))
    assert negative_delta>.001
    output=os.environ.get('AUSC_PARITY_OUTPUT')
    if output:
        import ruckig,sys
        from sim.ausc_policy import m2_given_contact_controller,m2_given_contact_intent,m2_act_controller,m2_stage
        from sim.auscultation_pi05_v6 import execution
        root=Path(output); root.mkdir(parents=True,exist_ok=True)
        arrays=root/(case+'.npz')
        np.savez_compressed(arrays,teacher=a,raw_intent_student=b,applied_label_negative=negative_trace,
                            times=np.arange(1,len(a)+1)/240.)
        result=dict(schema='m2_given_actual_dispatcher_cpu_parity_v1',case=case,
            ruckig_version=ruckig.__version__,python=sys.version,executable=sys.executable,
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu_work=False,
            native_physics=False,physical_qualification=False,training_eligible=False,
            feedback='explicit CPU instantaneous commanded-position fixture',
            native_ticks=len(a),raw_intent_max_q_difference_rad=delta,
            raw_intent_max_derivative_difference=float(np.max(abs(a[:,7:]-b[:,7:]))),
            applied_label_negative_max_q_difference_rad=negative_delta,
            queries=len(student.l2_queries),history_wait_cells=len(waits),
            immutable_reference_sha256=teacher.reference.sha256,
            student_reference_sha256=student.reference.sha256,
            test_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            controlled_coarse_stop='CPU fixture reports first coarse entry at native row 901; not a measured physical gate',
            teacher_setters=teacher._intent_dispatcher.events,
            student_setters=student._intent_dispatcher.events,
            arrays=dict(file=str(arrays),bytes=arrays.stat().st_size,
                sha256=hashlib.sha256(arrays.read_bytes()).hexdigest()),
            source_hashes={str(Path(module.__file__).resolve()):hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                for module in (m2_given_contact_controller,m2_given_contact_intent,m2_act_controller,m2_stage,execution)})
        (root/(case+'.json')).write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
