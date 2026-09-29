"""CPU start-boundary checks; fake physics never certifies physical acquisition."""
import ast
import copy
import importlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as S
from unittest.mock import patch
import numpy as np
import pytest

Q=np.array([.12,-.06,.06,.12,-.04,.04,0.])


def module(name='start_learned_controller'):
    full='sim.auscultation_pi05_v7.'+name
    assert importlib.util.find_spec(full) is not None, 'New independent start runtime is required'
    return importlib.import_module(full)


def fixture(tmp_path):
    m=module();t=m.StartLearnedController.__new__(m.StartLearnedController)
    t.start_manifest={'target':'R4','requested_start_q_rad':Q.tolist(),'start_family':'train_yaw_positive_v1',
                      'parent_plan_sha256':'a'*64}
    t.start_manifest_sha256='b'*64;t.experiment_protocol_sha256='c'*64;t.episode_id='R4_start_learned_v1'
    t.start_receipt=m.new_receipt(t.start_manifest,'b'*64,'c'*64,t.episode_id)
    t.learned_rows=[];t.start_role=None;t.start_prepared=False;t.sequence_index=0;t.task_scope='full'
    t.q=np.zeros(7);t.joint_speed_limit=.175;t.actor_phase=None;t.target=S(name='R4')
    t.env=S(rows=[],out=tmp_path,last_contacts={})
    state={'time_s':0.,'q':t.q.copy(),'dq':np.zeros(7),'hall_N':0.,'compression_m':0.}
    t.env.state=lambda:state;t.sensor=S(rows=[],out=tmp_path)
    t.manager=S(state=S(phase=m.Phase.APPROACH));t.checker=object()
    canonical=np.array([np.zeros(7),Q+.4,Q+.5])
    t._canonical=lambda row:(canonical,1)
    t._tracked_metrics=lambda state:({},S(displacement_estimate_m=0.,spring_force_estimate_n=1.5),None)
    def capture(state,phase,target,row,**kwargs):
        raw=tmp_path/('raw_'+str(len(t.sensor.rows))+'.npz');np.savez(raw,rgb=np.zeros((2,2,3),np.uint8))
        t.sensor.rows.append(dict(index=len(t.sensor.rows),time_s=state['time_s'],target=target,phase=phase,
            q=state['q'].tolist(),dq=state['dq'].tolist(),physics_row=row,state=np.zeros(32).tolist(),
            raw_file=raw.name,raw_sha256=m.sha(raw),**kwargs))
    t.sensor.capture=capture
    t.events=[]
    def advance(destination,subphase,ticks):
        t.events.append(subphase)
        for _ in range(ticks):
            t.q=np.asarray(destination).copy();state.update(time_s=state['time_s']+1/240.,q=t.q.copy())
            t.env.rows.append(dict(q_command=t.q.tolist(),q=t.q.tolist(),dq=state['dq'].tolist(),
                time_s=state['time_s'],subphase=subphase,recovery=False,target='R4'))
            t._capture_observed_tick(state,subphase,False,False)
    t._joint_segment=lambda q,phase:advance(q,phase,12)
    t._hold=lambda seconds,phase:advance(t.q,phase,round(seconds*240))
    return t,state,canonical


def test_component_required():
    assert module().CLASS.endswith('.StartLearnedController')


@pytest.mark.parametrize('change',['home','position','speed','nan','shape'])
def test_start_measurement_gate_rejects_bad_states(change):
    m=module();state=dict(q=Q.copy(),dq=np.zeros(7),time_s=1.)
    if change=='home':state['q']=np.zeros(7)
    elif change=='position':state['q']=Q+.030001
    elif change=='speed':state['dq'][4]=.030001
    elif change=='nan':state['time_s']=float('nan')
    else:state['dq']=np.zeros(6)
    with pytest.raises((ValueError,m.EpisodeRejected)):m.checked_start_state(state,Q)


def test_actual_nonzero_q_dq_are_preserved():
    m=module();q=Q+.002;dq=np.linspace(-.01,.01,7)
    actual=m.checked_start_state(dict(q=q,dq=dq,time_s=1.),Q)
    np.testing.assert_array_equal(actual[0],q);np.testing.assert_array_equal(actual[1],dq)


def test_contact_return_and_later_points_are_exact_parent_behavior(tmp_path):
    m=module();t,_,_=fixture(tmp_path)
    for name in ('_contact_cycle','_return_home','desired_normal_velocity'):
        assert getattr(m.StartLearnedController,name) is getattr(m.LearnedController,name)
    t.sequence_index=1
    with patch.object(m.LearnedController,'_move_to_prepose',return_value='parent') as p:
        assert t._move_to_prepose({'next':True})=='parent'
        p.assert_called_once_with({'next':True})
    assert t.events==[]


def test_preparation_then_model_only_no_teacher_route_or_home_washout(tmp_path):
    m=module();t,state,canonical=fixture(tmp_path)
    calls=[]
    def infer(phase,active):
        calls.append((phase,active));t._bind_first_query(phase,active)
    t._infer_step=infer
    t._initialize_contact_supervisor=lambda:S(phase=m.Phase.CONTACT)
    with patch.object(m,'checked_edge',return_value={'clear':True}),patch.object(m,'JointTrajectory') as jt,patch.object(
            m.phase_contract,'precontact_phase',side_effect=['FREE','APPROACH']):
        actual=t._move_to_prepose({})
    assert calls==[('FREE',False)]
    assert t.events==['initial_home','settle','settle']
    np.testing.assert_array_equal(actual,canonical);np.testing.assert_array_equal(t.canonical_transfer,canonical)
    assert t.start_receipt['preparation_physics_range']==[0,12]
    assert t.start_receipt['settle_physics_range']==[12,252]
    assert t.start_receipt['observation_physics_range']==[252,264]
    assert t.start_receipt['learned_interval_start_physics_row']==264
    assert t.start_receipt['first_query']['physics_row_start']==264
    assert t.start_receipt['first_query']['observation_physics_row']==263
    assert t.start_receipt['first_query']['raw_sensor_sha256']==t.sensor.rows[-1]['raw_sha256']
    assert t.start_receipt['first_query']['measured_q_rad']==Q.tolist()
    assert t.start_receipt['first_query']['measured_dq_rad_s']==state['dq'].tolist()
    assert len(t.sensor.rows)==22 and all(f['eligible'] is False for f in t.sensor.rows)
    assert t.start_role is None and t.start_prepared is True
    jt.assert_called_once()


def prepared(t,state):
    t.start_prepared=True;t.q=Q.copy();state.update(time_s=1.,q=Q.copy())
    t.env.rows.append(dict(q=Q.tolist(),dq=state['dq'].tolist(),q_command=Q.tolist(),time_s=1.))
    t.sensor.capture(state,'APPROACH','R4',0,eligible=False,subphase='settle')
    t.start_receipt['learned_interval_start_physics_row']=1


def test_first_request_reuses_immutable_frame_and_delegates_absolute_policy(tmp_path):
    m=module();t,state,_=fixture(tmp_path);prepared(t,state)
    before=copy.deepcopy(t.sensor.rows)
    with patch.object(m.LearnedController,'_infer_step',return_value='original') as p:
        assert t._infer_step('FREE',active=False)=='original'
        p.assert_called_once_with('FREE',active=False)
    assert t.sensor.rows==before
    assert t.start_receipt['first_query']['acquired_observation_phase']=='APPROACH'
    assert t.start_receipt['first_query']['requested_policy_phase']=='FREE'


@pytest.mark.parametrize('change',['stale','changed_q','changed_dq','raw_tamper','wrong_phase','unprepared','boundary'])
def test_first_request_fails_before_model_on_unbound_observation(tmp_path,change):
    m=module();t,state,_=fixture(tmp_path);prepared(t,state);phase='FREE'
    if change=='stale':state['time_s']+=1/240.
    elif change=='changed_q':t.sensor.rows[-1]['q'][0]+=.001
    elif change=='changed_dq':t.sensor.rows[-1]['dq'][0]+=.001
    elif change=='raw_tamper':(tmp_path/t.sensor.rows[-1]['raw_file']).write_bytes(b'changed')
    elif change=='wrong_phase':phase='CONTACT'
    elif change=='unprepared':t.start_prepared=False
    else:t.start_receipt['learned_interval_start_physics_row']=0
    with patch.object(m.LearnedController,'_infer_step') as p:
        with pytest.raises((ValueError,m.EpisodeRejected)):t._infer_step(phase,active=False)
        p.assert_not_called()


def test_failed_preparation_retains_partial_receipt_and_class_identity(tmp_path):
    m=module();t,_,_=fixture(tmp_path)
    def failure(*args):raise m.EpisodeRejected('real preparation failure')
    t._joint_segment=failure
    with patch.object(m,'checked_edge',return_value={'clear':True}),pytest.raises(m.EpisodeRejected):
        t._move_to_prepose({})
    assert t.start_receipt['preparation_physics_range']==[0,0]
    result=t.finalize_start_receipt()
    assert result['complete'] is False and result['executed_controller_class']==m.CLASS
    assert result['first_query'] is None and result['qualification_passed'] is False


def test_run_outcome_is_new_class_not_frozen_v6(tmp_path):
    m=module();t,_,_=fixture(tmp_path)
    with patch.object(m.LearnedController,'run',return_value={'success':False,'executed_controller_class':'old'}):
        result=t.run([(S(name='R4'),{})])
    assert result['executed_controller_class']==m.CLASS
    assert result['controller_source']==m.SOURCE
    assert result['training_dataset_eligible'] is False


def test_entry_cli_and_snapshot_do_not_hide_new_runtime():
    r=module('run_start_isaac')
    tree=ast.parse(Path(r.__file__).read_text())
    top=[]
    for n in tree.body:
        if isinstance(n,ast.Import):top.extend(a.name for a in n.names)
        if isinstance(n,ast.ImportFrom):top.append(n.module or '')
    assert not any(x.startswith(('isaac','omni','pxr')) for x in top)
    p=r.argument_parser();options={x for a in p._actions for x in a.option_strings}
    assert {'--start-manifest','--start-manifest-sha256','--protocol','--protocol-sha256','--points'}<=options
    assert not {'--task-scope','--armature','--position-iterations','--speed-scale'}&options
    paths=r.runtime_source_paths()
    for name in ('start_learned_controller.py','run_start_isaac.py','start_collection.py'):
        assert paths['pi05_v7/'+name].is_file()
    assert paths['pi05_v6/learned_controller.py'].is_file()


@pytest.mark.parametrize('physical_load,adc_compression,adc_spring',[(.2,0.,1.5),(0.,.00012,4.),(0.,.00016,1.5)])
def test_released_gate_checks_both_actual_physics_and_decoded_adc(tmp_path,physical_load,adc_compression,adc_spring):
    m=module();t,state,_=fixture(tmp_path);state.update(q=Q.copy(),hall_N=physical_load)
    t._tracked_metrics=lambda state:({},S(displacement_estimate_m=adc_compression,spring_force_estimate_n=adc_spring),None)
    with pytest.raises(m.EpisodeRejected):t._require_released_start(state)


def test_new_runner_clean_import_defers_physics_until_kit():
    import subprocess
    import sys
    directory=str(Path(module('run_start_isaac').__file__).parent)
    code=("import sys; import sim.auscultation_pi05_v7 as package; "
          "package.__path__.insert(0,"+repr(directory)+"); "
          "import sim.auscultation_pi05_v7.run_start_isaac; "
          "assert not any(k.startswith(('pxr','omni.','isaacsim','sim.auscultation_pi05_v6.learned_controller')) for k in sys.modules)")
    subprocess.run([sys.executable,'-B','-c',code],check=True,timeout=15)


def test_cli_preserves_sequence_order_with_one_bound_initial_target():
    r=module('run_start_isaac')
    argv=['--source-project','source','--output','new','--annotation','a','--client-config','c',
          '--dynamics-contract','d','--episode-id','test','--start-manifest','s',
          '--start-manifest-sha256','a'*64,'--protocol','p','--protocol-sha256','b'*64,
          '--points','R4','L1','R1','R2','R3','L2','L3','L4','R5','L5']
    args=r.parse_args(argv)
    assert args.point=='R4' and args.points==argv[-10:]
    assert args.task_scope=='full' and args.speed_scale==.7 and args.armature==.02 and args.position_iterations==32
    with pytest.raises(SystemExit):r.parse_args(argv+['--task-scope','contact'])
    with pytest.raises(SystemExit):r.parse_args(argv[:-1]+['R4'])
