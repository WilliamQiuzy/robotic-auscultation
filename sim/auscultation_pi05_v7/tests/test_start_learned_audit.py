"""Independent synthetic ledger tampering checks; no qualification is produced."""
import copy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
from functools import lru_cache
import numpy as np
import pytest

Q=np.array([.12,-.06,.06,.12,-.04,.04,0.])
CLASS='sim.auscultation_pi05_v7.start_learned_controller.StartLearnedController'
SOURCE='pi05_absolute_joint_policy_with_bound_start_preparation'


def module(name='audit_start_learned_actions'):
    full='sim.auscultation_pi05_v7.'+name
    assert importlib.util.find_spec(full) is not None, 'Independent start learned audit required'
    return importlib.import_module(full)


@lru_cache(None)
def preparation():
    # Real frozen parent command generator; fixture does not call the new auditor.
    from sim.curobo_expert_v1.timing import smooth_commands
    return smooth_commands(np.vstack([np.zeros(7),Q]),speed_limit=.175,acceleration_limit=1.,
        jerk_limit=1.5,maximum_seconds=300.,minimum_seconds=1/.7,maximum_compute_seconds=15.)


def fixture(tmp_path):
    m=module();command=preparation();n=len(command);end=n+252
    physics=[]
    for i,q in enumerate(np.vstack([command,np.tile(Q,(252+12,1))])):
        role='move_to_declared_start' if i<n else 'settle_at_declared_start' if i<n+240 else 'start_observation'
        row=dict(time_s=(i+1)/240.,q=q.tolist(),dq=[0.]*7,q_command=q.tolist(),hall_N=0.,compression_m=0.,
                 diaphragm_N=0.,lip_N=0.,contact_rim_N=0.,frame_N=0.,forbidden_N=0.,carriage_other_N=0.,
                 subphase='initial_home' if i<n else 'settle',recovery=False,target='R4')
        if i<end:row['preparation_role']=role
        else:row['subphase']='learned_free'
        physics.append(row)
    raw=tmp_path/'raw.npz';np.savez(raw,rgb=np.zeros((2,2,3),np.uint8));raw_sha=hashlib.sha256(raw.read_bytes()).hexdigest()
    frame=dict(index=0,physics_row=end-1,time_s=end/240.,target='R4',phase='APPROACH',q=Q.tolist(),dq=[0.]*7,
               eligible=False,raw_file=raw.name,raw_sha256=raw_sha)
    from sim.auscultation_vla_v2.hall import HallCalibration
    cal_path=m.package_path('v6').parent/'auscultation_vla_v2/calibration/veo6_tared_snapshot.json'
    cal=HallCalibration.from_path(cal_path)
    frame['hall']=dict(raw=int(round(cal.voltage_at_displacement(0.)/cal.supply_v*cal.adc_full_scale)),
        sample_time_s=frame['time_s'],received_time_s=frame['time_s'],calibration_sha256=cal.sha256,source='simulated_adc')
    query=dict(index=0,phase='FREE',target='R4',physics_row_start=end,physics_row_end_exclusive=end+12,
               observation_frame=0,raw_sensor_sha256=raw_sha,time_s=end/240.,measured_q=Q.tolist(),
               inference={'request_id':'first-real-bound-request'})
    receipt=dict(schema='wam_v7_learned_measured_start_receipt_v1',episode_id='case',target='R4',start_family='family',
        executed_controller_class=CLASS,controller_source=SOURCE,requested_start_q_rad=Q.tolist(),
        start_manifest_sha256='a'*64,experiment_protocol_sha256='b'*64,source_plan_sha256='c'*64,
        start_position_tolerance_rad=.03,start_speed_limit_rad_s=.03,preparation_excluded_from_learned_interval=True,
        preparation_origin=dict(physics_row_start=0,time_s=0.,measured_q_rad=[0.]*7,
            measured_dq_rad_s=[0.]*7,retained_command_q_rad=[0.]*7),
        preparation_physics_range=[0,n],settle_physics_range=[n,n+240],observation_physics_range=[n+240,end],
        learned_interval_start_physics_row=end,complete=True,qualification_passed=False,
        first_query=dict(role='first_learned_policy_decision',requested_policy_phase='FREE',acquired_observation_phase='APPROACH',
            acquired_frame_eligible=False,observation_frame=0,observation_physics_row=end-1,physics_row_start=end,
            time_s=end/240.,measured_q_rad=Q.tolist(),measured_dq_rad_s=[0.]*7,retained_command_q_rad=Q.tolist(),
            raw_sensor_file=raw.name,raw_sensor_sha256=raw_sha,model_request_receipt_present=True,
            learned_query_index=0,request_id=query['inference']['request_id']))
    outcome=dict(executed_controller_class=CLASS,controller_source=SOURCE,episode_id='case',start_family='family',
        task_scope='full',requested_points=['R4'],start_manifest_sha256='a'*64,experiment_protocol_sha256='b'*64,
        start_acceptance_audit_required=True,expert_fallbacks=0,training_dataset_eligible=False,speed_scale=.7,
        sequence_execution=dict(scene_instances=1,resets_between_points=0))
    start=dict(target='R4',start_family='family',requested_start_q_rad=Q.tolist(),parent_plan_sha256='c'*64,
        proposed_starts_sha256='d'*64,split='train',geometry_preflight_passed=True,
        requested_physics_profile={'position_iterations':32,'velocity_iterations':1,'armature_kg_m2':.02})
    protocol=dict(schema='wam_v7_frozen_start_family_protocol_v1',actual_start_tolerance_rad=.03,start_spec_sha256='d'*64,
        cases={'family::R4':dict(start_manifest_sha256='a'*64,source_plan_sha256='c'*64,target='R4',start_family='family')},
        families=[dict(family_id='family',requested_q_rad=Q.tolist(),split='train')])
    return dict(outcome=outcome,receipt=receipt,start=start,protocol=protocol,physics=physics,frames=[frame],
                learned=[query],raw_sensor_root=tmp_path)


def test_distinct_start_auditor_required():
    assert module().SCHEMA=='wam_v7_bound_start_learned_action_audit_v1'


def test_independent_start_ledger_and_real_preparation_generator(tmp_path):
    result=module().verify_start_records(**fixture(tmp_path))
    assert result['first_query_bound'] and result['preparation_command_replay_verified']
    assert result['classical_preparation_steps']>252


@pytest.mark.parametrize('change',['old_class','old_source','boundary_gap','extra_classical','wrong_request','wrong_q',
    'moving','raw_tamper','frame_time','frame_relabel','teacher_transfer','prep_command','settle_command','first_nonfree',
    'later_preparation','protocol','reset','origin_home','source_plan'])
def test_rejects_false_start_acceptance(tmp_path,change):
    m=module();v=fixture(tmp_path);r=v['receipt'];p=v['physics'];f=v['frames'][0];q=v['learned'][0]
    if change=='old_class':v['outcome']['executed_controller_class']='sim.auscultation_pi05_v6.learned_controller.LearnedController'
    elif change=='old_source':v['outcome']['controller_source']='pi05_absolute_joint_policy'
    elif change=='boundary_gap':q['physics_row_start']+=1
    elif change=='extra_classical':r['observation_physics_range'][1]+=12
    elif change=='wrong_request':r['first_query']['request_id']='another-request'
    elif change=='wrong_q':f['q'][0]+=.001
    elif change=='moving':f['dq'][0]=.04
    elif change=='raw_tamper':(tmp_path/'raw.npz').write_bytes(b'wrong')
    elif change=='frame_time':f['time_s']-=1/240.
    elif change=='frame_relabel':f['eligible']=True
    elif change=='teacher_transfer':p[10]['subphase']='transfer'
    elif change=='prep_command':p[10]['q_command'][0]+=.00001
    elif change=='settle_command':p[r['settle_physics_range'][0]]['q_command'][0]+=.00001
    elif change=='first_nonfree':q['phase']='CONTACT'
    elif change=='later_preparation':p[-1]['preparation_role']='move_to_declared_start'
    elif change=='protocol':v['protocol']['families'][0]['requested_q_rad'][0]+=.01
    elif change=='reset':v['outcome']['sequence_execution']['resets_between_points']=1
    elif change=='origin_home':r['preparation_origin']['measured_q_rad'][0]=.04
    else:v['start']['parent_plan_sha256']='e'*64
    with pytest.raises((ValueError,KeyError)):m.verify_start_records(**v)


def test_new_assessor_fails_closed_on_original_action_audit():
    a=module('assess_start_trials')
    errors=a.check_documents({}, {}, {'schema':'wam_learned_absolute_joint_action_evidence_audit_v7'}, {},
        checkpoint='a'*64,points=['R4'],scope='full',outcome_sha='b'*64,start_family='family',protocol_sha='c'*64)
    assert any('start action audit' in e for e in errors)
    assert any('controller' in e for e in errors)


def saved_episode(tmp_path):
    """A synthetic new-boundary fixture; original policy auditor is tested separately."""
    m=module();v=fixture(tmp_path);(tmp_path/'sensors').mkdir()
    snapshot=tmp_path/'input_snapshot';objects=snapshot/'objects';objects.mkdir(parents=True)
    files={}
    def add(key,data):
        h=hashlib.sha256(data).hexdigest();(objects/h).write_bytes(data);files[key]={'sha256':h};return h
    plan_sha=add('R4/plan',b'{}\n')
    v['start']['parent_plan_sha256']=plan_sha;v['receipt']['source_plan_sha256']=plan_sha
    v['protocol']['cases']['family::R4']['source_plan_sha256']=plan_sha
    start_sha=add('source/v7/start_manifest',json.dumps(v['start']).encode())
    v['outcome']['start_manifest_sha256']=v['receipt']['start_manifest_sha256']=start_sha
    v['protocol']['cases']['family::R4']['start_manifest_sha256']=start_sha
    # The actual protocol has this extra provenance field; it must not be discarded.
    v['protocol']['cases']['family::R4']['acquisition_kind']='new_independent_closed_loop_episode'
    protocol_sha=add('source/v7/start_protocol',(json.dumps(v['protocol'],sort_keys=True,separators=(',',':'))+'\n').encode())
    v['outcome']['experiment_protocol_sha256']=v['receipt']['experiment_protocol_sha256']=protocol_sha
    for key,path in m.audit_source_paths().items():
        label=('source/pi05_'+key if key.startswith(('v6/','v7/')) else
            'source/code/controller.py' if key=='sim/auscultation_expert_layout_v1/controller.py' else
            'source/runtime/'+key.removeprefix('sim/'))
        add(label,path.read_bytes())
    parent=m.package_path('v6');overlay=Path(m.__file__).parent
    for name in ('learned_controller.py','observed_supervisor.py'):
        add('source/pi05_v6/'+name,(parent/name).read_bytes())
    declared={key:m.sha(path) for key,path in {
        'pi05_v7/run_start_isaac.py':overlay/'run_start_isaac.py',
        'pi05_v7/start_learned_controller.py':overlay/'start_learned_controller.py',
        'pi05_v7/start_collection.py':overlay/'start_collection.py',
        'pi05_v6/learned_controller.py':parent/'learned_controller.py','pi05_v6/execution.py':parent/'execution.py'}.items()}
    v['outcome']['start_runtime_source_hashes']=v['receipt']['start_runtime_source_hashes']=declared
    v['receipt']['controller_source_sha256']=declared['pi05_v7/start_learned_controller.py']
    add('source/vla/hall_calibration.json',(parent.parent/'auscultation_vla_v2/calibration/veo6_tared_snapshot.json').read_bytes())
    manifest=snapshot/'manifest.json';manifest.write_text(json.dumps({'files':files}))
    v['outcome']['input_snapshot']='input_snapshot/manifest.json'
    v['outcome']['input_snapshot_manifest_sha256']=v['receipt']['input_snapshot_manifest_sha256']=m.sha(manifest)
    for filename,rows in [('expert_trace.jsonl',v['physics']),('sensors/frames.jsonl',v['frames']),('learned_actions.jsonl',v['learned'])]:
        (tmp_path/filename).write_text(''.join(json.dumps(x)+'\n' for x in rows))
    (tmp_path/'control_commands.json').write_text(json.dumps([{'physics_row':0,'time_s':0.,'previous_q_command':[0.]*7,'observation':{'q':[0.]*7,'dq':[0.]*7}}]))
    v['receipt']['artifact_sha256']={name:m.sha(tmp_path/name) for name in ('expert_trace.jsonl','sensors/frames.jsonl','learned_actions.jsonl')}
    receipt=tmp_path/'learned_start_receipt.json';receipt.write_text(json.dumps(v['receipt']))
    v['outcome']['learned_start_receipt']={'path':receipt.name,'sha256':m.sha(receipt)}
    (tmp_path/'outcome.json').write_text(json.dumps(v['outcome']))
    return v


@pytest.mark.parametrize('tamper',[None,'source','trace','receipt','protocol','command_origin'])
def test_saved_boundary_wrapper_checks_actual_objects_and_receipts(tmp_path,monkeypatch,tamper):
    m=module();saved_episode(tmp_path)
    monkeypatch.setattr(m.base_audit,'audit_episode',lambda *args,**kwargs:dict(
        schema='wam_learned_absolute_joint_action_evidence_audit_v7',passed=True,errors=[],artifact_sha256={}))
    if tamper=='source':
        p=tmp_path/'input_snapshot/manifest.json';mft=json.loads(p.read_text())
        h=mft['files']['source/pi05_v7/run_start_isaac.py']['sha256'];(p.parent/'objects'/h).write_bytes(b'wrong')
    elif tamper=='trace':(tmp_path/'expert_trace.jsonl').write_text('{}\n')
    elif tamper=='receipt':(tmp_path/'learned_start_receipt.json').write_text('{}')
    elif tamper=='protocol':
        p=tmp_path/'input_snapshot/manifest.json';mft=json.loads(p.read_text())
        h=mft['files']['source/v7/start_protocol']['sha256'];(p.parent/'objects'/h).write_bytes(b'wrong')
    elif tamper=='command_origin':
        p=tmp_path/'control_commands.json';p.write_text(json.dumps([{'physics_row':0,'previous_q_command':[.01]*7}]))
    result=m.audit_episode(tmp_path,expected_checkpoint_metadata_sha256='f'*64,raw_sensor_root=tmp_path)
    assert result['passed'] is (tamper is None),result
    if tamper is None:
        assert result['start_boundary_verified'] and result['start_source_snapshot_verified']
        assert result['schema']==m.SCHEMA


@pytest.mark.parametrize('change',[None,'adc_loaded','physics_loaded','contact','clock'])
def test_independent_actual_adc_and_truth_release_gate(tmp_path,change):
    m=module();v=fixture(tmp_path);first=v['physics'][v['receipt']['learned_interval_start_physics_row']-1];frame=v['frames'][0]
    cal=m.base_audit.HallCalibration.from_path(m.package_path('v6').parent/'auscultation_vla_v2/calibration/veo6_tared_snapshot.json')
    if change=='adc_loaded':frame['hall']['raw']=int(round(cal.voltage_at_displacement(.001)/cal.supply_v*cal.adc_full_scale))
    elif change=='physics_loaded':first['hall_N']=.2
    elif change=='contact':first['lip_N']=.2
    elif change=='clock':frame['hall']['sample_time_s']-=.051
    if change is None:assert m.verify_start_release(first,frame,cal)['adc_release_verified']
    else:
        with pytest.raises(ValueError):m.verify_start_release(first,frame,cal)


def valid_documents():
    a=module('assess_start_trials');c=importlib.import_module('sim.auscultation_pi05_v7.policy_client')
    from sim.auscultation_pi05_v6.training_data import MODALITIES
    from sim.auscultation_pi05_v6.task_observation import observation_binding
    from sim.auscultation_pi05_v7.action_encoding import action_encoding_binding
    binding=dict(checkpoint_metadata_sha256='a'*64,data_hash='b'*64,
        inference_source_sha256={name:'c'*64 for name in c.inference_source_paths()},
        sampling_steps=10,sampling_seed=917,sampling_policy='fixed common Gaussian seed',
        action_semantics='absolute_joint_command_v1',model_action_semantics='future_command_minus_fixed_observation_measured_q',
        training_recipe=c.RECIPE,modalities=MODALITIES,audio='absent',dynamics_recipe_sha256='d'*64,
        dynamics_contract_sha256='e'*64,execution_contract_sha256='f'*64,
        observation_contract=observation_binding(),action_encoding_binding=action_encoding_binding())
    outcome=dict(controller_source=SOURCE,executed_controller_class=CLASS,start_acceptance_audit_required=True,
        start_family='family',experiment_protocol_sha256='c'*64,start_manifest_sha256='d'*64,
        training_dataset_eligible=False,expert_demonstration=False,expert_fallbacks=0,learned_policy_binding=binding,
        requested_points=['R4'],task_scope='full',learned_policy_queries=10,success=True,
        points=[dict(target='R4',success=True,observed_hold_s=3.,verified_front_contact_hold_s=3.,final_phase='DONE')])
    collection=dict(passed=True,smoothness_target_met=True,audit={'accepted':True},teardown={'close_returned':True,'errors':[]})
    actions=dict(schema='wam_v7_bound_start_learned_action_audit_v1',base_action_audit_schema='wam_learned_absolute_joint_action_evidence_audit_v7',
        start_boundary_verified=True,start_source_snapshot_verified=True,start_family='family',experiment_protocol_sha256='c'*64,
        start_manifest_sha256='d'*64,expected_checkpoint_metadata_sha256='a'*64,verified_executed_steps=12,
        failed_queries=0,unverified_requests=0,artifact_sha256={'/trial/outcome.json':'b'*64})
    for key in ('passed','evidence_consistent','command_binding_verified','sensor_payload_binding_verified',
        'policy_task_observation_verified','request_receipt_binding_verified','server_request_log_verified','inference_source_snapshot_verified'):
        actions[key]=True
    supervisor=dict(schema='wam_observed_supervisor_replay_audit_v1',passed=True,artifact_sha256={'outcome.json':'b'*64},
        replay=dict(every_observation_recomputed=True,every_manager_update_replayed=True,all_active_ticks_covered=True))
    return [outcome,collection,actions,supervisor]


@pytest.mark.parametrize('change',[None,'old_class','old_audit','unbound_start','no_service','mechanics','short_hold','residual_on_wire'])
def test_assessor_requires_all_original_gates_plus_real_start_gate(change):
    a=module('assess_start_trials');docs=valid_documents();o,c,actions,s=docs
    if change=='old_class':o['executed_controller_class']='sim.auscultation_pi05_v6.learned_controller.LearnedController'
    elif change=='old_audit':actions['schema']='wam_learned_absolute_joint_action_evidence_audit_v7'
    elif change=='unbound_start':actions['start_boundary_verified']=False
    elif change=='no_service':actions['server_request_log_verified']=False
    elif change=='mechanics':c['smoothness_target_met']=False
    elif change=='short_hold':o['points'][0]['observed_hold_s']=2.9
    elif change=='residual_on_wire':o['learned_policy_binding']['action_semantics']='joint_delta'
    errors=a.check_documents(*docs,checkpoint='a'*64,points=['R4'],scope='full',outcome_sha='b'*64,
        start_family='family',protocol_sha='c'*64)
    assert (errors==[]) is (change is None),errors
