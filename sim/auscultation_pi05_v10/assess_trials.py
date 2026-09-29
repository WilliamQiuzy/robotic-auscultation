"""V10 selected recovery admission plus unchanged physical/Headroom acceptance.

Only zero-home full single points and one continuous sequence are in this
version. This does not certify start-family, unseen-patient or hardware tasks.
"""
import argparse
import json
from pathlib import Path
from sim.auscultation_pi05_v7 import coverage_pilot_evaluation as E
from . import runtime_contract as C
from .audit_actions import SCHEMA,audit_source_paths
from .policy_contract import verify_policy_binding
from .policy_contract import RECIPE
from .runtime_contract import package_path
from sim.auscultation_pi05_v6.training_data import MODALITIES
from sim.auscultation_pi05_v6.task_observation import observation_binding
from sim.auscultation_pi05_v8.action_encoding import action_encoding_binding as codec_binding, B
def action_encoding_binding():return codec_binding(B)
POINTS=tuple(side+str(i) for i in range(1,6) for side in 'RL')
def inherited_check_documents(outcome,collection,actions,supervisor,*,checkpoint,points,scope,outcome_sha):
    errors=[]
    def need(ok,message):
        if not ok:errors.append(message)
    binding=outcome.get('learned_policy_binding',{})
    need(outcome.get('controller_source')=='pi05_absolute_joint_policy_with_runtime_jerk_headroom','wrong learned runtime identity')
    need(outcome.get('executed_controller_class')=='sim.auscultation_pi05_v7.headroom_learned_controller.HeadroomLearnedController','wrong headroom controller class')
    from sim.auscultation_pi05_v7.headroom_execution import verify_contract_bindings
    try:verify_contract_bindings(outcome,binding)
    except (ValueError,KeyError,TypeError) as error:errors.append(str(error))
    from sim.auscultation_pi05_v7.run_headroom_isaac import PARENT_INVENTORY_SHA256
    need(outcome.get('parent_runtime_inventory_sha256')==PARENT_INVENTORY_SHA256,'original runtime inventory differs')
    need(outcome.get('original_failed_case_reclassified') is False,'original failed trial cannot be reclassified')
    need(all(actions.get(k) is True for k in ('runtime_adaptation_verified','runtime_source_snapshot_verified',
         'original_whole_stream_gate_verified')),'runtime adaptation/unchanged gate audit incomplete')
    need(actions.get('trained_execution_sha256')==outcome.get('trained_execution_sha256')
         and actions.get('runtime_execution_sha256')==outcome.get('runtime_execution_sha256'),'action audit execution contracts differ')
    need(outcome.get('expert_demonstration') is False and outcome.get('expert_fallbacks')==0,
         'expert identity/fallbacks are incompatible with learned acceptance')
    need(binding.get('checkpoint_metadata_sha256')==checkpoint,'checkpoint differs')
    try:verify_policy_binding(binding,checkpoint)
    except ValueError as error:errors.append(str(error))
    need(binding.get('training_recipe')==RECIPE,'wrong policy recipe')
    need(binding.get('observation_contract')==observation_binding(),'task conditioning binding differs')
    need(binding.get('action_encoding_binding')==action_encoding_binding(),'fixed-observation residual encoding differs')
    need(binding.get('modalities')==MODALITIES and binding.get('audio')=='absent','modality binding differs')
    need(outcome.get('requested_points')==points and outcome.get('task_scope')==scope,'requested task differs')
    need(type(outcome.get('learned_policy_queries')) is int and outcome['learned_policy_queries']>0,'no learned queries')
    need(outcome.get('success') is True,'episode failed')
    actual=outcome.get('points',[])
    need([p.get('target') for p in actual]==points and all(p.get('success') is True for p in actual),
         'not every requested point completed')
    need(all(p.get('observed_hold_s',0)>=3-1e-9 and p.get('verified_front_contact_hold_s',0)>=3-2/240
             and p.get('final_phase')=='DONE' for p in actual),'hold/release incomplete')
    need(collection.get('passed') is True and collection.get('smoothness_target_met') is True
         and collection.get('audit',{}).get('accepted') is True,'independent physical gates failed')
    teardown=collection.get('teardown',{})
    need(teardown.get('close_returned') is True and teardown.get('errors')==[],'shutdown incomplete')
    need(actions.get('schema')=='wam_pi05_v10_recovery_headroom_replay_v1','wrong action audit')
    need(actions.get('expected_checkpoint_metadata_sha256')==checkpoint,'action audit checkpoint differs')
    need(all(actions.get(k) is True for k in ('passed','evidence_consistent','command_binding_verified',
        'sensor_payload_binding_verified','policy_task_observation_verified','request_receipt_binding_verified','server_request_log_verified','inference_source_snapshot_verified')),
         'learned proposal/request/execution evidence incomplete')
    need(actions.get('verified_executed_steps',0)>0 and actions.get('failed_queries')==0
         and actions.get('unverified_requests')==0,'failed or unverified learned actions')
    hashes=[v for k,v in actions.get('artifact_sha256',{}).items() if Path(k).name=='outcome.json']
    need(hashes==[outcome_sha],'action audit is not bound to this outcome')
    need(supervisor.get('schema')=='wam_observed_supervisor_replay_audit_v1'
         and supervisor.get('passed') is True,'measured contact replay failed')
    need(supervisor.get('artifact_sha256',{}).get('outcome.json')==outcome_sha,'contact audit outcome differs')
    need(all(supervisor.get('replay',{}).get(k) is True for k in ('every_observation_recomputed',
         'every_manager_update_replayed','all_active_ticks_covered')),'contact replay is incomplete')
    if len(points)>1:
        sequence=outcome.get('sequence_execution',{})
        need(scope=='full' and sequence.get('scene_instances')==1 and sequence.get('resets_between_points')==0,
             'sequence was reset or substituted with component tasks')
    return errors




def verify_outer_audit(actions):
    E.need(actions.get('schema')==SCHEMA,'New V10 recovery action audit required')
    E.need(all(actions.get(k) is True for k in ('passed','recovery_binding_verified',
        'recovery_admission_verified','connection_source_snapshot_verified','recovery_request_bindings_verified','admission_bridge_verified','codec_decode_verified')),
        'V10 recovery action binding/selection/source audit incomplete')
    inherited=actions.get('inherited_action_audit',{})
    E.need(inherited.get('schema')=='wam_pi05_v10_recovery_headroom_replay_v1' and inherited.get('passed') is True,
        'Complete inherited Headroom replay required')
    E.need(actions.get('errors')==[] and type(actions.get('recovery_queries_checked')) is int
        and actions['recovery_queries_checked']>0,'V10 recovery query audit incomplete')
    return inherited


def assess_case(record,checkpoint):
    root=Path(record['directory']).resolve();artifacts={}
    report=dict(schema='wam_pi05_v10_recovery_headroom_case_assessment_v1',case_id=record['case_id'],directory=str(root),
        points=record['points'],scope=record['scope'],passed=False,errors=[],artifact_sha256=artifacts,
        aggregate_protocol_acceptance=False,admission_bridge_verified=False)
    try:
        def read(name):
            path=root/name;artifacts[name]=E.file_sha(path);return E.read(path)
        outcome=read('outcome.json');collection=read('collection_result.json')
        actions=read('learned_actions_audit.json');supervisor=read('observed_supervisor_audit.json')
        inherited=verify_outer_audit(actions);saved=actions['recovery_runtime_binding']
        admission=C.selected_admission(saved['admission_binding']['path'],saved['admission_binding']['sha256'],verifier_binding=saved['admission_verifier_binding'])
        E.need(admission==saved,'Selected model admission changed after action audit')
        C.verify_connection_outcome(outcome,admission)
        E.need(admission['checkpoint_metadata_sha256']==checkpoint,'Plan selected checkpoint differs')
        E.need(actions.get('expected_checkpoint_metadata_sha256')==checkpoint,'V10 recovery audit checkpoint differs')
        E.need(actions.get('runtime_connection_contract')==C.runtime_connection_contract(),'V10 recovery audit runtime contract differs')
        C.verify_artifacts(actions.get('artifact_sha256',{}))
        C.B.verify_artifacts(admission)
        expected_sources={name:E.file_sha(path) for name,path in audit_source_paths(admission).items()}
        E.need(actions.get('audit_source_sha256')==expected_sources,'V10 recovery auditor/source inventory differs')
        report['errors']+=inherited_check_documents(outcome,collection,inherited,supervisor,checkpoint=checkpoint,
            points=record['points'],scope=record['scope'],outcome_sha=artifacts['outcome.json'])
        from .audit_actions import inherited_audit_source_paths as inherited_sources
        E.need(inherited.get('audit_source_sha256')=={k:E.file_sha(p) for k,p in inherited_sources().items()},
            'Inherited auditor source map differs')
        from .runtime_contract import package_path
        E.need(supervisor.get('audit_source_sha256')==E.file_sha(package_path('v6')/'audit_observed_supervisor.py'),
            'Observed supervisor audit source differs')
        for name,digest in artifacts.items():E.need(E.file_sha(root/name)==digest,'Assessment evidence changed')
        report.update(passed=not report['errors'],admission_bridge_verified=True,admission_binding=admission['admission_binding'],
            checkpoint_binding=admission['checkpoint_binding'],training_source_sha256=admission['training_source_sha256'],
            runtime_connection_contract=C.runtime_connection_contract())
    except (OSError,ValueError,KeyError,TypeError,RuntimeError) as error:
        report['errors'].append(type(error).__name__+': '+str(error))
        if isinstance(error,C.B.AdmissionFailure):report['admission_bridge_failure']=error.receipt
    return report


def assess(plan):
    E.need(plan.get('schema')=='wam_pi05_v10_recovery_headroom_acceptance_plan_v1','New V10 recoveryed acceptance plan required')
    records=plan['cases'];checkpoint=plan['checkpoint_metadata_sha256'];selection=plan['admission_binding']
    E.need(len({r['case_id'] for r in records})==len(records),'Duplicate case identity')
    nominal=[r for r in records if r.get('role')=='nominal_single_point']
    sequences=[r for r in records if r.get('role')=='nominal_continuous_sequence']
    E.need(len(records)==11 and {tuple(r['points']) for r in nominal}=={(p,) for p in POINTS} and len(nominal)==10
        and all(r['scope']=='full' for r in records) and len(sequences)==1 and sequences[0]['points']==list(POINTS),
        'Exactly ten full singles and the ordered continuous sequence required')
    cases=[dict(assess_case(r,checkpoint),role=r['role']) for r in records]
    for row in cases:
        if row.get('admission_binding')!=selection:
            row['passed']=False;row['errors'].append('Case pair admission differs from fixed plan')
    return dict(schema='wam_pi05_v10_recovery_headroom_acceptance_summary_v1',checkpoint_metadata_sha256=checkpoint,
        admission_binding=selection,cases=cases,all_declared_tests_passed=all(r['passed'] for r in cases),
        fixed_scene_mechanical_task_passed=all(r['passed'] for r in cases),requested_cases=11,
        start_family_generalization_validated=False,unseen_body_pose_motion_validated=False,
        acoustic_or_clinical_validated=False,hardware_validated=False,realtime_validated=False,
        assessor_sha256=E.file_sha(__file__))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=assess(E.read(a.plan))
    result['plan_sha256']=E.file_sha(a.plan);E.write_new(a.output,result)
    print(json.dumps(dict(all_declared_tests_passed=result['all_declared_tests_passed'])),flush=True)
    raise SystemExit(0 if result['all_declared_tests_passed'] else 1)


if __name__=='__main__':main()
