"""Assess declared-start learned cases with an additional independent start gate.

This report distinguishes contact components, full individual points and a
continuous sequence. It never infers unseen-person or hardware capability.
"""
import argparse
import json
from pathlib import Path

from sim.auscultation_pi05_v6.dynamics_contract import sha256_file
from sim.auscultation_pi05_v6.training_data import MODALITIES
from sim.auscultation_pi05_v6.task_observation import observation_binding
from .action_encoding import action_encoding_binding
from .policy_client import verify_policy_binding, RECIPE, package_path

POINTS=tuple(side+str(i) for i in range(1,6) for side in 'RL')


def check_documents(outcome,collection,actions,supervisor,*,checkpoint,points,scope,outcome_sha,start_family,protocol_sha):
    errors=[]
    def need(ok,message):
        if not ok:errors.append(message)
    binding=outcome.get('learned_policy_binding',{})
    need(outcome.get('controller_source')=='pi05_absolute_joint_policy_with_bound_start_preparation'
         and outcome.get('executed_controller_class')=='sim.auscultation_pi05_v7.start_learned_controller.StartLearnedController',
         'not the declared-start learned controller')
    need(scope=='full' and outcome.get('start_acceptance_audit_required') is True,'full learned-start scope required')
    need(outcome.get('start_family')==start_family and outcome.get('experiment_protocol_sha256')==protocol_sha,
         'declared start family/protocol differs')
    need(outcome.get('training_dataset_eligible') is False,'learned trial cannot be training-qualified expert data')
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
    need(actions.get('schema')=='wam_v7_bound_start_learned_action_audit_v1'
         and actions.get('base_action_audit_schema')=='wam_learned_absolute_joint_action_evidence_audit_v7','wrong start action audit')
    need(actions.get('start_boundary_verified') is True and actions.get('start_source_snapshot_verified') is True,
         'independent measured-start/command/source verification incomplete')
    need(actions.get('start_family')==start_family and actions.get('experiment_protocol_sha256')==protocol_sha
         and actions.get('start_manifest_sha256')==outcome.get('start_manifest_sha256'),'start audit binding differs')
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


def assess_case(record,checkpoint):
    root=Path(record['directory']).resolve();artifacts={}
    report=dict(case_id=record['case_id'],directory=str(root),points=record['points'],scope=record['scope'],
                passed=False,errors=[],artifact_sha256=artifacts)
    try:
        def read(name):
            path=root/name;artifacts[name]=sha256_file(path);return json.loads(path.read_text())
        outcome=read('outcome.json');collection=read('collection_result.json')
        actions=read('learned_start_actions_audit.json');supervisor=read('observed_supervisor_audit.json')
        report['errors']=check_documents(outcome,collection,actions,supervisor,checkpoint=checkpoint,
            points=record['points'],scope=record['scope'],outcome_sha=artifacts['outcome.json'],
            start_family=record['start_family'],protocol_sha=record['experiment_protocol_sha256'])
        if outcome.get('start_manifest_sha256')!=record['start_manifest_sha256']:
            report['errors'].append('actual start manifest differs from the predeclared case')
        from .audit_start_learned_actions import audit_source_paths
        paths=audit_source_paths()
        if set(actions.get('audit_source_sha256',{}))!=set(paths):
            report['errors'].append('action audit source binding incomplete')
        for name,expected in actions.get('audit_source_sha256',{}).items():
            if name not in paths or sha256_file(paths[name])!=expected:
                report['errors'].append('action auditor source differs: '+name)
        for path,expected in actions.get('artifact_sha256',{}).items():
            if sha256_file(path)!=expected:report['errors'].append('action evidence changed: '+path)
        for relative,expected in supervisor.get('artifact_sha256',{}).items():
            path=(root/relative).resolve()
            if not path.is_relative_to(root) or sha256_file(path)!=expected:
                report['errors'].append('supervisor evidence changed: '+relative)
        parent=package_path('v6')
        if supervisor.get('audit_source_sha256')!=sha256_file(parent/'audit_observed_supervisor.py'):
            report['errors'].append('contact auditor source differs')
        for name,expected in artifacts.items():
            if sha256_file(root/name)!=expected:report['errors'].append('evidence changed during assessment')
        report['passed']=not report['errors']
    except (OSError,ValueError,KeyError,TypeError) as error:
        report['errors'].append(type(error).__name__+': '+str(error))
    return report


def assess(plan):
    if plan.get('schema')!='wam_v7_bound_start_learned_assessment_plan_v1':
        raise ValueError('Explicit bound-start learned assessment plan required')
    records=plan['cases'];checkpoint=plan['checkpoint_metadata_sha256']
    if not records or len({r['case_id'] for r in records})!=len(records):
        raise ValueError('Nonempty uniquely identified cases required')
    identities=[]
    for record in records:
        points=record['points']
        if (record['scope']!='full' or not points or len(set(points))!=len(points)
                or any(p not in POINTS for p in points)
                or record['role'] not in {'bound_start_single_point','bound_start_continuous_sequence'}
                or (record['role']=='bound_start_single_point' and len(points)!=1)
                or (record['role']=='bound_start_continuous_sequence' and (len(points)!=10 or set(points)!=set(POINTS)))):
            raise ValueError('Only a full individual point or continuous ten-target start case is declared')
        for key in ('start_manifest_sha256','experiment_protocol_sha256'):
            value=record[key]
            if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Explicit SHA required: '+key)
        identities.append((record['start_family'],tuple(points),record['experiment_protocol_sha256']))
    if len(set(identities))!=len(identities):raise ValueError('Duplicate declared start/target case')
    results=[dict(assess_case(r,checkpoint),role=r['role'],start_family=r['start_family']) for r in records]
    return dict(schema='wam_v7_bound_start_isaac_acceptance_summary_v1',checkpoint_metadata_sha256=checkpoint,
        all_declared_tests_passed=all(r['passed'] for r in results),cases=results,
        original_zero_home_rollout_repaired=False,unseen_body_pose_motion_validated=False,realtime_validated=False,
        acoustic_or_clinical_validated=False,hardware_validated=False,
        scope='Only explicitly declared start-family learned cases; never substitutes for original zero-home evaluation',
        assessor_sha256=sha256_file(__file__))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=assess(json.loads(args.plan.read_text()));result['plan_sha256']=sha256_file(args.plan)
    with args.output.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(all_declared_tests_passed=result['all_declared_tests_passed'],cases=len(result['cases']))))
    raise SystemExit(0 if result['all_declared_tests_passed'] else 1)
