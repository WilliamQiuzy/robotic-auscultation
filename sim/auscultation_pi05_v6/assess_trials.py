"""Combine completed learned Isaac evidence; experts/offline tests cannot pass.

This report distinguishes contact components, full individual points and a
continuous sequence. It never infers unseen-person or hardware capability.
"""
import argparse
import json
from pathlib import Path

from .dynamics_contract import sha256_file
from .training_data import MODALITIES
from .task_observation import observation_binding
from .action_encoding import action_encoding_binding

POINTS=tuple(side+str(i) for i in range(1,6) for side in 'RL')


def check_documents(outcome,collection,actions,supervisor,*,checkpoint,points,scope,outcome_sha):
    errors=[]
    def need(ok,message):
        if not ok:errors.append(message)
    binding=outcome.get('learned_policy_binding',{})
    need(outcome.get('controller_source')=='pi05_absolute_joint_policy','not a learned pi05 controller')
    need(outcome.get('expert_demonstration') is False and outcome.get('expert_fallbacks')==0,
         'expert identity/fallbacks are incompatible with learned acceptance')
    need(binding.get('checkpoint_metadata_sha256')==checkpoint,'checkpoint differs')
    need(binding.get('training_recipe')=='dynamics_bound_absolute_joint_actions_v6_task_pose_logit_v1','wrong policy recipe')
    need(binding.get('observation_contract')==observation_binding(),'task conditioning binding differs')
    need(binding.get('action_encoding_binding')==action_encoding_binding(),'bounded action encoding differs')
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
    need(actions.get('schema')=='wam_learned_absolute_joint_action_evidence_audit_v6','wrong action audit')
    need(actions.get('expected_checkpoint_metadata_sha256')==checkpoint,'action audit checkpoint differs')
    need(all(actions.get(k) is True for k in ('passed','evidence_consistent','command_binding_verified',
        'sensor_payload_binding_verified','policy_task_observation_verified','request_receipt_binding_verified','server_request_log_verified')),
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
        actions=read('learned_actions_audit.json');supervisor=read('observed_supervisor_audit.json')
        report['errors']=check_documents(outcome,collection,actions,supervisor,checkpoint=checkpoint,
            points=record['points'],scope=record['scope'],outcome_sha=artifacts['outcome.json'])
        for name,expected in actions.get('audit_source_sha256',{}).items():
            if Path(name).name!=name or sha256_file(Path(__file__).with_name(name))!=expected:
                report['errors'].append('action auditor source differs: '+name)
        if set(actions.get('audit_source_sha256',{}))!={'audit_learned_actions.py','execution.py','action_contract.py','task_observation.py','action_encoding.py'}:
            report['errors'].append('action audit source binding incomplete')
        if supervisor.get('audit_source_sha256')!=sha256_file(Path(__file__).with_name('audit_observed_supervisor.py')):
            report['errors'].append('contact auditor source differs')
        for name,expected in artifacts.items():
            if sha256_file(root/name)!=expected:report['errors'].append('evidence changed during assessment')
        report['passed']=not report['errors']
    except (OSError,ValueError,KeyError,TypeError) as error:
        report['errors'].append(type(error).__name__+': '+str(error))
    return report


def assess(plan):
    records=plan['cases'];checkpoint=plan['checkpoint_metadata_sha256']
    if len({r['case_id'] for r in records})!=len(records):raise ValueError('Duplicate case identity')
    nominal=[r for r in records if r.get('role')=='nominal_single_point']
    sequences=[r for r in records if r.get('role')=='nominal_continuous_sequence']
    if ({tuple(r['points']) for r in nominal}!={(p,) for p in POINTS} or len(nominal)!=10
            or any(r['scope']!='full' for r in nominal) or len(sequences)!=1
            or sequences[0]['points']!=list(POINTS)
            or sequences[0]['scope']!='full'):
        raise ValueError('Predeclared ten full individual points and one full continuous sequence required')
    results=[dict(assess_case(r,checkpoint),role=r['role']) for r in records]
    fixed=all(r['passed'] for r in results if r['role'] in {'nominal_single_point','nominal_continuous_sequence'})
    return dict(schema='wam_pi05_isaac_acceptance_summary_v6',checkpoint_metadata_sha256=checkpoint,
        fixed_scene_mechanical_task_passed=fixed,all_declared_tests_passed=all(r['passed'] for r in results),
        cases=results,unseen_body_pose_motion_validated=False,realtime_validated=False,
        acoustic_or_clinical_validated=False,hardware_validated=False,
        scope='Recorded learned task execution under the declared simulation conditions only',
        assessor_sha256=sha256_file(__file__))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=assess(json.loads(args.plan.read_text()));result['plan_sha256']=sha256_file(args.plan)
    with args.output.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(fixed_scene_mechanical_task_passed=result['fixed_scene_mechanical_task_passed'],
                         all_declared_tests_passed=result['all_declared_tests_passed'],cases=len(result['cases']))))
