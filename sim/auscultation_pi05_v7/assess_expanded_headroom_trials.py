"""Expanded experiment admission plus unchanged physical/Headroom acceptance.

Only zero-home full single points and one continuous sequence are in this
version. This does not certify start-family, unseen-patient or hardware tasks.
"""
import argparse
import json
from pathlib import Path
from . import expanded_evaluation as E
from . import expanded_runtime_contract as C
from .audit_expanded_headroom_actions import SCHEMA,audit_source_paths
from .assess_headroom_trials import check_documents as inherited_check_documents,POINTS


def verify_outer_audit(actions):
    E.require(actions.get('schema')==SCHEMA,'New expanded action audit required')
    E.require(all(actions.get(k) is True for k in ('passed','experiment_binding_verified',
        'selected_checkpoint_lock_verified','connection_source_snapshot_verified','experiment_request_bindings_verified')),
        'Expanded action binding/selection/source audit incomplete')
    inherited=actions.get('inherited_action_audit',{})
    E.require(inherited.get('schema')=='wam_v7_headroom_learned_action_audit_v1' and inherited.get('passed') is True,
        'Complete inherited Headroom replay required')
    E.require(actions.get('errors')==[] and type(actions.get('expanded_queries_checked')) is int
        and actions['expanded_queries_checked']>0,'Expanded query audit incomplete')
    return inherited


def assess_case(record,checkpoint):
    root=Path(record['directory']).resolve();artifacts={}
    report=dict(schema='wam_v7_expanded_headroom_case_assessment_v1',case_id=record['case_id'],directory=str(root),
        points=record['points'],scope=record['scope'],passed=False,errors=[],artifact_sha256=artifacts,
        aggregate_protocol_acceptance=False)
    try:
        def read(name):
            path=root/name;artifacts[name]=E.sha(path);return E.read(path)
        outcome=read('outcome.json');collection=read('collection_result.json')
        actions=read('learned_actions_audit.json');supervisor=read('observed_supervisor_audit.json')
        inherited=verify_outer_audit(actions);saved=actions['experiment_runtime_binding']
        admission=C.selected_admission(saved['selection_lock_path'],saved['selection_lock_sha256'],saved['manifest_path'],saved['experiment_path'])
        E.require(admission==saved,'Selected model admission changed after action audit')
        C.verify_connection_outcome(outcome,admission)
        E.require(admission['checkpoint_metadata_sha256']==checkpoint,'Plan selected checkpoint differs')
        E.require(actions.get('expected_checkpoint_metadata_sha256')==checkpoint,'Expanded audit checkpoint differs')
        E.require(actions.get('runtime_connection_contract')==C.runtime_connection_contract(),'Expanded audit runtime contract differs')
        C.verify_artifacts(actions.get('artifact_sha256',{}))
        expected_sources={name:E.sha(path) for name,path in audit_source_paths().items()}
        E.require(actions.get('audit_source_sha256')==expected_sources,'Expanded auditor/source inventory differs')
        report['errors']+=inherited_check_documents(outcome,collection,inherited,supervisor,checkpoint=checkpoint,
            points=record['points'],scope=record['scope'],outcome_sha=artifacts['outcome.json'])
        from .audit_headroom_learned_actions import audit_source_paths as inherited_sources
        E.require(inherited.get('audit_source_sha256')=={k:E.sha(p) for k,p in inherited_sources().items()},
            'Inherited auditor source map differs')
        from .policy_client import package_path
        E.require(supervisor.get('audit_source_sha256')==E.sha(package_path('v6')/'audit_observed_supervisor.py'),
            'Observed supervisor audit source differs')
        for name,digest in artifacts.items():E.require(E.sha(root/name)==digest,'Assessment evidence changed')
        report.update(passed=not report['errors'],selection_lock_sha256=admission['selection_lock_sha256'],
            experiment_binding=admission['experiment_binding'],experiment_source_sha256=admission['experiment_source_sha256'],
            runtime_connection_contract=C.runtime_connection_contract())
    except (OSError,ValueError,KeyError,TypeError,RuntimeError) as error:
        report['errors'].append(type(error).__name__+': '+str(error))
    return report


def assess(plan):
    E.require(plan.get('schema')=='wam_v7_expanded_headroom_acceptance_plan_v1','New expanded acceptance plan required')
    records=plan['cases'];checkpoint=plan['checkpoint_metadata_sha256'];selection=plan['selection_lock_sha256']
    E.require(len({r['case_id'] for r in records})==len(records),'Duplicate case identity')
    nominal=[r for r in records if r.get('role')=='nominal_single_point']
    sequences=[r for r in records if r.get('role')=='nominal_continuous_sequence']
    E.require(len(records)==11 and {tuple(r['points']) for r in nominal}=={(p,) for p in POINTS} and len(nominal)==10
        and all(r['scope']=='full' for r in records) and len(sequences)==1 and sequences[0]['points']==list(POINTS),
        'Exactly ten full singles and the ordered continuous sequence required')
    cases=[dict(assess_case(r,checkpoint),role=r['role']) for r in records]
    for row in cases:
        if row.get('selection_lock_sha256')!=selection:
            row['passed']=False;row['errors'].append('Case selection lock differs from fixed plan')
    return dict(schema='wam_v7_expanded_headroom_acceptance_summary_v1',checkpoint_metadata_sha256=checkpoint,
        selection_lock_sha256=selection,cases=cases,all_declared_tests_passed=all(r['passed'] for r in cases),
        fixed_scene_mechanical_task_passed=all(r['passed'] for r in cases),requested_cases=11,
        start_family_generalization_validated=False,unseen_body_pose_motion_validated=False,
        acoustic_or_clinical_validated=False,hardware_validated=False,realtime_validated=False,
        assessor_sha256=E.sha(__file__))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=assess(E.read(a.plan))
    result['plan_sha256']=E.sha(a.plan);E.write_new(a.output,result)
    print(json.dumps(dict(all_declared_tests_passed=result['all_declared_tests_passed'])),flush=True)
    raise SystemExit(0 if result['all_declared_tests_passed'] else 1)


if __name__=='__main__':main()
