"""Independently replay the unchanged policy executor and every gate exception."""
import argparse
import importlib
import json
from pathlib import Path
import numpy as np
from . import audit_learned_actions as base
from .numerical_command_gate import CertificateTracker,compare_command,numerical_gate_contract,contract_sha256
from sim.auscultation_pi05_v6.execution import JointTrajectory

CLASS='sim.auscultation_pi05_v7.numerical_learned_controller.NumericalLearnedController'
SOURCE='pi05_absolute_joint_policy_with_certified_numerical_comparison'
MODULES=('numerical_command_gate.py','numerical_learned_controller.py','run_numerical_isaac.py',
         'audit_numerical_learned_actions.py','assess_numerical_trials.py')
SCHEMA='wam_v7_numerical_gate_learned_action_audit_v1'


def audit_source_paths():
    paths=base.audit_source_paths()
    paths.update({'v7/'+name:Path(__file__).with_name(name) for name in MODULES})
    paths['skill_v1/executor.py']=Path(importlib.import_module('sim.auscultation_skill_v1.executor').__file__)
    paths['skill_v1/controller.py']=Path(importlib.import_module('sim.auscultation_skill_v1.controller').__file__)
    return paths


def replay_gate_records(learned,physics,events,receipt):
    """Reconstruct commands and certificates; saved derived verdicts are compared."""
    event_by_row={}
    for index,event in enumerate(events):
        row=event.get('physics_row')
        base.require(event.get('index')==index and type(row) is int and 3<=row<=len(physics),'Invalid numerical event index/row')
        base.require(row not in event_by_row,'Duplicate numerical event')
        event_by_row[row]=event
    query_by_row={};starts={}
    for query in learned:
        if query.get('accepted') is not True:continue
        start=query['physics_row_start'];end=query['physics_row_end_exclusive']
        base.require(start not in starts,'Duplicate accepted query start');starts[start]=query
        for row in range(start,end):
            base.require(row not in query_by_row,'Overlapping query intervals');query_by_row[row]=query
    trajectory=None;target=None;tracker=CertificateTracker();release=False;seen=[];interventions=0;verified=0
    def check(row,command,subphase,contact_active,executed):
        nonlocal interventions,verified
        if row<3:return
        history=np.asarray([p['q_command'] for p in physics[row-3:row]],float)
        result=compare_command(history,command,velocity=.175,acceleration=.5 if contact_active else 1.)
        if result['raw_passed']:
            base.require(row not in event_by_row,'Recorded numerical exception when raw gate passed');return
        cert=tracker.certificate()
        result=compare_command(history,command,velocity=.175,acceleration=.5 if contact_active else 1.,certificate=cert)
        base.require(row in event_by_row,'Missing numerical raw-failure event')
        event=event_by_row[row];seen.append(row)
        base.require(event['history']==history.tolist() and event['command']==np.asarray(command,float).tolist(),'Numerical input command/history differs')
        base.require(event['certificate']==cert,'Numerical certificate differs from independent continuous Ruckig replay')
        base.require(event['comparison']==result,'Numerical verdict differs from independent exact computation')
        base.require(event['subphase']==subphase and event['contact_active'] is contact_active
                     and event['active'] is False and event['recovery'] is False,'Numerical phase/physical mode differs')
        base.require(event['physics_executed'] is executed,'Numerical event execution marker differs')
        base.require(event['time_s']==physics[row-1]['time_s'],'Numerical event clock differs')
        if executed:base.require(result['passed'],'Executed command failed the certified comparison')
        interventions+=int(result['intervened']);verified+=1
    for row,physical in enumerate(physics):
        command=np.asarray(physical['q_command'],float);subphase=physical.get('subphase','')
        if row in starts:
            query=starts[row]
            if trajectory is None or query['target']!=target:
                trajectory=JointTrajectory(physics[row-1]['q_command'],velocity=.175,acceleration=.5,jerk=8.)
                target=query['target'];release=False
            trajectory.set_target(query['raw_actions'][0])
        if row in query_by_row:
            expected=trajectory.step();base.require(np.array_equal(command,expected),'Numerical auditor requires bit-exact executed Ruckig q')
            tracker.observe(trajectory,expected,row,subphase)
        elif subphase=='learned_release_stop':
            base.require(trajectory is not None,'Release command has no continuous Ruckig state')
            if not release:trajectory.set_target(physics[row-1]['q_command']);release=True
            expected=trajectory.step();base.require(np.array_equal(command,expected),'Release command differs from continuous Ruckig')
            tracker.observe(trajectory,expected,row,subphase)
        else:tracker.observe(None,command,row,subphase)
        contact_active='observed_supervisor_phase' in physical
        check(row,command,subphase,contact_active,True)
    pending=int(len(physics) in event_by_row)
    if pending:
        event=event_by_row[len(physics)];last=learned[-1]
        base.require(last.get('failure_reason') and last.get('accepted') is True
                     and last['physics_row_end_exclusive']==len(physics),'Unexecuted gate event lacks a failed accepted query')
        base.require(trajectory is not None,'Unexecuted event has no trajectory')
        # If failure preceded the first tick, its target was not visited above.
        if last['physics_row_start']==len(physics):trajectory.set_target(last['raw_actions'][0])
        command=trajectory.step();subphase='learned_'+last['phase'].lower()
        tracker.observe(trajectory,command,len(physics),subphase)
        contact_active=(last['phase'] in {'CONTACT','HOLD','RETRACT'} or 'observed_supervisor_phase' in physics[-1])
        check(len(physics),command,subphase,contact_active,False)
    base.require(set(seen)==set(event_by_row),'Unverified numerical event')
    base.require(receipt['raw_rejected_ticks']==len(events) and receipt['numerical_interventions']==interventions,'Numerical receipt counts differ')
    # A non-gate physical failure can occur after comparison but before append.
    compared=receipt['compared_ticks'];minimum=max(0,len(physics)-3)
    base.require(type(compared) is int and minimum+pending<=compared<=minimum+1,'Numerical comparison coverage differs')
    return dict(numerical_gate_verified=True,raw_rejected_ticks=len(events),numerical_interventions=interventions,
                verified_numerical_events=verified,executed_rows_checked=len(physics),unexecuted_events=pending)


def audit_episode(directory,*,expected_checkpoint_metadata_sha256,binding_path=None,server_requests_path=None,raw_sensor_root=None):
    directory=Path(directory)
    report=base.audit_episode(directory,expected_checkpoint_metadata_sha256=expected_checkpoint_metadata_sha256,
        binding_path=binding_path,server_requests_path=server_requests_path,raw_sensor_root=raw_sensor_root)
    report['base_action_audit_schema']=report['schema'];report['schema']=SCHEMA
    sources=report['artifact_sha256']
    def read(name,rows=False):
        path=directory/name;data=path.read_bytes();sources[str(path.resolve())]=base.digest(data)
        return [json.loads(line) for line in data.splitlines()] if rows else json.loads(data)
    try:
        outcome=read('outcome.json');snapshot=read(outcome['input_snapshot']);folder=(directory/outcome['input_snapshot']).parent
        base.require(outcome.get('executed_controller_class')==CLASS and outcome.get('controller_source')==SOURCE,'Wrong numerical controller identity')
        base.require(outcome.get('numerical_gate_contract')==numerical_gate_contract()
                     and outcome.get('numerical_gate_contract_sha256')==contract_sha256(),'Numerical contract differs')
        from .run_numerical_isaac import PARENT_INVENTORY_SHA256
        base.require(outcome.get('parent_runtime_inventory_sha256')==PARENT_INVENTORY_SHA256,'Wrong inherited runtime inventory')
        base.require(outcome.get('original_execution_contract_sha256')==outcome['learned_policy_binding']['execution_contract_sha256'],'Original Ruckig contract changed')
        for name in MODULES:
            entry=snapshot['files']['source/pi05_v7/'+name];obj=folder/'objects'/entry['sha256']
            base.require(base.digest(obj.read_bytes())==entry['sha256']==base.digest(Path(__file__).with_name(name).read_bytes()),'Numerical source differs: '+name)
            sources[str(obj.resolve())]=entry['sha256']
        for key,expected in [('source/pi05_v7/numerical_gate_contract',outcome['numerical_gate_contract_file_sha256']),
                             ('source/pi05_v7/parent_runtime_inventory',PARENT_INVENTORY_SHA256)]:
            identity=snapshot['files'][key]['sha256'];obj=folder/'objects'/identity
            base.require(identity==expected and base.digest(obj.read_bytes())==identity,'Numerical public manifest binding differs')
            sources[str(obj.resolve())]=identity
            if 'numerical_gate_contract' in key:base.require(json.loads(obj.read_text())==numerical_gate_contract(),'Numerical contract object differs')
        receipt=read('numerical_gate_receipt.json');events=read('numerical_gate_decisions.jsonl',True)
        base.require(outcome['numerical_gate_receipt']==dict(path='numerical_gate_receipt.json',sha256=sources[str((directory/'numerical_gate_receipt.json').resolve())]),'Numerical receipt outcome binding differs')
        base.require(receipt['schema']=='wam_numerical_command_gate_receipt_v1' and receipt['controller_class']==CLASS,'Wrong numerical receipt schema/identity')
        base.require(receipt['numerical_gate_contract']==numerical_gate_contract() and receipt['numerical_gate_contract_sha256']==contract_sha256(),'Receipt contract differs')
        base.require(receipt['ledger_path']=='numerical_gate_decisions.jsonl' and receipt['ledger_sha256']==sources[str((directory/'numerical_gate_decisions.jsonl').resolve())],'Numerical ledger hash differs')
        base.require(receipt['controller_source_sha256']==base.digest(Path(__file__).with_name('numerical_learned_controller.py').read_bytes()),'Numerical controller source differs')
        base.require(receipt['flushed_on_run_exit'] is True and receipt['trajectory_changed'] is False and receipt['nominal_jerk_rad_s3']==8.,'Wrong numerical execution scope')
        report.update(replay_gate_records(read('learned_actions.jsonl',True),read('expert_trace.jsonl',True),events,receipt))
        report['numerical_source_snapshot_verified']=True
    except (ValueError,TypeError,KeyError,IndexError,OSError,RuntimeError) as error:
        report['passed']=report['evidence_consistent']=False;report['numerical_gate_verified']=False
        report['errors'].append('Numerical audit: '+str(error))
    report['audit_source_sha256']={name:base.digest(path.read_bytes()) for name,path in audit_source_paths().items()}
    for path,expected in sources.items():
        if base.digest(Path(path).read_bytes())!=expected:
            report['passed']=report['evidence_consistent']=False;report['errors'].append('Numerical evidence changed: '+path)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path)
    parser.add_argument('--checkpoint-metadata-sha256',required=True);parser.add_argument('--binding',type=Path);parser.add_argument('--server-requests',type=Path)
    args=parser.parse_args();result=audit_episode(args.directory,expected_checkpoint_metadata_sha256=args.checkpoint_metadata_sha256,binding_path=args.binding,server_requests_path=args.server_requests)
    (args.directory/'learned_actions_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result));raise SystemExit(0 if result['passed'] else 1)

if __name__=='__main__':main()
