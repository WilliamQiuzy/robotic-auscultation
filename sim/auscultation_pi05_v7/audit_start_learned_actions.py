"""Additional independent boundary audit for declared-start learned episodes.

The original V7 auditor still verifies every service payload and Ruckig tick.
This audit additionally reconstructs the sole classical preparation and binds
its end to the first real learned query. It does not certify physical success.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import numpy as np
from . import audit_learned_actions as base_audit
from .policy_client import package_path

SCHEMA='wam_v7_bound_start_learned_action_audit_v1'
CLASS='sim.auscultation_pi05_v7.start_learned_controller.StartLearnedController'
SOURCE='pi05_absolute_joint_policy_with_bound_start_preparation'
DT=1/240.
require=base_audit.require
close=base_audit.close
vector=base_audit.vector


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*2**20),b''):h.update(block)
    return h.hexdigest()


def audit_source_paths():
    paths=base_audit.audit_source_paths()
    paths.update({'v7/'+name:Path(__file__).with_name(name) for name in (
        'audit_start_learned_actions.py','start_learned_controller.py','run_start_isaac.py','start_collection.py')})
    # These frozen sources implement the preparation command clock/route.
    for name in ('sim.curobo_expert_v1.timing','sim.curobo_expert_v1.controller','sim.curobo_expert_v1.withdrawal',
                 'sim.auscultation_skill_v1.controller','sim.expert_quality_20260924.candidate_controller',
                 'sim.expert_quality_20260924.candidate_planner','sim.auscultation_expert_layout_v1.controller'):
        paths[name.replace('.','/')+'.py']=Path(importlib.import_module(name).__file__)
    return paths


def verify_start_records(*,outcome,receipt,start,protocol,physics,frames,learned,raw_sensor_root):
    require(outcome.get('executed_controller_class')==CLASS and outcome.get('controller_source')==SOURCE,
            'Wrong declared-start controller identity')
    require(receipt.get('schema')=='wam_v7_learned_measured_start_receipt_v1'
            and receipt.get('executed_controller_class')==CLASS and receipt.get('controller_source')==SOURCE,
            'Wrong learned start receipt identity')
    require(receipt.get('complete') is True and receipt.get('qualification_passed') is False,
            'Missing actual first service request or false producer qualification')
    require(outcome.get('task_scope')=='full' and outcome.get('speed_scale')==.7
            and outcome.get('expert_fallbacks')==0 and outcome.get('training_dataset_eligible') is False
            and outcome.get('start_acceptance_audit_required') is True,'Incomplete learned-start task contract')
    sequence=outcome.get('sequence_execution',{})
    require(sequence.get('scene_instances')==1 and sequence.get('resets_between_points')==0,
            'Start sequence was reset')
    points=outcome['requested_points'];require(points and points[0]==start['target'],'First target differs from declared start')
    for key in ('episode_id','start_family','start_manifest_sha256','experiment_protocol_sha256'):
        require(receipt.get(key)==outcome.get(key),'Start receipt binding differs: '+key)
    require(receipt['target']==start['target'] and receipt['start_family']==start['start_family']
            and receipt['source_plan_sha256']==start['parent_plan_sha256'],'Start plan/family differs')
    require(start.get('geometry_preflight_passed') is True and start.get('requested_physics_profile')==
            {'position_iterations':32,'velocity_iterations':1,'armature_kg_m2':.02},'Unbound start geometry/profile')
    requested=vector(start['requested_start_q_rad'],(7,),'declared start')
    require(np.max(abs(requested))>.06,'Requested start is not a genuine nonzero family')
    close(receipt['requested_start_q_rad'],requested,'Receipt declared start')
    require(protocol.get('schema')=='wam_v7_frozen_start_family_protocol_v1'
            and protocol.get('actual_start_tolerance_rad')==.03
            and protocol.get('start_spec_sha256')==start['proposed_starts_sha256'],'Wrong start experiment protocol')
    family=[f for f in protocol['families'] if f['family_id']==start['start_family']]
    case=protocol['cases'][start['start_family']+'::'+start['target']]
    require(len(family)==1 and family[0]['split']==start['split'],'Start family split differs')
    close(family[0]['requested_q_rad'],requested,'Protocol start')
    require(all(case.get(key)==value for key,value in dict(start_manifest_sha256=outcome['start_manifest_sha256'],
        source_plan_sha256=start['parent_plan_sha256'],target=start['target'],start_family=start['start_family']).items()),
        'Frozen protocol case differs')
    require(receipt.get('start_position_tolerance_rad')==.03 and receipt.get('start_speed_limit_rad_s')==.03
            and receipt.get('preparation_excluded_from_learned_interval') is True,'Wrong preparation limits/scope')
    origin=receipt['preparation_origin']
    origin_command=vector(origin['retained_command_q_rad'],(7,),'preparation retained origin')
    require(origin['physics_row_start']==0 and np.max(abs(origin_command))<=.03
            and np.max(abs(vector(origin['measured_q_rad'],(7,),'preparation measured origin')))<=.03,
            'Preparation did not begin at actual zero-home neighborhood')
    vector(origin['measured_dq_rad_s'],(7,),'preparation measured dq')
    require(physics and abs(physics[0]['time_s']-origin['time_s']-DT)<1e-8,'Preparation source clock differs')
    from sim.curobo_expert_v1.timing import smooth_commands
    commands=smooth_commands(np.vstack([origin_command,requested]),speed_limit=.175,acceleration_limit=1.,
        jerk_limit=1.5,maximum_seconds=300.,minimum_seconds=1/.7,maximum_compute_seconds=15.)
    n=len(commands);end=n+252
    require(receipt['preparation_physics_range']==[0,n] and receipt['settle_physics_range']==[n,n+240]
            and receipt['observation_physics_range']==[n+240,end]
            and receipt['learned_interval_start_physics_row']==end,'Classical/learned boundary differs')
    require(len(physics)>=end and learned,'Missing learned execution after preparation')
    for index,row in enumerate(physics):
        if index>=end:
            require(row.get('preparation_role') is None,'Classical preparation reappeared inside learned sequence')
            continue
        role='move_to_declared_start' if index<n else 'settle_at_declared_start' if index<n+240 else 'start_observation'
        require(row.get('preparation_role')==role and row['subphase']==('initial_home' if index<n else 'settle')
                and row.get('recovery') is False and row['target']==start['target'],'Unexplained preparation command role')
        close(row['q_command'],commands[index] if index<n else requested,'Classical preparation replay',atol=1e-10)
    query=learned[0];first=receipt['first_query']
    require(query['index']==0 and query['phase']=='FREE' and query['target']==start['target']
            and query['physics_row_start']==end,'First policy query escaped start boundary')
    require(first['role']=='first_learned_policy_decision' and first['requested_policy_phase']=='FREE'
            and first['physics_row_start']==end and first['observation_physics_row']==end-1
            and first['learned_query_index']==0 and first['model_request_receipt_present'] is True
            and first['request_id']==query['inference']['request_id'],'First actual model request differs')
    frame=frames[query['observation_frame']];last=physics[end-1]
    require(frame['index']==query['observation_frame']==first['observation_frame'] and frame['physics_row']==end-1
            and frame['target']==start['target'] and frame['phase']==first['acquired_observation_phase']
            and frame['eligible'] is False and first['acquired_frame_eligible'] is False,'First raw acquisition identity differs')
    for stamp in (frame['time_s'],first['time_s'],query['time_s']):
        close(stamp,last['time_s'],'First measured/request clock',atol=1e-9)
    measured=vector(last['q'],(7,),'measured first q');dq=vector(last['dq'],(7,),'measured first dq')
    require(np.max(abs(measured-requested))<=.03 and np.max(abs(measured))>.03
            and np.max(abs(dq))<=.03,'Measured start tolerance/speed failed')
    for q in (frame['q'],first['measured_q_rad'],query['measured_q']):close(q,measured,'First measured q',atol=1e-12)
    for d in (frame['dq'],first['measured_dq_rad_s']):close(d,dq,'First measured dq',atol=1e-12)
    close(first['retained_command_q_rad'],last['q_command'],'First retained command')
    require(all(f['eligible'] is False for f in frames if f['physics_row']<end),
            'Preparation acquisition was relabeled eligible')
    require(frame['raw_file']==first['raw_sensor_file'] and frame['raw_sha256']==first['raw_sensor_sha256']
            ==query['raw_sensor_sha256'],'First raw file/query hash differs')
    root=Path(raw_sensor_root).resolve();raw=(root/frame['raw_file']).resolve()
    require(raw.is_relative_to(root) and raw.is_file() and sha(raw)==frame['raw_sha256'],'First raw bytes differ')
    return dict(first_query_bound=True,preparation_command_replay_verified=True,
        classical_preparation_steps=end,first_query_physics_row=end,first_frame=frame['index'],
        initial_measured_q_rad=measured.tolist(),initial_measured_dq_rad_s=dq.tolist(),
        preparation_not_learned=True,continuous_sequence_preparation_count=1)


def verify_start_release(physics_row,frame,calibration):
    values=vector([physics_row[name] for name in ('hall_N','compression_m','diaphragm_N','lip_N',
        'contact_rim_N','frame_N','forbidden_N','carriage_other_N')],(8,),'actual release physics')
    require(np.all(values>=0) and values[0]<.2 and values[1]<=.00015 and sum(values[2:])<.2,
            'Actual start physical release gate failed')
    packet=base_audit.HallPacket(**frame['hall'])
    hall=calibration.decode(packet,now_s=frame['time_s'],max_age_s=.05)
    load=0. if hall.displacement_estimate_m<.0001 else hall.spring_force_estimate_n
    require(load<.2 and hall.displacement_estimate_m<=.00015,'Actual start ADC release gate failed')
    return dict(adc_release_verified=True,true_physics_release_verified=True,
                hall_sample_time_s=hall.sample_time_s,hall_displacement_estimate_m=hall.displacement_estimate_m)


def audit_episode(directory,*,expected_checkpoint_metadata_sha256,binding_path=None,
                  server_requests_path=None,raw_sensor_root=None):
    root=Path(directory).resolve()
    report=dict(schema=SCHEMA,passed=False,errors=[],expected_checkpoint_metadata_sha256=expected_checkpoint_metadata_sha256)
    artifacts={}
    try:
        base=base_audit.audit_episode(root,expected_checkpoint_metadata_sha256=expected_checkpoint_metadata_sha256,
            binding_path=binding_path,server_requests_path=server_requests_path,raw_sensor_root=raw_sensor_root)
        report.update(base);report['base_action_audit_schema']=base['schema'];report['schema']=SCHEMA
        artifacts.update(base.get('artifact_sha256',{}))
        def read(path,rows=False):
            path=Path(path);artifacts[str(path)]=sha(path);data=path.read_text()
            return [json.loads(line) for line in data.splitlines()] if rows else json.loads(data)
        outcome=read(root/'outcome.json');receipt_path=root/'learned_start_receipt.json'
        require(outcome.get('learned_start_receipt')==dict(path=receipt_path.name,sha256=sha(receipt_path)),
                'Outcome start receipt binding differs')
        receipt=read(receipt_path);snapshot_path=root/outcome['input_snapshot'];snapshot=read(snapshot_path)
        require(receipt['input_snapshot_manifest_sha256']==outcome['input_snapshot_manifest_sha256'],
                'Start receipt snapshot differs')
        def snap(key,current=None):
            identity=snapshot['files'][key]['sha256'];path=snapshot_path.parent/'objects'/identity
            require(sha(path)==identity,'Start snapshot object changed: '+key)
            artifacts[str(path)]=identity
            if current is not None:require(sha(current)==identity,'Actually imported start source differs: '+key)
            return path,identity
        start_path,start_sha=snap('source/v7/start_manifest');protocol_path,protocol_sha=snap('source/v7/start_protocol')
        require(start_sha==outcome['start_manifest_sha256'] and protocol_sha==outcome['experiment_protocol_sha256'],
                'Explicit start/protocol SHA differs')
        start=read(start_path);protocol=read(protocol_path)
        require(protocol_path.read_bytes()==(json.dumps(protocol,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode(),
                'Noncanonical frozen protocol')
        require(snap(start['target']+'/plan')[1]==start['parent_plan_sha256'],'Actual first plan differs')
        paths=audit_source_paths()
        for key,path in paths.items():
            if key.startswith(('v6/','v7/')):snapshot_key='source/pi05_'+key
            elif key=='sim/auscultation_expert_layout_v1/controller.py':snapshot_key='source/code/controller.py'
            else:snapshot_key='source/runtime/'+key.removeprefix('sim/')
            snap(snapshot_key,path)
        for name in ('learned_controller.py','observed_supervisor.py'):
            snap('source/pi05_v6/'+name,package_path('v6')/name)
        declared={key:sha(path) for key,path in {
            'pi05_v7/run_start_isaac.py':Path(__file__).with_name('run_start_isaac.py'),
            'pi05_v7/start_learned_controller.py':Path(__file__).with_name('start_learned_controller.py'),
            'pi05_v7/start_collection.py':Path(__file__).with_name('start_collection.py'),
            'pi05_v6/learned_controller.py':package_path('v6')/'learned_controller.py',
            'pi05_v6/execution.py':package_path('v6')/'execution.py'}.items()}
        require(outcome['start_runtime_source_hashes']==receipt['start_runtime_source_hashes']==declared,
                'Start runtime source declaration differs')
        require(receipt['controller_source_sha256']==declared['pi05_v7/start_learned_controller.py'],
                'Start controller receipt source differs')
        required_artifacts=('expert_trace.jsonl','sensors/frames.jsonl','learned_actions.jsonl')
        require(set(receipt['artifact_sha256'])==set(required_artifacts),'Incomplete start receipt artifact map')
        for name in required_artifacts:require(sha(root/name)==receipt['artifact_sha256'][name],'Start receipt artifact differs: '+name)
        physics=read(root/'expert_trace.jsonl',True);frames=read(root/'sensors/frames.jsonl',True)
        learned=read(root/'learned_actions.jsonl',True);controls=read(root/'control_commands.json')
        require(controls and controls[0]['physics_row']==0,'No bound initial preparation command')
        close(controls[0]['previous_q_command'],receipt['preparation_origin']['retained_command_q_rad'],
              'Initial command origin',atol=1e-12)
        close(controls[0]['time_s'],receipt['preparation_origin']['time_s'],'Initial command time',atol=1e-9)
        for field,key in (('q','measured_q_rad'),('dq','measured_dq_rad_s')):
            close(controls[0]['observation'][field],receipt['preparation_origin'][key],'Initial measured '+field,atol=1e-12)
        report['start_boundary']=verify_start_records(outcome=outcome,receipt=receipt,start=start,protocol=protocol,
            physics=physics,frames=frames,learned=learned,raw_sensor_root=raw_sensor_root or root/'sensors')
        cal=base_audit.HallCalibration.from_path(snap('source/vla/hall_calibration.json')[0])
        frame=frames[learned[0]['observation_frame']]
        report['start_boundary'].update(verify_start_release(physics[learned[0]['physics_row_start']-1],frame,cal))
        report['start_boundary_verified']=True;report['start_source_snapshot_verified']=True
        report.update(start_family=start['start_family'],experiment_protocol_sha256=protocol_sha,start_manifest_sha256=start_sha)
        report['audit_source_sha256']={key:sha(path) for key,path in paths.items()}
        require(all(sha(path)==expected for path,expected in artifacts.items()),'Evidence changed during start audit')
        report['passed']=bool(base['passed'])
    except (ValueError,KeyError,TypeError,OSError,IndexError,RuntimeError) as error:
        report['passed']=False;report['errors'].append(type(error).__name__+': '+str(error))
    report['artifact_sha256']=artifacts
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    p.add_argument('--expected-checkpoint-metadata-sha256',required=True)
    p.add_argument('--binding',type=Path);p.add_argument('--server-requests',type=Path)
    p.add_argument('--raw-sensor-root',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report=audit_episode(a.directory,expected_checkpoint_metadata_sha256=a.expected_checkpoint_metadata_sha256,
        binding_path=a.binding,server_requests_path=a.server_requests,raw_sensor_root=a.raw_sensor_root)
    with a.output.open('x') as stream:stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=report['passed'],errors=report['errors'])))
    raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__':main()
