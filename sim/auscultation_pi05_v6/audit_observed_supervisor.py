"""Independently replay acquired ADC/FK/RGB-D contact decisions without Isaac.

This checks evidence and continuous dwell accounting, not clinical suitability
or policy success. No missing high-rate sensor sample is reconstructed.
"""
import argparse
from dataclasses import asdict
import importlib
import json
from pathlib import Path

import numpy as np

from sim.auscultation_expert_layout_v1.phase_manager import PhaseManager, PhaseConfig, ContactObservation
from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
from sim.auscultation_vla_v2.hall import HallCalibration, HallPacket
from .dynamics_contract import sha256_file, supervisor_contract
from .observed_supervisor import VERSION
from .supervision import contact_load_estimate

SCHEMA='wam_observed_supervisor_replay_audit_v1'
DT=1/240


def require(condition,message):
    if not condition:raise ValueError(message)


def close(actual,expected,label,tolerance=1e-9):
    a,b=np.asarray(actual,float),np.asarray(expected,float)
    require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            and np.allclose(a,b,atol=tolerance,rtol=0),label+' mismatch')


def compare_state(actual,expected,label):
    expected=asdict(expected)
    require(set(actual)==set(expected),label+' fields differ')
    for key,value in expected.items():
        if isinstance(value,float):close(actual[key],value,label+'.'+key)
        else:require(actual[key]==value,label+'.'+key+' mismatch')


def replay(records,physics,frames,body_poses,registrations,configs,kin,cal,base):
    """Replay every saved manager call, including zero-time phase handoffs."""
    require(records,'Missing observed supervisor decisions')
    frame_map={row['index']:row for row in frames}
    body_map={row['frame_index']:row for row in body_poses}
    require(len(frame_map)==len(frames)==len(body_map),'Duplicate/missing camera body poses')
    managers={};last_rows={};counts={};max_hold={};transitions=[];active_targets=set();packets={}
    frames_by_physics={}
    for frame in frames:frames_by_physics.setdefault(frame['physics_row'],[]).append(frame)
    def bind_packet(pi,packet):
        if pi in packets:require(packet==packets[pi],'Conflicting ADC history at one physics sample')
        packets[pi]=packet
        for frame in frames_by_physics.get(pi,[]):
            require(frame['hall']==packet,'Supervisor ADC differs from policy camera observation')
    for name,registration in registrations.items():
        first=frames[0]
        require(registration['source_frame']==first['index'],'Task not bound to first body frame')
        close(registration['source_time_s'],first['time_s'],'Task registration time')
        body=np.asarray(body_map[first['index']]['body_reference_to_current_base'])
        close(registration['initial_body_reference_to_current_base'],body,'Initial body')
        task=np.eye(4);task[:3,3]=registration['initial_task_position'];task[:3,:3]=registration['initial_task_rotation']
        close(body@np.asarray(registration['task_pose_in_body_reference']),task,'Task registration transform')
    for index,row in enumerate(records):
        require(row['schema']=='wam_observed_supervisor_decision_v1' and row['index']==index,'Decision sequence changed')
        name=row['target'];require(name in registrations and name in configs,'Unregistered decision target')
        pi=row['physics_row'];require(type(pi) is int and 0<=pi<len(physics),'Invalid physics row')
        state=physics[pi];stamp=row['time_s'];dt=row['dt_s']
        require(state['target']==name,'Physics target changed')
        close(stamp,state['time_s'],'Decision timestamp')
        close(row['measured_q'],state['q'],'Measured q',1e-12)
        close(row['measured_dq'],state['dq'],'Measured dq',1e-12)
        require(row['phase_config']==configs[name],'Supervisor thresholds changed')
        if name not in managers:
            require(row['decision_kind']=='initialization' and dt==0,'First call must be zero-time initialization')
            managers[name]=PhaseManager(PhaseConfig(**configs[name]));counts[name]=0;max_hold[name]=0.
        elif row['decision_kind']=='initialization':
            require(dt==0 and name not in active_targets and pi>=last_rows[name],
                    'Unexpected reinitialization after active contact or reversed time')
        else:
            require(row['decision_kind']=='physics_tick','Invalid supervisor decision kind')
            close(dt,DT,'Physics update interval',1e-12)
            require(pi==last_rows[name]+1,'Missing/duplicated contact physics decision')
            close(stamp-physics[last_rows[name]]['time_s'],DT,'Elapsed physics interval',1e-8)
        if dt:active_targets.add(name)
        last_rows[name]=pi;counts[name]+=1
        geometry=row['geometry'];fi=geometry['frame_index'];require(fi in frame_map,'Missing geometry frame')
        frame=frame_map[fi];body=body_map[fi]
        require(frame['target']==name,'Wrong target camera frame')
        require(geometry['raw_sensor_sha256']==frame['raw_sha256'],'Geometry image identity changed')
        close(geometry['time_s'],frame['time_s'],'Geometry timestamp')
        close(body['time_s'],frame['time_s'],'Body timestamp')
        close(geometry['body_reference_to_current_base'],body['body_reference_to_current_base'],'Body transform')
        require(-1e-9<=stamp-frame['time_s']<=.15+1e-9,'Missing/stale geometry')
        require(frame['physics_row']<=pi,'Geometry came from future physics')
        # Contact updates precede the optional camera capture at this same tick.
        eligible=[f for f in frames if f['physics_row']<pi or
                  (row['decision_kind']=='initialization' and f['physics_row']<=pi)]
        require(eligible and max(f['index'] for f in eligible)==fi,'Supervisor did not use latest available camera frame')
        bind_packet(pi,row['raw_hall_packet'])
        packet=HallPacket(**row['raw_hall_packet']);hall=cal.decode(packet,now_s=stamp)
        pose=kin.tcp(row['measured_q'],hall.displacement_estimate_m);world=np.asarray(base)@pose
        prior=row['speed_reference'];pidx=prior['physics_row']
        require(type(pidx) is int and pidx==pi-1 and pidx>=0,'Speed requires preceding real physics row')
        close(prior['time_s'],physics[pidx]['time_s'],'Previous speed timestamp')
        close(stamp-prior['time_s'],DT,'Speed time interval',1e-8)
        close(prior['measured_q'],physics[pidx]['q'],'Previous measured q',1e-12)
        bind_packet(pidx,prior['raw_hall_packet'])
        previous_hall=cal.decode(HallPacket(**prior['raw_hall_packet']),now_s=prior['time_s'])
        previous_world=np.asarray(base)@kin.tcp(prior['measured_q'],previous_hall.displacement_estimate_m)
        close(prior['tcp_position_m'],previous_world[:3,3],'Previous measured TCP')
        speed=float(np.linalg.norm(world[:3,3]-previous_world[:3,3])/(stamp-prior['time_s']))
        task=np.asarray(geometry['body_reference_to_current_base'])@np.asarray(registrations[name]['task_pose_in_body_reference'])
        normal=-task[:3,2];error=pose[:3,3]-task[:3,3];gap=float(error@normal)
        observation=ContactObservation(contact_load_estimate(hall),hall.displacement_estimate_m,gap,
            float(np.linalg.norm(error-gap*normal)),
            float(np.rad2deg(np.arccos(np.clip(pose[:3,2]@task[:3,2],-1,1)))),speed,
            observation_age_s=max(stamp-frame['time_s'],stamp-hall.sample_time_s),
            tcp_position_m=tuple(world[:3,3]))
        saved=row['observation'];require(set(saved)==set(asdict(observation)),'Observation fields changed')
        for key,value in asdict(observation).items():
            if isinstance(value,bool):require(saved[key] is value,'Target validity changed')
            else:close(saved[key],value,'Observation '+key,1e-8)
        manager=managers[name];compare_state(row['prior_state'],manager.state,'Prior state')
        old=manager.state.phase.value;current=manager.update(observation,dt)
        compare_state(row['post_state'],current,'Post state')
        max_hold[name]=max(max_hold[name],current.hold_elapsed)
        if old!=current.phase.value:transitions.append(dict(target=name,time_s=stamp,previous=old,current=current.phase.value))
        if dt:
            require(state['observed_supervisor_phase']==current.phase.value,'Physics phase annotation changed')
            close(state['hold_elapsed_s'],current.hold_elapsed,'Physics hold credit')
            expected_phase='RETRACT' if current.phase.value=='DONE' else current.phase.value
            for acquired in frames_by_physics.get(pi,[]):
                require(acquired['target']==name and acquired['phase']==expected_phase,
                        'Acquired contact phase differs from replayed post-update supervisor')
    active_subphases={'contact_cycle','normal_unload','normal_released_retreat',
                     'normal_release_confirm','learned_contact','learned_hold',
                     'learned_retract','learned_release_stop'}
    expected_rows={i for i,row in enumerate(physics) if 'observed_supervisor_phase' in row
                   or row.get('subphase') in active_subphases}
    actual_rows={row['physics_row'] for row in records if row['dt_s']>0}
    require(expected_rows==actual_rows,'Incomplete active-physics coverage')
    return dict(decisions=len(records),targets=counts,max_continuous_hold_s=max_hold,
        final_states={k:asdict(v.state) for k,v in managers.items()},transitions=transitions,
        every_observation_recomputed=True,every_manager_update_replayed=True,all_active_ticks_covered=True)


def audit_episode(directory):
    root=Path(directory).resolve();artifacts={}
    report=dict(schema=SCHEMA,passed=False,errors=[],artifact_sha256=artifacts,
        source_supervisor_version=VERSION,
        source_supervisor_sha256=sha256_file(Path(__file__).with_name('observed_supervisor.py')),
        policy_acceptance='not_evaluated',scope='Acquired ADC/FK/RGB-D observations and contact supervisor decisions')
    def read(name,rows=False):
        path=(root/name).resolve();require(path.is_relative_to(root),'Artifact escapes trial')
        artifacts[str(path.relative_to(root))]=sha256_file(path)
        data=path.read_bytes()
        return [json.loads(line) for line in data.splitlines()] if rows else json.loads(data)
    try:
        outcome=read('outcome.json');manifest=read(outcome['input_snapshot'])
        require(artifacts[outcome['input_snapshot']]==outcome['input_snapshot_manifest_sha256'],'Snapshot manifest changed')
        def obj(key,module=None):
            identity=manifest['files'][key]['sha256']
            path=Path(outcome['input_snapshot']).parent/'objects'/identity
            require(sha256_file(root/path)==identity,'Snapshot object changed: '+key)
            artifacts[str(path)]=identity
            if module:require(sha256_file(importlib.import_module(module).__file__)==identity,'Audit dependency differs: '+key)
            return root/path
        for key,module in [('source/pi05_v6/observed_supervisor.py',__package__+'.observed_supervisor'),
            ('source/pi05_v6/supervision.py',__package__+'.supervision'),
            ('source/code/phase_manager.py','sim.auscultation_expert_layout_v1.phase_manager'),
            ('source/code/kinematics.py','sim.auscultation_expert_layout_v1.kinematics'),
            ('source/code/tool_frames.py','sim.auscultation_expert_layout_v1.tool_frames'),
            ('source/vla/hall.py','sim.auscultation_vla_v2.hall'),
            ('source/vla/force_sensor.py','auscultation.mechanical_sensor.force_feedback.force_sensor')]:obj(key,module)
        cal=HallCalibration.from_path(obj('source/vla/hall_calibration.json'))
        kin=OpenArmKinematics(obj('source/urdf'))
        ledger=read('supervisor_decisions.jsonl',True);receipt=read('supervisor_decisions_receipt.json')
        for owner in (outcome,receipt):
            require(owner['observed_supervisor_version']==VERSION and
                    owner['observed_supervisor_sha256']==report['source_supervisor_sha256'],'Observed supervisor identity mismatch')
        require(receipt['schema']=='wam_observed_supervisor_ledger_receipt_v1' and
                receipt['ledger_path']=='supervisor_decisions.jsonl' and
                receipt['ledger_sha256']==artifacts['supervisor_decisions.jsonl'] and
                receipt['decision_count']==len(ledger) and receipt['flushed_on_run_exit'] is True,'Ledger receipt invalid')
        require(outcome['observed_supervisor_ledger']==receipt,'Outcome ledger receipt mismatch')
        registrations=read('task_frame_registration.json');require(registrations==outcome['task_frame_registration'],'Task registration changed')
        base=receipt['world_from_left_base'];configs={}
        for name,registration in registrations.items():
            plan=json.loads(obj(name+'/plan').read_text())
            close(base,plan['world_from_left_base'],'Approved robot base',1e-7)
            point=next(p for p in plan['points'] if p['name']==name)
            target=(np.linalg.inv(np.asarray(plan['world_from_left_base']))@np.r_[point['position_world'],1])[:3]
            close(registration['initial_task_position'],target,'Approved target',1e-8)
            close(registration['initial_task_rotation'],np.asarray(base)[:3,:3].T@np.asarray(point['command_rotation_world']),'Approved tool orientation')
            gap=point.get('free_approach_start_gap_m',.06);config=supervisor_contract()
            config.update(precontact_gap_m=min(.04,gap),retract_clearance_m=min(.04,gap-.005));configs[name]=config
        require(outcome['phase_config']==supervisor_contract(),'Base supervisor thresholds changed')
        report['replay']=replay(ledger,read('expert_trace.jsonl',True),read('sensors/frames.jsonl',True),
            read('sensors/body_poses.jsonl',True),registrations,configs,kin,cal,base)
        if outcome.get('success') is True:
            require(set(report['replay']['final_states'])==set(registrations),'Successful outcome omits target decisions')
            require(all(state['success'] is True for state in report['replay']['final_states'].values()),
                    'Successful outcome disagrees with replayed terminal supervisor')
            require(all(value>=supervisor_contract()['hold_duration_s']-1e-9
                        for value in report['replay']['max_continuous_hold_s'].values()),
                    'Successful outcome lacks complete measured continuous hold')
            for point in outcome['points']:
                own=[r for r in ledger if r['target']==point['target']]
                entered=next(r['physics_row'] for r in own if r['post_state']['phase']=='CONTACT')
                active=[r['physics_row'] for r in own if r['dt_s']>0]
                require(active==list(range(entered+1,point['contact_end_physics_row'])),
                        'Successful contact interval has missing or extra supervisor updates')
        for name,identity in artifacts.items():require(sha256_file(root/name)==identity,'Evidence changed during audit: '+name)
        report['passed']=True
    except (ValueError,KeyError,TypeError,IndexError,OSError,StopIteration) as error:
        report['errors'].append(type(error).__name__+': '+str(error))
    report['audit_source_sha256']=sha256_file(__file__)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    parser.add_argument('--output',type=Path);args=parser.parse_args()
    report=audit_episode(args.directory)
    path=args.output or args.directory/'observed_supervisor_audit.json'
    with path.open('x') as stream:stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='artifact_sha256'},indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__':main()
