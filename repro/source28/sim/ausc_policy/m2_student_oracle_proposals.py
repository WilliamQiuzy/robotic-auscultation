"""Genuine TRAIN histories for unexecuted CPU-oracle diagnostics.

This is deliberately not a native demonstration reader or training loader.
Restored command dynamics and proposed setter cells are computational evidence;
they establish neither future measured tracking nor a successful contact cycle.
"""
from copy import deepcopy
from pathlib import Path
import os
import sys

import numpy as np

from . import m2_student_recovery_prefix as student
from . import m2_given_contact_recovery as recovery
from .retainer_contact import ALL_FORCE_GROUPS

rt=student.rt
need=rt.need
SCHEMA='m2_student_cpu_oracle_proposed_block_v1'


def authenticate_source(source):
    need(isinstance(source,dict) and set(source)=={'config','owner','child','native'},
        'Explicit original student source references required')
    config=rt.read_bound(source['config'])
    meta=config.get(student.adapter.ROUTE_KEY,{}).get('metadata',{})
    need(meta.get('split')=='train','CPU oracle sources must be TRAIN before native payload access')
    return student.authenticate(*(source[k] for k in ('config','owner','child','native')))


def query_at(binding,row):
    """Use actual input observations and the pre-query retained command seam."""
    need(type(row) is int and row>=12 and row%12==0,'Actual complete query boundary required')
    meta=binding['metadata'];need(meta.get('split')=='train','TRAIN-only oracle inputs required')
    reference=rt.pose.require_route(meta)
    requests=[r for r in binding['requests'] if r.get('context',{}).get('physics_row')==row]
    need(len(requests)==1,'One authenticated actual request at the selected query is required')
    request=requests[0];events=binding['events']
    need(row<len(events),'Actual observed-before row is missing')
    context=request['context'];stamp=rt.shared.contract._scalar(context['observed_at_s'],'actual query time')
    need(context['context'][-1]=='APPROACH' and abs(stamp-row/240.)<=1e-8,
        'Original APPROACH prerequisite required; unsupported FREE queries are rejected')
    for event in events[row-12:row]:
        need(event.get('state_audit')=='completed' and event.get('tick_error') is None and
            event.get('physical_violations')==[] and event.get('actual_control_owner')=='ACT_L2' and
            event.get('controller_before',{}).get('subphase') in ('given_l2_motion','given_l2_history_wait') and
            event.get('controller_before',{}).get('actor',{}).get('phase')=='APPROACH',
            'Actual causal history crosses an incomplete/failed/different-owner native tick')
    history=np.asarray(request['history'],float);times=np.asarray(request['history_times'],float)
    need(history.shape==(2,17) and times.shape==(2,), 'Actual H2 observations required')
    for j,index in enumerate((row-12,row)):
        observed=events[index]['observed_before']
        encoded=rt.shared.contract.encode_state(observed['q'],reference.pose_base,1.,metadata=meta)
        need(observed['physics_row']==index and abs(observed['time_s']-index/240.)<=1e-8 and
            times[j]==observed['time_s'] and np.array_equal(history[j],encoded),
            'H2 frame/clock differs from the actual native observation')
    measured=events[row]['observed_before']['q']
    need(request['measured_q']==measured and events[row]['observed_before']['time_s']==stamp,
        'Actual query measured origin or clock differs')
    previous=events[row-1];native=previous['native_row']
    need(previous['physics_row']==row-1 and abs(previous['time_s']-stamp)<=1e-8 and
        np.array_equal(native['q'],measured), 'Previous completed measured state differs from query input')
    snapshot=deepcopy(previous['interpolator']['input'])
    commands=np.asarray([e['q_command'] for e in events[row-3:row]],float)
    need(commands.shape==(3,7) and np.isfinite(commands).all() and
        np.array_equal(commands[-1],snapshot['fields']['current_position']),
        'Pre-query command history and retained interpolator seam differ')
    forces=np.asarray([native[k+'_N'] for k in ALL_FORCE_GROUPS],float)
    need(np.isfinite(forces).all() and np.all(forces>=0.),'Actual nonnegative physical force ledger required')
    query=dict(measured_q=deepcopy(measured),measured_dq=deepcopy(native['dq']),
        history=history.tolist(),history_times=times.tolist(),time_s=stamp,physics_row=row,
        reference_sha256=reference.sha256,context=deepcopy(context['context']),
        hall_N=native['hall_N'],physical_contact_N=float(forces.sum()),compression_m=native['compression_m'],
        metadata=deepcopy(meta))
    recovery._validate_query(query,reference)
    return query,snapshot,commands


def proposed_block(proposal,query):
    """Package proposed setters, never applied commands or executed futures."""
    preview=proposal.get('preview',{})
    need(proposal.get('kind')=='recovery_proposal' and
        preview.get('all_240hz_commands_verified') is True and preview.get('physics_executed') is False,
        'Complete computational preview of unexecuted teacher proposal required')
    cells=np.asarray(preview.get('cells'),float)
    origin=rt.shared.contract._array(query['measured_q'],(7,),'actual fixed measured origin')
    need(cells.ndim==2 and cells.shape[1:]==(7,) and 0<len(cells)<=2400 and np.isfinite(cells).all(),
        'Finite bounded proposed raw setter cells required')
    count=min(20,len(cells));relative=np.zeros((20,7),dtype=np.float64)
    relative[:count]=cells[:count]-origin
    return dict(schema=SCHEMA,source_observed_history='actual_authenticated_TRAIN_student_query',
        proposed_absolute_setters=cells[:count].copy(),proposed_relative_setters=relative,
        proposed_is_pad=np.arange(20)>=count,valid_proposed_cells=count,
        actions_executed=False,physics_qualified=False,training_eligible=False,
        interpretation='CPU raw setter proposal; no future measured observations or native contact qualification')


def diagnose(source,plan_config_ref,rows,output):
    """Finite same-scene CPU proposals; no simulator, native labels or trainer."""
    need(sys.platform=='linux' and os.environ.get('CUDA_VISIBLE_DEVICES')=='',
        'Linux CPU-only proposal diagnostic required')
    need(isinstance(rows,list) and 1<=len(rows)<=6 and len(rows)==len(set(rows)) and
        all(type(r) is int for r in rows),'One through six declared distinct actual queries required')
    binding=authenticate_source(source)  # Reject non-TRAIN before native reads.
    reference=rt.pose.require_route(binding['metadata'])
    plan=rt.pose.bound_json(reference.snapshot()['sources']['fresh_plan'])
    config=rt.read_bound(plan_config_ref)
    from .geometry_epoch import require_geometry_artifacts,contact_frame
    from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
    from sim.auscultation_expert_layout_v1.posture import PostureLimits,PostureChecker
    kin=OpenArmKinematics(contact_frame=contact_frame)
    limits=PostureLimits.from_dict(config['posture_limits'],kin)
    need(config.get('collision_profile')=='full' and limits.to_dict()==plan['posture_limits'] and
        config['patient_receipt']==reference.snapshot()['sources']['patient_receipt'] and
        np.array_equal(config['world_from_left_base'],plan['world_from_left_base']) and
        np.array_equal(config['world_from_left_base'],reference.snapshot()['world_from_base']),
        'Original full collision profile, posture, installed patient and base required')
    geometry=require_geometry_artifacts(config['collision_directory'],config)
    need(geometry==plan['geometry_artifacts'] and
        rt.reference(config['geometry_source'])['sha256']==config['geometry_source_sha256'],
        'Actual canonical plan geometry bytes differ')
    from .contact_collision import ContactCollisionChecker
    from sim.auscultation_expert_layout_v1.plan_low_elbow import checked_transfer
    base=np.asarray(config['world_from_left_base'],float)
    checker=PostureChecker(ContactCollisionChecker(kin,base,config['collision_directory'],config,profile='full'),base,limits)
    output=Path(output).resolve();need(not output.exists(),'Fresh CPU diagnostic output required');output.mkdir(parents=True)
    rt.write_new(output/'REQUEST.json',dict(source=source,plan_config=plan_config_ref,physics_rows=rows,
        generator=rt.reference(__file__),actions_executed=False,physics_qualified=False,training_eligible=False))
    outcomes=[];labels=[]
    for row in rows:
        outcome=dict(physics_row=row,proposal_accepted=False,actions_executed=False,physics_qualified=False,training_eligible=False)
        try:
            query,snapshot,commands=query_at(binding,row)
            trajectory=rt.handoff.restore_trajectory(snapshot)
            rng=np.random.default_rng(65)
            def verify(q):
                need(limits.evaluate(kin,base,q,contact=False)['clear'],'Original posture rejected CPU proposal')
                checked=checker.check(q,clearance=.003,allow_patient_contact=False,
                    compression=max(0.,float(query['compression_m'])))
                need(checked.get('clear') is True and np.isfinite(checked.get('minimum_distance_m',np.nan)) and
                    checked['minimum_distance_m']>=.003-1e-9,'Original collision rejected CPU proposal')
            proposal=recovery.propose_recovery(query,trajectory,commands,np.asarray(plan['transfer_q'][-1]),
                reference=reference,planner=lambda start,goal,deadline:checked_transfer(checker,limits,start,goal,kin,base,rng,deadline),
                verify_command=verify,numeric_detours=True,all_joint_detours=True)
            saved=rt.write_new(output/f'query-{row}.json',proposal)
            outcome['computational_proposal']=saved
            need(proposal['kind']=='recovery_proposal',proposal.get('reason','CPU oracle abstained'))
            block=proposed_block(proposal,query)
            labels.append(dict(query=query,block=block))
            outcome.update(proposal_accepted=True,valid_proposed_cells=block['valid_proposed_cells'])
        except (ValueError,KeyError,TypeError,RuntimeError,TimeoutError) as error:
            outcome['rejection']=type(error).__name__+': '+str(error)
        outcomes.append(outcome)
        rt.write_new(output/f'outcome-{row}.json',outcome)
    arrays=None
    if labels:
        path=output/'UNEXECUTED_PROPOSALS.npz'
        with path.open('xb') as stream:
            np.savez_compressed(stream,source_observed_histories=np.asarray([v['query']['history'] for v in labels]),
                source_observation_times=np.asarray([v['query']['history_times'] for v in labels]),
                source_physics_rows=np.asarray([v['query']['physics_row'] for v in labels],dtype=np.int64),
                proposed_relative_setters=np.asarray([v['block']['proposed_relative_setters'] for v in labels]),
                proposed_is_pad=np.asarray([v['block']['proposed_is_pad'] for v in labels]))
        arrays=rt.reference(path)
    for ref in [*source.values(),plan_config_ref]:rt.bound_path(ref)
    reference.verify_sources();need(require_geometry_artifacts(config['collision_directory'],config)==geometry,
        'Geometry changed during CPU diagnostic')
    return rt.write_new(output/'COMPLETE.json',dict(schema='m2_student_cpu_oracle_diagnostic_v1',
        complete=True,outcomes=outcomes,attempted_queries=len(rows),accepted_proposals=len(labels),
        arrays=arrays,actions_executed=False,physics_qualified=False,training_eligible=False,
        source_subject_id=binding['metadata']['subject_id'],source_split='train',
        source_episode_verdict_unchanged=True,gpu_launched=False))
