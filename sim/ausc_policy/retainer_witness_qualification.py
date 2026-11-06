"""CPU-only independent issuer for a fresh, terminal native contact oracle.

No simulator is imported or launched. The original 72-case source/owner chain,
raw callback journal, ordered native points, ticks and stored residuals must
agree. Analytic sector planes and full coordinate solves independently check
p+s*n and front/side/back controls before the unchanged runtime classifier is
compared. Only shape-0 convexHull/staticTriangleMesh semantics are qualified.
"""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import itertools
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from . import contact_semantics_oracle as oracle
from . import m2_given_contact_runtime as runtime
from . import retainer_witness as witness
from .geometry_epoch import GEOMETRY_SOURCE_SHA256

need=runtime.need
TOL=.0001
GUARD=float(16*np.finfo(np.float32).eps)
ALPHAS=(-1.,-.5,0.,.5,1.)
SOURCE='omni.physx.IPhysx.get_rigidbody_transformation'


def rigid(value):
    m=np.asarray(value,dtype=float)
    need(m.shape==(4,4) and np.isfinite(m).all() and np.allclose(m[3],[0,0,0,1],atol=1e-10,rtol=0) and
         np.allclose(m[:3,:3].T@m[:3,:3],np.eye(3),atol=1e-7,rtol=0) and
         abs(np.linalg.det(m[:3,:3])-1.)<=1e-7,'Invalid native rigid frame')
    return m


def analytic_geometry():
    """Independent six equations, in authored front/back/lower/outer/upper/inner order."""
    a=np.pi/32;theta=2*a;c,s=np.cos(a),np.sin(a)
    normals=np.array([[0,0,-1],[0,0,1],[0,-1,0],[c,s,0],[-np.sin(theta),np.cos(theta),0],[-c,-s,0]])
    offsets=np.array([.047,-.0455,0,.023*c,0,-.021*c])
    xy=[(r*np.cos(t),r*np.sin(t)) for r,t in [(.021,0),(.023,0),(.023,theta),(.021,theta)]]
    vertices=np.array([(x,y,z) for z in (-.047,-.0455) for x,y in xy])
    return normals,offsets,vertices


def independent_feature(matrix,position,normal):
    m=rigid(matrix);p=np.asarray(position,float);n=np.asarray(normal,float)
    need(p.shape==n.shape==(3,) and np.isfinite(p).all() and np.isfinite(n).all() and
         abs(n@n-1.)<=GUARD,'Finite raw native unit normal and witness required')
    local=np.linalg.solve(m,np.r_[p,1.])[:3]
    direction=np.linalg.solve(m,np.r_[-n,0.])[:3]
    planes,offsets,_=analytic_geometry();residuals=planes@local-offsets;scores=planes@direction
    radial=np.hypot(local[0],local[1])
    passed=bool(abs(local[2]+.047)<=TOL and .021*np.cos(np.pi/32)-TOL<=radial<=.023+TOL and
        (residuals<=TOL).all() and abs(residuals[0])<=TOL and scores[0]>GUARD and
        (scores[0]-scores[1:]>GUARD).all())
    return dict(passed=passed,local=local,plane_residuals_m=residuals,normal_scores=scores)


def checked_cases(specs):
    controls=list(itertools.product(('front','side','back'),(0.,.00015,.00035),(0.,.02),(False,True),(False,True)))
    need(isinstance(specs,list) and len(specs)==72,'Full original 72-case denominator required')
    planes,_,vertices=analytic_geometry()
    for i,(c,control) in enumerate(zip(specs,controls)):
        feature,gap,speed,transformed,reverse=control;cid='case_'+str(i).zfill(3);root='/World/Cases/'+cid
        need(c.get('id')==cid and tuple(c.get(k) for k in ('feature','gap_m','speed_m_s','transformed','reverse_creation'))==control,
             'Original ordered oracle controls changed')
        need(c.get('body_path')==root+'/Wedge' and c.get('shape_path')==root+'/Wedge/Shape' and c.get('plate_path')==root+'/Plate',
             'Original convex/static fixture paths changed')
        r=Rotation.from_euler('xyz',[19,27,31],degrees=True) if transformed else Rotation.identity()
        translation=np.array([(i%9)*.25,(i//9)*.25,1.]);face_index={'front':0,'side':3,'back':1}[feature]
        ids={'front':[0,1,2,3],'side':[1,2,5,6],'back':[4,5,6,7]}[feature]
        normal=planes[face_index];center=vertices[ids].mean(axis=0);axis=r.as_matrix()@normal
        face=c.get('support_face',{})
        for actual,expected in [(face.get('outward_local'),normal),(face.get('center_local'),center),
            (face.get('support_m'),normal@center),(c.get('initial_body_position_world'),translation),
            (c.get('initial_body_quaternion_xyzw'),r.as_quat()),(c.get('initial_velocity_world'),speed*axis),
            (c.get('wedge_to_plate_axis_world'),axis)]:
            need(np.allclose(np.asarray(actual,float),expected,rtol=0,atol=1e-12),'Authored known fixture geometry changed')
        plane=r.as_matrix()@center+translation+gap*axis
        ref=np.array([0.,0.,1.]) if abs(axis[2])<.9 else np.array([1.,0.,0.])
        u=np.cross(axis,ref);u/=np.linalg.norm(u);v=np.cross(axis,u)
        corners=np.array([plane+sx*.06*u+sy*.06*v for sx,sy in [(-1,-1),(1,-1),(1,1),(-1,1)]],np.float32).astype(float)
        plate_normal=np.cross(corners[2]-corners[0],corners[1]-corners[0]);plate_normal/=np.linalg.norm(plate_normal)
        need(np.allclose(c['plate_origin_world'],corners.mean(axis=0),atol=1e-12,rtol=0) and
             np.allclose(c['plate_normal_world'],plate_normal,atol=1e-12,rtol=0),'Float32 static triangle fixture changed')
    return {c['id']:c for c in specs}


def checked_frame(sample,c,label):
    need(isinstance(sample,dict) and sample.get('available') is True and sample.get('source')==SOURCE and
         sample.get('path')==c['body_path'] and sample.get('sample')==label and
         sample.get('contact_generation_pose_claimed') is False and not sample.get('error'),
         'Missing actual native '+label+' rigid sample')
    return rigid(sample.get('world_from_wedge'))


def journal_reconstruction(journal,specs):
    events=[];ticks=[];pending=[];case_ids=set(specs)
    for row in journal:
        if row.get('kind')=='contact_callback':
            event=deepcopy(row.get('event'))
            need(isinstance(event,dict) and event.get('event_index')==len(events),'Noncontiguous callback journal event index')
            need(event.get('case') in case_ids,'Unknown native callback case')
            if event.get('execution_stage')=='controlled_step':pending.append(event['event_index'])
            else:need(event.get('execution_stage')=='initialization_reset' and not ticks and event.get('physics_row') is None,
                      'Unexpected initialization/native callback stage')
            events.append(event)
        elif row.get('kind')=='post_step':
            tick=row.get('tick',{});step=len(ticks)
            need(step<12 and tick.get('physics_row')==step and row.get('updated_event_indices')==pending and
                 tick.get('events')==len(pending) and set(row.get('post',{}))==case_ids,
                 'Native journal post-step event/tick coverage differs')
            for i in pending:
                need(events[i].get('physics_row')==step,'Native callback is assigned to the wrong physics tick')
                events[i]['post']=deepcopy(row['post'][events[i]['case']])
            for cid,sample in row['post'].items():checked_frame(sample,specs[cid],'post_step_current')
            before=tick.get('time_before_s');after=tick.get('time_after_s')
            need(type(before) in (int,float) and type(after) in (int,float) and np.isfinite([before,after]).all() and
                 before>=0 and abs(after-before-1/240)<=1e-7 and
                 (not ticks or abs(before-ticks[-1]['time_after_s'])<=1e-10),'Native physics clock mismatch')
            ticks.append(tick);pending=[]
        else:raise ValueError('Native journal callback error or unknown entry')
    need(len(ticks)==12 and not pending,'Missing native journal controlled ticks')
    return events,ticks


def analyze(cases,events,journal,ticks,residuals):
    specs=checked_cases(cases);rebuilt,journal_ticks=journal_reconstruction(journal,specs)
    need(rebuilt==events and ticks==journal_ticks,'Raw native journal reconstruction differs from archived contacts/ticks')
    need(isinstance(residuals,list) and len(events)==len(residuals) and events,'Raw/residual event count differs')
    counts={f:dict(positive=0,accepted=0) for f in ('front','side','back')};coverage=set();positive_controls=set()
    case_counts={c['id']:dict(case=c['id'],**{k:c[k] for k in ('feature','gap_m','speed_m_s','transformed','reverse_creation')},
        contacts=0,positive_contacts=0,control_status='available',unavailable_counts={},observed_retainer_pair_indices=[0]) for c in cases}
    controlled=warmup=0;midpoint=raw_control=ring_error=plate_error=classifier_error=stored_error=0.
    for i,(e,stored) in enumerate(zip(events,residuals)):
        need(e.get('event_index')==i and e.get('case') in specs,'Native event index/case mismatch')
        c=specs[e['case']];raw=e['raw'];is_control=e['execution_stage']=='controlled_step'
        need(raw.get('colliders')==[c['shape_path'],c['plate_path']],'Only actual retainer shape-0 pair is qualified')
        need(raw.get('native_fields_missing')==[] and raw.get('native_field_errors')=={} and
             raw.get('header_field_errors')=={} and raw.get('contact_generation_pose_claimed',False) is False and
             all(type(raw.get(k)) is int and 0<=raw[k]<4294967295 for k in ('face_index0','face_index1')),
             'Missing/malformed raw native contact/header fields')
        p=np.asarray(raw['position_world'],float);n=np.asarray(raw['normal_world'],float);impulse=np.asarray(raw['impulse_Ns'],float)
        separation=raw['separation_m']
        need(p.shape==n.shape==impulse.shape==(3,) and np.isfinite([p,n,impulse]).all() and
             type(separation) in (int,float) and np.isfinite(separation) and abs(n@n-1.)<=GUARD,
             'Finite unmodified native contact vectors required')
        force=float(np.linalg.norm(impulse)*240);positive=force>0
        need(stored.get('case')==c['id'] and stored.get('feature')==c['feature'] and stored.get('physics_row')==e['physics_row'] and
             stored.get('execution_stage')==e['execution_stage'] and stored.get('retainer_pair_index')==0 and
             stored.get('native_field_errors')=={} and stored.get('separation_m')==separation and
             abs(stored.get('force_N',float('inf'))-force)<=1e-10 and
             all(stored.get(k)==raw[k] for k in ('face_index0','face_index1')),'Stored native residual metadata differs')
        if is_control:
            controlled+=1;coverage.add((c['id'],e['physics_row']))
            case_counts[c['id']]['contacts']+=1;case_counts[c['id']]['positive_contacts']+=int(positive)
            need(type(e['physics_row']) is int and 0<=e['physics_row']<12 and stored.get('unavailable')==[],
                 'All controlled native fields/poses must be available')
        else:warmup+=1
        frame_results=[]
        for label,native_label in [('pre','pre_step_current'),('callback_current','callback_current'),('post','post_step_current')]:
            if not e[label].get('available'):
                need(not is_control and label in ('pre','post'),'Controlled native pose is missing');continue
            matrix=checked_frame(e[label],c,native_label);by_pose=stored.get('by_pose',{}).get(label)
            need(isinstance(by_pose,dict),'Stored native pose residuals missing')
            fn=np.asarray(c['support_face']['outward_local']);support=c['support_face']['support_m']
            for alpha in ALPHAS:
                candidate=p+alpha*separation*n;local=np.linalg.solve(matrix,np.r_[candidate,1.])[:3]
                ring=float(fn@local-support);plate=float(np.asarray(c['plate_normal_world'])@(candidate-np.asarray(c['plate_origin_world'])))
                for value,field in [(ring,'ring_face_residual_m'),(plate,'plate_face_residual_m')]:
                    error=abs(value-by_pose[field][str(alpha)]);stored_error=max(stored_error,error)
                    need(error<=1e-10,'Independent raw residual differs from stored oracle arithmetic')
                if is_control and alpha==1.:ring_error=max(ring_error,abs(ring))
                if is_control and alpha==0.:plate_error=max(plate_error,abs(plate))
                if is_control and label=='pre' and c['speed_m_s']==0 and c['gap_m']==.00035:
                    if alpha==.5:midpoint=max(midpoint,abs(ring))
                    if alpha==0.:raw_control=max(raw_control,abs(ring))
            if is_control and label in ('pre','post'):
                point=p+separation*n;independent=independent_feature(matrix,point,n)
                actual=witness.sector_feature_audit(matrix,point,n,0)
                error=max(float(np.max(np.abs(independent['plane_residuals_m']-actual['plane_residuals_m']))),
                          float(np.max(np.abs(independent['normal_scores']-actual['normal_scores']))))
                classifier_error=max(classifier_error,error)
                need(error<=1e-10 and independent['passed']==actual['passed'],'Independent feature classifier disagrees with runtime')
                direction=np.linalg.solve(matrix,np.r_[-n,0.])[:3]
                need(fn@direction>.99,'Native shape1-to-shape0 normal does not match known contacted face')
                frame_results.append(independent['passed'])
        if is_control and positive:
            counts[c['feature']]['positive']+=1;accepted=all(frame_results) and len(frame_results)==2
            counts[c['feature']]['accepted']+=int(accepted)
            positive_controls.add((c['feature'],c['transformed'],c['reverse_creation']))
            need(accepted==(c['feature']=='front'),'Positive front/side/back conservative classifier control failed')
    need(coverage=={(cid,step) for cid in specs for step in range(12)},'Every original case needs all controlled ticks')
    need(positive_controls==set(itertools.product(('front','side','back'),(False,True),(False,True))),
         'Missing positive front/side/back rotation/creation controls')
    need(ring_error<=TOL and plate_error<=TOL and midpoint>TOL and raw_control>2*TOL,
         'p+s*n convention or raw/midpoint negative controls failed at original 0.1mm tolerance')
    return dict(schema='independent_native_retainer_witness_analysis_v1',passed=True,cases=len(cases),
        controlled_rows=controlled,initialization_rows=warmup,feature_counts=counts,journal_reconstruction_exact=True,
        case_counts=list(case_counts.values()),
        shape0_ring_max_abs_residual_m=ring_error,raw_plate_max_abs_residual_m=plate_error,
        midpoint_negative_control_max_abs_m=midpoint,raw_point_negative_control_max_abs_m=raw_control,
        maximum_stored_residual_error_m=stored_error,maximum_classifier_error=classifier_error,
        pair1_qualified=False,independent_method='analytic six planes and full linear coordinate solve',
        physical_episode_qualified=False,training_eligible=False,test_payloads_opened=False)


def validate_summary(complete,analysis,ticks):
    need(complete.get('contact_rows')==analysis['controlled_rows']+analysis['initialization_rows'] and
         complete.get('cases')==analysis['case_counts'] and
         complete.get('positive_impulse_cases')==sum(c['positive_contacts']>0 for c in analysis['case_counts']) and
         complete.get('warmup_contact_rows')==analysis['initialization_rows'] and
         complete.get('controlled_start_time_s')==ticks[0]['time_before_s'] and
         complete.get('native_pair_reversal_status')=='unavailable',
         'Native COMPLETE case/contact counts differ from independent raw reconstruction')


def qualification_document(owner_ref):
    owner=runtime.read_bound(owner_ref)
    need(owner.get('cpu_diagnostic_only') is not True,'CPU diagnostic fixture cannot issue a native witness receipt')
    run=Path(owner_ref['path']).resolve().parent;review=run/'output/review'
    guard=owner.get('resource_guard',{});guard_path=Path(guard.get('path',''))
    need(guard_path.is_absolute() and guard_path.name=='process_guard.py','Actual native owner source root required')
    root=guard_path.resolve().parents[2]
    refs={key:dict(file=str(path),sha256=runtime.reference(path)['sha256']) for key,path in {
        'owner_terminal':Path(owner_ref['path']).resolve(),'owner_exit':run/'output/native_worker_exit.json',
        'complete':review/'COMPLETE.json','cases':review/'cases.json','native':review/'native_contacts.jsonl',
        'residuals':review/'residuals.json'}.items()}
    raw={key:witness._bound_bytes(ref) for key,ref in refs.items()};complete=json.loads(raw['complete'])
    need(all(json.loads(raw[key]).get('cpu_diagnostic_only') is not True for key in ('owner_exit','complete')),
         'CPU fixture documents cannot qualify native execution')
    binary=Path(complete['native_binding_file']);binary_ref=dict(file=str(binary),sha256=complete['native_binding_sha256'])
    need(binary.is_absolute() and binary.suffix=='.so' and witness._sha(binary)==binary_ref['sha256'],
         'Actual hash-bound native Linux PhysX binary required')
    with binary.open('rb') as stream:need(stream.read(4)==b'\x7fELF','Actual native ELF binary required')
    commit=root/'CODE_COMMIT';manifest=root/'SOURCE_MANIFEST.json'
    need(commit.is_file() and manifest.is_file() and owner.get('code_commit')==commit.read_text().strip(),
         'Actual archived source commit must match native owner')
    inventory=json.loads(manifest.read_text());names=set()
    need(isinstance(inventory,list) and inventory,'Fresh archived source inventory required')
    for entry in inventory:
        name=entry.get('file');need(isinstance(name,str) and name not in names and not Path(name).is_absolute(),'Unique source inventory path required')
        names.add(name);path=runtime.dataset._path(name,root)
        need(root in path.parents and runtime.reference(path)['sha256']==entry.get('sha256') and
             ('bytes' not in entry or path.stat().st_size==entry['bytes']),'Native source archive changed: '+name)
    for name,expected in [('contact_semantics_oracle.py',complete.get('source_sha256')),
                          ('retainer_geometry.py',complete.get('retainer_geometry_source_sha256'))]:
        need('sim/ausc_policy/'+name in names and witness._sha(root/'sim/ausc_policy'/name)==expected,
             'Native fixture authoring/shape source differs from imported archive')
        need(witness._sha(Path(__file__).with_name(name))==expected,
             'Issuer reviewed oracle/geometry differs from actual native source')
    need(Path(guard.get('path','')).resolve()==root/'sim/ausc_policy/process_guard.py' and
         witness._sha(guard['path'])==guard.get('sha256')==witness._sha(Path(__file__).with_name('process_guard.py')),
         'Original native owner guard source changed')
    native_helper=root/'sim/ausc_policy/native_contact_evidence.py'
    need('sim/ausc_policy/native_contact_evidence.py' in names and
         witness._sha(native_helper)==witness._sha(Path(__file__).with_name(native_helper.name)),
         'Actual native contact recorder differs from reviewed issuer source')
    stage=owner.get('stage',{});worker=json.loads(raw['owner_exit']);group=stage.get('process_identity',{})
    need(stage.get('errors')==[] and stage.get('signals')==[] and type(worker.get('pid')) is int and
         str(worker['pid']) in group.get('members',{}) and type(group.get('pgid')) is int and group.get('pgid')==group.get('sid'),
         'Fresh native worker must belong to original drained owner group without signals/errors')
    document=dict(schema=witness.QUALIFICATION_SCHEMA,qualified=True,scope=deepcopy(witness.SCOPE),rule=deepcopy(witness.RULE),
        execution=dict(terminal=True,exit_code=0,complete=True),native_binding=binary_ref,evidence=refs,
        geometry_source_sha256=GEOMETRY_SOURCE_SHA256,retainer_geometry_source_sha256=complete['retainer_geometry_source_sha256'],
        classifier_source_sha256=witness._sha(witness.__file__),issuer_source_sha256=witness._sha(__file__),
        source_archive=dict(code_commit=owner['code_commit'],source_manifest=runtime.reference(manifest)),
        qualified_at_utc=datetime.now(timezone.utc).isoformat(),training_eligible=False,model_validated=False,
        test_payloads_opened=False,expert_episode_requalified=False,old_run_verdicts_unchanged=True,
        qualification_scope='Fresh native shape-0 witness semantics and conservative six-face classification only.')
    witness._validate_oracle_chain(document,raw)
    need(complete.get('controlled_simulation_steps')==12 and complete.get('physics_dt_s')==1/240 and
         complete.get('runtime_shutdown',{}).get('app_close_returned') is True and
         complete.get('runtime_shutdown',{}).get('fast_shutdown') is False,'Actual complete native close/clocks required')
    artifacts={}
    for name in ('cases.json','native_contacts.jsonl','native_journal.jsonl','physics_ticks.json','residuals.json'):
        ref=runtime.reference(review/name);declared=complete.get('artifacts',{}).get(name,{})
        need(all(declared.get(key)==ref[key] for key in ('bytes','sha256')),'Raw native artifact differs from COMPLETE: '+name)
        artifacts[name]=ref
    return document,artifacts


def qualify(owner_ref,output):
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Explicit CPU-only CUDA_VISIBLE_DEVICES required')
    output=Path(output).resolve();need(not output.exists(),'Fresh immutable qualification output required')
    document,artifacts=qualification_document(owner_ref)
    def read(name):return runtime.read_bound(artifacts[name])
    def lines(name):return [json.loads(line) for line in runtime.bound_path(artifacts[name]).read_text().splitlines()]
    analysis=analyze(read('cases.json'),lines('native_contacts.jsonl'),lines('native_journal.jsonl'),
                     read('physics_ticks.json'),read('residuals.json'))
    validate_summary(json.loads(witness._bound_bytes(document['evidence']['complete'])),analysis,read('physics_ticks.json'))
    # Reauthenticate every input before publishing a fresh receipt.
    qualification_document(owner_ref)
    for ref in artifacts.values():runtime.bound_path(ref)
    output.mkdir(parents=True)
    analysis.update(owner_terminal=owner_ref,artifacts=artifacts,issuer=runtime.reference(Path(__file__)))
    independent=runtime.write_new(output/'INDEPENDENT_ANALYSIS.json',analysis)
    document['independent_analysis']=dict(file=independent['path'],sha256=independent['sha256'])
    qualified=runtime.write_new(output/'QUALIFIED.json',document)
    contract=dict(schema=witness.SCHEMA,qualification=dict(file=qualified['path'],sha256=qualified['sha256']))
    witness.load_contract(contract)
    runtime.write_new(output/'CONTRACT.json',contract)
    return contract


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--owner',required=True)
    parser.add_argument('--sha256',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(qualify(dict(path=args.owner,sha256=args.sha256),args.output)))


if __name__=='__main__':main()
