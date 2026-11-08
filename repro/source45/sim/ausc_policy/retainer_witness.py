"""Qualified shape-0 convex retainer witness audit; opt-in and fail closed.

The receipt qualifies native semantics, not robot performance. CPU fixtures
may exercise this module but cannot initialize its native runtime provider.
Cooked PhysX face indices are retained only as raw evidence, never CAD labels.
"""
from copy import deepcopy
from functools import lru_cache
import hashlib
import importlib
import json
from pathlib import Path
import re

import numpy as np

from .retainer_contact import ROOT, is_retainer_patient_pair
from .retainer_geometry import ring_sector_meshes, RING_FRONT_Z_M, INNER_RADIUS_M, OUTER_RADIUS_M, SECTOR_COUNT

SCHEMA='retainer_front_audit_v3'
QUALIFICATION_SCHEMA='qualified_native_retainer_witness_v1'
BODY='/World/Stethoscope/Carriage'
POSE_SOURCE='omni.physx.IPhysx.get_rigidbody_transformation'
LOCATION_TOLERANCE_M=.0001
# Native normals originate as float32. Sixteen float32 roundoffs bound the
# 3-term frame dot product and subsequent 3-term support dot/comparison. This
# guard resolves floating-point ties only; it is not an angular permission.
NORMAL_SCORE_GUARD=float(16*np.finfo(np.float32).eps)
SCOPE=dict(retainer_pair_index=0,retainer_shape='convexHull',patient_shape='staticTriangleMesh')
RULE=dict(witness='p+s*n',normal_direction='shape1_to_shape0',
          pose_frames=['pre_step_current','post_step_current'],pair1_qualified=False)
EVIDENCE_KEYS={'owner_terminal','owner_exit','complete','cases','native','residuals'}


def _need(ok,message):
    if not ok:raise ValueError(message)


def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):digest.update(chunk)
    return digest.hexdigest()


def _reference(ref):
    _need(isinstance(ref,dict) and set(ref)=={'file','sha256'}
          and isinstance(ref['file'],str) and Path(ref['file']).is_absolute()
          and isinstance(ref['sha256'],str) and re.fullmatch('[0-9a-f]{64}',ref['sha256']) is not None,
          'Explicit absolute file/SHA256 qualification reference required')
    return ref


def _bound_bytes(ref):
    _reference(ref);data=Path(ref['file']).read_bytes()
    _need(hashlib.sha256(data).hexdigest()==ref['sha256'],'Qualification artifact hash changed: '+ref['file'])
    return data


def require_contract(contract):
    _need(isinstance(contract,dict) and set(contract)=={'schema','qualification'}
          and contract['schema']==SCHEMA,'Explicit native_retainer_audit retainer_front_audit_v3 required')
    _reference(contract['qualification'])
    return contract


def _validate_oracle_chain(receipt,data):
    evidence=receipt['evidence']
    owner=json.loads(data['owner_terminal']);worker=json.loads(data['owner_exit']);complete=json.loads(data['complete'])
    _need(all(isinstance(value,dict) for value in (owner,worker,complete)),
          'Native qualification documents must be JSON objects')
    owner_path=Path(evidence['owner_terminal']['file']).resolve();run=owner_path.parent
    _need(owner_path.name=='result.json' and
          Path(evidence['owner_exit']['file']).resolve()==run/'output/native_worker_exit.json' and
          Path(evidence['complete']['file']).resolve()==run/'output/review/COMPLETE.json',
          'Qualification owner/worker/COMPLETE path binding mismatch')
    stage=owner.get('stage',{});config=owner.get('config',{})
    _need(isinstance(stage,dict) and isinstance(config,dict),'Malformed native owner stage/config')
    _need(owner.get('completed') is True and owner.get('gpu_released') is True and
          config.get('experiment')=='contact_semantics_oracle' and
          isinstance(owner.get('ended_utc'),str) and bool(owner['ended_utc']) and
          stage.get('passed') is True and stage.get('process_group_drained') is True and
          type(stage.get('exit_code')) is int and stage['exit_code']==0,
          'Native oracle owner is failed, undrained or not released')
    _need(worker.get('worker')=='sim.ausc_policy.contact_semantics_oracle' and
          type(worker.get('returncode')) is int and worker['returncode']==0 and
          worker.get('clean_exit') is True and worker.get('timed_out') is False and
          worker.get('complete_file_present') is True,'Native oracle worker lacks actual clean exit-0')
    _need(complete.get('schema')=='native_convex_triangle_contact_oracle_v1' and
          complete.get('completed') is True and complete.get('execution_completed') is True and
          complete.get('evidence_complete') is True and complete.get('evidence_status')=='complete',
          'Incomplete native contact oracle evidence')
    _need(complete.get('native_binding_file')==receipt['native_binding']['file'] and
          complete.get('native_binding_sha256')==receipt['native_binding']['sha256'] and
          complete.get('retainer_geometry_source_sha256')==receipt['retainer_geometry_source_sha256'],
          'Native oracle COMPLETE binding/geometry differs from qualification')
    # COMPLETE's candidate semantics flag remains false by design. Independent
    # receipt qualification is a separate analysis, never inferred from it.
    artifacts=complete.get('artifacts',{})
    _need(isinstance(artifacts,dict),'Malformed native COMPLETE artifact inventory')
    for key,name in [('cases','cases.json'),('native','native_contacts.jsonl'),('residuals','residuals.json')]:
        artifact=artifacts.get(name,{})
        _need(isinstance(artifact,dict),'Malformed native COMPLETE artifact reference')
        _need(Path(evidence[key]['file']).resolve()==run/'output/review'/name and
              artifact.get('sha256')==evidence[key]['sha256'] and
              type(artifact.get('bytes')) is int and artifact['bytes']==len(data[key]),
              'Native oracle artifact does not match COMPLETE: '+name)


def load_contract(contract):
    """Authenticate a separate qualified receipt and every raw evidence ref."""
    from .geometry_epoch import GEOMETRY_SOURCE_SHA256
    require_contract(contract)
    receipt=json.loads(_bound_bytes(contract['qualification']))
    _need(isinstance(receipt,dict),'Qualification receipt must be a JSON object')
    _need(receipt.get('schema')==QUALIFICATION_SCHEMA and receipt.get('qualified') is True,
          'Independently qualified native witness receipt required')
    _need(receipt.get('scope')==SCOPE and receipt.get('rule')==RULE and
          type(receipt['scope']['retainer_pair_index']) is int and
          receipt['rule']['pair1_qualified'] is False,'Unqualified native witness scope or rule')
    _need(receipt.get('execution')==dict(terminal=True,exit_code=0,complete=True) and
          receipt['execution']['terminal'] is True and receipt['execution']['complete'] is True and
          type(receipt['execution']['exit_code']) is int,
          'Terminal exit-0 native oracle qualification required')
    _need(receipt.get('geometry_source_sha256')==GEOMETRY_SOURCE_SHA256 and
          receipt.get('retainer_geometry_source_sha256')==_sha(Path(__file__).with_name('retainer_geometry.py')),
          'Qualified retainer geometry source changed')
    _reference(receipt.get('native_binding'))
    evidence=receipt.get('evidence')
    _need(isinstance(evidence,dict) and set(evidence)==EVIDENCE_KEYS,'Complete qualified oracle artifact references required')
    data={key:_bound_bytes(ref) for key,ref in evidence.items()}
    _validate_oracle_chain(receipt,data)
    if 'independent_analysis' in receipt:_bound_bytes(receipt['independent_analysis'])
    return receipt


def verify_native_binding(contract,native_binding_file=None):
    """Lazy native import; compare the actual installed binary before movement."""
    receipt=load_contract(contract)
    if native_binding_file is None:
        _need(receipt.get('cpu_diagnostic_only') is not True,'CPU diagnostic receipt cannot authorize a native runtime')
        native_binding_file=importlib.import_module('omni.physx.bindings._physx').__file__
    _need(str(Path(native_binding_file).resolve())==str(Path(receipt['native_binding']['file']).resolve()) and
          _sha(native_binding_file)==receipt['native_binding']['sha256'],
          'Actual PhysX native binary differs from qualified binary')
    return receipt


def settings(fresh):
    """An absent selector is the archival v2 route; a present one must qualify."""
    declared=fresh.get('native_retainer_audit')
    capture=fresh.get('raw_multimodal_capture',{})
    metadata=capture.get('compact',{}).get('metadata',{}) if isinstance(capture,dict) else {}
    captured=metadata.get('native_retainer_audit')
    if captured!=declared and (captured is not None or declared is not None) and metadata:
        raise ValueError('Native retainer runtime/capture binding mismatch')
    if 'native_retainer_audit' not in fresh:return None
    contract=deepcopy(declared);load_contract(contract)
    return contract


class NativeRetainerAudit:
    """One validated qualification per run; no native/evidence rehash per tick."""
    def __init__(self,contract, *, native_binding_file=None):
        self.contract=deepcopy(contract)
        self.receipt=verify_native_binding(self.contract,native_binding_file)

    def audit(self,points,evidence,reported_force_N, *, dt,physics_row,time_s):
        return _audit(points,evidence,reported_force_N,self.contract,dt=dt,physics_row=physics_row,time_s=time_s)


def _rigid(value):
    m=np.asarray(value,float)
    _need(m.shape==(4,4) and np.isfinite(m).all() and
          np.allclose(m[3],[0,0,0,1],atol=1e-10,rtol=0) and
          np.allclose(m[:3,:3].T@m[:3,:3],np.eye(3),atol=1e-7,rtol=0) and
          np.isclose(np.linalg.det(m[:3,:3]),1.,atol=1e-7,rtol=0),'Invalid native rigid matrix')
    return m


def _frames(evidence,dt,physics_row,time_s):
    _need(isinstance(evidence,dict) and evidence.get('schema')=='native_contact_tick_v1'
          and type(physics_row) is int and physics_row>=0
          and type(evidence.get('physics_row')) is int and evidence['physics_row']==physics_row,
          'Native physics tick index mismatch')
    start=evidence.get('interval_start_s');native_dt=evidence.get('physics_dt_s')
    _need(isinstance(start,(int,float)) and np.isfinite(start) and start>=0 and
          isinstance(native_dt,(int,float)) and np.isfinite(native_dt) and
          np.isclose(native_dt,dt,atol=1e-12,rtol=0) and
          np.isfinite(time_s) and np.isclose(start+dt,time_s,atol=1e-10,rtol=0),
          'Native physics tick time mismatch')
    _need(evidence.get('observer_error') is None,'Native contact observer error')
    result={}
    for key,label in [('pre_step','pre_step_current'),('post_step','post_step_current')]:
        frame=evidence.get(key)
        _need(isinstance(frame,dict) and frame.get('available') is True and
              frame.get('source')==POSE_SOURCE and frame.get('path')==BODY and frame.get('sample')==label
              and not frame.get('error'),'Missing or unqualified native '+key+' frame')
        result[key]=_rigid(frame.get('world_from_carriage'))
    return result


@lru_cache(maxsize=SECTOR_COUNT)
def sector_planes(index):
    """Six outward support planes derived from the 12 authored triangles."""
    vertices,triangles=ring_sector_meshes()[index];center=vertices.mean(axis=0);planes=[]
    for ids in triangles:
        a,b,c=vertices[ids];n=np.cross(b-a,c-a);n/=np.linalg.norm(n)
        if n@(a-center)<0:n=-n
        b=float(n@a)
        if any(np.allclose(n,p[0],atol=1e-12,rtol=0) and abs(b-p[1])<=1e-12 for p in planes):continue
        _need(np.max(vertices@n-b)<=1e-12,'Nonconvex authored retainer sector')
        planes.append((n,b))
    _need(len(planes)==6,'Expected six authored convex sector planes')
    normals=np.asarray([p[0] for p in planes]);offsets=np.asarray([p[1] for p in planes])
    front=np.flatnonzero(np.all(np.isclose(normals,[0,0,-1],atol=1e-12,rtol=0),axis=1))
    _need(len(front)==1,'Unique authored front plane required')
    normals.setflags(write=False);offsets.setflags(write=False)
    return normals,offsets,int(front[0])


def _frame_audit(matrix,witness,normal,index):
    local=matrix[:3,:3].T@(witness-matrix[:3,3]);direction=-matrix[:3,:3].T@normal
    normals,offsets,front=sector_planes(index)
    residuals=normals@local-offsets;scores=normals@direction
    other=np.delete(scores,front)
    front_unique=bool(scores[front]>NORMAL_SCORE_GUARD and
                      np.all(scores[front]-other>NORMAL_SCORE_GUARD))
    radial=float(np.linalg.norm(local[:2]))
    axial=bool(abs(local[2]-RING_FRONT_Z_M)<=LOCATION_TOLERANCE_M)
    annulus=bool(INNER_RADIUS_M*np.cos(np.pi/SECTOR_COUNT)-LOCATION_TOLERANCE_M<=radial<=OUTER_RADIUS_M+LOCATION_TOLERANCE_M)
    inside=bool(np.all(residuals<=LOCATION_TOLERANCE_M))
    incident=bool(abs(residuals[front])<=LOCATION_TOLERANCE_M)
    reasons=[]
    for ok,reason in [(axial,'outside_front_axial_band'),(annulus,'outside_front_annulus'),
                      (inside,'outside_actual_sector'),(incident,'not_front_incident'),
                      (front_unique,'front_not_unique_positive_support')]:
        if not ok:reasons.append(reason)
    return dict(passed=not reasons,reasons=reasons,carriage_witness_m=local.tolist(),
        direction_local=direction.tolist(),normal_scores=scores.tolist(),plane_residuals_m=residuals.tolist(),
        outward_plane_normals=normals.tolist(),plane_offsets_m=offsets.tolist(),front_plane_index=front,
        front_unique_max=front_unique,inside_sector=inside,front_incident=incident,
        axial_band=axial,annulus=annulus)


def sector_feature_audit(world_from_body,witness_world,normal_world,sector_index):
    """Pure geometric diagnostic; does not qualify a native pose or receipt."""
    _need(type(sector_index) is int and 0<=sector_index<SECTOR_COUNT,'Invalid retainer sector')
    matrix=_rigid(world_from_body)
    witness=np.asarray(witness_world,float);normal=np.asarray(normal_world,float)
    _need(witness.shape==(3,) and normal.shape==(3,) and np.isfinite(witness).all() and
          np.isfinite(normal).all() and abs(float(normal@normal)-1.)<=NORMAL_SCORE_GUARD,
          'Finite witness and raw native unit normal required')
    return _frame_audit(matrix,witness,normal,sector_index)


def audit_retainer_witness(points,evidence,reported_force_N,contract, *, dt=1/240,physics_row,time_s):
    load_contract(contract)
    return _audit(points,evidence,reported_force_N,contract,dt=dt,physics_row=physics_row,time_s=time_s)


def _audit(points,evidence,reported_force_N,contract, *, dt,physics_row,time_s):
    require_contract(contract)
    _need(np.isfinite(dt) and dt>0 and np.isfinite(reported_force_N) and reported_force_N>=0,
          'Invalid retainer impulse ledger input')
    frame_error=None
    try:frames=_frames(evidence,dt,physics_row,time_s)
    except (ValueError,TypeError,KeyError) as error:frames={};frame_error=str(error)
    total=bad=0.;locations=[];positive_qualified=0;zero_points=0
    for point in points:
        pair=point['colliders']
        _need(len(pair)==2 and all(isinstance(p,str) for p in pair),'Invalid raw contact pair')
        if not is_retainer_patient_pair(*pair):continue
        impulse=np.asarray(point['impulse_Ns'],float)
        _need(impulse.shape==(3,) and np.isfinite(impulse).all(),'Invalid raw retainer impulse')
        force=float(np.linalg.norm(impulse)/dt);total+=force
        ri=0 if pair[0].startswith(ROOT+'DeviceContactRing/') else 1
        if force==0.:
            # Keep the raw callback untouched and an observable zero-load row,
            # but never invent a positive feature qualification from it.
            zero_points+=1
            locations.append(dict(force_N=0.,positive_impulse=False,on_front=None,
                retainer_pair_index=ri,witness_world_m=None,witness_rule=None,
                reasons=['zero_impulse_not_feature_qualified'],frames={},
                raw_face_index0=point.get('face_index0'),raw_face_index1=point.get('face_index1')))
            continue
        reasons=[];frame_reports={};witness=None
        if ri!=0:reasons.append('unqualified_retainer_pair_index')
        if frame_error:reasons.append(frame_error)
        try:
            p=np.asarray(point.get('position_world'),float);n=np.asarray(point.get('normal_world'),float)
            s=float(point.get('separation_m'))
            _need(p.shape==(3,) and n.shape==(3,) and np.isfinite(p).all() and np.isfinite(n).all()
                  and np.isfinite(s),'Missing or nonfinite native witness fields')
            _need(abs(float(n@n)-1.)<=NORMAL_SCORE_GUARD,'Native normal is not unit length')
            _need(not point.get('native_field_errors') and
                  'normal' not in point.get('native_fields_missing',[]),'Native witness field extraction error')
            if ri==0:
                # Use the raw signed separation and raw unit normal unchanged.
                witness=p+s*n
                index=int(pair[ri].rsplit('_',1)[1])
                for key,matrix in frames.items():
                    report=_frame_audit(matrix,witness,n,index);frame_reports[key]=report
                    reasons.extend(key+':'+reason for reason in report['reasons'])
        except (ValueError,TypeError,KeyError) as error:reasons.append(str(error))
        qualified=not reasons
        if force>0.:
            if not qualified:bad+=force
            else:positive_qualified+=1
        locations.append(dict(force_N=force,positive_impulse=force>0.,on_front=qualified,
            retainer_pair_index=ri,witness_world_m=None if witness is None else witness.tolist(),
            witness_rule='p+s*n' if ri==0 else None,reasons=reasons,frames=frame_reports,
            raw_face_index0=point.get('face_index0'),raw_face_index1=point.get('face_index1')))
    matched=bool(np.isclose(total,reported_force_N,atol=1e-8,rtol=1e-8))
    return dict(schema=SCHEMA,native_retainer_audit=deepcopy(contract),passed=bool(matched and bad==0.),
        force_N=total,force_matches_ledger=matched,off_front_force_N=bad,
        location_tolerance_m=LOCATION_TOLERANCE_M,normal_score_precision_guard=NORMAL_SCORE_GUARD,
        positive_points_qualified=positive_qualified,zero_impulse_points=zero_points,points=locations)
