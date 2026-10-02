"""Actual saved carriage-pose ray gaps against the recorded static triangle mesh.

CPU geometry diagnostic only. Missing rays remain unknown; sparse sampling is
not a seal, area, continuous-contact or acoustic certificate.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation


def ray_offsets(points, direction, triangles):
    points=np.asarray(points,float);d=np.asarray(direction,float);tri=np.asarray(triangles,float)
    if (points.ndim!=2 or points.shape[1]!=3 or tri.ndim!=3 or tri.shape[1:]!=(3,3)
            or d.shape!=(3,) or not all(np.isfinite(x).all() for x in (points,d,tri))
            or not np.isclose(np.linalg.norm(d),1.,atol=1e-6,rtol=0)):
        raise ValueError('Finite points, triangles and unit direction required')
    offsets=[];e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0]
    pvec=np.cross(d,e2);det=np.einsum('ij,ij->i',e1,pvec)
    valid=np.abs(det)>1e-12;inverse=np.zeros_like(det);inverse[valid]=1/det[valid]
    for point in points:
        origin=point-.05*d;tvec=origin-tri[:,0]
        u=np.einsum('ij,ij->i',tvec,pvec)*inverse
        qvec=np.cross(tvec,e1);v=(qvec@d)*inverse
        distance=np.einsum('ij,ij->i',e2,qvec)*inverse
        hit=valid&(u>=-1e-9)&(v>=-1e-9)&(u+v<=1+1e-9)&(distance>=0)&(distance<=.1)
        offsets.append(float(distance[hit].min()-.05) if hit.any() else np.nan)
    return np.asarray(offsets)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def front_samples(vertices):
    v=np.asarray(vertices,float)
    if v.shape!=(8,3) or not np.isfinite(v).all():
        raise ValueError('Unexpected ring sector vertices')
    # The collision exporter stores float32: -0.047 becomes -0.04699999839.
    # Ten nanometres selects that saved face without moving any vertex.
    front=v[np.isclose(v[:,2],-.047,atol=1e-8,rtol=0)]
    if len(front)!=4:raise ValueError('Unexpected front sector geometry')
    order=np.argsort(np.linalg.norm(front[:,:2],axis=1))
    return np.asarray([front[order[:2]].mean(axis=0),front.mean(axis=0),front[order[2:]].mean(axis=0)])


def verify_geometry_binding(directory,native):
    directory=Path(directory)
    for filename,key in [('collision_geometry.json','collision_geometry_sha256'),
                         ('collision_geometry.npz','collision_arrays_sha256')]:
        if sha(directory/filename)!=native.get('geometry_artifacts',{}).get(key):
            raise ValueError('Original geometry SHA binding differs: '+filename)


def verify_point_coverage(buckets,native):
    expected=[p['name'] for p in native['plan_bindings']]
    nominal=[s+str(i) for i in range(1,6) for s in 'LR']
    if expected!=nominal or list(buckets)!=expected:
        raise ValueError('Incomplete or changed nominal ten point order')


def audit(run,output):
    started=time.monotonic();run=Path(run);output=Path(output)
    if output.exists():raise FileExistsError('Existing diagnostic must be retained')
    case=run/'output/review';data_path=case/'fresh_expert/fresh_expert_replay.npz'
    metadata_path=case/'geometry/collision_geometry.json';mesh_path=case/'geometry/collision_geometry.npz'
    native=json.loads((case/'fresh_expert_result.json').read_text())
    verify_geometry_binding(case/'geometry',native)
    geometry=json.loads(metadata_path.read_text())['colliders']
    patient=[e for e in geometry if e['path'].startswith('/World/Patient/')]
    if not patient or any(e['body']!='world' or e['approximation']!='none' for e in patient):
        raise ValueError('Recorded static patient triangle surfaces required')
    ring=sorted([e for e in geometry if '/DeviceContactRing/sector_' in e['path']],key=lambda e:e['path'])
    if len(ring)!=32:raise ValueError('All 32 ring sectors required')
    mesh=[];local=[]
    with np.load(mesh_path,allow_pickle=False) as a:
        for e in patient:
            v=np.asarray(a[e['key']+'_v'],float);f=np.asarray(a[e['key']+'_f']);t=np.asarray(e['world_from_body'])
            mesh.append((v@t[:3,:3].T+t[:3,3])[f])
        for e in ring:
            if e['body']!='/World/Stethoscope/Carriage':raise ValueError('Unexpected ring body')
            local.extend(front_samples(a[e['key']+'_v']))
    mesh=np.concatenate(mesh);local=np.asarray(local);bounds_lo=mesh.min(axis=1);bounds_hi=mesh.max(axis=1)
    with np.load(data_path,allow_pickle=False) as a:
        paths=[str(x) for x in a['paths']];indices=a['row_indices'];all_poses=a['poses']
        if paths.count('/World/Stethoscope/Carriage')!=1 or not np.array_equal(indices,np.arange(7,native['physics_steps'],8)):
            raise ValueError('Actual pose sampling ledger differs')
        poses=all_poses[:,paths.index('/World/Stethoscope/Carriage')].copy()
    wanted={int(x):i for i,x in enumerate(indices)};buckets=defaultdict(list);digest=hashlib.sha256()
    count=0
    for line in (case/'fresh_expert/fresh_expert_trace.jsonl').open('rb'):
        digest.update(line);row=json.loads(line);i=count;count+=1
        if row['phase']!='HOLD' or i not in wanted:continue
        pose=poses[wanted[i]]
        if not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:])-1)>1e-6:raise ValueError('Invalid native carriage pose')
        rotation=Rotation.from_quat(pose[[4,5,6,3]]).as_matrix()
        world=local@rotation.T+pose[:3];direction=-rotation[:,2]
        # Conservative ray-segment AABB prefilter, never a skin approximation.
        ends=np.r_[world-.05*direction,world+.05*direction]
        selected=np.all(bounds_hi>=ends.min(axis=0)-1e-9,axis=1)&np.all(bounds_lo<=ends.max(axis=0)+1e-9,axis=1)
        offsets=ray_offsets(world,direction,mesh[selected])
        axis_angle=float(np.degrees(np.arccos(np.clip(direction@np.asarray(row['axis_world']),-1,1))))
        buckets[row['target']].append(dict(time_s=row['time_s'],row=i,offsets=offsets,
            wrist_carriage_axis_angle_deg=axis_angle))
    if digest.hexdigest()!=native['trace_sha256'] or count!=native['physics_steps']:
        raise ValueError('Trace SHA/count mismatch')
    verify_point_coverage(buckets,native)
    verify_geometry_binding(case/'geometry',native)
    points=[]
    for target,rows in buckets.items():
        if len(rows)!=90:raise ValueError('Expected 90 actual 30 Hz HOLD poses')
        gaps=np.asarray([r['offsets'] for r in rows]);finite=gaps[np.isfinite(gaps)]
        if not finite.size:raise ValueError('No measured mesh intersections')
        points.append(dict(target=target,poses=len(rows),rays=int(gaps.size),missing_rays=int(np.isnan(gaps).sum()),
            gap_mm=dict(min=float(finite.min()*1000),median=float(np.median(finite)*1000),
                p95=float(np.percentile(finite,95)*1000),max=float(finite.max()*1000)),
            fraction_within_0_1mm=float(np.mean(np.abs(gaps)<=.0001)),
            fraction_within_0_5mm=float(np.mean(np.abs(gaps)<=.0005)),
            fraction_within_1mm=float(np.mean(np.abs(gaps)<=.001)),
            native_wrist_carriage_axis_max_deg=max(r['wrist_carriage_axis_angle_deg'] for r in rows),
            middle_frame=dict(time_s=rows[45]['time_s'],row=rows[45]['row'],
                sector_inner_center_outer_gap_mm=[[float(x*1000) if np.isfinite(x) else None for x in values]
                    for values in rows[45]['offsets'].reshape(32,3)])))
    report=dict(schema='native_carriage_retainer_surface_gap_v1',run_id=run.name,points=points,
        trace_sha256=digest.hexdigest(),physics_rows=count,wall_s=time.monotonic()-started,
        source_sha256=sha(__file__),source_bindings={str(p.relative_to(run)):sha(p) for p in (case/'fresh_expert_result.json',data_path,metadata_path,mesh_path)},
        geometry='recorded static patient triangles; actual saved carriage pose; three front samples per each of 32 sectors',
        positive_gap='patientward ray from rim sample to first outer triangle hit; positive open gap, negative penetration',
        sampled_pose_hz=30,ray_half_length_m=.05,flatness_qualification=False,training_eligible=False,
        scope='Sampled rigid geometry; not all ring points, skin deformation, acoustic seal or sound quality. Threshold fractions are descriptive, not an acoustic gate.')
    output.mkdir(parents=True,exist_ok=False)
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(points=len(points),poses=sum(p['poses'] for p in points),wall_s=report['wall_s'],output=str(output))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run');p.add_argument('output');a=p.parse_args()
    audit(a.run,a.output)
