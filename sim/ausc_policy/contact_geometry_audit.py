"""Read-only contact accounting and mesh distances at saved actual rigid poses.

No physics execution, FK-substituted pose, scene edit or dataset qualification.
This diagnoses the recorded simulation, not real acoustic coupling.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


GROUPS=('diaphragm','lip','contact_rim','carriage_other','frame','forbidden')
CARRIAGE='/World/Stethoscope/Carriage'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def group(pair):
    if len(pair)!=2 or any(not isinstance(p,str) for p in pair):
        raise ValueError('Invalid collider pair')
    patient=any(p.startswith('/World/Patient/') for p in pair)
    module=next((p for p in pair if p.startswith('/World/Stethoscope/')),'')
    if patient and module:
        if '/Fixed/' in module:return 'frame'
        if module.endswith('/Diaphragm'):return 'diaphragm'
        if '/RetentionLip/' in module:return 'lip'
        if '/ContactRim/' in module:return 'contact_rim'
        return 'carriage_other'
    if patient or module or any('openarm_left_link' in p for p in pair):return 'forbidden'
    return None


def audit_contacts(rows):
    hold=[r for r in rows if r['phase']=='HOLD']
    if not hold:raise ValueError('No actual HOLD rows')
    values=[];counts={k:0 for k in GROUPS};pairs={};max_error=0.
    for row in hold:
        forces={k:0. for k in GROUPS}
        for point in row['contact_points']:
            impulse=np.asarray(point['impulse_Ns'],float)
            if impulse.shape!=(3,) or not np.isfinite(impulse).all():raise ValueError('Invalid recorded impulse')
            label=group(point['colliders'])
            if label is None:continue
            force=float(np.linalg.norm(impulse)*240)
            forces[label]+=force;counts[label]+=1
            name=' | '.join(sorted(point['colliders']))
            pairs[name]=pairs.get(name,0)+1
        expected=np.asarray([row[k+'_N'] for k in GROUPS],float)
        actual=np.asarray([forces[k] for k in GROUPS])
        error=float(np.max(np.abs(expected-actual)));max_error=max(error,max_error)
        if not np.isfinite(expected).all() or error>1e-9:
            raise ValueError('Raw contact impulses disagree with recorded component force')
        values.append(actual)
    values=np.asarray(values)
    return dict(hold_physics_rows=len(hold),hold_duration_s=len(hold)/240,
        contact_impulse_force_max_error_N=max_error,component_contact_point_counts=counts,
        diaphragm_contact_points=counts['diaphragm'],contact_pair_point_counts=pairs,
        component_force_peak_N=dict(zip(GROUPS,values.max(axis=0).tolist())),
        component_force_mean_N=dict(zip(GROUPS,values.mean(axis=0).tolist())))


def sampled_hold_poses(rows,arrays):
    paths=[str(x) for x in arrays['paths']];indices=np.asarray(arrays['row_indices']);poses=np.asarray(arrays['poses'])
    if (paths.count(CARRIAGE)!=1 or len(set(paths))!=len(paths) or indices.dtype.kind not in 'iu'
            or not np.array_equal(indices,np.arange(7,len(rows),8))
            or poses.shape!=(len(indices),len(paths),7) or not np.isfinite(poses).all()):
        raise ValueError('Missing or mismatched actual rigid pose ledger')
    j=paths.index(CARRIAGE);result=[]
    for i,index in enumerate(indices):
        if rows[int(index)]['phase']!='HOLD':continue
        pose=poses[i,j]
        if abs(np.linalg.norm(pose[3:])-1)>1e-5:raise ValueError('Invalid saved quaternion')
        result.append((int(index),j,pose.copy()))
    if not result:raise ValueError('No saved actual HOLD poses')
    return result


def mesh_object(entry,arrays,fcl):
    v=np.asarray(arrays[entry['key']+'_v'],np.float64)
    faces=np.asarray(arrays[entry['key']+'_f'],np.int32)
    model=fcl.BVHModel();model.beginModel(len(v),len(faces));model.addSubModel(v,faces);model.endModel()
    transform=np.asarray(entry['world_from_body'],float)
    return fcl.CollisionObject(model,fcl.Transform(transform[:3,:3],transform[:3,3]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();c=json.loads(args.config.read_text())
    if (sys.platform!='linux' or os.environ.get('AUSC_OWNED_RUN')!='1' or os.environ.get('CUDA_VISIBLE_DEVICES')!=''
            or c['experiment']!='contact_geometry_audit'):
        raise ValueError('Owned Linux CPU contact audit required')
    root=Path(c['source_run']);case=root/'output'/c['case_id']
    required={
        'result.json',*('output/'+c['case_id']+'/'+n for n in ('result.json','outcome.json',
        'expert_trace.jsonl','expert_replay.npz','geometry/collision_geometry.json','geometry/collision_geometry.npz'))}
    if set(c['source_files'])!=required:raise ValueError('Incomplete source bindings')
    def check_bindings():
        for name,digest in c['source_files'].items():
            path=root/name
            if not path.resolve().is_relative_to(root.resolve()) or sha(path)!=digest:
                raise ValueError('Changed contact-audit source: '+name)
    check_bindings()
    owner=json.loads((root/'result.json').read_text());outcome=json.loads((case/'outcome.json').read_text())
    if (not owner['completed'] or not owner['gpu_released'] or not owner['stage']['process_group_drained']
            or owner['stage']['exit_code']!=0 or not outcome['expert_demonstration']
            or outcome['learned_policy'] or outcome['training_dataset_eligible']):
        raise ValueError('Completed expert-only actual case required')
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[json.loads(x) for x in (case/'expert_trace.jsonl').open()]
    accounting=audit_contacts(rows)
    with np.load(case/'expert_replay.npz',allow_pickle=False) as data:poses=sampled_hold_poses(rows,data)
    import fcl
    from scipy.spatial.transform import Rotation
    geometry=json.loads((case/'geometry/collision_geometry.json').read_text())['colliders']
    membranes=[e for e in geometry if e['path'].endswith('/Diaphragm')]
    patients=[e for e in geometry if e['path'].startswith('/World/Patient/')]
    if len(membranes)!=1 or membranes[0]['body']!=CARRIAGE or not patients:
        raise ValueError('Unexpected membrane or patient geometry')
    if any(e['body']!='world' or e['approximation']!='none' for e in patients):
        raise ValueError('This audit requires the recorded static patient triangle meshes')
    with np.load(case/'geometry/collision_geometry.npz',allow_pickle=False) as arrays:
        membrane=mesh_object(membranes[0],arrays,fcl)
        objects=[mesh_object(e,arrays,fcl) for e in patients]
        lip=[e for e in geometry if '/RetentionLip/' in e['path']]
        membrane_front=float(np.min(arrays[membranes[0]['key']+'_v'][:,2]))
        lip_front=min(float(np.min(arrays[e['key']+'_v'][:,2])) for e in lip)
    distances=[]
    for row_index,_,pose in poses:
        # Saved SingleRigidPrim pose is position + wxyz; no FK or policy estimate.
        rotation=Rotation.from_quat(pose[[4,5,6,3]]).as_matrix()
        membrane.setTransform(fcl.Transform(rotation,pose[:3]))
        values=[float(fcl.distance(membrane,obj)) for obj in objects]
        if not np.isfinite(values).all():raise ValueError('Nonfinite physical mesh distance')
        nearest=int(np.argmin(values))
        distances.append(dict(physics_row=row_index,time_s=rows[row_index]['time_s'],
            membrane_patient_distance_m=values[nearest],nearest_patient=patients[nearest]['path']))
    d=np.asarray([x['membrane_patient_distance_m'] for x in distances])
    check_bindings()
    report=dict(completed=True,accounting=accounting,actual_pose_samples=len(poses),distance_samples=distances,
        distance_min_m=float(d.min()),distance_median_m=float(np.median(d)),distance_max_m=float(d.max()),
        geometry_membrane_recess_behind_lip_m=membrane_front-lip_front,
        source_files=c['source_files'],source_run=str(root),case_id=c['case_id'],
        actual_physics_executed=False,scene_modified=False,model_loaded=False,
        m0_passed=False,training_dataset_eligible=False,
        scope='Recounted raw PhysX contact impulses and mesh distance at saved actual 30 Hz HOLD poses; '
              'not a fresh contact test, dense swept clearance, real skin model or acoustic-coupling measurement')
    (args.output/'COMPLETE.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('actual_pose_samples','distance_min_m','distance_max_m',
        'geometry_membrane_recess_behind_lip_m','scene_modified','m0_passed')}),flush=True)


if __name__=='__main__':main()
