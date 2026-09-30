"""Fresh ten-point geometry qualification in owned Linux Isaac; no old paths."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from .geometry_epoch import GEOMETRY_ID, contact_frame, require_current


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def moving_inertia(stage, priors):
    """Re-estimate translated head and opened bore; never claim bench calibration."""
    import trimesh
    from pxr import UsdGeom, UsdPhysics, Gf
    from scipy.spatial.transform import Rotation
    from .tool_protrusion import MOVING, HEAD_VISUALS, HEAD_SHIFT_M
    parts=[]
    for old in priors['parts']:
        mass=float(old['mass_kg_estimate'])
        center=np.array(old['center_of_mass_m_estimate'])
        inertia=np.array(old['inertia_about_com_kg_m2_estimate'])
        if old['name'] in HEAD_VISUALS:
            center[2]+=HEAD_SHIFT_M
        elif old['name'] in ('left_clamshell','right_clamshell'):
            mesh=UsdGeom.Mesh(stage.GetPrimAtPath(MOVING+'/Visuals/'+old['name']))
            ids=np.asarray(mesh.GetFaceVertexIndicesAttr().Get());faces=[];index=0
            for count in mesh.GetFaceVertexCountsAttr().Get():
                poly=ids[index:index+count];index+=count
                faces.extend([[poly[0],poly[i],poly[i+1]] for i in range(1,count-1)])
            shape=trimesh.Trimesh(np.asarray(mesh.GetPointsAttr().Get()),faces,process=True)
            if not shape.is_watertight or shape.volume<=0:
                raise ValueError('Opened clamp mesh must have closed positive volume')
            shape.density=old['density_kg_m3_estimate']
            mass=float(shape.mass);center=shape.center_mass;inertia=shape.moment_inertia
        parts.append(dict(name=old['name'],mass_kg=mass,com_m=center.tolist(),inertia_kg_m2=inertia.tolist()))
    mass=sum(p['mass_kg'] for p in parts)
    center=sum(p['mass_kg']*np.array(p['com_m']) for p in parts)/mass
    inertia=np.zeros((3,3))
    for part in parts:
        d=np.array(part['com_m'])-center
        inertia+=np.array(part['inertia_kg_m2'])+part['mass_kg']*(d@d*np.eye(3)-np.outer(d,d))
    values,axes=np.linalg.eigh(inertia)
    if np.linalg.det(axes)<0:axes[:,0]*=-1
    quat=Rotation.from_matrix(axes).as_quat()
    if min(values)<=0:raise ValueError('Non-positive carriage inertia')
    api=UsdPhysics.MassAPI(stage.GetPrimAtPath(MOVING))
    before={a.GetName():str(a.Get()) for a in stage.GetPrimAtPath(MOVING).GetAttributes() if a.GetName().startswith('physics:')}
    api.CreateMassAttr(mass);api.CreateCenterOfMassAttr(Gf.Vec3f(*center))
    api.CreateDiagonalInertiaAttr(Gf.Vec3f(*values))
    api.CreatePrincipalAxesAttr(Gf.Quatf(float(quat[3]),Gf.Vec3f(*quat[:3])))
    return dict(mass_kg=mass,com_m=center.tolist(),inertia_kg_m2=inertia.tolist(),
        principal_values=values.tolist(),parts=parts,before=before,
        status='CAD volume/density and nominal head mass estimates; not bench measurements')


def search_targets(stage, entries, output, config):
    from pxr import UsdGeom
    from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
    from sim.auscultation_expert_layout_v1.scene import ROBOT
    from sim.auscultation_expert_layout_v1.posture import PostureLimits
    from sim.auscultation_expert_layout_v1.pose_ik import solve_posture_ik, IKBudgetExpired
    from sim.auscultation_skill_v1.certified_collision import CertifiedCollisionChecker
    from scipy.spatial.transform import Rotation
    import trimesh
    base=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(ROBOT+'/openarm_left_link0'))).T
    kin=OpenArmKinematics(contact_frame=contact_frame)
    limits=PostureLimits.from_dict(config['posture_limits'],kin)
    checker=CertifiedCollisionChecker(kin,base,directory=output/'geometry')
    arrays=np.load(output/'geometry/collision_geometry.npz')
    meshes=[]
    for e in entries:
        if not e['path'].startswith('/World/Patient/'):continue
        mesh=trimesh.Trimesh(arrays[e['key']+'_v'],arrays[e['key']+'_f'],process=False)
        mesh.apply_transform(np.asarray(e['world_from_body']));meshes.append(mesh)
    patient=trimesh.util.concatenate(meshes)
    lower=np.array(limits.joint_lower_rad);upper=np.array(limits.joint_upper_rad)
    rng=np.random.default_rng(930)
    seeds=[np.zeros(7),(lower+upper)/2]+[rng.uniform(lower,upper) for _ in range(config['search_starts']-2)]
    rows=[]
    for target in config['targets']:
        start=time.monotonic();deadline=start+config['per_target_seconds']
        position=np.array(target['position_world_m']);normal=np.array(target['normal_world']);normal/=np.linalg.norm(normal)
        nearest,distance,triangle=patient.nearest.on_surface([position])
        # Regenerate local normal from current patient triangles, preserving original outward sign.
        centers=patient.triangles_center
        nearby=np.linalg.norm(centers-position,axis=1)<.022
        fitted=np.sum(patient.face_normals[nearby]*patient.area_faces[nearby,None],axis=0)
        if np.linalg.norm(fitted)<1e-8:fitted=patient.face_normals[triangle[0]].copy()
        fitted/=np.linalg.norm(fitted)
        if fitted@normal<0:fitted=-fitted
        center=nearest[0];normal=fitted
        axis=-normal;x=np.cross([0,0,1],axis);x/=np.linalg.norm(x)
        reference=np.column_stack([x,np.cross(axis,x),axis])
        candidates=[];attempts=0;failed={}
        # No saved plan, q, precontact distance or previous handoff tolerance enters this search.
        for index,seed in enumerate(seeds):
            if time.monotonic()>=deadline:break
            rotation=reference@Rotation.from_rotvec([0,0,np.deg2rad((0,45,-45,90,-90,180)[index%6])]).as_matrix()
            try:
                q,ok,pe,re=solve_posture_ik(kin,base,limits,base[:3,:3].T@(center-base[:3,3]),
                    base[:3,:3].T@rotation,seed,maxiter=120,deadline=deadline,contact=True)
            except IKBudgetExpired:break
            attempts+=1
            if not ok:failed['ik']=failed.get('ik',0)+1;continue
            collision=checker.check(q,allow_patient_contact=True,details=True)
            candidate=dict(q=q.tolist(),position_error_m=pe,rotation_error_rad=re,
                j2_upper_urdf_margin_deg=float(np.rad2deg(kin.upper_limits[1]-q[1])),
                j2_upper_safety_margin_deg=float(np.rad2deg(upper[1]-q[1])),
                minimum_joint_safety_margin_deg=float(np.rad2deg(np.min(np.r_[q-lower,upper-q]))),
                rotation_world=rotation.tolist(),collision=collision,
                posture=limits.evaluate(kin,base,q,contact=True))
            candidates.append(candidate)
        candidates.sort(key=lambda r:(not r['collision']['clear'],-r['minimum_joint_safety_margin_deg']))
        row=dict(name=target['name'],fresh_target_world_m=center.tolist(),normal_world=normal.tolist(),
            raw_target_world_m=position.tolist(),surface_projection_distance_m=float(distance[0]),
            attempts=attempts,failures=failed,ik_feasible=bool(candidates),
            collision_clear=any(c['collision']['clear'] for c in candidates),candidates=candidates,
            elapsed_s=time.monotonic()-start)
        rows.append(row);write(output/'reachability.json',dict(geometry_id=GEOMETRY_ID,points=rows,
            world_from_left_base=base.tolist(),complete=len(rows)==10,physics_executed=False,training_eligible=False))
        print('REACHABILITY',row['name'],row['ik_feasible'],row['collision_clear'],len(candidates),flush=True)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();config=json.loads(args.config.read_text());require_current(config)
    if sys.platform!='linux' or os.environ.get('AUSC_OWNED_RUN')!='1':raise RuntimeError('Owned Linux Isaac required')
    args.output.mkdir(parents=True,exist_ok=False)
    source=Path(config['geometry_source'])
    if hashlib.sha256(source.read_bytes()).hexdigest()!=config['geometry_source_sha256']:raise ValueError('Geometry changed')
    from isaacsim import SimulationApp
    app=SimulationApp(dict(headless=True))
    try:
        from pxr import Usd, UsdGeom
        from .geometry_export import export_geometry
        from sim.auscultation_repair_v3.physics_profile import PhysicsProfile,apply_profile
        stage=Usd.Stage.Open(str(source));stage=Usd.Stage.Open(stage.Flatten())
        mass=moving_inertia(stage,config['component_priors']);write(args.output/'mass_estimates.json',mass)
        profile=apply_profile(stage,PhysicsProfile(**config['physics']));write(args.output/'physics_profile.json',profile)
        stage.GetRootLayer().customLayerData=dict(stage.GetRootLayer().customLayerData,wamGeometryVersion=GEOMETRY_ID)
        filename=args.output/'candidate_physics.usdc';stage.GetRootLayer().Export(str(filename))
        entries=export_geometry(stage,args.output/'geometry')
        result=search_targets(stage,entries,args.output,config)
        write(args.output/'COMPLETE.json',dict(geometry_id=GEOMETRY_ID,source_sha256=config['geometry_source_sha256'],
            effective_scene_sha256=hashlib.sha256(filename.read_bytes()).hexdigest(),
            ten_point_search_complete=len(result)==10,ik_feasible=sum(r['ik_feasible'] for r in result),
            collision_clear=sum(r['collision_clear'] for r in result),physical_validation=False,
            training_eligible=False,m0_pipeline_passed=True))
    except Exception as error:
        traceback.print_exc();write(args.output/'FAILED.json',dict(error=repr(error)))
        raise
    finally:app.close()


if __name__=='__main__':main()
