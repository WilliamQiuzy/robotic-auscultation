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


def preparation_backend(config, cpu_prepare_only, platform, environment):
    """CPU authoring cannot enter any simulation branch or claim GPU ownership."""
    if platform != 'linux':
        raise RuntimeError('Linux scene preparation required')
    if cpu_prepare_only:
        if config.get('prepare_only') is not True or not config.get('patient_asset'):
            raise ValueError('CPU authoring requires prepare_only and authenticated patient')
        if environment.get('CUDA_VISIBLE_DEVICES') != '':
            raise RuntimeError('CPU authoring requires explicitly empty CUDA_VISIBLE_DEVICES')
        return 'cpu_usd_authoring_no_simulation_app'
    if environment.get('AUSC_OWNED_RUN') != '1':
        raise RuntimeError('Owned Linux Isaac required')
    return 'native_isaac_simulation_app'


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
    from .geometry_epoch import plane_lower_clearance
    planes=json.loads((output/'geometry/collision_geometry.json').read_text())['analytic_planes']
    class WithPlanes(CertifiedCollisionChecker):
        def check(self,q,**kwargs):
            result=super().check(q,**kwargs)
            for plane in planes:
                distance=plane_lower_clearance(self.bounds[self.dynamic],plane['normal'],plane['point'])
                if distance<kwargs.get('clearance',.003):
                    return dict(clear=False,minimum_distance_m=distance,pair=['dynamic bounds',plane['path']])
            return result
    checker=WithPlanes(kin,base,directory=output/'geometry')
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
            if config.get('axis_seed_search',False):
                proposal=kin.solve_ik(base[:3,:3].T@(center-base[:3,3]),
                    approach_axis_base=base[:3,:3].T@(-normal),seed=seed,max_starts=1,max_nfev=100)
                if proposal.success:
                    seed=proposal.q
                    rotation=base[:3,:3]@kin.tcp(seed)[:3,:3]
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
    parser.add_argument('--cpu-prepare-only',action='store_true',
        help='Author prepare-only USD using the native SDK, without SimulationApp or physics')
    args=parser.parse_args();config=json.loads(args.config.read_text());require_current(config)
    if 'matched_scene_eval' in config:
        from .m2_rebuild_paired_eval import seed_worker
        seed_worker(config,output=args.output)
    if os.environ.get('AUSC_VAL_SCENARIO_PROTOCOL') is not None:
        from .m2_given_contact_expanded_val_scenarios import seed_worker
        seed_worker(config,output=args.output)
    if 'm2_act_onpolicy_teacher_collection' in config:
        from .m2_act_onpolicy_collection import preflight
        preflight(config,output=args.output)
    if 'm2_given_contact_recovery' in config:
        from .m2_given_contact_recovery import preflight
        preflight(config)  # TRAIN metadata and original scene projection before Kit/patient payloads.
    formal_test=None
    if 'm2_given_contact_evaluation' in config:
        from . import m2_given_contact_evaluation as evaluation
        route=config[evaluation.ROUTE_KEY]
        schema=route.get('schema') if isinstance(route,dict) else None
        if schema==evaluation.PREP_ROUTE_SCHEMA and not args.cpu_prepare_only:
            raise ValueError('Sealed preparation requires explicit --cpu-prepare-only')
        if schema==evaluation.NATIVE_ROUTE_SCHEMA and args.cpu_prepare_only:
            raise ValueError('Sealed native execution cannot use --cpu-prepare-only')
        formal_test=evaluation.preflight(config,output=args.output)
    if config.get('m2_act_deployment',{}).get('schema')=='m2_act_pilot_history2_deployment_v2':
        from .m2_pilot_deployment import preflight
        preflight(config)  # VALIDATION identities before any scene/patient payload or SimulationApp
    if 'm2_given_contact_deployment' in config:
        from .m2_given_contact_deployment import preflight
        preflight(config,output=args.output)  # Formal metadata before the first patient/USD/SimulationApp read.
    if 'm2_given_contact_pilot_deployment' in config:
        from .m2_given_contact_pilot_deployment import preflight
        preflight(config,output=args.output)  # Diagnostic TRAIN/VAL binding before scene payloads or Kit.
    if 'm2_given_contact_expanded_pilot_deployment' in config:
        from .m2_given_contact_expanded_pilot_deployment import preflight
        preflight(config,output=args.output)
    if 'm2_given_contact_expanded_val_scenarios' in config:
        from .m2_given_contact_expanded_val_scenarios import preflight
        preflight(config,output=args.output)
    backend=preparation_backend(config,args.cpu_prepare_only,sys.platform,os.environ)
    shared_scene='compact_scene_storage' in config
    isolated_preflight=None
    if shared_scene:
        from .scene_store import validate_config,SDK_LAYER_SCHEMA
        if args.cpu_prepare_only or config.get('prepare_only') is True:
            raise ValueError('CPU preparation must retain the fixed prepared scene, not scene references')
        if config['compact_scene_storage'].get('schema')==SDK_LAYER_SCHEMA:
            from .native_usd_preflight import validate_before_kit
            isolated_preflight=validate_before_kit(config,args.config)
        else:validate_config(config)
    settling_inputs=None
    if 'm2_render_settling_probe' in config:
        from .render_settling_probe import preflight
        settling_inputs=preflight(config,args.config)
    args.output.mkdir(parents=True,exist_ok=False)
    if isolated_preflight is not None:write(args.output/'native_usd_preflight.json',isolated_preflight)
    source=Path(config['geometry_source'])
    if not shared_scene and hashlib.sha256(source.read_bytes()).hexdigest()!=config['geometry_source_sha256']:
        raise ValueError('Geometry changed')
    app=None
    if not args.cpu_prepare_only:
        from isaacsim import SimulationApp
        app_settings=dict(headless=True)
        # Isaac 6 fast shutdown exits the process from close(), before our final
        # CAS readback/byte ledger. The opted-in path must return from close().
        if shared_scene:
            from .native_lifecycle import shared_app_settings
            app_settings=shared_app_settings()
        app=SimulationApp(app_settings)
    result=None;shutdown_attempted=False;shutdown_returned=False
    try:
        if shared_scene:
            from .scene_store import prepared_runtime,finalize_job
            from .fresh_expert import run_probe
            stage,mass=prepared_runtime(config,args.output)
            result=run_probe(app,stage,args.output,config,mass)
            result['training_scene_sha256']=config['compact_scene_storage']['sources']['scene']['sha256']
            from .native_lifecycle import close_shared_runtime
            closing_app,app=app,None;shutdown_attempted=True
            result['runtime_shutdown']=close_shared_runtime(closing_app,
                progress_path=args.output/'runtime_shutdown_progress.jsonl')
            shutdown_returned=True  # Final SDK/CAS readback follows normal native close.
            finalize_job(args.output,result)
            with (args.output/'COMPLETE.json').open('rb') as stream:os.fsync(stream.fileno())
            return True
        from pxr import Usd, UsdGeom
        from .geometry_export import export_geometry
        from sim.auscultation_repair_v3.physics_profile import PhysicsProfile,apply_profile
        stage=Usd.Stage.Open(str(source));stage=Usd.Stage.Open(stage.Flatten())
        from .approved_camera import apply_approved_camera
        write(args.output/'approved_camera_restoration.json',apply_approved_camera(stage))
        patient=None
        if config.get('patient_asset'):
            stool=None
            if config['patient_asset'].get('layout') is not None:
                from .m2_stool_layout import install_stool_layout
                binding=config['stool_source']
                stool=install_stool_layout(stage,binding['directory'],binding,
                    config['patient_asset']['layout'],patient_spec=config['patient_asset'])
                write(args.output/'stool_layout_receipt.json',stool)
            from .patient_scene import install_patient
            patient=install_patient(stage,config['patient_asset'],**({'formal_config':config} if formal_test else {}))
            patient['non_patient_scene_unchanged_scope']='patient replacement only; after explicit stool installation'
            if stool is not None:
                patient['stool_installation_sha256']=hashlib.sha256(
                    (args.output/'stool_layout_receipt.json').read_bytes()).hexdigest()
            write(args.output/'patient_receipt.json',patient)
        mass=moving_inertia(stage,config['component_priors']);write(args.output/'mass_estimates.json',mass)
        profile=apply_profile(stage,PhysicsProfile(**config['physics']));write(args.output/'physics_profile.json',profile)
        from .passive_preload import apply_single_explicit_preload, verify_preload_readback
        preload=apply_single_explicit_preload(stage,config.get('fresh_expert'))
        if preload is not None:write(args.output/'passive_preload_authoring.json',preload)
        stage.GetRootLayer().customLayerData=dict(stage.GetRootLayer().customLayerData,wamGeometryVersion=GEOMETRY_ID)
        filename=args.output/'candidate_physics.usdc';stage.GetRootLayer().Export(str(filename))
        if preload is not None:
            saved=Usd.Stage.Open(str(filename))
            write(args.output/'passive_preload_saved_usd_readback.json',
                verify_preload_readback(saved,config['fresh_expert'],preload))
        entries=export_geometry(stage,args.output/'geometry')
        if config.get('prepare_only') is True:
            if patient is None:raise ValueError('M2 preparation requires an authenticated patient')
            from sim.auscultation_expert_layout_v1.scene import ROBOT
            base=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(
                stage.GetPrimAtPath(ROBOT+'/openarm_left_link0'))).T
            write(args.output/'world_from_left_base.json',base.tolist())
            write(args.output/'COMPLETE.json',dict(completed=True,
                geometry_id=GEOMETRY_ID,source_sha256=config['geometry_source_sha256'],
                effective_scene_sha256=hashlib.sha256(filename.read_bytes()).hexdigest(),
                patient=patient,stool=stool,collider_count=len(entries),
                geometry_artifacts={key:hashlib.sha256((args.output/'geometry'/name).read_bytes()).hexdigest()
                    for name,key in [('collision_geometry.json','collision_geometry_sha256'),
                                     ('collision_geometry.npz','collision_arrays_sha256')]},
                world_from_left_base=base.tolist(),physics_executed=False,
                physical_validation=False,training_eligible=False,preparation_backend=backend))
            return
        if settling_inputs is not None:
            from .render_settling_probe import run_probe
            run_probe(app,stage,args.output,config,mass,checked_inputs=settling_inputs)
            return
        if config.get('m2_sensor_probe'):
            from .m2_sensor_probe import run_probe
            result=run_probe(app,stage,args.output,config,mass)
            write(args.output/'COMPLETE.json',result)
            return
        if config.get('patient_preview'):
            if patient is None:raise ValueError('Patient preview requires explicit new asset')
            from .patient_scene import render_preview
            preview=render_preview(app,filename,args.output,patient)
            write(args.output/'COMPLETE.json',dict(completed=True,patient=patient,preview=preview,
                physical_validation=False,training_eligible=False))
            return
        if config.get('fresh_expert'):
            from .fresh_expert import run_probe
            result=run_probe(app,stage,args.output,config,mass)
            write(args.output/'COMPLETE.json',result)
            return
        if config.get('contact_probe'):
            from .geometry_contact_probe import run_probe
            result=run_probe(app,stage,args.output,config,mass)
            write(args.output/'COMPLETE.json',result)
            return
        result=search_targets(stage,entries,args.output,config)
        write(args.output/'COMPLETE.json',dict(geometry_id=GEOMETRY_ID,source_sha256=config['geometry_source_sha256'],
            effective_scene_sha256=hashlib.sha256(filename.read_bytes()).hexdigest(),
            ten_point_search_complete=len(result)==10,ik_feasible=sum(r['ik_feasible'] for r in result),
            collision_clear=sum(r['collision_clear'] for r in result),physical_validation=False,
            training_eligible=False,m0_pipeline_passed=True))
    except Exception as error:
        traceback.print_exc();write(args.output/'FAILED.json',dict(error=repr(error)))
        if shared_scene:
            if app is not None:
                from .native_lifecycle import close_shared_runtime
                closing_app,app=app,None;shutdown_attempted=True
                try:
                    close_shared_runtime(closing_app,progress_path=args.output/'runtime_shutdown_progress.jsonl')
                    shutdown_returned=True
                except Exception as close_error:
                    write(args.output/'FAILED.json',dict(error=repr(error),shutdown_error=repr(close_error)))
            if (not shutdown_attempted or shutdown_returned) and (args.output/'scene_ref.json').is_file():
                from .scene_store import finalize_job
                failed=result if isinstance(result,dict) else {}
                failed.update(success=False,compact_capture_complete=False,physical_validation=False,
                    training_eligible=False,failure_reason=repr(error),
                    training_scene_sha256=config['compact_scene_storage']['sources']['scene']['sha256'])
                try:finalize_job(args.output,failed)
                except Exception as final_error:
                    # Integrity failures already replace stale success and count
                    # FAILED.json; preserve the original exception for the owner.
                    print('COMPACT_SCENE_FAILURE_FINALIZATION',repr(final_error),flush=True)
        raise
    finally:
        if app is not None:app.close()


def native_entrypoint():
    # Shared captures finalize after close. Avoid Python native finalizers only
    # after native shutdown and byte/integrity receipts have completed normally.
    shared_finalized=main()
    if shared_finalized is True:
        sys.stdout.flush();sys.stderr.flush();os._exit(0)


if __name__=='__main__':native_entrypoint()
