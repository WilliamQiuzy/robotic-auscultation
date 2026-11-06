"""Synthetic convex-wedge/triangle-plate diagnostic, never a robot qualification."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys

import numpy as np

from .native_contact_evidence import contact_point_record,contact_header_record,native_pose_sample
from .retainer_geometry import ring_sector_meshes

DT=1/240
STEPS=12


def cases():
    result=[]
    for index,(feature,gap,speed,transformed,reverse) in enumerate(itertools.product(
            ('front','side','back'),(0.,.00015,.00035),(0.,.02),(False,True),(False,True))):
        result.append(dict(id='case_'+str(index).zfill(3),feature=feature,gap_m=gap,
            speed_m_s=speed,transformed=transformed,reverse_creation=reverse))
    return result


def support_face(feature):
    vertices,_=ring_sector_meshes()[0]
    if feature=='front':ids=[0,1,2,3];normal=np.array([0.,0.,-1.])
    elif feature=='back':ids=[4,5,6,7];normal=np.array([0.,0.,1.])
    elif feature=='side':
        ids=[1,2,5,6];normal=np.array([np.cos(np.pi/32),np.sin(np.pi/32),0.])
    else:raise ValueError('Unknown synthetic support feature')
    center=vertices[ids].mean(axis=0)
    return dict(outward_local=normal.tolist(),center_local=center.tolist(),support_m=float(normal@center))


def point_residuals(position,normal,separation,ring_index,world_from_body,plate_origin,
                    plate_normal,face):
    if ring_index not in (0,1):raise ValueError('Ordered native pair index required')
    p,n=np.asarray(position,float),np.asarray(normal,float)
    m=np.asarray(world_from_body,float)
    fn=np.asarray(face['outward_local'])
    residuals={};plate_residuals={}
    for a in (-1.,-.5,0.,.5,1.):
        candidate=p+a*separation*n
        local=m[:3,:3].T@(candidate-m[:3,3])
        residuals[str(a)]=float(fn@local-face['support_m'])
        plate_residuals[str(a)]=float(np.asarray(plate_normal)@(candidate-plate_origin))
    return dict(plate_raw_residual_m=plate_residuals['0.0'],plate_face_residual_m=plate_residuals,
        ring_face_residual_m=residuals,expected_pair_sign=1 if ring_index==0 else -1,
        evidence_only=True,production_rule_authorized=False)


def plate_fixture(center,wedge_to_plate_axis):
    axis=np.asarray(wedge_to_plate_axis,float)
    ref=np.array([0.,0.,1.]) if abs(axis[2])<.9 else np.array([1.,0.,0.])
    u=np.cross(axis,ref);u/=np.linalg.norm(u);v=np.cross(axis,u)
    points=np.array([center+sx*.06*u+sy*.06*v for sx,sy in
                    [(-1,-1),(1,-1),(1,1),(-1,1)]],dtype=np.float32).astype(float)
    normal=np.cross(points[2]-points[0],points[1]-points[0]);normal/=np.linalg.norm(normal)
    return dict(points=points.tolist(),plate_origin_world=points.mean(axis=0).tolist(),
        plate_normal_world=normal.tolist(),wedge_to_plate_axis_world=axis.tolist())


def evidence_status(specs,summaries):
    per_case=[]
    for c in specs:
        rows=[r for r in summaries if r['case']==c['id'] and r['execution_stage']=='controlled_step']
        positive=[r for r in rows if r['force_N']>0.]
        unavailable={}
        for row in rows:
            for field in row['unavailable']:unavailable[field]=unavailable.get(field,0)+1
        status=('unavailable_no_contacts' if not rows else
                'unavailable_missing_fields' if unavailable else 'available')
        per_case.append(dict(case=c['id'],feature=c['feature'],gap_m=c['gap_m'],speed_m_s=c['speed_m_s'],
            transformed=c['transformed'],reverse_creation=c['reverse_creation'],contacts=len(rows),
            positive_contacts=len(positive),control_status=status,unavailable_counts=unavailable,
            observed_retainer_pair_indices=sorted({r['retainer_pair_index'] for r in rows})))
    available=sum(c['control_status']=='available' for c in per_case)
    pair_indices={r['retainer_pair_index'] for r in summaries}
    return dict(cases=per_case,evidence_complete=available==len(specs),
        evidence_status='complete' if available==len(specs) else 'partial' if available else 'unavailable',
        positive_impulse_cases=sum(c['positive_contacts']>0 for c in per_case),
        native_pair_reversal_status='observed' if pair_indices=={0,1} else 'unavailable',
        native_pair_reversal_unavailable_reason=None if pair_indices=={0,1} else
            'Creation order does not establish native pair reversal; both indices were not observed.',
        warmup_contact_rows=sum(r['execution_stage']=='initialization_reset' for r in summaries))


def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def run(output):
    from scipy.spatial.transform import Rotation
    import omni.usd
    from pxr import UsdGeom,UsdPhysics,PhysxSchema,Gf,Vt,PhysicsSchemaTools
    from omni.physx import get_physx_interface,get_physx_simulation_interface
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleRigidPrim
    from omni.physx.bindings import _physx
    import omni.kit.app

    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    omni.usd.get_context().new_stage()
    stage=omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage,1.)
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    scene=UsdPhysics.Scene.Define(stage,'/World/PhysicsScene')
    scene.CreateGravityMagnitudeAttr(0.)
    vertices,faces=ring_sector_meshes()[0]
    specs=cases();by_collider={};bodies={};events=[];unexpected=[]
    context=dict(step=None,execution_stage='initialization_reset',pre={})
    for i,c in enumerate(specs):
        rotation=Rotation.from_euler('xyz',[19,27,31],degrees=True) if c['transformed'] else Rotation.identity()
        matrix=rotation.as_matrix();translation=np.array([(i%9)*.25,(i//9)*.25,1.])
        face=support_face(c['feature']);outward=matrix@np.asarray(face['outward_local'])
        plane=matrix@np.asarray(face['center_local'])+translation+c['gap_m']*outward
        fixture=plate_fixture(plane,outward);pts=np.asarray(fixture.pop('points'),dtype=np.float32)
        root='/World/Cases/'+c['id'];body_path=root+'/Wedge';shape_path=body_path+'/Shape';plate_path=root+'/Plate'
        c.update(body_path=body_path,shape_path=shape_path,plate_path=plate_path,
            **fixture,support_face=face,
            initial_body_position_world=translation.tolist(),initial_body_quaternion_xyzw=rotation.as_quat().tolist(),
            initial_velocity_world=(c['speed_m_s']*outward).tolist())
        def make_plate():
            mesh=UsdGeom.Mesh.Define(stage,plate_path)
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
            mesh.CreateFaceVertexCountsAttr([3,3]);mesh.CreateFaceVertexIndicesAttr([0,2,1,0,3,2])
            mesh.CreateSubdivisionSchemeAttr('none')
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('none')
            p=PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
            p.CreateContactOffsetAttr(.001);p.CreateRestOffsetAttr(0.)
            PhysxSchema.PhysxContactReportAPI.Apply(mesh.GetPrim()).CreateThresholdAttr(0.)
        def make_wedge():
            body=UsdGeom.Xform.Define(stage,body_path)
            body.AddTranslateOp().Set(Gf.Vec3d(*translation))
            q=rotation.as_quat();body.AddOrientOp().Set(Gf.Quatf(float(q[3]),Gf.Vec3f(*q[:3])))
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            mass=UsdPhysics.MassAPI.Apply(body.GetPrim());mass.CreateMassAttr(.1)
            mass.CreateCenterOfMassAttr(Gf.Vec3f(*vertices.mean(axis=0)))
            p=PhysxSchema.PhysxRigidBodyAPI.Apply(body.GetPrim())
            p.CreateDisableGravityAttr(True);p.CreateSolverPositionIterationCountAttr(32)
            p.CreateSolverVelocityIterationCountAttr(1)
            PhysxSchema.PhysxContactReportAPI.Apply(body.GetPrim()).CreateThresholdAttr(0.)
            mesh=UsdGeom.Mesh.Define(stage,shape_path)
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
            mesh.CreateFaceVertexCountsAttr([3]*len(faces));mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
            mesh.CreateSubdivisionSchemeAttr('none')
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
            p=PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
            p.CreateContactOffsetAttr(.00003);p.CreateRestOffsetAttr(0.)
        if c['reverse_creation']:make_plate();make_wedge()
        else:make_wedge();make_plate()
        by_collider[shape_path]=c;by_collider[plate_path]=c
    reader=get_physx_interface().get_rigidbody_transformation
    (output/'cases.json').write_text(json.dumps(specs,indent=2)+'\n')
    journal=(output/'native_journal.jsonl').open('x')
    callback_errors=[]
    def pose(c,label):
        row=native_pose_sample(reader,c['body_path'],label)
        row['world_from_wedge']=row.pop('world_from_carriage')
        return row
    def collect_callback(headers,data):
        pose_cache={}
        for h in headers:
            pair=[str(PhysicsSchemaTools.intToSdfPath(v)) for v in (h.collider0,h.collider1)]
            if not h.num_contact_data:continue
            a,b=[by_collider.get(path) for path in pair]
            if a is None or b is None or a['id']!=b['id']:
                unexpected.append(dict(pair=pair,count=h.num_contact_data));continue
            c=a;key=c['id']
            if key not in pose_cache:pose_cache[key]=pose(c,'callback_current')
            header=contact_header_record(h,PhysicsSchemaTools.intToSdfPath)
            for index in range(h.contact_data_offset,h.contact_data_offset+h.num_contact_data):
                record=contact_point_record(data[index],pair,actors=header['actors'])
                record.update(header)
                event=dict(event_index=len(events),case=key,physics_row=context['step'],raw=record,
                    execution_stage=context['execution_stage'],
                    pre=context['pre'].get(key,dict(available=False,error='No pre pose for SDK initialization step')),
                    callback_current=pose_cache[key],post=dict(available=False,error='No explicit post pose for SDK initialization step'))
                events.append(event)
                journal.write(json.dumps(dict(kind='contact_callback',event=event),allow_nan=False)+'\n')
        journal.flush()
    def callback(headers,data):
        # Native bindings may log and swallow a Python callback exception.
        # Persist it and reject the diagnostic after returning from physics.
        try:collect_callback(headers,data)
        except Exception as error:
            callback_errors.append(repr(error)[:1000])
            journal.write(json.dumps(dict(kind='callback_error',error=repr(error)[:1000]))+'\n')
            journal.flush()
    # SDK reset/play internally simulates. Subscribe before World construction,
    # retaining initialization events separately from the 12 controlled steps.
    subscription=get_physx_simulation_interface().subscribe_contact_report_events(callback)
    ticks=[];world=None
    try:
        world=World(physics_dt=DT,rendering_dt=DT,stage_units_in_meters=1.,physics_prim_path='/World/PhysicsScene')
        world.get_physics_context().set_gravity(0.)
        world.reset();world.play()
        for c in specs:
            body=SingleRigidPrim(c['body_path'],name=c['id']);body.initialize()
            body.set_linear_velocity(np.asarray(c['initial_velocity_world']))
            bodies[c['id']]=body
        controlled_start_time=float(world.current_time)
        if callback_errors:raise RuntimeError('Native contact callback failed: '+str(callback_errors))
        for step in range(STEPS):
            context.clear();context.update(step=step,execution_stage='controlled_step',
                pre={c['id']:pose(c,'pre_step_current') for c in specs})
            start=len(events);before=float(world.current_time);world.step(render=False)
            after=float(world.current_time)
            if abs(after-before-DT)>1e-7:raise ValueError('Unexpected synthetic physics clock')
            post={c['id']:pose(c,'post_step_current') for c in specs}
            for record in events[start:]:record['post']=post[record['case']]
            tick=dict(physics_row=step,time_before_s=before,time_after_s=after,events=len(events)-start)
            ticks.append(tick)
            journal.write(json.dumps(dict(kind='post_step',tick=tick,post=post,
                updated_event_indices=[r['event_index'] for r in events[start:]]),allow_nan=False)+'\n')
            journal.flush()
            if callback_errors:raise RuntimeError('Native contact callback failed: '+str(callback_errors))
        if unexpected:raise ValueError('Synthetic fixtures contacted another case')
    finally:
        subscription=None
        try:
            if world is not None:world.stop()
        finally:journal.close()
    spec_by_id={c['id']:c for c in specs}
    summaries=[]
    for record in events:
        c=spec_by_id[record['case']];raw=record['raw'];ri=raw['colliders'].index(c['shape_path'])
        result=dict(case=c['id'],feature=c['feature'],physics_row=record['physics_row'],
            execution_stage=record['execution_stage'],
            force_N=float(np.linalg.norm(raw['impulse_Ns'])/DT),retainer_pair_index=ri,
            separation_m=raw['separation_m'],face_index0=raw['face_index0'],face_index1=raw['face_index1'],
            native_field_errors=raw['native_field_errors'],by_pose={},unavailable=[])
        if raw['normal_world'] is None:result['unavailable'].append('normal_world')
        for label in ('pre','callback_current','post'):
            sample=record[label]
            if not sample['available']:result['unavailable'].append(label)
            if not sample['available'] or raw['normal_world'] is None:continue
            result['by_pose'][label]=point_residuals(raw['position_world'],raw['normal_world'],raw['separation_m'],ri,
                sample['world_from_wedge'],np.asarray(c['plate_origin_world']),np.asarray(c['plate_normal_world']),c['support_face'])
        summaries.append(result)
    with (output/'native_contacts.jsonl').open('x') as stream:
        for row in events:stream.write(json.dumps(row,allow_nan=False)+'\n')
    (output/'physics_ticks.json').write_text(json.dumps(ticks,indent=2)+'\n')
    (output/'residuals.json').write_text(json.dumps(summaries,indent=2)+'\n')
    status=evidence_status(specs,summaries)
    result=dict(schema='native_convex_triangle_contact_oracle_v1',
        execution_completed=True,completed=status['evidence_complete'],**status,
        controlled_simulation_steps=STEPS,controlled_start_time_s=controlled_start_time,
        initialization_step_count=None,initialization_step_count_unavailable_reason='SDK reset/play performs internal simulations without a public step counter; all observed callbacks retained.',
        physics_dt_s=DT,contact_rows=len(events),
        physics_settings=dict(gravity=0.,position_iterations=32,velocity_iterations=1,
            dynamic_mass_kg=.1,retainer_contact_offset_m=.00003,plate_contact_offset_m=.001,
            rest_offsets_m=0.,report_threshold_N=0.,report_threshold_is_not_acceptance_threshold=True),
        native_binding_file=_physx.__file__,native_binding_sha256=_sha(_physx.__file__),
        source_sha256=_sha(__file__),retainer_geometry_source_sha256=_sha(Path(__file__).with_name('retainer_geometry.py')),
        artifacts={p.name:dict(bytes=p.stat().st_size,sha256=_sha(p)) for p in output.iterdir() if p.is_file()},
        physical_validation=False,training_eligible=False,semantics_rule_qualified=False,
        test_payloads_opened=False,real_robot_operated=False,
        scope='Independent known front/side/back convex wedge vs static triangle mesh; no patient or robot; hypotheses only.')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    if sys.platform!='linux' or os.environ.get('AUSC_OWNED_RUN')!='1' or config['experiment']!='contact_semantics_oracle':
        raise RuntimeError('Finite owned Linux semantic diagnostic required')
    from isaacsim import SimulationApp
    # Default Kit fast shutdown terminates the process inside app.close(),
    # preventing both post-close completion and failure receipts.
    app=SimulationApp(dict(headless=True,fast_shutdown=False))
    close_returned=False
    try:
        try:result=run(args.output)
        finally:
            app.close();close_returned=True
    except BaseException as error:
        args.output.mkdir(parents=True,exist_ok=True)
        failure=dict(execution_completed=False,completed=False,error=repr(error)[:2000],
            app_close_returned=close_returned,
            evidence_complete=False,semantics_rule_qualified=False,
            partial_native_journal='native_journal.jsonl',training_eligible=False,
            artifacts={p.name:dict(bytes=p.stat().st_size,sha256=_sha(p)) for p in args.output.iterdir() if p.is_file()})
        (args.output/'FAILED.json').write_text(json.dumps(failure,indent=2)+'\n')
        if isinstance(error,SystemExit) and error.code in (None,0):
            raise RuntimeError('SDK requested a zero-code exit before diagnostic completion') from error
        raise
    result['runtime_shutdown']=dict(app_close_returned=close_returned,fast_shutdown=False,
        interpreter_exit_policy='success_only_os_exit_after_native_close_and_fsynced_receipt',
        reason='Avoid Python finalizers after native plugin teardown; historical nonzero exit cause remains unproven.')
    with (args.output/'COMPLETE.json').open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())


if __name__=='__main__':
    main()
    sys.stdout.flush();sys.stderr.flush()
    os._exit(0)
