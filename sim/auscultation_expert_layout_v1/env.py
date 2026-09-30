"""PhysX-only OpenArm runtime. No CAN, ROS, or physical robot imports."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from .scene import ROOT, ROBOT, WRIST, FIXED, MOVING, prepare_stage

DT=1/240

class IsaacContactEnv:
    contact_groups = ('diaphragm','lip','contact_rim','carriage_other','frame','forbidden')

    def contact_group(self,a,b):
        patient=a.startswith('/World/Patient/') or b.startswith('/World/Patient/')
        module=a if a.startswith('/World/Stethoscope/') else b if b.startswith('/World/Stethoscope/') else ''
        if patient and module:
            if '/Fixed/' in module:return 'frame'
            if module.endswith('/Diaphragm'):return 'diaphragm'
            if '/RetentionLip/' in module:return 'lip'
            if '/ContactRim/' in module:return 'contact_rim'
            return 'carriage_other'
        if patient or module or 'openarm_left_link' in a or 'openarm_left_link' in b:
            return 'forbidden'
        return None

    def __init__(self, output_dir=None, *, stage_factory=None,
                 contact_offset_m=.1262, moving_mass_kg=.067155):
        import carb, omni.usd
        from pxr import UsdGeom, UsdPhysics, PhysicsSchemaTools
        from omni.physx import get_physx_simulation_interface
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
        self.out=Path(output_dir or ROOT/'results').resolve();self.out.mkdir(parents=True,exist_ok=True)
        self.contact_offset_m=float(contact_offset_m);self.moving_mass_kg=float(moving_mass_kg)
        stage=(stage_factory or prepare_stage)(self.out/'training_scene.usdc')
        self.base=np.array(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(ROBOT+'/openarm_left_link0'))).T
        omni.usd.get_context().open_stage(str(self.out/'training_scene.usdc'))
        # SimulationApp is owned by entrypoint; app.update needed before World.
        import omni.kit.app
        for _ in range(8): omni.kit.app.get_app().update()
        self.stage=omni.usd.get_context().get_stage()
        carb.settings.get_settings().set('/physics/updateToUsd',True)
        self.world=World(physics_dt=DT,rendering_dt=1/30,stage_units_in_meters=1.,physics_prim_path='/World/PhysicsScene')
        self.world.reset();self.world.play()
        self.robot=SingleArticulation(ROBOT,name='ausc_robot');self.robot.initialize()
        self.left=np.array([self.robot.get_dof_index(f'openarm_left_joint{i}') for i in range(1,8)])
        self.slider=self.robot.get_dof_index('PassiveCompression')
        self.wrist=SingleRigidPrim(WRIST,name='ausc_wrist');self.wrist.initialize()
        self.bodies=[]
        for p in self.stage.Traverse():
            path=str(p.GetPath())
            if p.HasAPI(UsdPhysics.RigidBodyAPI) and (path.startswith(ROBOT) or path in (FIXED,MOVING)):
                b=SingleRigidPrim(path,name=f'record_{len(self.bodies)}');b.initialize();self.bodies.append((path,b))
        self.contact_impulses=defaultdict(float);self.contact_pairs=defaultdict(float);self.contact_points=[]
        def callback(headers,data):
            for h in headers:
                a,b=[str(PhysicsSchemaTools.intToSdfPath(x)) for x in [h.collider0,h.collider1]]
                impulse=0.
                for i in range(h.contact_data_offset,h.contact_data_offset+h.num_contact_data):
                    impulse+=float(np.linalg.norm(data[i].impulse))
                    if hasattr(data[i],'position'):
                        self.contact_points.append({'colliders':[a,b],
                            'position_world':list(data[i].position),'impulse_Ns':list(data[i].impulse),
                            'separation_m':float(data[i].separation)})
                self.contact_pairs[tuple(sorted((a,b)))]+=impulse
                group=self.contact_group(a,b)
                if group is not None:self.contact_impulses[group]+=impulse
        self.contact_subscription=get_physx_simulation_interface().subscribe_contact_report_events(callback)
        self.t=0.;self.filtered_hall_force=0.;self.rows=[];self.poses=[];self.pose_row_indices=[]
        self.previous_tcp=None;self.last_contacts={};self.last_pairs={}

    def state(self):
        from scipy.spatial.transform import Rotation
        q_all=self.robot.get_joint_positions();dq_all=self.robot.get_joint_velocities()
        p,quat=self.wrist.get_world_pose()
        R=Rotation.from_quat(np.asarray(quat)[[1,2,3,0]]).as_matrix()
        compression=float(q_all[self.slider]);velocity=float(dq_all[self.slider])
        # Positive compression moves the carriage toward the wrist. The actual
        # diaphragm, not the adapter origin, defines the contact pose.
        tcp=np.asarray(p)+R[:,2]*(self.contact_offset_m-compression)
        outward=-R[:,2]
        hall_force=0. if compression<.0001 else max(0.,1.5+1000*compression+3.2*velocity+self.moving_mass_kg*9.81*outward[2])
        base_tcp=self.base[:3,:3].T@(tcp-self.base[:3,3])
        return {'q':np.asarray(q_all[self.left]),'dq':np.asarray(dq_all[self.left]),
                'tcp_world':tcp,'tcp_base':base_tcp,'rotation_world':R,
                'rotation_base':self.base[:3,:3].T@R,'axis_world':R[:,2],
                'compression_m':compression,'hall_raw_N':hall_force,
                'hall_N':self.filtered_hall_force,'time_s':self.t}

    def step(self, q_target, label='', target='', meta=None):
        from isaacsim.core.utils.types import ArticulationAction
        q_target=np.asarray(q_target,dtype=float)
        if q_target.shape!=(7,) or not np.isfinite(q_target).all():raise ValueError('Invalid joint command')
        self.robot.apply_action(ArticulationAction(joint_positions=q_target,joint_indices=self.left))
        self.contact_impulses.clear();self.contact_pairs.clear();self.contact_points.clear()
        self.world.step(render=False);self.t+=DT
        s=self.state()
        self.filtered_hall_force+=DT/(.02+DT)*(s['hall_raw_N']-self.filtered_hall_force)
        s['hall_N']=self.filtered_hall_force
        self.last_contacts={k:v/DT for k,v in self.contact_impulses.items()}
        self.last_pairs={' | '.join(k):v/DT for k,v in self.contact_pairs.items() if v/DT>.05}
        velocity=0. if self.previous_tcp is None else float(np.linalg.norm(s['tcp_world']-self.previous_tcp)/DT)
        self.previous_tcp=s['tcp_world'].copy()
        row={'time_s':self.t,'phase':label,'target':target,'compression_m':s['compression_m'],
             'hall_N':s['hall_N'],'tcp_speed_m_s':velocity,
             **{k+'_N':self.last_contacts.get(k,0.) for k in self.contact_groups},
             'contact_pairs_N':self.last_pairs.copy(),
             'contact_points':self.contact_points.copy(),
             'q':s['q'].tolist(),'dq':s['dq'].tolist(),'q_command':q_target.tolist(),
             'tcp_world':s['tcp_world'].tolist(),'axis_world':s['axis_world'].tolist()}
        row['tcp_base_m'] = s['tcp_base'].tolist()
        row['tcp_rotation_base'] = (self.base[:3,:3].T @ s['rotation_world']).tolist()
        if meta:row.update(meta)
        self.rows.append(row)
        if len(self.rows)%8==0:
            self.poses.append([np.r_[b.get_world_pose()[0],b.get_world_pose()[1]] for _,b in self.bodies])
            self.pose_row_indices.append(len(self.rows)-1)
        return s

    def save(self, name='expert'):
        with (self.out/(name+'_trace.jsonl')).open('w') as f:
            for row in self.rows:f.write(json.dumps(row)+'\n')
        np.savez_compressed(self.out/(name+'_replay.npz'),poses=np.asarray(self.poses),
                            row_indices=self.pose_row_indices,paths=[p for p,_ in self.bodies])
        (self.out/'world_from_left_base.json').write_text(json.dumps(self.base.tolist(),indent=2))
