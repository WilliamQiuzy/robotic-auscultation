"""Force-source diagnostic from a fresh simulation reset, not an expert rollout.

Free-space reset requires unchanged 3 mm clearance. Contact geometry uses the
existing finite-head shell threshold (0.3 mm, audit_tool_fit.py); robot, fixed
frame and clamp retain 3 mm. This probe cannot qualify a transfer or M2 episode.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from .geometry_epoch import contact_frame, GEOMETRY_ID, qualify_contact, plane_lower_clearance


def run_probe(app,stage,output,config,mass):
    from pxr import UsdGeom
    from sim.auscultation_expert_layout_v1.env import IsaacContactEnv,DT
    from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
    from sim.auscultation_expert_layout_v1.posture import PostureLimits
    from sim.auscultation_expert_layout_v1.pose_ik import solve_posture_ik
    from sim.auscultation_skill_v1.certified_collision import CertifiedCollisionChecker
    from sim.auscultation_expert_layout_v1.scene import ROBOT
    import fcl
    binding=config['contact_probe'];source=Path(binding['reachability'])
    if hashlib.sha256(source.read_bytes()).hexdigest()!=binding['sha256']:raise ValueError('Fresh reachability changed')
    report=json.loads(source.read_text())
    if report['geometry_id']!=GEOMETRY_ID:raise ValueError('Wrong probe geometry')
    row=next(r for r in report['points'] if r['name']==binding['target'])
    candidate=row['candidates'][binding['candidate_index']]
    center=np.asarray(row['fresh_target_world_m']);normal=np.asarray(row['normal_world'])
    rotation=np.asarray(candidate['rotation_world']);base=np.asarray(report['world_from_left_base'])
    kin=OpenArmKinematics(contact_frame=contact_frame)
    limits=PostureLimits.from_dict(config['posture_limits'],kin)
    checker=CertifiedCollisionChecker(kin,base,directory=output/'geometry')
    geometry=json.loads((output/'geometry/collision_geometry.json').read_text())
    # Independent contact-pose check: only the intended diaphragm is exempt.
    # The previously existing shell-fit threshold is explicit, never called a 3 mm pass.
    q=np.asarray(candidate['q']);checker.update(q)
    minimum_shell=.0003;rejected=[]
    for i,j in checker.pairs:
        a,b=checker.entries[i],checker.entries[j]
        patient=b['path'].startswith('/World/Patient/')
        diaphragm=a['path'].endswith('/Diaphragm')
        if patient and diaphragm:continue
        head=patient and a['path'].split('/')[-1] in ('DeviceEdge','DeviceCone','DeviceDial')
        threshold=.0003 if head else .003
        gap=np.maximum(0.,np.maximum(checker.bounds[i,0]-checker.bounds[j,1],checker.bounds[j,0]-checker.bounds[i,1]))
        if np.linalg.norm(gap)>=threshold:continue
        d=float(fcl.distance(checker.objects[i],checker.objects[j]))
        if not np.isfinite(d):raise ValueError('Nonfinite contact geometry')
        if head:minimum_shell=min(minimum_shell,d)
        if d<threshold-1e-8:rejected.append(dict(pair=[a['path'],b['path']],distance_m=d,threshold_m=threshold))
    if rejected:raise ValueError('Finite-head diagnostic pose rejected: '+str(rejected[:3]))
    def solve(gap,seed):
        q,ok,pe,re=solve_posture_ik(kin,base,limits,base[:3,:3].T@(center+normal*gap-base[:3,3]),
            base[:3,:3].T@rotation,seed,maxiter=100,contact=True)
        if not ok:raise ValueError('Fresh normal contact IK failed')
        return q
    initial_gap=.012
    q=solve(initial_gap,q)
    free=checker.check(q,clearance=.003)
    if not free['clear']:raise ValueError('Unsafe reset pose: '+str(free))
    for plane in geometry['analytic_planes']:
        if plane_lower_clearance(checker.bounds[checker.dynamic],plane['normal'],plane['point'])<.003:
            raise ValueError('Unsafe reset ground clearance')
    # All approach samples are generated fresh; the reset is explicitly not a home transfer.
    def factory(destination):
        stage.GetRootLayer().Export(str(destination));return stage
    env=IsaacContactEnv(output/'contact',stage_factory=factory,contact_offset_m=.132,
        moving_mass_kg=mass['mass_kg'])
    np.testing.assert_allclose(env.base,base,atol=1e-7)
    env.robot.set_joint_positions(q,joint_indices=env.left)
    env.robot.set_joint_velocities(np.zeros(7),joint_indices=env.left)
    reason=None;phase='SETTLE';gap=initial_gap;hold=0.;completed=False
    nominal_q=q.copy();frames=[]
    annotator=product=None
    try:
        import omni.replicator.core as rep
        product=rep.create.render_product('/World/RobotCam',(640,360))
        annotator=rep.AnnotatorRegistry.get_annotator('rgb');annotator.attach([product])
        for index in range(60*240):
            state=env.state()
            if index%12==0:
                if index<240:phase='SETTLE'
                elif phase=='SETTLE':phase='CONTACT'
                if phase=='CONTACT':
                    velocity=float(np.clip((state['hall_N']-4.)*.001,-.0014,.0014))
                    gap+=velocity*.05
                    if 3.5<=state['hall_N']<=4.5:hold+=.05
                    else:hold=0.
                    if hold>=3.5:phase='RETRACT'
                elif phase=='RETRACT':gap+=.005*.05
                if gap<-.006:raise ValueError('Normal travel bound')
                next_q=solve(gap,nominal_q)
                if np.max(abs(next_q-nominal_q))/.05>.175:raise ValueError('Joint command speed bound')
                block=np.linspace(nominal_q,next_q,13)[1:];nominal_q=next_q
            state=env.step(block[index%12],phase,binding['target'],dict(diagnostic=True,training_eligible=False))
            contacts=env.last_contacts
            if sum(contacts.values())>6. or state['hall_N']>=5.5 or state['compression_m']>.006:
                raise ValueError('Original force/stroke stop')
            if any(contacts.get(k,0.)>.5 for k in ('carriage_other','frame','forbidden')):
                raise ValueError('Non-diaphragm physical contact stop')
            if contacts.get('lip',0.)>0. or contacts.get('contact_rim',0.)>0.:
                raise ValueError('Lip or clamp rim contact; diagnostic failed')
            if not limits.evaluate(kin,base,state['q'],observed=True,contact=True)['clear']:
                raise ValueError('Observed posture stop')
            if index%24==0:
                env.world.render()
                frame=np.asarray(annotator.get_data())
                if frame.shape==(360,640,4):frames.append(frame[:,:,:3].copy())
            if phase=='RETRACT' and gap>=initial_gap and state['hall_N']<.2:
                completed=True;break
    except Exception as error:
        reason=repr(error)
    finally:
        env.save('geometry_probe')
        if annotator is not None:annotator.detach()
        if product is not None:product.destroy()
    physics=qualify_contact(env.rows)
    result=dict(geometry_id=GEOMETRY_ID,physical_validation=bool(completed and physics['passed']),
        source_sha256=config['geometry_source_sha256'],diagnostic_completed=True,withdrawal_complete=completed,
        force_audit=physics,failure_reason=reason,physics_steps=len(env.rows),
        scope='Actual approved patient contact from a fresh free-space reset; no home transfer/expert/student qualification',
        contact_shell_threshold_m=.0003,contact_shell_minimum_m=minimum_shell,
        free_space_threshold_m=.003,training_eligible=False,learned_policy=False)
    if frames:
        import subprocess
        video=output/'actual_contact_probe.mp4'
        command=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','640x360',
            '-r','10','-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p',str(video)]
        proc=subprocess.run(command,input=np.asarray(frames,dtype=np.uint8).tobytes(),capture_output=True)
        if proc.returncode:raise RuntimeError('Video encode failed: '+proc.stderr.decode()[-500:])
        result['video_frames']=len(frames)
    return result
