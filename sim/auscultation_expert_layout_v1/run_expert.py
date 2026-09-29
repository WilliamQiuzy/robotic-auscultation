"""Run one fully measured, fail-closed simulated contact episode."""
import argparse,json,os,time,hashlib
from dataclasses import asdict,replace
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--point',default='R1');args=parser.parse_args()
os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
from isaacsim import SimulationApp
app=SimulationApp({'headless':True})
import numpy as np
from .env import IsaacContactEnv,DT
from .scene import ROOT
from .kinematics import OpenArmKinematics
from .targets import load_targets
from .phase_manager import PhaseManager,PhaseConfig,Phase,ContactObservation

plan_file=ROOT/'results'/('planned_paths_'+args.point+'.json')
plan=json.loads(plan_file.read_text());row=next(p for p in plan['points'] if p['name']==args.point)
if not row['success']:raise RuntimeError('No checked motion plan')
out=ROOT/'results'/('expert_'+args.point)
env=IsaacContactEnv(out);kin=OpenArmKinematics()
target_path=plan.get('targets_file')
if target_path:target_path=ROOT/'results'/Path(target_path).name
target=next(t for t in load_targets(env.base,path=target_path) if t.name==args.point)
normal=np.asarray(row.get('command_normal_world',target.normal_world));normal/=np.linalg.norm(normal)
normal_base=env.base[:3,:3].T@normal
rotation=np.asarray(row['command_rotation_world']) if 'command_rotation_world' in row else None
rotation_base=env.base[:3,:3].T@rotation if rotation is not None else None
# On a curved rigid surface the membrane center can stand off while its rim
# actually touches. Only the diaphragm and its adjacent front rings may touch.
config=PhaseConfig(contact_normal_tolerance_m=.006,approach_timeout_s=120,episode_timeout_s=180,
                   max_compression_m=.006,hold_tangential_tolerance_m=.003)
manager=PhaseManager(config)
outcome={'success':False,'training_accepted':False,'target':target.name,'scope':'simulation contact mechanics only; no acoustic or clinical success',
         'phase_config':asdict(config),'plan_sha256':hashlib.sha256(plan_file.read_bytes()).hexdigest(),
         'world_target_m':target.position_world.tolist(),'command_normal_world':normal.tolist(),
         'point_provenance':target.provenance,'warnings':['Finite contact patch uses up to6mm center standoff; target center is unchanged.']}
q=np.zeros(7);face_filtered=0.;total_filtered=0.;audit_hold=0.;best_audit_hold=0.;control_rows=[]

def metrics(s):
    error=s['tcp_world']-target.position_world;gap=float(error@normal)
    return {'gap_m':gap,'tangent_m':float(np.linalg.norm(error-gap*normal)),
            'angle_deg':float(np.rad2deg(np.arccos(np.clip(s['axis_world']@(-normal),-1,1))))}

def tick(command,subphase):
    global face_filtered,total_filtered,audit_hold,best_audit_hold
    s=env.step(command,manager.state.phase.value,target.name,{'subphase':subphase})
    m=metrics(s);env.rows[-1].update(m)
    obs=ContactObservation(s['hall_N'],max(0.,s['compression_m']),m['gap_m'],m['tangent_m'],
        m['angle_deg'],env.rows[-1]['tcp_speed_m_s'],tcp_position_m=tuple(s['tcp_world']))
    previous=manager.state;state=manager.update(obs,DT)
    face=sum(env.last_contacts.get(k,0.) for k in ('diaphragm','lip','contact_rim'))
    total=sum(env.last_contacts.get(k,0.) for k in ('diaphragm','lip','contact_rim','carriage_other','frame'))
    alpha=DT/(.02+DT);face_filtered+=alpha*(face-face_filtered);total_filtered+=alpha*(total-total_filtered)
    real_contact=face_filtered>.15 and face_filtered>.1*max(total_filtered,.01)
    if previous.phase==Phase.HOLD and state.hold_elapsed>previous.hold_elapsed and real_contact:audit_hold+=DT
    elif state.phase==Phase.HOLD:audit_hold=0.
    best_audit_hold=max(best_audit_hold,audit_hold)
    env.rows[-1].update(hold_elapsed_s=state.hold_elapsed,audit_hold_s=audit_hold,
        front_contact_filtered_N=face_filtered,total_filtered_N=total_filtered)
    for key in ['frame','forbidden','carriage_other']:
        if env.last_contacts.get(key,0.)>.5:raise RuntimeError('Rejected '+key+' contact: '+str(env.last_pairs))
    if total>6. or s['compression_m']>.006:raise RuntimeError('Independent physical force/stroke limit')
    return s,state

try:
    for _ in range(240):s,state=tick(q,'settle')
    for destination in row['transfer_q'][1:]:
        destination=np.asarray(destination);start=q.copy();seconds=max(1.,1.875*np.max(np.abs(destination-start))/.25)
        steps=int(np.ceil(seconds/DT))
        for i in range(1,steps+1):
            u=i/steps;blend=10*u**3-15*u**4+6*u**5;q=start+(destination-start)*blend
            s,state=tick(q,'transfer')
        q=destination
    for _ in range(240):s,state=tick(q,'align')
    gap=.06
    for i in range(int(80/DT)):
        if i%12==0:
            s=env.state();state=manager.state
            if state.phase==Phase.RETRACT:speed=.01
            elif s['hall_N']>.3 or s['compression_m']>.00015:speed=float(np.clip((s['hall_N']-config.target_force_N)*.001,-.002,.002))
            else:speed=-.012 if gap>.020 else -.002
            gap+=speed*12*DT
            if gap<-.01:raise RuntimeError('Contact search exceeded displacement budget')
            kw={'orientation_base':rotation_base} if rotation_base is not None else {'approach_axis_base':-normal_base}
            sol=kin.solve_ik(target.position_base+normal_base*gap,seed=np.clip(q,kin.lower_limits,kin.upper_limits),
                            max_starts=1,max_nfev=80,**kw)
            if not sol.success:raise RuntimeError('Pose IK failed in '+state.phase.value)
            q=sol.q
            control_rows.append({'physics_row':len(env.rows),'q_command':q.tolist(),
                                 'phase':state.phase.value,'command_gap_m':gap})
        s,state=tick(q,'contact_cycle')
        if i%240==0:print('EXPERT',args.point,state.phase.value,metrics(s),'hall',s['hall_N'],
             'front_contact',face_filtered,'hold',state.hold_elapsed,'audit',audit_hold,flush=True)
        if state.completed:break
    if not manager.state.success:raise RuntimeError('State machine failed: '+str(manager.state.failure_reason))
    if best_audit_hold<config.hold_duration_s-2*DT:raise RuntimeError('Observable hold lacked independently verified front contact')
    outcome.update(success=True,training_accepted=True,verified_front_contact_hold_s=best_audit_hold)
    print('EXPERT_ACCEPTED',target.name,best_audit_hold,flush=True)
except Exception as error:
    outcome['failure_reason']=str(error)
    print('EXPERT_REJECTED',target.name,str(error),flush=True)
finally:
    outcome.update(final_phase=manager.state.phase.value,observed_hold_s=manager.state.hold_elapsed,
                   verified_front_contact_hold_s=best_audit_hold,physics_steps=len(env.rows))
    env.save('expert')
    (out/'outcome.json').write_text(json.dumps(outcome,indent=2))
    (out/'control_commands.json').write_text(json.dumps(control_rows))
    app.close()
if not outcome['success']:raise SystemExit(1)
