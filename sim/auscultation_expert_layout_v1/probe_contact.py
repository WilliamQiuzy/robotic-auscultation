"""Physical expert development probe; produces evidence, never accepted IL data."""
import argparse,os,json,sys
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--point',default='R1');args=parser.parse_args()
os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
from isaacsim import SimulationApp
app=SimulationApp({'headless':True})
import numpy as np
from .env import IsaacContactEnv,DT
from .kinematics import OpenArmKinematics
from .targets import load_targets
from .scene import ROOT

env=IsaacContactEnv(ROOT/'results'/('probe_'+args.point));kin=OpenArmKinematics()
planned=json.loads((ROOT/'results'/('planned_paths_'+args.point+'.json')).read_text())
target_path=planned.get('targets_file')
if target_path: target_path=ROOT/'results'/Path(target_path).name
targets={t.name:t for t in load_targets(env.base,path=target_path)};target=targets[args.point]
row=next(p for p in planned['points'] if p['name']==args.point)
if not row['success']:raise RuntimeError('No geometrically checked approach route')
def metrics(state):
    error=state['tcp_world']-target.position_world
    gap=float(error@target.normal_world)
    return {'gap_m':gap,'tangent_m':float(np.linalg.norm(error-gap*target.normal_world)),
            'angle_deg':float(np.rad2deg(np.arccos(np.clip(state['axis_world']@(-target.normal_world),-1,1))))}
q=np.zeros(7);outcome={'scope':'debug-only contact probe; never accepted imitation data','success':False}
def checked_step(q,phase,meta=None):
    s=env.step(q,phase,target.name,meta)
    env.rows[-1].update(metrics(s))
    force=sum(env.last_contacts.get(k,0.) for k in ('diaphragm','lip','carriage_other','frame'))
    if env.last_contacts.get('frame',0)>.5:raise RuntimeError('Frame contacted patient: '+str(env.last_pairs))
    if env.last_contacts.get('forbidden',0)>.5:raise RuntimeError('Illegal contact: '+str(env.last_pairs))
    if force>6. or s['compression_m']>.006:raise RuntimeError('Force/stroke exceeded probe limit')
    return s
try:
    for _ in range(240):checked_step(q,'settle')
    for destination in row['transfer_q'][1:]:
        destination=np.asarray(destination);start=q.copy()
        seconds=max(1.,1.875*float(np.max(np.abs(destination-start)))/.25)
        for step in range(1,int(seconds/DT)+1):
            u=min(1,step*DT/seconds);smooth=10*u**3-15*u**4+6*u**5
            q=start+(destination-start)*smooth
            s=checked_step(q,'transfer')
        q=destination
    for _ in range(360):s=checked_step(q,'align')
    print('PRECONTACT',metrics(s),s['q'],flush=True)
    gap=.06;stable=0.;hold_started=False;seen=False
    for tick in range(12000):
        if tick%12==0:
            s=env.state();m=metrics(s)
            if s['hall_N']>2. and s['compression_m']>.0002:seen=True
            if seen:
                speed=np.clip((s['hall_N']-4.)*.001,-.002,.002)
            else:speed=-.012 if gap>.02 else -.002
            gap+=float(speed)*12*DT
            sol=kin.solve_ik(target.position_base+target.normal_base*gap,target.approach_axis_base,seed=np.clip(q,kin.lower_limits,kin.upper_limits),max_starts=1,max_nfev=60)
            if not sol.success:raise RuntimeError('IK lost at '+str(gap))
            q=sol.q
        s=checked_step(q,'hold_probe' if seen else 'contact_probe',{'command_gap_m':gap})
        m=metrics(s)
        if seen and 3.5<=s['hall_N']<=4.5 and env.last_contacts.get('diaphragm',0)>.15 and m['tangent_m']<.003 and abs(m['gap_m'])<.003 and m['angle_deg']<5:stable+=DT
        else:stable=0.
        if tick%240==0:print('CONTACT',tick,metrics(s),'hall',s['hall_N'],'qmm',s['compression_m']*1000,'physx',env.last_contacts,flush=True)
        if stable>=3.5:break
        if gap<-.012:raise RuntimeError('No valid contact within search stroke')
    if stable<3.5:raise RuntimeError('No continuous valid diaphragm hold before timeout')
    for tick in range(2400):
        if tick%12==0:
            gap=min(.06,gap+.01*12*DT)
            sol=kin.solve_ik(target.position_base+target.normal_base*gap,target.approach_axis_base,seed=q,max_starts=1,max_nfev=60)
            if not sol.success:raise RuntimeError('Retraction IK failed')
            q=sol.q
        s=checked_step(q,'retract')
        if gap>=.06 and s['compression_m']<.0001:break
    if metrics(s)['gap_m']<.04 or s['compression_m']>.0001:raise RuntimeError('Release not verified')
    outcome.update(success=True,stable_s=stable)
    print('PROBE_COMPLETE',metrics(s),'stable_s',stable,flush=True)
except Exception as e:
    print('PROBE_REJECTED',repr(e),flush=True)
    outcome['reason']=str(e)
finally:
    (env.out/'outcome.json').write_text(json.dumps(outcome,indent=2))
    env.save('probe');app.close()
if not outcome['success']:raise SystemExit(1)
