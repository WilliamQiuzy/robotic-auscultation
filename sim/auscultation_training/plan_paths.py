"""Pose IK plus independent whole-arm collision checked approach paths."""
import argparse,json,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
from .kinematics import OpenArmKinematics
from .targets import load_targets
from .collision import CollisionChecker
from .scene import ROOT

def densify(a,b,step=.02):
    return np.linspace(a,b,max(2,int(np.ceil(np.max(np.abs(b-a))/step))+1))

def rrt_path(checker,start,goal,rng,max_iterations=1500):
    if checker.edge(start,goal)['clear']:return [start,goal]
    # Single-tree RRT with goal bias; shortcut only through checked edges.
    nodes=[start.copy()];parents=[-1]
    for iteration in range(max_iterations):
        candidate=goal if iteration%5==0 else rng.uniform(checker.kinematics.lower_limits,checker.kinematics.upper_limits)
        i=int(np.argmin(np.linalg.norm(np.asarray(nodes)-candidate,axis=1)))
        delta=candidate-nodes[i]; candidate=nodes[i]+delta*min(1.,.22/max(np.linalg.norm(delta),1e-12))
        if not checker.edge(nodes[i],candidate)['clear']:continue
        nodes.append(candidate);parents.append(i)
        if np.linalg.norm(candidate-goal)<.4 and checker.edge(candidate,goal)['clear']:
            path=[goal];i=len(nodes)-1
            while i>=0:path.append(nodes[i]);i=parents[i]
            path=path[::-1]
            for _ in range(80):
                if len(path)<3:break
                a,b=sorted(rng.choice(len(path),size=2,replace=False))
                if b>a+1 and checker.edge(path[a],path[b])['clear']:path=path[:a+1]+path[b:]
            return path
    return None

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--point',default=None);parser.add_argument('--seeds',type=int,default=40)
    parser.add_argument('--targets',type=Path,default=None)
    args=parser.parse_args()
    base=np.asarray(json.loads((ROOT/'runtime_inspection.json').read_text())['world_from_left_base'])
    kin=OpenArmKinematics();checker=CollisionChecker(kin,base);targets=load_targets(base,path=args.targets)
    rng=np.random.default_rng(14);home=np.zeros(7)
    print('HOME',checker.check(home),flush=True)
    result={'world_from_left_base':base.tolist(),'points':[], 'targets_file':str(args.targets) if args.targets else None,
            'scope':'pose and geometric path checks; dynamics and contact not yet certified'}
    for target in targets:
        if args.point and target.name!=args.point:continue
        start=time.monotonic();candidates=[];failures=[]
        for i in range(args.seeds):
            seed=home if i==0 else rng.uniform(kin.lower_limits,kin.upper_limits)
            pre=target.position_base+target.normal_base*.06
            sol=kin.solve_ik(pre,target.approach_axis_base,seed=seed,max_starts=1,max_nfev=200)
            if not sol.success:continue
            test=checker.check(sol.q)
            if not test['clear']:
                failures.append(test);continue
            # A reachable prepose is insufficient: reject roll/posture branches
            # whose fixed frame or guide heads obstruct loaded contact.
            loaded=kin.solve_ik(target.position_base,target.approach_axis_base,
                                seed=sol.q,compression=.004,max_starts=1,max_nfev=120)
            if not loaded.success:continue
            test=checker.check(loaded.q,clearance=.001,allow_patient_contact=True,compression=.004)
            if not test['clear']:
                failures.append(test);continue
            # Cartesian normal approach, retaining the selected posture branch.
            q=sol.q;approach=[q];ok=True
            for gap in np.linspace(.06,.015,24)[1:]:
                ik=kin.solve_ik(target.position_base+target.normal_base*gap,target.approach_axis_base,seed=q,max_starts=1,max_nfev=80)
                if not ik.success or np.max(np.abs(ik.q-q))>.2:ok=False;break
                test=checker.edge(q,ik.q)
                if not test['clear']:ok=False;failures.append(test);break
                q=ik.q;approach.append(q)
            if ok:candidates.append((sol.q,approach))
            if len(candidates)>=3:break
        candidates.sort(key=lambda x:np.linalg.norm(x[0]-home))
        path=None;chosen=None
        for pre,approach in candidates:
            path=rrt_path(checker,home,pre,rng)
            if path is not None:chosen=approach;break
        row={'name':target.name,'success':path is not None,'candidates':len(candidates),'seconds':time.monotonic()-start}
        if path is not None:row.update(transfer_q=np.asarray(path).tolist(),approach_q=np.asarray(chosen).tolist(),position_world=target.position_world.tolist(),normal_world=target.normal_world.tolist())
        else:row['collision_examples']=failures[:5]
        result['points'].append(row)
        print('PLAN',target.name,row['success'],'candidates',len(candidates),'seconds',row['seconds'],flush=True)
        (ROOT/'results'/('planned_paths'+('_'+args.point if args.point else '')+'.json')).write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
