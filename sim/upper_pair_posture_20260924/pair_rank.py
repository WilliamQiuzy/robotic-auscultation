"""Bounded nominal R1-L1-R2 route comparison; physical release is checked later."""
import argparse
import itertools
import json
from pathlib import Path
import time

import numpy as np
from .contract import PLANS, RUNTIME, ROOT, PROJECT, read, sha, require, verify_contract
from .search import joint_metrics, motion_rank


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--search',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-pairs',type=int,default=8)
    parser.add_argument('--max-seconds',type=float,default=300.)
    args=parser.parse_args()
    require(not args.output.exists(),'pair report must be new')
    contract=read(ROOT/'contract.json');verify_contract(contract)
    summary=read(args.search);require(summary.get('completed') is True,'search must be complete')
    sources={str(args.search.resolve().relative_to(PROJECT)):sha(args.search)}
    options={n:[] for n in ('R1','L1')}
    for record in summary['candidates']:
        p=(args.search.parent/record['plan_path']).resolve()
        require(sha(p)==record['plan_sha256'],'candidate bytes changed')
        sources[str(p.relative_to(PROJECT))]=sha(p)
        options[record['name']].append({**record,'path':str(p.relative_to(PROJECT)),'row':read(p)['points'][0]})
    for name in options:
        p=PLANS/f'planned_paths_{name}.json';row=read(p)['points'][0]
        options[name].append({'id':'baseline_'+name,'name':name,'path':str(p.relative_to(PROJECT)),
                              'row':row,'relative_roll_deg':0.,'qpre':row['approach_q'][0]})
    r2=read(PLANS/'planned_paths_R2.json')['points'][0]
    pairs=[]
    for a,b in itertools.product(options['R1'],options['L1']):
        heuristic=joint_metrics(np.vstack([a['row']['transfer_q'],b['qpre'],r2['approach_q'][0]]))
        pairs.append((motion_rank(heuristic),a,b))
    pairs.sort(key=lambda item:item[0])
    # Always include the unchanged pair for a geometrically matched reference.
    baseline=next(x for x in pairs if x[1]['id']=='baseline_R1' and x[2]['id']=='baseline_L1')
    selected=[baseline]+[x for x in pairs if x is not baseline][:args.max_pairs]
    from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
    from sim.auscultation_expert_layout_v1.posture import PostureLimits
    from sim.auscultation_expert_layout_v1.replan_layout import SafetyChecker
    from sim.auscultation_expert_layout_v1.plan_low_elbow import checked_transfer
    from sim.expert_quality_20260924.candidate_planner import checked_shortcut
    from sim.curobo_expert_v1.fast_collision import FastCollisionChecker
    kin=OpenArmKinematics(RUNTIME/'openarm_v10_follower_stetho.urdf')
    base=np.asarray(read(PLANS/'planned_paths_R1.json')['world_from_left_base'])
    limits=PostureLimits.from_dict(read(PLANS/'posture_limits.json'),kin)
    checker=SafetyChecker(FastCollisionChecker(kin,base,directory=PLANS),base,limits,margin_deg=3.)
    started=time.monotonic();deadline=started+args.max_seconds;records=[]
    for _,a,b in selected:
        if time.monotonic()>=deadline:break
        result={'r1':a['id'],'l1':b['id'],'r1_plan':a['path'],'l1_plan':b['path'],
                'relative_roll_deg':[a['relative_roll_deg'],b['relative_roll_deg']],
                'baseline':a['id']=='baseline_R1' and b['id']=='baseline_L1','feasible':False,'transfers':[]}
        records.append(result)
        try:
            route=list(np.asarray(a['row']['transfer_q']))
            for origin,destination in ((a['row'],b['row']),(b['row'],r2)):
                # Include nominal approach, loaded corridor and retracing. Dynamics are not inferred.
                approach=np.asarray(origin['approach_q'])
                corridor=np.asarray(origin['contact_corridor_posture_q'])
                route.extend(approach[1:]);route.extend(corridor[1:]);route.extend(corridor[-2::-1]);route.extend(approach[-2::-1])
                canonical=np.asarray(destination['transfer_q'])
                stage=np.asarray(destination['lower_staging_q'])
                indices=np.flatnonzero(np.all(np.isclose(canonical,stage,atol=1e-9,rtol=0),axis=1))
                require(len(indices)==1,'ambiguous lower stage')
                bridge,search=checked_transfer(checker,limits,approach[0],stage,kin,base,
                    np.random.default_rng(265),min(deadline,time.monotonic()+30.))
                require(bridge is not None,'nominal inter-point bridge unavailable')
                staged=np.vstack([bridge,canonical[indices[0]+1:]])
                transfer,evidence=checked_shortcut(staged,checker,min(deadline,time.monotonic()+30.))
                route.extend(transfer[1:])
                result['transfers'].append({'from':origin['name'],'to':destination['name'],
                    'joint_waypoints_q':transfer.tolist(),'bridge_search':search,'check':evidence})
            result.update(feasible=True,metrics=joint_metrics(route))
        except (ValueError,TimeoutError) as error:result['failure']=str(error)
        print(json.dumps({k:result[k] for k in ('r1','l1','feasible')}),flush=True)
    for rel,digest in sources.items():require(sha(PROJECT/rel)==digest,'search source changed')
    verify_contract(contract)
    alternatives=sorted([r for r in records if r['feasible'] and not r['baseline']],key=lambda r:motion_rank(r['metrics']))
    result={'schema':'upper_pair_nominal_comparison_v1','source_sha256':sources,'records':records,
        'ranked_alternatives':[{'r1':r['r1'],'l1':r['l1'],'r1_plan':r['r1_plan'],'l1_plan':r['l1_plan'],
                              'metrics':r['metrics']} for r in alternatives],
        'elapsed_seconds':time.monotonic()-started,'max_seconds':args.max_seconds,'max_pairs':args.max_pairs,
        'scope':'Nominal geometric paths through R2 prepose. Actual loaded posture, release, measured inter-point origin and dynamics need physics. No impossibility or safety certification.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
