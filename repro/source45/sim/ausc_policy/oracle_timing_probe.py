"""Compare timing on the same planned route from one actual stopped query."""
import argparse
import os
from pathlib import Path
import sys
import time
import numpy as np
from . import act_baseline as a
from .isaac_case import prepare
from .recovery_oracle import StationaryRecoveryOracle


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();c=a.read_json(args.config)
    a.need(sys.platform=='linux' and os.environ.get('AUSC_OWNED_RUN')=='1' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
        and c['experiment']=='oracle_timing_probe' and c['strategies']==['global_cubic','checked_edges_v1'],'Owned paired CPU timing probe required')
    root=Path(c['source_run']);owner=a.pinned_json(root/'result.json',c['source_owner_sha256'])
    case=root/'output/history20';query=a.pinned_json(case/'oracle_query.json',c['query_sha256'])
    outcome=a.pinned_json(case/'outcome.json',c['outcome_sha256'])
    a.need(owner['completed'] and owner['gpu_released'] and owner['stage']['process_group_drained']
        and not outcome['success'] and outcome['expert_demonstration'] and query['kind']=='abstain'
        and 'search' in query and query['stationarity']['passed'] and outcome['oracle_receipt']['query_sha256']==c['query_sha256'],
        'Actual stopped query with rejected timed recovery required')
    trace=case/'expert_trace.jsonl';a.need(a.sha256(trace)==c['trace_sha256'],'Acquired physics changed')
    import json
    physics=[json.loads(x) for x in trace.open()];state=query['measured_query']['state'];context=query['measured_query']['context']
    for r in state['recent_measurements']:
        old=physics[r['physics_row']]
        a.need(r['time_s']==old['time_s'] and all(np.array_equal(r[k],old[k]) for k in ('q','q_command')),'Actual stop history differs')
    args.output.mkdir(parents=True,exist_ok=False);bundle,_,_=prepare(c,args.output/'geometry_source')
    from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
    from sim.auscultation_expert_layout_v1.posture import PostureChecker,PostureLimits
    from sim.auscultation_expert_layout_v1.plan_low_elbow import checked_transfer
    from sim.auscultation_skill_v1.certified_collision import CertifiedCollisionChecker
    from sim.curobo_expert_v1.static_world import load_static_world,install_fcl_static_world
    kin=OpenArmKinematics();base=np.asarray(bundle.plan['world_from_left_base']);folder=args.output/'geometry_source'
    checker=CertifiedCollisionChecker(kin,base,directory=folder/'geometry')
    install_fcl_static_world(checker,load_static_world(folder/'static_world'),geometry_dir=folder/'geometry')
    limits=PostureLimits.from_dict(bundle.row['posture_limits'],kin);checker=PostureChecker(checker,base,limits)
    # Search once. Both timings receive the identical route, not two different
    # planner samples or a path selected because one timing looked better.
    route,search=checked_transfer(checker,limits,np.asarray(state['q']),np.asarray(context['q_goal']),kin,base,
        np.random.default_rng(65),time.monotonic()+30.)
    a.need(route is not None,'Same-state route search failed')
    route=np.asarray(route);a.save_json(args.output/'shared_route.json',dict(waypoints=route.tolist(),search=search))
    records=[]
    for strategy in c['strategies']:
        oracle=StationaryRecoveryOracle(checker,lambda *unused:(route.copy(),search),lower=limits.joint_lower_rad,
            upper=limits.joint_upper_rad,timing_strategy=strategy)
        report=oracle.query(state,context);a.save_json(args.output/(strategy+'.json'),report)
        records.append(dict(strategy=strategy,kind=report['kind'],reason=report.get('reason'),
            checked_samples=report.get('checked_samples'),timing=report.get('timing'),
            rejected_command_index=report.get('checking_command_index'),wall_time_s=report['wall_time_s']))
        print(dict(strategy=strategy,kind=report['kind'],reason=report.get('reason')),flush=True)
    a.save_json(args.output/'COMPLETE.json',dict(completed=True,cases=records,same_route=True,
        shared_route_sha256=a.sha256(args.output/'shared_route.json'),actual_physics_executed=False,
        training_dataset_eligible=False,m0_passed=False,model_loaded=False))


if __name__=='__main__':main()
