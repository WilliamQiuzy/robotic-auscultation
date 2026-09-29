"""Native CPU comparison on one pinned saved ACT reference/clock sequence.

The zero-hint branch must reproduce every actually executed old command.
Counterfactual branches share the old references; there is no new observation,
policy inference, collision validation, PhysX or contact success here.
"""
from collections import Counter
from pathlib import Path
import json
import time
import numpy as np

from . import act_baseline as a
from .motion_hints import velocity_hints
from .motion_preview import LIMITS,choose_velocity_hint
from sim.auscultation_pi05_v6.execution import JointTrajectory


def compare(config,inspect,kin,output):
    source=Path(config['reference_run']);owner=a.read_json(source/'result.json')
    a.need(owner['completed'] is True and owner['gpu_released'] is True,'Reference owner still active')
    case=source/'output/case'
    outcome=a.pinned_json(case/'outcome.json',config['reference_outcome_sha256'])
    a.need(outcome['motion_velocity']=='zero' and len(outcome['d0_handoffs'])==1
           and not outcome['d0_handoffs'][0]['candidate'],'Require original zero-hint no-stop D0')
    trace_path=case/'expert_trace.jsonl';a.need(a.sha256(trace_path)==config['reference_trace_sha256'],'Trace binding differs')
    trace=[json.loads(line) for line in trace_path.read_text().splitlines()]
    start=outcome['d0_handoffs'][0]['l2_start_physics_row'];initial=np.asarray(trace[start-1]['q_command'])
    a.need(np.array_equal(initial,np.zeros(7)),'Unexpected command initialization')
    schedule=[];cursor=start
    for action in outcome['l2_actions']:
        first,last=action['physics_row_start'],action['physics_row_end_exclusive']
        a.need(first==cursor and 0<=last-first<=12,'Noncontiguous source action clock')
        cursor=last
        if last==first:continue  # The original refused command is never turned into an executed tick.
        query=outcome['l2_queries'][action['query_index']];chunk=np.asarray(query['raw_absolute_chunk'])
        np.testing.assert_array_equal(chunk[action['prefix_index']],action['proposal'])
        schedule.append(dict(target=np.asarray(action['proposal']),ticks=last-first,
            observed_commands=np.asarray([r['q_command'] for r in trace[first:last]]),
            velocity=velocity_hints(chunk)['velocity'][action['prefix_index']],completed=action['completed']))
    a.need(cursor==len(trace),'Source physics rows not fully covered')
    deadline=time.monotonic()+500.;branches={};arrays={}
    for mode in ('zero','first_feasible','nearest_reference'):
        trajectory=JointTrajectory(initial,**LIMITS);rows=[];counts=Counter();matched=0.;tick_count=0;fail=None;comparisons=[]
        try:
            for item in schedule:
                a.need(time.monotonic()<deadline,'Finite reference replay budget expired')
                if mode=='zero':terminal=np.zeros(7)
                else:
                    terminal,receipt=choose_velocity_hint(trajectory,item['target'],item['velocity'],inspect,selection=mode)
                    counts[str(receipt['selected_factor'])]+=1
                    if mode=='nearest_reference':
                        zero=receipt['zero_hint_endpoint_joint_mse_rad2'];best=receipt['selected_endpoint_joint_mse_rad2']
                        if zero is not None:
                            a.need(best<=zero,'Selected endpoint is worse than feasible zero-hint candidate')
                            comparisons.append(dict(selected_mse_rad2=best,zero_mse_rad2=zero))
                trajectory.set_target(item['target']);trajectory.input.target_velocity=terminal.tolist();trajectory.input.target_acceleration=[0.]*7
                for k in range(item['ticks']):
                    command=trajectory.step();tick_count+=1
                    a.need(inspect(command)['clear'],'Generated command violates original posture bounds')
                    if mode=='zero':matched=max(matched,float(np.max(np.abs(command-item['observed_commands'][k]))))
                if item['completed']:rows.append((command.copy(),item['target'].copy()))
            if mode=='zero':a.need(matched<=1e-12,'Native zero-hint branch does not reconstruct actual command history')
        except Exception as error:
            fail=dict(error=type(error).__name__,reason=str(error),preview=getattr(error,'receipt',None))
        a.need(rows,'No completed native reference intervals')
        command,reference=(np.stack(v) for v in zip(*rows))
        position=np.stack([kin.tcp(q)[:3,3] for q in command]);goal=np.stack([kin.tcp(q)[:3,3] for q in reference])
        branches[mode]=dict(completed=fail is None,failure=fail,ticks=tick_count,completed_intervals=len(rows),
            original_command_max_abs_difference_rad=matched if mode=='zero' else None,
            joint_mean_abs_error_rad=float(np.abs(command-reference).mean()),
            tcp_mean_distance_mm=float(np.linalg.norm(position-goal,axis=1).mean()*1000),
            factor_counts=dict(counts),local_zero_comparisons=len(comparisons),
            local_zero_comparisons_all_not_worse=all(x['selected_mse_rad2']<=x['zero_mse_rad2'] for x in comparisons)
                if mode=='nearest_reference' else None)
        arrays.update({mode+'_command':command,mode+'_reference':reference})
    np.savez_compressed(output/'reference_endpoints.npz',**arrays)
    return dict(source_run=str(source),source_outcome_sha256=config['reference_outcome_sha256'],
        source_trace_sha256=config['reference_trace_sha256'],branches=branches,
        fixed_actual_action_intervals=len(schedule),fixed_actual_ticks=sum(x['ticks'] for x in schedule),
        all_branches_completed=all(v['completed'] for v in branches.values()),
        endpoint_arrays_sha256=a.sha256(output/'reference_endpoints.npz'),
        scope='Native mathematical commands only, identical old ACT references and actual executed clock; no new closed loop or collision proof')
