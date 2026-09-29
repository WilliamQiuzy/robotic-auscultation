"""Compare completed, mirrored M0 D0 trials; preserve every original verdict.

This is descriptive evidence from one approved body, not a capture-region or
generalization estimate. No training, rendering, simulator or robot imports.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(root, run_id):
    if Path(run_id).name != run_id:
        raise ValueError('Invalid run ID')
    root=root.resolve(); manifest=read(root/'MIRROR_SHA256.json')
    for name, binding in manifest.items():
        path=root/name
        if root not in path.resolve().parents or path.stat().st_size!=binding['bytes'] or digest(path)!=binding['sha256']:
            raise ValueError('Evidence binding changed: '+name)
    def bound(relative):
        name=run_id+'/'+relative
        if name not in manifest:
            raise ValueError('Required evidence is unbound: '+name)
        return root/name
    owner=read(bound('result.json'));case='output/case/'
    result=read(bound(case+'result.json'));outcome=read(bound(case+'outcome.json'))
    if owner.get('completed') is not True or owner.get('gpu_released') is not True or result.get('execution_completed') is not True:
        raise ValueError('Only actually completed/released trials can be compared')
    trace=[json.loads(line) for line in bound(case+'expert_trace.jsonl').read_text().splitlines()]
    commands=read(bound(case+'control_commands.json'))
    times=np.asarray([r['time_s'] for r in trace])
    if not len(trace) or not np.allclose(np.diff(times),1/240,rtol=0,atol=1e-8):
        raise ValueError('Physics history is missing or discontinuous')
    cursor=0;zero=[]
    for i,row in enumerate(commands):
        start,end=row['physics_row'],row['physics_row_end_exclusive']
        if start!=cursor or end<start or end-start!=row['executed_physics_steps']:
            raise ValueError('Command/physics coverage mismatch')
        if start==end:zero.append(i)
        cursor=end
    if cursor!=len(trace):raise ValueError('Uncovered physics rows')
    errors=[];factors=Counter();rejected=0
    for action in outcome['l2_actions']:
        preview=action.get('posture_hint_preview')
        if preview:
            if preview.get('accepted'):factors[str(preview['selected_factor'])]+=1
            else:rejected+=1
        if not action['completed']:continue
        start,end=action['physics_row_start'],action['physics_row_end_exclusive']
        if end-start!=12 or abs(times[end-1]-action['label_time_s'])>1e-8:
            raise ValueError('Completed action clock/coverage mismatch')
        errors.append(np.asarray(trace[end-1]['q_command'])-action['proposal'])
    dynamics=read(bound(case+'measured_dynamics.json'))
    last=trace[-1];queries=outcome['l2_queries'];handoffs=outcome['d0_handoffs']
    summary=dict(run_id=run_id,source_commit=owner['code_commit'],case_passed=result['case_passed'],
        execution_completed=True,m0_passed=False,approved_bodies=1,target='R4',
        checkpoint_sha256=outcome['checkpoint_receipt']['checkpoint_sha256'],
        mirror_manifest_sha256=digest(root/'MIRROR_SHA256.json'),
        source_sha256={name:digest(bound(case+name)) for name in
            ('outcome.json','expert_trace.jsonl','control_commands.json','result.json')},
        physics_steps=len(trace),simulation_end_s=float(times[-1]),policy_queries=len(queries),
        completed_actions=len(errors),attempted_actions=len(outcome['l2_actions']),
        zero_execution_command_indices=zero,original_audit_verdicts_preserved=True,
        motion_velocity=outcome.get('motion_velocity','zero'),posture_hint_preview=outcome.get('posture_hint_preview',False),
        preview_selected_factors=dict(factors),preview_all_candidates_rejected=rejected,
        l3_entries=sum(bool(h.get('stationary_entry')) for h in handoffs),
        l3_completions=sum(bool(h.get('l3_completed')) for h in handoffs),
        final=dict(gap_mm=1000*last['gap_m'],tangent_mm=1000*last['tangent_m'],
            tool_axis_deg=last['angle_deg'],hall_N=last['hall_N'],hold_elapsed_s=last['hold_elapsed_s']),
        max_hall_N=max(r['hall_N'] for r in trace),max_hold_elapsed_s=max(r['hold_elapsed_s'] for r in trace),
        observed_centered_jerk_rad_s3=dynamics['centered']['jerk_rad_s3']['max'],
        command_target_mean_abs_rad=float(np.abs(errors).mean()) if errors else None,
        command_target_error_scope='Shaped command minus ACT reference at 50ms; not prediction error',
        handoffs=handoffs,failure_reason=outcome['failure_reason'])
    return summary,trace,outcome


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True);parser.add_argument('--baseline-run',required=True)
    parser.add_argument('--candidate',type=Path,required=True);parser.add_argument('--candidate-run',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    values=[load(args.baseline,args.baseline_run),load(args.candidate,args.candidate_run)]
    if values[0][0]['checkpoint_sha256']!=values[1][0]['checkpoint_sha256']:
        raise ValueError('This comparison requires identical ACT weights')
    report=dict(scope='Descriptive single-body R4 executor comparison; not a success-rate estimate',
                m0_passed=False,rows=[v[0] for v in values])
    (args.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True,layout='constrained')
    for (summary,trace,outcome),label,color in zip(values,
            ('Original: zero terminal velocity','Quadratic velocity + posture preview'),('#97543d','#1b7784')):
        first=outcome['l2_queries'][0]['time_s'] if outcome['l2_queries'] else float('inf')
        rows=[r for i,r in enumerate(trace) if r['time_s']>=first and i%12==0]
        for ax,key,scale in zip(axes,('gap_m','tangent_m','angle_deg'),(1000,1000,1)):
            ax.plot([r['time_s'] for r in rows],[r[key]*scale for r in rows],label=label,color=color,lw=1.5)
    for ax,label,threshold in zip(axes,('Normal gap (mm)','Tangential error (mm)','Tool-axis error (deg)'),(25,3,5)):
        ax.axhline(threshold,color='#777',ls='--',lw=1);ax.set_ylabel(label);ax.grid(alpha=.15)
    axes[0].axhspan(23,27,color='#888',alpha=.12);axes[0].legend()
    axes[-1].set_xlabel('Simulator time (s); inference pauses excluded')
    fig.suptitle('Same ACT checkpoint, same R4: actual execution comparison\nDashed lines: original handoff gates; each run stops at its own terminal state',fontsize=12)
    fig.savefig(args.output/'comparison.svg');fig.savefig(args.output/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(report,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
