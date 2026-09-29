"""Reproduce the saved first-command posture failure with native CPU Ruckig."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from . import act_baseline as a
from .executor_probe import load_native, LIMITS, FLAGS
from .motion_hints import velocity_hints
from .motion_preview import choose_velocity_hint
from sim.auscultation_pi05_v6.execution import JointTrajectory


def run(config, output):
    a.need(config['experiment']=='motion_preview_probe' and config['device']=='cpu'
           and config['time_limit_seconds']==600,'Only the declared CPU preview probe is supported')
    native=load_native(config)
    source=Path(config['source_run']);owner=a.read_json(source/'result.json')
    a.need(owner['completed'] and owner['gpu_released'],'Source run must have actually terminated')
    case=source/'output/case'
    outcome=a.pinned_json(case/'outcome.json',config['source_outcome_sha256'])
    trace_path=case/'expert_trace.jsonl'
    a.need(a.sha256(trace_path)==config['source_trace_sha256'],'Source trace changed')
    trace=[json.loads(line) for line in trace_path.read_text().splitlines()]
    a.need(outcome['motion_velocity']=='quadratic5' and len(outcome['l2_queries'])==1
           and len(outcome['l2_actions'])==1 and not outcome['l2_actions'][0]['completed']
           and outcome['l2_actions'][0]['physics_row_start']==len(trace)
           and outcome['l2_actions'][0]['physics_row_end_exclusive']==len(trace)
           and 'posture_joint_envelope' in outcome['failure_reason'],'Unexpected source diagnostic')
    snapshot=a.read_json(case/'input_snapshot/manifest.json')
    plan_binding=snapshot['files']['R4/plan']
    plan=a.pinned_json(plan_binding['source_path'],plan_binding['sha256'])
    from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
    from sim.auscultation_expert_layout_v1.posture import PostureLimits
    kin=OpenArmKinematics()
    a.need(a.sha256(kin.urdf_path)==snapshot['files']['R4/kinematics_urdf']['sha256'],'Source URDF changed')
    posture=PostureLimits.from_dict(outcome['posture_limits'],kin)
    base=np.asarray(plan['world_from_left_base'])
    initial=np.asarray(trace[-1]['q_command'],float)
    # D0 has just initialized a new command trajectory at this retained q.
    a.need(np.array_equal(initial,np.zeros(7)), 'Saved initial-home trajectory contract changed')
    trajectory=JointTrajectory(initial,**LIMITS)
    query=outcome['l2_queries'][0];chunk=np.asarray(query['raw_absolute_chunk'])
    hint=velocity_hints(chunk)
    np.testing.assert_array_equal(hint['velocity'],query['velocity_hint']['velocity'])
    def inspect(command):
        if np.any(command<kin.lower_limits) or np.any(command>kin.upper_limits):
            return dict(clear=False,reason='exact command URDF bounds')
        return posture.evaluate(kin,base,command,observed=False,contact=False)
    before={key:list(getattr(trajectory.input,key)) for key in
            ('current_position','current_velocity','current_acceleration','target_position','target_velocity')}
    velocity,receipt=choose_velocity_hint(trajectory,chunk[0],hint['velocity'][0],inspect)
    after={key:list(getattr(trajectory.input,key)) for key in before}
    a.need(before==after,'Preview mutated the active command state')
    a.need(receipt['selected_factor']<1. and receipt['trials'][0]['first_rejection'] is not None,
           'Original full-hint failure was not reproduced')
    trajectory.set_target(chunk[0]);trajectory.input.target_velocity=velocity.tolist()
    actual=np.asarray([trajectory.step() for _ in range(12)])
    expected=np.asarray(receipt['trials'][-1]['preview_commands'])
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)
    a.need(all(inspect(q)['clear'] for q in actual),'Selected CPU command crosses the original posture bounds')
    report=dict(completed=True,preview_passed=True,**FLAGS,
        scope='Native CPU command-generation reproduction only; no model, PhysX, collision or contact test',
        source_run=str(source),source_outcome_sha256=config['source_outcome_sha256'],
        source_trace_sha256=config['source_trace_sha256'],source_plan_sha256=plan_binding['sha256'],
        motion_preview_source_sha256=a.sha256(Path(__file__).with_name('motion_preview.py')),
        urdf_sha256=a.sha256(kin.urdf_path),native_ruckig_sha256=a.sha256(native.__file__),
        limits=LIMITS,factors=[1.,.5,.25,.125,0.],ticks=12,
        original_failure_reproduced=True,active_state_unchanged=True,
        preview_vs_generation_max_abs_rad=float(np.max(np.abs(actual-expected))),
        initial_command_q=initial.tolist(),raw_position=chunk[0].tolist(),
        estimated_velocity=hint['velocity'][0].tolist(),selected_velocity=velocity.tolist(),preview=receipt)
    a.save_json(output/'metrics.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if sys.platform!='linux' or os.environ.get('AUSC_OWNED_RUN')!='1':
        raise RuntimeError('Existing guarded Linux CPU owner required')
    config=a.read_json(args.config);args.output.mkdir(parents=True,exist_ok=False)
    a.save_json(args.output/'configuration.json',config)
    try:
        report=run(config,args.output)
        a.save_json(args.output/'COMPLETE.json',dict(completed=True,preview_passed=report['preview_passed'],
            **FLAGS,metrics_sha256=a.sha256(args.output/'metrics.json')))
    except Exception as error:
        a.save_json(args.output/'FAILED.json',dict(completed=False,**FLAGS,error=type(error).__name__,reason=str(error),
            preview=getattr(error,'receipt',None)))
        raise
    print('NATIVE_POSTURE_PREVIEW_COMPLETE',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
