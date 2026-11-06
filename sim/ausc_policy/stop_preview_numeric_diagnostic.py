"""Replay a failed expert capture on CPU; raw archives stay on Linux.

Future held-waypoint commands and constant tail are numerical hypotheses only.
They do not establish future observed geometry, contact or measured settling.
"""
import argparse
from copy import deepcopy
import gzip
import json
import os
from pathlib import Path
import sys

import numpy as np

from . import m2_given_contact_handoff as handoff
from . import m2_given_contact_runtime as runtime
from sim.auscultation_pi05_v6.execution import JointTrajectory,FinishedStationaryRoundoffError,FINISHED_ROUNDOFF_ULPS
from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
from sim.auscultation_skill_v1.executor import command_derivatives


def replay(episode,output,*,requested_jerk=None):
    runtime.need(sys.platform=='linux' and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Linux CPU-only diagnostic required')
    import ruckig
    runtime.need(ruckig.__version__=='0.15.3' and FINISHED_ROUNDOFF_ULPS==64,'Original native solver/bound required')
    episode=Path(episode).resolve();output=Path(output).resolve();runtime.need(not output.exists(),'Fresh diagnostic output required')
    report_ref=runtime.reference(episode/'fresh_expert_result.json')
    audit_ref=runtime.reference(episode/'compact_multimodal/native_audit.jsonl.gz')
    report=runtime.read_bound(report_ref)
    runtime.need(report.get('success') is False and 'Finished stationary output exceeds numerical roundoff bound' in
        str(report.get('failure_reason')),'Actual retained numerical failure required')
    with gzip.open(runtime.bound_path(audit_ref),'rt') as stream:audit=[json.loads(line) for line in stream]
    setters=report['raw_target_setters'];first=setters[0]['physics_row_start'];by_row={s['physics_row_start']:s for s in setters}
    runtime.need(report['physics_steps']==len(audit) and len(by_row)==len(setters),'Exact actual audit/setter count required')
    limits=dict(report['raw_target_interpolation_limits']);counterfactual=requested_jerk is not None
    if counterfactual:
        runtime.need(requested_jerk==handoff.PLANNING_LIMITS['jerk']<limits['jerk'],'Explicit tighter current requested jerk required')
        limits['jerk']=requested_jerk
    runtime.need(limits==handoff.PLANNING_LIMITS,'Actual replay requires its original planning-source version; use explicit tighter hypothesis separately')
    trajectory=JointTrajectory(audit[0]['q_command'],**limits)
    commands=[];prefix_history=np.asarray([r['q_command'] for r in audit[first-3:first]],dtype=float)
    peaks={key:0. for key in ('velocity','acceleration','jerk')};maximum_delta=0.
    for i in range(first,len(audit)):
        if i in by_row:trajectory.set_target(by_row[i]['target_q'])
        q=trajectory.step()
        if not counterfactual:runtime.need(np.array_equal(q,audit[i]['q_command']),'CPU replay differs from issued native command')
        maximum_delta=max(maximum_delta,float(np.max(abs(q-audit[i]['q_command']))))
        values=command_derivatives(prefix_history,q)
        runtime.need(values['velocity']<=.175 and values['acceleration']<=.5+1e-8 and values['jerk']<=1.5+1e-8,
            'Original scheduled command derivative gate failed')
        for key in peaks:peaks[key]=max(peaks[key],values[key])
        commands.append(q.copy());prefix_history=np.vstack([prefix_history[1:],q])
    history=prefix_history;kin=OpenArmKinematics();continuation=[]
    def verify(q):
        runtime.need(np.all(q>=kin.lower_limits) and np.all(q<=kin.upper_limits),'Original strict joint bound failed')
    def live():return dict(input=handoff.input_snapshot(trajectory),output=handoff.output_snapshot(trajectory),
        history=history.tolist(),object_ids=[id(v) for v in (trajectory,trajectory.input,trajectory.output,trajectory.generator)])
    attempts=[];proposal=None;start_input=handoff.input_snapshot(trajectory)
    for extra in range(0,handoff.MAX_PARK_STEPS-handoff.SETTLE_INTERVALS):
        if extra%12==0:
            before=deepcopy(live());entry=dict(physics_row=len(audit)+extra,extra_live_ticks=extra,
                input_state=before['input'],frozen_command_history=history.tolist(),passed=False,stop_setter_issued=False)
            attempts.append(entry)
            try:
                proposal=handoff.preview_stop(trajectory,history,verify,live)
                runtime.need(extra+len(proposal.commands)+24<=1200,'Numerical hypothesis exceeds single parking budget')
                entry.update(passed=True,stop_steps=len(proposal.commands),preview=proposal.report)
                break
            except FinishedStationaryRoundoffError as error:
                entry.update(error=str(error),details=deepcopy(error.details))
            runtime.need(live()==before,'Rejected clone mutated live interpolator/history')
        # The current recorded waypoint is unchanged; every 20Hz refresh sets
        # that exact same target. This is not a future actual observation.
        if extra%12==0:trajectory.set_target(setters[-1]['target_q'])
        q=trajectory.step();verify(q);values=command_derivatives(history,q)
        runtime.need(values['velocity']<=.175 and values['acceleration']<=.5+1e-8 and values['jerk']<=1.5+1e-8,
            'Original continuation command derivative gate failed')
        for key in peaks:peaks[key]=max(peaks[key],values[key])
        continuation.append(q.copy());history=np.vstack([history[1:],q])
    runtime.need(proposal is not None and attempts[-1]['passed'],'No legal bounded numerical preview candidate')
    stop_before=handoff.input_snapshot(trajectory)
    handoff.apply_stop_profile(trajectory,expected_input=stop_before);trajectory.set_target(history[-1]);issued=[]
    for expected in proposal.commands:
        q=trajectory.step();runtime.need(np.array_equal(q,expected),'Numerical preview/live STOP hypothesis differs')
        verify(q);issued.append(q.copy());history=np.vstack([history[1:],q])
    runtime.need(np.array_equal(history,np.tile(history[-1],(3,1))),'Numerical stationary history missing')
    tail=handoff.constant_tail_snapshot(trajectory,history[-1],len(continuation)+len(issued))
    for _ in range(24):runtime.need(handoff.constant_tail_snapshot(trajectory,history[-1],tail['start_physics_row'])==tail,
        'Constant numerical tail changed interpolator state')
    runtime.bound_path(report_ref);runtime.bound_path(audit_ref)
    output.mkdir(parents=True)
    path=output/'numerical_commands.npz'
    with path.open('xb') as stream:np.savez(stream,**{
        ('counterfactual_schedule' if counterfactual else 'actual_prefix_replay'):np.asarray(commands),
        'hypothetical_l2_continuation':np.asarray(continuation),'hypothetical_stop':np.asarray(issued)})
    return runtime.write_new(output/'COMPLETE.json',dict(schema='m2_actual_stop_numeric_diagnostic_v1',complete=True,
        source_report=report_ref,source_audit=audit_ref,source_helper=runtime.reference(Path(__file__)),
        actual_prefix_rows=len(audit),actual_interpolator_rows=len(commands),all_issued_commands_bit_identical=not counterfactual,
        counterfactual_profile=counterfactual,source_requested_limits=report['raw_target_interpolation_limits'],
        requested_limits=limits,max_abs_archive_command_delta_rad=maximum_delta,command_derivative_maxima=peaks,
        actual_initial_stop_input=start_input,attempts=attempts,numerical_tail=tail,arrays=runtime.reference(path),
        hypothetical_wait_ticks=len(continuation),hypothetical_stop_ticks=len(issued),
        hypothetical_total_parking_ticks=len(continuation)+len(issued)+24,roundoff_ulps=64,
        physics_executed=False,gpu_launched=False,geometry_future_verified=False,measured_settling_verified=False,
        physical_qualification=False,training_eligible=False,original_failed_episode_verdict_unchanged=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--episode',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--requested-jerk',type=float)
    args=parser.parse_args();print(json.dumps(replay(args.episode,args.output,requested_jerk=args.requested_jerk)))


if __name__=='__main__':main()
