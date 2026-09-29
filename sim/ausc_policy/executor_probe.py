"""Finite CPU command-shaping probe; no model, PhysX, sensors or robot control.

Replay only pinned R4 TRAIN labels through the existing JointTrajectory and
ActionPrefix. Each proven continuous origin interval is an independent probe:
measured q at its first origin, zero initial command velocity/acceleration.
H10/E5 tails are omitted, never padded or joined across a preparation boundary.
The comparison is endpoint command/FK error, not observed motion or safety.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np

from . import act_baseline as a
from .prefix import ActionPrefix, PrefixContext, HORIZON, EXECUTION_PREFIX, PERIOD_S
from sim.auscultation_pi05_v6.execution import JointTrajectory

EPISODE = 'R4_observed_armature02_v1'
LIMITS = dict(dt=1/240, velocity=.175, acceleration=.5, jerk=7.99999)
TICKS = 12
MODES = ('zero_terminal_velocity', 'adjacent_target_velocity')
FLAGS = dict(m0_passed=False, closed_loop_evaluated=False, physics_executed=False,
             expert_demonstration=False, training_dataset_eligible=False,
             training_accepted=False, contact_evaluated=False, model_loaded=False)


def load_source(config):
    """Open only the named R4 cache; preparation already proved raw L2 labels."""
    a.need(config['episode_id'] == EPISODE, 'Only the declared R4 TRAIN episode is allowed')
    receipt = a.pinned_json(config['prepared_manifest'], config['prepared_manifest_sha256'])
    a.need(receipt.get('schema') == 'ausc_m0_act_preparation_v1'
           and receipt.get('action_representation') == a.ABSOLUTE
           and receipt.get('horizon') == HORIZON and receipt.get('execute_steps') == EXECUTION_PREFIX
           and receipt.get('period_s') == PERIOD_S and receipt.get('heldout_arrays_opened') is False
           and receipt.get('image_arrays_opened') is False, 'Prepared source contract differs')
    matches = [r for r in receipt['episodes'] if r.get('episode_id') == EPISODE]
    a.need(len(matches) == 1, 'Missing/duplicate R4 preparation')
    row = matches[0]; source = row['source_record']; boundary = row['boundary']
    a.need(row.get('split') == source.get('split') == 'train' and source.get('target') == 'R4'
           and source.get('episode_id') == EPISODE, 'R4 source is not the declared TRAIN episode')
    a.need(boundary.get('stationary_handoff_proved') is True
           and boundary.get('boundary_source') == 'raw align to contact_cycle, before all contact actions'
           and type(boundary.get('first_contact_physics_row')) is int
           and boundary['first_contact_physics_row'] > 0, 'Raw L2 cutoff is unproved')
    a.need(set(row['cache']) == set(a.MEMBERS), 'Unexpected prepared cache inventory')
    arrays = {}
    for name, binding in row['cache'].items():
        path = Path(binding['path'])
        a.need(path.is_absolute() and path.stat().st_size <= 16*1024**2
               and a.sha256(path) == binding['sha256'], 'Prepared cache changed/oversized: '+name)
        value = np.load(path, mmap_mode='r', allow_pickle=False)
        a.need(isinstance(value, np.memmap) and not value.flags.writeable, 'Read-only cache required')
        arrays[name] = value
    a.need(len(arrays['times']) == boundary['verified_label_pairs']
           and 0 < boundary['l2_label_pairs'] <= len(arrays['times']), 'Prepared boundary counts differ')
    provenance = dict(prepared_manifest=str(config['prepared_manifest']),
        prepared_manifest_sha256=config['prepared_manifest_sha256'], episode_id=EPISODE,
        cache=row['cache'], boundary=boundary, heldout_arrays_opened=False, image_arrays_opened=False,
        action_units='original absolute joint radians', label_clock='actions[i] is command at times[i]+0.05; no second shift')
    return arrays, row, provenance


def load_native(config):
    """Verify the existing native dependency before making its directory importable."""
    directories = config.get('dependency_paths',[]); files = config.get('dependency_files',{})
    a.need(isinstance(directories,list) and directories and isinstance(files,dict) and files,
           'Explicit pinned native dependency paths/files required')
    a.need(all(Path(d).is_absolute() and Path(d).is_dir() for d in directories),
           'Native dependency directories must be existing absolute paths')
    approved = {}
    for value,digest in files.items():
        path = Path(value)
        a.need(path.is_absolute() and a.sha256(path) == digest, 'Native dependency hash differs: '+value)
        approved[str(path.resolve())] = digest
    for directory in reversed(directories):
        sys.path.insert(0,str(Path(directory).resolve()))
    import ruckig
    a.need(ruckig.__version__ == '0.12.2', 'The actual pinned runtime Ruckig 0.12.2 is required')
    resolved = str(Path(ruckig.__file__).resolve())
    a.need(resolved in approved and a.sha256(resolved) == approved[resolved],
           'Imported Ruckig resolved outside the pinned native dependency')
    return ruckig


def make_segments(arrays, starts):
    """Consecutive qualified origins define independent, fully bounded probes."""
    times = arrays['times']; n = len(times)
    a.need(0 < n <= 10000 and times.shape == (n,) and times.dtype == np.float64
           and np.isfinite(times).all() and np.all(np.diff(times) > 0), 'Invalid bounded source clock')
    a.need(arrays['state'].shape == (n,41) and arrays['actions'].shape == (n,7)
           and all(arrays[k].shape == (n,) for k in ('phases','targets','target_sources'))
           and np.isfinite(arrays['state']).all() and np.isfinite(arrays['actions']).all(), 'Invalid source arrays')
    a.need(isinstance(starts,list) and starts and all(type(s) is int and 0 <= s <= n-HORIZON for s in starts)
           and starts == sorted(set(starts)), 'Invalid qualified H10 origins')
    runs = []
    for start in starts:
        stop = start+HORIZON
        phase = str(arrays['phases'][start])
        a.need(phase in ('FREE','APPROACH') and np.all(arrays['phases'][start:stop] == phase)
               and np.all(arrays['targets'][start:stop] == 'R4')
               and np.all(arrays['target_sources'][start:stop] == 'rgbd_surface_tracking')
               and np.allclose(np.diff(times[start:stop]), PERIOD_S, rtol=0, atol=1e-7),
               'Prepared H10 phase/target/source/clock boundary changed')
        if not runs or start != runs[-1][-1]+1:
            runs.append([])
        runs[-1].append(start)
    result = []
    for number, run in enumerate(runs):
        origins = run[::EXECUTION_PREFIX]
        indices = [i for origin in origins for i in range(origin, origin+EXECUTION_PREFIX)]
        result.append(dict(segment_id=number, source_start=run[0], source_stop_exclusive=run[-1]+HORIZON,
            start_time_s=float(times[run[0]]), phase=str(arrays['phases'][run[0]]),
            eligible_window_origins=len(run), query_origins=origins, executed_indices=indices,
            executed_actions=len(indices), duration_s=len(indices)*PERIOD_S,
            unused_tail_actions=run[-1]+HORIZON-(indices[-1]+1),
            unused_tail_valid_window_origins=run[-1]-origins[-1],
            skipped_overlapping_window_origins=len(run)-len(origins),
            initial_measured_q=np.asarray(arrays['state'][run[0],:7],float).tolist(),
            initial_recorded_dq=np.asarray(arrays['state'][run[0],7:14],float).tolist(),
            first_target_minus_initial_measured_q_rad=(arrays['actions'][run[0]]-arrays['state'][run[0],:7]).tolist()))
    return result


def derive_velocity(chunk):
    """Forward secants at the five executed target times, using H10 lookahead.

    v[k]=(a[k+1]-a[k])/50ms estimates outgoing velocity at target a[k]. It is
    not a measured derivative, nor an exact teacher derivative. Terminal a=0
    stays unchanged. No difference ever crosses a chunk/phase/L2 boundary.
    """
    chunk = np.asarray(chunk, float)
    a.need(chunk.shape == (HORIZON,7) and np.isfinite(chunk).all(), 'Full finite H10 required for velocity derivation')
    velocity = np.diff(chunk[:EXECUTION_PREFIX+1], axis=0)/PERIOD_S
    peak = float(np.max(np.abs(velocity)))
    a.need(np.isfinite(velocity).all() and peak <= LIMITS['velocity'],
           f'Derived terminal velocity max {peak:.9g} exceeds 0.175 rad/s; refused without clipping')
    return velocity


def _pack(rows):
    widths = dict(teacher_target_q=7, command_q=7, teacher_measured_q_at_label=7,
                  target_velocity=7, command_velocity=7, command_acceleration=7)
    names = ('segment_id','source_index','origin_index','prefix_index','label_time_s','elapsed_s',*widths)
    return {name:np.asarray([r[name] for r in rows], dtype=np.int64 if name.endswith('_id') or name.endswith('_index') else float)
            .reshape((-1,widths[name]) if name in widths else (-1,)) for name in names}


def replay(arrays, segments, mode, *, trajectory_factory=JointTrajectory, deadline=None):
    """No feedback substitution: pass_to_input preserves command q/v/a each tick."""
    a.need(mode in MODES, 'Unknown executor comparison')
    rows = []; failure = None; tick_count = 0
    peaks = dict(velocity_rad_s=0., acceleration_rad_s2=0., jerk_rad_s3=0.)
    sid = origin = index = None
    try:
        # Reject any infeasible FF before executing this branch, preserving a
        # transparent all-or-refused comparison against the zero-velocity run.
        velocities = {}
        if mode == MODES[1]:
            for segment in segments:
                sid = segment['segment_id']
                for origin in segment['query_origins']:
                    velocities[(sid,origin)] = derive_velocity(arrays['actions'][origin:origin+HORIZON])
        for segment in segments:
            sid = segment['segment_id']; origin = segment['source_start']
            trajectory = trajectory_factory(np.array(arrays['state'][origin,:7],float), **LIMITS)
            prefix = ActionPrefix()
            context = PrefixContext(EPISODE, 'R4', sid, segment['phase'], 'CPU_EXECUTOR_PROBE')
            previous_acceleration = np.zeros(7)
            for origin in segment['query_origins']:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError('Finite CPU probe deadline')
                stamp = float(arrays['times'][origin])
                chunk = np.array(arrays['actions'][origin:origin+HORIZON],float)
                installed = prefix.install(chunk, observed_at_s=stamp, now_s=stamp, context=context)
                a.need(installed.status == 'installed', 'Real prefix rejected chunk: '+installed.reason)
                for k in range(EXECUTION_PREFIX):
                    index = origin+k; now = float(arrays['times'][index])
                    proposal = prefix.pop(now_s=now,context=context,observation_time_s=now,
                        feedback_time_s=now,hall_time_s=now,target_visible=True,safety_ok=True)
                    a.need(proposal.status == 'action' and proposal.index == k
                           and abs(proposal.execute_at_s-(now+PERIOD_S)) < 1e-7,
                           'Prefix label clock/proposal differs')
                    trajectory.set_target(proposal.proposal)
                    terminal = velocities[(sid,origin)][k] if mode == MODES[1] else np.zeros(7)
                    if mode == MODES[1]:
                        trajectory.input.target_velocity = terminal.tolist()
                    # Match Ruckig.update/calculate's own false/true validation,
                    # preserving the baseline's treatment of generated current
                    # state. Do not add a stricter current-limit roundoff gate.
                    # Ruckig 0.12.2 exposes these arguments, verified against
                    # https://github.com/pantor/ruckig/blob/v0.12.2/src/ruckig/python.cpp
                    if not trajectory.generator.validate_input(trajectory.input,
                            check_current_state_within_limits=False,check_target_state_within_limits=True):
                        raise ValueError('Native Ruckig input validation rejected target/state')
                    for _ in range(TICKS):
                        command = trajectory.step(); tick_count += 1
                        jerk = (trajectory.acceleration-previous_acceleration)/LIMITS['dt']
                        for key,value in (('velocity_rad_s',trajectory.velocity),
                                          ('acceleration_rad_s2',trajectory.acceleration),('jerk_rad_s3',jerk)):
                            peaks[key] = max(peaks[key],float(np.max(np.abs(value))))
                        previous_acceleration = trajectory.acceleration.copy()
                    a.need(abs(float(arrays['times'][index+1])-proposal.execute_at_s) < 1e-7,
                           'Measured teacher label-time reference is not adjacent')
                    rows.append(dict(segment_id=sid,source_index=index,origin_index=origin,prefix_index=k,
                        label_time_s=proposal.execute_at_s,elapsed_s=(index-segment['source_start']+1)*PERIOD_S,
                        teacher_target_q=proposal.proposal.copy(),command_q=np.array(command,float),
                        teacher_measured_q_at_label=np.array(arrays['state'][index+1,:7],float),
                        target_velocity=terminal.copy(),command_velocity=trajectory.velocity.copy(),
                        command_acceleration=trajectory.acceleration.copy()))
    except Exception as error:
        failure = dict(segment_id=sid,query_origin=origin,source_index=index,error=type(error).__name__,reason=str(error))
    return dict(completed=failure is None, failure=failure, arrays=_pack(rows), physics_tick_count=0,
                generated_command_ticks=tick_count, analytic_command_derivative_peaks=peaks,
                native_trajectory=trajectory_factory is JointTrajectory)


def _summary(values):
    values = np.asarray(values,float)
    return dict(mean=float(values.mean()),median=float(np.median(values)),p95=float(np.percentile(values,95)),max=float(values.max()))


def metrics(arrays, kin):
    """Zero-compression FK in left-base coordinates; never a PhysX measurement."""
    count = len(arrays['source_index'])
    if not count:
        return dict(all=dict(samples=0),first_1s=dict(samples=0),after_1s=dict(samples=0),segments=[])
    poses = {name:np.stack([kin.tcp(q) for q in arrays[name]]) for name in
             ('command_q','teacher_target_q','teacher_measured_q_at_label')}
    from scipy.spatial.transform import Rotation
    def compare(mask, reference):
        error = arrays['command_q'][mask]-arrays[reference][mask]
        left, right = poses['command_q'][mask], poses[reference][mask]
        angles = Rotation.from_matrix(left[:,:3,:3] @ right[:,:3,:3].transpose(0,2,1)).magnitude()
        return dict(joint_mae_rad=float(np.abs(error).mean()),joint_max_abs_rad=float(np.abs(error).max()),
            joint_signed_bias_rad=error.mean(axis=0).tolist(),joint_mae_per_joint_rad=np.abs(error).mean(axis=0).tolist(),
            tcp_position_m=_summary(np.linalg.norm(left[:,:3,3]-right[:,:3,3],axis=1)),
            tcp_orientation_rad=_summary(angles))
    def group(mask):
        n = int(mask.sum())
        if not n:return dict(samples=0)
        return dict(samples=n,command_vs_teacher_target=compare(mask,'teacher_target_q'),
                    command_vs_recorded_teacher_measurement=compare(mask,'teacher_measured_q_at_label'))
    startup = arrays['elapsed_s'] <= 1.+1e-9
    return dict(all=group(np.ones(count,bool)),first_1s=group(startup),after_1s=group(~startup),
        segments=[dict(segment_id=int(sid),all=group(arrays['segment_id']==sid),
                       first_1s=group((arrays['segment_id']==sid)&startup),
                       after_1s=group((arrays['segment_id']==sid)&~startup)) for sid in np.unique(arrays['segment_id'])])


def run(config, output):
    a.need(config.get('experiment') == 'executor_probe' and config.get('mode') == 'teacher_replay'
           and config.get('device') == 'cpu' and config.get('time_limit_seconds') == 600,
           'Only the finite CPU teacher executor probe is declared')
    started = time.monotonic(); deadline = started+config['time_limit_seconds']
    arrays,row,provenance = load_source(config)
    segments = make_segments(arrays,row['starts'])
    ruckig = load_native(config)
    from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
    kin = OpenArmKinematics(); branches = {}; small = {}
    for mode in MODES:
        branch = replay(arrays,segments,mode,deadline=deadline)
        data = branch.pop('arrays'); branch['metrics'] = metrics(data,kin)
        branches[mode] = branch
        small.update({mode+'__'+key:value for key,value in data.items()})
    np.savez_compressed(output/'endpoints.npz',**small)
    comparable = all(branch['completed'] for branch in branches.values())
    comparison = dict(full_identical_schedule_completed=comparable)
    if comparable:
        comparison['ff_minus_zero'] = {}
        for group in ('all','first_1s','after_1s'):
            zero,ff = (branches[mode]['metrics'][group] for mode in MODES)
            if zero['samples']:
                z,f = zero['command_vs_teacher_target'],ff['command_vs_teacher_target']
                comparison['ff_minus_zero'][group] = dict(samples=zero['samples'],
                    joint_mae_rad=f['joint_mae_rad']-z['joint_mae_rad'],
                    mean_tcp_position_m=f['tcp_position_m']['mean']-z['tcp_position_m']['mean'],
                    mean_tcp_orientation_rad=f['tcp_orientation_rad']['mean']-z['tcp_orientation_rad']['mean'])
    report = dict(completed=True,**FLAGS,scope='CPU command/FK compatibility diagnostic only',
        source=provenance,segments=segments,branches=branches,comparison=comparison,
        schedule=dict(horizon=HORIZON,execute_steps=EXECUTION_PREFIX,period_s=PERIOD_S,
                      ticks_per_action=TICKS,limits=LIMITS,tail_policy='omit incomplete H10; no padding, stopping extension, or boundary bridge'),
        initialization='Each segment restarts from its recorded measured_q with command v/a=0. Initial q-to-first-command includes teacher servo lag; first 1s reported separately.',
        feedforward='Outgoing secant (a[k+1]-a[k])/0.05 within the same complete H10; a_target stays zero. Reject infeasible velocity/native input; never clamp.',
        limitations='No model/PhysX/contact/collision/observations. Kinematic target lag mixes initialization and executor shaping; recorded teacher measured q is a separate reference, not a simulated result. No runtime change is authorized by this probe.',
        native_ruckig_version=ruckig.__version__,native_ruckig_path=str(ruckig.__file__),
        native_ruckig_sha256=a.sha256(ruckig.__file__),urdf_sha256=a.sha256(kin.urdf_path),
        endpoint_arrays_sha256=a.sha256(output/'endpoints.npz'),elapsed_wall_s=time.monotonic()-started)
    a.save_json(output/'metrics.json',report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if sys.platform != 'linux' or os.environ.get('AUSC_OWNED_RUN') != '1':
        raise RuntimeError('The bounded native CPU probe requires the existing Linux owner')
    config = a.read_json(args.config); args.output.mkdir(parents=True,exist_ok=False)
    a.save_json(args.output/'configuration.json',config)
    try:
        report = run(config,args.output)
        a.save_json(args.output/'COMPLETE.json',dict(completed=True,**FLAGS,
            comparison_completed=report['comparison']['full_identical_schedule_completed'],
            metrics_sha256=a.sha256(args.output/'metrics.json')))
    except Exception as error:
        a.save_json(args.output/'FAILED.json',dict(completed=False,**FLAGS,error=type(error).__name__,reason=str(error)))
        raise
    print('CPU_EXECUTOR_PROBE_COMPLETE',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
