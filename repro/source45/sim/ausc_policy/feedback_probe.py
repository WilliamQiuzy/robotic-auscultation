"""Finite native ACT input-response diagnostic, never a recovery oracle.

Translate the registered target in its two tangent axes while preserving q,
rotation, phase and confidence. Compare predicted FK to the unperturbed output.
No observation acquisition, physics, images, Hall intervention or training.
"""
import argparse
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from . import act_baseline as a
from .act_runtime import InferenceService, NativeEngine, load_artifacts
from .executor_probe import load_source
from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics

FLAGS = dict(m0_passed=False, training_updates=0, physics_executed=False,
             closed_loop_evaluated=False, contact_evaluated=False,
             expert_demonstration=False, recovery_oracle=False,
             image_arrays_opened=False, heldout_arrays_opened=False)


def relative_pose(state):
    state = np.asarray(state, float)
    a.need(state.shape == (19,) and np.isfinite(state).all(), 'Invalid state19')
    columns = state[10:16].reshape(2, 3).T
    rotation = np.column_stack((columns, np.cross(columns[:, 0], columns[:, 1])))
    a.need(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0), 'Invalid state rotation')
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_matrix(rotation).as_matrix()
    pose[:3, 3] = state[7:10]
    return pose


def shift_target(state, axis, displacement):
    a.need(axis in (0, 1) and displacement in (-.005, -.001, .001, .005), 'Undeclared target intervention')
    relative_pose(state)
    result = np.asarray(state, np.float32).copy()
    result[7 + axis] -= displacement
    return result


def shift_robot(state, joint, displacement, kin):
    """Change one joint and its FK together, preserving the inferred task frame."""
    a.need(type(joint) is int and 0 <= joint < 7 and displacement in (-.02, -.005, .005, .02),
           'Undeclared robot intervention')
    task_to_tcp = relative_pose(state)
    state = np.asarray(state, np.float32)
    task = kin.tcp(state[:7]) @ np.linalg.inv(task_to_tcp)
    result = state.copy(); result[joint] += displacement
    a.need(np.all(result[:7] >= kin.lower_limits) and np.all(result[:7] <= kin.upper_limits),
           'Robot intervention outside joint bounds; no clipping')
    changed = np.linalg.inv(task) @ kin.tcp(result[:7])
    result[7:10] = changed[:3, 3]
    result[10:16] = changed[:3, :2].T.reshape(-1)
    return result


def robot_responses(state, original, source_start, kin, service, displacements):
    inverse = relative_pose(state) @ np.linalg.inv(kin.tcp(state[:7]))
    clean_origin = (inverse @ kin.tcp(state[:7]))[:3, 3]
    clean_output = (inverse @ kin.tcp(original[4]))[:3, 3]
    rows, skipped = [], []
    for joint in range(7):
        for displacement in displacements:
            try:
                changed = shift_robot(state, joint, displacement, kin)
            except ValueError as error:
                if 'outside joint bounds' not in str(error):
                    raise
                skipped.append(dict(source_start=source_start, joint=joint,
                    joint_shift_rad=displacement, reason=str(error)))
                continue
            raw = service.predict_state(changed)
            initial_delta = (inverse @ kin.tcp(changed[:7]))[:3, 3]-clean_origin
            output_delta = (inverse @ kin.tcp(raw[4]))[:3, 3]-clean_output
            initial_norm = float(np.linalg.norm(initial_delta))
            # Both predictions use the same fixed task. These are differences
            # from clean proposals, not executed corrections or oracle actions.
            rows.append(dict(source_start=source_start, joint=joint, joint_shift_rad=displacement,
                initial_tcp_change_m=initial_delta.tolist(), proposed_h5_tcp_change_m=output_delta.tolist(),
                tcp_perturbation_ratio=float(np.linalg.norm(output_delta)/initial_norm) if initial_norm > 1e-9 else None,
                changed_joint_response_gain=float((raw[4,joint]-original[4,joint])/displacement),
                initial_joint_delta_rad=(changed[:7]-state[:7]).tolist(),
                proposed_h5_joint_delta_rad=(raw[4]-original[4]).tolist(),
                block_relative_h5_change_rad=((raw[4]-changed[:7])-(original[4]-state[:7])).tolist()))
    return rows, skipped


def robot_groups(responses):
    groups = {}
    for magnitude in (.005, .02):
        rows = [r for r in responses if abs(r['joint_shift_rad']) == magnitude]
        a.need(rows, 'No usable robot interventions')
        gains = [r['changed_joint_response_gain'] for r in rows]
        ratios = [r['tcp_perturbation_ratio'] for r in rows if r['tcp_perturbation_ratio'] is not None]
        a.need(ratios, 'No nondegenerate TCP interventions')
        groups[str(magnitude)] = dict(count=len(rows), nondegenerate_tcp_count=len(ratios),
            median_joint_response_gain=float(np.median(gains)),
            median_tcp_perturbation_ratio=float(np.median(ratios)),
            tcp_ratio_percentiles=np.percentile(ratios,[10,50,90]).tolist(),
            tcp_ratio_below_one_count=sum(x < 1 for x in ratios),
            copy_current_q_reference_gain=1., copy_current_q_reference_tcp_ratio=1.)
    return groups


def geometry(pose):
    # The registered task's +Z is the inward tool axis; outward gap is -Z.
    return dict(gap_mm=float(-1000 * pose[2, 3]),
        tangent_mm=float(1000 * np.linalg.norm(pose[:2, 3])),
        axis_deg=float(np.rad2deg(np.arccos(np.clip(pose[2, 2], -1, 1)))),
        so3_rad=float(Rotation.from_matrix(pose[:3, :3]).magnitude()))


def run(config, output):
    robot = config['mode'] == 'robot_state_response'
    a.need(config['experiment'] == 'feedback_probe' and config['mode'] in ('target_geometry_response','robot_state_response')
        and config['device'] == 'cpu' and config['time_limit_seconds'] == 600
        and config['checkpoint_step'] == 2000 and config['anchor_count'] == 16
        and (config.get('displacements_rad') == [-.02, -.005, .005, .02] if robot
             else config.get('displacements_m') == [-.005, -.001, .001, .005]), 'Finite probe scope differs')
    source = Path(config['run_root'])
    owner = a.read_json(source / 'result.json')
    a.need(owner['completed'] is True and owner['gpu_released'] is True, 'Training source still active')
    artifacts = load_artifacts(source / 'output/configuration.json', source / 'output/checkpoint_2000',
                              config['checkpoint_sha256'])
    arrays, row, provenance = load_source(config)
    starts = [s for s in row['starts'] if arrays['phases'][s] == 'APPROACH']
    a.need(len(starts) >= 16, 'Insufficient declared APPROACH anchors')
    chosen = [starts[i] for i in np.rint(np.linspace(0, len(starts)-1, 16)).astype(int)]
    a.need(len(set(chosen)) == 16, 'Duplicate anchors')
    service = InferenceService(artifacts, NativeEngine(artifacts))
    kin = OpenArmKinematics()
    a.need(a.sha256(kin.urdf_path) == config['urdf_sha256'], 'Kinematics binding differs')
    records, responses, repeats, skipped = [], [], [], []
    for s in chosen:
        state = a.small_state(arrays['state'][s], 'R4', 'APPROACH')
        original = service.predict_state(state)
        task_to_tcp = relative_pose(state)
        base_to_task = kin.tcp(state[:7]) @ np.linalg.inv(task_to_tcp)
        inverse = np.linalg.inv(base_to_task)
        predicted = [inverse @ kin.tcp(original[k]) for k in (0, 4, 9)]
        reference = [inverse @ kin.tcp(arrays['actions'][s+k]) for k in (0, 4, 9)]
        records.append(dict(source_start=s, origin=geometry(task_to_tcp),
            predicted=[geometry(p) for p in predicted], reference=[geometry(p) for p in reference]))
        if robot:
            values, omitted = robot_responses(state, original, s, kin, service, config['displacements_rad'])
            responses.extend(values); skipped.extend(omitted)
        for axis in (() if robot else (0, 1)):
            for displacement in config['displacements_m']:
                changed = shift_target(state, axis, displacement)
                raw = service.predict_state(changed)
                # Compare both predictions in the SAME original task frame.
                delta = (inverse @ kin.tcp(raw[4]))[:3, 3] - predicted[1][:3, 3]
                responses.append(dict(source_start=s, axis=axis, target_shift_m=displacement,
                    proposed_h5_tcp_change_m=delta.tolist(),
                    signed_response_gain=float(delta[axis] / displacement),
                    raw_action_max_abs_change_rad=float(np.max(np.abs(raw-original))),
                    expected_direction_agreement=bool(delta[axis] / displacement > 0)))
        repeats.append(float(np.max(np.abs(service.predict_state(state)-original))))
        print('FEEDBACK_ANCHOR', len(records), 'responses', len(responses), 'skipped', len(skipped), flush=True)
    a.need(max(repeats) == 0., 'Deterministic clean-repeat mismatch')
    groups = {}
    for axis in (() if robot else (0, 1)):
        for magnitude in (.001, .005):
            rows = [r for r in responses if r['axis'] == axis and abs(r['target_shift_m']) == magnitude]
            gains = np.asarray([r['signed_response_gain'] for r in rows])
            groups[f'axis{axis}_{magnitude}m'] = dict(count=len(rows),
                direction_agreement_count=sum(r['expected_direction_agreement'] for r in rows),
                median_gain=float(np.median(gains)), minimum_gain=float(gains.min()), maximum_gain=float(gains.max()))
    if robot:
        groups = robot_groups(responses)
    report = dict(completed=True, **FLAGS, anchors=records, responses=responses, groups=groups,
        intervention=config['mode'], skipped=skipped,
        clean_repeat_max_abs_rad=max(repeats), inference_count=32+len(responses),
        receipt=artifacts.receipt, provenance=provenance,
        scope='Counterfactual registered target geometry, q fixed; h5 zero-compression raw-proposal FK, '
              'task frame reconstructed from state19 at origin; no execution or correctness oracle. '
              'Direction is a local response descriptor, not a universally correct collision-aware action label. '
              'No raw depth/RGB/Hall intervention or modality-use conclusion.')
    if robot:
        report['scope'] = ('Counterfactual robot joints and zero-compression TCP changed consistently; '
            'fixed inferred target, phase and confidence. h5 raw proposals, no execution or oracle. '
            'Ratios compare perturbed minus clean FK outputs; ratio below one is a local input-response '
            'descriptor, NOT proof of feedback stability, task convergence, collision safety or generalization. '
            'Only URDF bounds screened, not posture or collision feasibility. No RGB/depth/Hall intervention.')
    a.save_json(output / 'metrics.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != 'linux' or os.environ.get('AUSC_OWNED_RUN') != '1':
        raise RuntimeError('Original guarded Linux CPU owner required')
    config = a.read_json(args.config)
    args.output.mkdir(parents=True, exist_ok=False)
    a.save_json(args.output / 'configuration.json', config)
    try:
        run(config, args.output)
        a.save_json(args.output / 'COMPLETE.json', dict(completed=True, **FLAGS,
            metrics_sha256=a.sha256(args.output / 'metrics.json')))
    except Exception as error:
        a.save_json(args.output / 'FAILED.json', dict(completed=False, **FLAGS,
            error=type(error).__name__, reason=str(error)))
        raise
    print('FEEDBACK_PROBE_COMPLETE', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
