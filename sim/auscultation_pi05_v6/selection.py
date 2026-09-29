"""Predeclared validation-only sampled-action ranking; no physics acceptance."""
import math
from .action_contract import PHASES


def select_checkpoint(reports, *, data_hash):
    if not reports:raise ValueError('No completed validation evaluations')
    candidates=[];reference_records=None
    for report in reports:
        if (report.get('selection_only') is not True or set(report.get('splits',{}))!={'validation'}
                or report.get('data_hash')!=data_hash or report.get('seed')!=917
                or report.get('sampling_policy')!='fixed deployment seed per query'
                or report.get('sampling_steps')!=10 or report.get('per_phase')!=20):
            raise ValueError('Identical deployment-sampler validation-only reports required')
        validation=report['splits']['validation']
        if reference_records is None:reference_records=validation['records']
        if not reference_records or validation['records']!=reference_records:
            raise ValueError('Checkpoint validation windows differ')
        phases=validation['baseline_by_phase']
        if set(phases)!=PHASES:raise ValueError('Every task phase required for checkpoint ranking')
        errors=[phases[phase]['first_action_joint_mae_rad'] for phase in sorted(PHASES)]
        fraction=validation['first_action_bound_pass_fraction']['baseline']
        if (any(not math.isfinite(e) or e<0 for e in errors)
                or not math.isfinite(fraction) or not 0<=fraction<=1):
            raise ValueError('Invalid candidate action metric')
        candidates.append(dict(checkpoint=report['checkpoint'],checkpoint_step=report['checkpoint_step'],
            checkpoint_metadata_sha256=report['checkpoint_metadata_sha256'],
            first_action_bound_pass_fraction=fraction,
            phase_balanced_first_action_joint_mae_rad=sum(errors)/len(errors)))
    selected=min(candidates,key=lambda item:(1-item['first_action_bound_pass_fraction'],
        item['phase_balanced_first_action_joint_mae_rad'],item['checkpoint_step']))
    return dict(schema='wam_pi05_validation_action_selection_v1',selected=selected,candidates=candidates,
        criterion='Minimize first-action URDF bound failures, then equal-phase first-action MAE in radians, then earlier step',
        data_hash=data_hash,test_split_used=False,closed_loop_validated=False,policy_qualified=False)
