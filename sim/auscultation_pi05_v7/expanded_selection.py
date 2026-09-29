"""CPU-only feasible-first selection for the declared 30/10/10 experiment.

Reconstructs every numeric validation report and known-start bounds veto from
qualified source labels. Training errors and any test predictions never rank.
"""
import argparse
from pathlib import Path
from . import expanded_evaluation as E

SCHEMA='wam_pi05_v7_expanded_selection_v1'
LOCK_SCHEMA='wam_pi05_v7_expanded_selection_lock_v1'
CRITERION=('Require every raw validation chunk and every first eligible FREE train '
    'sentinel chunk inside unchanged URDF limits, with no clipping or tolerance. '
    'Among feasible checkpoints minimize equal-weight episode/phase macro first-action '
    'MAE, macro full-chunk MAE, then earlier optimizer step. No train-error ranking.')


def pick_feasible(candidates):
    feasible=[r for r in candidates if r.get('feasible') is True]
    E.require(feasible,'No feasible checkpoint; test/deployment prohibited')
    return min(feasible,key=lambda r:(r['macro_first_action_mae_rad'],r['macro_full_chunk_mae_rad'],r['step']))


def build_selection(context,evaluation_path):
    """No model import or prediction; exact fresh numeric reconstruction only."""
    evaluation_path=Path(evaluation_path).resolve(strict=True); root=evaluation_path.parent
    report=E.read(evaluation_path); d=context['dataset']; b=context['binding']
    E.require(report.get('schema')=='wam_pi05_v7_expanded_evaluation_completed_v1' and
        report.get('evaluation_complete') is True and report.get('test_predictions_generated') is False and
        report.get('closed_loop_validated') is False,'Completed development-only evaluation required')
    E.require(report.get('binding')==b,'Evaluation source/data/checkpoint binding changed')
    validation=E.build_validation_schedule(d); sentinels=E.build_train_sentinels(d)
    E.require(report.get('validation_schedule')==validation and report.get('train_sentinel_schedule')==sentinels,
        'Evaluation schedule differs from fresh dataset')
    expected_reports=[]; candidates=[]; artifacts={str(evaluation_path):E.sha(evaluation_path)}
    for checkpoint,metadata in zip(context['checkpoints'],context['metadata']):
        step=metadata['step']; binding=dict(b,step=step,checkpoint=str(checkpoint)); results={}
        for kind,rows in [('validation',validation),('train_sentinel',sentinels)]:
            path=root/('step_%06d'%step)/kind/'report.json'
            result=E.verify_numeric_report(path,d,rows,kind=kind,binding=binding)
            expected_reports.append(dict(step=step,kind=kind,path=str(path),sha256=E.sha(path)))
            for artifact in (path,path.with_name('predictions.npz')):artifacts[str(artifact)]=E.sha(artifact)
            results[kind]=result
        val=results['validation']; train=results['train_sentinel']
        E.require(val['scores']['bucket_count']==50,'All fifty validation bins required')
        feasible=val['bounds']['all_passed'] and train['bounds']['all_passed']
        reasons=[]
        if not val['bounds']['all_passed']:reasons.append('validation_raw_urdf_violation')
        if not train['bounds']['all_passed']:reasons.append('known_start_train_raw_urdf_violation')
        candidates.append(dict(step=step,checkpoint=str(checkpoint),checkpoint_sha256=b['checkpoint_inventory'][str(checkpoint)],
            feasible=bool(feasible),filter_reasons=reasons,
            macro_first_action_mae_rad=val['scores']['macro_first_action_mae_rad'],
            macro_full_chunk_mae_rad=val['scores']['macro_full_chunk_mae_rad'],
            validation_bounds=val['bounds'],known_start_bounds=train['bounds']))
    E.require(report.get('reports')==expected_reports,'Report list/source bindings differ')
    try:selected=pick_feasible(candidates)
    except ValueError:selected=None
    E.verify_context_unchanged(context)
    for path,digest in artifacts.items():E.require(E.sha(path)==digest,'Numeric evidence changed during selection')
    return dict(schema=SCHEMA,criterion=CRITERION,binding=b,evaluation_path=str(evaluation_path),
        evaluation_sha256=E.sha(evaluation_path),artifact_sha256=artifacts,candidates=candidates,
        feasible_steps=[c['step'] for c in candidates if c['feasible']],selected=selected,
        status='diagnostic_until_actual_rollout' if selected else 'rejected_no_feasible_checkpoint',
        validation_schedule_sha256=E.json_sha(validation),train_sentinel_schedule_sha256=E.json_sha(sentinels),
        test_predictions_generated=False,closed_loop_validated=False,train_errors_used_for_ranking=False)


def lock_payload(selection,selection_path):
    E.require(selection.get('selected') is not None,'No feasible selection to lock')
    return dict(schema=LOCK_SCHEMA,selection_path=str(Path(selection_path).resolve(strict=True)),
        selection_sha256=E.sha(selection_path),selection=selection,
        test_policy='One final-test attempt per trained run, only this locked checkpoint. Interrupted attempts remain consumed.',
        final_test_marker=str(Path(selection['binding']['run'])/'FINAL_TEST_STARTED.json'),
        closed_loop_validated=False)


def _marker_payload(lock_path,lock):
    return dict(schema='wam_pi05_v7_expanded_selection_locked_v1',
        selection_lock=str(Path(lock_path).resolve(strict=True)),selection_lock_sha256=E.sha(lock_path))


def bind_selection_marker(lock_path):
    lock=E.read(lock_path);run=Path(lock['selection']['binding']['run'])
    E.write_new(run/'SELECTION_LOCKED.json',_marker_payload(lock_path,lock))


def verify_selection_marker(lock_path,lock):
    marker=Path(lock['selection']['binding']['run'])/'SELECTION_LOCKED.json'
    E.require(marker.is_file() and E.read(marker)==_marker_payload(lock_path,lock),
        'Trained-run selection marker differs from this lock')


def verify_selection_lock(path,expected_sha):
    path=Path(path).resolve(strict=True)
    E.require(E.sha(path)==expected_sha,'Selection lock SHA mismatch')
    lock=E.read(path); E.require(lock.get('schema')==LOCK_SCHEMA,'New selection lock schema required')
    verify_selection_marker(path,lock)
    selection_path=Path(lock['selection_path']).resolve(strict=True)
    E.require(E.sha(selection_path)==lock['selection_sha256'],'Selection evidence changed')
    old=lock['selection']; b=old['binding']
    context=E.load_run_context(b['manifest_path'],b['run'],b['experiment_path'])
    rebuilt=build_selection(context,old['evaluation_path'])
    E.require(rebuilt==old and E.read(selection_path)==rebuilt,'Selection differs from fresh numeric reconstruction')
    E.require(lock_payload(rebuilt,selection_path)==lock,'Selection lock fields changed')
    E.require(E.sha(path)==expected_sha,'Selection lock changed during verification')
    return lock,context


def run(args):
    context=E.load_run_context(args.manifest,args.run,args.experiment)
    result=build_selection(context,args.evaluation)
    output=E.require_fresh_output(args.output,context['binding']);output.mkdir(parents=True,exist_ok=False)
    path=output/'selection.json';E.write_new(path,result)
    # No-feasible evidence survives and deliberately has no deployment/test lock.
    E.require(result['selected'] is not None,'No feasible checkpoint; selection evidence preserved')
    lock_path=output/'selection.lock.json';E.write_new(lock_path,lock_payload(result,path));lock_path.chmod(0o444)
    bind_selection_marker(lock_path)
    return dict(selection=str(path),lock=str(lock_path),lock_sha256=E.sha(lock_path),selected_step=result['selected']['step'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('manifest','experiment','run','evaluation','output'):p.add_argument('--'+key,type=Path,required=True)
    import json
    print(json.dumps(run(p.parse_args())),flush=True)


if __name__=='__main__':main()
