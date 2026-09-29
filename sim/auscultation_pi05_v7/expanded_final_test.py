"""Single locked-checkpoint final test; never selects or changes a checkpoint.

The run-scoped consumed marker is created atomically before model/test sampling.
An interrupted attempt stays consumed. Integrity/reconstruction can read sealed
artifacts; test predictions become exposed only in this explicit final stage.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
from pathlib import Path
from . import expanded_evaluation as E
from . import expanded_selection as S


def read_selection_lock(path,expected_sha):
    E.require(E.sha(path)==expected_sha,'Selection lock SHA mismatch')
    lock=E.read(path)
    E.require(lock.get('schema')==S.LOCK_SCHEMA,'Successful expanded selection lock required')
    return lock


@contextmanager
def claim_test_once(lock_path,lock_sha,output,*,marker_path=None):
    """Exclusive creation is the attempt boundary; never remove marker on failure."""
    marker=Path(marker_path) if marker_path is not None else Path(lock_path).parent/'final_test_started.json'
    output=Path(output).resolve()
    E.require(not output.exists(),'Fresh final-test output required')
    # The production caller derives marker from the verified trained-run binding,
    # not from the user output; copying the lock cannot create another attempt.
    E.write_new(marker,dict(schema='wam_pi05_v7_expanded_test_started_v1',
        selection_lock=str(Path(lock_path).resolve()),selection_lock_sha256=lock_sha,output=str(output),
        started_utc=datetime.now(timezone.utc).isoformat(),test_family_exposed=True,
        failure_does_not_release_attempt=True))
    output.mkdir(parents=True,exist_ok=False)
    try:yield output
    except BaseException as exc:
        E.write_new(output/'failed.json',dict(schema='wam_pi05_v7_expanded_test_failed_v1',
            error=type(exc).__name__+': '+str(exc),test_family_exposed=True,attempt_consumed=True))
        raise


def run(args):
    # Fresh source/label/numeric reconstruction has no test inference or model init.
    read_selection_lock(args.lock,args.lock_sha256)
    lock,context=S.verify_selection_lock(args.lock,args.lock_sha256)
    E.require_fresh_output(args.output,context['binding'])
    selected=lock['selection']['selected']; checkpoint=Path(selected['checkpoint'])
    E.require(E.checkpoint_inventory(checkpoint)==selected['checkpoint_sha256'],'Locked checkpoint bytes changed')
    with claim_test_once(args.lock,args.lock_sha256,args.output,marker_path=lock['final_test_marker']) as output:
        d=context['dataset']; rows=E._phase_schedule(d,'test')
        from . import model
        runtime=model.initialize(61,5e-5)
        E.require(runtime['assets']==context['config']['assets'],'Official base assets differ')
        model.restore(runtime,checkpoint,d.data_hash,context['config'],d.normalization)
        predictor=E.make_predictor(context,runtime,model.sampler(runtime,steps=10))
        prediction=E._predict_rows(d,rows,predictor); reference,anchor=E.reference_arrays(d,rows)
        binding=dict(context['binding'],step=selected['step'],checkpoint=str(checkpoint),
            selection_lock_sha256=args.lock_sha256)
        E.save_numeric_report(output/'numeric',prediction,reference,anchor,rows,kind='final_test',binding=binding)
        E.verify_context_unchanged(context)
        E.require(E.sha(args.lock)==args.lock_sha256,'Selection lock changed during test')
        report=dict(schema='wam_pi05_v7_expanded_final_test_v1',selection_lock_sha256=args.lock_sha256,
            selected_step=selected['step'],selected_checkpoint=str(checkpoint),checkpoint_sha256=selected['checkpoint_sha256'],
            numeric_report_sha256=E.sha(output/'numeric/report.json'),numeric_sha256=E.sha(output/'numeric/predictions.npz'),
            binding=binding,test_family_exposed=True,offline_test_complete=True,selection_unchanged=True,
            closed_loop_validated=False,scope='Known targets/body; held-out initial-start family. No interpoint or patient generalization claim.')
        E.write_new(output/'final_test.json',report)
        return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lock',type=Path,required=True);p.add_argument('--lock-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    from sim.auscultation_pi05_v6.train_openpi import ROOT
    import fcntl,subprocess,json
    with (ROOT/'locks/isaac_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        E.require(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU busy')
        result=run(args);print(json.dumps(dict(selected_step=result['selected_step'],offline_test_complete=True)),flush=True)


if __name__=='__main__':main()
