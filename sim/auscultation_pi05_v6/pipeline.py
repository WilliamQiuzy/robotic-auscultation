"""Run bounded training and offline evaluation; never award Isaac acceptance."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .train_openpi import _atomic_json, _sha, load_dataset
from .selection import select_checkpoint


def run(manifest, output, *, steps=4000, seed=61):
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    if steps < 1000 or steps % 1000 or seed < 0:
        raise ValueError('A positive multiple of 1000 steps and nonnegative seed are required')
    if output.exists():
        raise ValueError('New run directory required; resume checkpoints explicitly')
    dataset = load_dataset(manifest)
    if not dataset.indices['test']:
        raise ValueError('The frozen protocol requires train, validation and test episodes')
    output.mkdir(parents=True)
    report = dict(schema='wam_pi05_v6_pipeline_v1', manifest=str(manifest),
        manifest_sha256=_sha(manifest), data_hash=dataset.data_hash,
        dynamics_recipe_sha256=dataset.dynamics['recipe_sha256'],
        source_sha256={p.name:_sha(p) for p in Path(__file__).parent.glob('*.py')},
        steps_requested=steps, seed=seed, started_utc=datetime.now(timezone.utc).isoformat(),
        scope='Diagnostic one-body/layout training with held-out target plans; no unseen-person or motion evidence',
        selection='Validation-only sampled first actions: fewer URDF bound failures, then equal-phase joint MAE; evaluated every 1000-step checkpoint with deployment sampling',
        training_complete=False, offline_evaluation_complete=False,
        closed_loop_validated=False, policy_qualified=False, stages=[])
    def save(): _atomic_json(output/'pipeline.json', report)
    def stage(name, arguments):
        command=[sys.executable,'-u','-B','-m','sim.auscultation_pi05_v6.'+arguments[0],*arguments[1:]]
        record=dict(name=name, command=command, started_utc=datetime.now(timezone.utc).isoformat())
        report['stages'].append(record);save()
        started=time.monotonic()
        with (output/(name+'.log')).open('x') as log:
            result=subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False,
                                  env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        record.update(exit_code=result.returncode, seconds=time.monotonic()-started)
        save()
        if result.returncode:
            raise RuntimeError(name+' failed; retained log and last complete checkpoint')
    save()
    try:
        stage('training',['train_openpi','--manifest',str(manifest),'--output',str(output/'model'),
            '--steps',str(steps),'--batch-size','4','--learning-rate','0.00005','--seed',str(seed),
            '--save-every','1000','--evaluate-every','1000'])
        final=output/'model'/('step-%012d'%steps)
        if not final.is_dir(): raise ValueError('Successful process did not produce the final checkpoint')
        report['training_complete']=True
        report['minimum_flow_loss_pointer']=json.loads((output/'model/best.json').read_text())
        candidates=[]
        for step in range(1000,steps+1,1000):
            candidate=output/'model'/('step-%012d'%step)
            evaluation_output=output/('selection_step_%d'%step)
            stage('selection_step_%d'%step,['evaluate_openpi','--manifest',str(manifest),
                '--checkpoint',str(candidate),'--output',str(evaluation_output),'--selection-only'])
            evaluation=json.loads((evaluation_output/'report.json').read_text())
            if (evaluation.get('checkpoint')!=str(candidate)
                    or evaluation.get('checkpoint_metadata_sha256')!=_sha(candidate/'metadata.json')):
                raise ValueError('Candidate validation checkpoint binding invalid')
            candidates.append(evaluation)
        selection=select_checkpoint(candidates,data_hash=dataset.data_hash)
        _atomic_json(output/'checkpoint_selection.json',selection)
        chosen=selection['selected'];selected=Path(chosen['checkpoint'])
        if (selected.parent!=output/'model' or not selected.is_dir()
                or _sha(selected/'metadata.json')!=chosen['checkpoint_metadata_sha256']):
            raise ValueError('Selected checkpoint binding invalid')
        report['selected_checkpoint']=str(selected)
        report['selected_checkpoint_metadata_sha256']=chosen['checkpoint_metadata_sha256'];save()
        stage('evaluation',['evaluate_openpi','--manifest',str(manifest),
            '--checkpoint',str(selected),'--output',str(output/'evaluation')])
        evaluation=json.loads((output/'evaluation/report.json').read_text())
        if (evaluation.get('data_hash')!=dataset.data_hash
                or evaluation.get('checkpoint_metadata_sha256')!=chosen['checkpoint_metadata_sha256']
                or not evaluation.get('splits') or evaluation.get('closed_loop_validated') is not False):
            raise ValueError('Offline evaluation did not produce the bound diagnostic report')
        report['offline_evaluation_complete']=True
    except BaseException as exc:
        report['error']=type(exc).__name__+': '+str(exc)
        raise
    finally:
        report['finished_utc']=datetime.now(timezone.utc).isoformat();save()
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=4000)
    parser.add_argument('--seed',type=int,default=61)
    args=parser.parse_args()
    print(json.dumps(run(args.manifest,args.output,steps=args.steps,seed=args.seed),indent=2))
