"""Bounded encoding-repair experiment using the unchanged archived dataset."""
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime,timezone

ROOT=Path('/home/william/wam-vla-v2')


def main():
    run=ROOT/'runs/state_encoding_repair_v1';run.mkdir(parents=True,exist_ok=False)
    py=str(ROOT/'vendor/openpi/.venv/bin/python');manifest=str(ROOT/'datasets/contact_v1/manifest.json')
    env=dict(os.environ,JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',OPENBLAS_NUM_THREADS='1')
    report=dict(scope='state representation ablation on archived data; no new generalization test',
        task_training_complete=False,closed_loop_validated=False,stages=[],started_utc=datetime.now(timezone.utc).isoformat())
    def save():
        tmp=run/'pipeline.partial';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(run/'pipeline.json')
    def stage(name,command,timeout):
        item=dict(name=name,command=command,started_utc=datetime.now(timezone.utc).isoformat());report['stages'].append(item);save()
        start=time.monotonic()
        with (run/(name+'.log')).open('w') as stream:
            result=subprocess.run(command,cwd=ROOT/'project',env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
        item.update(exit_code=result.returncode,wall_seconds=time.monotonic()-start);save()
        if result.returncode:raise RuntimeError(name+' failed; see log')
    save()
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('GPU has an existing compute process')
    try:
        stage('training',[py,'-B','-u','-m','sim.auscultation_repair_v3.pi05.train_openpi','--manifest',manifest,
            '--output',str(run/'model'),'--steps','1000','--batch-size','1','--seed','42','--learning-rate','0.0001',
            '--save-every','100','--evaluate-every','200'],14400)
        latest=json.loads((run/'model/latest.json').read_text())
        metadata=json.loads((run/'model'/latest['directory']/'metadata.json').read_text())
        if metadata['step']!=1000:raise ValueError('Training ended before planned step')
        report['task_training_complete']=True;save()
        best=json.loads((run/'model/best.json').read_text());checkpoint=run/'model'/best['directory']
        report['selected_checkpoint']=str(checkpoint);save()
        stage('evaluation',[py,'-B','-u','-m','sim.auscultation_repair_v3.pi05.evaluate_openpi',
            '--manifest',manifest,'--checkpoint',str(checkpoint),'--output',str(run/'evaluation')],7200)
        report['offline_evaluation_complete']=True
    except Exception as error:
        report['error']=repr(error);raise
    finally:report['finished_utc']=datetime.now(timezone.utc).isoformat();save()


if __name__=='__main__':main()
