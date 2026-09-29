"""Finite, separately versioned development run; no acceptance from loss."""
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime,timezone

ROOT=Path('/home/william/wam-vla-v2')


def main():
    run=ROOT/'runs/pi05_v4_phase_v1';run.mkdir(parents=True,exist_ok=False)
    py=str(ROOT/'vendor/openpi/.venv/bin/python')
    manifest=str(ROOT/'datasets/contact_v1/manifest.json')
    env=dict(os.environ,JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',OPENBLAS_NUM_THREADS='1')
    report=dict(scope='Development training on historical data; no new generalization evidence',
                policy_qualified=False,closed_loop_validated=False,stages=[],
                started_utc=datetime.now(timezone.utc).isoformat())
    def save():
        p=run/'pipeline.partial';p.write_text(json.dumps(report,indent=2)+'\n');p.replace(run/'pipeline.json')
    def stage(name,command,timeout):
        item=dict(name=name,command=command);report['stages'].append(item);save();start=time.monotonic()
        with (run/(name+'.log')).open('w') as f:
            process=subprocess.run(command,cwd=ROOT/'project',env=env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
        item.update(exit_code=process.returncode,seconds=time.monotonic()-start);save()
        if process.returncode:raise RuntimeError(name+' failed')
    save()
    try:
        stage('training',[py,'-u','-B','-m','sim.auscultation_pi05_v4.train_openpi',
              '--manifest',manifest,'--output',str(run/'model'),'--steps','4000','--batch-size','4',
              '--learning-rate','0.00005','--seed','51','--save-every','1000','--evaluate-every','1000'],14400)
        best=json.loads((run/'model/best.json').read_text());checkpoint=run/'model'/best['directory']
        report['selected_checkpoint']=str(checkpoint);report['training_complete']=True;save()
        stage('evaluation',[py,'-u','-B','-m','sim.auscultation_pi05_v4.evaluate_openpi',
              '--manifest',manifest,'--checkpoint',str(checkpoint),'--output',str(run/'evaluation')],7200)
    except Exception as e:report['error']=repr(e);raise
    finally:report['finished_utc']=datetime.now(timezone.utc).isoformat();save()


if __name__=='__main__':main()
