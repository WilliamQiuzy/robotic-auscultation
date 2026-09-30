"""Persist native finite-head rejection evidence; no IK or physics execution."""
import hashlib
import json
from pathlib import Path
from .fit_refinement import FitRefinementConfig,refine_finite_head_fit


def run_fit_diagnostic(oracle,targets,output,settings):
    config=FitRefinementConfig(**settings);output=Path(output)
    names=[target['name'] for target in targets]
    if (not names or len(set(names))!=len(names) or
            any(name not in {s+str(i) for s in 'LR' for i in range(1,6)} for name in names)):
        raise ValueError('Invalid diagnostic target list')
    rows=[]
    for target in targets:
        name=target['name'];ledger=output/(name+'_fit_attempts.jsonl')
        with ledger.open('x') as stream:
            def sink(attempt):
                stream.write(json.dumps(attempt,allow_nan=False)+'\n');stream.flush()
            report=refine_finite_head_fit(oracle,target['position_world_m'],target['normal_world'],
                                         sink=sink,config=config,compression=.0025)
        report.update(name=name,physics_executed=False,training_eligible=False,
            raw_attempts_file=ledger.name,raw_attempts_sha256=hashlib.sha256(ledger.read_bytes()).hexdigest())
        with (output/(name+'_refined_fits.json')).open('x') as stream:
            stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
        row=dict(name=name,fit_candidates=len(report['candidates']),
            orientation_attempts=report['orientation_attempts'],stage_counts=report['stage_counts'],
            raw_attempts_sha256=report['raw_attempts_sha256'],physics_executed=False,training_eligible=False)
        rows.append(row);print('REFINED_FITS',row,flush=True)
    return rows
