"""Report paired offline outcomes, including regressions and rejected release."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--old',type=Path,required=True)
    p.add_argument('--new',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();old,new=[json.loads(path.read_text()) for path in [args.old,args.new]]
    for key in ('data_hash','sampling_steps','seed','per_phase'):
        if old[key]!=new[key]:raise ValueError('Unpaired evaluation: '+key)
    result=dict(scope='paired diagnostic on previously inspected single-layout plans; no fresh generalization claim',
        data_hash=old['data_hash'],old_report_sha256=hashlib.sha256(args.old.read_bytes()).hexdigest(),
        new_report_sha256=hashlib.sha256(args.new.read_bytes()).hexdigest(),splits={},
        task_training_complete=True,policy_qualified=False,closed_loop_validated=False,
        reasons=['below simple phase-mean baseline on mean translation error',
                 'large-error tail and proposal-bound pass rate regressed',
                 'no fresh held-out body/layout/motion evaluation',
                 'no learned-policy closed-loop Isaac execution',
                 'useful RGB/Hall contribution not established by these interventions'])
    for split in old['splits']:
        a,b=old['splits'][split],new['splits'][split]
        if a['records']!=b['records']:raise ValueError('Evaluation windows differ')
        result['splits'][split]=dict(old=a['metrics']['baseline'],new=b['metrics']['baseline'],
            phase_mean=b['descriptive_baselines']['training_phase_mean'],
            old_bound_pass=a['bound_pass_fraction_after_inactive_force_mask']['baseline'],
            new_bound_pass=b['bound_pass_fraction_after_inactive_force_mask']['baseline'],
            new_by_phase=b['baseline_by_phase'],new_interventions=b['metrics'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:dict(old_mean=v['old']['mean_translation_error_mm'],new_mean=v['new']['mean_translation_error_mm'],
        old_p95=v['old']['p95_translation_error_mm'],new_p95=v['new']['p95_translation_error_mm'],
        old_pass=v['old_bound_pass'],new_pass=v['new_bound_pass']) for k,v in result['splits'].items()},indent=2))


if __name__=='__main__':main()
