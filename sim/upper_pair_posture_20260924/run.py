"""Run upper-posture alternatives with the unchanged latest B controller."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from .contract import (ROOT, PROJECT, RUNTIME, TARGETS, read, require, sha,
                       validate_materialized_set, candidate_motion_changed)


def validate_points(points):
    require(points in (['R1','L1','R2'],list(TARGETS)),'run only matched three-point screen or full ten')
    return points


def validate_outcome_contract(outcome,contract,points):
    require(outcome.get('success') is True,'physics episode did not complete')
    require(outcome.get('candidate_policy')==contract['controller_policy'],'controller policy changed')
    require(outcome.get('speed_scale')==contract['speed_scale'],'speed scale changed')
    require(outcome.get('phase_config')==contract['baseline_phase_config'],'contact phase config changed')
    require(outcome.get('requested_points')==points,'requested point sequence changed')
    require([p['target'] for p in outcome.get('points',[])]==points,'incomplete point sequence')


def verify_selection(input_dir,contract_path=ROOT/'contract.json'):
    from .materialize import selected_source_record,selected_source_document
    input_dir=Path(input_dir)
    selection_path=input_dir/'selection.json'
    selection=read(selection_path)
    require(selection.get('schema')=='upper_pair_selection_v1','invalid selection manifest')
    require(selection.get('contract_sha256')==sha(contract_path),'selection contract changed')
    require(selection.get('changed_points')==['R1','L1'],'selection point scope changed')
    require(set(selection.get('selected_plans',{}))=={'R1','L1'},'selection requires both named upper sources')
    require(set(selection.get('materialized_before_revalidation_sha256',{}))==set(TARGETS),
            'selection must bind every materialized point')
    summary_relative=selection.get('search_summary_source')
    require(isinstance(summary_relative,str),'missing selected search summary')
    expected_sources={summary_relative,*(row['source_path'] for row in selection['selected_plans'].values())}
    if selection.get('pair_report_source') is not None:expected_sources.add(selection['pair_report_source'])
    require(set(selection.get('source_sha256',{}))==expected_sources,'selection source set differs')
    captured={}
    for relative,digest in selection['source_sha256'].items():
        path=(PROJECT/relative).resolve()
        require(PROJECT in path.parents,'selection source outside project: '+relative)
        raw=path.read_bytes();captured[relative]=raw
        require(hashlib.sha256(raw).hexdigest()==digest,'selection source changed: '+relative)
    for name,digest in selection['materialized_before_revalidation_sha256'].items():
        path=input_dir/f'planned_paths_{name}.json'
        if name in ('R1','L1'):path=path.with_suffix('.before_revalidation.json')
        require(sha(path)==digest,'materialized selection changed: '+name)
    contract=read(contract_path)
    summary=json.loads(captured[summary_relative]);summary_path=PROJECT/summary_relative
    changed=False
    for name,record in selection['selected_plans'].items():
        relative=record['source_path'];raw=captured[relative]
        expected=selected_source_record(name,PROJECT/relative,raw,summary_path,summary,contract)
        require(all(record.get(key)==value for key,value in expected.items()),'selected source identity changed: '+name)
        before=input_dir/f'planned_paths_{name}.before_revalidation.json'
        require(record.get('materialized_before_revalidation_sha256')==sha(before),
                'selected source-to-before digest changed: '+name)
        chosen=selected_source_document(raw)
        require(read(before)==chosen,'materialized before plan differs from selected source: '+name)
        changed=changed or candidate_motion_changed(chosen,read(PROJECT/contract['baseline_plans'][name]['path']))
    require(changed,'selection contains no changed upper posture or motion')
    return selection_path,selection


def worker(args):
    from sim.curobo_expert_v1 import run_expert
    contract=read(args.contract)
    validate_materialized_set(args.input_dir,contract,contract_path=args.contract)
    selection_path,selection=verify_selection(args.input_dir,args.contract)
    original=run_expert.implementation_sources
    production=('contract.py','materialize.py','run.py','search.py','search_cli.py','revalidate.py','pair_rank.py')
    captured={name:ROOT/name for name in production}
    captured['contract.json']=args.contract
    captured['selection.json']=selection_path
    for index,relative in enumerate(sorted(selection['source_sha256'])):
        captured['selection_input_'+str(index)]=(PROJECT/relative).resolve()
    run_expert.implementation_sources=lambda bridge,launcher:{**original(bridge,launcher),
        **{'new_expert/upper_pair_posture/'+name:path for name,path in captured.items()}}
    forwarded=['--points',*args.points,'--input-dir',str(args.input_dir),
        '--runtime-dir',str(RUNTIME),'--output',str(args.output),
        '--speed-scale',str(contract['speed_scale'])]
    oldargs=argparse.Namespace(policy=contract['controller_policy'],bridge_dir=None,
                               launcher=None,fk_request=None,fk_result=None)
    try:return run_expert.worker(forwarded,oldargs)
    finally:run_expert.implementation_sources=original


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',type=Path,default=ROOT/'contract.json')
    parser.add_argument('--input-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--points',nargs='+',default=list(TARGETS))
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args(argv)
    args.contract=args.contract.resolve();args.input_dir=args.input_dir.resolve();args.output=args.output.resolve()
    validate_points(args.points)
    contract=read(args.contract)
    plans=validate_materialized_set(args.input_dir,contract,contract_path=args.contract)
    verify_selection(args.input_dir,args.contract)
    if args.worker:return worker(args)
    require(not args.output.exists(),'output must be new')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    log=args.output.with_suffix('.log')
    with log.open('x') as stream:
        try:
            code=subprocess.run([sys.executable,'-B','-m','sim.upper_pair_posture_20260924.run',
                '--worker',*argv],cwd=PROJECT,stdout=stream,stderr=subprocess.STDOUT,timeout=7200).returncode
        except subprocess.TimeoutExpired:code=None
    from sim.auscultation_expert_layout_v1.run_low_elbow import verify_saved
    from sim.auscultation_expert_layout_v1.controller import verify_snapshot
    from sim.curobo_expert_v1.provenance import verify_candidate_sources
    report=verify_saved(args.output,code)
    try:
        verify_snapshot(args.output/'input_snapshot',verify_sources=True)
        outcome=read(args.output/'outcome.json')
        validate_outcome_contract(outcome,contract,args.points)
        binding=verify_candidate_sources(args.output,outcome)
        require(binding['status']=='passed','candidate source binding rejected: '+str(binding['reasons']))
        require(validate_materialized_set(args.input_dir,contract,contract_path=args.contract)==plans,'plans changed during execution')
        verify_selection(args.input_dir,args.contract)
        manifest=read(args.output/'input_snapshot/manifest.json')
        for name in args.points:
            require(manifest['files'][name+'/plan']['sha256']==plans[name],'executed plan differs: '+name)
        report['upper_pair_contract']={'status':'passed','contract_sha256':sha(args.contract),
            'selection_sha256':sha(args.input_dir/'selection.json'),'point_plan_sha256':plans}
        report['new_source_binding']=binding
    except (OSError,ValueError,KeyError,TypeError) as error:
        report['accepted']=False;report.setdefault('reasons',[]).append('upper_pair_contract:'+str(error))
    destination=args.output/'runner_summary.json' if args.output.exists() else log.with_suffix('.runner.json')
    destination.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'accepted':report['accepted'],'reasons':report.get('reasons',[]),
                      'report':str(destination)}),flush=True)
    return 0 if report['accepted'] else 1


if __name__=='__main__':raise SystemExit(main())
