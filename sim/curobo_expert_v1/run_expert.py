"""Run new isolated expert policies without modifying any frozen release."""
import argparse
from functools import partial
import json
from pathlib import Path
import subprocess
import sys

from .provenance import ROOT, POLICIES, implementation_sources, bind_manifest, verify_candidate_sources


def worker(argv,args):
    from sim.auscultation_expert_layout_v1 import run_sequence
    from sim.auscultation_expert_layout_v1.run_low_elbow import audit_saved
    from .controller import ExpertController
    if args.bridge_dir is not None:
        from .robot_bridge import verify_bridge
        verify_bridge(args.bridge_dir)
    if args.policy=='curobo':
        from .preflight import validate_preflight
        if args.fk_request is None or args.fk_result is None:raise ValueError('actual GPU FK preflight is required')
        validate_preflight(args.bridge_dir,args.fk_request,args.fk_result)
    original=run_sequence.source_paths
    sources=implementation_sources(args.bridge_dir,args.launcher)
    if args.policy=='curobo':
        sources.update({'new_expert/preflight/fk_request.json':args.fk_request.resolve(),
                        'new_expert/preflight/fk_result.json':args.fk_result.resolve()})
    run_sequence.source_paths=lambda root:{**original(root),**sources}
    from sim.auscultation_expert_layout_v1 import collision
    from .fast_collision import FastCollisionChecker
    original_checker=collision.CollisionChecker
    collision.CollisionChecker=FastCollisionChecker
    def audit(output):
        bind_manifest(output,args.policy)
        return audit_saved(output)
    try:
        return run_sequence.main(argv,controller_class=partial(ExpertController,
            generator_policy=args.policy,bridge_dir=args.bridge_dir,launcher=args.launcher),result_auditor=audit)
    finally:
        run_sequence.source_paths=original
        collision.CollisionChecker=original_checker


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',choices=POLICIES,required=True)
    parser.add_argument('--bridge-dir',type=Path)
    parser.add_argument('--launcher',type=Path)
    parser.add_argument('--fk-request',type=Path)
    parser.add_argument('--fk-result',type=Path)
    parser.add_argument('--worker',action='store_true')
    args,forwarded=parser.parse_known_args(argv)
    if args.worker:return worker(forwarded,args)
    sub=argparse.ArgumentParser(add_help=False);sub.add_argument('--output',type=Path,required=True)
    paths,_=sub.parse_known_args(forwarded);output=paths.output.resolve()
    if output.exists():parser.error('output must be a new directory')
    output.parent.mkdir(parents=True,exist_ok=True);log=output.with_suffix('.log')
    command=[sys.executable,'-m','sim.curobo_expert_v1.run_expert','--worker',*argv]
    with log.open('x') as stream:
        try:code=subprocess.run(command,cwd=ROOT.parents[1],stdout=stream,
                                stderr=subprocess.STDOUT,timeout=7200).returncode
        except subprocess.TimeoutExpired:code=None
    from sim.auscultation_expert_layout_v1.run_low_elbow import verify_saved
    from sim.auscultation_expert_layout_v1.controller import verify_snapshot
    report=verify_saved(output,code)
    try:
        verify_snapshot(output/'input_snapshot',verify_sources=True)
        outcome=json.loads((output/'outcome.json').read_text())
        binding=verify_candidate_sources(output,outcome);report['new_source_binding']=binding
        if binding['status']!='passed':raise ValueError(str(binding['reasons']))
        if outcome.get('candidate_policy')!=args.policy:raise ValueError('outcome policy differs from request')
    except (OSError,ValueError,KeyError) as error:
        report['accepted']=False;report['reasons'].append('independent_sources:'+str(error))
    destination=output/'runner_summary.json' if output.exists() else log.with_suffix('.runner.json')
    destination.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    return 0 if report['accepted'] else 1


if __name__=='__main__':raise SystemExit(main())
