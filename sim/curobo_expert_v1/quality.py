"""Reuse sealed numerical equations with explicit new provenance verification.

An isolated module instance keeps the original import and source files unchanged.
Only the derived-source verifier is extended; every numerical gate is identical.
"""
import importlib.util
import json
from pathlib import Path

from sim.expert_quality_20260924 import quality as original
from .provenance import CLASS, verify_candidate_sources

spec=importlib.util.spec_from_file_location('wam_quality_v1_equations',original.__file__)
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
old_verifier=core.verify_candidate_sources


def _verify(source,outcome):
    if outcome.get('candidate_method')==CLASS:
        return verify_candidate_sources(source,outcome)
    return old_verifier(source,outcome)


core.verify_candidate_sources=_verify
sha256_file=core.sha256_file
verify_hashes=core.verify_hashes
validate_episode_provenance=core.validate_episode_provenance


def audit_episode(episode,output,profile=None):
    own_sources={name:Path(__file__).with_name(name) for name in ('quality.py','provenance.py')}
    expected={name:sha256_file(path) for name,path in own_sources.items()}
    report=core.audit_episode(episode,output,profile)
    report.setdefault('audit_sources',{}).update({name:{'path':str(own_sources[name]),'sha256':value}
                                   for name,value in expected.items()})
    for name,path in own_sources.items():
        if sha256_file(path)!=expected[name]:raise RuntimeError('adapter changed during audit')
    Path(output).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def audit_free_motion(episode,output):
    from sim.expert_quality_20260924 import audit_free_motion as original_free
    spec=importlib.util.spec_from_file_location('sim.expert_quality_20260924._new_free_equations',original_free.__file__)
    free=importlib.util.module_from_spec(spec);spec.loader.exec_module(free)
    free.validate_episode_provenance=validate_episode_provenance
    report=free.audit_free_motion(episode,output)
    report.setdefault('audit_sources',{}).update({'new_quality_adapter':sha256_file(__file__),
                                  'new_provenance':sha256_file(Path(__file__).with_name('provenance.py'))})
    Path(output).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode',type=Path);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    report=audit_episode(args.episode,args.output_dir/'quality_report.json')
    free=audit_free_motion(args.episode,args.output_dir/'free_motion_report.json')
    print(json.dumps({'decision':report['simulation_quality_decision'],'hard_failures':report['hard_failures'],
                      'free_status':free['status'],'free_failures':free['failures']},indent=2))
    return 1 if report['hard_failures'] or free['status']=='failed' else 0


if __name__=='__main__':raise SystemExit(main())
