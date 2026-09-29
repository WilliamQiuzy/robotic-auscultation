"""Build a new ten-plan input set while preserving eight baseline plans bytewise."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from .contract import (ROOT, PROJECT, PLANS, TARGETS, read, sha, require,
                       validate_materialized_set, verify_contract, candidate_motion_changed)


def selected_source_document(raw):
    document=json.loads(raw)
    document.pop('revalidation',None)
    return document


def selected_source_record(name,path,raw,summary_path,summary,contract):
    """A selected source is an exact pinned baseline or an exact search member."""
    path=Path(path).resolve();summary_path=Path(summary_path).resolve()
    require(PROJECT in path.parents,'selection source outside project')
    digest=hashlib.sha256(raw).hexdigest()
    document=json.loads(raw)
    require(len(document.get('points',[]))==1 and document['points'][0].get('name')==name,
            'selected point mismatch: '+name)
    result={'source_path':str(path.relative_to(PROJECT)),'source_sha256':digest}
    if digest==contract['baseline_plans'][name]['sha256']:
        return dict(result,source_kind='pinned_baseline',search_candidate_id=None)
    matches=[]
    for candidate in summary.get('candidates',[]):
        relative=candidate.get('plan_path')
        if not isinstance(relative,str) or Path(relative).is_absolute():continue
        expected=(summary_path.parent/relative).resolve()
        if (summary_path.parent in expected.parents and expected==path
                and candidate.get('name')==name and candidate.get('plan_sha256')==digest):
            matches.append(candidate)
    require(len(matches)==1 and isinstance(matches[0].get('id'),str),'selected plan is not one exact search candidate: '+name)
    return dict(result,source_kind='search_candidate',search_candidate_id=matches[0]['id'])


def materialize(output,r1_plan,l1_plan,search_summary,contract_path=ROOT/'contract.json',pair_report=None):
    output=Path(output).resolve()
    require(PROJECT in output.parents and not output.exists(),'candidate output must be a new project directory')
    contract_raw=Path(contract_path).read_bytes();contract=json.loads(contract_raw);verify_contract(contract)
    sources=[Path(r1_plan).resolve(),Path(l1_plan).resolve(),Path(search_summary).resolve()]
    require(all(PROJECT in p.parents and p.is_file() for p in sources),'selection inputs must be project files')
    if pair_report is not None:
        sources.append(Path(pair_report).resolve())
        require(PROJECT in sources[-1].parents and sources[-1].is_file(),'pair report must be a project file')
    # Capture once. All membership, serialization, and manifest hashes derive
    # from these bytes, then every source is checked again before publication.
    captured={path:path.read_bytes() for path in sources}
    summary=json.loads(captured[sources[2]])
    selected={name:selected_source_record(name,path,captured[path],sources[2],summary,contract)
              for name,path in zip(('R1','L1'),sources[:2])}
    chosen={name:selected_source_document(captured[path]) for name,path in zip(('R1','L1'),sources[:2])}
    require(any(candidate_motion_changed(doc,read(PROJECT/contract['baseline_plans'][name]['path']))
                for name,doc in chosen.items()),'at least one selected upper plan must change actual posture or motion')
    output.mkdir(parents=True)
    filenames=set()
    for name in TARGETS:
        reference=read(PLANS/f'planned_paths_{name}.json')
        filenames.add(f'planned_paths_{name}.json')
        for key in ('audit','targets','collision_metadata','collision_arrays'):
            filenames.add(reference['provenance'][key+'_file'])
    filenames.update(n for n in ('posture_limits.json','solver_posture_limits.json','reference_posture_limits.json')
                     if (PLANS/n).is_file())
    for name in sorted(filenames):shutil.copyfile(PLANS/name,output/name)
    for name,doc in chosen.items():
        (output/f'planned_paths_{name}.json').write_text(json.dumps(doc,indent=2,allow_nan=False)+'\n')
    paths=validate_materialized_set(output,contract,require_revalidation=False,contract_path=contract_path)
    for name in ('R1','L1'):
        selected[name]['materialized_before_revalidation_sha256']=paths[name]
        require(read(output/f'planned_paths_{name}.json')==chosen[name],'materialized source semantics changed: '+name)
    require(Path(contract_path).read_bytes()==contract_raw,'contract changed during materialization')
    require(all(path.read_bytes()==raw for path,raw in captured.items()),'selection source changed during materialization')
    selection={'schema':'upper_pair_selection_v1','changed_points':['R1','L1'],
        'contract_sha256':hashlib.sha256(contract_raw).hexdigest(),
        'source_sha256':{str(p.relative_to(PROJECT)):hashlib.sha256(raw).hexdigest() for p,raw in captured.items()},
        'selected_plans':selected,'search_summary_source':str(sources[2].relative_to(PROJECT)),
        'pair_report_source':None if pair_report is None else str(sources[3].relative_to(PROJECT)),
        'materialized_before_revalidation_sha256':paths,
        'scope':'Geometry candidate only; independent geometry and new physics required.'}
    (output/'selection.json').write_text(json.dumps(selection,indent=2,allow_nan=False)+'\n')
    return selection


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('output','r1-plan','l1-plan','search-summary'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--contract',type=Path,default=ROOT/'contract.json')
    parser.add_argument('--pair-report',type=Path,help='optional exact ranking report to pin with the selection')
    args=parser.parse_args()
    print(json.dumps(materialize(args.output,args.r1_plan,args.l1_plan,args.search_summary,args.contract,args.pair_report),indent=2))


if __name__=='__main__':main()
