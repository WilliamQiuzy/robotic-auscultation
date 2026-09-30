"""Identity-isolated M2 inventory. No expert episodes or training are generated."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from .geometry_epoch import require_current


def canonical_shape(sequence_shapes):
    values=np.asarray(sequence_shapes,float)
    if (values.ndim!=2 or values.shape[1]!=10 or not len(values)
            or not np.isfinite(values).all() or np.max(np.ptp(values,axis=0))>1e-4):
        raise ValueError('Inconsistent identity shape; quarantine all its sequences pending review')
    return np.median(values,axis=0)


def identity_split(identities, seed):
    unique=set(identities)
    if not unique or any(not re.fullmatch(r'p\d+', value) for value in unique):
        raise ValueError('HuMMan person IDs required')
    ranked=sorted(unique,key=lambda value:hashlib.sha256(f'{seed}:{value}'.encode()).digest())
    heldout=max(1,int(len(ranked)*.1))
    if 2*heldout>=len(ranked):raise ValueError('Too few identities for three splits')
    return dict(train=sorted(ranked[2*heldout:]),validation=sorted(ranked[heldout:2*heldout]),
                test=sorted(ranked[:heldout]))


def require_bulk_qualification(report, split_sha):
    require_current(report)
    total=report.get('randomized_cases',0);success=report.get('successes',0)
    if (type(total) is not int or type(success) is not int or total<100
            or not 0<=success<=total or success/total<.95
            or report.get('physics_executed') is not True
            or report.get('identity_split_sha256')!=split_sha):
        raise ValueError('Bulk generation requires >=95% actual randomized expert success on >=100 cases')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    files=sorted(args.root.glob('*/smpl_params.npz'));records=[];bodies={};errors=[]
    for path in files:
        match=re.fullmatch(r'(p\d+)_a\d+',path.parent.name)
        if not match:raise ValueError('Unexpected sequence directory')
        identity=match[1]
        try:
            with np.load(path,allow_pickle=False) as arrays:
                values={k:np.asarray(arrays[k]) for k in ('global_orient','body_pose','betas','transl')}
            count=len(values['body_pose'])
            if count==0 or any(values[k].shape!=(count,width) or not np.isfinite(values[k]).all()
                for k,width in [('global_orient',3),('body_pose',69),('betas',10),('transl',3)]):
                raise ValueError('Invalid shapes, empty or nonfinite SMPL parameters')
            beta=values['betas'];median=np.median(beta,axis=0)
            bodies.setdefault(identity,[]).append(median)
            records.append(dict(path=path.relative_to(args.root).as_posix(),identity=identity,frames=count,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                beta_within_sequence_max_abs=float(np.max(abs(beta-median)))))
        except (ValueError,KeyError) as error:errors.append(dict(sequence=path.parent.name,error=str(error)))
    split=identity_split(bodies,20260930)
    encoded=(json.dumps(dict(seed=20260930,splits=split),indent=2)+'\n').encode()
    split_sha=hashlib.sha256(encoded).hexdigest()
    (args.output/'identity_split.json').write_bytes(encoded)
    across=[float(np.max(np.ptp(np.asarray(v),axis=0))) for v in bodies.values()]
    eligible={};quarantined=[]
    for identity,values in bodies.items():
        try:eligible[identity]=canonical_shape(values).tolist()
        except ValueError:quarantined.append(identity)
    quality=dict(eligible_shape_betas=eligible,quarantined_identities=quarantined,
        criterion='Across-sequence beta max difference <=1e-4; not an anatomical accuracy claim',
        identity_split_sha256=split_sha)
    summary=dict(sequences_total=len(files),sequences_valid=len(records),identities=len(bodies),
        counts={k:len(v) for k,v in split.items()},identity_split_sha256=split_sha,
        source_inventory_sha256=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest(),
        rejected_sequences=len(errors),beta_within_sequence_max_abs=max(r['beta_within_sequence_max_abs'] for r in records),
        beta_across_sequences_max_abs=max(across),identities_with_beta_variation_above_1e_4=sum(v>1e-4 for v in across),
        eligible_consistent_identities=len(eligible),quarantined_identities=len(quarantined),
        eligible_split_counts={k:sum(i in eligible for i in ids) for k,ids in split.items()},
        gender='not stored; use existing project SMPL_NEUTRAL convention, not inferred patient sex',
        smpl_topology_validated=False,body_generation_complete=False,expert_qualification_complete=False,
        bulk_generation_allowed=False,training_eligible=False)
    for name,value in [('summary.json',summary),('source_inventory.json',records),('rejected_sequences.json',errors),('identity_quality.json',quality)]:
        (args.output/name).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
