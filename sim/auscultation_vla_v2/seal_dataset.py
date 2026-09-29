"""Bundle all ten independently qualified target episodes under frozen splits.

Inputs are JSON records with episode_dir, qualification_dir and episode_id.
This packaging step never manufactures qualifications or substitutes failed
episodes. Raw sensor evidence remains in the collector archives on Linux.
"""
import argparse
import json
from pathlib import Path
import shutil
import re
from .train_openpi import _sha,_read_json,_atomic_json,load_dataset
from .training_data import MODALITIES


def seal(records,protocol_path,output):
    protocol_path=Path(protocol_path);protocol=_read_json(protocol_path)
    output=Path(output)
    if output.exists():raise ValueError('new dataset directory required')
    expected={side+str(i) for side in 'RL' for i in range(1,6)}
    target_split={}
    for split in ('train','validation','test'):
        key='validation_targets' if split=='validation' else split+'_targets'
        for target in protocol[key]:
            if target in target_split:raise ValueError('protocol target assigned twice')
            target_split[target]=split
    if set(target_split)!=expected or protocol['heldout_axes']!=['motion_family']:
        raise ValueError('frozen ten-target motion-family protocol required')
    prepared=[];seen=set()
    for item in records:
        directory=Path(item['episode_dir']);qualified=Path(item['qualification_dir']);identity=item['episode_id']
        if not isinstance(identity,str) or not re.fullmatch('[A-Za-z0-9_-]+',identity):
            raise ValueError('invalid episode filename identity')
        outcome=_read_json(directory/'outcome.json');report_path=qualified/(identity+'.qualification.json')
        report=_read_json(report_path);array_path=qualified/(identity+'.npz')
        requested=outcome.get('requested_points')
        if (not isinstance(requested,list) or len(requested)!=1 or requested[0] not in expected
                or requested[0] in seen):raise ValueError('unique single-target episode required')
        target=requested[0];seen.add(target)
        if (report.get('qualified') is not True or report.get('episode_id')!=identity
                or report.get('arrays_sha256')!=_sha(array_path)
                or report.get('evidence',{}).get('artifact_sha256',{}).get('outcome.json')!=_sha(directory/'outcome.json')):
            raise ValueError('qualification or collector outcome binding mismatch: '+target)
        if any(outcome.get(key)!=protocol[key] for key in ('body_source','layout_family')):
            raise ValueError('unplanned body or layout')
        if outcome.get('motion_family')!='independent_target_plan_'+target:
            raise ValueError('motion ancestry differs from independently solved target plan')
        prepared.append((identity,target,outcome,directory,report_path,array_path))
    if seen!=expected:raise ValueError('all ten qualified targets required before sealing; missing '+str(sorted(expected-seen)))
    output.mkdir(parents=True)
    episodes=[]
    for identity,target,outcome,directory,report_path,array_path in prepared:
        for path in (report_path,array_path):
            shutil.copyfile(path,output/path.name)
            if _sha(path)!=_sha(output/path.name):raise ValueError('dataset copy hash mismatch')
        outcome_file=output/(identity+'.outcome.json');shutil.copyfile(directory/'outcome.json',outcome_file)
        episodes.append(dict(episode_id=identity,split=target_split[target],target=target,
            **{key:outcome[key] for key in ('body_source','layout_family','motion_family')},
            path=array_path.name,sha256=_sha(output/array_path.name),qualification_path=report_path.name,
            qualification_sha256=_sha(output/report_path.name),outcome_path=outcome_file.name,
            outcome_sha256=_sha(outcome_file)))
    shutil.copyfile(protocol_path,output/'protocol.json')
    manifest=dict(schema='wam_vla_training_manifest_v1',sealed=True,modalities=MODALITIES,
        heldout_axes=['motion_family'],horizon=10,action_period_s=.05,episodes=episodes,
        protocol_sha256=_sha(output/'protocol.json'),scope=protocol['purpose'])
    # A preflight filename cannot be mistaken for a committed dataset seal.
    candidate=output/'manifest.pending.json';_atomic_json(candidate,manifest)
    dataset=load_dataset(candidate)
    candidate.replace(output/'manifest.json')
    receipt=dict(data_hash=dataset.data_hash,chunks={key:len(value) for key,value in dataset.indices.items()},
                 episodes=len(episodes),unseen_body_evaluation=False,patient_motion_evaluation=False,
                 manifest_sha256=_sha(output/'manifest.json'))
    _atomic_json(output/'seal_receipt.json',receipt)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('records','protocol','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(seal(_read_json(a.records),a.protocol,a.output),indent=2))


if __name__=='__main__':main()
