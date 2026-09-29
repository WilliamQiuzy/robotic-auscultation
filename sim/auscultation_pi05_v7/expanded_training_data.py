"""Fail-closed expanded experiment view; original dataset bytes stay untouched.

Structural checks may read held-out arrays. Gradients, normalization fitting and
periodic diagnostics cannot consume test predictions. The frozen residual view
refits its state/residual statistics using all thirty new training-role episodes.
"""
from collections import Counter
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import numpy as np
from . import training_data as legacy
from .training_data import observation,stack_observations,decode_prediction,core_sources,RECIPE
from .expanded_contract import (VERSION,PROTOCOL_SHA256,PROTOCOL_HELPER_SHA256,FAMILIES,TARGETS,PHASES,
    read_json,file_sha,is_sha,training_plan,experiment_binding,experiment_source_hashes,json_bytes)

SIDECAR_SCHEMA='wam_pi05_v7_expanded_experiment_sidecar_v1'
ASSEMBLER_SHA256='6b332f3c937820f662fd16ee21151f75e2e55a41cec667c415a56c08791e9539'

def absolute_bound(binding):
    if not isinstance(binding,dict) or set(binding)!={'path','sha256'}:raise ValueError('Exact experiment path/hash binding required')
    path=Path(binding['path'])
    if not path.is_absolute() or path!=path.resolve() or path.is_symlink() or not path.is_file() or not is_sha(binding['sha256']) or file_sha(path)!=binding['sha256']:
        raise ValueError('Experiment evidence path/hash mismatch')
    return path.resolve()

def verify_metadata(manifest_path,sidecar):
    """Re-run frozen metadata qualification links, not just their pass booleans."""
    manifest_path=Path(manifest_path).resolve()
    if not isinstance(sidecar,dict) or sidecar.get('schema')!=SIDECAR_SCHEMA or sidecar.get('training_plan')!=training_plan():
        raise ValueError('Required expanded_experiment sidecar/plan missing or different')
    paths={name:absolute_bound(sidecar.get(name)) for name in ('manifest','protocol','preparation','assembly_receipt','packets','protocol_helper')}
    if paths['manifest']!=manifest_path:raise ValueError('Experiment sidecar belongs to another manifest')
    if sidecar['protocol']['sha256']!=PROTOCOL_SHA256 or sidecar['protocol_helper']['sha256']!=PROTOCOL_HELPER_SHA256:
        raise ValueError('Frozen experiment protocol/helper changed')
    protocol=read_json(paths['protocol']);prep=read_json(paths['preparation']);receipt=read_json(paths['assembly_receipt'])
    if (receipt.get('schema')!='wam_v7_start_family_assembly_receipt_v1' or receipt.get('manifest_sha256')!=file_sha(manifest_path)
            or receipt.get('protocol_sha256')!=PROTOCOL_SHA256 or receipt.get('inputs_preserved') is not True
            or receipt.get('episodes')!=50 or receipt.get('role_counts')!={'train':30,'validation':10,'test':10}
            or Path(receipt.get('dataset_root','')).resolve()!=manifest_path.parent):
        raise ValueError('Actual complete assembly receipt differs')
    evidence=Path(receipt.get('retained_evidence_root','')).resolve()
    if paths['preparation']!=evidence/'preparation.json' or paths['packets']!=evidence/'packets.json':
        raise ValueError('Preparation is not actual assembly evidence')
    sources=receipt.get('source_sha256')
    if not isinstance(sources,dict) or ASSEMBLER_SHA256 not in sources.values() or PROTOCOL_HELPER_SHA256 not in sources.values():
        raise ValueError('Frozen assembler/protocol source inventory missing')
    for name,digest in sources.items():absolute_bound(dict(path=name,sha256=digest))
    spec=importlib.util.spec_from_file_location('_expanded_frozen_protocol',paths['protocol_helper'])
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    actual=helper.prepare_seal(protocol,read_json(paths['packets']))
    if actual!=prep or prep.get('metadata_ready') is not True or prep.get('missing_cases') or len(prep.get('records',[]))!=50:
        raise ValueError('Exact fifty-case preparation replay differs')
    manifest=read_json(manifest_path);records=manifest.get('episodes',[])
    if len(records)!=50 or manifest.get('heldout_axes')!=['motion_family']:raise ValueError('Fifty start-family episodes required')
    originals={r['episode_id']:r for r in prep['records']}
    slots=set();ids=set()
    for record in records:
        identity=record.get('episode_id');family=record.get('start_family');target=record.get('target')
        slot=(family,target)
        if identity in ids or slot in slots or family not in FAMILIES or target not in TARGETS:raise ValueError('Duplicate/missing family-target episode')
        ids.add(identity);slots.add(slot)
        original=originals.get(identity,{})
        for key in ('episode_id','target','split','motion_family','start_family','experiment_role','experiment_ancestry'):
            if record.get(key)!=original.get(key):raise ValueError('Experiment ancestry/role changed: '+key)
        if record['split']!=FAMILIES[family] or record['experiment_role']!=FAMILIES[family] or record['motion_family']!='initial_start_family_'+family:
            raise ValueError('Frozen family role changed')
    if slots!={(family,target) for family in FAMILIES for target in TARGETS}:raise ValueError('Incomplete actual family-target coverage')
    return paths,manifest,receipt

def verify_windows(dataset):
    counts={}
    for ei,episode in enumerate(dataset.episodes):
        split=episode['split'];starts=[start for e,start in dataset.indices[split] if e==ei]
        values=Counter(str(episode['phases'][s]) for s in starts)
        if set(values)!=set(PHASES) or any(values[p]<=0 for p in PHASES):
            raise ValueError('Empty episode/phase valid-window bin: '+episode['episode_id'])
        counts[episode['episode_id']]=dict(values)
    if Counter(e['split'] for e in dataset.episodes)!={'train':30,'validation':10,'test':10}:
        raise ValueError('Thirty/ten/ten actual roles required')
    return counts

@dataclass
class Dataset:
    inherited:object
    experiment_binding:dict
    experiment_source_sha256:dict
    manifest_path:Path
    experiment_path:Path
    window_counts:dict
    def __getattr__(self,name):return getattr(self.inherited,name)
    def rgb(self,episode_index):return self.inherited.parent.rgb(episode_index)

def load_dataset(manifest,experiment=None):
    if experiment is None:raise ValueError('Required expanded_experiment sidecar missing')
    manifest,experiment=Path(manifest).resolve(),Path(experiment).resolve()
    paths,_,receipt=verify_metadata(manifest,read_json(experiment))
    base=legacy.load_dataset(manifest)
    if receipt.get('data_hash')!=base.parent.data_hash or receipt.get('dynamics_recipe_sha256')!=base.dynamics['recipe_sha256']:
        raise ValueError('Assembly numeric/dynamics view changed')
    counts=verify_windows(base)
    binding=experiment_binding(file_sha(manifest),file_sha(paths['preparation']),file_sha(experiment))
    return Dataset(base,binding,experiment_source_hashes(),manifest,experiment,counts)

def make_batch(dataset,split,indices,tokenizer):
    if split not in ('train','validation'):raise ValueError('Training/diagnostic entry forbids test batches')
    return legacy.make_batch(dataset,split,indices,tokenizer)

def prepare_experiment(manifest,protocol,preparation,assembly_receipt,protocol_helper,output):
    """Create only after real complete assembly. Does not create a manifest."""
    output=Path(output).resolve()
    protected=[Path(path).resolve().parent for path in (manifest,protocol,preparation,assembly_receipt,protocol_helper)]
    protected.append(Path(__file__).resolve().parent)
    for root in protected:
        if output==root or root in output.parents or output in root.parents:
            raise ValueError('Experiment sidecar output overlaps immutable source/data/evidence tree')
    if output.exists():raise ValueError('New sidecar output required')
    def bind(path):
        path=Path(path).resolve();return dict(path=str(path),sha256=file_sha(path))
    sidecar=dict(schema=SIDECAR_SCHEMA,training_plan=training_plan(),
        **{k:bind(v) for k,v in dict(manifest=manifest,protocol=protocol,preparation=preparation,
            assembly_receipt=assembly_receipt,protocol_helper=protocol_helper,packets=Path(preparation).with_name('packets.json')).items()})
    _,_,verified_receipt=verify_metadata(manifest,sidecar)
    for name in verified_receipt['source_sha256']:
        root=Path(name).resolve().parent
        if output==root or root in output.parents or output in root.parents:
            raise ValueError('Experiment sidecar output overlaps original raw/qualification/source evidence tree')
    base=legacy.load_dataset(manifest);verify_windows(base)
    receipt=read_json(assembly_receipt)
    if receipt.get('data_hash')!=base.parent.data_hash:raise ValueError('Assembly numeric view differs')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream:stream.write(json_bytes(sidecar))
    return sidecar


def main():
    import argparse,json
    parser=argparse.ArgumentParser(description='Bind a real complete fifty-episode assembly; never fabricate a manifest')
    for name in ('manifest','protocol','preparation','assembly-receipt','protocol-helper','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    prepare_experiment(args.manifest,args.protocol,args.preparation,args.assembly_receipt,args.protocol_helper,args.output)
    print(json.dumps(dict(experiment_path=str(args.output.resolve()),experiment_sha256=file_sha(args.output))))

if __name__=='__main__':main()
