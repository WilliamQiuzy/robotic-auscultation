"""Bind each actual expert acquisition to one simulator recipe before training."""
import json
from pathlib import Path
import shutil
from .dynamics_contract import require_compatible,sha256_file,execution_contract,sha256_json,supervisor_contract


def verify_manifest_dynamics(manifest_path, manifest):
    from .train_openpi import _bound_file,_read_json
    from .observed_supervisor import VERSION as OBSERVED_SUPERVISOR_VERSION
    root=Path(manifest_path).resolve().parent
    common_path=_bound_file(root,manifest.get('dynamics_contract_path'),manifest.get('dynamics_contract_sha256'))
    common=_read_json(common_path)
    require_compatible(common,common,scope='simulator_recipe')
    recipe=common['recipe_sha256']
    if manifest.get('dynamics_recipe_sha256')!=recipe:
        raise ValueError('Manifest dynamics recipe differs from bound contract')
    for record in manifest['episodes']:
        path=_bound_file(root,record.get('dynamics_contract_path'),record.get('dynamics_contract_sha256'))
        own=_read_json(path)
        require_compatible(common,own,scope='simulator_recipe')
        outcome_path=_bound_file(root,record.get('outcome_path'),record.get('outcome_sha256'))
        outcome=_read_json(outcome_path)
        qualification=_read_json(_bound_file(root,record.get('qualification_path'),record.get('qualification_sha256')))
        if qualification.get('evidence',{}).get('artifact_sha256',{}).get('outcome.json')!=sha256_file(outcome_path):
            raise ValueError('Dynamics outcome is not the qualified acquisition')
        if outcome.get('dynamics_recipe_sha256')!=recipe or record.get('dynamics_recipe_sha256')!=recipe:
            raise ValueError('Missing or mixed acquired simulator recipe')
        if (outcome.get('phase_config')!=supervisor_contract() or outcome.get('speed_scale')!=.7
                or outcome.get('joint_speed_limit_rad_s')!=.175):
            raise ValueError('Acquired timing/supervisor differs from declared gentle execution contract')
        if (outcome.get('observed_supervisor_version')!=OBSERVED_SUPERVISOR_VERSION
                or outcome.get('observed_supervisor_sha256')!=sha256_file(Path(__file__).with_name('observed_supervisor.py'))):
            raise ValueError('Demonstration did not execute the shared measured ADC/FK/RGB-D supervisor')
    return dict(recipe_sha256=recipe,contract_sha256=sha256_file(common_path),contract=common,
        execution_contract_sha256=sha256_json(execution_contract()))


def bind_dataset(parent_manifest, records, output):
    """Create a new sealed copy; never retrofit a missing receipt into an episode."""
    from .train_openpi import _read_json,_atomic_json
    source=Path(parent_manifest).resolve();output=Path(output).resolve()
    if output.exists() or output.is_relative_to(source.parent):raise ValueError('New external dataset output required')
    mapping={r['episode_id']:Path(r['dynamics_contract_path']) for r in records}
    if len(mapping)!=len(records):raise ValueError('Duplicate dynamics source record')
    manifest=_read_json(source)
    if set(mapping)!={r['episode_id'] for r in manifest['episodes']}:raise ValueError('All episode receipts required')
    original={str(p.relative_to(source.parent)):sha256_file(p) for p in source.parent.rglob('*') if p.is_file()}
    shutil.copytree(source.parent,output)
    (output/'manifest.json').unlink()
    shutil.copyfile(source,output/'unbound_parent_manifest.json')
    if (output/'seal_receipt.json').exists():
        (output/'seal_receipt.json').rename(output/'unbound_seal_receipt.json')
    dynamics=output/'dynamics';dynamics.mkdir()
    common=None
    for record in manifest['episodes']:
        src=mapping[record['episode_id']]
        receipt=_read_json(src)
        if common is None:common=receipt
        require_compatible(common,receipt,scope='simulator_recipe')
        dest=dynamics/(record['episode_id']+'.json');shutil.copyfile(src,dest)
        record.update(dynamics_contract_path=str(dest.relative_to(output)),
            dynamics_contract_sha256=sha256_file(dest),dynamics_recipe_sha256=receipt['recipe_sha256'])
    if common is None:raise ValueError('Empty dataset')
    _atomic_json(dynamics/'common.json',common)
    manifest.update(dynamics_contract_path='dynamics/common.json',
        dynamics_contract_sha256=sha256_file(dynamics/'common.json'),dynamics_recipe_sha256=common['recipe_sha256'])
    _atomic_json(output/'manifest.pending.json',manifest)
    verify_manifest_dynamics(output/'manifest.pending.json',manifest)
    for name,expected in original.items():
        if sha256_file(source.parent/name)!=expected:raise ValueError('Parent dataset changed while binding')
    (output/'manifest.pending.json').replace(output/'manifest.json')
    _atomic_json(output/'seal_receipt.json',dict(schema='wam_dynamics_dataset_binding_v1',
        manifest_sha256=sha256_file(output/'manifest.json'),parent_manifest_sha256=sha256_file(source),
        dynamics_recipe_sha256=common['recipe_sha256'],episodes=len(manifest['episodes']),
        source_preserved=True,scope='Acquisition/recipe binding; no learned-policy acceptance'))
    return manifest


def main():
    import argparse
    parser=argparse.ArgumentParser()
    for key in ('parent-manifest','records','output'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    result=bind_dataset(args.parent_manifest,json.loads(args.records.read_text()),args.output)
    print(json.dumps(dict(episodes=len(result['episodes']),dynamics_recipe_sha256=result['dynamics_recipe_sha256'])))


if __name__=='__main__':main()
