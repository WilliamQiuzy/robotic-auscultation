"""Package ten qualified real Isaac recordings without changing source evidence."""
import argparse
import json
from pathlib import Path
import shutil

from sim.auscultation_vla_v2.seal_dataset import seal
from sim.auscultation_expert_layout_v1.kinematics import DEFAULT_URDF_PATH
from .dynamics_contract import sha256_file
from .dynamics_binding import bind_dataset
from .export_joint_dataset import export_dataset
from .derive_task_dataset import derive_dataset
from .train_openpi import _atomic_json, _read_json, load_dataset


def assemble(records, protocol, output):
    output=Path(output).resolve()
    if output.exists():raise ValueError('New assembly directory required')
    if len(records)!=10:raise ValueError('All ten qualified acquisitions required')
    prepared=[];receipt_records=[];sources={};trace_inputs={}
    for record in records:
        episode=Path(record['episode_dir']).resolve()
        qualification=Path(record['qualification_dir']).resolve()
        identity=record['episode_id']
        if not isinstance(identity,str) or Path(identity).name!=identity or identity in {'.','..'}:
            raise ValueError('Invalid episode identity')
        own=qualification/'dynamics_contract.json'
        binding=_read_json(qualification/'runtime_recipe_binding.json')
        acquired=episode/'dynamics_contract.json'
        if (binding.get('complete') is not True
                or binding.get('contract_sha256')!=sha256_file(own)
                or sha256_file(own)!=sha256_file(acquired)):
            raise ValueError('Acquired dynamics receipt does not match independent qualification')
        audit_path=episode/'observed_supervisor_audit.json'
        audit=_read_json(audit_path)
        if audit.get('schema')!='wam_observed_supervisor_replay_audit_v1' or audit.get('passed') is not True:
            raise ValueError('Independent measured-supervisor replay must pass before assembly')
        trace_inputs[identity]=dict(audit['artifact_sha256'])
        trace_inputs[identity]['observed_supervisor_audit.json']=sha256_file(audit_path)
        for relative,expected in trace_inputs[identity].items():
            path=episode/relative
            if (Path(relative).is_absolute() or not path.resolve().is_relative_to(episode)
                    or path.is_symlink() or sha256_file(path)!=expected):
                raise ValueError('Observed-supervisor source artifact changed: '+relative)
            sources[str(path)]=expected
        for path in [own,acquired,episode/'expert_trace.jsonl',episode/'sensors/frames.jsonl',
                     qualification/(identity+'.npz'),qualification/(identity+'.qualification.json'),
                     episode/'outcome.json',qualification/'runtime_recipe_binding.json']:
            sources[str(path)]=sha256_file(path)
        prepared.append(dict(episode_dir=str(episode),qualification_dir=str(qualification),episode_id=identity))
        receipt_records.append(dict(episode_id=identity,dynamics_contract_path=str(acquired)))
    output.mkdir(parents=True)
    _atomic_json(output/'input_records.json',prepared)
    _atomic_json(output/'source_hashes.json',sources)
    trace_root=output/'trace_views';trace_root.mkdir()
    for record in prepared:
        # Only immutable trace copies in the exporter's established directory
        # convention; no synthetic frames, actions or qualifications are created.
        view=trace_root/('expert_rgbd_'+record['episode_id']);(view/'sensors').mkdir(parents=True)
        source=Path(record['episode_dir'])
        for name in trace_inputs[record['episode_id']]:
            (view/name).parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source/name,view/name)
            if sha256_file(view/name)!=sources[str(source/name)]:raise ValueError('Trace copy mismatch')
    parent=output/'qualified_pose'
    parent_receipt=seal(prepared,protocol,parent)
    derived=output/'absolute_unbound'
    derivation=export_dataset(parent/'manifest.json',trace_root,derived,DEFAULT_URDF_PATH)
    conditioned=output/'task_conditioned_unbound'
    derive_dataset(derived/'manifest.json',trace_root,conditioned)
    final=output/'task_conditioned_bound'
    bind_dataset(conditioned/'manifest.json',receipt_records,final)
    dataset=load_dataset(final/'manifest.json')
    for name,expected in sources.items():
        if sha256_file(name)!=expected:raise ValueError('Input changed during assembly: '+name)
    receipt=dict(schema='wam_pi05_v6_assembly_v1',manifest=str(final/'manifest.json'),
        manifest_sha256=sha256_file(final/'manifest.json'),data_hash=dataset.data_hash,
        dynamics_recipe_sha256=dataset.dynamics['recipe_sha256'],
        parent_receipt=parent_receipt,derived_rows=derivation['rows'],episodes=len(dataset.episodes),
        chunks={split:len(values) for split,values in dataset.indices.items()},
        inputs_preserved=True,learned_policy_qualified=False,
        action_encoding_support=dataset.groups['action_encoding_support'],
        scope='Ten qualified same-body/layout acquisitions; no new generalization or policy acceptance')
    _atomic_json(output/'assembly_receipt.json',receipt)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('records','protocol','output'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(assemble(_read_json(args.records),args.protocol,args.output),indent=2))
