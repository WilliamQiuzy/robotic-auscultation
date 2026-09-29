"""Qualify a completed V6 measured-supervisor expert in a separate working copy.

The unchanged V2 qualifier replays raw RGB-D/Hall and retains acquired phases,
exact label pairs and transition exclusions. Its report remains verbatim in
parent_v2/. A distinct report binds the independently replayed V6 supervisor;
no acquisition data, original report, phase or acceptance threshold is edited.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*2**20),b''):digest.update(block)
    return digest.hexdigest()


def inventory(root):
    files={}
    for p in sorted(root.rglob('*')):
        if p.is_symlink():raise ValueError('symlink in source: '+str(p))
        if p.is_file():files[str(p.relative_to(root))]={'sha256':sha(p),'bytes':p.stat().st_size}
    return files



def audit_source_path():
    from . import audit_observed_supervisor
    return Path(audit_observed_supervisor.__file__)


def contact_verifier():
    from .export_joint_dataset import verify_contact_supervision
    return verify_contact_supervision


def verified_collector_sources(episode,outcome):
    from .collect_demo import collector_source_paths,CLASS
    episode=Path(episode).resolve()
    relative=Path(outcome['input_snapshot'])
    manifest_path=episode/relative
    if relative.is_absolute() or '..' in relative.parts or not manifest_path.resolve().is_relative_to(episode):
        raise ValueError('Collector snapshot escapes source episode')
    if sha(manifest_path)!=outcome['input_snapshot_manifest_sha256']:
        raise ValueError('Collector input snapshot manifest differs')
    manifest=json.loads(manifest_path.read_text())
    if (outcome.get('executed_controller_class')!=CLASS or outcome.get('speed_scale')!=.7
            or outcome.get('controller_source')!='analytic_expert_shared_measured_supervisor'):
        raise ValueError('Outcome does not identify the V6 measured collector and fixed timing')
    sources={}
    for key,path in collector_source_paths().items():
        expected=manifest['files']['source/'+key]['sha256']
        if sha(manifest_path.parent/'objects'/expected)!=expected or sha(path)!=expected:
            raise ValueError('Collector source differs from snapshot: '+key)
        if outcome.get('collector_source_hashes',{}).get(key)!=expected:
            raise ValueError('Outcome collector source identity differs: '+key)
        sources[str(path.resolve())]=expected
    return sources


def augment_qualification(parent,episode,outcome,parent_relative_path,parent_sha256):
    """Add new evidence without rewriting the parent or its selected pairs."""
    episode=Path(episode).resolve();child=copy.deepcopy(parent)
    evidence=child.setdefault('evidence',{})
    sources=evidence.setdefault('source_sha256',{})
    sources.update(verified_collector_sources(episode,outcome))
    audit_path=episode/'observed_supervisor_audit.json'
    audit=json.loads(audit_path.read_text())
    audit_source=audit_source_path()
    if audit.get('passed') is not True or audit.get('audit_source_sha256')!=sha(audit_source):
        raise ValueError('Independent supervisor replay failed or auditor source differs')
    sources[str(audit_source.resolve())]=sha(audit_source)
    sources[str(Path(__file__).resolve())]=sha(Path(__file__))
    artifacts=evidence.setdefault('artifact_sha256',{})
    for name,expected in audit['artifact_sha256'].items():
        path=episode/name
        if (Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink()
                or not path.resolve().is_relative_to(episode) or sha(path)!=expected):
            raise ValueError('Observed audit artifact changed or escapes source: '+name)
        if name in artifacts and artifacts[name]!=expected:
            raise ValueError('Parent qualification and supervisor audit disagree: '+name)
        artifacts[name]=expected
    artifacts['observed_supervisor_audit.json']=sha(audit_path)
    contact,_=contact_verifier()(outcome,evidence,episode,sha(episode/'outcome.json'))
    if contact.get('contact_supervisor_equivalence_verified') is not True:
        raise ValueError('Shared observed contact supervision is not verified')
    child['checks']['observed_contact_supervision_verified']=True
    child['qualified']=bool(parent['qualified'] and all(child['checks'].values()) and not child['reasons'])
    evidence['observed_contact_supervision']=contact
    evidence['parent_v2_qualification']=dict(path=parent_relative_path,sha256=parent_sha256,
        semantics='Verbatim independent V2 sensor/mechanics/pose-label report; acquired phases and selected pairs retained')
    evidence['qualification_origin']='V6 measured-supervisor acquisition; unchanged V2 replay plus independent actual supervisor ledger replay'
    return child


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--working-copy',type=Path,required=True)
    parser.add_argument('--episode-id',required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--camera-source',type=Path,required=True)
    parser.add_argument('--expected-recipe-sha256',required=True)
    args=parser.parse_args()
    episode=args.episode.resolve();output=args.output.resolve();working=args.working_copy.resolve()
    if output.exists() or working.exists():raise ValueError('fresh output and working copy required')
    if output==episode or working==episode or output.is_relative_to(episode) or working.is_relative_to(episode):
        raise ValueError('qualification and working copy must be separate from original source')
    from sim.auscultation_vla_v2 import qualify_episode as qe
    collection=json.loads((episode/'collection_result.json').read_text())
    outcome=json.loads((episode/'outcome.json').read_text())
    if not all((collection.get('passed') is True,collection.get('smoothness_target_met') is True,
                outcome.get('success') is True,outcome.get('training_accepted') is True)):
        raise ValueError('candidate collection gates rejected; no qualification attempt')
    protocol=json.loads(args.protocol.read_text())
    if sha(args.protocol)!='4ff1cd5156eeb195ce6f410393abe1a1e194f03e2d11b914fbcfea23829bf2a1':
        raise ValueError('frozen parent protocol hash mismatch')
    if sha(Path(qe.__file__).with_name('action_contract.py'))!=protocol['action_contract_sha256']:
        raise ValueError('frozen V2 action contract hash mismatch')
    requested=outcome['requested_points']
    if len(requested)!=1:raise ValueError('exactly one requested target required')
    split_matches=[name for name,key in [('train','train_targets'),('val','validation_targets'),('test','test_targets')] if requested[0] in protocol[key]]
    if len(split_matches)!=1:raise ValueError('unexpected point or split provenance')
    split=split_matches[0]
    profile=json.loads((episode/'physics_profile.json').read_text())
    if sha(episode/'training_scene.usdc')!=profile['effective_scene_sha256']:
        raise ValueError('effective scene hash mismatch')
    snapshot=json.loads((episode/outcome['input_snapshot']).read_text())
    if snapshot['files']['source/repair/physics_profile.py']['sha256']!=profile['source_sha256']:
        raise ValueError('physics profile source is not snapshot-bound')
    from sim.auscultation_repair_v3 import metrics
    metrics_sha=sha(Path(metrics.__file__))
    if snapshot['files']['source/repair/metrics.py']['sha256']!=metrics_sha:
        raise ValueError('dynamic metric source differs from collection snapshot')
    physics_rows=[json.loads(line) for line in (episode/'expert_trace.jsonl').open()]
    dynamics=metrics.motion_metrics([r['time_s'] for r in physics_rows],
                                   [r['q'] for r in physics_rows],[r['dq'] for r in physics_rows])
    if dynamics!=json.loads((episode/'measured_dynamics.json').read_text()):
        raise ValueError('recorded dynamics differs from independent trace recomputation')
    if dynamics['centered']['acceleration_rad_s2']['max']>2 or dynamics['centered']['jerk_rad_s3']['max']>20:
        raise ValueError('independently recomputed smoothness gate failed')
    from . import dynamics_contract as dc
    dynamics_path=episode/'dynamics_contract.json'
    contract=json.loads(dynamics_path.read_text())
    if snapshot['files']['source/pi05_v6/dynamics_contract.py']['sha256']!=sha(Path(dc.__file__)):
        raise ValueError('dynamics contract verifier differs from frozen snapshot')
    dc.require_compatible(contract,contract,scope='simulator_recipe')
    if contract.get('complete') is not True or contract.get('selected_scope')!='simulator_recipe':
        raise ValueError('incomplete or wrong compatibility scope')
    if contract.get('recipe_sha256')!=args.expected_recipe_sha256 or outcome.get('dynamics_recipe_sha256')!=args.expected_recipe_sha256:
        raise ValueError('episode recipe differs from frozen batch or outcome')
    if contract['evidence'].get('capture')!='actual running Isaac process before controlled motion':
        raise ValueError('missing actual-runtime capture provenance')
    if contract['evidence'].get('recipe_source_sha256')!=snapshot['files']['source/pi05_v6/runtime_dynamics.py']['sha256']:
        raise ValueError('runtime capture source is not snapshot-bound')
    if contract['recipe']['effective_scene_sha256']!=profile['effective_scene_sha256'] or contract['recipe']['physics_profile']!=profile['profile']:
        raise ValueError('recipe and exact scene/profile mismatch')
    dynamics_binding={'schema':'wam_episode_runtime_recipe_binding_v1',
                      'contract_relative_path':'dynamics_contract.json','contract_sha256':sha(dynamics_path),
                      'recipe_sha256':contract['recipe_sha256'],'binding_sha256':contract['binding_sha256'],
                      'selected_scope':'simulator_recipe','complete':True,
                      'runtime_versions':contract['runtime_versions'],
                      'snapshot_manifest_sha256':sha(episode/outcome['input_snapshot']),
                      'capture_source_sha256':contract['evidence']['recipe_source_sha256'],
                      'scope':'matching execution-time simulator recipe; auto-computed effective values and physical calibration unverified'}
    # Verify the new producer and required passed audit before exporting arrays.
    verified_collector_sources(episode,outcome)
    augment_qualification(dict(qualified=False,checks={},reasons=[],evidence={}),episode,outcome,'pending','0'*64)
    # Qualifier checks every source object and exact replay code below. Preserve
    # original files and run its intentionally mutating audits in a working copy.
    before=inventory(episode)
    def copy_immutable_binary(source,destination):
        source=Path(source);relative=source.relative_to(episode)
        mutable_reports={'independent_audit.json','posture_audit.json','verification_summary.json'}
        if str(relative) in mutable_reports:shutil.copy2(source,destination)
        else:os.link(source,destination)
    shutil.copytree(episode,working,copy_function=copy_immutable_binary)
    if inventory(working)!=before:raise ValueError('working-copy content differs before replay')
    output.mkdir(parents=True)
    parent_output=output/'parent_v2'
    parent_result=qe.qualify_episode(working,parent_output,episode_id=args.episode_id,
                                    camera_source_path=args.camera_source,rgb_width=224)
    if inventory(episode)!=before:
        raise RuntimeError('source episode changed during independent qualification')
    parent_report=parent_output/(args.episode_id+'.qualification.json')
    result=augment_qualification(parent_result,episode,outcome,
        str(parent_report.relative_to(output)),sha(parent_report))
    (output/(args.episode_id+'.qualification.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if result['qualified']:
        source=parent_output/(args.episode_id+'.npz')
        if sha(source)!=result['arrays_sha256']:raise ValueError('Parent array hash differs')
        shutil.copy2(source,output/source.name)
    for name in ('observed_supervisor_audit.json','supervisor_decisions_receipt.json',
                 'task_frame_registration.json','outcome.json'):
        shutil.copy2(episode/name,output/name)
    after=inventory(episode)
    receipt={'schema':'wam_v6_observed_expert_qualification_receipt_v1','episode':str(episode),
             'working_copy':str(working),'output':str(output),'episode_id':args.episode_id,
             'qualified':result['qualified'],'policy_acceptance':'not_evaluated',
             'original_all_files_unchanged':before==after,'original_artifacts':before,
             'profile':profile,'runtime_recipe_binding':dynamics_binding,
             'independently_recomputed_dynamics':dynamics,
             'dynamic_metric_source_sha256':metrics_sha,
             'frozen_parent_protocol_sha256':sha(args.protocol),
             'frozen_parent_protocol_source':str(args.protocol),
             'qualifier_source_sha256':sha(Path(qe.__file__)),'helper_source_sha256':sha(Path(__file__)),
             'source_plan_split':split,'source_plan_target':requested[0],
             'ancestry_rule':'Repeated target plan retains frozen target split',
             'no_dataset_mutation':'Standalone V2 pose-action arrays from V6 acquisition; no label relabeling, manifest merge or joint export',
             'output_artifacts':inventory(output),'command_argv':sys.argv}
    (output/'runtime_recipe_binding.json').write_text(json.dumps(dynamics_binding,indent=2)+chr(10))
    shutil.copy2(dynamics_path,output/'dynamics_contract.json')
    shutil.copy2(args.protocol,output/'frozen_parent_protocol.json')
    shutil.copy2(__file__,output/'qualify_expert_source.py')
    receipt['output_artifacts']=inventory(output)
    (output/'derivation_receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'qualified':result['qualified'],'checks':result['checks'],'reasons':result['reasons'],
                      'arrays_sha256':result['arrays_sha256'],'frames':result['evidence'].get('raw_frames_verified'),
                      'labels':result['label_diagnostics'].get('label_count'),
                      'chunks':result['evidence'].get('full_ten_step_chunks'),
                      'original_all_files_unchanged':before==after,
                      'receipt_sha256':sha(output/'derivation_receipt.json')},indent=2),flush=True)
    if before!=after:raise RuntimeError('source episode changed during qualification')
    return 0 if result['qualified'] else 1


if __name__=='__main__':raise SystemExit(main())
