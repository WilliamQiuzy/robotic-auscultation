"""Additional experiment/selection replay around the unchanged Headroom auditor.

The inherited audit keeps its original schema as a nested result. Its physical,
ADC, source, raw action and whole-stream checks are never weakened or relabeled.
"""
import argparse
import json
from pathlib import Path
from . import expanded_evaluation as E
from . import expanded_runtime_contract as C
from .expanded_contract import experiment_source_paths

SCHEMA='wam_v7_expanded_headroom_action_audit_v1'


def audit_source_paths():
    from .audit_headroom_learned_actions import audit_source_paths as inherited_paths
    return {**inherited_paths(),**experiment_source_paths(),**C.connection_source_paths()}


def verify_experiment_requests(learned,requests,binding,admission):
    C.verify_policy_against_admission(binding,admission)
    E.require(isinstance(learned,list) and learned and isinstance(requests,list) and requests,
        'Actual query and server request records required')
    server={}
    for row in requests:
        identity=row.get('request_id');E.require(isinstance(identity,str) and identity and identity not in server,
            'Duplicate or missing server request identity')
        for field in ('checkpoint_metadata_sha256','experiment_binding','experiment_source_sha256'):
            E.require(row.get(field)==binding[field],'Server experiment/source binding differs: '+field)
        server[identity]=row
    seen=set()
    for row in learned:
        receipt=row.get('inference',{});identity=receipt.get('request_id')
        E.require(identity in server and identity not in seen,'Missing or duplicate matching server request')
        C.verify_policy_against_admission(receipt,admission)
        seen.add(identity)
    return len(seen)


def verify_connection_snapshot(directory,outcome,admission,artifacts):
    directory=Path(directory).resolve();relative=outcome['input_snapshot']
    E.require(isinstance(relative,str) and not Path(relative).is_absolute(),'Contained snapshot path required')
    path=(directory/relative).resolve();E.require(directory in path.parents,'Snapshot escapes episode')
    E.require(E.sha(path)==outcome['input_snapshot_manifest_sha256'],'Snapshot manifest changed')
    artifacts[str(path)]=E.sha(path);snapshot=E.read(path)
    def obj(key):
        digest=snapshot['files'][key]['sha256'];E.require(isinstance(digest,str) and len(digest)==64 and all(c in '0123456789abcdef' for c in digest),'Invalid snapshot object SHA')
        item=path.parent/'objects'/digest;E.require(E.sha(item)==digest,'Snapshot object changed: '+key)
        artifacts[str(item)]=digest;return item
    for label,source in {**experiment_source_paths(),**C.connection_source_paths()}.items():
        generation,name=label.split('/')
        E.require(E.sha(obj('source/pi05_'+generation+'/'+name))==E.sha(source),'Actual expanded runtime source differs: '+label)
    for key,expected in [('expanded_selection_lock',admission['selection_lock_sha256']),
            ('expanded_experiment',admission['experiment_sha256']),('expanded_manifest',admission['manifest_sha256'])]:
        E.require(E.sha(obj('source/pi05_v7/'+key))==expected,'Selected experiment snapshot differs: '+key)
    E.require(E.read(obj('source/pi05_v7/expanded_selected_admission'))==admission,'Admission snapshot differs')
    contract=obj('source/pi05_v7/expanded_connection_contract')
    E.require(E.read(contract)==C.runtime_connection_contract() and
        E.sha(contract)==outcome.get('runtime_connection_contract_file_sha256'),'Connection contract file differs')


def audit_episode(directory,*,expected_checkpoint_metadata_sha256,binding_path=None,server_requests_path=None,raw_sensor_root=None):
    directory=Path(directory).resolve();artifacts={}
    result=dict(schema=SCHEMA,passed=False,errors=[],inherited_action_audit=None,
        expected_checkpoint_metadata_sha256=expected_checkpoint_metadata_sha256,
        experiment_binding_verified=False,selected_checkpoint_lock_verified=False,
        connection_source_snapshot_verified=False,experiment_request_bindings_verified=False,
        artifact_sha256=artifacts,closed_loop_acceptance=False)
    def read(path,rows=False):
        path=Path(path).resolve();artifacts[str(path)]=E.sha(path)
        return [json.loads(line) for line in path.read_text().splitlines()] if rows else E.read(path)
    try:
        outcome=read(directory/'outcome.json');saved=outcome.get('experiment_runtime_binding',{})
        admission=C.selected_admission(saved['selection_lock_path'],saved['selection_lock_sha256'],
            saved['manifest_path'],saved['experiment_path'])
        E.require(admission==saved,'Actual selected admission differs from fresh evidence')
        E.require(admission['checkpoint_metadata_sha256']==expected_checkpoint_metadata_sha256,'Expected selected checkpoint differs')
        C.verify_connection_outcome(outcome,admission)
        verify_connection_snapshot(directory,outcome,admission,artifacts)
        binding=outcome['learned_policy_binding']
        if binding_path is not None:
            external=read(binding_path);C.verify_policy_against_admission(external,admission)
            E.require(external==binding,'External service binding differs from episode')
        E.require(server_requests_path is not None,'Actual server request log required')
        learned=read(directory/'learned_actions.jsonl',rows=True);requests=read(server_requests_path,rows=True)
        checked=verify_experiment_requests(learned,requests,binding,admission)
        from .audit_headroom_learned_actions import audit_episode as inherited_audit
        inherited=inherited_audit(directory,expected_checkpoint_metadata_sha256=expected_checkpoint_metadata_sha256,
            binding_path=binding_path,server_requests_path=server_requests_path,raw_sensor_root=raw_sensor_root)
        E.require(inherited.get('schema')=='wam_v7_headroom_learned_action_audit_v1','Inherited audit schema changed')
        artifacts.update(inherited.get('artifact_sha256',{}));artifacts.update(admission['artifact_sha256'])
        result.update(inherited_action_audit=inherited,experiment_binding_verified=True,
            selected_checkpoint_lock_verified=True,connection_source_snapshot_verified=True,
            experiment_request_bindings_verified=True,expanded_queries_checked=checked,
            experiment_runtime_binding=admission,runtime_connection_contract=C.runtime_connection_contract(),
            passed=inherited.get('passed') is True)
        if not result['passed']:result['errors'].append('Inherited complete Headroom replay failed')
        C.verify_artifacts(artifacts)
    except (OSError,ValueError,KeyError,TypeError,IndexError,RuntimeError) as error:
        result['passed']=False;result['errors'].append(type(error).__name__+': '+str(error))
    result['audit_source_sha256']={name:E.sha(path) for name,path in audit_source_paths().items()}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    p.add_argument('--expected-checkpoint-metadata-sha256',required=True)
    p.add_argument('--binding',type=Path);p.add_argument('--server-requests',type=Path,required=True)
    p.add_argument('--raw-sensor-root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report=audit_episode(a.directory,expected_checkpoint_metadata_sha256=a.expected_checkpoint_metadata_sha256,
        binding_path=a.binding,server_requests_path=a.server_requests,raw_sensor_root=a.raw_sensor_root)
    E.write_new(a.output,report);print(json.dumps(dict(passed=report['passed'],errors=report['errors'])),flush=True)
    raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__':main()
