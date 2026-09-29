"""Reconstruct V8 pair admission in OpenPI CPU, without changing Isaac NumPy."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
from sim.auscultation_pi05_v7 import coverage_pilot_admission_bridge as owned

BASE = owned.BASE
PARENT_INVENTORY = BASE / 'cloud-runtime-v8/horizon-pair-20260928-v1.source_inventory.json'
PARENT_SHA = '144d91be27011df7992fe7784e800b77a01eacab5f7c639837520def6a7c37fe'
OWNED_SHA = '95e39b8dd739ba3772d91892bf7c729ee8fd3a97d5b9eedc59afdbc13e05f10d'
REQUEST_SCHEMA = 'wam_horizon_pair_cpu_admission_request_v1'
RESPONSE_SCHEMA = 'wam_horizon_pair_cpu_admission_response_v1'
RECEIPT_SCHEMA = 'wam_horizon_pair_cpu_admission_bridge_receipt_v1'
ADMISSION_SCHEMA = 'wam_horizon_pair_selected_runtime_admission_v1'
AdmissionFailure = owned.AdmissionFailure
AdmissionInterrupted = owned.AdmissionInterrupted
need, file_sha, digest = owned.need, owned.file_sha, owned.digest
regular, json_bytes, read_json_bytes = owned.regular, owned.json_bytes, owned.read_json_bytes


def root_path():
    return Path(__file__).resolve().parents[2]


def source_contract():
    root = root_path()
    need(root.parent == BASE / 'cloud-runtime-v8', 'Independent frozen V8 snapshot required')
    need(file_sha(regular(PARENT_INVENTORY)) == PARENT_SHA, 'Original343 inventory changed')
    ancestor = read_json_bytes(PARENT_INVENTORY.read_bytes())['source_files']
    need(len(ancestor) == 343, 'Original343 source set required')
    inventory_path = root.with_name(root.name + '.source_inventory.json')
    inventory = read_json_bytes(regular(inventory_path).read_bytes())
    need(inventory.get('schema') == 'wam_horizon_pair_physical_runtime_inventory_v1' and
         inventory.get('source_project') == str(root), 'Explicit physical runtime inventory required')
    files = inventory['source_files']
    need(set(ancestor) < set(files) and all(files[n] == h for n, h in ancestor.items()), 'Original343 ancestry changed')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() or p.is_symlink()}
    need(actual == set(files), 'Exact physical runtime tree required')
    for name, expected in files.items():
        relative = Path(name)
        need(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe inventory path')
        need(file_sha(regular(root / relative)) == expected, 'Frozen runtime source changed: ' + name)
    need(Path(owned.__file__).resolve() == root / 'sim/auscultation_pi05_v7/coverage_pilot_admission_bridge.py' and
         file_sha(owned.__file__) == OWNED_SHA, 'Reviewed CPU process implementation changed')
    need(file_sha(owned.OPENPI_PYTHON) == owned.PYTHON_SHA, 'Bound OpenPI interpreter changed')
    site = owned.OPENPI_PREFIX / 'lib/python3.11/site-packages'
    numerical = {str(site / 'numpy/__init__.py'): owned.NUMPY_SHA,
                 str(site / 'scipy/__init__.py'): owned.SCIPY_SHA}
    for path, expected in numerical.items():
        need(file_sha(regular(path)) == expected, 'Numerical package binding changed')
    need(file_sha(regular(owned.GUARD_PATH)) == owned.GUARD_SHA, 'Owned cleanup helper changed')
    return dict(schema='wam_horizon_pair_cpu_bridge_contract_v1', source_project=str(root),
        parent343_inventory=dict(path=str(PARENT_INVENTORY), sha256=PARENT_SHA),
        runtime_inventory=dict(path=str(inventory_path), sha256=file_sha(inventory_path)),
        extension_source_sha256={n: h for n, h in files.items() if n not in ancestor},
        worker_python=dict(path=str(owned.OPENPI_PYTHON), sha256=owned.PYTHON_SHA, prefix=str(owned.OPENPI_PREFIX)),
        numerical_package_files=numerical, worker_numpy_version='1.26.4', worker_scipy_version='1.15.3',
        owned_process_implementation_sha256=OWNED_SHA,
        owned_process_guard=dict(path=str(owned.GUARD_PATH), sha256=owned.GUARD_SHA),
        timeout_s=owned.TIMEOUT_S, stdout_limit_bytes=owned.STDOUT_LIMIT, stderr_limit_bytes=owned.STDERR_LIMIT,
        verifier='sim.auscultation_pi05_v8.pair_admission.verify_pair_admission',
        cancellation='AdmissionInterrupted propagates after owned cleanup',
        physical_numpy_unchanged=True, model_initialized=False, private_configuration_used=False)


def verify_request(request):
    contract = source_contract()
    need(set(request) == {'schema', 'source_contract', 'source_project', 'pair_admission_path', 'pair_admission_sha256'},
         'Exact public pair admission request required')
    need(request['schema'] == REQUEST_SCHEMA and request['source_contract'] == contract and
         request['source_project'] == str(root_path()), 'CPU request source differs')
    path = Path(request['pair_admission_path'])
    need(path.is_absolute() and path.resolve() == path and not path.is_symlink(), 'Canonical pair admission path required')
    if path.exists():
        need(file_sha(regular(path)) == request['pair_admission_sha256'], 'Pair admission bytes changed')
    else:
        need(request['pair_admission_sha256'] is None, 'Missing admission cannot have a claimed SHA')
    return contract


def read_bound(path, expected_sha):
    raw = regular(path).read_bytes()
    need(owned.file_bytes_sha(raw) == expected_sha, 'Bound input bytes changed before parsing: ' + str(path))
    return read_json_bytes(raw)


def add_artifacts(artifacts, additions):
    for path, expected in additions.items():
        need(path not in artifacts or artifacts[path] == expected, 'Cannot rebind existing artifact: ' + path)
        artifacts[path] = expected


def build_runtime_admission(path, expected_sha):
    from sim.auscultation_pi05_v8 import pair_admission, train
    pair = pair_admission.verify_pair_admission(path, expected_sha)
    need(pair.get('candidate_admitted') is True and pair.get('service_eligible') is True, 'Positive reconstructed pair admission required')
    checkpoint = pair['admitted_checkpoint_binding']
    need(train.checkpoint_binding(checkpoint['checkpoint_path']) == checkpoint, 'Admitted B checkpoint changed')
    metadata_path = Path(checkpoint['checkpoint_path']) / 'metadata.json'
    metadata = read_bound(metadata_path, checkpoint['metadata_sha256'])
    config = metadata['config']
    config_path = Path(config['training_run_path']) / 'configuration.json'
    lock_ref = pair['b_selection_lock_binding']
    lock = read_bound(lock_ref['path'], lock_ref['sha256'])
    need(lock['selection']['run'] == config['training_run_path'], 'Locked original B run differs')
    config_sha = lock['selection']['artifact_sha256'][str(config_path)]
    saved = read_bound(config_path, config_sha)
    need(saved['config'] == config and saved['normalization'] == metadata['normalization'] and
         saved['data_hash'] == checkpoint['data_hash'], 'Exact admitted metadata/config/data differs')
    manifest_sha = saved['manifest_sha256']
    read_bound(saved['manifest'], manifest_sha)
    artifacts = dict(pair['artifact_sha256'])
    add_artifacts(artifacts, {str(path): expected_sha, str(metadata_path): checkpoint['metadata_sha256'],
        str(config_path): config_sha, str(saved['manifest']): manifest_sha,
        lock_ref['path']: lock_ref['sha256']})
    result = dict(schema=ADMISSION_SCHEMA, pair_admission_path=str(path), pair_admission_sha256=expected_sha,
        pair_admission=pair, selected_checkpoint=checkpoint['checkpoint_path'],
        checkpoint_metadata_sha256=checkpoint['metadata_sha256'], checkpoint_binding=checkpoint,
        manifest_path=saved['manifest'], manifest_sha256=manifest_sha,
        dataset_codec_binding=pair['admitted_dataset_codec_binding'], data_hash=checkpoint['data_hash'],
        training_source_sha256=checkpoint['source_sha256'], artifact_sha256=artifacts,
        pair_freshly_reconstructed=True, physical_accepted=False, generalization_validated=False,
        **{k: config[k] for k in ('dynamics_recipe_sha256', 'dynamics_contract_sha256', 'execution_contract_sha256')})
    owned.verify_admission_artifacts(result)
    return result


def worker_response(request):
    environment = owned.verify_worker_environment()
    contract = verify_request(request)
    result = dict(schema=RESPONSE_SCHEMA, request_sha256=digest(request), source_contract=contract,
        environment=environment, status='not_ready', admission=None, pair_reconstructed=False,
        model_initialized=False, service_contacted=False)
    if not Path(request['pair_admission_path']).is_file():
        return result
    admission = build_runtime_admission(request['pair_admission_path'], request['pair_admission_sha256'])
    need(source_contract() == contract, 'Sources changed during CPU reconstruction')
    need(not any(n == 'fcl' or n.startswith('fcl.') or n in {'isaacsim', 'pxr', 'omni'} or
         n.startswith(('isaacsim.', 'pxr.', 'omni.')) for n in sys.modules), 'CPU verifier imported physical SDK')
    result.update(status='verified', admission=admission, pair_reconstructed=True)
    return result


def verify_receipt(receipt, admission):
    need(receipt.get('schema') == RECEIPT_SCHEMA, 'New V8 bridge receipt required')
    need(type(receipt.get('process_exit_code')) is int and receipt['process_exit_code'] == 0 and
         receipt.get('process_group_drained') is True and receipt.get('cancelled') is False,
         'CPU process did not complete cleanly')
    response = receipt['worker_response']
    contract = source_contract()
    need(response.get('schema') == RESPONSE_SCHEMA and response.get('status') == 'verified' and
         response.get('pair_reconstructed') is True and response.get('source_contract') == receipt.get('source_contract') == contract,
         'Fresh V8 CPU reconstruction required')
    request = receipt['request']
    verify_request(request)
    need(response.get('request_sha256') == receipt.get('request_sha256') == digest(request), 'CPU request digest differs')
    need(request['pair_admission_path'] == admission.get('pair_admission_path') and
         request['pair_admission_sha256'] == admission.get('pair_admission_sha256'), 'Requested pair differs')
    identity, parent = receipt['process_identity'], receipt['parent_pid']
    need(type(parent) is int and parent > 1 and type(identity.get('pid')) is int and identity['pid'] > 1 and
         identity.get('pgid') == identity.get('sid') == identity['pid'] and
         isinstance(identity.get('start_ticks'), str) and identity['start_ticks'].isdecimal(), 'Owned CPU identity invalid')
    need(receipt.get('command') == [str(owned.OPENPI_PYTHON), '-B', str(Path(__file__).resolve()), '--worker', '--parent-pid', str(parent)],
         'CPU worker command differs')
    need(response.get('environment') == dict(python=str(owned.OPENPI_PYTHON), prefix=str(owned.OPENPI_PREFIX),
        numpy='1.26.4', scipy='1.15.3', cuda_visible_devices='', jax_platforms='cpu', pid=identity['pid'], parent_pid=parent),
        'Actual CPU environment differs')
    need(response.get('model_initialized') is False and response.get('service_contacted') is False and
         receipt.get('sdk_numpy_unchanged') is True and receipt.get('private_configuration_used') is False,
         'CPU scope changed')
    elapsed = receipt.get('elapsed_s')
    need(type(elapsed) in (int, float) and math.isfinite(elapsed) and 0 <= elapsed <= owned.TIMEOUT_S + 212, 'CPU duration invalid')
    need(receipt.get('process_errors') == [] and receipt.get('cleanup_signals') == [], 'CPU cleanup/error present')
    need(response.get('admission') == admission and receipt.get('admission_sha256') == digest(admission), 'CPU admission bytes differ')
    need(receipt.get('stdout_sha256') == digest(response) and receipt.get('stdout_bytes') == len(json_bytes(response)) <= owned.STDOUT_LIMIT,
         'Worker stdout differs')
    need(type(receipt.get('stderr_bytes')) is int and 0 <= receipt['stderr_bytes'] <= owned.STDERR_LIMIT, 'Worker stderr bound differs')
    need(receipt.get('artifact_sha256_reverified') == owned.verify_admission_artifacts(admission), 'Physical SDK artifact rehash differs')
    return True


def selected_admission_with_receipt(path, expected_sha):
    contract = source_contract()
    request = dict(schema=REQUEST_SCHEMA, source_contract=contract, source_project=str(root_path()),
                   pair_admission_path=str(Path(path).resolve()), pair_admission_sha256=expected_sha)
    verify_request(request)
    command = [str(owned.OPENPI_PYTHON), '-B', str(Path(__file__).resolve()), '--worker', '--parent-pid', str(os.getpid())]
    run = owned.run_bounded(command, json_bytes(request), env=owned.child_environment(root_path()), cwd=root_path())
    try:
        need(not run['errors'] and not run['cancelled'] and run['process_group_drained'], 'CPU pair admission interrupted or failed')
        need(run['exit_code'] in (0, 2), 'CPU pair reconstruction failed')
        response = read_json_bytes(run['stdout'])
        need(run['exit_code'] == 0 and response.get('status') == 'verified', 'Actual pair admission is not ready')
        admission = response.get('admission')
        need(isinstance(admission, dict), 'Missing original pair reconstruction')
        receipt = dict(schema=RECEIPT_SCHEMA, process_exit_code=run['exit_code'], process_group_drained=run['process_group_drained'],
            process_identity=run['process_identity'], cancelled=run['cancelled'], elapsed_s=run['elapsed_s'],
            command=command, parent_pid=os.getpid(), request=request, request_sha256=digest(request),
            source_contract=contract, worker_response=response, process_errors=run['errors'], cleanup_signals=run['signals'],
            stdout_sha256=owned.file_bytes_sha(run['stdout']), stdout_bytes=len(run['stdout']),
            stderr_sha256=owned.file_bytes_sha(run['stderr']), stderr_bytes=len(run['stderr']),
            admission_sha256=digest(admission), artifact_sha256_reverified=owned.verify_admission_artifacts(admission),
            sdk_numpy_unchanged=True, private_configuration_used=False)
        verify_receipt(receipt, admission)
        return admission, receipt
    except (ValueError, KeyError, TypeError, UnicodeError, OSError) as error:
        failure = dict(schema='wam_horizon_pair_cpu_admission_failure_v1', reason=type(error).__name__ + ': ' + str(error),
            source_contract=contract, request=request, command=command, parent_pid=os.getpid(),
            **{k: run[k] for k in ('exit_code', 'process_group_drained', 'errors', 'signals', 'cancelled', 'elapsed_s')},
            process_identity=run.get('process_identity'), stdout_sha256=owned.file_bytes_sha(run['stdout']),
            stderr_sha256=owned.file_bytes_sha(run['stderr']), stderr_excerpt_utf8=run['stderr'][:8192].decode('utf8', errors='replace'))
        kind = AdmissionInterrupted if run['cancelled'] else AdmissionFailure
        raise kind(failure['reason'], failure) from error


def selected_admission(path, expected_sha):
    return selected_admission_with_receipt(path, expected_sha)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--parent-pid', type=int, required=True)
    args = parser.parse_args()
    need(args.worker, 'Explicit internal CPU worker required')
    owned._parent_death_guard(args.parent_pid)
    raw = sys.stdin.buffer.read(65537)
    need(len(raw) <= 65536, 'CPU request exceeds bound')
    response = worker_response(read_json_bytes(raw))
    sys.stdout.buffer.write(json_bytes(response))
    sys.stdout.buffer.flush()
    return 0 if response['status'] == 'verified' else 2


if __name__ == '__main__':
    raise SystemExit(main())
