"""CPU orchestration regressions; retained owned guard supplies real cleanup."""
import copy
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
from sim.auscultation_pi05_v8 import admission_bridge as B


def fixture(monkeypatch, tmp_path):
    contract = {'schema': 'synthetic CPU contract, not model evidence'}
    path = tmp_path / 'admission.json'
    path.write_bytes(b'{}\n')
    admission = dict(pair_admission_path=str(path), pair_admission_sha256=B.file_sha(path),
                     artifact_sha256={str(path): B.file_sha(path)})
    monkeypatch.setattr(B, 'source_contract', lambda: contract)
    parent = os.getpid()
    identity = dict(pid=12345, pgid=12345, sid=12345, start_ticks='100')

    def run(command, payload, **kwargs):
        req = B.read_json_bytes(payload)
        response = dict(schema=B.RESPONSE_SCHEMA, request_sha256=B.digest(req), source_contract=contract,
            status='verified', admission=admission, pair_reconstructed=True, model_initialized=False, service_contacted=False,
            environment=dict(python=str(B.owned.OPENPI_PYTHON), prefix=str(B.owned.OPENPI_PREFIX), numpy='1.26.4', scipy='1.15.3',
                             cuda_visible_devices='', jax_platforms='cpu', pid=identity['pid'], parent_pid=parent))
        return dict(exit_code=0, process_group_drained=True, errors=[], signals=[], cancelled=False,
            process_identity=identity, stdout=B.json_bytes(response), stderr=b'', elapsed_s=1.)

    monkeypatch.setattr(B.owned, 'run_bounded', run)
    return path, admission, run


def test_verified_cpu_receipt_binds_original_admission(monkeypatch, tmp_path):
    path, expected, _ = fixture(monkeypatch, tmp_path)
    admission, receipt = B.selected_admission_with_receipt(path, B.file_sha(path))
    assert admission == expected and B.verify_receipt(receipt, admission)
    assert receipt['worker_response']['environment']['cuda_visible_devices'] == ''
    assert receipt['private_configuration_used'] is False
    path.write_bytes(b'{"changed":true}\n')
    with pytest.raises(ValueError, match='bytes changed'):
        B.verify_receipt(receipt, admission)


@pytest.mark.parametrize('change', ['cancel', 'exit', 'drain', 'stdout', 'errors'])
def test_failed_owned_cpu_cannot_admit(monkeypatch, tmp_path, change):
    path, _, original = fixture(monkeypatch, tmp_path)

    def changed(*args, **kwargs):
        run = original(*args, **kwargs)
        if change == 'cancel': run['cancelled'] = True
        if change == 'exit': run['exit_code'] = 1
        if change == 'drain': run['process_group_drained'] = False
        if change == 'stdout': run['stdout'] = b'{"duplicate":1,"duplicate":2}'
        if change == 'errors': run['errors'] = ['simulated cleanup failure']
        return run

    monkeypatch.setattr(B.owned, 'run_bounded', changed)
    exception = B.AdmissionInterrupted if change == 'cancel' else B.AdmissionFailure
    with pytest.raises(exception) as captured:
        B.selected_admission_with_receipt(path, B.file_sha(path))
    assert captured.value.receipt['cancelled'] is (change == 'cancel')
    if change == 'cancel':
        assert isinstance(captured.value, KeyboardInterrupt)
        assert not isinstance(captured.value, Exception)


@pytest.mark.parametrize('field,value', [('command', ['foreign-worker']), ('process_errors', ['error']),
    ('cleanup_signals', ['SIGTERM']), ('process_exit_code', True), ('cancelled', True), ('stdout_sha256', '0' * 64)])
def test_saved_cpu_receipt_tampering_rejects(monkeypatch, tmp_path, field, value):
    path, _, _ = fixture(monkeypatch, tmp_path)
    admission, receipt = B.selected_admission_with_receipt(path, B.file_sha(path))
    receipt = copy.deepcopy(receipt)
    receipt[field] = value
    with pytest.raises(ValueError):
        B.verify_receipt(receipt, admission)


def test_missing_pair_never_calls_reconstruction(monkeypatch, tmp_path):
    contract = {'synthetic': True}
    monkeypatch.setattr(B, 'source_contract', lambda: contract)
    monkeypatch.setattr(B.owned, 'verify_worker_environment', lambda: {'synthetic CPU only': True})
    monkeypatch.setattr(B, 'build_runtime_admission', lambda *_: pytest.fail('must not reconstruct absent pair'))
    request = dict(schema=B.REQUEST_SCHEMA, source_contract=contract, source_project=str(B.root_path()),
                   pair_admission_path=str(tmp_path / 'missing.json'), pair_admission_sha256=None)
    result = B.worker_response(request)
    assert result['status'] == 'not_ready' and result['admission'] is None and not result['pair_reconstructed']


def test_cpu_child_environment_hides_gpu_and_uses_bound_root():
    env = B.owned.child_environment(B.root_path())
    assert env['CUDA_VISIBLE_DEVICES'] == '' and env['JAX_PLATFORMS'] == 'cpu'
    assert env['PYTHONPATH'] == str(B.root_path())
    assert 'LD_LIBRARY_PATH' not in env


def test_existing_artifact_cannot_be_rebound():
    artifacts = {'/synthetic/fixed.json': 'a' * 64}
    with pytest.raises(ValueError, match='Cannot rebind'):
        B.add_artifacts(artifacts, {'/synthetic/fixed.json': 'b' * 64})
    assert artifacts['/synthetic/fixed.json'] == 'a' * 64


def test_metadata_change_after_checkpoint_verification_rejects(monkeypatch, tmp_path):
    from sim.auscultation_pi05_v8 import train
    import sim.auscultation_pi05_v8 as package
    checkpoint_path = tmp_path / 'checkpoint'
    checkpoint_path.mkdir()
    metadata = checkpoint_path / 'metadata.json'
    metadata.write_text('{"config":{"dynamics_recipe_sha256":"original"}}\n')
    checkpoint = dict(checkpoint_path=str(checkpoint_path), metadata_sha256=B.file_sha(metadata))
    pair = dict(candidate_admitted=True, service_eligible=True, admitted_checkpoint_binding=checkpoint,
                artifact_sha256={str(metadata): B.file_sha(metadata)})
    mock_pair = SimpleNamespace(verify_pair_admission=lambda *_: pair)
    monkeypatch.setitem(sys.modules, 'sim.auscultation_pi05_v8.pair_admission', mock_pair)
    monkeypatch.setattr(package, 'pair_admission', mock_pair, raising=False)

    def verified_then_changed(_):
        metadata.write_text('{"config":{"dynamics_recipe_sha256":"changed"}}\n')
        return checkpoint

    monkeypatch.setattr(train, 'checkpoint_binding', verified_then_changed)
    with pytest.raises(ValueError, match='bytes changed before parsing'):
        B.build_runtime_admission(str(tmp_path / 'admission.json'), 'a' * 64)
