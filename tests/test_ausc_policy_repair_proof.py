from copy import deepcopy
import hashlib
from pathlib import Path
import pytest

from sim.ausc_policy.recovery_case import validate_repair_proof


def proof(monkeypatch):
    root=Path('/test/actual20');audit_root=Path('/test/raw20');case=root/'output/history20'
    config=dict(source_run='/test/student',source_files={'trace':'original'},repair_proof=dict(
        run=str(root),owner_sha256='a'*64,case_sha256='b'*64,raw_audit_run=str(audit_root),
        raw_owner_sha256='c'*64,raw_report='report.json',raw_report_sha256='d'*64))
    done=dict(completed=True,gpu_released=True,stage=dict(process_group_drained=True,exit_code=0))
    owner=dict(done,config=dict(moving_origin='history_free20_v1',oracle_timing='checked_edges_v1',
        source_run=config['source_run'],source_files=config['source_files']))
    audited=dict(done,config=dict(experiment='recovery_sensor_audit',cases=[dict(path=str(case),
        owner_sha256='a'*64,case_sha256='b'*64)]))
    code=Path(__file__).resolve().parents[1]/'sim/ausc_policy'
    report=dict(passed=True,training_dataset_eligible=False,learned_policy=False,
        source_hashes={str(code/n):hashlib.sha256((code/n).read_bytes()).hexdigest()
            for n in ('edge_timing.py','recovery_oracle.py','moving_brake.py')})
    records={str(root/'result.json'):owner,str(case/'result.json'):dict(case_passed=True,training_dataset_eligible=False),
        str(audit_root/'result.json'):audited,str(audit_root/'output/report.json'):report}
    monkeypatch.setattr('sim.ausc_policy.act_baseline.pinned_json',lambda p,d:deepcopy(records[str(p)]))
    return config,records,report,audited


def test_completed_mechanical_repair_and_matching_raw_audit_are_required(monkeypatch):
    config,records,report,audited=proof(monkeypatch);validate_repair_proof(config)
    report['passed']=False
    with pytest.raises(ValueError):validate_repair_proof(config)


def test_different_raw_case_binding_cannot_authorize_new_origin(monkeypatch):
    config,records,report,audited=proof(monkeypatch)
    audited['config']['cases'][0]['case_sha256']='e'*64
    with pytest.raises(ValueError):validate_repair_proof(config)


def test_changed_recovery_algorithm_requires_its_own_actual_verification(monkeypatch):
    config,records,report,audited=proof(monkeypatch)
    report['source_hashes'][next(iter(report['source_hashes']))]='e'*64
    with pytest.raises(ValueError,match='algorithm changed'):validate_repair_proof(config)
