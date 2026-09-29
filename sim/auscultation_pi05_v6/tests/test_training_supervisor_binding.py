import json
from pathlib import Path
import pytest
from sim.auscultation_pi05_v6 import observed_supervisor
from sim.auscultation_pi05_v6.train_openpi import verify_observed_audit,_sha


def fixture(tmp_path):
    artifacts={'outcome.json':'b'*64};source=_sha(Path(observed_supervisor.__file__))
    audit=dict(schema='wam_observed_supervisor_replay_audit_v1',passed=True,
        source_supervisor_version=observed_supervisor.VERSION,source_supervisor_sha256=source,
        audit_source_sha256=_sha(Path(observed_supervisor.__file__).with_name('audit_observed_supervisor.py')),
        artifact_sha256=artifacts,replay=dict(every_observation_recomputed=True,
            every_manager_update_replayed=True,all_active_ticks_covered=True))
    path=tmp_path/'audit.json';path.write_text(json.dumps(audit))
    derivation=dict(observed_supervisor_version=observed_supervisor.VERSION,observed_supervisor_sha256=source,
        independent_supervisor_audit=dict(path=path.name,sha256=_sha(path),artifact_sha256=artifacts))
    return path,audit,derivation,{'outcome_sha256':'b'*64}


def test_copied_audit_binds_source_outcome_and_complete_replay(tmp_path):
    _,_,derivation,record=fixture(tmp_path)
    verify_observed_audit(tmp_path,derivation,record)


@pytest.mark.parametrize('change',['bytes','outcome','source','coverage','old_auditor'])
def test_forged_true_flags_do_not_bypass_audit_binding(tmp_path,change):
    path,audit,derivation,record=fixture(tmp_path)
    if change=='bytes':path.write_text('{}')
    elif change=='outcome':record['outcome_sha256']='c'*64
    else:
        if change=='source':audit['source_supervisor_sha256']='d'*64
        elif change=='old_auditor':audit['audit_source_sha256']='e'*64
        else:audit['replay']['all_active_ticks_covered']=False
        path.write_text(json.dumps(audit));derivation['independent_supervisor_audit']['sha256']=_sha(path)
    with pytest.raises(ValueError):verify_observed_audit(tmp_path,derivation,record)
