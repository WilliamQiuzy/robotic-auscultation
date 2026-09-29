import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as S
import pytest
from sim.auscultation_pi05_v6.qualify_expert import augment_qualification


def fixture(tmp_path,monkeypatch):
    from sim.auscultation_pi05_v6 import qualify_expert as module
    audit_source=tmp_path/'auditor.py';audit_source.write_text('audit code')
    monkeypatch.setattr(module,'audit_source_path',lambda:audit_source)
    root=tmp_path/'raw';root.mkdir()
    artifact=root/'outcome.json';artifact.write_text('{"success": true}\n')
    ah=hashlib.sha256(artifact.read_bytes()).hexdigest()
    audit=dict(schema='wam_observed_supervisor_replay_audit_v1',passed=True,errors=[],
        audit_source_sha256=hashlib.sha256(audit_source.read_bytes()).hexdigest(),
        artifact_sha256={'outcome.json':ah})
    (root/'observed_supervisor_audit.json').write_text(json.dumps(audit))
    source=tmp_path/'observed_supervisor.py';source.write_text('shared source')
    monkeypatch.setattr(module,'verified_collector_sources',lambda *args:{str(source):hashlib.sha256(source.read_bytes()).hexdigest()})
    calls=[]
    def verifier(outcome,evidence,raw_root,outcome_sha):
        calls.append(copy.deepcopy(evidence))
        assert evidence['artifact_sha256']['observed_supervisor_audit.json']==module.sha(root/'observed_supervisor_audit.json')
        assert raw_root==root and outcome_sha==ah
        return dict(contact_supervisor_equivalence_verified=True),{}
    monkeypatch.setattr(module,'contact_verifier',lambda:verifier)
    parent=dict(schema='wam_vla_episode_qualification_v1',qualified=True,checks={'actions_bounded':True},
        reasons=[],evidence={'artifact_sha256':{'outcome.json':ah},'source_sha256':{},
                            'label_pairs':[{'frame_index':8,'following_frame_index':9}]},
        label_diagnostics={'exclusions':[{'frame_index':7,'reason':'phase_target_or_time_boundary'}]})
    return module,root,parent,calls


def test_wrapper_preserves_parent_labels_and_adds_bound_audit(tmp_path,monkeypatch):
    module,root,parent,calls=fixture(tmp_path,monkeypatch);before=copy.deepcopy(parent)
    child=augment_qualification(parent,root,{},'parent_v2/p.qualification.json','a'*64)
    assert parent==before
    assert child['checks']['observed_contact_supervision_verified'] is True
    assert child['evidence']['label_pairs']==parent['evidence']['label_pairs']
    assert child['label_diagnostics']==parent['label_diagnostics']
    assert child['evidence']['parent_v2_qualification']['sha256']=='a'*64
    assert len(calls)==1


def test_failed_parent_remains_failed(tmp_path,monkeypatch):
    module,root,parent,_=fixture(tmp_path,monkeypatch)
    parent['qualified']=False;parent['checks']['actions_bounded']=False
    child=augment_qualification(parent,root,{},'parent_v2/p.json','a'*64)
    assert child['qualified'] is False


@pytest.mark.parametrize('change',['artifact','audit_source','audit_pass','qualified_artifact'])
def test_changed_audit_or_artifact_is_rejected(tmp_path,monkeypatch,change):
    module,root,parent,_=fixture(tmp_path,monkeypatch)
    path=root/'observed_supervisor_audit.json';audit=json.loads(path.read_text())
    if change=='artifact':(root/'outcome.json').write_text('{}')
    elif change=='audit_source':audit['audit_source_sha256']='0'*64
    elif change=='audit_pass':audit['passed']=False
    else:parent['evidence']['artifact_sha256']['outcome.json']='0'*64
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError):augment_qualification(parent,root,{},'parent_v2/p.json','a'*64)
