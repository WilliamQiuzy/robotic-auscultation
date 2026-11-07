from copy import deepcopy
import hashlib
import json
import pytest
from sim.ausc_policy import m2_given_contact_runtime as rt


def scope_fixture(tmp_path):
    from sim.ausc_policy import m2_rebuild_capture_dataset as rebuild
    split=json.loads(rt.shared.contract.IDENTITY_SPLIT_PATH.read_text())
    excluded=list(rebuild.TRAIN_IDS)+[rebuild.SELECTION_ID]+list(rebuild.FROZEN_IDS)
    pool=[i for i in split['splits']['validation'] if i not in excluded and i not in rt.shared.contract.QUARANTINED_IDENTITIES][:3]
    quality=rt.write_new(tmp_path/'quality.json',dict(eligible_shape_betas={i:[] for i in pool},
        quarantined_identities=[],identity_split_sha256=rt.shared.contract.IDENTITY_SPLIT_SHA256))
    quality=dict(file=quality['path'],sha256=quality['sha256'])
    chosen=sorted(pool,key=lambda i:hashlib.sha256(('20261106:act-recovery-new-heldout:'+i).encode()).digest())[:2]
    return dict(schema='wam_act_recovery_new_identity_cohort_v1',seed=20261106,
        cohorts=dict(train=list(rebuild.TRAIN_IDS),selection_validation=[rebuild.SELECTION_ID],frozen_evaluation=chosen),
        excluded_prior_cohort_ids=excluded,identity_split=dict(file=str(rt.shared.contract.IDENTITY_SPLIT_PATH),
            sha256=rt.shared.contract.IDENTITY_SPLIT_SHA256),identity_quality=quality,
        pose_ids=list(rebuild.POSE_IDS),target_ids=['L1','R1'],fixed_denominator=8,
        frozen_before_training=True,model_outcomes_consulted=False,timestamp_utc='2026-11-06T16:00:00Z')


def test_fresh_identity_scope_is_deterministic_and_disjoint(tmp_path):
    from sim.ausc_policy import m2_act_recovery_scope as scope
    value=scope_fixture(tmp_path);ref=rt.write_new(tmp_path/'cohort.json',value)
    got=scope.validate(ref)
    assert got==value['cohorts']


@pytest.mark.parametrize('fault',['old_id','selected_after_outcomes','training_overlap','changed_order','denominator'])
def test_fresh_scope_rejects_contamination_and_posthoc_selection(tmp_path,fault):
    from sim.ausc_policy import m2_act_recovery_scope as scope
    value=scope_fixture(tmp_path)
    if fault=='old_id':value['cohorts']['frozen_evaluation'][0]='p001030'
    elif fault=='selected_after_outcomes':value['model_outcomes_consulted']=True
    elif fault=='training_overlap':value['cohorts']['frozen_evaluation'][0]='p001120'
    elif fault=='changed_order':value['cohorts']['frozen_evaluation'].reverse()
    else:value['fixed_denominator']=7
    with pytest.raises(ValueError):scope.validate(rt.write_new(tmp_path/'bad.json',value))
