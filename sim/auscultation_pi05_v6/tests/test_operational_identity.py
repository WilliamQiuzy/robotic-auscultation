import pytest
from sim.auscultation_pi05_v6.operational_identity import qualification_id
from sim.auscultation_pi05_v6.cleanup_qualified_arrays import qualified_array_paths


def test_only_original_or_explicit_candidate_identity_allowed():
    assert qualification_id('R4_expert_observed_v1')=='R4_observed_v1'
    assert qualification_id('R5_expert_observed_redundancy30_v1')=='R5_observed_redundancy30_v1'
    for value in ('R5_expert_observed_redundancy31_v1','L5_expert_observed_redundancy30_v1','../R5_expert_observed_redundancy30_v1','R5_expert_observed_v1/','R5_expert_recipe_v1'):
        with pytest.raises(ValueError):qualification_id(value)


def test_candidate_array_cleanup_selects_only_two_exact_qualified_arrays():
    qid='R5_observed_redundancy30_v1';receipt=dict(verified_identical=True,episode_id=qid,arrays_sha256='a'*64,files_sha256={qid+'.npz':'a'*64,'parent_v2/'+qid+'.npz':'a'*64,'other.npz':'b'*64})
    assert qualified_array_paths('R5_expert_observed_redundancy30_v1',receipt)==(qid,{qid+'.npz':'a'*64,'parent_v2/'+qid+'.npz':'a'*64})
