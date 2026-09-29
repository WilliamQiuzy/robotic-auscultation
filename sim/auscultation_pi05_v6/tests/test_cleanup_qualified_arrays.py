import pytest
from sim.auscultation_pi05_v6.cleanup_qualified_arrays import qualified_array_paths


def receipt():
    return dict(verified_identical=True,episode_id='R4_observed_v1',arrays_sha256='a'*64,
        files_sha256={'R4_observed_v1.npz':'a'*64,'parent_v2/R4_observed_v1.npz':'a'*64,'other.npz':'b'*64})


def test_only_exact_two_qualified_array_names_selected():
    qid,selected=qualified_array_paths('R4_expert_observed_v1',receipt())
    assert qid=='R4_observed_v1'
    assert list(selected)==['R4_observed_v1.npz','parent_v2/R4_observed_v1.npz']


@pytest.mark.parametrize('episode', ['../R4_expert_observed_v1','R4_expert_recipe_v1','R4_expert_observed_v1/'])
def test_unowned_episode_path_rejected(episode):
    with pytest.raises(ValueError):qualified_array_paths(episode,receipt())


def test_unverified_copy_or_different_arrays_rejected():
    record=receipt();record['verified_identical']=False
    with pytest.raises(ValueError):qualified_array_paths('R4_expert_observed_v1',record)
    record=receipt();record['files_sha256']['parent_v2/R4_observed_v1.npz']='b'*64
    with pytest.raises(ValueError):qualified_array_paths('R4_expert_observed_v1',record)
