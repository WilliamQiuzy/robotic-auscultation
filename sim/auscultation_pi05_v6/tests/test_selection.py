from copy import deepcopy
import pytest
from sim.auscultation_pi05_v6.selection import select_checkpoint
from sim.auscultation_pi05_v6.action_contract import PHASES


def report(step,error,fraction=1.):
    return dict(selection_only=True,data_hash='same',seed=917,sampling_steps=10,per_phase=20,
        sampling_policy='fixed deployment seed per query',checkpoint='/model/'+str(step),checkpoint_step=step,
        checkpoint_metadata_sha256='a'*64,splits={'validation':dict(records=[{'frame':1}],
            first_action_bound_pass_fraction={'baseline':fraction},
            baseline_by_phase={phase:{'first_action_joint_mae_rad':error} for phase in PHASES})})


def test_ranks_first_action_error_without_awarding_physics_acceptance():
    result=select_checkpoint([report(1000,.1),report(2000,.01)],data_hash='same')
    assert result['selected']['checkpoint_step']==2000
    assert not result['closed_loop_validated'] and not result['policy_qualified']


def test_bounds_precede_small_prediction_error():
    result=select_checkpoint([report(1000,.1),report(2000,.01,.99)],data_hash='same')
    assert result['selected']['checkpoint_step']==1000


def test_rejects_test_split_and_different_validation_windows():
    candidate=report(2000,.01);candidate['splits']['test']={}
    with pytest.raises(ValueError,match='validation-only'):
        select_checkpoint([candidate],data_hash='same')
    candidate=report(2000,.01);candidate['splits']['validation']['records']=[{'frame':2}]
    with pytest.raises(ValueError,match='windows differ'):
        select_checkpoint([report(1000,.1),candidate],data_hash='same')
