"""Physical labels stay unchanged while the flow operates in logit coordinates."""
import copy
import numpy as np
import pytest

from sim.auscultation_pi05_v6.action_contract import LOWER,UPPER,PHASES,bounded
from sim.auscultation_pi05_v6.action_encoding import action_encoding_binding,inspect_support
from sim.auscultation_pi05_v6.training_data import (
    physical_codec,fit_normalization,action_stats,normalize,denormalize)


def episode():
    phases=np.repeat(sorted(PHASES),8)
    fraction=np.linspace(.15,.85,len(phases))[:,None]
    commands=(np.asarray(LOWER)+(np.asarray(UPPER)-LOWER)*fraction).astype(np.float32)
    codec=physical_codec();state=np.repeat(((codec.low+codec.high)/2)[None],len(phases),axis=0)
    return dict(episode_id='acquired_train',split='train',phases=phases,actions=commands,state=state)


def test_phase_logit_scaling_preserves_original_physical_labels():
    data=episode();original=data['actions'].copy();normalization=fit_normalization([data])
    assert normalization['action_encoding_binding']==action_encoding_binding()
    for phase in PHASES:
        labels=data['actions'][data['phases']==phase]
        stats=action_stats(normalization,phase)
        restored=denormalize(normalize(labels,stats),stats)
        np.testing.assert_allclose(restored,labels,atol=2e-6,rtol=0)
        assert bounded(restored,phase)
    np.testing.assert_array_equal(data['actions'],original)
    inspect_support(original)


def test_finite_extreme_flow_endpoints_stay_in_true_urdf_bounds():
    normalization=fit_normalization([episode()]);stats=action_stats(normalization,'FREE')
    q=denormalize(np.asarray([[-1e6]*7,[1e6]*7],np.float32),stats)
    assert bounded(q,'FREE')


def test_changed_action_encoding_binding_rejected_before_inference():
    normalization=fit_normalization([episode()]);wrong=copy.deepcopy(normalization)
    wrong['action_encoding_binding']['schema']='unbound-decoder'
    with pytest.raises(ValueError,match='binding'):action_stats(wrong,'FREE')
    stats=copy.deepcopy(action_stats(normalization,'FREE'))
    stats['action_encoding']['schema']='unbound-decoder'
    with pytest.raises(ValueError,match='encoding'):normalize(episode()['actions'],stats)
    with pytest.raises(ValueError,match='encoding'):denormalize(np.zeros((10,7)),stats)


def test_validation_labels_cannot_fit_action_scaling():
    data=episode();data['split']='validation'
    with pytest.raises(ValueError,match='training episodes only'):fit_normalization([data])


def test_physical_boundary_label_is_rejected_without_silent_epsilon():
    data=episode();data['actions'][0,3]=0
    with pytest.raises(ValueError):fit_normalization([data])

