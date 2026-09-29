"""Prediction metrics and controlled modality interventions, no robot execution."""
import importlib
import numpy as np
import pytest


def api():
    try:return importlib.import_module('sim.auscultation_vla_v2.evaluation')
    except ModuleNotFoundError:pytest.fail('VLA evaluation helpers missing')


def test_action_metrics_keep_translation_rotation_force_separate():
    a=np.zeros((2,10,7));b=a.copy();b[:,:,0]=.001;b[:,:,3]=.01;b[:,:,6]=.5
    r=api().action_metrics(a,b)
    assert r['mean_translation_error_mm']==pytest.approx(1)
    assert r['mean_rotation_error_rad']==pytest.approx(.01)
    assert r['mean_force_error_N']==pytest.approx(.5)
    assert r['first_action_translation_error_mm']==pytest.approx(1)
    with pytest.raises(ValueError):api().action_metrics(a,b[:,:9])


@pytest.mark.parametrize('name,indices',[
    ('hall',[20,21]),('robot',list(range(14))),
    ('depth_geometry',list(range(14,20))+list(range(22,32)))])
def test_state_intervention_changes_exact_declared_channels(name,indices):
    x=np.arange(32,dtype=float)+1;original=x.copy()
    y=api().intervene_state(x,name)
    expected=x.copy();expected[indices]=0
    np.testing.assert_array_equal(y,expected);np.testing.assert_array_equal(x,original)
    with pytest.raises(ValueError):api().intervene_state(x,'audio')


def test_phase_balanced_evaluation_selection_is_deterministic_no_duplicates():
    phases=['FREE']*30+['HOLD']*4+['RETRACT']*12
    selected=api().balanced_indices(phases,per_phase=5)
    assert len(selected)==14 and len(set(selected))==14
    assert sum(phases[i]=='HOLD' for i in selected)==4
    assert selected==api().balanced_indices(phases,per_phase=5)
    with pytest.raises(ValueError):api().balanced_indices(['UNKNOWN'],per_phase=5)


def test_inactive_force_mask_does_not_clip_motion_or_active_force():
    a=np.ones((10,7));a[:,0]=.1;a[:,6]=5.
    b=api().mask_inactive_force(a,'FREE')
    np.testing.assert_array_equal(b[:,:6],a[:,:6]);assert np.all(b[:,6]==0)
    np.testing.assert_array_equal(api().mask_inactive_force(a,'CONTACT'),a)
    assert np.all(a[:,6]==5.)
    with pytest.raises(ValueError):api().mask_inactive_force(a,'INVALID')


def test_constant_action_baseline_uses_only_training_frames():
    a=np.zeros((3,7));a[:,0]=[.001,.003,.009];a[:,6]=[0,0,4]
    episodes=[dict(split='train',actions=a,phases=np.array(['FREE','FREE','HOLD']))]
    means=api().training_phase_means(episodes)
    assert means['FREE'][0]==pytest.approx(.002)
    assert means['HOLD'][6]==4
    with pytest.raises(ValueError):
        api().training_phase_means([dict(episodes[0],split='test')])
