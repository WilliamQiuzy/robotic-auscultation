"""Training split and temporal contracts; these fixtures are not robot demos."""
import importlib
import numpy as np
import pytest


def api():
    try:
        module = importlib.import_module('sim.auscultation_vla_v2.training_data')
    except ModuleNotFoundError:
        module = None
    assert module is not None, 'multimodal training data contract is missing'
    return module


def entry(name, split, **groups):
    return dict(episode_id=name, split=split, body_source='same_body',
                layout_family=groups.get('layout_family',name),
                motion_family=groups.get('motion_family',name))


def test_groups_cannot_cross_a_declared_heldout_axis():
    m=api(); rows=[entry('one','train'),entry('two','validation',layout_family='one')]
    with pytest.raises(ValueError,match='layout_family'):
        m.validate_groups(rows,heldout_axes=('layout_family','motion_family'))


def test_same_body_does_not_get_reported_as_unseen_people():
    m=api(); rows=[entry('one','train'),entry('two','validation')]
    report=m.validate_groups(rows,heldout_axes=('layout_family','motion_family'))
    assert report['unseen_body_evaluation'] is False
    with pytest.raises(ValueError,match='body_source'):
        m.validate_groups(rows,heldout_axes=('body_source',))


@pytest.mark.parametrize('mutation',[
    lambda r:r[1].update(episode_id='one'),
    lambda r:r[1].update(split='random_frames'),
    lambda r:r[1].update(motion_family=''),
])
def test_invalid_group_registry_rejected(mutation):
    m=api(); rows=[entry('one','train'),entry('two','validation')];mutation(rows)
    with pytest.raises(ValueError):m.validate_groups(rows,heldout_axes=('layout_family',))


def test_normalization_cannot_fit_validation_or_test_frames():
    m=api(); x=np.arange(640,dtype=float).reshape(20,32);a=np.ones((20,7))
    with pytest.raises(ValueError,match='train'):
        m.fit_normalization([dict(split='validation',state=x,actions=a,episode_id='v')])
    stats=m.fit_normalization([dict(split='train',state=x,actions=a,episode_id='t')])
    assert stats['fit_episode_ids']==['t']
    assert len(stats['state']['low'])==32 and len(stats['actions']['low'])==7
    assert np.isfinite(m.normalize(x,stats['state'])).all()
    assert np.allclose(m.denormalize(m.normalize(a,stats['actions']),stats['actions']),a)


def test_constant_training_channel_does_not_amplify_a_tiny_noise_spike():
    m=api();state=np.zeros((20,32));actions=np.zeros((20,7))
    stats=m.fit_normalization([dict(split='train',state=state,actions=actions,episode_id='t')])
    changed=state.copy();changed[:,20]=.001
    assert np.max(np.abs(m.normalize(changed,stats['state'])))<.1


def test_chunks_use_only_valid_future_actions_from_same_target_and_phase():
    m=api();t=np.arange(32)*.05;phase=np.array(['APPROACH']*16+['CONTACT']*16)
    target=np.array(['R1']*32);actions=np.arange(224,dtype=float).reshape(32,7)
    chunks=m.full_chunks(t,phase,target,actions,horizon=10,period_s=.05)
    assert [c['start'] for c in chunks]==list(range(7))+list(range(16,23))
    assert np.array_equal(chunks[1]['actions'],actions[1:11])
    assert all(c['actions'].shape==(10,7) for c in chunks)


def test_chunks_never_pad_or_bridge_dropped_frames_or_target_changes():
    m=api();t=np.arange(24)*.05;t[12:]+=.05
    phases=['HOLD']*24;target=['R1']*6+['L1']*18
    chunks=m.full_chunks(t,phases,target,np.zeros((24,7)),horizon=10,period_s=.05)
    assert [c['start'] for c in chunks]==[12,13,14]
    with pytest.raises(ValueError):m.full_chunks(t,phases,target,np.zeros((24,32)),horizon=10)


@pytest.mark.parametrize('origin_s',[0.,100.])
def test_chunks_reject_float32_timestamps_before_silently_discarding_valid_windows(origin_s):
    m=api();times=(origin_s+np.arange(200)*.05).astype(np.float32)
    with pytest.raises(ValueError,match='float64'):
        m.full_chunks(times,['HOLD']*200,['R1']*200,np.zeros((200,7)))


@pytest.mark.parametrize('origin_s',[0.,100.,1_000_000.])
@pytest.mark.parametrize('dropped_frame',[False,True])
def test_float64_timestamps_preserve_all_contiguous_windows_and_reject_real_gaps(origin_s,dropped_frame):
    m=api();times=origin_s+np.arange(200,dtype=np.float64)*.05
    if dropped_frame:times[100:]+=.05
    chunks=m.full_chunks(times,['HOLD']*200,['R1']*200,np.zeros((200,7)))
    expected=list(range(91))+list(range(100,191)) if dropped_frame else list(range(191))
    assert [chunk['start'] for chunk in chunks]==expected


def test_modality_declaration_must_include_measured_depth_hall_and_robot_inputs():
    m=api(); valid={'rgb':'camera','depth':'camera_metric_optical_z','hall':'adc_voltage',
        'robot':'measured_q_dq_fk','target':'rgbd_surface_tracking','audio':'absent'}
    m.validate_modalities(valid)
    for key,value in [('depth','not_used'),('hall','physx_contact_force'),('target','isaac_ground_truth'),('audio','synthetic')]:
        bad={**valid,key:value}
        with pytest.raises(ValueError):m.validate_modalities(bad)
