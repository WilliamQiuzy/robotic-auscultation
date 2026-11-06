"""Real truncated teacher futures, never synthetic STOP or alignment labels."""
from copy import deepcopy
import numpy as np
import pytest

from sim.ausc_policy import m2_given_contact_act as given, m2_act as shared
from test_ausc_given_contact_capture import new_inputs


def test_truncated_real_cells_retain_last_causal_origins_and_legacy_is_unchanged(tmp_path):
    ref, records, audit, meta = new_inputs(tmp_path, 24)
    old = given.assemble_samples(records, audit, meta)
    new = given.assemble_samples(records, audit, meta, allow_truncated=True)
    assert new['schema'] == given.MASKED_DATASET_SCHEMA
    assert len(new['states']) > len(old['states'])
    np.testing.assert_array_equal(new['actions'][:len(old['states'])], old['actions'])
    lengths = (~new['action_is_pad']).sum(1)
    assert lengths[-1] == 1 and lengths[0] == 20
    assert np.all(new['actions'][new['action_is_pad']] == 0)
    assert new['derived_index']['windows'][-1]['valid_future_cells'] == 1
    start = new['derived_index']['windows'][-1]['observation_indices'][-1]
    np.testing.assert_array_equal(new['histories'][-1, 0], records[start-1]['state17'])
    assert 'action_is_pad' not in old


def test_phase_transition_truncates_prefix_without_labeling_new_phase(tmp_path):
    _, records, audit, meta = new_inputs(tmp_path, 24)
    audit[100]['controller_before']['actor']['phase'] = 'FREE'
    new = given.assemble_samples(records, audit, meta, allow_truncated=True)
    first = new['derived_index']['windows'][0]
    assert first['valid_future_cells'] == 6  # Future ticks24..95, never tick100.
    assert first['inclusive_native_ticks'][1] == 95


def test_one_real_history_is_never_padded_even_for_short_future(tmp_path):
    _, records, audit, meta = new_inputs(tmp_path, 8)
    new = given.assemble_samples(records, audit, meta, allow_truncated=True)
    assert new['derived_index']['windows'][0]['observation_indices'] == [0, 1]
    assert new['derived_index']['windows'][-1]['valid_future_cells'] == 1
    assert len(new['states']) == 6


def numerical(tmp_path):
    _, records, audit, meta = new_inputs(tmp_path, 24)
    return given.assemble_samples(records, audit, meta, allow_truncated=True)


def test_mask_excludes_unavailable_suffix_from_normalization_metrics_and_loss(tmp_path):
    data = numerical(tmp_path)
    stats = given.fit_normalization(data, split='train')
    valid = ~data['action_is_pad']
    np.testing.assert_allclose(stats['action_mean'], data['actions'][valid].mean(0))
    batch = given.pack_batch(data, stats, np.arange(len(valid)), training=False)
    np.testing.assert_array_equal(batch['action_is_pad'], ~valid)
    prediction = data['actions'].copy()
    prediction[~valid] = 1e6
    result = shared.evaluate_predictions(prediction, data)
    assert result['act']['full_chunk_mae_rad'] == 0
    assert result['valid_future_cells'] == int(valid.sum())
    assert result['per_horizon_valid_cells'] == valid.sum(0).tolist()


@pytest.mark.parametrize('fault', ['all_pad','hole','wrong_dtype','shape','nonzero_suffix','untyped_mask'])
def test_invalid_or_untyped_padding_cannot_become_teacher_labels(tmp_path, fault):
    data = numerical(tmp_path)
    if fault == 'all_pad': data['action_is_pad'][0] = True
    elif fault == 'hole': data['action_is_pad'][0, 3] = True
    elif fault == 'wrong_dtype': data['action_is_pad'] = data['action_is_pad'].astype(int)
    elif fault == 'shape': data['action_is_pad'] = data['action_is_pad'][:, :19]
    elif fault == 'nonzero_suffix': data['actions'][-1, -1] = .123
    else: data['schema'] = given.DATASET_SCHEMA
    with pytest.raises(ValueError): given.fit_normalization(data, split='train')


def test_raw_intent_corruption_is_still_a_hard_error_for_short_chunks(tmp_path):
    _, records, audit, meta = new_inputs(tmp_path, 8)
    audit[24]['raw_intent']['live_target_q'][0] += .01
    with pytest.raises(ValueError): given.assemble_samples(records, audit, meta, allow_truncated=True)


def test_masked_dataset_loader_has_explicit_version_and_seventh_array(tmp_path):
    from test_ausc_given_contact_expanded_pilot import bundle, trainer
    from sim.ausc_policy import m2_given_contact_runtime as rt
    value = bundle(tmp_path)
    value['schema'] = trainer().MASKED_PREPARATION_SCHEMA
    for split, ref in value['arrays'].items():
        with np.load(ref['path']) as archive: arrays = {k: archive[k] for k in archive.files}
        arrays['action_is_pad'] = np.zeros((len(arrays['states']),20), bool)
        path = tmp_path/(split+'-masked.npz')
        np.savez(path, **arrays)
        value['arrays'][split] = rt.reference(path)
    loaded = trainer().load_dataset(rt.write_new(tmp_path/'masked-complete.json', value))
    assert loaded['train']['schema'] == given.MASKED_DATASET_SCHEMA
    assert loaded['train']['action_is_pad'].dtype == bool


def test_recovery_short_windows_do_not_train_on_perturbation_cells(tmp_path):
    from test_ausc_given_contact_recovery import ledger
    from sim.ausc_policy import m2_given_contact_recovery_dataset as rec
    _, records, audit, meta = new_inputs(tmp_path,34)
    sidecar=ledger(audit,meta)
    samples=given.assemble_samples(records,audit,meta,allow_truncated=True)
    got=rec.filter_recovery_samples(samples,audit,meta,sidecar)
    assert len(got['states'])==24
    assert got['action_is_pad'].shape==(24,20)
    assert (~got['action_is_pad'][-1]).sum()==1
    for window in got['derived_index']['windows']:
        assert window['raw_intent_native_ticks'][0]>=120


def test_masked_export_combination_preserves_masks_and_split_ranges(tmp_path):
    from sim.ausc_policy import m2_given_contact_masked_dataset as export
    from test_ausc_given_contact_expanded_pilot import bundle
    from sim.ausc_policy import m2_given_contact_runtime as rt
    value=bundle(tmp_path)
    decoded=[]
    for split,ref in value['arrays'].items():
        for e in [r for r in value['episodes'] if r['split']==split]:
            begin,end=e['split_sample_range']
            with np.load(ref['path']) as archive:s={k:archive[k][begin:end] for k in archive.files}
            s.update(schema=given.MASKED_DATASET_SCHEMA,task_id=given.TASK_ID,
                action_semantics_id=given.ACTION_SEMANTICS_ID,observation_contract=given.observation_contract(),
                label_protocol=shared.MASKED_LABEL_PROTOCOL,action_is_pad=np.zeros((end-begin,20),bool),
                derived_index=dict(windows=[dict(valid_future_cells=20)]*(end-begin)))
            row=dict(e,target_id='L1');meta=dict(reference_sha256='a'*64,given_reference={})
            receipt=deepcopy(e);receipt['native_reset_identity']=dict(pid=100+len(decoded),sid=90,start_ticks='1000')
            decoded.append((row,(s,receipt,meta)))
    result=export.combine(decoded)
    assert result['train']['action_is_pad'].shape==(9,20)
    assert result['counts']==dict(train=9,validation=4)
    assert result['normalization']==given.fit_normalization(result['train'],split='train')
    decoded[1][1][1]['native_reset_identity']=deepcopy(decoded[0][1][1]['native_reset_identity'])
    with pytest.raises(ValueError,match='reset'):export.combine(decoded)


def test_masked_recovery_decoder_does_not_require_a_full_h20_prefix_first(tmp_path,monkeypatch):
    from sim.ausc_policy import m2_given_contact_masked_dataset as export
    from sim.ausc_policy import m2_given_contact_runtime as rt
    from sim.ausc_policy import m2_given_contact_recovery_dataset as recovery
    config=rt.write_new(tmp_path/'cfg.json',dict(m2_given_contact_recovery={}))
    expected=dict(states=np.zeros((1,17)));seen=[]
    def read(row,base,*,allow_truncated=False):
        assert allow_truncated is True
        seen.append(row)
        return expected,{},{}
    monkeypatch.setattr(recovery,'read_episode',read)
    row=dict(config=config)
    assert export.read_episode(row,tmp_path)[0] is expected
    assert seen==[row]


def test_sidecar_binding_is_to_actual_native_fresh_expert_output(tmp_path):
    from sim.ausc_policy import m2_given_contact_recovery_dataset as recovery
    path=tmp_path/'fresh_expert/given_recovery.json';path.parent.mkdir();path.write_text('{}')
    ref=dict(path=str(path),sha256='a'*64)
    recovery.require_sidecar_location(ref,tmp_path)
    with pytest.raises(ValueError):recovery.require_sidecar_location(dict(ref,path=str(tmp_path/'given_recovery.json')),tmp_path)


def test_masked_training_balances_real_near_goal_origins_without_state_noise():
    from sim.ausc_policy import m2_given_contact_expanded_pilot_train as train
    states=np.zeros((100,17));states[:,7]=1.;states[:10,7]=.03
    sample=dict(states=states,label_protocol=shared.MASKED_LABEL_PROTOCOL)
    ids=train.training_indices(np.random.default_rng(1),sample,64)
    assert len(ids)==64 and np.all((ids>=0)&(ids<100))
    assert sum(ids<10)>=32
    assert np.all(states[:10,7]==.03)
    legacy=dict(states=states)
    np.testing.assert_array_equal(train.training_indices(np.random.default_rng(1),legacy,64),
        np.random.default_rng(1).integers(100,size=64))
