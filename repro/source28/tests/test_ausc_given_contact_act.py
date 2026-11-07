import numpy as np
import pytest

from test_ausc_given_contact_pose import reference, metadata
from test_ausc_policy_m2_contract import metadata as old_metadata


def module():
    from sim.ausc_policy import m2_given_contact_act
    return m2_given_contact_act


def samples(ref):
    from sim.ausc_policy.m2_contract import encode_state
    state = encode_state(np.zeros(7), ref.pose_base, 1., metadata=metadata(ref))
    return dict(schema=module().DATASET_SCHEMA, task_id=module().TASK_ID,
        action_semantics_id=module().ACTION_SEMANTICS_ID,
        observation_contract=module().observation_contract(), states=np.tile(state, (2, 1)),
        histories=np.tile(state, (2, 2, 1)), actions=np.zeros((2, 20, 7)))


def test_new_normalization_is_distinct_and_train_only(tmp_path):
    ref = reference(tmp_path); data = samples(ref)
    stats = module().fit_normalization(data, split='train')
    assert stats['schema'] == module().NORMALIZATION_SCHEMA
    batch = module().pack_batch(data, stats, [0], training=False)
    assert batch['observation.state'].shape == (1, 7)
    assert batch['observation.environment_state'].shape == (1, 27)
    assert batch['action'].shape == (1, 20, 7)
    with pytest.raises(ValueError): module().fit_normalization(data, split='validation')
    with pytest.raises(ValueError): module().fit_normalization(dict(data, schema='m2_old_dataset'), split='train')
    from sim.ausc_policy import m2_act
    with pytest.raises(ValueError): m2_act.pack_batch(data, stats, [0], training=False)


def test_new_checkpoint_rejects_equal_dimension_old_weights(tmp_path):
    from sim.ausc_policy import m2_act
    m = module(); ref = reference(tmp_path)
    cp = dict(schema=m.CHECKPOINT_SCHEMA, upstream_commit=m2_act.UPSTREAM_COMMIT,
        action_semantics_id=m.ACTION_SEMANTICS_ID,
        model=m2_act.native_spec(), selection_split='validation', action_representation='block_relative_joint',
        state_only=True, metadata=metadata(ref), observation_contract=m.observation_contract(), cpu_fixture_only=True)
    m.validate_checkpoint(cp, allow_cpu_fixture=True)
    with pytest.raises(ValueError): m.validate_checkpoint(dict(cp, schema=m2_act.SCHEMA), allow_cpu_fixture=True)
    with pytest.raises(ValueError): m.validate_checkpoint(dict(cp, metadata=old_metadata()), allow_cpu_fixture=True)
    with pytest.raises(ValueError): m2_act.validate_checkpoint(cp, allow_cpu_fixture=True)


def test_native_production_loader_is_explicitly_not_ready():
    with pytest.raises(RuntimeError, match='bridge'):
        module().make_controller(deployment={'schema': module().DEPLOYMENT_SCHEMA})


def test_new_query_rejects_old_semantic_action_block_before_any_execution(tmp_path, monkeypatch):
    from test_ausc_policy_m2_act_controller import bare_controller, query_history
    from sim.ausc_policy import m2_contract, m2_act_controller
    ref = reference(tmp_path); c, state = bare_controller(monkeypatch)
    c.metadata = metadata(ref)
    observation = dict(source_id='given-contact-'+ref.sha256, observed_at_s=0.,
        context=('episode', 'L1', 0, 'APPROACH'), estimated_target_base=ref.pose_base, confidence=1.,
        state17=m2_contract.encode_state(np.zeros(7), ref.pose_base, 1., metadata=c.metadata))
    query_history(c, state, observation)
    c.predictor = lambda *a, **k: m2_contract.JointActionBlock('block_relative_joint',
        np.zeros((20, 7)), np.zeros(7), old_metadata())
    with pytest.raises(ValueError, match='semantic'): c._query(observation)
    assert c.prefix is None


def test_old_controller_factory_rejects_new_route_before_loading_weights(tmp_path, monkeypatch):
    from sim.ausc_policy import m2_act_controller as old
    ref = reference(tmp_path)
    monkeypatch.setattr(old.act, 'load_checkpoint', lambda *a, **k: pytest.fail('Old weights loaded for new task'))
    deployment=dict(schema=old.DEPLOYMENT_SCHEMA,checkpoint='best.pt',checkpoint_sha256='a'*64,
        lerobot_root='/official',metadata=metadata(ref),episode_id='episode')
    with pytest.raises(ValueError): old.make_controller(deployment=deployment,estimate_provider=lambda *_:None)
