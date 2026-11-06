"""CPU contracts for the explicitly diagnostic eight-case ACT pilot."""
from copy import deepcopy
import importlib
import json

import numpy as np
import pytest

from sim.ausc_policy import m2_given_contact_runtime as runtime


def module():
    name='sim.ausc_policy.m2_given_contact_pilot_train'
    assert importlib.util.find_spec(name) is not None, 'Eight-case pilot trainer missing'
    return importlib.import_module(name)


def bundle(tmp_path):
    ids=json.loads(runtime.shared.contract.IDENTITY_SPLIT_PATH.read_text())['splits']['train']
    ids=[value for value in ids if value not in runtime.shared.contract.QUARANTINED_IDENTITIES][:7]
    episodes=[dict(episode_id='train-'+str(i),split='train',subject_id=subject) for i,subject in enumerate(ids)]
    episodes.append(dict(episode_id='val',split='validation',subject_id='p001005'))
    manifest=runtime.write_new(tmp_path/'manifest.json',dict(schema=runtime.dataset.MANIFEST_SCHEMA,
        task_id=runtime.pose.TASK_ID,action_semantics_id=runtime.given.ACTION_SEMANTICS_ID,episodes=episodes))
    refs={};rows=[];train=None
    for split,n in [('train',3908),('validation',500)]:
        states=np.zeros((n,17));histories=np.zeros((n,2,17))
        actions=np.arange(n*20*7,dtype=float).reshape(n,20,7)*1e-7
        samples=dict(states=states,histories=histories,actions=actions,origins=np.zeros((n,7)),
            velocities=np.zeros((n,7)),observation_times=np.arange(n)*.05)
        with (tmp_path/(split+'.npz')).open('xb') as stream:np.savez(stream,**samples)
        refs[split]=runtime.reference(tmp_path/(split+'.npz'))
        available=[e for e in episodes if e['split']==split]
        rows.extend(dict(split=split,sample_index=i,episode_id=available[i%len(available)]['episode_id']) for i in range(n))
        samples.update(schema=runtime.given.DATASET_SCHEMA,task_id=runtime.pose.TASK_ID,
            action_semantics_id=runtime.given.ACTION_SEMANTICS_ID,observation_contract=runtime.pose.observation_contract())
        if split=='train':train=samples
    stats=runtime.given.fit_normalization(train,split='train')
    value=dict(schema='m2_given_eight_pilot_preparation_v1',source_manifest=manifest,arrays=refs,
        normalization=runtime.write_new(tmp_path/'normalization.json',stats),
        index=runtime.write_new(tmp_path/'index.json',dict(schema=runtime.dataset.INDEX_SCHEMA,samples=rows,
            manifest_sha256=manifest['sha256'],task_id=runtime.pose.TASK_ID,
            action_semantics_id=runtime.given.ACTION_SEMANTICS_ID)),
        task_contract=runtime.task_contract(),training_eligible=False,test_payloads_opened=False)
    return value


def test_load_preserves_raw_labels_train_normalization_and_identity_scope(tmp_path):
    m=module();value=bundle(tmp_path)
    loaded=m.load_dataset(runtime.write_new(tmp_path/'COMPLETE.json',value))
    packed=runtime.given.pack_batch(loaded['train'],loaded['normalization'],np.array([0,3907]),training=False)
    assert packed['observation.state'].shape==(2,7)
    assert packed['observation.environment_state'].shape==(2,27)
    restored=packed['action']*np.asarray(loaded['normalization']['action_std'])+np.asarray(loaded['normalization']['action_mean'])
    assert np.allclose(restored,loaded['train']['actions'][[0,3907]])
    assert loaded['scope']['validation']['subject_ids']==['p001005']
    assert loaded['scope']['train']['window_count']==3908
    assert len(loaded['scope']['train']['subject_ids'])==7


def test_pilot_rejects_formal_qualification_and_test_scope(tmp_path):
    m=module();value=bundle(tmp_path)
    value['training_eligible']=True
    with pytest.raises(ValueError,match='diagnostic'):
        m.load_dataset(runtime.write_new(tmp_path/'formal.json',value))
    value['training_eligible']=False
    manifest=runtime.read_bound(value['source_manifest']);manifest['episodes'][-1]['split']='test'
    value['source_manifest']=runtime.write_new(tmp_path/'test-manifest.json',manifest)
    with pytest.raises(ValueError,match='TRAIN or VALIDATION'):
        m.load_dataset(runtime.write_new(tmp_path/'test.json',value))


def test_pilot_validates_immutable_arrays_and_full_validation_selection(tmp_path):
    m=module();value=bundle(tmp_path)
    with open(value['arrays']['validation']['path'],'ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match='changed'):
        m.load_dataset(runtime.write_new(tmp_path/'changed.json',value))
    rows=[dict(step=500,validation={'act':{'full_chunk_mae_rad':.2}}),
          dict(step=1000,validation={'act':{'full_chunk_mae_rad':.1}}),
          dict(step=1500,validation={'act':{'full_chunk_mae_rad':.1}})]
    assert m.selected_step(rows)==1000


def test_pilot_config_is_finite_and_mapped_without_formal_changes(tmp_path):
    m=module();value=bundle(tmp_path)
    config=dict(experiment=m.EXPERIMENT,base='/workspace/wam',time_limit_seconds=7200,
        dataset=runtime.write_new(tmp_path/'COMPLETE.json',value),lerobot_root=runtime.VENDOR_ROOT,
        steps=5000,batch_size=64,eval_every=500,seed=20261026,lr=1e-4,weight_decay=1e-4,
        q_noise_std=.002,q_dropout=.05)
    m.validate_config(config)
    from sim.ausc_policy import run
    assert run.MODULES[m.EXPERIMENT]=='sim.ausc_policy.m2_given_contact_pilot_train'
    altered=deepcopy(config);altered['resume']='last_pilot.pt'
    with pytest.raises(ValueError):m.validate_config(altered)
