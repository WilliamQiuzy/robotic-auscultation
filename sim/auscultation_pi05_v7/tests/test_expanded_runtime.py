"""CPU-only experiment guards; toy metadata never represents qualified data."""
import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

def module(name):
    return importlib.import_module('sim.auscultation_pi05_v7.expanded_'+name)

def test_fixed_plan_and_no_old_checkpoint_resume():
    contract=module('contract');plan=contract.training_plan()
    assert plan['total_steps']==12000 and plan['candidate_steps']==list(range(1000,12001,1000))
    assert (plan['batch_size'],plan['seed'],plan['learning_rate'])==(4,61,5e-5)
    assert plan['diagnostic_splits']==['train','validation']
    args=SimpleNamespace(steps=12000,batch_size=4,seed=61,learning_rate=5e-5,save_every=1000,evaluate_every=1000,stop_at=12000,resume=None)
    module('train_openpi').validate_arguments(args)
    for key,value in [('steps',4000),('seed',62),('learning_rate',1e-4),('save_every',250),('resume',Path('old3750'))]:
        altered=copy.copy(args);setattr(altered,key,value)
        with pytest.raises(ValueError):module('train_openpi').validate_arguments(altered)

def test_training_batch_rejects_test_before_any_dataset_access():
    class Forbidden:
        def __getattribute__(self,name):raise AssertionError('test data touched: '+name)
    with pytest.raises(ValueError,match='test'):
        module('training_data').make_batch(Forbidden(),'test',[0],None)

def test_diagnostic_schedule_never_visits_test_arrays():
    class TestEpisode(dict):
        def __getitem__(self,key):
            if key not in ('split','episode_id'):raise AssertionError('test array touched')
            return super().__getitem__(key)
    dataset=SimpleNamespace(episodes=[{'episode_id':'seen','split':'train','phases':['FREE']*20},TestEpisode(split='test',episode_id='heldout')],indices={'train':[(0,0),(0,10)],'validation':[],'test':[(1,0)]})
    rows=module('train_openpi').diagnostic_schedule(dataset,per_phase=2)
    assert len(rows)==1 and rows[0]['episode_index']==0

def test_missing_expanded_receipts_fail_before_frozen_loader(tmp_path,monkeypatch):
    data=module('training_data');path=tmp_path/'manifest.json';path.write_text(json.dumps({'schema':'wam_vla_training_manifest_v1','sealed':True,'episodes':[]}))
    monkeypatch.setattr(data.legacy,'load_dataset',lambda _:(_ for _ in ()).throw(AssertionError('old loader reached')))
    with pytest.raises(ValueError,match='expanded_experiment'):data.load_dataset(path)

def test_actor_configuration_refuses_original_trainer_hash():
    actor=module('actor')
    with pytest.raises(ValueError,match='expanded'):
        actor.verify_configuration(SimpleNamespace(),{'trainer_sha256':'a'*64})

def test_service_requires_new_experiment_fields():
    from sim.auscultation_pi05_v7.tests.test_runtime_binding import binding
    client=module('policy_client')
    with pytest.raises(ValueError,match='experiment'):
        client.verify_policy_binding(binding(),'a'*64)


def complete_binding():
    from sim.auscultation_pi05_v7.tests.test_runtime_binding import binding
    info=binding();contract=module('contract')
    info['experiment_binding']=contract.experiment_binding('1'*64,'2'*64,'3'*64)
    info['experiment_source_sha256']={name:'4'*64 for name in contract.experiment_source_paths()}
    return info

@pytest.mark.parametrize('change',['source_missing','source_extra','source_zero','plan','manifest','missing_sidecar'])
def test_expanded_wire_rejects_incomplete_or_changed_protocol(change):
    info=complete_binding();client=module('policy_client')
    client.verify_policy_binding(info,'a'*64)
    if change=='source_missing':info['experiment_source_sha256'].pop(next(iter(info['experiment_source_sha256'])))
    elif change=='source_extra':info['experiment_source_sha256']['v7/not_reviewed.py']='5'*64
    elif change=='source_zero':info['experiment_source_sha256'][next(iter(info['experiment_source_sha256']))]='0'*64
    elif change=='plan':info['experiment_binding']['training_plan']['total_steps']=4000
    elif change=='manifest':info['experiment_binding']['manifest_sha256']='not-hash'
    elif change=='missing_sidecar':info['experiment_binding'].pop('experiment_sha256')
    with pytest.raises(ValueError):client.verify_policy_binding(info,'a'*64)


def test_reply_cannot_drop_experiment_binding_after_valid_health(tmp_path,monkeypatch):
    import io
    client=module('policy_client');info=complete_binding()
    config=dict(token='synthetic-unit-test-only',client_port=1,**{'expected_'+k:info[k] for k in client.CONFIG_FIELDS})
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    def response(request,timeout):
        reply=copy.deepcopy(info)
        if request.data:
            reply['request_id']=json.loads(request.data)['request_id'];reply.pop('experiment_binding')
        return io.BytesIO(json.dumps(reply).encode())
    monkeypatch.setattr(client.urllib.request,'urlopen',response)
    actor=client.PolicyClient(path)
    import numpy as np
    with pytest.raises(ValueError,match='binding changed: experiment_binding'):
        actor.predict(np.zeros((3,4,3),dtype=np.uint8),np.zeros(41),'FREE','R1')


def test_actual_configuration_requires_truthful_trainer_source(monkeypatch):
    actor=module('actor');contract=module('contract')
    from sim.auscultation_pi05_v6 import train_openpi as parent
    from sim.auscultation_pi05_v6.task_observation import observation_binding
    from sim.auscultation_pi05_v7.action_encoding import action_encoding_binding
    binding=contract.experiment_binding('1'*64,'2'*64,'3'*64)
    sources=contract.experiment_source_hashes()
    dataset=SimpleNamespace(experiment_binding=binding,experiment_source_sha256=sources,
        parent=SimpleNamespace(data_hash='4'*64),dynamics={'contract_sha256':'5'*64,'recipe_sha256':'6'*64,'execution_contract_sha256':'7'*64})
    config=dict(recipe=actor.RECIPE,training_run_path='/explicit-cpu-fixture/run',experiment_binding=binding,experiment_source_sha256=sources,
        action_dim=7,action_horizon=10,state_width=41,max_token_len=256,source_sha256=actor.core_sources(),
        openpi_commit=parent.OPENPI_COMMIT,parent_data_hash='4'*64,action_semantics='absolute_joint_command_v1',
        model_action_semantics='future_command_minus_fixed_observation_measured_q',canonical_phase_instructions=dict(parent.INSTRUCTIONS),
        instruction_policy='canonical-phase-and-target-v1',observation_contract=observation_binding(),action_encoding_binding=action_encoding_binding(),
        dynamics_contract_sha256='5'*64,dynamics_recipe_sha256='6'*64,execution_contract_sha256='7'*64,
        trainer_sha256=contract.file_sha(Path(actor.__file__).with_name('expanded_train_openpi.py')))
    config.update({k:contract.training_plan()[k] for k in ('seed','batch_size','learning_rate','total_steps','save_every','evaluate_every')})
    actor.verify_configuration(dataset,config)
    config['trainer_sha256']=contract.file_sha(Path(actor.__file__).with_name('train_openpi.py'))
    with pytest.raises(ValueError,match='Expanded training source changed'):actor.verify_configuration(dataset,config)


def test_empty_phase_bin_in_one_episode_is_not_hidden_by_other_episodes():
    data=module('training_data');phases=module('contract').PHASES
    episodes=[];indices={'train':[],'validation':[],'test':[]}
    for ei in range(50):
        split='train' if ei<30 else 'validation' if ei<40 else 'test'
        own=list(phases) if ei!=39 else list(phases[:-1])
        episodes.append(dict(episode_id='toy'+str(ei),split=split,phases=own))
        indices[split].extend((ei,s) for s in range(len(own)))
    with pytest.raises(ValueError,match='toy39'):data.verify_windows(SimpleNamespace(episodes=episodes,indices=indices))


def test_periodic_diagnostic_prediction_callback_never_receives_test(tmp_path,monkeypatch):
    import sys
    import types
    import numpy as np
    trainer=module('train_openpi')
    monkeypatch.setitem(sys.modules,'jax',SimpleNamespace(device_put=lambda x:x,random=SimpleNamespace(key=lambda x:x)))
    models=types.ModuleType('openpi.models');models.model=SimpleNamespace(Observation=lambda **x:x)
    monkeypatch.setitem(sys.modules,'openpi',types.ModuleType('openpi'))
    monkeypatch.setitem(sys.modules,'openpi.models',models)
    phases=np.array(['FREE']*12);q=np.array([.1,-.2,.2,1.,-.1,.2,.1])
    episodes=[dict(episode_id=name,split=split,phases=phases,state=np.tile(np.r_[q,np.zeros(34)],(12,1)),
        actions=np.tile(q,(12,1)),targets=np.array(['R1']*12)) for name,split in [('seen','train'),('validation','validation'),('never','test')]]
    dataset=SimpleNamespace(episodes=episodes,indices={'train':[(0,0)],'validation':[(1,0)],'test':[(2,0)]},
        data_hash='1'*64,experiment_binding={},experiment_source_sha256={})
    calls=[]
    def batch(d,split,indices,tokenizer):
        assert split!='test';calls.append(split);return {'split':split},None
    monkeypatch.setattr(trainer,'make_batch',batch)
    monkeypatch.setattr(trainer,'decode_prediction',lambda d,n,state,phase:np.tile(q,(10,1)))
    report=trainer.sampled_diagnostic(dataset,dict(tokenizer=None,trainable=None,frozen=None),
        lambda p,f,obs,seed:np.zeros((1,10,7)),1000,tmp_path/'diagnostic')
    assert calls==['train','validation']
    assert set(report['episodes'])=={'seen','validation'}
    with np.load(tmp_path/'diagnostic/predictions.npz') as saved:assert not any('never' in k for k in saved.files)


def test_protocol_sidecar_cannot_be_published_with_incomplete_real_evidence(tmp_path):
    data=module('training_data');manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    for name in ('protocol','preparation','assembly','helper','packets'):(tmp_path/(name+'.json')).write_text('{}')
    out=tmp_path/'must_not_exist.json'
    with pytest.raises(ValueError):data.prepare_experiment(manifest,tmp_path/'protocol.json',tmp_path/'preparation.json',
        tmp_path/'assembly.json',tmp_path/'helper.json',out)
    assert not out.exists()


def test_same_run_resume_is_latest_complete_only_and_rng_sequence_exact(tmp_path):
    import numpy as np
    from sim.auscultation_pi05_v6 import train_openpi as frozen
    trainer=module('train_openpi');run=tmp_path/'run';run.mkdir()
    config={'toy':'explicit CPU numeric checkpoint fixture','training_run_path':str(run.resolve())};normalization={'toy':[1,2]}
    rng=np.random.default_rng(61);first=rng.integers(0,500,23)
    checkpoint=frozen.save_checkpoint(run/'recovery',step=23,arrays={'weight':np.array([.25]),'optimizer':np.array([.75]),'rng':np.array([61])},
        metadata=dict(data_hash='d'*64,config=config,normalization=normalization,sampler_rng_state=rng.bit_generator.state))
    uninterrupted=rng.integers(0,500,100)
    chosen=trainer.resolve_same_run_resume(run,checkpoint,data_hash='d'*64,config=config,normalization=normalization)
    arrays,meta=frozen.load_checkpoint(chosen,data_hash='d'*64,config=config,normalization=normalization)
    restored=np.random.default_rng();restored.bit_generator.state=meta['sampler_rng_state']
    np.testing.assert_array_equal(uninterrupted,restored.integers(0,500,100))
    np.testing.assert_array_equal(arrays['optimizer'],[.75])
    latest=frozen.save_checkpoint(run/'recovery',step=24,arrays={'weight':np.array([.3])},metadata=dict(data_hash='d'*64,config=config,normalization=normalization))
    with pytest.raises(ValueError,match='latest'):trainer.resolve_same_run_resume(run,checkpoint,data_hash='d'*64,config=config,normalization=normalization)
    (run/'SELECTION_LOCKED.json').write_text('{}')
    with pytest.raises(ValueError,match='locked'):trainer.resolve_same_run_resume(run,latest,data_hash='d'*64,config=config,normalization=normalization)


def test_resume_rejects_other_run_and_corrupt_latest_without_falling_back(tmp_path):
    import numpy as np
    from sim.auscultation_pi05_v6 import train_openpi as frozen
    trainer=module('train_openpi');run=tmp_path/'run';run.mkdir()
    settings=dict(data_hash='d'*64,config={'training_run_path':str(run.resolve())},normalization={})
    old=frozen.save_checkpoint(tmp_path/'other/model',step=10,arrays={'x':np.ones(1)},metadata=settings)
    with pytest.raises(ValueError,match='same run'):trainer.resolve_same_run_resume(run,old,**settings)
    own=frozen.save_checkpoint(run/'recovery',step=20,arrays={'x':np.ones(1)},metadata=settings)
    (own/'arrays.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash mismatch'):trainer.resolve_same_run_resume(run,own,**settings)


def test_resume_numerical_optimizer_and_sampler_suffix_matches_uninterrupted(tmp_path):
    import numpy as np
    from sim.auscultation_pi05_v6 import train_openpi as frozen
    # Explicit CPU state-serialization fixture; no model/qualification claim.
    def advance(weight,momentum,rng,n):
        chosen=[]
        for _ in range(n):
            x=int(rng.integers(0,50));chosen.append(x)
            gradient=weight-(x/50.)
            momentum=.9*momentum+.1*gradient;weight=weight-5e-5*momentum
        return weight,momentum,chosen
    seed=np.random.default_rng(61);w,m,_=advance(np.array([.17]),np.array([0.]),seed,37)
    metadata=dict(data_hash='a'*64,config={'fixture':'toy CPU optimizer','training_run_path':str((tmp_path/'run').resolve())},normalization={},sampler_rng_state=copy.deepcopy(seed.bit_generator.state))
    run=tmp_path/'run';checkpoint=frozen.save_checkpoint(run/'recovery',step=37,arrays={'w':w,'m':m},metadata=metadata)
    expected_w,expected_m,expected_rows=advance(w.copy(),m.copy(),seed,81)
    chosen=module('train_openpi').resolve_same_run_resume(run,checkpoint,data_hash=metadata['data_hash'],config=metadata['config'],normalization={})
    arrays,saved=frozen.load_checkpoint(chosen,data_hash=metadata['data_hash'],config=metadata['config'],normalization={})
    recovered=np.random.default_rng();recovered.bit_generator.state=saved['sampler_rng_state']
    actual_w,actual_m,actual_rows=advance(arrays['w'],arrays['m'],recovered,81)
    assert actual_rows==expected_rows
    np.testing.assert_array_equal(actual_w,expected_w);np.testing.assert_array_equal(actual_m,expected_m)

@pytest.mark.parametrize('protected',['dataset','evidence','protocol','helper'])
def test_experiment_sidecar_cannot_add_files_to_original_evidence(tmp_path,monkeypatch,protected):
    data=module('training_data');paths={}
    for name in ('dataset','evidence','protocol','helper'):
        folder=tmp_path/name;folder.mkdir();paths[name]=folder
    args=dict(manifest=paths['dataset']/'manifest.json',protocol=paths['protocol']/'protocol.json',
        preparation=paths['evidence']/'preparation.json',assembly_receipt=paths['dataset']/'assembly_receipt.json',
        protocol_helper=paths['helper']/'protocol.py')
    for path in args.values():path.write_text('{}')
    (paths['evidence']/'packets.json').write_text('[]')
    # Isolate write protection from heavy qualification. This is no actual receipt.
    monkeypatch.setattr(data,'verify_metadata',lambda *args:None)
    monkeypatch.setattr(data.legacy,'load_dataset',lambda _:SimpleNamespace(parent=SimpleNamespace(data_hash=None)))
    monkeypatch.setattr(data,'verify_windows',lambda _:None)
    output=paths[protected]/'new_sidecar.json'
    with pytest.raises(ValueError,match='overlaps'):data.prepare_experiment(**args,output=output)
    assert not output.exists()


def test_copied_run_cannot_impersonate_original_resume_location(tmp_path):
    import shutil
    import numpy as np
    from sim.auscultation_pi05_v6 import train_openpi as frozen
    trainer=module('train_openpi');original=tmp_path/'original';copy_run=tmp_path/'copy'
    settings=dict(data_hash='d'*64,config={'training_run_path':str(original.resolve())},normalization={})
    saved=frozen.save_checkpoint(original/'recovery',step=20,arrays={'x':np.ones(1)},metadata=settings)
    shutil.copytree(original,copy_run)
    with pytest.raises(ValueError,match='original run path'):
        trainer.resolve_same_run_resume(copy_run,copy_run/'recovery'/saved.name,**settings)


def test_experiment_sidecar_cannot_add_to_indirect_raw_source_tree(tmp_path,monkeypatch):
    data=module('training_data');direct=tmp_path/'direct';raw=tmp_path/'raw_episode';direct.mkdir();raw.mkdir()
    args={name:direct/(name+'.json') for name in ('manifest','protocol','preparation','assembly_receipt','protocol_helper')}
    for path in args.values():path.write_text('{}')
    (direct/'packets.json').write_text('[]');source=raw/'outcome.json';source.write_text('{}')
    receipt={'source_sha256':{str(source):'explicit-toy-hash'},'data_hash':None}
    monkeypatch.setattr(data,'verify_metadata',lambda *args:({}, {}, receipt))
    monkeypatch.setattr(data,'read_json',lambda _:receipt)
    monkeypatch.setattr(data.legacy,'load_dataset',lambda _:SimpleNamespace(parent=SimpleNamespace(data_hash=None)))
    monkeypatch.setattr(data,'verify_windows',lambda _:None)
    output=raw/'new_experiment.json'
    with pytest.raises(ValueError,match='overlaps'):data.prepare_experiment(**args,output=output)
    assert not output.exists()
