"""Variable-size diagnostic training contracts and actual native dispatch."""
from copy import deepcopy
import importlib
import importlib.util
import json

import numpy as np
import pytest

from sim.ausc_policy import m2_given_contact_runtime as rt
from test_ausc_given_contact_pose import reference, metadata


def trainer():
    name='sim.ausc_policy.m2_given_contact_expanded_pilot_train'
    assert importlib.util.find_spec(name) is not None, 'Expanded diagnostic trainer missing'
    return importlib.import_module(name)


def deployment():
    name='sim.ausc_policy.m2_given_contact_expanded_pilot_deployment'
    assert importlib.util.find_spec(name) is not None, 'Expanded diagnostic deployment missing'
    return importlib.import_module(name)


def bundle(tmp_path):
    splits=json.loads(rt.shared.contract.IDENTITY_SPLIT_PATH.read_text())['splits']
    selected={s:[x for x in splits[s] if x not in rt.shared.contract.QUARANTINED_IDENTITIES][:n]
              for s,n in [('train',3),('validation',2)]}
    episodes=[];rows=[];refs={};counts={'train':9,'validation':4};train=None
    for split,count in counts.items():
        cursor=0
        for i,subject in enumerate(selected[split]):
            n=count//len(selected[split]);episode_id=split+'-'+str(i)
            episodes.append(dict(episode_id=episode_id,subject_id=subject,split=split,target_id='L1',
                windows=n,split_sample_range=[cursor,cursor+n]))
            rows.extend(dict(episode_id=episode_id,split=split,sample_index=j) for j in range(cursor,cursor+n))
            cursor+=n
        states=np.zeros((count,17));states[:,0]=np.arange(count)*.01
        histories=np.stack([states,states],axis=1)
        samples=dict(states=states,histories=histories,actions=np.zeros((count,20,7)),
            origins=states[:,:7].copy(),velocities=np.zeros((count,7)),observation_times=np.arange(count)*.05)
        path=tmp_path/(split+'.npz')
        with path.open('xb') as stream:np.savez(stream,**samples)
        refs[split]=rt.reference(path)
        samples.update(schema=rt.given.DATASET_SCHEMA,task_id=rt.pose.TASK_ID,
            action_semantics_id=rt.given.ACTION_SEMANTICS_ID,observation_contract=rt.pose.observation_contract())
        if split=='train':train=samples
    value=dict(schema='m2_given_diverse_pilot_preparation_v1',complete=True,counts=counts,episodes=episodes,
        split_subject_ids=selected,arrays=refs,task_contract=rt.task_contract(),fixed_denominator=40,
        index=rt.write_new(tmp_path/'index.json',dict(schema=rt.dataset.INDEX_SCHEMA,samples=rows,
            task_id=rt.pose.TASK_ID,action_semantics_id=rt.given.ACTION_SEMANTICS_ID)),
        normalization=rt.write_new(tmp_path/'normalization.json',rt.given.fit_normalization(train,split='train')),
        diagnostic_only=True,formal_acceptance=False,training_eligible=False,test_payloads_opened=False)
    return value


def test_variable_counts_keep_raw_h20_h2_and_disjoint_train_val(tmp_path):
    m=trainer();value=bundle(tmp_path)
    loaded=m.load_dataset(rt.write_new(tmp_path/'COMPLETE.json',value))
    assert loaded['train']['actions'].shape==(9,20,7)
    assert loaded['validation']['histories'].shape==(4,2,17)
    assert loaded['scope']['train']['episode_count']==3
    assert loaded['scope']['validation']['window_count']==4
    assert loaded['scope']['validation']['subject_ids']==sorted(value['split_subject_ids']['validation'])
    packed=rt.given.pack_batch(loaded['train'],loaded['normalization'],np.array([0,8]),training=False)
    assert packed['observation.state'].shape==(2,7)
    assert packed['observation.environment_state'].shape==(2,27)


@pytest.mark.parametrize('alter',[
    lambda v:v.update(schema='m2_given_eight_pilot_preparation_v1'),
    lambda v:v['counts'].update(train=8),
    lambda v:v['episodes'][0].update(split='test'),
    lambda v:v['episodes'][0].update(windows=0),
    lambda v:v['episodes'][0].update(subject_id=v['episodes'][-1]['subject_id']),
    lambda v:v['split_subject_ids']['validation'].append(v['episodes'][0]['subject_id']),
    lambda v:v.update(training_eligible=True),
])
def test_expanded_rejects_count_identity_scope_and_schema_changes(tmp_path,alter):
    m=trainer();value=bundle(tmp_path);alter(value)
    with pytest.raises(ValueError):m.load_dataset(rt.write_new(tmp_path/'bad.json',value))


def test_expanded_config_has_zero_q_augmentation_configurable_finite_steps_and_dispatch(tmp_path):
    m=trainer();config=dict(m.SCHEDULE,base='/workspace/wam',dataset=rt.write_new(tmp_path/'data.json',bundle(tmp_path)))
    assert config['steps']==20000 and config['q_noise_std']==config['q_dropout']==0
    assert rt.shared.native_spec()['dropout']==.1
    assert m.validate_config(dict(config,steps=1200,eval_every=500))['steps']==1200
    assert m.evaluation_steps(dict(config,steps=1200,eval_every=500))==[500,1000,1200]
    from sim.ausc_policy import run
    assert run.MODULES[m.EXPERIMENT]=='sim.ausc_policy.m2_given_contact_expanded_pilot_train'
    for extra in ({'resume':'last.pt'},{'q_noise_std':.002},{'q_dropout':.05},{'steps':True},{'steps':0}):
        with pytest.raises(ValueError):m.validate_config(dict(config,**extra))


def test_expanded_protocol_is_distinct_and_native_router_selects_it(monkeypatch,tmp_path):
    m=deployment()
    from sim.ausc_policy import fresh_expert, m2_given_contact_pilot_deployment as old
    marker=object();monkeypatch.setattr(m,'run_native_probe',lambda *args:marker)
    assert fresh_expert.run_probe(None,None,None,{m.ROUTE_KEY:{}},None) is marker
    ref=reference(tmp_path);meta=dict(metadata(ref),target_id='L1')
    q=np.zeros(7);state=rt.shared.contract.encode_state(q,ref.pose_base,1.,metadata=meta)
    dep=dict(metadata=meta,episode_id='expanded',source_episode_id='source',
        model={'checkpoint':{'sha256':'a'*64}},source_config={'sha256':'b'*64})
    request=m.make_request(dep,1,q,ref.pose_base,1.,metadata=meta,history=np.stack([state,state]),
        history_times=[0.,.05],context=dict(observed_at_s=.05,physics_row=12,
            context=['expanded','L1',0,'FREE'],source_id='given-contact-'+ref.sha256))
    assert request['schema']==m.PROTOCOL_SCHEMA!=old.PROTOCOL_SCHEMA
    m.validate_request(request,dep,sequence=1)
    with pytest.raises(ValueError):old.validate_request(request,dep,sequence=1)
    block=m.decode_response(m.response_for(request,np.zeros((20,7))),request)
    assert block.metadata['checkpoint_policy']==trainer().PILOT_SCHEMA
    from sim.ausc_policy.m2_act_controller import FixedOriginPrefix
    prefix=FixedOriginPrefix(block,observed_q=q,context=('expanded','L1',0,'FREE'),stamp=.05)
    for i in range(10):np.testing.assert_array_equal(prefix.pop(.05+i*.05,prefix.context),q)
    assert prefix.remaining==0


def test_expanded_derivation_preserves_physical_config_and_original_pilot_route(tmp_path):
    m=deployment();meta=dict(metadata(reference(tmp_path)),subject_id='p000449',split='train',
        episode_id='source',target_id='L1')
    original=dict(physics={'position_iterations':64},fresh_expert={'force_feedback':{'frozen':True},
        'raw_multimodal_capture':{'enabled':True}},m2_given_contact_control={'metadata':meta,'episode_id':'source'})
    frozen=deepcopy(original)
    derived=m.derive_config(original,source_config={'sha256':'b'*64},model={'checkpoint':{'sha256':'a'*64}},
        run_id='expanded',source_inventory={},scope={'train':{'subject_ids':['p000449'],'episode_ids':['source']}})
    assert original==frozen and derived['physics']==original['physics']
    assert derived['fresh_expert']['force_feedback']==original['fresh_expert']['force_feedback']
    assert m.ROUTE_KEY in derived and 'm2_given_contact_pilot_deployment' not in derived
    assert derived[m.ROUTE_KEY]['schema']==m.DEPLOYMENT_SCHEMA


def trained_run(tmp_path):
    m=trainer();prepared=m.load_dataset(rt.write_new(tmp_path/'dataset.json',bundle(tmp_path)))
    dataset=rt.reference(tmp_path/'dataset.json')
    config=dict(m.SCHEDULE,base='/workspace/wam',dataset=dataset,steps=1200,eval_every=500)
    inputs=dict(config=rt.write_new(tmp_path/'config.json',config),dataset=dataset,code=m.code_binding())
    weights=tmp_path/'weights.pt';weights.write_bytes(b'weights hash checked without deserializing')
    best=rt.reference(weights)
    metrics={s:rt.metrics(np.zeros_like(prepared[s]['actions']),prepared[s]) for s in ('train','validation')}
    evaluations=[dict(step=s,**deepcopy(metrics)) for s in [500,1000,1200]]
    selection=dict(schema=m.SELECTION_SCHEMA,inputs=inputs,best=best,selected_step=500,
        protocol=m.training_protocol(config),window_order=prepared['window_order'],scope=prepared['scope'],**m.FLAGS)
    cpu=rt.write_new(tmp_path/'cpu.json',dict(schema=m.CPU_SCHEMA,passed=True,checkpoint=best,selected_step=500,
        scope=prepared['scope'],selection=rt.write_new(tmp_path/'selected.json',selection),
        comparisons={s:{'passed':True} for s in ('train','validation')},**m.FLAGS))
    complete=dict(selection,schema=m.PILOT_SCHEMA,complete=True,completed_steps=1200,scope=prepared['scope'],
        task_contract=rt.task_contract(),model=rt.shared.native_spec(),schedule=config,evaluations=evaluations,
        selected_metrics=metrics,cpu_equivalence=cpu)
    return complete


def test_expanded_selected_binding_requires_full_val_earliest_minimum_cpu_receipt_and_actual_weights(tmp_path):
    m=deployment();complete=trained_run(tmp_path)
    bound=rt.write_new(tmp_path/'COMPLETE.json',complete)
    model,_=m.model_binding(bound)
    assert model['checkpoint']==complete['best']
    for altered in (dict(complete,selected_step=1000),dict(complete,completed_steps=5000),
                    dict(complete,schema='m2_given_contact_act_pilot_v1'),dict(complete,test_payloads_opened=True)):
        with pytest.raises(ValueError):m.validate_complete(altered)
    altered=deepcopy(complete);altered['evaluations'][0]['validation']['sample_count']-=1
    with pytest.raises(ValueError,match='population'):m.validate_complete(altered)
    cpu=rt.read_bound(complete['cpu_equivalence']);cpu['passed']=False
    altered=dict(complete,cpu_equivalence=rt.write_new(tmp_path/'cpu-failed.json',cpu))
    with pytest.raises(ValueError,match='CPU'):m.model_binding(rt.write_new(tmp_path/'bad-complete.json',altered))
    with open(complete['best']['path'],'ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match='changed'):m.model_binding(bound)


def test_expanded_reuses_original_controller_and_bridge_with_explicit_adapter(monkeypatch):
    m=deployment()
    from sim.ausc_policy import m2_given_contact_pilot_deployment as old
    observed={};sentinel=object()
    class Env:
        rows=[]
        def state(self):return dict(q=np.zeros(7),time_s=0.)
    class Bridge:
        runtime={'cpu':True}
        def __init__(self,dep,*,output,adapter):observed.update(adapter=adapter,dep=dep)
        def close(self):observed['closed']=True
    def factory(*args,**kwargs):observed['factory']=kwargs;return sentinel
    monkeypatch.setattr(old,'CPUBridge',Bridge);monkeypatch.setattr(m,'PilotGivenACT',factory)
    dep=dict(model={},metadata={},source_inventory={},episode_id='expanded')
    result=m.make_controller(Env(),None,None,deployment=dep,estimate_provider=None,output='/unused')
    assert result is sentinel and observed['adapter'] is m
    assert observed['factory']['policy_receipt']['schema']==m.DEPLOYMENT_SCHEMA
    assert issubclass(m.ExpandedGivenACT,old.PilotGivenACT)
