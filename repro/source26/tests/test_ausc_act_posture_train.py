"""Explicit v2 dispatch and unchanged legacy ACT numerical admission."""
from copy import deepcopy
import numpy as np
import pytest
from sim.ausc_policy import m2_act_auxiliary_train as train,m2_given_contact_expanded_pilot_train as expanded
from sim.ausc_policy import m2_given_contact_expanded_pilot_deployment as deployment,m2_rebuild_paired_eval as paired
from sim.ausc_policy import m2_act_posture_head as head,m2_act_chunk_data as data,m2_given_contact_runtime as rt
from test_ausc_act_posture_head import spec


def test_v2_has_distinct_schemas_and_protocol_head_in_loss_and_metrics():
    assert hasattr(train,'CONFIG_SCHEMA_V2'), 'Explicit full-prefix/head training v2 schema required'
    config=dict(train.SCHEDULE,schema=train.CONFIG_SCHEMA_V2,posture_head=spec())
    p=train.training_protocol(config)
    assert p['schema']==train.PROTOCOL_SCHEMA_V2 and p['posture_head']==config['posture_head']
    assert p['parameter_count']==2658823 and p['history_frames']==2 and p['action_horizon']==20 and p['execution_prefix']==10
    assert p['auxiliary_admission']['preparation_schema']==data.SCHEMA
    assert p['head_active_in_loss_and_metrics'] is True
    old=train.training_protocol(dict(train.SCHEDULE,schema=train.CONFIG_SCHEMA))
    assert old['schema']==train.PROTOCOL_SCHEMA and 'posture_head' not in old


def test_explicit_v2_dispatch_routes_new_config_selection_complete_and_cpu(monkeypatch):
    assert hasattr(train,'CONFIG_SCHEMA_V2'), 'V2 dispatcher required'
    sentinel=object()
    for entry in ('validate_config','training_protocol','load_selected_policy','verify_selected','fit','validate_complete','model_binding'):
        monkeypatch.setattr(train,entry,lambda *a,**k:sentinel)
    assert expanded.validate_config({'schema':train.CONFIG_SCHEMA_V2}) is sentinel
    assert expanded.training_protocol({'schema':train.CONFIG_SCHEMA_V2}) is sentinel
    assert expanded.load_selected_policy({'schema':train.SELECTION_SCHEMA_V2}) is sentinel
    assert deployment.validate_complete({'schema':train.PILOT_SCHEMA_V2}) is sentinel
    monkeypatch.setattr(rt,'read_bound',lambda ref:dict(schema=train.PILOT_SCHEMA_V2))
    assert deployment.model_binding({}) is sentinel


def test_v1_metadata_cannot_silently_admit_v2_bank(tmp_path):
    ref=rt.write_new(tmp_path/'v2.json',dict(schema=data.SCHEMA,complete=True,optimization_admission=data.ADMISSION,**data.old.FLAGS))
    with pytest.raises(ValueError):train.auxiliary_metadata(ref)


def test_head_runtime_profile_equality_is_shared_and_preserves_gates(monkeypatch):
    s=spec();complete=dict(schema='m2_act_auxiliary_act_pilot_v2',protocol=dict(posture_head=s))
    monkeypatch.setattr(head,'validate_bound_spec',lambda s,n:s)
    config=dict(posture_limits=deepcopy(s['physical_posture_profile']))
    head.validate_runtime_head(complete,config)
    config['posture_limits']['max_elbow_z_m']+=.001
    with pytest.raises(ValueError,match='profile'):head.validate_runtime_head(complete,config)
    head.validate_runtime_head(dict(schema=train.PILOT_SCHEMA),{})


def test_v2_oracle_authentication_requires_actual_command_snapshot(tmp_path,monkeypatch):
    from test_ausc_act_chunk_data import imported
    got,_=imported(tmp_path,monkeypatch);s=got['samples'];source=s['sources'][0]
    proposal=rt.read_bound(source['bindings']['proposal'])
    with pytest.raises(ValueError,match='command|seam'):
        data.validate_oracle_sample(s,0,proposal,proposal['query'],snapshot={'actual':'q/v/a'})


@pytest.mark.parametrize('adapter',[deployment,paired])
def test_v2_checkpoint_bridge_roundtrip(adapter,tmp_path):
    from test_ausc_given_contact_pose import reference,metadata
    ref=reference(tmp_path);meta=dict(metadata(ref),episode_id='new-act',target_id='L1');q=np.zeros(7)
    state=rt.shared.contract.encode_state(q,ref.pose_base,1.,metadata=meta)
    dep=dict(metadata=meta,episode_id='new-act',source_episode_id='native',
        model=dict(checkpoint={'sha256':'a'*64},checkpoint_schema='m2_act_auxiliary_act_pilot_v2'),
        source_config={'sha256':'b'*64},protocol={'sha256':'c'*64},scenario_id='scenario')
    request=adapter.make_request(dep,1,q,ref.pose_base,1.,metadata=meta,history=np.stack([state,state]),
        history_times=[0.,.05],context=dict(observed_at_s=.05,physics_row=12,
        context=['new-act','L1',0,'FREE'],source_id='given-contact-'+ref.sha256))
    block=adapter.decode_response(adapter.response_for(request,np.zeros((20,7))),request)
    assert block.metadata['checkpoint_policy']=='m2_act_auxiliary_act_pilot_v2'


@pytest.mark.parametrize('loader',[train.validate_complete,train.load_selected_policy])
def test_v2_marker_cannot_relabel_a_v1_config_or_model(tmp_path,monkeypatch,loader):
    from test_ausc_act_auxiliary_train import completed_run
    value=completed_run(tmp_path,monkeypatch);value['schema']=train.PILOT_SCHEMA_V2
    with pytest.raises(ValueError,match='version|Version'):loader(value)


def test_factory_initializes_only_new_training_and_never_selected_inference(monkeypatch):
    calls=[];policy=object();software=object();torch=object()
    monkeypatch.setattr(rt,'make_model',lambda *a,**k:(torch,policy,software))
    monkeypatch.setattr(head,'install',lambda p,s:calls.append('install'))
    monkeypatch.setattr(head,'initialize_training',lambda p,s:calls.append('training_init'))
    config=dict(schema=train.CONFIG_SCHEMA_V2,lerobot_root='/official',posture_head=spec())
    assert train.make_model(config,device='cpu')==(torch,policy,software)
    assert calls==['install']
    train.make_model(config,device='cuda',training=True)
    assert calls==['install','install','training_init']
    train.make_model(dict(schema=train.CONFIG_SCHEMA,lerobot_root='/official'),device='cpu',training=True)
    assert calls==['install','install','training_init']


def test_paired_all_arms_share_the_original_runtime_head_profile_check(tmp_path,monkeypatch):
    from test_ausc_rebuild_paired_eval import setup
    from sim.ausc_policy import isaac_batch
    m,source,model,_,_=setup(tmp_path,monkeypatch)
    prepared=rt.read_bound(m.prepare(model['complete']['path'],source['path'],tmp_path/'prepared',astra_settings=m.DEFAULT_ASTRA))
    observed=[]
    monkeypatch.setattr(head,'validate_runtime_head',lambda complete,config:observed.append(config[m.DESCRIPTOR]['arm']))
    for arm in m.ARMS:
        batch=rt.read_bound(prepared['batches'][arm]);m.preflight_arm(isaac_batch.validate_jobs(batch)[0]['settings'])
    assert set(observed)==set(m.ARMS)


def test_v2_head_field_does_not_require_an_optional_replacement_gpu_binding(monkeypatch):
    from test_ausc_act_auxiliary_train import scope_value
    frozen=scope_value();aux_ref=dict(path='/opaque/aux.json',bytes=1,sha256='a'*64)
    native=dict(counts=train.NATIVE_COUNTS,split_subject_ids=dict(train=['p001120']))
    config=dict(train.SCHEDULE,schema=train.CONFIG_SCHEMA_V2,base='/workspace/wam',dataset=train.NATIVE_DATASET,
        auxiliary=aux_ref,evaluation_cohort=train.TRAIN16_SCOPE,posture_head=spec())
    monkeypatch.setattr(train.scope,'validate',lambda ref:frozen['cohorts'])
    monkeypatch.setattr(rt,'read_bound',lambda ref:native if ref==train.NATIVE_DATASET else frozen)
    monkeypatch.setattr(train,'native_normalization',lambda config:config['posture_head']['normalization'])
    monkeypatch.setattr(head,'validate_bound_spec',lambda value,stats:value)
    monkeypatch.setattr(train,'auxiliary_metadata',lambda ref,**k:dict(sources=[dict(subject_id='p000758')]))
    monkeypatch.setattr(train.binding,'validate_dataset',lambda *a,**k:None)
    monkeypatch.setattr(train.scope,'validate_train16_metadata',lambda *a,**k:None)
    monkeypatch.setattr(train.binding,'validate',lambda *a,**k:pytest.fail('Optional GPU binding validation called'))
    assert train.validate_config(config) is config
