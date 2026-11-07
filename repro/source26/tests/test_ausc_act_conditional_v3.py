"""V3 fixtures exercise software contracts; they are not physical qualification."""
from copy import deepcopy
import importlib
import importlib.util

import numpy as np
import pytest


def data():
    name='sim.ausc_policy.m2_act_conditional_data'
    assert importlib.util.find_spec(name), 'Explicit typed conditional V3 data API required'
    return importlib.import_module(name)


def producer():
    name='sim.ausc_policy.m2_act_conditional_supervision'
    assert importlib.util.find_spec(name), 'Finite conditional V3 producer required'
    return importlib.import_module(name)


def test_v3_pins_parent_and_source24_without_weakening_v2(monkeypatch):
    m=data();from sim.ausc_policy import m2_act_chunk_data as v2
    assert m.PARENT_COMPLETE['bytes']==1129
    assert m.PARENT_COMPLETE['sha256']=='6b08048ff52454254eeda57cf161fdfcdf37324cdfba53fa11c501999d087c14'
    assert m.SOURCE24_MANIFEST['bytes']==286136
    generator=dict(path=m.SOURCE24_MANIFEST['path'].replace('SOURCE_MANIFEST.json','sim/ausc_policy/m2_student_oracle_proposals.py'),
        bytes=v2.GENERATOR_BYTES,sha256=v2.GENERATOR_SHA256)
    monkeypatch.setattr(m.rt,'bound_path',lambda ref:None)
    m.require_source24(m.SOURCE24_MANIFEST,generator)
    with pytest.raises(ValueError):v2.require_frozen_archive(m.SOURCE24_MANIFEST,generator)
    with pytest.raises(ValueError):m.require_source24(dict(m.SOURCE24_MANIFEST,sha256='a'*64),generator)


def test_anchor_selection_uses_ideal_approach_and_deterministic_distinct_ties():
    m=producer();h=np.zeros((6,2,17));h[:,1,7]=[.08,.06,.04,.03,.03,.02]
    windows=[dict(input_index=i,phase='APPROACH') for i in range(6)]
    assert m.anchor_indices(h,windows)==[1,2,3,5]
    h[:,1,7]=.04
    assert m.anchor_indices(h,windows)==[0,1,2,5]
    with pytest.raises(ValueError):m.anchor_indices(h[:3],windows[:3])


def test_orthogonal_static_seed_deterministic_and_stationary_fallback():
    m=producer();q=np.zeros(7);tangent=np.array([1.,0,0,0,0,0,0])
    a,rule=m.perturb_seed(q,tangent,seed=42,slot_index=3,scale_rad=.02)
    b,_=m.perturb_seed(q,tangent,seed=42,slot_index=3,scale_rad=.02)
    np.testing.assert_array_equal(a,b)
    assert abs(a@tangent)<1e-12 and np.linalg.norm(a)==pytest.approx(.02)
    assert rule=='orthogonal_local_teacher_tangent'
    c,rule=m.perturb_seed(q,np.zeros(7),seed=42,slot_index=3,scale_rad=.02)
    assert rule=='stationary_tangent_isotropic_fallback' and np.linalg.norm(c)==pytest.approx(.02)


def test_canonical_prefix_mask_or_computed_hold_never_masks_failed_cell():
    m=producer();raw=np.arange(21*7,dtype=float).reshape(21,7)
    cells,mask=m.continuation(raw,19,'mask_absent_canonical_suffix')
    np.testing.assert_array_equal(cells,raw[19:])
    assert mask.tolist()==[False,False]+[True]*18
    cells,mask=m.continuation(raw,19,'computed_constant_coarse_target_hold')
    np.testing.assert_array_equal(cells[:2],raw[19:])
    np.testing.assert_array_equal(cells[2:],np.repeat(raw[-1:],18,axis=0))
    assert not mask.any()
    with pytest.raises(ValueError):m.continuation(raw,19,'mask_failed_preview_cells')


def test_v3_identity_gate_precedes_numerical_or_reference_access(monkeypatch):
    m=data()
    source=dict(schema=m.SOURCE_SCHEMA,kind=m.CONDITIONAL_KIND,subject_id='p000923',split='validation',metadata={})
    monkeypatch.setattr(m.rt,'bound_path',lambda *a:pytest.fail('Reference read before TRAIN identity gate'))
    class Poison(dict):
        def __getitem__(self,key):
            assert key not in m.old.ARRAY_KEYS, 'Numeric read before identity gate'
            return super().__getitem__(key)
    with pytest.raises(ValueError,match='TRAIN'):m.validate_samples(Poison(),[source],{'p001120'})


def test_parent_proof_covers_every_array_source_id_and_rejection():
    m=data();arrays={k:np.arange(3) for k in m.old.ARRAY_KEYS}
    parent=dict(samples=dict(arrays,sources=[{'source_id':'old-id'}]),slot_ledger=[{'status':'rejected','rejection':'old reason'}],
        request_refs=['old-request'])
    candidate=dict(samples=dict({k:np.r_[v,99] for k,v in arrays.items()},sources=[{'source_id':'old-id'},{}]),
        slot_ledger=deepcopy(parent['slot_ledger'])+[{}],request_refs=['old-request','new-request'])
    m.prove_parent(parent,candidate)
    candidate['samples']['sources'][0]['source_id']='resealed'
    with pytest.raises(ValueError,match='parent'):m.prove_parent(parent,candidate)
    candidate['samples']['sources'][0]['source_id']='old-id'
    candidate['slot_ledger'][0]['rejection']='replacement reason'
    with pytest.raises(ValueError,match='parent'):m.prove_parent(parent,candidate)


def test_v3_four_group_means_and_old_protocol_unchanged():
    from sim.ausc_policy import m2_act_auxiliary_train as train
    assert hasattr(train,'CONFIG_SCHEMA_V3'), 'Explicit V3 training/config/checkpoint/CPU dispatch required'
    assert train.GROUPS==dict(native=dict(size=32,coefficient=.5),planned=dict(size=16,coefficient=.25),oracle=dict(size=16,coefficient=.25))
    batches={};calls=[]
    class Official:
        def __call__(self,batch):
            calls.append(batch);valid=~batch['action_is_pad']
            return (np.abs(batch['action'])*valid[...,None]).sum()/(valid.sum()*7),{}
    for i,(name,group) in enumerate(train.GROUPS_V3.items(),1):
        mask=np.tile(np.arange(20)>=i,(group['size'],1));action=np.full((group['size'],20,7),float(i));action[mask]=12345.
        batches[name]=dict(action=action,action_is_pad=mask)
    loss,logged=train.mixed_loss(Official(),batches,version=3)
    assert loss==pytest.approx(.5*1+.125*2+.125*3+.25*4) and len(calls)==4
    assert logged['conditional']['examples']==8
    from test_ausc_act_posture_head import spec
    from sim.ausc_policy import m2_act_posture_head as head
    legacy=spec();new=head.make_spec(legacy['physical_posture_profile'],legacy['profile_sources'],legacy['normalization'],version=2)
    protocol=train.training_protocol(dict(train.SCHEDULE,schema=train.CONFIG_SCHEMA_V3,posture_head=new))
    assert protocol['schema']==train.PROTOCOL_SCHEMA_V3 and protocol['objective']['conditional']['coefficient']==.125
    assert protocol['augmentation']==dict(q_noise_std=0.,q_dropout=0.)


def test_existing_v2_cache_preserves_three_group_sampling(tmp_path,monkeypatch):
    from sim.ausc_policy import m2_act_auxiliary_train as train
    from test_ausc_act_auxiliary_train import cache_inputs
    prepared,auxiliary=cache_inputs(tmp_path);auxiliary['samples']['schema']=train.chunk_data.SCHEMA
    monkeypatch.setattr(train.chunk_data,'pack_batch',lambda s,n,i:dict(action=s['relative_setters'],action_is_pad=s['setter_is_pad']))
    cache=train.build_cache(prepared,auxiliary)
    ids=train.sample_cached_indices(np.random.default_rng(42),cache)
    assert {k:len(v) for k,v in ids.items()}=={'native':32,'planned':16,'oracle':16}


def test_v3_runtime_and_cpu_routing_preserve_checkpoint_marker(tmp_path,monkeypatch):
    from sim.ausc_policy import m2_act_auxiliary_train as train,m2_given_contact_expanded_pilot_train as expanded
    assert hasattr(train,'SELECTION_SCHEMA_V3')
    sentinel=object();monkeypatch.setattr(train,'load_selected_policy',lambda s:sentinel)
    assert expanded.load_selected_policy({'schema':train.SELECTION_SCHEMA_V3}) is sentinel
    from sim.ausc_policy import m2_given_contact_expanded_pilot_deployment as dep,m2_rebuild_paired_eval as paired
    from test_ausc_given_contact_pose import reference,metadata
    ref=reference(tmp_path);meta=dict(metadata(ref),episode_id='v3',target_id='L1');q=np.zeros(7)
    state=dep.runtime.shared.contract.encode_state(q,ref.pose_base,1.,metadata=meta)
    config=dict(metadata=meta,episode_id='v3',source_episode_id='native',model=dict(checkpoint={'sha256':'a'*64},
        checkpoint_schema=train.PILOT_SCHEMA_V3),source_config={'sha256':'b'*64},protocol={'sha256':'c'*64},scenario_id='s')
    for adapter in (dep,paired):
        request=adapter.make_request(config,1,q,ref.pose_base,1.,metadata=meta,history=np.stack([state,state]),
            history_times=[0.,.05],context=dict(observed_at_s=.05,physics_row=12,context=['v3','L1',0,'FREE'],
                source_id='given-contact-'+ref.sha256))
        adapter.validate_request(request,config,sequence=1)
        block=adapter.decode_response(adapter.response_for(request,np.zeros((20,7))),request)
        assert block.metadata['checkpoint_policy']==train.PILOT_SCHEMA_V3


def test_head_v2_is_explicit_and_preserves_physical_box_and_teacher_representability(tmp_path):
    from sim.ausc_policy import m2_act_posture_head as head
    from test_ausc_act_posture_head import spec as old_spec
    legacy=old_spec()
    assert hasattr(head,'SCHEMA_V2'), 'Explicit head v2 required; do not relabel legacy v1'
    new=head.make_spec(legacy['physical_posture_profile'],legacy['profile_sources'],legacy['normalization'],version=2)
    assert new['schema']==head.SCHEMA_V2 and new['softness_rad']==.010
    assert legacy['schema']==head.SCHEMA and legacy['softness_rad']==.001
    assert new['physical_posture_profile']==legacy['physical_posture_profile'] and new['inset_rad']==legacy['inset_rad']==.001
    for value in (dict(legacy,softness_rad=.010),dict(new,softness_rad=.001)):
        with pytest.raises(ValueError):head.validate_spec(value)
    head.audit_targets(np.zeros((1,20,7)),np.zeros((1,20),bool),np.zeros((1,2,17)),new)
    labels=np.zeros((1,20,7));labels[0,0,0]=1.7995
    with pytest.raises(ValueError,match='representable|inset'):
        head.audit_targets(labels,np.zeros((1,20),bool),np.zeros((1,2,17)),new)
    assert labels[0,0,0]==1.7995


def test_v2_softness_has_useful_gradient_for_measured_40_50mrad_excursions():
    torch=pytest.importorskip('torch')
    from sim.ausc_policy import m2_act_posture_head as head
    x=torch.tensor([-.050,-.040,1.040,1.050],dtype=torch.float64,requires_grad=True)
    y=head.bounded_targets(x,torch.tensor(0.),torch.tensor(1.),.010);y.sum().backward()
    assert torch.all(torch.isfinite(y)) and torch.all(y>0) and torch.all(y<1)
    assert torch.all(x.grad>.006) and torch.all(x.grad<.019)
    f=torch.tensor([-100.,-.05,0.,.5,1.,1.05,100.],dtype=torch.float32)
    result=head.bounded_targets(f,torch.tensor(.001),torch.tensor(.999),.010)
    assert torch.all(torch.isfinite(result)) and torch.all(result[1:]>=result[:-1])
    assert torch.all(result>=.001) and torch.all(result<=.999)


def case_request(tmp_path,monkeypatch):
    m=producer();from test_ausc_act_planned_supervision import scene_fixture
    from test_ausc_given_contact_handoff import FixtureTrajectory
    scene,meta,config=scene_fixture(tmp_path)
    parent=m.planned.generate_planned_scene(scene,'L1',metadata=meta,plan_config_ref=config,
        allowed_train_ids={meta['subject_id']},trajectory_factory=FixtureTrajectory,
        verify_factory=lambda *a:lambda q:None,artifact_output=tmp_path/'old-raw')
    raw=parent['samples']['sources'][0]['bindings']['raw_schedule']
    cases=[]
    for i in range(64):
        case=dict(case_index=i,slot_id=f'planned:{i}',subject_id=meta['subject_id'],pose_id=scene['pose_id'],
            target_id='L1',canonical_status='rejected',canonical_rejection='original unavailable',anchors=[])
        if i==0:
            case.update(canonical_status='accepted',metadata=meta,scene=scene,plan_config=config,raw_schedule=raw)
            for a,anchor in enumerate(m.ANCHORS):
                seeds=[]
                for p in range(4):
                    q,rule=m.perturb_seed(np.zeros(7),np.zeros(7),seed=42,slot_index=a*4+p,scale_rad=0.)
                    seeds.append(dict(slot_index=a*4+p,perturbation=p,q=q.tolist(),rule=rule,scale_rad=0.))
                case['anchors'].append(dict(anchor=anchor,input_index=0,parent_row=0,q=[0.]*7,tangent=[0.]*7,seeds=seeds))
        cases.append(case)
    request=dict(schema=m.REQUEST_SCHEMA,parent_complete=m.data.PARENT_COMPLETE,case_denominator=64,slot_denominator=1024,
        seed=42,scales_rad=[0.]*4,perturbation_scale_semantics='joint_l2_radius_rad',
        suffix_policy=m.POLICIES[1],cases=cases,code=m._code(),frozen_before_preview=True,
        selection_validation_used=False,fresh_payloads_opened=False,**m.old.FLAGS)
    ref=m.rt.write_new(tmp_path/'request.json',request)
    return m,ref,meta,FixtureTrajectory


def test_static_fixture_checks_seed_all_240hz_commands_and_is_never_production(tmp_path,monkeypatch):
    m,request,meta,trajectory=case_request(tmp_path,monkeypatch);checks=[];factories=[]
    def factory(*args):
        factories.append(1)
        return lambda q:checks.append(q.copy())
    got=m.generate_case(request,0,tmp_path/'case',allowed_train_ids={meta['subject_id']},production=False,
        trajectory_factory=trajectory,verify_factory=factory)
    assert len(factories)==1 and len(checks)==16*(1+240)
    assert len(got['parts'])==16 and all(s['status']=='accepted' for s in got['slot_ledger'])
    sample=got['parts'][0];source=sample['sources'][0]
    np.testing.assert_array_equal(sample['source_histories'][0,0],sample['source_histories'][0,1])
    assert source['observation_origin']==m.data.ORIGIN and source['generated_static_input'] is True
    assert not sample['setter_is_pad'].any() and all(source[k] is False for k in m.old.FLAGS)
    candidate=m.rt.read_bound(source['bindings']['candidate'])
    assert candidate['computed_hold_cells']==18 and candidate['logical_geometry_checks']==candidate['actual_geometry_checks']==241
    outcome=m.rt.read_bound(got['outcomes'])
    assert outcome['slot_ledger']==got['slot_ledger'] and outcome['original_verification_calls']==16*241
    with pytest.raises(ValueError,match='fixture|injected|Production'):
        m.data.validate_samples(sample,sample['sources'],{meta['subject_id']})
    with pytest.raises(ValueError,match='fixture|injected|Production'):
        m.generate_case(request,0,tmp_path/'forbidden',allowed_train_ids={meta['subject_id']},trajectory_factory=trajectory)


def test_unsafe_seed_rejects_whole_slot_and_unavailable_case_retains_sixteen(tmp_path,monkeypatch):
    m,request,meta,trajectory=case_request(tmp_path,monkeypatch)
    def reject(*args):
        def check(q):raise ValueError('full original posture rejects seed')
        return check
    got=m.generate_case(request,0,tmp_path/'unsafe',allowed_train_ids={meta['subject_id']},production=False,
        trajectory_factory=trajectory,verify_factory=reject)
    assert not got['parts'] and len(got['slot_ledger'])==16
    assert all(s['status']=='rejected' and 'rejects seed' in s['rejection'] for s in got['slot_ledger'])
    unavailable=m.generate_case(request,1,tmp_path/'unavailable',allowed_train_ids={meta['subject_id']})
    assert len(unavailable['slot_ledger'])==16 and not unavailable['parts']
    assert all(s['rejection']=='canonical unavailable: original unavailable' for s in unavailable['slot_ledger'])


def test_failed_later_preview_tick_never_becomes_shorter_positive_mask(tmp_path,monkeypatch):
    m,request,meta,trajectory=case_request(tmp_path,monkeypatch);calls=[]
    def factory(*args):
        def verify(q):
            calls.append(q.copy())
            if len(calls)==14:raise ValueError('unsafe second-cell command')
        return verify
    got=m.generate_case(request,0,tmp_path/'failed-cell',allowed_train_ids={meta['subject_id']},production=False,
        trajectory_factory=trajectory,verify_factory=factory)
    assert got['slot_ledger'][0]['status']=='rejected' and 'second-cell' in got['slot_ledger'][0]['rejection']
    assert len(got['parts'])==15 and all(p['sources'][0]['slot_index']!=0 for p in got['parts'])
    assert all(not p['setter_is_pad'].any() for p in got['parts'])


def test_resealed_candidate_cannot_invent_velocity_or_measured_provenance(tmp_path,monkeypatch):
    m,request,meta,trajectory=case_request(tmp_path,monkeypatch)
    got=m.generate_case(request,0,tmp_path/'candidate',allowed_train_ids={meta['subject_id']},production=False,
        trajectory_factory=trajectory,verify_factory=lambda *a:lambda q:None)
    part=got['parts'][0];source=deepcopy(part['sources'][0]);candidate=m.rt.read_bound(source['bindings']['candidate'])
    candidate['generated_velocity'][0]=.01
    source['bindings']['candidate']=m.rt.write_new(tmp_path/'fake-velocity.json',candidate)
    arrays={k:part[k] for k in m.old.ARRAY_KEYS}
    with pytest.raises(ValueError,match='static|provenance'):
        m.data.bind_samples(arrays,[source],{meta['subject_id']},production=False)
    source=deepcopy(part['sources'][0]);source['observation_origin']='actual_authenticated_TRAIN_student_query'
    with pytest.raises(ValueError,match='provenance|measured'):
        m.data.bind_samples(arrays,[source],{meta['subject_id']},production=False)
    source=deepcopy(part['sources'][0]);source['actual_history_authenticated']=True
    with pytest.raises(ValueError,match='fields|measured'):
        m.data.bind_samples(arrays,[source],{meta['subject_id']},production=False)


def test_all_rejected_qualification_keeps_ledger_and_cannot_train(tmp_path,monkeypatch):
    m,request,meta,_=case_request(tmp_path,monkeypatch)
    monkeypatch.setattr(m,'validate_request',m.request_metadata)
    complete=m.generate(request,tmp_path/'qualification',allowed_train_ids={meta['subject_id']},case_indices=[1])
    value=m.rt.read_bound(complete);metadata=m.rt.read_bound(value['metadata'])
    assert value['complete'] is False and value['attempted_slots']==16 and len(metadata['slot_ledger'])==16
    with pytest.raises(ValueError,match='qualification|1024|complete'):m.load_generation(complete,{meta['subject_id']})


def test_v3_sampler_balances_anchors_and_counts_actual_sources_once():
    from sim.ausc_policy import m2_act_auxiliary_train as train
    assert hasattr(train,'record_draws'), 'Actual source/kind/identity gradient accounting required'
    tree={'body':{'pose':{'R3':{'near60mm':list(range(100)), 'near40mm':[100]}}}}
    cache=dict(near=np.array([0]),native_count=2,version=3,groups=train.GROUPS_V3,
        trees=dict(planned={'body':{'pose':{'R3':[0]}}},conditional=tree,oracle={'body':{'pose':{'R3':[0]}}}),
        subjects=dict(native=np.array(['body']*2,object),auxiliary=np.array(['body']*101,object)),
        source_ids=dict(native=np.array(['native:episode']*2,object),auxiliary=np.array(['aux:'+str(i) for i in range(101)],object)),
        kinds=dict(native=np.array(['native']*2,object),auxiliary=np.array(['x']*101,object)))
    rng=np.random.default_rng(42);counts=[]
    for _ in range(200):
        ids=train.sample_cached_indices(rng,cache)
        assert {k:len(v) for k,v in ids.items()}=={k:v['size'] for k,v in train.GROUPS_V3.items()}
        counts.extend(ids['conditional']==100)
    assert .4<np.mean(counts)<.6
    value=dict(gradient_examples={k:0 for k in train.GROUPS_V3},gradient_subject_examples={k:{} for k in train.GROUPS_V3},
        gradient_source_examples={k:{} for k in train.GROUPS_V3},gradient_kind_examples={k:{} for k in train.GROUPS_V3},
        actual_optimized_train_ids=[])
    train.record_draws(value,cache,ids)
    assert value['gradient_examples']['conditional']==8
    assert sum(value['gradient_source_examples']['conditional'].values())==8
    assert value['gradient_kind_examples']['native']=={'native':32} and value['actual_optimized_train_ids']==['body']


def test_v3_genuine_import_retains_every_full_cell_and_actual_h2(tmp_path,monkeypatch):
    m=data();from test_ausc_act_auxiliary_data import oracle_fixture
    complete,frozen,meta,_,_=oracle_fixture(tmp_path)
    monkeypatch.setattr(m,'require_source24',lambda *a:None)  # explicitly unqualified fixture only
    monkeypatch.setattr(m,'_require_closed_owner',lambda *a:None)
    got=m.import_student_oracle(complete,frozen,allowed_train_ids={meta['subject_id']});s=got['samples']
    assert s['setter_is_pad'].tolist()==[[False,False]+[True]*18]*2
    assert got['slot_ledger'][1]['status']=='rejected'
    assert all(source['schema']==m.SOURCE_SCHEMA and source['observation_origin']==m.old.ORIGINS['student_oracle_unexecuted']
        for source in s['sources'])
    with pytest.raises(ValueError):m.old.validate_samples(s,s['sources'],{meta['subject_id']})
    with pytest.raises(ValueError):m.v2.validate_samples(s,s['sources'],{meta['subject_id']})
    proposal=m.rt.read_bound(s['sources'][0]['bindings']['proposal']);actual=deepcopy(proposal['query'])
    actual['history'][0][0]+=.001
    with pytest.raises(ValueError,match='genuine|query|authenticated'):
        m.v2.validate_oracle_sample(s,0,proposal,actual)


def test_v3_unpinned_genuine_archive_rejected_before_arrays(tmp_path,monkeypatch):
    m=data();from test_ausc_act_auxiliary_data import oracle_fixture
    complete,frozen,meta,_,_=oracle_fixture(tmp_path)
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('Unpinned source24 array read'))
    with pytest.raises(ValueError,match='source24|Pinned'):m.import_student_oracle(complete,frozen,allowed_train_ids={meta['subject_id']})


def test_v3_requires_whole_owner_closed_before_any_oracle_arrays(tmp_path,monkeypatch):
    m=data();from test_ausc_act_auxiliary_data import oracle_fixture
    complete,frozen,meta,_,_=oracle_fixture(tmp_path)
    monkeypatch.setattr(m,'require_source24',lambda *a:None)
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('Unclosed owner oracle arrays opened'))
    with pytest.raises(ValueError,match='owner|closed|drained'):m.import_student_oracle(complete,frozen,allowed_train_ids={meta['subject_id']})


def test_v3_rejects_legacy_head_in_protocol_and_model_before_model_factory(monkeypatch):
    from sim.ausc_policy import m2_act_auxiliary_train as train
    from test_ausc_act_posture_head import spec
    config=dict(train.SCHEDULE,schema=train.CONFIG_SCHEMA_V3,posture_head=spec())
    monkeypatch.setattr(train.runtime,'make_model',lambda *a,**k:pytest.fail('Wrong-version head reached model factory'))
    with pytest.raises(ValueError,match='head|Head|version'):train.training_protocol(config)
    with pytest.raises(ValueError,match='head|Head|version'):train.make_model(config,device='cpu')


def test_new_head_float32_runtime_decode_box_and_inference_never_reinitializes():
    torch=pytest.importorskip('torch')
    from sim.ausc_policy import m2_act_posture_head as head
    from test_ausc_act_posture_head import spec
    legacy=spec();s=head.make_spec(legacy['physical_posture_profile'],legacy['profile_sources'],legacy['normalization'],version=2)
    rng=np.random.default_rng(19);stats=s['normalization'];q=rng.uniform(-1.8,1.8,(200,7))
    obs=((q-np.array(stats['state_mean'])[:7])/np.array(stats['state_std'])[:7]).astype('float32')
    raw=torch.tensor(rng.normal(size=(200,20,7))*1e4,dtype=torch.float32)
    result=head.transform(raw,{'observation.state':torch.from_numpy(obs)},s)
    decoded=result.detach().numpy().astype(float)*np.array(stats['action_std'])+np.array(stats['action_mean'])+q[:,None]
    assert np.all(decoded>=-1.8) and np.all(decoded<=1.8)
    class Policy(torch.nn.Module):
        def __init__(self):super().__init__();self.model=torch.nn.Linear(7,7)
    policy=Policy();before=deepcopy(policy.state_dict());head.install(policy,s)
    assert all(torch.equal(before[k],v) for k,v in policy.state_dict().items())


def test_new_head_same_forward_graph_for_loss_prediction_and_inference():
    torch=pytest.importorskip('torch')
    from sim.ausc_policy import m2_act_posture_head as head
    from test_ausc_act_posture_head import spec
    legacy=spec();s=head.make_spec(legacy['physical_posture_profile'],legacy['profile_sources'],legacy['normalization'],version=2)
    class Backbone(torch.nn.Module):
        def __init__(self):super().__init__();self.raw=torch.nn.Parameter(torch.zeros(20,7))
        def forward(self,batch):return self.raw[None].expand(len(batch['observation.state']),-1,-1),(None,None)
    class Policy(torch.nn.Module):
        def __init__(self):super().__init__();self.model=Backbone()
    policy=Policy();head.install(policy,s)
    stats=s['normalization'];q=np.zeros(7)
    observed=torch.tensor(((q-np.array(stats['state_mean'])[:7])/np.array(stats['state_std'])[:7])[None],dtype=torch.float32)
    class NoLabels(dict):
        def __getitem__(self,key):
            assert key!='action', 'Head must not consult labels'
            return super().__getitem__(key)
    batch=NoLabels({'observation.state':observed});train=policy.model(batch)[0]
    train.sum().backward();assert bool(torch.all(policy.model.raw.grad>0))
    policy.eval();inference=policy.model(batch)[0];torch.testing.assert_close(train,inference,rtol=0,atol=0)
    assert head.diagnostics(policy)['forward_calls']==2


def test_full_request_freezes_all_1024_slots_including_64_unavailable(tmp_path,monkeypatch):
    m=producer();from sim.ausc_policy.m2_act_recovery_scope import TRAIN16_IDS
    scenes=[];ledger=[];sources=[];histories=[];ids=[]
    for subject in TRAIN16_IDS:
        for pose in ('pose1','pose2'):
            scene=dict(subject_id=subject,split='train',cohort='train',pose_id=pose,
                plans={target:{'plan':{'opaque':subject+pose+target}} for target in ('R3','L3')})
            scenes.append(scene)
            for target in ('R3','L3'):
                index=len(ledger);slot=dict(slot_id='planned:'+subject+':'+pose+':'+target,
                    subject_id=subject,pose_id=pose,target_id=target,kind='planned_ideal_tracking',**m.old.FLAGS)
                if index<4:slot.update(status='rejected',rejection='canonical unavailable '+str(index))
                else:
                    source_id='parent-source-'+str(index);source_index=len(sources)
                    slot.update(status='accepted',source_id=source_id)
                    sources.append(dict(source_id=source_id,kind='planned_ideal_tracking',metadata=dict(subject_id=subject,split='train'),
                        windows=[dict(input_index=j,phase='APPROACH') for j in range(4)],
                        bindings=dict(geometry={'opaque':'geometry'},raw_schedule={'opaque':'raw'})))
                    h=np.zeros((4,2,17));h[:,1,7]=[.06,.04,.03,.02];h[:,1,0]=np.arange(4)*.001
                    histories.extend(h);ids.extend([source_index]*4)
                ledger.append(slot)
    manifest=m.rt.write_new(tmp_path/'scenes.json',dict(schema='m2_rebuild_scene_manifest_v1',purpose='capture',scenes=scenes,
        targets=['R3','L3'],requested_scenes=32,requested_target_slots=64,subject_ids=list(TRAIN16_IDS)))
    parent=dict(samples=dict(sources=sources,source_histories=np.array(histories),source_indices=np.array(ids)),
        request_refs=[manifest],slot_ledger=ledger)
    request=m._request(manifest,parent,set(TRAIN16_IDS),seed=42,scales_rad=[.001,.002,.003,.004],suffix_policy=m.POLICIES[1])
    assert len(request['cases'])==64 and request['slot_denominator']==1024
    assert sum(c['canonical_status']=='rejected' for c in request['cases'])*16==64
    assert sum(len(a['seeds']) for c in request['cases'] for a in c['anchors'])==960
    assert request==m._request(manifest,parent,set(TRAIN16_IDS),seed=42,scales_rad=[.001,.002,.003,.004],suffix_policy=m.POLICIES[1])
    original=m.perturb_seed
    def degenerate(q,tangent,**kwargs):
        if kwargs['slot_index']==64:raise ValueError('Degenerate perturbation seed rejected; do not resample')
        return original(q,tangent,**kwargs)
    monkeypatch.setattr(m,'perturb_seed',degenerate)
    rejected=m._request(manifest,parent,set(TRAIN16_IDS),seed=42,scales_rad=[.001,.002,.003,.004],suffix_policy=m.POLICIES[1])
    assert len(rejected['cases'])==64 and rejected['cases'][4]['anchors'][0]['seeds'][0]['seed_rejection'].startswith('ValueError: Degenerate')


def test_v3_draw_validation_binds_source_subject_kind_counts(monkeypatch):
    from sim.ausc_policy import m2_act_auxiliary_train as train
    monkeypatch.setattr(train.scope,'validate_train16_metadata',lambda *a,**k:None)
    frozen=dict(optimized_train_ids=['p001120'])
    inventory={name:{name+':one':dict(subject_id='p001120',kind=name)} for name in train.GROUPS_V3}
    value=dict(gradient_examples={k:v['size'] for k,v in train.GROUPS_V3.items()},
        gradient_subject_examples={k:{'p001120':v['size']} for k,v in train.GROUPS_V3.items()},
        gradient_source_examples={k:{k+':one':v['size']} for k,v in train.GROUPS_V3.items()},
        gradient_kind_examples={k:{k:v['size']} for k,v in train.GROUPS_V3.items()},actual_optimized_train_ids=['p001120'])
    train._validate_draws(value,1,frozen,['p001120'],version=3,inventory=inventory)
    for key in ('gradient_source_examples','gradient_kind_examples'):
        bad=deepcopy(value);bad[key]['conditional']={'invented':8}
        with pytest.raises(ValueError,match='source|kind|contribution'):
            train._validate_draws(bad,1,frozen,['p001120'],version=3,inventory=inventory)


def test_v3_completed_model_requires_exact_new_cpu_schema_and_source_contributions(tmp_path,monkeypatch):
    from sim.ausc_policy import m2_act_auxiliary_train as train,m2_given_contact_expanded_pilot_deployment as deployment
    from test_ausc_act_auxiliary_train import completed_run
    from test_ausc_act_posture_head import spec
    from sim.ausc_policy import m2_act_posture_head as head
    complete=completed_run(tmp_path,monkeypatch);config=deepcopy(complete['schedule']);legacy=spec()
    config.update(schema=train.CONFIG_SCHEMA_V3,posture_head=head.make_spec(legacy['physical_posture_profile'],
        legacy['profile_sources'],legacy['normalization'],version=2))
    complete.update(schema=train.PILOT_SCHEMA_V3,schedule=config,protocol=train.training_protocol(config))
    complete['inputs']['config']=train.runtime.write_new(tmp_path/'config-v3.json',config)
    complete['inputs']['code']=train.code_binding(version=3)
    actual=complete['optimization_scope']['optimized_train_ids'];native=complete['native_scope']['train']['subject_ids']
    inventory={name:{name+':'+subject:dict(subject_id=subject,kind=name) for subject in (native if name=='native' else actual)}
        for name in train.GROUPS_V3}
    monkeypatch.setattr(train,'contribution_inventory',lambda c:inventory)
    def draws(steps):
        subjects={name:{subject:(steps*group['size']-len(inventory[name])+1 if i==0 else 1)
            for i,subject in enumerate(native if name=='native' else actual)} for name,group in train.GROUPS_V3.items()}
        return dict(actual_optimized_train_ids=actual,gradient_examples={k:steps*v['size'] for k,v in train.GROUPS_V3.items()},
            gradient_subject_examples=subjects,gradient_source_examples={k:{k+':'+subject:n for subject,n in row.items()}
                for k,row in subjects.items()},gradient_kind_examples={k:{k:steps*v['size']} for k,v in train.GROUPS_V3.items()})
    complete.update(optimization=draws(20000),selected_optimization=draws(500))
    complete['auxiliary_counts'][data().CONDITIONAL_KIND]=dict(attempted_slots=1024,accepted_sources=960,samples=960,supervised_cells=19200)
    selection={k:deepcopy(v) for k,v in complete.items() if k not in ('schedule','evaluations','complete','completed_steps','cpu_equivalence','task_contract','model')}
    selection['schema']=train.SELECTION_SCHEMA_V3
    cpu=train.runtime.read_bound(complete['cpu_equivalence']);cpu.update(schema=train.CPU_SCHEMA_V3,
        selection=train.runtime.write_new(tmp_path/'SELECTED-v3.json',selection))
    complete['cpu_equivalence']=train.runtime.write_new(tmp_path/'CPU-v3.json',cpu)
    ref=train.runtime.write_new(tmp_path/'COMPLETE-v3.json',complete)
    assert deployment.model_binding(ref)[0]['checkpoint_schema']==train.PILOT_SCHEMA_V3
    cpu['schema']=train.CPU_SCHEMA_V2;complete['cpu_equivalence']=train.runtime.write_new(tmp_path/'wrong-cpu-v3.json',cpu)
    with pytest.raises(ValueError,match='CPU'):deployment.model_binding(train.runtime.write_new(tmp_path/'wrong-complete-v3.json',complete))
