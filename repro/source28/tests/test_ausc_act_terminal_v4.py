"""Data-only V4 fixtures; no native/GPU/model execution or physical qualification."""
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def data():
    name='sim.ausc_policy.m2_act_terminal_data'
    assert importlib.util.find_spec(name), 'Separate terminal-query V4 admission API required'
    return importlib.import_module(name)


def bridge():
    name='sim.ausc_policy.m2_act_frozen26_data'
    assert importlib.util.find_spec(name), 'Pinned original26 isolated validation bridge required'
    return importlib.import_module(name)


def parent_fixture():
    m=data();n=2
    arrays=dict(source_histories=np.arange(n*34,dtype='<f8').reshape(n,2,17),
        source_times=np.array([[0.,.05],[.05,.1]]),relative_setters=np.zeros((n,20,7)),
        setter_is_pad=np.tile(np.arange(20)>=1,(n,1)),source_indices=np.zeros(n,dtype=np.int64))
    source=dict(source_id='immutable-parent',schema='m2_act_auxiliary_source_v3',kind='conditional_ideal_continuation',
        subject_id='p001120',split='train',metadata=dict(subject_id='p001120',split='train'),windows=[{},{}])
    return dict(samples=dict(arrays,sources=[source]),sources=[source],slot_ledger=[dict(status='rejected',rejection='original')],
        request_refs=[dict(path='/old/request',bytes=1,sha256='a'*64)],provenance={'original':'unchanged'})


def request_fixture():
    m=data();slots=[]
    for semantic,row in zip(m.SEMANTICS,(14400,14400,None)):
        slots.append(dict(slot_id='R3:'+semantic,job='r3',subject_id='p001120',split='train',pose_id='pose',
            target_id='R3',semantic=semantic,physics_row=row))
    request=dict(schema=m.REQUEST_SCHEMA,slots=slots,oracle_request_refs=[],cases=[dict(job='r3',subject_id='p001120',split='train',
        pose_id='pose',target_id='R3',source=dict(config={},owner={},child={},native={}))],**m.old.FLAGS)
    ledger=[dict(slots[0],status='accepted',source_id='new-one',**m.old.FLAGS),
        dict(slots[1],status='duplicate',duplicate_of=slots[0]['slot_id'],**m.old.FLAGS),
        dict(slots[2],status='missing',rejection='No actual eligible query',**m.old.FLAGS)]
    imported=[dict(slot_id='oracle:r3:14400',subject_id='p001120',pose_id='pose',target_id='R3',
        kind='student_oracle_unexecuted',physics_row=14400,status='accepted',source_id='new-one',**m.old.FLAGS)]
    return request,ledger,imported


def test_pins_entire_act5_parent_source26_scope_and_keeps_data_api_independent():
    m=data();b=bridge()
    assert m.PARENT_COMPLETE==dict(path='/opt/wam-runtime/fast-wam/datasets/act-conditional-full-chunks-20261107-v1/COMPLETE.json',
        bytes=1202,sha256='d84afc9f705f25de34bc2aeb253ee79b43a52981ad6fdc57e3f19265da47b075')
    assert (m.PARENT_ROWS,m.PARENT_SOURCES,m.PARENT_SLOTS)==(35982,934,1109)
    assert b.SOURCE26_ROOT=='/opt/wam-runtime/code/wam-act-recovery-20261106-v26'
    assert b.SOURCE26_COMMIT=='05e972ca184383a525aff293fc9f3590e546da9c'
    assert b.SOURCE26_MANIFEST['bytes']==287532 and b.SOURCE26_MANIFEST['sha256']=='315f021e76f4e606f4a327b053abc22bacf8a1b8c649d35b4bd8ba9e644c743d'
    assert m.TRAIN16_SCOPE['bytes']==1521 and m.TRAIN16_SCOPE['sha256']=='adf6b0bd95669719989f6795c177f0a8419921dd79a257ca0c725f99b3220e71'
    assert not hasattr(m,'fit') and not hasattr(m,'make_model')
    assert m.ADMISSION['computational_optimization_eligible'] is True
    assert all(m.ADMISSION[k] is False for k in m.old.FLAGS)


@pytest.mark.parametrize('key',['source_histories','source_times','relative_setters','setter_is_pad','source_indices','sources','ledger','requests','dtype','signed_zero'])
def test_parent_proof_is_bit_exact_and_preserves_every_ordered_prefix(key):
    m=data();parent=parent_fixture();candidate=deepcopy(parent)
    m.assert_ordered_parent(parent,candidate)
    if key=='sources':candidate['samples']['sources'][0]['source_id']='resealed'
    elif key=='ledger':candidate['slot_ledger'][0]['rejection']='rewritten'
    elif key=='requests':candidate['request_refs'][0]['sha256']='b'*64
    elif key=='dtype':candidate['samples']['source_times']=candidate['samples']['source_times'].astype('float32')
    elif key=='signed_zero':candidate['samples']['relative_setters'][0,0,0]=-0.
    elif key=='setter_is_pad':candidate['samples'][key][0,1]=False
    else:candidate['samples'][key].flat[0]+=1
    with pytest.raises(ValueError,match='parent'):m.assert_ordered_parent(parent,candidate)


def test_all_identity_metadata_precedes_bound_source_or_numeric_payload_reads(monkeypatch):
    m=data();parent=parent_fixture()
    parent['sources'].append(dict(subject_id='p000923',split='train',metadata=dict(subject_id='p000923',split='train')))
    monkeypatch.setattr(m.rt,'bound_path',lambda *a:pytest.fail('Reference opened before entire TRAIN gate'))
    with pytest.raises(ValueError,match='TRAIN'):m._identity_gate(parent['sources'],{'p001120'})
    parent['sources'][-1].update(subject_id='p001120',split='validation')
    with pytest.raises(ValueError,match='TRAIN'):m._identity_gate(parent['sources'],{'p001120'})


def test_semantic_missing_duplicate_slots_retained_without_repeated_weight():
    m=data();request,ledger,imported=request_fixture()
    assert m.validate_terminal_ledger(request,ledger,imported,{'p001120'})==ledger
    changed=deepcopy(ledger);changed[1].pop('duplicate_of');changed[1].update(status='accepted',source_id='new-one')
    with pytest.raises(ValueError):m.validate_terminal_ledger(request,changed,imported,{'p001120'})
    with pytest.raises(ValueError):m.validate_terminal_ledger(request,ledger[:-1],imported,{'p001120'})
    changed=deepcopy(ledger);changed[2]['physics_row']=21600
    with pytest.raises(ValueError):m.validate_terminal_ledger(request,changed,imported,{'p001120'})


def test_semantic_rejection_and_duplicate_of_abstention_keep_original_reason():
    m=data();request,ledger,imported=request_fixture()
    imported[0].update(status='rejected',rejection='Original full geometry abstained');imported[0].pop('source_id')
    ledger[0].update(status='rejected',rejection=imported[0]['rejection']);ledger[0].pop('source_id')
    assert m.validate_terminal_ledger(request,ledger,imported,{'p001120'})==ledger
    ledger[0]['rejection']='replacement'
    with pytest.raises(ValueError):m.validate_terminal_ledger(request,ledger,imported,{'p001120'})


def inventory_fixture(tmp_path):
    b=bridge();root=tmp_path/'archive';root.mkdir()
    (root/'sim').mkdir();(root/'sim'/'__init__.py').write_text('')
    rows=[dict(file='sim/__init__.py',bytes=0,sha256=hashlib.sha256(b'').hexdigest())]
    (root/'CODE_COMMIT').write_text(b.SOURCE26_COMMIT+'\n')
    manifest=b.rt.write_new(root/'SOURCE_MANIFEST.json',rows)
    return root,manifest


@pytest.mark.parametrize('fault',['bytes','symlink','directory_symlink','unknown','traversal','duplicate','commit','count'])
def test_full_archive_inventory_rejects_before_any_python_execution(tmp_path,fault):
    b=bridge();root,manifest=inventory_fixture(tmp_path)
    if fault=='bytes':(root/'sim/__init__.py').write_text('tampered')
    elif fault=='symlink':(root/'sim/__init__.py').unlink();(root/'sim/__init__.py').symlink_to(tmp_path/'absent')
    elif fault=='directory_symlink':(root/'sim/__init__.py').unlink();(root/'sim').rmdir();(root/'sim').symlink_to(tmp_path)
    elif fault=='unknown':(root/'extra.py').write_text('')
    elif fault=='commit':(root/'CODE_COMMIT').write_text('0'*40)
    elif fault in ('traversal','duplicate'):
        rows=b.rt.read_bound(manifest)
        if fault=='traversal':rows[0]['file']='../outside.py'
        else:rows.append(rows[0])
        (root/'SOURCE_MANIFEST.json').write_text(json.dumps(rows));manifest=b.rt.reference(root/'SOURCE_MANIFEST.json')
    with pytest.raises(ValueError):b._verify_inventory(root,manifest,expected_count=2 if fault=='count' else 1)


def test_production_archive_cannot_accept_source24_or_an_arbitrary_root(tmp_path,monkeypatch):
    b=bridge();root,manifest=inventory_fixture(tmp_path)
    monkeypatch.setattr(b.rt,'bound_path',lambda *a:pytest.fail('Unpinned archive read'))
    with pytest.raises(ValueError,match='source26'):b.require_archive(manifest)
    from sim.ausc_policy import m2_act_conditional_data as v3
    with pytest.raises(ValueError,match='source26'):b.require_archive(v3.SOURCE24_MANIFEST)


def test_array_proof_includes_storage_dtype_shape_and_bytes():
    b=bridge();a=np.array([0.,1.],dtype='<f8');proof=b.array_proof(a)
    assert proof['dtype']==a.dtype.str and proof['shape']==[2]
    assert proof['sha256']==hashlib.sha256(a.tobytes()).hexdigest()
    assert proof!=b.array_proof(a.astype('float32'))
    other=a.copy();other[0]=-0.
    assert proof!=b.array_proof(other)


def test_run_headers_gate_every_config_before_archive_native_or_arrays(tmp_path,monkeypatch):
    m=data();rt=m.rt
    configs=[rt.write_new(tmp_path/f'config-{i}.json',dict(m2_given_contact_expanded_pilot_deployment=dict(metadata=dict(
        subject_id='p001120' if i==0 else 'p000923',split='train')))) for i in range(2)]
    cases=[];results=[]
    for i,config in enumerate(configs):
        directory=tmp_path/f'job-{i}';directory.mkdir()
        source=dict(config=config,owner={'poison':'owner'},child={'poison':'child'},native={'poison':'native'})
        rt.write_new(directory/'REQUEST.json',dict(source=source,physics_rows=[24],generator={'poison':'generator'}))
        per=rt.write_new(directory/'COMPLETE.json',dict(source_subject_id='p001120' if i==0 else 'p000923',source_split='train'))
        cases.append(dict(job=str(i),source=source));results.append(dict(job=str(i),result=per))
    frozen=rt.write_new(tmp_path/'freeze.json',dict(cases=cases,source_manifest={'poison':'manifest'}))
    complete=rt.write_new(tmp_path/'root.json',dict(results=results))
    monkeypatch.setattr(m.frozen26,'require_archive',lambda *a:pytest.fail('Archive opened before every source TRAIN gate'))
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('Numerical archive opened before every TRAIN gate'))
    assert hasattr(m,'_run_headers'), 'Metadata-only original diagnostic preflight required'
    with pytest.raises(ValueError,match='TRAIN'):m._run_headers([dict(complete=complete,frozen_request=frozen)],{'p001120'})


def new_query_fixture(tmp_path,monkeypatch):
    m=data()
    from test_ausc_act_auxiliary_data import oracle_fixture
    complete,frozen,meta,_,_=oracle_fixture(tmp_path)
    # These fixture seams are explicitly unqualified. No injection is exposed by
    # the production API, which must authenticate the real full source26 tree.
    monkeypatch.setattr(m.scope,'TRAIN16_IDS',(*m.scope.TRAIN16_IDS,meta['subject_id']))
    frozen_value=m.rt.read_bound(frozen)
    generator=m.rt.read_bound(m.rt.reference(Path(complete['path']).parent/'diagnostic/REQUEST.json'))['generator']
    monkeypatch.setattr(m.frozen26,'require_archive',lambda *a:{'sim/ausc_policy/m2_student_oracle_proposals.py':generator})
    monkeypatch.setattr(m.frozen26,'SOURCE26_MANIFEST',frozen_value['source_manifest'])
    owner=m.rt.read_bound(frozen_value['cases'][0]['source']['owner'])
    owner.update(completed=True,gpu_released=True,stage=dict(passed=True,exit_code=0,process_group_drained=True,
        errors=[],signals=[],deferred_signals=[]))
    Path(frozen_value['cases'][0]['source']['owner']['path']).write_text(json.dumps(owner))
    # Rebind the request/freeze to the closed owner bytes.
    owner_ref=m.rt.reference(frozen_value['cases'][0]['source']['owner']['path'])
    frozen_value['cases'][0]['source']['owner']=owner_ref
    path=Path(complete['path']).parent/'diagnostic/REQUEST.json';request=m.rt.read_bound(m.rt.reference(path))
    request['source']['owner']=owner_ref;path.write_text(json.dumps(request))
    frozen_path=tmp_path/'fixture-freeze.json';frozen=m.rt.write_new(frozen_path,frozen_value)
    authenticated={}
    for row in (24,48):
        path=tmp_path/'diagnostic'/f'query-{row}.json';proposal=m.rt.read_bound(m.rt.reference(path))
        snapshot=dict(fields=dict(current_position=[.01]*7,current_velocity=[.02]*7,current_acceleration=[.03]*7))
        proposal['preview'].update(input_state=snapshot,input_sha256=m.rt.digest(snapshot))
        path.write_text(json.dumps(proposal));proposal_ref=m.rt.reference(path)
        outcome_path=tmp_path/'diagnostic'/f'outcome-{row}.json';outcome=m.rt.read_bound(m.rt.reference(outcome_path))
        outcome['computational_proposal']=proposal_ref;outcome_path.write_text(json.dumps(outcome))
        authenticated[row]=dict(query=proposal['query'],snapshot=snapshot,commands=[[.01]*7]*3,
            proposal_authenticated_sha256=proposal_ref['sha256'])
    per_path=tmp_path/'diagnostic/COMPLETE.json';per=m.rt.read_bound(m.rt.reference(per_path))
    per['outcomes']=[m.rt.read_bound(m.rt.reference(tmp_path/'diagnostic'/f'outcome-{r}.json')) for r in (24,36,48)]
    per_path.write_text(json.dumps(per));root=m.rt.read_bound(complete);root['results'][0]['result']=m.rt.reference(per_path)
    complete=m.rt.write_new(tmp_path/'fixture-root.json',root)
    # Rejected proposal row still authenticates its actual query.
    authenticated[36]=dict(authenticated[24],query=dict(authenticated[24]['query'],physics_row=36))
    monkeypatch.setattr(m.frozen26,'authenticate_queries',lambda groups:[ [deepcopy(authenticated[r]) for r in g['rows']] for g in groups])
    return complete,frozen,meta,authenticated


def test_new_full_query_import_uses_real_h2_and_keeps_abstention(tmp_path,monkeypatch):
    m=data();assert hasattr(m,'import_student_oracle'), 'New source26 V4 full-prefix importer required'
    complete,frozen,meta,_=new_query_fixture(tmp_path,monkeypatch)
    got=m.import_student_oracle(complete,frozen,allowed_train_ids={meta['subject_id']})
    assert [r['status'] for r in got['slot_ledger']]==['accepted','rejected','accepted']
    samples=got['samples'];assert samples['schema']==m.SCHEMA
    assert len(samples['sources'])==2 and all(s['schema']==m.SOURCE_SCHEMA for s in samples['sources'])
    assert samples['setter_is_pad'].tolist()==[[False,False]+[True]*18]*2
    np.testing.assert_allclose(samples['relative_setters'][0,:2],[[.676]*7,[.776]*7])
    assert all(s[k] is False for s in samples['sources'] for k in m.old.FLAGS)
    assert samples['optimization_admission']['computational_optimization_eligible'] is True


@pytest.mark.parametrize('fault',['history','origin','snapshot','teacher_prefix','mask','proposal_receipt','owner'])
def test_new_authenticated_query_or_computed_prefix_cannot_be_resealed(tmp_path,monkeypatch,fault):
    m=data();assert hasattr(m,'import_student_oracle'), 'New source26 V4 authentication required'
    complete,frozen,meta,authenticated=new_query_fixture(tmp_path,monkeypatch)
    got=m.import_student_oracle(complete,frozen,allowed_train_ids={meta['subject_id']})
    if fault=='owner':
        ref=got['samples']['sources'][0]['bindings']['owner'];Path(ref['path']).write_text('{}')
        with pytest.raises(ValueError):m.import_student_oracle(complete,frozen,allowed_train_ids={meta['subject_id']})
        return
    if fault=='history':authenticated[24]['query']['history'][0][0]+=.001
    elif fault=='origin':authenticated[24]['query']['measured_q'][0]+=.001
    elif fault=='snapshot':authenticated[24]['snapshot']['fields']['current_acceleration'][0]+=.001
    elif fault=='proposal_receipt':authenticated[24]['proposal_authenticated_sha256']='0'*64
    else:
        samples=got['samples']
        if fault=='teacher_prefix':samples['relative_setters'][0,1,0]+=.001
        else:samples['setter_is_pad'][0,1]=True;samples['relative_setters'][0,1]=0.
        # Reseal source hashes; authenticity must still reject.
        source=samples['sources'][0];source['sample_sha256']=m.old._row_digest(samples,[0]);source['source_id']=m.old._source_id(source)
        samples['source_table_sha256']=m.rt.digest(samples['sources'])
        with pytest.raises(ValueError):m.authenticate_new_genuine_groups(dict(samples=samples,sources=samples['sources']),{meta['subject_id']})
        return
    with pytest.raises(ValueError):m.import_student_oracle(complete,frozen,allowed_train_ids={meta['subject_id']})


def test_pack_uses_original_normalization_and_ignores_padded_future_poison(tmp_path,monkeypatch):
    m=data();assert hasattr(m,'pack_batch'), 'V4 packing API required'
    from sim.ausc_policy import m2_given_contact_act as native
    from test_ausc_act_auxiliary_data import fixture
    old=fixture(tmp_path);samples=dict(old,schema=m.SCHEMA,optimization_admission=m.ADMISSION)
    stats=native.fit_normalization(dict(schema=native.MASKED_DATASET_SCHEMA,task_id=native.TASK_ID,
        observation_contract=native.observation_contract(),action_semantics_id=native.ACTION_SEMANTICS_ID,
        states=old['source_histories'][:,1],histories=old['source_histories'],actions=old['relative_setters'],
        action_is_pad=old['setter_is_pad'],label_protocol=m.rt.shared.MASKED_LABEL_PROTOCOL),split='train')
    before=m.pack_batch(samples,stats,[0,2]);samples['relative_setters'][samples['setter_is_pad']]=np.nan
    after=m.pack_batch(samples,stats,[0,2])
    for key in before:np.testing.assert_array_equal(after[key],before[key])
    assert np.isfinite(after['action']).all() and samples['schema']==m.SCHEMA


def export_fixture(tmp_path,monkeypatch):
    m=data();parent=parent_fixture();parent['samples']['schema']='m2_act_auxiliary_preparation_v3'
    monkeypatch.setattr(m.frozen26,'SOURCE26_MANIFEST',m.rt.write_new(tmp_path/'archive-manifest.json',[]))
    parent['sources'][0].update(m.old.FLAGS);parent['sources'][0]['windows']=[dict(input_index=1,phase='APPROACH'),dict(input_index=2,phase='APPROACH')]
    parent['samples'].update(allowed_train_ids=['p001120'],source_table_sha256=m.rt.digest(parent['sources']),**m.old.FLAGS)
    parent['archive_validation']=dict(manifest=m.frozen26.SOURCE26_MANIFEST,code_commit=m.frozen26.SOURCE26_COMMIT,
        array_proofs={k:m.frozen26.array_proof(parent['samples'][k]) for k in m.ARRAY_KEYS})
    arrays=tmp_path/'PARENT.npz'
    with arrays.open('xb') as stream:np.savez_compressed(stream,**{k:parent['samples'][k] for k in m.ARRAY_KEYS})
    # The pinned parent fixture contains an immutable rejection and metadata-only request.
    parent['request_refs']=[m.rt.write_new(tmp_path/'parent-request.json',dict(original=True))]
    meta={k:v for k,v in parent['samples'].items() if k not in m.ARRAY_KEYS}
    meta.update(slot_ledger=parent['slot_ledger'],request_refs=parent['request_refs'],provenance=parent['provenance'])
    metaref=m.rt.write_new(tmp_path/'parent-metadata.json',meta)
    parentref=m.rt.write_new(tmp_path/'parent-complete.json',dict(schema='m2_act_auxiliary_preparation_v3',complete=True,
        metadata=metaref,metadata_sha256=metaref['sha256'],arrays=m.rt.reference(arrays),**m.old.FLAGS))
    monkeypatch.setattr(m,'PARENT_COMPLETE',parentref);monkeypatch.setattr(m,'PARENT_ROWS',2)
    monkeypatch.setattr(m,'PARENT_SOURCES',1);monkeypatch.setattr(m,'PARENT_SLOTS',1)
    calls=[]
    def archived(ref,allowed):
        assert ref==parentref;calls.append('original26 full parent validation');return deepcopy(parent)
    monkeypatch.setattr(m.frozen26,'load_parent',archived)
    scoperef=m.rt.write_new(tmp_path/'scope.json',dict(fixture_metadata_only=True))
    monkeypatch.setattr(m,'TRAIN16_SCOPE',scoperef)
    monkeypatch.setattr(m.scope,'validate_train16_metadata',lambda value:dict(train=list(m.scope.TRAIN16_IDS)))
    request,ledger,_=request_fixture()
    for slot in request['slots']:slot['physics_row']=None
    ledger=[dict(slot,status='missing',rejection='No genuine query for requested time',**m.old.FLAGS) for slot in request['slots']]
    request['oracle_request_refs']=[]
    request['cases'][0]['source']=dict(
        config=m.rt.write_new(tmp_path/'missing-config.json',dict(m2_given_contact_expanded_pilot_deployment=dict(metadata=dict(
            subject_id='p001120',split='train',target_id='R3')))),
        owner=m.rt.write_new(tmp_path/'missing-owner.json',dict(completed=True,gpu_released=True,config=dict(experiment='isaac_batch',jobs=[{}]*4),stage=dict(
            passed=True,exit_code=0,process_group_drained=True,errors=[],signals=[],deferred_signals=[]))),
        child=m.rt.write_new(tmp_path/'missing-child.json',{}),native=m.rt.write_new(tmp_path/'missing-native.json',dict(success=False)))
    monkeypatch.setattr(m.frozen26,'authenticate_sources',lambda sources:[dict(metadata=dict(subject_id='p001120',split='train',target_id='R3'),pose_id='pose')])
    reqref=m.rt.write_new(tmp_path/'TERMINAL_REQUEST.json',request)
    return parent,reqref,ledger,calls


def test_export_and_full_load_rerun_original_parent_without_resealing(tmp_path,monkeypatch):
    m=data();assert hasattr(m,'export_from_parent'), 'V4 parent export API required'
    parent,request,ledger,calls=export_fixture(tmp_path,monkeypatch)
    ref=m.export_from_parent(tmp_path/'v4',oracle_runs=[],terminal_request_ref=request,slot_ledger=ledger,allowed_train_ids={'p001120'})
    assert len(calls)==1
    complete,meta=m.metadata(ref,{'p001120'})
    assert len(calls)==1, 'Metadata admission must not replay the full archived parent'
    assert len(meta['sources'])==1 and meta['counts']['new_source26']['accepted_sources']==0
    loaded=m.load_preparation(ref,{'p001120'})
    assert len(calls)==2, 'Full training-array admission must execute the original validator again'
    assert loaded['slot_ledger']==parent['slot_ledger']+ledger
    assert loaded['request_refs'][:len(parent['request_refs'])]==parent['request_refs']
    m.assert_ordered_parent(parent,loaded)
    for key in m.ARRAY_KEYS:assert loaded['samples'][key].tobytes()==parent['samples'][key].tobytes()
    inventory=m.contribution_inventory(loaded)
    assert set(inventory['groups'])=={'native','planned','conditional','oracle'}
    assert inventory['groups']['native']['included_in_auxiliary'] is False


@pytest.mark.parametrize('fault',['parent_array','parent_source','parent_request','parent_proof','v4_array','flag'])
def test_resealed_v4_export_cannot_substitute_parent_or_claim_measured_future(tmp_path,monkeypatch,fault):
    m=data();assert hasattr(m,'export_from_parent'), 'V4 parent/lineage authority required'
    parent,request,ledger,calls=export_fixture(tmp_path,monkeypatch)
    ref=m.export_from_parent(tmp_path/'v4',oracle_runs=[],terminal_request_ref=request,slot_ledger=ledger,allowed_train_ids={'p001120'})
    complete=m.rt.read_bound(ref);meta=m.rt.read_bound(complete['metadata'])
    if fault=='parent_array':parent['samples']['relative_setters'][0,0,0]=.01
    elif fault=='parent_source':meta['sources'][0]['source_id']='relabelled';meta['source_table_sha256']=m.rt.digest(meta['sources'])
    elif fault=='parent_request':meta['request_refs'][0]=request
    elif fault=='parent_proof':meta['provenance']['ordered_parent_proof']['relative_setters']['sha256']='0'*64
    elif fault=='flag':meta['optimization_admission']['future_measured_histories']=True
    else:
        path=m.rt.bound_path(complete['arrays'])
        with np.load(path) as archive:arrays={k:archive[k].copy() for k in archive.files}
        arrays['relative_setters'][0,0,0]=.01
        with path.open('wb') as stream:np.savez(stream,**arrays)
        complete['arrays']=m.rt.reference(path)
    complete['metadata']=m.rt.write_new(tmp_path/'changed-meta.json',meta);complete['metadata_sha256']=complete['metadata']['sha256']
    changed=m.rt.write_new(tmp_path/'changed-complete.json',complete)
    with pytest.raises(ValueError):m.load_preparation(changed,{'p001120'})


def test_metadata_hashes_all_bound_references_but_never_loads_training_arrays(tmp_path,monkeypatch):
    m=data();assert hasattr(m,'export_from_parent'), 'V4 metadata admission required'
    _,request,ledger,calls=export_fixture(tmp_path,monkeypatch)
    ref=m.export_from_parent(tmp_path/'v4',oracle_runs=[],terminal_request_ref=request,slot_ledger=ledger,allowed_train_ids={'p001120'})
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('Metadata admission opened training NPZ'))
    m.metadata(ref,{'p001120'});assert len(calls)==1
    Path(request['path']).write_text('{}')
    with pytest.raises(ValueError):m.metadata(ref,{'p001120'})


def test_metadata_cohort_gate_precedes_array_hashing(tmp_path,monkeypatch):
    m=data();assert hasattr(m,'export_from_parent'), 'V4 metadata isolation required'
    _,request,ledger,calls=export_fixture(tmp_path,monkeypatch)
    ref=m.export_from_parent(tmp_path/'v4',oracle_runs=[],terminal_request_ref=request,slot_ledger=ledger,allowed_train_ids={'p001120'})
    complete=m.rt.read_bound(ref);meta=m.rt.read_bound(complete['metadata']);meta['sources'][0]['split']='validation'
    complete['metadata']=m.rt.write_new(tmp_path/'bad-meta.json',meta);complete['metadata_sha256']=complete['metadata']['sha256']
    changed=m.rt.write_new(tmp_path/'bad-complete.json',complete)
    monkeypatch.setattr(m.rt,'bound_path',lambda *a:pytest.fail('Payload reference hashed before all TRAIN metadata'))
    with pytest.raises(ValueError,match='TRAIN'):m.metadata(changed,{'p001120'})


def test_v4_code_binding_uses_actual_loaded_modules_and_exposes_no_injection_api():
    m=data();assert hasattr(m,'code_binding'), 'V4 code binding helper required'
    binding=m.code_binding()
    assert binding['files']['data']==m.rt.reference(m.__file__)
    assert binding['files']['source26_bridge']==m.rt.reference(m.frozen26.__file__)
    import inspect
    assert 'archive_root' not in inspect.signature(m.export_from_parent).parameters
    assert 'production' not in inspect.signature(m.export_from_parent).parameters


@pytest.mark.parametrize('fault',['live_owner','wrong_owner_batch','native_passed','missing_case','case_val','forged_config_val','case_source_mismatch'])
def test_all_missing_target_still_requires_negative_original_train_owner(tmp_path,monkeypatch,fault):
    m=data();parent,request,ledger,calls=export_fixture(tmp_path,monkeypatch)
    value=m.rt.read_bound(request)
    assert 'cases' in value, 'Missing-query cases must retain original source ownership refs'
    source=value['cases'][0]['source']
    if fault=='live_owner':
        owner=m.rt.read_bound(source['owner']);owner['gpu_released']=False
        value['cases'][0]['source']['owner']=m.rt.write_new(tmp_path/'live-owner.json',owner)
    elif fault=='wrong_owner_batch':
        owner=m.rt.read_bound(source['owner']);owner['config']=dict(experiment='isaac_batch',jobs=[{}]*3)
        value['cases'][0]['source']['owner']=m.rt.write_new(tmp_path/'short-owner.json',owner)
    elif fault=='native_passed':value['cases'][0]['source']['native']=m.rt.write_new(tmp_path/'passed.json',dict(success=True))
    elif fault=='missing_case':value['cases']=[]
    elif fault=='case_val':value['cases'][0]['split']='validation'
    elif fault=='forged_config_val':
        config=m.rt.read_bound(source['config']);config['m2_given_contact_expanded_pilot_deployment']['metadata']['split']='validation'
        value['cases'][0]['source']['config']=m.rt.write_new(tmp_path/'validation-config.json',config)
    else:
        monkeypatch.setattr(m.frozen26,'authenticate_sources',lambda sources:[dict(metadata=dict(subject_id='p001120',split='train',target_id='L3'),pose_id='pose')])
    changed=m.rt.write_new(tmp_path/'bad-terminal-request.json',value)
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('Missing target failure must precede numerical payloads'))
    with pytest.raises(ValueError):m.export_from_parent(tmp_path/'invalid-v4',oracle_runs=[],terminal_request_ref=changed,
        slot_ledger=ledger,allowed_train_ids={'p001120'})


def test_original26_bridge_parent_cannot_load_an_arbitrary_dataset(monkeypatch):
    b=bridge()
    monkeypatch.setattr(b,'_run',lambda *a:pytest.fail('Arbitrary parent replay attempted'))
    with pytest.raises(ValueError,match='parent'):b.load_parent(dict(path='/arbitrary/COMPLETE.json',bytes=1,sha256='a'*64),{'p001120'})


def test_bridge_runs_original_package_in_isolated_cpu_child_and_verifies_transport(tmp_path,monkeypatch):
    b=bridge();root=tmp_path/'source26-fixture';(root/'sim/ausc_policy').mkdir(parents=True)
    (root/'sim/__init__.py').write_text('')
    (root/'sim/ausc_policy/__init__.py').write_text("INITIALIZER_TOKEN='unchanged fixture initializer'\n")
    (root/'sim/ausc_policy/m2_student_oracle_proposals.py').write_text('')
    (root/'sim/ausc_policy/m2_act_conditional_data.py').write_text('''import hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import sim.ausc_policy as package
class Runtime:
    def reference(self,path):
        path=Path(path)
        return dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    def write_new(self,path,value):
        Path(path).write_text(json.dumps(value));return self.reference(path)
rt=Runtime()
old=SimpleNamespace(ARRAY_KEYS=('source_histories','source_times','relative_setters','setter_is_pad','source_indices'))
def load_preparation(ref,allowed):
    assert package.INITIALIZER_TOKEN=='unchanged fixture initializer'
    assert sys.flags.isolated and sys.dont_write_bytecode
    assert sys.path[0]==str(Path(__file__).parents[2])
    assert allowed==['p001120']
    arrays=dict(source_histories=np.zeros((1,2,17)),source_times=np.array([[0.,.05]]),relative_setters=np.full((1,20,7),-0.),setter_is_pad=np.ones((1,20),bool),source_indices=np.zeros(1,dtype=np.int64))
    return dict(samples=dict(arrays,sources=[]),sources=[],validation_calls=['load_preparation'])
def validate_lineage(value,allowed):
    value['validation_calls'].append('validate_lineage')
''')
    (root/'CODE_COMMIT').write_text(b.SOURCE26_COMMIT+'\n')
    rows=[]
    for path in sorted(root.rglob('*.py')):
        ref=b.rt.reference(path);rows.append(dict(file=path.relative_to(root).as_posix(),bytes=ref['bytes'],sha256=ref['sha256']))
    manifest=b.rt.write_new(root/'SOURCE_MANIFEST.json',rows)
    monkeypatch.setattr(b,'SOURCE26_ROOT',str(root));monkeypatch.setattr(b,'SOURCE26_MANIFEST',manifest)
    monkeypatch.setattr(b,'SOURCE26_FILES',len(rows));monkeypatch.setattr(b.sys,'platform','linux')
    monkeypatch.setenv('PYTHONPATH','/future28/must-not-be-imported')
    parent=b.load_parent(b.PARENT_COMPLETE,{'p001120'})
    assert parent['validation_calls']==['load_preparation','validate_lineage']
    assert np.signbit(parent['samples']['relative_setters']).all()
    assert parent['archive_validation']['array_proofs']['relative_setters']==b.array_proof(parent['samples']['relative_setters'])
    assert not list(root.rglob('__pycache__')), '-B must leave immutable archive untouched'


def test_direct_new_import_refuses_train_identity_outside_original_train16(tmp_path,monkeypatch):
    m=data()
    from test_ausc_act_auxiliary_data import oracle_fixture
    complete,frozen,meta,_,_=oracle_fixture(tmp_path)
    assert meta['subject_id'] not in m.scope.TRAIN16_IDS
    monkeypatch.setattr(m.frozen26,'require_archive',lambda *a:pytest.fail('Out-of-scope archive read'))
    with pytest.raises(ValueError,match='TRAIN16'):m.import_student_oracle(complete,frozen,allowed_train_ids={meta['subject_id']})


def terminal_oracle_fixture(tmp_path,monkeypatch):
    """One genuine-shaped TRAIN16 query, independently typed unqualified fixture."""
    m=data();tmp_path.mkdir();rt=m.rt
    from test_ausc_given_contact_pose import inputs,bound
    from test_ausc_policy_m2_act import inputs as numeric_inputs
    from test_ausc_act_auxiliary_data import POSE
    point,plan,patient=inputs(tmp_path)
    p=rt.pose.bound_json(patient);p.update(identity='p001120',proposal='p001120_'+POSE);p['targets'][0]['name']='R3'
    patient=bound(tmp_path,'terminal-patient.json',p)
    planvalue=rt.pose.bound_json(plan);planvalue.update(name='R3');planvalue['patient_provenance'].update(identity='p001120',binding=patient)
    plan=bound(tmp_path,'terminal-plan.json',planvalue);point.update(name='R3',manifest_ref=patient)
    reference=rt.pose.build_given_reference(point,plan,patient,np.eye(4),0.)
    _,_,meta=numeric_inputs();meta.update(subject_id='p001120',target_id='R3')
    meta=rt.pose.route_metadata(meta,reference)
    source=dict(config=rt.write_new(tmp_path/'config.json',dict(m2_given_contact_expanded_pilot_deployment=dict(metadata=meta))),
        owner=rt.write_new(tmp_path/'owner.json',dict(completed=True,gpu_released=True,config=dict(experiment='isaac_batch',jobs=[{}]*4),stage=dict(passed=True,exit_code=0,
            process_group_drained=True,errors=[],signals=[],deferred_signals=[]))),child=rt.write_new(tmp_path/'child.json',{}),
        native=rt.write_new(tmp_path/'native.json',dict(success=False)))
    generator=rt.reference(m.old.__file__)
    manifest=rt.write_new(tmp_path/'SOURCE_MANIFEST.json',[dict(file='sim/ausc_policy/m2_student_oracle_proposals.py',
        bytes=generator['bytes'],sha256=generator['sha256'])])
    # Keep the archived generator path convention while using harmless fixture bytes.
    archive=tmp_path/'archive';(archive/'sim/ausc_policy').mkdir(parents=True)
    path=archive/'sim/ausc_policy/m2_student_oracle_proposals.py';path.write_text('# fixture only\n');generator=rt.reference(path)
    manifest=rt.write_new(archive/'SOURCE_MANIFEST.json',[dict(file='sim/ausc_policy/m2_student_oracle_proposals.py',
        bytes=generator['bytes'],sha256=generator['sha256'])])
    monkeypatch.setattr(m.frozen26,'SOURCE26_MANIFEST',manifest)
    monkeypatch.setattr(m.frozen26,'require_archive',lambda *a:{'sim/ausc_policy/m2_student_oracle_proposals.py':generator})
    row=24;q=np.full(7,.024);h=np.array([rt.shared.contract.encode_state(q-.001,reference.pose_base,1.,metadata=meta),
        rt.shared.contract.encode_state(q,reference.pose_base,1.,metadata=meta)]);times=np.array([.05,.1])
    query=dict(metadata=meta,measured_q=q.tolist(),measured_dq=[0.]*7,history=h.tolist(),history_times=times.tolist(),
        time_s=.1,physics_row=row,context=[meta['episode_id'],'R3',0,'APPROACH'],reference_sha256=reference.sha256)
    snapshot=dict(fields=dict(current_position=[.01]*7,current_velocity=[.02]*7,current_acceleration=[.03]*7))
    proposal=dict(kind='recovery_proposal',query=query,query_sha256=rt.digest(query),physics_executed=False,training_eligible=False,
        preview=dict(cells=[[.7]*7,[.8]*7],input_state=snapshot,input_sha256=rt.digest(snapshot),all_240hz_commands_verified=True,physics_executed=False))
    directory=tmp_path/'diagnostic';directory.mkdir();proposalref=rt.write_new(directory/'query-24.json',proposal)
    outcome=dict(physics_row=row,proposal_accepted=True,computational_proposal=proposalref,valid_proposed_cells=2,
        actions_executed=False,physics_qualified=False,training_eligible=False)
    rt.write_new(directory/'outcome-24.json',outcome)
    relative=np.zeros((1,20,7));relative[0,:2]=np.array([[.7]*7,[.8]*7])-q
    path=directory/'UNEXECUTED_PROPOSALS.npz'
    with path.open('xb') as stream:np.savez_compressed(stream,source_observed_histories=h[None],source_observation_times=times[None],
        source_physics_rows=np.array([row],np.int64),proposed_relative_setters=relative,proposed_is_pad=(np.arange(20)>=2)[None])
    flags=dict(actions_executed=False,physics_qualified=False,training_eligible=False)
    planref=rt.write_new(tmp_path/'plan-config.json',dict(metadata=meta))
    rt.write_new(directory/'REQUEST.json',dict(source=source,plan_config=planref,physics_rows=[row],generator=generator,**flags))
    per=rt.write_new(directory/'COMPLETE.json',dict(schema='m2_student_cpu_oracle_diagnostic_v1',complete=True,outcomes=[outcome],
        attempted_queries=1,accepted_proposals=1,arrays=rt.reference(path),source_subject_id='p001120',source_split='train',
        source_episode_verdict_unchanged=True,gpu_launched=False,**flags))
    freeze=rt.write_new(tmp_path/'FROZEN_REQUEST.json',dict(cases=[dict(job='r3',source=source,plan_config=planref,physics_rows=[row])],
        fixed_query_denominator=1,source_manifest=manifest,**flags))
    complete=rt.write_new(tmp_path/'COMPLETE.json',dict(results=[dict(job='r3',result=per)],fixed_query_denominator=1,gpu_launched=False,**flags))
    record=dict(query=query,snapshot=snapshot,commands=[[.01]*7]*3,proposal_authenticated_sha256=proposalref['sha256'])
    monkeypatch.setattr(m.frozen26,'authenticate_queries',lambda groups:[[deepcopy(record)] for g in groups])
    monkeypatch.setattr(m.frozen26,'authenticate_sources',lambda sources:[dict(metadata=deepcopy(meta),pose_id=POSE) for _ in sources])
    return dict(complete=complete,frozen_request=freeze),source,POSE


def test_accepted_tail_roundtrip_adds_one_source_for_two_semantic_slots(tmp_path,monkeypatch):
    m=data();parent,_,_,_=export_fixture(tmp_path,monkeypatch)
    run,source,pose=terminal_oracle_fixture(tmp_path/'new',monkeypatch)
    parent['archive_validation']['manifest']=m.frozen26.SOURCE26_MANIFEST
    imported=m.import_student_oracle(run['complete'],run['frozen_request'],allowed_train_ids={'p001120'})
    request,ledger,_=request_fixture()
    request['cases'][0].update(pose_id=pose,source=source)
    for slot in request['slots']:slot['pose_id']=pose
    request['slots'][0]['physics_row']=24;request['slots'][1]['physics_row']=24
    ledger=[dict(request['slots'][0],status='accepted',source_id=imported['sources'][0]['source_id'],**m.old.FLAGS),
        dict(request['slots'][1],status='duplicate',duplicate_of=request['slots'][0]['slot_id'],**m.old.FLAGS),
        dict(request['slots'][2],status='missing',rejection='No actual last eligible query retained',**m.old.FLAGS)]
    request['oracle_request_refs']=[run['frozen_request']]
    requestref=m.rt.write_new(tmp_path/'terminal-with-query.json',request)
    result=m.export_from_parent(tmp_path/'v4',oracle_runs=[run],terminal_request_ref=requestref,slot_ledger=ledger,allowed_train_ids={'p001120'})
    loaded=m.load_preparation(result,{'p001120'})
    m.assert_ordered_parent(parent,loaded)
    assert len(loaded['sources'])==2 and len(loaded['samples']['source_indices'])==3
    assert loaded['counts']['new_source26']==dict(accepted_sources=1,samples=1,attempted_slots=3,subject_ids=['p001120'])
    assert loaded['slot_ledger'][1:]==ledger
    np.testing.assert_array_equal(loaded['samples']['source_histories'][2:],imported['samples']['source_histories'])


@pytest.mark.parametrize('entry',['authenticate_sources','authenticate_queries'])
def test_bridge_direct_queries_gate_train16_before_native_archive_execution(tmp_path,monkeypatch,entry):
    b=bridge();source=dict(config=b.rt.write_new(tmp_path/'config.json',dict(m2_given_contact_expanded_pilot_deployment=dict(
        metadata=dict(subject_id='p000439',split='train')))),owner={},child={},native={})
    monkeypatch.setattr(b,'_run',lambda *a:pytest.fail('Out-of-TRAIN16 source reached archive execution'))
    value=[source] if entry=='authenticate_sources' else [dict(source=source,rows=[24],proposals={})]
    with pytest.raises(ValueError,match='TRAIN'):getattr(b,entry)(value)
