"""V2 full computed prefix labels, with explicit legacy boundaries."""
from copy import deepcopy
import importlib
import importlib.util

import numpy as np
import pytest

from sim.ausc_policy import m2_given_contact_runtime as rt,m2_act_auxiliary_data as old
from test_ausc_act_auxiliary_data import oracle_fixture,fixture


def module():
    name='sim.ausc_policy.m2_act_chunk_data'
    assert importlib.util.find_spec(name) is not None, 'Explicit full-prefix auxiliary v2 API required'
    return importlib.import_module(name)


def imported(tmp_path,monkeypatch):
    m=module();tmp_path.mkdir(parents=True,exist_ok=True);complete,frozen,meta,_,_=oracle_fixture(tmp_path)
    # Unit fixtures are unqualified; production pin validator has its own rejection test.
    monkeypatch.setattr(m,'require_frozen_archive',lambda *a:None)
    return m.import_student_oracle(complete,frozen,allowed_train_ids={meta['subject_id']}),meta


def test_full_prefix_is_bound_to_computed_cells_and_no_future_histories(tmp_path,monkeypatch):
    m=module();got,meta=imported(tmp_path,monkeypatch);s=got['samples']
    assert s['schema']==m.SCHEMA and s['optimization_admission']==m.ADMISSION
    assert s['setter_is_pad'].tolist()==[[False,False]+[True]*18]*2
    np.testing.assert_allclose(s['relative_setters'][0,:2],[[.676]*7,[.776]*7])
    assert s['source_histories'].shape==(2,2,17)
    assert all(s[k] is False for k in old.FLAGS)
    assert got['slot_ledger'][1]['status']=='rejected'
    m.validate_samples(s,s['sources'],{meta['subject_id']})
    with pytest.raises(ValueError):old.validate_samples(s,s['sources'],{meta['subject_id']})


@pytest.mark.parametrize('fault',['tail','history','mask','origin'])
def test_resealed_v2_arrays_cannot_change_any_computed_label(tmp_path,monkeypatch,fault):
    m=module();got,meta=imported(tmp_path,monkeypatch);s=got['samples']
    arrays={k:s[k].copy() for k in old.ARRAY_KEYS}
    if fault=='tail':arrays['relative_setters'][0,1,2]+=.001
    elif fault=='history':arrays['source_histories'][0,0,2]+=.001
    elif fault=='mask':arrays['setter_is_pad'][0,1]=True;arrays['relative_setters'][0,1]=0
    else:arrays['source_histories'][0,-1,2]+=.001
    with pytest.raises(ValueError,match='prefix|genuine|bound|computed'):
        m.bind_samples(arrays,s['sources'],{meta['subject_id']})


def test_unpinned_archive_rejected_without_loading_arrays_or_executing_python(tmp_path,monkeypatch):
    m=module();complete,frozen,meta,_,_=oracle_fixture(tmp_path)
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('Unpinned numeric archive opened'))
    with pytest.raises(ValueError,match='pinned|Pinned|frozen|Frozen'):
        m.import_student_oracle(complete,frozen,allowed_train_ids={meta['subject_id']})


@pytest.mark.parametrize('split',['validation','test'])
def test_v2_split_rejection_precedes_numeric_reads(tmp_path,monkeypatch,split):
    m=module();s=fixture(tmp_path);sources=deepcopy(s['sources']);sources[0]['schema']=m.SOURCE_SCHEMA
    sources[0]['split']=split
    class Poison(dict):
        def __getitem__(self,key):
            assert key not in old.ARRAY_KEYS,'Array opened before source eligibility'
            return super().__getitem__(key)
    with pytest.raises(ValueError,match='TRAIN'):m.validate_samples(Poison(s),sources,s['allowed_train_ids'])


def test_v2_roundtrip_preserves_requests_rejections_and_full_masks(tmp_path,monkeypatch):
    m=module();got,meta=imported(tmp_path/'src',monkeypatch);sources=got['samples']['sources']
    frozen=sources[0]['bindings']['frozen_request']
    ref=m.export_preparation(tmp_path/'bank',got['samples'],got['slot_ledger'],{meta['subject_id']},request_refs=[frozen],provenance={})
    loaded=m.load_preparation(ref,{meta['subject_id']})
    assert loaded['slot_ledger']==got['slot_ledger']
    np.testing.assert_array_equal(loaded['samples']['relative_setters'],got['samples']['relative_setters'])
    with pytest.raises(ValueError):old.load_preparation(ref,{meta['subject_id']})


def test_source23_both_targets_required_and_legacy_prefix_exact(tmp_path,monkeypatch):
    m=module();got,meta=imported(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='source23|R3|L3'):m.require_current_targets(got['samples']['sources'])
    a={k:np.zeros((2,)) for k in old.ARRAY_KEYS};b={k:v.copy() for k,v in a.items()}
    m.prove_ordered_prefix(a,b,2)
    b['relative_setters'][1]=1
    with pytest.raises(ValueError,match='prefix'):m.prove_ordered_prefix(a,b,2)


def test_lineage_reproduces_pinned_original_array_prefix_and_rejects_resealed_change(monkeypatch):
    m=module();n=2;arrays={k:np.zeros((n,)) for k in old.ARRAY_KEYS}
    original=dict(samples=dict(arrays,sources=[]))
    monkeypatch.setattr(m,'PLANNED_COUNT',n)
    monkeypatch.setattr(old,'load_preparation',lambda ref,ids:original)
    monkeypatch.setattr(rt,'read_bound',lambda ref:{'arrays':{'bound':'old'},'request_refs':['original'],
        'slot_ledger':[dict(status='rejected',rejection='original rejection')]})
    monkeypatch.setattr(m,'_upgrade_legacy_samples',lambda value,ids:value,raising=False)
    monkeypatch.setattr(m,'validate_new_runs',lambda *a:None)
    monkeypatch.setattr(m,'require_current_targets',lambda sources:None)
    proof=m.prove_ordered_prefix(arrays,arrays,n)
    aux=dict(samples=dict(arrays,sources=[]),request_refs=['original','new'],
        slot_ledger=[dict(status='rejected',rejection='original rejection')],
        provenance=dict(legacy_complete=m.LEGACY_COMPLETE,legacy_arrays={'bound':'old'},planned_count=n,
            planned_ordered_prefix=proof,legacy_source_ids=[],legacy_converted_oracles=6,original_request_slots=70))
    m.validate_lineage(aux,set())
    aux['slot_ledger'][0]['rejection']='rewritten original outcome'
    with pytest.raises(ValueError,match='original|Original'):m.validate_lineage(aux,set())
    aux['slot_ledger'][0]['rejection']='original rejection'
    aux['samples']['source_times']=np.ones(n)
    with pytest.raises(ValueError,match='prefix'):m.validate_lineage(aux,set())


def test_new_rejection_ledger_is_bound_to_actual_completed_frozen_diagnostic(tmp_path,monkeypatch):
    m=module();got,meta=imported(tmp_path,monkeypatch);s=got['samples'];refs=s['sources'][0]['bindings']
    monkeypatch.setattr(m,'SOURCE23_MANIFEST',refs['frozen_manifest'])
    aux=dict(got,request_refs=[refs['frozen_request']],provenance=dict(new_oracle_runs=[
        dict(complete=refs['diagnostic_root'],frozen_request=refs['frozen_request'])]))
    original=dict(sources=[],**{k:s[k][:0].copy() for k in old.ARRAY_KEYS})
    m.validate_new_runs(aux,dict(request_refs=[],slot_ledger=[]),original,{meta['subject_id']})
    aux['slot_ledger'][1]['rejection']='replacement rejection'
    with pytest.raises(ValueError,match='outcome|ledger|diagnostic'):
        m.validate_new_runs(aux,dict(request_refs=[],slot_ledger=[]),original,{meta['subject_id']})
