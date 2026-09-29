"""Normalized parity is required; old decoded-coordinate parity is not."""
from types import SimpleNamespace
import copy
import numpy as np
import pytest
from sim.auscultation_pi05_v11 import parity as P,train as T


def actual_fixture(monkeypatch):
    row=dict(episode_id='R1_x',start=0,phase='FREE',target='R1')
    oldraw=np.full((10,7),.1,np.float32);newraw=np.full((10,7),.2,np.float32);z=np.zeros((10,7),np.float32)
    history=dict(references={},mapping={('R1_x',0):0},rows={'original120':[row],'fixed160':[row]},
        arrays={'original120':{'normalized_predictions':z[None],'predictions':oldraw[None]},'fixed160':{'pass_000_normalized':z,'pass_000_raw':oldraw}})
    data=SimpleNamespace(indices={'train':[(0,0)]},episodes=[{'state':np.zeros((1,41))}])
    numeric={f'{family}_000_{name}':v.copy() for family in ('original120','fixed160') for name,v in [('normalized',z[None]),('raw',newraw)]}
    def decode(d,zn,state,phase,target):
        assert phase=='FREE' and target=='R1';return newraw.copy()
    monkeypatch.setattr(P.D,'decode_prediction',decode)
    return data,history,numeric


def test_new_decode_can_differ_but_normalized_tensor_must_match(monkeypatch):
    d,h,n=actual_fixture(monkeypatch);P._check_actual(d,h,n)
    n['fixed160_000_normalized'][0,0,0]=.001
    with pytest.raises(ValueError,match='normalized'):P._check_actual(d,h,n)


def test_actual_new_decoder_bytes_cannot_be_substituted_with_old_raw(monkeypatch):
    d,h,n=actual_fixture(monkeypatch);n['original120_000_raw']=h['arrays']['original120']['predictions'][0].copy()
    with pytest.raises(ValueError,match='canonical decode'):P._check_actual(d,h,n)


@pytest.mark.parametrize('field,value',[('schema','wam_pi05_v10_recovery_checkpoint_binding_v1'),('local_step',1),('optimizer_step',True),('schedule_cursor',4),('cumulative_trainable_update_age',7301)])
def test_cross_version_or_clock_rejected_before_checkpoint(monkeypatch,field,value):
    d=SimpleNamespace(normalization={});binding=dict(schema='wam_pi05_v11_bounded_checkpoint_binding_v1',local_step=0,optimizer_step=0,schedule_cursor=0,cumulative_trainable_update_age=7300)
    binding[field]=value;monkeypatch.setattr(P.D,'dataset_binding',lambda d:{})
    monkeypatch.setattr(T,'checkpoint_binding',lambda *a:pytest.fail('Bad context reached weights'))
    with pytest.raises(ValueError):P._context(d,binding)


def test_no_heldout_label_encoding_in_trainer_or_parity():
    import ast
    from pathlib import Path
    for module in (T,P):
        tree=ast.parse(Path(module.__file__).read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='D' and n.func.attr=='make_batch']
        assert all(isinstance(n.args[1],ast.Constant) and n.args[1].value=='train' for n in calls)
    assert 'decoded_action_equivalence_to_parent=False' in Path(P.__file__).read_text()
