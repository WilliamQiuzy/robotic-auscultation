"""CPU numerical tests for the explicit ACT constraint parameterization."""
from copy import deepcopy
import importlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


def module():
    name='sim.ausc_policy.m2_act_posture_head'
    assert importlib.util.find_spec(name) is not None, 'Explicit bounded ACT head required'
    return importlib.import_module(name)


def stats():
    from sim.ausc_policy import m2_given_contact_runtime as rt
    return dict(schema=rt.given.NORMALIZATION_SCHEMA,task_id=rt.pose.TASK_ID,
        action_semantics_id=rt.given.ACTION_SEMANTICS_ID,fitted_split='train',
        observation_contract=rt.pose.observation_contract(),state_mean=np.linspace(-.3,.3,17).tolist(),
        state_std=np.linspace(.1,.7,17).tolist(),action_mean=np.linspace(-.002,.003,7).tolist(),
        action_std=np.linspace(.02,.04,7).tolist())


def profile():
    return dict(schema='low_elbow_posture_v1',elbow_link='openarm_left_link4',max_elbow_z_m=1.,
        joint_lower_rad=[-1.8]*7,joint_upper_rad=[1.8]*7,observed_joint_tolerance_rad=.005,
        max_elbow_above_shoulder_m=0.,min_contact_elbow_below_wrist_m=0.)


def spec():
    m=module()
    return m.make_spec(profile(),[dict(path='/bound/profile.json',bytes=1,sha256='a'*64)],stats())


def test_head_declares_parameterization_and_exact_metadata():
    m=module();s=spec()
    assert s['schema']==m.SCHEMA and s['inset_rad']==s['softness_rad']==.001
    assert s['physical_posture_profile']==profile() and s['normalization']==stats()
    assert s['guarantee']=='target_separable_joint_box_only'
    assert m.validate_spec(s) is s
    for key in ('inset_rad','softness_rad'):
        wrong=deepcopy(s);wrong[key]=.01
        with pytest.raises(ValueError):m.validate_spec(wrong)


@pytest.mark.parametrize('dtype_name',['float32','float64'])
def test_map_finite_extremes_monotone_interior_and_finite_gradients(dtype_name):
    torch=pytest.importorskip('torch');m=module();dtype=getattr(torch,dtype_name)
    bound=torch.tensor([[-1.8]*7,[1.8]*7],dtype=dtype)
    x=torch.linspace(-2.,2.,2001,dtype=dtype)[:,None].repeat(1,7).requires_grad_(True)
    y=m.bounded_targets(x,bound[0]+.001,bound[1]-.001,.001)
    assert bool(torch.all(y>=bound[0]+.001)) and bool(torch.all(y<=bound[1]-.001))
    assert bool(torch.all(torch.diff(y[:,0])>=0))
    torch.testing.assert_close(y[1000],x[1000],rtol=0,atol=2e-7)
    y.sum().backward();assert bool(torch.isfinite(x.grad).all()) and bool(torch.all(x.grad>=0))
    extreme=torch.tensor([-torch.finfo(dtype).max,torch.finfo(dtype).max],dtype=dtype)[:,None].repeat(1,7).requires_grad_(True)
    out=m.bounded_targets(extreme,bound[0]+.001,bound[1]-.001,.001)
    assert bool(torch.isfinite(out).all());out.sum().backward()
    assert bool(torch.isfinite(extreme.grad).all())


def test_same_forward_output_train_infer_poison_labels_and_state_dict_ownership():
    torch=pytest.importorskip('torch');m=module();s=spec()
    class SingleForward(torch.nn.Module):
        def __init__(self):
            super().__init__();self.raw=torch.nn.Parameter(torch.ones(20,7)*100);self.calls=0
        def forward(self,batch):
            self.calls+=1
            return self.raw[None].expand(len(batch['observation.state']),-1,-1),(None,None)
    class Policy(torch.nn.Module):
        def __init__(self):super().__init__();self.model=SingleForward()
    p=Policy();keys=set(p.state_dict());params=list(p.parameters());m.install(p,s)
    assert set(p.state_dict())==keys and list(p.parameters())==params
    class Poison(dict):
        def __getitem__(self,key):
            assert key!='action','Head must never read action labels'
            return super().__getitem__(key)
    q=np.linspace(-1.,1.,7);st=stats()
    obs=torch.tensor(((q-np.array(st['state_mean'])[:7])/np.array(st['state_std'])[:7])[None],dtype=torch.float32)
    batch=Poison({'observation.state':obs})
    train=p.model(batch)[0];p.eval();infer=p.model(batch)[0]
    torch.testing.assert_close(train,infer,rtol=0,atol=0)
    assert p.model.calls==2
    decoded=train.detach().numpy()*np.array(st['action_std'])+np.array(st['action_mean'])+q
    assert np.all(decoded>=-1.8) and np.all(decoded<=1.8)
    train.sum().backward();assert bool(torch.isfinite(p.model.raw.grad).all())
    d=m.diagnostics(p);assert d['forward_calls']==2 and d['max_displacement_rad']>0
    with pytest.raises(ValueError,match='once|installed'):m.install(p,s)


def test_full_float32_normalization_head_runtime_decode_box():
    torch=pytest.importorskip('torch');m=module();s=spec();rng=np.random.default_rng(19);st=stats()
    q=rng.uniform(-1.8,1.8,(200,7));obs=((q-np.array(st['state_mean'])[:7])/np.array(st['state_std'])[:7]).astype('float32')
    raw=torch.tensor(rng.normal(size=(200,20,7))*1e4,dtype=torch.float32)
    result=m.transform(raw,{'observation.state':torch.from_numpy(obs)},s)
    decoded=result.detach().numpy().astype(float)*np.array(st['action_std'])+np.array(st['action_mean'])+q[:,None]
    assert np.all(decoded>=-1.8) and np.all(decoded<=1.8)


def test_representability_audit_rejects_labels_without_clipping():
    m=module();s=spec();labels=np.zeros((2,20,7));mask=np.tile(np.arange(20)>=2,(2,1));labels[mask]=999
    histories=np.zeros((2,2,17));result=m.audit_targets(labels,mask,histories,s)
    assert result['valid_cells']==4 and result['minimum_inset_margin_rad']>0
    labels[0,1,6]=1.7995
    with pytest.raises(ValueError,match='representable|inset'):m.audit_targets(labels,mask,histories,s)
    assert labels[0,1,6]==1.7995


def test_official_act_single_forward_loss_and_prediction_use_the_head(monkeypatch):
    from sim.ausc_policy import m2_given_contact_runtime as rt
    if Path(sys.prefix).resolve()!=(Path(rt.VENDOR_ROOT)/'.venv').resolve():
        pytest.skip('Official ACT integration requires the pinned LeRobot virtualenv')
    torch=pytest.importorskip('torch')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    # Pytest lives only in an isolated fixture target. Exclude its metadata
    # while invoking the unchanged strict official dependency inventory.
    with monkeypatch.context() as isolated:
        isolated.setattr(sys,'path',[p for p in sys.path if '/test-deps' not in p])
        torch,policy,_=rt.make_model({'lerobot_root':rt.VENDOR_ROOT},device='cpu')
    assert sum(p.numel() for p in policy.parameters())==2658823
    m=module();s=spec();m.install(policy,s);seen=[]
    handle=policy.model.register_forward_hook(lambda module,args,out:seen.append(out[0]))
    batch={'observation.state':torch.zeros(4,7),'observation.environment_state':torch.zeros(4,27),
        'action':torch.ones(4,20,7),'action_is_pad':torch.zeros(4,20,dtype=torch.bool)}
    batch['action_is_pad'][0,11:]=True
    policy.train();loss,_=policy(batch)
    assert len(seen)==1 and m.diagnostics(policy)['forward_calls']==1
    valid=~batch['action_is_pad'];expected=(seen[0]-batch['action']).abs()[valid].mean()
    torch.testing.assert_close(loss,expected)
    loss.backward();assert any(p.grad is not None and bool(torch.any(p.grad!=0)) for p in policy.parameters())
    handle.remove();policy.eval();q=np.array(s['normalization']['state_mean'])[:7]
    prediction=policy.predict_action_chunk({k:v for k,v in batch.items() if k.startswith('observation.')})
    decoded=prediction.detach().numpy()*s['normalization']['action_std']+s['normalization']['action_mean']+q
    assert np.all(decoded>=-1.8) and np.all(decoded<=1.8)
    assert m.diagnostics(policy)['forward_calls']==2 and not torch.cuda.is_initialized()


def test_training_only_initialization_sets_zero_relative_and_does_not_touch_other_parameters():
    torch=pytest.importorskip('torch');m=module();s=spec()
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__();self.action_head=torch.nn.Linear(13,7);self.encoder=torch.nn.Linear(7,13)
        def forward(self,batch):
            raw=self.action_head(self.encoder(batch['observation.state']))[:,None].expand(-1,20,-1)
            return raw,(None,None)
    class Policy(torch.nn.Module):
        def __init__(self):super().__init__();self.model=Backbone()
    p=Policy();other={k:v.clone() for k,v in p.state_dict().items() if not k.startswith('model.action_head.')}
    receipt=m.initialize_training(p,s)
    assert receipt['schema']==m.INIT_SCHEMA and receipt['training_only'] is True
    assert bool(torch.all(p.model.action_head.weight==0))
    raw=p.model({'observation.state':torch.zeros(3,7)})[0].detach().numpy()
    relative=raw*np.array(s['normalization']['action_std'])+np.array(s['normalization']['action_mean'])
    np.testing.assert_allclose(relative,0,atol=1e-9)
    assert all(torch.equal(p.state_dict()[k],v) for k,v in other.items())
    with pytest.raises(ValueError,match='once|initialized'):m.initialize_training(p,s)
    # Installing the inference head must not reset a learned final linear layer.
    with torch.no_grad():p.model.action_head.weight.fill_(.123);p.model.action_head.bias.fill_(.456)
    learned={k:v.clone() for k,v in p.state_dict().items()};m.install(p,s)
    assert all(torch.equal(p.state_dict()[k],v) for k,v in learned.items())
