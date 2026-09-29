"""Synthetic source-root fixtures; task replay is stubbed, never scientific qualification."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[3]
META=ROOT/'results/pi05_v6_pilot/armature02_training/task_conditioned_bound'
B=ROOT/'results/pi05_v7_repair/horizon_pair_B_completed_mirror_v1/verified_files/runs/horizon-pair-original8-v1-B/configuration.json'
SPEC=ROOT/'results/diagnostics/pi05_recovery_data_training_design_v1/machine_spec_v2.json'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n');return p

def module():
    try:return importlib.import_module('sim.auscultation_pi05_v10.training_data')
    except ModuleNotFoundError as e:
        if e.name not in ('sim.auscultation_pi05_v10','sim.auscultation_pi05_v10.training_data'):raise
        return SimpleNamespace(load_dataset=lambda *a,**k:pytest.fail('V10 scoped data API not implemented'))

@pytest.fixture
def fixture(tmp_path,monkeypatch):
    D=module()
    from sim.auscultation_pi05_v6 import train_openpi as old, derive_task_dataset as task
    from sim.auscultation_pi05_v6 import dynamics_contract as dyn, observed_supervisor as sup
    design=json.loads(SPEC.read_text());original=json.loads((META/'manifest.json').read_text());norm=json.loads(B.read_text())['normalization']
    normpath=write(tmp_path/'norm.json',norm)
    allids=design['data']['train_existing5']+design['data']['train_new3']
    spec_by_id={e['episode_id']:e for e in original['episodes']}
    for eid in allids:
        target=eid.split('_')[0];src=copy.deepcopy(next(e for e in original['episodes'] if e['target']==target))
        src.update(episode_id=eid,split='train',motion_family='independent_target_plan_'+target);spec_by_id[eid]=src
    sources=[];mapping=[];manifests={};records={};arrays={}
    common=json.loads((META/'dynamics/common.json').read_text())
    for sid,ids in [('original10',[e['episode_id'] for e in original['episodes']])]+[(e,[e]) for e in allids]:
        srcroot=tmp_path/sid;srcroot.mkdir();cm=write(srcroot/'common.json',common);recs=[]
        for ei,eid in enumerate(ids):
            rec=copy.deepcopy(spec_by_id[eid]);n=100
            q=np.array([.01,-.5,-.3,1.,.2,.1,.2],np.float32);state=np.zeros((n,41),np.float32);state[:,:7]=q
            act=np.tile(q,(n,1))+np.arange(n,dtype=np.float32)[:,None]*np.float32(1e-5)
            phases=np.repeat(['FREE','APPROACH','CONTACT','HOLD','RETRACT'],20)
            ar=dict(state=state,actions=act,rgb=np.full((n,2,2,3),ei,np.uint8),times=np.arange(n,dtype=np.float64)*.05,
                phases=phases,targets=np.repeat(rec['target'],n),prompts=np.array(['prompt '+str(p) for p in phases]),target_sources=np.repeat('rgbd_surface_tracking',n))
            npz=srcroot/(eid+'.npz');np.savez(npz,**ar);arrays[eid]=ar
            # Make each archive distinct across source roots too; a state channel is legitimate synthetic identity.
            ar['state'][:,7]=np.float32(len(records)*1e-5);np.savez(npz,**ar)
            outcome=write(srcroot/(eid+'.outcome.json'),dict(dynamics_recipe_sha256=common['recipe_sha256'],phase_config=dyn.supervisor_contract(),speed_scale=.7,joint_speed_limit_rad_s=.175,observed_supervisor_version=sup.VERSION,observed_supervisor_sha256=sha(Path(sup.__file__))))
            artifacts={'outcome.json':sha(outcome)}
            audit=write(srcroot/(eid+'.audit.json'),dict(schema='wam_observed_supervisor_replay_audit_v1',passed=True,source_supervisor_version=sup.VERSION,source_supervisor_sha256=sha(Path(sup.__file__)),audit_source_sha256=sha(ROOT/'sim/auscultation_pi05_v6/audit_observed_supervisor.py'),artifact_sha256=artifacts,replay=dict(every_observation_recomputed=True,every_manager_update_replayed=True,all_active_ticks_covered=True)))
            capsule=write(srcroot/(eid+'.capsule.json'),{'synthetic':True})
            phase=dict(phase_contract_sha256=original['phase_contract_sha256'],cross_boundary_label_indices=[19,39,59,79],contact_supervisor_equivalence_verified=True,observed_supervisor_version=sup.VERSION,observed_supervisor_sha256=sha(Path(sup.__file__)),independent_supervisor_audit=dict(path=audit.name,sha256=sha(audit),artifact_sha256=artifacts))
            qual=dict(schema='wam_vla_episode_qualification_v1',episode_id=eid,arrays_sha256=sha(npz),data_kind='closed_loop_rollout',qualified=True,phase_contract_version=original['phase_contract_version'],phase_contract_sha256=original['phase_contract_sha256'],action_contract_version=old.ACTION_CONTRACT_VERSION,action_contract_sha256=sha(ROOT/'sim/auscultation_pi05_v6/action_contract.py'),checks={k:True for k in old.QUALIFICATION_CHECKS},modalities=original['modalities'],evidence=dict(artifact_sha256=artifacts,joint_action_derivation={'phase_derivation':phase},task_observation_derivation=dict(parent_episode={'path':npz.name,'sha256':sha(npz)},evidence_root='.',input_sha256={capsule.name:sha(capsule)})))
            qp=write(srcroot/(eid+'.qualification.json'),qual)
            dr=write(srcroot/(eid+'.derivation.json'),{'synthetic':True})
            rec.update(derivation_path=dr.name,derivation_sha256=sha(dr),path=npz.name,sha256=sha(npz),qualification_path=qp.name,qualification_sha256=sha(qp),outcome_path=outcome.name,outcome_sha256=sha(outcome),dynamics_contract_path=cm.name,dynamics_contract_sha256=sha(cm),dynamics_recipe_sha256=common['recipe_sha256'])
            recs.append(rec);records[eid]=(srcroot,rec);mapping.append(dict(episode_id=eid,source_id=sid))
        manifest=copy.deepcopy(original);manifest.pop('protocol_sha256',None);manifest.update(episodes=recs,dynamics_contract_path=cm.name,dynamics_contract_sha256=sha(cm))
        mp=write(srcroot/'manifest.json',manifest);sources.append(dict(id=sid,manifest=dict(path=str(mp),sha256=sha(mp))));manifests[sid]=mp
    composite=write(tmp_path/'composite.json',dict(schema='wam_pi05_v10_recovery_composite_manifest_v1',sealed=True,design_binding=dict(path=str(SPEC),sha256=sha(SPEC)),sources=sources,episodes=mapping))
    calls=[]
    monkeypatch.setattr(task,'verify_task_derivation',lambda root,record,report,values:calls.append(record['episode_id']) or True)
    if hasattr(D,'ORIGINAL_MANIFEST_SHA256'):monkeypatch.setattr(D,'ORIGINAL_MANIFEST_SHA256',sha(manifests['original10']))
    def load(scopes=('train',),**kw):return D.load_dataset(composite,scopes=scopes,normalization_path=normpath,expected_normalization_sha256=sha(normpath),**kw)
    return SimpleNamespace(D=D,load=load,composite=composite,norm=norm,normpath=normpath,original=original,design=design,records=records,manifests=manifests,calls=calls,arrays=arrays)

def test_scoped_load_preserves_full_identity_without_test_arrays(fixture,monkeypatch):
    f=fixture;opened=[];load=np.load
    def watched(path,*a,**kw):opened.append(str(path));assert 'L5_observed' not in str(path);return load(path,*a,**kw)
    monkeypatch.setattr(np,'load',watched)
    d=f.load();assert d.loaded_scopes==('train',) and len(d.episodes)==18 and len(f.calls)==16
    assert len(d.indices['train'])>0 and d.indices['validation']==d.indices['test']==[]
    assert all('state' not in e for e in d.episodes if e['split']!='train')
    assert d.normalization==f.norm and d.normalization['residual_actions']['fit_episode_ids']==f.design['data']['train_original8']
    first=f.D.dataset_binding(d);v=f.load(('train','validation'));assert len(f.calls)==33
    assert f.D.dataset_binding(v)==first and v.qualification_scope_binding!=d.qualification_scope_binding
    assert first['schema']=='wam_pi05_v10_recovery_dataset_binding_v1'
    assert set(d.qualification_scope_binding['verified_episode_ids'])==set(f.design['data']['train_original8']+f.design['data']['train_existing5']+f.design['data']['train_new3'])

def test_phase_safe_last_action_and_original_view(fixture):
    f=fixture;d=f.load();e=0
    assert (e,9) in d.indices['train'] and (e,10) not in d.indices['train']  # action19 crosses phase
    view=f.D.original_train_view(d)
    assert len(view.episodes)==8 and len(view.indices['train'])==408
    assert view.qualification_scope_binding==d.qualification_scope_binding and view.parent is d
    assert view.normalization is d.normalization
    from sim.auscultation_pi05_v7.coverage_pilot_evaluation import build_diagnostic_schedule
    assert len(build_diagnostic_schedule(view))==120

def test_batch_and_decode_use_fixed_B_formula_and_old_observation(fixture,monkeypatch):
    f=fixture;d=f.load();from sim.auscultation_pi05_v7 import training_data as old
    seen=[]
    monkeypatch.setattr(old,'observation',lambda rgb,state,phase,target,tokenizer,norm:seen.append(norm) or {'state':state.copy()})
    obs,z=f.D.make_batch(d,'train',[0,1],object());assert len(seen)==2 and all(x is d.normalization for x in seen)
    for i,(ei,start) in enumerate(d.indices['train'][:2]):
        e=d.episodes[ei];q=f.D.decode_prediction(d,z[i],e['state'][start],str(e['phases'][start]))
        np.testing.assert_allclose(q,e['actions'][start:start+10],atol=2e-6,rtol=0)
    for split,ix in [('validation',[0]),('test',[0]),('train',[True]),('train',[-1]),('train',[]),('train',[99999])]:
        with pytest.raises(ValueError):f.D.make_batch(d,split,ix,None)

@pytest.mark.parametrize('scopes',[(),('bad',),('train','train'),('test',)])
def test_scope_and_test_authorization_rejected_before_npz(fixture,monkeypatch,scopes):
    f=fixture;monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('Unauthorized scope opened arrays'))
    with pytest.raises(ValueError):f.load(scopes)

@pytest.mark.parametrize('mutation',['wrong_norm_sha','refit_norm','missing_source','override_split','duplicate_episode','foreign_test_authorization'])
def test_manifest_and_norm_negatives_before_arrays(fixture,monkeypatch,mutation):
    f=fixture;manifest=json.loads(f.composite.read_text());kw={}
    if mutation=='wrong_norm_sha':kw['expected_normalization_sha256']='0'*64
    elif mutation=='refit_norm':n=copy.deepcopy(f.norm);n['state']['schema']='changed';write(f.normpath,n)
    elif mutation=='missing_source':manifest['sources'].pop();write(f.composite,manifest)
    elif mutation=='override_split':manifest['episodes'][0]['split']='test';write(f.composite,manifest)
    elif mutation=='duplicate_episode':manifest['episodes'][1]=manifest['episodes'][0];write(f.composite,manifest)
    else:kw['test_authorization']={'approved':True}
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('Invalid identity opened arrays'))
    with pytest.raises(ValueError):f.D.load_dataset(f.composite,scopes=('test',) if mutation=='foreign_test_authorization' else ('train',),normalization_path=f.normpath,expected_normalization_sha256=kw.pop('expected_normalization_sha256',sha(f.normpath)),**kw)

def test_scope_artifact_tamper_and_norm_mutation_rejected(fixture):
    f=fixture;d=f.load();root,rec=f.records[f.design['data']['train_original8'][0]]
    (root/rec['outcome_path']).write_text('{}\n')
    with pytest.raises(ValueError):f.D.dataset_binding(d)
    d.normalization['mode']='wrong'
    with pytest.raises(ValueError):f.D.make_batch(d,'train',[0],None)

def test_original_task_and_supervisor_checks_cannot_be_waived(fixture,monkeypatch):
    f=fixture;from sim.auscultation_pi05_v6 import derive_task_dataset as task
    def fail(*a,**k):raise ValueError('Actual task derivation rejected')
    monkeypatch.setattr(task,'verify_task_derivation',fail)
    with pytest.raises(ValueError,match='task derivation'):f.load()


def rebind_record(f,eid,monkeypatch,*,change_arrays=None,change_report=None,change_record=None):
    root,record=f.records[eid];record=copy.deepcopy(record)
    qp=root/record['qualification_path'];report=json.loads(qp.read_text())
    if change_arrays:
        with np.load(root/record['path'],allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        change_arrays(arrays);np.savez(root/record['path'],**arrays)
        record['sha256']=sha(root/record['path']);report['arrays_sha256']=record['sha256'];report['evidence']['task_observation_derivation']['parent_episode']['sha256']=record['sha256']
    if change_report:change_report(report)
    write(qp,report);record['qualification_sha256']=sha(qp)
    if change_record:change_record(record)
    mp=root/'manifest.json';m=json.loads(mp.read_text());m['episodes']=[record if x['episode_id']==eid else x for x in m['episodes']];write(mp,m)
    composite=json.loads(f.composite.read_text())
    for s in composite['sources']:
        if Path(s['manifest']['path'])==mp:
            s['manifest']['sha256']=sha(mp)
            if s['id']=='original10':monkeypatch.setattr(f.D,'ORIGINAL_MANIFEST_SHA256',sha(mp))
    write(f.composite,composite)


def test_original_state_domain_gate_survives_no_refit(fixture,monkeypatch):
    f=fixture;f.load()  # Ensure this is a feature assertion rather than an import failure.
    rebind_record(f,f.design['data']['train_original8'][0],monkeypatch,change_arrays=lambda a:a['state'].__setitem__((0,14),9.))
    with pytest.raises(ValueError,match='physical domain'):f.load()


def test_positive_test_authorization_is_exact_and_selected_only(fixture,monkeypatch,tmp_path):
    import sys
    f=fixture;dataset=f.load();identity=f.D.dataset_binding(dataset);run=tmp_path/'run';run.mkdir()
    lockpath=write(run/'selection.lock.json',{'synthetic':True});lb={'path':str(lockpath),'sha256':sha(lockpath)}
    cp={'data_hash':identity['data_hash'],'checkpoint_path':str(run/'model/step-000000000500'),'local_step':500}
    lock={'schema':'wam_pi05_v10_recovery_r5_selection_lock_v1','selection_path':str(run/'selection.json'),'selection_sha256':'1'*64,'selection':{'run':str(run),'dataset_binding':identity,'selected':{'checkpoint_binding':cp}}}
    calls=[]
    def verify(path,expected_sha=None):calls.append((path,expected_sha));return lock
    monkeypatch.setitem(sys.modules,'sim.auscultation_pi05_v10.evaluation',SimpleNamespace(verify_selection_lock=verify))
    value=dict(schema='wam_pi05_v10_recovery_final_test_reservation_v1',run=str(run),selection_lock_binding=lb,selected_checkpoint_binding=cp,dataset_binding=identity,test_split='test',episode_id='L5_observed_armature02_v1',status='reserved')
    rp=write(run/'FINAL_TEST_STARTED.json',value);rb={'path':str(rp),'sha256':sha(rp)};auth={'selection_lock_binding':lb,'reservation_binding':rb}
    test=f.load(('test',),test_authorization=auth)
    assert test.loaded_scopes==('test',) and test.indices['train']==test.indices['validation']==[]
    assert test.qualification_scope_binding['verified_episode_ids']==['L5_observed_armature02_v1'] and calls==[(str(lockpath),lb['sha256'])]
    assert f.D.dataset_binding(test)==identity
    # A re-bound but incorrect reservation must still fail before any array access.
    value['status']='completed';write(rp,value);auth['reservation_binding']['sha256']=sha(rp)
    monkeypatch.setattr(np,'load',lambda *a,**kw:pytest.fail('Invalid reservation opened arrays'))
    with pytest.raises(ValueError,match='reservation'):f.load(('test',),test_authorization=auth)


@pytest.mark.parametrize('case',['qualification_false','missing_check','boundary_bool','cadence','legacy_boundary','supervisor_audit','source_escape','duplicate_content','cross_split_group'])
def test_inherited_per_record_and_global_gates(fixture,monkeypatch,case):
    f=fixture;eid=f.design['data']['train_original8'][0]
    if case=='qualification_false':rebind_record(f,eid,monkeypatch,change_report=lambda r:r.update(qualified=False))
    elif case=='missing_check':rebind_record(f,eid,monkeypatch,change_report=lambda r:r['checks'].update(camera_valid=False))
    elif case=='boundary_bool':rebind_record(f,eid,monkeypatch,change_report=lambda r:r['evidence']['joint_action_derivation']['phase_derivation'].update(cross_boundary_label_indices=[True]))
    elif case=='cadence':rebind_record(f,eid,monkeypatch,change_arrays=lambda a:a['times'].__setitem__(slice(None),np.arange(100)*.1))
    elif case=='legacy_boundary':rebind_record(f,eid,monkeypatch,change_arrays=lambda a:a['actions'].__setitem__((0,3),0.))
    elif case=='supervisor_audit':rebind_record(f,eid,monkeypatch,change_report=lambda r:r['evidence']['joint_action_derivation']['phase_derivation'].update(observed_supervisor_sha256='0'*64))
    elif case=='source_escape':rebind_record(f,eid,monkeypatch,change_record=lambda r:r.update(path='../foreign.npz'))
    elif case=='duplicate_content':rebind_record(f,eid,monkeypatch,change_record=lambda r:r.update(sha256=f.records[f.design['data']['train_original8'][1]][1]['sha256']))
    else:rebind_record(f,eid,monkeypatch,change_record=lambda r:r.update(motion_family='independent_target_plan_R5'))
    with pytest.raises(ValueError):f.load()


def test_consumed_qualification_rebind_during_task_check_rejected(fixture,monkeypatch):
    f=fixture;from sim.auscultation_pi05_v6 import derive_task_dataset as task
    def replace(root,record,report,arrays):
        (Path(root)/record['qualification_path']).write_text('{}\n');return True
    monkeypatch.setattr(task,'verify_task_derivation',replace)
    with pytest.raises(ValueError,match='SHA mismatch'):f.load()
