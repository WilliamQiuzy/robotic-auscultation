"""Synthetic optimizer/checkpoint invariants; never actual model evidence."""
import importlib
from pathlib import Path
import json
import hashlib
import numpy as np
import pytest

def module():
 p=Path(__file__).parents[1]/'coverage_pilot_train.py';assert p.is_file(),'Pilot trainer not implemented yet'
 return importlib.import_module('sim.auscultation_pi05_v7.coverage_pilot_train')

def test_step_zero_exact_arrays_preserves_rng_and_optimizer():
 m=module();a=dict(leaf_00000=np.ones((2,3),np.float32),leaf_00001=np.asarray(4000,np.int32),jax_rng=np.array([123,456],np.uint32))
 b={k:v.copy() for k,v in a.items()};assert m.assert_array_identity(a,b)['passed']
 for key in a:
  bad={k:v.copy() for k,v in b.items()};bad[key].flat[0]+=1
  with pytest.raises(ValueError):m.assert_array_identity(a,bad)
 bad={k:v.copy() for k,v in b.items()};bad['leaf_00000']=bad['leaf_00000'].astype(np.float64)
 with pytest.raises(ValueError):m.assert_array_identity(a,bad)

def test_optimizer_age_is_not_local_step():
 m=module();contract=[dict(path='[1][1][0].count',shape=[],dtype='int32')]
 m.verify_optimizer_age({'leaf_00000':np.asarray(4007,np.int32)},contract,7)
 with pytest.raises(ValueError):m.verify_optimizer_age({'leaf_00000':np.asarray(7,np.int32)},contract,7)
 with pytest.raises(ValueError):m.verify_optimizer_age({'leaf_00000':np.asarray(4007,np.int64)},contract,7)

def checkpoint(m,root,step,config):
 p=root/'model'/('step-%012d'%step);p.mkdir(parents=True)
 arrays=dict(leaf_00000=np.asarray(4000+step,np.int32),jax_rng=np.array([1,2],np.uint32));np.savez(p/'arrays.npz',**arrays)
 meta=dict(schema='wam_pi05_resume_v1',step=step,optimizer_step=4000+step,schedule_cursor=step*4,data_hash='data',config=config,normalization={},arrays_sha256=m.parent._sha(p/'arrays.npz'),leaf_contract=[dict(path='[1][1][0].count',shape=[],dtype='int32')],sampler_rng_state=dict(cursor=step*4,schedule_sha256='s'*64))
 (p/'metadata.json').write_text(json.dumps(meta));(p/'COMPLETE.json').write_text(json.dumps(dict(metadata_sha256=m.parent._sha(p/'metadata.json'))));return p

def test_resume_latest_and_original_namespace_only(tmp_path):
 m=module();run=tmp_path/'run';config=dict(training_run_path=str(run.resolve()),schedule_binding=dict(sha256='s'*64),total_steps=10,candidate_steps=[0,5,10])
 old=checkpoint(m,run,0,config);latest=checkpoint(m,run,5,config)
 assert m.resolve_same_run_resume(run,latest,data_hash='data',config=config,normalization={})['step']==5
 with pytest.raises(ValueError):m.resolve_same_run_resume(run,old,data_hash='data',config=config,normalization={})
 import shutil
 copied=tmp_path/'copied';shutil.copytree(run,copied)
 with pytest.raises(ValueError):m.resolve_same_run_resume(copied,copied/'model'/latest.name,data_hash='data',config=config,normalization={})
 (run/'PILOT_SELECTION_LOCKED.json').write_text('{}')
 with pytest.raises(ValueError):m.resolve_same_run_resume(run,latest,data_hash='data',config=config,normalization={})

def test_corrupt_newest_cannot_fall_back(tmp_path):
 m=module();run=tmp_path/'run';config=dict(training_run_path=str(run.resolve()),schedule_binding=dict(sha256='s'*64),total_steps=10,candidate_steps=[0,5,10])
 old=checkpoint(m,run,0,config);new=checkpoint(m,run,5,config);(new/'arrays.npz').write_bytes(b'bad')
 with pytest.raises(ValueError):m.resolve_same_run_resume(run,old,data_hash='data',config=config,normalization={})
 with pytest.raises(ValueError):m.resolve_same_run_resume(run,new,data_hash='data',config=config,normalization={})

def test_resume_cursor_cannot_advance_or_reset(tmp_path):
 m=module();run=tmp_path/'run';config=dict(training_run_path=str(run.resolve()),schedule_binding=dict(sha256='s'*64),total_steps=10,candidate_steps=[0,5,10])
 p=checkpoint(m,run,5,config);x=json.loads((p/'metadata.json').read_text());x['schedule_cursor']=0;(p/'metadata.json').write_text(json.dumps(x));(p/'COMPLETE.json').write_text(json.dumps(dict(metadata_sha256=m.parent._sha(p/'metadata.json'))))
 with pytest.raises(ValueError):m.resolve_same_run_resume(run,p,data_hash='data',config=config,normalization={})

def test_batch_suffix_and_optimizer_rng_serialization_identical(tmp_path):
 import importlib.util
 spec=importlib.util.spec_from_file_location('schedule_test_fixture',Path(__file__).with_name('test_coverage_pilot_schedule.py'));fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
 inputs,schedule_module=fixture.inputs,fixture.module
 m=module();smod=schedule_module();records,ids=inputs();s=smod.build_schedule(records,ids)
 def advance(weights,momentum,age,rng,start,end):
  visited=[]
  for step in range(start,end):
   idx=smod.batch_indices(s,step);noise=rng.normal(size=4);g=(idx*.0001+noise).mean();momentum=.9*momentum+.1*g;weights-=.00005*momentum;age+=1;visited.extend(idx)
  return weights,momentum,age,visited
 rng=np.random.default_rng(88);full=advance(1.,.2,4000,rng,0,35)
 rng=np.random.default_rng(88);part=advance(1.,.2,4000,rng,0,13);state=json.loads(json.dumps(rng.bit_generator.state))
 restored=np.random.default_rng();restored.bit_generator.state=state;tail=advance(*part[:3],restored,13,35)
 assert full[:3]==tail[:3] and full[3]==part[3]+tail[3]

def test_training_output_rejects_entire_frozen_snapshot_before_loading(tmp_path,monkeypatch):
 from types import SimpleNamespace
 m=module();snapshot=tmp_path/'sealed';fake=snapshot/'sim/auscultation_pi05_v7/coverage_pilot_train.py';fake.parent.mkdir(parents=True);fake.touch()
 monkeypatch.setattr(m,'__file__',str(fake));monkeypatch.setattr(m,'load_inputs',lambda *_:pytest.fail('Must reject before reading data'))
 a=SimpleNamespace(output=snapshot/'sim/auscultation_pi05_v6/new_run',manifest=tmp_path/'dataset/manifest.json',bootstrap=tmp_path/'oldrun/model/step-000000004000',resume=None,verify_only=True)
 with pytest.raises(ValueError,match='overlaps'):m.run(a)

def test_failed_selection_reservation_blocks_resume(tmp_path):
 m=module();run=tmp_path/'run';config=dict(training_run_path=str(run.resolve()),schedule_binding=dict(sha256='s'*64),total_steps=10,candidate_steps=[0,5,10])
 p=checkpoint(m,run,5,config);(run/'pilot_selection').mkdir()
 with pytest.raises(ValueError):m.resolve_same_run_resume(run,p,data_hash='data',config=config,normalization={})
