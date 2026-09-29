"""Synthetic CPU-only bridge protocol and real owned-child regressions."""
import importlib.util
from pathlib import Path
import json
import sys
import pytest

MODULE=Path(__file__).parents[1]/'coverage_pilot_admission_bridge.py'

def bridge():
 assert MODULE.is_file(),'Independent bridge implementation absent'
 s=importlib.util.spec_from_file_location('bridge_test',MODULE);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_environment_drops_secrets_and_hides_gpu():
 m=bridge();env=m.child_environment(Path('/SYNTHETIC/source'))
 assert env['CUDA_VISIBLE_DEVICES']=='' and env['JAX_PLATFORMS']=='cpu'
 assert env['PYTHONNOUSERSITE']=='1' and env['PYTHONPATH']=='/SYNTHETIC/source'
 assert set(env)<=m.CHILD_ENV_KEYS

def test_strict_json_rejects_duplicate_and_nonfinite():
 m=bridge()
 for raw in (b'{"a":1,"a":2}',b'{"a":NaN}'):
  with pytest.raises(ValueError):m.read_json_bytes(raw)

def test_missing_lock_cannot_mint_admission(tmp_path,monkeypatch):
 m=bridge();request={'selection_lock':str(tmp_path/'missing'),'selection_lock_sha256':None,'manifest':str(tmp_path/'manifest')}
 monkeypatch.setattr(m,'verify_worker_environment',lambda:{'SYNTHETIC':True})
 monkeypatch.setattr(m,'verify_request',lambda q:dict(request=q))
 response=m.worker_response(request)
 assert response['status']=='not_ready' and response['admission'] is None
 assert response['selection_reconstructed'] is False

def synthetic_receipt(m,monkeypatch):
 # Protocol-only fixture: never a real selected model or qualification.
 contract={'SYNTHETIC':True}
 request={'selection_lock':'/SYNTHETIC/lock','selection_lock_sha256':'a'*64,'manifest':'/SYNTHETIC/manifest','manifest_sha256':'b'*64}
 admission={'artifact_sha256':{},'SYNTHETIC':True,'selection_lock_path':request['selection_lock'],'selection_lock_sha256':request['selection_lock_sha256'],'manifest_path':request['manifest'],'manifest_sha256':request['manifest_sha256']}
 response={'schema':m.RESPONSE_SCHEMA,'status':'verified','admission':admission,'selection_reconstructed':True,'request_sha256':m.digest(request),'source_contract':contract,
  'environment':dict(python=str(m.OPENPI_PYTHON),prefix=str(m.OPENPI_PREFIX),numpy='1.26.4',scipy='1.15.3',cuda_visible_devices='',jax_platforms='cpu',pid=100,parent_pid=99),
  'model_initialized':False,'service_contacted':False}
 receipt={'schema':m.RECEIPT_SCHEMA,'process_exit_code':0,'process_group_drained':True,'cancelled':False,'worker_response':response,'stdout_sha256':m.digest(response),'stdout_bytes':len(m.json_bytes(response)),
  'request':request,'request_sha256':m.digest(request),'source_contract':contract,'admission_sha256':m.digest(admission),'artifact_sha256_reverified':{},'stderr_bytes':0,'stderr_sha256':m.file_bytes_sha(b''),
  'command':[str(m.OPENPI_PYTHON),'-B',str(m.MODULE) if hasattr(m,'MODULE') else str(Path(m.__file__).resolve()),'--worker','--parent-pid','99'],
  'parent_pid':99,'process_identity':dict(pid=100,pgid=100,sid=100,start_ticks='1234'),'elapsed_s':1.1,'process_errors':[],'cleanup_signals':[],
  'sdk_numpy_unchanged':True,'private_configuration_used':False}
 monkeypatch.setattr(m,'source_contract',lambda:contract)
 monkeypatch.setattr(m,'verify_request',lambda a:contract)
 monkeypatch.setattr(m,'verify_admission_artifacts',lambda a:{})
 return receipt,admission


def test_saved_bridge_receipt_requires_zero_exit_and_actual_response(monkeypatch):
 m=bridge();receipt,admission=synthetic_receipt(m,monkeypatch)
 assert m.verify_receipt(receipt,admission)
 for key,value in [('process_exit_code',1),('process_exit_code',None),('process_group_drained',False),('stdout_sha256','bad'),('cancelled',True),
                   ('command',['/other/python']),('elapsed_s',float('nan')),('elapsed_s',99999),('process_errors',['bad']),('parent_pid',0),('cleanup_signals',['SIGTERM'])]:
  with pytest.raises(ValueError):m.verify_receipt(dict(receipt,**{key:value}),admission)


def test_receipt_rejects_rebound_request_and_environment(monkeypatch):
 import copy
 m=bridge();receipt,admission=synthetic_receipt(m,monkeypatch)
 for section,key,value in [('request','manifest','/SYNTHETIC/other'),('environment','numpy','1.26.0'),('environment','cuda_visible_devices','0'),('environment','pid',1000)]:
  changed=copy.deepcopy(receipt)
  target=changed['request'] if section=='request' else changed['worker_response']['environment']
  target[key]=value
  changed['stdout_sha256']=m.digest(changed['worker_response']);changed['stdout_bytes']=len(m.json_bytes(changed['worker_response']))
  with pytest.raises(ValueError):m.verify_receipt(changed,admission)

def test_every_bound_artifact_is_freshly_hashed(tmp_path):
 m=bridge();f=tmp_path/'actual';f.write_bytes(b'original')
 a={'artifact_sha256':{str(f):m.file_sha(f)}}
 assert m.verify_admission_artifacts(a)==a['artifact_sha256']
 f.write_bytes(b'changed')
 with pytest.raises(ValueError):m.verify_admission_artifacts(a)

@pytest.mark.skipif(sys.platform!='linux',reason='Actual /proc ownership semantics')
def test_real_cpu_process_output_limit_timeout_and_exit(tmp_path):
 m=bridge()
 # An owned diagnostic child only; no original selector/model/SDK/GPU.
 good=m.run_bounded([sys.executable,'-c','print("ok")'],b'',env=m.child_environment(tmp_path),cwd=tmp_path,timeout_s=3,stdout_limit=64,stderr_limit=64)
 assert good['exit_code']==0 and good['stdout']==b'ok\n' and good['process_group_drained']
 for code,limit,timeout in [('print("x"*1000)',16,3),('import time;time.sleep(30)',64,.15),('raise SystemExit(7)',64,3)]:
  r=m.run_bounded([sys.executable,'-c',code],b'',env=m.child_environment(tmp_path),cwd=tmp_path,timeout_s=timeout,stdout_limit=limit,stderr_limit=64)
  assert r['process_group_drained'] and (r['errors'] or r['exit_code']!=0)

@pytest.mark.skipif(sys.platform!='linux',reason='Actual /proc ownership semantics')
def test_real_child_startup_interrupt_is_reaped(tmp_path,monkeypatch):
 m=bridge();g=m.guard_module();original=g.process_identity;captured=[]
 def interrupted(pid):
  captured.append(pid)
  if len(captured)==1:raise KeyboardInterrupt('SYNTHETIC interruption immediately after Popen')
  return original(pid)
 monkeypatch.setattr(g,'process_identity',interrupted);monkeypatch.setattr(m,'guard_module',lambda:g)
 result=m.run_bounded([sys.executable,'-c','import time;time.sleep(30)'],b'',env=m.child_environment(tmp_path),cwd=tmp_path,timeout_s=3)
 assert result['process_group_drained'] and result['exit_code'] is not None and result['errors']
 assert not Path('/proc',str(captured[0])).exists()

@pytest.mark.skipif(sys.platform!='linux',reason='Actual signals / owned child')
def test_late_signal_in_final_close_cancels(tmp_path,monkeypatch):
 import os,signal
 m=bridge();factory=m.selectors.DefaultSelector
 def selector():
  obj=factory();close=obj.close
  def interrupted_close():
   os.kill(os.getpid(),signal.SIGTERM);close()
  obj.close=interrupted_close;return obj
 monkeypatch.setattr(m.selectors,'DefaultSelector',selector)
 r=m.run_bounded([sys.executable,'-c','print("ok")'],b'',env=m.child_environment(tmp_path),cwd=tmp_path,timeout_s=3)
 assert r['exit_code']==0 and r['process_group_drained'] and r['cancelled'] and r['deferred_signals']==['SIGTERM']
 with pytest.raises(m.AdmissionInterrupted) as caught:m.raise_process_failure(ValueError('cancelled'),{'SYNTHETIC':True},{'SYNTHETIC':True},['SYNTHETIC'],r)
 assert caught.value.receipt['process_exit_code']==0 and caught.value.receipt['process_group_drained']

@pytest.mark.parametrize('exit_code,stdout,errors,cancelled',[(7,b'',[],False),(0,b'invalid JSON',[],False),(None,b'',['timeout'],False),(0,b'{}',[],True)])
def test_failed_worker_keeps_actual_process_evidence(tmp_path,monkeypatch,exit_code,stdout,errors,cancelled):
 m=bridge();manifest=tmp_path/'manifest.json';manifest.write_text('{}');lock=tmp_path/'lock.json';lock.write_text('{}')
 monkeypatch.setattr(m,'source_contract',lambda:{'SYNTHETIC':True});monkeypatch.setattr(m,'root_path',lambda:tmp_path)
 monkeypatch.setattr(m,'verify_request',lambda r:None)
 monkeypatch.setattr(m,'run_bounded',lambda *a,**k:dict(exit_code=exit_code,stdout=stdout,stderr=b'bounded public worker error',errors=errors,cancelled=cancelled,process_group_drained=True,signals=[],elapsed_s=.1,process_identity={'pid':123}))
 with pytest.raises(m.AdmissionInterrupted if cancelled else m.AdmissionFailure) as caught:m.selected_admission_with_receipt(lock,'a'*64,manifest)
 r=caught.value.receipt
 assert r['process_exit_code']==exit_code and r['process_group_drained'] is True
 assert r['stdout_sha256']==m.file_bytes_sha(stdout) and r['process_identity']=={'pid':123}
 assert r['stderr_excerpt_utf8']=='bounded public worker error' and not r['passed']

def test_parent_cancellation_propagates_beyond_validation_catches():
 m=bridge();run=dict(exit_code=-2,process_group_drained=True,cancelled=True,process_identity={'pid':123},errors=['KeyboardInterrupt: actual parent cancellation'],signals=[],elapsed_s=.1,stdout=b'',stderr=b'')
 with pytest.raises(KeyboardInterrupt) as caught:
  m.raise_process_failure(ValueError('cancelled'),{'SYNTHETIC':True},{'SYNTHETIC':True},['SYNTHETIC'],run)
 assert caught.value.receipt['cancelled'] is True
 assert not isinstance(caught.value,ValueError)
