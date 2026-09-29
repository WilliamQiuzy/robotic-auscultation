"""Synthetic bridge boundary checks and exact inherited physics AST parity."""
import ast
import copy
import importlib
from pathlib import Path
import sys
import pytest
P=Path(__file__).parents[1]
def parsed(name):return ast.parse((P/name).read_text())
def functions(name):return {n.name:n for n in parsed(name).body if isinstance(n,ast.FunctionDef)}
def mod(name):
 if sys.platform=='darwin':pytest.skip('Inherited native/runtime imports run in cloud CPU tree')
 return importlib.import_module('sim.auscultation_pi05_v7.'+name)

def test_original_selection_algorithm_not_copied_or_monkeypatched():
 tree=parsed('coverage_pilot_bridge_runtime_contract.py')
 selected=functions('coverage_pilot_bridge_runtime_contract.py')['selected_admission_with_receipt']
 assert len(selected.body)==2 and isinstance(selected.body[1],ast.Return)
 assert ast.unparse(selected.body[1].value)=='B.selected_admission_with_receipt(lock_path, lock_sha, manifest)'
 assert not any(isinstance(n,ast.Name) and n.id=='verify_selection_lock' for n in ast.walk(tree))
 for name in ('coverage_pilot_admission_bridge.py','coverage_pilot_bridge_runtime_contract.py'):
  assert not any(isinstance(x,ast.Attribute) and x.attr in ('__version__','selected_admission','verify_selection_lock') for n in ast.walk(parsed(name)) if isinstance(n,ast.Assign) for target in n.targets for x in ast.walk(target))


def test_all_copied_replay_functions_identical_except_new_schema():
 old=functions('audit_coverage_pilot_headroom_actions.py');new=functions('audit_coverage_pilot_bridge_headroom_actions.py')
 for name,node in old.items():
  if name in ('audit_episode','audit_source_paths','verify_connection_snapshot','main'):continue
  actual=copy.deepcopy(new[name])
  for n in ast.walk(actual):
   if isinstance(n,ast.Constant) and n.value=='wam_v7_coverage_pilot_bridge_headroom_replay_v1':n.value='wam_v7_coverage_pilot_headroom_replay_v1'
  assert ast.dump(node)==ast.dump(actual),name


def test_assessment_all_physical_adc_and_completion_checks_identical():
 old=functions('assess_coverage_pilot_headroom_trials.py')['inherited_check_documents']
 actual=copy.deepcopy(functions('assess_coverage_pilot_bridge_headroom_trials.py')['inherited_check_documents'])
 for n in ast.walk(actual):
  if isinstance(n,ast.Constant) and n.value=='wam_v7_coverage_pilot_bridge_headroom_replay_v1':n.value='wam_v7_coverage_pilot_headroom_replay_v1'
 assert ast.dump(old)==ast.dump(actual)


def test_runner_capture_only_adds_receipt_and_actual_identity():
 old=functions('run_coverage_pilot_headroom_isaac.py')['capture']
 new=copy.deepcopy(functions('run_coverage_pilot_bridge_headroom_isaac.py')['capture'])
 for node in ast.walk(new):
  if isinstance(node,ast.ImportFrom) and node.module=='coverage_pilot_bridge_runtime_contract':node.module='coverage_pilot_runtime_contract'
  if isinstance(node,ast.Call):node.keywords=[k for k in node.keywords if k.arg!='admission_bridge_receipt']
 assert ast.dump(old)==ast.dump(new)
 tree=parsed('run_coverage_pilot_bridge_headroom_isaac.py')
 assert not any(isinstance(n,ast.ClassDef) for n in ast.walk(tree))
 main=functions('run_coverage_pilot_bridge_headroom_isaac.py')['main']
 checked=next(i for i,n in enumerate(main.body) if any(isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='verify_coverage_pilot_inputs' for x in ast.walk(n)))
 kit=next(i for i,n in enumerate(main.body) if isinstance(n,ast.ImportFrom) and n.module=='isaacsim')
 assert checked<kit


def test_bridge_assessor_requires_explicit_bridge_flag():
 m=mod('assess_coverage_pilot_bridge_headroom_trials')
 value=dict(schema='wam_v7_coverage_pilot_bridge_headroom_action_audit_v1',passed=True,errors=[],coverage_pilot_queries_checked=1,
  pilot_binding_verified=True,selected_checkpoint_lock_verified=True,connection_source_snapshot_verified=True,pilot_request_bindings_verified=True,
  inherited_action_audit=dict(schema='wam_v7_coverage_pilot_bridge_headroom_replay_v1',passed=True))
 with pytest.raises(ValueError):m.verify_outer_audit(value)
 value['admission_bridge_verified']=False
 with pytest.raises(ValueError):m.verify_outer_audit(value)
 value['admission_bridge_verified']=True
 assert m.verify_outer_audit(value)==value['inherited_action_audit']
 for bad in ('wam_v7_coverage_pilot_headroom_action_audit_v1','wam_v7_headroom_learned_action_audit_v1'):
  with pytest.raises(ValueError):m.verify_outer_audit(dict(value,schema=bad))


def test_contract_source_map_binds_original_and_new_execution(monkeypatch):
 m=mod('coverage_pilot_bridge_runtime_contract')
 monkeypatch.setattr(m.B,'source_contract',lambda:{'SYNTHETIC':True})
 d=m.runtime_connection_contract()
 assert len(d['connection_source_sha256'])==5 and len(d['original_connection_source_sha256'])==4 and len(d['pilot_source_sha256'])==7
 assert d['executed_runner_module']=='sim.auscultation_pi05_v7.run_coverage_pilot_bridge_headroom_isaac'
 assert d['headroom_implementation']['runtime_execution_contract']['jerk_rad_s3']==[7.99999]*7
 assert d['headroom_implementation']['original_whole_stream_gate']['joint_jerk_rad_s3']==8.


def test_new_contract_admission_delegates_original_result_and_receipt(monkeypatch):
 m=mod('coverage_pilot_bridge_runtime_contract');calls=[]
 monkeypatch.setattr(m.B,'selected_admission_with_receipt',lambda *args:(calls.append(args) or ({'SYNTHETIC_admission':True},{'SYNTHETIC_receipt':True})))
 assert m.selected_admission('lock','sha','manifest')=={'SYNTHETIC_admission':True}
 assert calls==[('lock','sha','manifest')]


def test_actual_runner_classes_unchanged():
 r=mod('run_coverage_pilot_bridge_headroom_isaac');c,e=r.physical_classes()
 assert c.__module__=='sim.auscultation_pi05_v7.headroom_learned_controller'
 assert e.__module__=='sim.auscultation_pi05_v7.headroom_execution'

@pytest.mark.parametrize('name',['audit_coverage_pilot_bridge_headroom_actions','assess_coverage_pilot_bridge_headroom_trials'])
def test_bridge_cancel_escapes_audit_validation_handlers(tmp_path,monkeypatch,name):
 import json
 m=mod(name);receipt={'SYNTHETIC_cancelled':True}
 admission=dict(selection_lock_path='/SYNTHETIC/lock',selection_lock_sha256='a'*64,manifest_path='/SYNTHETIC/manifest')
 for file,value in [('outcome.json',{'pilot_runtime_binding':admission}),('collection_result.json',{}),('learned_actions_audit.json',{'pilot_runtime_binding':admission}),('observed_supervisor_audit.json',{})]:
  (tmp_path/file).write_text(json.dumps(value))
 def interrupted(*args):raise m.C.B.AdmissionInterrupted('SYNTHETIC parent cancellation',receipt)
 monkeypatch.setattr(m.C,'selected_admission',interrupted)
 with pytest.raises(m.C.B.AdmissionInterrupted) as caught:
  if name.startswith('audit'):m.audit_episode(tmp_path,expected_checkpoint_metadata_sha256='b'*64)
  else:
   monkeypatch.setattr(m,'verify_outer_audit',lambda actions:{})
   m.assess_case(dict(case_id='SYNTHETIC',directory=str(tmp_path),points=['R4'],scope='full'),'b'*64)
 assert caught.value.receipt is receipt
