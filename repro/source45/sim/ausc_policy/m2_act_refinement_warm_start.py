"""V9 initialization and data-only collection from exact ACT10 selected tensors.

Only an isolated original Source41 official CPU child admits the historical
model. Its tensor-only export is initialization provenance, never a V9 result.
"""
from copy import deepcopy
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys

import numpy as np

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from sim.ausc_policy import m2_given_contact_runtime as rt
    from sim.ausc_policy import m2_act_phase_warm_start as shared
    from sim.ausc_policy import m2_act_posture_head as head
else:
    from . import m2_given_contact_runtime as rt
    from . import m2_act_phase_warm_start as shared
    from . import m2_act_posture_head as head

need=rt.need
tensor_proofs=shared.tensor_proofs
PROBE_TRAIN_INDICES=list(shared.PROBE_TRAIN_INDICES)
OFFICIAL_PYTHON=shared.OFFICIAL_PYTHON
SCHEMA='m2_act10_selected_cpu_initialization_v9'
EQUIVALENCE_SCHEMA='m2_act_refinement_initialization_equivalence_v9'
COLLECTION_SCHEMA='m2_act10_collection_only_bootstrap_v1'
ORIGIN='ACT10 V8 selected17000 of completed20000; original41 CPU authenticated'
_ROOT='/opt/wam-runtime/code/wam-act-recovery-20261106-v41'
_COMMIT='c6590db44ff94448a7da34488b5ceaa7810f1134'
SOURCE41=dict(root=_ROOT,code_commit=_COMMIT,manifest_files=1706,
    code_commit_file=dict(path=_ROOT+'/CODE_COMMIT',bytes=41,sha256=hashlib.sha256((_COMMIT+'\n').encode()).hexdigest()),
    source_manifest=dict(path=_ROOT+'/SOURCE_MANIFEST.json',bytes=295288,
        sha256='f11af1049afabcd7c1d17bf724b86f6d3157897b01ae4a6c73db749bfebdd1ff'))
CONTROL=dict(path='/opt/wam-runtime/fast-wam/diagnostics/act10-diverse-v8-training-20261107-v1/COMPLETE.json',bytes=2757,
    sha256='68422ea72f34087ee6c19c78a034e03fded828ab130f29ff3510cdcf5318247c')
_BASE='/opt/wam-runtime/fast-wam/runs/ausc_policy/act10-diverse-v8-train-20261107-v1/output/'
COMPLETE=dict(path=_BASE+'COMPLETE.json',bytes=4016445,sha256='f54c68cf03c883610617cf9beaac0a8d8f20d59d473e1458b07ded031730ae5e')
WEIGHTS=dict(path=_BASE+'best_auxiliary_act.pt',bytes=11805569,sha256='9de4ffff75bdf01f3855a872bb0315256ee23c240a6c5c3d37221e4cb35c764d')
SELECTED=dict(path=_BASE+'SELECTED.json',bytes=3752763,sha256='55b09e463e8101771cea8cfe819f200e238dd537574f7b60ef50a2aaf15f3146')
CPU_EQUIVALENCE=dict(path=_BASE+'CPU_EQUIVALENCE.json',bytes=15458,sha256='5909b1e21ba17b201792012951f10b24f21d60a722e26e1264f2528c4fe0fdb2')
OWNER=dict(path=str(Path(_BASE).parent/'result.json'),bytes=8471,sha256='20d8c9c2c919f85507db8598b39da0d3f6f8513361df747966dcb24e07227195')
PARENT_MODEL=dict(checkpoint_schema='m2_act_auxiliary_act_pilot_v8',data_version=5,model_optimizer_version=8,evaluation_candidates=40)
_IMPORT_MARKERS=('_posture_training_initialization','_act6_initialization','_act7_initialization','_act8_initialization','_act9_initialization','_act9_collection_import')


def _regular_unaliased(path):
    need(all(not p.is_symlink() for p in (path,*path.parents)),'Original41 source aliases forbidden')
    info=path.lstat()
    need(stat.S_ISREG(info.st_mode) and info.st_nlink==1,'Regular original41 source without hardlink aliases required')


def require_archive():
    """Hash every regular Source41 file, including controls; reject all aliases."""
    root=Path(SOURCE41['root']);need(root.is_absolute() and root.resolve()==root,'Canonical original41 source required')
    for ref in (SOURCE41['source_manifest'],SOURCE41['code_commit_file']):
        _regular_unaliased(Path(ref['path']));rt.bound_path(ref)
    rows=rt.read_bound(SOURCE41['source_manifest'])
    need(isinstance(rows,list) and len(rows)==SOURCE41['manifest_files'],'Complete original41 inventory required')
    need((root/'CODE_COMMIT').read_text()==SOURCE41['code_commit']+'\n','Original41 commit differs')
    names=set()
    for row in rows:
        name=row.get('file');relative=Path(name) if isinstance(name,str) else None
        need(set(row)=={'file','bytes','sha256'} and relative is not None and name==relative.as_posix() and
            not relative.is_absolute() and relative.parts and '..' not in relative.parts and name not in names and
            name not in ('SOURCE_MANIFEST.json','CODE_COMMIT'),'Unique safe original41 file paths required')
        path=root/relative;_regular_unaliased(path)
        rt.bound_path(dict(path=str(path),bytes=row['bytes'],sha256=row['sha256']));names.add(name)
    actual=set()
    for path in root.rglob('*'):
        need(not path.is_symlink(),'No original41 symlinks allowed')
        mode=path.lstat().st_mode
        need(stat.S_ISDIR(mode) or stat.S_ISREG(mode),'Only directories/regular files allowed in original41')
        if stat.S_ISREG(mode):_regular_unaliased(path);actual.add(path.relative_to(root).as_posix())
    need(actual==names|{'SOURCE_MANIFEST.json','CODE_COMMIT'},'Unknown or omitted original41 files')


def _validate_owner(value):
    need(value.get('run_id')=='act10-diverse-v8-train-20261107-v1' and value.get('code_commit')==SOURCE41['code_commit'] and
        value.get('completed') is True and value.get('gpu_released') is True,'Exact normally closed ACT10 owner required')
    stage=value.get('stage',{})
    need(stage.get('stage')=='m2_given_contact_expanded_pilot_train' and stage.get('passed') is True and
        type(stage.get('exit_code')) is int and stage['exit_code']==0 and stage.get('process_group_drained') is True and
        all(stage.get(k)==[] for k in ('errors','signals','deferred_signals')),'ACT10 owner must drain normally without signals')


def _parent_graph():
    control=rt.read_bound(CONTROL);complete=rt.read_bound(COMPLETE);cpu=rt.read_bound(CPU_EQUIVALENCE)
    selected=rt.read_bound(SELECTED);_validate_owner(rt.read_bound(OWNER))
    need(control.get('schema')=='act10_diverse_v8_single_owned_training_execution_v1' and control.get('complete') is True and
        control.get('normal_owner') is True and control.get('training')==COMPLETE and control.get('training_owner')==OWNER and
        control.get('model')==dict(checkpoint=WEIGHTS,checkpoint_schema=PARENT_MODEL['checkpoint_schema'],complete=COMPLETE) and
        control.get('cpu_equivalence')==CPU_EQUIVALENCE and control.get('selection')==SELECTED and
        control.get('completed_steps')==20000 and control.get('selected_step')==17000 and control.get('evaluation_candidates')==40 and
        control.get('source')==dict(code_commit=SOURCE41['code_commit'],files=SOURCE41['manifest_files'],manifest=SOURCE41['source_manifest'],root=SOURCE41['root']),
        'Pinned ACT10 normally closed training/model/source graph required')
    need(complete.get('schema')==PARENT_MODEL['checkpoint_schema'] and complete.get('complete') is True and
        complete.get('completed_steps')==20000 and complete.get('selected_step')==17000 and complete.get('best')==WEIGHTS and
        complete.get('cpu_equivalence')==CPU_EQUIVALENCE and cpu.get('schema')=='m2_act_auxiliary_act_cpu_equivalence_v8' and
        cpu.get('passed') is True and cpu.get('checkpoint')==WEIGHTS and cpu.get('selection')==SELECTED and cpu.get('selected_step')==17000,
        'Exact ACT10 V8 selected17000 of completed20000 and own CPU equivalence required')
    need(selected.get('schema')=='m2_act_auxiliary_act_selection_v8' and selected.get('selected_step')==17000 and selected.get('best')==WEIGHTS and
        selected.get('completed_steps')==20000 and all(selected.get(k)==complete.get(k) for k in
        ('inputs','protocol','initialization','scope','native_scope','optimization_scope','diverse_inventory','measured_counts',
         'selected_predictions','selection_policy','terminal_pool_inventory','evaluation_candidates','optimization','selected_optimization')) and
        all(value.get('data_version')==5 and value.get('model_optimizer_version')==8 for value in (selected,complete,cpu)) and
        [row.get('step') for row in complete.get('evaluation_candidates',[])]==list(range(500,20001,500)) and
        isinstance(complete.get('optimization'),dict) and isinstance(selected.get('selected_optimization'),dict),
        'Original V8 all40/selection/optimizer-count metadata must remain unchanged')
    root=Path(SOURCE41['root']);code=selected['inputs']['code']
    need(code.get('code_commit')==SOURCE41['code_commit'] and code.get('files') and all(
        Path(ref['path'])==root/'sim/ausc_policy'/name for name,ref in code['files'].items()),'ACT10 original41 numerical origins required')
    software=rt.read_bound(selected['inputs']['software']);vendor=software.get('vendor',{})
    need(vendor.get('root')==rt.VENDOR_ROOT and vendor.get('upstream_commit')==rt.UPSTREAM_COMMIT and
        software.get('prefix')==rt.VENDOR_ROOT+'/.venv' and software.get('source_bindings') and
        all(Path(path).is_relative_to(root) for path in software['source_bindings']) and
        vendor.get('files',{}).get('src/lerobot/policies/act/modeling_act.py',{}).get('sha256')==
        '83d954f79ccd3eaa6774dbea92524939c8ac08359bb355ba7cd37b8601e1e3af','Official ACT10 CPU vendor/environment/origins required')
    return selected,rt.read_bound(selected['inputs']['dataset'])


def validate_receipt(ref,*,normalization=None,posture_head=None):
    """Metadata-only admission; load tensor bytes only in the import kernel."""
    value=rt.read_bound(ref);selected,native=_parent_graph();inputs=value.get('inputs',{})
    need(value.get('schema')==SCHEMA and value.get('complete') is True and value.get('origin')==ORIGIN and value.get('source41')==SOURCE41 and
        value.get('completed_steps')==20000 and value.get('selected_step')==17000 and value.get('parent_model')==PARENT_MODEL and
        value.get('historical_optimization')==dict(completed=selected['optimization'],selected=selected['selected_optimization']) and
        value.get('parameter_count')==2658823 and value.get('model')==rt.shared.native_spec() and value.get('task_contract')==rt.task_contract() and
        value.get('optimizer_imported') is False and value.get('scheduler_imported') is False,'Exact typed ACT10 V8 to V9 initialization required')
    need(inputs==dict(control=CONTROL,complete=COMPLETE,weights=WEIGHTS,cpu_equivalence=CPU_EQUIVALENCE,selection=SELECTED,owner=OWNER,
        software=selected['inputs']['software'],normalization=native['normalization'],source_manifest=SOURCE41['source_manifest']),
        'All pinned ACT10 initialization provenance references required')
    need(rt.validate_normalization(value.get('normalization',{}))==rt.read_bound(native['normalization']) and
        value.get('posture_head')==selected['protocol']['posture_head'],'ACT10 original native normalization/head differs')
    head.validate_spec(value['posture_head'])
    need(value['posture_head']['schema']==head.SCHEMA_V2 and value['posture_head']['normalization']==value['normalization'],
        'Unchanged original posture head v2/native normalization required')
    if normalization is not None:need(value['normalization']==normalization,'V9 normalization differs from ACT10')
    if posture_head is not None:need(value['posture_head']==posture_head,'V9 posture head differs from ACT10')
    probes=value.get('probes',{});obs=probes.get('observations',{})
    need(probes.get('split')=='train' and probes.get('dataset')==selected['inputs']['dataset'] and probes.get('indices')==PROBE_TRAIN_INDICES and
        set(obs)=={'observation.state','observation.environment_state'} and np.asarray(obs['observation.state']).shape==(4,7) and
        np.asarray(obs['observation.environment_state']).shape==(4,27) and np.asarray(probes.get('raw_predictions')).shape==(4,20,7) and
        all(np.isfinite(np.asarray(v)).all() for v in [*obs.values(),probes['raw_predictions']]),'Fixed actual native TRAIN-only probes required')
    need(value.get('execution')==dict(device='cpu',intraop_threads=1,interop_threads=1,cuda_visible_devices=''),
        'Single-thread CUDA-hidden original41 CPU export required')
    proofs=value.get('tensor_proofs')
    need(isinstance(proofs,dict) and len(proofs)==54 and all(isinstance(k,str) and isinstance(p,dict) and
        set(p)=={'shape','dtype','bytes','sha256'} and isinstance(p['shape'],list) and all(type(n) is int and n>=0 for n in p['shape']) and
        isinstance(p['dtype'],str) and type(p['bytes']) is int and p['bytes']>=0 and
        isinstance(p['sha256'],str) and re.fullmatch('[0-9a-f]{64}',p['sha256']) for k,p in proofs.items()),'All54 tensor bit proofs required')
    rt._reference_shape(value.get('state_dict'))
    need(set(value['state_dict'])=={'path','bytes','sha256'} and type(value['state_dict']['bytes']) is int and value['state_dict']['bytes']>0,
        'Full tensor-only packet reference required')
    return value


def protocol(ref):
    value=validate_receipt(ref)
    return dict(schema=SCHEMA,receipt=deepcopy(ref),inputs=deepcopy(value['inputs']),method='ACT10 V8 selected17000 full state_dict -> V9',
        final_linear_reset=False,optimizer='fresh AdamW',optimizer_initial_step=0,optimizer_imported=False,scheduler_imported=False,
        initial_tensor_proofs=deepcopy(value['tensor_proofs']),original_evaluations='initialization provenance only; excluded from V9 evaluations')


def _import_policy(torch,policy,ref,value,*,collection_only):
    need(not any(hasattr(policy,k) for k in _IMPORT_MARKERS),'V9 import cannot follow old initialization/reset/collection or be repeated')
    need(getattr(policy,'_posture_head_spec',None)==value['posture_head'],'Exactly installed original head v2 required before import')
    state=torch.load(io.BytesIO(rt.bound_path(value['state_dict']).read_bytes()),map_location='cpu',weights_only=True)
    need(tensor_proofs(state)==value['tensor_proofs'] and len(state)==54,'All54 authenticated tensor-only bytes required')
    current=policy.state_dict()
    need(set(state)==set(current) and all(state[k].shape==current[k].shape and state[k].dtype==current[k].dtype for k in state),
        'Exact unchanged architecture tensor names/shapes/dtypes required')
    policy.load_state_dict(state,strict=True);actual=tensor_proofs(policy.state_dict())
    need(actual==value['tensor_proofs'],'Bit-identical imported tensors required')
    policy.eval();device=next(policy.parameters()).device
    batch={k:torch.tensor(v,dtype=torch.float32,device=device) for k,v in value['probes']['observations'].items()}
    with torch.no_grad():prediction=policy.predict_action_chunk(batch).detach().cpu().numpy().astype(np.float64)
    normalization=value['normalization'];prediction=prediction*np.asarray(normalization['action_std'])+np.asarray(normalization['action_mean'])
    expected=np.asarray(value['probes']['raw_predictions'],dtype=np.float64)
    need(prediction.shape==expected.shape and np.isfinite(prediction).all() and
        np.all(abs(prediction-expected)<=1e-6+1e-5*np.maximum(abs(prediction),abs(expected))),'V9 initial prediction equivalence to actual ACT10 failed')
    result=dict(schema=EQUIVALENCE_SCHEMA,passed=True,initialization=deepcopy(ref),initial_tensor_proofs=actual,
        prediction_max_abs_rad=float(abs(prediction-expected).max()),prediction_rtol=1e-5,prediction_atol=1e-6,prediction_device=str(device),
        probe_split='train',probe_indices=list(PROBE_TRAIN_INDICES),optimizer_imported=False,scheduler_imported=False,final_linear_reset=False)
    if collection_only:
        result.update(collection_only=True,optimizer_created=False);policy._act9_collection_import=deepcopy(result)
    else:
        result['optimizer_step']=0;policy._act9_initialization=deepcopy(result)
    return result


def initialize_policy(torch,policy,ref,*,normalization,posture_head):
    value=validate_receipt(ref,normalization=normalization,posture_head=posture_head)
    return _import_policy(torch,policy,ref,value,collection_only=False)


def fresh_optimizer(torch,policy,*,lr,weight_decay):
    need(hasattr(policy,'_act9_initialization') and not hasattr(policy,'_act9_collection_import') and lr==1e-4 and weight_decay==1e-4,
        'Authenticated V9 training initialization and exact fresh AdamW required')
    optimizer=torch.optim.AdamW(policy.parameters(),lr=lr,weight_decay=weight_decay,betas=(.9,.999),eps=1e-8)
    need(not optimizer.state,'Fresh AdamW step0 required')
    return optimizer,dict(name='AdamW',initial_step=0,initial_state_entries=0,lr=lr,weight_decay=weight_decay,betas=[.9,.999],eps=1e-8,
        optimizer_imported=False,scheduler_imported=False)


def make_collection_policy(initialization_ref):
    """Source41-selected data-only bootstrap; never normal model admission."""
    value=validate_receipt(initialization_ref)
    torch,policy,software=rt.make_model(dict(lerobot_root=rt.VENDOR_ROOT),device='cpu')
    head.install(policy,value['posture_head'])
    equivalence=_import_policy(torch,policy,initialization_ref,value,collection_only=True)
    packet=dict(schema=COLLECTION_SCHEMA,collection_only=True,benchmark_model=False,optimizer_created=False,
        initialization=deepcopy(initialization_ref),normalization=deepcopy(value['normalization']),posture_head=deepcopy(value['posture_head']),
        parent_model=deepcopy(value['parent_model']),historical_optimization=deepcopy(value['historical_optimization']),
        source_evidence=dict(parent=deepcopy(SOURCE41),current=dict(software=software,helper=rt.reference(Path(__file__).resolve()))),
        tensor_proofs=deepcopy(equivalence['initial_tensor_proofs']),equivalence=equivalence)
    return torch,policy,packet


# This child imports only the authenticated original41 numerical loader/runtime.
_PROGRAM='''import hashlib,json,os,sys
from pathlib import Path
root=Path(sys.argv[1]);sys.path.insert(0,str(root))
from sim.ausc_policy import m2_act_auxiliary_train as original
from sim.ausc_policy import m2_given_contact_runtime as runtime
assert Path(original.__file__).resolve()==root/'sim/ausc_policy/m2_act_auxiliary_train.py'
assert Path(runtime.__file__).resolve()==root/'sim/ausc_policy/m2_given_contact_runtime.py'
payload=json.load(sys.stdin)
assert str(Path(sys.prefix))==runtime.VENDOR_ROOT+'/.venv'
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
for key in ('control','complete','weights','cpu_equivalence','selection','owner'):runtime.bound_path(payload[key])
model,complete=original.model_binding(payload['complete'])
assert complete['schema']=='m2_act_auxiliary_act_pilot_v8'
assert complete['best']==payload['weights'] and complete['cpu_equivalence']==payload['cpu_equivalence']
assert complete['completed_steps']==20000 and complete['selected_step']==17000
assert [row['step'] for row in complete['evaluation_candidates']]==list(range(500,20001,500))
selection=runtime.read_bound(payload['selection'])
assert runtime.read_bound(payload['cpu_equivalence'])['selection']==payload['selection']
torch,policy,packet=original.load_selected_policy(selection)
torch.set_num_threads(1)
assert torch.get_num_interop_threads()==1 and not torch.cuda.is_initialized()
assert sum(p.numel() for p in policy.parameters())==2658823 and len(policy.state_dict())==54
assert selection['inputs']['code']['code_commit']==payload['source41']['code_commit']
assert all(Path(r['path'])==root/'sim/ausc_policy'/name for name,r in selection['inputs']['code']['files'].items())
assert packet['step']==selection['selected_step']==17000
assert packet['data_version']==selection['data_version']==5 and packet['model_optimizer_version']==selection['model_optimizer_version']==8
assert packet['optimization']==selection['selected_optimization']
state={k:v.detach().cpu().contiguous().clone() for k,v in policy.state_dict().items()}
proofs={}
for k,v in sorted(state.items()):
    assert not v.is_floating_point() or bool(torch.isfinite(v).all())
    body=v.reshape(-1).view(torch.uint8).numpy().tobytes()
    proofs[k]=dict(shape=list(v.shape),dtype=str(v.dtype),bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
path=Path(payload['output'])/'INITIAL_STATE.pt'
with path.open('xb') as stream:torch.save(state,stream)
assert set(torch.load(path,map_location='cpu',weights_only=True))==set(state)
native=original.expanded.load_dataset(selection['inputs']['dataset'])
assert len(native['train']['states'])==8373
batch=runtime.pack_observations(native['train'],packet['normalization'],payload['indices'])
policy.eval()
with torch.no_grad():prediction=policy.predict_action_chunk({k:torch.from_numpy(v) for k,v in batch.items()}).cpu().numpy().astype('float64')
import numpy as np
prediction=prediction*np.asarray(packet['normalization']['action_std'])+np.asarray(packet['normalization']['action_mean'])
native_complete=runtime.read_bound(selection['inputs']['dataset'])
value=dict(schema=payload['schema'],complete=True,origin=payload['origin'],source41=payload['source41'],
    inputs=dict(control=payload['control'],complete=payload['complete'],weights=payload['weights'],cpu_equivalence=payload['cpu_equivalence'],
        owner=payload['owner'],selection=payload['selection'],software=selection['inputs']['software'],normalization=native_complete['normalization'],source_manifest=payload['source41']['source_manifest']),
    selected_step=17000,completed_steps=20000,parent_model=payload['parent_model'],
    historical_optimization=dict(completed=complete['optimization'],selected=selection['selected_optimization']),
    parameter_count=2658823,model=runtime.shared.native_spec(),task_contract=runtime.task_contract(),
    normalization=packet['normalization'],posture_head=selection['protocol']['posture_head'],state_dict=runtime.reference(path),tensor_proofs=proofs,
    optimizer_imported=False,scheduler_imported=False,
    probes=dict(split='train',dataset=selection['inputs']['dataset'],indices=payload['indices'],observations={k:v.tolist() for k,v in batch.items()},raw_predictions=prediction.tolist()),
    execution=dict(device='cpu',intraop_threads=1,interop_threads=1,cuda_visible_devices=''))
print(json.dumps(value,allow_nan=False))
'''


def export(output):
    need(sys.platform=='linux','Warm export requires actual Linux official CPU environment')
    require_archive();_parent_graph()
    output=Path(output).resolve();need(not output.exists(),'Fresh V9 tensor-only export directory required');output.mkdir(parents=True)
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    environment.pop('PYTHONPATH',None)
    payload=dict(schema=SCHEMA,origin=ORIGIN,control=CONTROL,complete=COMPLETE,weights=WEIGHTS,cpu_equivalence=CPU_EQUIVALENCE,
        selection=SELECTED,owner=OWNER,source41=SOURCE41,parent_model=PARENT_MODEL,indices=PROBE_TRAIN_INDICES,output=str(output))
    try:
        result=subprocess.run([OFFICIAL_PYTHON,'-I','-B','-c',_PROGRAM,SOURCE41['root']],input=json.dumps(payload,allow_nan=False),text=True,
            capture_output=True,check=False,timeout=1800,env=environment,cwd=SOURCE41['root'])
    finally:
        require_archive();_parent_graph()
    need(result.returncode==0,'Truthful original41 ACT10 CPU export failed: '+result.stderr[-8000:])
    value=json.loads(result.stdout);rt.bound_path(value['state_dict'])
    ref=rt.write_new(output/'INITIALIZATION.json',value);validate_receipt(ref);return ref


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    print(json.dumps(export(parser.parse_args().output),allow_nan=False),flush=True)


if __name__=='__main__':main()
