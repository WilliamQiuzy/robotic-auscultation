"""V8 initialization from the exact ACT9 V7 selected policy, in Source34.

The exporter launches only the pinned original source loader in official CPU
Python. Inference validates provenance without replaying any historical model.
"""
from copy import deepcopy
import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from sim.ausc_policy import m2_given_contact_runtime as rt
    from sim.ausc_policy import m2_act_phase_warm_start as shared
    from sim.ausc_policy import m2_teacher_stop_reader as teacher
    from sim.ausc_policy import m2_act_posture_head as head
else:
    from . import m2_given_contact_runtime as rt
    from . import m2_act_phase_warm_start as shared
    from . import m2_teacher_stop_reader as teacher
    from . import m2_act_posture_head as head

need=rt.need
tensor_proofs=shared.tensor_proofs
PROBE_TRAIN_INDICES=list(shared.PROBE_TRAIN_INDICES)
OFFICIAL_PYTHON=shared.OFFICIAL_PYTHON
SCHEMA='m2_act9_selected_cpu_initialization_v8'
EQUIVALENCE_SCHEMA='m2_act_diverse_initialization_equivalence_v8'
ORIGIN='ACT9 selected1000; original34 CPU authenticated'
SOURCE34=deepcopy(teacher.SOURCE34)
CONTROL=dict(path='/opt/wam-runtime/fast-wam/diagnostics/act9-measured-training-20261107-v1/COMPLETE.json',bytes=2220,
    sha256='93ecf09940471ed62e23f757f9f885cf9f6efa08b6433a4deed4a84ae90bf6fe')
_BASE='/opt/wam-runtime/fast-wam/runs/ausc_policy/act9-measured-train-20261107-v1/output/'
COMPLETE=dict(path=_BASE+'COMPLETE.json',bytes=2019252,sha256='3be44244e867846db274a6f558d583377a8d8fe33d95846e06d8b81c7f2489f7')
WEIGHTS=dict(path=_BASE+'best_auxiliary_act.pt',bytes=11278593,sha256='7523475be10b1ad9edb1b2080513c748d7d3d0dcbc90b19fbefc5539ac27baec')
CPU_EQUIVALENCE=dict(path=_BASE+'CPU_EQUIVALENCE.json',bytes=13757,sha256='a6f6cc75977f52363892fe4694707b160c85b71d2fff88da1593dc2966ad4b04')
SELECTED=dict(path=_BASE+'SELECTED.json',bytes=1996150,sha256='72b6bec3e60e52c9ab4a964d90e15cbdb6266ea31a899580e87c1c5b22863988')
OWNER=dict(path=str(Path(_BASE).parent/'result.json'),bytes=8214,sha256='d4d39e4ea488f104d19a7ba94a17bf17fa6f463ce345e0e9cfe4e350edeaef55')


def _validate_owner(value):
    need(value.get('run_id')=='act9-measured-train-20261107-v1' and value.get('code_commit')==SOURCE34['code_commit'] and
        value.get('completed') is True and value.get('gpu_released') is True,'Exact normally closed ACT9 owner required')
    stage=value.get('stage',{})
    need(stage.get('stage')=='m2_given_contact_expanded_pilot_train' and stage.get('passed') is True and
        type(stage.get('exit_code')) is int and stage['exit_code']==0 and stage.get('process_group_drained') is True and
        all(stage.get(k)==[] for k in ('errors','signals','deferred_signals')), 'ACT9 owner stage must close normally without signals')


def require_archive():
    """Authenticate every original34 file only when exporting actual tensors."""
    import stat
    root=Path(SOURCE34['root']);need(root.is_absolute() and root.resolve()==root,'Canonical original34 source required')
    rows=rt.read_bound(SOURCE34['source_manifest']);need(isinstance(rows,list) and len(rows)==SOURCE34['manifest_files'],'Complete original34 inventory required')
    need((root/'CODE_COMMIT').read_text().strip()==SOURCE34['code_commit'],'Original34 commit differs')
    names=set()
    for row in rows:
        name=row.get('file');relative=Path(name) if isinstance(name,str) else None
        need(set(row)=={'file','bytes','sha256'} and relative is not None and name==relative.as_posix() and
            not relative.is_absolute() and relative.parts and '..' not in relative.parts and name not in names and
            name not in ('SOURCE_MANIFEST.json','CODE_COMMIT'),'Unique safe original34 file paths required')
        path=root/relative
        need(all(not p.is_symlink() for p in (path,*path.parents)) and stat.S_ISREG(path.lstat().st_mode),'Regular original34 files without aliases required')
        rt.bound_path(dict(path=str(path),bytes=row['bytes'],sha256=row['sha256']));names.add(name)
    actual=set()
    for path in root.rglob('*'):
        need(not path.is_symlink(),'No original34 symlinks allowed')
        if path.is_file():actual.add(path.relative_to(root).as_posix())
    need(actual==names|{'SOURCE_MANIFEST.json','CODE_COMMIT'},'Unknown or omitted original34 files')


def _parent_graph():
    control=rt.read_bound(CONTROL);complete=rt.read_bound(COMPLETE);cpu=rt.read_bound(CPU_EQUIVALENCE)
    selected=rt.read_bound(SELECTED);_validate_owner(rt.read_bound(OWNER))
    need(control.get('schema')=='act9_single_owned_training_execution_v1' and control.get('complete') is True and
        control.get('normal_owner') is True and control.get('training')==COMPLETE and control.get('training_owner')==OWNER and
        control.get('model')==dict(checkpoint=WEIGHTS,checkpoint_schema='m2_act_auxiliary_act_pilot_v7',complete=COMPLETE) and
        control.get('cpu_equivalence')==CPU_EQUIVALENCE and control.get('completed_steps')==control.get('selected_step')==1000 and
        control.get('source')==dict(code_commit=SOURCE34['code_commit'],files=SOURCE34['manifest_files'],manifest=SOURCE34['source_manifest'],root=SOURCE34['root']),
        'Pinned ACT9 closed training control/model/source graph required')
    need(complete.get('schema')=='m2_act_auxiliary_act_pilot_v7' and complete.get('complete') is True and
        complete.get('completed_steps')==complete.get('selected_step')==1000 and complete.get('best')==WEIGHTS and
        complete.get('cpu_equivalence')==CPU_EQUIVALENCE and cpu.get('schema')=='m2_act_auxiliary_act_cpu_equivalence_v7' and
        cpu.get('passed') is True and cpu.get('checkpoint')==WEIGHTS and cpu.get('selection')==SELECTED and cpu.get('selected_step')==1000,
        'Exact ACT9 V7 selected1000 and own CPU equivalence required')
    need(selected.get('schema')=='m2_act_auxiliary_act_selection_v7' and selected.get('selected_step')==1000 and selected.get('best')==WEIGHTS and
        all(selected.get(k)==complete.get(k) for k in ('inputs','protocol','initialization','scope','optimization_scope','measured_inventory','selected_predictions')),
        'Original ACT9 selected provenance differs')
    root=Path(SOURCE34['root']);code=selected['inputs']['code']
    need(code.get('code_commit')==SOURCE34['code_commit'] and code.get('files') and all(
        Path(ref['path'])==root/'sim/ausc_policy'/name for name,ref in code['files'].items()),'ACT9 original34 numerical origins required')
    software=rt.read_bound(selected['inputs']['software']);vendor=software.get('vendor',{})
    need(vendor.get('root')==rt.VENDOR_ROOT and vendor.get('upstream_commit')==rt.UPSTREAM_COMMIT and
        software.get('prefix')==rt.VENDOR_ROOT+'/.venv' and software.get('source_bindings') and
        all(Path(path).is_relative_to(root) for path in software['source_bindings']) and
        vendor.get('files',{}).get('src/lerobot/policies/act/modeling_act.py',{}).get('sha256')==
        '83d954f79ccd3eaa6774dbea92524939c8ac08359bb355ba7cd37b8601e1e3af','Official ACT9 CPU vendor/environment/origins required')
    native=rt.read_bound(selected['inputs']['dataset'])
    return selected,native


def validate_receipt(ref,*,normalization=None,posture_head=None):
    """Metadata-only import admission. Tensor bytes are checked at initialization."""
    value=rt.read_bound(ref);selected,native=_parent_graph();inputs=value.get('inputs',{})
    need(value.get('schema')==SCHEMA and value.get('complete') is True and value.get('origin')==ORIGIN and
        value.get('source34')==SOURCE34 and value.get('completed_steps')==value.get('selected_step')==1000 and
        value.get('parameter_count')==2658823 and value.get('model')==rt.shared.native_spec() and value.get('task_contract')==rt.task_contract() and
        value.get('optimizer_imported') is False and value.get('scheduler_imported') is False,'Exact typed ACT9 V8 initialization required')
    need(inputs==dict(control=CONTROL,complete=COMPLETE,weights=WEIGHTS,cpu_equivalence=CPU_EQUIVALENCE,selection=SELECTED,owner=OWNER,
        software=selected['inputs']['software'],normalization=native['normalization'],source_manifest=SOURCE34['source_manifest']),
        'All pinned ACT9 initialization provenance references required')
    need(rt.validate_normalization(value.get('normalization',{}))==rt.read_bound(native['normalization']) and
        value.get('posture_head')==selected['protocol']['posture_head'],'ACT9 original native normalization/head differs')
    head.validate_spec(value['posture_head'])
    need(value['posture_head']['schema']==head.SCHEMA_V2 and value['posture_head']['normalization']==value['normalization'],'Unchanged original posture head v2 required')
    if normalization is not None:need(value['normalization']==normalization,'V8 normalization differs from ACT9')
    if posture_head is not None:need(value['posture_head']==posture_head,'V8 posture head differs from ACT9')
    probes=value.get('probes',{});obs=probes.get('observations',{})
    need(probes.get('split')=='train' and probes.get('dataset')==selected['inputs']['dataset'] and probes.get('indices')==PROBE_TRAIN_INDICES and
        set(obs)=={'observation.state','observation.environment_state'} and np.asarray(obs['observation.state']).shape==(4,7) and
        np.asarray(obs['observation.environment_state']).shape==(4,27) and np.asarray(probes.get('raw_predictions')).shape==(4,20,7) and
        all(np.isfinite(np.asarray(v)).all() for v in [*obs.values(),probes['raw_predictions']]),'Fixed actual native TRAIN-only probes required')
    need(value.get('execution')==dict(device='cpu',intraop_threads=1,interop_threads=1,cuda_visible_devices=''),'Single-thread CUDA-hidden original34 CPU export required')
    proofs=value.get('tensor_proofs')
    need(isinstance(proofs,dict) and len(proofs)==54 and all(isinstance(k,str) and isinstance(p,dict) and
        set(p)=={'shape','dtype','bytes','sha256'} and isinstance(p['shape'],list) and all(type(n) is int and n>=0 for n in p['shape']) and
        isinstance(p['dtype'],str) and type(p['bytes']) is int and p['bytes']>=0 and isinstance(p['sha256'],str) and len(p['sha256'])==64
        for k,p in proofs.items()),'All54 tensor bit proofs required')
    rt._reference_shape(value.get('state_dict'));need(set(value['state_dict'])=={'path','bytes','sha256'} and type(value['state_dict']['bytes']) is int and value['state_dict']['bytes']>0,
        'Full tensor-only packet reference required')
    return value


def protocol(ref):
    value=validate_receipt(ref)
    return dict(schema=SCHEMA,receipt=deepcopy(ref),inputs=deepcopy(value['inputs']),method='ACT9 selected1000 full state_dict -> V8; exactly20000 new optimizer updates',
        final_linear_reset=False,optimizer='fresh AdamW',optimizer_initial_step=0,optimizer_imported=False,scheduler_imported=False,
        initial_tensor_proofs=deepcopy(value['tensor_proofs']),original_evaluations='initialization provenance only; excluded from V8 evaluations')


def initialize_policy(torch,policy,ref,*,normalization,posture_head):
    value=validate_receipt(ref,normalization=normalization,posture_head=posture_head)
    need(not any(hasattr(policy,k) for k in ('_posture_training_initialization','_act7_initialization','_act8_initialization')),
        'V8 import cannot follow an old initialization/reset or be repeated')
    state=torch.load(io.BytesIO(rt.bound_path(value['state_dict']).read_bytes()),map_location='cpu',weights_only=True)
    need(len(state)==54 and tensor_proofs(state)==value['tensor_proofs'],'All54 authenticated tensor bytes required')
    current=policy.state_dict()
    need(set(state)==set(current) and all(state[k].shape==current[k].shape and state[k].dtype==current[k].dtype for k in state),
        'Exact unchanged architecture tensor names/shapes/dtypes required')
    policy.load_state_dict(state,strict=True);actual=tensor_proofs(policy.state_dict());need(actual==value['tensor_proofs'],'Bit-identical initial tensors required')
    policy.eval();device=next(policy.parameters()).device
    batch={k:torch.tensor(v,dtype=torch.float32,device=device) for k,v in value['probes']['observations'].items()}
    with torch.no_grad():prediction=policy.predict_action_chunk(batch).detach().cpu().numpy().astype(np.float64)
    prediction=prediction*np.asarray(normalization['action_std'])+np.asarray(normalization['action_mean']);expected=np.asarray(value['probes']['raw_predictions'])
    need(prediction.shape==expected.shape and np.isfinite(prediction).all() and np.all(abs(prediction-expected)<=1e-6+1e-5*np.maximum(abs(prediction),abs(expected))),
        'V8 initial prediction equivalence to actual ACT9 failed')
    result=dict(schema=EQUIVALENCE_SCHEMA,passed=True,initialization=deepcopy(ref),initial_tensor_proofs=actual,prediction_max_abs_rad=float(abs(prediction-expected).max()),
        prediction_rtol=1e-5,prediction_atol=1e-6,prediction_device=str(device),probe_split='train',probe_indices=PROBE_TRAIN_INDICES,
        optimizer_step=0,optimizer_imported=False,scheduler_imported=False,final_linear_reset=False)
    policy._act8_initialization=deepcopy(result)
    return result


def fresh_optimizer(torch,policy,*,lr,weight_decay):
    need(hasattr(policy,'_act8_initialization') and lr==1e-4 and weight_decay==1e-4,'Authenticated V8 initialization and exact fresh AdamW required')
    optimizer=torch.optim.AdamW(policy.parameters(),lr=lr,weight_decay=weight_decay,betas=(.9,.999),eps=1e-8)
    need(not optimizer.state,'Fresh AdamW step0 required')
    return optimizer,dict(name='AdamW',initial_step=0,initial_state_entries=0,lr=lr,weight_decay=weight_decay,betas=[.9,.999],eps=1e-8,optimizer_imported=False,scheduler_imported=False)


# This child imports ONLY the true original34 model/runtime. The current helper
# runs outside it and never changes origins or weakens the historical inventory.
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
assert complete['schema']=='m2_act_auxiliary_act_pilot_v7'
assert complete['best']==payload['weights'] and complete['cpu_equivalence']==payload['cpu_equivalence']
assert complete['completed_steps']==complete['selected_step']==1000
selection=runtime.read_bound(payload['selection'])
assert runtime.read_bound(payload['cpu_equivalence'])['selection']==payload['selection']
torch,policy,packet=original.load_selected_policy(selection)
torch.set_num_threads(1)
assert torch.get_num_interop_threads()==1 and not torch.cuda.is_initialized()
assert sum(p.numel() for p in policy.parameters())==2658823 and len(policy.state_dict())==54
assert selection['inputs']['code']['code_commit']==payload['source34']['code_commit']
assert all(Path(r['path'])==root/'sim/ausc_policy'/name for name,r in selection['inputs']['code']['files'].items())
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
value=dict(schema=payload['schema'],complete=True,origin=payload['origin'],source34=payload['source34'],
    inputs=dict(control=payload['control'],complete=payload['complete'],weights=payload['weights'],cpu_equivalence=payload['cpu_equivalence'],
        owner=payload['owner'],selection=payload['selection'],software=selection['inputs']['software'],normalization=native_complete['normalization'],source_manifest=payload['source34']['source_manifest']),
    selected_step=1000,completed_steps=1000,parameter_count=2658823,model=runtime.shared.native_spec(),task_contract=runtime.task_contract(),
    normalization=packet['normalization'],posture_head=selection['protocol']['posture_head'],state_dict=runtime.reference(path),tensor_proofs=proofs,
    optimizer_imported=False,scheduler_imported=False,
    probes=dict(split='train',dataset=selection['inputs']['dataset'],indices=payload['indices'],observations={k:v.tolist() for k,v in batch.items()},raw_predictions=prediction.tolist()),
    execution=dict(device='cpu',intraop_threads=1,interop_threads=1,cuda_visible_devices=''))
print(json.dumps(value,allow_nan=False))
'''


def export(output):
    need(sys.platform=='linux','Warm export requires actual Linux official CPU environment')
    require_archive();_parent_graph()
    output=Path(output).resolve();need(not output.exists(),'Fresh V8 tensor-only export directory required');output.mkdir(parents=True)
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    environment.pop('PYTHONPATH',None)
    payload=dict(schema=SCHEMA,origin=ORIGIN,control=CONTROL,complete=COMPLETE,weights=WEIGHTS,cpu_equivalence=CPU_EQUIVALENCE,
        selection=SELECTED,owner=OWNER,source34=SOURCE34,indices=PROBE_TRAIN_INDICES,output=str(output))
    result=subprocess.run([OFFICIAL_PYTHON,'-I','-B','-c',_PROGRAM,SOURCE34['root']],input=json.dumps(payload,allow_nan=False),text=True,
        capture_output=True,check=False,timeout=1800,env=environment,cwd=SOURCE34['root'])
    need(result.returncode==0,'Truthful original34 ACT9 CPU export failed: '+result.stderr[-8000:])
    require_archive();_parent_graph();value=json.loads(result.stdout);rt.bound_path(value['state_dict'])
    ref=rt.write_new(output/'INITIALIZATION.json',value);validate_receipt(ref);return ref


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    print(json.dumps(export(parser.parse_args().output),allow_nan=False),flush=True)


if __name__=='__main__':main()
