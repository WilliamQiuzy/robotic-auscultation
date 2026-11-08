"""Explicit ACT7 initialization from authenticated ACT6 selected CPU tensors.

Original ACT6 admission runs only in its immutable original source28 process.
Current V5 inference never runs this exporter or loads historical weights.
"""
from pathlib import Path
import sys

# Also support the root's explicit official-Python -I -B direct-file CLI.
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from sim.ausc_policy import m2_given_contact_runtime as rt
else:
    from . import m2_given_contact_runtime as rt

need=rt.need
SCHEMA='m2_act6_selected_cpu_initialization_v1'
EQUIVALENCE_SCHEMA='m2_act7_initialization_equivalence_v1'
OFFICIAL_PYTHON='/opt/wam-runtime/vendor/lerobot-eb7a54c/.venv/bin/python'
_BASE='/opt/wam-runtime/fast-wam/runs/ausc_policy/act6-terminal-train-20261107-v1/output/'
COMPLETE=dict(path=_BASE+'COMPLETE.json',bytes=562776,sha256='c4b145d38e4374de317cf4c070ed5a74c16cc9a54d6c992b2d75c210a5d672f1')
WEIGHTS=dict(path=_BASE+'best_auxiliary_act.pt',bytes=10843905,sha256='822bbaf5298695a3ceeb3ab00e741e49a220bca7d458824e20beabb9ef542efe')
CPU_EQUIVALENCE=dict(path=_BASE+'CPU_EQUIVALENCE.json',bytes=13263,sha256='aa73d61b8a4e3fbe4cc03d7454693419e6bd587d094f6ac8d18a7f3b12bb3138')
SELECTED=dict(path=_BASE+'SELECTED.json',bytes=299637,sha256='c1c0ad76d3173ce2fe0d421b18aafb1d068141610266a73e6f0f86b735ef1292')
OWNER=dict(path=str(Path(_BASE).parent/'result.json'),bytes=8157,sha256='0b6d1943fa936296f955946f80dea635a23677e826efde177d9825c2b892b494')
SOURCE28_ROOT='/opt/wam-runtime/code/wam-act-recovery-20261106-v28'
SOURCE28_COMMIT='d932993b13030da7873d04a4a9471df99bb836ec'
SOURCE28_FILES=1673
SOURCE28_MANIFEST=dict(path=SOURCE28_ROOT+'/SOURCE_MANIFEST.json',bytes=289698,
    sha256='f05a78822bcc6febbaa7f4a7a8de2cabd9aa33daed3d648b90053cb336edb89c')
PROBE_TRAIN_INDICES=[0,64,4186,8372]

import argparse
from copy import deepcopy
import hashlib
import io
import json
import os
import subprocess

import numpy as np


def _validate_owner(owner):
    # The complete owner record itself is pinned; this is not a new GPU lease.
    need(isinstance(owner,dict) and owner.get('run_id')=='act6-terminal-train-20261107-v1' and
        owner.get('code_commit')==SOURCE28_COMMIT and owner.get('completed') is True and owner.get('gpu_released') is True,
        'Original ACT6 training owner must have closed normally')
    stage=owner.get('stage',{})
    need(stage.get('stage')=='m2_given_contact_expanded_pilot_train' and stage.get('passed') is True and
        type(stage.get('exit_code')) is int and stage['exit_code']==0 and stage.get('process_group_drained') is True and
        all(stage.get(key)==[] for key in ('errors','signals','deferred_signals')),
        'Original ACT6 owner stage must have closed normally without signals')


def require_archive():
    """Authenticate the full pinned original28 closure only at export time."""
    import stat
    root=Path(SOURCE28_ROOT)
    need(root.is_absolute() and root.resolve()==root,'Canonical original28 source root required')
    rows=rt.read_bound(SOURCE28_MANIFEST)
    need(isinstance(rows,list) and len(rows)==SOURCE28_FILES,'Complete1673 original28 manifest required')
    need((root/'CODE_COMMIT').read_text().strip()==SOURCE28_COMMIT,'Original28 commit differs')
    names=set()
    for row in rows:
        name=row.get('file');relative=Path(name) if isinstance(name,str) else None
        need(set(row)=={'file','bytes','sha256'} and relative is not None and name==relative.as_posix() and
            not relative.is_absolute() and relative.parts and '..' not in relative.parts and name not in names and
            name not in ('SOURCE_MANIFEST.json','CODE_COMMIT'),'Unique safe original28 manifest paths required')
        path=root/relative
        need(all(not part.is_symlink() for part in (path,*path.parents)) and stat.S_ISREG(path.lstat().st_mode),
            'Regular original28 files without symlinks required')
        rt.bound_path(dict(path=str(path),bytes=row['bytes'],sha256=row['sha256']));names.add(name)
    actual=set()
    for path in root.rglob('*'):
        need(not path.is_symlink(),'No symlinks in original28 archive')
        if path.is_file():actual.add(path.relative_to(root).as_posix())
    need(actual==names|{'SOURCE_MANIFEST.json','CODE_COMMIT'},'Unknown or omitted original28 file')


def tensor_proofs(state):
    """Hash every tensor's storage bytes, including buffers and signed zero."""
    import torch
    need(isinstance(state,dict) and state and all(isinstance(k,str) and isinstance(v,torch.Tensor) for k,v in state.items()),
        'Nonempty tensor-only state_dict required; optimizer/scheduler packets rejected')
    result={}
    for key,value in sorted(state.items()):
        value=value.detach().cpu().contiguous()
        need(value.layout==torch.strided and (not value.is_floating_point() or bool(torch.isfinite(value).all())),
            'Finite dense initialization tensors required')
        body=value.reshape(-1).view(torch.uint8).numpy().tobytes()
        result[key]=dict(shape=list(value.shape),dtype=str(value.dtype),bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
    return result


def validate_receipt(ref,*,normalization=None,posture_head=None):
    """Metadata/hash admission only: never load old or exported model tensors.

    Config pins this export's complete reference. Original28 already admitted the
    model when exporting; current inference reads only immutable provenance.
    """
    value=rt.read_bound(ref);inputs=value.get('inputs',{})
    need(value.get('schema')==SCHEMA and value.get('complete') is True and
        value.get('origin')=='ACT6 selected18000; original28 CPU authenticated' and
        value.get('optimizer_imported') is False and value.get('scheduler_imported') is False and
        value.get('selected_step')==18000 and value.get('completed_steps')==20000 and
        value.get('parameter_count')==2658823 and value.get('model')==rt.shared.native_spec() and
        value.get('task_contract')==rt.task_contract(), 'Exact authenticated ACT6 initialization origin/model required')
    need(inputs.get('complete')==COMPLETE and inputs.get('weights')==WEIGHTS and
        inputs.get('cpu_equivalence')==CPU_EQUIVALENCE and inputs.get('selection')==SELECTED and inputs.get('owner')==OWNER and inputs.get('source_manifest')==SOURCE28_MANIFEST and
        value.get('source28')==dict(code_commit=SOURCE28_COMMIT,file_count=SOURCE28_FILES,root=SOURCE28_ROOT),
        'Pinned ACT6 complete/weights/CPU receipt/original28 manifest required')
    complete=rt.read_bound(COMPLETE);cpu=rt.read_bound(CPU_EQUIVALENCE)
    _validate_owner(rt.read_bound(OWNER))
    need(complete.get('schema')=='m2_act_auxiliary_act_pilot_v4' and complete.get('complete') is True and
        complete.get('completed_steps')==20000 and complete.get('selected_step')==18000 and complete.get('best')==WEIGHTS and
        complete.get('cpu_equivalence')==CPU_EQUIVALENCE and cpu.get('passed') is True and
        cpu.get('checkpoint')==WEIGHTS and cpu.get('selected_step')==18000 and inputs.get('selection')==cpu.get('selection'),
        'Original completed selected18000 ACT6 and CPU selection required')
    selected=rt.read_bound(inputs['selection'])
    need(selected.get('schema')=='m2_act_auxiliary_act_selection_v4' and selected.get('best')==WEIGHTS and
        selected.get('selected_step')==18000 and all(selected.get(k)==complete.get(k) for k in ('inputs','protocol')) and
        inputs.get('software')==selected['inputs'].get('software'), 'Original ACT6 selected provenance differs')
    code=selected['inputs']['code'];root=Path(SOURCE28_ROOT)
    need(code.get('code_commit')==SOURCE28_COMMIT and code.get('files') and all(
        Path(item['path'])==root/'sim/ausc_policy'/name for name,item in code['files'].items()),
        'Original ACT6 numerical source paths must remain actual28')
    software=rt.read_bound(inputs['software']);vendor=software.get('vendor',{})
    need(vendor.get('root')==rt.VENDOR_ROOT and vendor.get('upstream_commit')==rt.UPSTREAM_COMMIT and
        software.get('prefix')==rt.VENDOR_ROOT+'/.venv' and software.get('source_bindings') and
        all(Path(path).is_relative_to(root) for path in software['source_bindings']) and
        vendor.get('files',{}).get('src/lerobot/policies/act/modeling_act.py',{}).get('sha256')==
            '83d954f79ccd3eaa6774dbea92524939c8ac08359bb355ba7cd37b8601e1e3af',
        'Original official ACT6 vendor/environment/origin required')
    native=rt.read_bound(selected['inputs']['dataset'])
    need(inputs.get('normalization')==native.get('normalization') and
        rt.validate_normalization(value.get('normalization',{}))==rt.read_bound(inputs['normalization']) and
        value.get('posture_head')==selected['protocol'].get('posture_head'), 'Original ACT6 head/native normalization differs')
    # Version two is a model property; never weaken it to accept a relabeled head.
    if __package__ in (None,''):
        from sim.ausc_policy import m2_act_posture_head as head
    else:
        from . import m2_act_posture_head as head
    head.validate_spec(value['posture_head'])
    need(value['posture_head']['schema']==head.SCHEMA_V2 and value['posture_head']['normalization']==value['normalization'],
        'Original ACT6 head v2/native normalization required')
    if normalization is not None:need(value['normalization']==normalization,'New model normalization differs from ACT6')
    if posture_head is not None:need(value['posture_head']==posture_head,'New model head differs from ACT6')
    probes=value.get('probes',{});observations=probes.get('observations',{})
    need(probes.get('split')=='train' and probes.get('dataset')==selected['inputs']['dataset'] and
        probes.get('indices')==PROBE_TRAIN_INDICES and set(observations)=={'observation.state','observation.environment_state'} and
        np.asarray(observations['observation.state']).shape==(4,7) and
        np.asarray(observations['observation.environment_state']).shape==(4,27) and
        np.asarray(probes.get('raw_predictions')).shape==(4,20,7) and
        all(np.isfinite(np.asarray(v)).all() for v in [*observations.values(),probes['raw_predictions']]),
        'Fixed known native TRAIN-only initial prediction probes required')
    need(value.get('execution')==dict(device='cpu',intraop_threads=1,interop_threads=1,cuda_visible_devices=''),
        'Original28 single-thread CUDA-hidden CPU export required')
    proofs=value.get('tensor_proofs')
    need(isinstance(proofs,dict) and len(proofs)==54 and all(isinstance(k,str) and isinstance(p,dict) and
        set(p)=={'shape','dtype','bytes','sha256'} and isinstance(p['shape'],list) and
        all(type(n) is int and n>=0 for n in p['shape']) and isinstance(p['dtype'],str) and
        type(p['bytes']) is int and p['bytes']>=0 and isinstance(p['sha256'],str) and len(p['sha256'])==64
        for k,p in proofs.items()), 'Complete initialization tensor bit proofs required')
    rt._reference_shape(value.get('state_dict'))
    need(set(value['state_dict'])=={'path','bytes','sha256'} and type(value['state_dict']['bytes']) is int and
        value['state_dict']['bytes']>0,'Full initialization tensor artifact reference required')
    return value


def protocol(ref):
    value=validate_receipt(ref)
    return dict(schema=SCHEMA,receipt=deepcopy(ref),inputs=deepcopy(value['inputs']),
        method='ACT6 selected18000 full state_dict -> ACT7; exactly1000 new optimizer steps',
        final_linear_reset=False,optimizer='fresh AdamW',optimizer_initial_step=0,
        optimizer_imported=False,scheduler_imported=False,initial_tensor_proofs=deepcopy(value['tensor_proofs']),
        original_evaluations='initialization provenance only; excluded from ACT7 evaluations')


def initialize_policy(torch,policy,ref,*,normalization,posture_head):
    """Strictly load all authenticated CPU tensors once before fresh optimization."""
    value=validate_receipt(ref,normalization=normalization,posture_head=posture_head)
    need(not hasattr(policy,'_posture_training_initialization') and not hasattr(policy,'_act7_initialization'),
        'ACT7 initialization cannot follow zero reset or be repeated')
    state=torch.load(io.BytesIO(rt.bound_path(value['state_dict']).read_bytes()),map_location='cpu',weights_only=True)
    need(tensor_proofs(state)==value['tensor_proofs'],'Authenticated initialization tensor bytes differ')
    need(len(state)==54,'All54 ACT6 initialization tensors required')
    current=policy.state_dict()
    need(set(state)==set(current) and all(state[k].shape==current[k].shape and state[k].dtype==current[k].dtype for k in state),
        'Exact complete initialization tensor names/shapes/dtypes required')
    policy.load_state_dict(state,strict=True)
    actual=tensor_proofs(policy.state_dict());need(actual==value['tensor_proofs'],'Initial model tensors differ after strict load')
    policy.eval();device=next(policy.parameters()).device
    batch={k:torch.tensor(v,dtype=torch.float32,device=device) for k,v in value['probes']['observations'].items()}
    with torch.no_grad():prediction=policy.predict_action_chunk(batch).detach().cpu().numpy().astype(np.float64)
    prediction=prediction*np.asarray(normalization['action_std'])+np.asarray(normalization['action_mean'])
    expected=np.asarray(value['probes']['raw_predictions'],dtype=np.float64)
    need(prediction.shape==expected.shape and np.isfinite(prediction).all() and
        np.all(abs(prediction-expected)<=1e-6+1e-5*np.maximum(abs(prediction),abs(expected))),
        'ACT7 initial prediction equivalence to original ACT6 failed')
    result=dict(schema=EQUIVALENCE_SCHEMA,passed=True,initialization=deepcopy(ref),initial_tensor_proofs=actual,
        prediction_max_abs_rad=float(abs(prediction-expected).max()),prediction_rtol=1e-5,prediction_atol=1e-6,
        prediction_device=str(device),probe_split='train',probe_indices=PROBE_TRAIN_INDICES,
        optimizer_step=0,optimizer_imported=False,scheduler_imported=False,final_linear_reset=False)
    policy._act7_initialization=deepcopy(result)
    return result


def fresh_optimizer(torch,policy,*,lr,weight_decay):
    need(hasattr(policy,'_act7_initialization'),'Authenticated ACT7 initialization required before optimizer construction')
    optimizer=torch.optim.AdamW(policy.parameters(),lr=lr,weight_decay=weight_decay,betas=(.9,.999),eps=1e-8)
    need(not optimizer.state,'Fresh AdamW must start at step0 with no imported state')
    return optimizer,dict(name='AdamW',initial_step=0,initial_state_entries=0,lr=lr,weight_decay=weight_decay,
        betas=[.9,.999],eps=1e-8,optimizer_imported=False,scheduler_imported=False)


# This child deliberately imports NO current numerical or helper module.
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
for key in ('complete','weights','cpu_equivalence','selection','owner'):runtime.bound_path(payload[key])
model,complete=original.model_binding(payload['complete'])
assert complete['best']==payload['weights'] and complete['cpu_equivalence']==payload['cpu_equivalence']
assert complete['schema']=='m2_act_auxiliary_act_pilot_v4'
assert complete['completed_steps']==20000 and complete['selected_step']==18000
owner=runtime.read_bound(payload['owner'])
assert owner['run_id']=='act6-terminal-train-20261107-v1' and owner['code_commit']==payload['code_commit']
assert owner['completed'] is True and owner['gpu_released'] is True
stage=owner['stage']
assert stage['stage']=='m2_given_contact_expanded_pilot_train' and stage['passed'] is True
assert type(stage['exit_code']) is int and stage['exit_code']==0 and stage['process_group_drained'] is True
assert all(stage[k]==[] for k in ('errors','signals','deferred_signals'))
cpu=runtime.read_bound(payload['cpu_equivalence'])
assert cpu['selection']==payload['selection']
selection=runtime.read_bound(payload['selection'])
torch,policy,packet=original.load_selected_policy(selection)
torch.set_num_threads(1)
assert torch.get_num_interop_threads()==1
assert not torch.cuda.is_initialized()
assert sum(p.numel() for p in policy.parameters())==2658823
assert selection['inputs']['code']['code_commit']==payload['code_commit']
assert all(Path(r['path'])==root/'sim/ausc_policy'/name for name,r in selection['inputs']['code']['files'].items())
assert len(policy.state_dict())==54
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
indices=payload['indices']
batch=runtime.pack_observations(native['train'],packet['normalization'],indices)
policy.eval()
with torch.no_grad():prediction=policy.predict_action_chunk({k:torch.from_numpy(v) for k,v in batch.items()}).cpu().numpy().astype('float64')
import numpy as np
prediction=prediction*np.asarray(packet['normalization']['action_std'])+np.asarray(packet['normalization']['action_mean'])
native_complete=runtime.read_bound(selection['inputs']['dataset'])
value=dict(schema=payload['schema'],complete=True,origin='ACT6 selected18000; original28 CPU authenticated',
    inputs=dict(complete=payload['complete'],weights=payload['weights'],cpu_equivalence=payload['cpu_equivalence'],
        owner=payload['owner'],selection=payload['selection'],software=selection['inputs']['software'],normalization=native_complete['normalization'],source_manifest=payload['manifest']),
    source28=dict(code_commit=payload['code_commit'],file_count=1673,root=str(root)),
    selected_step=18000,completed_steps=20000,parameter_count=2658823,model=runtime.shared.native_spec(),task_contract=runtime.task_contract(),
    normalization=packet['normalization'],posture_head=selection['protocol']['posture_head'],state_dict=runtime.reference(path),tensor_proofs=proofs,
    optimizer_imported=False,scheduler_imported=False,
    probes=dict(split='train',dataset=selection['inputs']['dataset'],indices=indices,observations={k:v.tolist() for k,v in batch.items()},raw_predictions=prediction.tolist()),
    execution=dict(device='cpu',intraop_threads=1,interop_threads=1,cuda_visible_devices=''))
print(json.dumps(value,allow_nan=False))
'''


def export(output):
    """Export pinned ACT6 initialization in original28, without any GPU owner."""
    need(sys.platform=='linux','ACT6 warm-start export requires the actual Linux official CPU environment')
    require_archive()
    for ref in (COMPLETE,WEIGHTS,CPU_EQUIVALENCE,SELECTED,OWNER):rt.bound_path(ref)
    output=Path(output).resolve();need(not output.exists(),'Fresh ACT6 initialization export directory required')
    output.mkdir(parents=True)
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    environment.pop('PYTHONPATH',None)
    payload=dict(schema=SCHEMA,complete=COMPLETE,weights=WEIGHTS,cpu_equivalence=CPU_EQUIVALENCE,selection=SELECTED,owner=OWNER,
        manifest=SOURCE28_MANIFEST,code_commit=SOURCE28_COMMIT,indices=PROBE_TRAIN_INDICES,output=str(output))
    result=subprocess.run([OFFICIAL_PYTHON,'-I','-B','-c',_PROGRAM,SOURCE28_ROOT],input=json.dumps(payload,allow_nan=False),
        text=True,capture_output=True,check=False,timeout=1800,env=environment,cwd=SOURCE28_ROOT)
    need(result.returncode==0,'Original28 ACT6 selected CPU export failed: '+result.stderr[-8000:])
    require_archive()
    for ref in (COMPLETE,WEIGHTS,CPU_EQUIVALENCE,SELECTED,OWNER):rt.bound_path(ref)
    value=json.loads(result.stdout);rt.bound_path(value['state_dict'])
    ref=rt.write_new(output/'INITIALIZATION.json',value);validate_receipt(ref)
    return ref


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(export(args.output),allow_nan=False),flush=True)


if __name__=='__main__':main()
