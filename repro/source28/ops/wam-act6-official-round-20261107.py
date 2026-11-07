#!/usr/bin/env python3
"""Finite ACT6 source28 CPU integration, then owned 20k training and four DEV.

External to the frozen source tree. Only the original continuation helper starts
GPU/native owners. No retries, resumes, fresh cases, expert fallback or reset.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib
from importlib.machinery import ModuleSpec
import json
import math
import os
from pathlib import Path
import random
import stat
import sys
import time
import traceback
from types import ModuleType, SimpleNamespace

CODE=Path('/opt/wam-runtime/code/wam-act-recovery-20261106-v28')
COMMIT='d932993b13030da7873d04a4a9471df99bb836ec'
MANIFEST_SHA256='f05a78822bcc6febbaa7f4a7a8de2cabd9aa33daed3d648b90053cb336edb89c'
FILE_COUNT=1673
BASE=Path('/opt/wam-runtime/fast-wam')
OFFICIAL='/opt/wam-runtime/vendor/lerobot-eb7a54c/.venv/bin/python'
NATIVE='/opt/wam-runtime/envs/isaacsim60/bin/python'
INTEGRATION_CONTROL=BASE/'diagnostics/act6-official-integration-20261107-v1'
CONTROL=BASE/'diagnostics/act6-terminal-development-20261107-v1'
DATA_CONTROL=BASE/'diagnostics/act6-terminal-data-export-20261107-v1/COMPLETE.json'
DATA_PREPARATION=BASE/'datasets/act6-terminal-full-chunks-20261107-v1/COMPLETE.json'
TRAIN_RUN='act6-terminal-train-20261107-v1'
SMOKE_RUN='act6-terminal-smoke-20261107-v1'
CONFIG_SCHEMA='m2_act_auxiliary_training_config_v4'
INTEGRATION_SCHEMA='act6_official_cpu_integration_v1'
PILOT_SCHEMA='m2_act_auxiliary_act_pilot_v4'
INIT_REF=dict(path=str(BASE/'diagnostics/act6-initialization-20261107-v1/INITIALIZATION.json'),bytes=47401,
    sha256='63a65a064605c48f89ecfa8718c1d1a7b0681e91812dcd35c40521b46aa81c0e')
APPROVED_CONFIG=dict(path=str(BASE/'diagnostics/act-conditional-official-integration-20261107-v1/train-config.json'),bytes=5346,
    sha256='17bd35c9f0bd39590d18b42f3f9c5e6f5d9751391dd296255a966b7b01ac3dfc')
LEGACY_OWNER=dict(path=str(BASE/'runs/ausc_policy/act5-legacy-stop-smoke-20261107-v1/result.json'),bytes=3401,
    sha256='f23f66c6e6d5b37b1c73ab2cc67a46f95c80891668702a531b0b0f57a92692df')
SOURCE26=Path('/opt/wam-runtime/code/wam-act-recovery-20261106-v26')
COMMIT26='05e972ca184383a525aff293fc9f3590e546da9c'
GROUPS=dict(native=dict(size=32,coefficient=.50),planned=dict(size=8,coefficient=.125),
    conditional=dict(size=8,coefficient=.125),oracle=dict(size=16,coefficient=.25))
STOP_KEY='stop_residual_trajectory'
STOP_CONFIG=dict(schema='m2_stop_residual_native_cache_replan_v1',enabled=True,max_cache_recalculations=8)
POSE_ID='p000474_a001230_f0000'
CASE_SPECS=(('p001120','train','R3'),('p001120','train','L3'),
    ('p000923','validation','R3'),('p000923','validation','L3'))
FALSE_FLAGS=dict(actual_training_started=False,actual_physics=False,fresh_payloads_opened=False,normalization_refit=False)


def need(condition,message):
    if not condition:raise ValueError(message)


def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference(path):
    path=Path(path);body=path.read_bytes()
    return dict(path=str(path),bytes=len(body),sha256=hashlib.sha256(body).hexdigest())


def require_host():
    need(sys.platform=='linux' and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and
        sys.executable==OFFICIAL and sys.flags.isolated==1 and not sys.flags.optimize,
        'Unoptimized official Linux -I -B CPU operator with hidden CUDA required')
    need(os.environ.get('AUSC_OWNED_RUN') is None,'External operator must not inherit GPU ownership')


def source_closure():
    """Full regular-file and closed-inventory authentication, never a filename-only check."""
    need(CODE.resolve()==CODE,'Exact resolved source28 root required')
    need((CODE/'CODE_COMMIT').read_text().strip()==COMMIT,'Exact source28 commit required')
    manifest=CODE/'SOURCE_MANIFEST.json'
    need(sha256(manifest)==MANIFEST_SHA256,'Exact source28 full manifest required')
    rows=json.loads(manifest.read_text());need(isinstance(rows,list) and len(rows)==FILE_COUNT,'Exact source28 file count required')
    names=set()
    for row in rows:
        need(isinstance(row,dict) and set(row)=={'file','bytes','sha256'},'Exact source inventory row required')
        name=row['file']
        need(isinstance(name,str) and bool(name) and not Path(name).is_absolute() and '..' not in Path(name).parts and
            Path(name).as_posix()==name and name not in names and name not in ('CODE_COMMIT','SOURCE_MANIFEST.json') and
            type(row['bytes']) is int and row['bytes']>=0,'Safe unique source inventory path required')
        names.add(name)
    actual=set()
    for path in CODE.rglob('*'):
        info=path.lstat();need(not stat.S_ISLNK(info.st_mode),'Symlink in immutable source28')
        if stat.S_ISDIR(info.st_mode):continue
        need(stat.S_ISREG(info.st_mode) and info.st_nlink==1,'Nonregular or hardlinked source28 file')
        actual.add(path.relative_to(CODE).as_posix())
    need(actual==names|{'CODE_COMMIT','SOURCE_MANIFEST.json'},'Unknown or omitted source inventory file')
    for row in rows:
        path=CODE/row['file']
        need(path.resolve()==path and reference(path)==dict(path=str(path),bytes=row['bytes'],sha256=row['sha256']),
            'Frozen source28 file changed: '+row['file'])
    return dict(root=str(CODE),code_commit=COMMIT,file_count=FILE_COUNT,
        source_manifest=reference(manifest),code_commit_file=reference(CODE/'CODE_COMMIT'))


def project_name(name):return name in ('sim','scripts','auscultation') or name.startswith(('sim.','scripts.','auscultation.'))


def module_origins(inventory=None):
    found={}
    for name,module in tuple(sys.modules.items()):
        if not project_name(name):continue
        path=getattr(module,'__file__',None);spec=getattr(module,'__spec__',None)
        if path is None:
            locations=list(getattr(module,'__path__',[]))
            need(locations and all(Path(p).resolve().is_relative_to(CODE) for p in locations),
                'Project namespace outside exact source28: '+name)
            continue
        need(isinstance(path,str) and spec is not None and spec.origin==path and Path(path).resolve()==Path(path) and
            Path(path).is_relative_to(CODE),'Project module origin outside actual source28: '+name)
        found[name]=dict(path=path,origin=spec.origin)
    if inventory is not None:
        for name,ref in inventory['files'].items():
            full='sim.ausc_policy.'+name[:-3]
            need(found.get(full)==dict(path=ref['path'],origin=ref['path']),
                'All22 actual V4 module origins required: '+full)
    return found


def pin_project_namespaces():
    # The official SDK may have installed a benign root `sim` namespace only.
    # It has executed no initializer; foreign project files/subnamespaces fail.
    for name,module in tuple(sys.modules.items()):
        if not project_name(name):continue
        path=getattr(module,'__file__',None)
        if path is not None:
            need(Path(path).resolve().is_relative_to(CODE),'Preimported project source outside source28: '+name)
        elif name not in ('sim','scripts'):
            locations=list(getattr(module,'__path__',[]))
            need(locations and all(Path(p).resolve().is_relative_to(CODE) for p in locations),
                'Preimported project namespace outside source28: '+name)
    for name in ('sim','scripts'):
        path=CODE/name;need(path.is_dir() and path.resolve()==path,'Exact source28 namespace root required')
        module=sys.modules.get(name)
        if module is None or getattr(module,'__file__',None) is None:
            module=ModuleType(name);module.__package__=name;module.__loader__=None
            module.__spec__=ModuleSpec(name,loader=None,is_package=True)
        module.__path__=[str(path)];module.__spec__.submodule_search_locations=module.__path__;sys.modules[name]=module


def current_inventory(runtime):
    value=runtime.aux.code_binding(version=4)
    need(value.get('code_commit')==COMMIT and len(value.get('files',{}))==22,'Exact current22 V4 numerical inventory required')
    for name,ref in value['files'].items():
        need(ref['path']==str(CODE/'sim/ausc_policy'/name) and reference(ref['path'])==ref,'Actual source28 numerical reference required')
        importlib.import_module('sim.ausc_policy.'+name[:-3])
    module_origins(value)
    return value


def bind_runtime():
    require_host();source_closure();sys.dont_write_bytecode=True;os.environ['PYTHONDONTWRITEBYTECODE']='1'
    os.chdir(CODE);sys.path.insert(0,str(CODE));pin_project_namespaces()
    modules=dict(development='scripts.act_student_recovery_round_20261106',rt='sim.ausc_policy.m2_given_contact_runtime',
        aux='sim.ausc_policy.m2_act_auxiliary_train',warm='sim.ausc_policy.m2_act_warm_start',head='sim.ausc_policy.m2_act_posture_head',
        scene_store='sim.ausc_policy.scene_store',residual='sim.ausc_policy.m2_stop_residual_trajectory',
        deployment='sim.ausc_policy.m2_given_contact_expanded_pilot_deployment',audit='sim.ausc_policy.m2_given_contact_pilot_native_audit',
        isaac_batch='sim.ausc_policy.isaac_batch',process_guard='sim.ausc_policy.process_guard')
    runtime=SimpleNamespace(**{key:importlib.import_module(name) for key,name in modules.items()})
    need(runtime.development.CODE==CODE and runtime.development.BASE==BASE and runtime.development.OFFICIAL==OFFICIAL and
        runtime.development.NATIVE==NATIVE,'Original owner helper binding changed')
    current_inventory(runtime);source_closure();return runtime


def checked_ref(rt,path,expected_sha256):
    ref=rt.reference(Path(path).resolve());need(ref['sha256']==expected_sha256,'External input SHA differs from actual pinned receipt')
    return ref


def validate_new_metadata(meta):
    counts=meta.get('counts',{})
    need(counts.get('parent')==dict(rows=35982,sources=934,attempted_slots=1109) and
        counts.get('new_source26')==dict(accepted_sources=6,samples=6,attempted_slots=6,subject_ids=['p001120']) and
        counts.get('groups',{}).get('oracle',{}).get('samples')==26 and
        counts['groups']['oracle'].get('supervised_cells')==518,'Actual complete ordered parent and new-six/120-cell data required')
    sources=[s for s in meta.get('sources',[]) if s.get('schema')=='m2_act_auxiliary_source_v4']
    targets={target:sum(s.get('target_id')==target for s in sources) for target in ('R3','L3')}
    need(len(sources)==6 and targets=={'R3':3,'L3':3} and len({s.get('source_id') for s in sources})==6 and all(
        s.get('subject_id')=='p001120' and s.get('split')=='train' and s.get('pose_id')==POSE_ID and
        s.get('kind')=='student_oracle_unexecuted' and len(s.get('windows',[]))==1 for s in sources),
        'Exactly six genuine original TRAIN p001120 sources, three each R3/L3, required')
    return dict(new_sources=6,new_samples=6,new_computed_cells=120,targets=targets,subject_ids=['p001120'],
        source_ids=[s['source_id'] for s in sources],actions_executed=False,physics_qualified=False)


def validate_export(runtime,ref):
    need(ref['path']==str(DATA_CONTROL),'Exact actual V4 data-export control path required')
    value=runtime.rt.read_bound(ref)
    need(value.get('passed') is True and value.get('code_commit')==COMMIT and
        value.get('source_manifest')==runtime.rt.reference(CODE/'SOURCE_MANIFEST.json') and
        all(value.get(k) is v for k,v in FALSE_FLAGS.items()),'Closed CPU source28 actual V4 export PASS required')
    prep=value.get('preparation');need(isinstance(prep,dict) and prep.get('path')==str(DATA_PREPARATION),
        'Actual exported preparation reference required')
    need(runtime.rt.reference(prep['path'])==prep,'Actual V4 preparation changed')
    meta=runtime.aux.auxiliary_metadata(prep,version=4)
    need(meta.get('counts')==value.get('counts') and len(meta.get('sources',[]))==940 and len(meta.get('slot_ledger',[]))==1115,
        'Actual940-source/1115-slot V4 metadata/export binding required')
    proof=validate_new_metadata(meta)
    return value,meta,proof


def new_source_array_proof(auxiliary):
    """Count the six actually loaded H2/H20 sources, including every pad mask."""
    import numpy as np
    sources=auxiliary['sources'];samples=auxiliary['samples']
    selected=[i for i,s in enumerate(sources) if s.get('schema')=='m2_act_auxiliary_source_v4']
    rows=np.flatnonzero(np.isin(samples['source_indices'],selected))
    need(len(selected)==len(rows)==6 and all(np.count_nonzero(samples['source_indices']==i)==1 for i in selected),
        'Actual six unique new-source rows required')
    mask=samples['setter_is_pad'][rows];histories=samples['source_histories'][rows]
    need(mask.shape==(6,20) and mask.dtype==np.bool_ and int((~mask).sum())==120 and
        histories.shape==(6,2,17) and np.isfinite(histories).all(),'Actual six H2state17/H20 sources with120 computed cells required')
    return dict(source_indices=selected,row_indices=rows.tolist(),computed_cells=120,actual_h2_shape=list(histories.shape),
        full_horizon=20,all_new_cells_valid=True,future_measured_histories=False,actions_executed=False)


def build_config(runtime,data_export):
    rt,aux,head=runtime.rt,runtime.aux,runtime.head
    need(rt.reference(APPROVED_CONFIG['path'])==APPROVED_CONFIG,'Exact original approved training configuration required')
    old=rt.read_bound(APPROVED_CONFIG);gpu=old.get('replacement_pod_gpu_binding')
    need(isinstance(gpu,dict) and bool(gpu),'Existing approved replacement-pod binding required; no substitute')
    need(rt.reference(INIT_REF['path'])==INIT_REF,'Exact actual CPU initialization receipt required')
    config=dict(aux.SCHEDULE,schema=aux.CONFIG_SCHEMA_V4,base=str(BASE),dataset=deepcopy(aux.NATIVE_DATASET),
        auxiliary=deepcopy(data_export['preparation']),evaluation_cohort=deepcopy(aux.TRAIN16_SCOPE),
        posture_head=head.pinned_spec(aux.native_normalization({'dataset':aux.NATIVE_DATASET}),version=2),
        initialization=deepcopy(INIT_REF),replacement_pod_gpu_binding=deepcopy(gpu))
    need(config['schema']==CONFIG_SCHEMA and aux._groups(4)==GROUPS,'Exact V4 mixed objective required')
    aux.validate_config(config);return config


def numerical_input_refs(rt,config):
    """Pin only already-authorized exported tensors/arrays and native statistics."""
    initial=rt.read_bound(config['initialization']);native=rt.read_bound(config['dataset']);auxiliary=rt.read_bound(config['auxiliary'])
    refs=dict(initial_state=initial['state_dict'],native_normalization=native['normalization'],
        native_train=native['arrays']['train'],native_validation=native['arrays']['validation'],
        auxiliary_metadata=auxiliary['metadata'],auxiliary_arrays=auxiliary['arrays'])
    for name,ref in refs.items():need(rt.reference(ref['path'])==ref,'Frozen numerical input changed: '+name)
    return deepcopy(refs)


def cpu_loss_probe(runtime,torch,policy,optimizer,batches):
    """One real training-mode loss/backward; never call optimizer.step()."""
    import numpy as np
    need(hasattr(policy,'_act6_initialization') and policy._act6_initialization.get('passed') is True,
        'Authenticated initialization required before CPU integration loss')
    need(not optimizer.state and not torch.cuda.is_initialized(),'Empty step0 optimizer and uninitialized CUDA required')
    before=runtime.warm.tensor_proofs(policy.state_dict());policy.train();optimizer.zero_grad(set_to_none=True)
    rng=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    loss,groups=runtime.aux.mixed_loss(policy,batches,version=4)
    need(loss.ndim==0 and bool(torch.isfinite(loss)),'Finite scalar official V4 loss required')
    poisoned={name:{key:value.clone() for key,value in batch.items()} for name,batch in batches.items()}
    pads=0
    for name,batch in poisoned.items():
        need(batch['action'].ndim==3 and tuple(batch['action'].shape[1:])==(20,7) and
            tuple(batch['action_is_pad'].shape)==tuple(batch['action'].shape[:2]),'Full original H20 pad mask required')
        mask=batch['action_is_pad'];pads+=int(mask.sum());batch['action'][mask]=1e6
        need(groups[name]['examples']==GROUPS[name]['size'] and groups[name]['coefficient']==GROUPS[name]['coefficient'],
            'Unchanged32/8/8/16 four-group objective required')
    need(pads>0,'Actual sampled batch must contain padding for a meaningful poison check')
    random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2])
    poison_loss,poison_groups=runtime.aux.mixed_loss(policy,poisoned,version=4)
    need(bool(torch.isfinite(poison_loss)) and torch.equal(loss.detach(),poison_loss.detach()) and groups==poison_groups,
        'Pad-only1e6 poisoning changed exact RNG-matched official loss')
    loss.backward();grads=[p.grad for p in policy.parameters() if p.grad is not None]
    need(grads and all(bool(torch.isfinite(g).all()) for g in grads),'Finite nonempty backward gradients required')
    norm=math.sqrt(sum(float(g.detach().double().square().sum()) for g in grads))
    need(math.isfinite(norm) and norm>0,'Positive finite CPU gradient norm required')
    after=runtime.warm.tensor_proofs(policy.state_dict())
    need(before==after and not optimizer.state and not torch.cuda.is_initialized(),
        'CPU integration updated policy tensors, optimizer state or CUDA')
    return dict(passed=True,loss=float(loss.detach()),groups=groups,pad_poison_equal=True,padded_cells_poisoned=pads,
        poisoned_action_value=1e6,gradient_norm=norm,gradient_tensors=len(grads),finite_gradients=True,
        policy_tensor_proofs_before=before,policy_tensor_proofs_after=after,all_tensor_bytes_unchanged=True,
        optimizer_state_entries=0,optimizer_steps=0,parameter_updates=0,cuda_initialized=False)


def actual_draws(cache,indices):
    rows={}
    for name,ids in indices.items():
        group='native' if name=='native' else 'auxiliary'
        rows[name]=[dict(index=int(i),subject_id=str(cache['subjects'][group][i]),
            source_id=str(cache['source_ids'][group][i]),kind=str(cache['kinds'][group][i])) for i in ids]
    return dict(purpose='CPU diagnostic sample only; zero optimized examples',groups=rows,optimizer_steps=0)


def run_integration(runtime,data_path,data_sha256):
    require_host();start=source_closure();rt=runtime.rt;operator=rt.reference(__file__)
    need(BASE.resolve()==BASE and not INTEGRATION_CONTROL.exists(),'Unique CPU integration control required; no retry')
    need(not Path(operator['path']).is_relative_to(CODE),'Operator must remain external')
    data_ref=checked_ref(rt,data_path,data_sha256);need(data_ref['path']==str(DATA_CONTROL),'Exact data export control required')
    inventory=current_inventory(runtime);INTEGRATION_CONTROL.mkdir(parents=True,exist_ok=False)
    rt.write_new(INTEGRATION_CONTROL/'STARTED.json',dict(operator=operator,source28=start,data_export_control=data_ref,
        started_utc=runtime.process_guard.now(),cpu_only=True,**FALSE_FLAGS))
    refs=[data_ref,deepcopy(INIT_REF),deepcopy(APPROVED_CONFIG)]
    def stable():
        need(source_closure()==start and rt.reference(__file__)==operator,'CPU source/operator changed')
        need(current_inventory(runtime)==inventory,'Current22 V4 numerical inventory changed')
        for ref in refs:need(rt.reference(ref['path'])==ref,'Frozen CPU integration input changed')
    try:
        data,meta,newproof=validate_export(runtime,data_ref);config=build_config(runtime,data)
        config_ref=rt.write_new(INTEGRATION_CONTROL/'train-config.json',config);refs.extend([config_ref,config['auxiliary'],config['dataset'],config['evaluation_cohort']])
        numeric_refs=numerical_input_refs(rt,config);refs.extend(numeric_refs.values())
        prepared=runtime.aux.load_inputs(config)
        need(prepared['auxiliary']['counts']==meta['counts'],'Loaded full V4 metadata/counts changed')
        need(len(prepared['auxiliary']['samples']['source_indices'])==35988,'Actual35988 V4 origins required')
        loaded_new=validate_new_metadata(prepared['auxiliary']);need(loaded_new==newproof,'Loaded new genuine source proofs changed')
        new_arrays=new_source_array_proof(prepared['auxiliary'])
        torch,policy,software=runtime.aux.make_model(config,device='cpu',training=False)
        runtime.aux.validate_official_forward(software)
        need(not torch.cuda.is_initialized() and sum(p.numel() for p in policy.parameters())==2658823,
            'Actual official CPU model2658823 with uninitialized CUDA required')
        execution=rt.execution_settings(torch,policy,device='cpu')
        initial=runtime.warm.initialize_policy(torch,policy,config['initialization'],
            normalization=prepared['normalization'],posture_head=config['posture_head'])
        need(initial.get('passed') is True and initial.get('initialization')==INIT_REF and initial.get('optimizer_step')==0 and
            initial.get('probe_indices')==[0,64,4186,8372] and initial.get('final_linear_reset') is False,
            'Actual strict tensor/prediction initialization PASS required')
        initial_ref=rt.write_new(INTEGRATION_CONTROL/'INITIALIZATION_EQUIVALENCE.json',initial);refs.append(initial_ref)
        optimizer,optimizer_initialization=runtime.warm.fresh_optimizer(torch,policy,lr=config['lr'],weight_decay=config['weight_decay'])
        cache=runtime.aux.build_cache(prepared,prepared['auxiliary'],torch=torch,device='cpu')
        import numpy as np
        indices=runtime.aux.sample_cached_indices(np.random.default_rng(config['seed']),cache)
        batches=runtime.aux.cached_batches(cache,indices);probe=cpu_loss_probe(runtime,torch,policy,optimizer,batches)
        need(probe['policy_tensor_proofs_before']==initial['initial_tensor_proofs'],
            'All model tensor bytes must remain exactly authenticated initialization bytes')
        stable();origins=module_origins(inventory)
        result=dict(schema=INTEGRATION_SCHEMA,passed=True,operator=operator,training_config=config_ref,
            data_export_control=data_ref,data_preparation=config['auxiliary'],initialization=deepcopy(INIT_REF),
            initialization_equivalence=initial_ref,optimizer_initialization=optimizer_initialization,
            source28=start,source28_end=source_closure(),code_inventory22=inventory,module_origins=origins,
            official_software=software,execution=execution,official_environment=dict(executable=sys.executable,prefix=sys.prefix,
                version=sys.version,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),cuda_initialized=torch.cuda.is_initialized()),
            cpu_integration=probe,actual_sample_draws=actual_draws(cache,indices),new_source_proof=newproof,new_source_array_proof=new_arrays,
            counts=meta['counts'],posture_target_audit=prepared['posture_target_audit'],
            numerical_input_refs=numeric_refs,stable_input_refs=deepcopy(refs),
            oracle_authentication=prepared['oracle_authentication'],cpu_only=True,cuda_initialized=False,
            parameter_updates=0,optimizer_steps=0,elapsed_s=time.monotonic()-START_TIME,**FALSE_FLAGS)
        stable();ref=rt.write_new(INTEGRATION_CONTROL/'COMPLETE.json',result)
        print(json.dumps(dict(complete=ref,training_config=config_ref,passed=True)),flush=True);return ref,result
    except BaseException as error:
        (INTEGRATION_CONTROL/'traceback.txt').write_text(traceback.format_exc())
        rt.write_new(INTEGRATION_CONTROL/'FAILED.json',dict(schema=INTEGRATION_SCHEMA,passed=False,operator=operator,
            source28=start,data_export_control=data_ref,error=repr(error),cpu_only=True,**FALSE_FLAGS));raise


def validate_integration_witness(value,config):
    source=value.get('source28',{});ref=value.get('training_config',{})
    need(value.get('schema')==INTEGRATION_SCHEMA and value.get('passed') is True and value.get('cpu_only') is True and
        value.get('cuda_initialized') is False and all(value.get(k) is v for k,v in FALSE_FLAGS.items()) and
        type(value.get('parameter_updates')) is int and value['parameter_updates']==0 and
        type(value.get('optimizer_steps')) is int and value['optimizer_steps']==0 and
        value.get('cpu_integration',{}).get('passed') is True,'Closed actual CPU-only integration PASS required before any GPU')
    need(source.get('root')==str(CODE) and source.get('code_commit')==COMMIT and source.get('file_count')==FILE_COUNT and
        source.get('source_manifest',{}).get('path')==str(CODE/'SOURCE_MANIFEST.json') and
        source['source_manifest'].get('sha256')==MANIFEST_SHA256 and len(value.get('code_inventory22',{}).get('files',{}))==22,
        'Actual full source28 closure/current22 integration binding required')
    need(ref.get('path')==str(INTEGRATION_CONTROL/'train-config.json') and config.get('schema')==CONFIG_SCHEMA and
        config.get('base')==str(BASE) and config.get('initialization')==value.get('initialization')==INIT_REF and
        config.get('auxiliary')==value.get('data_preparation') and
        value.get('data_export_control',{}).get('path')==str(DATA_CONTROL),
        'Exact new V4 config/initialization/data/control integration binding required')
    return value


class FiniteRound:
    def __init__(self,runtime):
        self.runtime=runtime;self.rt=runtime.rt
        self.source=self.source28=self.source28_end=self.operator=self.integration=self.training_config=None
        self.inventory=self.model=self.source_inventory=self.training=None
        self.data_export_control=self.data_preparation=None
        self.training_owner=self.smoke_owner=None
        self.normal_training_owner=self.normal_smoke_owner=self.native_requested=False
        self.inputs=[];self.config_refs=[];self.protocol_errors=[]
        self.outcomes=[dict(id=s+'-'+t.lower(),subject_id=s,split=split,target_id=t,pose_id=POSE_ID,
            config=None,source_episode_id=None,preflight_passed=False,qualification=None,qualification_attempted=False,
            passed=False,executed=False,infrastructure_error=False,error='Declared case has no independent physical qualification')
            for s,split,t in CASE_SPECS]

    def pin(self,ref):
        need(self.rt.reference(ref['path'])==ref,'Frozen actual round input changed')
        self.inputs.append(deepcopy(ref));return self.rt.read_bound(ref)

    def unchanged(self):
        current=source_closure();need(current==self.source28,'Full actual source28 closure changed');self.source28_end=current
        self.runtime.development.unchanged(self.source) # Truthful original legacy continuation catalog.
        need(current_inventory(self.runtime)==self.inventory,'Actual V4 model22 source inventory changed')
        if self.model is not None:
            need(self.runtime.deployment.model_sources(self.model)==self.source_inventory,'Actual model-specific native V4 inventory changed')
        need(self.rt.reference(__file__)==self.operator,'Frozen external operator changed')
        for ref in self.inputs+self.config_refs:need(self.rt.reference(ref['path'])==ref,'Frozen round/config input changed')

    def bind_integration(self,ref):
        value=self.pin(ref);config=self.pin(value.get('training_config',{}))
        validate_integration_witness(value,config)
        need(value.get('operator')==self.operator and value.get('source28')==value.get('source28_end')==self.source28 and
            value.get('code_inventory22')==self.inventory,'Exact external operator/source28/current22 integration required')
        self.runtime.aux.validate_config(config)
        need(self.rt.reference(INIT_REF['path'])==INIT_REF and config==build_config(self.runtime,{'preparation':config['auxiliary']}),
            'Actual initialization and approved unchanged GPU binding/config required')
        numeric_refs=numerical_input_refs(self.rt,config)
        need(numeric_refs==value.get('numerical_input_refs'),'Actual integrated numerical input references changed')
        self.inputs.extend(numeric_refs.values())
        data=self.pin(value['data_export_control']);_,_,newproof=validate_export(self.runtime,value['data_export_control'])
        self.data_export_control=deepcopy(value['data_export_control']);self.data_preparation=deepcopy(config['auxiliary'])
        need(newproof==value.get('new_source_proof') and data['preparation']==config['auxiliary'],
            'Actual data export/new-six integration proof differs')
        environment=value.get('official_environment',{})
        need(environment.get('executable')==OFFICIAL and environment.get('prefix')==str(Path(OFFICIAL).parent.parent) and
            environment.get('cuda_visible_devices')=='' and environment.get('cuda_initialized') is False,
            'Actual official uninitialized CPU integration environment required')
        self.runtime.rt.validate_execution_settings(value.get('execution',{}),device='cpu')
        self.runtime.aux.validate_official_forward(value.get('official_software',{}))
        origins=value.get('module_origins',{})
        for name,item in self.inventory['files'].items():
            need(origins.get('sim.ausc_policy.'+name[:-3])==dict(path=item['path'],origin=item['path']),
                'All22 actual recorded CPU module origins required')
        for name,item in origins.items():
            need(project_name(name) and item.get('path')==item.get('origin') and
                Path(item['path']).is_relative_to(CODE),'Recorded CPU project module outside current28')
        new_arrays=value.get('new_source_array_proof',{})
        need(new_arrays.get('computed_cells')==120 and new_arrays.get('actual_h2_shape')==[6,2,17] and
            new_arrays.get('all_new_cells_valid') is True,'Actual CPU six-source120-cell array proof required')
        initial=self.pin(value['initialization_equivalence']);probe=value['cpu_integration']
        receipt=self.pin(INIT_REF)
        need(initial.get('passed') is True and initial.get('initialization')==INIT_REF and
            initial.get('initial_tensor_proofs')==receipt.get('tensor_proofs')==probe.get('policy_tensor_proofs_before')==probe.get('policy_tensor_proofs_after') and
            probe.get('all_tensor_bytes_unchanged') is True and probe.get('pad_poison_equal') is True and
            probe.get('finite_gradients') is True and type(probe.get('gradient_norm')) in (int,float) and
            math.isfinite(probe['gradient_norm']) and probe['gradient_norm']>0 and probe.get('optimizer_steps')==0 and
            probe.get('optimizer_state_entries')==0 and probe.get('parameter_updates')==0,
            'Full actual tensor/pad/backward/step0 integration witness required')
        self.training_config=value['training_config']
        for key in ('dataset','auxiliary','evaluation_cohort'):self.pin(config[key])
        self.validate_legacy()
        native=self.rt.read_bound(config['dataset']);rows=[]
        for case in self.outcomes:
            matches=[row for row in native.get('episodes',[]) if row.get('origin')=='qualified_actual_nominal' and
                all(row.get(k)==case[k] for k in ('subject_id','split','target_id','pose_id'))]
            need(len(matches)==1,'Exact unique original four-case DEV bank row required')
            row=matches[0];original=self.pin(row['config']);route=original.get('m2_given_contact_control',{});meta=route.get('metadata',{})
            need(route.get('episode_id')==meta.get('episode_id')==row.get('episode_id') and
                all(meta.get(k)==case[k] for k in ('subject_id','split','target_id')),'Original DEV config identity differs')
            case['source_episode_id']=row['episode_id'];rows.append(row)
        need(len({r['episode_id'] for r in rows})==4,'Fixed four unique original DEV episodes required')
        return config,rows

    def validate_legacy(self):
        owner=self.pin(LEGACY_OWNER);guard=self.rt.reference(SOURCE26/'sim/ausc_policy/process_guard.py')
        expected=dict(path=guard['path'],sha256=guard['sha256'],gpu_exit_grace_s=self.runtime.process_guard.EXIT_GPU_GRACE_S)
        need(self.runtime.development.normal_owner(owner) and owner.get('run_id')=='act5-legacy-stop-smoke-20261107-v1' and
            owner.get('code_commit')==COMMIT26 and owner.get('resource_guard')==expected,
            'Prior original26 legacy owner must already be normal, closed, drained and GPU released')
        need(self.source['guard']['sha256']==guard['sha256'] and self.source['guard']['bytes']==guard['bytes'],
            'Original guard bytes must remain unchanged on28')

    def normal_owner(self,owner,run_id,config_ref):
        expected=dict(path=self.source['guard']['path'],sha256=self.source['guard']['sha256'],
            gpu_exit_grace_s=self.runtime.process_guard.EXIT_GPU_GRACE_S)
        return (self.runtime.development.normal_owner(owner) and owner.get('run_id')==run_id and
            owner.get('code_commit')==COMMIT and owner.get('config_sha256')==config_ref['sha256'] and owner.get('resource_guard')==expected)

    def prepare(self,complete,rows):
        deployment=self.runtime.deployment;need(self.runtime.residual.CONFIG==STOP_CONFIG,'Exact original STOP option required')
        jobs=[]
        for case,row in zip(self.outcomes,rows):
            try:
                original=self.rt.read_bound(row['config'])
                config=deployment.derive_config(original,source_config=row['config'],model=self.model,run_id=SMOKE_RUN,
                    source_inventory=self.source_inventory,scope=complete['scope'],stop_residual_trajectory=deepcopy(STOP_CONFIG))
                case['config']=self.rt.write_new(CONTROL/'smokes'/(case['id']+'.json'),config);self.config_refs.append(case['config'])
                need(STOP_KEY in config and config[STOP_KEY]==STOP_CONFIG and type(config[STOP_KEY]['enabled']) is bool and
                    type(config[STOP_KEY]['max_cache_recalculations']) is int,'Explicit exact STOP key required')
                projected=deepcopy(original);projected.pop('m2_given_contact_control');projected['fresh_expert'].pop('raw_multimodal_capture',None)
                if 'compact_scene_storage' in projected:projected['compact_scene_storage']['batch_id']=SMOKE_RUN
                projected=self.runtime.residual.configuration_projection(projected,config,metadata=config[deployment.ROUTE_KEY]['metadata'])
                need(projected=={k:v for k,v in config.items() if k!=deployment.ROUTE_KEY},
                    'Exact STOP projection must preserve original geometry/camera/force/dynamics/posture')
                output=BASE/'runs/ausc_policy'/SMOKE_RUN/'output'/case['id']
                deployment.preflight(config,output=output);self.runtime.scene_store.validate_config(config)
                case.update(preflight_passed=True,error=None)
                jobs.append(dict(id=case['id'],config=case['config']['path'],sha256=case['config']['sha256']))
            except Exception as error:
                case.update(passed=False,infrastructure_error=True,preparation_error=repr(error),error=repr(error))
        self.rt.write_new(CONTROL/'PREPARATION.json',dict(outcomes=deepcopy(self.outcomes),fixed_denominator=4))
        need(all(c['preflight_passed'] for c in self.outcomes),'Four-case CPU preflight failed; every case retained')
        batch=dict(experiment='isaac_batch',base=str(BASE),parallelism=2,time_limit_seconds=20000,jobs=jobs)
        self.runtime.isaac_batch.validate_jobs(batch);ref=self.rt.write_new(CONTROL/'smokes/batch.json',batch)
        self.config_refs.append(ref);return ref

    def qualify_all(self):
        for case in self.outcomes:
            if case['qualification_attempted']:continue
            case['qualification_attempted']=True
            output=BASE/'runs/ausc_policy'/SMOKE_RUN/'output'/case['id']
            config_path=case['config']['path'] if isinstance(case['config'],dict) else str(CONTROL/'smokes'/(case['id']+'.json'))
            try:
                # Missing config/native files are an explicit independent audit
                # outcome, including when preparation or original owner failed.
                ref=self.runtime.audit.qualify(output,config_path,receipt_path=CONTROL/'qualifications'/(case['id']+'.json'))
                receipt=self.rt.read_bound(ref);need(type(receipt.get('passed')) is bool,'Independent audit boolean verdict required')
                files=all((output/name).is_file() for name in ('fresh_expert_result.json','COMPLETE.json'))
                case.update(qualification=ref,passed=receipt['passed'],error=receipt.get('error'),native_files_present=files,
                    infrastructure_error=bool(case.get('infrastructure_error') or not files))
                case['physical_gate_passed']=receipt.get('checks',{}).get('unchanged_physical_gate_passed') is True
                runref=receipt.get('inputs',{}).get('run')
                if isinstance(runref,dict):
                    run=self.rt.read_bound(runref);steps=run.get('physics_steps')
                    case['executed']=type(steps) is int and steps>0
                    case['reported_physics_steps']=steps if type(steps) is int else None
                need(not case['passed'] or case['physical_gate_passed'],'Independent passed receipt must prove unchanged physical gate')
            except Exception as error:
                text=type(error).__name__+': '+str(error)
                case.update(passed=False,qualification=None,error=text,infrastructure_error=True)
                try:
                    case['audit_error']=self.rt.write_new(CONTROL/'qualifications'/(case['id']+'-AUDIT_ERROR.json'),
                        dict(schema='act6_external_audit_error_v1',passed=False,error=text,configuration=case['config'],actual_output=str(output)))
                except Exception as receipt_error:case['audit_receipt_error']=repr(receipt_error)

    def result(self):
        qualified=(self.normal_training_owner and self.normal_smoke_owner and not self.protocol_errors and
            len(self.outcomes)==4 and all(c['passed'] is True and c['qualification_attempted'] is True and
                c.get('physical_gate_passed') is True and c['executed'] is True and not c['infrastructure_error'] for c in self.outcomes))
        return dict(schema='act6_finite_development_v1',source=self.source,source28=self.source28,source28_end=self.source28_end,operator=self.operator,
            code_inventory22=self.inventory,source_inventory=self.source_inventory,integration=self.integration,
            continuation_snapshot_catalog='original unchanged helper legacy V3',data_export_control=self.data_export_control,
            data_preparation=self.data_preparation,
            training_config=self.training_config,initialization=deepcopy(INIT_REF),training=self.training,model=self.model,
            training_owner=self.training_owner,normal_training_owner=self.normal_training_owner,legacy_owner=LEGACY_OWNER,
            smoke_owner=self.smoke_owner,native_owner_requested=self.native_requested,normal_smoke_owner=self.normal_smoke_owner,
            outcomes=deepcopy(self.outcomes),protocol_errors=deepcopy(self.protocol_errors),fixed_denominator=4,declared_cases=4,
            native_requested_cases=4 if self.native_requested else 0,executed_cases=sum(c['executed'] is True for c in self.outcomes),
            physical_successes=sum(c['passed'] is True for c in self.outcomes),
            qualification_attempted_cases=sum(c['qualification_attempted'] for c in self.outcomes),
            stop_configuration=deepcopy(STOP_CONFIG),stop_adapter_enabled=True,stop_adapter_requested=self.native_requested,
            stop_adapter_tested=qualified,combined_full_treatment_tested=qualified,qualified_dev_only=qualified,
            model_qualified_dev_only=qualified,qualification_scope='four original known TRAIN/selection-VAL development cases only',
            treatment='fine-tuned ACT6: ACT5 selected19000 initialization plus20000 new optimizer steps, terminal-query data and explicit STOP',
            model_qualified=False,formal_acceptance=False,heldout_executed=False,generalization_qualified=False,
            fresh_identities_exposed=False,fresh_unexposed=True,physics_gates_changed=False,expert_l2_fallback=False,
            automatic_retry=False,automatic_resume=False)

    def recover_owner(self,run_id):
        path=BASE/'runs/ausc_policy'/run_id/'result.json'
        return self.rt.reference(path) if path.is_file() else None

    def run(self,integration_path,integration_sha256):
        require_host();self.source28=source_closure()
        need(BASE.resolve()==BASE and not CONTROL.exists(),'Unique finite round control required; no retry/resume')
        for run_id in (TRAIN_RUN,SMOKE_RUN):need(not (BASE/'runs/ausc_policy'/run_id).exists(),'Unique owner run required; no retry/resume')
        self.integration=checked_ref(self.rt,integration_path,integration_sha256)
        need(self.integration['path']==str(INTEGRATION_CONTROL/'COMPLETE.json'),'Exact closed CPU integration path required')
        self.inventory=current_inventory(self.runtime);self.source=self.runtime.development.snapshot();self.operator=self.rt.reference(__file__)
        need(self.source['source_manifest']==self.rt.reference(CODE/'SOURCE_MANIFEST.json') and
            self.source['code_commit']==self.rt.reference(CODE/'CODE_COMMIT'),'Original continuation snapshot must retain actual28 manifest/commit')
        need(not Path(self.operator['path']).is_relative_to(CODE),'External operator required')
        CONTROL.mkdir(parents=True,exist_ok=False)
        self.rt.write_new(CONTROL/'STARTED.json',dict(source=self.source,source28=self.source28,operator=self.operator,integration=self.integration,
            code_inventory22=self.inventory,started_utc=self.runtime.process_guard.now(),training_run=TRAIN_RUN,smoke_run=SMOKE_RUN,
            declared_cases=deepcopy(self.outcomes),fixed_denominator=4,heldout_execution=False,automatic_retry=False,automatic_resume=False))
        phase='closed_cpu_integration_admission'
        try:
            config,rows=self.bind_integration(self.integration);self.unchanged();phase='owned_new20k_training'
            try:
                self.training_owner,owner=self.runtime.development.run_owner(
                    self.training_config,TRAIN_RUN,self.runtime.development.OFFICIAL,self.source,control=CONTROL)
                self.normal_training_owner=self.normal_owner(owner,TRAIN_RUN,self.training_config)
                need(self.normal_training_owner,'New original training owner did not close normally/drain/release')
            except BaseException:
                self.training_owner=self.recover_owner(TRAIN_RUN);raise
            self.unchanged();self.training=self.rt.reference(BASE/'runs/ausc_policy'/TRAIN_RUN/'output/COMPLETE.json')
            self.inputs.append(deepcopy(self.training));self.model,complete=self.runtime.deployment.model_binding(self.training)
            need(complete.get('schema')==PILOT_SCHEMA and complete.get('complete') is True and complete.get('completed_steps')==20000 and
                complete.get('inputs',{}).get('config')==self.training_config and complete['inputs'].get('initialization')==INIT_REF and
                complete.get('schedule')==config and len(complete.get('evaluations',[]))==40 and
                self.model.get('checkpoint_schema')==PILOT_SCHEMA and self.model.get('complete')==self.training,
                'Actual new V4 full20k/40-evaluation completed model required; never reuse ACT5')
            self.runtime.aux.validate_code(complete['inputs']['code'],version=4)
            need(complete['inputs']['code']==self.inventory,'Actual new model numerical22/current28 binding required')
            self.source_inventory=self.runtime.deployment.model_sources(self.model)
            need(all(self.source_inventory['files'].get(k)==v for k,v in self.inventory['files'].items()),
                'Model-specific deployment inventory must include every current V4 module')
            for key in ('best','cpu_equivalence'):self.pin(complete[key])
            self.rt.write_new(CONTROL/'TRAINED.json',dict(source=self.source,source28=self.source28,operator=self.operator,
                training_config=self.training_config,initialization=INIT_REF,training_owner=self.training_owner,
                complete=self.training,model=self.model,source_inventory=self.source_inventory))
            phase='four_dev_cpu_preflight';batch=self.prepare(complete,rows);self.unchanged()
            phase='owned_four_dev';self.native_requested=True;owner_error=None
            try:
                self.smoke_owner,owner=self.runtime.development.run_owner(
                    batch,SMOKE_RUN,self.runtime.development.NATIVE,self.source,control=CONTROL)
                self.normal_smoke_owner=self.normal_owner(owner,SMOKE_RUN,batch)
                need(self.normal_smoke_owner,'Original native owner did not close normally/drain/release')
            except BaseException as error:self.smoke_owner=self.recover_owner(SMOKE_RUN);owner_error=error
            phase='independent_four_dev_qualification';self.qualify_all();self.unchanged()
            if owner_error is not None:raise owner_error
            result=self.result();self.rt.write_new(CONTROL/'DEVELOPMENT_RESULTS.json',result)
            terminal='DEVELOPMENT_COMPLETE.json' if result['qualified_dev_only'] else 'NEEDS_REPAIR.json'
            self.rt.write_new(CONTROL/terminal,result)
            print(json.dumps(dict(terminal=self.rt.reference(CONTROL/terminal),physical_successes=result['physical_successes'],fixed_denominator=4)),flush=True)
            return result
        except BaseException as error:
            self.protocol_errors.append(dict(phase=phase,error=repr(error)))
            (CONTROL/'traceback.txt').write_text(traceback.format_exc())
            # All four audits run even with missing configs/native output and any
            # abnormal owner; independent diagnostic failures cannot vanish.
            self.qualify_all();result=self.result()
            if not (CONTROL/'DEVELOPMENT_RESULTS.json').exists():self.rt.write_new(CONTROL/'DEVELOPMENT_RESULTS.json',result)
            self.rt.write_new(CONTROL/'FAILED.json',dict(result,phase=phase,error=repr(error)))
            raise


START_TIME=time.monotonic()

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='phase',required=True)
    p=sub.add_parser('integration');p.add_argument('--data-complete',required=True);p.add_argument('--data-sha256',required=True)
    p=sub.add_parser('round');p.add_argument('--integration-complete',required=True);p.add_argument('--integration-sha256',required=True)
    args=parser.parse_args();runtime=bind_runtime()
    if args.phase=='integration':run_integration(runtime,args.data_complete,args.data_sha256)
    else:FiniteRound(runtime).run(args.integration_complete,args.integration_sha256)


if __name__=='__main__':main()
