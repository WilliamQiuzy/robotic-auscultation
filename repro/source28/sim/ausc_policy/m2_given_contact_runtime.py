"""Official given-contact ACT runtime and finite numerical protocol.

This module imports no Torch until an explicit official-environment constructor.
Inference only consumes two causal observation frames. Synthetic runtime probes
are separately typed; they cannot qualify a native capture or a trained model.
The existing production factory remains refused until the owned Task5C bridge.
"""
from copy import deepcopy
from contextlib import contextmanager
import argparse
import base64
import csv
import fcntl
import hashlib
from importlib import metadata as distributions
import inspect
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

import numpy as np

from . import m2_act as shared, m2_given_contact_act as given
from . import m2_given_contact_dataset as dataset, m2_given_contact_pose as pose
from . import m2_given_contact_handoff as handoff, geometry_epoch as geometry

need=shared.need
VENDOR_ROOT='/opt/wam-runtime/vendor/lerobot-eb7a54c'
UPSTREAM_COMMIT=shared.UPSTREAM_COMMIT
VENDOR_PYTHON_FILES=880
RUNTIME_SCHEMA='m2_given_contact_official_runtime_v1'
FIXTURE_SCHEMA='m2_given_contact_numeric_runtime_fixture_v1'
FIXTURE_RESULT_SCHEMA='m2_given_contact_numeric_runtime_result_v1'
FORMAL_RUN_SCHEMA='m2_given_contact_formal_act_training_v1'
MODEL_SEED=20261026
FIXTURE_SEED=20261028
EVAL_STEPS=list(range(500,20001,500))
BATCH_SIZE=64
OBSERVATION_KEYS=('observation.state','observation.environment_state')
GPU_LOCK='/workspace/wam/locks/isaac_gpu.lock'
GPU_LOCK_ID=(47,288098408)
GUARD_SHA256='35305b9151a1da5046023c4462acbfb50aadc2d997b073f4acabc356bd16dc5d'
OWNED_STAGE_SCHEMA='m2_given_contact_owned_stage_v1'
OWNED_MODULES={'m2_given_contact_numeric_fixture':'sim.ausc_policy.m2_given_contact_runtime',
    'm2_given_contact_train':'sim.ausc_policy.m2_given_contact_train'}


def numerical_protocol():
    return dict(steps=20000,batch_size=64,model_seed=MODEL_SEED,minibatch_seed=MODEL_SEED,
        augmentation_seed=20261027,learning_rate=1e-4,weight_decay=1e-4,betas=[.9,.999],eps=1e-8,
        grad_norm=1.,q_noise_std=.002,q_dropout=.05,dtype='float32',amp=False,tf32=False,
        deterministic_algorithms=True,cublas_workspace_config=':4096:8',float32_matmul_precision='highest',
        intraop_threads=2,interop_threads=1,evaluation_steps=EVAL_STEPS,wall_timeout_s=21600,
        prediction_atol=1e-6,prediction_rtol=1e-5,metric_atol=1e-7,metric_rtol=1e-5,
        cpu_prediction_identity='C-order little-endian float64 bytes',selection='earliest exact full-VAL H20 MAE argmin',
        train_gate='ACT <= .8 HOLD AND ACT <= .8 CV; no tolerance',automatic_resume=False)


@contextmanager
def owned_gpu(binding=None):
    """Exactly one child lock; default original resource identity is unchanged."""
    if binding is not None:
        from .rebuild_gpu_binding import owned_gpu as replacement_owned_gpu
        with replacement_owned_gpu(binding) as ownership:yield ownership
        return
    from . import process_guard as guard
    need(sys.platform=='linux' and os.environ.get('AUSC_OWNED_RUN')=='1' and
         os.environ.get('CUDA_VISIBLE_DEVICES')=='0', 'Original owned GPU process required')
    pids=sorted(guard.gpu_pids());need(not pids, 'Conflicting GPU process before CUDA initialization')
    path=Path(GPU_LOCK);need(not path.is_symlink(), 'Original GPU lock cannot be replaced by a link')
    with path.open('r+') as lock:
        stat=os.fstat(lock.fileno());need((stat.st_dev,stat.st_ino)==GPU_LOCK_ID, 'Original GPU lock identity changed')
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield dict(path=str(path),device=stat.st_dev,inode=stat.st_ino,pid=os.getpid(),checks=dict(gpu_pids_before=pids))
        finally:fcntl.flock(lock.fileno(),fcntl.LOCK_UN)


def reference(path):
    path=dataset._path(str(path),Path.cwd())
    need(path.is_file(), 'Required immutable runtime input missing')
    return dict(path=str(path),bytes=path.stat().st_size,sha256=shared.sha256(path))


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,allow_nan=False,sort_keys=True,indent=2);stream.write('\n')
    return reference(path)


def digest(value):return hashlib.sha256(pose.canonical(value)).hexdigest()


def task_contract():
    return dict(task_id=pose.TASK_ID,target_source=pose.TARGET_SOURCE,contract_id=pose.CONTRACT_ID,
        geometry_id=geometry.GEOMETRY_ID,geometry_source_sha256=geometry.GEOMETRY_SOURCE_SHA256,
        identity_split_sha256=shared.contract.IDENTITY_SPLIT_SHA256,action_semantics_id=given.ACTION_SEMANTICS_ID,
        handoff_policy_id=handoff.POLICY_ID,action_representation='block_relative_joint',
        observation_contract=pose.observation_contract(),state_dim=17,history_steps=2,horizon=20,
        execution_prefix=10,physics_hz=240,control_hz=20)


def validate_normalization(stats):
    need(stats.get('schema')==given.NORMALIZATION_SCHEMA and stats.get('task_id')==pose.TASK_ID and
         stats.get('action_semantics_id')==given.ACTION_SEMANTICS_ID and stats.get('fitted_split')=='train' and
         stats.get('observation_contract')==pose.observation_contract(), 'Formal TRAIN-only normalization required')
    for key,n in [('state_mean',17),('state_std',17),('action_mean',7),('action_std',7)]:
        value=np.asarray(stats.get(key),dtype=np.float64)
        need(value.shape==(n,) and np.isfinite(value).all() and ('std' not in key or (value>=1e-6).all()),
             'Invalid finite normalization '+key)
    return stats


def canonical_predictions(value):
    value=np.asarray(value)
    need(value.ndim==3 and value.shape[0]>0 and value.shape[1:]==(20,7) and value.dtype.kind in 'fiu' and
         np.isfinite(value).all(), 'Complete finite prediction array [N,20,7] required')
    return np.ascontiguousarray(value,dtype='<f8')


def save_predictions(path,value,*,window_order_sha256):
    value=canonical_predictions(value)
    need(re.fullmatch('[0-9a-f]{64}',str(window_order_sha256)), 'Exact prediction window order required')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:np.lib.format.write_array(stream,value,version=(1,0),allow_pickle=False)
    return dict(reference=reference(path),window_order_sha256=window_order_sha256,
        array=dict(shape=list(value.shape),dtype='<f8',order='C',sha256=hashlib.sha256(value.tobytes()).hexdigest()))


def load_predictions(binding,*,window_order_sha256):
    ref=binding.get('reference');path=dataset._ref_path(ref,Path.cwd())
    need(reference(path)==ref and binding.get('window_order_sha256')==window_order_sha256,
         'Prediction file/window identity changed')
    value=np.load(path,allow_pickle=False)
    expected=binding.get('array',{})
    need(value.dtype.str=='<f8' and value.flags.c_contiguous and expected.get('shape')==list(value.shape) and
         expected.get('dtype')=='<f8' and expected.get('order')=='C' and
         hashlib.sha256(value.tobytes()).hexdigest()==expected.get('sha256'), 'Canonical prediction shape/dtype/bytes changed')
    return canonical_predictions(value)


def metrics(predicted,samples):
    result=shared.evaluate_predictions(canonical_predictions(predicted),samples)
    result['metric_space']='raw target-space MAE; not measured tracking'
    validate_metrics(result)
    return result


def validate_metrics(value):
    need(type(value.get('sample_count')) is int and value['sample_count']>0 and value.get('horizon')==20 and
         value.get('selection_metric')=='act.full_chunk_mae_rad', 'Complete full-H20 metric/count required')
    for name in ('act','hold','constant_velocity'):
        part=value.get(name,{})
        score=part.get('full_chunk_mae_rad');per=np.asarray(part.get('per_horizon_mae_rad'),dtype=np.float64)
        need(type(score) in (int,float) and np.isfinite(score) and score>=0 and per.shape==(20,) and
             np.isfinite(per).all() and (per>=0).all(), 'Finite nonnegative full/per-horizon metric required: '+name)
    return value


def admission_gate(value):
    validate_metrics(value)
    act=value['act']['full_chunk_mae_rad'];hold=value['hold']['full_chunk_mae_rad'];cv=value['constant_velocity']['full_chunk_mae_rad']
    a=act<=.8*hold;b=act<=.8*cv
    return dict(passed=a and b,beats_hold_20_percent=a,beats_cv_20_percent=b,
        act_over_hold=None if hold==0 else act/hold,act_over_cv=None if cv==0 else act/cv,
        hold_threshold_rad=.8*hold,cv_threshold_rad=.8*cv,metric_space='raw target-space MAE; no tolerance added to admission')


def earliest_argmin(rows):
    need(isinstance(rows,list) and [r.get('step') for r in rows]==EVAL_STEPS,
         'Exactly forty complete ordered VAL snapshots required')
    for row in rows:validate_metrics(row.get('metrics',{}))
    need(len({r['metrics']['sample_count'] for r in rows})==1, 'Validation window count changed across snapshots')
    return min(rows,key=lambda r:(r['metrics']['act']['full_chunk_mae_rad'],r['step']))['step']


def compare_devices(cpu,cuda,cpu_metrics,cuda_metrics):
    cpu=canonical_predictions(cpu);cuda=canonical_predictions(cuda)
    need(cpu.shape==cuda.shape and np.all(abs(cpu-cuda)<=1e-6+1e-5*np.maximum(abs(cpu),abs(cuda))),
         'CPU/CUDA prediction shape or elementwise tolerance failed')
    validate_metrics(cpu_metrics);validate_metrics(cuda_metrics)
    need(cpu_metrics['sample_count']==cuda_metrics['sample_count']==len(cpu), 'CPU/CUDA window count differs')
    for name in ('act','hold','constant_velocity'):
        for key in ('full_chunk_mae_rad','per_horizon_mae_rad'):
            a=np.asarray(cpu_metrics[name][key]);b=np.asarray(cuda_metrics[name][key])
            need(a.shape==b.shape and np.all(abs(a-b)<=1e-7+1e-5*np.maximum(abs(a),abs(b))),
                 'CPU/CUDA full/per-horizon metric tolerance failed')
    return dict(passed=True,prediction_max_abs_rad=float(abs(cpu-cuda).max()),
        prediction_atol=1e-6,prediction_rtol=1e-5,metric_atol=1e-7,metric_rtol=1e-5)


def pack_observations(samples,stats,indices):
    validate_normalization(stats)
    history=np.asarray(samples['histories'],dtype=np.float64);states=np.asarray(samples['states'],dtype=np.float64)
    need(history.shape==(len(states),2,17) and states.ndim==2 and states.shape[1:]==(17,) and
         np.isfinite(history).all() and np.array_equal(history[:,1],states), 'Two real finite causal observation frames required')
    history=(history[indices]-np.asarray(stats['state_mean']))/np.asarray(stats['state_std'])
    return {'observation.state':history[:,1,:7].astype(np.float32),
        'observation.environment_state':np.concatenate([history[:,1,7:],history[:,0]],axis=1).astype(np.float32)}


def validate_runtime_reference(checkpoint,metadata):
    need(checkpoint.get('task_contract')==task_contract() and metadata.get('handoff_policy_id')==handoff.POLICY_ID,
         'Checkpoint/runtime common task/geometry/action/handoff contract differs')
    ref=pose.require_route(metadata)
    # No comparison with the first (or any) TRAIN reference. Runtime provenance
    # is fresh, while checkpoint provenance retains every training reference.
    return ref


def _dependency_inventory(installed=None):
    result=[];cached={};prefix=Path(sys.prefix).resolve()
    def payload(path,algorithm='sha256'):
        need(path.is_file() and not any(p.is_symlink() for p in (path,*path.parents)), 'Dependency RECORD payload missing or linked')
        cache_key=(str(path),algorithm)
        if cache_key not in cached:
            before=path.stat();digest=hashlib.new(algorithm);actual=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):
                    digest.update(block)
                    if algorithm!='sha256':actual.update(block)
            after=path.stat()
            need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==
                 (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns), 'Dependency payload changed while hashing')
            cached[cache_key]=(dict(path=str(path),bytes=after.st_size,
                sha256=digest.hexdigest() if algorithm=='sha256' else actual.hexdigest()),
                base64.urlsafe_b64encode(digest.digest()).decode().rstrip('='))
        return cached[cache_key]
    for distribution in distributions.distributions() if installed is None else installed:
        raw_location=Path(distribution._path)
        need(not any(p.is_symlink() for p in (raw_location,*raw_location.parents)), 'Dependency metadata location cannot be linked')
        location=raw_location.resolve()
        need(location.is_dir(), 'Dependency metadata location missing')
        files={}
        for path in sorted(location.rglob('*')):
            need(not path.is_symlink(), 'Dependency metadata cannot hide linked files')
            if path.is_file():files[path.relative_to(location).as_posix()]=payload(path)[0]
        name=distribution.metadata['Name'];record=location/'RECORD';payloads={}
        if record.is_file():
            need(prefix in location.parents, 'Dependency RECORD metadata lies outside official prefix')
            with record.open(newline='') as stream:entries=list(csv.reader(stream))
            need(entries and all(len(row)==3 for row in entries), 'Dependency RECORD requires exact path/hash/size rows')
            for relative,declared,size in entries:
                need(relative and not Path(relative).is_absolute() and '\\' not in relative, 'Invalid dependency RECORD path')
                # Wheel script paths such as ../../../bin/foo are legitimate.
                # Resolve them against this distribution's actual metadata parent,
                # then require the installed prefix, without following links.
                raw=location.parent/relative
                need(not any(p.is_symlink() for p in (raw,*raw.parents)), 'Dependency RECORD path traverses a link')
                path=raw.resolve()
                need(prefix in path.parents and str(path) not in payloads, 'Dependency RECORD escape or duplicate path')
                scope='wheel_record_hash';algorithm='sha256';expected=None
                if declared:
                    parts=declared.split('=',1)
                    need(len(parts)==2 and parts[0] in ('sha256','sha384','sha512') and
                         re.fullmatch('[A-Za-z0-9_-]+',parts[1]), 'Unsupported dependency RECORD hash')
                    algorithm,expected=parts
                    need(size.isdecimal(), 'Dependency RECORD hash requires declared bytes')
                else:
                    need(path.parent==location and path.name in ('RECORD','INSTALLER','REQUESTED','direct_url.json'),
                         'Dependency RECORD blank hash outside explicit installer metadata scope')
                    scope='installer_record_self' if path==record else 'installer_generated_metadata'
                ref,actual=payload(path,algorithm)
                need((expected is None or actual==expected) and (not size or size.isdecimal() and int(size)==ref['bytes']),
                     'Dependency RECORD payload hash/size changed: '+str(path))
                payloads[str(path)]=dict(reference=ref,record_path=relative,record_hash=declared,
                    record_size=size,record_scope=scope)
            need(str(record) in payloads, 'Dependency RECORD must bind its own installer metadata bytes')
            source=dict(kind='wheel_RECORD_files',record=files['RECORD'])
        else:
            # The installed editable distribution also exposes its original
            # egg-info occurrence. Its code is the separately verified full
            # 882-file pinned vendor inventory, not a blanket no-RECORD escape.
            need(name.lower()=='lerobot' and location==Path(VENDOR_ROOT)/'src/lerobot.egg-info' and
                 {'PKG-INFO','SOURCES.txt'}<=set(files), 'Dependency lacks an authenticated wheel RECORD or exact editable vendor role')
            source=dict(kind='editable_pinned_vendor',root=VENDOR_ROOT,upstream_commit=UPSTREAM_COMMIT,
                inventory='software.vendor.files')
        result.append(dict(name=distribution.metadata['Name'],version=distribution.version,
            metadata_location=str(location),files=files,payloads=payloads,payload_source=source))
    # Preserve every distribution occurrence, including repeated names/paths.
    return sorted(result,key=lambda r:(r['name'].lower(),r['metadata_location'],r['version']))


def vendor_inventory(root):
    root=dataset._path(str(root),Path.cwd())
    commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    need(commit==UPSTREAM_COMMIT, 'Official vendor commit differs')
    raw=subprocess.check_output(['git','-C',str(root),'ls-files','-z','--','*.py','pyproject.toml','uv.lock'])
    names=raw.decode().split('\0');names=[n for n in names if n]
    need(len(names)==VENDOR_PYTHON_FILES+2 and sum(n.endswith('.py') for n in names)==VENDOR_PYTHON_FILES and
         {'pyproject.toml','uv.lock'}<=set(names) and len(names)==len(set(names)), 'Full official vendor Python/config/lock inventory required')
    untracked=subprocess.check_output(['git','-C',str(root),'ls-files','--others','--exclude-standard','--','*.py'])
    live={p.relative_to(root).as_posix() for p in (root/'src').rglob('*.py')}
    need(not untracked and live=={n for n in names if n.startswith('src/') and n.endswith('.py')},
         'Official vendor has untracked importable Python sources')
    entries=subprocess.check_output(['git','-C',str(root),'ls-tree','-rz','HEAD','--',*names]).split(b'\0')
    blobs={}
    for entry in filter(None,entries):
        head,name=entry.split(b'\t',1);mode,kind,sha=head.split()
        need(mode in (b'100644',b'100755') and kind==b'blob', 'Regular pinned vendor blob required')
        blobs[name.decode()]=sha.decode()
    need(set(blobs)==set(names), 'Complete pinned vendor blob inventory required')
    for name,expected in blobs.items():
        path=dataset._path(str(root/name),root);body=path.read_bytes()
        need(hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()==expected,
             'Official vendor bytes differ from pinned HEAD blob: '+name)
    subprocess.run(['git','-C',str(root),'diff','--exit-code','HEAD','--',*names],check=True,stdout=subprocess.DEVNULL)
    return dict(upstream_commit=commit,root=str(root),files={name:reference(root/name) for name in sorted(names)})


def software_inventory(root):
    vendor=vendor_inventory(root)
    return dict(schema=RUNTIME_SCHEMA,vendor=vendor,dependencies=_dependency_inventory(),
        python=reference(Path(sys.executable).resolve()),prefix=sys.prefix,python_version=sys.version,
        source_bindings={str(Path(m.__file__).resolve()):shared.sha256(m.__file__) for m in (shared,given,pose,handoff,sys.modules[__name__])})


def make_model(config,*,device):
    root=Path(config.get('lerobot_root',''))
    need(str(root)==VENDOR_ROOT and Path(sys.prefix).resolve()==(root/'.venv').resolve(), 'Use the pinned official ACT environment')
    need(device in ('cpu','cuda'), 'Explicit CPU/CUDA device required')
    if device=='cpu':need(os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'CPU runtime requires CUDA hidden')
    else:need(sys.platform=='linux' and os.environ.get('AUSC_OWNED_RUN')=='1', 'CUDA requires an owned Linux job')
    need(os.environ.get('CUBLAS_WORKSPACE_CONFIG',':4096:8')==':4096:8', 'Deterministic CUBLAS workspace must be fixed before CUDA')
    if 'torch' in sys.modules:need(not sys.modules['torch'].cuda.is_initialized(), 'Fresh official process required before CUDA initialization')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    if str(root/'src') not in sys.path:sys.path.insert(0,str(root/'src'))
    inventory=software_inventory(root)
    import torch
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(2)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    random.seed(MODEL_SEED);np.random.seed(MODEL_SEED);torch.manual_seed(MODEL_SEED)
    if device=='cuda':
        need(torch.cuda.is_available(), 'CUDA unavailable; no device substitution')
        torch.cuda.manual_seed_all(MODEL_SEED)
    from lerobot.configs.types import FeatureType,NormalizationMode,PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    need(Path(inspect.getfile(ACTPolicy)).resolve()==root/'src/lerobot/policies/act/modeling_act.py', 'Wrong official ACT import')
    cfg=ACTConfig(input_features={'observation.state':PolicyFeature(FeatureType.STATE,(7,)),
        'observation.environment_state':PolicyFeature(FeatureType.ENV,(27,))},output_features={'action':PolicyFeature(FeatureType.ACTION,(7,))},
        normalization_mapping={key:NormalizationMode.IDENTITY for key in ('STATE','ENV','ACTION')},
        **shared.native_spec(),device=device,use_amp=False,push_to_hub=False,optimizer_lr=1e-4,optimizer_weight_decay=1e-4)
    need(cfg.device==device, 'Unexpected automatic device substitution')
    policy=ACTPolicy(cfg).to(device)
    need(not hasattr(policy.model,'backbone') and not hasattr(policy.model,'vae_encoder'), 'Image/VAE encoder forbidden')
    need(all(p.dtype==torch.float32 and p.device.type==device for p in policy.parameters()), 'Explicit FP32 model/device required')
    return torch,policy,inventory


def execution_settings(torch,policy,*,device):
    value=dict(device=device,parameter_devices=sorted({p.device.type for p in policy.parameters()}),
        parameter_dtypes=sorted({str(p.dtype) for p in policy.parameters()}),
        default_dtype=str(torch.get_default_dtype()),intraop_threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        matmul_precision=torch.get_float32_matmul_precision(),matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        torch_version=str(torch.__version__),cuda_version=torch.version.cuda,
        device_name=torch.cuda.get_device_name(0) if device=='cuda' else 'cpu')
    validate_execution_settings(value,device=device);return value


def validate_execution_settings(value,*,device):
    expected=dict(device=device,parameter_devices=[device],parameter_dtypes=['torch.float32'],default_dtype='torch.float32',
        intraop_threads=2,interop_threads=1,deterministic_algorithms=True,matmul_precision='highest',matmul_tf32=False,
        cudnn_tf32=False,cudnn_deterministic=True,cudnn_benchmark=False,cublas_workspace_config=':4096:8')
    need(all(value.get(key)==wanted for key,wanted in expected.items()) and value.get('torch_version') and
         value.get('device_name') and (device=='cpu' or value.get('cuda_version')), 'Actual deterministic FP32/thread/device settings differ')
    return value


def _reference_shape(ref):
    need(isinstance(ref,dict) and Path(str(ref.get('path',''))).is_absolute() and
         re.fullmatch('[0-9a-f]{64}',str(ref.get('sha256',''))), 'Absolute immutable path/SHA256 required')


def read_bound(ref):
    _reference_shape(ref)
    value,path=dataset._bound(ref,Path.cwd())
    if 'bytes' in ref:need(path.stat().st_size==ref['bytes'], 'Immutable input byte count differs')
    return value


def bound_path(ref):
    _reference_shape(ref);path=dataset._ref_path(ref,Path.cwd())
    need(path.is_file() and shared.sha256(path)==ref['sha256'] and
         ('bytes' not in ref or path.stat().st_size==ref['bytes']), 'Immutable artifact path/size/SHA256 changed')
    return path


def same_reference(a,b):return reference(bound_path(a))==reference(bound_path(b))


def validate_checkpoint(value):
    need(value.get('schema')==given.CHECKPOINT_SCHEMA and value.get('formal_run_schema')==FORMAL_RUN_SCHEMA and
         value.get('upstream_commit')==UPSTREAM_COMMIT and value.get('model')==shared.native_spec() and
         value.get('task_contract')==task_contract() and value.get('selection_split')=='validation' and
         value.get('state_only') is True and value.get('action_representation')=='block_relative_joint' and
         value.get('cpu_fixture_only') is False and value.get('step') in EVAL_STEPS,
         'Formal checkpoint architecture/common contract/completed evaluation step required')
    validate_normalization(value.get('normalization',{}));validate_metrics(value.get('metrics',{}))
    inventory=value.get('reference_inventory')
    need(isinstance(inventory,list) and inventory and len({r.get('episode_id') for r in inventory})==len(inventory),
         'Complete unique per-episode reference inventory required')
    for row in inventory:
        need(isinstance(row.get('episode_id'),str) and row['episode_id'] and
             pose.GivenReference.from_snapshot(row.get('reference',{})).sha256==row.get('reference_sha256'),
             'Episode reference inventory digest differs')
    inputs=value.get('inputs',{})
    need(set(inputs)=={'dataset','config','software','normalization','reference_inventory'}, 'Complete checkpoint input provenance required')
    need(set(inputs['dataset'])=={'plan','seal','manifest','qualification','reconstruction'}, 'Sealed dataset input inventory required')
    for ref in [*inputs['dataset'].values(),*(inputs[k] for k in inputs if k!='dataset')]:_reference_shape(ref)
    need(isinstance(value.get('state_dict'),dict), 'Strict model state dictionary required')
    return value


def validate_software(expected,current):
    need(expected==current, 'Official vendor/dependency/processing/Python source inventory changed')


def infer(torch,policy,samples,stats,*,device):
    """Fixed 64+tail observation-only inference; labels never enter the model."""
    validate_normalization(stats);policy.eval();parts=[]
    count=len(samples['states']);need(count>0, 'Complete nonempty inference population required')
    with torch.no_grad():
        for start in range(0,count,BATCH_SIZE):
            observations=pack_observations(samples,stats,slice(start,min(start+BATCH_SIZE,count)))
            batch={key:torch.from_numpy(value).to(device) for key,value in observations.items()}
            value=policy.predict_action_chunk(batch).detach().cpu().numpy()
            need(value.shape==(min(BATCH_SIZE,count-start),20,7), 'Official ACT returned wrong complete chunk shape')
            parts.append(np.asarray(value,dtype=np.float64)*np.asarray(stats['action_std'])+np.asarray(stats['action_mean']))
    return canonical_predictions(np.concatenate(parts))


def validate_complete(value):
    need(value.get('schema')==FORMAL_RUN_SCHEMA and value.get('completed_steps')==20000 and
         value.get('complete') is True and value.get('source_audit_passed') is True and
         value.get('cpu_fixture_only') is False and value.get('test_payloads_opened') is False and
         value.get('task_contract')==task_contract(), 'Complete source-audited formal run required')
    rows=value.get('evaluations');selected=earliest_argmin(rows)
    need(value.get('selected_step')==selected, 'Selected checkpoint is not exact earliest full-VAL argmin')
    for row in rows:
        _reference_shape(row.get('checkpoint'));need(isinstance(row.get('predictions'),dict), 'Every VAL snapshot/prediction required')
    for key in ('best','last'):_reference_shape(value.get(key))
    need(value['best']['sha256']==next(r['checkpoint']['sha256'] for r in rows if r['step']==selected),
         'Best checkpoint bytes differ from selected snapshot')
    need(set(value.get('selected_predictions',{}))=={'train','validation'}, 'Selected full TRAIN and VAL predictions required')
    need(set(value.get('selected_metrics',{}))=={'train','validation'}, 'Selected full TRAIN and VAL metrics required')
    for metric in value['selected_metrics'].values():validate_metrics(metric)
    return value


def load_checkpoint(model,*,device='cpu'):
    """No controller or admission is created by this strict numerical loader."""
    need(set(model)=={'checkpoint','complete'}, 'Exact checkpoint and COMPLETE references required')
    complete=validate_complete(read_bound(model['complete']))
    need(model['checkpoint']==complete['best'], 'Only the unchanged CUDA-selected checkpoint may load')
    software=read_bound(complete['inputs']['software'])
    config=read_bound(complete['inputs']['config']);processing_audit(config['processing'])
    torch,policy,current=make_model(config,device=device)
    validate_software(software,current)
    path=dataset._ref_path(model['checkpoint'],Path.cwd())
    need(reference(path)==model['checkpoint'], 'Checkpoint file changed')
    checkpoint=validate_checkpoint(torch.load(path,map_location='cpu',weights_only=True))
    need(checkpoint['step']==complete['selected_step'] and checkpoint['inputs']==complete['inputs'] and
         read_bound(checkpoint['inputs']['normalization'])==checkpoint['normalization'] and
         read_bound(checkpoint['inputs']['reference_inventory'])==checkpoint['reference_inventory'],
         'Checkpoint normalization/reference/input binding differs')
    policy.load_state_dict(checkpoint['state_dict'],strict=True)
    return torch,policy,checkpoint,complete


class ChunkPredictor:
    """Fresh runtime references share contracts, never a universal TRAIN pose."""
    def __init__(self,torch,policy,checkpoint,metadata,*,device='cpu'):
        validate_checkpoint(checkpoint);validate_runtime_reference(checkpoint,metadata)
        self.torch,self.policy,self.checkpoint,self.metadata,self.device=torch,policy,checkpoint,deepcopy(metadata),device

    def __call__(self,measured_q,estimated_target_base,confidence,*,metadata,history,history_times):
        ref=validate_runtime_reference(self.checkpoint,metadata)
        need(metadata==self.metadata and confidence==1 and
             np.array_equal(np.asarray(estimated_target_base),ref.pose_base),
             'Inference fresh reference/context changed')
        state=shared.contract.encode_state(measured_q,estimated_target_base,confidence,metadata=metadata)
        history=np.asarray(history,dtype=np.float64);times=np.asarray(history_times,dtype=np.float64)
        need(history.shape==(2,17) and times.shape==(2,) and np.isfinite(history).all() and np.isfinite(times).all() and
             times[0]>=0 and abs(times[1]-times[0]-.05)<=1e-8 and np.array_equal(history[1],state),
             'Two real causal inference frames/time/current measured origin required')
        value=infer(self.torch,self.policy,dict(states=state[None],histories=history[None]),
            self.checkpoint['normalization'],device=self.device)[0]
        return shared.contract.JointActionBlock('block_relative_joint',value,state[:7],
            dict(metadata,training_eligible=False,physical_qualification=False,checkpoint_policy=given.CHECKPOINT_SCHEMA))


def fixture_spec():
    return dict(schema=FIXTURE_SCHEMA,synthetic=True,runtime_only=True,training_eligible=False,
        generator='numpy.Generator(PCG64)',seed=FIXTURE_SEED,model_seed=MODEL_SEED,count=65,batches=[64,1],
        draw_order=[['uniform',-.8,.8,[65,7]],['normal',0,.01,[65,7]],
            ['uniform',-.06,.06,[65,3]],['normal',0,.002,[65,3]],
            ['uniform',-.4,.4,[65,3]],['normal',0,.01,[65,3]],['standard_normal',[65,20,7]]],
        rotation='Rz(z) @ Ry(y) @ Rx(x)',state='[q7,p3,Rcol0,Rcol1,1]',
        actions='.02 * noise + .005 * (k+1) * [1,-1,1,-1,1,-1,1]',
        velocity='(q_current-q_previous)/.05',normalization='shared TRAIN mean/std axes (0,1), floor 1e-6',
        model=shared.native_spec(),task_contract=task_contract(),numerics=numerical_protocol())


def _fixture_samples(rng):
    # This function is only called once by the guarded official CPU freezer.
    q0=rng.uniform(-.8,.8,(65,7));q1=q0+rng.normal(0,.01,(65,7))
    p0=rng.uniform(-.06,.06,(65,3));p1=p0+rng.normal(0,.002,(65,3))
    e0=rng.uniform(-.4,.4,(65,3));e1=e0+rng.normal(0,.01,(65,3))
    noise=rng.standard_normal((65,20,7))
    def rotations(e):
        result=[]
        for x,y,z in e:
            cx,sx,cy,sy,cz,sz=np.cos(x),np.sin(x),np.cos(y),np.sin(y),np.cos(z),np.sin(z)
            rx=np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
            ry=np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
            rz=np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
            result.append(rz@ry@rx)
        return np.asarray(result)
    frames=[]
    for q,p,e in ((q0,p0,e0),(q1,p1,e1)):
        r=rotations(e);frames.append(np.c_[q,p,r[:,:,0],r[:,:,1],np.ones(65)])
    return dict(schema=given.DATASET_SCHEMA,task_id=pose.TASK_ID,action_semantics_id=given.ACTION_SEMANTICS_ID,
        observation_contract=pose.observation_contract(),synthetic=True,runtime_only=True,training_eligible=False,
        histories=np.stack(frames,axis=1),states=frames[1],velocities=(q1-q0)/.05,
        actions=.02*noise+.005*np.arange(1,21)[None,:,None]*np.array([1,-1,1,-1,1,-1,1])[None,None,:])


def processing_audit(binding):
    need(isinstance(binding,dict) and set(binding)=={'plan','processing_freeze'}, 'Frozen final processing source binding required')
    plan=read_bound(binding['plan']);path=Path(binding['plan']['path'])
    return dataset._processing_preflight(plan,dict(processing_freeze=binding['processing_freeze']),path.parent)


def source_identity(report):
    # Every call already reopens the closed source tree, archive and history.
    # The measurement timestamp is evidence about the observation, not a file.
    return {key:value for key,value in report.items() if key!='observed_at'}


def _array_binding(value):
    value=np.ascontiguousarray(value,dtype='<f8')
    return dict(shape=list(value.shape),dtype='<f8',order='C',sha256=hashlib.sha256(value.tobytes()).hexdigest())


def _fixture_data(manifest):
    need(manifest.get('spec')==fixture_spec(), 'Frozen numeric fixture recipe changed')
    binding=manifest['samples'];path=dataset._ref_path(binding,Path.cwd())
    need(reference(path)==binding, 'Frozen synthetic sample bytes changed')
    with np.load(path,allow_pickle=False) as archive:
        need(set(archive.files)=={'states','histories','actions','velocities'}, 'Complete fixed fixture arrays required')
        values={key:archive[key] for key in archive.files}
    for key,shape in [('states',(65,17)),('histories',(65,2,17)),('actions',(65,20,7)),('velocities',(65,7))]:
        value=values[key]
        need(value.shape==shape and value.dtype.str=='<f8' and np.isfinite(value).all() and
             _array_binding(value)==manifest['arrays'][key], 'Fixed fixture shape/dtype/full array bytes changed')
    values.update(schema=given.DATASET_SCHEMA,task_id=pose.TASK_ID,observation_contract=pose.observation_contract(),
        action_semantics_id=given.ACTION_SEMANTICS_ID,synthetic=True,training_eligible=False)
    need(np.array_equal(values['states'],values['histories'][:,1]) and
         np.array_equal(values['velocities'],(values['histories'][:,1,:7]-values['histories'][:,0,:7])/.05),
         'Frozen fixture history/current/CV relation differs')
    stats=validate_normalization(read_bound(manifest['normalization']))
    need(stats==given.fit_normalization(values,split='train'), 'Frozen fixture normalization does not reconstruct')
    return values,stats


def freeze_fixture(config_ref,output):
    config=read_bound(config_ref)
    need(config.get('lerobot_root')==VENDOR_ROOT and Path(sys.prefix).resolve()==(Path(VENDOR_ROOT)/'.venv').resolve(),
         'Fixture generation requires the pinned official CPU environment')
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'Fixture initialization must be CPU only')
    need(config.get('fixture_spec')==fixture_spec(), 'Predeclared exact runtime fixture specification required')
    source=processing_audit(config.get('processing'))
    output=Path(output).resolve();need(not output.exists(), 'New immutable numeric fixture directory required')
    torch,policy,software=make_model(config,device='cpu')
    samples=_fixture_samples(np.random.Generator(np.random.PCG64(FIXTURE_SEED)))
    stats=given.fit_normalization(samples,split='train');output.mkdir(parents=True)
    keys=('states','histories','actions','velocities')
    with (output/'samples.npz').open('xb') as stream:
        np.savez(stream,**{key:np.ascontiguousarray(samples[key],dtype='<f8') for key in keys})
    normalization=write_new(output/'normalization.json',stats)
    software_ref=write_new(output/'software.json',software)
    with (output/'initial_state.pt').open('xb') as stream:
        torch.save(dict(schema=FIXTURE_SCHEMA,cpu_fixture_only=True,model=shared.native_spec(),
            task_contract=task_contract(),model_seed=MODEL_SEED,state_dict=policy.state_dict()),stream)
    need(source_identity(source)==source_identity(processing_audit(config['processing'])), 'Processing source freeze changed during fixture initialization')
    need(read_bound(config_ref)==config, 'Fixture config changed during initialization')
    validate_software(software,software_inventory(VENDOR_ROOT))
    result=dict(schema=FIXTURE_SCHEMA,spec=fixture_spec(),config=config_ref,software=software_ref,
        processing=config['processing'],processing_audit=source,samples=reference(output/'samples.npz'),
        normalization=normalization,initial_state=reference(output/'initial_state.pt'),
        arrays={key:_array_binding(samples[key]) for key in keys},
        window_order_sha256=digest(list(range(65))),runtime_only=True,synthetic=True,training_eligible=False)
    write_new(output/'manifest.json',result)
    return result


def _identity():
    from . import process_guard as guard
    identity=guard.process_identity(os.getpid())
    need(identity is not None, 'Fresh Linux process identity required')
    return identity


def _run_fixture(fixture_ref,output,device,*,owned_stage=None,ownership=None):
    manifest=read_bound(fixture_ref);samples,stats=_fixture_data(manifest)
    source=processing_audit(manifest['processing']);config=read_bound(manifest['config'])
    torch,policy,software=make_model(config,device=device)
    validate_software(read_bound(manifest['software']),software)
    path=dataset._ref_path(manifest['initial_state'],Path.cwd())
    need(reference(path)==manifest['initial_state'], 'Frozen initial model bytes changed')
    packet=torch.load(path,map_location='cpu',weights_only=True)
    need(packet.get('schema')==FIXTURE_SCHEMA and packet.get('cpu_fixture_only') is True and
         packet.get('model')==shared.native_spec() and packet.get('task_contract')==task_contract() and
         packet.get('model_seed')==MODEL_SEED, 'Only the frozen CPU-initialized fixture state may load')
    policy.load_state_dict(packet['state_dict'],strict=True)
    output=Path(output).resolve();need(not output.exists(), 'Fresh fixture execution directory required');output.mkdir(parents=True)
    prediction=infer(torch,policy,samples,stats,device=device)
    result=dict(schema=FIXTURE_RESULT_SCHEMA,fixture=fixture_ref,device=device,identity=_identity(),
        execution=execution_settings(torch,policy,device=device),
        software=manifest['software'],initial_state=manifest['initial_state'],batches=[64,1],
        predictions=save_predictions(output/'predictions.npy',prediction,window_order_sha256=manifest['window_order_sha256']),
        metrics=metrics(prediction,samples),runtime_only=True,synthetic=True,training_eligible=False)
    if device=='cuda':
        need(owned_stage is not None and ownership is not None, 'CUDA fixture requires original-owner launch and lock evidence')
        result.update(owned_stage=owned_stage,ownership=ownership)
    need(source_identity(source)==source_identity(processing_audit(manifest['processing'])), 'Fixture processing sources changed')
    validate_software(software,software_inventory(VENDOR_ROOT));_fixture_data(read_bound(fixture_ref))
    result_ref=write_new(output/'result.json',result)
    if device=='cuda':write_new(output/'COMPLETE.json',dict(schema='m2_given_contact_owned_fixture_complete_v1',
        result=result_ref,identity=result['identity'],owned_stage=owned_stage,ownership=ownership))
    return result


def run_fixture(fixture_ref,output,*,device):
    need(device=='cpu', 'CUDA fixture must use the original run owner and explicit finite stage config')
    return _run_fixture(fixture_ref,output,device)


def _compare_fixture(fixture_ref,result_refs,cuda_owner_terminal):
    manifest=read_bound(fixture_ref);samples,_=_fixture_data(manifest)
    bound_path(manifest['initial_state']);validate_software(read_bound(manifest['software']),software_inventory(VENDOR_ROOT))
    need(len(result_refs)==3, 'Two fresh CPU results followed by one owned CUDA result required')
    results=[read_bound(ref) for ref in result_refs];arrays=[]
    for result,device in zip(results,('cpu','cpu','cuda')):
        need(result.get('schema')==FIXTURE_RESULT_SCHEMA and result.get('device')==device and
             same_reference(result.get('fixture'),fixture_ref) and result.get('initial_state')==manifest['initial_state'] and
             result.get('software')==manifest['software'] and result.get('batches')==[64,1] and
             result.get('runtime_only') is True and result.get('training_eligible') is False,
             'Complete exact fixture/device/software result required')
        validate_execution_settings(result.get('execution',{}),device=device)
        value=load_predictions(result['predictions'],window_order_sha256=manifest['window_order_sha256'])
        need(len(value)==65 and metrics(value,samples)==result.get('metrics'), 'Fixture full metrics do not reconstruct')
        arrays.append(value)
    need(results[0]['identity']!=results[1]['identity'] and
         results[0]['identity']['pid']!=results[1]['identity']['pid'], 'CPU fixture results must come from two fresh processes')
    need(arrays[0].tobytes()==arrays[1].tobytes() and results[0]['metrics']==results[1]['metrics'],
         'Fresh CPU fixture arrays/metrics are not byte-identical')
    comparison=compare_devices(arrays[0],arrays[2],results[0]['metrics'],results[2]['metrics'])
    cuda=results[2];complete_ref=reference(bound_path(result_refs[2]).parent/'COMPLETE.json')
    owner=verify_owned_stage(cuda_owner_terminal,complete_ref=complete_ref,child_identity=cuda['identity'],
        experiment='m2_given_contact_numeric_fixture',child_result_ref=result_refs[2])
    need(same_reference(cuda.get('owned_stage',{}).get('input'),fixture_ref), 'CUDA owner used a different frozen numeric fixture')
    processing_audit(manifest['processing'])
    return dict(schema='m2_given_contact_numeric_runtime_qualification_v1',fixture=fixture_ref,results=result_refs,
        passed=True,cpu_byte_identical=True,comparison=comparison,spec=fixture_spec(),
        processing=manifest['processing'],software=manifest['software'],cuda_owner_terminal=cuda_owner_terminal,
        cuda_owner_audit=owner,runtime_only=True,synthetic=True,training_eligible=False)


def qualify_fixture(fixture_ref,result_refs,output,*,cuda_owner_terminal):
    try:result=_compare_fixture(fixture_ref,result_refs,cuda_owner_terminal)
    except Exception as error:
        write_new(output,dict(schema='m2_given_contact_numeric_runtime_qualification_v1',fixture=fixture_ref,
            results=result_refs,cuda_owner_terminal=cuda_owner_terminal,passed=False,error=type(error).__name__+': '+str(error),runtime_only=True,
            synthetic=True,training_eligible=False));raise
    write_new(output,result);return result


def verify_fixture(qualification_ref):
    value=read_bound(qualification_ref)
    need(value.get('passed') is True, 'A failed or missing numerical runtime qualification blocks training')
    need(value==_compare_fixture(value['fixture'],value['results'],value.get('cuda_owner_terminal')), 'Fresh numeric fixture qualification differs')
    validate_software(read_bound(value['software']),software_inventory(VENDOR_ROOT))
    return value


def _stage_config(config_ref,experiment):
    value=read_bound(config_ref)
    need(set(value)=={'schema','experiment','base','time_limit_seconds','input'} and
         value.get('schema')==OWNED_STAGE_SCHEMA and value.get('experiment')==experiment and experiment in OWNED_MODULES and
         Path(str(value.get('base',''))).is_absolute() and
         value.get('time_limit_seconds')==(21600 if experiment=='m2_given_contact_train' else 3600),
         'Exact finite original-owner stage config required')
    _reference_shape(value.get('input'));return value


def _same_process(a,b):
    return isinstance(a,dict) and isinstance(b,dict) and all(a.get(key)==b.get(key) and a.get(key) is not None
        for key in ('pid','pgid','sid','start_ticks')) and a['pid']==a['pgid']==a['sid']


def _process_command(pid):
    return (Path('/proc')/str(pid)/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')


def start_owned_stage(config_ref,output,*,experiment):
    """Authenticate this actual child and original run parent before GPU use."""
    from . import process_guard as guard
    need(sys.platform=='linux' and os.environ.get('AUSC_OWNED_RUN')=='1', 'Original Linux run owner required')
    config=_stage_config(config_ref,experiment);output=dataset._path(str(output),Path.cwd());folder=output.parent
    run_ref=reference(folder/'run.json');run=read_bound(run_ref);identity=_identity()
    parent_pid=os.getppid();parent=guard.process_identity(parent_pid)
    command=_process_command(parent_pid)
    need('-m' in command, 'Actual original run parent command required')
    marker=command.index('-m')
    need(Path(command[0]).resolve()==Path(sys.executable).resolve() and all(arg in ('-u','-B') for arg in command[1:marker]) and
         command[marker:]==['-m','sim.ausc_policy.run','--config',str(bound_path(config_ref)),'--run-id',folder.name],
         'Actual parent is not the declared original run invocation')
    need(_same_process(identity,identity) and run.get('run_id')==folder.name and
         folder==Path(config['base'])/'runs/ausc_policy'/folder.name and output.name=='output' and
         run.get('config')==config and run.get('config_sha256')==config_ref['sha256'] and run.get('completed') is False,
         'Actual child/config/output does not match original prelaunch run record')
    module=OWNED_MODULES[experiment]
    expected=[sys.executable,'-u','-B','-m',module,'--config',str(bound_path(config_ref)),'--output',str(output)]
    source=reference(guard.__file__)
    need(source['sha256']==GUARD_SHA256 and run.get('command')==expected and run.get('resource_guard')==
         dict(path=source['path'],sha256=source['sha256'],gpu_exit_grace_s=guard.EXIT_GPU_GRACE_S),
         'Original unchanged guard/child command required')
    return dict(run_id=folder.name,run=run_ref,config=reference(bound_path(config_ref)),input=config['input'],output=str(output),
        experiment=experiment,module=module,child_identity=identity,parent_identity=parent,parent_command=command,guard=source)


def verify_owned_stage(terminal_ref,*,complete_ref,child_identity,experiment,child_result_ref=None):
    """Reopen original run/process/COMPLETE after exit; no child self-qualification."""
    from . import process_guard as guard
    terminal=read_bound(terminal_ref);terminal_path=bound_path(terminal_ref)
    complete=read_bound(complete_ref);start=complete.get('owned_stage',{})
    need(start.get('experiment')==experiment and start.get('module')==OWNED_MODULES.get(experiment),
         'Declared original owned stage required')
    config=_stage_config(start.get('config'),experiment);run=read_bound(start.get('run'))
    folder=Path(config['base'])/'runs/ausc_policy'/start.get('run_id','')
    need(terminal_path==folder/'result.json' and bound_path(start['run'])==folder/'run.json' and
         bound_path(complete_ref)==folder/'output/COMPLETE.json' and start.get('output')==str(folder/'output') and
         config['input']==start.get('input') and run.get('config')==config and run.get('config_sha256')==start['config']['sha256'],
         'Original owner/config/input/output immutable path binding differs')
    common=('run_id','code_commit','config','config_sha256','command','resource_guard','started_utc')
    need(all(terminal.get(key)==run.get(key) for key in common) and terminal.get('completed') is True and
         terminal.get('gpu_released') is True, 'Original owner did not complete and release GPU normally')
    source=reference(guard.__file__)
    need(source['sha256']==GUARD_SHA256 and start.get('guard')==source and terminal.get('resource_guard')==
         dict(path=source['path'],sha256=source['sha256'],gpu_exit_grace_s=guard.EXIT_GPU_GRACE_S), 'Original guard source binding changed')
    expected=[sys.executable,'-u','-B','-m',OWNED_MODULES[experiment],'--config',start['config']['path'],'--output',str(folder/'output')]
    need(terminal.get('command')==expected, 'Original official child command differs')
    stage=terminal.get('stage',{});process_ref=reference(folder/'lifecycle'/(experiment+'.process.json'))
    need(read_bound(process_ref)==stage and stage.get('passed') is True and stage.get('exit_code')==0 and
         stage.get('process_group_drained') is True and stage.get('errors')==[] and stage.get('signals')==[] and
         stage.get('deferred_signals')==[] and _same_process(stage.get('process_identity'),child_identity) and
         _same_process(start.get('child_identity'),child_identity), 'Original child exit/identity/drain evidence failed')
    if experiment=='m2_given_contact_train':
        need(complete.get('schema')==FORMAL_RUN_SCHEMA and same_reference(config['input'],complete['inputs']['config']),
             'Original owner used a different formal training config')
        ownership=read_bound(complete['resources'])['owner']
    else:
        need(complete.get('schema')=='m2_given_contact_owned_fixture_complete_v1' and child_result_ref is not None and
             complete.get('result')==reference(bound_path(child_result_ref)),
             'Original owner COMPLETE does not bind exact CUDA fixture result')
        child=read_bound(child_result_ref)
        need(child.get('owned_stage')==start and _same_process(child.get('identity'),child_identity) and
             child.get('ownership')==complete.get('ownership'), 'CUDA child/result/COMPLETE provenance differs')
        ownership=complete.get('ownership',{})
    need(ownership.get('path')==GPU_LOCK and (ownership.get('device'),ownership.get('inode'))==GPU_LOCK_ID and
         ownership.get('pid')==child_identity['pid'] and ownership.get('checks')==dict(gpu_pids_before=[]),
         'Original single GPU lock and pre-CUDA idle checks required')
    return dict(passed=True,terminal=terminal_ref,run=start['run'],process=process_ref,complete=complete_ref,
        child_identity={key:child_identity[key] for key in ('pid','pgid','sid','start_ticks')},ownership=ownership)


def run_owned_fixture(stage_ref,output):
    start=start_owned_stage(stage_ref,output,experiment='m2_given_contact_numeric_fixture')
    with owned_gpu() as ownership:
        return _run_fixture(start['input'],output,'cuda',owned_stage=start,ownership=ownership)


def main():
    if len(sys.argv)>1 and sys.argv[1]=='--config':
        parser=argparse.ArgumentParser(description='Original-owner finite CUDA fixture stage')
        parser.add_argument('--config',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
        result=run_owned_fixture(reference(args.config),args.output)
        print(json.dumps(dict(output=str(Path(args.output).resolve()),runtime_only=True,training_eligible=False)));return
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='operation',required=True)
    freeze=sub.add_parser('freeze-fixture');freeze.add_argument('--config',required=True);freeze.add_argument('--config-sha256',required=True)
    run=sub.add_parser('run-fixture');run.add_argument('--fixture',required=True);run.add_argument('--fixture-sha256',required=True)
    run.add_argument('--device',required=True,choices=['cpu','cuda'])
    qualify=sub.add_parser('qualify-fixture');qualify.add_argument('--fixture',required=True);qualify.add_argument('--fixture-sha256',required=True)
    qualify.add_argument('--results',required=True,nargs=3);qualify.add_argument('--result-sha256',required=True,nargs=3)
    qualify.add_argument('--cuda-owner-terminal',required=True);qualify.add_argument('--cuda-owner-sha256',required=True)
    for child in (freeze,run,qualify):child.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.operation=='freeze-fixture':result=freeze_fixture(dict(path=str(Path(args.config).resolve()),sha256=args.config_sha256),args.output)
    else:
        fixture=dict(path=str(Path(args.fixture).resolve()),sha256=args.fixture_sha256)
        if args.operation=='run-fixture':result=run_fixture(fixture,args.output,device=args.device)
        else:result=qualify_fixture(fixture,[dict(path=str(Path(path).resolve()),sha256=sha) for path,sha in
            zip(args.results,args.result_sha256)],args.output,cuda_owner_terminal=dict(
                path=str(Path(args.cuda_owner_terminal).resolve()),sha256=args.cuda_owner_sha256))
    print(json.dumps(dict(operation=args.operation,output=str(Path(args.output).resolve()),training_eligible=False,
        passed=result.get('passed')),allow_nan=False))


if __name__=='__main__':main()
