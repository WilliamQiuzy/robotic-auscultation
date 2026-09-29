"""Inactive conditional C trainer software; actual entry requires completed B.

The official model, loss, optimizer and sampling loop are unchanged. C is freshly
initialized, never restored from B weights. Every real run and verify-only path
reconstructs the full condition, loads saved C normalization, and matches B's
exact schedule and numerical initialization. Importing this module does none of
that work and does not authorize a training run.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import uuid
import numpy as np
from sim.auscultation_pi05_v7 import coverage_pilot_schedule as schedule_api
from sim.auscultation_pi05_v8 import train as previous
from . import action_encoding as codec,training_data as data_api,trigger

VERSION='wam_rational_q4_training_v1'
RECIPE='pi05_v9_rational_q4_fresh_base_v1'
INITIALIZATION_SCHEMA='wam_rational_q4_fresh_initialization_v1'
SPEC_SHA256='cfcf4573ba222656735cb5e99bc521e684d02dc0513a143b2e37bd83e2eecff1'
NORMALIZATION_FILE_SHA256='4801ae589fda5ca98e6236e8307234dc05a4c70574682b52063217f7273c1f34'
PARENT_TRAINER_SHA256='89dbf8238e4041966cb221865f2269624dd870b70393568526c51ef0056f1d39'
PAIR_SOURCE_SHA256='95fa3bf8444d2b90df57c8afcb647fa93ed80b1c3d9b2aa378b43a57bb2999c9'
C_INPUT_SOURCES={'action_encoding.py':'2f50e76cc06602e9eebba856002ab9178cce79d3fbe025325e4a907f72b35526',
 'training_data.py':'1134b0276219ab9ff6057a2d28f00d2eb509524a700b064c7a4fb9476c1337c9',
 'trigger.py':'ce80edb60460e52a0f3271e7353c4007f88226b49fd6ba704e339940eea9eb08'}
ORIGINAL_B_SOURCES=dict(previous.data_api.EXPECTED_PARENT_SOURCES,
 **{'v7/'+n:h for n,h in previous.ANCESTRY_SOURCES.items()},
 **{'v8/action_encoding.py':'a694add3d8becbc99fdc18b9c4ad9d10823339f2a674fcc62130c9147331ae23',
    'v8/training_data.py':'86dc1e157012fec6983a68ee90940d5718b716ee6cd3a83f8faef447cc761172',
    'v8/train.py':PARENT_TRAINER_SHA256,'v8/evaluation.py':'1babc07f8f53c833d39d761efb077e74ec1e9a8f4a305669b1bc7c05b848c5b9'})
ROOT=Path('/home/william/wam-vla-v2')
SEED=61
LEARNING_RATE=5e-5
TOTAL_STEPS=14600
EXPECTED_STEPS=(0,1217,2434,3650,4867,6084,7300,8517,9734,10950,12167,13384,14600)
LOCK_NAMES=('RATIONAL_SELECTION_LOCKED.json','rational_selection',*previous.LOCK_NAMES)
need=previous.need
utc=previous.utc
read=previous.read
parent_module=previous.parent_module
freeze=previous.freeze
separate=previous.separate
records_for=previous.records_for
fingerprint_leaves=previous.fingerprint_leaves
_validate_numerical=previous._validate_numerical
verify_optimizer_age=previous.verify_optimizer_age
verify_step_zero_arrays=previous.verify_step_zero_arrays
_numeric_initialization=previous._numeric_initialization
validate_limit=previous.validate_limit


def source_hashes():
    """Original B24 retained under its own keys; five new C sources are additional."""
    need(schedule_api.file_sha(previous.__file__)==PARENT_TRAINER_SHA256,'Original B trainer changed')
    old=previous.source_hashes();need(old==ORIGINAL_B_SOURCES,'Original B24 source closure changed')
    own=Path(__file__).parent;pair=own.with_name('auscultation_pi05_v8')/'pair_admission.py'
    need(schedule_api.file_sha(pair)==PAIR_SOURCE_SHA256,'Original pair verifier changed')
    result=dict(old,**{'v8/pair_admission.py':PAIR_SOURCE_SHA256})
    for name in ('action_encoding.py','training_data.py','trigger.py','train.py','evaluation.py'):
        path=own/name;need(path.is_file() and not path.is_symlink(),'Complete new C source set required')
        actual=schedule_api.file_sha(path)
        need(name not in C_INPUT_SOURCES or actual==C_INPUT_SOURCES[name],'Reviewed C input source changed')
        result['v9/'+name]=actual
    need(len(result)==30,'Exact C30 source closure required')
    return result


def _pinned_bytes(binding):
    need(isinstance(binding,dict) and set(binding)=={'path','sha256'},'Exact input path/hash binding required')
    path=trigger.canonical(binding['path']);raw=path.read_bytes()
    need(isinstance(binding['sha256'],str) and hashlib.sha256(raw).hexdigest()==binding['sha256'],'Bound input changed before parse')
    return raw


def _pinned(binding):return json.loads(_pinned_bytes(binding))


def _original_a_initialization(binding):
    initial=previous._verify_initialization(_pinned(binding))
    path=Path(binding['path'])
    need(path.name=='initialization.json' and initial['mode']==previous.codec.A
         and initial['training_run_path']==str(path.parent)
         and initial['source_sha256']==ORIGINAL_B_SOURCES,'Bound original A initialization required')
    return initial


def reference_from_condition(condition_binding,*,reconstruct):
    """Full scientific reconstruction only at actual entry/context boundaries.

    Checkpoint-local verification pins the already frozen condition and consumed
    B bytes. It is not a replacement for the full run-context reconstruction.
    """
    condition=trigger.verify_receipt(condition_binding['path'],condition_binding['sha256']) if reconstruct else _pinned(condition_binding)
    need(condition.get('schema')==trigger.SCHEMA and condition.get('condition_satisfied') is True,'Completed B condition must be freshly satisfied')
    need(_pinned(condition_binding)==condition,'Condition bytes changed during reconstruction')
    spec=condition['specification_binding'];need(spec['sha256']==SPEC_SHA256,'Approved conditional design differs');_pinned(spec)
    artifacts=condition['artifact_sha256'];run=trigger.B_RUN.resolve()
    def bound(name):
        path=run/name;need(str(path) in artifacts,'B dependency missing from condition closure')
        return dict(path=str(path),sha256=artifacts[str(path)])
    cb=bound('configuration.json');ib=bound('initialization.json');sb=bound('training_schedule.json')
    saved=_pinned(cb);config=saved['config'];initial=previous._verify_initialization(_pinned(ib))
    need(config['mode']==previous.codec.B and config['training_schema']==previous.VERSION and config['recipe']==previous.RECIPE
         and config['training_run_path']==str(run) and config['source_sha256']==ORIGINAL_B_SOURCES,'Original B configuration identity differs')
    need(initial['mode']==previous.codec.B and initial['training_run_path']==str(run)
         and initial['source_sha256']==ORIGINAL_B_SOURCES and initial['assets']==config['assets']
         and initial['config_sha256']==schedule_api.digest(config),'Original B initialization/config binding differs')
    ab=config['paired_initialization']
    need(isinstance(ab,dict) and set(ab)=={'path','sha256'} and artifacts.get(ab['path'])==ab['sha256'],
         'Original A initialization missing from verified condition closure')
    ai=_original_a_initialization(ab)
    need(initial['paired_a_receipt_integrity']==ai['integrity_sha256'] and ai['assets']==initial['assets']
         and ai['numerical_initialization']==initial['numerical_initialization'],'Original B/A initialization provenance differs')
    need(config['schedule_binding']['path']==sb['path'] and config['schedule_binding']['sha256']==sb['sha256'],'Original B schedule path/hash differs')
    schedule=_pinned(sb)
    need(schedule_api.load_schedule(sb['path'],config['schedule_binding'])==schedule,'Original B schedule metadata differs')
    need(schedule_api.json_bytes(schedule)==_pinned_bytes(sb),'Original B schedule bytes are not canonical')
    need(schedule['total_steps']==14600 and schedule['total_draws']==58400 and schedule['valid_window_count']==12759
         and tuple(schedule['candidate_steps'])==EXPECTED_STEPS,'Original B complete budget differs')
    for binding in (condition_binding,spec,cb,ib,sb,ab):_pinned_bytes(binding)
    return dict(run_path=str(run),configuration_binding=cb,initialization_binding=ib,schedule_binding=config['schedule_binding'],
        data_hash=saved['data_hash'],source_sha256=config['source_sha256'],assets=config['assets'],pair_spec=config['pair_spec'],
        paired_A_initialization_binding=ab)


def _protect_original_run_outputs(output,condition_binding,reference):
    """Use freshly pinned consumed provenance before any C output mutation."""
    condition=_pinned(condition_binding);ab=reference.get('paired_A_initialization_binding')
    need(isinstance(ab,dict) and set(ab)=={'path','sha256'}
         and condition['artifact_sha256'].get(ab['path'])==ab['sha256'],
         'Bound original A tree required before output admission')
    ai=_original_a_initialization(ab)
    need(reference['run_path']==str(trigger.B_RUN.resolve()),'Original B tree differs')
    separate(output,[Path(ai['training_run_path']),Path(reference['run_path'])])


def load_inputs(manifest,mode,*,condition_binding,saved_normalization_binding):
    codec.action_encoding_binding(mode)
    reference=reference_from_condition(condition_binding,reconstruct=True)  # must precede loader/assets/model
    source_hashes()
    need(saved_normalization_binding['sha256']==NORMALIZATION_FILE_SHA256,'Canonical prepared C normalization required')
    norm=_pinned(saved_normalization_binding)
    dataset=data_api.load_dataset(manifest,mode,normalization=norm,expected_normalization_sha256=data_api._digest(norm))
    need(dataset.parent.data_hash==reference['data_hash'],'C parent is not the original qualified B data')
    records,ids=records_for(dataset)
    sb=reference['schedule_binding'];schedule=_pinned({k:sb[k] for k in ('path','sha256')})
    schedule_api.verify_schedule(schedule,records,ids)
    condition=_pinned(condition_binding);spec=condition['specification_binding'];_pinned(spec)
    _,_,assets=parent_module()._verify_public_assets(ROOT)
    need(assets==reference['assets'],'Original official B assets differ')
    return dataset,schedule,assets,spec,reference


def save_reference_schedule(path,schedule,reference):
    binding=reference['schedule_binding'];raw=_pinned_bytes({k:binding[k] for k in ('path','sha256')})
    need(raw==schedule_api.json_bytes(schedule),'C schedule differs from original B bytes')
    path=Path(path)
    with path.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    need(schedule_api.file_sha(path)==binding['sha256'],'Copied schedule hash differs')
    return dict(binding,path=str(path.resolve()))


def make_initialization(config,numerical,paired_b):
    _validate_numerical(numerical);codec.action_encoding_binding(config['mode'])
    need(config['seed']==61 and config['learning_rate']==5e-5 and config['batch_size']==4,'Original initialization settings required')
    need(paired_b is not None,'C requires original B initialization');previous._verify_initialization(paired_b)
    need(paired_b['mode']==previous.codec.B and paired_b['training_run_path']==config['b_reference']['run_path'],'Original B initialization required')
    need(paired_b['source_sha256']==config['b_reference']['source_sha256']==ORIGINAL_B_SOURCES,'Original B source provenance differs')
    need({k:config['source_sha256'][k] for k in ORIGINAL_B_SOURCES}==ORIGINAL_B_SOURCES,'Unchanged original B source subset differs')
    need(paired_b['assets']==config['assets'] and paired_b['pair_spec']==config['b_reference']['pair_spec'],'Original B assets/spec differ')
    need(paired_b['numerical_initialization']==numerical,'Fresh C and original B numerical initialization differ')
    result=dict(schema=INITIALIZATION_SCHEMA,mode=codec.C,training_run_path=config['training_run_path'],seed=61,optimizer_step=0,
        source_sha256=config['source_sha256'],assets=config['assets'],conditional_spec=config['conditional_spec'],
        condition_binding=config['condition_binding'],saved_normalization_binding=config['saved_normalization_binding'],
        b_reference=config['b_reference'],config_sha256=schedule_api.digest(config),numerical_initialization=numerical,
        paired_B_numerics_equal=True,paired_B_receipt_integrity=paired_b['integrity_sha256'])
    result['integrity_sha256']=schedule_api.digest(result)
    return result


def _verify_initialization(receipt):
    need(isinstance(receipt,dict) and receipt.get('schema')==INITIALIZATION_SCHEMA,'Fresh C initialization receipt required')
    codec.action_encoding_binding(receipt.get('mode'));_validate_numerical(receipt['numerical_initialization'])
    need(receipt.get('seed')==61 and receipt.get('optimizer_step')==0 and receipt.get('paired_B_numerics_equal') is True,'Fresh matched C age/seed required')
    need(receipt.get('integrity_sha256')==schedule_api.digest({k:v for k,v in receipt.items() if k!='integrity_sha256'}),'C initialization integrity differs')
    return receipt


def make_config(dataset,schedule,schedule_binding,assets,spec,output,condition_binding,saved_normalization_binding,reference):
    parent=parent_module()
    return dict(training_schema=VERSION,recipe=RECIPE,mode=dataset.mode,training_run_path=str(Path(output).resolve()),
        source_sha256=source_hashes(),trainer_sha256=schedule_api.file_sha(__file__),dataset_binding=data_api.dataset_binding(dataset),
        parent_data_hash=dataset.parent.data_hash,openpi_commit=parent.OPENPI_COMMIT,assets=assets,conditional_spec=spec,
        condition_binding=condition_binding,saved_normalization_binding=saved_normalization_binding,b_reference=reference,
        seed=61,learning_rate=5e-5,batch_size=4,total_steps=14600,candidate_steps=schedule['candidate_steps'],schedule_binding=schedule_binding,
        evaluate_every='predeclared_candidate_steps',save_every='predeclared_candidates_plus_interruption_recovery',
        action_dim=7,action_horizon=10,state_width=41,max_token_len=256,
        model_architecture='unmodified official pi05 with V6 seven-channel LoRA/projection partition',
        initialization='independent fresh official base; B-identical numeric initialization, Adam0 and JAX key61; no trained checkpoint bootstrap',
        action_semantics='absolute_joint_command_v1',model_action_semantics='signed_rational_measured_anchor_q4_plus_six_original_B_residuals',
        canonical_phase_instructions=dict(parent.INSTRUCTIONS),instruction_policy='canonical-phase-and-target-v1',
        observation_contract=dataset.manifest['observation_contract'],action_encoding_binding=dataset.normalization['action_encoding_binding'],
        dynamics_contract_sha256=dataset.dynamics['contract_sha256'],dynamics_recipe_sha256=dataset.dynamics['recipe_sha256'],
        execution_contract_sha256=dataset.dynamics['execution_contract_sha256'])


def verify_configuration(dataset,config,schedule,assets,spec,reference):
    run=config.get('training_run_path');need(isinstance(run,str) and str(Path(run).resolve())==run,'Canonical original C run required')
    need(reference_from_condition(config['condition_binding'],reconstruct=False)==reference==config['b_reference'],'Frozen B provenance differs')
    need(_pinned(config['saved_normalization_binding'])==dataset.normalization
         and config['saved_normalization_binding']['sha256']==NORMALIZATION_FILE_SHA256,'Saved C normalization changed')
    expected=make_config(dataset,schedule,config['schedule_binding'],assets,spec,run,config['condition_binding'],config['saved_normalization_binding'],reference)
    need(config==expected,'C configuration/source/mode/condition differs')
    need(config['schedule_binding']['path']==str(Path(run)/'training_schedule.json'),'Canonical C schedule required')
    need(schedule_api.load_schedule(config['schedule_binding']['path'],config['schedule_binding'])==schedule,'Saved C schedule differs')
    need(_pinned_bytes({k:config['schedule_binding'][k] for k in ('path','sha256')})==_pinned_bytes({k:reference['schedule_binding'][k] for k in ('path','sha256')}),'C schedule bytes differ from B')
    return expected


def _load_initialization(config):
    path=Path(config['training_run_path'])/'initialization.json';need(path.is_file() and not path.is_symlink(),'C initialization receipt required')
    receipt=_verify_initialization(read(path))
    reference=reference_from_condition(config['condition_binding'],reconstruct=False)
    need(reference==config['b_reference'],'B reference changed after C initialization')
    b=_pinned(reference['initialization_binding'])
    need(make_initialization(config,receipt['numerical_initialization'],b)==receipt,'Matched C initialization proof differs')
    return receipt,schedule_api.file_sha(path)

def _verified_metadata(path,*,data_hash,config,normalization):
    path=Path(path)
    need(not path.is_symlink() and all((path/n).is_file() and not (path/n).is_symlink() for n in ('metadata.json','arrays.npz','COMPLETE.json')),'Regular complete checkpoint required')
    m=parent_module()._verified_checkpoint_metadata(path,data_hash=data_hash,config=config,normalization=normalization)
    need(config.get('training_schema')==VERSION and config.get('recipe')==RECIPE and config.get('source_sha256')==source_hashes(),'New truthful C training sources required')
    need(config.get('mode')==normalization.get('mode') and config['dataset_binding']['mode']==config['mode']
         and config['dataset_binding']['data_hash']==data_hash,'Foreign C mode/data checkpoint')
    stats_sha=codec.validate_stats(normalization['residual_actions'],mode=config['mode'])
    need(config['dataset_binding'].get('normalization_sha256')==data_api._digest(normalization)
         and config['dataset_binding'].get('residual_stats_sha256')==stats_sha,'Checkpoint normalization differs from qualified C binding')
    need(_pinned(config['saved_normalization_binding'])==normalization and config['saved_normalization_binding']['sha256']==NORMALIZATION_FILE_SHA256,'Checkpoint differs from pinned prepared C normalization')
    run=Path(config['training_run_path']);need(path.resolve().parent.parent==run and str(run.resolve())==str(run),'Foreign/copied checkpoint run')
    need(re.fullmatch(r'step-[0-9]{12}',path.name) is not None and m['step']==int(path.name[5:]),'Checkpoint directory clock differs')
    need(m.get('training_schema')==VERSION and type(m.get('optimizer_step')) is int and m['optimizer_step']==m['step']
         and m.get('schedule_cursor')==4*m['step'] and m.get('sampler_rng_state')==dict(cursor=4*m['step'],schedule_sha256=config['schedule_binding']['sha256']),
         'Fresh checkpoint clock/cursor differs')
    initial,ih=_load_initialization(config);need(m.get('initialization_sha256')==ih,'Checkpoint initialization lineage differs')
    with np.load(path/'arrays.npz',allow_pickle=False) as arrays:
        verify_optimizer_age(arrays,m['leaf_contract'],m['step'])
        if m['step']==0:verify_step_zero_arrays(arrays,m['leaf_contract'],initial['numerical_initialization'])
    return m


def checkpoint_binding(path):
    path=Path(path).resolve(strict=True)
    metadata=dict(path=str(path/'metadata.json'),sha256=schedule_api.file_sha(path/'metadata.json'))
    m=_pinned(metadata);c=m['config'];run=Path(c['training_run_path'])
    configuration=dict(path=str(run/'configuration.json'),sha256=schedule_api.file_sha(run/'configuration.json'))
    complete=dict(path=str(path/'COMPLETE.json'),sha256=schedule_api.file_sha(path/'COMPLETE.json'))
    saved=_pinned(configuration)
    need(saved['config']==c and saved['data_hash']==m['data_hash'] and saved['normalization']==m['normalization'],'Original C run configuration differs')
    verified=_verified_metadata(path,data_hash=saved['data_hash'],config=c,normalization=saved['normalization'])
    need(verified==m,'Checkpoint metadata changed during verification')
    for binding in (metadata,configuration,complete):_pinned_bytes(binding)
    need(schedule_api.file_sha(path/'arrays.npz')==m['arrays_sha256'],'Checkpoint arrays changed during verification')
    return dict(schema='wam_rational_q4_checkpoint_binding_v1',mode=c['mode'],checkpoint_path=str(path),local_step=m['step'],optimizer_step=m['optimizer_step'],
        metadata_sha256=metadata['sha256'],arrays_sha256=m['arrays_sha256'],complete_sha256=complete['sha256'],
        data_hash=m['data_hash'],normalization_sha256=schedule_api.digest(m['normalization']),config_sha256=schedule_api.digest(c),
        schedule_sha256=c['schedule_binding']['sha256'],source_sha256=c['source_sha256'])


def resolve_same_run_resume(output,resume,*,data_hash,config,normalization):
    output=Path(output).resolve();resume=Path(resume).resolve()
    need(config.get('training_run_path')==str(output),'Copied/different run cannot resume')
    need(not any((output/n).exists() for n in LOCK_NAMES),'Selected/consumed run cannot resume')
    paths=[]
    for name in ('model','recovery'):
        folder=output/name
        if not folder.exists():continue
        for p in folder.iterdir():
            if re.fullmatch(r'step-[0-9]{12}',p.name):
                need(p.is_dir() and not p.is_symlink() and (p/'COMPLETE.json').is_file(),'Incomplete/corrupt declared checkpoint');paths.append(p.resolve())
    need(paths,'No complete same-run checkpoint');latest=max(paths,key=lambda p:int(p.name[5:]))
    need(resume==latest and sum(p.name==latest.name for p in paths)==1,'Only unambiguous latest complete same-run checkpoint can resume')
    m=_verified_metadata(latest,data_hash=data_hash,config=config,normalization=normalization)
    need(m['step']==int(latest.name[5:]) and m.get('optimizer_step')==m['step'] and m.get('schedule_cursor')==4*m['step'],'Fresh checkpoint age/cursor differs')
    with np.load(latest/'arrays.npz',allow_pickle=False) as arrays:verify_optimizer_age(arrays,m['leaf_contract'],m['step'])
    return m


def _terminal_segment_binding(run,receipt,receipt_sha,expected=None):
    """Bind the current segment only, preserving lawful later same-run resume."""
    raw_path=receipt.get('current_segment_path')
    need(isinstance(raw_path,str),'Current terminal segment path required')
    segment=Path(raw_path)
    need(segment.is_absolute() and segment.parent==run/'training_segments' and segment.suffix=='.json'
         and not segment.name.endswith(('.cancelled.json','.previous.json')),'Direct original terminal segment required')
    segment=trigger.canonical(segment)
    need(str(segment)==raw_path,'Canonical original terminal segment path required')
    cancellation=segment.with_suffix('.cancelled.json')
    need(not cancellation.exists() and not cancellation.is_symlink(),'Current training segment was cancelled')
    binding=dict(path=str(segment),sha256=receipt_sha,cancellation_path=str(cancellation))
    need(expected is None or binding==expected,'Terminal segment binding changed')
    need(_pinned({'path':binding['path'],'sha256':receipt_sha})==receipt,'Current segment and terminal receipt differ')
    need(not cancellation.exists() and not cancellation.is_symlink(),'Current training segment was cancelled')
    return binding


def verified_run_context(run):
    run=Path(run).resolve(strict=True);saved=read(run/'configuration.json');config=saved['config']
    need(config.get('training_run_path')==str(run),'Copied run cannot select')
    dataset,schedule,assets,spec,reference=load_inputs(saved['manifest'],config['mode'],condition_binding=config['condition_binding'],saved_normalization_binding=config['saved_normalization_binding']);verify_configuration(dataset,config,schedule,assets,spec,reference)
    need(saved['data_hash']==dataset.data_hash and saved['normalization']==dataset.normalization and saved['manifest_sha256']==data_api.MANIFEST_SHA256,'Saved dataset binding changed')
    _load_initialization(config)
    receipt_file=dict(path=str(run/'training_run.json'),sha256=schedule_api.file_sha(run/'training_run.json'))
    receipt=_pinned(receipt_file)
    need(receipt.get('schema')==VERSION and receipt.get('mode')==dataset.mode and receipt.get('training_complete') is True
         and receipt.get('local_step')==receipt.get('optimizer_step')==TOTAL_STEPS and receipt.get('schedule_cursor')==58400
         and receipt.get('error') is None and receipt.get('interrupted') is False,'Complete uninterrupted 14600-update receipt required')
    terminal=_terminal_segment_binding(run,receipt,receipt_file['sha256'])
    paths=[run/'model'/('step-%012d'%s) for s in schedule['candidate_steps']]
    from . import evaluation
    rows=evaluation.build_diagnostic_schedule(dataset)
    need(read(run/'diagnostic_schedule.json')==dict(data_hash=dataset.data_hash,rows=rows),'Diagnostic rows differ')
    for p in paths:
        binding=checkpoint_binding(p);evaluation.verify_checkpoint_report(run/'diagnostics'/('local_%06d'%binding['local_step']),dataset,binding)
    _pinned_bytes(receipt_file);_terminal_segment_binding(run,receipt,receipt_file['sha256'],terminal)
    return dict(dataset=dataset,config=config,schedule=schedule,candidate_paths=paths,completed=True,terminal_segment_binding=terminal)


def _finish_training(receipt,stop,handlers,save_receipt,write_cancellation):
    """Restore every handler and fail if cancellation arrived during final save.

    A separate cancellation marker survives a failed main-receipt rewrite when
    its storage remains writable. Any persistence failure still propagates as a
    nonzero process exit; it cannot turn cancellation into successful return.
    """
    errors=[]
    receipt['finished_utc']=utc()
    if stop[0]:receipt.update(training_complete=False,requested_segment_complete=False,interrupted=True)
    try:save_receipt()
    except BaseException as error:errors.append(error)
    finally:
        for sig,handler in handlers.items():
            try:signal.signal(sig,handler)
            except BaseException as error:errors.append(error)
    # The temporary handler may run during the last write or either restoration.
    # Recheck after all restorations before permitting a successful return.
    if stop[0]:
        receipt.update(training_complete=False,requested_segment_complete=False,interrupted=True,
            cancelled_utc=utc(),error=receipt.get('error') or 'InterruptedError: training cancelled during execution or finalization')
        cancellation=dict(schema='wam_rational_q4_training_cancellation_v1',training_complete=False,interrupted=True,
            local_step=receipt.get('local_step'),cancelled_utc=receipt['cancelled_utc'],error=receipt['error'])
        try:write_cancellation(cancellation)
        except BaseException as error:errors.append(error)
        try:save_receipt()
        except BaseException as error:errors.append(error)
        raise InterruptedError('Training cancelled; successful completion is invalid') from (errors[0] if errors else None)
    if errors:raise errors[0]


def run(args):
    output=Path(args.output).resolve();manifest=Path(args.manifest).resolve()
    condition=dict(path=str(Path(args.condition_receipt).resolve()),sha256=args.condition_sha256)
    normalization=dict(path=str(Path(args.normalization).resolve()),sha256=args.normalization_sha256)
    protected=[Path(__file__).resolve().parents[2],manifest.parent,ROOT/'public-assets',
               Path(condition['path']).parent,Path(normalization['path']).parent,trigger.B_RUN]
    separate(output,protected);validate_limit(args.stop_at,0)
    need(not any((output/n).exists() for n in LOCK_NAMES),'Locked C run')
    need(not output.exists() or args.resume is not None,'Fresh output or explicit same-run resume required')
    dataset,schedule,assets,spec,reference=load_inputs(manifest,args.mode,condition_binding=condition,saved_normalization_binding=normalization)
    _protect_original_run_outputs(output,condition,reference)
    if args.verify_only:
        if args.resume:
            saved=read(output/'configuration.json');verify_configuration(dataset,saved['config'],schedule,assets,spec,reference)
            need(saved['config']['condition_binding']==condition and saved['config']['saved_normalization_binding']==normalization,'Resume frozen inputs changed')
            resolve_same_run_resume(output,args.resume,data_hash=dataset.data_hash,config=saved['config'],normalization=dataset.normalization)
        return dict(schema=VERSION,mode=args.mode,verified=True,train_episodes=8,valid_windows=schedule['valid_window_count'],total_steps=schedule['total_steps'],total_draws=schedule['total_draws'],schedule_sha256=schedule_api.digest(schedule),source_sha256=source_hashes(),gpu_initialized=False,output_changed=False)
    output.mkdir(parents=True,exist_ok=True);schedule_path=output/'training_schedule.json'
    if args.resume:
        saved=read(output/'configuration.json');config=saved['config'];verify_configuration(dataset,config,schedule,assets,spec,reference)
        need(config['condition_binding']==condition and config['saved_normalization_binding']==normalization,'Resume frozen inputs changed')
        restored=resolve_same_run_resume(output,args.resume,data_hash=dataset.data_hash,config=config,normalization=dataset.normalization);step=restored['step']
    else:
        sb=save_reference_schedule(schedule_path,schedule,reference);config=make_config(dataset,schedule,sb,assets,spec,output,condition,normalization,reference);step=0
        freeze(output/'configuration.json',dict(config=config,data_hash=dataset.data_hash,normalization=dataset.normalization,manifest=str(manifest),manifest_sha256=data_api.MANIFEST_SHA256))
    limit=validate_limit(args.stop_at,step)
    from . import evaluation
    diagnostic=evaluation.build_diagnostic_schedule(dataset);freeze(output/'diagnostic_schedule.json',dict(data_hash=dataset.data_hash,rows=diagnostic))
    stop=[False];handlers={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    for sig in handlers:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    segments=output/'training_segments';segments.mkdir(exist_ok=True);segment=segments/(utc().replace(':','-')+'-'+uuid.uuid4().hex+'.json')
    receipt=dict(schema=VERSION,mode=args.mode,started_utc=utc(),pid=os.getpid(),training_run_path=str(output),local_step=step,optimizer_step=step,
        schedule_cursor=4*step,training_complete=False,interrupted=False,error=None,resume=None if args.resume is None else str(Path(args.resume).resolve()),source_sha256=config['source_sha256'],closed_loop_validated=False,current_segment_path=str(segment))
    parent=parent_module()
    if (output/'training_run.json').exists():
        with segment.with_suffix('.previous.json').open('xb') as f:f.write((output/'training_run.json').read_bytes())
    def save_receipt():
        receipt.update(local_step=step,optimizer_step=step,schedule_cursor=4*step,interrupted=stop[0]);parent._atomic_json(segment,receipt);parent._atomic_json(output/'training_run.json',receipt)
    save_receipt()
    try:
        import jax
        import jax.numpy as jnp
        import flax.nnx as nnx
        import optax
        from openpi.models import model as model_api
        from sim.auscultation_pi05_v7 import model as implementation
        runtime=implementation.initialize(61,5e-5);need(runtime['assets']==assets,'Official public assets changed during initialization')
        rng=jax.random.key(61);numeric=_numeric_initialization(runtime,rng,jax)
        initial=make_initialization(config,numeric,_pinned(reference['initialization_binding']))
        if args.resume:
            saved_initial,initial_sha=_load_initialization(config);need(saved_initial==initial,'Independent fresh initialization changed before restore')
            restored,key=implementation.restore(runtime,args.resume,dataset.data_hash,config,dataset.normalization);rng=jax.random.wrap_key_data(key)
        else:
            freeze(output/'initialization.json',initial);initial_sha=schedule_api.file_sha(output/'initialization.json')
        sample=implementation.sampler(runtime,steps=10)
        def objective(parameters,frozen,obs,actions,key):
            model=nnx.merge(runtime['graph'],parameters,frozen);model.train();return model.compute_loss(key,obs,actions,train=True).mean()
        gradient=jax.jit(jax.value_and_grad(objective,argnums=0))
        @jax.jit
        def update(parameters,state,grads):
            changes,state=runtime['optimizer'].update(grads,state,parameters);return optax.apply_updates(parameters,changes),state
        def save_checkpoint():
            need(source_hashes()==config['source_sha256'] and schedule_api.file_sha(schedule_path)==config['schedule_binding']['sha256'],'Training source or schedule changed')
            leaves=jax.tree.leaves((runtime['trainable'],runtime['opt_state']));arrays={'leaf_%05d'%i:np.asarray(v) for i,v in enumerate(jax.device_get(leaves))};arrays['jax_rng']=np.asarray(jax.random.key_data(rng))
            verify_optimizer_age(arrays,runtime['leaf_contract'],step)
            if step==0:verify_step_zero_arrays(arrays,runtime['leaf_contract'],initial['numerical_initialization'])
            metadata=dict(training_schema=VERSION,data_hash=dataset.data_hash,config=config,normalization=dataset.normalization,leaf_contract=runtime['leaf_contract'],
                optimizer_step=step,schedule_cursor=4*step,sampler_rng_state=dict(cursor=4*step,schedule_sha256=config['schedule_binding']['sha256']),
                initialization_sha256=initial_sha,best_step=None,best_validation_loss=None,heldout_axes=['diagnostic_original_train_only'])
            return parent.save_checkpoint(output/('model' if step in schedule['candidate_steps'] else 'recovery'),step=step,arrays=arrays,metadata=metadata)
        def diagnose(checkpoint):
            dest=output/'diagnostics'/('local_%06d'%step);binding=checkpoint_binding(checkpoint)
            if dest.exists():evaluation.verify_checkpoint_report(dest,dataset,binding);return
            predicted=[];normalized=[]
            for row in diagnostic:
                obs,_=data_api.make_batch(dataset,'train',[row['train_index']],runtime['tokenizer'])
                value=np.asarray(sample(runtime['trainable'],runtime['frozen'],model_api.Observation(**jax.device_put(obs)),jax.random.key(917)))
                need(value.shape==(1,10,7) and np.isfinite(value).all(),'Malformed diagnostic output')
                e=dataset.episodes[row['episode_index']];state=e['state'][row['start']]
                normalized.append(value[0]);predicted.append(data_api.decode_prediction(dataset,value[0],state,row['phase']))
            evaluation.save_checkpoint_report(dest,dataset,np.asarray(predicted),np.asarray(normalized),binding)
        last_saved=step
        if not args.resume:
            checkpoint=save_checkpoint();receipt['latest_checkpoint']=str(checkpoint)
            if not stop[0]:diagnose(checkpoint)
        elif step in schedule['candidate_steps'] and not stop[0]:diagnose(Path(args.resume))
        while step<limit and not stop[0]:
            indices=schedule_api.batch_indices(schedule,step);obs,actions=data_api.make_batch(dataset,'train',indices,runtime['tokenizer']);rng,key=jax.random.split(rng)
            loss,grads=gradient(runtime['trainable'],runtime['frozen'],model_api.Observation(**jax.device_put(obs)),jax.device_put(actions),key)
            need(bool(jax.device_get(jnp.all(jnp.stack([jnp.isfinite(g).all() for g in jax.tree.leaves(grads)])))) and np.isfinite(float(loss)),'Nonfinite training update')
            runtime['trainable'],runtime['opt_state']=update(runtime['trainable'],runtime['opt_state'],grads);jax.block_until_ready((runtime['trainable'],runtime['opt_state']));step+=1
            metrics=dict(mode=args.mode,local_step=step,optimizer_step=step,train_loss=float(loss),schedule_cursor=4*step,train_indices=indices.tolist(),segment=str(segment))
            with (output/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics,allow_nan=False)+'\n')
            if step in schedule['candidate_steps'] or step==limit or stop[0]:
                checkpoint=save_checkpoint();last_saved=step;receipt['latest_checkpoint']=str(checkpoint)
                if step in schedule['candidate_steps'] and not stop[0]:diagnose(checkpoint)
            if step%25==0:print(json.dumps(metrics),flush=True)
            save_receipt()
        if step>last_saved:receipt['latest_checkpoint']=str(save_checkpoint())
        receipt.update(training_complete=step==TOTAL_STEPS and not stop[0],requested_segment_complete=step==limit)
    except BaseException as error:
        receipt['error']=type(error).__name__+': '+str(error);raise
    finally:
        _finish_training(receipt,stop,handlers,save_receipt,
            lambda value:parent._atomic_json(segment.with_suffix('.cancelled.json'),value))
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--mode',choices=codec.MODES,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--condition-receipt',type=Path,required=True);p.add_argument('--condition-sha256',required=True)
    p.add_argument('--normalization',type=Path,required=True);p.add_argument('--normalization-sha256',required=True)
    p.add_argument('--resume',type=Path);p.add_argument('--stop-at',type=int);p.add_argument('--verify-only',action='store_true');args=p.parse_args()
    if args.verify_only:print(json.dumps(run(args),indent=2));return
    # No GPU resource acquisition until the scientific condition reconstructs.
    trigger.verify_receipt(str(args.condition_receipt.resolve()),args.condition_sha256)
    with (ROOT/'locks/isaac_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        need(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'Another GPU workload is active')
        print(json.dumps(run(args),indent=2),flush=True)


if __name__=='__main__':main()
