"""Versioned state-only ACT optimization with separate computational sources.

Native demonstration admission and normalization remain unchanged. Planned
commands and genuine student first-cell oracle proposals are explicitly admitted
only by this optimization protocol; they never become physical demonstrations.
"""
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

from . import m2_given_contact_runtime as runtime
from . import m2_given_contact_expanded_pilot_train as expanded
from . import m2_act_auxiliary_data as data
from . import m2_act_recovery_scope as scope
from . import rebuild_gpu_binding as binding
from . import m2_act_chunk_data as chunk_data
from . import m2_act_posture_head as posture_head
from . import m2_act_conditional_data as conditional_data
from . import m2_act_terminal_data as terminal_data

need=runtime.need
CONFIG_SCHEMA='m2_act_auxiliary_training_config_v1'
PILOT_SCHEMA='m2_act_auxiliary_act_pilot_v1'
SELECTION_SCHEMA='m2_act_auxiliary_act_selection_v1'
CPU_SCHEMA='m2_act_auxiliary_act_cpu_equivalence_v1'
PROTOCOL_SCHEMA='m2_act_auxiliary_act_protocol_v1'
CONFIG_SCHEMA_V2='m2_act_auxiliary_training_config_v2'
PILOT_SCHEMA_V2='m2_act_auxiliary_act_pilot_v2'
SELECTION_SCHEMA_V2='m2_act_auxiliary_act_selection_v2'
CPU_SCHEMA_V2='m2_act_auxiliary_act_cpu_equivalence_v2'
PROTOCOL_SCHEMA_V2='m2_act_auxiliary_act_protocol_v2'
CONFIG_SCHEMA_V3='m2_act_auxiliary_training_config_v3'
PILOT_SCHEMA_V3='m2_act_auxiliary_act_pilot_v3'
SELECTION_SCHEMA_V3='m2_act_auxiliary_act_selection_v3'
CPU_SCHEMA_V3='m2_act_auxiliary_act_cpu_equivalence_v3'
PROTOCOL_SCHEMA_V3='m2_act_auxiliary_act_protocol_v3'
CONFIG_SCHEMA_V4='m2_act_auxiliary_training_config_v4'
PILOT_SCHEMA_V4='m2_act_auxiliary_act_pilot_v4'
SELECTION_SCHEMA_V4='m2_act_auxiliary_act_selection_v4'
CPU_SCHEMA_V4='m2_act_auxiliary_act_cpu_equivalence_v4'
PROTOCOL_SCHEMA_V4='m2_act_auxiliary_act_protocol_v4'
TRAIN16_SCOPE=dict(path='/opt/wam-runtime/fast-wam/diagnostics/act-auxiliary-cohort-20261106-v2/COHORT_FREEZE.json',
    bytes=1521,sha256='adf6b0bd95669719989f6795c177f0a8419921dd79a257ca0c725f99b3220e71')
SCHEDULE=deepcopy(expanded.SCHEDULE)
FLAGS=deepcopy(expanded.FLAGS)
NATIVE_DATASET=dict(path='/opt/wam-runtime/fast-wam/datasets/act-student-recovery-masked-20261106-v1/COMPLETE.json',
    bytes=164297,sha256='265495e293b4ba95ca779cdc9693762716a3ea0010c6430ad2fde3b456f30cee')
NATIVE_COUNTS=dict(train=8373,validation=2345)
OFFICIAL_FORWARD_SHA256='83d954f79ccd3eaa6774dbea92524939c8ac08359bb355ba7cd37b8601e1e3af'
SOURCE20_MANIFEST=dict(path='/opt/wam-runtime/code/wam-act-recovery-20261106-v20/SOURCE_MANIFEST.json',
    bytes=282448,sha256='34a722d283db0dfa78718e037b2473739eaf066fc5c455343b8740b7b929c844')
SOURCE20_GENERATOR=dict(path='/opt/wam-runtime/code/wam-act-recovery-20261106-v20/sim/ausc_policy/m2_student_oracle_proposals.py',
    bytes=11637,sha256='b507ed21c339d2ca371ccb9b08472aeca4a1de2abb89080b45cf9ac1fb594e5e')
SOURCE20_REQUEST_SHA256='2ffcf68f41d1ffd634d4e97568b831976c71653394bf2eb61c9104eae5bd470d'
GROUPS=dict(native=dict(size=32,coefficient=.50),planned=dict(size=16,coefficient=.25),oracle=dict(size=16,coefficient=.25))
GROUPS_V3=dict(native=dict(size=32,coefficient=.50),planned=dict(size=8,coefficient=.125),
    conditional=dict(size=8,coefficient=.125),oracle=dict(size=16,coefficient=.25))


def _version(value):
    if value.get('schema') in (CONFIG_SCHEMA_V4,PILOT_SCHEMA_V4,SELECTION_SCHEMA_V4,CPU_SCHEMA_V4,PROTOCOL_SCHEMA_V4,terminal_data.SCHEMA):return 4
    if value.get('schema') in (CONFIG_SCHEMA_V3,PILOT_SCHEMA_V3,SELECTION_SCHEMA_V3,conditional_data.SCHEMA):return 3
    return 2 if _v2(value) else 1


def _api(version):
    need(type(version) is int and version in (1,2,3,4),'Explicit supported auxiliary data version required')
    return {1:data,2:chunk_data,3:conditional_data,4:terminal_data}[version]


def _groups(version):
    need(type(version) is int and version in (1,2,3,4),'Explicit supported optimizer version required')
    return GROUPS_V3 if version>=3 else GROUPS


def _kinds(version):
    return conditional_data.KINDS if version>=3 else data.KINDS


def _v2(value):
    return value.get('schema') in (CONFIG_SCHEMA_V2,PILOT_SCHEMA_V2,SELECTION_SCHEMA_V2,chunk_data.SCHEMA)


def _schemas(value):
    if _version(value)==4:return PILOT_SCHEMA_V4,SELECTION_SCHEMA_V4,CPU_SCHEMA_V4,PROTOCOL_SCHEMA_V4
    if _version(value)==3:return PILOT_SCHEMA_V3,SELECTION_SCHEMA_V3,CPU_SCHEMA_V3,PROTOCOL_SCHEMA_V3
    return (PILOT_SCHEMA_V2,SELECTION_SCHEMA_V2,CPU_SCHEMA_V2,PROTOCOL_SCHEMA_V2) if _v2(value) else (PILOT_SCHEMA,SELECTION_SCHEMA,CPU_SCHEMA,PROTOCOL_SCHEMA)


def make_model(config,*,device,training=False):
    _require_head_version(config)
    torch,policy,software=runtime.make_model({'lerobot_root':config['lerobot_root']},device=device)
    if _version(config)>=2:
        posture_head.install(policy,config['posture_head'])
        if training and _version(config)!=4:posture_head.initialize_training(policy,config['posture_head'])
    return torch,policy,software


def _require_head_version(config):
    version=_version(config)
    if version>=2:
        spec=config.get('posture_head',{});posture_head.validate_spec(spec)
        need(spec['schema']==(posture_head.SCHEMA_V2 if version>=3 else posture_head.SCHEMA),
            'Model/config head version mismatch; legacy weights cannot be relabeled')


def auxiliary_metadata(ref,*,version=1):
    """Reject identity/profile/eligibility faults before any numerical loader."""
    api=_api(version)
    if version==4:
        _,metadata=terminal_data.metadata(ref,scope.TRAIN16_IDS)
        need(all(s.get('pose_id') in runtime_scope_poses() and s.get('target_id') in ('R3','L3') for s in metadata['sources']) and
            {s['kind'] for s in metadata['sources']}==set(_kinds(version)),
            'All original separate known-pose TRAIN computational kinds required')
        return metadata
    if version==3:
        _,metadata=conditional_data.metadata(ref,scope.TRAIN16_IDS)
        need(all(s.get('pose_id') in runtime_scope_poses() and s.get('target_id') in ('R3','L3') for s in metadata['sources']),
             'Explicit known-pose R3/L3 computational sources required')
        need({s['kind'] for s in metadata['sources']}==set(conditional_data.KINDS),
             'All three separate computational source kinds required')
        current={s['target_id'] for s in metadata['sources'] if s['schema']==conditional_data.SOURCE_SCHEMA and s['kind']=='student_oracle_unexecuted'}
        need(current=={'R3','L3'},'New genuine source24 TRAIN oracle queries required for both R3 and L3')
        return metadata
    expanded.full_reference(ref);complete=runtime.read_bound(ref)
    need(complete.get('schema')==api.SCHEMA and complete.get('complete') is True and
         all(complete.get(k) is v for k,v in data.FLAGS.items()),'Separate unexecuted auxiliary preparation required')
    metadata=runtime.read_bound(complete['metadata'])
    binding.validate_auxiliary_dataset(metadata,schema=api.SOURCE_SCHEMA,train_pool_ids=scope.TRAIN16_IDS)
    for source in metadata['sources']:
        kind=source['kind']
        need(kind in data.KINDS and source.get('pose_id') in runtime_scope_poses() and
             source.get('target_id') in ('R3','L3'),'Explicit known-pose R3/L3 computational source required')
        if kind=='planned_ideal_tracking':
            need(source.get('verification_profile')=='original_full_cpu_preview' and source.get('injected_factories')==[],
                 'Production original_full_cpu_preview required; injected CPU fixture profiles rejected')
        else:
            need(source.get('observation_origin')==data.ORIGINS[kind] and
                 source.get('source_episode_verdict_unchanged') is True and source.get('original_source_success') is False and
                 {'generator','native','owner','child','proposal','outcome','request','diagnostic','diagnostic_root',
                  'frozen_request','frozen_manifest','original_arrays','source','geometry'}<=set(source.get('bindings',{})),
                 'Genuine frozen authenticated negative TRAIN student oracle source required')
            if version==2:
                chunk_data.require_frozen_archive(source['bindings']['frozen_manifest'],source['bindings']['generator'])
            else:_require_source20(source['bindings'])
    need({s['kind'] for s in metadata['sources']}==set(data.KINDS),'Both planned and student oracle sources are required')
    if version==2:
        need(complete.get('optimization_admission')==chunk_data.ADMISSION and metadata.get('optimization_admission')==chunk_data.ADMISSION,
             'Explicit full computed prefix optimization admission required')
        chunk_data.require_current_targets(metadata['sources'])
    return metadata


def runtime_scope_poses():
    return ('p000474_a001230_f0000','p000476_a001231_f0023')


def validate_config(config):
    version=_version(config) if isinstance(config,dict) else 1
    v2=version>=2
    optional={'replacement_pod_gpu_binding'}&set(config) if isinstance(config,dict) else set()
    if v2:optional=optional|{'posture_head'}
    required={'schema','base','dataset','auxiliary','evaluation_cohort'}|({'initialization'} if version==4 else set())
    need(isinstance(config,dict) and set(config)==set(SCHEDULE)|required|optional and
         config.get('schema') in (CONFIG_SCHEMA,CONFIG_SCHEMA_V2,CONFIG_SCHEMA_V3,CONFIG_SCHEMA_V4) and all(config.get(k)==v and type(config[k]) is type(v) for k,v in SCHEDULE.items()) and
         Path(str(config.get('base',''))).is_absolute(),'Exact finite 20k/64/500 auxiliary optimization schedule required; no resume')
    for key in ('dataset','auxiliary','evaluation_cohort'):expanded.full_reference(config[key])
    need(config['dataset']==NATIVE_DATASET,'Exact immutable native recovery29 bank required')
    # Read metadata only. No native or auxiliary numerical arrays before overlap checks.
    cohorts=scope.validate(config['evaluation_cohort']);frozen=runtime.read_bound(config['evaluation_cohort'])
    need(frozen.get('schema')==scope.TRAIN16_SCHEMA,'Explicit new TRAIN16 scope required; legacy TRAIN4 admission is separate')
    if v2:
        need(config['evaluation_cohort']==TRAIN16_SCOPE,'Pinned accepted TRAIN16 pretraining scope required')
        if version>=3:posture_head.validate_bound_spec(config['posture_head'],native_normalization(config),version=2)
        else:posture_head.validate_bound_spec(config['posture_head'],native_normalization(config))
    if version==4:
        from . import m2_act_warm_start as warm_start
        expanded.full_reference(config['initialization'])
        warm_start.validate_receipt(config['initialization'],normalization=native_normalization(config),posture_head=config['posture_head'])
    auxiliary=auxiliary_metadata(config['auxiliary'],version=version) if v2 else auxiliary_metadata(config['auxiliary'])
    native=runtime.read_bound(config['dataset'])
    binding.validate_dataset(native,schema=expanded.MASKED_PREPARATION_SCHEMA,task_contract=runtime.task_contract())
    need(native.get('counts')==NATIVE_COUNTS,'Unchanged native TRAIN8373/selection-VAL2345 population required')
    actual=set(native['split_subject_ids']['train'])|{s['subject_id'] for s in auxiliary['sources']}
    scope.validate_train16_metadata(frozen,actual_train_ids=actual)
    need(not actual&set(cohorts['selection_validation']+cohorts['frozen_evaluation']),
         'Auxiliary/native optimizer overlaps selection or fresh evaluation identities')
    if 'replacement_pod_gpu_binding' in optional:binding.validate(config['replacement_pod_gpu_binding'],base=config['base'])
    return config


def training_protocol(config):
    _require_head_version(config)
    result=dict(schema=_schemas(config)[3],schedule=deepcopy(config),model=runtime.shared.native_spec(),parameter_count=2658823,
        state_only=True,observation_dimensions={'state':7,'environment':27,'total':34},raw_state_dimensions=17,
        history_frames=2,action_horizon=20,execution_prefix=10,model_dropout=.1,task_contract=runtime.task_contract(),
        evaluation_steps=expanded.evaluation_steps(config),selection='earliest minimum full native selection-VAL2345 raw masked MAE',
        label_protocol=runtime.shared.MASKED_LABEL_PROTOCOL,normalization='pinned real native TRAIN only; no auxiliary refit',
        objective={name:dict(**value,loss='official forward masked valid-cell/joint mean') for name,value in _groups(_version(config)).items()},
        native_sampling='half real near60mm origins, half uniform real origins',
        auxiliary_sampling='separate kind then uniform body -> pose -> target -> window',
        auxiliary_admission=dict(schema=binding.AUXILIARY_ADMISSION_SCHEMA,preparation_schema=data.SCHEMA,
            computational_optimization_eligible=True,**data.FLAGS),
        augmentation={'q_noise_std':0.,'q_dropout':0.},gradient_clip_norm=1.,gradient_clip_error_if_nonfinite=True,
        action_clipping=False,
        selection_auxiliary_used=False,selection_fresh_used=False,official_forward_sha256=OFFICIAL_FORWARD_SHA256,**FLAGS)
    if _version(config)>=2:
        result.update(posture_head=deepcopy(config['posture_head']),
            posture_initialization=posture_head.initialization_spec(config['posture_head']),head_active_in_loss_and_metrics=True,
            model_definition='official ACT backbone with deterministic smooth absolute posture-box output parameterization',
            oracle_supervision='full computed prefix up to20; genuine current H2 only; no future measured histories')
        result['auxiliary_admission']=dict(chunk_data.ADMISSION,preparation_schema=chunk_data.SCHEMA)
    if _version(config)>=3:
        result['auxiliary_admission']=dict(conditional_data.ADMISSION,preparation_schema=conditional_data.SCHEMA)
        result['auxiliary_sampling']='uniform body -> pose -> target -> anchor-or-window -> perturbation; groups separately sampled'
        result['conditional_supervision']='generated static ideal H2/q/v=a=0; canonical L2 continuation; no actual/future measurements'
        result['treatment']='combined conditional/genuine coverage, STOP residual adapter and head v2 softness; not a causal ablation'
    if _version(config)==4:
        from . import m2_act_warm_start as warm_start
        result['posture_initialization']=warm_start.protocol(config['initialization'])
        result['auxiliary_admission']=dict(terminal_data.ADMISSION,preparation_schema=terminal_data.SCHEMA)
        result['treatment']='fine-tuned ACT6: ACT5 selected19000 initialization plus 20000 new optimizer steps and terminal-query data; not a causal data-only ablation'
    return result


def code_binding(*,version=1):
    value=expanded.code_binding();root=Path(__file__).resolve().parent
    for name in ('m2_act_auxiliary_train.py','m2_act_auxiliary_data.py','m2_act_planned_supervision.py',
                 'm2_given_contact_recovery.py','m2_student_oracle_proposals.py','m2_student_recovery_prefix.py'):
        value['files'][name]=runtime.reference(root/name)
    if version>=2:
        for name in ('m2_act_chunk_data.py','m2_act_posture_head.py'):
            value['files'][name]=runtime.reference(root/name)
    if version>=3:
        for name in ('m2_act_conditional_data.py','m2_act_conditional_supervision.py'):
            value['files'][name]=runtime.reference(root/name)
    if version==4:
        for name in ('m2_act_terminal_data.py','m2_act_frozen26_data.py','m2_act_warm_start.py'):
            value['files'][name]=runtime.reference(root/name)
    return value


def validate_code(value,*,version=1):
    current=code_binding(version=version)
    if version==4:
        need(value==current,'Current V4 numerical source origins/commit/bytes changed')
        return
    need(set(value.get('files',{}))==set(current['files']) and all(
         {k:ref[k] for k in ('bytes','sha256')}=={k:current['files'][name][k] for k in ('bytes','sha256')}
         for name,ref in value['files'].items()),'Mixed-source trained numerical/protocol source bytes changed')


def validate_official_forward(software):
    ref=software.get('vendor',{}).get('files',{}).get('src/lerobot/policies/act/modeling_act.py',{})
    need(ref.get('sha256')==OFFICIAL_FORWARD_SHA256,'Pinned official masked valid-cell mean forward bytes required')


def _require_source20(refs):
    need(refs.get('frozen_manifest')==SOURCE20_MANIFEST and refs.get('generator')==SOURCE20_GENERATOR and
         refs.get('frozen_request',{}).get('sha256')==SOURCE20_REQUEST_SHA256,
         'Pinned actual frozen source20 manifest/generator/request required; arbitrary archives cannot self-authenticate')


def native_normalization(config):
    native=runtime.read_bound(config['dataset'])
    return runtime.validate_normalization(runtime.read_bound(native['normalization']))


def validate_oracle_sample(samples,source_index,proposal,authenticated_query):
    """Resealing an export cannot substitute genuine H2 or first-cell labels."""
    source=samples['sources'][source_index];row=source['frozen_physics_row']
    need(proposal.get('query')==authenticated_query,'Oracle query differs from genuine authenticated student input')
    history,times,relative,_=data._oracle_query(proposal,source['metadata'],row)
    ids=np.flatnonzero(samples['source_indices']==source_index)
    need(len(ids)==1,'Each genuine oracle source must retain its single actual query')
    i=int(ids[0]);mask=samples['setter_is_pad'][i]
    need(np.array_equal(samples['source_histories'][i],history) and np.array_equal(samples['source_times'][i],times) and
         np.array_equal(samples['relative_setters'][i,0],relative[0]) and not mask[0] and np.all(mask[1:]) and
         np.all(samples['relative_setters'][i,1:]==0),
         'Oracle exported arrays must equal genuine H2 and the bound first raw setter only')


def authenticate_oracles(auxiliary):
    """Authenticate genuine queries once in their original frozen source20 code.

    Original model admission remains entirely in that immutable code archive.
    Only actual pre-query H2/current origin is compared; no future measurement
    or physical-success label is invented by the computational oracle.
    """
    version=_version(auxiliary.get('samples',{}));v2=version>=2
    if version==4:
        # load_preparation has just rerun ALL original parent admission in26.
        # Its historical source20/23/24 paths must never be read by a28 V3 API.
        provenance=auxiliary['provenance'];proof=provenance['parent_archive_validation']
        need(provenance.get('parent_complete')==terminal_data.PARENT_COMPLETE and
            proof.get('manifest')==terminal_data.frozen26.SOURCE26_MANIFEST and proof.get('array_proofs'),
            'Full original26 parent authentication required before new V4 queries')
        parent=dict(parent_complete=terminal_data.PARENT_COMPLETE,manifest=terminal_data.frozen26.SOURCE26_MANIFEST,
            parent_original26_validation=deepcopy(proof),actual_history_authenticated=True,future_measurements_claimed=False)
        return [parent]+terminal_data.authenticate_new_genuine_groups(auxiliary,scope.TRAIN16_IDS)
    groups={}
    for source_index,source in enumerate(auxiliary['sources']):
        if source['kind']!='student_oracle_unexecuted':continue
        refs=source['bindings']
        if version==3:conditional_data.require_oracle_archive(source)
        elif v2:chunk_data.require_frozen_archive(refs['frozen_manifest'],refs['generator'])
        else:_require_source20(refs)
        request=runtime.read_bound(refs['request'])
        key=runtime.digest(dict(source=request['source'],manifest=refs['frozen_manifest']))
        groups.setdefault(key,dict(refs=refs,request=request,sources=[]))['sources'].append((source_index,source))
    program='''import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from sim.ausc_policy import m2_student_oracle_proposals as original
assert Path(original.__file__).resolve() == Path(sys.argv[1])/'sim/ausc_policy/m2_student_oracle_proposals.py'
payload=json.load(sys.stdin)
binding=original.authenticate_source(payload['source'])
queried=[original.query_at(binding,row) for row in payload['rows']]
queries=[dict(query=q,snapshot=snapshot,commands=commands.tolist()) for q,snapshot,commands in queried] if payload.get('full_prefix') else [q for q,_,_ in queried]
print(json.dumps(queries,allow_nan=False))
'''
    receipts=[]
    for group in groups.values():
        refs=group['refs'];manifest_ref=refs['frozen_manifest'];root=Path(manifest_ref['path']).parent.resolve()
        manifest=runtime.read_bound(manifest_ref)
        need(isinstance(manifest,list) and manifest,'Complete frozen source20 inventory required')
        names=set()
        for item in manifest:
            relative=item.get('file');need(isinstance(relative,str),'Frozen manifest file path required')
            path=root/relative
            need(not path.is_symlink() and path.resolve().is_relative_to(root) and relative not in names,
                 'Unique regular files inside frozen oracle source archive required')
            names.add(relative);actual=runtime.reference(path)
            need(all(actual[k]==item.get(k) for k in ('bytes','sha256')),'Frozen oracle source archive bytes changed')
        generator=root/'sim/ausc_policy/m2_student_oracle_proposals.py'
        need('sim/ausc_policy/m2_student_oracle_proposals.py' in names and runtime.reference(generator)==refs['generator'],
             'Oracle generator must be its original fully bound frozen source20 module')
        rows=sorted({s['frozen_physics_row'] for _,s in group['sources']})
        payload=dict(source=group['request']['source'],rows=rows,full_prefix=v2)
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
        result=subprocess.run([sys.executable,'-I','-B','-c',program,str(root)],input=json.dumps(payload),
            text=True,capture_output=True,check=False,timeout=1800,env=environment,cwd=root)
        need(result.returncode==0,'Frozen source20 actual student authentication failed: '+result.stderr[-8000:])
        queries=json.loads(result.stdout);need(len(queries)==len(rows),'Complete authenticated source query set required')
        by_row=dict(zip(rows,queries))
        for source_index,source in group['sources']:
            proposal=runtime.read_bound(source['bindings']['proposal'])
            authenticated=by_row[source['frozen_physics_row']]
            if v2:
                chunk_data.validate_oracle_sample(auxiliary['samples'],source_index,proposal,authenticated['query'],snapshot=authenticated['snapshot'])
            else:validate_oracle_sample(auxiliary['samples'],source_index,proposal,authenticated)
        receipts.append(dict(source=group['request']['source'],manifest=manifest_ref,physics_rows=rows,
            actual_history_authenticated=True,future_measurements_claimed=False))
    need(receipts,'At least one authenticated genuine student oracle source required')
    return receipts


def _audit_posture_inputs(config,prepared,auxiliary):
    """Audit every retained target and original source profile without clipping."""
    spec=config['posture_head'];profile=spec['physical_posture_profile'];refs=[]
    for episode in prepared['preparation']['episodes']:refs.append(episode['config'])
    for source in auxiliary['sources']:
        refs.extend((source['bindings']['source'],source['bindings']['geometry']))
    seen=set()
    for ref in refs:
        key=runtime.digest(ref)
        if key in seen:continue
        seen.add(key);value=runtime.read_bound(ref)
        need(value.get('posture_limits')==profile,'Native/planned/oracle source must retain the sole original full posture profile')
    audits={}
    for split in ('train','validation'):
        samples=prepared[split]
        audits[split]=posture_head.audit_targets(samples['actions'],samples['action_is_pad'],samples['histories'],spec)
    samples=auxiliary['samples']
    audits['auxiliary']=posture_head.audit_targets(samples['relative_setters'],samples['setter_is_pad'],samples['source_histories'],spec)
    prepared['posture_target_audit']=audits


def load_inputs(config):
    validate_config(config)
    # Full native validation and normalization are exactly the original loader.
    prepared=expanded.load_dataset(config['dataset'])
    auxiliary=_api(_version(config)).load_preparation(config['auxiliary'],scope.TRAIN16_IDS)
    if _version(config)>=2:
        if _version(config)==2:chunk_data.validate_lineage(auxiliary,scope.TRAIN16_IDS)
        _audit_posture_inputs(config,prepared,auxiliary)
    authentication=authenticate_oracles(auxiliary)
    frozen=runtime.read_bound(config['evaluation_cohort'])
    native_scope=deepcopy(prepared['scope'])
    prepared['scope']['train'].update(subject_ids=frozen['optimized_train_ids'],
        native_subject_ids=native_scope['train']['subject_ids'],native_episode_count=native_scope['train']['episode_count'],
        native_window_count=native_scope['train']['window_count'])
    prepared.update(auxiliary=auxiliary,optimization_scope=frozen,native_scope=native_scope,oracle_authentication=authentication)
    return prepared


def _tree(sources,source_indices,kind,*,anchor_balanced=False):
    tree={}
    for i,source in enumerate(sources):
        if source['kind']!=kind:continue
        targets=tree.setdefault(source['subject_id'],{}).setdefault(source['pose_id'],{})
        rows=np.flatnonzero(source_indices==i).tolist()
        if anchor_balanced:targets.setdefault(source['target_id'],{}).setdefault(source['anchor'],[]).extend(rows)
        else:targets.setdefault(source['target_id'],[]).extend(rows)
    return tree


def build_cache(prepared,auxiliary,*,torch=None,device='cpu'):
    """Pack validated arrays and sampling groups once, before optimization."""
    version=_version(auxiliary['samples']);groups=_groups(version)
    need({s['kind'] for s in auxiliary['sources']}==set(_kinds(version)),'Both planned and oracle plus declared conditional auxiliary kinds required')
    native=prepared['train'];samples=auxiliary['samples']
    packed=dict(native=runtime.given.pack_batch(native,prepared['normalization'],np.arange(len(native['states'])),training=False),
        auxiliary=_api(version).pack_batch(samples,prepared['normalization'],np.arange(len(samples['source_indices']))))
    near=np.flatnonzero(np.linalg.norm(native['states'][:,7:10],axis=1)<=.060)
    need(len(near)>0,'Real near60mm TRAIN origins required')
    native_subjects=np.empty(len(native['states']),dtype=object);native_subjects[:]=None
    native_sources=np.empty(len(native['states']),dtype=object);native_sources[:]=None
    for episode in prepared['preparation']['episodes']:
        if episode['split']=='train':
            start,end=episode['split_sample_range'];native_subjects[start:end]=episode['subject_id']
            native_sources[start:end]='native:'+episode['episode_id']
    need(all(isinstance(s,str) for s in native_subjects),'Every real origin needs actual native subject identity')
    trees={name:_tree(auxiliary['sources'],samples['source_indices'],kind,anchor_balanced=version>=3 and name=='conditional')
        for name,kind in dict(planned='planned_ideal_tracking',oracle='student_oracle_unexecuted',
            **({'conditional':conditional_data.CONDITIONAL_KIND} if version>=3 else {})).items()}
    need(all(trees.values()),'Both auxiliary kinds need nonempty balanced sampling groups')
    if torch is not None:
        packed={group:{key:torch.from_numpy(value).to(device) for key,value in batch.items()} for group,batch in packed.items()}
    return dict(packed=packed,near=near,native_count=len(native['states']),trees=trees,version=version,groups=groups,
        source_ids=dict(native=native_sources,auxiliary=np.asarray([auxiliary['sources'][i].get('source_id','fixture-unqualified')
            for i in samples['source_indices']],dtype=object)),
        kinds=dict(native=np.full(len(native_subjects),'native',dtype=object),auxiliary=np.asarray([
            auxiliary['sources'][i]['kind'] for i in samples['source_indices']],dtype=object)),
        subjects=dict(native=native_subjects,auxiliary=np.asarray([auxiliary['sources'][i]['subject_id']
            for i in samples['source_indices']],dtype=object)))


def sample_cached_indices(rng,cache):
    near=cache['near'];ids=dict(native=np.r_[near[rng.integers(len(near),size=16)],rng.integers(cache['native_count'],size=16)])
    def choose(values):return values[int(rng.integers(len(values)))]
    for kind,tree in cache['trees'].items():
        selected=[]
        for _ in range(cache.get('groups',GROUPS)[kind]['size']):
            poses=tree[choose(sorted(tree))];targets=poses[choose(sorted(poses))]
            windows=targets[choose(sorted(targets))]
            if isinstance(windows,dict):windows=windows[choose(sorted(windows))]
            selected.append(choose(windows))
        ids[kind]=np.asarray(selected,dtype=np.int64)
    return ids


def cached_batches(cache,indices):
    return {name:{key:value[indices[name]] for key,value in cache['packed']['native' if name=='native' else 'auxiliary'].items()}
        for name in cache.get('groups',GROUPS)}


def mixed_loss(policy,batches,*,version=1):
    """One official valid-cell mean per declared group, weighted exactly once."""
    groups=_groups(version)
    need(set(batches)==set(groups),'Exactly declared native/planned/conditional/oracle loss groups required')
    total=None;logged={}
    for name,contract in groups.items():
        batch=batches[name];valid=int((~batch['action_is_pad']).sum().item())
        need(valid>0 and len(batch['action'])==contract['size'],'Nonempty fixed declared supervision group size required')
        loss,_=policy(batch);need(loss.ndim==0,'Official ACT scalar loss required')
        weighted=contract['coefficient']*loss;total=weighted if total is None else total+weighted
        scalar=float(loss.detach().cpu()) if hasattr(loss,'detach') else float(loss)
        logged[name]=dict(loss=scalar,coefficient=contract['coefficient'],valid_cells=valid,valid_joints=valid*7,
            examples=len(batch['action']))
    return total,logged


def record_draws(value,cache,indices):
    """Call only after a successful optimizer step; count actual selected examples."""
    seen=set(value['actual_optimized_train_ids'])
    for name,index in indices.items():
        group='native' if name=='native' else 'auxiliary'
        value['gradient_examples'][name]+=len(index)
        for subject in cache['subjects'][group][index]:
            counts=value['gradient_subject_examples'][name];counts[subject]=counts.get(subject,0)+1;seen.add(subject)
        if cache.get('version',1)>=3:
            for key,field in (('gradient_source_examples','source_ids'),('gradient_kind_examples','kinds')):
                for label in cache[field][group][index]:
                    counts=value[key][name];counts[label]=counts.get(label,0)+1
    value['actual_optimized_train_ids']=sorted(seen)


def contribution_inventory(config):
    """Bind reported source counts to actual immutable metadata, without arrays."""
    native=runtime.read_bound(config['dataset']);auxiliary=auxiliary_metadata(config['auxiliary'],version=_version(config))
    inventory={k:{} for k in GROUPS_V3}
    for episode in native['episodes']:
        if episode['split']=='train':inventory['native']['native:'+episode['episode_id']]=dict(subject_id=episode['subject_id'],kind='native')
    mapping={'planned_ideal_tracking':'planned','student_oracle_unexecuted':'oracle',conditional_data.CONDITIONAL_KIND:'conditional'}
    for source in auxiliary['sources']:
        inventory[mapping[source['kind']]][source['source_id']]=dict(subject_id=source['subject_id'],kind=source['kind'])
    return inventory


def checkpoint(policy,prepared,inputs,step,metrics,protocol,optimization):
    value=expanded.original.checkpoint(policy,prepared,inputs,step,metrics)
    value.update(schema={PROTOCOL_SCHEMA:PILOT_SCHEMA,PROTOCOL_SCHEMA_V2:PILOT_SCHEMA_V2,PROTOCOL_SCHEMA_V3:PILOT_SCHEMA_V3,PROTOCOL_SCHEMA_V4:PILOT_SCHEMA_V4}[protocol['schema']],protocol=protocol,optimization_scope=prepared['optimization_scope'],
        native_scope=prepared['native_scope'],auxiliary_counts=prepared['auxiliary']['counts'],
        oracle_authentication=prepared['oracle_authentication'],optimization=deepcopy(optimization))
    if protocol['schema'] in (PROTOCOL_SCHEMA_V2,PROTOCOL_SCHEMA_V3,PROTOCOL_SCHEMA_V4):
        value['posture_head_diagnostics']=posture_head.diagnostics(policy)
        value['posture_target_audit']=deepcopy(prepared['posture_target_audit'])
    if protocol['schema']==PROTOCOL_SCHEMA_V4:value['initialization']=deepcopy(prepared.get('initialization'))
    return value


def _validate_initialization(config,inputs,value):
    """Validate pinned provenance and the actual step0 check, without old loading."""
    from . import m2_act_warm_start as warm_start
    receipt=warm_start.validate_receipt(config['initialization'],normalization=native_normalization(config),posture_head=config['posture_head'])
    need(inputs.get('initialization')==config['initialization'] and isinstance(value,dict) and
        value.get('receipt')==config['initialization'] and value.get('equivalence')==inputs.get('initialization_equivalence'),
        'Exact V4 initialization receipt/equivalence input references required')
    expanded.full_reference(value['equivalence']);equivalence=runtime.read_bound(value['equivalence'])
    difference=equivalence.get('prediction_max_abs_rad')
    need(equivalence.get('schema')==warm_start.EQUIVALENCE_SCHEMA and equivalence.get('passed') is True and
        equivalence.get('initialization')==config['initialization'] and equivalence.get('initial_tensor_proofs')==receipt['tensor_proofs'] and
        equivalence.get('optimizer_step')==0 and equivalence.get('optimizer_imported') is False and
        equivalence.get('scheduler_imported') is False and equivalence.get('final_linear_reset') is False and
        equivalence.get('probe_split')=='train' and equivalence.get('probe_indices')==warm_start.PROBE_TRAIN_INDICES and
        equivalence.get('prediction_rtol')==1e-5 and equivalence.get('prediction_atol')==1e-6 and
        type(difference) in (int,float) and np.isfinite(difference) and difference>=0,
        'Actual ACT6 initial tensor/prediction equivalence and optimizer step0 required')
    need(value.get('optimizer')==dict(name='AdamW',initial_step=0,initial_state_entries=0,lr=config['lr'],
        weight_decay=config['weight_decay'],betas=[.9,.999],eps=1e-8,optimizer_imported=False,scheduler_imported=False),
        'ACT6 must use fresh AdamW step0; historical optimizer/scheduler cannot resume')


def load_selected_policy(selection):
    pilot_schema,selection_schema,_,_=_schemas(selection)
    need(selection.get('schema') in (pilot_schema,selection_schema) and
         all(selection.get(k) is v for k,v in FLAGS.items()),'Explicit mixed-source ACT selected weights required')
    config=validate_config(runtime.read_bound(selection['inputs']['config']))
    need(_version(selection)==_version(config),'Selected model/config version mismatch; head cannot relabel legacy weights')
    need(selection.get('protocol')==training_protocol(config) and selection['inputs']['dataset']==config['dataset'] and
         selection['inputs'].get('auxiliary')==config['auxiliary'] and
         selection.get('optimization_scope')==runtime.read_bound(config['evaluation_cohort']),
         'Exact native/auxiliary/scope/protocol binding differs')
    if _version(selection)==4:_validate_initialization(config,selection['inputs'],selection.get('initialization'))
    scope.validate_train16_metadata(selection['optimization_scope'],actual_train_ids=selection['scope']['train']['subject_ids'])
    _validate_draws(selection.get('selected_optimization'),selection['selected_step'],selection['optimization_scope'],
        selection['native_scope']['train']['subject_ids'],version=_version(selection),
        inventory=contribution_inventory(config) if _version(selection)>=3 else None)
    validate_code(selection['inputs']['code'],version=_version(selection))
    torch,policy,software=make_model(config,device='cpu')
    validate_official_forward(software);saved=runtime.read_bound(selection['inputs']['software'])
    for key in ('vendor','dependencies','python','prefix','python_version','source_bindings'):
        need(saved[key]==software[key],'Official mixed-source CPU environment changed: '+key)
    packet=torch.load(io.BytesIO(runtime.bound_path(selection['best']).read_bytes()),map_location='cpu',weights_only=True)
    need(packet.get('schema')==pilot_schema and packet.get('step')==selection['selected_step'] and
         packet.get('task_contract')==runtime.task_contract() and packet.get('model')==runtime.shared.native_spec() and
         all(packet.get(k) is v for k,v in FLAGS.items()) and
         all(packet.get(k)==selection.get(k) for k in ('inputs','scope','protocol','optimization_scope','native_scope',
              'auxiliary_counts','oracle_authentication')) and packet.get('optimization')==selection.get('selected_optimization'),
         'Exact selected mixed-source checkpoint required')
    if _version(selection)==4:need(packet.get('initialization')==selection['initialization'],'Selected V4 initialization binding differs')
    need(runtime.validate_normalization(packet['normalization'])==native_normalization(config),
         'Selected normalization must equal the pinned real native TRAIN normalization; no auxiliary refit')
    policy.load_state_dict(packet['state_dict'],strict=True);policy.eval()
    return torch,policy,packet


def _validate_draws(value,steps,frozen,native_subjects,*,version=1,inventory=None):
    groups=_groups(version)
    need(isinstance(value,dict) and set(value.get('gradient_examples',{}))==set(groups) and
         set(value.get('gradient_subject_examples',{}))==set(groups),'Complete actual gradient-contributing example counts required')
    subjects=set()
    for name,contract in groups.items():
        counts=value['gradient_subject_examples'][name]
        expected=steps*contract['size']
        need(value['gradient_examples'][name]==expected and isinstance(counts,dict) and counts and
             all(type(n) is int and n>0 for n in counts.values()) and sum(counts.values())==expected and
             set(counts)<=set(frozen['optimized_train_ids']) and (name!='native' or set(counts)<=set(native_subjects)),
             'Actual gradient-contributing identity/example counts differ: '+name)
        subjects.update(counts)
    need(value.get('actual_optimized_train_ids')==sorted(subjects),'Actual optimized identities differ from contributing records')
    scope.validate_train16_metadata(frozen,actual_train_ids=subjects)
    if version>=3:
        need(isinstance(inventory,dict) and set(inventory)==set(groups) and
            set(value.get('gradient_source_examples',{}))==set(groups) and
            set(value.get('gradient_kind_examples',{}))==set(groups),'V3 actual source and kind contribution inventory required')
        for name in groups:
            counts=value['gradient_source_examples'][name];by_subject={};by_kind={}
            need(isinstance(counts,dict) and counts and set(counts)<=set(inventory[name]) and
                all(type(n) is int and n>0 for n in counts.values()) and sum(counts.values())==value['gradient_examples'][name],
                'Actual V3 source contribution counts differ: '+name)
            for source_id,count in counts.items():
                entry=inventory[name][source_id];subject=entry['subject_id'];kind=entry['kind']
                by_subject[subject]=by_subject.get(subject,0)+count;by_kind[kind]=by_kind.get(kind,0)+count
            need(by_subject==value['gradient_subject_examples'][name] and by_kind==value['gradient_kind_examples'][name],
                'Actual V3 source/kind/subject contribution sums differ: '+name)


def validate_complete(value):
    pilot_schema,_,_,_=_schemas(value)
    need(value.get('schema')==pilot_schema and value.get('complete') is True and
         value.get('task_contract')==runtime.task_contract() and value.get('model')==runtime.shared.native_spec() and
         all(value.get(k) is v for k,v in FLAGS.items()),'Complete explicit mixed-source ACT diagnostic run required')
    config=validate_config(value.get('schedule',{}));inputs=value.get('inputs',{})
    need(_version(value)==_version(config),'Completed model/config version mismatch; head cannot relabel legacy weights')
    need(runtime.read_bound(inputs['config'])==config and inputs.get('dataset')==config['dataset'] and
         inputs.get('auxiliary')==config['auxiliary'] and value.get('protocol')==training_protocol(config) and
         value.get('completed_steps')==config['steps'],'Completed exact mixed-source inputs/schedule/protocol required')
    if _version(value)==4:_validate_initialization(config,inputs,value.get('initialization'))
    frozen=runtime.read_bound(config['evaluation_cohort']);selected=value.get('selected_step')
    need(value.get('optimization_scope')==frozen,'Actual new scope differs from pretraining freeze')
    native=value.get('native_scope',{});actual=value.get('scope',{})
    need(set(native)==set(actual)=={'train','validation'} and
         set(native['train'].get('subject_ids',[]))==set(binding.TRAIN_IDS) and
         native['validation'].get('subject_ids')==['p000923'] and
         actual['train'].get('subject_ids')==frozen['optimized_train_ids'] and actual['validation']==native['validation'] and
         actual['train'].get('native_subject_ids')==native['train']['subject_ids'] and
         actual['train'].get('native_episode_count')==native['train']['episode_count'] and
         actual['train'].get('native_window_count')==NATIVE_COUNTS['train'] and
         all(actual[s].get('window_count')==native[s].get('window_count')==NATIVE_COUNTS[s] and
             actual[s].get('episode_ids')==native[s].get('episode_ids') and native[s].get('episode_ids') for s in native),
         'Actual optimization identities and unchanged separately typed native evaluation population required')
    rows=value.get('evaluations',[])
    need([row.get('step') for row in rows]==expanded.evaluation_steps(config) and selected==expanded.selected_step(rows),
         'Earliest minimum complete native raw masked selection-VAL MAE required')
    for split,count in NATIVE_COUNTS.items():
        for row in rows:need(runtime.validate_metrics(row[split])['sample_count']==count,'Complete native evaluation population differs')
        need(runtime.validate_metrics(value['selected_metrics'][split])==next(row[split] for row in rows if row['step']==selected),
             'Selected native metrics differ from full selected evaluation')
    inventory=contribution_inventory(config) if _version(value)>=3 else None
    _validate_draws(value.get('optimization'),config['steps'],frozen,native['train']['subject_ids'],version=_version(value),inventory=inventory)
    _validate_draws(value.get('selected_optimization'),selected,frozen,native['train']['subject_ids'],version=_version(value),inventory=inventory)
    counts=value.get('auxiliary_counts',{})
    if _version(value)==4:
        need(counts==auxiliary_metadata(config['auxiliary'],version=4)['counts'], 'Actual V4 parent/new contribution counts differ')
        computational=[counts.get('groups',{}).get(name,{}) for name in ('planned','conditional','oracle')]
    else:
        need(set(counts)==set(_kinds(_version(value))),'Separate computational kinds required')
        computational=list(counts.values())
    need(all(row.get('samples',0)>0 and row.get('accepted_sources',0)>0 for row in computational) and
         value.get('oracle_authentication') and all(r.get('actual_history_authenticated') is True and
         r.get('future_measurements_claimed') is False for r in value['oracle_authentication']),
         'Separate computational counts and actual pre-query history authentication required')
    for key in ('best','cpu_equivalence'):expanded.full_reference(value.get(key))
    return value


def model_binding(ref):
    expanded.full_reference(ref);complete=validate_complete(runtime.read_bound(ref));runtime.bound_path(complete['best'])
    pilot_schema,selection_schema,cpu_schema,_=_schemas(complete)
    cpu=runtime.read_bound(complete['cpu_equivalence'])
    need(cpu.get('schema')==cpu_schema and cpu.get('passed') is True and cpu.get('checkpoint')==complete['best'] and
         cpu.get('selected_step')==complete['selected_step'] and cpu.get('scope')==complete['scope'] and
         cpu.get('optimization_scope')==complete['optimization_scope'] and all(cpu.get(k) is v for k,v in FLAGS.items()) and
         set(cpu.get('comparisons',{}))=={'train','validation'} and all(row.get('passed') is True and
             row.get('prediction_rtol')==1e-5 and row.get('prediction_atol')==1e-6 and
             row.get('metric_rtol')==1e-5 and row.get('metric_atol')==1e-7 for row in cpu['comparisons'].values()),
         'Actual selected mixed-source CPU equivalence under original tolerances required')
    selection=runtime.read_bound(cpu['selection'])
    need(selection.get('schema')==selection_schema and all(selection.get(k)==complete.get(k) for k in
         ('inputs','best','selected_step','scope','native_scope','optimization_scope','protocol','window_order',
          'selected_metrics','selected_predictions','selected_optimization','optimization','auxiliary_counts','oracle_authentication')),
         'Mixed-source CPU receipt must bind the exact selected provenance and population')
    if _version(complete)==4:
        need(selection.get('initialization')==complete['initialization']==cpu.get('initialization'),
            'V4 CPU/selected/completed initialization bindings differ')
    validate_code(complete['inputs']['code'],version=_version(complete))
    return dict(checkpoint=deepcopy(complete['best']),complete=deepcopy(ref),checkpoint_schema=pilot_schema),complete


def verify_selected(selection_ref,receipt_path):
    selection=runtime.read_bound(selection_ref);prepared=load_inputs(runtime.read_bound(selection['inputs']['config']))
    need(all(prepared[k]==selection[k] for k in ('scope','native_scope','optimization_scope','window_order')),
         'Selected mixed-source native scope/window ordering changed')
    torch,policy,packet=load_selected_policy(selection)
    execution=runtime.execution_settings(torch,policy,device='cpu');comparisons={};metrics={}
    for split in ('train','validation'):
        cpu=runtime.infer(torch,policy,prepared[split],packet['normalization'],device='cpu')
        cuda=runtime.load_predictions(selection['selected_predictions'][split],window_order_sha256=prepared['window_order'][split])
        metrics[split]=runtime.metrics(cpu,prepared[split])
        comparisons[split]=runtime.compare_devices(cpu,cuda,metrics[split],selection['selected_metrics'][split])
    value=dict(schema=_schemas(selection)[2],passed=True,selection=selection_ref,
        checkpoint=selection['best'],selected_step=selection['selected_step'],scope=selection['scope'],
        optimization_scope=selection['optimization_scope'],comparisons=comparisons,metrics=metrics,execution=execution,**FLAGS)
    if _version(selection)==4:value['initialization']=deepcopy(selection['initialization'])
    return runtime.write_new(receipt_path,value)


def fit(config_ref,output):
    started=time.monotonic();config=validate_config(runtime.read_bound(config_ref));prepared=load_inputs(config)
    pilot_schema,selection_schema,_,_=_schemas(config)
    output=Path(output).resolve();need(not output.exists(),'Fresh mixed-source output required; no resume');output.mkdir(parents=True)
    torch=None;policy=None;optimizer=None;completed=0;inputs=dict(config=config_ref,dataset=config['dataset'],
        auxiliary=config['auxiliary'],code=code_binding(version=_version(config)));protocol=training_protocol(config)
    groups=_groups(_version(config))
    optimization=dict(gradient_examples={k:0 for k in groups},gradient_subject_examples={k:{} for k in groups},
        actual_optimized_train_ids=[])
    if _version(config)>=3:
        optimization.update(gradient_source_examples={k:{} for k in groups},gradient_kind_examples={k:{} for k in groups})
    try:
        lease=runtime.owned_gpu(config['replacement_pod_gpu_binding']) if 'replacement_pod_gpu_binding' in config else runtime.owned_gpu()
        with lease as ownership:
            identity=runtime._identity();need(identity['pid']==identity['pgid']==identity['sid'],'Original guard-owned mixed-source session required')
            torch,policy,software=make_model(config,device='cuda',training=True)
            validate_official_forward(software);execution=runtime.execution_settings(torch,policy,device='cuda')
            need(sum(p.numel() for p in policy.parameters())==2658823,'Approved official ACT parameter count differs')
            if _version(config)==4:
                from . import m2_act_warm_start as warm_start
                inputs['initialization']=deepcopy(config['initialization'])
                initial=warm_start.initialize_policy(torch,policy,config['initialization'],normalization=prepared['normalization'],posture_head=config['posture_head'])
                inputs['initialization_equivalence']=runtime.write_new(output/'INITIALIZATION_EQUIVALENCE.json',initial)
                optimizer,optimizer_initialization=warm_start.fresh_optimizer(torch,policy,lr=config['lr'],weight_decay=config['weight_decay'])
                prepared['initialization']=dict(receipt=deepcopy(config['initialization']),equivalence=inputs['initialization_equivalence'],optimizer=optimizer_initialization)
            cache=build_cache(prepared,prepared['auxiliary'],torch=torch,device='cuda')
            inputs['software']=runtime.write_new(output/'software.json',software)
            runtime.write_new(output/'STARTED.json',dict(schema=pilot_schema,inputs=inputs,scope=prepared['scope'],
                optimization_scope=prepared['optimization_scope'],protocol=protocol,execution=execution,
                ownership=ownership,identity=identity,**FLAGS))
            rng=np.random.Generator(np.random.PCG64(config['seed']))
            if _version(config)!=4:optimizer=torch.optim.AdamW(policy.parameters(),lr=config['lr'],weight_decay=config['weight_decay'],betas=(.9,.999),eps=1e-8)
            evaluations=[];best_score=float('inf');last_metrics=None;seen=set()
            with (output/'metrics.jsonl').open('x',buffering=1) as log:
                for step in range(1,config['steps']+1):
                    need(time.monotonic()-started<config['time_limit_seconds'],'Finite mixed-source training deadline reached')
                    policy.train();ids=sample_cached_indices(rng,cache);batches=cached_batches(cache,ids)
                    optimizer.zero_grad(set_to_none=True);loss,losses=mixed_loss(policy,batches,version=_version(config))
                    need(bool(torch.isfinite(loss)),'Finite official mixed-source scalar ACT loss required')
                    loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),1.,error_if_nonfinite=True)
                    optimizer.step();completed=step
                    record_draws(optimization,cache,ids);seen=set(optimization['actual_optimized_train_ids'])
                    if step==1 or step%50==0:
                        row=dict(step=step,train_loss=float(loss.detach().cpu()),groups=losses,grad_norm=float(norm),
                            elapsed_wall_s=time.monotonic()-started)
                        if _version(config)>=2:row['posture_head_diagnostics']=posture_head.diagnostics(policy)
                        log.write(json.dumps(row,allow_nan=False)+'\n');print(json.dumps(row),flush=True)
                    if step%config['eval_every']:continue
                    row={'step':step}
                    for split in ('train','validation'):
                        prediction=runtime.infer(torch,policy,prepared[split],prepared['normalization'],device='cuda')
                        row[split]=runtime.metrics(prediction,prepared[split])
                    evaluations.append(row);last_metrics=row
                    log.write(json.dumps(row,allow_nan=False)+'\n');print(json.dumps(row),flush=True)
                    score=row['validation']['act']['full_chunk_mae_rad']
                    if score<best_score:
                        best_score=score;expanded.save_checkpoint(torch,output/'best_auxiliary_act.pt',
                            checkpoint(policy,prepared,inputs,step,row,protocol,optimization))
                scope.validate_train16_metadata(prepared['optimization_scope'],actual_train_ids=seen)
                last_packet=checkpoint(policy,prepared,inputs,completed,last_metrics,protocol,optimization)
                last_packet['optimizer_state_dict']=optimizer.state_dict()
                last=expanded.save_checkpoint(torch,output/'last_auxiliary_act.pt',last_packet)
            selected=expanded.selected_step(evaluations);best=runtime.reference(output/'best_auxiliary_act.pt')
            packet=torch.load(output/'best_auxiliary_act.pt',map_location='cpu',weights_only=True)
            need(packet['step']==selected,'Earliest minimum full raw masked VAL selected checkpoint differs')
            scope.validate_train16_metadata(prepared['optimization_scope'],actual_train_ids=packet['optimization']['actual_optimized_train_ids'])
            policy.load_state_dict(packet['state_dict'],strict=True);predictions={};selected_metrics={}
            for split in ('train','validation'):
                value=runtime.infer(torch,policy,prepared[split],prepared['normalization'],device='cuda')
                predictions[split]=runtime.save_predictions(output/('selected-'+split+'.npy'),value,window_order_sha256=prepared['window_order'][split])
                selected_metrics[split]=runtime.metrics(value,prepared[split])
            selection={k:deepcopy(packet[k]) for k in ('scope','native_scope','optimization_scope','auxiliary_counts','oracle_authentication')}
            if _version(config)==4:selection['initialization']=deepcopy(packet['initialization'])
            selection.update(schema=selection_schema,selected_step=selected,best=best,inputs=inputs,
                window_order=prepared['window_order'],selected_predictions=predictions,selected_metrics=selected_metrics,
                selected_optimization=packet['optimization'],optimization=optimization,protocol=protocol,**FLAGS)
            if _version(config)>=2:
                selection.update(posture_target_audit=packet['posture_target_audit'],
                    posture_head_diagnostics=posture_head.diagnostics(policy))
            selected_ref=runtime.write_new(output/'SELECTED.json',selection)
            environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
            subprocess.run([sys.executable,'-B','-m',expanded.MODULE,'--verify-selected',selected_ref['path'],
                '--receipt',str(output/'CPU_EQUIVALENCE.json')],check=True,env=environment,timeout=1800)
            cpu=runtime.reference(output/'CPU_EQUIVALENCE.json');need(runtime.read_bound(cpu).get('passed') is True,'Actual CPU equivalence required')
            complete=dict(selection,schema=pilot_schema,complete=True,completed_steps=completed,last=last,
                task_contract=runtime.task_contract(),model=runtime.shared.native_spec(),evaluations=evaluations,cpu_equivalence=cpu,
                execution=execution,ownership=ownership,identity=identity,schedule=deepcopy(config),
                elapsed_wall_s=time.monotonic()-started,training_log=runtime.reference(output/'metrics.jsonl'))
            runtime.write_new(output/'COMPLETE.json',complete);return complete
    except BaseException as error:
        partial=None
        if policy is not None:
            packet=checkpoint(policy,prepared,inputs,completed,None,protocol,optimization);packet.update(complete=False,error=repr(error))
            if optimizer is not None:packet['optimizer_state_dict']=optimizer.state_dict()
            partial=expanded.save_checkpoint(torch,output/'failed_auxiliary_act.pt',packet)
        (output/'traceback.txt').write_text(traceback.format_exc())
        runtime.write_new(output/'FAILED.json',dict(schema=pilot_schema,complete=False,completed_steps=completed,
            inputs=inputs,partial_checkpoint=partial,error=repr(error),traceback=runtime.reference(output/'traceback.txt'),**FLAGS))
        raise
