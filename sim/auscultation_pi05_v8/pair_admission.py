"""Post-lock paired TRAIN diagnostics. CPU verification never authorizes execution.

Only the explicit collect command initializes the original model. An external
owner must preserve the real child exit and process-group teardown evidence.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import numpy as np
from sim.auscultation_pi05_v6.action_contract import LOWER, UPPER
from sim.auscultation_pi05_v7.coverage_pilot_schedule import digest, file_sha, json_bytes
from . import action_encoding as C, training_data as D, evaluation as E, train as T

PLAN_SCHEMA='wam_horizon_pair_fixed160_plan_v1'
FIXED_SCHEMA='wam_horizon_pair_fixed160_receipt_v1'
ADMISSION_SCHEMA='wam_horizon_pair_fixed160_admission_v1'
PROCESS_EXIT_SCHEMA='wam_horizon_pair_fixed160_process_exit_v1'
PROCESS_EXIT_REQUIRED=('schema','process_identity','command','exit_code','process_group_drained','errors','deferred_signals','receipt_sha256','guard_evidence_binding','guard_source_binding')
GUARD_SHA256='48c47cf0d6d8f3d770028931a61d2882c5281684053a6149dacfcbe7b2483ff2'
ROLES=('A_comparator','B_candidate')
INVENTORY=T.ROOT/'cloud-runtime-v8/horizon-pair-20260928-v1.source_inventory.json'
INVENTORY_SHA='144d91be27011df7992fe7784e800b77a01eacab5f7c639837520def6a7c37fe'
SAMPLER=dict(E.SAMPLER,passes=2,key_rule='fresh key917 per row and repeat')
need=E.need
read=E.read


def regular(path):
    path=Path(path);need(path.is_file() and not path.is_symlink(),'Regular evidence file required: '+str(path));return path.resolve()


def binding(path):
    path=regular(path);return dict(path=str(path),sha256=file_sha(path))


def bound(value):
    p=regular(value['path']);need(file_sha(p)==value['sha256'],'Bound artifact changed');return p


def read_pinned(value):
    raw=regular(value['path']).read_bytes();need(hashlib.sha256(raw).hexdigest()==value['sha256'],'Pinned JSON changed before parsing');return json.loads(raw)


def add_artifact(result,path,expected_sha):
    path=regular(path);key=str(path)
    need(key not in result or result[key]==expected_sha,'Conflicting artifact hashes')
    need(file_sha(path)==expected_sha,'Consumed artifact changed');result[key]=expected_sha


def verified_json_binding(path,expected):
    path=regular(path);raw=path.read_bytes();need(json.loads(raw)==expected,'Verified JSON changed before artifact capture')
    value=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest());bound(value);return value


def source_binding():return {'sim/auscultation_pi05_v8/pair_admission.py':file_sha(regular(__file__))}


def diagnostic_artifact(value):
    path=bound(value);need(file_sha(path)==file_sha(regular(__file__)),'Diagnostic producer/verifier bytes differ');return value


def collection_command(plan,role,output):
    need(role in ROLES,'Exact collection role required')
    return [str(T.ROOT/'vendor/openpi/.venv/bin/python'),'-B','-m','sim.auscultation_pi05_v8.pair_admission',
            '--plan',str(Path(plan).resolve()),'--role',role,'--output',str(Path(output).resolve()),'--collect']


def array_contract(value):
    a=np.asarray(value);need(a.dtype.kind in 'biuf','Numeric array required')
    return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())


def same(a,b):return array_contract(a)==array_contract(b)


def input_contract(tree):
    result=[]
    def visit(value,path):
        if isinstance(value,dict):
            for k in sorted(value):need(isinstance(k,str),'Input keys must be strings');visit(value[k],path+[k])
        else:
            a=np.asarray(value);need(np.isfinite(a).all(),'Nonfinite model input');result.append(dict(path='/'.join(path),**array_contract(a)))
    visit(tree,[]);need(result,'Nonempty model inputs required');return result


def fixed_rows(dataset):
    # The canonical loader owns qualification; this additionally validates every
    # original train origin rather than trusting a caller-supplied first20 list.
    episodes=[(i,e) for i,e in enumerate(dataset.episodes) if e['split']=='train']
    ids=[e['episode_id'] for _,e in episodes]
    need(len(ids)==len(set(ids))==8 and ids==dataset.normalization['residual_actions']['fit_episode_ids'],'Original TRAIN fit order required')
    grouped={x:[] for x in ids};seen=set();targets={}
    for ti,pair in enumerate(dataset.indices['train']):
        need(len(pair)==2 and all(isinstance(x,(int,np.integer)) and not isinstance(x,(bool,np.bool_)) for x in pair),'Integer train origin required')
        ei,start=map(int,pair);need(0<=ei<len(dataset.episodes) and start>=0 and (ei,start) not in seen,'Invalid/duplicate train origin');seen.add((ei,start))
        e=dataset.episodes[ei];need(e['split']=='train' and e['episode_id'] in grouped,'Heldout train origin')
        phase=str(e['phases'][start]);target=str(e['targets'][start])
        need(target in E.parent.TARGETS and targets.setdefault(e['episode_id'],target)==target,'Original single target per episode required')
        need(len(e['actions'][start:start+10])==10 and all(str(x)==phase for x in e['phases'][start:start+10])
             and all(str(x)==target for x in e['targets'][start:start+10]),'Cross-phase/target or incomplete qualified origin')
        if phase=='FREE':grouped[e['episode_id']].append(dict(episode_id=e['episode_id'],episode_index=ei,train_index=ti,start=start,phase=phase,target=target))
    need(set(targets.values())==E.parent.TARGETS and len(targets)==8,'All original eight targets required');result=[]
    for eid in ids:
        rows=sorted(grouped[eid],key=lambda x:x['start']);need(len(rows)>=20,'Twenty qualified FREE starts per TRAIN episode required')
        for ordinal,row in enumerate(rows[:20]):result.append(dict(row,row_id=len(result),ordinal=ordinal))
    return result


def fixed_references(dataset,rows):
    need(rows==fixed_rows(dataset),'Fixed160 rows changed')
    ref=np.asarray([dataset.episodes[r['episode_index']]['actions'][r['start']:r['start']+10] for r in rows])
    anchors=np.asarray([dataset.episodes[r['episode_index']]['state'][r['start'],:7] for r in rows])
    need(ref.shape==(160,10,7) and anchors.shape==(160,7) and np.isfinite(ref).all() and np.isfinite(anchors).all(),'Finite original labels/anchors required')
    return ref,anchors


def runtime_inventory():
    p=regular(INVENTORY);need(file_sha(p)==INVENTORY_SHA,'Frozen343 inventory differs');value=read(p);root=Path(value['source_project'])
    need(len(value['source_files'])==343,'Exact343 source parent required')
    for n,h in value['source_files'].items():
        q=regular(root/n);need(q.is_relative_to(root) and file_sha(q)==h,'Frozen343 source changed')
    for name,h in T.source_hashes().items():
        relative='sim/auscultation_pi05_'+name
        need(value['source_files'].get(relative)==h,'Training24 differs from frozen343')
    return binding(p)


def _contexts(a_lock,b_lock):
    contexts={};locks={}
    for role,path,mode in zip(ROLES,(a_lock,b_lock),(C.A,C.B)):
        lock_binding=binding(path);lock=E.verify_selection_lock(bound(lock_binding),lock_binding['sha256']);need(lock['selection']['mode']==mode,'Wrong paired lock mode')
        ctx=T.verified_run_context(lock['selection']['run']);need(ctx['completed'] and ctx['dataset'].mode==mode,'Complete original arm required')
        bound(lock_binding);ctx['lock_binding']=lock_binding
        first=lock['selection']['candidates'][0]['checkpoint_binding']
        metadata=read_pinned(dict(path=str(Path(first['checkpoint_path'])/'metadata.json'),sha256=first['metadata_sha256']))
        ctx['initialization_binding']=dict(path=str(Path(ctx['config']['training_run_path'])/'initialization.json'),sha256=metadata['initialization_sha256']);bound(ctx['initialization_binding'])
        locks[role]=lock;contexts[role]=ctx
    validate_pair(contexts)
    return contexts,locks


def validate_pair(contexts):
    a,b=[contexts[k] for k in ROLES];ac,bc=a['config'],b['config']
    need(ac['mode']==C.A and bc['mode']==C.B,'Distinct matching A/B modes required')
    for k in ('source_sha256','assets','pair_spec','seed','learning_rate','batch_size','observation_contract','parent_data_hash'):
        need(ac[k]==bc[k],'Pair '+k+' differs')
    need(ac['seed']==61 and ac['batch_size']==4 and ac['learning_rate']==5e-5,'Original fresh settings required')
    need(a['schedule']==b['schedule'] and a['schedule']['total_steps']==14600 and a['schedule']['total_draws']==58400,'Identical complete training schedule required')
    need(tuple(a['schedule']['candidate_steps'])==E.EXPECTED_STEPS,'Thirteen candidates required')
    ai,ah=T._load_initialization(ac);bi,bh=T._load_initialization(bc)
    need(ai['numerical_initialization']==bi['numerical_initialization'] and bi['paired_a_numerics_equal'] is True,'Exact fresh numerical initialization differs')
    ap=Path(ac['training_run_path'])/'initialization.json'
    need(bc['paired_initialization']==binding(ap) and ah==file_sha(ap) and bi['paired_a_receipt_integrity']==ai['integrity_sha256'],'B must reference exact A initialization')
    ar,br=[T.records_for(x['dataset']) for x in (a,b)];need(ar==br,'Original TRAIN fit origins differ')
    need(fixed_rows(a['dataset'])==fixed_rows(b['dataset']),'A/B fixed160 origins differ')
    for x,y in zip(fixed_references(a['dataset'],fixed_rows(a['dataset'])),fixed_references(b['dataset'],fixed_rows(b['dataset']))):need(same(x,y),'A/B original labels/anchors differ')


def chosen_binding(selection,role):
    need(role in ROLES,'Exact checkpoint role required');candidate=selection['selected']
    if candidate is None:
        if role==ROLES[1]:return None
        matches=[x for x in selection['candidates'] if x['local_step']==14600];need(len(matches)==1,'Unique descriptive A final checkpoint required');candidate=matches[0]
    expected=candidate['checkpoint_binding'];need(T.checkpoint_binding(expected['checkpoint_path'])==expected,'Checkpoint changed after original selection lock verification');return expected


def _build_plan(a_lock,b_lock,diagnostic_source=None):
    contexts,locks=_contexts(a_lock,b_lock);a,b=[contexts[x] for x in ROLES];ac,bc=a['config'],b['config'];rows=fixed_rows(a['dataset'])
    ca=locks[ROLES[0]]['selection']['selected'];cb=locks[ROLES[1]]['selection']['selected']
    a_binding=chosen_binding(locks[ROLES[0]]['selection'],ROLES[0]);b_binding=chosen_binding(locks[ROLES[1]]['selection'],ROLES[1])
    diagnostic_source=binding(__file__) if diagnostic_source is None else diagnostic_artifact(diagnostic_source)
    plan=dict(schema=PLAN_SCHEMA,approved_experiment_binding=ac['pair_spec'],diagnostic_source_sha256=source_binding(),diagnostic_source_artifact_binding=diagnostic_source,frozen_runtime_inventory_binding=runtime_inventory(),
        a_selection_lock_binding=a['lock_binding'],b_selection_lock_binding=b['lock_binding'],a_run_path=ac['training_run_path'],b_run_path=bc['training_run_path'],
        a_dataset_binding=D.dataset_binding(a['dataset']),b_dataset_binding=D.dataset_binding(b['dataset']),
        a_initialization_binding=a['initialization_binding'],b_initialization_binding=b['initialization_binding'],
        numerical_initialization_equal=True,b_references_exact_a_initialization=True,training_schedule_bindings=[ac['schedule_binding'],bc['schedule_binding']],identical_schedule_content=True,
        candidate_steps=list(E.EXPECTED_STEPS),both_budgets_and_13_reports_complete=True,common_parent_data_hash=ac['parent_data_hash'],common_fit_windows_sha256=digest(T.records_for(a['dataset'])),
        common_training_source_sha256=ac['source_sha256'],a_comparator_role='locked_selected' if ca else 'final14600_descriptive_only',a_checkpoint_binding=a_binding,b_checkpoint_binding=b_binding,
        fixed160_rows=rows,fixed160_rows_sha256=digest(rows),sampler=SAMPLER,no_reselection=True,status='ready_for_owned_collection' if cb else 'rejected_no_eligible_B')
    return plan,contexts,locks


def _fresh(output,contexts,extra=()):
    output=Path(output).resolve();protected=[Path(__file__).resolve().parents[2],T.ROOT/'public-assets',INVENTORY.parent,Path(T.PAIR_SPEC_PATH).parent,*extra]
    for x in contexts.values():
        run=Path(x['config']['training_run_path']);protected.append(run)
        protected.append(Path(read(run/'configuration.json')['manifest']).resolve().parent)
    T.separate(output,protected);need(not output.exists(),'Fresh independent output required');output.mkdir(parents=True);return output


def prepare_pair(a_lock,b_lock,output):
    plan,contexts,_=_build_plan(a_lock,b_lock);out=_fresh(output,contexts);E.write_new(out/'plan.json',plan);return binding(out/'plan.json')


def _verify_plan(plan):
    identity=binding(plan);saved=read_pinned(identity);rebuilt,contexts,locks=_build_plan(bound(saved['a_selection_lock_binding']),bound(saved['b_selection_lock_binding']),saved['diagnostic_source_artifact_binding'])
    need(saved==rebuilt,'Plan differs from fresh paired reconstruction');bound(identity);return saved,contexts,locks


def _tokenizer(config):
    import sentencepiece
    path=T.ROOT/'public-assets/paligemma_tokenizer.model';need(file_sha(regular(path))==config['assets']['tokenizer_sha256'],'Tokenizer differs')
    return sentencepiece.SentencePieceProcessor(model_file=str(path))


def _canonical_inputs(dataset,rows,tokenizer):
    return [input_contract(D.make_batch(dataset,'train',[r['train_index']],tokenizer)[0]) for r in rows]


def _overlap(dataset,checkpoint,rows):
    p=E._report_path(checkpoint);E.verify_checkpoint_report(p,dataset,checkpoint);old=E.build_diagnostic_schedule(dataset)
    result={}
    with np.load(p.with_name('predictions.npz'),allow_pickle=False) as f:
        for i,r in enumerate(rows):
            if r['ordinal']!=0:continue
            matches=[j for j,v in enumerate(old) if v['episode_id']==r['episode_id'] and v['phase']=='FREE' and v['position']=='first' and v['start']==r['start']]
            need(len(matches)==1,'Frozen120 first FREE overlap missing');j=matches[0]
            result[i]={'normalized':f['normalized_predictions'][j],'raw':f['predictions'][j]}
    need(len(result)==8,'All eight frozen120 overlaps required');return result


def describe(prediction,reference,anchors,rows=None):
    p,ref,q=map(lambda x:np.asarray(x,np.float64),(prediction,reference,anchors));need(p.shape==ref.shape==(160,10,7) and q.shape==(160,7) and all(np.isfinite(v).all() for v in (p,ref,q)),'Finite fixed160 required')
    good=(p>=LOWER)&(p<=UPPER);bad=np.argwhere(~good);delta=p-ref;hold=np.broadcast_to(q[:,None,:],p.shape)
    margins=np.minimum(p-LOWER,UPPER-p)
    bounds=dict(coordinate_count=11200,violation_count=len(bad),all_passed=bool(good.all()),tolerance_rad=0.,first_pass_fraction=float(good[:,0].all(axis=1).mean()),full_pass_fraction=float(good.all(axis=(1,2)).mean()),
        minimum_margin_per_joint_rad=margins.min(axis=(0,1)).tolist(),violations=[dict(row=int(i),horizon=int(h),joint=int(j),value_rad=float(p[i,h,j]),lower_rad=float(LOWER[j]),upper_rad=float(UPPER[j])) for i,h,j in bad])
    metrics=dict(first_mae_rad=float(np.abs(delta[:,0]).mean()),full_mae_rad=float(np.abs(delta).mean()),signed_bias_per_joint_rad=delta.mean(axis=(0,1)).tolist(),
        hold_first_mae_rad=float(np.abs(hold[:,0]-ref[:,0]).mean()),hold_full_mae_rad=float(np.abs(hold-ref).mean()),worst_first_max_abs_rad=float(np.abs(delta[:,0]).max()))
    metrics['per_episode']=[]
    for ei in range(8):
        s=slice(ei*20,(ei+1)*20);metrics['per_episode'].append(dict(episode_ordinal=ei,episode_id=None if rows is None else rows[ei*20]['episode_id'],
            first_mae_rad=float(np.abs(delta[s,0]).mean()),full_mae_rad=float(np.abs(delta[s]).mean()),worst_first_max_abs_rad=float(np.abs(delta[s,0]).max()),
            signed_bias_per_joint_rad=delta[s].mean(axis=(0,1)).tolist(),hold_first_mae_rad=float(np.abs(hold[s,0]-ref[s,0]).mean()),hold_full_mae_rad=float(np.abs(hold[s]-ref[s]).mean())))
    return dict(bounds=bounds,metrics=metrics)


def verify_numeric(dataset,rows,arrays,statuses,inputs,canonical_inputs,overlap,role):
    need(role in ROLES and len(rows)==160 and len(statuses)==len(inputs)==320 and len(canonical_inputs)==160,'Complete two-pass denominator required')
    ref=np.asarray([dataset.episodes[r['episode_index']]['actions'][r['start']:r['start']+10] for r in rows]);q=np.asarray([dataset.episodes[r['episode_index']]['state'][r['start'],:7] for r in rows])
    need(same(arrays['reference'],ref) and same(arrays['anchors'],q),'Original labels/anchors byte mismatch')
    expected={'reference','anchors'};failures=[];raw=[]
    for i in range(160):
        a,b=statuses[i],statuses[160+i]
        need(a.get('pass_name')=='pass' and b.get('pass_name')=='replay' and a.get('row_id')==b.get('row_id')==i,'All attempts must be explicitly accounted in order')
        need({k:v for k,v in a.items() if k!='pass_name'}=={k:v for k,v in b.items() if k!='pass_name'},'Pass/replay status differs')
        need(inputs[i]==inputs[160+i]==canonical_inputs[i],'Actual model inputs differ from original canonical inputs')
        status=a['status'];need(status in ('finite','nonfinite_return','floating_point_error'),'Unaccounted/arbitrary model failure')
        if status=='floating_point_error':need(set(a)=={'pass_name','row_id','status','error'} and isinstance(a['error'],str) and a['error'],'Explicit numeric exception evidence required')
        else:
            for label in ('pass','replay'):
                key=f'{label}_{i:03d}_normalized';expected.add(key);need(key in arrays,'Missing retained model output');z=np.asarray(arrays[key]);need(z.shape==(10,7) and z.dtype.kind=='f','Malformed model output')
                need(bool(np.isfinite(z).all())==(status=='finite'),'False numeric status')
            need(same(arrays[f'pass_{i:03d}_normalized'],arrays[f'replay_{i:03d}_normalized']),'Full160 normalized replay differs')
        if status=='finite':
            for label in ('pass','replay'):
                key=f'{label}_{i:03d}_raw';expected.add(key);need(key in arrays,'Missing retained raw output')
                decoded=C.decode_chunk(arrays[f'{label}_{i:03d}_normalized'],q[i],'FREE',dataset.normalization['residual_actions'],mode=dataset.mode)
                need(same(arrays[key],decoded),'Raw output differs from exact original codec decode')
            need(same(arrays[f'pass_{i:03d}_raw'],arrays[f'replay_{i:03d}_raw']),'Full160 raw replay differs');raw.append(arrays[f'pass_{i:03d}_raw'])
        else:failures.append(i)
        if i in overlap:
            need(status=='finite' and same(arrays[f'pass_{i:03d}_normalized'],overlap[i]['normalized']) and same(arrays[f'pass_{i:03d}_raw'],overlap[i]['raw']),'Frozen120 overlap not reproduced')
    need(set(overlap)==set(range(0,160,20)) and set(arrays)==expected,'Missing/extra numeric artifact')
    result=dict(status='numerical_failure' if failures else 'finite',replay_byte_comparison=True,frozen120_first_free_overlap_comparison=True,canonical_codec_decode_comparison=True,numerical_failure_evidence=failures)
    result.update(dict(bounds=None,metrics=None) if failures else describe(np.asarray(raw),ref,q,rows));return result


def verify_execution(receipt,producer):
    p=Path(receipt).with_name('execution.json');need(p.is_file() and not p.is_symlink(),'Actual external process exit required');identity_binding=binding(p);value=read_pinned(identity_binding)
    need(all(k in value for k in PROCESS_EXIT_REQUIRED) and value['schema']==PROCESS_EXIT_SCHEMA,'External process schema differs')
    for k in ('process_identity','command'):need(value[k]==producer[k],'Producer identity/argv mismatch')
    identity=value['process_identity'];need(set(identity)=={'pid','start_ticks'} and all(type(identity[k]) is int and identity[k]>0 for k in identity),'Actual process identity required')
    need(type(value['exit_code']) is int and value['exit_code']==0 and value['process_group_drained'] is True and value['errors']==[] and value['deferred_signals']==[] and value['receipt_sha256']==file_sha(receipt),'Unsuccessful/interrupted or unbound producer exit')
    need(file_sha(bound(value['guard_source_binding']))==GUARD_SHA256,'Reviewed guard source differs')
    gp=bound(value['guard_evidence_binding']);need(gp.name.endswith('.process.json'),'Original guard process evidence required');guard=read_pinned(value['guard_evidence_binding'])
    late=gp.with_name(gp.name[:-len('.process.json')]+'.late_interrupt.json');need(not late.exists() and not late.is_symlink(),'Guard late interruption recorded')
    gi=guard['process_identity'];need(gi['pid']==identity['pid'] and int(gi['start_ticks'])==identity['start_ticks'] and gi['pgid']==gi['sid']==identity['pid'],'Guard-owned process session differs')
    need(guard['command']==value['command'] and guard['passed'] is True and type(guard['exit_code']) is int and guard['exit_code']==0 and guard['process_group_drained'] is True
         and guard['errors']==guard['deferred_signals']==guard['signals']==[],'Actual owned guard did not complete cleanly')
    bound(identity_binding);bound(value['guard_evidence_binding']);return identity_binding


def _environment():
    return dict(python_executable=str(Path(sys.executable).resolve()),python_version=sys.version,numpy_version=np.__version__,jax_version=importlib.metadata.version('jax'),openpi_python_prefix=str(Path(sys.prefix).resolve()))


def _producer():
    text=Path('/proc/self/stat').read_text();ticks=int(text[text.rfind(')')+2:].split()[19]);argv=Path('/proc/self/cmdline').read_bytes().rstrip(b'\0').split(b'\0')
    return dict(process_identity=dict(pid=os.getpid(),start_ticks=ticks),command=[x.decode() for x in argv],source_file_binding=binding(__file__))


def _record(plan_path,plan,ctx,role,output,statuses,inputs,arrays,producer,environment):
    ck=plan['a_checkpoint_binding' if role==ROLES[0] else 'b_checkpoint_binding'];d=ctx['dataset'];npz=output/'numeric.npz'
    with npz.open('xb') as f:np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
    canonical=_canonical_inputs(d,plan['fixed160_rows'],_tokenizer(ctx['config']))
    numeric=verify_numeric(d,plan['fixed160_rows'],arrays,statuses,inputs,canonical,_overlap(d,ck,plan['fixed160_rows']),role)
    record=dict(schema=FIXED_SCHEMA,role=role,mode=d.mode,plan_binding=binding(plan_path),checkpoint_binding=ck,dataset_binding=D.dataset_binding(d),
        selection_lock_binding=plan['a_selection_lock_binding' if role==ROLES[0] else 'b_selection_lock_binding'],sampler=SAMPLER,rows_sha256=plan['fixed160_rows_sha256'],
        reference_array_binding=array_contract(arrays['reference']),anchors_array_binding=array_contract(arrays['anchors']),actual_consumed_input_leaf_hashes=inputs,
        numeric_file_binding=binding(npz),numeric_array_contracts={k:array_contract(v) for k,v in sorted(arrays.items())},producer_execution_binding=producer,
        producer_source_sha256=source_binding(),runtime_inventory_binding=plan['frozen_runtime_inventory_binding'],official_openpi_commit=ctx['config']['openpi_commit'],
        official_assets_binding=ctx['config']['assets'],runtime_environment=environment,per_row_pass_and_replay_status=statuses,errors=[],training_rng_consumed_false=True,
        heldout_predictions_generated_false=True,closed_loop_validated_false=True,artifact_sha256={str(npz):file_sha(npz)},**numeric)
    E.write_new(output/'receipt.json',record);return binding(output/'receipt.json')


def collect_fixed160(plan,role,output):
    """GPU-owning external supervisor only; this function never manufactures exit evidence."""
    need(role in ROLES,'Exact fixed role required');p,contexts,_=_verify_plan(plan);need(p['b_checkpoint_binding'] is not None,'Negative B has no fallback or inference')
    ctx=contexts[role];ck=p['a_checkpoint_binding' if role==ROLES[0] else 'b_checkpoint_binding'];out=_fresh(output,contexts,[Path(plan).resolve().parent])
    producer=_producer();need(producer['command']==collection_command(plan,role,out) and producer['source_file_binding']==p['diagnostic_source_artifact_binding'],'Original explicit producer command/source required')
    E.write_new(out/'started.json',dict(plan_binding=binding(plan),role=role,producer=producer,source_sha256=source_binding()))
    arrays={};statuses=[];inputs=[]
    try:
        import jax
        from openpi.models import model as model_api
        from sim.auscultation_pi05_v7 import model as model
        runtime=model.initialize(61,5e-5);need(runtime['assets']==ctx['config']['assets'],'Official assets changed')
        model.restore(runtime,ck['checkpoint_path'],ctx['dataset'].data_hash,ctx['config'],ctx['dataset'].normalization);sample=model.sampler(runtime,steps=10)
        rows=p['fixed160_rows'];arrays['reference'],arrays['anchors']=fixed_references(ctx['dataset'],rows)
        for label in ('pass','replay'):
            for i,row in enumerate(rows):
                obs,_=D.make_batch(ctx['dataset'],'train',[row['train_index']],runtime['tokenizer']);inputs.append(input_contract(obs));status=dict(pass_name=label,row_id=i)
                try:value=np.asarray(sample(runtime['trainable'],runtime['frozen'],model_api.Observation(**jax.device_put(obs)),jax.random.key(917)))
                except FloatingPointError as error:status.update(status='floating_point_error',error=str(error)[:2048] or type(error).__name__)
                else:
                    need(value.shape==(1,10,7) and value.dtype.kind=='f','Malformed model output')
                    arrays[f'{label}_{i:03d}_normalized']=value[0]
                    status['status']='finite' if np.isfinite(value).all() else 'nonfinite_return'
                    if status['status']=='finite':arrays[f'{label}_{i:03d}_raw']=D.decode_prediction(ctx['dataset'],value[0],ctx['dataset'].episodes[row['episode_index']]['state'][row['start']],'FREE')
                statuses.append(status)
        need(_verify_plan(plan)[0]==p,'Pair evidence changed during collection')
        return _record(plan,p,ctx,role,out,statuses,inputs,arrays,producer,_environment())
    except BaseException as error:
        # Preserve actual partial outputs; no placeholder zeros or accepted receipt.
        with (out/'failed_numeric.npz').open('xb') as f:np.savez(f,**arrays)
        E.write_new(out/'failure.json',dict(error=type(error).__name__+': '+str(error),statuses=statuses,inputs=inputs,producer=producer,source_sha256=source_binding(),failed_numeric_sha256=file_sha(out/'failed_numeric.npz')))
        raise


def _verify_fixed(plan_path,p,contexts,receipt):
    identity=binding(receipt);path=regular(receipt);r=read_pinned(identity);role=r['role'];need(role in ROLES and p['b_checkpoint_binding'] is not None,'No eligible B or wrong role')
    ctx=contexts[role];d=ctx['dataset'];ck=p['a_checkpoint_binding' if role==ROLES[0] else 'b_checkpoint_binding']
    for k,v in dict(schema=FIXED_SCHEMA,mode=d.mode,plan_binding=verified_json_binding(plan_path,p),checkpoint_binding=ck,dataset_binding=D.dataset_binding(d),selection_lock_binding=p['a_selection_lock_binding' if role==ROLES[0] else 'b_selection_lock_binding'],sampler=SAMPLER,rows_sha256=p['fixed160_rows_sha256'],producer_source_sha256=source_binding(),runtime_inventory_binding=p['frozen_runtime_inventory_binding'],official_openpi_commit=ctx['config']['openpi_commit'],official_assets_binding=ctx['config']['assets'],errors=[],training_rng_consumed_false=True,heldout_predictions_generated_false=True,closed_loop_validated_false=True).items():need(r.get(k)==v,'Fixed160 '+k+' differs')
    env=r['runtime_environment'];need(env['numpy_version']==ctx['schedule']['numpy_version'] and env['openpi_python_prefix']==str((T.ROOT/'vendor/openpi/.venv').resolve()) and Path(env['python_executable']).is_absolute(),'Producer environment differs')
    producer=r['producer_execution_binding'];need(producer['command']==collection_command(plan_path,role,path.parent) and producer['source_file_binding']==p['diagnostic_source_artifact_binding'],'Original collection argv/source differs')
    diagnostic_artifact(producer['source_file_binding'])
    need(bound(r['numeric_file_binding'])==path.with_name('numeric.npz'),'Original numeric path required')
    with np.load(path.with_name('numeric.npz'),allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
    need(r['numeric_array_contracts']=={k:array_contract(v) for k,v in sorted(arrays.items())} and r['reference_array_binding']==array_contract(arrays['reference']) and r['anchors_array_binding']==array_contract(arrays['anchors']),'Array shape/dtype/bytes differ')
    result=verify_numeric(d,p['fixed160_rows'],arrays,r['per_row_pass_and_replay_status'],r['actual_consumed_input_leaf_hashes'],_canonical_inputs(d,p['fixed160_rows'],_tokenizer(ctx['config'])),_overlap(d,ck,p['fixed160_rows']),role)
    for k,v in result.items():need(r.get(k)==v,'Fixed160 numeric/metric reconstruction differs')
    need(r['artifact_sha256']=={str(path.with_name('numeric.npz')):file_sha(path.with_name('numeric.npz'))},'Artifact map differs')
    exit_binding=verify_execution(path,r['producer_execution_binding'])
    bound(identity);return dict(receipt=r,receipt_binding=identity,execution_binding=exit_binding)


def verify_fixed160(plan,receipt):
    p,contexts,_=_verify_plan(plan);return _verify_fixed(plan,p,contexts,receipt)


def research_comparison(a,b):
    result={}
    for k in ('worst_initial_first_max_abs_rad','macro_first_mae_rad','macro_full_mae_rad'):
        av,bv=float(a[k]),float(b[k]);need(np.isfinite(av) and np.isfinite(bv) and av>=0 and bv>=0,'Finite research scores required')
        result[k]=dict(a=av,b=bv,absolute_change=bv-av,ratio=None if av==0 else bv/av,relative_reduction=None if av==0 else 1-bv/av,status='zero_control' if av==0 else 'defined')
    result.update(criteria_passed=bool(result['worst_initial_first_max_abs_rad']['ratio'] is not None and result['worst_initial_first_max_abs_rad']['ratio']<=.75 and all(result[k]['ratio'] is not None and result[k]['ratio']<=1.05 for k in ('macro_first_mae_rad','macro_full_mae_rad'))),affects_admission=False)
    return result


def _scores(context,step):
    ck=T.checkpoint_binding(Path(context['config']['training_run_path'])/'model'/('step-%012d'%step));return E.verify_checkpoint_report(E._report_path(ck),context['dataset'],ck)['metrics']['scores']


def dataset_artifacts(manifest_binding,result):
    """Flatten the same contained qualification files consumed by the V6 loader."""
    manifest=read_pinned(manifest_binding);root=Path(manifest_binding['path']).parent
    def relative(path,sha):
        full=(root/path).resolve();need(full.is_relative_to(root),'Dataset evidence escapes qualified root');add_artifact(result,full,sha);return dict(path=str(full),sha256=sha)
    add_artifact(result,**dict(path=manifest_binding['path'],expected_sha=manifest_binding['sha256']))
    if 'protocol_sha256' in manifest:relative('protocol.json',manifest['protocol_sha256'])
    relative(manifest['dynamics_contract_path'],manifest['dynamics_contract_sha256'])
    for record in manifest['episodes']:
        for path_key,sha_key in (('path','sha256'),('dynamics_contract_path','dynamics_contract_sha256'),('outcome_path','outcome_sha256')):relative(record[path_key],record[sha_key])
        qualification=relative(record['qualification_path'],record['qualification_sha256']);report=read_pinned(qualification)
        task=report['evidence']['task_observation_derivation'];relative(task['parent_episode']['path'],task['parent_episode']['sha256'])
        for name,sha in task['input_sha256'].items():relative(str(Path(task['evidence_root'])/name),sha)
        audit=report['evidence']['joint_action_derivation']['phase_derivation']['independent_supervisor_audit'];relative(audit['path'],audit['sha256'])


def artifact_closure(plan_identity,p,contexts,locks,fixed):
    """Use already-verified hashes, reject conflicts, and recheck the full closure.

    Public base parameters retain the existing full initialize/load_inputs check;
    their verified receipt and tokenizer are included, not another 12GB scan.
    """
    result={}
    def add(value):add_artifact(result,value['path'],value['sha256'])
    add(plan_identity)
    for key in ('approved_experiment_binding','frozen_runtime_inventory_binding','diagnostic_source_artifact_binding','a_selection_lock_binding','b_selection_lock_binding','a_initialization_binding','b_initialization_binding'):add(p[key])
    inv=read_pinned(p['frozen_runtime_inventory_binding']);producer_root=Path(p['diagnostic_source_artifact_binding']['path']).parents[2]
    for name,sha in inv['source_files'].items():
        add_artifact(result,Path(inv['source_project'])/name,sha)
        add_artifact(result,producer_root/name,sha)
    for role in ROLES:
        ctx=contexts[role];lock=locks[role];selection=lock['selection'];run=Path(ctx['config']['training_run_path'])
        for path,sha in selection['artifact_sha256'].items():add_artifact(result,path,sha)
        add(dict(path=lock['selection_path'],sha256=lock['selection_sha256']))
        marker=run/E.MARKER;expected_marker=E._marker(bound(p['a_selection_lock_binding' if role==ROLES[0] else 'b_selection_lock_binding']))
        add(verified_json_binding(marker,expected_marker))
        candidates=selection['candidates'];need(tuple(c['local_step'] for c in candidates)==E.EXPECTED_STEPS,'Complete26 checkpoint closure required')
        for candidate in candidates:
            ck=candidate['checkpoint_binding'];directory=Path(ck['checkpoint_path'])
            for name,key in (('metadata.json','metadata_sha256'),('arrays.npz','arrays_sha256'),('COMPLETE.json','complete_sha256')):add_artifact(result,directory/name,ck[key])
        config_path=run/'configuration.json';config_binding=dict(path=str(config_path),sha256=selection['artifact_sha256'][str(config_path)])
        configuration=read_pinned(config_binding);need(configuration['config']==ctx['config'],'Configuration changed after verification')
        add(ctx['config']['schedule_binding']);dataset_artifacts(dict(path=configuration['manifest'],sha256=configuration['manifest_sha256']),result)
        add_artifact(result,T.ROOT/'public_checkpoint_verified.json',ctx['config']['assets']['checkpoint_receipt_sha256'])
        add_artifact(result,T.ROOT/'public-assets/paligemma_tokenizer.model',ctx['config']['assets']['tokenizer_sha256'])
    for own in fixed:
        if own is None:continue
        add(own['receipt_binding']);add(own['execution_binding']);execution=read_pinned(own['execution_binding'])
        add(execution['guard_evidence_binding']);add(execution['guard_source_binding'])
        add(own['receipt']['producer_execution_binding']['source_file_binding'])
        for path,sha in own['receipt']['artifact_sha256'].items():add_artifact(result,path,sha)
    for path,sha in result.items():need(file_sha(path)==sha,'Evidence changed during artifact closure verification')
    return result


def _final(plan_path,p,contexts,locks,a_receipt,b_receipt):
    plan_identity=verified_json_binding(plan_path,p)
    a,b=[contexts[x] for x in ROLES];primary=research_comparison(_scores(a,14600),_scores(b,14600));bsel=locks[ROLES[1]]['selection']['selected']
    secondary=None;af=bf=None;reasons=[]
    if bsel is None:
        need(a_receipt is None and b_receipt is None,'Negative B forbids fallback inference');status='rejected_no_eligible_B';reasons=[status]
    else:
        need(a_receipt is not None and b_receipt is not None,'Both real fixed160 outputs required')
        af=_verify_fixed(plan_path,p,contexts,a_receipt);bf=_verify_fixed(plan_path,p,contexts,b_receipt)
        need(af['receipt']['role']==ROLES[0] and bf['receipt']['role']==ROLES[1],'Exact A/B fixed160 roles required')
        need(af['receipt']['actual_consumed_input_leaf_hashes']==bf['receipt']['actual_consumed_input_leaf_hashes'],'Actual A/B model inputs differ')
        need(bsel['bounds']['all_passed'] and bsel['moving_buckets']['passed']==16,'B locked120 original eligibility required')
        secondary=research_comparison(_scores(a,p['a_checkpoint_binding']['local_step']),_scores(b,p['b_checkpoint_binding']['local_step']))
        if bf['receipt']['status']!='finite':reasons.append('rejected_B_numerical_output')
        elif not bf['receipt']['bounds']['all_passed']:reasons.append('rejected_B_fixed160_raw_bound')
        status=reasons[0] if reasons else 'simulator_candidate_only'
    artifacts=artifact_closure(plan_identity,p,contexts,locks,(af,bf))
    admitted=not reasons;d=b['dataset']
    return dict(schema=ADMISSION_SCHEMA,status=status,plan_binding=plan_identity,a_selection_lock_binding=p['a_selection_lock_binding'],b_selection_lock_binding=p['b_selection_lock_binding'],
        pair_prerequisites=dict(both_budgets_and_13_reports_complete=True,numerical_initialization_equal=True,identical_schedule_content=True),a_comparator_role=p['a_comparator_role'],
        a_fixed160_receipt_binding=None if af is None else af['receipt_binding'],b_fixed160_receipt_binding=None if bf is None else bf['receipt_binding'],
        a_control_outcome=None if af is None else dict(status=af['receipt']['status'],bounds=af['receipt']['bounds'],metrics=af['receipt']['metrics']),
        b_locked120_bounds_and_moving_gates=None if bsel is None else dict(bounds=bsel['bounds'],moving_buckets=bsel['moving_buckets']),b_fixed160_bounds=None if bf is None else bf['receipt']['bounds'],
        research_primary_final14600=primary,research_secondary_locked_or_fallback=secondary,fixed160_descriptive_comparison=None if af is None else dict(a=af['receipt']['metrics'],b=bf['receipt']['metrics']),
        candidate_admitted=admitted,service_eligible=admitted,admitted_checkpoint_binding=p['b_checkpoint_binding'] if admitted else None,
        admitted_dataset_codec_binding=dict(dataset_binding=D.dataset_binding(d),normalization_sha256=digest(d.normalization),action_encoding_binding=d.normalization['action_encoding_binding']) if admitted else None,
        no_reselection=True,rejection_reasons=reasons,authorization_granted_by_this_receipt_false=True,physical_accepted_false=True,closed_loop_validated_false=True,generalization_validated_false=True,artifact_sha256=artifacts)


def finalize_pair_admission(plan,a_receipt,b_receipt,output):
    p,contexts,locks=_verify_plan(plan);result=_final(plan,p,contexts,locks,a_receipt,b_receipt)
    extra=[Path(plan).resolve().parent]+[Path(x).resolve().parent for x in (a_receipt,b_receipt) if x is not None]
    out=_fresh(output,contexts,extra);path=out/'admission.json';E.write_new(path,result);path.chmod(0o444);return binding(path)


def verify_pair_admission(receipt,expected_sha256):
    path=regular(receipt);need(file_sha(path)==expected_sha256,'Admission SHA differs');r=read(path);plan=bound(r['plan_binding']);p,contexts,locks=_verify_plan(plan)
    a=None if r['a_fixed160_receipt_binding'] is None else bound(r['a_fixed160_receipt_binding']);b=None if r['b_fixed160_receipt_binding'] is None else bound(r['b_fixed160_receipt_binding'])
    expected=_final(plan,p,contexts,locks,a,b);need(r==expected and file_sha(path)==expected_sha256,'Admission differs from complete evidence reconstruction');return expected


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path);parser.add_argument('--a-lock',type=Path);parser.add_argument('--b-lock',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--role',choices=ROLES)
    parser.add_argument('--collect',action='store_true');parser.add_argument('--verify-only',action='store_true');parser.add_argument('--a-receipt',type=Path);parser.add_argument('--b-receipt',type=Path);parser.add_argument('--finalize',action='store_true')
    args=parser.parse_args()
    if args.collect:
        need(not args.verify_only and not args.finalize and args.plan and args.role and args.output,'Explicit collect arguments required');result=collect_fixed160(args.plan,args.role,args.output)
    elif args.finalize:
        need(not args.verify_only and args.plan and args.output,'Explicit finalize arguments required');result=finalize_pair_admission(args.plan,args.a_receipt,args.b_receipt,args.output)
    elif args.verify_only or args.output is None:
        if args.plan:p,_,_=_verify_plan(args.plan)
        else:need(args.a_lock and args.b_lock,'Two actual locks required');p,_,_=_build_plan(args.a_lock,args.b_lock)
        result=dict(verified=True,status=p['status'],model_initialized=False,output_changed=False)
    else:
        need(args.a_lock and args.b_lock,'Two actual locks required');result=prepare_pair(args.a_lock,args.b_lock,args.output)
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=='__main__':main()
