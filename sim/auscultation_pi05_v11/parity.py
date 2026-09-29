"""Exact B7300 step0 normalized-tensor/input equivalence, with two distinct input evidence classes.

Only run_step0 calls its caller-supplied predictor. Verification reconstructs
original TRAIN inputs without inference or an old full-scope dataset loader.
"""
import copy
import os
from pathlib import Path
import numpy as np
from sim.auscultation_pi05_v7 import coverage_pilot_evaluation as Q, training_data as O
from sim.auscultation_pi05_v8 import pair_admission as F
from . import training_data as D

SCHEMA='wam_pi05_v11_step0_parity_v1'
RECONSTRUCTION_SCHEMA='wam_pi05_v11_original120_input_reconstruction_v1'
INPUT_SCHEMA='wam_pi05_v11_step0_consumed_inputs_v1'
FAILURE_SCHEMA='wam_pi05_v11_step0_parity_failure_v1'
CANCEL_SCHEMA='wam_pi05_v11_step0_parity_cancellation_v1'
PAIR_SOURCE_SHA='95fa3bf8444d2b90df57c8afcb647fa93ed80b1c3d9b2aa378b43a57bb2999c9'
ROOT=Path('/home/william/wam-vla-v2')
need=D.need
file_sha=D.file_sha
write_new=Q.write_new


def _read(path,sha,artifacts):return D._read_pinned(path,sha,artifacts)
def _pin(path,sha,artifacts):return D._pin(path,sha,artifacts)
def _binding(path):return dict(path=str(D._regular(path)),sha256=file_sha(path))

def _cancel_path(path):return Path(path).with_suffix('.cancelled.json')
def _not_cancelled(path):
    p=_cancel_path(path);need(not p.exists() and not p.is_symlink(),'Step0 parity cancelled')

def _check_callback(cancelled):
    if cancelled():raise InterruptedError('Step0 parity cancelled')


def _context(dataset,binding):
    from . import train as T
    identity=D.dataset_binding(dataset)
    need(type(binding) is dict and binding.get('schema')=='wam_pi05_v11_bounded_checkpoint_binding_v1','V11 checkpoint binding required')
    need(all(type(binding.get(k)) is int and binding[k]==v for k,v in dict(local_step=0,optimizer_step=0,schedule_cursor=0,cumulative_trainable_update_age=7300).items()),'Exact step0 checkpoint required')
    need(binding['data_hash']==identity['data_hash'] and binding['normalization_sha256']==D.digest(dataset.normalization),'Step0 data/normalization differs')
    checkpoint=Path(binding['checkpoint_path']);run=checkpoint.parents[1]
    need(checkpoint==run/'model/step-000000000000' and checkpoint==checkpoint.resolve(),'Original step0 model path required')
    need(T.checkpoint_binding(checkpoint)==binding,'Current step0 weights/configuration differs')
    artifacts=dict(dataset.qualification_scope_binding['artifact_sha256'])
    for name,key in [('metadata.json','metadata_sha256'),('arrays.npz','arrays_sha256'),('COMPLETE.json','complete_sha256')]:_pin(checkpoint/name,binding[key],artifacts)
    metadata=_read(checkpoint/'metadata.json',binding['metadata_sha256'],artifacts);config=metadata['config']
    need(D.digest(config)==binding['config_sha256'] and config['dataset_binding']==identity and metadata['normalization']==dataset.normalization,'Checkpoint context differs')
    saved=_read(run/'configuration.json',file_sha(run/'configuration.json'),artifacts)
    need(saved['config']==config and saved['data_hash']==identity['data_hash'] and saved['normalization']==dataset.normalization,'Saved step0 configuration differs')
    _pin(run/'initialization.json',metadata['initialization_sha256'],artifacts)
    _pin(run/'training_schedule.json',config['schedule_binding']['sha256'],artifacts)
    design_binding=identity['design_binding'];need(design_binding['sha256']==D.DESIGN_SHA256,'Approved parity design required')
    design=_read(design_binding['path'],design_binding['sha256'],artifacts)
    need(config['parent_checkpoint_binding']==design['training']['parent_checkpoint_binding'],'Parent must be exact historical B7300')
    project=Path(__file__).resolve().parents[2]
    for name,sha in binding['source_sha256'].items():_pin(project/name,sha,artifacts)
    _pin(Path(__file__).resolve(),binding['source_sha256']['sim/auscultation_pi05_v11/parity.py'],artifacts)
    _pin(Path(F.__file__).resolve(),PAIR_SOURCE_SHA,artifacts)
    tokenpath=ROOT/'public-assets/paligemma_tokenizer.model';_pin(tokenpath,config['assets']['tokenizer_sha256'],artifacts)
    import sentencepiece
    tokenizer=sentencepiece.SentencePieceProcessor(model_file=str(tokenpath))
    D._recheck(artifacts)
    return dict(dataset_binding=identity,checkpoint_binding=copy.deepcopy(binding),parent_checkpoint_binding=design['training']['parent_checkpoint_binding'],design=design,artifact_sha256=artifacts,tokenizer=tokenizer)


def _history_paths(parent):
    return (Path(parent['checkpoint_path']).parents[1]/'diagnostics/local_007300/report.json',
            ROOT/'diagnostics/horizon_fixed160_B_selected7300_v1/raw/receipt.json')


def _origins(dataset):
    view=D.original_train_view(dataset)
    rows120=Q.build_diagnostic_schedule(view);rows160=F.fixed_rows(view)
    mapping={(e['episode_id'],start):i for i,(ei,start) in enumerate(dataset.indices['train']) for e in [dataset.episodes[ei]]}
    need(len(mapping)==len(dataset.indices['train']),'Unique qualified TRAIN origins required')
    refs120,anchors120=Q.references(view,rows120);refs160,anchors160=F.fixed_references(view,rows160)
    return view,rows120,rows160,mapping,dict(original120_reference=refs120,original120_anchors=anchors120,fixed160_reference=refs160,fixed160_anchors=anchors160)


def _reference_input(dataset,row,index,tokenizer):
    ei,start=dataset.indices['train'][index];episode=dataset.episodes[ei]
    need(episode['split']=='train' and episode['episode_id']==row['episode_id'] and start==row['start'],'Original TRAIN-only reconstruction required')
    need(str(episode['phases'][start])==row['phase'] and str(episode['targets'][start])==row['target'],'Original observation descriptor differs')
    obs=O.observation(dataset.rgb(ei)[start],episode['state'][start],row['phase'],row['target'],tokenizer,dataset.normalization)
    return F.input_contract(O.stack_observations([obs]))


def _npz(path,sha,artifacts):
    p=_pin(path,sha,artifacts)
    with np.load(p,allow_pickle=False) as archive:result={k:archive[k] for k in archive.files}
    _pin(p,sha,artifacts);return result


def _guard_markers(history):
    p=history['guard_marker'];need(not p.exists() and not p.is_symlink(),'Historical fixed160 guard cancellation recorded')


def _history(dataset,context):
    artifacts=dict(context['artifact_sha256']);parent=context['parent_checkpoint_binding'];oldb=context['design']['data']['qualified_parent_dataset_binding']['design_binding'];old_design=_read(oldb['path'],oldb['sha256'],artifacts);classes=old_design['warm_start']['step0_evidence_classes']
    reportpath,fixedpath=_history_paths(parent)
    report=_read(reportpath,classes['original120']['historical_output_reference']['report_sha256'],artifacts)
    fixed=_read(fixedpath,classes['fixed160']['historical_input_receipt_binding']['sha256'],artifacts)
    view,rows120,rows160,mapping,references=_origins(dataset)
    expected=dict(schema='wam_horizon_pair_train_numeric_report_v1',checkpoint_binding=parent,mode=D.parent_D.codec.B,local_step=7300,optimizer_step=7300,sampler=Q.SAMPLER,rows=rows120,diagnostic_schedule_sha256=Q.digest(rows120),train_predictions_only=True,heldout_predictions_generated=False,closed_loop_validated=False,initialization_baseline_only=False)
    need(all(report.get(k)==v for k,v in expected.items()),'Historical120 identity/rows/sampler differs')
    need(report['numeric_sha256']==classes['original120']['historical_output_reference']['numeric_sha256_from_report'],'Historical120 numeric identity differs')
    a120=_npz(reportpath.with_name('predictions.npz'),report['numeric_sha256'],artifacts)
    need(set(a120)=={'normalized_predictions','predictions','reference','anchors'},'Historical120 numeric set differs')
    expected=dict(schema=F.FIXED_SCHEMA,role='B_candidate',mode=D.parent_D.codec.B,checkpoint_binding=parent,sampler=F.SAMPLER,rows_sha256=Q.digest(rows160),status='finite',errors=[],replay_byte_comparison=True,canonical_codec_decode_comparison=True,frozen120_first_free_overlap_comparison=True)
    need(all(fixed.get(k)==v for k,v in expected.items()),'Historical fixed160 identity/status differs')
    pb=fixed['plan_binding'];plan=_read(pb['path'],pb['sha256'],artifacts)
    need(plan.get('schema')==F.PLAN_SCHEMA and plan.get('b_checkpoint_binding')==parent and plan.get('fixed160_rows')==rows160 and plan.get('fixed160_rows_sha256')==Q.digest(rows160) and plan.get('sampler')==F.SAMPLER,'Historical fixed160 plan differs')
    nb=fixed['numeric_file_binding'];need(Path(nb['path'])==fixedpath.with_name('numeric.npz'),'Historical numeric sibling required')
    a160=_npz(nb['path'],nb['sha256'],artifacts)
    need(fixed['numeric_array_contracts']=={k:F.array_contract(v) for k,v in sorted(a160.items())},'Historical numeric array contracts differ')
    need(fixed['reference_array_binding']==F.array_contract(a160['reference']) and fixed['anchors_array_binding']==F.array_contract(a160['anchors']),'Historical fixed160 reference binding differs')
    for family,arrays in [('original120',a120),('fixed160',a160)]:
        need(F.same(arrays['reference'],references[family+'_reference']) and F.same(arrays['anchors'],references[family+'_anchors']),'Historical reference/anchors differ')
    need(a120['normalized_predictions'].shape==a120['predictions'].shape==(120,10,7),'Complete120 output required')
    for i,row in enumerate(rows120):
        z=a120['normalized_predictions'][i];raw=a120['predictions'][i]
        need(z.dtype==raw.dtype==np.float32 and np.isfinite(z).all() and np.isfinite(raw).all(),'Finite float32 historical120 required')
        q=view.episodes[row['episode_index']]['state'][row['start']]
        need(F.same(raw,D.parent_D.decode_prediction(dataset.parent,z,q,row['phase'])),'Historical120 canonical decode differs')
    inputs120=[_reference_input(dataset,r,mapping[r['episode_id'],r['start']],context['tokenizer']) for r in rows120]
    inputs160=[_reference_input(dataset,r,mapping[r['episode_id'],r['start']],context['tokenizer']) for r in rows160]
    statuses=fixed['per_row_pass_and_replay_status'];inputs=fixed['actual_consumed_input_leaf_hashes']
    need(len(statuses)==len(inputs)==320,'All320 historical input/status records required')
    expected_keys={'reference','anchors'}
    for i,row in enumerate(rows160):
        q=view.episodes[row['episode_index']]['state'][row['start']]
        for pi,label in enumerate(('pass','replay')):
            need(statuses[160*pi+i]==dict(pass_name=label,row_id=i,status='finite'),'Historical finite pass/replay status differs')
            need(inputs[160*pi+i]==inputs160[i],'Historical fixed160 input consumption differs')
            nk=f'{label}_{i:03d}_normalized';rk=f'{label}_{i:03d}_raw';expected_keys.update((nk,rk))
            need(nk in a160 and rk in a160,'Historical numeric row missing')
            z=a160[nk];raw=a160[rk]
            need(z.shape==raw.shape==(10,7) and z.dtype==raw.dtype==np.float32 and np.isfinite(z).all() and np.isfinite(raw).all(),'Historical finite float32 row required')
            need(F.same(raw,D.parent_D.decode_prediction(dataset.parent,z,q,'FREE')),'Historical fixed160 canonical decode differs')
        need(F.same(a160[f'pass_{i:03d}_normalized'],a160[f'replay_{i:03d}_normalized']) and F.same(a160[f'pass_{i:03d}_raw'],a160[f'replay_{i:03d}_raw']),'Historical replay bytes differ')
    need(set(a160)==expected_keys,'Historical numeric denominator differs')
    overlap=[]
    for i,row in enumerate(rows160):
        if row['ordinal']!=0:continue
        matches=[j for j,r in enumerate(rows120) if r['episode_id']==row['episode_id'] and r['start']==row['start'] and r['phase']=='FREE' and r['position']=='first']
        need(len(matches)==1,'Explicit eight-row overlap missing');j=matches[0]
        need(inputs120[j]==inputs160[i] and F.same(a120['normalized_predictions'][j],a160[f'pass_{i:03d}_normalized']) and F.same(a120['predictions'][j],a160[f'pass_{i:03d}_raw']),'Historical overlap differs')
        overlap.append(dict(episode_id=row['episode_id'],start=row['start'],original120_row=j,fixed160_row=i,historical_fixed160_records=[i,160+i]))
    need(len(overlap)==8,'Exactly eight overlap rows required')
    executionpath=fixedpath.with_name('execution.json');execution=_read(executionpath,file_sha(executionpath),artifacts)
    for key in ('guard_evidence_binding','guard_source_binding'):_pin(execution[key]['path'],execution[key]['sha256'],artifacts)
    source=fixed['producer_execution_binding']['source_file_binding'];_pin(source['path'],source['sha256'],artifacts);need(source['sha256']==PAIR_SOURCE_SHA,'Historical producer source differs')
    need(F.verify_execution(fixedpath,fixed['producer_execution_binding'])==_binding(executionpath),'Actual historical fixed160 execution required')
    gp=Path(execution['guard_evidence_binding']['path']);need(gp.name.endswith('.process.json'),'Original guard path required')
    result=dict(rows={'original120':rows120,'fixed160':rows160},inputs={'original120':inputs120,'fixed160':inputs160},mapping=mapping,references=references,arrays={'original120':a120,'fixed160':a160},overlap=overlap,artifact_sha256=artifacts,guard_marker=gp.with_name(gp.name[:-len('.process.json')]+'.late_interrupt.json'),historical_bindings=dict(original120_report=_binding(reportpath),original120_numeric=_binding(reportpath.with_name('predictions.npz')),fixed160_receipt=_binding(fixedpath),fixed160_numeric=_binding(nb['path']),fixed160_plan=pb,fixed160_execution=_binding(executionpath)))
    _guard_markers(result);D._recheck(artifacts);return result


def _record(family,i,row,index,contract):
    return dict(evidence_class=family,row_id=i,origin=row,v11_train_index=index,input_leaf_hashes=contract,status='finite',historical_fixed160_input_record_indices=[i,160+i] if family=='fixed160' else [],new_original120_reconstruction_row=i if family=='original120' else None)


def _reconstruction(context,history):
    return dict(schema=RECONSTRUCTION_SCHEMA,evidence_class='new_CPU_reconstruction_from_locked_original_TRAIN_and_original_preprocessing',historical_input_consumption_receipt=False,dataset_binding=context['dataset_binding'],parent_checkpoint_binding=context['parent_checkpoint_binding'],rows=[dict(origin=r,input_leaf_hashes=c) for r,c in zip(history['rows']['original120'],history['inputs']['original120'])])


def _expected_inputs(history):
    return dict(schema=INPUT_SCHEMA,model_batch_size=1,model_call_count=280,rows=[_record(family,i,r,history['mapping'][r['episode_id'],r['start']],history['inputs'][family][i]) for family in ('original120','fixed160') for i,r in enumerate(history['rows'][family])])


def _check_actual(dataset,history,numeric):
    expected=set(history['references'])
    for name,value in history['references'].items():need(name in numeric and F.same(numeric[name],value),'Retained original reference/anchor differs')
    for family in ('original120','fixed160'):
        hist=history['arrays'][family]
        for i,row in enumerate(history['rows'][family]):
            nk=f'{family}_{i:03d}_normalized';rk=f'{family}_{i:03d}_raw';expected.update((nk,rk))
            need(nk in numeric and rk in numeric,'Actual280 output denominator incomplete')
            z=numeric[nk];raw=numeric[rk]
            need(z.shape==(1,10,7) and z.dtype==np.float32 and np.isfinite(z).all(),'Actual finite float32 sampler output required')
            need(raw.shape==(10,7) and raw.dtype==np.float32 and np.isfinite(raw).all(),'Actual finite raw output required')
            ei,start=dataset.indices['train'][history['mapping'][row['episode_id'],row['start']]]
            need(F.same(raw,D.decode_prediction(dataset,z[0],dataset.episodes[ei]['state'][start],row['phase'],row['target'])),'Actual all70 canonical decode differs')
            if family=='original120':hz,hr=hist['normalized_predictions'][i],hist['predictions'][i]
            else:hz,hr=hist[f'pass_{i:03d}_normalized'],hist[f'pass_{i:03d}_raw']
            need(F.same(z[0],hz),'Actual normalized tensor differs from historical B7300')
    need(set(numeric)==expected,'Actual numeric set differs')


def _final_check(dataset,binding,context,history,path):
    _not_cancelled(path);_guard_markers(history);D._recheck(history['artifact_sha256'])
    fresh=_context(dataset,binding)
    need(fresh['dataset_binding']==context['dataset_binding'] and fresh['checkpoint_binding']==context['checkpoint_binding'] and fresh['artifact_sha256']==context['artifact_sha256'],'Step0 context changed during consumption')
    D._recheck(history['artifact_sha256']);_not_cancelled(path);_guard_markers(history)


def _receipt(context,history,folder,numeric,written):
    artifacts=dict(history['artifact_sha256'])
    for name in ('numeric.npz','consumed_inputs.json','original120_reconstruction.json'):_pin(folder/name,written[str(folder/name)],artifacts)
    return dict(schema=SCHEMA,passed=True,checkpoint_binding=context['checkpoint_binding'],dataset_binding=context['dataset_binding'],parent_checkpoint_binding=context['parent_checkpoint_binding'],sampler=Q.SAMPLER,model_call_count=280,original120_historical_input_consumption_receipt=False,historical_fixed160_input_records=320,original120_input_evidence_class=RECONSTRUCTION_SCHEMA,actual_input_evidence_class=INPUT_SCHEMA,historical_bindings=history['historical_bindings'],overlap=history['overlap'],numeric_array_contracts={k:F.array_contract(v) for k,v in sorted(numeric.items())},artifact_sha256=artifacts,qualified_input_scope='original8 TRAIN only; R5/L5 never predicted by parity',normalized_tensor_and_input_equivalence=True,decoded_action_equivalence_to_parent=False,bounded_coordinate_changed=True,scope='step0 normalized tensor/input equivalence only; new canonical decoded outputs retained; full training, selection, L5 and physics remain separate')


def _numeric_write(path,arrays):
    with Path(path).open('xb') as stream:np.savez(stream,**arrays);stream.flush();os.fsync(stream.fileno())


def run_step0(output,dataset,binding,tokenizer,predict,*,cancelled):
    """Run exactly120+160 supplied-model calls; immutable failure evidence on error."""
    path=Path(output);_check_callback(cancelled);_not_cancelled(path)
    need(path==Path(binding['checkpoint_path']).parents[1]/'step0_parity.json' and path==path.resolve(),'Canonical current-run parity output required')
    folder=path.with_suffix('');need(not path.exists() and not path.is_symlink() and not folder.exists() and not folder.is_symlink(),'Fresh parity output required')
    context=_context(dataset,binding);history=_history(dataset,context);_check_callback(cancelled);_not_cancelled(path)
    folder.mkdir();numeric={k:v.copy() for k,v in history['references'].items()};records=[]
    try:
        for family in ('original120','fixed160'):
            for i,row in enumerate(history['rows'][family]):
                _check_callback(cancelled);index=history['mapping'][row['episode_id'],row['start']]
                obs=D.make_observation_batch(dataset,'train',[index],tokenizer);contract=F.input_contract(obs)
                need(contract==history['inputs'][family][i],'Actual model input differs from original preprocessing/history')
                record=_record(family,i,row,index,contract);record['status']='started';records.append(record)
                try:returned=predict(obs)
                except BaseException as error:
                    record.update(status='exception',error=type(error).__name__+': '+str(error));raise
                if isinstance(returned,np.ndarray) and returned.dtype.kind in 'buif':numeric[f'{family}_{i:03d}_normalized']=returned.copy()
                need(F.input_contract(obs)==contract,'Model input mutated during consumption')
                need(isinstance(returned,np.ndarray) and returned.shape==(1,10,7) and returned.dtype==np.float32 and np.isfinite(returned).all(),'Actual finite float32 sampler output required')
                ei,start=dataset.indices['train'][index]
                raw=D.decode_prediction(dataset,returned[0],dataset.episodes[ei]['state'][start],row['phase'],row['target']);numeric[f'{family}_{i:03d}_raw']=raw.copy();record['status']='finite'
                hist=history['arrays'][family]
                hz,hr=(hist['normalized_predictions'][i],hist['predictions'][i]) if family=='original120' else (hist[f'pass_{i:03d}_normalized'],hist[f'pass_{i:03d}_raw'])
                need(F.same(returned[0],hz),'Actual normalized tensor differs from historical B7300')
                _check_callback(cancelled)
        consumed=dict(schema=INPUT_SCHEMA,model_batch_size=1,model_call_count=len(records),rows=records)
        need(consumed==_expected_inputs(history),'Actual consumed input denominator differs');_check_actual(dataset,history,numeric)
        _final_check(dataset,binding,context,history,path);_check_callback(cancelled)
        reconstruction=_reconstruction(context,history)
        _numeric_write(folder/'numeric.npz',numeric);write_new(folder/'consumed_inputs.json',consumed);write_new(folder/'original120_reconstruction.json',reconstruction)
        # Re-read the bytes actually retained, pinning JSON to the values written.
        # Never legitimize replacement evidence by capturing its new SHA later.
        written={}
        _read(folder/'consumed_inputs.json',D.digest(consumed),written)
        _read(folder/'original120_reconstruction.json',D.digest(reconstruction),written)
        retained=_npz(folder/'numeric.npz',file_sha(folder/'numeric.npz'),written)
        need(set(retained)==set(numeric) and all(F.same(retained[k],numeric[k]) for k in numeric),'Persisted actual numeric bytes differ')
        receipt=_receipt(context,history,folder,numeric,written)
        _final_check(dataset,binding,context,history,path);_check_callback(cancelled);D._recheck(receipt['artifact_sha256'])
        receipt_sha=D.digest(receipt)
        write_new(path,receipt)
        _check_callback(cancelled);_final_check(dataset,binding,context,history,path);D._recheck(receipt['artifact_sha256']);_check_callback(cancelled)
        _pin(path,receipt_sha,{})
        need(not (folder/'failure.json').exists() and not (folder/'failure.json').is_symlink(),'Late parity failure recorded')
        return receipt
    except BaseException as error:
        if isinstance(error,InterruptedError) or cancelled():
            cp=_cancel_path(path)
            if not cp.exists() and not cp.is_symlink():write_new(cp,dict(schema=CANCEL_SCHEMA,passed=False,error=type(error).__name__+': '+str(error)))
        _numeric_write(folder/'failed_numeric.npz',numeric)
        failure_path=folder/'failure.json'
        if failure_path.exists() or failure_path.is_symlink():failure_path=folder/'publication_failure.json'
        write_new(failure_path,dict(schema=FAILURE_SCHEMA,passed=False,error=type(error).__name__+': '+str(error),accounted_input_prefix=records,numeric_file_binding=_binding(folder/'failed_numeric.npz'),artifact_sha256=history['artifact_sha256'],scope='retained actual attempted prefix; no complete parity or policy acceptance'))
        raise


def verify_receipt(path,dataset,binding):
    """Read-only full recomputation; no predictor, old loader, or model import."""
    path=Path(path);_not_cancelled(path)
    need(path==Path(binding['checkpoint_path']).parents[1]/'step0_parity.json','Original run parity path required')
    identity=_binding(path);receipt=_read(path,identity['sha256'],{})
    need(receipt.get('schema')==SCHEMA and receipt.get('passed') is True,'Positive completed parity receipt required')
    folder=path.with_suffix('');need(folder==folder.resolve() and folder.is_dir() and not folder.is_symlink(),'Original parity evidence directory required')
    need(not (folder/'failure.json').exists() and not (folder/'failure.json').is_symlink(),'Failed parity cannot verify')
    context=_context(dataset,binding);history=_history(dataset,context);artifacts=receipt['artifact_sha256']
    D._recheck(artifacts)
    numeric=_npz(folder/'numeric.npz',artifacts[str(folder/'numeric.npz')],{})
    consumed=_read(folder/'consumed_inputs.json',artifacts[str(folder/'consumed_inputs.json')],{})
    reconstruction=_read(folder/'original120_reconstruction.json',artifacts[str(folder/'original120_reconstruction.json')],{})
    need(consumed==_expected_inputs(history),'Retained actual model input contracts differ')
    need(reconstruction==_reconstruction(context,history),'New120 CPU input reconstruction differs')
    _check_actual(dataset,history,numeric)
    need(receipt==_receipt(context,history,folder,numeric,artifacts),'Parity receipt/source/closure differs')
    _final_check(dataset,binding,context,history,path);D._recheck(artifacts);_pin(path,identity['sha256'],{});_not_cancelled(path)
    need(not (folder/'failure.json').exists() and not (folder/'failure.json').is_symlink(),'Late parity failure recorded')
    # The receipt cannot contain its own hash. Expose the first-read pin only
    # in this return value so enclosing T/E consumers retain the early closure.
    result=copy.deepcopy(receipt);result['receipt_binding']=identity
    need(str(path) not in result['artifact_sha256'],'Self-referential parity artifact forbidden')
    result['artifact_sha256'][str(path)]=identity['sha256']
    return result
