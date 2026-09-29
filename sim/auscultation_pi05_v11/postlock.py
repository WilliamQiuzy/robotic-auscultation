"""Selected V11 TRAIN replay and once-only L5; no physical acceptance.

Only --collect initializes the official model. External owned exit evidence is
written by the root guard supervisor, never by this worker itself.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import sys
import numpy as np
from sim.auscultation_pi05_v8 import pair_admission as F
from sim.auscultation_pi05_v11 import training_data as D, evaluation as E, train as T

SCHEMA='wam_pi05_v11_postlock_collection_v1'
ADMISSION_SCHEMA='wam_pi05_v11_bounded_admission_v1'
EXECUTION_SCHEMA='wam_pi05_v11_postlock_process_exit_v1'
RESERVATION_SCHEMA='wam_pi05_v11_bounded_final_test_reservation_v1'
GUARD_SHA='48c47cf0d6d8f3d770028931a61d2882c5281684053a6149dacfcbe7b2483ff2'
PAIR_SHA='95fa3bf8444d2b90df57c8afcb647fa93ed80b1c3d9b2aa378b43a57bb2999c9'
ROOT=T.ROOT
PRODUCER_FILE=ROOT/'diagnostics/pi05_v11_postlock_runtime_v1/postlock.py'
need=D.need
file_sha=D.file_sha
digest=D.digest
write_new=E.write_new


def binding(path):return dict(path=str(D._regular(path)),sha256=file_sha(path))
def read_binding(b,artifacts):
    need(type(b) is dict and set(b)=={'path','sha256'},'Exact path/SHA binding required')
    return D._read_pinned(b['path'],b['sha256'],artifacts)
def merge(artifacts,more):
    for p,h in more.items():D._pin(p,h,artifacts)
def key(pi,i,kind):return f'{pi}_{i:06d}_{kind}'
def source_binding():
    need(file_sha(F.__file__)==PAIR_SHA,'Frozen pure numeric/process helper changed')
    result={str(Path(__file__).resolve()):file_sha(__file__),str(Path(F.__file__).resolve()):PAIR_SHA}
    project=Path(T.__file__).resolve().parents[2]
    for name,h in T.source_hashes().items():D._pin(project/name,h,result)
    return result

def output_path(run,kind):
    need(kind in ('probes','test'),'Known collection kind required')
    return Path(run)/'postlock'/kind

def require_output(run,kind,output):
    path=Path(output);need(path.is_absolute() and path==path.resolve() and path==output_path(run,kind),'Fixed original output directory required')
    return path

def no_failure(folder):
    folder=Path(folder)
    for name in ('failure.json','publication_failure.json','cancelled.json'):
        p=folder/name;need(not p.exists() and not p.is_symlink(),'Failed/cancelled collection cannot be accepted')

def check_cancel(cancelled):
    if cancelled():raise InterruptedError('Postlock collection cancelled')

def runtime_predicates():
    repo=ROOT/'vendor/openpi'
    need(Path(sys.prefix).resolve()==(repo/'.venv').resolve(),'Original OpenPI interpreter required')
    commit=T.parent_module().verify_openpi_checkout(repo,repo/'src/openpi/models/pi0_config.py')
    return dict(python_prefix=str(Path(sys.prefix).resolve()),python_executable=str(Path(sys.executable).resolve()),numpy_version=np.__version__,openpi_commit=commit)

def noncontent_predicates(context):
    # Only same-operation evidence pins are reused; all absence/path/environment
    # predicates below remain fresh. No checkpoint/report science is cached.
    need(runtime_predicates()==context['runtime'],'Scientific interpreter/checkout changed')
    need(source_binding()==context['producer_sources'],'Scientific source identity changed')
    run=context['run'];a=context['artifact_sha256'];p=run/'training_run.json'
    old=D._read_pinned(p,a[str(p)],{})
    need(T.verify_terminal(run)==old,'Current training segment changed or cancelled')
    pointer=run/'TRAINING_OWNER.json';v=D._read_pinned(pointer,a[str(pointer)],{})
    life=v['lifecycle'];D._pin(life['path'],a[life['path']],{})
    from sim.auscultation_pi05_v11 import execution
    execution.no_cancel(Path(life['path']).parent)
    if 'dataset' in context:need(D.dataset_binding(context['dataset'])==context['dataset_binding'],'In-memory qualified inputs changed')

def _locked(lb):
    environment=runtime_predicates();initial_sources=source_binding();artifacts={};merge(artifacts,initial_sources)
    for name in ('public_checkpoint_verified.json','public_tokenizer_verified.json'):
        path=ROOT/name;D._read_pinned(path,file_sha(path),artifacts)
    old=read_binding(lb,artifacts);lock=E.verify_selection_lock(lb['path'],lb['sha256']);need(lock==old,'Selection changed')
    selection=lock['selection'];chosen=selection['selected'];need(type(chosen) is dict,'Positive selected checkpoint required; no fallback')
    cp=chosen['checkpoint_binding'];need(T.checkpoint_binding(cp['checkpoint_path'])==cp,'Selected checkpoint changed')
    run=Path(selection['run']);need(run==run.resolve(),'Original selected run required')
    merge(artifacts,selection['artifact_sha256'])
    for p in (run/E.MARKER,run/'selection/selection.json'):D._pin(p,file_sha(p),artifacts)
    configpath=run/'configuration.json';saved=D._read_pinned(configpath,artifacts[str(configpath)],artifacts);config=saved['config']
    need(digest(config)==cp['config_sha256'] and config['dataset_binding']==selection['dataset_binding'],'Selected configuration/data differs')
    # Public assets are verified by the original full context first. Retain every
    # consumed file hash (not a newly computed replacement hash) for end checks.
    public=ROOT/'public_checkpoint_verified.json'
    inventory=D._read_pinned(public,config['assets']['checkpoint_receipt_sha256'],artifacts)
    for record in inventory['files']:
        prefix='checkpoints/pi05_base/';need(record['name'].startswith(prefix),'Unexpected public asset')
        rel=record['name'][len(prefix):];need(not Path(rel).is_absolute() and '..' not in Path(rel).parts,'Relative public asset required')
        path=ROOT/'public-assets/pi05_base'/rel;D._pin(path,record['sha256'],artifacts);need(path.stat().st_size==record['bytes'],'Public asset size differs')
    tr=ROOT/'public_tokenizer_verified.json';tm=D._read_pinned(tr,file_sha(tr),artifacts)
    need(tm['sha256']==config['assets']['tokenizer_sha256'],'Tokenizer receipt differs')
    D._pin(ROOT/'public-assets/paligemma_tokenizer.model',tm['sha256'],artifacts)
    D._recheck(artifacts)
    result=dict(run=run,lock_binding=copy.deepcopy(lb),checkpoint_binding=cp,dataset_binding=selection['dataset_binding'],config=config,artifact_sha256=artifacts,runtime=environment,producer_sources=initial_sources)
    noncontent_predicates(result);return result

def _reservation(rb,info,artifacts):
    r=read_binding(rb,artifacts)
    expected=dict(schema=RESERVATION_SCHEMA,run=str(info['run']),selection_lock_binding=info['lock_binding'],selected_checkpoint_binding=info['checkpoint_binding'],dataset_binding=info['dataset_binding'],test_split='test',episode_id='L5_observed_armature02_v1',status='reserved')
    need(Path(rb['path'])==info['run']/'FINAL_TEST_STARTED.json' and r==expected,'Original exclusive test reservation differs')
    return r

def reserve_test(lock):
    return _reserve_test(_locked(lock))

def _reserve_test(info):
    artifacts=dict(info['artifact_sha256']);merge(artifacts,_passed_probes(info))
    run=info['run'];p=run/'FINAL_TEST_STARTED.json';need(not p.is_symlink(),'Existing test reservation')
    value=dict(schema=RESERVATION_SCHEMA,run=str(run),selection_lock_binding=info['lock_binding'],selected_checkpoint_binding=info['checkpoint_binding'],dataset_binding=info['dataset_binding'],test_split='test',episode_id='L5_observed_armature02_v1',status='reserved')
    D._recheck(artifacts);write_new(p,value)
    D._pin(p,digest(value),{});D._recheck(artifacts);_probe_end(info,artifacts)
    return dict(path=str(p),sha256=digest(value))

def _passed_probes(info):
    """CPU reconstruct already-consumed TRAIN evidence before any L5 attempt."""
    path=output_path(info['run'],'probes')/'receipt.json'
    p=_verify(path,'probes',locked=info)
    need(p['result']['passed'] is True,'L5 requires passing completed TRAIN probes')
    need(p['lock_binding']==info['lock_binding'] and p['checkpoint_binding']==info['checkpoint_binding'] and p['dataset_binding']==info['dataset_binding'],'Probes belong to another selected checkpoint/data')
    a=dict(p['artifact_sha256']);merge(a,verify_execution(path)['artifact_sha256'])
    D._recheck(a);no_failure(path.parent)
    return a


def _probe_end(info,artifacts):
    path=output_path(info['run'],'probes')/'receipt.json'
    merge(dict(artifacts),verify_execution(path)['artifact_sha256'])
    no_failure(path.parent);D._recheck(artifacts);no_failure(path.parent)


def _context(lb,kind,reservation=None,*,locked=None):
    info=_locked(lb) if locked is None else locked;need(info['lock_binding']==lb,'Operation lock differs');a=dict(info['artifact_sha256']);identity=info['dataset_binding']
    auth=None
    if kind=='test':
        merge(a,_passed_probes(info))
        _reservation(reservation,info,a);auth=dict(selection_lock_binding=lb,reservation_binding=reservation)
    split='train' if kind=='probes' else 'test'
    data=D.load_dataset(identity['manifest_binding']['path'],scopes=(split,),normalization_path=identity['normalization_file_binding']['path'],expected_normalization_sha256=identity['normalization_file_binding']['sha256'],test_authorization=auth)
    need(D.dataset_binding(data)==identity,'Qualified scoped identity differs');merge(a,data.qualification_scope_binding['artifact_sha256'])
    rows=E.build_probe_schedule(data) if kind=='probes' else E.build_report_schedule(data,'test')
    if kind=='probes':
        p=info['run']/'diagnostic_schedule.json';v=D._read_pinned(p,a[str(p)],a);need(v['probes']==rows,'Pre-training fixed probes changed')
    reference,anchors=E.references(data,rows);plans,posture=E.posture_contract();D._recheck(a)
    return dict(info,dataset=data,rows=rows,reference=reference,anchors=anchors,plans=plans,posture=posture,kind=kind,reservation_binding=reservation,artifact_sha256=a)

def collection_command(lock,kind,output):
    need(kind in ('probes','test'),'Known collection kind required')
    return [str(ROOT/'vendor/openpi/.venv/bin/python'),'-B',str(PRODUCER_FILE),'--lock',lock['path'],'--lock-sha256',lock['sha256'],'--kind',kind,'--output',str(output),'--collect']

def _producer(lb,kind,folder):
    need(Path(__file__).resolve()==PRODUCER_FILE,'Original external scientific worker required')
    raw=Path('/proc/self/stat').read_text();ticks=int(raw[raw.rfind(')')+2:].split()[19]);command=[s.decode() for s in Path('/proc/self/cmdline').read_bytes().rstrip(b'\0').split(b'\0')]
    need(command==collection_command(lb,kind,folder),'Only exact owned model CLI may collect actual evidence')
    return dict(process_identity=dict(pid=os.getpid(),start_ticks=ticks),command=command,source_binding=source_binding())

class SelectedPredictor:
    """Construct only for explicit worker collection; no test/verification imports JAX."""
    def __init__(self,info):
        import jax
        from openpi.models import model as model_api
        from sim.auscultation_pi05_v7 import model as M
        cp=info['checkpoint_binding'];c=info['config'];runtime=M.initialize(61,5e-5)
        need(runtime['assets']==c['assets'],'Public model assets differ')
        p=Path(cp['checkpoint_path']);m=D._read_pinned(p/'metadata.json',cp['metadata_sha256'],{})
        M.restore(runtime,p,cp['data_hash'],c,m['normalization'])
        need(T.checkpoint_binding(p)==cp,'Checkpoint changed while restoring selected model')
        self.checkpoint_binding=copy.deepcopy(cp);self.tokenizer=runtime['tokenizer'];self.source_binding=source_binding()
        self._runtime=runtime;self._sample=M.sampler(runtime,steps=10);self._jax=jax;self._model_api=model_api
    def __call__(self,obs):
        jax=self._jax;r=self._runtime
        return np.asarray(self._sample(r['trainable'],r['frozen'],self._model_api.Observation(**jax.device_put(obs)),jax.random.key(917)))

def require_predictor(predict,tokenizer,context):
    need(type(predict) is SelectedPredictor and tokenizer is predict.tokenizer,'Official selected model wrapper required')
    need(predict.checkpoint_binding==context['checkpoint_binding'] and predict.source_binding==source_binding(),'Predictor checkpoint/source changed')

def row_anchor(data,row):
    """Match the explicit row target before canonical full70 decode.

    Read only state/phase/target metadata; observation creation never encodes
    heldout reference actions. Qualified-reference support is checked by E.
    """
    ei=row['episode_index'];start=row['start']
    need(type(ei) is int and 0<=ei<len(data.episodes) and type(start) is int and start>=0,'Invalid row origin')
    e=data.episodes[ei]
    need(e['episode_id']==row['episode_id'] and e['split']==row['split'] and e['split'] in data.loaded_scopes,'Decoded origin outside qualified split')
    need(e['target']==row['target'] and len(e['targets'][start:start+10])==10 and all(str(t)==row['target'] for t in e['targets'][start:start+10]),'Decoded row target mismatch')
    need(len(e['phases'][start:start+10])==10 and all(str(p)==row['phase'] for p in e['phases'][start:start+10]),'Decoded row phase mismatch')
    q=np.asarray(e['state'][start]);need(q.shape==(41,) and np.isfinite(q).all(),'Original finite measured41 anchor required')
    return q


def summarize(data,rows,arrays,statuses,plans,passes):
    n=len(rows);need(passes in (1,2) and len(statuses)==n*passes,'Complete accounted denominator required')
    need(arrays['reference'].shape==(n,10,7) and arrays['anchors'].shape==(n,7),'Reference shapes differ')
    expected={'reference','anchors'};raw=[];bad=[]
    for pi in range(passes):
        values=[]
        for i,row in enumerate(rows):
            nk=key(pi,i,'normalized');rk=key(pi,i,'raw');expected.add(nk);need(nk in arrays,'Missing actual normalized output')
            z=arrays[nk];need(z.shape==(1,10,7) and z.dtype==np.float32,'Actual float32 full70 output required')
            status=statuses[pi*n+i];finite=bool(np.isfinite(z).all())
            if not finite:
                need(status=='nonfinite_return' and rk not in arrays,'Nonfinite raw cannot be invented');bad.append(i);values.append(None);continue
            expected.add(rk);need(rk in arrays,'Missing canonical raw output')
            q=row_anchor(data,row)
            decoded=D.decode_prediction(data,z[0],q,row['phase'],row['target']);p=arrays[rk]
            need(F.same(p,decoded) and p.shape==(10,7) and p.dtype==np.float32,'All70 original measured-anchor decode differs')
            need(status==('finite' if np.isfinite(p).all() else 'nonfinite_decoded'),'Numeric status differs')
            if status!='finite':bad.append(i)
            values.append(p)
        raw.append(values)
    need(set(arrays)==expected,'Unexpected or missing numeric output')
    replay=True
    if passes==2:
        replay=statuses[:n]==statuses[n:] and all(F.same(arrays[key(0,i,'normalized')],arrays[key(1,i,'normalized')]) and ((key(0,i,'raw') not in arrays and key(1,i,'raw') not in arrays) or (key(0,i,'raw') in arrays and key(1,i,'raw') in arrays and F.same(arrays[key(0,i,'raw')],arrays[key(1,i,'raw')]))) for i in range(n))
    # Unavailable raw coordinates are explicit nonfinite gate placeholders only;
    # they are never written as actual model outputs to the numeric archive.
    p_all=np.asarray([v if v is not None else np.full((10,7),np.nan,np.float32) for values in raw for v in values])
    gates=E.raw_gates(p_all,rows*passes,plans);metrics=None
    if not bad:metrics=E.phase_metrics(np.asarray(raw[0]),arrays['reference'],arrays['anchors'],rows)
    passed=not bad and replay and all(gates[k]['all_passed'] for k in ('urdf','posture'))
    coordinate=E.coordinate_diagnostics(data,p_all,rows*passes)
    return dict(passed=bool(passed),complete=True,rows=n,passes=passes,coordinate_count=n*70,total_coordinate_count=n*70*passes,replay_equal=bool(replay),nonfinite_rows=sorted(set(bad)),raw_gates=gates,metrics=metrics,coordinate_diagnostics=coordinate,bounded_by_codec=True,learned_safety_validated=False,clipping=False,physical_acceptance=False,population_generalization=False)

def numeric_write(path,values):
    with Path(path).open('xb') as f:np.savez(f,**values);f.flush();os.fsync(f.fileno())

def numeric_read(path,h):
    D._pin(path,h,{})
    with np.load(path,allow_pickle=False) as z:values={k:z[k] for k in z.files}
    D._pin(path,h,{});return values

def _claim(reservation,info,producer):
    a={};_reservation(reservation,info,a);p=info['run']/'FINAL_TEST_INFERENCE_STARTED.json'
    value=dict(schema='wam_pi05_v11_once_test_inference_v1',reservation_binding=reservation,producer=producer)
    write_new(p,value);D._pin(p,digest(value),a);return dict(path=str(p),sha256=digest(value))

def _fresh(context,folder,lb,kind,reservation,claim):
    no_failure(folder);D._recheck(context['artifact_sha256'])
    info=context;need(info['lock_binding']==lb,'Operation lock differs');noncontent_predicates(context)
    if kind=='test':
        _probe_end(context,context['artifact_sha256'])
        _reservation(reservation,info,{});need(Path(claim['path'])==info['run']/'FINAL_TEST_INFERENCE_STARTED.json','Original single-test claim path required');v=read_binding(claim,{})
        need(v['reservation_binding']==reservation,'Test claim changed')
    D._recheck(context['artifact_sha256']);noncontent_predicates(context);no_failure(folder)

def _record(context,folder,producer,claim,inputs,statuses,arrays,written,result):
    artifacts=dict(context['artifact_sha256']);merge(artifacts,written);merge(artifacts,producer['source_binding'])
    started=dict(kind=context['kind'],producer=producer,lock_binding=context['lock_binding'],test_claim_binding=claim)
    D._read_pinned(folder/'started.json',digest(started),artifacts)
    if claim:D._pin(claim['path'],claim['sha256'],artifacts)
    return dict(schema=SCHEMA,kind=context['kind'],lock_binding=context['lock_binding'],checkpoint_binding=context['checkpoint_binding'],dataset_binding=context['dataset_binding'],rows=context['rows'],rows_sha256=digest(context['rows']),sampler=E.SAMPLER,passes=2 if context['kind']=='probes' else 1,model_call_count=len(inputs),producer=producer,reservation_binding=context.get('reservation_binding'),test_claim_binding=claim,actual_inputs=inputs,statuses=statuses,result=result,numeric_array_contracts={k:F.array_contract(v) for k,v in sorted(arrays.items())},numeric_file_binding=dict(path=str(folder/'numeric.npz'),sha256=written[str(folder/'numeric.npz')]),artifact_sha256=artifacts)

def _collect(lb,kind,folder,tokenizer,predict,cancelled,reservation=None,*,locked=None):
    check_cancel(cancelled);info=_locked(lb) if locked is None else locked;need(info['lock_binding']==lb,'Operation lock differs');folder=require_output(info['run'],kind,folder)
    need(not folder.exists() and not folder.is_symlink(),'Collection already attempted')
    producer=_producer(lb,kind,folder);claim=None
    if kind=='test':claim=_claim(reservation,info,producer)
    folder.parent.mkdir(exist_ok=True);folder.mkdir();write_new(folder/'started.json',dict(kind=kind,producer=producer,lock_binding=lb,test_claim_binding=claim))
    arrays={};inputs=[];statuses=[];context=None
    try:
        context=_context(lb,kind,reservation,locked=info);require_predictor(predict,tokenizer,context)
        arrays=dict(reference=context['reference'].copy(),anchors=context['anchors'].copy());rows=context['rows'];passes=2 if kind=='probes' else 1
        for pi in range(passes):
            for i,row in enumerate(rows):
                check_cancel(cancelled);index=row['train_index'] if kind=='probes' else row['split_index']
                obs=D.make_observation_batch(context['dataset'],'train' if kind=='probes' else 'test',[index],tokenizer);contract=F.input_contract(obs)
                if pi:need(contract==inputs[i]['leaves'],'Replay input bytes changed')
                inputs.append(dict(pass_index=pi,row_index=i,leaves=contract));z=predict(obs)
                if isinstance(z,np.ndarray) and z.dtype.kind in 'bifu':arrays[key(pi,i,'normalized')]=z.copy()
                need(F.input_contract(obs)==contract,'Model mutated input during consumption')
                need(isinstance(z,np.ndarray) and z.shape==(1,10,7) and z.dtype==np.float32,'Actual selected sampler output malformed')
                if np.isfinite(z).all():
                    q=row_anchor(context['dataset'],row);raw=D.decode_prediction(context['dataset'],z[0],q,row['phase'],row['target']);arrays[key(pi,i,'raw')]=raw.copy()
                    statuses.append('finite' if np.isfinite(raw).all() else 'nonfinite_decoded')
                else:statuses.append('nonfinite_return')
                check_cancel(cancelled)
        result=summarize(context['dataset'],rows,arrays,statuses,context['plans'],passes)
        _fresh(context,folder,lb,kind,reservation,claim);check_cancel(cancelled)
        numeric_write(folder/'numeric.npz',arrays);nh=file_sha(folder/'numeric.npz');retained=numeric_read(folder/'numeric.npz',nh)
        need(set(retained)==set(arrays) and all(F.same(retained[k],arrays[k]) for k in arrays),'Persisted actual numeric changed')
        written={str(folder/'numeric.npz'):nh};record=_record(context,folder,producer,claim,inputs,statuses,arrays,written,result)
        h=digest(record);_fresh(context,folder,lb,kind,reservation,claim);D._recheck(record['artifact_sha256']);check_cancel(cancelled)
        write_new(folder/'receipt.json',record)
        _fresh(context,folder,lb,kind,reservation,claim);D._recheck(record['artifact_sha256']);D._pin(folder/'receipt.json',h,{});check_cancel(cancelled);no_failure(folder)
        return record
    except BaseException as error:
        if isinstance(error,InterruptedError) or cancelled():
            p=folder/'cancelled.json'
            if not p.exists() and not p.is_symlink():write_new(p,dict(cancelled=True,error=type(error).__name__+': '+str(error)))
        numeric_write(folder/'failed_numeric.npz',arrays)
        p=folder/'failure.json'
        if p.exists() or p.is_symlink():p=folder/'publication_failure.json'
        write_new(p,dict(schema='wam_pi05_v11_postlock_incomplete_v1',complete=False,kind=kind,producer=producer,attempted_inputs=inputs,statuses=statuses,error=type(error).__name__+': '+str(error),numeric_file_binding=binding(folder/'failed_numeric.npz')))
        raise

def collect_probes(lock,output,tokenizer,predict,*,cancelled):return _collect(lock,'probes',output,tokenizer,predict,cancelled)
def collect_test(reservation,output,tokenizer,predict,*,cancelled):
    r=read_binding(reservation,{})
    return _collect(r['selection_lock_binding'],'test',output,tokenizer,predict,cancelled,reservation)

def _verify(path,kind,*,locked=None):
    path=Path(path);rb=binding(path);a={};r=read_binding(rb,a);need(r.get('schema')==SCHEMA and r.get('kind')==kind,'Postlock receipt identity differs')
    context=_context(r['lock_binding'],kind,r['reservation_binding'],locked=locked);folder=require_output(context['run'],kind,path.parent);need(path.name=='receipt.json','Original receipt name required');no_failure(folder)
    merge(a,context['artifact_sha256']);merge(a,r['artifact_sha256'])
    claim=r['test_claim_binding'];producer=r['producer'];need(producer['source_binding']==source_binding(),'Actual producer source changed')
    need(producer['command']==collection_command(r['lock_binding'],kind,folder),'Actual selected worker command differs')
    if kind=='test':
        need(Path(claim['path'])==context['run']/'FINAL_TEST_INFERENCE_STARTED.json','Original test claim path required')
        need(read_binding(claim,a)==dict(schema='wam_pi05_v11_once_test_inference_v1',reservation_binding=r['reservation_binding'],producer=producer),'Original test inference claim differs')
    else:need(claim is None and r['reservation_binding'] is None,'TRAIN probe cannot consume test reservation')
    arrays=numeric_read(folder/'numeric.npz',r['numeric_file_binding']['sha256']);need(F.same(arrays['reference'],context['reference']) and F.same(arrays['anchors'],context['anchors']),'Actual reference/anchor changed')
    rows=context['rows'];inputs=[]
    # Tokenizer is CPU-only and initialized independently from the selected model.
    tokenizer=_tokenizer(context)
    for pi in range(2 if kind=='probes' else 1):
        for i,row in enumerate(rows):
            index=row['train_index'] if kind=='probes' else row['split_index'];obs=D.make_observation_batch(context['dataset'],'train' if kind=='probes' else 'test',[index],tokenizer)
            inputs.append(dict(pass_index=pi,row_index=i,leaves=F.input_contract(obs)))
    need(inputs==r['actual_inputs'],'Actual model input byte contracts differ')
    result=summarize(context['dataset'],rows,arrays,r['statuses'],context['plans'],2 if kind=='probes' else 1)
    expected=_record(context,folder,producer,claim,inputs,r['statuses'],arrays,{str(folder/'numeric.npz'):r['numeric_file_binding']['sha256']},result)
    need(expected==r,'Receipt scientific reconstruction differs');_fresh(context,folder,r['lock_binding'],kind,r['reservation_binding'],claim);D._recheck(a);no_failure(folder)
    return dict(r,receipt_binding=rb,artifact_sha256=dict(r['artifact_sha256'],**{rb['path']:rb['sha256']}))

def _tokenizer(context):
    import sentencepiece
    p=ROOT/'public-assets/paligemma_tokenizer.model';D._pin(p,context['config']['assets']['tokenizer_sha256'],{})
    return sentencepiece.SentencePieceProcessor(model_file=str(p))

def verify_probes(receipt):return _verify(receipt,'probes')
def verify_test(receipt):return _verify(receipt,'test')

def verify_execution(receipt):
    receipt=Path(receipt);artifacts={};rb=binding(receipt);r=read_binding(rb,artifacts);producer=r['producer']
    eb=binding(receipt.with_name('execution.json'));v=read_binding(eb,artifacts)
    need(v.get('schema')==EXECUTION_SCHEMA,'Actual external execution receipt required')
    need(v['process_identity']==producer['process_identity'] and v['command']==producer['command'] and v['receipt_sha256']==rb['sha256'],'Actual worker identity/command/output differs')
    identity=v['process_identity'];need(set(identity)=={'pid','start_ticks'} and all(type(x) is int and x>0 for x in identity.values()),'Actual Linux identity required')
    need(type(v['exit_code']) is int and v['exit_code']==0 and v['process_group_drained'] is True and v['errors']==v['deferred_signals']==[],'Owned worker must exit0/drain without cancellation')
    gs=v['guard_source_binding'];D._pin(gs['path'],gs['sha256'],artifacts);need(gs['sha256']==GUARD_SHA,'Original resource guard changed')
    gp=v['guard_evidence_binding'];guard=read_binding(gp,artifacts);g=guard['process_identity'];need(g['pid']==g['pgid']==g['sid']==identity['pid'] and str(g['start_ticks']).isascii() and str(g['start_ticks']).isdecimal() and int(g['start_ticks'])==identity['start_ticks'],'Actual owned guard session differs')
    need(guard['command']==v['command'] and type(guard['exit_code']) is int and guard['exit_code']==0 and guard['passed'] is True and guard['process_group_drained'] is True and guard['errors']==guard['signals']==guard['deferred_signals']==[],'Guard did not succeed cleanly')
    p=Path(gp['path']);need(p.name.endswith('.process.json'),'Original process receipt name required')
    late=p.with_name(p.name[:-len('.process.json')]+'.late_interrupt.json')
    def cancel():
        for q in (late,p.parent/'late_interrupt.json'):need(not q.exists() and not q.is_symlink(),'Guard final interruption recorded')
    cancel();D._recheck(artifacts);cancel();return dict(binding=eb,artifact_sha256=artifacts)

def _end_admission(value,info):
    a=value['artifact_sha256'];D._recheck(a);noncontent_predicates(info)
    for b in (value['probes_binding'],value['test_binding']):
        r=D._read_pinned(b['path'],b['sha256'],{})
        merge(dict(a),verify_execution(b['path'])['artifact_sha256']);no_failure(Path(b['path']).parent)
        if r['kind']=='test':
            _reservation(r['reservation_binding'],info,{})
            claim=r['test_claim_binding'];need(Path(claim['path'])==info['run']/'FINAL_TEST_INFERENCE_STARTED.json','Original test claim path required')
            need(read_binding(claim,{})==dict(schema='wam_pi05_v11_once_test_inference_v1',reservation_binding=r['reservation_binding'],producer=r['producer']),'Original test claim changed')
    noncontent_predicates(info)

def _admission(probes,test,*,_operation=None):
    # One operation owns its early full selected context. This local reference is
    # never persisted or reused across API calls/processes. D's test-scope loader
    # still performs its original authorization independently.
    a={};pb=binding(probes);tb=binding(test);before=read_binding(pb,a);read_binding(tb,a)
    info=_locked(before['lock_binding'])
    p=_verify(probes,'probes',locked=info);t=_verify(test,'test',locked=info);merge(a,p['artifact_sha256']);merge(a,t['artifact_sha256'])
    need(p['lock_binding']==t['lock_binding'] and p['checkpoint_binding']==t['checkpoint_binding'] and p['dataset_binding']==t['dataset_binding'],'Both actual collections must use same selected model/data')
    pe=verify_execution(probes);te=verify_execution(test);merge(a,pe['artifact_sha256']);merge(a,te['artifact_sha256']);merge(a,info['artifact_sha256'])
    passed=p['result']['passed'] and t['result']['passed'];value=dict(schema=ADMISSION_SCHEMA,service_eligible=bool(passed),candidate_admitted=bool(passed),admitted_checkpoint_binding=p['checkpoint_binding'] if passed else None,dataset_binding=p['dataset_binding'],lock_binding=p['lock_binding'],probes_binding=p['receipt_binding'],test_binding=t['receipt_binding'],source_binding=source_binding(),artifact_sha256=a,bounded_by_codec=True,learned_safety_validated=False,physical_acceptance=False,population_generalization=False,status='simulator_candidate' if passed else 'rejected_probes_or_test',no_reselection=True)
    need(verify_execution(probes)==pe and verify_execution(test)==te,'External process evidence changed')
    _end_admission(value,info)
    if _operation is not None:_operation['context']=info
    return value

def artifact_admission(probes,test,output):
    operation={};value=_admission(probes,test,_operation=operation);info=operation['context'];p=Path(output);need(p==info['run']/'postlock/admission.json' and p==p.resolve(),'Original admission path required')
    invalid=p.with_suffix('.invalid.json')
    need(not invalid.exists() and not invalid.is_symlink(),'Admission publication invalid')
    h=digest(value)
    try:
        write_new(p,value)
        _end_admission(value,info);D._pin(p,h,{})
        need(not invalid.exists() and not invalid.is_symlink(),'Late admission invalidation')
    except BaseException:
        if not invalid.exists() and not invalid.is_symlink():write_new(invalid,dict(invalid=True))
        raise
    return value

def verify_admission(receipt,expected_sha=None):
    p=Path(receipt);rb=binding(p);need(expected_sha is None or rb['sha256']==expected_sha,'Admission SHA differs');a={};v=read_binding(rb,a)
    invalid=p.with_suffix('.invalid.json');need(not invalid.exists() and not invalid.is_symlink(),'Admission publication invalid')
    operation={};rebuilt=_admission(v['probes_binding']['path'],v['test_binding']['path'],_operation=operation);need(v==rebuilt,'Admission reconstruction differs');info=operation['context'];need(p==info['run']/'postlock/admission.json','Original admission path differs')
    _end_admission(v,info);D._recheck(a);no_failure(Path(v['probes_binding']['path']).parent);no_failure(Path(v['test_binding']['path']).parent);need(not invalid.exists() and not invalid.is_symlink(),'Late admission invalidation');return dict(v,receipt_binding=rb,artifact_sha256=dict(v['artifact_sha256'],**{rb['path']:rb['sha256']}))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--lock',type=Path);parser.add_argument('--lock-sha256');parser.add_argument('--kind',choices=('probes','test'));parser.add_argument('--output',type=Path);parser.add_argument('--collect',action='store_true');parser.add_argument('--verify-admission',type=Path);parser.add_argument('--admission-sha256');args=parser.parse_args()
    if args.verify_admission is not None:
        need(args.admission_sha256 is not None and not args.collect and all(x is None for x in (args.lock,args.lock_sha256,args.kind,args.output)),'Exclusive read-only admission verification arguments required')
        print(json.dumps(verify_admission(args.verify_admission,args.admission_sha256),allow_nan=False));return
    need(args.admission_sha256 is None and all(x is not None for x in (args.lock,args.lock_sha256,args.kind,args.output)),'Explicit lock/kind/output required')
    lb=dict(path=str(args.lock),sha256=args.lock_sha256)
    if not args.collect:
        p=args.output/'receipt.json'
        if not p.exists():print(json.dumps(dict(status='not_ready',model_initialized=False)));raise SystemExit(2)
        r=verify_probes(p) if args.kind=='probes' else verify_test(p);need(r['lock_binding']==lb,'Caller lock differs');print(json.dumps(r,allow_nan=False));return
    info=_locked(lb);require_output(info['run'],args.kind,args.output);reservation=_reserve_test(info) if args.kind=='test' else None
    stop=[False];old={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    for sig in old:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    try:
        model=SelectedPredictor(info);check_cancel(lambda:stop[0])
        r=_collect(lb,args.kind,args.output,model.tokenizer,model,lambda:stop[0],reservation,locked=info)
        print(json.dumps(r,allow_nan=False),flush=True)
    finally:
        for sig,handler in old.items():signal.signal(sig,handler)
        if stop[0]:
            folder=args.output
            if folder.is_dir() and not (folder/'cancelled.json').exists():write_new(folder/'cancelled.json',dict(cancelled=True,reason='CLI final interruption'))
            raise InterruptedError('CLI interrupted')

if __name__=='__main__':main()
