"""Inactive conditional C comparison and simulator-candidate admission.

Only explicit collection may initialize a model. Negative-B fixed160 collection
is intentionally unsupported; native selected-B receipts keep their V8 identity.
Every public verifier reconstructs provenance; stored booleans are not admission.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import numpy as np
from sim.auscultation_pi05_v8 import pair_admission as P
from sim.auscultation_pi05_v7.coverage_pilot_schedule import digest,file_sha,json_bytes
from sim.auscultation_pi05_v6.action_contract import LOWER,UPPER
from . import action_encoding as C,training_data as D,evaluation as E,train as T,trigger

PLAN_SCHEMA='wam_rational_q4_comparison_plan_v1'
FIXED_SCHEMA='wam_rational_q4_fixed160_receipt_v1'
PROCESS_EXIT_SCHEMA='wam_rational_q4_fixed160_process_exit_v1'
ADMISSION_SCHEMA='wam_rational_q4_comparison_admission_v1'
PROCESS_EXIT_REQUIRED=('schema','process_identity','command','exit_code','process_group_drained','errors','signals','deferred_signals','receipt_sha256','guard_evidence_binding','guard_source_binding')
GUARD_SHA256=P.GUARD_SHA256
SAMPLER=dict(E.SAMPLER,passes=2,key_rule='fresh key917 per row and repeat')
PARENT_INVENTORY=T.ROOT/'cloud-runtime-v9/rational-q4-20260928-v1.source_inventory.json'
PARENT_INVENTORY_SHA='517946f8d5ed9150c1d0d36891d1c84d1e8637a2e28fbef9afd5d4c3d55f87cc'
INVENTORY_SCHEMA='wam_rational_q4_comparison_runtime_inventory_v1'
SOURCE_KEY='sim/auscultation_pi05_v9/comparison_admission.py'
TEST_KEY='sim/auscultation_pi05_v9/tests/test_comparison_admission.py'
need=E.need
array_contract=P.array_contract
same=P.same
input_contract=P.input_contract
fixed_rows=P.fixed_rows
fixed_references=P.fixed_references
describe=P.describe
research_comparison=P.research_comparison

class UnsupportedRoute(ValueError):pass


def canonical(path,*,file=True):
    path=Path(path)
    need(path.is_absolute() and path.resolve()==path and all(not p.is_symlink() for p in (path,*path.parents)), 'Canonical non-symlink absolute path required')
    if file:need(path.is_file(),'Regular evidence file required')
    return path


def binding(path):
    path=canonical(Path(path).absolute());return dict(path=str(path),sha256=file_sha(path))


def bound(value):
    need(isinstance(value,dict) and set(value)=={'path','sha256'},'Exact path/SHA binding required')
    need(isinstance(value['sha256'],str) and re.fullmatch('[0-9a-f]{64}',value['sha256']) is not None,'SHA256 required')
    path=canonical(value['path']);need(file_sha(path)==value['sha256'],'Bound artifact changed');return path


def read_pinned(value):
    path=bound(value);raw=path.read_bytes();need(hashlib.sha256(raw).hexdigest()==value['sha256'],'Consumed JSON changed')
    return json.loads(raw)


def add_binding(closure,value):
    path=bound(value);key=str(path)
    need(key not in closure or closure[key]==value['sha256'],'Conflicting consumed artifact identity')
    closure[key]=value['sha256'];return value


def add_map(closure,values):
    for path,sha in values.items():add_binding(closure,dict(path=path,sha256=sha))


def recheck(closure):
    for path,sha in closure.items():bound(dict(path=path,sha256=sha))


def no_cancel(path):
    path=Path(path);need(not path.exists() and not path.is_symlink(),'Cancellation marker present: '+str(path))


def fresh_output(output,closure,*,protected_trees=()):
    output=canonical(Path(output).absolute(),file=False)
    trees=[Path(__file__).resolve().parents[2],T.ROOT/'runs',T.ROOT/'datasets',T.ROOT/'public-assets',T.ROOT/'vendor',*map(Path,protected_trees)]
    # Evidence directories are inputs too. The fixed public checkpoint receipt
    # in ROOT is a singleton file; treating its parent as an evidence tree would
    # prohibit every independent diagnostic output under the project base.
    trees.extend(Path(p).parent for p in closure if Path(p).parent!=T.ROOT)
    T.separate(output,trees)
    for raw in closure:
        path=Path(raw)
        need(output!=path and output not in path.parents,'Output would contain consumed evidence')
    need(not output.exists(),'Fresh independent output required');output.mkdir(parents=True);return output


def write_immutable(path,value):
    E.write_new(path,value);Path(path).chmod(0o444);return binding(path)


def verify_condition(value):
    saved=read_pinned(value);actual=trigger.verify_receipt(value['path'],value['sha256'])
    need(actual==saved and saved.get('schema')==trigger.SCHEMA and saved.get('condition_satisfied') is True,'Actual completed-B condition required')
    bound(value)
    if saved['route']!='locked_B_fixed160':raise UnsupportedRoute('negative_B_selection fixed160 comparator is not implemented; original P gate is never bypassed')
    return saved


def collection_command(plan_binding,role,output):
    if role!='C_candidate':raise UnsupportedRoute('Selected B must reuse its bound native V8 receipt; B_comparator collection is unsupported')
    plan=bound(plan_binding);output=canonical(Path(output).absolute(),file=False)
    return [str(T.ROOT/'vendor/openpi/.venv/bin/python'),'-B','-m','sim.auscultation_pi05_v9.comparison_admission',
        '--plan',str(plan),'--plan-sha256',plan_binding['sha256'],'--role',role,'--output',str(output),'--collect']


def verify_numeric(dataset,rows,arrays,statuses,inputs,canonical_inputs,overlap):
    need(len(rows)==160 and len(statuses)==len(inputs)==320 and len(canonical_inputs)==160,'Complete two-pass denominator required')
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
                key=f'{label}_{i:03d}_normalized';expected.add(key);need(key in arrays,'Missing retained model output');z=np.asarray(arrays[key]);need(z.shape==(10,7) and z.dtype==np.float32,'Malformed model output')
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



def verify_guard(value,command,identity=None):
    need(value.get('command')==command and value.get('passed') is True and type(value.get('exit_code')) is int
         and value['exit_code']==0 and value.get('process_group_drained') is True
         and all(value.get(k)==[] for k in ('errors','signals','deferred_signals')),'Clean actual owned process required')
    gi=value['process_identity'];actual=dict(pid=gi['pid'],start_ticks=int(gi['start_ticks']))
    need(type(actual['pid']) is int and actual['pid']>0 and actual['start_ticks']>0
         and gi['pgid']==gi['sid']==actual['pid'],'Original owned process group required')
    need(identity is None or actual==identity,'Owned process identity differs');return actual


def verify_execution(receipt_binding,producer,closure):
    rp=bound(receipt_binding);value_binding=binding(rp.with_name('execution.json'));add_binding(closure,value_binding)
    value=read_pinned(value_binding)
    need(all(k in value for k in PROCESS_EXIT_REQUIRED) and value['schema']==PROCESS_EXIT_SCHEMA,'New external exit schema required')
    ident=producer['process_identity']
    need(set(ident)=={'pid','start_ticks'} and all(type(v) is int and v>0 for v in ident.values()),'Actual PID/start ticks required')
    need(value['process_identity']==ident and value['command']==producer['command'] and type(value['exit_code']) is int
         and value['exit_code']==0 and value['process_group_drained'] is True and value['receipt_sha256']==receipt_binding['sha256']
         and all(value[k]==[] for k in ('errors','signals','deferred_signals')),'External exit incomplete, cancelled or unbound')
    for key in ('guard_source_binding','guard_evidence_binding'):add_binding(closure,value[key])
    need(value['guard_source_binding']['sha256']==GUARD_SHA256,'Original process guard source required')
    gp=Path(value['guard_evidence_binding']['path']);need(gp.name.endswith('.process.json'),'Original guard evidence path required')
    late=gp.with_name(gp.name[:-len('.process.json')]+'.late_interrupt.json')
    for p in (late,rp.with_name('late_interrupt.json'),rp.with_name('cancelled.json')):no_cancel(p)
    verify_guard(read_pinned(value['guard_evidence_binding']),producer['command'],ident)
    recheck(closure);bound(receipt_binding)
    for p in (late,rp.with_name('late_interrupt.json'),rp.with_name('cancelled.json')):no_cancel(p)
    return value_binding


def verify_native_b(condition,checkpoint,closure):
    pb,rb=condition['fixed_plan_binding'],condition['fixed_receipt_binding']
    need(pb is not None and rb is not None,'Native selected-B plan and receipt required')
    add_binding(closure,pb);add_binding(closure,rb)
    ep=Path(rb['path']).with_name('execution.json');expected=condition['artifact_sha256'].get(str(ep))
    need(expected is not None,'Native actual exit absent from condition closure');add_binding(closure,dict(path=str(ep),sha256=expected))
    native=P.verify_fixed160(bound(pb),bound(rb))
    need(native['receipt_binding']==rb and native['execution_binding']==dict(path=str(ep),sha256=expected)
         and native['receipt']['schema']==P.FIXED_SCHEMA and native['receipt']['role']=='B_candidate'
         and native['receipt']['checkpoint_binding']==checkpoint and native['receipt']['status']=='finite','Original locked B comparator differs')
    recheck(closure);return native


def verify_selected120(selected):
    need(type(selected['local_step']) is int and selected['local_step']>0 and selected['eligible'] is True
         and selected['bounds']['all_passed'] is True and selected['bounds']['coordinate_count']==8400
         and selected['bounds']['violating_coordinates']==0 and selected['moving_buckets']['passed']==16,'Original locked C120 eligibility required')


def admission_decision(selected,numeric):
    if selected is None:
        need(numeric is None,'Negative C forbids fallback inference');return ['rejected_no_eligible_C']
    verify_selected120(selected)
    need(numeric is not None,'Actual C160 required')
    if numeric['status']=='numerical_failure':return ['rejected_C_numerical_output']
    need(numeric['status']=='finite' and numeric['bounds']['coordinate_count']==11200,'Complete finite C160 required')
    return [] if numeric['bounds']['all_passed'] is True and numeric['bounds']['violation_count']==0 else ['rejected_C_fixed160_raw_bound']


def verify_inventory(value,closure,diagnostic=None):
    add_binding(closure,value);inv=read_pinned(value);parent_binding=dict(path=str(PARENT_INVENTORY),sha256=PARENT_INVENTORY_SHA)
    add_binding(closure,parent_binding);parent=read_pinned(parent_binding)
    need(inv.get('schema')==INVENTORY_SCHEMA and inv.get('parent_inventory')==parent_binding,'Explicit comparison producer inventory required')
    files=inv['source_files'];old=parent['source_files'];need(len(old)==373 and set(files)==set(old)|{SOURCE_KEY,TEST_KEY}
         and all(files[k]==v for k,v in old.items()),'Exact immutable373 plus comparison source/test required')
    root=canonical(inv['source_project'],file=False);oldroot=canonical(parent['source_project'],file=False)
    need(root.is_dir() and root!=oldroot,'Independent producer runtime required')
    tree=[root,*root.rglob('*')]
    need(all(not p.is_symlink() and (p.is_file() or p.is_dir()) and not p.stat().st_mode&0o222 for p in tree),'Read-only regular producer tree required')
    need({str(p.relative_to(root)) for p in tree if p.is_file()}==set(files),'Exact producer file tree required')
    for name,h in files.items():
        rel=Path(name);need(not rel.is_absolute() and '..' not in rel.parts and str(rel)==name,'Contained source names required')
        path=root/rel;need(path.stat().st_nlink==1,'Independent regular producer files required');add_binding(closure,dict(path=str(path),sha256=h))
        if name in old:
            ancestor=oldroot/rel;add_binding(closure,dict(path=str(ancestor),sha256=h));need(not os.path.samefile(path,ancestor),'Independent source copies required')
    need(files[SOURCE_KEY]==file_sha(__file__),'Current comparison source differs')
    for name,h in T.source_hashes().items():need(files.get('sim/auscultation_pi05_'+name)==h,'C30 source differs from producer inventory')
    own=dict(path=str(root/SOURCE_KEY),sha256=files[SOURCE_KEY]);need(diagnostic is None or diagnostic==own,'Original diagnostic source binding differs')
    recheck(closure);return inv,own


def _review_sources(prep,closure,launcher):
    inv=read_pinned(dict(path=prep['inventory_path'],sha256=prep['inventory_sha256']))
    need(prep['inventory_sha256']==PARENT_INVENTORY_SHA and prep['inventory_path']==str(PARENT_INVENTORY)
         and prep['runtime_project']==inv['source_project'],'Original C training snapshot required')
    for path,h in inv['source_files'].items():add_binding(closure,dict(path=str(Path(inv['source_project'])/path),sha256=h))
    prefix='sim/auscultation_pi05_v9/'
    scopes={'codec':{prefix+n for n in ('__init__.py','action_encoding.py','training_data.py','tests/test_action_encoding.py','tests/test_training_data.py')},
        **{name:{prefix+name+'.py',prefix+'tests/test_'+name+'.py'} for name in ('trigger','evaluation','train')}}
    scopes['trainer']=scopes.pop('train')
    scopes['launcher']={'results/diagnostics/rational_training_cloud_launcher_v1/'+n for n in ('launch.py','test_launch.py')}
    required={name+'_'+kind:files for name,files in scopes.items() for kind in ('spec','quality')}
    need(set(prep['reviews'])==set(required),'Ten approved C training roles required')
    live=dict(inv['source_files'])
    for name in ('launch.py','test_launch.py'):
        source=launcher.with_name(name);b=binding(source);add_binding(closure,b)
        live['results/diagnostics/rational_training_cloud_launcher_v1/'+name]=b['sha256']
    for role,scope in required.items():
        rb=prep['reviews'][role];add_binding(closure,rb);review=read_pinned(rb)
        need(review.get('passed') is True and review.get('P1_findings',review.get('P1'))==[] and review.get('P2_findings',review.get('P2'))==[],'Approved source-bound '+role+' review required')
        maps=[review[k] for k in ('source_sha256','reviewed_source_sha256') if k in review];need(maps,'Review source map required');covered={}
        for values in maps:
            for key,h in values.items():need(key not in covered or covered[key]==h,'Review source conflict');covered[key]=h
        need(scope<=set(covered) and all(covered[k]==live[k] for k in scope),'Current review scope bytes required')
        need(all(k not in live or live[k]==h for k,h in covered.items()),'Stale reviewed source claim')


def verify_training_lifecycle(value,ctx,condition_binding,closure):
    add_binding(closure,value);life=read_pinned(value);run=Path(ctx['config']['training_run_path']);folder=Path(value['path']).parent
    need(run==T.ROOT/'runs/rational-q4-original8-v1-C','Original approved single C output namespace required')
    need(Path(value['path']).name=='lifecycle.json' and life.get('schema')=='wam_rational_q4_training_lifecycle_v1'
         and life.get('mode')==C.C and life.get('output')==str(run) and life.get('execute') is True
         and life.get('completed') is True and life.get('training_complete') is True and life.get('errors')==[]
         and life.get('selected') is False and life.get('model_accepted') is False and life.get('gpu_released') is True
         and life.get('outer_flock') is False and life.get('trainer_owns_shared_lock') is True,'Actual clean C training owner required')
    no_cancel(folder/'late_interrupt.json');stages=life['stages']
    need([s['stage'] for s in stages]==['cpu_preflight','training','completion_verification'],'Exact three owned stages required')
    final=stages[-1]['command'];need(len(final)==11 and final[:3]==[str(T.ROOT/'vendor/openpi/.venv/bin/python'),'-u','-B'] and final[4]=='--preparation'
         and final[6]=='--preparation-sha256' and final[8]=='--evidence' and final[9]==str(folder) and final[10]=='--verify-completion','Exact guarded completion command required')
    preparation=dict(path=final[5],sha256=final[7]);add_binding(closure,preparation);prep=read_pinned(preparation)
    need(prep==life['preparation'] and preparation['sha256']==life['preparation_sha256'] and prep['schema']=='wam_rational_q4_launch_preparation_v1','Original C dispatch preparation required')
    cfg=ctx['config'];need(prep['mode']==C.C and prep['output']==str(run) and prep['condition_receipt']==condition_binding
         and prep['saved_normalization']==cfg['saved_normalization_binding'] and prep['training_source_sha256']==cfg['source_sha256'],'Dispatched training inputs differ')
    need(prep['experiment_spec']['sha256']==T.SPEC_SHA256 and prep['saved_normalization']['sha256']==T.NORMALIZATION_FILE_SHA256
         and prep['manifest']==dict(path=str(T.ROOT/'datasets/pi05_v6_task_pose_armature02_v1/task_conditioned_bound/manifest.json'),sha256=D.MANIFEST_SHA256),
         'Approved original specification, saved stats and manifest required')
    for name,expected in dict(training_steps=14600,sample_draws=58400,candidate_steps=list(E.EXPECTED_STEPS),seed=61,batch_size=4,learning_rate=5e-5).items():
        need(type(prep.get(name)) is type(expected) and prep[name]==expected,'Fixed dispatched training setting differs')
    launcher=canonical(final[3]);add_binding(closure,dict(path=str(launcher),sha256=prep['launcher_sha256']))
    _review_sources(prep,closure,launcher)
    for key in ('experiment_spec','condition_receipt','saved_normalization','manifest'):add_binding(closure,prep[key])
    add_binding(closure,dict(path=prep['inventory_path'],sha256=prep['inventory_sha256']))
    guard=dict(path=str(T.ROOT/'diagnostics/coverage_pilot_cloud_launcher/process_guard.py'),sha256=GUARD_SHA256);add_binding(closure,guard)
    need(prep['guard_sha256']==GUARD_SHA256,'Original training process guard required')
    command=[str(T.ROOT/'vendor/openpi/.venv/bin/python'),'-u','-B','-m','sim.auscultation_pi05_v9.train','--manifest',prep['manifest']['path'],
        '--mode',C.C,'--output',str(run),'--condition-receipt',condition_binding['path'],'--condition-sha256',condition_binding['sha256'],
        '--normalization',prep['saved_normalization']['path'],'--normalization-sha256',prep['saved_normalization']['sha256']]
    actual=life['command'];need(actual[:len(command)]==command,'Original training argv required')
    if len(actual)>len(command):
        need(len(actual)==len(command)+2 and actual[-2]=='--resume','Only explicit same-run resume allowed')
        cp=canonical(actual[-1],file=False);need(cp.parent in (run/'model',run/'recovery') and re.fullmatch('step-[0-9]{12}',cp.name),'Original same-run recovery path required')
    for stage,argv in zip(stages,(actual+['--verify-only'],actual,final)):
        path=folder/(stage['stage']+'.process.json');sb=binding(path);add_binding(closure,sb)
        need(read_pinned(sb)==stage,'Actual process stage record differs');verify_guard(stage,argv)
        no_cancel(folder/(stage['stage']+'.late_interrupt.json'))
    receipt_path=run/'training_run.json';need(life['training_receipt_sha256']==file_sha(receipt_path),'Actual training owner receipt SHA differs')
    main=read_pinned(dict(path=str(receipt_path),sha256=life['training_receipt_sha256']));need(main==life['training_receipt'],'Actual training owner consumed another receipt')
    cb=dict(path=str(folder/'completion.json'),sha256=life['completion_sha256']);add_binding(closure,cb);complete=read_pinned(cb)
    need(complete.get('schema')=='wam_rational_q4_full_context_completion_v1' and complete.get('verified') is True
         and complete['output']==str(run) and complete['preparation_sha256']==preparation['sha256']
         and complete['source_sha256']==cfg['source_sha256'] and complete['candidate_steps']==list(E.EXPECTED_STEPS)
         and complete['terminal_segment_binding']==ctx['terminal_segment_binding'] and complete['training_receipt']==main
         and complete['selected'] is False and complete['model_accepted'] is False,'Actual full C completion proof differs')
    expected={str(run/n) for n in E.CONTEXT_FILES}|{ctx['terminal_segment_binding']['path']}
    for step in E.EXPECTED_STEPS:
        expected|={str(run/'model'/('step-%012d'%step)/n) for n in ('metadata.json','arrays.npz','COMPLETE.json')}
        expected|={str(run/'diagnostics'/('local_%06d'%step)/n) for n in ('report.json','predictions.npz')}
    need(set(complete['artifact_sha256'])==expected,'Complete original checkpoint/report closure required');add_map(closure,complete['artifact_sha256'])
    dispatch=folder/'dispatch.json';db=binding(dispatch);add_binding(closure,db);v=read_pinned(db)
    need(v==dict(schema='wam_rational_q4_owned_dispatch_v1',evidence=str(folder),preparation_sha256=preparation['sha256'],output=str(run),launcher_sha256=prep['launcher_sha256']),'Original owned dispatch differs')
    for stage in stages:no_cancel(folder/(stage['stage']+'.late_interrupt.json'))
    no_cancel(folder/'late_interrupt.json');E.verify_terminal_segment(run,ctx['terminal_segment_binding'],ctx['verified_input_sha256']);recheck(closure)
    return value


def _lock_context(value,arm,closure):
    api,trainer=(P.E,P.T) if arm=='B' else (E,T)
    add_binding(closure,value);saved=read_pinned(value);need(saved['schema']==api.LOCK_SCHEMA,'Original arm selection schema required')
    selection=saved['selection'];run=Path(selection['run']);marker=binding(run/api.MARKER);add_binding(closure,marker)
    add_binding(closure,dict(path=saved['selection_path'],sha256=saved['selection_sha256']));add_map(closure,selection['artifact_sha256'])
    need(tuple(c['local_step'] for c in selection['candidates'])==tuple(E.EXPECTED_STEPS),'Full13 locked denominator required')
    for candidate in selection['candidates']:
        ck=candidate['checkpoint_binding'];directory=Path(ck['checkpoint_path'])
        for name,key in (('metadata.json','metadata_sha256'),('arrays.npz','arrays_sha256'),('COMPLETE.json','complete_sha256')):
            add_binding(closure,dict(path=str(directory/name),sha256=ck[key]))
    need(api.verify_selection_lock(bound(value),value['sha256'])==saved,'Original selection changed during verification')
    ctx=api.load_context(run);need(ctx['completed'] is True,'Complete original arm context required')
    if arm=='C':E.verify_terminal_segment(run,ctx['terminal_segment_binding'],ctx['verified_input_sha256'])
    config_binding=dict(path=str(run/'configuration.json'),sha256=selection['artifact_sha256'][str(run/'configuration.json')])
    configuration=read_pinned(config_binding);need(configuration['config']==ctx['config'],'Consumed configuration differs')
    dataset_closure={};P.dataset_artifacts(dict(path=configuration['manifest'],sha256=configuration['manifest_sha256']),dataset_closure);add_map(closure,dataset_closure)
    add_binding(closure,dict(path=str(T.ROOT/'public_checkpoint_verified.json'),sha256=ctx['config']['assets']['checkpoint_receipt_sha256']))
    add_binding(closure,dict(path=str(T.ROOT/'public-assets/paligemma_tokenizer.model'),sha256=ctx['config']['assets']['tokenizer_sha256']))
    for candidate in selection['candidates']:
        expected=candidate['checkpoint_binding'];need(trainer.checkpoint_binding(expected['checkpoint_path'])==expected,'Locked original checkpoint differs')
    recheck(closure);return ctx,saved


def validate_pair(contexts,condition_binding):
    b,c=contexts['B'],contexts['C'];bc,cc=b['config'],c['config']
    need(bc['mode']==P.C.B and cc['mode']==C.C and cc['condition_binding']==condition_binding,'Original B and conditional C identities required')
    need(bc['source_sha256']==T.ORIGINAL_B_SOURCES and cc['source_sha256']==T.source_hashes()
         and {k:cc['source_sha256'][k] for k in bc['source_sha256']}==bc['source_sha256'],'Distinct original B24/C30 source maps required')
    for key in ('assets','observation_contract','seed','learning_rate','batch_size'):
        need(bc[key]==cc[key],'B/C original '+key+' differs')
    need(cc['seed']==61 and cc['batch_size']==4 and cc['learning_rate']==5e-5,'Original model settings required')
    need(c['dataset'].parent.data_hash==b['dataset'].data_hash and cc['parent_data_hash']==b['dataset'].data_hash,'C original qualified B view required')
    need(b['schedule']==c['schedule'] and c['schedule']['total_steps']==14600 and c['schedule']['total_draws']==58400
         and tuple(c['schedule']['candidate_steps'])==tuple(E.EXPECTED_STEPS),'Identical full schedule and ordered draws required')
    need(bc['schedule_binding']['sha256']==cc['schedule_binding']['sha256'] and bound({k:bc['schedule_binding'][k] for k in ('path','sha256')}).read_bytes()==bound({k:cc['schedule_binding'][k] for k in ('path','sha256')}).read_bytes(),'Schedule content bytes differ')
    bi,bh=P.T._load_initialization(bc);ci,ch=T._load_initialization(cc)
    need(ci['numerical_initialization']==bi['numerical_initialization'] and ci['paired_B_numerics_equal'] is True
         and ci['paired_B_receipt_integrity']==bi['integrity_sha256'],'Fresh complete B/C numerical initialization differs')
    need(cc['b_reference']['initialization_binding']==dict(path=str(Path(bc['training_run_path'])/'initialization.json'),sha256=bh),'C references another B initialization')
    need(T.records_for(c['dataset'])==P.T.records_for(b['dataset']),'Original fit origins differ')
    rows=fixed_rows(b['dataset']);need(rows==fixed_rows(c['dataset'])==fixed_rows(c['dataset'].parent),'Canonical first20 FREE origins differ')
    for x,y in zip(fixed_references(b['dataset'],rows),fixed_references(c['dataset'],rows)):need(same(x,y),'Original fixed160 labels/anchors differ')
    need(E.build_diagnostic_schedule(c['dataset'])==P.E.build_diagnostic_schedule(b['dataset']),'Original120 row ordering differs')
    for x,y in zip(E.references(c['dataset'],E.build_diagnostic_schedule(c['dataset'])),P.E.references(b['dataset'],P.E.build_diagnostic_schedule(b['dataset']))):need(same(x,y),'Original120 labels/anchors differ')
    return rows


def _build_plan(condition_binding,c_lock_binding,*,c_lifecycle_binding,producer_inventory_binding,b_fixed160_binding=None,diagnostic=None):
    closure={};add_binding(closure,condition_binding);add_binding(closure,c_lock_binding);add_binding(closure,c_lifecycle_binding);add_binding(closure,producer_inventory_binding)
    condition=verify_condition(condition_binding);add_map(closure,condition['artifact_sha256'])
    b_lock=condition['b_selection_lock_binding'];b,bl=_lock_context(b_lock,'B',closure);c,cl=_lock_context(c_lock_binding,'C',closure)
    contexts={'B':b,'C':c};locks={'B':bl,'C':cl};rows=validate_pair(contexts,condition_binding)
    bselected=bl['selection']['selected'];need(bselected is not None and bselected['local_step']>0,'Original positive locked B required')
    cselected=cl['selection']['selected'];bck=bselected['checkpoint_binding'];cck=None if cselected is None else cselected['checkpoint_binding']
    if cselected is not None:verify_selected120(cselected)
    if b_fixed160_binding is not None:need(b_fixed160_binding==condition['fixed_receipt_binding'],'Cannot substitute B evidence or comparator')
    native=verify_native_b(condition,bck,closure)
    oldplan=read_pinned(condition['fixed_plan_binding'])
    need(oldplan['fixed160_rows']==rows and oldplan['fixed160_rows_sha256']==digest(rows) and native['receipt']['sampler']==SAMPLER,'Native B origins/sampler differ')
    verify_training_lifecycle(c_lifecycle_binding,c,condition_binding,closure)
    inv,source=verify_inventory(producer_inventory_binding,closure,diagnostic)
    recheck(closure)
    plan=dict(schema=PLAN_SCHEMA,specification_binding=condition['specification_binding'],condition_binding=condition_binding,trigger_route=condition['route'],
        b_selection_lock_binding=b_lock,c_selection_lock_binding=c_lock_binding,b_checkpoint_binding=bck,c_checkpoint_binding=cck,b_comparator_kind='original_locked_selected',
        b_evidence_reuse=dict(native_schema=P.FIXED_SCHEMA,original_role='B_candidate',plan_binding=condition['fixed_plan_binding'],receipt_binding=native['receipt_binding'],execution_binding=native['execution_binding']),
        b_dataset_binding=P.D.dataset_binding(b['dataset']),c_dataset_binding=D.dataset_binding(c['dataset']),
        b_initialization_binding=dict(path=str(Path(b['config']['training_run_path'])/'initialization.json'),sha256=bl['selection']['artifact_sha256'][str(Path(b['config']['training_run_path'])/'initialization.json')]),
        c_initialization_binding=dict(path=str(Path(c['config']['training_run_path'])/'initialization.json'),sha256=cl['selection']['artifact_sha256'][str(Path(c['config']['training_run_path'])/'initialization.json')]),
        training_schedule_bindings=[b['config']['schedule_binding'],c['config']['schedule_binding']],b_training_lifecycle_binding=condition['lifecycle_binding'],c_training_lifecycle_binding=c_lifecycle_binding,
        fixed160_rows=rows,fixed160_rows_sha256=digest(rows),sampler=SAMPLER,diagnostic_source_binding=source,frozen_producer_inventory_binding=producer_inventory_binding,
        source_sha256=dict(T.source_hashes(),**{'v9/comparison_admission.py':source['sha256']}),
        status='rejected_no_eligible_C' if cselected is None else 'ready_for_owned_collection',artifact_sha256=closure,
        numerical_initialization_equal=True,identical_schedule_content=True,negative_B_route_supported=False,model_initialized=False,authorization_granted=False)
    return plan,contexts,locks,native


def _protected(plan):
    values=[plan[k] for k in ('condition_binding','b_selection_lock_binding','c_selection_lock_binding','b_training_lifecycle_binding','c_training_lifecycle_binding','frozen_producer_inventory_binding','diagnostic_source_binding')]
    reuse=plan['b_evidence_reuse'];values.extend(reuse[k] for k in ('plan_binding','receipt_binding','execution_binding'))
    return [Path(v['path']).parent for v in values]


def prepare_comparison(condition_binding,c_lock_binding,output,*,c_lifecycle_binding,producer_inventory_binding,b_fixed160_binding=None):
    p,ctx,locks,native=_build_plan(condition_binding,c_lock_binding,c_lifecycle_binding=c_lifecycle_binding,producer_inventory_binding=producer_inventory_binding,b_fixed160_binding=b_fixed160_binding)
    out=fresh_output(output,p['artifact_sha256'],protected_trees=_protected(p));recheck(p['artifact_sha256']);return write_immutable(out/'plan.json',p)


def _verify_plan(value):
    saved=read_pinned(value)
    need(saved.get('schema')==PLAN_SCHEMA,'New conditional comparison plan required')
    p,ctx,locks,native=_build_plan(saved['condition_binding'],saved['c_selection_lock_binding'],c_lifecycle_binding=saved['c_training_lifecycle_binding'],
        producer_inventory_binding=saved['frozen_producer_inventory_binding'],b_fixed160_binding=saved['b_evidence_reuse']['receipt_binding'],diagnostic=saved['diagnostic_source_binding'])
    need(saved==p,'Plan differs from original fresh reconstruction');bound(value);return p,ctx,locks,native


def _canonical_inputs(ctx,rows):
    tokenizer=P._tokenizer(ctx['config'])
    return [input_contract(D.make_batch(ctx['dataset'],'train',[row['train_index']],tokenizer)[0]) for row in rows]


def _overlap(ctx,ck,rows,closure):
    report_path=E._report_path(ck);rb=binding(report_path);add_binding(closure,rb);report=read_pinned(rb)
    nb=dict(path=str(report_path.with_name('predictions.npz')),sha256=report['numeric_sha256']);add_binding(closure,nb)
    E.verify_checkpoint_report(report_path,ctx['dataset'],ck);old=E.build_diagnostic_schedule(ctx['dataset']);result={}
    with np.load(bound(nb),allow_pickle=False) as f:
        for i,row in enumerate(rows):
            if row['ordinal']!=0:continue
            matches=[j for j,x in enumerate(old) if x['episode_id']==row['episode_id'] and x['phase']=='FREE' and x['position']=='first' and x['start']==row['start']]
            need(len(matches)==1,'Original frozen120 overlap missing');j=matches[0];result[i]=dict(normalized=f['normalized_predictions'][j],raw=f['predictions'][j])
    need(set(result)==set(range(0,160,20)),'All8 frozen overlaps required');recheck(closure);return result


def _record(plan_binding,plan,ctx,out,arrays,statuses,inputs,producer,environment):
    rows=plan['fixed160_rows'];closure=dict(plan['artifact_sha256'])
    numeric=verify_numeric(ctx['dataset'],rows,arrays,statuses,inputs,_canonical_inputs(ctx,rows),_overlap(ctx,plan['c_checkpoint_binding'],rows,closure))
    npz=out/'numeric.npz'
    with npz.open('xb') as f:np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
    record=dict(schema=FIXED_SCHEMA,role='C_candidate',mode=C.C,plan_binding=plan_binding,checkpoint_binding=plan['c_checkpoint_binding'],dataset_binding=plan['c_dataset_binding'],selection_lock_binding=plan['c_selection_lock_binding'],
        sampler=SAMPLER,rows_sha256=plan['fixed160_rows_sha256'],numeric_file_binding=binding(npz),numeric_array_contracts={k:array_contract(v) for k,v in sorted(arrays.items())},
        actual_consumed_input_leaf_hashes=inputs,per_row_pass_and_replay_status=statuses,producer_execution_binding=producer,producer_source_binding=plan['diagnostic_source_binding'],runtime_inventory_binding=plan['frozen_producer_inventory_binding'],
        runtime_environment=environment,official_openpi_commit=ctx['config']['openpi_commit'],official_assets_binding=ctx['config']['assets'],errors=[],training_rng_consumed=False,heldout_predictions_generated=False,
        artifact_sha256={str(npz):file_sha(npz)},**numeric)
    recheck(closure);bound(plan_binding);return write_immutable(out/'receipt.json',record)


def _producer():
    raw=Path('/proc/self/stat').read_text();ticks=int(raw[raw.rfind(')')+2:].split()[19]);argv=Path('/proc/self/cmdline').read_bytes().rstrip(b'\0').split(b'\0')
    return dict(process_identity=dict(pid=os.getpid(),start_ticks=ticks),command=[x.decode() for x in argv],source_file_binding=binding(__file__))


def collect_fixed160(plan_binding,role,output):
    command=collection_command(plan_binding,role,output);plan,contexts,locks,native=_verify_plan(plan_binding)
    need(plan['status']=='ready_for_owned_collection' and plan['c_checkpoint_binding'] is not None,'Negative C is not collectable')
    ctx=contexts['C'];out=fresh_output(output,plan['artifact_sha256'],protected_trees=[*_protected(plan),Path(plan_binding['path']).parent])
    producer=_producer();need(producer['command']==command and producer['source_file_binding']==plan['diagnostic_source_binding'],'Explicit original collection process required')
    E.write_new(out/'started.json',dict(plan_binding=plan_binding,role=role,producer=producer))
    arrays={};statuses=[];inputs=[]
    try:
        import jax
        from openpi.models import model as model_api
        from sim.auscultation_pi05_v7 import model
        runtime=model.initialize(61,5e-5);need(runtime['assets']==ctx['config']['assets'],'Official assets changed')
        model.restore(runtime,plan['c_checkpoint_binding']['checkpoint_path'],ctx['dataset'].data_hash,ctx['config'],ctx['dataset'].normalization)
        sample=model.sampler(runtime,steps=10);rows=plan['fixed160_rows'];arrays['reference'],arrays['anchors']=fixed_references(ctx['dataset'],rows)
        for label in ('pass','replay'):
            for i,row in enumerate(rows):
                obs,_=D.make_batch(ctx['dataset'],'train',[row['train_index']],runtime['tokenizer']);inputs.append(input_contract(obs));status=dict(pass_name=label,row_id=i)
                try:value=np.asarray(sample(runtime['trainable'],runtime['frozen'],model_api.Observation(**jax.device_put(obs)),jax.random.key(917)))
                except FloatingPointError as error:status.update(status='floating_point_error',error=str(error)[:2048] or type(error).__name__)
                else:
                    need(value.shape==(1,10,7) and value.dtype==np.float32,'Actual float32 model output required')
                    arrays[f'{label}_{i:03d}_normalized']=value[0];status['status']='finite' if np.isfinite(value).all() else 'nonfinite_return'
                    if status['status']=='finite':arrays[f'{label}_{i:03d}_raw']=D.decode_prediction(ctx['dataset'],value[0],ctx['dataset'].episodes[row['episode_index']]['state'][row['start']],'FREE')
                statuses.append(status)
        need(_verify_plan(plan_binding)[0]==plan,'Plan changed during actual collection')
        return _record(plan_binding,plan,ctx,out,arrays,statuses,inputs,producer,P._environment())
    except BaseException as error:
        with (out/'failed_numeric.npz').open('xb') as f:np.savez(f,**arrays)
        E.write_new(out/'failure.json',dict(error=type(error).__name__+': '+str(error),statuses=statuses,inputs=inputs,producer=producer,failed_numeric_sha256=file_sha(out/'failed_numeric.npz')))
        raise


def _verify_c_fixed(plan_binding,plan,ctx,receipt_binding,closure):
    add_binding(closure,receipt_binding);path=bound(receipt_binding);r=read_pinned(receipt_binding)
    need(plan['c_checkpoint_binding'] is not None and path.name=='receipt.json','Original positive C and worker receipt required')
    expected=dict(schema=FIXED_SCHEMA,role='C_candidate',mode=C.C,plan_binding=plan_binding,
        checkpoint_binding=plan['c_checkpoint_binding'],dataset_binding=plan['c_dataset_binding'],selection_lock_binding=plan['c_selection_lock_binding'],
        sampler=SAMPLER,rows_sha256=plan['fixed160_rows_sha256'],producer_source_binding=plan['diagnostic_source_binding'],runtime_inventory_binding=plan['frozen_producer_inventory_binding'],
        official_openpi_commit=ctx['config']['openpi_commit'],official_assets_binding=ctx['config']['assets'],errors=[],training_rng_consumed=False,heldout_predictions_generated=False)
    for key,value in expected.items():need(r.get(key)==value and type(r.get(key)) is type(value),'C fixed160 '+key+' differs')
    env=r['runtime_environment']
    need(env['numpy_version']==ctx['schedule']['numpy_version'] and env['openpi_python_prefix']==str((T.ROOT/'vendor/openpi/.venv').resolve())
         and Path(env['python_executable']).is_absolute(),'Original producer numerical environment required')
    producer=r['producer_execution_binding']
    need(producer['command']==collection_command(plan_binding,'C_candidate',path.parent) and producer['source_file_binding']==plan['diagnostic_source_binding'],'Original collection command/source required')
    add_binding(closure,producer['source_file_binding']);add_binding(closure,r['numeric_file_binding'])
    need(bound(r['numeric_file_binding'])==path.with_name('numeric.npz'),'Original numeric artifact path required')
    with np.load(bound(r['numeric_file_binding']),allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
    need(r['numeric_array_contracts']=={k:array_contract(v) for k,v in sorted(arrays.items())},'Retained numeric contracts differ')
    need(r['artifact_sha256']=={r['numeric_file_binding']['path']:r['numeric_file_binding']['sha256']},'Worker numeric closure differs')
    result=verify_numeric(ctx['dataset'],plan['fixed160_rows'],arrays,r['per_row_pass_and_replay_status'],r['actual_consumed_input_leaf_hashes'],
        _canonical_inputs(ctx,plan['fixed160_rows']),_overlap(ctx,plan['c_checkpoint_binding'],plan['fixed160_rows'],closure))
    for key,value in result.items():need(r.get(key)==value,'C fixed160 numeric reconstruction differs')
    exit_binding=verify_execution(receipt_binding,producer,closure)
    recheck(closure);bound(receipt_binding)
    return dict(receipt=r,receipt_binding=receipt_binding,execution_binding=exit_binding)


def verify_fixed160(plan_binding,receipt_binding):
    plan,contexts,locks,native=_verify_plan(plan_binding);closure=dict(plan['artifact_sha256']);add_binding(closure,plan_binding)
    if receipt_binding==plan['b_evidence_reuse']['receipt_binding']:result=native
    else:result=_verify_c_fixed(plan_binding,plan,contexts['C'],receipt_binding,closure)
    _finish_checks(plan,contexts,closure,[result]);return result


def physical_summary(predicted,reference,anchors,rows):
    p,ref,q=[np.asarray(x,np.float64) for x in (predicted,reference,anchors)]
    need(p.shape==ref.shape==(120,10,7) and q.shape==(120,7) and len(rows)==120 and all(np.isfinite(v).all() for v in (p,ref,q)),'Original finite120 physical arrays required')
    error=p-ref;per={}
    for name,value,axes in (('first',error[:,0],0),('full',error,(0,1))):
        per[name+'_mae_rad']=np.abs(value).mean(axis=axes).tolist();per[name+'_signed_bias_rad']=value.mean(axis=axes).tolist();per[name+'_max_abs_rad']=np.abs(value).max(axis=axes).tolist()
    violations=[]
    for i,h,j in np.argwhere((p<LOWER)|(p>UPPER)):
        row=rows[int(i)];violations.append(dict(row=int(i),episode_id=row['episode_id'],phase=row['phase'],position=row['position'],start=row['start'],
            horizon=int(h),joint=int(j),side='lower' if p[i,h,j]<LOWER[j] else 'upper',value_rad=float(p[i,h,j]),lower_rad=float(LOWER[j]),upper_rad=float(UPPER[j])))
    displacement={}
    for name,value in (('predicted',p-q[:,None,:]),('reference',ref-q[:,None,:])):
        displacement[name]=dict(first_mae_per_joint_rad=np.abs(value[:,0]).mean(axis=0).tolist(),full_mae_per_joint_rad=np.abs(value).mean(axis=(0,1)).tolist(),
            first_signed_bias_per_joint_rad=value[:,0].mean(axis=0).tolist(),full_max_abs_per_joint_rad=np.abs(value).max(axis=(0,1)).tolist())
    first=[dict(episode_id=row['episode_id'],row=i,max_abs_rad=float(np.abs(error[i,0]).max()),mae_rad=float(np.abs(error[i,0]).mean())) for i,row in enumerate(rows) if row['phase']=='FREE' and row['position']=='first']
    return dict(per_joint=per,initial_FREE_first_errors=first,raw_bound_violations=violations,anchor_displacement=displacement,normalized_values_compared=False)


def _diagnostic(ctx,candidate,arm,closure):
    api=P.E if arm=='B' else E;ck=candidate['checkpoint_binding'];rp=api._report_path(ck)
    rb=dict(path=str(rp),sha256=candidate['report_sha256']);add_binding(closure,rb);saved=read_pinned(rb)
    nb=dict(path=str(rp.with_name('predictions.npz')),sha256=saved['numeric_sha256']);add_binding(closure,nb)
    need(api.verify_checkpoint_report(rp,ctx['dataset'],ck)==saved,'Original bound checkpoint report differs')
    with np.load(bound(nb),allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
    rows=api.build_diagnostic_schedule(ctx['dataset']);ref,q=api.references(ctx['dataset'],rows)
    need(same(arrays['reference'],ref) and same(arrays['anchors'],q),'Original diagnostic labels/anchors differ')
    result=dict(checkpoint_binding=ck,eligible=candidate['eligible'],report_binding=rb,numeric_binding=nb,metrics=saved['metrics'],
        physical=physical_summary(arrays['predictions'],ref,q,rows),representation_metrics=saved.get('representation_metrics'),
        reference_contract=array_contract(ref),anchors_contract=array_contract(q),rows_sha256=digest(rows))
    recheck(closure);return result


def _step_comparison(b,c):
    need(b['reference_contract']==c['reference_contract'] and b['anchors_contract']==c['anchors_contract'] and b['rows_sha256']==c['rows_sha256'],'B/C physical diagnostic origins/labels/anchors differ')
    return dict(B=b,C=c,research_claim_criteria=research_comparison(b['metrics']['scores'],c['metrics']['scores']),
        normalized_q4_compared=False,affects_admission=False)


def _diagnostic_comparisons(contexts,locks,closure):
    bc=locks['B']['selection']['candidates'];cc=locks['C']['selection']['candidates']
    need(tuple(c['local_step'] for c in bc)==tuple(c['local_step'] for c in cc)==tuple(E.EXPECTED_STEPS),'Complete step-matched13 required')
    values={arm:{} for arm in ('B','C')};curves=[]
    for b,c in zip(bc,cc):
        step=b['local_step'];bv=_diagnostic(contexts['B'],b,'B',closure);cv=_diagnostic(contexts['C'],c,'C',closure)
        values['B'][step]=bv;values['C'][step]=cv;curves.append(dict(local_step=step,comparison=_step_comparison(bv,cv)))
    bs=locks['B']['selection']['selected'];cs=locks['C']['selection']['selected']
    secondary=None if cs is None else _step_comparison(values['B'][bs['local_step']],values['C'][cs['local_step']])
    return _step_comparison(values['B'][14600],values['C'][14600]),curves,secondary


def _finish_checks(plan,contexts,closure,fixed):
    """Late marker absence is a live condition, not a saved hash or boolean."""
    recheck(closure)
    # The trigger reconstructs the old native B lifecycle/collection cancellation
    # paths. Current C segment markers are deliberately distinguished from history.
    condition=verify_condition(plan['condition_binding']);need(condition['b_selection_lock_binding']==plan['b_selection_lock_binding'],'B condition changed')
    c=contexts['C'];verify_training_lifecycle(plan['c_training_lifecycle_binding'],c,plan['condition_binding'],closure)
    for item in fixed:
        if item is None:continue
        if item['receipt']['schema']==FIXED_SCHEMA:
            need(verify_execution(item['receipt_binding'],item['receipt']['producer_execution_binding'],closure)==item['execution_binding'],'Original external execution changed')
    E.verify_terminal_segment(Path(c['config']['training_run_path']),c['terminal_segment_binding'],c['verified_input_sha256'])
    recheck(closure)
    _late_markers(plan,contexts,fixed)


def _late_markers(plan,contexts,fixed):
    # Check absence AFTER the expensive closure sweep. Include original B
    # evidence even when a negative C lock forbids both collection arguments.
    no_cancel(contexts['C']['terminal_segment_binding']['cancellation_path'])
    for key in ('b_training_lifecycle_binding','c_training_lifecycle_binding'):
        folder=Path(plan[key]['path']).parent
        for p in (folder/'late_interrupt.json',*folder.glob('*.late_interrupt.json')):no_cancel(p)
    bindings=[item['execution_binding'] for item in fixed if item is not None]
    if plan.get('b_evidence_reuse') is not None:bindings.append(plan['b_evidence_reuse']['execution_binding'])
    for value in bindings:
        execution=read_pinned(value);gp=Path(execution['guard_evidence_binding']['path']);folder=Path(value['path']).parent
        need(gp.name.endswith('.process.json'),'Original guard evidence path required')
        for p in (gp.with_name(gp.name[:-len('.process.json')]+'.late_interrupt.json'),folder/'late_interrupt.json',folder/'cancelled.json'):no_cancel(p)


def _final(plan_binding,plan,contexts,locks,native,b_receipt_binding,c_receipt_binding):
    closure=dict(plan['artifact_sha256']);add_binding(closure,plan_binding)
    primary,curves,secondary=_diagnostic_comparisons(contexts,locks,closure)
    selected=locks['C']['selection']['selected'];bf=cf=None
    if selected is None:
        need(b_receipt_binding is None and c_receipt_binding is None,'Negative C lock forbids collection/fallback')
        reasons=admission_decision(None,None)
    else:
        need(b_receipt_binding==plan['b_evidence_reuse']['receipt_binding'] and c_receipt_binding is not None,'Exact native B and actual C fixed160 required')
        bf=native;cf=_verify_c_fixed(plan_binding,plan,contexts['C'],c_receipt_binding,closure)
        need(bf['receipt']['actual_consumed_input_leaf_hashes']==cf['receipt']['actual_consumed_input_leaf_hashes'],'Actual B/C model observation bytes differ')
        reasons=admission_decision(selected,cf['receipt'])
    for item in (bf,cf):
        if item is None:continue
        add_binding(closure,item['receipt_binding']);add_binding(closure,item['execution_binding'])
        add_map(closure,item['receipt']['artifact_sha256'])
        add_binding(closure,item['receipt']['producer_execution_binding']['source_file_binding'])
        execution=read_pinned(item['execution_binding'])
        for key in ('guard_source_binding','guard_evidence_binding'):add_binding(closure,execution[key])
    admitted=not reasons;data=contexts['C']['dataset']
    result=dict(schema=ADMISSION_SCHEMA,status=reasons[0] if reasons else 'simulator_candidate_only',plan_binding=plan_binding,condition_binding=plan['condition_binding'],
        b_selection_lock_binding=plan['b_selection_lock_binding'],c_selection_lock_binding=plan['c_selection_lock_binding'],b_comparator_kind=plan['b_comparator_kind'],
        b_checkpoint_binding=plan['b_checkpoint_binding'],c_checkpoint_binding=plan['c_checkpoint_binding'],
        b_fixed160_receipt_binding=None if bf is None else bf['receipt_binding'],c_fixed160_receipt_binding=None if cf is None else cf['receipt_binding'],
        b_fixed160_execution_binding=None if bf is None else bf['execution_binding'],c_fixed160_execution_binding=None if cf is None else cf['execution_binding'],
        primary_final14600=primary,all13_step_matched_comparison=curves,secondary_locked_comparison=secondary,
        fixed160_descriptive_comparison=None if cf is None else dict(B=dict(status=bf['receipt']['status'],bounds=bf['receipt']['bounds'],metrics=bf['receipt']['metrics']),C=dict(status=cf['receipt']['status'],bounds=cf['receipt']['bounds'],metrics=cf['receipt']['metrics']),affects_admission=False),
        c_locked120_bounds_and_moving_gates=None if selected is None else dict(bounds=selected['bounds'],moving_buckets=selected['moving_buckets']),
        c_fixed160_bounds=None if cf is None else cf['receipt']['bounds'],rejection_reasons=reasons,artifact_sha256=closure,
        candidate_admitted=admitted,service_eligible=admitted,admitted_checkpoint_binding=plan['c_checkpoint_binding'] if admitted else None,
        admitted_dataset_codec_binding=dict(dataset_binding=D.dataset_binding(data),normalization_sha256=digest(data.normalization),action_encoding_binding=data.normalization['action_encoding_binding']) if admitted else None,
        no_reselection=True,physical_accepted=False,closed_loop_validated=False,generalization_validated=False,authorization_granted=False)
    _finish_checks(plan,contexts,closure,[bf,cf]);return result


def _reconstruct_final(plan_binding,b_receipt_binding,c_receipt_binding):
    plan,contexts,locks,native=_verify_plan(plan_binding)
    return _final(plan_binding,plan,contexts,locks,native,b_receipt_binding,c_receipt_binding)


def finalize_comparison(plan_binding,b_receipt_binding,c_receipt_binding,output):
    plan,contexts,locks,native=_verify_plan(plan_binding)
    result=_final(plan_binding,plan,contexts,locks,native,b_receipt_binding,c_receipt_binding)
    extra=[Path(plan_binding['path']).parent]+[Path(b['path']).parent for b in (b_receipt_binding,c_receipt_binding) if b is not None]
    out=fresh_output(output,result['artifact_sha256'],protected_trees=[*_protected(plan),*extra])
    _finish_checks(plan,contexts,result['artifact_sha256'],[])  # Live training markers after output validation.
    # Collection cancellation must also be checked at the final publication boundary.
    if c_receipt_binding is not None:
        r=read_pinned(c_receipt_binding);verify_execution(c_receipt_binding,r['producer_execution_binding'],result['artifact_sha256'])
    recheck(result['artifact_sha256'])
    current=[]
    if c_receipt_binding is not None:
        current=[dict(execution_binding=dict(path=str(Path(c_receipt_binding['path']).with_name('execution.json')),sha256=result['artifact_sha256'][str(Path(c_receipt_binding['path']).with_name('execution.json'))]))]
    _late_markers(plan,contexts,current);return write_immutable(out/'admission.json',result)


def verify_comparison_admission(receipt_path,expected_sha256):
    identity=dict(path=str(receipt_path),sha256=expected_sha256);saved=read_pinned(identity)
    need(saved.get('schema')==ADMISSION_SCHEMA,'New conditional C admission schema required')
    result=_reconstruct_final(saved['plan_binding'],saved['b_fixed160_receipt_binding'],saved['c_fixed160_receipt_binding'])
    need(result==saved,'Admission differs from complete evidence reconstruction');bound(identity);return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','condition','c-lock','c-lifecycle','producer-inventory','b-receipt','c-receipt','admission'):
        parser.add_argument('--'+name);parser.add_argument('--'+name+'-sha256')
    parser.add_argument('--output');parser.add_argument('--role',choices=['C_candidate']);action=parser.add_mutually_exclusive_group()
    for name in ('collect','prepare','finalize'):action.add_argument('--'+name,action='store_true')
    args=parser.parse_args(argv)
    def identity(name,optional=False):
        key=name.replace('-','_');path,h=getattr(args,key),getattr(args,key+'_sha256')
        if optional and path is None and h is None:return None
        need(path is not None and h is not None,'Explicit '+name+' path and expected SHA required');return dict(path=path,sha256=h)
    if args.collect:
        need(args.output and args.role=='C_candidate','Explicit C collection output/role required');result=collect_fixed160(identity('plan'),args.role,args.output)
    elif args.prepare:
        need(args.output,'Fresh plan output required');result=prepare_comparison(identity('condition'),identity('c-lock'),args.output,c_lifecycle_binding=identity('c-lifecycle'),producer_inventory_binding=identity('producer-inventory'),b_fixed160_binding=identity('b-receipt',True))
    elif args.finalize:
        need(args.output,'Fresh admission output required');result=finalize_comparison(identity('plan'),identity('b-receipt',True),identity('c-receipt',True),args.output)
    elif args.admission:
        need(args.output is None and args.role is None,'CPU verification cannot write/collect');i=identity('admission');result=verify_comparison_admission(i['path'],i['sha256'])
    else:
        need(args.output is None and args.role is None,'CPU verification cannot write/collect');p,_,_,_=_verify_plan(identity('plan'));result=dict(status=p['status'],verified=True,model_initialized=False)
    print(json.dumps(result,sort_keys=True,allow_nan=False));return result


if __name__=='__main__':main()
