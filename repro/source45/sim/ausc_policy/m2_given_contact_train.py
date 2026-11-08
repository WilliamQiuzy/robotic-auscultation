"""One finite official given-contact ACT run. No resume or TEST route."""
import argparse
from copy import deepcopy
import json
import hashlib
import stat
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np

from . import m2_given_contact_runtime as runtime

need=runtime.need
CONFIG_SCHEMA='m2_given_contact_formal_act_config_v1'


def validate_config(config):
    need(set(config)=={'schema','lerobot_root','device','task_contract','model','numerics','fixture_spec',
        'dataset','processing','runtime_qualification'} and config.get('schema')==CONFIG_SCHEMA and
        config.get('lerobot_root')==runtime.VENDOR_ROOT and config.get('device')=='cuda' and
        config.get('task_contract')==runtime.task_contract() and config.get('model')==runtime.shared.native_spec() and
        config.get('numerics')==runtime.numerical_protocol() and config.get('fixture_spec')==runtime.fixture_spec(),
        'One exact reviewed formal schedule/model/config required; no resume or device substitution')
    refs=config.get('dataset',{})
    need(set(refs)=={'plan','seal','manifest','qualification','reconstruction'}, 'Only sealed TRAIN/VAL dataset references are permitted')
    for ref in [*refs.values(),config.get('runtime_qualification')]:runtime._reference_shape(ref)
    binding=config.get('processing',{})
    need(set(binding)=={'plan','processing_freeze'} and binding['plan']==refs['plan'], 'Same original plan and final processing freeze required')
    runtime._reference_shape(binding['processing_freeze'])
    return config


def window_order(prepared):
    rows=prepared['derived_index']['samples'];result={}
    need(all(row.get('split') in ('train','validation') for row in rows), 'TEST/quarantine or unknown derived row refused')
    for split in ('train','validation'):
        selected=[r for r in rows if r['split']==split]
        need([r.get('sample_index') for r in selected]==list(range(len(prepared[split]['states']))),
             'Complete original ordered inference windows required')
        result[split]=runtime.digest(selected)
    return result


class _InputClosure:
    """Private fixed baseline. Terminal verification never discovers new refs."""
    __slots__=('_declarations','_baseline')

    def __init__(self,roots,refs,*,expected_trees=()):
        refs=list(refs)
        for role,root,inventory in expected_trees:
            need(isinstance(inventory,dict) and inventory, 'Original exhaustive sealed/capture inventory required')
            for name,binding in inventory.items():
                need(isinstance(name,str) and name and name!='.' and not Path(name).is_absolute() and
                     '..' not in Path(name).parts and Path(name).as_posix()==name and
                     isinstance(binding,dict) and set(binding)=={'bytes','sha256'} and
                     type(binding['bytes']) is int and binding['bytes']>=0, 'Exact original relative artifact path/bytes/SHA256 required')
                ref=dict(binding,path=str(runtime.dataset._path(name,Path(root))))
                runtime._reference_shape(ref);refs.append((role,ref,False))
        self._declarations=runtime.pose.canonical(dict(roots=roots,refs=refs,expected_trees=expected_trees))
        try:self._baseline=runtime.pose.canonical(self._snapshot())
        except OSError as error:raise ValueError('Original readonly inventory missing or changed: '+str(error)) from error

    def _snapshot(self):
        declarations=json.loads(self._declarations);entries={};roles={};allow={}
        for role,ref,linked in declarations['refs']:
            need(not linked or role in ('sdk','other_runtime'), 'Only authenticated installed SDK/runtime link exception permitted')
            path=runtime.dataset._path(ref['path'],Path.cwd());key=str(path)
            roles.setdefault(key,[]).append(dict(role=role,reference=ref))
            allow[key]=allow.get(key,False) or linked
        def visit(path,closed=False):
            path=runtime.dataset._path(str(path),Path.cwd());key=str(path)
            if key in entries:
                need(not closed or entries[key]['kind']=='directory' or entries[key]['nlink']==1,
                     'Closed source/metadata/capture trees cannot use runtime link exceptions')
                return
            info=path.lstat()
            common=dict(device=info.st_dev,inode=info.st_ino,mode=info.st_mode,nlink=info.st_nlink)
            if stat.S_ISDIR(info.st_mode):
                entries[key]=dict(common,kind='directory')
                for child in sorted(path.iterdir()):visit(child,closed=True)
            else:
                need(stat.S_ISREG(info.st_mode) and (info.st_nlink==1 or (not closed and allow.get(key,False))),
                     'Linked or special readonly input refused: '+key)
                descriptor=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
                with os.fdopen(descriptor,'rb') as stream:
                    before=os.fstat(stream.fileno());digest=hashlib.sha256()
                    for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
                    after=os.fstat(stream.fileno())
                def stable(item):return (item.st_dev,item.st_ino,item.st_mode,item.st_nlink,item.st_size,item.st_mtime_ns,item.st_ctime_ns)
                need(stable(before)==stable(after)==stable(path.lstat())==stable(info),
                     'Readonly input changed while hashing: '+key)
                entries[key]=dict(common,kind='file',bytes=info.st_size,sha256=digest.hexdigest())
        for paths in declarations['roots'].values():
            for root in paths:
                path=runtime.dataset._path(root,Path.cwd());need(path.is_dir(), 'Declared closed input tree missing')
                visit(path,closed=True)
        for key,bindings in roles.items():
            visit(Path(key));actual=entries[key]
            need(actual['kind']=='file' and all(actual['sha256']==b['reference']['sha256'] and
                 ('bytes' not in b['reference'] or actual['bytes']==b['reference']['bytes']) for b in bindings),
                 'Fixed input role/path/size/SHA256 changed: '+key)
        for role,root,inventory in declarations['expected_trees']:
            root=runtime.dataset._path(root,Path.cwd())
            expected_files={str(root/name) for name in inventory}
            expected_dirs={str(root)}|{str(p) for name in inventory for p in (root/name).parents if p==root or root in p.parents}
            actual_files={p for p,row in entries.items() if root in Path(p).parents and row['kind']=='file'}
            actual_dirs={p for p,row in entries.items() if (Path(p)==root or root in Path(p).parents) and row['kind']=='directory'}
            need(actual_files==expected_files and actual_dirs==expected_dirs,
                 'Original exhaustive '+role+' file/directory path set changed before baseline')
        return dict(roots=declarations['roots'],roles=roles,entries=entries)

    def assert_output(self,output):
        output=runtime.dataset._path(str(output),Path.cwd());value=json.loads(self._declarations)
        inputs=[Path(p) for paths in value['roots'].values() for p in paths]+[Path(r['path']) for _,r,_ in value['refs']]
        need(all(output!=p and output not in p.parents and p not in output.parents for p in inputs),
             'Current output overlaps a readonly input file/tree')

    def verify(self):
        try:current=runtime.pose.canonical(self._snapshot())
        except (OSError,KeyError,TypeError) as error:raise ValueError('Readonly input closure changed: '+str(error)) from error
        need(current==self._baseline, 'Fixed readonly input tree, inode, role or content changed')
        rows=json.loads(self._baseline)['entries']
        return dict(passed=True,inventory_sha256=hashlib.sha256(self._baseline).hexdigest(),
            file_count=sum(row['kind']=='file' for row in rows.values()),
            bytes=sum(row.get('bytes',0) for row in rows.values()))

    def inventory(self):return json.loads(self._baseline)


def _declared_refs(value,role,base):
    """Walk only a chosen authenticated document, never follow its JSON refs."""
    result=[]
    if isinstance(value,dict):
        name=value.get('path',value.get('file'))
        if isinstance(name,str) and isinstance(value.get('sha256'),str):
            path=runtime.dataset._path(name,base)
            ref=dict(path=str(path),sha256=value['sha256'])
            if 'bytes' in value:ref['bytes']=value['bytes']
            result.append((role,ref,role in ('sdk','other_runtime')))
        else:
            for item in value.values():result.extend(_declared_refs(item,role,base))
    elif isinstance(value,list):
        for item in value:result.extend(_declared_refs(item,role,base))
    return result


def _freeze_inputs(config_ref,config,prepared):
    """Called only after this process's full public seal+dataset reconstruction."""
    plan=runtime.read_bound(config['dataset']['plan']);base=Path(config['dataset']['plan']['path']).parent
    scope=plan['accounting_scope'];processing=scope['processing_sources'];historical=processing['original_execution_freeze']
    seal=runtime.read_bound(config['dataset']['seal'])
    expected_trees=[('sealed_metadata',scope['operational_metadata']['root'],seal.get('metadata_inventory'))]
    need(len(plan['attempts'])==240 and len(scope['owner_roots'])==3, 'Complete fixed population input scope required')
    roots=dict(sealed_metadata=[scope['operational_metadata']['root']],collection_owners=scope['owner_roots'],
        configurations=[str(runtime.bound_path(plan['config_inventory']).parent)],collector=[plan['collector']['root']],
        final_sources=[processing['live_root'],processing['archive_root']],historical_sources=[historical['root']],
        captures=[],qualified_smoke_owners=[])
    refs=_declared_refs(config,'fit_config',base)+[('fit_config',config_ref,False)]
    # No storage_roots/CPU-bank traversal. These are the authenticated finite
    # plan references, not a recursive crawl of arbitrary referenced documents.
    for key in ('allocation_proposal','allocation_mapping','source_recipe','qualification','config_inventory',
                'previous_owner','phase_batches','collector'):
        refs.extend(_declared_refs(plan[key],'plan_'+key,base))
    refs.extend(_declared_refs({k:v for k,v in historical.items() if k!='environment'},'source',base))
    refs.extend(_declared_refs(historical.get('environment',{}),'other_runtime',base))
    for entry in scope['extra_input_refs']:
        role=entry['category'];need(role in ('source','config','sdk','prepared','other_runtime'), 'Unknown frozen input role')
        refs.extend(_declared_refs(entry['reference'],role,base))
    account=prepared['qualification']['accounting']
    need(account.get('all_accounted_jobs_within_budget') is True, 'Qualified complete shared accounting required')
    for ref in account['unique_union']['shared_inputs']:
        role='sdk' if any(k.startswith('sdk_mdl_') for k in ref.get('kinds',[])) else 'prepared'
        refs.extend(_declared_refs(ref,role,base))
    refs.extend(_declared_refs(account.get('additional_inputs',[]),'prepared',base))
    rows=runtime.read_bound(config['dataset']['manifest'])['episodes']
    smokes=runtime.read_bound(plan['qualification'])['episodes']
    need(len(smokes)==5, 'Exactly five actual qualified collector smoke inputs required')
    roots['qualified_smoke_owners']=[str(runtime.bound_path(row['owner_terminal']).parent) for row in smokes]
    def sources(row):
        refs.extend(_declared_refs(row,'episode_receipt',base))
        freeze=runtime.read_bound(row['source_freeze'])
        refs.extend(_declared_refs(freeze,'source',Path(row['source_freeze']['path']).parent))
        for original,archived in freeze['files'].items():
            refs.append(('original_source',dict(path=original,bytes=archived['bytes'],sha256=archived['sha256']),False))
        value=runtime.read_bound(row['config'])
        storage=value.get('compact_scene_storage',{}).get('sources',{})
        for name,ref in storage.items():refs.extend(_declared_refs(ref,'sdk' if name.startswith('sdk_mdl_') else 'prepared',base))
        value=deepcopy(value)
        if 'compact_scene_storage' in value:value['compact_scene_storage'].pop('sources',None)
        refs.extend(_declared_refs(value,'config_input',base))
    for row in [*plan['attempts'],*smokes]:sources(row)
    for row in rows:
        need(row.get('split') in ('train','validation'), 'TEST/quarantine input closure refused')
        roots['captures'].append(row['capture_root']);refs.extend(_declared_refs(row,'episode_receipt',base))
    for row in [*rows,*smokes]:
        root=str(runtime.dataset._path(row['capture_root'],base))
        expected_trees.append(('capture_artifact',root,row.get('artifacts')))
    return _InputClosure(roots,refs,expected_trees=expected_trees)


def _freeze_model_inputs(model,complete,*,model_ref=None,owner_terminal=None):
    """Completed model/owner inputs, excluding current admission outputs."""
    refs=_declared_refs(complete,'completed_model',Path.cwd())+_declared_refs(model,'model',Path.cwd())
    if model_ref is not None:refs.append(('model_declaration',model_ref,False))
    roots={}
    start=complete.get('owned_stage')
    if start is not None:
        terminal=runtime.reference(runtime.bound_path(start['run']).parent/'result.json')
        need(owner_terminal is None or runtime.same_reference(owner_terminal,terminal), 'Completed model uses another original owner terminal')
        owner_terminal=terminal
        runtime.verify_owned_stage(terminal,complete_ref=model['complete'],
            child_identity=runtime.read_bound(complete['resources'])['identity'],experiment='m2_given_contact_train')
    if owner_terminal is not None:
        refs.append(('training_owner_terminal',owner_terminal,False))
        roots['completed_training_owner']=[str(runtime.bound_path(owner_terminal).parent)]
    return _InputClosure(roots,refs)


def postflight(pre,output,*,model_inputs=None):
    """Rebind fixed bytes and live prepared arrays; do not decode again."""
    closure=pre['_input_closure'];closure.assert_output(output)
    if model_inputs is not None:model_inputs.assert_output(output)
    baseline=json.loads(pre['_initial_state'])
    need(pre['config']==baseline['config'] and pre['fingerprint']==baseline['fingerprint'] and
         pre['window_order']==baseline['window_order'], 'Initial preflight baseline mutated')
    fingerprint=runtime.dataset._fingerprint(pre['prepared']);orders=window_order(pre['prepared'])
    need(fingerprint==baseline['fingerprint'] and orders==baseline['window_order'] and
         runtime.given.fit_normalization(pre['prepared']['train'],split='train')==pre['prepared']['normalization'],
         'Prepared arrays, reference inventory, window order or TRAIN normalization changed')
    source=runtime.processing_audit(pre['config']['processing'])
    need(runtime.source_identity(source)==baseline['processing_source_identity'], 'Terminal processing sources changed')
    fixture=runtime.verify_fixture(pre['config']['runtime_qualification'])
    need(fixture==baseline['runtime_qualification'], 'Terminal fixture arrays/weights/owner binding changed')
    # One complete terminal SHA pass over the fixed baseline, after all model
    # and software checks. No terminal declaration can extend this inventory.
    audit=dict(closure.verify(),fingerprint=fingerprint,window_order=orders,
        processing_audit=source,full_decode_repeated=False)
    if model_inputs is not None:audit['completed_model']=model_inputs.verify()
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    audit['fixed_inventory']=runtime.write_new(output/'readonly_inputs.json',closure.inventory())
    if model_inputs is not None:audit['completed_model_inventory']=runtime.write_new(output/'completed_model_inputs.json',model_inputs.inventory())
    audit['evidence']=runtime.write_new(output/'input_rebinding.json',audit)
    return audit


def preflight(config_ref):
    """Always reconstruct from the new original archives, never a caller cache."""
    config=validate_config(runtime.read_bound(config_ref));refs=config['dataset']
    seal_audit=runtime.dataset.verify_final_seal(refs['plan'],refs['seal'])
    seal=runtime.read_bound(refs['seal'])
    need(seal_audit.get('passed') is True and seal_audit.get('training_eligible') is True and
         seal_audit.get('fingerprint')==refs['reconstruction'] and
         all(seal.get(key)==refs[key] for key in ('manifest','qualification')) and
         seal.get('fingerprint')==refs['reconstruction'], 'Exact original qualified final seal/bundle required')
    prepared=runtime.dataset.prepare_dataset(refs['manifest']['path'],refs['manifest']['sha256'],refs['qualification'])
    need(prepared.get('training_eligible') is True and prepared.get('qualification',{}).get('passed') is True and
         prepared.get('test_payloads_opened') is False and prepared.get('task_contract')==runtime.task_contract(),
         'Fresh qualified native TRAIN/VAL reconstruction/common contract required')
    fingerprint=runtime.read_bound(refs['reconstruction'])
    need(runtime.dataset._fingerprint(prepared)==fingerprint, 'Sealed arrays/index/counts/stats/current-source fingerprint changed')
    runtime.validate_normalization(prepared['normalization'])
    need(runtime.given.fit_normalization(prepared['train'],split='train')==prepared['normalization'], 'TRAIN normalization changed')
    fixture=runtime.verify_fixture(config['runtime_qualification'])
    need(fixture.get('passed') is True and fixture.get('processing')==config['processing'], 'Same frozen official numerical runtime qualification required')
    source=runtime.processing_audit(config['processing'])
    need(runtime.read_bound(config_ref)==config, 'Fit config changed during reconstruction')
    pre=dict(config=config,prepared=prepared,fingerprint=fingerprint,seal_audit=seal_audit,
        runtime_qualification=fixture,processing_audit=source,window_order=window_order(prepared))
    pre['_input_closure']=_freeze_inputs(config_ref,config,prepared)
    pre['_initial_state']=runtime.pose.canonical(dict(config=config,fingerprint=fingerprint,window_order=pre['window_order'],
        processing_source_identity=runtime.source_identity(source),runtime_qualification=fixture))
    return pre


def _rng_state(torch,rng,augmentation):
    state=np.random.get_state()
    return dict(minibatch=deepcopy(rng.bit_generator.state),augmentation=deepcopy(augmentation.bit_generator.state),
        python=random.getstate(),numpy=dict(algorithm=state[0],keys=state[1].tolist(),position=state[2],
            has_gauss=state[3],cached_gaussian=state[4]),torch_cpu=torch.get_rng_state(),
        torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def _save_checkpoint(torch,path,value):
    runtime.validate_checkpoint(value)
    with Path(path).open('xb') as stream:torch.save(value,stream)
    return runtime.reference(path)


def _checkpoint(policy,prepared,inputs,step,metric):
    return dict(schema=runtime.given.CHECKPOINT_SCHEMA,formal_run_schema=runtime.FORMAL_RUN_SCHEMA,
        upstream_commit=runtime.UPSTREAM_COMMIT,model=runtime.shared.native_spec(),task_contract=runtime.task_contract(),
        step=step,selection_split='validation',state_only=True,action_representation='block_relative_joint',
        cpu_fixture_only=False,normalization=prepared['normalization'],reference_inventory=prepared['reference_inventory'],
        inputs=inputs,metrics=metric,state_dict=policy.state_dict(),closed_loop_evaluated=False)


def _optimize(torch,policy,prepared,inputs,output,orders,check,log,*,device):
    """Private loop; public fit fixes CUDA, ownership and all 20,000 steps."""
    train=prepared['train'];validation=prepared['validation'];stats=prepared['normalization']
    rng=np.random.Generator(np.random.PCG64(runtime.MODEL_SEED))
    augmentation=np.random.Generator(np.random.PCG64(20261027))
    optimizer=torch.optim.AdamW(policy.parameters(),lr=1e-4,weight_decay=1e-4,betas=(.9,.999),eps=1e-8)
    evaluations=[];step=0;completed=0;last_metric=None
    try:
        for step in range(1,20001):
            check();policy.train();ids=rng.integers(len(train['states']),size=64)
            batch=runtime.given.pack_batch(train,stats,ids,training=True,rng=augmentation,q_noise_std=.002,q_dropout=.05)
            tensors={key:torch.from_numpy(value).to(device) for key,value in batch.items()}
            optimizer.zero_grad(set_to_none=True);loss,_=policy(tensors)
            need(loss.ndim==0 and bool(torch.isfinite(loss)), 'Nonfinite/non-scalar official ACT loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            completed=step
            log.write(json.dumps(dict(step=step,train_loss=float(loss.detach().cpu()),grad_norm=float(norm)),allow_nan=False)+'\n')
            if step not in runtime.EVAL_STEPS:continue
            prediction=runtime.infer(torch,policy,validation,stats,device=device)
            last_metric=runtime.metrics(prediction,validation)
            row=dict(step=step,metrics=last_metric,predictions=runtime.save_predictions(output/('val-'+str(step)+'.npy'),
                prediction,window_order_sha256=orders['validation']))
            row['checkpoint']=_save_checkpoint(torch,output/('step-'+str(step)+'.pt'),
                _checkpoint(policy,prepared,inputs,step,last_metric))
            evaluations.append(row);log.write(json.dumps(dict(step=step,split='validation',metrics=last_metric),allow_nan=False)+'\n');log.flush()
        # Save the exact terminal optimizer and RNG before loading selected weights.
        packet=_checkpoint(policy,prepared,inputs,20000,last_metric)
        packet['training_state']=dict(completed_steps=20000,optimizer=optimizer.state_dict(),rng=_rng_state(torch,rng,augmentation))
        last=_save_checkpoint(torch,output/'last.pt',packet)
        return evaluations,last
    except BaseException as error:
        # This state is explicit partial evidence, never a formal checkpoint or resume input.
        partial=dict(schema='m2_given_contact_formal_act_partial_v1',completed_steps=completed,
            state_dict=policy.state_dict(),optimizer=optimizer.state_dict(),rng=_rng_state(torch,rng,augmentation),
            inputs=inputs,error=type(error).__name__+': '+str(error),formal_complete=False,automatic_resume=False)
        with (output/'partial.pt').open('xb') as stream:torch.save(partial,stream)
        raise


def fit(stage_config_ref,output):
    started=time.monotonic();output=Path(output).resolve()
    need(not output.exists(), 'One new immutable training run directory required')
    owned_stage=runtime.start_owned_stage(stage_config_ref,output,experiment='m2_given_contact_train')
    config_ref=owned_stage['input']
    with runtime.owned_gpu() as ownership:
        pre=preflight(config_ref);config=pre['config'];prepared=pre['prepared']
        pre['_input_closure'].assert_output(output)
        torch,policy,software=runtime.make_model(config,device='cuda')
        execution=runtime.execution_settings(torch,policy,device='cuda')
        output.mkdir(parents=True,exist_ok=False)
        inputs=dict(dataset=deepcopy(config['dataset']),config=config_ref,
            software=runtime.write_new(output/'software.json',software),
            normalization=runtime.write_new(output/'normalization.json',prepared['normalization']),
            reference_inventory=runtime.write_new(output/'reference_inventory.json',prepared['reference_inventory']))
        from . import process_guard as guard
        identity=runtime._identity();need(identity['pgid']==identity['pid']==identity['sid'], 'Original guard-owned training session required')
        resource_rows=[];sampled=[-float('inf')]
        def check():
            elapsed=time.monotonic()-started
            need(elapsed<21600, 'Six-hour finite training deadline reached')
            if time.monotonic()-sampled[0]>=10:
                sample=guard.sample_resources(output,identity);sampled[0]=time.monotonic();resource_rows.append(sample)
                violation=guard.guard_violation(sample,elapsed,21600);need(violation is None,violation or '')
        try:
            with (output/'metrics.jsonl').open('x') as log:
                evaluations,last=_optimize(torch,policy,prepared,inputs,output,pre['window_order'],check,log,device='cuda')
            selected=runtime.earliest_argmin(evaluations)
            chosen=next(row for row in evaluations if row['step']==selected)
            with runtime.bound_path(chosen['checkpoint']).open('rb') as source,(output/'best.pt').open('xb') as destination:
                shutil.copyfileobj(source,destination)
            best=runtime.reference(output/'best.pt')
            need(best['sha256']==chosen['checkpoint']['sha256'], 'Best copy differs from exact CUDA-selected snapshot')
            packet=runtime.validate_checkpoint(torch.load(runtime.bound_path(chosen['checkpoint']),map_location='cpu',weights_only=True))
            policy.load_state_dict(packet['state_dict'],strict=True)
            predictions={};metrics={}
            for split in ('train','validation'):
                check();value=runtime.infer(torch,policy,prepared[split],prepared['normalization'],device='cuda')
                predictions[split]=runtime.save_predictions(output/('selected-'+split+'.npy'),value,window_order_sha256=pre['window_order'][split])
                metrics[split]=runtime.metrics(value,prepared[split])
            after=postflight(pre,output)
            runtime.validate_software(software,runtime.software_inventory(runtime.VENDOR_ROOT));check()
            resources=runtime.write_new(output/'resources.json',dict(owner=ownership,identity=identity,samples=resource_rows,
                elapsed_wall_s=time.monotonic()-started,maximum_wall_s=21600,guard_source=runtime.reference(guard.__file__)))
            complete=dict(schema=runtime.FORMAL_RUN_SCHEMA,completed_steps=20000,complete=True,source_audit_passed=True,
                cpu_fixture_only=False,test_payloads_opened=False,task_contract=runtime.task_contract(),inputs=inputs,
                evaluations=evaluations,selected_step=selected,best=best,last=last,selected_predictions=predictions,
                selected_metrics=metrics,window_order=pre['window_order'],processing=config['processing'],
                numerical_protocol=runtime.numerical_protocol(),resources=resources,owned_stage=owned_stage,
                input_rebinding=after['evidence'],execution=execution,training_log=runtime.reference(output/'metrics.jsonl'),
                closed_loop_evaluated=False,physical_acceptance=False,
                admission_pending=True,training_eligible_model=False)
            runtime.validate_complete(complete);runtime.write_new(output/'COMPLETE.json',complete)
            return complete
        except BaseException as error:
            runtime.write_new(output/'PARTIAL.json',dict(schema=runtime.FORMAL_RUN_SCHEMA,complete=False,
                inputs=inputs,error=type(error).__name__+': '+str(error),elapsed_wall_s=time.monotonic()-started,
                resources=resource_rows,test_payloads_opened=False,automatic_resume=False));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--config-sha256')
    parser.add_argument('--output',required=True);args=parser.parse_args()
    config_ref=runtime.reference(args.config)
    if args.config_sha256:need(config_ref['sha256']==args.config_sha256, 'Original stage config SHA256 differs')
    result=fit(config_ref,args.output)
    print(json.dumps(dict(completed_steps=result['completed_steps'],selected_step=result['selected_step'],admission_pending=True)))


if __name__=='__main__':main()
