"""Independent full-population CPU admission for one completed formal ACT run."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from . import m2_given_contact_runtime as runtime, m2_given_contact_train as train

need=runtime.need
WORKER_SCHEMA='m2_given_contact_formal_cpu_reconstruction_v1'
WORKER_TIMEOUT_S=21600
MODULE='sim.ausc_policy.m2_given_contact_admission'


def _predictions(result,prepared,orders):
    need(set(result.get('selected_predictions',{}))==set(result.get('selected_metrics',{}))=={'train','validation'},
         'Complete selected TRAIN and VAL reconstruction required')
    arrays={}
    for split in ('train','validation'):
        value=runtime.load_predictions(result['selected_predictions'][split],window_order_sha256=orders[split])
        need(len(value)==len(prepared[split]['states']) and runtime.metrics(value,prepared[split])==result['selected_metrics'][split],
             'Full selected CPU/CUDA arrays/counts/metrics do not reconstruct')
        arrays[split]=value
    return arrays


def compare_cpu_results(model,complete,prepared,workers):
    """Reopen every canonical prediction; tolerances never change admission."""
    need(len(workers)==2, 'Exactly two fresh independent CPU reconstructions required')
    orders=train.window_order(prepared)
    need(orders==complete.get('window_order'), 'Original full TRAIN/VAL window order changed')
    for worker in workers:
        need(worker.get('schema')==WORKER_SCHEMA and worker.get('model')==model and worker.get('inputs')==complete['inputs'] and
             worker.get('device')=='cpu' and worker.get('window_order')==orders and worker.get('source_audit_passed') is True,
             'Strict selected-model fresh CPU provenance required; fixtures are refused')
        runtime.validate_execution_settings(worker.get('execution',{}),device='cpu')
    need(workers[0]['identity']['pid']!=workers[1]['identity']['pid'] and workers[0]['identity']!=workers[1]['identity'],
         'Selected weights must be reconstructed in two fresh CPU processes')
    cpu=[_predictions(worker,prepared,orders) for worker in workers]
    cuda=_predictions(complete,prepared,orders)
    for split in ('train','validation'):
        need(cpu[0][split].tobytes()==cpu[1][split].tobytes() and
             workers[0]['selected_metrics'][split]==workers[1]['selected_metrics'][split],
             'Fresh CPU prediction arrays and full metrics must be byte-identical')
    rows=workers[0].get('evaluations');cpu_selected=runtime.earliest_argmin(rows)
    need(cpu_selected==complete['selected_step']==runtime.earliest_argmin(complete['evaluations']),
         'CPU/CUDA earliest argmin differs; device substitution/reselection refused')
    snapshot_comparisons=[]
    for cpu_row,cuda_row in zip(rows,complete['evaluations']):
        need(cpu_row.get('checkpoint')==cuda_row['checkpoint'], 'CPU VAL snapshot weight identity/order changed')
        runtime.bound_path(cpu_row['checkpoint'])
        a=runtime.load_predictions(cpu_row['predictions'],window_order_sha256=orders['validation'])
        b=runtime.load_predictions(cuda_row['predictions'],window_order_sha256=orders['validation'])
        need(len(a)==len(b)==len(prepared['validation']['states']) and
             runtime.metrics(a,prepared['validation'])==cpu_row['metrics'] and
             runtime.metrics(b,prepared['validation'])==cuda_row['metrics'], 'All forty full VAL metrics must reconstruct')
        snapshot_comparisons.append(dict(step=cpu_row['step'],**runtime.compare_devices(a,b,cpu_row['metrics'],cuda_row['metrics'])))
    selected_comparisons={split:runtime.compare_devices(cpu[0][split],cuda[split],workers[0]['selected_metrics'][split],
        complete['selected_metrics'][split]) for split in ('train','validation')}
    selected_row=next(row for row in rows if row['step']==cpu_selected)
    selected_val=runtime.load_predictions(selected_row['predictions'],window_order_sha256=orders['validation'])
    need(selected_val.tobytes()==cpu[0]['validation'].tobytes(), 'CPU selected VAL differs from the same exact snapshot reconstruction')
    cpu_gate=runtime.admission_gate(workers[0]['selected_metrics']['train'])
    cuda_gate=runtime.admission_gate(complete['selected_metrics']['train'])
    flags=('passed','beats_hold_20_percent','beats_cv_20_percent')
    need(all(cpu_gate[key]==cuda_gate[key] for key in flags), 'CPU/CUDA exact TRAIN admission gate booleans disagree')
    return dict(passed=cpu_gate['passed'],cpu_byte_identical=True,cpu_selected_step=cpu_selected,
        train_gate=cpu_gate,cuda_train_gate=cuda_gate,selected_comparisons=selected_comparisons,
        snapshot_comparisons=snapshot_comparisons,metrics=workers[0]['selected_metrics'],
        validation_comparisons_reported=True,test_payloads_opened=False)


def _load_snapshot(torch,policy,row,inputs,prepared):
    packet=runtime.validate_checkpoint(torch.load(runtime.bound_path(row['checkpoint']),map_location='cpu',weights_only=True))
    need(packet['inputs']==inputs and packet['step']==row['step'] and packet['metrics']==row['metrics'] and
         packet['normalization']==prepared['normalization'] and packet['reference_inventory']==prepared['reference_inventory'],
         'Snapshot source/normalization/reference/selection binding differs')
    policy.load_state_dict(packet['state_dict'],strict=True)


def cpu_worker(model_ref,output,*,all_snapshots):
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'Independent CPU worker requires CUDA hidden')
    model=runtime.read_bound(model_ref);complete=runtime.validate_complete(runtime.read_bound(model['complete']))
    pre=train.preflight(complete['inputs']['config']);prepared=pre['prepared']
    need(complete['inputs']['dataset']==pre['config']['dataset'] and complete['processing']==pre['config']['processing'] and
         complete['window_order']==pre['window_order'], 'Completed run uses different sealed data/source/window order')
    model_inputs=train._freeze_model_inputs(model,complete,model_ref=model_ref)
    pre['_input_closure'].assert_output(output);model_inputs.assert_output(output)
    torch,policy,checkpoint,loaded=runtime.load_checkpoint(model,device='cpu')
    execution=runtime.execution_settings(torch,policy,device='cpu')
    runtime.validate_execution_settings(complete.get('execution',{}),device='cuda')
    need(complete.get('numerical_protocol')==runtime.numerical_protocol(), 'Completed run numerical protocol differs')
    runtime.bound_path(complete['last']);runtime.bound_path(complete['training_log'])
    need(loaded==complete and checkpoint['normalization']==prepared['normalization'] and
         checkpoint['reference_inventory']==prepared['reference_inventory'], 'Strict selected weights do not match full reconstruction')
    output=Path(output).resolve();need(not output.exists(), 'Fresh independent CPU worker directory required');output.mkdir(parents=True)
    evaluations=[]
    if all_snapshots:
        for row in complete['evaluations']:
            _load_snapshot(torch,policy,row,complete['inputs'],prepared)
            value=runtime.infer(torch,policy,prepared['validation'],prepared['normalization'],device='cpu')
            evaluations.append(dict(step=row['step'],checkpoint=row['checkpoint'],metrics=runtime.metrics(value,prepared['validation']),
                predictions=runtime.save_predictions(output/('val-'+str(row['step'])+'.npy'),value,
                    window_order_sha256=pre['window_order']['validation'])))
        runtime.earliest_argmin(evaluations)
    selected=next(row for row in complete['evaluations'] if row['step']==complete['selected_step'])
    _load_snapshot(torch,policy,selected,complete['inputs'],prepared)
    predictions={};metrics={}
    for split in ('train','validation'):
        value=runtime.infer(torch,policy,prepared[split],prepared['normalization'],device='cpu')
        predictions[split]=runtime.save_predictions(output/('selected-'+split+'.npy'),value,window_order_sha256=pre['window_order'][split])
        metrics[split]=runtime.metrics(value,prepared[split])
    after=train.postflight(pre,output,model_inputs=model_inputs)
    need(runtime.read_bound(model_ref)==model and runtime.read_bound(model['complete'])==complete, 'Completed model changed during CPU reconstruction')
    runtime.validate_software(runtime.read_bound(complete['inputs']['software']),runtime.software_inventory(runtime.VENDOR_ROOT))
    result=dict(schema=WORKER_SCHEMA,model=model,inputs=complete['inputs'],identity=runtime._identity(),device='cpu',
        window_order=pre['window_order'],selected_predictions=predictions,selected_metrics=metrics,evaluations=evaluations,
        input_rebinding=after['evidence'],source_audit_passed=True,execution=execution,processing=pre['config']['processing'],test_payloads_opened=False)
    runtime.write_new(output/'result.json',result);return result


def _spawn_worker(model_ref,output,*,all_snapshots):
    from . import process_guard as guard
    root=Path(__file__).resolve().parents[2];output=Path(output)
    command=[sys.executable,'-u','-B','-m',MODULE,'--worker','--model',model_ref['path'],
        '--model-sha256',model_ref['sha256'],'--output',str(output)]
    if all_snapshots:command.append('--all-snapshots')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
        PYTHONPATH=str(root),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    for key in ('PYTHONHOME','PYTHONUSERBASE','HF_TOKEN'):env.pop(key,None)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.with_suffix('.log').open('xb') as log:
        process=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        identity=None;valid_session=False
        try:
            identity=guard.process_identity(process.pid);identity['members']={str(process.pid):identity['start_ticks']}
            valid_session=identity['pid']==identity['pgid']==identity['sid']==process.pid
            need(valid_session, 'Independent CPU child session missing')
            started=time.monotonic()
            while process.poll() is None:
                guard.remember_group(identity)
                need(time.monotonic()-started<WORKER_TIMEOUT_S, 'Finite independent CPU reconstruction deadline reached')
                try:process.wait(timeout=1)
                except subprocess.TimeoutExpired:pass
            need(process.returncode==0 and not guard.group_members(identity), 'CPU reconstruction failed or left an owned descendant')
        finally:
            if not valid_session:
                # Direct Popen child is still owned and unreaped. Never signal
                # an unverified process group when startup identity fails.
                for action in (process.terminate,process.kill):
                    if process.poll() is not None:break
                    action()
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:pass
                need(process.poll() is not None, 'CPU startup child could not be reaped')
            elif process.poll() is None or guard.group_members(identity):
                for sig,seconds in ((signal.SIGTERM,5),(signal.SIGKILL,5)):
                    guard.signal_owned_group(process,identity,sig)
                    try:process.wait(timeout=seconds)
                    except subprocess.TimeoutExpired:continue
                    if not guard.group_members(identity):break
            if valid_session:need(process.poll() is not None and not guard.group_members(identity), 'Independent CPU process could not be reaped/drained')
    result_ref=runtime.reference(output/'result.json');result=runtime.read_bound(result_ref)
    need(all(result['identity'][key]==identity[key] for key in ('pid','pgid','sid','start_ticks')), 'CPU result child identity differs')
    return result_ref


def audit_selected_model(model,output,*,cuda_owner_terminal):
    """One external owner terminal plus two fresh observation-only CPU loads."""
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'Admission runs only in the pinned CPU environment')
    complete=runtime.validate_complete(runtime.read_bound(model['complete']))
    runtime.validate_execution_settings(complete.get('execution',{}),device='cuda')
    need(complete.get('numerical_protocol')==runtime.numerical_protocol(), 'Completed run numerical protocol differs')
    need(model.get('checkpoint')==complete['best'], 'Only original CUDA VAL-selected model can be admitted')
    pre=train.preflight(complete['inputs']['config'])
    need(complete['inputs']['dataset']==pre['config']['dataset'] and complete['processing']==pre['config']['processing'],
         'Admission requires original exact final dataset seal and processing root')
    # Implemented with the shared original-run stage verifier; no self-reported
    # device string or child completion can replace the parent's terminal proof.
    owner=runtime.verify_owned_stage(cuda_owner_terminal,complete_ref=model['complete'],
        child_identity=runtime.read_bound(complete['resources'])['identity'],experiment='m2_given_contact_train')
    model_inputs=train._freeze_model_inputs(model,complete,owner_terminal=cuda_owner_terminal)
    pre['_input_closure'].assert_output(output);model_inputs.assert_output(output)
    output=Path(output).resolve();need(not output.exists(), 'New immutable admission directory required');output.mkdir(parents=True)
    model_ref=runtime.write_new(output/'model.json',model);results=[]
    try:
        results.append(_spawn_worker(model_ref,output/'cpu-one',all_snapshots=True))
        results.append(_spawn_worker(model_ref,output/'cpu-two',all_snapshots=False))
        workers=[runtime.read_bound(ref) for ref in results]
        comparison=compare_cpu_results(model,complete,pre['prepared'],workers)
        after=train.postflight(pre,output,model_inputs=model_inputs)
        need(runtime.read_bound(model['complete'])==complete, 'Completed model record changed during admission')
        runtime.bound_path(model['checkpoint'])
        runtime.verify_owned_stage(cuda_owner_terminal,complete_ref=model['complete'],
            child_identity=runtime.read_bound(complete['resources'])['identity'],experiment='m2_given_contact_train')
        receipt=dict(schema=runtime.given.ADMISSION_SCHEMA,formal_run_schema=runtime.FORMAL_RUN_SCHEMA,
            task_contract=runtime.task_contract(),task_id=runtime.pose.TASK_ID,action_semantics_id=runtime.given.ACTION_SEMANTICS_ID,
            handoff_policy_id=runtime.handoff.POLICY_ID,model=model,inputs=complete['inputs'],processing=complete['processing'],
            input_rebinding=after['evidence'],cuda_owner_terminal=cuda_owner_terminal,cuda_owner_audit=owner,cpu_results=results,
            normalization=complete['inputs']['normalization'],reference_inventory=complete['inputs']['reference_inventory'],
            source_inventory=complete['inputs']['software'],comparison=comparison,passed=comparison['passed'],
            test_payloads_opened=False,test_sealed=True,closed_loop_evaluated=False,physical_acceptance=False,
            production_ready=False,production_bridge_pending=True)
        runtime.write_new(output/'admission.json',receipt);return receipt
    except BaseException as error:
        runtime.write_new(output/'admission.json',dict(schema=runtime.given.ADMISSION_SCHEMA,model=model,
            cuda_owner_terminal=cuda_owner_terminal,cpu_results=results,passed=False,test_sealed=True,
            error=type(error).__name__+': '+str(error),production_ready=False));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True);parser.add_argument('--model-sha256',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--worker',action='store_true')
    parser.add_argument('--all-snapshots',action='store_true');parser.add_argument('--cuda-owner-terminal');parser.add_argument('--cuda-owner-sha256')
    args=parser.parse_args();model_ref=dict(path=str(Path(args.model).resolve()),sha256=args.model_sha256)
    if args.worker:result=cpu_worker(model_ref,args.output,all_snapshots=args.all_snapshots)
    else:
        need(not args.all_snapshots and args.cuda_owner_terminal and args.cuda_owner_sha256, 'Admission requires exact original CUDA owner terminal')
        result=audit_selected_model(runtime.read_bound(model_ref),args.output,cuda_owner_terminal=dict(
            path=str(Path(args.cuda_owner_terminal).resolve()),sha256=args.cuda_owner_sha256))
    print(json.dumps(dict(output=str(Path(args.output).resolve()),passed=result.get('passed'),test_sealed=True)))


if __name__=='__main__':main()
