"""CPU-only admission negatives. No synthetic result is a trained-model permit."""
from copy import deepcopy
import importlib
import numpy as np
import pytest

from test_ausc_given_contact_runtime import module as runtime_module,batch


def module():
    name='sim.ausc_policy.m2_given_contact_admission'
    assert importlib.util.find_spec(name) is not None, 'Independent formal CPU admission missing'
    return importlib.import_module(name)


def comparison_fixture(tmp_path):
    m=module();r=runtime_module();value,_=batch(tmp_path);value['velocities'][:]=0;value['actions'][0,0,0]=0
    prepared=dict(train=value,validation=deepcopy(value),derived_index=dict(samples=[
        dict(split=split,sample_index=i) for split in ('train','validation') for i in range(2)]))
    from sim.ausc_policy import m2_given_contact_train as train
    orders=train.window_order(prepared);inputs=dict(dataset={},software={});model={'checkpoint':{},'complete':{}}
    rows=[];cpu_rows=[]
    for step in r.EVAL_STEPS:
        (tmp_path/('weight'+str(step))).write_bytes(str(step).encode());weight=r.reference(tmp_path/('weight'+str(step)))
        prediction=r.save_predictions(tmp_path/('cuda'+str(step)+'.npy'),value['actions'],window_order_sha256=orders['validation'])
        row=dict(step=step,checkpoint=weight,metrics=r.metrics(value['actions'],value),predictions=prediction)
        rows.append(row);cpu_rows.append(deepcopy(row))
    selected={split:r.save_predictions(tmp_path/(split+'.npy'),prepared[split]['actions'],window_order_sha256=orders[split])
        for split in ('train','validation')}
    metrics={split:r.metrics(prepared[split]['actions'],prepared[split]) for split in ('train','validation')}
    complete=dict(inputs=inputs,evaluations=rows,selected_step=500,selected_predictions=selected,selected_metrics=metrics,window_order=orders)
    workers=[dict(schema=m.WORKER_SCHEMA,model=model,inputs=inputs,identity=dict(pid=i+100,start_ticks=str(i)),
        execution=dict(device='cpu',parameter_devices=['cpu'],parameter_dtypes=['torch.float32'],default_dtype='torch.float32',
            intraop_threads=2,interop_threads=1,deterministic_algorithms=True,matmul_precision='highest',matmul_tf32=False,
            cudnn_tf32=False,cudnn_deterministic=True,cudnn_benchmark=False,cublas_workspace_config=':4096:8',
            torch_version='CPU unit fixture',cuda_version=None,device_name='cpu'),
        device='cpu',window_order=orders,selected_predictions=deepcopy(selected),selected_metrics=deepcopy(metrics),
        evaluations=deepcopy(cpu_rows) if i==0 else [],source_audit_passed=True) for i in range(2)]
    return m,r,model,complete,prepared,workers


def test_independent_cpu_full_arrays_metrics_and_all_forty_argmin_are_required(tmp_path):
    m,r,model,complete,prepared,workers=comparison_fixture(tmp_path)
    result=m.compare_cpu_results(model,complete,prepared,workers)
    assert result['passed'] is True and result['cpu_byte_identical'] is True and result['cpu_selected_step']==500
    assert result['train_gate']['passed'] is True and 'validation_gate' not in result


@pytest.mark.parametrize('fault',['same_process','negative_zero','reordered','missing_snapshot','other_argmin','wrong_metric','missing_window','fixture'])
def test_independent_admission_rejects_partial_stale_or_favorable_device_substitution(tmp_path,fault):
    m,r,model,complete,prepared,workers=comparison_fixture(tmp_path)
    if fault=='same_process':workers[1]['identity']=workers[0]['identity']
    elif fault=='negative_zero':
        a=np.array(prepared['train']['actions']);a[0,0,0]=-0.
        # Different bytes must fail even if both worker metrics are internally recomputed.
        workers[1]['selected_predictions']['train']=r.save_predictions(tmp_path/'different.npy',a,window_order_sha256=complete['window_order']['train'])
        workers[1]['selected_metrics']['train']=r.metrics(a,prepared['train'])
    elif fault=='reordered':workers[1]['window_order']['train']='f'*64
    elif fault=='missing_snapshot':workers[0]['evaluations'].pop()
    elif fault=='other_argmin':
        a=prepared['validation']['actions']+1e-10;row=workers[0]['evaluations'][0]
        row['predictions']=r.save_predictions(tmp_path/'early.npy',a,window_order_sha256=complete['window_order']['validation'])
        row['metrics']=r.metrics(a,prepared['validation'])
    elif fault=='wrong_metric':workers[0]['selected_metrics']['train']['act']['per_horizon_mae_rad'][-1]+=.01
    elif fault=='missing_window':
        workers[1]['selected_predictions']['train']=r.save_predictions(tmp_path/'short.npy',prepared['train']['actions'][:1],
            window_order_sha256=complete['window_order']['train'])
    else:workers[0]['schema']=r.FIXTURE_RESULT_SCHEMA
    with pytest.raises(ValueError):m.compare_cpu_results(model,complete,prepared,workers)


def test_cpu_cuda_gate_disagreement_fails_even_inside_numeric_tolerance(tmp_path):
    m,r,model,complete,prepared,workers=comparison_fixture(tmp_path)
    sample=prepared['train'];threshold=.8*r.metrics(np.zeros_like(sample['actions']),sample)['hold']['full_chunk_mae_rad']
    a=sample['actions']+threshold-5e-8;b=sample['actions']+threshold+5e-8
    # Use constant exact threshold-side metrics derived from complete arrays.
    for worker in workers:
        worker['selected_predictions']['train']=r.save_predictions(tmp_path/('cpu'+str(worker['identity']['pid'])+'.npy'),a,
            window_order_sha256=complete['window_order']['train'])
        worker['selected_metrics']['train']=r.metrics(a,sample)
    complete['selected_predictions']['train']=r.save_predictions(tmp_path/'cuda-gate.npy',b,window_order_sha256=complete['window_order']['train'])
    complete['selected_metrics']['train']=r.metrics(b,sample)
    assert r.admission_gate(workers[0]['selected_metrics']['train'])['passed']!=r.admission_gate(complete['selected_metrics']['train'])['passed']
    with pytest.raises(ValueError,match='gate'):m.compare_cpu_results(model,complete,prepared,workers)


def test_cpu_child_startup_identity_error_reaps_only_its_direct_child(tmp_path,monkeypatch):
    m=module();r=runtime_module()
    from sim.ausc_policy import process_guard as guard
    class Child:
        pid=123;returncode=None;terminated=False
        def poll(self):return self.returncode
        def terminate(self):self.terminated=True;self.returncode=-15
        def kill(self):self.returncode=-9
        def wait(self,timeout):return self.returncode
    child=Child();calls=[]
    def popen(command,**kwargs):
        calls.append((command,kwargs));return child
    def missing(pid):raise FileNotFoundError('identity disappeared')
    monkeypatch.setattr(m.subprocess,'Popen',popen);monkeypatch.setattr(guard,'process_identity',missing)
    monkeypatch.setattr(guard,'signal_owned_group',lambda *a:pytest.fail('Unverified group signalled'))
    with pytest.raises(FileNotFoundError):m._spawn_worker(dict(path='/model.json',sha256='a'*64),tmp_path/'cpu',all_snapshots=True)
    assert child.terminated and child.poll()==-15
    command,options=calls[0]
    assert command[4]==m.MODULE and '-B' in command and '--all-snapshots' in command
    assert options['start_new_session'] is True and options['env']['CUDA_VISIBLE_DEVICES']==''
    assert options['env']['PYTHONDONTWRITEBYTECODE']=='1'


def test_cpu_worker_keeps_one_initial_public_preflight_all_forty_inferences_and_fixed_postflight(tmp_path,monkeypatch):
    from test_ausc_given_contact_train import config_fixture
    prepared_root=tmp_path/'prepared';prepared_root.mkdir()
    t,r,config,prepared,calls=config_fixture(prepared_root,monkeypatch)
    model_root=tmp_path/'model';model_root.mkdir()
    m,r,_,complete,_,workers=comparison_fixture(model_root)
    config_ref=r.write_new(tmp_path/'config.json',config)
    complete.update(inputs=dict(config=config_ref,dataset=config['dataset'],software=r.write_new(tmp_path/'software.json',{})),
        processing=config['processing'],window_order=t.window_order(prepared),execution={},numerical_protocol=r.numerical_protocol(),
        last=complete['evaluations'][-1]['checkpoint'],training_log=r.write_new(tmp_path/'training-log.json',{}))
    model=dict(checkpoint=complete['evaluations'][0]['checkpoint'],complete=r.write_new(model_root/'COMPLETE.json',complete))
    model_ref=r.write_new(tmp_path/'model.json',model)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(r,'validate_complete',lambda v:v)
    monkeypatch.setattr(r,'load_checkpoint',lambda *a,**k:(None,None,dict(normalization=prepared['normalization'],reference_inventory=prepared['reference_inventory']),complete))
    monkeypatch.setattr(r,'execution_settings',lambda *a,**k:workers[0]['execution'])
    monkeypatch.setattr(r,'validate_execution_settings',lambda *a,**k:None)
    monkeypatch.setattr(r,'validate_software',lambda *a:None)
    monkeypatch.setattr(r,'software_inventory',lambda *a:{})
    monkeypatch.setattr(r,'_identity',lambda:workers[0]['identity'])
    snapshots=[];inferences=[]
    monkeypatch.setattr(m,'_load_snapshot',lambda torch,policy,row,*a:snapshots.append(row['step']))
    def infer(torch,policy,samples,*a,**k):inferences.append(len(samples['states']));return samples['actions']
    monkeypatch.setattr(r,'infer',infer)
    result=m.cpu_worker(model_ref,tmp_path/'cpu-output',all_snapshots=True)
    assert [row[0] for row in calls]==['seal','reconstruct']
    assert snapshots==r.EVAL_STEPS+[500] and len(inferences)==42
    assert len(result['evaluations'])==40 and r.read_bound(result['input_rebinding'])['full_decode_repeated'] is False
