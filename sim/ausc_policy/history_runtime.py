"""Explicit causal-history ACT inference; original 19D runtime stays default.

Only measured past q enters the extra 28 features. Each sparse history is bound
to real frame identities and simulation times; no interpolation or padding.
The service opens checkpoint metadata, never training arrays or an expert.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import numpy as np
from . import act_baseline as a, act_runtime as r
from .history_fit import VARIANTS, history_deltas
from .history_prepare import OFFSETS


def validate_receipt(receipt):
    r.validate_receipt(receipt)
    a.need(receipt.get('history_variant') in VARIANTS
        and receipt.get('environment_width')==40 and receipt.get('auxiliary_width')==28
        and receipt.get('history_offsets_frames')==list(OFFSETS)
        and all(r.is_digest(receipt.get(k)) for k in ('history_contract_sha256','history_preparation_sha256',
                  'paired_owner_sha256','paired_complete_sha256')), 'Invalid causal history receipt')


def load_artifacts(config):
    root=Path(config['run_root']);owner=a.pinned_json(root/'result.json',config['owner_sha256'])
    complete=a.pinned_json(root/'output/COMPLETE.json',config['complete_sha256'])
    variant=config['variant'];step=config['checkpoint_step']
    a.need(variant in VARIANTS and step in (512,2000) and owner['completed'] and owner['gpu_released']
        and owner['stage']['process_group_drained'] and owner['stage']['exit_code']==0
        and owner['config']['experiment']=='history_fit' and complete['completed']
        and complete['identical_initial_weights'],'Completed source-bound paired fit required')
    records=[v for v in complete['variants'] if v['variant']==variant]
    a.need(len(records)==1 and records[0]['steps']==step and records[0]['maximum_reload_error_rad']==0
        and records[0]['final_weights_sha256']==config['checkpoint_sha256'],'Paired final checkpoint differs')
    folder=root/'output'/variant;checkpoint=folder/f'checkpoint_{step:04d}'
    artifacts=r.load_artifacts(folder/'configuration.json',checkpoint,config['checkpoint_sha256'],environment_width=40)
    a.need(artifacts.config==a.pinned_json(owner['config']['baseline_config'],owner['config']['baseline_config_sha256']),
           'Paired source configuration changed')
    history=a.pinned_json(folder/'history_contract.json',config['history_contract_sha256']);aux=history['auxiliary_normalization']
    a.need(history['variant']==variant and history['offsets_frames']==list(OFFSETS)
        and history['single_frame_state_dim']==19 and history['auxiliary_width']==28
        and history['environment_width']==40
        and history['history_owner_sha256']==owner['config']['history_owner_sha256']
        and history['history_complete_sha256']==owner['config']['history_complete_sha256'], 'History contract differs')
    low,high=(np.asarray(aux[k],float) for k in ('low','high'))
    a.need(low.shape==high.shape==(28,) and np.isfinite(low).all() and np.isfinite(high).all()
        and np.all(high>low) and aux['actual_clamp'] is False and aux['span_floor_rad']==.002,
        'Invalid frozen history normalization')
    artifacts.receipt.update(history_variant=variant,environment_width=40,auxiliary_width=28,
        history_offsets_frames=list(OFFSETS),history_contract_sha256=a.sha256(folder/'history_contract.json'),
        history_preparation_sha256=history['history_complete_sha256'],
        paired_owner_sha256=config['owner_sha256'],paired_complete_sha256=config['complete_sha256'])
    validate_receipt(artifacts.receipt)
    return artifacts,aux


def validate_history(history,request):
    a.need(isinstance(history,list) and len(history)==5,'Five actual sparse history frames required')
    states=[]
    for frame,offset in zip(history,OFFSETS):
        a.need(isinstance(frame,dict) and set(frame)=={'frame_id','sim_stamp_s','target_id','q'},'History fields differ')
        stamp=frame['sim_stamp_s'];index=frame['frame_id']
        a.need(type(index) is int and index>=0 and index==request['frame_id']+offset
            and type(stamp) in (int,float) and np.isfinite(stamp) and stamp>=0
            and abs(stamp-request['sim_stamp_s']-offset*.05)<=1e-8
            and frame['target_id']==request['target_id'],'History frame, target or causal clock differs')
        q=np.asarray(frame['q'])
        a.need(q.shape==(7,) and q.dtype.kind in 'fiu' and np.isfinite(q).all(), 'Invalid measured history q')
        states.append(q.astype(np.float32))
    values=np.stack(states)
    a.need(np.isfinite(values).all() and np.array_equal(values[-1],r.validate_state(request['state'])[:7]),
           'History current q differs from acquired block anchor')
    return values


def history_from_frames(frames,current):
    """Select original acquired frames, including the current endpoint."""
    by={f['index']:f for f in frames[-11:]}
    a.need(len(by)==len(frames[-11:]),'Duplicate actual history frames')
    result=[]
    for offset in OFFSETS:
        index=current['index']+offset
        a.need(index in by,'No full acquired half-second history; padding forbidden')
        f=by[index]
        result.append(dict(frame_id=f['index'],sim_stamp_s=f['time_s'],target_id=f['target'],q=list(f['q'])))
    return result


def input_digest(state,history):
    return r.digest_json(dict(state=r.validate_state(state).tolist(),history=history))


def validate_request(request,receipt):
    validate_receipt(receipt)
    a.need(isinstance(request,dict) and request.get('schema')=='ausc_act_history_request_v1', 'Explicit history request required')
    base={k:v for k,v in request.items() if k!='history'};base['schema']='ausc_act_request_v1'
    state=r.validate_request(base,receipt)
    return state,validate_history(request.get('history'),request)


def pack_history(state,q,stats,aux,variant):
    state=r.validate_state(state);q=np.asarray(q,np.float32)
    a.need(q.shape==(5,7) and np.isfinite(q).all() and np.array_equal(q[-1],state[:7]), 'History block anchor differs')
    history=np.zeros((1,5,19),np.float32);history[0,:,:7]=q
    batch=a.pack_batch([dict(state=state)],stats,training=False)
    extra=a.scale(history_deltas(history,variant),aux)
    batch['observation.environment_state']=np.concatenate((batch['observation.environment_state'],extra),axis=1)
    return batch


class NativeEngine(r.NativeEngine):
    def __init__(self,artifacts):super().__init__(artifacts,environment_width=40)


class InferenceService:
    def __init__(self,artifacts,aux,engine):
        validate_receipt(artifacts.receipt);self.artifacts,self.aux,self.engine=artifacts,aux,engine

    def predict_state(self,state,q):
        state=r.validate_state(state)
        batch=pack_history(state,q,self.artifacts.stats,self.aux,self.artifacts.receipt['history_variant'])
        prediction=np.asarray(self.engine.predict(batch))
        a.need(prediction.shape==(1,10,7) and np.isfinite(prediction).all(),'Invalid history ACT output')
        return a.decode_actions(a.unscale(prediction,self.artifacts.stats['action']),state[None,:7],a.BLOCK_RELATIVE)[0]

    def handle(self,request):
        started=time.perf_counter();state,q=validate_request(request,self.artifacts.receipt)
        raw=self.predict_state(state,q)
        return dict(schema='ausc_act_history_response_v1',status='ok',raw=raw.tolist(),
            **{k:request[k] for k in r.BINDINGS},input_sha256=input_digest(state,request['history']),
            receipt=copy.deepcopy(self.artifacts.receipt),receipt_sha256=r.digest_json(self.artifacts.receipt),
            checkpoint_sha256=self.artifacts.receipt['checkpoint_sha256'],latency_ms=(time.perf_counter()-started)*1000)


def validate_response(response,request,receipt):
    a.need(response.get('schema')=='ausc_act_history_response_v1'
        and response.get('input_sha256')==input_digest(request['state'],request['history']), 'History response input binding differs')
    base=dict(response,schema='ausc_act_response_v1',input_sha256=r.input_digest(request['state']))
    r.validate_response(base,request,receipt)
    return response


class HistoryClient:
    def __init__(self,port,receipt,*,timeout_s=5.):
        validate_receipt(receipt)
        a.need(type(port) is int and 0<port<=65535 and np.isfinite(timeout_s) and timeout_s>0,'Invalid history client port/deadline')
        self.port,self.receipt,self.timeout_s=port,copy.deepcopy(receipt),timeout_s

    def predict(self,state,*,history,**bindings):
        request=r.build_request(state,self.receipt,**bindings)
        request.update(schema='ausc_act_history_request_v1',history=history);validate_request(request,self.receipt)
        deadline=time.monotonic()+self.timeout_s
        with socket.create_connection((r.HOST,self.port),timeout=self.timeout_s) as connection:
            r.send_message(connection,request,deadline);response=r.read_message(connection,deadline)
        return validate_response(response,request,self.receipt)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();config=a.read_json(args.config)
    a.need(sys.platform=='linux' and os.environ.get('AUSC_OWNED_RUN')=='1'
        and config['experiment']=='history_runtime' and config['mode'] in ('verify_reload','serve'),'Owned explicit history runtime required')
    args.output.mkdir(parents=True,exist_ok=False)
    artifacts,aux=load_artifacts(config)
    if config['mode']=='serve':
        a.need(not {'inputs_npz','reference_npz'}&config.keys(),'Online history service opens no label/observation datasets')
        service=InferenceService(artifacts,aux,NativeEngine(artifacts))
        with r.make_server(service,port=config.get('port',0),request_timeout_s=5.) as server:
            a.save_json(args.output/'ready.json',dict(host=r.HOST,port=server.server_address[1],receipt=artifacts.receipt,
                receipt_sha256=r.digest_json(artifacts.receipt),scope='inference_only'))
            server.serve_forever(poll_interval=.1)
    else:
        a.need(config['atol_rad']==1e-6 and config['time_limit_seconds']==600,'Original CPU reload gate required')
        states=r.read_small_npz(config['inputs_npz'],'state19',(19,));history=r.read_small_npz(config['inputs_npz'],'q_history',(5,7))
        reference=r.read_small_npz(config['reference_npz'],'raw',(10,7))
        a.need(a.sha256(config['inputs_npz'])==config['inputs_sha256'] and a.sha256(config['reference_npz'])==config['reference_sha256']
            and len(states)==len(history)==len(reference)==32,'Bound original 32 history inputs required')
        outputs=[]
        for _ in range(2):
            service=InferenceService(artifacts,aux,NativeEngine(artifacts))
            outputs.append(np.stack([service.predict_state(s,h) for s,h in zip(states,history)]));del service
        error=float(abs(outputs[0]-reference).max());reload_error=float(abs(outputs[0]-outputs[1]).max())
        passed=error<=config['atol_rad'] and reload_error==0
        evidence=dict(passed=passed,completed=passed,count=32,native_model_loaded=True,device='cpu',
            max_reference_difference_rad=error,max_reload_difference_rad=reload_error,receipt=artifacts.receipt,
            inputs_sha256=config['inputs_sha256'],reference_sha256=config['reference_sha256'],
            m0_passed=False,closed_loop_evaluated=False)
        np.savez(args.output/'predictions.npz',raw=outputs[0],reference=reference)
        a.save_json(args.output/('COMPLETE.json' if passed else 'FAILED.json'),evidence)
        a.need(passed,'History CPU reload differs from GPU reference')


if __name__=='__main__':main()
