"""Paired finite ACT fit with true past q deltas versus repeated-current control.

Same architecture, parameters at step0, examples, labels, draws and optimizer.
The extra 28 inputs are four measured past-minus-current seven-joint vectors;
they are not PhysX's reported dq, future commands, or expert path progress.
This M0 diagnostic is state-only, not the requested final multimodal VLA.
"""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import sys
import numpy as np
from . import act_baseline as a
from .history_prepare import OFFSETS

VARIANTS=('repeat_current','causal_history')


def history_deltas(history,variant):
    history=np.asarray(history,np.float32)
    if (history.ndim!=3 or history.shape[1:]!=(5,19) or not np.isfinite(history).all()
            or variant not in VARIANTS):
        raise ValueError('Declared actual history batch and condition required')
    if variant=='repeat_current':return np.zeros((len(history),28),np.float32)
    return (history[:,:4,:7]-history[:,4:5,:7]).reshape(len(history),28).copy()


def history_stats(values):
    values=np.asarray(values,float)
    if values.ndim!=2 or values.shape[1]!=28 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Finite full TRAIN history deltas required')
    q01,q99=np.quantile(values,[.01,.99],axis=0);width=np.maximum(q99-q01,.002)
    low=(q01+q99-width)/2;high=low+width;z=2*(values-low)/width-1
    return dict(low=low.tolist(),high=high.tolist(),q01=q01.tolist(),q99=q99.tolist(),
        span_floor_used=((q99-q01)<.002).tolist(),potential_clamp_fraction=(abs(z)>1).mean(axis=0).tolist(),
        actual_clamp=False,units='rad',span_floor_rad=.002,
        order='t-500ms q-current q; t-250ms; t-100ms; t-50ms; seven joints each')


def load_histories(config,episodes):
    root=Path(config['history_run']);owner=a.pinned_json(root/'result.json',config['history_owner_sha256'])
    receipt=a.pinned_json(root/'output/COMPLETE.json',config['history_complete_sha256'])
    a.need(owner['completed'] and owner['gpu_released'] and owner['stage']['process_group_drained']
        and owner['stage']['exit_code']==0 and owner['config']['experiment']=='history_prepare'
        and receipt['completed'] and receipt['current_inputs_exactly_reproduced']
        and receipt['offsets_frames']==list(OFFSETS) and receipt['heldout_opened'] is False,
        'Completed original causal history preparation required')
    a.need([r['episode_id'] for r in receipt['episodes']]==list(a.TRAIN16),'History TRAIN identities differ')
    rows={};features=[]
    for episode,record in zip(episodes,receipt['episodes']):
        a.need(episode['episode_id']==record['episode_id'],'History source reordered')
        index=a.pinned_json(record['index_path'],record['index_sha256'])
        a.need(index['starts']==episode['starts'] and index['offsets_frames']==list(OFFSETS),'History rows remapped')
        a.need(a.sha256(record['history_path'])==record['history_sha256'],'History cache changed')
        history=np.load(record['history_path'],allow_pickle=False,mmap_mode='r')
        a.need(history.shape==(len(episode['starts']),5,19) and np.isfinite(history).all(),'History cache shape differs')
        for k,start in enumerate(episode['starts']):
            row=a.row_from_episode(episode,start,10)
            a.need(np.array_equal(history[k,-1],row['state']),'History current endpoint changed')
            rows[(episode['episode_id'],start)]=history[k]
        features.append(history_deltas(history,'causal_history'))
    a.need(len(rows)==receipt['total_windows'],'History row count differs')
    return rows,history_stats(np.concatenate(features)),receipt


def make_packer(histories,history_normalization,variant):
    def pack(rows,stats,**kwargs):
        # Decode/labels always retain the original measured-q block origin.
        batch=a.pack_batch(rows,stats,**kwargs)
        history=np.stack([histories[(r['episode_id'],r['start'])] for r in rows])
        a.need(np.array_equal(history[:,-1],np.stack([r['state'] for r in rows])), 'Batch history pairing changed')
        auxiliary=a.scale(history_deltas(history,variant),history_normalization)
        batch['observation.environment_state']=np.concatenate((batch['observation.environment_state'],auxiliary),axis=1)
        return batch
    return pack


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();config=a.read_json(args.config)
    a.need(sys.platform=='linux' and os.environ.get('AUSC_OWNED_RUN')=='1'
        and config['experiment']=='history_fit' and config['variants']==list(VARIANTS), 'Owned explicit paired fit required')
    baseline=a.pinned_json(config['baseline_config'],config['baseline_config_sha256']);a.validate_config(baseline)
    a.need(baseline['mode']=='train32' and baseline['device']=='cuda' and baseline['action_representation']==a.BLOCK_RELATIVE,
           'Initial history experiment is paired finite 512-step fit only')
    args.output.mkdir(parents=True,exist_ok=False)
    with (Path(config['base'])/'locks/isaac_gpu.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        episodes,stats,receipt=a.load_prepared(baseline['prepared_manifest'],baseline)
        histories,aux,history_receipt=load_histories(config,episodes);variants=[]
        native=lambda c:a.make_native(c,environment_width=40)
        torch,smoke,_=native(dict(baseline,device='cpu'));smoke_rows=[a.row_from_episode(e,e['starts'][0],10) for e in episodes[:2]]
        smoke_receipts=[]
        for variant in VARIANTS:
            packer=make_packer(histories,aux,variant);smoke.zero_grad(set_to_none=True)
            loss,_=smoke({k:torch.from_numpy(v) for k,v in packer(smoke_rows,stats,training=True,action_representation=a.BLOCK_RELATIVE).items()})
            a.need(bool(torch.isfinite(loss)),'Native history loss nonfinite');loss.backward()
            gradients=[p.grad for p in smoke.parameters() if p.grad is not None]
            a.need(gradients and all(bool(torch.isfinite(g).all()) for g in gradients),'Native history gradient invalid')
            smoke.eval()
            with torch.inference_mode():prediction=smoke.predict_action_chunk({k:torch.from_numpy(v) for k,v in packer(smoke_rows,stats,training=False).items()})
            a.need(tuple(prediction.shape)==(2,10,7) and bool(torch.isfinite(prediction).all()),'Native history shape invalid')
            smoke.train();smoke_receipts.append(dict(variant=variant,passed=True,loss=float(loss.detach()),gradient_tensors=len(gradients)))
        a.save_json(args.output/'native_smoke.json',dict(conditions=smoke_receipts,device='cpu',optimizer_updates=0,environment_width=40))
        del smoke,gradients,loss,prediction;gc.collect()
        for variant in VARIANTS:
            folder=args.output/variant;folder.mkdir();packer=make_packer(histories,aux,variant)
            a.save_json(folder/'configuration.json',baseline);a.save_json(folder/'normalization.json',stats)
            a.save_action_contract(folder,baseline,receipt)
            a.save_json(folder/'history_contract.json',dict(variant=variant,auxiliary_normalization=aux,
                history_owner_sha256=config['history_owner_sha256'],history_complete_sha256=config['history_complete_sha256'],
                offsets_frames=list(OFFSETS),single_frame_state_dim=19,auxiliary_width=28,environment_width=40,
                features='Observed past q minus current q, or zero for repeated-current control',
                current_q_dropout_only=True,training_scope='original32 paired fit, no new expert labels'))
            a.fit(baseline,folder,episodes,stats,receipt,batch_packer=packer,native_factory=native)
            # A strict actual-weight roundtrip is required before a completed
            # result. This does not make the old 19D live server compatible.
            torch,policy,_=native(baseline)
            from safetensors.torch import load_model
            checkpoint=folder/'checkpoint_0512';missing,unexpected=load_model(policy,str(checkpoint/'model.safetensors'),strict=True,device='cuda')
            a.need(not missing and not unexpected,'History checkpoint state differs')
            policy.eval();policy.requires_grad_(False)
            probe=a.read_json(folder/'probe_rows.json');by={e['episode_id']:e for e in episodes}
            rows=[a.row_from_episode(by[r['episode_id']],r['start'],10) for r in probe]
            with torch.inference_mode():
                prediction=policy.predict_action_chunk({k:torch.from_numpy(v).cuda() for k,v in packer(rows,stats,training=False).items()})
            raw=a.decode_actions(a.unscale(prediction.cpu().numpy(),stats['action']),np.stack([r['measured_q'] for r in rows]),a.BLOCK_RELATIVE)
            with np.load(checkpoint/'predictions.npz',allow_pickle=False) as saved:error=float(abs(raw-saved['raw']).max())
            a.need(error<=1e-7,'History checkpoint reload changes predictions')
            metrics=a.read_json(checkpoint/'metrics.json')
            row=dict(variant=variant,steps=512,maximum_reload_error_rad=error,metrics=metrics,
                initial_weights_sha256=a.sha256(folder/'checkpoint_0000/model.safetensors'),
                final_weights_sha256=a.sha256(checkpoint/'model.safetensors'))
            a.save_json(folder/'roundtrip.json',row);variants.append(row)
            del policy;gc.collect();torch.cuda.empty_cache()
        a.need(variants[0]['initial_weights_sha256']==variants[1]['initial_weights_sha256'], 'Paired initial tensors differ')
        a.need(a.read_json(args.output/'repeat_current/probe_rows.json')==a.read_json(args.output/'causal_history/probe_rows.json'),'Paired origins differ')
        a.save_json(args.output/'COMPLETE.json',dict(completed=True,variants=variants,identical_initial_weights=True,
            original_history_windows=history_receipt['total_windows'],baseline_config_sha256=config['baseline_config_sha256'],
            m0_passed=False,closed_loop_evaluated=False,heldout_opened=False,
            scope='Paired TRAIN32 numerical fit only; not VLA, sensor-use proof or deployment-ready policy'))


if __name__=='__main__':main()
