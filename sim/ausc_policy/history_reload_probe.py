"""Finite reload inference diagnosis of an already failed checkpoint; no updates."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import numpy as np
from . import act_baseline as a
from .history_fit import load_histories,make_packer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();config=a.read_json(args.config)
    a.need(sys.platform=='linux' and os.environ.get('AUSC_OWNED_RUN')=='1'
        and config['experiment']=='history_reload_probe','Owned finite reload diagnosis required')
    run=Path(config['failed_run']);owner=a.pinned_json(run/'result.json',config['owner_sha256'])
    a.need(owner['completed'] is False and owner['gpu_released'] and owner['stage']['process_group_drained']
        and owner['stage']['exit_code']==1 and owner['config']['experiment']=='history_fit','Closed failed fit required')
    outer=owner['config'];baseline=a.pinned_json(outer['baseline_config'],outer['baseline_config_sha256'])
    folder=run/'output/repeat_current';checkpoint=folder/'checkpoint_0512'
    a.need(a.sha256(checkpoint/'model.safetensors')==config['weights_sha256']
        and a.sha256(checkpoint/'predictions.npz')==config['predictions_sha256'],'Failed saved artifacts changed')
    args.output.mkdir(parents=True,exist_ok=False)
    with (Path(config['base'])/'locks/isaac_gpu.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        episodes,stats,_=a.load_prepared(baseline['prepared_manifest'],baseline)
        histories,aux,_=load_histories(outer,episodes);packer=make_packer(histories,aux,'repeat_current')
        probe=a.read_json(folder/'probe_rows.json');by={e['episode_id']:e for e in episodes}
        rows=[a.row_from_episode(by[r['episode_id']],r['start'],10) for r in probe]
        with np.load(checkpoint/'predictions.npz',allow_pickle=False) as saved:
            expected={k:saved[k].copy() for k in ('raw','reference','measured_q','model_action_physical')}
        a.need(np.array_equal(expected['measured_q'],np.stack([r['measured_q'] for r in rows]))
            and np.array_equal(expected['reference'],np.stack([r['actions'] for r in rows])),'Saved original probe origins differ')
        torch,policy,_=a.make_native(baseline,environment_width=40)
        from safetensors.torch import load_model,load_file
        missing,unexpected=load_model(policy,str(checkpoint/'model.safetensors'),strict=True,device='cuda')
        a.need(not missing and not unexpected,'Saved tensors do not load')
        on_disk=load_file(str(checkpoint/'model.safetensors'),device='cpu')
        a.need(set(on_disk)==set(policy.state_dict()) and all(torch.equal(value.cpu(),on_disk[key]) for key,value in policy.state_dict().items()),'Loaded tensor values differ')
        policy.eval();batch=packer(rows,stats,training=False)
        results=[];predictions={};original_fastpath=torch.backends.mha.get_fastpath_enabled()
        try:
            for requires_grad in (True,False):
                policy.requires_grad_(requires_grad)
                for fastpath in (True,False):
                    torch.backends.mha.set_fastpath_enabled(fastpath)
                    outputs=[]
                    for repeat in range(2):
                        with torch.inference_mode():out=policy.predict_action_chunk({k:torch.from_numpy(v).cuda() for k,v in batch.items()})
                        outputs.append(out.cpu().numpy().copy())
                    physical=a.unscale(outputs[0],stats['action'])
                    raw=a.decode_actions(physical,expected['measured_q'],a.BLOCK_RELATIVE)
                    key=f'grad{int(requires_grad)}_fast{int(fastpath)}';predictions[key]=raw
                    result=dict(requires_grad=requires_grad,mha_fastpath=fastpath,
                        repeated_normalized_output_max=float(abs(outputs[0]-outputs[1]).max()),
                        saved_raw_max_error_rad=float(abs(raw-expected['raw']).max()),
                        saved_relative_max_error_rad=float(abs(physical-expected['model_action_physical']).max()))
                    results.append(result);print(result,flush=True)
        finally:torch.backends.mha.set_fastpath_enabled(original_fastpath)
        np.savez(args.output/'predictions.npz',**predictions,reference=expected['raw'])
        a.save_json(args.output/'COMPLETE.json',dict(completed=True,results=results,original_fastpath=original_fastpath,
            source_owner_sha256=config['owner_sha256'],source_weights_sha256=config['weights_sha256'],
            exact_tensor_load=True,original_origins_verified=True,optimizer_updates=0,m0_passed=False,
            scope='Fixed failed weights inference diagnosis, not repair, retraining or physical acceptance'))


if __name__=='__main__':main()
