"""Prepare only 32 bound history probe inputs for native runtime validation."""
import argparse
import os
from pathlib import Path
import sys
import numpy as np
from . import act_baseline as a
from .history_fit import load_histories,VARIANTS


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();config=a.read_json(args.config)
    a.need(sys.platform=='linux' and os.environ.get('AUSC_OWNED_RUN')=='1'
        and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and config['experiment']=='history_runtime_prepare',
        'Owned CPU-only history validation preparation required')
    root=Path(config['run_root']);owner=a.pinned_json(root/'result.json',config['owner_sha256'])
    complete=a.pinned_json(root/'output/COMPLETE.json',config['complete_sha256'])
    a.need(owner['completed'] and owner['gpu_released'] and owner['stage']['process_group_drained']
        and owner['stage']['exit_code']==0 and complete['completed'] and complete['mode']=='train'
        and complete['steps_per_variant']==2000 and complete['identical_initial_weights'], 'Completed finite paired TRAIN16 required')
    outer=owner['config'];baseline=a.pinned_json(outer['baseline_config'],outer['baseline_config_sha256'])
    episodes,stats,_=a.load_prepared(baseline['prepared_manifest'],baseline)
    histories,aux,_=load_histories(outer,episodes);by={e['episode_id']:e for e in episodes}
    args.output.mkdir(parents=True,exist_ok=False);records=[];prior=None
    for variant in VARIANTS:
        folder=root/'output'/variant;probe=a.read_json(folder/'probe_rows.json')
        a.need(len(probe)==32 and (prior is None or prior==probe),'Original paired probe rows differ');prior=probe
        rows=[a.row_from_episode(by[r['episode_id']],r['start'],10) for r in probe]
        history=np.stack([histories[(r['episode_id'],r['start'])] for r in rows])
        state=np.stack([r['state'] for r in rows])
        a.need(np.array_equal(history[:,-1],state),'Actual current endpoint differs')
        reference=folder/'checkpoint_2000/predictions.npz'
        with np.load(reference,allow_pickle=False) as saved:
            a.need(np.array_equal(saved['measured_q'],state[:,:7]) and np.array_equal(saved['reference'],np.stack([r['actions'] for r in rows])),
                'Saved predictions paired with different observations or labels')
        hc=a.read_json(folder/'history_contract.json')
        a.need(hc['auxiliary_normalization']==aux and a.read_json(folder/'normalization.json')==stats,'Original normalization changed')
        inputs=args.output/f'{variant}_inputs.npz'
        np.savez(inputs,state19=state,q_history=history[:,:,:7])
        weights=folder/'checkpoint_2000/model.safetensors'
        result=dict(experiment='history_runtime',mode='verify_reload',base=config['base'],run_root=str(root),
            owner_sha256=config['owner_sha256'],complete_sha256=config['complete_sha256'],variant=variant,checkpoint_step=2000,
            checkpoint_sha256=a.sha256(weights),history_contract_sha256=a.sha256(folder/'history_contract.json'),
            inputs_npz=str(inputs.resolve()),inputs_sha256=a.sha256(inputs),reference_npz=str(reference),reference_sha256=a.sha256(reference),
            atol_rad=1e-6,time_limit_seconds=600)
        a.save_json(args.output/f'{variant}_reload.json',result);records.append(result)
    a.save_json(args.output/'COMPLETE.json',dict(completed=True,variants=records,count_per_variant=32,
        original_inputs_labels_exact=True,heldout_opened=False,optimizer_updates=0,m0_passed=False))


if __name__=='__main__':main()
