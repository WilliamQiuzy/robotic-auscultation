"""Fixed final history models: full TRAIN coverage and actual-query CPU replay.

Zero optimizer updates. Early segments are fixed first 20% of eligible origins,
not selected using model errors. Counterfactual or oracle labels are not made.
"""
import argparse
import gc
import os
from pathlib import Path
import sys
import time
import numpy as np
from . import act_baseline as a,history_runtime as h
from .history_fit import load_histories,make_packer,VARIANTS


def groups(rows):
    ids=[(r['episode_id'],r['phase']) for r in rows]
    result={'all':np.arange(len(rows))}
    for episode in a.TRAIN16:
        indices=np.array([i for i,(e,p) in enumerate(ids) if e==episode],int)
        a.need(len(indices)>0,'Missing declared TRAIN episode')
        result[episode]=indices;result[episode+'/early20pct']=indices[:max(1,int(np.ceil(len(indices)*.2)))]
    for phase in ('FREE','APPROACH'):result[phase]=np.array([i for i,(e,p) in enumerate(ids) if p==phase],int)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();config=a.read_json(args.config)
    a.need(sys.platform=='linux' and os.environ.get('AUSC_OWNED_RUN')=='1' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
        and config['experiment']=='history_path_probe' and config['variants']==list(VARIANTS), 'Owned fixed paired CPU probe required')
    args.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();records=[]
    root=Path(config['training_run']);owner=a.pinned_json(root/'result.json',config['training_owner_sha256'])
    complete=a.pinned_json(root/'output/COMPLETE.json',config['training_complete_sha256'])
    a.need(owner['completed'] and owner['gpu_released'] and complete['completed'] and complete['steps_per_variant']==2000,'Original paired TRAIN16 required')
    outer=owner['config'];baseline=a.pinned_json(outer['baseline_config'],outer['baseline_config_sha256'])
    episodes,stats,_=a.load_prepared(baseline['prepared_manifest'],baseline)
    histories,aux,_=load_histories(outer,episodes)
    rows=[dict(a.row_from_episode(e,s,10),phase=str(e['arrays']['phases'][s])) for e in episodes for s in e['starts']]
    reference=np.stack([r['actions'] for r in rows]);q=np.stack([r['measured_q'] for r in rows]);dq=np.stack([r['measured_dq'] for r in rows])
    group_indices=groups(rows)
    observed_velocity=np.stack([(histories[(r['episode_id'],r['start'])][-1,:7]-histories[(r['episode_id'],r['start'])][-2,:7])/.05 for r in rows])
    candidate_baselines=dict(hold=np.repeat(q[:,None],10,axis=1),api_velocity=q[:,None]+dq[:,None]*np.arange(1,11)[None,:,None]*.05,
        observed_position_velocity=q[:,None]+observed_velocity[:,None]*np.arange(1,11)[None,:,None]*.05)
    base_metrics={name:{key:a.error_summary(raw[indices]-reference[indices]) for key,indices in group_indices.items()} for name,raw in candidate_baselines.items()}
    for variant in VARIANTS:
        selected=config['runtime_configs'][variant];runtime=a.pinned_json(selected['path'],selected['sha256'])
        artifacts,normalization=h.load_artifacts(runtime)
        a.need(artifacts.receipt['paired_owner_sha256']==config['training_owner_sha256']
            and artifacts.receipt['history_variant']==variant and normalization==aux,'Selected fixed model/normalization differs')
        engine=h.NativeEngine(artifacts);service=h.InferenceService(artifacts,normalization,engine)
        packer=make_packer(histories,aux,variant);predicted=[]
        # Native CPU inference in fixed batches, no labels enter the model.
        for offset in range(0,len(rows),32):
            selected_rows=rows[offset:offset+32];batch=packer(selected_rows,stats,training=False)
            z=engine.predict(batch)
            predicted.extend(a.decode_actions(a.unscale(z,stats['action']),np.stack([r['measured_q'] for r in selected_rows]),a.BLOCK_RELATIVE))
        raw=np.asarray(predicted);a.need(raw.shape==reference.shape,'Full TRAIN output shape differs')
        np.savez(args.output/f'{variant}_train_predictions.npz',raw=raw,reference=reference,measured_q=q)
        metrics={key:a.error_summary(raw[indices]-reference[indices]) for key,indices in group_indices.items()}
        # Preserve every first eligible origin, even when its motion is tiny.
        # Signed vectors are evidence for inspection, not a post-hoc pass gate.
        first_origins=[]
        for episode in a.TRAIN16:
            i=int(group_indices[episode][0])
            first_origins.append(dict(episode_id=episode,start=int(rows[i]['start']),
                state19=rows[i]['state'].tolist(),history_q=histories[(episode,rows[i]['start'])][:,:7].tolist(),
                predicted_relative=(raw[i]-q[i]).tolist(),reference_relative=(reference[i]-q[i]).tolist(),
                error=a.error_summary((raw[i]-reference[i])[None])))
        a.save_json(args.output/f'{variant}_first_origins.json',first_origins)
        replay=[]
        for source in config['cases']:
            case_root=Path(source['run']);case_owner=a.pinned_json(case_root/'result.json',source['owner_sha256'])
            outcome=a.pinned_json(case_root/'output/case/outcome.json',source['outcome_sha256'])
            a.need(case_owner['completed'] and case_owner['gpu_released'] and case_owner['stage']['process_group_drained'], 'Actual case must be closed')
            if case_owner['config']['variant']!=variant:continue
            a.need(outcome['learned_policy_receipt']==artifacts.receipt,'Actual case used another model')
            errors=[];predictions=[]
            for query in outcome['l2_queries']:
                if not query['accepted']:continue
                request=dict(frame_id=query['observation_frame'],sim_stamp_s=query['time_s'],target_id=query['target'],state=query['state19'])
                history=h.validate_history(query['causal_history'],request)
                candidate=service.predict_state(query['state19'],history)
                errors.append(float(abs(candidate-np.asarray(query['raw_absolute_chunk'],np.float32)).max()));predictions.append(candidate)
            a.need(errors and max(errors)==0,'Actual CPU service response does not exactly replay')
            np.savez(args.output/f'{variant}_actual_replay.npz',raw=np.asarray(predictions))
            replay.append(dict(run_id=case_root.name,accepted_queries=len(errors),maximum_error_rad=max(errors)))
        a.need(len(replay)==1,'Exactly one same-version case per model required')
        records.append(dict(variant=variant,checkpoint_sha256=artifacts.receipt['checkpoint_sha256'],metrics=metrics,actual_replay=replay))
        print(dict(variant=variant,all=metrics['all'],R4_early=metrics['R4_observed_armature02_v1/early20pct'],actual_replay=replay),flush=True)
        del service,engine;gc.collect()
    a.save_json(args.output/'rows.json',[dict(episode_id=r['episode_id'],start=r['start'],phase=r['phase']) for r in rows])
    a.save_json(args.output/'COMPLETE.json',dict(completed=True,records=records,baselines=base_metrics,
        train_windows=len(rows),group_sizes={k:len(v) for k,v in group_indices.items()},elapsed_s=time.monotonic()-start,
        optimizer_updates=0,heldout_opened=False,new_physics_executed=False,m0_passed=False,
        scope='Full original TRAIN fit plus exact actual-query reproduction; no new labels or generalization claim'))


if __name__=='__main__':main()
