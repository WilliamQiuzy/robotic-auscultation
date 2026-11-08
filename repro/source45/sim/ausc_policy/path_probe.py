"""All qualified R4 TRAIN windows and saved-query CPU inference diagnostics.

No optimizer, simulator, trajectory generation, heldout arrays or images.
Nearest TRAIN states describe support; their labels are not counterfactual
oracle actions and never feed a live control process.
"""
from __future__ import annotations
import argparse
from collections import Counter
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from . import act_baseline as a
from .act_runtime import InferenceService,NativeEngine,load_artifacts
from .executor_probe import load_source

FLAGS=dict(m0_passed=False,closed_loop_evaluated=False,physics_executed=False,
           contact_evaluated=False,training_updates=0,heldout_arrays_opened=False,image_arrays_opened=False)


def describe(value):
    value=np.asarray(value,float)
    a.need(value.size>0 and np.isfinite(value).all(),'Finite nonempty metric required')
    return dict(mean=float(value.mean()),median=float(np.median(value)),
                p95=float(np.quantile(value,.95)),max=float(value.max()))


def rotations(states):
    columns=np.asarray(states)[:,10:16].reshape(-1,2,3).transpose(0,2,1)
    return np.concatenate([columns,np.cross(columns[:,:,0],columns[:,:,1])[:,:,None]],axis=2)


def coverage(queries,states,starts,phases,stats):
    z=a.scale(states,stats['state']); output=[]
    for query in queries:
        state=np.asarray(query['state19'],np.float32)
        candidates=np.flatnonzero(phases==query['phase'])
        a.need(len(candidates)>0,'Online phase absent from original R4 TRAIN')
        distance=np.linalg.norm(z[candidates]-a.scale(state,stats['state']),axis=1)
        nearest=candidates[np.argmin(distance)]
        output.append(dict(time_s=query['time_s'],phase=query['phase'],source_start=int(starts[nearest]),
            normalized_state_l2=float(distance.min()),
            joint_max_abs_rad=float(np.abs(state[:7]-states[nearest,:7]).max()),
            task_relative_tcp_distance_mm=float(np.linalg.norm(state[7:10]-states[nearest,7:10])*1000),
            task_relative_rotation_deg=float(np.rad2deg(Rotation.from_matrix(
                rotations(state[None])[0]@rotations(states[nearest:nearest+1])[0].T).magnitude()))))
    return output


def run(config,output):
    a.need(config['experiment']=='path_probe' and config['mode']=='saved_state_inference' and config['device']=='cpu'
           and config['checkpoint_step']==2000 and config['time_limit_seconds']==600,'Unexpected finite CPU scope')
    source=Path(config['run_root']);run_receipt=a.read_json(source/'result.json')
    a.need(run_receipt['completed'] is True and run_receipt['gpu_released'] is True,'Incomplete training source')
    artifacts=load_artifacts(source/'output/configuration.json',source/'output/checkpoint_2000',config['checkpoint_sha256'])
    arrays,row,provenance=load_source(config)
    starts=np.asarray(row['starts']);phases=np.asarray(arrays['phases'][starts]);q=np.asarray(arrays['state'][starts,:7])
    states=np.stack([a.small_state(arrays['state'][s],'R4',str(arrays['phases'][s])) for s in starts])
    reference=np.stack([arrays['actions'][s:s+10] for s in starts])
    service=InferenceService(artifacts,NativeEngine(artifacts))
    predictions=np.stack([service.predict_state(state) for state in states])
    hold=np.repeat(q[:,None],10,axis=1)
    cv=q[:,None]+arrays['state'][starts,7:14][:,None]*np.arange(1,11)[None,:,None]*.05
    metrics={}
    for name,mask in [('ALL',np.ones(len(states),bool))]+[(p,phases==p) for p in ('FREE','APPROACH')]:
        a.need(mask.any(),'Missing declared phase')
        metrics[name]=dict(count=int(mask.sum()),methods={})
        for method,raw in (('act',predictions),('hold',hold),('constant_velocity',cv)):
            difference=np.abs(raw[mask]-reference[mask])
            metrics[name]['methods'][method]=dict(h0_rad=describe(difference[:,0].mean(1)),
                h5_rad=describe(difference[:,:5].mean((1,2))),h10_rad=describe(difference.mean((1,2))),
                signed_joint_bias_by_horizon_rad=(raw[mask]-reference[mask]).mean(0).tolist())
    archive=dict(states=states,source_starts=starts,phases=phases,reference=reference,
                 raw=predictions,hold=hold,constant_velocity=cv)
    online={}
    for binding in config['online_runs']:
        p=Path(binding['path']);owner=a.read_json(p/'result.json')
        a.need(owner['completed'] is True and owner['gpu_released'] is True,'Online source still active')
        value=a.pinned_json(p/'output/case/outcome.json',binding['outcome_sha256'])
        a.need(value['checkpoint_receipt']==artifacts.receipt,'Online source used different policy artifacts')
        queries=value['l2_queries'];a.need(0<len(queries)<=500,'Unexpected query count')
        replay=np.stack([service.predict_state(item['state19']) for item in queries])
        saved=np.asarray([item['raw_absolute_chunk'] for item in queries])
        discrepancy=float(np.max(np.abs(replay-saved)))
        a.need(discrepancy<=1e-6,'Saved online inference is not reproducible')
        support=coverage(queries,states,starts,phases,artifacts.stats)
        prefix=p.name.replace('-','_');archive[prefix+'_replayed']=replay
        online[p.name]=dict(outcome_sha256=binding['outcome_sha256'],queries=len(queries),
            max_replay_difference_rad=discrepancy,phase_counts=dict(Counter(s['phase'] for s in support)),
            support_by_phase={phase:{key:describe([s[key] for s in support if s['phase']==phase]) for key in
                ('normalized_state_l2','joint_max_abs_rad','task_relative_tcp_distance_mm','task_relative_rotation_deg')}
                for phase in sorted(set(s['phase'] for s in support))},nearest_train=support,
            scope='Nearest nominal R4 TRAIN state in frozen normalized feature space; descriptive, not a true oracle or generalization test')
    np.savez_compressed(output/'predictions.npz',**archive)
    report=dict(completed=True,**FLAGS,native_model_loaded=True,device='cpu',
        scope='All qualified R4 TRAIN windows, one fixed checkpoint, no selection or additional fitting',
        receipt=artifacts.receipt,provenance=provenance,train_metrics=metrics,online=online,
        predictions_sha256=a.sha256(output/'predictions.npz'))
    a.save_json(output/'metrics.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if sys.platform!='linux' or os.environ.get('AUSC_OWNED_RUN')!='1':
        raise RuntimeError('Original guarded Linux CPU owner required')
    config=a.read_json(args.config);args.output.mkdir(parents=True,exist_ok=False)
    a.save_json(args.output/'configuration.json',config)
    try:
        run(config,args.output)
        a.save_json(args.output/'COMPLETE.json',dict(completed=True,**FLAGS,
            metrics_sha256=a.sha256(args.output/'metrics.json')))
    except Exception as error:
        a.save_json(args.output/'FAILED.json',dict(completed=False,**FLAGS,error=type(error).__name__,reason=str(error)))
        raise
    print('PATH_PROBE_COMPLETE',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
