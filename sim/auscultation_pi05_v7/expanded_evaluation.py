"""Declared 50-episode experiment: development predictions on train/validation only.

Integrity checks may validate sealed test artifacts. This entry point never
samples a test observation. Test inference is exclusively in expanded_final_test.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from sim.auscultation_pi05_v6.action_contract import LOWER, UPPER

PHASES = ('FREE', 'APPROACH', 'CONTACT', 'HOLD', 'RETRACT')
TARGETS = frozenset(side+str(i) for i in range(1, 6) for side in ('R', 'L'))
STEPS = tuple(range(1000, 12001, 1000))
SAMPLER = dict(seed=917, sampling_steps=10, fixed_noise_per_query=True)
SCHEMA = 'wam_pi05_v7_expanded_numeric_evaluation_v1'


def require(condition, message):
    if not condition: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    with path.open('x') as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')
        stream.flush(); os.fsync(stream.fileno())


def assert_same(actual, expected, label):
    # Report numbers are deterministic recomputations from the saved raw arrays.
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), label+' keys differ')
        for key, value in expected.items(): assert_same(actual[key], value, label+'/'+key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label+' length differs')
        for i, value in enumerate(expected): assert_same(actual[i], value, label+'/'+str(i))
    elif isinstance(expected, float):
        require(type(actual) in (int, float) and math.isfinite(actual) and
                math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), label+' numeric score differs')
    else:
        require(type(actual) == type(expected) and actual == expected, label+' differs')


def _row(dataset, split, index):
    ei, start = dataset.indices[split][index]; e = dataset.episodes[ei]
    require(e['split'] == split, 'Episode/sample split differs')
    require(type(start) in (int, np.int64, np.int32) and start >= 0, 'Invalid window origin')
    phase = str(e['phases'][start]); target = str(e['targets'][start])
    require(phase in PHASES and target in TARGETS, 'Unknown phase/target')
    require(len(e['phases'][start:start+10]) == 10 and
            all(str(x) == phase for x in e['phases'][start:start+10]), 'Window crosses phase')
    require(all(str(x) == target for x in e['targets'][start:start+10]), 'Window crosses target')
    return dict(episode_id=e['episode_id'], split=split, episode_index=int(ei),
                dataset_index=int(index), frame=int(start), phase=phase, target=target)


def _phase_schedule(dataset, split):
    require(split in ('validation', 'test'), 'Phase schedule only for validation/final test')
    episodes = [(i,e) for i,e in enumerate(dataset.episodes) if e['split'] == split]
    require(len(episodes) == 10, 'Exactly ten '+split+' episodes required')
    all_rows = [_row(dataset, split, i) for i in range(len(dataset.indices[split]))]
    require(len({(r['episode_id'],r['frame']) for r in all_rows}) == len(all_rows), 'Duplicate window')
    rows=[]; seen_targets=[]
    for ei, e in sorted(episodes, key=lambda x: x[1]['episode_id']):
        own=[r for r in all_rows if r['episode_index']==ei]
        targets={r['target'] for r in own}; require(len(targets)==1, 'Missing or mixed episode target')
        seen_targets.extend(targets)
        for phase in PHASES:
            bucket=sorted((r for r in own if r['phase']==phase),key=lambda r:r['frame'])
            require(bucket, 'empty '+split+' episode/phase bin: '+e['episode_id']+'/'+phase)
            positions=np.rint(np.linspace(0,len(bucket)-1,min(20,len(bucket)))).astype(int)
            require(len(set(positions))==len(positions), 'Duplicate selected positions')
            rows.extend(bucket[int(p)] for p in positions)
    require(set(seen_targets)==TARGETS and len(set(seen_targets))==10, 'All ten targets required once')
    return rows


def build_validation_schedule(dataset):
    return _phase_schedule(dataset, 'validation')


def build_train_sentinels(dataset):
    episodes=[(i,e) for i,e in enumerate(dataset.episodes) if e['split']=='train']
    require(len(episodes)==30, 'Exactly thirty train episodes required')
    all_rows=[_row(dataset,'train',i) for i in range(len(dataset.indices['train']))]
    require(len({(r['episode_id'],r['frame']) for r in all_rows})==len(all_rows), 'Duplicate train window')
    rows=[]
    for ei,e in sorted(episodes,key=lambda x:x[1]['episode_id']):
        own=[r for r in all_rows if r['episode_index']==ei and r['phase']=='FREE']
        require(own,'No first eligible FREE sentinel for '+e['episode_id'])
        rows.append(min(own,key=lambda r:r['frame']))
    require(Counter(r['target'] for r in rows)=={t:3 for t in TARGETS}, 'Three training starts per target required')
    return rows


def reference_arrays(dataset, rows):
    require(rows, 'Nonempty rows required')
    refs=[]; anchors=[]
    for row in rows:
        expected=_row(dataset,row['split'],row['dataset_index'])
        require(row==expected,'Row differs from actual qualified index')
        e=dataset.episodes[row['episode_index']]; s=row['frame']
        refs.append(e['actions'][s:s+10]); anchors.append(e['state'][s,None,:7])
    refs,anchors=map(np.asarray,(refs,anchors))
    require(refs.shape==(len(rows),10,7) and anchors.shape==(len(rows),1,7) and
            np.isfinite(refs).all() and np.isfinite(anchors).all(),'Invalid actual reference/anchor arrays')
    return refs,anchors


def bounds_report(prediction):
    p=np.asarray(prediction)
    require(p.ndim==3 and p.shape[1:]==(10,7) and len(p)>0 and p.dtype.kind=='f', 'Expected floating [N,10,7]')
    require(np.isfinite(p).all(), 'Nonfinite prediction: no feasible selection permitted')
    invalid=(p<np.asarray(LOWER))|(p>np.asarray(UPPER))
    return dict(all_passed=not bool(invalid.any()), full_chunk_bound_pass_fraction=float(np.mean(~invalid.any(axis=(1,2)))),
        first_action_bound_pass_fraction=float(np.mean(~invalid[:,0].any(axis=1))),
        coordinate_count=int(p.size), violating_coordinates=int(invalid.sum()),
        violations=[dict(record_index=int(i),horizon_index=int(h),joint_index=int(j),proposed_rad=float(p[i,h,j]))
                    for i,h,j in np.argwhere(invalid)],
        per_joint_min_margin_rad=np.minimum(p-np.asarray(LOWER),np.asarray(UPPER)-p).min(axis=(0,1)).tolist(),
        lower_rad=list(LOWER),upper_rad=list(UPPER),tolerance_rad=0.,clipping=False)


def score_arrays(prediction, reference, anchor, rows):
    p,r,a=map(lambda x:np.asarray(x,dtype=np.float64),(prediction,reference,anchor))
    require(p.shape==r.shape==(len(rows),10,7) and a.shape==(len(rows),1,7) and
            np.isfinite([p,r]).all() and np.isfinite(a).all(),'Invalid numeric scoring arrays')
    error=np.abs(p-r); hold=np.abs(a-r)
    buckets=[]
    for eid,phase in sorted({(row['episode_id'],row['phase']) for row in rows}):
        ix=np.array([(row['episode_id'],row['phase'])==(eid,phase) for row in rows])
        delta=r[ix]-a[ix]; proposed=p[ix]-a[ix]; denominator=float(np.sum(delta**2)); dot=float(np.sum(delta*proposed))
        buckets.append(dict(episode_id=eid,phase=phase,target=rows[int(np.flatnonzero(ix)[0])]['target'],count=int(ix.sum()),
            first_action_mae_rad=float(error[ix,0].mean()),full_chunk_mae_rad=float(error[ix].mean()),
            max_abs_error_rad=float(error[ix].max()),
            measured_hold_first_mae_rad=float(hold[ix,0].mean()),measured_hold_full_mae_rad=float(hold[ix].mean()),
            reference_motion_l2_rad=float(np.sqrt(denominator)),
            squared_error_relative_to_measured_hold=None if denominator<=1e-20 else float(np.sum((p[ix]-r[ix])**2))/denominator,
            least_squares_motion_scale=None if denominator<=1e-20 else dot/denominator))
    return dict(macro_first_action_mae_rad=float(np.mean([b['first_action_mae_rad'] for b in buckets])),
        macro_full_chunk_mae_rad=float(np.mean([b['full_chunk_mae_rad'] for b in buckets])),
        micro_first_action_mae_rad=float(error[:,0].mean()),micro_full_chunk_mae_rad=float(error.mean()),
        bucket_count=len(buckets),buckets=buckets,bounds=bounds_report(p))


def evaluate_development_rows(dataset, rows, predict):
    require(rows and all(r['split'] in ('train','validation') for r in rows), 'Development cannot predict test rows')
    return _predict_rows(dataset,rows,predict)


def _predict_rows(dataset,rows,predict):
    """Internal sampling primitive. Final-test caller must claim its one-shot lease."""
    reference_arrays(dataset,rows)  # Validate all identities before any prediction.
    values=[]
    for row in rows:
        e=dataset.episodes[row['episode_index']]; s=row['frame']
        result=predict(dataset.rgb(row['episode_index'])[s],e['state'][s],row['phase'],row['target'])
        value=result[0] if isinstance(result,tuple) else result
        value=np.asarray(value)
        require(value.shape==(10,7) and value.dtype.kind=='f' and np.isfinite(value).all(),'Invalid raw decoded prediction')
        values.append(value.copy())
    return np.asarray(values)


def save_numeric_report(output,pred,ref,anchor,rows,*,kind,binding):
    require(kind in ('validation','train_sentinel','final_test'),'Unknown evaluation kind')
    expected_split={'validation':'validation','train_sentinel':'train','final_test':'test'}[kind]
    require(rows and all(r['split']==expected_split for r in rows),'Report split/kind mismatch')
    bounds=bounds_report(pred)
    require(np.asarray(ref).shape==np.asarray(pred).shape and np.asarray(anchor).shape==(len(rows),1,7),'Numeric array shape mismatch')
    scores=None if kind=='train_sentinel' else score_arrays(pred,ref,anchor,rows)
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'predictions.npz',prediction=pred,reference=ref,measured_anchor=anchor)
    report=dict(schema=SCHEMA,kind=kind,binding=binding,sampler=SAMPLER,records=rows,
        records_sha256=json_sha(rows),predictions_sha256=sha(output/'predictions.npz'),bounds=bounds,scores=scores,
        train_sentinel_used_for_ranking=False,closed_loop_validated=False)
    write_new(output/'report.json',report)
    return report


def verify_numeric_report(path,dataset,rows,*,kind,binding):
    path=Path(path); report=read(path)
    for key,value in dict(schema=SCHEMA,kind=kind,binding=binding,sampler=SAMPLER,records=rows,
            records_sha256=json_sha(rows),train_sentinel_used_for_ranking=False,closed_loop_validated=False).items():
        require(report.get(key)==value,'Numeric report binding differs: '+key)
    numeric=path.with_name('predictions.npz'); require(sha(numeric)==report['predictions_sha256'],'Numeric SHA changed')
    with np.load(numeric,allow_pickle=False) as arrays:
        require(set(arrays.files)=={'prediction','reference','measured_anchor'},'Unexpected numeric keys')
        p,r,a=(arrays[k] for k in ('prediction','reference','measured_anchor'))
    refs,anchors=reference_arrays(dataset,rows)
    require(p.shape==(len(rows),10,7),'Prediction row count differs')
    require(np.array_equal(r,refs) and np.array_equal(a,anchors),'Numeric reference/anchor differs from fresh source')
    assert_same(report.get('bounds'),bounds_report(p),'bounds')
    assert_same(report.get('scores'),None if kind=='train_sentinel' else score_arrays(p,r,a,rows),'scores')
    return report


def require_fresh_output(output,binding,*,source_root=None):
    output=Path(output).resolve()
    protected=[Path(binding['run']).resolve(),Path(binding['manifest_path']).resolve().parent,
        Path(binding['experiment_path']).resolve().parent,
        Path(__file__).resolve().parent if source_root is None else Path(source_root).resolve()]
    for root in protected:
        require(output!=root and root not in output.parents and output not in root.parents,
            'Output overlaps protected input/source tree: '+str(root))
    require(not output.exists(), 'Fresh output required')
    return output


def checkpoint_inventory(path):
    return {name:sha(Path(path)/name) for name in ('metadata.json','arrays.npz','COMPLETE.json')}


def verify_training_run_identity(config,run):
    require(config.get('training_run_path')==str(Path(run).resolve(strict=True)),
        'Copied run is not the bound original training run / marker namespace')


def load_run_context(manifest,run,experiment):
    """CPU-only complete-source/data/plan/checkpoint verification, no model creation."""
    from .expanded_training_data import load_dataset
    from .expanded_actor import verify_configuration
    from . import expanded_contract as contract
    from sim.auscultation_pi05_v6 import train_openpi as parent
    manifest,run=Path(manifest).resolve(strict=True),Path(run).resolve(strict=True)
    experiment=Path(experiment).resolve(strict=True)
    dataset=load_dataset(manifest,experiment); record=read(run/'configuration.json'); config=record['config']
    verify_configuration(dataset,config)
    verify_training_run_identity(config,run)
    require(record['data_hash']==dataset.data_hash and record['normalization']==dataset.normalization and
            record['manifest_sha256']==sha(manifest) and record.get('experiment_sha256')==sha(experiment) and
            Path(record.get('experiment','')).resolve()==experiment,'Training dataset/normalization differs')
    receipt=read(run/'training_run.json')
    require(receipt.get('schema')=='wam_pi05_v7_expanded_training_run_v1' and
            receipt.get('data_hash')==dataset.data_hash and receipt.get('experiment_binding')==dataset.experiment_binding and
            receipt.get('experiment_source_sha256')==dataset.experiment_source_sha256,'Training receipt binding differs')
    require(receipt.get('training_complete') is True and receipt.get('step')==12000 and not receipt.get('error'),
            'Complete successful 12000-step training required')
    require(config.get('seed')==61 and config.get('learning_rate')==5e-5 and config.get('batch_size')==4,
            'Predeclared optimizer settings differ')
    plan=contract.training_plan()
    require(tuple(plan['candidate_steps'])==STEPS and plan['sampling_seed']==SAMPLER['seed'] and
            plan['sampling_steps']==SAMPLER['sampling_steps'],'Evaluator and frozen plan differ')
    schedules=dict(experiment_binding=dataset.experiment_binding,data_hash=dataset.data_hash,
        train_sentinels=build_train_sentinels(dataset),validation=build_validation_schedule(dataset))
    require(read(run/'predeclared_schedules.json')==schedules and
        sha(run/'predeclared_schedules.json')==config.get('predeclared_schedules_sha256'),
        'Pre-training schedule differs from qualified dataset')
    checkpoints=sorted((run/'model').glob('step-*'))
    require(len(checkpoints)==len(STEPS),'Exactly twelve scheduled checkpoints required')
    inventory={}; metadata=[]
    for path,step in zip(checkpoints,STEPS):
        meta=parent._verified_checkpoint_metadata(path,data_hash=dataset.data_hash,config=config,normalization=dataset.normalization)
        require(meta['step']==step,'Checkpoint schedule differs')
        inventory[str(path)]=checkpoint_inventory(path); metadata.append(meta)
    source=dict(dataset.experiment_source_sha256)
    binding=dict(manifest_path=str(manifest),manifest_sha256=sha(manifest),experiment_path=str(experiment),
        experiment_sha256=sha(experiment),data_hash=dataset.data_hash,
        parent_data_hash=dataset.parent.data_hash,normalization_sha256=json_sha(dataset.normalization),
        experiment_binding=dataset.experiment_binding,experiment_source_sha256=source,training_plan=plan,
        configuration_sha256=sha(run/'configuration.json'),training_run_sha256=sha(run/'training_run.json'),
        run=str(run),predeclared_schedules_sha256=sha(run/'predeclared_schedules.json'),checkpoint_inventory=inventory)
    return dict(dataset=dataset,config=config,binding=binding,checkpoints=checkpoints,metadata=metadata)


def verify_context_unchanged(context):
    from . import expanded_contract as contract
    b=context['binding']
    require(sha(b['manifest_path'])==b['manifest_sha256'],'Manifest changed during evaluation')
    require(sha(b['experiment_path'])==b['experiment_sha256'],'Experiment sidecar changed')
    run=Path(b['run'])
    for key,file in [('configuration_sha256','configuration.json'),('training_run_sha256','training_run.json'),
                     ('predeclared_schedules_sha256','predeclared_schedules.json')]:
        require(sha(run/file)==b[key],'Training input changed')
    for path,expected in b['checkpoint_inventory'].items(): require(checkpoint_inventory(path)==expected,'Checkpoint changed')
    require(contract.experiment_source_hashes()==b['experiment_source_sha256'],'Experiment source changed')
    from .training_data import core_sources
    require(core_sources()==context['config']['source_sha256'],'Inherited training core changed')


def make_predictor(context,runtime,sample):
    import jax
    from openpi.models import model as model_api
    from .training_data import observation,stack_observations,decode_prediction
    dataset=context['dataset']
    def predict(rgb,state,phase,target):
        raw=np.asarray(state,np.float32)
        row=observation(rgb,raw,phase,target,runtime['tokenizer'],dataset.normalization)
        normalized=np.asarray(sample(runtime['trainable'],runtime['frozen'],
            model_api.Observation(**jax.device_put(stack_observations([row]))),jax.random.key(917)))
        require(normalized.shape==(1,10,7) and np.isfinite(normalized).all(),'Invalid sampled flow tensor')
        return decode_prediction(dataset,normalized[0],raw,phase)
    return predict


def run(args):
    from . import model
    context=load_run_context(args.manifest,args.run,args.experiment); d=context['dataset']
    validation=build_validation_schedule(d); sentinels=build_train_sentinels(d)
    output=require_fresh_output(args.output,context['binding']); output.mkdir(parents=True,exist_ok=False)
    write_new(output/'started.json',dict(schema='wam_pi05_v7_expanded_evaluation_started_v1',binding=context['binding']))
    runtime=model.initialize(61,5e-5)
    require(runtime['assets']==context['config']['assets'],'Official base assets differ')
    sample=model.sampler(runtime,steps=10); predict=make_predictor(context,runtime,sample)
    reports=[]
    for path,meta in zip(context['checkpoints'],context['metadata']):
        model.restore(runtime,path,d.data_hash,context['config'],d.normalization)
        binding=dict(context['binding'],step=meta['step'],checkpoint=str(path))
        for kind,rows in [('validation',validation),('train_sentinel',sentinels)]:
            pred=evaluate_development_rows(d,rows,predict); ref,anchor=reference_arrays(d,rows)
            report_path=output/('step_%06d'%meta['step'])/kind
            save_numeric_report(report_path,pred,ref,anchor,rows,kind=kind,binding=binding)
            reports.append(dict(step=meta['step'],kind=kind,path=str(report_path/'report.json'),sha256=sha(report_path/'report.json')))
    verify_context_unchanged(context)
    result=dict(schema='wam_pi05_v7_expanded_evaluation_completed_v1',binding=context['binding'],
        validation_schedule=validation,train_sentinel_schedule=sentinels,reports=reports,
        evaluation_complete=True,test_predictions_generated=False,closed_loop_validated=False)
    write_new(output/'evaluation.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('manifest','experiment','run','output'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args()
    from sim.auscultation_pi05_v6.train_openpi import ROOT
    import fcntl,subprocess
    with (ROOT/'locks/isaac_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU busy')
        result=run(args); print(json.dumps(dict(evaluation_complete=result['evaluation_complete'])),flush=True)


if __name__=='__main__':main()
