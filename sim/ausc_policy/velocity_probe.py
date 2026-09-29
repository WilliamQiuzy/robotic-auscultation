"""CPU-only causal velocity baseline audit on SHA-bound original TRAIN arrays.

Never infer a policy, train, open heldout data, or modify the original scores.
Position history uses only t and t-50ms, never a future/centered difference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .act_baseline import TRAIN16

PREPARED_SHA = 'f4d72efa2a1aec07c47e5f8c068f4406201204946fe109f7aea926f9836d889b'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def causal_velocity(q, times, origin):
    q, times = np.asarray(q), np.asarray(times)
    if (q.ndim != 2 or q.shape[1] != 7 or times.shape != (len(q),)
            or type(origin) is not int or not 1 <= origin < len(q)):
        raise ValueError('Two actual prior/current observations required')
    positions = np.asarray(q[origin-1:origin+1],float)
    clocks = np.asarray(times[origin-1:origin+1],float)
    if (not np.isfinite(positions).all() or not np.isfinite(clocks).all()
            or abs(clocks[1]-clocks[0]-.05) > 1e-8):
        raise ValueError('Nonfinite or nonconsecutive observation history')
    return (positions[1]-positions[0])/(clocks[1]-clocks[0])


def proposals(q, reported_dq, history_dq):
    q, reported_dq, history_dq = [np.asarray(v,float) for v in (q,reported_dq,history_dq)]
    if any(v.shape != (7,) or not np.isfinite(v).all() for v in (q,reported_dq,history_dq)):
        raise ValueError('Finite seven-joint observations required')
    horizons = np.arange(1,11)[:,None]*.05
    return dict(hold=np.broadcast_to(q,(10,7)).copy(),
                api_dq_cv=q+reported_dq*horizons,
                position_history_cv=q+history_dq*horizons)


def summarize(records):
    result = {}
    for phase in ('ALL','FREE','APPROACH'):
        rows = [r for r in records if phase == 'ALL' or r['phase'] == phase]
        if not rows:
            continue
        methods = {}
        for method in rows[0]['errors']:
            errors = np.stack([r['errors'][method] for r in rows])
            methods[method] = {}
            for name, values in [('h1',errors[:,0].mean(axis=1)),
                                 ('h10',errors.mean(axis=(1,2)))]:
                methods[method][name] = dict(median_rad=float(np.median(values)),
                    mean_rad=float(np.mean(values)),p95_rad=float(np.quantile(values,.95)))
        differences = np.stack([r['reported_dq']-r['history_dq'] for r in rows])
        result[phase] = dict(samples=len(rows),methods=methods,
            velocity_difference_mae_rad_s=float(abs(differences).mean()),
            velocity_difference_peak_rad_s=float(abs(differences).max()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--act-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    root = args.inputs.resolve()
    manifest = json.loads((root/'MIRROR_SHA256.json').read_text())
    for name,binding in manifest.items():
        path = (root/name).resolve()
        if (not path.is_relative_to(root) or path.stat().st_size != binding['bytes']
                or sha(path) != binding['sha256']):
            raise ValueError('Changed mirrored input: '+name)
    prepared = root/'prepared.json'
    if sha(prepared) != PREPARED_SHA:
        raise ValueError('Original preparation changed')
    data = json.loads(prepared.read_text())
    if tuple(e['episode_id'] for e in data['episodes']) != TRAIN16:
        raise ValueError('Original TRAIN16 order/identity changed')
    act = args.act_run
    probe = json.loads((act/'probe_rows.json').read_text())
    if len(probe) != 32 or len({(p['episode_id'],p['start']) for p in probe}) != 32:
        raise ValueError('Exact unique 32-observation ACT probe required')
    with np.load(act/'checkpoint_2000/predictions.npz',allow_pickle=False) as arrays:
        saved = {k: arrays[k] for k in ('raw','reference','measured_q','measured_dq')}
    if saved['raw'].shape != (32,10,7) or not np.isfinite(saved['raw']).all():
        raise ValueError('Saved original ACT predictions missing')
    lookup = {(r['episode_id'],r['start']):i for i,r in enumerate(probe)}
    all_rows,probe_rows,exclusions,seen = [],[],[],set()
    for episode in data['episodes']:
        eid = episode['episode_id']
        if episode['split'] != 'train':
            raise ValueError('Heldout arrays forbidden')
        arrays = {}
        for key in ('state','actions','times','phases'):
            path = root/eid/(key+'.npy')
            if sha(path) != episode['cache'][key]['sha256']:
                raise ValueError('Cached array binding changed')
            arrays[key] = np.load(path,allow_pickle=False,mmap_mode='r')
        state,commands,times,phases = [arrays[k] for k in ('state','actions','times','phases')]
        if (state.shape != (len(times),41) or commands.shape != (len(times),7)
                or phases.shape != times.shape):
            raise ValueError('Original array shape differs')
        for start in episode['starts']:
            index = lookup.get((eid,start))
            q,dq = state[start,:7],state[start,7:14]
            reference = commands[start:start+10]
            if (reference.shape != (10,7) or not np.isfinite(reference).all()
                    or phases[start] not in ('FREE','APPROACH')
                    or not np.allclose(np.diff(times[start:start+10]),.05,atol=1e-8,rtol=0)):
                raise ValueError('Changed original eligible L2 window')
            if index is not None:
                seen.add(index)
                if not (np.array_equal(q,saved['measured_q'][index])
                        and np.array_equal(dq,saved['measured_dq'][index])
                        and np.array_equal(reference,saved['reference'][index])):
                    raise ValueError('ACT prediction/reference pairing differs')
            try:
                velocity = causal_velocity(state[:,:7],times,start)
            except ValueError as error:
                exclusions.append(dict(episode_id=eid,start=start,reason=str(error),probe=index))
                continue
            candidates = proposals(q,dq,velocity)
            errors = {k:abs(v-reference) for k,v in candidates.items()}
            row = dict(episode_id=eid,start=start,phase=str(phases[start]),time_s=float(times[start]),
                reported_dq=dq,history_dq=velocity,errors=errors)
            all_rows.append(row)
            if index is not None:
                probe_rows.append(dict(row,errors=dict(errors,act=abs(saved['raw'][index]-reference))))
    if seen != set(range(32)):
        raise ValueError('Saved probe outside original eligible TRAIN windows')
    report = dict(scope='Read-only original TRAIN16 velocity-definition diagnostic; single body',
        m0_passed=False,optimizer_updates=0,heldout_arrays_opened=False,
        prepared_sha256=PREPARED_SHA,mirror_manifest_sha256=sha(root/'MIRROR_SHA256.json'),
        act_prediction_sha256=sha(act/'checkpoint_2000/predictions.npz'),
        act_probe_rows_sha256=sha(act/'probe_rows.json'),
        history_definition='Backward difference of t and t-50ms measured q; no future, no smoothing, no gaps filled',
        original_eligible_windows=sum(len(e['starts']) for e in data['episodes']),
        exclusions=exclusions,full_baselines=summarize(all_rows),matched_act_probe=summarize(probe_rows),
        input_fairness='ACT takes current state19, not dq/history; baselines are information-rich diagnostics, not matched-input policies',
        target_semantics='Errors against future expert joint COMMANDS, not future actual measured positions',
        interpretation='API dq, measured finite difference and commanded motion are separate quantities; none alone certifies learned movement or physics fidelity')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    methods=('hold','api_dq_cv','position_history_cv','act')
    names=('Hold','API dq CV','Measured-history CV','ACT checkpoint 2000')
    for ax,phase in zip(axes,('FREE','APPROACH')):
        group=report['matched_act_probe'][phase]
        values=[group['methods'][m]['h10']['median_rad'] for m in methods]
        bars=ax.bar(np.arange(4),np.asarray(values)*1000,color=['#b8b1a2','#ac614c','#36777d','#3f5877'])
        ax.bar_label(bars,labels=[f'{v*1000:.3f}' for v in values],padding=3)
        ax.set(xticks=np.arange(4),xticklabels=names,ylabel='H10 per-observation MAE median (mrad)',
               title=f"{phase} — {group['samples']} matched TRAIN observations")
        ax.tick_params(axis='x',labelrotation=20)
        ax.spines[['right','top']].set_visible(False)
        ax.set_ylim(0,max(values)*1300)
    fig.suptitle('Two velocity definitions; predictions evaluated against expert commands')
    for suffix in ('png','svg'): fig.savefig(args.output/('comparison.'+suffix),dpi=150)
    plt.close(fig)
    print(json.dumps({k:report[k] for k in ('full_baselines','matched_act_probe')}))


if __name__ == '__main__':
    main()
