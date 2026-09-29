"""Read-only TRAIN diagnostic: can causal history distinguish near-state pairs?

No training/label changes. Near is a declared metric, not observation identity
or an information-theoretic impossibility claim. No future enters history.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from .act_baseline import small_state, TRAIN16
from .velocity_probe import PREPARED_SHA

LAGS=(.05,.1,.25,.5)


def past_index(times,origin,lag):
    times=np.asarray(times,float)
    if times.ndim!=1 or type(origin) is not int or not 0<=origin<len(times) or lag not in LAGS:
        raise ValueError('Declared causal lag and actual origin required')
    wanted=times[origin]-lag;candidate=int(np.searchsorted(times,wanted))
    indices=[i for i in (candidate-1,candidate) if 0<=i<origin and np.isfinite(times[i]) and abs(times[i]-wanted)<1e-7]
    if len(indices)!=1: raise ValueError('Missing actual prior sample; no padding or interpolation')
    return indices[0]


def near_state(left,right):
    left,right=np.asarray(left),np.asarray(right)
    if left.shape!=(19,) or right.shape!=(19,) or not np.isfinite(np.r_[left,right]).all():
        raise ValueError('Finite state19 pair required')
    delta=abs(left-right)
    return bool(delta[:7].max()<=.0005 and np.linalg.norm(delta[7:10])<=.0002
        and np.linalg.norm(delta[10:16])<=.001 and np.array_equal(left[17:],right[17:]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.inputs.resolve()
    def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    manifest=json.loads((root/'MIRROR_SHA256.json').read_text())
    for name,b in manifest.items():
        file=(root/name).resolve()
        if not file.is_relative_to(root) or file.stat().st_size!=b['bytes'] or sha(file)!=b['sha256']:
            raise ValueError('Changed mirrored TRAIN source')
    if sha(root/'prepared.json')!=PREPARED_SHA: raise ValueError('Changed preparation')
    prepared=json.loads((root/'prepared.json').read_text())
    if tuple(e['episode_id'] for e in prepared['episodes'])!=TRAIN16: raise ValueError('Original TRAIN16 required')
    groups={};episodes={}
    for e in prepared['episodes']:
        if e['split']!='train': raise ValueError('Heldout forbidden')
        eid=e['episode_id'];target=e['source_record']['target'];folder=root/eid
        arrays={k:np.load(folder/(k+'.npy'),allow_pickle=False,mmap_mode='r') for k in ('state','actions','times','phases')}
        for k in arrays:
            if sha(folder/(k+'.npy'))!=e['cache'][k]['sha256']: raise ValueError('Prepared source remapped')
        if not np.isfinite(arrays['times']).all() or not (np.diff(arrays['times'])>0).all(): raise ValueError('Invalid clock')
        states={i:small_state(arrays['state'][i],target,str(phase)) for i,phase in enumerate(arrays['phases']) if phase in ('FREE','APPROACH')}
        episodes[eid]=(arrays,states)
        for start in e['starts']:
            state=states[start];relative=arrays['actions'][start:start+10]-arrays['state'][start,:7]
            groups.setdefault((target,str(arrays['phases'][start])),[]).append((eid,start,state,relative))
    pairs=[]
    for key,rows in groups.items():
        states=np.stack([r[2][:16] for r in rows]);scale=np.r_[np.full(10,.001),np.full(6,.01)]
        _,indices=cKDTree(states/scale).query(states/scale,k=min(16,len(rows)))
        for i,neighbors in enumerate(indices):
            for j in np.atleast_1d(neighbors):
                j=int(j)
                if i>=j: continue
                left,right=rows[i],rows[j]
                if left[0]==right[0] and abs(left[1]-right[1])<=10: continue
                if not near_state(left[2],right[2]): continue
                history={};available=True;still_near=True
                for lag in LAGS:
                    try:
                        values=[]
                        for eid,start,_,_ in (left,right):
                            arrays,allstates=episodes[eid]
                            values.append(allstates[past_index(arrays['times'],start,lag)])
                        still_near=still_near and near_state(*values)
                    except (ValueError,KeyError): available=False
                    history[str(lag)]=dict(available=available,near_all_declared_lags=still_near if available else None)
                pairs.append(dict(target=key[0],phase=key[1],left=list(left[:2]),right=list(right[:2]),
                    relative_h10_mae_rad=float(abs(left[3]-right[3]).mean()),history=history))
    summary={}
    for lag in LAGS:
        usable=[r for r in pairs if r['history'][str(lag)]['available']]
        high=[r for r in usable if r['relative_h10_mae_rad']>=.002]
        summary[str(lag)]=dict(available_pairs=len(usable),missing_history_pairs=len(pairs)-len(usable),
            high_label_difference_pairs=len(high),high_difference_still_near=sum(r['history'][str(lag)]['near_all_declared_lags'] for r in high))
    common=[r for r in pairs if r['history']['0.5']['available']]
    common_high=[r for r in common if r['relative_h10_mae_rad']>=.002]
    matched=dict(pairs=len(common),high_label_difference_pairs=len(common_high),
        high_difference_still_near={'0.0':len(common_high),**{str(lag):sum(r['history'][str(lag)]['near_all_declared_lags'] for r in common_high) for lag in LAGS}})
    result=dict(near_pairs=len(pairs),summary=summary,pairs=pairs,source_prepared_sha256=PREPARED_SHA,
        matched_complete_history_cohort=matched,
        source_mirror_sha256=sha(root/'MIRROR_SHA256.json'),probe_source_sha256=sha(Path(__file__)),
        heldout_opened=False,optimizer_updates=0,labels_changed=False,m0_passed=False,
        search='Same predeclared nearest16 current-state screen. Add cumulative causal lags 50/100/250/500ms; missing samples excluded explicitly.',
        limitation='Descriptive sensitivity to history, not exact state equivalence, a learned policy, causal proof or instruction to discard labels.')
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(near_pairs=len(pairs),summary=summary)))


if __name__=='__main__': main()
