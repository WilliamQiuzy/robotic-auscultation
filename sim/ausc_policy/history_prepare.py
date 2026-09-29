"""Recover causal observation history from original, qualified TRAIN captures.

Existing labels and eligibility are immutable. No raw images, model, optimizer
or simulator is used. Historical frames may be ineligible as action origins;
they remain real observations, never new expert action labels.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
from .act_baseline import TRAIN16, pinned_json, sha256, small_state, save_json
from .velocity_probe import PREPARED_SHA

OFFSETS=(-10,-5,-2,-1,0)


def history_frame_indices(frames,origin):
    if type(origin) is not int or not 10<=origin<len(frames):
        raise ValueError('Full actual half-second history required')
    indices=[origin+i for i in OFFSETS];current=frames[origin]
    for index,offset in zip(indices,OFFSETS):
        row=frames[index]
        if (row['index']!=index or row['target']!=current['target']
                or not np.isfinite(row['time_s'])
                or abs(row['time_s']-current['time_s']-offset*.05)>1e-8):
            raise ValueError('Historical frame identity, target or clock differs')
    return indices


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();config=json.loads(a.config.read_text())
    if (sys.platform!='linux' or os.environ.get('AUSC_OWNED_RUN')!='1'
            or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or config['experiment']!='history_prepare'
            or config['offsets_frames']!=list(OFFSETS) or config['prepared_sha256']!=PREPARED_SHA):
        raise ValueError('Declared owned Linux CPU-only causal preparation required')
    prepared=pinned_json(config['prepared'],PREPARED_SHA)
    if tuple(e['episode_id'] for e in prepared['episodes'])!=TRAIN16: raise ValueError('Original TRAIN16 required')
    # The task decoder independently matches this external calibration to each
    # acquisition's retained snapshot. Do not reconstruct Hall or camera data.
    sys.path.append(config['asset_root'])
    from sim.auscultation_pi05_v6.derive_task_dataset import derive_states
    from sim.auscultation_pi05_v6.phase_contract import precontact_phase
    a.output.mkdir(parents=True,exist_ok=False);records=[];total=0
    for e in prepared['episodes']:
        if e['split']!='train': raise ValueError('Heldout forbidden')
        eid=e['episode_id'];source=e['source_record']
        report_path=Path(source['report_path']);report=pinned_json(report_path,source['qualification_sha256'])
        if report['qualified'] is not True or report['arrays_sha256']!=source['sha256']:
            raise ValueError('Original qualification changed')
        derivation=report['evidence']['task_observation_derivation']
        evidence=(report_path.parent/derivation['evidence_root']).resolve()
        if not evidence.is_relative_to(report_path.parent.resolve()):raise ValueError('Evidence path escaped')
        bound={}
        for name,digest in derivation['input_sha256'].items():
            file=(evidence/name).resolve()
            if not file.is_relative_to(evidence) or sha256(file)!=digest:raise ValueError('Changed original history evidence')
            bound[str(file)]=digest
        pairs=report['evidence']['label_pairs']
        frames=[json.loads(line) for line in (evidence/'sensors/frames.jsonl').open()]
        if [f['index'] for f in frames]!=list(range(len(frames))):raise ValueError('Nonconsecutive original frames')
        histories=[history_frame_indices(frames,pairs[start]['frame_index']) for start in e['starts']]
        wanted=sorted(set(i for h in histories for i in h))
        target=source['target']
        if any(frames[i]['target']!=target for i in wanted):raise ValueError('History changed target')
        # Same immutable decoder as the qualified state41, also for real past
        # observations that were excluded only from action-origin selection.
        decoded=derive_states(np.array([frames[i]['state'] for i in wanted],np.float32),
                              [dict(frame_index=i) for i in wanted],evidence)
        decoded_by=dict(zip(wanted,decoded));phases={};current='FREE';active=None
        for frame in frames[:max(wanted)+1]:
            if frame['target']!=active:active=frame['target'];current='FREE'
            if frame['eligible'] and frame['phase'] in ('FREE','APPROACH'):
                current=precontact_phase(current,frame['state'])
            phases[frame['index']]=current
        arrays={}
        for key in ('state','times','phases'):
            item=e['cache'][key]
            if sha256(item['path'])!=item['sha256']:raise ValueError('Original cache changed')
            arrays[key]=np.load(item['path'],allow_pickle=False,mmap_mode='r')
        packed=[];ineligible_history=0
        for start,indices in zip(e['starts'],histories):
            origin=indices[-1];frame=frames[origin]
            if (not np.array_equal(decoded_by[origin],arrays['state'][start])
                    or abs(frame['time_s']-arrays['times'][start])>1e-8
                    or phases[origin]!=str(arrays['phases'][start])):
                raise ValueError('Current history endpoint differs from qualified training input')
            packed.append([small_state(decoded_by[i],target,phases[i]) for i in indices])
            ineligible_history+=sum(not frames[i]['eligible'] for i in indices[:-1])
        folder=a.output/eid;folder.mkdir()
        array=np.asarray(packed,np.float32)
        if array.shape!=(len(e['starts']),5,19) or not np.isfinite(array).all():raise ValueError('Invalid causal history shape')
        np.save(folder/'history.npy',array,allow_pickle=False)
        save_json(folder/'index.json',dict(starts=e['starts'],frame_indices=histories,offsets_frames=list(OFFSETS)))
        for file,digest in bound.items():
            if sha256(file)!=digest:raise ValueError('History source changed during preparation')
        row=dict(episode_id=eid,windows=len(packed),observed_history_slots_ineligible_as_action_origins=ineligible_history,
            history_path=str((folder/'history.npy').resolve()),history_sha256=sha256(folder/'history.npy'),
            index_path=str((folder/'index.json').resolve()),index_sha256=sha256(folder/'index.json'),
            qualification_sha256=source['qualification_sha256'],bound_history_sources=bound)
        records.append(row);total+=len(packed);print(json.dumps({k:row[k] for k in ('episode_id','windows','observed_history_slots_ineligible_as_action_origins')}),flush=True)
    save_json(a.output/'COMPLETE.json',dict(completed=True,episodes=records,total_windows=total,
        prepared_sha256=PREPARED_SHA,offsets_frames=list(OFFSETS),history_shape_per_window=[5,19],
        current_inputs_exactly_reproduced=True,labels_exported=0,new_expert_data=False,
        original_qualification_unchanged=True,optimizer_updates=0,heldout_opened=False,m0_passed=False,
        scope='Causal history input cache for a controlled M0 fit; original qualified labels must be joined by exact episode/start. No policy or new dataset qualification.'))


if __name__=='__main__':main()
