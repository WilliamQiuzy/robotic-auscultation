"""Verify all source bindings and sequentially decode every review-video frame."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from sim.expert_quality_20260924.visual_judge import validate_review, decode_samples, sha256
from .quality import validate_episode_provenance,verify_hashes

PROJECT=Path(__file__).resolve().parents[2]


def _require(condition,message):
    if not condition:raise ValueError(message)


def _bound_source(data,name,project):
    matches=[(Path(project)/relative).resolve() for relative in data['source_sha256'] if Path(relative).name==name]
    _require(len(matches)==1,'exactly one bound '+name+' required')
    return matches[0]


def _same_number(actual,expected,label):
    _require(type(actual) in (int,float) and math.isfinite(actual)
             and type(expected) in (int,float) and math.isfinite(expected)
             and abs(actual-expected)<1e-8,'frame source mismatch: '+label)


def validate_review_sources(manifest,project=PROJECT):
    """Bind every displayed frame to the saved pose, trace and point interval."""
    data,items=validate_review(manifest,project=project)
    _require(abs(data['source_physics_dt_s']-1/240)<1e-12 and abs(data['source_pose_dt_s']-1/30)<1e-12,
             'unexpected authoritative source clocks')
    trace=_bound_source(data,'expert_trace.jsonl',project)
    replay=_bound_source(data,'expert_replay.npz',project)
    outcome_path=_bound_source(data,'outcome.json',project)
    _require(trace.parent==replay.parent==outcome_path.parent,'mixed episode source files')
    outcome=json.loads(outcome_path.read_text())
    points=outcome.get('points',[])
    _require(isinstance(points,list) and len({p['target'] for p in points})==len(points),'invalid point inventory')
    points={p['target']:p for p in points}
    intervals={};wanted=set()
    for item in items:
        clip=item['clip'];target=clip['target']
        _require(target in points,'clip target absent from outcome')
        _require(clip['clip_id']==target+'_'+clip['view'],'clip identity differs from target/view')
        start,end=(points[target][key] for key in ('start_physics_row','end_physics_row'))
        _require(type(start) is int and type(end) is int and 0<=start<end,'invalid point interval')
        _require(clip['point_start_physics_row']==start and clip['point_end_physics_row']==end,'clip point interval mismatch')
        expected_count=int(math.floor((end-start)/240*data['fps']+1e-7))
        _require(clip['frame_count']==expected_count,'clip omits part of bound point interval')
        intervals[target]=(start,end);wanted.add(start)
        for frame in clip['frames']:
            _require(start<=frame['measurement_row']<end,'frame lies outside point interval')
            wanted.update((frame['measurement_row'],frame['pose_row']))
    rows={};count=0
    with trace.open() as stream:
        for index,line in enumerate(stream):
            row=json.loads(line);count=index+1
            for target,(start,end) in intervals.items():
                if start<=index<end:_require(row['target']==target,'trace point interval target mismatch')
            if index in wanted:rows[index]=row
    _require(all(end<=count for _,end in intervals.values()) and wanted<=rows.keys(),'frame references absent trace rows')
    with np.load(replay,allow_pickle=False) as archive:
        indices=np.asarray(archive['row_indices']);poses=np.asarray(archive['poses']);paths=np.asarray(archive['paths'])
        _require(indices.ndim==1 and np.issubdtype(indices.dtype,np.integer) and len(indices)>0
                 and np.all(np.diff(indices)>0) and 0<=indices[0] and indices[-1]<count,'invalid replay row inventory')
        _require(paths.ndim==1 and len(paths)>0 and len(set(map(str,paths)))==len(paths),'invalid replay body inventory')
        _require(poses.shape==(len(indices),len(paths),7) and np.isfinite(poses).all(),'invalid replay poses')
        _require(np.all(np.abs(np.linalg.norm(poses[:,:,3:],axis=2)-1)<1e-4),'invalid replay rotation')
    for item in items:
        clip=item['clip'];start,end=intervals[clip['target']]
        origin=rows[start]['time_s']-1/240
        for frame in clip['frames']:
            row=rows[frame['measurement_row']]
            index=frame.get('pose_index')
            _require(type(index) is int and 0<=index<len(indices),'frame replay pose_index outside archive')
            _require(int(indices[index])==frame['pose_row'],'frame pose_row differs from replay index')
            _require(index==int(np.searchsorted(indices,frame['measurement_row'],side='right')-1),'frame did not use latest preceding replay pose')
            _same_number(frame['pose_time_s'],rows[int(indices[index])]['time_s'],'pose time')
            _same_number(frame['measurement_time_s'],row['time_s'],'measurement time')
            _same_number(frame['simulation_interval_end_s'],origin+(frame['frame_index']+1)/data['fps'],'point frame clock')
            expected_row=start+int(math.floor((frame['frame_index']+1)*240/data['fps']+1e-7))-1
            _require(frame['measurement_row']==expected_row,'frame omitted or shifted physical samples')
            for key in ('phase','subphase'):_require(frame.get(key)==row[key],'frame source mismatch: '+key)
            _same_number(frame['hall_N'],row['hall_N'],'hall_N')
            front=sum(float(row[key]) for key in ('lip_N','diaphragm_N','contact_rim_N'))
            _same_number(frame['front_N'],front,'front_N')
    return data,items


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('--output',type=Path,help='receipt destination; use a new package path for frozen historical media')
    args=parser.parse_args()
    manifest=args.manifest.resolve(); before=sha256(manifest);verifier_before=sha256(Path(__file__))
    data,items=validate_review_sources(manifest,project=PROJECT)
    episode=_bound_source(data,'outcome.json',PROJECT).parent
    provenance=validate_episode_provenance(episode)
    _require(provenance['status']=='passed','episode provenance rejected: '+str(provenance.get('reasons')))
    records=[]
    for item in items:
        _,sampling=decode_samples(item,data['fps'],data['resolution'],8)
        records.append(dict(clip=item['clip']['clip_id'],frame_count=item['clip']['frame_count'],
                            sha256=item['video_sha256'],decoded_full=True,
                            sampled_rgb_sha256=sampling['decoded_rgb_sha256']))
    validate_review_sources(manifest,project=PROJECT)
    if sha256(manifest)!=before:raise ValueError('manifest changed during verification')
    _require(sha256(Path(__file__))==verifier_before,'media verifier changed during verification')
    _require(not verify_hashes(episode,provenance['artifact_sha256']),'episode changed during media verification')
    report=dict(source_bindings_passed=True,manifest_sha256=before,videos=records,
                verifier_sha256=verifier_before,semantic_binding={'status':'passed',
                    'frames_verified':sum(item['clip']['frame_count'] for item in items),
                    'checks':['replay_index_and_row','trace_target_phase_subphase','Hall_and_front_force','point_interval_and_original_clock']},
                scope='episode provenance, every frame source semantic binding, dimensions, rate and complete sequential decode; no policy acceptance')
    output=args.output.resolve() if args.output else manifest.parent/'media_verification.json'
    protected={manifest,*[(PROJECT/p).resolve() for p in data['source_sha256']],
               *[Path(item['video_path']).resolve() for item in items]}
    _require(output not in protected and output!=episode and episode not in output.parents,
             'receipt must not overwrite evidence or immutable episode')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'verified_videos':len(records),'manifest':str(manifest)}),flush=True)


if __name__=='__main__':main()
