import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

from sim.curobo_expert_v1 import verify_media as subject


def fixture(tmp_path):
    episode=tmp_path/'episode';episode.mkdir()
    trace=episode/'expert_trace.jsonl'
    rows=[{'time_s':(i+1)/240,'target':'R1','phase':'HOLD','subphase':'contact',
           'hall_N':1.,'lip_N':.5,'diaphragm_N':.25,'contact_rim_N':.25} for i in range(32)]
    trace.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    replay=episode/'expert_replay.npz'
    poses=np.zeros((4,1,7));poses[:,:,3]=1.
    np.savez(replay,row_indices=np.array([0,8,16,24]),poses=poses,paths=np.array(['/body']))
    outcome=episode/'outcome.json';outcome.write_text(json.dumps({'points':[{'target':'R1','start_physics_row':0,'end_physics_row':32}]}))
    video=tmp_path/'clip.mp4';video.write_bytes(b'video identity fixture; no decode claimed by semantic check')
    frames=[{'frame_index':i,'video_time_s':i/15,'simulation_interval_end_s':(i+1)/15,
          'measurement_row':15+16*i,'measurement_time_s':(i+1)/15,
          'pose_index':1+2*i,'pose_row':8+16*i,'pose_time_s':(9+16*i)/240,
          'phase':'HOLD','subphase':'contact','hall_N':1.,'front_N':1.} for i in range(2)]
    clip={'clip_id':'R1_posture','target':'R1','view':'posture','video':'clip.mp4','video_sha256':subject.sha256(video),
          'frame_count':2,'duration_s':2/15,'point_start_physics_row':0,'point_end_physics_row':32,'frames':frames}
    manifest={'schema':'blind_recorded_review_v1','playback_speed':1.,'no_acceptance_or_model_scores_in_pixels':True,
        'no_interpolation':True,'source_files_unchanged':True,'fps':15,'resolution':[32,32],
        'source_physics_dt_s':1/240,'source_pose_dt_s':1/30,
        'source_sha256':{str(p.relative_to(tmp_path)):subject.sha256(p) for p in (trace,replay,outcome)},'clips':[clip]}
    path=tmp_path/'review_manifest.json'
    def save():path.write_text(json.dumps(manifest))
    save();return path,manifest,save


def test_valid_frame_source_semantics(tmp_path):
    path,_,_=fixture(tmp_path)
    data,items=subject.validate_review_sources(path,project=tmp_path)
    assert len(items)==1 and data['clips'][0]['target']=='R1'


@pytest.mark.parametrize('field,value',[('pose_index',12345),('pose_row',15),('hall_N',99.),('front_N',99.),
                                       ('phase','RETRACT'),('subphase','fabricated')])
def test_manifest_cannot_relabel_or_reindex_saved_measurements(tmp_path,field,value):
    path,manifest,save=fixture(tmp_path)
    manifest['clips'][0]['frames'][0][field]=value
    if field=='pose_row':manifest['clips'][0]['frames'][0]['pose_time_s']=16/240
    save()
    with pytest.raises(ValueError):subject.validate_review_sources(path,project=tmp_path)


def test_wrong_target_and_shortened_point_interval_rejected(tmp_path):
    path,manifest,save=fixture(tmp_path)
    manifest['clips'][0]['target']='L5';save()
    with pytest.raises(ValueError):subject.validate_review_sources(path,project=tmp_path)
    manifest['clips'][0]['target']='R1';manifest['clips'][0]['point_end_physics_row']=31;save()
    with pytest.raises(ValueError):subject.validate_review_sources(path,project=tmp_path)


def test_nonexistent_pose_row_fails_even_with_correct_trace_time(tmp_path):
    path,manifest,save=fixture(tmp_path)
    np.savez(tmp_path/'episode/expert_replay.npz',row_indices=np.array([0,8]),poses=np.zeros((2,1,7)),paths=np.array(['/body']))
    manifest['source_sha256']['episode/expert_replay.npz']=subject.sha256(tmp_path/'episode/expert_replay.npz');save()
    with pytest.raises(ValueError):subject.validate_review_sources(path,project=tmp_path)


def test_full_decode_receipt_can_be_saved_outside_frozen_review(tmp_path):
    cv2=pytest.importorskip('cv2')
    path,manifest,save=fixture(tmp_path)
    video=tmp_path/'clip.mp4'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),15,(32,32))
    assert writer.isOpened()
    for index in range(2):writer.write(np.full((32,32,3),index*100,dtype=np.uint8))
    writer.release();manifest['clips'][0]['video_sha256']=subject.sha256(video);save()
    output=tmp_path/'new_package/old_media_receipt.json'
    hashes={p.name:subject.sha256(p) for p in (tmp_path/'episode').iterdir()}
    with patch.object(subject,'PROJECT',tmp_path),patch.object(subject,'validate_episode_provenance',return_value={'status':'passed','artifact_sha256':hashes}),patch.object(sys,'argv',['verify_media',str(path),'--output',str(output)]):
        subject.main()
    receipt=json.loads(output.read_text())
    assert receipt['semantic_binding']=={'status':'passed','frames_verified':2,'checks':['replay_index_and_row','trace_target_phase_subphase','Hall_and_front_force','point_interval_and_original_clock']}
    assert receipt['videos'][0]['decoded_full'] and receipt['videos'][0]['frame_count']==2
    assert not (tmp_path/'media_verification.json').exists()
    assert subject.sha256(video)==manifest['clips'][0]['video_sha256']
