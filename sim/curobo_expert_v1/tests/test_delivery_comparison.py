import copy
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from sim.curobo_expert_v1 import build_comparison as subject


NAMES=[s+str(i) for i in range(1,6) for s in ('R','L')]


def fixture(tmp_path):
    episode=tmp_path/'episode';episode.mkdir()
    outcome={'success':True,'requested_points':NAMES,'points':[{'target':n,'success':True} for n in NAMES]}
    (episode/'outcome.json').write_text(json.dumps(outcome))
    (episode/'expert_trace.jsonl').write_text('synthetic bound trace')
    hashes={p.name:subject.sha256_file(p) for p in episode.iterdir()}
    measured={'total_travel_rad':10.,'travel_rad':[1.]*7,'reversal_count':2,
        'posture':{'max_elbow_above_shoulder_m':.04},
        'derivatives':{'raw':{'acceleration':{'max_abs_per_joint':[1.]*7},'jerk':{'max_abs_per_joint':[2.]*7}},
                       'filtered':{'jerk':{'max_abs_per_joint':[1.]*7}}}}
    quality={'artifact_sha256':hashes,'motion':{'global':{'measured':measured},'sampling':{'duration_s':100.}},
        'contact':{'legacy':{'accepted':True,'points':[{'target':n,'accepted':True,'max_physical_force_N':4.,'continuous_front_contact_hold_s':3.} for n in NAMES]}},
        'hard_failures':[],'simulation_quality_decision':'pending_evaluation'}
    free={'provenance':{'artifact_sha256':hashes},'status':'passed','failures':[],
          'actual_windows':{'jerk_rad_s3':{'peak':{'value':1.}}}}
    report=tmp_path/'report';report.mkdir()
    provenance={'status':'passed','artifact_sha256':hashes,'recorded_acceptance':{'independent_audit.json':True,'posture_audit.json':True}}
    def save():
        (report/'quality_report.json').write_text(json.dumps(quality));(report/'free_motion_report.json').write_text(json.dumps(free))
    save()
    return episode,outcome,quality,free,provenance,save


def collect(tmp_path,provenance,**extra):
    with patch.object(subject,'ROOT',tmp_path),patch.object(subject,'validate_episode_provenance',return_value=provenance):
        return subject.collect(dict(report_dir='report',episode_dir='episode',label='candidate',description='synthetic',**extra))


def test_complete_requires_bound_outcome_success_and_every_point_audit(tmp_path):
    episode,outcome,quality,free,provenance,save=fixture(tmp_path)
    assert collect(tmp_path,provenance)['complete']
    quality['contact']['legacy']['points'][-1]['accepted']=False
    save()
    assert not collect(tmp_path,provenance)['complete']
    quality['contact']['legacy']['points'][-1]['accepted']=True
    outcome['success']=False;outcome['points'][-1]['success']=False
    (episode/'outcome.json').write_text(json.dumps(outcome))
    provenance['artifact_sha256']['outcome.json']=subject.sha256_file(episode/'outcome.json')
    save()
    assert not collect(tmp_path,provenance)['complete']


def test_actual_episode_mutation_rejected_even_when_reports_agree(tmp_path):
    episode,_,_,_,provenance,_=fixture(tmp_path)
    (episode/'expert_trace.jsonl').write_text('modified after both reports')
    with pytest.raises(ValueError,match='artifact|evidence|hash'):
        collect(tmp_path,provenance)


def test_provenance_failure_or_missing_report_anchor_rejected(tmp_path):
    _,_,quality,_,provenance,save=fixture(tmp_path)
    provenance['status']='failed';provenance['reasons']=['anchor mismatch']
    with pytest.raises(ValueError,match='provenance'):
        collect(tmp_path,provenance)
    provenance['status']='passed'
    quality['artifact_sha256']=dict(quality['artifact_sha256'])
    del quality['artifact_sha256']['expert_trace.jsonl'];save()
    with pytest.raises(ValueError,match='artifact|anchor|evidence'):
        collect(tmp_path,provenance)


def test_explicit_episode_required(tmp_path):
    fixture(tmp_path)
    with patch.object(subject,'ROOT',tmp_path),pytest.raises(ValueError,match='episode_dir'):
        subject.collect({'report_dir':'report','label':'x','description':'x'})


def test_semantic_receipt_required_and_external_receipt_supported(tmp_path):
    episode,_,quality,_,provenance,save=fixture(tmp_path)
    replay=episode/'expert_replay.npz';replay.write_bytes(b'bound replay; semantic validation mocked independently')
    provenance['artifact_sha256']['expert_replay.npz']=subject.sha256_file(replay);save()
    review=tmp_path/'review';review.mkdir();manifest=review/'review_manifest.json';manifest.write_text('{}')
    video=review/'clip.mp4';video.write_bytes(b'bound movie')
    item={'clip':{'clip_id':'R1_posture','frame_count':2,'video_sha256':subject.sha256_file(video),'duration_s':2/15},'video_path':str(video)}
    data={'source_sha256':{str((episode/name).relative_to(tmp_path)):digest for name,digest in provenance['artifact_sha256'].items()}}
    receipt={'source_bindings_passed':True,'manifest_sha256':subject.sha256_file(manifest),
             'videos':[{'clip':'R1_posture','decoded_full':True,'sha256':subject.sha256_file(video),'frame_count':2}]}
    destination=tmp_path/'external_receipt.json';destination.write_text(json.dumps(receipt))
    with patch.object(subject,'validate_review_sources',return_value=(data,[item])):
        with pytest.raises(ValueError,match='media verification'):
            collect(tmp_path,provenance,review_dir='review',media_verification='external_receipt.json')
        receipt.update(semantic_binding={'status':'passed','frames_verified':2},
                       verifier_sha256=subject.sha256_file(Path(subject.__file__).with_name('verify_media.py')))
        destination.write_text(json.dumps(receipt))
        result=collect(tmp_path,provenance,review_dir='review',media_verification='external_receipt.json')
        assert result['media_verification']=='external_receipt.json'


def partial_fixture(tmp_path):
    episode,outcome,quality,free,provenance,save=fixture(tmp_path)
    outcome['points']=outcome['points'][:9]
    outcome['success']=False;outcome['points'][-1]['success']=False
    (episode/'outcome.json').write_text(json.dumps(outcome))
    provenance['artifact_sha256']['outcome.json']=subject.sha256_file(episode/'outcome.json')
    provenance['recorded_acceptance']={name:False for name in provenance['recorded_acceptance']}
    quality['contact']['legacy']['points']=quality['contact']['legacy']['points'][:8]
    for point in quality['contact']['legacy']['points']:point['accepted']=False
    quality['contact']['legacy']['accepted']=False
    quality['hard_failures']=['controller_rejected','invalid_point_interval']
    quality['simulation_quality_decision']='reject_hard_failure';save()
    return quality,provenance,save


def test_eight_controller_completions_are_not_erased_by_whole_episode_rejection(tmp_path):
    _,provenance,_=partial_fixture(tmp_path)
    result=collect(tmp_path,provenance)
    assert result['completed_points']==8
    assert result['audit_accepted_points']==0
    assert result['controller_attempted_points']==9
    assert result['recorded_points']==8
    assert result['complete'] is False


def test_completed_controller_targets_must_have_corresponding_contact_inventory(tmp_path):
    quality,provenance,save=partial_fixture(tmp_path)
    quality['contact']['legacy']['points'].pop(2);save()
    with pytest.raises(ValueError,match='inventory'):
        collect(tmp_path,provenance)


def test_page_labels_controller_counts_and_links_bounded_r5_diagnostic(tmp_path):
    _,provenance,_=partial_fixture(tmp_path)
    config={'variants':[{'report_dir':'report','episode_dir':'episode','label':'partial cuRobo','description':'partial'}],'notes':[]}
    path=tmp_path/'config.json';path.write_text(json.dumps(config));(tmp_path/'results').mkdir()
    with patch.object(subject,'ROOT',tmp_path),patch.object(subject,'validate_episode_provenance',return_value=provenance),patch.object(sys,'argv',['build_comparison','--config',str(path)]):
        subject.main()
    page=(tmp_path/'comparison.html').read_text()
    assert '完成点数（控制器记录）' in page and '<td>8/10</td>' in page
    assert 'results/full10_independent_review/r5_release_diagnostics.png' in page
    assert 'results/full10_independent_review/r5_release_diagnostics.pdf' in page
    assert '局部图不能代表全程或确定因果' in page
