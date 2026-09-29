"""Adversarial media binding tests; no Isaac or rendering jobs."""
import copy
import importlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest


def module():
    try:
        return importlib.import_module('sim.auscultation_skill_v1.verify_media')
    except ModuleNotFoundError:
        pytest.fail('independent skill media verifier missing')


def example():
    names = [s+str(i) for i in range(1, 6) for s in ('R', 'L')]
    rows, points = [], []
    for index, name in enumerate(names):
        start = len(rows)
        states = ([('APPROACH', 'initial_home')]*16+[('APPROACH', 'settle')]*16 if index == 0 else [])
        states += [('APPROACH', 'align')]*16+[('HOLD', 'contact_cycle')]*752
        states += [('RETRACT', 'normal_release_confirm')]*16
        if index == 9: states += [('DONE', 'home_settle')]*125
        for phase, subphase in states:
            rows.append(dict(time_s=(len(rows)+1)/240, target=name, phase=phase, subphase=subphase))
        points.append(dict(target=name, start_physics_row=start, end_physics_row=len(rows), success=True))
    outcome = dict(success=True, training_accepted=True, requested_points=names, points=points,
                   physics_steps=len(rows), phase_config=dict(hold_duration_s=3.))
    indices = np.arange(7, len(rows), 8)
    poses = np.zeros((len(indices), 1, 7)); poses[:, :, 3] = 1.
    paths = np.array(['/World/Body'])
    count = (len(rows)+15)//16
    frames=[]
    for index in range(count):
        end=min((index+1)*16,len(rows)); row=end-1; pose=end//8-1
        frames.append(dict(frame_index=index, video_time_s=index/15, simulation_interval_end_s=end/240,
            measurement_row=row, measurement_time_s=rows[row]['time_s'], pose_index=pose,
            pose_row=int(indices[pose]), pose_time_s=rows[indices[pose]]['time_s'],
            **{k:rows[row][k] for k in ('target','phase','subphase')}))
    manifest=dict(fps=15, playback_speed=1., resolution=[1920,720], frame_count=count,
        source_physics_rows=len(rows), source_duration_s=len(rows)/240, duration_s=count/15,
        terminal_video_padding_s=count/15-len(rows)/240, source_physics_dt_s=1/240,
        source_pose_dt_s=1/30, first_saved_pose_s=rows[indices[0]]['time_s'],
        last_saved_pose_s=rows[indices[-1]]['time_s'], frames=frames,
        point_intervals=[{k:p[k] for k in ('target','start_physics_row','end_physics_row')} for p in points])
    return manifest, outcome, rows, indices, poses, paths


def test_full_timeline_accepts_extra_hold_rows_and_fractional_final_frame():
    m=module(); data,outcome,rows,indices,poses,paths=example()
    report=m.validate_schedule(data,outcome,rows,indices,poses,paths)
    assert report['frame_count']==len(data['frames'])
    assert report['final_home_included'] is True
    assert data['terminal_video_padding_s']>0
    assert all(sum(r['phase']=='HOLD' for r in rows if r['target']==t)>720 for t in outcome['requested_points'])


@pytest.mark.parametrize('mutation', ['clock','skip','pose','timewarp','order','home','fps'])
def test_schedule_rejects_clock_mapping_or_coverage_tampering(mutation):
    m=module(); data,outcome,rows,indices,poses,paths=example()
    if mutation=='clock': rows[700]['time_s']+=1/240
    if mutation=='skip': data['frames'].pop(4)
    if mutation=='pose': data['frames'][4]['pose_row']+=8
    if mutation=='timewarp': data['frames'][4]['video_time_s']*=2
    if mutation=='order': outcome['requested_points'][0:2]=['L1','R1']
    if mutation=='home': rows[-1]['subphase']='normal_release_confirm'
    if mutation=='fps': data['fps']=30
    with pytest.raises(ValueError):m.validate_schedule(data,outcome,rows,indices,poses,paths)


def test_source_hashes_and_paths_fail_closed(tmp_path):
    m=module(); path=tmp_path/'source';path.write_bytes(b'bound')
    hashes={'source':m.sha(path)}
    assert m.verify_sources(hashes,tmp_path)=={'source':path}
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash'):m.verify_sources(hashes,tmp_path)
    with pytest.raises(ValueError,match='unsafe'):m.verify_sources({'../outside':'0'*64},tmp_path)
    with pytest.raises(ValueError,match='duplicate'):m.read_json_text('{"a":1,"a":2}')
    with pytest.raises(ValueError,match='nonfinite'):m.read_json_text('{"a":NaN}')


def test_rgbd_required_32_actual_pose_bindings_and_zero_time_skew():
    m=module();_,outcome,rows,indices,poses,paths=example()
    expected=m.expected_rgbd_samples(rows,indices,outcome['requested_points'])
    assert len(expected)==32
    samples=[dict(sample,capture_id=i,rgb_source_time_s=sample['simulation_time_s'],
                  depth_source_time_s=sample['simulation_time_s']) for i,sample in enumerate(expected)]
    m.validate_rgbd_bindings(samples,expected)
    broken=copy.deepcopy(samples);broken[3]['physics_row']+=8
    with pytest.raises(ValueError):m.validate_rgbd_bindings(broken,expected)
    broken=copy.deepcopy(samples);broken[3]['depth_source_time_s']+=1/240
    with pytest.raises(ValueError):m.validate_rgbd_bindings(broken,expected)
    with pytest.raises(ValueError):m.validate_rgbd_bindings(samples[:-1],expected)


def test_rgbd_file_triplet_checks_depth_mask_and_hashes(tmp_path):
    m=module();k=dict(width=6,height=4,fx=5.,fy=5.,cx=3.,cy=2.)
    image=np.zeros((4,6,3),np.uint8);image[:,3:]=200
    depth=np.ones((4,6),np.float32);depth[0,0]=np.inf;depth[0,1]=np.nan
    mask=np.isfinite(depth)&(depth>0)&(depth<100.)
    files={'rgb':'rgb.png','depth_m':'depth.npy','valid_mask':'mask.npy'}
    Image.fromarray(image).save(tmp_path/files['rgb']);np.save(tmp_path/files['depth_m'],depth);np.save(tmp_path/files['valid_mask'],mask)
    sample=dict(files=files,file_sha256={k:m.sha(tmp_path/v) for k,v in files.items()},valid_depth_fraction=float(mask.mean()))
    actual=m.validate_rgbd_triplet(tmp_path,sample,k)
    np.testing.assert_array_equal(np.isnan(actual),np.isnan(depth))
    mask[0,0]=True;np.save(tmp_path/files['valid_mask'],mask)
    with pytest.raises(ValueError,match='hash'):m.validate_rgbd_triplet(tmp_path,sample,k)
    sample['file_sha256']['valid_mask']=m.sha(tmp_path/files['valid_mask'])
    with pytest.raises(ValueError,match='mask'):m.validate_rgbd_triplet(tmp_path,sample,k)


def test_projection_recomputed_from_metric_depth_rejects_false_visibility():
    m=module();k=dict(width=5,height=5,fx=100.,fy=100.,cx=2.,cy=2.)
    depth=np.ones((5,5),np.float32);depth[2,2]=.7
    measured=float(depth[2,2])
    row=dict(name='R1',target_world_m=[0.,0.,1.],optical_depth_m=1.,inside=True,
        uv_px=[2.,2.],sample_pixel_xy=[2,2],valid_depth=True,measured_depth_m=measured,
        depth_residual_m=measured-1.,occluded_by_10mm=True,depth_matches_within_5mm=False,
        backprojected_world_m=[0.,0.,measured],measured_backprojection_residual_m=1.-measured,
        mathematical_roundtrip_residual_m=0.)
    assert m.validate_projections([row],{'R1':[0.,0.,1.]},depth,k,np.eye(4)) is False
    row['occluded_by_10mm']=False
    with pytest.raises(ValueError,match='occluded'):m.validate_projections([row],{'R1':[0.,0.,1.]},depth,k,np.eye(4))


def render_fixture(tmp_path,monkeypatch):
    m=module();data,outcome,rows,indices,poses,paths=example()
    episode=tmp_path/'episode';media=tmp_path/'media';episode.mkdir();media.mkdir()
    (episode/'training_scene.usdc').write_bytes(b'synthetic test source, never rendered')
    outcome.update(training_scene_sha256=m.sha(episode/'training_scene.usdc'),
                   executed_controller_class='sim.auscultation_skill_v1.student.StudentController')
    (episode/'outcome.json').write_text(json.dumps(outcome))
    (episode/'expert_trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    np.savez(episode/'expert_replay.npz',row_indices=indices,poses=poses,paths=paths)
    provenance=dict(status='passed',artifact_sha256={p.name:m.sha(p) for p in episode.iterdir()})
    monkeypatch.setattr(m,'validate_episode_provenance',lambda path:provenance)
    required=['sim/upper_pair_posture_20260924/render_episode.py','sim/route_b_full_video_20260924/timeline.py',
              'sim/auscultation_skill_v1/private_renderer.py','sim/auscultation_skill_v1/media_teardown.py']
    implementations=['sim/auscultation_skill_v1/'+name for name in ('render.py','quality.py','provenance.py','contract.py')]
    for name in required+implementations:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('test bound source')
    camera=tmp_path/m.CAMERA;camera.parent.mkdir(parents=True,exist_ok=True)
    camera.write_text(json.dumps(dict(downward_pitch_delta_deg=7.,camera_height_delta_m=.05)))
    video=media/'test.mp4';video.write_bytes(b'test codec adapter input')
    data.update(schema=m.RENDER_SCHEMA,episode='episode',source_sha256={str(p.relative_to(tmp_path)):m.sha(p)
        for p in list(episode.iterdir())+[camera]+[tmp_path/p for p in required]},
        renderer_implementation_sha256={p:m.sha(tmp_path/p) for p in implementations},
        inherited_renderer=required[0],executed_renderer=required[2],
        source_binding_validator='sim.auscultation_skill_v1.quality.validate_episode_provenance',
        executed_controller_class=outcome['executed_controller_class'],layout=m.LAYOUT,camera_config=m.CAMERA,
        source_files_unchanged=True,episode_provenance=provenance,video=video.name,video_sha256=m.sha(video),source_outcome_accepted=True)
    lifecycle,guard=guard_fixture(tmp_path,'render')
    data.update(lifecycle)
    manifest=media/'review_manifest.json';manifest.write_text(json.dumps(data))
    def decoder(path,count,resolution,fps):
        assert path==video and count==data['frame_count'] and resolution==(1920,720) and fps==15
        return dict(decoded_through_eof=True,pts_verified=True,decoded_frames=count)
    monkeypatch.setattr(m,'decode_video',decoder)
    return m,media,episode,manifest,video


def test_render_receipt_binds_new_schema_sources_manifest_and_video(tmp_path,monkeypatch):
    m,media,episode,manifest,video=render_fixture(tmp_path,monkeypatch)
    report=m.verify_render(media,project=tmp_path,guard_path=tmp_path/'render_guard.json')
    assert report['status']=='passed' and report['manifest_sha256']==m.sha(manifest)
    assert report['video_sha256']==m.sha(video) and report['decode']['decoded_through_eof'] is True
    assert report['process_completion']['guard_sha256']==m.sha(tmp_path/'render_guard.json')


@pytest.mark.parametrize('mutation',['video','source','implementation','old_schema','during_decode'])
def test_render_refuses_unbound_or_changed_artifacts(tmp_path,monkeypatch,mutation):
    m,media,episode,manifest,video=render_fixture(tmp_path,monkeypatch)
    if mutation=='video':video.write_bytes(b'replaced video')
    if mutation=='source':(episode/'expert_trace.jsonl').write_text('changed')
    if mutation=='implementation':(tmp_path/'sim/auscultation_skill_v1/render.py').write_text('changed')
    if mutation=='old_schema':
        data=json.loads(manifest.read_text());data['schema']='continuous_recorded_route_v1';manifest.write_text(json.dumps(data))
    if mutation=='during_decode':
        def decoder(*args,**kwargs):
            manifest.write_text(manifest.read_text()+' ')
            return dict(decoded_through_eof=True,pts_verified=True)
        monkeypatch.setattr(m,'decode_video',decoder)
    with pytest.raises(ValueError):m.verify_render(media,project=tmp_path,guard_path=tmp_path/'render_guard.json')


def guard_fixture(project,kind):
    launcher=project.parent/'run_isaac_linbox.sh'
    launcher.write_text('#!/bin/bash\n# synthetic owned launcher; never executed\n')
    lifecycle=dict(apps_created=1,close_returned=True,
        completion_stage='post_app_close_pre_process_exit',exit_policy='controlled_os_exit_after_close')
    data={'episode':'episode', 'render_lifecycle' if kind=='render' else 'capture_lifecycle':lifecycle,
          'resource_teardown':dict(policy='capture_scope_end_then_stage_close_before_normal_close_v1',
              replicator_wait_completed=True,annotators_detached=2,
              render_products_destroyed=2 if kind=='render' else 1,close_returned=True,
              capture_scope_ended=True,gc_before_stage_close=True,pre_stage_close_updates=5,
              explicit_stage_close_checked=True,stage_absent_after_drain=True,
              post_stage_close_updates=5,gc_before_app_close=True,errors=[])}
    guard=dict(command=['bash',str(launcher),'-m','sim.auscultation_skill_v1.'+kind,
                        '--episode','episode','--output','media'],
               returncode=0,guard_returncode=0,timed_out=False,owned_process_group=333,
               minimum_required_gib=8.,minimum_observed_gib=16.)
    path=project/(kind+'_guard.json');path.write_text(json.dumps(guard))
    return data,path


@pytest.mark.parametrize('kind',['render','rgbd'])
def test_guard_binds_actual_exit_owned_command_and_postclose(tmp_path,kind):
    m=module();data,path=guard_fixture(tmp_path,kind)
    record=m.verify_process_completion(tmp_path/'media',data,guard_path=path,
        project=tmp_path,module='sim.auscultation_skill_v1.'+kind)
    assert record['guard_path']==str(path.resolve()) and record['guard_sha256']==m.sha(path)
    assert record['launcher_path']==str((tmp_path.parent/'run_isaac_linbox.sh').resolve())
    assert record['module']=='sim.auscultation_skill_v1.'+kind
    assert record['episode']==str((tmp_path/'episode').resolve())
    assert record['output']==str((tmp_path/'media').resolve())


@pytest.mark.parametrize('kind',['render','rgbd'])
@pytest.mark.parametrize('mutation',['missing_lifecycle','false_close','wrong_stage','wrong_teardown',
    'missing_guard','signal_exit','guard_failure','timeout','low_memory','interrupted',
    'wrong_episode','wrong_output','wrong_module','wrong_launcher','duplicate_episode','unknown_flag',
    'boolean_exit','numeric_close','numeric_teardown_wait','missing_process_group','inside_media'])
def test_guard_rejects_incomplete_failed_or_other_invocations(tmp_path,kind,mutation):
    m=module();data,path=guard_fixture(tmp_path,kind);guard=json.loads(path.read_text())
    lifecycle='render_lifecycle' if kind=='render' else 'capture_lifecycle'
    if mutation=='missing_lifecycle':del data[lifecycle]
    if mutation=='false_close':data[lifecycle]['close_returned']=False
    if mutation=='wrong_stage':data[lifecycle]['completion_stage']='before_close'
    if mutation=='wrong_teardown':data['resource_teardown']['render_products_destroyed']=0
    if mutation=='signal_exit':guard['returncode']=-11
    if mutation=='guard_failure':guard['guard_returncode']=75
    if mutation=='timeout':guard['timed_out']=True
    if mutation=='low_memory':guard['aborted_low_memory']=True
    if mutation=='interrupted':guard['interrupted_signal']=15
    if mutation=='wrong_episode':guard['command'][5]='another_episode'
    if mutation=='wrong_output':guard['command'][7]='another_output'
    if mutation=='wrong_module':guard['command'][3]='sim.auscultation_skill_v1.run'
    if mutation=='wrong_launcher':guard['command'][1]='/tmp/unowned_launcher.sh'
    if mutation=='duplicate_episode':guard['command']+=['--episode','episode']
    if mutation=='unknown_flag':guard['command']+=['--unrecognized','value']
    if mutation=='boolean_exit':guard['returncode']=False
    if mutation=='numeric_close':data[lifecycle]['close_returned']=1
    if mutation=='numeric_teardown_wait':data['resource_teardown']['replicator_wait_completed']=1
    if mutation=='missing_process_group':del guard['owned_process_group']
    if mutation=='inside_media':
        (tmp_path/'media').mkdir();path=tmp_path/'media/guard.json'
    path.write_text(json.dumps(guard))
    if mutation=='missing_guard':path=None
    with pytest.raises(ValueError):
        m.verify_process_completion(tmp_path/'media',data,guard_path=path,
            project=tmp_path,module='sim.auscultation_skill_v1.'+kind)


@pytest.mark.parametrize('mutation',['guard','launcher'])
def test_render_rejects_process_evidence_changed_during_decode(tmp_path,monkeypatch,mutation):
    m,media,episode,manifest,video=render_fixture(tmp_path,monkeypatch)
    target=tmp_path/'render_guard.json' if mutation=='guard' else tmp_path.parent/'run_isaac_linbox.sh'
    def decoder(*args,**kwargs):
        target.write_text(target.read_text()+' ')
        return dict(decoded_through_eof=True,pts_verified=True)
    monkeypatch.setattr(m,'decode_video',decoder)
    with pytest.raises(ValueError,match='process|guard|launcher'):
        m.verify_render(media,project=tmp_path,guard_path=tmp_path/'render_guard.json')


@pytest.mark.parametrize('value',[np.inf,np.nan,0.])
def test_rgbd_rejects_all_invalid_depth_without_changing_occlusion_policy(tmp_path,value):
    m=module();pixels=np.zeros((4,6,3),np.uint8);pixels[:,3:]=200
    Image.fromarray(pixels).save(tmp_path/'rgb.png')
    depth=np.full((4,6),value,np.float32);mask=np.zeros((4,6),np.bool_)
    np.save(tmp_path/'depth.npy',depth);np.save(tmp_path/'mask.npy',mask)
    files=dict(rgb='rgb.png',depth_m='depth.npy',valid_mask='mask.npy')
    sample=dict(files=files,file_sha256={k:m.sha(tmp_path/v) for k,v in files.items()},valid_depth_fraction=0.)
    with pytest.raises(ValueError,match='valid depth'):
        m.validate_rgbd_triplet(tmp_path,sample,dict(width=6,height=4))
    depth[1,1]=.7;mask[1,1]=True
    np.save(tmp_path/'depth.npy',depth);np.save(tmp_path/'mask.npy',mask)
    sample.update(file_sha256={k:m.sha(tmp_path/v) for k,v in files.items()},valid_depth_fraction=1/24)
    assert m.validate_rgbd_triplet(tmp_path,sample,dict(width=6,height=4))[1,1]==np.float32(.7)


@pytest.mark.parametrize('arguments',[['--render','media'],['--rgbd','media'],
    ['--render-guard','guard.json'],['--rgbd-guard','guard.json']])
def test_cli_requires_matching_media_and_guard_before_reading_artifacts(tmp_path,arguments):
    with pytest.raises(SystemExit) as error:
        module().main(arguments+['--output',str(tmp_path/'receipt.json')])
    assert error.value.code==2 and not (tmp_path/'receipt.json').exists()


@pytest.mark.parametrize('kind',['render','rgbd'])
def test_guard_accepts_absolute_canonical_paths_and_only_render_label(tmp_path,kind):
    m=module();data,path=guard_fixture(tmp_path,kind);guard=json.loads(path.read_text())
    guard['command'][5]=str(tmp_path/'episode');guard['command'][7]=str(tmp_path/'media')
    if kind=='render':guard['command']+=['--label','SYNTHETIC REVIEW']
    path.write_text(json.dumps(guard))
    assert m.verify_process_completion(tmp_path/'media',data,guard_path=path,project=tmp_path,
        module='sim.auscultation_skill_v1.'+kind)['status']=='passed'
    if kind=='rgbd':
        guard['command']+=['--label','not supported'];path.write_text(json.dumps(guard))
        with pytest.raises(ValueError,match='argument'):
            m.verify_process_completion(tmp_path/'media',data,guard_path=path,project=tmp_path,
                module='sim.auscultation_skill_v1.'+kind)


@pytest.mark.parametrize('field', ['replicator_wait_completed','capture_scope_ended',
    'gc_before_stage_close','explicit_stage_close_checked','stage_absent_after_drain',
    'gc_before_app_close','close_returned'])
@pytest.mark.parametrize('value', [False,1,None])
def test_scoped_teardown_requires_every_actual_boolean(tmp_path,field,value):
    m=module();data,path=guard_fixture(tmp_path,'rgbd')
    data['resource_teardown'][field]=value
    with pytest.raises(ValueError,match='teardown'):
        m.verify_process_completion(tmp_path/'media',data,guard_path=path,
            project=tmp_path,module='sim.auscultation_skill_v1.rgbd')


@pytest.mark.parametrize('field,value', [('pre_stage_close_updates',4),('post_stage_close_updates',6),
    ('pre_stage_close_updates',5.),('errors',[{'stage':'capture','traceback':'error'}]),
    ('policy','owned_annotators_then_products_before_normal_close_v1')])
def test_scoped_teardown_rejects_incomplete_or_old_cleanup(tmp_path,field,value):
    m=module();data,path=guard_fixture(tmp_path,'render')
    data['resource_teardown'][field]=value
    with pytest.raises(ValueError,match='teardown'):
        m.verify_process_completion(tmp_path/'media',data,guard_path=path,
            project=tmp_path,module='sim.auscultation_skill_v1.render')


@pytest.mark.parametrize('source', ['private_renderer.py','media_teardown.py'])
@pytest.mark.parametrize('mutation', ['changed','removed_from_manifest'])
def test_scoped_native_implementation_is_explicitly_hash_bound(tmp_path,monkeypatch,source,mutation):
    m,media,episode,manifest,video=render_fixture(tmp_path,monkeypatch)
    relative='sim/auscultation_skill_v1/'+source
    if mutation=='changed':
        (tmp_path/relative).write_text('modified native cleanup code')
    else:
        data=json.loads(manifest.read_text());del data['source_sha256'][relative]
        manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        m.verify_render(media,project=tmp_path,guard_path=tmp_path/'render_guard.json')
