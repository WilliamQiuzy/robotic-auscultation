"""Small independent fixtures; no Isaac, GPU, remote host, or frozen writes."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import verify_full as verifier


class FullVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.episode = self.project / verifier.EPISODE_REL
        self.episode.mkdir(parents=True)
        self.render = self.project / 'render'
        self.render.mkdir()
        self.manifest = self.render / 'review_manifest.json'
        self.n = 332
        self.patch = mock.patch.object(verifier, 'EXPECTED_ROWS', self.n)
        self.patch.start(); self.addCleanup(self.patch.stop)
        self.patch_hold = mock.patch.object(verifier, 'HOLD_STEPS', 16)
        self.patch_hold.start(); self.addCleanup(self.patch_hold.stop)
        self.patch_home = mock.patch.object(verifier, 'HOME_SETTLE_STEPS', 12)
        self.patch_home.start(); self.addCleanup(self.patch_home.stop)
        points, self.rows = [], []
        for i, target in enumerate(verifier.TARGETS):
            start = i * 32
            end = (i + 1) * 32 if i < 9 else self.n
            points.append(dict(target=target, start_physics_row=start,
                               end_physics_row=end, success=True, training_accepted=True))
            for r in range(start, end):
                phase = 'HOLD' if r < start + 16 else 'DONE'
                subphase = 'home_settle' if r >= 320 else 'contact_cycle'
                self.rows.append(dict(time_s=(r+1)/240, target=target, phase=phase,
                                      subphase=subphase, hall_N=4.0, lip_N=1.0,
                                      diaphragm_N=3.0, contact_rim_N=0.0,
                                      hold_elapsed_s=(r-start+1)/240 if phase=='HOLD' else 0.0))
        self.outcome = dict(success=True, training_accepted=True, physics_steps=self.n,
                            points=points, training_scene_sha256=hashlib.sha256(b'scene').hexdigest())
        (self.episode/'training_scene.usdc').write_bytes(b'scene')
        self.write_trace()
        self.write_json(self.episode/'outcome.json', self.outcome)
        self.indices = np.arange(7, self.n, 8, dtype=np.int64)
        poses = np.zeros((len(self.indices), 1, 7)); poses[:, :, 3] = 1
        np.savez(self.episode/'expert_replay.npz', poses=poses, row_indices=self.indices,
                 paths=np.array(['/World/Body']))
        for name in ('independent_audit.json','posture_audit.json'):
            self.write_json(self.episode/name, dict(accepted=True))
        for relative in verifier.REQUIRED_STATIC_SOURCES:
            path = self.project/relative; path.parent.mkdir(parents=True, exist_ok=True)
            if relative == verifier.CAMERA_REL:
                self.write_json(path, dict(downward_pitch_delta_deg=7,
                                          camera_height_delta_m=.05,
                                          world_from_camera_usd=np.eye(4).tolist()))
            else: path.write_text('# fixture\n')
        (self.render/'route_b_full_1x.mp4').write_bytes(b'video fixture')
        self.provenance = dict(status='passed', reasons=[], artifact_sha256={
            name: verifier.sha256(self.episode/name) for name in
            ('outcome.json','expert_trace.jsonl','expert_replay.npz',
             'independent_audit.json','posture_audit.json')})
        self.provenance_patch = mock.patch.object(verifier, 'validate_episode_provenance',
                                                  return_value=self.provenance)
        self.provenance_patch.start(); self.addCleanup(self.provenance_patch.stop)
        frames = []
        pose_cursor = 0
        for index in range((self.n+15)//16):
            end = min((index+1)*16,self.n); row_index = end-1
            while pose_cursor+1<len(self.indices) and self.indices[pose_cursor+1]<=row_index:
                pose_cursor += 1
            pose_row = int(self.indices[pose_cursor]); row = self.rows[row_index]
            frames.append(dict(frame_index=index, video_time_s=index/15,
                               simulation_interval_end_s=end/240,
                               measurement_row=row_index, measurement_time_s=row['time_s'],
                               pose_index=pose_cursor, pose_row=pose_row,
                               pose_time_s=self.rows[pose_row]['time_s'],
                               **{k:row[k] for k in ('target','phase','subphase')}))
        self.data = dict(schema='continuous_recorded_route_v1', episode=verifier.EPISODE_REL,
                         video='route_b_full_1x.mp4', video_sha256=verifier.sha256(self.render/'route_b_full_1x.mp4'),
                         source_sha256={}, source_files_unchanged=True,
                         episode_provenance=self.provenance, resolution=[1920,720], fps=15,
                         playback_speed=1.0, frame_count=len(frames), source_physics_rows=self.n,
                         source_duration_s=self.n/240, duration_s=len(frames)/15,
                         terminal_video_padding_s=len(frames)/15-self.n/240,
                         source_pose_dt_s=1/30, source_physics_dt_s=1/240,
                         first_saved_pose_s=self.rows[7]['time_s'],
                         last_saved_pose_s=self.rows[int(self.indices[-1])]['time_s'],
                         source_outcome_accepted=True, camera_config=verifier.CAMERA_REL,
                         layout=verifier.EXPECTED_LAYOUT,
                         point_intervals=[{k:p[k] for k in ('target','start_physics_row','end_physics_row')} for p in points],
                         frames=frames)
        self.rebind()

    @staticmethod
    def write_json(path, data):
        path.write_text(json.dumps(data, allow_nan=False))

    def write_trace(self):
        (self.episode/'expert_trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))

    def rebind(self):
        self.data['source_sha256'] = {relative:verifier.sha256(self.project/relative)
                                       for relative in verifier.required_sources()}
        self.write_json(self.manifest,self.data)

    def validate(self):
        self.write_json(self.manifest,self.data)
        return verifier.validate_full_sources(self.manifest, project=self.project)

    def test_complete_integer_coverage_and_tail(self):
        result = self.validate()
        self.assertEqual(result['semantic_binding']['coverage']['physics_rows'],332)
        self.assertEqual(result['semantic_binding']['coverage']['last_interval_steps'],12)
        self.assertEqual(result['semantic_binding']['coverage']['frame_count'],21)
        self.assertAlmostEqual(result['semantic_binding']['coverage']['padding_s'],1/60)
        self.assertEqual(set(result['semantic_binding']['hold_video_frame_counts']),set(verifier.TARGETS))
        self.assertEqual(result['semantic_binding']['last_frame']['subphase'],'home_settle')

    def test_drop_tail_even_if_totals_are_rewritten(self):
        self.data['frames'].pop(); self.data['frame_count']-=1
        self.data['duration_s']=self.data['frame_count']/15
        with self.assertRaisesRegex(ValueError,'frame count'):self.validate()

    def test_shift_point_boundary_rejected(self):
        self.data['point_intervals'][1]['start_physics_row']+=1
        with self.assertRaisesRegex(ValueError,'point interval'):self.validate()

    def test_every_sample_binding_checked(self):
        original=copy.deepcopy(self.data)
        for key,value in [('measurement_row',30),('pose_index',2),('pose_row',23),
                          ('measurement_time_s',99),('simulation_interval_end_s',99),
                          ('target','L5'),('phase','RETRACT'),('subphase','return_home'),
                          ('frame_index',True),('video_time_s',99)]:
            with self.subTest(key=key):
                self.data=copy.deepcopy(original);self.data['frames'][0][key]=value
                with self.assertRaises(ValueError):self.validate()

    def test_bad_padding_and_missing_source_rejected(self):
        self.data['terminal_video_padding_s']=0
        with self.assertRaisesRegex(ValueError,'padding'):self.validate()
        self.data['terminal_video_padding_s']=1/60
        self.data['source_sha256'].pop(verifier.CAMERA_REL)
        with self.assertRaisesRegex(ValueError,'source inventory'):self.validate()

    def test_changed_source_or_video_rejected(self):
        path=self.episode/'training_scene.usdc';path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'source hash'):self.validate()
        path.write_bytes(b'scene');(self.render/'route_b_full_1x.mp4').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'video hash'):self.validate()

    def test_wrong_episode_and_unsafe_source_rejected(self):
        self.data['episode']='other'
        with self.assertRaisesRegex(ValueError,'canonical episode'):self.validate()
        self.data['episode']=verifier.EPISODE_REL
        self.data['source_sha256']['../outside']='0'*64
        with self.assertRaisesRegex(ValueError,'unsafe source'):self.validate()

    def test_fresh_provenance_failure_and_stale_manifest_rejected(self):
        with mock.patch.object(verifier,'validate_episode_provenance',return_value={'status':'failed','reasons':['tamper']}):
            with self.assertRaisesRegex(ValueError,'provenance rejected'):self.validate()
        self.data['episode_provenance']=dict(status='passed',artifact_sha256={})
        with self.assertRaisesRegex(ValueError,'provenance artifact'):self.validate()

    def test_source_clock_shift_gap_and_nan_rejected(self):
        original=copy.deepcopy(self.rows)
        for mode in ('shift','gap','nan'):
            with self.subTest(mode=mode):
                self.rows=copy.deepcopy(original)
                if mode=='shift':
                    for row in self.rows:row['time_s']+=2
                elif mode=='gap':self.rows[5]['time_s']+=1/240
                else:self.rows[5]['time_s']=float('nan')
                self.write_trace();self.rebind()
                with self.assertRaises(ValueError):self.validate()

    def test_missing_hold_home_or_unsuccessful_point_rejected(self):
        self.outcome['points'][3]['success']=False
        self.write_json(self.episode/'outcome.json',self.outcome);self.rebind()
        with self.assertRaisesRegex(ValueError,'point not successful'):self.validate()
        self.outcome['points'][3]['success']=True
        self.write_json(self.episode/'outcome.json',self.outcome)
        self.rows[0]['phase']='DONE';self.write_trace();self.rebind()
        with self.assertRaisesRegex(ValueError,'HOLD'):self.validate()
        self.rows[0]['phase']='HOLD';self.rows[-1]['subphase']='return_home'
        self.write_trace();self.rebind()
        with self.assertRaisesRegex(ValueError,'home_settle'):self.validate()

    def test_changed_pose_stride_and_nonunit_pose_rejected(self):
        with np.load(self.episode/'expert_replay.npz',allow_pickle=False) as z:
            poses=z['poses'];paths=z['paths']
        indices=self.indices.copy();indices[3]-=1
        np.savez(self.episode/'expert_replay.npz',poses=poses,paths=paths,row_indices=indices)
        self.rebind()
        with self.assertRaisesRegex(ValueError,'pose row'):self.validate()
        poses[0,0,3]=2
        np.savez(self.episode/'expert_replay.npz',poses=poses,paths=paths,row_indices=self.indices)
        self.rebind()
        with self.assertRaisesRegex(ValueError,'rotation'):self.validate()

    def test_decode_must_read_all_frames_and_reject_extra_or_bad_shape(self):
        import cv2
        class Capture:
            def __init__(self,frames):self.frames=list(frames);self.reads=0;self.released=False
            def isOpened(self):return True
            def get(self,key):return {cv2.CAP_PROP_FPS:15,cv2.CAP_PROP_FRAME_WIDTH:8,
                                     cv2.CAP_PROP_FRAME_HEIGHT:6,cv2.CAP_PROP_FRAME_COUNT:3,
                                     cv2.CAP_PROP_POS_MSEC:(self.reads-1)*1000/15}[key]
            def read(self):
                self.reads+=1
                return (True,self.frames.pop(0)) if self.frames else (False,None)
            def release(self):self.released=True
        frames=[np.full((6,8,3),i,np.uint8) for i in range(3)]
        capture=Capture(frames)
        with mock.patch.object(cv2,'VideoCapture',return_value=capture):
            result=verifier.decode_full(Path('fixture.mp4'),3,resolution=(8,6))
        self.assertEqual(result['decoded_frame_count'],3);self.assertEqual(capture.reads,4)
        self.assertTrue(capture.released)
        for bad in (frames[:-1],frames+[frames[-1]],frames[:1]+[np.zeros((5,8,3),np.uint8)]+frames[2:]):
            capture=Capture(bad)
            with mock.patch.object(cv2,'VideoCapture',return_value=capture):
                with self.assertRaises(ValueError):verifier.decode_full(Path('fixture.mp4'),3,resolution=(8,6))
            self.assertTrue(capture.released)
        capture=Capture(frames)
        real_get=capture.get
        capture.get=lambda key: 0.0 if key==cv2.CAP_PROP_POS_MSEC else real_get(key)
        with mock.patch.object(cv2,'VideoCapture',return_value=capture):
            with self.assertRaisesRegex(ValueError,'presentation timestamp'):
                verifier.decode_full(Path('fixture.mp4'),3,resolution=(8,6))

    def test_verification_writes_receipt_only_after_full_decode(self):
        self.write_json(self.manifest,self.data)
        output=self.render/'receipt.json'
        with mock.patch.object(verifier,'decode_full',return_value={'decoded_full':True,'decoded_frame_count':21}):
            verifier.verify_full(self.manifest,output,project=self.project)
        report=json.loads(output.read_text())
        self.assertEqual(report['status'],'passed')
        self.assertEqual(report['manifest_sha256'],verifier.sha256(self.manifest))
        with mock.patch.object(verifier,'decode_full',side_effect=ValueError('truncated')):
            with self.assertRaisesRegex(ValueError,'truncated'):
                verifier.verify_full(self.manifest,self.render/'absent.json',project=self.project)
        self.assertFalse((self.render/'absent.json').exists())

    def test_receipt_cannot_overwrite_bound_evidence_or_episode(self):
        self.write_json(self.manifest,self.data)
        for output in (self.manifest,self.render/'route_b_full_1x.mp4',self.episode/'receipt.json',
                       self.project/'sim/expert_quality_20260924/README.md',
                       self.project/'sim/curobo_expert_v1/another.json'):
            with self.subTest(output=output),self.assertRaisesRegex(ValueError,'receipt'):
                verifier.verify_full(self.manifest,output,project=self.project)


if __name__=='__main__':unittest.main()
