"""Numerical counterexamples: failures must survive aggregation and filtering."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


class QualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.q = importlib.import_module('sim.expert_quality_20260924.quality')

    def rows(self, hz=240, duration=4.0):
        n = int(hz*duration)
        result = []
        for i in range(n):
            hold = i < n-int(hz*.2)
            result.append(dict(time_s=(i+1)/hz, q=[0.]*7, q_command=[0.]*7,
                dq=[0.]*7, phase='HOLD' if hold else 'DONE', target='R1', subphase='contact',
                hall_N=4. if hold else 0., compression_m=.002 if hold else 0.,
                gap_m=.001 if hold else .03, tangent_m=0., angle_deg=0., tcp_speed_m_s=0.,
                diaphragm_N=4. if hold else 0., lip_N=0., contact_rim_N=0.,
                carriage_other_N=0., frame_N=0., forbidden_N=0.,
                tcp_world=[0.,0.,0.], axis_world=[0.,0.,-1.]))
        return result

    def outcome(self, rows):
        return dict(success=True, training_accepted=True, requested_points=['R1'], points=[dict(
            target='R1',success=True,training_accepted=True,start_physics_row=0,
            contact_end_physics_row=len(rows),end_physics_row=len(rows),
            command_normal_world=[0.,0.,1.],phase_config=dict(hold_duration_s=3.,
            contact_normal_tolerance_m=.006,retract_clearance_m=.02,max_compression_m=.006))])

    def test_nonfinite_and_time_gap(self):
        for key, value, reason in [('q',[float('nan')]*7,'nonfinite'),('time_s',.8,'timestamp')]:
            rows=self.rows();rows[100][key]=value
            with self.assertRaisesRegex(ValueError,reason):
                self.q.validate_trace(rows)

    def test_nonfinite_optional_numeric_is_rejected(self):
        rows=self.rows();rows[12]['contact_points']=[{'position_world':[0.,float('nan'),0.]}]
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            self.q.validate_trace(rows)

    def test_cumulative_small_step_reversal_and_jitter(self):
        vals=np.radians([0,.3,.6,.9,1.2,1.5,1.4,1.6,1.3,1.,.7,.4,.1])
        events=self.q.reversal_events(vals[:,None],np.arange(len(vals),dtype=float))
        self.assertEqual(len(events),1)
        self.assertEqual(events[0]['row'],11)
        self.assertEqual(events[0]['extreme_row'],7)

    def test_wrap_jump_not_unwrapped_or_smoothed_away(self):
        rows=self.rows();rows[480]['q'][0]=2*np.pi
        a=self.q.analyze_motion(rows,self.outcome(rows))
        self.assertAlmostEqual(a['global']['measured']['travel_rad'][0],4*np.pi)
        self.assertAlmostEqual(a['global']['measured']['peak_to_peak_rad'][0],2*np.pi)
        self.assertGreater(a['global']['measured']['max_step_rad'][0],6)
        self.assertGreater(a['global']['measured']['derivatives']['raw']['velocity']['max_abs_per_joint'][0],700)

    def test_hold_is_continuous_and_wrong_site_is_hard_failure(self):
        rows=self.rows();rows[470]['hall_N']=0.
        r=self.q.analyze_contact(rows,self.outcome(rows))
        self.assertIn('R1:continuous_contact_hold_too_short',r['hard_failures'])
        rows=self.rows();rows[100]['frame_N']=.6
        r=self.q.analyze_contact(rows,self.outcome(rows))
        self.assertIn('frame_N_contact',r['hard_failures'])
        self.assertEqual(r['events'][0]['row'],100)

    def test_loaded_transition_drag_and_draft_threshold(self):
        rows=self.rows()
        for i,row in enumerate(rows):
            row['phase']='APPROACH';row['subphase']='transfer';row['tcp_world']=[i*.0001,0.,0.]
        result=self.q.analyze_contact(rows,self.outcome(rows))
        self.assertGreater(result['loaded_drag']['total_tangential_distance_m'],.08)
        self.assertEqual(result['loaded_drag']['status'],'exceeded_pending_user_choice')
        profile=self.q.load_profile();profile['user_selected']=True
        result=self.q.analyze_contact(rows,self.outcome(rows),profile)
        self.assertEqual(result['loaded_drag']['status'],'failed')

    def test_derivative_frequency_and_dt_handling(self):
        peaks=[]
        for hz in (120,240,480):
            rows=self.rows(hz=hz)
            for row in rows:
                row['q'][0]=row['q_command'][0]=.2*np.sin(2*np.pi*.5*row['time_s'])
            result=self.q.analyze_motion(rows,self.outcome(rows),expected_dt_s=1/hz)
            derivative=result['global']['measured']['derivatives']['filtered']
            peaks.append([derivative[k]['max_abs_per_joint'][0] for k in ('velocity','acceleration','jerk')])
            self.assertAlmostEqual(result['sampling']['dt_s'],1/hz)
        np.testing.assert_allclose(peaks,np.tile([.2*np.pi,.2*np.pi**2,.2*np.pi**3],(3,1)),rtol=.03)
        with self.assertRaisesRegex(ValueError,'timestamp'):
            self.q.validate_trace(self.rows(hz=120))

    def test_segment_travel_accounts_for_boundaries(self):
        rows=self.rows()
        for i,r in enumerate(rows):
            r['q'][0]=r['q_command'][0]=i*.001
            if i>=480:r['target']='L1';r['subphase']='transfer'
        result=self.q.analyze_motion(rows,self.outcome(rows))
        total=sum(s['measured']['total_travel_rad'] for s in result['subphases'])
        self.assertAlmostEqual(total,result['global']['measured']['total_travel_rad'])

    def test_bound_evidence_mutation_and_missing_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'a').write_bytes(b'initial')
            expected={'a':hashlib.sha256(b'initial').hexdigest()}
            self.assertEqual(self.q.verify_hashes(p,expected),[])
            (p/'a').write_bytes(b'changed')
            self.assertIn('artifact_sha256_mismatch:a',self.q.verify_hashes(p,expected))
            (p/'a').unlink()
            self.assertIn('missing_artifact:a',self.q.verify_hashes(p,expected))

    def candidate_fixture(self,p):
        names=('candidate_controller.py','candidate_planner.py','run_candidate.py')
        objects=p/'input_snapshot'/'objects';objects.mkdir(parents=True)
        sources={};records={}
        for name in names:
            payload=name.encode();digest=hashlib.sha256(payload).hexdigest();(objects/digest).write_bytes(payload)
            sources['source/candidate/'+name]={'sha256':digest}
            records[name]={'path':'input_snapshot/objects/'+digest,'sha256':digest}
        snapshot=p/'input_snapshot'/'manifest.json';snapshot.write_text(json.dumps({'files':sources}))
        method='sim.expert_quality_20260924.candidate_controller.CandidateController'
        manifest={'schema':'wam_expert_candidate_sources_v1','candidate_policy':'staged_continuous',
            'derived_controller':method,'files':records}
        (p/'candidate_sources_manifest.json').write_text(json.dumps(manifest))
        return {'candidate_policy':'staged_continuous','candidate_method':method,
            'candidate_sources_manifest':'candidate_sources_manifest.json',
            'candidate_sources_manifest_sha256':self.q.sha256_file(p/'candidate_sources_manifest.json'),
            'input_snapshot':'input_snapshot/manifest.json','input_snapshot_manifest_sha256':self.q.sha256_file(snapshot)}

    def test_candidate_manifest_is_independently_hash_bound(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);outcome=self.candidate_fixture(p)
            result=self.q.verify_candidate_sources(p,outcome)
            self.assertEqual(result['status'],'passed')
            record=json.loads((p/'candidate_sources_manifest.json').read_text())['files']['candidate_controller.py']
            (p/record['path']).write_bytes(b'changed')
            self.assertEqual(self.q.verify_candidate_sources(p,outcome)['status'],'failed')

    def test_candidate_cannot_drop_manifest_sources_or_change_policy_class(self):
        self.assertEqual(self.q.verify_candidate_sources('.',{'candidate_policy':'staged_continuous'})['status'],'failed')
        self.assertEqual(self.q.verify_candidate_sources('.',{'candidate_method':'anything'})['status'],'failed')
        for field,value in [('candidate_policy','checked_shortcut'),('derived_controller','OtherController'),('files',{}),('files',['candidate_controller.py','candidate_planner.py','run_candidate.py'])]:
            with tempfile.TemporaryDirectory() as d:
                p=Path(d);outcome=self.candidate_fixture(p);path=p/'candidate_sources_manifest.json'
                manifest=json.loads(path.read_text());manifest[field]=value;path.write_text(json.dumps(manifest))
                outcome['candidate_sources_manifest_sha256']=self.q.sha256_file(path)
                self.assertEqual(self.q.verify_candidate_sources(p,outcome)['status'],'failed')

    def test_candidate_sources_must_match_original_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);outcome=self.candidate_fixture(p);path=p/'candidate_sources_manifest.json'
            manifest=json.loads(path.read_text());data=b'different';digest=hashlib.sha256(data).hexdigest()
            (p/'changed.py').write_bytes(data)
            manifest['files']['candidate_controller.py']={'path':'changed.py','sha256':digest}
            path.write_text(json.dumps(manifest));outcome['candidate_sources_manifest_sha256']=self.q.sha256_file(path)
            self.assertEqual(self.q.verify_candidate_sources(p,outcome)['status'],'failed')

    def test_loaded_retraction_is_diagnostic_and_point_change_separate(self):
        rows=self.rows()
        for i,r in enumerate(rows):
            r['tcp_world']=[0.,0.,i*.00001]
            if i>=10:r['phase']='RETRACT'
        contact=self.q.analyze_contact(rows,self.outcome(rows))
        event=contact['loaded_drag']['events'][0]
        self.assertEqual(event['category'],'retraction')
        self.assertFalse(event['exceeded']);self.assertEqual(event['severity'],'diagnostic')
        self.assertGreater(event['normal_absolute_distance_m'],0.)
        self.assertEqual(event['distance_m'],0.)
        self.assertEqual(self.q.select_worst_events({'preference_events':[]},contact),[])
        for r in rows[480:]:r['target']='L1'
        outcome=self.outcome(rows);point=dict(outcome['points'][0],target='L1');outcome['points'].append(point)
        contact=self.q.analyze_contact(rows,outcome)
        self.assertIn('point_change',[e['category'] for e in contact['loaded_drag']['events']])

    def test_provenance_api_rejects_replay_replacement(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);snapshot=p/'input_snapshot';(snapshot/'objects').mkdir(parents=True)
            data=b'geometry';digest=hashlib.sha256(data).hexdigest();(snapshot/'objects'/digest).write_bytes(data)
            (snapshot/'manifest.json').write_text(json.dumps({'files':{'source/collision_regeneration':{'sha256':digest}}}))
            outcome={'input_snapshot':'input_snapshot/manifest.json',
                'input_snapshot_manifest_sha256':self.q.sha256_file(snapshot/'manifest.json'),'collision_regeneration_sha256':digest}
            (p/'outcome.json').write_text(json.dumps(outcome))
            for name in ('expert_trace.jsonl','expert_replay.npz','world_from_left_base.json','control_commands.json'):(p/name).write_bytes(b'fixture')
            common={name:self.q.sha256_file(p/name) for name in ('expert_trace.jsonl','outcome.json')}
            mechanical={'accepted':True,'artifact_sha256':dict(common,**{'control_commands.json':self.q.sha256_file(p/'control_commands.json')})}
            urdf=Path(self.q.__file__).parents[1]/'auscultation_expert_layout_v1'/'openarm_v10_follower_stetho.urdf'
            posture={'accepted':True,'artifact_sha256':dict(common,**{'expert_replay.npz':self.q.sha256_file(p/'expert_replay.npz')}),
                'source_sha256':{'world_from_left_base':{'sha256':self.q.sha256_file(p/'world_from_left_base.json')},
                    'kinematics_urdf':{'sha256':self.q.sha256_file(urdf)}}}
            (p/'independent_audit.json').write_text(json.dumps(mechanical));(p/'posture_audit.json').write_text(json.dumps(posture))
            before={f.relative_to(p):f.read_bytes() for f in p.rglob('*') if f.is_file()}
            self.assertEqual(self.q.validate_episode_provenance(p)['status'],'passed')
            self.assertEqual(before,{f.relative_to(p):f.read_bytes() for f in p.rglob('*') if f.is_file()})
            (p/'expert_replay.npz').write_bytes(b'replacement')
            result=self.q.validate_episode_provenance(p)
            self.assertEqual(result['status'],'failed')
            self.assertIn('posture_audit.json:artifact_sha256_mismatch:expert_replay.npz',result['reasons'])

    def test_posture_segment_has_max_timestamp_and_integral(self):
        times=np.arange(1,6)*.1
        p=self.q.summarize_landmarks(np.array([.8,.9,1.1,1.2,1.0]),np.array([1.,1.,1.,1.,1.]),
            1.,times,1,4,np.array([False,True,True,True,False]))
        self.assertAlmostEqual(p['max_elbow_above_shoulder_m'],.2)
        self.assertAlmostEqual(p['above_shoulder_integral_m_s'],.03)
        self.assertEqual(p['max_elbow_row'],3)
        self.assertAlmostEqual(p['minimum_contact_elbow_below_wrist_m'],-.2)

    def test_hall_and_stroke_failures_have_time_evidence(self):
        rows=self.rows();rows[130]['hall_N']=6.;rows[140]['compression_m']=.01
        result=self.q.analyze_contact(rows,self.outcome(rows))
        events={e['reason']:e for e in result['events']}
        self.assertEqual(events['hall_force_limit']['row'],130)
        self.assertEqual(events['stroke_limit']['row'],140)

    def test_missing_drag_normals_propagates_pending_decision(self):
        rows=self.rows();outcome=self.outcome(rows);del outcome['points'][0]['command_normal_world']
        profile=self.q.load_profile();profile['user_selected']=True
        contact=self.q.analyze_contact(rows,outcome,profile)
        self.assertEqual(contact['loaded_drag']['status'],'pending')
        self.assertEqual(self.q.simulation_quality_decision([], 'passed',contact['loaded_drag']['status'],profile),'pending_evaluation')
        self.assertEqual(self.q.simulation_quality_decision([],None,'passed',profile),'pending_evaluation')
        self.assertEqual(self.q.simulation_quality_decision(['hard_failure'],'passed','pending',profile),'reject_hard_failure')

    def test_audit_never_writes_input_directory(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            with self.assertRaisesRegex(ValueError,'input directory'):
                self.q.audit_episode(p,p/'quality.json')

if __name__=='__main__':unittest.main()
