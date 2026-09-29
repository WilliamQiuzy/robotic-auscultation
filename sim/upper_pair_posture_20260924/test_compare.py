"""Synthetic metric contracts; no rendering, FCL, remote host or source edits."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import compare


def episode_fixture(targets=compare.TARGETS, extra_home=0, dt=1/240):
    rows=[];points=[]
    for k,target in enumerate(targets):
        start=len(rows)
        count=40+(extra_home if k==len(targets)-1 else 0)
        for i in range(count):
            t=(len(rows)+1)*dt
            phase=('APPROACH' if i<8 else 'CONTACT' if i<16 else 'HOLD'
                   if i<24 else 'RETRACT' if i<32 else 'DONE')
            sub=('transfer' if phase=='APPROACH' else 'contact_cycle' if phase in ('CONTACT','HOLD')
                 else 'normal_unload' if phase=='RETRACT' else 'withdraw_settle')
            if k==len(targets)-1 and i>=32:sub='home_settle'
            force=4.0 if phase in ('CONTACT','HOLD') else 0.0
            q=[.01*(j+1)*t for j in range(7)]
            rows.append(dict(time_s=t,target=target,phase=phase,subphase=sub,
                q=q,q_command=[v-.001*(j+1) for j,v in enumerate(q)],dq=[.01*(j+1) for j in range(7)],
                tcp_world=[t*.001,0.,0.],axis_world=[0.,0.,1.],hall_N=force,
                diaphragm_N=0.,lip_N=force,contact_rim_N=0.,carriage_other_N=0.,frame_N=0.,forbidden_N=0.,
                compression_m=.001 if force else 0.,gap_m=0.,tangent_m=0.,angle_deg=0.,tcp_speed_m_s=.001,
                hold_elapsed_s=(i-15)*dt if phase=='HOLD' else 0.,physical_violations=[],posture_violations=[]))
        points.append(dict(target=target,success=True,training_accepted=True,start_physics_row=start,
                           end_physics_row=len(rows),contact_end_physics_row=start+32,
                           command_normal_world=[0.,0.,1.],phase_config=dict(target_force_N=4.,force_tolerance_N=.5,
                           hold_duration_s=8*dt,release_force_limit_N=.2),observed_hold_s=8*dt))
    outcome=dict(requested_points=list(targets),points=points,success=True,training_accepted=True,
                 physics_steps=len(rows),physics_dt_s=dt,posture_limits=dict(joint_lower_rad=[-3.]*7,
                 joint_upper_rad=[3.]*7,observed_joint_tolerance_rad=.005,max_elbow_z_m=2.,
                 min_contact_elbow_below_wrist_m=0.),phase_config=points[0]['phase_config'],inter_point_routes=[])
    return rows,outcome


class MetricTests(unittest.TestCase):
    def prepare(self,targets=compare.TARGETS,**kwargs):
        rows,outcome=episode_fixture(targets,**kwargs)
        return compare.prepare_episode(rows,outcome,np.eye(4)),rows,outcome

    def test_primary_screen_stops_at_r2_hold_not_its_return_home(self):
        baseline,_,_=self.prepare()
        rows,outcome=episode_fixture(compare.TARGETS[:3],extra_home=32)
        for row in rows[112:]:row['q'][6]+=2
        alternative=compare.prepare_episode(rows,outcome,np.eye(4))
        report=compare.compare_prepared(baseline,alternative)
        primary=report['primary_comparison']
        self.assertEqual(primary['window_kind'],'requested_prefix_through_final_HOLD')
        self.assertEqual(primary['baseline']['window']['end_row_exclusive'],104)
        self.assertEqual(primary['alternative']['window']['end_row_exclusive'],104)
        self.assertAlmostEqual(primary['delta']['actual_total_travel_rad'],0)
        self.assertIsNone(report['full10_comparison'])
        self.assertEqual(report['terminal_diagnostics']['alternative']['window']['end_row_exclusive'],152)

    def test_raw_derivatives_are_computed_before_window_slice(self):
        rows,outcome=episode_fixture(compare.TARGETS[:3])
        rows[104]['q'][0]+=.1
        prepared=compare.prepare_episode(rows,outcome,np.eye(4))
        result=compare.summarize(prepared,100,104)
        values=np.asarray([row['q'] for row in rows])
        velocity=np.gradient(values,1/240,axis=0,edge_order=2)
        acceleration=np.gradient(velocity,1/240,axis=0,edge_order=2)
        jerk=np.gradient(acceleration,1/240,axis=0,edge_order=2)
        self.assertAlmostEqual(result['actual']['raw']['jerk']['max_abs_per_joint'][0],float(np.max(np.abs(jerk[100:104,0]))))
        self.assertGreater(result['actual']['raw']['jerk']['max_abs_per_joint'][0],1)
        sliced=np.gradient(np.gradient(np.gradient(values[100:104],1/240,axis=0,edge_order=2),1/240,axis=0,edge_order=2),1/240,axis=0,edge_order=2)
        self.assertGreater(abs(result['actual']['raw']['jerk']['max_abs_per_joint'][0]-np.max(np.abs(sliced[:,0]))),1)

    def test_actual_command_and_recorded_velocity_are_distinct(self):
        rows,outcome=episode_fixture(compare.TARGETS[:3])
        for row in rows:row['q_command']=[0.]*7;row['dq']=[99.]*7
        prepared=compare.prepare_episode(rows,outcome,np.eye(4));metric=compare.summarize(prepared,0,len(rows))
        self.assertGreater(metric['actual']['total_travel_rad'],0)
        self.assertEqual(metric['commanded']['total_travel_rad'],0)
        self.assertAlmostEqual(metric['actual']['raw']['velocity']['max_abs_per_joint'][0],.01)
        self.assertEqual(metric['recorded_actual_dq']['max_abs_per_joint'][0],99.)
        self.assertNotEqual(metric['tracking']['rms_per_joint'][0],0)

    def test_wrist_indices_and_wrapped_jump_retained(self):
        rows,outcome=episode_fixture(compare.TARGETS[:3])
        for row in rows:row['q']=[0.]*7
        rows[50]['q'][4]=3.1;rows[51]['q'][4]=-3.1
        prepared=compare.prepare_episode(rows,outcome,np.eye(4));m=compare.summarize(prepared,0,len(rows))['actual']
        self.assertAlmostEqual(m['wrist_joints_5_7_travel_rad'],12.4)
        self.assertAlmostEqual(m['max_step_rad'][4],6.2)
        self.assertEqual(m['elbow_flexion_joint_4_travel_rad'],0)
        self.assertEqual(compare.JOINT_CONVENTION['wrist_zero_based_indices'],[4,5,6])

    def test_travel_increments_assigned_to_arrival_without_boundary_loss(self):
        prepared,_,_=self.prepare(compare.TARGETS[:3])
        whole=compare.summarize(prepared,0,120)['actual']['total_travel_rad']
        pieces=sum(compare.summarize(prepared,start,start+40)['actual']['total_travel_rad'] for start in (0,40,80))
        self.assertAlmostEqual(whole,pieces)

    def test_missing_final_hold_does_not_win_with_partial_total(self):
        baseline,_,_=self.prepare()
        rows,outcome=episode_fixture(compare.TARGETS[:3])
        rows=rows[:89];outcome['physics_steps']=89;outcome['success']=False
        outcome['points'][-1].update(end_physics_row=89,success=False,training_accepted=False)
        outcome['points'][-1].pop('contact_end_physics_row')
        alternative=compare.prepare_episode(rows,outcome,np.eye(4))
        report=compare.compare_prepared(baseline,alternative)
        self.assertEqual(report['primary_comparison']['status'],'unavailable_incomplete_requested_window')
        self.assertIsNone(report['primary_comparison']['delta'])
        self.assertFalse(report['episode_diagnostics']['alternative']['recorded_success'])
        self.assertEqual(report['episode_diagnostics']['alternative']['recorded_completed_points'],2)
        self.assertEqual(report['point_comparisons'][0]['status'],'matched')
        self.assertIsNone(report['point_comparisons'][-1]['delta'])

    def test_short_hold_followed_by_abort_is_not_completed_prefix(self):
        baseline,_,_=self.prepare()
        rows,outcome=episode_fixture(compare.TARGETS[:3])
        for row in rows[100:]:row['phase']='ABORT'
        outcome['success']=False;outcome['points'][-1]['success']=False
        report=compare.compare_prepared(baseline,compare.prepare_episode(rows,outcome,np.eye(4)))
        self.assertEqual(report['primary_comparison']['status'],'unavailable_incomplete_requested_window')
        self.assertIsNone(report['primary_comparison']['delta'])

    def test_earlier_failed_point_and_stages_cannot_win_by_truncation(self):
        baseline,_,_=self.prepare()
        rows,outcome=episode_fixture(compare.TARGETS[:3])
        rows=rows[:27];outcome['physics_steps']=len(rows);outcome['points']=outcome['points'][:1]
        outcome['points'][0].update(end_physics_row=len(rows),success=False)
        outcome['points'][0].pop('contact_end_physics_row');outcome['success']=False
        report=compare.compare_prepared(baseline,compare.prepare_episode(rows,outcome,np.eye(4)))
        self.assertEqual(report['point_comparisons'][0]['status'],'incomplete_point_diagnostic')
        self.assertIsNone(report['point_comparisons'][0]['delta'])
        self.assertTrue(all(stage['delta'] is None for stage in report['stage_comparisons'] if stage['key'][0]=='R1'))

    def test_full10_only_for_complete_ten_point_run(self):
        baseline,_,_=self.prepare();alternative,_,_=self.prepare()
        report=compare.compare_prepared(baseline,alternative)
        self.assertEqual(report['full10_comparison']['baseline']['window']['end_row_exclusive'],400)
        self.assertEqual(report['full10_comparison']['delta']['actual_total_travel_rad'],0)
        self.assertEqual(report['upper_pair_comparison']['baseline']['window']['end_row_exclusive'],104)

    def test_hold_stability_and_release_force_zero_event(self):
        rows,outcome=episode_fixture(compare.TARGETS[:3]);rows[17]['hall_N']=5.
        prepared=compare.prepare_episode(rows,outcome,np.eye(4));contact=compare.contact_point(prepared,'R1')
        self.assertAlmostEqual(contact['hold']['duration_s'],8/240)
        self.assertAlmostEqual(contact['hold']['hall']['mean_N'],4.125)
        self.assertEqual(contact['hold']['hall']['peak_N'],5.)
        self.assertEqual(contact['hold']['hall']['outside_declared_band_samples'],1)
        self.assertEqual(contact['release']['first_exact_zero_both_forces']['row'],24)
        self.assertEqual(contact['release']['first_exact_zero_both_forces']['time_s'],25/240)

    def test_posture_reports_measured_fk_not_joint4_as_elbow_flip(self):
        prepared,_,_=self.prepare(compare.TARGETS[:3]);m=compare.summarize(prepared,0,120)
        self.assertIn('elbow_max_z_m',m['posture_measured_q_fk'])
        self.assertIn('minimum_contact_wrist_minus_elbow_z_m',m['posture_measured_q_fk'])
        self.assertIn('not a spatial elbow flip',compare.JOINT_CONVENTION['elbow_flexion_scope'])

    def test_phase_coverage_mismatch_is_explicit(self):
        baseline,_,_=self.prepare();rows,outcome=episode_fixture(compare.TARGETS[:3])
        for row in rows[8:16]:row['subphase']='different_contact'
        report=compare.compare_prepared(baseline,compare.prepare_episode(rows,outcome,np.eye(4)))
        unmatched=[x for x in report['stage_comparisons'] if x['status']!='matched']
        self.assertTrue(any(x['key'][0]=='R1' for x in unmatched))

    def test_nonfinite_and_time_gap_rejected(self):
        for change in ('time','nan'):
            rows,outcome=episode_fixture(compare.TARGETS[:3])
            if change=='time':rows[4]['time_s']+=.01
            else:rows[4]['q'][2]=float('nan')
            with self.assertRaises(ValueError):compare.prepare_episode(rows,outcome,np.eye(4))

    def test_derivative_units_follow_sample_dt(self):
        first,_,_=self.prepare(compare.TARGETS[:3],dt=1/240)
        second,_,_=self.prepare(compare.TARGETS[:3],dt=1/120)
        a=compare.summarize(first,10,30)['actual']['raw']['velocity']['max_abs_per_joint']
        b=compare.summarize(second,10,30)['actual']['raw']['velocity']['max_abs_per_joint']
        np.testing.assert_allclose(a,b,atol=1e-10)

    def test_capped_fcl_values_never_claim_exact_margin(self):
        rows,outcome=episode_fixture(compare.TARGETS[:3])
        outcome['inter_point_routes']=[dict(from_target='R1',to_target='L1',free_clearance_m=.003,
                                          edge_checks=[dict(clear=True,minimum_distance_m=.003,samples=5)])]
        prepared=compare.prepare_episode(rows,outcome,np.eye(4))
        geometry=compare.geometry_diagnostics(prepared)
        self.assertIsNone(geometry['exact_clearance_margin_m'])
        self.assertTrue(geometry['records'][0]['clear_values_are_capped_at_gate'])


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.episode=self.root/'episode';self.episode.mkdir()
        rows,outcome=episode_fixture(compare.TARGETS[:3]);self.rows=rows;self.outcome=outcome
        (self.episode/'expert_trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        (self.episode/'world_from_left_base.json').write_text(json.dumps(np.eye(4).tolist()))
        (self.episode/'outcome.json').write_text(json.dumps(outcome))
        for name in ('independent_audit.json','posture_audit.json'):
            (self.episode/name).write_text(json.dumps(dict(accepted=True,reasons=[])))
        indices=np.arange(7,len(rows),8);poses=np.zeros((len(indices),2,7));poses[:,:,3]=1
        np.savez(self.episode/'expert_replay.npz',row_indices=indices,poses=poses,
                 paths=np.array(['/World/openarm_left_link4','/World/openarm_left_link7']))
        snapshot=self.episode/'input_snapshot';(snapshot/'objects').mkdir(parents=True)
        files={}
        for target in compare.TARGETS[:3]:
            value=json.dumps(dict(points=[dict(name=target,geometric_clearance_m=.003,
                              loaded_noncontact_clearance_m=.0003)]));digest=compare.sha256_bytes(value.encode())
            (snapshot/'objects'/digest).write_text(value);files[target+'/plan']=dict(sha256=digest)
        (snapshot/'manifest.json').write_text(json.dumps(dict(files=files)))
        self.outcome['input_snapshot']='input_snapshot/manifest.json'
        (self.episode/'outcome.json').write_text(json.dumps(self.outcome))
        hashes={str(p.relative_to(self.episode)):compare.sha256(p) for p in self.episode.rglob('*') if p.is_file()}
        self.provenance=dict(status='passed',reasons=[],artifact_sha256=hashes,recorded_acceptance={'independent_audit.json':True,'posture_audit.json':True})
        patch=mock.patch.object(compare,'validate_episode_provenance',return_value=self.provenance)
        patch.start();self.addCleanup(patch.stop)

    def test_load_pins_actual_artifacts_and_captured_plans(self):
        ep=compare.load_episode(self.episode)
        self.assertEqual(set(ep['sources']['captured_plans']),set(compare.TARGETS[:3]))
        self.assertEqual(ep['sources']['artifact_sha256']['expert_trace.jsonl'],compare.sha256(self.episode/'expert_trace.jsonl'))

    def test_tamper_rejected_despite_passed_status_string(self):
        (self.episode/'expert_trace.jsonl').write_text((self.episode/'expert_trace.jsonl').read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'artifact'):compare.load_episode(self.episode)

    def test_failed_provenance_rejected(self):
        with mock.patch.object(compare,'validate_episode_provenance',return_value=dict(status='failed',reasons=['bad source'])):
            with self.assertRaisesRegex(ValueError,'provenance'):compare.load_episode(self.episode)

    def test_json_markdown_saved_outside_inputs_and_no_acceptance_claim(self):
        result=compare.compare_episodes(self.episode,self.episode,self.root/'comparison')
        self.assertIsNone(result['physics_acceptance_inferred'])
        self.assertTrue((self.root/'comparison/comparison.json').is_file())
        text=(self.root/'comparison/comparison.md').read_text()
        self.assertIn('原始',text);self.assertIn('HOLD',text);self.assertIn('不推断',text)
        with self.assertRaisesRegex(ValueError,'output'):
            compare.compare_episodes(self.episode,self.episode,self.episode/'output')


if __name__=='__main__':unittest.main()
