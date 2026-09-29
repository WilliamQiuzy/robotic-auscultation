"""Saved-command audit tests, especially boundary windows hidden by padding."""
import importlib
import unittest

import numpy as np

DT=1/240.


class FreeMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit=importlib.import_module('sim.expert_quality_20260924.audit_free_motion')

    def fixture(self):
        # In units k, segment A ends with increments [1,2,2,1,0],
        # and segment B starts with them. Each padded third difference <=k;
        # the real splice includes [1,0,1], whose second difference is 2k.
        k=15*DT**3
        a=np.cumsum(np.r_[np.zeros(7),[k,2*k,2*k,k,0.]])
        b=a[-1]+np.cumsum(np.r_[[k,2*k,2*k,k,0.],np.zeros(7)])
        q=np.zeros((24,7));q[:,0]=np.r_[a,b]
        rows=[dict(time_s=(i+1)*DT,q_command=q[i].tolist(),subphase='transfer',target='R1',phase='APPROACH') for i in range(24)]
        records=[]
        for start,end in ((0,12),(12,24)):
            records.append(dict(start_physics_row=start,end_physics_row_exclusive=end,motion_complete=True,
                duration_s=12*DT,physics_dt_s=DT,endpoint_stationary_padding_samples=3,
                target='R1',subphase='transfer',velocity_limit_rad_s=.25,acceleration_limit_rad_s2=2.,jerk_limit_rad_s3=20.,
                peaks={'velocity_rad_s':2*k/DT,'acceleration_rad_s2':k/DT**2,'jerk_rad_s3':15.}))
        outcome=dict(candidate_policy='staged_continuous',free_motion_segments=records,
            free_motion_command_bounds=dict(joint_velocity_rad_s=.25,joint_acceleration_rad_s2=2.,joint_jerk_rad_s3=20.))
        return rows,outcome

    def test_each_segment_passes_but_real_splice_fails(self):
        rows,outcome=self.fixture()
        result=self.audit.analyze_free_motion(rows,outcome,initial_q=np.zeros(7))
        self.assertTrue(all(x['reported_peaks_match'] for x in result['segments']))
        self.assertTrue(all(x['padded_limits_passed'] for x in result['segments']))
        peak=result['actual_windows']['jerk_rad_s3']['cross_segment_peak']
        self.assertAlmostEqual(peak['value'],30.)
        self.assertTrue(peak['cross_segment_boundary'])
        self.assertIn('actual_free_motion_jerk_rad_s3_limit',result['failures'])

    def test_contact_ik_windows_are_excluded_with_counts(self):
        rows,outcome=self.fixture()
        # An unrelated large contact IK command must not be called a free-generator failure.
        for i in range(24,36):
            rows.append(dict(time_s=(i+1)*DT,q_command=[2.]*7,subphase='contact',target='R1',phase='CONTACT'))
        result=self.audit.analyze_free_motion(rows,outcome,initial_q=np.zeros(7))
        for order,name in enumerate(('velocity_rad_s','acceleration_rad_s2','jerk_rad_s3'),1):
            metric=result['actual_windows'][name]
            self.assertEqual(metric['excluded_mixed_free_and_nonfree_windows'],order)
            self.assertEqual(metric['excluded_all_nonfree_windows'],12-order)
            self.assertLess(metric['peak']['value'],100.)

    def test_report_peak_count_and_coverage_tampering_rejected(self):
        for mutation,reason in [('peak','reported_peak_mismatch'),('count','command_count_mismatch'),('hole','uncovered_free_subphase_rows')]:
            rows,outcome=self.fixture()
            if mutation=='peak':outcome['free_motion_segments'][0]['peaks']['jerk_rad_s3']=1.
            if mutation=='count':outcome['free_motion_segments'][0]['duration_s']+=DT
            if mutation=='hole':outcome['free_motion_segments'][0]['start_physics_row']=1;outcome['free_motion_segments'][0]['duration_s']-=DT
            result=self.audit.analyze_free_motion(rows,outcome,initial_q=np.zeros(7))
            self.assertTrue(any(reason in x for x in result['failures']),result['failures'])

    def test_baseline_without_segment_metadata_has_limited_scope(self):
        rows,_=self.fixture()
        result=self.audit.analyze_free_motion(rows,{},initial_q=np.zeros(7))
        self.assertEqual(result['metadata_scope'],'subphase_only_baseline_diagnostic')
        self.assertFalse(result['controller_claims_verified'])
        self.assertGreater(result['actual_windows']['jerk_rad_s3']['evaluated_windows'],0)

    def test_initial_command_missing_is_pending_not_assumed_zero(self):
        rows,outcome=self.fixture()
        result=self.audit.analyze_free_motion(rows,outcome)
        self.assertIn('initial_command_before_first_row_unavailable',result['pending'])
        self.assertIsNone(result['segments'][0]['reported_peaks_match'])

    def test_candidate_missing_metadata_is_failed_not_baseline_diagnostic(self):
        rows,_=self.fixture()
        result=self.audit.analyze_free_motion(rows,{'candidate_policy':'staged_continuous'},initial_q=np.zeros(7))
        self.assertEqual(result['status'],'failed')
        self.assertIn('candidate_missing_free_motion_segments',result['failures'])

    def test_actual_below_limits_and_bound_sources_can_pass_claims(self):
        rows,outcome=self.fixture()
        for row in rows:row['q_command']=(np.asarray(row['q_command'])*.5).tolist()
        for record in outcome['free_motion_segments']:
            record['peaks']={key:value*.5 for key,value in record['peaks'].items()}
        result=self.audit.analyze_free_motion(rows,outcome,initial_q=np.zeros(7),source_binding={'status':'passed'})
        self.assertEqual(result['status'],'passed')
        self.assertTrue(result['controller_claims_verified'])
        result=self.audit.analyze_free_motion(rows,outcome,initial_q=np.zeros(7),source_binding={'status':'failed'})
        self.assertEqual(result['status'],'failed')

    def test_stricter_declared_segment_limit_is_enforced(self):
        rows,outcome=self.fixture()
        outcome['free_motion_segments'][0]['jerk_limit_rad_s3']=10.
        result=self.audit.analyze_free_motion(rows,outcome,initial_q=np.zeros(7),source_binding={'status':'passed'})
        self.assertFalse(result['segments'][0]['padded_limits_passed'])
        self.assertIn('segment_0:padded_command_limit',result['failures'])

    def test_nan_and_bad_timing_rejected(self):
        for key,value in [('q_command',[float('nan')]*7),('time_s',1.)]:
            rows,outcome=self.fixture();rows[2][key]=value
            with self.assertRaises(ValueError):self.audit.analyze_free_motion(rows,outcome,initial_q=np.zeros(7))

if __name__=='__main__':unittest.main()
