"""Fixed derivative estimates are reference hints, never waypoint feasibility."""
import importlib
import unittest

import numpy as np


class MotionHintTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module('sim.ausc_policy.motion_hints')
        except ModuleNotFoundError:
            self.fail('The fixed motion hint helper is not implemented')

    def test_quadratic_derivative_is_at_original_label_times_including_first_two(self):
        m=self.module();t=(np.arange(10)+1)*.05
        offset=np.linspace(-.3,.3,7);v=np.linspace(-.02,.03,7);acc=np.linspace(.01,.08,7)
        q=offset+t[:,None]*v+.5*t[:,None]**2*acc
        before=q.copy();result=m.velocity_hints(q)
        np.testing.assert_allclose(result['velocity'],v+t[:5,None]*acc,rtol=0,atol=2e-15)
        np.testing.assert_array_equal(q,before)
        np.testing.assert_array_equal(result['raw_secant'],np.diff(q[:6],axis=0)/.05)
        self.assertEqual(result['method'],'quadratic5')

    def test_fixed_windows_use_only_current_chunk_and_constant_offsets_cancel(self):
        m=self.module();rng=np.random.default_rng(4);q=rng.normal(0,.0005,(10,7))
        first=m.velocity_hints(q)['velocity']
        shifted=m.velocity_hints(q+2.)['velocity']
        np.testing.assert_allclose(first,shifted,rtol=0,atol=1e-14)
        q[7:]=100.
        np.testing.assert_array_equal(m.velocity_hints(q)['velocity'],first)
        np.testing.assert_array_equal(m.velocity_hints(np.full((10,7),2.))['velocity'],np.zeros((5,7)))

    def test_raw_secant_excess_remains_visible_when_estimate_is_bounded(self):
        m=self.module();q=np.zeros((10,7));q[:,0]=np.arange(10)*.005+(-1.)**np.arange(10)*.002
        result=m.velocity_hints(q)
        self.assertTrue(result['raw_secant_exceeds_limit'])
        self.assertGreater(result['summary']['raw_secant_max_abs_rad_s'],.175)
        self.assertLess(result['summary']['estimated_velocity_max_abs_rad_s'],.175)
        self.assertGreater(result['summary']['raw_secant_exceedance_count'],0)
        self.assertGreater(result['summary']['estimated_minus_raw_max_abs_rad_s'],0.)
        np.testing.assert_array_equal(result['raw_secant'],np.diff(q[:6],axis=0)/.05)
        with self.assertRaisesRegex(ValueError,'velocity'):m.velocity_hints(q,'adjacent')

    def test_true_excess_is_refused_without_clipping_and_zero_keeps_raw_flags(self):
        m=self.module();q=np.arange(10)[:,None]*np.full((1,7),.01)
        for method in ('quadratic5','adjacent'):
            with self.subTest(method=method),self.assertRaisesRegex(ValueError,'velocity'):
                m.velocity_hints(q,method)
        result=m.velocity_hints(q,'zero')
        np.testing.assert_array_equal(result['velocity'],np.zeros((5,7)))
        self.assertTrue(result['raw_secant_exceeds_limit'])

    def test_wrong_shape_dtype_nonfinite_and_unknown_method_fail_closed(self):
        m=self.module()
        for value in (np.zeros((5,7)),np.zeros((10,8)),np.full((10,7),np.nan),
                      np.full((10,7),'0.1'),np.ones((10,7),dtype=complex)):
            with self.subTest(shape=value.shape,dtype=value.dtype),self.assertRaises(ValueError):
                m.velocity_hints(value)
        with self.assertRaises(ValueError):m.velocity_hints(np.zeros((10,7)),'auto')


if __name__=='__main__':unittest.main()
