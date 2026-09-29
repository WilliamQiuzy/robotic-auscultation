import unittest

import numpy as np

from sim.auscultation_pi05_v6 import action_encoding as codec


class ActionEncodingTests(unittest.TestCase):
    def setUp(self):
        binding=codec.action_encoding_binding()
        self.lo=np.asarray(binding['float32_lower_rad'],np.float64)
        self.hi=np.asarray(binding['float32_upper_rad'],np.float64)
        self.mid=(self.lo+self.hi)/2

    def test_codomain_strictly_inside_raw_urdf(self):
        binding=codec.action_encoding_binding()
        self.assertTrue(np.all(self.lo>np.asarray(binding['urdf_lower_rad'])))
        self.assertTrue(np.all(self.hi<np.asarray(binding['urdf_upper_rad'])))
        np.testing.assert_array_equal(self.lo,self.lo.astype(np.float32).astype(np.float64))
        np.testing.assert_array_equal(self.hi,self.hi.astype(np.float32).astype(np.float64))
        self.assertEqual(binding['action_contract_version'],'absolute_joint_command_v1')
        self.assertEqual(len(binding['source_sha256']),64)

    def test_arbitrary_leading_shape(self):
        q=np.broadcast_to(self.mid,(2,10,7)).copy()
        z=codec.encode(q);out=codec.decode(z)
        self.assertEqual(z.shape,q.shape);self.assertEqual(z.dtype,np.float32)
        self.assertEqual(out.shape,q.shape);self.assertEqual(out.dtype,np.float32)
        np.testing.assert_allclose(out,q,rtol=0,atol=2e-6)

    def test_random_full_domain_roundtrip(self):
        rng=np.random.default_rng(1024)
        q=self.lo+(self.hi-self.lo)*rng.uniform(1e-5,1-1e-5,(2000,7))
        report=codec.inspect_support(q)
        self.assertTrue(report['strict_interior_supported'])
        self.assertTrue(report['roundtrip_within_declared_tolerance'])
        self.assertLessEqual(report['roundtrip_max_abs_rad'],2e-6)

    def test_known_joint4_near_boundary(self):
        q=self.mid.copy();q[3]=2.2e-6
        z=codec.encode(q);out=codec.decode(z)
        self.assertAlmostEqual(float(z[3]),-13.920467,places=5)
        self.assertGreater(float(out[3]),0.)
        self.assertLess(abs(float(out[3])-q[3]),2e-11)

    def test_exact_effective_endpoints_rejected(self):
        for boundary in (self.lo,self.hi):
            with self.assertRaisesRegex(ValueError,'strictly inside'):
                codec.encode(boundary)
            report=codec.inspect_support(boundary)
            self.assertFalse(report['strict_interior_supported'])
            self.assertIsNone(report['roundtrip_max_abs_rad'])

    def test_raw_endpoint_and_outside_rejected(self):
        binding=codec.action_encoding_binding()
        for q in (np.asarray(binding['urdf_lower_rad']),np.asarray(binding['urdf_upper_rad']),self.hi+1):
            with self.assertRaisesRegex(ValueError,'strictly inside'):codec.encode(q)

    def test_nearest_float32_inside_effective_bounds(self):
        lower=np.nextafter(self.lo.astype(np.float32),np.float32(np.inf))
        upper=np.nextafter(self.hi.astype(np.float32),np.float32(-np.inf))
        for q in (lower,upper):
            report=codec.inspect_support(q)
            self.assertTrue(report['strict_interior_supported'])
            self.assertTrue(report['roundtrip_within_declared_tolerance'])

    def test_extreme_finite_latents_saturate_declared_codomain(self):
        z=np.array([[-1e308]*7,[1e308]*7],np.float64)
        out=codec.decode(z)
        np.testing.assert_array_equal(out[0],self.lo.astype(np.float32))
        np.testing.assert_array_equal(out[1],self.hi.astype(np.float32))
        self.assertGreater(out[0,3],0.)

    def test_decoder_is_monotone(self):
        z=np.repeat(np.linspace(-120,120,1001)[:,None],7,axis=1)
        out=codec.decode(z)
        self.assertTrue(np.all(np.diff(out,axis=0)>=0))
        self.assertTrue(np.all(out>=self.lo) and np.all(out<=self.hi))

    def test_nonfinite_rejected(self):
        for value in (np.nan,np.inf,-np.inf):
            q=self.mid.copy();q[3]=value
            for fn in (codec.encode,codec.decode,codec.inspect_support):
                with self.assertRaisesRegex(ValueError,'finite numeric'):fn(q)

    def test_invalid_shapes_and_booleans_rejected(self):
        for value in ([],np.ones(6),np.empty((0,7)),np.ones((2,6)),True,np.ones(7,dtype=bool),['x']*7):
            for fn in (codec.encode,codec.decode,codec.inspect_support):
                with self.assertRaises(ValueError):fn(value)

    def test_input_bytes_unchanged(self):
        q=np.repeat(self.mid[None],3,axis=0).astype(np.float32);before=q.tobytes()
        codec.encode(q);codec.inspect_support(q)
        self.assertEqual(q.tobytes(),before)

    def test_precision_report_does_not_claim_exact_labels(self):
        q=self.lo+(self.hi-self.lo)*np.array([.000123,.00456,.12345,.33333,.6789,.987,.9999])
        report=codec.inspect_support(q)
        expected=float(np.max(np.abs(codec.decode(codec.encode(q)).astype(np.float64)-q)))
        self.assertEqual(report['roundtrip_max_abs_rad'],expected)
        self.assertEqual(report['roundtrip_tolerance_rad'],2e-6)
        self.assertIn('float32_reference_exact_coordinate_count',report)


if __name__=='__main__':unittest.main()
