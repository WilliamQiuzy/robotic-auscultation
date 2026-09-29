import importlib.util
from itertools import product
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog


class ObjectPose:
    def __init__(self, vertices, translation=None):
        self.vertices = vertices
        self.translation = np.zeros(3) if translation is None else np.asarray(translation)
    def getRotation(self):
        return np.eye(3)
    def getTranslation(self):
        return self.translation


class SurfaceChecker:
    def __init__(self, inner, outer, pairs=None, source_report=None):
        self.entries = [{'key':'moving', 'path':'/moving', 'body':'moving'},
                        {'key':'static', 'path':'/static', 'body':'static'}]
        self.dynamic = [0]
        self.pairs = [(0, 1)] if pairs is None else pairs
        self.pair_indices = np.array(self.pairs, dtype=int).reshape(-1, 2)
        self.objects = [ObjectPose(inner), ObjectPose(outer)]
        self.bounds = np.array([[v.min(0), v.max(0)] for v in (inner, outer)])
        self.report = source_report
    def check(self, q, clearance=.003, **kwargs):
        if self.report is not None:
            return dict(self.report)
        return {'clear':True, 'minimum_distance_m':clearance, 'pair':None}


def bundle_for(vertices):
    return {'geometry':{'static':{'vertices_world':vertices, 'body':'static'}}}


class StaticContainmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.corners = np.array(list(product((-1., 1.), repeat=3)))
        self.inner = self.corners*.001
        self.outer = self.corners*.025
        np.savez(self.directory/'collision_geometry.npz', moving_v=self.inner)
    def tearDown(self):
        self.temp.cleanup()
    def install(self, checker, outer=None):
        from sim.curobo_expert_v1.static_containment import install_static_containment
        return install_static_containment(checker, bundle_for(self.outer if outer is None else outer), self.directory)
    def test_dynamic_inside_static_envelope_is_rejected(self):
        checker = SurfaceChecker(self.inner, self.outer)
        self.assertTrue(checker.check(np.zeros(7))['clear'])
        receipt = self.install(checker)
        result = checker.check(np.zeros(7))
        self.assertFalse(result['clear'])
        self.assertEqual(result['reason'], 'inside_static_swept_envelope')
        self.assertEqual(result['pair'], ['/moving', '/static'])
        self.assertEqual(receipt['checked_original_pairs'], 1)
    def test_only_original_pairs_are_checked(self):
        checker = SurfaceChecker(self.inner, self.outer, pairs=[])
        self.install(checker)
        self.assertTrue(checker.check(np.zeros(7))['clear'])
    def test_world_transform_and_broadphase_are_used(self):
        checker = SurfaceChecker(self.inner, self.outer)
        checker.objects[0].translation = np.array([.05, 0, 0])
        checker.bounds[0] += [.05, 0, 0]
        self.install(checker)
        self.assertTrue(checker.check(np.zeros(7))['clear'])
    def test_existing_surface_failure_is_preserved(self):
        report = {'clear':False, 'minimum_distance_m':.001, 'pair':['/a','/b']}
        checker = SurfaceChecker(self.inner, self.outer, source_report=report)
        self.install(checker)
        self.assertEqual(checker.check(np.zeros(7)), report)
    def test_boundary_vertex_is_conservatively_rejected(self):
        checker = SurfaceChecker(self.inner, self.outer)
        checker.objects[0].translation = np.array([.026, 0, 0])
        checker.bounds[0] += [.026, 0, 0]
        self.install(checker)
        self.assertFalse(checker.check(np.zeros(7))['clear'])
    def test_missing_source_vertices_fails_closed(self):
        np.savez(self.directory/'collision_geometry.npz', wrong_v=self.inner)
        with self.assertRaises((KeyError, ValueError)):
            self.install(SurfaceChecker(self.inner, self.outer))
    @unittest.skipUnless(importlib.util.find_spec('fcl'), 'native FCL is Linux-only')
    def test_actual_finger_sweep_native_fcl_containment_counterexample(self):
        import fcl
        import json
        source = Path(__file__).resolve().parents[2]/'auscultation_expert_layout_v1/results/expert_v2/plans'
        metadata = json.loads((source/'collision_geometry.json').read_text())
        with np.load(source/'collision_geometry.npz') as arrays:
            finger = next(e for e in metadata['colliders'] if e['body'].endswith('openarm_right_right_finger'))
            parent = next(e for e in metadata['colliders'] if e['body'].endswith('openarm_right_link7'))
            vertices = arrays[finger['key']+'_v']
        transform = np.asarray(parent['world_from_body']); origin = np.array([0,0,.1025]); axis = np.array([0,-1,0])
        local = np.concatenate([vertices+origin, vertices+origin+.044*axis])
        padding = .0004+1e-7+2*np.sin(.001/2)*(np.linalg.norm(local,axis=1).max()+np.linalg.norm(vertices,axis=1).max())
        world = local@transform[:3,:3].T+transform[:3,3]
        outer = (world[:,None]+padding*self.corners[None]).reshape(-1,3)
        hull = ConvexHull(outer)
        result = linprog([0,0,0,-1], A_ub=np.c_[hull.equations[:,:3], np.ones(len(hull.equations))],
                         b_ub=-hull.equations[:,3], bounds=[(None,None)]*3+[(0,None)], method='highs')
        self.assertTrue(result.success)
        center = result.x[:3]
        def obj(v):
            triangles = ConvexHull(v).simplices.astype(np.int32)
            mesh = fcl.BVHModel(); mesh.beginModel(len(v),len(triangles)); mesh.addSubModel(v,triangles); mesh.endModel()
            return fcl.CollisionObject(mesh)
        inner_object = obj(self.inner)
        inner_object.setTransform(fcl.Transform(np.eye(3),center))
        outer_object = obj(outer)
        distance = float(fcl.distance(inner_object,outer_object,fcl.DistanceRequest(),fcl.DistanceResult()))
        self.assertGreater(distance,.003)
        checker = SurfaceChecker(self.inner, outer, source_report={'clear':True,'minimum_distance_m':.003,'pair':None})
        checker.objects = [inner_object,outer_object]
        checker.bounds[0] += center
        self.install(checker,outer)
        self.assertFalse(checker.check(np.zeros(7))['clear'])


if __name__ == '__main__':
    unittest.main()
