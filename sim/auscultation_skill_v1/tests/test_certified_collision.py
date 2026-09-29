"""CPU-only native FCL tests for an optional, unintegrated optimization."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytestmark = pytest.mark.skipif(importlib.util.find_spec('fcl') is None, reason='native FCL CPU environment required')
PROJECT = Path(__file__).resolve().parents[3]


def test_rigid_motion_bound_is_conservative_for_adversarial_point_sets():
    from sim.auscultation_skill_v1.certified_collision import distance_lower_bound
    rng = np.random.default_rng(740)
    vertices = [rng.normal(size=(23, 3)), rng.normal(size=(17, 3))]
    radii = np.array([np.linalg.norm(v, axis=1).max() for v in vertices])
    old_r = np.stack([np.eye(3), np.eye(3)])
    old_t = np.array([[0., 0., 0.], [5., 0., 0.]])
    def distance(rotations, translations):
        a, b = [v @ r.T + t for v, r, t in zip(vertices, rotations, translations)]
        return np.linalg.norm(a[:, None]-b[None], axis=2).min()
    previous = distance(old_r, old_t)
    for scale in (0., 1e-10, .001, .1, 1., 10.):
        for _ in range(10):
            rotations = Rotation.from_rotvec(rng.normal(size=(2, 3))*scale).as_matrix()
            translations = old_t + rng.normal(size=(2, 3))*scale
            lower = distance_lower_bound(previous, old_r, old_t, rotations, translations, radii, 1e-8)
            assert lower <= distance(rotations, translations) + 1e-12


def test_numeric_guard_and_nonfinite_data_cannot_certify_a_pair():
    from sim.auscultation_skill_v1.certified_collision import distance_lower_bound
    rotations = np.stack([np.eye(3)]*2); translations = np.zeros((2, 3)); radii = np.ones(2)
    assert distance_lower_bound(.0003, rotations, translations, rotations, translations, radii, 1e-8) < .0003
    for value in (float('nan'), float('inf'), -.001):
        assert distance_lower_bound(value, rotations, translations, rotations, translations, radii, 1e-8) is None
    invalid = translations.copy(); invalid[0, 0] = np.nan
    assert distance_lower_bound(.1, rotations, translations, rotations, invalid, radii, 1e-8) is None
    with pytest.raises(ValueError):
        distance_lower_bound(.1, rotations, translations, rotations, translations, radii, 1e-9)


def test_batched_bounds_match_scalar_proof_and_reject_each_invalid_record():
    from sim.auscultation_skill_v1.certified_collision import distance_lower_bound, distance_lower_bounds
    rng = np.random.default_rng(157)
    old_r = Rotation.random(24, random_state=rng).as_matrix().reshape(12, 2, 3, 3)
    new_r = Rotation.random(24, random_state=rng).as_matrix().reshape(12, 2, 3, 3)
    old_t, new_t = rng.normal(size=(12, 2, 3)), rng.normal(size=(12, 2, 3))
    radii = rng.uniform(.01, 1., size=(12, 2)); distances = rng.uniform(.003, 1., size=12)
    expected = np.array([distance_lower_bound(d, a, b, c, e, r) for d, a, b, c, e, r in
                         zip(distances, old_r, old_t, new_r, new_t, radii)])
    np.testing.assert_allclose(distance_lower_bounds(distances, old_r, old_t, new_r, new_t, radii), expected, atol=1e-14, rtol=0.)
    distances[0] = np.inf; old_r[1, 0, 0, 0] = np.nan; radii[2, 0] = -1.
    actual = distance_lower_bounds(distances, old_r, old_t, new_r, new_t, radii)
    assert np.isnan(actual[:3]).all()
    np.testing.assert_allclose(actual[3:], expected[3:], atol=1e-14, rtol=0.)


@pytest.fixture(scope='module')
def native_checkers():
    from sim.auscultation_skill_v1.certified_collision import CertifiedCollisionChecker
    from sim.curobo_expert_v1.fast_collision import FastCollisionChecker
    from sim.curobo_expert_v1.static_world import build_static_world, install_fcl_static_world
    from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
    plan = json.loads((PROJECT/'sim/upper_pair_posture_20260924/results/candidate_b_plans/planned_paths_L1.json').read_text())
    kin = OpenArmKinematics(); base = np.array(plan['world_from_left_base'])
    geometry = PROJECT/'sim/auscultation_expert_layout_v1/results/expert_v2/plans'
    ordinary = FastCollisionChecker(kin, base, geometry)
    candidate = CertifiedCollisionChecker(kin, base, geometry)
    bundle = build_static_world()
    for checker in (ordinary, candidate):
        install_fcl_static_world(checker, bundle)
    return ordinary, candidate, plan['points'][0]


def test_repeated_safe_pose_reuses_only_certified_distances_and_keeps_pair_checks(native_checkers):
    ordinary, candidate, row = native_checkers
    q = np.asarray(row['loaded_q'])
    options = dict(clearance=.0003, compression=.002, allow_patient_contact=True, details=True)
    candidate.invalidate_certificates()
    expected = ordinary.check(q, **options)
    assert candidate.check(q, **options) == expected
    assert candidate.last_exact_distance_calls > 0
    assert candidate.check(q, **options) == expected
    assert candidate.last_certificate_reuses > 0
    assert candidate.last_checked_aabb_pair_count == len(ordinary.pairs)
    assert candidate.last_updated_dynamic_objects == ordinary.last_updated_dynamic_objects
    assert candidate._static_containment_installed is True


def test_contact_exemptions_clearance_changes_and_large_motion_match_native(native_checkers):
    ordinary, candidate, row = native_checkers
    q = np.asarray(row['loaded_q']); candidate.invalidate_certificates()
    for pose, compression, clearance, allowed, details in [
        (q, .002, .0003, True, True), (q, .002, .003, False, True),
        (np.zeros(7), 0., .003, False, False), (q, .0035, .0003, True, True),
        (q, .002, .0003, True, False)]:
        options = dict(compression=compression, clearance=clearance, allow_patient_contact=allowed, details=details)
        assert candidate.check(pose, **options) == ordinary.check(pose, **options)


def test_static_object_replacement_and_mutation_epoch_invalidate_old_certificates(native_checkers):
    import fcl
    ordinary, candidate, row = native_checkers
    q = np.asarray(row['loaded_q']); options = dict(compression=.002, clearance=.0003, allow_patient_contact=True)
    candidate.check(q, **options); candidate.check(q, **options)
    old = candidate.objects[:]
    # Replacing an identity with a geometrically equal native object must not
    # accidentally retain a certificate through Python id reuse.
    index = next(i for i in candidate.dynamic if any(i in key for key in candidate._certificates))
    entry = candidate.entries[index]
    arrays = np.load(PROJECT/'sim/auscultation_expert_layout_v1/results/expert_v2/plans/collision_geometry.npz')
    vertices, faces = arrays[entry['key']+'_v'], arrays[entry['key']+'_f']
    model = fcl.BVHModel(); model.beginModel(len(vertices), len(faces)); model.addSubModel(vertices, faces); model.endModel()
    candidate.objects[index] = fcl.CollisionObject(model)
    try:
        assert candidate.check(q, **options) == ordinary.check(q, **options)
        assert candidate.last_invalidated_certificates > 0
        candidate.invalidate_certificates()
        assert not candidate._certificates
    finally:
        candidate.objects = old
        candidate.invalidate_certificates()


def nested_mesh_checker():
    import fcl
    from itertools import product
    from scipy.spatial import ConvexHull
    from sim.auscultation_skill_v1.certified_collision import CertifiedCollisionChecker
    vertices = [np.array(list(product((-1., 1.), repeat=3)))*size for size in (.001, .025)]
    class Nested(CertifiedCollisionChecker):
        def __init__(self):
            self.entries = [{'key': 'moving', 'path': '/moving', 'body': 'moving'},
                            {'key': 'static', 'path': '/static', 'body': 'static'}]
            self.dynamic = [0]; self.pairs = [(0, 1)]; self.pair_indices = np.array(self.pairs)
            self._pair_a, self._pair_b = self.pair_indices.T
            self.objects = []; self.local_corners = vertices
            for v in vertices:
                triangles = ConvexHull(v).simplices.astype(np.int32)
                mesh = fcl.BVHModel(); mesh.beginModel(len(v), len(triangles)); mesh.addSubModel(v, triangles); mesh.endModel()
                self.objects.append(fcl.CollisionObject(mesh))
            self.numeric_guard_m = 1e-8; self._certificates = {}; self._radii = {}
            self.total_certificate_reuses = self.total_exact_distance_calls = 0
        def _update_bounds(self, q, compression=0.):
            self.objects[0].setTransform(fcl.Transform(Rotation.from_rotvec(q[3:6]).as_matrix(), q[:3]))
            self.bounds = np.array([[points.min(0), points.max(0)] for points in
                [v@obj.getRotation().T+obj.getTranslation() for v, obj in zip(vertices, self.objects)]])
        def _sync_near_objects(self, pairs):
            pass
    return Nested(), vertices


def test_original_containment_wrapper_rejects_even_when_all_mesh_distances_are_reused(tmp_path):
    from sim.curobo_expert_v1.static_containment import install_static_containment
    checker, vertices = nested_mesh_checker()
    np.savez(tmp_path/'collision_geometry.npz', moving_v=vertices[0])
    install_static_containment(checker, {'geometry': {'static': {'body': 'static', 'vertices_world': vertices[1]}}}, tmp_path)
    first = checker.check(np.zeros(7))
    second = checker.check(np.zeros(7))
    assert first == second
    assert second['clear'] is False and second['reason'] == 'inside_static_swept_envelope'
    assert checker.last_certificate_reuses == 1 and checker.last_exact_distance_calls == 0


def test_boundary_rotation_and_large_translation_force_exact_fallback():
    from sim.curobo_expert_v1.fast_collision import FastCollisionChecker
    checker, _ = nested_mesh_checker()
    for x, angle in [(0., 0.), (.01, 0.), (.02099999999, 0.), (.02100000001, 0.),
                     (.0215, 0.), (.0215, 0.), (.021, .7), (-.0215, -.9), (0., 0.)]:
        q = np.array([x, 0., 0., 0., 0., angle, 0.])
        expected = FastCollisionChecker.check(checker, q, clearance=.003, details=True)
        actual = checker.check(q, clearance=.003, details=True)
        assert actual == expected
        if not actual['clear']:
            assert checker.last_exact_distance_calls == 1
            assert not checker._certificates


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_native_fcl_distance_fails_closed(monkeypatch, invalid):
    import fcl
    checker, _ = nested_mesh_checker()
    monkeypatch.setattr(fcl, 'distance', lambda *args, **kwargs: invalid)
    with pytest.raises(ValueError, match='nonfinite native FCL distance'):
        checker.check(np.zeros(7))
    assert not checker._certificates


@pytest.mark.parametrize('options', [{'compression': float('nan')}, {'compression': float('inf')},
                                     {'clearance': float('inf')}, {'clearance': -.003}])
def test_invalid_query_cannot_silently_skip_all_pairs(options):
    checker, _ = nested_mesh_checker()
    with pytest.raises(ValueError, match='invalid collision query'):
        checker.check(np.zeros(7), **options)


def test_invalid_bounds_are_rejected_before_near_pair_filter(monkeypatch):
    checker, _ = nested_mesh_checker()
    original = checker._update_bounds
    def invalid(*args, **kwargs):
        original(*args, **kwargs)
        checker.bounds[0, 0, 0] = np.nan
    monkeypatch.setattr(checker, '_update_bounds', invalid)
    with pytest.raises(ValueError, match='invalid collision bounds'):
        checker.check(np.zeros(7))


@pytest.mark.parametrize('rotation,translation', [(np.eye(3).reshape(9), np.zeros(3)),
                                                  (np.eye(3), np.zeros((1, 3))),
                                                  (2.*np.eye(3), np.zeros(3)),
                                                  (np.eye(3), np.array([np.nan, 0., 0.]))])
def test_batch_native_pose_validation_keeps_shape_and_rigidity_guards(rotation, translation):
    from types import SimpleNamespace
    checker, _ = nested_mesh_checker()
    checker.objects[0] = SimpleNamespace(getRotation=lambda: rotation, getTranslation=lambda: translation)
    with pytest.raises(ValueError, match='invalid native collision transform'):
        checker._native_poses(np.array([0]))
