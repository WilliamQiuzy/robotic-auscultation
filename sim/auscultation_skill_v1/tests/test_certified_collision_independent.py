"""Independent adversarial checks; no production checker integration."""
import importlib.util
from itertools import product

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytestmark = pytest.mark.skipif(importlib.util.find_spec('fcl') is None,
                               reason='native FCL CPU environment required')


def two_boxes():
    import fcl
    from sim.auscultation_skill_v1.certified_collision import CertifiedCollisionChecker
    class Boxes(CertifiedCollisionChecker):
        def __init__(self):
            self.entries = [{'path': '/robot/Diaphragm', 'body': 'moving'},
                            {'path': '/World/Patient/shirt', 'body': 'patient'}]
            self.objects = [fcl.CollisionObject(fcl.Box(.002, .004, .006)) for _ in range(2)]
            corners = np.array(list(product((-.001, .001), (-.002, .002), (-.003, .003))))
            self.local_corners = [corners.copy(), corners.copy()]
            self.dynamic = [0]; self.pairs = [(0, 1)]; self.pair_indices = np.array(self.pairs)
            self._pair_a, self._pair_b = self.pair_indices.T
            self.numeric_guard_m = 1e-8; self._certificates = {}; self._radii = {}
            self.total_certificate_reuses = self.total_exact_distance_calls = 0
            self.second_rotation = np.eye(3); self.second_position = np.array([.011, 0., 0.])
        def _update_bounds(self, q, compression=0.):
            q = np.asarray(q)
            transforms = [(Rotation.from_rotvec(q[3:6]).as_matrix(), q[:3]),
                          (self.second_rotation, self.second_position + [0., 0., compression])]
            bounds = []
            for obj, corners, (rotation, translation) in zip(self.objects, self.local_corners, transforms):
                obj.setTransform(fcl.Transform(rotation, translation))
                points = corners @ rotation.T + translation
                # A conservative enlarged AABB forces examination of this pair.
                bounds.append([points.min(0)-.1, points.max(0)+.1])
            self.bounds = np.asarray(bounds)
        def _sync_near_objects(self, pairs):
            self.synced_pairs = np.asarray(pairs).copy()
    return Boxes()


def test_native_clearance_equality_remains_clear_but_cannot_be_reused(monkeypatch):
    import fcl
    checker = two_boxes()
    monkeypatch.setattr(fcl, 'distance', lambda *args: .003)
    for _ in range(2):
        result = checker.check(np.zeros(7), clearance=.003)
        assert result == {'clear': True, 'minimum_distance_m': .003, 'pair': None}
        assert checker.last_exact_distance_calls == 1
        assert checker.last_certificate_reuses == 0
        assert not checker._certificates


def test_pair_is_checked_again_when_contact_permission_is_removed():
    checker = two_boxes(); checker.second_position[:] = 0.
    allowed = checker.check(np.zeros(7), allow_patient_contact=True)
    assert allowed['clear'] and checker.last_exact_distance_calls == 0
    forbidden = checker.check(np.zeros(7), allow_patient_contact=False)
    assert not forbidden['clear'] and checker.last_exact_distance_calls == 1
    assert not checker._certificates


def test_cached_pair_survives_broadphase_absence_but_large_return_motion_falls_back():
    checker = two_boxes(); q = np.zeros(7)
    assert checker.check(q)['clear']
    assert checker.check(q)['clear'] and checker.last_certificate_reuses == 1
    q[0] = -2.
    assert checker.check(q)['clear'] and checker.last_near_pair_count == 0
    q[0] = .010
    assert not checker.check(q)['clear']
    assert checker.last_certificate_reuses == 0 and checker.last_exact_distance_calls == 1


def test_both_body_rigid_displacements_bound_actual_native_distance():
    import fcl
    from sim.auscultation_skill_v1.certified_collision import distance_lower_bound
    checker = two_boxes(); checker.check(np.zeros(7))
    old_r = np.asarray([o.getRotation() for o in checker.objects])
    old_t = np.asarray([o.getTranslation() for o in checker.objects])
    radii = np.asarray([np.linalg.norm(c, axis=1).max() for c in checker.local_corners])
    old_distance = fcl.distance(*checker.objects)
    rng = np.random.default_rng(93017)
    for index in range(80):
        scale = (1e-4, .1, 1.)[index % 3]
        q = np.r_[rng.uniform(-.008, .008, 3)*scale, rng.uniform(-np.pi, np.pi, 3)*scale, 0.]
        checker.second_rotation = Rotation.from_rotvec(rng.uniform(-np.pi, np.pi, 3)*scale).as_matrix()
        checker.second_position = old_t[1] + rng.uniform(-.006, .006, 3)*scale
        checker._update_bounds(q)
        rotations = np.asarray([o.getRotation() for o in checker.objects])
        translations = np.asarray([o.getTranslation() for o in checker.objects])
        native = fcl.distance(*checker.objects)
        lower = distance_lower_bound(old_distance, old_r, old_t, rotations, translations, radii)
        # Negative native values encode collision; unsigned separation is zero.
        assert lower <= max(0., native) + 1e-12, index


@pytest.mark.parametrize('distance', [np.nan, np.inf, -np.inf])
def test_nonfinite_native_result_fails_closed_and_does_not_seed_cache(monkeypatch, distance):
    import fcl
    checker = two_boxes()
    monkeypatch.setattr(fcl, 'distance', lambda *args: distance)
    with pytest.raises(ValueError):
        checker.check(np.zeros(7), details=True)
    assert not checker._certificates


@pytest.mark.parametrize('kind', ['q', 'compression', 'clearance', 'bounds'])
def test_invalid_input_or_broadphase_bounds_cannot_skip_every_pair(kind, monkeypatch):
    checker = two_boxes(); q = np.zeros(7); options = {}
    if kind == 'q':
        q[0] = np.nan
    elif kind in ('compression', 'clearance'):
        options[kind] = np.nan
    def update(*args):
        checker.bounds = np.zeros((2, 2, 3))
        if kind == 'bounds':
            checker.bounds[0, 0, 0] = np.nan
    monkeypatch.setattr(checker, '_update_bounds', update)
    with pytest.raises(ValueError):
        checker.check(q, **options)


def test_new_larger_geometry_object_invalidates_old_distance_and_radius():
    import fcl
    checker = two_boxes(); q = np.zeros(7)
    assert checker.check(q)['clear']
    assert checker.check(q)['clear'] and checker.last_certificate_reuses == 1
    checker.objects[1] = fcl.CollisionObject(fcl.Box(.040, .040, .040))
    checker.local_corners[1] = np.array(list(product((-.020, .020), repeat=3)))
    result = checker.check(q)
    assert not result['clear']
    assert checker.last_invalidated_certificates == 1
    assert checker.last_exact_distance_calls == 1
    assert checker._radii[1][1] == pytest.approx(np.sqrt(3)*.020)


def test_batched_mixed_pair_indices_exemptions_replacements_and_early_return(monkeypatch):
    import fcl
    from dataclasses import replace
    checker = two_boxes()
    checker.entries = [{'path': '/robot/Diaphragm', 'body': 'a'},
                       {'path': '/robot/frame', 'body': 'b'},
                       {'path': '/World/Patient/shirt', 'body': 'c'},
                       {'path': '/world/table', 'body': 'd'}]
    checker.objects = [fcl.CollisionObject(fcl.Box(.002, .002, .002),
                       fcl.Transform(np.eye(3), [index*.01, 0., 0.])) for index in range(4)]
    checker.local_corners = [np.array(list(product((-.001, .001), repeat=3))) for _ in range(4)]
    # The active/cached subsets deliberately do not follow sorted object IDs.
    checker.pairs = [(0, 2), (0, 1), (1, 3), (2, 3)]
    checker.pair_indices = np.array(checker.pairs)
    checker._pair_a, checker._pair_b = checker.pair_indices.T
    def bounds(*args):
        checker.bounds = np.array([[obj.getTranslation()-.1, obj.getTranslation()+.1]
                                   for obj in checker.objects])
    monkeypatch.setattr(checker, '_update_bounds', bounds)
    q = np.zeros(7)
    assert checker.check(q)['clear'] and checker.last_exact_distance_calls == 4
    assert checker.check(q)['clear'] and checker.last_certificate_reuses == 4
    checker.objects[1] = fcl.CollisionObject(fcl.Box(.002, .002, .002),
                                           fcl.Transform(np.eye(3), [.01, 0., 0.]))
    assert checker.check(q, allow_patient_contact=True)['clear']
    assert checker.last_invalidated_certificates == 2
    assert checker.last_exact_distance_calls == 2 and checker.last_certificate_reuses == 1
    assert checker.check(q)['clear'] and checker.last_certificate_reuses == 4
    checker._certificates[(1, 3)] = replace(checker._certificates[(1, 3)], distance=np.nan)
    assert checker.check(q)['clear']
    assert checker.last_exact_distance_calls == 1 and checker.last_certificate_reuses == 3
    for certificate in checker._certificates.values():
        assert not any(value.flags.writeable for value in
                       (certificate.rotations, certificate.translations, certificate.radii))
    collision = checker.check(q, clearance=.01)
    assert not collision['clear']
    assert collision['pair'] == ['/robot/Diaphragm', '/robot/frame']
    assert collision['minimum_distance_m'] == pytest.approx(.008)
    assert checker.last_certificate_reuses == 1 and checker.last_exact_distance_calls == 1
