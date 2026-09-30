"""Contract regressions runnable without native FCL on the development Mac."""
import numpy as np
import pytest

from sim.ausc_policy.contact_collision import (
    ContactCollisionChecker, _ContactPolicyMixin, pair_clearance,
)


FACE = '/World/Stethoscope/Carriage/Colliders/Diaphragm'
LIP = '/World/Stethoscope/Carriage/Colliders/RetentionLip/sector_00'
RIM = '/World/Stethoscope/Carriage/Colliders/ContactRim/sector_00'
SHELL = '/World/Stethoscope/Carriage/Colliders/DeviceEdge'
PATIENT = '/World/Patient/Body'
ROBOT = '/World/StethoArmMount/OpenArmStetho/openarm_left_link3/collision'


class FakeObject:
    def getRotation(self):
        return np.eye(3)

    def getTranslation(self):
        return np.zeros(3)


class FakeChecker(_ContactPolicyMixin):
    """Native-distance oracle plus bounds/FK call recording, no FCL substitute."""
    def __init__(self, paths, distances, *, bounds=None, planes=()):
        self.entries = [{'path': p} for p in paths]
        self.pairs = [(i, len(paths)-1) for i in range(len(paths)-1)]
        self.pair_indices = np.asarray(self.pairs, int).reshape(-1, 2)
        self._pair_a, self._pair_b = self.pair_indices.T
        self.dynamic = list(range(len(paths)-1))
        self.bounds = np.asarray(bounds if bounds is not None else
            [[[0, 0, .01], [.01, .01, .02]]] * len(paths), float)
        self._body_transforms = np.repeat(np.eye(4)[None], len(self.dynamic), axis=0)
        self.objects = [FakeObject() for _ in paths]
        self.analytic_planes = list(planes)
        self.distances = distances
        self.distance_calls = []
        self.update_calls = []

    def _update_bounds(self, q, compression=0.):
        self.update_calls.append((q.copy(), compression))

    def _sync_near_objects(self, pairs):
        self.synced_pairs = pairs.copy()

    def _native_distance(self, i, j):
        self.distance_calls.append((i, j))
        return self.distances[i]


def test_only_diaphragm_is_exempt_and_contact_policy_is_symmetric():
    for a, b in ((FACE, PATIENT), (PATIENT, FACE)):
        assert pair_clearance(a, b, .0003, True) is None
    for path in (LIP, RIM, ROBOT, '/World/Other/Diaphragm'):
        assert pair_clearance(path, PATIENT, .0003, True) == .003
    assert pair_clearance(FACE, '/World/Table/table_low', .0003, True) == .003
    assert pair_clearance(SHELL, PATIENT, .003, True) == .0003
    assert pair_clearance(SHELL, PATIENT, .0003, False) == .003
    assert pair_clearance(ROBOT, PATIENT, .005, True) == .005


@pytest.mark.parametrize('forbidden', [LIP, RIM, ROBOT])
def test_contact_and_small_requested_clearance_cannot_exempt_other_parts(forbidden):
    checker = FakeChecker([FACE, forbidden, PATIENT], [-1., .002])
    result = checker.check(np.zeros(7), clearance=.0003, allow_patient_contact=True)
    assert not result['clear']
    assert result['pair'] == [forbidden, PATIENT]
    assert checker.distance_calls == [(1, 2)]
    assert checker.last_checked_aabb_pair_count == 2
    assert len(checker.update_calls) == 1


def test_shell_and_membrane_contact_require_distinct_thresholds():
    checker = FakeChecker([FACE, SHELL, RIM, PATIENT], [-1., .0004, .0031])
    assert checker.check(np.zeros(7), allow_patient_contact=True)['clear']
    checker.distances[1] = .0002
    assert not checker.check(np.zeros(7), allow_patient_contact=True)['clear']
    checker.distances[1] = .0004
    assert not checker.check(np.zeros(7), allow_patient_contact=False)['clear']


def test_broadphase_uses_general_floor_even_when_caller_requests_less():
    bounds = [[[0, 0, .01], [.01, .01, .02]],
              [[.012, 0, .01], [.022, .01, .02]]]
    checker = FakeChecker([RIM, PATIENT], [.002], bounds=bounds)
    assert not checker.check(np.zeros(7), clearance=.0003, allow_patient_contact=True)['clear']
    assert checker.distance_calls == [(0, 1)]


def test_all_pairs_and_exact_compressed_fk_are_checked_for_each_query():
    checker = FakeChecker([SHELL, RIM, PATIENT], [.0004, .0031])
    for q, compression in ((np.zeros(7), .001), (np.full(7, .2), .004)):
        assert checker.check(q, allow_patient_contact=True, compression=compression)['clear']
        assert checker.last_checked_aabb_pair_count == 2
        np.testing.assert_array_equal(checker.update_calls[-1][0], q)
        assert checker.update_calls[-1][1] == compression
    assert checker.distance_calls == [(0, 2), (1, 2), (0, 2), (1, 2)]


def test_plane_is_checked_outside_finite_ground_mesh_at_general_clearance():
    bounds = [[[100, 100, .002], [101, 101, .01]],
              [[0, 0, 1], [1, 1, 2]]]
    checker = FakeChecker([ROBOT, PATIENT], [10.], bounds=bounds,
        planes=[dict(path='/World/GroundPlane/collisionPlane', normal=[0, 0, 1], point=[0, 0, 0])])
    result = checker.check(np.zeros(7), clearance=.0003, allow_patient_contact=True)
    assert not result['clear']
    assert result['minimum_distance_m'] == pytest.approx(.002)
    assert result['reason'] == 'analytic_plane_clearance'


@pytest.mark.parametrize('kwargs', [
    {'q': np.full(7, np.nan)}, {'q': np.zeros(6)}, {'clearance': np.nan},
    {'clearance': 0.}, {'compression': np.inf}, {'compression': -.0001},
    {'compression': .0061}, {'allow_patient_contact': 'yes'},
])
def test_invalid_query_never_passes(kwargs):
    checker = FakeChecker([RIM, PATIENT], [.01])
    args = {'q': np.zeros(7), **kwargs}
    with pytest.raises(ValueError):
        checker.check(**args)


def test_nonfinite_bounds_or_native_distance_never_passes():
    checker = FakeChecker([RIM, PATIENT], [np.nan])
    with pytest.raises(ValueError, match='distance'):
        checker.check(np.zeros(7))
    checker.bounds[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='bounds'):
        checker.check(np.zeros(7))


def test_retired_geometry_fails_before_native_fcl_import(tmp_path):
    with pytest.raises(ValueError, match='Retired'):
        ContactCollisionChecker(None, np.eye(4), tmp_path, {})


def test_current_epoch_still_requires_both_artifact_hashes_before_native_import(tmp_path):
    from sim.ausc_policy.geometry_epoch import GEOMETRY_ID, GEOMETRY_SOURCE_SHA256
    metadata = dict(geometry_id=GEOMETRY_ID, geometry_source_sha256=GEOMETRY_SOURCE_SHA256,
                    freshly_generated=True)
    with pytest.raises(ValueError, match='artifact hash'):
        ContactCollisionChecker(None, np.eye(4), tmp_path, metadata)


def test_constructor_installs_current_static_sweep_and_containment(monkeypatch, tmp_path):
    import hashlib
    import sys
    from types import ModuleType
    import sim.ausc_policy.contact_collision as module
    from sim.ausc_policy.geometry_epoch import GEOMETRY_ID, GEOMETRY_SOURCE_SHA256
    metadata = dict(geometry_id=GEOMETRY_ID, geometry_source_sha256=GEOMETRY_SOURCE_SHA256,
                    freshly_generated=True)
    for name, key in (('collision_geometry.json', 'collision_geometry_sha256'),
                      ('collision_geometry.npz', 'collision_arrays_sha256')):
        (tmp_path / name).write_bytes(b'fixture')
        metadata[key] = hashlib.sha256(b'fixture').hexdigest()
    monkeypatch.setattr(module, '_validate_geometry', lambda directory: {'analytic_planes': []})
    calls = []
    fake_fcl = ModuleType('fcl')
    fake_fast = ModuleType('sim.curobo_expert_v1.fast_collision')

    class FastBase:
        def __init__(self, kin, base, directory):
            calls.append(('native', directory))

    fake_fast.FastCollisionChecker = FastBase
    fake_static = ModuleType('sim.curobo_expert_v1.static_world')

    def build(*, geometry_dir, urdf_path):
        calls.append(('build', geometry_dir, urdf_path))
        return {'current': True}

    def install(checker, bundle, *, geometry_dir):
        assert isinstance(checker, _ContactPolicyMixin)
        assert bundle == {'current': True}
        calls.append(('install', geometry_dir))
        checker._static_containment_installed = True
        return {'containment_gate': 'fixture'}

    fake_static.build_static_world = build
    fake_static.install_fcl_static_world = install
    for name, value in (('fcl', fake_fcl), (fake_fast.__name__, fake_fast),
                        (fake_static.__name__, fake_static)):
        monkeypatch.setitem(sys.modules, name, value)
    checker = ContactCollisionChecker(None, np.eye(4), tmp_path, metadata)
    assert [c[0] for c in calls] == ['native', 'build', 'install']
    assert all(c[1] == tmp_path for c in calls)
    assert calls[1][2].name == 'openarm_v10_follower_stetho.urdf'
    assert calls[1][2].is_file()
    assert checker._static_containment_installed


def test_edge_rechecks_each_sample_and_carries_compression():
    wrapper = ContactCollisionChecker.__new__(ContactCollisionChecker)
    wrapper.checker = FakeChecker([RIM, PATIENT], [.0031])
    result = wrapper.edge(np.zeros(7), np.full(7, .03), compression=.002)
    assert result['clear'] and result['samples'] == 3
    assert len(wrapper.checker.update_calls) == 3
    assert [call[1] for call in wrapper.checker.update_calls] == [.002, .002, .002]
    with pytest.raises(ValueError, match='joint_step'):
        wrapper.edge(np.zeros(7), np.zeros(7), joint_step=.1)
