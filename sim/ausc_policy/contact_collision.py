"""Fresh-geometry clearance policy with membrane and device-retainer contact.

Uses the original FastCollisionChecker FK, collider inventory, allowed body
pairs and conservative broad phase. Only its overly broad patient-contact
exemption is replaced. Static sweep envelopes and their containment gate are
installed for every instance. This is a sampled geometric gate, not a physical
contact or continuous swept-motion certificate.

Native FCL is imported only when a checker is constructed. Threshold and fake
distance-oracle regressions therefore run on the development Mac without FCL.
"""
from pathlib import Path
import json

import numpy as np

from .geometry_epoch import require_current, plane_lower_clearance
from .retainer_contact import CONTACT_PATHS


GENERAL_CLEARANCE_M = .003
SHELL_CLEARANCE_M = .0003
_COLLIDER_ROOT = '/World/Stethoscope/Carriage/Colliders/'
_FACE = _COLLIDER_ROOT + 'Diaphragm'
_SHELLS = frozenset(_COLLIDER_ROOT + name for name in
                    ('DeviceEdge', 'DeviceCone', 'DeviceDial'))
_DEFAULT_URDF = (Path(__file__).resolve().parents[1]
                 / 'auscultation_expert_layout_v1/openarm_v10_follower_stetho.urdf')


def _query(q, clearance, allow_patient_contact, compression):
    q = np.asarray(q, float)
    if (q.shape != (7,) or not np.isfinite(q).all()
            or not np.isscalar(clearance) or not np.isscalar(compression)
            or not np.isfinite([clearance, compression]).all()
            or clearance <= 0. or not 0. <= compression <= .006
            or not isinstance(allow_patient_contact, (bool, np.bool_))):
        raise ValueError('invalid collision query')
    return q, max(GENERAL_CLEARANCE_M, float(clearance)), float(compression)


def pair_clearance(path_a, path_b, clearance=.003, allow_patient_contact=False):
    """Return separation, or None for a permitted patient contact interface.

The shell's 0.3 mm contact threshold applies only to its three named parts
against the patient. Contact mode never changes table/robot/tool thresholds.
Calling with a small clearance cannot lower the 3 mm general safety floor.
"""
    if (not np.isscalar(clearance) or not np.isfinite(clearance) or clearance <= 0.
            or not isinstance(allow_patient_contact, (bool, np.bool_))
            or not isinstance(path_a, str) or not isinstance(path_b, str)):
        raise ValueError('invalid pair clearance query')
    if allow_patient_contact:
        for component, patient in ((path_a, path_b), (path_b, path_a)):
            if patient.startswith('/World/Patient/'):
                if component in CONTACT_PATHS:
                    return None
                if component in _SHELLS:
                    return SHELL_CLEARANCE_M
    return max(GENERAL_CLEARANCE_M, float(clearance))


def _rigid_matrices(matrices, label):
    matrices = np.asarray(matrices, float)
    if (matrices.ndim != 3 or matrices.shape[1:] != (4, 4)
            or not np.isfinite(matrices).all()
            or not np.allclose(matrices[:, 3, :], [0, 0, 0, 1], rtol=0., atol=1e-8)
            or not np.allclose(matrices[:, :3, :3].transpose(0, 2, 1)
                               @ matrices[:, :3, :3], np.eye(3), rtol=0., atol=1e-7)
            or not np.allclose(np.linalg.det(matrices[:, :3, :3]), 1., rtol=0., atol=1e-7)):
        raise ValueError('invalid ' + label + ' rigid transform')


class _ContactPolicyMixin:
    """New check algorithm; FastCollisionChecker supplies FK/bounds/native sync."""
    def _validate_native(self, pairs):
        indices = np.unique(pairs)
        matrices = np.repeat(np.eye(4)[None], len(indices), axis=0)
        for slot, index in enumerate(indices):
            rotation = np.asarray(self.objects[index].getRotation(), float)
            translation = np.asarray(self.objects[index].getTranslation(), float)
            if rotation.shape != (3, 3) or translation.shape != (3,):
                raise ValueError('invalid native collision transform shape')
            matrices[slot, :3, :3] = rotation
            matrices[slot, :3, 3] = translation
        _rigid_matrices(matrices, 'native collision')

    def check(self, q, clearance=.003, allow_patient_contact=False, compression=0., details=False):
        q, general, compression = _query(q, clearance, allow_patient_contact, compression)
        self._update_bounds(q, compression)
        if (self.bounds.shape != (len(self.entries), 2, 3)
                or not np.isfinite(self.bounds).all()
                or np.any(self.bounds[:, 0] > self.bounds[:, 1])):
            raise ValueError('invalid collision bounds')
        _rigid_matrices(self._body_transforms, 'dynamic FK')
        ia, ib = self._pair_a, self._pair_b
        gap = np.maximum(0., np.maximum(self.bounds[ia, 0]-self.bounds[ib, 1],
                                       self.bounds[ib, 0]-self.bounds[ia, 1]))
        # Test every inherited pair at the GENERAL floor. In particular a caller
        # requesting shell clearance must not skip a lip 2 mm from the patient.
        broad_distance = np.linalg.norm(gap, axis=1)
        if not np.isfinite(gap).all() or not np.isfinite(broad_distance).all():
            raise ValueError('nonfinite collision broadphase distance')
        near = broad_distance < general + 1e-9
        near_pairs = self.pair_indices[near]
        self.last_checked_aabb_pair_count = len(self.pairs)
        self.last_near_pair_count = len(near_pairs)
        self.last_exact_distance_calls = 0
        self._sync_near_objects(near_pairs)
        self._validate_native(near_pairs)
        worst = {'clear': True, 'minimum_distance_m': general, 'pair': None,
                 'general_clearance_m': general, 'violations': []}
        for plane in self.analytic_planes:
            distance = plane_lower_clearance(self.bounds[self.dynamic], plane['normal'], plane['point'])
            if not np.isfinite(distance):
                raise ValueError('nonfinite analytic plane distance')
            pair = ['dynamic bounds', plane['path']]
            if distance < worst['minimum_distance_m']:
                worst.update(minimum_distance_m=distance, pair=pair)
            if distance < general:
                worst['clear'] = False
                worst['reason'] = 'analytic_plane_clearance'
                worst['violations'].append(dict(pair=pair, distance_m=distance, required_m=general))
                if not details:
                    return worst
        # Original pair order retained; no triangle/collider removal or cached
        # previous configuration's distance. Only the defined B/C contact
        # interface is exempt, never the printed lip or backing shell.
        for i, j in near_pairs:
            a, b = self.entries[i]['path'], self.entries[j]['path']
            required = pair_clearance(a, b, general, allow_patient_contact)
            if required is None:
                continue
            distance = float(self._native_distance(int(i), int(j)))
            self.last_exact_distance_calls += 1
            if not np.isfinite(distance):
                raise ValueError('nonfinite native FCL distance')
            if distance < worst['minimum_distance_m']:
                worst.update(minimum_distance_m=distance, pair=[a, b])
            if distance < required:
                worst['clear'] = False
                worst['violations'].append(dict(pair=[a, b], distance_m=distance, required_m=required))
                if not details:
                    return worst
        return worst


def _validate_geometry(directory):
    metadata = json.loads((directory / 'collision_geometry.json').read_text())
    entries = metadata.get('colliders', [])
    planes = metadata.get('analytic_planes')
    if not entries or metadata.get('unsupported'):
        raise ValueError('missing or unsupported collision geometry')
    if not isinstance(planes, list) or not planes:
        raise ValueError('analytic ground plane is required')
    for plane in planes:
        if not isinstance(plane.get('path'), str):
            raise ValueError('analytic plane path is required')
        plane_lower_clearance(np.zeros((1, 2, 3)), plane['normal'], plane['point'])
    _rigid_matrices([entry['world_from_body'] for entry in entries], 'source collider')
    keys = [entry['key'] for entry in entries]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate collision geometry key')
    with np.load(directory / 'collision_geometry.npz', allow_pickle=False) as arrays:
        for entry in entries:
            if not isinstance(entry.get('path'), str) or not isinstance(entry.get('body'), str):
                raise ValueError('collision geometry path/body is required')
            vertices = np.asarray(arrays[entry['key'] + '_v'])
            faces = np.asarray(arrays[entry['key'] + '_f'])
            if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices)
                    or not np.isfinite(vertices).all() or faces.ndim != 2
                    or faces.shape[1:] != (3,) or not len(faces)
                    or not np.issubdtype(faces.dtype, np.integer)
                    or faces.min() < 0 or faces.max() >= len(vertices)):
                raise ValueError('invalid source collision mesh: ' + entry['key'])
    return metadata


class ContactCollisionChecker:
    """Strict 5 mm geometry checker, including current static sweep/containment.

    metadata must bind the active epoch AND SHA256 of both geometry artifacts.
    No historical default geometry directory is used. The supplied kinematics
    must be the caller's current-geometry kinematics; FK is rerun per query.
    """
    def __init__(self, kinematics, world_from_base, directory, metadata, urdf_path=None):
        require_current(metadata)
        from .geometry_epoch import require_geometry_artifacts
        directory = Path(directory)
        self.geometry_artifacts = require_geometry_artifacts(directory, metadata)
        source = _validate_geometry(directory)
        base = np.asarray(world_from_base, float)
        _rigid_matrices(base[None], 'robot base')
        import fcl
        from sim.curobo_expert_v1.fast_collision import FastCollisionChecker
        from sim.curobo_expert_v1.static_world import build_static_world, install_fcl_static_world

        class NativeChecker(_ContactPolicyMixin, FastCollisionChecker):
            def _native_distance(self, i, j):
                return fcl.distance(self.objects[i], self.objects[j], fcl.DistanceRequest(), fcl.DistanceResult())

        self.checker = NativeChecker(kinematics, base, directory=directory)
        self.checker.analytic_planes = source['analytic_planes']
        self.static_bundle = build_static_world(geometry_dir=directory,
            urdf_path=Path(urdf_path) if urdf_path is not None else _DEFAULT_URDF)
        self.static_installation = install_fcl_static_world(self.checker, self.static_bundle,
                                                          geometry_dir=directory)

    def __getattr__(self, name):
        return getattr(self.checker, name)

    def check(self, *args, **kwargs):
        return self.checker.check(*args, **kwargs)

    def edge(self, start, end, clearance=.003, allow_patient_contact=False,
             joint_step=.015, compression=0.):
        """Check both endpoints and every <= joint_step sample, at given load."""
        start, _, compression = _query(start, clearance, allow_patient_contact, compression)
        end, _, _ = _query(end, clearance, allow_patient_contact, compression)
        if not np.isscalar(joint_step) or not np.isfinite(joint_step) or not 0. < joint_step <= .015:
            raise ValueError('joint_step must be in (0, 0.015] rad')
        n = max(1, int(np.ceil(np.max(np.abs(end-start)) / joint_step)))
        minimum = float('inf')
        for u in np.linspace(0., 1., n+1):
            report = self.check(start + (end-start)*u, clearance, allow_patient_contact, compression)
            minimum = min(minimum, report['minimum_distance_m'])
            if not report['clear']:
                return {**report, 'fraction': float(u), 'samples': n+1}
        return {'clear': True, 'minimum_distance_m': minimum, 'samples': n+1}
