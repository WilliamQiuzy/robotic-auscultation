"""Shared conservative static-arm geometry for cuRobo and independent FCL.

Right fingers may move through their full URDF prismatic range. Other rigid
motion is tightly bounded and checked from actual PhysX body poses at runtime.
Neither the frozen source geometry nor its old collision checker is modified.
"""
from __future__ import annotations

import hashlib
from itertools import product
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from sim.auscultation_expert_layout_v1.scene import ROBOT

PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT/'sim/auscultation_expert_layout_v1'
GEOMETRY = SOURCE/'results/expert_v2/plans'
TRANSLATION_TOLERANCE_M = .0002
ROTATION_TOLERANCE_RAD = .001
JOINT_NUMERICAL_TOLERANCE_M = 1e-7


def geometry_sha(vertices, faces):
    digest = hashlib.sha256()
    digest.update(np.asarray(vertices, dtype='<f8').tobytes())
    digest.update(np.asarray(faces, dtype='<i4').tobytes())
    return digest.hexdigest()


def contract_sha(contract):
    return hashlib.sha256(json.dumps(contract, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _convex_mesh(points):
    points = np.asarray(points, float)
    hull = ConvexHull(points)
    triangles = hull.simplices.copy()
    normals = np.cross(points[triangles[:, 1]]-points[triangles[:, 0]],
                       points[triangles[:, 2]]-points[triangles[:, 0]])
    flip = np.sum(normals*hull.equations[:, :3], axis=1) < 0
    triangles[flip] = triangles[flip][:, [0, 2, 1]]
    used = np.unique(triangles)
    return points[used], np.searchsorted(used, triangles).astype(np.int32)


def build_static_world(geometry_dir=GEOMETRY, urdf_path=SOURCE/'openarm_v10_follower_stetho.urdf'):
    """Return world-frame envelope meshes and a portable runtime pose contract."""
    directory = Path(geometry_dir)
    metadata = json.loads((directory/'collision_geometry.json').read_text())
    reference = {e['body']: np.asarray(e['world_from_body'], float) for e in metadata['colliders']
                 if e['body'].startswith(ROBOT+'/') and
                 ('/openarm_right_' in e['body'] or e['body'].endswith(('openarm_body_link0', 'openarm_left_link0')))}
    finger_specs = {}
    for joint in ET.parse(urdf_path).getroot().findall('joint'):
        if not joint.get('name', '').startswith('openarm_right_finger_joint'):
            continue
        if joint.get('type') != 'prismatic':
            raise ValueError('right finger must be prismatic')
        child = ROBOT+'/'+joint.find('child').get('link')
        parent = ROBOT+'/'+joint.find('parent').get('link')
        origin = np.array(joint.find('origin').get('xyz').split(), float)
        rpy = np.array(joint.find('origin').get('rpy', '0 0 0').split(), float)
        axis = np.array(joint.find('axis').get('xyz').split(), float)
        lower, upper = (float(joint.find('limit').get(name)) for name in ('lower', 'upper'))
        if not np.array_equal(rpy, np.zeros(3)) or not np.isclose(np.linalg.norm(axis), 1.) or (lower, upper) != (0., .044):
            raise ValueError('unexpected finger orientation/axis/range; review adapter')
        relative = np.linalg.inv(reference[parent]) @ reference[child]
        q_ref = float((relative[:3, 3]-origin)@axis)
        if (np.linalg.norm(relative[:3, 3]-origin-axis*q_ref) > 1e-6
                or np.linalg.norm(relative[:3, :3]-np.eye(3)) > 1e-6):
            raise ValueError('USD finger frame differs from URDF prismatic model')
        finger_specs[child] = {'parent': parent, 'joint_name': joint.get('name'), 'origin_xyz': origin.tolist(),
                               'axis': axis.tolist(), 'lower_m': lower, 'upper_m': upper,
                               'source_q_m': q_ref}
    if len(finger_specs) != 2:
        raise ValueError('exactly two right finger colliders must be represented')
    geometry, records = {}, {}
    corners = np.array(list(product((-1., 1.), repeat=3)))
    with np.load(directory/'collision_geometry.npz', allow_pickle=False) as arrays:
        for entry in metadata['colliders']:
            body, key = entry['body'], entry['key']
            if body not in reference:
                continue
            vertices = np.asarray(arrays[key+'_v'], float)
            radius = float(np.linalg.norm(vertices, axis=1).max())
            if body in finger_specs:
                spec = finger_specs[body]
                parent = reference[spec['parent']]
                origin, axis = np.array(spec['origin_xyz']), np.array(spec['axis'])
                local = np.concatenate([vertices+origin+spec['lower_m']*axis,
                                        vertices+origin+spec['upper_m']*axis])
                center_world = local @ parent[:3, :3].T + parent[:3, 3]
                # Triangle inequality: parent translation+rotation, child's
                # non-sliding residual translation+rotation, and q roundoff.
                padding = (2*TRANSLATION_TOLERANCE_M + JOINT_NUMERICAL_TOLERANCE_M
                           + 2*np.sin(ROTATION_TOLERANCE_RAD/2)*(float(np.linalg.norm(local, axis=1).max())+radius))
                kind = 'finger_sweep'
            else:
                transform = reference[body]
                center_world = vertices @ transform[:3, :3].T + transform[:3, 3]
                padding = TRANSLATION_TOLERANCE_M + 2*np.sin(ROTATION_TOLERANCE_RAD/2)*radius
                kind = 'bounded_static_body'
            expanded = (center_world[:, None]+padding*corners[None]).reshape(-1, 3)
            world_vertices, faces = _convex_mesh(expanded)
            sha = geometry_sha(world_vertices, faces)
            geometry[key] = {'body': body, 'kind': kind, 'vertices_world': world_vertices, 'faces': faces,
                              'source_vertices_body': vertices, 'padding_m': float(padding), 'sha256': sha}
            records[key] = {'body': body, 'source_path': entry['path'], 'kind': kind, 'padding_m': float(padding),
                            'geometry_sha256': sha, 'vertex_count': len(world_vertices), 'face_count': len(faces)}
    contract = {
        'schema': 'wam_static_world_sweep_v1', 'body_reference_poses': {k: v.tolist() for k, v in reference.items()},
        'finger_specs': finger_specs, 'geometry': records,
        'translation_tolerance_m': TRANSLATION_TOLERANCE_M, 'rotation_tolerance_rad': ROTATION_TOLERANCE_RAD,
        'finger_joint_numerical_tolerance_m': JOINT_NUMERICAL_TOLERANCE_M,
        'proof': 'Convex hull of each source collider translated across both finger range endpoints, plus an axis-aligned cube containing the full bounded displacement ball.',
        'scope': 'Engineering conservative model; actual poses checked every executed physics step; no 10mm static-pose tolerance.',
    }
    return {'contract': contract, 'contract_sha256': contract_sha(contract), 'geometry': geometry}


def save_static_world(directory, bundle):
    """Persist the exact shared meshes, avoiding a runtime hull re-triangulation."""
    root = Path(directory)
    (root/'static_world_contract.json').write_text(json.dumps(bundle['contract'], indent=2, allow_nan=False)+'\n')
    arrays = {}
    for key, value in bundle['geometry'].items():
        arrays[key+'_v'] = value['vertices_world']
        arrays[key+'_f'] = value['faces']
    np.savez_compressed(root/'static_world_geometry.npz', **arrays)


def load_static_world(directory):
    """Load bridge-hash-bound static geometry identically on every host."""
    root = Path(directory)
    contract = json.loads((root/'static_world_contract.json').read_text())
    if contract.get('schema') != 'wam_static_world_sweep_v1' or not isinstance(contract.get('geometry'), dict):
        raise ValueError('static-world contract schema mismatch')
    geometry = {}
    with np.load(root/'static_world_geometry.npz', allow_pickle=False) as arrays:
        if set(arrays.files) != {key+suffix for key in contract['geometry'] for suffix in ('_v', '_f')}:
            raise ValueError('static-world geometry inventory mismatch')
        for key, record in contract['geometry'].items():
            vertices = np.array(arrays[key+'_v'], float)
            faces = np.array(arrays[key+'_f'])
            if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or not np.isfinite(vertices).all()
                    or faces.ndim != 2 or faces.shape[1:] != (3,) or not np.issubdtype(faces.dtype, np.integer)
                    or not len(vertices) or not len(faces) or faces.min() < 0 or faces.max() >= len(vertices)
                    or geometry_sha(vertices, faces) != record['geometry_sha256']):
                raise ValueError('static-world mesh/hash mismatch: '+key)
            geometry[key] = {'body': record['body'], 'kind': record['kind'], 'padding_m': record['padding_m'],
                             'vertices_world': vertices, 'faces': faces, 'sha256': record['geometry_sha256']}
    return {'contract': contract, 'contract_sha256': contract_sha(contract), 'geometry': geometry}


def _matrix(pose):
    value = np.asarray(pose, float)
    if value.shape == (7,):
        if not np.isfinite(value).all() or not np.isclose(np.linalg.norm(value[3:]), 1., atol=1e-5, rtol=0):
            raise ValueError('invalid rigid-body quaternion')
        result = np.eye(4); result[:3, 3] = value[:3]
        result[:3, :3] = Rotation.from_quat(value[[4, 5, 6, 3]]).as_matrix()
        return result
    if (value.shape != (4, 4) or not np.isfinite(value).all()
            or not np.allclose(value[3], [0, 0, 0, 1], atol=1e-9)
            or not np.allclose(value[:3, :3].T@value[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(value[:3, :3]), 1., atol=1e-6)):
        raise ValueError('invalid rigid-body transform')
    return value


def _angle(a, b):
    return float(Rotation.from_matrix(a.T@b).magnitude())


def validate_static_poses(poses, contract):
    violations, matrices = [], {}
    position_max = rotation_max = finger_position_max = finger_rotation_max = 0.
    fingers = {}
    for body, reference_value in contract['body_reference_poses'].items():
        try:
            matrices[body] = _matrix(poses[body])
        except (KeyError, ValueError, TypeError) as error:
            violations.append({'body': body, 'reason': 'missing_or_invalid_pose', 'detail': str(error)})
    for body, expected in contract['body_reference_poses'].items():
        if body in contract['finger_specs'] or body not in matrices:
            continue
        expected, actual = np.asarray(expected), matrices[body]
        distance = float(np.linalg.norm(actual[:3, 3]-expected[:3, 3]))
        angle = _angle(expected[:3, :3], actual[:3, :3])
        position_max, rotation_max = max(position_max, distance), max(rotation_max, angle)
        if distance > contract['translation_tolerance_m']+1e-12 or angle > contract['rotation_tolerance_rad']+1e-12:
            violations.append({'body': body, 'reason': 'static_body_drift', 'translation_m': distance, 'rotation_rad': angle})
    for body, spec in contract['finger_specs'].items():
        if body not in matrices or spec['parent'] not in matrices:
            continue
        relative = np.linalg.inv(matrices[spec['parent']]) @ matrices[body]
        axis = np.asarray(spec['axis']); delta = relative[:3, 3]-spec['origin_xyz']
        q = float(delta@axis)
        distance = float(np.linalg.norm(delta-q*axis)); angle = _angle(np.eye(3), relative[:3, :3])
        fingers[body] = q
        finger_position_max, finger_rotation_max = max(finger_position_max, distance), max(finger_rotation_max, angle)
        tolerance = contract['finger_joint_numerical_tolerance_m']
        if (q < spec['lower_m']-tolerance or q > spec['upper_m']+tolerance
                or distance > contract['translation_tolerance_m']+1e-12 or angle > contract['rotation_tolerance_rad']+1e-12):
            violations.append({'body': body, 'reason': 'finger_outside_swept_contract', 'joint_m': q,
                               'non_sliding_translation_m': distance, 'relative_rotation_rad': angle})
    return {'clear': not violations, 'violations': violations, 'max_static_translation_m': position_max,
            'max_static_rotation_rad': rotation_max, 'max_finger_non_sliding_translation_m': finger_position_max,
            'max_finger_relative_rotation_rad': finger_rotation_max, 'finger_positions_m': fingers}


def install_fcl_static_world(checker, bundle, *, geometry_dir=None):
    """Replace only independent checker's static objects with shared envelopes."""
    import fcl
    exact = checker
    while hasattr(exact, 'checker'):
        exact = exact.checker
    replacements = []
    for index, entry in enumerate(exact.entries):
        key = entry['key']
        if key not in bundle['geometry']:
            continue
        if index in exact.dynamic:
            raise ValueError('static-world replacement must not overwrite moving geometry')
        replacement = bundle['geometry'][key]
        if replacement['body'] != entry['body']:
            raise ValueError('static collider body mismatch')
        vertices, faces = replacement['vertices_world'], replacement['faces']
        model = fcl.BVHModel(); model.beginModel(len(vertices), len(faces))
        model.addSubModel(vertices, faces); model.endModel()
        exact.objects[index] = fcl.CollisionObject(model, fcl.Transform(np.eye(3), np.zeros(3)))
        lower, upper = vertices.min(0), vertices.max(0)
        exact.bounds[index] = [lower, upper]
        exact.local_corners[index] = np.array(list(product(*zip(lower, upper))))
        replacements.append(key)
    if set(replacements) != set(bundle['geometry']):
        raise ValueError('independent FCL omitted required static envelope')
    from .static_containment import install_static_containment
    containment = install_static_containment(exact, bundle, geometry_dir=geometry_dir)
    return {'contract_sha256': bundle['contract_sha256'], 'geometry_sha256': {
        key: bundle['geometry'][key]['sha256'] for key in replacements}, 'colliders_replaced': replacements,
        'containment_gate': containment,
        'scope': 'identical right-arm/finger envelope geometry used by cuRobo and FCL; c002 remains FCL-only, left_link0 retains its native robot sphere cover'}


class StaticWorldMonitor:
    def __init__(self, env, checker, bundle=None, *, geometry_dir=None):
        self.env = env; self.bundle = bundle or build_static_world()
        self.installation = install_fcl_static_world(checker, self.bundle, geometry_dir=geometry_dir)
        self.bodies = {name: body for name, body in env.bodies if name in self.bundle['contract']['body_reference_poses']}
        self.checked_steps = 0
        self.last_report = None
        self.maximum = {key: 0. for key in ('max_static_translation_m', 'max_static_rotation_rad',
                                          'max_finger_non_sliding_translation_m', 'max_finger_relative_rotation_rad')}
        self.check()

    def check(self):
        poses = {}
        for name, body in self.bodies.items():
            position, quaternion = body.get_world_pose()
            poses[name] = np.r_[position, quaternion]
        report = validate_static_poses(poses, self.bundle['contract'])
        self.last_report = report
        for key in self.maximum:
            self.maximum[key] = max(self.maximum[key], report[key])
        if not report['clear']:
            raise ValueError('static world left verified envelope: '+str(report['violations']))
        return report

    def after_step(self):
        self.checked_steps += 1
        try:
            return self.check()
        finally:
            if self.last_report is not None:
                self.env.rows[-1]['static_world_check'] = dict(self.last_report)

    def summary(self):
        return {**self.installation, 'checked_physics_steps': self.checked_steps, 'maxima': self.maximum,
                'last_report': self.last_report, 'contract': self.bundle['contract']}
