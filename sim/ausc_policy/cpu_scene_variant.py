"""Build a CPU collision-bundle proposal entirely in memory.

This does not construct FCL, solve IK, install USD, or establish support. The
caller supplies an already placed observed mesh. Source artifact SHA-256s are
syntax-checked provenance, not verified file hashes; separate canonical hashes
bind the actual in-memory inputs and outputs. Rebuild the checker and perform
static QA and native installation acceptance before using any proposal.
"""
from copy import deepcopy
import hashlib
import json
import re

import numpy as np

from .geometry_epoch import require_current
from .surface_anchors import evaluate_anchors


STOOL_PATHS = tuple('/World/Stool/' + name for name in
                    ('seat', 'leg0', 'leg1', 'leg2', 'leg3',
                     'rung_x0', 'rung_x1', 'rung_y0', 'rung_y1'))
PATIENT_PATH = '/World/Patient/SMPLSurface'
TARGET_NAMES = tuple(side + str(i) for side in 'LR' for i in range(1, 6))
SOURCE_HASH_FIELDS = ('collision_geometry_sha256', 'collision_arrays_sha256',
                      'observed_placement_sha256', 'anchor_inventory_sha256')
_CLAIMS = ('static_support_qualified', 'ik_qualified', 'physics_qualified',
           'training_eligible', 'source_artifact_hashes_verified',
           'stool_height_tested', 'native_scene_installed')


def _json_bytes(value):
    """Canonical finite JSON: sorted string keys, UTF-8, no extra whitespace."""
    def check(item):
        if isinstance(item, dict):
            if not all(isinstance(key, str) for key in item):
                raise ValueError('Canonical JSON requires string keys')
            for child in item.values():
                check(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                check(child)
    check(value)
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'),
                          ensure_ascii=False, allow_nan=False).encode('utf-8')
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Content must be finite JSON data') from exc


def canonical_entry_sha256(entry):
    """Hash finite JSON content; dictionary insertion order is immaterial."""
    if not isinstance(entry, dict):
        raise ValueError('Entry must be a dictionary')
    return hashlib.sha256(_json_bytes(entry)).hexdigest()


def canonical_array_sha256(value):
    """Hash dtype/shape and little-endian C-order bytes, independent of strides.

    The hashed stream is canonical JSON {dtype, shape}, a NUL byte, then array
    bytes. Dtype and shape are significant; byte order and memory order are not.
    This is an array-content hash, deliberately distinct from an NPZ file hash.
    """
    array = np.asarray(value)
    if (array.dtype.kind not in 'biuf' or array.dtype.itemsize > 8
            or not np.isfinite(array).all()):
        raise ValueError('Array hash requires finite real numeric content')
    dtype = array.dtype.newbyteorder('<')
    canonical = np.ascontiguousarray(array, dtype=dtype)
    description = {'dtype': dtype.str, 'shape': list(array.shape)}
    digest = hashlib.sha256(_json_bytes(description) + b'\0')
    digest.update(canonical.tobytes(order='C'))
    return digest.hexdigest()


def _real_array(value, label):
    array = np.asarray(value)
    if (array.dtype.kind not in 'iuf' or array.dtype.itemsize > 8
            or not np.isfinite(array).all()):
        raise ValueError(label + ' must contain finite real numbers')
    return array


def _mesh(vertices, faces, label):
    vertices = _real_array(vertices, label + ' vertices')
    faces = np.asarray(faces)
    if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 3
            or faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces)
            or faces.dtype.kind not in 'iu' or faces.dtype.itemsize > 8
            or faces.min() < 0 or faces.max() >= len(vertices)
            or np.any(faces[:, 0] == faces[:, 1])
            or np.any(faces[:, 1] == faces[:, 2])
            or np.any(faces[:, 0] == faces[:, 2])):
        raise ValueError('Invalid ' + label + ' mesh topology')
    return vertices, faces


def _transform(value):
    matrix = _real_array(value, 'world_from_body').astype(np.float64)
    if (matrix.shape != (4, 4)
            or not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-8)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3],
                               np.eye(3), rtol=0, atol=1e-7)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, rtol=0, atol=1e-7)):
        raise ValueError('world_from_body must be a proper rigid transform')
    return matrix


def _patient_path(path):
    return path == '/World/Patient' or path.startswith('/World/Patient/')


def _validate_bundle(metadata, arrays):
    if not isinstance(metadata, dict) or not isinstance(arrays, dict):
        raise ValueError('Collision metadata and arrays must be dictionaries')
    _json_bytes(metadata)
    if any(name in metadata for name in ('patient_receipt', 'non_patient_scene_unchanged')):
        raise ValueError('Native patient receipt cannot be carried into a CPU proposal')
    if any(metadata.get(name, False) is not False for name in _CLAIMS):
        raise ValueError('Source qualification cannot qualify a changed CPU proposal')
    entries = metadata.get('colliders')
    if not isinstance(entries, list) or not entries:
        raise ValueError('Missing colliders')
    for name in ('allowed_body_pairs', 'analytic_planes', 'unsupported'):
        if not isinstance(metadata.get(name), list):
            raise ValueError('Missing collision metadata list: ' + name)
    keys, paths, expected_arrays = set(), set(), set()
    transforms = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('Invalid collider entry')
        key, path, body = (entry.get(name) for name in ('key', 'path', 'body'))
        if (not isinstance(key, str) or not re.fullmatch(r'[A-Za-z0-9_]+', key)
                or not isinstance(path, str) or not path.startswith('/World/')
                or not isinstance(body, str) or not body
                or key in keys or path in paths):
            raise ValueError('Invalid or duplicate collider key/path/body')
        if 'approximation' not in entry:
            raise ValueError('Missing collider approximation')
        keys.add(key)
        paths.add(path)
        transforms[key] = _transform(entry.get('world_from_body'))
        bounds = _real_array(entry.get('bounds_local'), 'bounds_local')
        if bounds.shape != (2, 3) or np.any(bounds[0] > bounds[1]):
            raise ValueError('Invalid bounds_local')
        vkey, fkey = key + '_v', key + '_f'
        if vkey not in arrays or fkey not in arrays:
            raise ValueError('Missing collider array pair: ' + key)
        _mesh(arrays[vkey], arrays[fkey], path)
        expected_arrays.update((vkey, fkey))
    if set(arrays) != expected_arrays:
        raise ValueError('Orphan or unrecognized collision arrays')
    stool_paths = {path for path in paths if path.startswith('/World/Stool/')}
    if stool_paths != set(STOOL_PATHS):
        raise ValueError('Expected exactly the nine named stool colliders')
    return transforms


def _bundle_bindings(metadata, arrays):
    return {
        'metadata_sha256': canonical_entry_sha256(metadata),
        'entry_sha256': {entry['key']: canonical_entry_sha256(entry)
                         for entry in metadata['colliders']},
        'array_sha256': {key: canonical_array_sha256(arrays[key]) for key in sorted(arrays)},
    }


def build_cpu_scene_variant(metadata, arrays, patient_vertices_world, patient_faces,
                            anchors, *, stool_xy_translation_m, source_bindings):
    """Replace patient meshes and translate all nine stool parts in world XY.

    Input collision arrays use ``key_v``/``key_f`` with vertices in the entry's
    body frame. The supplied patient vertices are already in world metres and
    are quantized to float32 *before* anchor evaluation. Stool body transforms
    are preserved; a world XY translation is expressed in each local frame.
    Stool height is unchanged, not tested. All returned data owns its memory.
    """
    if not isinstance(source_bindings, dict):
        raise ValueError('Missing source bindings')
    require_current(source_bindings)
    for name in SOURCE_HASH_FIELDS:
        if not isinstance(source_bindings.get(name), str) or not re.fullmatch(
                r'[0-9a-f]{64}', source_bindings[name]):
            raise ValueError('Missing or malformed SHA-256 source hash: ' + name)
    _json_bytes(source_bindings)
    transforms = _validate_bundle(metadata, arrays)
    xy = _real_array(stool_xy_translation_m, 'stool_xy_translation_m').astype(np.float64)
    if xy.shape != (2,):
        raise ValueError('stool_xy_translation_m must have exactly two components')
    delta = np.array([xy[0], xy[1], 0.], dtype=np.float64)
    vertices, faces = _mesh(patient_vertices_world, patient_faces, 'patient')
    with np.errstate(over='ignore', invalid='ignore'):
        patient_vertices = vertices.astype('<f4', copy=True)
    if not np.isfinite(patient_vertices).all():
        raise ValueError('Patient vertices overflow float32')
    patient_faces_copy = faces.copy()
    if not isinstance(anchors, dict) or set(anchors) != set(TARGET_NAMES):
        raise ValueError('Expected exactly ten V6A anchors L1-L5/R1-R5')
    _json_bytes(anchors)
    try:
        evaluated = evaluate_anchors(patient_vertices, patient_faces_copy, anchors)
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError('Malformed V6A anchors') from exc
    # An unambiguous reserved key cannot overwrite a preserved collider.
    retained = [entry for entry in metadata['colliders'] if not _patient_path(entry['path'])]
    patient_key = 'cpu_patient_smpl'
    if any(entry['key'] == patient_key for entry in retained):
        raise ValueError('Reserved patient key collides with a retained collider')
    result_metadata = deepcopy(metadata)
    result_metadata['colliders'] = []
    result_arrays = {}
    for original in retained:
        entry = deepcopy(original)
        key = entry['key']
        local_vertices = np.array(arrays[key + '_v'], copy=True)
        if entry['path'] in STOOL_PATHS:
            # Solve using the stored matrix rather than its transpose: rounded
            # source rotations need not be exactly orthogonal at bit precision.
            local_delta = np.linalg.solve(transforms[key][:3, :3], delta)
            with np.errstate(over='ignore', invalid='ignore'):
                local_vertices = local_vertices.astype(np.float64) + local_delta
            if not np.isfinite(local_vertices).all():
                raise ValueError('Translated stool vertices are nonfinite')
            entry['bounds_local'] = [local_vertices.min(0).tolist(), local_vertices.max(0).tolist()]
        result_arrays[key + '_v'] = local_vertices
        result_arrays[key + '_f'] = np.array(arrays[key + '_f'], copy=True)
        result_metadata['colliders'].append(entry)
    result_arrays[patient_key + '_v'] = patient_vertices
    result_arrays[patient_key + '_f'] = patient_faces_copy
    result_metadata['colliders'].append({
        'key': patient_key, 'path': PATIENT_PATH, 'body': 'world',
        'world_from_body': np.eye(4).tolist(), 'approximation': 'none',
        'bounds_local': [patient_vertices.min(0).tolist(), patient_vertices.max(0).tolist()],
    })
    targets = [{'name': name, 'position_world_m': evaluated[name]['position_m'],
                'normal_world': evaluated[name]['normal']} for name in TARGET_NAMES]
    return {
        'schema_version': 1, 'kind': 'cpu_collision_bundle_proposal',
        'metadata': result_metadata, 'arrays': result_arrays, 'targets': targets,
        'source_bindings': deepcopy(source_bindings),
        'content_bindings': {
            'canonical_format': 'finite-json-utf8-sorted-v1; array-json-dtype-shape-nul-le-c-v1',
            'source': _bundle_bindings(metadata, arrays),
            'proposal': _bundle_bindings(result_metadata, result_arrays),
            'patient_input_vertices_sha256': canonical_array_sha256(vertices),
            'patient_input_faces_sha256': canonical_array_sha256(faces),
            'anchors_sha256': canonical_entry_sha256(anchors),
            'targets_sha256': hashlib.sha256(_json_bytes(targets)).hexdigest(),
        },
        'stool_translation_world_m': delta.tolist(),
        'changed_stool_paths': list(STOOL_PATHS),
        'pending_gates': ['checker_rebuild', 'static_scene_qa', 'native_installation_acceptance',
                          'support', 'ik', 'physics', 'training_qualification'],
        **{name: False for name in _CLAIMS},
    }
