"""Bounded CPU rays through separate, unchanged native patient mesh layers.

The Linux-only CLI reads an immutable v10 run and writes a new external output
directory. Missing intersections are unknown within the finite ray segment;
neither topology nor sparse hits prove continuous skin or acoustic coupling.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

if __package__ in (None, ''):
    from contact_surface_gap import front_samples, sha, verify_geometry_binding, verify_point_coverage
else:
    from .contact_surface_gap import front_samples, sha, verify_geometry_binding, verify_point_coverage


CARRIAGE = '/World/Stethoscope/Carriage'
LAYER_SUFFIXES = {
    'trans__organic__malecaucasianskin': 'skin',
    'opaque__fabric__scrubsshirt': 'shirt',
    'opaque__fabric__scrubspants': 'pants',
    'opaque__leather__sneakers': 'shoes',
}
RADII_M = (.0115, .021, .023)


def _vector(value, name, *, unit=False):
    result = np.asarray(value, float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError('Finite three-vector required: ' + name)
    if unit and not np.isclose(np.linalg.norm(result), 1., atol=1e-6, rtol=0):
        raise ValueError('Unit direction required: ' + name)
    return result


def _half_length(value):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number))
            or not np.isfinite(value) or value <= 0):
        raise ValueError('Positive finite ray half length required')
    return float(value)


def ray_hits(points, direction, triangles, *, half_length_m=.05):
    """Return signed offset and input triangle row for the first segment hit.

    The segment starts at ``point - half_length * direction`` and travels in
    the unit ``direction``. This chooses the first surface, not the intersection
    nearest the point. Missing rays return NaN and -1. Shared-edge ties use the
    lowest input triangle row. No triangle winding is changed or culled.
    """
    half = _half_length(half_length_m)
    d = _vector(direction, 'ray', unit=True)
    points, tri = np.asarray(points, float), np.asarray(triangles, float)
    if (points.ndim != 2 or points.shape[1:] != (3,) or tri.ndim != 3
            or tri.shape[1:] != (3, 3) or not np.isfinite(points).all() or not np.isfinite(tri).all()):
        raise ValueError('Finite points and triangles required')
    offsets = np.full(len(points), np.nan)
    indices = np.full(len(points), -1, dtype=np.int64)
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    pvec = np.cross(d, e2)
    det = np.einsum('ij,ij->i', e1, pvec)
    valid = np.abs(det) > 1e-12
    inverse = np.zeros_like(det)
    inverse[valid] = 1. / det[valid]
    for i, point in enumerate(points):
        tvec = point - half * d - tri[:, 0]
        u = np.einsum('ij,ij->i', tvec, pvec) * inverse
        qvec = np.cross(tvec, e1)
        v = (qvec @ d) * inverse
        distance = np.einsum('ij,ij->i', e2, qvec) * inverse
        hit = (valid & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
               & (distance >= 0) & (distance <= 2 * half))
        candidates = np.flatnonzero(hit)
        if len(candidates):
            index = int(candidates[np.argmin(distance[candidates])])
            offsets[i], indices[i] = distance[index] - half, index
    return offsets, indices


def circular_patch(center, normal):
    """Anchor, then 32 angles at each fixed radius; stable least-aligned basis."""
    center = _vector(center, 'patch center')
    normal = _vector(normal, 'patch normal', unit=True)
    reference = np.eye(3)[np.argmin(np.abs(normal))]
    tangent = np.cross(normal, reference)
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)
    angles = np.arange(32) * (2 * np.pi / 32)
    directions = np.cos(angles)[:, None] * tangent + np.sin(angles)[:, None] * bitangent
    return np.concatenate([center[None], *(center + radius * directions for radius in RADII_M)])


def physical_transform(value):
    """Validate a proper rigid transform; reject scale, shear and reflection."""
    transform = np.asarray(value, float)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1., atol=1e-6, rtol=0)):
        raise ValueError('Proper physical rigid transform required')
    return transform


def _mesh_arrays(vertices, faces):
    v, f = np.asarray(vertices), np.asarray(faces)
    if (v.dtype.kind not in 'iuf' or v.ndim != 2 or v.shape[1:] != (3,)
            or not np.isfinite(v).all() or f.dtype.kind not in 'iu'
            or f.ndim != 2 or f.shape[1:] != (3,)
            or (f.size and (f.min() < 0 or f.max() >= len(v)))):
        raise ValueError('Malformed native vertices or triangle indices')
    return v.astype(float, copy=False), f.astype(np.int64, copy=False)


def mesh_topology(vertices, faces):
    """Describe indexed geometry without welding, repairing or coverage claims."""
    v, f = _mesh_arrays(vertices, faces)
    edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    _, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    triangles = v[f]
    area2 = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    return dict(vertex_count=len(v), triangle_count=len(f), used_vertex_count=len(np.unique(f)),
                unused_vertex_count=len(v) - len(np.unique(f)), edge_count=len(counts),
                boundary_edge_count=int(np.sum(counts == 1)), nonmanifold_edge_count=int(np.sum(counts > 2)),
                duplicate_triangle_count=len(f) - len(np.unique(np.sort(f, axis=1), axis=0)),
                degenerate_triangle_count=int(np.sum(area2 <= 1e-12)),
                surface_coverage_proven=False, self_intersection_checked=False,
                scope='Indexed mesh counts only; no welding, repair, watertightness or local skin coverage proof')


def patient_meshes(entries, arrays):
    """Load static native patient triangles and classify exact path components."""
    patient = [e for e in entries if e['path'].startswith('/World/Patient/')]
    if not patient or any(e['body'] != 'world' or e['approximation'] != 'none' for e in patient):
        raise ValueError('Recorded static patient triangle colliders required')
    if len({e['path'] for e in patient}) != len(patient) or len({e['key'] for e in patient}) != len(patient):
        raise ValueError('Duplicate native collider path or key')
    colliders = []
    for entry in patient:
        suffix = entry['path'].rsplit('/', 1)[-1]
        if suffix not in LAYER_SUFFIXES:
            raise ValueError('Unclassified native patient collider: ' + entry['path'])
        transform = physical_transform(entry['world_from_body'])
        vertices, faces = _mesh_arrays(arrays[entry['key'] + '_v'], arrays[entry['key'] + '_f'])
        world = vertices @ transform[:3, :3].T + transform[:3, 3]
        triangles = world[faces]
        colliders.append(dict(path=entry['path'], key=entry['key'], layer=LAYER_SUFFIXES[suffix],
                              triangles=triangles, bounds_lo=triangles.min(axis=1), bounds_hi=triangles.max(axis=1),
                              world_from_body=transform.tolist(), topology=mesh_topology(vertices, faces)))
    return colliders


def _nullable(values):
    return [float(x) if np.isfinite(x) else None for x in values]


def _stats(values):
    finite = np.asarray(values)[np.isfinite(values)]
    if not len(finite):
        return None
    return dict(min=float(finite.min()), median=float(np.median(finite)),
                p05=float(np.percentile(finite, 5)), p95=float(np.percentile(finite, 95)), max=float(finite.max()))


def layer_rays(points, direction, colliders, *, half_length_m=.05):
    """Same finite rays independently against each layer and all patient meshes."""
    half = _half_length(half_length_m)
    direction = _vector(direction, 'inward ray', unit=True)
    points = np.asarray(points, float)
    # Validate even when every layer is empty.
    ray_hits(points, direction, np.empty((0, 3, 3)), half_length_m=half)
    if not len(points):
        raise ValueError('At least one patch sample required')
    ends = np.concatenate([points - half * direction, points + half * direction])
    lo, hi = ends.min(axis=0) - 1e-9, ends.max(axis=0) + 1e-9
    selected = []
    for collider in colliders:
        ids = np.flatnonzero(np.all(collider['bounds_hi'] >= lo, axis=1)
                             & np.all(collider['bounds_lo'] <= hi, axis=1))
        selected.append((collider, ids))
    reports, all_offsets = {}, {}
    for layer in ('skin', 'shirt', 'pants', 'shoes', 'all_patient'):
        chunks, lookup = [], []
        for collider, ids in selected:
            if layer == 'all_patient' or collider['layer'] == layer:
                chunks.append(collider['triangles'][ids])
                lookup.extend((collider, int(i)) for i in ids)
        triangles = np.concatenate(chunks) if chunks else np.empty((0, 3, 3))
        offsets, hits = ray_hits(points, direction, triangles, half_length_m=half)
        all_offsets[layer] = offsets
        paths, layers, keys, ids, normals, angles, acute = [], [], [], [], [], [], []
        for hit in hits:
            if hit < 0:
                paths.append(None); layers.append(None); keys.append(None); ids.append(-1)
                normals.append(None); angles.append(None); acute.append(None)
                continue
            collider, triangle_index = lookup[hit]
            triangle = collider['triangles'][triangle_index]
            normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            normal /= np.linalg.norm(normal)
            cosine = float(np.clip(normal @ -direction, -1, 1))
            paths.append(collider['path']); layers.append(collider['layer']); keys.append(collider['key'])
            ids.append(triangle_index); normals.append(normal.tolist())
            angles.append(float(np.degrees(np.arccos(cosine))))
            acute.append(float(np.degrees(np.arccos(abs(cosine)))))
        reports[layer] = dict(hit_count=int(np.isfinite(offsets).sum()), missing_count=int(np.isnan(offsets).sum()),
                              signed_offset_m=_nullable(offsets), signed_offset_stats_m=_stats(offsets),
                              first_hit_collider=paths, first_hit_layer=layers, first_hit_key=keys,
                              triangle_index=ids, triangle_normal_world=normals,
                              geometry_normal_angle_deg=angles, unoriented_normal_angle_deg=acute,
                              geometry_normal_angle_stats_deg=_stats(np.array(angles, float)))
    skin, shirt = all_offsets['skin'], all_offsets['shirt']
    separation = skin - shirt
    return dict(ray_count=len(points), points_world_m=points.tolist(), direction_world=direction.tolist(),
                ray_half_length_m=half, layers=reports,
                paired_skin_shirt_hit_count=int(np.isfinite(separation).sum()),
                shirt_skin_separation_m=_nullable(separation), shirt_skin_separation_stats_m=_stats(separation),
                separation_definition='skin offset minus shirt offset along the same inward ray; positive means shirt hit is outward of skin hit',
                missing_skin_coverage=bool(np.isnan(skin).any()),
                missing_interpretation='No hit within this bounded ray; this does not establish absence of skin globally')


def _middle_hold(case, native):
    data_path = case / 'fresh_expert/fresh_expert_replay.npz'
    count_expected = native['physics_steps']
    if type(count_expected) is not int or count_expected <= 0:
        raise ValueError('Invalid native physics step count')
    with np.load(data_path, allow_pickle=False) as arrays:
        paths = [str(x) for x in arrays['paths']]
        indices, poses = arrays['row_indices'], arrays['poses']
        if (paths.count(CARRIAGE) != 1 or len(set(paths)) != len(paths) or indices.dtype.kind not in 'iu'
                or not np.array_equal(indices, np.arange(7, count_expected, 8))
                or poses.shape != (len(indices), len(paths), 7)):
            raise ValueError('Actual pose sampling ledger differs')
        carriage = poses[:, paths.index(CARRIAGE)].copy()
    if (not np.isfinite(carriage).all()
            or not np.allclose(np.linalg.norm(carriage[:, 3:], axis=1), 1., atol=1e-6, rtol=0)):
        raise ValueError('Invalid native carriage poses')
    wanted = {int(row): i for i, row in enumerate(indices)}
    buckets, digest, count = {}, hashlib.sha256(), 0
    trace = case / 'fresh_expert/fresh_expert_trace.jsonl'
    with trace.open('rb') as stream:
        for i, line in enumerate(stream):
            digest.update(line); count += 1
            row = json.loads(line)
            if row.get('phase') != 'HOLD' or i not in wanted:
                continue
            name = row['target']
            if not np.isfinite(row['time_s']):
                raise ValueError('Nonfinite native HOLD timestamp')
            axis = _vector(row['axis_world'], 'native wrist axis', unit=True)
            buckets.setdefault(name, []).append(dict(row=i, time_s=float(row['time_s']),
                                                       pose=carriage[wanted[i]], axis_world=axis))
    if digest.hexdigest() != native['trace_sha256'] or count != count_expected:
        raise ValueError('Trace SHA/count mismatch')
    verify_point_coverage(buckets, native)
    if any(len(rows) != 90 for rows in buckets.values()):
        raise ValueError('Expected 90 actual 30 Hz HOLD poses per point')
    if any(not np.array_equal(np.diff([r['row'] for r in rows]), np.full(89, 8))
           or np.any(np.diff([r['time_s'] for r in rows]) <= 0) for rows in buckets.values()):
        raise ValueError('Native HOLD samples must be one ordered continuous interval per point')
    return {name: rows[len(rows) // 2] for name, rows in buckets.items()}


def audit(run, output, *, expected_replay_sha256, half_length_m=.05):
    """Audit native files against a replay SHA from prior immutable evidence.

    The caller must supply the expected digest; hashing the current replay and
    treating that value as an expectation would not establish source identity.
    Only a unique Linux CPU output is written.
    """
    started = time.monotonic()
    half = _half_length(half_length_m)
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Existing diagnostic must be retained')
    if sys.platform != 'linux' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Raw layer audit requires Linux CPU with CUDA_VISIBLE_DEVICES empty')
    if output.is_relative_to(run):
        raise ValueError('Diagnostic output must be outside immutable source run')
    if (not isinstance(expected_replay_sha256, str) or len(expected_replay_sha256) != 64
            or not set(expected_replay_sha256) <= set('0123456789abcdef')):
        raise ValueError('Expected replay SHA-256 must be a lowercase 64-character hex digest')
    case, directory = run / 'output/review', run / 'output/review/geometry'
    replay_path = case / 'fresh_expert/fresh_expert_replay.npz'
    if sha(replay_path) != expected_replay_sha256:
        raise ValueError('Native replay SHA differs from externally expected immutable binding')
    native_path = case / 'fresh_expert_result.json'
    native_before = sha(native_path)
    native = json.loads(native_path.read_text())
    verify_geometry_binding(directory, native)
    verify_point_coverage(dict.fromkeys(b['name'] for b in native['plan_bindings']), native)
    source_paths = [native_path, directory / 'collision_geometry.json', directory / 'collision_geometry.npz',
                    case / 'fresh_expert/fresh_expert_trace.jsonl', replay_path]
    plans = []
    for binding in native['plan_bindings']:
        path = Path(binding['file']).resolve()
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != binding['sha256']:
            raise ValueError('Plan SHA changed: ' + str(path))
        plan = json.loads(data)
        if plan['name'] != binding['name'] or plan['geometry_artifacts'] != native['geometry_artifacts']:
            raise ValueError('Plan name or native geometry binding differs')
        _vector(plan['position_world'], 'original anchor')
        _vector(plan['normal_world'], 'original normal', unit=True)
        plans.append(plan); source_paths.append(path)
    before = {str(path): sha(path) for path in source_paths}
    if before[str(replay_path)] != expected_replay_sha256:
        raise ValueError('Native replay SHA changed during source binding')
    if before[str(native_path)] != native_before:
        raise ValueError('Native result changed during source binding')
    # Recheck plan bytes against the hashes just read, closing pre-snapshot races.
    for binding in native['plan_bindings']:
        if before[str(Path(binding['file']).resolve())] != binding['sha256']:
            raise ValueError('Plan SHA changed during source binding')
    entries = json.loads((directory / 'collision_geometry.json').read_text())['colliders']
    ring = sorted([e for e in entries if '/DeviceContactRing/sector_' in e['path']], key=lambda e: e['path'])
    if (len(ring) != 32 or len({e['path'] for e in ring}) != 32
            or any(e['body'] != CARRIAGE for e in ring)):
        raise ValueError('All 32 original carriage ring sectors required')
    with np.load(directory / 'collision_geometry.npz', allow_pickle=False) as arrays:
        colliders = patient_meshes(entries, arrays)
        local = []
        for entry in ring:
            physical_transform(entry['world_from_body'])
            local.extend(front_samples(arrays[entry['key'] + '_v']))
    local = np.asarray(local)
    if not np.allclose(local[:, 2], local[0, 2], atol=1e-10, rtol=0):
        raise ValueError('Native ring front planes differ')
    actual_local_patch = circular_patch([0, 0, local[0, 2]], [0, 0, 1])
    middle = _middle_hold(case, native)
    points = []
    for plan in plans:
        name, normal = plan['name'], np.asarray(plan['normal_world'], float)
        anchor = circular_patch(plan['position_world'], normal)
        row = middle[name]
        pose = row['pose']
        rotation = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
        direction = -rotation[:, 2]
        actual = actual_local_patch @ rotation.T + pose[:3]
        front = local @ rotation.T + pose[:3]
        points.append(dict(target=name, position_world=plan['position_world'], normal_world=plan['normal_world'],
                           anchor_patch=layer_rays(anchor, -normal, colliders, half_length_m=half),
                           native_middle_hold=dict(row=row['row'], time_s=row['time_s'],
                               pose_world_xyz_wxyz=pose.tolist(), native_axis_world=row['axis_world'].tolist(),
                               wrist_carriage_axis_angle_deg=float(np.degrees(np.arccos(np.clip(direction @ row['axis_world'], -1, 1)))),
                               circular_patch=layer_rays(actual, direction, colliders, half_length_m=half),
                               retainer_front=layer_rays(front, direction, colliders, half_length_m=half))))
    verify_geometry_binding(directory, native)
    after = {str(path): sha(path) for path in source_paths}
    if before != after:
        raise ValueError('Source files changed during layer audit')
    report = dict(schema='native_patient_contact_layer_audit_v1', run_id=run.name, source_run=str(run),
                  points=points, plan_bindings=native['plan_bindings'], geometry_artifacts=native['geometry_artifacts'],
                  source_sha256_before=before, source_sha256_after=after, source_sha256=sha(__file__),
                  expected_replay_sha256=expected_replay_sha256,
                  contact_surface_gap_source_sha256=sha(Path(__file__).with_name('contact_surface_gap.py')),
                  colliders=[{k: c[k] for k in ('path', 'key', 'layer', 'world_from_body', 'topology')} for c in colliders],
                  ray_half_length_m=half, patch_radii_m=list(RADII_M), patch_angle_count=32,
                  patch_order='center; increasing radius, each with angles 0..31 times 2pi/32',
                  native_middle_selection='index 45 of 90 saved HOLD carriage poses at 30 Hz, matched by physics trace row',
                  signed_offset_definition='First intersection from point-half_length*inward_direction; signed relative to point',
                  normal_definition='Raw triangle winding normal versus negative ray direction; unoriented angle also reported, no flipping or averaging',
                  qualification=dict(physics_qualified=False, flatness_qualified=False, continuous_skin_coverage=False,
                                     acoustic_coupling_qualified=False, thin_cloth_qualified=False,
                                     training_eligible=False, clinical_qualified=False),
                  nominal_scrubs_status='Unmodified nominal geometry only; thickness, fit, mechanics and acoustics are unqualified',
                  physics_executed=False, scene_modified=False, wall_s=time.monotonic() - started,
                  scope='Finite rigid geometry samples only. Missing rays are unknown within the search interval, never zero or global absence. No plane fit, skin deformation, cloth model or acoustic claim.')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(points=len(points), output=str(output), wall_s=report['wall_s'])), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--expected-replay-sha256', required=True,
                        help='Replay SHA-256 from prior immutable evidence, not the current file')
    parser.add_argument('--ray-half-length-m', type=float, default=.05)
    args = parser.parse_args()
    audit(args.run, args.output, expected_replay_sha256=args.expected_replay_sha256,
          half_length_m=args.ray_half_length_m)
