"""Pure height/XY proposals for the nine frozen 32-sided stool cylinders.

This is deliberately not generalized CAD: seat/legs have world-Z axes, and
rung_x/rung_y have world-X/world-Y axes. The supplied mesh, including its caps
and faceting, is preserved. No asset I/O, scene installation or qualification
is performed. External artifact hashes are syntax-checked; canonical content
hashes separately bind the actual arrays and metadata supplied to this call.
"""
from collections import Counter
from copy import deepcopy
import re

import numpy as np

from . import cpu_scene_variant as bundle
from .geometry_epoch import require_current


GEOMETRY_TOL_M = 1e-7
STOOL_PATHS = bundle.STOOL_PATHS


def _scalar(value, name):
    array = bundle._real_array(value, name)
    if array.shape != ():
        raise ValueError(name + ' must be a finite scalar')
    return float(array)


def _bounds(vertices):
    return np.array([vertices.min(0), vertices.max(0)])


def _cylinder(world, faces, axis, path):
    """Validate two matching regular 32-point rings, cap centers and topology."""
    error = 'Unsupported axis-aligned closed 32-sided cylinder: ' + path
    if world.shape != (66, 3) or faces.shape != (128, 3) or not np.isfinite(world).all():
        raise ValueError(error)
    bounds = _bounds(world)
    if np.any(bounds[1] - bounds[0] <= 2*GEOMETRY_TOL_M):
        raise ValueError(error)
    cross = [i for i in range(3) if i != axis]
    at_low = np.abs(world[:, axis]-bounds[0, axis]) <= GEOMETRY_TOL_M
    at_high = np.abs(world[:, axis]-bounds[1, axis]) <= GEOMETRY_TOL_M
    if at_low.sum() != 33 or at_high.sum() != 33 or not (at_low | at_high).all():
        raise ValueError(error)
    centers, rings = [], []
    center = world[:, cross].mean(0)
    distances = np.linalg.norm(world[:, cross]-center, axis=1)
    for mask in (at_low, at_high):
        cap_center = np.flatnonzero(mask & (distances <= GEOMETRY_TOL_M))
        ring = np.flatnonzero(mask & (distances > GEOMETRY_TOL_M))
        if len(cap_center) != 1 or len(ring) != 32:
            raise ValueError(error)
        angles = np.arctan2(world[ring, cross[1]]-center[1], world[ring, cross[0]]-center[0])
        order = np.argsort(angles)
        gaps = np.diff(np.r_[angles[order], angles[order][0]+2*np.pi])
        radius = float(distances[ring].mean())
        if radius <= 2*GEOMETRY_TOL_M or not np.allclose(
                gaps, 2*np.pi/32, rtol=0, atol=4*GEOMETRY_TOL_M/radius):
            raise ValueError(error)
        centers.append(int(cap_center[0])); rings.append(ring[order])
    radius = float(distances[np.r_[rings[0], rings[1]]].mean())
    if np.max(np.abs(distances[np.r_[rings[0], rings[1]]]-radius)) > GEOMETRY_TOL_M:
        raise ValueError(error)
    # Match by actual profile coordinates, avoiding the -pi/pi branch cut.
    delta = world[rings[0]][:, None, cross] - world[rings[1]][None, :, cross]
    pair_distances = np.linalg.norm(delta, axis=2)
    matches = pair_distances.argmin(1)
    if len(set(matches.tolist())) != 32 or pair_distances[np.arange(32), matches].max() > GEOMETRY_TOL_M:
        raise ValueError(error)
    rings[1] = rings[1][matches]
    slots = {int(vertex): i for ring in rings for i, vertex in enumerate(ring)}
    cap_sets = [set(ring.tolist()) for ring in rings]
    cap_edges = [Counter(), Counter()]
    side_counts, directed = Counter(), Counter()
    for triangle in faces:
        indices = [int(i) for i in triangle]
        for a, b in zip(indices, indices[1:]+indices[:1]):
            directed[(a, b)] += 1
        cap = next((i for i, vertex in enumerate(centers) if vertex in indices), None)
        if cap is not None:
            edge = [i for i in indices if i != centers[cap]]
            if len(edge) != 2 or not set(edge) <= cap_sets[cap]:
                raise ValueError(error)
            angular = sorted(slots[i] for i in edge)
            cap_edges[cap][tuple(angular)] += 1
        else:
            if not all(i in slots for i in indices):
                raise ValueError(error)
            angular = sorted({slots[i] for i in indices})
            if len(angular) != 2 or all(i in cap_sets[0] for i in indices) or all(i in cap_sets[1] for i in indices):
                raise ValueError(error)
            side_counts[tuple(angular)] += 1
    edges = {tuple(sorted((i, (i+1) % 32))) for i in range(32)}
    if (any(set(cap) != edges or any(n != 1 for n in cap.values()) for cap in cap_edges)
            or set(side_counts) != edges or any(n != 2 for n in side_counts.values())
            or any(n != 1 or directed[(b, a)] != 1 for (a, b), n in directed.items())):
        raise ValueError(error)
    return bounds, radius


def build_stool_geometry(metadata, arrays, *, new_seat_top_z_m,
                         stool_xy_translation_m, floor_z_m, source_bindings):
    """Return independent copies containing exact proposed meshes and bindings.

    Source legs must span the explicit floor to seat underside within 1e-7 m.
    Their world-Z endpoints follow the new floor/underside interval; XY and
    every face index remain unchanged. Rungs translate to retain their center
    fraction of the leg height, preserving their complete original profile.
    All nine parts receive the same world XY translation. Stored rigid frames
    remain unchanged; new vertices are solved back into those local frames.

    Original height and zero XY are a numerical identity within 1e-7 m (and
    preserve vertex bytes when the requested top equals the computed top).
    Non-stool entries, arrays, allowed pairs, planes and camera data are copied
    unchanged. This cannot establish floor contact, support, IK or native QA.
    """
    if not isinstance(source_bindings, dict):
        raise ValueError('Explicit source bindings required')
    require_current(source_bindings)
    for name in ('collision_geometry_sha256', 'collision_arrays_sha256'):
        if not isinstance(source_bindings.get(name), str) or not re.fullmatch('[0-9a-f]{64}', source_bindings[name]):
            raise ValueError('Missing or malformed source SHA-256: ' + name)
    bundle._json_bytes(source_bindings)
    transforms = bundle._validate_bundle(metadata, arrays)
    if any(entry['path'] == '/World/Stool' for entry in metadata['colliders']):
        raise ValueError('Expected exactly the nine named stool colliders')
    top = _scalar(new_seat_top_z_m, 'new_seat_top_z_m')
    floor = _scalar(floor_z_m, 'floor_z_m')
    xy = bundle._real_array(stool_xy_translation_m, 'stool_xy_translation_m').astype(float)
    if xy.shape != (2,):
        raise ValueError('World XY translation must have two finite components')
    sources = {}
    for entry in metadata['colliders']:
        key, path = entry['key'], entry['path']
        local = np.asarray(arrays[key+'_v'])
        if not np.allclose(entry['bounds_local'], _bounds(local), rtol=0, atol=GEOMETRY_TOL_M):
            raise ValueError('Source bounds_local contradict mesh: ' + path)
        if path not in STOOL_PATHS:
            continue
        matrix = transforms[key]
        world = local @ matrix[:3, :3].T + matrix[:3, 3]
        axis = 0 if '/rung_x' in path else 1 if '/rung_y' in path else 2
        bounds, radius = _cylinder(world, np.asarray(arrays[key+'_f']), axis, path)
        sources[path] = dict(world=world, bounds=bounds, radius=radius, axis=axis)
    seat = sources['/World/Stool/seat']['bounds']
    thickness = float(seat[1, 2]-seat[0, 2])
    old_length, new_length = float(seat[0, 2]-floor), top-thickness-floor
    if min(old_length, new_length) <= 2*GEOMETRY_TOL_M:
        raise ValueError('Seat underside must leave positive leg length above floor')
    for path, source in sources.items():
        bounds = source['bounds']
        if '/leg' in path and not np.allclose(bounds[:, 2], [floor, seat[0, 2]], rtol=0, atol=GEOMETRY_TOL_M):
            raise ValueError('Source legs must meet the fixed floor and seat underside')
        if '/rung_' in path and (bounds[0, 2] <= floor+GEOMETRY_TOL_M or bounds[1, 2] >= seat[0, 2]-GEOMETRY_TOL_M):
            raise ValueError('Source rungs must be fully above floor and below seat')

    result_metadata = deepcopy(metadata)
    result_arrays = {key: np.array(value, copy=True) for key, value in arrays.items()}
    report = dict(source_seat_top_z_m=float(seat[1, 2]), new_seat_top_z_m=top,
        floor_z_m=floor, stool_xy_translation_m=xy.tolist(), seat_thickness_m=thickness,
        source_leg_length_m=old_length, proposal_leg_length_m=new_length,
        geometry_tolerance_m=GEOMETRY_TOL_M, parts={})
    identity = top == seat[1, 2] and np.array_equal(xy, [0., 0.])
    for entry in result_metadata['colliders']:
        key, path = entry['key'], entry['path']
        if path not in sources:
            continue
        source = sources[path]
        world = source['world'].copy()
        fraction = None
        if '/rung_' in path:
            center = float(source['bounds'][:, 2].mean())
            fraction = (center-floor)/old_length
        if not identity:
            world[:, :2] += xy
            if path.endswith('/seat'):
                world[:, 2] += top-seat[1, 2]
            elif '/leg' in path:
                # Every cylinder vertex lies on a cap; fix both intended end
                # planes exactly rather than amplifying source roundoff.
                upper = world[:, 2] > source['bounds'][:, 2].mean()
                world[:, 2] = np.where(upper, floor+new_length, floor)
            else:
                world[:, 2] += floor+fraction*new_length-center
            matrix = transforms[key]
            local = np.linalg.solve(matrix[:3, :3], (world-matrix[:3, 3]).T).T
            result_arrays[key+'_v'] = local
        local = result_arrays[key+'_v']
        matrix = transforms[key]
        actual_world = local @ matrix[:3, :3].T + matrix[:3, 3]
        bounds, radius = _cylinder(actual_world, result_arrays[key+'_f'], source['axis'], path)
        if '/rung_' in path and (bounds[0, 2] <= floor+GEOMETRY_TOL_M or bounds[1, 2] >= floor+new_length-GEOMETRY_TOL_M):
            raise ValueError('Proposed height cannot fit rung thickness above floor and below seat')
        entry['bounds_local'] = _bounds(local).tolist()
        report['parts'][path] = dict(source_bounds_world_m=source['bounds'].tolist(),
            proposal_bounds_world_m=bounds.tolist(), axis_world='xyz'[source['axis']],
            radius_m=radius, center_height_fraction=fraction)
    return dict(schema_version=1, kind='stool_geometry_proposal', metadata=result_metadata,
        arrays=result_arrays, source_bindings=deepcopy(source_bindings), generation_report=report,
        content_bindings=dict(canonical_format='finite-json-utf8-sorted-v1; array-json-dtype-shape-nul-le-c-v1',
            source=bundle._bundle_bindings(metadata, arrays),
            proposal=bundle._bundle_bindings(result_metadata, result_arrays)),
        stool_height_parameterized=True,
        pending_gates=['checker_rebuild', 'static_scene_qa', 'native_installation_acceptance',
                       'support', 'ik', 'physics', 'training_qualification'],
        **{name: False for name in bundle._CLAIMS})
