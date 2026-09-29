"""Reject dynamic source vertices embedded in the new static convex envelopes.

This supplements the unchanged triangle-distance policy. It does not claim a
continuous collision or general solid-containment proof for the old geometry.
"""
from functools import wraps
import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

GEOMETRY = Path(__file__).resolve().parents[1]/'auscultation_expert_layout_v1/results/expert_v2/plans'


def install_static_containment(checker, bundle, geometry_dir=None):
    """Wrap the exact checker in place after installing its static envelopes."""
    exact = checker
    while hasattr(exact, 'checker'):
        exact = exact.checker
    if getattr(exact, '_static_containment_installed', False):
        raise ValueError('static containment gate already installed')
    dynamic = set(exact.dynamic)
    hulls = {}
    for index, entry in enumerate(exact.entries):
        if entry['key'] not in bundle['geometry']:
            continue
        replacement = bundle['geometry'][entry['key']]
        if index in dynamic or replacement['body'] != entry['body']:
            raise ValueError('static containment geometry/body mismatch')
        vertices = np.asarray(replacement['vertices_world'], float)
        if not np.isfinite(vertices).all():
            raise ValueError('nonfinite static envelope')
        hulls[index] = ConvexHull(vertices).equations
    # Keep the original pair order and allowed-pair policy. No new exemptions.
    pairs = [(i, j) for i, j in exact.pairs if i in dynamic and j in hulls]
    source = Path(geometry_dir or GEOMETRY)/'collision_geometry.npz'
    vertices_by_index = {}
    with np.load(source, allow_pickle=False) as arrays:
        for index in {i for i, _ in pairs}:
            vertices = np.asarray(arrays[exact.entries[index]['key']+'_v'], float)
            if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
                raise ValueError('invalid source vertices for containment')
            vertices_by_index[index] = vertices
    pair_array = np.asarray(pairs, dtype=int).reshape(-1, 2)
    original_check = exact.check

    @wraps(original_check)
    def check(*args, **kwargs):
        report = original_check(*args, **kwargs)
        if not report['clear'] or not len(pair_array):
            return report
        a, b = pair_array.T
        bounds = exact.bounds
        if not np.isfinite(bounds[pair_array]).all():
            raise ValueError('nonfinite containment broadphase bounds')
        # Zero-distance AABB overlap implies FastCollision synchronized each
        # relevant object's native transform during the original near-pair pass.
        overlap = np.all((bounds[a, 0] <= bounds[b, 1]+1e-10)
                         & (bounds[b, 0] <= bounds[a, 1]+1e-10), axis=1)
        world_vertices = {}
        for i, j in pair_array[overlap]:
            if i not in world_vertices:
                obj = exact.objects[i]
                rotation, translation = np.asarray(obj.getRotation()), np.asarray(obj.getTranslation())
                world_vertices[i] = vertices_by_index[i]@rotation.T + translation
            equations = hulls[j]
            inside = np.all(world_vertices[i]@equations[:, :3].T+equations[:, 3] <= 1e-10, axis=1)
            if np.any(inside):
                return {**report, 'clear':False, 'minimum_distance_m':0.,
                        'pair':[exact.entries[i]['path'], exact.entries[j]['path']],
                        'reason':'inside_static_swept_envelope'}
        return report

    exact.check = check
    exact._static_containment_installed = True
    return {'method':'dynamic_source_vertex_inside_static_convex_envelope',
            'checked_original_pairs':len(pairs), 'static_envelope_count':len(hulls),
            'source_npz_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'boundary_tolerance_m':1e-10,
            'scope':'supplemental containment rejection for new static envelopes only; original surface checks remain'}
