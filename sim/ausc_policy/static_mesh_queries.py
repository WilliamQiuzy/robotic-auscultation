"""Bounded numerical queries on unmodified world-space triangle meshes.

These are finite-precision engineering screens, not exact CAD predicates or
physics/native/training qualification. Closed connected components are treated
as a union of filled shells (not a convention for nested cavity shells).
No welding, hole filling, winding repair or convex-hull replacement is used.
Time budgets are cooperative: atomic NumPy/Rtree/FCL calls cannot be interrupted,
but an over-budget result is never returned as pass.
"""
from collections import Counter, defaultdict
import importlib
import math
import time

import numpy as np


def _report(status, reason, **fields):
    return dict(status=status, reason=reason, physics_qualified=False,
                native_scene_installed=False, training_eligible=False, **fields)


class _Unresolved(Exception):
    def __init__(self, reason, backend_exception_type=None):
        super().__init__(reason)
        self.backend_exception_type = backend_exception_type


class _Budget:
    def __init__(self, max_candidates, time_budget_s):
        if (type(max_candidates) is not int or max_candidates < 0 or
                isinstance(time_budget_s, bool) or not isinstance(time_budget_s, (int, float)) or
                not math.isfinite(time_budget_s) or time_budget_s <= 0):
            raise ValueError('Nonnegative candidate and positive finite time budgets required')
        self.maximum, self.seconds = max_candidates, time_budget_s
        self.started, self.used = time.monotonic(), 0

    def take(self, count=0):
        if time.monotonic()-self.started >= self.seconds:
            raise _Unresolved('time_budget_exhausted')
        if self.used+count > self.maximum:
            raise _Unresolved('candidate_budget_exhausted')
        self.used += count


def _finish(budget, status, reason, **fields):
    return _report(status, reason, candidate_count=budget.used,
                   elapsed_s=time.monotonic()-budget.started, **fields)


def _coverage(total):
    return dict(total_faces=total, tested_faces=0, pending_faces=total)


def _arrays(vertices, faces, tolerance_m):
    if (isinstance(tolerance_m, bool) or not isinstance(tolerance_m, (int, float)) or
            not math.isfinite(tolerance_m) or tolerance_m <= 0):
        raise ValueError('Positive finite tolerance_m required')
    v, f = np.asarray(vertices), np.asarray(faces)
    if (v.dtype.kind not in 'iuf' or v.ndim != 2 or v.shape[1:] != (3,) or len(v) < 3 or
            not np.isfinite(v).all() or f.dtype.kind not in 'iu' or
            f.ndim != 2 or f.shape[1:] != (3,) or not len(f) or f.min() < 0 or f.max() >= len(v)):
        raise ValueError('Malformed/nonfinite triangle mesh')
    v, f = v.astype(float, copy=True), f.astype(np.int64, copy=True)
    triangles = v[f]
    area2 = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1)
    if not np.isfinite(area2).all() or np.any(area2 <= tolerance_m**2):
        raise ValueError('Degenerate or numerically unrepresentable triangle')
    return v, f


def _topology(v, f):
    edges, links = defaultdict(list), defaultdict(lambda: defaultdict(set))
    for index, face in enumerate(f):
        for a, b, c in (face, np.roll(face, -1), np.roll(face, -2)):
            edges[tuple(sorted((int(a), int(b))))].append((index, int(a), int(b)))
            links[int(a)][int(b)].add(int(c)); links[int(a)][int(c)].add(int(b))
    issues = []
    if len(np.unique(np.sort(f, axis=1), axis=0)) != len(f): issues.append('duplicate_faces')
    if len(np.unique(f)) != len(v): issues.append('unused_vertices')
    if any(len(rows) == 1 for rows in edges.values()): issues.append('open_boundary')
    if any(len(rows) > 2 for rows in edges.values()): issues.append('nonmanifold_edge')
    if any(len(rows) == 2 and rows[0][1:] != rows[1][1:][::-1] for rows in edges.values()):
        issues.append('inconsistent_winding')
    adjacent = [set() for _ in f]
    for rows in edges.values():
        ids = [row[0] for row in rows]
        for other in ids[1:]:
            adjacent[ids[0]].add(other); adjacent[other].add(ids[0])
    components, unseen = [], set(range(len(f)))
    while unseen:
        todo, component = [min(unseen)], []
        while todo:
            face = todo.pop()
            if face not in unseen: continue
            unseen.remove(face); component.append(face); todo.extend(adjacent[face] & unseen)
        components.append(sorted(component))
    for neighbors in links.values():
        seen, todo = set(), [next(iter(neighbors))]
        while todo:
            vertex = todo.pop()
            if vertex in seen: continue
            seen.add(vertex); todo.extend(neighbors[vertex]-seen)
        if len(seen) != len(neighbors) or any(len(row) != 2 for row in neighbors.values()):
            issues.append('nonmanifold_vertex_link'); break
    return _report('fail' if issues else 'pass', 'invalid_topology' if issues else 'closed_oriented_manifold',
        issues=issues, vertex_count=len(v), face_count=len(f), component_count=len(components),
        component_face_ids=components, self_intersection_checked=False)


def validate_mesh(vertices, faces, *, tolerance_m=1e-9):
    """Validate finite nondegenerate, closed oriented edge/vertex topology only.

    A topology pass does not imply an embedded or non-self-intersecting solid.
    Face components are returned explicitly, including disconnected components.
    """
    try:
        v, f = _arrays(vertices, faces, tolerance_m)
        return _topology(v, f)
    except (ValueError, TypeError, OverflowError) as exc:
        return _report('fail', 'invalid_mesh', issues=[str(exc)], component_face_ids=[],
                       component_count=0, self_intersection_checked=False)


def _inside(point, triangle, normal, tol):
    for a, b in zip(triangle, np.roll(triangle, -1, axis=0)):
        if np.dot(np.cross(b-a, point-a), normal) < -tol*np.linalg.norm(b-a): return False
    return True


def _contacts(a, b, tol):
    """Intersection polygon/segment points, including coplanar extra overlap."""
    na = np.cross(a[1]-a[0], a[2]-a[0]); na /= np.linalg.norm(na)
    nb = np.cross(b[1]-b[0], b[2]-b[0]); nb /= np.linalg.norm(nb)
    da, db = (a-b[0]) @ nb, (b-a[0]) @ na
    if any(np.all(d > tol) or np.all(d < -tol) for d in (da, db)): return []
    if np.linalg.norm(np.cross(na, nb)) <= 1e-10 and np.max(np.abs(da)) <= tol:
        polygon = [p.copy() for p in a]
        for start, end in zip(b, np.roll(b, -1, axis=0)):
            output = []
            if not polygon: break
            for p, q in zip(polygon, polygon[1:]+polygon[:1]):
                dp, dq = np.dot(np.cross(end-start, p-start), nb), np.dot(np.cross(end-start, q-start), nb)
                inside_p, inside_q = dp >= -tol*np.linalg.norm(end-start), dq >= -tol*np.linalg.norm(end-start)
                if inside_p: output.append(p)
                if inside_p != inside_q and dp != dq: output.append(p+(q-p)*dp/(dp-dq))
            polygon = output
        return polygon
    points = []
    for source, target, distances, normal in ((a, b, da, nb), (b, a, db, na)):
        for i in range(3):
            j = (i+1) % 3
            if abs(distances[i]) <= tol and _inside(source[i], target, normal, tol): points.append(source[i])
            if distances[i]*distances[j] < 0:
                p = source[i]+(source[j]-source[i])*distances[i]/(distances[i]-distances[j])
                if _inside(p, target, normal, tol): points.append(p)
    return points


def _point_segment(point, a, b):
    direction = b-a
    t = np.clip(np.dot(point-a, direction)/np.dot(direction, direction), 0., 1.)
    return float(np.linalg.norm(point-a-t*direction))


def _beyond_shared(points, shared, v, tol):
    if not points: return False
    if len(shared) == 1: return any(np.linalg.norm(p-v[shared[0]]) > tol for p in points)
    if len(shared) == 2: return any(_point_segment(p, *v[shared]) > tol for p in points)
    return True


def _tree(triangles, tol, budget):
    budget.take()
    try:
        trimesh = importlib.import_module('trimesh')
    except (ImportError, OSError) as exc:
        raise _Unresolved('rtree_or_trimesh_unavailable') from exc
    bounds = np.stack((triangles.min(1)-tol, triangles.max(1)+tol), axis=1)
    try:
        tree = trimesh.util.bounds_tree(bounds)
    except Exception as exc:
        # RTreeError is a direct Exception subclass. This boundary contains
        # only the external construction call, not our bounds computation.
        raise _Unresolved('rtree_build_failed', type(exc).__name__) from exc
    budget.take()
    return tree


def _candidates(tree, bounds):
    """Catch backend creation/next failures without wrapping caller geometry."""
    try:
        candidates = iter(tree.intersection(bounds))
    except Exception as exc:
        raise _Unresolved('rtree_query_failed', type(exc).__name__) from exc
    while True:
        try:
            candidate = next(candidates)
        except StopIteration:
            return
        except Exception as exc:
            raise _Unresolved('rtree_query_failed', type(exc).__name__) from exc
        yield candidate


def _scan(v, f, tol, budget, coverage):
    triangles = v[f]; tree = _tree(triangles, tol, budget)
    for i, a in enumerate(triangles):
        budget.take()
        for j in _candidates(tree, np.r_[a.min(0)-tol, a.max(0)+tol]):
            if j <= i: continue
            budget.take(1)
            shared = np.intersect1d(f[i], f[j])
            if _beyond_shared(_contacts(a, triangles[j], tol), shared, v, tol):
                if not _beyond_shared(_contacts(a, triangles[j], 0.), shared, v, tol):
                    raise _Unresolved('self_contact_within_tolerance')
                return [[i, int(j)]]
        coverage.update(tested_faces=i+1, pending_faces=len(f)-i-1)
    budget.take()
    return []


def scan_self_intersections(vertices, faces, *, tolerance_m=1e-9, max_candidates=100000, time_budget_s=10.):
    """Scan all faces, exempting only actual shared vertices/edges, not pairs.

    Invalid topology remains fail even if this numerical intersection scan is
    clear. A found crossing may stop early; pass requires complete coverage.
    """
    budget = _Budget(max_candidates, time_budget_s)
    base = dict(scope='whole_mesh', scan_complete=False, intersecting_pairs=[], coverage=None)
    try:
        v, f = _arrays(vertices, faces, tolerance_m); topology = _topology(v, f)
        base.update(coverage=_coverage(len(f)), topology=topology)
        hits = _scan(v, f, tolerance_m, budget, base['coverage'])
        base.update(intersecting_pairs=hits, scan_complete=not bool(hits), topology=topology)
        return _finish(budget, 'fail' if hits or topology['status'] == 'fail' else 'pass',
            'self_intersection' if hits else topology['reason'], **base)
    except _Unresolved as exc:
        return _finish(budget, 'unresolved', str(exc), backend_exception_type=exc.backend_exception_type, **base)
    except (ValueError, TypeError, OverflowError) as exc:
        return _finish(budget, 'fail', 'invalid_mesh', detail=str(exc), **base)


def _segment_distance(a, b, c, d):
    value = min(_point_segment(a, c, d), _point_segment(b, c, d),
                _point_segment(c, a, b), _point_segment(d, a, b))
    u, v, w = b-a, d-c, a-c
    normal = np.cross(u, v)
    denominator = np.dot(normal, normal)
    if denominator > 0:
        # The equivalent Gram determinant |u|²|v|²-(u.v)² loses the
        # entire angle term for nearly parallel edges. Triple products keep
        # that term; the normal projection avoids cancelling closest points.
        s = np.dot(np.cross(v, w), normal)/denominator
        t = np.dot(np.cross(u, w), normal)/denominator
        if 0 <= s <= 1 and 0 <= t <= 1:
            value = min(value, float(abs(np.dot(w, normal))/np.sqrt(denominator)))
    return value


def _closest_distances(trimesh, triangles, points):
    try:
        closest = trimesh.triangles.closest_point(triangles, points)
    except Exception as exc:
        raise _Unresolved('trimesh_closest_point_failed', type(exc).__name__) from exc
    try:
        closest = np.asarray(closest)
    except (TypeError, ValueError) as exc:
        raise _Unresolved('invalid_trimesh_closest_points') from exc
    if (closest.shape != points.shape or closest.dtype.kind not in 'iuf'
            or closest.dtype.itemsize > 8 or not np.isfinite(closest).all()):
        raise _Unresolved('invalid_trimesh_closest_points')
    with np.errstate(over='ignore', invalid='ignore'):
        distances = np.linalg.norm(closest-points, axis=1)
    if not np.isfinite(distances).all():
        raise _Unresolved('nonfinite_trimesh_closest_distances')
    return distances


def _distance(va, fa, vb, fb, backend, budget, coverage):
    if backend not in ('auto', 'numpy', 'fcl'): raise ValueError('Unknown distance_backend')
    if backend != 'numpy':
        try:
            fcl = importlib.import_module('fcl')
        except (ImportError, OSError) as exc:
            if backend == 'fcl': raise _Unresolved('python_fcl_unavailable') from exc
        else:
            coverage.update(method='native_bvh_distance', total_pairs=None, tested_pairs=None,
                            pending_pairs=None, complete=False)
            try:
                objects = []
                for vertices, faces in ((va, fa), (vb, fb)):
                    model = fcl.BVHModel(); model.beginModel(len(vertices), len(faces))
                    model.addSubModel(vertices, faces); model.endModel()
                    objects.append(fcl.CollisionObject(model))
                    budget.take()
                value = float(fcl.distance(*objects, fcl.DistanceRequest(), fcl.DistanceResult()))
            except (RuntimeError, ValueError, OSError) as exc:
                raise _Unresolved('python_fcl_query_failed') from exc
            budget.take()
            if not math.isfinite(value) or value < 0: raise _Unresolved('invalid_fcl_distance')
            coverage['complete'] = True
            return value, 'python_fcl'
    trimesh = importlib.import_module('trimesh')
    total = len(fa)*len(fb)
    coverage.update(method='exhaustive_triangle_distance', total_pairs=total, tested_pairs=0,
                    pending_pairs=total, complete=False)
    best = math.inf
    for a in va[fa]:
        for b in vb[fb]:
            budget.take(1)
            # Vertex/face and edge/edge features both matter for mesh distance.
            da = _closest_distances(trimesh, np.tile(b, (3, 1, 1)), a)
            db = _closest_distances(trimesh, np.tile(a, (3, 1, 1)), b)
            best = min(best, float(da.min()), float(db.min()))
            for i in range(3):
                for j in range(3): best = min(best, _segment_distance(a[i], a[(i+1) % 3], b[j], b[(j+1) % 3]))
            coverage['tested_pairs'] += 1
            coverage['pending_pairs'] -= 1
    budget.take()
    if not math.isfinite(best): raise _Unresolved('nonfinite_surface_distance')
    coverage['complete'] = True
    return best, 'numpy_triangle_pairs'


def _inside_component(point, triangles, budget):
    angle = 0.
    for start in range(0, len(triangles), 1024):
        rows = triangles[start:start+1024]-point
        budget.take(len(rows))
        a, b, c = rows[:, 0], rows[:, 1], rows[:, 2]
        la, lb, lc = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1), np.linalg.norm(c, axis=1)
        numerator = np.einsum('ij,ij->i', a, np.cross(b, c))
        denominator = la*lb*lc + np.einsum('ij,ij->i', a, b)*lc + np.einsum('ij,ij->i', b, c)*la + np.einsum('ij,ij->i', c, a)*lb
        angle += float(np.sum(2*np.arctan2(numerator, denominator)))
    winding = abs(angle/(4*np.pi))
    if abs(winding-1) <= 1e-6: return True
    if winding <= 1e-6: return False
    raise _Unresolved('ambiguous_solid_angle_winding')


def _containment(va, fa, ca, vb, fb, cb, budget, coverage):
    result = []
    for component in ca:
        point = va[fa[component[0], 0]]
        inside = False
        for target in cb:
            inside = _inside_component(point, vb[fb[target]], budget) or inside
        result.append(inside)
        coverage['tested_component_directions'] += 1
        coverage['pending_component_directions'] -= 1
    return result


def query_mesh_pair(vertices_a, faces_a, vertices_b, faces_b, *, tolerance_m=1e-9,
                    max_candidates=100000, time_budget_s=10., distance_backend='auto'):
    """Full numerical pair screen: topology, self/cross contact, distance, containment.

    All connected shells are checked in both directions after complete surface
    intersection scans. Each shell's representative point suffices only under
    these closed, embedded, disjoint-surface preconditions. NumPy distance is
    exhaustive triangle-feature distance, bounded by the shared query budget;
    auto prefers lazy python-fcl when installed. Candidate counts include
    triangle tests and containment triangle evaluations. No safety clearance
    threshold is inferred beyond the explicit numerical tolerance.
    """
    budget = _Budget(max_candidates, time_budget_s)
    fields = dict(surface_distance_m=None, distance_backend=None, containment=None, coverage={})
    try:
        va, fa = _arrays(vertices_a, faces_a, tolerance_m)
        vb, fb = _arrays(vertices_b, faces_b, tolerance_m)
        ta, tb = _topology(va, fa), _topology(vb, fb)
        fields['topology'] = [ta, tb]
        if any(t['status'] != 'pass' for t in (ta, tb)):
            return _finish(budget, 'fail', 'invalid_topology', **fields)
        fields['coverage'] = dict(self_a=_coverage(len(fa)), self_b=_coverage(len(fb)),
            surface=_coverage(len(fa)), distance=dict(complete=False), containment=dict(complete=False))
        for label, v, f in (('self_a', va, fa), ('self_b', vb, fb)):
            if _scan(v, f, tolerance_m, budget, fields['coverage'][label]):
                return _finish(budget, 'fail', 'self_intersection', **fields)
        triangles_b = vb[fb]; tree = _tree(triangles_b, tolerance_m, budget)
        for i, a in enumerate(va[fa]):
            budget.take()
            for j in _candidates(tree, np.r_[a.min(0)-tolerance_m, a.max(0)+tolerance_m]):
                budget.take(1)
                if _contacts(a, triangles_b[j], tolerance_m):
                    if not _contacts(a, triangles_b[j], 0.):
                        raise _Unresolved('surface_contact_within_tolerance')
                    fields['surface_distance_m'] = 0.
                    return _finish(budget, 'fail', 'surface_intersection', **fields)
            fields['coverage']['surface'].update(tested_faces=i+1, pending_faces=len(fa)-i-1)
        distance, backend = _distance(va, fa, vb, fb, distance_backend, budget, fields['coverage']['distance'])
        fields.update(surface_distance_m=distance, distance_backend=backend)
        if distance <= tolerance_m:
            raise _Unresolved('surface_contact_within_tolerance')
        ca, cb = ta['component_face_ids'], tb['component_face_ids']
        coverage = fields['coverage']['containment']
        coverage.update(total_component_directions=len(ca)+len(cb), tested_component_directions=0,
                        pending_component_directions=len(ca)+len(cb))
        a_in_b = _containment(va, fa, ca, vb, fb, cb, budget, coverage)
        b_in_a = _containment(vb, fb, cb, va, fa, ca, budget, coverage)
        fields['containment'] = dict(a_inside_b=a_in_b, b_inside_a=b_in_a,
            method='component_union_solid_angle_after_full_surface_checks')
        budget.take()
        coverage['complete'] = True
        contained = any(a_in_b) or any(b_in_a)
        return _finish(budget, 'fail' if contained else 'pass', 'solid_containment' if contained else 'disjoint_solids', **fields)
    except _Unresolved as exc:
        return _finish(budget, 'unresolved', str(exc), backend_exception_type=exc.backend_exception_type, **fields)
    except (ImportError, ModuleNotFoundError) as exc:
        return _finish(budget, 'unresolved', 'backend_unavailable', detail=str(exc), **fields)
    except (ValueError, TypeError, OverflowError) as exc:
        return _finish(budget, 'fail', 'invalid_mesh_or_query', detail=str(exc), **fields)
