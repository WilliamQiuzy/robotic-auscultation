"""Real NumPy/trimesh/Rtree geometry tests; FCL tests run only when installed."""
import importlib
from copy import deepcopy

import numpy as np
import pytest
import trimesh


def module():
    name = 'sim.ausc_policy.static_mesh_queries'
    assert importlib.util.find_spec(name), 'Bounded static mesh queries are missing'
    return importlib.import_module(name)


def box(extents=(1., 1., 1.), center=(0., 0., 0.)):
    mesh = trimesh.creation.box(extents=extents)
    return mesh.vertices.copy()+center, mesh.faces.copy()


def joined(*meshes):
    vertices, faces, offset = [], [], 0
    for v, f in meshes:
        vertices.append(v); faces.append(f+offset); offset += len(v)
    return np.concatenate(vertices), np.concatenate(faces)


def torus():
    # Closed ring with a real concave hole; no convex hull or repair operation.
    n, m = 12, 8
    vertices = []
    for i in range(n):
        u = 2*np.pi*i/n
        for j in range(m):
            v = 2*np.pi*j/m
            radius = 2.+.45*np.cos(v)
            vertices.append([radius*np.cos(u), radius*np.sin(u), .45*np.sin(v)])
    faces = []
    for i in range(n):
        for j in range(m):
            a, b = i*m+j, ((i+1) % n)*m+j
            c, d = i*m+(j+1) % m, ((i+1) % n)*m+(j+1) % m
            faces.extend([[a, b, c], [b, d, c]])
    return np.asarray(vertices), np.asarray(faces)


def pair(a, b, **kwargs):
    return module().query_mesh_pair(*a, *b, distance_backend='numpy', **kwargs)


def test_valid_mesh_reports_all_closed_components_without_mutating_inputs():
    v, f = joined(box(), box(center=(3., 0., 0.)))
    before = (v.copy(), f.copy()); v.setflags(write=False); f.setflags(write=False)
    result = module().validate_mesh(v, f)
    assert result['status'] == 'pass' and result['component_count'] == 2
    assert sorted(len(ids) for ids in result['component_face_ids']) == [12, 12]
    np.testing.assert_array_equal(v, before[0]); np.testing.assert_array_equal(f, before[1])
    assert result['physics_qualified'] is False and result['training_eligible'] is False


@pytest.mark.parametrize('kind', ['open', 'degenerate', 'duplicate', 'nonfinite', 'bad_indices',
                                  'float_faces', 'inconsistent_winding', 'nonmanifold_vertex'])
def test_invalid_mesh_never_passes_validation(kind):
    v, f = box()
    if kind == 'open': f = f[:-1]
    elif kind == 'degenerate': v[1] = v[0]
    elif kind == 'duplicate': f = np.vstack((f, f[0]))
    elif kind == 'nonfinite': v[0, 0] = np.nan
    elif kind == 'bad_indices': f[0, 0] = len(v)
    elif kind == 'float_faces': f = f.astype(float)
    elif kind == 'inconsistent_winding': f[0] = f[0, ::-1]
    else:
        # Two closed shells sharing only one vertex index form a pinched link.
        other_v, other_f = box(center=(1., 1., 1.))
        v, f = joined((v, f), (other_v, other_f))
        duplicate = np.flatnonzero(np.all(v == [.5, .5, .5], axis=1))
        f[f == duplicate[1]] = duplicate[0]
    assert module().validate_mesh(v, f)['status'] == 'fail'


def test_normal_shared_edges_and_vertices_on_a_closed_box_are_allowed():
    result = module().scan_self_intersections(*box())
    assert result['status'] == 'pass' and result['scan_complete']
    assert result['intersecting_pairs'] == []
    assert result['candidate_count'] > 0


def test_sharing_one_vertex_does_not_hide_intersection_elsewhere():
    v = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.],
                  [1., 1., 1.], [1., 1., -1.]])
    result = module().scan_self_intersections(v, np.array([[0, 1, 2], [0, 3, 4]]))
    assert result['status'] == 'fail'
    assert result['intersecting_pairs'] == [[0, 1]]


def test_coplanar_extra_overlap_and_duplicate_faces_fail():
    v = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.], [.8, .8, 0.]])
    result = module().scan_self_intersections(v, np.array([[0, 1, 2], [0, 1, 3]]))
    assert result['intersecting_pairs'] == [[0, 1]]
    v, f = box(); f = np.vstack((f, f[0]))
    assert module().scan_self_intersections(v, f)['status'] == 'fail'


def test_whole_mesh_scan_finds_crossings_between_disconnected_components():
    result = module().scan_self_intersections(*joined(box(), box(center=(.3, .2, .1))))
    assert result['status'] == 'fail' and result['intersecting_pairs']
    assert result['scope'] == 'whole_mesh'


@pytest.mark.parametrize('swap', [False, True])
def test_complete_containment_is_failure_despite_positive_surface_distance(swap):
    outer, inner = box((4., 4., 4.)), box()
    a, b = (inner, outer) if swap else (outer, inner)
    result = pair(a, b)
    assert result['status'] == 'fail'
    assert result['surface_distance_m'] == pytest.approx(1.5)
    assert result['containment']['a_inside_b'] == ([True] if swap else [False])
    assert result['containment']['b_inside_a'] == ([False] if swap else [True])
    assert result['reason'] == 'solid_containment'


def test_concave_hole_is_outside_even_inside_its_convex_hull():
    result = pair(torus(), box((.3, .3, .3)), max_candidates=100000, time_budget_s=30.)
    assert result['status'] == 'pass'
    assert result['surface_distance_m'] > 1.
    assert result['containment']['a_inside_b'] == [False]
    assert result['containment']['b_inside_a'] == [False]


def test_every_disconnected_component_is_checked_in_both_directions():
    a = joined(box((.2, .2, .2)), box((.2, .2, .2), center=(3., 0., 0.)))
    result = pair(a, box((2., 2., 2.)))
    assert result['status'] == 'fail'
    assert result['containment']['a_inside_b'] == [True, False]


def test_surface_crossing_and_contact_fail_and_separated_meshes_pass():
    assert pair(box(), box(center=(.6, .2, .1)))['reason'] == 'surface_intersection'
    assert pair(box(), box(center=(1., 0., 0.)))['status'] == 'fail'
    result = pair(box(), box(center=(2., 0., 0.)))
    assert result['status'] == 'pass'
    assert result['surface_distance_m'] == pytest.approx(1.)
    assert result['distance_backend'] == 'numpy_triangle_pairs'


def test_edge_edge_distance_is_measured_instead_of_vertex_only_approximation():
    # Perpendicular long thin boxes: nearest features are interior edge points.
    a = box((4., .1, .1))
    b = box((.1, 4., .1), center=(0., 0., .3))
    result = pair(a, b)
    assert result['status'] == 'pass'
    assert result['surface_distance_m'] == pytest.approx(.2)


@pytest.mark.parametrize('limit', [dict(max_candidates=0), dict(time_budget_s=1e-12)])
def test_budget_exhaustion_is_unresolved_never_pass(limit):
    result = module().scan_self_intersections(*box(), **limit)
    assert result['status'] == 'unresolved' and not result['scan_complete']
    result = pair(box(), box(center=(2., 0., 0.)), **limit)
    assert result['status'] == 'unresolved'
    assert result['surface_distance_m'] is None


def test_open_or_degenerate_mesh_cannot_be_clear_in_pair_query():
    v, f = box()
    assert pair((v, f[:-1]), box(center=(3., 0., 0.)))['status'] == 'fail'
    v[1] = v[0]
    assert pair((v, f), box(center=(3., 0., 0.)))['status'] == 'fail'


@pytest.mark.skipif(importlib.util.find_spec('fcl') is None, reason='python-fcl is unavailable on this Mac')
def test_actual_fcl_surface_distance_and_containment():
    result = module().query_mesh_pair(*box((4., 4., 4.)), *box(), distance_backend='fcl')
    assert result['distance_backend'] == 'python_fcl'
    assert result['surface_distance_m'] == pytest.approx(1.5)
    assert result['status'] == 'fail' and result['reason'] == 'solid_containment'


def test_forced_unavailable_fcl_is_unresolved(monkeypatch):
    mod = module()
    original = importlib.import_module
    def missing(name, *args, **kwargs):
        if name == 'fcl': raise ImportError('Test unavailable optional backend')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(mod.importlib, 'import_module', missing)
    result = mod.query_mesh_pair(*box(), *box(center=(2., 0., 0.)), distance_backend='fcl')
    assert result['status'] == 'unresolved' and result['surface_distance_m'] is None


def test_pair_queries_leave_readonly_mesh_arrays_unchanged():
    a, b = box(), box(center=(2., 0., 0.))
    copies = deepcopy((a, b))
    for mesh in (a, b):
        for array in mesh: array.setflags(write=False)
    assert pair(a, b)['status'] == 'pass'
    for mesh, before in zip((a, b), copies):
        for array, original in zip(mesh, before): np.testing.assert_array_equal(array, original)


def test_positive_gap_inside_numerical_tolerance_is_unresolved():
    result = pair(box(), box(center=(1.+5e-10, 0., 0.)))
    assert result['status'] == 'unresolved'
    assert 'tolerance' in result['reason']


def test_scan_reports_elapsed_time_and_completed_or_pending_face_coverage():
    complete = module().scan_self_intersections(*box())
    assert complete['elapsed_s'] >= 0.
    assert complete['coverage'] == dict(total_faces=12, tested_faces=12, pending_faces=0)
    partial = module().scan_self_intersections(*box(), max_candidates=1)
    assert partial['status'] == 'unresolved'
    assert partial['coverage']['pending_faces'] > 0
    assert partial['coverage']['tested_faces'] + partial['coverage']['pending_faces'] == 12


def test_distance_budget_reports_pending_pairs_without_a_partial_minimum():
    result = pair(box(), box(center=(2., 0., 0.)), max_candidates=120)
    assert result['status'] == 'unresolved'
    assert result['surface_distance_m'] is None
    distance = result['coverage']['distance']
    assert distance['total_pairs'] == 144 and distance['pending_pairs'] > 0
    assert distance['tested_pairs'] + distance['pending_pairs'] == 144
    assert result['elapsed_s'] >= 0.


@pytest.mark.skipif(importlib.util.find_spec('fcl') is None, reason='python-fcl unavailable')
def test_native_backend_failure_is_unresolved_not_a_clear_or_crashed_query(monkeypatch):
    import fcl
    def failed(*args, **kwargs): raise RuntimeError('Synthetic backend failure')
    monkeypatch.setattr(fcl, 'distance', failed)
    result = module().query_mesh_pair(*box(), *box(center=(2., 0., 0.)), distance_backend='fcl')
    assert result['status'] == 'unresolved' and result['surface_distance_m'] is None


@pytest.mark.parametrize('query', ['scan', 'pair'])
def test_rtree_build_runtime_failure_is_unresolved_in_both_queries(monkeypatch, query):
    def failed(*args, **kwargs): raise RuntimeError('Injected Rtree construction failure')
    monkeypatch.setattr(trimesh.util, 'bounds_tree', failed)
    if query == 'scan':
        result = module().scan_self_intersections(*box())
        assert result['scan_complete'] is False
    else:
        result = pair(box(), box(center=(2., 0., 0.)))
        assert result['surface_distance_m'] is None
    assert result['status'] == 'unresolved'
    assert result['reason'] == 'rtree_build_failed'


@pytest.mark.parametrize('tolerance', [1e-12, 1e-10])
def test_nearparallel_edges_cannot_overestimate_gap_into_a_false_clear(tolerance):
    vertices = np.array([[-1., 0., 0.], [1., 0., 0.], [0., -1., -1.], [0., 1., -1.]])
    angle, gap = 1e-8, 1e-10
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                         [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    reflected = vertices * [1., 1., -1.]
    a = trimesh.convex.convex_hull(vertices)
    b = trimesh.convex.convex_hull(reflected @ rotation.T + [0., 0., gap])
    result = pair((a.vertices, a.faces), (b.vertices, b.faces), tolerance_m=tolerance)
    assert result['surface_distance_m'] == pytest.approx(gap, rel=1e-5, abs=1e-16)
    if tolerance >= gap:
        assert result['status'] == 'unresolved'
    else:
        assert result['status'] == 'pass'


@pytest.mark.parametrize('query', ['scan', 'pair'])
def test_rtree_import_inside_builder_is_unresolved_in_both_queries(monkeypatch, query):
    def missing(*args, **kwargs): raise ImportError('No module named rtree')
    monkeypatch.setattr(trimesh.util, 'bounds_tree', missing)
    result = module().scan_self_intersections(*box()) if query == 'scan' else pair(box(), box(center=(2., 0., 0.)))
    assert result['status'] == 'unresolved'
    assert result['reason'] == 'rtree_build_failed'


@pytest.mark.parametrize('phase', ['scan', 'pair_self', 'pair_surface'])
def test_lazy_rtree_failure_after_yield_is_unresolved(monkeypatch, phase):
    class BrokenTree:
        def intersection(self, *args):
            yield 0
            raise RuntimeError('Lazy Rtree iteration failed')
    original = trimesh.util.bounds_tree
    calls = 0
    def tree(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs) if phase == 'pair_surface' and calls < 3 else BrokenTree()
    monkeypatch.setattr(trimesh.util, 'bounds_tree', tree)
    result = module().scan_self_intersections(*box()) if phase == 'scan' else pair(box(), box(center=(2., 0., 0.)))
    assert result['status'] == 'unresolved'
    assert result['reason'] == 'rtree_query_failed'
    if phase == 'scan': assert result['scan_complete'] is False
    else: assert result['surface_distance_m'] is None


@pytest.mark.parametrize('failure', ['runtime', 'nan', 'shape'])
def test_closest_point_backend_failure_or_invalid_result_is_unresolved(monkeypatch, failure):
    def failed(triangles, points):
        if failure == 'runtime': raise RuntimeError('Triangle backend failed')
        return np.full_like(points, np.nan) if failure == 'nan' else np.zeros((1, 3))
    monkeypatch.setattr(trimesh.triangles, 'closest_point', failed)
    result = pair(box(), box(center=(2., 0., 0.)))
    assert result['status'] == 'unresolved'
    assert result['surface_distance_m'] is None
    assert result['coverage']['distance']['complete'] is False


@pytest.mark.parametrize('query', ['scan', 'pair'])
def test_backend_wrappers_do_not_swallow_internal_geometry_runtime_errors(monkeypatch, query):
    def bug(*args, **kwargs): raise RuntimeError('Internal geometry bug')
    monkeypatch.setattr(module(), '_contacts', bug)
    with pytest.raises(RuntimeError, match='Internal geometry bug'):
        if query == 'scan': module().scan_self_intersections(*box())
        else: pair(box(), box(center=(2., 0., 0.)))


@pytest.mark.parametrize('operation', ['build', 'create_iterator', 'next'])
@pytest.mark.parametrize('phase', ['scan', 'pair_self', 'pair_surface'])
def test_actual_rtree_exception_type_is_unresolved_at_each_backend_boundary(monkeypatch, operation, phase):
    from rtree.exceptions import RTreeError
    class BrokenTree:
        def intersection(self, *args):
            if operation == 'create_iterator': raise RTreeError('Native Rtree iterator creation failed')
            def candidates():
                yield 0
                raise RTreeError('Native Rtree lazy traversal failed')
            return candidates()
    original = trimesh.util.bounds_tree
    calls = 0
    def tree(*args, **kwargs):
        nonlocal calls
        calls += 1
        if phase == 'pair_surface' and calls < 3: return original(*args, **kwargs)
        if operation == 'build': raise RTreeError('Native Rtree construction failed')
        return BrokenTree()
    monkeypatch.setattr(trimesh.util, 'bounds_tree', tree)
    result = module().scan_self_intersections(*box()) if phase == 'scan' else pair(box(), box(center=(2., 0., 0.)))
    assert result['status'] == 'unresolved'
    assert result['backend_exception_type'] == 'RTreeError'
    assert result['reason'] == ('rtree_build_failed' if operation == 'build' else 'rtree_query_failed')
    if phase == 'scan': assert result['scan_complete'] is False
    else: assert result['surface_distance_m'] is None


@pytest.mark.parametrize('exception', [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize('query', ['scan', 'pair'])
def test_backend_boundary_does_not_catch_base_exceptions(monkeypatch, exception, query):
    def interrupted(*args, **kwargs): raise exception('Explicit termination')
    monkeypatch.setattr(trimesh.util, 'bounds_tree', interrupted)
    with pytest.raises(exception):
        if query == 'scan': module().scan_self_intersections(*box())
        else: pair(box(), box(center=(2., 0., 0.)))


@pytest.mark.parametrize('failure', ['value_error', 'custom_exception'])
def test_closest_point_external_exception_is_unresolved_with_type(monkeypatch, failure):
    class ExternalClosestFailure(Exception):
        pass
    exception = ValueError if failure == 'value_error' else ExternalClosestFailure
    def failed(*args, **kwargs): raise exception('External closest-point failure')
    monkeypatch.setattr(trimesh.triangles, 'closest_point', failed)
    result = pair(box(), box(center=(2., 0., 0.)))
    assert result['status'] == 'unresolved'
    assert result['reason'] == 'trimesh_closest_point_failed'
    assert result['backend_exception_type'] == exception.__name__
    assert result['surface_distance_m'] is None
    assert result['coverage']['distance']['complete'] is False


@pytest.mark.parametrize('exception', [KeyboardInterrupt, SystemExit])
def test_closest_point_boundary_does_not_catch_base_exceptions(monkeypatch, exception):
    def interrupted(*args, **kwargs): raise exception('Explicit termination')
    monkeypatch.setattr(trimesh.triangles, 'closest_point', interrupted)
    with pytest.raises(exception):
        pair(box(), box(center=(2., 0., 0.)))
