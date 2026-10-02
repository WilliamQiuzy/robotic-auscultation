"""Bounded height-envelope diagnostic for fixed, unmodified surface samples.

The linear program removes affine tilt from a fixed local sample footprint.
It is not finite-head fitting: a rotated disk samples different locations,
may hit different triangles, and must still pass FiniteHeadFit and robot QA.
Residual span is measured along the supplied reference normal, not the fitted
normal. No acoustic, area, compliance, motion or training claim follows.
"""
import numpy as np
from scipy.optimize import linprog
from .contact_surface_gap import ray_offsets


def offset_surface_proxy(vertices, faces, offset_m):
    """Parallel-surface sensitivity proxy; never a calibrated clothing model.

    Whole source topology is retained for this offline geometric comparison.
    This is not a garment pattern or a native simulation installation. Global
    self-intersections are not removed; local folds must be screened separately.
    """
    v = np.asarray(vertices, float); f = np.asarray(faces)
    if (v.ndim != 2 or v.shape[1:] != (3,) or not np.isfinite(v).all()
            or f.ndim != 2 or f.shape[1:] != (3,) or len(f) == 0
            or f.dtype.kind not in 'iu' or f.min() < 0 or f.max() >= len(v)):
        raise ValueError('Finite indexed triangles required')
    if (isinstance(offset_m, (bool, np.bool_)) or not np.isscalar(offset_m)
            or not np.isfinite(offset_m) or not 0 < offset_m <= .002):
        raise ValueError('Explicit sensitivity offset in (0, 2 mm] required')
    from collections import defaultdict
    edges = defaultdict(list)
    for face in f:
        for a, b in zip(face, np.roll(face, -1)):
            edges[tuple(sorted((int(a), int(b))))].append((int(a), int(b)))
    if any(len(e) != 2 or e[0] != e[1][::-1] for e in edges.values()):
        raise ValueError('Closed consistent winding required for offset proxy')
    tri = v[f]; area_vectors = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    if np.any(np.linalg.norm(area_vectors, axis=1) < 1e-14):
        raise ValueError('Degenerate source triangles')
    centered = tri - v.mean(axis=0)
    volume = np.sum(np.einsum('ij,ij->i', centered[:, 0],
                    np.cross(centered[:, 1], centered[:, 2])))/6
    if volume <= 0:
        raise ValueError('Positive oriented volume / outward winding required')
    normals = np.zeros_like(v)
    for corner in range(3):
        np.add.at(normals, f[:, corner], area_vectors)
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths < 1e-14):
        raise ValueError('Undefined source vertex normal')
    shifted = v + offset_m*normals/lengths[:, None]
    out_tri = shifted[f]
    new_vectors = np.cross(out_tri[:, 1]-out_tri[:, 0], out_tri[:, 2]-out_tri[:, 0])
    dots = np.einsum('ij,ij->i', area_vectors, new_vectors)
    receipt = dict(offset_m=float(offset_m), faces_unchanged=True,
        maximum_vertex_displacement_m=float(np.linalg.norm(shifted-v, axis=1).max()),
        local_face_orientation_reversals=int(np.sum(dots <= 0)),
        global_self_intersections_checked=False, calibrated_cloth=False,
        scope='Area-weighted source vertex normals; uncalibrated parallel surface for offline geometry sensitivity only. No garment cut, wrinkles, material, compression, friction or acoustics.',
        physics_qualified=False, training_eligible=False)
    return shifted, receipt


def probe_surface_patch(triangles, position, normal):
    """97 fixed ray samples over a 46 mm disk, and 64 over its 42--46 mm rim."""
    position = np.asarray(position, float); normal = np.asarray(normal, float)
    if (position.shape != (3,) or normal.shape != (3,)
            or not np.isfinite(position).all() or not np.isfinite(normal).all()
            or not np.isclose(np.linalg.norm(normal), 1., atol=1e-8, rtol=0)):
        raise ValueError('Finite anchor and unit normal required')
    seed = np.eye(3)[int(np.argmin(abs(normal)))]
    u = np.cross(seed, normal); u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    angles = np.arange(32)*2*np.pi/32
    samples = np.concatenate([position[None, :]] + [position + r*(np.cos(angles)[:, None]*u +
                  np.sin(angles)[:, None]*v) for r in (.0115, .021, .023)])
    offsets = ray_offsets(samples, -normal, triangles)
    mask = np.isfinite(offsets)
    hits = samples - offsets[:, None]*normal
    return dict(requested_samples=len(samples), missing_samples=int((~mask).sum()),
        points_world_m=samples.tolist(), inward_direction_world=(-normal).tolist(),
        signed_offsets_m=[float(x) if np.isfinite(x) else None for x in offsets],
        plane=fit_height_envelope(hits, normal) if mask.all() else None,
        ring_plane=fit_height_envelope(hits[33:], normal) if mask[33:].all() else None,
        scope='CPU fixed disk/rim samples on unchanged triangles. Missing rays remain unknown. No continuous footprint, layer disambiguation, collision or physical qualification.',
        training_eligible=False)


def fit_height_envelope(points, reference_normal, *, slope_component_limit_deg=20.):
    p = np.asarray(points, dtype=float)
    n = np.asarray(reference_normal, dtype=float)
    if (p.ndim != 2 or p.shape[1:] != (3,) or len(p) < 3
            or not np.isfinite(p).all() or n.shape != (3,)
            or not np.isfinite(n).all()
            or not np.isclose(np.linalg.norm(n), 1., atol=1e-8, rtol=0)):
        raise ValueError('Finite complete samples and unit reference normal required')
    if (isinstance(slope_component_limit_deg, (bool, np.bool_))
            or not np.isscalar(slope_component_limit_deg)
            or not np.isfinite(slope_component_limit_deg)
            or not 0 < slope_component_limit_deg < 45):
        raise ValueError('Explicit slope component limit between zero and 45 degrees required')
    n = n / np.linalg.norm(n)
    seed = np.eye(3)[int(np.argmin(abs(n)))]
    u = np.cross(seed, n); u /= np.linalg.norm(u)
    v = np.cross(n, u)
    basis = np.column_stack((u, v, n))
    origin = p.mean(axis=0)
    local = (p - origin) @ basis
    if np.linalg.matrix_rank(np.c_[local[:, :2], np.ones(len(p))], tol=1e-10) < 3:
        raise ValueError('Sample footprint is degenerate')
    # Millimetres improve LP conditioning; slopes remain dimensionless.
    x, y, z = (1000 * local).T
    zeros, ones = np.zeros(len(p)), np.ones(len(p))
    # lo <= z - a*x - b*y <= hi; minimize hi-lo.
    matrix = np.r_[np.c_[x, y, ones, zeros], np.c_[-x, -y, zeros, -ones]]
    bound = float(np.tan(np.deg2rad(slope_component_limit_deg)))
    result = linprog([0., 0., -1., 1.], A_ub=matrix, b_ub=np.r_[z, -z],
        bounds=[(-bound, bound), (-bound, bound), (None, None), (None, None)],
        method='highs', options={'primal_feasibility_tolerance': 1e-9,
                                 'dual_feasibility_tolerance': 1e-9})
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError('Height-envelope optimization did not converge')
    a, b, low, high = result.x
    residual = z - a*x - b*y
    if np.max(matrix @ result.x - np.r_[z, -z]) > 1e-7:
        raise ValueError('Height-envelope constraints not satisfied')
    fitted_normal = basis @ [-a, -b, 1.]
    fitted_normal /= np.linalg.norm(fitted_normal)
    return dict(schema='fixed_footprint_height_envelope_v1', samples=len(p),
        reference_normal_world=n.tolist(), origin_world_m=origin.tolist(),
        basis_world_columns=basis.tolist(), slopes=[float(a), float(b)],
        slope_component_limit_deg=float(slope_component_limit_deg),
        maximum_box_corner_normal_angle_deg=float(np.degrees(np.arctan(np.sqrt(2)*bound))),
        normal_world=fitted_normal.tolist(),
        normal_angle_deg=float(np.degrees(np.arctan(np.hypot(a, b)))),
        original_height_span_m=float(np.ptp(z)/1000),
        residual_span_m=float(np.ptp(residual)/1000),
        low_height_m=float(low/1000), high_height_m=float(high/1000),
        solver_objective_span_m=float(result.fun/1000),
        maximum_constraint_violation_m=float(max(0., np.max(matrix @ result.x - np.r_[z, -z]))/1000),
        scope='Fixed sampled footprint height span along reference normal; slope box is diagnostic, not an expert tilt gate. No rotated footprint or full mesh clearance checked.',
        acoustic_qualified=False, finite_head_qualified=False, physics_qualified=False,
        training_eligible=False)
