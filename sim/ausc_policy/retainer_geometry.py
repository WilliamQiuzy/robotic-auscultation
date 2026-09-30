"""Recessed membrane B and device contact ring C; printed clamp A is unchanged.

Coordinates are carriage-local metres, with patientward along -Z. This is a
new review candidate, not a real-world contact or dynamics qualification.
"""
import hashlib
import json

import numpy as np

from .tool_protrusion import MOVING, LIP_Z_M, wrist_contact_offset

GEOMETRY_VERSION = 'stethoscope_retainer_contact_5mm_recess1mm_v2'
RING_ROOT = MOVING + '/Colliders/DeviceContactRing'
MEMBRANE_PATH = MOVING + '/Colliders/Diaphragm'
BACKING_PATH = MOVING + '/Colliders/DeviceEdge'
VISUAL_PATH = MOVING + '/Visuals/thinklabs_black_edge'
RING_FRONT_Z_M = -.047
MEMBRANE_FRONT_Z_M = -.046
PARTITION_Z_M = -.0455
BACKING_REAR_Z_M = -.041
INNER_RADIUS_M = .021
OUTER_RADIUS_M = .023
RECESS_M = .001
SECTOR_COUNT = 32
_CHANGED_ROOTS = (MEMBRANE_PATH, BACKING_PATH, VISUAL_PATH, RING_ROOT)


def geometry_metrics():
    """Nominal dimensions and their provenance, suitable for a JSON receipt."""
    return dict(geometry_id=GEOMETRY_VERSION, ring_front_z_m=RING_FRONT_Z_M,
        membrane_front_z_m=MEMBRANE_FRONT_Z_M, membrane_recess_m=RECESS_M,
        ring_back_z_m=PARTITION_Z_M, backing_rear_z_m=BACKING_REAR_Z_M,
        nominal_protrusion_m=LIP_Z_M-RING_FRONT_Z_M,
        contact_wrist_offset_m=wrist_contact_offset(),
        membrane_wrist_offset_m=wrist_contact_offset()-RECESS_M,
        ring_inner_vertex_radius_m=INNER_RADIUS_M, ring_outer_vertex_radius_m=OUTER_RADIUS_M,
        ring_inner_chord_radius_m=float(INNER_RADIUS_M*np.cos(np.pi/SECTOR_COUNT)),
        ring_inner_chord_sagitta_m=float(INNER_RADIUS_M*(1-np.cos(np.pi/SECTOR_COUNT))),
        ring_sector_count=SECTOR_COUNT,
        dimension_provenance=dict(
            membrane_recess_m='User approximate measurement: B is about 1 mm behind C.',
            protrusion_m='User specified nominal 5 mm beyond printed clamp A; retained from v1.',
            radii_m='Legacy collision proxy radii 21 mm and 23 mm, not a new real-object measurement.',
            ring_axial_partition_m='1.5 mm collision discretization partition, not a measured retainer thickness.'),
        contact_structure='Recessed membrane B plus device contact ring C; backing DeviceEdge stays shell.',
        printed_clamp_contact_allowed=False, physical_validation=False, training_eligible=False,
        inertia_status='Original mass and inertia retained; dynamics are not qualified.')


def ring_sector_meshes():
    """32 individually convex, closed wedges; no hull ever spans the opening."""
    faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], dtype=np.int32)
    result = []
    for index in range(SECTOR_COUNT):
        a, b = 2*np.pi*np.array([index, index+1])/SECTOR_COUNT
        xy = [(r*np.cos(t), r*np.sin(t)) for r, t in
            [(INNER_RADIUS_M, a), (OUTER_RADIUS_M, a), (OUTER_RADIUS_M, b), (INNER_RADIUS_M, b)]]
        result.append((np.array([(x, y, z) for z in (RING_FRONT_Z_M, PARTITION_Z_M)
            for x, y in xy]), faces.copy()))
    return result


def recessed_black_edge_mesh(vertices, faces):
    """Replace only the original flat cap; retain the exact outer and rear shell.

    The replacement has a front annulus, a 1 mm inner wall, and a disk at B.
    Its outer boundary reuses the original tessellation without resizing it.
    """
    vertices = np.asarray(vertices, dtype=float)
    raw_faces = np.asarray(faces)
    if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices)
            or not np.isfinite(vertices).all() or raw_faces.ndim != 2
            or raw_faces.shape[1:] != (3,) or not len(raw_faces)
            or not np.issubdtype(raw_faces.dtype, np.integer)
            or raw_faces.min() < 0 or raw_faces.max() >= len(vertices)):
        raise ValueError('Finite triangle mesh required')
    faces = raw_faces.astype(np.int32)
    front_vertices = np.isclose(vertices[:, 2], RING_FRONT_Z_M, atol=1e-8, rtol=0)
    front_faces = front_vertices[faces].all(axis=1)
    if (not front_faces.any() or not np.isclose(vertices[:, 2].min(), RING_FRONT_Z_M, atol=1e-8, rtol=0)
            or not np.isclose(vertices[:, 2].max(), BACKING_REAR_Z_M, atol=1e-8, rtol=0)):
        raise ValueError('Expected the unchanged v1 black-edge front and rear')
    boundary = np.flatnonzero(front_vertices)
    radii = np.linalg.norm(vertices[boundary, :2], axis=1)
    if not np.allclose(radii, OUTER_RADIUS_M, atol=1e-8, rtol=0):
        raise ValueError('Expected legacy circular front boundary without internal vertices')
    # CAD tessellation repeats coordinates on its face boundaries.
    _, unique = np.unique(np.round(vertices[boundary, :2], 10), axis=0, return_index=True)
    boundary = boundary[unique]
    boundary = boundary[np.argsort(np.arctan2(vertices[boundary, 1], vertices[boundary, 0]))]
    n = len(boundary)
    if n < SECTOR_COUNT:
        raise ValueError('Insufficient front boundary resolution')
    xy = vertices[boundary, :2]
    xy = xy * (INNER_RADIUS_M/np.linalg.norm(xy, axis=1))[:, None]
    inner_front = np.c_[xy, np.full(n, RING_FRONT_Z_M)]
    inner_back = np.c_[xy, np.full(n, MEMBRANE_FRONT_Z_M)]
    start = len(vertices)
    result_vertices = np.vstack([vertices, inner_front, inner_back, [0, 0, MEMBRANE_FRONT_Z_M]])
    added = []
    center = start+2*n
    for i in range(n):
        j = (i+1) % n
        a, b, c, d = int(boundary[i]), int(boundary[j]), start+i, start+j
        e, f = start+n+i, start+n+j
        # Right-handed winding: cap points patientward; bore wall points inward.
        added.extend([[a, c, b], [b, c, d], [c, e, d], [d, e, f], [center, f, e]])
    return result_vertices, np.vstack([faces[~front_faces], np.asarray(added, dtype=np.int32)])


def protected_scene_snapshot(stage):
    """Hash prim metadata, attributes (including samples), and relationships.

    Exact subtree boundaries exclude only B, backing, black-edge visual and C.
    ContactTCP, clamp, camera, joints, materials and physics remain protected.
    """
    def value_fingerprint(value):
        # Hash exact numeric buffers: large patient meshes need neither lossy
        # string formatting nor several hundred MB of temporary decimal text.
        if type(value).__module__ == 'pxr.Vt':
            array = np.asarray(value)
            if array.dtype.kind in 'biufc':
                return dict(dtype=array.dtype.str, shape=array.shape,
                    sha256=hashlib.sha256(array.tobytes()).hexdigest())
        return str(value)
    rows = []
    for prim in stage.TraverseAll():
        path = str(prim.GetPath())
        if any(path == root or path.startswith(root+'/') for root in _CHANGED_ROOTS):
            continue
        attrs = {a.GetName(): dict(value=value_fingerprint(a.Get()), metadata={k: str(v) for k, v in a.GetAllMetadata().items()},
            samples=[(t, value_fingerprint(a.Get(t))) for t in a.GetTimeSamples()]) for a in prim.GetAttributes()}
        rels = {r.GetName(): dict(targets=[str(t) for t in r.GetTargets()],
            metadata={k: str(v) for k, v in r.GetAllMetadata().items()}) for r in prim.GetRelationships()}
        rows.append((path, {k: str(v) for k, v in prim.GetAllMetadata().items()}, attrs, rels))
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def apply_to_stage(stage):
    """Apply to an independent flattened v1 candidate, with source preflight."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, Vt
    if stage.GetPrimAtPath(RING_ROOT):
        raise ValueError('Retainer geometry revision already applied')
    if (len(stage.GetLayerStack(includeSessionLayers=False)) != 1
            or UsdGeom.GetStageMetersPerUnit(stage) != 1.
            or str(UsdGeom.GetStageUpAxis(stage)) != 'Z'):
        raise ValueError('Independent flattened metric Z-up stage required')
    cache = UsdGeom.XformCache()
    moving = stage.GetPrimAtPath(MOVING)
    if not moving:
        raise ValueError('Missing carriage')
    body_inverse = cache.GetLocalToWorldTransform(moving).GetInverse()
    def local(prim):
        return cache.GetLocalToWorldTransform(prim)*body_inverse
    def cylinder_preflight(path, radius, front, rear):
        prim = stage.GetPrimAtPath(path)
        cylinder = UsdGeom.Cylinder(prim)
        if (not cylinder or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                or cylinder.GetAxisAttr().Get() != 'Z'
                or not np.isclose(cylinder.GetRadiusAttr().Get(), radius, atol=1e-9, rtol=0)):
            raise ValueError('Missing expected live legacy cylinder: '+path)
        h = cylinder.GetHeightAttr().Get()
        np.testing.assert_allclose(local(prim.GetParent()), np.eye(4), atol=1e-8, rtol=0)
        np.testing.assert_allclose(local(prim), Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, 0, (front+rear)/2)), atol=1e-8, rtol=0)
        if not np.isclose(h, rear-front, atol=1e-9, rtol=0):
            raise ValueError('Unexpected legacy cylinder span: '+path)
        for name in ('physxCollision:contactOffset', 'physxCollision:restOffset'):
            value = prim.GetAttribute(name).Get()
            if value is None or not np.isfinite(value) or (name.endswith('restOffset') and value > 0):
                raise ValueError('Invalid inherited '+name)
        if any(a.GetNumTimeSamples() for a in prim.GetAttributes()):
            raise ValueError('Static source geometry required')
        return cylinder
    diaphragm = cylinder_preflight(MEMBRANE_PATH, INNER_RADIUS_M, -.047, -.0465)
    backing = cylinder_preflight(BACKING_PATH, OUTER_RADIUS_M, -.0467, BACKING_REAR_Z_M)
    tcp = stage.GetPrimAtPath(MOVING+'/ContactTCP')
    if not tcp:
        raise ValueError('Missing v1 ContactTCP')
    np.testing.assert_allclose(local(tcp).ExtractTranslation(), [0, 0, RING_FRONT_Z_M], atol=1e-9, rtol=0)
    visual = UsdGeom.Mesh(stage.GetPrimAtPath(VISUAL_PATH))
    if not visual:
        raise ValueError('Missing black-edge visual')
    np.testing.assert_allclose(local(visual.GetPrim()), np.eye(4), atol=1e-8, rtol=0)
    counts = np.asarray(visual.GetFaceVertexCountsAttr().Get())
    if not (counts == 3).all():
        raise ValueError('Expected legacy triangle visual')
    vertices, faces = recessed_black_edge_mesh(visual.GetPointsAttr().Get(),
        np.asarray(visual.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3))
    before = protected_scene_snapshot(stage)
    for cylinder, front, rear in [(diaphragm, MEMBRANE_FRONT_Z_M, PARTITION_Z_M),
            (backing, PARTITION_Z_M, BACKING_REAR_Z_M)]:
        cylinder.CreateHeightAttr(rear-front)
        cylinder.ClearXformOpOrder()
        cylinder.AddTranslateOp(opSuffix='retainerV2').Set(Gf.Vec3d(0, 0, (front+rear)/2))
    UsdGeom.Xform.Define(stage, RING_ROOT)
    layer = stage.GetRootLayer()
    template = backing.GetPrim()
    for i, (v, f) in enumerate(ring_sector_meshes()):
        mesh = UsdGeom.Mesh.Define(stage, RING_ROOT+f'/sector_{i:02d}')
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v.astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr([3]*len(f))
        mesh.CreateFaceVertexIndicesAttr(f.reshape(-1).tolist())
        mesh.CreateSubdivisionSchemeAttr('none')
        mesh.CreateExtentAttr([Gf.Vec3f(*v.min(axis=0)), Gf.Vec3f(*v.max(axis=0))])
        prim = mesh.GetPrim()
        prim.SetMetadata('apiSchemas', template.GetMetadata('apiSchemas'))
        # Copy authored physics values and material relationships with metadata.
        # The same parent preserves inherited bindings. No positive rest offset.
        for prop in template.GetAuthoredProperties():
            if prop.GetName().startswith(('physics:', 'physx', 'material:')):
                Sdf.CopySpec(layer, prop.GetPath(), layer, prim.GetPath().AppendProperty(prop.GetName()))
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexHull')
    visual.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
    visual.GetFaceVertexCountsAttr().Set([3]*len(faces))
    visual.GetFaceVertexIndicesAttr().Set(faces.reshape(-1).tolist())
    visual.CreateExtentAttr([Gf.Vec3f(*vertices.min(axis=0)), Gf.Vec3f(*vertices.max(axis=0))])
    after = protected_scene_snapshot(stage)
    if before != after:
        raise ValueError('Unexpected change outside the retainer geometry revision')
    layer.customLayerData = dict(layer.customLayerData or {},
        wamGeometryVersion=GEOMETRY_VERSION, wamTrainingEligible=False)
    report = geometry_metrics()
    report.update(changed_prims=list(_CHANGED_ROOTS), protected_scene_unchanged=True,
        protected_scene_sha256=after, contact_tcp_unchanged=True,
        original_black_edge_outer_and_rear_shell_retained=True,
        collision_margins_inherited=dict(contact_offset_m=float(template.GetAttribute('physxCollision:contactOffset').Get()),
            rest_offset_m=float(template.GetAttribute('physxCollision:restOffset').Get())),
        collision_material_bindings_inherited=True)
    return report
