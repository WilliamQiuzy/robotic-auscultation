"""Versioned stethoscope geometry candidate; the legacy scene stays reproducible."""
import numpy as np

LIP_Z_M = -.042
FACE_Z_M = -.047
OLD_FACE_Z_M = -.0412
HEAD_SHIFT_M = FACE_Z_M - OLD_FACE_Z_M
BORE_RADIUS_M = .0235
MOVING = '/World/Stethoscope/Carriage'
HEAD_VISUALS = ('thinklabs_black_edge', 'thinklabs_cone', 'thinklabs_dial')
HEAD_COLLIDERS = ('Diaphragm', 'DeviceEdge', 'DeviceCone', 'DeviceDial')


def wrist_contact_offset(compression=0.):
    if not np.isfinite(compression) or not 0 <= compression <= .00725:
        raise ValueError('Invalid compression')
    return .085 - FACE_Z_M - compression


def points(vertices):
    value = np.asarray(vertices, dtype=float)
    if value.ndim != 2 or value.shape[1] != 3 or not np.isfinite(value).all():
        raise ValueError('Finite Nx3 geometry required')
    return value.copy()


def shift_head(vertices):
    value = points(vertices)
    value[:, 2] += HEAD_SHIFT_M
    return value


def open_lower_bore(vertices):
    value = points(vertices)
    radius = np.linalg.norm(value[:, :2], axis=1)
    if np.any(radius < 1e-9):
        raise ValueError('Unexpected axial clamp vertex')
    selected = (value[:, 2] <= OLD_FACE_Z_M + 1e-7) & (radius < BORE_RADIUS_M - 1e-7)
    value[selected, :2] *= (BORE_RADIUS_M / radius[selected])[:, None]
    return value


def apply_to_stage(stage):
    """Edit only a disposable candidate; legacy policies must not use its old TCP."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt
    if stage.GetPrimAtPath(MOVING + '/ContactTCP'):
        raise ValueError('Candidate patch already applied')
    changed = []
    for name in HEAD_VISUALS:
        path = MOVING + '/Visuals/' + name
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(path))
        if not mesh:
            raise ValueError('Missing device visual: ' + path)
        vertices = shift_head(mesh.GetPointsAttr().Get())
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
        mesh.CreateExtentAttr([Gf.Vec3f(*vertices.min(axis=0)), Gf.Vec3f(*vertices.max(axis=0))])
        changed.append(path)
    for name in HEAD_COLLIDERS:
        path = MOVING + '/Colliders/' + name
        prim = stage.GetPrimAtPath(path)
        if not prim or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            raise ValueError('Missing live head collider: ' + path)
        UsdGeom.Xformable(prim).AddTranslateOp(opSuffix='protrusionV1').Set(Gf.Vec3d(0, 0, HEAD_SHIFT_M))
        changed.append(path)
    for name in ('left_clamshell', 'right_clamshell'):
        path = MOVING + '/Visuals/' + name
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(path))
        vertices = open_lower_bore(mesh.GetPointsAttr().Get())
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
        mesh.CreateExtentAttr([Gf.Vec3f(*vertices.min(axis=0)), Gf.Vec3f(*vertices.max(axis=0))])
        changed.append(path)
    for prim in Usd.PrimRange(stage.GetPrimAtPath(MOVING + '/Colliders/RetentionLip')):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        vertices = open_lower_bore(mesh.GetPointsAttr().Get())
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
        changed.append(str(prim.GetPath()))
    tcp = UsdGeom.Xform.Define(stage, MOVING + '/ContactTCP')
    tcp.AddTranslateOp().Set(Gf.Vec3d(0, 0, FACE_Z_M))
    tcp.GetPrim().CreateAttribute('wam:nominalWristOffsetM', Sdf.ValueTypeNames.Double).Set(wrist_contact_offset())
    tcp.GetPrim().CreateAttribute('wam:protrusionM', Sdf.ValueTypeNames.Double).Set(LIP_Z_M - FACE_Z_M)
    tcp.GetPrim().CreateAttribute('wam:dimensionStatus', Sdf.ValueTypeNames.String).Set('user-requested nominal 5 mm; not measured from real assembly')
    stage.GetRootLayer().customLayerData = dict(stage.GetRootLayer().customLayerData or {},
        wamGeometryVersion='stethoscope_protruding_5mm_review_v1',
        wamTrainingEligible=False)
    return dict(changed_prims=changed, nominal_protrusion_m=LIP_Z_M - FACE_Z_M,
        head_translation_m=HEAD_SHIFT_M, diaphragm_wrist_offset_m=wrist_contact_offset(),
        clamp_bore_diameter_m=2 * BORE_RADIUS_M, dimension_status='User-requested nominal 5 mm; not bench measurements',
        lower_retention='Straight bore preserved from user-reviewed 2 mm candidate; mechanical retention not load-validated.',
        physical_validation=False, training_dataset_eligible=False, m0_passed=False,
        inertia_status='Original estimated mass and inertia retained for static geometry review; dynamics not qualified.',
        integration='Use candidate ContactTCP; historical policies and precomputed plans retain their original scene/TCP.')
