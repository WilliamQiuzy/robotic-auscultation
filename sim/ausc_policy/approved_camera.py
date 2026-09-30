"""Restore the already approved 7-degree camera layer before any new run.

The 5 mm tool source inherited the pre-review camera. This explicit overlay
restores the approved pose; it does not change camera optics or physical assets.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

CAMERA_SHA256='aeac00eb5a963520e869c60efff52a300484bd1c4a49232067a669c525ad07f0'
CAMERA_FILE=Path(__file__).resolve().parents[1]/'camera_framing_20260923/recommended_camera.json'
PATHS=('/World/RobotCam','/World/RobotCamBody')


def load_approved_camera(data):
    if not isinstance(data,bytes) or hashlib.sha256(data).hexdigest()!=CAMERA_SHA256:
        raise ValueError('Exact approved camera bytes required')
    return json.loads(data)


def local_camera_transform(world_pose,parent_world):
    matrices=[np.asarray(value,float) for value in (world_pose,parent_world)]
    for matrix in matrices:
        if (matrix.shape!=(4,4) or not np.isfinite(matrix).all()
                or not np.allclose(matrix[3],[0,0,0,1],rtol=0,atol=1e-9)
                or not np.allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),rtol=0,atol=1e-8)
                or not np.isclose(np.linalg.det(matrix[:3,:3]),1.,rtol=0,atol=1e-8)):
            raise ValueError('Camera requires rigid transforms')
    return np.linalg.inv(matrices[1])@matrices[0]


def _require_static_ancestors(prim,rigid_body_api):
    while prim and not prim.IsPseudoRoot():
        if prim.HasAPI(rigid_body_api):
            raise ValueError('Approved camera cannot have a physical parent')
        if any(a.GetNumTimeSamples() for a in prim.GetAttributes() if a.GetName().startswith('xformOp')):
            raise ValueError('Approved camera ancestry must be static')
        prim=prim.GetParent()


def apply_approved_camera(stage):
    """Only author the camera and visual body transforms in a flattened stage."""
    from pxr import Usd,UsdGeom,UsdPhysics,Gf
    data=CAMERA_FILE.read_bytes();approved=load_approved_camera(data)
    if len(stage.GetLayerStack(includeSessionLayers=False))!=1:
        raise ValueError('Camera restoration requires an independent flattened stage')
    if UsdGeom.GetStageMetersPerUnit(stage)!=1.:
        raise ValueError('Camera restoration requires original metric stage')
    desired=np.array(approved['world_from_camera_usd'])
    prims=[stage.GetPrimAtPath(path) for path in PATHS]
    if any(not p for p in prims) or not prims[0].IsA(UsdGeom.Camera):
        raise ValueError('Approved camera or visual body missing')
    for root in prims:
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.CollisionAPI):
                raise ValueError('Camera overlay must not move physical geometry')
    def protected():
        rows=[]
        for prim in stage.Traverse():
            path=str(prim.GetPath())
            attrs={a.GetName():str(a.Get()) for a in prim.GetAttributes()
                if not (path in PATHS and a.GetName().startswith('xformOp'))}
            rows.append((path,attrs,{r.GetName():str(r.GetTargets()) for r in prim.GetRelationships()}))
        return hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    before_hash=protected();before={}
    # Preflight every parent and static transform before authoring either prim.
    locals_=[]
    for prim in prims:
        _require_static_ancestors(prim,UsdPhysics.RigidBodyAPI)
        cache=UsdGeom.XformCache()
        before[str(prim.GetPath())]=np.array(cache.GetLocalToWorldTransform(prim)).T.tolist()
        parent_world=np.array(cache.GetLocalToWorldTransform(prim.GetParent())).T
        locals_.append(local_camera_transform(desired,parent_world))
    for prim,local in zip(prims,locals_):
        xf=UsdGeom.Xformable(prim);xf.ClearXformOpOrder()
        xf.AddTransformOp(opSuffix='approved7deg').Set(Gf.Matrix4d(local.T))
    after={str(p.GetPath()):np.array(UsdGeom.XformCache().GetLocalToWorldTransform(p)).T.tolist() for p in prims}
    if any(not np.allclose(value,desired,atol=1e-9,rtol=0) for value in after.values()):
        raise ValueError('Approved world camera transform was not restored')
    after_hash=protected()
    if before_hash!=after_hash:raise ValueError('Camera overlay changed protected scene attributes')
    return dict(camera_sha256=CAMERA_SHA256,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        before=before,after=after,changed_paths=list(PATHS),protected_scene_sha256=after_hash,
        non_camera_attributes_unchanged=True,physical_geometry_unchanged=True,
        camera_optics_unchanged=True,restored_user_approved_pose=True,
        physics_executed=False,training_eligible=False)
