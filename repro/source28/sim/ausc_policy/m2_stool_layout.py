"""Prepare/install a bounded stool layout from authenticated original geometry.

Pure preparation calls the existing exact nine-cylinder stool_geometry builder.
Native installation requires an already initialized USD/Isaac stage; this module
never creates SimulationApp, imports PhysxSchema, starts physics, or qualifies
support/collisions/training. Call before install_patient with the SAME layout.
"""
import hashlib
import io
import json
from pathlib import Path
import numpy as np

from .geometry_epoch import require_current
from .stool_geometry import build_stool_geometry, STOOL_PATHS

# Same frozen original collision exports as m2_retarget.FROZEN; do not substitute
# an already translated/exported scene and compound the layout translation.
FROZEN_GEOMETRY_SHA256='11a2fc922602a70a97d83b26cd0cd9e0868a535c9d62e0fd10a38ca975b76275'
FROZEN_ARRAYS_SHA256='0cd0e871d723a3505e5c565be688c5c5860d678faac2534f8bb837af77974a05'
HEIGHTS=(-.06,-.03,0.,.03,.06)
FLOAT32_TOL_M=2e-6
FLAGS=dict(physics_qualified=False,training_eligible=False,collision_qualified=False,
           support_qualified=False,physics_executed=False)


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return digest.hexdigest()


def validate_layout(layout):
    need(isinstance(layout,dict) and set(layout)=={'xy_m','height_m'},'Explicit common xy_m/height_m layout required')
    xy=np.asarray(layout['xy_m']);height=layout['height_m']
    need(xy.shape==(2,) and xy.dtype.kind in 'fiu' and np.isfinite(xy).all() and
         np.all(abs(xy)<=.15+1e-12),'Stool layout XY must remain within +/- .15 m')
    need(isinstance(height,(int,float)) and not isinstance(height,bool) and height in HEIGHTS,
         'Stool layout height must be one of five approved values')
    return dict(xy_m=xy.astype(float).tolist(),height_m=float(height))


def require_matching_layout(layout,patient_spec):
    expected=validate_layout(layout)
    need(validate_layout(patient_spec.get('layout'))==expected,'Stool and patient layout must match exactly')


def prepare_stool_layout(geometry_directory,metadata,layout):
    """Pure authenticated proposal; returned world meshes are float32-ready."""
    layout=validate_layout(layout);require_current(metadata)
    bindings=dict(metadata,**metadata.get('geometry_artifacts',{}))
    for key,expected in [('collision_geometry_sha256',FROZEN_GEOMETRY_SHA256),
                         ('collision_arrays_sha256',FROZEN_ARRAYS_SHA256)]:
        need(bindings.get(key)==expected,'Original frozen source artifact hash required: '+key)
    directory=Path(geometry_directory)
    geometry=directory/'collision_geometry.json';arrays_file=directory/'collision_geometry.npz'
    geometry_bytes=geometry.read_bytes();arrays_bytes=arrays_file.read_bytes()
    need(hashlib.sha256(geometry_bytes).hexdigest()==FROZEN_GEOMETRY_SHA256 and
         hashlib.sha256(arrays_bytes).hexdigest()==FROZEN_ARRAYS_SHA256,
         'Frozen stool source artifact hash mismatch')
    source=json.loads(geometry_bytes)
    with np.load(io.BytesIO(arrays_bytes),allow_pickle=False) as z:arrays={key:np.array(z[key],copy=True) for key in z.files}
    planes=source.get('analytic_planes',[])
    need(len(planes)==1 and np.allclose(planes[0].get('normal'),[0,0,1],rtol=0,atol=1e-12),
         'Single unchanged horizontal floor required')
    floor=float(planes[0]['point'][2])
    seat=next(row for row in source['colliders'] if row['path']=='/World/Stool/seat')
    matrix=np.asarray(seat['world_from_body'],float)
    seat_world=arrays[seat['key']+'_v']@matrix[:3,:3].T+matrix[:3,3]
    top=float(seat_world[:,2].max())
    proposal=build_stool_geometry(source,arrays,new_seat_top_z_m=top+layout['height_m'],
        stool_xy_translation_m=layout['xy_m'],floor_z_m=floor,source_bindings=bindings)
    parts={}
    for row in proposal['metadata']['colliders']:
        path=row['path']
        if path not in STOOL_PATHS:continue
        matrix=np.asarray(row['world_from_body'],float);key=row['key']
        world=proposal['arrays'][key+'_v']@matrix[:3,:3].T+matrix[:3,3]
        vertices=world.astype(np.float32);faces=proposal['arrays'][key+'_f'].astype(np.int32)
        extent=np.array([vertices.min(0),vertices.max(0)],np.float32)
        need(np.allclose(vertices,world,rtol=0,atol=FLOAT32_TOL_M),'Stool float32 conversion exceeds tolerance')
        report=proposal['generation_report']['parts'][path]
        parts[path]=dict(vertices_world_f32=vertices.tolist(),faces=faces.tolist(),extent_world_f32=extent.tolist(),
            before_bounds_world_m=report['source_bounds_world_m'],after_bounds_world_m=report['proposal_bounds_world_m'],
            points_sha256=hashlib.sha256(vertices.astype('<f4').tobytes()).hexdigest(),
            faces_sha256=hashlib.sha256(faces.astype('<i4').tobytes()).hexdigest())
    need(set(parts)==set(STOOL_PATHS),'Exactly nine stool meshes required')
    return dict(schema='m2_stool_layout_proposal_v1',layout=layout,parts=parts,floor_z_m=floor,
        floor_plane=planes[0],source_bindings=bindings,content_bindings=proposal['content_bindings'],
        float32_tolerance_m=FLOAT32_TOL_M,native_scene_installed=False,**FLAGS)


def _protected_snapshot(stage):
    """Serialize resolved stage excluding exactly the nine allowed leaf prims."""
    from pxr import Usd
    layer=stage.Flatten();copy=Usd.Stage.Open(layer)
    for path in STOOL_PATHS:
        need(copy.RemovePrim(path),'Cannot isolate protected stage content')
    return layer.ExportToString()


def _properties(prim,excluded):
    values={}
    for prop in prim.GetAuthoredProperties():
        name=prop.GetName()
        if name in excluded:continue
        # Metadata includes authored defaults; preserve time samples/connections
        # and material/physics relationship targets as well.
        record=dict(metadata=repr(sorted(prop.GetAllMetadata().items())))
        if hasattr(prop,'GetTargets'):record['targets']=[str(p) for p in prop.GetTargets()]
        else:
            record['connections']=[str(p) for p in prop.GetConnections()]
            record['samples']=[(float(t),repr(prop.Get(t))) for t in prop.GetTimeSamples()]
        values[name]=record
    return values


def install_stool_layout(stage,geometry_directory,metadata,layout,*,patient_spec=None):
    """Install same-path world meshes, verify protected scene, rollback on error.

    Parent must call this before patient installation and pass the same layout.
    Original authored material/physics/visual properties remain; mesh topology,
    extent, xformOpOrder and MeshCollisionAPI/approximation are the deliberate
    geometry representation changes. Native installation is not qualification.
    """
    proposal=prepare_stool_layout(geometry_directory,metadata,layout)
    if patient_spec is not None:require_matching_layout(layout,patient_spec)
    from pxr import Usd,UsdGeom,UsdPhysics,Vt,Gf
    need(len(stage.GetLayerStack(includeSessionLayers=False))==1,
         'Stool installation requires one editable source layer')
    root=stage.GetRootLayer()
    need(stage.GetEditTarget().GetLayer()==root,'Stool installation must edit source root layer')
    backup=root.ExportToString();protected=_protected_snapshot(stage)
    # Cylinder-only schema defaults cease to exist after changing the type.
    # Geometry is instead authenticated by the exact Mesh world points/faces
    # and extents below. Material, physics and all other properties stay bound.
    excluded={'points','faceVertexCounts','faceVertexIndices','extent','xformOpOrder',
              'subdivisionScheme','physics:approximation','radius','height','axis'}
    saved={};before_bounds={};before_apis={}
    cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_])
    for path,part in proposal['parts'].items():
        prim=stage.GetPrimAtPath(path)
        need(bool(prim) and prim.GetTypeName()=='Cylinder' and not prim.GetChildren(),
             'Expected original leaf stool cylinder: '+path)
        need(prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get(),
             'Original stool collision must remain enabled')
        bound=cache.ComputeWorldBound(prim).ComputeAlignedBox()
        bounds=np.array([bound.GetMin(),bound.GetMax()],float)
        need(np.allclose(bounds,part['before_bounds_world_m'],rtol=0,atol=FLOAT32_TOL_M),
             'Live stool bounds differ from authenticated original: '+path)
        before_bounds[path]=bounds.tolist();saved[path]=_properties(prim,excluded)
        before_apis[path]=set(prim.GetAppliedSchemas())
    installed={}
    try:
        for path,part in proposal['parts'].items():
            prim=stage.GetPrimAtPath(path);prim.SetTypeName('Mesh')
            mesh=UsdGeom.Mesh(prim);transform=UsdGeom.Xformable(prim)
            transform.ClearXformOpOrder();transform.SetResetXformStack(True)
            vertices=np.asarray(part['vertices_world_f32'],np.float32);faces=np.asarray(part['faces'],np.int32)
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
            mesh.CreateFaceVertexCountsAttr([3]*len(faces));mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
            mesh.CreateSubdivisionSchemeAttr('none')
            mesh.CreateExtentAttr([Gf.Vec3f(*map(float,vertices.min(0))),
                                   Gf.Vec3f(*map(float,vertices.max(0)))])
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('none')
            need(_properties(prim,excluded)==saved[path] and before_apis[path]<=set(prim.GetAppliedSchemas()),
                 'Stool material/physics/visual properties changed: '+path)
            actual=np.asarray(mesh.GetPointsAttr().Get(),np.float32)
            matrix=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).T
            world=actual@matrix[:3,:3].T+matrix[:3,3]
            extent=np.asarray(mesh.GetExtentAttr().Get(),np.float32)
            need(transform.GetResetXformStack() and np.allclose(matrix,np.eye(4),rtol=0,atol=1e-12) and
                 np.array_equal(actual,vertices) and np.array_equal(extent,np.array([vertices.min(0),vertices.max(0)])) and
                 np.allclose(world,vertices,rtol=0,atol=FLOAT32_TOL_M), 'Installed stool world points/extent mismatch')
            need(np.array_equal(mesh.GetFaceVertexIndicesAttr().Get(),faces.reshape(-1)) and
                 list(mesh.GetFaceVertexCountsAttr().Get())==[3]*len(faces),'Installed stool topology mismatch')
            installed[path]=dict(before_bounds_world_m=before_bounds[path],
                after_bounds_world_m=[world.min(0).tolist(),world.max(0).tolist()],
                points_sha256=part['points_sha256'],faces_sha256=part['faces_sha256'])
        after=_protected_snapshot(stage)
        need(after==protected,'A non-stool prim, floor, robot, tool, body or camera changed')
    except BaseException:
        root.ImportFromString(backup)
        raise
    return dict(schema='m2_stool_layout_installation_v1',layout=proposal['layout'],
        source_bindings=proposal['source_bindings'],content_bindings=proposal['content_bindings'],parts=installed,
        protected_scene_sha256=hashlib.sha256(protected.encode()).hexdigest(),non_stool_scene_unchanged=True,
        floor_plane_unchanged=True,floor_z_m=proposal['floor_z_m'],float32_tolerance_m=FLOAT32_TOL_M,
        native_scene_installed=True,native_installation_acceptance=False,**FLAGS)
