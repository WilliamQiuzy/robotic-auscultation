"""Prepare a disposable, fully collidable simulation stage. Never edits input layers."""
from pathlib import Path
import json

ROBOT = '/World/StethoArmMount/OpenArmStetho'
WRIST = ROBOT + '/openarm_left_link7'
FIXED = '/World/Stethoscope/Fixed'
MOVING = '/World/Stethoscope/Carriage'
ROOT = Path(__file__).resolve().parent

def prepare_stage(destination):
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf
    source = Usd.Stage.Open(str(ROOT.parent/'auscultation_assets_20260922/contact_static.usda'))
    stage = Usd.Stage.Open(source.Flatten())
    for path in ['/Render','/Replicator','/Orchestrator','/World/ContactFixture','/World/ContactFixtureGuide']:
        if stage.GetPrimAtPath(path): stage.RemovePrim(path)
    while True:
        instances = [p for p in stage.Traverse() if p.IsInstance()]
        if not instances: break
        for p in instances: p.SetInstanceable(False)
    # Restore VEO6 CAD envelopes rather than the old whole-width lug bars.
    # Four separated ears preserve the open seam. Bore holes remain a
    # conservative omission in these simple convex contact proxies.
    def box(path,size,center):
        cube=UsdGeom.Cube.Define(stage,path);cube.CreateSizeAttr(1.)
        xf=UsdGeom.Xformable(cube);xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*center));xf.AddScaleOp().Set(Gf.Vec3f(*size))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
        api=PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
        api.CreateContactOffsetAttr(.00003);api.CreateRestOffsetAttr(0.)
    for name in ['FrontLugs','RearLugs']:
        stage.RemovePrim(MOVING+'/Colliders/'+name)
    for side,x in [('Left',-.010),('Right',.010)]:
        for end,y in [('Front',-.0295),('Rear',.0295)]:
            box(MOVING+'/Colliders/'+end+side+'Ear',(.014,.011,.011),(x,y,-.033))
    box(MOVING+'/Colliders/MagnetTower',(.010,.009,.01325),(.025,-.016,-.026375))
    # CAD lower bore Ø47 mm, outside Ø53 mm. Preserve the existing lip bore.
    import numpy as np
    for name in ['ClampWall','RetentionLip']:
        prim=stage.GetPrimAtPath(MOVING+'/Colliders/'+name)
        for p in Usd.PrimRange(prim):
            if not p.IsA(UsdGeom.Mesh):continue
            mesh=UsdGeom.Mesh(p);points=np.asarray(mesh.GetPointsAttr().Get(),dtype=float)
            radii=np.linalg.norm(points[:,:2],axis=1)
            # The ring exporter uses tessellated radial bands.
            outer=radii>.025;inner=(radii>.022)&~outer
            points[outer,:2]*=(.0265/radii[outer])[:,None]
            points[inner,:2]*=(.0235/radii[inner])[:,None]
            mesh.GetPointsAttr().Set([Gf.Vec3f(*p) for p in points])
    # Split off the intended front edge without changing the occupied geometry.
    rim_parent=MOVING+'/Colliders/ContactRim'
    UsdGeom.Xform.Define(stage,rim_parent)
    for prim in list(Usd.PrimRange(stage.GetPrimAtPath(MOVING+'/Colliders/ClampWall'))):
        if not prim.IsA(UsdGeom.Mesh):continue
        path=str(prim.GetPath());dest=rim_parent+'/'+prim.GetName()
        Sdf.CopySpec(stage.GetRootLayer(),path,stage.GetRootLayer(),dest)
        wall=UsdGeom.Mesh(prim);rim=UsdGeom.Mesh(stage.GetPrimAtPath(dest))
        points=np.asarray(wall.GetPointsAttr().Get(),dtype=float)
        front=points[:,2]<-.04
        wp=points.copy();wp[front,2]=-.0402
        rp=points.copy();rp[~front,2]=-.0402
        wall.GetPointsAttr().Set([Gf.Vec3f(*p) for p in wp])
        rim.GetPointsAttr().Set([Gf.Vec3f(*p) for p in rp])
    repaired=[]
    for p in stage.Traverse():
        path=str(p.GetPath())
        if p.HasAPI(UsdPhysics.CollisionAPI) and p.GetTypeName()=='Xform':
            p.RemoveAPI(UsdPhysics.MeshCollisionAPI); p.RemoveAPI(UsdPhysics.CollisionAPI)
        arm_mesh = path.startswith(ROBOT) and '/collisions/' in path and p.IsA(UsdGeom.Mesh)
        table_mesh = path.startswith('/World/Table/') and p.IsA(UsdGeom.Mesh)
        if arm_mesh or table_mesh:
            UsdPhysics.CollisionAPI.Apply(p).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr('convexHull' if arm_mesh else 'none')
            api=PhysxSchema.PhysxCollisionAPI.Apply(p)
            api.CreateContactOffsetAttr(.001); api.CreateRestOffsetAttr(0.)
            repaired.append(path)
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr(0.)
            PhysxSchema.PhysxRigidBodyAPI.Apply(p).CreateSolverPositionIterationCountAttr(32)
            PhysxSchema.PhysxRigidBodyAPI.Apply(p).CreateSolverVelocityIterationCountAttr(8)
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            a=PhysxSchema.PhysxArticulationAPI.Apply(p)
            a.CreateEnabledSelfCollisionsAttr(True)
            a.CreateSolverPositionIterationCountAttr(32); a.CreateSolverVelocityIterationCountAttr(8)
        if p.IsA(UsdPhysics.RevoluteJoint) and path.startswith(ROBOT):
            name=p.GetName(); i=int(name[-1]) if name[-1].isdigit() else 1
            effort=[40,40,27,27,7,7,7][min(i,7)-1]
            drive=UsdPhysics.DriveAPI.Apply(p,'angular')
            drive.CreateStiffnessAttr(1600.); drive.CreateDampingAttr(80.)
            drive.CreateMaxForceAttr(float(effort));drive.CreateTargetPositionAttr(0.)
    # The two wrist axes share a compact housing: link5's convex hub and
    # link7's cylindrical collision proxy overlap at their common bearing.
    # Filter only this internal assembly pair; exterior collisions stay live.
    for side in ('left','right'):
        a=stage.GetPrimAtPath(ROBOT+f'/openarm_{side}_link5')
        UsdPhysics.FilteredPairsAPI.Apply(a).CreateFilteredPairsRel().AddTarget(ROBOT+f'/openarm_{side}_link7')
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    stage.GetRootLayer().Export(str(destination))
    (destination.parent/'scene_repairs.json').write_text(json.dumps({
        'source':'../auscultation_assets_20260922/contact_static.usda',
        'repaired_collider_meshes':repaired,'self_collisions_enabled':True,
        'arm_effort_limits_Nm':[40,40,27,27,7,7,7],
        'internal_wrist_housing_pairs_filtered':[['link5','link7']],
        'module_proxy_repairs':{'ears':'four14x11x11mm,centers(+-10,+-29.5,-33)mm',
                                'magnet_tower':'10x9x13.25mm,z[-33,-19.75]mm',
                                'ring_outer_radius_mm':26.5,'ring_bore_radius_mm':23.5,
                                'front_contact_band_z_mm':[-41.2,-40.2],
                                'cad_source':'thinklab_openarm_veo6_realshape_mount.py'},
        'original_layers_modified':False},indent=2))
    return stage
