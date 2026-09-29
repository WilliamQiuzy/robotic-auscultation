"""Build the editable real2sim scene inside Isaac Sim, using metre-valued CAD meshes."""
from pathlib import Path
import json, math
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema, Vt

ROBOT='/World/StethoArmMount/OpenArmStetho'
WRIST=ROBOT+'/openarm_left_link7'
FIXED='/World/Stethoscope/Fixed'
MOVING='/World/Stethoscope/Carriage'
MOUNT_Z=0.085
CONTACT_Z=-0.0412
TRAVEL=0.00725

def matrix_prim(stage,path,matrix):
    p=UsdGeom.Xform.Define(stage,path)
    p.ClearXformOpOrder();p.AddTransformOp().Set(matrix)
    return p.GetPrim()

def material(stage,name,color,metal=0.,rough=.45):
    m=UsdShade.Material.Define(stage,'/World/Real2SimMaterials/'+name)
    s=UsdShade.Shader.Define(stage,str(m.GetPath())+'/Shader')
    s.CreateIdAttr('UsdPreviewSurface')
    s.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    s.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(metal)
    s.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(rough)
    m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(),'surface')
    return m

def bind(p,m):
    UsdShade.MaterialBindingAPI.Apply(p).Bind(m)

def collision(p):
    UsdPhysics.CollisionAPI.Apply(p).CreateCollisionEnabledAttr(True)
    a=PhysxSchema.PhysxCollisionAPI.Apply(p)
    a.CreateContactOffsetAttr(0.00003)
    a.CreateRestOffsetAttr(0.)

def cube(stage,path,size,center,mat=None,collide=True):
    s=UsdGeom.Cube.Define(stage,path);s.CreateSizeAttr(1.)
    s.AddTranslateOp().Set(Gf.Vec3d(*center));s.AddScaleOp().Set(Gf.Vec3f(*size))
    if mat:bind(s.GetPrim(),mat)
    if collide:collision(s.GetPrim())
    return s

def cylinder(stage,path,r,h,center,mat=None,collide=True):
    s=UsdGeom.Cylinder.Define(stage,path);s.CreateRadiusAttr(r);s.CreateHeightAttr(h);s.CreateAxisAttr('Z')
    s.AddTranslateOp().Set(Gf.Vec3d(*center))
    if mat:bind(s.GetPrim(),mat)
    if collide:collision(s.GetPrim())
    return s

def mesh(stage,path,verts,faces,mat=None,collide=False):
    m=UsdGeom.Mesh.Define(stage,path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(verts,dtype=np.float32)))
    f=np.asarray(faces,dtype=np.int32)
    m.CreateFaceVertexCountsAttr([3]*len(f));m.CreateFaceVertexIndicesAttr(f.reshape(-1).tolist())
    m.CreateSubdivisionSchemeAttr('none');m.CreateDoubleSidedAttr(True)
    if mat:bind(m.GetPrim(),mat)
    if collide:
        collision(m.GetPrim());UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr('convexHull')
    return m

def ring_colliders(stage,parent,r0,r1,z0,z1,n=24):
    faces=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    for i in range(n):
        a,b=2*math.pi*i/n,2*math.pi*(i+1)/n
        xy=[(r0*math.cos(a),r0*math.sin(a)),(r1*math.cos(a),r1*math.sin(a)),(r1*math.cos(b),r1*math.sin(b)),(r0*math.cos(b),r0*math.sin(b))]
        v=[(x,y,z) for z in (z0,z1) for x,y in xy]
        mesh(stage,parent+f'/sector_{i:02}',v,faces,collide=True)

def rigid(p,mass,com=(0,0,0),inertia=(.00008,.00008,.00008)):
    r=UsdPhysics.RigidBodyAPI.Apply(p);r.CreateRigidBodyEnabledAttr(True)
    m=UsdPhysics.MassAPI.Apply(p);m.CreateMassAttr(mass);m.CreateCenterOfMassAttr(Gf.Vec3f(*com));m.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
    q=PhysxSchema.PhysxRigidBodyAPI.Apply(p);q.CreateSolverPositionIterationCountAttr(32);q.CreateSolverVelocityIterationCountAttr(8)
    q.CreateSleepThresholdAttr(0.)
    PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr(0.)

def springs(stage,q):
    for i,(sx,sy) in enumerate(((.034,.01),(-.034,-.01))):
        p=UsdGeom.BasisCurves.Get(stage,FIXED+f'/Visuals/Spring_{i}')
        if not p:
            p=UsdGeom.BasisCurves.Define(stage,FIXED+f'/Visuals/Spring_{i}')
            p.CreateTypeAttr('linear');p.CreateWrapAttr('nonperiodic');p.CreateCurveVertexCountsAttr([626])
            p.CreateWidthsAttr([.0006]);p.SetWidthsInterpolation('constant')
            bind(p.GetPrim(),material(stage,'SpringSteel',(.13,.14,.15),.75,.27))
        t=np.linspace(0,1,626);ang=np.pi/2+2*np.pi*19.5*t
        low=-.0295+q+.00069;high=-.0063
        v=np.c_[sx+.0027*np.cos(ang),sy+.0027*np.sin(ang),low+(high-low)*t]
        p.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(v.astype(np.float32)))

def build(stage,root):
    root=Path(root)
    # Flattened original preserves scene assets; this layer is a new candidate.
    manifest=json.loads((root/'assets/component_manifest.json').read_text())
    for suffix in ['openarm_left_left_finger','openarm_left_right_finger','openarm_left_hand_tcp',
                   'joints/openarm_left_finger_joint1','joints/openarm_left_finger_joint2','joints/openarm_left_hand_tcp_joint']:
        p=stage.GetPrimAtPath(ROBOT+'/'+suffix)
        if p:p.SetActive(False)
    for suffix in ['/StethoModule','/visuals','/collisions']:
        p=stage.GetPrimAtPath(WRIST+suffix)
        if p:p.SetActive(False)
    # Wrist visual retains the official arm mesh up to the gripper mounting region.
    # Removed distal palm/fingers cannot continue to produce invisible collisions.
    original=Usd.Stage.Open(str(root/'results/baseline_flattened.usdc'))
    ov=original.GetPrimAtPath(WRIST+'/visuals')
    oldxf=UsdGeom.XformCache();w_inv=oldxf.GetLocalToWorldTransform(original.GetPrimAtPath(WRIST)).GetInverse()
    wrist_mat=material(stage,'WristMetal',(.28,.30,.32),.65,.3)
    n=0
    for p in Usd.PrimRange(ov,Usd.TraverseInstanceProxies()):
        if not p.IsA(UsdGeom.Mesh):continue
        m=UsdGeom.Mesh(p);counts=list(m.GetFaceVertexCountsAttr().Get());ids=list(m.GetFaceVertexIndicesAttr().Get())
        tr=oldxf.GetLocalToWorldTransform(p)*w_inv
        v=np.array([tuple(tr.Transform(Gf.Vec3d(*x))) for x in m.GetPointsAttr().Get()])
        ff=[];offset=0
        for count in counts:
            poly=ids[offset:offset+count];offset+=count
            for j in range(1,count-1):
                tri=[poly[0],poly[j],poly[j+1]]
                if v[tri,2].max()<=.073:ff.append(tri)
        mesh(stage,WRIST+f'/RetainedWrist/Part{n}',v,ff,wrist_mat);n+=1
    # The adapter fills the mating interface; pose is an explicitly labeled estimate.
    adapter=material(stage,'AdapterAluminum',(.55,.58,.62),.8,.23)
    cylinder(stage,WRIST+'/Adapter',.0285,.01025,(0,0,.078125),adapter)
    cylinder(stage,WRIST+'/AdapterSleeve',.016,.031,(0,0,.0575),adapter,collide=False)
    cylinder(stage,WRIST+'/WristCollision',.029,.081,(0,0,.0325),collide=True).CreateVisibilityAttr('invisible')
    # Keep the original serial articulation and finite torque drives.
    for p in stage.Traverse():
        if p.IsA(UsdPhysics.RevoluteJoint) and str(p.GetPath()).startswith(ROBOT):
            d=UsdPhysics.DriveAPI.Apply(p,'angular');d.CreateTargetPositionAttr(0.)
            d.CreateStiffnessAttr(400.);d.CreateDampingAttr(40.);d.CreateMaxForceAttr(80.)
    w=UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(WRIST))
    local=Gf.Matrix4d(1.);local.SetRotate(Gf.Rotation(Gf.Vec3d(1,0,0),180));local.SetTranslateOnly(Gf.Vec3d(0,0,MOUNT_Z))
    world=local*w
    f=matrix_prim(stage,FIXED,world);c=matrix_prim(stage,MOVING,world)
    group_mass={}
    for group,prim in [('fixed',f),('moving',c)]:
        parts=[p for p in manifest['components'] if p['group']==group and p.get('install_default',True)]
        mass=sum(p['mass_kg_estimate'] for p in parts)
        com=sum(p['mass_kg_estimate']*np.array(p['center_of_mass_m_estimate']) for p in parts)/mass
        inertia=np.zeros((3,3))
        for p in parts:
            dx=np.array(p['center_of_mass_m_estimate'])-com
            inertia+=np.array(p['inertia_about_com_kg_m2_estimate'])+p['mass_kg_estimate']*(np.dot(dx,dx)*np.eye(3)-np.outer(dx,dx))
        rigid(prim,mass,com,tuple(np.diag(inertia)))
        group_mass[group]=mass
    j=UsdPhysics.FixedJoint.Define(stage,'/World/Stethoscope/WristMount')
    j.CreateBody0Rel().SetTargets([WRIST]);j.CreateBody1Rel().SetTargets([FIXED])
    j.CreateLocalPos0Attr(Gf.Vec3f(0,0,MOUNT_Z));j.CreateLocalRot0Attr(Gf.Quatf(0,Gf.Vec3f(1,0,0)))
    j.CreateLocalPos1Attr(Gf.Vec3f(0));j.CreateLocalRot1Attr(Gf.Quatf(1))
    j.CreateCollisionEnabledAttr(False)
    j=UsdPhysics.PrismaticJoint.Define(stage,'/World/Stethoscope/PassiveCompression')
    j.CreateBody0Rel().SetTargets([FIXED]);j.CreateBody1Rel().SetTargets([MOVING]);j.CreateAxisAttr('Z')
    j.CreateLowerLimitAttr(0.);j.CreateUpperLimitAttr(TRAVEL);j.CreateCollisionEnabledAttr(False)
    for meth in [j.CreateLocalPos0Attr,j.CreateLocalPos1Attr]:meth(Gf.Vec3f(0))
    for meth in [j.CreateLocalRot0Attr,j.CreateLocalRot1Attr]:meth(Gf.Quatf(1))
    d=UsdPhysics.DriveAPI.Apply(j.GetPrim(),'linear');d.CreateTypeAttr('force')
    d.CreateStiffnessAttr(1000.);d.CreateDampingAttr(3.2);d.CreateTargetPositionAttr(-.0015);d.CreateMaxForceAttr(100.)
    # Exclude internal wrist/module pairs, keep every exterior module collider live.
    for a,b in [(FIXED,MOVING),(WRIST,FIXED),(WRIST,MOVING),(ROBOT+'/openarm_left_link6',FIXED),(ROBOT+'/openarm_left_link6',MOVING)]:
        UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(a)).CreateFilteredPairsRel().AddTarget(b)
    components=manifest['components']
    if isinstance(components,dict):components=[dict(v,name=k) for k,v in components.items()]
    for part in components:
        if part.get('install_default',True) is False or part.get('group')=='spring':continue
        name=part['name'];parent=FIXED if part['group']=='fixed' else MOVING
        fn=part.get('npz_path',part.get('npz',part.get('file')))
        data=np.load(root/'assets'/fn)
        color=part.get('display_color_srgb',part.get('color_rgb',part.get('color',(.045,.048,.052))))
        if isinstance(color,str):color=tuple(int(color.lstrip('#')[i:i+2],16)/255 for i in (0,2,4))
        metal=0.7 if any(k in name for k in ['rod','screw','bolt','cone','dial','usb']) else 0.
        m=material(stage,name,color,part.get('metallic',metal),part.get('roughness',.42))
        mesh(stage,parent+'/Visuals/'+name,data['vertices'],data['faces'],m)
    # Analytical compound collisions preserve holes and the 0.8 mm recessed diaphragm.
    fc=FIXED+'/Colliders';mc=MOVING+'/Colliders'
    for path in [fc,mc]:UsdGeom.Xform.Define(stage,path)
    cylinder(stage,fc+'/FramePlate',.044,.0035,(0,0,0))
    cube(stage,fc+'/ElectronicsWing',(.023,.029,.008),(0,-.039,.0025))
    cube(stage,fc+'/HallPillar',(.006,.009,.011),(.025,-.016,-.006))
    for i,x in enumerate((-.034,.034)):
        cylinder(stage,fc+f'/GuidePost{i}',.0025,.0332,(x,0,-.0166))
        cylinder(stage,fc+f'/PostHead{i}',.004,.0015,(x,0,-.03395))
        cylinder(stage,mc+f'/Bushing{i}',.006,.014,(x,0,-.026))
    ring_colliders(stage,mc+'/ClampWall',.0233,.0263,-.0412,-.0300)
    ring_colliders(stage,mc+'/RetentionLip',.021,.0263,-.0420,-.0412)
    for y in (-.030,.030):cube(stage,mc+('/FrontLugs' if y<0 else '/RearLugs'),(.036,.010,.012),(0,y,-.031))
    cube(stage,mc+'/MagnetTower',(.010,.010,.019),(.025,-.016,-.02925))
    cylinder(stage,mc+'/Diaphragm',.021,.0005,(0,0,-.04095))
    cylinder(stage,mc+'/DeviceEdge',.023,.0057,(0,0,-.03805))
    # Convex hull of the actual tapered CAD cone preserves its exterior radius.
    cone_part=next(p for p in components if p['name']=='thinklabs_cone')
    cone=np.load(root/'assets'/cone_part['npz'])
    mesh(stage,mc+'/DeviceCone',cone['vertices'],cone['faces'],collide=True)
    cylinder(stage,mc+'/DeviceDial',.019,.012,(0,0,-.0182))
    for p in [stage.GetPrimAtPath(fc),stage.GetPrimAtPath(mc)]:UsdGeom.Imageable(p).CreateVisibilityAttr('invisible')
    springs(stage,0.)
    # Contact fixture reproduces the hand pressing the mounted carriage in the reference clip.
    testmat=material(stage,'TestPad',(.045,.35,.42),.05,.55)
    q=Gf.Matrix4d(world);q.SetTranslateOnly(world.Transform(Gf.Vec3d(0,0,CONTACT_Z-.014)))
    p=matrix_prim(stage,'/World/ContactFixture',q);rigid(p,.2)
    UsdPhysics.RigidBodyAPI(p).CreateKinematicEnabledAttr(False)
    PhysxSchema.PhysxRigidBodyAPI.Apply(p).CreateDisableGravityAttr(True)
    # A one-axis test stand keeps the press parallel without prescribing penetration.
    guide=UsdPhysics.PrismaticJoint.Define(stage,'/World/ContactFixtureGuide')
    guide.CreateBody1Rel().SetTargets(['/World/ContactFixture']);guide.CreateAxisAttr('Z')
    guide.CreateLocalPos0Attr(Gf.Vec3f(*world.Transform(Gf.Vec3d(0,0,CONTACT_Z))))
    rot=world.ExtractRotationQuat()
    guide.CreateLocalRot0Attr(Gf.Quatf(float(rot.GetReal()),Gf.Vec3f(*rot.GetImaginary())))
    guide.CreateLocalPos1Attr(Gf.Vec3f(0));guide.CreateLocalRot1Attr(Gf.Quatf(1))
    guide.CreateLowerLimitAttr(-.1);guide.CreateUpperLimitAttr(.1)
    cylinder(stage,'/World/ContactFixture/CentralProbe',.016,.012,(0,0,-.006),testmat)
    cylinder(stage,'/World/ContactFixture/BroadPad',.052,.010,(0,0,-.005),testmat).CreateVisibilityAttr('invisible')
    UsdPhysics.CollisionAPI(stage.GetPrimAtPath('/World/ContactFixture/BroadPad')).GetCollisionEnabledAttr().Set(False)
    cylinder(stage,'/World/ContactFixture/FrameProbe',.005,.012,(0,0,-.006),testmat).CreateVisibilityAttr('invisible')
    UsdPhysics.CollisionAPI(stage.GetPrimAtPath('/World/ContactFixture/FrameProbe')).GetCollisionEnabledAttr().Set(False)
    # Static props retain collision; gravity and a small contact margin are explicit.
    scene=UsdPhysics.Scene.Get(stage,'/World/PhysicsScene');scene.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1));scene.CreateGravityMagnitudeAttr(9.81)
    pa=PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim());pa.CreateTimeStepsPerSecondAttr(240);pa.CreateSolverTypeAttr('TGS');pa.CreateEnableGPUDynamicsAttr(False)
    physicsmat=UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath('/World/Physics_Materials/physics_material'))
    physicsmat.CreateRestitutionAttr(0.)
    UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,'Z')
    metadata={'mount_translation_wrist_m':[0,0,MOUNT_Z],'mount_rotation_cad_to_wrist_deg':[180,0,0],
              'mount_translation_status':'photo/official-wrist-envelope estimate; bench flange measurement pending',
              'stiffness_N_m':1000,'preload_N':1.5,'damping_Ns_m':3.2,'travel_m':TRAVEL,
              'fixed_mass_kg':group_mass['fixed'],'moving_mass_kg':group_mass['moving'],'mass_status':'component volume/density estimates; diagonalized inertia approximation',
              'diaphragm_z_m':CONTACT_Z,'lip_z_m':-.042,'contact_offset_m':.00003,'rest_offset_m':0.,
              'fixed_world_matrix':[[float(v) for v in row] for row in world]}
    (root/'results/build_metadata.json').write_text(json.dumps(metadata,indent=2))
    return metadata
