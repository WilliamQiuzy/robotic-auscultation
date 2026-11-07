"""Export current live USD collision geometry, never load a historical plan."""
from pathlib import Path
import json
import numpy as np

def export_geometry(s, output):
    import trimesh
    from pxr import Usd, UsdGeom, UsdPhysics
    OUT = Path(output)
    OUT.mkdir(parents=True, exist_ok=False)
    cache = UsdGeom.XformCache()
    arrays, entries, ignored, planes = {}, [], [], []
    ROBOT = '/World/StethoArmMount/OpenArmStetho'
    def world(p): return np.array(cache.GetLocalToWorldTransform(p)).T
    def owner(p):
        a=p
        while a and not a.IsPseudoRoot():
            if a.HasAPI(UsdPhysics.RigidBodyAPI): return str(a.GetPath())
            a=a.GetParent()
        return 'world'
    allowed=[]
    for p in Usd.PrimRange.Stage(s, Usd.TraverseInstanceProxies()):
        if p.IsA(UsdPhysics.Joint):
            j=UsdPhysics.Joint(p)
            if not j.GetCollisionEnabledAttr().Get():
                a,b=j.GetBody0Rel().GetTargets(),j.GetBody1Rel().GetTargets()
                if a and b: allowed.append([str(a[0]),str(b[0])])
        if p.HasAPI(UsdPhysics.FilteredPairsAPI):
            for other in UsdPhysics.FilteredPairsAPI(p).GetFilteredPairsRel().GetTargets():
                allowed.append([str(p.GetPath()),str(other)])
        if not p.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get(): continue
        path=str(p.GetPath())
        if path.startswith('/World/ContactFixture'): continue
        kind=p.GetTypeName(); mesh=None
        if kind=='Plane':
            axis=UsdGeom.Plane(p).GetAxisAttr().Get(); transform=world(p)
            normal=transform[:3,:3]@np.eye(3)['XYZ'.index(axis)]
            normal/=np.linalg.norm(normal)
            if owner(p)!='world':raise ValueError('Moving infinite plane not supported')
            planes.append(dict(path=path,normal=normal.tolist(),point=transform[:3,3].tolist()))
            continue
        if kind=='Mesh':
            m=UsdGeom.Mesh(p); v=np.asarray(m.GetPointsAttr().Get(), dtype=float)
            ids=np.asarray(m.GetFaceVertexIndicesAttr().Get()); faces=[]; start=0
            for n in m.GetFaceVertexCountsAttr().Get():
                poly=ids[start:start+n]; start+=n
                faces.extend([[poly[0],poly[i],poly[i+1]] for i in range(1,n-1)])
            mesh=trimesh.Trimesh(v,faces,process=False)
        elif kind=='Cube': mesh=trimesh.creation.box(extents=[UsdGeom.Cube(p).GetSizeAttr().Get()]*3)
        elif kind=='Sphere': mesh=trimesh.creation.icosphere(subdivisions=2,radius=UsdGeom.Sphere(p).GetRadiusAttr().Get())
        elif kind in ('Cylinder','Capsule','Cone'):
            g=getattr(UsdGeom,kind)(p); radius=g.GetRadiusAttr().Get(); height=g.GetHeightAttr().Get()
            if kind=='Cylinder': mesh=trimesh.creation.cylinder(radius=radius,height=height,sections=32)
            elif kind=='Capsule':
                mesh=trimesh.creation.capsule(radius=radius,height=height,count=[16,16]);mesh.apply_translation([0,0,-height/2])
            else:
                mesh=trimesh.creation.cone(radius=radius,height=height,sections=32);mesh.apply_translation([0,0,-height/2])
            axis=g.GetAxisAttr().Get()
            if axis!='Z': mesh.apply_transform(trimesh.geometry.align_vectors([0,0,1],[1,0,0] if axis=='X' else [0,1,0]))
        else: ignored.append({'path':path,'type':str(kind)});continue
        body=owner(p); wb=np.eye(4) if body=='world' else world(s.GetPrimAtPath(body))
        local=np.linalg.inv(wb)@world(p)
        mesh.apply_transform(local)
        approximation=UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get() if p.HasAPI(UsdPhysics.MeshCollisionAPI) else None
        # PhysX dynamic mesh colliders use a convex approximation in this asset.
        if approximation=='convexHull': mesh=mesh.convex_hull
        key=f'c{len(entries):03d}';arrays[key+'_v']=np.asarray(mesh.vertices);arrays[key+'_f']=np.asarray(mesh.faces,dtype=np.int32)
        entries.append({'key':key,'path':path,'body':body,'world_from_body':wb.tolist(),'approximation':approximation,'bounds_local':mesh.bounds.tolist()})
    np.savez_compressed(OUT/'collision_geometry.npz',**arrays)
    (OUT/'collision_geometry.json').write_text(json.dumps({'colliders':entries,'allowed_body_pairs':allowed,'unsupported':ignored,'analytic_planes':planes},indent=2))
    if ignored: raise ValueError('Unsupported collision shapes: '+str(ignored))
    return entries
