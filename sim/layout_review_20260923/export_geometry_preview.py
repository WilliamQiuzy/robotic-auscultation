"""Export actual USD visible control geometry for a non-Isaac camera preview.

Only an anonymous session layer is modified. No source assets are written.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, Usd, UsdGeom, UsdShade

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
MODULE = PROJECT / "sim/auscultation_training"
SOURCE = MODULE / "results/low_elbow_release_v1/physics_all10/training_scene.usdc"
PLANS = MODULE / "results/low_elbow_release_v1/plans"
OUT = HERE / "geometry_export"
NAMES = [f"{s}{i}" for i in range(1, 6) for s in ("R", "L")]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def camera_state(stage):
    c = UsdGeom.Camera(stage.GetPrimAtPath("/World/RobotCam"))
    world = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(c.GetPrim()),float).T
    f,ha,va = [float(a.Get()) for a in (c.GetFocalLengthAttr(),c.GetHorizontalApertureAttr(),c.GetVerticalApertureAttr())]
    ho,vo = float(c.GetHorizontalApertureOffsetAttr().Get()),float(c.GetVerticalApertureOffsetAttr().Get())
    return {"world_from_optical":(world@np.diag([1.,-1.,-1.,1.])).tolist(),
        "world_from_usd_camera":world.tolist(),"fx":f/ha*1920,"fy":f/va*1080,
        "cx":960-ho/ha*1920,"cy":540+vo/va*1080,"width":1920,"height":1080,
        "focal_length":f,"horizontal_aperture":ha,"vertical_aperture":va,
        "horizontal_aperture_offset":ho,"vertical_aperture_offset":vo,
        "clipping_range":list(c.GetClippingRangeAttr().Get()),"projection":str(c.GetProjectionAttr().Get())}


def bounds(stage,path):
    r=UsdGeom.BBoxCache(Usd.TimeCode.Default(),["default","render"]).ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
    return np.asarray([r.GetMin(),r.GetMax()],float)


def move_world(stage,path,delta):
    p=stage.GetPrimAtPath(path);cache=UsdGeom.XformCache()
    world=cache.GetLocalToWorldTransform(p)
    world.SetTranslateOnly(world.ExtractTranslation()+Gf.Vec3d(*map(float,delta)))
    local=world*cache.GetLocalToWorldTransform(p.GetParent()).GetInverse()
    xf=UsdGeom.Xformable(p);xf.ClearXformOpOrder();xf.AddTransformOp(opSuffix="geometryLayoutReview").Set(local)


def move_local_z(p,dz):
    xf=UsdGeom.Xformable(p);matrix=xf.GetLocalTransformation();v=matrix.ExtractTranslation()
    matrix.SetTranslateOnly(Gf.Vec3d(v[0],v[1],v[2]+dz))
    xf.ClearXformOpOrder();xf.AddTransformOp(opSuffix="geometryLayoutReview").Set(matrix)


def material_color(prim):
    try:
        material=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        if material:
            for shader_prim in Usd.PrimRange(material.GetPrim()):
                if not shader_prim.IsA(UsdShade.Shader): continue
                shader=UsdShade.Shader(shader_prim)
                for name in ("diffuseColor","diffuse_color_constant","baseColor","base_color","albedo"):
                    inp=shader.GetInput(name)
                    value=inp.Get() if inp else None
                    if value is not None and not isinstance(value,(str,float,int)):
                        arr=np.asarray(value,float)
                        if arr.shape==(3,) and np.isfinite(arr).all():
                            return np.clip(arr,0,1),str(shader_prim.GetPath())+".inputs:"+name
    except (ValueError,TypeError,RuntimeError):
        pass
    return None,None


def fallback_color(path):
    p=path.lower()
    if "skin" in p:return [0.62,0.40,0.30]
    if "scrubspants" in p:return [0.16,0.23,0.29]
    if "scrubsshirt" in p:return [0.19,0.34,0.41]
    if "sneakers" in p:return [0.16,0.17,0.18]
    if "/table" in p:return [0.50,0.48,0.44]
    if "/stool" in p:return [0.32,0.30,0.27]
    if "ground" in p:return [0.34,0.35,0.37]
    if "openarm" in p:return [0.27,0.29,0.32]
    return [0.5,0.52,0.55]


def mesh_geometry(prim):
    mesh=UsdGeom.Mesh(prim)
    vertices=np.asarray(mesh.GetPointsAttr().Get(),float)
    counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get(),int)
    indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),int)
    holes=set(map(int,mesh.GetHoleIndicesAttr().Get() or []))
    triangles=[];source_faces=[];corner_triangles=[];cursor=0
    for fi,n in enumerate(counts):
        face=indices[cursor:cursor+n]
        if fi not in holes:
            for j in range(1,n-1):
                triangles.append([face[0],face[j],face[j+1]])
                source_faces.append(fi);corner_triangles.append([cursor,cursor+j,cursor+j+1])
        cursor+=n
    triangles=np.asarray(triangles,np.int32).reshape(-1,3)
    if mesh.GetOrientationAttr().Get()==UsdGeom.Tokens.leftHanded:
        triangles=triangles[:,[0,2,1]]
    return vertices,triangles,np.asarray(source_faces,int),np.asarray(corner_triangles,int),len(counts)


def primitive_geometry(prim):
    typ=prim.GetTypeName()
    if typ=="Cube":
        obj=trimesh.creation.box(extents=[float(UsdGeom.Cube(prim).GetSizeAttr().Get())]*3)
    elif typ=="Sphere":
        obj=trimesh.creation.uv_sphere(radius=float(UsdGeom.Sphere(prim).GetRadiusAttr().Get()),count=[20,40])
    elif typ=="Cylinder":
        c=UsdGeom.Cylinder(prim);obj=trimesh.creation.cylinder(radius=float(c.GetRadiusAttr().Get()),height=float(c.GetHeightAttr().Get()),sections=48)
    elif typ=="Capsule":
        c=UsdGeom.Capsule(prim);h=float(c.GetHeightAttr().Get());r=float(c.GetRadiusAttr().Get())
        obj=trimesh.creation.uv_sphere(radius=r,count=[20,40]);v=np.asarray(obj.vertices).copy()
        v[:,2]+=np.where(v[:,2]>=0,h/2,-h/2);obj.vertices=v
    else:
        raise ValueError(typ)
    vertices=np.asarray(obj.vertices,float);faces=np.asarray(obj.faces,np.int32)
    if typ in ("Cylinder","Capsule"):
        axis=str(prim.GetAttribute("axis").Get())
        if axis=="X":vertices=vertices[:,[2,0,1]]
        elif axis=="Y":vertices=vertices[:,[1,2,0]]
        else:assert axis=="Z"
    return vertices,faces


def triangle_colors(prim,triangles,source_faces,corner_triangles,face_count):
    color,source=material_color(prim)
    if color is None:color,source=np.asarray(fallback_color(str(prim.GetPath()))),"semantic neutral fallback; no texture reconstruction"
    colors=np.tile(color,(len(triangles),1))
    pv=UsdGeom.PrimvarsAPI(prim).GetPrimvar("displayColor")
    if pv and pv.HasValue():
        values=np.asarray(pv.ComputeFlattened(),float)
        interp=str(pv.GetInterpolation())
        if values.ndim==2 and values.shape[1]==3 and np.isfinite(values).all():
            if len(values)==1:colors[:]=values[0];source="displayColor:constant"
            elif source_faces is not None and interp=="uniform" and len(values)==face_count:
                colors=values[source_faces];source="displayColor:uniform"
            elif interp in ("vertex","varying") and len(values)>int(triangles.max()):
                colors=values[triangles].mean(axis=1);source="displayColor:"+interp+" averaged over triangle"
            elif corner_triangles is not None and interp=="faceVarying" and len(values)>int(corner_triangles.max()):
                colors=values[corner_triangles].mean(axis=1);source="displayColor:faceVarying averaged over triangle"
    subsets=[]
    if source_faces is not None:
        for child in prim.GetChildren():
            if not child.IsA(UsdGeom.Subset):continue
            sub=UsdGeom.Subset(child)
            if str(sub.GetElementTypeAttr().Get())!="face":continue
            subcolor,subsource=material_color(child)
            if subcolor is not None:
                idx=np.asarray(sub.GetIndicesAttr().Get(),int)
                colors[np.isin(source_faces,idx)]=subcolor
                subsets.append({"path":str(child.GetPath()),"source":subsource,"faces":len(idx)})
    return np.clip(colors,0,1),source,subsets


def export(stage,stem,delta,base_metadata):
    cache=UsdGeom.XformCache();vertices=[];triangles=[];colors=[];normals=[];ids=[];rows=[];excluded=[];offset=0
    for prim in stage.Traverse():
        typ=prim.GetTypeName()
        if typ not in ("Mesh","Cube","Cylinder","Sphere","Capsule"):continue
        path=str(prim.GetPath());imageable=UsdGeom.Imageable(prim)
        purpose=str(imageable.ComputePurpose());visible=str(imageable.ComputeVisibility())!="invisible"
        if not visible or purpose not in ("default","render") or "/collisions/" in path or prim.GetName().endswith("Collision"):
            excluded.append({"path":path,"visible":visible,"purpose":purpose});continue
        if typ=="Mesh":v,f,faceids,corners,nfaces=mesh_geometry(prim)
        else:v,f=primitive_geometry(prim);faceids=corners=None;nfaces=None
        if not len(v) or not len(f):continue
        matrix=np.asarray(cache.GetLocalToWorldTransform(prim),float).T
        world=v@matrix[:3,:3].T+matrix[:3,3]
        if np.linalg.det(matrix[:3,:3])<0:f=f[:,[0,2,1]]
        color,color_source,subsets=triangle_colors(prim,f,faceids,corners,nfaces)
        normal=np.cross(world[f[:,1]]-world[f[:,0]],world[f[:,2]]-world[f[:,0]])
        lengths=np.linalg.norm(normal,axis=1);valid=lengths>1e-14
        f,color,normal=f[valid],color[valid],normal[valid]/lengths[valid,None]
        if not len(f):continue
        mesh_id=len(rows)
        rows.append({"mesh_id":mesh_id,"path":path,"type":typ,"vertex_count":len(v),"triangle_count":len(f),
            "vertex_offset":offset,"world_from_local":matrix.tolist(),"bounds_world_m":[world.min(0).tolist(),world.max(0).tolist()],
            "color_source":color_source,"material_subsets":subsets,
            "subdivision_scheme":str(UsdGeom.Mesh(prim).GetSubdivisionSchemeAttr().Get()) if typ=="Mesh" else None})
        vertices.append(world.astype(np.float32));triangles.append((f+offset).astype(np.int32));colors.append(color.astype(np.float32));normals.append(normal.astype(np.float32));ids.append(np.full(len(f),mesh_id,np.int32));offset+=len(v)
    arrays={"vertices":np.concatenate(vertices),"triangles":np.concatenate(triangles),"triangle_colors":np.concatenate(colors),
        "triangle_normals":np.concatenate(normals),"triangle_mesh_id":np.concatenate(ids)}
    assert np.isfinite(arrays["vertices"]).all() and arrays["triangles"].max()<len(arrays["vertices"])
    dest=OUT/(stem+".npz");np.savez_compressed(dest,**arrays)
    metadata={**base_metadata,"variant":stem,"patient_translation_world_m":list(delta),"camera":camera_state(stage),
        "targets":[{"name":n,"position_world":(np.asarray(json.loads((PLANS/f"planned_paths_{n}.json").read_text())["points"][0]["position_world"])+delta).tolist()} for n in NAMES],
        "mesh_file":dest.name,"mesh_file_sha256":sha(dest),"meshes":rows,"excluded_meshes":excluded,
        "counts":{"vertices":len(arrays["vertices"]),"triangles":len(arrays["triangles"]),"meshes":len(rows)},
        "bounds_world_m":{n:bounds(stage,"/World/"+n).tolist() for n in ("Patient","Stool","Table")}}
    (OUT/(stem+".json")).write_text(json.dumps(metadata,indent=2)+"\n")
    print(stem,metadata["counts"],flush=True)
    return metadata


def main():
    OUT.mkdir(exist_ok=False)
    inputs=[SOURCE,Path(__file__),*[PLANS/f"planned_paths_{n}.json" for n in NAMES]]
    hashes={str(p.relative_to(PROJECT)):sha(p) for p in inputs}
    print("Opening source USD once",flush=True)
    stage=Usd.Stage.Open(str(SOURCE));assert stage
    assert UsdGeom.GetStageMetersPerUnit(stage)==1 and str(UsdGeom.GetStageUpAxis(stage))=="Z"
    stage.SetEditTarget(stage.GetSessionLayer())
    camera=camera_state(stage)
    initial={n:bounds(stage,"/World/"+n) for n in ("Patient","Stool","Table")}
    seat=bounds(stage,"/World/Stool/seat");legs={p.GetName():bounds(stage,str(p.GetPath()))[0,2] for p in stage.GetPrimAtPath("/World/Stool").GetChildren() if p.GetName().startswith("leg")}
    base_metadata={"schema":"usd_geometry_camera_preview_v1","scope":"Actual visible authored control meshes with exact original RobotCam projection; geometric preview only, not Isaac rendered appearance, material/texture reproduction, physics or trajectory validation.",
        "color_encoding":"Authored displayColor or material constants in scene-linear RGB when available; documented neutral fallback otherwise; textures and lighting omitted.",
        "source_sha256":hashes,"meters_per_unit":1,"up_axis":"Z","arm_pose":"Authored static source scene pose unchanged in both layouts.",
        "topology":"Mesh polygons fan triangulated, holes skipped, no subdivision; primitive tessellation approximates analytic surfaces.",
        "camera":camera}
    a=export(stage,"baseline",[0.,0.,0.],base_metadata)
    delta=[0.,.05,-.02]
    move_world(stage,"/World/Patient",delta);move_world(stage,"/World/Stool",[0.,.05,0.])
    move_local_z(stage.GetPrimAtPath("/World/Stool/seat"),-.02)
    for child in stage.GetPrimAtPath("/World/Stool").GetChildren():
        if child.GetName().startswith("leg"):
            cylinder=UsdGeom.Cylinder(child);assert cylinder and str(cylinder.GetAxisAttr().Get())=="Z"
            height=float(cylinder.GetHeightAttr().Get())-.02;assert height>0
            cylinder.GetHeightAttr().Set(height);move_local_z(child,-.01)
        elif child.GetName().startswith("rung"):
            z=float(UsdGeom.Xformable(child).GetLocalTransformation().ExtractTranslation()[2]);move_local_z(child,z*(-.02)/seat[1,2])
    assert camera_state(stage)==camera
    np.testing.assert_allclose(bounds(stage,"/World/Patient"),initial["Patient"]+delta,atol=1e-6,rtol=0)
    np.testing.assert_allclose(bounds(stage,"/World/Table"),initial["Table"],atol=1e-9,rtol=0)
    np.testing.assert_allclose(bounds(stage,"/World/Stool/seat"),seat+delta,atol=1e-6,rtol=0)
    for name,z in legs.items():assert abs(bounds(stage,"/World/Stool/"+name)[0,2]-z)<1e-6
    b=export(stage,"farther5_lower2",delta,base_metadata)
    assert a["camera"]==b["camera"]
    unchanged=all(sha(PROJECT/p)==h for p,h in hashes.items());assert unchanged
    (OUT/"verification.json").write_text(json.dumps({"accepted":True,"source_files_unchanged":unchanged,
        "camera_exactly_unchanged":True,"patient_translation_matches":True,"table_unchanged":True,"seat_translation_matches":True,
        "stool_leg_foot_height_unchanged":True,"scope":"Static geometry export invariants only.","inputs":hashes},indent=2)+"\n")


if __name__=="__main__":main()
