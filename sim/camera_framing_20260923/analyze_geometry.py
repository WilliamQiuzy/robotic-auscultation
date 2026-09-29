import json, math, hashlib
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom
ROOT=Path(__file__).resolve().parent
P=ROOT.parents[1]
scene=P/'sim/auscultation_expert_layout_v1/results/expert_v2/physics_all10/training_scene.usdc'
a=json.load(open(P/'sim/auscultation_expert_layout_v1/layout_inputs.json'))
s=Usd.Stage.Open(str(scene)); cam=UsdGeom.Camera(s.GetPrimAtPath('/World/RobotCam'))
world=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(cam.GetPrim()),float).T
opt=world@np.diag([1.,-1.,-1.,1.]); orig=np.array(a['camera']['world_from_optical'])
assert np.allclose(opt,orig,atol=1e-12)
p=next(p for p in s.Traverse() if p.IsA(UsdGeom.Mesh) and str(p.GetPath()).endswith('opaque__fabric__scrubsshirt'))
m=UsdGeom.Mesh(p); verts=np.asarray(m.GetPointsAttr().Get(),float); mat=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(p),float)
verts=(np.c_[verts,np.ones(len(verts))]@mat)[:,:3]
counts=np.array(m.GetFaceVertexCountsAttr().Get()); ids=np.array(m.GetFaceVertexIndicesAttr().Get()); tris=[]; off=0
for count in counts:
 face=ids[off:off+count]; off+=count
 for k in range(1,count-1):tris.append([face[0],face[k],face[k+1]])
tris=verts[np.asarray(tris)]
# Posterior control-mesh surface: raycast along +worldY at each x/z sample,
# taking the first triangle intersection (minimum Y). This is geometric sampling,
# not rasterization and does not claim anatomical landmarks or occlusion visibility.
xz=tris[:,:,[0,2]]
e1=xz[:,1]-xz[:,0]; e2=xz[:,2]-xz[:,0]
den=e1[:,0]*e2[:,1]-e2[:,0]*e1[:,1]
nonzero=np.abs(den)>1e-10
safe_den=np.where(nonzero,den,1)
def posterior(x,z):
 delta=np.array([x,z])-xz[:,0]
 u=(delta[:,0]*e2[:,1]-e2[:,0]*delta[:,1])/safe_den
 v=(e1[:,0]*delta[:,1]-delta[:,0]*e1[:,1])/safe_den
 w=1-u-v
 hit=nonzero&(u>=-1e-7)&(v>=-1e-7)&(w>=-1e-7)
 if not hit.any():return None
 ys=w[hit]*tris[hit,0,1]+u[hit]*tris[hit,1,1]+v[hit]*tris[hit,2,1]
 return float(ys.min())
zmin,zmax=float(verts[:,2].min()),float(verts[:,2].max()); zs=np.linspace(zmin+.0001,zmax-.0001,1201)
center=np.array([[0,y,z] for z in zs if (y:=posterior(0,z)) is not None])
# Equal-width and equal-world-height samples on a deliberately defined central shirt-back proxy.
# Surface-area fraction not computed; fraction is x/z projected sample count.
roi=np.array([[x,y,z] for z in np.linspace(zmin+.005,zmax-.005,301) for x in np.linspace(-.15,.15,61) if (y:=posterior(x,z)) is not None])
points=a['approved_variant']['projected_existing_points']; pw=np.array([p['world_position_m'] for p in points]); names=[p['name'] for p in points]
K=a['camera']; W,H=K['width'],K['height']
def project(pts,pose):
 q=np.c_[pts,np.ones(len(pts))]@np.linalg.inv(pose).T
 uv=q[:,:2]/q[:,2,None]*[K['fx'],K['fy']]+[K['cx'],K['cy']]
 mask=(q[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<W)&(uv[:,1]>=0)&(uv[:,1]<H)
 return uv,mask
results=[]
for d in [0,5,8,10,12]:
 # Optical x points right, y points down, z looks forward. Negative optical-X rotation
 # rotates +Z toward +Y (down). The same local negative-X rotation applies to USD camera.
 ang=math.radians(-d); c,ss=math.cos(ang),math.sin(ang)
 R=np.array([[1,0,0,0],[0,c,-ss,0],[0,ss,c,0],[0,0,0,1.]])
 O=opt@R
 uv,ok=project(pw,O); uc,kc=project(center,O); ur,kr=project(roi,O)
 cv=center[kc]; margin=np.minimum.reduce([uv[:,0],W-uv[:,0],uv[:,1],H-uv[:,1]])
 results.append(dict(down_pitch_deg=d,world_from_optical=O.tolist(),optical_axis_world=O[:3,2].tolist(),point_uv=[dict(name=n,uv_px=v.tolist(),minimum_margin_px=float(mm),inside_frame=bool(k)) for n,v,k,mm in zip(names,uv,ok,margin)],point_minimum_margin_px=float(margin.min()),point_y_range_px=[float(uv[:,1].min()),float(uv[:,1].max())],center_back_world_z_range_m=[float(center[:,2].min()),float(center[:,2].max())],center_back_framed_world_z_range_m=[float(cv[:,2].min()),float(cv[:,2].max())],center_back_height_fraction_in_frame=float((cv[:,2].max()-cv[:,2].min())/(center[:,2].max()-center[:,2].min())),center_back_samples_fraction_in_frame=float(kc.mean()),central_shirt_proxy_samples_fraction_in_frame=float(kr.mean()),whole_shirt_projected_bbox_px=[project(verts,O)[0].min(0).tolist(),project(verts,O)[0].max(0).tolist()]))
res=dict(schema='camera_framing_geometric_math_v1',scene=str(scene),scene_sha256=hashlib.sha256(scene.read_bytes()).hexdigest(),camera_verified_against_layout_inputs=True,camera_original=K,shirt_path=str(p.GetPath()),shirt_world_bounds_m=[verts.min(0).tolist(),verts.max(0).tolist()],shirt_geometry_scope='Authored USD visible control-mesh triangles; no subdivision smoothing and no rasterization.',center_back_definition='Closest shirt control-mesh triangle surface along +world Y, at world X=0 for a uniform Z sweep. Bounds are cloth-derived, not anatomical landmarks.',central_shirt_proxy_definition='Closest shirt triangle surface along +world Y at world X in [-.15,.15] m and Z in [shirt_min_z+.005,shirt_max_z-.005], uniformly sampled in X/Z. This is a central posterior cloth patch, NOT clinical/anatomical full-back ground truth. Values measure frustum coverage only, not occlusion.',center_count=len(center),roi_count=len(roi),rotation='Use negative local X angle. World_from_optical_new = world_from_optical_old @ Rx(-angle). World_from_USD_new = world_from_USD_old @ Rx(-angle); USD/Gf matrix is transpose. Camera center and lens are unchanged.',results=results)
np.savez_compressed(ROOT/'back_roi_samples.npz', center=center, roi=roi)
(ROOT/'geometric_math.json').write_text(json.dumps(res,indent=2)+'\n')
for r in results:print(r['down_pitch_deg'],'points margin',round(r['point_minimum_margin_px'],1),'y',np.round(r['point_y_range_px'],1),'center height',round(r['center_back_height_fraction_in_frame']*100,1),'proxy',round(r['central_shirt_proxy_samples_fraction_in_frame']*100,1),'center visibleZ',np.round(r['center_back_framed_world_z_range_m'],3))
