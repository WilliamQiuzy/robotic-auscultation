"""Pure CPU projection screen for camera-height candidates; no occlusion claim."""
from pathlib import Path
import hashlib, json, math
import numpy as np
P=Path(__file__).resolve().parent
base=json.loads((P/'comparison_v1/comparison.json').read_text()); K=base['intrinsics']
O=np.asarray(base['variants']['down_0deg']['world_from_optical'],float)
samples=np.load(P/'back_roi_samples.npz'); center=samples['center']; roi=samples['roi']
layout=json.loads((P.parent/'auscultation_expert_layout_v1/layout_inputs.json').read_text())
targets=layout['approved_variant']['projected_existing_points']; names=[x['name'] for x in targets]; points=np.asarray([x['world_position_m'] for x in targets])
W,H=K['width'],K['height']
def project(p,o):
 q=np.c_[p,np.ones(len(p))]@np.linalg.inv(o).T; uv=q[:,:2]/q[:,2,None]*[K['fx'],K['fy']]+[K['cx'],K['cy']]
 ok=(q[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<W)&(uv[:,1]>=0)&(uv[:,1]<H)
 return uv,ok
results=[]
for dz in [.05,.08,.10]:
 for deg in np.arange(8.,23.,1.):
  r=math.radians(-deg);t=np.eye(4);t[1:3,1:3]=[[math.cos(r),-math.sin(r)],[math.sin(r),math.cos(r)]]
  pose=O@t;pose[2,3]+=dz
  uv,ok=project(points,pose); ur,kr=project(roi,pose);uc,kc=project(center,pose)
  upper=roi[:,2]>1.43; up=ur[upper]; margin=np.minimum.reduce([uv[:,0],W-1-uv[:,0],uv[:,1],H-1-uv[:,1]])
  cv=center[kc]
  results.append(dict(height_increase_m=dz,downward_pitch_delta_deg=float(deg),camera_height_m=float(pose[2,3]),look_elevation_deg=float(math.degrees(math.asin(pose[2,2]))),world_from_optical=pose.tolist(),world_from_camera_usd=(pose@np.diag([1.,-1.,-1.,1.])).tolist(),targets_inside=int(ok.sum()),target_minimum_margin_px=float(margin.min()),target_y_range_px=[float(uv[:,1].min()),float(uv[:,1].max())],point_uv=[dict(name=n,uv_px=v.tolist())for n,v in zip(names,uv)],roi_frustum_fraction=float(kr.mean()),center_frustum_fraction=float(kc.mean()),center_back_visible_world_z_range_m=[float(cv[:,2].min()),float(cv[:,2].max())],upper_central_shirt_minimum_y_px=float(up[:,1].min()),upper_central_shirt_minimum_side_margin_px=float(np.minimum(up[:,0],W-1-up[:,0]).min()),upper_central_shirt_scope='Central posterior cloth samples at world Z>1.43m, not head mesh nor anatomical landmarks.'))
res=dict(schema='elevated_camera_geometric_candidates_v1',scope='Pure geometric frustum screen only; no rendered-depth occlusion results, no new physics run. Original XY/lens/patient/robot are unchanged. Upper-edge proxy is central shirt only, NOT full head.',source_sha256={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [P/'comparison_v1/comparison.json',P/'back_roi_samples.npz',Path(__file__)]},results=results)
(P/'elevated_camera_math.json').write_text(json.dumps(res,indent=2)+'\n')
for dz in [.05,.08,.10]:
 for r in results:
  if r['height_increase_m']==dz and r['upper_central_shirt_minimum_y_px']>=120 and r['roi_frustum_fraction']>.75:
   print('dz',dz,'down',r['downward_pitch_delta_deg'],'roi%',round(r['roi_frustum_fraction']*100,1),'center%',round(r['center_frustum_fraction']*100,1),'upperedge',round(r['upper_central_shirt_minimum_y_px'],1),'pointmargin',round(r['target_minimum_margin_px'],1))
