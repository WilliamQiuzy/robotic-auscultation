"""Estimate finite-footprint contact normals from the existing RGB-D observation."""
import json,copy
from pathlib import Path
import numpy as np
from .scene import ROOT

def fit_normal(points, center, outward, radius=.025):
    points=np.asarray(points);center=np.asarray(center);outward=np.asarray(outward)
    delta=points-center
    along=delta@outward
    radial=np.linalg.norm(delta-along[:,None]*outward,axis=1)
    selected=np.isfinite(points).all(axis=1)&(radial<radius)&(abs(along)<.025)
    cloud=points[selected]
    if len(cloud)<50:raise ValueError('Insufficient connected depth samples')
    weight=np.exp(-.5*(radial[selected]/(.6*radius))**2)
    mean=np.average(cloud,axis=0,weights=weight)
    centered=cloud-mean
    cov=(centered*weight[:,None]).T@centered/weight.sum()
    eig,vec=np.linalg.eigh(cov);normal=vec[:,0]
    if normal@outward<0:normal=-normal
    return normal, {'radius_m':radius,'samples':len(cloud),'plane_rms_m':float(np.sqrt(eig[0])),
                    'change_deg':float(np.rad2deg(np.arccos(np.clip(normal@outward,-1,1))))}

def main():
    assets=ROOT.parent/'auscultation_assets_20260922'
    pred=json.loads((assets/'model_predictions.json').read_text())
    manifest=json.loads((assets/'asset_manifest.json').read_text())
    depth=np.load(assets/'raised_depth_m.npy')
    k=manifest['intrinsics_rendered'];T=np.array(manifest['world_from_camera_optical'])
    for point in pred['frames']['raised']['points']:
        u,v=map(int,point['uv_px']);radius=110
        yy,xx=np.mgrid[max(0,v-radius):min(depth.shape[0],v+radius+1),max(0,u-radius):min(depth.shape[1],u+radius+1)]
        z=depth[yy,xx]
        camera=np.stack(((xx-k['cx'])*z/k['fx'],(yy-k['cy'])*z/k['fy'],z),axis=-1).reshape(-1,3)
        world=camera@T[:3,:3].T+T[:3,3]
        original=point['normal_world_from_depth']
        normal,quality=fit_normal(world,point['world_xyz_m'],original)
        point['normal_world_point_patch']=original
        point['normal_world_from_depth']=normal.tolist()
        point['contact_normal_fit']=quality
        print(point['name'],quality,flush=True)
    pred['normal_source']='Gaussian-weighted PCA over 25 mm radius RGB-D footprint; no collision mesh input'
    (ROOT/'results/contact_targets.json').write_text(json.dumps(pred,indent=2))

if __name__=='__main__':main()
