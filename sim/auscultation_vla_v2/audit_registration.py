"""Offline perception diagnostic. Saved-pose images are never training episodes.

The polygon below is a manual image annotation of visible torso support in the
approved diagnostic camera, not a simulator segmentation mask. Reference truth
positions are read ONLY after registration, for scoring, and initial marked
pixels proxy an operator annotation. No result authorizes robot movement.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from .rigid_registration import register

POLYGON=np.array([[650,350],[1180,350],[1250,620],[1150,1040],
                  [700,1040],[590,620]],np.int32)


def cloud(depth,valid,k,polygon=POLYGON,stride=8):
    depth=np.asarray(depth);valid=np.asarray(valid)
    if depth.ndim!=2 or depth.shape!=valid.shape or valid.dtype!=bool:
        raise ValueError('bad depth validity pairing')
    y,x=np.mgrid[:depth.shape[0],:depth.shape[1]]
    safe=np.where(valid&np.isfinite(depth)&(depth>0),depth,0.)
    xyz=np.stack([(x-k[0,2])*safe/k[0,0],(y-k[1,2])*safe/k[1,1],safe],axis=-1)
    a=xyz[2:,1:-1]-xyz[:-2,1:-1];b=xyz[1:-1,2:]-xyz[1:-1,:-2]
    n=np.cross(a,b);length=np.linalg.norm(n,axis=-1)
    n/=np.maximum(length[...,None],1e-12)
    mask=np.zeros(depth.shape,np.uint8);cv2.fillPoly(mask,[polygon],1)
    ok=mask[1:-1,1:-1].astype(bool)&valid[1:-1,1:-1]
    ok&=valid[2:,1:-1]&valid[:-2,1:-1]&valid[1:-1,2:]&valid[1:-1,:-2]
    ok&=(length>1e-10)&(np.linalg.norm(a,axis=-1)<.008)&(np.linalg.norm(b,axis=-1)<.008)
    keep=ok[::stride,::stride]
    return xyz[1:-1,1:-1][::stride,::stride][keep],n[::stride,::stride][keep]


def run(source,output):
    source=Path(source);manifest=json.loads((source/'rgbd_manifest.json').read_text())
    k=manifest['intrinsics'];K=np.array([[k['fx'],0,k['cx']],[0,k['fy'],k['cy']],[0,0,1.]])
    first=manifest['samples'][0]
    def load(sample):
        return cloud(np.load(source/sample['files']['depth_m']),
                     np.load(source/sample['files']['valid_mask']).astype(bool),K)
    p,pn=load(first);rows=[]
    for sample in manifest['samples']:
        row=dict(capture_id=sample['capture_id'],target=sample['target'],kind=sample['kind'])
        try:
            q,qn=load(sample);fit=register(p,pn,q,qn)
            # Body is static in these saved poses. This score is not a motion test.
            moved=p@fit['transform'][:3,:3].T+fit['transform'][:3,3]
            row.update(accepted=True,static_drift_p95_m=float(np.quantile(np.linalg.norm(moved-p,axis=1),.95)),
                       **{key:value for key,value in fit.items() if key not in ('transform','reference_indices','current_indices')})
        except ValueError as error:row.update(accepted=False,reason=str(error))
        rows.append(row)
        print(json.dumps(row),flush=True)
    result=dict(schema='wam_rigid_registration_diagnostic_v1',source=str(source),
        scope='Static saved-pose depth registration only; no live tracking, no RGB gating, no training data',
        support_polygon_px=POLYGON.tolist(),total=len(rows),accepted=sum(r['accepted'] for r in rows),rows=rows)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.source,args.output)
