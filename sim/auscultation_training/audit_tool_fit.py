"""Search finite-head geometry at the retained targets; no robot motion or data acceptance."""
import argparse,json
import numpy as np
import fcl
from .scene import ROOT,FIXED,MOVING

def merged_object(entries,arrays):
    vertices=[];faces=[];offset=0
    for e in entries:
        v=arrays[e['key']+'_v'];f=arrays[e['key']+'_f']
        vertices.append(v);faces.append(f+offset);offset+=len(v)
    v=np.concatenate(vertices);f=np.concatenate(faces)
    m=fcl.BVHModel();m.beginModel(len(v),len(f));m.addSubModel(v,f);m.endModel()
    return fcl.CollisionObject(m)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--point',default='R1');parser.add_argument('--compression',type=float,default=.004);args=parser.parse_args()
    meta=json.loads((ROOT/'results/collision_geometry.json').read_text());arr=np.load(ROOT/'results/collision_geometry.npz')
    es=meta['colliders'];patient=merged_object([e for e in es if e['path'].startswith('/World/Patient/')],arr)
    fixed=merged_object([e for e in es if e['body']==FIXED],arr)
    shell=merged_object([e for e in es if e['body']==MOVING and not(e['path'].endswith('/Diaphragm') or '/RetentionLip/' in e['path'] or '/ContactRim/' in e['path'])],arr)
    face=merged_object([e for e in es if e['path'].endswith('/Diaphragm')],arr)
    points=json.loads((ROOT/'results/contact_targets.json').read_text())['frames']['raised']['points']
    target=next(p for p in points if p['name']==args.point)
    p=np.array(target['world_xyz_m']);n=np.array(target['normal_world_from_depth'])
    x=np.cross(n,[0,0,1]);x/=np.linalg.norm(x);y=np.cross(n,x)
    candidates=[];best=-1;stats={'fixed_collision':0,'shell_collision':0,'membrane_gap':0,'passed':0}
    for u in np.linspace(-.6,.6,13):
        for v in np.linspace(-.6,.6,13):
            normal=n+u*x+v*y;normal/=np.linalg.norm(normal)
            tangent=x-normal*(x@normal);tangent/=np.linalg.norm(tangent);bitangent=np.cross(normal,tangent)
            for roll in np.linspace(0,2*np.pi,36,endpoint=False):
                a=np.cos(roll)*tangent+np.sin(roll)*bitangent;b=np.cross(normal,a);R=np.stack((a,b,normal),axis=1)
                # Keep membrane center at current target; original rendered depth
                # differs from collider by about 1mm. Offset is reported, not hidden.
                for gap in [0.,.0015,.003,.005]:
                    center=p+gap*n
                    fixed.setTransform(fcl.Transform(R,center+normal*(.0412-args.compression)))
                    shell.setTransform(fcl.Transform(R,center+normal*.0412))
                    face.setTransform(fcl.Transform(R,center+normal*.0412))
                    df=fcl.distance(fixed,patient)
                    if df<.001:stats['fixed_collision']+=1;continue
                    ds=fcl.distance(shell,patient)
                    if ds<.0003:stats['shell_collision']+=1;continue
                    dm=fcl.distance(face,patient)
                    if dm>.002:stats['membrane_gap']+=1;continue
                    angle=float(np.arccos(np.clip(normal@n,-1,1)))
                    candidates.append({'normal_world':normal.tolist(),'wrist_rotation_world':(R@np.diag([1,-1,-1])).tolist(),
                                       'gap_m':gap,'angle_change_deg':float(np.rad2deg(angle)),'fixed_clearance_m':df,
                                       'shell_clearance_m':ds,'face_distance_m':dm})
                    stats['passed']+=1
        print('FIT_PROGRESS',args.point,u,stats,flush=True)
    candidates.sort(key=lambda c:(c['angle_change_deg'],-c['shell_clearance_m']))
    result={'name':args.point,'compression_m':args.compression,'scope':'geometry hypotheses; none dynamically validated','stats':stats,'candidates':candidates}
    (ROOT/'results'/('tool_fit_'+args.point+'_'+str(round(args.compression*1000,2))+'mm.json')).write_text(json.dumps(result,indent=2))
    print('FIT_DONE',args.point,len(candidates),candidates[:1],flush=True)

if __name__=='__main__':main()
