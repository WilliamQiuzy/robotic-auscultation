"""V6A anchors on posed SMPL surfaces; numeric asset pilot, no simulator rendering."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def evaluate_anchors(vertices, faces, anchors):
    vertices=np.asarray(vertices,float);faces=np.asarray(faces)
    if (vertices.ndim!=2 or vertices.shape[1]!=3 or not np.isfinite(vertices).all()
            or faces.ndim!=2 or faces.shape[1]!=3 or faces.dtype.kind not in 'iu'
            or faces.min()<0 or faces.max()>=len(vertices)):
        raise ValueError('Invalid surface topology')
    topology={tuple(sorted(f)):f for f in faces.tolist()};result={}
    for name,anchor in anchors.items():
        ids=np.asarray(anchor['smpl_vertex_indices']);weights=np.asarray(anchor['barycentric_weights'],float)
        if (ids.shape!=(3,) or ids.dtype.kind not in 'iu' or len(set(ids))!=3
                or weights.shape!=(3,) or not np.isfinite(weights).all()
                or np.min(weights)<0 or abs(weights.sum()-1)>1e-10
                or tuple(sorted(ids)) not in topology):raise ValueError('Anchor is not on an actual SMPL triangle: '+name)
        ordered=vertices[topology[tuple(sorted(ids))]]
        normal=np.cross(ordered[1]-ordered[0],ordered[2]-ordered[0]);area2=np.linalg.norm(normal)
        if area2<1e-12:raise ValueError('Degenerate posed anchor triangle')
        result[name]=dict(position_m=(weights@vertices[ids]).tolist(),normal=(normal/area2).tolist(),
            triangle_area_m2=float(area2/2))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--identity',type=Path,required=True);p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    import smplx
    import torch
    torch.set_num_threads(2)
    inventory=json.loads(args.inventory.read_text());quality=json.loads((args.identity/'identity_quality.json').read_text())
    split=json.loads((args.identity/'identity_split.json').read_text())
    split_sha=hashlib.sha256((args.identity/'identity_split.json').read_bytes()).hexdigest()
    if split_sha!=quality['identity_split_sha256']:raise ValueError('Identity split changed')
    train=[i for i in split['splits']['train'] if i in quality['eligible_shape_betas']]
    chosen=np.random.default_rng(20260930).choice(train,8,replace=False).tolist()
    model=smplx.SMPL(model_path=str(args.model),gender='neutral',batch_size=1,num_betas=10)
    poses={'rest':np.zeros(69,dtype=np.float32),'seated_asset':np.zeros(69,dtype=np.float32),
           'seated_lean_asset':np.zeros(69,dtype=np.float32)}
    for label in ('seated_asset','seated_lean_asset'):
        poses[label][0]=poses[label][3]=-1.25
        poses[label][9]=poses[label][12]=1.4
    poses['seated_lean_asset'][6]=.12
    records=[]
    with torch.no_grad():
        for number,identity in enumerate(chosen):
            beta=torch.tensor(quality['eligible_shape_betas'][identity],dtype=torch.float32).reshape(1,10)
            for label,pose in poses.items():
                value=model(betas=beta,body_pose=torch.tensor(pose).reshape(1,69),return_verts=True)
                vertices=value.vertices[0].numpy();joints=value.joints[0].numpy()
                anchored=evaluate_anchors(vertices,model.faces,inventory['anchors'])
                filename=f'body{number:02d}_{label}.npz'
                np.savez_compressed(args.output/filename,vertices=vertices,faces=model.faces,joints=joints,
                    anchor_positions=np.array([a['position_m'] for a in anchored.values()]),
                    anchor_normals=np.array([a['normal'] for a in anchored.values()]),
                    anchor_names=np.array(list(anchored)))
                records.append(dict(identity=identity,pose=label,file=filename,
                    sha256=hashlib.sha256((args.output/filename).read_bytes()).hexdigest(),anchors=anchored))
    summary=dict(identities=8,split='train only',posed_bodies=len(records),anchors_per_body=len(inventory['anchors']),
        all_anchors_on_actual_smpl_faces=True,all_normals_finite_unit=True,
        identity_split_sha256=split_sha,smpl_neutral_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
        anchor_inventory_sha256=hashlib.sha256(args.inventory.read_bytes()).hexdigest(),
        coordinates='SMPL model metres; not placed into approved Isaac scene',
        pose_status='unqualified synthetic asset poses; no sitting/foot/clothing/robot reachability claims',
        isaac_rendered=False,expert_generated=False,training_eligible=False)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (args.output/'asset_manifest.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
