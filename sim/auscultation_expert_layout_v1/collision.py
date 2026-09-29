"""Independent FCL collision/clearance checks over exported USD collider meshes."""
import json
from pathlib import Path
import numpy as np
import fcl
from .scene import ROBOT, WRIST, FIXED, MOVING, ROOT

class CollisionChecker:
    def __init__(self, kinematics, world_from_base, directory=None):
        self.kinematics=kinematics;self.base=np.asarray(world_from_base)
        directory=Path(directory or ROOT/'results')
        metadata=json.loads((directory/'collision_geometry.json').read_text())
        arrays=np.load(directory/'collision_geometry.npz')
        self.entries=metadata['colliders'];self.objects=[];self.local_corners=[];self.bounds=[]
        self.allowed={frozenset(x) for x in metadata['allowed_body_pairs']}
        self.dynamic=[]
        for i,e in enumerate(self.entries):
            v=np.asarray(arrays[e['key']+'_v'],dtype=np.float64)
            faces=np.asarray(arrays[e['key']+'_f'],dtype=np.int32)
            geom=fcl.BVHModel();geom.beginModel(len(v),len(faces));geom.addSubModel(v,faces);geom.endModel()
            T=np.asarray(e['world_from_body'])
            self.objects.append(fcl.CollisionObject(geom,fcl.Transform(T[:3,:3],T[:3,3])))
            from itertools import product
            corners=np.array(list(product(*zip(v.min(axis=0),v.max(axis=0)))))
            self.local_corners.append(corners)
            points=corners@T[:3,:3].T+T[:3,3]
            self.bounds.append([points.min(axis=0),points.max(axis=0)])
            if ('/openarm_left_link' in e['body'] and not e['body'].endswith('link0')) or e['body'] in (FIXED,MOVING):self.dynamic.append(i)
        self.pairs=[]
        for i in self.dynamic:
            for j,e in enumerate(self.entries):
                if j==i or (j in self.dynamic and j<i):continue
                a,b=self.entries[i]['body'],e['body']
                if a==b or frozenset((a,b)) in self.allowed:continue
                self.pairs.append((i,j))
        self.bounds=np.asarray(self.bounds);self.pair_indices=np.asarray(self.pairs)

    def update(self,q,compression=0.):
        frames={ROBOT+'/'+name:self.base@T for name,T in self.kinematics.link_transforms(q).items()}
        adapter=np.diag([1.,-1.,-1.,1.]);adapter[2,3]=.085
        frames[FIXED]=frames[WRIST]@adapter
        slider=np.eye(4);slider[2,3]=compression
        frames[MOVING]=frames[FIXED]@slider
        for i in self.dynamic:
            T=frames[self.entries[i]['body']]
            self.objects[i].setTransform(fcl.Transform(T[:3,:3],T[:3,3]))
            points=self.local_corners[i]@T[:3,:3].T+T[:3,3]
            self.bounds[i]=[points.min(axis=0),points.max(axis=0)]

    def check(self,q,clearance=.003,allow_patient_contact=False,compression=0.,details=False):
        self.update(q,compression)
        # AABB distance is a conservative lower bound. Only nearby pairs require
        # triangle checks. Report a certified floor capped at requested clearance.
        worst={'clear':True,'minimum_distance_m':clearance,'pair':None}
        ia,ib=self.pair_indices.T
        gap=np.maximum(0.,np.maximum(self.bounds[ia,0]-self.bounds[ib,1],self.bounds[ib,0]-self.bounds[ia,1]))
        near=np.linalg.norm(gap,axis=1)<clearance+1e-9
        for i,j in self.pair_indices[near]:
            a,b=self.entries[i],self.entries[j]
            contact_face=a['path'].endswith('/Diaphragm') or '/RetentionLip/' in a['path'] or '/ContactRim/' in a['path']
            if allow_patient_contact and contact_face and b['path'].startswith('/World/Patient/'):continue
            # FCL unsigned distance is exact for the exported triangle surfaces.
            result=fcl.DistanceResult()
            d=float(fcl.distance(self.objects[i],self.objects[j],fcl.DistanceRequest(),result))
            if d<worst['minimum_distance_m']:
                worst.update(minimum_distance_m=d,pair=[a['path'],b['path']])
            if d<clearance:
                worst['clear']=False
                if not details:return worst
        return worst

    def edge(self,start,end,clearance=.003,allow_patient_contact=False,joint_step=.015):
        start,end=np.asarray(start),np.asarray(end)
        n=max(1,int(np.ceil(np.max(np.abs(end-start))/joint_step)))
        minimum=float('inf')
        for u in np.linspace(0,1,n+1):
            r=self.check(start+(end-start)*u,clearance,allow_patient_contact)
            minimum=min(minimum,r['minimum_distance_m'])
            if not r['clear']:return {**r,'fraction':float(u),'samples':n+1}
        return {'clear':True,'minimum_distance_m':minimum,'samples':n+1}
