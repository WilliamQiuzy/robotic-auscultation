"""Fresh finite-head surface fitting; no previous plans, IK or motion."""
import json
from pathlib import Path
import numpy as np


def component_frames(center, wrist_rotation, compression):
    """Actual new diaphragm centre to fixed and sliding component frames."""
    center=np.asarray(center,float);rotation=np.asarray(wrist_rotation,float)
    if (center.shape!=(3,) or rotation.shape!=(3,3) or not np.isfinite(center).all()
            or not np.isfinite(rotation).all() or not np.isfinite(compression)
            or not 0<=compression<=.006
            or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-8,rtol=0)
            or not np.isclose(np.linalg.det(rotation),1.,atol=1e-8,rtol=0)):
        raise ValueError('Invalid current-geometry contact frame')
    fixed=np.eye(4);moving=np.eye(4)
    fixed[:3,:3]=moving[:3,:3]=rotation@np.diag([1.,-1.,-1.])
    outward=-rotation[:,2]
    fixed[:3,3]=center+outward*(.047-compression)
    moving[:3,3]=center+outward*.047
    return fixed,moving


class FiniteHeadFit:
    """Same finite-head fit gates, rebuilt from the current exported meshes.

    Fitted tool normal is distinct from local triangle/averaged normal. It is
    the fitted normal that the retained controller tracks within five degrees,
    as in the existing audit_tool_fit -> command_normal_world contract.
    """
    def __init__(self,directory,metadata):
        from .geometry_epoch import require_geometry_artifacts
        self.artifact_hashes=require_geometry_artifacts(directory,metadata)
        import fcl
        self.fcl=fcl
        directory=Path(directory)
        entries=json.loads((directory/'collision_geometry.json').read_text())['colliders']
        arrays=np.load(directory/'collision_geometry.npz',allow_pickle=False)
        groups={k:[] for k in ('patient','fixed','clamp','shell','diaphragm')}
        for e in entries:
            path=e['path'];group=None
            if path.startswith('/World/Patient/'):group='patient'
            elif e['body']=='/World/Stethoscope/Fixed':group='fixed'
            elif e['body']=='/World/Stethoscope/Carriage':
                suffix=path.split('/')[-1]
                group=('diaphragm' if suffix=='Diaphragm' else 'shell'
                       if suffix in ('DeviceEdge','DeviceCone','DeviceDial') else 'clamp')
            if group is None:continue
            vertices=np.asarray(arrays[e['key']+'_v'],float)
            faces=np.asarray(arrays[e['key']+'_f'],np.int32)
            if group=='patient':
                matrix=np.asarray(e['world_from_body'])
                vertices=vertices@matrix[:3,:3].T+matrix[:3,3]
            groups[group].append((vertices,faces))
        arrays.close()
        self.objects={}
        for group,meshes in groups.items():
            if not meshes:raise ValueError('Missing current component geometry: '+group)
            vertices=[];faces=[];offset=0
            for v,f in meshes:vertices.append(v);faces.append(f+offset);offset+=len(v)
            v=np.concatenate(vertices);f=np.concatenate(faces)
            model=fcl.BVHModel();model.beginModel(len(v),len(f));model.addSubModel(v,f);model.endModel()
            self.objects[group]=fcl.CollisionObject(model)

    def check(self,center,rotation,compression=.0025):
        fixed,moving=component_frames(center,rotation,compression)
        for name in ('fixed','clamp','shell','diaphragm'):
            t=fixed if name=='fixed' else moving
            self.objects[name].setTransform(self.fcl.Transform(t[:3,:3],t[:3,3]))
        distances={name:float(self.fcl.distance(self.objects[name],self.objects['patient']))
                   for name in ('fixed','clamp','shell','diaphragm')}
        if not np.isfinite(list(distances.values())).all():raise ValueError('Nonfinite finite-head distance')
        return dict(passed=bool(distances['fixed']>=.003 and distances['clamp']>=.003
                and distances['shell']>=.0003 and distances['diaphragm']<=.002),distances_m=distances,
            full_robot_collision_checked=False,physics_executed=False)

    def candidates(self,position,local_normal,*,compression=.0025):
        """Regenerate the original bounded normal/roll/gap family on new meshes.

        Original fitting family n+u*x+v*y, u,v in [-.6,.6], gives a maximum
        40.3155-degree local-normal deviation. This is not the tracking error
        tolerance; every candidate still requires full robot/physics validation.
        """
        position=np.asarray(position,float);normal=np.asarray(local_normal,float)
        if position.shape!=(3,) or normal.shape!=(3,) or not np.isfinite(position).all() or not np.isclose(np.linalg.norm(normal),1.):
            raise ValueError('Invalid surface anchor')
        # Preserve the original audit's square-bounded tangent coordinates.
        x=np.cross(normal,[0.,0.,1.])
        if np.linalg.norm(x)<1e-8:x=np.cross(normal,[1.,0.,0.])
        x/=np.linalg.norm(x)
        y=np.cross(normal,x);rows=[];attempts=0
        for angle in (0.,5.,10.,15.,20.,25.,30.,35.,40.):
            accepted=[]
            for azimuth in ([0.] if angle==0 else np.linspace(0,2*np.pi,16,endpoint=False)):
                uv=np.tan(np.deg2rad(angle))*np.array([np.cos(azimuth),np.sin(azimuth)])
                if np.max(abs(uv))>.6+1e-12:continue
                n=np.cos(np.deg2rad(angle))*normal+np.sin(np.deg2rad(angle))*(np.cos(azimuth)*x+np.sin(azimuth)*y)
                tangent=x-n*(x@n);tangent/=np.linalg.norm(tangent)
                for roll in np.linspace(0,2*np.pi,12,endpoint=False):
                    a=np.cos(roll)*tangent+np.sin(roll)*np.cross(n,tangent)
                    rotation=np.column_stack((a,np.cross(-n,a),-n))
                    for gap in (0.,.0015,.003,.005):
                        center=position+gap*normal;fit=self.check(center,rotation,compression);attempts+=1
                        if fit['passed']:
                            accepted.append(dict(center_world_m=center.tolist(),normal_world=n.tolist(),
                                rotation_world=rotation.tolist(),surface_gap_m=gap,
                                normal_deviation_from_surface_deg=angle,roll_rad=float(roll),fit=fit))
            # Preserve diversity at every tilt rather than spend all IK on one roll.
            if accepted:
                ids=np.linspace(0,len(accepted)-1,min(16,len(accepted)),dtype=int)
                rows.extend(accepted[i] for i in ids)
        return dict(candidates=rows,fit_attempts=attempts,loaded_compression_m=compression,
                    geometry_artifacts=getattr(self,'artifact_hashes',{}),
                    scope='finite tool surface hypotheses only; no robot, physics or training qualification')
