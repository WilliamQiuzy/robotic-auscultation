"""Fresh finite-head surface fitting; no previous plans, IK or motion."""
import json
from pathlib import Path
import numpy as np


FIRST_CONTACT_RESOLUTION_M = 1e-6
CONTACT_NUMERIC_TOLERANCE_M = 1e-9
FIRST_CONTACT_MIN_STEP_M = 1e-9
DIAPHRAGM_GAP_BAND_M = (1e-5, 3e-5)
FIRST_CONTACT_INTERVAL_M = (0., .005)


def _surface_anchor(position, local_normal):
    position=np.array(position,float,copy=True);normal=np.array(local_normal,float,copy=True)
    if (position.shape!=(3,) or normal.shape!=(3,) or not np.isfinite(position).all()
            or not np.isfinite(normal).all()
            or not np.isclose(np.linalg.norm(normal),1.,atol=1e-6,rtol=0)):
        raise ValueError('Invalid surface anchor')
    normal/=np.linalg.norm(normal)
    return position,normal


def _observation(report):
    """Recompute gates from distances; never trust a caller's coarse-fit flag."""
    distances=report.get('distances_m',{})
    try:
        values=np.asarray([distances[k] for k in ('fixed','clamp','shell','diaphragm')],float)
    except (KeyError,TypeError,ValueError) as error:
        raise ValueError('Invalid finite-head distance report') from error
    collision=report.get('diaphragm_collision')
    if (values.shape!=(4,) or not np.isfinite(values).all()
            or not isinstance(collision,(bool,np.bool_))
            or (values[3]<0 and not collision)):
        raise ValueError('Invalid or nonfinite finite-head distance/collision report')
    clear=bool(values[0]>=.003 and values[1]>=.003 and values[2]>=.0003)
    touch=bool(collision or values[3]<=CONTACT_NUMERIC_TOLERANCE_M)
    passed=bool(clear and not touch and DIAPHRAGM_GAP_BAND_M[0]<=values[3]<=DIAPHRAGM_GAP_BAND_M[1])
    return {**report,'passed':passed,'noncontact_clear':clear,'diaphragm_touch':touch,
            'diaphragm_gap_band_m':list(DIAPHRAGM_GAP_BAND_M)}


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
        count=self.fcl.collide(self.objects['diaphragm'],self.objects['patient'],
                              self.fcl.CollisionRequest(),self.fcl.CollisionResult())
        if not np.isscalar(count) or not np.isfinite(count) or count<0:
            raise ValueError('Invalid native diaphragm collision result')
        return _observation(dict(distances_m=distances,diaphragm_collision=bool(count>0),
            full_robot_collision_checked=False,physics_executed=False))

    def first_contact(self,position,local_normal,rotation,compression=.0025):
        """Locate first surface contact on the unchanged command-normal line.

        The old 0--5 mm gap range is a SEARCH interval, not accepted contact.
        Start outside and conservatively advance using the current triangle
        distance (translation cannot reduce distance faster than travel). The
        final 1 nanometre minimum step resolves contact numerically; this is
        explicitly a sampled engineering gate, not an exact CCD certificate.
        Refine the first clear/touch bracket and back off to a measured 10--30
        micrometre gap. Recheck all noncontact parts at the final position.
        A missing contact bracket, obstruction or exhausted budget rejects the
        rotation; there is no inward extension or silent anchor replacement.
        """
        position,normal=_surface_anchor(position,local_normal)
        rotation=np.asarray(rotation,float)
        component_frames(position,rotation,compression)  # rigid frame/domain validation
        direction=-rotation[:,2]
        x=np.cross(normal,[0.,0.,1.])
        if np.linalg.norm(x)<1e-8:x=np.cross(normal,[1.,0.,0.])
        x/=np.linalg.norm(x);y=np.cross(normal,x)
        dot=float(direction@normal)
        if dot<=0 or np.max(abs(np.array([direction@x,direction@y])/dot))>.6+1e-12:
            raise ValueError('First-contact direction exceeds the original fitted-normal family')
        evaluations=0
        common=dict(raw_anchor_world_m=position.tolist(),gap_axis='command_normal',
            gap_direction_world=direction.tolist(),
            first_contact_search_resolution_m=FIRST_CONTACT_RESOLUTION_M,
            first_contact_search_min_step_m=FIRST_CONTACT_MIN_STEP_M,
            contact_numeric_tolerance_m=CONTACT_NUMERIC_TOLERANCE_M,
            bracket_noncontact_numeric_tolerance_m=CONTACT_NUMERIC_TOLERANCE_M,
            first_contact_search_interval_m=list(FIRST_CONTACT_INTERVAL_M),
            full_robot_collision_checked=False,physics_executed=False)

        def evaluate(gap):
            nonlocal evaluations
            evaluations+=1
            return _observation(self.check(position+direction*gap,rotation,compression))

        def rejected(reason,report=None):
            result={**common,'passed':False,'reason':reason,'fit_evaluations':evaluations}
            if report is not None:result['rejected_fit']=report
            return result

        def bracket_clear(report):
            # Only discovery/refinement of the contact boundary gets this
            # numerical cushion. The final candidate gate stays strictly at
            # 3 mm / 0.3 mm, including its separate fresh evaluation below.
            d=report['distances_m'];tol=CONTACT_NUMERIC_TOLERANCE_M
            return d['fixed']>=.003-tol and d['clamp']>=.003-tol and d['shell']>=.0003-tol

        inner,outer=FIRST_CONTACT_INTERVAL_M
        gap=outer;report=evaluate(gap)
        if not report['noncontact_clear']:return rejected('noncontact_obstruction',report)
        if report['diaphragm_touch']:return rejected('outer_endpoint_not_clear',report)
        history=[(gap,report)]
        bracket=None
        for _ in range(512):
            if gap<=inner:break
            distance=report['distances_m']['diaphragm']
            # Near a shell exactly 0.3 mm behind the face, a 1 um step could
            # physically pass the boundary. Use a 1 nm terminal step instead.
            step=max(.8*max(0.,distance-CONTACT_NUMERIC_TOLERANCE_M),FIRST_CONTACT_MIN_STEP_M)
            next_gap=max(inner,gap-step)
            following=evaluate(next_gap)
            if not bracket_clear(following):return rejected('noncontact_obstruction',following)
            if following['diaphragm_touch']:
                bracket=[next_gap,gap]
                break
            gap,report=next_gap,following
            history.append((gap,report))
        if bracket is None:
            return rejected('missing_contact_bracket' if gap<=inner else 'contact_search_budget_exhausted',report)

        for _ in range(64):
            if bracket[1]-bracket[0]<=FIRST_CONTACT_RESOLUTION_M:break
            middle=.5*(bracket[0]+bracket[1]);sample=evaluate(middle)
            if not bracket_clear(sample):return rejected('noncontact_obstruction',sample)
            if sample['diaphragm_touch']:bracket[0]=middle
            else:
                bracket[1]=middle;history.append((middle,sample))
        if bracket[1]-bracket[0]>FIRST_CONTACT_RESOLUTION_M:
            return rejected('contact_bracket_resolution_not_reached')

        # Retain the last outside sample with enough space to bracket a 20 um
        # residual. Distances need not be monotonic across the entire mesh.
        target=.5*sum(DIAPHRAGM_GAP_BAND_M)
        outside=next(((g,r) for g,r in reversed(history)
                      if r['distances_m']['diaphragm']>=target),None)
        candidate=None
        if outside is None:
            candidate=next(((g,r) for g,r in reversed(history) if r['passed']),None)
        else:
            low,high=bracket[0],outside[0]
            for _ in range(64):
                middle=.5*(low+high);sample=evaluate(middle)
                if not sample['noncontact_clear']:return rejected('noncontact_obstruction',sample)
                if sample['passed']:
                    candidate=(middle,sample);break
                if sample['diaphragm_touch'] or sample['distances_m']['diaphragm']<DIAPHRAGM_GAP_BAND_M[0]:
                    low=middle
                else:high=middle
                if high-low<FIRST_CONTACT_RESOLUTION_M:break
        if candidate is None:return rejected('near_contact_gap_not_resolved')
        final_gap=candidate[0]
        final=evaluate(final_gap)  # Freshly recheck every noncontact component.
        if not final['noncontact_clear']:return rejected('noncontact_obstruction',final)
        if not final['passed']:return rejected('final_contact_gap_changed',final)
        return {**common,'passed':True,'center_world_m':(position+direction*final_gap).tolist(),
            'surface_gap_m':float(final_gap),'contact_surface_gap_m':float(np.mean(bracket)),
            'contact_surface_gap_bracket_m':list(map(float,bracket)),
            'contact_bracket_width_m':float(bracket[1]-bracket[0]),
            'diaphragm_residual_distance_m':float(final['distances_m']['diaphragm']),
            'fit_evaluations':evaluations,'fit':final}

    def candidates(self,position,local_normal,*,compression=.0025):
        """Regenerate the original bounded normal/roll/gap family on new meshes.

        Original fitting family n+u*x+v*y, u,v in [-.6,.6], gives a maximum
        40.3155-degree local-normal deviation. This is not the tracking error
        tolerance; every candidate still requires full robot/physics validation.
        """
        position,normal=_surface_anchor(position,local_normal)
        # Preserve the original audit's square-bounded tangent coordinates.
        x=np.cross(normal,[0.,0.,1.])
        if np.linalg.norm(x)<1e-8:x=np.cross(normal,[1.,0.,0.])
        x/=np.linalg.norm(x)
        y=np.cross(normal,x);rows=[];attempts=0;orientations=0;rejections={}
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
                    found=self.first_contact(position,normal,rotation,compression)
                    attempts+=found['fit_evaluations'];orientations+=1
                    if found['passed']:
                        accepted.append({**found,'normal_world':n.tolist(),
                            'rotation_world':rotation.tolist(),
                            'normal_deviation_from_surface_deg':angle,'roll_rad':float(roll)})
                    else:
                        reason=found['reason'];rejections[reason]=rejections.get(reason,0)+1
            # Preserve diversity at every tilt rather than spend all IK on one roll.
            if accepted:
                ids=np.linspace(0,len(accepted)-1,min(16,len(accepted)),dtype=int)
                rows.extend(accepted[i] for i in ids)
        return dict(candidates=rows,fit_attempts=attempts,orientation_attempts=orientations,
                    rejected_orientations=rejections,loaded_compression_m=compression,
                    geometry_artifacts=getattr(self,'artifact_hashes',{}),
                    first_contact_search_resolution_m=FIRST_CONTACT_RESOLUTION_M,
                    diaphragm_gap_band_m=list(DIAPHRAGM_GAP_BAND_M),gap_axis='command_normal',
                    scope='sampled finite tool near-contact hypotheses only; no CCD, robot, physics or training qualification')
