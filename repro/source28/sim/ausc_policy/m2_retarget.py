"""CPU-only observed-pose retarget assets; static numeric screening, never physics.

Declared frozen TRAIN or VALIDATION identities are selected. Two TRAIN poses are copied
without joint authoring, fitted to canonical identity betas, floor-aligned and
yawed toward world +Y. Full exact stool-footprint triangle clipping determines
seating height. No global embedded-solid/self-intersection gate is attempted.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from .body_placement import anatomical_frame
from .observed_placement import transform_observed_arrays
from .observed_seated_poses import _SMPLCPUAdapter, _identity_sources
from .surface_anchors import evaluate_anchors
from .geometry_epoch import GEOMETRY_ID, GEOMETRY_SOURCE_SHA256

SCHEMA = 'm2_observed_seated_retarget_v1'
FROZEN = {
    'split':'0e50ebce15f6e2816a914494cdf721acc6029db7d59cd0c9caeab0685a523699',
    'quality':'0b7f7fbd9843e82a0b5c3d781b0e852fd78f377f56f14685cab29a47c7d9ec24',
    'anchors':'6a185c28dbb6c321d3201f1b34d3077501d2e5fa2254231423634c17f682b8b3',
    'model':'b93df2dd228acc91d4ea6ede5f83f81e5171e7bee35c39695e6e27cb25ee1d4a',
    'poses':'1d22ed86e3fecff09301559ba7ba9901bdef7bf4713aa34bd94e42b0f0d366c8',
    'floor':'3ab01f81eedfeb90f8f9ce6835bc4700e5d101e809ab704f207312c2dcc1420b',
    'geometry':'11a2fc922602a70a97d83b26cd0cd9e0868a535c9d62e0fd10a38ca975b76275',
    'geometry_arrays':'0cd0e871d723a3505e5c565be688c5c5860d678faac2534f8bb837af77974a05',
}
POSES = (('p000474_a001230',0),('p000476_a001231',23))
FLAGS = dict(physics_qualified=False,training_eligible=False,seat_support_qualified=False,
             ik_qualified=False,static_scene_qa_passed=False)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return digest.hexdigest()


def bound_json(path,digest):
    data=Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Frozen artifact changed: '+str(path))
    return json.loads(data)


def write_json(path,value):
    def convert(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False,default=convert)+'\n')


def select_train_identities(split,quality,split_sha,count,seed):
    return select_identities(split,quality,split_sha,count,seed,'train')


def select_identities(split,quality,split_sha,count,seed,expected_split):
    if type(count) is not int or count<1 or type(seed) is not int:
        raise ValueError('Positive integer identity count and integer seed required')
    if expected_split not in ('train','validation'):
        raise ValueError('This generator accepts TRAIN or VALIDATION only; TEST remains unopened')
    _,eligible,_=_identity_sources(split,quality,split_sha,expected_split,None)
    if count>len(eligible):raise ValueError('Requested count exceeds eligible '+expected_split+' identities')
    return sorted(eligible,key=lambda x:hashlib.sha256(f'{seed}:{x}'.encode()).digest())[:count]


def retarget_parameters(candidate,betas):
    values={key:np.array(value,dtype=float,copy=True) for key,value in candidate['observed_parameters'].items()}
    values['betas']=np.array(betas,dtype=float,copy=True)
    for key,width in [('betas',10),('body_pose',69),('global_orient',3),('transl',3)]:
        if key not in values or values[key].shape!=(width,) or not np.isfinite(values[key]).all():
            raise ValueError('Invalid observed retarget parameter: '+key)
    return {key:values[key] for key in ('betas','body_pose','global_orient','transl')}


def clip_polygon(triangle,equations):
    """Clip a 3-D polygon by halfspaces a*x+b*y+c*z+d <= 0."""
    polygon=[p.copy() for p in np.asarray(triangle,float)]
    for equation in equations:
        out=[]
        for p,q in zip(polygon,polygon[1:]+polygon[:1]):
            dp=equation[:3]@p+equation[3];dq=equation[:3]@q+equation[3]
            ip,iq=dp<=1e-12,dq<=1e-12
            if ip:out.append(p)
            if ip!=iq and dp!=dq:out.append(p+(q-p)*dp/(dp-dq))
        polygon=out
        if not polygon:break
    return polygon


def footprint_minimum(vertices,faces,seat_vertices):
    """Minimum over every triangle clipped to the exact convex seat XY hull."""
    from scipy.spatial import ConvexHull
    v=np.asarray(vertices,float);f=np.asarray(faces);seat=np.asarray(seat_vertices,float)
    hull=ConvexHull(seat[:,:2]);equations=np.column_stack([hull.equations[:,:2],np.zeros(len(hull.equations)),hull.equations[:,2]])
    tri=v[f];lo=seat[:,:2].min(0);hi=seat[:,:2].max(0)
    ids=np.flatnonzero(np.all(tri[:,:,:2].max(1)>=lo-1e-12,axis=1)&np.all(tri[:,:,:2].min(1)<=hi+1e-12,axis=1))
    best=None;nonempty=0
    for i in ids:
        polygon=clip_polygon(tri[i],equations)
        if polygon:
            nonempty+=1;p=min(polygon,key=lambda p:p[2])
            if best is None or p[2]<best[0]:best=(float(p[2]),int(i),p)
    if best is None:raise ValueError('No patient footprint overlap')
    return dict(min_z_m=best[0],face_id=best[1],point_world_m=best[2].tolist(),
                all_triangles=len(f),xy_candidates=len(ids),clipped_triangles=nonempty,
                seat_hull_vertices=len(hull.vertices))


def _clipped_edge_has_length(p,q,equations):
    """Whether an original edge survives the same patch clipping halfspaces."""
    delta=q-p;lower=0.;upper=1.
    for equation in equations:
        offset=float(equation[:3]@p+equation[3]);slope=float(equation[:3]@delta)
        if abs(slope)<=1e-15:
            if offset>1e-12:return False
            continue
        crossing=-offset/slope
        if slope>0:upper=min(upper,crossing)
        else:lower=max(lower,crossing)
        if upper<=lower:return False
    return bool((upper-lower)*np.linalg.norm(delta)>1e-12)


def fit_contact_patch(vertices,faces,anchor,raw_normal,*,seed_vertex_ids,radius=.023,sides=96):
    """Continuous surface-area TLS on an anchor-connected clipped mesh patch.

    The tangent disk is a 96-sided inscribed polygon (max radial deficit 12.4
    micrometres at 23mm), clipped also to +/- radius depth. Polygon triangles
    contribute exact first/second surface moments, independent of tessellation.
    Connectivity uses only shared edges with positive length after the same
    disk and depth clipping; a meeting outside the patch cannot connect faces.
    """
    v=np.asarray(vertices,float);f=np.asarray(faces);p=np.asarray(anchor,float);n=np.asarray(raw_normal,float)
    if radius<=0 or not np.isfinite(radius) or sides<16 or not np.isclose(np.linalg.norm(n),1.,atol=1e-8):
        raise ValueError('Finite positive radius and unit raw normal required')
    u=np.cross(n,np.eye(3)[np.argmin(abs(n))]);u/=np.linalg.norm(u);w=np.cross(n,u)
    basis=np.column_stack([u,w,n]);local=(v-p)@basis;tri=local[f]
    theta=(np.arange(sides)+.5)*2*np.pi/sides
    equations=np.column_stack([np.cos(theta),np.sin(theta),np.zeros(sides),np.full(sides,-radius*np.cos(np.pi/sides))])
    equations=np.vstack([equations,[0.,0.,1.,-radius],[0.,0.,-1.,-radius]])
    ids=np.flatnonzero(np.all(tri.max(1)>=-radius,axis=1)&np.all(tri.min(1)<=radius,axis=1))
    polygons={int(i):clip_polygon(tri[i],equations) for i in ids}
    polygons={i:poly for i,poly in polygons.items() if len(poly)>=3}
    seed=[i for i in polygons if set(map(int,f[i]))==set(map(int,seed_vertex_ids))]
    if len(seed)!=1:raise ValueError('Contact patch seed triangle missing or ambiguous')
    edges={}
    for i in polygons:
        for a,b in zip(f[i],np.roll(f[i],-1)):
            if _clipped_edge_has_length(local[a],local[b],equations):
                edges.setdefault(tuple(sorted((int(a),int(b)))),[]).append(i)
    connected=set(seed);queue=list(seed)
    while queue:
        i=queue.pop()
        for a,b in zip(f[i],np.roll(f[i],-1)):
            for j in edges.get(tuple(sorted((int(a),int(b)))),()):
                if j not in connected:connected.add(j);queue.append(j)
    total=0.;first=np.zeros(3);second=np.zeros((3,3))
    for i in connected:
        poly=np.array(polygons[i])
        for k in range(1,len(poly)-1):
            t=poly[[0,k,k+1]];area=np.linalg.norm(np.cross(t[1]-t[0],t[2]-t[0]))/2
            summed=t.sum(0);total+=area;first+=area*summed/3
            second+=area*(t.T@t+np.outer(summed,summed))/12
    if total<1e-10:raise ValueError('Insufficient local surface area')
    centroid=first/total;cov=second/total-np.outer(centroid,centroid)
    values,vectors=np.linalg.eigh(cov)
    if values[1]<1e-12:raise ValueError('Rank-deficient local contact patch')
    normal=basis@vectors[:,0]
    if normal@n<0:normal=-normal
    return dict(normal_world=normal.tolist(),centroid_world_m=(p+basis@centroid).tolist(),
                radius_m=float(radius),surface_area_m2=float(total),local_triangles=len(connected),
                rms_residual_m=float(np.sqrt(max(0.,values[0]))),
                raw_to_patch_angle_deg=float(np.degrees(np.arccos(np.clip(normal@n,-1,1)))),
                disk_polygon_sides=sides,max_radial_deficit_m=float(radius*(1-np.cos(np.pi/sides))),
                method='surface_area_integrated_orthogonal_least_squares_clipped_edge_connected_triangles_v2',
                raw_anchor_position_unchanged=True,ray_audit_performed=False)


def place_retarget(vertices,faces,joints,anchors,seat_vertices,floor_normal,floor_binding,seat_clearance=1e-6):
    raw=evaluate_anchors(vertices,faces,anchors);names=list(raw)
    positions=np.array([raw[n]['position_m'] for n in names]);normals=np.array([raw[n]['normal'] for n in names])
    common=dict(source_floor_normal=floor_normal,source_anchor_m=joints[0],source_floor_binding=floor_binding)
    aligned=transform_observed_arrays(vertices,joints,positions,normals,world_yaw_rad=0.,world_anchor_m=[0.,0.,0.],**common)
    front=anatomical_frame(aligned['joints'])[:,2]
    if np.linalg.norm(front[:2])<1e-8:raise ValueError('Projected anatomical front is degenerate')
    yaw=float(np.pi/2-np.arctan2(front[1],front[0]))
    center=(np.min(seat_vertices,axis=0)+np.max(seat_vertices,axis=0))/2;center[2]=0
    placed=transform_observed_arrays(vertices,joints,positions,normals,world_yaw_rad=yaw,world_anchor_m=center,**common)
    minimum=footprint_minimum(placed['vertices'],faces,seat_vertices)
    top=float(np.max(seat_vertices,axis=0)[2]);shift=top+seat_clearance-minimum['min_z_m']
    placed['vertices'][:,2]+=shift;placed['joints'][:,2]+=shift;placed['world_from_source'][2,3]+=shift
    saved=placed['vertices'].astype('<f4');after=footprint_minimum(saved,faces,seat_vertices)
    evaluated=evaluate_anchors(saved,faces,anchors)
    return dict(vertices=saved,faces=np.asarray(faces,dtype=np.int32),joints=placed['joints'].astype('<f4'),
                anchor_names=np.array(names),anchor_positions=np.array([evaluated[n]['position_m'] for n in names]),
                anchor_normals=np.array([evaluated[n]['normal'] for n in names]),world_from_source=placed['world_from_source'],
                placement=dict(world_yaw_rad=yaw,world_z_translation_m=shift,seat_top_m=top,
                    seat_gap_m=after['min_z_m']-top,requested_numeric_clearance_m=seat_clearance,
                    footprint_minimum_before=minimum,footprint_minimum_after=after,
                    front_alignment='projected anatomical front to world +Y; torso tilt preserved',
                    invariance=placed['invariance'],source_floor_binding=floor_binding,
                    source_floor_assumption='floor assumed level; gravity not measured',
                    foot_support_verified=False,torso_leveling_applied=False))


class ExternalSurfaceScreen:
    """FCL triangle surface checks; no solid containment or global safety claim."""
    def __init__(self,metadata,arrays):
        import fcl
        self.fcl=fcl;self.entries=[];self.objects={};self.geometry={}
        for e in metadata['colliders']:
            if (e['path'].startswith(('/World/Stool/','/World/Table/')) or
                (e['body'].startswith('/World/StethoArmMount/OpenArmStetho/') and
                 ('/openarm_right_' in e['body'] or e['body'].endswith(('openarm_body_link0','openarm_left_link0'))))):
                t=np.asarray(e['world_from_body']);v=arrays[e['key']+'_v'].astype(float)@t[:3,:3].T+t[:3,3]
                f=arrays[e['key']+'_f'];self.entries.append(e);self.geometry[e['path']]=(v,f)
                self.objects[e['path']]=self.object(v,f)
        if len(self.entries)!=22:raise ValueError('Expected 22 frozen external static mesh colliders')

    def object(self,v,f):
        model=self.fcl.BVHModel();model.beginModel(len(v),len(f));model.addSubModel(v,np.asarray(f,np.int32));model.endModel()
        return self.fcl.CollisionObject(model)

    def check(self,vertices,faces):
        patient=self.object(vertices.astype(float),faces);pairs=[]
        for entry in self.entries:
            result=self.fcl.CollisionResult();other=self.objects[entry['path']]
            self.fcl.collide(patient,other,self.fcl.CollisionRequest(num_max_contacts=4,enable_contact=True),result)
            distance=self.fcl.DistanceResult();d=float(self.fcl.distance(patient,other,self.fcl.DistanceRequest(),distance))
            if not np.isfinite(d):raise ValueError('Nonfinite FCL surface distance')
            pairs.append(dict(path=entry['path'],surface_intersection=bool(result.is_collision),
                distance_m=0. if result.is_collision else d,
                contacts=[dict(patient_face=int(c.b1),external_face=int(c.b2),point_world_m=np.asarray(c.pos).tolist()) for c in result.contacts[:4]]))
        return dict(surface_pairs_completed=len(pairs),pairs=pairs,
                    surface_intersections=sum(p['surface_intersection'] for p in pairs),
                    floor_min_z_m=float(vertices[:,2].min()),solid_containment_checked=False,
                    global_self_intersection_checked=False,**FLAGS)


def generate(base,output,count=40,seed=20261026,*,expected_split='train'):
    """Generate actual frozen assets on Linux CPU only; output must be new."""
    if sys.platform!='linux' or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('Actual generation requires Linux and CUDA_VISIBLE_DEVICES empty')
    started=time.perf_counter();base=Path(base).resolve();output=Path(output).resolve();runs=base/'runs/ausc_policy'
    paths=dict(split=runs/'m2_identity_5mm_v2/identity_split.json',quality=runs/'m2_identity_5mm_v2/identity_quality.json',
               anchors=runs/'asset_inventory/inventory.json',model=base/'m2-assets/smpl/SMPL_NEUTRAL.pkl',
               poses=runs/'observed_seated_train_label_pilot_20260930_v1/candidate_manifest.json',
               floor=runs/'source_floor_fit_20260930_v1/summary.json',
               geometry=runs/'retainer-5mm-recess1mm-review-20260930-v2/output/review/geometry/collision_geometry.json',
               geometry_arrays=runs/'retainer-5mm-recess1mm-review-20260930-v2/output/review/geometry/collision_geometry.npz')
    inputs={k:bound_json(p,FROZEN[k]) for k,p in paths.items() if k not in ('model','geometry_arrays')}
    for k in ('model','geometry_arrays'):
        if sha(paths[k])!=FROZEN[k]:raise ValueError('Frozen artifact changed: '+k)
    chosen=select_identities(inputs['split'],inputs['quality'],FROZEN['split'],count,seed,expected_split)
    anchors=inputs['anchors']['anchors']
    if len(anchors)!=10:raise ValueError('Expected 10 V6A anchors')
    sources=[]
    for sequence,frame in POSES:
        candidates=[c for c in inputs['poses']['candidates'] if c['source']['sequence_path']==sequence+'/smpl_params.npz' and c['source']['frame_index']==frame]
        if len(candidates)!=1:raise ValueError('Observed pose source is not unique')
        c=candidates[0];identity=c['identity']
        if c['split']!='train' or identity not in inputs['split']['splits']['train']:raise ValueError('Observed pose is not train')
        path=base/'m2-assets/smpl_params'/c['source']['sequence_path']
        if sha(path)!=c['source']['sequence_sha256']:raise ValueError('Observed sequence changed')
        # Only these two observed TRAIN sequence arrays are opened, never target or heldout sequences.
        with np.load(path,allow_pickle=False) as z:
            for key in ('body_pose','global_orient','transl','betas'):
                if not np.array_equal(z[key][frame],np.asarray(c['observed_parameters'][key])):raise ValueError('Observed parameters differ from source bytes')
        row=next(r for r in inputs['floor']['records'] if r['sequence']==sequence)
        plane=bound_json(paths['floor'].parent/row['plane_file'],row['plane_sha256'])
        binding=dict(method='floor_normal_assuming_level',source_plane_sha256=row['plane_sha256'],source_report_sha256=FROZEN['floor'],source_frame=plane['source_frame'])
        sources.append((sequence,c,plane,binding))
    with np.load(paths['geometry_arrays'],allow_pickle=False) as z:arrays={k:np.array(z[k]) for k in z.files}
    screen=ExternalSurfaceScreen(inputs['geometry'],arrays);seat=screen.geometry['/World/Stool/seat'][0]
    adapter=_SMPLCPUAdapter(paths['model']);output.mkdir(parents=True,exist_ok=False)
    records=[];failures=[]
    for identity in chosen:
        for sequence,c,plane,binding in sources:
            label=f'{identity}_{sequence}_f{c["source"]["frame_index"]:04d}';t=time.perf_counter()
            try:
                params=retarget_parameters(c,inputs['quality']['eligible_shape_betas'][identity]);body=adapter(**params)
                if body['vertices'].shape!=(6890,3) or body['faces'].shape!=(13776,3):raise ValueError('Unexpected SMPL topology')
                asset=place_retarget(body['vertices'],body['faces'],body['joints'],anchors,seat,plane['normal_source'],binding)
                placement=asset.pop('placement');patches={}
                for name,p,n in zip(asset['anchor_names'],asset['anchor_positions'],asset['anchor_normals']):
                    patches[str(name)]=fit_contact_patch(asset['vertices'],asset['faces'],p,n,seed_vertex_ids=anchors[str(name)]['smpl_vertex_indices'])
                asset['contact_patch_normals']=np.array([patches[str(n)]['normal_world'] for n in asset['anchor_names']])
                numeric=screen.check(asset['vertices'],asset['faces'])
                passed=bool(numeric['surface_intersections']==0 and numeric['floor_min_z_m']>=-1e-6 and abs(placement['seat_gap_m']-1e-6)<=1e-7)
                filename=label+'.npz';np.savez_compressed(output/filename,**asset)
                provenance=dict(identity=identity,canonical_betas=params['betas'],shape_source=dict(kind='frozen_identity_quality_canonical_beta',quality_sha256=FROZEN['quality']),
                    pose_source=c['source'],observed_parameters=c['observed_parameters'],retargeted_parameters=params,
                    candidate_manifest_sha256=FROZEN['poses'],source_floor_plane_sha256=binding['source_plane_sha256'],
                    placement=placement,contact_patches=patches,external_static_screen=numeric,**FLAGS)
                detail=label+'.json';write_json(output/detail,provenance)
                records.append(dict(identity=identity,proposal=label,split=expected_split,file=filename,sha256=sha(output/filename),
                    provenance_file=detail,provenance_sha256=sha(output/detail),source_pose=c['source'],mesh_scale=1.,
                    numeric_candidate_screen_passed=passed,external_surface_intersections=numeric['surface_intersections'],
                    seat_gap_m=placement['seat_gap_m'],floor_min_z_m=numeric['floor_min_z_m'],
                    pose_kind='observed_train_pose_retargeted_to_frozen_identity_shape',synthetic_pose=True,
                    vertices=6890,faces=13776,anchors=10,wall_seconds=time.perf_counter()-t,**FLAGS))
            except Exception as error:
                failures.append(dict(identity=identity,proposal=label,error_type=type(error).__name__,error=str(error)))
            print(json.dumps(dict(identity=identity,pose=sequence,assets=len(records),failures=len(failures))),flush=True)
    write_json(output/'asset_manifest.json',records);write_json(output/'generation_failures.json',failures)
    summary=dict(schema=SCHEMA,geometry_id=GEOMETRY_ID,geometry_source_sha256=GEOMETRY_SOURCE_SHA256,freshly_generated=True,
        identity_split_sha256=FROZEN['split'],identity_quality_sha256=FROZEN['quality'],anchor_inventory_sha256=FROZEN['anchors'],smpl_neutral_sha256=FROZEN['model'],
        input_bindings={k:dict(path=str(p),sha256=FROZEN[k]) for k,p in paths.items()},
        code_sha256={Path(p).name:sha(p) for p in [__file__,sys.modules[_SMPLCPUAdapter.__module__].__file__,sys.modules[transform_observed_arrays.__module__].__file__,sys.modules[evaluate_anchors.__module__].__file__,sys.modules[anatomical_frame.__module__].__file__,sys.modules['sim.ausc_policy.m2_identity'].__file__,sys.modules['sim.ausc_policy.geometry_epoch'].__file__]},
        selected_train_identities=chosen if expected_split=='train' else [],
        selected_identities=chosen,target_identity_split=expected_split,
        seed=seed,requested_identities=count,poses_per_identity=len(POSES),requested_assets=count*len(POSES),
        generated_assets=len(records),generated_identities=len({r['identity'] for r in records}),
        numeric_screen_passed_assets=sum(r['numeric_candidate_screen_passed'] for r in records),
        numeric_screen_passed_identities=len({r['identity'] for r in records if r['numeric_candidate_screen_passed']}),
        generation_failures=len(failures),manifest_sha256=sha(output/'asset_manifest.json'),
        eligible_frozen_split_counts={k:sum(i in inputs['quality']['eligible_shape_betas'] for i in ids) for k,ids in inputs['split']['splits'].items()},
        coordinates='metres; nominal approved scene world frame; patient front projected to +Y',
        normal_contract='NPZ anchor_normals are original face normals; contact_patch_normals are radius23mm area-integrated TLS normals; raw anchors unchanged',
        limitations='Static triangle-surface screening only; known SMPL axilla self-intersections not globally qualified; solid containment, foot support, robot reachability and physics unverified. Source floor assumed level. Retargeted bodies combine two known TRAIN poses with shapes from the explicitly declared target split; pose generalization is untested.',
        cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],wall_seconds=time.perf_counter()-started,**FLAGS)
    write_json(output/'summary.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--identities',type=int,default=40);parser.add_argument('--seed',type=int,default=20261026)
    parser.add_argument('--split',choices=('train','validation'),default='train')
    args=parser.parse_args();summary=generate(args.base,args.output,args.identities,args.seed,expected_split=args.split)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
