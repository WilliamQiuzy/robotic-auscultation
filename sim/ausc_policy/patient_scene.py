"""Bind a numerically screened SMPL candidate into a separate simulation scene.

The approved robot/tool/table/stool/camera source is unchanged. Replacing the
patient is explicit M2 scene variation, not qualification of the new patient.
"""
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from .geometry_epoch import require_current
from .surface_anchors import evaluate_anchors


def _read(path,digest):
    data=Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Patient source hash mismatch: '+str(path))
    return data


def layout_translation(spec):
    layout=spec.get('layout')
    if layout is None:return np.zeros(3)
    if set(layout)!={'xy_m','height_m'}:raise ValueError('Unsupported patient layout fields')
    xy=np.asarray(layout['xy_m'],float);height=layout['height_m']
    if (xy.shape!=(2,) or not np.isfinite(xy).all() or np.max(abs(xy))>.15+1e-12
            or isinstance(height,bool) or height not in (-.06,-.03,0.,.03,.06)):
        raise ValueError('Layout exceeds frozen floor/height search domain')
    return np.r_[xy,float(height)]


def load_candidate(spec,*,formal_config=None):
    if formal_config is not None:
        from .m2_given_contact_evaluation import authorize_patient
        authorize_patient(spec,formal_config)
    elif spec.get('expected_split')=='test':
        raise ValueError('TEST patient requires an exact issued formal capability')
    require_current(spec)
    root=Path(spec['directory'])
    manifest=json.loads(_read(root/'asset_manifest.json',spec['manifest_sha256']))
    summary=json.loads(_read(root/'summary.json',spec['summary_sha256']))
    from .m2_contract import IDENTITY_SPLIT_SHA256, QUARANTINED_IDENTITIES
    if summary['identity_split_sha256']!=IDENTITY_SPLIT_SHA256:
        raise ValueError('Patient frozen identity split gate differs')
    split=json.loads(_read(spec['identity_split'],summary['identity_split_sha256']))['splits']
    anchors=json.loads(_read(spec['anchor_inventory'],summary['anchor_inventory_sha256']))['anchors']
    records=[r for r in manifest if r['file']==spec['file']]
    if len(records)!=1:raise ValueError('Patient candidate is not unique')
    record=records[0]
    expected_split=spec.get('expected_split','train')
    reconstructed=False
    if ('reconstruction_source' in spec or
            summary.get('schema')=='m2_reconstructed_observed_seated_retarget_v1'):
        from .observed_reconstruction import validate_reconstruction
        validate_reconstruction(spec,summary,record)
        reconstructed=True
    observed=summary.get('schema')=='m2_observed_seated_retarget_v1' or reconstructed
    if (record.get('numeric_candidate_screen_passed') is not True or record.get('mesh_scale')!=1.
            or expected_split not in ('train','validation','test')
            or record.get('split')!=expected_split
            or [name for name,ids in split.items() if record['identity'] in ids]!=[expected_split]
            or record['identity'] in QUARANTINED_IDENTITIES
            or (not observed and summary.get('schema')!='m2_authored_seated_arm_pose_pilot_v1')):
        raise ValueError('Patient candidate failed numeric or identity gate')
    file=root/spec['file']
    if file.resolve().parent!=root.resolve():raise ValueError('Patient candidate path escapes manifest')
    data=_read(file,record['sha256'])
    with np.load(io.BytesIO(data),allow_pickle=False) as z:
        arrays={key:np.array(z[key],copy=True) for key in
            ('vertices','faces','joints','anchor_names','anchor_positions','anchor_normals')}
        if observed:arrays['contact_patch_normals']=np.array(z['contact_patch_normals'],copy=True)
    if arrays['vertices'].shape!=(6890,3) or arrays['faces'].shape!=(13776,3):
        raise ValueError('Unexpected SMPL topology')
    calculated=evaluate_anchors(arrays['vertices'],arrays['faces'],anchors)
    if arrays['anchor_names'].shape!=(10,) or set(arrays['anchor_names'].tolist())!=set(calculated):
        raise ValueError('Incomplete patient anchors')
    for key,field in [('anchor_positions','position_m'),('anchor_normals','normal')]:
        expected=np.array([calculated[n][field] for n in arrays['anchor_names']])
        if not np.allclose(arrays[key],expected,atol=1e-7,rtol=0):
            raise ValueError('Patient anchors differ from actual mesh')
    targets=[dict(name=str(n),position_world_m=p.tolist(),normal_world=v.tolist())
        for n,p,v in zip(arrays['anchor_names'],arrays['anchor_positions'],arrays['anchor_normals'])]
    if observed:
        normals=arrays['contact_patch_normals']
        if (normals.shape!=(10,3) or not np.isfinite(normals).all()
                or not np.allclose(np.linalg.norm(normals,axis=1),1.,atol=1e-8,rtol=0)):
            raise ValueError('Invalid finite contact patch normals')
        provenance_path=root/record['provenance_file']
        if provenance_path.resolve().parent!=root.resolve():raise ValueError('Patient provenance path escaped')
        provenance=json.loads(_read(provenance_path,record['provenance_sha256']))
        for target,normal in zip(targets,normals):
            patch=provenance['contact_patches'][target['name']]
            if (patch['radius_m']!=.023 or not np.allclose(patch['normal_world'],normal,atol=1e-10,rtol=0)):
                raise ValueError('Contact patch provenance differs')
            target['contact_plane_normal_world']=normal.tolist()
            target['contact_plane_radius_m']=.023
    delta=layout_translation(spec)
    if np.any(delta):
        if not observed:raise ValueError('M2 layout requires observed retargeted body')
        from .m2_retarget import fit_contact_patch
        arrays['vertices']=(arrays['vertices'].astype(float)+delta).astype(np.float32)
        arrays['joints']=arrays['joints']+delta
        evaluated=evaluate_anchors(arrays['vertices'],arrays['faces'],anchors)
        for index,target in enumerate(targets):
            name=target['name'];value=evaluated[name]
            arrays['anchor_positions'][index]=value['position_m']
            arrays['anchor_normals'][index]=value['normal']
            patch=fit_contact_patch(arrays['vertices'],arrays['faces'],value['position_m'],value['normal'],
                seed_vertex_ids=anchors[name]['smpl_vertex_indices'])
            arrays['contact_patch_normals'][index]=patch['normal_world']
            target.update(position_world_m=value['position_m'],normal_world=value['normal'],
                          contact_plane_normal_world=patch['normal_world'])
    receipt=dict(identity=record['identity'],proposal=record['proposal'],asset_sha256=record['sha256'],
        identity_split_sha256=summary['identity_split_sha256'],targets=targets,
        split=expected_split,mesh_scale=1.,clothing='none',body_is_static=True,synthetic_pose=not observed,
        pose_source='observed_pose_retargeted_to_canonical_shape' if observed else 'authored_pose',
        synthetic_shape_pose_combination=observed,
        layout_translation_world_m=delta.tolist(),layout_changes_stool_and_patient=bool(np.any(delta)),
        post_layout_external_collision_qualified=False,
        physics_qualified=False,training_eligible=False)
    if reconstructed:
        receipt.update(reconstruction_source=spec['reconstruction_source'],
            reconstruction_cohort=spec['reconstruction_cohort'],
            original_depth_plane_bytes_recovered=False)
    return arrays,receipt


def install_patient(stage,spec,*,formal_config=None):
    arrays,receipt=load_candidate(spec,formal_config=formal_config)
    from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,PhysxSchema,Gf,Vt
    if len(stage.GetLayerStack(includeSessionLayers=False))!=1:
        raise ValueError('Patient replacement requires a flattened independent stage')
    if not stage.GetPrimAtPath('/World/Patient'):raise ValueError('Original patient missing')
    def protected():
        return {str(p.GetPath()):dict(attrs={a.GetName():str(a.Get()) for a in p.GetAttributes()},
            rels={r.GetName():str(r.GetTargets()) for r in p.GetRelationships()})
            for p in stage.Traverse() if not (str(p.GetPath())=='/World/Patient' or str(p.GetPath()).startswith('/World/Patient/'))}
    before=protected();colliders=[]
    for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/Patient')):
        if prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            material,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose='physics')
            contact=PhysxSchema.PhysxCollisionAPI(prim)
            colliders.append((str(material.GetPath()),contact.GetContactOffsetAttr().Get(),contact.GetRestOffsetAttr().Get()))
    if not colliders or len(set(colliders))!=1 or colliders[0][0]!='/World/TrainingContactMaterial':
        raise ValueError('Original patient contact recipe is missing or inconsistent')
    material_path,contact_offset,rest_offset=colliders[0]
    if contact_offset is None or rest_offset is None:raise ValueError('Missing original contact offsets')
    material=UsdShade.Material(stage.GetPrimAtPath(material_path))
    physics_material=UsdPhysics.MaterialAPI(material.GetPrim())
    receipt['contact_recipe']=dict(material=material_path,contact_offset_m=contact_offset,
        rest_offset_m=rest_offset,static_friction=physics_material.GetStaticFrictionAttr().Get(),
        dynamic_friction=physics_material.GetDynamicFrictionAttr().Get(),
        restitution=physics_material.GetRestitutionAttr().Get(),source='unchanged nominal patient physics material')
    if not stage.RemovePrim('/World/Patient') or stage.GetPrimAtPath('/World/Patient'):
        raise ValueError('Previous composed patient survived removal')
    patient=UsdGeom.Xform.Define(stage,'/World/Patient')
    patient.SetResetXformStack(True)  # Candidate arrays are already world coordinates.
    mesh=UsdGeom.Mesh.Define(stage,'/World/Patient/SMPLSurface')
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(arrays['vertices'].astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr([3]*len(arrays['faces']))
    mesh.CreateFaceVertexIndicesAttr(arrays['faces'].reshape(-1).astype(int).tolist())
    mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateDoubleSidedAttr(False)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(.55,.38,.29)])
    mesh.CreateExtentAttr([Gf.Vec3f(*map(float,arrays['vertices'].min(0))),
                          Gf.Vec3f(*map(float,arrays['vertices'].max(0)))])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('none')
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material,materialPurpose='physics')
    contact=PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
    contact.CreateContactOffsetAttr(contact_offset);contact.CreateRestOffsetAttr(rest_offset)
    if protected()!=before:raise ValueError('Non-patient scene properties changed')
    receipt['patient_prim']='/World/Patient/SMPLSurface'
    receipt['non_patient_scene_unchanged']=True
    return receipt


def render_preview(app,filename,output,receipt):
    """Actual stopped-timeline Isaac images and metric depth; no physics claims."""
    from pxr import UsdGeom,Gf
    from PIL import Image
    import omni.usd
    import omni.timeline
    import omni.replicator.core as rep
    output=Path(output)
    omni.usd.get_context().open_stage(str(filename))
    for _ in range(35):app.update()
    stage=omni.usd.get_context().get_stage()
    omni.timeline.get_timeline_interface().stop()
    rgb=rep.AnnotatorRegistry.get_annotator('rgb')
    depth=rep.AnnotatorRegistry.get_annotator('distance_to_image_plane')
    product=rep.create.render_product('/World/RobotCam',(960,540))
    rgb.attach([product]);depth.attach([product])
    try:
        rep.orchestrator.step(rt_subframes=12,pause_timeline=True,delta_time=0.)
        image=np.asarray(rgb.get_data())[:,:,:3];d=np.asarray(depth.get_data())
        if image.shape!=(540,960,3) or image.std()<3:raise ValueError('Missing patient RGB')
        if d.shape!=(540,960):raise ValueError('Missing patient metric depth')
        Image.fromarray(image).save(output/'patient_rgb.png')
        np.savez_compressed(output/'patient_depth.npz',depth_m=d,valid=np.isfinite(d)&(d>0))
        # Markers are visualization-only and are added AFTER saving the clean
        # scene/RGB/depth. None is a collider or part of a training observation.
        for target in receipt['targets']:
            mark=UsdGeom.Sphere.Define(stage,'/World/M2AnchorReview/'+target['name'])
            mark.CreateRadiusAttr(.003)
            mark.AddTranslateOp().Set(Gf.Vec3d(*target['position_world_m']))
            mark.CreateDisplayColorAttr([Gf.Vec3f(.1,.9,.3)])
        rep.orchestrator.step(rt_subframes=12,pause_timeline=True,delta_time=0.)
        Image.fromarray(np.asarray(rgb.get_data())[:,:,:3]).save(output/'patient_anchor_review.png')
    finally:rgb.detach();depth.detach();product.destroy()
    return dict(actual_isaac_images=True,physics_executed=False,training_eligible=False,
                camera='/World/RobotCam',camera_pose_unchanged=True,
                clean_rgb='patient_rgb.png',metric_depth='patient_depth.npz',
                diagnostic_markers='patient_anchor_review.png')
