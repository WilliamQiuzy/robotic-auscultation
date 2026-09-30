import numpy as np
import pytest

CONTACT_SURFACE = "membrane_and_device_retainer_v2"
from sim.ausc_policy import geometry_epoch
from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics


def test_component_frames_match_actual_132mm_kinematics_under_compression():
    from sim.ausc_policy import finite_head
    kin = OpenArmKinematics(contact_frame=geometry_epoch.contact_frame)
    q = np.array([.2,-.5,.2,1.1,-.3,.1,.2])
    for compression in (0.,.0025,.006):
        wrist = kin.fk(q); tcp = kin.tcp(q,compression)
        fixed, moving = finite_head.component_frames(tcp[:3,3],wrist[:3,:3],compression)
        adapter=np.diag([1.,-1.,-1.,1.]);adapter[2,3]=.085
        slider=np.eye(4);slider[2,3]=compression
        np.testing.assert_allclose(fixed,wrist@adapter,atol=1e-12)
        np.testing.assert_allclose(moving,wrist@adapter@slider,atol=1e-12)


def test_contact_frames_reject_invalid_physical_domain():
    from sim.ausc_policy import finite_head
    for compression in (-.001,.0061,float('nan')):
        with pytest.raises(ValueError):finite_head.component_frames(np.zeros(3),np.eye(3),compression)
    with pytest.raises(ValueError):finite_head.component_frames(np.zeros(3),np.zeros((3,3)),0.)


def test_candidate_family_reaches_diagonals_without_expanding_original_square():
    from sim.ausc_policy.finite_head import FiniteHeadFit
    fit=object.__new__(FiniteHeadFit)
    fit.first_contact=lambda position,normal,rotation,compression: dict(
        contact_surface=CONTACT_SURFACE, passed=True, center_world_m=(np.asarray(position)+.00002*normal).tolist(),
        surface_gap_m=.00002, contact_surface_gap_m=0.,
        contact_face_residual_distance_m=.00002, first_contact_search_resolution_m=.000001,
        contact_bracket_width_m=.000001, contact_numeric_tolerance_m=1e-9,
        raw_anchor_world_m=np.asarray(position).tolist(), fit=dict(passed=True),
        fit_evaluations=1)
    normal=np.array([.3,.4,np.sqrt(.75)])
    x=np.cross(normal,[0,0,1]);x/=np.linalg.norm(x);y=np.cross(normal,x)
    result=fit.candidates(np.zeros(3),normal)
    assert {35.,40.}.issubset({c['normal_deviation_from_surface_deg'] for c in result['candidates']})
    for c in result['candidates']:
        n=np.array(c['normal_world']);uv=np.array([n@x,n@y])/(n@normal)
        assert np.max(abs(uv))<=.6+1e-12


def plane_fit(*, tilt_touch_m=0., blocked_below_m=None, never_touch=False,
              contact_roundoff=False):
    """Analytical plane/disc oracle; tests search only, not native FCL."""
    from sim.ausc_policy.finite_head import FiniteHeadFit
    fit=object.__new__(FiniteHeadFit)
    queries=[]
    def query(center, rotation, compression):
        gap=float(center[2])-tilt_touch_m
        distance=(gap+.001 if never_touch else max(gap,0.))
        collision=gap<=0 and not never_touch and not contact_roundoff
        if contact_roundoff and gap==0:distance=1e-17
        distances=dict(fixed=.01,clamp=.01,shell=.001,contact_face=distance)
        if blocked_below_m is not None and center[2]<blocked_below_m:
            distances['clamp']=.0025
        queries.append((np.asarray(center).copy(),compression))
        return dict(contact_surface=CONTACT_SURFACE,passed=False,distances_m=distances,contact_face_collision=collision,
                    full_robot_collision_checked=False,physics_executed=False)
    fit.check=query
    return fit,queries


@pytest.mark.parametrize('roundoff', [False,True])
def test_first_contact_plane_replaces_millimetre_gap_with_measured_near_contact(roundoff):
    fit,queries=plane_fit(contact_roundoff=roundoff)
    anchor=np.zeros(3);before=anchor.copy()
    result=fit.first_contact(anchor,[0,0,1],np.diag([1.,-1.,-1.]),.0025)
    assert result['passed']
    assert result['contact_surface']==CONTACT_SURFACE
    assert 1e-5<=result['contact_face_residual_distance_m']<=3e-5
    assert result['surface_gap_m']<.00003
    assert abs(result['contact_surface_gap_m'])<=result['first_contact_search_resolution_m']
    assert result['contact_bracket_width_m']<=result['first_contact_search_resolution_m']
    np.testing.assert_array_equal(anchor,before)
    np.testing.assert_array_equal(result['raw_anchor_world_m'],anchor)
    np.testing.assert_allclose(result['center_world_m'],queries[-1][0],atol=0,rtol=0)
    assert all(q[1]==.0025 for q in queries)


def test_tilted_finite_disc_stops_at_first_edge_contact_not_raw_anchor():
    from scipy.spatial.transform import Rotation
    angle=np.deg2rad(20.)
    radius=.01
    touch=radius*np.sin(angle)
    fit,_=plane_fit(tilt_touch_m=touch)
    rotation=Rotation.from_euler('y',angle).as_matrix()@np.diag([1.,-1.,-1.])
    result=fit.first_contact(np.zeros(3),[0,0,1],rotation,.003)
    assert result['passed']
    assert abs(result['contact_surface_gap_m']-radius*np.tan(angle))<=1e-6
    assert result['surface_gap_m']>radius*np.tan(angle)
    assert 1e-5<=result['contact_face_residual_distance_m']<=3e-5
    np.testing.assert_allclose(result['center_world_m'],-rotation[:,2]*result['surface_gap_m'])
    assert result['gap_axis']=='command_normal'
    assert result['raw_anchor_world_m']==[0.,0.,0.]


def test_lip_obstruction_cannot_be_accepted_by_changing_shell_threshold():
    fit,_=plane_fit(blocked_below_m=.0002)
    result=fit.first_contact(np.zeros(3),[0,0,1],np.diag([1.,-1.,-1.]),.0025)
    assert not result['passed']
    assert result['reason']=='noncontact_obstruction'
    assert result['rejected_fit']['distances_m']['clamp']==.0025


def test_missing_contact_bracket_is_rejected_even_with_old_1_5mm_fit():
    fit,_=plane_fit(never_touch=True)
    result=fit.first_contact(np.zeros(3),[0,0,1],np.diag([1.,-1.,-1.]),.0025)
    assert not result['passed']
    assert result['reason']=='missing_contact_bracket'


@pytest.mark.parametrize('kwargs', [
    dict(position=[np.nan,0,0]),dict(local_normal=[0,0,0]),
    dict(local_normal=[0,0,np.nan]),dict(rotation=np.zeros((3,3))),
    dict(compression=-.001),dict(compression=.0061),
])
def test_first_contact_rejects_bad_inputs(kwargs):
    fit,_=plane_fit()
    inputs=dict(position=np.zeros(3),local_normal=[0,0,1],rotation=np.diag([1.,-1.,-1.]),compression=.0025)
    inputs.update(kwargs)
    with pytest.raises(ValueError):fit.first_contact(**inputs)


def test_first_contact_rejects_nonfinite_distance():
    fit,_=plane_fit()
    fit.check=lambda *args:dict(contact_surface=CONTACT_SURFACE,distances_m=dict(fixed=.01,clamp=.01,shell=.001,contact_face=np.nan),
                               contact_face_collision=False)
    with pytest.raises(ValueError,match='distance'):
        fit.first_contact(np.zeros(3),[0,0,1],np.diag([1.,-1.,-1.]),.0025)


def test_first_contact_finds_thin_surface_even_when_both_search_endpoints_are_clear():
    fit,_=plane_fit()
    def thin_surface(center,rotation,compression):
        z=float(center[2]);lower=.003;upper=.0031
        inside=lower<=z<=upper
        return dict(contact_surface=CONTACT_SURFACE,distances_m=dict(fixed=.01,clamp=.01,shell=.001,
            contact_face=0. if inside else min(abs(z-lower),abs(z-upper))),contact_face_collision=inside)
    fit.check=thin_surface
    assert not thin_surface([0,0,0],None,None)['contact_face_collision']
    assert not thin_surface([0,0,.005],None,None)['contact_face_collision']
    result=fit.first_contact(np.zeros(3),[0,0,1],np.diag([1.,-1.,-1.]),.0025)
    assert result['passed']
    assert abs(result['contact_surface_gap_m']-.0031)<=1e-6
    assert result['surface_gap_m']>.0031


@pytest.mark.parametrize('part,clearance', [('fixed',.002999),('shell',.000299)])
def test_first_contact_never_lowers_noncontact_thresholds(part,clearance):
    fit,_=plane_fit()
    original=fit.check
    def blocked(*args):
        result=original(*args);result['distances_m'][part]=clearance
        return result
    fit.check=blocked
    result=fit.first_contact(np.zeros(3),[0,0,1],np.diag([1.,-1.,-1.]),.0025)
    assert not result['passed'] and result['reason']=='noncontact_obstruction'


def test_final_near_contact_position_rechecks_all_noncontact_parts():
    fit,_=plane_fit()
    original=fit.check;counts={}
    def changing_final(*args):
        result=original(*args)
        key=tuple(args[0]);counts[key]=counts.get(key,0)+1
        d=result['distances_m']['contact_face']
        if 1e-5<=d<=3e-5 and counts[key]>1:
            result['distances_m']['clamp']=.002
        return result
    fit.check=changing_final
    result=fit.first_contact(np.zeros(3),[0,0,1],np.diag([1.,-1.,-1.]),.0025)
    assert not result['passed'] and result['reason']=='noncontact_obstruction'
    assert result['rejected_fit']['distances_m']['clamp']==.002


def test_already_touching_outer_endpoint_has_no_valid_outside_bracket():
    fit,_=plane_fit(tilt_touch_m=.0051)
    result=fit.first_contact(np.zeros(3),[0,0,1],np.diag([1.,-1.,-1.]),.0025)
    assert not result['passed'] and result['reason']=='outer_endpoint_not_clear'


@pytest.mark.parametrize('roundoff', [-1e-16,0.,1e-16])
def test_contact_bracket_handles_equal_noncontact_boundaries_without_relaxing_final_gate(roundoff):
    fit,_=plane_fit(contact_roundoff=True)
    original=fit.check
    def boundary_roundoff(*args):
        result=original(*args);gap=float(args[0][2])
        result['distances_m'].update(fixed=.003+gap+roundoff,clamp=.003+gap+roundoff,
                                    shell=.0003+gap+roundoff)
        return result
    fit.check=boundary_roundoff
    result=fit.first_contact(np.zeros(3),[0,0,1],np.diag([1.,-1.,-1.]),.0025)
    assert result['passed']
    assert result['fit']['distances_m']['fixed']>=.003
    assert result['fit']['distances_m']['clamp']>=.003
    assert result['fit']['distances_m']['shell']>=.0003
    assert result['bracket_noncontact_numeric_tolerance_m']==1e-9
    assert result['first_contact_search_min_step_m']==1e-9


def test_contact_boundary_violation_larger_than_one_nm_is_rejected():
    fit,_=plane_fit();original=fit.check
    def blocked_boundary(*args):
        result=original(*args)
        result['distances_m']['shell']=.0003+float(args[0][2])-3e-9
        return result
    fit.check=blocked_boundary
    result=fit.first_contact(np.zeros(3),[0,0,1],np.diag([1.,-1.,-1.]),.0025)
    assert not result['passed'] and result['reason']=='noncontact_obstruction'


def test_final_near_contact_has_no_one_nm_clearance_exemption():
    fit,_=plane_fit();original=fit.check;counts={}
    def final_roundoff(*args):
        result=original(*args);key=tuple(args[0]);counts[key]=counts.get(key,0)+1
        if 1e-5<=result['distances_m']['contact_face']<=3e-5 and counts[key]>1:
            result['distances_m']['shell']=.0003-1e-16
        return result
    fit.check=final_roundoff
    result=fit.first_contact(np.zeros(3),[0,0,1],np.diag([1.,-1.,-1.]),.0025)
    assert not result['passed'] and result['reason']=='noncontact_obstruction'


@pytest.mark.parametrize('policy', [None, 'diaphragm_only_v1'])
def test_legacy_contact_observation_cannot_be_relabelled_as_current(policy):
    from sim.ausc_policy.finite_head import _observation
    report=dict(distances_m=dict(fixed=.01,clamp=.01,shell=.001,contact_face=.00002),
                contact_face_collision=False)
    if policy is not None:report['contact_surface']=policy
    with pytest.raises(ValueError,match='contact.surface'):_observation(report)


def exported_plane_fixture(tmp_path, monkeypatch, *, omitted=(), extra=()):
    """Record exported meshes and inject analytic plane distances; no native FCL.

    Real constructor, grouping, transform and search code run. The backend's
    only geometric oracle is the minimum transformed z above an infinite plane.
    """
    import hashlib
    import json
    import sys
    from types import SimpleNamespace

    class Model:
        def beginModel(self,*args):pass
        def addSubModel(self,vertices,faces):
            self.vertices=vertices.copy();self.faces=faces.copy()
        def endModel(self):pass
    class Object:
        def __init__(self,model):
            self.model=model;self.transform=(np.eye(3),np.zeros(3))
        def setTransform(self,transform):self.transform=transform
    def signed_height(obj,patient):
        rotation,translation=obj.transform
        return np.min((obj.model.vertices@rotation.T+translation)[:,2])
    monkeypatch.setitem(sys.modules,'fcl',SimpleNamespace(
        BVHModel=Model,CollisionObject=Object,Transform=lambda r,t:(r,t),
        distance=lambda a,b:max(0.,signed_height(a,b)),
        collide=lambda a,b,*args:int(signed_height(a,b)<=0.),
        CollisionRequest=object,CollisionResult=object))
    prefix='/World/Stethoscope/Carriage/Colliders/'
    specs=[('/World/Patient/Body','/World/Patient',0.),
           ('/World/Stethoscope/Fixed/Colliders/Frame','/World/Stethoscope/Fixed',-.03),
           (prefix+'ContactRim/sector_00','/World/Stethoscope/Carriage',-.042),
           (prefix+'RetentionLip/sector_00','/World/Stethoscope/Carriage',-.041),
           (prefix+'Diaphragm','/World/Stethoscope/Carriage',-.046)]
    specs += [(prefix+part,'/World/Stethoscope/Carriage',-.0455)
              for part in ('DeviceEdge','DeviceCone','DeviceDial')]
    specs += [(prefix+f'DeviceContactRing/sector_{i:02d}','/World/Stethoscope/Carriage',-.047)
              for i in range(32)]
    specs += list(extra)
    arrays={};entries=[]
    for i,(path,body,z) in enumerate(specs):
        if path in omitted:continue
        key=f'mesh_{i}'
        arrays[key+'_v']=np.array([[-.01,-.01,z],[.01,-.01,z],[0,.01,z]])
        arrays[key+'_f']=np.array([[0,1,2]])
        entries.append(dict(path=path,body=body,key=key,world_from_body=np.eye(4).tolist()))
    (tmp_path/'collision_geometry.json').write_text(json.dumps(dict(colliders=entries)))
    np.savez(tmp_path/'collision_geometry.npz',**arrays)
    metadata=dict(geometry_id=geometry_epoch.GEOMETRY_ID,freshly_generated=True,
                  geometry_source_sha256=geometry_epoch.GEOMETRY_SOURCE_SHA256)
    for filename,key in [('collision_geometry.json','collision_geometry_sha256'),
                         ('collision_geometry.npz','collision_arrays_sha256')]:
        metadata[key]=hashlib.sha256((tmp_path/filename).read_bytes()).hexdigest()
    return metadata


def test_exported_membrane_and_all_retainer_sectors_form_contact_union(tmp_path,monkeypatch):
    from sim.ausc_policy.finite_head import FiniteHeadFit
    metadata=exported_plane_fixture(tmp_path,monkeypatch)
    fit=FiniteHeadFit(tmp_path,metadata)
    assert set(fit.objects)=={'patient','fixed','clamp','shell','contact_face'}
    face=fit.objects['contact_face'].model
    assert len(face.faces)==33
    assert set(face.vertices[:,2])=={-.046,-.047}
    assert len(fit.objects['shell'].model.faces)==3
    assert len(fit.objects['clamp'].model.faces)==2
    result=fit.first_contact(np.zeros(3),[0,0,1],np.diag([1.,-1.,-1.]))
    assert result['passed']
    assert 1e-5<=result['contact_face_residual_distance_m']<=3e-5
    assert abs(result['contact_surface_gap_m'])<=1e-6
    assert result['surface_gap_m']<3e-5  # C first; membrane B remains recessed by 1 mm.
    assert result['fit']['contact_surface']==CONTACT_SURFACE
    assert 'diaphragm' not in repr(result)


@pytest.mark.parametrize('missing', ['all_ring','one_sector','membrane'])
def test_legacy_or_incomplete_contact_geometry_is_rejected(tmp_path,monkeypatch,missing):
    from sim.ausc_policy.finite_head import FiniteHeadFit
    prefix='/World/Stethoscope/Carriage/Colliders/'
    omitted=([prefix+f'DeviceContactRing/sector_{i:02d}' for i in range(32)] if missing=='all_ring'
             else [prefix+'DeviceContactRing/sector_17'] if missing=='one_sector'
             else [prefix+'Diaphragm'])
    metadata=exported_plane_fixture(tmp_path,monkeypatch,omitted=omitted)
    with pytest.raises(ValueError,match='contact.surface|contact geometry'):
        FiniteHeadFit(tmp_path,metadata)


def test_retainer_name_on_other_path_cannot_remove_clamp_clearance(tmp_path,monkeypatch):
    from sim.ausc_policy.finite_head import FiniteHeadFit
    metadata=exported_plane_fixture(tmp_path,monkeypatch,extra=[
        ('/World/Stethoscope/Carriage/Other/DeviceContactRing/sector_00',
         '/World/Stethoscope/Carriage',-.046)])
    fit=FiniteHeadFit(tmp_path,metadata)
    result=fit.first_contact(np.zeros(3),[0,0,1],np.diag([1.,-1.,-1.]))
    assert not result['passed'] and result['reason']=='noncontact_obstruction'
    assert result['rejected_fit']['distances_m']['clamp']<.003


@pytest.mark.parametrize('missing_array',['vertices','faces'])
def test_retainer_sector_with_empty_mesh_cannot_certify_current_contact_surface(tmp_path,monkeypatch,missing_array):
    import hashlib
    import json
    from sim.ausc_policy.finite_head import FiniteHeadFit
    metadata=exported_plane_fixture(tmp_path,monkeypatch)
    entries=json.loads((tmp_path/'collision_geometry.json').read_text())['colliders']
    key=next(e['key'] for e in entries if e['path'].endswith('DeviceContactRing/sector_17'))
    path=tmp_path/'collision_geometry.npz'
    with np.load(path) as saved:arrays={name:saved[name] for name in saved.files}
    arrays[key+('_v' if missing_array=='vertices' else '_f')]=np.empty((0,3))
    np.savez(path,**arrays)
    metadata['collision_arrays_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='contact geometry'):FiniteHeadFit(tmp_path,metadata)


@pytest.mark.parametrize('corruption',[
    'all_ring_keys_alias_membrane','shared_key_other_group','shared_key_ignored_entry',
    'duplicate_path','negative_face_index','fractional_face_index','float_face_dtype',
    'face_index_out_of_range','unsigned_face_index_out_of_range','nonfinite_vertices',
    'wrong_vertex_shape','wrong_face_shape','nonnumeric_vertices','empty_shell_vertices',
    'empty_patient_faces',
])
def test_hash_bound_corrupt_mesh_export_rejects_before_native_calls(tmp_path,monkeypatch,corruption):
    """A matching artifact hash does not make aliased or invalid meshes safe."""
    import hashlib
    import json
    import sys
    from sim.ausc_policy.finite_head import FiniteHeadFit

    metadata=exported_plane_fixture(tmp_path,monkeypatch)
    manifest=tmp_path/'collision_geometry.json';archive=tmp_path/'collision_geometry.npz'
    entries=json.loads(manifest.read_text())['colliders']
    with np.load(archive) as saved:arrays={name:saved[name] for name in saved.files}
    membrane=next(e for e in entries if e['path'].endswith('/Diaphragm'))
    retainer=next(e for e in entries if e['path'].endswith('DeviceContactRing/sector_17'))
    key=retainer['key']
    if corruption=='all_ring_keys_alias_membrane':
        for entry in entries:
            if '/DeviceContactRing/' in entry['path']:entry['key']=membrane['key']
    elif corruption=='shared_key_other_group':
        retainer['key']=entries[0]['key']
    elif corruption=='shared_key_ignored_entry':
        entries.append(dict(path='/World/Other/Mesh',body='/World/Other',key=key))
    elif corruption=='duplicate_path':
        # Use a non-contact path so the contact inventory still has 33 entries.
        duplicate=dict(entries[1],key='duplicate_mesh')
        entries.append(duplicate)
        arrays['duplicate_mesh_v']=arrays[entries[1]['key']+'_v'].copy()
        arrays['duplicate_mesh_f']=arrays[entries[1]['key']+'_f'].copy()
    elif corruption=='negative_face_index':arrays[key+'_f']=np.array([[-1,1,2]])
    elif corruption=='fractional_face_index':arrays[key+'_f']=np.array([[0.,1.5,2.]])
    elif corruption=='float_face_dtype':arrays[key+'_f']=np.array([[0.,1.,2.]])
    elif corruption=='face_index_out_of_range':arrays[key+'_f']=np.array([[0,1,3]])
    elif corruption=='unsigned_face_index_out_of_range':
        arrays[key+'_f']=np.array([[0,1,2**32]],dtype=np.uint64)
    elif corruption=='nonfinite_vertices':arrays[key+'_v'][1,1]=np.nan
    elif corruption=='wrong_vertex_shape':arrays[key+'_v']=np.zeros((3,2))
    elif corruption=='wrong_face_shape':arrays[key+'_f']=np.array([[0,1]])
    elif corruption=='nonnumeric_vertices':arrays[key+'_v']=arrays[key+'_v'].astype(str)
    elif corruption=='empty_shell_vertices':
        shell=next(e for e in entries if e['path'].endswith('/DeviceEdge'))
        arrays[shell['key']+'_v']=np.empty((0,3))
    elif corruption=='empty_patient_faces':arrays[entries[0]['key']+'_f']=np.empty((0,3),int)
    manifest.write_text(json.dumps(dict(colliders=entries)));np.savez(archive,**arrays)
    metadata['collision_geometry_sha256']=hashlib.sha256(manifest.read_bytes()).hexdigest()
    metadata['collision_arrays_sha256']=hashlib.sha256(archive.read_bytes()).hexdigest()

    # A failed preflight must precede native model allocation, not rely on an
    # FCL failure after lossy dtype conversion to discover invalid input.
    native_calls=[];original=sys.modules['fcl'].BVHModel
    def model():
        native_calls.append('BVHModel')
        return original()
    monkeypatch.setattr(sys.modules['fcl'],'BVHModel',model)
    with pytest.raises(ValueError,match='geometry|mesh'):
        FiniteHeadFit(tmp_path,metadata)
    assert native_calls==[]
