"""Synthetic CPU controls exercise analysis; they cannot issue a native permit."""
from copy import deepcopy
import importlib
import importlib.util
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sim.ausc_policy import contact_semantics_oracle as oracle
from sim.ausc_policy import m2_given_contact_runtime as rt


def module():
    name='sim.ausc_policy.retainer_witness_qualification'
    assert importlib.util.find_spec(name) is not None,'Independent native witness issuer missing'
    return importlib.import_module(name)


def fixture():
    specs=oracle.cases();events=[];journal=[];ticks=[]
    for i,c in enumerate(specs):
        rotation=Rotation.from_euler('xyz',[19,27,31],degrees=True) if c['transformed'] else Rotation.identity()
        r=rotation.as_matrix();translation=np.array([(i%9)*.25,(i//9)*.25,1.])
        face=oracle.support_face(c['feature']);outward=r@np.array(face['outward_local'])
        plane=r@np.array(face['center_local'])+translation+c['gap_m']*outward
        plate=oracle.plate_fixture(plane,outward);plate.pop('points')
        root='/World/Cases/'+c['id']
        c.update(body_path=root+'/Wedge',shape_path=root+'/Wedge/Shape',plate_path=root+'/Plate',
            support_face=face,initial_body_position_world=translation.tolist(),
            initial_body_quaternion_xyzw=rotation.as_quat().tolist(),
            initial_velocity_world=(c['speed_m_s']*outward).tolist(),**plate)
    def frame(c,step,label):
        r=Rotation.from_quat(c['initial_body_quaternion_xyzw']).as_matrix();m=np.eye(4);m[:3,:3]=r
        m[:3,3]=np.array(c['initial_body_position_world'])+step*1e-6*np.array(c['initial_velocity_world'])/.02
        return dict(sample=label,source='omni.physx.IPhysx.get_rigidbody_transformation',path=c['body_path'],
            available=True,world_from_wedge=m.tolist(),contact_generation_pose_claimed=False)
    for step in range(12):
        post={c['id']:frame(c,step+1,'post_step_current') for c in specs};indices=[]
        for c in specs:
            n=-np.array(c['wedge_to_plate_axis_world']);s=c['gap_m']-step*1e-6*c['speed_m_s']/.02
            positive=c['gap_m']==0. or c['speed_m_s']>0.
            raw=dict(colliders=[c['shape_path'],c['plate_path']],actors=[c['body_path'],c['plate_path']],
                position_world=c['plate_origin_world'],normal_world=n.tolist(),separation_m=s,
                impulse_Ns=(n*oracle.DT*(1. if positive else 0.)).tolist(),
                native_fields_missing=[],native_field_errors={},header_field_errors={},
                face_index0=0,face_index1=1,face_index0_available=True,face_index1_available=True,
                callback_index=None,event_type='CONTACT_PERSIST',stage_id=1)
            index=len(events);indices.append(index)
            event=dict(event_index=index,case=c['id'],physics_row=step,raw=raw,execution_stage='controlled_step',
                pre=frame(c,step,'pre_step_current'),callback_current=frame(c,step+1,'callback_current'),
                post=dict(available=False,error='No explicit post pose for SDK initialization step'))
            journal.append(dict(kind='contact_callback',event=deepcopy(event)))
            event['post']=post[c['id']];events.append(event)
        tick=dict(physics_row=step,time_before_s=step*oracle.DT,time_after_s=(step+1)*oracle.DT,events=len(indices))
        ticks.append(tick);journal.append(dict(kind='post_step',tick=tick,post=post,updated_event_indices=indices))
    by_id={c['id']:c for c in specs};summaries=[]
    for e in events:
        c=by_id[e['case']];raw=e['raw']
        row=dict(case=c['id'],feature=c['feature'],physics_row=e['physics_row'],execution_stage=e['execution_stage'],
            force_N=float(np.linalg.norm(raw['impulse_Ns'])/oracle.DT),retainer_pair_index=0,
            separation_m=raw['separation_m'],face_index0=0,face_index1=1,native_field_errors={},by_pose={},unavailable=[])
        for label in ('pre','callback_current','post'):
            row['by_pose'][label]=oracle.point_residuals(raw['position_world'],raw['normal_world'],raw['separation_m'],0,
                e[label]['world_from_wedge'],np.array(c['plate_origin_world']),np.array(c['plate_normal_world']),c['support_face'])
        summaries.append(row)
    return specs,events,journal,ticks,summaries


def test_full_raw_matrix_independent_rule_and_front_side_back_negative_controls():
    result=module().analyze(*fixture())
    assert result['passed'] is True and result['cases']==72 and result['controlled_rows']==864
    assert result['journal_reconstruction_exact'] is True
    assert result['feature_counts']['front']['positive']>0
    assert result['feature_counts']['front']['accepted']==result['feature_counts']['front']['positive']
    assert result['feature_counts']['side']['accepted']==result['feature_counts']['back']['accepted']==0
    assert result['midpoint_negative_control_max_abs_m']>1e-4
    assert result['pair1_qualified'] is False


@pytest.mark.parametrize('feature',['front','side','back'])
def test_analytic_classifier_agrees_with_original_without_using_its_planes(feature,monkeypatch):
    m=module();face=oracle.support_face(feature);w=np.array(face['center_local']);n=-np.array(face['outward_local'])
    result=m.independent_feature(np.eye(4),w,n)
    assert result['passed'] is (feature=='front')
    from sim.ausc_policy import retainer_witness as witness
    original=witness.sector_feature_audit(np.eye(4),w,n,0)
    assert original['passed']==result['passed']
    monkeypatch.setattr(witness,'sector_planes',lambda *args:(_ for _ in ()).throw(AssertionError('not independent')))
    assert m.independent_feature(np.eye(4),w,n)['passed']==result['passed']


@pytest.mark.parametrize('alter',[
    lambda v:v[1][0]['raw'].update(separation_m=.02),
    lambda v:v[2][0]['event']['raw'].update(separation_m=.02),
    lambda v:v[2].pop(),
    lambda v:v[2].append(dict(kind='callback_error',error='native callback failed')),
    lambda v:v[3][0].update(events=0),
    lambda v:v[4][0]['by_pose']['pre']['ring_face_residual_m'].update({'1.0':.001}),
    lambda v:v[0][0].update(gap_m=.1),
    lambda v:v[1][0]['pre'].update(available=False),
    lambda v:v[1][0]['raw'].update(native_fields_missing=['normal']),
    lambda v:v[1][0]['raw'].update(colliders=list(reversed(v[1][0]['raw']['colliders']))),
])
def test_tampering_and_missing_native_controls_fail_closed(alter):
    values=list(fixture());alter(values)
    with pytest.raises(ValueError):module().analyze(*values)


def test_independent_classifier_rejects_flipped_side_front_normal_and_nonrigid_frames():
    m=module();face=oracle.support_face('side');w=np.array(face['center_local'])
    assert not m.independent_feature(np.eye(4),w,[0,0,1])['passed']
    frame=np.eye(4);frame[0,0]=1.01
    with pytest.raises(ValueError):m.independent_feature(frame,w,[0,0,1])


def test_cpu_fixture_cannot_issue_receipt_even_with_terminal_status(tmp_path,monkeypatch):
    m=module();monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    ref=rt.write_new(tmp_path/'result.json',dict(cpu_diagnostic_only=True,completed=True,gpu_released=True,
        ended_utc='2026-11-05T00:00:00Z',config=dict(experiment='contact_semantics_oracle'),
        stage=dict(passed=True,exit_code=0,process_group_drained=True)))
    with pytest.raises(ValueError,match='CPU|native'):m.qualify(ref,tmp_path/'issued')
    assert not (tmp_path/'issued'/'QUALIFIED.json').exists()


def test_complete_case_and_contact_counts_are_independently_checked():
    m=module();values=fixture();analysis=m.analyze(*values)
    complete=oracle.evidence_status(values[0],values[4])
    complete.update(contact_rows=len(values[1]),controlled_start_time_s=values[3][0]['time_before_s'])
    m.validate_summary(complete,analysis,values[3])
    complete['cases'][0]['contacts']-=1
    with pytest.raises(ValueError,match='count|case'):m.validate_summary(complete,analysis,values[3])
