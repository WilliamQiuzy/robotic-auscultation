"""Declared 35-node R4 direct-entry geometry screen, never a capture success rate.

CPU-only IK, approved posture/FCL and original full normal-curve checks. No
physics resets, prepared observations, learned actions or tolerance changes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation


COMPONENT_NODES={'entry-tangent-plus':'t0_+0.00001','entry-normal-plus':'t2_+0.00200'}


def load_geometry_case(config):
    binding=config['geometry_probe'];root=Path(binding['path']);name=config['case_id']
    if name not in COMPONENT_NODES or set(binding['cases'])!=set(config['case_ids']):
        raise ValueError('Unregistered actual component geometry')
    item=binding['cases'][name]
    if item['node']!=COMPONENT_NODES[name]: raise ValueError('Geometry node remapped')
    files={'result.json':binding['owner_sha256'],'output/COMPLETE.json':binding['complete_sha256'],
           'output/'+item['node']+'.json':item['sha256']}
    for relative,digest in files.items():
        if hashlib.sha256((root/relative).read_bytes()).hexdigest()!=digest:
            raise ValueError('Completed geometry source changed')
    owner=json.loads((root/'result.json').read_text())
    complete=json.loads((root/'output/COMPLETE.json').read_text())
    case=json.loads((root/'output'/(item['node']+'.json')).read_text())
    if not (owner['completed'] and owner['gpu_released'] and owner['stage']['process_group_drained']
            and owner['stage']['exit_code']==0 and owner['config']['experiment']=='capture_probe'
            and owner['config']['core_assets']==config['core_assets'] and owner['config']['point']=='R4'
            and complete['completed'] and complete['case_count']==35 and case in complete['cases']
            and case['case']['id']==item['node']
            and case['ik_accurate'] and case['free_geometry_passed'] and case['direct_entry_geometry_passed']
            and case['full_curve_geometry_passed'] and case['training_dataset_eligible'] is False):
        raise ValueError('Completed explicit direct-entry geometry required')
    q=np.asarray(case['q'],float)
    if q.shape!=(7,) or not np.isfinite(q).all(): raise ValueError('Invalid geometry command')
    return case


def declared_cases():
    cases=[dict(id='center',kind='center',axis=None,value=0.)]
    for axis in range(3):
        values=(.00001,.00002,.0001) if axis<2 else (.0005,.002)
        for amount in values:
            for sign in (-1,1):
                value=sign*amount
                cases.append(dict(id=f't{axis}_{value:+.5f}',kind='translation',axis=axis,value=value))
    for axis in range(3):
        for amount in (.0001,.0002,.001):
            for sign in (-1,1):
                value=sign*amount
                cases.append(dict(id=f'r{axis}_{value:+.4f}',kind='rotation',axis=axis,value=value))
    return cases


def perturb_pose(pose,normal,case):
    pose=np.asarray(pose,float);normal=np.asarray(normal,float)
    if (pose.shape!=(4,4) or normal.shape!=(3,) or not np.isfinite(pose).all()
            or not np.isfinite(normal).all() or not np.allclose(pose[3],[0,0,0,1],atol=1e-12,rtol=0)
            or not np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-8,rtol=0)
            or not np.isclose(np.linalg.det(pose[:3,:3]),1,atol=1e-8,rtol=0)
            or not np.allclose(normal,-pose[:3,2],atol=1e-6,rtol=0)
            or case not in declared_cases()):
        raise ValueError('Approved node and orthogonal tangent/outward-normal basis required')
    result=pose.copy()
    if case['kind']=='center': return result
    axis=pose[:3,case['axis']] if case['axis']<2 else normal
    if case['kind']=='translation': result[:3,3]+=axis*case['value']
    else: result[:3,:3]=Rotation.from_rotvec(axis*case['value']).as_matrix()@pose[:3,:3]
    return result


def main():
    from .isaac_case import prepare,write
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();config=json.loads(a.config.read_text())
    if (sys.platform!='linux' or os.environ.get('AUSC_OWNED_RUN')!='1'
            or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or config['experiment']!='capture_probe'
            or config['point']!='R4' or config['case_definition']!='handoff-report-35-v1'):
        raise ValueError('Owned finite CPU-only R4 direct-entry geometry screen required')
    a.output.mkdir(parents=True,exist_ok=False)
    bundle,_,_=prepare(config,a.output/'geometry_source')
    from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
    from sim.auscultation_expert_layout_v1.pose_ik import solve_posture_ik, IKBudgetExpired
    from sim.auscultation_expert_layout_v1.posture import PostureChecker,PostureLimits
    from sim.auscultation_skill_v1.certified_collision import CertifiedCollisionChecker
    from sim.auscultation_skill_v1.curve import FixedNormalCurve,force_derived_domain
    from sim.auscultation_skill_v1.controller import CONTACT_DEFAULTS
    from sim.curobo_expert_v1.static_world import load_static_world,install_fcl_static_world
    from .contact_entry import check_retained_pose
    kin=OpenArmKinematics();base=np.asarray(bundle.plan['world_from_left_base']);row=bundle.row
    folder=a.output/'geometry_source';checker=CertifiedCollisionChecker(kin,base,directory=folder/'geometry')
    install_fcl_static_world(checker,load_static_world(folder/'static_world'),geometry_dir=folder/'geometry')
    limits=PostureLimits.from_dict(row['posture_limits'],kin);checker=PostureChecker(checker,base,limits)
    q0=np.asarray(row['transfer_q'][-1]);target=np.asarray(bundle.target.position_base)
    rotation=base[:3,:3].T@np.asarray(row['command_rotation_world']);normal=-rotation[:,2]
    nominal=kin.tcp(q0)
    # Retain the approved pose/frame, including its own numerical residual.
    # All requested perturbations are applied around that actual command FK.
    if not np.allclose(nominal[:3,2],rotation[:,2],atol=1e-6,rtol=0):
        raise ValueError('Canonical pose axis differs from approved task')
    loaded_gap=float((kin.tcp(np.asarray(row['loaded_q']))[:3,3]-target)@normal)
    domain=force_derived_domain(loaded_gap,row['loaded_compression_m'],base[:3,:3]@normal,
                               4.,CONTACT_DEFAULTS['nominal_compression_reserve_m'])
    def contact_check(q,gap):
        return checker.check(q,clearance=CONTACT_DEFAULTS['contact_clearance_m'],
            allow_patient_contact=True,compression=max(0.,domain['nominal_contact_center_gap_m']-gap))
    reports=[];deadline=time.monotonic()+1500
    for case in declared_cases():
        if time.monotonic()>deadline: raise TimeoutError('Declared total geometry budget exhausted')
        started=time.monotonic();report=dict(case=case,ik_accurate=False,free_geometry_passed=False,
            direct_entry_geometry_passed=False,full_curve_geometry_passed=False,
            physics_executed=False,training_dataset_eligible=False)
        try:
            desired=perturb_pose(nominal,normal,case)
            if case['kind']=='center': q=q0.copy();accepted=True
            else:
                q,accepted,_,_=solve_posture_ik(kin,base,limits,desired[:3,3],desired[:3,:3],q0,
                    maxiter=100,deadline=min(deadline,time.monotonic()+15),planning_margin_m=.002,
                    joint_margin_rad=0.,contact=False)
            attained=kin.tcp(q)
            pe=float(np.linalg.norm(attained[:3,3]-desired[:3,3]))
            ae=float(Rotation.from_matrix(attained[:3,:3]@desired[:3,:3].T).magnitude())
            report.update(q=q.tolist(),requested_pose=desired.tolist(),attained_pose=attained.tolist(),
                ik_position_error_m=pe,ik_rotation_error_rad=ae,
                maximum_joint_change_rad=float(np.max(abs(q-q0))))
            # Probe accuracy is tighter than the unchanged physical curve gate;
            # inaccurate requests stay in the denominator as explicit failures.
            if not accepted or pe>1e-7 or ae>1e-6: raise ValueError('Probe IK does not resolve the requested perturbation')
            report['ik_accurate']=True
            check=checker.check(q,clearance=.003,allow_patient_contact=False,compression=0.)
            report['free_geometry']=check
            if check['clear'] is not True: raise ValueError('Perturbed free pose geometry refused')
            report['free_geometry_passed']=True
            gate=check_retained_pose(attained,target,normal,rotation,
                position_tolerance_m=CONTACT_DEFAULTS['curve_position_tolerance_m'],
                angle_tolerance_rad=CONTACT_DEFAULTS['curve_angle_tolerance_rad'])
            report['direct_entry_gate']=gate
            if not gate['passed']: raise ValueError('Original direct-entry gate refused: '+','.join(gate['failures']))
            report['direct_entry_geometry_passed']=True
            curve=FixedNormalCurve.build(kin,base,limits,q,target,normal,rotation,domain['minimum_gap_m'],
                spacing=CONTACT_DEFAULTS['curve_spacing_m'],
                position_tolerance=CONTACT_DEFAULTS['curve_position_tolerance_m'],
                angle_tolerance=CONTACT_DEFAULTS['curve_angle_tolerance_rad'],collision_check=contact_check,
                wall_budget_s=min(120.,deadline-time.monotonic()))
            report.update(full_curve_geometry_passed=True,curve=curve.report)
        except (ValueError,RuntimeError,TimeoutError,IKBudgetExpired) as error:
            report['failure_reason']=type(error).__name__+': '+str(error)
        report['wall_time_s']=time.monotonic()-started
        write(a.output/(case['id']+'.json'),report);reports.append(report)
        print(json.dumps({k:report.get(k) for k in ('case','ik_accurate','direct_entry_geometry_passed','full_curve_geometry_passed','failure_reason')}),flush=True)
    write(a.output/'COMPLETE.json',dict(completed=True,case_count=len(reports),
        counts={key:sum(r[key] for r in reports) for key in ('ik_accurate','free_geometry_passed','direct_entry_geometry_passed','full_curve_geometry_passed')},
        cases=reports,contact_defaults=CONTACT_DEFAULTS,domain=domain,physics_executed=False,
        capture_success_rate=None,m0_passed=False,training_dataset_eligible=False,model_loaded=False,
        scope='35 prospective single-body geometry nodes, not actual attained states or a dynamic capture region'))


if __name__=='__main__': main()
