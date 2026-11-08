"""CPU representation regressions, never native or physical qualification."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sim.ausc_policy import geometry_epoch, m2_act_onpolicy_collection as collection
from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics


def native_like_pose(kin,q,base,compression):
    """Independent float32 chain/storage, then the real env.state TCP algebra.

    This emulates transform precision only. It is not a PhysX solver and cannot
    establish a solver accuracy bound or qualify a captured episode.
    """
    wrist=np.eye(4,dtype=np.float32)
    for joint,angle in zip(kin._joints,np.asarray(q,dtype=np.float32)):
        motion=np.eye(4,dtype=np.float32)
        motion[:3,:3]=Rotation.from_rotvec(joint.axis*float(angle)).as_matrix()
        wrist=wrist@joint.origin.astype(np.float32)@motion
    world=base.astype(np.float32)@wrist
    # Native wrist readback stores position/quaternion separately. SciPy
    # normalizes the stored quaternion, exactly as IsaacContactEnv.state does.
    quat=Rotation.from_matrix(world[:3,:3]).as_quat().astype(np.float32)
    rotation=Rotation.from_quat(quat).as_matrix()
    tcp=world[:3,3].astype(float)+rotation[:,2]*(kin.contact_frame()[2,3]-compression)
    pose=np.eye(4);pose[:3,:3]=base[:3,:3].T@rotation
    pose[:3,3]=base[:3,:3].T@(tcp-base[:3,3])
    return pose


def inputs(compression=-1.297662244e-8):
    kin=OpenArmKinematics(contact_frame=geometry_epoch.contact_frame)
    q=np.array([.27,-.42,.35,.7,-.24,.18,-.33],dtype=np.float32).astype(float)
    base=np.eye(4);base[:3,:3]=Rotation.from_euler('x',-np.pi/2-3.57456062284e-6).as_matrix()
    base[:3,3]=[0.,.3309999994933605,1.376228643993393]
    compression=float(np.float32(compression))
    pose=native_like_pose(kin,q,base,compression)
    return kin,q,compression,pose,base


@pytest.mark.parametrize('compression',[-1.297662244e-8,0.,.00014])
def test_native_transform_storage_and_signed_compression_are_not_analytic_equality(compression):
    args=inputs(compression);kin,q,compression,pose,base=args
    original=[a.copy() if isinstance(a,np.ndarray) else a for a in args[1:]]
    assert not np.allclose(kin.tcp(q),pose,atol=1e-10,rtol=0)
    proof=collection.verify_native_tcp_consistency(*args)
    assert proof['position_error_m']<proof['position_tolerance_m']<20e-6
    assert proof['rotation_error_rad']<proof['rotation_tolerance_rad']<8e-6
    assert proof['units']==dict(position='m',rotation='rad')
    assert proof['native_roundoff_scope']=='representation consistency only; not PhysX accuracy or physical acceptance'
    for before,after in zip(original,args[1:]):np.testing.assert_array_equal(before,after)
    # No relaxation of geometry's command/IK API to accept signed sensor residuals.
    if compression<0:
        with pytest.raises(ValueError,match='Invalid compression'):geometry_epoch.contact_frame(compression)


@pytest.mark.parametrize('fault',['old_tcp','wrong_frame','wrong_q','rotation','compression','translation_outside',
    'rotation_outside','nonrigid','reflection','homogeneous','pose_nan','pose_inf','q_nan','compression_nan',
    'base_nan','base_nonrigid','base_far','compression_far','q_limit','pose_shape','pose_text','compression_bool'])
def test_native_pose_contract_rejects_geometry_frame_and_invalid_measurements(fault):
    kin,q,compression,pose,base=inputs()
    if fault=='old_tcp':pose[:3,3]-=pose[:3,2]*.0058
    elif fault=='wrong_frame':pose=base@pose
    elif fault=='wrong_q':q[0]+=.001
    elif fault=='rotation':pose[:3,:3]=pose[:3,:3]@Rotation.from_rotvec([0.,0.,.001]).as_matrix()
    elif fault=='compression':compression=.00014
    elif fault in ('translation_outside','rotation_outside'):
        proof=collection.verify_native_tcp_consistency(kin,q,compression,pose,base)
        pose=kin.tcp(q);pose[:3,3]-=pose[:3,2]*compression
        if fault=='translation_outside':pose[0,3]+=proof['position_tolerance_m']*1.01
        else:pose[:3,:3]=pose[:3,:3]@Rotation.from_rotvec([proof['rotation_tolerance_rad']*1.01,0.,0.]).as_matrix()
    elif fault=='nonrigid':pose[0,0]+=.001
    elif fault=='reflection':pose[:3,0]*=-1
    elif fault=='homogeneous':pose[3,0]=.001
    elif fault=='pose_nan':pose[0,3]=np.nan
    elif fault=='pose_inf':pose[0,3]=np.inf
    elif fault=='q_nan':q[0]=np.nan
    elif fault=='compression_nan':compression=np.nan
    elif fault=='base_nan':base[0,3]=np.nan
    elif fault=='base_nonrigid':base[0,0]+=.001
    elif fault=='base_far':base[:3,3]=[0.,0.,2.000001]
    elif fault=='compression_far':compression=-1.
    elif fault=='q_limit':q[0]=kin.upper_limits[0]+1e-6
    elif fault=='pose_shape':pose=pose[:3]
    elif fault=='pose_text':pose=pose.astype(str)
    else:compression=True
    with pytest.raises(ValueError):collection.verify_native_tcp_consistency(kin,q,compression,pose,base)


def test_separate_meter_and_radian_bounds_have_explicit_float32_scale():
    kin,q,compression,pose,base=inputs()
    proof=collection.verify_native_tcp_consistency(kin,q,compression,pose,base)
    u=2.**-24;gamma=64*u/(1-64*u)
    chain_length=.6235988938841939
    assert proof['schema']=='m2_act_onpolicy_native_tcp_consistency_v1'
    assert proof['rotation_tolerance_rad']==2*gamma
    assert proof['position_tolerance_m']==pytest.approx(gamma*(chain_length+abs(.132-compression)+2*np.linalg.norm(base[:3,3])))
    for kind in ('position','rotation'):
        candidate=kin.tcp(q);candidate[:3,3]-=candidate[:3,2]*compression
        if kind=='position':candidate[0,3]+=proof['position_tolerance_m']*.99
        else:candidate[:3,:3]=candidate[:3,:3]@Rotation.from_rotvec([proof['rotation_tolerance_rad']*.99,0.,0.]).as_matrix()
        collection.verify_native_tcp_consistency(kin,q,compression,candidate,base)


def test_submicrometre_and_microradian_native_like_readback_residuals():
    kin,q,compression,_,base=inputs()
    pose=kin.tcp(q);pose[:3,3]-=pose[:3,2]*compression
    # Rounded magnitudes from Root's independent native diagnostics, not raw
    # episode data and not a fit used to choose either numerical ceiling.
    pose[0,3]+=4.53e-7
    pose[:3,:3]=pose[:3,:3]@Rotation.from_rotvec([0.,1.31e-6,0.]).as_matrix()
    proof=collection.verify_native_tcp_consistency(kin,q,compression,pose,base)
    assert proof['position_error_m']==pytest.approx(4.53e-7)
    assert proof['rotation_error_rad']==pytest.approx(1.31e-6)
