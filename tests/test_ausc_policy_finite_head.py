import numpy as np
import pytest
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
    fit.check=lambda *a: dict(passed=True)
    normal=np.array([.3,.4,np.sqrt(.75)])
    x=np.cross(normal,[0,0,1]);x/=np.linalg.norm(x);y=np.cross(normal,x)
    result=fit.candidates(np.zeros(3),normal)
    assert {35.,40.}.issubset({c['normal_deviation_from_surface_deg'] for c in result['candidates']})
    for c in result['candidates']:
        n=np.array(c['normal_world']);uv=np.array([n@x,n@y])/(n@normal)
        assert np.max(abs(uv))<=.6+1e-12
