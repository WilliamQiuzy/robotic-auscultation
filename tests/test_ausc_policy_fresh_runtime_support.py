import numpy as np
import pytest
from types import SimpleNamespace


def test_monitor_reuses_verified_installation_but_still_checks_pose(monkeypatch):
    from sim.curobo_expert_v1 import static_world as s
    contract=dict(body_reference_poses={'fixed':np.eye(4).tolist()},finger_specs={},
        translation_tolerance_m=.001,rotation_tolerance_rad=.001,
        finger_joint_numerical_tolerance_m=.0001)
    bundle=dict(contract=contract,contract_sha256=s.contract_sha(contract),geometry={})
    installed=dict(contract_sha256=bundle['contract_sha256'],geometry_sha256={})
    checker=SimpleNamespace(static_bundle=bundle,static_installation=installed,
                            _static_containment_installed=True)
    monkeypatch.setattr(s,'install_fcl_static_world',lambda *a,**k:pytest.fail('double install'))
    position=np.zeros(3)
    body=SimpleNamespace(get_world_pose=lambda:(position,np.array([1.,0.,0.,0.])))
    monitor=s.StaticWorldMonitor(SimpleNamespace(bodies=[('fixed',body)],rows=[]),checker,bundle)
    assert monitor.last_report['clear']
    position[0]=.01
    with pytest.raises(ValueError,match='static world'):monitor.check()
    checker.static_installation={**installed,'contract_sha256':'wrong'}
    with pytest.raises(ValueError,match='installation'):s.StaticWorldMonitor(SimpleNamespace(bodies=[]),checker,bundle)


def test_contact_domain_uses_current_moving_mass():
    from sim.auscultation_skill_v1.curve import force_derived_domain
    normal=np.array([0.,-.8,.6])
    report=force_derived_domain(-.002,.0025,normal,4.,.0005,moving_mass_kg=.09)
    assert report['static_hall_model']['moving_mass_kg']==.09
    assert report['static_target_compression_m']==pytest.approx((4-1.5-.09*9.81*.6)/1000.)
    with pytest.raises(ValueError):force_derived_domain(0.,.0025,normal,4.,.0005,moving_mass_kg=float('nan'))
