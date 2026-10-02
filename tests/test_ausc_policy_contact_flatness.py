import math
import numpy as np
import pytest
from sim.ausc_policy import contact_flatness as F
from sim.ausc_policy.retainer_contact import ROOT


def contact(sector, force=1., reverse=False):
    paths=[ROOT+f'DeviceContactRing/sector_{sector:02d}','/World/Patient/body']
    return dict(colliders=paths[::-1] if reverse else paths,
                impulse_Ns=[0.,0.,(-force if reverse else force)/240])


def row(points, axis=(0.,0.,-1.)):
    return dict(axis_world=list(axis),retainer_N=sum(np.linalg.norm(p['impulse_Ns'])*240 for p in points),
                contact_points=points)


def test_parallel_head_and_all_sectors():
    out=F.measure(row([contact(i,1/8) for i in range(32)]),[0,0,1],[0,0,1])
    assert out['target_normal_angle_deg']==0
    assert out['sectors_gt_0_05N']==32
    assert out['loaded_center_arc_deg']==348.75
    assert out['largest_sector_force_fraction']==pytest.approx(1/32)


def test_seventeen_degree_tool_is_not_flat_despite_perfect_command_tracking():
    t=math.radians(17);axis=[-math.sin(t),0,-math.cos(t)]
    out=F.measure(row([contact(0,4.)],axis),[0,0,1],[-x for x in axis])
    assert out['target_normal_angle_deg']==pytest.approx(17.)
    assert out['command_normal_angle_deg']==pytest.approx(0.,abs=1e-6)
    assert out['tangent_plane_opposite_edge_height_mm']==pytest.approx(46*math.sin(t))
    assert out['sectors_gt_0_05N']==1


def test_zero_impulse_candidates_do_not_count_as_loaded():
    out=F.measure(row([contact(0,4.)]+[contact(i,0.) for i in range(1,32)]),[0,0,1],[0,0,1])
    assert out['reported_candidate_sectors']==32
    assert out['sectors_gt_zero']==1


def test_contacts_sum_by_sector_and_collider_order():
    out=F.measure(row([contact(31,.03),contact(31,.03,True),contact(0,3.94)]),[0,0,1],[0,0,1])
    assert out['sectors_gt_0_05N']==2
    assert out['loaded_center_arc_deg']==11.25
    assert out['normal_resultant_angle_deg']==0.


def test_opposite_normal_is_not_silently_flipped():
    out=F.measure(row([contact(1,4)]),[0,0,-1],[0,0,1])
    assert out['target_normal_angle_deg']==180.


@pytest.mark.parametrize('normal',[[0,0,0],[0,0,2],[0,0,float('nan')]])
def test_invalid_normals_fail(normal):
    with pytest.raises(ValueError):F.measure(row([contact(1,4)]),normal,[0,0,1])


def test_component_force_mismatch_fails():
    r=row([contact(1,4)]);r['retainer_N']=3.
    with pytest.raises(ValueError,match='ledger'):F.measure(r,[0,0,1],[0,0,1])
