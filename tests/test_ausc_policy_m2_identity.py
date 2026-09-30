import pytest
import numpy as np


def test_identity_split_is_order_independent_and_disjoint():
    from sim.ausc_policy.m2_identity import identity_split
    ids = ['p%06d' % i for i in range(340)]
    a = identity_split(ids, 20260930)
    assert a == identity_split(ids[::-1] + ids[:10], 20260930)
    assert {k:len(v) for k,v in a.items()} == {'train':272,'validation':34,'test':34}
    assert set(a['train']).isdisjoint(a['test'])
    assert set(a['validation']).isdisjoint(a['test'])


def test_bulk_requires_current_actual_randomized_expert_qualification():
    from sim.ausc_policy.m2_identity import require_bulk_qualification
    from sim.ausc_policy.geometry_epoch import GEOMETRY_ID
    good = dict(geometry_id=GEOMETRY_ID,geometry_source_sha256='a'*64,freshly_generated=True,
        randomized_cases=100,successes=95,physics_executed=True,identity_split_sha256='b'*64)
    require_bulk_qualification(good,'b'*64)
    for changes in ({'physics_executed':False},{'successes':94},{'randomized_cases':10},
                    {'identity_split_sha256':'c'*64},{'geometry_id':'old'}):
        with pytest.raises(ValueError):require_bulk_qualification(dict(good,**changes),'b'*64)


def test_inconsistent_identity_shape_is_quarantined_not_averaged():
    from sim.ausc_policy.m2_identity import canonical_shape
    np.testing.assert_allclose(canonical_shape([np.zeros(10),np.zeros(10)]),0.)
    with pytest.raises(ValueError):canonical_shape([np.zeros(10),np.ones(10)])
