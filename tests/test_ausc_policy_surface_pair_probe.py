import hashlib
import importlib
import json

import numpy as np
import pytest


def test_cell_binding_rejects_changed_asset(tmp_path):
    module = importlib.import_module('sim.ausc_policy.surface_pair_probe')
    file = tmp_path/'collision_geometry.json'
    file.write_text('{}')
    cell = {'artifacts': {'collision_geometry.json': {'path': str(file), 'sha256': '0'*64}}}
    with pytest.raises(ValueError, match='hash'):
        module.bound_cell_sources(cell)


def test_bound_sources_read_exact_bytes(tmp_path):
    module = importlib.import_module('sim.ausc_policy.surface_pair_probe')
    paths = {}
    for name in ('collision_geometry.json', 'collision_geometry.npz', 'v6a_anchors.json'):
        file = tmp_path/name
        file.write_bytes(name.encode())
        paths[name] = {'path': str(file), 'sha256': hashlib.sha256(file.read_bytes()).hexdigest()}
    result = module.bound_cell_sources({'artifacts': paths})
    assert result == {name: name.encode() for name in paths}


def test_unknown_or_incomplete_case_selection_cannot_silently_drop_denominator():
    module = importlib.import_module('sim.ausc_policy.surface_pair_probe')
    with pytest.raises(ValueError, match='case'):
        module.select_cells({'cells': [{'cell_id': 'yaw_p000474_35'}]},
                            ['yaw_p000474_35', 'yaw_p000476_35'])


def test_proxy_anchor_moves_only_along_original_normal():
    module = importlib.import_module('sim.ausc_policy.surface_pair_probe')
    mesh = np.array([[[-1,-1,.0005],[1,-1,.0005],[1,1,.0005]],
                     [[-1,-1,.0005],[1,1,.0005],[-1,1,.0005]]])
    mapped = module.map_anchor(mesh, [0,0,0], [0,0,1])
    assert mapped['position_world_m'] == pytest.approx([0,0,.0005])
    assert mapped['tangential_displacement_m'] == pytest.approx(0.)


def test_no_surface_hit_does_not_create_anchor():
    module = importlib.import_module('sim.ausc_policy.surface_pair_probe')
    with pytest.raises(ValueError, match='missing'):
        module.map_anchor(np.empty((0,3,3)), [0,0,0], [0,0,1])


def test_missing_central_hit_retained_without_aborting_following_patch():
    module = importlib.import_module('sim.ausc_policy.surface_pair_probe')
    mesh = np.array([[[-1,-1,0],[1,-1,0],[1,1,0]], [[-1,-1,0],[1,1,0],[-1,1,0]]])
    results = [module.evaluate_patch(mesh, p, [0,0,1]) for p in ([2,2,0], [0,0,0])]
    assert len(results) == 2
    assert results[0]['status'] == 'unresolved_missing_central_hit'
    assert results[0]['anchor_map'] is None
    assert results[0]['sampled_geometry']['missing_samples'] == 97
    assert results[1]['status'] == 'sampled_patch_complete'
