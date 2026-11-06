import pytest

from sim.ausc_policy.geometry_qualification import preparation_backend


def test_cpu_authoring_requires_explicit_non_simulation_boundary():
    config = dict(prepare_only=True, patient_asset={'authenticated_later': True})
    assert preparation_backend(config, True, 'linux', {'CUDA_VISIBLE_DEVICES': ''}) == 'cpu_usd_authoring_no_simulation_app'
    for changes in ({'prepare_only': False}, {'prepare_only': 1}, {'patient_asset': None}):
        with pytest.raises(ValueError):
            preparation_backend(dict(config, **changes), True, 'linux', {'CUDA_VISIBLE_DEVICES': ''})
    for env in ({}, {'CUDA_VISIBLE_DEVICES': '0'}, {'CUDA_VISIBLE_DEVICES': '', 'AUSC_OWNED_RUN': '1'}):
        if env.get('CUDA_VISIBLE_DEVICES') == '':
            assert preparation_backend(config, True, 'linux', env).startswith('cpu_')
        else:
            with pytest.raises(RuntimeError):
                preparation_backend(config, True, 'linux', env)
    with pytest.raises(RuntimeError):
        preparation_backend(config, True, 'darwin', {'CUDA_VISIBLE_DEVICES': ''})


def test_native_path_still_requires_owned_linux():
    with pytest.raises(RuntimeError):
        preparation_backend({}, False, 'linux', {})
    with pytest.raises(RuntimeError):
        preparation_backend({}, False, 'darwin', {'AUSC_OWNED_RUN': '1'})
    assert preparation_backend({}, False, 'linux', {'AUSC_OWNED_RUN': '1'}) == 'native_isaac_simulation_app'
