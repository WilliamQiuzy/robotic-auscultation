import json
from copy import deepcopy
import pytest
from pathlib import Path
from sim.auscultation_pi05_v6 import observed_supervisor
from sim.auscultation_pi05_v5.tests.test_dynamics_contract import snapshot
from sim.auscultation_pi05_v6.dynamics_contract import build_contract,sha256_file,supervisor_contract
from sim.auscultation_pi05_v6.dynamics_binding import bind_dataset,verify_manifest_dynamics


def prepare(tmp_path,snapshot,*,mixed=False,missing_outcome_receipt=False,wrong_timing=False,ideal_supervisor=False):
    parent=tmp_path/'parent';parent.mkdir()
    records=[];sources=[]
    for identity in ('a','b'):
        physics=deepcopy(snapshot)
        if mixed and identity=='b':physics['prims']['/joint2']['attributes']['physxJoint:armature']=.01
        recipe=dict(baseline_scene_sha256='a'*64,effective_scene_sha256='b'*64,
            physics_profile={'armature':.01},runtime_settings={'dt':1/240})
        doc=build_contract(physics,[f'/joint{i}' for i in range(7)],
            runtime_versions=dict(isaac_sim='5.1',isaac_sim_build='test-build',usd='0.24',physx_schema='test-schema'),
            recipe=recipe,evidence={'source_usd_sha256':'b'*64},scope='simulator_recipe')
        contract=tmp_path/(identity+'.dynamics.json');contract.write_text(json.dumps(doc))
        outcome=parent/(identity+'.outcome.json')
        outcome.write_text(json.dumps({} if missing_outcome_receipt else dict(dynamics_recipe_sha256=doc['recipe_sha256'],
            phase_config=supervisor_contract(),speed_scale=1. if wrong_timing else .7,joint_speed_limit_rad_s=.175,
            observed_supervisor_version='old_ideal_force' if ideal_supervisor else observed_supervisor.VERSION,
            observed_supervisor_sha256=sha256_file(Path(observed_supervisor.__file__)))) )
        qualification=parent/(identity+'.qualification.json')
        qualification.write_text(json.dumps({'evidence':{'artifact_sha256':{'outcome.json':sha256_file(outcome)}}}))
        records.append(dict(episode_id=identity,outcome_path=outcome.name,outcome_sha256=sha256_file(outcome),
            qualification_path=qualification.name,qualification_sha256=sha256_file(qualification)))
        sources.append(dict(episode_id=identity,dynamics_contract_path=str(contract)))
    manifest=parent/'manifest.json';manifest.write_text(json.dumps(dict(episodes=records)))
    return manifest,sources


def test_binds_qualified_outcome_and_preserves_parent(tmp_path,snapshot):
    source,records=prepare(tmp_path,snapshot);before=source.read_bytes();out=tmp_path/'bound'
    manifest=bind_dataset(source,records,out)
    verified=verify_manifest_dynamics(out/'manifest.json',manifest)
    assert verified['recipe_sha256']==manifest['dynamics_recipe_sha256']
    assert source.read_bytes()==before
    path=out/manifest['episodes'][0]['dynamics_contract_path'];path.write_text('{}')
    with pytest.raises(ValueError,match='hash'):verify_manifest_dynamics(out/'manifest.json',manifest)


@pytest.mark.parametrize('option',['mixed','missing_outcome_receipt','wrong_timing','ideal_supervisor'])
def test_historical_or_mixed_recipe_cannot_be_retrofitted(tmp_path,snapshot,option):
    source,records=prepare(tmp_path,snapshot,**{option:True})
    with pytest.raises(ValueError):bind_dataset(source,records,tmp_path/'rejected')
    assert not (tmp_path/'rejected/manifest.json').exists()
