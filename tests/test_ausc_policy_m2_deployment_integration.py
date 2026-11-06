"""Stubbed CPU runtime ownership tests; no trained model or physics claims."""
from copy import deepcopy
import hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from sim.ausc_policy import fresh_expert as fresh,contact_collision,m2_act_controller as adapter
from sim.ausc_policy.geometry_epoch import GEOMETRY_ID,GEOMETRY_SOURCE_SHA256
from test_ausc_policy_m2_estimate_provider import sources
from test_ausc_policy_expert_capture import binding


def setup_runtime(tmp_path,monkeypatch,fault='none'):
    # Finalization imports this real class hierarchy; load it before replacing
    # FreshExpert with the sentinel that forbids the expert execution branch.
    from sim.ausc_policy import m2_given_contact_controller
    provider,meta,patient,*_=sources(tmp_path)
    checkpoint=tmp_path/'checkpoint.pt';checkpoint.write_bytes(b'CPU stub; never loaded or qualified')
    deployment=dict(schema=adapter.DEPLOYMENT_SCHEMA,checkpoint=str(checkpoint),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),lerobot_root='/unused-test-only',metadata=meta,episode_id='new-episode')
    config=dict(geometry_id=GEOMETRY_ID,geometry_source_sha256=GEOMETRY_SOURCE_SHA256,
        fresh_expert=dict(passive_preload_N=1.5),m2_act_deployment=deployment,m2_target_estimate=provider)
    runtime_meta=dict(geometry_id=GEOMETRY_ID,geometry_source_sha256=GEOMETRY_SOURCE_SHA256,freshly_generated=True,
        geometry_artifacts={},world_from_left_base=np.eye(4).tolist(),plans=[],patient_provenance=patient)
    events=[];created=[]
    class Env:
        def __init__(self,output,**kwargs):
            events.append('env');self.rows=[];self.out=Path(output);self.out.mkdir();self.base=np.eye(4)
            self.world=SimpleNamespace(stop=lambda:events.append('stop'));self.value=dict(q=np.zeros(7),time_s=0.)
        def state(self):return deepcopy(self.value)
        def save(self,name):
            events.append('trace')
            if fault=='trace':raise OSError('trace fixture failure')
            (self.out/(name+'_trace.jsonl')).write_text(''.join(json.dumps(row)+'\n' for row in self.rows))
    class Recorder:
        def __init__(self,*args):events.append('video_start')
        def close(self):events.append('video_close');return dict(file='CPU-stub-video')
    class Controller:
        def __init__(self,env,estimate_provider):
            self.env=env;self.control_rows=[];self.episode_id='new-episode';self.target=SimpleNamespace(name='L1')
            self.provider=estimate_provider;self.policy_receipt=dict(checkpoint_sha256=deployment['checkpoint_sha256'])
            self.l2_handoffs=[];self.l2_queries=[];self.l2_actions=[]
        def run(self,points):
            self.env.value['time_s']=1/240
            self.env.rows.append(dict(q_command=[0.]*7,target='L1',learned_policy=False,expert_demonstration=True))
            self.control_rows.append(dict(learned_policy=False,expert_demonstration=True))
            acquired=self.provider(self,self.env.state())
            self.l2_queries.append(dict(time_s=acquired['observed_at_s']))
            if fault=='run':raise RuntimeError('controller fixture failure')
            if fault=='fingerprint':checkpoint.write_bytes(b'changed')
            self.l2_handoffs.append(dict(target='L1',l2_handoff_success=True,l3_completed=True))
            return dict(success=True,learned_policy=False,points=[dict(target='L1',success=True,expert_demonstration=True)])
    def factory(env,kin,checker,*,deployment,estimate_provider,**kwargs):
        events.append('factory')
        assert not hasattr(env,'raw_capture')
        assert deployment['metadata']==estimate_provider.metadata
        assert deployment['metadata']['episode_id']==deployment['episode_id']=='new-episode'
        if fault=='factory':raise RuntimeError('checkpoint loading fixture failure')
        obj=Controller(env,estimate_provider);created.append(obj);return obj
    monkeypatch.setattr(sys,'platform','linux');monkeypatch.setenv('AUSC_OWNED_RUN','1')
    monkeypatch.setattr(fresh,'read_plans',lambda *a:([(SimpleNamespace(name='L1'),{})],deepcopy(runtime_meta)))
    monkeypatch.setattr(contact_collision,'ContactCollisionChecker',lambda *a,**kw:SimpleNamespace(profile=kw['profile'],static_bundle={}))
    monkeypatch.setattr(fresh,'FreshContactEnv',Env);monkeypatch.setattr(fresh,'_VideoRecorder',Recorder)
    monkeypatch.setattr(adapter,'make_controller',factory)
    monkeypatch.setattr(fresh,'FreshExpert',lambda *a,**kw:pytest.fail('Expert branch used for learned rollout'))
    monkeypatch.setattr(fresh,'create_expert_capture',lambda *a,**kw:pytest.fail('Expert capture attached to learned rollout'))
    def audit_physics(outcome,rows,*,retainer_witness_contract):
        assert retainer_witness_contract is None
        return dict(outcome,physical_validation=bool(outcome.get('success')))
    monkeypatch.setattr(fresh,'audit_physics',audit_physics)
    monkeypatch.setattr(fresh,'require_geometry_artifacts',lambda *a:{})
    return config,events,created


@pytest.mark.parametrize('fault',['none','factory','run','trace','fingerprint'])
def test_optional_branch_preserves_learned_labels_sources_and_cleanup(tmp_path,monkeypatch,fault):
    config,events,created=setup_runtime(tmp_path,monkeypatch,fault)
    result=fresh.run_probe(None,None,tmp_path,config,dict(mass_kg=.067))
    assert result['learned_policy'] is True and result['expert_demonstration'] is False
    assert result['training_eligible'] is False and result['training_accepted'] is False
    assert result['success'] is (fault=='none'), result.get('errors')
    assert events[-3:]==['trace','video_close','stop']
    assert result['video']==dict(file='CPU-stub-video') and not result['raw_capture_complete']
    assert result['l2_handoff_attempt_count']==1
    hashes=result['runtime_sources_sha256']
    for name in ('m2_act_controller.py','m2_estimate_provider.py','m2_act.py','m2_contract.py','checkpoint.pt'):
        assert any(Path(path).name==name for path in hashes)
    snapshot=tmp_path/'fresh_expert_inputs'
    assert (snapshot/'m2_target_noise_receipt.json').read_bytes()==Path(config['m2_target_estimate']['noise_receipt']['file']).read_bytes()
    assert json.loads((snapshot/'manifest.json').read_text())['learned_policy'] is True
    saved=json.loads((tmp_path/'fresh_expert_result.json').read_text());assert saved['learned_policy'] is True
    if created:
        assert all(row['learned_policy'] and not row['expert_demonstration'] for row in created[0].env.rows+created[0].control_rows)
        assert result['l2_queries'][0]['time_s']==pytest.approx(1/240)
    if fault=='fingerprint':assert any(e['stage']=='source_fingerprint' for e in result['errors'])


@pytest.mark.parametrize('fault',['capture','missing_estimate','checkpoint','invalid_deployment','noise_hash','identity'])
def test_preflight_failure_is_labeled_learned_and_starts_no_environment(tmp_path,monkeypatch,fault):
    config,events,_=setup_runtime(tmp_path,monkeypatch)
    if fault=='capture':config['fresh_expert']['raw_multimodal_capture']=binding()
    if fault=='missing_estimate':config.pop('m2_target_estimate')
    if fault=='checkpoint':config['m2_act_deployment']['checkpoint_sha256']='0'*64
    if fault=='invalid_deployment':config['m2_act_deployment']=None
    if fault=='noise_hash':config['m2_target_estimate']['noise_receipt']['sha256']='0'*64
    if fault=='identity':config['m2_act_deployment']['metadata']['subject_id']='p000452'
    result=fresh.run_probe(None,None,tmp_path,config,dict(mass_kg=.067))
    assert not result['success'] and result['learned_policy'] and not result['expert_demonstration']
    assert result['l2_handoff_attempt_count']==1 and result['l2_handoff_success_count']==0
    assert events==[] and result['physics_steps']==0
    assert json.loads((tmp_path/'fresh_expert_result.json').read_text())['learned_policy'] is True


def test_unexpected_post_run_error_keeps_actual_step_count_and_video(tmp_path,monkeypatch):
    config,events,_=setup_runtime(tmp_path,monkeypatch)
    monkeypatch.setattr(fresh,'audit_physics',lambda *args:{})
    result=fresh.run_probe(None,None,tmp_path,config,dict(mass_kg=.067))
    assert not result['success'] and result['learned_policy'] and result['physics_steps']==1
    assert result['video']==dict(file='CPU-stub-video') and events[-1]=='stop'
    assert result['runtime_sources_sha256']
