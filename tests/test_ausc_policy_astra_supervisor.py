"""CPU bridge tests; these never qualify a model or native rollout."""
from copy import deepcopy
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest


def observation(boundary='HOME'):
    return dict(schema='ausc_astra_native_observation_v1', simulation_only=True,
        observation_id='episode:0', episode_id='episode', physics_row=0, time_s=0.,
        boundary=boundary, phase='APPROACH', q_rad=[0.]*7, dq_rad_s=[0.]*7,
        hall_force_N=0., hall_source='native_axial_proxy_filtered', compression_m=0.,
        target_id='L1', target_source='given_contact_pose', reference_sha256='a'*64,
        reference_pose_base=np.eye(4).tolist(), rgb_sha256=None,
        local_executor='GivenContactExpert', local_motor_assistance=True)


def decision(obs, skill=None):
    return dict(observation_id=obs['observation_id'], skill=skill or
        {'HOME':'approach_target','PARKED':'continue_local_control','RELEASED':'retract'}[obs['boundary']],
        target_id=obs['target_id'], reason='Use the bounded local skill.')


def test_decision_cannot_change_target_observation_skill_or_thresholds():
    from sim.ausc_policy import astra_supervisor as m
    obs=observation()
    assert m.validate_decision(decision(obs),obs)==decision(obs)
    for change in (dict(target_id='R1'),dict(observation_id='stale'),dict(skill='retract'),
                   dict(force_limit_N=9),dict(reason='')):
        with pytest.raises(ValueError):m.validate_decision(dict(decision(obs),**change),obs)


@pytest.mark.parametrize('boundary,allowed', [('HOME','approach_target'),
    ('PARKED','continue_local_control'),('RELEASED','retract')])
def test_only_safe_boundary_skills_and_abort_are_admitted(boundary,allowed):
    from sim.ausc_policy import astra_supervisor as m
    obs=observation(boundary)
    assert m.validate_decision(decision(obs,allowed),obs)['skill']==allowed
    assert m.validate_decision(dict(decision(obs,'abort'),target_id=None),obs)['skill']=='abort'
    for skill in set(m.SKILLS)-{allowed,'abort'}:
        with pytest.raises(ValueError):m.validate_decision(decision(obs,skill),obs)


def test_http_client_is_loopback_only_and_requires_token(monkeypatch):
    from sim.ausc_policy import astra_supervisor as m
    for url in ('https://example.com/decision','http://0.0.0.0:8123/decision'):
        with pytest.raises(ValueError):m.HTTPClient(url,'token')
    with pytest.raises(ValueError):m.HTTPClient('http://127.0.0.1:8123/decision','')


class LocalSkills:
    def _move_to_prepose(self,row):self.executed.append('approach');return 'transfer'
    def _contact_cycle(self):self.executed.append('contact');return None
    def _return_home(self,transfer):self.executed.append(('return',transfer));return None
    def _tick(self,*args,**kwargs):
        self.env.rows.append(dict(command_source_id='given_contact_original_analytic_l3_v1'))
        return dict()


def bridge(tmp_path,skill=None,mutate=False):
    from sim.ausc_policy.astra_native import AstraSupervisorMixin
    class Controller(AstraSupervisorMixin,LocalSkills):pass
    c=Controller.__new__(Controller)
    c.executed=[];c.env=SimpleNamespace(rows=[])
    state=dict(q=np.zeros(7),dq=np.zeros(7),time_s=0.)
    c.env.state=lambda:deepcopy(state)
    def client(obs,rgb):
        if mutate:state['time_s']+=1/240
        return dict(model='gpt-6-astra',decision=decision(obs,skill),backend='cpu_test',
                    inference_wall_s=.01)
    c._init_astra(client=client,observation_provider=lambda c,b:(observation(b),None),output=tmp_path)
    return c


def test_real_skill_dispatch_and_ack_are_distinct_from_model_proposal(tmp_path):
    c=bridge(tmp_path)
    assert c._move_to_prepose({})=='transfer'
    c._contact_cycle();c._return_home('transfer')
    assert c.executed==['approach','contact',('return','transfer')]
    assert len(c.astra_decisions)==3
    assert all(x['execution']['completed'] for x in c.astra_decisions)
    assert all(x['execution']['physical_success'] is None for x in c.astra_decisions)
    assert len(list(tmp_path.glob('*/proposal.json')))==3


def test_abort_and_invalid_response_execute_no_local_skill(tmp_path):
    c=bridge(tmp_path,skill='abort')
    with pytest.raises(RuntimeError,match='abort'):c._move_to_prepose({})
    assert not c.executed
    assert c.astra_decisions[0]['execution']['status']=='aborted'
    c=bridge(tmp_path/'invalid',skill='retract')
    with pytest.raises(ValueError):c._move_to_prepose({})
    assert not c.executed and c.astra_decisions[0]['execution']['status']=='rejected'


def test_inference_must_not_advance_native_state(tmp_path):
    c=bridge(tmp_path,mutate=True)
    with pytest.raises(ValueError,match='Physics'):c._move_to_prepose({})
    assert not c.executed


def test_native_failure_ack_is_retained_and_not_retried(tmp_path):
    c=bridge(tmp_path)
    def fail(*args):raise RuntimeError('native mesh gate')
    LocalSkills._move_to_prepose,old=fail,LocalSkills._move_to_prepose
    try:
        with pytest.raises(RuntimeError,match='native mesh gate'):c._move_to_prepose({})
    finally:LocalSkills._move_to_prepose=old
    assert c.astra_decisions[0]['execution']['status']=='failed'
    assert c.astra_decisions[0]['execution']['completed'] is False
    assert len(c.astra_decisions)==1


def test_motor_source_is_retained_and_supervisor_is_separately_attributed(tmp_path):
    c=bridge(tmp_path);c._move_to_prepose({});c._tick(np.zeros(7),'contact')
    row=c.env.rows[-1]
    assert row['command_source_id']=='given_contact_original_analytic_l3_v1'
    assert row['supervisor_model']=='gpt-6-astra'
    assert row['expert_demonstration'] is False and row['local_motor_assistance'] is True
    assert row['astra_decision_id']==0


def test_cli_client_pins_model_and_deletes_ephemeral_rgb(tmp_path,monkeypatch):
    from sim.ausc_policy import astra_service as m
    seen=[]
    def run(argv,**kwargs):
        seen.append(argv);path=Path(argv[argv.index('-i')+1]);assert path.read_bytes()==b'png'
        Path(argv[argv.index('-o')+1]).write_text(json.dumps(decision(observation())))
        return SimpleNamespace(returncode=0,stdout='{"type":"turn.completed","usage":{"input_tokens":1}}\n',stderr='')
    monkeypatch.setattr(m.subprocess,'run',run)
    result=m.CodexClient('/fake/codex')(observation(),b'png')
    assert result['model']=='gpt-6-astra'
    argv=seen[0]
    assert argv[argv.index('--model')+1]=='gpt-6-astra'
    assert '--ephemeral' in argv and '--ignore-user-config' in argv
    assert argv[argv.index('--sandbox')+1]=='read-only'
    assert not Path(argv[argv.index('-i')+1]).exists()


def test_cli_failure_has_no_model_or_analytic_fallback(monkeypatch):
    from sim.ausc_policy import astra_service as m
    calls=[]
    def run(*args,**kwargs):calls.append(args);raise m.subprocess.TimeoutExpired('codex',1)
    monkeypatch.setattr(m.subprocess,'run',run)
    with pytest.raises(m.subprocess.TimeoutExpired):m.CodexClient('/fake/codex')(observation(),None)
    assert len(calls)==1


def config():
    from sim.ausc_policy.astra_native import SCHEMA
    from sim.ausc_policy.m2_given_contact_handoff import POLICY_ID
    return dict(astra_supervisor=dict(schema=SCHEMA,model='gpt-6-astra',
        endpoint='http://127.0.0.1:8123/decision',token_env='AUSC_ASTRA_RUN_TOKEN',timeout_s=150.),
        fresh_expert={},m2_given_contact_control=dict(handoff_policy_id=POLICY_ID,
            metadata=dict(handoff_policy_id=POLICY_ID)))


def test_native_opt_in_forbids_lost_parking_capture_or_act_relabel(monkeypatch):
    from sim.ausc_policy import astra_native as m
    monkeypatch.setenv('AUSC_ASTRA_RUN_TOKEN','x'*32)
    good=config();assert m.settings(good)['model']=='gpt-6-astra'
    for bad in (dict(good,m2_act_deployment={}),
                dict(good,fresh_expert=dict(raw_multimodal_capture={'enabled':True})),
                dict(good,m2_given_contact_control=dict(metadata={})),
                dict(good,astra_supervisor=dict(good['astra_supervisor'],model='other'))):
        with pytest.raises(ValueError):m.settings(bad)


def test_run_probe_routes_astra_before_any_legacy_policy_adapter(monkeypatch,tmp_path):
    from sim.ausc_policy import fresh_expert,astra_native
    seen=[]
    monkeypatch.setattr(astra_native,'run_native_probe',lambda *args:seen.append(args) or {'astra':True})
    result=fresh_expert.run_probe(None,None,tmp_path,dict(astra_supervisor={}),{})
    assert result=={'astra':True} and len(seen)==1


def test_failure_before_controller_creation_retains_astra_identity(monkeypatch,tmp_path):
    from sim.ausc_policy import fresh_expert,astra_native
    monkeypatch.setenv('AUSC_ASTRA_RUN_TOKEN','x'*32)
    def fail(*args,**kwargs):raise ValueError('no verified physical assets')
    monkeypatch.setattr(fresh_expert,'_run_probe',fail)
    result=astra_native.run_native_probe(None,None,tmp_path,config(),{})
    assert result['success'] is False and result['supervisor_model']=='gpt-6-astra'
    assert result['closed_loop_evaluated'] is False and result['local_motor_assistance'] is True
    assert result['astra_decisions']==[]
    assert result['supervisor_configured'] is True and result['supervisor_inference'] is False


def test_loopback_auth_and_size_rejections_never_invoke_inference():
    from sim.ausc_policy import astra_service as m
    from sim.ausc_policy.astra_supervisor import MAX_REQUEST_BYTES
    calls=[];token='test-'+32*'x'
    server=m.make_server(0,token,lambda *args:calls.append(args))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    url=f'http://127.0.0.1:{server.server_port}/decision'
    try:
        for headers,status in [({},401),({'Authorization':'Bearer '+token,
                                         'Content-Length':str(MAX_REQUEST_BYTES+1)},400)]:
            with pytest.raises(HTTPError) as error:
                urlopen(Request(url,b'{}',headers=headers,method='POST'),timeout=2)
            assert error.value.code==status
        assert calls==[]
    finally:server.shutdown();server.server_close();thread.join(timeout=2)


def test_http_roundtrip_keeps_bound_observation_image_and_model_provenance():
    import hashlib
    from sim.ausc_policy import astra_service as service,astra_supervisor as protocol
    calls=[];token='test-'+32*'x'
    def infer(obs,rgb):
        calls.append((obs,rgb))
        return dict(model=protocol.MODEL,backend='codex_exec',decision=decision(obs),events=[],
            service_source_sha256=hashlib.sha256(Path(service.__file__).read_bytes()).hexdigest(),
            protocol_source_sha256=hashlib.sha256(Path(protocol.__file__).read_bytes()).hexdigest())
    server=service.make_server(0,token,infer)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        obs=observation();rgb=b'actual-native-image';obs['rgb_sha256']=hashlib.sha256(rgb).hexdigest()
        result=protocol.HTTPClient(f'http://127.0.0.1:{server.server_port}/decision',token)(obs,rgb)
        assert result['decision']['observation_id']==obs['observation_id']
        assert calls==[(obs,rgb)]
    finally:server.shutdown();server.server_close();thread.join(timeout=2)


def test_boundary_request_hashes_bind_authoritative_linux_artifacts(tmp_path):
    import hashlib
    c=bridge(tmp_path);c._move_to_prepose({})
    row=c.astra_decisions[0]
    assert row['request_sha256']==hashlib.sha256((tmp_path/'0000/request.json').read_bytes()).hexdigest()
    assert row['observation_sha256']==hashlib.sha256((tmp_path/'0000/observation.json').read_bytes()).hexdigest()


def test_native_bridge_cannot_claim_qualified_success_from_skill_ack(tmp_path):
    from sim.ausc_policy.astra_native import tag_result
    c=bridge(tmp_path);c._move_to_prepose({})
    result=tag_result(dict(success=False,physical_validation=False,points=[]),c)
    assert result['success'] is False and result['physical_validation'] is False
    assert result['learned_policy'] is False and result['expert_demonstration'] is False
    assert result['closed_loop_evaluated'] is False


def test_http_rejects_service_source_drift_before_any_native_dispatch():
    from sim.ausc_policy import astra_service as service,astra_supervisor as protocol
    token='test-'+32*'x'
    def infer(obs,rgb):
        return dict(model=protocol.MODEL,backend='codex_exec',decision=decision(obs),
                    service_source_sha256='0'*64,protocol_source_sha256='0'*64)
    server=service.make_server(0,token,infer)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        with pytest.raises(ValueError,match='source'):
            protocol.HTTPClient(f'http://127.0.0.1:{server.server_port}/decision',token)(observation(),None)
    finally:server.shutdown();server.server_close();thread.join(timeout=2)


def test_native_astra_rejects_relaxed_plane_before_any_inference():
    from sim.ausc_policy.astra_native import AstraGivenContactExpert
    c=AstraGivenContactExpert.__new__(AstraGivenContactExpert)
    row=dict(selected_fit=dict(first_contact=dict(contact_plane_normal_world=[0.,0.,1.],
        contact_plane_radius_m=.023,plane_tilt_limit_deg=10.,tilt_relaxed=True)))
    with pytest.raises(ValueError,match='strict'):c.run([(None,row)])


def test_native_astra_preserves_original_local_skill_result_for_strict_plane(monkeypatch):
    from sim.ausc_policy import astra_native as m
    c=m.AstraGivenContactExpert.__new__(m.AstraGivenContactExpert);c.astra_decisions=[]
    seen=[]
    def run(self,points):seen.append(points);return dict(success=False,physical_validation=False,points=[])
    monkeypatch.setattr(m.GivenContactExpert,'run',run)
    row=dict(selected_fit=dict(first_contact=dict(contact_plane_normal_world=[0.,0.,1.],
        contact_plane_radius_m=.023,plane_tilt_limit_deg=3.,tilt_relaxed=False)))
    result=c.run([(None,row)])
    assert len(seen)==1 and seen[0][0][1] is row
    assert result['physical_validation'] is False and result['supervisor_model']=='gpt-6-astra'
