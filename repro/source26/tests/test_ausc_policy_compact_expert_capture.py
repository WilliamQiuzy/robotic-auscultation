"""CPU native-hook tests; no physics or learned-policy qualification."""
from copy import deepcopy
import json

import numpy as np
import pytest

from test_ausc_policy_expert_capture import Env, Backend, GEOMETRY, binding, labels
from test_ausc_policy_training_capture import metadata, bindings
from sim.ausc_policy import expert_capture, training_capture


def compact_binding(**overrides):
    meta = metadata()
    meta['episode_id'] = binding()['episode_id']
    settings = dict(metadata=meta, code_bindings=bindings(), max_frames=10,
                    native_image_size=[640,360],
                    estimated_target_base=np.eye(4).tolist(), target_confidence=.8,
                    target_estimate_source_id='seeded-estimate-1')
    settings.update(overrides)
    return dict(binding(), profile='compact', compact=settings)


def test_compact_installed_body_identity_and_split_must_match(tmp_path):
    geometry=deepcopy(GEOMETRY)
    geometry['patient_provenance']=dict(identity='p000438',split='test',asset_sha256='d'*64,
        receipt_sha256='e'*64)
    with pytest.raises(ValueError,match='installed patient'):
        expert_capture.create_expert_capture(tmp_path/'capture',binding=compact_binding(),
            geometry_metadata=geometry)


def test_production_compact_requires_explicit_authenticated_body(tmp_path):
    value=compact_binding();value['compact']['metadata']['source_kind']='isaac_live_physics'
    with pytest.raises(ValueError,match='authenticated installed patient'):
        expert_capture.create_expert_capture(tmp_path/'capture',binding=value,geometry_metadata=GEOMETRY)


def make(tmp_path, **overrides):
    env = Env()
    capture = expert_capture.create_expert_capture(tmp_path/'capture',
        binding=compact_binding(**overrides), geometry_metadata=GEOMETRY)
    capture.start(env, backend_factory=Backend)
    return env, capture


def actor_labels(t, value=None):
    from sim.ausc_policy.m2_stage import RULE_ID,NATIVE_SOURCE
    value=deepcopy(value or labels())
    value['actor']=dict(phase='APPROACH',owner='L2',sampled_at_s=t,rule_id=RULE_ID,
        source=NATIVE_SOURCE,estimated_tcp_distance_m=.08)
    return value


def tick(env, capture, *, before=None, after=None, clean=True, with_actor=True):
    index = len(env.rows)
    env.advance()
    qcmd = np.arange(7, dtype=float)/100 + (index+1)/1000
    env.rows[-1]['q_command'] = qcmd.tolist()
    capture.after_physics(env,index)
    command = dict(q_command=qcmd, issued_at_s=index/240.,
        endpoint_at_s=(index+1)/240., readback_at_s=(index+1)/240., source_id=f'command-{index}')
    capture.finish_tick(env,index,actor_labels(index/240.,before) if with_actor else before or labels(),
        actor_labels((index+1)/240.,after) if with_actor else after or labels(),
        clean_command=command if clean else None,control_owner='L2')
    return qcmd


def test_profiles_are_explicit_and_full_raw_remains_compatible(tmp_path):
    assert expert_capture.capture_settings({'raw_multimodal_capture':compact_binding()})['profile']=='compact'
    raw = dict(binding(),profile='full_raw')
    capture = expert_capture.create_expert_capture(tmp_path/'raw',binding=raw,geometry_metadata=GEOMETRY)
    assert type(capture) is expert_capture.ExpertCapture
    capture.close()


def test_direct_raw_constructor_cannot_silently_record_compact_as_full_raw(tmp_path):
    with pytest.raises(ValueError,match='factory|create_expert_capture'):
        expert_capture.ExpertCapture(tmp_path/'raw',binding=compact_binding(),geometry_metadata=GEOMETRY)


@pytest.mark.parametrize('change',[dict(subject_id='p000452'),dict(split='test')])
def test_compact_rejects_quarantined_or_wrong_split_identity(change):
    value=compact_binding()
    value['compact']['metadata'].update(change)
    with pytest.raises(ValueError,match='identity|split|quarantined'):
        expert_capture.capture_settings({'raw_multimodal_capture':value})


def test_compact_hook_pairs_past_observation_with_future_clean_command(tmp_path):
    env,capture = make(tmp_path)
    for _ in range(27): tick(env,capture)
    capture.close()
    result = capture.result(27,execution_completed=True)
    assert result['compact_capture_complete'] and not result['raw_capture_complete']
    assert result['capture_profile']=='compact' and not result['training_eligible']
    rows=list(training_capture.iter_training_records(capture.output/'compact'))
    assert len(rows)==2 and rows[0]['observed_at_s']==.05
    assert rows[0]['action_endpoint_at_s']==.1
    np.testing.assert_allclose(rows[0]['q_command'],np.arange(7)/100+.024)
    np.testing.assert_array_equal(rows[0]['measured_q'],np.zeros(7))
    assert rows[0]['rgb'].shape==(144,256,3)
    assert rows[0]['state17'].shape==(17,)
    assert rows[0]['hall_received_time_s']<=.05
    assert 'q_command' not in rows[1]
    terminal=json.loads((capture.output/'compact/terminal.json').read_text())
    assert terminal['final_native_step_count']==27 and terminal['terminal_tick_count']==3
    assert result['total_bytes']<=100_000_000
    assert not (capture.output/'hall').exists() and not (capture.output/'rgbd').exists()
    assert not list(capture.output.rglob('frame-*.npz'))


def test_sensor_zero_boundary_keeps_raw_physics_and_audits_mapping(tmp_path):
    import gzip
    env,capture=make(tmp_path)
    env.advance();raw=-5.130387226159883e-9
    env.rows[-1]['compression_m']=raw;env.rows[-1]['q_command']=np.zeros(7).tolist()
    capture.after_physics(env,0)
    command=dict(q_command=np.zeros(7),issued_at_s=0.,endpoint_at_s=1/240.,
                 readback_at_s=1/240.,source_id='command-0')
    capture.finish_tick(env,0,actor_labels(0.),actor_labels(1/240.),clean_command=command,control_owner='L2')
    assert env.rows[-1]['compression_m']==raw
    capture.close()
    with gzip.open(capture.output/'native_audit.jsonl.gz','rt') as stream:row=json.loads(next(stream))
    assert row['measurements']['compression_m']==raw
    assert row['hall_sensor_input']==dict(raw_compression_m=raw,sensor_compression_m=0.,zero_boundary_delta_m=-raw)
    assert capture.result(1,execution_completed=True)['hall_zero_conversion']['normalized_ticks']==1


@pytest.mark.parametrize('compression',['below-bound',.0072500001,float('nan'),float('inf')])
def test_sensor_zero_conversion_does_not_hide_real_travel_errors(tmp_path,compression):
    env,capture=make(tmp_path);env.advance()
    if compression=='below-bound':compression=np.nextafter(-capture.hall_zero_tolerance_m,-np.inf)
    env.rows[-1]['compression_m']=compression
    with pytest.raises(ValueError):capture.after_physics(env,0)
    assert np.isnan(env.rows[-1]['compression_m']) if np.isnan(compression) else env.rows[-1]['compression_m']==compression
    capture.close();assert not capture.result(1,execution_completed=True)['compact_capture_complete']


def test_exact_lower_adc_equivalent_bound_has_same_packets_as_zero(tmp_path):
    env,a=make(tmp_path/'bounded');zero,b=make(tmp_path/'zero')
    eps=a.hall_zero_tolerance_m
    assert eps==pytest.approx(3.809856306074158e-6,rel=1e-12)
    policy=a.hall_zero_policy
    codes=np.rint(np.linspace(policy['zero_quantizer_input'],policy['lower_endpoint_linear_quantizer_input'],10001))
    assert np.array_equal(np.unique(codes),[1974])
    for i in range(12):
        env.advance();zero.advance()
        env.rows[-1]['compression_m']=-eps;zero.rows[-1]['compression_m']=0.
        a.after_physics(env,i);b.after_physics(zero,i)
        assert a.sensor.read((i+1)/240.)==b.sensor.read((i+1)/240.)
    assert a._hall_zero_count==12 and b._hall_zero_count==0
    assert a._hall_zero_max_delta_m==eps
    a.close();b.close()


def test_compact_rejects_missing_clean_command_instead_of_using_measured_q(tmp_path):
    env,capture=make(tmp_path)
    with pytest.raises(ValueError,match='clean command'):
        tick(env,capture,clean=False)
    capture.close()
    assert not capture.result(1,execution_completed=True)['compact_capture_complete']


def test_scalar_or_wrong_command_cannot_become_action_label(tmp_path):
    env,capture=make(tmp_path)
    env.advance(); env.rows[-1]['q_command']=np.zeros(7).tolist()
    capture.after_physics(env,0)
    command=dict(q_command=1.,issued_at_s=0.,endpoint_at_s=1/240.,
                 readback_at_s=1/240.,source_id='bad')
    with pytest.raises(ValueError,match='seven|7|command'):
        capture.finish_tick(env,0,actor_labels(0.),actor_labels(1/240.),clean_command=command,control_owner='L2')
    capture.close()


def test_compact_retains_intervening_context_transitions(tmp_path):
    import gzip
    env,capture=make(tmp_path)
    for i in range(24):
        tick(env,capture,after=labels('HOLD') if i==15 else labels())
    capture.close()
    with gzip.open(capture.output/'native_audit.jsonl.gz','rt') as stream:
        events=[json.loads(line) for line in stream]
    assert len(events)==24
    assert events[15]['controller_after']['phase']=='HOLD'
    assert events[16]['controller_after']['phase']=='APPROACH'
    assert all(event['state_audit']=='completed' for event in events)


def test_compact_camera_preserves_render_clock_and_closes_resources(tmp_path):
    env,capture=make(tmp_path)
    backend=capture.camera._backend
    backend.render=lambda: setattr(env.world,'current_time',env.world.current_time+.01)
    with pytest.raises(ValueError,match='clock'):
        for _ in range(12): tick(env,capture)
    assert not capture.camera.finalize()['completed']
    capture.close()
    assert 'destroy' in backend.events
    assert not capture.result(12,execution_completed=True)['compact_capture_complete']


def test_missing_audit_file_returns_failed_receipt_instead_of_raising(tmp_path):
    env,capture=make(tmp_path)
    for _ in range(12):tick(env,capture)
    capture.close();(capture.output/'native_audit.jsonl.gz').unlink()
    result=capture.result(12,execution_completed=True)
    assert not result['compact_capture_complete']
    assert any(row['stage']=='artifact_inventory' for row in result['errors'])


def test_late_budget_failure_receipt_counts_its_error_bytes(tmp_path):
    env,capture=make(tmp_path)
    for _ in range(12):tick(env,capture)
    capture.close();capture.limit_bytes=100
    result=capture.result(12,execution_completed=True)
    assert not result['compact_capture_complete']
    assert result['total_bytes']==sum(p.stat().st_size for p in capture.output.rglob('*') if p.is_file())


def test_compact_bound_failure_never_reports_success(tmp_path):
    env,capture=make(tmp_path,max_frames=1)
    with pytest.raises(ValueError,match='maximum|bound'):
        for _ in range(24): tick(env,capture)
    capture.close()
    result=capture.result(24,execution_completed=True)
    assert not result['compact_capture_complete'] and result['total_bytes']<=100_000_000


def test_compact_missing_native_audit_cannot_complete(tmp_path):
    env,capture=make(tmp_path)
    for _ in range(12): tick(env,capture)
    env.advance();env.rows[-1]['q_command']=np.zeros(7).tolist()
    capture.after_physics(env,12)
    capture.close()
    assert not capture.result(13,execution_completed=True)['compact_capture_complete']


@pytest.mark.parametrize('count',[0,1,11,12,13,24,25])
def test_compact_adaptive_terminal_counts_do_not_add_native_steps(tmp_path,count):
    env,capture=make(tmp_path)
    for _ in range(count): tick(env,capture)
    capture.close()
    result=capture.result(count,execution_completed=True)
    assert result['compact_capture_complete']
    receipt=result['receipts']['compact']
    assert receipt['frames']==count//12
    assert receipt['complete_pairs']==max(0,count//12-1)
    assert receipt['terminal_tick_count']==count%12
    assert len(env.rows)==count
    assert result['total_bytes']==sum(path.stat().st_size for path in capture.output.rglob('*') if path.is_file())


def test_compact_auxiliary_and_payload_share_one_byte_cap(tmp_path):
    env,capture=make(tmp_path,limit_bytes=262144,max_frames=100)
    rng=np.random.default_rng(51)
    capture.camera._backend.read=lambda:(rng.integers(0,256,(360,640,3),dtype=np.uint8),
        rng.uniform(.1,2.,(360,640)).astype(np.float32))
    with pytest.raises(ValueError,match='byte budget'):
        for _ in range(120): tick(env,capture)
    capture.close()
    result=capture.result(len(env.rows),execution_completed=True)
    assert not result['compact_capture_complete']
    assert sum(path.stat().st_size for path in capture.output.rglob('*') if path.is_file())<=262144
