"""Synthetic CPU recording tests, never native physics/training qualification."""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from sim.ausc_policy import m2_contract as contract
from sim.ausc_policy.geometry_epoch import GEOMETRY_ID, GEOMETRY_SOURCE_SHA256
from sim.auscultation_vla_v2.hall import HallCalibration, HallPacket

ROOT = Path(__file__).resolve().parents[1]
CAMERA = (ROOT/'sim/camera_framing_20260923/recommended_camera.json').read_bytes()
CALIBRATION = (ROOT/'sim/auscultation_vla_v2/calibration/veo6_tared_snapshot.json').read_bytes()
CAL = HallCalibration(CALIBRATION)
Q = np.array([.1, -.2, -.1, .3, .1, .1, -.2])


def module():
    assert importlib.util.find_spec('sim.ausc_policy.training_capture') is not None, 'compact training capture is missing'
    return importlib.import_module('sim.ausc_policy.training_capture')


def metadata():
    return dict(geometry_id=GEOMETRY_ID, geometry_source_sha256=GEOMETRY_SOURCE_SHA256,
                collision_geometry_sha256='a'*64, collision_arrays_sha256='b'*64,
                freshly_generated=True, contract_id=contract.CONTRACT_ID, state_dim=17,
                horizon=20, execution_prefix=10, control_hz=20, joint_unit='rad',
                position_unit='m', target_pose_frame='openarm_left_link0',
                target_source='rgbd_surface_tracking', episode_id='synthetic-r3',
                target_id='R3', subject_id='p000439', split='train',
                identity_split_sha256=contract.IDENTITY_SPLIT_SHA256,
                source_kind='cpu_synthetic_test', training_eligible=True,
                physical_qualification=True, artifact_hashes_verified=True)


def bindings():
    return {key:'c'*64 for key in ('native_backend', 'state_adapter', 'target_estimator', 'action_source', 'body')}


def writer(tmp_path, **kwargs):
    options = dict(metadata=metadata(), camera_bytes=CAMERA, calibration_bytes=CALIBRATION,
                   code_bindings=bindings(), expected_frames=2)
    options.update(kwargs)
    return module().CompactTrainingCapture(tmp_path/'training_capture', **options)


def packet(sample, received=None, raw=None):
    return HallPacket(int(CAL.voltage_at_displacement(.001)/CAL.supply_v*CAL.adc_full_scale) if raw is None else raw,
                      sample, sample if received is None else received, CAL.sha256, 'simulated_adc')


def frame(i=0):
    t=(i+1)*.05
    estimate=np.eye(4); estimate[:3,3]=[.05, .1, .8]
    depth=np.full((360,640), .54321, np.float32)
    depth[:20,:20]=np.nan
    return dict(episode_id='synthetic-r3',target_id='R3',observed_at_s=t, rgb=np.full((360,640,3), [40,100,160],np.uint8),
                depth_m=depth, depth_valid=np.isfinite(depth)&(depth>0),
                measured_q=Q.copy(), estimated_target_base=estimate, target_confidence=.8,
                state17=contract.encode_state(Q,estimate,.8,metadata=metadata()),
                target_source='rgbd_surface_tracking', target_estimate_time_s=t,
                observation_source_id=f'observation-{i}', target_estimate_source_id=f'estimate-{i}',
                phase='APPROACH', control_owner='L2',
                action=dict(q_command=Q+.01, issued_at_s=t+.001,
                            endpoint_at_s=t+.05, readback_at_s=t+.05,
                            source_id=f'action-{i}'),
                hall_packets=[packet(t-.04+j*.01) for j in range(5)])


def test_roundtrip_keeps_metric_depth_state_action_hall_and_explicit_provenance(tmp_path):
    w=writer(tmp_path)
    rows=[frame(i) for i in range(2)]
    for row in rows: w.append(**row)
    receipt=w.close()
    assert receipt['completed'] and receipt['frames']==2
    assert receipt['total_bytes']==sum(p.stat().st_size for p in w.output.iterdir())
    assert receipt['total_bytes']<100_000_000
    assert not any(receipt[key] for key in ('training_eligible','physical_qualification','artifact_hashes_verified'))
    manifest=json.loads((w.output/'manifest.json').read_text())
    assert manifest['policy_image_size']==[256,144]
    assert manifest['native_image_size']==[640,360]
    assert manifest['depth_unit_m']==.0001
    assert manifest['hall_downsampling']=='latest causally received packet; no averaging'
    assert manifest['camera_sha256']==hashlib.sha256(CAMERA).hexdigest()
    assert manifest['calibration_sha256']==CAL.sha256
    assert manifest['code_bindings']['target_estimator']=='c'*64
    assert 'writer' in manifest['code_bindings'] and 'action_contract' in manifest['code_bindings']
    assert not any(manifest['metadata'][key] for key in ('training_eligible','physical_qualification','artifact_hashes_verified'))
    original=json.loads(CAMERA)['intrinsics']
    assert manifest['policy_intrinsics']['fx']==pytest.approx(original['fx']*256/1920)
    saved=list(module().iter_training_records(w.output))
    for i,record in enumerate(saved):
        assert record['rgb'].shape==(144,256,3)
        assert np.max(np.abs(record['rgb'].astype(int)-rows[i]['rgb'][0,0]))<=3
        np.testing.assert_array_equal(record['state17'],rows[i]['state17'])
        np.testing.assert_array_equal(record['q_command'], rows[i]['action']['q_command'])
        assert record['observed_at_s']==(i+1)*.05
        assert record['action_endpoint_at_s']==pytest.approx((i+2)*.05)
        assert record['action_issued_at_s']<record['action_readback_at_s']
        assert record['depth_q'].dtype==np.dtype('<u4')
        assert record['depth_valid'].dtype==bool and not record['depth_valid'][0,0]
        assert record['depth_q'][0,0]==0
        np.testing.assert_allclose(record['depth_q'][record['depth_valid']]*.0001,.54321,atol=.00005)
        assert record['hall_sample_time_s']==pytest.approx((i+1)*.05)
        decoded=CAL.decode(rows[i]['hall_packets'][-1],now_s=(i+1)*.05)
        assert record['hall_spring_force_estimate_n']==decoded.spring_force_estimate_n
        assert record['hall_voltage_v']==decoded.hall_voltage_v
    assert not list(w.output.rglob('*.jsonl'))
    assert not any(p.is_dir() for p in w.output.iterdir())
    assert w.close()==receipt


@pytest.mark.parametrize('change,match',[
    ({'observed_at_s':.051},'20 Hz'),
    ({'rgb':np.zeros((144,256,3),np.uint8)},'RGB'),
    ({'rgb':np.zeros((360,640,3),float)},'RGB'),
    ({'depth_m':np.zeros((360,640,3))},'depth'),
    ({'depth_valid':np.ones((360,640),np.uint8)},'validity'),
    ({'state17':np.zeros(17)},'state17'),
    ({'state17':np.full(17,np.nan)},'state17'),
    ({'measured_q':np.zeros(6)},'measured_q'),
    ({'target_source':'ground_truth'},'target source'),
    ({'target_estimate_time_s':.051},'estimate'),
    ({'observation_source_id':''},'source'),
    ({'hall_packets':[]},'Hall'),
    ({'hall_packets':[packet(.05,.051)]},'Hall'),
    ({'hall_packets':[packet(.04),packet(.03)]},'Hall'),
    ({'hall_packets':[dict(force=4.)]},'Hall'),
    ({'hall_packets':[packet(.05,raw=0)]},'Hall'),
    ({'hall_packets':[packet(.05,raw=60000)]},'Hall'),
])
def test_bad_modality_latches_failure_and_cannot_silently_complete(tmp_path,change,match):
    w=writer(tmp_path,expected_frames=1)
    row=frame();row.update(change)
    with pytest.raises(ValueError,match=match):w.append(**row)
    with pytest.raises(RuntimeError,match='failed'):w.append(**frame())
    with pytest.raises(RuntimeError,match='incomplete'):w.close()
    receipt=json.loads((w.output/'receipt.json').read_text())
    assert not receipt['completed'] and receipt['frames']==0


@pytest.mark.parametrize('change',[
    dict(issued_at_s=.049),dict(issued_at_s=.101),dict(endpoint_at_s=.05),
    dict(readback_at_s=.05),dict(q_command=np.full(7,np.nan)),dict(q_command=np.zeros(8)),
    dict(source_id=''),
])
def test_action_must_be_future_endpoint_with_causal_command_and_readback(tmp_path,change):
    w=writer(tmp_path,expected_frames=1);row=frame();row['action'].update(change)
    with pytest.raises(ValueError,match='action'):w.append(**row)
    w.abort('test rejected action')


def test_held_or_missing_adc_is_not_replaced_by_future_packet_or_invented_force(tmp_path):
    w=writer(tmp_path)
    first=frame();first['hall_packets']=[packet(.025,.03),packet(.035,.05)]
    w.append(**first)
    second=frame(1);second['hall_packets']=[packet(.06,.09),packet(.075,.10)]
    w.append(**second);w.close()
    records=list(module().iter_training_records(w.output))
    assert [row['hall_sample_time_s'] for row in records]==[.035,.075]
    assert [row['hall_received_time_s'] for row in records]==[.05,.10]


def test_stale_or_reused_hall_and_duplicate_clock_are_rejected(tmp_path):
    w=writer(tmp_path);w.append(**frame())
    second=frame(1);second['hall_packets']=[packet(.04)]
    with pytest.raises(ValueError,match='Hall'):w.append(**second)
    w.abort('test stale packet')
    other=writer(tmp_path/'duplicate');other.append(**frame())
    with pytest.raises(ValueError,match='20 Hz'):other.append(**frame())
    other.abort('test duplicated frame')


@pytest.mark.parametrize('kind',['range','mismatch','all_invalid'])
def test_depth_rejects_silent_saturation_or_false_validity(tmp_path,kind):
    w=writer(tmp_path,expected_frames=1);row=frame()
    if kind=='range':row['depth_m'][40:80,40:80]=500_000.
    elif kind=='mismatch':row['depth_valid'][40,40]=False
    else:row['depth_m'][:]=np.nan;row['depth_valid'][:]=False
    with pytest.raises(ValueError,match='depth|validity'):w.append(**row)
    w.abort('test bad depth')


def test_partial_close_and_context_exception_leave_explicit_incomplete_receipt(tmp_path):
    w=writer(tmp_path);w.append(**frame())
    with pytest.raises(RuntimeError,match='incomplete'):w.close()
    assert not json.loads((w.output/'receipt.json').read_text())['completed']
    with pytest.raises(RuntimeError,match='incomplete'):list(module().iter_training_records(w.output))
    with pytest.raises(RuntimeError,match='interrupted'):
        with writer(tmp_path/'other') as capture:
            capture.append(**frame())
            raise RuntimeError('interrupted')
    assert not json.loads((tmp_path/'other/training_capture/receipt.json').read_text())['completed']


def test_budget_is_actual_whole_episode_bytes_and_overflow_is_fail_closed(tmp_path):
    w=writer(tmp_path,expected_frames=1,limit_bytes=30_000)
    row=frame();row['rgb']=np.random.default_rng(4).integers(0,256,(360,640,3),dtype=np.uint8)
    row['depth_m']=np.random.default_rng(5).uniform(.2,5,(360,640)).astype(np.float32)
    row['depth_valid'][:]=True
    with pytest.raises(ValueError,match='byte budget'):w.append(**row)
    with pytest.raises(RuntimeError,match='incomplete'):w.close()
    receipt=json.loads((w.output/'receipt.json').read_text())
    assert not receipt['completed'] and receipt['frames']==0
    assert receipt['total_bytes']<=30_000
    with pytest.raises(ValueError,match='limit'):writer(tmp_path/'too_big',limit_bytes=100_000_001)


@pytest.mark.parametrize('change',[
    dict(target_source='ground_truth'),dict(target_id='L99'),dict(subject_id=''),
    dict(collision_geometry_sha256=None),dict(identity_split_sha256='f'*64),
])
def test_invalid_episode_provenance_is_rejected_before_output(tmp_path,change):
    meta=metadata();meta.update(change)
    with pytest.raises(ValueError):writer(tmp_path,metadata=meta)
    assert not (tmp_path/'training_capture').exists()


def test_noisy_target_requires_bound_measured_noise_model(tmp_path):
    meta=metadata();meta['target_source']='gt_plus_measured_noise'
    with pytest.raises(ValueError,match='noise_model'):writer(tmp_path,metadata=meta)
    codes=bindings();codes['noise_model']='d'*64
    w=writer(tmp_path,metadata=meta,code_bindings=codes,expected_frames=1)
    row=frame();row['target_source']='gt_plus_measured_noise';w.append(**row);w.close()
    assert json.loads((w.output/'manifest.json').read_text())['metadata']['target_source']=='gt_plus_measured_noise'


def test_real_acquisition_is_linux_only_cpu_fixture_is_explicit(tmp_path,monkeypatch):
    mod=module();monkeypatch.setattr(mod.sys,'platform','darwin')
    meta=metadata();meta['source_kind']='isaac_live_physics'
    with pytest.raises(RuntimeError,match='Linux'):writer(tmp_path,metadata=meta)


@pytest.mark.parametrize('filename',['rgb.mjpeg','numeric.bin','manifest.json'])
def test_reader_rejects_corrupted_complete_bundle(tmp_path,filename):
    w=writer(tmp_path,expected_frames=1);w.append(**frame());w.close()
    path=w.output/filename;data=bytearray(path.read_bytes());data[len(data)//2]^=1;path.write_bytes(data)
    with pytest.raises(ValueError,match='integrity'):list(module().iter_training_records(w.output))


def test_reader_rejects_truncation_and_extra_frame(tmp_path):
    w=writer(tmp_path,expected_frames=1);w.append(**frame());w.close()
    data=(w.output/'numeric.bin').read_bytes();(w.output/'numeric.bin').write_bytes(data[:-7])
    with pytest.raises(ValueError,match='integrity'):list(module().iter_training_records(w.output))


def test_frame_context_must_match_single_target_and_episode(tmp_path):
    w=writer(tmp_path,expected_frames=1)
    row=frame();row.update(episode_id='synthetic-r3',target_id='R3')
    w.append(**row);w.close()
    saved=next(module().iter_training_records(w.output))
    assert saved['episode_id']=='synthetic-r3' and saved['target_id']=='R3'
    for key,value in [('episode_id','another'),('target_id','L4')]:
        other=writer(tmp_path/key,expected_frames=1)
        changed=deepcopy(row);changed[key]=value
        with pytest.raises(ValueError,match='episode|target'):other.append(**changed)
        other.abort('crossing context')


def test_missing_frame_context_cannot_be_inferred_from_manifest(tmp_path):
    w=writer(tmp_path,expected_frames=1)
    row=frame();del row['episode_id'];del row['target_id']
    with pytest.raises(ValueError,match='modality|context'):w.append(**row)
    with pytest.raises(RuntimeError,match='incomplete'):w.close()


def test_unexpected_output_file_cannot_get_complete_receipt(tmp_path):
    w=writer(tmp_path,expected_frames=1);w.append(**frame())
    (w.output/'unbound.txt').write_text('external bytes')
    with pytest.raises(RuntimeError,match='incomplete'):w.close()
    assert not json.loads((w.output/'receipt.json').read_text())['completed']


def test_metric_depth_quantization_error_bound_uses_input_float_value(tmp_path):
    w=writer(tmp_path,expected_frames=1);row=frame()
    row['depth_m'][:]=np.float32(.50005);row['depth_valid'][:]=True
    source=float(row['depth_m'][0,0]);w.append(**row);w.close()
    record=next(module().iter_training_records(w.output))
    assert np.max(np.abs(record['depth_q'].astype(float)*.0001-source))<=.00005+1e-12


@pytest.mark.parametrize('missing',['rgb','depth_m','depth_valid','state17','action','hall_packets'])
def test_omitted_modality_latches_failure_even_after_expected_frames(tmp_path,missing):
    w=writer(tmp_path,expected_frames=1);w.append(**frame());row=frame(1);del row[missing]
    with pytest.raises(ValueError,match='modality'):w.append(**row)
    with pytest.raises(RuntimeError,match='incomplete'):w.close()


def test_partial_write_cannot_publish_successful_recording(tmp_path):
    w=writer(tmp_path,expected_frames=1)
    underlying=w._numeric
    class FailNumericWrite:
        def write(self,payload):
            underlying.write(payload[:10]);underlying.flush()
            raise OSError('injected interrupted numeric write')
        def close(self):underlying.close()
    w._numeric=FailNumericWrite()
    with pytest.raises(OSError,match='interrupted'):w.append(**frame())
    with pytest.raises(RuntimeError,match='incomplete'):w.close()
    receipt=json.loads((w.output/'receipt.json').read_text())
    assert receipt['frames']==0 and not receipt['completed']
    assert receipt['total_bytes']==sum(p.stat().st_size for p in w.output.iterdir())


def adaptive_writer(tmp_path, **kwargs):
    return writer(tmp_path,expected_frames=None,max_frames=kwargs.pop('max_frames',20),**kwargs)


def test_adaptive_episode_closes_shorter_than_bound_with_actual_native_endpoint(tmp_path):
    with adaptive_writer(tmp_path,max_frames=10) as w:
        w.append(**frame());w.append(**frame(1))
        receipt=w.close(final_native_step_count=36,final_native_time_s=.15)
    assert receipt['completed'] and receipt['frames']==2
    assert receipt['expected_frames'] is None and receipt['max_frames']==10
    assert receipt['duration_mode']=='adaptive'
    assert receipt['final_native_step_count']==36 and receipt['final_native_time_s']==.15
    assert not any(receipt[k] for k in ('training_eligible','physical_qualification','artifact_hashes_verified'))
    assert len(list(module().iter_training_records(w.output)))==2


@pytest.mark.parametrize('closing',[
    {},dict(final_native_step_count=48,final_native_time_s=.20),
    dict(final_native_step_count=24,final_native_time_s=.10),
    dict(final_native_step_count=35,final_native_time_s=35/240),
    dict(final_native_step_count=36,final_native_time_s=.151),
    dict(final_native_step_count=True,final_native_time_s=.15),
    dict(final_native_step_count=36,final_native_time_s=float('nan')),
])
def test_adaptive_close_requires_exact_final_native_count_and_time(tmp_path,closing):
    w=adaptive_writer(tmp_path);w.append(**frame());w.append(**frame(1))
    with pytest.raises(RuntimeError,match='incomplete'):w.close(**closing)
    receipt=json.loads((w.output/'receipt.json').read_text())
    assert not receipt['completed'] and 'native' in receipt['failure']


def test_adaptive_clock_gap_remains_fatal_even_if_native_end_count_is_plausible(tmp_path):
    w=adaptive_writer(tmp_path);w.append(**frame())
    with pytest.raises(ValueError,match='20 Hz'):w.append(**frame(2))
    with pytest.raises(RuntimeError,match='incomplete'):
        w.close(final_native_step_count=24,final_native_time_s=.1)


def test_adaptive_maximum_cannot_be_exceeded_or_empty_capture_accepted(tmp_path):
    w=adaptive_writer(tmp_path,max_frames=1);w.append(**frame())
    with pytest.raises(ValueError,match='count'):w.append(**frame(1))
    with pytest.raises(RuntimeError,match='incomplete'):
        w.close(final_native_step_count=24,final_native_time_s=.1)
    empty=adaptive_writer(tmp_path/'empty')
    with pytest.raises(RuntimeError,match='incomplete'):
        empty.close(final_native_step_count=12,final_native_time_s=.05)


def test_adaptive_budget_overflow_cannot_be_accepted_as_shorter_episode(tmp_path):
    w=adaptive_writer(tmp_path,limit_bytes=30_000)
    row=frame();row['depth_m']=np.random.default_rng(9).uniform(.2,5,(360,640)).astype(np.float32)
    row['depth_valid'][:]=True
    with pytest.raises(ValueError,match='byte budget'):w.append(**row)
    with pytest.raises(RuntimeError,match='incomplete'):
        w.close(final_native_step_count=12,final_native_time_s=.05)
    receipt=json.loads((w.output/'receipt.json').read_text())
    assert not receipt['completed'] and receipt['total_bytes']<=30_000


@pytest.mark.parametrize('options',[
    dict(expected_frames=None,max_frames=None),dict(expected_frames=2,max_frames=3),
    dict(expected_frames=None,max_frames=0),dict(expected_frames=None,max_frames=True),
])
def test_duration_mode_needs_exactly_one_positive_frame_bound(tmp_path,options):
    with pytest.raises(ValueError,match='frame'):writer(tmp_path,**options)
    assert not (tmp_path/'training_capture').exists()


@pytest.mark.parametrize('received',[.05,.05+5e-10])
def test_hall_batch_cannot_defer_packet_available_at_prior_observation(tmp_path,received):
    w=writer(tmp_path)
    first=frame();first['hall_packets']=[packet(.035,.05)];w.append(**first)
    second=frame(1);second['hall_packets']=[packet(.05,received)]
    with pytest.raises(ValueError,match='Hall.*prior observation'):w.append(**second)
    with pytest.raises(RuntimeError,match='incomplete'):w.close()


def test_hall_first_batch_allows_episode_start_later_batches_allow_new_delayed_arrival(tmp_path):
    w=writer(tmp_path)
    first=frame();first['hall_packets']=[packet(0.,0.),packet(.035,.05)];w.append(**first)
    second=frame(1);second['hall_packets']=[packet(.05,.051),packet(.08,.095)];w.append(**second)
    w.close()
    records=list(module().iter_training_records(w.output))
    assert [float(row['hall_received_time_s']) for row in records]==[.05,.095]


def test_unicode_abort_reason_fits_reserved_receipt_bytes(tmp_path):
    probe=writer(tmp_path/'probe')
    initial=sum(p.stat().st_size for p in probe.output.iterdir());probe.abort('probe')
    limit=initial+module().RECEIPT_RESERVE+100
    w=writer(tmp_path/'bounded',limit_bytes=limit)
    receipt=w.abort('错误'*1024)
    assert not receipt['completed'] and receipt['total_bytes']<=limit
    assert (w.output/'receipt.json').is_file()
    assert len(json.dumps(receipt['failure'],ensure_ascii=True).encode())<=2048
    with pytest.raises(RuntimeError,match='incomplete'):w.close()


def test_receipt_io_failure_is_latched_but_can_retry_incomplete_receipt(tmp_path,monkeypatch):
    w=writer(tmp_path,expected_frames=1);w.append(**frame())
    original=Path.open;failed=[False]
    def fail_receipt_once(path,*args,**kwargs):
        mode=args[0] if args else kwargs.get('mode','r')
        if path.name in ('receipt.json','.receipt.json.tmp') and any(c in mode for c in 'wx') and not failed[0]:
            failed[0]=True
            raise OSError('injected receipt write failure')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',fail_receipt_once)
    with pytest.raises(OSError,match='receipt write failure'):w.close()
    with pytest.raises(RuntimeError,match='failed'):w.append(**frame(1))
    with pytest.raises(RuntimeError,match='incomplete'):w.close()
    receipt=json.loads((w.output/'receipt.json').read_text())
    assert not receipt['completed'] and 'receipt' in receipt['failure']
    assert receipt['total_bytes']==sum(p.stat().st_size for p in w.output.iterdir())


def test_predeclared_frame_bound_cannot_exceed_supported_native_counter(tmp_path):
    with pytest.raises(ValueError,match='frame'):
        writer(tmp_path,expected_frames=10**4000)
    assert not (tmp_path/'training_capture').exists()


def test_uint32_metric_depth_keeps_near_foreground_and_far_native_background(tmp_path):
    w=writer(tmp_path,expected_frames=1);row=frame()
    row['depth_m'][180:,:]=np.float32(18.330701828)
    source=row['depth_m'].copy();w.append(**row);w.close()
    record=next(module().iter_training_records(w.output))
    assert record['depth_q'].dtype==np.dtype('<u4')
    ys=((np.arange(144)+.5)*360/144).astype(int);xs=((np.arange(256)+.5)*640/256).astype(int)
    sampled=source[np.ix_(ys,xs)].astype(float);valid=record['depth_valid']
    assert np.max(np.abs(record['depth_q'][valid].astype(float)*.0001-sampled[valid]))<=.00005+1e-12
    assert record['depth_valid'][100,:].all()
    assert float(record['depth_q'][100,100])*.0001>18.
    manifest=json.loads((w.output/'manifest.json').read_text())
    assert manifest['depth_storage_dtype']=='<u4'
    assert manifest['depth_max_m']==np.iinfo(np.uint32).max*.0001
    assert manifest['schema'].endswith('_v3')


def test_depth_above_uint32_metric_range_is_rejected_without_clipping(tmp_path):
    w=writer(tmp_path,expected_frames=1);row=frame()
    row['depth_m']=np.full((360,640),np.iinfo(np.uint32).max*.0001+.001,dtype=float)
    row['depth_valid'][:]=True
    with pytest.raises(ValueError,match='depth.*range'):w.append(**row)
    with pytest.raises(RuntimeError,match='incomplete'):w.close()


def test_reader_checks_metric_depth_dtype_even_when_bundle_hashes_are_recomputed(tmp_path):
    import io
    import zlib
    w=writer(tmp_path,expected_frames=1);w.append(**frame());w.close()
    numeric=w.output/'numeric.bin';raw=numeric.read_bytes()
    with np.load(io.BytesIO(raw[module().HEADER.size:]),allow_pickle=False) as archive:
        values={key:archive[key] for key in archive.files}
    values['depth_code']=values['depth_code'].astype('<i4')
    payload=io.BytesIO();np.savez_compressed(payload,**values);data=payload.getvalue()
    numeric.write_bytes(module().HEADER.pack(len(data),zlib.crc32(data))+data)
    receipt_path=w.output/'receipt.json';receipt=json.loads(receipt_path.read_text())
    receipt['artifacts']['numeric.bin']=dict(bytes=numeric.stat().st_size,sha256=hashlib.sha256(numeric.read_bytes()).hexdigest())
    base=sum(p.stat().st_size for p in w.output.iterdir() if p.name!='receipt.json')
    encode=lambda:(json.dumps(receipt,sort_keys=True,separators=(',',':'))+'\n').encode()
    while receipt['total_bytes']!=base+len(encode()):receipt['total_bytes']=base+len(encode())
    receipt_path.write_bytes(encode())
    with pytest.raises(ValueError,match='depth'):list(module().iter_training_records(w.output))


def test_owned_receipt_staging_replaced_by_symlink_cannot_write_external_file(tmp_path,monkeypatch):
    import os
    w=writer(tmp_path,expected_frames=1);w.append(**frame())
    def fail_publish(*args,**kwargs):raise OSError('injected receipt publication failure')
    with monkeypatch.context() as patch:
        patch.setattr(Path,'replace',fail_publish)
        patch.setattr(os,'link',fail_publish)
        with pytest.raises(OSError,match='publication'):w.close()
    staging=w.output/'.receipt.json.tmp'
    assert staging.is_file()
    staging.unlink();sentinel=tmp_path/'external.txt';sentinel.write_text('protected external bytes')
    staging.symlink_to(sentinel)
    with pytest.raises((OSError,RuntimeError,ValueError)):w.close()
    assert sentinel.read_text()=='protected external bytes'
    assert not (w.output/'receipt.json').exists()


def test_receipt_publication_does_not_overwrite_existing_foreign_file(tmp_path):
    w=writer(tmp_path,expected_frames=1);w.append(**frame())
    receipt=w.output/'receipt.json';receipt.write_text('existing foreign receipt')
    with pytest.raises((OSError,RuntimeError,ValueError)):w.close()
    assert receipt.read_text()=='existing foreign receipt'


def numeric_records(path):
    import io
    with path.open('rb') as stream:
        while True:
            header=stream.read(module().HEADER.size)
            if not header:break
            length,_=module().HEADER.unpack(header)
            with np.load(io.BytesIO(stream.read(length)),allow_pickle=False) as archive:
                yield {key:archive[key] for key in archive.files}


def rewrite_numeric_records(output,records):
    import io
    import zlib
    numeric=output/'numeric.bin'
    with numeric.open('wb') as stream:
        for record in records:
            payload=io.BytesIO();np.savez_compressed(payload,**record);data=payload.getvalue()
            stream.write(module().HEADER.pack(len(data),zlib.crc32(data))+data)
    receipt_path=output/'receipt.json';receipt=json.loads(receipt_path.read_text())
    receipt['artifacts']['numeric.bin']=dict(bytes=numeric.stat().st_size,sha256=hashlib.sha256(numeric.read_bytes()).hexdigest())
    base=sum(p.stat().st_size for p in output.iterdir() if p.name!='receipt.json')
    encode=lambda:(json.dumps(receipt,sort_keys=True,separators=(',',':'))+'\n').encode()
    while receipt['total_bytes']!=base+len(encode()):receipt['total_bytes']=base+len(encode())
    receipt_path.write_bytes(encode())


def test_depth_xor_keyframes_and_moving_validity_roundtrip_exact_quantized_values(tmp_path):
    w=writer(tmp_path,expected_frames=22);expected=[]
    ys=((np.arange(144)+.5)*360/144).astype(int);xs=((np.arange(256)+.5)*640/256).astype(int)
    for i in range(22):
        row=frame(i);row['depth_m'][180:,:]=np.float32(18.3307+i*.001)
        row['depth_m'][80:100,80:100]=np.nan if i%3 else np.float32(.7321)
        row['depth_valid']=np.isfinite(row['depth_m'])&(row['depth_m']>0)
        sampled=row['depth_m'][np.ix_(ys,xs)].astype(float);valid=row['depth_valid'][np.ix_(ys,xs)]
        quantized=np.zeros(valid.shape,dtype='<u4');quantized[valid]=np.rint(sampled[valid]/.0001).astype('<u4')
        expected.append((quantized,valid));w.append(**row)
    w.close();encoded=list(numeric_records(w.output/'numeric.bin'))
    assert [i for i,row in enumerate(encoded) if bool(row['depth_keyframe'])]==[0,20]
    for i,row in enumerate(encoded):
        assert row['depth_code'].dtype==np.dtype('<u4') and 'depth_q' not in row
        code=expected[i][0] if i%20==0 else np.bitwise_xor(expected[i][0],expected[i-1][0])
        np.testing.assert_array_equal(row['depth_code'],code)
    for (quantized,valid),decoded in zip(expected,module().iter_training_records(w.output)):
        np.testing.assert_array_equal(decoded['depth_q'],quantized)
        np.testing.assert_array_equal(decoded['depth_valid'],valid)
    manifest=json.loads((w.output/'manifest.json').read_text())
    assert manifest['depth_codec']=='uint32_xor_previous_with_periodic_absolute_keyframes'
    assert manifest['depth_keyframe_interval']==20
    assert manifest['schema'].endswith('_v3')


@pytest.mark.parametrize('problem',['first_delta','extra_keyframe','flag_dtype','code_dtype','missing_valid'])
def test_depth_reader_rejects_corrupted_temporal_contract(tmp_path,problem):
    w=writer(tmp_path);w.append(**frame());w.append(**frame(1));w.close()
    records=list(numeric_records(w.output/'numeric.bin'))
    if problem=='first_delta':records[0]['depth_keyframe']=np.bool_(False)
    elif problem=='extra_keyframe':records[1]['depth_keyframe']=np.bool_(True)
    elif problem=='flag_dtype':records[0]['depth_keyframe']=np.int64(1)
    elif problem=='code_dtype':records[0]['depth_code']=records[0]['depth_code'].astype('<i8')
    else:del records[0]['depth_valid']
    rewrite_numeric_records(w.output,records)
    with pytest.raises(ValueError,match='depth'):list(module().iter_training_records(w.output))


def test_caller_mutating_yielded_depth_cannot_change_later_delta_decoding(tmp_path):
    w=writer(tmp_path);w.append(**frame());w.append(**frame(1));w.close()
    records=module().iter_training_records(w.output)
    first=next(records);reference=first['depth_q'].copy();first['depth_q'][:]=0
    np.testing.assert_array_equal(next(records)['depth_q'],reference)


def test_failed_append_does_not_advance_temporal_depth_state(tmp_path):
    w=writer(tmp_path,expected_frames=2,limit_bytes=50_000);w.append(**frame())
    previous=w._previous_depth.copy()
    row=frame(1);row['depth_m']=np.random.default_rng(7).uniform(.2,18,(360,640)).astype(np.float32)
    row['depth_valid'][:]=True
    with pytest.raises(ValueError,match='byte budget'):w.append(**row)
    np.testing.assert_array_equal(w._previous_depth,previous)
    assert w._count==1
    with pytest.raises(RuntimeError,match='incomplete'):w.close()
