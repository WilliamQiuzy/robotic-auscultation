"""Pure CPU Hall acquisition: real calibration mapping, no force-label input."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

from sim.ausc_policy import m2_sensors as sensors
from sim.ausc_policy.geometry_epoch import GEOMETRY_ID, GEOMETRY_SOURCE_SHA256
from sim.auscultation_vla_v2.hall import HallPacket, HallCalibration

CALIBRATION = (Path(__file__).resolve().parents[1] /
               'sim/auscultation_vla_v2/calibration/veo6_tared_snapshot.json').read_bytes()
GEOMETRY = dict(geometry_id=GEOMETRY_ID, geometry_source_sha256=GEOMETRY_SOURCE_SHA256,
                freshly_generated=True)


def stream(tmp_path, **kwargs):
    return sensors.M2HallStream(tmp_path/'sensors', geometry_metadata=GEOMETRY,
        calibration_bytes=CALIBRATION, episode_id='m2-test', **kwargs)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_real_calibration_240hz_updates_preserve_100hz_packets_and_references(tmp_path):
    acquisition=stream(tmp_path)
    outputs=[acquisition.append(i,(i+1)/240,i*.00008) for i in range(25)]
    assert len(outputs)==25
    new=[out for out in outputs if out['is_new_packet']]
    assert len(new)==11
    assert outputs[0]['packet_id']==outputs[1]['packet_id']==outputs[2]['packet_id']
    assert outputs[3]['packet_id']!=outputs[2]['packet_id']
    assert outputs[1]['packet']==outputs[0]['packet']
    assert outputs[1]['packet']['sample_time_s']==1/240
    cal=HallCalibration(CALIBRATION)
    for out in outputs:
        packet=HallPacket(**out['packet'])
        assert out['decoded']==asdict(cal.decode(packet,now_s=out['time_s']))
        assert out['training_eligible'] is False
    physics=rows(tmp_path/'sensors/physics.jsonl')
    hall=rows(tmp_path/'sensors/hall.jsonl')
    assert len(physics)==len(hall)==25
    assert hall[1]['packet'] is None and hall[1]['packet_id']==hall[0]['packet_id']
    assert physics[-1]['compression_m']==pytest.approx(.00192)
    snapshot=acquisition.snapshot()
    assert snapshot['physics_hz']==240 and snapshot['adc_nominal_hz']==100
    assert snapshot['parameters']['sample_period_s']==.01
    assert not snapshot['bench_verified']
    receipt=acquisition.finalize()
    assert receipt['completed'] and not receipt['training_eligible']
    audit=sensors.audit_hall_stream(tmp_path/'sensors')
    assert audit['passed'] and audit['physics_rows']==25 and audit['unique_packets']==11
    assert not audit['training_eligible']


@pytest.mark.parametrize('row,stamp,compression', [
    (1,1/240,0.),(0,0.,0.),(0,float('nan'),0.),(0,1/240,float('inf')),
    (True,1/240,0.),(0,True,0.),(0,1/240,False)])
def test_invalid_clock_or_values_are_rejected_and_never_accepted(tmp_path,row,stamp,compression):
    acquisition=stream(tmp_path)
    with pytest.raises(ValueError):acquisition.append(row,stamp,compression)
    receipt=acquisition.finalize()
    assert not receipt['completed'] and receipt['failure']
    assert not rows(tmp_path/'sensors/physics.jsonl')
    assert not receipt['training_eligible']


@pytest.mark.parametrize('next_row,next_stamp',[(1,1/240),(2,3/240),(1,3/240)])
def test_duplicate_missing_or_discontinuous_steps_preserve_the_written_original(tmp_path,next_row,next_stamp):
    acquisition=stream(tmp_path)
    acquisition.append(0,1/240,0.)
    original=(tmp_path/'sensors/physics.jsonl').read_bytes()
    with pytest.raises(ValueError):acquisition.append(next_row,next_stamp,0.)
    assert (tmp_path/'sensors/physics.jsonl').read_bytes()==original
    with pytest.raises(RuntimeError,match='failed'):acquisition.append(1,2/240,0.)
    assert not acquisition.finalize()['completed']


def test_out_of_calibration_compression_keeps_raw_physics_and_hall_error(tmp_path):
    acquisition=stream(tmp_path)
    acquisition.append(0,1/240,.001)
    with pytest.raises(ValueError,match='travel'):acquisition.append(1,2/240,-.000001)
    physics=rows(tmp_path/'sensors/physics.jsonl');hall=rows(tmp_path/'sensors/hall.jsonl')
    assert physics[-1]['compression_m']==-.000001
    assert hall[-1]['error'] and hall[-1]['physics_row']==1
    assert len(physics)==len(hall)==2
    assert not acquisition.finalize()['completed']


def test_injected_simulator_wrong_calibration_keeps_raw_packet_before_decode_fails(tmp_path):
    class WrongSHA:
        def __init__(self,calibration,**parameters):pass
        def update(self,time_s,compression_m,**kwargs):self.time=time_s
        def read(self,now_s):return HallPacket(31577,self.time,self.time,'0'*64,'simulated_adc')
    acquisition=stream(tmp_path,simulator_factory=WrongSHA)
    with pytest.raises(ValueError,match='identity'):acquisition.append(0,1/240,0.)
    hall=rows(tmp_path/'sensors/hall.jsonl')
    assert hall[0]['packet']['calibration_sha256']=='0'*64
    assert hall[0]['error'] and hall[0]['decoded'] is None
    assert not acquisition.finalize()['completed']


def test_latency_pending_packets_and_seed_are_explicit(tmp_path):
    acquisition=stream(tmp_path,condition=dict(latency_s=.02,noise_std_v=.0003,seed=27))
    first=acquisition.append(0,1/240,0.)
    assert first['status']=='pending' and first['packet'] is None
    for i in range(1,20):acquisition.append(i,(i+1)/240,.001)
    snapshot=acquisition.snapshot()
    assert snapshot['parameters']['seed']==27
    assert snapshot['parameters']['latency_s']==.02
    acquisition.finalize()
    assert sensors.audit_hall_stream(tmp_path/'sensors')['passed']


def test_geometry_calibration_and_unknown_parameters_fail_before_output(tmp_path):
    with pytest.raises(ValueError):
        sensors.M2HallStream(tmp_path/'old',geometry_metadata={**GEOMETRY,'geometry_id':'old1262'},
            calibration_bytes=CALIBRATION,episode_id='bad')
    assert not (tmp_path/'old').exists()
    class WrongCalibration(HallCalibration):
        def __init__(self,data):super().__init__(data);self.sha256='0'*64
    with pytest.raises(ValueError,match='calibration'):
        stream(tmp_path,calibration_factory=WrongCalibration)
    assert not (tmp_path/'sensors').exists()
    with pytest.raises(ValueError):stream(tmp_path,condition={'sample_period_s':1/240})
    assert not (tmp_path/'sensors').exists()


def test_readonly_replay_refuses_modified_adc_bytes(tmp_path):
    acquisition=stream(tmp_path)
    for i in range(15):acquisition.append(i,(i+1)/240,.001)
    acquisition.finalize()
    path=tmp_path/'sensors/hall.jsonl'
    original=rows(path);original[0]['packet']['raw']+=1
    path.write_text(''.join(json.dumps(row)+'\n' for row in original))
    with pytest.raises(ValueError,match='changed'):sensors.audit_hall_stream(tmp_path/'sensors')


def test_no_samples_cannot_become_a_completed_stream(tmp_path):
    acquisition=stream(tmp_path)
    receipt=acquisition.finalize()
    assert not receipt['completed']
    with pytest.raises(ValueError,match='incomplete'):sensors.audit_hall_stream(tmp_path/'sensors')


def test_replay_reconstructs_adc_not_only_file_hashes(tmp_path):
    import hashlib
    acquisition=stream(tmp_path)
    for i in range(15):acquisition.append(i,(i+1)/240,.001)
    acquisition.finalize()
    path=tmp_path/'sensors/hall.jsonl'
    altered=rows(path);altered[0]['packet']['raw']+=1
    path.write_text(''.join(json.dumps(row)+'\n' for row in altered))
    receipt_path=tmp_path/'sensors/receipt.json'
    receipt=json.loads(receipt_path.read_text())
    receipt['artifacts_sha256']['hall.jsonl']=hashlib.sha256(path.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='replay'):sensors.audit_hall_stream(tmp_path/'sensors')


def test_replay_binds_source_bytes_across_hosts_without_requiring_identical_paths(tmp_path):
    import hashlib
    acquisition=stream(tmp_path)
    for i in range(4):acquisition.append(i,(i+1)/240,0.)
    acquisition.finalize()
    manifest_path=tmp_path/'sensors/manifest.json'
    manifest=json.loads(manifest_path.read_text())
    for record in manifest['source_bindings'].values():record['path']='/remote/checkout/'+record['module']+'.py'
    manifest_path.write_text(json.dumps(manifest))
    receipt_path=tmp_path/'sensors/receipt.json';receipt=json.loads(receipt_path.read_text())
    receipt['artifacts_sha256']['manifest.json']=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt))
    assert sensors.audit_hall_stream(tmp_path/'sensors')['passed']


def test_raw_adc_is_flushed_before_injected_decoder_runs_or_raises(tmp_path):
    class FailingDecoder(HallCalibration):
        def decode(self,packet,**kwargs):
            acquired=rows(tmp_path/'sensors/adc_acquisition.jsonl')
            assert len(acquired)==1
            assert acquired[0]['packet']==asdict(packet)
            assert acquired[0]['acquisition_id']=='adc00000000'
            assert acquired[0]['training_eligible'] is False
            raise ValueError('decoder failed after durable raw acquisition')
    acquisition=stream(tmp_path,calibration_factory=FailingDecoder)
    with pytest.raises(ValueError,match='decoder failed'):acquisition.append(0,1/240,.001)
    hall=rows(tmp_path/'sensors/hall.jsonl')
    assert hall[0]['acquisition_id']=='adc00000000' and hall[0]['error']
    receipt=acquisition.finalize()
    assert not receipt['completed'] and receipt['adc_acquisitions']==1
    assert 'adc_acquisition.jsonl' in receipt['artifacts_sha256']


@pytest.mark.parametrize('invalid_time',[float('nan'),float('inf'),-float('inf')])
def test_nonfinite_packet_clock_is_preserved_as_lossless_invalid_bits(tmp_path,invalid_time):
    import struct
    class BadClock:
        def __init__(self,calibration,**parameters):self.cal=calibration
        def update(self,time_s,compression_m,**kwargs):self.time=time_s
        def read(self,now_s):return HallPacket(31577,invalid_time,self.time,self.cal.sha256,'simulated_adc')
    acquisition=stream(tmp_path,simulator_factory=BadClock)
    with pytest.raises(ValueError,match='clock'):acquisition.append(0,1/240,.001)
    acquired=rows(tmp_path/'sensors/adc_acquisition.jsonl')
    tagged=acquired[0]['packet']['sample_time_s']
    assert tagged['invalid_type']=='nonfinite_float'
    assert tagged['ieee754_binary64_be']==struct.pack('>d',invalid_time).hex()
    assert acquired[0]['valid_json_packet'] is False
    hall=rows(tmp_path/'sensors/hall.jsonl')
    assert hall[0]['error'] and hall[0]['decoded'] is None
    assert hall[0]['acquisition_id']==acquired[0]['acquisition_id']
    assert hall[0]['packet']['sample_time_s']==tagged
    assert len(rows(tmp_path/'sensors/physics.jsonl'))==1
    assert not acquisition.finalize()['completed']


def test_decoder_termination_leaves_raw_packet_and_cannot_finalize_complete(tmp_path):
    class TerminatingDecoder(HallCalibration):
        def decode(self,packet,**kwargs):
            assert rows(tmp_path/'sensors/adc_acquisition.jsonl')[0]['packet']==asdict(packet)
            raise SystemExit('decoder terminated')
    acquisition=stream(tmp_path,calibration_factory=TerminatingDecoder)
    with pytest.raises(SystemExit):acquisition.append(0,1/240,.001)
    receipt=acquisition.finalize()
    assert not receipt['completed']
    assert receipt['adc_acquisitions']==receipt['physics_rows']==1
    assert receipt['hall_rows']==0
    assert receipt['failure']['stage']=='incomplete_ledgers'


def test_large_ledger_hash_is_read_in_bounded_chunks(tmp_path,monkeypatch):
    import hashlib
    path=tmp_path/'large_hall.jsonl'
    block=(b'{"raw":31577,"time_s":0.05}\n'*2500)
    expected=hashlib.sha256()
    with path.open('wb') as ledger:
        for _ in range(129):ledger.write(block);expected.update(block)
    assert path.stat().st_size>8*1024*1024
    real_open=Path.open
    reads=[]
    class BoundedReader:
        def __init__(self,handle):self.handle=handle
        def __enter__(self):return self
        def __exit__(self,*args):return self.handle.__exit__(*args)
        def read(self,size=-1):
            assert 0<size<=1024*1024, 'unbounded or oversized ledger read'
            reads.append(size)
            return self.handle.read(size)
    def bounded_open(self,*args,**kwargs):
        handle=real_open(self,*args,**kwargs)
        return BoundedReader(handle) if self==path else handle
    monkeypatch.setattr(Path,'open',bounded_open)
    assert sensors._file_sha256(path)==expected.hexdigest()
    assert len(reads)>8


def test_finalization_and_replay_never_read_entire_ledgers_or_source_files(tmp_path,monkeypatch):
    original=Path.read_bytes
    def guarded_read_bytes(self):
        assert self.suffix not in {'.jsonl','.py'}, 'full ledger/source read is forbidden'
        return original(self)
    monkeypatch.setattr(Path,'read_bytes',guarded_read_bytes)
    acquisition=stream(tmp_path)
    for i in range(30):acquisition.append(i,(i+1)/240,.001)
    assert acquisition.finalize()['completed']
    assert sensors.audit_hall_stream(tmp_path/'sensors')['passed']


def test_tagged_numpy_clock_is_archived_but_never_decoded_or_completed(tmp_path):
    calls=[]
    class Decoder(HallCalibration):
        def decode(self,packet,**kwargs):
            calls.append(packet)
            return super().decode(packet,**kwargs)
    class NumpyClock:
        def __init__(self,calibration,**parameters):self.cal=calibration
        def update(self,time_s,compression_m,**kwargs):self.time=np.float64(time_s)
        def read(self,now_s):return HallPacket(31577,self.time,self.time,self.cal.sha256,'simulated_adc')
    acquisition=stream(tmp_path,calibration_factory=Decoder,simulator_factory=NumpyClock)
    with pytest.raises(ValueError,match='representation'):acquisition.append(0,1/240,.001)
    acquired=rows(tmp_path/'sensors/adc_acquisition.jsonl')[0]
    assert acquired['valid_json_packet'] is False
    assert acquired['packet']['sample_time_s']['invalid_type']=='numpy_scalar'
    assert not calls
    hall=rows(tmp_path/'sensors/hall.jsonl')[0]
    assert hall['status']=='error' and hall['decoded'] is None
    assert not acquisition.finalize()['completed']


def test_same_numeric_packet_with_invalid_numpy_type_gets_its_own_raw_evidence(tmp_path):
    class ChangedType:
        def __init__(self,calibration,**parameters):self.cal=calibration;self.count=0
        def update(self,time_s,compression_m,**kwargs):self.count+=1
        def read(self,now_s):
            stamp=1/240 if self.count==1 else np.float64(1/240)
            return HallPacket(31577,stamp,stamp,self.cal.sha256,'simulated_adc')
    acquisition=stream(tmp_path,simulator_factory=ChangedType)
    first=acquisition.append(0,1/240,.001)
    assert first['status']=='available'
    with pytest.raises(ValueError,match='representation'):acquisition.append(1,2/240,.001)
    acquired=rows(tmp_path/'sensors/adc_acquisition.jsonl')
    assert len(acquired)==2
    assert acquired[0]['valid_json_packet'] is True and acquired[1]['valid_json_packet'] is False
    assert acquired[0]['acquisition_id']!=acquired[1]['acquisition_id']
    assert acquired[1]['packet']['sample_time_s']['invalid_type']=='numpy_scalar'
    hall=rows(tmp_path/'sensors/hall.jsonl')
    assert hall[1]['acquisition_id']==acquired[1]['acquisition_id']
    assert hall[1]['packet']==acquired[1]['packet']
    assert hall[1]['status']=='error' and hall[1]['decoded'] is None
    assert not acquisition.finalize()['completed']


def test_latest_evidence_binds_full_return_and_raw_physics_without_aliases(tmp_path):
    import hashlib
    acquisition=stream(tmp_path)
    assert hasattr(acquisition,'latest_evidence'), 'Missing live Hall evidence accessor'
    with pytest.raises(RuntimeError): acquisition.latest_evidence()
    first=acquisition.append(0,1/240,.001)
    evidence=acquisition.latest_evidence()
    assert evidence['physics']==dict(physics_row=0,time_s=1/240,compression_m=.001,training_eligible=False)
    assert evidence['hall']==first and evidence['snapshot']==acquisition.snapshot()
    assert evidence['manifest_sha256']==hashlib.sha256((tmp_path/'sensors/manifest.json').read_bytes()).hexdigest()
    first['packet']['raw']=1
    evidence['physics']['compression_m']=99.
    evidence['hall']['packet']['raw']=2
    evidence['snapshot']['episode_id']='other'
    fresh=acquisition.latest_evidence()
    assert fresh['physics']['compression_m']==.001 and fresh['hall']['packet']['raw'] not in (1,2)
    assert fresh['snapshot']['episode_id']=='m2-test'
    second=acquisition.append(1,2/240,.002)
    assert not second['is_new_packet'] and second['packet'] is not None
    assert acquisition.latest_evidence()['hall']==second
    assert acquisition.latest_evidence()['physics']['compression_m']==.002
    acquisition.finalize()
    with pytest.raises(RuntimeError): acquisition.latest_evidence()


@pytest.mark.parametrize('failure',['append','incomplete','manifest'])
def test_latest_evidence_refuses_failed_incomplete_or_changed_stream(tmp_path,failure):
    acquisition=stream(tmp_path)
    assert hasattr(acquisition,'latest_evidence'), 'Missing live Hall evidence accessor'
    acquisition.append(0,1/240,.001)
    if failure=='append':
        with pytest.raises(ValueError): acquisition.append(2,3/240,.001)
    elif failure=='incomplete': acquisition._physics_rows+=1
    else: (tmp_path/'sensors/manifest.json').write_text('{}\n')
    with pytest.raises(RuntimeError): acquisition.latest_evidence()
    acquisition.finalize()
