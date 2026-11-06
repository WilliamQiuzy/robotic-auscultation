"""Lossless native transport; no policy, native physics, or acceptance changes."""
import gzip
import json
import lzma

import pytest

from sim.ausc_policy import m2_given_contact_evaluation as neutral
from sim.ausc_policy import m2_given_contact_pilot_native_audit as audit
from sim.ausc_policy import m2_given_contact_runtime as rt
from sim.ausc_policy import m2_rebuild_paired_eval as paired


def rows():
    result=[]
    for i in range(2):
        before=dict(physics_row=i,time_s=i/240.,q=[0.]*7,pose_base=rt.np.eye(4).tolist())
        after=dict(before,physics_row=i+1,time_s=(i+1)/240.)
        native=dict(time_s=after['time_s'],q=[0.]*7,q_command=[0.]*7,physical_violations=[])
        result.append(dict(schema='m2_given_contact_neutral_native_tick_v1',physics_row=i,time_s=after['time_s'],
            observed_before=before,observed_after=after,native_row=native,q_command=[0.]*7,
            state_audit='completed',physical_violations=[],tick_error=None))
    return result


def receipt(tmp_path,suffix='.xz',events=None,payload=None):
    events=rows() if events is None else events
    data=(''.join(json.dumps(row,sort_keys=True)+'\n' for row in events)).encode() if payload is None else payload
    path=tmp_path/('native.jsonl'+suffix)
    path.write_bytes(lzma.compress(data,format=lzma.FORMAT_XZ,preset=3) if suffix=='.xz' else
                     gzip.compress(data) if suffix=='.gz' else data)
    value=dict(schema='m2_given_contact_neutral_native_audit_v1',metadata={'source':'fixture'},
        complete=True,errors=[],rows=len(events),trace=rt.reference(path))
    value['index']=rt.write_new(tmp_path/'index.json',value)
    return value,data


@pytest.mark.parametrize('suffix,opener',[('.gz',gzip.open),('.xz',lzma.open)])
def test_buffered_transport_preserves_exact_json_bytes_and_close_rows(tmp_path,suffix,opener):
    path=tmp_path/('audit.jsonl'+suffix);stream=audit.BufferedGzipText(path)
    payload=[json.dumps(dict(index=i,value=-0.0,text=' spaces, \\" quotes ',nested=rows()),sort_keys=True)+'\n' for i in range(241)]
    for line in payload[:239]:stream.write(line);stream.flush()
    assert stream.pending_rows==239 and stream.written_rows==0
    stream.write(payload[239]);stream.flush()
    assert stream.pending_rows==0 and stream.written_rows==240
    stream.write(payload[240]);stream.close()
    assert stream.written_rows==241 and stream.pending_rows==0
    with opener(path,'rb') as incoming:assert incoming.read()==''.join(payload).encode()


@pytest.mark.parametrize('suffix',['.gz','.xz',''])
def test_unchanged_reader_validates_all_rows_in_legacy_and_new_transport(tmp_path,suffix):
    value,_=receipt(tmp_path,suffix)
    assert neutral.read_native_trace(value,value['metadata'])==rows()


def test_xz_hash_corruption_is_rejected_before_decode(tmp_path):
    value,_=receipt(tmp_path);path=rt.bound_path(value['trace']);path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='changed|digest|bytes|SHA|hash|Checksum'):
        neutral._read_native_events(value,value['metadata'])


@pytest.mark.parametrize('failure',['truncate','row_count','wrong_codec','terminal_failure'])
def test_xz_keeps_transport_row_coverage_and_failure_gates(tmp_path,failure):
    events=rows()
    if failure=='terminal_failure':
        events[-1]['tick_error']='original physical stop';events[-1]['physical_violations']=['original gate']
        events[-1]['native_row']['physical_violations']=['original gate']
    value,_=receipt(tmp_path,events=events);value.pop('index')
    if failure=='truncate':
        path=rt.bound_path(value['trace']);path.write_bytes(path.read_bytes()[:-10]);value['trace']=rt.reference(path)
    elif failure=='row_count':value['rows']+=1
    elif failure=='wrong_codec':
        path=rt.bound_path(value['trace']);path.write_bytes(gzip.compress(b'{}\n'));value['trace']=rt.reference(path)
    else:value.update(complete=False,errors=[dict(stage='original_tick',physics_row=1,error='original physical stop')])
    value['index']=rt.write_new(tmp_path/'negative-index.json',value)
    if failure=='terminal_failure':
        assert len(neutral._read_native_events(value,value['metadata']))==2
        with pytest.raises(ValueError,match='without errors'):neutral.read_native_trace(value,value['metadata'])
    else:
        with pytest.raises((EOFError,lzma.LZMAError,ValueError)):neutral._read_native_events(value,value['metadata'])


def test_only_new_matched_adapter_requests_xz_legacy_stays_gzip(tmp_path,monkeypatch):
    from test_ausc_given_contact_pilot_native_audit import setup
    _,legacy,_,_,_,_=setup(tmp_path,monkeypatch)
    assert legacy.path.name=='given_control_native.jsonl.gz';legacy.close()
    assert paired.NATIVE_TRACE_FILENAME=='given_control_native.jsonl.xz'


def test_learned_attach_honors_explicit_matched_transport(tmp_path,monkeypatch):
    from test_ausc_given_contact_pilot_native_audit import setup
    from sim.ausc_policy import m2_given_contact_pilot_deployment as pilot
    monkeypatch.setattr(pilot,'NATIVE_TRACE_FILENAME',paired.NATIVE_TRACE_FILENAME,raising=False)
    _,observer,controller,_,_,_=setup(tmp_path,monkeypatch)
    controller._tick([0.]*7,'staging');receipt=observer.close()
    assert receipt['trace']['path'].endswith('.jsonl.xz')
    assert len(neutral.read_native_trace(receipt,controller.metadata))==1


def test_reader_rejects_undeclared_transport_suffix(tmp_path):
    value,_=receipt(tmp_path,'.bin')
    with pytest.raises(ValueError,match='transport'):neutral._read_native_events(value,value['metadata'])


def test_unsupported_buffer_codec_rejected(tmp_path):
    with pytest.raises(ValueError,match='transport'):audit.BufferedGzipText(tmp_path/'audit.zip')
