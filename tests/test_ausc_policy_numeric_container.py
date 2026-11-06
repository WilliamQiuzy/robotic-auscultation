"""Lossless numeric-container CPU fixtures; never native qualification."""
import gzip
import hashlib
import io
import json

import numpy as np
import pytest

from sim.ausc_policy import expert_capture, training_capture as capture
from test_ausc_policy_training_capture import (
    adaptive_writer, frame, rebind_artifacts, terminal_frame, terminal_ticks, writer,
)
from test_ausc_policy_compact_expert_capture import compact_binding, make, tick


CODEC = 'framed_npz_gzip_single_member_v1'


def record(tmp_path, *, codec=CODEC, frames=3, remainder=0):
    options = {} if codec is None else dict(numeric_container_codec=codec)
    w = adaptive_writer(tmp_path, max_frames=max(1, frames), **options)
    for i in range(frames):
        if i == frames-1: w.append_terminal(**terminal_frame(i))
        else: w.append(**frame(i))
    n = frames*12+remainder
    w.close(final_native_step_count=n, final_native_time_s=n/240., terminal_ticks=terminal_ticks(n))
    return w.output


def replace_numeric(output, physical, *, logical=None):
    (output/'numeric.bin').write_bytes(physical)
    if logical is not None:
        path = output/'receipt.json'
        value = json.loads(path.read_text())
        value['numeric_container'].update(uncompressed_bytes=len(logical),
            uncompressed_sha256=hashlib.sha256(logical).hexdigest())
        path.write_bytes(capture._json_bytes(value))
    rebind_artifacts(output, ['numeric.bin'])


@pytest.mark.parametrize('frames,remainder', [(0,0),(0,1),(0,11),(1,0),(2,1),(21,11)])
def test_gzip_is_exact_deterministic_legacy_framing_with_native_terminal(tmp_path, frames, remainder):
    legacy = record(tmp_path/'legacy', codec=None, frames=frames, remainder=remainder)
    first = record(tmp_path/'first', frames=frames, remainder=remainder)
    second = record(tmp_path/'second', frames=frames, remainder=remainder)
    logical = (legacy/'numeric.bin').read_bytes()
    physical = (first/'numeric.bin').read_bytes()
    assert capture.NUMERIC_CONTAINER_CODEC == CODEC
    assert physical == (second/'numeric.bin').read_bytes()
    assert gzip.decompress(physical) == logical
    assert physical[:10] == b'\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\x03'
    assert (first/'rgb.mjpeg').read_bytes() == (legacy/'rgb.mjpeg').read_bytes()
    assert (first/'terminal.json').read_bytes() == (legacy/'terminal.json').read_bytes()
    expected = list(capture.iter_training_records(legacy))
    actual = list(capture.iter_training_records(first))
    assert len(actual) == len(expected) == frames
    for a, b in zip(actual, expected):
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
            assert np.asarray(a[key]).dtype == np.asarray(b[key]).dtype
    receipt = json.loads((first/'receipt.json').read_text())
    assert receipt['numeric_container'] == dict(codec=CODEC, uncompressed_bytes=len(logical),
        uncompressed_sha256=hashlib.sha256(logical).hexdigest(), frame_count=frames)
    assert receipt['artifacts']['numeric.bin']['bytes'] == len(physical)
    assert receipt['total_bytes'] == sum(p.stat().st_size for p in first.iterdir())
    assert 'numeric_container_codec' not in json.loads((legacy/'manifest.json').read_text())
    assert 'numeric_container' not in json.loads((legacy/'receipt.json').read_text())


def test_fixed_recording_and_explicit_none_keep_legacy_unchanged(tmp_path):
    old = writer(tmp_path/'old', expected_frames=1)
    explicit = writer(tmp_path/'explicit', expected_frames=1, numeric_container_codec=None)
    new = writer(tmp_path/'new', expected_frames=1, numeric_container_codec=CODEC)
    for w in (old, explicit, new): w.append(**frame()); w.close()
    for name in ('manifest.json','numeric.bin','receipt.json'):
        assert (old.output/name).read_bytes() == (explicit.output/name).read_bytes()
    assert gzip.decompress((new.output/'numeric.bin').read_bytes()) == (old.output/'numeric.bin').read_bytes()
    assert len(list(capture.iter_training_records(new.output))) == 1


@pytest.mark.parametrize('codec', ['', 'gzip', 'framed_npz_gzip_single_member_v2', True, {}])
def test_unknown_writer_codec_rejected_before_output(tmp_path, codec):
    with pytest.raises(ValueError, match='codec|container'):
        writer(tmp_path, numeric_container_codec=codec)
    assert not (tmp_path/'training_capture').exists()


@pytest.mark.parametrize('field', ['manifest', 'receipt', 'missing', 'legacy_receipt'])
def test_unknown_or_mismatched_container_declaration_rejected(tmp_path, field):
    output = record(tmp_path)
    manifest = json.loads((output/'manifest.json').read_text())
    receipt = json.loads((output/'receipt.json').read_text())
    if field == 'manifest': manifest['numeric_container_codec'] = 'unknown_v2'
    elif field == 'receipt': receipt['numeric_container']['codec'] = 'unknown_v2'
    elif field == 'missing': del receipt['numeric_container']
    else: del manifest['numeric_container_codec']
    (output/'manifest.json').write_bytes(capture._json_bytes(manifest))
    (output/'receipt.json').write_bytes(capture._json_bytes(receipt))
    rebind_artifacts(output, ['manifest.json'])
    with pytest.raises(ValueError, match='integrity'):
        list(capture.iter_training_records(output))


@pytest.mark.parametrize('problem', ['second_member','empty_member','suffix','prefix','trailer_crc','trailer_size',
    'deflate_corruption','bad_header','wrong_sha','wrong_size','wrong_count','boolean_count','impossible_size'])
def test_rebound_container_rejects_malformed_stream_and_metadata(tmp_path, problem):
    output = record(tmp_path)
    raw = (output/'numeric.bin').read_bytes()
    path = output/'receipt.json'
    receipt = json.loads(path.read_text())
    if problem == 'second_member': raw += gzip.compress(b'extra', mtime=0)
    elif problem == 'empty_member': raw += gzip.compress(b'', mtime=0)
    elif problem == 'suffix': raw += b'EXTRA'
    elif problem == 'prefix': raw = b'EXTRA'+raw
    elif problem in ('trailer_crc', 'trailer_size'):
        at = -8 if problem == 'trailer_crc' else -4
        raw = bytearray(raw); raw[at] ^= 1; raw = bytes(raw)
    elif problem == 'deflate_corruption':
        raw = bytearray(raw); raw[len(raw)//2] ^= 128; raw = bytes(raw)
    elif problem == 'bad_header': raw = b'\x00'+raw[1:]
    elif problem == 'wrong_sha': receipt['numeric_container']['uncompressed_sha256'] = '0'*64
    elif problem == 'wrong_size': receipt['numeric_container']['uncompressed_bytes'] += 1
    elif problem == 'wrong_count': receipt['numeric_container']['frame_count'] += 1
    elif problem == 'boolean_count': receipt['numeric_container']['frame_count'] = True
    else: receipt['numeric_container']['uncompressed_bytes'] = 10**12
    path.write_bytes(capture._json_bytes(receipt))
    replace_numeric(output, raw)
    with pytest.raises(ValueError, match='integrity'):
        list(capture.iter_training_records(output))


@pytest.mark.parametrize('cut', [0,1,2,9,10,11,30,-9,-8,-7,-4,-1])
def test_rebound_truncation_rejected_at_header_payload_and_eof(tmp_path, cut):
    output = record(tmp_path)
    raw = (output/'numeric.bin').read_bytes()
    replace_numeric(output, raw[:cut])
    with pytest.raises(ValueError, match='integrity'):
        list(capture.iter_training_records(output))


@pytest.mark.parametrize('problem', ['oversized_frame','crc','extra_frame','missing_frame','terminal_kind'])
def test_original_frame_bounds_crc_count_and_terminal_checks_survive_recompression(tmp_path, problem):
    output = record(tmp_path, frames=1)
    logical = gzip.decompress((output/'numeric.bin').read_bytes())
    if problem == 'oversized_frame':
        # The declared decoded byte count remains within the original bound;
        # the original frame header itself must reject this promised length.
        logical = capture.HEADER.pack(capture.MAX_RECORD_BYTES+1, 0)+b'0'*capture.MAX_RECORD_BYTES
    elif problem == 'crc': logical = logical[:4]+b'\x00'*4+logical[8:]
    elif problem == 'extra_frame': logical += logical
    elif problem == 'missing_frame': logical = b''
    else:
        values = capture._read_numeric_payload(logical[8:])
        values['record_kind'] = np.array('paired')
        stream = io.BytesIO(); np.savez_compressed(stream, **values)
        logical = capture._framed(stream.getvalue())
    replace_numeric(output, gzip.compress(logical, mtime=0), logical=logical)
    with pytest.raises(ValueError, match='integrity'):
        list(capture.iter_training_records(output))


def test_expansion_bomb_stops_after_original_records_without_allocating_expansion(tmp_path, monkeypatch):
    output = record(tmp_path, frames=1)
    logical = gzip.decompress((output/'numeric.bin').read_bytes())
    # Bind the valid original count/size/hash while presenting a 32 MiB tail.
    replace_numeric(output, gzip.compress(logical+b'0'*(32*1024*1024), mtime=0))
    factory = capture.zlib.decompressobj
    reads = []
    class GuardedDecoder:
        def __init__(self, *args): self.decoder = factory(*args)
        def decompress(self, data, max_length=0):
            assert 0 < max_length <= capture.MAX_RECORD_BYTES
            result = self.decoder.decompress(data, max_length)
            reads.append(len(result))
            return result
        def __getattr__(self, name): return getattr(self.decoder, name)
    # Only instrument the outer decoder; NPZ member decoding is also bounded
    # but uses the original function through the selected wbits dispatch.
    monkeypatch.setattr(capture.zlib, 'decompressobj', lambda bits=15:
        GuardedDecoder(bits) if bits == 31 else factory(bits))
    with pytest.raises(ValueError, match='integrity.*expansion'):
        list(capture.iter_training_records(output))
    assert sum(reads) == len(logical)+1


def test_incompressible_record_crosses_incremental_input_blocks(tmp_path):
    row = terminal_frame(); rng = np.random.default_rng(902)
    row['depth_m'] = rng.uniform(.2,4000,(360,640)).astype(np.float32)
    row['depth_valid'][:] = True
    outputs = []
    for name, codec in [('old',None),('new',CODEC)]:
        w = adaptive_writer(tmp_path/name, numeric_container_codec=codec)
        w.append_terminal(**row)
        w.close(final_native_step_count=12, final_native_time_s=.05, terminal_ticks=[])
        outputs.append(w.output)
    assert (outputs[1]/'numeric.bin').stat().st_size > 64*1024
    assert gzip.decompress((outputs[1]/'numeric.bin').read_bytes()) == (outputs[0]/'numeric.bin').read_bytes()
    old = list(capture.iter_training_records(outputs[0]))[0]
    new = list(capture.iter_training_records(outputs[1]))[0]
    np.testing.assert_array_equal(new['depth_q'], old['depth_q'])


def test_physical_budget_uses_compressed_bytes_and_overflow_aborts(tmp_path):
    reference = record(tmp_path/'reference', frames=24)
    limit = sum(p.stat().st_size for p in reference.iterdir())+capture.RECEIPT_RESERVE+2000
    new = adaptive_writer(tmp_path/'fits', max_frames=24, numeric_container_codec=CODEC, limit_bytes=limit)
    old = adaptive_writer(tmp_path/'legacy', max_frames=24, limit_bytes=limit)
    for i in range(23): new.append(**frame(i))
    new.append_terminal(**terminal_frame(23))
    receipt = new.close(final_native_step_count=288, final_native_time_s=1.2, terminal_ticks=[])
    assert receipt['completed'] and receipt['total_bytes'] <= limit
    with pytest.raises(ValueError, match='byte budget'):
        for i in range(24): old.append(**frame(i))
    old.abort('expected legacy budget failure')
    small = writer(tmp_path/'overflow', expected_frames=1, numeric_container_codec=CODEC, limit_bytes=30_000)
    row = frame(); rng = np.random.default_rng(13)
    row['rgb'] = rng.integers(0,256,(360,640,3),dtype=np.uint8)
    row['depth_m'] = rng.uniform(.2,5,(360,640)).astype(np.float32); row['depth_valid'][:] = True
    with pytest.raises(ValueError, match='byte budget'): small.append(**row)
    failed = small.abort('expected budget failure')
    assert not failed['completed'] and failed['frames'] == 0 and failed['total_bytes'] <= 30_000
    assert gzip.decompress((small.output/'numeric.bin').read_bytes()) == b''


def test_abort_finalizes_single_member_and_preserves_actual_native_suffix(tmp_path):
    w = adaptive_writer(tmp_path, numeric_container_codec=CODEC)
    w.append(**frame())
    receipt = w.abort('injected interruption', final_native_step_count=13,
        final_native_time_s=13/240., terminal_ticks=terminal_ticks(13))
    assert not receipt['completed'] and receipt['frames'] == 1
    assert receipt['final_native_step_count'] == 13 and receipt['terminal_tick_count'] == 1
    logical = gzip.decompress((w.output/'numeric.bin').read_bytes())
    assert receipt['numeric_container']['uncompressed_bytes'] == len(logical)
    assert receipt['total_bytes'] == sum(p.stat().st_size for p in w.output.iterdir())
    assert w.abort('second abort') == receipt
    with pytest.raises(RuntimeError, match='incomplete'): list(capture.iter_training_records(w.output))


@pytest.mark.parametrize('stage', ['record','footer'])
def test_interrupted_compressed_record_or_footer_cannot_publish_success(tmp_path, stage):
    w = writer(tmp_path, expected_frames=1, numeric_container_codec=CODEC)
    if stage == 'footer': w.append(**frame())
    underlying = w._numeric
    class InterruptedWrite:
        def write(self, payload):
            underlying.write(payload[:3]); underlying.flush()
            raise OSError('injected interrupted '+stage)
        def flush(self): underlying.flush()
        def close(self): underlying.close()
    w._numeric = InterruptedWrite()
    if stage == 'record':
        with pytest.raises(OSError, match='injected'): w.append(**frame())
    with pytest.raises(RuntimeError, match='incomplete'): w.close()
    receipt = json.loads((w.output/'receipt.json').read_text())
    assert not receipt['completed'] and receipt['frames'] == int(stage == 'footer')
    assert receipt['total_bytes'] == sum(p.stat().st_size for p in w.output.iterdir())
    with pytest.raises(RuntimeError, match='incomplete'): list(capture.iter_training_records(w.output))


def test_expert_settings_validate_and_forward_only_explicit_codec(tmp_path):
    value = compact_binding(numeric_container_codec=CODEC)
    assert expert_capture.capture_settings({'raw_multimodal_capture':value})['compact']['numeric_container_codec'] == CODEC
    env, w = make(tmp_path, numeric_container_codec=CODEC)
    for _ in range(25): tick(env, w)
    w.close()
    assert w.result(25, execution_completed=True)['compact_capture_complete']
    assert len(list(capture.iter_training_records(w.output/'compact'))) == 2
    assert json.loads((w.output/'compact/manifest.json').read_text())['numeric_container_codec'] == CODEC
    value['compact']['numeric_container_codec'] = 'unknown_v2'
    with pytest.raises(ValueError, match='codec|container'):
        expert_capture.capture_settings({'raw_multimodal_capture':value})
