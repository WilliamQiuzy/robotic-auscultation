"""Read-only audits of CPU-injected raw ledgers, never qualified task data."""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from sim.ausc_policy.expert_capture import ExpertCapture
from sim.ausc_policy.retainer_contact import ALL_FORCE_GROUPS, CONTACT_SURFACE
from test_ausc_policy_expert_capture import Env, Backend, binding, labels, GEOMETRY, ROOT


def module():
    name = 'sim.ausc_policy.expert_capture_audit'
    assert importlib.util.find_spec(name), 'New explicit prefix/complete audit is missing'
    return importlib.import_module(name)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, default=lambda v: v.tolist(), allow_nan=False) + '\n')


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def tree_hashes(root):
    return {str(p.relative_to(root)): sha(p) for p in root.rglob('*') if p.is_file()}


def episode(path, *, success=True, steps=24):
    review = path / 'output/review'
    geo = deepcopy(GEOMETRY)
    for name, key in [('collision_geometry.json', 'collision_geometry_sha256'),
                      ('collision_geometry.npz', 'collision_arrays_sha256')]:
        artifact = review / 'geometry' / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b'CPU geometry binding fixture: ' + name.encode())
        geo['geometry_artifacts'][key] = sha(artifact)
    geo['contact_surface'] = CONTACT_SURFACE
    env = Env()
    capture = ExpertCapture(review/'raw_multimodal', binding=binding(), geometry_metadata=geo)
    capture.start(env, backend_factory=Backend)
    for i in range(steps):
        env.advance()
        env.rows[-1].update(q_command=[0.]*7, tcp_world=[0.]*3, axis_world=[0.,0.,1.],
            **{key+'_N': 4. if key == 'retainer' else 0. for key in ALL_FORCE_GROUPS},
            contact_surface=CONTACT_SURFACE,
            retainer_front_audit=dict(passed=True, force_matches_ledger=True, force_N=4.,
                                      off_front_force_N=0., points=[]))
        capture.after_physics(env, i)
        capture.finish_tick(env, i, labels(), labels('HOLD'))
    capture.close(None if success else 'original controller failure')
    receipt = capture.result(steps, execution_completed=success)
    trace_path = review/'fresh_expert/fresh_expert_trace.jsonl'
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(''.join(json.dumps(row, default=lambda v:v.tolist())+'\n' for row in env.rows))
    result = dict(success=success, physical_validation=success, errors=[], diagnostic_completed=True,
                  raw_capture_complete=receipt['raw_capture_complete'], raw_multimodal_capture=receipt,
                  geometry_id=geo['geometry_id'], geometry_source_sha256=geo['geometry_source_sha256'],
                  geometry_artifacts=geo['geometry_artifacts'], contact_surface=CONTACT_SURFACE,
                  trace_sha256=sha(trace_path), physics_steps=steps, training_eligible=False)
    write(review/'fresh_expert_result.json', result)
    write(review/'fresh_expert_inputs/manifest.json', geo)
    write(path/'run.json', dict(config=dict(fresh_expert=dict(raw_multimodal_capture=binding()))))
    write(path/'result.json', dict(completed=True, gpu_released=True))
    return path


def refresh_stream(run, category, filename):
    root = run/'output/review/raw_multimodal'
    receipt = read(root/category/'receipt.json')
    receipt['artifacts_sha256'][filename] = sha(root/category/filename)
    write(root/category/'receipt.json', receipt)
    top = read(root/'receipt.json'); top['receipts']['camera' if category=='rgbd' else 'hall'] = receipt
    write(root/'receipt.json', top)
    result = read(run/'output/review/fresh_expert_result.json'); result['raw_multimodal_capture'] = top
    write(run/'output/review/fresh_expert_result.json', result)


def test_complete_without_baseline_replays_and_never_mutates_evidence(tmp_path):
    audit = module(); run = episode(tmp_path/'run'); before = tree_hashes(run)
    report = audit.audit_capture(run, mode='complete')
    assert report['consistency_passed'] and report['complete_capture_audited']
    assert report['baseline_comparison'] is None
    assert report['physics_rows']==24 and report['rgbd_frames']==2
    assert not report['training_eligible'] and not report['physical_qualification']
    assert tree_hashes(run)==before


def test_prefix_keeps_failure_flags_and_complete_rejects_original_failure(tmp_path):
    audit = module(); run = episode(tmp_path/'run', success=False)
    before = tree_hashes(run)
    report = audit.audit_capture(run, mode='prefix')
    assert report['consistency_passed'] and not report['complete_capture_audited']
    assert report['original']['raw_capture_complete'] is False
    assert report['original']['hall_receipt']['failure']
    with pytest.raises(ValueError, match='complete'):
        audit.audit_capture(run, mode='complete')
    assert tree_hashes(run)==before


@pytest.mark.parametrize('mutation', ['top_disagrees', 'camera_failed', 'hall_failed', 'top_errors', 'attempt_error'])
def test_complete_rejects_inconsistent_or_failed_original_receipts(tmp_path, mutation):
    audit = module(); run = episode(tmp_path/'run'); raw=run/'output/review/raw_multimodal'
    if mutation=='top_disagrees':
        top=read(raw/'receipt.json');top['raw_capture_complete']=False;write(raw/'receipt.json',top)
    elif mutation in ('camera_failed','hall_failed'):
        category='rgbd' if mutation=='camera_failed' else 'hall'
        receipt=read(raw/category/'receipt.json');receipt['completed']=False
        write(raw/category/'receipt.json',receipt)
        refresh_stream(run,category,'manifest.json')
    elif mutation=='top_errors':
        top=read(raw/'receipt.json');top['errors']=[{'stage':'late','error':'original'}]
        write(raw/'receipt.json',top)
        result=read(run/'output/review/fresh_expert_result.json');result['raw_multimodal_capture']=top
        write(run/'output/review/fresh_expert_result.json',result)
    else:
        path=raw/'rgbd/attempts.jsonl'; data=rows(path);data[-1]['status']='error'
        path.write_text(''.join(json.dumps(r)+'\n' for r in data));refresh_stream(run,'rgbd','attempts.jsonl')
    with pytest.raises(ValueError): audit.audit_capture(run,mode='complete')


@pytest.mark.parametrize('field', ['retainer_N','retainer_front_audit','q_command'])
def test_baseline_detects_new_and_old_physics_differences(tmp_path,field):
    audit=module();run=episode(tmp_path/'run');base=episode(tmp_path/'baseline')
    path=base/'output/review/fresh_expert/fresh_expert_trace.jsonl';data=rows(path)
    if field=='retainer_N':data[3][field]=3.
    elif field=='q_command':data[3][field][0]=.01
    else:data[3][field]['passed']=False
    path.write_text(''.join(json.dumps(r)+'\n' for r in data))
    result=read(base/'output/review/fresh_expert_result.json');result['trace_sha256']=sha(path)
    write(base/'output/review/fresh_expert_result.json',result)
    comparison=audit.audit_capture(run,mode='complete',baseline=base)['baseline_comparison']
    assert not comparison['all_compared_physics_exact']
    assert field in comparison['differing_fields']
    assert len([f for f in comparison['compared_fields'] if f.endswith('_N')])==7


@pytest.mark.parametrize('kind',['geometry','artifact','surface'])
def test_baseline_rejects_unbound_epoch_artifacts_or_contact_surface(tmp_path,kind):
    audit=module();run=episode(tmp_path/'run');base=episode(tmp_path/'baseline')
    path=base/'output/review/fresh_expert_inputs/manifest.json';m=read(path)
    if kind=='geometry':m['geometry_id']='retired'
    elif kind=='artifact':m['geometry_artifacts']['collision_arrays_sha256']='0'*64
    else:m['contact_surface']='membrane_only'
    write(path,m)
    with pytest.raises(ValueError):audit.audit_capture(run,mode='prefix',baseline=base)


def test_prefix_without_one_camera_period_has_no_fake_depth_statistics(tmp_path):
    audit=module();run=episode(tmp_path/'run',success=False,steps=5)
    report=audit.audit_capture(run,mode='prefix')
    assert report['rgbd_frames']==0 and report['depth_valid_fraction_range'] is None


def test_rehashed_adc_tampering_is_detected_by_real_replay(tmp_path):
    audit=module();run=episode(tmp_path/'run');raw=run/'output/review/raw_multimodal'
    path=raw/'hall/adc_acquisition.jsonl';data=rows(path);data[0]['packet']['raw']+=1
    path.write_text(''.join(json.dumps(r)+'\n' for r in data));refresh_stream(run,'hall','adc_acquisition.jsonl')
    with pytest.raises(ValueError,match='ADC|replay'):audit.audit_capture(run,mode='prefix')


def source_snapshot(run,snapshot):
    raw=run/'output/review/raw_multimodal';sources=set()
    for path in [raw/'manifest.json',raw/'hall/manifest.json',raw/'rgbd/manifest.json',raw/'receipt.json']:
        m=read(path)
        sources.update(m.get('runtime_sources_sha256',{}))
        sources.update(s['path'] for s in m.get('source_bindings',{}).values())
    sources.update(str(ROOT/p) for p in [
        'sim/ausc_policy/__init__.py','sim/ausc_policy/geometry_epoch.py','sim/ausc_policy/tool_protrusion.py',
        'sim/auscultation_vla_v2/__init__.py','sim/auscultation_pi05_v6/__init__.py',
        'sim/auscultation_pi05_v6/hall_conditions.py'])
    for source in sources:
        target=snapshot/Path(source).relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    # Recorded absolute Linux paths are provenance, not local filesystem locations.
    for path in run.rglob('*.json'):
        path.write_text(path.read_text().replace(str(ROOT),'/recorded/project'))
    for category in ['hall','rgbd']:
        refresh_stream(run,category,'manifest.json')
    hm=read(raw/'hall/manifest.json');cm=read(raw/'rgbd/manifest.json')
    cm['hall_stream']={'snapshot':hm,'manifest_sha256':sha(raw/'hall/manifest.json')}
    write(raw/'rgbd/manifest.json',cm);refresh_stream(run,'rgbd','manifest.json')
    path=raw/'rgbd/frames.jsonl';data=rows(path)
    for row in data:row['hall_stream_manifest_sha256']=sha(raw/'hall/manifest.json')
    path.write_text(''.join(json.dumps(r)+'\n' for r in data));refresh_stream(run,'rgbd','frames.jsonl')


def test_explicit_source_mapping_replays_snapshot_and_does_not_create_pycache(tmp_path):
    audit=module();run=episode(tmp_path/'run');snapshot=tmp_path/'snapshot'
    source_snapshot(run,snapshot);before=tree_hashes(snapshot)
    report=audit.audit_capture(run,mode='complete',recorded_project_root='/recorded/project',source_snapshot_root=snapshot)
    assert report['complete_capture_audited'] and report['sources']['mapped']
    assert report['hall_replay']['implementation']=='source_snapshot'
    assert tree_hashes(snapshot)==before
    assert not list(snapshot.rglob('__pycache__'))


@pytest.mark.parametrize('fault',['missing_mapping','hash','module','outside_root'])
def test_source_mapping_never_silently_skips_identity_checks(tmp_path,fault):
    audit=module();run=episode(tmp_path/'run');snapshot=tmp_path/'snapshot';source_snapshot(run,snapshot)
    kwargs=dict(recorded_project_root='/recorded/project',source_snapshot_root=snapshot)
    if fault=='missing_mapping':kwargs.pop('source_snapshot_root')
    elif fault=='hash':
        path=snapshot/'sim/ausc_policy/m2_sensors.py';path.write_text(path.read_text()+'\n# changed\n')
    else:
        raw=run/'output/review/raw_multimodal';path=raw/'rgbd/manifest.json';m=read(path)
        m['source_bindings']['capture']['module' if fault=='module' else 'path']='wrong.module' if fault=='module' else '/outside/m2_capture.py'
        write(path,m);refresh_stream(run,'rgbd','manifest.json')
    with pytest.raises(ValueError):audit.audit_capture(run,mode='complete',**kwargs)


def test_mode_is_required_and_only_two_explicit_modes_exist(tmp_path):
    audit=module()
    with pytest.raises(TypeError):audit.audit_capture(tmp_path)
    with pytest.raises(ValueError,match='mode'):audit.audit_capture(tmp_path,mode='auto')


def test_snapshot_replay_rejects_any_import_from_outside_snapshot(tmp_path):
    audit=module();run=episode(tmp_path/'run');empty=tmp_path/'empty-snapshot';empty.mkdir()
    with pytest.raises(ValueError,match='outside.*snapshot'):
        audit._replay_hall(run/'output/review/raw_multimodal/hall',source_root=empty)


@pytest.mark.parametrize('mode', ['prefix', 'complete'])
def test_two_completed_attempts_cannot_claim_the_same_frame(tmp_path, mode):
    audit=module();run=episode(tmp_path/'run');raw=run/'output/review/raw_multimodal'
    path=raw/'rgbd/attempts.jsonl';data=rows(path)
    for key in ('index','raw_file','raw_sha256'):
        data[-1][key]=data[1][key]
    path.write_text(''.join(json.dumps(r)+'\n' for r in data));refresh_stream(run,'rgbd','attempts.jsonl')
    with pytest.raises(ValueError,match='attempt|frame'):
        audit.audit_capture(run,mode=mode)


@pytest.mark.parametrize('field,value', [('physics_row','999'), ('supplied_time_s','999.'),
                                       ('phase',"'CONTACT'"), ('target',"'R4'")])
@pytest.mark.parametrize('change_started', [False, True])
def test_attempt_identity_must_match_start_terminal_and_frame(tmp_path, field, value, change_started):
    audit=module();run=episode(tmp_path/'run');raw=run/'output/review/raw_multimodal'
    path=raw/'rgbd/attempts.jsonl';data=rows(path);data[-1][field]=value
    if change_started:data[-2][field]=value
    path.write_text(''.join(json.dumps(r)+'\n' for r in data));refresh_stream(run,'rgbd','attempts.jsonl')
    with pytest.raises(ValueError,match='attempt|frame'):
        audit.audit_capture(run,mode='complete')


def test_attempts_cannot_swap_otherwise_valid_frame_bindings(tmp_path):
    audit=module();run=episode(tmp_path/'run');raw=run/'output/review/raw_multimodal'
    path=raw/'rgbd/attempts.jsonl';data=rows(path)
    data=[dict(data[2],attempt=0),dict(data[3],attempt=0),dict(data[0],attempt=1),dict(data[1],attempt=1)]
    path.write_text(''.join(json.dumps(r)+'\n' for r in data));refresh_stream(run,'rgbd','attempts.jsonl')
    with pytest.raises(ValueError,match='attempt|frame'):
        audit.audit_capture(run,mode='complete')


@pytest.mark.parametrize('saved_frame',[False,True])
def test_prefix_retains_genuine_final_failed_attempt_with_or_without_saved_frame(tmp_path,saved_frame):
    audit=module();run=episode(tmp_path/'run',success=False);raw=run/'output/review/raw_multimodal'
    path=raw/'rgbd/attempts.jsonl';data=rows(path)
    data[-1].update(status='error',failure={'stage':'attempt_complete_write' if saved_frame else 'render',
                                          'error':'original camera failure'})
    if not saved_frame:
        for key in ('raw_file','raw_sha256','partial_file'):data[-1].pop(key,None)
        frames_path=raw/'rgbd/frames.jsonl';frames=rows(frames_path)[:1]
        frames_path.write_text(''.join(json.dumps(r)+'\n' for r in frames))
        receipt=read(raw/'rgbd/receipt.json');receipt['frames']=1;write(raw/'rgbd/receipt.json',receipt)
        refresh_stream(run,'rgbd','frames.jsonl')
    path.write_text(''.join(json.dumps(r)+'\n' for r in data));refresh_stream(run,'rgbd','attempts.jsonl')
    before=tree_hashes(run)
    report=audit.audit_capture(run,mode='prefix')
    assert report['consistency_passed'] and not report['complete_capture_audited']
    assert report['original']['camera_receipt']['completed'] is False
    assert report['rgbd_frames']==(2 if saved_frame else 1)
    assert tree_hashes(run)==before


def test_raw_complete_is_independent_of_original_physical_failure(tmp_path):
    audit=module();run=episode(tmp_path/'run');path=run/'output/review/fresh_expert_result.json'
    result=read(path);result.update(success=False,physical_validation=False,
                                   failure_reason='original physical hold requirement not met')
    write(path,result)
    report=audit.audit_capture(run,mode='complete')
    assert report['complete_capture_audited'] and report['original']['raw_capture_complete']
    assert report['original']['success'] is False and report['original']['physical_validation'] is False
    assert not report['physical_qualification'] and not report['training_eligible']


@pytest.mark.parametrize('role',['capture','backend','environment','hall_stream','calibration',
                                 'host_calibration','geometry_gate'])
def test_rgbd_source_roles_cannot_be_omitted_from_hash_verification(tmp_path,role):
    audit=module();run=episode(tmp_path/'run');raw=run/'output/review/raw_multimodal'
    path=raw/'rgbd/manifest.json';manifest=read(path);del manifest['source_bindings'][role]
    write(path,manifest);refresh_stream(run,'rgbd','manifest.json')
    with pytest.raises(ValueError,match='source|role'):
        audit.audit_capture(run,mode='complete')


@pytest.mark.parametrize('mode', ['prefix', 'complete'])
@pytest.mark.parametrize('alias', ['same_name', 'symlink', 'wrong_frame_number'])
def test_frames_require_distinct_files_named_for_their_own_index(tmp_path, mode, alias):
    audit=module();run=episode(tmp_path/'run');raw=run/'output/review/raw_multimodal'
    frames_path=raw/'rgbd/frames.jsonl';frames=rows(frames_path)
    first,second=raw/'rgbd'/frames[0]['raw_file'],raw/'rgbd'/frames[1]['raw_file']
    second.unlink()
    if alias=='same_name': frames[1]['raw_file']=frames[0]['raw_file']
    elif alias=='symlink': second.symlink_to(first.name)
    else:
        frames[1]['raw_file']='frame-00000099.npz'
        (raw/'rgbd'/frames[1]['raw_file']).write_bytes(first.read_bytes())
    frames[1]['raw_sha256']=sha(first)
    frames_path.write_text(''.join(json.dumps(r)+'\n' for r in frames))
    refresh_stream(run,'rgbd','frames.jsonl')
    path=raw/'rgbd/attempts.jsonl';attempts=rows(path)
    for key in ('raw_file','raw_sha256'): attempts[-1][key]=frames[1][key]
    path.write_text(''.join(json.dumps(r)+'\n' for r in attempts));refresh_stream(run,'rgbd','attempts.jsonl')
    with pytest.raises(ValueError,match='file|frame'):
        audit.audit_capture(run,mode=mode)


def test_identical_raw_bytes_in_distinct_correctly_named_frames_are_valid(tmp_path):
    audit=module();run=episode(tmp_path/'run');raw=run/'output/review/raw_multimodal'
    path=raw/'rgbd/frames.jsonl';frames=rows(path)
    (raw/'rgbd'/frames[1]['raw_file']).write_bytes((raw/'rgbd'/frames[0]['raw_file']).read_bytes())
    frames[1]['raw_sha256']=frames[0]['raw_sha256']
    path.write_text(''.join(json.dumps(r)+'\n' for r in frames));refresh_stream(run,'rgbd','frames.jsonl')
    path=raw/'rgbd/attempts.jsonl';attempts=rows(path);attempts[-1]['raw_sha256']=frames[1]['raw_sha256']
    path.write_text(''.join(json.dumps(r)+'\n' for r in attempts));refresh_stream(run,'rgbd','attempts.jsonl')
    assert audit.audit_capture(run,mode='complete')['complete_capture_audited']


@pytest.mark.parametrize('role', ['capture','hall_stream','calibration','host_calibration'])
def test_rgbd_source_roles_cannot_exchange_otherwise_valid_source_records(tmp_path,role):
    audit=module();run=episode(tmp_path/'run');raw=run/'output/review/raw_multimodal'
    path=raw/'rgbd/manifest.json';manifest=read(path);bindings=manifest['source_bindings']
    bindings[role],bindings['geometry_gate']=bindings['geometry_gate'],bindings[role]
    write(path,manifest);refresh_stream(run,'rgbd','manifest.json')
    with pytest.raises(ValueError,match='source.*role|role.*module'):
        audit.audit_capture(run,mode='complete')


@pytest.mark.parametrize('role',['backend','environment'])
def test_native_rgbd_source_roles_require_the_production_modules(tmp_path,role):
    audit=module();run=episode(tmp_path/'run');raw=run/'output/review/raw_multimodal'
    path=raw/'rgbd/manifest.json';manifest=read(path)
    manifest['injected_backend']=False
    manifest['source_bindings']['backend']=deepcopy(manifest['source_bindings']['capture'])
    env=ROOT/'sim/ausc_policy/fresh_expert.py'
    manifest['source_bindings']['environment']=dict(module='sim.ausc_policy.fresh_expert',path=str(env),sha256=sha(env))
    manifest['source_bindings'][role]=deepcopy(manifest['source_bindings']['geometry_gate'])
    write(path,manifest);refresh_stream(run,'rgbd','manifest.json')
    with pytest.raises(ValueError,match='source.*role|role.*module'):
        audit.audit_capture(run,mode='complete')


def test_rgbd_hall_source_bytes_must_match_the_bound_hall_stream(tmp_path):
    audit=module();run=episode(tmp_path/'run');raw=run/'output/review/raw_multimodal'
    path=raw/'rgbd/manifest.json';manifest=read(path)
    record=manifest['source_bindings']['hall_stream']
    other=tmp_path/'other/sim/ausc_policy/m2_sensors.py';other.parent.mkdir(parents=True)
    other.write_bytes(Path(record['path']).read_bytes()+b'\n# other acquisition source\n')
    record.update(path=str(other),sha256=sha(other))
    write(path,manifest);refresh_stream(run,'rgbd','manifest.json')
    with pytest.raises(ValueError,match='source.*identity|Hall.*source'):
        audit.audit_capture(run,mode='complete')


def test_injected_capture_is_explicitly_reported_as_injected_evidence(tmp_path):
    report=module().audit_capture(episode(tmp_path/'run'),mode='complete')
    assert report['sources']['rgbd_injected_backend'] is True


def test_unknown_rgbd_schema_cannot_inherit_current_frame_or_role_rules(tmp_path):
    audit=module();run=episode(tmp_path/'run');raw=run/'output/review/raw_multimodal'
    path=raw/'rgbd/manifest.json';manifest=read(path);manifest['schema']='unknown_future_schema'
    write(path,manifest);refresh_stream(run,'rgbd','manifest.json')
    with pytest.raises(ValueError,match='schema'):
        audit.audit_capture(run,mode='complete')


@pytest.mark.parametrize('location',['frame','scheduled_event','unscheduled_event','error_event'])
def test_complete_rejects_recorded_tick_errors_even_with_error_free_receipts(tmp_path,location):
    audit=module();run=episode(tmp_path/'run');raw=run/'output/review/raw_multimodal'
    if location=='frame':
        path=raw/'rgbd/frames.jsonl';records=rows(path);records[0]['context']['tick_error']='original tick failure'
        path.write_text(''.join(json.dumps(r)+'\n' for r in records));refresh_stream(run,'rgbd','frames.jsonl')
    else:
        path=raw/'events.jsonl';records=rows(path)
        if location=='error_event':
            records.append(dict(status='error',stage='controller_tick',error='original failure',training_eligible=False))
        else: records[11 if location=='scheduled_event' else 0]['context']['tick_error']='original tick failure'
        path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    with pytest.raises(ValueError,match='event|tick|context'):
        audit.audit_capture(run,mode='complete')


@pytest.mark.parametrize('fault',['missing_file','missing_unscheduled_tick','duplicate','clock','camera_due'])
def test_complete_requires_the_entire_recorded_tick_event_sequence(tmp_path,fault):
    audit=module();run=episode(tmp_path/'run');raw=run/'output/review/raw_multimodal'
    path=raw/'events.jsonl';records=rows(path)
    if fault=='missing_file':path.unlink()
    else:
        if fault=='missing_unscheduled_tick':del records[0]
        elif fault=='duplicate':records.insert(1,deepcopy(records[0]))
        elif fault=='clock':records[0]['time_s']=99.
        else:records[0]['camera_due']=True
        path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    with pytest.raises(ValueError,match='event|tick'):
        audit.audit_capture(run,mode='complete')


def test_prefix_preserves_recorded_tick_failure_and_missing_tail_event(tmp_path):
    audit=module();run=episode(tmp_path/'run',success=False,steps=25);raw=run/'output/review/raw_multimodal'
    path=raw/'events.jsonl';events=rows(path)[:-1]
    events[11]['context']['tick_error']='original controller failure'
    events.insert(11,dict(status='error',stage='controller_tick',error='original controller failure',training_eligible=False))
    path.write_text(''.join(json.dumps(r)+'\n' for r in events))
    path=raw/'rgbd/frames.jsonl';frames=rows(path);frames[0]['context']['tick_error']='original controller failure'
    path.write_text(''.join(json.dumps(r)+'\n' for r in frames));refresh_stream(run,'rgbd','frames.jsonl')
    before=tree_hashes(run);report=audit.audit_capture(run,mode='prefix')
    assert report['events']['tick_error_rows']==[11]
    assert report['events']['error_records'][0]['stage']=='controller_tick'
    assert report['events']['missing_physics_tick_count']==1
    assert not report['complete_capture_audited'] and not report['training_eligible']
    assert tree_hashes(run)==before


@pytest.mark.parametrize('index', [0, 3])
def test_prefix_cannot_omit_an_event_before_later_recorded_ticks(tmp_path,index):
    audit=module();run=episode(tmp_path/'run',success=False);raw=run/'output/review/raw_multimodal'
    path=raw/'events.jsonl';events=rows(path);del events[index]
    path.write_text(''.join(json.dumps(r)+'\n' for r in events))
    with pytest.raises(ValueError,match='event|tick'):
        audit.audit_capture(run,mode='prefix')


@pytest.mark.parametrize('location', ['frame', 'event', 'context'])
@pytest.mark.parametrize('key,value', [('status','error'), ('tick_error','recorded failure')])
def test_explicit_record_errors_are_rejected_by_complete_and_preserved_by_prefix(tmp_path,location,key,value):
    audit=module();run=episode(tmp_path/'run');raw=run/'output/review/raw_multimodal'
    event_path=raw/'events.jsonl';events=rows(event_path)
    frame_path=raw/'rgbd/frames.jsonl';frames=rows(frame_path)
    if location=='frame':frames[0][key]=value
    elif location=='event':events[0][key]=value
    else:
        events[11]['context'][key]=value
        frames[0]['context'][key]=value
    event_path.write_text(''.join(json.dumps(r)+'\n' for r in events))
    frame_path.write_text(''.join(json.dumps(r)+'\n' for r in frames))
    refresh_stream(run,'rgbd','frames.jsonl')
    before=tree_hashes(run)
    with pytest.raises(ValueError,match='error'):
        audit.audit_capture(run,mode='complete')
    report=audit.audit_capture(run,mode='prefix')
    assert not report['complete_capture_audited'] and not report['training_eligible']
    if location=='event':assert report['events']['tick_error_rows']==[0]
    else:assert report['frame_error_records'][0]['physics_row']==11
    assert tree_hashes(run)==before
