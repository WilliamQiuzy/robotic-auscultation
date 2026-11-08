"""Data-only V5 phase queries over a sealed, already admitted V4 snapshot.

The pinned producer authenticated native H2, command seams, the failed original
Native source and source28 closure. We inherit that proof; we do not replay a
model, open native trace payloads, rerun historical factories, or fit statistics.
Only original unexecuted oracle proposals supply future setter labels.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import numpy as np

from . import m2_act_auxiliary_data as old, m2_act_terminal_data as v4
from . import m2_act_recovery_scope as scope

rt = old.rt
need = rt.need
ARRAY_KEYS = old.ARRAY_KEYS
SCHEMA = 'm2_act_auxiliary_preparation_v5'
SOURCE_SCHEMA = 'm2_act_auxiliary_source_v5'
ADMISSION_SCHEMA = 'm2_act_phase_computational_admission_v5'
PROBE_SCHEMA = 'act7_phase_oracle_probe_v1'
PROBE_FLAGS = dict(actions_executed=False, physics_qualified=False, training_eligible=False,
    training_started=False, gpu_launched=False, fresh_payloads_opened=False,
    validation_histories_opened=False, physics_gates_changed=False, automatic_retry=False,
    automatic_resume=False, formal_acceptance=False)
SLOT_IDS = ('near_entry.first','near_entry.next','before60.previous','before60.anchor','last.previous','last.anchor')
SOURCE28_COMMIT = 'd932993b13030da7873d04a4a9471df99bb836ec'
ORACLE_KEYS = ('source_observed_histories','source_observation_times','source_physics_rows',
    'proposed_relative_setters','proposed_is_pad')
_BASE = '/opt/wam-runtime/fast-wam/'
_CONTROL = _BASE+'diagnostics/act7-phase-oracle-probe-20261107-v1/'
_PARENT = _BASE+'datasets/act6-terminal-full-chunks-20261107-v1/'
_CODE = '/opt/wam-runtime/code/wam-act-recovery-20261106-v28/'


@dataclass(frozen=True)
class _Inputs:
    parent_complete: dict
    parent_metadata: dict
    parent_arrays: dict
    producer_complete: dict
    producer_manifest: dict
    producer_operator: dict
    frozen_request: dict
    final_slot_ledger: dict
    oracle_complete: dict
    oracle_arrays: dict
    source28_manifest: dict
    parent_rows: int = 35988
    parent_sources: int = 940
    parent_slots: int = 1115
    source28_files: int = 1673
    subject_id: str = 'p001120'
    pose_id: str = 'p000474_a001230_f0000'
    target_id: str = 'L3'


@dataclass(frozen=True)
class ProductionInputs(_Inputs):
    """Only the exact closed production descriptor is accepted."""


@dataclass(frozen=True)
class LocalFixtureInputs(_Inputs):
    """Explicit synthetic local binding; never described as production evidence."""


PRODUCTION_INPUTS = ProductionInputs(
    parent_complete=dict(path=_PARENT+'COMPLETE.json',bytes=1129,sha256='d5ec96f999498dd3133dc8984bc79b44ad2eb00be24f14bc1bebb269045bf7ce'),
    parent_metadata=dict(path=_PARENT+'METADATA.json',bytes=13439079,sha256='133a78792793f071f0637de8673db1511b0174ef12209336635b0cfe320cb049'),
    parent_arrays=dict(path=_PARENT+'AUXILIARY_ARRAYS.npz',bytes=8338733,sha256='d1360dfcaab41741e7f17ce70c38f18506d4bccef883dc81e6971a7bb13075ba'),
    producer_complete=dict(path=_CONTROL+'COMPLETE.json',bytes=5610,sha256='146cefaeea1094b1fcd47b3dfd28bb78ca5f7e45475c3b3f299691c7b3e2a78e'),
    producer_manifest=dict(path=_CONTROL+'OUTPUT_MANIFEST.json',bytes=6425,sha256='e07021c7b20aa69992c1c6665eadd34596811caf6c53e049fe84c9ab81c5e87a'),
    producer_operator=dict(path=_BASE+'diagnostics/run_act7_phase_oracle_probe_20261107.py',bytes=39293,sha256='11054e4fc24a531a6cefe49202b29b286926c54078430601a254806338655e4e'),
    frozen_request=dict(path=_CONTROL+'FROZEN_REQUEST.json',bytes=196259,sha256='e168d985005843a33d68fa61b480dc850e21f8f93852a14cef63417c17dcb355'),
    final_slot_ledger=dict(path=_CONTROL+'FINAL_SLOT_LEDGER.json',bytes=7896,sha256='2c342776f4e5d19c02be33ca73a10b901efc126e74b5ea1133c0a8ab40dae4e4'),
    oracle_complete=dict(path=_CONTROL+'oracle/COMPLETE.json',bytes=3464,sha256='ebd92bae3816ff48437e43c6ad031e31019e84ba2644e0a2ddb576e145769a40'),
    oracle_arrays=dict(path=_CONTROL+'oracle/UNEXECUTED_PROPOSALS.npz',bytes=2861,sha256='a5cca28b88d33021ccd53706a140eba38494f3226dc7bc7767e423c00b64a046'),
    source28_manifest=dict(path=_CODE+'SOURCE_MANIFEST.json',bytes=289698,sha256='f05a78822bcc6febbaa7f4a7a8de2cabd9aa33daed3d648b90053cb336edb89c'))
_PRODUCTION_DIGEST = rt.digest(asdict(PRODUCTION_INPUTS))


def _inputs(value):
    value = PRODUCTION_INPUTS if value is None else value
    need(type(value) in (ProductionInputs,LocalFixtureInputs), 'Explicit typed V5 inputs required')
    if type(value) is ProductionInputs:
        need(rt.digest(asdict(value)) == _PRODUCTION_DIGEST, 'Exact immutable production V5 input descriptor required')
    for key,ref in asdict(value).items():
        if isinstance(ref,dict):
            rt._reference_shape(ref)
            need(set(ref)=={'path','bytes','sha256'} and type(ref['bytes']) is int and ref['bytes']>=0,
                'Complete immutable input reference required: '+key)
            if type(value) is LocalFixtureInputs:
                need(not Path(ref['path']).is_relative_to('/opt/wam-runtime'), 'Synthetic descriptor must use explicit local files')
    return value


def _profile(inputs):
    return 'production_closed_inputs' if type(inputs) is ProductionInputs else 'local_synthetic_fixture'


def _flags(value,keys=PROBE_FLAGS):
    need(all(value.get(k) is False for k in keys), 'Unexecuted, unqualified data-only flags required')


def _allowed(allowed):
    need(isinstance(allowed,(list,tuple,set,frozenset)) and bool(allowed) and set(allowed)<=set(scope.TRAIN16_IDS),
        'Explicit original TRAIN16 identity pool required')


def _identities(sources,allowed):
    _allowed(allowed)
    need(isinstance(sources,list), 'Explicit source table required')
    for source in sources:
        old._train(source.get('subject_id'),source.get('split'),allowed)
        meta = source.get('metadata',{})
        old._train(meta.get('subject_id'),meta.get('split'),allowed)
        need(all(source.get(k)==meta.get(k) for k in ('subject_id','split','target_id')), 'Source metadata identity/target differs')


def array_proof(array):
    value = np.asarray(array)
    return dict(shape=list(value.shape),dtype=value.dtype.str,
        sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())


def code_binding():
    return dict(schema='m2_act_phase_data_code_v5',files=dict(data=rt.reference(__file__),
        inherited_v4=rt.reference(v4.__file__),generic_protocol=rt.reference(old.__file__),
        runtime=rt.reference(rt.__file__),train16_scope=rt.reference(scope.__file__)))


def _parent_header(inputs,allowed):
    complete = rt.read_bound(inputs.parent_complete)
    need(complete.get('metadata')==inputs.parent_metadata and complete.get('arrays')==inputs.parent_arrays,
        'Exact three-file immutable V4 parent required')
    meta = rt.read_bound(inputs.parent_metadata); _identities(meta.get('sources'),allowed)
    need(complete.get('schema')==meta.get('schema')==v4.SCHEMA and complete.get('complete') is True and
        complete.get('metadata_sha256')==inputs.parent_metadata['sha256'] and
        complete.get('optimization_admission')==meta.get('optimization_admission')==v4.ADMISSION,
        'Explicit already-admitted closed V4 parent required')
    _flags(complete,old.FLAGS); _flags(meta,old.FLAGS)
    need(meta.get('source_table_sha256')==rt.digest(meta['sources']) and meta.get('allowed_train_ids')==sorted(allowed) and
        len(meta['sources'])==inputs.parent_sources and len(meta.get('slot_ledger',[]))==inputs.parent_slots and
        sum(len(s.get('windows',[])) for s in meta['sources'])==inputs.parent_rows and
        isinstance(meta.get('request_refs'),list) and bool(meta['request_refs']) and
        isinstance(meta.get('provenance'),dict) and bool(meta['provenance']),
        'Complete V4 source/slot/request provenance and fixed counts required')
    rt.bound_path(inputs.parent_arrays)  # binary hash only, never JSON decoding
    return complete,meta


def _seal(inputs):
    seal = rt.read_bound(inputs.producer_manifest); _flags(seal)
    need(seal.get('schema')==PROBE_SCHEMA and isinstance(seal.get('files'),list), 'Original sealed producer manifest required')
    root = Path(inputs.producer_complete['path']).parent
    refs = {}
    for ref in seal['files']:
        rt._reference_shape(ref); path=Path(ref['path'])
        need(path.is_relative_to(root) and path!=Path(inputs.producer_manifest['path']) and ref['path'] not in refs,
            'Unique producer output files within sealed root required')
        refs[ref['path']]=ref
    actual={str(p) for p in root.rglob('*') if p.is_file() and p!=Path(inputs.producer_manifest['path'])}
    need(actual==set(refs), 'Sealed producer manifest omitted or added an output file')
    for key in ('producer_complete','frozen_request','final_slot_ledger','oracle_complete','oracle_arrays'):
        ref=getattr(inputs,key);need(refs.get(ref['path'])==ref, 'Exact pinned output missing from sealed manifest: '+key)
    # Metadata identity checks happen before payload hashing/opening.
    return refs


def _sealed_json(ref,refs):
    need(refs.get(ref.get('path'))==ref, 'Original producer JSON reference is outside sealed manifest')
    need(Path(ref['path']).suffix=='.json', 'Only explicit JSON metadata may be decoded')
    return rt.read_bound(ref)


def _source28(inputs,closure,generator):
    need(closure.get('code_commit')==SOURCE28_COMMIT and closure.get('source_manifest')==inputs.source28_manifest and
        closure.get('file_count')==inputs.source28_files and closure.get('root')==str(Path(inputs.source28_manifest['path']).parent),
        'Pinned inherited original source28 closure required')
    rows=rt.read_bound(inputs.source28_manifest)
    need(isinstance(rows,list) and len(rows)==inputs.source28_files, 'Complete original source28 manifest inventory required')
    entries={}
    for row in rows:
        name=row.get('file')
        need(set(row)=={'file','bytes','sha256'} and isinstance(name,str) and name and not Path(name).is_absolute() and
            '..' not in Path(name).parts and name not in entries, 'Safe unique source28 file inventory required')
        entries[name]=row
    name='sim/ausc_policy/m2_student_oracle_proposals.py'; entry=entries.get(name,{})
    need(generator==dict(path=str(Path(closure['root'])/name),bytes=entry.get('bytes'),sha256=entry.get('sha256')),
        'Original oracle generator must bind source28 manifest entry')
    rt.bound_path(generator)
    commit_ref=closure.get('code_commit_file',{})
    need(commit_ref.get('path')==str(Path(closure['root'])/'CODE_COMMIT'), 'Original source28 commit-file binding required')
    need(rt.bound_path(commit_ref).read_text().strip()==SOURCE28_COMMIT, 'Original source28 commit bytes differ')


def _plan(metadata,source):
    """Verify bound plan/target/reference; never invent a prepose gap."""
    rt.geometry.require_current(metadata); reference=rt.pose.require_route(metadata)
    fresh=reference.snapshot()['sources']['fresh_plan']
    plan_ref=rt.reference(fresh['file'])
    need(plan_ref['sha256']==fresh['sha256'] and source['bindings']['source']==plan_ref,
        'Source plan path/SHA differs from bound canonical reference')
    plan=rt.read_bound(plan_ref); gap=plan.get('free_approach_start_gap_m')
    need(plan.get('name',plan.get('target'))==source['target_id']==reference.snapshot()['target_id'] and
        type(gap) in (int,float) and np.isfinite(gap) and .025<=gap<=.06,
        'Target-aligned canonical plan actual prepose gap required')
    return float(gap),reference,plan_ref


def _geometry(frozen,inputs):
    meta=frozen['source_metadata']; reference=rt.pose.require_route(meta)
    fresh=reference.snapshot()['sources']['fresh_plan']; plan_ref=rt.reference(fresh['file'])
    need(plan_ref['sha256']==fresh['sha256'], 'Bound canonical plan SHA differs')
    plan=rt.read_bound(plan_ref); gap=plan.get('free_approach_start_gap_m'); g=frozen.get('precontact_geometry',{})
    need(type(gap) in (int,float) and np.isfinite(gap) and .025<=gap<=.06 and
        plan.get('name',plan.get('target'))==inputs.target_id and g.get('fresh_plan')==fresh and
        g.get('point_plan_provenance',{}).get('plan_file')==fresh['file'] and
        g.get('point_plan_provenance',{}).get('plan_sha256')==fresh['sha256'] and
        g.get('point_native_report')==frozen['source']['native'] and g.get('point_index')==0 and
        g.get('point_target_id')==inputs.target_id and
        all(type(g.get(k)) in (int,float) and g[k]==gap for k in ('point_phase_precontact_gap_m',
            'point_checked_prepose_gap_m','plan_free_approach_start_gap_m','precontact_gap_m')) and
        g.get('desired_skin_relative_position_m')==[0.,0.,-gap] and
        g.get('state17_position_semantics')=='Rref.T @ (TCP - skin_contact_point)' and
        g.get('reference_normal_semantics')=='minus_reference_pose_Z_axis' and
        g.get('selection_error_semantics')=='norm(state17[7:10] - [0,0,-actual_point_gap])',
        'Original point/plan/target/gap/reference coordinate identity differs')
    return g


def _roles(frozen,role,final,root):
    slots=frozen.get('slots'); eligible=frozen.get('eligible_queries')
    need(isinstance(slots,list) and len(slots)==6 and role.get('schema')==final.get('schema')=='act7_phase_slot_roles_v1' and
        role.get('slots')==slots and len(final.get('slots',[]))==6, 'All six frozen semantic roles and outcomes required')
    need(isinstance(eligible,list) and all(e.get('eligible_index')==i for i,e in enumerate(eligible)),
        'Original eligible request order required')
    need(all(type(e.get('physics_row')) is int and e['physics_row']>=12 and e['physics_row']%12==0 and
        abs(e['time_s']-e['physics_row']/240.)<=1e-8 and type(e.get('request_index')) is int and
        all(type(e.get(k)) in (int,float) and np.isfinite(e[k]) and e[k]>=0 for k in
            ('precontact_relative_position_error_norm_m','skin_relative_position_norm_m')) for e in eligible) and
        all(a['physics_row']<b['physics_row'] and a['request_index']<b['request_index'] for a,b in zip(eligible,eligible[1:])),
        'Finite ordered eligible native query clocks/positions required')
    near=next((i for i,e in enumerate(eligible) if e['precontact_relative_position_error_norm_m']<=.006),None)
    before=next((i for i in range(len(eligible)-1,-1,-1) if eligible[i]['time_s']<=60.+1e-8),None)
    indices=[near,None if near is None else near+1,None if before is None else before-1,before,len(eligible)-2,len(eligible)-1]
    reserved={eligible[i]['physics_row'] for i in indices[4:] if i is not None and 0<=i<len(eligible)};seen=set()
    for n,(slot,index,outcome) in enumerate(zip(slots,indices,final['slots'])):
        exists=index is not None and 0<=index<len(eligible)
        intended='potential_train' if n<4 else 'reserved_diagnostic'
        effective=intended if exists else 'missing';status='selected' if exists else 'missing';reason=None
        if exists and n<4:
            row=eligible[index]['physics_row']
            if row in reserved:effective=status='duplicate_excluded';reason='reserved_diagnostic_role_wins'
            elif row in seen:effective=status='duplicate_excluded';reason='earlier_potential_train_slot_owns_unique_row'
            else:seen.add(row)
        need(slot.get('slot_id')==SLOT_IDS[n] and slot.get('slot_index')==n and
            slot.get('pair_id')==('near_entry','before60','last')[n//2] and slot.get('intended_role')==intended and
            slot.get('effective_role')==effective and slot.get('status')==status and slot.get('training_eligible') is False and
            all(outcome.get(k)==v for k,v in slot.items()), 'Frozen role/source ordering or exclusion semantics changed')
        need((exists and all(slot.get(k)==v for k,v in eligible[index].items())) or
            (not exists and all(slot.get(k) is None for k in ('physics_row','eligible_index','request_index','query_id','time_s'))),
            'Frozen role must match original eligible query or explicit missing slot')
        if reason:need(slot.get('exclusion_reason')==reason, 'Duplicate exclusion reason must survive')
    for n in (0,2,4):
        a,b=slots[n:n+2]; adjacent=a['eligible_index'] is not None and b['eligible_index'] is not None and b['eligible_index']==a['eligible_index']+1
        need(a.get('pair_is_adjacent_eligible_requests') is adjacent and b.get('pair_is_adjacent_eligible_requests') is adjacent,
            'Original eligible-query adjacency differs')
    rows=sorted({s['physics_row'] for s in slots if s['physics_row'] is not None},key=lambda row:next(e['eligible_index'] for e in eligible if e['physics_row']==row))
    need(frozen.get('physics_rows')==rows and frozen.get('distinct_query_denominator')==root.get('distinct_query_denominator')==len(rows) and
        root.get('eligible_query_count')==len(eligible), 'Fixed distinct query/eligible denominator differs')
    for value in (root,role):
        need(value.get('semantic_slot_denominator')==6 and value.get('potential_train_slot_denominator')==4 and
            value.get('reserved_diagnostic_slot_denominator')==2, 'Fixed six/four/two role denominators required')
    need(final.get('semantic_slot_denominator')==frozen.get('semantic_slot_denominator')==6, 'Six semantic attempts must survive')
    return slots,rows


def _probe_header(inputs,allowed):
    refs=_seal(inputs)
    root=_sealed_json(inputs.producer_complete,refs); frozen=_sealed_json(inputs.frozen_request,refs)
    # Inspect authoritative source metadata/config before queries, proposals or NPZ.
    meta=frozen.get('source_metadata',{});old._train(meta.get('subject_id'),meta.get('split'),allowed)
    need(meta.get('subject_id')==inputs.subject_id and meta.get('target_id')==inputs.target_id,
        'Original permitted TRAIN target identity required')
    rt.geometry.require_current(meta);reference=rt.pose.require_route(meta)
    source=frozen.get('source',{})
    need(set(source)=={'config','owner','child','native'} and root.get('source')==source,
        'Authentic original source config/owner/child/Native refs required')
    config=rt.read_bound(source['config']);actual=config.get('m2_given_contact_expanded_pilot_deployment',{}).get('metadata',{})
    old._train(actual.get('subject_id'),actual.get('split'),allowed)
    need(actual==meta, 'Source configuration metadata differs')
    patient=rt.pose.bound_json(reference.snapshot()['sources']['patient_receipt'])
    need(patient.get('identity')==inputs.subject_id and patient.get('proposal')==inputs.subject_id+'_'+inputs.pose_id,
        'Actual bound installed body/pose differs')
    for value in (root,frozen):_flags(value)
    need(root.get('schema')==frozen.get('schema')==PROBE_SCHEMA and root.get('operator')==frozen.get('operator')==inputs.producer_operator and
        root.get('frozen_request')==inputs.frozen_request and root.get('final_slot_ledger')==inputs.final_slot_ledger and
        root.get('oracle_result')==inputs.oracle_complete and root.get('model')==frozen.get('model') and
        root.get('source28')==root.get('source28_end')==frozen.get('source28') and
        type(root.get('source_native_physics_steps')) is int and root['source_native_physics_steps']>0,
        'Pinned closed probe/source/model/operator/Native evidence differs')
    rt.bound_path(inputs.producer_operator);_source28(inputs,root['source28'],frozen['generator'])
    for ref in source.values():rt.bound_path(ref)  # inherit native authentication; never decode opaque/weight packets
    geometry=_geometry(frozen,inputs)
    role=_sealed_json(root.get('slot_role_ledger',{}),refs); final=_sealed_json(inputs.final_slot_ledger,refs)
    _flags(role);_flags(final)
    need(root['slot_role_ledger']==frozen.get('slot_role_ledger')==final.get('frozen_slot_role_ledger') and
        final.get('frozen_request')==inputs.frozen_request, 'Original role/frozen/final ledger references differ')
    slots,rows=_roles(frozen,role,final,root)
    # All sealed files are now authenticated; binary inputs are hash-only here.
    for ref in refs.values():rt.bound_path(ref)
    return dict(root=root,frozen=frozen,role=role,final=final,slots=slots,rows=rows,refs=refs,geometry=geometry)


def _archive(ref,keys):
    with np.load(rt.bound_path(ref),allow_pickle=False) as archive:
        need(set(archive.files)==set(keys), 'Exact separate binary NPZ array keys required')
        return {k:archive[k].copy() for k in keys}


def _query(record,slot_rows,header,inputs):
    _flags(record)
    need(record.get('schema')=='act7_authenticated_measured_query_v1' and record.get('source')==header['frozen']['source'] and
        record.get('physics_row') in slot_rows and record.get('actual_student_output') is True and
        record.get('positive_teacher_label') is False and record.get('future_measured_observations') is False and
        record.get('precontact_geometry')==header['geometry'], 'Original measured query provenance differs')
    row=record['physics_row'];members=[s for s in header['slots'] if s['physics_row']==row]
    need(record.get('slot_ids')==[s['slot_id'] for s in members] and record.get('slot_roles')==[
        {k:s[k] for k in ('slot_id','intended_role','effective_role')} for s in members] and
        record.get('reserved_diagnostic_only') is any(s['intended_role']=='reserved_diagnostic' for s in members),
        'Measured query roles/reserved attribution differs')
    query=record.get('query',{});meta=header['frozen']['source_metadata']
    old._train(query.get('metadata',{}).get('subject_id'),query.get('metadata',{}).get('split'),{inputs.subject_id})
    need(query.get('metadata')==meta and query.get('reference_sha256')==meta['reference_sha256'], 'Measured query reference/metadata differs')
    h=np.asarray(query.get('history'),float); times=np.asarray(query.get('history_times'),float);q=np.asarray(query.get('measured_q'),float)
    need(h.shape==(2,17) and times.shape==(2,) and q.shape==(7,) and np.isfinite(h).all() and np.isfinite(times).all() and
        np.isfinite(q).all() and query.get('physics_row')==row and type(query.get('time_s')) in (int,float) and
        abs(query['time_s']-row/240.)<=1e-8 and abs(times[1]-query['time_s'])<=1e-8 and
        abs(times[1]-times[0]-.05)<=1e-8 and np.array_equal(h[-1,:7],q) and
        query.get('context')==[meta['episode_id'],inputs.target_id,0,'APPROACH'], 'Measured H2/origin/context/clock differs')
    skin=float(np.linalg.norm(h[-1,7:10]));delta=h[-1,7:10].copy();delta[2]+=header['geometry']['precontact_gap_m'];error=float(np.linalg.norm(delta))
    need(abs(record.get('skin_relative_position_norm_m',-1)-skin)<=1e-10 and
        abs(record.get('precontact_relative_position_error_norm_m',-1)-error)<=1e-10 and
        all(abs(s['skin_relative_position_norm_m']-skin)<=1e-10 and abs(s['precontact_relative_position_error_norm_m']-error)<=1e-10 for s in members),
        'Measured query skin/prepose position norms differ')
    student=np.asarray(record.get('actual_original_student_absolute_h20'),float)
    seam=np.asarray(record.get('pre_query_three_command_seam'),float);snapshot=record.get('retained_interpolator_snapshot',{})
    fields=snapshot.get('fields',{})
    need(student.shape==(20,7) and np.isfinite(student).all() and seam.shape==(3,7) and np.isfinite(seam).all() and
        all(np.asarray(fields.get(k)).shape==(7,) and np.isfinite(np.asarray(fields[k],float)).all() for k in
            ('current_position','current_velocity','current_acceleration')) and np.array_equal(seam[-1],fields['current_position']),
        'Original retained student/dynamics/three-command seam differs')
    return query


def _additions(header,inputs,allowed,parent_arrays):
    refs=header['refs'];frozen=header['frozen'];root=header['root'];rows=header['rows'];meta=frozen['source_metadata']
    query_refs=frozen.get('diagnostic_queries')
    need(isinstance(query_refs,list) and len(query_refs)==len(rows), 'Every distinct original measured query artifact required')
    records=[_sealed_json(ref,refs) for ref in query_refs]
    queries=[_query(record,rows,header,inputs) for record in records]
    need([q['physics_row'] for q in queries]==rows, 'Original measured query artifact order differs')
    oracle=_sealed_json(inputs.oracle_complete,refs);_flags(oracle,('actions_executed','physics_qualified','training_eligible','gpu_launched'))
    need(oracle.get('schema')=='m2_student_cpu_oracle_diagnostic_v1' and oracle.get('complete') is True and
        oracle.get('source_subject_id')==inputs.subject_id and oracle.get('source_split')=='train' and
        oracle.get('source_episode_verdict_unchanged') is True and oracle.get('arrays')==inputs.oracle_arrays,
        'Original unexecuted TRAIN oracle outcome required')
    request_ref=refs.get(str(Path(inputs.oracle_complete['path']).parent/'REQUEST.json'),{})
    request=_sealed_json(request_ref,refs)
    need(request.get('source')==frozen['source'] and request.get('physics_rows')==rows and
        request.get('plan_config')==frozen.get('plan_config') and request.get('generator')==frozen['generator'],
        'Original oracle REQUEST source/plan/rows/generator differs')
    _flags(request,('actions_executed','physics_qualified','training_eligible'))
    outcomes=oracle.get('outcomes',[])
    need(len(outcomes)==len(rows) and [o.get('physics_row') for o in outcomes]==rows and
        oracle.get('attempted_queries')==len(rows) and all(type(o.get('proposal_accepted')) is bool for o in outcomes),
        'Every attempted original oracle outcome must survive in order')
    accepted=[];by_row={}
    for n,(outcome,record,query) in enumerate(zip(outcomes,records,queries)):
        row=rows[n];_flags(outcome,('actions_executed','physics_qualified','training_eligible'))
        saved=refs.get(str(Path(inputs.oracle_complete['path']).parent/('outcome-'+str(row)+'.json')), {})
        need(_sealed_json(saved,refs)==outcome, 'Original per-query outcome differs')
        if outcome['proposal_accepted']:
            pr=outcome.get('computational_proposal',{});proposal=_sealed_json(pr,refs)
            need(Path(pr['path'])==Path(inputs.oracle_complete['path']).parent/('query-'+str(row)+'.json') and
                proposal.get('query')==query, 'Original teacher proposal must bind the measured query')
            h,t,relative,mask=old._oracle_query(proposal,meta,row)
            preview=proposal['preview'];snapshot=record['retained_interpolator_snapshot']
            need(preview.get('input_state')==snapshot and preview.get('input_sha256')==rt.digest(snapshot) and
                outcome.get('valid_proposed_cells')==int(np.count_nonzero(~mask)), 'Original teacher command seam/mask/count differs')
            value=dict(outcome=outcome,query_ref=query_refs[n],proposal_ref=pr,history=h,times=t,relative=relative,mask=mask)
            accepted.append(value);by_row[row]=value
        else:
            need(isinstance(outcome.get('rejection'),str) and bool(outcome['rejection']) and len(outcome['rejection'])<=1200,
                'Bounded original rejection reason required')
    need(oracle.get('accepted_proposals')==root.get('proposals_accepted')==len(accepted) and
        root.get('proposal_rejections')==len(rows)-len(accepted), 'Original accepted/rejected proposal denominators differ')
    arrays=_archive(inputs.oracle_arrays,ORACLE_KEYS)
    shapes=((len(accepted),2,17),(len(accepted),2),(len(accepted),),(len(accepted),20,7),(len(accepted),20))
    for key,shape in zip(ORACLE_KEYS,shapes):
        a=arrays[key];need(a.shape==shape,'Original oracle binary shape differs: '+key)
        need(a.dtype==np.dtype(bool) if key=='proposed_is_pad' else a.dtype.kind in ('iu' if key=='source_physics_rows' else 'fiu'),
            'Original oracle binary dtype differs: '+key)
        need(np.isfinite(a).all(),'Finite original oracle binary required')
    need(arrays['source_physics_rows'].tolist()==[v['outcome']['physics_row'] for v in accepted], 'Original oracle accepted binary row order differs')
    for i,value in enumerate(accepted):
        for key,item in zip(ORACLE_KEYS,(value['history'],value['times'],value['outcome']['physics_row'],value['relative'],value['mask'])):
            need(np.array_equal(arrays[key][i],item), 'Original teacher binary/query/proposal alignment differs: '+key)
    sources=[];parts={k:[] for k in ARRAY_KEYS};ledger=deepcopy(header['final']['slots']);seen=set()
    for slot,entry in zip(header['slots'],ledger):
        row=slot['physics_row'];outcome=next((o for o in outcomes if o['physics_row']==row),None)
        need(entry.get('proposal_attempted') is (outcome is not None) and entry.get('proposal_accepted')==(outcome['proposal_accepted'] if outcome else None) and
            entry.get('proposal_outcome')==outcome, 'Every missing/duplicate/rejected/reserved outcome must survive')
        if outcome and not outcome['proposal_accepted']:need(entry.get('rejection')==outcome['rejection'], 'Original rejected proposal reason lost')
        if slot['effective_role']!='potential_train' or row not in by_row:continue
        need(row not in seen,'Duplicate optimization origin forbidden');seen.add(row);value=by_row[row]
        source=dict(schema=SOURCE_SCHEMA,kind='student_oracle_unexecuted',subject_id=inputs.subject_id,split='train',
            target_id=inputs.target_id,pose_id=inputs.pose_id,metadata=deepcopy(meta),
            observation_origin=old.ORIGINS['student_oracle_unexecuted'],frozen_physics_row=row,
            windows=[dict(input_index=row//12,phase='APPROACH')],future_measured_histories=False,generated_static_input=False,
            source_episode_verdict_unchanged=True,original_source_success=False,role='potential_train',
            supervision='original teacher full computed prefix up to20; measured H2 only; future unexecuted',
            inherited_native_authentication='pinned producer original source28 native H2 and command seam proof',
            bindings=dict(source=frozen['source']['config'],owner=frozen['source']['owner'],child=frozen['source']['child'],
                native=frozen['source']['native'],request=inputs.frozen_request,query=value['query_ref'],proposal=value['proposal_ref'],
                oracle=inputs.oracle_complete,producer=inputs.producer_complete,producer_operator=inputs.producer_operator,
                frozen_manifest=inputs.source28_manifest,generator=frozen['generator']),**old.FLAGS)
        local=dict(source_histories=value['history'][None],source_times=value['times'][None],relative_setters=value['relative'][None],
            setter_is_pad=value['mask'][None],source_indices=np.array([len(sources)]))
        for key in ARRAY_KEYS:
            cast=local[key].astype(parent_arrays[key].dtype)
            need(np.array_equal(cast,local[key]),'New rows cannot lose information in exact parent storage dtype')
            parts[key].append(cast)
        source['sample_sha256']=old._row_digest(local,[0]);source['source_id']=old._source_id(source)
        entry['optimization_source_id']=source['source_id'];sources.append(source)
    potential=sum(s['effective_role']=='potential_train' and s.get('proposal_accepted') is True for s in ledger)
    reserved=len({s['physics_row'] for s in ledger if s['effective_role']=='reserved_diagnostic' and s.get('proposal_accepted') is True})
    need(root.get('potential_train_distinct_accepted')==potential==len(sources) and root.get('reserved_distinct_accepted')==reserved,
        'Fixed potential/reserved accepted role counts differ')
    arrays={k:np.concatenate(v) if v else np.empty((0,*parent_arrays[k].shape[1:]),dtype=parent_arrays[k].dtype) for k,v in parts.items()}
    return sources,arrays,ledger


def _admission(admitted):
    return dict(schema=ADMISSION_SCHEMA,computational_optimization_eligible=admitted,
        status='admitted' if admitted else 'rejected_incomplete_four_origin_cohort',expected_training_origins=4,
        oracle_labels='bound_original_teacher_computed_prefix_up_to20',future_measured_histories=False,**old.FLAGS)


def _assemble(inputs,allowed):
    _,parent_meta=_parent_header(inputs,allowed); header=_probe_header(inputs,allowed)
    parent=_archive(inputs.parent_arrays,ARRAY_KEYS)
    old._numeric(parent,parent_meta['sources'],full_oracle_prefix=True)
    need(len(parent['source_indices'])==inputs.parent_rows,'Exact closed V4 parent row count required')
    new_sources,tail,phase_ledger=_additions(header,inputs,allowed,parent)
    sources=deepcopy(parent_meta['sources'])+new_sources
    need(len({s['source_id'] for s in sources})==len(sources),'Repeated source IDs forbidden')
    offset=inputs.parent_sources
    need(offset+len(new_sources)-1<=np.iinfo(parent['source_indices'].dtype).max,'V5 source-index storage overflow')
    tail['source_indices']=tail['source_indices']+offset
    arrays={k:np.concatenate([parent[k],tail[k]]) for k in ARRAY_KEYS}
    proof={k:array_proof(parent[k]) for k in ARRAY_KEYS}
    need(all(array_proof(arrays[k][:inputs.parent_rows])==proof[k] for k in ARRAY_KEYS),'Exact parent byte prefix changed')
    admission=_admission(len(new_sources)==4)
    samples=dict(schema=SCHEMA,sources=sources,source_table_sha256=rt.digest(sources),allowed_train_ids=sorted(allowed),
        optimization_admission=admission,input_profile=_profile(inputs),**old.FLAGS,**arrays)
    old._numeric(samples,sources,full_oracle_prefix=True)
    provenance=dict(inputs=asdict(inputs),input_profile=_profile(inputs),data_code=code_binding(),
        parent_admission=dict(mode='inherited_hash_bound_closed_snapshot',fresh_historical_factory_replay=False,
            complete=inputs.parent_complete,metadata=inputs.parent_metadata,arrays=inputs.parent_arrays,
            inherited_optimization_admission=deepcopy(parent_meta['optimization_admission']),
            inherited_provenance=deepcopy(parent_meta['provenance'])),ordered_parent_proof=proof,
        ordered_parent_metadata_proof={k:rt.digest(parent_meta[k]) for k in ('sources','slot_ledger','request_refs')},
        producer_admission=dict(mode='inherited_hash_bound_source28_native_query_authentication',
            new_live_query_performed=False,producer=inputs.producer_complete,manifest=inputs.producer_manifest,
            source28=header['root']['source28'],source=header['frozen']['source'],model=header['frozen']['model']))
    counts=dict(inherited_sources=inputs.parent_sources,inherited_origins=inputs.parent_rows,inherited_attempted_slots=inputs.parent_slots,
        new_measured_query_origins=len(new_sources),total_new_measured_query_origins=len(header['rows']),
        optimization_measured_query_origins=len(new_sources),new_valid_computed_future_cells=int(np.count_nonzero(~tail['setter_is_pad'])),
        reserved_diagnostic_sources=len({s['physics_row'] for s in phase_ledger if s['effective_role']=='reserved_diagnostic' and s['physics_row'] is not None}),
        semantic_slot_denominator=6,potential_train_slot_denominator=4,reserved_diagnostic_slot_denominator=2,
        attempted_distinct_queries=len(header['rows']),new_physical_demonstrations=0,
        native_bank='external unchanged original29 episodes:25 TRAIN four IDs;4 selectionVAL one ID',
        auxiliary_normalization_fitted=False,optimization_admitted=admission['computational_optimization_eligible'])
    return dict(samples=samples,sources=sources,slot_ledger=deepcopy(parent_meta['slot_ledger'])+deepcopy(phase_ledger),
        phase_slot_ledger=phase_ledger,request_refs=deepcopy(parent_meta['request_refs'])+[deepcopy(inputs.frozen_request)],
        provenance=provenance,counts=counts)


def _metadata(value):
    meta={k:deepcopy(v) for k,v in value['samples'].items() if k not in ARRAY_KEYS}
    meta.update({k:deepcopy(value[k]) for k in ('slot_ledger','phase_slot_ledger','request_refs','provenance','counts')})
    return meta


def export_from_phase_probe(output,*,allowed_train_ids,inputs=None):
    """Create one fresh V5 preparation, preserving incomplete six-slot ledgers.

    Missing/rejected/duplicate training origins produce a closed artifact whose
    explicit admission is false. Reserved roles never enter arrays or batches.
    LocalFixtureInputs are visibly synthetic; omitted inputs use exact real pins.
    """
    inputs=_inputs(inputs);output=Path(output)
    need(not output.exists(),'Fresh V5 preparation directory required; no retry/resume')
    value=_assemble(inputs,allowed_train_ids); output.mkdir(parents=True)
    path=output/'AUXILIARY_ARRAYS.npz'
    with path.open('xb') as stream:np.savez_compressed(stream,**{k:value['samples'][k] for k in ARRAY_KEYS})
    bound=rt.write_new(output/'METADATA.json',_metadata(value))
    return rt.write_new(output/'COMPLETE.json',dict(schema=SCHEMA,complete=True,metadata=bound,arrays=rt.reference(path),
        metadata_sha256=bound['sha256'],optimization_admission=deepcopy(value['samples']['optimization_admission']),
        input_profile=_profile(inputs),**old.FLAGS))


def metadata(complete_ref,allowed_train_ids,*,inputs=None):
    """Hash/identity admission metadata; no NPZ decoding or historical replay."""
    inputs=_inputs(inputs);complete=rt.read_bound(complete_ref)
    need(complete.get('schema')==SCHEMA and complete.get('complete') is True and complete.get('input_profile')==_profile(inputs),
        'Complete explicitly typed V5 preparation required')
    _flags(complete,old.FLAGS);meta=rt.read_bound(complete['metadata']);_identities(meta.get('sources'),allowed_train_ids)
    need(meta.get('schema')==SCHEMA and meta.get('source_table_sha256')==rt.digest(meta['sources']) and
        meta.get('allowed_train_ids')==sorted(allowed_train_ids) and meta.get('input_profile')==_profile(inputs) and
        complete.get('metadata_sha256')==complete['metadata']['sha256'] and
        complete.get('optimization_admission')==meta.get('optimization_admission') and
        meta.get('optimization_admission') in (_admission(True),_admission(False)), 'Explicit V5 schema/admission/source table differs')
    _flags(meta,old.FLAGS);p=meta.get('provenance',{})
    need(p.get('inputs')==asdict(inputs) and p.get('input_profile')==_profile(inputs) and p.get('data_code')==code_binding(),
        'V5 input/code binding differs')
    _,parent=_parent_header(inputs,allowed_train_ids);_probe_header(inputs,allowed_train_ids)
    for key in ('sources','slot_ledger','request_refs'):
        need(rt.digest(meta[key][:len(parent[key])])==rt.digest(parent[key]),'Exact inherited parent metadata prefix differs: '+key)
    rt.bound_path(complete['arrays'])
    return complete,meta


def load_preparation(complete_ref,allowed_train_ids,*,inputs=None):
    """Deterministic closed-input verification, without reopening old factories."""
    inputs=_inputs(inputs);complete,meta=metadata(complete_ref,allowed_train_ids,inputs=inputs)
    expected=_assemble(inputs,allowed_train_ids)
    need(rt.digest(meta)==rt.digest(_metadata(expected)), 'V5 source/provenance/slot/count ledger differs from immutable inputs')
    arrays=_archive(complete['arrays'],ARRAY_KEYS)
    for key in ARRAY_KEYS:need(array_proof(arrays[key])==array_proof(expected['samples'][key]),'Exact V5 array bytes differ: '+key)
    expected['samples'].update(arrays)
    return expected


def pack_batch(samples,real_native_stats,indices):
    """Normalize solely with external native TRAIN stats; ignore poisoned padding."""
    need(samples.get('schema')==SCHEMA and samples.get('optimization_admission')==_admission(True),
        'Explicit optimization-admitted complete V5 samples required')
    stats=rt.validate_normalization(real_native_stats);ids=np.asarray(indices)
    need(ids.ndim==1 and ids.dtype.kind in 'iu' and np.all(ids>=0) and np.all(ids<len(samples['source_indices'])),
        'Valid V5 batch row indices required')
    h=(samples['source_histories'][ids]-np.asarray(stats['state_mean']))/np.asarray(stats['state_std'])
    mask=samples['setter_is_pad'][ids].copy()
    raw=np.where(mask[...,None],0.,samples['relative_setters'][ids])
    action=(raw-np.asarray(stats['action_mean']))/np.asarray(stats['action_std'])
    need(np.isfinite(h).all() and np.isfinite(action).all(),'Finite valid V5 packed values required')
    return {'observation.state':h[:,1,:7].astype(np.float32),
        'observation.environment_state':np.concatenate([h[:,1,7:],h[:,0]],axis=1).astype(np.float32),
        'action':action.astype(np.float32),'action_is_pad':mask}


def terminal_pool_inventory(auxiliary,*,allowed_train_ids,expected_strata=()):
    """Pure existing planned/conditional TRAIN inventory near canonical PREPOSE.

    This diagnostic 6mm criterion changes no physical handoff gate or sampler.
    Counts refer to eligible windows, grouped by body, pose and target. Explicit
    requested strata with no eligible window remain visible, as do invalid plans.
    """
    sources=auxiliary['sources'];_identities(sources,allowed_train_ids)
    samples=auxiliary['samples'];rows=[];invalid=[];present=set();requested=set();counts={k:{} for k in ('body','pose','target','body_pose_target')}
    kinds=('planned_ideal_tracking','conditional_ideal_continuation')
    for index,source in enumerate(sources):
        if source.get('kind') not in kinds:continue
        stratum=(source['subject_id'],source['pose_id'],source['target_id'])
        requested.add(stratum)
        try:
            gap,_,_=_plan(source['metadata'],source)
            indices=np.flatnonzero(np.asarray(samples['source_indices'])==index)
            need(len(indices)==len(source['windows']),'Source/window inventory differs')
            for row in indices:
                position=np.asarray(samples['source_histories'][row,-1,7:10],float)
                need(position.shape==(3,) and np.isfinite(position).all(),'Finite existing skin-relative state17 position required')
                delta=position.copy();delta[2]+=gap;error=float(np.linalg.norm(delta))
                if error>.006:continue
                rows.append(dict(row_index=int(row),source_index=index,source_id=source['source_id'],subject_id=stratum[0],
                    pose_id=stratum[1],target_id=stratum[2],actual_gap_m=gap,precontact_error_norm_m=error,
                    skin_relative_norm_m=float(np.linalg.norm(position))))
                present.add(stratum)
                for key,value in zip(counts,(*stratum,'|'.join(stratum))):counts[key][value]=counts[key].get(value,0)+1
        except (ValueError,KeyError,TypeError,OSError) as error:
            # Fixed bounded message; never exception repr or opaque payload text.
            invalid.append(dict(stratum=list(stratum),source_index=index,error_type=type(error).__name__,reason='Invalid bound source plan/reference/window'))
    expected=requested
    for stratum in expected_strata:
        need(isinstance(stratum,(tuple,list)) and len(stratum)==3 and all(isinstance(x,str) and x for x in stratum),
            'Explicit body/pose/target strata required')
        expected.add(tuple(stratum))
    return dict(schema='m2_act_diverse_terminal_pool_inventory_v5',eligible_row_indices=[r['row_index'] for r in rows],rows=rows,
        bucket_counts=counts,missing_strata=[list(s) for s in sorted(expected-present)],invalid_strata=invalid,
        diagnostic_threshold_m=.006,coordinate_error='norm([px,py,pz+source_canonical_gap])',
        physical_gate_changed=False,sampler_changed=False,arrays_created=False,normalization_fitted=False)
