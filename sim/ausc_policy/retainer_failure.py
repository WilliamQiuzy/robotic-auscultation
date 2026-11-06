"""Bounded first-failure native contact evidence; no control or acceptance gates."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from .retainer_contact import RETAINER_PATHS,is_retainer_patient_pair

MAX_BYTES=128*1024
SCHEMA='native_retainer_first_failure_v1'
TRIGGER='device retainer side/back or unreconciled contact'


def _encode(document):
    return json.dumps(document,separators=(',',':'),allow_nan=False).encode()+b'\n'


def _bounded(document,limit):
    """Remove whole point records, retaining counts and explicit missing data."""
    document['artifact_limit_bytes']=limit
    while True:
        count=document['counts']['retainer_patient_points']
        document['points_omitted']=None if count is None else count-len(document['points'])
        document['truncated']=bool(document['points_omitted'])
        payload=_encode(document)
        if len(payload)<=limit:return payload
        if not document['points']:
            raise ValueError('Retainer failure evidence exceeds available auxiliary byte budget')
        document['points'].pop()


def _snapshot(env,world):
    row=env.rows[-1];points=env.contact_points;audit=env.retainer_front_audit
    counts=dict(contact_points=len(points),retainer_patient_points=0,positive_impulse_points=0,
        zero_impulse_points=0,positive_impulse_off_front_points=0,zero_impulse_off_front_points=0)
    document=dict(schema=SCHEMA,trigger=TRIGGER,physics_row=len(env.rows)-1,
        time_s=row.get('time_s'),phase=row.get('phase'),target=row.get('target'),subphase=row.get('subphase'),
        world_from_carriage=None if world is None else np.asarray(world,float).tolist(),last_contacts=deepcopy(env.last_contacts),
        retainer_front_audit={key:deepcopy(value) for key,value in audit.items() if key!='points'},
        audit_point_count=len(audit.get('points',[])),counts=counts,points=[],missing_fields=[],
        zero_impulse_points_are_force_evidence=False,physical_gates_unchanged=True,
        evidence_only=True,training_eligible=False)
    if world is None:document['missing_fields'].append('world_from_carriage')
    if hasattr(env,'contact_tick_evidence'):
        document['contact_tick_evidence']=deepcopy(env.contact_tick_evidence)
    for key in ('time_s','phase','target'):
        if key not in row:document['missing_fields'].append(key)
    used=len(_encode(document))+4096
    for raw_index,point in enumerate(points):
        if not is_retainer_patient_pair(*point['colliders']):continue
        index=counts['retainer_patient_points'];counts['retainer_patient_points']+=1
        positive=bool(np.linalg.norm(point['impulse_Ns'])>0.)
        counts['positive_impulse_points' if positive else 'zero_impulse_points']+=1
        location=audit.get('points',[])[index] if index<len(audit.get('points',[])) else None
        if location is not None and location.get('on_front') is False:
            counts['positive_impulse_off_front_points' if positive else 'zero_impulse_off_front_points']+=1
        missing=[key for key in ('colliders','position_world','impulse_Ns','separation_m') if key not in point]
        if location is None:missing.append('audit_point')
        for key in missing:
            if key not in document['missing_fields']:document['missing_fields'].append(key)
        sector=next(path for path in point['colliders'] if path in RETAINER_PATHS)
        record=dict(raw_contact_index=raw_index,retainer_point_index=index,
            sector=int(sector.rsplit('_',1)[1]),raw=point,audit=location,missing_fields=missing)
        size=len(_encode(record))
        if used+size<=MAX_BYTES:
            document['points'].append(deepcopy(record));used+=size
    if counts['retainer_patient_points']!=document['audit_point_count']:
        document['missing_fields'].append('raw_audit_point_count_mismatch')
    if env.last_contacts.get('retainer',0.)>0. and not counts['retainer_patient_points']:
        document['missing_fields'].append('positive_ledger_without_retainer_points')
    return _bounded(document,MAX_BYTES)


def latch_retainer_failure(env,world_from_carriage):
    """Snapshot once at the existing failed-audit predicate; never raise a new gate."""
    audit=getattr(env,'retainer_front_audit',None)
    if audit is None or audit.get('passed') is True or getattr(env,'_retainer_failure_latched',False):return
    env._retainer_failure_latched=True
    env.retainer_failure_evidence=dict(saved=False,missing=True,status='pending_finalization',limit_bytes=MAX_BYTES)
    try:env._retainer_failure_payload=_snapshot(env,world_from_carriage)
    except Exception as error:
        # Evidence collection cannot replace the initiating physical verdict.
        env._retainer_failure_payload=_encode(dict(schema=SCHEMA,trigger=TRIGGER,
            snapshot_error=repr(error)[:512],missing_fields=['snapshot_failed'],
            counts=dict(retainer_patient_points=None),points=[],points_omitted=None,truncated=False,
            evidence_only=True,physical_gates_unchanged=True,training_eligible=False))


def save_retainer_failure(env,output,*,capture=None):
    """Finalize inside compact's existing auxiliary cap, or the diagnostic output.

    The compact result's existing recursive inventory includes this artifact.
    Its receipt reserve is left intact; there is no additional episode budget.
    """
    return save_retainer_snapshot(env,output,capture=capture,
        payload_attr='_retainer_failure_payload',receipt_attr='retainer_failure_evidence',
        file_name='retainer_failure.json',schema=SCHEMA)


def save_retainer_snapshot(env,output,*,capture,payload_attr,receipt_attr,file_name,schema):
    """Share the same bounded auxiliary budget for native evidence artifacts."""
    payload=getattr(env,payload_attr,None)
    if payload is None:return None
    compact=capture is not None and getattr(capture,'binding',{}).get('profile')=='compact'
    directory=Path(capture.output if compact else output);path=directory/file_name
    receipt=dict(schema=schema,file=str(path),saved=False,missing=True,sha256=None,bytes=0,
        limit_bytes=MAX_BYTES,budget_accounting=('compact_auxiliary_included_in_episode_total'
            if compact else 'bounded_runtime_failure_diagnostic'),training_eligible=False)
    setattr(env,receipt_attr,receipt)
    if compact:capture.receipts[receipt_attr]=receipt
    try:
        limit=MAX_BYTES
        if compact:
            if not capture._closed:raise ValueError('Close compact capture before budgeting failure evidence')
            used=sum(item.stat().st_size for item in directory.iterdir() if item.is_file())
            available=capture.auxiliary_limit-used-capture.AUXILIARY_RESERVE_BYTES
            receipt.update(auxiliary_bytes_before=used,auxiliary_limit_bytes=capture.auxiliary_limit,
                           receipt_reserve_bytes=capture.AUXILIARY_RESERVE_BYTES)
            limit=min(limit,max(0,available))
        document=json.loads(payload)
        document['budget_accounting']=receipt['budget_accounting']
        receipt['limit_bytes']=limit
        payload=_bounded(document,limit)
        with path.open('xb') as stream:stream.write(payload)
        if path.read_bytes()!=payload:raise OSError('Retainer failure artifact readback differs')
        receipt.update(saved=True,missing=False,sha256=hashlib.sha256(payload).hexdigest(),
                       bytes=len(payload),truncated=document['truncated'],missing_fields=document['missing_fields'])
    except Exception as error:
        receipt['error']=repr(error)[:512]
        try:receipt['missing']=not path.is_file()
        except OSError:receipt['existence_check_failed']=True
        raise
    return deepcopy(receipt)
