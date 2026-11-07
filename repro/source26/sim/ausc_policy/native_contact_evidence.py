"""Native contact diagnostics; no point correction or acceptance decision.

PhysX's Python callback exposes current transforms, not contact-event poses.
Recorded samples therefore make no contact-generation-pose claim. Face indices
remain native indices: a convex cooked hull's face is not an authored triangle.
"""
import numpy as np

from .retainer_contact import is_retainer_patient_pair


def contact_point_record(data, colliders, *, actors=None, callback_index=None):
    record = dict(colliders=list(colliders), position_world=list(data.position),
                  impulse_Ns=list(data.impulse), separation_m=float(data.separation),
                  actors=None if actors is None else list(actors),
                  callback_index=callback_index, native_fields_missing=[],native_field_errors={})
    for source, destination in (('normal', 'normal_world'),
                                ('face_index0', 'face_index0'),
                                ('face_index1', 'face_index1')):
        value = None
        try:
            raw = getattr(data, source, None)
            if raw is not None:
                value = list(raw) if source == 'normal' else int(raw)
        except Exception as error:
            record['native_field_errors'][source] = repr(error)[:256]
        if value is None:
            record[destination] = None
            record['native_fields_missing'].append(source)
        else:
            record[destination] = value
        if source != 'normal':
            record[destination+'_available'] = value is not None and value != 4294967295
    return record


def contact_header_record(header, path_from_id):
    record = dict(actors=None,event_type=None,stage_id=None,header_field_errors={})
    for key,read in (
        ('actors',lambda:[str(path_from_id(header.actor0)),str(path_from_id(header.actor1))]),
        ('event_type',lambda:str(header.type)),('stage_id',lambda:int(header.stage_id))):
        try:record[key] = read()
        except Exception as error:record['header_field_errors'][key] = repr(error)[:256]
    return record


def native_pose_sample(reader, path, sample):
    result = dict(sample=sample, source='omni.physx.IPhysx.get_rigidbody_transformation',
        path=path, available=False, world_from_carriage=None,
        contact_generation_pose_claimed=False)
    try:
        raw = reader(path)
        if not raw.get('ret_val'):
            raise ValueError('Native rigid-body transformation unavailable')
        position = np.asarray(raw['position'], float)
        quat = np.asarray(raw['rotation'], float)
        if (position.shape != (3,) or quat.shape != (4,)
                or not np.isfinite(position).all() or not np.isfinite(quat).all()
                or not np.isclose(quat @ quat, 1., rtol=0, atol=1e-6)):
            raise ValueError('Invalid native position or xyzw quaternion')
        from scipy.spatial.transform import Rotation
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat(quat).as_matrix()
        matrix[:3, 3] = position
        result.update(available=True, world_from_carriage=matrix.tolist(),
                      position_world=position.tolist(), quaternion_xyzw=quat.tolist())
    except Exception as error:
        result['error'] = repr(error)[:512]
    return result


def latch_first_positive_contact(env, world_from_carriage):
    if getattr(env, '_retainer_first_contact_latched', False):
        return
    if not any(is_retainer_patient_pair(*point['colliders'])
               and np.linalg.norm(point['impulse_Ns']) > 0.
               for point in env.contact_points):
        return
    env._retainer_first_contact_latched = True
    from .retainer_failure import _snapshot, _bounded, _encode, MAX_BYTES
    try:
        import json
        document = json.loads(_snapshot(env, world_from_carriage))
        document.update(schema='native_retainer_first_contact_v1',
                        event_kind='first_positive_retainer_contact',
                        trigger='first_positive_retainer_contact')
        env._retainer_first_contact_payload = _bounded(document, MAX_BYTES)
    except Exception as error:
        env._retainer_first_contact_payload = _encode(dict(
            schema='native_retainer_first_contact_v1',
            snapshot_error=repr(error)[:512], missing_fields=['snapshot_failed'],
            counts=dict(retainer_patient_points=None), points=[], points_omitted=None,
            truncated=False, evidence_only=True, physical_gates_unchanged=True,
            training_eligible=False))


def save_first_positive_contact(env, output, *, capture=None):
    from .retainer_failure import save_retainer_snapshot
    return save_retainer_snapshot(env, output, capture=capture,
        payload_attr='_retainer_first_contact_payload',
        receipt_attr='retainer_first_contact_evidence',
        file_name='retainer_first_contact.json', schema='native_retainer_first_contact_v1')
