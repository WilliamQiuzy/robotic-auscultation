"""Explicit known-goal control reference. No estimator or normal-only roll fit.

The reference is available before motion and contains no route/joint labels.
It describes prescribed-pose control, never visual or sensor robustness.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from . import m2_contract as contract
from .geometry_epoch import require_current

TASK_ID = 'm2_given_contact_pose_control_v1'
TARGET_SOURCE = 'given_contact_pose'
CONTRACT_ID = 'm2_given_contact_state17_raw_target_h20_e10_v2'
OBSERVATION_ID = 'm2_given_contact_state17_history2_v1'
REFERENCE_SCHEMA = 'm2_given_contact_pose_reference_v1'
CAPTURE_SCHEMA = 'm2_given_contact_compact_expert_capture_v2'
STAGE_RULE = 'm2_given_reference_tcp_distance_latch_9cm_v1'
NATIVE_SOURCE = 'native_measured_q_given_contact_pose_v1'
L2_ROLES = frozenset(('given_l2_motion',))


def need(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def bound_json(ref):
    need(isinstance(ref, dict) and set(ref) == {'file', 'sha256'}, 'Explicit hash-bound source required')
    data = Path(ref['file']).read_bytes()
    need(hashlib.sha256(data).hexdigest() == ref['sha256'], 'Reference source hash changed')
    return json.loads(data)


def observation_contract():
    result = contract.observation_contract()
    result.update(schema=OBSERVATION_ID, route=OBSERVATION_ID, stage_rule_id=STAGE_RULE,
                  task_id=TASK_ID, target_source=TARGET_SOURCE,
                  confidence_semantics='available prescribed reference; not visual accuracy')
    return result


@dataclass(frozen=True)
class GivenReference:
    """Canonical owned bytes prevent nested array/dict mutation."""
    payload: bytes

    def __post_init__(self):
        value = json.loads(self.payload)
        need(value.get('schema') == REFERENCE_SCHEMA and value.get('task_id') == TASK_ID and
             value.get('target_source') == TARGET_SOURCE and value.get('revision') == 0,
             'Invalid prescribed reference schema/revision')
        pose = contract._pose(value['reference_pose_base'], 'given reference')
        base = contract._pose(value['world_from_base'], 'reference base')
        normal = contract._array(value['normal_base'], (3,), 'reference normal')
        point = contract._array(value['point_world_m'], (3,), 'reference point')
        need(np.allclose(normal, -pose[:3, 2], atol=1e-9, rtol=0) and
             np.allclose((base @ pose)[:3, 3], point, atol=1e-10, rtol=0), 'Reference frame disagreement')
        need(value.get('frame') == 'openarm_left_link0' and value.get('position_unit') == 'm' and
             value.get('rotation_unit') == 'rotation_matrix' and value.get('translation_noise') == 'absent_named_known_goal_task',
             'Reference units/source semantics mismatch')
        contract._scalar(value['available_at_s'], 'reference availability')
        need(set(value['sources']) == {'task_manifest', 'fresh_plan', 'patient_receipt'}, 'Missing reference provenance')
        object.__setattr__(self, 'payload', canonical(value))

    @classmethod
    def from_snapshot(cls, value):
        return cls(canonical(value))

    def snapshot(self):
        return json.loads(self.payload)

    @property
    def sha256(self):
        return hashlib.sha256(self.payload).hexdigest()

    @property
    def pose_base(self):
        value = np.asarray(self.snapshot()['reference_pose_base'], dtype=np.float64)
        return np.frombuffer(value.tobytes(), dtype=np.float64).reshape(4, 4)

    @property
    def normal_base(self):
        value = np.asarray(self.snapshot()['normal_base'], dtype=np.float64)
        return np.frombuffer(value.tobytes(), dtype=np.float64)

    def verify_sources(self):
        for ref in self.snapshot()['sources'].values():
            bound_json(ref)


def build_given_reference(task_point, fresh_plan_ref, patient_receipt_ref, world_from_base, available_at_s):
    """Bind user surface point, installed patient, fresh selected FULL command R."""
    plan = bound_json(fresh_plan_ref); require_current(plan)
    patient = bound_json(patient_receipt_ref)
    manifest_ref = task_point['manifest_ref']; manifest = bound_json(manifest_ref)
    name = task_point['name']
    need(name in [s+str(i) for s in 'LR' for i in range(1, 6)] and plan.get('name') == name and
         plan.get('success') is True, 'Fresh successful target plan required')
    need(patient.get('non_patient_scene_unchanged') is True and
         plan.get('patient_provenance', {}).get('binding') == patient_receipt_ref and
         all(plan['patient_provenance'].get(k) == patient.get(k) for k in ('identity', 'asset_sha256')),
         'Fresh plan installed patient binding mismatch')
    targets = [x for x in manifest['targets'] if x['name'] == name]
    installed = [x for x in patient['targets'] if x['name'] == name]
    need(len(targets) == len(installed) == 1, 'Unique task point and installed patient point required')
    point = contract._array(task_point['position_world_m'], (3,), 'surface point')
    raw_normal = contract._array(task_point['normal_world'], (3,), 'surface normal')
    for target in [targets[0], installed[0]]:
        need(np.allclose(target['position_world_m'], point, atol=1e-10, rtol=0) and
             np.allclose(target['normal_world'], raw_normal, atol=1e-10, rtol=0), 'Task point manifest disagreement')
    need(np.allclose(plan['position_world'], point, atol=1e-10, rtol=0) and
         np.allclose(plan['normal_world'], raw_normal, atol=1e-10, rtol=0), 'Fresh plan task point disagreement')
    base = contract._pose(world_from_base, 'world_from_base')
    need(np.allclose(base, plan['world_from_left_base'], atol=1e-9, rtol=0), 'Fresh plan base disagreement')
    desired = np.eye(4); desired[:3, :3] = plan['command_rotation_world']; desired[:3, 3] = point
    desired = contract._pose(desired, 'fresh selected full rotation')
    need(np.allclose(-desired[:3, 2], plan['command_normal_world'], atol=1e-9, rtol=0),
         'Selected rotation and command normal disagree')
    pose = np.linalg.inv(base) @ desired
    return GivenReference(canonical(dict(schema=REFERENCE_SCHEMA, task_id=TASK_ID, target_source=TARGET_SOURCE,
        revision=0, target_id=name, reference_pose_base=pose.tolist(), normal_base=(-pose[:3, 2]).tolist(),
        point_world_m=point.tolist(), world_from_base=base.tolist(), frame='openarm_left_link0',
        position_unit='m', rotation_unit='rotation_matrix', translation_noise='absent_named_known_goal_task',
        available_at_s=contract._scalar(available_at_s, 'reference availability'),
        sources=dict(task_manifest=deepcopy(manifest_ref), fresh_plan=deepcopy(fresh_plan_ref),
                     patient_receipt=deepcopy(patient_receipt_ref)))))


def require_route(metadata):
    from .m2_given_contact_intent import ACTION_SEMANTICS_ID
    need(metadata.get('action_semantics_id') == ACTION_SEMANTICS_ID, 'Given raw target action semantics required')
    need(metadata.get('task_id') == TASK_ID and metadata.get('target_source') == TARGET_SOURCE and
         metadata.get('contract_id') == CONTRACT_ID and metadata.get('observation_contract') == observation_contract(),
         'Explicit given-contact task/source/observation contract required')
    ref = GivenReference.from_snapshot(metadata['given_reference'])
    need(metadata.get('reference_sha256') == ref.sha256 and metadata.get('reference_revision') == 0,
         'Given reference bytes/revision mismatch')
    if 'target_id' in metadata:
        need(metadata['target_id'] == ref.snapshot()['target_id'], 'Given reference target mismatch')
    if 'installed_patient_receipt_sha256' in metadata:
        need(metadata['installed_patient_receipt_sha256'] == ref.snapshot()['sources']['patient_receipt']['sha256'],
             'Given reference installed patient receipt mismatch')
    return ref


def route_metadata(base_metadata, reference):
    from .m2_given_contact_intent import ACTION_SEMANTICS_ID
    need(isinstance(reference, GivenReference), 'Immutable given reference required')
    result = deepcopy(base_metadata)
    result.update(task_id=TASK_ID, target_source=TARGET_SOURCE, contract_id=CONTRACT_ID,
        observation_contract=observation_contract(), given_reference=reference.snapshot(),
        reference_sha256=reference.sha256, reference_revision=0, action_semantics_id=ACTION_SEMANTICS_ID)
    contract.require_metadata(result)
    return result


class GivenPoseProvider:
    """Same reference on every query, real measured q and causal distance latch."""
    def __init__(self, reference_ref, episode_id, metadata):
        self.reference = (reference_ref if isinstance(reference_ref, GivenReference) else
                          GivenReference.from_snapshot(bound_json(reference_ref)))
        self.metadata = deepcopy(metadata)
        need(require_route(self.metadata).sha256 == self.reference.sha256, 'Provider reference mismatch')
        need(isinstance(episode_id, str) and bool(episode_id), 'Episode identity required')
        self.episode_id = episode_id; self.bound = None
        from .m2_stage import CausalStage
        self.stage = CausalStage()

    def bind_before_motion(self, controller):
        need(not controller.env.rows and self.bound is None, 'Reference must be bound before first motion')
        now = contract._scalar(controller.env.state()['time_s'], 'binding clock')
        need(self.reference.snapshot()['available_at_s'] <= now, 'Reference not yet available before motion')
        self.reference.verify_sources()
        need(np.allclose(controller.env.base, self.reference.snapshot()['world_from_base'],
                         atol=1e-9, rtol=0), 'Actual base differs from reference')
        self.bound = dict(bound_at_s=now, physics_row=0, reference_sha256=self.reference.sha256)

    def __call__(self, controller, state):
        need(self.bound is not None, 'Given reference must be bound before first motion')
        ref = require_route(self.metadata)
        need(ref.sha256 == self.bound['reference_sha256'] and
             controller.target.name == ref.snapshot()['target_id'], 'Given reference/target changed')
        need(np.allclose(controller.env.base, ref.snapshot()['world_from_base'], atol=1e-9, rtol=0),
             'Actual base differs from reference')
        stamp = contract._scalar(state['time_s'], 'given observation clock')
        need(stamp >= self.bound['bound_at_s'], 'Given reference clock moved backwards')
        q = contract._array(state['q'], (7,), 'given measured q')
        encoded = contract.encode_state(q, ref.pose_base, 1., metadata=self.metadata)
        phase = self.stage.update(float(np.linalg.norm(encoded[7:10])), stamp)
        return dict(episode_id=self.episode_id, target_id=controller.target.name, revision=0, phase=phase,
                    observed_at_s=stamp, measured_q=q, estimated_target_base=ref.pose_base,
                    confidence=1., source_id='given-contact-'+ref.sha256)

    def snapshot(self):
        return dict(reference=self.reference.snapshot(), reference_sha256=self.reference.sha256,
                    availability_binding=deepcopy(self.bound), task_id=TASK_ID, state_only=True,
                    visual_or_sensor_robustness_claim=False)
