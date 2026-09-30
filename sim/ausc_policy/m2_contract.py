"""M2 L2 numerical and provenance contract for the approved 5 mm tool.

L2 receives RGB plus this 17-value state. Depth informs the estimated target;
Hall/ADC and velocities remain available to L3 and safety, never to this state.
This module does not implement a depth encoder, IK, an executor, or a trainer.
Its provenance checks bind files to a receipt, not certify their physics or
authorize bulk data generation. The separate expert qualification gate remains.

All poses are rigid base-from-frame matrices, translations in metres. Rotation
6D is the first column followed by the second column (not row-major flattening).
Actions label t + (k+1)/20 seconds, not the observation at t. Relative-joint
blocks use the measured q at block start once, with no per-step reanchoring.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .geometry_epoch import require_current
from ..auscultation_expert_layout_v1.kinematics import OpenArmKinematics


CONTRACT_ID = 'm2_l2_5mm_state17_h20_e10_v1'
HORIZON = 20
EXECUTION_PREFIX = 10
PERIOD_S = .05
STATE_DIM = 17
ACTION_DIM = 7
UNCOMPRESSED_TCP_M = .132
IDENTITY_SPLIT_SHA256 = '0e50ebce15f6e2816a914494cdf721acc6029db7d59cd0c9caeab0685a523699'
IDENTITY_SPLIT_PATH = Path(__file__).resolve().parents[2] / 'configs/ausc_policy/m2_identity_split_20260930.json'
# Frozen 20260930 HuMMan identity audit: inconsistent shape betas across sequences.
QUARANTINED_IDENTITIES = frozenset(('p000452', 'p000471'))
REQUIRED_ARTIFACTS = ('scene', 'body', 'rgb', 'depth', 'depth_valid', 'joints',
                      'hall_adc', 'physics_trace', 'target_estimates', 'expert_queries')
TARGET_SOURCES = frozenset(('rgbd_surface_tracking', 'gt_plus_measured_noise'))
JOINT_ARMS = frozenset(('block_relative_joint', 'absolute_joint'))
L2_PHASES = frozenset(('FREE', 'APPROACH'))
_KIN = OpenArmKinematics()


def _array(value, shape, name):
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in 'fiu' or not np.isfinite(raw).all():
        raise ValueError(f'{name} must be finite numeric {shape}')
    return raw.astype(np.float64, copy=True)


def _pose(value, name):
    value = _array(value, (4, 4), name)
    r = value[:3, :3]
    if (not np.allclose(value[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-7, rtol=0)
            or not np.isclose(np.linalg.det(r), 1., atol=1e-7, rtol=0)):
        raise ValueError(name + ' must be a proper rigid transform')
    return value


def _scalar(value, name, *, minimum=0., maximum=np.inf):
    if (not isinstance(value, (int, float, np.integer, np.floating))
            or isinstance(value, (bool, np.bool_)) or not np.isfinite(value)
            or not minimum <= value <= maximum):
        raise ValueError(name + ' is invalid')
    return float(value)


def require_metadata(metadata):
    """Reject legacy 41D/single-step and unbound or truth-target records."""
    require_current(metadata)
    expected = dict(contract_id=CONTRACT_ID, state_dim=STATE_DIM, horizon=HORIZON,
                    execution_prefix=EXECUTION_PREFIX, control_hz=20, joint_unit='rad',
                    position_unit='m', target_pose_frame='openarm_left_link0')
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError('M2 contract mismatch: ' + key)
    if metadata.get('target_source') not in TARGET_SOURCES:
        raise ValueError('Target must be an explicitly labeled deployment estimate or measured-noise simulation estimate')


def target_relative_pose(tcp_base, estimated_target_base):
    """Return estimated-target-from-TCP; both inputs must use the same frame."""
    tcp = _pose(tcp_base, 'tcp_base')
    target = _pose(estimated_target_base, 'estimated_target_base')
    relative = np.eye(4)
    relative[:3, :3] = target[:3, :3].T @ tcp[:3, :3]
    relative[:3, 3] = target[:3, :3].T @ (tcp[:3, 3] - target[:3, 3])
    return relative


def encode_state(measured_q, estimated_target_base, confidence, *, metadata):
    """q7 + target-relative TCP position3/rotation6D + estimate confidence1.

    FK uses an explicit uncompressed 132 mm wrist offset; it deliberately does
    not call the legacy 126.2 mm ``kin.tcp`` or infer compression from Hall.
    Target ID and phase belong in the prompt, never in this numerical state.
    """
    require_metadata(metadata)
    q = _array(measured_q, (7,), 'measured_q')
    confidence = _scalar(confidence, 'confidence', maximum=1.)
    offset = np.eye(4)
    offset[2, 3] = UNCOMPRESSED_TCP_M
    tcp = _KIN.fk(q) @ offset
    relative = target_relative_pose(tcp, estimated_target_base)
    return np.r_[q, relative[:3, 3], relative[:3, 0], relative[:3, 1], confidence]


@dataclass(frozen=True)
class JointActionBlock:
    """An immutable numerical block; decode is idempotent with its fixed origin."""
    arm: str
    values: np.ndarray
    observed_q: np.ndarray
    metadata: object

    def __post_init__(self):
        if self.arm not in JOINT_ARMS:
            raise ValueError('Unknown joint action arm')
        require_metadata(self.metadata)
        values = _array(self.values, (HORIZON, ACTION_DIM), 'actions')
        q = _array(self.observed_q, (7,), 'observed_q')
        # Immutable bytes own the storage: setflags(write=True) cannot reopen
        # a mutation path to the fixed block origin or encoded action values.
        values = np.frombuffer(values.tobytes(), dtype=np.float64).reshape(values.shape)
        q = np.frombuffer(q.tobytes(), dtype=np.float64).reshape(q.shape)
        object.__setattr__(self, 'values', values)
        object.__setattr__(self, 'observed_q', q)
        object.__setattr__(self, 'metadata', MappingProxyType(dict(self.metadata)))


def encode_joint_chunk(absolute_commands, observed_q, *, arm, metadata):
    """Encode raw-radian H20 labels; neither clipping nor dynamics shaping."""
    require_metadata(metadata)
    commands = _array(absolute_commands, (HORIZON, ACTION_DIM), 'absolute_commands')
    q = _array(observed_q, (7,), 'observed_q')
    if arm not in JOINT_ARMS:
        raise ValueError('Unknown joint action arm')
    values = commands - q if arm == 'block_relative_joint' else commands
    return JointActionBlock(arm, values, q, metadata)


def decode_joint_chunk(block):
    """Return absolute proposals using the stored start q; no current-q input.

    All posture, collision, joint, velocity, force and execution gates remain
    mandatory downstream. Repeated calls return the same absolute proposals.
    """
    if not isinstance(block, JointActionBlock):
        raise ValueError('Expected an encoded JointActionBlock')
    require_metadata(block.metadata)
    if block.arm == 'block_relative_joint':
        return block.values + block.observed_q
    return block.values.copy()


def encode_target_pose_chunk(tcp_commands_base, *, metadata):
    """Reserved D7 arm: pose6 + elbow psi requires a shared mathematical definition.

    The current spec has no psi reference plane, sign, singularity handling or
    matching IK constraint. Encoding an arbitrary seventh scalar would hide
    that gap. Define and validate that shared oracle/executor contract first.
    """
    require_metadata(metadata)
    raise NotImplementedError('Target-pose arm awaits a defined elbow psi and matching IK constraint')


@dataclass(frozen=True)
class ChunkContext:
    episode_id: str
    target_id: str
    phase: str
    control_owner: str


def _valid_context(context):
    if (not isinstance(context, ChunkContext)
            or any(not isinstance(value, str) or not value.strip()
                   for value in (context.episode_id, context.target_id, context.phase, context.control_owner))
            or context.phase not in L2_PHASES or context.control_owner != 'L2'):
        raise ValueError('Chunk context must remain in L2 FREE/APPROACH')


def validate_chunk_timing(observed_at_s, action_times_s, observation_context,
                          action_contexts, *, metadata):
    """Accept a full future H20 block with one unchanged phase; never pad.

    Each context labels the actual command endpoint. Acquisition must also
    stop a block immediately on any intervening context event; an endpoint-only
    log cannot establish absence of an unrecorded transient event.
    """
    require_metadata(metadata)
    start = _scalar(observed_at_s, 'observed_at_s')
    times = _array(action_times_s, (HORIZON,), 'action_times_s')
    expected = start + np.arange(1, HORIZON+1) * PERIOD_S
    if not np.allclose(times, expected, atol=1e-7, rtol=0):
        raise ValueError('Actions must label future 20 Hz command endpoints t+(k+1)/20')
    _valid_context(observation_context)
    if len(action_contexts) != HORIZON:
        raise ValueError('Incomplete action contexts; padding is forbidden')
    for context in action_contexts:
        _valid_context(context)
        if (context.episode_id != observation_context.episode_id
                or context.target_id != observation_context.target_id):
            raise ValueError('Chunk crosses episode or target change')
        if context.phase != observation_context.phase:
            raise ValueError('Chunk crosses phase change')


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require_episode(metadata, *, expected_split, artifact_root,
                    identity_split_path=IDENTITY_SPLIT_PATH):
    """Bind new Isaac episode sources and frozen identity split before ingestion.

    Returns bound artifact paths. This is not a physics-success verdict; the
    actual trace/collision/sensor validators and >=95% expert gate are separate.
    Raw RGB-D, Hall ADC and robot inputs remain archived even though Hall and
    depth pixels are not numerical L2 state channels.
    """
    require_metadata(metadata)
    if expected_split not in ('train', 'validation', 'test') or metadata.get('split') != expected_split:
        raise ValueError('Unexpected episode split')
    if (metadata.get('source_kind') != 'isaac_live_physics' or metadata.get('replay') is not False
            or not isinstance(metadata.get('episode_id'), str) or not metadata['episode_id'].strip()):
        raise ValueError('Actual Isaac physics source receipt required; replay/static sources are ineligible')
    if (metadata.get('identity_split_sha256') != IDENTITY_SPLIT_SHA256
            or _sha256(identity_split_path) != IDENTITY_SPLIT_SHA256):
        raise ValueError('Frozen identity split mismatch')
    splits = json.loads(Path(identity_split_path).read_text())['splits']
    subject = metadata.get('subject_id')
    memberships = [name for name, identities in splits.items() if subject in identities]
    if memberships != [expected_split] or subject in QUARANTINED_IDENTITIES:
        raise ValueError('Subject is outside expected split or quarantined')
    artifacts = metadata.get('artifacts', {})
    required = REQUIRED_ARTIFACTS + (('noise_model',) if metadata['target_source'] == 'gt_plus_measured_noise' else ())
    root = Path(artifact_root).resolve()
    bound = {}
    for name in required:
        item = artifacts.get(name, {})
        filename, checksum = item.get('path'), item.get('sha256')
        if not isinstance(filename, str) or not isinstance(checksum, str) or len(checksum) != 64:
            raise ValueError('Missing source artifact: ' + name)
        path = (root / filename).resolve()
        if root not in path.parents or not path.is_file() or _sha256(path) != checksum:
            raise ValueError('Source artifact changed or outside bundle: ' + name)
        bound[name] = str(path)
    return bound
