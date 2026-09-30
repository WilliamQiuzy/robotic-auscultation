"""CPU-only, boundary-aware assembly of M2 numerical joint action samples.

This explicit input schema is not an adapter for raw expert capture: callers
must supply measured observation joints and a separately sourced target
estimate. Source IDs bind the supplied records only; they do not authenticate
artifact hashes, prove measurement provenance, qualify an episode or permit
training. RGB loading, pose+psi actions, trainers and promotion are out of scope.
"""
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields

import numpy as np

from .m2_contract import (ChunkContext, HORIZON, JOINT_ARMS, JointActionBlock,
                         L2_PHASES, encode_joint_chunk, encode_state,
                         require_metadata, validate_chunk_timing)


PHYSICS_HZ = 240
PHYSICS_STEPS_PER_ACTION = 12
FUTURE_PHYSICS_ROWS = HORIZON*PHYSICS_STEPS_PER_ACTION
TIMESTAMP_TOLERANCE_S = 1e-9  # Floating point clock serialization, never resampling.
TARGET_IDS = frozenset(side+str(index) for side in 'LR' for index in range(1, 6))


@dataclass(frozen=True)
class M2Observation:
    source_id: str
    physics_row: int
    time_s: float
    measured_q: object
    estimated_target_base: object
    target_confidence: float
    target_source: str
    context: ChunkContext


@dataclass(frozen=True)
class PhysicsCommand:
    source_id: str
    physics_row: int
    time_s: float
    q_command: object
    context: ChunkContext


@dataclass(frozen=True)
class SourceRow:
    source_id: str
    physics_row: int
    time_s: float
    context: ChunkContext


@dataclass(frozen=True)
class M2JointSample:
    state: np.ndarray
    action_block: JointActionBlock
    observation_source: SourceRow
    command_sources: tuple
    action_sources: tuple
    training_eligible: bool = field(default=False, init=False)
    physical_qualification: bool = field(default=False, init=False)
    artifact_hashes_verified: bool = field(default=False, init=False)
    source_binding_scope: str = field(
        default='references to supplied records only; no artifact or episode qualification', init=False)


def _record(value, record_type):
    if isinstance(value, record_type): return value
    required = {item.name for item in fields(record_type)}
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError(record_type.__name__+' requires explicit fields: '+', '.join(sorted(required)))
    return record_type(**value)


def _context(value):
    context = _record(value, ChunkContext)
    if (any(not isinstance(item, str) or not item.strip() for item in
            (context.episode_id, context.target_id, context.phase, context.control_owner))
            or context.phase not in L2_PHASES or context.control_owner != 'L2'):
        raise ValueError('Invalid L2 context: explicit episode, target, FREE/APPROACH and L2 owner required')
    if context.target_id not in TARGET_IDS:
        raise ValueError('Unknown target ID in L2 context; require L1..L5 or R1..R5')
    return context


def _source(record):
    if not isinstance(record.source_id, str) or not record.source_id.strip():
        raise ValueError('Explicit nonempty source_id required')
    row = record.physics_row
    if (not isinstance(row, (int, np.integer)) or isinstance(row, (bool, np.bool_)) or row < 0):
        raise ValueError('physics_row must be a nonnegative integer')
    time = record.time_s
    if (not isinstance(time, (int, float, np.integer, np.floating))
            or isinstance(time, (bool, np.bool_)) or not np.isfinite(time) or time < 0):
        raise ValueError('Invalid record timestamp time_s')
    return SourceRow(record.source_id, int(row), float(time), _context(record.context))


def _command_q(value):
    q = np.asarray(value)
    if q.shape != (7,) or q.dtype.kind not in 'fiu' or not np.isfinite(q).all():
        raise ValueError('q_command must be a finite numeric seven-joint vector')
    return q.astype(np.float64, copy=True)


def assemble_joint_sample(observation, future_commands, *, arm, metadata):
    """Assemble exactly t+12,...,t+240 from all 240 consecutive future rows.

    ``future_commands`` must contain rows t+1 through t+240 in recorded order;
    t is the supplied observation row. The current command is never a label.
    Every intervening row must preserve the explicit observation context and
    clock. Missing records, interpolation, padding and boundary inference are
    forbidden. The fixed relative-joint origin is measured_q at t, regardless
    of any commanded positions. Only the existing two raw-radian arms exist.
    """
    if not isinstance(metadata, Mapping): raise ValueError('Explicit M2 metadata required')
    require_metadata(metadata)
    if arm not in JOINT_ARMS: raise ValueError('Unknown joint action arm; pose+psi remains deferred')
    observed = _record(observation, M2Observation)
    observation_source = _source(observed)
    if observed.target_source != metadata['target_source']:
        raise ValueError('Observation target source differs from explicit M2 estimate source')
    if ('episode_id' in metadata and metadata['episode_id'] != observation_source.context.episode_id):
        raise ValueError('Metadata episode differs from observation context')
    # Numerical assembly establishes none of these reserved qualifications,
    # including inside the action block's copied metadata.
    sample_metadata = dict(metadata, training_eligible=False,
                           physical_qualification=False, artifact_hashes_verified=False)
    state = encode_state(observed.measured_q, observed.estimated_target_base,
                         observed.target_confidence, metadata=sample_metadata)
    if (not isinstance(future_commands, Sequence) or isinstance(future_commands, (str, bytes))
            or len(future_commands) != FUTURE_PHYSICS_ROWS):
        raise ValueError('Require exactly 240 consecutive future physics command rows; no padding')
    commands = []; sources = []; seen_sources = {observation_source.source_id}
    for tick, value in enumerate(future_commands, 1):
        command = _record(value, PhysicsCommand); source = _source(command)
        if source.physics_row != observation_source.physics_row+tick:
            raise ValueError('Command rows must be consecutive future physics rows in recorded order')
        expected_time = observation_source.time_s+tick/PHYSICS_HZ
        if abs(source.time_s-expected_time) > TIMESTAMP_TOLERANCE_S:
            raise ValueError('Command timestamp differs from the observation-anchored 240 Hz clock')
        if source.context != observation_source.context:
            raise ValueError('Chunk crosses an intervening episode/target/phase/owner context boundary')
        if source.source_id in seen_sources:
            raise ValueError('Duplicate source_id in observation/future command records')
        seen_sources.add(source.source_id)
        commands.append(_command_q(command.q_command)); sources.append(source)
    action_sources = tuple(sources[PHYSICS_STEPS_PER_ACTION-1::PHYSICS_STEPS_PER_ACTION])
    validate_chunk_timing(observation_source.time_s, [row.time_s for row in action_sources],
        observation_source.context, [row.context for row in action_sources], metadata=sample_metadata)
    block = encode_joint_chunk(np.asarray(commands)[PHYSICS_STEPS_PER_ACTION-1::PHYSICS_STEPS_PER_ACTION],
                               state[:7], arm=arm, metadata=sample_metadata)
    # Source references and numerical values must not alias mutable input rows.
    state = np.frombuffer(state.tobytes(), dtype=np.float64)
    return M2JointSample(state, block, observation_source, tuple(sources), action_sources)
