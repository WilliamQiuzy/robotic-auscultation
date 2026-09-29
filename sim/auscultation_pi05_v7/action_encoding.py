"""Train-only physical joint deltas anchored at the chunk's observed q.

Every target in the ten-step horizon is an immutable absolute command minus
the SAME measured q from the initial observation. Normalization is affine in
radians, with no logit transform or clipping. Decoding adds that same anchor.

This representation does not bound an action proposal. Callers must check the
decoded absolute trajectory with the original joint, collision, posture, force
and dynamics gates before execution. A zero residual is not a task controller.
"""
import hashlib
from pathlib import Path

import numpy as np

from sim.auscultation_pi05_v6 import action_contract

VERSION = 'fixed_observed_q_chunk_delta_rad_v1'
SCHEMA = 'wam_fixed_observation_delta_normalization_v1'
HORIZON = 10
MINIMUM_FULL_SPAN_RAD = .002
ROUNDTRIP_TOLERANCE_RAD = 2e-6
QUANTILES = (.01, .99)


def action_encoding_binding():
    """Bind source, physical units, anchor semantics and the original raw gate."""
    return dict(schema='wam_delta_action_encoding_binding_v1', version=VERSION,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        horizon=HORIZON, action_width=7, units='rad',
        formula='delta[k]=absolute_command[k]-observation_q; absolute_command[k]=delta[k]+observation_q',
        anchor_policy='one measured q at the initial observation for the complete horizon',
        normalization='train-only 1%-99% phase quantiles over explicitly supplied valid windows',
        minimum_full_span_rad=MINIMUM_FULL_SPAN_RAD,
        encoded_dtype='float32', decoded_dtype='float32', intermediate_dtype='float64_cpu',
        roundtrip_tolerance_rad=ROUNDTRIP_TOLERANCE_RAD,
        action_clipping=False, bounded_by_codec=False,
        original_absolute_action_contract=action_contract.VERSION,
        original_absolute_action_contract_sha256=hashlib.sha256(Path(action_contract.__file__).read_bytes()).hexdigest(),
        execution_policy='apply original absolute joint and other execution gates after decoding')


def _array(value, shape, name):
    try:
        raw = np.asarray(value)
        if raw.shape != shape or raw.dtype.kind not in 'fiu':
            raise ValueError('Invalid '+name+' shape or numeric dtype')
        result = raw.astype(np.float64)
    except (ValueError, TypeError, OverflowError):
        raise ValueError('Invalid '+name+' shape or numeric dtype') from None
    if not np.isfinite(result).all():
        raise ValueError('Invalid nonfinite '+name)
    return result


def fit_stats(episodes):
    """Fit only caller-supplied, already cadence-qualified horizon starts.

    Each episode supplies ``episode_id, split, actions[N,7], state[N,41],
    phases[N], starts``. The caller supplies V6's validated starts; this function
    never regenerates omitted windows. It checks each supplied start is complete
    and within one known phase. Cadence/source qualification remains the caller's
    responsibility because this interface intentionally receives no timestamps.
    """
    if not isinstance(episodes, (list, tuple)) or not episodes:
        raise ValueError('Nonempty training episodes required')
    ids, residuals, window_counts = [], {}, {}
    for episode in episodes:
        if not isinstance(episode, dict) or episode.get('split') != 'train':
            raise ValueError('Normalization may fit train episodes only')
        identity = episode.get('episode_id')
        if not isinstance(identity, str) or not identity.strip() or identity in ids:
            raise ValueError('Unique nonempty training episode identity required')
        ids.append(identity)
        raw_actions = np.asarray(episode.get('actions'))
        count = raw_actions.shape[0] if raw_actions.ndim else 0
        if count < HORIZON:
            raise ValueError('Episode has no complete action window')
        actions = _array(raw_actions, (count, 7), 'absolute actions')
        states = _array(episode.get('state'), (count, 41), 'measured state')
        phases = np.asarray(episode.get('phases'))
        if phases.shape != (count,) or phases.dtype.kind != 'U' or not set(phases) <= action_contract.PHASES:
            raise ValueError('Known phase labels required')
        starts = episode.get('starts')
        if (not isinstance(starts, (list, tuple, np.ndarray))
                or (isinstance(starts, np.ndarray) and starts.ndim != 1) or len(starts) == 0):
            raise ValueError('Explicit nonempty valid window starts required')
        if any(isinstance(start, (bool, np.bool_)) or not isinstance(start, (int, np.integer))
               or not 0 <= int(start) <= count-HORIZON for start in starts):
            raise ValueError('Invalid complete-window start')
        if len(set(map(int, starts))) != len(starts):
            raise ValueError('Duplicate window starts')
        for start in starts:
            start = int(start)
            phase = str(phases[start])
            if np.any(phases[start:start+HORIZON] != phase):
                raise ValueError('Window crosses a phase boundary')
            # Read-only arithmetic; each window's anchor is measured once.
            delta = actions[start:start+HORIZON] - states[start, :7]
            if not np.isfinite(delta).all():
                raise ValueError('Nonfinite physical window residual')
            residuals.setdefault(phase, []).append(delta)
            window_counts[phase] = window_counts.get(phase, 0) + 1
    by_phase = {}
    for phase in sorted(residuals):
        values = np.concatenate(residuals[phase], axis=0)
        low, high = np.quantile(values, QUANTILES, axis=0)
        center = (low + high) / 2
        span = np.maximum(high-low, MINIMUM_FULL_SPAN_RAD)
        by_phase[phase] = dict(low=(center-span/2).tolist(), high=(center+span/2).tolist(),
            observed_min=values.min(axis=0).tolist(), observed_max=values.max(axis=0).tolist(),
            window_count=window_counts[phase], residual_count=len(values), units='rad')
    return dict(schema=SCHEMA, version=VERSION, horizon=HORIZON, units='rad',
        fit_episode_ids=ids, quantiles=list(QUANTILES), minimum_full_span_rad=MINIMUM_FULL_SPAN_RAD,
        fitting_weight='each explicitly supplied window contributes its ten fixed-anchor residuals',
        action_encoding_binding=action_encoding_binding(), actions_by_phase=by_phase)


def _phase_range(phase, stats):
    if not isinstance(stats, dict) or stats.get('action_encoding_binding') != action_encoding_binding():
        raise ValueError('Action encoding source binding changed')
    if (stats.get('schema') != SCHEMA or stats.get('version') != VERSION
            or stats.get('horizon') != HORIZON or stats.get('units') != 'rad'
            or stats.get('minimum_full_span_rad') != MINIMUM_FULL_SPAN_RAD):
        raise ValueError('Invalid fixed-anchor normalization contract')
    if not isinstance(phase, str) or phase not in action_contract.PHASES or phase not in stats.get('actions_by_phase', {}):
        raise ValueError('Unknown or unfitted action phase')
    own = stats['actions_by_phase'][phase]
    low = _array(own.get('low'), (7,), 'phase low')
    high = _array(own.get('high'), (7,), 'phase high')
    if own.get('units') != 'rad' or np.any(high <= low):
        raise ValueError('Invalid physical phase normalization range')
    return low, high


def encode_chunk(absolute_commands, observation_q, phase, stats):
    """Return float32[10,7] targets; values outside training support stay outside."""
    commands = _array(absolute_commands, (HORIZON, 7), 'absolute action chunk')
    anchor = _array(observation_q, (7,), 'observation q anchor')
    low, high = _phase_range(phase, stats)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        result = (2*((commands-anchor)-low)/(high-low)-1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite encoded action chunk')
    return result


def decode_chunk(normalized_chunk, observation_q, phase, stats):
    """Return raw absolute q; the caller must apply original execution gates."""
    normalized = _array(normalized_chunk, (HORIZON, 7), 'normalized action chunk')
    anchor = _array(observation_q, (7,), 'observation q anchor')
    low, high = _phase_range(phase, stats)
    with np.errstate(over='ignore', invalid='ignore'):
        result = ((normalized+1)/2*(high-low)+low+anchor).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite decoded absolute action chunk')
    return result
