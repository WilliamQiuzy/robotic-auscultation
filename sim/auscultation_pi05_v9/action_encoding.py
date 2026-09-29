"""Explicit learned rational q4 coordinate; six frozen B residual coordinates.

Fitting is TRAIN-only preparation. Encode/decode validate and use saved statistics,
never refit. This bounds q4 support only; original all-joint and physical gates
remain necessary. No clipping, alternate anchor, or corrective projection occurs.
"""
import copy
import hashlib
from pathlib import Path
import numpy as np
from sim.auscultation_pi05_v8 import action_encoding as parent

C = 'C_rational_q4'
MODES = (C,)
HORIZON = 10
MINIMUM_FULL_SPAN_RAD = .002
ROUNDTRIP_TOLERANCE_RAD = 2e-6
SCHEMA = 'wam_rational_q4_mixed_normalization_v1'
VERSION = 'fixed_observed_q_signed_rational_q4_rad_v1'
PARENT_CODEC_SHA256 = 'a694add3d8becbc99fdc18b9c4ad9d10823339f2a674fcc62130c9147331ae23'
SPEC_SHA256 = 'cfcf4573ba222656735cb5e99bc521e684d02dc0513a143b2e37bd83e2eecff1'
U_SAFE_RAD = 2.4434609413146973
SIX = (0, 1, 2, 4, 5, 6)
_need = parent._need
_digest = parent._digest
_array = parent.parent._array


def _mode(mode):
    _need(isinstance(mode, str) and mode == C, 'Explicit rational q4 mode required')
    return mode


def recipe(mode=C):
    return 'pi05_v9_signed_rational_q4_v1_' + _mode(mode)


def action_encoding_binding(mode=C):
    _mode(mode)
    old = parent.action_encoding_binding(parent.B)
    _need(old['source_sha256'] == PARENT_CODEC_SHA256, 'Frozen B codec changed')
    limits = parent.parent.action_contract
    _need(float(limits.LOWER[3]) == 0 and U_SAFE_RAD <= float(limits.UPPER[3])
          < float(np.nextafter(np.float32(U_SAFE_RAD), np.float32(np.inf))), 'Original q4 bounds changed')
    return dict(schema='wam_rational_q4_action_binding_v1', version=VERSION, mode=mode,
        recipe=recipe(mode), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        parent_binding=old, approved_design_sha256=SPEC_SHA256, horizon=10, action_width=7,
        units='rad', q4_index=3, U_safe_rad=U_SAFE_RAD, unchanged_joint_indices=list(SIX),
        anchor_policy='one original measured q for all ten horizons',
        q4_inverse='below:a*((target-a)/target); above:(U-a)*((target-a)/(U-target)); equal:0',
        q4_decode='negative:a*a/(a-r); positive:a+(U-a)*(r/(U-a+r)); zero:a',
        q4_fit_axis='FREE_APPROACH phase_horizon; CONTACT_HOLD_RETRACT phase_pooled_horizon',
        quantiles=[.01, .99], quantile_method='linear', minimum_full_span_rad=.002,
        action_clipping=False, bounded_by_codec={'joint3': True, 'other6': False},
        intermediate_dtype='float64_cpu', encoded_dtype='float32', decoded_dtype='float32',
        denormalized_q4_residual_cast='none', roundtrip_tolerance_rad=ROUNDTRIP_TOLERANCE_RAD,
        execution_policy='apply unchanged original raw absolute-joint and physical gates')


def _numeric(value, name):
    raw = np.asarray(value)
    _need(raw.dtype.kind in 'fiu', 'Invalid numeric dtype for ' + name)
    with np.errstate(over='ignore', invalid='ignore'):
        result = raw.astype(np.float64)
    _need(np.isfinite(result).all(), 'Nonfinite ' + name)
    return result


def _domain(value, name):
    result = _numeric(value, name)
    _need(np.all((result >= 0) & (result <= U_SAFE_RAD)), 'Out-of-domain ' + name)
    return result


def inverse_q4(target, anchor):
    """Float64 signed residual; non-anchor endpoints cannot be finite targets."""
    q, a = np.broadcast_arrays(_domain(target, 'q4 target'), _domain(anchor, 'q4 anchor'))
    _need(not np.any(((q == 0) | (q == U_SAFE_RAD)) & (q != a)), 'Non-anchor q4 endpoint has infinite inverse')
    r = np.zeros(q.shape, np.float64)
    below, above = q < a, q > a
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        r[below] = a[below] * ((q[below] - a[below]) / q[below])
        r[above] = (U_SAFE_RAD-a[above]) * ((q[above]-a[above]) / (U_SAFE_RAD-q[above]))
        serialized = r.astype(np.float32)
    _need(np.isfinite(r).all() and np.isfinite(serialized).all(), 'Inverse outside finite float32 support')
    return r


def decode_q4(residual, anchor):
    """Stable float64 branch arithmetic, then one final float32 command cast."""
    r, a = np.broadcast_arrays(_numeric(residual, 'q4 residual'), _domain(anchor, 'q4 anchor'))
    q = np.empty(r.shape, np.float64)
    below, above, zero = r < 0, r > 0, r == 0
    q[zero] = a[zero]
    with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
        q[below] = a[below]*a[below] / (a[below]-r[below])
        b = U_SAFE_RAD-a[above]
        q[above] = a[above] + b * (r[above] / (b+r[above]))
        out = q.astype(np.float32)
    _need(np.isfinite(q).all() and np.isfinite(out).all()
          and np.all((q >= 0) & (q <= U_SAFE_RAD))
          and np.all((out >= 0) & (out.astype(np.float64) <= U_SAFE_RAD)), 'Invalid decoded q4; no repair permitted')
    return out


def fit_stats(episodes, mode=C, *, parent_stats=None):
    """Fit each explicit qualified TRAIN window once; never generate starts."""
    binding = action_encoding_binding(mode)
    fitted_B = parent.fit_stats(episodes, parent.B)  # unchanged numeric validation/axes
    if parent_stats is not None:
        parent.validate_stats(parent_stats, mode=parent.B)
        _need(parent_stats == fitted_B, 'Supplied B statistics do not match original TRAIN windows')
    old = copy.deepcopy(fitted_B if parent_stats is None else parent_stats)
    by_phase = copy.deepcopy(old['actions_by_phase'])
    for episode in episodes:
        _domain(np.asarray(episode['state'])[:, 3], 'TRAIN q4 anchors')
        _domain(np.asarray(episode['actions'])[:, 3], 'TRAIN q4 labels')
    for phase, own in by_phase.items():
        values = np.stack([inverse_q4(e['actions'][s:s+10, 3], e['state'][s, 3])
                           for e in episodes for s in map(int, e['starts']) if str(e['phases'][s]) == phase])
        horizon = phase in parent.CHANGED_PHASES
        lo, hi = np.quantile(values, [.01, .99], axis=0 if horizon else None, method='linear')
        center = (lo+hi)/2
        span = np.maximum(hi-lo, MINIMUM_FULL_SPAN_RAD)
        replacements = dict(low=center-span/2, high=center+span/2,
                            observed_min=values.min(axis=0) if horizon else values.min(),
                            observed_max=values.max(axis=0) if horizon else values.max())
        for key, value in replacements.items():
            full = np.asarray(own[key], np.float64).copy()
            full[..., 3] = value
            own[key] = full.tolist()
    result = dict(schema=SCHEMA, version=VERSION, recipe=recipe(mode), mode=mode,
        horizon=10, units='rad', quantiles=[.01, .99], quantile_method='linear',
        minimum_full_span_rad=.002, action_encoding_binding=binding,
        fit_episode_ids=old['fit_episode_ids'], fit_window_count=old['fit_window_count'],
        fit_windows_sha256=old['fit_windows_sha256'], fitting_weight=old['fitting_weight'],
        parent_B_stats=old, parent_B_normalization_sha256=parent.normalization_digest(old, mode=parent.B),
        actions_by_phase=by_phase)
    result['normalization_sha256'] = _digest(result)
    validate_stats(result, mode=mode)
    return result


def validate_stats(stats, *, mode=C, expected_sha256=None):
    binding = action_encoding_binding(mode)
    _need(isinstance(stats, dict) and stats.get('schema') == SCHEMA and stats.get('version') == VERSION
          and stats.get('mode') == mode and stats.get('recipe') == recipe(mode), 'Wrong rational normalization identity')
    _need(stats.get('action_encoding_binding') == binding, 'Rational codec/source binding differs')
    _need(stats.get('horizon') == 10 and stats.get('units') == 'rad' and stats.get('quantiles') == [.01,.99]
          and stats.get('quantile_method') == 'linear' and stats.get('minimum_full_span_rad') == .002, 'Rational numeric contract differs')
    actual = _digest({k:v for k,v in stats.items() if k != 'normalization_sha256'})
    _need(stats.get('normalization_sha256') == actual and (expected_sha256 is None or actual == expected_sha256), 'Rational normalization hash differs')
    old = stats.get('parent_B_stats', {})
    old_hash = parent.validate_stats(old, mode=parent.B)
    _need(stats.get('parent_B_normalization_sha256') == old_hash, 'B normalization parent differs')
    for key in ('fit_episode_ids', 'fit_window_count', 'fit_windows_sha256', 'fitting_weight'):
        _need(stats.get(key) == old[key], 'B fitting identity differs')
    phases = stats.get('actions_by_phase')
    _need(isinstance(phases, dict) and set(phases) == set(old['actions_by_phase']), 'B fitted phase set differs')
    for phase, own in phases.items():
        ref = old['actions_by_phase'][phase]
        shape = (10,7) if phase in parent.CHANGED_PHASES else (7,)
        _need(isinstance(own, dict) and set(own) == set(ref), 'Mixed phase metadata differs')
        for key in ref:
            if key not in ('low','high','observed_min','observed_max'):
                _need(own[key] == ref[key], 'Original B phase metadata differs')
            else:
                value = _array(own[key], shape, 'mixed phase '+key)
                expected = _array(ref[key], shape, 'B phase '+key)
                _need(value[..., SIX].tobytes() == expected[..., SIX].tobytes(), 'Unchanged six B statistics differ')
        low, high = np.asarray(own['low']), np.asarray(own['high'])
        _need(np.all(high > low), 'Invalid mixed normalization range')
    return actual


def normalization_digest(stats, *, mode=C):
    return validate_stats(stats, mode=mode)


def _range(phase, stats, mode):
    validate_stats(stats, mode=mode)
    _need(isinstance(phase, str) and phase in stats['actions_by_phase'], 'Unknown or unfitted phase')
    own = stats['actions_by_phase'][phase]
    return np.asarray(own['low'], np.float64), np.asarray(own['high'], np.float64)


def encode_chunk(absolute_commands, observation_q, phase, stats, *, mode=C):
    low, high = _range(phase, stats, mode)
    commands = _array(absolute_commands, (10,7), 'absolute action chunk')
    anchor = _array(observation_q, (7,), 'observation anchor')
    r = inverse_q4(commands[:,3], anchor[3])
    # Delegate six unchanged coordinates to exact B operation order.
    out = parent.encode_chunk(commands, anchor, phase, stats['parent_B_stats'], mode=parent.B)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        out[:,3] = (2*(r-low[...,3])/(high[...,3]-low[...,3])-1).astype(np.float32)
    _need(np.isfinite(out).all(), 'Nonfinite rational normalized target')
    return out


def decode_chunk(normalized_chunk, observation_q, phase, stats, *, mode=C):
    low, high = _range(phase, stats, mode)
    normalized = _array(normalized_chunk, (10,7), 'normalized action chunk')
    with np.errstate(over='ignore', invalid='ignore'):
        serialized = normalized.astype(np.float32)
    _need(np.isfinite(serialized).all() and np.array_equal(serialized.astype(np.float64), normalized),
          'Normalized model coordinates must retain exact finite float32 values')
    anchor = _array(observation_q, (7,), 'observation anchor')
    with np.errstate(over='ignore', invalid='ignore'):
        r = (normalized[:,3]+1)/2*(high[...,3]-low[...,3])+low[...,3]
    q4 = decode_q4(r, anchor[3])
    # Avoid an irrelevant additive-q4 overflow before replacing it with rational q4.
    unchanged = normalized.copy()
    unchanged[:,3] = 0
    out = parent.decode_chunk(unchanged, anchor, phase, stats['parent_B_stats'], mode=parent.B)
    out[:,3] = q4
    _need(np.isfinite(out).all(), 'Nonfinite raw mixed action proposal')
    return out
