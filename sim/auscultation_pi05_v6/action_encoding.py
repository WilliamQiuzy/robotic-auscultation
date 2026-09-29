"""Declared bounded action coordinates; not a post-hoc command clamp.

The original absolute-q labels remain the physical ground truth. Encode their
strictly interior values as log odds, fit phase normalization in that latent
space using training data only, and keep the official flow velocity head
unbounded. Decode only the final sampled/denormalized latent trajectory.

Real arithmetic maps finite latents to an open interval. Float32 quantization
may return either declared endpoint; both endpoints are strictly inside the
original URDF limits. This numeric saturation is part of the representation,
not evidence that a command is collision-safe or dynamically executable.
"""
import hashlib
from pathlib import Path

import numpy as np

from . import action_contract

VERSION = 'float32_interior_urdf_logit_v1'
ROUNDTRIP_TOLERANCE_RAD = 2e-6
_URDF_LOWER = np.asarray(action_contract.LOWER, np.float64)
_URDF_UPPER = np.asarray(action_contract.UPPER, np.float64)


def _interior_bounds():
    lower = _URDF_LOWER.astype(np.float32)
    upper = _URDF_UPPER.astype(np.float32)
    # Choose the closest representable values strictly INSIDE the true bounds.
    # Only the fixed codec domain is constructed here; action data is not moved.
    lower = np.where(lower.astype(np.float64) > _URDF_LOWER, lower,
                     np.nextafter(lower, np.float32(np.inf))).astype(np.float32)
    upper = np.where(upper.astype(np.float64) < _URDF_UPPER, upper,
                     np.nextafter(upper, np.float32(-np.inf))).astype(np.float32)
    if not (np.all(lower.astype(np.float64) > _URDF_LOWER)
            and np.all(upper.astype(np.float64) < _URDF_UPPER) and np.all(upper > lower)):
        raise ValueError('No finite float32 interior URDF action domain')
    return lower.astype(np.float64), upper.astype(np.float64)


_LOWER, _UPPER = _interior_bounds()


def _array(value, name):
    message = 'Invalid '+name+': expected finite numeric (..., 7) with nonempty leading dimensions'
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in 'fiu' or raw.ndim < 1 or raw.shape[-1] != 7 or not raw.size:
            raise ValueError(message)
        result = raw.astype(np.float64)
    except (ValueError, TypeError, OverflowError):
        raise ValueError(message) from None
    if not np.isfinite(result).all():
        raise ValueError(message)
    return result


def encode(actions):
    """Return float32 log-odds, rejecting endpoints/outside labels unchanged."""
    q = _array(actions, 'absolute actions')
    if not np.all((q > _LOWER) & (q < _UPPER)):
        raise ValueError('Absolute action labels must be strictly inside the declared float32 URDF domain; no clipping')
    # Difference of logs avoids cancellation of 1-p near either endpoint.
    z = (np.log(q-_LOWER)-np.log(_UPPER-q)).astype(np.float32)
    if not np.isfinite(z).all():
        raise ValueError('Nonfinite encoded action latent')
    return z


def decode(latent):
    """Decode finite latents with stable sigmoid arithmetic, without q clipping.

    Float64 CPU arithmetic precedes the one explicit float32 output cast. Keep
    this codec outside bf16/float16 model arithmetic, including endpoint values.
    """
    z = _array(latent, 'action latent')
    with np.errstate(over='raise', invalid='raise', under='ignore'):
        exponential = np.exp(-np.abs(z))
        tail = exponential/(1.+exponential)
        width = _UPPER-_LOWER
        q = np.where(z >= 0., _UPPER-width*tail, _LOWER+width*tail).astype(np.float32)
    q64 = q.astype(np.float64)
    if (not np.isfinite(q).all() or not np.all((q64 >= _LOWER) & (q64 <= _UPPER))
            or not np.all((q64 > _URDF_LOWER) & (q64 < _URDF_UPPER))):
        raise ValueError('Decoded action violated the declared float32 URDF domain')
    return q


def action_encoding_binding():
    """Bind source, parent action meaning, bounds, dtypes and numerical policy."""
    return dict(schema='wam_action_encoding_binding_v1', version=VERSION,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        action_contract_version=action_contract.VERSION,
        action_contract_sha256=hashlib.sha256(Path(action_contract.__file__).read_bytes()).hexdigest(),
        urdf_lower_rad=_URDF_LOWER.tolist(), urdf_upper_rad=_URDF_UPPER.tolist(),
        float32_lower_rad=_LOWER.tolist(), float32_upper_rad=_UPPER.tolist(),
        encoded_dtype='float32', decoded_dtype='float32', intermediate_dtype='float64_cpu',
        formula='z=log(q-lower)-log(upper-q); q=lower+(upper-lower)*sigmoid(z)',
        normalization='train-only per-phase statistics on encoded latent, never physical-q statistics',
        endpoint_policy='labels strictly interior; finite decoded latents may quantize to declared interior endpoints',
        nonfinite_policy='reject before encoding or decoding', action_clipping=False,
        roundtrip_tolerance_rad=ROUNDTRIP_TOLERANCE_RAD,
        evaluation_reference='original physical action labels, never decoded normalized labels',
        scope='action support only; original raw joint, dynamics, collision, posture and force gates remain required')


def inspect_support(actions):
    """Report source-label support and actual float32 encode/decode error.

    Numeric/shape errors raise. Endpoint/domain violations return an explicit
    unsupported report, with no adjusted labels or attempted infinite encoding.
    This audit does not include a subsequent phase normalization round trip;
    integration must audit that complete pipeline against the same raw labels.
    """
    q = _array(actions, 'absolute actions').reshape(-1, 7)
    lower_margin, upper_margin = q-_LOWER, _UPPER-q
    invalid = (lower_margin <= 0.) | (upper_margin <= 0.)
    supported = not bool(invalid.any())
    report = dict(schema='wam_action_encoding_support_v1', version=VERSION,
        sample_count=len(q), strict_interior_supported=supported,
        invalid_coordinate_count=int(invalid.sum()),
        invalid_coordinates_per_joint=invalid.sum(axis=0).astype(int).tolist(),
        minimum_lower_margin_rad_per_joint=lower_margin.min(axis=0).tolist(),
        minimum_upper_margin_rad_per_joint=upper_margin.min(axis=0).tolist(),
        roundtrip_tolerance_rad=ROUNDTRIP_TOLERANCE_RAD,
        roundtrip_max_abs_rad=None, roundtrip_max_abs_rad_per_joint=None,
        roundtrip_within_declared_tolerance=False,
        float32_reference_exact_coordinate_count=None,
        scope='codec-only actual float32 roundtrip; phase normalization requires a separate full-pipeline audit')
    if supported:
        z = encode(q)
        decoded = decode(z)
        errors = np.abs(decoded.astype(np.float64)-q)
        maximum = float(errors.max())
        report.update(roundtrip_max_abs_rad=maximum,
            roundtrip_max_abs_rad_per_joint=errors.max(axis=0).tolist(),
            roundtrip_within_declared_tolerance=maximum <= ROUNDTRIP_TOLERANCE_RAD,
            float32_reference_exact_coordinate_count=int((decoded == q.astype(np.float32)).sum()),
            latent_min_per_joint=z.min(axis=0).astype(float).tolist(),
            latent_max_per_joint=z.max(axis=0).astype(float).tolist())
    return report
