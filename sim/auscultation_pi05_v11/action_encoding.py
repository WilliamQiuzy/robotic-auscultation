"""Source-bound all70 softplus interval coordinates, with no output projection.

``build_bounds_contract(project)`` reads the twelve approved physical sources
once. Its portable table can then be validated without rereading the plans.
``fit_stats(qualified_episode_views, bounds_contract)`` fits every explicit TRAIN
start once, using independent horizons for FREE/APPROACH and pooled horizons for
CONTACT/HOLD/RETRACT. Acquisition/cadence qualification and the canonical16/24397
membership are the caller's responsibility; this module checks numerical views.

``encode_chunk(q, anchor, phase, target, stats)`` and ``decode_chunk(z, anchor,
phase, target, stats)`` use saved statistics only. ``denormalize_chunk`` exposes
the float64 latent residual for evidence/replay. A single measured7-vector is
used for all10 horizons. The caller must match target to its row/request; joint
values alone cannot identify the requested target. Zero residual is NOT a hold
near a bound. Bounded support is not learned safety or physical acceptance.
"""
import ast
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np

MODE = 'V11_target_interval_latent_residual_v1'
VERSION = 'wam_pi05_v11_target_interval_latent_residual_v1'
SCHEMA = 'wam_pi05_v11_target_interval_latent_normalization_v1'
BOUNDS_SCHEMA = 'wam_pi05_v11_source_bound_target_intervals_v1'
RECIPE = 'pi05_v11_all70_softplus_interval_latent_residual_v1'
SPEC_SHA256 = 'aaaac1cdf4d46fa67db49360701e7dd5d8f3cef3960ea8442b8a1d194cb11c6e'
PROPOSAL_SHA256 = '38f6de40961bd7deaddf7e2816d5bb33c9272d8fa52c265826fcd37647b31a89'
HORIZON, ACTION_WIDTH = 10, 7
TAU_RAD = .001
MINIMUM_FULL_SPAN_RAD = .002
ROUNDTRIP_TOLERANCE_RAD = 2e-6
PHASES = ('FREE', 'APPROACH', 'CONTACT', 'HOLD', 'RETRACT')
HORIZON_PHASES = ('FREE', 'APPROACH')
TARGETS = ('R1', 'L1', 'R2', 'L2', 'R3', 'L3', 'R4', 'L4', 'R5', 'L5')
ACTION_CONTRACT_PATH = 'sim/auscultation_pi05_v6/action_contract.py'
PLAN_PREFIX = 'sim/upper_pair_posture_20260924/results/candidate_b_plans/planned_paths_'
APPROVED_PROJECTION_SHA256 = 'ddd67a7863df2fc2417a41c48b9a2f4060f49b2f237d40e2dbd73a61637249d5'
SOURCE_SHA256 = MappingProxyType({
    PLAN_PREFIX+'R1.json':'8a28db5771cd12f394f07019075b9c91508ef5b00931ae0fa844cc58a10d3026',
    PLAN_PREFIX+'L1.json':'b3c655ee7b7370b4492090eab3800e9e3132477408d76b05d63e8af3e7d548be',
    PLAN_PREFIX+'R2.json':'1d9b95c3386c15c444d8ee6b7e3fefd3daba9a0fa65c714874b2e6c9170ade1b',
    PLAN_PREFIX+'L2.json':'b277b04f14de68f1009d0600bc4b3b1f9257b1f131fdaffcd9df2e9be10beb82',
    PLAN_PREFIX+'R3.json':'2b7749a1cf6497960bfde933c6ab33553bf5e34420c20e19828d3384617d5a11',
    PLAN_PREFIX+'L3.json':'f160e28fc7d029bced5330b6fffff88485809b1c48f68a307a2de69277566beb',
    PLAN_PREFIX+'R4.json':'a5350dc5f5da21c5e476264da927636ea3c2dbf3c7a8a77d08898c4561a0e53f',
    PLAN_PREFIX+'L4.json':'aea2b7f66c7cba10ac23f83cb3f5518876290f8e383d7f42ce2f06ac92693b64',
    PLAN_PREFIX+'R5.json':'6603f93b6365840dede7e75def86490f76e938e050c75bc1edd74d7970e4f79f',
    PLAN_PREFIX+'L5.json':'4d2da4c86861d18f646d01763a0a5dfbebaa9d3b6d598c0e27d00f9b0524aa75',
    'sim/auscultation_expert_layout_v1/posture.py':'87ae90861e3221796910eebe620b5b7f8c94bca11992d084a4b2e3d07a3ceac2',
    ACTION_CONTRACT_PATH:'d21e73ec9e530b0a3785debb9e78cd42648f4de45880221317ba982aa44b7596',
})


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    """Portable canonical JSON object digest, not a file-byte digest."""
    try:
        raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    except (TypeError, ValueError, OverflowError):
        raise ValueError('Non-canonical JSON value') from None
    return hashlib.sha256(raw).hexdigest()


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _boolean_leaf(value):
    return isinstance(value, (bool, np.bool_)) or (isinstance(value, (tuple, list)) and any(_boolean_leaf(v) for v in value))


def _array(value, shape, name):
    try:
        raw = np.asarray(value)
        _need(raw.shape == shape and raw.dtype.kind in 'fiu' and not _boolean_leaf(value), 'Invalid '+name+' shape or numeric dtype')
        with np.errstate(over='ignore', invalid='ignore'):
            out = raw.astype(np.float64)
    except (TypeError, OverflowError):
        raise ValueError('Invalid '+name+' shape or numeric dtype') from None
    _need(np.isfinite(out).all(), 'Nonfinite '+name)
    return out


def inward_endpoints(lower, upper):
    """Nearest finite float32 endpoints contained in the original7 intervals."""
    low, high = _array(lower, (7,), 'lower bounds'), _array(upper, (7,), 'upper bounds')
    _need(np.all(low < high), 'Empty original interval')
    with np.errstate(over='ignore', invalid='ignore'):
        lo, hi = low.astype(np.float32), high.astype(np.float32)
        lo = np.where(lo.astype(np.float64) < low, np.nextafter(lo, np.float32(np.inf)), lo)
        hi = np.where(hi.astype(np.float64) > high, np.nextafter(hi, np.float32(-np.inf)), hi)
    lo, hi = lo.astype(np.float64), hi.astype(np.float64)
    _need(np.isfinite(lo).all() and np.isfinite(hi).all() and np.all(lo < hi)
          and np.all(lo >= low) and np.all(hi <= high), 'Empty or nonfinite float32 inward interval')
    return lo, hi


def _table(projection):
    urdf_low = _array(projection['urdf_lower_rad'], (7,), 'URDF lower')
    urdf_high = _array(projection['urdf_upper_rad'], (7,), 'URDF upper')
    _need(digest(projection) == APPROVED_PROJECTION_SHA256, 'Approved limits numeric projection differs')
    targets = {}
    for target in TARGETS:
        posture = projection['posture_limits'][target]
        pl = _array(posture['joint_lower_rad'], (7,), 'posture lower')
        pu = _array(posture['joint_upper_rad'], (7,), 'posture upper')
        lower, upper = np.maximum(urdf_low, pl), np.minimum(urdf_high, pu)
        lo, hi = inward_endpoints(lower, upper)
        targets[target] = dict(target=target, posture_lower_rad=pl.tolist(), posture_upper_rad=pu.tolist(),
                               original_lower_rad=lower.tolist(), original_upper_rad=upper.tolist(),
                               lower_rad=lo.tolist(), upper_rad=hi.tolist())
    result = dict(schema=BOUNDS_SCHEMA, version=VERSION, mode=MODE, source_sha256=dict(SOURCE_SHA256),
                  projection_sha256=APPROVED_PROJECTION_SHA256, urdf_lower_rad=urdf_low.tolist(),
                  urdf_upper_rad=urdf_high.tolist(), targets=targets)
    result['bounds_sha256'] = digest(result)
    return result


def build_bounds_contract(source_root):
    """Read/hash approved sources before parsing; never read dataset/model files."""
    root = Path(source_root)
    payloads = {}
    for relative, expected in SOURCE_SHA256.items():
        path = root / relative
        _need(path.is_file() and not path.is_symlink(), 'Missing or nonordinary bounds source')
        raw = path.read_bytes()
        _need(hashlib.sha256(raw).hexdigest() == expected, 'Approved bounds source hash differs: '+relative)
        payloads[relative] = raw
    literals = {}
    for node in ast.parse(payloads[ACTION_CONTRACT_PATH]).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ('LOWER', 'UPPER'):
            literals[node.targets[0].id] = list(ast.literal_eval(node.value))
    posture = {}
    for target in TARGETS:
        plan = json.loads(payloads[PLAN_PREFIX+target+'.json'])
        _need(len(plan['points']) == 1 and plan['points'][0]['name'] == target, 'Plan target identity differs')
        own = plan['points'][0]['posture_limits']
        posture[target] = {k: own[k] for k in ('joint_lower_rad', 'joint_upper_rad')}
    result = _table(dict(urdf_lower_rad=literals['LOWER'], urdf_upper_rad=literals['UPPER'], posture_limits=posture))
    # Retain the early consumed bytes as authority, and reject rebinding before return.
    for relative, raw in payloads.items():
        path = root / relative
        _need(path.is_file() and not path.is_symlink() and hashlib.sha256(path.read_bytes()).hexdigest() == hashlib.sha256(raw).hexdigest(), 'Bounds source changed during construction')
    return result


def validate_bounds_contract(contract, expected_sha256=None):
    """Reconstruct exact approved projection/endpoints; no live-plan I/O/cache."""
    _need(type(contract) is dict and contract.get('schema') == BOUNDS_SCHEMA
          and contract.get('mode') == MODE and contract.get('version') == VERSION, 'Wrong bounds contract identity')
    _need(contract.get('source_sha256') == dict(SOURCE_SHA256), 'Approved limits source map differs')
    targets = contract.get('targets')
    _need(type(targets) is dict and set(targets) == set(TARGETS), 'Exact canonical target set required')
    projection = dict(urdf_lower_rad=contract.get('urdf_lower_rad'), urdf_upper_rad=contract.get('urdf_upper_rad'), posture_limits={})
    for target in TARGETS:
        own = targets[target]
        _need(type(own) is dict and own.get('target') == target, 'Bound target identity differs')
        projection['posture_limits'][target] = dict(joint_lower_rad=own.get('posture_lower_rad'), joint_upper_rad=own.get('posture_upper_rad'))
    expected = _table(projection)
    _need(contract == expected, 'Bounds hash or approved interval differs')
    _need(expected_sha256 is None or expected_sha256 == expected['bounds_sha256'], 'Expected bounds hash differs')
    return expected['bounds_sha256']


def action_encoding_binding(bounds_contract):
    bh = validate_bounds_contract(bounds_contract)
    return dict(schema='wam_pi05_v11_target_interval_action_binding_v1', version=VERSION, mode=MODE, recipe=RECIPE,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                approved_machine_sha256=SPEC_SHA256, approved_proposal_sha256=PROPOSAL_SHA256,
                bounds_sha256=bh, limits_source_sha256=dict(SOURCE_SHA256),
                horizon=10, action_width=7, units='latent_rad', tau_rad=TAU_RAD,
                latent_label='F_inverse(q_ref[h,j])-raw_measured_anchor[j]',
                decode='float32(F(raw_measured_anchor[j]+latent_residual[h,j]))',
                anchor_policy='one original raw measured7-vector for all10 horizons', zero_residual_is_hold=False,
                per_horizon_joint_phases=list(HORIZON_PHASES), pooled_horizon_per_joint_phases=list(PHASES[2:]),
                quantiles=[.01, .99], quantile_method='linear', minimum_full_span_latent_rad=.002,
                bounded_by_codec=True, action_clipping=False, posthoc_projection=False, saturation_possible=True,
                intermediate_dtype='float64_cpu', encoded_dtype='float32', decoded_dtype='float32',
                roundtrip_tolerance_rad=ROUNDTRIP_TOLERANCE_RAD, original_physical_gates_required=True)


def _limits(target, bounds):
    _need(type(target) is str and target in TARGETS, 'Explicit canonical target required')
    own = bounds['targets'][target]
    return np.asarray(own['lower_rad'], np.float64), np.asarray(own['upper_rad'], np.float64)


def _inverse(q, lower, upper):
    _need(np.all((q > lower) & (q < upper)), 'Reference must be strictly inside inward interval; no clamps')
    with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
        result = q + TAU_RAD * (np.log(-np.expm1(-(q-lower)/TAU_RAD)) - np.log(-np.expm1(-(upper-q)/TAU_RAD)))
    _need(np.isfinite(result).all(), 'Nonfinite inverse coordinate')
    return result


def _softplus(value):
    return np.maximum(value, 0) + np.log1p(np.exp(-np.abs(value)))


def _forward(x, lower, upper):
    x, lower, upper = np.broadcast_arrays(x, lower, upper)
    bottom = x <= lower + (upper-lower)/2
    out = np.empty(x.shape, np.float64)
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        out[bottom] = lower[bottom] + TAU_RAD * (_softplus((x[bottom]-lower[bottom])/TAU_RAD) - _softplus((x[bottom]-upper[bottom])/TAU_RAD))
        out[~bottom] = upper[~bottom] - TAU_RAD * (_softplus((upper[~bottom]-x[~bottom])/TAU_RAD) - _softplus((lower[~bottom]-x[~bottom])/TAU_RAD))
        result = out.astype(np.float32)
    _need(np.isfinite(out).all() and np.isfinite(result).all() and np.all(result >= lower)
          and np.all(result <= upper), 'Invalid interval decode; no corrective projection permitted')
    return result


def inverse_coordinates(q_reference, target, bounds_contract):
    validate_bounds_contract(bounds_contract)
    q = _array(q_reference, (10, 7), 'reference chunk')
    return _inverse(q, *_limits(target, bounds_contract))


def forward_coordinates(latent_coordinates, target, bounds_contract):
    validate_bounds_contract(bounds_contract)
    x = _array(latent_coordinates, (10, 7), 'latent coordinate chunk')
    return _forward(x, *_limits(target, bounds_contract))


def fit_stats(qualified_episode_views, bounds_contract):
    """Fit all supplied explicit TRAIN starts once, irrespective of draw weights."""
    binding = action_encoding_binding(bounds_contract)
    bounds = json.loads(json.dumps(bounds_contract, allow_nan=False))
    _need(isinstance(qualified_episode_views, (list, tuple)) and len(qualified_episode_views) > 0, 'Nonempty qualified TRAIN views required')
    ids, records, fit_rows, values = [], [], [], {p: [] for p in PHASES}
    references, anchors, limits = ({p: [] for p in PHASES} for _ in range(3))
    for episode in qualified_episode_views:
        _need(type(episode) is dict and episode.get('split') == 'train', 'Only qualified TRAIN views may fit normalization')
        eid = episode.get('episode_id')
        _need(type(eid) is str and eid.strip() == eid and bool(eid) and eid not in ids, 'Unique nonempty TRAIN episode identity required')
        ids.append(eid)
        shape = np.asarray(episode.get('actions')).shape
        _need(len(shape) == 2 and shape[1] == 7 and shape[0] >= 10, 'Complete action windows required')
        count = shape[0]
        q = _array(episode['actions'], (count, 7), 'TRAIN actions')
        states = _array(episode.get('state'), (count, 41), 'TRAIN measured state')
        phases, targets = np.asarray(episode.get('phases')), np.asarray(episode.get('targets'))
        _need(phases.shape == (count,) and phases.dtype.kind == 'U' and set(phases) <= set(PHASES), 'Known TRAIN phases required')
        _need(targets.shape == (count,) and targets.dtype.kind == 'U' and set(targets) <= set(TARGETS), 'Known TRAIN targets required')
        starts = episode.get('starts')
        _need(isinstance(starts, (list, tuple, np.ndarray)) and np.asarray(starts).ndim == 1 and len(starts) > 0, 'Explicit nonempty qualified starts required')
        _need(all(_integer(s) and 0 <= int(s) <= count-10 for s in starts), 'Invalid complete-window start')
        starts = list(map(int, starts))
        _need(len(starts) == len(set(starts)), 'Duplicate TRAIN start')
        phase_counts = {}
        for start in starts:
            phase, target = str(phases[start]), str(targets[start])
            _need(np.all(phases[start:start+10] == phase) and np.all(targets[start:start+10] == target), 'TRAIN chunk crosses phase or target')
            interval = _limits(target, bounds)
            x = _inverse(q[start:start+10], *interval)
            residual = x-states[start, :7]
            _need(np.isfinite(residual).all(), 'Nonfinite TRAIN latent residual')
            values[phase].append(residual)
            references[phase].append(q[start:start+10])
            anchors[phase].append(states[start, :7].copy())
            limits[phase].append(interval)
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
            fit_rows.append(dict(episode_id=eid, start=start, phase=phase, target=target))
        records.append(dict(episode_id=eid, window_count=len(starts), starts_sha256=digest(starts),
                            phase_window_counts=phase_counts, targets=sorted({str(targets[s]) for s in starts})))
    by_phase, roundtrip_error = {}, 0.
    for phase, windows in values.items():
        if not windows:
            continue
        array = np.stack(windows)
        axes = 0 if phase in HORIZON_PHASES else (0, 1)
        qlo, qhi = np.quantile(array, [.01, .99], axis=axes, method='linear')
        center, span = (qlo+qhi)/2, np.maximum(qhi-qlo, .002)
        by_phase[phase] = dict(low=(center-span/2).tolist(), high=(center+span/2).tolist(),
                               quantile_low=qlo.tolist(), quantile_high=qhi.tolist(),
                               observed_min=array.min(axis=axes).tolist(), observed_max=array.max(axis=axes).tolist(),
                               window_count=len(windows), residual_count=len(windows)*10, units='latent_rad',
                               fit_axis='window_only_separate_horizon_joint' if phase in HORIZON_PHASES else 'window_and_horizon_per_joint',
                               floor_active_coordinates=int(((qhi-qlo) < .002).sum()))
        # Verify the actual float32 labels and saved B-axis ranges in this fit.
        # Reuse collected windows; do not reload/refit a dataset or sample draws.
        low, high = np.asarray(by_phase[phase]['low']), np.asarray(by_phase[phase]['high'])
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            z = (2*(array-low)/(high-low)-1).astype(np.float32)
            restored = (z.astype(np.float64)+1)/2*(high-low)+low+np.stack(anchors[phase])[:, None, :]
        _need(np.isfinite(z).all() and np.isfinite(restored).all(), 'Nonfinite fit roundtrip coordinate')
        interval = np.asarray(limits[phase], np.float64)
        decoded = _forward(restored, interval[:, 0, None, :], interval[:, 1, None, :])
        error = float(np.abs(decoded.astype(np.float64)-np.stack(references[phase])).max())
        _need(error <= ROUNDTRIP_TOLERANCE_RAD, 'TRAIN normalized roundtrip exceeds original tolerance')
        roundtrip_error = max(roundtrip_error, error)
    result = dict(schema=SCHEMA, version=VERSION, mode=MODE, recipe=RECIPE, horizon=10, action_width=7,
                  units='latent_rad', tau_rad=TAU_RAD, bounded_by_codec=True, quantiles=[.01, .99], quantile_method='linear',
                  minimum_full_span_latent_rad=.002, fit_episode_ids=ids, fit_window_count=len(fit_rows),
                  fit_windows_sha256=digest(fit_rows), fit_episode_windows=records,
                  fitting_weight='each explicitly supplied qualified TRAIN window exactly once; no schedule weighting',
                  fit_roundtrip_max_error_rad=roundtrip_error, fit_roundtrip_tolerance_rad=ROUNDTRIP_TOLERANCE_RAD,
                  bounds_contract=bounds, action_encoding_binding=binding, actions_by_phase=by_phase)
    result['normalization_sha256'] = digest(result)
    _need(validate_bounds_contract(bounds_contract) == bounds['bounds_sha256'], 'Bounds changed during fitting')
    validate_stats(result)
    return result


def validate_stats(stats, expected_sha256=None, expected_bounds_sha256=None):
    """Validate saved mode/source/bounds/axes/shape/integrity, without fitting."""
    _need(type(stats) is dict and stats.get('schema') == SCHEMA and stats.get('version') == VERSION
          and stats.get('mode') == MODE and stats.get('recipe') == RECIPE, 'Wrong interval normalization identity')
    bounds = stats.get('bounds_contract')
    validate_bounds_contract(bounds, expected_bounds_sha256)
    _need(stats.get('action_encoding_binding') == action_encoding_binding(bounds), 'Codec source or action binding changed')
    _need(type(stats.get('horizon')) is int and stats['horizon'] == 10 and type(stats.get('action_width')) is int
          and stats['action_width'] == 7 and stats.get('units') == 'latent_rad' and stats.get('tau_rad') == .001
          and stats.get('bounded_by_codec') is True and stats.get('quantiles') == [.01, .99]
          and stats.get('quantile_method') == 'linear' and stats.get('minimum_full_span_latent_rad') == .002, 'Interval normalization numerical contract differs')
    actual = digest({k: v for k, v in stats.items() if k != 'normalization_sha256'})
    _need(stats.get('normalization_sha256') == actual and (expected_sha256 is None or expected_sha256 == actual), 'Normalization hash differs')
    error = stats.get('fit_roundtrip_max_error_rad')
    _need(type(error) in (int, float) and np.isfinite(error) and 0 <= error <= ROUNDTRIP_TOLERANCE_RAD
          and stats.get('fit_roundtrip_tolerance_rad') == ROUNDTRIP_TOLERANCE_RAD, 'Original fit roundtrip tolerance differs')
    ids, records = stats.get('fit_episode_ids'), stats.get('fit_episode_windows')
    _need(type(ids) is list and len(ids) > 0 and all(type(e) is str and bool(e) and e.strip() == e for e in ids)
          and len(set(ids)) == len(ids) and type(records) is list and len(records) == len(ids), 'Malformed fit episode identities')
    phase_counts = {}
    for eid, record in zip(ids, records):
        _need(type(record) is dict and set(record) == {'episode_id', 'window_count', 'starts_sha256', 'phase_window_counts', 'targets'}
              and record['episode_id'] == eid and type(record['window_count']) is int and record['window_count'] > 0
              and _hash(record['starts_sha256']), 'Malformed TRAIN fit record')
        counts = record['phase_window_counts']; targets = record['targets']
        _need(type(counts) is dict and bool(counts) and set(counts) <= set(PHASES)
              and all(type(n) is int and n > 0 for n in counts.values()) and sum(counts.values()) == record['window_count'], 'Fit phase counts differ')
        _need(type(targets) is list and bool(targets) and all(type(t) is str and t in TARGETS for t in targets)
              and targets == sorted(set(targets)), 'Malformed TRAIN target membership')
        for phase, count in counts.items():
            phase_counts[phase] = phase_counts.get(phase, 0) + count
    _need(type(stats.get('fit_window_count')) is int and stats['fit_window_count'] == sum(phase_counts.values())
          and _hash(stats.get('fit_windows_sha256')) and stats.get('fitting_weight') == 'each explicitly supplied qualified TRAIN window exactly once; no schedule weighting', 'TRAIN fit count or provenance differs')
    by_phase = stats.get('actions_by_phase')
    _need(type(by_phase) is dict and set(by_phase) == set(phase_counts), 'Fitted phase membership differs')
    for phase, own in by_phase.items():
        shape = (10, 7) if phase in HORIZON_PHASES else (7,)
        _need(type(own) is dict and set(own) == {'low','high','quantile_low','quantile_high','observed_min','observed_max','window_count','residual_count','units','fit_axis','floor_active_coordinates'}, 'Phase statistic fields differ')
        low, high, qlo, qhi, minimum, maximum = [_array(own[k], shape, phase+' '+k) for k in ('low','high','quantile_low','quantile_high','observed_min','observed_max')]
        _need(np.all(minimum <= qlo) and np.all(qlo <= qhi) and np.all(qhi <= maximum), 'Invalid phase quantile range')
        with np.errstate(over='ignore', invalid='ignore'):
            center, span = (qlo+qhi)/2, np.maximum(qhi-qlo, .002)
            extreme_scale = span * float(np.finfo(np.float32).max)
        _need(np.isfinite(extreme_scale).all() and np.all(high > low)
              and np.array_equal(low, center-span/2) and np.array_equal(high, center+span/2), 'Invalid saved phase range/floor')
        axis = 'window_only_separate_horizon_joint' if phase in HORIZON_PHASES else 'window_and_horizon_per_joint'
        _need(type(own['window_count']) is int and own['window_count'] == phase_counts[phase]
              and type(own['residual_count']) is int and own['residual_count'] == 10*phase_counts[phase]
              and own['units'] == 'latent_rad' and own['fit_axis'] == axis
              and type(own['floor_active_coordinates']) is int and own['floor_active_coordinates'] == int(((qhi-qlo) < .002).sum()), 'Phase fit axis/count differs')
    return actual


def normalization_digest(stats):
    return validate_stats(stats)


def _range(phase, target, stats):
    validate_stats(stats)
    _need(type(phase) is str and phase in stats['actions_by_phase'], 'Unknown or unfitted phase')
    _limits(target, stats['bounds_contract'])
    own = stats['actions_by_phase'][phase]
    return np.asarray(own['low'], np.float64), np.asarray(own['high'], np.float64)


def encode_chunk(q_reference, anchor, phase, target, stats):
    low, high = _range(phase, target, stats)
    q = _array(q_reference, (10, 7), 'reference chunk')
    measured = _array(anchor, (7,), 'original measured anchor')
    residual = _inverse(q, *_limits(target, stats['bounds_contract']))-measured
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        encoded = (2*(residual-low)/(high-low)-1).astype(np.float32)
    _need(np.isfinite(encoded).all(), 'Encoded label outside finite float32 support')
    return encoded


def denormalize_chunk(normalized_chunk, phase, target, stats):
    """Return float64 residuals unchanged; no anchor addition or output bound map."""
    low, high = _range(phase, target, stats)
    normalized = _array(normalized_chunk, (10, 7), 'normalized action chunk')
    with np.errstate(over='ignore', invalid='ignore'):
        serialized = normalized.astype(np.float32)
        residual = (normalized+1)/2*(high-low)+low
    _need(np.isfinite(serialized).all() and np.array_equal(serialized.astype(np.float64), normalized), 'Normalized actions must retain exact finite float32 values')
    _need(np.isfinite(residual).all(), 'Nonfinite unnormalized latent residual')
    return residual


def decode_chunk(normalized_chunk, anchor, phase, target, stats):
    residual = denormalize_chunk(normalized_chunk, phase, target, stats)
    measured = _array(anchor, (7,), 'original measured anchor')
    with np.errstate(over='ignore', invalid='ignore'):
        x = residual+measured
    _need(np.isfinite(x).all(), 'Nonfinite latent coordinate')
    return _forward(x, *_limits(target, stats['bounds_contract']))
