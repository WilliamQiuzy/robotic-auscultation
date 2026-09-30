"""Bounded finite-tool fit diagnostics, not a reachability or training gate.

The injected oracle owns first-contact discovery and its unchanged safety gates.
This module samples orientations, retains raw evidence, and uses clearances only
to choose where to sample next. It never changes anchors or extends the gap line.
"""
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .finite_head import (
    CONTACT_SURFACE,
    CONTACT_FACE_GAP_BAND_M,
    FIRST_CONTACT_INTERVAL_M,
    FIRST_CONTACT_RESOLUTION_M,
    _observation,
    _surface_anchor,
)


LOADED_COMPRESSION_M = .0025
NONCONTACT_CLEARANCE_FLOORS_M = {'fixed': .003, 'clamp': .003, 'shell': .0003}
COARSE_ORIENTATIONS = 1260
MAX_ORACLE_CALLS = 10000
MAX_REFINEMENT_SEEDS = 35
COARSE_TILT_DEG = (0., 5., 10., 15., 20., 25., 30., 35., 40.)
# Alternate sides before moving farther away; at zero, negative tilt samples
# the opposite tangent direction and stays within the same square family.
TILT_OFFSETS_DEG = (-1., 1., -2., 2., 0.)
AZIMUTH_OFFSETS_DEG = (0., -3.75, 3.75, -7.5, 7.5, -11.25, 11.25)
ROLL_OFFSETS_DEG = (0., -5., 5., -10., 10., -15., 15.)
REFINEMENTS_PER_SEED = len(TILT_OFFSETS_DEG)*len(AZIMUTH_OFFSETS_DEG)*len(ROLL_OFFSETS_DEG)
SCOPE = ('sampled finite tool near-contact hypotheses only; '
         'no CCD, full-robot collision, IK, physics or training qualification')


@dataclass(frozen=True)
class FitRefinementConfig:
    """At most 9,835 raw proposals and 10,000 oracle calls per anchor.

    The full 1,260-orientation coarse family is always attempted. A smaller
    max_calls can truncate refinement; the summary explicitly records this.
    """
    top_k: int = 8
    max_calls: int = MAX_ORACLE_CALLS

    def __post_init__(self):
        for name, lower, upper in (('top_k', 0, MAX_REFINEMENT_SEEDS),
                                   ('max_calls', COARSE_ORIENTATIONS, MAX_ORACLE_CALLS)):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not lower <= value <= upper:
                raise ValueError(f'{name} must be an integer in [{lower}, {upper}]')


def _basis(normal):
    x = np.cross(normal, [0., 0., 1.])
    if np.linalg.norm(x) < 1e-8:
        x = np.cross(normal, [1., 0., 0.])
    x /= np.linalg.norm(x)
    return x, np.cross(normal, x)


def _orientation(normal, x, y, tilt_deg, azimuth_rad, roll_rad):
    tilt_rad = np.deg2rad(tilt_deg)
    uv = np.tan(tilt_rad)*np.array([np.cos(azimuth_rad), np.sin(azimuth_rad)])
    if np.max(abs(uv)) > .6 + 1e-12:
        return None
    n = np.cos(tilt_rad)*normal + np.sin(tilt_rad)*(np.cos(azimuth_rad)*x + np.sin(azimuth_rad)*y)
    tangent = x - n*(x@n)
    tangent /= np.linalg.norm(tangent)
    a = np.cos(roll_rad)*tangent + np.sin(roll_rad)*np.cross(n, tangent)
    return np.column_stack((a, np.cross(-n, a), -n))


def _coarse_angles():
    # Deliberately retain FiniteHeadFit.candidates order and angular samples.
    for tilt in COARSE_TILT_DEG:
        for azimuth in ([0.] if tilt == 0 else np.linspace(0, 2*np.pi, 16, endpoint=False)):
            uv = np.tan(np.deg2rad(tilt))*np.array([np.cos(azimuth), np.sin(azimuth)])
            if np.max(abs(uv)) > .6 + 1e-12:
                continue
            for roll in np.linspace(0, 2*np.pi, 12, endpoint=False):
                yield tilt, float(azimuth), float(roll)


def _refined_angles(seed):
    for azimuth_offset in AZIMUTH_OFFSETS_DEG:
        for roll_offset in ROLL_OFFSETS_DEG:
            for tilt_offset in TILT_OFFSETS_DEG:
                yield (seed['angles_deg']['tilt'] + tilt_offset,
                       (seed['azimuth_rad'] + np.deg2rad(azimuth_offset)) % (2*np.pi),
                       (seed['roll_rad'] + np.deg2rad(roll_offset)) % (2*np.pi))


def _validate_output(found):
    if (not isinstance(found, dict) or not isinstance(found.get('passed'), (bool, np.bool_))
            or isinstance(found.get('fit_evaluations'), (bool, np.bool_))
            or not isinstance(found.get('fit_evaluations'), Integral)
            or found['fit_evaluations'] < 1):
        raise ValueError('Invalid first_contact output')
    if found.get('contact_surface')!=CONTACT_SURFACE:
        raise ValueError('Missing or retired first_contact contact surface policy')
    if found['passed']:
        # Fail closed on a contradictory injected oracle, reusing the original
        # gate without rewriting or discarding its raw output in the sink.
        if not isinstance(found.get('fit'), dict) or not _observation(found['fit'])['passed']:
            raise ValueError('Invalid first_contact accepted fit')
    else:
        if not isinstance(found.get('reason'), str) or not found['reason']:
            raise ValueError('Invalid first_contact rejection reason')
        report=found.get('rejected_fit')
        if report is not None and (not isinstance(report,dict) or report.get('contact_surface')!=CONTACT_SURFACE):
            raise ValueError('Missing or retired rejected-fit contact surface policy')


def _clearance_priority(record):
    """Ranking only: larger worst clearance margin, then total margin, wins."""
    report = record['first_contact'].get('rejected_fit')
    if report is None:
        return (float('inf'), float('inf'), record['attempt_index']), None
    try:
        margins = {part: float(report['distances_m'][part]) - floor
                   for part, floor in NONCONTACT_CLEARANCE_FLOORS_M.items()}
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('Invalid first_contact rejected distances') from error
    if not np.isfinite(list(margins.values())).all():
        raise ValueError('Invalid first_contact rejected distances')
    return (-min(margins.values()), -sum(margins.values()), record['attempt_index']), margins


def _search_definition(config):
    return dict(
        contact_surface=CONTACT_SURFACE,
        family='normalize(n + u*x + v*y), abs(u)<=0.6 and abs(v)<=0.6',
        coarse_tilt_deg=list(COARSE_TILT_DEG), coarse_azimuth_step_deg=22.5,
        coarse_roll_step_deg=30., zero_tilt_coarse_azimuth_deg=[0.],
        refinement_tilt_offsets_deg=list(TILT_OFFSETS_DEG),
        refinement_azimuth_offsets_deg=list(AZIMUTH_OFFSETS_DEG),
        refinement_roll_offsets_deg=list(ROLL_OFFSETS_DEG),
        refinement_order='seed rank, azimuth offset, roll offset, signed tilt offset',
        negative_tilt='opposite tangent direction; normal deviation is absolute tilt',
        top_k=int(config.top_k), max_calls=int(config.max_calls),
        raw_orientation_upper_bound=COARSE_ORIENTATIONS + int(config.top_k)*REFINEMENTS_PER_SEED,
        call_budget_unit='first_contact calls, not internal distance evaluations',
        ranking=('descending minimum(distance - original floor), descending sum of margins, '
                 'ascending coarse attempt_index; missing rejected_fit sorts last'),
        noncontact_clearance_floors_m=dict(NONCONTACT_CLEARANCE_FLOORS_M),
        rotation_deduplication='matrix entries rounded to 12 decimal places across both stages',
        first_contact_search_interval_m=list(FIRST_CONTACT_INTERVAL_M),
        first_contact_search_resolution_m=FIRST_CONTACT_RESOLUTION_M,
        contact_face_gap_band_m=list(CONTACT_FACE_GAP_BAND_M),
        compression_m=LOADED_COMPRESSION_M, gap_axis='command_normal',
        scope=SCOPE,
    )


def refine_finite_head_fit(oracle, position, local_normal, *, sink,
                           config=None, compression=LOADED_COMPRESSION_M):
    """Run coarse plus local refinement and stream every attempted orientation.

    ``oracle.first_contact(position, normal, rotation, compression)`` must use
    the existing FiniteHeadFit contract. ``sink(record)`` is mandatory and is
    called synchronously with a deep copy of angles, inputs, and the entire
    first_contact return value, including rejected_fit. Callers should persist
    each record as it arrives. An oracle exception additionally streams the
    attempted inputs and exception, then propagates. A sink failure propagates
    immediately without further oracle calls. No partial success is returned.

    Exhausting this finite set cannot establish global unreachability. Even a
    passing finite-tool hypothesis still requires independent robot and physics
    validation; this routine cannot qualify an expert or enable training.
    """
    if config is None:
        config = FitRefinementConfig()
    if not isinstance(config, FitRefinementConfig):
        raise TypeError('config must be FitRefinementConfig')
    if not callable(getattr(oracle, 'first_contact', None)) or not callable(sink):
        raise TypeError('oracle.first_contact and sink must be callable')
    if not np.isscalar(compression) or not np.isfinite(compression) or compression != LOADED_COMPRESSION_M:
        raise ValueError('Diagnostic compression must remain exactly 0.0025 m')
    position, normal = _surface_anchor(position, local_normal)
    x, y = _basis(normal)
    seen = set()
    candidates, failures = [], []
    counts = {'coarse': 0, 'refine': 0}
    rejections = {}
    fit_evaluations = 0
    duplicates = 0
    outside_family = 0
    budget_exhausted = False

    def attempt(stage, angles, seed_index=None):
        nonlocal fit_evaluations, duplicates, outside_family, budget_exhausted
        tilt, azimuth, roll = angles
        rotation = _orientation(normal, x, y, tilt, azimuth, roll)
        if rotation is None:
            outside_family += 1
            return True
        key = tuple(np.round(rotation.ravel(), 12))
        if key in seen:
            duplicates += 1
            return True
        if sum(counts.values()) >= config.max_calls:
            budget_exhausted = True
            return False
        seen.add(key)
        record = dict(
            contact_surface=CONTACT_SURFACE,
            attempt_index=sum(counts.values()), stage=stage, seed_attempt_index=seed_index,
            angles_deg=dict(tilt=float(tilt), azimuth=float(np.rad2deg(azimuth)), roll=float(np.rad2deg(roll))),
            azimuth_rad=float(azimuth), roll_rad=float(roll),
            position_world_m=position.tolist(), local_normal_world=normal.tolist(),
            rotation_world=rotation.tolist(), compression_m=LOADED_COMPRESSION_M,
        )
        counts[stage] += 1
        try:
            found = oracle.first_contact(position.copy(), normal.copy(), rotation.copy(), LOADED_COMPRESSION_M)
        except Exception as error:
            record.update(first_contact=None, exception=dict(type=type(error).__name__, message=str(error)))
            sink(deepcopy(record))
            raise
        record['first_contact'] = deepcopy(found)
        sink(deepcopy(record))
        _validate_output(found)
        fit_evaluations += int(found['fit_evaluations'])
        if found['passed']:
            candidates.append({**deepcopy(found), 'attempt_index': record['attempt_index'], 'stage': stage,
                               'normal_world': (-rotation[:, 2]).tolist(),
                               'rotation_world': rotation.tolist(),
                               'normal_deviation_from_surface_deg': abs(float(tilt)),
                               'angles_deg': dict(record['angles_deg']),
                               'azimuth_rad': float(azimuth), 'roll_rad': float(roll)})
        else:
            reason = found['reason']
            rejections[reason] = rejections.get(reason, 0) + 1
            priority, margins = _clearance_priority(record)
            if stage == 'coarse':
                failures.append((priority, record, margins))
        return True

    for angles in _coarse_angles():
        if not attempt('coarse', angles):
            break
    selected = sorted(failures, key=lambda item: item[0])[:config.top_k]
    for _, seed, _ in selected:
        for angles in _refined_angles(seed):
            if not attempt('refine', angles, seed['attempt_index']):
                break
        if budget_exhausted:
            break
    return dict(
        contact_surface=CONTACT_SURFACE,
        candidates=candidates, orientation_attempts=sum(counts.values()), fit_attempts=fit_evaluations,
        stage_counts=counts, rejected_orientations=rejections,
        duplicate_orientations_skipped=duplicates, outside_family_proposals_skipped=outside_family,
        selected_seed_attempts=[dict(attempt_index=seed['attempt_index'], angles_deg=seed['angles_deg'],
                                     noncontact_clearance_margins_m=margins) for _, seed, margins in selected],
        loaded_compression_m=LOADED_COMPRESSION_M,
        geometry_artifacts=deepcopy(getattr(oracle, 'artifact_hashes', {})),
        search_definition=_search_definition(config),
        budget_exhausted=budget_exhausted, search_complete=not budget_exhausted,
        outcome='finite_tool_hypotheses_found' if candidates else 'no_fit_found_in_sampled_search',
        globally_unreachable=None, scope=SCOPE,
    )
