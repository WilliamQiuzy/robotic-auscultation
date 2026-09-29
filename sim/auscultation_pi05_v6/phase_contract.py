"""Shared observed pre-contact phases; acquired contact labels stay provenance.

The 90 mm switch uses the RGB-D/Hall/FK state given to the policy, never a
teacher waypoint index. Once entered, APPROACH persists until the independent
contact supervisor takes over. Each target starts a fresh pre-contact sequence.
"""
from collections import Counter

import numpy as np

VERSION = 'observed_monotonic_precontact_v1'
APPROACH_DISTANCE_M = .09
PHASES = {'FREE', 'APPROACH', 'CONTACT', 'HOLD', 'RETRACT'}


def precontact_phase(previous, observed_state):
    state = np.asarray(observed_state, dtype=np.float32)
    if previous not in {'FREE', 'APPROACH'} or state.shape != (32,) or not np.isfinite(state).all():
        raise ValueError('Invalid observed pre-contact phase/state')
    return ('APPROACH' if previous == 'APPROACH' or
            float(np.linalg.norm(state[14:17])) < APPROACH_DISTANCE_M else 'FREE')


def derive_phases(arrays, frames, pairs):
    """Relabel the exact acquired pair origins without altering their selection.

    All eligible frames participate in the monotonic selector, including frames
    excluded from action pairs at old boundaries. Original contact/hold/retract
    supervision is retained explicitly; this is not a replay of missing 240 Hz
    ADC observations and does not certify equivalence of teacher supervision.
    """
    if len(pairs) != len(arrays['phases']) or len(pairs) != len(arrays['state']):
        raise ValueError('Phase pair/state count mismatch')
    derived, current, active_target = {}, 'FREE', None
    for expected, frame in enumerate(frames):
        if frame['index'] != expected:
            raise ValueError('Invalid observed frame order')
        if not frame.get('eligible'):
            continue
        if frame['target'] != active_target:
            active_target, current = frame['target'], 'FREE'
        acquired = frame['phase']
        if acquired not in PHASES:
            raise ValueError('Unknown acquired phase')
        if acquired in {'FREE', 'APPROACH'}:
            current = precontact_phase(current, frame['state'])
            derived[expected] = current
        else:
            derived[expected] = acquired
    selected = []
    for index, pair in enumerate(pairs):
        origin = pair['frame_index']
        if origin not in derived:
            raise ValueError('Selected phase origin is not eligible')
        frame = frames[origin]
        measured = np.asarray(frame['state'], np.float32)
        if (measured.shape != (32,) or not np.isfinite(measured).all() or
                not np.array_equal(measured, arrays['state'][index])):
            raise ValueError('Frozen state differs from phase observation')
        if arrays['phases'][index] != frame['phase']:
            raise ValueError('Frozen acquired phase mismatch')
        selected.append(dict(label_index=index, frame_index=origin,
                             following_frame_index=pair['following_frame_index'],
                             acquired_phase=frame['phase'], acquired_subphase=frame['subphase'],
                             derived_phase=derived[origin]))
    phases = np.asarray([item['derived_phase'] for item in selected])
    changes = Counter(item['acquired_phase'] + '->' + item['derived_phase'] for item in selected
                      if item['acquired_phase'] != item['derived_phase'])
    return phases, dict(phase_contract_version=VERSION, approach_distance_m=APPROACH_DISTANCE_M,
                        selector='monotonic per target using measured state[14:17] norm',
                        selected_frames=selected, changed_label_count=sum(changes.values()),
                        changes=dict(changes), exact_parent_pair_selection=True,
                        cross_boundary_label_indices=[index for index,pair in enumerate(pairs)
                            if derived.get(pair['following_frame_index']) != derived[pair['frame_index']]],
                        contact_phase_source='acquired supervisor labels, not reconstructed ADC replay',
                        contact_supervisor_equivalence_verified=False,
                        chunk_rule='reject any ten-action chunk crossing derived phase, target or time boundary')
