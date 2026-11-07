"""Given-task raw setter intent, separate from issued native commands.

Each ordinary setter owns the twelve *subsequent* native ticks. A separately
owned first coarse-entry stop may interrupt a cell; it never supplies labels.
No measured positions or interpolated endpoints are inferred as raw targets.
"""
from copy import deepcopy
import hashlib
import json
import numpy as np

ACTION_SEMANTICS_ID = 'm2_given_raw_target_fixed_origin_cell_end_h20_e10_v2'
INTENT_SCHEMA = 'm2_given_joint_target_setter_v1'
AUDIT_SCHEMA = 'm2_given_live_target_tick_v1'


def need(ok, message):
    if not ok: raise ValueError(message)


def _q(value):
    q = np.asarray(value)
    need(q.shape == (7,) and q.dtype.kind in 'fiu' and np.isfinite(q).all(), 'Invalid raw target q')
    return q.astype(float)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


class IntentDispatcher:
    """The shared actual teacher/student setter, retaining interpolator q/v/a."""
    def __init__(self, trajectory, *, reference_sha256):
        self.trajectory = trajectory
        self.reference_sha256 = reference_sha256
        self.events = []
        self.current = None

    def issue(self, q, *, time_s, physics_row, context, owner, role, stop=False):
        q = _q(q)
        need(type(physics_row) is int and physics_row >= 0 and type(time_s) in (int,float) and np.isfinite(time_s) and
             abs(time_s-physics_row/240.) <= 1e-8, 'Setter native clock mismatch')
        need(stop or physics_row % 12 == 0, 'Raw intent setter must use the 20 Hz grid')
        need((stop and owner == 'L2_STOP' and role == 'given_l2_park') or
             (not stop and owner in ('TEACHER_L2', 'ACT_L2') and
              role in ('given_l2_motion', 'given_l2_history_wait')), 'Invalid raw intent owner/role')
        need(len(context) == 4 and context[2] == 0 and context[3] in ('FREE', 'APPROACH'),
             'Invalid setter reference context')
        if self.current is not None:
            previous = self.current
            elapsed = physics_row-previous['physics_row_start']
            need(not previous['offgrid_stop'] and (elapsed == 12 or (stop and 0 < elapsed <= 12)),
                 'Setter replaced before its twelve ticks or after stop')
        self.trajectory.set_target(q)
        live = _q(self.trajectory.input.target_position)
        need(np.array_equal(live, q), 'Setter live target readback mismatch')
        setter = dict(schema=INTENT_SCHEMA, action_semantics_id=ACTION_SEMANTICS_ID,
            intent_id=len(self.events), target_q=q.tolist(), issued_at_s=float(time_s),
            physics_row_start=physics_row, context=list(context), control_owner=owner,
            execution_role=role, reference_sha256=self.reference_sha256, reference_revision=0,
            offgrid_stop=bool(stop))
        setter['setter_sha256'] = hashlib.sha256(_canonical(setter)).hexdigest()
        self.events.append(deepcopy(setter)); self.current = setter
        return deepcopy(setter)

    def snapshot(self, *, time_s, physics_row):
        need(self.current is not None, 'Missing actual raw target setter')
        setter = self.current
        tick = physics_row-setter['physics_row_start']
        need(0 <= tick and (setter['offgrid_stop'] or tick < 12), 'Raw intent cell expired')
        need(abs(time_s-physics_row/240.) <= 1e-8, 'Raw intent tick clock mismatch')
        live = _q(self.trajectory.input.target_position)
        need(np.array_equal(live, setter['target_q']), 'Raw intent live target changed without setter')
        return dict(schema=AUDIT_SCHEMA, action_semantics_id=ACTION_SEMANTICS_ID,
            setter=deepcopy(setter), native_tick_in_cell=tick, live_target_q=live.tolist())


def validate_native_intents(audit, metadata):
    """Integrity is global; natural actor-stage flips only exclude windows."""
    setters = {}; previous_id = None; previous_tick = None
    for i, event in enumerate(audit):
        need('q_command' in event, 'Given native audit lacks actual q_command')
        _q(event['q_command'])
        evidence = event.get('raw_intent')
        role = event['controller_before']['subphase']
        actual_owner=event['controller_before'].get('actual_control_owner')
        need(actual_owner in ('PREPARATION','TEACHER_L2','ACT_L2','L2_STOP','L3_ENTRY',
             'ANALYTIC_L3_ALIGN','ANALYTIC_L3','STOPPED') and
             event['controller_after'].get('actual_control_owner')==actual_owner and
             event.get('actual_control_owner')==actual_owner, 'Given actual owner annotation mismatch')
        if evidence is None:
            need(role not in ('given_l2_motion', 'given_l2_history_wait', 'given_l2_park'),
                 'Missing actual raw target setter evidence')
            continue
        need(isinstance(evidence,dict) and set(evidence) == {'schema','action_semantics_id','setter',
             'native_tick_in_cell','live_target_q','live_target_after_step_q'} and evidence.get('schema') == AUDIT_SCHEMA and
             evidence.get('action_semantics_id') == ACTION_SEMANTICS_ID, 'Wrong raw target audit semantics')
        setter = deepcopy(evidence['setter']); digest = setter.pop('setter_sha256', None)
        need(set(setter) == {'schema','action_semantics_id','intent_id','target_q','issued_at_s',
             'physics_row_start','context','control_owner','execution_role','reference_sha256',
             'reference_revision','offgrid_stop'}, 'Malformed raw target setter fields')
        need(digest == hashlib.sha256(_canonical(setter)).hexdigest(), 'Corrupt raw intent ID/bytes')
        need(setter['schema'] == INTENT_SCHEMA and setter['action_semantics_id'] == ACTION_SEMANTICS_ID,
             'Wrong raw target setter semantics')
        identity = setter['intent_id']; start = setter['physics_row_start']
        need(type(identity) is int and identity >= 0 and type(start) is int and start >= 0 and
             abs(setter['issued_at_s']-start/240.) <= 1e-8, 'Late raw target setter or invalid ID')
        need(setter['reference_sha256'] == metadata['reference_sha256'] and setter['reference_revision'] == 0 and
             setter['context'][:3] == [metadata['episode_id'], metadata['target_id'], 0] and
             setter['context'][3] in ('FREE','APPROACH'), 'Raw intent reference/context changed')
        stop = setter['offgrid_stop']
        need(setter['control_owner']==actual_owner, 'Given actual owner differs from held raw setter')
        need(type(stop) is bool and ((stop and setter['control_owner'] == 'L2_STOP' and
             setter['execution_role'] == 'given_l2_park') or (not stop and start % 12 == 0 and
             setter['control_owner'] in ('TEACHER_L2','ACT_L2') and
             setter['execution_role'] in ('given_l2_motion','given_l2_history_wait'))), 'Invalid setter owner/grid')
        need(role == setter['execution_role'], 'Native role differs from actual setter owner')
        tick = i-start
        need(evidence['native_tick_in_cell'] == tick and tick >= 0 and (stop or tick < 12),
             'Raw target cell does not own exactly its subsequent ticks')
        q = _q(setter['target_q'])
        need(np.array_equal(_q(evidence['live_target_q']), q) and
             np.array_equal(_q(evidence['live_target_after_step_q']), q), 'Raw target live readback mismatch')
        if identity in setters:
            need(setters[identity] == setter and previous_id == identity and previous_tick+1 == tick,
                 'Corrupt/reused raw intent ID or nonconsecutive tick')
        else:
            need(identity == len(setters) and tick == 0, 'Missing/late setter intent ID')
            if role == 'given_l2_motion':
                need(event['controller_before']['actor']['phase'] == setter['context'][3],
                     'Raw intent stage differs from actual actor at setter issue')
            if previous_id is not None:
                prior = setters[previous_id]
                need(not prior['offgrid_stop'] and (previous_tick == 11 or (stop and previous_tick < 11)),
                     'Raw target setter changed inside its cell')
            setters[identity] = setter
        previous_id, previous_tick = identity, tick
    return setters


def raw_cell(audit, start, metadata):
    """Return a real complete future cell or None for excluded STOP/WAIT."""
    events = audit[start:start+12]
    need(len(events) == 12, 'Incomplete raw target label cell')
    evidence = events[0].get('raw_intent')
    if evidence is None: return None
    setter = evidence['setter']
    if setter['offgrid_stop'] or setter['execution_role'] != 'given_l2_motion': return None
    need(setter['physics_row_start'] == start and evidence['native_tick_in_cell'] == 0,
         'Late or missing raw target cell-start label')
    if any(x.get('raw_intent', {}).get('setter') != setter or
           x['raw_intent']['native_tick_in_cell'] != k for k,x in enumerate(events)):
        return None  # A separately owned first coarse STOP interrupts a cell.
    return _q(setter['target_q'])
