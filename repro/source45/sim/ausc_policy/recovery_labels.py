"""Compare a stationary oracle's first block against actual subsequent execution.

Diagnostic only: no array export, eligibility changes, or rolling-query claim.
"""
import numpy as np


def audit_capture_entry(receipt,physics):
    """Bind the prepared physical entry without inventing an oracle query."""
    from .recovery_oracle import measured_stationarity
    if (receipt.get('preparation_completed') is not True or receipt.get('local_alignment_used') is not False
            or receipt.get('physical_joint_resets')!=0):
        raise ValueError('Direct actual preparation without alignment or reset required')
    entry=receipt['entry'];origin=entry['physics_row']
    if type(origin) is not int or not 12<=origin<len(physics)-1:
        raise ValueError('Capture entry outside actual history')
    current=physics[origin]
    if (entry['time_s']!=current['time_s']
            or any(not np.array_equal(entry[k],current[k]) for k in ('q','dq'))
            or not np.array_equal(receipt['requested_command'],current['q_command'])
            or any(r['oracle_role']!='PREPARATION' for r in physics[:origin+1])
            or physics[origin+1]['oracle_role']!='MEASURED_ANALYTIC_L3'):
        raise ValueError('Capture origin/roles differ; no hidden oracle correction allowed')
    for index,row in enumerate(entry['recent_measurements'],start=origin-12):
        actual=physics[index]
        if (row['physics_row']!=index or row['time_s']!=actual['time_s']
                or any(not np.array_equal(row[k],actual[k]) for k in ('q','q_command'))):
            raise ValueError('Capture measured history changed')
    stationary=measured_stationarity(entry)
    if not stationary['passed']: raise ValueError('Actual capture was not stationary')
    return dict(entry_physics_row=origin,entry_time_s=current['time_s'],stationarity=stationary,
        direct_l3_first_physics_row=origin+1,training_dataset_eligible=False,
        scope='Actual direct component preparation only; no expert query label or training export')


def audit_query_labels(query,physics):
    return _audit_commands(query,physics,kind='trajectory',role='EXPERT_RECOVERY',
        state=query['measured_query']['state'],anchor='acquired measured q at actual stationary query',
        scope='First stationary-query H10 proposal executed as part of full expert trajectory; no moving-state oracle labels')


def audit_brake_labels(query,physics):
    state=query['measured_query'];origin=state['physics_row']
    if type(origin) is not int or not 12<=origin<len(physics):
        raise ValueError('Brake origin outside actual history')
    history=state['recent_measurements']
    if (len(history)!=13 or [r['physics_row'] for r in history]!=list(range(origin-12,origin+1))
            or not np.array_equal(state['command_state']['position'],physics[origin]['q_command'])):
        raise ValueError('Brake does not bind actual history and retained command')
    for recorded,actual in zip(history,physics[origin-12:origin+1]):
        if (recorded['time_s']!=actual['time_s'] or
                any(not np.array_equal(recorded[k],actual[k]) for k in ('q','q_command'))):
            raise ValueError('Brake query history differs from actual acquisition')
    return _audit_commands(query,physics,kind='braking_proposal',role='EXPERT_BRAKE',state=state,
        anchor='acquired measured q at actual moving brake query',
        scope='Braking proposal actually executed; not goal-directed moving oracle supervision or dataset qualification')


def _audit_commands(query,physics,*,kind,role,state,anchor,scope):
    if (query.get('kind')!=kind or query.get('horizon')!=10
            or query.get('executed_prefix_steps')!=5 or query.get('period_s')!=.05):
        raise ValueError('Original H10/E5 query required')
    origin=state['physics_row']
    if type(origin) is not int or not 0<=origin<len(physics):
        raise ValueError('Query origin outside acquired physics')
    row=physics[origin];q=np.asarray(state['q'],float)
    if (q.shape!=(7,) or not np.isfinite(q).all() or not np.array_equal(q,row['q'])
            or state['time_s']!=row['time_s']):
        raise ValueError('Query does not bind actual measured origin')
    commands=np.asarray(query['commands_240hz'],float)
    if (commands.ndim!=2 or commands.shape[1:]!=(7,) or len(commands)<120
            or not np.isfinite(commands).all() or origin+len(commands)>=len(physics)):
        raise ValueError('Complete acquired expert proposal required')
    executed=physics[origin+1:origin+len(commands)+1]
    if (any(r['oracle_role']!=role for r in executed)
            or not np.array_equal(np.asarray([r['q_command'] for r in executed]),commands)
            or not np.allclose([r['time_s']-state['time_s'] for r in executed],
                               np.arange(1,len(commands)+1)/240,atol=1e-8,rtol=0)):
        raise ValueError('Oracle proposal differs from actual commands, roles or future clock')
    relative=np.asarray(query['action_chunk_relative_rad'],float)
    if relative.shape!=(10,7) or not np.array_equal(relative,commands[11:120:12]-q):
        raise ValueError('Relative label shifted or rebased instead of current measured q')
    label_rows=list(range(origin+12,origin+121,12))
    return dict(h10_actual_command_match=True,proposal_physics_commands_matched=len(commands),
        query_physics_row=origin,label_physics_rows=label_rows,
        action0_time_s=physics[label_rows[0]]['time_s'],query_time_s=state['time_s'],
        action_period_s=.05,anchor=anchor,
        training_dataset_eligible=False,arrays_exported=0,
        scope=scope)
