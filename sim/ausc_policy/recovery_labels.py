"""Compare a stationary oracle's first block against actual subsequent execution.

Diagnostic only: no array export, eligibility changes, or rolling-query claim.
"""
import numpy as np


def audit_query_labels(query,physics):
    if (query.get('kind')!='trajectory' or query.get('horizon')!=10
            or query.get('executed_prefix_steps')!=5 or query.get('period_s')!=.05):
        raise ValueError('Original stationary H10/E5 query required')
    state=query['measured_query']['state'];origin=state['physics_row']
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
    if (any(r['oracle_role']!='EXPERT_RECOVERY' for r in executed)
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
        action_period_s=.05,anchor='acquired measured q at actual stationary query',
        training_dataset_eligible=False,arrays_exported=0,
        scope='First stationary-query H10 proposal executed as part of full expert trajectory; no moving-state oracle labels')
