"""Budget arithmetic for an inference probe; not a physical closed-loop result."""
import numpy as np


def assess(measurements_s, *, prediction_steps, execution_steps, action_period_s,
           sensor_age_s, transport_s, max_observation_age_s):
    values=np.asarray(measurements_s,float)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all() or np.any(values<0):
        raise ValueError('invalid latency measurements')
    if type(prediction_steps) is not int or type(execution_steps) is not int or not 0<execution_steps<=prediction_steps:
        raise ValueError('invalid horizon')
    limits=np.asarray([action_period_s,sensor_age_s,transport_s,max_observation_age_s],float)
    if not np.isfinite(limits).all() or action_period_s<=0 or sensor_age_s<0 or transport_s<0 or max_observation_age_s<=0:
        raise ValueError('invalid latency budget')
    p95=float(np.percentile(values,95)); prefix=execution_steps*action_period_s
    oldest=sensor_age_s+p95+transport_s+(execution_steps-1)*action_period_s
    replenish=p95+transport_s<=prefix+1e-12
    fresh=oldest<=max_observation_age_s+1e-12
    return dict(samples=len(values),median_s=float(np.median(values)),p95_s=p95,
        prediction_horizon_s=prediction_steps*action_period_s,execution_horizon_s=prefix,
        oldest_executed_observation_age_s=oldest,can_replenish_prefix=bool(replenish),
        observation_age_passed=bool(fresh),budget_feasible_for_assumed_overheads=bool(replenish and fresh),
        closed_loop_ready=False,scope='synthetic model-path measurements plus assumed sensor/transport overhead; no measured tracker, actuator or contact dynamics')
