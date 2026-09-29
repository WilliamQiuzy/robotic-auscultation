"""Explicit sensor stress settings; engineering assumptions, not measured noise."""
import math
from sim.auscultation_vla_v2.hall import HallSimulator

DEFAULT=dict(noise_std_v=0.,voltage_bias_v=0.,latency_s=0.,curve_warp=0.,seed=0)


def validate_condition(value=None):
    if value is None:value={}
    if not isinstance(value,dict) or set(value)-set(DEFAULT):raise ValueError('Unknown Hall condition fields')
    result=dict(DEFAULT,**value)
    for k in ('noise_std_v','voltage_bias_v','latency_s','curve_warp'):
        if type(result[k]) not in (int,float) or not math.isfinite(result[k]):raise ValueError('Nonfinite Hall condition')
    if (not 0<=result['noise_std_v']<=.003 or abs(result['voltage_bias_v'])>.003
        or not 0<=result['latency_s']<=.03 or abs(result['curve_warp'])>.05
        or type(result['seed']) is not int or result['seed']<0):raise ValueError('Hall stress outside declared domain')
    return result


def make_sensor(calibration,condition=None):
    return HallSimulator(calibration,**validate_condition(condition))
