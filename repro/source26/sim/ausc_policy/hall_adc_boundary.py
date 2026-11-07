"""Shared existing deterministic ADC-equivalent lower-end conversion.

Extracted unchanged from CompactExpertCapture. This changes only the forward
sensor model input, never raw physics, a force controller or a safety threshold.
"""
import numpy as np


def lower_endpoint_policy(calibration):
    v0=calibration.voltage_at_displacement(0.)
    slope=(calibration._volts[1]-v0)/calibration._compression_m[1]
    if not np.isfinite(slope) or slope>=0.:
        raise ValueError('Pinned Hall tare must have a decreasing finite slope')
    tolerance=float(calibration.supply_v/(2.*4095.*abs(slope)))
    code0=v0/calibration.supply_v*4095.
    if np.rint(code0)!=np.rint(code0+.5):
        raise ValueError('Half-LSB lower endpoint does not preserve the pinned zero ADC code')
    return dict(method='deterministic_adc_equivalent_lower_endpoint_v1',
        lower_tolerance_m=tolerance,voltage_slope_v_per_m=float(slope),
        native_adc_bits=12,native_adc_zero_code=int(np.rint(code0)),
        lower_endpoint_linear_quantizer_input=float(code0+.5),zero_quantizer_input=float(code0),
        noise_bias_drift_warp_zero=True,upper_clamping=False,raw_physics_unchanged=True,
        physical_safety_gates_unchanged=True,bench_verified_extrapolation=False)


def sensor_input(calibration, raw):
    """Use the exact existing bound, deriving it rather than accepting a tolerance."""
    if isinstance(raw,(bool,np.bool_)) or not isinstance(raw,(int,float,np.integer,np.floating)) or not np.isfinite(raw):
        raise ValueError('Hall raw compression must be finite numeric')
    raw=float(raw);policy=lower_endpoint_policy(calibration)
    mapped=0. if -policy['lower_tolerance_m'] <= raw < 0. else raw
    calibration.voltage_at_displacement(mapped)  # no upper clamp or lower extrapolation
    return dict(raw_compression_m=raw,sensor_compression_m=mapped,zero_boundary_delta_m=mapped-raw)
