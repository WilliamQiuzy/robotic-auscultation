"""Separate axial Hall proxy control from independent normal-contact evidence.

The opt-in margin waits near the existing 4 N setpoint before crediting HOLD.
It never changes the sensor, dwell duration, safety caps or physical audit.
Native normal/friction impulses are diagnostic labels only, not actor inputs.
"""
from dataclasses import replace
import numpy as np

MODE = 'axial_proxy_margin_v1'


def hold_config(config, setting):
    if setting is None:
        return config
    if type(setting) is not dict or setting != {'mode': MODE}:
        raise ValueError('force_feedback requires mode: '+MODE)
    if config.target_force_N != 4. or config.force_tolerance_N < .1:
        raise ValueError('force_feedback requires existing 4 N target and cannot widen a hold band')
    return replace(config, force_tolerance_N=.1)


def balance_receipt(report, path_name, allowed_pair, moving_path, state,
                    slider_velocity, mass, dt):
    """Read native pair/anchor offsets and orient both impulses onto carriage.

    PhysX contact impulse uses collider0 orientation. Friction anchors have the
    same pair binding. Keep vectors and signed projections so sign/closure can
    be checked against native measurements, not selected to fit the Hall value.
    """
    if len(report) != 3:
        raise ValueError('Full normal/friction report required')
    headers, points, anchors = report
    outward = -np.asarray(state['axis_world'], float)
    scalars = [state['compression_m'], state['hall_raw_N'], state['hall_N'], slider_velocity, mass, dt]
    if (outward.shape != (3,) or not np.isfinite(outward).all()
            or not np.isclose(np.linalg.norm(outward), 1.)
            or not np.isfinite(scalars).all() or mass <= 0 or dt <= 0):
        raise ValueError('Invalid force-balance state')
    normal = np.zeros(3); friction = np.zeros(3); normal_total = 0.; pairs = []
    for h in headers:
        a, b = path_name(h.collider0), path_name(h.collider1)
        if not allowed_pair(a, b):
            continue
        if moving_path(a) == moving_path(b):
            raise ValueError('Ambiguous moving contact body')
        sign = 1. if moving_path(a) else -1.
        n = np.zeros(3); f = np.zeros(3); pair_total = 0.
        for array, start, count, kind in (
                (points, h.contact_data_offset, h.num_contact_data, 'normal'),
                (anchors, h.friction_anchors_offset, h.num_friction_anchors_data, 'friction')):
            if start < 0 or count < 0 or start+count > len(array):
                raise ValueError('Incomplete native '+kind+' report')
            for index in range(start, start+count):
                impulse = np.asarray(array[index].impulse, float)
                if impulse.shape != (3,) or not np.isfinite(impulse).all():
                    raise ValueError('Invalid native impulse')
                force = sign*impulse/dt
                if kind == 'normal':
                    n += force; pair_total += float(np.linalg.norm(force))
                else:
                    f += force
        normal += n; friction += f; normal_total += pair_total
        pairs.append(dict(colliders=[a,b],normal_force_vector_N=n.tolist(),
                          friction_force_vector_N=f.tolist(),friction_anchor_count=int(h.num_friction_anchors_data)))
    spring = 1.5+1000*float(state['compression_m'])
    damping = 3.2*float(slider_velocity)
    gravity = float(mass*9.81*outward[2])
    combined = float((normal+friction)@outward)
    return dict(schema='native_axial_force_balance_v1',actor_input=False,
        normal_force_N=normal_total,normal_force_vector_N=normal.tolist(),
        friction_force_vector_N=friction.tolist(),normal_axial_N=float(normal@outward),
        friction_axial_N=float(friction@outward),combined_axial_N=combined,
        hall_raw_N=float(state['hall_raw_N']),hall_filtered_N=float(state['hall_N']),
        spring_preload_N=spring,damping_N=damping,gravity_N=gravity,
        slider_velocity_m_s=float(slider_velocity),
        axial_model_minus_report_N=spring+damping+gravity-combined,
        pairs=pairs,scope='read-only native labels; acceleration and solver residual not removed')
