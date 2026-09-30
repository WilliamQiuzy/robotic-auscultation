"""Measured single-scene wall cost; no extrapolated parallel performance."""
import math


def runtime_cost(*, physics_steps, physics_dt_s, probe_wall_s,
                 controller_wall_s, setup_wall_s, capture_profile):
    if type(physics_steps) is not int or physics_steps < 0:
        raise ValueError('Nonnegative actual physics step count required')
    for name, value in [('dt',physics_dt_s),('probe',probe_wall_s),
                        ('controller',controller_wall_s),('setup',setup_wall_s)]:
        if value is None and name in ('controller','setup'):
            continue
        if (isinstance(value,bool) or not isinstance(value,(int,float)) or
                not math.isfinite(value) or value < 0 or (name=='dt' and value==0)):
            raise ValueError('Invalid measured wall/physics time: '+name)
    if (controller_wall_s is None) != (setup_wall_s is None):
        raise ValueError('Setup and controller timing must both be measured or unavailable')
    remainder=None if controller_wall_s is None else probe_wall_s-controller_wall_s-setup_wall_s
    if remainder is not None and remainder < -1e-8:
        raise ValueError('Overlapping or inconsistent measured wall intervals')
    if capture_profile not in ('diagnostic_video','full_raw','training'):
        raise ValueError('Explicit capture profile required')
    simulated=physics_steps*physics_dt_s
    return dict(schema='isaac_probe_wall_cost_v1',environment_count=1,
        physics_steps=physics_steps,physics_dt_s=physics_dt_s,simulation_s=simulated,
        capture_profile=capture_profile,probe_wall_s=probe_wall_s,
        probe_setup_wall_s=setup_wall_s,controller_wall_s=controller_wall_s,
        finalize_and_audit_wall_s=None if remainder is None else max(0.,remainder),
        controller_wall_s_per_sim_s=(None if not simulated or controller_wall_s is None
                                    else controller_wall_s/simulated),
        probe_wall_s_per_sim_s=None if not simulated else probe_wall_s/simulated,
        isaac_startup_wall_s=None,render_capture_encode_wall_s=None,
        physical_validation=False,training_eligible=False,
        scope='Probe ends after audit and before result serialization; earlier Kit startup, result serialization and worker teardown are excluded. Controller interval includes planning, physics, safety and rendering/capture. Unmeasured overheads remain null; not Isaac Lab throughput.')
