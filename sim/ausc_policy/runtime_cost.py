"""Pure summaries of labeled wall clocks; no extrapolated parallel performance."""
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


def summarize_run(*, run_id, context_id, gpu, scene_count, resolution, capture_tier,
                  simulation_s, active_wall_s, total_wall_s, startup_wall_s=None,
                  teardown_wall_s=None, capture_wall_s=None, capture_scope=None,
                  serialization_wall_s=None, serialization_scope=None):
    """Summarize one measured run, without filling missing clocks with estimates.

    ``context_id`` identifies a fixed workload, runtime and measurement protocol;
    ``gpu`` identifies hardware (use an explicit CPU label for CPU runs).
    Simulation seconds are elapsed scene time, not summed scene-seconds.

    Startup, active, teardown and ``disjoint`` clocks are mutually exclusive
    phases of the total process lifetime. ``within_active`` capture/serialization
    clocks may overlap each other and are never added or subtracted from active.
    Only disjoint measured clocks establish separate overhead. Other overlap
    patterns must be isolated by the caller before using this schema.

    A scope may be labeled even when its duration is unavailable. If all outer
    phases are measured, they must account for total; otherwise they bound it
    from below. Uninstrumented gaps must stay unavailable, not become zero.
    """
    for name, value in dict(run_id=run_id, context_id=context_id, gpu=gpu,
                            capture_tier=capture_tier).items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'Explicit {name} label required')
    if type(scene_count) is not int or scene_count <= 0:
        raise ValueError('Positive scene_count required')
    if (not isinstance(resolution, (list, tuple)) or len(resolution) != 2 or
            any(type(value) is not int or value <= 0 for value in resolution)):
        raise ValueError('Positive [width, height] resolution required')

    durations = dict(startup=startup_wall_s, active=active_wall_s,
        simulation=simulation_s, capture=capture_wall_s,
        serialization=serialization_wall_s, teardown=teardown_wall_s, total=total_wall_s)
    for name, value in durations.items():
        if value is None and name not in ('simulation', 'active', 'total'):
            continue
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or value < 0 or
                (name == 'simulation' and value == 0)):
            raise ValueError(f'Invalid measured duration: {name}')

    scopes = dict(capture=capture_scope, serialization=serialization_scope)
    outer = [startup_wall_s, active_wall_s, teardown_wall_s]
    complete = all(value is not None for value in outer)
    for name, scope in scopes.items():
        value = durations[name]
        if scope not in (None, 'within_active', 'disjoint') or (value is not None and scope is None):
            raise ValueError(f'Explicit measured timing scope required: {name}')
        if scope == 'within_active':
            if value is not None and value > active_wall_s + 1e-8:
                raise ValueError(f'{name} exceeds its active parent interval')
        else:
            complete = complete and scope == 'disjoint' and value is not None
            if scope == 'disjoint':
                outer.append(value)
    accounted = sum(value for value in outer if value is not None)
    if not math.isfinite(accounted) or (accounted > total_wall_s and
            not math.isclose(accounted, total_wall_s, rel_tol=1e-9, abs_tol=1e-8)):
        raise ValueError('Measured disjoint components exceed total wall time')
    if complete and not math.isclose(accounted, total_wall_s, rel_tol=1e-9, abs_tol=1e-8):
        raise ValueError('Complete timing components contradict total wall time')

    ratios = {name: None if value is None else value / simulation_s
              for name, value in durations.items() if name != 'simulation'}
    if any(value is not None and not math.isfinite(value) for value in ratios.values()):
        raise ValueError('Wall time per simulation second must be finite')
    return dict(schema='runtime_cost_run_v1', run_id=run_id,
        context=dict(context_id=context_id, gpu=gpu, scene_count=scene_count,
            resolution=list(resolution), capture_tier=capture_tier,
            capture_scope=capture_scope, serialization_scope=serialization_scope),
        durations_s=durations, wall_s_per_sim_s=ratios,
        separate_overhead_s={name: durations[name] if scope == 'disjoint' else None
                             for name, scope in scopes.items()})


def _quantiles(values):
    values = sorted(value for value in values if value is not None)
    result = dict(n=len(values))
    for name, fraction in (('p50', .5), ('p90', .9), ('p95', .95)):
        if not values:
            result[name] = None
            continue
        position = (len(values) - 1) * fraction
        lower, upper = math.floor(position), math.ceil(position)
        weight = position - lower
        result[name] = values[lower] + (values[upper] - values[lower]) * weight
    return result


def aggregate_runs(runs):
    """Validate raw ``summarize_run`` keyword dictionaries and report quantiles.

    Quantiles use linear interpolation at (n - 1) * p, giving each run equal
    weight. Per-metric ``n`` counts available measurements only; it is not a
    confidence interval or a prediction of parallel/Isaac Lab throughput.
    Hardware, workload, capture settings and clock scopes must match exactly.
    """
    summaries = [summarize_run(**run) for run in runs]
    if not summaries:
        raise ValueError('At least one measured run required')
    run_ids = [run['run_id'] for run in summaries]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError('Unique run_id labels required')
    context = summaries[0]['context']
    if any(run['context'] != context for run in summaries[1:]):
        raise ValueError('Incompatible runtime measurement context')
    result = dict(schema='runtime_cost_aggregate_v1', context=context,
        sample_count=len(summaries), run_ids=run_ids,
        quantile_method='linear interpolation at (n - 1) * p')
    for group in ('durations_s', 'wall_s_per_sim_s', 'separate_overhead_s'):
        result[group] = {name: _quantiles(run[group][name] for run in summaries)
                         for name in summaries[0][group]}
    return result
