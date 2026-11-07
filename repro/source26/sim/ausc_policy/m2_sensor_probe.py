"""One second of actual home-state RGB-D/Hall acquisition, not a demonstration."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np

from .geometry_epoch import require_current
from .m2_capture import M2RGBDCapture, APPROVED_CAMERA_SHA256
from .m2_sensors import M2HallStream, audit_hall_stream

CALIBRATION_SHA256 = 'ce6bcda3bef2282e07be0f44686dc007756ede57cdfcd732d35e28c1d469d054'


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def capture_home(env, checker, posture, hall, camera, *, steps=240, video_step=None):
    """Consecutive actual physics with unchanged free-space and force gates."""
    if type(steps) is not int or steps != 240:
        raise ValueError('This bounded smoke requires exactly 240 physics steps')
    for index in range(steps):
        command = np.zeros(7)
        check = checker.check(command, clearance=.003, allow_patient_contact=False,
                              compression=max(0., float(env.state()['compression_m'])))
        if check.get('clear') is not True:
            raise ValueError('command home geometry rejected: '+str(check))
        state = env.step(command, 'RAW_CAPTURE_SMOKE', 'R3',
                         dict(training_eligible=False, learned_policy=False))
        # Persist every real step even if its subsequent safety audit fails.
        sample = hall.append(index, state['time_s'], state['compression_m'])
        failure = None
        try:
            observed = checker.check(state['q'], clearance=.003, allow_patient_contact=False,
                                     compression=max(0., float(state['compression_m'])))
            if observed.get('clear') is not True:
                raise ValueError('observed home geometry rejected: '+str(observed))
            if posture(state['q']).get('clear') is not True:
                raise ValueError('observed home posture rejected')
            if not np.isfinite(state['compression_m']) or abs(state['compression_m']) > .006:
                raise ValueError('home slider stroke rejected')
            if any(not np.isfinite(value) or value != 0. for value in env.last_contacts.values()):
                raise ValueError('home smoke unexpectedly contacted an object')
        except Exception as error:
            failure = error
        try:
            if index % 12 == 11:
                try:
                    camera.capture(state, physics_row=index, phase='RAW_CAPTURE_SMOKE', target='R3',
                        hall=sample, context=dict(home_hold_only=True, expert_demonstration=False,
                                                 physical_audit_error=None if failure is None else repr(failure)))
                except Exception as capture_error:
                    if failure is not None: raise failure from capture_error
                    raise
            if failure is not None: raise failure
        finally:
            # Do not carry a traceback -> frame -> exception cycle into app.close.
            failure = None
        if video_step is not None:
            from .m2_capture import _state, _same_state, _scalar
            before = _state(state); clock = _scalar(env.world.current_time, 'world clock')
            video_step()
            _same_state(before, _state(env.state()), 'video render')
            if abs(_scalar(env.world.current_time, 'world clock')-clock) > 1e-8:
                raise ValueError('video render advanced physics clock')


def close_resources(camera, hall, recorder, env, error=None):
    """Keep partial evidence; independently attempt every owned teardown."""
    receipts = {}; errors = []
    def attempt(name, operation):
        try: receipts[name] = operation()
        except BaseException as caught: errors.append(dict(stage=name,error=repr(caught)))
    if camera is not None: attempt('camera', lambda: camera.finalize(error))
    if hall is not None: attempt('hall', lambda: hall.finalize(error))
    if env is not None:
        env.video_recorder = None
        attempt('trace', lambda: env.save('sensor_probe'))
    if recorder is not None: attempt('video', recorder.close)
    if env is not None: attempt('world_stop', env.world.stop)
    return receipts, errors


def run_probe(app, stage, output, config, mass):
    if sys.platform != 'linux' or os.environ.get('AUSC_OWNED_RUN') != '1':
        raise RuntimeError('Native sensor smoke requires the owned Linux runner')
    require_current(config)
    from .fresh_expert import FreshContactEnv, _VideoRecorder
    from .contact_collision import ContactCollisionChecker
    from .geometry_epoch import contact_frame
    from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
    from sim.auscultation_expert_layout_v1.posture import PostureLimits

    output = Path(output); binding = config['m2_sensor_probe']
    camera_path = Path(binding['camera_file']); calibration_path = Path(binding['calibration_file'])
    camera_bytes = camera_path.read_bytes(); calibration_bytes = calibration_path.read_bytes()
    if (hashlib.sha256(camera_bytes).hexdigest() != APPROVED_CAMERA_SHA256 or
            binding.get('camera_sha256') != APPROVED_CAMERA_SHA256 or
            hashlib.sha256(calibration_bytes).hexdigest() != CALIBRATION_SHA256 or
            binding.get('calibration_sha256') != CALIBRATION_SHA256):
        raise ValueError('Pinned camera or Hall calibration changed')
    geometry = dict(geometry_id=config['geometry_id'], freshly_generated=True,
        geometry_source_sha256=config['geometry_source_sha256'], geometry_artifacts=dict(
            collision_geometry_sha256=_sha(output/'geometry/collision_geometry.json'),
            collision_arrays_sha256=_sha(output/'geometry/collision_geometry.npz')))
    sources = {str(Path(inspect.getfile(obj))): _sha(inspect.getfile(obj)) for obj in
               (run_probe, FreshContactEnv, M2HallStream, M2RGBDCapture, ContactCollisionChecker)}
    env = hall = camera = recorder = None; error = failure_text = None; finished = False
    def factory(destination):
        stage.GetRootLayer().Export(str(destination)); return stage
    try:
        env = FreshContactEnv(output/'raw_sensor_physics', stage_factory=factory,
                             contact_offset_m=.132, moving_mass_kg=float(mass['mass_kg']))
        env.passive_preload_N = 1.5; env.video_recorder = None
        if np.max(np.abs(env.state()['q'])) > .03:
            raise ValueError('Sensor smoke must begin at the unchanged home state')
        kin = OpenArmKinematics(contact_frame=contact_frame)
        limits = PostureLimits.from_dict(config['posture_limits'], kin)
        checker = ContactCollisionChecker(kin, env.base, output/'geometry',
                                          {**geometry, **geometry['geometry_artifacts']})
        recorder = _VideoRecorder(env, output)
        hall = M2HallStream(output/'raw_hall', geometry_metadata=geometry,
            calibration_bytes=calibration_bytes, episode_id=binding['episode_id'])
        camera = M2RGBDCapture(env, output/'raw_rgbd', geometry_metadata=geometry,
            camera_bytes=camera_bytes, camera_sha256=APPROVED_CAMERA_SHA256,
            calibration_bytes=calibration_bytes, episode_id=binding['episode_id'],
            hall_stream=hall)
        capture_home(env, checker, lambda q: limits.evaluate(kin,env.base,q,observed=True,contact=False),
                     hall, camera, steps=binding['physics_steps'], video_step=recorder.after_step)
        finished = True
    except BaseException as caught:
        import traceback
        error = repr(caught); failure_text = traceback.format_exc()
    finally:
        receipts, errors = close_resources(camera, hall, recorder, env, error)
    if error is not None:
        errors.insert(0,dict(stage='capture',error=error))
        try: (output/'sensor_probe_failure.txt').write_text(failure_text)
        except Exception as caught: errors.append(dict(stage='failure_text_save',error=repr(caught)))
    if hall is not None and receipts.get('hall',{}).get('completed'):
        try: receipts['hall_replay'] = audit_hall_stream(hall.output)
        except Exception as caught: errors.append(dict(stage='hall_replay',error=repr(caught)))
    for path, expected in sources.items():
        try:
            if _sha(path) != expected: raise ValueError('Source changed during acquisition')
        except Exception as caught: errors.append(dict(stage='source',path=path,error=repr(caught)))
    completed = bool(finished and not errors and receipts.get('camera',{}).get('completed')
        and receipts['camera']['frames'] == 20 and receipts.get('hall',{}).get('completed')
        and receipts['hall']['physics_rows'] == 240 and receipts.get('hall_replay',{}).get('passed')
        and receipts.get('video',{}).get('frames') == 10)
    result = dict(capture_completed=completed, physics_steps=0 if env is None else len(env.rows),
        geometry=geometry, receipts=receipts, errors=errors, runtime_sources_sha256=sources,
        training_eligible=False, physical_validation=False, learned_policy=False,
        scope='One second of actual home-state raw RGB-D/Hall acquisition only; no expert or model acceptance')
    (output/'sensor_probe_result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result
