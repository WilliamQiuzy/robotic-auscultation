"""Bounded Astra skill protocol. This module applies no motor commands."""
from copy import deepcopy
import base64
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, build_opener, ProxyHandler

MODEL = 'gpt-6-astra'
SKILLS = ('approach_target', 'continue_local_control', 'retract', 'abort')
BOUNDARY_SKILL = dict(HOME='approach_target', PARKED='continue_local_control', RELEASED='retract')
MAX_IMAGE_BYTES = 1024 * 1024
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_RESPONSE_BYTES = 128 * 1024
SYSTEM = '''You supervise SIMULATION-ONLY robotic auscultation using safe local skills.
Read the current single-camera RGB image, metric optical-Z depth summary grid
and validity counts, measured joints, simulated calibrated Hall voltage/decoded
spring-load estimate, phase, and prescribed contact pose. The exact numeric depth
map/mask stay on Linux. A depth probe selected by projecting the prescribed pose
is a depth measurement, NOT an independent anatomical target estimate. Hall ADC
is explicitly simulated from measured slider displacement using the pinned
existing calibration, not an actual hardware ADC reading or bench pressure.
The local GivenContactExpert owns all motors, alignment, parking,
force feedback, collision checks and physical scoring. Select the boundary's
one available motion skill or abort. HOME permits approach_target; PARKED permits
continue_local_control (complete local alignment/contact/hold/release); RELEASED
permits retract (checked return home). There is no remote inference during an
ongoing contact skill. Never change joints, force thresholds, gains, success
labels or targets. Match observation_id and target_id. A request is not evidence
of execution or success. Return only the requested JSON decision; use no tools.'''


def need(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def vector(value, size):
    return isinstance(value, list) and len(value) == size and all(
        type(x) in (int, float) and math.isfinite(x) for x in value)


def validate_observation(obs):
    need(isinstance(obs, dict) and obs.get('schema') == 'ausc_astra_native_observation_v1'
         and obs.get('simulation_only') is True, 'Explicit native simulation observation required')
    need(isinstance(obs.get('observation_id'), str) and 0 < len(obs['observation_id']) <= 256,
         'Observation identity required')
    need(isinstance(obs.get('episode_id'), str) and bool(obs['episode_id']), 'Episode identity required')
    need(obs.get('boundary') in BOUNDARY_SKILL, 'Safe skill boundary required')
    need(obs.get('target_id') in {s+str(i) for s in 'LR' for i in range(1, 6)}, 'Invalid target')
    need(obs.get('target_source') == 'given_contact_pose' and
         obs.get('local_motor_assistance') is True and obs.get('local_executor') == 'GivenContactExpert',
         'Prescribed reference and analytic motor attribution required')
    need(vector(obs.get('q_rad'), 7) and vector(obs.get('dq_rad_s'), 7), 'Finite measured joints required')
    for key in ('time_s', 'hall_force_N', 'compression_m'):
        need(type(obs.get(key)) in (int, float) and math.isfinite(obs[key]), 'Finite '+key+' required')
    need(type(obs.get('physics_row')) is int and obs['physics_row'] >= 0, 'Native physics row required')
    pose = obs.get('reference_pose_base')
    need(isinstance(pose, list) and len(pose) == 4 and all(vector(r, 4) for r in pose),
         'Finite full prescribed pose required')
    digest = obs.get('reference_sha256')
    need(isinstance(digest, str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest),
         'Hash-bound prescribed reference required')
    if 'sensor_contract' in obs or 'sensor_bundle' in obs:
        validate_sensor_fields(obs)
    canonical(obs)
    return obs


def validate_sensor_fields(obs):
    """Standard-library validation also runs on the inference-only Mac."""
    need(obs.get('sensor_contract') == 'ausc_astra_rgbd_hall_adc_v1', 'RGB-D/Hall sensor contract required')
    bundle=obs.get('sensor_bundle',{});hall=bundle.get('hall',{});depth=bundle.get('depth',{})
    packet=hall.get('packet',{});decoded=hall.get('decoded',{})
    need(bundle.get('schema') == obs['sensor_contract'] and
         bundle.get('single_camera_path') == '/World/RobotCam' and
         bundle.get('physics_row_end_exclusive') == obs['physics_row'] and
         bundle.get('observed_at_s') == obs['time_s'] and hall.get('observed_at_s') == obs['time_s'],
         'Sensor/native observation binding differs')
    cal='ce6bcda3bef2282e07be0f44686dc007756ede57cdfcd732d35e28c1d469d054'
    need(packet.get('source') == 'simulated_adc' and hall.get('actual_adc_measurement') is False and
         hall.get('force_label_used') is False and hall.get('bench_verified') is False and all(v == cal for v in
         (packet.get('calibration_sha256'),decoded.get('calibration_sha256'),hall.get('calibration_sha256'))),
         'Pinned simulated Hall calibration/source required')
    need(type(packet.get('raw')) is int and 0 < packet['raw'] < 65535, 'Valid simulated ADC packet required')
    voltage=decoded.get('hall_voltage_v');force=decoded.get('spring_force_estimate_n')
    need(vector([voltage,force],2) and voltage > 0 and obs.get('hall_voltage_v') == voltage and
         obs['hall_force_N'] == force and obs.get('hall_voltage_source') == 'simulated_adc' and
         obs.get('hall_source') == 'calibration_decoded_simulated_adc_spring_load',
         'Model Hall feedback differs from calibrated voltage decode')
    times=[packet.get('sample_time_s'),packet.get('received_time_s'),hall.get('observed_at_s')]
    need(vector(times,3) and 0 <= times[0] <= times[1] <= times[2] and times[2]-times[0] <= .05+1e-9,
         'Hall observation clock/freshness differs')
    need(depth.get('units') == 'm' and depth.get('semantics') == 'metric_optical_z' and
         depth.get('annotator') == 'distance_to_image_plane' and
         depth.get('grid_shape') == [18,32] and
         type(depth.get('valid_count')) is int and depth['valid_count'] > 0,
         'Metric optical-Z depth summary required')
    grid=depth.get('median_grid_m');counts=depth.get('valid_count_grid')
    need(isinstance(grid,list) and len(grid)==18 and isinstance(counts,list) and len(counts)==18,
         'Bounded depth grid/mask summary required')
    for row,count in zip(grid,counts):
        need(isinstance(row,list) and len(row)==32 and isinstance(count,list) and len(count)==32,
             'Depth grid/mask dimensions differ')
        for value,n in zip(row,count):
            need(type(n) is int and 0 <= n <= 400 and
                 (value is None and n == 0 or vector([value],1) and value > 0 and n > 0),
                 'Invalid depth grid value/mask binding')
    need(sum(sum(row) for row in counts) == depth['valid_count'] and
         depth.get('invalid_count') == 360*640-depth['valid_count'] and
         not {'depth_m','depth_valid'} & set(depth) and
         bundle.get('raw_rgbd',{}).get('exact_arrays_transferred') is False,
         'Depth summary counts or raw transfer contract differ')


def decision_schema(obs):
    validate_observation(obs)
    properties = dict(observation_id=dict(type='string'),
        skill=dict(type='string', enum=[BOUNDARY_SKILL[obs['boundary']], 'abort']),
        target_id=dict(type=['string', 'null'], enum=[obs['target_id'], None]),
        reason=dict(type='string'))
    return dict(type='object', properties=properties, required=list(properties), additionalProperties=False)


def validate_decision(value, obs):
    validate_observation(obs)
    need(isinstance(value, dict) and set(value) == {'observation_id', 'skill', 'target_id', 'reason'},
         'Unexpected decision fields')
    need(value['observation_id'] == obs['observation_id'], 'Stale observation identity')
    need(value['skill'] in (BOUNDARY_SKILL[obs['boundary']], 'abort'), 'Skill forbidden at this boundary')
    need(value['target_id'] == obs['target_id'] or value['skill'] == 'abort' and value['target_id'] is None,
         'Decision target differs from native target')
    need(isinstance(value['reason'], str) and 0 < len(value['reason'].strip()) <= 4096, 'Reason required')
    return deepcopy(value)


def request_payload(obs, rgb):
    validate_observation(obs)
    need(rgb is None or isinstance(rgb, bytes) and 0 < len(rgb) <= MAX_IMAGE_BYTES, 'RGB image size invalid')
    digest = None if rgb is None else hashlib.sha256(rgb).hexdigest()
    need(obs.get('rgb_sha256') == digest, 'RGB/observation identity differs')
    return dict(observation=obs, rgb_png_base64=None if rgb is None else base64.b64encode(rgb).decode())


class HTTPClient:
    """Per-run bearer capability through an explicit loopback SSH tunnel."""
    def __init__(self, endpoint, token, timeout_s=150.):
        url = urlparse(endpoint)
        need(url.scheme == 'http' and url.hostname == '127.0.0.1' and url.path == '/decision'
             and not url.username and not url.password and not url.query and not url.fragment,
             'Loopback /decision endpoint required')
        need(isinstance(token, str) and bool(token) and '\n' not in token and '\r' not in token,
             'Explicit per-run bearer token required')
        need(type(timeout_s) in (int, float) and 0 < timeout_s <= 180, 'Bounded HTTP timeout required')
        self.endpoint, self.token, self.timeout_s = endpoint, token, float(timeout_s)
        self.opener = build_opener(ProxyHandler({}))

    def __call__(self, obs, rgb):
        data = canonical(request_payload(obs, rgb))
        need(len(data) <= MAX_REQUEST_BYTES, 'Request too large')
        req = Request(self.endpoint, data=data, headers={'Content-Type':'application/json',
            'Authorization':'Bearer '+self.token}, method='POST')
        with self.opener.open(req, timeout=self.timeout_s) as response:
            raw = response.read(MAX_RESPONSE_BYTES+1)
        need(len(raw) <= MAX_RESPONSE_BYTES, 'Response too large')
        result = json.loads(raw)
        need(result.get('model') == MODEL and result.get('backend') == 'codex_exec', 'Inference source mismatch')
        for name, path in [('service_source_sha256', Path(__file__).with_name('astra_service.py')),
                           ('protocol_source_sha256', Path(__file__))]:
            need(result.get(name) == hashlib.sha256(path.read_bytes()).hexdigest(), 'Astra service source mismatch')
        validate_decision(result.get('decision'), obs)
        return result
