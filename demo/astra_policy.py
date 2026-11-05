"""Astra skill-request prototype for simulation; it never applies robot motion.

The cloud model chooses a bounded skill. The native local executor retains
IK, path checks, force control and actual outcome scoring. API / Isaac joint
execution has not been benchmarked by this release.
"""
import argparse
import base64
from copy import deepcopy
import json
import math
from pathlib import Path
import time

MODEL = 'gpt-6-astra'
TARGETS = {side + str(i) for side in 'LR' for i in range(1, 6)}
SKILLS = ['approach_target', 'continue_local_control', 'retract', 'abort']
SYSTEM = '''You supervise a SIMULATION-ONLY robotic auscultation task.
Select one requested skill using the visible RGB image (when supplied),
depth-derived target geometry, current joint state, Hall voltage and phase.
Only select provided pending targets. During contact/HOLD the local force
controller is authoritative: continue it, retract or abort. You cannot set
force, joint angles, thresholds, success labels or controller gains. A new
approach is permitted only from HOME or FREE. Match observation_id exactly.
Requesting a skill does not prove it was executed or succeeded.'''


def need(condition, message):
    if not condition:
        raise ValueError(message)


def finite_vector(value, size):
    return isinstance(value, list) and len(value) == size and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in value)


def validate_observation(obs):
    need(obs.get('schema') == 'ausc-astra-observation-v1' and obs.get('simulation_only') is True,
         'Explicit simulation-only observation required')
    need(isinstance(obs.get('observation_id'), str) and bool(obs['observation_id']), 'Observation identity required')
    need(obs.get('phase') in {'HOME', 'FREE', 'APPROACH', 'ALIGN', 'CONTACT', 'HOLD', 'RETRACT', 'FAILED', 'COMPLETE'},
         'Unknown local controller phase')
    need(finite_vector(obs.get('q_rad'), 7), 'Seven finite joint positions required')
    v = obs.get('hall_voltage_v')
    need(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v), 'Finite Hall voltage required')
    points = obs.get('depth_target_points_m', {})
    need(isinstance(points, dict) and bool(points) and set(points) <= TARGETS and
         all(finite_vector(v, 3) for v in points.values()), 'Finite provided depth-derived targets required')
    pending = obs.get('pending_target_ids', []); done = obs.get('completed_target_ids', [])
    need(isinstance(pending, list) and isinstance(done, list) and len(set(pending)) == len(pending) and
         len(set(done)) == len(done) and set(pending) <= set(points) and set(done) <= TARGETS and
         not set(pending) & set(done), 'Invalid pending/completed targets')
    camera = obs.get('camera', {})
    need(camera.get('frame') == 'robot_base' and camera.get('depth_units') == 'm', 'Robot-base geometry in meters required')


def build_request(obs, image_path=None):
    validate_observation(obs)
    content = [{'type': 'input_text', 'text': json.dumps(obs, allow_nan=False)}]
    if image_path is not None:
        p = Path(image_path); need(p.suffix.lower() in {'.png', '.jpg', '.jpeg'}, 'Use a PNG or JPEG observation')
        data = p.read_bytes(); need(0 < len(data) <= 5 * 1024 * 1024, 'Observation image must be at most 5 MiB')
        mime = 'image/png' if p.suffix.lower() == '.png' else 'image/jpeg'
        content.append({'type': 'input_image', 'image_url': 'data:' + mime + ';base64,' + base64.b64encode(data).decode()})
    properties = dict(observation_id={'type': 'string'}, skill={'type': 'string', 'enum': SKILLS},
                      target_id={'type': ['string', 'null'], 'enum': sorted(TARGETS) + [None]},
                      reason={'type': 'string'})
    return dict(model=MODEL, reasoning={'effort': 'medium'}, max_output_tokens=4096,
                instructions=SYSTEM, input=[{'role': 'user', 'content': content}],
                tools=[{'type': 'function', 'name': 'request_auscultation_skill',
                        'description': 'Request one simulation skill for the current observation.', 'strict': True,
                        'parameters': {'type': 'object', 'properties': properties,
                                       'required': list(properties), 'additionalProperties': False}}],
                tool_choice={'type': 'function', 'name': 'request_auscultation_skill'},
                parallel_tool_calls=False, store=False)


def validate_decision(decision, obs):
    validate_observation(obs)
    need(set(decision) == {'observation_id', 'skill', 'target_id', 'reason'}, 'Unexpected decision fields')
    need(decision['observation_id'] == obs['observation_id'], 'Stale response observation identity')
    skill = decision['skill']; target = decision['target_id']
    need(skill in SKILLS and isinstance(decision['reason'], str) and bool(decision['reason']), 'Invalid skill or reason')
    need(target is None or target in TARGETS, 'Invented anatomical target')
    if obs['phase'] in {'FAILED', 'COMPLETE'}:
        need(skill == 'abort', 'A stopped controller cannot continue motion')
    if skill == 'approach_target':
        need(obs['phase'] in {'HOME', 'FREE'} and target in obs['pending_target_ids'], 'Approach not allowed in this state')
    elif target is not None:
        need(target in obs['depth_target_points_m'], 'Target geometry is unavailable')
    return deepcopy(decision)


def propose(client, obs, image_path=None):
    started = time.monotonic()
    response = client.responses.create(**build_request(obs, image_path))
    calls = [item for item in response.output if item.type == 'function_call']
    need(len(calls) == 1 and calls[0].name == 'request_auscultation_skill', 'One supported tool request required')
    decision = validate_decision(json.loads(calls[0].arguments), obs)
    return dict(schema='ausc-astra-skill-proposal-v1', model=MODEL, decision=decision,
                elapsed_wall_s=time.monotonic() - started, response_id=response.id,
                executed=False, physical_success=None,
                usage=response.usage.model_dump() if response.usage is not None else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observation', required=True); parser.add_argument('--rgb')
    parser.add_argument('--output', required=True); parser.add_argument('--live', action='store_true')
    args = parser.parse_args(); obs = json.loads(Path(args.observation).read_text())
    if args.live:
        from openai import OpenAI
        result = propose(OpenAI(), obs, args.rgb)
    else:
        result = dict(dry_run=True, api_called=False, executed=False,
                      request=build_request(obs, args.rgb))
    with Path(args.output).open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write('\n')


if __name__ == '__main__':
    main()
