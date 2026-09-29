"""Check a second Isaac host against acquired physics before scheduling work."""
import argparse
import json
import os
from pathlib import Path
import sys


def capture(app,errors,released,output,expected):
    from sim.auscultation_repair_v3.physics_profile import make_env,PhysicsProfile
    from sim.auscultation_pi05_v5.runtime_dynamics import capture_runtime_contract
    from sim.auscultation_pi05_v5.dynamics_contract import require_compatible
    env=make_env(output,PhysicsProfile(32,1,.01))
    actual=capture_runtime_contract(env)
    require_compatible(json.loads(expected.read_text()),actual,scope='simulator_recipe')
    return dict(recipe_sha256=actual['recipe_sha256'],matches_acquired_recipe=True,
                controlled_physics_steps=0,scope='Host physics recipe probe; no rollout or model acceptance')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expected',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('New probe output required')
    args.output.mkdir(parents=True)
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'fast_shutdown':False})
    from sim.auscultation_skill_v1.media_teardown import run_scoped_capture
    code=1
    try:
        payload,receipt=run_scoped_capture(app,capture,args.output,args.expected)
        (args.output/'probe.json').write_text(json.dumps(dict(**payload,teardown=receipt),indent=2)+'\n')
        print('RECIPE_PROBE',json.dumps(payload),flush=True);code=0
    except Exception as error:
        (args.output/'failure.json').write_text(json.dumps(dict(error=type(error).__name__+': '+str(error)))+'\n')
        print('RECIPE_PROBE_FAILURE',str(error),flush=True)
    sys.stdout.flush();sys.stderr.flush();os._exit(code)
