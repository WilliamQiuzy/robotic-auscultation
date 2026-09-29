"""Bounded no-contact PhysX validation of the approved physical scene."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.argv = [sys.argv[0]]
    os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'multi_gpu': False})
    result = {'accepted': False, 'scope': 'Bounded home hold; contact force delivery must be proven by a contact episode.'}
    env = None
    try:
        from pxr import UsdGeom
        from .env import IsaacContactEnv, DT
        from .kinematics import OpenArmKinematics
        from .scene import ROOT, ROBOT
        env = IsaacContactEnv(args.output)
        kin = OpenArmKinematics()
        assert all(env.robot.get_dof_index('openarm_left_joint'+str(i)) >= 0 for i in range(1, 8))
        assert env.slider >= 0 and env.contact_subscription is not None
        initial = env.state()
        errors, forces, compressions, hall = [], [], [], []
        for step in range(480):
            state = env.step(np.zeros(7), label='home_smoke')
            for name in ['q', 'dq', 'tcp_world', 'rotation_world', 'compression_m', 'hall_N']:
                assert np.isfinite(state[name]).all(), name
            assert np.all(state['q'] >= kin.lower_limits-.005) and np.all(state['q'] <= kin.upper_limits+.005)
            wrist = env.base @ kin._chain(state['q'])[0][kin.wrist_link]
            # Preserve raw physical angles and signed slider displacement.
            expected_tcp = wrist[:3, 3] + wrist[:3, 2]*(.1262-state['compression_m'])
            errors.append(float(np.linalg.norm(expected_tcp-state['tcp_world'])))
            force = sum(env.last_contacts.get(name, 0.) for name in ['diaphragm','lip','contact_rim','carriage_other','frame','forbidden'])
            forces.append(force); compressions.append(state['compression_m']); hall.append(state['hall_N'])
            assert force < .05, ('unexpected_home_contact', force)
            assert np.max(np.abs(state['q'])) < .02, 'home joint drift'
            assert -.0001 <= state['compression_m'] < .0005, 'unloaded compression'
            assert state['hall_N'] < .2, 'unloaded Hall force'
            assert errors[-1] < .002, ('FK disagreement', errors[-1])
            assert abs(state['time_s']-(step+1)*DT) < 1e-10
        cache = UsdGeom.XformCache()
        current_base = np.asarray(cache.GetLocalToWorldTransform(env.stage.GetPrimAtPath(ROBOT+'/openarm_left_link0'))).T
        base_translation_error = float(np.linalg.norm(current_base[:3,3]-env.base[:3,3]))
        base_rotation_error = float(np.linalg.norm(current_base[:3,:3]-env.base[:3,:3]))
        # PhysX stores transforms in float32; the authored base is float64.
        assert base_translation_error <= 1e-6 and base_rotation_error <= 1e-6
        assert len(env.rows) == 480
        assert len(env.poses) == 60
        assert env.pose_row_indices == list(range(7, 480, 8))
        runtime = {'scope': result['scope'], 'world_from_left_base': env.base.tolist(),
                   'dof_names': list(env.robot.dof_names), 'initial_q': initial['q'].tolist(),
                   'settled_q': state['q'].tolist(), 'home_steps': 480,
                   'source_scene_sha256': hashlib.sha256((args.output/'training_scene.usdc').read_bytes()).hexdigest()}
        (ROOT/'runtime_inspection.json').write_text(json.dumps(runtime, indent=2)+'\n')
        result.update(accepted=True, steps=480, physics_dt_s=DT,
                      max_fk_error_m=max(errors), peak_home_contact_N=max(forces),
                      peak_home_hall_N=max(hall), compression_range_m=[min(compressions),max(compressions)],
                      max_abs_final_joint_rad=float(np.max(np.abs(state['q']))),
                      replay_frames=len(env.poses), base_static_within_numeric_tolerance=True,
                      base_translation_error_m=base_translation_error,
                      base_rotation_matrix_error=base_rotation_error,
                      authored_world_from_base=env.base.tolist(),
                      final_runtime_world_from_base=current_base.tolist(),
                      base_translation_tolerance_m=1e-6, base_rotation_matrix_tolerance=1e-6)
    except Exception as error:
        result['accepted'] = False
        result['error'] = str(error) or type(error).__name__
        raise
    finally:
        if env is not None:
            env.save('home_smoke')
        (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        print('HOME_PHYSICS_SMOKE', json.dumps(result), flush=True)
        app.close()


if __name__ == '__main__':
    main()
