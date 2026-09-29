import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


class StaticWorldTests(unittest.TestCase):
    def test_static_world_implementation_exists(self):
        self.assertIsNotNone(importlib.util.find_spec('sim.curobo_expert_v1.static_world'))

    def test_entire_finger_stroke_and_pose_budget_inside_conservative_volume(self):
        from sim.curobo_expert_v1.static_world import build_static_world
        bundle = build_static_world()
        contract = bundle['contract']
        rng = np.random.default_rng(7)
        for key, entry in bundle['geometry'].items():
            if entry['kind'] != 'finger_sweep':
                continue
            spec = contract['finger_specs'][entry['body']]
            parent = np.asarray(contract['body_reference_poses'][spec['parent']])
            vertices = entry['source_vertices_body']
            hull = ConvexHull(entry['vertices_world']).equations
            for q in np.linspace(0, .044, 9):
                relative = np.eye(4)
                relative[:3, 3] = np.asarray(spec['origin_xyz']) + q*np.asarray(spec['axis'])
                # Combined rigid parent drift and non-prismatic child residual.
                parent_actual = parent.copy()
                parent_actual[:3, :3] = parent[:3, :3] @ Rotation.from_rotvec([.001, 0, 0]).as_matrix()
                parent_actual[:3, 3] += [.0002, 0, 0]
                relative[:3, :3] = Rotation.from_rotvec([0, .001, 0]).as_matrix()
                relative[:3, 3] += [0, 0, .0002]
                world = parent_actual @ relative
                points = vertices @ world[:3, :3].T + world[:3, 3]
                self.assertLessEqual(float((points @ hull[:, :3].T + hull[:, 3]).max()), 1e-9)
            self.assertGreater(entry['padding_m'], .0004)

    def test_full_stroke_allowed_but_non_sliding_motion_rejected(self):
        from sim.curobo_expert_v1.static_world import build_static_world, validate_static_poses
        bundle = build_static_world(); c = bundle['contract']
        poses = {name: np.array(t) for name, t in c['body_reference_poses'].items()}
        for name, spec in c['finger_specs'].items():
            rel = np.eye(4); rel[:3, 3] = np.array(spec['origin_xyz']) + .044*np.array(spec['axis'])
            poses[name] = poses[spec['parent']] @ rel
        self.assertTrue(validate_static_poses(poses, c)['clear'])
        finger = next(iter(c['finger_specs']))
        poses[finger][:3, 3] += [.001, 0, 0]
        self.assertFalse(validate_static_poses(poses, c)['clear'])

    def test_other_arm_drift_missing_pose_and_nonfinite_cannot_pass(self):
        from sim.curobo_expert_v1.static_world import build_static_world, validate_static_poses
        contract = build_static_world()['contract']
        poses = {name: np.array(t) for name, t in contract['body_reference_poses'].items()}
        arm = next(n for n in poses if n.endswith('openarm_right_link4'))
        poses[arm][0, 3] += .001
        self.assertFalse(validate_static_poses(poses, contract)['clear'])
        poses[arm][0, 3] = np.nan
        self.assertFalse(validate_static_poses(poses, contract)['clear'])
        del poses[arm]
        self.assertFalse(validate_static_poses(poses, contract)['clear'])

    def test_joint_limit_excess_is_rejected_despite_inflation(self):
        from sim.curobo_expert_v1.static_world import build_static_world, validate_static_poses
        contract = build_static_world()['contract']
        poses = {name: np.array(t) for name, t in contract['body_reference_poses'].items()}
        name, spec = next(iter(contract['finger_specs'].items()))
        rel = np.eye(4); rel[:3, 3] = np.array(spec['origin_xyz']) + .045*np.array(spec['axis'])
        poses[name] = poses[spec['parent']] @ rel
        self.assertFalse(validate_static_poses(poses, contract)['clear'])

    def test_fcl_adapter_installs_same_world_geometry_and_preserves_pairs(self):
        from sim.curobo_expert_v1.static_world import build_static_world, install_fcl_static_world, geometry_sha
        bundle = build_static_world()
        entries = [{'key': key, 'body': value['body']} for key, value in bundle['geometry'].items()]
        entries.append({'key': 'dynamic', 'body': 'dynamic'})
        checker = SimpleNamespace(entries=entries, dynamic=[len(entries)-1], objects=[object() for _ in entries],
                                  bounds=np.zeros((len(entries), 2, 3)), local_corners=[None for _ in entries],
                                  pairs=[(len(entries)-1, index) for index in range(len(entries)-1)])
        dynamic_object, pairs = checker.objects[-1], checker.pairs.copy()
        class Model:
            def beginModel(self, vertices, faces): pass
            def addSubModel(self, vertices, faces): self.vertices, self.faces = vertices.copy(), faces.copy()
            def endModel(self): pass
        fake = SimpleNamespace(BVHModel=Model, Transform=lambda rotation, translation: (rotation, translation),
                               CollisionObject=lambda model, transform: SimpleNamespace(model=model, transform=transform))
        with patch.dict('sys.modules', {'fcl': fake}), patch('sim.curobo_expert_v1.static_containment.install_static_containment', return_value={'tested_separately': True}):
            report = install_fcl_static_world(SimpleNamespace(checker=checker), bundle)
        self.assertEqual(report['contract_sha256'], bundle['contract_sha256'])
        self.assertEqual(checker.pairs, pairs)
        self.assertIs(checker.objects[-1], dynamic_object)
        for index, entry in enumerate(entries[:-1]):
            value = bundle['geometry'][entry['key']]; obj = checker.objects[index]
            self.assertEqual(geometry_sha(obj.model.vertices, obj.model.faces), value['sha256'])
            np.testing.assert_array_equal(obj.transform[0], np.eye(3))
            np.testing.assert_array_equal(obj.transform[1], np.zeros(3))
            np.testing.assert_array_equal(checker.bounds[index], [value['vertices_world'].min(0), value['vertices_world'].max(0)])

    def test_monitor_reads_each_body_once_and_records_failed_step(self):
        from sim.curobo_expert_v1.static_world import build_static_world, StaticWorldMonitor
        bundle = build_static_world()
        class Body:
            def __init__(self, matrix): self.matrix=np.array(matrix); self.calls=0
            def get_world_pose(self):
                self.calls += 1
                q=Rotation.from_matrix(self.matrix[:3, :3]).as_quat()[[3, 0, 1, 2]]
                return self.matrix[:3, 3], q
        env=SimpleNamespace(rows=[], bodies=[(name, Body(t)) for name, t in bundle['contract']['body_reference_poses'].items()])
        with patch('sim.curobo_expert_v1.static_world.install_fcl_static_world', return_value={}):
            monitor=StaticWorldMonitor(env, None, bundle)
        self.assertTrue(all(body.calls == 1 for _, body in env.bodies))
        env.rows.append({})
        monitor.after_step()
        arm=next(body for name, body in env.bodies if name.endswith('openarm_right_link4'))
        arm.matrix[0, 3] += .001
        env.rows.append({})
        with self.assertRaisesRegex(ValueError, 'verified envelope'):
            monitor.after_step()
        self.assertFalse(env.rows[-1]['static_world_check']['clear'])
        self.assertEqual(monitor.summary()['checked_physics_steps'], 2)

    def test_controller_rejects_unreleased_compression_for_both_policies(self):
        from sim.curobo_expert_v1.controller import ExpertController
        from sim.auscultation_expert_layout_v1.controller import EpisodeRejected
        for policy in ('curobo', 'revised_shortcut'):
            for compression in (.00015001, float('nan'), float('inf')):
                actor=ExpertController.__new__(ExpertController)
                actor.sequence_index=1; actor.generator_policy=policy
                actor.env=SimpleNamespace(state=lambda: {'compression_m': compression})
                with self.assertRaisesRegex(EpisodeRejected, 'compression'):
                    actor._move_to_prepose({})

    def test_controller_checks_static_world_before_and_after_every_tick(self):
        from sim.curobo_expert_v1.controller import ExpertController
        from sim.expert_quality_20260924.candidate_controller import CandidateController
        from sim.auscultation_expert_layout_v1.controller import EpisodeRejected
        actor=ExpertController.__new__(ExpertController)
        calls=[]; actor.env=SimpleNamespace(rows=[])
        actor.static_world=SimpleNamespace(check=lambda: calls.append('before'), after_step=lambda: calls.append('after'))
        def step(*a, **k):
            calls.append('tick');actor.env.rows.append({});return {'sample': 1}
        with patch.object(CandidateController, '_tick', side_effect=step):
            self.assertEqual(actor._tick(np.zeros(7), 'test'), {'sample': 1})
        self.assertEqual(calls, ['before', 'tick', 'after'])
        actor.static_world.check=lambda: (_ for _ in ()).throw(ValueError('invalid static pose'))
        with patch.object(CandidateController, '_tick') as step:
            with self.assertRaisesRegex(EpisodeRejected, 'static'):
                actor._tick(np.zeros(7), 'test')
            step.assert_not_called()


if __name__ == '__main__':
    unittest.main()
