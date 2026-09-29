"""CPU contracts for the pinned cuRobo geometry bridge."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


class BridgeAvailability(unittest.TestCase):
    def test_bridge_implementation_exists(self):
        self.assertIsNotNone(importlib.util.find_spec('sim.curobo_expert_v1.robot_bridge'))


class RobotBridgeTests(unittest.TestCase):
    def setUp(self):
        from sim.curobo_expert_v1 import robot_bridge
        self.bridge = robot_bridge

    def test_inverse_world_transform_preserves_points_and_rejects_reflection(self):
        t = np.array([[1, 0, 0, .2], [0, 0, 1, .331], [0, -1, 0, 1.376], [0, 0, 0, 1.]])
        points = np.array([[.3, .9, 1.2], [-.2, .6, 1.1]])
        base = self.bridge.points_in_base(points, t)
        np.testing.assert_allclose(base @ t[:3, :3].T + t[:3, 3], points, atol=1e-12)
        t[0, 0] = -1
        with self.assertRaises(ValueError):
            self.bridge.points_in_base(points, t)

    def test_conservative_cells_cover_volume_not_only_vertices(self):
        vertices = np.array([[x, y, z] for x in [-.042, .042] for y in [-.004, .004] for z in [-.001, .001]])
        spheres, report = self.bridge.cover_convex_mesh(vertices, max_radius=.008)
        self.assertGreater(len(spheres), 3)
        self.assertLessEqual(max(s['radius'] for s in spheres), .008 + 1e-9)
        rng = np.random.default_rng(4)
        sample = rng.uniform(vertices.min(0), vertices.max(0), (1000, 3))
        self.assertLessEqual(self.bridge.maximum_uncovered_distance(sample, spheres), 1e-9)
        self.assertEqual(report['guarantee'], 'entire_convex_hull_volume')
        with self.assertRaises(ValueError):
            self.bridge.cover_convex_mesh(vertices * np.nan)

    def test_tool_frames_match_independent_contact_kinematics(self):
        from sim.auscultation_expert_layout_v1.tool_frames import wrist_from_diaphragm
        fixed, carriage, diaphragm = self.bridge.tool_transforms()
        np.testing.assert_allclose(fixed[:3, :3], np.diag([1., -1., -1.]))
        np.testing.assert_allclose(fixed[:3, 3], [0, 0, .085])
        np.testing.assert_allclose(carriage, fixed)
        np.testing.assert_allclose(diaphragm, wrist_from_diaphragm())

    def test_real_export_has_every_collider_and_verified_relocatable_sources(self):
        from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'bridge'
            result = self.bridge.export_bridge(output)
            robot, world, manifest = self.bridge.load_bridge(output)
            kin = robot['robot_cfg']['kinematics']
            self.assertTrue(Path(kin['urdf_path']).is_absolute())
            self.assertEqual(kin['cspace']['joint_names'], list(OpenArmKinematics().joint_names))
            self.assertEqual(kin['base_link'], 'openarm_left_link0')
            self.assertEqual(kin['ee_link'], 'stetho_diaphragm')
            original_kin = OpenArmKinematics()
            derived_kin = OpenArmKinematics(kin['urdf_path'])
            probe = (original_kin.lower_limits + original_kin.upper_limits) / 2
            for link, transform in original_kin.link_transforms(probe).items():
                np.testing.assert_allclose(derived_kin.link_transforms(probe)[link], transform, atol=1e-12)
            self.assertEqual(len(manifest['collider_ledger']), 125)
            self.assertEqual(sum(x['role'] == 'moving_robot' for x in manifest['collider_ledger']), 98)
            self.assertIn('openarm_left_link0', kin['collision_link_names'])
            self.assertIn('openarm_left_link1', kin['self_collision_ignore']['openarm_left_link0'])
            self.assertNotIn('openarm_left_link3', kin['self_collision_ignore']['openarm_left_link0'])
            self.assertTrue(any('right_link' in x['source_body'] for x in manifest['collider_ledger'] if x['role'] == 'world'))
            self.assertEqual(len(world['mesh']), 25)
            self.assertEqual(manifest['intentionally_omitted_colliders'][0]['key'], 'c002')
            self.assertEqual(manifest['intentionally_omitted_colliders'][0]['independent_gate'], 'original_full_mesh_FCL')
            self.assertLess(manifest['sphere_count'], 1024)
            self.assertEqual(manifest['self_collision_fit']['final_overlapping_pairs'], 0)
            self.assertGreater(manifest['self_collision_fit']['iterations'], 0)
            self.assertTrue(all(Path(x['file_path']).is_absolute() for x in world['mesh'].values()))
            from sim.curobo_expert_v1.static_world import build_static_world, load_static_world
            bundle = build_static_world()
            loaded = load_static_world(output)
            self.assertEqual(manifest['static_world_contract_sha256'], bundle['contract_sha256'])
            self.assertEqual(loaded['contract_sha256'], bundle['contract_sha256'])
            for key in ('c013', 'c020', 'c021'):
                expected_world = bundle['geometry'][key]['vertices_world']
                actual_base = np.array([[float(v) for v in line.split()[1:]] for line in (output/f'world_meshes/{key}.obj').read_text().splitlines() if line.startswith('v ')])
                base = np.asarray(manifest['world_from_left_base'])
                np.testing.assert_allclose(actual_base @ base[:3, :3].T + base[:3, 3], expected_world, atol=1e-12)
                np.testing.assert_array_equal(loaded['geometry'][key]['vertices_world'], expected_world)
                self.assertEqual(loaded['geometry'][key]['sha256'], bundle['geometry'][key]['sha256'])
            self.assertEqual(manifest['unsupported_source_geometry'][0]['resolution'], 'covered_by_exported_ground_mesh_plus_conservative_floor_cuboid')
            self.assertEqual(result['manifest_path'], str(output.resolve()/'bridge_manifest.json'))
            self.assertEqual(self.bridge.verify_bridge(output)['status'], 'passed')
            with self.assertRaisesRegex(ValueError, 'empty'):
                self.bridge.export_bridge(output)
            obj = next(output.glob('world_meshes/*.obj'))
            obj.write_text(obj.read_text() + '\n# changed\n')
            with self.assertRaisesRegex(ValueError, 'hash'):
                self.bridge.load_bridge(output)

    def test_verifier_rejects_missing_hash_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'bridge'
            self.bridge.export_bridge(output)
            path = output/'bridge_manifest.json'
            manifest = json.loads(path.read_text())
            del manifest['output_sha256']['robot_config.json']
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'anchor'):
                self.bridge.verify_bridge(output)

    def test_fk_response_requires_exact_order_and_all_links(self):
        from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
        kin = OpenArmKinematics()
        q = (kin.lower_limits + kin.upper_limits) / 2
        reference = self.bridge.reference_fk([q])
        response = {'joint_names': list(kin.joint_names), 'link_names': reference['link_names'], 'poses': reference['poses']}
        self.assertEqual(self.bridge.validate_backend_fk([q], response)['status'], 'passed')
        response['joint_names'] = response['joint_names'][::-1]
        with self.assertRaisesRegex(ValueError, 'joint'):
            self.bridge.validate_backend_fk([q], response)
        response['joint_names'] = list(kin.joint_names)
        response['poses'][0][0][0] += .001
        with self.assertRaisesRegex(ValueError, 'FK'):
            self.bridge.validate_backend_fk([q], response)

    def test_backend_native_fk_schema_requires_base_frame_and_success(self):
        from sim.auscultation_expert_layout_v1.kinematics import OpenArmKinematics
        kin = OpenArmKinematics()
        q = (kin.lower_limits + kin.upper_limits) / 2
        reference = self.bridge.reference_fk([q])
        response = {'success': True, 'metadata': {'joint_names': list(kin.joint_names),
                    'pose_frame': 'robot_base', 'quaternion_order': 'wxyz'},
                    'link_poses': {name: [reference['poses'][0][i]] for i, name in enumerate(reference['link_names'])}}
        self.assertEqual(self.bridge.validate_backend_fk([q], response)['status'], 'passed')
        response['metadata']['pose_frame'] = 'world'
        with self.assertRaisesRegex(ValueError, 'frame'):
            self.bridge.validate_backend_fk([q], response)


if __name__ == '__main__':
    unittest.main()
