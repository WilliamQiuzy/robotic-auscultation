"""CPU contract tests; GPU acceptance is recorded by probe_curobo.py."""
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import tempfile
import time
import unittest

import numpy as np


def backend(test):
    test.assertIsNotNone(importlib.util.find_spec("sim.curobo_expert_v1.curobo_backend"))
    return importlib.import_module("sim.curobo_expert_v1.curobo_backend")


class BackendContractTest(unittest.TestCase):
    def test_rejects_nonseven_joint_input_before_cuda_import(self):
        b = backend(self)
        with self.assertRaisesRegex(ValueError, "seven"):
            b.validate_joint_vector([0.0] * 6)

    def test_rejects_nonfinite_joint_input(self):
        b = backend(self)
        with self.assertRaisesRegex(ValueError, "finite"):
            b.validate_joint_vector([0, 0, 0, 0, 0, 0, np.nan])

    def test_reorders_planner_output_to_declared_joint_order(self):
        b = backend(self)
        names = [f"j{i}" for i in range(7)]
        q = np.tile(np.arange(7)[::-1], (3, 1))
        ordered = b.ordered_path(q, names[::-1], names)
        np.testing.assert_array_equal(ordered, np.tile(np.arange(7), (3, 1)))

    def test_missing_joint_is_not_silently_accepted(self):
        b = backend(self)
        with self.assertRaisesRegex(ValueError, "joint"):
            b.ordered_path(np.zeros((2, 7)), [f"j{i}" for i in range(7)], [f"k{i}" for i in range(7)])

    def test_planner_failure_never_returns_a_fallback_path(self):
        b = backend(self)
        r = b.BackendResult.failed("GRAPH_FAIL", "No graph path", {"planner": "cuRobo"})
        self.assertFalse(r.success)
        self.assertEqual(r.path.shape, (0, 7))
        self.assertEqual(r.to_dict()["failure"]["code"], "GRAPH_FAIL")

    def test_endpoint_residual_is_a_gate(self):
        b = backend(self)
        with self.assertRaisesRegex(ValueError, "endpoint"):
            b.validate_output_path(np.zeros((3, 7)), np.zeros(7), np.ones(7), 0.002)

    def test_backend_metadata_does_not_claim_exact_collision_or_physics(self):
        b = backend(self)
        m = b.base_metadata()
        self.assertEqual(m["curobo_expected_commit"], "2fbffc35225398cf9d5f382804faa9de2608753b")
        self.assertEqual(m["exact_geometry_audit"], "required_separately")
        self.assertEqual(m["contact_physics"], "not_executed")

    def test_guard_memory_parse_uses_available_not_free(self):
        self.assertIsNotNone(importlib.util.find_spec("sim.curobo_expert_v1.guarded_run"))
        guard = importlib.import_module("sim.curobo_expert_v1.guarded_run")
        self.assertEqual(guard.mem_available_gib("MemFree: 1 kB\nMemAvailable: 9437184 kB\n"), 9.0)

    def test_sphere_count_over_kernel_limit_is_rejected_before_cuda(self):
        b = backend(self)
        cfg = {"kinematics": {"cspace": {"joint_names": [f"j{i}" for i in range(7)]},
                              "collision_spheres": {"tool": [{}] * 1024}}}
        with self.assertRaisesRegex(ValueError, "1024"):
            b._prepare_robot(cfg)

    def test_timeout_cleans_owned_group_and_preserves_unrelated_process(self):
        guard = importlib.import_module("sim.curobo_expert_v1.guarded_run")
        self.assertTrue(hasattr(guard, "run_guard"))
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                mem = root / "meminfo"
                mem.write_text("MemAvailable: 9437184 kB\n")
                start = time.monotonic()
                code = guard.run_guard([sys.executable, "-c", "import time; time.sleep(30)"],
                    minimum_gib=8, lock_path=root / "lock", record_path=root / "record.json",
                    timeout_seconds=1, meminfo_path=mem)
                self.assertEqual(code, 124)
                self.assertLess(time.monotonic() - start, 5)
                report = json.loads((root / "record.json").read_text())
                self.assertTrue(report["timed_out"])
                self.assertIsNotNone(report["returncode"])
                self.assertIsNone(unrelated.poll())
        finally:
            unrelated.terminate()
            unrelated.wait()

    def test_sigterm_cleans_child_and_writes_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mem = root / "meminfo"
            mem.write_text("MemAvailable: 9437184 kB\n")
            pidfile = root / "child.pid"
            record = root / "record.json"
            child_code = f"import os,time;open({str(pidfile)!r},'w').write(str(os.getpid()));time.sleep(30)"
            code = ("from pathlib import Path;from sim.curobo_expert_v1.guarded_run import run_guard;"
                    f"raise SystemExit(run_guard({[sys.executable, '-c', child_code]!r},minimum_gib=8,"
                    f"lock_path=Path({str(root / 'lock')!r}),record_path=Path({str(record)!r}),"
                    f"meminfo_path=Path({str(mem)!r})))")
            guard = subprocess.Popen([sys.executable, "-c", code])
            try:
                deadline = time.monotonic() + 5
                while not pidfile.exists() and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertTrue(pidfile.exists())
                owned_pid = int(pidfile.read_text())
                guard.send_signal(signal.SIGTERM)
                self.assertEqual(guard.wait(timeout=5), 143)
                self.assertEqual(json.loads(record.read_text())["interrupted_signal"], signal.SIGTERM)
                with self.assertRaises(ProcessLookupError):
                    os.kill(owned_pid, 0)
            finally:
                if guard.poll() is None:
                    guard.kill()
                    guard.wait()

    def test_fk_and_plan_config_binding_includes_actual_asset_bytes(self):
        b = backend(self)
        self.assertTrue(hasattr(b, "configuration_metadata"))
        with tempfile.TemporaryDirectory() as temp:
            urdf = Path(temp) / "robot.urdf"
            urdf.write_text("<robot name='test'/>")
            robot = {"kinematics": {"urdf_path": str(urdf)}}
            meta = b.configuration_metadata(robot, {})
            self.assertEqual(meta["robot_config_sha256"], b._sha(robot))
            self.assertEqual(meta["asset_sha256"][str(urdf)], b.hashlib.sha256(urdf.read_bytes()).hexdigest())

    def test_reordered_path_is_contiguous_for_cuda_kernel(self):
        b = backend(self)
        names = [f"j{i}" for i in range(7)]
        q = b.ordered_path(np.arange(21).reshape(3, 7), names, names[::-1])
        self.assertTrue(q.flags.c_contiguous)

    def test_graph_only_is_explicit_and_requires_graph_enabled(self):
        b = backend(self)
        self.assertIn("enable_opt", b.PlannerOptions.__dataclass_fields__)
        with self.assertRaisesRegex(ValueError, "graph"):
            b.PlannerOptions(enable_opt=False, enable_graph=False)
        self.assertEqual(b.PlannerOptions(enable_opt=False, enable_graph=True).solver_mode,
                         "native_graph_only")
        self.assertEqual(b.PlannerOptions().solver_mode, "native_trajectory_optimization")


if __name__ == "__main__":
    unittest.main()
