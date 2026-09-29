"""Pinned legacy cuRobo v0.7.6 joint-space adapter, isolated from Isaac Sim.

CLI requests and results are files so upstream solver logs cannot corrupt JSON.
The sphere-model planner is never an exact geometry or contact-physics verdict.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, field
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from typing import Any

import numpy as np

EXPECTED_COMMIT = "2fbffc35225398cf9d5f382804faa9de2608753b"


def validate_joint_vector(value: Any) -> np.ndarray:
    q = np.asarray(value, dtype=np.float64)
    if q.shape != (7,):
        raise ValueError("Expected exactly seven joint positions (shape [7])")
    if not np.isfinite(q).all():
        raise ValueError("Joint positions must be finite")
    return q


def ordered_path(path: Any, actual_names: list[str], requested_names: list[str]) -> np.ndarray:
    q = np.asarray(path, dtype=np.float64)
    if len(actual_names) != len(set(actual_names)) or set(actual_names) != set(requested_names):
        raise ValueError("Planner joint names differ from the declared seven-joint contract")
    if q.ndim != 2 or q.shape[1] != 7 or len(requested_names) != 7:
        raise ValueError("Planner path must contain seven joint columns")
    return np.ascontiguousarray(q[:, [actual_names.index(name) for name in requested_names]])


def validate_output_path(path: np.ndarray, start: np.ndarray, end: np.ndarray, tolerance: float) -> dict:
    if path.ndim != 2 or path.shape[1] != 7 or len(path) < 2 or not np.isfinite(path).all():
        raise ValueError("Planner output must be a finite Nx7 path with N >= 2")
    residual = {"start_max_abs_rad": float(np.max(np.abs(path[0] - start))),
                "end_max_abs_rad": float(np.max(np.abs(path[-1] - end)))}
    if max(residual.values()) > tolerance:
        raise ValueError(f"Planner endpoint residual exceeds {tolerance}: {residual}")
    return residual


def base_metadata() -> dict:
    return {"backend": "curobo", "api_family": "legacy_v0.7.6",
            "curobo_expected_commit": EXPECTED_COMMIT,
            "planner_method": "MotionGen.plan_single_js",
            "collision_representation": "robot_spheres_vs_world_meshes_or_cuboids",
            "exact_geometry_audit": "required_separately",
            "contact_physics": "not_executed", "fallback_used": False}


@dataclass
class BackendResult:
    success: bool
    path: np.ndarray = field(default_factory=lambda: np.empty((0, 7)))
    timestamps: np.ndarray = field(default_factory=lambda: np.empty(0))
    metadata: dict = field(default_factory=dict)
    failure: dict | None = None

    @classmethod
    def failed(cls, code: str, message: str, metadata: dict | None = None) -> "BackendResult":
        return cls(False, metadata=metadata or base_metadata(), failure={"code": code, "message": message})

    def to_dict(self) -> dict:
        return {"success": self.success, "path": self.path.tolist(),
                "timestamps": self.timestamps.tolist(), "metadata": self.metadata,
                "failure": self.failure}


@dataclass(frozen=True)
class PlannerOptions:
    interpolation_dt: float = 0.01
    endpoint_tolerance_rad: float = 0.002
    max_attempts: int = 8
    timeout_s: float = 20.0
    num_trajopt_seeds: int = 4
    num_graph_seeds: int = 4
    trajopt_tsteps: int = 32
    collision_activation_distance: float = 0.005
    use_cuda_graph: bool = True
    graph_seed: int = 1531
    ik_seed: int = 1531
    enable_graph: bool = False
    enable_opt: bool = True
    enable_graph_attempt: int = 2

    def __post_init__(self):
        if not (0 < self.interpolation_dt <= 0.05):
            raise ValueError("interpolation_dt must be in (0, 0.05]")
        if not (0 < self.endpoint_tolerance_rad <= 0.01):
            raise ValueError("endpoint_tolerance_rad must be in (0, 0.01]")
        if self.max_attempts < 1 or self.timeout_s <= 0:
            raise ValueError("Planning requires positive attempts and timeout")
        if self.collision_activation_distance < 0:
            raise ValueError("collision_activation_distance cannot be negative")
        if not self.enable_opt and not self.enable_graph:
            raise ValueError("Disabling trajectory optimization requires explicit graph planning")

    @property
    def solver_mode(self) -> str:
        return "native_trajectory_optimization" if self.enable_opt else "native_graph_only"


def _config(value: dict | str | Path) -> dict:
    if isinstance(value, (str, Path)):
        path = Path(value)
        if path.suffix.lower() == ".json":
            return json.loads(path.read_text())
        import yaml
        return yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a mapping or a JSON/YAML path")
    return copy.deepcopy(value)


def _sha(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _asset_hashes(robot: dict, world: dict) -> dict:
    files = {}
    kin = robot["kinematics"]
    for key in ("urdf_path", "collision_spheres"):
        value = kin.get(key)
        if isinstance(value, str) and Path(value).is_file():
            files[value] = hashlib.sha256(Path(value).read_bytes()).hexdigest()
    for mesh in world.get("mesh", {}).values():
        value = mesh.get("file_path")
        if value and Path(value).is_file():
            files[value] = hashlib.sha256(Path(value).read_bytes()).hexdigest()
    return files


def configuration_metadata(robot: dict, world: dict) -> dict:
    return {"robot_config_sha256": _sha(robot), "world_config_sha256": _sha(world),
            "asset_sha256": _asset_hashes(robot, world)}


def _runtime_metadata() -> dict:
    import curobo
    import torch
    root = next((p for p in Path(curobo.__file__).resolve().parents if (p / ".git").exists()), None)
    commit = None
    dirty = None
    if root:
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(root), "diff", "--name-only"], text=True).strip())
    if commit != EXPECTED_COMMIT or dirty:
        raise RuntimeError(f"cuRobo source must be clean pinned commit {EXPECTED_COMMIT}; got {commit}, dirty={dirty}")
    if not torch.cuda.is_available():
        raise RuntimeError("cuRobo backend requires the authorized Linux CUDA host")
    return {"curobo_commit": commit, "curobo_source_dirty": dirty,
            "curobo_package_version": importlib.metadata.version("nvidia_curobo"),
            "python_version": platform.python_version(), "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "cuda_capability": list(torch.cuda.get_device_capability(0)),
            "warp_version": importlib.metadata.version("warp-lang"),
            "backend_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _prepare_robot(value: dict | str | Path) -> tuple[dict, list[str]]:
    config = _config(value)
    robot = config.get("robot_cfg", config)
    kin = robot["kinematics"]
    locked = kin.get("lock_joints") or {}
    names = [x for x in kin["cspace"]["joint_names"] if x not in locked]
    if len(names) != 7 or len(set(names)) != 7:
        raise ValueError("Robot config must declare exactly seven unique active joint names")
    spheres = kin.get("collision_spheres", {})
    if isinstance(spheres, str) and Path(spheres).is_file():
        spheres = _config(spheres)
    if isinstance(spheres, dict):
        spheres = spheres.get("collision_spheres", spheres)
        count = sum(len(v) for v in spheres.values()) + sum((kin.get("extra_collision_spheres") or {}).values())
        if count >= 1024:
            raise ValueError(f"Legacy cuRobo self-collision kernel requires fewer than 1024 spheres; got {count}")
    return robot, names


class CuroboPlanner:
    """Reusable CUDA planner; the file CLI deliberately creates one per request."""

    def __init__(self, robot_config: dict | str | Path, world: dict | str | Path,
                 options: PlannerOptions | dict | None = None):
        self.options = options if isinstance(options, PlannerOptions) else PlannerOptions(**(options or {}))
        self.robot, self.joint_names = _prepare_robot(robot_config)
        self.world = _config(world)
        self.metadata = base_metadata() | configuration_metadata(self.robot, self.world) | {
            "joint_names": self.joint_names, "options": asdict(self.options),
            "solver_mode": self.options.solver_mode,
            "time_parameterization": "native_optimizer" if self.options.enable_opt else "geometric_interpolation_requires_external_retiming"}
        self.metadata.update(_runtime_metadata())
        import torch
        from curobo.types.base import TensorDeviceType
        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
        from curobo.geom.sdf.world import CollisionCheckerType
        self.tensor_args = TensorDeviceType(device=torch.device("cuda:0"))
        torch.manual_seed(self.options.graph_seed)
        np.random.seed(self.options.graph_seed)
        o = self.options
        started = time.perf_counter()
        cfg = MotionGenConfig.load_from_robot_config(
            {"robot_cfg": self.robot}, self.world, tensor_args=self.tensor_args,
            interpolation_dt=o.interpolation_dt, cspace_threshold=o.endpoint_tolerance_rad,
            num_trajopt_seeds=o.num_trajopt_seeds, num_graph_seeds=o.num_graph_seeds,
            trajopt_tsteps=o.trajopt_tsteps, use_cuda_graph=o.use_cuda_graph,
            self_collision_check=True, self_collision_opt=True,
            collision_checker_type=CollisionCheckerType.MESH,
            collision_activation_distance=o.collision_activation_distance,
            evaluate_interpolated_trajectory=True, graph_seed=o.graph_seed, ik_seed=o.ik_seed,
            trajopt_js_fix_terminal_action=True, trajopt_fix_terminal_action=True,
        )
        self.motion_gen = MotionGen(cfg)
        self.metadata["planner_initialization_s"] = time.perf_counter() - started

    def plan(self, start7: Any, end7: Any) -> BackendResult:
        start, end = validate_joint_vector(start7), validate_joint_vector(end7)
        from curobo.types.state import JointState
        from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
        import torch
        meta = copy.deepcopy(self.metadata)
        started = time.perf_counter()
        o = self.options
        try:
            start_state = JointState.from_position(self.tensor_args.to_device(start[None]), joint_names=self.joint_names)
            end_state = JointState.from_position(self.tensor_args.to_device(end[None]), joint_names=self.joint_names)
            result = self.motion_gen.plan_single_js(start_state, end_state, MotionGenPlanConfig(
                max_attempts=o.max_attempts, timeout=o.timeout_s,
                enable_graph=o.enable_graph, enable_graph_attempt=o.enable_graph_attempt,
                enable_opt=o.enable_opt, enable_finetune_trajopt=o.enable_opt,
                fail_on_invalid_query=True, check_start_validity=True,
            ))
            torch.cuda.synchronize()
            meta.update({"planner_elapsed_s": time.perf_counter() - started,
                         "planner_status": str(result.status), "valid_query": bool(result.valid_query),
                         "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                         "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
            for name in ("solve_time", "graph_time", "trajopt_time", "finetune_time", "attempts"):
                value = getattr(result, name, None)
                if value is not None:
                    meta[name] = float(value)
            if not bool(result.success.item()):
                return BackendResult.failed("PLANNER_FAILURE", str(result.status), meta)
            trajectory = result.get_interpolated_plan()
            path = ordered_path(trajectory.position.detach().cpu().numpy(), trajectory.joint_names, self.joint_names)
            meta["endpoint_residual"] = validate_output_path(path, start, end, o.endpoint_tolerance_rad)
            dt = float(result.interpolation_dt)
            meta["interpolation_dt"] = dt
            meta["path_samples"] = len(path)
            meta["joint_travel_rad"] = np.abs(np.diff(path, axis=0)).sum(axis=0).tolist()
            return BackendResult(True, path, np.arange(len(path)) * dt, meta)
        except Exception as exc:
            meta["planner_elapsed_s"] = time.perf_counter() - started
            return BackendResult.failed(type(exc).__name__, str(exc), meta)


def plan_joint_path(robot_config: dict | str | Path, world: dict | str | Path,
                    start7: Any, end7: Any, *, options: PlannerOptions | dict | None = None) -> BackendResult:
    # Validate before importing CUDA, so a malformed runtime request is cheap and explicit.
    start, end = validate_joint_vector(start7), validate_joint_vector(end7)
    try:
        return CuroboPlanner(robot_config, world, options).plan(start, end)
    except Exception as exc:
        return BackendResult.failed(type(exc).__name__, str(exc))


def forward_kinematics(robot_config: dict | str | Path, joint_vectors: Any, link_names: list[str]) -> dict:
    robot, names = _prepare_robot(robot_config)
    vectors = np.asarray([validate_joint_vector(q) for q in joint_vectors])
    if not len(vectors) or not link_names:
        raise ValueError("FK requires joint_vectors and link_names")
    meta = base_metadata() | configuration_metadata(robot, {}) | _runtime_metadata() | {
        "joint_names": names, "pose_frame": "robot_base", "quaternion_order": "wxyz"}
    robot["kinematics"]["link_names"] = list(link_names)
    from curobo.types.base import TensorDeviceType
    from curobo.types.robot import RobotConfig
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
    tensor_args = TensorDeviceType()
    rc = RobotConfig.from_dict(robot, tensor_args)
    model = CudaRobotModel(rc.kinematics)
    actual = list(model.joint_names)
    q = ordered_path(vectors, names, actual)
    state = model.get_state(tensor_args.to_device(q).contiguous())
    poses = {name: np.concatenate([state.link_poses[name].position.detach().cpu().numpy(),
                                  state.link_poses[name].quaternion.detach().cpu().numpy()], axis=-1).tolist()
             for name in link_names}
    return {"success": True, "path": [], "timestamps": [], "failure": None,
            "metadata": meta, "link_poses": poses}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; use a new evidence path")
    request_bytes = args.request.read_bytes()
    try:
        request = json.loads(request_bytes)
        operation = request.get("operation", "plan")
        if operation == "plan":
            result = plan_joint_path(request["robot_config"], request["world"], request["start7"],
                                     request["end7"], options=request.get("options")).to_dict()
        elif operation == "fk":
            result = forward_kinematics(request["robot_config"], request["joint_vectors"], request["link_names"])
        else:
            raise ValueError(f"Unknown operation {operation}")
    except Exception as exc:
        result = BackendResult.failed(type(exc).__name__, str(exc)).to_dict()
    result["metadata"]["request_sha256"] = hashlib.sha256(request_bytes).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(temp, args.output)
    return 0 if result["success"] else 2


if __name__ == "__main__":
    sys.exit(main())
