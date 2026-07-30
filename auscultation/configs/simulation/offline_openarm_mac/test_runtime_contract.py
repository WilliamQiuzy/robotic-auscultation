from __future__ import annotations

import base64
import hashlib
import importlib
import json
import os
import re
import stat
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


EXPECTED_DIRECT_DISTRIBUTIONS = {
    "numpy": "2.4.2",
    "mujoco": "3.10.0",
    "usd-core": "24.8",
    "trimesh": "4.12.2",
    "pycollada": "0.9.3",
    "rtree": "1.4.1",
    "scipy": "1.17.1",
    "opencv-python-headless": "5.0.0.93",
    "imageio": "2.37.3",
    "imageio-ffmpeg": "0.6.0",
    "pytest": "9.1.1",
}


def _contract():
    return importlib.import_module("runtime_contract")


def _fingerprint(path: Path) -> tuple[str, int, int] | None:
    if not path.exists():
        return None
    file_stat = path.stat()
    return (
        hashlib.sha256(path.read_bytes()).hexdigest(),
        file_stat.st_size,
        file_stat.st_mtime_ns,
    )


def _snapshot_mujoco_logs(root: Path) -> dict[Path, tuple[str, int, int]]:
    return {
        path.resolve(strict=True): fingerprint
        for path in root.rglob("MUJOCO_LOG.TXT")
        if (fingerprint := _fingerprint(path)) is not None
    }


def _record_sha256(payload: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
    return "sha256=" + digest.rstrip(b"=").decode("ascii")


class _FakeDistribution:
    def __init__(self, root: Path, files: list[Path], version: str = "1.0"):
        self._root = root
        self.files = files
        self.version = version

    def locate_file(self, path: object) -> Path:
        return self._root / str(path)


def test_project_and_lock_capture_exact_direct_dependencies_with_hashes():
    contract = _contract()

    project = contract.validate_project_contract()

    assert project["requires_python"] == "==3.11.*"
    assert project["direct_dependencies"] == EXPECTED_DIRECT_DISTRIBUTIONS
    assert project["lock_version"] == 1
    assert project["lock_revision"] == 3
    assert len(project["config_sha256"]) == 64
    assert len(project["lock_sha256"]) == 64
    assert project["locked_package_count"] >= len(EXPECTED_DIRECT_DISTRIBUTIONS)


@pytest.mark.parametrize(
    ("field", "current", "invalid"),
    [
        ("version", "version = 1\n", "version = 2\n"),
        ("revision", "revision = 3\n", "revision = 4\n"),
    ],
)
def test_project_contract_rejects_lock_schema_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    current: str,
    invalid: str,
):
    contract = _contract()
    invalid_lock = tmp_path / "uv.lock"
    invalid_lock.write_text(
        contract.LOCK_PATH.read_text(encoding="utf-8").replace(current, invalid, 1),
        encoding="utf-8",
    )
    monkeypatch.setattr(contract, "LOCK_PATH", invalid_lock)

    with pytest.raises(contract.RuntimeContractError, match=rf"uv\.lock {field}"):
        contract.validate_project_contract()


def test_runtime_identity_rejects_x86_rosetta_wrong_python_and_mjx():
    contract = _contract()
    env_root = contract.EXPECTED_ENV_ROOT
    valid = contract.RuntimeIdentity(
        sys_platform="darwin",
        platform_system="Darwin",
        platform_machine="arm64",
        uname_machine="arm64",
        arch_command="arm64",
        sysconfig_platform="macosx-15.0-arm64",
        python_version=(3, 11, 15),
        python_version_string="3.11.15",
        executable=str(env_root / "bin/python"),
        prefix=str(env_root),
        rosetta_translated=False,
    )

    with pytest.raises(contract.RuntimeContractError, match="arm64"):
        contract.validate_runtime_identity(
            replace(
                valid,
                platform_machine="x86_64",
                uname_machine="x86_64",
                arch_command="i386",
                sysconfig_platform="macosx-15.0-x86_64",
            )
        )
    with pytest.raises(contract.RuntimeContractError, match="Rosetta"):
        contract.validate_runtime_identity(replace(valid, rosetta_translated=True))
    with pytest.raises(contract.RuntimeContractError, match="Python 3.11"):
        contract.validate_runtime_identity(
            replace(
                valid,
                python_version=(3, 12, 0),
                python_version_string="3.12.0",
            )
        )

    mjx_root = Path.home() / "envs/mjx"
    with pytest.raises(contract.RuntimeContractError, match="mjx"):
        contract.validate_runtime_identity(
            replace(
                valid,
                executable=str(mjx_root / "bin/python"),
                prefix=str(mjx_root),
            )
        )


@pytest.mark.parametrize(
    ("sys_path", "pythonpath", "source"),
    [
        (["/safe/runtime/path"], "/safe/pythonpath", "cwd"),
        (["", "/safe/runtime/path"], "/safe/pythonpath", "sys.path"),
        (["/safe/runtime/path"], os.pathsep + "/safe/pythonpath", "PYTHONPATH"),
    ],
)
def test_runtime_path_guard_rejects_mjx_cwd_and_empty_path_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sys_path: list[str],
    pythonpath: str,
    source: str,
):
    contract = _contract()
    mjx_root = tmp_path / "mjx"
    mjx_root.mkdir()
    monkeypatch.setattr(contract, "FORBIDDEN_MJX_ROOT", mjx_root)
    monkeypatch.chdir(mjx_root)
    monkeypatch.setattr(sys, "path", sys_path)
    monkeypatch.setenv("PYTHONPATH", pythonpath)

    with pytest.raises(contract.RuntimeContractError, match=rf"{source}.*mjx"):
        contract._reject_mjx_paths()


def test_distribution_provenance_rejects_external_symlink_and_bad_record_hash(
    tmp_path: Path,
):
    contract = _contract()
    env_root = tmp_path / "env"
    site_packages = env_root / "lib/python3.11/site-packages"
    dist_info = site_packages / "demo-1.0.dist-info"
    dist_info.mkdir(parents=True)
    metadata_payload = b"Name: demo\nVersion: 1.0\n"
    (dist_info / "METADATA").write_bytes(metadata_payload)
    outside = tmp_path / "outside.py"
    outside_payload = b"outside = True\n"
    outside.write_bytes(outside_payload)
    (site_packages / "demo.py").symlink_to(outside)
    record = "\n".join(
        [
            f"demo.py,{_record_sha256(outside_payload)},{len(outside_payload)}",
            (
                "demo-1.0.dist-info/METADATA,"
                f"{_record_sha256(metadata_payload)},{len(metadata_payload)}"
            ),
            "demo-1.0.dist-info/RECORD,,",
        ]
    )
    (dist_info / "RECORD").write_text(record + "\n", encoding="utf-8")
    distribution = _FakeDistribution(
        site_packages,
        [
            Path("demo.py"),
            Path("demo-1.0.dist-info/METADATA"),
            Path("demo-1.0.dist-info/RECORD"),
        ],
    )

    with pytest.raises(contract.RuntimeContractError, match="outside isolated env"):
        contract.validate_distribution_provenance(
            "demo",
            distribution,
            env_root=env_root,
            compatible_locked_artifacts=[
                {"filename": "demo-1.0-py3-none-any.whl", "sha256": "0" * 64}
            ],
        )

    (site_packages / "demo.py").unlink()
    (site_packages / "demo.py").write_bytes(outside_payload)
    bad_record = record.replace(_record_sha256(outside_payload), "sha256=" + "A" * 43)
    (dist_info / "RECORD").write_text(bad_record + "\n", encoding="utf-8")
    with pytest.raises(contract.RuntimeContractError, match="RECORD sha256 mismatch"):
        contract.validate_distribution_provenance(
            "demo",
            distribution,
            env_root=env_root,
            compatible_locked_artifacts=[
                {"filename": "demo-1.0-py3-none-any.whl", "sha256": "0" * 64}
            ],
        )


def test_distribution_provenance_rejects_unapproved_unhashed_record_entry(
    tmp_path: Path,
):
    contract = _contract()
    env_root = tmp_path / "env"
    site_packages = env_root / "lib/python3.11/site-packages"
    dist_info = site_packages / "demo-1.0.dist-info"
    dist_info.mkdir(parents=True)
    (site_packages / "demo.py").write_bytes(b"demo = True\n")
    metadata_payload = b"Name: demo\nVersion: 1.0\n"
    (dist_info / "METADATA").write_bytes(metadata_payload)
    (dist_info / "RECORD").write_text(
        "demo.py,,\n"
        "demo-1.0.dist-info/METADATA,"
        f"{_record_sha256(metadata_payload)},{len(metadata_payload)}\n"
        "demo-1.0.dist-info/RECORD,,\n",
        encoding="utf-8",
    )
    distribution = _FakeDistribution(
        site_packages,
        [
            Path("demo.py"),
            Path("demo-1.0.dist-info/METADATA"),
            Path("demo-1.0.dist-info/RECORD"),
        ],
    )

    with pytest.raises(contract.RuntimeContractError, match="unhashed RECORD entry"):
        contract.validate_distribution_provenance(
            "demo",
            distribution,
            env_root=env_root,
            compatible_locked_artifacts=[
                {"filename": "demo-1.0-py3-none-any.whl", "sha256": "0" * 64}
            ],
        )


def test_live_runtime_is_native_arm64_python311_inside_isolated_env():
    contract = _contract()

    manifest = contract.build_runtime_manifest()
    runtime = manifest["runtime"]

    assert runtime["sys_platform"] == "darwin"
    assert runtime["platform_system"] == "Darwin"
    assert runtime["platform_machine"] == "arm64"
    assert runtime["uname_machine"] == "arm64"
    assert runtime["arch_command"] == "arm64"
    assert runtime["rosetta_translated"] is False
    assert runtime["python_version"][:2] == [3, 11]
    assert Path(runtime["executable_realpath"]).is_relative_to(
        contract.EXPECTED_ENV_ROOT.resolve()
    )
    assert Path(runtime["prefix_realpath"]) == contract.EXPECTED_ENV_ROOT.resolve()
    assert {
        name: record["version"]
        for name, record in manifest["direct_distributions"].items()
    } == EXPECTED_DIRECT_DISTRIBUTIONS
    assert all(
        Path(record["location_realpath"]).is_relative_to(
            contract.EXPECTED_ENV_ROOT.resolve()
        )
        for record in manifest["direct_distributions"].values()
    )
    from packaging.tags import sys_tags
    from packaging.utils import parse_wheel_filename

    compatible_tags = set(sys_tags())
    for record in manifest["direct_distributions"].values():
        assert record["record_file_count"] > 0
        assert re.fullmatch(r"[0-9a-f]{64}", record["record_tree_sha256"])
        assert Path(record["metadata_dir_realpath"]).is_relative_to(
            contract.EXPECTED_ENV_ROOT.resolve()
        )
        assert record["compatible_locked_artifacts"]
        for artifact in record["compatible_locked_artifacts"]:
            assert re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"])
            _, _, _, wheel_tags = parse_wheel_filename(artifact["filename"])
            assert wheel_tags & compatible_tags


def test_canonical_manifest_is_sorted_compact_and_nan_free():
    contract = _contract()

    assert contract.canonical_json_bytes({"z": 2, "a": [3, 1]}) == (
        b'{"a":[3,1],"z":2}'
    )
    with pytest.raises(ValueError, match="JSON"):
        contract.canonical_json_bytes({"bad": float("nan")})


def test_atomic_manifest_writer_uses_explicit_test_path_and_fsyncs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    contract = _contract()
    formal_before = _fingerprint(contract.MANIFEST_PATH)
    output_path = tmp_path / "nested/runtime_manifest.json"
    fsync_kinds: list[bool] = []
    real_fsync = os.fsync

    def tracked_fsync(file_descriptor: int) -> None:
        fsync_kinds.append(stat.S_ISDIR(os.fstat(file_descriptor).st_mode))
        real_fsync(file_descriptor)

    monkeypatch.setattr(contract.os, "fsync", tracked_fsync)
    manifest = contract.write_runtime_manifest(output_path)
    payload = output_path.read_bytes()
    assert payload == contract.canonical_json_bytes(manifest)
    assert b"NaN" not in payload
    assert json.loads(payload) == manifest
    assert fsync_kinds == [False, True]
    assert _fingerprint(contract.MANIFEST_PATH) == formal_before


def test_atomic_manifest_writer_cleans_temp_after_replace_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    contract = _contract()
    output_path = tmp_path / "runtime_manifest.json"
    observed_temp: list[Path] = []
    monkeypatch.setattr(
        contract,
        "build_runtime_manifest",
        lambda: {"schema": "wam.offline_openarm_mac.runtime.v1"},
    )

    def fail_replace(source: str | Path, destination: str | Path) -> None:
        observed_temp.append(Path(source))
        raise OSError("replace failed")

    monkeypatch.setattr(contract.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        contract.write_runtime_manifest(output_path)

    assert observed_temp
    assert not observed_temp[0].exists()
    assert not output_path.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_formal_manifest_is_current_canonical_and_read_only():
    contract = _contract()
    formal_before = _fingerprint(contract.MANIFEST_PATH)

    payload = contract.MANIFEST_PATH.read_bytes()
    manifest = json.loads(payload)

    assert manifest["schema"] == "wam.offline_openarm_mac.runtime.v1"
    assert payload == contract.canonical_json_bytes(manifest)
    assert manifest == contract.build_runtime_manifest()
    assert _fingerprint(contract.MANIFEST_PATH) == formal_before


def test_mujoco_renders_rgb_metric_depth_then_two_geom_segmentation_at_64px(
    monkeypatch: pytest.MonkeyPatch,
):
    contract = _contract()
    workspace_logs_before = _snapshot_mujoco_logs(contract.WORKSPACE_ROOT)
    isolated_log_dir = contract.EXPECTED_ENV_ROOT / "logs"
    isolated_log_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(isolated_log_dir)

    import mujoco

    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco model="two_visible_geoms">
          <visual>
            <map znear="0.01" zfar="10"/>
          </visual>
          <worldbody>
            <geom name="left_geom" type="box" pos="-0.45 0 0"
                  size="0.30 0.30 0.30" rgba="1 0 0 1"/>
            <geom name="right_geom" type="box" pos="0.45 0 0"
                  size="0.30 0.30 0.30" rgba="0 1 0 1"/>
            <camera name="render_camera" pos="0 -3 0"
                    xyaxes="1 0 0 0 0 1" fovy="45"/>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=64, width=64)

    try:
        renderer.update_scene(data, camera="render_camera")
        rgb = renderer.render()

        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera="render_camera")
        depth_m = renderer.render()
        renderer.disable_depth_rendering()

        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera="render_camera")
        segmentation = renderer.render()
        renderer.disable_segmentation_rendering()
    finally:
        renderer.close()

    assert rgb.shape == (64, 64, 3)
    assert rgb.dtype == np.uint8
    assert np.unique(rgb.reshape(-1, 3), axis=0).shape[0] > 2
    assert depth_m.shape == (64, 64)
    assert np.issubdtype(depth_m.dtype, np.floating)
    assert np.isfinite(depth_m).all()

    assert segmentation.shape == (64, 64, 2)
    left_id = model.geom("left_geom").id
    right_id = model.geom("right_geom").id
    assert left_id != right_id
    geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)
    left_mask = (segmentation[..., 0] == left_id) & (
        segmentation[..., 1] == geom_type
    )
    right_mask = (segmentation[..., 0] == right_id) & (
        segmentation[..., 1] == geom_type
    )
    assert left_mask.any()
    assert right_mask.any()

    visible_depth_m = depth_m[left_mask | right_mask]
    assert np.median(visible_depth_m) == pytest.approx(2.7, abs=0.15)

    workspace_logs_after = _snapshot_mujoco_logs(contract.WORKSPACE_ROOT)
    changed_logs = {
        path
        for path in workspace_logs_before.keys() | workspace_logs_after.keys()
        if workspace_logs_before.get(path) != workspace_logs_after.get(path)
    }
    isolated_log_root = isolated_log_dir.resolve(strict=True)
    isolated_mujoco_log = (isolated_log_dir / "MUJOCO_LOG.TXT").resolve(strict=True)
    assert changed_logs
    assert all(path.is_relative_to(isolated_log_root) for path in changed_logs)
    assert isolated_mujoco_log in workspace_logs_after
    assert isolated_mujoco_log in changed_logs
