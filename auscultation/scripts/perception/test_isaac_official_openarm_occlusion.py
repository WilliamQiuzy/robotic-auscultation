from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ausc.perception.tracking import official_openarm_experiment as experiment
from scripts.perception import isaac_official_openarm_occlusion as runner


def _required_cli_arguments(tmp_path: Path) -> list[str]:
    return [
        "--asset-bundle",
        str(tmp_path / "bundle"),
        "--asset-manifest",
        str(tmp_path / "bundle" / "manifest.json"),
        "--out",
        str(tmp_path / "out"),
    ]


def _write_fake_media_tool(
    path: Path,
    tool_name: str,
    *,
    version: str = "7.1-test",
) -> str:
    path.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' '{tool_name} version {version}' 'configuration: test-only'\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _media_identity(path: Path) -> dict[str, int]:
    info = path.stat()
    return {
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "size_bytes": int(info.st_size),
        "mtime_ns": int(info.st_mtime_ns),
        "ctime_ns": int(info.st_ctime_ns),
    }


def _single_frame_probe_stdout() -> str:
    return json.dumps(
        {
            "streams": [
                {
                    "codec_name": "h264",
                    "pix_fmt": "yuv420p",
                    "width": 640,
                    "height": 360,
                    "nb_frames": "1",
                    "r_frame_rate": "12/1",
                }
            ]
        }
    )


def _verified_fake_media_tools(
    monkeypatch: pytest.MonkeyPatch,
    ffmpeg: Path,
    ffprobe: Path,
) -> dict:
    identities = {
        (ffmpeg.stat().st_dev, ffmpeg.stat().st_ino): "ffmpeg",
        (ffprobe.stat().st_dev, ffprobe.stat().st_ino): "ffprobe",
    }

    def pinned_version_run(command, **kwargs):
        pass_fds = kwargs.get("pass_fds")
        assert isinstance(pass_fds, tuple) and len(pass_fds) == 1
        descriptor = pass_fds[0]
        info = os.fstat(descriptor)
        name = identities[(info.st_dev, info.st_ino)]
        fd_root = (
            "/proc/self/fd"
            if runner.sys.platform.startswith("linux")
            else "/dev/fd"
        )
        assert command == [f"{fd_root}/{descriptor}", "-version"]
        return SimpleNamespace(
            stdout=f"{name} version 7.1-test\nconfiguration: test-only\n",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(runner.subprocess, "run", pinned_version_run)
    return runner._verified_media_tools(
        ffmpeg_path=str(ffmpeg),
        ffmpeg_sha256=hashlib.sha256(ffmpeg.read_bytes()).hexdigest(),
        ffprobe_path=str(ffprobe),
        ffprobe_sha256=hashlib.sha256(ffprobe.read_bytes()).hexdigest(),
        _allow_test_fd_fallback=True,
    )


def test_isolated_code_release_has_only_the_official_experiment_runtime_closure() -> None:
    assert runner.ISOLATED_CODE_RELEASE_PATHS == (
        "auscultation/ausc/__init__.py",
        "auscultation/ausc/perception/tracking/experimental_visibility_geometry.py",
        "auscultation/ausc/perception/tracking/official_asset_contract.py",
        "auscultation/ausc/perception/tracking/official_openarm_experiment.py",
        "auscultation/scripts/perception/isaac_official_openarm_occlusion.py",
    )
    assert "auscultation/ausc/perception/tracking/__init__.py" not in (
        runner.ISOLATED_CODE_RELEASE_PATHS
    )


def test_runtime_provenance_requires_exact_isaac_51_and_python_311() -> None:
    versions = {
        "isaacsim": "5.1.0.0",
        "isaacsim-kernel": "5.1.0.0",
    }
    evidence = runner._verified_runtime_provenance(
        version_lookup=versions.__getitem__,
        executable=runner.sys.executable,
        python_version=(3, 11, 9),
    )

    assert evidence["python_version"] == "3.11.9"
    assert evidence["isaacsim_version"] == "5.1.0.0"
    assert evidence["isaacsim_kernel_version"] == "5.1.0.0"
    assert Path(evidence["python_executable_realpath"]).is_file()

    for distribution, value in (
        ("isaacsim", "5.0.0.0"),
        ("isaacsim-kernel", "5.1.0.1"),
    ):
        wrong = dict(versions)
        wrong[distribution] = value
        with pytest.raises(RuntimeError, match="Isaac 5.1 runtime"):
            runner._verified_runtime_provenance(
                version_lookup=wrong.__getitem__,
                executable=runner.sys.executable,
                python_version=(3, 11, 9),
            )

    with pytest.raises(RuntimeError, match="Python 3.11"):
        runner._verified_runtime_provenance(
            version_lookup=versions.__getitem__,
            executable=runner.sys.executable,
            python_version=(3, 10, 14),
        )


def test_runtime_provenance_is_attached_after_exact_stage_validation() -> None:
    stage = {
        "meters_per_unit": 1.0,
        "import_root_scale": 1.0,
        "link_count": 24,
        "joint_count": 23,
        "revolute_joint_names": list(
            experiment.LEFT_FOLLOWER_JOINTS + experiment.RIGHT_REFERENCE_JOINTS
        ),
        "animated_joint_names": list(experiment.LEFT_FOLLOWER_JOINTS),
        "robot_pixel_count": 1200,
        "human_pixel_count": 8000,
        "human_asset_sha256": experiment.NVIDIA_HUMAN_SHA256,
        "robot_urdf_sha256": experiment.OFFICIAL_CLEAN_URDF_SHA256,
    }
    provenance = {
        "python_version": "3.11.9",
        "isaacsim_version": "5.1.0.0",
        "isaacsim_kernel_version": "5.1.0.0",
    }

    combined = runner._validated_stage_inventory(stage, provenance)

    assert set(stage) == {
        "meters_per_unit",
        "import_root_scale",
        "link_count",
        "joint_count",
        "revolute_joint_names",
        "animated_joint_names",
        "robot_pixel_count",
        "human_pixel_count",
        "human_asset_sha256",
        "robot_urdf_sha256",
    }
    assert combined["runtime_provenance"] == provenance
    assert "runtime_provenance" not in stage


def test_smoke_arguments_do_not_require_media_tools(tmp_path: Path) -> None:
    args = runner._arguments([*_required_cli_arguments(tmp_path), "--mode", "smoke"])

    assert args.mode == "smoke"
    assert args.ffmpeg_path is None
    assert args.ffmpeg_sha256 is None
    assert args.ffprobe_path is None
    assert args.ffprobe_sha256 is None


@pytest.mark.parametrize(
    "missing_option",
    (
        "--ffmpeg-path",
        "--ffmpeg-sha256",
        "--ffprobe-path",
        "--ffprobe-sha256",
    ),
)
def test_full_arguments_require_all_explicit_media_tool_bindings(
    tmp_path: Path, missing_option: str
) -> None:
    media_options = {
        "--ffmpeg-path": str(tmp_path / "ffmpeg"),
        "--ffmpeg-sha256": "a" * 64,
        "--ffprobe-path": str(tmp_path / "ffprobe"),
        "--ffprobe-sha256": "b" * 64,
    }
    argv = [*_required_cli_arguments(tmp_path), "--mode", "full"]
    for option, value in media_options.items():
        if option != missing_option:
            argv.extend((option, value))

    with pytest.raises(SystemExit):
        runner._arguments(argv)


def test_media_tools_are_verified_and_attached_to_runtime_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ffmpeg = tmp_path / "ffmpeg"
    ffprobe = tmp_path / "ffprobe"
    ffmpeg_sha = _write_fake_media_tool(ffmpeg, "ffmpeg")
    ffprobe_sha = _write_fake_media_tool(ffprobe, "ffprobe")

    version_fds: list[int] = []

    def pinned_version_run(command, **kwargs):
        pass_fds = kwargs.get("pass_fds")
        assert isinstance(pass_fds, tuple) and len(pass_fds) == 1
        descriptor = pass_fds[0]
        assert command[0] in (
            f"/proc/self/fd/{descriptor}",
            f"/dev/fd/{descriptor}",
        )
        assert command[1:] == ["-version"]
        info = os.fstat(descriptor)
        name = "ffmpeg" if info.st_ino == ffmpeg.stat().st_ino else "ffprobe"
        version_fds.append(descriptor)
        return SimpleNamespace(
            stdout=f"{name} version 7.1-test\nconfiguration: test-only\n",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(runner.subprocess, "run", pinned_version_run)

    media_tools = runner._verified_media_tools(
        ffmpeg_path=str(ffmpeg),
        ffmpeg_sha256=ffmpeg_sha,
        ffprobe_path=str(ffprobe),
        ffprobe_sha256=ffprobe_sha,
        _allow_test_fd_fallback=True,
    )

    assert media_tools == {
        "ffmpeg": {
            "supplied_path": str(ffmpeg),
            "resolved_realpath": str(ffmpeg.resolve(strict=True)),
            "size_bytes": ffmpeg.stat().st_size,
            "sha256": ffmpeg_sha,
            "version_line": "ffmpeg version 7.1-test",
            "identity": _media_identity(ffmpeg),
        },
        "ffprobe": {
            "supplied_path": str(ffprobe),
            "resolved_realpath": str(ffprobe.resolve(strict=True)),
            "size_bytes": ffprobe.stat().st_size,
            "sha256": ffprobe_sha,
            "version_line": "ffprobe version 7.1-test",
            "identity": _media_identity(ffprobe),
        },
    }
    assert len(version_fds) == 2
    for descriptor in version_fds:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    base = {"python_version": "3.11.9", "isaacsim_version": "5.1.0.0"}
    runtime = runner._runtime_provenance_with_media_tools(base, media_tools)
    assert runtime["media_tools"] == media_tools
    assert "media_tools" not in base

    stage = {
        "meters_per_unit": 1.0,
        "import_root_scale": 1.0,
        "link_count": 24,
        "joint_count": 23,
        "revolute_joint_names": list(
            experiment.LEFT_FOLLOWER_JOINTS + experiment.RIGHT_REFERENCE_JOINTS
        ),
        "animated_joint_names": list(experiment.LEFT_FOLLOWER_JOINTS),
        "robot_pixel_count": 1200,
        "human_pixel_count": 8000,
        "human_asset_sha256": experiment.NVIDIA_HUMAN_SHA256,
        "robot_urdf_sha256": experiment.OFFICIAL_CLEAN_URDF_SHA256,
    }
    inventory = runner._validated_stage_inventory(stage, runtime)
    assert inventory["runtime_provenance"]["media_tools"] == media_tools


def test_media_tool_verification_rejects_bad_hash_path_and_symlink(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "ffmpeg"
    actual_sha = _write_fake_media_tool(executable, "ffmpeg")

    with pytest.raises(ValueError, match="canonical lowercase"):
        runner._verified_media_tool(
            str(executable), actual_sha.upper(), tool_name="ffmpeg"
        )
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        runner._verified_media_tool(
            str(executable), "0" * 64, tool_name="ffmpeg"
        )
    with pytest.raises(RuntimeError, match="unavailable"):
        runner._verified_media_tool(
            str(tmp_path / "missing"), actual_sha, tool_name="ffmpeg"
        )

    non_executable = tmp_path / "non-executable"
    non_executable.write_bytes(executable.read_bytes())
    non_executable.chmod(0o644)
    with pytest.raises(RuntimeError, match="executable regular file"):
        runner._verified_media_tool(
            str(non_executable), actual_sha, tool_name="ffmpeg"
        )

    symlink = tmp_path / "ffmpeg-link"
    symlink.symlink_to(executable)
    with pytest.raises(RuntimeError, match="symlink"):
        runner._verified_media_tool(str(symlink), actual_sha, tool_name="ffmpeg")


def test_fd_execution_path_is_linux_only_unless_test_fallback_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "ffmpeg"
    _write_fake_media_tool(executable, "ffmpeg")
    descriptor = os.open(executable, os.O_RDONLY)
    try:
        identity = _media_identity(executable)
        monkeypatch.setattr(runner.sys, "platform", "darwin")
        with pytest.raises(RuntimeError, match="Linux /proc"):
            runner._media_fd_execution_path(
                descriptor,
                expected_identity=identity,
                tool_name="ffmpeg",
            )
        assert runner._media_fd_execution_path(
            descriptor,
            expected_identity=identity,
            tool_name="ffmpeg",
            allow_test_fallback=True,
        ) == f"/dev/fd/{descriptor}"

        real_stat = os.stat
        monkeypatch.setattr(runner.sys, "platform", "linux")
        monkeypatch.setattr(
            runner.os,
            "stat",
            lambda value: (
                os.fstat(descriptor)
                if value == f"/proc/self/fd/{descriptor}"
                else real_stat(value)
            ),
        )
        assert runner._media_fd_execution_path(
            descriptor,
            expected_identity=identity,
            tool_name="ffmpeg",
        ) == f"/proc/self/fd/{descriptor}"
    finally:
        os.close(descriptor)


def test_encode_is_fd_pinned_during_swap_exec_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "frame_00000.png").write_bytes(b"frame")
    ffmpeg = tmp_path / "ffmpeg"
    ffprobe = tmp_path / "ffprobe"
    ffmpeg_sha = _write_fake_media_tool(ffmpeg, "ffmpeg")
    ffprobe_sha = _write_fake_media_tool(ffprobe, "ffprobe")
    ffmpeg_provenance = {
        "supplied_path": str(ffmpeg),
        "resolved_realpath": str(ffmpeg.resolve(strict=True)),
        "size_bytes": ffmpeg.stat().st_size,
        "sha256": ffmpeg_sha,
        "version_line": "ffmpeg version 7.1-test",
        "identity": _media_identity(ffmpeg),
    }
    ffprobe_provenance = {
        "supplied_path": str(ffprobe),
        "resolved_realpath": str(ffprobe.resolve(strict=True)),
        "size_bytes": ffprobe.stat().st_size,
        "sha256": ffprobe_sha,
        "version_line": "ffprobe version 7.1-test",
        "identity": _media_identity(ffprobe),
    }
    original_inode = ffmpeg.stat().st_ino
    held_original = tmp_path / "ffmpeg-original-held"
    malicious = tmp_path / "ffmpeg-malicious"
    displaced = tmp_path / "ffmpeg-malicious-displaced"
    _write_fake_media_tool(malicious, "ffmpeg", version="9.9-malicious")
    actual_fds: list[int] = []

    def swap_restore_run(command, **kwargs):
        invoked = list(command)
        if invoked[1:] == ["-version"]:
            return SimpleNamespace(
                stdout="ffmpeg version 7.1-test\n",
                stderr="",
                returncode=0,
            )
        pass_fds = kwargs.get("pass_fds")
        assert isinstance(pass_fds, tuple) and len(pass_fds) == 1
        descriptor = pass_fds[0]
        assert invoked[0] in (
            f"/proc/self/fd/{descriptor}",
            f"/dev/fd/{descriptor}",
        )
        assert os.fstat(descriptor).st_ino == original_inode
        actual_fds.append(descriptor)
        os.replace(ffmpeg, held_original)
        os.replace(malicious, ffmpeg)
        assert ffmpeg.stat().st_ino != os.fstat(descriptor).st_ino
        os.replace(ffmpeg, displaced)
        os.replace(held_original, ffmpeg)
        assert ffmpeg.stat().st_ino == original_inode
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", swap_restore_run)

    with pytest.raises(RuntimeError, match="changed since startup"):
        runner._encode_h264(
            frames,
            tmp_path / "evidence.mp4",
            fps=12,
            expected_width=640,
            expected_height=360,
            expected_frames=1,
            ffmpeg_provenance=ffmpeg_provenance,
            ffprobe_provenance=ffprobe_provenance,
            _allow_test_fd_fallback=True,
        )
    assert len(actual_fds) == 1
    with pytest.raises(OSError):
        os.fstat(actual_fds[0])


def test_encode_h264_uses_only_explicit_verified_command_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "frame_00000.png").write_bytes(b"frame")
    destination = tmp_path / "evidence.mp4"
    ffmpeg = tmp_path / "verified-ffmpeg"
    ffprobe = tmp_path / "verified-ffprobe"
    _write_fake_media_tool(ffmpeg, "ffmpeg")
    _write_fake_media_tool(ffprobe, "ffprobe")
    media_tools = _verified_fake_media_tools(monkeypatch, ffmpeg, ffprobe)
    identities = {
        ffmpeg.stat().st_ino: "ffmpeg",
        ffprobe.stat().st_ino: "ffprobe",
    }
    calls: list[tuple[str, list[str], int]] = []

    def fake_run(command, **kwargs):
        invoked = list(command)
        pass_fds = kwargs.get("pass_fds")
        assert isinstance(pass_fds, tuple) and len(pass_fds) == 1
        descriptor = pass_fds[0]
        name = identities[os.fstat(descriptor).st_ino]
        assert invoked[0] in (
            f"/proc/self/fd/{descriptor}",
            f"/dev/fd/{descriptor}",
        )
        calls.append((name, invoked, descriptor))
        if name == "ffprobe":
            return SimpleNamespace(
                stdout=_single_frame_probe_stdout(), stderr="", returncode=0
            )
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(
        runner.shutil,
        "which",
        lambda _name: pytest.fail("PATH lookup must not be used for media tools"),
    )

    runner._encode_h264(
        frames,
        destination,
        fps=12,
        expected_width=640,
        expected_height=360,
        expected_frames=1,
        ffmpeg_provenance=media_tools["ffmpeg"],
        ffprobe_provenance=media_tools["ffprobe"],
        _allow_test_fd_fallback=True,
    )

    assert [name for name, _, _ in calls] == ["ffmpeg", "ffprobe"]
    assert not any(command[1:] == ["-version"] for _, command, _ in calls)
    for _, _, descriptor in calls:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_encode_rejects_atomic_byte_replacement_before_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "frame_00000.png").write_bytes(b"frame")
    ffmpeg = tmp_path / "ffmpeg"
    ffprobe = tmp_path / "ffprobe"
    _write_fake_media_tool(ffmpeg, "ffmpeg")
    _write_fake_media_tool(ffprobe, "ffprobe")
    media_tools = _verified_fake_media_tools(monkeypatch, ffmpeg, ffprobe)
    replacement = tmp_path / "ffmpeg-replacement"
    _write_fake_media_tool(replacement, "ffmpeg", version="9.9-replaced")
    os.replace(replacement, ffmpeg)
    calls: list[list[str]] = []

    def unexpected_run(command, **_kwargs):
        calls.append(list(command))
        pytest.fail("replaced ffmpeg bytes must not be executed")

    monkeypatch.setattr(runner.subprocess, "run", unexpected_run)

    with pytest.raises(RuntimeError, match="changed since startup"):
        runner._encode_h264(
            frames,
            tmp_path / "evidence.mp4",
            fps=12,
            expected_width=640,
            expected_height=360,
            expected_frames=1,
            ffmpeg_provenance=media_tools["ffmpeg"],
            ffprobe_provenance=media_tools["ffprobe"],
            _allow_test_fd_fallback=True,
        )
    assert calls == []


def test_encode_rejects_same_byte_inode_replacement_before_actual_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "frame_00000.png").write_bytes(b"frame")
    ffmpeg = tmp_path / "ffmpeg"
    ffprobe = tmp_path / "ffprobe"
    _write_fake_media_tool(ffmpeg, "ffmpeg")
    _write_fake_media_tool(ffprobe, "ffprobe")
    media_tools = _verified_fake_media_tools(monkeypatch, ffmpeg, ffprobe)
    original_inode = media_tools["ffmpeg"]["identity"]["inode"]
    replacement = tmp_path / "ffmpeg-replacement"
    _write_fake_media_tool(replacement, "ffmpeg")
    os.replace(replacement, ffmpeg)
    assert ffmpeg.stat().st_ino != original_inode
    calls: list[list[str]] = []

    def unexpected_run(command, **_kwargs):
        calls.append(list(command))
        pytest.fail("replacement inode must be rejected before any execution")

    monkeypatch.setattr(runner.subprocess, "run", unexpected_run)

    with pytest.raises(RuntimeError, match="changed since startup"):
        runner._encode_h264(
            frames,
            tmp_path / "evidence.mp4",
            fps=12,
            expected_width=640,
            expected_height=360,
            expected_frames=1,
            ffmpeg_provenance=media_tools["ffmpeg"],
            ffprobe_provenance=media_tools["ffprobe"],
            _allow_test_fd_fallback=True,
        )
    assert calls == []


@pytest.mark.parametrize("replaced_tool", ("ffmpeg", "ffprobe"))
def test_encode_revalidates_after_each_media_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replaced_tool: str,
) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "frame_00000.png").write_bytes(b"frame")
    paths = {"ffmpeg": tmp_path / "ffmpeg", "ffprobe": tmp_path / "ffprobe"}
    _write_fake_media_tool(paths["ffmpeg"], "ffmpeg")
    _write_fake_media_tool(paths["ffprobe"], "ffprobe")
    media_tools = _verified_fake_media_tools(
        monkeypatch, paths["ffmpeg"], paths["ffprobe"]
    )
    replacement = tmp_path / f"{replaced_tool}-replacement"
    _write_fake_media_tool(replacement, replaced_tool, version="9.9-replaced")
    inode_to_name = {
        media_tools[name]["identity"]["inode"]: name
        for name in ("ffmpeg", "ffprobe")
    }
    actual_calls: list[str] = []

    def replace_during_actual_run(command, **kwargs):
        invoked = list(command)
        pass_fds = kwargs.get("pass_fds")
        assert isinstance(pass_fds, tuple) and len(pass_fds) == 1
        descriptor = pass_fds[0]
        name = inode_to_name[os.fstat(descriptor).st_ino]
        assert invoked[0] in (
            f"/proc/self/fd/{descriptor}",
            f"/dev/fd/{descriptor}",
        )
        actual_calls.append(name)
        if name == replaced_tool:
            os.replace(replacement, paths[replaced_tool])
        if name == "ffprobe":
            return SimpleNamespace(
                stdout=_single_frame_probe_stdout(), stderr="", returncode=0
            )
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", replace_during_actual_run)

    with pytest.raises(RuntimeError, match="changed since startup"):
        runner._encode_h264(
            frames,
            tmp_path / "evidence.mp4",
            fps=12,
            expected_width=640,
            expected_height=360,
            expected_frames=1,
            ffmpeg_provenance=media_tools["ffmpeg"],
            ffprobe_provenance=media_tools["ffprobe"],
            _allow_test_fd_fallback=True,
        )
    expected_actual_tools = ["ffmpeg"] if replaced_tool == "ffmpeg" else ["ffmpeg", "ffprobe"]
    assert actual_calls == expected_actual_tools


def test_encode_rejects_incomplete_provenance_before_actual_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "frame_00000.png").write_bytes(b"frame")
    ffmpeg = tmp_path / "ffmpeg"
    ffprobe = tmp_path / "ffprobe"
    _write_fake_media_tool(ffmpeg, "ffmpeg")
    _write_fake_media_tool(ffprobe, "ffprobe")
    media_tools = _verified_fake_media_tools(monkeypatch, ffmpeg, ffprobe)
    incomplete_ffmpeg = dict(media_tools["ffmpeg"])
    incomplete_ffmpeg.pop("identity")

    def unexpected_run(_command, **_kwargs):
        pytest.fail("incomplete provenance must fail before media execution")

    monkeypatch.setattr(runner.subprocess, "run", unexpected_run)

    with pytest.raises(RuntimeError, match="changed since startup"):
        runner._encode_h264(
            frames,
            tmp_path / "evidence.mp4",
            fps=12,
            expected_width=640,
            expected_height=360,
            expected_frames=1,
            ffmpeg_provenance=incomplete_ffmpeg,
            ffprobe_provenance=media_tools["ffprobe"],
            _allow_test_fd_fallback=True,
        )


def test_pinned_media_tool_preserves_body_failure_over_verify_and_close_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = tmp_path / "ffmpeg"
    sha256 = _write_fake_media_tool(tool, "ffmpeg")
    provenance = {
        "supplied_path": str(tool),
        "resolved_realpath": str(tool.resolve(strict=True)),
        "size_bytes": tool.stat().st_size,
        "sha256": sha256,
        "version_line": "ffmpeg version 7.1-test",
        "identity": _media_identity(tool),
    }
    original_close = runner.os.close

    def fail_verify(_descriptor, _provenance, *, tool_name):
        assert tool_name == "ffmpeg"
        raise RuntimeError("synthetic cleanup verification failure")

    def close_then_fail(descriptor: int) -> None:
        original_close(descriptor)
        raise RuntimeError("synthetic descriptor close failure")

    monkeypatch.setattr(runner, "_verify_pinned_media_tool", fail_verify)
    monkeypatch.setattr(runner.os, "close", close_then_fail)

    with pytest.raises(RuntimeError, match="synthetic primary encode failure") as caught:
        with runner._pinned_media_tool(
            provenance,
            tool_name="ffmpeg",
            allow_test_fallback=True,
        ):
            raise RuntimeError("synthetic primary encode failure")

    notes = getattr(caught.value, "__notes__", [])
    if notes:
        assert any("cleanup verification failure" in note for note in notes)
        assert any("descriptor close failure" in note for note in notes)


def test_pinned_media_tool_surfaces_verify_before_close_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = tmp_path / "ffprobe"
    sha256 = _write_fake_media_tool(tool, "ffprobe")
    provenance = {
        "supplied_path": str(tool),
        "resolved_realpath": str(tool.resolve(strict=True)),
        "size_bytes": tool.stat().st_size,
        "sha256": sha256,
        "version_line": "ffprobe version 7.1-test",
        "identity": _media_identity(tool),
    }
    original_close = runner.os.close

    monkeypatch.setattr(
        runner,
        "_verify_pinned_media_tool",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic post-use verification failure")
        ),
    )

    def close_then_fail(descriptor: int) -> None:
        original_close(descriptor)
        raise RuntimeError("synthetic descriptor close failure")

    monkeypatch.setattr(runner.os, "close", close_then_fail)
    with pytest.raises(RuntimeError, match="synthetic post-use verification failure"):
        with runner._pinned_media_tool(
            provenance,
            tool_name="ffprobe",
            allow_test_fallback=True,
        ):
            pass


@pytest.mark.parametrize("open_kind", ["startup", "reuse"])
def test_media_open_cleanup_preserves_primary_hash_failure_over_close_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    open_kind: str,
) -> None:
    tool_name = "ffmpeg"
    tool = tmp_path / tool_name
    sha256 = _write_fake_media_tool(tool, tool_name)
    provenance = {
        "supplied_path": str(tool),
        "resolved_realpath": str(tool.resolve(strict=True)),
        "size_bytes": tool.stat().st_size,
        "sha256": sha256,
        "version_line": "ffmpeg version 7.1-test",
        "identity": _media_identity(tool),
    }
    original_close = runner.os.close

    monkeypatch.setattr(
        runner,
        "_hash_media_descriptor",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic primary media hash failure")
        ),
    )

    def close_then_fail(descriptor: int) -> None:
        original_close(descriptor)
        raise RuntimeError("synthetic media descriptor close failure")

    monkeypatch.setattr(runner.os, "close", close_then_fail)
    with pytest.raises(RuntimeError, match="synthetic primary media hash failure") as caught:
        if open_kind == "startup":
            runner._verified_media_tool(
                tool,
                sha256,
                tool_name=tool_name,
                _allow_test_fd_fallback=True,
            )
        else:
            runner._open_pinned_media_tool(
                provenance,
                tool_name=tool_name,
                allow_test_fallback=True,
            )

    notes = list(getattr(caught.value, "__notes__", [])) + list(
        getattr(caught.value, "_wam_secondary_failures", [])
    )
    assert any("descriptor close failure" in note for note in notes)


class _FakeReplicatorOrchestrator:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.calls: list[bool] = []

    def set_capture_on_play(self, enabled: bool) -> None:
        self.calls.append(enabled)
        self.settings.values["/omni/replicator/captureOnPlay"] = enabled


class _FakeCarbSettings:
    def __init__(self) -> None:
        self.values = {"/omni/replicator/captureOnPlay": True}

    def get(self, key: str):
        return self.values.get(key)


def test_replicator_capture_on_play_is_explicitly_disabled() -> None:
    settings = _FakeCarbSettings()
    rep = SimpleNamespace(
        orchestrator=_FakeReplicatorOrchestrator(settings)
    )

    evidence = runner._configure_replicator_explicit_capture(rep, settings)

    assert rep.orchestrator.calls == [False]
    assert evidence == {
        "capture_on_play": False,
        "capture_trigger": "explicit_zero_delta_orchestrator_step_only",
    }


class _FakeSyncWorld:
    def __init__(
        self,
        *,
        fail_step: bool = False,
        fail_pause: bool = False,
        fail_restore: bool = False,
        silent_pause_failure: bool = False,
        silent_restore_failure: bool = False,
    ) -> None:
        self._playing = False
        self._physics_dt = 1.0 / 60.0
        self._rendering_dt = 1.0 / 60.0
        self.current_time = 2.5
        self.current_time_step_index = 8
        self.fail_step = fail_step
        self.fail_pause = fail_pause
        self.fail_restore = fail_restore
        self.silent_pause_failure = silent_pause_failure
        self.silent_restore_failure = silent_restore_failure
        self.calls: list[object] = []

    def is_playing(self) -> bool:
        return self._playing

    def is_stopped(self) -> bool:
        return False

    def get_physics_dt(self) -> float:
        return self._physics_dt

    def get_rendering_dt(self) -> float:
        return self._rendering_dt

    def set_simulation_dt(self, *, physics_dt: float) -> None:
        self.calls.append(("set_dt", physics_dt))
        if self.fail_restore and physics_dt != 0.0:
            raise RuntimeError("synthetic dt restore failure")
        if self.silent_restore_failure and physics_dt != 0.0:
            return
        self._physics_dt = physics_dt

    def play(self) -> None:
        self.calls.append("play")
        self._playing = True

    def pause(self) -> None:
        self.calls.append("pause")
        if self.fail_pause:
            raise RuntimeError("synthetic pause failure")
        if self.silent_pause_failure:
            return
        self._playing = False

    def step(self, *, render: bool, step_sim: bool, update_fabric: bool) -> None:
        self.calls.append(
            ("step", render, step_sim, update_fabric, self._physics_dt)
        )
        timeline = getattr(self, "timeline", None)
        if timeline is not None and timeline.is_auto_updating():
            raise RuntimeError("timeline auto-update remained enabled during flush")
        if self.fail_step:
            raise RuntimeError("synthetic primary sync failure")
        self.current_time += self._physics_dt
        if self._physics_dt > 0.0:
            self.current_time_step_index += 1

    def render(self) -> None:
        self.calls.append("render")


class _FakeTimeline:
    def __init__(
        self,
        world: _FakeSyncWorld,
        *,
        fail_restore_auto_update: bool = False,
        silent_restore_auto_update: bool = False,
    ) -> None:
        self.world = world
        self.world.timeline = self
        self.current_time = 4.25
        self.auto_update = True
        self.fail_restore_auto_update = fail_restore_auto_update
        self.silent_restore_auto_update = silent_restore_auto_update
        self.calls: list[object] = []

    def get_current_time(self) -> float:
        return self.current_time

    def is_playing(self) -> bool:
        return self.world.is_playing()

    def is_stopped(self) -> bool:
        return False

    def is_auto_updating(self) -> bool:
        return self.auto_update

    def set_auto_update(self, enabled: bool) -> None:
        self.calls.append(("set_auto_update", enabled))
        if self.fail_restore_auto_update and enabled:
            raise RuntimeError("synthetic timeline auto-update restore failure")
        if self.silent_restore_auto_update and enabled:
            return
        self.auto_update = enabled

    def commit_silently(self) -> None:
        self.calls.append("commit_silently")


def test_zero_dt_fabric_sync_restores_timeline_dt_and_time() -> None:
    world = _FakeSyncWorld()
    timeline = _FakeTimeline(world)

    evidence = runner._synchronize_articulation_render_state(
        world, timeline=timeline
    )

    assert evidence == {
        "method": "zero_physics_dt_step_update_fabric_then_render",
        "original_physics_dt_s": pytest.approx(1.0 / 60.0),
        "restored_physics_dt_s": pytest.approx(1.0 / 60.0),
        "original_rendering_dt_s": pytest.approx(1.0 / 60.0),
        "restored_rendering_dt_s": pytest.approx(1.0 / 60.0),
        "simulation_time_before_s": 2.5,
        "simulation_time_after_s": 2.5,
        "physics_step_index_before": 8,
        "physics_step_index_after": 8,
        "physics_step_index_delta": 0,
        "fabric_flush_call_count": 1,
        "timeline_time_before_s": 4.25,
        "timeline_time_after_s": 4.25,
        "timeline_auto_update_before_sync": True,
        "timeline_auto_update_after_sync": True,
        "timeline_auto_update_disabled_during_flush": True,
        "timeline_paused_after_sync": True,
        "timeline_stopped_after_sync": False,
        "physics_integration_used": False,
        "claim_scope": "visual_pose_and_fabric_flush_only",
        "render_gate": {
            "label": "world_render_after_fabric_flush",
            "world_time_delta_s": 0.0,
            "physics_step_index_delta": 0,
            "timeline_time_delta_s": 0.0,
            "physics_dt_unchanged": True,
            "rendering_dt_unchanged": True,
            "timeline_auto_update_unchanged": True,
            "timeline_paused_after": True,
            "timeline_stopped_after": False,
        },
    }
    assert world.calls == [
        ("set_dt", 0.0),
        "play",
        ("step", False, True, True, 0.0),
        "pause",
        ("set_dt", pytest.approx(1.0 / 60.0)),
        "render",
    ]
    assert timeline.calls == [
        ("set_auto_update", False),
        "commit_silently",
        ("set_auto_update", True),
        "commit_silently",
    ]


def test_zero_dt_fabric_sync_restores_state_when_step_fails() -> None:
    world = _FakeSyncWorld(fail_step=True)
    timeline = _FakeTimeline(world)

    with pytest.raises(RuntimeError, match="synthetic primary sync failure"):
        runner._synchronize_articulation_render_state(world, timeline=timeline)

    assert world.is_playing() is False
    assert world.get_physics_dt() == pytest.approx(1.0 / 60.0)
    assert world.calls[-2:] == ["pause", ("set_dt", pytest.approx(1.0 / 60.0))]
    assert "render" not in world.calls


def test_zero_dt_fabric_sync_preserves_primary_failure_over_cleanup_failures() -> None:
    world = _FakeSyncWorld(
        fail_step=True,
        fail_pause=True,
        fail_restore=True,
    )
    timeline = _FakeTimeline(world, fail_restore_auto_update=True)

    with pytest.raises(RuntimeError, match="synthetic primary sync failure") as caught:
        runner._synchronize_articulation_render_state(world, timeline=timeline)

    notes = list(getattr(caught.value, "__notes__", [])) + list(
        getattr(caught.value, "_wam_secondary_failures", [])
    )
    assert any("synthetic pause failure" in note for note in notes)
    assert any("synthetic dt restore failure" in note for note in notes)
    assert any("timeline auto-update restore failure" in note for note in notes)
    assert world.calls[-2:] == [
        "pause",
        ("set_dt", pytest.approx(1.0 / 60.0)),
    ]


def test_zero_dt_fabric_sync_audits_silent_cleanup_failure_before_primary_rethrow() -> None:
    world = _FakeSyncWorld(
        fail_step=True,
        silent_pause_failure=True,
        silent_restore_failure=True,
    )
    timeline = _FakeTimeline(world, silent_restore_auto_update=True)

    with pytest.raises(RuntimeError, match="synthetic primary sync failure") as caught:
        runner._synchronize_articulation_render_state(world, timeline=timeline)

    notes = list(getattr(caught.value, "__notes__", [])) + list(
        getattr(caught.value, "_wam_secondary_failures", [])
    )
    assert any("cleanup audit" in note for note in notes)


def test_zero_dt_fabric_sync_audits_silent_auto_update_restore_failure() -> None:
    world = _FakeSyncWorld(fail_step=True)
    timeline = _FakeTimeline(world, silent_restore_auto_update=True)

    with pytest.raises(RuntimeError, match="synthetic primary sync failure") as caught:
        runner._synchronize_articulation_render_state(world, timeline=timeline)

    notes = list(getattr(caught.value, "__notes__", [])) + list(
        getattr(caught.value, "_wam_secondary_failures", [])
    )
    assert any("auto-update" in note for note in notes)


def test_zero_time_operation_rejects_world_or_timeline_advance() -> None:
    world = _FakeSyncWorld()
    timeline = _FakeTimeline(world)

    result, evidence = runner._run_zero_time_operation(
        world,
        timeline,
        lambda: "captured",
        label="test_capture",
    )
    assert result == "captured"
    assert evidence["physics_step_index_delta"] == 0
    assert evidence["timeline_time_delta_s"] == 0.0
    assert evidence["timeline_auto_update_unchanged"] is True

    def advance_timeline() -> None:
        timeline.current_time += 0.01

    with pytest.raises(RuntimeError, match="advanced the USD timeline"):
        runner._run_zero_time_operation(
            world,
            timeline,
            advance_timeline,
            label="bad_capture",
        )


def test_render_sync_evidence_requires_motion_exact_readback_and_paused_timeline() -> None:
    initial_robot = np.zeros((20, 20), dtype=bool)
    moved_robot = np.zeros((20, 20), dtype=bool)
    initial_robot[2:12, 2:10] = True
    moved_robot[7:17, 10:18] = True
    initial_human = np.ones((20, 20), dtype=bool)
    moved_human = initial_human.copy()
    restored_robot = initial_robot.copy()
    restored_human = initial_human.copy()
    initial = np.zeros(7)
    commanded = initial.copy()
    commanded[0] = 0.45
    observed = commanded.copy()
    observed[0] -= 1.0e-8
    sync = {
        "method": "zero_physics_dt_step_update_fabric_then_render",
        "original_physics_dt_s": 1.0 / 60.0,
        "restored_physics_dt_s": 1.0 / 60.0,
        "original_rendering_dt_s": 1.0 / 60.0,
        "restored_rendering_dt_s": 1.0 / 60.0,
        "simulation_time_before_s": 0.0,
        "simulation_time_after_s": 0.0,
        "physics_step_index_delta": 0,
        "fabric_flush_call_count": 1,
        "timeline_time_before_s": 0.0,
        "timeline_time_after_s": 0.0,
        "timeline_auto_update_before_sync": True,
        "timeline_auto_update_after_sync": True,
        "timeline_auto_update_disabled_during_flush": True,
        "timeline_paused_after_sync": True,
        "timeline_stopped_after_sync": False,
        "physics_integration_used": False,
        "claim_scope": "visual_pose_and_fabric_flush_only",
    }

    evidence = runner._validated_render_sync_evidence(
        initial_robot_mask=initial_robot,
        moved_robot_mask=moved_robot,
        initial_human_mask=initial_human,
        moved_human_mask=moved_human,
        restored_robot_mask=restored_robot,
        restored_human_mask=restored_human,
        initial_joint_positions=initial,
        commanded_joint_positions=commanded,
        observed_joint_positions=observed,
        sync_evidence=sync,
    )

    assert evidence["robot_mask_changed_pixels"] >= 64
    assert evidence["joint_motion_max_abs_rad"] == pytest.approx(0.45)
    assert evidence["readback_max_abs_error_rad"] == pytest.approx(1.0e-8)

    broken_sync_evidence = []
    missing_index_delta = dict(sync)
    missing_index_delta.pop("physics_step_index_delta")
    broken_sync_evidence.append(missing_index_delta)
    for key, bad_value in (
        ("physics_step_index_delta", 1),
        ("fabric_flush_call_count", 0),
        ("timeline_time_after_s", 0.01),
        ("timeline_auto_update_before_sync", False),
        ("timeline_auto_update_after_sync", False),
        ("timeline_auto_update_disabled_during_flush", False),
    ):
        broken = dict(sync)
        broken[key] = bad_value
        broken_sync_evidence.append(broken)
    for broken in broken_sync_evidence:
        with pytest.raises(RuntimeError, match="timing (evidence|contract)"):
            runner._validated_render_sync_evidence(
                initial_robot_mask=initial_robot,
                moved_robot_mask=moved_robot,
                initial_human_mask=initial_human,
                moved_human_mask=moved_human,
                restored_robot_mask=restored_robot,
                restored_human_mask=restored_human,
                initial_joint_positions=initial,
                commanded_joint_positions=commanded,
                observed_joint_positions=observed,
                sync_evidence=broken,
            )

    with pytest.raises(RuntimeError, match="robot mask did not follow"):
        runner._validated_render_sync_evidence(
            initial_robot_mask=initial_robot,
            moved_robot_mask=initial_robot.copy(),
            initial_human_mask=initial_human,
            moved_human_mask=moved_human,
            restored_robot_mask=restored_robot,
            restored_human_mask=restored_human,
            initial_joint_positions=initial,
            commanded_joint_positions=commanded,
            observed_joint_positions=observed,
            sync_evidence=sync,
        )
    with pytest.raises(RuntimeError, match="joint readback"):
        runner._validated_render_sync_evidence(
            initial_robot_mask=initial_robot,
            moved_robot_mask=moved_robot,
            initial_human_mask=initial_human,
            moved_human_mask=moved_human,
            restored_robot_mask=restored_robot,
            restored_human_mask=restored_human,
            initial_joint_positions=initial,
            commanded_joint_positions=commanded,
            observed_joint_positions=commanded + 2.0e-6,
            sync_evidence=sync,
        )

    bad_restore = restored_robot.copy()
    bad_restore[12:17, 10:18] = True
    with pytest.raises(RuntimeError, match="did not return to the initial pose"):
        runner._validated_render_sync_evidence(
            initial_robot_mask=initial_robot,
            moved_robot_mask=moved_robot,
            initial_human_mask=initial_human,
            moved_human_mask=moved_human,
            restored_robot_mask=bad_restore,
            restored_human_mask=restored_human,
            initial_joint_positions=initial,
            commanded_joint_positions=commanded,
            observed_joint_positions=observed,
            sync_evidence=sync,
        )


def test_render_sync_motion_threshold_scales_with_robot_union() -> None:
    initial_robot = np.ones((100, 100), dtype=bool)
    moved_robot = initial_robot.copy()
    moved_robot[:8, :10] = False
    human = np.ones((100, 100), dtype=bool)
    initial = np.zeros(7)
    commanded = initial.copy()
    commanded[0] = 0.45
    sync = {
        "method": "zero_physics_dt_step_update_fabric_then_render",
        "original_physics_dt_s": 1.0 / 60.0,
        "restored_physics_dt_s": 1.0 / 60.0,
        "original_rendering_dt_s": 1.0 / 60.0,
        "restored_rendering_dt_s": 1.0 / 60.0,
        "simulation_time_before_s": 0.0,
        "simulation_time_after_s": 0.0,
        "physics_step_index_delta": 0,
        "fabric_flush_call_count": 1,
        "timeline_time_before_s": 0.0,
        "timeline_time_after_s": 0.0,
        "timeline_auto_update_before_sync": True,
        "timeline_auto_update_after_sync": True,
        "timeline_auto_update_disabled_during_flush": True,
        "timeline_paused_after_sync": True,
        "timeline_stopped_after_sync": False,
        "physics_integration_used": False,
        "claim_scope": "visual_pose_and_fabric_flush_only",
    }

    with pytest.raises(RuntimeError, match="robot mask did not follow"):
        runner._validated_render_sync_evidence(
            initial_robot_mask=initial_robot,
            moved_robot_mask=moved_robot,
            initial_human_mask=human,
            moved_human_mask=human,
            restored_robot_mask=initial_robot,
            restored_human_mask=human,
            initial_joint_positions=initial,
            commanded_joint_positions=commanded,
            observed_joint_positions=commanded,
            sync_evidence=sync,
        )


def _trajectory_mask_sample(
    frame_index: int,
    joints: np.ndarray,
    mask: np.ndarray,
    phase: str,
) -> dict:
    return {
        "frame_index": frame_index,
        "joint_positions_rad": joints,
        "robot_mask": mask,
        "motion_phase": phase,
    }


def test_trajectory_mask_keyframes_require_relative_motion_and_home_restore() -> None:
    home_joints = np.zeros(7)
    home_mask = np.zeros((100, 100), dtype=bool)
    home_mask[30:70, 20:50] = True
    samples = [
        _trajectory_mask_sample(0, home_joints, home_mask, "initial_home"),
    ]
    for index, shift in enumerate((10, 20, 30), start=1):
        moved_joints = home_joints.copy()
        moved_joints[0] = 0.3 + index * 0.1
        moved_mask = np.zeros_like(home_mask)
        moved_mask[30:70, 20 + shift : 50 + shift] = True
        samples.append(
            _trajectory_mask_sample(
                index * 8,
                moved_joints,
                moved_mask,
                "nominal_contact",
            )
        )
    samples.append(
        _trajectory_mask_sample(32, home_joints, home_mask, "return_home")
    )

    evidence = runner._validated_trajectory_mask_keyframes(samples)

    assert evidence["meaningful_motion_keyframe_count"] == 3
    assert evidence["return_home_keyframe_count"] == 1
    assert evidence["all_meaningful_motion_keyframes_passed"] is True
    assert evidence["all_return_home_keyframes_passed"] is True


def test_trajectory_mask_keyframes_reject_hash_only_single_pixel_noise() -> None:
    home_joints = np.zeros(7)
    home_mask = np.zeros((100, 100), dtype=bool)
    home_mask[30:70, 20:50] = True
    samples = [
        _trajectory_mask_sample(0, home_joints, home_mask, "initial_home"),
    ]
    for index in range(1, 4):
        moved_joints = home_joints.copy()
        moved_joints[0] = 0.3 + index * 0.1
        noisy_mask = home_mask.copy()
        noisy_mask[index, index] = True
        samples.append(
            _trajectory_mask_sample(
                index * 8,
                moved_joints,
                noisy_mask,
                "nominal_contact",
            )
        )
    samples.append(
        _trajectory_mask_sample(32, home_joints, home_mask, "return_home")
    )

    with pytest.raises(RuntimeError, match="keyframe mask did not follow"):
        runner._validated_trajectory_mask_keyframes(samples)


def test_trajectory_mask_keyframes_reject_bad_home_restore() -> None:
    home_joints = np.zeros(7)
    home_mask = np.zeros((100, 100), dtype=bool)
    home_mask[30:70, 20:50] = True
    samples = [
        _trajectory_mask_sample(0, home_joints, home_mask, "initial_home"),
    ]
    for index, shift in enumerate((10, 20, 30), start=1):
        moved_joints = home_joints.copy()
        moved_joints[0] = 0.3 + index * 0.1
        moved_mask = np.zeros_like(home_mask)
        moved_mask[30:70, 20 + shift : 50 + shift] = True
        samples.append(
            _trajectory_mask_sample(
                index * 8,
                moved_joints,
                moved_mask,
                "nominal_contact",
            )
        )
    bad_restore = home_mask.copy()
    bad_restore[0:2, 0:10] = True
    samples.append(
        _trajectory_mask_sample(32, home_joints, bad_restore, "return_home")
    )

    with pytest.raises(RuntimeError, match="return-home mask did not restore"):
        runner._validated_trajectory_mask_keyframes(samples)


def test_simulation_app_config_uses_verified_fast_process_shutdown() -> None:
    config = runner._simulation_app_config(
        SimpleNamespace(width=640, height=360)
    )

    assert config["fast_shutdown"] is True
    assert config["headless"] is True
    assert config["width"] == 640
    assert config["height"] == 360


class _FakeSimulationApp:
    def __init__(self, *, fail_close: bool = False) -> None:
        self.fail_close = fail_close
        self.close_calls: list[tuple[bool, bool]] = []

    def close(self, *, wait_for_replicator: bool, skip_cleanup: bool) -> None:
        self.close_calls.append((wait_for_replicator, skip_cleanup))
        if self.fail_close:
            raise RuntimeError("synthetic close failure")


class _ProcessExit(BaseException):
    def __init__(self, code: int) -> None:
        super().__init__(code)
        self.code = code


def _raise_process_exit(code: int) -> None:
    raise _ProcessExit(code)


class _FailingDiagnosticStream(io.StringIO):
    def __init__(self, *, fail_write: bool = False, fail_flush: bool = False) -> None:
        super().__init__()
        self.fail_write = fail_write
        self.fail_flush = fail_flush

    def write(self, value: str) -> int:
        if self.fail_write:
            raise BrokenPipeError("synthetic diagnostic write failure")
        return super().write(value)

    def flush(self) -> None:
        if self.fail_flush:
            raise BrokenPipeError("synthetic diagnostic flush failure")
        super().flush()


def test_isolated_process_persists_failure_before_nonzero_exit_without_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _FakeSimulationApp(fail_close=True)
    args = SimpleNamespace(out=str(tmp_path / "failed-run"), mode="smoke")
    stdout = io.StringIO()
    stderr = io.StringIO()

    def fail_run(_args, *, runtime_provenance):
        assert runtime_provenance == {"runtime": "test"}
        raise RuntimeError("synthetic primary run failure")

    monkeypatch.setattr(runner, "_run", fail_run)
    with pytest.raises(_ProcessExit) as caught:
        runner._run_isolated_simulation_process(
            args,
            app,
            {"runtime": "test"},
            exit_process=_raise_process_exit,
            stdout=stdout,
            stderr=stderr,
        )

    assert caught.value.code == 1
    assert app.close_calls == []
    assert "synthetic primary run failure" in stderr.getvalue()
    assert "SMOKE_PASS" not in stdout.getvalue()
    terminal = json.loads(runner._process_terminal_path(args.out).read_text())
    assert terminal["terminal_state"] == "RUN_BODY_FAILED"
    assert terminal["expected_process_exit_code"] == 1
    assert terminal["exception_type"] == "RuntimeError"


def test_isolated_process_persists_success_before_fast_close_and_zero_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _FakeSimulationApp()
    args = SimpleNamespace(out=str(tmp_path / "successful-run"), mode="smoke")
    stdout = io.StringIO()
    stderr = io.StringIO()
    summary = {"status": "SMOKE_PASS"}
    monkeypatch.setattr(
        runner,
        "_run",
        lambda _args, *, runtime_provenance: summary,
    )

    with pytest.raises(_ProcessExit) as caught:
        runner._run_isolated_simulation_process(
            args,
            app,
            {"runtime": "test"},
            exit_process=_raise_process_exit,
            stdout=stdout,
            stderr=stderr,
        )

    assert caught.value.code == 0
    assert app.close_calls == [(False, True)]
    assert json.loads(stdout.getvalue()) == summary
    assert stderr.getvalue() == ""
    terminal = json.loads(runner._process_terminal_path(args.out).read_text())
    assert terminal["terminal_state"] == "RUN_BODY_PASS_AWAITING_SUPERVISOR_EXIT"
    assert terminal["expected_process_exit_code"] == 0
    assert terminal["summary"] == summary
    assert Path(args.out) not in runner._process_terminal_path(args.out).parents


def test_isolated_process_close_failure_cannot_produce_zero_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _FakeSimulationApp(fail_close=True)
    args = SimpleNamespace(out=str(tmp_path / "close-failed-run"), mode="smoke")
    stdout = io.StringIO()
    stderr = io.StringIO()
    summary = {"status": "SMOKE_PASS"}
    monkeypatch.setattr(
        runner,
        "_run",
        lambda _args, *, runtime_provenance: summary,
    )

    with pytest.raises(_ProcessExit) as caught:
        runner._run_isolated_simulation_process(
            args,
            app,
            {"runtime": "test"},
            exit_process=_raise_process_exit,
            stdout=stdout,
            stderr=stderr,
        )

    assert caught.value.code == 1
    assert app.close_calls == [(False, True)]
    assert "synthetic close failure" in stderr.getvalue()


def test_isolated_process_success_output_failure_hard_exits_one_without_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _FakeSimulationApp()
    args = SimpleNamespace(out=str(tmp_path / "output-failed-run"), mode="smoke")
    stdout = _FailingDiagnosticStream(fail_flush=True)
    stderr = _FailingDiagnosticStream(fail_write=True, fail_flush=True)
    monkeypatch.setattr(
        runner,
        "_run",
        lambda _args, *, runtime_provenance: {"status": "SMOKE_PASS"},
    )

    with pytest.raises(_ProcessExit) as caught:
        runner._run_isolated_simulation_process(
            args,
            app,
            {"runtime": "test"},
            exit_process=_raise_process_exit,
            stdout=stdout,
            stderr=stderr,
        )

    assert caught.value.code == 1
    assert app.close_calls == []


def test_isolated_process_failure_diagnostic_io_cannot_block_nonzero_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _FakeSimulationApp()
    args = SimpleNamespace(out=str(tmp_path / "diagnostic-failed-run"), mode="smoke")
    stdout = _FailingDiagnosticStream(fail_flush=True)
    stderr = _FailingDiagnosticStream(fail_write=True, fail_flush=True)

    def fail_run(_args, *, runtime_provenance):
        raise RuntimeError("synthetic run failure before broken diagnostics")

    monkeypatch.setattr(runner, "_run", fail_run)
    with pytest.raises(_ProcessExit) as caught:
        runner._run_isolated_simulation_process(
            args,
            app,
            {"runtime": "test"},
            exit_process=_raise_process_exit,
            stdout=stdout,
            stderr=stderr,
        )

    assert caught.value.code == 1
    assert app.close_calls == []


def test_process_terminal_atomic_commit_never_overwrites_racing_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = runner._process_terminal_path(tmp_path / "race-run")
    rival = b"RIVAL_MARKER\n"
    original_link = runner.os.link

    def racing_link(source, target, **kwargs):
        Path(target).write_bytes(rival)
        return original_link(source, target, **kwargs)

    monkeypatch.setattr(runner.os, "link", racing_link)
    with pytest.raises(RuntimeError, match="already exists|commit"):
        runner._write_process_terminal(
            destination,
            {"schema": "test", "terminal_state": "BODY_PASS"},
        )

    assert destination.read_bytes() == rival
    assert not list(tmp_path.glob(f"{destination.name}.tmp-*"))


def test_process_terminal_cleanup_preserves_primary_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = runner._process_terminal_path(tmp_path / "write-failed-run")
    original_close = runner.os.close
    monkeypatch.setattr(
        runner.os,
        "write",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic primary terminal write failure")
        ),
    )

    def close_then_fail(descriptor: int) -> None:
        original_close(descriptor)
        raise RuntimeError("synthetic terminal descriptor close failure")

    monkeypatch.setattr(runner.os, "close", close_then_fail)
    with pytest.raises(
        RuntimeError, match="synthetic primary terminal write failure"
    ) as caught:
        runner._write_process_terminal(
            destination,
            {"schema": "test", "terminal_state": "BODY_PASS"},
        )

    notes = list(getattr(caught.value, "__notes__", [])) + list(
        getattr(caught.value, "_wam_secondary_failures", [])
    )
    assert any("descriptor close failure" in note for note in notes)
    assert not destination.exists()
    assert not list(tmp_path.glob(f"{destination.name}.tmp-*"))


def _successful_supervisor_fixture(output_root: Path) -> tuple[Path, dict, dict]:
    output_root.mkdir()
    (output_root / "frame.png").write_bytes(b"official-frame")
    nested = output_root / "raw"
    nested.mkdir()
    (nested / "metrics.json").write_text('{"status":"SMOKE_PASS"}\n')
    integrity = runner._write_sha256s(output_root)
    summary = {
        "status": "SMOKE_PASS",
        "mode": "smoke",
        "out": str(output_root),
        "robot_pixels": 100,
        "human_pixels": 200,
        "mesh_count": 42,
        "output_integrity": integrity,
    }
    terminal = runner._process_terminal_path(output_root)
    payload = {
        "schema": "wam_isolated_simulation_process_terminal_v1",
        "terminal_state": "RUN_BODY_PASS_AWAITING_SUPERVISOR_EXIT",
        "mode": "smoke",
        "out": str(output_root),
        "expected_process_exit_code": 0,
        "summary": summary,
        "shutdown": {
            "method": "SimulationApp.close",
            "wait_for_replicator": False,
            "skip_cleanup": True,
            "fast_shutdown": True,
            "supervisor_exit_status_required": True,
        },
        "production_activation": False,
    }
    runner._write_process_terminal(terminal, payload)
    properties = {
        "LoadState": "loaded",
        "ActiveState": "active",
        "SubState": "exited",
        "Result": "success",
        "ExecMainCode": "1",
        "ExecMainStatus": "0",
    }
    return terminal, payload, properties


def test_output_integrity_is_canonical_exact_and_round_trips(tmp_path: Path) -> None:
    (tmp_path / "b.bin").write_bytes(b"b")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "a.bin").write_bytes(b"a")

    integrity = runner._write_sha256s(tmp_path)
    verified = runner._verify_output_integrity(tmp_path)

    assert verified == integrity
    assert integrity["schema"] == "wam_output_integrity_v1"
    assert integrity["file_count"] == 2
    assert integrity["sha256s_sha256"] == integrity["payload_inventory_sha256"]
    rows = (tmp_path / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    assert [row[66:] for row in rows] == ["b.bin", "nested/a.bin"]
    assert all(len(row[:64]) == 64 and row[64:66] == "  " for row in rows)


def test_output_integrity_rejects_symlink_payload(tmp_path: Path) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(b"target")
    (tmp_path / "alias.bin").symlink_to(target)

    with pytest.raises(RuntimeError, match="symlink|regular"):
        runner._write_sha256s(tmp_path)


def test_output_integrity_pins_manifest_against_lstat_read_symlink_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "payload.bin").write_bytes(b"payload")
    runner._write_sha256s(tmp_path)
    manifest = tmp_path / "SHA256SUMS.txt"
    external = tmp_path.with_name(f"{tmp_path.name}-external-manifest.txt")
    external.write_bytes(manifest.read_bytes())
    original_read_bytes = Path.read_bytes
    original_open = runner.os.open
    raced = False

    def race_manifest_path() -> None:
        nonlocal raced
        if raced:
            return
        raced = True
        manifest.unlink()
        manifest.symlink_to(external)

    def racing_read_bytes(path: Path) -> bytes:
        if path == manifest:
            race_manifest_path()
        return original_read_bytes(path)

    def racing_open(path, flags, *args, **kwargs):
        if Path(path) == manifest:
            race_manifest_path()
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", racing_read_bytes)
    monkeypatch.setattr(runner.os, "open", racing_open)
    with pytest.raises(RuntimeError, match="integrity manifest|pinned|symlink"):
        runner._verify_output_integrity(tmp_path)

    assert raced is True
    assert manifest.is_symlink()


def test_output_integrity_rechecks_manifest_after_payload_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "payload.bin").write_bytes(b"payload")
    runner._write_sha256s(tmp_path)
    manifest = tmp_path / "SHA256SUMS.txt"
    replacement = tmp_path.with_name(f"{tmp_path.name}-replacement-manifest.txt")
    replacement_bytes = b"0" * 64 + b"  forged.bin\n"
    replacement.write_bytes(replacement_bytes)
    original_inventory = runner._output_payload_paths
    raced = False

    def racing_inventory(root: Path):
        nonlocal raced
        if not raced:
            raced = True
            os.replace(replacement, manifest)
        return original_inventory(root)

    monkeypatch.setattr(runner, "_output_payload_paths", racing_inventory)
    with pytest.raises(RuntimeError, match="integrity manifest|changed"):
        runner._verify_output_integrity(tmp_path)

    assert raced is True
    assert manifest.read_bytes() == replacement_bytes
    assert not manifest.is_symlink()


@pytest.mark.parametrize("mutation", ["modify", "add", "delete"])
def test_output_integrity_rejects_any_post_manifest_file_set_change(
    tmp_path: Path, mutation: str
) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"original")
    runner._write_sha256s(tmp_path)

    if mutation == "modify":
        payload.write_bytes(b"modified")
    elif mutation == "add":
        (tmp_path / "late.bin").write_bytes(b"late")
    else:
        payload.unlink()

    with pytest.raises(RuntimeError, match="integrity|inventory|SHA-256|file set"):
        runner._verify_output_integrity(tmp_path)


def test_supervisor_completion_requires_marker_rc_and_bound_inventory(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "run"
    terminal, payload, properties = _successful_supervisor_fixture(output_root)

    validated = runner._validated_supervisor_completion(
        terminal,
        properties,
        output_root,
    )

    assert validated["terminal"] == payload
    assert validated["systemd"] == properties
    assert validated["output_integrity"] == payload["summary"]["output_integrity"]
    assert terminal.parent == output_root.parent
    assert terminal not in output_root.rglob("*")


@pytest.mark.parametrize(
    ("property_name", "bad_value"),
    [
        ("LoadState", "not-found"),
        ("ActiveState", "inactive"),
        ("SubState", "dead"),
        ("Result", "core-dump"),
        ("ExecMainCode", "0"),
        ("ExecMainCode", "2"),
        ("ExecMainCode", "3"),
        ("ExecMainCode", "killed"),
        ("ExecMainStatus", "11"),
    ],
)
def test_supervisor_completion_rejects_each_bad_process_property(
    tmp_path: Path, property_name: str, bad_value: str
) -> None:
    output_root = tmp_path / "run"
    terminal, _payload, properties = _successful_supervisor_fixture(output_root)
    properties[property_name] = bad_value

    with pytest.raises(RuntimeError, match="supervisor|systemd|process"):
        runner._validated_supervisor_completion(terminal, properties, output_root)


@pytest.mark.parametrize(
    ("field_path", "bad_value"),
    [
        (("terminal_state",), "RUN_BODY_FAILED"),
        (("expected_process_exit_code",), 1),
        (("production_activation",), True),
        (("mode",), "full"),
        (("summary", "mode"), "full"),
        (("summary", "out"), "/tmp/other-run"),
        (("summary", "output_integrity", "file_count"), 999),
        (("shutdown", "supervisor_exit_status_required"), False),
    ],
)
def test_supervisor_completion_rejects_each_bad_terminal_binding(
    tmp_path: Path, field_path: tuple[str, ...], bad_value: object
) -> None:
    output_root = tmp_path / "run"
    terminal, payload, properties = _successful_supervisor_fixture(output_root)
    cursor = payload
    for field in field_path[:-1]:
        cursor = cursor[field]
    cursor[field_path[-1]] = bad_value
    terminal.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="terminal|completion|integrity"):
        runner._validated_supervisor_completion(terminal, properties, output_root)


def test_supervisor_completion_rejects_output_tamper_after_success_marker(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "run"
    terminal, _payload, properties = _successful_supervisor_fixture(output_root)
    (output_root / "frame.png").write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="integrity|SHA-256"):
        runner._validated_supervisor_completion(terminal, properties, output_root)


URDF = Path(__file__).resolve().parents[2] / "ausc/perception/robot/openarm_v10.urdf"


def test_frozen_robot_root_inverse_has_unit_scale_and_exact_yaw() -> None:
    local = np.array([[0.20, 0.40, 0.80], [-0.10, 0.55, 1.20]])
    rotation = np.diag([-1.0, -1.0, 1.0])
    translation = np.array([0.15, 1.15, 0.0])
    world = local @ rotation.T + translation

    assert np.array_equal(runner._world_to_robot_root(world), local)
    assert np.array_equal(runner._world_to_robot_root(world[0]), local[0])


def test_ik_seed_table_is_deterministic_finite_and_inside_official_limits() -> None:
    limits = experiment.left_joint_limits(URDF.read_bytes())

    first = runner._ik_seeds(limits)
    second = runner._ik_seeds(limits)

    assert first.shape == (37, 7)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert np.all(first >= limits[:, 0])
    assert np.all(first <= limits[:, 1])
    assert np.array_equal(first[0], [-1.0, -1.2, 0.0, 1.15, 0.0, 0.0, -0.30])


def test_json_and_hash_evidence_are_canonical_and_fail_closed_for_nonfinite(tmp_path: Path) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"official-openarm")
    runner._write_json(
        tmp_path / "value.json",
        runner._json_safe(
            {
                "finite": 1.0,
                "nan": float("nan"),
                "inf": np.inf,
                "nested_array": np.array([[1.0, np.nan], [np.inf, 2.0]]),
            }
        ),
    )
    runner._write_sha256s(tmp_path)

    raw = (tmp_path / "value.json").read_text(encoding="utf-8")
    assert '"nan": null' in raw
    assert '"inf": null' in raw
    assert '"nested_array": [' in raw
    assert raw.count("null") == 4
    expected = hashlib.sha256(payload.read_bytes()).hexdigest()
    sums = (tmp_path / "SHA256SUMS.txt").read_text(encoding="utf-8")
    assert f"{expected}  payload.bin" in sums
    assert "SHA256SUMS.txt" not in sums


def test_instance_masks_are_selected_by_exact_scene_root_not_semantic_class() -> None:
    packet = {
        "data": np.array([[0, 11, 12], [21, 22, 31]], dtype=np.uint32),
        "info": {
            "idToLabels": {
                "11": "/World/OpenArm/link1/visual",
                "12": "/World/OpenArm/link2/visual",
                "21": "/World/Human/labcoat",
                "22": "/World/Human/body",
                "31": "/World/OpenArmFake/link",
            }
        },
    }

    robot = runner._instance_mask(packet, "/World/OpenArm")
    human = runner._instance_mask(packet, "/World/Human")

    assert np.array_equal(robot, [[False, True, True], [False, False, False]])
    assert np.array_equal(human, [[False, False, False], [True, True, False]])
    with pytest.raises(RuntimeError, match="did not render"):
        runner._instance_mask(packet, "/World/Missing")


def test_usd_camera_parameters_reproduce_full_fxfycxcy_contract() -> None:
    scaled = runner._scaled_intrinsics(640, 360)
    camera = runner._usd_camera_parameters(scaled, horizontal_aperture_mm=36.0)

    assert np.isclose(camera["focal_length_mm"] / camera["horizontal_aperture_mm"] * 640, scaled["fx"])
    assert np.isclose(camera["focal_length_mm"] / camera["vertical_aperture_mm"] * 360, scaled["fy"])
    assert np.isclose(
        320.0 - camera["horizontal_aperture_offset_mm"] / camera["horizontal_aperture_mm"] * 640,
        scaled["cx"],
    )
    assert np.isclose(
        180.0 + camera["vertical_aperture_offset_mm"] / camera["vertical_aperture_mm"] * 360,
        scaled["cy"],
    )


def test_media_probe_gate_requires_exact_h264_stream_contract() -> None:
    valid = {
        "streams": [
            {
                "codec_name": "h264",
                "pix_fmt": "yuv420p",
                "width": 640,
                "height": 360,
                "nb_frames": "161",
                "r_frame_rate": "12/1",
            }
        ]
    }

    assert runner._validate_video_probe(
        valid,
        expected_width=640,
        expected_height=360,
        expected_fps=12,
        expected_frames=161,
    )["codec_name"] == "h264"
    for field, bad in (
        ("codec_name", "mpeg4"),
        ("pix_fmt", "yuv444p"),
        ("width", 320),
        ("height", 180),
        ("nb_frames", "160"),
        ("r_frame_rate", "24/1"),
    ):
        changed = {"streams": [dict(valid["streams"][0], **{field: bad})]}
        with pytest.raises(RuntimeError, match="video stream contract"):
            runner._validate_video_probe(
                changed,
                expected_width=640,
                expected_height=360,
                expected_fps=12,
                expected_frames=161,
            )


def test_holdout_parameter_trials_are_frozen_stratified_and_not_physical_claims() -> None:
    points = np.array(
        [
            [-0.12, -0.15, 1.0], [0.12, -0.15, 1.0],
            [-0.12, 0.15, 1.0], [0.12, 0.15, 1.0],
            [-0.08, -0.08, 1.0], [0.08, -0.08, 1.0],
            [-0.08, 0.08, 1.0], [0.08, 0.08, 1.0],
            [-0.02, -0.12, 1.0], [0.02, 0.12, 1.0],
        ],
        dtype=np.float64,
    )
    visibility = tuple(
        experiment.RenderedVisibility(
            name=name,
            visible_fraction=0.9,
            robot_front_fraction=0.1,
            valid_depth_fraction=1.0,
            mask_edge_distance_px=12.0,
        )
        for name in experiment.POINT_NAMES
    )

    first = runner._holdout_parameter_trials(
        points, visibility, target_index=4, seed=17, trials_per_stratum=30
    )
    second = runner._holdout_parameter_trials(
        points, visibility, target_index=4, seed=17, trials_per_stratum=30
    )

    assert first == second
    assert first["scope"] == "parameter_sensitivity_only"
    assert first["physical_accuracy_pass"] is False
    assert first["independent_trajectory_claim"] is False
    assert first["trials_per_stratum"] == 30
    assert len(first["strata"]) == 4
    assert all(row["trial_count"] == 30 for row in first["strata"])
    assert all(row["error_p50_m"] <= row["error_p95_m"] <= row["error_p99_m"] for row in first["strata"])
    assert all(len(row["bootstrap_error_p95_95ci_m"]) == 2 for row in first["strata"])


def test_smoke_anchor_alignment_gate_rejects_points_off_the_rendered_back() -> None:
    projected = np.column_stack((np.linspace(120, 520, 10), np.linspace(100, 300, 10)))
    aligned = tuple(
        experiment.RenderedVisibility(name, 0.8, 0.0, 1.0, 20.0)
        for name in experiment.POINT_NAMES
    )
    off_back = tuple(
        experiment.RenderedVisibility(name, 0.8 if index < 7 else 0.0, 0.0, 1.0, 20.0)
        for index, name in enumerate(experiment.POINT_NAMES)
    )

    assert runner._validate_smoke_anchor_alignment(
        projected, aligned, width=640, height=360
    )["aligned_anchor_count"] == 10
    with pytest.raises(RuntimeError, match="official back"):
        runner._validate_smoke_anchor_alignment(
            projected, off_back, width=640, height=360
        )


def test_joint_readback_gate_binds_saved_pose_to_paused_articulation_state() -> None:
    class FakeArticulation:
        def __init__(self, left, right):
            self.left = np.asarray(left, dtype=np.float64)
            self.right = np.asarray(right, dtype=np.float64)

        def get_joint_positions(self, joint_indices):
            return self.left if np.array_equal(joint_indices, np.arange(7)) else self.right

    commanded = np.linspace(-0.3, 0.3, 7)
    right = np.zeros(7)
    indices = np.arange(7)
    gate = runner._validate_joint_readback(
        FakeArticulation(commanded, right),
        left_indices=indices,
        right_indices=indices + 7,
        commanded_left=commanded,
        frozen_right=right,
    )

    assert np.array_equal(gate["observed_left"], commanded)
    assert np.array_equal(gate["observed_right"], right)
    with pytest.raises(RuntimeError, match="left joint readback"):
        runner._validate_joint_readback(
            FakeArticulation(commanded + 1.0e-3, right),
            left_indices=indices,
            right_indices=indices + 7,
            commanded_left=commanded,
            frozen_right=right,
        )
    with pytest.raises(RuntimeError, match="right reference chain"):
        runner._validate_joint_readback(
            FakeArticulation(commanded, right + 1.0e-3),
            left_indices=indices,
            right_indices=indices + 7,
            commanded_left=commanded,
            frozen_right=right,
        )


def test_targeted_occlusion_status_does_not_call_motion_alone_occlusion() -> None:
    missing = runner._targeted_occlusion_status(
        max_robot_front_fraction_sum=0.0,
        max_single_point_robot_front_fraction=0.0,
    )
    observed = runner._targeted_occlusion_status(
        max_robot_front_fraction_sum=0.45,
        max_single_point_robot_front_fraction=0.31,
    )

    assert missing["status"] == "NO_TARGETED_OCCLUSION_EVIDENCE"
    assert missing["observed"] is False
    assert observed["status"] == "TARGETED_OCCLUSION_OBSERVED"
    assert observed["observed"] is True


def test_render_sampling_can_represent_direct_then_bounded_freeze_then_lost() -> None:
    waypoints = np.array(
        [
            [-1.0, -1.2, 0.0, 1.15, 0.0, 0.0, -0.30],
            [-0.9, -1.1, 0.1, 1.20, 0.1, 0.1, 0.20],
        ]
    )
    trajectory = experiment.build_minimum_jerk_trajectory(
        waypoints, frames_per_segment=9, seconds_per_segment=2.8
    )
    dt = float(trajectory.time_s[1] - trajectory.time_s[0])

    short = experiment.decide_occlusion_policy(
        geometry_accepted=False,
        last_valid_age_s=dt,
        last_target_error_p95_m=0.004,
        patient_motion_bound_m_s=0.010,
        max_freeze_age_s=0.40,
        max_freeze_error_p95_m=0.010,
    )
    later = experiment.decide_occlusion_policy(
        geometry_accepted=False,
        last_valid_age_s=2.0 * dt,
        last_target_error_p95_m=0.004,
        patient_motion_bound_m_s=0.010,
        max_freeze_age_s=0.40,
        max_freeze_error_p95_m=0.010,
    )

    assert dt < 0.40
    assert short.state == "BOUNDED_PATIENT_FRAME_FREEZE"
    assert later.state == "LOST"


def test_target_direct_update_gate_is_point_specific_and_robot_mask_aware() -> None:
    accepted = runner._target_direct_update_gate(
        geometry_surface_accepted=True,
        target_rendered_observation_available=True,
        target_geometry_inlier=True,
        target_visible_fraction=0.70,
        target_robot_front_fraction=0.0,
        target_valid_depth_fraction=0.80,
    )
    occluded = runner._target_direct_update_gate(
        geometry_surface_accepted=True,
        target_rendered_observation_available=True,
        target_geometry_inlier=True,
        target_visible_fraction=0.70,
        target_robot_front_fraction=0.25,
        target_valid_depth_fraction=1.0,
    )
    low_support = runner._target_direct_update_gate(
        geometry_surface_accepted=True,
        target_rendered_observation_available=True,
        target_geometry_inlier=True,
        target_visible_fraction=0.34,
        target_robot_front_fraction=0.0,
        target_valid_depth_fraction=0.34,
    )
    bad_geometry = runner._target_direct_update_gate(
        geometry_surface_accepted=False,
        target_rendered_observation_available=True,
        target_geometry_inlier=True,
        target_visible_fraction=1.0,
        target_robot_front_fraction=0.0,
        target_valid_depth_fraction=1.0,
    )
    rendered_outlier = runner._target_direct_update_gate(
        geometry_surface_accepted=True,
        target_rendered_observation_available=True,
        target_geometry_inlier=False,
        target_visible_fraction=1.0,
        target_robot_front_fraction=0.0,
        target_valid_depth_fraction=1.0,
    )

    assert accepted["accepted"] is True
    assert accepted["fixed_visible_point_count_rule_used"] is False
    assert occluded["accepted"] is False
    assert "target_robot_front_occlusion" in occluded["failure_reasons"]
    assert low_support["accepted"] is False
    assert set(low_support["failure_reasons"]) == {
        "target_visible_support_below_threshold",
        "target_valid_depth_below_threshold",
    }
    assert bad_geometry["failure_reasons"] == ["geometry_surface_rejected"]
    assert rendered_outlier["accepted"] is False
    assert rendered_outlier["failure_reasons"] == [
        "target_rendered_geometry_outlier"
    ]


def test_target_direct_update_gate_threshold_boundaries_are_fail_closed() -> None:
    at_valid_boundaries = runner._target_direct_update_gate(
        geometry_surface_accepted=True,
        target_rendered_observation_available=True,
        target_geometry_inlier=True,
        target_visible_fraction=0.35,
        target_robot_front_fraction=np.nextafter(0.25, 0.0),
        target_valid_depth_fraction=0.35,
    )
    at_occlusion_boundary = runner._target_direct_update_gate(
        geometry_surface_accepted=True,
        target_rendered_observation_available=True,
        target_geometry_inlier=True,
        target_visible_fraction=1.0,
        target_robot_front_fraction=0.25,
        target_valid_depth_fraction=1.0,
    )

    assert at_valid_boundaries["accepted"] is True
    assert at_occlusion_boundary["accepted"] is False


def test_geometry_target_estimate_applies_audited_rigid_transform() -> None:
    transform = np.eye(4)
    transform[:3, :3] = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    transform[:3, 3] = [0.10, -0.20, 0.30]
    point = np.array([0.20, 0.40, 0.80])

    estimate = runner._geometry_target_estimate_camera_m(
        point,
        SimpleNamespace(transform=transform),
    )

    assert np.allclose(estimate, [-0.30, 0.0, 1.10])
    with pytest.raises(RuntimeError, match="geometry target estimate"):
        runner._geometry_target_estimate_camera_m(
            point,
            SimpleNamespace(transform=np.full((4, 4), np.nan)),
        )


def test_rendered_anchor_observations_use_unoccluded_human_depth() -> None:
    depth = np.full((7, 7), 2.0, dtype=np.float32)
    human = np.zeros((7, 7), dtype=bool)
    robot = np.zeros((7, 7), dtype=bool)
    human[2, 2] = True
    human[5, 4] = True
    robot[5, 5] = True

    observed = runner._rendered_anchor_observations(
        projected_pixels=np.array([[2.0, 2.0], [5.0, 5.0]]),
        depth_m=depth,
        human_mask=human,
        robot_mask=robot,
        intrinsics={"fx": 2.0, "fy": 4.0, "cx": 0.0, "cy": 0.0},
        support_radius_px=1,
        robot_mask_dilation_px=1,
    )

    assert np.array_equal(observed["available"], [True, False])
    assert np.allclose(observed["camera_points_m"][0], [2.0, 1.0, 2.0])
    assert np.array_equal(observed["sample_pixels"][0], [2.0, 2.0])
    assert observed["sample_distance_px"][0] == 0.0
    assert np.isnan(observed["camera_points_m"][1]).all()
    assert observed["source"] == "nearest_unoccluded_human_instance_depth_sample"


def _rendered_geometry_inlier_fixture() -> tuple[np.ndarray, list]:
    reference = np.array(
        [
            [-0.22, -0.30, 0.95],
            [0.20, -0.28, 1.00],
            [-0.18, -0.12, 1.05],
            [0.22, -0.10, 0.98],
            [-0.20, 0.05, 1.02],
            [0.21, 0.08, 1.06],
            [-0.17, 0.22, 0.97],
            [0.19, 0.24, 1.03],
            [-0.14, 0.34, 1.01],
            [0.16, 0.36, 1.04],
        ],
        dtype=np.float64,
    )
    visibility = [SimpleNamespace(visible_fraction=1.0) for _ in range(10)]
    return reference, visibility


def test_rendered_geometry_inlier_selection_rejects_one_gross_depth_outlier() -> None:
    reference, visibility = _rendered_geometry_inlier_fixture()
    observed = reference + np.array([0.002, -0.001, 0.003])
    observed[4] += np.array([0.0, 0.0, 0.080])

    selection = runner._rendered_geometry_inlier_selection(
        reference,
        visibility,
        observed,
        np.ones(10, dtype=bool),
    )

    assert set(selection) == {
        "initial_fit",
        "residual_m",
        "inlier_mask",
        "threshold_m",
    }
    assert selection["threshold_m"] == pytest.approx(0.006)
    assert np.asarray(selection["initial_fit"].transform).shape == (4, 4)
    assert np.isfinite(selection["initial_fit"].transform).all()
    assert selection["residual_m"].shape == (10,)
    assert selection["inlier_mask"].dtype == np.bool_
    assert selection["inlier_mask"].shape == (10,)
    assert selection["inlier_mask"].sum() == 9
    assert not selection["inlier_mask"][4]
    assert selection["residual_m"][4] > selection["threshold_m"]


def test_rendered_geometry_inlier_selection_threshold_is_inclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference, visibility = _rendered_geometry_inlier_fixture()
    reference[0, 0] = 0.0
    reference[1, 0] = 0.0
    observed = reference.copy()
    observed[0, 0] += 0.006
    observed[1, 0] += np.nextafter(0.006, 1.0)
    initial_fit = SimpleNamespace(transform=np.eye(4, dtype=np.float64))
    monkeypatch.setattr(
        runner,
        "_evaluate_geometry",
        lambda *args, **kwargs: initial_fit,
    )

    selection = runner._rendered_geometry_inlier_selection(
        reference,
        visibility,
        observed,
        np.ones(10, dtype=bool),
    )

    assert selection["initial_fit"] is initial_fit
    assert selection["residual_m"][0] == pytest.approx(0.006, abs=1e-15)
    assert selection["inlier_mask"][0]
    assert selection["residual_m"][1] > 0.006
    assert not selection["inlier_mask"][1]


def test_rendered_geometry_inlier_selection_marks_unavailable_observation_invalid() -> None:
    reference, visibility = _rendered_geometry_inlier_fixture()
    observed = reference + np.array([0.002, -0.001, 0.003])
    observed[-1] = np.nan
    valid = np.ones(10, dtype=bool)
    valid[-1] = False

    selection = runner._rendered_geometry_inlier_selection(
        reference,
        visibility,
        observed,
        valid,
    )

    assert np.isnan(selection["residual_m"][-1])
    assert not selection["inlier_mask"][-1]


@pytest.mark.parametrize(
    "mutation",
    ["reference_shape", "visibility_length", "valid_dtype", "valid_nan"],
)
def test_rendered_geometry_inlier_selection_rejects_invalid_inputs(
    mutation: str,
) -> None:
    reference, visibility = _rendered_geometry_inlier_fixture()
    observed = reference.copy()
    valid = np.ones(10, dtype=bool)
    if mutation == "reference_shape":
        reference = reference[:-1]
    elif mutation == "visibility_length":
        visibility = visibility[:-1]
    elif mutation == "valid_dtype":
        valid = valid.astype(np.int8)
    else:
        observed[0] = np.nan

    with pytest.raises(RuntimeError, match="rendered geometry inlier"):
        runner._rendered_geometry_inlier_selection(
            reference,
            visibility,
            observed,
            valid,
        )


def test_geometry_evaluation_uses_independent_rendered_observations() -> None:
    reference, visibility = _rendered_geometry_inlier_fixture()
    translation = np.array([0.006, -0.004, 0.003])
    observed = reference + translation

    result = runner._evaluate_geometry(
        reference,
        visibility,
        3,
        observed_camera_points=observed,
        observed_valid=np.ones(10, dtype=bool),
    )
    unavailable = runner._evaluate_geometry(
        reference,
        visibility,
        3,
        observed_camera_points=np.full((10, 3), np.nan),
        observed_valid=np.zeros(10, dtype=bool),
    )

    assert result.accepted is True
    assert np.allclose(result.transform[:3, 3], translation, atol=1e-8)
    assert np.allclose(
        runner._geometry_target_estimate_camera_m(reference[3], result),
        observed[3],
        atol=1e-8,
    )
    assert unavailable.accepted is False


@pytest.mark.parametrize(
    "kwargs",
    [
            {
                "geometry_surface_accepted": 1,
                "target_rendered_observation_available": True,
                "target_geometry_inlier": True,
            "target_visible_fraction": 1.0,
            "target_robot_front_fraction": 0.0,
            "target_valid_depth_fraction": 1.0,
        },
            {
                "geometry_surface_accepted": True,
                "target_rendered_observation_available": True,
                "target_geometry_inlier": True,
            "target_visible_fraction": float("nan"),
            "target_robot_front_fraction": 0.0,
            "target_valid_depth_fraction": 1.0,
        },
            {
                "geometry_surface_accepted": True,
                "target_rendered_observation_available": True,
                "target_geometry_inlier": True,
            "target_visible_fraction": 1.1,
            "target_robot_front_fraction": 0.0,
            "target_valid_depth_fraction": 1.0,
        },
        {
            "geometry_surface_accepted": True,
            "target_rendered_observation_available": False,
            "target_geometry_inlier": True,
            "target_visible_fraction": 1.0,
            "target_robot_front_fraction": 0.0,
            "target_valid_depth_fraction": 1.0,
        },
    ],
)
def test_target_direct_update_gate_rejects_invalid_evidence(kwargs: dict) -> None:
    with pytest.raises(RuntimeError, match="target direct-update"):
        runner._target_direct_update_gate(**kwargs)


def test_per_point_lock_policy_holds_last_3d_then_fails_closed_without_stopping_robot() -> None:
    visible_gate = runner._target_direct_update_gate(
        geometry_surface_accepted=True,
        target_rendered_observation_available=True,
        target_geometry_inlier=True,
        target_visible_fraction=0.70,
        target_robot_front_fraction=0.0,
        target_valid_depth_fraction=0.80,
    )
    occluded_gate = runner._target_direct_update_gate(
        geometry_surface_accepted=True,
        target_rendered_observation_available=False,
        target_geometry_inlier=False,
        target_visible_fraction=0.05,
        target_robot_front_fraction=0.75,
        target_valid_depth_fraction=1.0,
    )
    point = np.array([0.10, -0.02, 0.80])
    current = runner._advance_point_lock_policy(
        time_s=0.0,
        current_camera_point_m=point,
        current_target_error_p95_m=0.004,
        direct_gate=visible_gate,
        last_valid_time_s=None,
        last_valid_error_p95_m=None,
        last_valid_camera_point_m=None,
    )
    frozen = runner._advance_point_lock_policy(
        time_s=0.35,
        current_camera_point_m=np.full(3, np.nan),
        current_target_error_p95_m=float("inf"),
        direct_gate=occluded_gate,
        last_valid_time_s=current["next_last_valid_time_s"],
        last_valid_error_p95_m=current["next_last_valid_error_p95_m"],
        last_valid_camera_point_m=current["next_last_valid_camera_point_m"],
    )
    lost = runner._advance_point_lock_policy(
        time_s=0.70,
        current_camera_point_m=np.full(3, np.nan),
        current_target_error_p95_m=float("inf"),
        direct_gate=occluded_gate,
        last_valid_time_s=frozen["next_last_valid_time_s"],
        last_valid_error_p95_m=frozen["next_last_valid_error_p95_m"],
        last_valid_camera_point_m=frozen["next_last_valid_camera_point_m"],
    )

    assert current["policy_state"] == "DIRECT_GEOMETRY_UPDATE"
    assert current["estimate_source"] == "current_rendered_geometry"
    assert frozen["policy_state"] == "BOUNDED_PATIENT_FRAME_FREEZE"
    assert frozen["estimate_valid"] is True
    assert np.array_equal(frozen["estimate_camera_m"], point)
    assert frozen["estimate_source"] == "last_valid_patient_frame_lock"
    assert lost["policy_state"] == "LOST"
    assert lost["estimate_valid"] is False
    assert np.isnan(lost["estimate_camera_m"]).all()
    assert lost["estimate_source"] == "none_fail_closed"
    assert all(
        row["robot_motion_policy"] == "unchanged_by_point_visibility_policy"
        for row in (current, frozen, lost)
    )
    assert runner._ordered_policy_trace(
        [current["policy_state"], frozen["policy_state"], lost["policy_state"]]
    )


def test_per_point_lock_policy_without_history_is_immediately_lost() -> None:
    unavailable = runner._target_direct_update_gate(
        geometry_surface_accepted=False,
        target_rendered_observation_available=False,
        target_geometry_inlier=False,
        target_visible_fraction=0.0,
        target_robot_front_fraction=1.0,
        target_valid_depth_fraction=1.0,
    )
    result = runner._advance_point_lock_policy(
        time_s=0.0,
        current_camera_point_m=np.array([0.0, 0.0, 1.0]),
        current_target_error_p95_m=0.010,
        direct_gate=unavailable,
        last_valid_time_s=None,
        last_valid_error_p95_m=None,
        last_valid_camera_point_m=None,
    )

    assert result["policy_state"] == "LOST"
    assert result["estimate_valid"] is False
    assert result["history_available"] is False
    assert result["last_valid_age_s"] is None
    assert result["policy_reason"] == "no_last_valid_estimate"


def test_per_point_lock_policy_rejects_time_regression_on_direct_reacquisition() -> None:
    visible = runner._target_direct_update_gate(
        geometry_surface_accepted=True,
        target_rendered_observation_available=True,
        target_geometry_inlier=True,
        target_visible_fraction=0.8,
        target_robot_front_fraction=0.0,
        target_valid_depth_fraction=0.8,
    )
    with pytest.raises(RuntimeError, match="history|time"):
        runner._advance_point_lock_policy(
            time_s=0.5,
            current_camera_point_m=np.array([0.0, 0.0, 1.0]),
            current_target_error_p95_m=0.004,
            direct_gate=visible,
            last_valid_time_s=0.6,
            last_valid_error_p95_m=0.004,
            last_valid_camera_point_m=np.array([0.0, 0.0, 1.0]),
        )


def test_per_point_lock_policy_freeze_age_and_uncertainty_boundaries() -> None:
    unavailable = runner._target_direct_update_gate(
        geometry_surface_accepted=True,
        target_rendered_observation_available=False,
        target_geometry_inlier=False,
        target_visible_fraction=0.0,
        target_robot_front_fraction=1.0,
        target_valid_depth_fraction=1.0,
    )
    common = {
        "current_camera_point_m": np.full(3, np.nan),
        "current_target_error_p95_m": float("inf"),
        "direct_gate": unavailable,
        "last_valid_time_s": 0.0,
        "last_valid_camera_point_m": np.array([0.0, 0.0, 1.0]),
    }
    exact = runner._advance_point_lock_policy(
        time_s=0.40,
        last_valid_error_p95_m=0.006,
        **common,
    )
    too_old = runner._advance_point_lock_policy(
        time_s=np.nextafter(0.40, 1.0),
        last_valid_error_p95_m=0.006,
        **common,
    )
    too_uncertain = runner._advance_point_lock_policy(
        time_s=0.40,
        last_valid_error_p95_m=np.nextafter(0.006, 1.0),
        **common,
    )

    assert exact["policy_state"] == "BOUNDED_PATIENT_FRAME_FREEZE"
    assert np.isclose(exact["predicted_error_p95_m"], 0.010)
    assert too_old["policy_state"] == "LOST"
    assert too_uncertain["policy_state"] == "LOST"


def test_rep_camera_is_resolved_by_new_prim_identity_not_total_stage_camera_count() -> None:
    defaults = {
        "/OmniverseKit_Persp",
        "/OmniverseKit_Front",
        "/OmniverseKit_Top",
        "/OmniverseKit_Right",
    }
    after = defaults | {"/Replicator/Camera_Xform/Camera"}

    assert runner._unique_new_prim_path(defaults, after, label="camera") == (
        "/Replicator/Camera_Xform/Camera"
    )
    with pytest.raises(RuntimeError, match="one new camera"):
        runner._unique_new_prim_path(defaults, defaults, label="camera")
    with pytest.raises(RuntimeError, match="one new camera"):
        runner._unique_new_prim_path(
            defaults,
            after | {"/Replicator/Camera_Xform_01/Camera"},
            label="camera",
        )


def test_policy_trace_requires_one_ordered_direct_freeze_lost_episode() -> None:
    direct = "DIRECT_GEOMETRY_UPDATE"
    freeze = "BOUNDED_PATIENT_FRAME_FREEZE"
    lost = "LOST"

    assert runner._ordered_policy_trace([lost, direct, direct, freeze, freeze, lost])
    assert not runner._ordered_policy_trace([direct, lost, freeze])
    assert not runner._ordered_policy_trace([direct, freeze, direct, lost])
    assert not runner._ordered_policy_trace([direct, freeze])


def test_selected_policy_episode_binds_occlusion_and_states_to_same_target_wrist() -> None:
    def row(target, wrist, state, front):
        return {
            "target_name": target,
            "wrist_index": wrist,
            "policy_state": state,
            "robot_front_fraction_sum": front,
            "max_single_point_robot_front_fraction": front,
        }

    direct = "DIRECT_GEOMETRY_UPDATE"
    freeze = "BOUNDED_PATIENT_FRAME_FREEZE"
    lost = "LOST"
    same = [row("R3", 0, direct, 0.0), row("R3", 0, freeze, 0.31), row("R3", 0, lost, 0.30)]
    split_target = [row("R3", 0, direct, 0.31), row("L4", 0, freeze, 0.31), row("L4", 0, lost, 0.31)]
    split_wrist = [row("R3", 0, direct, 0.31), row("R3", 1, freeze, 0.31), row("R3", 1, lost, 0.31)]
    no_occlusion = [row("R3", 0, direct, 0.0), row("R3", 0, freeze, 0.0), row("R3", 0, lost, 0.0)]

    evidence = runner._targeted_policy_episode(same)
    assert evidence["observed"] is True
    assert evidence["target_name"] == "R3"
    assert evidence["wrist_index"] == 0
    assert evidence["frame_indices"] == [0, 1, 2]
    for rows in (split_target, split_wrist, no_occlusion):
        assert runner._targeted_policy_episode(rows)["observed"] is False


def test_selected_point_policy_episode_is_point_occlusion_specific_and_context_bound() -> None:
    def row(
        frame,
        point,
        trajectory_target,
        wrist,
        state,
        target_front,
        global_front,
        group=0,
    ):
        direct = state == "DIRECT_GEOMETRY_UPDATE"
        freeze = state == "BOUNDED_PATIENT_FRAME_FREEZE"
        estimate_valid = direct or freeze
        observation_available = direct
        age = 0.0 if direct else (0.35 if freeze else 0.70)
        failure_reasons = []
        if not observation_available:
            failure_reasons.append("target_rendered_depth_observation_unavailable")
        if not direct:
            failure_reasons.append("target_visible_support_below_threshold")
        if target_front >= 0.25:
            failure_reasons.append("target_robot_front_occlusion")
        return {
            "frame_index": frame,
            "joint_time_s": float(frame) * 0.35,
            "point_name": point,
            "point_index": experiment.POINT_NAMES.index(point),
            "trajectory_target_name": trajectory_target,
            "trajectory_target_index": experiment.POINT_NAMES.index(
                trajectory_target
            ),
            "wrist_index": wrist,
            "trajectory_group_index": group,
            "motion_phase": ("approach", "nominal_contact", "depart")[
                min(frame, 2)
            ],
            "policy_state": state,
            "policy_reason": (
                "current_rendered_support_passes_declared_geometry_surface"
                if direct
                else (
                    "short_hold_with_declared_motion_uncertainty_bound"
                    if freeze
                    else "freeze_age_exceeded+freeze_uncertainty_exceeded"
                )
            ),
            "predicted_error_p95_m": (
                0.004 if direct else (0.004 + 0.010 * age)
            ),
            "last_valid_age_s": age,
            "history_available": True,
            "estimate_valid": estimate_valid,
            "estimate_source": (
                "current_rendered_geometry"
                if direct
                else (
                    "last_valid_patient_frame_lock"
                    if freeze
                    else "none_fail_closed"
                )
            ),
            "estimate_camera_x_m": 0.10 if estimate_valid else None,
            "estimate_camera_y_m": -0.02 if estimate_valid else None,
            "estimate_camera_z_m": 0.80 if estimate_valid else None,
            "target_rendered_observation_available": observation_available,
            "target_geometry_inlier": direct,
            "direct_update_accepted": direct,
            "direct_update_failure_reasons": "+".join(failure_reasons),
            "geometry_surface_accepted": True,
            "geometry_failure_reasons": "",
            "geometry_target_error_p95_m": 0.004,
            "target_visible_fraction": 0.80 if direct else 0.05,
            "target_robot_front_fraction": target_front,
            "target_valid_depth_fraction": 1.0,
            "robot_front_fraction_sum": global_front,
            "max_single_point_robot_front_fraction": min(global_front, 1.0),
            "fixed_visible_point_count_rule_used": False,
            "robot_motion_policy": "unchanged_by_point_visibility_policy",
            "direct_estimate_source": (
                "rendered_depth_se3_transform_of_canonical_reference_point"
            ),
        }

    direct = "DIRECT_GEOMETRY_UPDATE"
    freeze = "BOUNDED_PATIENT_FRAME_FREEZE"
    lost = "LOST"
    rows = [
        row(0, "R2", "L1", 0, direct, 0.0, 0.8),
        row(0, "L2", "L1", 0, direct, 0.0, 0.8),
        row(1, "R2", "L1", 0, freeze, 0.75, 1.2),
        row(1, "L2", "L1", 0, freeze, 0.0, 1.2),
        row(2, "R2", "L1", 0, lost, 0.65, 1.0),
        row(2, "L2", "L1", 0, lost, 0.0, 1.0),
    ]

    evidence = runner._targeted_point_policy_episode(rows)

    assert evidence["observed"] is True
    assert evidence["point_name"] == "R2"
    assert evidence["trajectory_target_name"] == "L1"
    assert evidence["wrist_index"] == 0
    assert evidence["frame_indices"] == [0, 1, 2]
    only_other_point_occluded = [
        row(0, "L2", "L1", 0, direct, 0.0, 0.8),
        row(1, "L2", "L1", 0, freeze, 0.0, 1.2),
        row(2, "L2", "L1", 0, lost, 0.0, 1.0),
    ]
    split_context = [
        row(0, "R2", "L1", 0, direct, 0.0, 0.8),
        row(1, "R2", "L3", 0, freeze, 0.75, 1.2),
        row(2, "R2", "L3", 0, lost, 0.65, 1.0),
    ]
    split_group = [
        row(0, "R2", "L1", 0, direct, 0.0, 0.8, group=0),
        row(1, "R2", "L1", 0, freeze, 0.75, 1.2, group=1),
        row(2, "R2", "L1", 0, lost, 0.65, 1.0, group=1),
    ]
    discontinuous = [
        row(0, "R2", "L1", 0, direct, 0.0, 0.8),
        row(10, "R2", "L1", 0, freeze, 0.75, 1.2),
        row(11, "R2", "L1", 0, lost, 0.65, 1.0),
    ]
    assert runner._targeted_point_policy_episode(only_other_point_occluded)["observed"] is False
    assert runner._targeted_point_policy_episode(split_context)["observed"] is False
    assert runner._targeted_point_policy_episode(split_group)["observed"] is False
    assert runner._targeted_point_policy_episode(discontinuous)["observed"] is False

    contradictory = [dict(item) for item in rows]
    contradictory[2]["policy_state"] = "DIRECT_GEOMETRY_UPDATE"
    with pytest.raises(RuntimeError, match="state semantics"):
        runner._targeted_point_policy_episode(contradictory)


_EXPECTED_POINT_POLICY_CSV_FIELDS = (
    "frame_index",
    "joint_time_s",
    "point_name",
    "point_index",
    "trajectory_target_name",
    "trajectory_target_index",
    "trajectory_group_index",
    "wrist_index",
    "motion_phase",
    "policy_state",
    "policy_reason",
    "predicted_error_p95_m",
    "last_valid_age_s",
    "history_available",
    "estimate_valid",
    "estimate_source",
    "estimate_camera_x_m",
    "estimate_camera_y_m",
    "estimate_camera_z_m",
    "target_rendered_observation_available",
    "target_geometry_inlier",
    "direct_update_accepted",
    "direct_update_failure_reasons",
    "geometry_surface_accepted",
    "geometry_failure_reasons",
    "geometry_target_error_p95_m",
    "target_visible_fraction",
    "target_robot_front_fraction",
    "target_valid_depth_fraction",
    "robot_front_fraction_sum",
    "max_single_point_robot_front_fraction",
    "fixed_visible_point_count_rule_used",
    "robot_motion_policy",
    "direct_estimate_source",
)


def _canonical_per_point_policy_rows() -> list[dict]:
    rows = []
    frame_times = (0.0, 0.35, 0.70)
    phases = ("approach", "nominal_contact", "depart")
    target_front_by_frame = (0.0, 0.75, 0.65)
    for frame_index, (time_s, phase, selected_front) in enumerate(
        zip(frame_times, phases, target_front_by_frame)
    ):
        front_by_point = [0.0] * len(experiment.POINT_NAMES)
        front_by_point[0] = selected_front
        front_sum = float(sum(front_by_point))
        front_max = float(max(front_by_point))
        for point_index, point_name in enumerate(experiment.POINT_NAMES):
            if point_index == 0 and frame_index == 1:
                state = "BOUNDED_PATIENT_FRAME_FREEZE"
            elif point_index == 0 and frame_index == 2:
                state = "LOST"
            else:
                state = "DIRECT_GEOMETRY_UPDATE"
            direct = state == "DIRECT_GEOMETRY_UPDATE"
            freeze = state == "BOUNDED_PATIENT_FRAME_FREEZE"
            estimate_valid = direct or freeze
            age = 0.0 if direct else time_s
            point_front = front_by_point[point_index]
            observation_available = direct
            visible_fraction = 0.80 if direct else 0.05
            failure_reasons = []
            if not observation_available:
                failure_reasons.append(
                    "target_rendered_depth_observation_unavailable"
                )
            if visible_fraction < 0.35:
                failure_reasons.append("target_visible_support_below_threshold")
            if point_front >= 0.25:
                failure_reasons.append("target_robot_front_occlusion")
            base_estimate = np.asarray(
                [0.10 + 0.01 * point_index, -0.02, 0.80], dtype=np.float64
            )
            if direct and point_index != 0:
                base_estimate = base_estimate + np.asarray(
                    [0.001 * frame_index, 0.0, 0.0], dtype=np.float64
                )
            rows.append(
                {
                    "frame_index": frame_index,
                    "joint_time_s": time_s,
                    "point_name": point_name,
                    "point_index": point_index,
                    "trajectory_target_name": "R1",
                    "trajectory_target_index": 0,
                    "trajectory_group_index": 0,
                    "wrist_index": 0,
                    "motion_phase": phase,
                    "policy_state": state,
                    "policy_reason": (
                        "current_rendered_support_passes_declared_geometry_surface"
                        if direct
                        else (
                            "short_hold_with_declared_motion_uncertainty_bound"
                            if freeze
                            else "freeze_age_exceeded+freeze_uncertainty_exceeded"
                        )
                    ),
                    "predicted_error_p95_m": (
                        0.004 if direct else 0.004 + 0.010 * age
                    ),
                    "last_valid_age_s": age,
                    "history_available": True,
                    "estimate_valid": estimate_valid,
                    "estimate_source": (
                        "current_rendered_geometry"
                        if direct
                        else (
                            "last_valid_patient_frame_lock"
                            if freeze
                            else "none_fail_closed"
                        )
                    ),
                    "estimate_camera_x_m": (
                        float(base_estimate[0]) if estimate_valid else None
                    ),
                    "estimate_camera_y_m": (
                        float(base_estimate[1]) if estimate_valid else None
                    ),
                    "estimate_camera_z_m": (
                        float(base_estimate[2]) if estimate_valid else None
                    ),
                    "target_rendered_observation_available": (
                        observation_available
                    ),
                    "target_geometry_inlier": direct,
                    "direct_update_accepted": direct,
                    "direct_update_failure_reasons": "+".join(failure_reasons),
                    "geometry_surface_accepted": True,
                    "geometry_failure_reasons": "",
                    "geometry_target_error_p95_m": 0.004,
                    "target_visible_fraction": visible_fraction,
                    "target_robot_front_fraction": point_front,
                    "target_valid_depth_fraction": 1.0,
                    "robot_front_fraction_sum": front_sum,
                    "max_single_point_robot_front_fraction": front_max,
                    "fixed_visible_point_count_rule_used": False,
                    "robot_motion_policy": (
                        "unchanged_by_point_visibility_policy"
                    ),
                    "direct_estimate_source": (
                        "rendered_depth_se3_transform_of_canonical_reference_point"
                    ),
                }
            )
    return rows


def _persist_canonical_point_policy(tmp_path: Path) -> tuple[Path, Path, dict]:
    csv_path = tmp_path / "per_point_policy.csv"
    npz_path = tmp_path / "per_point_policy.npz"
    report = runner._persist_and_verify_per_point_policy(
        _canonical_per_point_policy_rows(),
        csv_path=csv_path,
        npz_path=npz_path,
        expected_point_names=experiment.POINT_NAMES,
    )
    return csv_path, npz_path, report


def _rewrite_csv_cell(
    path: Path,
    *,
    row_index: int,
    field_name: str,
    value: str,
) -> None:
    with path.open("r", newline="", encoding="utf-8") as handle:
        table = list(csv.reader(handle))
    column_index = table[0].index(field_name)
    table[row_index + 1][column_index] = value
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows(table)


def _rewrite_policy_npz(path: Path, mutation) -> None:
    with np.load(path, allow_pickle=False) as persisted:
        arrays = {name: persisted[name].copy() for name in persisted.files}
    mutation(arrays)
    np.savez_compressed(path, **arrays)


def _verify_policy_artifacts(csv_path: Path, npz_path: Path) -> dict:
    return runner._verify_persisted_per_point_policy(
        csv_path=csv_path,
        npz_path=npz_path,
        expected_point_names=experiment.POINT_NAMES,
    )


def test_per_point_policy_artifacts_have_exact_roundtrip_schema_and_evidence(
    tmp_path: Path,
) -> None:
    csv_path, npz_path, report = _persist_canonical_point_policy(tmp_path)

    required_checks = {
        "exact_csv_schema",
        "exact_npz_schema",
        "csv_npz_alignment",
        "frame_point_cartesian_complete",
        "mask_nan_contract",
        "history_state_contract",
        "direct_gate_recomputed",
        "robot_motion_unchanged",
        "state_counts_complete",
    }
    assert required_checks <= set(report["checks"])
    assert all(report["checks"][name] is True for name in required_checks)
    assert report["point_frame_state_counts"] == {
        "DIRECT_GEOMETRY_UPDATE": 28,
        "BOUNDED_PATIENT_FRAME_FREEZE": 1,
        "LOST": 1,
    }
    assert report["frames_with_point_state"] == {
        "DIRECT_GEOMETRY_UPDATE": 3,
        "BOUNDED_PATIENT_FRAME_FREEZE": 1,
        "LOST": 1,
    }
    assert report["selected_episode"] == {
        "observed": True,
        "minimum_fraction": 0.25,
        "point_name": "R1",
        "trajectory_target_name": "R1",
        "wrist_index": 0,
        "trajectory_group_index": 0,
        "frame_indices": [0, 1, 2],
        "state_trace": [
            "DIRECT_GEOMETRY_UPDATE",
            "BOUNDED_PATIENT_FRAME_FREEZE",
            "LOST",
        ],
        "target_robot_front_fraction": [0.0, 0.75, 0.65],
    }

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        csv_table = list(csv.reader(handle))
    assert tuple(csv_table[0]) == _EXPECTED_POINT_POLICY_CSV_FIELDS
    assert len(csv_table) == 1 + 3 * len(experiment.POINT_NAMES)
    bool_columns = [
        _EXPECTED_POINT_POLICY_CSV_FIELDS.index(name)
        for name in (
            "history_available",
            "estimate_valid",
            "target_rendered_observation_available",
            "target_geometry_inlier",
            "direct_update_accepted",
            "geometry_surface_accepted",
            "fixed_visible_point_count_rule_used",
        )
    ]
    assert {
        row[column]
        for row in csv_table[1:]
        for column in bool_columns
    } == {"true", "false"}

    point_count = len(experiment.POINT_NAMES)
    expected_npz = {
        "schema": (np.dtype("<U33"), ()),
        "point_names": (np.dtype("<U2"), (point_count,)),
        "frame_index": (np.dtype("<i8"), (3,)),
        "joint_time_s": (np.dtype("<f8"), (3,)),
        "trajectory_target_name": (np.dtype("<U2"), (3,)),
        "wrist_index": (np.dtype("<i8"), (3,)),
        "trajectory_group_index": (np.dtype("<i8"), (3,)),
        "policy_state": (np.dtype("<U28"), (3, point_count)),
        "estimate_valid": (np.dtype(np.bool_), (3, point_count)),
        "history_valid": (np.dtype(np.bool_), (3, point_count)),
        "estimate_camera_m": (np.dtype("<f8"), (3, point_count, 3)),
        "predicted_error_p95_m": (np.dtype("<f8"), (3, point_count)),
        "last_valid_age_s": (np.dtype("<f8"), (3, point_count)),
        "target_rendered_observation_available": (
            np.dtype(np.bool_),
            (3, point_count),
        ),
        "target_geometry_inlier": (
            np.dtype(np.bool_),
            (3, point_count),
        ),
        "direct_update_accepted": (np.dtype(np.bool_), (3, point_count)),
        "target_robot_front_fraction": (
            np.dtype("<f8"),
            (3, point_count),
        ),
        "robot_motion_unchanged": (np.dtype(np.bool_), (3,)),
    }
    with np.load(npz_path, allow_pickle=False) as persisted:
        assert set(persisted.files) == set(expected_npz)
        assert persisted["schema"].item() == "wam_per_point_occlusion_policy_v3"
        for name, (dtype, shape) in expected_npz.items():
            assert persisted[name].dtype == dtype, name
            assert persisted[name].shape == shape, name
        assert np.isfinite(persisted["estimate_camera_m"][:2, 0]).all()
        assert np.isnan(persisted["estimate_camera_m"][2, 0]).all()

    reloaded = _verify_policy_artifacts(csv_path, npz_path)
    assert reloaded["checks"] == report["checks"]
    assert reloaded["point_frame_state_counts"] == report[
        "point_frame_state_counts"
    ]
    assert reloaded["selected_episode"] == report["selected_episode"]


def test_per_point_policy_artifacts_preserve_exact_no_history_lost_prediction(
    tmp_path: Path,
) -> None:
    rows = _canonical_per_point_policy_rows()
    for frame_index in range(3):
        row = rows[frame_index * len(experiment.POINT_NAMES)]
        row.update(
            {
                "policy_state": "LOST",
                "policy_reason": "no_last_valid_estimate",
                "predicted_error_p95_m": 0.010
                + 0.010 * (0.40 + 0.01),
                "last_valid_age_s": None,
                "history_available": False,
                "estimate_valid": False,
                "estimate_source": "none_fail_closed",
                "estimate_camera_x_m": None,
                "estimate_camera_y_m": None,
                "estimate_camera_z_m": None,
                "target_rendered_observation_available": False,
                "target_geometry_inlier": False,
                "direct_update_accepted": False,
                "direct_update_failure_reasons": (
                    "target_rendered_depth_observation_unavailable+"
                    "target_visible_support_below_threshold"
                    + (
                        "+target_robot_front_occlusion"
                        if row["target_robot_front_fraction"] >= 0.25
                        else ""
                    )
                ),
                "target_visible_fraction": 0.0,
            }
        )

    report = runner._persist_and_verify_per_point_policy(
        rows,
        csv_path=tmp_path / "per_point_policy.csv",
        npz_path=tmp_path / "per_point_policy.npz",
        expected_point_names=experiment.POINT_NAMES,
    )

    assert report["point_frame_state_counts"]["LOST"] == 3
    assert report["selected_episode"]["observed"] is False


@pytest.mark.parametrize("mutation", ["header", "bool"])
def test_per_point_policy_verifier_rejects_noncanonical_csv(
    tmp_path: Path,
    mutation: str,
) -> None:
    csv_path, npz_path, _report = _persist_canonical_point_policy(tmp_path)
    if mutation == "header":
        text = csv_path.read_text(encoding="utf-8")
        csv_path.write_text(
            text.replace("frame_index,", "frame_id,", 1), encoding="utf-8"
        )
    else:
        _rewrite_csv_cell(
            csv_path,
            row_index=0,
            field_name="history_available",
            value="TRUE",
        )

    with pytest.raises(RuntimeError):
        _verify_policy_artifacts(csv_path, npz_path)


@pytest.mark.parametrize("mutation", ["extra_key", "wrong_dtype"])
def test_per_point_policy_verifier_rejects_noncanonical_npz(
    tmp_path: Path,
    mutation: str,
) -> None:
    csv_path, npz_path, _report = _persist_canonical_point_policy(tmp_path)

    def mutate(arrays: dict[str, np.ndarray]) -> None:
        if mutation == "extra_key":
            arrays["unexpected"] = np.asarray(1, dtype=np.int64)
        else:
            arrays["frame_index"] = arrays["frame_index"].astype(np.int32)

    _rewrite_policy_npz(npz_path, mutate)
    with pytest.raises(RuntimeError):
        _verify_policy_artifacts(csv_path, npz_path)


def test_per_point_policy_verifier_rejects_partial_nan_under_valid_mask(
    tmp_path: Path,
) -> None:
    csv_path, npz_path, _report = _persist_canonical_point_policy(tmp_path)
    _rewrite_csv_cell(
        csv_path,
        row_index=0,
        field_name="estimate_camera_x_m",
        value="",
    )

    def insert_partial_nan(arrays: dict[str, np.ndarray]) -> None:
        arrays["estimate_camera_m"][0, 0, 0] = np.nan

    _rewrite_policy_npz(npz_path, insert_partial_nan)
    with pytest.raises(RuntimeError):
        _verify_policy_artifacts(csv_path, npz_path)


def test_per_point_policy_verifier_rejects_history_state_contradiction(
    tmp_path: Path,
) -> None:
    csv_path, npz_path, _report = _persist_canonical_point_policy(tmp_path)
    _rewrite_csv_cell(
        csv_path,
        row_index=0,
        field_name="history_available",
        value="false",
    )

    def remove_direct_history(arrays: dict[str, np.ndarray]) -> None:
        arrays["history_valid"][0, 0] = False

    _rewrite_policy_npz(npz_path, remove_direct_history)
    with pytest.raises(RuntimeError):
        _verify_policy_artifacts(csv_path, npz_path)


def test_per_point_policy_verifier_rejects_direct_update_for_rendered_outlier(
    tmp_path: Path,
) -> None:
    csv_path, npz_path, _report = _persist_canonical_point_policy(tmp_path)
    _rewrite_csv_cell(
        csv_path,
        row_index=0,
        field_name="target_geometry_inlier",
        value="false",
    )

    def mark_direct_target_outlier(arrays: dict[str, np.ndarray]) -> None:
        arrays["target_geometry_inlier"][0, 0] = False

    _rewrite_policy_npz(npz_path, mark_direct_target_outlier)
    with pytest.raises(RuntimeError):
        _verify_policy_artifacts(csv_path, npz_path)


def _write_geometry_inlier_frame_product(path: Path) -> list[dict]:
    point_count = len(experiment.POINT_NAMES)
    rows = _canonical_per_point_policy_rows()[:point_count]
    np.savez_compressed(
        path,
        rendered_observation_available=np.ones(point_count, dtype=bool),
        rendered_geometry_residual_m=np.zeros(point_count, dtype=np.float64),
        rendered_geometry_inlier_mask=np.ones(point_count, dtype=bool),
        rendered_geometry_inlier_threshold_m=np.asarray(0.006, dtype=np.float64),
        rendered_geometry_initial_transform=np.eye(4, dtype=np.float64),
        rendered_geometry_initial_fit_recoverable=np.asarray(True, dtype=bool),
        visible_fraction=np.ones(point_count, dtype=np.float64),
        point_geometry_transform=np.repeat(
            np.eye(4, dtype=np.float64)[None, :, :], point_count, axis=0
        ),
        point_geometry_target_error_p95_m=np.full(
            point_count, 0.004, dtype=np.float64
        ),
        point_geometry_surface_accepted=np.ones(point_count, dtype=bool),
        point_geometry_failure_reasons=np.full(
            point_count, "", dtype="<U160"
        ),
        point_target_geometry_inlier=np.ones(point_count, dtype=bool),
        point_direct_update_accepted=np.ones(point_count, dtype=bool),
        frame_index=np.asarray(0, dtype=np.int64),
    )
    return rows


def test_geometry_inlier_frame_products_are_reopened_and_bound_to_policy(
    tmp_path: Path,
) -> None:
    root = tmp_path / "frame_products"
    root.mkdir()
    rows = _write_geometry_inlier_frame_product(root / "frame_00000.npz")

    report = runner._verify_rendered_geometry_inlier_frame_products(
        root, rows, frame_count=1
    )

    assert report["verified_frame_count"] == 1
    assert report["verified_point_frame_count"] == 10
    assert report["inlier_count_min"] == 10
    assert report["raw_observation_mask_bound"] is True
    assert report["policy_rows_bound"] is True


def test_geometry_inlier_frame_products_reject_unavailable_zero_residual(
    tmp_path: Path,
) -> None:
    root = tmp_path / "frame_products"
    root.mkdir()
    path = root / "frame_00000.npz"
    rows = _write_geometry_inlier_frame_product(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays["rendered_observation_available"][0] = False
    arrays["rendered_geometry_inlier_mask"][0] = False
    arrays["point_target_geometry_inlier"][0] = False
    rows[0]["target_geometry_inlier"] = False
    np.savez_compressed(path, **arrays)

    with pytest.raises(RuntimeError, match="frame-product masks"):
        runner._verify_rendered_geometry_inlier_frame_products(
            root, rows, frame_count=1
        )
