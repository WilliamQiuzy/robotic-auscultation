from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import h200_env as env  # noqa: E402
import artifact_contract as contract  # noqa: E402


REAL_RUNTIME_PROBE = env.probe_runtime


PYTHON_REPORT = {
    "abi": "cp312",
    "implementation": "CPython",
    "machine": "x86_64",
    "platform": "linux",
    "version": "3.12.13",
}
SYSTEM_REPORT = {
    "cuda_device_name": "NVIDIA H200 NVL",
    "glibc_version": "2.35",
    "libraries": [
        {
            "device": 2049,
            "inode": 424242,
            "name": "libcuda.so.1",
            "path": "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
            "sha256": "a" * 64,
        }
    ],
    "nvidia_driver_version": "570.124.06",
    "schema": "wam_h200_system_libraries_v1",
}
CORE_VERSIONS = {
    "lerobot": "0.3.3",
    "torch": "2.7.1",
    "torchaudio": "2.7.1",
    "torchvision": "0.22.1",
}
TRACKED_DIRECT_PACKAGES = {
    "joblib",
    "lerobot",
    "librosa",
    "matplotlib",
    "numpy",
    "opencv-python-headless",
    "pandas",
    "pillow",
    "pyarrow",
    "scikit-learn",
    "scipy",
    "smplx",
    "soundfile",
    "torch",
    "torchaudio",
    "torchvision",
    "transformers",
    "trimesh",
}


@pytest.fixture(autouse=True)
def _use_exact_fixture_runtime_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    fixture_base = tmp_path / "fixture-base" / "python3.12"
    fixture_base.parent.mkdir()
    os.link(Path(sys.executable).resolve(), fixture_base)
    monkeypatch.setattr(
        env, "resolve_base_python", lambda _value: str(fixture_base), raising=False
    )
    monkeypatch.setattr(
        env,
        "open_base_python",
        lambda _path: os.open(fixture_base, os.O_RDONLY),
        raising=False,
    )
    monkeypatch.setattr(
        env,
        "probe_runtime",
        lambda _python, *, pass_fds=(): (
            dict(PYTHON_REPORT),
            json.loads(json.dumps(SYSTEM_REPORT)),
        ),
        raising=False,
    )


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _wheel_filename(name: str, version: str) -> str:
    normalized = name.replace("-", "_")
    if name in {"torch", "torchaudio", "torchvision"}:
        return f"{normalized}-{version}+cu128-cp312-cp312-manylinux_2_28_x86_64.whl"
    return f"{normalized}-{version}-py3-none-any.whl"


def _write_lock_and_wheels(
    tmp_path: Path,
    packages: dict[str, str] | None = None,
) -> tuple[Path, Path]:
    packages = packages or CORE_VERSIONS
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    blocks = [
        "--index-url https://pypi.org/simple",
        "--extra-index-url https://download.pytorch.org/whl/cu128",
        "",
    ]
    for name, version in sorted(packages.items()):
        filename = _wheel_filename(name, version)
        data = ("wheel:" + filename).encode("utf-8")
        (wheelhouse / filename).write_bytes(data)
        blocks.extend(
            [
                f"{name}=={version} \\",
                f"    --hash=sha256:{_sha(data)}",
            ]
        )
    lock_path = tmp_path / "requirements-h200.lock.txt"
    lock_path.write_text("\n".join(blocks) + "\n", encoding="utf-8")
    return lock_path, wheelhouse


def _identity(tmp_path: Path, *, system_report: dict | None = None) -> tuple[dict, Path, Path]:
    lock_path, wheelhouse = _write_lock_and_wheels(tmp_path)
    identity = env.build_environment_identity(
        lock_path,
        wheelhouse,
        python_report=PYTHON_REPORT,
        system_library_report=system_report or SYSTEM_REPORT,
    )
    return identity, lock_path, wheelhouse


def _smoke(identity: dict, *, ok: bool = True) -> dict:
    checks = {
        "annotation_ui": ok,
        "audio": ok,
        "cuda_h200": ok,
        "dataset_reader": ok,
        "gpu_tensor": ok,
        "perception": ok,
        "reward_model": ok,
    }
    return {
        "checks": checks,
        "environment_id": identity["environment_id"],
        "packages": [
            {"name": row["name"], "version": row["version"]}
            for row in identity["payload"]["wheels"]
        ],
        "runtime": {
            "cuda_device_name": identity["payload"]["cuda"]["device_name"],
            "python": dict(identity["payload"]["python"]),
            "system_libraries": json.loads(
                json.dumps(identity["payload"]["system_libraries"])
            ),
            "torch_cuda": identity["payload"]["cuda"]["runtime"],
        },
        "schema": env.SMOKE_SCHEMA,
        "versions": CORE_VERSIONS,
    }


def _materialize_fake_venv(argv: list[str]) -> None:
    root = Path(argv[-1])
    launcher = Path(argv[0])
    python = root / "bin" / "python"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o755)
    python312 = root / "bin" / "python3.12"
    python312.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python312.chmod(0o755)
    (root / "pyvenv.cfg").write_text(
        "home = %s\n"
        "include-system-site-packages = false\n"
        "version = 3.12.13\n"
        "executable = %s\n"
        "command = %s -m venv --copies --without-pip %s\n"
        % (launcher.parent, launcher, launcher, root),
        encoding="utf-8",
    )


def _materialize_fake_ensurepip(argv: list[str]) -> None:
    pip = Path(argv[0]).parent / "pip"
    pip.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    pip.chmod(0o755)


def _runner(calls: list[list[str]]):
    def run(
        argv: list[str],
        *,
        process_env: dict[str, str] | None = None,
        pass_fds: tuple[int, ...] = (),
    ) -> None:
        del process_env
        del pass_fds
        calls.append(list(argv))
        if argv[1:5] == ["-I", "-S", "-m", "venv"]:
            _materialize_fake_venv(argv)
        elif argv[1:] == [
            "-I", "-m", "ensurepip", "--upgrade", "--default-pip",
        ]:
            _materialize_fake_ensurepip(argv)

    return run


def test_hash_lock_accepts_uv_style_continuations_and_binds_raw_bytes(tmp_path: Path) -> None:
    lock_path, _wheelhouse = _write_lock_and_wheels(tmp_path)

    parsed = env.parse_hash_lock(lock_path)

    assert parsed.lock_sha256 == _sha(lock_path.read_bytes())
    assert parsed.indexes == (
        "https://pypi.org/simple",
        "https://download.pytorch.org/whl/cu128",
    )
    assert {row.name: row.version for row in parsed.requirements} == CORE_VERSIONS
    assert all(row.hashes and all(len(value) == 64 for value in row.hashes) for row in parsed.requirements)


@pytest.mark.parametrize(
    "body, message",
    [
        ("demo>=1.0 --hash=sha256:" + "a" * 64 + "\n", "exactly pinned"),
        ("demo==1.0\n", "sha256"),
        ("demo==1.0 --hash=md5:" + "a" * 32 + "\n", "sha256"),
        ("demo @ https://example.test/demo.whl --hash=sha256:" + "a" * 64 + "\n", "URL"),
        ("-e git+https://example.test/repo.git#egg=demo\n", "editable"),
        ("demo==1.0 ; python_version >= '3.12' --hash=sha256:" + "a" * 64 + "\n", "marker"),
        (
            "demo==1.0 --hash=sha256:" + "a" * 64 + "\n"
            "Demo==1.0 --hash=sha256:" + "b" * 64 + "\n",
            "duplicate",
        ),
        ("--find-links https://example.test/wheels\n", "directive"),
        ("openpi==0.1 --hash=sha256:" + "a" * 64 + "\n", "openpi"),
    ],
)
def test_hash_lock_rejects_unsafe_or_ambiguous_input(
    tmp_path: Path, body: str, message: str
) -> None:
    lock_path = tmp_path / "bad.lock"
    lock_path.write_text(body, encoding="utf-8")

    with pytest.raises(env.LockValidationError, match=message):
        env.parse_hash_lock(lock_path)


def test_hash_lock_rejects_unapproved_or_duplicate_indexes(tmp_path: Path) -> None:
    for body in (
        "--index-url https://mirror.example.test/simple\n",
        "--index-url https://pypi.org/simple\n--index-url https://pypi.org/simple\n",
    ):
        lock_path = tmp_path / (hashlib.sha256(body.encode()).hexdigest() + ".lock")
        lock_path.write_text(body, encoding="utf-8")
        with pytest.raises(env.LockValidationError, match="index"):
            env.parse_hash_lock(lock_path)


def test_identity_binds_platform_lock_indexes_wheels_and_system_report(tmp_path: Path) -> None:
    identity, lock_path, wheelhouse = _identity(tmp_path)
    payload = identity["payload"]

    assert identity == {
        "environment_id": "wam-" + identity["payload_sha256"][:20],
        "payload": payload,
        "payload_sha256": identity["payload_sha256"],
        "schema": env.IDENTITY_ENVELOPE_SCHEMA,
    }
    assert identity["payload_sha256"] == _sha(env.canonical_json_bytes(payload))
    assert payload["lock"] == {
        "sha256": _sha(lock_path.read_bytes()),
        "size": len(lock_path.read_bytes()),
    }
    assert payload["python"] == PYTHON_REPORT
    assert payload["platform"] == {"machine": "x86_64", "system": "linux"}
    assert payload["cuda"] == {
        "device_name": "NVIDIA H200 NVL",
        "runtime": "12.8",
        "torch_build": "cu128",
    }
    assert payload["pytorch"] == {
        "torch": "2.7.1",
        "torchaudio": "2.7.1",
        "torchvision": "0.22.1",
    }
    assert payload["indexes"] == [
        "https://pypi.org/simple",
        "https://download.pytorch.org/whl/cu128",
    ]
    assert payload["system_libraries"] == SYSTEM_REPORT
    assert [row["filename"] for row in payload["wheels"]] == sorted(
        path.name for path in wheelhouse.glob("*.whl")
    )


def test_identity_changes_when_any_bound_input_changes(tmp_path: Path) -> None:
    identity, lock_path, wheelhouse = _identity(tmp_path)
    changed_report = json.loads(json.dumps(SYSTEM_REPORT))
    changed_report["nvidia_driver_version"] = "571.0"
    changed = env.build_environment_identity(
        lock_path,
        wheelhouse,
        python_report=PYTHON_REPORT,
        system_library_report=changed_report,
    )

    assert changed["environment_id"] != identity["environment_id"]


@pytest.mark.parametrize(
    "field, value",
    [
        ("implementation", "PyPy"),
        ("version", "3.11.9"),
        ("abi", "cp311"),
        ("platform", "darwin"),
        ("machine", "arm64"),
        ("version", "3.12.12"),
    ],
)
def test_identity_rejects_non_h200_python_target(tmp_path: Path, field: str, value: str) -> None:
    lock_path, wheelhouse = _write_lock_and_wheels(tmp_path)
    report = dict(PYTHON_REPORT)
    report[field] = value

    with pytest.raises(env.IdentityValidationError, match=field):
        env.build_environment_identity(
            lock_path,
            wheelhouse,
            python_report=report,
            system_library_report=SYSTEM_REPORT,
        )


@pytest.mark.parametrize(
    "forged, message",
    [("python", "version"), ("system", "probed")],
)
def test_build_rejects_caller_report_that_differs_from_live_probe(
    tmp_path: Path, forged: str, message: str
) -> None:
    lock_path, wheelhouse = _write_lock_and_wheels(tmp_path)
    python_report = dict(PYTHON_REPORT)
    system_report = json.loads(json.dumps(SYSTEM_REPORT))
    if forged == "python":
        python_report["version"] = "3.12.99"
    else:
        system_report["nvidia_driver_version"] = "999.0"

    with pytest.raises(env.IdentityValidationError, match=message):
        env.build_environment(
            lock_path,
            wheelhouse,
            releases_root=tmp_path / "envs" / "releases",
            python_report=python_report,
            system_library_report=system_report,
            command_runner=_runner([]),
            smoke_runner=lambda _path, identity: _smoke(identity),
        )


def test_build_probes_and_invokes_the_same_resolved_base_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path, wheelhouse = _write_lock_and_wheels(tmp_path)
    calls: list[list[str]] = []
    probed: list[str] = []
    executables: list[str | None] = []
    verified = tmp_path / "verified" / "bin" / "python3.12"
    verified.parent.mkdir(parents=True)
    os.link(Path(sys.executable).resolve(), verified)
    monkeypatch.setattr(
        env, "resolve_base_python", lambda _value: str(verified)
    )
    monkeypatch.setattr(
        env, "open_base_python", lambda path: os.open(path, os.O_RDONLY)
    )

    def runtime_probe(executable: str, *, pass_fds: tuple[int, ...] = ()):
        probed.append(executable)
        return dict(PYTHON_REPORT), json.loads(json.dumps(SYSTEM_REPORT))

    monkeypatch.setattr(env, "probe_runtime", runtime_probe)

    base_runner = _runner(calls)

    def runner(
        argv: list[str],
        *,
        process_env: dict[str, str] | None = None,
        pass_fds: tuple[int, ...] = (),
    ) -> None:
        executables.append(getattr(argv, "executable", None))
        base_runner(argv, process_env=process_env, pass_fds=pass_fds)

    env.build_environment(
        lock_path,
        wheelhouse,
        releases_root=tmp_path / "envs" / "releases",
        python_report=PYTHON_REPORT,
        system_library_report=SYSTEM_REPORT,
        command_runner=runner,
        smoke_runner=lambda _path, identity: _smoke(identity),
    )

    assert Path(calls[0][0]).name == "python3.12"
    assert executables[0] == (
        probed[0] if sys.platform.startswith("linux") else calls[0][0]
    )
    assert executables[1] is None


def test_build_rejects_base_python_path_swap_after_the_held_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path, wheelhouse = _write_lock_and_wheels(tmp_path)
    base_python = tmp_path / "python3.12"
    base_python.write_text("original interpreter\n", encoding="utf-8")
    base_python.chmod(0o755)
    original = base_python.stat()
    captured: dict[str, object] = {}

    monkeypatch.setattr(env, "resolve_base_python", lambda _value: str(base_python))
    monkeypatch.setattr(
        env, "open_base_python", lambda path: os.open(path, os.O_RDONLY)
    )

    def runtime_probe(executable: str, *, pass_fds: tuple[int, ...] = ()):
        assert len(pass_fds) == 1
        descriptor = pass_fds[0]
        info = os.fstat(descriptor)
        assert (info.st_dev, info.st_ino) == (original.st_dev, original.st_ino)
        captured["executable"] = executable
        captured["descriptor"] = descriptor
        base_python.rename(tmp_path / "original-moved")
        base_python.write_text("attacker replacement\n", encoding="utf-8")
        base_python.chmod(0o755)
        return dict(PYTHON_REPORT), json.loads(json.dumps(SYSTEM_REPORT))

    monkeypatch.setattr(env, "probe_runtime", runtime_probe)
    calls: list[list[str]] = []

    def runner(
        argv: list[str],
        *,
        process_env: dict[str, str] | None = None,
        pass_fds: tuple[int, ...] = (),
    ) -> None:
        del process_env
        calls.append(list(argv))
        if argv[1:5] == ["-I", "-S", "-m", "venv"]:
            _materialize_fake_venv(argv)

    with pytest.raises(env.IdentityValidationError, match="path.*changed"):
        env.build_environment(
            lock_path,
            wheelhouse,
            releases_root=tmp_path / "envs" / "releases",
            python_report=PYTHON_REPORT,
            system_library_report=SYSTEM_REPORT,
            python_executable=str(base_python),
            command_runner=runner,
            smoke_runner=lambda _path, identity: _smoke(identity),
        )

    assert calls == []
    assert not (tmp_path / "envs" / "releases").exists()
    assert base_python.stat().st_ino != original.st_ino


def test_real_venv_uses_stable_name_for_the_held_base_interpreter(
    tmp_path: Path,
) -> None:
    base_python = Path(sys.executable).resolve()
    base_python_fd = os.open(base_python, os.O_RDONLY)
    launcher_home = tmp_path / "bound-python-home"
    launcher_home.mkdir(mode=0o700)
    candidate = tmp_path / "candidate"
    try:
        command = env._bound_base_python_command(
            base_python_fd,
            launcher_home,
            [
                "-I", "-S", "-m", "venv", "--copies", "--without-pip",
                str(candidate),
            ],
        )
        env._run_command(command, pass_fds=(base_python_fd,))
        env._verify_bound_base_python(command[0], base_python_fd)
        candidate_python = candidate / "bin" / "python3.12"
        assert not (candidate / "bin" / "pip").exists()
        env._run_command([
            str(candidate_python), "-I", "-m", "ensurepip", "--upgrade",
            "--default-pip",
        ])
    finally:
        os.close(base_python_fd)

    assert (candidate / "bin" / "python3.12").is_file()
    assert (candidate / "bin" / "python").is_file()
    assert (candidate / "bin" / "pip").is_file()
    assert not os.path.lexists(str(candidate / "bin" / str(base_python_fd)))


def test_build_normalizes_pyvenv_config_away_from_the_ephemeral_launcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path, wheelhouse = _write_lock_and_wheels(tmp_path)
    canonical_base = tmp_path / "verified-base" / "bin" / "python3.12"
    canonical_base.parent.mkdir(parents=True)
    os.link(Path(sys.executable).resolve(), canonical_base)
    monkeypatch.setattr(
        env, "resolve_base_python", lambda _value: str(canonical_base)
    )
    monkeypatch.setattr(
        env, "open_base_python", lambda path: os.open(path, os.O_RDONLY)
    )
    base_runner = _runner([])

    def runner(
        argv: list[str],
        *,
        process_env: dict[str, str] | None = None,
        pass_fds: tuple[int, ...] = (),
    ) -> None:
        base_runner(argv, process_env=process_env, pass_fds=pass_fds)
        if argv[1:5] == ["-I", "-S", "-m", "venv"]:
            candidate = Path(argv[-1])
            launcher = Path(argv[0])
            (candidate / "pyvenv.cfg").write_text(
                "home = %s\n"
                "include-system-site-packages = false\n"
                "version = 3.12.13\n"
                "executable = %s\n"
                "command = %s -m venv --copies --without-pip %s\n"
                % (launcher.parent, launcher, launcher, candidate),
                encoding="utf-8",
            )

    final = env.build_environment(
        lock_path,
        wheelhouse,
        releases_root=tmp_path / "envs" / "releases",
        python_report=PYTHON_REPORT,
        system_library_report=SYSTEM_REPORT,
        command_runner=runner,
        smoke_runner=lambda _path, identity: _smoke(identity),
    )

    expected = (
        "home = %s\n"
        "include-system-site-packages = false\n"
        "version = 3.12.13\n"
        "executable = %s\n"
        "command = %s -m venv --copies --without-pip %s\n"
        % (canonical_base.parent, canonical_base, canonical_base, final)
    )
    assert (final / "pyvenv.cfg").read_text(encoding="utf-8") == expected
    assert ".h200-input-" not in expected
    assert subprocess.run(
        [str(final / "bin" / "python")], check=False,
    ).returncode == 0


def test_default_smoke_runner_disables_bytecode_before_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="{}", stderr="")

    monkeypatch.setattr(env.subprocess, "run", run)
    environment_id = "wam-" + "a" * 20
    assert env._default_smoke_runner(
        tmp_path / "candidate", {"environment_id": environment_id}
    ) == {}

    assert calls == [[
        str(tmp_path / "candidate" / "bin" / "python"),
        "-B",
        "-I",
        str(Path(env.__file__).with_name("h200_smoke.py")),
        "--environment-id",
        environment_id,
        "--identity",
        str(tmp_path / "candidate" / env.IDENTITY_NAME),
    ]]


def test_default_smoke_runner_reports_failure_without_request_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    environment_id = "wam-" + "a" * 20
    identity_path = tmp_path / "candidate" / env.IDENTITY_NAME

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured.update(kwargs)
        if kwargs.get("check") is True:
            raise subprocess.CalledProcessError(
                2, argv, output="stdout-secret", stderr="smoke exploded"
            )
        return subprocess.CompletedProcess(
            argv,
            2,
            stdout="stdout-secret",
            stderr="smoke exploded %s %s" % (environment_id, identity_path),
        )

    monkeypatch.setattr(env.subprocess, "run", run)
    monkeypatch.setattr(
        env,
        "_runtime_process_env",
        lambda _home: {"PRIVATE_SENTINEL": "process-env-secret"},
    )
    with pytest.raises(env.SmokeValidationError) as caught:
        env._default_smoke_runner(
            tmp_path / "candidate",
            {
                "environment_id": environment_id,
                "private": "identity-content-secret",
            },
        )

    message = str(caught.value)
    assert message == (
        "H200 smoke command failed (exit 2): "
        "smoke exploded <environment-id> <identity-path>"
    )
    assert captured["check"] is False
    assert captured["stdout"] is subprocess.PIPE
    assert captured["stderr"] is subprocess.PIPE
    assert captured["text"] is True
    assert "stdout-secret" not in message
    assert "process-env-secret" not in message
    assert "identity-content-secret" not in message
    assert environment_id not in message
    assert str(identity_path) not in message


@pytest.mark.parametrize(
    ("stderr", "diagnostic"),
    [
        ("", "<no stderr>"),
        (b"not text", "<non-text stderr>"),
    ],
)
def test_default_smoke_runner_handles_empty_or_nontext_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stderr: object,
    diagnostic: str,
) -> None:
    monkeypatch.setattr(
        env.subprocess,
        "run",
        lambda argv, **_kwargs: subprocess.CompletedProcess(
            argv, 2, stdout="ignored", stderr=stderr,
        ),
    )

    with pytest.raises(env.SmokeValidationError) as caught:
        env._default_smoke_runner(
            tmp_path / "candidate", {"environment_id": "wam-" + "b" * 20}
        )

    assert str(caught.value) == (
        "H200 smoke command failed (exit 2): %s" % diagnostic
    )


def test_default_smoke_runner_caps_oversize_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        env.subprocess,
        "run",
        lambda argv, **_kwargs: subprocess.CompletedProcess(
            argv, 2, stdout="ignored", stderr="💥" * 4096,
        ),
    )

    with pytest.raises(env.SmokeValidationError) as caught:
        env._default_smoke_runner(
            tmp_path / "candidate", {"environment_id": "wam-" + "c" * 20}
        )

    message = str(caught.value)
    assert message.startswith("H200 smoke command failed (exit 2): ")
    assert message.endswith("...[truncated]")
    assert len(message.split(": ", 1)[1].encode("utf-8")) <= 2048


@pytest.mark.parametrize("returncode", [True, False, 0.0, None, "2", 999, -999])
def test_default_smoke_runner_rejects_invalid_process_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, returncode: object,
) -> None:
    monkeypatch.setattr(
        env.subprocess,
        "run",
        lambda argv, **_kwargs: SimpleNamespace(
            args=argv, returncode=returncode, stdout="ignored", stderr="secret"
        ),
    )

    with pytest.raises(
        env.SmokeValidationError, match="invalid exit status"
    ) as caught:
        env._default_smoke_runner(
            tmp_path / "candidate", {"environment_id": "wam-" + "d" * 20}
        )

    assert "secret" not in str(caught.value)


def test_h200_base_interpreter_is_the_canonical_uv_managed_cp312() -> None:
    assert env.H200_BASE_PYTHON == (
        "/home/ssh-ziyue/.local/share/uv/python/"
        "cpython-3.12.13-linux-x86_64-gnu/bin/python3.12"
    )


def test_runtime_probe_wraps_a_missing_host_tool_as_identity_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(*_args, **_kwargs):
        raise FileNotFoundError("missing probe tool")

    monkeypatch.setattr(env.subprocess, "run", missing)
    with pytest.raises(env.IdentityValidationError, match="probe"):
        REAL_RUNTIME_PROBE("/verified/bin/python3.12")


def test_identity_rejects_missing_wrong_or_duplicate_wheels(tmp_path: Path) -> None:
    lock_path, wheelhouse = _write_lock_and_wheels(tmp_path)
    (wheelhouse / _wheel_filename("lerobot", "0.3.3")).unlink()
    with pytest.raises(env.IdentityValidationError, match="wheel"):
        env.build_environment_identity(
            lock_path,
            wheelhouse,
            python_report=PYTHON_REPORT,
            system_library_report=SYSTEM_REPORT,
        )

    wheel = next(wheelhouse.glob("*.whl"))
    wheel.write_bytes(b"tampered")
    with pytest.raises(env.IdentityValidationError, match="wheel"):
        env.build_environment_identity(
            lock_path,
            wheelhouse,
            python_report=PYTHON_REPORT,
            system_library_report=SYSTEM_REPORT,
        )


def test_build_uses_final_path_hash_install_and_marks_only_after_smoke(tmp_path: Path) -> None:
    lock_path, wheelhouse = _write_lock_and_wheels(tmp_path)
    releases = tmp_path / "envs" / "releases"
    calls: list[list[str]] = []

    final = env.build_environment(
        lock_path,
        wheelhouse,
        releases_root=releases,
        python_executable="python3.12",
        python_report=PYTHON_REPORT,
        system_library_report=SYSTEM_REPORT,
        command_runner=_runner(calls),
        smoke_runner=lambda _path, identity: _smoke(identity),
    )

    identity = env.verify_complete_environment(final)
    assert final == releases / identity["environment_id"]
    assert Path(calls[0][0]).name == "python3.12"
    assert calls[0][1:] == [
        "-I", "-S", "-m", "venv", "--copies", "--without-pip", str(final),
    ]
    assert calls[1] == [
        str(final / "bin" / "python3.12"),
        "-I",
        "-m",
        "ensurepip",
        "--upgrade",
        "--default-pip",
    ]
    assert calls[2][:8] == [
        str(final / "bin" / "python"),
        "-I",
        "-m",
        "pip",
        "--isolated",
        "install",
        "--no-index",
        "--find-links",
    ]
    assert calls[2][8].startswith(("/proc/self/fd/", "/dev/fd/"))
    assert calls[2][9:11] == ["--require-hashes", "-r"]
    assert calls[2][11].startswith(("/proc/self/fd/", "/dev/fd/"))
    marker_raw = (final / env.COMPLETE_NAME).read_bytes()
    marker = json.loads(marker_raw)
    assert marker_raw == env.canonical_json_bytes(marker)
    assert marker["payload_sha256"] == _sha(env.canonical_json_bytes(marker["payload"]))
    assert not (final.stat().st_mode & stat.S_IWUSR)


def test_build_commands_use_isolated_python_pip_and_private_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path, wheelhouse = _write_lock_and_wheels(tmp_path)
    monkeypatch.setenv("HOME", "/attacker/home")
    calls: list[list[str]] = []
    homes: list[str] = []
    base_runner = _runner(calls)

    def isolated_runner(
        argv: list[str],
        *,
        process_env: dict[str, str] | None = None,
        pass_fds: tuple[int, ...] = (),
    ) -> None:
        assert process_env is not None
        assert process_env["HOME"] != "/attacker/home"
        assert Path(process_env["HOME"]).is_dir()
        assert not any(
            key.startswith(("PYTHON", "LD_", "PIP_"))
            and key != "PIP_NO_INDEX"
            for key in process_env
        )
        homes.append(process_env["HOME"])
        base_runner(argv, process_env=process_env, pass_fds=pass_fds)

    env.build_environment(
        lock_path,
        wheelhouse,
        releases_root=tmp_path / "envs" / "releases",
        python_report=PYTHON_REPORT,
        system_library_report=SYSTEM_REPORT,
        command_runner=isolated_runner,
        smoke_runner=lambda _path, identity: _smoke(identity),
    )

    assert len(calls) == 3
    assert calls[0][1:5] == ["-I", "-S", "-m", "venv"]
    assert calls[1][1:4] == ["-I", "-m", "ensurepip"]
    assert calls[2][1:6] == ["-I", "-m", "pip", "--isolated", "install"]
    assert homes[0] == homes[1] == homes[2]
    assert not Path(homes[0]).exists()


def test_build_installs_from_a_descriptor_snapshot_when_inputs_are_swapped(
    tmp_path: Path,
) -> None:
    lock_path, wheelhouse = _write_lock_and_wheels(tmp_path)
    original_lock = lock_path.read_bytes()
    original_wheels = {path.name: path.read_bytes() for path in wheelhouse.iterdir()}
    calls = {"count": 0}

    def swapping_runner(
        argv: list[str],
        *,
        process_env: dict[str, str] | None = None,
        pass_fds: tuple[int, ...] = (),
    ) -> None:
        del process_env
        calls["count"] += 1
        if argv[1:5] == ["-I", "-S", "-m", "venv"]:
            _materialize_fake_venv(argv)
            return
        if argv[1:] == [
            "-I", "-m", "ensurepip", "--upgrade", "--default-pip",
        ]:
            _materialize_fake_ensurepip(argv)
            return
        lock_path.write_bytes(b"attacker lock bytes\n")
        first_wheel = next(wheelhouse.iterdir())
        first_wheel.write_bytes(b"attacker wheel bytes")
        snapshot_wheels = argv[argv.index("--find-links") + 1]
        snapshot_lock = argv[argv.index("-r") + 1]
        assert snapshot_wheels.startswith(("/proc/self/fd/", "/dev/fd/"))
        assert snapshot_lock.startswith(("/proc/self/fd/", "/dev/fd/"))
        assert len(pass_fds) >= 2
        lock_fd = pass_fds[1]
        os.lseek(lock_fd, 0, os.SEEK_SET)
        assert os.read(lock_fd, len(original_lock) + 1) == original_lock
        observed_wheels = {}
        for filename in os.listdir(pass_fds[0]):
            descriptor = os.open(filename, os.O_RDONLY, dir_fd=pass_fds[0])
            try:
                observed_wheels[filename] = os.read(descriptor, 1024 * 1024)
            finally:
                os.close(descriptor)
        assert observed_wheels == original_wheels

    final = env.build_environment(
        lock_path,
        wheelhouse,
        releases_root=tmp_path / "envs" / "releases",
        python_report=PYTHON_REPORT,
        system_library_report=SYSTEM_REPORT,
        command_runner=swapping_runner,
        smoke_runner=lambda _path, identity: _smoke(identity),
    )

    identity = env.verify_complete_environment(final)
    assert identity["payload"]["lock"]["sha256"] == _sha(original_lock)
    assert lock_path.read_bytes() != original_lock


def test_build_revalidates_the_descriptor_snapshot_after_pip(tmp_path: Path) -> None:
    lock_path, wheelhouse = _write_lock_and_wheels(tmp_path)

    def mutating_runner(
        argv: list[str],
        *,
        process_env: dict[str, str] | None = None,
        pass_fds: tuple[int, ...] = (),
    ) -> None:
        del process_env
        if argv[1:5] == ["-I", "-S", "-m", "venv"]:
            _materialize_fake_venv(argv)
            return
        if argv[1:] == [
            "-I", "-m", "ensurepip", "--upgrade", "--default-pip",
        ]:
            _materialize_fake_ensurepip(argv)
            return

        lock_fd = pass_fds[1]
        if sys.platform == "darwin":
            raw_path = fcntl.fcntl(lock_fd, 50, b"\0" * 1024)
            snapshot_lock = Path(raw_path.split(b"\0", 1)[0].decode())
        else:
            snapshot_lock = Path(os.readlink(f"/proc/self/fd/{lock_fd}"))
        os.fchmod(lock_fd, 0o600)
        with snapshot_lock.open("ab") as handle:
            handle.write(b"\n")

    with pytest.raises(
        env.IdentityValidationError,
        match="snapshot changed during installation",
    ):
        env.build_environment(
            lock_path,
            wheelhouse,
            releases_root=tmp_path / "envs" / "releases",
            python_report=PYTHON_REPORT,
            system_library_report=SYSTEM_REPORT,
            command_runner=mutating_runner,
            smoke_runner=lambda _path, identity: _smoke(identity),
        )


def test_post_install_revalidation_reads_the_held_descriptors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path, wheelhouse = _write_lock_and_wheels(tmp_path)
    real_builder = env.build_environment_identity
    first_identity: dict[str, object] = {}

    def stale_path_builder(*args: object, **kwargs: object) -> dict:
        if first_identity:
            return json.loads(json.dumps(first_identity))
        identity = real_builder(*args, **kwargs)
        first_identity.update(json.loads(json.dumps(identity)))
        return identity

    monkeypatch.setattr(env, "build_environment_identity", stale_path_builder)

    def mutating_runner(
        argv: list[str],
        *,
        process_env: dict[str, str] | None = None,
        pass_fds: tuple[int, ...] = (),
    ) -> None:
        del process_env
        if argv[1:5] == ["-I", "-S", "-m", "venv"]:
            _materialize_fake_venv(argv)
            return
        if argv[1:] == [
            "-I", "-m", "ensurepip", "--upgrade", "--default-pip",
        ]:
            _materialize_fake_ensurepip(argv)
            return
        lock_fd = pass_fds[1]
        os.fchmod(lock_fd, 0o600)
        os.lseek(lock_fd, 0, os.SEEK_END)
        writable_fd = os.open(
            f"/proc/self/fd/{lock_fd}" if sys.platform.startswith("linux") else (
                fcntl.fcntl(lock_fd, 50, b"\0" * 1024)
                .split(b"\0", 1)[0]
                .decode()
            ),
            os.O_WRONLY | os.O_APPEND,
        )
        try:
            os.write(writable_fd, b"\n")
        finally:
            os.close(writable_fd)

    with pytest.raises(
        env.IdentityValidationError,
        match="snapshot changed during installation",
    ):
        env.build_environment(
            lock_path,
            wheelhouse,
            releases_root=tmp_path / "envs" / "releases",
            python_report=PYTHON_REPORT,
            system_library_report=SYSTEM_REPORT,
            command_runner=mutating_runner,
            smoke_runner=lambda _path, identity: _smoke(identity),
        )


def test_completion_marker_binds_the_exact_installed_tree(tmp_path: Path) -> None:
    lock_path, wheelhouse = _write_lock_and_wheels(tmp_path)
    releases = tmp_path / "envs" / "releases"
    base_runner = _runner([])

    def runner(
        argv: list[str], *, process_env: dict[str, str] | None = None,
        pass_fds: tuple[int, ...] = (),
    ) -> None:
        base_runner(argv, process_env=process_env, pass_fds=pass_fds)
        if argv[1:5] == ["-I", "-S", "-m", "venv"]:
            package = Path(argv[-1]) / "lib" / "python3.12" / "site-packages" / "fixture.py"
            package.parent.mkdir(parents=True, exist_ok=True)
            package.write_text("VALUE = 1\n", encoding="utf-8")

    final = env.build_environment(
        lock_path,
        wheelhouse,
        releases_root=releases,
        python_report=PYTHON_REPORT,
        system_library_report=SYSTEM_REPORT,
        command_runner=runner,
        smoke_runner=lambda _path, identity: _smoke(identity),
    )

    marker = json.loads((final / env.COMPLETE_NAME).read_text(encoding="utf-8"))
    installed_tree = marker["payload"]["installed_tree"]
    rows = installed_tree["rows"]
    paths = {row["path"] for row in rows}
    assert "bin/python" in paths
    assert "lib/python3.12/site-packages/fixture.py" in paths
    assert env.IDENTITY_NAME in paths
    assert env.SMOKE_NAME in paths
    assert env.COMPLETE_NAME not in paths
    assert installed_tree["sha256"] == _sha(env.canonical_json_bytes(rows))
    assert all(set(row) == {"mode", "path", "sha256", "size"} for row in rows)


def test_complete_environment_rejects_installed_tree_tampering(tmp_path: Path) -> None:
    lock_path, wheelhouse = _write_lock_and_wheels(tmp_path)
    final = env.build_environment(
        lock_path,
        wheelhouse,
        releases_root=tmp_path / "envs" / "releases",
        python_report=PYTHON_REPORT,
        system_library_report=SYSTEM_REPORT,
        command_runner=_runner([]),
        smoke_runner=lambda _path, identity: _smoke(identity),
    )
    environment_python = final / "bin" / "python"
    environment_python.chmod(0o755)
    environment_python.write_text("#!/bin/sh\nexit 73\n", encoding="utf-8")
    environment_python.chmod(0o555)

    with pytest.raises(env.EnvironmentVerificationError, match="installed tree"):
        env.verify_complete_environment(final)
    current = tmp_path / "envs" / "canary" / "current"
    with pytest.raises(env.EnvironmentVerificationError, match="complete"):
        env.activate_canary(
            final,
            current,
            releases_root=tmp_path / "envs" / "releases",
            expected_current=None,
        )
    assert not current.exists()


@pytest.mark.parametrize("target_kind", ["directory", "marker"])
def test_complete_environment_rejects_any_writable_completed_state(
    tmp_path: Path, target_kind: str
) -> None:
    lock_path, wheelhouse = _write_lock_and_wheels(tmp_path)
    final = env.build_environment(
        lock_path,
        wheelhouse,
        releases_root=tmp_path / "envs" / "releases",
        python_report=PYTHON_REPORT,
        system_library_report=SYSTEM_REPORT,
        command_runner=_runner([]),
        smoke_runner=lambda _path, identity: _smoke(identity),
    )
    target = final / "bin" if target_kind == "directory" else final / env.COMPLETE_NAME
    target.chmod(stat.S_IMODE(target.stat().st_mode) | stat.S_IWUSR)

    with pytest.raises(env.EnvironmentVerificationError, match="writable|immutable"):
        env.verify_complete_environment(final)


def test_standard_lib64_symlink_is_removed_before_tree_sealing(tmp_path: Path) -> None:
    lock_path, wheelhouse = _write_lock_and_wheels(tmp_path)
    base_runner = _runner([])

    def linux_venv_runner(
        argv: list[str], *, process_env: dict[str, str] | None = None,
        pass_fds: tuple[int, ...] = (),
    ) -> None:
        base_runner(argv, process_env=process_env, pass_fds=pass_fds)
        if argv[1:5] == ["-I", "-S", "-m", "venv"]:
            root = Path(argv[-1])
            (root / "lib").mkdir(exist_ok=True)
            (root / "lib64").symlink_to("lib", target_is_directory=True)

    final = env.build_environment(
        lock_path,
        wheelhouse,
        releases_root=tmp_path / "envs" / "releases",
        python_report=PYTHON_REPORT,
        system_library_report=SYSTEM_REPORT,
        command_runner=linux_venv_runner,
        smoke_runner=lambda _path, identity: _smoke(identity),
    )

    assert not os.path.lexists(str(final / "lib64"))
    env.verify_complete_environment(final)


@pytest.mark.parametrize("entry_kind", ["symlink", "fifo"])
def test_build_rejects_symlink_or_nonregular_installed_entry(
    tmp_path: Path, entry_kind: str
) -> None:
    lock_path, wheelhouse = _write_lock_and_wheels(tmp_path)
    base_runner = _runner([])

    def unsafe_runner(
        argv: list[str], *, process_env: dict[str, str] | None = None,
        pass_fds: tuple[int, ...] = (),
    ) -> None:
        base_runner(argv, process_env=process_env, pass_fds=pass_fds)
        if argv[1:5] == ["-I", "-S", "-m", "venv"]:
            root = Path(argv[-1])
            if entry_kind == "symlink":
                (root / "unsafe-link").symlink_to("bin/python")
            else:
                os.mkfifo(root / "unsafe-fifo")

    with pytest.raises(env.EnvironmentVerificationError, match="installed tree"):
        env.build_environment(
            lock_path,
            wheelhouse,
            releases_root=tmp_path / "envs" / "releases",
            python_report=PYTHON_REPORT,
            system_library_report=SYSTEM_REPORT,
            command_runner=unsafe_runner,
            smoke_runner=lambda _path, identity: _smoke(identity),
        )


def test_smoke_runtime_must_match_the_environment_identity(tmp_path: Path) -> None:
    identity, _lock_path, _wheelhouse = _identity(tmp_path)
    smoke = _smoke(identity)
    smoke["runtime"]["system_libraries"]["nvidia_driver_version"] = "999.0"

    with pytest.raises(env.SmokeValidationError, match="runtime"):
        env._validate_smoke_report(smoke, identity)


def test_loaded_libcuda_must_match_identity_inode_and_hash(tmp_path: Path) -> None:
    import h200_smoke

    library = tmp_path / "libcuda.so.1"
    library.write_bytes(b"verified libcuda bytes")
    info = library.stat()
    maps = (
        "7f000000-7f001000 r-xp 00000000 %x:%x %s %s\n"
        % (os.major(info.st_dev), os.minor(info.st_dev), info.st_ino, library)
    )
    exact = [
        {
            "device": info.st_dev,
            "inode": info.st_ino,
            "name": "libcuda.so.1",
            "path": str(library),
            "sha256": _sha(library.read_bytes()),
        }
    ]

    assert h200_smoke._loaded_libcuda_inventory(exact, maps_text=maps) == exact
    for field, value in (("inode", info.st_ino + 1), ("sha256", "b" * 64)):
        expected = json.loads(json.dumps(exact))
        expected[0][field] = value
        with pytest.raises(h200_smoke.SmokeFailure, match="loaded libcuda"):
            h200_smoke._loaded_libcuda_inventory(expected, maps_text=maps)


def test_failed_smoke_cleans_only_the_new_final_path(tmp_path: Path) -> None:
    lock_path, wheelhouse = _write_lock_and_wheels(tmp_path)
    releases = tmp_path / "envs" / "releases"
    sibling = releases / "wam-user-owned"
    sibling.mkdir(parents=True)
    (sibling / "keep").write_text("keep", encoding="utf-8")
    identity = env.build_environment_identity(
        lock_path,
        wheelhouse,
        python_report=PYTHON_REPORT,
        system_library_report=SYSTEM_REPORT,
    )
    final = releases / identity["environment_id"]

    with pytest.raises(env.SmokeValidationError, match="smoke"):
        env.build_environment(
            lock_path,
            wheelhouse,
            releases_root=releases,
            python_report=PYTHON_REPORT,
            system_library_report=SYSTEM_REPORT,
            command_runner=_runner([]),
            smoke_runner=lambda _path, doc: _smoke(doc, ok=False),
        )

    assert not final.exists()
    assert (sibling / "keep").read_text(encoding="utf-8") == "keep"


def test_failed_explicit_ensurepip_cleans_only_the_owned_candidate(
    tmp_path: Path,
) -> None:
    lock_path, wheelhouse = _write_lock_and_wheels(tmp_path)
    releases = tmp_path / "envs" / "releases"
    sibling = releases / "wam-user-owned"
    sibling.mkdir(parents=True)
    (sibling / "keep").write_text("keep", encoding="utf-8")
    identity = env.build_environment_identity(
        lock_path,
        wheelhouse,
        python_report=PYTHON_REPORT,
        system_library_report=SYSTEM_REPORT,
    )
    final = releases / identity["environment_id"]

    def runner(
        argv: list[str],
        *,
        process_env: dict[str, str] | None = None,
        pass_fds: tuple[int, ...] = (),
    ) -> None:
        del process_env
        del pass_fds
        if argv[1:5] == ["-I", "-S", "-m", "venv"]:
            _materialize_fake_venv(argv)
            return
        if argv[1:4] == ["-I", "-m", "ensurepip"]:
            raise subprocess.CalledProcessError(1, argv)
        raise AssertionError("pip install must not run after ensurepip failure")

    with pytest.raises(subprocess.CalledProcessError):
        env.build_environment(
            lock_path,
            wheelhouse,
            releases_root=releases,
            python_report=PYTHON_REPORT,
            system_library_report=SYSTEM_REPORT,
            command_runner=runner,
            smoke_runner=lambda _path, identity: _smoke(identity),
        )

    assert not final.exists()
    assert (sibling / "keep").read_text(encoding="utf-8") == "keep"


def test_failed_build_never_cleans_a_replacement_user_directory(
    tmp_path: Path,
) -> None:
    lock_path, wheelhouse = _write_lock_and_wheels(tmp_path)
    releases = tmp_path / "envs" / "releases"
    calls = {"count": 0}
    replacement = {"sentinel": None}

    def replace_then_fail(
        argv: list[str], *, process_env: dict[str, str] | None = None,
        pass_fds: tuple[int, ...] = (),
    ) -> None:
        del process_env
        del pass_fds
        calls["count"] += 1
        final = Path(argv[-1]) if calls["count"] == 1 else Path(argv[0]).parents[1]
        if calls["count"] == 1:
            _materialize_fake_venv(argv)
            return
        final.rename(releases / "externally-moved-owned-candidate")
        final.mkdir()
        sentinel = final / "user-file"
        sentinel.write_text("do-not-delete", encoding="utf-8")
        replacement["sentinel"] = sentinel
        raise RuntimeError("injected install failure after external replacement")

    with pytest.raises(RuntimeError, match="external replacement"):
        env.build_environment(
            lock_path,
            wheelhouse,
            releases_root=releases,
            python_report=PYTHON_REPORT,
            system_library_report=SYSTEM_REPORT,
            command_runner=replace_then_fail,
            smoke_runner=lambda _path, identity: _smoke(identity),
        )

    sentinel = replacement["sentinel"]
    assert sentinel is not None
    assert sentinel.read_text(encoding="utf-8") == "do-not-delete"


def test_sigkill_leftover_is_quarantined_and_same_identity_retries(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "fork"):
        pytest.skip("requires fork to simulate an uncatchable process exit")
    lock_path, wheelhouse = _write_lock_and_wheels(tmp_path)
    releases = tmp_path / "envs" / "releases"
    identity = env.build_environment_identity(
        lock_path,
        wheelhouse,
        python_report=PYTHON_REPORT,
        system_library_report=SYSTEM_REPORT,
    )
    final = releases / identity["environment_id"]
    child = os.fork()
    if child == 0:
        calls = {"count": 0}

        def crash_during_install(
            argv: list[str], *, process_env: dict[str, str] | None = None,
            pass_fds: tuple[int, ...] = (),
        ) -> None:
            del process_env
            del pass_fds
            calls["count"] += 1
            if calls["count"] == 1:
                _materialize_fake_venv(argv)
                return
            os._exit(91)

        env.build_environment(
            lock_path,
            wheelhouse,
            releases_root=releases,
            python_report=PYTHON_REPORT,
            system_library_report=SYSTEM_REPORT,
            command_runner=crash_during_install,
            smoke_runner=lambda _path, document: _smoke(document),
        )
        os._exit(0)

    _pid, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 91
    assert final.is_dir()
    assert not (final / env.COMPLETE_NAME).exists()

    rebuilt = env.build_environment(
        lock_path,
        wheelhouse,
        releases_root=releases,
        python_report=PYTHON_REPORT,
        system_library_report=SYSTEM_REPORT,
        command_runner=_runner([]),
        smoke_runner=lambda _path, document: _smoke(document),
    )

    assert rebuilt == final
    env.verify_complete_environment(rebuilt)
    assert not list(releases.glob(".h200-incomplete-*"))


def test_quarantine_cleanup_is_descriptor_bound_not_path_rmtree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path, wheelhouse = _write_lock_and_wheels(tmp_path)
    identity = env.build_environment_identity(
        lock_path,
        wheelhouse,
        python_report=PYTHON_REPORT,
        system_library_report=SYSTEM_REPORT,
    )
    releases = env._assert_managed_root(tmp_path / "envs" / "releases")
    final = releases / identity["environment_id"]
    env._create_owned_final(releases, final, identity)
    nested = final / "partial" / "nested"
    nested.mkdir(parents=True)
    (nested / "payload").write_text("partial install\n", encoding="utf-8")

    def forbidden_path_rmtree(*_args, **_kwargs):
        raise AssertionError("quarantine cleanup used pathname rmtree")

    real_open = env.os.open
    real_close = env.os.close
    real_unlink = env.os.unlink
    real_rename = env._rename_noreplace_at
    payload_descriptors: set[int] = set()
    payload_tombstones: set[str] = set()

    def tracking_open(path, *args, **kwargs):
        descriptor = real_open(path, *args, **kwargs)
        if path == "payload":
            payload_descriptors.add(descriptor)
        return descriptor

    def tracking_close(descriptor: int) -> None:
        payload_descriptors.discard(descriptor)
        real_close(descriptor)

    def tracking_rename(
        source_fd: int,
        source_name: str,
        destination_fd: int,
        destination_name: str,
    ) -> None:
        real_rename(source_fd, source_name, destination_fd, destination_name)
        if source_name == "payload":
            payload_tombstones.add(destination_name)

    def checking_unlink(path, *args, **kwargs):
        if path == "payload":
            raise AssertionError("payload was unlinked by its caller-controlled name")
        if path in payload_tombstones:
            assert payload_descriptors, "payload descriptor was closed before unlink"
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(env.shutil, "rmtree", forbidden_path_rmtree)
    monkeypatch.setattr(env.os, "open", tracking_open)
    monkeypatch.setattr(env.os, "close", tracking_close)
    monkeypatch.setattr(env.os, "unlink", checking_unlink)
    monkeypatch.setattr(env, "_rename_noreplace_at", tracking_rename)
    env._recover_incomplete_candidate(releases, final, identity)

    assert not os.path.lexists(str(final))
    assert not list(releases.glob(".h200-incomplete-*"))
    assert payload_tombstones


def test_existing_final_is_never_overwritten_or_cleaned(tmp_path: Path) -> None:
    lock_path, wheelhouse = _write_lock_and_wheels(tmp_path)
    releases = tmp_path / "envs" / "releases"
    identity = env.build_environment_identity(
        lock_path,
        wheelhouse,
        python_report=PYTHON_REPORT,
        system_library_report=SYSTEM_REPORT,
    )
    final = releases / identity["environment_id"]
    final.mkdir(parents=True)
    sentinel = final / "user-file"
    sentinel.write_text("do-not-touch", encoding="utf-8")

    with pytest.raises(env.EnvironmentConflictError, match="exists"):
        env.build_environment(
            lock_path,
            wheelhouse,
            releases_root=releases,
            python_report=PYTHON_REPORT,
            system_library_report=SYSTEM_REPORT,
            command_runner=_runner([]),
            smoke_runner=lambda _path, doc: _smoke(doc),
        )

    assert sentinel.read_text(encoding="utf-8") == "do-not-touch"


def test_exclusive_lock_refuses_a_second_holder(tmp_path: Path) -> None:
    lock_path = tmp_path / ".h200-env.lock"

    with env.exclusive_lock(lock_path):
        with pytest.raises(env.EnvironmentConflictError, match="lock"):
            with env.exclusive_lock(lock_path):
                pass


def test_canary_activation_and_rollback_are_compare_and_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path, wheelhouse = _write_lock_and_wheels(tmp_path)
    releases = tmp_path / "envs" / "releases"
    calls: list[list[str]] = []
    old = env.build_environment(
        lock_path,
        wheelhouse,
        releases_root=releases,
        python_report=PYTHON_REPORT,
        system_library_report=SYSTEM_REPORT,
        command_runner=_runner(calls),
        smoke_runner=lambda _path, doc: _smoke(doc),
    )
    changed_report = json.loads(json.dumps(SYSTEM_REPORT))
    changed_report["nvidia_driver_version"] = "571.0"
    monkeypatch.setattr(
        env,
        "probe_runtime",
        lambda _python, *, pass_fds=(): (
            dict(PYTHON_REPORT),
            json.loads(json.dumps(changed_report)),
        ),
    )
    new = env.build_environment(
        lock_path,
        wheelhouse,
        releases_root=releases,
        python_report=PYTHON_REPORT,
        system_library_report=changed_report,
        command_runner=_runner(calls),
        smoke_runner=lambda _path, doc: _smoke(doc),
    )
    current = tmp_path / "envs" / "canary" / "current"

    assert env.activate_canary(old, current, releases_root=releases, expected_current=None) is None
    assert current.resolve() == old.resolve()
    previous = env.activate_canary(new, current, releases_root=releases, expected_current=old)
    assert previous == old.resolve()
    assert current.resolve() == new.resolve()
    restored = env.rollback_canary(
        current,
        previous,
        releases_root=releases,
        expected_current=new,
    )
    assert restored == new.resolve()
    assert current.resolve() == old.resolve()

    with pytest.raises(env.PointerConflictError, match="compare-and-swap"):
        env.activate_canary(new, current, releases_root=releases, expected_current=None)
    assert current.resolve() == old.resolve()


def test_canary_operations_refuse_production_legacy_and_unverified_targets(tmp_path: Path) -> None:
    releases = tmp_path / "envs" / "releases"
    candidate = releases / "wam-bad"
    candidate.mkdir(parents=True)

    for pointer in (
        Path.home() / "envs" / "wam-current",
        Path.home() / "envs" / "wam",
        tmp_path / "envs" / "wam-current",
    ):
        with pytest.raises(env.PointerPolicyError):
            env.activate_canary(
                candidate,
                pointer,
                releases_root=releases,
                expected_current=None,
            )

    with pytest.raises(env.EnvironmentVerificationError, match="complete"):
        env.activate_canary(
            candidate,
            tmp_path / "envs" / "canary" / "current",
            releases_root=releases,
            expected_current=None,
        )


def test_managed_roots_and_canaries_reject_protected_path_descendants() -> None:
    protected = [Path.home() / "envs" / "wam", Path.home() / "envs" / "wam-current"]
    for root in protected:
        with pytest.raises(env.PointerPolicyError, match="legacy|production"):
            env._assert_managed_root(root / "releases")
        with pytest.raises(env.PointerPolicyError, match="legacy|production"):
            env._assert_canary_pointer(root / "canary" / "current")


def test_build_and_smoke_environment_is_an_explicit_allowlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", "/attacker/home")
    monkeypatch.setenv("PYTHONPATH", "/attacker/python")
    monkeypatch.setenv("PYTHONHOME", "/attacker/home")
    monkeypatch.setenv("LD_PRELOAD", "/attacker/preload.so")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/attacker/lib")
    monkeypatch.setenv("PIP_INDEX_URL", "https://attacker.invalid/simple")
    monkeypatch.setenv("PIP_CONFIG_FILE", "/attacker/pip.conf")
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-propagate")

    safe_home = tmp_path / "isolated-home"
    safe_home.mkdir()
    process_env = env._runtime_process_env(safe_home)

    assert process_env["PIP_NO_INDEX"] == "1"
    assert process_env["HOME"] == str(safe_home)
    assert not any(
        key.startswith(("PYTHON", "LD_", "PIP_")) and key != "PIP_NO_INDEX"
        for key in process_env
    )
    assert "UNRELATED_SECRET" not in process_env


def test_tampered_marker_or_identity_cannot_activate(tmp_path: Path) -> None:
    lock_path, wheelhouse = _write_lock_and_wheels(tmp_path)
    releases = tmp_path / "envs" / "releases"
    final = env.build_environment(
        lock_path,
        wheelhouse,
        releases_root=releases,
        python_report=PYTHON_REPORT,
        system_library_report=SYSTEM_REPORT,
        command_runner=_runner([]),
        smoke_runner=lambda _path, doc: _smoke(doc),
    )
    final.chmod(0o755)
    marker = final / env.COMPLETE_NAME
    marker.chmod(0o644)
    marker.write_bytes(marker.read_bytes() + b" \n")

    with pytest.raises((env.EnvironmentVerificationError, env.PointerPolicyError)):
        env.activate_canary(
            final,
            tmp_path / "envs" / "canary" / "current",
            releases_root=releases,
            expected_current=None,
        )


def test_canary_rejects_a_symlinked_parent_even_when_name_says_canary(tmp_path: Path) -> None:
    lock_path, wheelhouse = _write_lock_and_wheels(tmp_path)
    releases = tmp_path / "envs" / "releases"
    final = env.build_environment(
        lock_path,
        wheelhouse,
        releases_root=releases,
        python_report=PYTHON_REPORT,
        system_library_report=SYSTEM_REPORT,
        command_runner=_runner([]),
        smoke_runner=lambda _path, doc: _smoke(doc),
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    canary = tmp_path / "envs" / "canary"
    canary.symlink_to(outside, target_is_directory=True)

    with pytest.raises(env.PointerPolicyError, match="symlink|isolated"):
        env.activate_canary(
            final,
            canary / "current",
            releases_root=releases,
            expected_current=None,
        )
    assert not (outside / "current").exists()


def test_symlinked_ancestors_cannot_receive_build_or_canary_side_effects(
    tmp_path: Path,
) -> None:
    lock_path, wheelhouse = _write_lock_and_wheels(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    redirected = tmp_path / "redirected"
    redirected.symlink_to(outside, target_is_directory=True)

    with pytest.raises(env.EnvironmentConflictError, match="symlink|unsafe"):
        env.build_environment(
            lock_path,
            wheelhouse,
            releases_root=redirected / "envs" / "releases",
            python_report=PYTHON_REPORT,
            system_library_report=SYSTEM_REPORT,
            command_runner=_runner([]),
            smoke_runner=lambda _path, identity: _smoke(identity),
        )
    assert not (outside / "envs").exists()

    releases = tmp_path / "safe" / "releases"
    candidate = env.build_environment(
        lock_path,
        wheelhouse,
        releases_root=releases,
        python_report=PYTHON_REPORT,
        system_library_report=SYSTEM_REPORT,
        command_runner=_runner([]),
        smoke_runner=lambda _path, identity: _smoke(identity),
    )
    with pytest.raises(env.PointerPolicyError, match="symlink|isolated"):
        env.activate_canary(
            candidate,
            redirected / "canary" / "nested" / "current",
            releases_root=releases,
            expected_current=None,
        )
    assert not (outside / "canary").exists()


def test_canary_restores_previous_if_post_switch_verification_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path, wheelhouse = _write_lock_and_wheels(tmp_path)
    releases = tmp_path / "envs" / "releases"
    old = env.build_environment(
        lock_path,
        wheelhouse,
        releases_root=releases,
        python_report=PYTHON_REPORT,
        system_library_report=SYSTEM_REPORT,
        command_runner=_runner([]),
        smoke_runner=lambda _path, doc: _smoke(doc),
    )
    changed_report = json.loads(json.dumps(SYSTEM_REPORT))
    changed_report["nvidia_driver_version"] = "571.0"
    monkeypatch.setattr(
        env,
        "probe_runtime",
        lambda _python, *, pass_fds=(): (
            dict(PYTHON_REPORT),
            json.loads(json.dumps(changed_report)),
        ),
    )
    new = env.build_environment(
        lock_path,
        wheelhouse,
        releases_root=releases,
        python_report=PYTHON_REPORT,
        system_library_report=changed_report,
        command_runner=_runner([]),
        smoke_runner=lambda _path, doc: _smoke(doc),
    )
    current = tmp_path / "envs" / "canary" / "current"
    env.activate_canary(old, current, releases_root=releases, expected_current=None)
    original = env._current_target
    calls = {"count": 0}

    def fail_post_switch(pointer: Path, root: Path):
        calls["count"] += 1
        if calls["count"] == 2:
            raise env.EnvironmentVerificationError("injected post-switch failure")
        return original(pointer, root)

    monkeypatch.setattr(env, "_current_target", fail_post_switch)
    with pytest.raises(env.EnvironmentVerificationError, match="injected"):
        env.activate_canary(new, current, releases_root=releases, expected_current=old)
    assert current.resolve() == old.resolve()


def _production_code_identity(seed: str = "c") -> tuple[str, str]:
    del seed
    body = b"frozen production code fixture\n"
    payload = {
        "artifact_kind": "code_release",
        "files": [
            {
                "mode": "0644",
                "path": "auscultation/train.py",
                "sha256": _sha(body),
                "size": len(body),
            }
        ],
        "schema": contract.PAYLOAD_SCHEMA,
        "source_role": "mac",
    }
    payload_sha256 = _sha(contract.canonical_json_bytes(payload))
    return "code_release-" + payload_sha256[:20], payload_sha256


def _stage_production_code(home: Path) -> tuple[str, str]:
    release_id, payload_sha256 = _production_code_identity()
    wam_root = home / "wam_ausc"
    releases = wam_root / ".releases"
    final = releases / release_id
    if not final.exists():
        source = home / "code-source"
        payload_file = source / "auscultation" / "train.py"
        payload_file.parent.mkdir(parents=True, exist_ok=True)
        payload_file.write_bytes(b"frozen production code fixture\n")
        manifest = contract.build_manifest(
            source,
            ["auscultation/train.py"],
            artifact_kind="code_release",
            source_role="mac",
        )
        assert manifest["release_id"] == release_id
        contract.write_release(source, manifest, final, target_role="h200")
        wam_root.mkdir(parents=True, exist_ok=True)
        (wam_root / "current").symlink_to(".releases/" + release_id)
    return release_id, payload_sha256


def _activate_production_for_test(candidate: Path, **arguments):
    home = arguments.pop("production_home")
    return env._activate_production_environment_at_home(
        candidate,
        production_home=home,
        **arguments,
    )


def _build_production_candidate(
    tmp_path: Path,
    *,
    system_report: dict | None = None,
) -> tuple[Path, Path]:
    home = tmp_path / "h200-home"
    input_root = tmp_path / (
        "inputs-" + (system_report or SYSTEM_REPORT)["nvidia_driver_version"]
    )
    input_root.mkdir()
    lock_path, wheelhouse = _write_lock_and_wheels(input_root)
    candidate = env.build_environment(
        lock_path,
        wheelhouse,
        releases_root=home / "envs" / "releases",
        python_report=PYTHON_REPORT,
        system_library_report=system_report or SYSTEM_REPORT,
        command_runner=_runner([]),
        smoke_runner=lambda _path, doc: _smoke(doc),
    )
    _stage_production_code(home)
    return home, candidate


def test_production_activation_is_distinct_double_confirmed_and_canonical(
    tmp_path: Path,
) -> None:
    home, candidate = _build_production_candidate(tmp_path)
    release_id, payload_sha256 = _production_code_identity()

    evidence = _activate_production_for_test(
        candidate,
        expected_current=None,
        frozen_code_release_id=release_id,
        frozen_code_payload_sha256=payload_sha256,
        confirm_code_release_id=release_id,
        confirm_code_payload_sha256=payload_sha256,
        production_home=home,
    )

    current = home / "envs" / "wam-current"
    assert current.is_symlink()
    assert current.resolve(strict=True) == candidate.resolve(strict=True)
    assert evidence["schema"] == env.PRODUCTION_ACTIVATION_EVIDENCE_SCHEMA
    assert evidence["status"] == "active"
    assert evidence["request"]["schema"] == env.PRODUCTION_ACTIVATION_REQUEST_SCHEMA
    assert evidence["request"]["frozen_code"]["payload_sha256"] == payload_sha256
    assert evidence["request"]["frozen_code"]["release_id"] == release_id
    assert set(evidence["request"]["frozen_code"]) == {
        "completion_sha256",
        "manifest_sha256",
        "payload_sha256",
        "release_id",
    }
    assert evidence["request"]["environment"] == {
        "environment_id": candidate.name,
        "payload_sha256": json.loads(
            (candidate / env.IDENTITY_NAME).read_text(encoding="utf-8")
        )["payload_sha256"],
    }
    assert evidence["request_sha256"] == _sha(
        env.canonical_json_bytes(evidence["request"])
    )


def test_public_production_capability_has_no_fake_home_or_environment_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "forged outside pytest")
    parameters = inspect.signature(env.activate_production_environment).parameters

    assert "_test_home" not in parameters
    assert "_allow_test_paths" not in parameters


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("confirm_code_release_id", "code_release-" + "d" * 20, "confirmation"),
        ("confirm_code_payload_sha256", "d" * 64, "confirmation"),
        ("frozen_code_release_id", "code_release-not-a-hash", "identity"),
        ("frozen_code_payload_sha256", "d" * 64, "identity"),
    ],
)
def test_production_activation_rejects_wrong_or_malformed_code_confirmation(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    home, candidate = _build_production_candidate(tmp_path)
    release_id, payload_sha256 = _production_code_identity()
    arguments = {
        "expected_current": None,
        "frozen_code_release_id": release_id,
        "frozen_code_payload_sha256": payload_sha256,
        "confirm_code_release_id": release_id,
        "confirm_code_payload_sha256": payload_sha256,
        "production_home": home,
    }
    arguments[field] = value

    with pytest.raises(env.PointerPolicyError, match=message):
        _activate_production_for_test(candidate, **arguments)

    assert not os.path.lexists(str(home / "envs" / "wam-current"))


def test_public_production_capability_ignores_forged_home_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home, candidate = _build_production_candidate(tmp_path)
    forged = tmp_path / "attacker-home"
    monkeypatch.setenv("HOME", str(forged))
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "forged runtime gate")
    monkeypatch.setattr(env, "H200_PRODUCTION_HOME", home)
    release_id, payload_sha256 = _production_code_identity()

    env.activate_production_environment(
        candidate,
        expected_current=None,
        frozen_code_release_id=release_id,
        frozen_code_payload_sha256=payload_sha256,
        confirm_code_release_id=release_id,
        confirm_code_payload_sha256=payload_sha256,
    )

    assert (home / "envs" / "wam-current").resolve(strict=True) == candidate
    assert not forged.exists()


def test_production_activation_rejects_missing_real_frozen_code_pointer(
    tmp_path: Path,
) -> None:
    home, candidate = _build_production_candidate(tmp_path)
    (home / "wam_ausc" / "current").unlink()
    release_id, payload_sha256 = _production_code_identity()

    with pytest.raises(env.PointerPolicyError, match="code.*current|frozen code"):
        _activate_production_for_test(
            candidate,
            expected_current=None,
            frozen_code_release_id=release_id,
            frozen_code_payload_sha256=payload_sha256,
            confirm_code_release_id=release_id,
            confirm_code_payload_sha256=payload_sha256,
            production_home=home,
        )

    assert not os.path.lexists(str(home / "envs" / "wam-current"))


@pytest.mark.parametrize("tamper", ["payload", "manifest", "completion"])
def test_production_activation_rejects_frozen_code_hash_or_marker_tampering(
    tmp_path: Path,
    tamper: str,
) -> None:
    home, candidate = _build_production_candidate(tmp_path)
    release_id, payload_sha256 = _production_code_identity()
    release = home / "wam_ausc" / ".releases" / release_id
    if tamper == "payload":
        (release / "auscultation" / "train.py").write_bytes(b"forged code bytes\n")
    elif tamper == "manifest":
        manifest = release / contract.MANIFEST_NAME
        manifest.write_bytes(manifest.read_bytes() + b" \n")
    else:
        complete = release / contract.COMPLETE_NAME
        complete.write_bytes(complete.read_bytes() + b" \n")

    with pytest.raises(env.PointerPolicyError, match="frozen code"):
        _activate_production_for_test(
            candidate,
            expected_current=None,
            frozen_code_release_id=release_id,
            frozen_code_payload_sha256=payload_sha256,
            confirm_code_release_id=release_id,
            confirm_code_payload_sha256=payload_sha256,
            production_home=home,
        )

    assert not os.path.lexists(str(home / "envs" / "wam-current"))


def test_frozen_code_pointer_race_after_env_switch_is_rejected_and_rolled_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home, candidate = _build_production_candidate(tmp_path)
    release_id, payload_sha256 = _production_code_identity()
    original = env._postverify_production_environment
    calls = {"count": 0}

    def change_code_pointer_after_switch(locked):
        result = original(locked)
        calls["count"] += 1
        if calls["count"] == 1:
            current = home / "wam_ausc" / "current"
            current.unlink()
            current.symlink_to(".releases/code_release-" + "d" * 20)
        return result

    monkeypatch.setattr(
        env, "_postverify_production_environment", change_code_pointer_after_switch
    )
    with pytest.raises(env.PointerPolicyError, match="code current.*changed"):
        _activate_production_for_test(
            candidate,
            expected_current=None,
            frozen_code_release_id=release_id,
            frozen_code_payload_sha256=payload_sha256,
            confirm_code_release_id=release_id,
            confirm_code_payload_sha256=payload_sha256,
            production_home=home,
        )

    assert not os.path.lexists(str(home / "envs" / "wam-current"))


def test_production_activation_takes_code_lock_before_environment_mutation(
    tmp_path: Path,
) -> None:
    home, candidate = _build_production_candidate(tmp_path)
    release_id, payload_sha256 = _production_code_identity()

    with env._locked_production_code(home, release_id, payload_sha256):
        with pytest.raises(env.EnvironmentConflictError, match="code lock"):
            _activate_production_for_test(
                candidate,
                expected_current=None,
                frozen_code_release_id=release_id,
                frozen_code_payload_sha256=payload_sha256,
                confirm_code_release_id=release_id,
                confirm_code_payload_sha256=payload_sha256,
                production_home=home,
            )

    assert not os.path.lexists(str(home / "envs" / "wam-current"))


def test_production_activation_is_idempotent_and_cas_guarded(tmp_path: Path) -> None:
    home, candidate = _build_production_candidate(tmp_path)
    release_id, payload_sha256 = _production_code_identity()
    arguments = {
        "frozen_code_release_id": release_id,
        "frozen_code_payload_sha256": payload_sha256,
        "confirm_code_release_id": release_id,
        "confirm_code_payload_sha256": payload_sha256,
        "production_home": home,
    }
    _activate_production_for_test(candidate, expected_current=None, **arguments)

    repeated = _activate_production_for_test(
        candidate,
        expected_current=candidate,
        **arguments,
    )
    assert repeated["previous_environment_id"] == candidate.name
    assert (home / "envs" / "wam-current").resolve(strict=True) == candidate

    with pytest.raises(env.PointerConflictError, match="compare-and-swap"):
        _activate_production_for_test(
            candidate,
            expected_current=None,
            **arguments,
        )


@pytest.mark.parametrize("unsafe", ["legacy", "outside", "alias"])
def test_production_activation_rejects_every_noncanonical_candidate_path(
    tmp_path: Path,
    unsafe: str,
) -> None:
    home, candidate = _build_production_candidate(tmp_path)
    release_id, payload_sha256 = _production_code_identity()
    if unsafe == "legacy":
        supplied = home / "envs" / "wam"
        supplied.mkdir()
    elif unsafe == "outside":
        supplied = home / "alternate-release"
        supplied.mkdir()
    else:
        supplied = home / "envs" / "releases" / "wam-alias"
        supplied.symlink_to(candidate, target_is_directory=True)

    with pytest.raises(
        (env.PointerPolicyError, env.EnvironmentVerificationError),
        match="canonical|real directory|complete|symlink|unsafe",
    ):
        _activate_production_for_test(
            supplied,
            expected_current=None,
            frozen_code_release_id=release_id,
            frozen_code_payload_sha256=payload_sha256,
            confirm_code_release_id=release_id,
            confirm_code_payload_sha256=payload_sha256,
            production_home=home,
        )

    assert not os.path.lexists(str(home / "envs" / "wam-current"))


@pytest.mark.parametrize("current_kind", ["file", "directory", "dangling", "absolute"])
def test_production_activation_rejects_noncanonical_existing_current(
    tmp_path: Path,
    current_kind: str,
) -> None:
    home, candidate = _build_production_candidate(tmp_path)
    current = home / "envs" / "wam-current"
    if current_kind == "file":
        current.write_text("not a pointer\n", encoding="utf-8")
    elif current_kind == "directory":
        current.mkdir()
    elif current_kind == "dangling":
        current.symlink_to("releases/wam-missing")
    else:
        current.symlink_to(candidate)
    release_id, payload_sha256 = _production_code_identity()

    with pytest.raises(
        (env.PointerConflictError, env.PointerPolicyError, env.EnvironmentVerificationError)
    ):
        _activate_production_for_test(
            candidate,
            expected_current=None,
            frozen_code_release_id=release_id,
            frozen_code_payload_sha256=payload_sha256,
            confirm_code_release_id=release_id,
            confirm_code_payload_sha256=payload_sha256,
            production_home=home,
        )


def test_production_activation_rolls_back_after_postverify_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home, old = _build_production_candidate(tmp_path)
    changed_report = json.loads(json.dumps(SYSTEM_REPORT))
    changed_report["nvidia_driver_version"] = "571.0"
    monkeypatch.setattr(
        env,
        "probe_runtime",
        lambda _python, *, pass_fds=(): (
            dict(PYTHON_REPORT),
            json.loads(json.dumps(changed_report)),
        ),
    )
    _same_home, new = _build_production_candidate(
        tmp_path,
        system_report=changed_report,
    )
    release_id, payload_sha256 = _production_code_identity()
    arguments = {
        "frozen_code_release_id": release_id,
        "frozen_code_payload_sha256": payload_sha256,
        "confirm_code_release_id": release_id,
        "confirm_code_payload_sha256": payload_sha256,
        "production_home": home,
    }
    _activate_production_for_test(old, expected_current=None, **arguments)
    def fail_once_after_switch(_locked):
        raise env.EnvironmentVerificationError("injected production postverify failure")

    monkeypatch.setattr(
        env, "_postverify_production_environment", fail_once_after_switch
    )
    with pytest.raises(env.EnvironmentVerificationError, match="injected"):
        _activate_production_for_test(new, expected_current=old, **arguments)

    assert (home / "envs" / "wam-current").resolve(strict=True) == old


def test_production_activation_preserves_previous_if_atomic_switch_fails_early(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home, old = _build_production_candidate(tmp_path)
    changed_report = json.loads(json.dumps(SYSTEM_REPORT))
    changed_report["nvidia_driver_version"] = "571.0"
    monkeypatch.setattr(
        env,
        "probe_runtime",
        lambda _python, *, pass_fds=(): (
            dict(PYTHON_REPORT),
            json.loads(json.dumps(changed_report)),
        ),
    )
    _same_home, new = _build_production_candidate(
        tmp_path,
        system_report=changed_report,
    )
    release_id, payload_sha256 = _production_code_identity()
    arguments = {
        "frozen_code_release_id": release_id,
        "frozen_code_payload_sha256": payload_sha256,
        "confirm_code_release_id": release_id,
        "confirm_code_payload_sha256": payload_sha256,
        "production_home": home,
    }
    _activate_production_for_test(old, expected_current=None, **arguments)

    def fail_before_switch(_locked, _target_name: str, **_kwargs) -> None:
        raise OSError("injected atomic switch failure")

    monkeypatch.setattr(env, "_replace_production_pointer", fail_before_switch)
    with pytest.raises(OSError, match="injected atomic switch"):
        _activate_production_for_test(new, expected_current=old, **arguments)

    assert (home / "envs" / "wam-current").resolve(strict=True) == old


def test_production_activation_cas_preserves_same_target_replacement_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home, old = _build_production_candidate(tmp_path)
    changed_report = json.loads(json.dumps(SYSTEM_REPORT))
    changed_report["nvidia_driver_version"] = "571.0"
    monkeypatch.setattr(
        env,
        "probe_runtime",
        lambda _python, *, pass_fds=(): (
            dict(PYTHON_REPORT),
            json.loads(json.dumps(changed_report)),
        ),
    )
    _same_home, new = _build_production_candidate(
        tmp_path,
        system_report=changed_report,
    )
    release_id, payload_sha256 = _production_code_identity()
    arguments = {
        "frozen_code_release_id": release_id,
        "frozen_code_payload_sha256": payload_sha256,
        "confirm_code_release_id": release_id,
        "confirm_code_payload_sha256": payload_sha256,
        "production_home": home,
    }
    _activate_production_for_test(old, expected_current=None, **arguments)
    original = env._replace_production_pointer
    raced_identity = {}

    def replace_same_target_then_activate(
        locked, target_name: str, **kwargs
    ) -> None:
        current_target = os.readlink("wam-current", dir_fd=locked.envs_fd)
        os.unlink("wam-current", dir_fd=locked.envs_fd)
        os.symlink(current_target, "wam-current", dir_fd=locked.envs_fd)
        info = os.stat(
            "wam-current", dir_fd=locked.envs_fd, follow_symlinks=False
        )
        raced_identity["value"] = (
            info.st_dev,
            info.st_ino,
            stat.S_IFMT(info.st_mode),
            info.st_mtime_ns,
            info.st_ctime_ns,
        )
        original(locked, target_name, **kwargs)

    monkeypatch.setattr(
        env, "_replace_production_pointer", replace_same_target_then_activate
    )
    with pytest.raises(env.PointerConflictError, match="CAS|identity|changed"):
        _activate_production_for_test(new, expected_current=old, **arguments)

    current = home / "envs" / "wam-current"
    info = current.lstat()
    assert (
        info.st_dev,
        info.st_ino,
        stat.S_IFMT(info.st_mode),
        info.st_mtime_ns,
        info.st_ctime_ns,
    ) == raced_identity["value"]
    assert os.readlink(current) == "releases/%s" % old.name


def test_production_rollback_cas_preserves_same_target_replacement_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home, old = _build_production_candidate(tmp_path)
    changed_report = json.loads(json.dumps(SYSTEM_REPORT))
    changed_report["nvidia_driver_version"] = "571.0"
    monkeypatch.setattr(
        env,
        "probe_runtime",
        lambda _python, *, pass_fds=(): (
            dict(PYTHON_REPORT),
            json.loads(json.dumps(changed_report)),
        ),
    )
    _same_home, new = _build_production_candidate(
        tmp_path,
        system_report=changed_report,
    )
    release_id, payload_sha256 = _production_code_identity()
    arguments = {
        "frozen_code_release_id": release_id,
        "frozen_code_payload_sha256": payload_sha256,
        "confirm_code_release_id": release_id,
        "confirm_code_payload_sha256": payload_sha256,
        "production_home": home,
    }
    _activate_production_for_test(old, expected_current=None, **arguments)
    raced_identity = {}

    def replace_same_target_then_fail(locked) -> None:
        target = os.readlink("wam-current", dir_fd=locked.envs_fd)
        os.unlink("wam-current", dir_fd=locked.envs_fd)
        os.symlink(target, "wam-current", dir_fd=locked.envs_fd)
        info = os.stat(
            "wam-current", dir_fd=locked.envs_fd, follow_symlinks=False
        )
        raced_identity["value"] = (
            info.st_dev,
            info.st_ino,
            stat.S_IFMT(info.st_mode),
            info.st_mtime_ns,
            info.st_ctime_ns,
        )
        raise env.EnvironmentVerificationError("injected postverify failure")

    monkeypatch.setattr(
        env, "_postverify_production_environment", replace_same_target_then_fail
    )
    with pytest.raises(env.PointerConflictError, match="rollback|compare-and-swap"):
        _activate_production_for_test(new, expected_current=old, **arguments)

    current = home / "envs" / "wam-current"
    info = current.lstat()
    assert (
        info.st_dev,
        info.st_ino,
        stat.S_IFMT(info.st_mode),
        info.st_mtime_ns,
        info.st_ctime_ns,
    ) == raced_identity["value"]
    assert os.readlink(current) == "releases/%s" % new.name


def test_production_activation_rolls_back_when_desired_target_fails_exact_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home, old = _build_production_candidate(tmp_path)
    changed_report = json.loads(json.dumps(SYSTEM_REPORT))
    changed_report["nvidia_driver_version"] = "571.0"
    monkeypatch.setattr(
        env,
        "probe_runtime",
        lambda _python, *, pass_fds=(): (
            dict(PYTHON_REPORT),
            json.loads(json.dumps(changed_report)),
        ),
    )
    _same_home, new = _build_production_candidate(
        tmp_path,
        system_report=changed_report,
    )
    release_id, payload_sha256 = _production_code_identity()
    arguments = {
        "frozen_code_release_id": release_id,
        "frozen_code_payload_sha256": payload_sha256,
        "confirm_code_release_id": release_id,
        "confirm_code_payload_sha256": payload_sha256,
        "production_home": home,
    }
    _activate_production_for_test(old, expected_current=None, **arguments)
    original = env._postverify_production_environment

    def tamper_desired_after_switch(locked):
        new.chmod(0o755)
        marker = new / env.COMPLETE_NAME
        marker.chmod(0o644)
        marker.write_bytes(marker.read_bytes() + b" \n")
        return original(locked)

    monkeypatch.setattr(
        env, "_postverify_production_environment", tamper_desired_after_switch
    )
    with pytest.raises((env.EnvironmentVerificationError, env.PointerPolicyError)):
        _activate_production_for_test(new, expected_current=old, **arguments)

    assert (home / "envs" / "wam-current").resolve(strict=True) == old


def test_production_activation_uses_the_existing_h200_env_lock(tmp_path: Path) -> None:
    home, candidate = _build_production_candidate(tmp_path)
    release_id, payload_sha256 = _production_code_identity()

    with env.exclusive_lock(home / "envs" / ".h200-env.lock"):
        with pytest.raises(env.EnvironmentConflictError, match="lock"):
            _activate_production_for_test(
                candidate,
                expected_current=None,
                frozen_code_release_id=release_id,
                frozen_code_payload_sha256=payload_sha256,
                confirm_code_release_id=release_id,
                confirm_code_payload_sha256=payload_sha256,
                production_home=home,
            )

    assert not os.path.lexists(str(home / "envs" / "wam-current"))


def test_production_home_with_symlinked_ancestor_is_rejected(tmp_path: Path) -> None:
    home, candidate = _build_production_candidate(tmp_path)
    redirected = tmp_path / "redirected-h200-home"
    redirected.symlink_to(home, target_is_directory=True)
    release_id, payload_sha256 = _production_code_identity()

    with pytest.raises(env.PointerPolicyError, match="symlink"):
        _activate_production_for_test(
            candidate,
            expected_current=None,
            frozen_code_release_id=release_id,
            frozen_code_payload_sha256=payload_sha256,
            confirm_code_release_id=release_id,
            confirm_code_payload_sha256=payload_sha256,
            production_home=redirected,
        )


def test_production_activation_rejects_releases_root_symlink_swap_during_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home, candidate = _build_production_candidate(tmp_path)
    release_id, payload_sha256 = _production_code_identity()
    releases = home / "envs" / "releases"
    outside = home / "attacker-releases"
    original = env._replace_production_pointer

    def switch_then_escape(locked, target_name: str, **kwargs):
        activated_identity = original(locked, target_name, **kwargs)
        releases.rename(outside)
        releases.symlink_to(outside, target_is_directory=True)
        return activated_identity

    monkeypatch.setattr(env, "_replace_production_pointer", switch_then_escape)
    with pytest.raises(env.PointerPolicyError, match="releases|symlink|changed"):
        _activate_production_for_test(
            candidate,
            expected_current=None,
            frozen_code_release_id=release_id,
            frozen_code_payload_sha256=payload_sha256,
            confirm_code_release_id=release_id,
            confirm_code_payload_sha256=payload_sha256,
            production_home=home,
        )

    current = home / "envs" / "wam-current"
    assert not os.path.lexists(str(current))


def _good_smoke_snapshot() -> dict:
    return {
        "abi": "cp312",
        "capability": [9, 0],
        "cuda_available": True,
        "device_name": "NVIDIA H200 NVL",
        "implementation": "CPython",
        "imports": {
            "annotation_ui": True,
            "audio": True,
            "dataset_reader": True,
            "perception": True,
            "reward_model": True,
        },
        "lerobot_version": "0.3.3",
        "glibc_version": SYSTEM_REPORT["glibc_version"],
        "libraries": SYSTEM_REPORT["libraries"],
        "machine": "x86_64",
        "nvidia_driver_version": SYSTEM_REPORT["nvidia_driver_version"],
        "packages": [
            {"name": "lerobot", "version": "0.3.3"},
            {"name": "torch", "version": "2.7.1+cu128"},
            {"name": "torchaudio", "version": "2.7.1+cu128"},
            {"name": "torchvision", "version": "0.22.1+cu128"},
        ],
        "platform": "linux",
        "python_version": "3.12.13",
        "tensor_result": 14.0,
        "torch_cuda": "12.8",
        "torch_version": "2.7.1+cu128",
        "torchaudio_version": "2.7.1+cu128",
        "torchvision_version": "0.22.1+cu128",
    }


def test_smoke_report_requires_all_h200_runtime_probes() -> None:
    import h200_smoke

    packages = _good_smoke_snapshot()["packages"]
    report = h200_smoke.evaluate_snapshot(
        "wam-" + "a" * 20, _good_smoke_snapshot(), packages
    )

    assert report == {
        "checks": {
            "annotation_ui": True,
            "audio": True,
            "cuda_h200": True,
            "dataset_reader": True,
            "gpu_tensor": True,
            "perception": True,
            "reward_model": True,
        },
        "environment_id": "wam-" + "a" * 20,
        "packages": packages,
        "runtime": {
            "cuda_device_name": "NVIDIA H200 NVL",
            "python": PYTHON_REPORT,
            "system_libraries": SYSTEM_REPORT,
            "torch_cuda": "12.8",
        },
        "schema": env.SMOKE_SCHEMA,
        "versions": CORE_VERSIONS,
    }


@pytest.mark.parametrize(
    "field, value",
    [
        ("implementation", "PyPy"),
        ("python_version", "3.11.9"),
        ("abi", "cp311"),
        ("platform", "darwin"),
        ("machine", "arm64"),
        ("cuda_available", False),
        ("device_name", "NVIDIA A100-SXM4-80GB"),
        ("capability", [8, 0]),
        ("tensor_result", 13.0),
        ("torch_cuda", "12.6"),
        ("torch_version", "2.6.0+cu128"),
        ("torchvision_version", "0.21.0+cu128"),
        ("torchaudio_version", "2.6.0+cu128"),
        ("lerobot_version", "0.3.2"),
    ],
)
def test_smoke_rejects_wrong_runtime_or_package(field: str, value: object) -> None:
    import h200_smoke

    snapshot = _good_smoke_snapshot()
    snapshot[field] = value
    with pytest.raises(h200_smoke.SmokeFailure):
        h200_smoke.evaluate_snapshot(
            "wam-" + "a" * 20, snapshot, _good_smoke_snapshot()["packages"]
        )


def test_smoke_rejects_any_missing_core_import() -> None:
    import h200_smoke

    for name in _good_smoke_snapshot()["imports"]:
        snapshot = _good_smoke_snapshot()
        snapshot["imports"][name] = False
        with pytest.raises(h200_smoke.SmokeFailure, match=name):
            h200_smoke.evaluate_snapshot(
                "wam-" + "a" * 20, snapshot, _good_smoke_snapshot()["packages"]
            )


def test_smoke_rejects_installed_package_inventory_drift() -> None:
    import h200_smoke

    expected = _good_smoke_snapshot()["packages"]
    snapshot = _good_smoke_snapshot()
    snapshot["packages"][0] = {"name": "lerobot", "version": "0.3.2"}
    with pytest.raises(h200_smoke.SmokeFailure, match="package inventory"):
        h200_smoke.evaluate_snapshot("wam-" + "a" * 20, snapshot, expected)


def test_smoke_identity_projects_filename_order_into_canonical_package_order(
    tmp_path: Path,
) -> None:
    import h200_smoke

    packages = dict(CORE_VERSIONS)
    packages.update({"InquirerPy": "0.3.4", "annotated-types": "0.7.0"})
    lock_path, wheelhouse = _write_lock_and_wheels(tmp_path, packages)
    identity = env.build_environment_identity(
        lock_path,
        wheelhouse,
        python_report=PYTHON_REPORT,
        system_library_report=SYSTEM_REPORT,
    )
    identity_path = tmp_path / env.IDENTITY_NAME
    identity_path.write_bytes(env.canonical_json_bytes(identity))

    wheel_names = [row["name"] for row in identity["payload"]["wheels"]]
    assert wheel_names.index("inquirerpy") < wheel_names.index("annotated-types")

    expected = h200_smoke._identity_expectations(
        identity_path, identity["environment_id"]
    )

    assert [row["name"] for row in expected["packages"]] == [
        "annotated-types",
        "inquirerpy",
        "lerobot",
        "torch",
        "torchaudio",
        "torchvision",
    ]


def test_h200_completion_accepts_only_canonical_package_order_for_filename_ordered_identity(
    tmp_path: Path,
) -> None:
    packages = dict(CORE_VERSIONS)
    packages.update({"InquirerPy": "0.3.4", "annotated-types": "0.7.0"})
    lock_path, wheelhouse = _write_lock_and_wheels(tmp_path, packages)
    identity = env.build_environment_identity(
        lock_path,
        wheelhouse,
        python_report=PYTHON_REPORT,
        system_library_report=SYSTEM_REPORT,
    )
    smoke = _smoke(identity)
    with pytest.raises(env.SmokeValidationError, match="inventory drifted"):
        env._validate_smoke_report(smoke, identity)

    smoke["packages"] = sorted(
        smoke["packages"], key=lambda row: env._normalize_name(row["name"])
    )

    assert env._validate_smoke_report(smoke, identity) == smoke


def test_smoke_identity_still_rejects_noncanonical_wheel_filename_order(
    tmp_path: Path,
) -> None:
    import h200_smoke

    packages = dict(CORE_VERSIONS)
    packages.update({"InquirerPy": "0.3.4", "annotated-types": "0.7.0"})
    lock_path, wheelhouse = _write_lock_and_wheels(tmp_path, packages)
    identity = env.build_environment_identity(
        lock_path,
        wheelhouse,
        python_report=PYTHON_REPORT,
        system_library_report=SYSTEM_REPORT,
    )
    identity["payload"]["wheels"] = sorted(
        identity["payload"]["wheels"], key=lambda row: row["name"]
    )
    payload_bytes = env.canonical_json_bytes(identity["payload"])
    identity["payload_sha256"] = _sha(payload_bytes)
    identity["environment_id"] = "wam-" + identity["payload_sha256"][:20]
    identity_path = tmp_path / env.IDENTITY_NAME
    identity_path.write_bytes(env.canonical_json_bytes(identity))

    with pytest.raises(h200_smoke.SmokeFailure, match="wheels are not sorted"):
        h200_smoke._identity_expectations(
            identity_path, identity["environment_id"]
        )


@pytest.mark.parametrize(
    "rows, message",
    [
        (
            [
                {"name": "Foo_Bar", "version": "1.0"},
                {"name": "foo-bar", "version": "2.0"},
            ],
            "duplicate",
        ),
        ([{"name": "-", "version": "1.0"}], "value is invalid"),
    ],
)
def test_smoke_package_inventory_rejects_normalized_collisions_and_invalid_names(
    rows: list[dict[str, str]], message: str
) -> None:
    import h200_smoke

    with pytest.raises(h200_smoke.SmokeFailure, match=message):
        h200_smoke._package_rows(rows)


def test_installed_package_inventory_preserves_canonical_expected_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import h200_smoke

    expected = [
        {"name": "annotated-types", "version": "0.7.0"},
        {"name": "InquirerPy", "version": "0.3.4"},
    ]
    queried: list[str] = []

    def installed_version(name: str) -> str:
        queried.append(name)
        return {"annotated-types": "0.7.0", "InquirerPy": "0.3.4"}[name]

    monkeypatch.setattr(h200_smoke.importlib.metadata, "version", installed_version)

    assert h200_smoke._installed_package_inventory(expected) == expected
    assert queried == ["annotated-types", "InquirerPy"]


def test_tracked_requirements_input_is_exact_pinned_wam_core_without_openpi() -> None:
    requirements_in = HERE.parents[1] / "requirements-h200.in"
    rows = []
    for physical in requirements_in.read_text(encoding="utf-8").splitlines():
        line = physical.split("#", 1)[0].strip()
        if line and not line.startswith("--"):
            rows.append(line)
    assert all("==" in row and ";" not in row and " @ " not in row for row in rows)
    names = {env._normalize_name(row.split("==", 1)[0]) for row in rows}
    assert names == TRACKED_DIRECT_PACKAGES
    assert not any(name.startswith("openpi") for name in names)


def test_blocker_metadata_fixture_remains_fail_closed(tmp_path: Path) -> None:
    requirements_in = tmp_path / "requirements-h200.in"
    requirements_in.write_text("lerobot==0.3.3\n", encoding="utf-8")
    lock_path = tmp_path / "requirements-h200.lock.txt"
    lock_path.write_text(
        "\n".join(
            [
                "# schema: wam_h200_lock_blocker_v1",
                "# status: blocked",
                "# target: cpython-3.12-linux-x86_64-manylinux_2_31-cu128",
                "# resolver: uv 0.10.12",
                "# requirements_in_sha256: %s" % _sha(requirements_in.read_bytes()),
                "# command: %s" % env.LOCK_COMPILE_COMMAND,
                "# blocker: evdev has no compatible wheel for this target",
                "# remediation: publish an audited evdev wheel and rerun the exact command",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(env.LockValidationError, match="no installable requirements"):
        env.parse_hash_lock(lock_path)
    blocker = env.read_blocked_lock_metadata(lock_path, requirements_in)
    assert blocker["schema"] == env.LOCK_BLOCKER_SCHEMA
    assert blocker["status"] == "blocked"
    assert blocker["target"] == "cpython-3.12-linux-x86_64-manylinux_2_31-cu128"
    assert blocker["resolver"] == "uv 0.10.12"
    assert blocker["requirements_in_sha256"] == _sha(requirements_in.read_bytes())
    assert blocker["command"] == env.LOCK_COMPILE_COMMAND
    assert "evdev" in blocker["blocker"] and "wheel" in blocker["blocker"]


def test_tracked_linux_cp312_lock_is_complete_and_core_pinned() -> None:
    lock_path = HERE.parents[1] / "requirements-h200.lock.txt"

    parsed = env.parse_hash_lock(lock_path)
    by_name = env._validate_core_lock(parsed)

    assert parsed.indexes == env.APPROVED_INDEXES
    assert len(parsed.requirements) == 140
    assert {name: env._base_version(by_name[name].version) for name in CORE_VERSIONS} == CORE_VERSIONS
    assert by_name["evdev"].version == "1.9.3"
    assert "34970e279c00ba5007b713bf8664bb541a5caa9e1b208334fa90c700ca254b23" in (
        by_name["evdev"].hashes
    )
    assert not any(row.name.startswith("openpi") for row in parsed.requirements)
