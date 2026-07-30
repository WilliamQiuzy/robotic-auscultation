from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest


DEPLOYMENT_DIR = Path(__file__).resolve().parent
PROFILE = DEPLOYMENT_DIR / "release_profile.json"
sys.path.insert(0, str(DEPLOYMENT_DIR))

import artifact_contract as contract  # noqa: E402
import simulation_release as simulation  # noqa: E402


def _source(root: Path, body: bytes = b"evidence\n") -> Path:
    root.mkdir(parents=True)
    payload = root / "payload.bin"
    payload.write_bytes(body)
    payload.chmod(0o640)
    return root


def _test_policy(tmp_path: Path) -> dict:
    base = tmp_path / "wam_data" / "simulation"
    return {
        "assets": {
            "artifact_kind": "simulation_asset",
            "current_link": (base / "assets" / "current").as_posix(),
            "releases_root": (base / "assets" / "releases").as_posix(),
            "staging_root": (base / "assets" / ".staging").as_posix(),
        },
        "code": {
            "artifact_kind": "simulation_code",
            "current_link": (base / "code" / "current").as_posix(),
            "releases_root": (base / "code" / "releases").as_posix(),
            "staging_root": (base / "code" / ".staging").as_posix(),
        },
        "results": {
            "artifact_kind": "simulation_result",
            "current_link": (base / "results" / "current").as_posix(),
            "releases_root": (base / "results" / "releases").as_posix(),
            "staging_root": (base / "results" / ".staging").as_posix(),
        },
    }


def _profile_for(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    audited = _test_policy(tmp_path)
    monkeypatch.setattr(simulation, "AUDITED_SIMULATION_POLICY", audited)
    value = json.loads(PROFILE.read_text(encoding="utf-8"))
    value["hosts"]["linbox"]["simulation"] = audited
    profile = tmp_path / "release_profile.json"
    profile.write_text(json.dumps(value), encoding="utf-8")
    return profile


def _stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    namespace: str = "code",
    nonce: str = "1" * 32,
) -> tuple[Path, Path]:
    profile = _profile_for(tmp_path, monkeypatch)
    source = _source(tmp_path / "source")
    staged = simulation.stage_simulation_release(
        source,
        ["payload.bin"],
        namespace=namespace,
        nonce=nonce,
        profile_path=profile,
    )
    return profile, staged


def test_checked_in_profile_has_exact_isolated_linbox_roots() -> None:
    value = json.loads(PROFILE.read_text(encoding="utf-8"))

    assert value["hosts"]["linbox"]["simulation"] == {
        "assets": {
            "artifact_kind": "simulation_asset",
            "current_link": "/home/robot/wam_data/simulation/assets/current",
            "releases_root": "/home/robot/wam_data/simulation/assets/releases",
            "staging_root": "/home/robot/wam_data/simulation/assets/.staging",
        },
        "code": {
            "artifact_kind": "simulation_code",
            "current_link": "/home/robot/wam_data/simulation/code/current",
            "releases_root": "/home/robot/wam_data/simulation/code/releases",
            "staging_root": "/home/robot/wam_data/simulation/code/.staging",
        },
        "results": {
            "artifact_kind": "simulation_result",
            "current_link": "/home/robot/wam_data/simulation/results/current",
            "releases_root": "/home/robot/wam_data/simulation/results/releases",
            "staging_root": "/home/robot/wam_data/simulation/results/.staging",
        },
    }
    assert simulation.load_simulation_policy() == simulation.AUDITED_SIMULATION_POLICY


def test_policy_rejects_a_substituted_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile_for(tmp_path, monkeypatch)
    value = json.loads(profile.read_text(encoding="utf-8"))
    value["hosts"]["linbox"]["simulation"]["results"]["releases_root"] = (
        tmp_path / "substitution"
    ).as_posix()
    profile.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(simulation.SimulationPolicyError, match="audited"):
        simulation.load_simulation_policy(profile)


@pytest.mark.parametrize(
    "namespace,kind,source_role",
    [
        ("assets", "simulation_asset", "linbox"),
        ("code", "simulation_code", "mac"),
        ("results", "simulation_result", "linbox"),
    ],
)
def test_stage_writes_a_completed_exact_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    namespace: str,
    kind: str,
    source_role: str,
) -> None:
    profile = _profile_for(tmp_path, monkeypatch)
    source = _source(tmp_path / "source")

    staged = simulation.stage_simulation_release(
        source,
        ["payload.bin"],
        namespace=namespace,
        nonce="2" * 32,
        profile_path=profile,
    )

    envelope = contract.verify_release(staged)
    marker = json.loads((staged / contract.COMPLETE_NAME).read_text(encoding="utf-8"))
    assert envelope["payload"]["artifact_kind"] == kind
    assert envelope["payload"]["source_role"] == source_role
    assert marker["release_id"] == envelope["release_id"]
    assert marker["payload_sha256"] == envelope["payload_sha256"]
    assert marker["target_role"] == "linbox"
    assert staged.parent == Path(
        simulation.load_simulation_policy(profile)[namespace]["staging_root"]
    )


def test_stage_rejects_source_symlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile_for(tmp_path, monkeypatch)
    source = _source(tmp_path / "source")
    (source / "payload.bin").unlink()
    (source / "payload.bin").symlink_to(tmp_path / "outside")

    with pytest.raises(contract.ArtifactError, match="symlink"):
        simulation.stage_simulation_release(
            source,
            ["payload.bin"],
            namespace="code",
            nonce="3" * 32,
            profile_path=profile,
        )


def test_asset_promotion_uses_only_the_isolated_asset_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, staged = _stage(
        tmp_path, monkeypatch, namespace="assets", nonce="4" * 32,
    )

    installed = simulation.promote_simulation_release(
        staged,
        namespace="assets",
        expected_current=None,
        profile_path=profile,
    )

    policy = simulation.load_simulation_policy(profile)
    asset_paths = policy["assets"]
    assert installed.parent == Path(asset_paths["releases_root"])
    assert installed.name.startswith("simulation_asset-")
    assert os.readlink(asset_paths["current_link"]) == "releases/" + installed.name
    for other_namespace in ("code", "results"):
        assert not os.path.lexists(policy[other_namespace]["current_link"])


def test_first_result_promotion_requires_absent_current_and_runs_post_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, staged = _stage(
        tmp_path, monkeypatch, namespace="results", nonce="a" * 32,
    )
    verified = []

    def verify_result(release: Path) -> None:
        envelope = contract.verify_release(release)
        assert envelope["payload"]["artifact_kind"] == "simulation_result"
        assert envelope["payload"]["source_role"] == "linbox"
        verified.append(release)

    installed = simulation.promote_simulation_release(
        staged,
        namespace="results",
        expected_current=None,
        profile_path=profile,
        post_verify=verify_result,
    )

    policy = simulation.load_simulation_policy(profile)
    result_paths = policy["results"]
    assert verified == [installed]
    assert installed.parent == Path(result_paths["releases_root"])
    assert os.readlink(result_paths["current_link"]) == "releases/" + installed.name

    source = _source(tmp_path / "result-source-2", b"new evidence\n")
    second = simulation.stage_simulation_release(
        source,
        ["payload.bin"],
        namespace="results",
        nonce="b" * 32,
        profile_path=profile,
    )
    with pytest.raises(contract.ConflictError, match="compare-and-swap"):
        simulation.promote_simulation_release(
            second,
            namespace="results",
            expected_current=None,
            profile_path=profile,
            post_verify=verify_result,
        )

    assert verified == [installed]
    assert os.readlink(result_paths["current_link"]) == "releases/" + installed.name


@pytest.mark.parametrize(
    "source_role,target_role,error",
    [
        ("mac", "linbox", "source role"),
        ("linbox", "h200", "target role"),
    ],
)
def test_asset_promotion_rejects_wrong_role_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_role: str,
    target_role: str,
    error: str,
) -> None:
    profile = _profile_for(tmp_path, monkeypatch)
    source = _source(tmp_path / "asset-source")
    envelope = contract.build_manifest(
        source,
        ["payload.bin"],
        artifact_kind="simulation_asset",
        source_role=source_role,
    )
    staging_root = Path(
        simulation.load_simulation_policy(profile)["assets"]["staging_root"]
    )
    staging_root.mkdir(parents=True)
    staged = staging_root / (
        "incoming-%s-%s" % (envelope["release_id"], "7" * 32)
    )
    contract.write_release(
        source, envelope, staged, target_role=target_role,
    )

    with pytest.raises(contract.VerificationError, match=error):
        simulation.promote_simulation_release(
            staged,
            namespace="assets",
            expected_current=None,
            profile_path=profile,
        )


def test_failed_asset_post_verification_restores_the_verified_asset_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, first = _stage(
        tmp_path, monkeypatch, namespace="assets", nonce="5" * 32,
    )
    first_installed = simulation.promote_simulation_release(
        first,
        namespace="assets",
        expected_current=None,
        profile_path=profile,
    )
    asset_policy = simulation.load_simulation_policy(profile)["assets"]
    current = Path(asset_policy["current_link"])
    previous_target = os.readlink(current)
    source = _source(tmp_path / "asset-source-2", b"new asset\n")
    second = simulation.stage_simulation_release(
        source,
        ["payload.bin"],
        namespace="assets",
        nonce="6" * 32,
        profile_path=profile,
    )

    def fail(_release: Path) -> None:
        raise RuntimeError("asset post verification failed")

    with pytest.raises(RuntimeError, match="asset post verification"):
        simulation.promote_simulation_release(
            second,
            namespace="assets",
            expected_current=previous_target,
            profile_path=profile,
            post_verify=fail,
        )

    assert os.readlink(current) == previous_target
    assert current.resolve() == first_installed.resolve()


@pytest.mark.parametrize("namespace", ["assets", "code"])
def test_stage_rejects_a_symlinked_audited_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, namespace: str,
) -> None:
    profile = _profile_for(tmp_path, monkeypatch)
    source = _source(tmp_path / "source")
    policy = simulation.load_simulation_policy(profile)
    staging = Path(policy[namespace]["staging_root"])
    staging.parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    staging.symlink_to(outside, target_is_directory=True)

    with pytest.raises(simulation.SimulationPolicyError, match="symlink"):
        simulation.stage_simulation_release(
            source,
            ["payload.bin"],
            namespace=namespace,
            nonce="4" * 32,
            profile_path=profile,
        )


def test_stage_publication_is_atomic_no_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile_for(tmp_path, monkeypatch)
    source = _source(tmp_path / "source")
    real_rename = contract._rename_noreplace_at

    def race(
        source_descriptor: int,
        source_name: str,
        destination_descriptor: int,
        destination_name: str,
    ) -> None:
        os.mkdir(destination_name, dir_fd=destination_descriptor)
        destination = contract._open_directory_at(
            destination_descriptor, destination_name,
        )
        try:
            descriptor = os.open(
                "interloper",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o644,
                dir_fd=destination,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write("preserve")
        finally:
            os.close(destination)
        real_rename(
            source_descriptor,
            source_name,
            destination_descriptor,
            destination_name,
        )

    monkeypatch.setattr(contract, "_rename_noreplace_at", race)
    with pytest.raises(contract.ConflictError):
        simulation.stage_simulation_release(
            source,
            ["payload.bin"],
            namespace="code",
            nonce="5" * 32,
            profile_path=profile,
        )

    staging = Path(simulation.load_simulation_policy(profile)["code"]["staging_root"])
    destinations = [path for path in staging.iterdir() if path.is_dir()]
    assert len(destinations) == 1
    assert (destinations[0] / "interloper").read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize("namespace", ["assets", "code"])
def test_stage_fails_closed_if_staging_root_is_swapped_after_no_follow_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    namespace: str,
) -> None:
    profile = _profile_for(tmp_path, monkeypatch)
    source = _source(tmp_path / "source")
    policy = simulation.load_simulation_policy(profile)[namespace]
    staging = Path(policy["staging_root"])
    staging.mkdir(parents=True)
    parked = staging.with_name(".staging-parked-by-race")
    outside = tmp_path / "outside"
    outside.mkdir()
    real_write = contract.write_release_at

    def swap_then_write(*args, **kwargs):
        staging.rename(parked)
        staging.symlink_to(outside, target_is_directory=True)
        return real_write(*args, **kwargs)

    monkeypatch.setattr(contract, "write_release_at", swap_then_write)

    with pytest.raises(simulation.SimulationPolicyError, match="identity changed"):
        simulation.stage_simulation_release(
            source,
            ["payload.bin"],
            namespace=namespace,
            nonce="a" * 32,
            profile_path=profile,
        )

    assert list(outside.iterdir()) == []
    assert list(parked.glob("incoming-*")) == []


@pytest.mark.parametrize("mutation", ["extra", "payload", "marker"])
def test_promotion_rejects_non_exact_or_unbound_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    profile, staged = _stage(tmp_path, monkeypatch)
    if mutation == "extra":
        (staged / "extra").write_text("unexpected", encoding="utf-8")
    elif mutation == "payload":
        (staged / "payload.bin").write_bytes(b"changed")
    else:
        marker_path = staged / contract.COMPLETE_NAME
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["payload_sha256"] = "0" * 64
        marker_path.write_bytes(contract.canonical_json_bytes(marker))

    with pytest.raises(contract.VerificationError):
        simulation.promote_simulation_release(
            staged,
            namespace="code",
            expected_current=None,
            profile_path=profile,
        )


def test_promotion_rejects_a_staged_path_outside_the_audited_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, staged = _stage(tmp_path, monkeypatch)
    escaped = tmp_path / staged.name
    staged.rename(escaped)

    with pytest.raises(simulation.SimulationPolicyError, match="direct child"):
        simulation.promote_simulation_release(
            escaped,
            namespace="code",
            expected_current=None,
            profile_path=profile,
        )


@pytest.mark.parametrize("namespace", ["assets", "code"])
def test_stale_cas_rejects_activation_and_preserves_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, namespace: str,
) -> None:
    profile, first = _stage(
        tmp_path, monkeypatch, namespace=namespace, nonce="6" * 32,
    )
    installed = simulation.promote_simulation_release(
        first,
        namespace=namespace,
        expected_current=None,
        profile_path=profile,
    )
    policy = simulation.load_simulation_policy(profile)[namespace]
    current = Path(policy["current_link"])
    first_target = os.readlink(current)
    source = _source(tmp_path / "source-2", b"next\n")
    second = simulation.stage_simulation_release(
        source,
        ["payload.bin"],
        namespace=namespace,
        nonce="7" * 32,
        profile_path=profile,
    )

    with pytest.raises(contract.ConflictError, match="compare-and-swap"):
        simulation.promote_simulation_release(
            second,
            namespace=namespace,
            expected_current="releases/stale-release",
            profile_path=profile,
        )

    assert current.resolve() == installed.resolve()
    assert os.readlink(current) == first_target


def test_failed_post_verification_rolls_back_only_simulation_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, first = _stage(tmp_path, monkeypatch, nonce="8" * 32)
    first_installed = simulation.promote_simulation_release(
        first,
        namespace="code",
        expected_current=None,
        profile_path=profile,
    )
    policy = simulation.load_simulation_policy(profile)["code"]
    current = Path(policy["current_link"])
    previous_target = os.readlink(current)
    source = _source(tmp_path / "source-2", b"next\n")
    second = simulation.stage_simulation_release(
        source,
        ["payload.bin"],
        namespace="code",
        nonce="9" * 32,
        profile_path=profile,
    )

    def fail(_release: Path) -> None:
        raise RuntimeError("post verification failed")

    with pytest.raises(RuntimeError, match="post verification"):
        simulation.promote_simulation_release(
            second,
            namespace="code",
            expected_current=previous_target,
            profile_path=profile,
            post_verify=fail,
        )

    assert os.readlink(current) == previous_target
    assert current.resolve() == first_installed.resolve()
    checked_in = json.loads(PROFILE.read_text(encoding="utf-8"))
    assert checked_in["hosts"]["linbox"]["production"]["current_link"] == (
        "/home/robot/Desktop/stethoscope/.wam-current"
    )
    assert checked_in["hosts"]["linbox"]["perception_runtime"]["current_link"] == (
        "/home/robot/wam_data/perception_runtime/current"
    )


def test_unsafe_previous_release_symlink_is_cleared_and_never_restored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, candidate = _stage(tmp_path, monkeypatch, nonce="d" * 32)
    policy = simulation.load_simulation_policy(profile)["code"]
    releases = Path(policy["releases_root"])
    releases.mkdir(parents=True)
    outside = tmp_path / "outside-release"
    outside.mkdir()
    unsafe_id = "simulation_code-" + "e" * 20
    (releases / unsafe_id).symlink_to(outside, target_is_directory=True)
    current = Path(policy["current_link"])
    current.symlink_to("releases/" + unsafe_id)
    callbacks = []

    def fail(_release: Path) -> None:
        callbacks.append(True)
        raise RuntimeError("post verification failed")

    with pytest.raises(simulation.SimulationPolicyError, match="previous current"):
        simulation.promote_simulation_release(
            candidate,
            namespace="code",
            expected_current="releases/" + unsafe_id,
            profile_path=profile,
            post_verify=fail,
        )

    assert callbacks == []
    assert not os.path.lexists(current)
    assert (releases / unsafe_id).is_symlink()
    assert list(outside.iterdir()) == []


def test_rollback_clears_current_if_verified_previous_child_is_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, first = _stage(tmp_path, monkeypatch, nonce="e" * 32)
    previous_release = simulation.promote_simulation_release(
        first,
        namespace="code",
        expected_current=None,
        profile_path=profile,
    )
    policy = simulation.load_simulation_policy(profile)["code"]
    current = Path(policy["current_link"])
    previous_target = os.readlink(current)
    source = _source(tmp_path / "source-2", b"next\n")
    second = simulation.stage_simulation_release(
        source,
        ["payload.bin"],
        namespace="code",
        nonce="f" * 32,
        profile_path=profile,
    )
    parked = previous_release.with_name(previous_release.name + "-parked-by-race")
    outside = tmp_path / "outside-release"
    outside.mkdir()

    def replace_previous_then_fail(_release: Path) -> None:
        previous_release.rename(parked)
        previous_release.symlink_to(outside, target_is_directory=True)
        raise RuntimeError("post verification failed")

    with pytest.raises(RuntimeError, match="post verification"):
        simulation.promote_simulation_release(
            second,
            namespace="code",
            expected_current=previous_target,
            profile_path=profile,
            post_verify=replace_previous_then_fail,
        )

    assert not os.path.lexists(current)
    assert previous_release.is_symlink()
    assert parked.is_dir()
    assert list(outside.iterdir()) == []


def test_promotion_fails_closed_if_releases_root_is_swapped_during_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, first = _stage(tmp_path, monkeypatch, nonce="b" * 32)
    simulation.promote_simulation_release(
        first,
        namespace="code",
        expected_current=None,
        profile_path=profile,
    )
    policy = simulation.load_simulation_policy(profile)["code"]
    current = Path(policy["current_link"])
    source = _source(tmp_path / "source-2", b"next\n")
    second = simulation.stage_simulation_release(
        source,
        ["payload.bin"],
        namespace="code",
        nonce="c" * 32,
        profile_path=profile,
    )
    second_id = contract.verify_release(second)["release_id"]
    releases = Path(policy["releases_root"])
    parked = releases.with_name("releases-parked-by-race")
    outside = tmp_path / "outside"
    outside.mkdir()
    real_write = contract.write_release_at

    def swap_then_write(*args, **kwargs):
        releases.rename(parked)
        releases.symlink_to(outside, target_is_directory=True)
        return real_write(*args, **kwargs)

    monkeypatch.setattr(contract, "write_release_at", swap_then_write)

    with pytest.raises(simulation.SimulationPolicyError, match="identity changed"):
        simulation.promote_simulation_release(
            second,
            namespace="code",
            expected_current=os.readlink(current),
            profile_path=profile,
        )

    assert list(outside.iterdir()) == []
    assert not os.path.lexists(current)
    assert not (parked / second_id).exists()
