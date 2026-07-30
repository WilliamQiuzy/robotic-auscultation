from __future__ import annotations

import base64
import copy
import hashlib
import os
from pathlib import Path
import shutil
import socket
import stat
import sys
import tempfile

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent))

import perception_environment_manifest as manifest  # noqa: E402


def _record_hash(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


class _Distribution:
    def __init__(
        self,
        site_packages: Path,
        *,
        name: str = "Demo_Pkg",
        version: str = "1.2.3",
    ) -> None:
        self.metadata = {"Name": name}
        self.version = version
        self._site_packages = site_packages
        normalized = name.lower().replace("_", "-")
        self._path = site_packages / f"{normalized}-{version}.dist-info"
        self._path.mkdir(parents=True)

    def locate_file(self, relative: str) -> Path:
        return self._site_packages / relative


def _distribution_fixture(root: Path) -> tuple[_Distribution, bytes, bytes]:
    site = root / "lib/python3.11/site-packages"
    site.mkdir(parents=True)
    distribution = _Distribution(site)
    package = site / "demo_pkg"
    package.mkdir()
    hashed = b"VERSION = 1\n"
    unhashed = b"generated metadata\n"
    (package / "__init__.py").write_bytes(hashed)
    (distribution._path / "INSTALLER").write_bytes(unhashed)
    record_rows = (
        f"demo_pkg/__init__.py,{_record_hash(hashed)},{len(hashed)}\n"
        f"{distribution._path.name}/INSTALLER,,\n"
        "missing.bin,sha256=AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA,1\n"
        f"{distribution._path.name}/RECORD,,\n"
    )
    (distribution._path / "RECORD").write_text(record_rows, encoding="utf-8")
    return distribution, hashed, unhashed


def test_record_observation_keeps_every_expected_and_actual_claim(tmp_path: Path) -> None:
    root = tmp_path / "env"
    distribution, hashed, unhashed = _distribution_fixture(root)

    observation = manifest.collect_record_observation(
        [distribution], environment_root=root
    )

    assert observation["algorithm"] == manifest.RECORD_OBSERVATION_ALGORITHM
    assert observation["distribution_count"] == 1
    assert observation["claim_count"] == 4
    assert observation["status_counts"] == {
        "exact": 1,
        "metadata_mismatch": 0,
        "missing": 1,
        "omitted_generated_pyc": 0,
        "record_unhashed": 2,
        "relocation_mismatch": 0,
    }
    rows = observation["distributions"][0]["claims"]
    assert [row["record_index"] for row in rows] == [1, 2, 3, 4]
    assert rows[0] == {
        "actual": {
            "sha256": hashlib.sha256(hashed).hexdigest(),
            "size": len(hashed),
        },
        "expected": {
            "hash_field": _record_hash(hashed),
            "sha256": hashlib.sha256(hashed).hexdigest(),
            "size": len(hashed),
            "size_field": str(len(hashed)),
        },
        "installed_path": "lib/python3.11/site-packages/demo_pkg/__init__.py",
        "record_index": 1,
        "record_path": "demo_pkg/__init__.py",
        "status": "exact",
    }
    assert rows[1]["expected"] == {
        "hash_field": "",
        "sha256": None,
        "size": None,
        "size_field": "",
    }
    assert rows[1]["actual"] == {
        "sha256": hashlib.sha256(unhashed).hexdigest(),
        "size": len(unhashed),
    }
    assert rows[1]["status"] == "record_unhashed"
    assert rows[2]["actual"] == {"sha256": None, "size": None}
    assert rows[2]["status"] == "missing"
    assert len(observation["sha256"]) == 64


def test_duplicate_record_rows_are_preserved_with_distinct_expected_metadata(
    tmp_path: Path,
) -> None:
    root = tmp_path / "env"
    distribution, hashed, _unhashed = _distribution_fixture(root)
    record = distribution._path / "RECORD"
    record.write_text(
        "demo_pkg/__init__.py,,\n"
        f"demo_pkg/__init__.py,{_record_hash(hashed)},{len(hashed)}\n",
        encoding="utf-8",
    )

    observation = manifest.collect_record_observation(
        [distribution], environment_root=root
    )

    claims = observation["distributions"][0]["claims"]
    assert len(claims) == 2
    assert claims[0]["installed_path"] == claims[1]["installed_path"]
    assert claims[0]["status"] == "record_unhashed"
    assert claims[1]["status"] == "exact"
    assert claims[0]["expected"] != claims[1]["expected"]


def _relocation_fixture(tmp_path: Path, *, installed_path: str = "../../../bin/demo"):
    source = tmp_path / "source"
    candidate = tmp_path / "candidate"
    source_site = source / "lib/python3.11/site-packages"
    candidate_site = candidate / "lib/python3.11/site-packages"
    source_site.mkdir(parents=True)
    candidate_site.mkdir(parents=True)
    distribution = _Distribution(candidate_site)
    source_script = (
        b"#!" + os.fsencode(source) + b"/bin/python3.11\n"
        b"from demo_pkg import main\nraise SystemExit(main())\n"
    )
    candidate_script = source_script.replace(
        os.fsencode(source), os.fsencode(candidate), 1
    )
    (source / "bin").mkdir()
    (candidate / "bin").mkdir(exist_ok=True)
    (source / "bin/demo").write_bytes(source_script)
    (candidate / "bin/demo").write_bytes(candidate_script)
    record_bytes = (
        f"{installed_path},{_record_hash(source_script)},{len(source_script)}\n"
    ).encode("utf-8")
    (distribution._path / "RECORD").write_bytes(record_bytes)
    source_record = source / distribution._path.relative_to(candidate) / "RECORD"
    source_record.parent.mkdir(parents=True)
    source_record.write_bytes(record_bytes)
    return source, candidate, distribution, source_script, candidate_script


def test_console_script_relocation_requires_only_exact_shebang_prefix_change(
    tmp_path: Path,
) -> None:
    source, candidate, distribution, source_bytes, candidate_bytes = (
        _relocation_fixture(tmp_path)
    )

    observation = manifest.collect_record_observation(
        [distribution],
        environment_root=candidate,
        relocation_source_root=source,
    )

    claim = observation["distributions"][0]["claims"][0]
    assert claim["status"] == "relocation_mismatch"
    assert claim["actual"] == {
        "sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "size": len(candidate_bytes),
    }
    assert claim["relocation"] == {
        "interpreter_suffix": "/bin/python3.11",
        "new_prefix": str(candidate),
        "old_prefix": str(source),
        "source_record_index": 1,
        "source_record_sha256": hashlib.sha256(
            (source / distribution._path.relative_to(candidate) / "RECORD").read_bytes()
        ).hexdigest(),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_size": len(source_bytes),
    }


def test_console_script_relocation_accepts_a_recipe_bound_source_already_relocated(
    tmp_path: Path,
) -> None:
    source, candidate, distribution, source_bytes, _candidate_bytes = (
        _relocation_fixture(tmp_path)
    )
    original_install_bytes = source_bytes.replace(
        os.fsencode(source), b"/opt/original-install", 1,
    )
    record_bytes = (
        f"../../../bin/demo,{_record_hash(original_install_bytes)},"
        f"{len(original_install_bytes)}\n"
    ).encode("utf-8")
    (distribution._path / "RECORD").write_bytes(record_bytes)
    source_record = source / distribution._path.relative_to(candidate) / "RECORD"
    source_record.write_bytes(record_bytes)

    observation = manifest.collect_record_observation(
        [distribution],
        environment_root=candidate,
        relocation_source_root=source,
    )
    claim = observation["distributions"][0]["claims"][0]

    assert claim["status"] == "relocation_mismatch"
    assert claim["relocation"]["source_sha256"] == hashlib.sha256(
        source_bytes
    ).hexdigest()
    assert claim["relocation"]["source_sha256"] != claim["expected"]["sha256"]
    assert manifest.validate_record_observation(
        observation, require_resolved=True,
    ) == observation


@pytest.mark.parametrize(
    "mutation",
    [
        "body", "wrong-prefix", "source-bytes-drift",
        "source-record-row-drift", "outside-bin",
    ],
)
def test_console_script_relocation_rejects_any_weaker_equivalence(
    tmp_path: Path, mutation: str,
) -> None:
    installed_path = (
        "demo_pkg/tool.py" if mutation == "outside-bin" else "../../../bin/demo"
    )
    source, candidate, distribution, source_bytes, candidate_bytes = (
        _relocation_fixture(tmp_path, installed_path=installed_path)
    )
    if mutation == "body":
        (candidate / "bin/demo").write_bytes(candidate_bytes + b"# changed\n")
    elif mutation == "wrong-prefix":
        (candidate / "bin/demo").write_bytes(
            candidate_bytes.replace(os.fsencode(candidate), b"/other/root", 1)
        )
    elif mutation == "source-bytes-drift":
        (source / "bin/demo").write_bytes(source_bytes + b"# source drift\n")
    elif mutation == "source-record-row-drift":
        source_record = (
            source / distribution._path.relative_to(candidate) / "RECORD"
        )
        source_record.write_text(
            f"{installed_path},{_record_hash(source_bytes)},{len(source_bytes) + 1}\n",
            encoding="utf-8",
        )
    else:
        package = candidate / "lib/python3.11/site-packages/demo_pkg"
        package.mkdir()
        (package / "tool.py").write_bytes(candidate_bytes)
        source_package = source / "lib/python3.11/site-packages/demo_pkg"
        source_package.mkdir()
        (source_package / "tool.py").write_bytes(source_bytes)

    observation = manifest.collect_record_observation(
        [distribution],
        environment_root=candidate,
        relocation_source_root=source,
    )

    claim = observation["distributions"][0]["claims"][0]
    assert claim["status"] == "metadata_mismatch"
    assert "relocation" not in claim


def test_only_exact_audited_generated_pyc_can_be_required_absent(
    tmp_path: Path,
) -> None:
    root = tmp_path / "env"
    site = root / "lib/python3.11/site-packages"
    site.mkdir(parents=True)
    distribution = _Distribution(site, name="numpy", version="1.26.4")
    source = site / "numpy/distutils/conv_template.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"SOURCE = 1\n")
    omitted = manifest.GENERATED_PYC_OMISSION
    other = "lib/python3.11/site-packages/numpy/other.cpython-311.pyc"
    (distribution._path / "RECORD").write_text(
        "numpy/distutils/__pycache__/conv_template.cpython-311.pyc,sha256=abc,9\n"
        "numpy/other.cpython-311.pyc,sha256=def,10\n",
        encoding="utf-8",
    )
    evidence = manifest.generated_pyc_omission_evidence(
        path=omitted,
        source_path="lib/python3.11/site-packages/numpy/distutils/conv_template.py",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        inspection_sha256="a" * 64,
    )

    observation = manifest.collect_record_observation(
        [distribution],
        environment_root=root,
        omitted_generated_pyc={omitted: evidence},
    )

    claims = observation["distributions"][0]["claims"]
    assert claims[0]["installed_path"] == omitted
    assert claims[0]["status"] == "omitted_generated_pyc"
    assert claims[0]["omission_evidence_sha256"] == hashlib.sha256(
        manifest.canonical_json_bytes(evidence)
    ).hexdigest()
    assert claims[1]["installed_path"] == other
    assert claims[1]["status"] == "missing"


def test_tree_manifest_covers_directories_regular_files_symlinks_and_modes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "env"
    package = root / "lib/pkg"
    package.mkdir(parents=True)
    binary = package / "tool.bin"
    binary.write_bytes(b"binary-v1")
    binary.chmod(0o550)
    link = root / "bin-tool"
    link.symlink_to("lib/pkg/tool.bin")

    first = manifest.collect_environment_tree(root)

    assert first["algorithm"] == manifest.ENVIRONMENT_TREE_ALGORITHM
    assert first["directories"] == 3
    assert first["regular_files"] == 1
    assert first["symlinks"] == 1
    rows = {row["path"]: row for row in first["entries"]}
    assert rows["."] == {"kind": "directory", "mode": "0755", "path": "."}
    assert rows["lib/pkg/tool.bin"] == {
        "kind": "regular",
        "mode": "0550",
        "path": "lib/pkg/tool.bin",
        "sha256": hashlib.sha256(b"binary-v1").hexdigest(),
        "size": 9,
    }
    assert rows["bin-tool"] == {
        "kind": "symlink",
        "mode": format(stat.S_IMODE(link.lstat().st_mode), "04o"),
        "path": "bin-tool",
        "resolved_path": "lib/pkg/tool.bin",
        "target": "lib/pkg/tool.bin",
    }

    binary.chmod(0o750)
    binary.write_bytes(b"binary-v2")
    binary.chmod(0o550)
    second = manifest.collect_environment_tree(root)
    assert second["sha256"] != first["sha256"]
    binary.chmod(0o750)
    binary.write_bytes(b"binary-v1")
    binary.chmod(0o500)
    third = manifest.collect_environment_tree(root)
    assert third["sha256"] != first["sha256"]


def test_tree_policy_has_only_two_self_references_and_requires_pyc_absent(
    tmp_path: Path,
) -> None:
    root = tmp_path / "env"
    root.mkdir()
    for relative in manifest.SELF_REFERENCE_EXCLUSIONS:
        (root / relative).write_text("self-reference\n", encoding="utf-8")
    tree = manifest.collect_environment_tree(root)
    assert tree["exclusions"] == {
        "required_absent_generated_cache": [manifest.GENERATED_PYC_OMISSION],
        "self_referential": list(manifest.SELF_REFERENCE_EXCLUSIONS),
    }
    assert tree["regular_files"] == 0
    assert all(
        row["path"] not in manifest.SELF_REFERENCE_EXCLUSIONS
        for row in tree["entries"]
    )

    omitted = root / manifest.GENERATED_PYC_OMISSION
    omitted.parent.mkdir(parents=True)
    omitted.write_bytes(b"must be absent")
    with pytest.raises(manifest.ManifestError, match="required absent"):
        manifest.collect_environment_tree(root)


@pytest.mark.parametrize("kind", ["absolute", "escape", "dangling", "fifo", "socket"])
def test_tree_manifest_rejects_unsafe_or_nonfilesystem_payloads(
    tmp_path: Path, kind: str,
) -> None:
    short_root = None
    if kind == "socket":
        short_root = Path(tempfile.mkdtemp(prefix="wam-tree-", dir="/tmp"))
        root = short_root
    else:
        root = tmp_path / "env"
        root.mkdir()
    candidate = root / "unsafe"
    held_socket = None
    if kind == "absolute":
        target = root / "target"
        target.write_bytes(b"target")
        candidate.symlink_to(target)
    elif kind == "escape":
        outside = tmp_path / "outside"
        outside.write_bytes(b"outside")
        candidate.symlink_to("../outside")
    elif kind == "dangling":
        candidate.symlink_to("missing")
    elif kind == "fifo":
        os.mkfifo(candidate)
    else:
        held_socket = socket.socket(socket.AF_UNIX)
        held_socket.bind(str(candidate))
    try:
        with pytest.raises(manifest.ManifestError, match="symlink|non-regular|FIFO|socket"):
            manifest.collect_environment_tree(root)
    finally:
        if held_socket is not None:
            held_socket.close()
        if short_root is not None:
            shutil.rmtree(short_root)


def test_tree_manifest_rejects_regular_file_hardlinks(tmp_path: Path) -> None:
    root = tmp_path / "env"
    root.mkdir()
    first = root / "first"
    first.write_bytes(b"same inode")
    os.link(first, root / "second")

    with pytest.raises(manifest.ManifestError, match="hardlink"):
        manifest.collect_environment_tree(root)

    source_snapshot = manifest.collect_environment_tree(
        root, allow_hardlinks=True
    )
    assert source_snapshot["hardlinks"] == [{
        "link_count": 2,
        "paths": ["first", "second"],
    }]

    (root / "second").unlink()
    (root / "second").write_bytes(b"same inode")
    copied = manifest.collect_environment_tree(root, allow_hardlinks=True)
    assert copied["hardlinks"] == []
    assert copied["sha256"] != source_snapshot["sha256"]


def test_tree_manifest_rejects_hardlink_bytes_changing_between_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "env"
    root.mkdir()
    first = root / "first"
    second = root / "second"
    first.write_bytes(b"before")
    os.link(first, second)
    original_read_regular = manifest._read_regular

    def read_then_mutate(path: Path, label: str):
        data, info = original_read_regular(path, label)
        if path == first:
            second.write_bytes(b"after-and-different")
        return data, info

    monkeypatch.setattr(manifest, "_read_regular", read_then_mutate)

    with pytest.raises(manifest.ManifestError, match="hardlink identity changed"):
        manifest.collect_environment_tree(root, allow_hardlinks=True)


def test_record_validator_recomputes_every_row_and_rejects_unresolved_strong_status(
    tmp_path: Path,
) -> None:
    root = tmp_path / "env"
    distribution, _hashed, _unhashed = _distribution_fixture(root)
    observation = manifest.collect_record_observation(
        [distribution], environment_root=root
    )

    assert manifest.validate_record_observation(observation) == observation
    with pytest.raises(manifest.ManifestError, match="unresolved RECORD statuses"):
        manifest.validate_record_observation(observation, require_resolved=True)

    tampered = copy.deepcopy(observation)
    tampered["distributions"][0]["claims"][0]["actual"]["size"] += 1
    with pytest.raises(manifest.ManifestError, match="fingerprint|contradicts"):
        manifest.validate_record_observation(tampered)


def test_tree_validator_recomputes_manifest_and_record_tree_crosscheck(
    tmp_path: Path,
) -> None:
    root = tmp_path / "env"
    distribution, _hashed, _unhashed = _distribution_fixture(root)
    missing = root / "lib/python3.11/site-packages/missing.bin"
    missing.write_bytes(b"x")
    record = distribution._path / "RECORD"
    rows = record.read_text(encoding="utf-8").splitlines()
    rows[2] = f"missing.bin,{_record_hash(b'x')},1"
    record.write_text("\n".join(rows) + "\n", encoding="utf-8")
    observation = manifest.collect_record_observation(
        [distribution], environment_root=root
    )
    tree = manifest.collect_environment_tree(root)

    assert manifest.validate_environment_tree(tree) == tree
    manifest.cross_check_record_tree(observation, tree)

    tampered = copy.deepcopy(tree)
    regular = next(row for row in tampered["entries"] if row["kind"] == "regular")
    regular["mode"] = "0777"
    with pytest.raises(manifest.ManifestError, match="fingerprint"):
        manifest.validate_environment_tree(tampered)

    drifted = copy.deepcopy(tree)
    installed = next(
        row for row in drifted["entries"]
        if row["path"] == "lib/python3.11/site-packages/demo_pkg/__init__.py"
    )
    installed["sha256"] = "0" * 64
    with pytest.raises(manifest.ManifestError, match="RECORD/tree"):
        manifest.cross_check_record_tree(observation, drifted)
