"""Fail-closed tests for the official OpenArm/Isaac asset manifest."""

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import pytest

# Import ``tracking`` without importing the heavier ``ausc`` package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tracking import official_asset_contract as contract  # noqa: E402


CHECKOUT_PATH = "openarm_description"
URDF_PATH = (
    "openarm_description/assets/robot/openarm_v1.0/urdf/example/v1.urdf"
)
HUMAN_USD_PATH = "human/male_adult_medical_01.usd"
TEXTURE_PATHS = (
    "human/textures/male_adult_medical_01_BaseColor.png",
    "human/textures/male_adult_medical_01_Normal.png",
    "human/textures/male_adult_medical_01_ORM.png",
)
DEPENDENCY_REPORT_PATH = "human/compute_all_dependencies.json"
LICENSE_PAGE_PATH = "human/licenses/isaac_sim_additional_license.html"
LICENSE_EVIDENCE_PATH = "human/licenses/license_evidence.json"
ADDITIONAL_LICENSE_URL = (
    "https://docs.isaacsim.omniverse.nvidia.com/latest/common/"
    "license-isaac-sim-additional.html"
)
OMNIPBR_RECORD = {
    "authored_uri": "OmniPBR.mdl",
    "resolved_path": "",
    "resolution_kind": "kit_mdl_search_path",
    "runtime_provenance": {
        "distribution": "isaacsim-kernel",
        "record_hash": "sha256=aFqCwYs4HxzSmDFJDkw1bmHb6sLYaAO6vBFc1qSoRds",
        "record_size": 15591,
        "version": "5.1.0.0",
    },
    "runtime_relative_path": "isaacsim/kit/mdl/core/Base/OmniPBR.mdl",
    "sha256": "685a82c18b381f1cd29831490e4c356e61dbeac2d86803babc115cd6a4a845db",
    "size_bytes": 15591,
}
TEXTURE_AUTHORED_URIS = {
    TEXTURE_PATHS[0]: "./textures/male_adult_medical_01_BaseColor.png",
    TEXTURE_PATHS[1]: "./textures/male_adult_medical_01_Normal.png",
    TEXTURE_PATHS[2]: "./textures/male_adult_medical_01_ORM.png",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _write(root: Path, relative: str, data: bytes) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _official_shape_urdf() -> bytes:
    robot = ET.Element("robot", {"name": "openarm"})
    links = []
    for index in range(24):
        links.append(ET.SubElement(robot, "link", {"name": f"link_{index}"}))

    for reference_index in range(42):
        visual = ET.SubElement(links[reference_index % len(links)], "visual")
        geometry = ET.SubElement(visual, "geometry")
        mesh_index = reference_index % 20
        scale = "0.001 -0.001 0.001" if reference_index % 3 == 0 else "0.001 0.001 0.001"
        ET.SubElement(
            geometry,
            "mesh",
            {
                "filename": (
                    "package://openarm_description/assets/robot/openarm_v1.0/mesh/"
                    f"test/mesh_{mesh_index:02d}.dae"
                ),
                "scale": scale,
            },
        )

    joint_names = [f"openarm_left_joint{index}" for index in range(1, 8)]
    joint_names += [f"openarm_right_joint{index}" for index in range(1, 8)]
    joint_names += [f"fixed_joint_{index}" for index in range(1, 10)]
    for index, name in enumerate(joint_names):
        joint_type = "revolute" if index < 14 else "fixed"
        joint = ET.SubElement(robot, "joint", {"name": name, "type": joint_type})
        ET.SubElement(joint, "parent", {"link": f"link_{index}"})
        ET.SubElement(joint, "child", {"link": f"link_{index + 1}"})
    return ET.tostring(robot, encoding="utf-8")


def _mesh_paths(urdf: bytes) -> list[str]:
    root = ET.fromstring(urdf)
    prefix = "package://openarm_description/"
    return sorted(
        {
            f"openarm_description/{mesh.attrib['filename'][len(prefix):]}"
            for mesh in root.iter("mesh")
        }
    )


def _git_evidence(required_paths: list[str]) -> dict[str, object]:
    return {
        "detached_head": True,
        "head": contract.ENACTIC_COMMIT,
        "remote_url": contract.ENACTIC_REPOSITORY_URL,
        "status_porcelain": "",
        "tracked_paths": sorted(required_paths),
    }


@pytest.fixture
def asset_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "assets"
    root.mkdir()
    urdf = _official_shape_urdf()
    _write(root, URDF_PATH, urdf)
    for mesh_path in _mesh_paths(urdf):
        _write(root, mesh_path, f"mesh:{mesh_path}".encode("utf-8"))

    human_usd = b"#usda 1.0\n(def Xform \"MedicalHuman\" {})\n"
    textures = {
        TEXTURE_PATHS[0]: b"base-color-texture",
        TEXTURE_PATHS[1]: b"normal-texture",
        TEXTURE_PATHS[2]: b"orm-texture",
    }
    _write(root, HUMAN_USD_PATH, human_usd)
    for path, data in textures.items():
        _write(root, path, data)

    monkeypatch.setattr(contract, "OFFICIAL_URDF_SHA256", _sha(urdf), raising=False)
    monkeypatch.setattr(
        contract,
        "OFFICIAL_URDF_CANONICAL_XML_SHA256",
        _sha(ET.tostring(ET.fromstring(urdf), encoding="utf-8")),
        raising=False,
    )
    monkeypatch.setattr(contract, "HUMAN_USD_SHA256", _sha(human_usd), raising=False)
    monkeypatch.setattr(contract, "HUMAN_USD_SIZE", len(human_usd), raising=False)
    monkeypatch.setattr(
        contract,
        "NVIDIA_TEXTURE_SHA256",
        {path: _sha(data) for path, data in textures.items()},
        raising=False,
    )
    monkeypatch.setattr(
        contract,
        "NVIDIA_TEXTURE_SIZE",
        {path: len(data) for path, data in textures.items()},
        raising=False,
    )

    report = {
        "asset_count": 4,
        "asset_url": contract.NVIDIA_MEDICAL_HUMAN_URL,
        "assets": [
            copy.deepcopy(OMNIPBR_RECORD),
            *[
                {
                    "authored_uri": TEXTURE_AUTHORED_URIS[path],
                    "resolution_kind": "usd_relative",
                    "resolved_path": path,
                    "sha256": _sha(textures[path]),
                    "size_bytes": len(textures[path]),
                }
                for path in TEXTURE_PATHS
            ],
        ],
        "isaac_version": "5.1.0.0",
        "layer_count": 1,
        "resolver": "UsdUtils.ComputeAllDependencies",
        "root_layer": {
            "resolved_path": HUMAN_USD_PATH,
            "sha256": _sha(human_usd),
            "size_bytes": len(human_usd),
        },
        "schema": "isaac_compute_all_dependencies_v2",
        "unresolved": [],
        "unresolved_count": 0,
    }
    _write(root, DEPENDENCY_REPORT_PATH, _canonical(report))

    license_page = b"<html><title>NVIDIA Isaac Sim Additional License</title></html>"
    _write(root, LICENSE_PAGE_PATH, license_page)
    monkeypatch.setattr(
        contract, "NVIDIA_LICENSE_PAGE_SHA256", _sha(license_page), raising=False
    )
    monkeypatch.setattr(
        contract, "NVIDIA_LICENSE_PAGE_SIZE", len(license_page), raising=False
    )
    license_evidence = {
        "additional_license_url": ADDITIONAL_LICENSE_URL,
        "asset_url": contract.NVIDIA_MEDICAL_HUMAN_URL,
        "page_path": LICENSE_PAGE_PATH,
        "page_sha256": _sha(license_page),
        "page_size": len(license_page),
        "retrieved_at": "2026-07-15T12:00:00Z",
        "schema": "nvidia_isaac_additional_license_evidence_v1",
    }
    _write(root, LICENSE_EVIDENCE_PATH, _canonical(license_evidence))

    def fake_probe(_checkout: Path, required_paths: list[str]) -> dict[str, object]:
        return _git_evidence(required_paths)

    monkeypatch.setattr(contract, "_probe_git_checkout", fake_probe, raising=False)
    return root


def _manifest(root: Path) -> dict[str, object]:
    return contract.build_manifest(root)


def _verify(root: Path, manifest: dict[str, object]) -> dict[str, object]:
    return contract.verify_manifest(contract.canonical_json_bytes(manifest), root)


def _rewrite_json(root: Path, relative: str, mutate) -> None:
    path = root / relative
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    path.write_bytes(_canonical(value))


def test_valid_manifest_binds_git_structure_nvidia_camera_and_files(asset_root: Path):
    manifest = _manifest(asset_root)
    assert manifest["clean_source"] is True
    assert manifest["root_scale"] == 1
    assert manifest["primitive_fallback"] is False
    assert manifest["robot_source"]["checkout_path"] == CHECKOUT_PATH
    assert manifest["robot_source"]["urdf_path"] == URDF_PATH
    assert manifest["robot_source"]["urdf_sha256"] == contract.OFFICIAL_URDF_SHA256
    assert manifest["robot_source"]["canonical_xml_sha256"] == (
        contract.OFFICIAL_URDF_CANONICAL_XML_SHA256
    )
    assert manifest["robot_source"]["git_evidence"]["detached_head"] is True
    assert manifest["dependencies"]["human_usd"] == list(TEXTURE_PATHS)
    assert manifest["human_source"]["usd_sha256"] == contract.HUMAN_USD_SHA256
    report = json.loads((asset_root / DEPENDENCY_REPORT_PATH).read_text(encoding="utf-8"))
    assert report["root_layer"]["resolved_path"] == HUMAN_USD_PATH
    assert report["asset_count"] == len(report["assets"]) == 4
    assert report["assets"][0] == OMNIPBR_RECORD
    assert HUMAN_USD_PATH not in {row.get("resolved_path") for row in report["assets"]}
    assert manifest["camera"]["distortion"] == "unavailable"
    assert manifest["camera"]["depth_bias"] == "unavailable"
    assert manifest["camera"]["camera_to_base_covariance"] == "unavailable"
    assert manifest["camera"]["installation_specific_pass"] is False
    assert contract.verify_manifest(contract.canonical_json_bytes(manifest), asset_root) == manifest


def test_caller_cannot_self_attest_clean_git(asset_root: Path):
    with pytest.raises(TypeError):
        contract.build_manifest(
            asset_root,
            robot_urdf_path=URDF_PATH,
            human_usd_path=HUMAN_USD_PATH,
            human_dependency_paths=TEXTURE_PATHS,
            license_evidence_paths=(LICENSE_EVIDENCE_PATH,),
            source_is_clean=True,
        )


def test_builder_rejects_boolean_git_probe(asset_root: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(contract, "_probe_git_checkout", lambda *_args: True, raising=False)
    with pytest.raises(contract.AssetContractError, match="git evidence"):
        contract.build_manifest(asset_root)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("remote_url", "https://example.invalid/fork.git"),
        ("head", "0" * 40),
        ("detached_head", False),
        ("status_porcelain", "?? untracked.bin"),
        ("tracked_paths", []),
    ],
)
def test_verifier_rejects_false_git_evidence(
    asset_root: Path, field: str, bad_value: object
):
    manifest = _manifest(asset_root)
    manifest["robot_source"]["git_evidence"][field] = bad_value
    with pytest.raises(contract.AssetContractError, match="git"):
        _verify(asset_root, manifest)


def test_single_link_zero_joint_urdf_fails_even_if_hashes_are_expected(
    asset_root: Path, monkeypatch: pytest.MonkeyPatch
):
    fake = (
        b'<robot name="openarm"><link name="only"><visual><geometry>'
        b'<mesh filename="package://openarm_description/assets/fake.dae" '
        b'scale="0.001 0.001 0.001"/></geometry></visual></link></robot>'
    )
    _write(asset_root, URDF_PATH, fake)
    _write(asset_root, "openarm_description/assets/fake.dae", b"fake")
    monkeypatch.setattr(contract, "OFFICIAL_URDF_SHA256", _sha(fake), raising=False)
    monkeypatch.setattr(
        contract,
        "OFFICIAL_URDF_CANONICAL_XML_SHA256",
        _sha(ET.tostring(ET.fromstring(fake), encoding="utf-8")),
        raising=False,
    )
    with pytest.raises(contract.AssetContractError, match="24 links|23 joints|14 revolute"):
        contract.build_manifest(asset_root)


def test_verifier_rejects_wrong_official_urdf_hash(
    asset_root: Path, monkeypatch: pytest.MonkeyPatch
):
    manifest = _manifest(asset_root)
    monkeypatch.setattr(contract, "OFFICIAL_URDF_SHA256", "f" * 64, raising=False)
    with pytest.raises(contract.AssetContractError, match="official URDF.*sha256"):
        _verify(asset_root, manifest)


def test_verifier_rejects_wrong_official_human_hash(
    asset_root: Path, monkeypatch: pytest.MonkeyPatch
):
    manifest = _manifest(asset_root)
    monkeypatch.setattr(contract, "HUMAN_USD_SHA256", "f" * 64, raising=False)
    with pytest.raises(contract.AssetContractError, match="human USD.*sha256"):
        _verify(asset_root, manifest)


def test_verifier_rejects_wrong_official_texture_hash(
    asset_root: Path, monkeypatch: pytest.MonkeyPatch
):
    manifest = _manifest(asset_root)
    wrong = dict(contract.NVIDIA_TEXTURE_SHA256)
    wrong[TEXTURE_PATHS[0]] = "f" * 64
    monkeypatch.setattr(contract, "NVIDIA_TEXTURE_SHA256", wrong, raising=False)
    with pytest.raises(contract.AssetContractError, match="texture.*sha256"):
        _verify(asset_root, manifest)


def test_builder_rejects_unresolved_isaac_dependency(asset_root: Path):
    def add_unresolved(value: dict[str, object]) -> None:
        value["unresolved"] = ["missing.usd"]
        value["unresolved_count"] = 1

    _rewrite_json(asset_root, DEPENDENCY_REPORT_PATH, add_unresolved)
    with pytest.raises(contract.AssetContractError, match="unresolved"):
        contract.build_manifest(asset_root)


def test_builder_rejects_missing_required_texture(asset_root: Path):
    (asset_root / TEXTURE_PATHS[1]).unlink()
    with pytest.raises(contract.AssetContractError, match="missing|texture"):
        contract.build_manifest(asset_root)


def test_builder_rejects_extra_human_dependency(asset_root: Path):
    def add_extra(value: dict[str, object]) -> None:
        value["asset_count"] = 5
        value["assets"].append(
            {
                "authored_uri": "./textures/unreviewed.png",
                "resolution_kind": "usd_relative",
                "resolved_path": "human/textures/unreviewed.png",
                "sha256": _sha(b"extra"),
                "size_bytes": 5,
            }
        )

    _write(asset_root, "human/textures/unreviewed.png", b"extra")
    _rewrite_json(asset_root, DEPENDENCY_REPORT_PATH, add_extra)
    with pytest.raises(contract.AssetContractError, match="dependency|resolved_paths|assets"):
        contract.build_manifest(asset_root)


def test_builder_rejects_root_layer_counted_as_dependency_asset(asset_root: Path):
    def substitute_root_for_omnipbr(value: dict[str, object]) -> None:
        root_layer = value["root_layer"]
        value["assets"][0] = {
            "authored_uri": HUMAN_USD_PATH,
            "resolution_kind": "bundle_root_layer",
            "resolved_path": HUMAN_USD_PATH,
            "sha256": root_layer["sha256"],
            "size_bytes": root_layer["size_bytes"],
        }

    _rewrite_json(asset_root, DEPENDENCY_REPORT_PATH, substitute_root_for_omnipbr)
    with pytest.raises(contract.AssetContractError, match="root layer|OmniPBR|asset records"):
        contract.build_manifest(asset_root)


def test_builder_rejects_legacy_report_that_counts_root_as_fourth_asset(
    asset_root: Path,
):
    root_bytes = (asset_root / HUMAN_USD_PATH).read_bytes()
    legacy = {
        "asset_url": contract.NVIDIA_MEDICAL_HUMAN_URL,
        "assets": 4,
        "isaac_version": "5.1.0.0",
        "layers": 1,
        "resolved_paths": [HUMAN_USD_PATH, *TEXTURE_PATHS],
        "resolver": "Isaac ComputeAllDependencies",
        "root_usd_path": HUMAN_USD_PATH,
        "root_usd_sha256": _sha(root_bytes),
        "schema": "isaac_compute_all_dependencies_v1",
        "unresolved": 0,
    }
    (asset_root / DEPENDENCY_REPORT_PATH).write_bytes(_canonical(legacy))
    with pytest.raises(contract.AssetContractError, match="root layer|schema v2|OmniPBR"):
        contract.build_manifest(asset_root)


def test_builder_rejects_missing_omnipbr_asset_record(asset_root: Path):
    def remove_omnipbr(value: dict[str, object]) -> None:
        value["assets"] = value["assets"][1:]
        value["asset_count"] = 3

    _rewrite_json(asset_root, DEPENDENCY_REPORT_PATH, remove_omnipbr)
    with pytest.raises(contract.AssetContractError, match="OmniPBR|assets=4|asset records"):
        contract.build_manifest(asset_root)


def test_builder_rejects_wrong_omnipbr_runtime_provenance(asset_root: Path):
    def change_record_hash(value: dict[str, object]) -> None:
        value["assets"][0]["runtime_provenance"]["record_hash"] = "sha256=forged"

    _rewrite_json(asset_root, DEPENDENCY_REPORT_PATH, change_record_hash)
    with pytest.raises(contract.AssetContractError, match="OmniPBR|runtime provenance|asset records"):
        contract.build_manifest(asset_root)


def test_builder_rejects_arbitrary_license_text(asset_root: Path):
    (asset_root / LICENSE_EVIDENCE_PATH).write_text("license seems fine", encoding="utf-8")
    with pytest.raises(contract.AssetContractError, match="license.*JSON"):
        contract.build_manifest(asset_root)


def test_builder_rejects_wrong_license_url(asset_root: Path):
    _rewrite_json(
        asset_root,
        LICENSE_EVIDENCE_PATH,
        lambda value: value.__setitem__(
            "additional_license_url", "https://example.invalid/license"
        ),
    )
    with pytest.raises(contract.AssetContractError, match="license.*URL"):
        contract.build_manifest(asset_root)


def test_builder_rejects_forged_license_page_even_when_metadata_matches(
    asset_root: Path,
):
    forged = b"<html><title>Forged license snapshot</title></html>"
    (asset_root / LICENSE_PAGE_PATH).write_bytes(forged)

    def bind_forged_page(value: dict[str, object]) -> None:
        value["page_sha256"] = _sha(forged)
        value["page_size"] = len(forged)

    _rewrite_json(asset_root, LICENSE_EVIDENCE_PATH, bind_forged_page)
    with pytest.raises(contract.AssetContractError, match="license.*snapshot|license.*sha256"):
        contract.build_manifest(asset_root)


def test_verifier_rejects_tampered_dependency_report(asset_root: Path):
    manifest = _manifest(asset_root)
    (asset_root / DEPENDENCY_REPORT_PATH).write_bytes(b"{}")
    with pytest.raises(contract.AssetContractError, match="size|sha256"):
        _verify(asset_root, manifest)


def test_verifier_rejects_symlinked_asset(asset_root: Path):
    manifest = _manifest(asset_root)
    texture = asset_root / TEXTURE_PATHS[0]
    outside = asset_root.parent / "outside.png"
    outside.write_bytes(texture.read_bytes())
    texture.unlink()
    texture.symlink_to(outside)
    with pytest.raises(contract.AssetContractError, match="symlink"):
        _verify(asset_root, manifest)


def test_verifier_rejects_mesh_scale_not_pinned_by_urdf(asset_root: Path):
    manifest = _manifest(asset_root)
    mesh = next(row for row in manifest["files"] if row["role"] == "robot_mesh")
    mesh["scale"] = [0.002, 0.001, 0.001]
    with pytest.raises(contract.AssetContractError, match="scale"):
        _verify(asset_root, manifest)


def test_verifier_rejects_noncanonical_json(asset_root: Path):
    manifest = _manifest(asset_root)
    pretty = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    with pytest.raises(contract.AssetContractError, match="canonical"):
        contract.verify_manifest(pretty, asset_root)


def test_verifier_rejects_extra_schema_key(asset_root: Path):
    manifest = _manifest(asset_root)
    manifest["unreviewed"] = True
    with pytest.raises(contract.AssetContractError, match="keys"):
        _verify(asset_root, manifest)


@pytest.mark.parametrize(
    "invalid_role",
    [[], {}, None, 7],
    ids=["list", "dict", "null", "number"],
)
def test_verifier_rejects_non_string_file_role(
    asset_root: Path, invalid_role: object
):
    manifest = _manifest(asset_root)
    manifest["files"][0]["role"] = invalid_role
    with pytest.raises(contract.AssetContractError, match="role.*string"):
        _verify(asset_root, manifest)


def test_verifier_rejects_guessed_commissioned_pose(asset_root: Path):
    manifest = copy.deepcopy(_manifest(asset_root))
    manifest["camera"]["pose"] = {"pose_status": "installation_pose_commissioned"}
    with pytest.raises(
        contract.AssetContractError,
        match="commissioned.*transform.*provenance",
    ):
        _verify(asset_root, manifest)


@pytest.fixture
def real_clean_detached_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    if shutil.which("git") is None:
        pytest.skip("git is required for checkout-probe tests")
    checkout = tmp_path / "checkout"
    checkout.mkdir()

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=checkout,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.stdout.strip()

    git("init", "-q")
    git("config", "user.email", "asset-contract@example.invalid")
    git("config", "user.name", "Asset Contract Test")
    (checkout / "tracked.txt").write_text("tracked", encoding="utf-8")
    git("add", "tracked.txt")
    git("commit", "-q", "-m", "fixture")
    git("remote", "add", "origin", contract.ENACTIC_REPOSITORY_URL)
    head = git("rev-parse", "HEAD")
    git("checkout", "-q", "--detach", head)
    monkeypatch.setattr(contract, "ENACTIC_COMMIT", head)
    return checkout


def _git(checkout: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=checkout,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_real_git_probe_accepts_clean_detached_tracked_checkout(
    real_clean_detached_checkout: Path,
):
    evidence = contract._probe_git_checkout(
        real_clean_detached_checkout, ["tracked.txt"]
    )
    assert evidence["detached_head"] is True
    assert evidence["status_porcelain"] == ""
    assert evidence["tracked_paths"] == ["tracked.txt"]


def test_real_git_probe_rejects_untracked_file(real_clean_detached_checkout: Path):
    (real_clean_detached_checkout / "rogue.bin").write_bytes(b"rogue")
    with pytest.raises(contract.AssetContractError, match="clean|untracked|status"):
        contract._probe_git_checkout(real_clean_detached_checkout, ["tracked.txt"])


def test_real_git_probe_disables_hostile_core_fsmonitor_without_hanging(
    real_clean_detached_checkout: Path,
):
    marker = real_clean_detached_checkout.parent / "fsmonitor-executed"
    hostile = real_clean_detached_checkout.parent / "hostile-fsmonitor.sh"
    hostile.write_text(
        "#!/bin/sh\n"
        f"printf executed > {str(marker)!r}\n"
        "sleep 2\n"
        "exit 0\n",
        encoding="utf-8",
    )
    hostile.chmod(0o755)
    _git(
        real_clean_detached_checkout,
        "config",
        "core.fsmonitor",
        str(hostile),
    )

    started = time.monotonic()
    evidence = contract._probe_git_checkout(
        real_clean_detached_checkout, ["tracked.txt"]
    )
    elapsed = time.monotonic() - started

    assert evidence["status_porcelain"] == ""
    assert not marker.exists()
    assert elapsed < 1.5


@pytest.mark.parametrize("filter_key", ["clean", "process"])
def test_real_git_probe_never_executes_hostile_content_filter(
    real_clean_detached_checkout: Path,
    monkeypatch: pytest.MonkeyPatch,
    filter_key: str,
):
    checkout = real_clean_detached_checkout
    marker = checkout.parent / f"filter-{filter_key}-executed"
    hostile = checkout.parent / f"hostile-filter-{filter_key}.sh"
    script = [
        "#!/bin/sh",
        f"printf executed > {str(marker)!r}",
        "sleep 2",
    ]
    if filter_key == "clean":
        script.append("cat")
    else:
        script.append("exit 1")
    hostile.write_text("\n".join(script) + "\n", encoding="utf-8")
    hostile.chmod(0o755)

    (checkout / ".gitattributes").write_text(
        "tracked.txt filter=hostile\n", encoding="utf-8"
    )
    _git(checkout, "add", ".gitattributes")
    _git(checkout, "commit", "-q", "-m", "add hostile filter attribute")
    head = _git(checkout, "rev-parse", "HEAD").stdout.strip()
    monkeypatch.setattr(contract, "ENACTIC_COMMIT", head)
    _git(checkout, "config", f"filter.hostile.{filter_key}", str(hostile))

    tracked = checkout / "tracked.txt"
    tracked.write_bytes(tracked.read_bytes())
    started = time.monotonic()
    evidence = contract._probe_git_checkout(checkout, ["tracked.txt"])
    elapsed = time.monotonic() - started

    assert evidence["status_porcelain"] == ""
    assert not marker.exists()
    assert elapsed < 1.5


def test_real_git_probe_rejects_shared_clone_with_external_object_alternate(
    real_clean_detached_checkout: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = real_clean_detached_checkout
    shared = source.parent / "shared-checkout"
    subprocess.run(
        ["git", "clone", "-q", "--shared", str(source), str(shared)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _git(shared, "remote", "set-url", "origin", contract.ENACTIC_REPOSITORY_URL)
    head = _git(shared, "rev-parse", "HEAD").stdout.strip()
    _git(shared, "checkout", "-q", "--detach", head)
    monkeypatch.setattr(contract, "ENACTIC_COMMIT", head)
    alternates = shared / ".git" / "objects" / "info" / "alternates"
    assert alternates.is_file()
    assert Path(alternates.read_text(encoding="utf-8").strip()).is_absolute()

    with pytest.raises(contract.AssetContractError, match="alternate|object storage"):
        contract._probe_git_checkout(shared, ["tracked.txt"])


def test_real_git_probe_rejects_external_commondir(
    real_clean_detached_checkout: Path,
):
    checkout = real_clean_detached_checkout
    external_common = checkout.parent / "external-common"
    shutil.copytree(checkout / ".git", external_common)
    (checkout / ".git" / "commondir").write_text(
        str(external_common) + "\n", encoding="utf-8"
    )

    with pytest.raises(contract.AssetContractError, match="commondir|metadata"):
        contract._probe_git_checkout(checkout, ["tracked.txt"])


@pytest.mark.parametrize("metadata_name", ["objects", "config"])
def test_real_git_probe_rejects_symlinked_git_metadata(
    real_clean_detached_checkout: Path,
    metadata_name: str,
):
    checkout = real_clean_detached_checkout
    metadata = checkout / ".git" / metadata_name
    external = checkout.parent / f"external-{metadata_name}"
    if metadata.is_dir():
        shutil.copytree(metadata, external)
        shutil.rmtree(metadata)
        metadata.symlink_to(external, target_is_directory=True)
    else:
        shutil.copy2(metadata, external)
        metadata.unlink()
        metadata.symlink_to(external)

    with pytest.raises(contract.AssetContractError, match="symlink|metadata|object"):
        contract._probe_git_checkout(checkout, ["tracked.txt"])


def test_real_git_probe_rejects_assume_unchanged_tamper(
    real_clean_detached_checkout: Path,
):
    _git(
        real_clean_detached_checkout,
        "update-index",
        "--assume-unchanged",
        "tracked.txt",
    )
    (real_clean_detached_checkout / "tracked.txt").write_text(
        "tampered behind assume-unchanged", encoding="utf-8"
    )
    status = _git(
        real_clean_detached_checkout,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    assert status.stdout == ""
    with pytest.raises(
        contract.AssetContractError,
        match="assume-unchanged|index flag|committed blob",
    ):
        contract._probe_git_checkout(real_clean_detached_checkout, ["tracked.txt"])


def test_real_git_probe_rejects_skip_worktree_tamper(
    real_clean_detached_checkout: Path,
):
    _git(
        real_clean_detached_checkout,
        "update-index",
        "--skip-worktree",
        "tracked.txt",
    )
    (real_clean_detached_checkout / "tracked.txt").write_text(
        "tampered behind skip-worktree", encoding="utf-8"
    )
    status = _git(
        real_clean_detached_checkout,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    assert status.stdout == ""
    with pytest.raises(
        contract.AssetContractError,
        match="skip-worktree|index flag|committed blob",
    ):
        contract._probe_git_checkout(real_clean_detached_checkout, ["tracked.txt"])
