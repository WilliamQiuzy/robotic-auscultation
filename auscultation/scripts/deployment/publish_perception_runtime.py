"""Build and activate immutable Linbox perception-runtime releases.

The generic :mod:`artifact_contract` owns byte/mode/exact-set integrity and
atomic publication.  This module adds the perception-specific semantic
contract: the sole v9 auscultation-point detector, a box-only YOLO person
presence detector with operator back confirmation, an offline DINOv2 cache,
one verified code release, and one exact Linbox environment identity must
travel together.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
from typing import Dict, Iterable, Mapping, Union

import artifact_contract as contract
import perception_environment_manifest as environment_manifest


RUNTIME_MANIFEST_NAME = "PERCEPTION_RUNTIME.json"
RUNTIME_MANIFEST_SCHEMA = "wam_perception_runtime_v4"
V3_RUNTIME_MANIFEST_SCHEMA = "wam_perception_runtime_v3"
LEGACY_RUNTIME_MANIFEST_SCHEMA = "wam_perception_runtime_v2"
ENVIRONMENT_LOCK_SCHEMA = "wam_perception_environment_lock_v2"
RELOCATION_AWARE_ENVIRONMENT_LOCK_SCHEMA = "wam_perception_environment_lock_v3"
LEGACY_ENVIRONMENT_LOCK_SCHEMA = "wam_perception_environment_lock_v1"
RUNTIME_ARTIFACT_KIND = "perception_runtime"

CHECKPOINT_PATH = "model/perc_lmh_v9_occ/best.pt"
POSTERIOR_GATE_CHECKPOINT_PATH = "model/posterior_gate/yolo11x-pose.pt"
TORCH_HOME_PATH = "torch_home"
DINO_CHECKPOINT_PATH = "torch_home/hub/checkpoints/dinov2_vitb14_pretrain.pth"
DINO_REPOSITORY_PATH = "torch_home/hub/facebookresearch_dinov2_main"
CODE_PATH = "code"
ENVIRONMENT_LOCK_PATH = "environment/linbox-perception.lock.json"

LANDMARK_ORDER = [
    "R1", "L1", "R2", "L2", "R3", "L3", "R4", "L4", "R5", "L5",
]
V3_REQUIRED_HARDENED_CODE_FILES = frozenset({
    "auscultation/ausc/audio/audio_stream.py",
    "auscultation/ausc/models/iuv_model.py",
    "auscultation/ausc/perception/camera_stream.py",
    "auscultation/ausc/perception/tracking/__init__.py",
    "auscultation/ausc/perception/tracking/detector.py",
    "auscultation/ausc/perception/tracking/model_tracker.py",
    "auscultation/ausc/perception/tracking/runtime.py",
    "auscultation/ausc/record_session.py",
    "auscultation/scripts/collect_server.py",
    "auscultation/scripts/collection_platform.py",
    "auscultation/scripts/collection_supervisor.py",
    "auscultation/scripts/perception_capture.py",
    "auscultation/scripts/perception_worker.py",
    "auscultation/scripts/posterior_gate.py",
})
REQUIRED_HARDENED_CODE_FILES = frozenset({
    *V3_REQUIRED_HARDENED_CODE_FILES,
    "auscultation/ausc/realtime_q.py",
    "auscultation/ausc/perception/production_runtime_activation.py",
    "auscultation/ausc/perception/robot/camera_exposure_evidence.py",
    "auscultation/ausc/perception/robot/frame_evidence_bundle.py",
    "auscultation/ausc/perception/robot/openarm_pair_activation.py",
    "auscultation/ausc/perception/robot/robot_capture_coordinator.py",
    "auscultation/ausc/perception/robot/robot_mask_provider.py",
    "auscultation/ausc/perception/robot/robot_session_capture.py",
    "auscultation/ausc/perception/robot/robot_socket_receiver.py",
    "auscultation/ausc/perception/robot/robot_state_stream.py",
    "auscultation/ausc/perception/tracking/back_mask_packet.py",
    "auscultation/ausc/perception/tracking/depth_noise_profile.py",
    "auscultation/ausc/perception/tracking/experimental_visibility_geometry.py",
    "auscultation/ausc/perception/tracking/geometry.py",
    "auscultation/ausc/perception/tracking/geometry_channel_packet.py",
    "auscultation/ausc/perception/tracking/occlusion_runtime.py",
    "auscultation/ausc/perception/tracking/patient_frame_evidence.py",
    "auscultation/ausc/perception/tracking/patient_pose_provider.py",
    "auscultation/ausc/perception/tracking/posterior_back_mask_provider.py",
    "auscultation/ausc/perception/tracking/production_evidence_capture.py",
    "auscultation/ausc/perception/tracking/production_geometry_gate.py",
    "auscultation/mechanical_sensor/force_feedback/force_capture.py",
    "auscultation/mechanical_sensor/force_feedback/force_sensor.py",
    "auscultation/scripts/perception/activate_handeye.py",
})
# Runtime schema v3 made a stronger claim than file integrity alone: YOLO is
# not an orientation authority, and Start requires an operator confirmation
# bound to a displayed frame.  Pin the two files that implement that claim so
# an older pose-orientation gate cannot be repackaged as v3 merely by supplying
# a different set of externally confirmed hashes.  This mapping is frozen for
# rollback verification; v4 has its own independent production-chain pins.
V3_REQUIRED_BACK_GATE_SEMANTIC_CODE_SHA256 = {
    "auscultation/scripts/collect_server.py": (
        "ede93e9197bef9bdfd56c8abda12517de4da70d175fbcf788feb764d08cbdc87"
    ),
    "auscultation/scripts/posterior_gate.py": (
        "7684d3558550f95f27ef9e46e81dc3aef5e63d5aa35d365cd2cd8bc3f1367f51"
    ),
    "auscultation/scripts/perception_worker.py": (
        "bbb756fa3fbd943e17760f70215152c59d732593dd6441a394b16138d6fca47e"
    ),
}
# Backwards-compatible public name for tooling that inspects the frozen v3
# rollback contract.  New v4 builds use the independent production-chain pin
# below and never reinterpret this mapping as v4 evidence.
REQUIRED_BACK_GATE_SEMANTIC_CODE_SHA256 = (
    V3_REQUIRED_BACK_GATE_SEMANTIC_CODE_SHA256
)

# This independently reviewed v4 mapping deliberately does not replace the
# frozen v3 mapping above.  A v3 rollback therefore retains its original
# 14-file claim, while a new v4 build must match the complete production chain.
REQUIRED_PRODUCTION_CHAIN_SEMANTIC_CODE_SHA256 = {
    "auscultation/ausc/audio/audio_stream.py": "91781c9564f41545ea3330dc38645fffe055961140bea412202cd933d290402a",
    "auscultation/ausc/models/iuv_model.py": "2bd72dc0a33e06e29c88e64480ad20b5586e361a8d7d6958e66fd08aea127cbf",
    "auscultation/ausc/perception/camera_stream.py": "9d62914b30ee6fd114763d31bf167afedc28ecc36b87943e3f55b004c4eaa34f",
    "auscultation/ausc/perception/production_runtime_activation.py": "3072a280196f29c7ce7f86fd45a07c3df6b3c67a4da0a9348f46a6554bac256b",
    "auscultation/ausc/perception/robot/camera_exposure_evidence.py": "ba07a9c3c90d7799ffc327adcd29b3228a842e73c12ceb94554e3de36fd27b34",
    "auscultation/ausc/perception/robot/frame_evidence_bundle.py": "53c3c7f1e7b9a2543f9836e2b16a3d420fc19075081c86d33091959b318c9dac",
    "auscultation/ausc/perception/robot/openarm_pair_activation.py": "52ac7798fd9cd800118d94bed4a4efd08b5fdd651d8d701095058e0aeeb5f1c9",
    "auscultation/ausc/perception/robot/robot_capture_coordinator.py": "c89e289830a510fcdc82cd18f0a6aafeea8648114856822dc4fa39ec72242444",
    "auscultation/ausc/perception/robot/robot_mask_provider.py": "a6dd78c4fe7203fc035fc4123a1f00efbc53a888195323898cf6a257c86219e2",
    "auscultation/ausc/perception/robot/robot_session_capture.py": "ce7842e338a1f4b8fcd24ef7855bbddb911bdd1ae9372b6ad657a1738badfc3d",
    "auscultation/ausc/perception/robot/robot_socket_receiver.py": "75ab952d59a5174a78baf10f7b5d79fda508afabaa77a48eb6ffa5fb472a26d9",
    "auscultation/ausc/perception/robot/robot_state_stream.py": "35b9dc85084ca1db0cc4f57079a81fbcf7da5da91d2cefe4316da71bd46b962d",
    "auscultation/ausc/perception/tracking/__init__.py": "e135d34389019e483400c79b001d9b9e84d13a0cacebe1585174cad2048bccd6",
    "auscultation/ausc/perception/tracking/back_mask_packet.py": "4aba720dcefa8f80c6baf3558e58e4d0ee34b7727ee385fca1344d97d26df4e0",
    "auscultation/ausc/perception/tracking/depth_noise_profile.py": "eebb5d23d8033594e3fc580f00bddf4e4a32b0d536ae9a8b7a578b433ce264fc",
    "auscultation/ausc/perception/tracking/detector.py": "834d2a65fa5236a7240731bb594c521eb76057e103012f46925ec696f363606e",
    "auscultation/ausc/perception/tracking/experimental_visibility_geometry.py": "ebe51a692a2e871e47415cb6dbecbb96d14bb7f410deceb2d8b1e485c4d52e33",
    "auscultation/ausc/perception/tracking/geometry.py": "04ef04d473b0d5348cb4bacf9b6168053bd0ac969a8868eb3b845036d6603696",
    "auscultation/ausc/perception/tracking/geometry_channel_packet.py": "491968eb8cab001b97a9663ed7ca24142e69ac2f576c15b6474288f56b94b923",
    "auscultation/ausc/perception/tracking/model_tracker.py": "4d9172b0bfe9e76695c31a1542b42a6bc92d23fd145a0bbeadbf755ca1c5f3af",
    "auscultation/ausc/perception/tracking/occlusion_runtime.py": "b67f61972344842375ec085c02527e3811f3441ac8bce1108fbddedd6591e392",
    "auscultation/ausc/perception/tracking/patient_frame_evidence.py": "c3958cf6a5aa5b8d6f856900598b2c402101c78be866e4393db75743fa0f7639",
    "auscultation/ausc/perception/tracking/patient_pose_provider.py": "a4c0da10e99b91c3e0ba414ab5eaf1469bc77becf167fe2c65f23fdcea8ff589",
    "auscultation/ausc/perception/tracking/posterior_back_mask_provider.py": "7f97668e9fff41ca3eab945a5133f62beacd9535cd23010dde7bc75fa5d941f3",
    "auscultation/ausc/perception/tracking/production_evidence_capture.py": "ff8025fbfac0c1aa382c6f804602b563570531dca97b9a14a96bd395e0ea5aa6",
    "auscultation/ausc/perception/tracking/production_geometry_gate.py": "48111d9c1253d192370aa2080042bc159f58ef9a55d0724f39e0a8bd046cdc85",
    "auscultation/ausc/perception/tracking/runtime.py": "47f35928e9f94691ec5ea35ba8f73c1957dc5fed637be729ca68d247dfb81187",
    "auscultation/ausc/realtime_q.py": "e193389433ba95d872a8f1ae34760ee170e58a3b40307ed53f7bf017a39b6023",
    "auscultation/ausc/record_session.py": "bb05ac01a7e605e93b7bb46d920dd5e664e21ff9ba7b112988909e2276a17410",
    "auscultation/mechanical_sensor/force_feedback/force_capture.py": "516c2a9c7f35ce0758fa0b7a2727e267444acfa051dbfd603719ef9f56c3160a",
    "auscultation/mechanical_sensor/force_feedback/force_sensor.py": "09f656f179301728ec955d7aea385c13a8968589359974c15b91c6f40261abce",
    "auscultation/scripts/collect_server.py": "cd95db5a3d8313aa2583991d198dea2ccf64367d513693c429c2b5aae30633d7",
    "auscultation/scripts/collection_platform.py": "7b5da0189549764c17234ecb7c1978f29f34dafbce7a96f61a486e214d0df011",
    "auscultation/scripts/collection_supervisor.py": "03f265cc6540770066f8ed5ca11b609e13a842e59aa94fe211c806de4508d420",
    "auscultation/scripts/perception_capture.py": "401c5f5d2cb77e14456082a21124cca3053046b645952511c8375f4116a05a49",
    "auscultation/scripts/perception/activate_handeye.py": "952dcf7048a71aa2682f3f18af2fef9cff29ff0429d9a1e0e65be095fa49d360",
    "auscultation/scripts/perception_worker.py": "3f2d25e21eeb381812ddc75977e3cb0b0985274ee333b5adeb8a2f50b436e4ac",
    "auscultation/scripts/posterior_gate.py": "47693f1605f041471f0159d70ba73269f845c7b3774dcec24c129cbe129f50c4",
}
APPROVED_LEGACY_V2_ROLLBACKS = {
    "perception_runtime-33641f649a1fd658101e": {
        "payload_sha256": (
            "33641f649a1fd658101e4cbb5cdd5cf87d2144c4868d8f2abfd5054b3376c6a1"
        ),
        "code_release_id": "code_release-0eb0e5ea614946b258ff",
        "code_payload_sha256": (
            "0eb0e5ea614946b258ffb9b30213b6d583f81b36bef901313b32012d553c2f1a"
        ),
    },
}
LEGACY_SUPPORTED_PACKAGES = {
    "numpy": "1.26.4",
    "opencv-python": "4.11.0.86",
    "opencv-python-headless": "4.13.0.92",
    "pyorbbecsdk2": "2.1.1",
    "scipy": "1.17.1",
    "torch": "2.11.0+cu128",
    "torchvision": "0.26.0+cu128",
    "yourdfpy": "0.0.60",
}
SUPPORTED_PACKAGES = {
    # One distribution must own ``cv2``.  The legacy clone inherited both
    # opencv-python and a newer opencv-python-headless whose NumPy requirement
    # was unsatisfied; overlapping RECORDs make that environment impossible to
    # reproduce or audit honestly.  Runtime v2 keeps the compatible 4.11 wheel.
    **{
        name: version for name, version in LEGACY_SUPPORTED_PACKAGES.items()
        if name != "opencv-python-headless"
    },
    "polars": "1.41.2",
    "polars-runtime-32": "1.41.2",
    "ultralytics": "8.4.64",
    "ultralytics-thop": "2.0.20",
}
SUPPORTED_MODULES = frozenset({"cv2", "pyorbbecsdk", "torch", "ultralytics"})
RELOCATION_AWARE_SUPPORTED_PACKAGES = {
    **SUPPORTED_PACKAGES,
    "imageio-ffmpeg": "0.6.0",
}
RELOCATION_AWARE_SUPPORTED_MODULES = frozenset({
    *SUPPORTED_MODULES,
    "imageio_ffmpeg",
})
LEGACY_SUPPORTED_MODULES = frozenset({"cv2", "pyorbbecsdk", "torch"})
DISTRIBUTION_INVENTORY_ALGORITHM = (
    "sha256-canonical-json-normalized-name-version-record-and-installed-path-bytes-size-sorted-owners-v3"
)
LEGACY_BYTE_DISTRIBUTION_INVENTORY_ALGORITHM = (
    "sha256-canonical-json-normalized-name-version-record-and-installed-files-bytes-size-v2"
)
LEGACY_DISTRIBUTION_INVENTORY_ALGORITHM = (
    "sha256-canonical-json-normalized-name-version-record-text-v1"
)
MODULE_TREE_ALGORITHM = "sha256-canonical-json-environment-relative-module-tree-v1"
SOURCE_ENVIRONMENT_FINGERPRINT_ALGORITHM = (
    "sha256-canonical-wam-perception-source-observation-v3"
)
LEGACY_BYTE_SOURCE_ENVIRONMENT_FINGERPRINT_ALGORITHM = (
    "sha256-canonical-wam-perception-source-observation-v2"
)
LEGACY_SOURCE_ENVIRONMENT_FINGERPRINT_ALGORITHM = (
    "sha256-canonical-wam-perception-source-observation-v1"
)
ENVIRONMENT_IDENTITY_ALGORITHM = (
    "sha256-canonical-wam-perception-environment-lock-identity-v2"
)
RELOCATION_AWARE_ENVIRONMENT_IDENTITY_ALGORITHM = (
    "sha256-canonical-wam-perception-environment-lock-identity-v3"
)
REPAIR_WHEELHOUSE_ALGORITHM = (
    "sha256-canonical-json-exact-perception-repair-wheelhouse-v1"
)
FINAL_PATH_BUILD_RECIPE_ALGORITHM = (
    "sha256-canonical-json-perception-final-path-build-recipe-v1"
)
LINBOX_PERCEPTION_RELEASES_ROOT = (
    "/home/robot/wam_data/perception_envs/releases"
)
FINAL_PATH_BUILD_OPERATIONS = (
    "conda-clone-copy-to-final-path",
    "remove-opencv-python-headless",
    "install-exact-opencv-wheel",
    "install-exact-gate-wheelhouse",
    "repair-pip-and-install-imageio-ffmpeg",
    "omit-audited-numpy-generated-pyc",
    "smoke-without-bytecode",
    "freeze-and-lock-tree",
    "write-completion-marker-last",
)
BUNDLED_FFMPEG_RELATIVE_PATH = (
    "lib/python3.11/site-packages/imageio_ffmpeg/binaries/"
    "ffmpeg-linux-x86_64-v7.0.2"
)
EXACT_REPAIR_WHEELS = [
    {
        "filename": "imageio_ffmpeg-0.6.0-py3-none-manylinux2014_x86_64.whl",
        "name": "imageio-ffmpeg",
        "sha256": "c7e46fcec401dd990405049d2e2f475e2b397779df2519b544b8aab515195282",
        "size": 29498237,
        "url": (
            "https://files.pythonhosted.org/packages/a0/2d/"
            "43c8522a2038e9d0e7dbdf3a61195ecc31ca576fb1527a528c877e87d973/"
            "imageio_ffmpeg-0.6.0-py3-none-manylinux2014_x86_64.whl"
        ),
        "version": "0.6.0",
    },
    {
        "filename": "pip-26.1.2-py3-none-any.whl",
        "name": "pip",
        "sha256": "382ff9f685ee3bc25864f820aa50505825f10f5458ffff07e30a6d96e5715cab",
        "size": 1813144,
        "url": (
            "https://files.pythonhosted.org/packages/5d/95/"
            "6b5cb3461ea5673ba0995989746db58eb18b91b54dbf331e72f569540946/"
            "pip-26.1.2-py3-none-any.whl"
        ),
        "version": "26.1.2",
    },
]
SUPPORTED_CUDA_ARCHITECTURES = [
    "sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120",
]
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ENVIRONMENT_ID = re.compile(r"^linbox-perception-[0-9a-f]{20}$")
_FORBIDDEN_TREE_PARTS = frozenset((".git", "__pycache__", ".pytest_cache"))


class PerceptionRuntimeError(contract.ArtifactError):
    """A perception release violates its semantic identity contract."""


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise PerceptionRuntimeError("%s must be a lowercase SHA-256" % label)
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise PerceptionRuntimeError("%s must be a relative POSIX path" % label)
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or any(part in ("", ".", "..") for part in candidate.parts):
        raise PerceptionRuntimeError("%s must not escape the runtime release" % label)
    if candidate.as_posix() != value:
        raise PerceptionRuntimeError("%s is not canonical" % label)
    return value


def _tree_rows(root: Union[str, Path], *, ignore_transient: bool = False) -> list[dict]:
    base = Path(root)
    if not base.is_dir() or base.is_symlink():
        raise PerceptionRuntimeError("tree root must be a real directory: %s" % base)
    rows = []
    for directory, dirnames, filenames in os.walk(str(base), followlinks=False):
        parent = Path(directory)
        for name in list(dirnames):
            candidate = parent / name
            if name in _FORBIDDEN_TREE_PARTS:
                if ignore_transient:
                    dirnames.remove(name)
                    continue
                raise PerceptionRuntimeError("tree contains transient metadata: %s" % candidate)
            if candidate.is_symlink():
                raise PerceptionRuntimeError("tree contains a symlink: %s" % candidate)
        for name in filenames:
            candidate = parent / name
            relative = candidate.relative_to(base).as_posix()
            if (name in _FORBIDDEN_TREE_PARTS or name.endswith((".pyc", ".pyo"))
                    or any(part in _FORBIDDEN_TREE_PARTS for part in PurePosixPath(relative).parts)):
                if ignore_transient:
                    continue
                raise PerceptionRuntimeError("tree contains transient metadata: %s" % candidate)
            info = candidate.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise PerceptionRuntimeError("tree contains a non-regular file: %s" % candidate)
            rows.append({
                "mode": format(stat.S_IMODE(info.st_mode), "04o"),
                "path": relative,
                "sha256": _sha256_file(candidate),
                "size": info.st_size,
            })
    rows.sort(key=lambda row: row["path"])
    return rows


def hash_regular_tree(root: Union[str, Path]) -> dict:
    """Return the canonical identity of an exact, symlink-free regular-file tree."""
    rows = _tree_rows(root)
    return {
        "algorithm": "sha256",
        "files": len(rows),
        "sha256": hashlib.sha256(contract.canonical_json_bytes(rows)).hexdigest(),
    }


def hash_dino_source_tree(root: Union[str, Path]) -> dict:
    """Hash executable DINO source while excluding host-specific cache metadata."""
    rows = _tree_rows(root, ignore_transient=True)
    return {
        "algorithm": "sha256",
        "files": len(rows),
        "sha256": hashlib.sha256(contract.canonical_json_bytes(rows)).hexdigest(),
    }


def _regular_files(root: Path) -> Iterable[str]:
    return (row["path"] for row in _tree_rows(root))


def _load_canonical_object(path: Union[str, Path], label: str) -> tuple[dict, bytes]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise PerceptionRuntimeError("%s must be a regular file" % label)
    raw = candidate.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PerceptionRuntimeError("%s must be valid UTF-8 JSON" % label) from exc
    if not isinstance(value, dict) or raw != contract.canonical_json_bytes(value):
        raise PerceptionRuntimeError("%s must be a canonical JSON object" % label)
    return value, raw


def _exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise PerceptionRuntimeError(
            "%s fields mismatch; missing=%r extra=%r"
            % (label, sorted(expected - actual), sorted(actual - expected))
        )


def environment_identity_sha256(value: Mapping[str, object]) -> str:
    """Hash the complete runtime identity without its circular display ID."""

    required = (
        (
            "schema", "python", "record_observation", "environment_tree",
            "packages", "modules", "ffmpeg", "cuda", "provisioning",
        )
        if value.get("schema") == RELOCATION_AWARE_ENVIRONMENT_LOCK_SCHEMA
        else (
            "schema", "python", "distribution_inventory", "packages",
            "modules", "cuda", "provisioning",
        )
    )
    try:
        projection = {name: value[name] for name in required}
    except (KeyError, TypeError) as exc:
        raise PerceptionRuntimeError(
            "environment identity projection is incomplete"
        ) from exc
    return hashlib.sha256(contract.canonical_json_bytes(projection)).hexdigest()


def source_environment_fingerprint_sha256(observation: Mapping[str, object]) -> str:
    """Hash a source probe without host paths or transient GPU availability."""

    try:
        python = observation["python"]
        modules = observation["modules"]
        cuda = observation["cuda"]
        projection = {
            "algorithm": SOURCE_ENVIRONMENT_FINGERPRINT_ALGORITHM,
            "python": {
                name: python[name] for name in (
                    "implementation", "version", "abi", "platform",
                    "executable_sha256",
                )
            },
            "distribution_inventory": observation["distribution_inventory"],
            "packages": observation["packages"],
            "modules": {
                name: {
                    key: modules[name][key]
                    for key in ("algorithm", "count", "sha256")
                }
                for name in sorted(modules)
            },
            "cuda": {
                name: cuda[name] for name in ("build", "architectures")
            },
        }
    except (KeyError, TypeError) as exc:
        raise PerceptionRuntimeError(
            "source environment observation is incomplete"
        ) from exc
    return hashlib.sha256(contract.canonical_json_bytes(projection)).hexdigest()


def build_recipe_sha256(value: Mapping[str, object]) -> str:
    """Hash a final-path recipe without its circular display digest."""

    if not isinstance(value, Mapping):
        raise PerceptionRuntimeError("build recipe must be an object")
    projection = {key: value[key] for key in value if key != "sha256"}
    return hashlib.sha256(contract.canonical_json_bytes(projection)).hexdigest()


def validate_final_path_build_recipe(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PerceptionRuntimeError("final-path build recipe must be an object")
    _exact_keys(value, {
        "algorithm", "operations", "sha256", "source_environment", "target",
        "wheelhouse_manifests",
    }, "final-path build recipe")
    if value["algorithm"] != FINAL_PATH_BUILD_RECIPE_ALGORITHM:
        raise PerceptionRuntimeError("final-path build recipe algorithm is unsupported")
    if value["operations"] != list(FINAL_PATH_BUILD_OPERATIONS):
        raise PerceptionRuntimeError("final-path build recipe operations are unsupported")
    source = value["source_environment"]
    if not isinstance(source, Mapping):
        raise PerceptionRuntimeError("build recipe source environment is invalid")
    _exact_keys(source, {
        "environment_id", "lock_sha256", "marker_sha256", "path",
        "snapshot_sha256",
    }, "build recipe source environment")
    source_id = source["environment_id"]
    if not isinstance(source_id, str) or not _ENVIRONMENT_ID.fullmatch(source_id):
        raise PerceptionRuntimeError("build recipe source environment_id is invalid")
    _require_sha256(source["lock_sha256"], "build recipe source lock SHA-256")
    _require_sha256(source["marker_sha256"], "build recipe source marker SHA-256")
    _require_sha256(source["snapshot_sha256"], "build recipe source snapshot SHA-256")
    expected_source_path = "%s/%s" % (
        LINBOX_PERCEPTION_RELEASES_ROOT, source_id,
    )
    if source["path"] != expected_source_path:
        raise PerceptionRuntimeError(
            "build recipe source must be the immutable verified release path"
        )
    target = value["target"]
    if not isinstance(target, Mapping):
        raise PerceptionRuntimeError("build recipe target is invalid")
    _exact_keys(target, {
        "abi", "cuda_build", "platform", "python_version", "releases_root",
        "strategy",
    }, "build recipe target")
    if dict(target) != {
        "abi": "cp311",
        "cuda_build": "12.8",
        "platform": "linux-x86_64",
        "python_version": "3.11.15",
        "releases_root": LINBOX_PERCEPTION_RELEASES_ROOT,
        "strategy": "reserve-and-build-in-final-path-no-rename",
    }:
        raise PerceptionRuntimeError("build recipe target profile is unsupported")
    wheelhouses = value["wheelhouse_manifests"]
    if not isinstance(wheelhouses, list) or len(wheelhouses) != 3:
        raise PerceptionRuntimeError("build recipe wheelhouse manifests are invalid")
    expected_purposes = ["gate-additions", "opencv-repair", "runtime-repairs"]
    if [row.get("purpose") for row in wheelhouses if isinstance(row, Mapping)] != (
        expected_purposes
    ):
        raise PerceptionRuntimeError("build recipe wheelhouse purposes are invalid")
    for row in wheelhouses:
        if not isinstance(row, Mapping):
            raise PerceptionRuntimeError("build recipe wheelhouse row is invalid")
        _exact_keys(row, {"purpose", "sha256"}, "build recipe wheelhouse")
        _require_sha256(row["sha256"], "build recipe wheelhouse SHA-256")
    expected_sha256 = build_recipe_sha256(value)
    if value["sha256"] != expected_sha256:
        raise PerceptionRuntimeError("final-path build recipe SHA-256 mismatch")
    return value


def expected_final_environment_root(value: Mapping[str, object]) -> Path:
    """Return the permanent release path selected before environment build."""

    recipe = validate_final_path_build_recipe(value["build_recipe"])
    environment_id = value.get("environment_id")
    if not isinstance(environment_id, str) or not _ENVIRONMENT_ID.fullmatch(
        environment_id
    ):
        raise PerceptionRuntimeError("environment_id is invalid")
    if environment_id != "linbox-perception-" + recipe["sha256"][:20]:
        raise PerceptionRuntimeError("environment_id does not bind the build recipe")
    return Path(LINBOX_PERCEPTION_RELEASES_ROOT) / environment_id


def _validate_relocation_aware_environment_lock(value: dict) -> dict:
    _exact_keys(value, {
        "schema", "environment_id", "build_recipe", "python", "record_observation",
        "environment_tree", "packages", "modules", "ffmpeg", "cuda",
        "provisioning",
    }, "relocation-aware environment lock")

    recipe = validate_final_path_build_recipe(value["build_recipe"])
    expected_environment_id = "linbox-perception-" + recipe["sha256"][:20]
    environment_id = value["environment_id"]
    if (
        not isinstance(environment_id, str)
        or not _ENVIRONMENT_ID.fullmatch(environment_id)
        or environment_id != expected_environment_id
    ):
        raise PerceptionRuntimeError(
            "environment_id does not bind the final-path build recipe"
        )
    expected_root = Path(LINBOX_PERCEPTION_RELEASES_ROOT) / environment_id

    python = value["python"]
    if not isinstance(python, dict):
        raise PerceptionRuntimeError("environment python identity is invalid")
    _exact_keys(python, {
        "implementation", "version", "abi", "platform", "executable_sha256",
    }, "environment python")
    if (
        python["implementation"] != "CPython"
        or python["version"] != "3.11.15"
        or python["abi"] != "cp311"
        or python["platform"] != "linux-x86_64"
    ):
        raise PerceptionRuntimeError(
            "environment is not the selected Linbox CPython 3.11 profile"
        )
    _require_sha256(python["executable_sha256"], "python executable_sha256")

    try:
        record = environment_manifest.validate_record_observation(
            value["record_observation"], require_resolved=True,
        )
        tree = environment_manifest.validate_environment_tree(
            value["environment_tree"]
        )
        environment_manifest.cross_check_record_tree(record, tree)
    except environment_manifest.ManifestError as exc:
        raise PerceptionRuntimeError(
            "relocation-aware environment manifest is invalid: %s" % exc
        ) from exc
    source_root = recipe["source_environment"]["path"]
    for distribution in record["distributions"]:
        for claim in distribution["claims"]:
            relocation = claim.get("relocation")
            if relocation is not None and (
                relocation["old_prefix"] != source_root
                or relocation["new_prefix"] != str(expected_root)
            ):
                raise PerceptionRuntimeError(
                    "RECORD relocation does not bind the immutable source and final paths"
                )

    packages = value["packages"]
    if not isinstance(packages, list):
        raise PerceptionRuntimeError("environment packages must be a list")
    normalized = {}
    record_by_name = {row["name"]: row for row in record["distributions"]}
    for row in packages:
        if not isinstance(row, dict):
            raise PerceptionRuntimeError("environment package row is invalid")
        _exact_keys(row, {"name", "version", "record_sha256"}, "environment package")
        name = row["name"]
        if not isinstance(name, str) or name in normalized:
            raise PerceptionRuntimeError(
                "environment package names must be unique strings"
            )
        if not isinstance(row["version"], str) or not row["version"]:
            raise PerceptionRuntimeError("environment package version is invalid")
        _require_sha256(row["record_sha256"], "%s RECORD sha256" % name)
        normalized[name] = row["version"]
        recorded = record_by_name.get(name)
        if (
            recorded is None
            or recorded["version"] != row["version"]
            or recorded["record_sha256"] != row["record_sha256"]
        ):
            raise PerceptionRuntimeError(
                "critical package does not match the complete RECORD observation: %s"
                % name
            )
    if [row["name"] for row in packages] != sorted(normalized):
        raise PerceptionRuntimeError("environment package rows must be name-sorted")
    if normalized != RELOCATION_AWARE_SUPPORTED_PACKAGES:
        raise PerceptionRuntimeError(
            "environment critical package profile is unsupported"
        )

    modules = value["modules"]
    if (
        not isinstance(modules, dict)
        or set(modules) != RELOCATION_AWARE_SUPPORTED_MODULES
    ):
        raise PerceptionRuntimeError("environment module identity set is invalid")
    for name, identity in modules.items():
        if not isinstance(identity, dict):
            raise PerceptionRuntimeError(
                "environment module identity is invalid: %s" % name
            )
        _exact_keys(
            identity, {"algorithm", "count", "sha256"},
            "environment module %s" % name,
        )
        if identity["algorithm"] != MODULE_TREE_ALGORITHM:
            raise PerceptionRuntimeError(
                "environment module tree algorithm is unsupported: %s" % name
            )
        if type(identity["count"]) is not int or identity["count"] <= 0:
            raise PerceptionRuntimeError(
                "environment module tree count is invalid: %s" % name
            )
        _require_sha256(identity["sha256"], "%s module sha256" % name)

    ffmpeg = value["ffmpeg"]
    if not isinstance(ffmpeg, dict):
        raise PerceptionRuntimeError("bundled ffmpeg identity is invalid")
    _exact_keys(ffmpeg, {
        "distribution", "mode", "path", "resolver", "sha256", "size",
        "version",
    }, "bundled ffmpeg")
    if (
        ffmpeg["distribution"] != "imageio-ffmpeg"
        or ffmpeg["version"] != "0.6.0"
        or ffmpeg["resolver"] != "imageio_ffmpeg.get_ffmpeg_exe"
        or ffmpeg["path"] != BUNDLED_FFMPEG_RELATIVE_PATH
        or not isinstance(ffmpeg["mode"], str)
        or not re.fullmatch(r"0[0-7]{3}", ffmpeg["mode"])
        or type(ffmpeg["size"]) is not int
        or ffmpeg["size"] <= 0
    ):
        raise PerceptionRuntimeError("bundled ffmpeg profile is unsupported")
    _require_sha256(ffmpeg["sha256"], "bundled ffmpeg SHA-256")
    tree_by_path = {row["path"]: row for row in tree["entries"]}
    ffmpeg_tree = tree_by_path.get(BUNDLED_FFMPEG_RELATIVE_PATH)
    if (
        not isinstance(ffmpeg_tree, Mapping)
        or ffmpeg_tree.get("kind") != "regular"
        or {
            key: ffmpeg_tree.get(key) for key in ("mode", "sha256", "size")
        } != {
            key: ffmpeg[key] for key in ("mode", "sha256", "size")
        }
    ):
        raise PerceptionRuntimeError(
            "bundled ffmpeg does not match the complete environment tree"
        )

    cuda = value["cuda"]
    if not isinstance(cuda, dict):
        raise PerceptionRuntimeError("CUDA identity is invalid")
    _exact_keys(cuda, {"build", "architectures"}, "CUDA identity")
    if (
        cuda["build"] != "12.8"
        or cuda["architectures"] != SUPPORTED_CUDA_ARCHITECTURES
    ):
        raise PerceptionRuntimeError("CUDA build/architectures are unsupported")

    provisioning = value["provisioning"]
    if not isinstance(provisioning, dict):
        raise PerceptionRuntimeError("environment provisioning identity is invalid")
    _exact_keys(provisioning, {
        "generated_pyc_omission", "mode", "repair_wheelhouse",
        "required_service_environment", "source_environment_fingerprint_algorithm",
        "source_environment_fingerprint_sha256", "wheelhouse_complete",
    }, "environment provisioning")
    if (
        provisioning["mode"] != "verified_clone_with_exact_repairs"
        or provisioning["wheelhouse_complete"] is not False
        or provisioning["required_service_environment"]
        != {"PYTHONDONTWRITEBYTECODE": "1"}
        or provisioning["source_environment_fingerprint_algorithm"]
        != SOURCE_ENVIRONMENT_FINGERPRINT_ALGORITHM
    ):
        raise PerceptionRuntimeError(
            "relocation-aware environment provisioning policy is unsupported"
        )
    _require_sha256(
        provisioning["source_environment_fingerprint_sha256"],
        "source environment fingerprint",
    )
    evidence = provisioning["generated_pyc_omission"]
    if not isinstance(evidence, Mapping):
        raise PerceptionRuntimeError("generated pyc omission evidence is invalid")
    try:
        expected_evidence = environment_manifest.generated_pyc_omission_evidence(
            path=evidence.get("path"),
            source_path=evidence.get("source_path"),
            source_sha256=evidence.get("source_sha256"),
            inspection_sha256=evidence.get("inspection_sha256"),
        )
    except environment_manifest.ManifestError as exc:
        raise PerceptionRuntimeError(
            "generated pyc omission evidence is invalid"
        ) from exc
    if dict(evidence) != expected_evidence:
        raise PerceptionRuntimeError("generated pyc omission evidence is noncanonical")
    evidence_sha256 = hashlib.sha256(
        contract.canonical_json_bytes(expected_evidence)
    ).hexdigest()
    omitted_claims = [
        claim
        for distribution in record["distributions"]
        for claim in distribution["claims"]
        if claim["status"] == "omitted_generated_pyc"
    ]
    if (
        not omitted_claims
        or {claim["installed_path"] for claim in omitted_claims}
        != {environment_manifest.GENERATED_PYC_OMISSION}
        or {claim["omission_evidence_sha256"] for claim in omitted_claims}
        != {evidence_sha256}
    ):
        raise PerceptionRuntimeError(
            "generated pyc omission does not match its RECORD claim"
        )
    wheelhouse = provisioning["repair_wheelhouse"]
    if not isinstance(wheelhouse, dict):
        raise PerceptionRuntimeError("repair wheelhouse identity is invalid")
    _exact_keys(
        wheelhouse, {"algorithm", "manifest_sha256", "wheels"},
        "repair wheelhouse",
    )
    if (
        wheelhouse["algorithm"] != REPAIR_WHEELHOUSE_ALGORITHM
        or wheelhouse["wheels"] != EXACT_REPAIR_WHEELS
    ):
        raise PerceptionRuntimeError("repair wheelhouse profile is unsupported")
    _require_sha256(
        wheelhouse["manifest_sha256"], "repair wheelhouse manifest SHA-256",
    )
    runtime_recipe_rows = [
        row for row in recipe["wheelhouse_manifests"]
        if row["purpose"] == "runtime-repairs"
    ]
    if (
        len(runtime_recipe_rows) != 1
        or runtime_recipe_rows[0]["sha256"] != wheelhouse["manifest_sha256"]
    ):
        raise PerceptionRuntimeError(
            "repair wheelhouse does not match the final-path build recipe"
        )
    return value


def validate_environment_lock(value: object, *, require_posterior_gate: bool = False) -> dict:
    """Validate the exact verified-clone profile selected by the Linbox probe."""
    if not isinstance(value, dict):
        raise PerceptionRuntimeError("environment lock must be a JSON object")
    if value.get("schema") == RELOCATION_AWARE_ENVIRONMENT_LOCK_SCHEMA:
        return _validate_relocation_aware_environment_lock(value)
    _exact_keys(value, {
        "schema", "environment_id", "python", "distribution_inventory",
        "packages", "modules", "cuda", "provisioning",
    }, "environment lock")
    if value["schema"] not in {
        ENVIRONMENT_LOCK_SCHEMA, LEGACY_ENVIRONMENT_LOCK_SCHEMA,
    }:
        raise PerceptionRuntimeError("environment lock schema is invalid")
    gate_profile = value["schema"] == ENVIRONMENT_LOCK_SCHEMA
    if require_posterior_gate and not gate_profile:
        raise PerceptionRuntimeError(
            "environment lock does not include the posterior gate runtime profile"
        )

    python = value["python"]
    if not isinstance(python, dict):
        raise PerceptionRuntimeError("environment python identity is invalid")
    _exact_keys(python, {
        "implementation", "version", "abi", "platform", "executable_sha256",
    }, "environment python")
    if (
        python["implementation"] != "CPython"
        or python["version"] != "3.11.15"
        or python["abi"] != "cp311"
        or python["platform"] != "linux-x86_64"
    ):
        raise PerceptionRuntimeError("environment is not the selected Linbox CPython 3.11 profile")
    _require_sha256(python["executable_sha256"], "python executable_sha256")

    inventory = value["distribution_inventory"]
    if not isinstance(inventory, dict):
        raise PerceptionRuntimeError("distribution inventory is invalid")
    inventory_keys = {"algorithm", "count", "sha256"} if gate_profile else {
        "count", "sha256",
    }
    _exact_keys(inventory, inventory_keys, "distribution inventory")
    strong_profile = gate_profile and inventory.get("algorithm") == (
        DISTRIBUTION_INVENTORY_ALGORITHM
    )
    legacy_byte_profile = gate_profile and inventory.get("algorithm") == (
        LEGACY_BYTE_DISTRIBUTION_INVENTORY_ALGORITHM
    )
    if gate_profile and inventory["algorithm"] not in {
        DISTRIBUTION_INVENTORY_ALGORITHM,
        LEGACY_BYTE_DISTRIBUTION_INVENTORY_ALGORITHM,
        LEGACY_DISTRIBUTION_INVENTORY_ALGORITHM,
    }:
        raise PerceptionRuntimeError("distribution inventory algorithm is unsupported")
    if require_posterior_gate and not strong_profile:
        raise PerceptionRuntimeError(
            "posterior gate runtime requires the strong installed-file byte lock"
        )
    if type(inventory["count"]) is not int or inventory["count"] <= 0:
        raise PerceptionRuntimeError("distribution inventory count must be positive")
    _require_sha256(inventory["sha256"], "distribution inventory sha256")

    packages = value["packages"]
    if not isinstance(packages, list):
        raise PerceptionRuntimeError("environment packages must be a list")
    normalized = {}
    for row in packages:
        if not isinstance(row, dict):
            raise PerceptionRuntimeError("environment package row is invalid")
        _exact_keys(row, {"name", "version", "record_sha256"}, "environment package")
        name = row["name"]
        if not isinstance(name, str) or name in normalized:
            raise PerceptionRuntimeError("environment package names must be unique strings")
        if not isinstance(row["version"], str) or not row["version"]:
            raise PerceptionRuntimeError("environment package version is invalid")
        _require_sha256(row["record_sha256"], "%s RECORD sha256" % name)
        normalized[name] = row["version"]
    if [row["name"] for row in packages] != sorted(normalized):
        raise PerceptionRuntimeError("environment package rows must be name-sorted")
    expected_packages = SUPPORTED_PACKAGES if gate_profile else LEGACY_SUPPORTED_PACKAGES
    if normalized != expected_packages:
        raise PerceptionRuntimeError("environment critical package profile is unsupported")

    modules = value["modules"]
    expected_modules = SUPPORTED_MODULES if gate_profile else LEGACY_SUPPORTED_MODULES
    if not isinstance(modules, dict) or set(modules) != expected_modules:
        raise PerceptionRuntimeError("environment module identity set is invalid")
    for name, identity in modules.items():
        if not isinstance(identity, dict):
            raise PerceptionRuntimeError("environment module identity is invalid: %s" % name)
        if strong_profile or legacy_byte_profile:
            _exact_keys(
                identity, {"algorithm", "count", "sha256"},
                "environment module %s" % name,
            )
            if identity["algorithm"] != MODULE_TREE_ALGORITHM:
                raise PerceptionRuntimeError(
                    "environment module tree algorithm is unsupported: %s" % name
                )
            if type(identity["count"]) is not int or identity["count"] <= 0:
                raise PerceptionRuntimeError(
                    "environment module tree count is invalid: %s" % name
                )
        else:
            _exact_keys(identity, {"sha256"}, "environment module %s" % name)
        _require_sha256(identity["sha256"], "%s module sha256" % name)

    cuda = value["cuda"]
    if not isinstance(cuda, dict):
        raise PerceptionRuntimeError("CUDA identity is invalid")
    _exact_keys(cuda, {"build", "architectures"}, "CUDA identity")
    if (
        cuda["build"] != "12.8"
        or cuda["architectures"] != SUPPORTED_CUDA_ARCHITECTURES
    ):
        raise PerceptionRuntimeError("CUDA build/architectures are unsupported")

    provisioning = value["provisioning"]
    if not isinstance(provisioning, dict):
        raise PerceptionRuntimeError("environment provisioning identity is invalid")
    provisioning_keys = {
        "mode", "source_environment_fingerprint_sha256", "wheelhouse_complete",
    }
    if gate_profile:
        provisioning_keys.add("source_environment_fingerprint_algorithm")
    _exact_keys(provisioning, provisioning_keys, "environment provisioning")
    if provisioning["mode"] != "verified_clone" or provisioning["wheelhouse_complete"] is not False:
        raise PerceptionRuntimeError(
            "Linbox perception environment must honestly identify the verified-clone strategy"
        )
    fingerprint = _require_sha256(
        provisioning["source_environment_fingerprint_sha256"],
        "source environment fingerprint",
    )
    if (
        gate_profile
        and provisioning["source_environment_fingerprint_algorithm"]
        != (
            SOURCE_ENVIRONMENT_FINGERPRINT_ALGORITHM
            if strong_profile
            else (
                LEGACY_BYTE_SOURCE_ENVIRONMENT_FINGERPRINT_ALGORITHM
                if legacy_byte_profile
                else LEGACY_SOURCE_ENVIRONMENT_FINGERPRINT_ALGORITHM
            )
        )
    ):
        raise PerceptionRuntimeError("source environment fingerprint algorithm is unsupported")
    environment_id = value["environment_id"]
    expected_environment_id = "linbox-perception-" + (
        environment_identity_sha256(value)[:20] if gate_profile else fingerprint[:20]
    )
    if (
        not isinstance(environment_id, str)
        or not _ENVIRONMENT_ID.fullmatch(environment_id)
        or environment_id != expected_environment_id
    ):
        raise PerceptionRuntimeError(
            "environment_id does not bind the complete environment identity"
        )
    return value


def is_strong_environment_lock(value: object) -> bool:
    validated = validate_environment_lock(value)
    if validated["schema"] == RELOCATION_AWARE_ENVIRONMENT_LOCK_SCHEMA:
        return True
    return (
        validated["schema"] == ENVIRONMENT_LOCK_SCHEMA
        and validated["distribution_inventory"].get("algorithm")
        == DISTRIBUTION_INVENTORY_ALGORITHM
        and all(
            identity.get("algorithm") == MODULE_TREE_ALGORITHM
            for identity in validated["modules"].values()
        )
    )


def is_relocation_aware_environment_lock(value: object) -> bool:
    """Return true only for the v3 RECORD-plus-full-tree lock."""

    validated = validate_environment_lock(value)
    return validated["schema"] == RELOCATION_AWARE_ENVIRONMENT_LOCK_SCHEMA


def _verify_expected_file(
    path: Union[str, Path], expected_sha256: object, expected_size: object, label: str,
) -> dict:
    candidate = Path(path)
    digest = _require_sha256(expected_sha256, "%s expected SHA-256" % label)
    if candidate.is_symlink() or not candidate.is_file():
        raise PerceptionRuntimeError("%s must be a regular file" % label)
    info = candidate.stat()
    if type(expected_size) is not int or expected_size < 0 or info.st_size != expected_size:
        raise PerceptionRuntimeError("%s size does not match the confirmed value" % label)
    actual = _sha256_file(candidate)
    if actual != digest:
        raise PerceptionRuntimeError("%s hash does not match the confirmed value" % label)
    return {"sha256": actual, "size": info.st_size}


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise PerceptionRuntimeError("runtime staging destination already exists")
    shutil.copy2(str(source), str(destination), follow_symlinks=False)
    if destination.is_symlink() or not destination.is_file():
        raise PerceptionRuntimeError("runtime staging copied a non-regular file")


def _copy_tree(source: Path, destination: Path) -> None:
    _tree_rows(source)
    shutil.copytree(str(source), str(destination), symlinks=True)
    _tree_rows(destination)


def _copy_dino_source_tree(source: Path, destination: Path) -> None:
    rows = _tree_rows(source, ignore_transient=True)
    destination.mkdir(parents=True)
    for row in rows:
        _copy_file(source / row["path"], destination / row["path"])
    _tree_rows(destination)


def _component_file(release: Path, relative: object, identity: Mapping[str, object], label: str) -> Path:
    normalized = _relative_path(relative, "%s path" % label)
    path = release / Path(*PurePosixPath(normalized).parts)
    _verify_expected_file(path, identity.get("sha256"), identity.get("size"), label)
    return path


def _verify_hardened_code_manifest(
    envelope: Mapping[str, object],
    expected_sha256: object = None,
    *,
    runtime_schema: str,
) -> dict[str, str]:
    if runtime_schema == RUNTIME_MANIFEST_SCHEMA:
        required_files = REQUIRED_HARDENED_CODE_FILES
        semantic_sha256 = REQUIRED_PRODUCTION_CHAIN_SEMANTIC_CODE_SHA256
        semantic_label = "production chain semantic code does not match runtime schema v4"
    elif runtime_schema == V3_RUNTIME_MANIFEST_SCHEMA:
        required_files = V3_REQUIRED_HARDENED_CODE_FILES
        semantic_sha256 = V3_REQUIRED_BACK_GATE_SEMANTIC_CODE_SHA256
        semantic_label = "operator back gate semantic code does not match runtime schema v3"
    elif runtime_schema == LEGACY_RUNTIME_MANIFEST_SCHEMA:
        required_files = V3_REQUIRED_HARDENED_CODE_FILES
        semantic_sha256 = {}
        semantic_label = ""
    else:
        raise PerceptionRuntimeError("hardened code runtime schema is unsupported")
    if runtime_schema == RUNTIME_MANIFEST_SCHEMA and set(semantic_sha256) != required_files:
        raise PerceptionRuntimeError(
            "runtime schema v4 production semantic pin set is incomplete"
        )
    payload = envelope.get("payload")
    rows = payload.get("files") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list):
        raise PerceptionRuntimeError("nested code release file manifest is invalid")
    rows_by_path = {
        row.get("path"): row for row in rows if isinstance(row, Mapping)
    }
    missing = sorted(required_files.difference(rows_by_path))
    if missing:
        raise PerceptionRuntimeError(
            "code release is missing required hardened files: %r" % missing
        )
    actual = {
        relative: _require_sha256(
            rows_by_path[relative].get("sha256"),
            "hardened code file %s SHA-256" % relative,
        )
        for relative in sorted(required_files)
    }
    if expected_sha256 is not None:
        if not isinstance(expected_sha256, Mapping):
            raise PerceptionRuntimeError("hardened code file confirmations must be a mapping")
        if set(expected_sha256) != required_files:
            raise PerceptionRuntimeError(
                "hardened code file confirmation set must be exact"
            )
        confirmed = {
            relative: _require_sha256(
                expected_sha256[relative],
                "confirmed hardened code file %s SHA-256" % relative,
            )
            for relative in sorted(required_files)
        }
        if confirmed != actual:
            raise PerceptionRuntimeError(
                "hardened code file hashes do not match the external confirmations"
            )
    if semantic_sha256:
        if {
            relative: actual[relative]
            for relative in semantic_sha256
        } != semantic_sha256:
            raise PerceptionRuntimeError(semantic_label)
    return actual


def _runtime_manifest(
    *, checkpoint: dict, posterior_gate_checkpoint: dict,
    dino_checkpoint: dict, dino_repo: dict,
    code_manifest: Mapping[str, object], environment: Mapping[str, object],
    environment_lock: dict, hardened_code_sha256: Mapping[str, str],
) -> dict:
    return {
        "schema": RUNTIME_MANIFEST_SCHEMA,
        "model": {
            "architecture": "NormalsDetector",
            "checkpoint_path": CHECKPOINT_PATH,
            "checkpoint_sha256": checkpoint["sha256"],
            "checkpoint_size": checkpoint["size"],
            "encoder": "dinov2_vitb14",
            "half": True,
            "landmark_order": LANDMARK_ORDER,
            "role": "sole_auscultation_point_source",
        },
        "posterior_gate": {
            "architecture": "YOLO11x-pose",
            "back_confirmation": "operator_required",
            "checkpoint_path": POSTERIOR_GATE_CHECKPOINT_PATH,
            "checkpoint_sha256": posterior_gate_checkpoint["sha256"],
            "checkpoint_size": posterior_gate_checkpoint["size"],
            "framework": "ultralytics",
            "module": "ultralytics",
            "orientation_authority": False,
            "person_presence_signal": "box_only",
            "produces_auscultation_points": False,
            "role": "single_person_presence_detector",
        },
        "dino": {
            "checkpoint_path": DINO_CHECKPOINT_PATH,
            "checkpoint_sha256": dino_checkpoint["sha256"],
            "checkpoint_size": dino_checkpoint["size"],
            "repository_path": DINO_REPOSITORY_PATH,
            "repository_files": dino_repo["files"],
            "repository_tree_sha256": dino_repo["sha256"],
        },
        "code": {
            "path": CODE_PATH,
            "release_id": code_manifest["release_id"],
            "payload_sha256": code_manifest["payload_sha256"],
            "hardened_file_sha256": dict(hardened_code_sha256),
        },
        "environment": {
            "environment_id": environment_lock["environment_id"],
            "lock_path": ENVIRONMENT_LOCK_PATH,
            "lock_sha256": environment["sha256"],
            "lock_size": environment["size"],
        },
        "entrypoint": {
            "detector_class": "tracking.detector.NormalsDetector",
            "runtime_module": "tracking.runtime",
            "tracker_class": "tracking.model_tracker.ModelBasedTracker",
        },
        "runtime": {
            "host_role": "linbox",
            "offline_assets_required": True,
            "torch_home": TORCH_HOME_PATH,
        },
    }


def verify_perception_runtime_release(release_dir: Union[str, Path]) -> dict:
    """Verify generic integrity and every perception-specific binding."""
    release = Path(release_dir)
    try:
        envelope = contract.verify_release(release)
    except contract.ArtifactError as exc:
        raise PerceptionRuntimeError("perception runtime outer release is invalid") from exc
    payload = envelope.get("payload")
    if not isinstance(payload, dict) or payload.get("artifact_kind") != RUNTIME_ARTIFACT_KIND:
        raise PerceptionRuntimeError("release is not a perception_runtime artifact")
    manifest, _raw = _load_canonical_object(
        release / RUNTIME_MANIFEST_NAME, "perception runtime manifest",
    )
    _exact_keys(manifest, {
        "schema", "model", "posterior_gate", "dino", "code", "environment",
        "entrypoint", "runtime",
    }, "perception runtime manifest")
    runtime_schema = manifest["schema"]
    if runtime_schema not in {
        RUNTIME_MANIFEST_SCHEMA,
        V3_RUNTIME_MANIFEST_SCHEMA,
        LEGACY_RUNTIME_MANIFEST_SCHEMA,
    }:
        raise PerceptionRuntimeError("perception runtime manifest schema is invalid")
    legacy_approval = None
    if runtime_schema == LEGACY_RUNTIME_MANIFEST_SCHEMA:
        legacy_approval = APPROVED_LEGACY_V2_ROLLBACKS.get(envelope["release_id"])
        if (
            legacy_approval is None
            or envelope["payload_sha256"] != legacy_approval["payload_sha256"]
        ):
            raise PerceptionRuntimeError(
                "legacy perception runtime is not an approved rollback"
            )

    model = manifest["model"]
    if not isinstance(model, dict):
        raise PerceptionRuntimeError("model identity is invalid")
    _exact_keys(model, {
        "architecture", "checkpoint_path", "checkpoint_sha256", "checkpoint_size",
        "encoder", "half", "landmark_order", "role",
    }, "model identity")
    if (
        model["architecture"] != "NormalsDetector"
        or model["encoder"] != "dinov2_vitb14"
        or model["half"] is not True
        or model["landmark_order"] != LANDMARK_ORDER
        or model["checkpoint_path"] != CHECKPOINT_PATH
        or model["role"] != "sole_auscultation_point_source"
    ):
        raise PerceptionRuntimeError("model semantic identity is unsupported")
    _component_file(release, model["checkpoint_path"], {
        "sha256": model["checkpoint_sha256"], "size": model["checkpoint_size"],
    }, "v9 checkpoint")

    posterior_gate = manifest["posterior_gate"]
    if not isinstance(posterior_gate, dict):
        raise PerceptionRuntimeError("posterior gate identity is invalid")
    if runtime_schema in {RUNTIME_MANIFEST_SCHEMA, V3_RUNTIME_MANIFEST_SCHEMA}:
        expected_posterior_gate = {
            "architecture": "YOLO11x-pose",
            "back_confirmation": "operator_required",
            "checkpoint_path": POSTERIOR_GATE_CHECKPOINT_PATH,
            "checkpoint_sha256": posterior_gate.get("checkpoint_sha256"),
            "checkpoint_size": posterior_gate.get("checkpoint_size"),
            "framework": "ultralytics",
            "module": "ultralytics",
            "orientation_authority": False,
            "person_presence_signal": "box_only",
            "produces_auscultation_points": False,
            "role": "single_person_presence_detector",
        }
    else:
        expected_posterior_gate = {
            "architecture": "YOLO11x-pose",
            "checkpoint_path": POSTERIOR_GATE_CHECKPOINT_PATH,
            "checkpoint_sha256": posterior_gate.get("checkpoint_sha256"),
            "checkpoint_size": posterior_gate.get("checkpoint_size"),
            "framework": "ultralytics",
            "module": "ultralytics",
            "produces_auscultation_points": False,
            "role": "posterior_human_pose_gate",
        }
    _exact_keys(
        posterior_gate, set(expected_posterior_gate), "posterior gate identity",
    )
    if posterior_gate != expected_posterior_gate:
        raise PerceptionRuntimeError("posterior gate semantic identity is unsupported")
    _component_file(release, posterior_gate["checkpoint_path"], {
        "sha256": posterior_gate["checkpoint_sha256"],
        "size": posterior_gate["checkpoint_size"],
    }, "posterior gate checkpoint")

    dino = manifest["dino"]
    if not isinstance(dino, dict):
        raise PerceptionRuntimeError("DINO identity is invalid")
    _exact_keys(dino, {
        "checkpoint_path", "checkpoint_sha256", "checkpoint_size", "repository_path",
        "repository_files", "repository_tree_sha256",
    }, "DINO identity")
    if dino["checkpoint_path"] != DINO_CHECKPOINT_PATH or dino["repository_path"] != DINO_REPOSITORY_PATH:
        raise PerceptionRuntimeError("DINO runtime layout is invalid")
    _component_file(release, dino["checkpoint_path"], {
        "sha256": dino["checkpoint_sha256"], "size": dino["checkpoint_size"],
    }, "DINO checkpoint")
    dino_root = release / Path(*PurePosixPath(dino["repository_path"]).parts)
    dino_identity = hash_regular_tree(dino_root)
    if (dino_identity["files"] != dino["repository_files"]
            or dino_identity["sha256"] != _require_sha256(
                dino["repository_tree_sha256"], "DINO repository tree SHA-256",
            )):
        raise PerceptionRuntimeError("DINO repository identity mismatch")
    if not (dino_root / "hubconf.py").is_file() or not (dino_root / "dinov2" / "__init__.py").is_file():
        raise PerceptionRuntimeError("DINO repository lacks its offline hub entrypoint")

    code = manifest["code"]
    if not isinstance(code, dict):
        raise PerceptionRuntimeError("code identity is invalid")
    _exact_keys(code, {
        "path", "release_id", "payload_sha256", "hardened_file_sha256",
    }, "code identity")
    if code["path"] != CODE_PATH:
        raise PerceptionRuntimeError("code release layout is invalid")
    nested = contract.verify_release(release / CODE_PATH)
    if (nested["payload"].get("artifact_kind") != "code_release"
            or nested["release_id"] != code["release_id"]
            or nested["payload_sha256"] != _require_sha256(
                code["payload_sha256"], "nested code payload SHA-256",
            )):
        raise PerceptionRuntimeError("nested code release identity mismatch")
    if legacy_approval is not None and (
        code["release_id"] != legacy_approval["code_release_id"]
        or code["payload_sha256"] != legacy_approval["code_payload_sha256"]
    ):
        raise PerceptionRuntimeError(
            "legacy perception runtime code is not the approved rollback code"
        )
    _verify_hardened_code_manifest(
        nested,
        code["hardened_file_sha256"],
        runtime_schema=runtime_schema,
    )

    environment = manifest["environment"]
    if not isinstance(environment, dict):
        raise PerceptionRuntimeError("environment identity is invalid")
    _exact_keys(environment, {
        "environment_id", "lock_path", "lock_sha256", "lock_size",
    }, "environment identity")
    if environment["lock_path"] != ENVIRONMENT_LOCK_PATH:
        raise PerceptionRuntimeError("environment lock layout is invalid")
    lock_path = _component_file(release, environment["lock_path"], {
        "sha256": environment["lock_sha256"], "size": environment["lock_size"],
    }, "environment lock")
    lock, _lock_raw = _load_canonical_object(lock_path, "environment lock")
    validate_environment_lock(lock, require_posterior_gate=True)
    if environment["environment_id"] != lock["environment_id"]:
        raise PerceptionRuntimeError("environment manifest does not bind the lock environment_id")

    expected_entrypoint = {
        "detector_class": "tracking.detector.NormalsDetector",
        "runtime_module": "tracking.runtime",
        "tracker_class": "tracking.model_tracker.ModelBasedTracker",
    }
    expected_runtime = {
        "host_role": "linbox", "offline_assets_required": True,
        "torch_home": TORCH_HOME_PATH,
    }
    if manifest["entrypoint"] != expected_entrypoint or manifest["runtime"] != expected_runtime:
        raise PerceptionRuntimeError("runtime entrypoint/environment semantics are invalid")
    return envelope


def build_perception_runtime_release(
    *, checkpoint: Union[str, Path], expected_checkpoint_sha256: str,
    expected_checkpoint_size: int,
    posterior_gate_checkpoint: Union[str, Path],
    expected_posterior_gate_checkpoint_sha256: str,
    expected_posterior_gate_checkpoint_size: int,
    dino_checkpoint: Union[str, Path],
    expected_dino_checkpoint_sha256: str, expected_dino_checkpoint_size: int,
    dino_repo: Union[str, Path], expected_dino_repo_sha256: str,
    code_release: Union[str, Path], expected_code_release_id: str,
    expected_code_payload_sha256: str,
    expected_hardened_code_sha256: Mapping[str, str],
    environment_lock: Union[str, Path],
    expected_environment_lock_sha256: str, releases_root: Union[str, Path],
) -> Path:
    """Build one completed local release after all identities are double-confirmed."""
    checkpoint_path = Path(checkpoint)
    posterior_gate_checkpoint_path = Path(posterior_gate_checkpoint)
    dino_checkpoint_path = Path(dino_checkpoint)
    dino_repo_path = Path(dino_repo)
    code_release_path = Path(code_release)
    environment_lock_path = Path(environment_lock)

    checkpoint_identity = _verify_expected_file(
        checkpoint_path, expected_checkpoint_sha256, expected_checkpoint_size, "v9 checkpoint",
    )
    posterior_gate_checkpoint_identity = _verify_expected_file(
        posterior_gate_checkpoint_path,
        expected_posterior_gate_checkpoint_sha256,
        expected_posterior_gate_checkpoint_size,
        "posterior gate checkpoint",
    )
    dino_checkpoint_identity = _verify_expected_file(
        dino_checkpoint_path, expected_dino_checkpoint_sha256,
        expected_dino_checkpoint_size, "DINO checkpoint",
    )
    dino_repo_identity = hash_dino_source_tree(dino_repo_path)
    if dino_repo_identity["sha256"] != _require_sha256(
        expected_dino_repo_sha256, "confirmed DINO repository SHA-256",
    ):
        raise PerceptionRuntimeError("DINO repository does not match the confirmed tree identity")
    if not (dino_repo_path / "hubconf.py").is_file() or not (dino_repo_path / "dinov2" / "__init__.py").is_file():
        raise PerceptionRuntimeError("DINO repository lacks its offline hub entrypoint")

    code_manifest = contract.verify_release(code_release_path)
    if (code_manifest["payload"].get("artifact_kind") != "code_release"
            or code_manifest["release_id"] != expected_code_release_id
            or code_manifest["payload_sha256"] != _require_sha256(
                expected_code_payload_sha256, "confirmed code payload SHA-256",
            )):
        raise PerceptionRuntimeError("code release does not match both confirmations")
    hardened_code_sha256 = _verify_hardened_code_manifest(
        code_manifest,
        expected_hardened_code_sha256,
        runtime_schema=RUNTIME_MANIFEST_SCHEMA,
    )

    lock, lock_raw = _load_canonical_object(environment_lock_path, "environment lock")
    validate_environment_lock(lock, require_posterior_gate=True)
    lock_sha256 = hashlib.sha256(lock_raw).hexdigest()
    if lock_sha256 != _require_sha256(
        expected_environment_lock_sha256, "confirmed environment lock SHA-256",
    ):
        raise PerceptionRuntimeError("environment lock does not match the confirmed identity")
    environment_identity = {"sha256": lock_sha256, "size": len(lock_raw)}

    releases = Path(releases_root)
    if releases.is_symlink():
        raise PerceptionRuntimeError("releases root must not be a symlink")
    releases.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix=".perception-runtime-build-", dir=str(releases.parent)))
    source = workspace / "source"
    source.mkdir()
    try:
        _copy_file(checkpoint_path, source / CHECKPOINT_PATH)
        _copy_file(
            posterior_gate_checkpoint_path,
            source / POSTERIOR_GATE_CHECKPOINT_PATH,
        )
        _copy_file(dino_checkpoint_path, source / DINO_CHECKPOINT_PATH)
        _copy_dino_source_tree(dino_repo_path, source / DINO_REPOSITORY_PATH)
        _copy_tree(code_release_path, source / CODE_PATH)
        _copy_file(environment_lock_path, source / ENVIRONMENT_LOCK_PATH)

        _verify_expected_file(
            source / CHECKPOINT_PATH, checkpoint_identity["sha256"],
            checkpoint_identity["size"], "staged v9 checkpoint",
        )
        _verify_expected_file(
            source / POSTERIOR_GATE_CHECKPOINT_PATH,
            posterior_gate_checkpoint_identity["sha256"],
            posterior_gate_checkpoint_identity["size"],
            "staged posterior gate checkpoint",
        )
        _verify_expected_file(
            source / DINO_CHECKPOINT_PATH, dino_checkpoint_identity["sha256"],
            dino_checkpoint_identity["size"], "staged DINO checkpoint",
        )
        if hash_regular_tree(source / DINO_REPOSITORY_PATH) != dino_repo_identity:
            raise PerceptionRuntimeError("DINO repository changed while staging")
        staged_code = contract.verify_release(source / CODE_PATH)
        if staged_code != code_manifest:
            raise PerceptionRuntimeError("code release changed while staging")
        staged_lock, staged_lock_raw = _load_canonical_object(
            source / ENVIRONMENT_LOCK_PATH, "staged environment lock",
        )
        validate_environment_lock(staged_lock, require_posterior_gate=True)
        if hashlib.sha256(staged_lock_raw).hexdigest() != lock_sha256:
            raise PerceptionRuntimeError("environment lock changed while staging")

        semantic = _runtime_manifest(
            checkpoint=checkpoint_identity,
            posterior_gate_checkpoint=posterior_gate_checkpoint_identity,
            dino_checkpoint=dino_checkpoint_identity,
            dino_repo=dino_repo_identity,
            code_manifest=code_manifest,
            environment=environment_identity,
            environment_lock=lock,
            hardened_code_sha256=hardened_code_sha256,
        )
        (source / RUNTIME_MANIFEST_NAME).write_bytes(contract.canonical_json_bytes(semantic))
        relative_paths = tuple(_regular_files(source))
        manifest = contract.build_manifest(
            source, relative_paths, artifact_kind=RUNTIME_ARTIFACT_KIND, source_role="mac",
        )
        destination = releases / manifest["release_id"]
        contract.write_release(source, manifest, destination, target_role="mac")
        verify_perception_runtime_release(destination)
        return destination
    finally:
        shutil.rmtree(str(workspace), ignore_errors=True)


def activate_perception_runtime_release(
    source_release: Union[str, Path], *, releases_root: Union[str, Path],
    current_link: Union[str, Path], expected_current: object,
) -> Path:
    """Install and CAS-activate one semantic release with rollback on post-check."""
    source = Path(source_release)
    verify_perception_runtime_release(source)
    return contract.install_release(
        source,
        releases_root,
        current_link,
        target_role="linbox",
        expected_current=expected_current,
        post_activate=verify_perception_runtime_release,
    )


def offline_runtime_environment(release_dir: Union[str, Path]) -> Dict[str, str]:
    """Return the fail-closed process environment for a verified release.

    The returned paths are release-local and absolute.  Callers should install
    this mapping into the perception process environment before importing the
    model modules; no user-global torch cache is part of this contract.
    """
    release = Path(release_dir).resolve(strict=True)
    verify_perception_runtime_release(release)
    return {
        "AUSC_DINOV2_CHECKPOINT": str(release / DINO_CHECKPOINT_PATH),
        "AUSC_DINOV2_REPO": str(release / DINO_REPOSITORY_PATH),
        "AUSC_OFFLINE_RUNTIME": "1",
        "AUSC_POSTERIOR_GATE_CHECKPOINT": str(
            release / POSTERIOR_GATE_CHECKPOINT_PATH
        ),
        "HF_HUB_OFFLINE": "1",
        "TORCH_HOME": str(release / TORCH_HOME_PATH),
        "TRANSFORMERS_OFFLINE": "1",
    }


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build, verify, or CAS-activate an immutable perception runtime",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="build a fully confirmed local release")
    build.add_argument("--checkpoint", required=True)
    build.add_argument("--checkpoint-sha256", required=True)
    build.add_argument("--checkpoint-size", required=True, type=int)
    build.add_argument("--posterior-gate-checkpoint", required=True)
    build.add_argument("--posterior-gate-checkpoint-sha256", required=True)
    build.add_argument("--posterior-gate-checkpoint-size", required=True, type=int)
    build.add_argument("--dino-checkpoint", required=True)
    build.add_argument("--dino-checkpoint-sha256", required=True)
    build.add_argument("--dino-checkpoint-size", required=True, type=int)
    build.add_argument("--dino-repo", required=True)
    build.add_argument("--dino-repo-sha256", required=True)
    build.add_argument("--code-release", required=True)
    build.add_argument("--code-release-id", required=True)
    build.add_argument("--code-payload-sha256", required=True)
    build.add_argument(
        "--hardened-code-sha256",
        required=True,
        help="canonical JSON mapping of required hardened paths to externally confirmed hashes",
    )
    build.add_argument("--environment-lock", required=True)
    build.add_argument("--environment-lock-sha256", required=True)
    build.add_argument("--releases-root", required=True)

    verify = commands.add_parser("verify", help="verify a release without mutating state")
    verify.add_argument("release")

    activate = commands.add_parser("activate", help="install and CAS-activate a release")
    activate.add_argument("release")
    activate.add_argument("--releases-root", required=True)
    activate.add_argument("--current-link", required=True)
    expectation = activate.add_mutually_exclusive_group(required=True)
    expectation.add_argument("--expected-current")
    expectation.add_argument("--expect-no-current", action="store_true")
    return parser


def _emit(value: object) -> None:
    sys.stdout.write(contract.canonical_json_bytes(value).decode("utf-8"))


def main(argv: object = None) -> int:
    """Run the explicit, force-free perception-runtime command line."""
    args = _argument_parser().parse_args(argv)
    if args.command == "build":
        hardened, _raw = _load_canonical_object(
            args.hardened_code_sha256, "hardened code hash confirmations",
        )
        release = build_perception_runtime_release(
            checkpoint=args.checkpoint,
            expected_checkpoint_sha256=args.checkpoint_sha256,
            expected_checkpoint_size=args.checkpoint_size,
            posterior_gate_checkpoint=args.posterior_gate_checkpoint,
            expected_posterior_gate_checkpoint_sha256=(
                args.posterior_gate_checkpoint_sha256
            ),
            expected_posterior_gate_checkpoint_size=(
                args.posterior_gate_checkpoint_size
            ),
            dino_checkpoint=args.dino_checkpoint,
            expected_dino_checkpoint_sha256=args.dino_checkpoint_sha256,
            expected_dino_checkpoint_size=args.dino_checkpoint_size,
            dino_repo=args.dino_repo,
            expected_dino_repo_sha256=args.dino_repo_sha256,
            code_release=args.code_release,
            expected_code_release_id=args.code_release_id,
            expected_code_payload_sha256=args.code_payload_sha256,
            expected_hardened_code_sha256=hardened,
            environment_lock=args.environment_lock,
            expected_environment_lock_sha256=args.environment_lock_sha256,
            releases_root=args.releases_root,
        ).resolve()
        _emit({
            "command": "build",
            "release_dir": str(release),
            "release_id": release.name,
        })
        return 0
    if args.command == "verify":
        envelope = verify_perception_runtime_release(args.release)
        _emit({"command": "verify", "release_id": envelope["release_id"]})
        return 0
    if args.command == "activate":
        installed = activate_perception_runtime_release(
            args.release,
            releases_root=args.releases_root,
            current_link=args.current_link,
            expected_current=None if args.expect_no_current else args.expected_current,
        ).resolve()
        _emit({
            "command": "activate",
            "current": os.readlink(args.current_link),
            "release_dir": str(installed),
            "release_id": installed.name,
        })
        return 0
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
