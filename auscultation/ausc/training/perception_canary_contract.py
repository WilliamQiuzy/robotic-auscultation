"""Fail-closed core identity contract for visible-stability perception canaries.

This module validates the immutable, H200-facing configuration envelope and
descriptor-pins, rehashes, semantically validates, and re-verifies its exact three
completed training-input releases.  The public API fixes all production roots;
private injection points exist only for deterministic contract tests.

Output reservation trusts processes running under the same effective UID as the
H200 training account.  Its owner-and-mode gate excludes writes by other users
and groups; it does not claim to defend against malicious same-EUID code.
"""

from copy import deepcopy
from contextlib import contextmanager
import csv
import ctypes
from dataclasses import dataclass, field
import errno
import hashlib
import inspect
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Callable, Mapping, Optional, Union
import uuid
import zipfile

from scripts.deployment import artifact_contract as _artifact_contract


CANARY_SCHEMA = "wam_perception_visible_stability_canary_v1"
RUN_ID_PREFIX = "perc_lmh_v9p1_visible_stability_canary-"
POLICY_RELATIVE_PATH = (
    "auscultation/configs/perception/visible-stability-mask-policy-v1.json"
)
URDF_RELATIVE_PATH = "auscultation/ausc/perception/robot/openarm_v10.urdf"

_DEFAULT_CODE_ROOT = Path("/home/ssh-ziyue/wam_ausc/.releases")
_DEFAULT_ENVIRONMENT_ROOT = Path("/home/ssh-ziyue/envs/releases")
_DEFAULT_OUTPUT_ROOT = Path(
    "/home/ssh-ziyue/wam_ausc/data/outputs/perception/canaries"
)
_DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]
PERCEPTION_INPUT_NAME = "PERCEPTION_INPUT.json"
PERCEPTION_INPUT_SCHEMA = "wam_perception_training_input_v1"
PERCEPTION_INPUT_RELEASE_ROOTS = {
    "geometry_source": Path(
        "/home/ssh-ziyue/wam_ausc/data/perception_geometry_source/releases"
    ),
    "stage0": Path(
        "/home/ssh-ziyue/wam_ausc/data/perception_stage0/releases"
    ),
    "train": Path(
        "/home/ssh-ziyue/wam_ausc/data/perception_session2_train/releases"
    ),
}
PERCEPTION_INPUT_FIELDS = frozenset(
    (
        "artifact_kind",
        "contract",
        "members",
        "resolvable_dependencies",
        "schema",
        "self_contained",
        "session_ids",
        "source_attestations",
        "source_role",
    )
)
PERCEPTION_INPUT_MEMBER_FIELDS = frozenset(
    ("content_origin", "path", "role", "sha256", "size")
)
PERCEPTION_SOURCE_ATTESTATION_FIELDS = frozenset(
    (
        "artifact_kind",
        "content_partition",
        "payload_sha256",
        "release_id",
        "release_manifest_sha256",
    )
)
MIXED_EVAL_CONTENT_PARTITION_FIELDS = frozenset(
    (
        "decoded_frame_hash_version",
        "schema",
        "session2_decoded_frame_sha256",
        "session2_selected_sha256",
        "session3_decoded_frame_sha256",
        "session3_denied_sha256",
        "shared_intrinsics_sha256",
        "source_manifest_sha256",
        "source_payload_sha256",
        "source_release_id",
    )
)
DENIED_TRAIN_CONTENT_SHA256 = frozenset(
    (
        "075671526052c4cfd0cb59706254dd8b29f011aa5c33b268022a37df3cdbee37",
    )
)
ALLOWED_SHARED_CONTENT_ORIGINS = frozenset(("shared_intrinsics",))
HISTORICAL_MIXED_EVAL_RELEASE_ID = "perception_eval-efc716189fc37d8886ba"
HISTORICAL_MIXED_EVAL_PAYLOAD_SHA256 = (
    "efc716189fc37d8886ba9d60ac3cdb0cbfc8204e9b149ccee91216e65fff49a9"
)
HISTORICAL_MIXED_EVAL_MANIFEST_SHA256 = (
    "952f1b266427828f8b83ba870f3a16d2867a2b11dd28f895ee6cf06497a6c5dd"
)
HISTORICAL_MIXED_EVAL_PARTITION_SHA256 = (
    "dace556e5191e3bfe95f1dbfd7484d633888a9ae1d5ee8a102a1638e735fa490"
)
HISTORICAL_SESSION2_SELECTED_LIST_SHA256 = (
    "f003515e44da86cde5c4a916eb65f9e51c6e2c3f87ef895f4d161b140db23b4b"
)
HISTORICAL_SESSION3_DENIED_LIST_SHA256 = (
    "bb90159332274b43d3d9f3f88fe02092dfe93422a91497512c39d37c1cad0f85"
)
HISTORICAL_SESSION2_DECODED_LIST_SHA256 = (
    "6043e8e62675dd9020d0b68ae7d71b50a08d5d7f4aa4d67c208d511f9dbbb612"
)
HISTORICAL_SESSION3_DECODED_LIST_SHA256 = (
    "4c824067e6d497e0b579429f7a775bf2987e19201ff50e08b2dd2ad5858ef2e9"
)
HISTORICAL_SHARED_INTRINSICS_LIST_SHA256 = (
    "61b8a4d5d54293a07e9bd52e2d984190aa803ac05c468b0e849b1014e169322c"
)
HISTORICAL_SESSION2_ORDERED_DEPTH_LIST_SHA256 = (
    "90e4337de51afb9b7d8e4f267bdd03341a228114f9302ae0b74a72a9335fee8f"
)
MIXED_EVAL_CONTENT_PARTITION_SCHEMA = "wam_perception_eval_content_partition_v1"
DECODED_FRAME_HASH_VERSION = "wam_bgr8_frame_sha256_v1"
DECODED_FRAME_DTYPE = "uint8"
DECODED_FRAME_SHAPE = (1080, 1920, 3)
STAGE0_SAMPLE_LEDGER_SCHEMA = "wam_perception_stage0_sample_ledger_v1"
STAGE0_SUBJECT_SPLIT_SCHEMA = "wam_perception_stage0_subject_split_v1"
STAGE0_SAMPLE_FORMAT = "wam_stage0_npz_depth_intrinsics_normals_points_v1"
STAGE0_PROVENANCE_SCHEMA = "wam_perception_stage0_provenance_v1"
MASK_POLICY_SHA256 = (
    "ad5a867cc62aa3164eb6f69123d72057c005acd0d6af8dd090e38ec7f5a5525b"
)
SESSION2_FRAME_LEDGER_SCHEMA = "wam_perception_session2_frame_ledger_v1"
STAGE0_IMAGE_SHAPE = (518, 518)
STAGE0_INTRINSICS_PX = (
    (560.0, 0.0, 259.0),
    (0.0, 560.0, 259.0),
    (0.0, 0.0, 1.0),
)
STAGE0_POINT_ORDER = (
    "R1",
    "L1",
    "R2",
    "L2",
    "R3",
    "L3",
    "R4",
    "L4",
    "R5",
    "L5",
)
STAGE0_DEPTH_HOLE_FRACTION_MAX = 0.4
# The geometry exporter already rejects a rendered back with fewer than 3,000
# supporting raster pixels.  The sealed sample must never weaken that gate.
STAGE0_MIN_POSITIVE_DEPTH_PIXELS = 3000
STAGE0_MIN_UNIQUE_VISIBLE_POINTS = 10
STAGE0_MIN_POINT_AXIS_SPAN_PX = 8.0
STAGE0_MIN_POINT_MINOR_SINGULAR_VALUE_PX = 2.0
STAGE0_MIN_VALID_NORMAL_FRACTION = 0.95
STAGE0_MIN_DECODED_NORMAL_NORM = 0.95
STAGE0_MAX_DECODED_NORMAL_NORM = 1.05
STAGE0_NPZ_MAX_COMPRESSED_BYTES = 2 * 1024 * 1024
STAGE0_NPY_MAX_HEADER_BYTES = 512
STAGE0_NPZ_MAX_COMPRESSION_RATIO = 512.0
_STAGE0_NPZ_ARRAY_SPECS = {
    "depth_m": ("float32", STAGE0_IMAGE_SHAPE, 4),
    "intrinsics_px": ("float64", (3, 3), 8),
    "normal_u8": ("uint8", STAGE0_IMAGE_SHAPE + (3,), 1),
    "point_order": ("<U2", (10,), 8),
    "points_px": ("float32", (10, 2), 4),
    "sample_id": ("<U18", (), 72),
    "sequence_id": ("<U15", (), 60),
    "source_frame_index": ("int32", (), 4),
    "subject_id": ("<U7", (), 28),
    "view_index": ("int16", (), 2),
    "visibility": ("bool", (10,), 1),
}
STAGE0_NPZ_MAX_EXPANDED_BYTES = sum(
    math.prod(shape or (1,)) * item_size + STAGE0_NPY_MAX_HEADER_BYTES
    for _, shape, item_size in _STAGE0_NPZ_ARRAY_SPECS.values()
)
_HUMMAN_SEQUENCE_ID = re.compile(r"^(p[0-9]{6})_a[0-9]{6}$")
_STAGE0_SAMPLE_ID = re.compile(r"^(p[0-9]{6}_a[0-9]{6})_v([0-3])$")

SESSION2_RGB_VIDEO_PATH = "video/session2.mp4"
SESSION2_SOURCE_FRAMES_PATH = "source/frames.csv"
SESSION2_INTRINSICS_PATH = "camera/intrinsics.json"
SESSION2_LABELS_PATH = "labels/dark.json"
SESSION2_FRAME_LEDGER_PATH = "ledgers/frames.json"
SESSION2_COORDINATE_METADATA_PATH = "metadata/coordinates.json"
SESSION2_COORDINATE_METADATA_SHA256 = (
    "b65512869467d5d4bb1d8b5b19d59c5590690a32ec64a850436c62cf3287eea3"
)
SESSION2_RGB_VIDEO_SHA256 = (
    "ac43ac4ef1f3fd977a83abdc22aa73c19ec4cd8f7034a0df44f1e76bf22c1699"
)
SESSION2_SOURCE_FRAMES_SHA256 = (
    "9fd13926e840a5e6d98fda651d889f4236292b66c600329c39dc6be5e0f8a625"
)
SESSION2_INTRINSICS_SHA256 = (
    "b05ce66978015518bcbe4dc39e09a5dbef6606153ccc04eab0eac0fc4f3875b3"
)
SESSION2_DARK_LABEL_SHA256 = (
    "5917d20aae86c2ee978dcc51ff42c61fa94fbf592ff9330a6ccf49e90a6577cb"
)
HUMMAN_SMPL_PARAMS_MANIFEST_SHA256 = (
    "4ceeaf2616a0b62948abed3bf366b73fc057e5c0cf68abe488dfbc250021350b"
)
GEOMETRY_SMPL_NEUTRAL_SHA256 = (
    "b93df2dd228acc91d4ea6ede5f83f81e5171e7bee35c39695e6e27cb25ee1d4a"
)
GEOMETRY_DENSEPOSE_UV_SHA256 = (
    "ef109a96e57ddadd67b567ac591ec0bfe728c8a567f29927fc85500eb3cd374b"
)
GEOMETRY_HUMMAN_LICENSE_SHA256 = (
    "bec0dbad332972acedb0963fe221053590d7f556a0c15ef562062abdc1894a0e"
)
GEOMETRY_SMPL_LICENSE_SHA256 = (
    "15df63171dab571bcbacd9a36e734583fdcd26833365c51c781fa1c46f543075"
)
GEOMETRY_DENSEPOSE_LICENSE_SHA256 = (
    "336255dc30193e8e15d689d9481bb05673d89055718f3a96923a7ffb99adbbaf"
)
_PRODUCTION_GEOMETRY_SUPPORT_SHA256 = {
    "densepose/UV_Processed.mat": GEOMETRY_DENSEPOSE_UV_SHA256,
    "licenses/DENSEPOSE_SOURCE_LICENSE.txt": GEOMETRY_DENSEPOSE_LICENSE_SHA256,
    "licenses/HUMMAN_LICENSE.txt": GEOMETRY_HUMMAN_LICENSE_SHA256,
    "licenses/SMPL_MODEL_LICENSE.html": GEOMETRY_SMPL_LICENSE_SHA256,
    "smpl/SMPL_NEUTRAL.pkl": GEOMETRY_SMPL_NEUTRAL_SHA256,
}


def _build_session2_coordinate_metadata_expected() -> dict:
    return {
        "decoded_frame_hash_version": DECODED_FRAME_HASH_VERSION,
        "depth_unit": "metres",
        "garment": "dark",
        "height_px": DECODED_FRAME_SHAPE[0],
        "landmark_order": list(STAGE0_POINT_ORDER),
        "pixel_axis_order": "xy",
        "pixel_index_base": 0,
        "pixel_origin": "top_left",
        "rgb_depth_registration": "registered",
        "schema": "wam_perception_session2_coordinate_metadata_v1",
        "session_id": "orbbec_session2",
        "subject_pseudonym": "same-subject-001",
        "width_px": DECODED_FRAME_SHAPE[1],
    }


SESSION2_COORDINATE_METADATA_EXPECTED = (
    _build_session2_coordinate_metadata_expected()
)


def _build_geometry_source_terms_expected(
    support_sha256: Mapping[str, str],
    smpl_manifest_sha256: str,
) -> dict:
    return {
        "commercial_use": False,
        "densepose": {
            "archive_sha256": (
                "607376797294512e2dd57caa45f90990406ba569f0607008ae6f4fd5754f1392"
            ),
            "archive_url": (
                "https://dl.fbaipublicfiles.com/densepose/"
                "densepose_uv_data.tar.gz"
            ),
            "asset_license_status": "UNRESOLVED",
            "asset_path": "densepose/UV_Processed.mat",
            "asset_sha256": support_sha256["densepose/UV_Processed.mat"],
            "license_path": "licenses/DENSEPOSE_SOURCE_LICENSE.txt",
            "license_sha256": support_sha256[
                "licenses/DENSEPOSE_SOURCE_LICENSE.txt"
            ],
            "object_version_id": "ac483846kSIll9eVFurMR6akUzJH05ZN",
            "redistribution": False,
            "revision": "51fb9f110a454c77b0c5acfa0c7cceb317f38aa1",
        },
        "humman": {
            "archive_sha256": (
                "2cccdcee712ea7f1f5a06e7621c9f39910317955b5063339d9d131d65399a4fd"
            ),
            "archive_url": (
                "https://huggingface.co/datasets/caizhongang/HuMMan/resolve/"
                "56a2a9a12e21abe10744580f96989534cdbdff76/"
                "humman_release_v1.0_point/smpl_params.7z"
            ),
            "license_path": "licenses/HUMMAN_LICENSE.txt",
            "license_sha256": support_sha256["licenses/HUMMAN_LICENSE.txt"],
            "member_count": 907,
            "member_manifest_sha256": smpl_manifest_sha256,
            "revision": "56a2a9a12e21abe10744580f96989534cdbdff76",
        },
        "intended_use": "noncommercial_scientific_research",
        "schema": "wam_perception_geometry_source_terms_v1",
        "scope": "private_h200_internal_only",
        "smpl": {
            "asset_path": "smpl/SMPL_NEUTRAL.pkl",
            "asset_sha256": support_sha256["smpl/SMPL_NEUTRAL.pkl"],
            "exportable": False,
            "license_acceptance_required": True,
            "license_path": "licenses/SMPL_MODEL_LICENSE.html",
            "license_sha256": support_sha256[
                "licenses/SMPL_MODEL_LICENSE.html"
            ],
            "license_url": "https://smpl.is.tue.mpg.de/modellicense.html",
            "source_version": "1.1.0-cleaned-numpy",
            "storage_scope": "private_h200_internal_only",
            "upstream_archive_transform_hash": "UNAVAILABLE_LEGACY",
        },
    }


GEOMETRY_SOURCE_TERMS_EXPECTED = _build_geometry_source_terms_expected(
    _PRODUCTION_GEOMETRY_SUPPORT_SHA256,
    HUMMAN_SMPL_PARAMS_MANIFEST_SHA256,
)
_PRODUCTION_PARTITION_LIST_SHA256 = {
    "session2_decoded_frame_sha256": HISTORICAL_SESSION2_DECODED_LIST_SHA256,
    "session2_selected_sha256": HISTORICAL_SESSION2_SELECTED_LIST_SHA256,
    "session3_decoded_frame_sha256": HISTORICAL_SESSION3_DECODED_LIST_SHA256,
    "session3_denied_sha256": HISTORICAL_SESSION3_DENIED_LIST_SHA256,
    "shared_intrinsics_sha256": HISTORICAL_SHARED_INTRINSICS_LIST_SHA256,
}
_PRODUCTION_TRAIN_MEMBER_SHA256 = {
    "coordinate_metadata": SESSION2_COORDINATE_METADATA_SHA256,
    "intrinsics": SESSION2_INTRINSICS_SHA256,
    "labels": SESSION2_DARK_LABEL_SHA256,
    "rgb_video": SESSION2_RGB_VIDEO_SHA256,
    "source_frame_index": SESSION2_SOURCE_FRAMES_SHA256,
}

_PERCEPTION_INPUT_EXPECTATIONS = {
    "geometry_source": {
        "artifact_kind": "perception_geometry_source",
        "source_role": "mac",
        "session_ids": [],
    },
    "stage0": {
        "artifact_kind": "perception_stage0",
        "source_role": "h200",
        "session_ids": [],
    },
    "train": {
        "artifact_kind": "perception_session2_train",
        "source_role": "h200",
        "session_ids": ["orbbec_session2"],
    },
}
_ENVIRONMENT_RELEASE_ID = "wam-5d2be01285ba34687f25"
_ENVIRONMENT_IDENTITY_SHA256 = (
    "5d2be01285ba34687f257479429f8c43f2046fbb05ae2fa43ecf82f40c5d65d7"
)
_URDF_SHA256 = "092ac325b834e69b3cca703c9711d0166f6c8cad54356926968bca562e13f0d5"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CODE_RELEASE_ID = re.compile(r"^code_release-([0-9a-f]{20})$")
_RUN_ID = re.compile(r"^" + re.escape(RUN_ID_PREFIX) + r"[0-9a-f]{20}$")
_FORBIDDEN_TERMS = ("holdout", "yellow", "evaluation")

_TOP_LEVEL_FIELDS = frozenset(
    (
        "schema",
        "code",
        "environment",
        "warm_start",
        "inputs",
        "seeds",
        "training",
        "mask_policy",
    )
)
_INPUT_FIELDS = frozenset(
    (
        "release_id",
        "payload_sha256",
        "release_manifest_sha256",
        "semantic_manifest_sha256",
    )
)
_INPUT_PREFIXES = {
    "geometry_source": "perception_geometry_source",
    "stage0": "perception_stage0",
    "train": "perception_session2_train",
}
_WARM_START = {
    "architecture": "NormalsDetector",
    "checkpoint_path": "model/perc_lmh_v9_occ/best.pt",
    "checkpoint_sha256": (
        "68037e6932fd7315af33722a69060f0d9838afbc1a60d6f0349da909917f6a14"
    ),
    "encoder": "dinov2_vitb14",
    "landmark_order": [
        "R1",
        "L1",
        "R2",
        "L2",
        "R3",
        "L3",
        "R4",
        "L4",
        "R5",
        "L5",
    ],
    "lora_alpha": 32,
    "lora_rank": 16,
    "payload_sha256": (
        "33641f649a1fd658101e4cbb5cdd5cf87d2144c4868d8f2abfd5054b3376c6a1"
    ),
    "release_id": "perception_runtime-33641f649a1fd658101e",
}
_EXPECTED_POLICY = {
    "depth_hole_fraction": {
        "max": STAGE0_DEPTH_HOLE_FRACTION_MAX,
        "min": 0.0,
    },
    "effective_visible_requires": [
        "in_image",
        "finite_valid_depth",
        "outside_dilated_robot_mask",
        "outside_depth_holes",
    ],
    "preprocessing": {
        "identifier": "wam_robot_mask_before_depth_to_normals_v1",
        "order": ["robot_mask", "depth_to_normals"],
    },
    "rasterization": {
        "camera_sampling": "random_valid_calibrated_camera",
        "geometry": "official_urdf_links",
        "pose_sampling": "random_within_urdf_joint_limits",
        "scale_up": False,
    },
    "robot_model": {
        "scale": 1.0,
        "units": "m",
        "urdf_path": URDF_RELATIVE_PATH,
        "urdf_sha256": _URDF_SHA256,
    },
    "schema": "wam_perception_visible_stability_mask_policy_v1",
    "tracker": {
        "masked_status": "never_measured",
        "policy": "model_based_tracker_visible_support",
        "predicted_coast": {
            "max_age_s": 0.4,
            "max_translation_m": 0.01,
            "then": "LOST",
        },
        "statuses": ["measured", "predicted", "LOST"],
        "version": "bounded_patient_relative_coast_v1",
    },
    "uncertainty_dilation_m": {
        "full_6x6_covariance_replay": False,
        "max": 0.02,
        "min": 0.0,
        "research_basis": (
            "envelope_around_provisional_approximately_0.005_m_translation_precision"
        ),
    },
    "version": "visible-stability-mask-policy-v1",
}


class CanaryContractError(RuntimeError):
    """The requested canary violates its immutable training contract."""


class _InputVerifierNotInstalled(CanaryContractError):
    """Internal marker for the intentional Task-2A production boundary."""


class _OutputRootAuthorityChanged(CanaryContractError):
    """Internal marker that prevents cleanup through a no-longer-trusted root."""


@dataclass(frozen=True)
class ValidatedCanaryIdentity:
    """Content-derived identity returned only after complete core validation."""

    canonical_config_bytes: bytes
    config_sha256: str
    run_id: str


@dataclass(frozen=True)
class InputVerificationReceipt:
    """Positive receipt binding a verifier result to exactly three input identities."""

    input_identity_sha256: str


@dataclass(frozen=True)
class _OutputRootAuthority:
    """Pinned inode, owner, and exact permission mode for the output root."""

    identity: tuple
    owner_uid: int
    mode: int


@dataclass
class ReservedOutputDirectory:
    """Owned output descriptors; ``path`` is display-only, never write authority."""

    path: Path
    _directory_fd: int = field(repr=False)
    _root_fd: int = field(repr=False)
    st_dev: int
    st_ino: int
    _closed: bool = field(default=False, init=False, repr=False)

    @property
    def directory_fd(self) -> int:
        return self.fileno()

    def fileno(self) -> int:
        if self._closed:
            raise OSError("reserved output directory is closed")
        os.fstat(self._directory_fd)
        return self._directory_fd

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            os.close(self._directory_fd)
        finally:
            os.close(self._root_fd)

    def __enter__(self) -> "ReservedOutputDirectory":
        self.fileno()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()


def _canonical_json_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise CanaryContractError("canary config must contain finite JSON values") from exc
    return (encoded + "\n").encode("utf-8")


def decoded_frame_hash_header_bytes() -> bytes:
    """Return the exact canonical header prepended to every decoded BGR8 frame."""
    return _canonical_json_bytes(
        {
            "dtype": DECODED_FRAME_DTYPE,
            "shape": list(DECODED_FRAME_SHAPE),
            "version": DECODED_FRAME_HASH_VERSION,
        }
    )


def bgr8_frame_sha256(contiguous_bgr8_bytes: bytes) -> str:
    """Hash one 1080x1920 C-contiguous BGR uint8 frame exactly."""
    expected_size = math.prod(DECODED_FRAME_SHAPE)
    if type(contiguous_bgr8_bytes) is not bytes or len(contiguous_bgr8_bytes) != expected_size:
        raise CanaryContractError(
            "decoded frame must be exact contiguous 1080x1920x3 BGR uint8 bytes"
        )
    return hashlib.sha256(
        decoded_frame_hash_header_bytes() + contiguous_bgr8_bytes
    ).hexdigest()


def expected_input_identity_sha256(
    inputs: Mapping[str, Mapping[str, str]],
) -> str:
    """Return the deterministic digest a successful input verifier must receipt."""
    return hashlib.sha256(_canonical_json_bytes(inputs)).hexdigest()


def _reject_json_constant(value: str) -> object:
    raise ValueError("non-finite JSON constant: " + value)


def _parse_config(config_or_canonical_bytes: Union[dict, bytes]) -> tuple:
    if type(config_or_canonical_bytes) is bytes:
        raw = config_or_canonical_bytes
        try:
            parsed = json.loads(
                raw.decode("utf-8"),
                parse_constant=_reject_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise CanaryContractError(
                "canary config bytes must be canonical UTF-8 JSON"
            ) from exc
        if type(parsed) is not dict:
            raise CanaryContractError("canary config must be a JSON object")
        canonical = _canonical_json_bytes(parsed)
        if raw != canonical:
            raise CanaryContractError("canary config bytes are not canonical JSON")
        return parsed, canonical
    if type(config_or_canonical_bytes) is not dict:
        raise CanaryContractError("canary config must be a dict or canonical bytes")
    config = deepcopy(config_or_canonical_bytes)
    return config, _canonical_json_bytes(config)


def _exact_keys(value: object, expected: frozenset, field: str) -> dict:
    if type(value) is not dict:
        raise CanaryContractError(field + " must be an object")
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(str(item) for item in expected - actual)
        extra = sorted(str(item) for item in actual - expected)
        raise CanaryContractError(
            "%s fields are not exact (missing=%s, extra=%s)"
            % (field, missing, extra)
        )
    return value


def _contains_forbidden_term(value: object) -> bool:
    if isinstance(value, str):
        lowered = value.lower()
        return any(term in lowered for term in _FORBIDDEN_TERMS)
    if type(value) is dict:
        return any(
            _contains_forbidden_term(key) or _contains_forbidden_term(child)
            for key, child in value.items()
        )
    if type(value) in (list, tuple):
        return any(_contains_forbidden_term(child) for child in value)
    return False


def _require_sha256(value: object, field: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise CanaryContractError(field + " must be a lowercase full SHA-256")
    return value


def _strict_equal(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return set(actual) == set(expected) and all(
            _strict_equal(actual[key], expected[key]) for key in expected
        )
    if type(expected) is list:
        return len(actual) == len(expected) and all(
            _strict_equal(left, right) for left, right in zip(actual, expected)
        )
    return bool(actual == expected)


def _canonical_absolute_path(value: Union[str, Path], field: str) -> Path:
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise CanaryContractError(field + " must be an absolute canonical path") from exc
    if type(raw) is not str or not raw or not raw.startswith("/"):
        raise CanaryContractError(field + " must be an absolute canonical path")
    if raw.startswith("//") or os.path.normpath(raw) != raw:
        raise CanaryContractError(field + " must be an absolute canonical path")
    return Path(raw)


def _require_descriptor_primitives(*functions: object) -> None:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise CanaryContractError("descriptor-safe no-follow operations are unavailable")
    if any(function not in os.supports_dir_fd for function in functions):
        raise CanaryContractError("required dir_fd operation is unavailable")


def _directory_flags() -> int:
    _require_descriptor_primitives(os.open)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    return flags


def _directory_walk_interleave(opened_path: Path, directory_fd: int) -> None:
    """Private deterministic race-test hook; production intentionally does nothing."""
    del opened_path, directory_fd


def _open_directory_no_follow(path: Path, field: str) -> int:
    candidate = _canonical_absolute_path(path, field)
    flags = _directory_flags()
    current_fd = None
    try:
        current_fd = os.open("/", flags)
        os.set_inheritable(current_fd, False)
        opened_path = Path("/")
        _directory_walk_interleave(opened_path, current_fd)
        for part in candidate.parts[1:]:
            child_fd = None
            try:
                child_fd = os.open(part, flags, dir_fd=current_fd)
                os.set_inheritable(child_fd, False)
                if not stat.S_ISDIR(os.fstat(child_fd).st_mode):
                    raise CanaryContractError(field + " contains a non-directory")
                opened_path = opened_path / part
                _directory_walk_interleave(opened_path, child_fd)
            except BaseException:
                if child_fd is not None:
                    os.close(child_fd)
                raise
            os.close(current_fd)
            current_fd = child_fd
        result = current_fd
        current_fd = None
        return result
    except CanaryContractError:
        raise
    except OSError as exc:
        raise CanaryContractError(
            field + " could not be opened by descriptor without following"
        ) from exc
    finally:
        if current_fd is not None:
            os.close(current_fd)


def _stat_identity(info: os.stat_result, *, stable: bool) -> tuple:
    identity = (info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode))
    if stable:
        identity += (
            info.st_size,
            getattr(info, "st_mtime_ns", None),
            getattr(info, "st_ctime_ns", None),
        )
    return identity


def _require_existing_no_follow(path: Path, field: str, *, directory: bool) -> None:
    if not directory:
        raise CanaryContractError(field + " must be checked as a pinned regular file")
    descriptor = _open_directory_no_follow(path, field)
    os.close(descriptor)


def _read_regular_stable(path: Path, field: str) -> bytes:
    candidate = _canonical_absolute_path(path, field)
    _require_descriptor_primitives(os.open, os.stat)
    parent_fd = _open_directory_no_follow(candidate.parent, field + " parent")
    descriptor = None
    try:
        flags = os.O_RDONLY | os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        named_before = os.stat(
            candidate.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        descriptor = os.open(candidate.name, flags, dir_fd=parent_fd)
        os.set_inheritable(descriptor, False)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CanaryContractError(field + " must be a regular file")
        if _stat_identity(named_before, stable=False) != _stat_identity(
            before, stable=False
        ):
            raise CanaryContractError(field + " changed while it was opened")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        named_after = os.stat(
            candidate.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if (
            _stat_identity(before, stable=True)
            != _stat_identity(after, stable=True)
            or _stat_identity(before, stable=True)
            != _stat_identity(named_after, stable=True)
        ):
            raise CanaryContractError(field + " changed while it was read")
        return b"".join(chunks)
    except CanaryContractError:
        raise
    except OSError as exc:
        raise CanaryContractError(
            field + " could not be opened by descriptor without following"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


def _selected_root(
    supplied: Optional[Union[str, Path]],
    default: Path,
    *,
    testing: bool,
    field: str,
) -> Path:
    if supplied is None:
        return default
    if testing is not True:
        raise CanaryContractError(field + " is a test-only injection")
    return _canonical_absolute_path(supplied, field)


def _validate_release_directory(
    identity: dict,
    *,
    root: Path,
    release_id: str,
    field: str,
) -> None:
    if type(identity.get("realpath")) is not str:
        raise CanaryContractError(field + ".realpath must be a string")
    expected = root / release_id
    expected_text = str(expected)
    if identity["realpath"] != expected_text:
        raise CanaryContractError(field + ".realpath is not the exact immutable release path")
    candidate = _canonical_absolute_path(identity["realpath"], field + ".realpath")
    if "current" in candidate.parts:
        raise CanaryContractError(field + ".realpath must not contain current")
    _require_existing_no_follow(candidate, field + ".realpath", directory=True)


def _validate_code(config: dict, *, root: Path) -> None:
    code = _exact_keys(
        config["code"],
        frozenset(("release_id", "payload_sha256", "realpath")),
        "code",
    )
    release_id = code["release_id"]
    match = _CODE_RELEASE_ID.fullmatch(release_id) if type(release_id) is str else None
    if match is None:
        raise CanaryContractError("code.release_id must be code_release- plus 20 lowercase hex")
    payload = _require_sha256(code["payload_sha256"], "code.payload_sha256")
    if match.group(1) != payload[:20]:
        raise CanaryContractError("code release ID does not match its payload SHA-256")
    _validate_release_directory(
        code,
        root=root,
        release_id=release_id,
        field="code",
    )


def _validate_environment(config: dict, *, root: Path) -> None:
    environment = _exact_keys(
        config["environment"],
        frozenset(("release_id", "identity_sha256", "realpath")),
        "environment",
    )
    if environment["release_id"] != _ENVIRONMENT_RELEASE_ID:
        raise CanaryContractError("environment.release_id is not the locked H200 release")
    if environment["identity_sha256"] != _ENVIRONMENT_IDENTITY_SHA256:
        raise CanaryContractError("environment.identity_sha256 is not the locked identity")
    _validate_release_directory(
        environment,
        root=root,
        release_id=_ENVIRONMENT_RELEASE_ID,
        field="environment",
    )


def _validate_warm_start(config: dict) -> None:
    warm_start = config["warm_start"]
    _exact_keys(warm_start, frozenset(_WARM_START), "warm_start")
    if not _strict_equal(warm_start, _WARM_START):
        raise CanaryContractError("warm_start does not exactly identify perc_lmh_v9_occ")


def _validate_inputs(config: dict) -> dict:
    inputs = _exact_keys(config["inputs"], frozenset(_INPUT_PREFIXES), "inputs")
    for name, prefix in _INPUT_PREFIXES.items():
        identity = _exact_keys(inputs[name], _INPUT_FIELDS, "inputs." + name)
        payload = _require_sha256(
            identity["payload_sha256"], "inputs.%s.payload_sha256" % name
        )
        _require_sha256(
            identity["release_manifest_sha256"],
            "inputs.%s.release_manifest_sha256" % name,
        )
        _require_sha256(
            identity["semantic_manifest_sha256"],
            "inputs.%s.semantic_manifest_sha256" % name,
        )
        expected_id = prefix + "-" + payload[:20]
        if identity["release_id"] != expected_id:
            raise CanaryContractError(
                "inputs.%s.release_id is not bound to its payload and exact kind" % name
            )
    return inputs


def _validate_seeds_and_training(config: dict) -> None:
    seeds = config["seeds"]
    if type(seeds) is not list or any(type(seed) is not int for seed in seeds):
        raise CanaryContractError("seeds must be unique integers")
    if len(seeds) != len(set(seeds)):
        raise CanaryContractError("seeds must be unique integers")
    if seeds != [17, 23, 42]:
        raise CanaryContractError("seeds must exactly equal [17, 23, 42]")

    training = _exact_keys(
        config["training"],
        frozenset(("epochs", "batch_size", "learning_rate")),
        "training",
    )
    if type(training["epochs"]) is not int or training["epochs"] != 5:
        raise CanaryContractError("training.epochs must be integer 5")
    if type(training["batch_size"]) is not int or training["batch_size"] != 8:
        raise CanaryContractError("training.batch_size must be integer 8")
    learning_rate = training["learning_rate"]
    if (
        type(learning_rate) is not float
        or not math.isfinite(learning_rate)
        or learning_rate != 1e-5
    ):
        raise CanaryContractError("training.learning_rate must be finite 1e-5")


def _validate_policy(
    config: dict,
    *,
    policy_root: Path,
    repo_root: Path,
) -> None:
    policy_ref = _exact_keys(
        config["mask_policy"], frozenset(("path", "sha256")), "mask_policy"
    )
    if policy_ref["path"] != POLICY_RELATIVE_PATH:
        raise CanaryContractError("mask_policy.path is not the canonical relative path")
    configured_sha = _require_sha256(policy_ref["sha256"], "mask_policy.sha256")
    expected_policy_bytes = _canonical_json_bytes(_EXPECTED_POLICY)
    policy_path = policy_root / POLICY_RELATIVE_PATH
    policy_bytes = _read_regular_stable(policy_path, "mask policy")
    if policy_bytes != expected_policy_bytes:
        raise CanaryContractError("mask policy bytes differ from the exact canonical policy")
    actual_policy_sha = hashlib.sha256(policy_bytes).hexdigest()
    if configured_sha != actual_policy_sha:
        raise CanaryContractError("mask_policy.sha256 does not match canonical policy bytes")

    urdf_path = repo_root / URDF_RELATIVE_PATH
    urdf_bytes = _read_regular_stable(urdf_path, "official OpenArm URDF")
    if hashlib.sha256(urdf_bytes).hexdigest() != _URDF_SHA256:
        raise CanaryContractError("official OpenArm URDF SHA-256 does not match policy")


def _verify_input_releases(
    inputs: Mapping[str, Mapping[str, str]],
    verifier: Callable[
        [Mapping[str, Mapping[str, str]]], InputVerificationReceipt
    ],
) -> None:
    """Explicit Task-2B boundary; no release path is inferred or opened here."""
    if not callable(verifier):
        raise CanaryContractError("verify_inputs must be an explicit callable")
    try:
        receipt = verifier(deepcopy(inputs))
    except _InputVerifierNotInstalled:
        raise
    except Exception as exc:
        raise CanaryContractError("input release verification failed") from exc
    if inspect.isawaitable(receipt) or inspect.isgenerator(receipt):
        close = getattr(receipt, "close", None)
        if callable(close):
            close()
        raise CanaryContractError("input verifier must return a synchronous receipt")
    if type(receipt) is not InputVerificationReceipt:
        raise CanaryContractError("input verifier did not return a verification receipt")
    digest = _require_sha256(
        receipt.input_identity_sha256,
        "input verification receipt input_identity_sha256",
    )
    if digest != expected_input_identity_sha256(inputs):
        raise CanaryContractError("input verification receipt does not match inputs")


def _parse_canonical_json_object(raw: bytes, field: str) -> dict:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CanaryContractError(field + " must be finite canonical UTF-8 JSON") from exc
    if type(value) is not dict or raw != _canonical_json_bytes(value):
        raise CanaryContractError(field + " must be a canonical JSON object")
    return value


def _relative_payload_path(value: object, field: str) -> str:
    if type(value) is not str or not value or "\\" in value or "\x00" in value:
        raise CanaryContractError(field + " must be a canonical relative POSIX path")
    candidate = PurePosixPath(value)
    if (
        candidate.is_absolute()
        or any(part in ("", ".", "..") for part in candidate.parts)
        or candidate.as_posix() != value
    ):
        raise CanaryContractError(field + " must be a canonical relative POSIX path")
    return value


def _read_regular_stable_at(
    root_fd: int,
    relative: str,
    field: str,
) -> bytes:
    """Read beneath a pinned release FD without resolving a pathname ancestor."""
    path = _relative_payload_path(relative, field)
    _require_descriptor_primitives(os.open, os.stat)
    directories = [os.dup(root_fd)]
    descriptor = None
    try:
        for component in PurePosixPath(path).parts[:-1]:
            info = os.stat(
                component,
                dir_fd=directories[-1],
                follow_symlinks=False,
            )
            if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                raise CanaryContractError(field + " contains a symlink or non-directory")
            child = os.open(component, _directory_flags(), dir_fd=directories[-1])
            os.set_inheritable(child, False)
            directories.append(child)
        name = PurePosixPath(path).parts[-1]
        named_before = os.stat(
            name,
            dir_fd=directories[-1],
            follow_symlinks=False,
        )
        flags = os.O_RDONLY | os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        descriptor = os.open(name, flags, dir_fd=directories[-1])
        os.set_inheritable(descriptor, False)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CanaryContractError(field + " must be a regular file")
        if _stat_identity(named_before, stable=False) != _stat_identity(
            before, stable=False
        ):
            raise CanaryContractError(field + " changed while it was opened")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        named_after = os.stat(
            name,
            dir_fd=directories[-1],
            follow_symlinks=False,
        )
        if (
            _stat_identity(before, stable=True) != _stat_identity(after, stable=True)
            or _stat_identity(before, stable=True)
            != _stat_identity(named_after, stable=True)
        ):
            raise CanaryContractError(field + " changed while it was read")
        return b"".join(chunks)
    except CanaryContractError:
        raise
    except OSError as exc:
        raise CanaryContractError(
            field + " could not be opened without following symlinks"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        for directory in reversed(directories):
            os.close(directory)


@contextmanager
def _open_regular_pinned_at(root_fd: int, relative: str, field: str):
    """Yield a no-follow file FD while retaining and rechecking its named binding."""
    path = _relative_payload_path(relative, field)
    _require_descriptor_primitives(os.open, os.stat)
    directories = [os.dup(root_fd)]
    descriptor = None
    before = None
    named_before = None
    name = PurePosixPath(path).parts[-1]
    try:
        for component in PurePosixPath(path).parts[:-1]:
            info = os.stat(
                component,
                dir_fd=directories[-1],
                follow_symlinks=False,
            )
            if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                raise CanaryContractError(field + " contains a symlink or non-directory")
            child = os.open(component, _directory_flags(), dir_fd=directories[-1])
            os.set_inheritable(child, False)
            directories.append(child)
        named_before = os.stat(
            name,
            dir_fd=directories[-1],
            follow_symlinks=False,
        )
        flags = os.O_RDONLY | os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        descriptor = os.open(name, flags, dir_fd=directories[-1])
        os.set_inheritable(descriptor, False)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CanaryContractError(field + " must be a regular file")
        if _stat_identity(named_before, stable=False) != _stat_identity(
            before, stable=False
        ):
            raise CanaryContractError(field + " changed while it was opened")
        yield descriptor
        after = os.fstat(descriptor)
        named_after = os.stat(
            name,
            dir_fd=directories[-1],
            follow_symlinks=False,
        )
        if (
            _stat_identity(before, stable=True) != _stat_identity(after, stable=True)
            or _stat_identity(before, stable=True)
            != _stat_identity(named_after, stable=True)
        ):
            raise CanaryContractError(field + " changed while it was consumed")
    except CanaryContractError:
        raise
    except OSError as exc:
        raise CanaryContractError(
            field + " could not be pinned without following symlinks"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        for directory in reversed(directories):
            os.close(directory)


def _open_release_at_root(root_fd: int, release_id: str, field: str) -> int:
    name = _single_component(release_id, field + ".release_id")
    try:
        named = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        if not stat.S_ISDIR(named.st_mode) or stat.S_ISLNK(named.st_mode):
            raise CanaryContractError(field + " release is a symlink or non-directory")
        descriptor = os.open(name, _directory_flags(), dir_fd=root_fd)
        os.set_inheritable(descriptor, False)
        opened = os.fstat(descriptor)
        if _stat_identity(named, stable=False) != _stat_identity(opened, stable=False):
            os.close(descriptor)
            raise CanaryContractError(field + " release changed while it was opened")
        return descriptor
    except CanaryContractError:
        raise
    except OSError as exc:
        raise CanaryContractError(field + " release is missing or unsafe") from exc


def _validate_release_envelope(
    envelope: object,
    manifest_bytes: bytes,
    completion_bytes: bytes,
    *,
    input_name: str,
    identity: Mapping[str, str],
) -> tuple:
    field = "inputs." + input_name
    envelope = _exact_keys(
        envelope,
        frozenset(("payload", "payload_sha256", "release_id", "schema")),
        field + " release envelope",
    )
    if envelope["schema"] != _artifact_contract.ENVELOPE_SCHEMA:
        raise CanaryContractError(field + " release envelope schema is not exact")
    payload = _exact_keys(
        envelope["payload"],
        frozenset(("artifact_kind", "files", "schema", "source_role")),
        field + " release payload",
    )
    expected = _PERCEPTION_INPUT_EXPECTATIONS[input_name]
    if payload["schema"] != _artifact_contract.PAYLOAD_SCHEMA:
        raise CanaryContractError(field + " release payload schema is not exact")
    if payload["artifact_kind"] != expected["artifact_kind"]:
        raise CanaryContractError(field + " artifact_kind is not exact")
    if payload["source_role"] != expected["source_role"]:
        raise CanaryContractError(field + " source_role is not exact")
    recomputed_payload_sha256 = hashlib.sha256(
        _canonical_json_bytes(payload)
    ).hexdigest()
    if envelope["payload_sha256"] != recomputed_payload_sha256:
        raise CanaryContractError(field + " release payload hash is not self-consistent")
    if envelope["release_id"] != (
        payload["artifact_kind"] + "-" + recomputed_payload_sha256[:20]
    ):
        raise CanaryContractError(field + " release ID is not payload-bound")
    if envelope["release_id"] != identity["release_id"]:
        raise CanaryContractError(field + " release_id does not match configured identity")
    if envelope["payload_sha256"] != identity["payload_sha256"]:
        raise CanaryContractError(field + " payload SHA-256 does not match configured identity")
    if hashlib.sha256(manifest_bytes).hexdigest() != identity["release_manifest_sha256"]:
        raise CanaryContractError(
            field + " release manifest SHA-256 does not match configured identity"
        )
    if manifest_bytes != _canonical_json_bytes(envelope):
        raise CanaryContractError(field + " release manifest is not strict canonical JSON")

    rows = payload["files"]
    if type(rows) is not list:
        raise CanaryContractError(field + " release file rows must be a list")
    normalized_rows = []
    seen = set()
    for index, row in enumerate(rows):
        row = _exact_keys(
            row,
            frozenset(("mode", "path", "sha256", "size")),
            "%s release file row %d" % (field, index),
        )
        path = _relative_payload_path(row["path"], field + " file path")
        if path in seen:
            raise CanaryContractError(field + " release has a duplicate file path")
        seen.add(path)
        _require_sha256(row["sha256"], field + " file SHA-256")
        if type(row["size"]) is not int or row["size"] < 0:
            raise CanaryContractError(field + " file size must be a nonnegative integer")
        if type(row["mode"]) is not str or re.fullmatch(r"0[0-7]{3}", row["mode"]) is None:
            raise CanaryContractError(field + " file mode is invalid")
        normalized_rows.append(row)
    if [row["path"] for row in normalized_rows] != sorted(seen):
        raise CanaryContractError(field + " release file rows are not path-sorted")
    if sum(row["path"] == PERCEPTION_INPUT_NAME for row in rows) != 1:
        raise CanaryContractError(field + " must contain exactly one semantic manifest")

    marker = _parse_canonical_json_object(completion_bytes, field + " completion marker")
    expected_marker = {
        "artifact_kind": payload["artifact_kind"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "payload_sha256": envelope["payload_sha256"],
        "release_id": envelope["release_id"],
        "schema": _artifact_contract.COMPLETE_SCHEMA,
        "source_role": payload["source_role"],
        "target_role": "h200",
        "verified_files": len(rows),
    }
    if not _strict_equal(marker, expected_marker):
        raise CanaryContractError(field + " completion marker is not exact for H200")
    return payload, normalized_rows


def _validate_semantic_member_rows(
    semantic: dict,
    outer_rows: list,
    *,
    input_name: str,
) -> tuple:
    members = semantic["members"]
    field = "inputs.%s semantic members" % input_name
    if type(members) is not list:
        raise CanaryContractError(field + " must be a list")
    by_path = {}
    hashes = {}
    for index, member in enumerate(members):
        member = _exact_keys(
            member,
            PERCEPTION_INPUT_MEMBER_FIELDS,
            "%s row %d" % (field, index),
        )
        path = _relative_payload_path(member["path"], field + " path")
        if path == PERCEPTION_INPUT_NAME or path in by_path:
            raise CanaryContractError(field + " contains a duplicate or reserved path")
        digest = _require_sha256(member["sha256"], field + " SHA-256")
        if type(member["size"]) is not int or member["size"] < 0:
            raise CanaryContractError(field + " size must be a nonnegative integer")
        if type(member["role"]) is not str or not member["role"]:
            raise CanaryContractError(field + " role must be a nonempty string")
        if type(member["content_origin"]) is not str or not member["content_origin"]:
            raise CanaryContractError(field + " content_origin must be a nonempty string")
        by_path[path] = member
        hashes.setdefault(digest, []).append(member)
    if [member["path"] for member in members] != sorted(by_path):
        raise CanaryContractError(field + " are not path-sorted")
    outer = {
        row["path"]: {key: row[key] for key in ("path", "sha256", "size")}
        for row in outer_rows
        if row["path"] != PERCEPTION_INPUT_NAME
    }
    projected = {
        path: {key: member[key] for key in ("path", "sha256", "size")}
        for path, member in by_path.items()
    }
    if projected != outer:
        raise CanaryContractError(field + " do not exactly match outer payload files")
    for duplicated in hashes.values():
        if len(duplicated) > 1 and any(
            row["content_origin"] not in ALLOWED_SHARED_CONTENT_ORIGINS
            for row in duplicated
        ):
            raise CanaryContractError(field + " contain an ambiguous duplicate content hash")
    return members, hashes


def _require_hash_list(
    value: object,
    field: str,
    *,
    length: int,
    sorted_unique: bool,
) -> list:
    if type(value) is not list or len(value) != length:
        raise CanaryContractError("%s must contain exactly %d hashes" % (field, length))
    for digest in value:
        _require_sha256(digest, field)
    if sorted_unique and value != sorted(set(value)):
        raise CanaryContractError(field + " must be sorted and unique")
    return value


def _validate_content_partition(
    partition: object,
    *,
    expected_partition_sha256: str,
    expected_partition_list_sha256: Mapping[str, str],
) -> dict:
    partition = _exact_keys(
        partition,
        MIXED_EVAL_CONTENT_PARTITION_FIELDS,
        "train source content_partition",
    )
    if partition["schema"] != MIXED_EVAL_CONTENT_PARTITION_SCHEMA:
        raise CanaryContractError("train source content_partition schema is not exact")
    if partition["decoded_frame_hash_version"] != DECODED_FRAME_HASH_VERSION:
        raise CanaryContractError(
            "train source decoded frame hash version is not exact"
        )
    expected_source = {
        "source_manifest_sha256": HISTORICAL_MIXED_EVAL_MANIFEST_SHA256,
        "source_payload_sha256": HISTORICAL_MIXED_EVAL_PAYLOAD_SHA256,
        "source_release_id": HISTORICAL_MIXED_EVAL_RELEASE_ID,
    }
    if any(partition[key] != value for key, value in expected_source.items()):
        raise CanaryContractError(
            "train source content_partition does not bind the historical mixed eval"
        )
    session2_selected = _require_hash_list(
        partition["session2_selected_sha256"],
        "content_partition.session2_selected_sha256",
        length=605,
        sorted_unique=True,
    )
    session3_denied = _require_hash_list(
        partition["session3_denied_sha256"],
        "content_partition.session3_denied_sha256",
        length=608,
        sorted_unique=True,
    )
    _require_hash_list(
        partition["session2_decoded_frame_sha256"],
        "content_partition.session2_decoded_frame_sha256",
        length=602,
        sorted_unique=False,
    )
    _require_hash_list(
        partition["session3_decoded_frame_sha256"],
        "content_partition.session3_decoded_frame_sha256",
        length=602,
        sorted_unique=False,
    )
    shared_values = _require_hash_list(
        partition["shared_intrinsics_sha256"],
        "content_partition.shared_intrinsics_sha256",
        length=1,
        sorted_unique=True,
    )
    shared = shared_values[0]
    expected_list_digests = _exact_keys(
        expected_partition_list_sha256,
        frozenset(_PRODUCTION_PARTITION_LIST_SHA256),
        "expected content partition list digests",
    )
    for list_field, expected_digest in expected_list_digests.items():
        _require_sha256(expected_digest, "content partition list digest")
        actual_digest = hashlib.sha256(
            _canonical_json_bytes(partition[list_field])
        ).hexdigest()
        if actual_digest != expected_digest:
            raise CanaryContractError(
                "content partition %s digest does not match" % list_field
            )
    if set(session2_selected) & set(session3_denied) != {shared}:
        raise CanaryContractError(
            "content partition overlap must be exactly shared intrinsics"
        )
    actual_partition_sha256 = hashlib.sha256(
        _canonical_json_bytes(partition)
    ).hexdigest()
    _require_sha256(
        expected_partition_sha256,
        "expected historical content partition SHA-256",
    )
    if actual_partition_sha256 != expected_partition_sha256:
        raise CanaryContractError(
            "historical mixed-eval content partition SHA-256 does not match"
        )
    return partition


def _validate_source_attestations(
    attestations: object,
    *,
    current_identity: Mapping[str, str],
    members: list,
    input_name: str,
    expected_partition_sha256: str,
    expected_partition_list_sha256: Mapping[str, str],
) -> Optional[dict]:
    field = "inputs.%s source_attestations" % input_name
    if type(attestations) is not list:
        raise CanaryContractError(field + " must be a list")
    if input_name != "train" and attestations:
        raise CanaryContractError(field + " must be empty")
    if input_name != "train":
        return None
    if input_name == "train" and len(attestations) != 1:
        raise CanaryContractError(field + " must contain exactly one audit attestation")
    for index, attestation in enumerate(attestations):
        attestation = _exact_keys(
            attestation,
            PERCEPTION_SOURCE_ATTESTATION_FIELDS,
            "%s row %d" % (field, index),
        )
        expected_identity = {
            "artifact_kind": "perception_eval",
            "payload_sha256": HISTORICAL_MIXED_EVAL_PAYLOAD_SHA256,
            "release_id": HISTORICAL_MIXED_EVAL_RELEASE_ID,
            "release_manifest_sha256": HISTORICAL_MIXED_EVAL_MANIFEST_SHA256,
        }
        if any(
            attestation[key] != value for key, value in expected_identity.items()
        ):
            raise CanaryContractError(
                field + " does not identify the exact historical mixed eval release"
            )
        partition = _validate_content_partition(
            attestation["content_partition"],
            expected_partition_sha256=expected_partition_sha256,
            expected_partition_list_sha256=expected_partition_list_sha256,
        )
        expected_selected = sorted(
            {
                member["sha256"]
                for member in members
                if member["role"]
                in {
                    "intrinsics",
                    "metric_depth",
                    "rgb_video",
                    "source_frame_index",
                }
            }
        )
        if partition["session2_selected_sha256"] != expected_selected:
            raise CanaryContractError(
                field + " selected content does not exactly bind copied Session-2 bytes"
            )
        denied = (
            set(partition["session3_denied_sha256"])
            | set(partition["session3_decoded_frame_sha256"])
            | set(DENIED_TRAIN_CONTENT_SHA256)
        )
        shared = partition["shared_intrinsics_sha256"][0]
        for member in members:
            if member["sha256"] not in denied:
                continue
            if not (
                member["sha256"] == shared
                and member["role"] == "intrinsics"
                and member["content_origin"] == "shared_intrinsics"
            ):
                raise CanaryContractError(
                    "Session-2 member collides with denied Session-3 content"
                )
        if any(
            attestation[key] == current_identity[current_key]
            for key, current_key in (
                ("release_id", "release_id"),
                ("payload_sha256", "payload_sha256"),
                ("release_manifest_sha256", "release_manifest_sha256"),
            )
        ):
            raise CanaryContractError(field + " circularly binds the containing release")
        return partition
    raise CanaryContractError(field + " is missing its audit attestation")


def _validate_kind_semantics(
    semantic: dict,
    members: list,
    *,
    input_name: str,
    content_partition: Optional[dict],
    expected_train_member_sha256: Mapping[str, str],
    expected_ordered_depth_list_sha256: str,
    expected_geometry_support_sha256: Mapping[str, str],
    expected_geometry_smpl_manifest_sha256: str,
) -> None:
    roles = [member["role"] for member in members]
    origins = {member["content_origin"] for member in members}
    contract = semantic["contract"]
    if type(contract) is not dict:
        raise CanaryContractError("inputs.%s contract must be an object" % input_name)
    if input_name == "geometry_source":
        expected_contract = {
            "densepose_uv_count": 1,
            "license_count": 3,
            "provenance_count": 1,
            "raw_rgb_count": 0,
            "smpl_neutral_count": 1,
            "smpl_params_count": 907,
        }
        if not _strict_equal(contract, expected_contract):
            raise CanaryContractError("geometry source contract is not exact")
        expected_role_counts = {
            "densepose_uv": 1,
            "license": 3,
            "provenance": 1,
            "smpl_params": 907,
            "smpl_neutral": 1,
        }
        if (
            len(members) != 913
            or set(roles) != set(expected_role_counts)
            or any(
                roles.count(role) != count
                for role, count in expected_role_counts.items()
            )
        ):
            raise CanaryContractError("geometry source member closure is not exact")
        expected_origins = {
            "densepose_uv": "densepose_uv",
            "license": "license",
            "provenance": "provenance",
            "smpl_neutral": "smpl_neutral",
            "smpl_params": "humman_smpl_params",
        }
        if any(
            member["content_origin"] != expected_origins[member["role"]]
            for member in members
        ):
            raise CanaryContractError("geometry source role/origin closure is not exact")
        by_role = {}
        for member in members:
            by_role.setdefault(member["role"], []).append(member)
        if (
            by_role["smpl_neutral"][0]["path"] != "smpl/SMPL_NEUTRAL.pkl"
            or by_role["densepose_uv"][0]["path"]
            != "densepose/UV_Processed.mat"
            or by_role["provenance"][0]["path"]
            != "provenance/SOURCE_TERMS.json"
            or {member["path"] for member in by_role["license"]}
            != {
                "licenses/HUMMAN_LICENSE.txt",
                "licenses/SMPL_MODEL_LICENSE.html",
                "licenses/DENSEPOSE_SOURCE_LICENSE.txt",
            }
        ):
            raise CanaryContractError("geometry source support paths are not exact")
        support_pins = _exact_keys(
            expected_geometry_support_sha256,
            frozenset(_PRODUCTION_GEOMETRY_SUPPORT_SHA256),
            "expected geometry support hashes",
        )
        member_by_path = {member["path"]: member for member in members}
        for path, expected_sha256 in support_pins.items():
            _require_sha256(expected_sha256, "expected geometry support hash")
            if member_by_path[path]["sha256"] != expected_sha256:
                raise CanaryContractError(
                    "geometry source support SHA-256 is not exact"
                )
        sequence_ids = []
        smpl_manifest_rows = []
        for member in by_role["smpl_params"]:
            parts = PurePosixPath(member["path"]).parts
            if (
                len(parts) != 4
                or parts[0:2] != ("humman", "smpl_params")
                or parts[3] != "smpl_params.npz"
                or _HUMMAN_SEQUENCE_ID.fullmatch(parts[2]) is None
            ):
                raise CanaryContractError(
                    "geometry smpl_params path is not a canonical HuMMan sequence path"
                )
            sequence_ids.append(parts[2])
            smpl_manifest_rows.append(
                (
                    "%s/smpl_params.npz" % parts[2],
                    member["sha256"],
                )
            )
        if len(sequence_ids) != len(set(sequence_ids)):
            raise CanaryContractError("geometry contains a duplicate HuMMan sequence ID")
        _require_sha256(
            expected_geometry_smpl_manifest_sha256,
            "expected HuMMan SMPL parameter manifest SHA-256",
        )
        smpl_manifest_bytes = "".join(
            "%s  %s\n" % (digest, sequence_path)
            for sequence_path, digest in sorted(smpl_manifest_rows)
        ).encode("ascii")
        if hashlib.sha256(smpl_manifest_bytes).hexdigest() != (
            expected_geometry_smpl_manifest_sha256
        ):
            raise CanaryContractError(
                "geometry HuMMan SMPL parameter manifest SHA-256 is not exact"
            )
    elif input_name == "stage0":
        expected_contract = {
            "provenance_path": "PROVENANCE.json",
            "sample_count": 3628,
            "sample_format": STAGE0_SAMPLE_FORMAT,
            "sample_ledger_path": "ledgers/sample_ids.json",
            "split_strategy": "subject_grouped_fixed_v1",
            "subject_split_ledger_path": "ledgers/subject_split.json",
        }
        if not _strict_equal(contract, expected_contract):
            raise CanaryContractError("Stage-0 contract is not exact")
        expected_role_counts = {
            "generated_sample": 3628,
            "provenance": 1,
            "sample_ledger": 1,
            "subject_split_ledger": 1,
        }
        if (
            len(members) != 3631
            or set(roles) != set(expected_role_counts)
            or any(
                roles.count(role) != count
                for role, count in expected_role_counts.items()
            )
        ):
            raise CanaryContractError(
                "Stage-0 must contain exactly 3,628 generated members and exact ledgers/provenance"
            )
        by_role = {}
        for member in members:
            by_role.setdefault(member["role"], []).append(member)
        expected_support = {
            "provenance": (contract["provenance_path"], "provenance"),
            "sample_ledger": (contract["sample_ledger_path"], "ledger"),
            "subject_split_ledger": (
                contract["subject_split_ledger_path"],
                "ledger",
            ),
        }
        if any(
            by_role[role][0]["path"] != path
            or by_role[role][0]["content_origin"] != origin
            for role, (path, origin) in expected_support.items()
        ):
            raise CanaryContractError("Stage-0 support member closure is not exact")
        generated = [member for member in members if member["role"] == "generated_sample"]
        sample_ids = []
        for member in generated:
            parts = PurePosixPath(member["path"]).parts
            if (
                member["content_origin"] != "generated"
                or len(parts) != 2
                or parts[0] != "samples"
                or not parts[1].endswith(".npz")
            ):
                raise CanaryContractError(
                    "Stage-0 generated sample path/origin is not canonical"
                )
            sample_id = parts[1][:-4]
            if _STAGE0_SAMPLE_ID.fullmatch(sample_id) is None:
                raise CanaryContractError("Stage-0 sample ID is not canonical")
            sample_ids.append(sample_id)
        if len(sample_ids) != len(set(sample_ids)):
            raise CanaryContractError("Stage-0 contains a duplicate generated sample ID")
    else:
        expected_keys = frozenset(
            (
                "decoded_frame_fingerprints",
                "depth_frame_count",
                "frame_count",
                "frame_ledger_path",
                "garment",
                "intrinsics_path",
                "labels_path",
                "ordered_depth_fingerprints",
                "rgb_video_path",
                "source_frames_path",
            )
        )
        _exact_keys(contract, expected_keys, "Session-2 contract")
        if (
            contract["garment"] != "dark"
            or contract["frame_count"] != 602
            or contract["depth_frame_count"] != 602
            or contract["frame_ledger_path"] != SESSION2_FRAME_LEDGER_PATH
            or contract["intrinsics_path"] != SESSION2_INTRINSICS_PATH
            or contract["labels_path"] != SESSION2_LABELS_PATH
            or contract["rgb_video_path"] != SESSION2_RGB_VIDEO_PATH
            or contract["source_frames_path"] != SESSION2_SOURCE_FRAMES_PATH
        ):
            raise CanaryContractError("Session-2 garment/frame contract is not exact")
        allowed_origins = {
            "orbbec_session2",
            "session2_dark_annotation",
            "shared_intrinsics",
            "builder_metadata",
        }
        if not origins <= allowed_origins:
            raise CanaryContractError("Session-2 contains an unsupported content origin")
        if not set(roles) <= {
            "rgb_video",
            "metric_depth",
            "intrinsics",
            "labels",
            "frame_ledger",
            "coordinate_metadata",
            "source_frame_index",
        }:
            raise CanaryContractError("Session-2 contains an unsupported member role")
        expected_paths = {
            "rgb_video": contract["rgb_video_path"],
            "intrinsics": contract["intrinsics_path"],
            "labels": contract["labels_path"],
            "source_frame_index": contract["source_frames_path"],
            "frame_ledger": contract["frame_ledger_path"],
            "coordinate_metadata": SESSION2_COORDINATE_METADATA_PATH,
        }
        by_role = {}
        for member in members:
            by_role.setdefault(member["role"], []).append(member)
        if any(
            len(by_role.get(role, [])) != 1
            or by_role[role][0]["path"] != path
            for role, path in expected_paths.items()
        ):
            raise CanaryContractError("Session-2 required member paths are not exact")
        if (
            len(by_role.get("metric_depth", [])) != 602
            or set(by_role) != set(expected_paths) | {"metric_depth"}
        ):
            raise CanaryContractError("Session-2 required member counts are not exact")
        expected_origins = {
            "coordinate_metadata": "builder_metadata",
            "frame_ledger": "builder_metadata",
            "intrinsics": "shared_intrinsics",
            "labels": "session2_dark_annotation",
            "rgb_video": "orbbec_session2",
            "source_frame_index": "orbbec_session2",
        }
        if any(
            by_role[role][0]["content_origin"] != origin
            for role, origin in expected_origins.items()
        ) or any(
            member["content_origin"] != "orbbec_session2"
            for member in by_role["metric_depth"]
        ):
            raise CanaryContractError(
                "Session-2 required role-to-origin bindings are not exact"
            )
        expected_depth_paths = {
            "depth/depth_%06d.npy" % index for index in range(602)
        }
        if {member["path"] for member in by_role["metric_depth"]} != expected_depth_paths:
            raise CanaryContractError("Session-2 metric depth paths are not exact")
        ordered_depth = _require_hash_list(
            contract["ordered_depth_fingerprints"],
            "Session-2 ordered depth fingerprints",
            length=602,
            sorted_unique=False,
        )
        _require_sha256(
            expected_ordered_depth_list_sha256,
            "expected Session-2 ordered depth list SHA-256",
        )
        if hashlib.sha256(_canonical_json_bytes(ordered_depth)).hexdigest() != (
            expected_ordered_depth_list_sha256
        ):
            raise CanaryContractError(
                "Session-2 ordered depth list SHA-256 does not match"
            )
        depth_by_path = {
            member["path"]: member for member in by_role["metric_depth"]
        }
        for index, expected_sha256 in enumerate(ordered_depth):
            expected_path = "depth/depth_%06d.npy" % index
            if depth_by_path[expected_path]["sha256"] != expected_sha256:
                raise CanaryContractError(
                    "Session-2 metric depth order does not match fixed fingerprints"
                )
        expected_pins = _exact_keys(
            expected_train_member_sha256,
            frozenset(_PRODUCTION_TRAIN_MEMBER_SHA256),
            "expected Session-2 member hashes",
        )
        for role, expected_sha256 in expected_pins.items():
            _require_sha256(expected_sha256, "expected Session-2 member hash")
            if by_role[role][0]["sha256"] != expected_sha256:
                raise CanaryContractError(
                    "Session-2 %s SHA-256 is not exact" % role
                )
        fingerprints = contract["decoded_frame_fingerprints"]
        if (
            type(fingerprints) is not list
            or len(fingerprints) != 602
        ):
            raise CanaryContractError(
                "Session-2 decoded frame fingerprints must contain 602 ordered hashes"
            )
        for digest in fingerprints:
            _require_sha256(digest, "Session-2 decoded frame fingerprint")
            if digest in DENIED_TRAIN_CONTENT_SHA256:
                raise CanaryContractError("Session-2 contains denied held-out content")
        if content_partition is None:
            raise CanaryContractError("Session-2 content partition is missing")
        if fingerprints != content_partition["session2_decoded_frame_sha256"]:
            raise CanaryContractError(
                "Session-2 decoded fingerprints do not match the content partition"
            )
        denied_decoded = set(content_partition["session3_decoded_frame_sha256"])
        if any(digest in denied_decoded for digest in fingerprints):
            raise CanaryContractError(
                "Session-2 decoded frame collides with denied Session-3 content"
            )


def _validate_perception_input_manifest(
    semantic: object,
    outer_rows: list,
    *,
    input_name: str,
    current_identity: Mapping[str, str],
    expected_partition_sha256: str,
    expected_partition_list_sha256: Mapping[str, str],
    expected_train_member_sha256: Mapping[str, str],
    expected_ordered_depth_list_sha256: str,
    expected_geometry_support_sha256: Mapping[str, str],
    expected_geometry_smpl_manifest_sha256: str,
) -> frozenset:
    """Strict Task-3 builder contract, shared by builders and this verifier."""
    semantic = _exact_keys(
        semantic,
        PERCEPTION_INPUT_FIELDS,
        "inputs.%s semantic manifest" % input_name,
    )
    expected = _PERCEPTION_INPUT_EXPECTATIONS[input_name]
    if semantic["schema"] != PERCEPTION_INPUT_SCHEMA:
        raise CanaryContractError("inputs.%s semantic schema is not exact" % input_name)
    for key in ("artifact_kind", "source_role", "session_ids"):
        if not _strict_equal(semantic[key], expected[key]):
            raise CanaryContractError("inputs.%s semantic %s is not exact" % (input_name, key))
    if semantic["self_contained"] is not True:
        raise CanaryContractError("inputs.%s must be self-contained" % input_name)
    if semantic["resolvable_dependencies"] != []:
        raise CanaryContractError("inputs.%s has a resolvable dependency" % input_name)
    members, hashes = _validate_semantic_member_rows(
        semantic,
        outer_rows,
        input_name=input_name,
    )
    if _contains_forbidden_term(
        {
            "contract": semantic["contract"],
            "members": members,
            "session_ids": semantic["session_ids"],
        }
    ):
        raise CanaryContractError("input semantic content names held-out/yellow/evaluation data")
    if input_name == "train" and any(
        digest in DENIED_TRAIN_CONTENT_SHA256 for digest in hashes
    ):
        raise CanaryContractError("Session-2 contains denied held-out content")
    content_partition = _validate_source_attestations(
        semantic["source_attestations"],
        current_identity=current_identity,
        members=members,
        input_name=input_name,
        expected_partition_sha256=expected_partition_sha256,
        expected_partition_list_sha256=expected_partition_list_sha256,
    )
    _validate_kind_semantics(
        semantic,
        members,
        input_name=input_name,
        content_partition=content_partition,
        expected_train_member_sha256=expected_train_member_sha256,
        expected_ordered_depth_list_sha256=expected_ordered_depth_list_sha256,
        expected_geometry_support_sha256=expected_geometry_support_sha256,
        expected_geometry_smpl_manifest_sha256=(
            expected_geometry_smpl_manifest_sha256
        ),
    )
    return frozenset(hashes)


def validate_perception_input_manifest(
    semantic: object,
    outer_rows: list,
    *,
    input_name: str,
    current_identity: Mapping[str, str],
) -> frozenset:
    """Validate a semantic manifest against the production content partition."""
    return _validate_perception_input_manifest(
        semantic,
        outer_rows,
        input_name=input_name,
        current_identity=current_identity,
        expected_partition_sha256=HISTORICAL_MIXED_EVAL_PARTITION_SHA256,
        expected_partition_list_sha256=_PRODUCTION_PARTITION_LIST_SHA256,
        expected_train_member_sha256=_PRODUCTION_TRAIN_MEMBER_SHA256,
        expected_ordered_depth_list_sha256=(
            HISTORICAL_SESSION2_ORDERED_DEPTH_LIST_SHA256
        ),
        expected_geometry_support_sha256=_PRODUCTION_GEOMETRY_SUPPORT_SHA256,
        expected_geometry_smpl_manifest_sha256=(
            HUMMAN_SMPL_PARAMS_MANIFEST_SHA256
        ),
    )


def _build_stage0_provenance_expected(
    geometry_source_identity: Mapping[str, str],
) -> dict:
    identity = _exact_keys(
        geometry_source_identity,
        _INPUT_FIELDS,
        "expected Stage-0 geometry source identity",
    )
    return {
        "NSEQ": 907,
        "NVIEW": 4,
        "environment": {
            "identity_sha256": _ENVIRONMENT_IDENTITY_SHA256,
            "release_id": _ENVIRONMENT_RELEASE_ID,
        },
        "geometry_source": dict(identity),
        "mask_policy_sha256": MASK_POLICY_SHA256,
        "preprocessing": {
            "identifier": "wam_robot_mask_before_depth_to_normals_v1"
        },
        "sample_count": 3628,
        "sample_format": STAGE0_SAMPLE_FORMAT,
        "schema": STAGE0_PROVENANCE_SCHEMA,
        "seed": 0,
    }


def _validate_geometry_source_terms(
    release_fd: int,
    *,
    expected_source_terms: Mapping[str, object],
) -> None:
    document = _parse_canonical_json_object(
        _read_regular_stable_at(
            release_fd,
            "provenance/SOURCE_TERMS.json",
            "geometry source terms",
        ),
        "geometry source terms",
    )
    if not _strict_equal(document, expected_source_terms):
        raise CanaryContractError("geometry source terms contract is not exact")


def _geometry_sequence_ids_from_semantic(semantic: dict) -> tuple:
    sequence_ids = tuple(
        sorted(
            PurePosixPath(member["path"]).parts[2]
            for member in semantic["members"]
            if member["role"] == "smpl_params"
        )
    )
    if (
        len(sequence_ids) != 907
        or len(set(sequence_ids)) != 907
        or any(
            _HUMMAN_SEQUENCE_ID.fullmatch(sequence_id) is None
            for sequence_id in sequence_ids
        )
    ):
        raise CanaryContractError(
            "verified geometry release does not expose 907 canonical HuMMan sequences"
        )
    return sequence_ids


def _validate_session2_coordinate_metadata(
    release_fd: int,
    semantic: dict,
) -> None:
    document = _parse_canonical_json_object(
        _read_regular_stable_at(
            release_fd,
            SESSION2_COORDINATE_METADATA_PATH,
            "Session-2 coordinate metadata",
        ),
        "Session-2 coordinate metadata",
    )
    expected = _build_session2_coordinate_metadata_expected()
    if not _strict_equal(document, expected):
        raise CanaryContractError(
            "Session-2 coordinate metadata contract is not exact"
        )
    member = next(
        (
            row
            for row in semantic["members"]
            if row["role"] == "coordinate_metadata"
        ),
        None,
    )
    if (
        member is None
        or member["path"] != SESSION2_COORDINATE_METADATA_PATH
        or member["sha256"] != SESSION2_COORDINATE_METADATA_SHA256
    ):
        raise CanaryContractError(
            "Session-2 coordinate metadata member pin is not exact"
        )


def _validate_stage0_provenance(
    release_fd: int,
    semantic: dict,
    *,
    geometry_source_identity: Mapping[str, str],
) -> None:
    provenance = _parse_canonical_json_object(
        _read_regular_stable_at(
            release_fd,
            semantic["contract"]["provenance_path"],
            "Stage-0 provenance",
        ),
        "Stage-0 provenance",
    )
    expected = _build_stage0_provenance_expected(geometry_source_identity)
    if not _strict_equal(provenance, expected):
        raise CanaryContractError("Stage-0 provenance contract is not exact")


def _validate_stage0_ledger_documents(
    semantic: dict,
    sample_ledger: dict,
    split_ledger: dict,
    *,
    expected_geometry_sequence_ids: tuple,
) -> list:
    if (
        type(expected_geometry_sequence_ids) is not tuple
        or len(expected_geometry_sequence_ids) != 907
        or tuple(sorted(set(expected_geometry_sequence_ids)))
        != expected_geometry_sequence_ids
        or any(
            type(sequence_id) is not str
            or _HUMMAN_SEQUENCE_ID.fullmatch(sequence_id) is None
            for sequence_id in expected_geometry_sequence_ids
        )
    ):
        raise CanaryContractError(
            "verified geometry sequence IDs must be 907 sorted unique HuMMan IDs"
        )
    _exact_keys(
        sample_ledger,
        frozenset(("samples", "schema")),
        "Stage-0 sample ledger",
    )
    if sample_ledger["schema"] != STAGE0_SAMPLE_LEDGER_SCHEMA:
        raise CanaryContractError("Stage-0 sample ledger schema is not exact")
    samples = sample_ledger["samples"]
    if type(samples) is not list or len(samples) != 3628:
        raise CanaryContractError("Stage-0 sample ledger must contain exactly 3628 rows")
    normalized = []
    for index, row in enumerate(samples):
        row = _exact_keys(
            row,
            frozenset(
                (
                    "sample_id",
                    "sample_path",
                    "sample_sha256",
                    "sequence_id",
                    "split",
                    "subject_id",
                    "view_index",
                )
            ),
            "Stage-0 sample ledger row %d" % index,
        )
        sample_match = (
            _STAGE0_SAMPLE_ID.fullmatch(row["sample_id"])
            if type(row["sample_id"]) is str
            else None
        )
        sequence_match = (
            _HUMMAN_SEQUENCE_ID.fullmatch(row["sequence_id"])
            if type(row["sequence_id"]) is str
            else None
        )
        if (
            type(row["sample_id"]) is not str
            or not row["sample_id"]
            or type(row["sequence_id"]) is not str
            or sequence_match is None
            or type(row["subject_id"]) is not str
            or re.fullmatch(r"p[0-9]{6}", row["subject_id"]) is None
            or type(row["view_index"]) is not int
            or row["view_index"] not in (0, 1, 2, 3)
            or row["split"] not in ("train", "validation")
            or sample_match is None
            or sample_match.group(1) != row["sequence_id"]
            or int(sample_match.group(2)) != row["view_index"]
            or sequence_match.group(1) != row["subject_id"]
            or row["sample_path"] != "samples/%s.npz" % row["sample_id"]
        ):
            raise CanaryContractError("Stage-0 sample ledger row is invalid")
        _require_sha256(row["sample_sha256"], "Stage-0 sample ledger SHA-256")
        normalized.append(row)
    sample_ids = [row["sample_id"] for row in normalized]
    if sample_ids != sorted(set(sample_ids)):
        raise CanaryContractError("Stage-0 sample ledger IDs must be sorted and unique")
    views_by_sequence = {}
    for row in normalized:
        views_by_sequence.setdefault(row["sequence_id"], []).append(
            row["view_index"]
        )
    if tuple(sorted(views_by_sequence)) != expected_geometry_sequence_ids:
        raise CanaryContractError(
            "Stage-0 sequences do not exactly match the verified geometry release"
        )
    if any(
        sorted(views) != [0, 1, 2, 3]
        for views in views_by_sequence.values()
    ):
        raise CanaryContractError(
            "Stage-0 must contain exactly views 0,1,2,3 for every geometry sequence"
        )
    generated = {
        member["path"]: member["sha256"]
        for member in semantic["members"]
        if member["role"] == "generated_sample"
    }
    ledger_generated = {
        row["sample_path"]: row["sample_sha256"] for row in normalized
    }
    if ledger_generated != generated:
        raise CanaryContractError(
            "Stage-0 sample ledger does not map one-to-one to generated payload members"
        )

    _exact_keys(
        split_ledger,
        frozenset(
            (
                "schema",
                "split_strategy",
                "train_subject_ids",
                "validation_subject_ids",
            )
        ),
        "Stage-0 subject split ledger",
    )
    if (
        split_ledger["schema"] != STAGE0_SUBJECT_SPLIT_SCHEMA
        or split_ledger["split_strategy"] != "subject_grouped_fixed_v1"
    ):
        raise CanaryContractError("Stage-0 subject split ledger metadata is not exact")
    train = split_ledger["train_subject_ids"]
    validation = split_ledger["validation_subject_ids"]
    for subjects, label in ((train, "train"), (validation, "validation")):
        if (
            type(subjects) is not list
            or not subjects
            or any(type(subject) is not str or not subject for subject in subjects)
            or subjects != sorted(set(subjects))
        ):
            raise CanaryContractError(
                "Stage-0 %s subject ledger must be nonempty, sorted, and unique"
                % label
            )
    train_set = set(train)
    validation_set = set(validation)
    if train_set & validation_set:
        raise CanaryContractError("Stage-0 subject ledgers must be disjoint")
    sample_subjects = {row["subject_id"] for row in normalized}
    if sample_subjects != train_set | validation_set:
        raise CanaryContractError("Stage-0 sample subjects do not match split ledger")
    if any(
        (row["split"] == "train" and row["subject_id"] not in train_set)
        or (
            row["split"] == "validation"
            and row["subject_id"] not in validation_set
        )
        for row in normalized
    ):
        raise CanaryContractError("Stage-0 sample split violates subject grouping")
    return normalized


def _validate_stage0_ledgers(
    release_fd: int,
    semantic: dict,
    *,
    expected_geometry_sequence_ids: tuple,
) -> list:
    contract = semantic["contract"]
    sample_ledger = _parse_canonical_json_object(
        _read_regular_stable_at(
            release_fd,
            contract["sample_ledger_path"],
            "Stage-0 sample ledger",
        ),
        "Stage-0 sample ledger",
    )
    split_ledger = _parse_canonical_json_object(
        _read_regular_stable_at(
            release_fd,
            contract["subject_split_ledger_path"],
            "Stage-0 subject split ledger",
        ),
        "Stage-0 subject split ledger",
    )
    return _validate_stage0_ledger_documents(
        semantic,
        sample_ledger,
        split_ledger,
        expected_geometry_sequence_ids=expected_geometry_sequence_ids,
    )


def _validate_stage0_npz_container(payload: bytes) -> frozenset:
    """Bound and validate every fixed ZIP/NPY header before ``np.load``."""
    import numpy as np

    expected_keys = frozenset(_STAGE0_NPZ_ARRAY_SPECS)
    expected_members = {key + ".npy" for key in expected_keys}
    try:
        with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if (
                len(names) != len(expected_members)
                or len(names) != len(set(names))
                or set(names) != expected_members
                or any(info.is_dir() or info.flag_bits & 0x1 for info in infos)
            ):
                raise CanaryContractError(
                    "Stage-0 NPZ ZIP member closure is not exact"
                )
            by_name = {info.filename: info for info in infos}
            total_expanded = 0
            total_compressed = 0
            for key, (dtype_name, shape, item_size) in (
                _STAGE0_NPZ_ARRAY_SPECS.items()
            ):
                info = by_name[key + ".npy"]
                expected_data_bytes = math.prod(shape or (1,)) * item_size
                if (
                    info.compress_type
                    not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                    or info.compress_size <= 0
                    or info.file_size < expected_data_bytes + 10
                    or info.file_size
                    > expected_data_bytes + STAGE0_NPY_MAX_HEADER_BYTES
                ):
                    raise CanaryContractError(
                        "Stage-0 NPY header uncompressed size is outside its exact bound"
                    )
                if (
                    info.file_size
                    > info.compress_size * STAGE0_NPZ_MAX_COMPRESSION_RATIO
                ):
                    raise CanaryContractError(
                        "Stage-0 NPZ member compression ratio exceeds its hard limit"
                    )
                total_expanded += info.file_size
                total_compressed += info.compress_size
                try:
                    with archive.open(info, mode="r") as member:
                        version = np.lib.format.read_magic(member)
                        if version == (1, 0):
                            actual_shape, fortran_order, actual_dtype = (
                                np.lib.format.read_array_header_1_0(
                                    member,
                                    max_header_size=STAGE0_NPY_MAX_HEADER_BYTES,
                                )
                            )
                        elif version == (2, 0):
                            actual_shape, fortran_order, actual_dtype = (
                                np.lib.format.read_array_header_2_0(
                                    member,
                                    max_header_size=STAGE0_NPY_MAX_HEADER_BYTES,
                                )
                            )
                        else:
                            raise CanaryContractError(
                                "Stage-0 NPY header version is not allowed"
                            )
                        header_bytes = member.tell()
                except CanaryContractError:
                    raise
                except (
                    EOFError,
                    OSError,
                    RuntimeError,
                    UnicodeError,
                    ValueError,
                    zipfile.BadZipFile,
                ) as exc:
                    raise CanaryContractError(
                        "Stage-0 NPY header is invalid or exceeds its hard limit"
                    ) from exc
                if (
                    actual_shape != shape
                    or fortran_order is not False
                    or actual_dtype != np.dtype(dtype_name)
                    or header_bytes > STAGE0_NPY_MAX_HEADER_BYTES
                    or info.file_size != header_bytes + expected_data_bytes
                ):
                    raise CanaryContractError(
                        "Stage-0 NPY header dtype/shape/layout is not exact"
                    )
            if (
                total_expanded > STAGE0_NPZ_MAX_EXPANDED_BYTES
                or total_compressed <= 0
                or total_expanded
                > total_compressed * STAGE0_NPZ_MAX_COMPRESSION_RATIO
            ):
                raise CanaryContractError(
                    "Stage-0 NPZ total expansion or compression ratio exceeds its hard limit"
                )
    except CanaryContractError:
        raise
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        raise CanaryContractError("Stage-0 sample is not a valid NPZ") from exc
    return expected_keys


def validate_stage0_npz_bytes(
    payload: bytes,
    *,
    expected_sample_id: str,
    expected_sequence_id: str,
    expected_subject_id: str,
    expected_view_index: int,
) -> None:
    """Validate one production Stage-0 sample as an exact, pickle-free NPZ."""
    sample_match = (
        _STAGE0_SAMPLE_ID.fullmatch(expected_sample_id)
        if type(expected_sample_id) is str
        else None
    )
    sequence_match = (
        _HUMMAN_SEQUENCE_ID.fullmatch(expected_sequence_id)
        if type(expected_sequence_id) is str
        else None
    )
    if (
        type(payload) is not bytes
        or sample_match is None
        or sequence_match is None
        or type(expected_subject_id) is not str
        or re.fullmatch(r"p[0-9]{6}", expected_subject_id) is None
        or type(expected_view_index) is not int
        or expected_view_index not in (0, 1, 2, 3)
        or sample_match.group(1) != expected_sequence_id
        or int(sample_match.group(2)) != expected_view_index
        or sequence_match.group(1) != expected_subject_id
    ):
        raise CanaryContractError("Stage-0 NPZ expected identity is invalid")
    if len(payload) > STAGE0_NPZ_MAX_COMPRESSED_BYTES:
        raise CanaryContractError(
            "Stage-0 NPZ compressed payload hard limit is exceeded"
        )
    expected_keys = _validate_stage0_npz_container(payload)

    try:
        import numpy as np

        with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
            if frozenset(arrays.files) != expected_keys:
                raise CanaryContractError("Stage-0 NPZ array keys are not exact")
            loaded = {key: arrays[key] for key in expected_keys}
    except CanaryContractError:
        raise
    except Exception as exc:
        raise CanaryContractError(
            "Stage-0 NPZ arrays could not be loaded without pickle"
        ) from exc

    depth = loaded["depth_m"]
    if (
        depth.dtype != np.dtype("float32")
        or depth.shape != STAGE0_IMAGE_SHAPE
        or not np.isfinite(depth).all()
        or np.any(depth < 0.0)
        or np.any(depth > 10.0)
    ):
        raise CanaryContractError("Stage-0 depth_m dtype/shape/range is invalid")
    positive_mask = depth > 0.0
    positive_depth = depth[positive_mask]
    if positive_depth.size < STAGE0_MIN_POSITIVE_DEPTH_PIXELS:
        raise CanaryContractError(
            "Stage-0 positive depth support is below the exporter gate"
        )
    if float(np.ptp(positive_depth)) <= 1e-6:
        raise CanaryContractError("Stage-0 depth_m is degenerate")
    positive_y, positive_x = np.nonzero(positive_mask)
    support_bbox_area = int(
        (positive_y.max() - positive_y.min() + 1)
        * (positive_x.max() - positive_x.min() + 1)
    )
    depth_hole_fraction = 1.0 - positive_depth.size / support_bbox_area
    if depth_hole_fraction > STAGE0_DEPTH_HOLE_FRACTION_MAX:
        raise CanaryContractError(
            "Stage-0 depth hole fraction exceeds the frozen 0.4 policy"
        )

    intrinsics = loaded["intrinsics_px"]
    if (
        intrinsics.dtype != np.dtype("float64")
        or intrinsics.shape != (3, 3)
        or not np.array_equal(
            intrinsics,
            np.asarray(STAGE0_INTRINSICS_PX, dtype=np.float64),
        )
    ):
        raise CanaryContractError("Stage-0 intrinsics_px is not exact")

    normals = loaded["normal_u8"]
    if (
        normals.dtype != np.dtype("uint8")
        or normals.shape != STAGE0_IMAGE_SHAPE + (3,)
        or np.any(normals > 254)
        or np.any(normals[~positive_mask] != 127)
    ):
        raise CanaryContractError("Stage-0 normal_u8 dtype/shape/background is invalid")
    encoded_background = np.all(normals == 127, axis=-1)
    valid_normal_mask = positive_mask & ~encoded_background
    valid_normal_count = int(np.count_nonzero(valid_normal_mask))
    if (
        valid_normal_count
        < positive_depth.size * STAGE0_MIN_VALID_NORMAL_FRACTION
    ):
        raise CanaryContractError(
            "Stage-0 valid foreground normal support is insufficient"
        )
    decoded_normals = (
        normals[valid_normal_mask].astype(np.float32) - 127.0
    ) / 127.0
    decoded_normal_norms = np.linalg.norm(decoded_normals, axis=1)
    if (
        not np.isfinite(decoded_normal_norms).all()
        or np.any(decoded_normal_norms < STAGE0_MIN_DECODED_NORMAL_NORM)
        or np.any(decoded_normal_norms > STAGE0_MAX_DECODED_NORMAL_NORM)
    ):
        raise CanaryContractError(
            "Stage-0 decoded foreground normal norm is outside the unit interval"
        )
    foreground_normals = normals[valid_normal_mask]
    if not np.any(foreground_normals != foreground_normals[0]):
        raise CanaryContractError("Stage-0 normal_u8 is degenerate")

    points = loaded["points_px"]
    if (
        points.dtype != np.dtype("float32")
        or points.shape != (10, 2)
        or not np.isfinite(points).all()
        or np.any(points[:, 0] < 0.0)
        or np.any(points[:, 0] >= STAGE0_IMAGE_SHAPE[1])
        or np.any(points[:, 1] < 0.0)
        or np.any(points[:, 1] >= STAGE0_IMAGE_SHAPE[0])
    ):
        raise CanaryContractError("Stage-0 points_px dtype/shape/range is invalid")
    point_x = np.floor(points[:, 0]).astype(np.intp)
    point_y = np.floor(points[:, 1]).astype(np.intp)
    unique_point_pixels = np.unique(
        np.column_stack((point_x, point_y)), axis=0
    )
    if len(unique_point_pixels) < STAGE0_MIN_UNIQUE_VISIBLE_POINTS:
        raise CanaryContractError(
            "Stage-0 does not contain enough unique visible points"
        )
    centered_points = points.astype(np.float64) - np.mean(points, axis=0)
    point_spans = np.ptp(points, axis=0)
    singular_values = np.linalg.svd(centered_points, compute_uv=False)
    if (
        np.any(point_spans < STAGE0_MIN_POINT_AXIS_SPAN_PX)
        or len(singular_values) < 2
        or singular_values[1] < STAGE0_MIN_POINT_MINOR_SINGULAR_VALUE_PX
    ):
        raise CanaryContractError(
            "Stage-0 visible points lack minimum 2D geometric spread"
        )
    if np.any(depth[point_y, point_x] <= 0.0):
        raise CanaryContractError("Stage-0 point has no positive metric depth")

    visibility = loaded["visibility"]
    if (
        visibility.dtype != np.dtype("bool")
        or visibility.shape != (10,)
        or not bool(np.all(visibility))
    ):
        raise CanaryContractError("Stage-0 visibility must be bool[10] all true")

    point_order = loaded["point_order"]
    if (
        point_order.dtype != np.dtype("<U2")
        or point_order.shape != (10,)
        or tuple(point_order.tolist()) != STAGE0_POINT_ORDER
    ):
        raise CanaryContractError("Stage-0 point_order is not exact")

    scalar_expectations = (
        ("sample_id", "<U18", expected_sample_id),
        ("sequence_id", "<U15", expected_sequence_id),
        ("subject_id", "<U7", expected_subject_id),
        ("view_index", "int16", expected_view_index),
    )
    for key, dtype, expected_value in scalar_expectations:
        value = loaded[key]
        if (
            value.shape != ()
            or value.dtype != np.dtype(dtype)
            or value.item() != expected_value
        ):
            raise CanaryContractError(
                "Stage-0 NPZ %s metadata is not exact" % key
            )
    source_frame_index = loaded["source_frame_index"]
    if (
        source_frame_index.shape != ()
        or source_frame_index.dtype != np.dtype("int32")
        or type(source_frame_index.item()) is not int
        or source_frame_index.item() < 0
    ):
        raise CanaryContractError(
            "Stage-0 source_frame_index metadata is invalid"
        )


def _read_descriptor_bytes_bounded(
    descriptor: int,
    *,
    initial_size: int,
    maximum_size: int,
) -> bytes:
    try:
        if initial_size < 0 or maximum_size < 0 or initial_size > maximum_size:
            raise CanaryContractError(
                "Stage-0 NPZ compressed payload hard limit is exceeded"
            )
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks = []
        remaining = initial_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise CanaryContractError(
                    "Stage-0 pinned sample shrank while it was read"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise CanaryContractError(
                "Stage-0 pinned sample grew while it was read"
            )
        return b"".join(chunks)
    except CanaryContractError:
        raise
    except OSError as exc:
        raise CanaryContractError("pinned payload descriptor could not be read") from exc


def _validate_stage0_npz_members(
    release_fd: int,
    ledger_rows: list,
    *,
    test_stage0_npz_validator_fd: Optional[Callable[[int, dict], None]],
) -> None:
    for row in ledger_rows:
        with _open_regular_pinned_at(
            release_fd,
            row["sample_path"],
            "Stage-0 sample " + row["sample_id"],
        ) as descriptor:
            try:
                payload_before = os.fstat(descriptor)
            except OSError as exc:
                raise CanaryContractError(
                    "Stage-0 pinned sample could not be sized"
                ) from exc
            if (
                not stat.S_ISREG(payload_before.st_mode)
                or payload_before.st_size > STAGE0_NPZ_MAX_COMPRESSED_BYTES
            ):
                raise CanaryContractError(
                    "Stage-0 NPZ compressed payload hard limit is exceeded"
                )
            payload = _read_descriptor_bytes_bounded(
                descriptor,
                initial_size=payload_before.st_size,
                maximum_size=STAGE0_NPZ_MAX_COMPRESSED_BYTES,
            )
            try:
                payload_after = os.fstat(descriptor)
            except OSError as exc:
                raise CanaryContractError(
                    "Stage-0 pinned sample could not be re-sized"
                ) from exc
            if (
                len(payload) != payload_before.st_size
                or _stat_identity(payload_before, stable=True)
                != _stat_identity(payload_after, stable=True)
            ):
                raise CanaryContractError(
                    "Stage-0 pinned sample changed while it was read"
                )
            if hashlib.sha256(payload).hexdigest() != row["sample_sha256"]:
                raise CanaryContractError(
                    "Stage-0 pinned sample SHA-256 does not match its ledger"
                )
            if test_stage0_npz_validator_fd is None:
                validate_stage0_npz_bytes(
                    payload,
                    expected_sample_id=row["sample_id"],
                    expected_sequence_id=row["sequence_id"],
                    expected_subject_id=row["subject_id"],
                    expected_view_index=row["view_index"],
                )
            else:
                try:
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    result = test_stage0_npz_validator_fd(descriptor, deepcopy(row))
                except CanaryContractError:
                    raise
                except Exception as exc:
                    raise CanaryContractError(
                        "test Stage-0 NPZ validator failed"
                    ) from exc
                if result is not None:
                    raise CanaryContractError(
                        "test Stage-0 NPZ validator must return None"
                    )


def _validate_session2_frame_ledger_document(
    semantic: dict,
    frame_ledger: dict,
) -> None:
    _exact_keys(
        frame_ledger,
        frozenset(("frames", "schema")),
        "Session-2 frame ledger",
    )
    if frame_ledger["schema"] != SESSION2_FRAME_LEDGER_SCHEMA:
        raise CanaryContractError("Session-2 frame ledger schema is not exact")
    frames = frame_ledger["frames"]
    if type(frames) is not list or len(frames) != 602:
        raise CanaryContractError("Session-2 frame ledger must contain 602 rows")
    depth_by_path = {
        member["path"]: member["sha256"]
        for member in semantic["members"]
        if member["role"] == "metric_depth"
    }
    ordered_depth = _require_hash_list(
        semantic["contract"]["ordered_depth_fingerprints"],
        "Session-2 ordered depth fingerprints",
        length=602,
        sorted_unique=False,
    )
    ledger_depth = {}
    rgb_hashes = []
    for expected_index, row in enumerate(frames):
        row = _exact_keys(
            row,
            frozenset(
                ("depth_path", "depth_sha256", "frame_index", "rgb_sha256")
            ),
            "Session-2 frame ledger row %d" % expected_index,
        )
        if type(row["frame_index"]) is not int or row["frame_index"] != expected_index:
            raise CanaryContractError(
                "Session-2 frame ledger indices must be contiguous and ordered"
            )
        depth_path = _relative_payload_path(
            row["depth_path"], "Session-2 frame ledger depth_path"
        )
        depth_sha256 = _require_sha256(
            row["depth_sha256"], "Session-2 frame ledger depth SHA-256"
        )
        expected_depth_path = "depth/depth_%06d.npy" % expected_index
        if (
            depth_path != expected_depth_path
            or depth_sha256 != ordered_depth[expected_index]
        ):
            raise CanaryContractError(
                "Session-2 frame ledger depth order is not exact"
            )
        rgb_hashes.append(
            _require_sha256(
                row["rgb_sha256"], "Session-2 frame ledger RGB SHA-256"
            )
        )
        if depth_path in ledger_depth:
            raise CanaryContractError(
                "Session-2 frame ledger contains a duplicate depth path"
            )
        ledger_depth[depth_path] = depth_sha256
    if ledger_depth != depth_by_path:
        raise CanaryContractError(
            "Session-2 frame ledger does not map one-to-one to metric depth members"
        )
    if rgb_hashes != semantic["contract"]["decoded_frame_fingerprints"]:
        raise CanaryContractError(
            "Session-2 frame ledger RGB hashes do not match ordered fingerprints"
        )
    partition = semantic["source_attestations"][0]["content_partition"]
    if rgb_hashes != partition["session2_decoded_frame_sha256"]:
        raise CanaryContractError(
            "Session-2 frame ledger RGB hashes do not match source partition"
        )


def _validate_session2_frame_ledger(release_fd: int, semantic: dict) -> None:
    path = semantic["contract"]["frame_ledger_path"]
    frame_ledger = _parse_canonical_json_object(
        _read_regular_stable_at(
            release_fd,
            path,
            "Session-2 frame ledger",
        ),
        "Session-2 frame ledger",
    )
    _validate_session2_frame_ledger_document(semantic, frame_ledger)


def _validate_source_frame_index_bytes(payload: bytes) -> None:
    """Validate the exact 602-row historical Orbbec frame-index semantics."""
    if type(payload) is not bytes:
        raise CanaryContractError("Session-2 source frame index must be bytes")
    try:
        text = payload.decode("utf-8", errors="strict")
        rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise CanaryContractError(
            "Session-2 source frame index is not strict UTF-8 CSV"
        ) from exc
    expected_header = ["idx", "t", "cam_index", "has_depth", "depth_scale"]
    if len(rows) != 603 or rows[0] != expected_header:
        raise CanaryContractError(
            "Session-2 source frame index schema/count is not exact"
        )
    previous_t = None
    previous_cam_index = None
    for expected_index, row in enumerate(rows[1:]):
        if len(row) != 5 or row[0] != str(expected_index):
            raise CanaryContractError(
                "Session-2 source frame index rows are not contiguous"
            )
        cam_index_token = row[2]
        if re.fullmatch(r"(?:0|[1-9][0-9]*)", cam_index_token) is None:
            raise CanaryContractError(
                "Session-2 source frame cam_index is not canonical"
            )
        try:
            timestamp = float(row[1])
            cam_index = int(cam_index_token)
            depth_scale = float(row[4])
        except ValueError as exc:
            raise CanaryContractError(
                "Session-2 source frame numeric values are malformed"
            ) from exc
        if (
            not math.isfinite(timestamp)
            or not math.isfinite(depth_scale)
            or row[3] != "1"
            or depth_scale.hex() != float(1.0).hex()
        ):
            raise CanaryContractError(
                "Session-2 source frame depth metadata is not exact"
            )
        if (
            previous_t is not None
            and (
                timestamp <= previous_t
                or cam_index <= previous_cam_index
            )
        ):
            raise CanaryContractError(
                "Session-2 source frame time/camera indices are not increasing"
            )
        previous_t = timestamp
        previous_cam_index = cam_index


def _validate_session2_source_frame_index(release_fd: int, semantic: dict) -> None:
    _validate_source_frame_index_bytes(
        _read_regular_stable_at(
            release_fd,
            semantic["contract"]["source_frames_path"],
            "Session-2 source frame index",
        )
    )


def _decode_session2_bgr8_hashes_from_fd(descriptor: int) -> list:
    """Production-only Linux/FFMPEG decode from a held no-follow release FD."""
    if not sys.platform.startswith("linux"):
        raise CanaryContractError("production video verification requires H200 Linux")
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise CanaryContractError("OpenCV is unavailable for production video verification") from exc
    os.lseek(descriptor, 0, os.SEEK_SET)
    capture = cv2.VideoCapture("/proc/self/fd/%d" % descriptor, cv2.CAP_FFMPEG)
    if not capture.isOpened():
        capture.release()
        raise CanaryContractError("descriptor-pinned Session-2 RGB video could not be decoded")
    hashes = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if len(hashes) >= 602:
                raise CanaryContractError("Session-2 RGB video contains more than 602 frames")
            if (
                str(frame.dtype) != DECODED_FRAME_DTYPE
                or tuple(frame.shape) != DECODED_FRAME_SHAPE
                or not bool(frame.flags.c_contiguous)
            ):
                raise CanaryContractError(
                    "decoded Session-2 frame is not contiguous 1080x1920x3 BGR uint8"
                )
            hashes.append(bgr8_frame_sha256(frame.tobytes(order="C")))
    finally:
        capture.release()
    if len(hashes) != 602:
        raise CanaryContractError("Session-2 RGB video must decode exactly 602 frames")
    return hashes


def _verify_session2_video_frames(
    release_fd: int,
    semantic: dict,
    *,
    test_frame_hashes_from_video_fd: Optional[Callable[[int], list]],
) -> None:
    path = semantic["contract"]["rgb_video_path"]
    with _open_regular_pinned_at(
        release_fd,
        path,
        "Session-2 RGB video",
    ) as descriptor:
        if test_frame_hashes_from_video_fd is None:
            hashes = _decode_session2_bgr8_hashes_from_fd(descriptor)
        else:
            try:
                hashes = test_frame_hashes_from_video_fd(descriptor)
            except CanaryContractError:
                raise
            except Exception as exc:
                raise CanaryContractError("test frame verifier failed") from exc
    hashes = _require_hash_list(
        hashes,
        "descriptor-decoded Session-2 frame hashes",
        length=602,
        sorted_unique=False,
    )
    partition = semantic["source_attestations"][0]["content_partition"]
    if (
        hashes != semantic["contract"]["decoded_frame_fingerprints"]
        or hashes != partition["session2_decoded_frame_sha256"]
    ):
        raise CanaryContractError(
            "descriptor-decoded Session-2 frame hashes do not match ledger/partition"
        )


def _verify_one_input_release(
    input_name: str,
    identity: Mapping[str, str],
    root_fd: int,
    *,
    expected_partition_sha256: str,
    expected_partition_list_sha256: Mapping[str, str],
    expected_train_member_sha256: Mapping[str, str],
    expected_ordered_depth_list_sha256: str,
    expected_geometry_support_sha256: Mapping[str, str],
    expected_geometry_smpl_manifest_sha256: str,
    expected_geometry_source_terms: Mapping[str, object],
    expected_geometry_source_identity: Mapping[str, str],
    expected_geometry_sequence_ids: tuple,
    test_stage0_npz_validator_fd: Optional[Callable[[int, dict], None]],
    test_frame_hashes_from_video_fd: Optional[Callable[[int], list]],
) -> tuple:
    release_fd = _open_release_at_root(root_fd, identity["release_id"], "inputs." + input_name)
    try:
        try:
            first_envelope = _artifact_contract.verify_release_fd(release_fd)
        except (_artifact_contract.ArtifactError, OSError) as exc:
            raise CanaryContractError(
                "inputs.%s artifact verification failed" % input_name
            ) from exc
        manifest_before = _read_regular_stable_at(
            release_fd, _artifact_contract.MANIFEST_NAME, "release manifest"
        )
        completion_before = _read_regular_stable_at(
            release_fd, _artifact_contract.COMPLETE_NAME, "completion marker"
        )
        semantic_bytes_before = _read_regular_stable_at(
            release_fd, PERCEPTION_INPUT_NAME, "perception semantic manifest"
        )
        strict_envelope = _parse_canonical_json_object(
            manifest_before, "release manifest"
        )
        if not _strict_equal(first_envelope, strict_envelope):
            raise CanaryContractError("artifact verifier envelope changed during consumption")
        payload, rows = _validate_release_envelope(
            strict_envelope,
            manifest_before,
            completion_before,
            input_name=input_name,
            identity=identity,
        )
        if hashlib.sha256(semantic_bytes_before).hexdigest() != identity[
            "semantic_manifest_sha256"
        ]:
            raise CanaryContractError(
                "inputs.%s semantic manifest SHA-256 does not match configured identity"
                % input_name
            )
        semantic = _parse_canonical_json_object(
            semantic_bytes_before,
            "inputs.%s semantic manifest" % input_name,
        )
        hashes = _validate_perception_input_manifest(
            semantic,
            rows,
            input_name=input_name,
            current_identity=identity,
            expected_partition_sha256=expected_partition_sha256,
            expected_partition_list_sha256=expected_partition_list_sha256,
            expected_train_member_sha256=expected_train_member_sha256,
            expected_ordered_depth_list_sha256=(
                expected_ordered_depth_list_sha256
            ),
            expected_geometry_support_sha256=expected_geometry_support_sha256,
            expected_geometry_smpl_manifest_sha256=(
                expected_geometry_smpl_manifest_sha256
            ),
        )
        if input_name == "geometry_source":
            _validate_geometry_source_terms(
                release_fd,
                expected_source_terms=expected_geometry_source_terms,
            )
            geometry_sequence_ids = _geometry_sequence_ids_from_semantic(semantic)
        elif input_name == "stage0":
            ledger_rows = _validate_stage0_ledgers(
                release_fd,
                semantic,
                expected_geometry_sequence_ids=expected_geometry_sequence_ids,
            )
            _validate_stage0_npz_members(
                release_fd,
                ledger_rows,
                test_stage0_npz_validator_fd=test_stage0_npz_validator_fd,
            )
            _validate_stage0_provenance(
                release_fd,
                semantic,
                geometry_source_identity=expected_geometry_source_identity,
            )
            geometry_sequence_ids = ()
        elif input_name == "train":
            _validate_session2_coordinate_metadata(release_fd, semantic)
            _validate_session2_frame_ledger(release_fd, semantic)
            _validate_session2_source_frame_index(release_fd, semantic)
            _verify_session2_video_frames(
                release_fd,
                semantic,
                test_frame_hashes_from_video_fd=test_frame_hashes_from_video_fd,
            )
            geometry_sequence_ids = ()
        try:
            second_envelope = _artifact_contract.verify_release_fd(release_fd)
        except (_artifact_contract.ArtifactError, OSError) as exc:
            raise CanaryContractError(
                "inputs.%s changed during artifact verification" % input_name
            ) from exc
        if not _strict_equal(first_envelope, second_envelope):
            raise CanaryContractError("input release envelope changed during verification")
        if (
            manifest_before
            != _read_regular_stable_at(
                release_fd, _artifact_contract.MANIFEST_NAME, "release manifest"
            )
            or completion_before
            != _read_regular_stable_at(
                release_fd, _artifact_contract.COMPLETE_NAME, "completion marker"
            )
            or semantic_bytes_before
            != _read_regular_stable_at(
                release_fd, PERCEPTION_INPUT_NAME, "perception semantic manifest"
            )
        ):
            raise CanaryContractError("input release metadata changed during verification")
        del payload
        return hashes, geometry_sequence_ids
    finally:
        os.close(release_fd)


def _verify_input_releases_at_roots(
    inputs: Mapping[str, Mapping[str, str]],
    release_roots: Mapping[str, Union[str, Path]],
    *,
    expected_partition_sha256: Optional[str] = None,
    expected_partition_list_sha256: Optional[Mapping[str, str]] = None,
    expected_train_member_sha256: Optional[Mapping[str, str]] = None,
    expected_ordered_depth_list_sha256: Optional[str] = None,
    expected_geometry_support_sha256: Optional[Mapping[str, str]] = None,
    expected_geometry_smpl_manifest_sha256: Optional[str] = None,
    expected_geometry_source_terms: Optional[Mapping[str, object]] = None,
    test_stage0_npz_validator_fd: Optional[
        Callable[[int, dict], None]
    ] = None,
    test_frame_hashes_from_video_fd: Optional[Callable[[int], list]] = None,
) -> InputVerificationReceipt:
    """Verify exactly three releases beneath explicitly pinned test/production roots."""
    checked_inputs = _validate_inputs({"inputs": deepcopy(inputs)})
    roots = _exact_keys(
        release_roots,
        frozenset(_PERCEPTION_INPUT_EXPECTATIONS),
        "perception input release roots",
    )
    selected_partition_sha256 = (
        HISTORICAL_MIXED_EVAL_PARTITION_SHA256
        if expected_partition_sha256 is None
        else expected_partition_sha256
    )
    selected_partition_list_sha256 = (
        _PRODUCTION_PARTITION_LIST_SHA256
        if expected_partition_list_sha256 is None
        else expected_partition_list_sha256
    )
    selected_train_member_sha256 = (
        _PRODUCTION_TRAIN_MEMBER_SHA256
        if expected_train_member_sha256 is None
        else expected_train_member_sha256
    )
    selected_ordered_depth_list_sha256 = (
        HISTORICAL_SESSION2_ORDERED_DEPTH_LIST_SHA256
        if expected_ordered_depth_list_sha256 is None
        else expected_ordered_depth_list_sha256
    )
    selected_geometry_support_sha256 = (
        _PRODUCTION_GEOMETRY_SUPPORT_SHA256
        if expected_geometry_support_sha256 is None
        else expected_geometry_support_sha256
    )
    selected_geometry_smpl_manifest_sha256 = (
        HUMMAN_SMPL_PARAMS_MANIFEST_SHA256
        if expected_geometry_smpl_manifest_sha256 is None
        else expected_geometry_smpl_manifest_sha256
    )
    selected_geometry_source_terms = (
        _build_geometry_source_terms_expected(
            _PRODUCTION_GEOMETRY_SUPPORT_SHA256,
            HUMMAN_SMPL_PARAMS_MANIFEST_SHA256,
        )
        if expected_geometry_source_terms is None
        else expected_geometry_source_terms
    )
    all_hashes = {}
    verified_geometry_sequence_ids = ()
    for input_name in ("geometry_source", "stage0", "train"):
        root = _canonical_absolute_path(roots[input_name], input_name + " release root")
        root_fd = _open_directory_no_follow(root, input_name + " release root")
        try:
            hashes, geometry_sequence_ids = _verify_one_input_release(
                input_name,
                checked_inputs[input_name],
                root_fd,
                expected_partition_sha256=selected_partition_sha256,
                expected_partition_list_sha256=selected_partition_list_sha256,
                expected_train_member_sha256=selected_train_member_sha256,
                expected_ordered_depth_list_sha256=(
                    selected_ordered_depth_list_sha256
                ),
                expected_geometry_support_sha256=selected_geometry_support_sha256,
                expected_geometry_smpl_manifest_sha256=(
                    selected_geometry_smpl_manifest_sha256
                ),
                expected_geometry_source_terms=selected_geometry_source_terms,
                expected_geometry_source_identity=checked_inputs["geometry_source"],
                expected_geometry_sequence_ids=verified_geometry_sequence_ids,
                test_stage0_npz_validator_fd=test_stage0_npz_validator_fd,
                test_frame_hashes_from_video_fd=test_frame_hashes_from_video_fd,
            )
        finally:
            os.close(root_fd)
        if input_name == "geometry_source":
            verified_geometry_sequence_ids = geometry_sequence_ids
        for digest in hashes:
            all_hashes.setdefault(digest, []).append(input_name)
    duplicated = {
        digest: names for digest, names in all_hashes.items() if len(names) > 1
    }
    if duplicated:
        raise CanaryContractError("input releases contain ambiguous cross-release content")
    return InputVerificationReceipt(expected_input_identity_sha256(checked_inputs))


def _validate_canary_config_core(
    config_or_canonical_bytes: Union[dict, bytes],
    *,
    code_root: Path,
    environment_root: Path,
    policy_root: Path,
    repo_root: Path,
    verify_inputs: Callable[
        [Mapping[str, Mapping[str, str]]], InputVerificationReceipt
    ],
) -> ValidatedCanaryIdentity:
    config, canonical = _parse_config(config_or_canonical_bytes)
    _exact_keys(config, _TOP_LEVEL_FIELDS, "canary config")
    if _contains_forbidden_term(config):
        raise CanaryContractError(
            "canary config keys and string values must not contain holdout, yellow, or evaluation"
        )
    if config["schema"] != CANARY_SCHEMA:
        raise CanaryContractError("canary config schema is not exact")

    _validate_code(config, root=code_root)
    _validate_environment(config, root=environment_root)
    _validate_warm_start(config)
    inputs = _validate_inputs(config)
    _validate_seeds_and_training(config)
    _validate_policy(
        config,
        policy_root=policy_root,
        repo_root=repo_root,
    )
    _verify_input_releases(inputs, verify_inputs)

    config_sha256 = hashlib.sha256(canonical).hexdigest()
    return ValidatedCanaryIdentity(
        canonical_config_bytes=canonical,
        config_sha256=config_sha256,
        run_id=RUN_ID_PREFIX + config_sha256[:20],
    )


def _production_verify_inputs(
    inputs: Mapping[str, Mapping[str, str]],
) -> InputVerificationReceipt:
    return _verify_input_releases_at_roots(inputs, PERCEPTION_INPUT_RELEASE_ROOTS)


def validate_canary_config(
    config_or_canonical_bytes: Union[dict, bytes],
) -> ValidatedCanaryIdentity:
    """Validate a production config and rehash its three completed H200 inputs."""
    return _validate_canary_config_core(
        config_or_canonical_bytes,
        code_root=_DEFAULT_CODE_ROOT,
        environment_root=_DEFAULT_ENVIRONMENT_ROOT,
        policy_root=_DEFAULT_REPO_ROOT,
        repo_root=_DEFAULT_REPO_ROOT,
        verify_inputs=_production_verify_inputs,
    )


def _validate_canary_config_for_test(
    config_or_canonical_bytes: Union[dict, bytes],
    *,
    code_root: Union[str, Path],
    environment_root: Union[str, Path],
    policy_root: Union[str, Path],
    repo_root: Union[str, Path],
    verify_inputs: Callable[
        [Mapping[str, Mapping[str, str]]], InputVerificationReceipt
    ],
) -> ValidatedCanaryIdentity:
    """Private deterministic harness with explicit non-production roots."""
    return _validate_canary_config_core(
        config_or_canonical_bytes,
        code_root=_canonical_absolute_path(code_root, "code_root"),
        environment_root=_canonical_absolute_path(
            environment_root, "environment_root"
        ),
        policy_root=_canonical_absolute_path(policy_root, "policy_root"),
        repo_root=_canonical_absolute_path(repo_root, "repo_root"),
        verify_inputs=verify_inputs,
    )


def _validated_output_identity(
    identity: ValidatedCanaryIdentity,
) -> ValidatedCanaryIdentity:
    if type(identity) is not ValidatedCanaryIdentity:
        raise CanaryContractError("output authority must be a validated canary identity")
    if type(identity.canonical_config_bytes) is not bytes:
        raise CanaryContractError("validated identity canonical bytes are invalid")
    _parse_config(identity.canonical_config_bytes)
    digest = hashlib.sha256(identity.canonical_config_bytes).hexdigest()
    if identity.config_sha256 != digest:
        raise CanaryContractError("validated identity config SHA-256 is forged")
    if identity.run_id != RUN_ID_PREFIX + digest[:20]:
        raise CanaryContractError("validated identity run_id is forged")
    return identity


def _output_path_for_identity(
    identity: ValidatedCanaryIdentity,
    *,
    host_role: str,
    output_root: Union[str, Path],
) -> tuple:
    if host_role != "h200":
        raise CanaryContractError("perception canary output is H200-only")
    checked = _validated_output_identity(identity)
    selected_root = _canonical_absolute_path(output_root, "output_root")
    if "current" in selected_root.parts:
        raise CanaryContractError("output paths must not contain current")
    return selected_root / checked.run_id, selected_root


def _require_output_absent(root_fd: int, run_id: str) -> None:
    _require_descriptor_primitives(os.stat)
    try:
        os.stat(run_id, dir_fd=root_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise CanaryContractError("output destination could not be inspected") from exc
    raise CanaryContractError("output destination already exists")


def _validate_output_identity_at_root(
    identity: ValidatedCanaryIdentity,
    *,
    host_role: str,
    output_root: Union[str, Path],
) -> Path:
    candidate, selected_root = _output_path_for_identity(
        identity,
        host_role=host_role,
        output_root=output_root,
    )
    root_fd = _open_directory_no_follow(selected_root, "output_root")
    try:
        _require_output_absent(root_fd, identity.run_id)
    finally:
        os.close(root_fd)
    return candidate


def validate_output_destination(
    config_or_canonical_bytes: Union[dict, bytes],
    *,
    host_role: str,
) -> Path:
    """Validate config and derive a side-effect-free, non-reserved output path."""
    identity = validate_canary_config(config_or_canonical_bytes)
    return _validate_output_identity_at_root(
        identity,
        host_role=host_role,
        output_root=_DEFAULT_OUTPUT_ROOT,
    )


def _validate_output_destination_for_test(
    identity: ValidatedCanaryIdentity,
    *,
    host_role: str,
    output_root: Union[str, Path],
) -> Path:
    """Private deterministic output-preflight harness."""
    return _validate_output_identity_at_root(
        identity,
        host_role=host_role,
        output_root=output_root,
    )


def _single_component(value: str, field_name: str) -> str:
    if (
        type(value) is not str
        or not value
        or value in (".", "..")
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise CanaryContractError(field_name + " must be one safe path component")
    return value


def _rename_noreplace_at(
    source_fd: int,
    source_name: str,
    destination_fd: int,
    destination_name: str,
) -> None:
    """Atomically publish one descriptor-relative entry without replacement."""
    source = _single_component(source_name, "staging name")
    destination = _single_component(destination_name, "output name")
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        operation = getattr(libc, "renameat2", None)
        if operation is None:
            raise CanaryContractError(
                "renameat2 is required for atomic no-replace output publication"
            )
        operation.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        operation.restype = ctypes.c_int
        result = operation(
            source_fd,
            os.fsencode(source),
            destination_fd,
            os.fsencode(destination),
            1,
        )
    elif sys.platform == "darwin":
        operation = getattr(libc, "renameatx_np", None)
        if operation is None:
            raise CanaryContractError(
                "renameatx_np is required for atomic no-replace output publication"
            )
        operation.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        operation.restype = ctypes.c_int
        result = operation(
            source_fd,
            os.fsencode(source),
            destination_fd,
            os.fsencode(destination),
            0x00000004,
        )
    else:
        raise CanaryContractError(
            "atomic descriptor-relative output publication is unsupported on "
            + sys.platform
        )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in (errno.EEXIST, errno.ENOTEMPTY):
        raise CanaryContractError("output destination is already reserved")
    raise CanaryContractError("atomic output publication failed") from OSError(
        error, os.strerror(error), destination
    )


def _mkdir_output_child_at(root_fd: int, child_name: str) -> tuple:
    _require_descriptor_primitives(os.mkdir)
    name = _single_component(child_name, "staging name")
    os.mkdir(name, 0o700, dir_fd=root_fd)
    child_fd = None
    try:
        child_fd = os.open(name, _directory_flags(), dir_fd=root_fd)
        os.set_inheritable(child_fd, False)
        opened = os.fstat(child_fd)
        named = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        opened_identity = _stat_identity(opened, stable=False)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or opened_identity != _stat_identity(named, stable=False)
            or os.listdir(child_fd)
        ):
            raise CanaryContractError("staging output identity is not owned")
        result = child_fd, opened_identity
        child_fd = None
        return result
    finally:
        if child_fd is not None:
            os.close(child_fd)


def _reservation_root_pinned(output_root: Path, root_fd: int) -> None:
    """Private deterministic race-test hook; production intentionally does nothing."""
    del output_root, root_fd


def _output_child_created(root_fd: int, staging_name: str) -> None:
    """Private deterministic staging-acquisition race-test hook."""
    del root_fd, staging_name


def _before_output_publication(
    root_fd: int, staging_name: str, run_id: str
) -> None:
    """Private deterministic pre-publication race-test hook."""
    del root_fd, staging_name, run_id


def _cleanup_owned_empty_child(
    root_fd: int,
    entry_name: str,
    child_fd: Optional[int],
    created_identity: Optional[tuple],
) -> None:
    if child_fd is None or created_identity is None:
        return
    try:
        opened = os.fstat(child_fd)
        named = os.stat(entry_name, dir_fd=root_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or _stat_identity(opened, stable=False)
            != _stat_identity(named, stable=False)
            or _stat_identity(named, stable=False) != created_identity
            or os.listdir(child_fd)
        ):
            return
        _require_descriptor_primitives(os.rmdir)
        os.rmdir(entry_name, dir_fd=root_fd)
    except OSError:
        return


def _attempt_cleanup_owned_empty_child(
    root_fd: int,
    entry_name: str,
    child_fd: Optional[int],
    created_identity: Optional[tuple],
) -> None:
    try:
        _cleanup_owned_empty_child(
            root_fd, entry_name, child_fd, created_identity
        )
    except Exception:
        # Setup cleanup is best-effort and must never mask the primary failure.
        return


def _output_root_stat(root_fd: int) -> os.stat_result:
    """Private seam for deterministic output-root authority tests."""
    return os.fstat(root_fd)


def _effective_uid() -> int:
    operation = getattr(os, "geteuid", None)
    if not callable(operation):
        raise CanaryContractError(
            "output root owner-control requires an effective UID"
        )
    try:
        effective_uid = operation()
    except Exception as exc:
        raise CanaryContractError(
            "output root owner-control requires an effective UID"
        ) from exc
    if type(effective_uid) is not int or effective_uid < 0:
        raise CanaryContractError(
            "output root owner-control requires an effective UID"
        )
    return effective_uid


def _capture_output_root_authority(root_fd: int) -> _OutputRootAuthority:
    try:
        info = _output_root_stat(root_fd)
        effective_uid = _effective_uid()
        mode = stat.S_IMODE(info.st_mode)
        owner_uid = info.st_uid
    except CanaryContractError:
        raise
    except Exception as exc:
        raise CanaryContractError(
            "output root must be an owner-controlled directory"
        ) from exc
    if (
        not stat.S_ISDIR(info.st_mode)
        or owner_uid != effective_uid
        or mode & (stat.S_IWGRP | stat.S_IWOTH)
    ):
        raise CanaryContractError(
            "output root must be an owner-controlled directory"
        )
    return _OutputRootAuthority(
        identity=_stat_identity(info, stable=False),
        owner_uid=owner_uid,
        mode=mode,
    )


def _require_root_authority(
    output_root: Path,
    root_fd: int,
    expected: _OutputRootAuthority,
) -> None:
    try:
        pinned = _capture_output_root_authority(root_fd)
    except CanaryContractError as exc:
        raise _OutputRootAuthorityChanged(
            "output root authority changed"
        ) from exc
    if pinned.identity != expected.identity:
        raise _OutputRootAuthorityChanged("output root binding changed")
    if pinned.owner_uid != expected.owner_uid or pinned.mode != expected.mode:
        raise _OutputRootAuthorityChanged("output root authority changed")

    rebound_fd = None
    try:
        rebound_fd = _open_directory_no_follow(output_root, "canonical output_root")
        rebound = _capture_output_root_authority(rebound_fd)
    except CanaryContractError as exc:
        raise _OutputRootAuthorityChanged("output root binding changed") from exc
    finally:
        if rebound_fd is not None:
            os.close(rebound_fd)
    if rebound.identity != expected.identity:
        raise _OutputRootAuthorityChanged("output root binding changed")
    if rebound.owner_uid != expected.owner_uid or rebound.mode != expected.mode:
        raise _OutputRootAuthorityChanged("output root authority changed")


def _require_named_identity(
    root_fd: int,
    entry_name: str,
    child_fd: int,
    expected_identity: tuple,
    field_name: str,
) -> None:
    opened = os.fstat(child_fd)
    named = os.stat(entry_name, dir_fd=root_fd, follow_symlinks=False)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or _stat_identity(opened, stable=False) != expected_identity
        or _stat_identity(named, stable=False) != expected_identity
    ):
        raise CanaryContractError(field_name + " identity changed")


def _reserve_output_identity_at_root(
    identity: ValidatedCanaryIdentity,
    *,
    host_role: str,
    output_root: Union[str, Path],
) -> ReservedOutputDirectory:
    checked = _validated_output_identity(identity)
    candidate, selected_root = _output_path_for_identity(
        checked,
        host_role=host_role,
        output_root=output_root,
    )
    _require_descriptor_primitives(os.mkdir, os.open, os.stat, os.rmdir)
    root_fd = _open_directory_no_follow(selected_root, "output_root")
    child_fd = None
    created = False
    created_identity = None
    staging_name = ".canary-reservation-" + uuid.uuid4().hex
    published = False
    try:
        root_authority = _capture_output_root_authority(root_fd)
        _reservation_root_pinned(selected_root, root_fd)
        try:
            child_fd, created_identity = _mkdir_output_child_at(
                root_fd, staging_name
            )
        except FileExistsError as exc:
            raise CanaryContractError("staging output name collision") from exc
        created = True
        _require_named_identity(
            root_fd,
            staging_name,
            child_fd,
            created_identity,
            "staging output",
        )
        _output_child_created(root_fd, staging_name)
        _require_named_identity(
            root_fd,
            staging_name,
            child_fd,
            created_identity,
            "staging output",
        )
        if os.listdir(child_fd):
            raise CanaryContractError("staging output must be empty before publication")
        _before_output_publication(root_fd, staging_name, checked.run_id)
        _require_named_identity(
            root_fd,
            staging_name,
            child_fd,
            created_identity,
            "staging output",
        )
        _require_root_authority(selected_root, root_fd, root_authority)
        _rename_noreplace_at(
            root_fd, staging_name, root_fd, checked.run_id
        )
        published = True
        _require_root_authority(selected_root, root_fd, root_authority)
        _require_named_identity(
            root_fd,
            checked.run_id,
            child_fd,
            created_identity,
            "published output",
        )
        os.fsync(root_fd)
        opened = os.fstat(child_fd)
        reservation = ReservedOutputDirectory(
            path=candidate,
            _directory_fd=child_fd,
            _root_fd=root_fd,
            st_dev=opened.st_dev,
            st_ino=opened.st_ino,
        )
        child_fd = None
        root_fd = None
        return reservation
    except CanaryContractError as exc:
        if created and not isinstance(exc, _OutputRootAuthorityChanged):
            _attempt_cleanup_owned_empty_child(
                root_fd,
                checked.run_id if published else staging_name,
                child_fd,
                created_identity,
            )
        raise
    except OSError as exc:
        if created:
            _attempt_cleanup_owned_empty_child(
                root_fd,
                checked.run_id if published else staging_name,
                child_fd,
                created_identity,
            )
        raise CanaryContractError("output destination reservation failed") from exc
    finally:
        if child_fd is not None:
            os.close(child_fd)
        if root_fd is not None:
            os.close(root_fd)


def reserve_output_destination(
    config_or_canonical_bytes: Union[dict, bytes],
    *,
    host_role: str,
) -> ReservedOutputDirectory:
    """Validate config, derive its run, and atomically publish an owned output."""
    identity = validate_canary_config(config_or_canonical_bytes)
    return _reserve_output_identity_at_root(
        identity,
        host_role=host_role,
        output_root=_DEFAULT_OUTPUT_ROOT,
    )


def _reserve_output_destination_for_test(
    identity: ValidatedCanaryIdentity,
    *,
    host_role: str,
    output_root: Union[str, Path],
) -> ReservedOutputDirectory:
    """Private deterministic reservation harness with an explicit root."""
    return _reserve_output_identity_at_root(
        identity,
        host_role=host_role,
        output_root=output_root,
    )
