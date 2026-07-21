from dataclasses import FrozenInstanceError
from copy import deepcopy
import hashlib
import io
import inspect
import json
import os
from pathlib import Path
import struct
from types import SimpleNamespace
from typing import Dict, List, Tuple
import zipfile

import pytest
import numpy as np

from scripts.deployment import artifact_contract

from ausc.training import perception_canary_contract as contract
from ausc.training.perception_canary_contract import (
    InputVerificationReceipt,
    ReservedOutputDirectory,
    CanaryContractError,
    ValidatedCanaryIdentity,
    _reserve_output_destination_for_test,
    _validate_canary_config_for_test,
    _validate_output_destination_for_test,
    expected_input_identity_sha256,
    reserve_output_destination,
    validate_canary_config,
    validate_output_destination,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
POLICY_RELATIVE_PATH = (
    "auscultation/configs/perception/visible-stability-mask-policy-v1.json"
)
POLICY_PATH = REPO_ROOT / POLICY_RELATIVE_PATH
URDF_RELATIVE_PATH = "auscultation/ausc/perception/robot/openarm_v10.urdf"

POINT_ORDER = ("R1", "L1", "R2", "L2", "R3", "L3", "R4", "L4", "R5", "L5")


def _valid_geometry_sequence_ids() -> tuple:
    return tuple(
        sorted(
            "p%06d_a%06d" % (index % 340, index // 340)
            for index in range(907)
        )
    )


def _coordinate_metadata_object() -> dict:
    return {
        "decoded_frame_hash_version": "wam_bgr8_frame_sha256_v1",
        "depth_unit": "metres",
        "garment": "dark",
        "height_px": 1080,
        "landmark_order": list(POINT_ORDER),
        "pixel_axis_order": "xy",
        "pixel_index_base": 0,
        "pixel_origin": "top_left",
        "rgb_depth_registration": "registered",
        "schema": "wam_perception_session2_coordinate_metadata_v1",
        "session_id": "orbbec_session2",
        "subject_pseudonym": "same-subject-001",
        "width_px": 1920,
    }


def _semantic_member(path: Path, relative: str, role: str, origin: str) -> dict:
    payload = path / relative
    return {
        "content_origin": origin,
        "path": relative,
        "role": role,
        "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
        "size": payload.stat().st_size,
    }


def _write_input_release(
    root: Path,
    *,
    artifact_kind: str,
    source_role: str,
    session_ids: list,
    files: list,
    contract_fields: dict,
    source_attestations: list = None,
    semantic_transform=None,
    semantic_bytes_override: bytes = None,
    semantic_encoder=None,
    target_role: str = "h200",
) -> tuple:
    source = root.parent / (artifact_kind + "-source")
    source.mkdir(parents=True)
    members = []
    for relative, body, role, origin in files:
        payload = source / relative
        payload.parent.mkdir(parents=True, exist_ok=True)
        payload.write_bytes(body)
        members.append(_semantic_member(source, relative, role, origin))
    semantic = {
        "artifact_kind": artifact_kind,
        "contract": contract_fields,
        "members": sorted(members, key=lambda row: row["path"]),
        "resolvable_dependencies": [],
        "schema": contract.PERCEPTION_INPUT_SCHEMA,
        "self_contained": True,
        "session_ids": session_ids,
        "source_attestations": source_attestations or [],
        "source_role": source_role,
    }
    if semantic_transform is not None:
        semantic_transform(semantic)
    if semantic_bytes_override is not None:
        semantic_bytes = semantic_bytes_override
    elif semantic_encoder is not None:
        semantic_bytes = semantic_encoder(semantic)
    else:
        semantic_bytes = _canonical_json_bytes(semantic)
    (source / contract.PERCEPTION_INPUT_NAME).write_bytes(semantic_bytes)
    paths = [relative for relative, *_ in files] + [contract.PERCEPTION_INPUT_NAME]
    envelope = artifact_contract.build_manifest(
        source,
        paths,
        artifact_kind=artifact_kind,
        source_role=source_role,
    )
    release_root = root / envelope["release_id"]
    artifact_contract.write_release(
        source, envelope, release_root, target_role=target_role
    )
    manifest_bytes = (release_root / artifact_contract.MANIFEST_NAME).read_bytes()
    return release_root, {
        "payload_sha256": envelope["payload_sha256"],
        "release_id": envelope["release_id"],
        "release_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "semantic_manifest_sha256": hashlib.sha256(semantic_bytes).hexdigest(),
    }


def _valid_stage0_payload() -> tuple:
    sample_files = []
    samples = []
    for sequence_id in _valid_geometry_sequence_ids():
        subject_id = sequence_id.split("_a", 1)[0]
        split = "train" if int(subject_id[1:]) < 300 else "validation"
        for view_index in range(4):
            sample_id = "%s_v%d" % (sequence_id, view_index)
            sample_path = "samples/%s.npz" % sample_id
            body = ("stage0-npz-%s" % sample_id).encode("ascii")
            digest = hashlib.sha256(body).hexdigest()
            sample_files.append(
                (sample_path, body, "generated_sample", "generated")
            )
            samples.append(
                {
                    "sample_id": sample_id,
                    "sample_path": sample_path,
                    "sample_sha256": digest,
                    "sequence_id": sequence_id,
                    "split": split,
                    "subject_id": subject_id,
                    "view_index": view_index,
                }
            )
    sample_ledger = {
        "samples": samples,
        "schema": contract.STAGE0_SAMPLE_LEDGER_SCHEMA,
    }
    split_ledger = {
        "schema": contract.STAGE0_SUBJECT_SPLIT_SCHEMA,
        "split_strategy": "subject_grouped_fixed_v1",
        "train_subject_ids": ["p%06d" % index for index in range(300)],
        "validation_subject_ids": ["p%06d" % index for index in range(300, 340)],
    }
    return (
        sample_files,
        _canonical_json_bytes(sample_ledger),
        _canonical_json_bytes(split_ledger),
    )


def _valid_source_frame_index_bytes() -> bytes:
    rows = ["idx,t,cam_index,has_depth,depth_scale"]
    rows.extend(
        "%d,%.9f,%d,1,1.0" % (index, index / 30.0, index)
        for index in range(602)
    )
    return ("\n".join(rows) + "\n").encode("utf-8")


def _valid_stage0_npz_arrays() -> dict:
    height = width = 518
    yy, xx = np.mgrid[0:height, 0:width]
    depth_m = np.zeros((height, width), dtype=np.float32)
    depth_m[40:480, 40:480] = (
        0.9 + xx[40:480, 40:480] * 0.0003 + yy[40:480, 40:480] * 0.0002
    ).astype(np.float32)
    normal_u8 = np.full((height, width, 3), 127, dtype=np.uint8)
    normal_x = (xx[40:480, 40:480] - 259.0) * 0.0005
    normal_y = (yy[40:480, 40:480] - 259.0) * 0.0003
    normal_z = -np.sqrt(1.0 - normal_x**2 - normal_y**2)
    encoded_normals = (
        np.stack((normal_x, normal_y, normal_z), axis=-1) * 127.0 + 127.0
    ).astype(np.uint8)
    normal_u8[40:480, 40:480] = encoded_normals
    points_px = np.asarray(
        [
            [170, 110],
            [348, 110],
            [165, 180],
            [353, 180],
            [160, 250],
            [358, 250],
            [168, 320],
            [350, 320],
            [180, 390],
            [338, 390],
        ],
        dtype=np.float32,
    )
    return {
        "depth_m": depth_m,
        "intrinsics_px": np.asarray(
            [[560.0, 0.0, 259.0], [0.0, 560.0, 259.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        "normal_u8": normal_u8,
        "point_order": np.asarray(POINT_ORDER, dtype="<U2"),
        "points_px": points_px,
        "sample_id": np.asarray("p000000_a000000_v0", dtype="<U18"),
        "sequence_id": np.asarray("p000000_a000000", dtype="<U15"),
        "source_frame_index": np.asarray(12, dtype=np.int32),
        "subject_id": np.asarray("p000000", dtype="<U7"),
        "view_index": np.asarray(0, dtype=np.int16),
        "visibility": np.ones(10, dtype=np.bool_),
    }


def _npz_bytes(arrays: dict) -> bytes:
    output = io.BytesIO()
    np.savez(output, **arrays)
    return output.getvalue()


def _compressed_npz_bytes(arrays: dict) -> bytes:
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    return output.getvalue()


def _replace_npz_member(payload: bytes, member_name: str, replacement: bytes) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(payload), mode="r") as source:
        with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as target:
            for info in source.infolist():
                target.writestr(
                    info.filename,
                    replacement if info.filename == member_name else source.read(info),
                )
    return output.getvalue()


def _npy_with_oversized_v1_header(array: np.ndarray) -> bytes:
    header_text = repr(
        {
            "descr": np.lib.format.dtype_to_descr(array.dtype),
            "fortran_order": False,
            "shape": array.shape,
        }
    ).encode("latin1")
    header_size = 630
    assert len(header_text) + 1 < header_size
    header = header_text + b" " * (header_size - len(header_text) - 1) + b"\n"
    return (
        np.lib.format.magic(1, 0)
        + struct.pack("<H", header_size)
        + header
        + array.tobytes(order="C")
    )


def _validate_stage0_npz(payload: bytes) -> None:
    contract.validate_stage0_npz_bytes(
        payload,
        expected_sample_id="p000000_a000000_v0",
        expected_sequence_id="p000000_a000000",
        expected_subject_id="p000000",
        expected_view_index=0,
    )


def _geometry_source_terms_object(
    support_sha256: dict,
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


def _stage0_provenance_object(geometry_identity: dict) -> dict:
    return {
        "NSEQ": 907,
        "NVIEW": 4,
        "environment": {
            "identity_sha256": (
                "5d2be01285ba34687f257479429f8c43f2046fbb05ae2fa43ecf82f40c5d65d7"
            ),
            "release_id": "wam-5d2be01285ba34687f25",
        },
        "geometry_source": dict(geometry_identity),
        "mask_policy_sha256": (
            "ad5a867cc62aa3164eb6f69123d72057c005acd0d6af8dd090e38ec7f5a5525b"
        ),
        "preprocessing": {
            "identifier": "wam_robot_mask_before_depth_to_normals_v1"
        },
        "sample_count": 3628,
        "sample_format": contract.STAGE0_SAMPLE_FORMAT,
        "schema": "wam_perception_stage0_provenance_v1",
        "seed": 0,
    }


def _write_valid_input_closure(tmp_path: Path) -> tuple:
    roots = {
        "geometry_source": tmp_path / "geometry" / "releases",
        "stage0": tmp_path / "stage0" / "releases",
        "train": tmp_path / "train" / "releases",
    }
    for root in roots.values():
        root.mkdir(parents=True)
    geometry_support_files = [
        ("smpl/SMPL_NEUTRAL.pkl", b"neutral", "smpl_neutral", "smpl_neutral"),
        ("densepose/UV_Processed.mat", b"uv", "densepose_uv", "densepose_uv"),
        ("licenses/HUMMAN_LICENSE.txt", b"humman-license", "license", "license"),
        (
            "licenses/SMPL_MODEL_LICENSE.html",
            b"smpl-license",
            "license",
            "license",
        ),
        (
            "licenses/DENSEPOSE_SOURCE_LICENSE.txt",
            b"densepose-license",
            "license",
            "license",
        ),
    ]
    geometry_smpl_files = [
        (
            "humman/smpl_params/%s/smpl_params.npz" % sequence_id,
            ("smpl-%s" % sequence_id).encode("ascii"),
            "smpl_params",
            "humman_smpl_params",
        )
        for sequence_id in _valid_geometry_sequence_ids()
    ]
    geometry_support_sha256 = {
        path: hashlib.sha256(body).hexdigest()
        for path, body, _, _ in geometry_support_files
    }
    geometry_smpl_manifest = "".join(
        "%s  %s/smpl_params.npz\n"
        % (
            hashlib.sha256(body).hexdigest(),
            Path(path).parts[2],
        )
        for path, body, _, _ in sorted(geometry_smpl_files)
    ).encode("ascii")
    geometry_source_terms = _geometry_source_terms_object(
        geometry_support_sha256,
        hashlib.sha256(geometry_smpl_manifest).hexdigest(),
    )
    geometry_files = geometry_support_files + [
        (
            "provenance/SOURCE_TERMS.json",
            _canonical_json_bytes(geometry_source_terms),
            "provenance",
            "provenance",
        ),
    ] + geometry_smpl_files
    _, geometry_identity = _write_input_release(
        roots["geometry_source"],
        artifact_kind="perception_geometry_source",
        source_role="mac",
        session_ids=[],
        files=geometry_files,
        contract_fields={
            "densepose_uv_count": 1,
            "license_count": 3,
            "provenance_count": 1,
            "raw_rgb_count": 0,
            "smpl_neutral_count": 1,
            "smpl_params_count": 907,
        },
    )
    stage0_sample_files, sample_ledger_bytes, split_ledger_bytes = (
        _valid_stage0_payload()
    )
    _, stage0_identity = _write_input_release(
        roots["stage0"],
        artifact_kind="perception_stage0",
        source_role="h200",
        session_ids=[],
        files=[
            ("ledgers/sample_ids.json", sample_ledger_bytes, "sample_ledger", "ledger"),
            ("ledgers/subject_split.json", split_ledger_bytes, "subject_split_ledger", "ledger"),
            (
                "PROVENANCE.json",
                _canonical_json_bytes(_stage0_provenance_object(geometry_identity)),
                "provenance",
                "provenance",
            ),
        ] + stage0_sample_files,
        contract_fields={
            "provenance_path": "PROVENANCE.json",
            "sample_count": 3628,
            "sample_format": contract.STAGE0_SAMPLE_FORMAT,
            "sample_ledger_path": "ledgers/sample_ids.json",
            "split_strategy": "subject_grouped_fixed_v1",
            "subject_split_ledger_path": "ledgers/subject_split.json",
        },
    )
    train_depth_files = [
        (
            "depth/depth_%06d.npy" % index,
            ("depth-%06d" % index).encode("ascii"),
            "metric_depth",
            "orbbec_session2",
        )
        for index in range(602)
    ]
    ordered_depth_fingerprints = [
        hashlib.sha256(("depth-%06d" % index).encode("ascii")).hexdigest()
        for index in range(602)
    ]
    decoded_fingerprints = [
        hashlib.sha256(("decoded-dark-%06d" % index).encode("ascii")).hexdigest()
        for index in range(602)
    ]
    frame_ledger = {
        "frames": [
            {
                "depth_path": "depth/depth_%06d.npy" % index,
                "depth_sha256": ordered_depth_fingerprints[index],
                "frame_index": index,
                "rgb_sha256": decoded_fingerprints[index],
            }
            for index in range(602)
        ],
        "schema": contract.SESSION2_FRAME_LEDGER_SCHEMA,
    }
    frame_ledger_bytes = _canonical_json_bytes(frame_ledger)
    source_frames_bytes = _valid_source_frame_index_bytes()
    shared_intrinsics_sha256 = hashlib.sha256(b"intrinsics").hexdigest()
    selected_session2_hashes = sorted(
        {
            hashlib.sha256(b"dark-rgb").hexdigest(),
            shared_intrinsics_sha256,
            hashlib.sha256(source_frames_bytes).hexdigest(),
            *(
                hashlib.sha256(
                    ("depth-%06d" % index).encode("ascii")
                ).hexdigest()
                for index in range(602)
            ),
        }
    )
    session3_denied_hashes = sorted(
        {
            shared_intrinsics_sha256,
            *(
                hashlib.sha256(
                    ("session3-source-%03d" % index).encode("ascii")
                ).hexdigest()
                for index in range(607)
            ),
        }
    )
    session3_decoded_hashes = [
        hashlib.sha256(("session3-bgr8-%06d" % index).encode("ascii")).hexdigest()
        for index in range(602)
    ]
    content_partition = {
        "decoded_frame_hash_version": contract.DECODED_FRAME_HASH_VERSION,
        "schema": contract.MIXED_EVAL_CONTENT_PARTITION_SCHEMA,
        "session2_decoded_frame_sha256": decoded_fingerprints,
        "session2_selected_sha256": selected_session2_hashes,
        "session3_decoded_frame_sha256": session3_decoded_hashes,
        "session3_denied_sha256": session3_denied_hashes,
        "shared_intrinsics_sha256": [shared_intrinsics_sha256],
        "source_manifest_sha256": contract.HISTORICAL_MIXED_EVAL_MANIFEST_SHA256,
        "source_payload_sha256": contract.HISTORICAL_MIXED_EVAL_PAYLOAD_SHA256,
        "source_release_id": contract.HISTORICAL_MIXED_EVAL_RELEASE_ID,
    }
    _, train_identity = _write_input_release(
        roots["train"],
        artifact_kind="perception_session2_train",
        source_role="h200",
        session_ids=["orbbec_session2"],
        files=[
            ("video/session2.mp4", b"dark-rgb", "rgb_video", "orbbec_session2"),
            ("camera/intrinsics.json", b"intrinsics", "intrinsics", "shared_intrinsics"),
            ("labels/dark.json", b"dark-labels", "labels", "session2_dark_annotation"),
            ("ledgers/frames.json", frame_ledger_bytes, "frame_ledger", "builder_metadata"),
            (
                "metadata/coordinates.json",
                _canonical_json_bytes(_coordinate_metadata_object()),
                "coordinate_metadata",
                "builder_metadata",
            ),
            ("source/frames.csv", source_frames_bytes, "source_frame_index", "orbbec_session2"),
        ] + train_depth_files,
        contract_fields={
            "decoded_frame_fingerprints": decoded_fingerprints,
            "depth_frame_count": 602,
            "frame_count": 602,
            "frame_ledger_path": "ledgers/frames.json",
            "garment": "dark",
            "intrinsics_path": "camera/intrinsics.json",
            "labels_path": "labels/dark.json",
            "ordered_depth_fingerprints": ordered_depth_fingerprints,
            "rgb_video_path": "video/session2.mp4",
            "source_frames_path": "source/frames.csv",
        },
        source_attestations=[
            {
                "artifact_kind": "perception_eval",
                "content_partition": content_partition,
                "payload_sha256": contract.HISTORICAL_MIXED_EVAL_PAYLOAD_SHA256,
                "release_id": contract.HISTORICAL_MIXED_EVAL_RELEASE_ID,
                "release_manifest_sha256": contract.HISTORICAL_MIXED_EVAL_MANIFEST_SHA256,
            }
        ],
    )
    return {
        "geometry_source": geometry_identity,
        "stage0": stage0_identity,
        "train": train_identity,
    }, roots


@pytest.fixture(scope="session")
def verified_input_closure(tmp_path_factory):
    return _write_valid_input_closure(
        tmp_path_factory.mktemp("perception-input-closure")
    )


def _write_stage0_release(
    tmp_path: Path,
    *,
    artifact_kind: str = "perception_stage0",
    source_role: str = "h200",
    target_role: str = "h200",
    semantic_transform=None,
    semantic_bytes_override: bytes = None,
    semantic_encoder=None,
) -> tuple:
    root = tmp_path / "stage0-invalid" / "releases"
    root.mkdir(parents=True)
    release, identity = _write_input_release(
        root,
        artifact_kind=artifact_kind,
        source_role=source_role,
        target_role=target_role,
        session_ids=[],
        files=[
            ("ledgers/sample_ids.json", b"[]\n", "sample_ledger", "ledger"),
            ("ledgers/subject_split.json", b"{}\n", "subject_split_ledger", "ledger"),
            ("PROVENANCE.json", b"stage0", "provenance", "provenance"),
        ],
        contract_fields={
            "provenance_path": "PROVENANCE.json",
            "sample_count": 3628,
            "sample_format": contract.STAGE0_SAMPLE_FORMAT,
            "sample_ledger_path": "ledgers/sample_ids.json",
            "split_strategy": "subject_grouped_fixed_v1",
            "subject_split_ledger_path": "ledgers/subject_split.json",
        },
        semantic_transform=semantic_transform,
        semantic_bytes_override=semantic_bytes_override,
        semantic_encoder=semantic_encoder,
    )
    return release, identity, root


def _write_input_member_variant(
    tmp_path: Path,
    *,
    inputs: dict,
    roots: dict,
    input_name: str,
    member_path: str,
    member_bytes: bytes,
) -> tuple:
    original_release = roots[input_name] / inputs[input_name]["release_id"]
    semantic, _ = _semantic_and_outer_rows(original_release)
    root = tmp_path / (input_name + "-variant") / "releases"
    root.mkdir(parents=True)
    _, identity = _write_input_release(
        root,
        artifact_kind=semantic["artifact_kind"],
        source_role=semantic["source_role"],
        session_ids=semantic["session_ids"],
        files=[
            (
                member["path"],
                (
                    member_bytes
                    if member["path"] == member_path
                    else (original_release / member["path"]).read_bytes()
                ),
                member["role"],
                member["content_origin"],
            )
            for member in semantic["members"]
        ],
        contract_fields=deepcopy(semantic["contract"]),
        source_attestations=deepcopy(semantic["source_attestations"]),
    )
    candidate_inputs = deepcopy(inputs)
    candidate_roots = dict(roots)
    candidate_inputs[input_name] = identity
    candidate_roots[input_name] = root
    return candidate_inputs, candidate_roots


def _semantic_and_outer_rows(release: Path) -> tuple:
    semantic = json.loads((release / contract.PERCEPTION_INPUT_NAME).read_text())
    envelope = json.loads(
        (release / artifact_contract.MANIFEST_NAME).read_text()
    )
    return semantic, envelope["payload"]["files"]


def _stage0_documents(release: Path) -> tuple:
    semantic, _ = _semantic_and_outer_rows(release)
    sample_ledger = json.loads(
        (release / semantic["contract"]["sample_ledger_path"]).read_text()
    )
    split_ledger = json.loads(
        (release / semantic["contract"]["subject_split_ledger_path"]).read_text()
    )
    return semantic, sample_ledger, split_ledger


def _session2_documents(release: Path) -> tuple:
    semantic, outer_rows = _semantic_and_outer_rows(release)
    frame_ledger = json.loads(
        (release / semantic["contract"]["frame_ledger_path"]).read_text()
    )
    return semantic, outer_rows, frame_ledger


def _partition_sha256_from_semantic(semantic: dict) -> str:
    partition = semantic["source_attestations"][0]["content_partition"]
    return hashlib.sha256(_canonical_json_bytes(partition)).hexdigest()


def _partition_list_sha256_from_semantic(semantic: dict) -> dict:
    partition = semantic["source_attestations"][0]["content_partition"]
    return {
        field: hashlib.sha256(_canonical_json_bytes(partition[field])).hexdigest()
        for field in contract._PRODUCTION_PARTITION_LIST_SHA256
    }


def _train_member_sha256_from_semantic(semantic: dict) -> dict:
    members_by_role = {member["role"]: member for member in semantic["members"]}
    return {
        role: members_by_role[role]["sha256"]
        for role in contract._PRODUCTION_TRAIN_MEMBER_SHA256
    }


def _ordered_depth_list_sha256_from_semantic(semantic: dict) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(semantic["contract"]["ordered_depth_fingerprints"])
    ).hexdigest()


def _geometry_support_sha256_from_semantic(semantic: dict) -> dict:
    pinned_paths = {
        "densepose/UV_Processed.mat",
        "licenses/DENSEPOSE_SOURCE_LICENSE.txt",
        "licenses/HUMMAN_LICENSE.txt",
        "licenses/SMPL_MODEL_LICENSE.html",
        "smpl/SMPL_NEUTRAL.pkl",
    }
    return {
        member["path"]: member["sha256"]
        for member in semantic["members"]
        if member["path"] in pinned_paths
    }


def _geometry_smpl_manifest_sha256_from_semantic(semantic: dict) -> str:
    lines = []
    for member in sorted(
        (
            member
            for member in semantic["members"]
            if member["role"] == "smpl_params"
        ),
        key=lambda member: member["path"],
    ):
        sequence_id = Path(member["path"]).parts[2]
        lines.append(
            "%s  %s/smpl_params.npz\n" % (member["sha256"], sequence_id)
        )
    return hashlib.sha256("".join(lines).encode("ascii")).hexdigest()


def _validate_fixture_stage0_sample_fd(descriptor: int, ledger_row: dict) -> None:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    assert b"".join(chunks) == (
        "stage0-npz-%s" % ledger_row["sample_id"]
    ).encode("ascii")


def _verify_input_closure(inputs: dict, roots: dict) -> InputVerificationReceipt:
    release = roots["train"] / inputs["train"]["release_id"]
    semantic, _ = _semantic_and_outer_rows(release)
    geometry_release = (
        roots["geometry_source"] / inputs["geometry_source"]["release_id"]
    )
    geometry_semantic, _ = _semantic_and_outer_rows(geometry_release)
    decoded_hashes = semantic["source_attestations"][0]["content_partition"][
        "session2_decoded_frame_sha256"
    ]

    def frame_hashes_from_video_fd(descriptor: int) -> list:
        os.lseek(descriptor, 0, os.SEEK_SET)
        while os.read(descriptor, 1024 * 1024):
            pass
        return decoded_hashes

    return contract._verify_input_releases_at_roots(
        inputs,
        roots,
        expected_partition_sha256=_partition_sha256_from_semantic(semantic),
        expected_partition_list_sha256=_partition_list_sha256_from_semantic(semantic),
        expected_train_member_sha256=_train_member_sha256_from_semantic(semantic),
        expected_ordered_depth_list_sha256=(
            _ordered_depth_list_sha256_from_semantic(semantic)
        ),
        expected_geometry_support_sha256=(
            _geometry_support_sha256_from_semantic(geometry_semantic)
        ),
        expected_geometry_smpl_manifest_sha256=(
            _geometry_smpl_manifest_sha256_from_semantic(geometry_semantic)
        ),
        expected_geometry_source_terms=_geometry_source_terms_object(
            _geometry_support_sha256_from_semantic(geometry_semantic),
            _geometry_smpl_manifest_sha256_from_semantic(geometry_semantic),
        ),
        test_stage0_npz_validator_fd=_validate_fixture_stage0_sample_fd,
        test_frame_hashes_from_video_fd=frame_hashes_from_video_fd,
    )


def _validate_semantic_for_test(
    semantic: dict,
    outer_rows: list,
    *,
    input_name: str,
    current_identity: dict,
    expected_partition_sha256: str = None,
    expected_partition_list_sha256: dict = None,
    expected_train_member_sha256: dict = None,
    expected_ordered_depth_list_sha256: str = None,
    expected_geometry_support_sha256: dict = None,
    expected_geometry_smpl_manifest_sha256: str = None,
    expected_label_sha256: str = None,
):
    selected_geometry_support = expected_geometry_support_sha256
    selected_geometry_manifest = expected_geometry_smpl_manifest_sha256
    if input_name == "geometry_source":
        if selected_geometry_support is None:
            selected_geometry_support = _geometry_support_sha256_from_semantic(
                semantic
            )
        if selected_geometry_manifest is None:
            selected_geometry_manifest = (
                _geometry_smpl_manifest_sha256_from_semantic(semantic)
            )
    else:
        selected_geometry_support = contract._PRODUCTION_GEOMETRY_SUPPORT_SHA256
        selected_geometry_manifest = contract.HUMMAN_SMPL_PARAMS_MANIFEST_SHA256
    selected = expected_partition_sha256
    if selected is None:
        selected = (
            _partition_sha256_from_semantic(semantic)
            if input_name == "train"
            else contract.HISTORICAL_MIXED_EVAL_PARTITION_SHA256
        )
    selected_list_digests = expected_partition_list_sha256
    selected_member_digests = expected_train_member_sha256
    selected_depth_list_digest = expected_ordered_depth_list_sha256
    if input_name == "train":
        if selected_list_digests is None:
            try:
                selected_list_digests = _partition_list_sha256_from_semantic(semantic)
            except (IndexError, KeyError, TypeError):
                selected_list_digests = contract._PRODUCTION_PARTITION_LIST_SHA256
        if selected_member_digests is None:
            try:
                selected_member_digests = _train_member_sha256_from_semantic(semantic)
            except (KeyError, TypeError):
                selected_member_digests = contract._PRODUCTION_TRAIN_MEMBER_SHA256
        else:
            selected_member_digests = dict(selected_member_digests)
        if expected_label_sha256 is not None:
            selected_member_digests["labels"] = expected_label_sha256
        if selected_depth_list_digest is None:
            try:
                selected_depth_list_digest = (
                    _ordered_depth_list_sha256_from_semantic(semantic)
                )
            except (KeyError, TypeError):
                selected_depth_list_digest = (
                    contract.HISTORICAL_SESSION2_ORDERED_DEPTH_LIST_SHA256
                )
    else:
        selected_list_digests = contract._PRODUCTION_PARTITION_LIST_SHA256
        selected_member_digests = contract._PRODUCTION_TRAIN_MEMBER_SHA256
        selected_depth_list_digest = (
            contract.HISTORICAL_SESSION2_ORDERED_DEPTH_LIST_SHA256
        )
    return contract._validate_perception_input_manifest(
        semantic,
        outer_rows,
        input_name=input_name,
        current_identity=current_identity,
        expected_partition_sha256=selected,
        expected_partition_list_sha256=selected_list_digests,
        expected_train_member_sha256=selected_member_digests,
        expected_ordered_depth_list_sha256=selected_depth_list_digest,
        expected_geometry_support_sha256=selected_geometry_support,
        expected_geometry_smpl_manifest_sha256=selected_geometry_manifest,
    )


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


EXPECTED_POLICY = {
    "depth_hole_fraction": {"max": 0.4, "min": 0.0},
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
        "urdf_sha256": (
            "092ac325b834e69b3cca703c9711d0166f6c8cad54356926968bca562e13f0d5"
        ),
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

EXPECTED_ENVIRONMENT_RELEASE = "wam-5d2be01285ba34687f25"
EXPECTED_ENVIRONMENT_SHA256 = (
    "5d2be01285ba34687f257479429f8c43f2046fbb05ae2fa43ecf82f40c5d65d7"
)
EXPECTED_WARM_START = {
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


@pytest.fixture
def valid_config(tmp_path: Path) -> Tuple[dict, Path, Path]:
    code_root = tmp_path / "code-releases"
    environment_root = tmp_path / "environment-releases"
    code_release_id = "code_release-" + "a" * 20
    (code_root / code_release_id).mkdir(parents=True)
    (environment_root / EXPECTED_ENVIRONMENT_RELEASE).mkdir(parents=True)
    config = {
        "schema": "wam_perception_visible_stability_canary_v1",
        "code": {
            "release_id": code_release_id,
            "payload_sha256": "a" * 64,
            "realpath": str(code_root / code_release_id),
        },
        "environment": {
            "release_id": EXPECTED_ENVIRONMENT_RELEASE,
            "identity_sha256": EXPECTED_ENVIRONMENT_SHA256,
            "realpath": str(environment_root / EXPECTED_ENVIRONMENT_RELEASE),
        },
        "warm_start": deepcopy(EXPECTED_WARM_START),
        "inputs": {
            "geometry_source": {
                "release_id": "perception_geometry_source-" + "b" * 20,
                "payload_sha256": "b" * 64,
                "release_manifest_sha256": "c" * 64,
                "semantic_manifest_sha256": "d" * 64,
            },
            "stage0": {
                "release_id": "perception_stage0-" + "e" * 20,
                "payload_sha256": "e" * 64,
                "release_manifest_sha256": "f" * 64,
                "semantic_manifest_sha256": "0" * 64,
            },
            "train": {
                "release_id": "perception_session2_train-" + "1" * 20,
                "payload_sha256": "1" * 64,
                "release_manifest_sha256": "2" * 64,
                "semantic_manifest_sha256": "3" * 64,
            },
        },
        "seeds": [17, 23, 42],
        "training": {
            "epochs": 5,
            "batch_size": 8,
            "learning_rate": 1e-5,
        },
        "mask_policy": {
            "path": POLICY_RELATIVE_PATH,
            "sha256": hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest(),
        },
    }
    return config, code_root, environment_root


def _valid_receipt(inputs: dict) -> InputVerificationReceipt:
    return InputVerificationReceipt(expected_input_identity_sha256(inputs))


def _validate(
    config: object,
    code_root: Path,
    environment_root: Path,
    verifier=_valid_receipt,
    *,
    policy_root: Path = REPO_ROOT,
    repo_root: Path = REPO_ROOT,
):
    return _validate_canary_config_for_test(
        config,
        code_root=code_root,
        environment_root=environment_root,
        policy_root=policy_root,
        repo_root=repo_root,
        verify_inputs=verifier,
    )


def _assert_invalid(
    config: object,
    code_root: Path,
    environment_root: Path,
    *,
    verifier=_valid_receipt,
    **kwargs,
) -> None:
    with pytest.raises(CanaryContractError):
        _validate(
            config,
            code_root,
            environment_root,
            verifier,
            **kwargs,
        )


def test_validate_canary_config_api_exists() -> None:
    assert callable(validate_canary_config)


def test_validated_identity_is_frozen() -> None:
    identity = ValidatedCanaryIdentity(b"{}\n", "a" * 64, "canary-a")

    with pytest.raises(FrozenInstanceError):
        identity.run_id = "canary-b"


def test_input_verification_receipt_is_frozen() -> None:
    receipt = InputVerificationReceipt("a" * 64)

    with pytest.raises(FrozenInstanceError):
        receipt.input_identity_sha256 = "b" * 64


def test_contract_error_is_a_runtime_error() -> None:
    assert issubclass(CanaryContractError, RuntimeError)


def test_checked_in_policy_is_exact_canonical_json() -> None:
    policy_bytes = POLICY_PATH.read_bytes()

    assert policy_bytes == _canonical_json_bytes(EXPECTED_POLICY)


def test_valid_config_returns_content_derived_identity_and_verifies_only_inputs(
    valid_config: Tuple[dict, Path, Path],
) -> None:
    config, code_root, environment_root = valid_config
    seen: List[Dict[str, dict]] = []
    expected_bytes = _canonical_json_bytes(config)
    expected_sha = hashlib.sha256(expected_bytes).hexdigest()

    def verify(inputs: Dict[str, dict]) -> InputVerificationReceipt:
        seen.append(inputs)
        return _valid_receipt(inputs)

    identity = _validate(config, code_root, environment_root, verify)

    assert identity == ValidatedCanaryIdentity(
        canonical_config_bytes=expected_bytes,
        config_sha256=expected_sha,
        run_id="perc_lmh_v9p1_visible_stability_canary-" + expected_sha[:20],
    )
    assert seen == [config["inputs"]]
    assert set(seen[0]) == {"geometry_source", "stage0", "train"}


def test_canonical_config_bytes_produce_the_same_identity(
    valid_config: Tuple[dict, Path, Path],
) -> None:
    config, code_root, environment_root = valid_config
    config_bytes = _canonical_json_bytes(config)

    from_dict = _validate(config, code_root, environment_root)
    from_bytes = _validate(config_bytes, code_root, environment_root)

    assert from_bytes == from_dict


@pytest.mark.parametrize("field", ["schema", "code", "environment", "warm_start", "inputs", "seeds", "training", "mask_policy"])
def test_missing_top_level_field_is_rejected(
    valid_config: Tuple[dict, Path, Path], field: str
) -> None:
    config, code_root, environment_root = valid_config
    del config[field]

    _assert_invalid(config, code_root, environment_root)


@pytest.mark.parametrize("field", ["run_id", "output_path", "unexpected"])
def test_caller_identity_or_extra_top_level_field_is_rejected(
    valid_config: Tuple[dict, Path, Path], field: str
) -> None:
    config, code_root, environment_root = valid_config
    config[field] = "caller-controlled"

    _assert_invalid(config, code_root, environment_root)


@pytest.mark.parametrize("forbidden", ["holdout", "Yellow", "EVALUATION"])
def test_forbidden_words_are_rejected_in_nested_keys_or_string_values(
    valid_config: Tuple[dict, Path, Path], forbidden: str
) -> None:
    base_config, code_root, environment_root = valid_config
    config = deepcopy(base_config)
    config["training"]["note_" + forbidden] = "safe"

    _assert_invalid(config, code_root, environment_root)

    config = deepcopy(base_config)
    config["inputs"]["train"]["release_id"] += "_" + forbidden
    _assert_invalid(config, code_root, environment_root)


@pytest.mark.parametrize(
    "raw",
    [
        b"{}",
        b"{not-json}\n",
        b"\xff\n",
        _canonical_json_bytes([]),
    ],
)
def test_bytes_must_be_canonical_utf8_json_object(
    valid_config: Tuple[dict, Path, Path], raw: bytes
) -> None:
    _, code_root, environment_root = valid_config

    _assert_invalid(raw, code_root, environment_root)


def test_noncanonical_encoding_of_a_valid_config_is_rejected(
    valid_config: Tuple[dict, Path, Path],
) -> None:
    config, code_root, environment_root = valid_config
    noncanonical = json.dumps(config, indent=2).encode("utf-8")

    _assert_invalid(noncanonical, code_root, environment_root)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("release_id", "code_release-" + "A" * 20),
        ("payload_sha256", "a" * 63),
        ("payload_sha256", "b" * 64),
        ("extra", "forbidden"),
    ],
)
def test_code_identity_is_exact_and_payload_bound(
    valid_config: Tuple[dict, Path, Path], field: str, value: object
) -> None:
    config, code_root, environment_root = valid_config
    config["code"][field] = value

    _assert_invalid(config, code_root, environment_root)


def test_missing_code_field_is_rejected(
    valid_config: Tuple[dict, Path, Path],
) -> None:
    config, code_root, environment_root = valid_config
    del config["code"]["payload_sha256"]

    _assert_invalid(config, code_root, environment_root)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("release_id", "wam-" + "0" * 20),
        ("identity_sha256", "5" * 64),
        ("extra", "forbidden"),
    ],
)
def test_environment_identity_is_exact(
    valid_config: Tuple[dict, Path, Path], field: str, value: object
) -> None:
    config, code_root, environment_root = valid_config
    config["environment"][field] = value

    _assert_invalid(config, code_root, environment_root)


def test_missing_environment_field_is_rejected(
    valid_config: Tuple[dict, Path, Path],
) -> None:
    config, code_root, environment_root = valid_config
    del config["environment"]["identity_sha256"]

    _assert_invalid(config, code_root, environment_root)


def test_public_config_api_has_no_test_roots_or_verifier() -> None:
    assert tuple(inspect.signature(validate_canary_config).parameters) == (
        "config_or_canonical_bytes",
    )
    assert tuple(
        inspect.signature(contract.validate_perception_input_manifest).parameters
    ) == (
        "semantic",
        "outer_rows",
        "input_name",
        "current_identity",
    )


def test_descriptor_pinned_production_input_verifier_accepts_three_completed_releases(
    verified_input_closure,
) -> None:
    inputs, roots = verified_input_closure

    receipt = _verify_input_closure(inputs, roots)

    assert receipt == InputVerificationReceipt(
        expected_input_identity_sha256(inputs)
    )


@pytest.mark.parametrize(
    "field", ["release_manifest_sha256", "semantic_manifest_sha256"]
)
def test_input_verifier_rejects_full_but_wrong_descriptor_hashes(
    verified_input_closure,
    field: str,
) -> None:
    inputs, roots = verified_input_closure
    poisoned = deepcopy(inputs)
    poisoned["stage0"][field] = "f" * 64

    with pytest.raises(CanaryContractError):
        _verify_input_closure(poisoned, roots)


def test_input_verifier_never_opens_mixed_eval_or_attestation_paths(
    verified_input_closure,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, roots = verified_input_closure
    opened = []
    real_open = contract._open_directory_no_follow

    def record_open(path: Path, field: str) -> int:
        opened.append(Path(path))
        return real_open(path, field)

    monkeypatch.setattr(contract, "_open_directory_no_follow", record_open)

    _verify_input_closure(inputs, roots)

    assert opened == [
        roots["geometry_source"],
        roots["stage0"],
        roots["train"],
    ]
    assert all("perception_eval" not in str(path) for path in opened)


@pytest.mark.parametrize(
    ("source_role", "target_role"),
    [("mac", "h200"), ("h200", "linbox")],
)
def test_stage0_release_requires_h200_source_and_target_roles(
    tmp_path: Path,
    verified_input_closure,
    source_role: str,
    target_role: str,
) -> None:
    _, identity, stage_root = _write_stage0_release(
        tmp_path,
        source_role=source_role,
        target_role=target_role,
    )
    inputs, roots = verified_input_closure
    candidate_inputs = deepcopy(inputs)
    candidate_roots = dict(roots)
    candidate_inputs["stage0"] = identity
    candidate_roots["stage0"] = stage_root

    with pytest.raises(CanaryContractError):
        _verify_input_closure(candidate_inputs, candidate_roots)


def test_stage0_does_not_trust_declared_count_when_ledgers_are_empty(
    tmp_path: Path,
    verified_input_closure,
) -> None:
    _, identity, stage_root = _write_stage0_release(tmp_path)
    inputs, roots = verified_input_closure
    candidate_inputs = deepcopy(inputs)
    candidate_roots = dict(roots)
    candidate_inputs["stage0"] = identity
    candidate_roots["stage0"] = stage_root

    with pytest.raises(CanaryContractError, match="Stage-0.*(ledger|3,628 generated)"):
        _verify_input_closure(candidate_inputs, candidate_roots)


def test_stage0_semantic_requires_3628_generated_sample_members(
    verified_input_closure,
) -> None:
    inputs, roots = verified_input_closure
    release = roots["stage0"] / inputs["stage0"]["release_id"]
    semantic, outer_rows = _semantic_and_outer_rows(release)
    removed = next(
        row for row in semantic["members"] if row["role"] == "generated_sample"
    )
    semantic["members"].remove(removed)
    outer_rows = [row for row in outer_rows if row["path"] != removed["path"]]

    with pytest.raises(CanaryContractError, match="3,628 generated"):
        _validate_semantic_for_test(
            semantic,
            outer_rows,
            input_name="stage0",
            current_identity=inputs["stage0"],
        )


@pytest.mark.parametrize("poison", ["extra_provenance", "wrong_provenance_path"])
def test_stage0_rejects_extra_or_noncanonical_provenance_member(
    verified_input_closure,
    poison: str,
) -> None:
    inputs, roots = verified_input_closure
    release = roots["stage0"] / inputs["stage0"]["release_id"]
    semantic, outer_rows = _semantic_and_outer_rows(release)
    if poison == "extra_provenance":
        body = b"hidden-stage0-member"
        digest = hashlib.sha256(body).hexdigest()
        semantic["members"].append(
            {
                "content_origin": "provenance",
                "path": "metadata/EXTRA_PROVENANCE.json",
                "role": "provenance",
                "sha256": digest,
                "size": len(body),
            }
        )
        outer_rows.append(
            {
                "path": "metadata/EXTRA_PROVENANCE.json",
                "sha256": digest,
                "size": len(body),
            }
        )
    else:
        member = next(
            row for row in semantic["members"] if row["role"] == "provenance"
        )
        old_path = member["path"]
        member["path"] = "metadata/renamed-provenance.json"
        next(row for row in outer_rows if row["path"] == old_path)["path"] = member[
            "path"
        ]
    semantic["members"].sort(key=lambda row: row["path"])
    outer_rows.sort(key=lambda row: row["path"])

    with pytest.raises(CanaryContractError):
        _validate_semantic_for_test(
            semantic,
            outer_rows,
            input_name="stage0",
            current_identity=inputs["stage0"],
        )


@pytest.mark.parametrize(
    "poison",
    ["sample_count", "mask_policy", "geometry_identity", "extra_key", "binary"],
)
def test_stage0_provenance_rejects_wrong_fixed_contract_or_geometry_identity(
    tmp_path: Path,
    verified_input_closure,
    poison: str,
) -> None:
    inputs, roots = verified_input_closure
    release = roots["stage0"] / inputs["stage0"]["release_id"]
    provenance_path = release / "PROVENANCE.json"
    if poison == "binary":
        poisoned_bytes = b"\x00raw-stage0-provenance"
    else:
        provenance = json.loads(provenance_path.read_text())
        if poison == "sample_count":
            provenance["sample_count"] = 3627
        elif poison == "mask_policy":
            provenance["mask_policy_sha256"] = "f" * 64
        elif poison == "geometry_identity":
            provenance["geometry_source"]["payload_sha256"] = "f" * 64
        else:
            provenance["encoded_payload"] = "Zm9yYmlkZGVu"
        poisoned_bytes = _canonical_json_bytes(provenance)
    candidate_inputs, candidate_roots = _write_input_member_variant(
        tmp_path,
        inputs=inputs,
        roots=roots,
        input_name="stage0",
        member_path="PROVENANCE.json",
        member_bytes=poisoned_bytes,
    )

    with pytest.raises(CanaryContractError):
        _verify_input_closure(candidate_inputs, candidate_roots)


def test_geometry_rejects_smpl_member_outside_canonical_sequence_path(
    verified_input_closure,
) -> None:
    inputs, roots = verified_input_closure
    release = roots["geometry_source"] / inputs["geometry_source"]["release_id"]
    semantic, outer_rows = _semantic_and_outer_rows(release)
    member = next(row for row in semantic["members"] if row["role"] == "smpl_params")
    old_path = member["path"]
    member["path"] = "renamed/sequence/smpl_params.npz"
    next(row for row in outer_rows if row["path"] == old_path)["path"] = member["path"]
    semantic["members"] = sorted(semantic["members"], key=lambda row: row["path"])
    outer_rows = sorted(outer_rows, key=lambda row: row["path"])

    with pytest.raises(CanaryContractError, match="canonical HuMMan sequence"):
        _validate_semantic_for_test(
            semantic,
            outer_rows,
            input_name="geometry_source",
            current_identity=inputs["geometry_source"],
        )


@pytest.mark.parametrize("poison", ["binary", "base64_wrapper", "extra_key"])
def test_geometry_source_terms_rejects_noncanonical_or_wrapped_content(
    tmp_path: Path,
    verified_input_closure,
    poison: str,
) -> None:
    inputs, roots = verified_input_closure
    release = roots["geometry_source"] / inputs["geometry_source"]["release_id"]
    source_terms_path = release / "provenance/SOURCE_TERMS.json"
    if poison == "binary":
        poisoned_bytes = b"\x00raw-rgb-or-binary"
    elif poison == "base64_wrapper":
        poisoned_bytes = _canonical_json_bytes(
            {"payload_base64": "cmF3LXJnYi1vci1iaW5hcnk="}
        )
    else:
        source_terms = json.loads(source_terms_path.read_text())
        source_terms["extra"] = "forbidden"
        poisoned_bytes = _canonical_json_bytes(source_terms)
    candidate_inputs, candidate_roots = _write_input_member_variant(
        tmp_path,
        inputs=inputs,
        roots=roots,
        input_name="geometry_source",
        member_path="provenance/SOURCE_TERMS.json",
        member_bytes=poisoned_bytes,
    )

    with pytest.raises(CanaryContractError):
        _verify_input_closure(candidate_inputs, candidate_roots)


def test_exported_geometry_source_terms_object_is_the_production_contract() -> None:
    assert contract.GEOMETRY_SOURCE_TERMS_EXPECTED == _geometry_source_terms_object(
        contract._PRODUCTION_GEOMETRY_SUPPORT_SHA256,
        contract.HUMMAN_SMPL_PARAMS_MANIFEST_SHA256,
    )


def test_geometry_rejects_hidden_builder_metadata_member(
    verified_input_closure,
) -> None:
    inputs, roots = verified_input_closure
    release = roots["geometry_source"] / inputs["geometry_source"]["release_id"]
    semantic, outer_rows = _semantic_and_outer_rows(release)
    body = b"hidden-raw-rgb-frame"
    digest = hashlib.sha256(body).hexdigest()
    semantic["members"].append(
        {
            "content_origin": "builder_metadata",
            "path": "metadata/frame.bin",
            "role": "builder_metadata",
            "sha256": digest,
            "size": len(body),
        }
    )
    outer_rows.append(
        {"path": "metadata/frame.bin", "sha256": digest, "size": len(body)}
    )
    semantic["members"].sort(key=lambda row: row["path"])
    outer_rows.sort(key=lambda row: row["path"])

    with pytest.raises(CanaryContractError):
        _validate_semantic_for_test(
            semantic,
            outer_rows,
            input_name="geometry_source",
            current_identity=inputs["geometry_source"],
        )


@pytest.mark.parametrize("poison", ["extra_license", "extra_provenance", "wrong_path"])
def test_geometry_rejects_extra_or_noncanonical_support_members(
    verified_input_closure,
    poison: str,
) -> None:
    inputs, roots = verified_input_closure
    release = roots["geometry_source"] / inputs["geometry_source"]["release_id"]
    semantic, outer_rows = _semantic_and_outer_rows(release)
    if poison == "wrong_path":
        member = next(row for row in semantic["members"] if row["role"] == "license")
        old_path = member["path"]
        member["path"] = "misc/renamed-license.txt"
        next(row for row in outer_rows if row["path"] == old_path)["path"] = member[
            "path"
        ]
    else:
        role = "license" if poison == "extra_license" else "provenance"
        body = ("extra-" + role).encode("ascii")
        digest = hashlib.sha256(body).hexdigest()
        path = "extra/%s.bin" % role
        semantic["members"].append(
            {
                "content_origin": role,
                "path": path,
                "role": role,
                "sha256": digest,
                "size": len(body),
            }
        )
        outer_rows.append({"path": path, "sha256": digest, "size": len(body)})
    semantic["members"].sort(key=lambda row: row["path"])
    outer_rows.sort(key=lambda row: row["path"])

    with pytest.raises(CanaryContractError):
        _validate_semantic_for_test(
            semantic,
            outer_rows,
            input_name="geometry_source",
            current_identity=inputs["geometry_source"],
        )


def test_geometry_support_hash_is_not_caller_substitutable(
    verified_input_closure,
) -> None:
    inputs, roots = verified_input_closure
    release = roots["geometry_source"] / inputs["geometry_source"]["release_id"]
    semantic, outer_rows = _semantic_and_outer_rows(release)
    expected_support = _geometry_support_sha256_from_semantic(semantic)
    member = next(
        row
        for row in semantic["members"]
        if row["path"] == "smpl/SMPL_NEUTRAL.pkl"
    )
    member["sha256"] = "f" * 64
    next(row for row in outer_rows if row["path"] == member["path"])[
        "sha256"
    ] = member["sha256"]

    with pytest.raises(CanaryContractError):
        _validate_semantic_for_test(
            semantic,
            outer_rows,
            input_name="geometry_source",
            current_identity=inputs["geometry_source"],
            expected_geometry_support_sha256=expected_support,
        )


def test_geometry_smpl_params_hashes_are_bound_in_sequence_path_order(
    verified_input_closure,
) -> None:
    inputs, roots = verified_input_closure
    release = roots["geometry_source"] / inputs["geometry_source"]["release_id"]
    semantic, outer_rows = _semantic_and_outer_rows(release)
    expected_manifest_sha256 = _geometry_smpl_manifest_sha256_from_semantic(semantic)
    smpl = sorted(
        (row for row in semantic["members"] if row["role"] == "smpl_params"),
        key=lambda row: row["path"],
    )
    first_sha256, second_sha256 = smpl[0]["sha256"], smpl[1]["sha256"]
    smpl[0]["sha256"], smpl[1]["sha256"] = second_sha256, first_sha256
    next(row for row in outer_rows if row["path"] == smpl[0]["path"])[
        "sha256"
    ] = second_sha256
    next(row for row in outer_rows if row["path"] == smpl[1]["path"])[
        "sha256"
    ] = first_sha256

    with pytest.raises(CanaryContractError):
        _validate_semantic_for_test(
            semantic,
            outer_rows,
            input_name="geometry_source",
            current_identity=inputs["geometry_source"],
            expected_geometry_smpl_manifest_sha256=expected_manifest_sha256,
        )


def test_geometry_production_authenticity_hash_constants_are_exact() -> None:
    assert contract.HUMMAN_SMPL_PARAMS_MANIFEST_SHA256 == (
        "4ceeaf2616a0b62948abed3bf366b73fc057e5c0cf68abe488dfbc250021350b"
    )
    assert contract.GEOMETRY_SMPL_NEUTRAL_SHA256 == (
        "b93df2dd228acc91d4ea6ede5f83f81e5171e7bee35c39695e6e27cb25ee1d4a"
    )
    assert contract.GEOMETRY_DENSEPOSE_UV_SHA256 == (
        "ef109a96e57ddadd67b567ac591ec0bfe728c8a567f29927fc85500eb3cd374b"
    )
    assert contract.GEOMETRY_HUMMAN_LICENSE_SHA256 == (
        "bec0dbad332972acedb0963fe221053590d7f556a0c15ef562062abdc1894a0e"
    )
    assert contract.GEOMETRY_SMPL_LICENSE_SHA256 == (
        "15df63171dab571bcbacd9a36e734583fdcd26833365c51c781fa1c46f543075"
    )
    assert contract.GEOMETRY_DENSEPOSE_LICENSE_SHA256 == (
        "336255dc30193e8e15d689d9481bb05673d89055718f3a96923a7ffb99adbbaf"
    )


@pytest.mark.parametrize("poison", ["missing", "renamed", "hash_mismatch"])
def test_stage0_ledger_maps_every_sample_one_to_one_to_payload(
    verified_input_closure,
    poison: str,
) -> None:
    inputs, roots = verified_input_closure
    release = roots["stage0"] / inputs["stage0"]["release_id"]
    semantic, sample_ledger, split_ledger = _stage0_documents(release)
    if poison == "missing":
        sample_ledger["samples"].pop()
    elif poison == "renamed":
        sample_ledger["samples"][0]["sample_path"] = "samples/renamed.npz"
    else:
        sample_ledger["samples"][0]["sample_sha256"] = "f" * 64

    with pytest.raises(CanaryContractError):
        contract._validate_stage0_ledger_documents(
            semantic,
            sample_ledger,
            split_ledger,
            expected_geometry_sequence_ids=_valid_geometry_sequence_ids(),
        )


def test_geometry_rejects_duplicate_canonical_sequence_identity(
    verified_input_closure,
) -> None:
    inputs, roots = verified_input_closure
    release = roots["geometry_source"] / inputs["geometry_source"]["release_id"]
    semantic, outer_rows = _semantic_and_outer_rows(release)
    smpl = [row for row in semantic["members"] if row["role"] == "smpl_params"]
    old_path = smpl[1]["path"]
    smpl[1]["path"] = smpl[0]["path"]
    next(row for row in outer_rows if row["path"] == old_path)["path"] = smpl[1]["path"]
    semantic["members"] = sorted(semantic["members"], key=lambda row: row["path"])
    outer_rows = sorted(outer_rows, key=lambda row: row["path"])

    with pytest.raises(CanaryContractError):
        _validate_semantic_for_test(
            semantic,
            outer_rows,
            input_name="geometry_source",
            current_identity=inputs["geometry_source"],
        )


def test_stage0_release_rejects_wrong_outer_artifact_kind_even_when_renamed(
    tmp_path: Path,
    verified_input_closure,
) -> None:
    release, identity, stage_root = _write_stage0_release(
        tmp_path,
        artifact_kind="perception_geometry_source",
        source_role="mac",
    )
    fake_id = "perception_stage0-" + identity["payload_sha256"][:20]
    release.rename(stage_root / fake_id)
    identity["release_id"] = fake_id
    inputs, roots = verified_input_closure
    candidate_inputs = deepcopy(inputs)
    candidate_roots = dict(roots)
    candidate_inputs["stage0"] = identity
    candidate_roots["stage0"] = stage_root

    with pytest.raises(CanaryContractError):
        _verify_input_closure(candidate_inputs, candidate_roots)


@pytest.mark.parametrize("poison", ["tamper", "extra", "mode"])
def test_artifact_verifier_rejects_payload_tamper_extra_and_mode_drift(
    tmp_path: Path,
    verified_input_closure,
    poison: str,
) -> None:
    release, identity, stage_root = _write_stage0_release(tmp_path)
    if poison == "tamper":
        (release / "PROVENANCE.json").write_bytes(b"tampered")
    elif poison == "extra":
        (release / "unexpected.bin").write_bytes(b"extra")
    else:
        (release / "PROVENANCE.json").chmod(0o600)
    inputs, roots = verified_input_closure
    candidate_inputs = deepcopy(inputs)
    candidate_roots = dict(roots)
    candidate_inputs["stage0"] = identity
    candidate_roots["stage0"] = stage_root

    with pytest.raises(CanaryContractError):
        _verify_input_closure(candidate_inputs, candidate_roots)


@pytest.mark.parametrize("encoding", ["extra", "noncanonical", "nan"])
def test_semantic_manifest_requires_exact_finite_canonical_schema(
    tmp_path: Path,
    verified_input_closure,
    encoding: str,
) -> None:
    transform = None
    encoder = None
    if encoding == "extra":
        transform = lambda semantic: semantic.__setitem__("extra", "forbidden")
    elif encoding == "noncanonical":
        encoder = lambda semantic: json.dumps(semantic, indent=2).encode("utf-8")
    else:
        transform = lambda semantic: semantic["contract"].__setitem__(
            "nonfinite", float("nan")
        )
        encoder = artifact_contract.canonical_json_bytes
    _, identity, stage_root = _write_stage0_release(
        tmp_path,
        semantic_transform=transform,
        semantic_encoder=encoder,
    )
    inputs, roots = verified_input_closure
    candidate_inputs = deepcopy(inputs)
    candidate_roots = dict(roots)
    candidate_inputs["stage0"] = identity
    candidate_roots["stage0"] = stage_root

    with pytest.raises(CanaryContractError):
        _verify_input_closure(candidate_inputs, candidate_roots)


@pytest.mark.parametrize("poison", ["duplicate", "size_mismatch"])
def test_semantic_members_must_exactly_match_outer_payload_without_duplicates(
    tmp_path: Path,
    verified_input_closure,
    poison: str,
) -> None:
    def mutate(semantic: dict) -> None:
        if poison == "duplicate":
            semantic["members"].append(deepcopy(semantic["members"][0]))
        else:
            semantic["members"][0]["size"] += 1

    _, identity, stage_root = _write_stage0_release(
        tmp_path,
        semantic_transform=mutate,
    )
    inputs, roots = verified_input_closure
    candidate_inputs = deepcopy(inputs)
    candidate_roots = dict(roots)
    candidate_inputs["stage0"] = identity
    candidate_roots["stage0"] = stage_root

    with pytest.raises(CanaryContractError):
        _verify_input_closure(candidate_inputs, candidate_roots)


@pytest.mark.parametrize(
    "poison",
    [
        "session3_path",
        "yellow_origin",
        "denied_member_sha",
        "denied_decoded_fingerprint",
        "resolvable_dependency",
        "attestation_path",
    ],
)
def test_train_semantics_reject_held_out_content_and_resolvers(
    verified_input_closure,
    poison: str,
) -> None:
    inputs, roots = verified_input_closure
    train_release = roots["train"] / inputs["train"]["release_id"]
    semantic, outer_rows = _semantic_and_outer_rows(train_release)
    expected_partition_sha256 = _partition_sha256_from_semantic(semantic)
    if poison == "session3_path":
        member = semantic["members"][0]
        old_path = member["path"]
        member["path"] = "session3/renamed.bin"
        next(row for row in outer_rows if row["path"] == old_path)["path"] = member["path"]
        semantic["members"] = sorted(semantic["members"], key=lambda row: row["path"])
        outer_rows = sorted(outer_rows, key=lambda row: row["path"])
    elif poison == "yellow_origin":
        semantic["members"][0]["content_origin"] = "yellow_garment"
    elif poison == "denied_member_sha":
        member = semantic["members"][0]
        member["sha256"] = next(iter(contract.DENIED_TRAIN_CONTENT_SHA256))
        next(row for row in outer_rows if row["path"] == member["path"])[
            "sha256"
        ] = member["sha256"]
    elif poison == "denied_decoded_fingerprint":
        fingerprints = semantic["contract"]["decoded_frame_fingerprints"]
        fingerprints[0] = next(iter(contract.DENIED_TRAIN_CONTENT_SHA256))
        semantic["contract"]["decoded_frame_fingerprints"] = sorted(fingerprints)
    elif poison == "resolvable_dependency":
        semantic["resolvable_dependencies"] = ["/forbidden/root"]
    else:
        semantic["source_attestations"][0]["path"] = "/mixed/eval/release"

    with pytest.raises(CanaryContractError):
        _validate_semantic_for_test(
            semantic,
            outer_rows,
            input_name="train",
            current_identity=inputs["train"],
            expected_partition_sha256=expected_partition_sha256,
        )


def test_exported_holdout_denylist_and_shared_intrinsics_exception_are_fixed() -> None:
    assert contract.DENIED_TRAIN_CONTENT_SHA256 == frozenset(
        {
            "075671526052c4cfd0cb59706254dd8b29f011aa5c33b268022a37df3cdbee37"
        }
    )
    assert contract.ALLOWED_SHARED_CONTENT_ORIGINS == frozenset(
        {"shared_intrinsics"}
    )


@pytest.mark.parametrize(
    "poison", ["missing", "wrong_identity", "omission", "extra"]
)
def test_train_attestation_exactly_binds_historical_eval_and_all_copied_content(
    verified_input_closure,
    poison: str,
) -> None:
    inputs, roots = verified_input_closure
    train_release = roots["train"] / inputs["train"]["release_id"]
    semantic, outer_rows = _semantic_and_outer_rows(train_release)
    expected_partition_sha256 = _partition_sha256_from_semantic(semantic)
    if poison == "missing":
        semantic["source_attestations"] = []
        attestation = None
    else:
        attestation = semantic["source_attestations"][0]
    if poison == "missing":
        pass
    elif poison == "wrong_identity":
        attestation["payload_sha256"] = "a" * 64
        attestation["release_id"] = "perception_eval-" + "a" * 20
    elif poison == "omission":
        attestation["content_partition"]["session2_selected_sha256"].pop()
    else:
        selected = attestation["content_partition"]["session2_selected_sha256"]
        attestation["content_partition"]["session2_selected_sha256"] = sorted(
            selected + ["0" * 64]
        )

    with pytest.raises(CanaryContractError):
        _validate_semantic_for_test(
            semantic,
            outer_rows,
            input_name="train",
            current_identity=inputs["train"],
            expected_partition_sha256=expected_partition_sha256,
        )


def test_historical_mixed_eval_identity_is_exactly_exported() -> None:
    assert contract.HISTORICAL_MIXED_EVAL_RELEASE_ID == (
        "perception_eval-efc716189fc37d8886ba"
    )
    assert contract.HISTORICAL_MIXED_EVAL_PAYLOAD_SHA256 == (
        "efc716189fc37d8886ba9d60ac3cdb0cbfc8204e9b149ccee91216e65fff49a9"
    )
    assert contract.HISTORICAL_MIXED_EVAL_MANIFEST_SHA256 == (
        "952f1b266427828f8b83ba870f3a16d2867a2b11dd28f895ee6cf06497a6c5dd"
    )


def test_historical_mixed_eval_partition_identity_is_exactly_exported() -> None:
    assert contract.HISTORICAL_MIXED_EVAL_PARTITION_SHA256 == (
        "dace556e5191e3bfe95f1dbfd7484d633888a9ae1d5ee8a102a1638e735fa490"
    )


def test_partition_schema_and_shared_intrinsics_shape_are_exact() -> None:
    assert contract.MIXED_EVAL_CONTENT_PARTITION_SCHEMA == (
        "wam_perception_eval_content_partition_v1"
    )


def test_synthetic_production_shape_partition_passes_with_private_digest_injection(
    verified_input_closure,
) -> None:
    inputs, roots = verified_input_closure
    release = roots["train"] / inputs["train"]["release_id"]
    semantic, outer_rows, _ = _session2_documents(release)
    partition = semantic["source_attestations"][0]["content_partition"]

    _validate_semantic_for_test(
        semantic,
        outer_rows,
        input_name="train",
        current_identity=inputs["train"],
        expected_partition_sha256=hashlib.sha256(
            _canonical_json_bytes(partition)
        ).hexdigest(),
    )

    assert len(partition["session2_selected_sha256"]) == 605
    assert len(partition["session3_denied_sha256"]) == 608
    assert len(partition["session2_decoded_frame_sha256"]) == 602
    assert len(partition["session3_decoded_frame_sha256"]) == 602
    assert len(partition["shared_intrinsics_sha256"]) == 1


def test_partition_tamper_is_rejected_against_pinned_digest(
    verified_input_closure,
) -> None:
    inputs, roots = verified_input_closure
    release = roots["train"] / inputs["train"]["release_id"]
    semantic, outer_rows, _ = _session2_documents(release)
    expected = _partition_sha256_from_semantic(semantic)
    denied = semantic["source_attestations"][0]["content_partition"][
        "session3_denied_sha256"
    ]
    denied[1] = "f" * 64
    denied.sort()

    with pytest.raises(CanaryContractError, match="partition SHA-256"):
        _validate_semantic_for_test(
            semantic,
            outer_rows,
            input_name="train",
            current_identity=inputs["train"],
            expected_partition_sha256=expected,
        )


def test_partition_list_tamper_is_rejected_even_when_full_digest_is_rebound(
    verified_input_closure,
) -> None:
    inputs, roots = verified_input_closure
    release = roots["train"] / inputs["train"]["release_id"]
    semantic, outer_rows, _ = _session2_documents(release)
    expected_list_digests = _partition_list_sha256_from_semantic(semantic)
    denied = semantic["source_attestations"][0]["content_partition"][
        "session3_denied_sha256"
    ]
    denied[1] = "f" * 64
    denied.sort()

    with pytest.raises(CanaryContractError, match="session3_denied_sha256 digest"):
        _validate_semantic_for_test(
            semantic,
            outer_rows,
            input_name="train",
            current_identity=inputs["train"],
            expected_partition_sha256=_partition_sha256_from_semantic(semantic),
            expected_partition_list_sha256=expected_list_digests,
        )


def test_renamed_session3_content_hash_is_rejected_even_with_session2_origin(
    verified_input_closure,
) -> None:
    inputs, roots = verified_input_closure
    release = roots["train"] / inputs["train"]["release_id"]
    semantic, outer_rows, _ = _session2_documents(release)
    partition = semantic["source_attestations"][0]["content_partition"]
    member = next(row for row in semantic["members"] if row["role"] == "metric_depth")
    old_sha = member["sha256"]
    denied_sha = partition["session3_decoded_frame_sha256"][0]
    member["sha256"] = denied_sha
    next(row for row in outer_rows if row["path"] == member["path"])[
        "sha256"
    ] = denied_sha
    selected = partition["session2_selected_sha256"]
    selected.remove(old_sha)
    selected.append(denied_sha)
    selected.sort()

    with pytest.raises(CanaryContractError, match="denied Session-3"):
        _validate_semantic_for_test(
            semantic,
            outer_rows,
            input_name="train",
            current_identity=inputs["train"],
        )


def test_shared_intrinsics_is_the_only_allowed_partition_overlap(
    verified_input_closure,
) -> None:
    inputs, roots = verified_input_closure
    release = roots["train"] / inputs["train"]["release_id"]
    semantic, outer_rows, _ = _session2_documents(release)
    partition = semantic["source_attestations"][0]["content_partition"]
    shared = partition["shared_intrinsics_sha256"][0]

    _validate_semantic_for_test(
        semantic,
        outer_rows,
        input_name="train",
        current_identity=inputs["train"],
    )

    assert set(partition["session2_selected_sha256"]) & set(
        partition["session3_denied_sha256"]
    ) == {shared}
    assert any(
        member["sha256"] == shared
        and member["role"] == "intrinsics"
        and member["content_origin"] == "shared_intrinsics"
        for member in semantic["members"]
    )


def test_train_rejects_decoy_orbbec_origin_on_wrong_required_role(
    verified_input_closure,
) -> None:
    inputs, roots = verified_input_closure
    release = roots["train"] / inputs["train"]["release_id"]
    semantic, outer_rows, _ = _session2_documents(release)
    source_frame_index = next(
        member
        for member in semantic["members"]
        if member["role"] == "source_frame_index"
    )
    coordinate = next(
        member
        for member in semantic["members"]
        if member["role"] == "coordinate_metadata"
    )
    source_frame_index["content_origin"] = "builder_metadata"
    coordinate["content_origin"] = "orbbec_session2"
    selected = semantic["source_attestations"][0]["content_partition"][
        "session2_selected_sha256"
    ]
    selected.remove(source_frame_index["sha256"])
    selected.append(coordinate["sha256"])
    selected.sort()

    with pytest.raises(CanaryContractError):
        _validate_semantic_for_test(
            semantic,
            outer_rows,
            input_name="train",
            current_identity=inputs["train"],
        )


@pytest.mark.parametrize(
    ("required_role", "replacement_sha256", "replacement_path"),
    [
        ("rgb_video", "a" * 64, "video/replacement.mp4"),
        ("metric_depth", "b" * 64, "depth/replacement.npy"),
    ],
)
def test_train_rejects_required_source_substitution_with_original_hash_hidden_in_wrong_role(
    verified_input_closure,
    required_role: str,
    replacement_sha256: str,
    replacement_path: str,
) -> None:
    inputs, roots = verified_input_closure
    release = roots["train"] / inputs["train"]["release_id"]
    semantic, outer_rows, _ = _session2_documents(release)
    expected_member_sha256 = _train_member_sha256_from_semantic(semantic)
    required = next(
        member for member in semantic["members"] if member["role"] == required_role
    )
    original_path = required["path"]
    original_sha256 = required["sha256"]
    original_size = required["size"]
    required.update(
        {
            "content_origin": "builder_metadata",
            "path": replacement_path,
            "sha256": replacement_sha256,
        }
    )
    required_outer = next(row for row in outer_rows if row["path"] == original_path)
    required_outer.update(
        {"path": replacement_path, "sha256": replacement_sha256}
    )
    semantic["members"].append(
        {
            "content_origin": "builder_metadata",
            "path": "metadata/stashed-original-%s.bin" % required_role,
            "role": "coordinate_metadata",
            "sha256": original_sha256,
            "size": original_size,
        }
    )
    outer_rows.append(
        {
            "path": "metadata/stashed-original-%s.bin" % required_role,
            "sha256": original_sha256,
            "size": original_size,
        }
    )
    semantic["members"].sort(key=lambda row: row["path"])
    outer_rows.sort(key=lambda row: row["path"])
    selected = semantic["source_attestations"][0]["content_partition"][
        "session2_selected_sha256"
    ]
    selected.remove(original_sha256)
    selected.append(replacement_sha256)
    selected.sort()

    with pytest.raises(CanaryContractError):
        _validate_semantic_for_test(
            semantic,
            outer_rows,
            input_name="train",
            current_identity=inputs["train"],
            expected_train_member_sha256=expected_member_sha256,
        )


def test_train_dark_label_hash_is_not_caller_substitutable(
    verified_input_closure,
) -> None:
    inputs, roots = verified_input_closure
    release = roots["train"] / inputs["train"]["release_id"]
    semantic, outer_rows, _ = _session2_documents(release)
    label = next(member for member in semantic["members"] if member["role"] == "labels")
    expected_label_sha256 = label["sha256"]
    label["sha256"] = "f" * 64
    next(row for row in outer_rows if row["path"] == label["path"])[
        "sha256"
    ] = label["sha256"]

    with pytest.raises(CanaryContractError):
        _validate_semantic_for_test(
            semantic,
            outer_rows,
            input_name="train",
            current_identity=inputs["train"],
            expected_label_sha256=expected_label_sha256,
        )


def test_partition_list_digest_constants_are_exact() -> None:
    assert contract.HISTORICAL_SESSION2_SELECTED_LIST_SHA256 == (
        "f003515e44da86cde5c4a916eb65f9e51c6e2c3f87ef895f4d161b140db23b4b"
    )
    assert contract.HISTORICAL_SESSION3_DENIED_LIST_SHA256 == (
        "bb90159332274b43d3d9f3f88fe02092dfe93422a91497512c39d37c1cad0f85"
    )
    assert contract.HISTORICAL_SESSION2_DECODED_LIST_SHA256 == (
        "6043e8e62675dd9020d0b68ae7d71b50a08d5d7f4aa4d67c208d511f9dbbb612"
    )
    assert contract.HISTORICAL_SESSION3_DECODED_LIST_SHA256 == (
        "4c824067e6d497e0b579429f7a775bf2987e19201ff50e08b2dd2ad5858ef2e9"
    )
    assert contract.HISTORICAL_SHARED_INTRINSICS_LIST_SHA256 == (
        "61b8a4d5d54293a07e9bd52e2d984190aa803ac05c468b0e849b1014e169322c"
    )


def test_session2_production_projection_paths_and_member_hashes_are_exact() -> None:
    assert contract.SESSION2_RGB_VIDEO_PATH == "video/session2.mp4"
    assert contract.SESSION2_SOURCE_FRAMES_PATH == "source/frames.csv"
    assert contract.SESSION2_INTRINSICS_PATH == "camera/intrinsics.json"
    assert contract.SESSION2_LABELS_PATH == "labels/dark.json"
    assert contract.SESSION2_FRAME_LEDGER_PATH == "ledgers/frames.json"
    assert contract.SESSION2_COORDINATE_METADATA_PATH == "metadata/coordinates.json"
    assert contract.SESSION2_RGB_VIDEO_SHA256 == (
        "ac43ac4ef1f3fd977a83abdc22aa73c19ec4cd8f7034a0df44f1e76bf22c1699"
    )
    assert contract.SESSION2_SOURCE_FRAMES_SHA256 == (
        "9fd13926e840a5e6d98fda651d889f4236292b66c600329c39dc6be5e0f8a625"
    )
    assert contract.SESSION2_INTRINSICS_SHA256 == (
        "b05ce66978015518bcbe4dc39e09a5dbef6606153ccc04eab0eac0fc4f3875b3"
    )
    assert contract.SESSION2_DARK_LABEL_SHA256 == (
        "5917d20aae86c2ee978dcc51ff42c61fa94fbf592ff9330a6ccf49e90a6577cb"
    )
    assert contract.HISTORICAL_SESSION2_ORDERED_DEPTH_LIST_SHA256 == (
        "90e4337de51afb9b7d8e4f267bdd03341a228114f9302ae0b74a72a9335fee8f"
    )


def test_session2_coordinate_metadata_constant_and_member_pin_are_exact() -> None:
    expected = _coordinate_metadata_object()
    expected_sha256 = hashlib.sha256(_canonical_json_bytes(expected)).hexdigest()

    assert contract.SESSION2_COORDINATE_METADATA_EXPECTED == expected
    assert contract.SESSION2_COORDINATE_METADATA_SHA256 == expected_sha256
    assert contract._PRODUCTION_TRAIN_MEMBER_SHA256["coordinate_metadata"] == (
        expected_sha256
    )


@pytest.mark.parametrize(
    "poison",
    ("ascii", "landmark_order", "depth_unit", "dimensions", "registration"),
)
def test_descriptor_coordinate_metadata_rejects_noncanonical_semantics(
    tmp_path: Path,
    verified_input_closure,
    poison: str,
) -> None:
    inputs, roots = verified_input_closure
    document = _coordinate_metadata_object()
    if poison == "ascii":
        body = b"coords"
    else:
        if poison == "landmark_order":
            document["landmark_order"][0:2] = ["L1", "R1"]
        elif poison == "depth_unit":
            document["depth_unit"] = "millimetres"
        elif poison == "dimensions":
            document["width_px"] = 1280
        else:
            document["rgb_depth_registration"] = "unregistered"
        body = _canonical_json_bytes(document)
    candidate_inputs, candidate_roots = _write_input_member_variant(
        tmp_path,
        inputs=inputs,
        roots=roots,
        input_name="train",
        member_path=contract.SESSION2_COORDINATE_METADATA_PATH,
        member_bytes=body,
    )

    with pytest.raises(CanaryContractError):
        _verify_input_closure(candidate_inputs, candidate_roots)


def test_stage0_ledger_is_exactly_four_views_per_verified_geometry_sequence(
    verified_input_closure,
) -> None:
    inputs, roots = verified_input_closure
    release = roots["stage0"] / inputs["stage0"]["release_id"]
    semantic, sample_ledger, split_ledger = _stage0_documents(release)

    contract._validate_stage0_ledger_documents(
        semantic,
        sample_ledger,
        split_ledger,
        expected_geometry_sequence_ids=_valid_geometry_sequence_ids(),
    )
    assert len({row["sequence_id"] for row in sample_ledger["samples"]}) == 907
    assert len({row["subject_id"] for row in sample_ledger["samples"]}) == 340


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("sequence_id", "p999999_a999999"),
        ("subject_id", "p999999"),
        ("view_index", 3),
    ),
)
def test_stage0_ledger_rejects_sequence_subject_or_view_mismatch(
    verified_input_closure,
    field: str,
    replacement: object,
) -> None:
    inputs, roots = verified_input_closure
    release = roots["stage0"] / inputs["stage0"]["release_id"]
    semantic, sample_ledger, split_ledger = _stage0_documents(release)
    sample_ledger["samples"][0][field] = replacement

    with pytest.raises(CanaryContractError):
        contract._validate_stage0_ledger_documents(
            semantic,
            sample_ledger,
            split_ledger,
            expected_geometry_sequence_ids=_valid_geometry_sequence_ids(),
        )


def test_stage0_npz_validator_accepts_exact_deterministic_schema() -> None:
    _validate_stage0_npz(_npz_bytes(_valid_stage0_npz_arrays()))


def test_stage0_npz_validator_accepts_bounded_deflate_compression() -> None:
    _validate_stage0_npz(_compressed_npz_bytes(_valid_stage0_npz_arrays()))


@pytest.mark.parametrize(
    "poison",
    (
        "ascii",
        "wrong_key",
        "wrong_dtype",
        "wrong_shape",
        "nan_depth",
        "constant_depth",
        "out_of_frame_point",
        "point_without_depth",
        "background_normal_not_127",
        "metadata_mismatch",
    ),
)
def test_stage0_npz_validator_rejects_invalid_payloads(poison: str) -> None:
    if poison == "ascii":
        payload = b"not-an-npz"
    else:
        arrays = _valid_stage0_npz_arrays()
        if poison == "wrong_key":
            arrays["depth"] = arrays.pop("depth_m")
        elif poison == "wrong_dtype":
            arrays["depth_m"] = arrays["depth_m"].astype(np.float64)
        elif poison == "wrong_shape":
            arrays["normal_u8"] = arrays["normal_u8"][:-1]
        elif poison == "nan_depth":
            arrays["depth_m"][50, 50] = np.nan
        elif poison == "constant_depth":
            arrays["depth_m"].fill(1.0)
        elif poison == "out_of_frame_point":
            arrays["points_px"][0] = [518.0, 100.0]
        elif poison == "point_without_depth":
            point_x, point_y = arrays["points_px"][0].astype(np.intp)
            arrays["depth_m"][point_y, point_x] = 0.0
            arrays["normal_u8"][point_y, point_x] = 127
        elif poison == "background_normal_not_127":
            arrays["normal_u8"][0, 0] = [127, 127, 126]
        else:
            arrays["sample_id"] = np.asarray(
                "p000000_a000000_v1", dtype="<U18"
            )
        payload = _npz_bytes(arrays)

    with pytest.raises(CanaryContractError):
        _validate_stage0_npz(payload)


def test_stage0_npz_accepts_zero_interior_depth_hole_fraction() -> None:
    arrays = _valid_stage0_npz_arrays()
    positive_y, positive_x = np.nonzero(arrays["depth_m"] > 0.0)
    bounding_area = (
        positive_y.max() - positive_y.min() + 1
    ) * (positive_x.max() - positive_x.min() + 1)

    assert np.count_nonzero(arrays["depth_m"] > 0.0) == bounding_area
    _validate_stage0_npz(_npz_bytes(arrays))


def test_stage0_npz_accepts_zero_global_depth_hole_fraction() -> None:
    arrays = _valid_stage0_npz_arrays()
    yy, xx = np.indices(arrays["depth_m"].shape)
    arrays["depth_m"] = (
        0.9 + xx * 0.0003 + yy * 0.0002
    ).astype(np.float32)
    normal_x = (xx - 259.0) * 0.0003
    normal_y = (yy - 259.0) * 0.0002
    normal_z = -np.sqrt(1.0 - normal_x**2 - normal_y**2)
    arrays["normal_u8"] = (
        np.stack((normal_x, normal_y, normal_z), axis=-1) * 127.0 + 127.0
    ).astype(np.uint8)

    _validate_stage0_npz(_npz_bytes(arrays))


def test_stage0_npz_rejects_depth_hole_fraction_above_frozen_policy() -> None:
    arrays = _valid_stage0_npz_arrays()
    yy, xx = np.indices(arrays["depth_m"].shape)
    positive = arrays["depth_m"] > 0.0
    arrays["depth_m"][positive & ((xx + yy) % 2 == 0)] = 0.0
    for point_x, point_y in arrays["points_px"].astype(np.intp):
        arrays["depth_m"][point_y, point_x] = (
            0.9 + point_x * 0.0003 + point_y * 0.0002
        )
    arrays["normal_u8"][arrays["depth_m"] == 0.0] = 127

    with pytest.raises(CanaryContractError, match="depth hole fraction"):
        _validate_stage0_npz(_npz_bytes(arrays))


def test_stage0_npz_rejects_tiny_positive_depth_support() -> None:
    arrays = _valid_stage0_npz_arrays()
    source_normals = arrays["normal_u8"][40:65, 40:65].copy()
    arrays["depth_m"].fill(0.0)
    arrays["normal_u8"].fill(127)
    yy, xx = np.mgrid[200:225, 200:225]
    arrays["depth_m"][200:225, 200:225] = (
        0.9 + xx * 0.0003 + yy * 0.0002
    ).astype(np.float32)
    arrays["normal_u8"][200:225, 200:225] = source_normals
    arrays["points_px"] = np.asarray(
        [
            [203, 203],
            [221, 203],
            [204, 208],
            [220, 208],
            [205, 213],
            [219, 213],
            [204, 218],
            [220, 218],
            [206, 222],
            [218, 222],
        ],
        dtype=np.float32,
    )

    with pytest.raises(CanaryContractError, match="positive depth support"):
        _validate_stage0_npz(_npz_bytes(arrays))


def test_stage0_npz_rejects_large_visible_point_overlap() -> None:
    arrays = _valid_stage0_npz_arrays()
    arrays["points_px"][:] = arrays["points_px"][0]

    with pytest.raises(CanaryContractError, match="unique visible points"):
        _validate_stage0_npz(_npz_bytes(arrays))


def test_stage0_npz_requires_all_ten_visible_points_to_land_on_unique_pixels() -> None:
    arrays = _valid_stage0_npz_arrays()
    arrays["points_px"][-1] = arrays["points_px"][0]

    with pytest.raises(CanaryContractError, match="unique visible points"):
        _validate_stage0_npz(_npz_bytes(arrays))


def test_stage0_npz_rejects_collinear_visible_point_geometry() -> None:
    arrays = _valid_stage0_npz_arrays()
    arrays["points_px"] = np.asarray(
        [[100 + 20 * index, 120 + 20 * index] for index in range(10)],
        dtype=np.float32,
    )

    with pytest.raises(CanaryContractError, match="2D geometric spread"):
        _validate_stage0_npz(_npz_bytes(arrays))


def test_stage0_npz_rejects_nonunit_normals_under_exporter_decode() -> None:
    arrays = _valid_stage0_npz_arrays()
    positive = arrays["depth_m"] > 0.0
    arrays["normal_u8"][positive] = [254, 254, 254]
    first_y, first_x = np.argwhere(positive)[0]
    arrays["normal_u8"][first_y, first_x] = [253, 254, 254]

    with pytest.raises(CanaryContractError, match="decoded foreground normal norm"):
        _validate_stage0_npz(_npz_bytes(arrays))


def test_stage0_npz_rejects_nearly_empty_valid_normal_support() -> None:
    arrays = _valid_stage0_npz_arrays()
    valid_patch = arrays["normal_u8"][40:50, 40:50].copy()
    arrays["normal_u8"][arrays["depth_m"] > 0.0] = 127
    arrays["normal_u8"][40:50, 40:50] = valid_patch

    with pytest.raises(CanaryContractError, match="valid foreground normal support"):
        _validate_stage0_npz(_npz_bytes(arrays))


@pytest.mark.parametrize("poison", ("wrong_shape", "wrong_dtype"))
def test_stage0_npz_rejects_array_header_before_numpy_load(
    monkeypatch: pytest.MonkeyPatch,
    poison: str,
) -> None:
    arrays = _valid_stage0_npz_arrays()
    if poison == "wrong_shape":
        arrays["depth_m"] = arrays["depth_m"][:-1]
    else:
        arrays["depth_m"] = arrays["depth_m"].astype(np.float64)
    payload = _compressed_npz_bytes(arrays)
    load_called = False

    def forbidden_load(*args, **kwargs):
        nonlocal load_called
        del args, kwargs
        load_called = True
        raise AssertionError("np.load must not run before bounded NPY header validation")

    monkeypatch.setattr(np, "load", forbidden_load)
    with pytest.raises(CanaryContractError, match="NPY header"):
        _validate_stage0_npz(payload)
    assert load_called is False


def test_stage0_npz_rejects_oversized_legal_header_before_numpy_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arrays = _valid_stage0_npz_arrays()
    payload = _replace_npz_member(
        _npz_bytes(arrays),
        "depth_m.npy",
        _npy_with_oversized_v1_header(arrays["depth_m"]),
    )
    load_called = False

    def forbidden_load(*args, **kwargs):
        nonlocal load_called
        del args, kwargs
        load_called = True
        raise AssertionError("np.load must not parse an oversized NPY header")

    monkeypatch.setattr(np, "load", forbidden_load)
    with pytest.raises(CanaryContractError, match="NPY header"):
        _validate_stage0_npz(payload)
    assert load_called is False


def test_stage0_npz_rejects_zip_bomb_ratio_before_numpy_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arrays = _valid_stage0_npz_arrays()
    arrays["normal_u8"].fill(127)
    payload = _compressed_npz_bytes(arrays)
    load_called = False

    def forbidden_load(*args, **kwargs):
        nonlocal load_called
        del args, kwargs
        load_called = True
        raise AssertionError("np.load must not expand a high-ratio NPZ")

    monkeypatch.setattr(np, "load", forbidden_load)
    with pytest.raises(CanaryContractError, match="compression ratio"):
        _validate_stage0_npz(payload)
    assert load_called is False


def test_stage0_npz_rejects_oversized_public_payload_before_zip_parse() -> None:
    with pytest.raises(CanaryContractError, match="compressed payload hard limit"):
        _validate_stage0_npz(b"x" * (3 * 1024 * 1024))


def test_stage0_member_fstat_limit_precedes_descriptor_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_id = "p000000_a000000_v0"
    sample_path = "samples/%s.npz" % sample_id
    absolute = tmp_path / sample_path
    absolute.parent.mkdir()
    with absolute.open("wb") as handle:
        handle.truncate(3 * 1024 * 1024)
    read_called = False

    def forbidden_read(*args, **kwargs) -> bytes:
        nonlocal read_called
        del args, kwargs
        read_called = True
        raise AssertionError("oversized compressed payload must not be read")

    monkeypatch.setattr(contract, "_read_descriptor_bytes_bounded", forbidden_read)
    root_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(CanaryContractError, match="compressed payload hard limit"):
            contract._validate_stage0_npz_members(
                root_fd,
                [
                    {
                        "sample_id": sample_id,
                        "sample_path": sample_path,
                        "sample_sha256": "0" * 64,
                        "sequence_id": "p000000_a000000",
                        "subject_id": "p000000",
                        "view_index": 0,
                    }
                ],
                test_stage0_npz_validator_fd=None,
            )
    finally:
        os.close(root_fd)
    assert read_called is False


def test_stage0_member_growth_during_read_is_bounded_to_initial_size_plus_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_id = "p000000_a000000_v0"
    sample_path = "samples/%s.npz" % sample_id
    absolute = tmp_path / sample_path
    absolute.parent.mkdir()
    initial_size = 1024 * 1024
    grown_size = 3 * 1024 * 1024
    with absolute.open("wb") as handle:
        handle.truncate(initial_size)

    real_read = os.read
    bytes_returned = 0
    growth_injected = False

    def grow_after_first_read(descriptor: int, count: int) -> bytes:
        nonlocal bytes_returned, growth_injected
        chunk = real_read(descriptor, count)
        bytes_returned += len(chunk)
        if chunk and not growth_injected:
            with absolute.open("r+b") as handle:
                handle.truncate(grown_size)
            growth_injected = True
        return chunk

    monkeypatch.setattr(contract.os, "read", grow_after_first_read)
    root_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(CanaryContractError) as caught:
            contract._validate_stage0_npz_members(
                root_fd,
                [
                    {
                        "sample_id": sample_id,
                        "sample_path": sample_path,
                        "sample_sha256": "0" * 64,
                        "sequence_id": "p000000_a000000",
                        "subject_id": "p000000",
                        "view_index": 0,
                    }
                ],
                test_stage0_npz_validator_fd=None,
            )
    finally:
        os.close(root_fd)

    assert growth_injected is True
    assert absolute.stat().st_size == grown_size
    assert bytes_returned <= min(
        initial_size,
        contract.STAGE0_NPZ_MAX_COMPRESSED_BYTES,
    ) + 1
    assert "grew while it was read" in str(caught.value)


def test_bgr8_frame_sha256_uses_exact_canonical_header_and_contiguous_bytes() -> None:
    header = (
        b'{"dtype":"uint8","shape":[1080,1920,3],'
        b'"version":"wam_bgr8_frame_sha256_v1"}\n'
    )
    frame = b"\x00" * (1080 * 1920 * 3)

    assert contract.decoded_frame_hash_header_bytes() == header
    assert contract.bgr8_frame_sha256(frame) == hashlib.sha256(
        header + frame
    ).hexdigest()


def test_bgr8_frame_sha256_rejects_wrong_frame_length() -> None:
    with pytest.raises(CanaryContractError):
        contract.bgr8_frame_sha256(b"\x00")


def test_private_verifier_pins_rgb_fd_before_test_frame_hash_injection(
    verified_input_closure,
) -> None:
    inputs, roots = verified_input_closure
    seen = []
    release = roots["train"] / inputs["train"]["release_id"]
    semantic, _, _ = _session2_documents(release)
    geometry_semantic, _ = _semantic_and_outer_rows(
        roots["geometry_source"] / inputs["geometry_source"]["release_id"]
    )

    def frame_hashes_from_fd(descriptor: int) -> list:
        os.lseek(descriptor, 0, os.SEEK_SET)
        seen.append(os.read(descriptor, 1024))
        return semantic["source_attestations"][0]["content_partition"][
            "session2_decoded_frame_sha256"
        ]

    contract._verify_input_releases_at_roots(
        inputs,
        roots,
        expected_partition_sha256=_partition_sha256_from_semantic(semantic),
        expected_partition_list_sha256=_partition_list_sha256_from_semantic(semantic),
        expected_train_member_sha256=_train_member_sha256_from_semantic(semantic),
        expected_ordered_depth_list_sha256=(
            _ordered_depth_list_sha256_from_semantic(semantic)
        ),
        expected_geometry_support_sha256=(
            _geometry_support_sha256_from_semantic(geometry_semantic)
        ),
        expected_geometry_smpl_manifest_sha256=(
            _geometry_smpl_manifest_sha256_from_semantic(geometry_semantic)
        ),
        expected_geometry_source_terms=_geometry_source_terms_object(
            _geometry_support_sha256_from_semantic(geometry_semantic),
            _geometry_smpl_manifest_sha256_from_semantic(geometry_semantic),
        ),
        test_stage0_npz_validator_fd=_validate_fixture_stage0_sample_fd,
        test_frame_hashes_from_video_fd=frame_hashes_from_fd,
    )

    assert seen == [b"dark-rgb"]


@pytest.mark.parametrize("poison", ["missing", "mismatch"])
def test_private_video_verifier_rejects_incomplete_or_mismatched_decode(
    verified_input_closure,
    poison: str,
) -> None:
    inputs, roots = verified_input_closure
    release = roots["train"] / inputs["train"]["release_id"]
    semantic, _, _ = _session2_documents(release)
    geometry_semantic, _ = _semantic_and_outer_rows(
        roots["geometry_source"] / inputs["geometry_source"]["release_id"]
    )
    decoded = list(
        semantic["source_attestations"][0]["content_partition"][
            "session2_decoded_frame_sha256"
        ]
    )
    if poison == "missing":
        decoded.pop()
    else:
        decoded[0] = "f" * 64

    with pytest.raises(CanaryContractError):
        contract._verify_input_releases_at_roots(
            inputs,
            roots,
            expected_partition_sha256=_partition_sha256_from_semantic(semantic),
            expected_partition_list_sha256=_partition_list_sha256_from_semantic(
                semantic
            ),
            expected_train_member_sha256=_train_member_sha256_from_semantic(semantic),
            expected_ordered_depth_list_sha256=(
                _ordered_depth_list_sha256_from_semantic(semantic)
            ),
            expected_geometry_support_sha256=(
                _geometry_support_sha256_from_semantic(geometry_semantic)
            ),
            expected_geometry_smpl_manifest_sha256=(
                _geometry_smpl_manifest_sha256_from_semantic(geometry_semantic)
            ),
            expected_geometry_source_terms=_geometry_source_terms_object(
                _geometry_support_sha256_from_semantic(geometry_semantic),
                _geometry_smpl_manifest_sha256_from_semantic(geometry_semantic),
            ),
            test_stage0_npz_validator_fd=_validate_fixture_stage0_sample_fd,
            test_frame_hashes_from_video_fd=lambda _: decoded,
        )


def test_source_frame_index_parser_rejects_malformed_rows() -> None:
    validator = getattr(contract, "_validate_source_frame_index_bytes", None)
    assert validator is not None
    malformed = _valid_source_frame_index_bytes().replace(
        b"37,1.233333333,37,1,1.0\n",
        b"37,nan,37,1,1.0\n",
    )

    with pytest.raises(CanaryContractError):
        validator(malformed)


def test_source_frame_index_parser_accepts_exact_ordered_602_rows() -> None:
    contract._validate_source_frame_index_bytes(_valid_source_frame_index_bytes())


def test_private_hash_injection_cannot_bypass_source_frame_index_semantics(
    tmp_path: Path,
    verified_input_closure,
) -> None:
    inputs, roots = verified_input_closure
    original_release = roots["train"] / inputs["train"]["release_id"]
    semantic, _ = _semantic_and_outer_rows(original_release)
    original_source_member = next(
        member
        for member in semantic["members"]
        if member["role"] == "source_frame_index"
    )
    malformed = _valid_source_frame_index_bytes().replace(
        b"37,1.233333333,37,1,1.0\n",
        b"37,1.233333333,37,0,1.0\n",
    )
    malformed_sha256 = hashlib.sha256(malformed).hexdigest()
    source_attestations = deepcopy(semantic["source_attestations"])
    selected = source_attestations[0]["content_partition"][
        "session2_selected_sha256"
    ]
    selected.remove(original_source_member["sha256"])
    selected.append(malformed_sha256)
    selected.sort()
    train_root = tmp_path / "malformed-train" / "releases"
    train_root.mkdir(parents=True)
    _, malformed_identity = _write_input_release(
        train_root,
        artifact_kind=semantic["artifact_kind"],
        source_role=semantic["source_role"],
        session_ids=semantic["session_ids"],
        files=[
            (
                member["path"],
                (
                    malformed
                    if member["role"] == "source_frame_index"
                    else (original_release / member["path"]).read_bytes()
                ),
                member["role"],
                member["content_origin"],
            )
            for member in semantic["members"]
        ],
        contract_fields=deepcopy(semantic["contract"]),
        source_attestations=source_attestations,
    )
    candidate_inputs = deepcopy(inputs)
    candidate_roots = dict(roots)
    candidate_inputs["train"] = malformed_identity
    candidate_roots["train"] = train_root

    with pytest.raises(CanaryContractError, match="source frame depth metadata"):
        _verify_input_closure(candidate_inputs, candidate_roots)


def test_session2_contract_binds_exact_frame_ledger_path(
    verified_input_closure,
) -> None:
    inputs, roots = verified_input_closure
    release = roots["train"] / inputs["train"]["release_id"]
    semantic, _ = _semantic_and_outer_rows(release)

    assert semantic["contract"]["frame_ledger_path"] == "ledgers/frames.json"


def test_session2_rejects_depth_swap_even_when_member_set_and_ledger_are_synchronized(
    verified_input_closure,
) -> None:
    inputs, roots = verified_input_closure
    release = roots["train"] / inputs["train"]["release_id"]
    semantic, outer_rows, frame_ledger = _session2_documents(release)
    first = next(
        member
        for member in semantic["members"]
        if member["path"] == "depth/depth_000000.npy"
    )
    second = next(
        member
        for member in semantic["members"]
        if member["path"] == "depth/depth_000001.npy"
    )
    first_sha256, second_sha256 = first["sha256"], second["sha256"]
    first["sha256"], second["sha256"] = second_sha256, first_sha256
    next(row for row in outer_rows if row["path"] == first["path"])[
        "sha256"
    ] = second_sha256
    next(row for row in outer_rows if row["path"] == second["path"])[
        "sha256"
    ] = first_sha256
    frame_ledger["frames"][0]["depth_sha256"] = second_sha256
    frame_ledger["frames"][1]["depth_sha256"] = first_sha256

    with pytest.raises(CanaryContractError):
        _validate_semantic_for_test(
            semantic,
            outer_rows,
            input_name="train",
            current_identity=inputs["train"],
        )
        contract._validate_session2_frame_ledger_document(semantic, frame_ledger)


def test_session2_ordered_depth_list_cannot_be_rebound_by_private_full_digest(
    verified_input_closure,
) -> None:
    inputs, roots = verified_input_closure
    release = roots["train"] / inputs["train"]["release_id"]
    semantic, outer_rows, frame_ledger = _session2_documents(release)
    expected_ordered_digest = _ordered_depth_list_sha256_from_semantic(semantic)
    first = next(
        member
        for member in semantic["members"]
        if member["path"] == "depth/depth_000000.npy"
    )
    second = next(
        member
        for member in semantic["members"]
        if member["path"] == "depth/depth_000001.npy"
    )
    first_sha256, second_sha256 = first["sha256"], second["sha256"]
    first["sha256"], second["sha256"] = second_sha256, first_sha256
    next(row for row in outer_rows if row["path"] == first["path"])[
        "sha256"
    ] = second_sha256
    next(row for row in outer_rows if row["path"] == second["path"])[
        "sha256"
    ] = first_sha256
    ordered = semantic["contract"]["ordered_depth_fingerprints"]
    ordered[0], ordered[1] = ordered[1], ordered[0]
    frame_ledger["frames"][0]["depth_sha256"] = ordered[0]
    frame_ledger["frames"][1]["depth_sha256"] = ordered[1]

    with pytest.raises(CanaryContractError, match="ordered depth list SHA-256"):
        _validate_semantic_for_test(
            semantic,
            outer_rows,
            input_name="train",
            current_identity=inputs["train"],
            expected_ordered_depth_list_sha256=expected_ordered_digest,
        )


@pytest.mark.parametrize(
    "poison", ["missing", "renamed_depth", "depth_hash", "rgb_hash"]
)
def test_session2_frame_ledger_binds_all_depth_and_ordered_rgb_hashes(
    verified_input_closure,
    poison: str,
) -> None:
    inputs, roots = verified_input_closure
    release = roots["train"] / inputs["train"]["release_id"]
    semantic, _, frame_ledger = _session2_documents(release)
    if poison == "missing":
        frame_ledger["frames"].pop()
    elif poison == "renamed_depth":
        frame_ledger["frames"][0]["depth_path"] = "depth/renamed.npy"
    elif poison == "depth_hash":
        frame_ledger["frames"][0]["depth_sha256"] = "f" * 64
    else:
        frame_ledger["frames"][0]["rgb_sha256"] = "f" * 64

    with pytest.raises(CanaryContractError):
        contract._validate_session2_frame_ledger_document(
            semantic,
            frame_ledger,
        )


def test_code_realpath_must_be_the_direct_immutable_release_child(
    valid_config: Tuple[dict, Path, Path],
) -> None:
    config, code_root, environment_root = valid_config
    config["code"]["realpath"] = str(code_root / "current")

    _assert_invalid(config, code_root, environment_root)


def test_environment_realpath_must_be_the_direct_immutable_release_child(
    valid_config: Tuple[dict, Path, Path],
) -> None:
    config, code_root, environment_root = valid_config
    config["environment"]["realpath"] = str(
        environment_root / "current" / EXPECTED_ENVIRONMENT_RELEASE
    )

    _assert_invalid(config, code_root, environment_root)


def test_symlinked_immutable_release_path_is_rejected(
    valid_config: Tuple[dict, Path, Path], tmp_path: Path
) -> None:
    config, code_root, environment_root = valid_config
    release_path = Path(config["code"]["realpath"])
    release_path.rmdir()
    target = tmp_path / "mutable-code"
    target.mkdir()
    release_path.symlink_to(target, target_is_directory=True)

    _assert_invalid(config, code_root, environment_root)


def test_missing_immutable_release_directory_is_rejected(
    valid_config: Tuple[dict, Path, Path],
) -> None:
    config, code_root, environment_root = valid_config
    Path(config["code"]["realpath"]).rmdir()

    _assert_invalid(config, code_root, environment_root)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("release_id", "perception_runtime-" + "0" * 20),
        ("payload_sha256", "0" * 64),
        ("checkpoint_path", "model/perc_lmh_v9_occ/last.pt"),
        ("checkpoint_sha256", "0" * 64),
        ("architecture", "OtherDetector"),
        ("encoder", "dinov2_vits14"),
        ("lora_rank", 8),
        ("lora_rank", True),
        ("lora_alpha", 16),
        ("landmark_order", ["L1", "R1", "R2", "L2", "R3", "L3", "R4", "L4", "R5", "L5"]),
        ("extra", "forbidden"),
    ],
)
def test_every_warm_start_identity_field_is_exact(
    valid_config: Tuple[dict, Path, Path], field: str, value: object
) -> None:
    config, code_root, environment_root = valid_config
    config["warm_start"][field] = value

    _assert_invalid(config, code_root, environment_root)


def test_missing_warm_start_field_is_rejected(
    valid_config: Tuple[dict, Path, Path],
) -> None:
    config, code_root, environment_root = valid_config
    del config["warm_start"]["architecture"]

    _assert_invalid(config, code_root, environment_root)


@pytest.mark.parametrize("input_name", ["geometry_source", "stage0", "train"])
def test_input_identity_requires_exact_fields(
    valid_config: Tuple[dict, Path, Path], input_name: str
) -> None:
    base_config, code_root, environment_root = valid_config
    config = deepcopy(base_config)
    del config["inputs"][input_name]["release_manifest_sha256"]

    _assert_invalid(config, code_root, environment_root)

    config = deepcopy(base_config)
    config["inputs"][input_name]["extra"] = "forbidden"
    _assert_invalid(config, code_root, environment_root)


@pytest.mark.parametrize(
    ("input_name", "field", "value"),
    [
        ("geometry_source", "release_id", "perception_stage0-" + "b" * 20),
        ("stage0", "release_id", "perception_stage0-" + "b" * 20),
        ("train", "release_id", "perception_session3_train-" + "1" * 20),
        ("geometry_source", "payload_sha256", "b" * 63),
        ("stage0", "release_manifest_sha256", "F" * 64),
        ("train", "semantic_manifest_sha256", "not-a-hash"),
    ],
)
def test_input_prefixes_hashes_and_payload_binding_are_exact(
    valid_config: Tuple[dict, Path, Path],
    input_name: str,
    field: str,
    value: str,
) -> None:
    config, code_root, environment_root = valid_config
    config["inputs"][input_name][field] = value

    _assert_invalid(config, code_root, environment_root)


def test_input_names_are_exact(
    valid_config: Tuple[dict, Path, Path],
) -> None:
    config, code_root, environment_root = valid_config
    config["inputs"]["validation"] = config["inputs"].pop("stage0")

    _assert_invalid(config, code_root, environment_root)


@pytest.mark.parametrize(
    "seeds",
    [
        [17, 23, 23],
        [17, 23, 43],
        [17, 23],
        [17, True, 42],
        [42, 23, 17],
    ],
)
def test_seeds_are_exact_unique_integers(
    valid_config: Tuple[dict, Path, Path], seeds: list
) -> None:
    config, code_root, environment_root = valid_config
    config["seeds"] = seeds

    _assert_invalid(config, code_root, environment_root)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("epochs", True),
        ("epochs", 6),
        ("batch_size", False),
        ("batch_size", 4),
        ("learning_rate", True),
        ("learning_rate", float("nan")),
        ("learning_rate", float("inf")),
        ("learning_rate", 10**10000),
        ("learning_rate", 2e-5),
        ("extra", 1),
    ],
)
def test_training_values_and_types_are_exact_finite_numbers(
    valid_config: Tuple[dict, Path, Path], field: str, value: object
) -> None:
    config, code_root, environment_root = valid_config
    config["training"][field] = value

    _assert_invalid(config, code_root, environment_root)


def test_missing_training_field_is_rejected(
    valid_config: Tuple[dict, Path, Path],
) -> None:
    config, code_root, environment_root = valid_config
    del config["training"]["epochs"]

    _assert_invalid(config, code_root, environment_root)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("path", "auscultation/configs/perception/other.json"),
        ("sha256", "0" * 64),
        ("sha256", "0" * 63),
        ("extra", "forbidden"),
    ],
)
def test_mask_policy_reference_is_exact_and_content_bound(
    valid_config: Tuple[dict, Path, Path], field: str, value: object
) -> None:
    config, code_root, environment_root = valid_config
    config["mask_policy"][field] = value

    _assert_invalid(config, code_root, environment_root)


def test_missing_mask_policy_field_is_rejected(
    valid_config: Tuple[dict, Path, Path],
) -> None:
    config, code_root, environment_root = valid_config
    del config["mask_policy"]["sha256"]

    _assert_invalid(config, code_root, environment_root)


def test_self_consistent_but_drifted_policy_is_rejected(
    valid_config: Tuple[dict, Path, Path], tmp_path: Path
) -> None:
    config, code_root, environment_root = valid_config
    drifted = deepcopy(EXPECTED_POLICY)
    drifted["depth_hole_fraction"]["max"] = 0.3
    policy_bytes = _canonical_json_bytes(drifted)
    policy_root = tmp_path / "policy-root"
    policy_path = policy_root / POLICY_RELATIVE_PATH
    policy_path.parent.mkdir(parents=True)
    policy_path.write_bytes(policy_bytes)
    config["mask_policy"]["sha256"] = hashlib.sha256(policy_bytes).hexdigest()

    _assert_invalid(
        config,
        code_root,
        environment_root,
        policy_root=policy_root,
    )


def test_noncanonical_policy_bytes_are_rejected(
    valid_config: Tuple[dict, Path, Path], tmp_path: Path
) -> None:
    config, code_root, environment_root = valid_config
    policy_bytes = POLICY_PATH.read_bytes() + b" \n"
    policy_root = tmp_path / "policy-root"
    policy_path = policy_root / POLICY_RELATIVE_PATH
    policy_path.parent.mkdir(parents=True)
    policy_path.write_bytes(policy_bytes)
    config["mask_policy"]["sha256"] = hashlib.sha256(policy_bytes).hexdigest()

    _assert_invalid(
        config,
        code_root,
        environment_root,
        policy_root=policy_root,
    )


def test_urdf_content_drift_is_rejected(
    valid_config: Tuple[dict, Path, Path], tmp_path: Path
) -> None:
    config, code_root, environment_root = valid_config
    repo_root = tmp_path / "repo-root"
    urdf_path = repo_root / URDF_RELATIVE_PATH
    urdf_path.parent.mkdir(parents=True)
    urdf_path.write_bytes((REPO_ROOT / URDF_RELATIVE_PATH).read_bytes() + b"\n")

    _assert_invalid(
        config,
        code_root,
        environment_root,
        repo_root=repo_root,
    )


def test_policy_read_stays_on_pinned_directory_when_ancestor_is_rebound(
    valid_config: Tuple[dict, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, code_root, environment_root = valid_config
    policy_root = tmp_path / "policy-root"
    policy_path = policy_root / POLICY_RELATIVE_PATH
    policy_path.parent.mkdir(parents=True)
    policy_path.write_bytes(POLICY_PATH.read_bytes())
    parked = tmp_path / "parked-policy-root"
    interleaved = []

    def rebind_after_open(opened_path: Path, _: int) -> None:
        if opened_path == policy_root and not interleaved:
            policy_root.rename(parked)
            replacement = policy_root / POLICY_RELATIVE_PATH
            replacement.parent.mkdir(parents=True)
            replacement.write_bytes(b"attacker-controlled-policy\n")
            interleaved.append(True)

    monkeypatch.setattr(
        contract, "_directory_walk_interleave", rebind_after_open, raising=False
    )

    identity = _validate(
        config,
        code_root,
        environment_root,
        policy_root=policy_root,
    )

    assert interleaved == [True]
    assert identity.run_id.startswith("perc_lmh_v9p1_visible_stability_canary-")


def test_directory_walk_rejects_symlink_swapped_before_component_open(
    valid_config: Tuple[dict, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, code_root, environment_root = valid_config
    policy_root = tmp_path / "policy-root"
    policy_path = policy_root / POLICY_RELATIVE_PATH
    policy_path.parent.mkdir(parents=True)
    policy_path.write_bytes(POLICY_PATH.read_bytes())
    parked = tmp_path / "parked-policy-root"
    interleaved = []

    def swap_before_open(opened_path: Path, _: int) -> None:
        if opened_path == policy_root.parent and not interleaved:
            policy_root.rename(parked)
            policy_root.symlink_to(parked, target_is_directory=True)
            interleaved.append(True)

    monkeypatch.setattr(
        contract, "_directory_walk_interleave", swap_before_open, raising=False
    )

    _assert_invalid(
        config,
        code_root,
        environment_root,
        policy_root=policy_root,
    )
    assert interleaved == [True]


def test_expected_input_identity_is_canonical_and_deterministic(
    valid_config: Tuple[dict, Path, Path],
) -> None:
    config, _, _ = valid_config
    reordered = dict(reversed(list(config["inputs"].items())))

    assert expected_input_identity_sha256(reordered) == hashlib.sha256(
        _canonical_json_bytes(config["inputs"])
    ).hexdigest()


@pytest.mark.parametrize("invalid_receipt", [False, None, iter(())])
def test_verifier_must_return_an_exact_receipt(
    valid_config: Tuple[dict, Path, Path], invalid_receipt: object
) -> None:
    config, code_root, environment_root = valid_config

    _assert_invalid(
        config,
        code_root,
        environment_root,
        verifier=lambda _: invalid_receipt,
    )


def test_verifier_receipt_digest_must_match_inputs(
    valid_config: Tuple[dict, Path, Path],
) -> None:
    config, code_root, environment_root = valid_config

    _assert_invalid(
        config,
        code_root,
        environment_root,
        verifier=lambda _: InputVerificationReceipt("0" * 64),
    )


def test_verifier_coroutine_is_closed_and_rejected(
    valid_config: Tuple[dict, Path, Path],
) -> None:
    config, code_root, environment_root = valid_config
    returned = []

    async def async_receipt() -> InputVerificationReceipt:
        return _valid_receipt(config["inputs"])

    def verify(_: dict):
        coroutine = async_receipt()
        returned.append(coroutine)
        return coroutine

    try:
        _assert_invalid(config, code_root, environment_root, verifier=verify)
        assert returned[0].cr_frame is None
    finally:
        if returned and returned[0].cr_frame is not None:
            returned[0].close()


def test_verifier_exception_is_wrapped_with_its_cause(
    valid_config: Tuple[dict, Path, Path],
) -> None:
    config, code_root, environment_root = valid_config
    cause = ValueError("verification failed")

    def verify(_: dict) -> InputVerificationReceipt:
        raise cause

    with pytest.raises(CanaryContractError, match="input release verification failed") as exc:
        _validate(config, code_root, environment_root, verify)

    assert exc.value.__cause__ is cause


def test_public_config_validation_rejects_synthetic_inputs_at_fixed_production_pins(
    valid_config: Tuple[dict, Path, Path],
    verified_input_closure,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, code_root, environment_root = valid_config
    inputs, roots = verified_input_closure
    config["inputs"] = inputs
    monkeypatch.setattr(contract, "_DEFAULT_CODE_ROOT", code_root)
    monkeypatch.setattr(contract, "_DEFAULT_ENVIRONMENT_ROOT", environment_root)
    monkeypatch.setattr(contract, "_DEFAULT_REPO_ROOT", REPO_ROOT)
    monkeypatch.setattr(contract, "PERCEPTION_INPUT_RELEASE_ROOTS", roots)

    with pytest.raises(CanaryContractError):
        validate_canary_config(config)


@pytest.mark.parametrize(
    "operation", [validate_output_destination, reserve_output_destination]
)
def test_public_output_operations_reject_synthetic_inputs_at_production_pins(
    valid_config: Tuple[dict, Path, Path],
    tmp_path: Path,
    verified_input_closure,
    monkeypatch: pytest.MonkeyPatch,
    operation,
) -> None:
    config, code_root, environment_root = valid_config
    inputs, roots = verified_input_closure
    config["inputs"] = inputs
    output_root = tmp_path / "canaries"
    output_root.mkdir()
    monkeypatch.setattr(contract, "_DEFAULT_CODE_ROOT", code_root)
    monkeypatch.setattr(contract, "_DEFAULT_ENVIRONMENT_ROOT", environment_root)
    monkeypatch.setattr(contract, "_DEFAULT_REPO_ROOT", REPO_ROOT)
    monkeypatch.setattr(contract, "_DEFAULT_OUTPUT_ROOT", output_root)
    monkeypatch.setattr(contract, "PERCEPTION_INPUT_RELEASE_ROOTS", roots)

    with pytest.raises(CanaryContractError):
        operation(config, host_role="h200")


OUTPUT_CONFIG_BYTES = b'{"test":"output-identity"}\n'
OUTPUT_CONFIG_SHA256 = hashlib.sha256(OUTPUT_CONFIG_BYTES).hexdigest()
OUTPUT_IDENTITY = ValidatedCanaryIdentity(
    canonical_config_bytes=OUTPUT_CONFIG_BYTES,
    config_sha256=OUTPUT_CONFIG_SHA256,
    run_id=(
        "perc_lmh_v9p1_visible_stability_canary-" + OUTPUT_CONFIG_SHA256[:20]
    ),
)
OUTPUT_RUN_ID = OUTPUT_IDENTITY.run_id


def _validate_output(output_root: Path, **kwargs):
    return _validate_output_destination_for_test(
        kwargs.pop("identity", OUTPUT_IDENTITY),
        host_role=kwargs.pop("host_role", "h200"),
        output_root=output_root,
        **kwargs,
    )


def _reserve_output(output_root: Path, **kwargs):
    return _reserve_output_destination_for_test(
        kwargs.pop("identity", OUTPUT_IDENTITY),
        host_role=kwargs.pop("host_role", "h200"),
        output_root=output_root,
        **kwargs,
    )


def test_fresh_direct_output_is_validated_without_creation(tmp_path: Path) -> None:
    output_root = tmp_path / "canaries"
    output_root.mkdir()
    destination = output_root / OUTPUT_RUN_ID

    validated = _validate_output(output_root)

    assert validated == destination
    assert not destination.exists()


@pytest.mark.parametrize("host_role", ["linbox", "mac", "H200"])
def test_output_is_h200_only(tmp_path: Path, host_role: str) -> None:
    output_root = tmp_path / "canaries"
    output_root.mkdir()

    with pytest.raises(CanaryContractError):
        _validate_output(output_root, host_role=host_role)


def test_output_name_is_derived_from_validated_identity(tmp_path: Path) -> None:
    output_root = tmp_path / "canaries"
    output_root.mkdir()

    assert _validate_output(output_root) == output_root / OUTPUT_RUN_ID


def test_existing_output_is_rejected(tmp_path: Path) -> None:
    output_root = tmp_path / "canaries"
    destination = output_root / OUTPUT_RUN_ID
    destination.mkdir(parents=True)

    with pytest.raises(CanaryContractError):
        _validate_output(output_root)


def test_symlink_output_is_rejected_even_when_broken(tmp_path: Path) -> None:
    output_root = tmp_path / "canaries"
    output_root.mkdir()
    destination = output_root / OUTPUT_RUN_ID
    destination.symlink_to(tmp_path / "missing-target", target_is_directory=True)

    with pytest.raises(CanaryContractError):
        _validate_output(output_root)


@pytest.mark.parametrize(
    "root_builder",
    [
        lambda root: str(root / "nested" / ".."),
        lambda root: root / "current",
        lambda root: Path("relative-canaries"),
    ],
)
def test_output_rejects_traversal_current_and_relative_roots(
    tmp_path: Path, root_builder
) -> None:
    output_root = tmp_path / "canaries"
    output_root.mkdir()

    with pytest.raises(CanaryContractError):
        _validate_output(root_builder(output_root))


def test_public_output_apis_have_no_test_root_switches() -> None:
    assert tuple(inspect.signature(validate_output_destination).parameters) == (
        "config_or_canonical_bytes",
        "host_role",
    )
    assert tuple(inspect.signature(reserve_output_destination).parameters) == (
        "config_or_canonical_bytes",
        "host_role",
    )


@pytest.mark.parametrize(
    "identity",
    [
        ValidatedCanaryIdentity(OUTPUT_CONFIG_BYTES, "0" * 64, OUTPUT_IDENTITY.run_id),
        ValidatedCanaryIdentity(
            OUTPUT_CONFIG_BYTES,
            OUTPUT_CONFIG_SHA256,
            "perc_lmh_v9p1_visible_stability_canary-" + "0" * 20,
        ),
    ],
)
def test_private_output_helper_rejects_forged_validated_identity(
    tmp_path: Path, identity: ValidatedCanaryIdentity
) -> None:
    output_root = tmp_path / "canaries"
    output_root.mkdir()

    with pytest.raises(CanaryContractError):
        _validate_output_destination_for_test(
            identity,
            host_role="h200",
            output_root=output_root,
        )


def test_symlinked_output_root_is_rejected(tmp_path: Path) -> None:
    real_root = tmp_path / "real-canaries"
    real_root.mkdir()
    output_root = tmp_path / "canaries"
    output_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(CanaryContractError):
        _validate_output(output_root)


@pytest.mark.parametrize("mode", [0o700, 0o755])
def test_atomic_reservation_returns_owned_directory_fd(
    tmp_path: Path, mode: int
) -> None:
    output_root = tmp_path / "canaries"
    output_root.mkdir()
    output_root.chmod(mode)
    destination = output_root / OUTPUT_RUN_ID

    reservation = _reserve_output(output_root)
    assert isinstance(reservation, ReservedOutputDirectory)
    with reservation as reserved:
        opened = os.fstat(reserved.directory_fd)
        named = os.stat(destination, follow_symlinks=False)
        assert (opened.st_dev, opened.st_ino) == (named.st_dev, named.st_ino)
        evidence_fd = os.open(
            "evidence.txt",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=reserved.directory_fd,
        )
        try:
            os.write(evidence_fd, b"partial evidence\n")
        finally:
            os.close(evidence_fd)

    assert destination.joinpath("evidence.txt").read_bytes() == b"partial evidence\n"
    with pytest.raises(OSError):
        reservation.fileno()


@pytest.mark.parametrize("mode", [0o775, 0o777])
def test_reservation_rejects_group_or_world_writable_root_before_creation(
    tmp_path: Path, mode: int
) -> None:
    output_root = tmp_path / "canaries"
    output_root.mkdir(mode=mode)
    output_root.chmod(mode)
    accepted = None

    try:
        with pytest.raises(CanaryContractError, match="owner-controlled"):
            accepted = _reserve_output(output_root)
    finally:
        if accepted is not None:
            accepted.close()

    assert list(output_root.iterdir()) == []


def test_reservation_rejects_wrong_output_root_owner_before_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "canaries"
    output_root.mkdir(mode=0o700)
    real_fstat = os.fstat
    accepted = None

    def wrong_owner(root_fd: int):
        info = real_fstat(root_fd)
        return SimpleNamespace(
            st_dev=info.st_dev,
            st_ino=info.st_ino,
            st_mode=info.st_mode,
            st_uid=info.st_uid + 1,
        )

    monkeypatch.setattr(contract, "_output_root_stat", wrong_owner, raising=False)

    try:
        with pytest.raises(CanaryContractError, match="owner-controlled"):
            accepted = _reserve_output(output_root)
    finally:
        if accepted is not None:
            accepted.close()

    assert list(output_root.iterdir()) == []


def test_reservation_rejects_when_effective_uid_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "canaries"
    output_root.mkdir(mode=0o700)
    monkeypatch.delattr(contract.os, "geteuid")

    with pytest.raises(CanaryContractError, match="effective UID"):
        _reserve_output(output_root)

    assert list(output_root.iterdir()) == []


def test_reservation_rejects_prepublication_root_mode_change_and_preserves_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "canaries"
    output_root.mkdir(mode=0o700)
    accepted = None

    def make_untrusted(root_fd: int, _: str, __: str) -> None:
        os.fchmod(root_fd, 0o777)

    monkeypatch.setattr(contract, "_before_output_publication", make_untrusted)

    try:
        with pytest.raises(CanaryContractError, match="authority changed"):
            accepted = _reserve_output(output_root)
    finally:
        if accepted is not None:
            accepted.close()

    assert not (output_root / OUTPUT_RUN_ID).exists()
    assert len(list(output_root.iterdir())) == 1


def test_reservation_rejects_postpublication_root_mode_change_and_preserves_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "canaries"
    output_root.mkdir(mode=0o700)
    real_publish = contract._rename_noreplace_at
    accepted = None

    def publish_then_make_untrusted(
        source_fd: int,
        source_name: str,
        destination_fd: int,
        destination_name: str,
    ) -> None:
        real_publish(
            source_fd,
            source_name,
            destination_fd,
            destination_name,
        )
        os.fchmod(source_fd, 0o777)

    monkeypatch.setattr(
        contract, "_rename_noreplace_at", publish_then_make_untrusted
    )

    try:
        with pytest.raises(CanaryContractError, match="authority changed"):
            accepted = _reserve_output(output_root)
    finally:
        if accepted is not None:
            accepted.close()

    assert (output_root / OUTPUT_RUN_ID).is_dir()
    assert list(output_root.iterdir()) == [output_root / OUTPUT_RUN_ID]


def test_untrusted_root_is_rejected_before_empty_replacement_race_can_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "canaries"
    output_root.mkdir(mode=0o777)
    output_root.chmod(0o777)
    attack_ran = []
    real_mkdir = contract._mkdir_output_child_at

    def empty_replacement(root_fd: int, name: str):
        attack_ran.append(True)
        result = real_mkdir(root_fd, name)
        os.rename(name, name + "-stolen", src_dir_fd=root_fd, dst_dir_fd=root_fd)
        os.mkdir(name, 0o700, dir_fd=root_fd)
        return result

    monkeypatch.setattr(contract, "_mkdir_output_child_at", empty_replacement)

    with pytest.raises(CanaryContractError, match="owner-controlled"):
        _reserve_output(output_root)

    assert attack_ran == []
    assert list(output_root.iterdir()) == []


@pytest.mark.parametrize("entry_kind", ["directory", "symlink"])
def test_atomic_reservation_rejects_preplaced_entries(
    tmp_path: Path, entry_kind: str
) -> None:
    output_root = tmp_path / "canaries"
    output_root.mkdir()
    destination = output_root / OUTPUT_RUN_ID
    if entry_kind == "directory":
        destination.mkdir()
    else:
        destination.symlink_to(tmp_path / "outside", target_is_directory=True)

    with pytest.raises(CanaryContractError, match="already reserved"):
        _reserve_output(output_root)


def test_reserved_fd_remains_authoritative_after_path_replacement(tmp_path: Path) -> None:
    output_root = tmp_path / "canaries"
    output_root.mkdir()
    destination = output_root / OUTPUT_RUN_ID
    parked = output_root / "parked"

    with _reserve_output(output_root) as reserved:
        destination.rename(parked)
        destination.mkdir()
        evidence_fd = os.open(
            "fd-only.txt",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=reserved.fileno(),
        )
        os.close(evidence_fd)

    assert (parked / "fd-only.txt").is_file()
    assert not (destination / "fd-only.txt").exists()


def test_reservation_preserves_partial_output_when_context_raises(tmp_path: Path) -> None:
    output_root = tmp_path / "canaries"
    output_root.mkdir()
    destination = output_root / OUTPUT_RUN_ID

    with pytest.raises(RuntimeError, match="training stopped"):
        with _reserve_output(output_root) as reserved:
            evidence_fd = os.open(
                "partial.txt",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=reserved.fileno(),
            )
            os.close(evidence_fd)
            raise RuntimeError("training stopped")

    assert (destination / "partial.txt").is_file()


def test_atomic_competitor_is_preserved_and_never_followed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "canaries"
    output_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = output_root / OUTPUT_RUN_ID
    real_mkdir = contract._mkdir_output_child_at
    raced_names = []

    def insert_competitor(root_fd: int, name: str) -> None:
        raced_names.append(name)
        os.symlink(str(outside), name, dir_fd=root_fd)
        real_mkdir(root_fd, name)

    monkeypatch.setattr(contract, "_mkdir_output_child_at", insert_competitor)

    with pytest.raises(CanaryContractError, match="staging output name collision"):
        _reserve_output(output_root)

    assert not destination.exists()
    assert (output_root / raced_names[0]).is_symlink()
    assert list(outside.iterdir()) == []


def test_reservation_rejects_canonical_root_rebind_without_touching_outside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "canaries"
    output_root.mkdir()
    parked = tmp_path / "parked-canaries"
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = output_root / OUTPUT_RUN_ID

    def rebind_root(_: Path, __: int) -> None:
        output_root.rename(parked)
        output_root.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(contract, "_reservation_root_pinned", rebind_root)

    with pytest.raises(CanaryContractError, match="root binding changed"):
        _reserve_output(output_root)

    assert list(outside.iterdir()) == []
    assert not (parked / OUTPUT_RUN_ID).exists()


def test_setup_cleanup_failure_never_masks_reservation_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "canaries"
    output_root.mkdir()
    parked = tmp_path / "parked-canaries"
    outside = tmp_path / "outside"
    outside.mkdir()

    def rebind_root(_: Path, __: int) -> None:
        output_root.rename(parked)
        output_root.symlink_to(outside, target_is_directory=True)

    def cleanup_failure(*_: object) -> None:
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(contract, "_reservation_root_pinned", rebind_root)
    monkeypatch.setattr(contract, "_cleanup_owned_empty_child", cleanup_failure)

    with pytest.raises(CanaryContractError, match="root binding changed"):
        _reserve_output(output_root)


def test_mkdir_to_open_replacement_is_not_followed_or_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "canaries"
    output_root.mkdir()
    destination = output_root / OUTPUT_RUN_ID
    stolen_name = "stolen-created-directory"
    raced_names = []

    def replace_created_child(root_fd: int, name: str) -> None:
        raced_names.append(name)
        os.rename(name, stolen_name, src_dir_fd=root_fd, dst_dir_fd=root_fd)
        os.mkdir(name, 0o700, dir_fd=root_fd)

    monkeypatch.setattr(contract, "_output_child_created", replace_created_child)

    with pytest.raises(CanaryContractError):
        _reserve_output(output_root)

    assert not destination.exists()
    assert (output_root / raced_names[0]).is_dir()
    assert (output_root / stolen_name).is_dir()


def test_mkdir_helper_replacement_before_return_never_yields_attacker_fd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "canaries"
    output_root.mkdir()
    destination = output_root / OUTPUT_RUN_ID
    stolen_name = "stolen-before-helper-return"
    real_mkdir = contract._mkdir_output_child_at
    accepted = None
    raced_names = []

    def replace_before_return(root_fd: int, name: str):
        raced_names.append(name)
        result = real_mkdir(root_fd, name)
        os.rename(name, stolen_name, src_dir_fd=root_fd, dst_dir_fd=root_fd)
        os.mkdir(name, 0o700, dir_fd=root_fd)
        replacement_fd = os.open(name, contract._directory_flags(), dir_fd=root_fd)
        try:
            attacker_fd = os.open(
                "attacker.txt",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=replacement_fd,
            )
            os.close(attacker_fd)
        finally:
            os.close(replacement_fd)
        return result

    monkeypatch.setattr(contract, "_mkdir_output_child_at", replace_before_return)

    try:
        with pytest.raises(CanaryContractError):
            accepted = _reserve_output(output_root)
    finally:
        if accepted is not None:
            accepted.close()

    assert not destination.exists()
    assert (output_root / raced_names[0] / "attacker.txt").is_file()
    assert (output_root / stolen_name).is_dir()
