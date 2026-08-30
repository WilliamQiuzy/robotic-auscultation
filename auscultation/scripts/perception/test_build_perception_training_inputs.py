import importlib.util
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import numpy as np

from ausc.training import perception_canary_contract as canary
from scripts.deployment import artifact_contract
from scripts.perception import build_perception_training_inputs as builder


POINT_ORDER = ("R1", "L1", "R2", "L2", "R3", "L3", "R4", "L4", "R5", "L5")


def test_builder_module_exists() -> None:
    assert importlib.util.find_spec(
        "scripts.perception.build_perception_training_inputs"
    ) is not None


def _canonical(value: object) -> bytes:
    return artifact_contract.canonical_json_bytes(value)


def _geometry_fixture(tmp_path: Path) -> tuple:
    data_root = tmp_path / "data"
    licenses_root = tmp_path / "licenses-source"
    releases_root = tmp_path / "geometry-releases"
    staging_root = tmp_path / ".staging"
    sequence_ids = tuple(
        sorted(
            "p%06d_a%06d" % (index % 340, index // 340)
            for index in range(907)
        )
    )
    for sequence_id in sequence_ids:
        path = (
            data_root
            / "humman"
            / "smpl_params"
            / sequence_id
            / "smpl_params.npz"
        )
        path.parent.mkdir(parents=True)
        path.write_bytes(("smpl:" + sequence_id).encode("ascii"))
    support_bodies = {
        "smpl/SMPL_NEUTRAL.pkl": b"neutral",
        "densepose/UV_Processed.mat": b"uv-map",
        "licenses/HUMMAN_LICENSE.txt": b"humman-license",
        "licenses/SMPL_MODEL_LICENSE.html": b"smpl-license",
        "licenses/DENSEPOSE_SOURCE_LICENSE.txt": b"densepose-license",
    }
    (data_root / "smpl").mkdir(parents=True)
    (data_root / "smpl" / "SMPL_NEUTRAL.pkl").write_bytes(
        support_bodies["smpl/SMPL_NEUTRAL.pkl"]
    )
    densepose = data_root / "densepose_coco" / "uv_data"
    densepose.mkdir(parents=True)
    (densepose / "UV_Processed.mat").write_bytes(
        support_bodies["densepose/UV_Processed.mat"]
    )
    licenses_root.mkdir()
    for relative, body in support_bodies.items():
        if not relative.startswith("licenses/"):
            continue
        (licenses_root / Path(relative).name).write_bytes(body)
    support_sha256 = {
        relative: hashlib.sha256(body).hexdigest()
        for relative, body in support_bodies.items()
    }
    manifest_rows = []
    for sequence_id in sequence_ids:
        body = ("smpl:" + sequence_id).encode("ascii")
        manifest_rows.append(
            "%s  %s/smpl_params.npz\n"
            % (hashlib.sha256(body).hexdigest(), sequence_id)
        )
    smpl_manifest_sha256 = hashlib.sha256(
        "".join(manifest_rows).encode("ascii")
    ).hexdigest()
    source_terms = canary._build_geometry_source_terms_expected(
        support_sha256,
        smpl_manifest_sha256,
    )
    return (
        data_root,
        licenses_root,
        releases_root,
        staging_root,
        support_sha256,
        smpl_manifest_sha256,
        source_terms,
    )


def _build_geometry_fixture(tmp_path: Path) -> tuple:
    fixture = _geometry_fixture(tmp_path)
    (
        data_root,
        licenses_root,
        releases_root,
        staging_root,
        support_sha256,
        smpl_manifest_sha256,
        source_terms,
    ) = fixture
    release = builder._build_geometry_source_release(
        data_root=data_root,
        licenses_root=licenses_root,
        releases_root=releases_root,
        staging_root=staging_root,
        expected_support_sha256=support_sha256,
        expected_smpl_manifest_sha256=smpl_manifest_sha256,
        expected_source_terms=source_terms,
    )
    return release, fixture


def test_geometry_builder_public_api_has_no_expected_hash_or_test_bypass() -> None:
    assert tuple(inspect.signature(builder.build_geometry_source_release).parameters) == (
        "data_root",
        "licenses_root",
        "releases_root",
    )
    assert tuple(inspect.signature(builder.verify_geometry_source_release).parameters) == (
        "release_dir",
    )


def test_geometry_builder_seals_exact_content_addressed_closure(tmp_path: Path) -> None:
    release, fixture = _build_geometry_fixture(tmp_path)
    (
        _data_root,
        _licenses_root,
        _releases_root,
        staging_root,
        support_sha256,
        smpl_manifest_sha256,
        source_terms,
    ) = fixture

    envelope = artifact_contract.verify_release(release)
    assert release.name == envelope["release_id"]
    assert envelope["payload"]["artifact_kind"] == "perception_geometry_source"
    assert envelope["payload"]["source_role"] == "mac"
    assert (release / artifact_contract.COMPLETE_NAME).is_file()
    assert list(staging_root.iterdir()) == []

    semantic_bytes = (release / canary.PERCEPTION_INPUT_NAME).read_bytes()
    semantic = json.loads(semantic_bytes)
    assert semantic_bytes == _canonical(semantic)
    assert semantic["contract"] == {
        "densepose_uv_count": 1,
        "license_count": 3,
        "provenance_count": 1,
        "raw_rgb_count": 0,
        "smpl_neutral_count": 1,
        "smpl_params_count": 907,
    }
    assert len(semantic["members"]) == 913
    assert all("rgb" not in member["path"].lower() for member in semantic["members"])
    assert json.loads(
        (release / "provenance" / "SOURCE_TERMS.json").read_bytes()
    ) == source_terms

    report = builder._verify_geometry_source_release(
        release,
        expected_support_sha256=support_sha256,
        expected_smpl_manifest_sha256=smpl_manifest_sha256,
        expected_source_terms=source_terms,
    )
    assert report["release_id"] == release.name
    assert report["sequence_count"] == 907


@pytest.mark.parametrize("poison", ("missing", "extra", "raw_rgb", "symlink"))
def test_geometry_builder_rejects_nonexact_or_unsafe_source_closure(
    tmp_path: Path,
    poison: str,
) -> None:
    fixture = _geometry_fixture(tmp_path)
    (
        data_root,
        licenses_root,
        releases_root,
        staging_root,
        support_sha256,
        smpl_manifest_sha256,
        source_terms,
    ) = fixture
    first = next(
        (data_root / "humman" / "smpl_params").glob("*/smpl_params.npz")
    )
    if poison == "missing":
        first.unlink()
    elif poison == "extra":
        (first.parent / "unexpected.bin").write_bytes(b"extra")
    elif poison == "raw_rgb":
        (first.parent / "frame.jpg").write_bytes(b"rgb")
    else:
        body = first.read_bytes()
        target = tmp_path / "outside.npz"
        target.write_bytes(body)
        first.unlink()
        first.symlink_to(target)

    with pytest.raises(builder.InputBuildError):
        builder._build_geometry_source_release(
            data_root=data_root,
            licenses_root=licenses_root,
            releases_root=releases_root,
            staging_root=staging_root,
            expected_support_sha256=support_sha256,
            expected_smpl_manifest_sha256=smpl_manifest_sha256,
            expected_source_terms=source_terms,
        )

    assert not releases_root.exists() or list(releases_root.iterdir()) == []
    assert not staging_root.exists() or list(staging_root.iterdir()) == []


def test_geometry_verifier_rejects_post_seal_extra_file(tmp_path: Path) -> None:
    release, fixture = _build_geometry_fixture(tmp_path)
    support_sha256 = fixture[4]
    smpl_manifest_sha256 = fixture[5]
    source_terms = fixture[6]
    (release / "raw.jpg").write_bytes(b"forbidden")

    with pytest.raises(builder.InputBuildError):
        builder._verify_geometry_source_release(
            release,
            expected_support_sha256=support_sha256,
            expected_smpl_manifest_sha256=smpl_manifest_sha256,
            expected_source_terms=source_terms,
        )


def _source_frames_bytes(*, session_number: int = 2) -> bytes:
    rows = ["idx,t,cam_index,has_depth,depth_scale"]
    rows.extend(
        "%d,%.9f,%d,1,1.0"
        % (index, index / 30.0 + (session_number - 2) * 0.000001, index)
        for index in range(602)
    )
    return ("\n".join(rows) + "\n").encode("utf-8")


def _write_tiny_video(path: Path, *, offset: int) -> None:
    import cv2

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0,
        (64, 48),
    )
    assert writer.isOpened()
    try:
        for index in range(602):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[..., 0] = (index + offset) % 251
            frame[..., 1] = np.arange(64, dtype=np.uint8)[None, :]
            frame[..., 2] = (
                np.arange(48, dtype=np.uint16)[:, None] + offset
            ) % 251
            writer.write(frame)
    finally:
        writer.release()


def _write_mixed_eval_fixture(tmp_path: Path, *, collide_rgb: bool = False) -> tuple:
    source = tmp_path / "mixed-source"
    source.mkdir()
    intrinsics = _canonical(
        {
            "height": 48,
            "intrinsics_px": [[50.0, 0.0, 32.0], [0.0, 50.0, 24.0], [0, 0, 1]],
            "width": 64,
        }
    )
    for session_number in (2, 3):
        frames = _source_frames_bytes(session_number=session_number)
        base = source / ("orbbec_session%d" % session_number)
        (base / "depth").mkdir(parents=True)
        _write_tiny_video(
            base / "rgb.mp4",
            offset=0 if session_number == 2 or collide_rgb else 73,
        )
        (base / "frames.csv").write_bytes(frames)
        (base / "intrinsics.json").write_bytes(intrinsics)
        for index in range(602):
            (base / "depth" / ("d%04d.npy" % index)).write_bytes(
                ("depth-%d-%04d" % (session_number, index)).encode("ascii")
            )
        for index in range(2 if session_number == 2 else 3):
            (base / ("diagnostic_%d.png" % index)).write_bytes(
                ("diagnostic-%d-%d" % (session_number, index)).encode("ascii")
            )
    paths = sorted(
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file()
    )
    manifest = artifact_contract.build_manifest(
        source,
        paths,
        artifact_kind="perception_eval",
        source_role="linbox",
    )
    release = tmp_path / "mixed-releases" / manifest["release_id"]
    artifact_contract.write_release(
        source,
        manifest,
        release,
        target_role="mac",
    )
    manifest_sha256 = hashlib.sha256(
        (release / artifact_contract.MANIFEST_NAME).read_bytes()
    ).hexdigest()
    identity = {
        "payload_sha256": manifest["payload_sha256"],
        "release_id": manifest["release_id"],
        "release_manifest_sha256": manifest_sha256,
    }
    return release, identity


def _label_document(
    *,
    session_number: int,
    garment: str,
    width: int = 64,
    height: int = 48,
) -> dict:
    prefix = "orb" if session_number == 2 else "orb3"
    domain = "clothed-orbbec" if garment == "dark" else "clothed-orbbec-yellow"
    document = {
        "_meta": {
            "annotator": "william",
            "domain": domain,
            "schema": "R1,L1..R5,L5 (R=图右 L=图左)",
            "space": "FULL native frame pixels",
        }
    }
    for ordinal, frame_index in enumerate(
        (60, 83, 107, 131, 155, 179, 203, 227, 251, 275, 299,
         322, 346, 370, 394, 418, 442, 466, 490, 514, 538, 562)
    ):
        document["%s_%04d" % (prefix, frame_index)] = {
            "anchors": [[1, 1], [2, 2], [3, 3]],
            "clip": "orbbec_session%d" % session_number,
            "frame": frame_index,
            "h": height,
            "points": {
                name: [float(5 + index * 4), float(6 + (index % 5) * 6)]
                for index, name in enumerate(POINT_ORDER)
            },
            "w": width,
        }
    return document


def _test_session_spec(
    *,
    session_number: int,
    garment: str,
    label_sha256: str,
    mixed_eval_release: Path,
) -> object:
    envelope = artifact_contract.verify_release(mixed_eval_release)
    by_path = {
        row["path"]: row for row in envelope["payload"]["files"]
    }
    base = "orbbec_session%d" % session_number
    return builder._SessionSpec(
        annotation_artifact_kind=(
            "perception_labels_train"
            if session_number == 2
            else "perception_labels_holdout"
        ),
        diagnostic_png_count=2 if session_number == 2 else 3,
        garment=garment,
        height_px=48,
        intrinsics_sha256=by_path[base + "/intrinsics.json"]["sha256"],
        label_domain=(
            "clothed-orbbec" if session_number == 2 else "clothed-orbbec-yellow"
        ),
        label_path="labels/%s.json" % garment,
        label_sha256=label_sha256,
        projection_artifact_kind=(
            "perception_session2_train"
            if session_number == 2
            else "perception_session3_holdout"
        ),
        rgb_sha256=by_path[base + "/rgb.mp4"]["sha256"],
        session_id=base,
        session_number=session_number,
        source_frames_sha256=by_path[base + "/frames.csv"]["sha256"],
        subject_pseudonym=(
            "same-subject-001"
        ),
        width_px=64,
    )


@pytest.fixture(scope="module")
def mixed_eval_fixture(tmp_path_factory):
    return _write_mixed_eval_fixture(tmp_path_factory.mktemp("mixed-eval"))


def test_annotation_public_api_has_no_expected_hash_or_test_bypass() -> None:
    assert tuple(
        inspect.signature(builder.build_session2_annotation_release).parameters
    ) == ("label_path", "mixed_eval_manifest", "releases_root")
    assert tuple(
        inspect.signature(builder.build_session3_annotation_release).parameters
    ) == ("label_path", "mixed_eval_manifest", "releases_root")
    assert tuple(
        inspect.signature(builder.verify_session2_annotation_release).parameters
    ) == ("release_dir", "mixed_eval_manifest")
    assert tuple(
        inspect.signature(builder.verify_session3_annotation_release).parameters
    ) == ("release_dir", "mixed_eval_manifest")


@pytest.mark.parametrize(
    ("session_number", "garment"),
    ((2, "dark"), (3, "yellow")),
)
def test_annotation_builder_binds_label_and_exact_selected_mixed_eval_members(
    tmp_path: Path,
    mixed_eval_fixture,
    session_number: int,
    garment: str,
) -> None:
    mixed_release, mixed_identity = mixed_eval_fixture
    label_document = _label_document(
        session_number=session_number,
        garment=garment,
    )
    label_bytes = json.dumps(
        label_document,
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8") + b"\n"
    label_path = tmp_path / (garment + ".json")
    label_path.write_bytes(label_bytes)
    spec = _test_session_spec(
        session_number=session_number,
        garment=garment,
        label_sha256=hashlib.sha256(label_bytes).hexdigest(),
        mixed_eval_release=mixed_release,
    )
    releases_root = tmp_path / "annotation-releases"
    staging_root = tmp_path / ".staging"

    release = builder._build_annotation_release(
        label_path=label_path,
        mixed_eval_manifest=mixed_release / artifact_contract.MANIFEST_NAME,
        releases_root=releases_root,
        staging_root=staging_root,
        spec=spec,
        expected_mixed_eval_identity=mixed_identity,
    )

    envelope = artifact_contract.verify_release(release)
    assert envelope["payload"]["source_role"] == "mac"
    assert envelope["payload"]["artifact_kind"] == spec.annotation_artifact_kind
    semantic = json.loads((release / canary.PERCEPTION_INPUT_NAME).read_bytes())
    attestation = semantic["source_attestations"][0]
    assert attestation["source_release_id"] == mixed_identity["release_id"]
    assert len(attestation["selected_members"]) == 605
    assert all(
        row["path"].startswith(spec.session_id + "/")
        for row in attestation["selected_members"]
    )
    assert not any(
        row["path"].endswith(".png")
        for row in attestation["selected_members"]
    )
    assert list(staging_root.iterdir()) == []

    report = builder._verify_annotation_release(
        release,
        mixed_eval_manifest=mixed_release / artifact_contract.MANIFEST_NAME,
        spec=spec,
        expected_mixed_eval_identity=mixed_identity,
    )
    assert report["labeled_frame_count"] == 22
    assert report["selected_member_count"] == 605


@pytest.mark.parametrize(
    "poison",
    (
        "frame_outside_video",
        "wrong_dimensions",
        "duplicate_frame",
        "missing_point",
        "nonfinite_coordinate",
        "out_of_frame_coordinate",
        "unknown_point",
        "wrong_clip",
        "wrong_domain",
    ),
)
def test_label_validator_rejects_invalid_video_binding(poison: str) -> None:
    document = _label_document(session_number=2, garment="dark")
    keys = [key for key in document if key != "_meta"]
    first = document[keys[0]]
    if poison == "frame_outside_video":
        first["frame"] = 602
    elif poison == "wrong_dimensions":
        first["w"] = 63
    elif poison == "duplicate_frame":
        document[keys[1]]["frame"] = first["frame"]
    elif poison == "missing_point":
        first["points"].pop("R1")
    elif poison == "nonfinite_coordinate":
        first["points"]["R1"][0] = float("nan")
    elif poison == "out_of_frame_coordinate":
        first["points"]["R1"][0] = 64.0
    elif poison == "unknown_point":
        first["points"]["X1"] = [1.0, 1.0]
    elif poison == "wrong_clip":
        first["clip"] = "orbbec_session3"
    else:
        document["_meta"]["domain"] = "clothed-orbbec-yellow"
    raw = json.dumps(document, ensure_ascii=False).encode("utf-8")
    simple_spec = type(
        "SimpleSpec",
        (),
        {
            "garment": "dark",
            "height_px": 48,
            "label_domain": "clothed-orbbec",
            "session_id": "orbbec_session2",
            "width_px": 64,
        },
    )

    with pytest.raises(builder.InputBuildError):
        builder._validate_label_bytes(raw, simple_spec)


def _decoded_video_hashes(path: Path, *, width: int, height: int) -> list:
    import cv2

    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened()
    header = _canonical(
        {
            "dtype": "uint8",
            "shape": [height, width, 3],
            "version": canary.DECODED_FRAME_HASH_VERSION,
        }
    )
    hashes = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            assert tuple(frame.shape) == (height, width, 3)
            assert frame.dtype == np.uint8
            hashes.append(
                hashlib.sha256(
                    header + np.ascontiguousarray(frame).tobytes(order="C")
                ).hexdigest()
            )
    finally:
        capture.release()
    assert len(hashes) == 602
    return hashes


def _coordinate_metadata(spec) -> dict:
    return {
        "decoded_frame_hash_version": canary.DECODED_FRAME_HASH_VERSION,
        "depth_unit": "metres",
        "garment": spec.garment,
        "height_px": spec.height_px,
        "landmark_order": list(POINT_ORDER),
        "pixel_axis_order": "xy",
        "pixel_index_base": 0,
        "pixel_origin": "top_left",
        "rgb_depth_registration": "registered",
        "schema": "wam_perception_session%d_coordinate_metadata_v1"
        % spec.session_number,
        "session_id": spec.session_id,
        "subject_pseudonym": spec.subject_pseudonym,
        "width_px": spec.width_px,
    }


def _projection_pins(
    mixed_release: Path,
    mixed_identity: dict,
    spec,
) -> object:
    envelope = artifact_contract.verify_release(mixed_release)
    rows = envelope["payload"]["files"]
    by_path = {row["path"]: row for row in rows}
    selected2 = sorted(
        {
            row["sha256"]
            for row in rows
            if row["path"].startswith("orbbec_session2/")
            and not row["path"].endswith(".png")
        }
    )
    denied3 = sorted(
        {
            row["sha256"]
            for row in rows
            if row["path"].startswith("orbbec_session3/")
        }
    )
    decoded2 = _decoded_video_hashes(
        mixed_release / "orbbec_session2" / "rgb.mp4",
        width=spec.width_px,
        height=spec.height_px,
    )
    decoded3 = _decoded_video_hashes(
        mixed_release / "orbbec_session3" / "rgb.mp4",
        width=spec.width_px,
        height=spec.height_px,
    )
    shared = [by_path["orbbec_session2/intrinsics.json"]["sha256"]]
    partition = {
        "decoded_frame_hash_version": canary.DECODED_FRAME_HASH_VERSION,
        "schema": canary.MIXED_EVAL_CONTENT_PARTITION_SCHEMA,
        "session2_decoded_frame_sha256": decoded2,
        "session2_selected_sha256": selected2,
        "session3_decoded_frame_sha256": decoded3,
        "session3_denied_sha256": denied3,
        "shared_intrinsics_sha256": shared,
        "source_manifest_sha256": mixed_identity["release_manifest_sha256"],
        "source_payload_sha256": mixed_identity["payload_sha256"],
        "source_release_id": mixed_identity["release_id"],
    }
    list_sha256 = {
        key: hashlib.sha256(_canonical(partition[key])).hexdigest()
        for key in (
            "session2_decoded_frame_sha256",
            "session2_selected_sha256",
            "session3_decoded_frame_sha256",
            "session3_denied_sha256",
            "shared_intrinsics_sha256",
        )
    }
    ordered_depth = [
        by_path[
            "%s/depth/d%04d.npy" % (spec.session_id, index)
        ]["sha256"]
        for index in range(602)
    ]
    coordinate_sha256 = hashlib.sha256(
        _canonical(_coordinate_metadata(spec))
    ).hexdigest()
    return builder._ProjectionPins(
        coordinate_metadata_sha256=coordinate_sha256,
        ordered_depth_list_sha256=hashlib.sha256(
            _canonical(ordered_depth)
        ).hexdigest(),
        partition_list_sha256=list_sha256,
        partition_sha256=hashlib.sha256(_canonical(partition)).hexdigest(),
    )


@pytest.fixture(scope="module")
def projection_input_fixture(tmp_path_factory, mixed_eval_fixture):
    root = tmp_path_factory.mktemp("projection-inputs")
    mixed_release, mixed_identity = mixed_eval_fixture
    specs = {}
    annotations = {}
    pins = {}
    for session_number, garment in ((2, "dark"), (3, "yellow")):
        label_document = _label_document(
            session_number=session_number,
            garment=garment,
        )
        label_bytes = json.dumps(
            label_document,
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8") + b"\n"
        label_path = root / (garment + ".json")
        label_path.write_bytes(label_bytes)
        spec = _test_session_spec(
            session_number=session_number,
            garment=garment,
            label_sha256=hashlib.sha256(label_bytes).hexdigest(),
            mixed_eval_release=mixed_release,
        )
        annotation = builder._build_annotation_release(
            label_path=label_path,
            mixed_eval_manifest=mixed_release / artifact_contract.MANIFEST_NAME,
            releases_root=root / ("annotation-releases-%d" % session_number),
            staging_root=root / (".annotation-staging-%d" % session_number),
            spec=spec,
            expected_mixed_eval_identity=mixed_identity,
        )
        specs[session_number] = spec
        annotations[session_number] = annotation
        pins[session_number] = _projection_pins(
            mixed_release,
            mixed_identity,
            spec,
        )
    return root, mixed_release, mixed_identity, specs, annotations, pins


def test_projection_public_api_has_no_expected_hash_or_test_bypass() -> None:
    assert tuple(
        inspect.signature(builder.build_session2_projection_release).parameters
    ) == ("mixed_eval_release", "annotation_release", "releases_root")
    assert tuple(
        inspect.signature(builder.build_session3_projection_release).parameters
    ) == ("mixed_eval_release", "annotation_release", "releases_root")
    assert tuple(
        inspect.signature(builder.verify_session2_projection_release).parameters
    ) == ("release_dir", "mixed_eval_release", "annotation_release")
    assert tuple(
        inspect.signature(builder.verify_session3_projection_release).parameters
    ) == ("release_dir", "mixed_eval_release", "annotation_release")


def test_projection_builders_seal_self_contained_pair_without_cross_session_content(
    tmp_path: Path,
    projection_input_fixture,
) -> None:
    (
        _root,
        mixed_release,
        mixed_identity,
        specs,
        annotations,
        pins,
    ) = projection_input_fixture
    releases = {}
    staging_roots = {}
    for session_number in (2, 3):
        staging = tmp_path / (".projection-staging-%d" % session_number)
        staging_roots[session_number] = staging
        releases[session_number] = builder._build_projection_release(
            mixed_eval_release=mixed_release,
            annotation_release=annotations[session_number],
            releases_root=tmp_path / ("projection-releases-%d" % session_number),
            staging_root=staging,
            spec=specs[session_number],
            peer_spec=specs[3 if session_number == 2 else 2],
            expected_mixed_eval_identity=mixed_identity,
            pins=pins[session_number],
            authoring_authorization=builder._test_h200_authorization(),
        )
        envelope = artifact_contract.verify_release(releases[session_number])
        assert envelope["payload"]["source_role"] == "h200"
        assert envelope["payload"]["artifact_kind"] == specs[
            session_number
        ].projection_artifact_kind
        assert len(envelope["payload"]["files"]) == 609
        semantic = json.loads(
            (releases[session_number] / canary.PERCEPTION_INPUT_NAME).read_bytes()
        )
        assert semantic["self_contained"] is True
        assert semantic["resolvable_dependencies"] == []
        assert len(
            [row for row in semantic["members"] if row["role"] == "metric_depth"]
        ) == 602
        assert not any(
            ("session3" if session_number == 2 else "session2") in row["path"]
            for row in envelope["payload"]["files"]
        )
        assert list(staging.iterdir()) == []

    pair = builder._verify_projection_pair(
        session2_release=releases[2],
        session3_release=releases[3],
        mixed_eval_release=mixed_release,
        session2_annotation_release=annotations[2],
        session3_annotation_release=annotations[3],
        session2_spec=specs[2],
        session3_spec=specs[3],
        expected_mixed_eval_identity=mixed_identity,
        session2_pins=pins[2],
        session3_pins=pins[3],
    )
    assert pair["shared_content_sha256"] == [specs[2].intrinsics_sha256]
    assert pair["session2_file_count"] == 609
    assert pair["session3_file_count"] == 609


def test_content_partition_rejects_cross_session_decoded_video_collision(
    tmp_path: Path,
) -> None:
    mixed_release, mixed_identity = _write_mixed_eval_fixture(
        tmp_path,
        collide_rgb=True,
    )
    envelope = artifact_contract.verify_release(mixed_release)
    by_path = {row["path"]: row for row in envelope["payload"]["files"]}
    dummy_label = hashlib.sha256(b"label").hexdigest()
    specs = {
        session_number: builder._SessionSpec(
            annotation_artifact_kind=(
                "perception_labels_train"
                if session_number == 2
                else "perception_labels_holdout"
            ),
            diagnostic_png_count=2 if session_number == 2 else 3,
            garment="dark" if session_number == 2 else "yellow",
            height_px=48,
            intrinsics_sha256=by_path[
                "orbbec_session%d/intrinsics.json" % session_number
            ]["sha256"],
            label_domain=(
                "clothed-orbbec"
                if session_number == 2
                else "clothed-orbbec-yellow"
            ),
            label_path=(
                "labels/dark.json" if session_number == 2 else "labels/yellow.json"
            ),
            label_sha256=dummy_label,
            projection_artifact_kind=(
                "perception_session2_train"
                if session_number == 2
                else "perception_session3_holdout"
            ),
            rgb_sha256=by_path[
                "orbbec_session%d/rgb.mp4" % session_number
            ]["sha256"],
            session_id="orbbec_session%d" % session_number,
            session_number=session_number,
            source_frames_sha256=by_path[
                "orbbec_session%d/frames.csv" % session_number
            ]["sha256"],
            subject_pseudonym="same-subject-001",
            width_px=64,
        )
        for session_number in (2, 3)
    }
    with pytest.raises(builder.InputBuildError):
        builder._build_content_partition(
            mixed_release,
            session2_spec=specs[2],
            session3_spec=specs[3],
            expected_mixed_eval_identity=mixed_identity,
        )


def test_module_cli_help_is_import_safe_and_executable() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "scripts.perception.build_perception_training_inputs",
            "--help",
        ],
        cwd=str(Path(__file__).resolve().parents[3]),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "build-geometry" in result.stdout
    assert "build-session2-projection" in result.stdout
    assert "verify-session3-projection" in result.stdout


def test_cli_emits_one_canonical_identity_document(monkeypatch, capfd) -> None:
    release = Path("/tmp/perception_geometry_source-deadbeef")
    report = {
        "file_count": 914,
        "payload_sha256": "a" * 64,
        "release_id": release.name,
        "release_manifest_sha256": "b" * 64,
        "semantic_manifest_sha256": "c" * 64,
        "sequence_count": 907,
        "total_bytes": 123,
    }
    monkeypatch.setattr(builder, "build_geometry_source_release", lambda *_: release)
    monkeypatch.setattr(builder, "verify_geometry_source_release", lambda *_: report)
    assert builder.main(
        [
            "build-geometry",
            "--data-root",
            "/tmp/data",
            "--licenses-root",
            "/tmp/licenses",
            "--releases-root",
            "/tmp/releases",
        ]
    ) == 0
    captured = capfd.readouterr()
    expected = _canonical(report).decode("utf-8")
    assert captured.out == expected
    assert captured.err == ""


def test_annotation_build_uses_manifest_snapshot_without_payload_tree(
    tmp_path: Path,
    mixed_eval_fixture,
) -> None:
    mixed_release, mixed_identity = mixed_eval_fixture
    snapshot = tmp_path / artifact_contract.MANIFEST_NAME
    snapshot.write_bytes(
        (mixed_release / artifact_contract.MANIFEST_NAME).read_bytes()
    )
    document = _label_document(session_number=2, garment="dark")
    label_bytes = json.dumps(document, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    label = tmp_path / "dark.json"
    label.write_bytes(label_bytes)
    spec = _test_session_spec(
        session_number=2,
        garment="dark",
        label_sha256=hashlib.sha256(label_bytes).hexdigest(),
        mixed_eval_release=mixed_release,
    )
    release = builder._build_annotation_release(
        label_path=label,
        mixed_eval_manifest=snapshot,
        releases_root=tmp_path / "annotation-releases",
        staging_root=tmp_path / ".annotation-staging",
        spec=spec,
        expected_mixed_eval_identity=mixed_identity,
    )
    assert builder._verify_annotation_release(
        release,
        mixed_eval_manifest=snapshot,
        spec=spec,
        expected_mixed_eval_identity=mixed_identity,
    )["selected_member_count"] == 605


@pytest.mark.parametrize("poison", (False, 0.0))
def test_strict_semantic_equality_rejects_bool_int_and_int_float(poison) -> None:
    assert builder._strict_equal({"value": 0}, {"value": poison}) is False


def _reseal_with_replaced_semantic(
    source_release: Path,
    destination_root: Path,
    semantic: dict,
) -> Path:
    source = destination_root / "source"
    shutil.copytree(source_release, source)
    (source / artifact_contract.MANIFEST_NAME).unlink()
    (source / artifact_contract.COMPLETE_NAME).unlink()
    (source / canary.PERCEPTION_INPUT_NAME).write_bytes(_canonical(semantic))
    paths = sorted(
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file()
    )
    original = artifact_contract.verify_release(source_release)
    manifest = artifact_contract.build_manifest(
        source,
        paths,
        artifact_kind=original["payload"]["artifact_kind"],
        source_role=original["payload"]["source_role"],
    )
    target = destination_root / manifest["release_id"]
    artifact_contract.write_release(
        source,
        manifest,
        target,
        target_role="mac",
    )
    return target


@pytest.mark.parametrize("poison", (False, 64.0))
def test_annotation_verifier_rejects_json_numeric_type_substitution(
    tmp_path: Path,
    mixed_eval_fixture,
    poison,
) -> None:
    mixed_release, mixed_identity = mixed_eval_fixture
    document = _label_document(session_number=2, garment="dark")
    label_bytes = json.dumps(document, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    label = tmp_path / "dark.json"
    label.write_bytes(label_bytes)
    spec = _test_session_spec(
        session_number=2,
        garment="dark",
        label_sha256=hashlib.sha256(label_bytes).hexdigest(),
        mixed_eval_release=mixed_release,
    )
    release = builder._build_annotation_release(
        label_path=label,
        mixed_eval_manifest=mixed_release / artifact_contract.MANIFEST_NAME,
        releases_root=tmp_path / "good",
        staging_root=tmp_path / ".staging-good",
        spec=spec,
        expected_mixed_eval_identity=mixed_identity,
    )
    semantic = json.loads((release / canary.PERCEPTION_INPUT_NAME).read_bytes())
    if poison is False:
        semantic["contract"]["pixel_index_base"] = False
    else:
        semantic["contract"]["width_px"] = poison
    poisoned = _reseal_with_replaced_semantic(release, tmp_path / "poisoned", semantic)
    with pytest.raises(builder.InputBuildError):
        builder._verify_annotation_release(
            poisoned,
            mixed_eval_manifest=mixed_release / artifact_contract.MANIFEST_NAME,
            spec=spec,
            expected_mixed_eval_identity=mixed_identity,
        )


def test_annotation_specialized_failure_never_publishes_final(
    tmp_path: Path,
    mixed_eval_fixture,
    monkeypatch,
) -> None:
    mixed_release, mixed_identity = mixed_eval_fixture
    document = _label_document(session_number=2, garment="dark")
    label_bytes = json.dumps(document, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    label = tmp_path / "dark.json"
    label.write_bytes(label_bytes)
    spec = _test_session_spec(
        session_number=2,
        garment="dark",
        label_sha256=hashlib.sha256(label_bytes).hexdigest(),
        mixed_eval_release=mixed_release,
    )
    releases = tmp_path / "releases"
    monkeypatch.setattr(
        builder,
        "_verify_annotation_release",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            builder.InputBuildError("specialized verification failed")
        ),
    )
    with pytest.raises(builder.InputBuildError):
        builder._build_annotation_release(
            label_path=label,
            mixed_eval_manifest=mixed_release / artifact_contract.MANIFEST_NAME,
            releases_root=releases,
            staging_root=tmp_path / ".staging",
            spec=spec,
            expected_mixed_eval_identity=mixed_identity,
        )
    assert not releases.exists() or list(releases.iterdir()) == []


def test_geometry_rejects_output_root_nested_in_source_before_write(tmp_path: Path) -> None:
    fixture = _geometry_fixture(tmp_path)
    data_root, licenses_root = fixture[0], fixture[1]
    nested_releases = data_root / "nested-releases"
    with pytest.raises(builder.InputBuildError):
        builder._build_geometry_source_release(
            data_root=data_root,
            licenses_root=licenses_root,
            releases_root=nested_releases,
            staging_root=tmp_path / ".staging",
            expected_support_sha256=fixture[4],
            expected_smpl_manifest_sha256=fixture[5],
            expected_source_terms=fixture[6],
        )
    assert not nested_releases.exists()


def test_projection_rejects_output_nested_in_mixed_eval_before_write(
    projection_input_fixture,
) -> None:
    _, mixed_release, mixed_identity, specs, annotations, pins = projection_input_fixture
    nested = mixed_release / "nested-output"
    with pytest.raises(builder.InputBuildError):
        builder._build_projection_release(
            mixed_eval_release=mixed_release,
            annotation_release=annotations[2],
            releases_root=nested,
            staging_root=mixed_release / ".nested-staging",
            spec=specs[2],
            peer_spec=specs[3],
            expected_mixed_eval_identity=mixed_identity,
            pins=pins[2],
            authoring_authorization=builder._test_h200_authorization(),
        )
    assert not nested.exists()


def test_verifier_rejects_release_renamed_away_from_content_id(tmp_path: Path) -> None:
    release, fixture = _build_geometry_fixture(tmp_path)
    renamed = release.with_name("renamed-release")
    release.rename(renamed)
    with pytest.raises(builder.InputBuildError):
        builder._verify_geometry_source_release(
            renamed,
            expected_support_sha256=fixture[4],
            expected_smpl_manifest_sha256=fixture[5],
            expected_source_terms=fixture[6],
        )


def test_bounded_reader_rejects_payload_over_hard_limit(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "large.bin").write_bytes(b"12345")
    with pytest.raises(builder.InputBuildError):
        builder._read_regular(
            root,
            "large.bin",
            "bounded fixture",
            maximum_size=4,
        )


def test_geometry_repeat_build_reuses_exact_release(tmp_path: Path) -> None:
    first, fixture = _build_geometry_fixture(tmp_path)
    second = builder._build_geometry_source_release(
        data_root=fixture[0],
        licenses_root=fixture[1],
        releases_root=fixture[2],
        staging_root=fixture[3],
        expected_support_sha256=fixture[4],
        expected_smpl_manifest_sha256=fixture[5],
        expected_source_terms=fixture[6],
    )
    assert second == first


def test_annotation_repeat_reuses_exact_but_rejects_corrupt_existing(
    tmp_path: Path,
    mixed_eval_fixture,
) -> None:
    mixed_release, mixed_identity = mixed_eval_fixture
    document = _label_document(session_number=2, garment="dark")
    label_bytes = json.dumps(document, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    label = tmp_path / "dark.json"
    label.write_bytes(label_bytes)
    spec = _test_session_spec(
        session_number=2,
        garment="dark",
        label_sha256=hashlib.sha256(label_bytes).hexdigest(),
        mixed_eval_release=mixed_release,
    )
    kwargs = {
        "label_path": label,
        "mixed_eval_manifest": mixed_release / artifact_contract.MANIFEST_NAME,
        "releases_root": tmp_path / "releases",
        "staging_root": tmp_path / ".staging",
        "spec": spec,
        "expected_mixed_eval_identity": mixed_identity,
    }
    first = builder._build_annotation_release(**kwargs)
    assert builder._build_annotation_release(**kwargs) == first
    (first / "extra.bin").write_bytes(b"corrupt")
    with pytest.raises(builder.InputBuildError):
        builder._build_annotation_release(**kwargs)


def test_session3_production_subject_is_within_subject_cross_garment() -> None:
    assert builder._production_session3_spec().subject_pseudonym == (
        builder._production_session2_spec().subject_pseudonym
    )


def test_projection_specialized_failure_never_publishes_final(
    tmp_path: Path,
    projection_input_fixture,
    monkeypatch,
) -> None:
    _, mixed_release, mixed_identity, specs, annotations, pins = projection_input_fixture
    releases = tmp_path / "projection-releases"
    monkeypatch.setattr(
        builder,
        "_verify_projection_release",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            builder.InputBuildError("specialized projection verification failed")
        ),
    )
    with pytest.raises(builder.InputBuildError):
        builder._build_projection_release(
            mixed_eval_release=mixed_release,
            annotation_release=annotations[2],
            releases_root=releases,
            staging_root=tmp_path / ".staging",
            spec=specs[2],
            peer_spec=specs[3],
            expected_mixed_eval_identity=mixed_identity,
            pins=pins[2],
            authoring_authorization=builder._test_h200_authorization(),
        )
    assert not releases.exists() or list(releases.iterdir()) == []


def test_projection_verifier_rejects_release_renamed_away_from_content_id(
    tmp_path: Path,
    projection_input_fixture,
) -> None:
    _, mixed_release, mixed_identity, specs, annotations, pins = projection_input_fixture
    release = builder._build_projection_release(
        mixed_eval_release=mixed_release,
        annotation_release=annotations[2],
        releases_root=tmp_path / "releases",
        staging_root=tmp_path / ".staging",
        spec=specs[2],
        peer_spec=specs[3],
        expected_mixed_eval_identity=mixed_identity,
        pins=pins[2],
        authoring_authorization=builder._test_h200_authorization(),
    )
    renamed = release.with_name("perception_session2_train-" + "f" * 20)
    release.rename(renamed)
    with pytest.raises(builder.InputBuildError):
        builder._verify_projection_release(
            renamed,
            mixed_eval_release=mixed_release,
            annotation_release=annotations[2],
            spec=specs[2],
            peer_spec=specs[3],
            expected_mixed_eval_identity=mixed_identity,
            pins=pins[2],
        )


def test_projection_verifier_rejects_json_numeric_type_substitution(
    tmp_path: Path,
    projection_input_fixture,
) -> None:
    _, mixed_release, mixed_identity, specs, annotations, pins = projection_input_fixture
    release = builder._build_projection_release(
        mixed_eval_release=mixed_release,
        annotation_release=annotations[2],
        releases_root=tmp_path / "releases",
        staging_root=tmp_path / ".staging",
        spec=specs[2],
        peer_spec=specs[3],
        expected_mixed_eval_identity=mixed_identity,
        pins=pins[2],
        authoring_authorization=builder._test_h200_authorization(),
    )
    semantic = json.loads((release / canary.PERCEPTION_INPUT_NAME).read_bytes())
    semantic["self_contained"] = 1  # bool -> int substitution
    poisoned = _reseal_with_replaced_semantic(release, tmp_path / "poisoned", semantic)
    with pytest.raises(builder.InputBuildError):
        builder._verify_projection_release(
            poisoned,
            mixed_eval_release=mixed_release,
            annotation_release=annotations[2],
            spec=specs[2],
            peer_spec=specs[3],
            expected_mixed_eval_identity=mixed_identity,
            pins=pins[2],
        )
