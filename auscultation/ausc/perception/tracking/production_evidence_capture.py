"""Lossless, fail-closed staging for production geometry model inputs.

This module deliberately implements only the acquisition-side input boundary.
It does not claim that geometry output is production-authorized and it does not
finalize a training session.  A detector may consume an input only after the
persisted NPY artifacts have been reopened, checked, and copied into immutable
arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
import errno
import hashlib
import io
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Optional, Sequence, Tuple

import numpy as np

from .geometry import Intrinsics
from .patient_frame_evidence import PatientFrameEvidence
from .production_geometry_gate import ProductionGeometryProfile


PERSISTED_GEOMETRY_MODEL_INPUT_SCHEMA = "wam_persisted_geometry_model_input_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_NPY_BYTES = 512 * 1024 * 1024


class ProductionEvidenceCaptureError(RuntimeError):
    """An input cannot safely be treated as exact persisted evidence."""


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value).tobytes(order="C")
    ).hexdigest()


def _hard_readonly(value: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(value)
    frozen = np.frombuffer(
        contiguous.tobytes(order="C"), dtype=contiguous.dtype,
    ).reshape(contiguous.shape)
    frozen.setflags(write=False)
    return frozen


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ProductionEvidenceCaptureError(f"{label} SHA-256 is invalid")
    return value


def _require_relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProductionEvidenceCaptureError(f"{label} path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ProductionEvidenceCaptureError(f"{label} path is unsafe")
    if str(path) != value:
        raise ProductionEvidenceCaptureError(f"{label} path is not canonical")
    return value


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        getattr(value, "st_mtime_ns", int(value.st_mtime * 1_000_000_000)),
        getattr(value, "st_ctime_ns", int(value.st_ctime * 1_000_000_000)),
    )


def _require_directory(path: Path, *, label: str) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ProductionEvidenceCaptureError(f"{label} is missing") from exc
    if stat.S_ISLNK(info.st_mode):
        raise ProductionEvidenceCaptureError(f"{label} is a symlink")
    if not stat.S_ISDIR(info.st_mode):
        raise ProductionEvidenceCaptureError(f"{label} is not a directory")


def _require_nonsymlink_components(session_dir: Path, path: Path) -> None:
    try:
        relative = path.relative_to(session_dir)
    except ValueError as exc:
        raise ProductionEvidenceCaptureError(
            "geometry artifact escapes the session directory"
        ) from exc
    _require_directory(session_dir, label="session directory")
    cursor = session_dir
    for part in relative.parts[:-1]:
        cursor = cursor / part
        _require_directory(cursor, label="geometry artifact parent")


def _stable_regular_bytes(
    session_dir: Path,
    relative_path: str,
    *,
    label: str,
) -> bytes:
    relative_path = _require_relative_path(relative_path, label=label)
    path = session_dir / relative_path
    _require_nonsymlink_components(session_dir, path)
    try:
        path_info = path.lstat()
    except OSError as exc:
        raise ProductionEvidenceCaptureError(f"{label} is missing") from exc
    if stat.S_ISLNK(path_info.st_mode):
        raise ProductionEvidenceCaptureError(f"{label} is a symlink")
    if not stat.S_ISREG(path_info.st_mode):
        raise ProductionEvidenceCaptureError(
            f"{label} must be a regular non-symlink file"
        )
    if path_info.st_size <= 0 or path_info.st_size > _MAX_NPY_BYTES:
        raise ProductionEvidenceCaptureError(f"{label} has an invalid size")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(path), flags)
    except OSError as exc:
        raise ProductionEvidenceCaptureError(f"{label} cannot be opened") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ProductionEvidenceCaptureError(
                f"{label} must be a regular file"
            )
        chunks = []
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(remaining, 1024 * 1024))
            if not block:
                raise ProductionEvidenceCaptureError(
                    f"{label} was truncated while reading"
                )
            chunks.append(block)
            remaining -= len(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _stat_identity(before) != _stat_identity(after):
        raise ProductionEvidenceCaptureError(f"{label} changed while reading")
    return b"".join(chunks)


def _decode_exact_npy(
    raw: bytes,
    *,
    label: str,
    expected_file_sha256: str,
    expected_array_sha256: str,
    expected_dtype: str,
    expected_shape: Tuple[int, ...],
) -> np.ndarray:
    if hashlib.sha256(raw).hexdigest() != _require_sha256(
        expected_file_sha256, label=f"{label} file",
    ):
        raise ProductionEvidenceCaptureError(f"{label} file SHA-256 mismatch")
    expected_array_sha256 = _require_sha256(
        expected_array_sha256, label=f"{label} array",
    )
    stream = io.BytesIO(raw)
    try:
        value = np.load(stream, allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise ProductionEvidenceCaptureError(f"{label} is not a valid NPY") from exc
    if stream.tell() != len(raw):
        raise ProductionEvidenceCaptureError(f"{label} has trailing bytes")
    if not isinstance(value, np.ndarray):
        raise ProductionEvidenceCaptureError(f"{label} is not an array")
    if value.dtype.str != expected_dtype:
        raise ProductionEvidenceCaptureError(f"{label} dtype mismatch")
    if tuple(value.shape) != tuple(expected_shape):
        raise ProductionEvidenceCaptureError(f"{label} shape mismatch")
    if not value.flags.c_contiguous:
        raise ProductionEvidenceCaptureError(f"{label} is not C-contiguous")
    if _array_sha256(value) != expected_array_sha256:
        raise ProductionEvidenceCaptureError(f"{label} array SHA-256 mismatch")
    return _hard_readonly(value)


@dataclass(frozen=True)
class ReloadedGeometryModelInput:
    """Immutable arrays reopened from the persisted evidence boundary."""

    rgb: np.ndarray
    depth: np.ndarray
    back_mask: Optional[np.ndarray]
    robot_mask: Optional[np.ndarray]


@dataclass(frozen=True)
class PersistedGeometryModelInput:
    """Frozen locator and exact identities for one detector input."""

    schema: str
    session_dir: Path
    episode_id: str
    sequence: int
    patient_frame_evidence_id: str
    frame_index: int
    camera_t: float
    depth_scale: float
    rgb_relative_path: str
    rgb_sha256: str
    rgb_file_sha256: str
    rgb_dtype: str
    rgb_shape: Tuple[int, ...]
    depth_relative_path: str
    depth_sha256: str
    depth_file_sha256: str
    depth_dtype: str
    depth_shape: Tuple[int, ...]
    back_mask_relative_path: Optional[str]
    back_mask_sha256: Optional[str]
    back_mask_file_sha256: Optional[str]
    robot_mask_relative_path: Optional[str]
    robot_mask_sha256: Optional[str]
    robot_mask_file_sha256: Optional[str]

    def __post_init__(self) -> None:
        if self.schema != PERSISTED_GEOMETRY_MODEL_INPUT_SCHEMA:
            raise ProductionEvidenceCaptureError(
                "persisted geometry model-input schema is unsupported"
            )
        if not isinstance(self.session_dir, Path) or not self.session_dir.is_absolute():
            raise ProductionEvidenceCaptureError("session directory must be absolute")
        _require_directory(self.session_dir, label="session directory")
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise ProductionEvidenceCaptureError("episode ID is missing")
        if type(self.sequence) is not int or self.sequence < 1:
            raise ProductionEvidenceCaptureError("sequence must be a positive integer")
        if not isinstance(self.patient_frame_evidence_id, str) or not self.patient_frame_evidence_id:
            raise ProductionEvidenceCaptureError("patient-frame evidence ID is missing")
        if type(self.frame_index) is not int or not 0 <= self.frame_index <= 999999:
            raise ProductionEvidenceCaptureError("frame index is invalid")
        if (
            isinstance(self.camera_t, bool)
            or not isinstance(self.camera_t, (int, float))
            or not math.isfinite(float(self.camera_t))
            or float(self.camera_t) <= 0.0
        ):
            raise ProductionEvidenceCaptureError("camera timestamp is invalid")
        if (
            isinstance(self.depth_scale, bool)
            or not isinstance(self.depth_scale, (int, float))
            or not math.isfinite(float(self.depth_scale))
            or float(self.depth_scale) <= 0.0
        ):
            raise ProductionEvidenceCaptureError("depth scale is invalid")

        rgb_hash = _require_sha256(self.rgb_sha256, label="RGB array")
        _require_sha256(self.rgb_file_sha256, label="RGB file")
        rgb_path = _require_relative_path(self.rgb_relative_path, label="RGB")
        if rgb_path != f"geometry/model_inputs/{rgb_hash}.npy":
            raise ProductionEvidenceCaptureError("RGB path is not content-addressed")
        if self.rgb_dtype != np.dtype(np.uint8).str:
            raise ProductionEvidenceCaptureError("RGB dtype contract is invalid")
        if (
            len(self.rgb_shape) != 3
            or self.rgb_shape[0] <= 0
            or self.rgb_shape[1] <= 0
            or self.rgb_shape[2] != 3
        ):
            raise ProductionEvidenceCaptureError("RGB shape contract is invalid")

        _require_sha256(self.depth_sha256, label="depth array")
        _require_sha256(self.depth_file_sha256, label="depth file")
        depth_path = _require_relative_path(self.depth_relative_path, label="depth")
        if depth_path != f"depth/depth_{self.frame_index:06d}.npy":
            raise ProductionEvidenceCaptureError("depth path is not canonical")
        if self.depth_dtype != np.dtype(np.uint16).str:
            raise ProductionEvidenceCaptureError("depth dtype contract is invalid")
        if tuple(self.depth_shape) != tuple(self.rgb_shape[:2]):
            raise ProductionEvidenceCaptureError("RGB/depth shape contract mismatch")

        self._validate_optional_mask("back", self.back_mask_relative_path,
                                    self.back_mask_sha256,
                                    self.back_mask_file_sha256)
        self._validate_optional_mask("robot", self.robot_mask_relative_path,
                                    self.robot_mask_sha256,
                                    self.robot_mask_file_sha256)

    @staticmethod
    def _validate_optional_mask(
        kind: str,
        relative_path: Optional[str],
        array_sha256: Optional[str],
        file_sha256: Optional[str],
    ) -> None:
        values = (relative_path, array_sha256, file_sha256)
        if all(value is None for value in values):
            return
        if any(value is None for value in values):
            raise ProductionEvidenceCaptureError(
                f"{kind} mask identity is incomplete"
            )
        assert relative_path is not None
        assert array_sha256 is not None
        assert file_sha256 is not None
        array_sha256 = _require_sha256(array_sha256, label=f"{kind} mask array")
        _require_sha256(file_sha256, label=f"{kind} mask file")
        relative_path = _require_relative_path(
            relative_path, label=f"{kind} mask",
        )
        expected = f"geometry/evidence_arrays/{kind}_{array_sha256}.npy"
        if relative_path != expected:
            raise ProductionEvidenceCaptureError(
                f"{kind} mask path is not content-addressed"
            )

    def reload_for_inference(self) -> ReloadedGeometryModelInput:
        """Reopen and verify every artifact before detector invocation."""

        self.__post_init__()
        rgb = _decode_exact_npy(
            _stable_regular_bytes(
                self.session_dir, self.rgb_relative_path, label="RGB model input",
            ),
            label="RGB model input",
            expected_file_sha256=self.rgb_file_sha256,
            expected_array_sha256=self.rgb_sha256,
            expected_dtype=self.rgb_dtype,
            expected_shape=self.rgb_shape,
        )
        depth = _decode_exact_npy(
            _stable_regular_bytes(
                self.session_dir, self.depth_relative_path, label="depth model input",
            ),
            label="depth model input",
            expected_file_sha256=self.depth_file_sha256,
            expected_array_sha256=self.depth_sha256,
            expected_dtype=self.depth_dtype,
            expected_shape=self.depth_shape,
        )
        back = self._reload_mask(
            "back", self.back_mask_relative_path, self.back_mask_sha256,
            self.back_mask_file_sha256,
        )
        robot = self._reload_mask(
            "robot", self.robot_mask_relative_path, self.robot_mask_sha256,
            self.robot_mask_file_sha256,
        )
        return ReloadedGeometryModelInput(
            rgb=rgb, depth=depth, back_mask=back, robot_mask=robot,
        )

    def _reload_mask(
        self,
        kind: str,
        relative_path: Optional[str],
        array_sha256: Optional[str],
        file_sha256: Optional[str],
    ) -> Optional[np.ndarray]:
        if relative_path is None:
            return None
        assert array_sha256 is not None
        assert file_sha256 is not None
        return _decode_exact_npy(
            _stable_regular_bytes(
                self.session_dir, relative_path, label=f"{kind} mask model input",
            ),
            label=f"{kind} mask model input",
            expected_file_sha256=file_sha256,
            expected_array_sha256=array_sha256,
            expected_dtype=np.dtype(np.bool_).str,
            expected_shape=self.depth_shape,
        )


class ProductionGeometryEvidenceCapture:
    """Stage exact geometry inputs without authorizing their outputs."""

    def __init__(
        self,
        session_dir: str | Path,
        *,
        intrinsics: Intrinsics,
        geometry_profile: ProductionGeometryProfile,
        point_names: Sequence[str],
    ) -> None:
        path = Path(session_dir).absolute()
        _require_directory(path, label="session directory")
        if not isinstance(intrinsics, Intrinsics):
            raise ProductionEvidenceCaptureError("intrinsics are invalid")
        if not isinstance(geometry_profile, ProductionGeometryProfile):
            raise ProductionEvidenceCaptureError("geometry profile is invalid")
        try:
            geometry_profile.__post_init__()
        except (TypeError, ValueError) as exc:
            raise ProductionEvidenceCaptureError("geometry profile is invalid") from exc
        names = tuple(point_names)
        if (
            not names
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ProductionEvidenceCaptureError("point names are invalid")
        self.session_dir = path
        self.intrinsics = intrinsics
        self.geometry_profile = geometry_profile
        self.point_names = names
        self._ensure_directory("depth")
        self._ensure_directory("geometry")
        self._ensure_directory("geometry/model_inputs")
        self._ensure_directory("geometry/evidence_arrays")

    def _ensure_directory(self, relative_path: str) -> Path:
        cursor = self.session_dir
        for part in PurePosixPath(relative_path).parts:
            cursor = cursor / part
            try:
                cursor.mkdir(mode=0o750)
            except FileExistsError:
                pass
            except OSError as exc:
                raise ProductionEvidenceCaptureError(
                    "geometry artifact directory cannot be created"
                ) from exc
            _require_directory(cursor, label="geometry artifact directory")
        return cursor

    def _persist_array(
        self,
        relative_path: str,
        value: np.ndarray,
        *,
        label: str,
    ) -> tuple[str, str]:
        value = np.ascontiguousarray(value)
        array_sha256 = _array_sha256(value)
        buffer = io.BytesIO()
        np.save(buffer, value, allow_pickle=False)
        raw = buffer.getvalue()
        file_sha256 = hashlib.sha256(raw).hexdigest()
        path = self.session_dir / _require_relative_path(
            relative_path, label=label,
        )
        _require_nonsymlink_components(self.session_dir, path)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(str(path), flags, 0o640)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise ProductionEvidenceCaptureError(
                    f"{label} cannot be created"
                ) from exc
            existing = _stable_regular_bytes(
                self.session_dir, relative_path, label=label,
            )
            if existing != raw:
                raise ProductionEvidenceCaptureError(
                    f"existing {label} does not match exact array"
                )
            return array_sha256, file_sha256
        try:
            view = memoryview(raw)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("artifact write made no progress")
                view = view[written:]
            os.fsync(descriptor)
        except BaseException:
            os.close(descriptor)
            path.unlink(missing_ok=True)
            raise
        else:
            os.close(descriptor)
        directory_descriptor = os.open(
            str(path.parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        return array_sha256, file_sha256

    def stage_model_input(
        self,
        patient_frame_evidence: PatientFrameEvidence,
        *,
        episode_id: str,
        sequence: int,
    ) -> PersistedGeometryModelInput:
        """Persist one exact RGB-D/mask input and return a reload-only handle."""

        if not isinstance(patient_frame_evidence, PatientFrameEvidence):
            raise ProductionEvidenceCaptureError(
                "patient-frame evidence is required"
            )
        try:
            patient_frame_evidence.__post_init__()
        except (TypeError, ValueError) as exc:
            raise ProductionEvidenceCaptureError(
                "patient-frame evidence is invalid"
            ) from exc
        if not isinstance(episode_id, str) or not episode_id:
            raise ProductionEvidenceCaptureError("episode ID is missing")
        if type(sequence) is not int or sequence < 1:
            raise ProductionEvidenceCaptureError(
                "sequence must be a positive integer"
            )
        if not 0 <= patient_frame_evidence.index <= 999999:
            raise ProductionEvidenceCaptureError("frame index exceeds canonical range")

        rgb = patient_frame_evidence.rgb
        depth = patient_frame_evidence.aligned_depth
        back = patient_frame_evidence.back_mask
        robot = patient_frame_evidence.robot_mask
        camera = patient_frame_evidence.frame_bundle.camera_frame

        rgb_relative_path = f"geometry/model_inputs/{camera.rgb_sha256}.npy"
        rgb_sha256, rgb_file_sha256 = self._persist_array(
            rgb_relative_path, rgb, label="RGB model input",
        )
        if rgb_sha256 != camera.rgb_sha256:
            raise ProductionEvidenceCaptureError("RGB evidence hash mismatch")

        depth_relative_path = f"depth/depth_{camera.index:06d}.npy"
        depth_sha256, depth_file_sha256 = self._persist_array(
            depth_relative_path, depth, label="depth model input",
        )
        if depth_sha256 != camera.depth_sha256:
            raise ProductionEvidenceCaptureError("depth evidence hash mismatch")

        back_hash = _array_sha256(np.asarray(back, dtype=np.bool_))
        back_relative_path = (
            f"geometry/evidence_arrays/back_{back_hash}.npy"
        )
        persisted_back_hash, back_file_sha256 = self._persist_array(
            back_relative_path,
            np.asarray(back, dtype=np.bool_),
            label="back mask model input",
        )
        if persisted_back_hash != patient_frame_evidence.back_mask_packet.mask_sha256:
            raise ProductionEvidenceCaptureError("back mask evidence hash mismatch")

        robot_hash = _array_sha256(np.asarray(robot, dtype=np.bool_))
        robot_relative_path = (
            f"geometry/evidence_arrays/robot_{robot_hash}.npy"
        )
        persisted_robot_hash, robot_file_sha256 = self._persist_array(
            robot_relative_path,
            np.asarray(robot, dtype=np.bool_),
            label="robot mask model input",
        )
        if (
            persisted_robot_hash
            != patient_frame_evidence.frame_bundle.robot_mask_packet.mask_sha256
        ):
            raise ProductionEvidenceCaptureError("robot mask evidence hash mismatch")

        return PersistedGeometryModelInput(
            schema=PERSISTED_GEOMETRY_MODEL_INPUT_SCHEMA,
            session_dir=self.session_dir,
            episode_id=episode_id,
            sequence=sequence,
            patient_frame_evidence_id=patient_frame_evidence.evidence_id,
            frame_index=camera.index,
            camera_t=float(camera.t),
            depth_scale=float(camera.depth_scale),
            rgb_relative_path=rgb_relative_path,
            rgb_sha256=rgb_sha256,
            rgb_file_sha256=rgb_file_sha256,
            rgb_dtype=rgb.dtype.str,
            rgb_shape=tuple(rgb.shape),
            depth_relative_path=depth_relative_path,
            depth_sha256=depth_sha256,
            depth_file_sha256=depth_file_sha256,
            depth_dtype=depth.dtype.str,
            depth_shape=tuple(depth.shape),
            back_mask_relative_path=back_relative_path,
            back_mask_sha256=persisted_back_hash,
            back_mask_file_sha256=back_file_sha256,
            robot_mask_relative_path=robot_relative_path,
            robot_mask_sha256=persisted_robot_hash,
            robot_mask_file_sha256=robot_file_sha256,
        )


__all__ = [
    "PERSISTED_GEOMETRY_MODEL_INPUT_SCHEMA",
    "PersistedGeometryModelInput",
    "ProductionEvidenceCaptureError",
    "ProductionGeometryEvidenceCapture",
    "ReloadedGeometryModelInput",
]
