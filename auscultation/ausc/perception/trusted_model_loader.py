"""Trusted loader for production model-replay verification.

A production replay verifier must never accept a caller-supplied scoring
callable — an arbitrary or forged runner could trivially declare a match and
authorize promotion.  Instead it constructs its runner ONLY through a
:class:`TrustedModelLoader`, which:

* binds an immutable :class:`TrustedModelActivation` whose ``release_id``
  content-addresses the exact model SHA-256 and names the preprocessing,
  postprocessing, tracker, calibration and runtime it is pinned to;
* re-reads the on-disk weight bytes at load time and refuses unless their
  SHA-256 equals the activation's;
* obtains a runner from a *privileged*, environment-provided constructor
  (the real Torch/CUDA loader on the box) — never from a value handed in at
  verification time.

This module is deliberately framework-free so the contract and its negative
paths are fully testable without Torch/CUDA; the privileged constructor is the
single seam the real environment supplies.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
import re
from typing import Callable

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RELEASE_ID = re.compile(r"^[a-z][a-z0-9_-]*-[0-9a-f]{20}$")

TRUSTED_MODEL_ACTIVATION_SCHEMA = "wam_trusted_model_activation_v1"


class TrustedModelLoaderError(RuntimeError):
    """A production model runner cannot be trusted for replay authorization."""


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise TrustedModelLoaderError(f"{label} SHA-256 is invalid")
    return value


def _require_identity(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrustedModelLoaderError(f"{label} identity is missing")
    return value


def _require_relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TrustedModelLoaderError(f"{label} path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix() or any(
        part in ("", ".", "..") for part in path.parts
    ):
        raise TrustedModelLoaderError(f"{label} path is unsafe")
    return value


@dataclass(frozen=True)
class TrustedModelActivation:
    """Immutable, verified identity of one activated production model.

    ``release_id`` must content-address ``model_sha256`` (``<kind>-<first20>``),
    so the release name cannot be pointed at different weights.
    """

    release_id: str
    model_relative_path: str
    model_sha256: str
    preprocessing_id: str
    postprocessing_id: str
    tracker_id: str
    calibration_id: str
    runtime_id: str

    def __post_init__(self) -> None:
        model_sha = _require_sha256(self.model_sha256, label="model")
        if not isinstance(self.release_id, str) or _RELEASE_ID.fullmatch(self.release_id) is None:
            raise TrustedModelLoaderError("model release id is invalid")
        if self.release_id.rsplit("-", 1)[-1] != model_sha[:20]:
            raise TrustedModelLoaderError(
                "model release id does not content-address the model SHA-256"
            )
        _require_relative_path(self.model_relative_path, label="model")
        _require_identity(self.preprocessing_id, label="preprocessing")
        _require_identity(self.postprocessing_id, label="postprocessing")
        _require_identity(self.tracker_id, label="tracker")
        _require_identity(self.calibration_id, label="calibration")
        _require_identity(self.runtime_id, label="runtime")

    @classmethod
    def from_manifest(cls, value: object) -> "TrustedModelActivation":
        if not isinstance(value, Mapping):
            raise TrustedModelLoaderError("activation manifest is not an object")
        expected = {
            "schema_version", "release_id", "model_relative_path", "model_sha256",
            "preprocessing_id", "postprocessing_id", "tracker_id",
            "calibration_id", "runtime_id",
        }
        if set(value) != expected:
            raise TrustedModelLoaderError("activation manifest field set is not exact")
        if value["schema_version"] != TRUSTED_MODEL_ACTIVATION_SCHEMA:
            raise TrustedModelLoaderError("activation manifest schema is unsupported")
        return cls(
            release_id=value["release_id"],
            model_relative_path=value["model_relative_path"],
            model_sha256=value["model_sha256"],
            preprocessing_id=value["preprocessing_id"],
            postprocessing_id=value["postprocessing_id"],
            tracker_id=value["tracker_id"],
            calibration_id=value["calibration_id"],
            runtime_id=value["runtime_id"],
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TrustedModelLoader:
    """Construct a production replay runner ONLY from a verified release.

    ``weights_constructor`` is the privileged, environment-provided function that
    turns the verified on-disk weight *bytes* into a runnable model.  It is not a
    scoring function and it is not the value a verification caller could hand in;
    the finalizer obtains a ``TrustedModelLoader`` for the activated release and
    never a bare callable, which is what stops an arbitrary runner from ever
    authorizing promotion.
    """

    def __init__(
        self,
        *,
        release_root: "str | Path",
        activation: TrustedModelActivation,
        weights_constructor: Callable[[bytes, TrustedModelActivation], object],
    ) -> None:
        if not isinstance(activation, TrustedModelActivation):
            raise TrustedModelLoaderError("activation is not a TrustedModelActivation")
        activation.__post_init__()
        if not callable(weights_constructor):
            raise TrustedModelLoaderError("weights constructor is not callable")
        root = Path(release_root)
        if not root.is_dir() or root.is_symlink():
            raise TrustedModelLoaderError("release root must be a real directory")
        self._root = root
        self._activation = activation
        self._weights_constructor = weights_constructor

    @property
    def activation(self) -> TrustedModelActivation:
        return self._activation

    def load(self) -> object:
        """Re-verify the on-disk weights and build the runner, or fail closed."""

        model_path = self._root / self._activation.model_relative_path
        if model_path.is_symlink() or not model_path.is_file():
            raise TrustedModelLoaderError("activated model file is missing")
        actual = _sha256_file(model_path)
        if actual != self._activation.model_sha256:
            raise TrustedModelLoaderError(
                "activated model file SHA-256 does not match the activation"
            )
        runner = self._weights_constructor(model_path.read_bytes(), self._activation)
        if runner is None or not callable(getattr(runner, "run", None)):
            raise TrustedModelLoaderError(
                "weights constructor did not return a runnable model"
            )
        return runner


def require_trusted_model_loader(candidate: object) -> TrustedModelLoader:
    """Return ``candidate`` iff it is a real :class:`TrustedModelLoader`.

    This is the single guard a production replay verifier uses at its boundary:
    a bare callable, a lambda, or any other value is refused, so a caller can
    never inject an arbitrary runner that would authorize promotion.
    """

    if not isinstance(candidate, TrustedModelLoader):
        raise TrustedModelLoaderError(
            "production replay requires a TrustedModelLoader constructed from a "
            "SHA/activation-verified release; an arbitrary callable can never "
            "authorize promotion"
        )
    return candidate


__all__ = [
    "TRUSTED_MODEL_ACTIVATION_SCHEMA",
    "TrustedModelActivation",
    "TrustedModelLoader",
    "TrustedModelLoaderError",
    "require_trusted_model_loader",
]
