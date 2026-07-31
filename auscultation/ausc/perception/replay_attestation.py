"""Signed model-replay attestation contract for production promotion.

A production component (geometry, quality) may only authorize training promotion
when the *deployment-activated* model has been independently re-run on the exact
persisted inputs and its outputs reproduce the recorded packets.  That replay
runs on the box with Torch/CUDA (see :mod:`ausc.perception.trusted_model_loader`),
which is framework-heavy and non-reproducible in unit tests.  So the box emits a
:class:`ReplayAttestation` artifact, and the *framework-free* finalizer verifies
it here.

The verifier is the anti-forgery boundary.  It fails closed unless ALL hold:

* the attestation binds, by SHA-256, the exact session artifacts it replayed
  (packet log, per-frame ledger / canonical ``audio.wav``) and every one still
  rehashes to that value on disk;
* the replayed model identity (``release_id`` content-addresses ``model_sha256``)
  equals the identity recorded in the session's own packets — a replay of some
  *other* model can never authorize this session;
* coverage is total: exactly one replayed item per recorded packet, and every
  item matched;
* the comparison is **bit-exact**, OR — only once a per-output epsilon ceiling
  has been *frozen from a calibration run* — within that frozen tolerance.  Until
  a ceiling is frozen, ``frozen_epsilon`` mode is refused, so the default is the
  strictest possible bar.

This module is deliberately Torch-free so every negative path is testable on any
host; the box-side runner that *produces* attestations lives separately and uses
the trusted loader as its single privileged seam.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RELEASE_ID = re.compile(r"^[a-z][a-z0-9_-]*-[0-9a-f]{20}$")
_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

REPLAY_ATTESTATION_SCHEMA = "wam_model_replay_attestation_v1"

COMPARISON_BIT_EXACT = "bit_exact"
COMPARISON_FROZEN_EPSILON = "frozen_epsilon"
_COMPARISON_MODES = frozenset({COMPARISON_BIT_EXACT, COMPARISON_FROZEN_EPSILON})

#: Every component must replay against these session artifacts, by relative path.
#: The verifier rehashes each and refuses any attestation that does not bind the
#: full required set (extra bound artifacts are allowed, missing ones are not).
#: ``geometry`` covers the perception channel and ``geometry_independent`` the
#: independent metric patient-pose channel; the finalizer requires BOTH so the
#: independent channel is replayed, not merely caller-self-reported.  Both bind
#: the same session artifacts.
REQUIRED_BOUND_ARTIFACTS: dict[str, frozenset[str]] = {
    "geometry": frozenset({
        "geometry_frame_packets.jsonl",
        "geometry/model_input_ledger.jsonl",
    }),
    "geometry_independent": frozenset({
        "geometry_frame_packets.jsonl",
        "geometry/model_input_ledger.jsonl",
    }),
    "quality": frozenset({
        "quality_inference_packets.jsonl",
        "audio.wav",
    }),
}

#: Canonical in-session location of each component's replay attestation, written
#: by the box-side runner and read by the framework-free finalizer.
ATTESTATION_RELATIVE_PATHS: dict[str, str] = {
    "geometry": "geometry/replay_attestation.json",
    "geometry_independent": "geometry/replay_attestation_independent.json",
    "quality": "quality/replay_attestation.json",
}

#: Per-component, per-output frozen tolerance ceilings.  DELIBERATELY EMPTY:
#: ``frozen_epsilon`` mode is refused for a component with no frozen ceiling, so
#: promotion requires bit-exact replay until a calibration run on the box proves
#: the activated model is not bit-reproducible and freezes a ceiling here.  A
#: ceiling, once added, is the hard maximum an attestation's epsilon may claim.
FROZEN_EPSILON_CEILINGS: dict[str, dict[str, float]] = {}

_ATTESTATION_FIELDS = frozenset({
    "schema_version",
    "component",
    "session_id",
    "bound_artifacts",
    "model_release_id",
    "model_sha256",
    "comparison_mode",
    "item_count",
    "matched_count",
    "max_abs_residual",
    "epsilon",
    "producer",
})

_PRODUCER_FIELDS = frozenset({
    "code_release_id",
    "module_sha256",
    "activation_sha256",
})


class ReplayAttestationError(RuntimeError):
    """A model-replay attestation cannot authorize production promotion."""


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ReplayAttestationError(f"{label} SHA-256 is invalid")
    return value


def _require_release_id(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _RELEASE_ID.fullmatch(value) is None:
        raise ReplayAttestationError(f"{label} release id is invalid")
    return value


def _require_relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReplayAttestationError(f"{label} path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix() or any(
        part in ("", ".", "..") for part in path.parts
    ):
        raise ReplayAttestationError(f"{label} path is unsafe")
    return value


def _require_finite_nonneg(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplayAttestationError(f"{label} is not a number")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")) or number < 0.0:
        raise ReplayAttestationError(f"{label} must be finite and non-negative")
    return number


def _require_positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ReplayAttestationError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True)
class ReplayAttestation:
    """Immutable, validated record of one component's model-replay result."""

    component: str
    session_id: str
    bound_artifacts: Mapping[str, str]
    model_release_id: str
    model_sha256: str
    comparison_mode: str
    item_count: int
    matched_count: int
    max_abs_residual: Mapping[str, float]
    epsilon: Mapping[str, float]
    producer: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.component not in REQUIRED_BOUND_ARTIFACTS:
            raise ReplayAttestationError("attestation component is unsupported")
        if not isinstance(self.session_id, str) or _SESSION_ID.fullmatch(self.session_id) is None:
            raise ReplayAttestationError("attestation session id is invalid")

        model_sha = _require_sha256(self.model_sha256, label="replayed model")
        _require_release_id(self.model_release_id, label="replayed model")
        if self.model_release_id.rsplit("-", 1)[-1] != model_sha[:20]:
            raise ReplayAttestationError(
                "replayed model release id does not content-address its SHA-256"
            )

        if not isinstance(self.bound_artifacts, Mapping) or not self.bound_artifacts:
            raise ReplayAttestationError("bound artifacts are missing")
        required = REQUIRED_BOUND_ARTIFACTS[self.component]
        for relative, digest in self.bound_artifacts.items():
            _require_relative_path(relative, label="bound artifact")
            _require_sha256(digest, label=f"bound artifact {relative}")
        if not required.issubset(set(self.bound_artifacts)):
            missing = sorted(required.difference(self.bound_artifacts))
            raise ReplayAttestationError(
                f"attestation does not bind required artifacts: {missing!r}"
            )

        _require_positive_int(self.item_count, label="item count")
        if type(self.matched_count) is not int or self.matched_count < 0:
            raise ReplayAttestationError("matched count is invalid")

        if not isinstance(self.max_abs_residual, Mapping) or not self.max_abs_residual:
            raise ReplayAttestationError("max residual evidence is missing")
        for name, residual in self.max_abs_residual.items():
            if not isinstance(name, str) or not name:
                raise ReplayAttestationError("residual output name is invalid")
            _require_finite_nonneg(residual, label=f"residual[{name}]")

        if self.comparison_mode not in _COMPARISON_MODES:
            raise ReplayAttestationError("comparison mode is unsupported")
        if not isinstance(self.epsilon, Mapping):
            raise ReplayAttestationError("epsilon map is invalid")
        for name, value in self.epsilon.items():
            if not isinstance(name, str) or not name:
                raise ReplayAttestationError("epsilon output name is invalid")
            _require_finite_nonneg(value, label=f"epsilon[{name}]")

        if not isinstance(self.producer, Mapping) or set(self.producer) != _PRODUCER_FIELDS:
            raise ReplayAttestationError("producer identity is not exact")
        _require_release_id(self.producer.get("code_release_id"), label="producer code")
        _require_sha256(self.producer.get("module_sha256"), label="producer module")
        _require_sha256(self.producer.get("activation_sha256"), label="producer activation")

    @classmethod
    def from_manifest(cls, value: object) -> "ReplayAttestation":
        if not isinstance(value, Mapping):
            raise ReplayAttestationError("attestation manifest is not an object")
        if set(value) != _ATTESTATION_FIELDS:
            raise ReplayAttestationError("attestation manifest field set is not exact")
        if value["schema_version"] != REPLAY_ATTESTATION_SCHEMA:
            raise ReplayAttestationError("attestation schema is unsupported")
        return cls(
            component=value["component"],
            session_id=value["session_id"],
            bound_artifacts=dict(value["bound_artifacts"])
            if isinstance(value["bound_artifacts"], Mapping) else value["bound_artifacts"],
            model_release_id=value["model_release_id"],
            model_sha256=value["model_sha256"],
            comparison_mode=value["comparison_mode"],
            item_count=value["item_count"],
            matched_count=value["matched_count"],
            max_abs_residual=dict(value["max_abs_residual"])
            if isinstance(value["max_abs_residual"], Mapping) else value["max_abs_residual"],
            epsilon=dict(value["epsilon"])
            if isinstance(value["epsilon"], Mapping) else value["epsilon"],
            producer=dict(value["producer"])
            if isinstance(value["producer"], Mapping) else value["producer"],
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rehash_bound_artifact(session: Path, relative: str) -> str:
    path = session / relative
    # Reject symlinked components anywhere along the path so a bound artifact can
    # never be redirected outside the session after the attestation was written.
    probe = session
    for part in PurePosixPath(relative).parts:
        probe = probe / part
        if probe.is_symlink():
            raise ReplayAttestationError(
                f"bound artifact {relative} traverses a symlink"
            )
    if not path.is_file():
        raise ReplayAttestationError(f"bound artifact {relative} is missing")
    return _sha256_file(path)


def load_session_attestation(
    session_dir: "str | Path", component: str,
) -> bytes | None:
    """Return the raw attestation bytes for ``component``, or ``None`` if absent.

    Absence is a normal state — a session that has not yet been replayed simply
    stays un-authorized.  Presence, however, is load-bearing: a symlinked path or
    a symlinked parent is refused, so a present attestation can never be
    redirected outside the session.
    """

    if component not in ATTESTATION_RELATIVE_PATHS:
        raise ReplayAttestationError("unsupported component for replay attestation")
    session = Path(session_dir).absolute()
    relative = ATTESTATION_RELATIVE_PATHS[component]
    probe = session
    for part in PurePosixPath(relative).parts:
        probe = probe / part
        if probe.is_symlink():
            raise ReplayAttestationError(
                f"replay attestation path {relative} traverses a symlink"
            )
    path = session / relative
    if not path.exists():
        return None
    if not path.is_file():
        raise ReplayAttestationError("replay attestation is not a regular file")
    return path.read_bytes()


def verify_replay_attestation(
    session_dir: "str | Path",
    attestation: object,
    *,
    component: str,
    recorded_identity: Mapping[str, object],
    recorded_item_count: int,
    trust_verifier: "Callable[[ReplayAttestation], bool]",
    epsilon_ceilings: Mapping[str, Mapping[str, float]] = FROZEN_EPSILON_CEILINGS,
) -> dict[str, object]:
    """Return promotion evidence iff ``attestation`` authorizes ``component``.

    ``recorded_identity`` is the model identity taken from the session's own
    verified packets (``{"release_id", "model_sha256"}``); ``recorded_item_count``
    is that component's packet count.

    ``trust_verifier`` is the MANDATORY trust root: a callable that proves the
    attestation was produced by verified, trusted execution (e.g. it is signed by,
    or registered under, the activated production runtime) rather than hand-written
    into the session.  It is checked before any content claim is trusted, and a
    falsy result refuses authorization — so a session-local JSON file with a
    forged producer can never authorize promotion, no matter how well its bound
    hashes and model identity line up.  Raises :class:`ReplayAttestationError` on
    any failure, so a caller that reaches a return value has proof — not a claim.
    """

    if not callable(trust_verifier):
        raise ReplayAttestationError(
            "a replay attestation trust root is mandatory; refusing to authorize"
        )
    session = Path(session_dir).absolute()
    if component not in REQUIRED_BOUND_ARTIFACTS:
        raise ReplayAttestationError("unsupported component for replay attestation")

    if isinstance(attestation, ReplayAttestation):
        record = attestation
        record.__post_init__()
    elif isinstance(attestation, (bytes, bytearray)):
        try:
            parsed = json.loads(bytes(attestation).decode("utf-8"))
        except (UnicodeError, ValueError) as exc:
            raise ReplayAttestationError("attestation bytes are not valid JSON") from exc
        record = ReplayAttestation.from_manifest(parsed)
    elif isinstance(attestation, Mapping):
        record = ReplayAttestation.from_manifest(attestation)
    else:
        raise ReplayAttestationError("attestation must be a mapping, bytes, or record")

    if record.component != component:
        raise ReplayAttestationError("attestation component does not match the request")

    # 0a) Session binding by identity: the attestation must name THIS session, so
    # an attestation minted for one session cannot be dropped into another.
    if record.session_id != session.name:
        raise ReplayAttestationError(
            "attestation session id does not match the session directory"
        )

    # 0b) Trust root FIRST: a forged, hand-written attestation is refused before
    # any of its self-reported content is examined.
    try:
        vouched = bool(trust_verifier(record))
    except ReplayAttestationError:
        raise
    except Exception as exc:  # a trust verifier that errors must never authorize
        raise ReplayAttestationError(
            f"replay attestation trust verification failed: {exc}"
        ) from exc
    if not vouched:
        raise ReplayAttestationError(
            "replay attestation is not vouched by a trusted root; refusing to authorize"
        )

    # 1) Model identity must equal the identity the session's own packets recorded.
    if not isinstance(recorded_identity, Mapping):
        raise ReplayAttestationError("recorded model identity is missing")
    expected_release = recorded_identity.get("release_id")
    expected_model_sha = recorded_identity.get("model_sha256")
    if (
        record.model_release_id != expected_release
        or record.model_sha256 != expected_model_sha
    ):
        raise ReplayAttestationError(
            "attestation replayed a different model than the session recorded"
        )

    # 2) Total coverage: one matched replay per recorded packet, nothing skipped.
    _require_positive_int(recorded_item_count, label="recorded item count")
    if record.item_count != recorded_item_count:
        raise ReplayAttestationError(
            "attestation item count does not equal the recorded packet count"
        )
    if record.matched_count != record.item_count:
        raise ReplayAttestationError(
            "attestation reports items that did not match the replay"
        )

    # 3) Session binding: every required artifact still rehashes to the bound SHA.
    for relative, bound_sha in sorted(record.bound_artifacts.items()):
        actual = _rehash_bound_artifact(session, relative)
        if actual != bound_sha:
            raise ReplayAttestationError(
                f"bound artifact {relative} no longer matches the attested SHA-256"
            )

    # 4) Comparison policy: bit-exact, or within a FROZEN ceiling only.
    if record.comparison_mode == COMPARISON_BIT_EXACT:
        if record.epsilon:
            raise ReplayAttestationError(
                "bit-exact attestation must not declare any epsilon"
            )
        for name, residual in record.max_abs_residual.items():
            if float(residual) != 0.0:
                raise ReplayAttestationError(
                    "bit-exact attestation has a non-zero residual"
                )
    else:  # COMPARISON_FROZEN_EPSILON
        ceilings = epsilon_ceilings.get(component)
        if not ceilings:
            raise ReplayAttestationError(
                "frozen-epsilon replay is refused until a tolerance is calibrated "
                "and frozen for this component; bit-exact replay is required"
            )
        if set(record.epsilon) != set(ceilings):
            raise ReplayAttestationError(
                "attestation epsilon outputs do not match the frozen policy outputs"
            )
        for name, claimed in record.epsilon.items():
            ceiling = float(ceilings[name])
            if float(claimed) > ceiling:
                raise ReplayAttestationError(
                    f"attestation epsilon[{name}] exceeds the frozen ceiling"
                )
        # Every recorded residual must be covered by a declared epsilon.
        if set(record.max_abs_residual) - set(record.epsilon):
            raise ReplayAttestationError(
                "attestation has residual outputs without a frozen epsilon"
            )
        for name, residual in record.max_abs_residual.items():
            if float(residual) > float(record.epsilon[name]):
                raise ReplayAttestationError(
                    f"attestation residual[{name}] exceeds its declared epsilon"
                )

    return {
        "schema_version": REPLAY_ATTESTATION_SCHEMA,
        "component": record.component,
        "session_id": record.session_id,
        "production_authorized": True,
        "comparison_mode": record.comparison_mode,
        "item_count": record.item_count,
        "matched_count": record.matched_count,
        "model_release_id": record.model_release_id,
        "model_sha256": record.model_sha256,
        "bound_artifacts": dict(sorted(record.bound_artifacts.items())),
        "producer": dict(record.producer),
    }


__all__ = [
    "ATTESTATION_RELATIVE_PATHS",
    "COMPARISON_BIT_EXACT",
    "COMPARISON_FROZEN_EPSILON",
    "FROZEN_EPSILON_CEILINGS",
    "REPLAY_ATTESTATION_SCHEMA",
    "REQUIRED_BOUND_ARTIFACTS",
    "ReplayAttestation",
    "ReplayAttestationError",
    "load_session_attestation",
    "verify_replay_attestation",
]
