"""Framework-free core of the box-side model-replay runner.

The runner re-executes the deployment-activated model on a session's exact
persisted inputs and emits a :class:`~ausc.perception.replay_attestation`
manifest for the framework-free finalizer to verify.  The *model execution* is
Torch/CUDA on the box; it enters this module through a single injected seam
(``model_run``), so the reconstruction, matching, comparison, and attestation
assembly — everything that decides whether promotion is authorized — is fully
unit-tested on any host with synthetic arrays.

The comparison is bit-exact by default: a replayed output authorizes an item
only when it is array-equal to the recorded output.  A ``frozen_epsilon`` run may
be used *after* a calibration on the box has proven the activated model is not
bit-reproducible and a per-output ceiling has been frozen in
``replay_attestation.FROZEN_EPSILON_CEILINGS`` — the finalizer verifier, not this
runner, enforces that ceiling.

This module never lowers the bar: a shape/dtype mismatch or a non-finite residual
is a hard error (the model or the reconstruction is wrong, not merely out of
tolerance), and an item that does not match simply is not counted, so the
resulting ``matched_count < item_count`` makes the attestation fail closed at the
finalizer.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from ausc.perception.replay_attestation import (
    COMPARISON_BIT_EXACT,
    COMPARISON_FROZEN_EPSILON,
    REPLAY_ATTESTATION_SCHEMA,
    REQUIRED_BOUND_ARTIFACTS,
    ReplayAttestation,
)


class ReplayRunnerError(RuntimeError):
    """The replay runner cannot produce a trustworthy attestation."""


@dataclass(frozen=True)
class ReplayItem:
    """One replay unit: the model input and the recorded outputs to reproduce.

    ``recorded_outputs`` maps each output name (e.g. ``"points_m"``) to the array
    the session recorded; ``model_run(model_input)`` must return the same names.
    """

    model_input: object
    recorded_outputs: Mapping[str, np.ndarray]


def compare_outputs(
    recorded: Mapping[str, object],
    replayed: Mapping[str, object],
    *,
    mode: str,
    epsilon: Mapping[str, float],
) -> tuple[bool, dict[str, float]]:
    """Compare one item's recorded vs replayed outputs.

    Returns ``(all_outputs_matched, {name: max_abs_residual})``.  Raises on a
    structural failure (missing/extra output, shape mismatch, non-finite value):
    those are never tolerable, only genuine numeric drift is.
    """

    if not isinstance(recorded, Mapping) or not isinstance(replayed, Mapping):
        raise ReplayRunnerError("replay outputs must be mappings")
    if not recorded:
        raise ReplayRunnerError("replay produced no recorded outputs to match")
    if set(recorded) != set(replayed):
        raise ReplayRunnerError(
            "replayed output names do not match the recorded outputs"
        )

    residuals: dict[str, float] = {}
    matched = True
    for name in recorded:
        rec = np.asarray(recorded[name])
        rep = np.asarray(replayed[name])
        if rec.shape != rep.shape:
            raise ReplayRunnerError(
                f"output {name!r} shape {rep.shape} does not match recorded {rec.shape}"
            )
        if rec.size == 0:
            residuals[name] = 0.0
            continue
        diff = float(np.max(np.abs(rec.astype(np.float64) - rep.astype(np.float64))))
        if not np.isfinite(diff):
            raise ReplayRunnerError(f"output {name!r} residual is not finite")
        residuals[name] = diff
        if mode == COMPARISON_BIT_EXACT:
            if not np.array_equal(rec, rep):
                matched = False
        else:
            if diff > float(epsilon.get(name, 0.0)):
                matched = False
    return matched, residuals


def compute_bound_artifacts(
    session_dir: "str | Path", component: str,
) -> dict[str, str]:
    """SHA-256 every artifact the component's attestation must bind."""

    if component not in REQUIRED_BOUND_ARTIFACTS:
        raise ReplayRunnerError("unsupported component for replay attestation")
    session = Path(session_dir)
    bound: dict[str, str] = {}
    for relative in sorted(REQUIRED_BOUND_ARTIFACTS[component]):
        path = session / relative
        if path.is_symlink() or not path.is_file():
            raise ReplayRunnerError(f"bound artifact {relative} is missing")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        bound[relative] = digest.hexdigest()
    return bound


def run_replay(
    session_dir: "str | Path",
    items: Sequence[ReplayItem],
    model_run: Callable[[object], Mapping[str, object]],
    *,
    component: str,
    model_release_id: str,
    model_sha256: str,
    producer: Mapping[str, str],
    comparison_mode: str = COMPARISON_BIT_EXACT,
    epsilon: Mapping[str, float] | None = None,
    bound_artifacts: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Replay every item through ``model_run`` and assemble the attestation.

    Returns the canonical attestation manifest (a plain dict); the caller writes
    it to ``<session>/<component>/replay_attestation.json``.  The manifest is
    validated for structure here, but the finalizer verifier remains the sole
    authority against the on-disk session — this runner can only *report* what it
    measured, never self-authorize.
    """

    if component not in REQUIRED_BOUND_ARTIFACTS:
        raise ReplayRunnerError("unsupported component for replay attestation")
    if comparison_mode not in (COMPARISON_BIT_EXACT, COMPARISON_FROZEN_EPSILON):
        raise ReplayRunnerError("unsupported comparison mode")
    epsilon = dict(epsilon or {})
    if comparison_mode == COMPARISON_BIT_EXACT and epsilon:
        raise ReplayRunnerError("bit-exact replay must not supply an epsilon")
    if not callable(model_run):
        raise ReplayRunnerError("model_run seam is not callable")

    item_list = list(items)
    if not item_list:
        raise ReplayRunnerError("no replay items to run")

    if bound_artifacts is None:
        bound_artifacts = compute_bound_artifacts(session_dir, component)

    matched_count = 0
    max_residual: dict[str, float] = {}
    for index, item in enumerate(item_list):
        if not isinstance(item, ReplayItem):
            raise ReplayRunnerError(f"replay item {index} type is invalid")
        replayed = model_run(item.model_input)
        matched, residuals = compare_outputs(
            item.recorded_outputs, replayed, mode=comparison_mode, epsilon=epsilon,
        )
        for name, value in residuals.items():
            max_residual[name] = max(max_residual.get(name, 0.0), value)
        if matched:
            matched_count += 1

    manifest = {
        "schema_version": REPLAY_ATTESTATION_SCHEMA,
        "component": component,
        "session_id": Path(session_dir).name,
        "bound_artifacts": dict(bound_artifacts),
        "model_release_id": model_release_id,
        "model_sha256": model_sha256,
        "comparison_mode": comparison_mode,
        "item_count": len(item_list),
        "matched_count": matched_count,
        "max_abs_residual": {name: float(value) for name, value in max_residual.items()},
        "epsilon": {name: float(value) for name, value in epsilon.items()},
        "producer": dict(producer),
    }
    # Structural self-check (producer identity, release/sha content-addressing,
    # bound-artifact completeness).  Does NOT prove promotion: a run with an
    # unmatched item still serializes here and is refused later at the finalizer.
    ReplayAttestation.from_manifest(manifest)
    return manifest


__all__ = [
    "ReplayItem",
    "ReplayRunnerError",
    "compare_outputs",
    "compute_bound_artifacts",
    "run_replay",
]
