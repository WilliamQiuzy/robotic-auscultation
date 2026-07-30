from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ausc.perception.trusted_model_loader import (
    TRUSTED_MODEL_ACTIVATION_SCHEMA,
    TrustedModelActivation,
    TrustedModelLoader,
    TrustedModelLoaderError,
    require_trusted_model_loader,
)


class _Runner:
    def __init__(self, model_bytes, activation):
        self._n = len(model_bytes)
        self.activation = activation

    def run(self, _inputs):
        return self._n


def _weights_constructor(model_bytes, activation):
    return _Runner(model_bytes, activation)


def _activation(model_sha256):
    return TrustedModelActivation(
        release_id=f"perc-q-{model_sha256[:20]}",
        model_relative_path="model/best.pt",
        model_sha256=model_sha256,
        preprocessing_id="pre-v1",
        postprocessing_id="post-v1",
        tracker_id="tracker-v1",
        calibration_id="calib-v1",
        runtime_id="runtime-v1",
    )


def _release(tmp_path, body=b"real-weight-bytes"):
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "best.pt").write_bytes(body)
    return hashlib.sha256(body).hexdigest()


def test_trusted_loader_loads_a_verified_release(tmp_path):
    sha = _release(tmp_path)
    loader = TrustedModelLoader(
        release_root=tmp_path,
        activation=_activation(sha),
        weights_constructor=_weights_constructor,
    )
    runner = loader.load()
    assert callable(runner.run)
    assert runner.activation.release_id == f"perc-q-{sha[:20]}"


def test_load_fails_closed_on_weight_sha_mismatch(tmp_path):
    sha = _release(tmp_path, body=b"real-weight-bytes")
    activation = _activation(sha)
    # Weights on disk are swapped after activation was pinned.
    (tmp_path / "model" / "best.pt").write_bytes(b"tampered-weight-bytes")
    loader = TrustedModelLoader(
        release_root=tmp_path, activation=activation,
        weights_constructor=_weights_constructor,
    )
    with pytest.raises(TrustedModelLoaderError, match="SHA-256 does not match"):
        loader.load()


def test_load_fails_closed_when_model_file_missing(tmp_path):
    sha = hashlib.sha256(b"never-written").hexdigest()
    loader = TrustedModelLoader(
        release_root=tmp_path, activation=_activation(sha),
        weights_constructor=_weights_constructor,
    )
    with pytest.raises(TrustedModelLoaderError, match="model file is missing"):
        loader.load()


def test_activation_rejects_release_id_that_does_not_address_the_weights():
    good = hashlib.sha256(b"x").hexdigest()
    with pytest.raises(TrustedModelLoaderError, match="content-address"):
        TrustedModelActivation(
            release_id="perc-q-" + "0" * 20,  # wrong first-20
            model_relative_path="model/best.pt",
            model_sha256=good,
            preprocessing_id="p", postprocessing_id="q", tracker_id="t",
            calibration_id="c", runtime_id="r",
        )


def test_activation_rejects_unsafe_model_path():
    sha = hashlib.sha256(b"x").hexdigest()
    with pytest.raises(TrustedModelLoaderError, match="path is unsafe"):
        TrustedModelActivation(
            release_id=f"perc-q-{sha[:20]}",
            model_relative_path="../escape/best.pt",
            model_sha256=sha,
            preprocessing_id="p", postprocessing_id="q", tracker_id="t",
            calibration_id="c", runtime_id="r",
        )


def test_loader_rejects_non_runnable_constructor_output(tmp_path):
    sha = _release(tmp_path)
    loader = TrustedModelLoader(
        release_root=tmp_path, activation=_activation(sha),
        weights_constructor=lambda _b, _a: object(),  # no .run
    )
    with pytest.raises(TrustedModelLoaderError, match="runnable"):
        loader.load()


def test_require_trusted_model_loader_refuses_an_arbitrary_callable(tmp_path):
    # The whole point: a bare callable can never stand in for the trusted loader.
    with pytest.raises(TrustedModelLoaderError, match="arbitrary callable"):
        require_trusted_model_loader(lambda _inputs: 0.99)
    with pytest.raises(TrustedModelLoaderError, match="arbitrary callable"):
        require_trusted_model_loader(object())
    sha = _release(tmp_path)
    real = TrustedModelLoader(
        release_root=tmp_path, activation=_activation(sha),
        weights_constructor=_weights_constructor,
    )
    assert require_trusted_model_loader(real) is real


def test_activation_manifest_round_trips_and_rejects_bad_schema():
    sha = hashlib.sha256(b"x").hexdigest()
    activation = _activation(sha)
    manifest = {
        "schema_version": TRUSTED_MODEL_ACTIVATION_SCHEMA,
        "release_id": activation.release_id,
        "model_relative_path": activation.model_relative_path,
        "model_sha256": activation.model_sha256,
        "preprocessing_id": activation.preprocessing_id,
        "postprocessing_id": activation.postprocessing_id,
        "tracker_id": activation.tracker_id,
        "calibration_id": activation.calibration_id,
        "runtime_id": activation.runtime_id,
    }
    assert TrustedModelActivation.from_manifest(manifest) == activation
    bad = dict(manifest, schema_version="wam_other")
    with pytest.raises(TrustedModelLoaderError, match="schema is unsupported"):
        TrustedModelActivation.from_manifest(bad)
    with pytest.raises(TrustedModelLoaderError, match="field set is not exact"):
        TrustedModelActivation.from_manifest({**manifest, "extra": 1})
