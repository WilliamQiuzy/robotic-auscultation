"""Canonical-v2 transition construction tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import rl_readiness_check as ready


def _df():
    return pd.DataFrame({
        "episode_index": [0, 0, 1, 1],
        "observation.state": [[0.0, 0.1], [1.0, 1.1], [10.0, 10.1], [11.0, 11.1]],
        "observation.contact_force_n": [[4.0], [4.1], [3.9], [4.0]],
        "observation.contact_force_valid": [[True], [True], [True], [True]],
        "observation.contact_force_age_s": [[0.01], [0.01], [0.01], [0.01]],
        "action": [[0.2, 0.3], [0.4, 0.5], [0.6, 0.7], [0.8, 0.9]],
        "next.reward": [[0.0], [0.5], [0.0], [0.8]],
        "next.done": [[False], [True], [False], [True]],
    })


def test_transitions_never_cross_done_or_episode_boundary():
    s, a, r, sn, done, returns, good = ready.transitions_from_dataframe(_df())
    assert s.shape == (4, 3)  # two joints + current force, no future Q/gate
    assert np.allclose(sn[1], s[1])
    assert np.allclose(sn[3], s[3])
    assert not np.allclose(sn[1], s[2])
    assert done.tolist() == [0.0, 1.0, 0.0, 1.0]
    assert returns[1] == pytest.approx(0.5)
    assert returns[3] == pytest.approx(0.8)


def test_readiness_rejects_future_derived_observation_fields():
    df = _df()
    df["observation.q_usable"] = 0.9
    with pytest.raises(ValueError, match="future-derived"):
        ready.transitions_from_dataframe(df)


def test_cli_dir_accepts_canonical_converter_root(tmp_path, monkeypatch):
    parquet = tmp_path / "data" / "chunk-000" / "episode_000000.parquet"
    parquet.parent.mkdir(parents=True)
    parquet.touch()
    received = []
    monkeypatch.setattr(ready, "main", lambda dirs: not received.extend(dirs))

    assert ready.cli(["--dir", str(tmp_path)]) == 0

    assert received == [str(tmp_path)]


@pytest.mark.parametrize(("is_ready", "expected_exit"), [(True, 0), (False, 1)])
def test_cli_exit_code_reflects_rl_readiness(tmp_path, monkeypatch, is_ready, expected_exit):
    parquet = tmp_path / "data" / "chunk-000" / "episode_000000.parquet"
    parquet.parent.mkdir(parents=True)
    parquet.touch()
    monkeypatch.setattr(ready, "main", lambda dirs: is_ready)

    assert ready.cli(["--dir", str(tmp_path)]) == expected_exit
