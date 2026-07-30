from __future__ import annotations

from pathlib import Path

import pytest

from auscultation.ausc.models import iuv_model


_RUNTIME_ENV = (
    "AUSC_DINOV2_CHECKPOINT",
    "AUSC_DINOV2_REPO",
    "AUSC_OFFLINE_RUNTIME",
    "TORCH_HOME",
)


@pytest.fixture(autouse=True)
def _clear_runtime_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _RUNTIME_ENV:
        monkeypatch.delenv(name, raising=False)


def test_offline_encoder_fails_before_hub_load_when_assets_are_unbound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setenv("AUSC_OFFLINE_RUNTIME", "1")
    monkeypatch.setattr(iuv_model.torch.hub, "load", lambda *args, **kwargs: calls.append((args, kwargs)))

    with pytest.raises(RuntimeError, match="AUSC_DINOV2_REPO"):
        iuv_model.DinoV2Encoder()

    assert calls == []


def test_offline_encoder_loads_only_the_release_local_repository_and_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    torch_home = tmp_path / "torch_home"
    repository = torch_home / "hub" / "facebookresearch_dinov2_main"
    checkpoint = torch_home / "hub" / "checkpoints" / "dinov2_vitb14_pretrain.pth"
    (repository / "dinov2").mkdir(parents=True)
    (repository / "hubconf.py").write_text("# fixture\n", encoding="utf-8")
    (repository / "dinov2" / "__init__.py").write_text("# fixture\n", encoding="utf-8")
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixture weights")
    monkeypatch.setenv("AUSC_OFFLINE_RUNTIME", "1")
    monkeypatch.setenv("AUSC_DINOV2_REPO", str(repository))
    monkeypatch.setenv("AUSC_DINOV2_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("TORCH_HOME", str(torch_home))
    calls = []
    sentinel = object()

    def fake_load(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(iuv_model.torch.hub, "load", fake_load)

    assert iuv_model._load_dinov2_backbone("dinov2_vitb14") is sentinel
    assert calls == [
        ((str(repository), "dinov2_vitb14"), {"pretrained": True, "source": "local"}),
    ]


def test_offline_encoder_fails_before_hub_load_without_the_bound_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    torch_home = tmp_path / "torch_home"
    repository = torch_home / "hub" / "facebookresearch_dinov2_main"
    (repository / "dinov2").mkdir(parents=True)
    (repository / "hubconf.py").write_text("# fixture\n", encoding="utf-8")
    (repository / "dinov2" / "__init__.py").write_text("# fixture\n", encoding="utf-8")
    monkeypatch.setenv("AUSC_OFFLINE_RUNTIME", "1")
    monkeypatch.setenv("AUSC_DINOV2_REPO", str(repository))
    monkeypatch.setenv("TORCH_HOME", str(torch_home))
    calls = []
    monkeypatch.setattr(iuv_model.torch.hub, "load", lambda *args, **kwargs: calls.append((args, kwargs)))

    with pytest.raises(RuntimeError, match="AUSC_DINOV2_CHECKPOINT"):
        iuv_model._load_dinov2_backbone("dinov2_vitb14")

    assert calls == []


def test_offline_encoder_rejects_assets_outside_the_release_torch_cache_layout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    torch_home = tmp_path / "torch_home"
    repository = torch_home / "hub" / "facebookresearch_dinov2_main"
    checkpoint = tmp_path / "global-cache" / "dinov2_vitb14_pretrain.pth"
    (repository / "dinov2").mkdir(parents=True)
    (repository / "hubconf.py").write_text("# fixture\n", encoding="utf-8")
    (repository / "dinov2" / "__init__.py").write_text("# fixture\n", encoding="utf-8")
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixture weights")
    monkeypatch.setenv("AUSC_OFFLINE_RUNTIME", "1")
    monkeypatch.setenv("AUSC_DINOV2_REPO", str(repository))
    monkeypatch.setenv("AUSC_DINOV2_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("TORCH_HOME", str(torch_home))
    calls = []
    monkeypatch.setattr(iuv_model.torch.hub, "load", lambda *args, **kwargs: calls.append((args, kwargs)))

    with pytest.raises(RuntimeError, match="release-local torch cache layout"):
        iuv_model._load_dinov2_backbone("dinov2_vitb14")

    assert calls == []


def test_offline_encoder_rejects_an_unpinned_dino_variant_before_hub_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    torch_home = tmp_path / "torch_home"
    repository = torch_home / "hub" / "facebookresearch_dinov2_main"
    checkpoint = torch_home / "hub" / "checkpoints" / "dinov2_vitb14_pretrain.pth"
    (repository / "dinov2").mkdir(parents=True)
    (repository / "hubconf.py").write_text("# fixture\n", encoding="utf-8")
    (repository / "dinov2" / "__init__.py").write_text("# fixture\n", encoding="utf-8")
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixture weights")
    monkeypatch.setenv("AUSC_OFFLINE_RUNTIME", "1")
    monkeypatch.setenv("AUSC_DINOV2_REPO", str(repository))
    monkeypatch.setenv("AUSC_DINOV2_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("TORCH_HOME", str(torch_home))
    calls = []
    monkeypatch.setattr(iuv_model.torch.hub, "load", lambda *args, **kwargs: calls.append((args, kwargs)))

    with pytest.raises(RuntimeError, match="dinov2_vitb14"):
        iuv_model._load_dinov2_backbone("dinov2_vits14")

    assert calls == []
