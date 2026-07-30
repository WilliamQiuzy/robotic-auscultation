#!/usr/bin/env python3
"""No-training smoke gate for one completed WAM-core H200 environment."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import platform
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Dict, List, Mapping, Optional, Sequence


SMOKE_SCHEMA = "wam_h200_smoke_v1"
SYSTEM_LIBRARIES_SCHEMA = "wam_h200_system_libraries_v1"
_ENVIRONMENT_ID = re.compile(r"^wam-[0-9a-f]{20}$")
_PYTHON_VERSION = re.compile(r"^3\.12\.13$")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_PACKAGE_NAME = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$"
)
_EXPECTED_VERSIONS = {
    "lerobot": "0.3.3",
    "torch": "2.7.1",
    "torchaudio": "2.7.1",
    "torchvision": "0.22.1",
}
_IMPORT_CHECKS = {
    "annotation_ui",
    "audio",
    "dataset_reader",
    "perception",
    "reward_model",
}


class SmokeFailure(RuntimeError):
    """The environment is not the exact H200 WAM-core runtime target."""


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _base_version(value: object) -> str:
    if not isinstance(value, str):
        raise SmokeFailure("package version is not a string")
    return value.split("+", 1)[0]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def _normalize_package_name(value: object) -> str:
    if not isinstance(value, str) or not _PACKAGE_NAME.fullmatch(value):
        raise SmokeFailure("locked package inventory value is invalid")
    return re.sub(r"[-_.]+", "-", value).lower()


def _package_rows(value: object) -> List[Dict[str, str]]:
    if not isinstance(value, (list, tuple)) or not value:
        raise SmokeFailure("locked package inventory is empty")
    rows: List[Dict[str, str]] = []
    seen = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != {"name", "version"}:
            raise SmokeFailure("locked package inventory row is invalid")
        name = item.get("name")
        version = item.get("version")
        if not isinstance(version, str) or not version:
            raise SmokeFailure("locked package inventory value is invalid")
        normalized_name = _normalize_package_name(name)
        if normalized_name in seen:
            raise SmokeFailure("locked package inventory has a duplicate")
        seen.add(normalized_name)
        rows.append({"name": name, "version": version})
    if rows != sorted(rows, key=lambda row: _normalize_package_name(row["name"])):
        raise SmokeFailure("locked package inventory is not sorted")
    return rows


def _wheel_name_version(filename: object) -> tuple[str, str]:
    if not isinstance(filename, str) or not filename.endswith(".whl"):
        raise SmokeFailure("environment identity wheel filename is invalid")
    parts = filename[:-4].split("-")
    if len(parts) < 5 or not parts[1]:
        raise SmokeFailure("environment identity wheel filename is invalid")
    try:
        name = _normalize_package_name(parts[0])
    except SmokeFailure as exc:
        raise SmokeFailure("environment identity wheel filename is invalid") from exc
    return name, parts[1]


def evaluate_snapshot(
    environment_id: str,
    snapshot: Mapping[str, object],
    expected_packages: Sequence[Mapping[str, str]],
) -> Dict[str, object]:
    """Validate a fully probed runtime snapshot and emit the exact smoke document."""
    _require(bool(_ENVIRONMENT_ID.fullmatch(environment_id)), "environment id is invalid")
    expected_fields = {
        "abi",
        "capability",
        "cuda_available",
        "device_name",
        "implementation",
        "imports",
        "lerobot_version",
        "glibc_version",
        "libraries",
        "machine",
        "nvidia_driver_version",
        "packages",
        "platform",
        "python_version",
        "tensor_result",
        "torch_cuda",
        "torch_version",
        "torchaudio_version",
        "torchvision_version",
    }
    _require(set(snapshot) == expected_fields, "runtime snapshot fields are invalid")
    _require(snapshot.get("implementation") == "CPython", "runtime is not CPython")
    python_version = snapshot.get("python_version")
    _require(
        isinstance(python_version, str) and bool(_PYTHON_VERSION.fullmatch(python_version)),
        "runtime is not Python 3.12.13",
    )
    _require(snapshot.get("abi") == "cp312", "runtime ABI is not cp312")
    _require(snapshot.get("platform") == "linux", "runtime platform is not linux")
    _require(snapshot.get("machine") == "x86_64", "runtime machine is not x86_64")
    _require(snapshot.get("cuda_available") is True, "CUDA is unavailable")
    device_name = snapshot.get("device_name")
    _require(
        isinstance(device_name, str) and "H200" in device_name.upper(),
        "CUDA device is not an H200",
    )
    capability = snapshot.get("capability")
    _require(
        isinstance(capability, list)
        and len(capability) == 2
        and all(isinstance(item, int) for item in capability)
        and tuple(capability) >= (9, 0),
        "H200 compute capability is invalid",
    )
    tensor_result = snapshot.get("tensor_result")
    _require(
        isinstance(tensor_result, (int, float)) and float(tensor_result) == 14.0,
        "GPU tensor operation returned the wrong result",
    )
    _require(snapshot.get("torch_cuda") == "12.8", "PyTorch CUDA runtime is not 12.8")
    glibc_version = snapshot.get("glibc_version")
    _require(
        isinstance(glibc_version, str)
        and bool(re.fullmatch(r"\d+\.\d+(?:\.\d+)?", glibc_version))
        and tuple(int(part) for part in glibc_version.split(".")[:2]) >= (2, 31),
        "runtime glibc version is invalid",
    )
    driver = snapshot.get("nvidia_driver_version")
    _require(isinstance(driver, str) and bool(driver), "runtime NVIDIA driver is invalid")
    source_libraries = snapshot.get("libraries")
    _require(isinstance(source_libraries, list) and bool(source_libraries), "runtime libraries are empty")
    libraries: List[Dict[str, str]] = []
    seen_libraries = set()
    for row in source_libraries:
        _require(
            isinstance(row, dict)
            and set(row) == {"device", "inode", "name", "path", "sha256"},
            "runtime library row is invalid",
        )
        device = row.get("device")
        inode = row.get("inode")
        name = row.get("name")
        path = row.get("path")
        digest = row.get("sha256")
        _require(
            isinstance(device, int)
            and not isinstance(device, bool)
            and device >= 0
            and isinstance(inode, int)
            and not isinstance(inode, bool)
            and inode > 0
            and isinstance(name, str)
            and bool(name)
            and isinstance(path, str)
            and Path(path).is_absolute()
            and isinstance(digest, str)
            and bool(_HASH.fullmatch(digest)),
            "runtime library value is invalid",
        )
        key = (name, path)
        _require(key not in seen_libraries, "runtime library inventory has a duplicate")
        seen_libraries.add(key)
        libraries.append(
            {
                "device": device,
                "inode": inode,
                "name": name,
                "path": path,
                "sha256": digest,
            }
        )
    libraries.sort(key=lambda row: (row["name"], row["path"]))

    versions = {
        "lerobot": _base_version(snapshot.get("lerobot_version")),
        "torch": _base_version(snapshot.get("torch_version")),
        "torchaudio": _base_version(snapshot.get("torchaudio_version")),
        "torchvision": _base_version(snapshot.get("torchvision_version")),
    }
    _require(versions == _EXPECTED_VERSIONS, "WAM-core package versions are invalid")
    expected_package_rows = _package_rows(expected_packages)
    observed_package_rows = _package_rows(snapshot.get("packages"))
    _require(
        observed_package_rows == expected_package_rows,
        "installed package inventory does not match locked wheel inventory",
    )
    imports = snapshot.get("imports")
    _require(isinstance(imports, dict) and set(imports) == _IMPORT_CHECKS, "import probes are incomplete")
    for name in sorted(_IMPORT_CHECKS):
        _require(imports.get(name) is True, "%s import/probe failed" % name)

    return {
        "checks": {
            "annotation_ui": True,
            "audio": True,
            "cuda_h200": True,
            "dataset_reader": True,
            "gpu_tensor": True,
            "perception": True,
            "reward_model": True,
        },
        "environment_id": environment_id,
        "packages": expected_package_rows,
        "runtime": {
            "cuda_device_name": device_name,
            "python": {
                "abi": snapshot["abi"],
                "implementation": snapshot["implementation"],
                "machine": snapshot["machine"],
                "platform": snapshot["platform"],
                "version": snapshot["python_version"],
            },
            "system_libraries": {
                "cuda_device_name": device_name,
                "glibc_version": glibc_version,
                "libraries": libraries,
                "nvidia_driver_version": driver,
                "schema": SYSTEM_LIBRARIES_SCHEMA,
            },
            "torch_cuda": snapshot["torch_cuda"],
        },
        "schema": SMOKE_SCHEMA,
        "versions": versions,
    }


def _load_annotation_module(auscultation_root: Path) -> object:
    path = auscultation_root / "scripts" / "audio" / "preference_annotator.py"
    spec = importlib.util.spec_from_file_location("wam_preference_annotator", str(path))
    if spec is None or spec.loader is None:
        raise SmokeFailure("annotation UI module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _installed_package_inventory(
    expected_packages: Sequence[Mapping[str, str]],
) -> List[Dict[str, str]]:
    from packaging.version import InvalidVersion, Version

    expected = _package_rows(expected_packages)
    for row in expected:
        try:
            installed = importlib.metadata.version(row["name"])
        except importlib.metadata.PackageNotFoundError as exc:
            raise SmokeFailure("locked package is not installed: %s" % row["name"]) from exc
        try:
            matches = Version(installed) == Version(row["version"])
        except InvalidVersion:
            matches = installed == row["version"]
        if not matches:
            raise SmokeFailure(
                "locked package version drift: %s expected %s found %s"
                % (row["name"], row["version"], installed)
            )
    return expected


def _loaded_libcuda_inventory(
    expected_libraries: Sequence[Mapping[str, object]],
    *,
    maps_text: Optional[str] = None,
) -> List[Dict[str, object]]:
    if len(expected_libraries) != 1:
        raise SmokeFailure("loaded libcuda identity requires exactly one expected library")
    expected = expected_libraries[0]
    if not isinstance(expected, dict) or set(expected) != {
        "device",
        "inode",
        "name",
        "path",
        "sha256",
    }:
        raise SmokeFailure("loaded libcuda expected identity is invalid")
    if expected.get("name") != "libcuda.so.1":
        raise SmokeFailure("loaded libcuda expected name is invalid")
    if maps_text is None:
        try:
            maps_text = Path("/proc/self/maps").read_text(encoding="utf-8")
        except OSError as exc:
            raise SmokeFailure("loaded libcuda map cannot be read") from exc
    observed = set()
    for line in maps_text.splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) != 6:
            continue
        raw_path = parts[5]
        if not Path(raw_path).name.startswith("libcuda.so"):
            continue
        try:
            major, minor = (int(item, 16) for item in parts[3].split(":", 1))
            mapped_inode = int(parts[4])
            path = Path(raw_path).resolve(strict=True)
            info = path.lstat()
        except (OSError, ValueError) as exc:
            raise SmokeFailure("loaded libcuda map entry is invalid") from exc
        mapped_device = os.makedev(major, minor)
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISREG(info.st_mode)
            or (info.st_dev, info.st_ino) != (mapped_device, mapped_inode)
        ):
            raise SmokeFailure("loaded libcuda map inode does not match its file")
        observed.add((info.st_dev, info.st_ino, path.as_posix()))
    if len(observed) != 1:
        raise SmokeFailure("loaded libcuda map does not identify one file")
    device, inode, raw_path = next(iter(observed))
    path = Path(raw_path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(path), flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (device, inode):
            raise SmokeFailure("loaded libcuda changed while being opened")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise SmokeFailure("loaded libcuda changed while being hashed")
    finally:
        os.close(descriptor)
    row = {
        "device": device,
        "inode": inode,
        "name": "libcuda.so.1",
        "path": raw_path,
        "sha256": digest.hexdigest(),
    }
    if row != expected:
        raise SmokeFailure("loaded libcuda inode or hash does not match environment identity")
    return [row]


def _nvidia_driver_version() -> str:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise SmokeFailure("NVIDIA driver probe failed: %s" % completed.stderr.strip())
    versions = {line.strip() for line in completed.stdout.splitlines() if line.strip()}
    if len(versions) != 1:
        raise SmokeFailure("NVIDIA driver probe did not return one consistent version")
    return next(iter(versions))


def collect_runtime_snapshot(
    expected_packages: Sequence[Mapping[str, str]],
    expected_libraries: Sequence[Mapping[str, str]],
) -> Dict[str, object]:
    """Exercise imports and a tiny GPU op without loading data or training."""
    auscultation_root = Path(__file__).resolve().parents[2]
    if str(auscultation_root) not in sys.path:
        sys.path.insert(0, str(auscultation_root))
    installed_packages = _installed_package_inventory(expected_packages)

    try:
        import numpy as np
        import torch
        import torchaudio
        import torchvision
    except Exception as exc:
        raise SmokeFailure("PyTorch stack import failed: %s" % exc) from exc

    imports = {name: False for name in _IMPORT_CHECKS}
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        import pandas as pd
        import pyarrow  # noqa: F401

        _require(LeRobotDataset is not None, "LeRobot dataset reader is unavailable")
        parquet = io.BytesIO()
        pd.DataFrame({"frame_index": [0], "value": [1.0]}).to_parquet(
            parquet, index=False
        )
        parquet.seek(0)
        restored = pd.read_parquet(parquet)
        _require(
            restored.to_dict(orient="records")
            == [{"frame_index": 0, "value": 1.0}],
            "Parquet dataset roundtrip failed",
        )
        imports["dataset_reader"] = True
    except Exception as exc:
        raise SmokeFailure("LeRobot dataset reader import failed: %s" % exc) from exc

    try:
        import scipy.signal as scipy_signal
        import soundfile as soundfile
        from transformers import ASTModel, AutoFeatureExtractor

        _require(
            scipy_signal is not None
            and soundfile is not None
            and ASTModel is not None
            and AutoFeatureExtractor is not None,
            "audio libraries are unavailable",
        )
        wav = io.BytesIO()
        soundfile.write(wav, np.zeros(160, np.float32), 16000, format="WAV")
        wav.seek(0)
        loaded, sample_rate = soundfile.read(wav, dtype="float32")
        _require(sample_rate == 16000 and loaded.shape == (160,), "audio WAV roundtrip failed")
        resampled = torchaudio.functional.resample(torch.zeros(1, 160), 16000, 8000)
        _require(tuple(resampled.shape) == (1, 80), "torchaudio resample probe failed")
        imports["audio"] = True
    except Exception as exc:
        raise SmokeFailure("audio import/probe failed: %s" % exc) from exc

    try:
        from ausc.audio.reward_model import BTReward

        reward = BTReward(dim=4, hidden=4, seed=0)
        scores = reward.score(np.zeros((2, 4), np.float32))
        _require(tuple(scores.shape) == (2,), "reward model output shape is invalid")
        imports["reward_model"] = True
    except Exception as exc:
        raise SmokeFailure("reward model import/probe failed: %s" % exc) from exc

    try:
        import cv2
        from ausc.models.iuv_model import IUVHead

        resized = cv2.resize(np.zeros((2, 2, 3), np.uint8), (4, 4))
        _require(tuple(resized.shape) == (4, 4, 3), "OpenCV perception probe failed")
        head = IUVHead(8, mid=32).cuda().eval()
        with torch.no_grad():
            part, uv = head(torch.zeros((1, 8, 2, 2), device="cuda"), (4, 4))
        _require(tuple(part.shape) == (1, 25, 4, 4), "perception part-head shape is invalid")
        _require(tuple(uv.shape) == (1, 2, 4, 4), "perception UV-head shape is invalid")
        imports["perception"] = True
    except Exception as exc:
        raise SmokeFailure("perception import/probe failed: %s" % exc) from exc

    try:
        annotation = _load_annotation_module(auscultation_root)
        queue = annotation.build_pair_queue(["a.wav", "b.wav", "c.wav"], False, n_pairs=8)
        _require(bool(queue), "annotation UI pair builder produced no pairs")
        imports["annotation_ui"] = True
    except Exception as exc:
        raise SmokeFailure("annotation UI import/probe failed: %s" % exc) from exc

    cuda_available = bool(torch.cuda.is_available())
    device_name = torch.cuda.get_device_name(0) if cuda_available else ""
    capability = list(torch.cuda.get_device_capability(0)) if cuda_available else [0, 0]
    tensor_result = None
    if cuda_available:
        values = torch.arange(4, dtype=torch.float32, device="cuda")
        tensor_result = float((values * values).sum().item())
        torch.cuda.synchronize()

    cache_tag = getattr(sys.implementation, "cache_tag", "") or ""
    abi_match = re.fullmatch(r"cpython-(\d{3})", cache_tag)
    abi = "cp%s" % abi_match.group(1) if abi_match else cache_tag
    return {
        "abi": abi,
        "capability": capability,
        "cuda_available": cuda_available,
        "device_name": device_name,
        "implementation": platform.python_implementation(),
        "imports": imports,
        "lerobot_version": importlib.metadata.version("lerobot"),
        "glibc_version": platform.libc_ver()[1],
        "libraries": _loaded_libcuda_inventory(expected_libraries),
        "machine": platform.machine().lower(),
        "nvidia_driver_version": _nvidia_driver_version(),
        "packages": installed_packages,
        "platform": sys.platform,
        "python_version": platform.python_version(),
        "tensor_result": tensor_result,
        "torch_cuda": torch.version.cuda,
        "torch_version": torch.__version__,
        "torchaudio_version": torchaudio.__version__,
        "torchvision_version": torchvision.__version__,
    }


def _identity_expectations(path: Path, environment_id: str) -> Dict[str, object]:
    try:
        raw = path.read_bytes()
        identity = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SmokeFailure("environment identity cannot be read") from exc
    _require(isinstance(identity, dict), "environment identity is not an object")
    _require(raw == _canonical_json_bytes(identity), "environment identity is not canonical JSON")
    _require(identity.get("environment_id") == environment_id, "environment identity id mismatch")
    payload = identity.get("payload")
    _require(isinstance(payload, dict), "environment identity payload is invalid")
    payload_digest = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    _require(identity.get("payload_sha256") == payload_digest, "environment identity hash mismatch")
    _require(
        environment_id == "wam-%s" % payload_digest[:20],
        "environment identity id does not bind its payload",
    )
    wheels = payload.get("wheels")
    _require(isinstance(wheels, list), "environment identity wheel inventory is invalid")
    _require(
        all(
            isinstance(row, dict)
            and {"filename", "name", "sha256", "size", "version"} == set(row)
            for row in wheels
        ),
        "environment identity wheel row is invalid",
    )
    previous_filename = None
    seen_names = set()
    for row in wheels:
        filename = row.get("filename")
        name = row.get("name")
        version = row.get("version")
        digest = row.get("sha256")
        size = row.get("size")
        _require(
            isinstance(filename, str)
            and isinstance(name, str)
            and isinstance(version, str)
            and isinstance(digest, str)
            and bool(_HASH.fullmatch(digest))
            and isinstance(size, int)
            and not isinstance(size, bool)
            and size > 0,
            "environment identity wheel value is invalid",
        )
        parsed_name, parsed_version = _wheel_name_version(filename)
        _require(
            parsed_name == name and parsed_version == version,
            "environment identity wheel filename is inconsistent",
        )
        _require(
            previous_filename is None or filename > previous_filename,
            "environment identity wheels are not sorted",
        )
        _require(
            parsed_name not in seen_names,
            "environment identity has duplicate wheel packages",
        )
        previous_filename = filename
        seen_names.add(parsed_name)
    system_libraries = payload.get("system_libraries")
    _require(
        isinstance(system_libraries, dict)
        and set(system_libraries)
        == {
            "cuda_device_name",
            "glibc_version",
            "libraries",
            "nvidia_driver_version",
            "schema",
        }
        and isinstance(system_libraries.get("libraries"), list),
        "environment identity system libraries are invalid",
    )
    packages = [
        {"name": row.get("name"), "version": row.get("version")}
        for row in wheels
    ]
    packages.sort(key=lambda row: _normalize_package_name(row["name"]))
    return {
        "packages": _package_rows(packages),
        "system_libraries": system_libraries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the WAM-core H200 environment")
    parser.add_argument("--environment-id", required=True)
    parser.add_argument("--identity", required=True, type=Path)
    args = parser.parse_args()
    try:
        expected = _identity_expectations(args.identity, args.environment_id)
        expected_packages = expected["packages"]
        system_libraries = expected["system_libraries"]
        report = evaluate_snapshot(
            args.environment_id,
            collect_runtime_snapshot(
                expected_packages,  # type: ignore[arg-type]
                system_libraries["libraries"],  # type: ignore[index]
            ),
            expected_packages,
        )
    except SmokeFailure as exc:
        print("H200 smoke failed: %s" % exc, file=sys.stderr)
        return 2
    sys.stdout.buffer.write(_canonical_json_bytes(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
