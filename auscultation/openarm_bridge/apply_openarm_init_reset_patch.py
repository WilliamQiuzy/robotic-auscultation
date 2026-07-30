#!/usr/bin/env python3
"""CAS-guarded F0 applier for the verified Lingbox OpenArm init reset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
BASELINE_PATH = HERE / "lingbox_openarm_init_reset_baseline.json"
PATCH_PATH = HERE / "openarm_init.full_reset.patch"


class PatchApplicationError(RuntimeError):
    pass


def _git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()  # noqa: S324 - Git identity


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_baseline() -> dict:
    try:
        value = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PatchApplicationError("cannot read init reset baseline") from exc
    if value.get("schema") != "wam_openarm_init_reset_patch_baseline_v1":
        raise PatchApplicationError("unsupported init reset baseline")
    return value


def _read_regular(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise PatchApplicationError(f"required regular source is missing: {path}")
    return path.read_bytes()


def _audit_complete_sequence(payload: bytes) -> None:
    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise PatchApplicationError("init reset source is not UTF-8") from exc
    if '#include <openarm/damiao_motor/dm_motor_constants.hpp>' not in text:
        raise PatchApplicationError("init reset is missing DM motor constants")
    ordered = (
        "CallbackMode::IGNORE",
        "refresh_all()",
        "recv_all()",
        "enable_all()",
        "CallbackMode::PARAM",
        "RID::MST_ID",
        "recv_all()",
        "CallbackMode::STATE",
        "recv_all()",
    )
    cursor = 0
    for token in ordered:
        cursor = text.find(token, cursor)
        if cursor < 0:
            raise PatchApplicationError(
                f"init reset sequence is missing or out of order: {token}"
            )
        cursor += len(token)


def _stage(source: bytes, relative: str) -> bytes:
    with tempfile.TemporaryDirectory(prefix="wam-openarm-init-reset-") as raw:
        root = Path(raw)
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source)
        result = subprocess.run(
            ["patch", "--batch", "--forward", "-F0", "-p1", "-i", str(PATCH_PATH)],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            raise PatchApplicationError(
                "versioned init reset patch did not apply exactly: "
                + (result.stderr.strip() or result.stdout.strip())
            )
        return path.read_bytes()


def _atomic_replace(path: Path, payload: bytes, mode: int) -> None:
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{path.name}.wam-init-reset-", dir=path.parent
    )
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def apply_patch(teleop_root: Path, *, check_only: bool) -> dict:
    root = teleop_root.expanduser().resolve()
    if not root.is_dir():
        raise PatchApplicationError(f"teleop root is not a directory: {root}")
    baseline = _load_baseline()
    relative = str(baseline["source_path"])
    path = root / relative
    source = _read_regular(path)
    source_blob = _git_blob_sha1(source)
    source_sha = _sha256(source)

    if (
        source_blob == baseline["output_git_blob_sha1"]
        and source_sha == baseline["output_sha256"]
    ):
        _audit_complete_sequence(source)
        mode = "already_applied"
        output = source
    else:
        if (
            source_blob != baseline["source_git_blob_sha1"]
            or source_sha != baseline["source_sha256"]
            or not source_blob.startswith(baseline["source_git_blob_prefix"])
        ):
            raise PatchApplicationError(
                "git blob prefix mismatch for OpenArm init source: "
                f"expected {baseline['source_git_blob_prefix']}, received {source_blob[:7]}"
            )
        output = _stage(source, relative)
        if (
            _git_blob_sha1(output) != baseline["output_git_blob_sha1"]
            or _sha256(output) != baseline["output_sha256"]
            or not _git_blob_sha1(output).startswith(
                baseline["output_git_blob_prefix"]
            )
        ):
            raise PatchApplicationError("staged init reset output hash mismatch")
        _audit_complete_sequence(output)
        mode = "check" if check_only else "applied"

    result = {
        "schema": "wam_openarm_init_reset_patch_result_v1",
        "mode": mode,
        "teleop_root": str(root),
        "source_path": relative,
        "input_git_blob_sha1": source_blob,
        "input_sha256": source_sha,
        "output_git_blob_sha1": _git_blob_sha1(output),
        "output_sha256": _sha256(output),
        "sequence": baseline["required_sequence"],
    }
    if check_only or mode == "already_applied":
        return result

    current = _read_regular(path)
    if current != source:
        raise PatchApplicationError("OpenArm init source changed before CAS commit")
    _atomic_replace(path, output, path.stat().st_mode & 0o777)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teleop-root", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        result = apply_patch(args.teleop_root, check_only=args.check)
    except (OSError, PatchApplicationError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
