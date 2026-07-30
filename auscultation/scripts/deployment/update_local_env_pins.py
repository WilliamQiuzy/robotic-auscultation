#!/usr/bin/env python3
"""Atomically update only non-secret WAM deployment pins in the local .env.

Malformed non-empty lines are never silently discarded.  An operator may remove
one only by confirming the SHA-256 of its exact text, which avoids printing a
possibly secret orphan token in logs or command output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Dict, Iterable, Mapping, Optional, Set


class EnvPinError(RuntimeError):
    """The requested .env update is ambiguous or outside the pin contract."""


_ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_PIN_PATTERNS = {
    "WAM_CODE_RELEASE_ID": re.compile(r"^code_release-[0-9a-f]{20}$"),
    "WAM_PERCEPTION_ENVIRONMENT_ID": re.compile(
        r"^linbox-perception-[0-9a-f]{20}$"
    ),
    "WAM_PERCEPTION_ENVIRONMENT_LOCK_SHA256": _HASH,
    "WAM_PERCEPTION_RUNTIME_RELEASE_ID": re.compile(
        r"^perception_runtime-[0-9a-f]{20}$"
    ),
    "WAM_PERCEPTION_RUNTIME_LOCK_SHA256": _HASH,
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ) + "\n"


def _validate_pins(pins: Mapping[str, str]) -> Dict[str, str]:
    if set(pins) != set(_PIN_PATTERNS):
        raise EnvPinError("deployment pin key set is incomplete or contains extras")
    result: Dict[str, str] = {}
    for key, pattern in _PIN_PATTERNS.items():
        value = pins.get(key)
        if not isinstance(value, str) or pattern.fullmatch(value) is None:
            raise EnvPinError("deployment pin has an invalid shape: %s" % key)
        result[key] = value
    return result


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def update_env_pins(
    path: Path,
    pins: Mapping[str, str],
    *,
    remove_invalid_sha256: Optional[Iterable[str]] = None,
    preserve_invalid_sha256: Optional[Iterable[str]] = None,
) -> Dict[str, object]:
    """Update the exact deployment pin set while preserving all valid secrets."""
    target = Path(path)
    try:
        info = target.lstat()
    except FileNotFoundError as exc:
        raise EnvPinError(".env does not exist") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise EnvPinError(".env must be a regular non-symlink file")
    validated = _validate_pins(pins)
    confirmations: Set[str] = set(remove_invalid_sha256 or ())
    preserve_confirmations: Set[str] = set(preserve_invalid_sha256 or ())
    if confirmations & preserve_confirmations:
        raise EnvPinError("an invalid line cannot be both removed and preserved")
    if any(
        _HASH.fullmatch(item) is None
        for item in confirmations | preserve_confirmations
    ):
        raise EnvPinError("invalid-line confirmation must be a SHA-256")

    try:
        raw = target.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise EnvPinError(".env is not readable UTF-8") from exc
    after = target.lstat()
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if identity(info) != identity(after):
        raise EnvPinError(".env changed while being read")

    output = []
    seen = set()
    used_confirmations = set()
    used_preserve_confirmations = set()
    removed = 0
    preserved = 0
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            output.append(line)
            continue
        match = _ASSIGNMENT.fullmatch(line)
        if match is None:
            digest = hashlib.sha256(line.encode("utf-8")).hexdigest()
            if digest in preserve_confirmations:
                output.append(line)
                used_preserve_confirmations.add(digest)
                preserved += 1
                continue
            if digest not in confirmations:
                raise EnvPinError("invalid line %d requires exact SHA-256 confirmation" % line_number)
            used_confirmations.add(digest)
            removed += 1
            continue
        key = match.group(1)
        if key in seen:
            raise EnvPinError("duplicate .env assignment: %s" % key)
        seen.add(key)
        if key in validated:
            output.append("%s=%s" % (key, validated[key]))
        else:
            output.append(line)

    unused = (confirmations - used_confirmations) | (
        preserve_confirmations - used_preserve_confirmations
    )
    if unused:
        raise EnvPinError("unused invalid-line confirmation")
    missing = [key for key in _PIN_PATTERNS if key not in seen]
    if missing:
        if output and output[-1] != "":
            output.append("")
        output.append("# Immutable WAM deployment pins (managed locally; non-secret)")
        output.extend("%s=%s" % (key, validated[key]) for key in missing)
    body = ("\n".join(output).rstrip("\n") + "\n").encode("utf-8")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % target.name, dir=str(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
            os.fchmod(handle.fileno(), 0o600)
            handle.flush()
            os.fsync(handle.fileno())
        current = target.lstat()
        if identity(current) != identity(after):
            raise EnvPinError(".env changed before atomic replacement")
        os.replace(str(temporary), str(target))
        _fsync_directory(target.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return {
        "preserved_invalid_lines": preserved,
        "removed_invalid_lines": removed,
        "schema": "wam_local_env_pin_update_v1",
        "updated_keys": sorted(validated),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Atomically update immutable WAM release pins in a local .env"
    )
    parser.add_argument("--env", required=True)
    parser.add_argument("--code-release-id", required=True)
    parser.add_argument("--perception-environment-id", required=True)
    parser.add_argument("--perception-environment-lock-sha256", required=True)
    parser.add_argument("--perception-runtime-release-id", required=True)
    parser.add_argument("--perception-runtime-lock-sha256", required=True)
    parser.add_argument("--remove-invalid-line-sha256", action="append", default=[])
    parser.add_argument("--preserve-invalid-line-sha256", action="append", default=[])
    args = parser.parse_args(argv)
    result = update_env_pins(
        Path(args.env),
        {
            "WAM_CODE_RELEASE_ID": args.code_release_id,
            "WAM_PERCEPTION_ENVIRONMENT_ID": args.perception_environment_id,
            "WAM_PERCEPTION_ENVIRONMENT_LOCK_SHA256": (
                args.perception_environment_lock_sha256
            ),
            "WAM_PERCEPTION_RUNTIME_RELEASE_ID": args.perception_runtime_release_id,
            "WAM_PERCEPTION_RUNTIME_LOCK_SHA256": args.perception_runtime_lock_sha256,
        },
        remove_invalid_sha256=args.remove_invalid_line_sha256,
        preserve_invalid_sha256=args.preserve_invalid_line_sha256,
    )
    print(_canonical_json(result), end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EnvPinError as exc:
        print(_canonical_json({
            "error": str(exc),
            "error_type": type(exc).__name__,
            "schema": "wam_local_env_pin_update_error_v1",
        }), end="")
        raise SystemExit(1)
