import hashlib
import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("update_local_env_pins.py")
SPEC = importlib.util.spec_from_file_location("update_local_env_pins", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


PINS = {
    "WAM_CODE_RELEASE_ID": "code_release-0123456789abcdef0123",
    "WAM_PERCEPTION_ENVIRONMENT_ID": "linbox-perception-0123456789abcdef0123",
    "WAM_PERCEPTION_ENVIRONMENT_LOCK_SHA256": "a" * 64,
    "WAM_PERCEPTION_RUNTIME_RELEASE_ID": "perception_runtime-0123456789abcdef0123",
    "WAM_PERCEPTION_RUNTIME_LOCK_SHA256": "b" * 64,
}


def test_updates_pins_atomically_without_exposing_or_changing_secrets(tmp_path):
    env = tmp_path / ".env"
    env.write_text("SECRET=do-not-print\nLINBOX_HOST=linbox\n", encoding="utf-8")
    env.chmod(0o600)

    result = module.update_env_pins(env, PINS)

    body = env.read_text(encoding="utf-8")
    assert body.startswith("SECRET=do-not-print\nLINBOX_HOST=linbox\n")
    assert env.stat().st_mode & 0o777 == 0o600
    assert all(f"{key}={value}\n" in body for key, value in PINS.items())
    assert "do-not-print" not in str(result)
    assert result["updated_keys"] == sorted(PINS)


def test_replaces_existing_pin_once_and_rejects_duplicate_assignments(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "WAM_CODE_RELEASE_ID=code_release-ffffffffffffffffffff\n",
        encoding="utf-8",
    )
    module.update_env_pins(env, PINS)
    assert env.read_text(encoding="utf-8").count("WAM_CODE_RELEASE_ID=") == 1

    env.write_text("A=1\nA=2\n", encoding="utf-8")
    try:
        module.update_env_pins(env, PINS)
    except module.EnvPinError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate assignments must fail closed")


def test_invalid_lines_require_exact_sha256_allowlist_for_removal(tmp_path):
    env = tmp_path / ".env"
    invalid = "orphan-secret-token"
    env.write_text(f"{invalid}\nGOOD=value\n", encoding="utf-8")

    try:
        module.update_env_pins(env, PINS)
    except module.EnvPinError as exc:
        assert "invalid line 1" in str(exc)
    else:
        raise AssertionError("invalid env syntax must fail closed")

    digest = hashlib.sha256(invalid.encode("utf-8")).hexdigest()
    result = module.update_env_pins(env, PINS, remove_invalid_sha256={digest})
    assert env.read_text(encoding="utf-8").startswith("GOOD=value\n")
    assert result["removed_invalid_lines"] == 1


def test_exact_sha256_can_preserve_legacy_opaque_lines_without_exposing_them(tmp_path):
    env = tmp_path / ".env"
    opaque = "legacy-token-that-must-remain-byte-exact"
    env.write_text(f"{opaque}\nGOOD=value\n", encoding="utf-8")
    digest = hashlib.sha256(opaque.encode("utf-8")).hexdigest()

    result = module.update_env_pins(
        env, PINS, preserve_invalid_sha256={digest}
    )

    assert env.read_text(encoding="utf-8").startswith(f"{opaque}\nGOOD=value\n")
    assert result["preserved_invalid_lines"] == 1
    assert opaque not in str(result)


def test_rejects_invalid_pin_shapes_and_unused_removal_confirmation(tmp_path):
    env = tmp_path / ".env"
    env.write_text("GOOD=value\n", encoding="utf-8")

    for bad in (
        {**PINS, "WAM_CODE_RELEASE_ID": "latest"},
        {**PINS, "WAM_PERCEPTION_ENVIRONMENT_LOCK_SHA256": "short"},
    ):
        try:
            module.update_env_pins(env, bad)
        except module.EnvPinError:
            pass
        else:
            raise AssertionError("invalid deployment pin must fail closed")

    try:
        module.update_env_pins(env, PINS, remove_invalid_sha256={"c" * 64})
    except module.EnvPinError as exc:
        assert "unused invalid-line confirmation" in str(exc)
    else:
        raise AssertionError("unused confirmation must fail closed")

    try:
        module.update_env_pins(
            env, PINS, preserve_invalid_sha256={"d" * 64}
        )
    except module.EnvPinError as exc:
        assert "unused invalid-line confirmation" in str(exc)
    else:
        raise AssertionError("unused preserve confirmation must fail closed")
