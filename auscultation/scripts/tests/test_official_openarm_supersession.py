from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[3]
RECORD = ROOT / "docs/runs/2026-07-15-official-openarm-occlusion/SUPERSEDES.json"
OLD_LOCAL = ROOT / "docs/runs/2026-07-15-isaac-occlusion-experiment/linbox_release"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_old_proxy_release_is_named_invalidated_and_hash_frozen() -> None:
    value = json.loads(RECORD.read_text(encoding="utf-8"))

    assert value["schema"] == "wam_simulation_supersession_v1"
    assert value["old_release"]["code_release_id"] == "isaac_occlusion_code-f8983f87d67760436914"
    assert value["old_release"]["result_release_id"] == "isaac_occlusion_result-6fbf3036639f15982b06"
    assert value["old_release"]["immutable"] is True
    assert value["invalidated_claims"] == [
        "analytic_capsule_occlusion_is_representative_of_openarm_v1",
        "fixed_six_visible_landmarks_is_a_valid_occlusion_rule",
    ]
    assert value["invalid_assumptions"]["severe_capsule_radius_m"] == 0.14
    assert value["replacement"]["namespace"] == "2026-07-15-official-openarm-occlusion"
    assert value["replacement"]["status"] in {
        "pending_lingbox_execution",
        "complete_parameter_sensitivity_only",
        "incomplete_ik_strata",
    }
    assert value["replacement"]["production_activation"] is False
    assert value["replacement"]["fixed_visible_point_count_rule_used"] is False
    assert value["mutation_destination"] != value["old_release"]["remote_path"]
    assert "isaac_occlusion_result-6fbf3036639f15982b06" not in value["mutation_destination"]

    hashes = value["old_release"]["local_snapshot_sha256"]
    assert set(hashes) == {path.name for path in OLD_LOCAL.iterdir() if path.is_file()}
    assert all(PurePosixPath(name).name == name for name in hashes)
    assert {name: _sha256(OLD_LOCAL / name) for name in sorted(hashes)} == hashes
