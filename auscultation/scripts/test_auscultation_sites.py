"""Tests for the shared auscultation-site semantics (the VLA "L" grounding)."""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import auscultation_sites as sites  # noqa: E402


def test_all_default_codes_are_mapped():
    codes = [f"{side}{i}" for side in ("R", "L") for i in range(1, 6)]
    assert sites.unmapped_codes(codes) == []


def test_instruction_is_anatomical_not_the_opaque_code():
    instr = sites.site_instruction("R1")
    assert "R1" not in instr                       # the opaque token is gone
    assert instr == "auscultate the right upper posterior lung field"
    # laterality is declared by the code's first letter (clinician-reviewed table)
    assert sites.site_anatomy("R3").startswith("right")
    assert sites.site_anatomy("L3").startswith("left")


def test_phrasing_is_deterministic_but_varied_with_anatomy_preserved():
    a = sites.phrase_site_instruction("R1", seed="sess1:R1")
    b = sites.phrase_site_instruction("R1", seed="sess1:R1")
    assert a == b                                  # reproducible given a seed
    assert "right upper posterior lung field" in a  # anatomy always present
    variants = {sites.phrase_site_instruction("R1", seed=f"s{i}:R1") for i in range(40)}
    assert len(variants) > 1                        # pool is actually explored across seeds
    assert all("{site}" in t for t in sites.INSTRUCTION_TEMPLATES)


def test_unmapped_code_falls_back_without_crashing():
    assert sites.site_anatomy("ZZ9") == sites.DEFAULT_SITE
    assert sites.unmapped_codes(["R1", "ZZ9"]) == ["ZZ9"]
    assert "posterior lung field" in sites.phrase_site_instruction("ZZ9", seed="x")


def test_instruction_map_is_self_describing():
    assert sites.instruction_map(["R1", "L5"]) == {
        "R1": "auscultate the right upper posterior lung field",
        "L5": "auscultate the left lower posterior lung field",
    }
