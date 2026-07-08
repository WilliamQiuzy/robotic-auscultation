"""Curate a doctor annotation pool from SPRSound (real lung sounds spanning quality).

For a preference-annotation pilot we want a few hundred REAL clips that span the quality range, so
doctors' pairwise comparisons are informative. SPRSound's `record_annotation` already gives strata
(Poor Quality vs Normal / CAS / DAS / CAS & DAS). This samples across them into a flat pool that
`preference_annotator.py --clips <pool>` serves. Runs where SPRSound is cloned (the H200).

    python build_sprsound_pool.py --sprsound ~/data/SPRSound --pool ~/wam_ausc/data/annotation_pool_sprsound
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import shutil


def build(sprsound_root: str, pool: str, n_poor: int = 120, n_per_class: int = 70, seed: int = 0) -> dict:
    wavs = {os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(os.path.join(sprsound_root, "**", "*.wav"), recursive=True)}
    poor: list[str] = []
    byclass: dict[str, list[str]] = {}
    for jp in glob.glob(os.path.join(sprsound_root, "**", "*.json"), recursive=True):
        s = os.path.splitext(os.path.basename(jp))[0]
        if s not in wavs:
            continue
        try:
            ra = json.load(open(jp)).get("record_annotation")
        except Exception:
            continue
        if ra is None:
            continue
        if ra.strip().lower() == "poor quality":
            poor.append(wavs[s])
        else:
            byclass.setdefault(ra.strip(), []).append(wavs[s])
    rng = random.Random(seed)
    sel = rng.sample(poor, min(n_poor, len(poor)))
    picks = {"Poor Quality": len(sel)}
    for cls, lst in sorted(byclass.items()):
        k = rng.sample(lst, min(n_per_class, len(lst)))
        sel += k; picks[cls] = len(k)
    os.makedirs(pool, exist_ok=True)
    for p in sel:
        shutil.copy(p, os.path.join(pool, os.path.basename(p)))
    print(f"[pool] {len(sel)} clips -> {pool}")
    for c, n in picks.items():
        print(f"    {c:16s} {n}")
    return {"total": len(sel), "picks": picks}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sprsound", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--n-poor", type=int, default=120)
    ap.add_argument("--n-per-class", type=int, default=70)
    a = ap.parse_args()
    build(a.sprsound, a.pool, a.n_poor, a.n_per_class)
