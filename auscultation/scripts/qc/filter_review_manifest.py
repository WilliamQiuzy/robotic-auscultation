#!/usr/bin/env python3
"""Filter a perception manifest using a realback review CSV."""
import argparse
import csv
import json
import os
import re


DEFAULT_EXCLUDE_STATUSES = {"exclude", "bad_label", "bad_image", "not_back", "uncertain"}


def subject_key(sid: str) -> str:
    m = re.match(r"(syn\d+)", sid)
    if m:
        return m.group(1)
    m = re.match(r"(p\d+)_a\d+", sid)
    if m:
        return m.group(1)
    prev = None
    cur = sid
    while cur != prev:
        prev = cur
        cur = re.sub(r"_(aug\d+|canon|neg|clin\d+)$", "", cur)
    return cur


def load_excluded_subjects(review_csv: str, exclude_statuses=None, exclude_auto_flags=None):
    exclude_statuses = set(exclude_statuses or DEFAULT_EXCLUDE_STATUSES)
    exclude_auto_flags = set(exclude_auto_flags or ())
    excluded = set()
    with open(review_csv, newline="") as f:
        for row in csv.DictReader(f):
            status = (row.get("review_status") or "").strip().lower()
            flags = {x.strip() for x in (row.get("auto_flags") or "").split(";") if x.strip()}
            if status in exclude_statuses or (flags & exclude_auto_flags):
                excluded.add(row["subject"])
    return excluded


def filter_manifest(manifest_in: str, review_csv: str, manifest_out: str, exclude_statuses=None, exclude_auto_flags=None):
    excluded = load_excluded_subjects(
        review_csv,
        exclude_statuses=exclude_statuses,
        exclude_auto_flags=exclude_auto_flags,
    )
    kept = dropped = 0
    os.makedirs(os.path.dirname(manifest_out), exist_ok=True)
    with open(manifest_in) as src, open(manifest_out, "w") as dst:
        for line in src:
            row = json.loads(line)
            if subject_key(row["id"]) in excluded:
                dropped += 1
                continue
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
    return kept, dropped, excluded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_in", required=True)
    ap.add_argument("--review_csv", required=True)
    ap.add_argument("--manifest_out", required=True)
    ap.add_argument("--exclude_statuses", default=",".join(sorted(DEFAULT_EXCLUDE_STATUSES)))
    ap.add_argument("--exclude_auto_flags", default="", help="Optional auto flags to drop, e.g. tiny,low_res,few_visible")
    args = ap.parse_args()
    statuses = {s.strip().lower() for s in args.exclude_statuses.split(",") if s.strip()}
    auto_flags = {s.strip() for s in args.exclude_auto_flags.split(",") if s.strip()}
    kept, dropped, excluded = filter_manifest(args.manifest_in, args.review_csv, args.manifest_out, statuses, auto_flags)
    print(f"[filter_review_manifest] kept={kept} dropped={dropped} excluded_subjects={len(excluded)} -> {args.manifest_out}")


if __name__ == "__main__":
    main()
