#!/usr/bin/env python3
"""标注预填 —— 用校准后的 Q_proxy 给一条/一批肺音生成【预填好的标注 JSON】。
标注者据此【确认/纠正】而非从零标(active learning)。schema 见 annotation_schema.json。

单条:
  python scripts/audio/prelabel.py --wav rec.wav --point R3 \
    --force_target 4 --force_measured 4.1 --on_lung_point 1 --out rec.annot.json
批量:
  python scripts/audio/prelabel.py --glob "session1/*.wav" --point_from_name --out_dir annots/
"""
import argparse, glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "ausc", "audio"))
import proxies as P  # noqa: E402  (自动加载 calibration.json)

# 子分低于阈值 → 建议失败原因。只用【校准证明可靠】的检测子(snr→噪声/运动, flattop→削波),
# 避免 band/lowfreq 这类跨数据集不稳的子分造成误报。闷/掉接触/心音 → 留给人工(脚本不猜)。
REASON_RULES = [
    ("noise_or_motion", ["snr"], 0.40),      # 低SNR: 环境噪声或运动伪影(人工区分; 力信号也帮判)
    ("clipping", ["flattop"], 0.40),
]


def suggest(m):
    sub = m["subscores"]; q = m["q_proxy"]
    reasons = []
    for reason, keys, thr in REASON_RULES:
        if any(sub.get(k, 1.0) < thr for k in keys):
            reasons.append(reason)
    overall = 2 if q >= 0.70 else (1 if q >= 0.50 else 0)
    return overall, reasons


def annotate(wav, point=None, ft=None, fm=None, on_lung=None):
    m = P.analyze(wav)
    if "error" in m:
        return {"wav": wav, "error": m["error"]}
    overall, reasons = suggest(m)
    return {
        "id": os.path.splitext(os.path.basename(wav))[0],
        "wav": wav,
        "context": {"task": "lung", "point": point,
                    "force_target_n": ft, "force_measured_n": fm, "on_lung_point": on_lung},
        "respiratory": {"clean_cycles": None, "air_entry": None, "cycle_boundaries_s": []},
        "quality": {"overall": None, "usable_for_diagnosis": None, "failure_reasons": []},
        "content": {"per_cycle": []},
        "prelabel": {
            "q_proxy": round(m["q_proxy"], 4),
            "subscores": {k: round(v, 3) for k, v in m["subscores"].items()},
            "suggested_overall": overall,
            "suggested_failure_reasons": reasons,
            "_note": "Q_proxy 已对噪声/运动/削波校准(AUC≈0.85–0.99); 闷/掉接触/心音弱→请人工确认"},
        "annotator": None, "annotated_at": None, "notes": ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav"); ap.add_argument("--glob"); ap.add_argument("--out"); ap.add_argument("--out_dir")
    ap.add_argument("--point"); ap.add_argument("--point_from_name", action="store_true")
    ap.add_argument("--force_target", type=float); ap.add_argument("--force_measured", type=float)
    ap.add_argument("--on_lung_point", type=int)
    a = ap.parse_args()
    on = None if a.on_lung_point is None else bool(a.on_lung_point)

    if a.wav:
        rec = annotate(a.wav, a.point, a.force_target, a.force_measured, on)
        out = a.out or (os.path.splitext(a.wav)[0] + ".annot.json")
        json.dump(rec, open(out, "w"), indent=2, ensure_ascii=False)
        print(f"wrote {out}  q_proxy={rec.get('prelabel', {}).get('q_proxy')}  "
              f"suggest={rec.get('prelabel', {}).get('suggested_failure_reasons')}")
    elif a.glob:
        os.makedirs(a.out_dir or ".", exist_ok=True)
        n = 0
        for w in sorted(glob.glob(a.glob)):
            pt = os.path.basename(w).split("_")[0] if a.point_from_name else a.point
            rec = annotate(w, pt, a.force_target, a.force_measured, on)
            json.dump(rec, open(os.path.join(a.out_dir or ".",
                      os.path.splitext(os.path.basename(w))[0] + ".annot.json"), "w"),
                      indent=2, ensure_ascii=False)
            n += 1
        print(f"wrote {n} prefilled annotations to {a.out_dir}")
    else:
        ap.error("need --wav or --glob")


if __name__ == "__main__":
    main()
