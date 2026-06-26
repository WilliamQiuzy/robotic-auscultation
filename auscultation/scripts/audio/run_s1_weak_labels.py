#!/usr/bin/env python3
"""S1 — 用自监督声学代理给【肺音】打弱质量标 + 在 SPRSound 'Poor Quality' 上校准。

对每个数据集的每条 wav 跑 ausc/audio/proxies.analyze → 写 manifest_<ds>.jsonl
(每行: id, wav, dataset, 指标, q_proxy)。
对 SPRSound: 读 record_annotation, 把 'Poor Quality' 当真值, 算
  - badness=(1-q_proxy) 判别 Poor Quality 的 AUC(代理有效性)
  - 每个子分单独 AUC(哪个代理最能分出差录音)
  - PQ vs 非PQ 的 q_proxy 均值
→ 写 calib_sprsound.json。

用法(pod 上):
  python scripts/audio/run_s1_weak_labels.py --root /workspace/data/audio \
    --dataset all --out_dir /workspace/data/audio/_s1 --limit 800 \
    --hflung_wav_dir /tmp/hflung
"""
import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "ausc"))
sys.path.insert(0, os.path.join(ROOT, "ausc", "audio"))
import proxies as P  # noqa: E402


def stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def sprsound_label_map(root):
    """SPRSound: {stem: record_annotation}, 从 Classification 的 json 递归读。"""
    out = {}
    base = os.path.join(root, "sprsound", "Classification")
    for j in glob.glob(os.path.join(base, "**", "*.json"), recursive=True):
        try:
            rec = json.load(open(j)).get("record_annotation")
            if rec:
                out[stem(j)] = rec
        except Exception:
            continue
    return out


def list_wavs(root, dataset, hflung_wav_dir):
    """各数据集的 wav 路径列表。"""
    if dataset == "sprsound":
        pats = [
            os.path.join(root, "sprsound", "Classification", "train_classification_wav", "*.wav"),
            os.path.join(root, "sprsound", "Classification", "valid_classification_wav", "*.wav"),
        ]
        ws = []
        for p in pats:
            ws += glob.glob(p)
        return ws
    if dataset == "icbhi":
        return glob.glob(os.path.join(root, "icbhi", "**", "*.wav"), recursive=True)
    if dataset == "hf_lung":
        d = hflung_wav_dir or os.path.join(root, "hf_lung_v1")
        return glob.glob(os.path.join(d, "**", "*.wav"), recursive=True)
    raise ValueError(dataset)


def run_dataset(root, dataset, out_dir, limit, hflung_wav_dir):
    wavs = sorted(list_wavs(root, dataset, hflung_wav_dir))
    total_found = len(wavs)
    if limit and limit > 0:
        wavs = wavs[:limit]
    os.makedirs(out_dir, exist_ok=True)
    man_path = os.path.join(out_dir, f"manifest_{dataset}.jsonl")
    qs, rows, errs = [], 0, 0
    t0 = time.time()
    with open(man_path, "w") as f:
        for i, w in enumerate(wavs):
            m = P.analyze(w)
            if "error" in m:
                errs += 1
                continue
            rec = {"id": stem(w), "wav": w, "dataset": dataset,
                   "fs_native": m["fs_native"], "dur_s": m["dur_s"],
                   "q_proxy": round(m["q_proxy"], 4),
                   "metrics": {k: round(v, 4) for k, v in m.items()
                               if isinstance(v, (int, float)) and k not in ("fs_native",)},
                   "subscores": {k: round(v, 4) for k, v in m["subscores"].items()}}
            f.write(json.dumps(rec) + "\n")
            qs.append(m["q_proxy"]); rows += 1
            if i % 100 == 0:
                rate = (i + 1) / max(1e-6, time.time() - t0)
                print(f"  [{dataset}] {i+1}/{len(wavs)}  {rate:.1f} wav/s  errs={errs}", flush=True)
    import numpy as np
    summ = {"dataset": dataset, "found": total_found, "processed": rows, "errors": errs,
            "q_mean": round(float(np.mean(qs)), 4) if qs else None,
            "q_median": round(float(np.median(qs)), 4) if qs else None,
            "q_p10": round(float(np.percentile(qs, 10)), 4) if qs else None,
            "q_p90": round(float(np.percentile(qs, 90)), 4) if qs else None,
            "manifest": man_path, "sec": round(time.time() - t0, 1)}
    print(f"[{dataset}] done: {summ}", flush=True)
    return summ, man_path


def calibrate_sprsound(root, out_dir, man_path):
    """读 SPRSound 标 + manifest, 算 q_proxy/子分 对 'Poor Quality' 的判别力。"""
    import numpy as np
    labels = sprsound_label_map(root)
    if not labels:
        print("[calib] no SPRSound labels found — skip", flush=True)
        return None
    rows = [json.loads(l) for l in open(man_path)]
    y, qp, subs = [], [], []
    for r in rows:
        rec = labels.get(r["id"])
        if rec is None:
            continue
        y.append(1 if rec == "Poor Quality" else 0)
        qp.append(r["q_proxy"]); subs.append(r["subscores"])
    if not y:
        print("[calib] no label/manifest join — skip", flush=True)
        return None
    y = np.asarray(y); qp = np.asarray(qp)
    npos = int(y.sum())
    badness = 1.0 - qp                                   # 代理"差"分, 应在 Poor Quality 上偏高
    sub_keys = list(subs[0].keys())
    sub_auc = {}
    for k in sub_keys:
        sk = np.array([s[k] for s in subs])
        sub_auc[k] = round(P.auc(1.0 - sk, y), 4)        # 1-子分=该维度的"差"
    out = {
        "n": int(len(y)), "n_poor_quality": npos, "pq_rate": round(npos / len(y), 4),
        "auc_qproxy_vs_PQ": round(P.auc(badness, y), 4),  # 主指标: >0.5 说明代理有效
        "q_mean_poor": round(float(qp[y == 1].mean()), 4) if npos else None,
        "q_mean_good": round(float(qp[y == 0].mean()), 4),
        "subscore_auc_vs_PQ": dict(sorted(sub_auc.items(), key=lambda kv: -kv[1])),
        "label_dist": {k: int(v) for k, v in zip(*np.unique([labels[r["id"]] for r in rows
                       if r["id"] in labels], return_counts=True))},
    }
    json.dump(out, open(os.path.join(out_dir, "calib_sprsound.json"), "w"), indent=2)
    print(f"[calib] {json.dumps(out, ensure_ascii=False)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="data/audio 根目录")
    ap.add_argument("--dataset", default="all", choices=["sprsound", "icbhi", "hf_lung", "all"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit", type=int, default=0, help="每集最多处理几条(0=全部)")
    ap.add_argument("--hflung_wav_dir", default="", help="HF_Lung 解压 wav 目录(默认 root/hf_lung_v1)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dsets = ["sprsound", "icbhi", "hf_lung"] if args.dataset == "all" else [args.dataset]
    report = {}
    for ds in dsets:
        summ, man = run_dataset(args.root, ds, args.out_dir, args.limit, args.hflung_wav_dir)
        report[ds] = summ
        if ds == "sprsound":
            summ["calibration"] = calibrate_sprsound(args.root, args.out_dir, man)
    json.dump(report, open(os.path.join(args.out_dir, "s1_report.json"), "w"), indent=2)
    print("\n=== S1 REPORT ===", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
