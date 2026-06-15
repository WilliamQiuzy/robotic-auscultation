#!/usr/bin/env python3
"""用【合成退化集】校准肺音质量代理 —— 对【我们的部署失败模式】定阈值(anchors)+权重。

为什么:S1 证明手工 SQI 对不上 SPRSound 临床 'Poor Quality'(那是"可标注性"不是信号脏)。
但我们部署的"差" = 机械臂落点失败 = 明显信号劣化。于是:
  取干净肺音 → 合成我们会遇到的退化(噪声/运动摩擦/削波/没贴住/掉接触/心音污染)→
  这就有了【匹配我们失败模式】的 good/bad 真值 → 据此重估每个子分的 (good,bad) 锚点 +
  按各子分对 good/bad 的判别力(AUC)定权重 → 写 calibration.json(proxies.load_calibration 加载)。

用法(pod):
  python scripts/audio/calibrate_proxies.py --clean_glob "/tmp/hflung/**/*.wav" \
    --n_sources 150 --out_dir /workspace/data/audio/_calib \
    --calib_out /workspace/code/auscultation/ausc/audio/calibration.json
"""
import argparse, glob, json, os, sys
import numpy as np
from scipy.signal import butter, filtfilt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "ausc", "audio"))
import proxies as P  # noqa: E402

SR = P.TARGET_SR


# ── 退化算子(输入 16k mono float, 峰值~0.3)──────────────────────────────────
def _rms(x):
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def deg_noise(x, snr_db, rng):
    pn = (_rms(x) ** 2) / (10 ** (snr_db / 10))
    return (x + rng.standard_normal(len(x)).astype(np.float32) * np.sqrt(pn)).astype(np.float32)


def deg_motion(x, rng):                                  # 低频隆隆 + 摩擦突发(真实运动伪影常很响)
    t = np.arange(len(x)) / SR
    y = x + (1.2 * _rms(x) * np.sin(2 * np.pi * rng.uniform(2, 8) * t)).astype(np.float32)
    for _ in range(int(rng.integers(3, 7))):
        s = int(rng.integers(0, max(1, len(x) - 2000))); L = min(int(rng.integers(800, 3000)), len(y) - s)
        y[s:s + L] += rng.standard_normal(L).astype(np.float32) * _rms(x) * 3.0
    return y.astype(np.float32)


def deg_clip(x, gain, rng):                              # 力太大 → 过载削波(先归一化峰值再过载)
    mx = float(np.max(np.abs(x))) + 1e-9
    return np.clip(x / mx * gain, -1.0, 1.0).astype(np.float32)


def deg_poor_contact(x, rng):                            # 没贴住: 低通(闷) + 衰减
    b, a = butter(4, 300 / (SR / 2), btype="low")
    return (filtfilt(b, a, x) * rng.uniform(0.2, 0.5)).astype(np.float32)


def deg_dropout(x, rng):                                 # 间歇掉接触: 段置零
    y = x.copy()
    for _ in range(int(rng.integers(2, 5))):
        s = int(rng.integers(0, max(1, len(x) - 4000))); L = min(int(rng.integers(1000, 4000)), len(y) - s)
        y[s:s + L] = 0.0
    return y.astype(np.float32)


def deg_heartbeat(x, rng):                               # 心音污染: 低频心跳 thump
    t = np.arange(len(x)) / SR; hr = rng.uniform(1.0, 1.6)
    beat = np.zeros(len(x), np.float32)
    for k in range(int(t[-1] * hr) + 1):
        c = int((k / hr) * SR)
        if c >= len(x):
            break
        w = min(int(0.08 * SR), len(x) - c)
        beat[c:c + w] += (np.hanning(w) * np.sin(2 * np.pi * 40 * t[c:c + w])).astype(np.float32)
    return (x + 1.5 * _rms(x) * beat).astype(np.float32)


# variant 列表: (mode, tier, fn)  tier: 1 mild / 2 moderate / 3 severe
def variants(x, rng):
    return [
        ("noise", 1, deg_noise(x, 15, rng)), ("noise", 2, deg_noise(x, 5, rng)), ("noise", 3, deg_noise(x, -3, rng)),
        ("motion", 2, deg_motion(x, rng)), ("motion", 3, deg_motion(deg_motion(x, rng), rng)),
        ("clip", 2, deg_clip(x, 4, rng)), ("clip", 3, deg_clip(x, 10, rng)),
        ("poor_contact", 2, deg_poor_contact(x, rng)),
        ("dropout", 2, deg_dropout(x, rng)),
        ("heartbeat", 2, deg_heartbeat(x, rng)),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_glob", required=True)
    ap.add_argument("--n_sources", type=int, default=150)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--calib_out", required=True)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # 1) 取相对干净的源(SNR 高、静音少 → 当 good 基线)
    cands = sorted(glob.glob(args.clean_glob, recursive=True))
    rng.shuffle(cands)
    rows = []                                            # (mode, tier, subscores dict, metrics dict)
    n_src = 0
    for path in cands:
        if n_src >= args.n_sources:
            break
        try:
            xn, srn = P.read_wav(path); x, sr = P.resample(xn, srn)
        except Exception:
            continue
        if len(x) < sr:
            continue
        x = (x / (np.max(np.abs(x)) + 1e-9) * 0.3).astype(np.float32)
        m0 = P.metrics_from_signal(x, sr)
        if m0["snr_db"] < 12 or m0["silence_ratio"] > 0.4:   # 源不够干净, 跳过
            continue
        n_src += 1
        rows.append(("clean", 0, m0["subscores"], m0))
        for mode, tier, xv in variants(x, rng):
            mv = P.metrics_from_signal(xv.astype(np.float32), sr)
            rows.append((mode, tier, mv["subscores"], mv))
    print(f"[calib] clean sources used = {n_src}, total clips = {len(rows)}", flush=True)
    if n_src < 20:
        print("[calib] too few clean sources — abort", flush=True); return

    keys = list(P._METRIC_KEY.keys())
    good = [r for r in rows if r[1] == 0]
    bad = [r for r in rows if r[1] >= 2]                 # moderate+severe 当 bad

    # 2) 重估 anchors: good_anchor=干净中位, bad_anchor=坏样中位(原始指标)。
    #    若 good≈bad(该指标分不开 → 退化锚点) → 保留默认锚, 不乱改。
    anchors = {}
    for k in keys:
        mk = P._METRIC_KEY[k]
        gv = float(np.median([r[3][mk] for r in good])); bv = float(np.median([r[3][mk] for r in bad]))
        if abs(gv - bv) < 0.02 * max(abs(gv), abs(bv), 1e-3):
            anchors[k] = list(P.DEFAULT_ANCHORS[k])
        else:
            anchors[k] = [round(gv, 4), round(bv, 4)]

    # 3) 权重 ∝ 各子分对 good/bad 的判别力(AUC-0.5)。用【derived anchors】重算子分, 保持一致。
    y = np.array([1] * len(good) + [0] * len(bad))
    sub_auc = {}
    for k in keys:
        s = np.array([P.subscores(r[3], anchors)[k] for r in good]
                     + [P.subscores(r[3], anchors)[k] for r in bad])
        sub_auc[k] = float(P.auc(s, y))                  # >0.5 = 该子分能分出好坏
    disc = {k: max(0.0, sub_auc[k] - 0.5) for k in keys}
    tot = sum(disc.values())
    weights = ({k: round(disc[k] / tot, 4) for k in keys} if tot > 1e-6
               else {k: round(1.0 / len(keys), 4) for k in keys})

    # 4) 用新 anchors+weights 重算 Q, 评 good/bad AUC + 每种失败模式 AUC
    def Q(metrics):
        return P.q_proxy(metrics, weights=weights, anchors=anchors)
    qg = np.array([Q(r[3]) for r in good]); qb = np.array([Q(r[3]) for r in bad])
    auc_overall = P.auc(np.concatenate([qg, qb]), y)
    per_mode = {}
    for mode in ["noise", "motion", "clip", "poor_contact", "dropout", "heartbeat"]:
        qm = np.array([Q(r[3]) for r in rows if r[0] == mode and r[1] >= 2])
        if len(qm):
            yy = np.array([1] * len(qg) + [0] * len(qm))
            per_mode[mode] = {"auc": round(float(P.auc(np.concatenate([qg, qm]), yy)), 3),
                              "q_mean_bad": round(float(qm.mean()), 3)}

    calib = {"anchors": anchors, "weights": weights,
             "meta": {"n_clean_sources": n_src, "n_good": len(good), "n_bad": len(bad),
                      "q_mean_good": round(float(qg.mean()), 3), "q_mean_bad": round(float(qb.mean()), 3),
                      "auc_Q_good_vs_bad": round(float(auc_overall), 3),
                      "subscore_auc": {k: round(sub_auc[k], 3) for k in keys},
                      "per_failure_mode": per_mode}}
    json.dump(calib, open(args.calib_out, "w"), indent=2, ensure_ascii=False)
    json.dump(calib, open(os.path.join(args.out_dir, "calibration.json"), "w"), indent=2, ensure_ascii=False)
    print("[calib] === CALIBRATION ===", flush=True)
    print(json.dumps(calib, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
