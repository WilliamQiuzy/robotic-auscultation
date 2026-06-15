#!/usr/bin/env python3
"""听诊声学质量模块 — 自监督质量代理 (S1 弱标 teacher)。

给一段【肺音】波形, 不用人工标, 算一组声学信号质量指标 + 一个综合质量分 Q_proxy∈[0,1]:
判这段录音是否够干净(够诊断)。指标分四组(对应路线图 S1 的 1/2/3/4):
  ① 呼吸周期/气流可检测性  —— respiratory_periodicity (包络自相关找呼吸节律)
  ② 基础噪声统计           —— snr_db / clipping_ratio / silence_ratio / energy_kurtosis(突发)
  ③ Grooby 式 SQI(肺)     —— band_power_ratio(肺带功率比) + 周期性 + SNR
  ④ 综合 Q_proxy           —— 上面子分加权(在 SPRSound 'Poor Quality' 上校准, 见 run_s1_weak_labels.py)

纯 numpy/scipy, 不依赖 librosa/soundfile(自带 wave 读取, 支持 16/24/32-bit PCM, 统一重采样 16k mono)。
注: Q_proxy 是**启发式弱标**(自监督), 权重见 DEFAULT_WEIGHTS; 真值校准用 SPRSound 'Poor Quality'。
"""
import wave
import numpy as np
from scipy import signal as sps
from scipy.stats import kurtosis, rankdata

TARGET_SR = 16000
LUNG_BAND = (100.0, 2000.0)        # 肺音主要能量带 (Hz; 含 crackle/wheeze 高频)
BREATH_LAG_S = (1.5, 6.0)          # 呼吸周期范围 (10–40 次/分)


# ── 音频读取(无外部依赖)──────────────────────────────────────────────────
def read_wav(path):
    """读 WAV → (float32 [-1,1] mono, sr)。支持 8/16/24/32-bit PCM。"""
    with wave.open(path, "rb") as w:
        sr, nch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if sw == 2:
        a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 1:
        a = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 3:                                       # 24-bit: 3 字节小端有符号
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        a = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        a = np.where(a & 0x800000, a - 0x1000000, a).astype(np.float32) / 8388608.0
    elif sw == 4:
        a = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width {sw} bytes")
    if nch > 1:
        a = a.reshape(-1, nch).mean(axis=1)
    return a.astype(np.float32), sr


def resample(x, sr, target_sr=TARGET_SR):
    if sr == target_sr or len(x) < 2:
        return x, sr
    g = np.gcd(int(sr), int(target_sr))
    y = sps.resample_poly(x, target_sr // g, sr // g).astype(np.float32)
    return y, target_sr


# ── 帧能量 ──────────────────────────────────────────────────────────────────
def _frame_rms(x, sr, win=0.05, hop=0.025):
    w = max(1, int(win * sr)); h = max(1, int(hop * sr))
    if len(x) < w:
        return np.array([np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12)])
    fr = np.lib.stride_tricks.sliding_window_view(x, w)[::h]
    return np.sqrt(np.mean(fr.astype(np.float64) ** 2, axis=1) + 1e-12)


def _c01(v):
    return float(min(1.0, max(0.0, v)))


# ── ② 基础噪声统计 ───────────────────────────────────────────────────────────
def clipping_ratio(x_native):
    """削波: |x| 接近满量程的样本占比(在原始采样率上算, 重采样会改峰值)。"""
    return float(np.mean(np.abs(x_native) > 0.99))


def silence_ratio(x, sr):
    """静音: 帧 RMS < 10% 中位数的帧占比(过多静音=没贴好/没信号)。"""
    r = _frame_rms(x, sr)
    thr = 0.1 * np.median(r) + 1e-9
    return float(np.mean(r < thr))


def snr_db(x, sr):
    """估 SNR: 活跃帧的 高/低百分位帧能量之比(无参考)。信号=90 分位, 噪声底=10 分位。
    只在活跃帧(>5% 峰值)上算噪声底 —— 否则长静音(如没贴好/无接触)把噪声底压到~0, SNR 虚高。"""
    r = _frame_rms(x, sr)
    if len(r) < 3:
        return 0.0
    active = r[r > 0.05 * np.max(r)]
    if len(active) < 3:                                  # 几乎全静音 → 视作无信号
        return 0.0
    noise = np.percentile(active, 10); sigp = np.percentile(active, 90)
    return float(20.0 * np.log10((sigp + 1e-9) / (noise + 1e-9)))


def energy_kurtosis(x, sr):
    """帧能量峰度: 高=突发噪声/敲击(spiky), 稳态信号≈0。"""
    r = _frame_rms(x, sr)
    return float(kurtosis(r, fisher=True, nan_policy="omit")) if len(r) > 3 else 0.0


# ── ③ Grooby 式 SQI: 肺带功率比 ─────────────────────────────────────────────
def band_power_ratio(x, sr, band=LUNG_BAND):
    """肺音主带 (100–1000Hz) 功率占总功率比。低=噪声/带外主导。"""
    f, pxx = sps.welch(x, fs=sr, nperseg=min(2048, len(x)))
    total = np.sum(pxx) + 1e-12
    in_band = np.sum(pxx[(f >= band[0]) & (f <= band[1])])
    return float(in_band / total)


def spectral_feats(x, sr, flat_band=(50.0, 2000.0)):
    """谱质心 / 谱平坦度(Wiener 熵, 高=噪声样) / 85% 滚降。
    平坦度只在 50–2000Hz 信号带上算(全带含大量近零高频 bin → 几何均值恒≈0, 失效)。"""
    f, pxx = sps.welch(x, fs=sr, nperseg=min(2048, len(x)))
    pxx = pxx + 1e-12
    centroid = float(np.sum(f * pxx) / np.sum(pxx))
    pb = pxx[(f >= flat_band[0]) & (f <= flat_band[1])]
    flatness = float(np.exp(np.mean(np.log(pb))) / np.mean(pb)) if pb.size else 0.0
    cum = np.cumsum(pxx); rolloff = float(f[np.searchsorted(cum, 0.85 * cum[-1])])
    return centroid, flatness, rolloff


# ── ① 呼吸周期 / 气流可检测性 ────────────────────────────────────────────────
def respiratory_periodicity(x, sr):
    """包络自相关找呼吸节律 → (周期性强度∈[0,1], 估计呼吸周期 s)。
    肺音随呼吸准周期起伏(~0.2–0.5Hz); 包络在呼吸周期 lag 上的自相关峰高 = 气流/呼吸可检测性。"""
    if len(x) < sr:                                      # < 1s 无法判呼吸
        return 0.0, 0.0
    env = np.abs(sps.hilbert(x))
    env_sr = 50
    env = sps.resample_poly(env, env_sr, sr)
    env = env - np.mean(env)
    if np.std(env) < 1e-6:
        return 0.0, 0.0
    ac = np.correlate(env, env, mode="full")[len(env) - 1:]
    ac = ac / (ac[0] + 1e-12)
    lo = int(BREATH_LAG_S[0] * env_sr); hi = int(min(BREATH_LAG_S[1] * env_sr, len(ac) - 1))
    if hi <= lo:
        return 0.0, 0.0
    k = lo + int(np.argmax(ac[lo:hi]))
    return float(max(0.0, ac[k])), float(k / env_sr)


# ── ④ 综合 Q_proxy ──────────────────────────────────────────────────────────
DEFAULT_WEIGHTS = {                                      # 子分加权(在 SPRSound PQ 上校准)
    "snr": 0.25, "periodicity": 0.20, "band": 0.20,
    "flatness": 0.15, "silence": 0.10, "clip": 0.05, "kurtosis": 0.05,
}


def subscores(m):
    """把原始指标映射成 [0,1] 子分(高=好), 用于综合 Q_proxy。"""
    return {
        "snr": _c01((m["snr_db"] - 6.0) / 18.0),                 # 6dB→0, 24dB→1
        "periodicity": _c01(m["periodicity"] / 0.40),            # ≥0.4→1
        "band": _c01((m["band_power_ratio"] - 0.15) / 0.50),     # 带功率比
        "flatness": 1.0 - _c01((m["spectral_flatness"] - 0.15) / 0.45),  # 越平=越噪→低分
        "silence": 1.0 - _c01((m["silence_ratio"] - 0.40) / 0.40),       # >40% 静音扣分
        "clip": 1.0 - _c01(m["clipping_ratio"] * 20.0),          # 5% 削波→0
        "kurtosis": 1.0 - _c01((m["energy_kurtosis"] - 4.0) / 16.0),     # 突发→低分
    }


def q_proxy(m, weights=DEFAULT_WEIGHTS):
    s = subscores(m)
    return float(sum(weights[k] * s[k] for k in weights))


# ── 顶层: 一条 wav → 全部指标 + Q_proxy ──────────────────────────────────────
def analyze(path, target_sr=TARGET_SR):
    """读 wav → dict(原始指标 + 子分 + q_proxy + 元信息)。失败返回 {'error':...}。"""
    try:
        x_nat, sr_nat = read_wav(path)
    except Exception as e:
        return {"error": f"read: {e}"}
    if len(x_nat) < 8:
        return {"error": "too short"}
    clip = clipping_ratio(x_nat)                          # 削波在原始 sr 上算
    x, sr = resample(x_nat, sr_nat, target_sr)
    centroid, flatness, rolloff = spectral_feats(x, sr)
    per, breath_T = respiratory_periodicity(x, sr)
    m = {
        "snr_db": snr_db(x, sr),
        "clipping_ratio": clip,
        "silence_ratio": silence_ratio(x, sr),
        "energy_kurtosis": energy_kurtosis(x, sr),
        "band_power_ratio": band_power_ratio(x, sr),
        "spectral_centroid": centroid,
        "spectral_flatness": flatness,
        "spectral_rolloff": rolloff,
        "periodicity": per,
        "breath_period_s": breath_T,
    }
    m["q_proxy"] = q_proxy(m)
    m["subscores"] = subscores(m)
    m["fs_native"] = int(sr_nat)
    m["dur_s"] = round(len(x_nat) / sr_nat, 3)
    return m


# ── 校准辅助: 无 sklearn 的 AUC(Mann–Whitney)──────────────────────────────
def auc(score_pos_is_high, labels):
    """AUC: score 对 label(1=正类) 的排序判别力。score 应在正类上偏高。"""
    s = np.asarray(score_pos_is_high, float); y = np.asarray(labels, int)
    npos = int((y == 1).sum()); nneg = int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))
