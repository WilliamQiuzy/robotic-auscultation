#!/usr/bin/env python3
"""S2 — 冻结音频编码器(可插拔)。把肺音 → 固定维 embedding, 喂给轻量质量头。
对标感知的 DINOv2(冻结)+ LoRA: 编码器不训, 只训上面的小头。

- LogMelStatsEncoder: 自带、可立即跑的基线(log-mel 统计 + 校准子分), 无需外部权重。
- OperaEncoder: 呼吸专用 foundation model(剑桥, CC-BY-NC, 研究可用)的 adapter 占位,
  需先 clone github.com/evelyn0414/OPERA + 载入 CT 权重(见 docstring)。

纯 numpy/scipy(基线);OPERA 路径需 torch + OPERA 仓库。
"""
import os, sys
import numpy as np
from scipy import signal as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import proxies as P  # noqa: E402


def _hz2mel(h):
    return 2595.0 * np.log10(1.0 + h / 700.0)


def _mel2hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def log_mel(x, sr=16000, n_mels=40, n_fft=512, hop=160, fmin=50.0, fmax=2000.0):
    """log-mel 谱 (n_mels, T)。肺音带 50–2000Hz。"""
    f, _, S = sps.spectrogram(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, mode="magnitude")
    mel_pts = _mel2hz(np.linspace(_hz2mel(fmin), _hz2mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, len(f)), np.float32)
    for i in range(n_mels):
        lo, ce, hi = mel_pts[i], mel_pts[i + 1], mel_pts[i + 2]
        fb[i] = np.clip(np.minimum((f - lo) / (ce - lo + 1e-9), (hi - f) / (hi - ce + 1e-9)), 0, None)
    return np.log((fb @ (S ** 2)) + 1e-6)


class LogMelStatsEncoder:
    """基线 frozen encoder: log-mel 时间统计(均值+标准差) + 10 个校准质量子分。
    维度 = n_mels*2 + 10。可立即跑, 作为 OPERA 就位前的 bootstrap。"""
    name = "logmel_stats"

    def __init__(self, n_mels=40):
        self.n_mels = n_mels
        self.dim = n_mels * 2 + len(P._METRIC_KEY)

    def embed_signal(self, x, sr=16000):
        lm = log_mel(x, sr, self.n_mels)
        stat = np.concatenate([lm.mean(1), lm.std(1)])
        m = P.metrics_from_signal(x, sr)
        sub = np.array([m["subscores"][k] for k in P._METRIC_KEY], np.float32)
        return np.concatenate([stat, sub]).astype(np.float32)

    def embed(self, wav):
        xn, srn = P.read_wav(wav)
        x, sr = P.resample(xn, srn)
        return self.embed_signal(x, sr)


class OperaEncoder:
    """OPERA(OPen Respiratory Acoustic foundation model)frozen embedder —— adapter 占位。
    SETUP(pod):
      git clone https://github.com/evelyn0414/OPERA /workspace/OPERA
      下权重: huggingface-cli download evelyn0414/OPERA --local-dir /workspace/OPERA/ckpt  (OPERA-CT)
      pip install 其依赖。然后在下面 _load/_encode 接入 OPERA-CT(16kHz, 8s, 64-mel → 768-d)。
    license: CC-BY-NC-4.0(研究可用, 发表署名+非商用)。"""
    name = "opera"
    dim = 768

    def __init__(self, ckpt=None):
        raise NotImplementedError(
            "OPERA adapter 未接: 先 clone evelyn0414/OPERA + 下 CT 权重, 再实现 embed()。"
            "在此之前用 build_encoder('logmel_stats') 跑通 S2 管线。")


def build_encoder(name="logmel_stats", **kw):
    if name == "logmel_stats":
        return LogMelStatsEncoder(**kw)
    if name == "opera":
        return OperaEncoder(**kw)
    raise ValueError(f"unknown encoder '{name}'")
