#!/usr/bin/env python3
"""S2 — 轻量质量头: embedding → Q∈[0,1] + 不确定性(MC-dropout)。
对标感知的轻量读出头: 编码器冻结, 只训这个小 MLP。"""
import torch
import torch.nn as nn


class QualityHead(nn.Module):
    def __init__(self, in_dim, hidden=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)              # logit

    @torch.no_grad()
    def predict(self, x, mc=0):
        """mc=0: 单次 Q=sigmoid(logit)。mc>0: 开 dropout 采样 mc 次 → (Q 均值, 不确定性 std)。"""
        if mc <= 0:
            self.eval()
            return torch.sigmoid(self.forward(x)), torch.zeros(x.shape[0], device=x.device)
        self.train()                                # 保持 dropout 开
        ps = torch.stack([torch.sigmoid(self.forward(x)) for _ in range(mc)])
        return ps.mean(0), ps.std(0)
