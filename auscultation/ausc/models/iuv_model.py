#!/usr/bin/env python3
"""稠密 IUV 感知模型(编码器可插拔)。

输出与 DensePose 同制式: 逐像素 part I(0=背景,1..24 身体块) + (U,V)∈[0,1]。
- 编码器: 预训练人体/通用 ViT(v0=DINOv2, 冻结; 将来可换 Sapiens)。
- 头: 轻量卷积解码器, 上采样到原分辨率, 输出 25 类 part logits + 2 通道 UV。

监督: part 用交叉熵(含背景), UV 用 masked Smooth-L1(仅前景)。这与 DensePose chart 头一致。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_PARTS = 25  # 0=背景 + 24 DensePose 身体块


# ── 编码器(可插拔)──────────────────────────────────────────────
class DinoV2Encoder(nn.Module):
    """DINOv2 ViT 主干(torch.hub 加载), 输出 patch 特征图 (B,C,h,w)。patch=14。"""
    def __init__(self, name: str = "dinov2_vitb14", frozen: bool = True):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", name)
        self.embed_dim = self.backbone.embed_dim
        self.patch = 14
        self.frozen = frozen
        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()        # 冻结主干始终 eval(关 dropout/不更新 BN)
        return self

    def forward(self, x):
        B, _, H, W = x.shape
        hh, ww = H // self.patch, W // self.patch
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            feat = self.backbone.forward_features(x)["x_norm_patchtokens"]  # (B, hh*ww, C)
        feat = feat.transpose(1, 2).reshape(B, self.embed_dim, hh, ww)
        return feat


def build_encoder(name: str, frozen: bool = True):
    if name.startswith("dinov2"):
        enc = DinoV2Encoder(name, frozen=frozen)
        return enc, enc.embed_dim, enc.patch
    raise ValueError(f"unknown encoder '{name}' (v0 支持 dinov2_*; Sapiens 待接)")


# ── LoRA(低秩适配, 替全解冻: 冻结基座, 只训低秩 A/B + 头, 强正则 + ckpt 小)──
class LoRALinear(nn.Module):
    """frozen base Linear + 低秩旁路 scale·B(A(x))。"""
    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Linear(base.in_features, r, bias=False)
        self.B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.B.weight)                       # 初始 LoRA=0 → 起点=原模型
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * self.B(self.A(x))


def apply_lora(model, r: int = 16, alpha: int = 32, targets=("qkv", "proj", "fc1", "fc2")):
    """把 DINOv2 注意力/MLP 的目标 Linear 换成 LoRALinear; 冻结基座, 仅 LoRA A/B + 头可训。"""
    enc = model.encoder.backbone
    enc.train()                                              # LoRA 需要 encoder forward 走 grad
    model.encoder.frozen = False
    for mod in enc.modules():
        for cn, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and cn in targets:
                setattr(mod, cn, LoRALinear(child, r, alpha))
    for p in model.encoder.parameters():                     # 先全冻
        p.requires_grad_(False)
    for m in model.encoder.modules():                        # 再只放开 LoRA A/B
        if isinstance(m, LoRALinear):
            for p in list(m.A.parameters()) + list(m.B.parameters()):
                p.requires_grad_(True)
    return model


# ── IUV 头 ──────────────────────────────────────────────────────
class IUVHead(nn.Module):
    """轻量卷积解码器: patch 特征 → (part_logits[25], uv[2]) 全分辨率。"""
    def __init__(self, in_dim: int, mid: int = 256, n_parts: int = N_PARTS):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, mid, 3, padding=1), nn.GroupNorm(32, mid), nn.GELU(),
            nn.Conv2d(mid, mid, 3, padding=1), nn.GroupNorm(32, mid), nn.GELU(),
        )
        self.part = nn.Conv2d(mid, n_parts, 1)
        self.uv = nn.Conv2d(mid, 2, 1)

    def forward(self, feat, out_hw):
        x = self.proj(feat)
        x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        part_logits = self.part(x)
        uv = torch.sigmoid(self.uv(x))      # U,V ∈ [0,1]
        return part_logits, uv


# ── 整体模型 ────────────────────────────────────────────────────
class DenseIUVModel(nn.Module):
    def __init__(self, encoder: str = "dinov2_vitb14", frozen_encoder: bool = True, mid: int = 256):
        super().__init__()
        self.encoder, feat_dim, self.patch = build_encoder(encoder, frozen=frozen_encoder)
        self.head = IUVHead(feat_dim, mid=mid)

    def forward(self, img):
        feat = self.encoder(img)
        return self.head(feat, out_hw=img.shape[-2:])   # part_logits (B,25,H,W), uv (B,2,H,W)


def iuv_loss(part_logits, uv, part_tgt, uv_tgt, lambda_uv: float = 1.0):
    """part 交叉熵(全像素) + UV masked Smooth-L1(仅前景 part>0)。"""
    ce = F.cross_entropy(part_logits, part_tgt)
    fg = (part_tgt > 0).unsqueeze(1).float()            # (B,1,H,W)
    denom = fg.sum().clamp(min=1.0)
    l1 = (F.smooth_l1_loss(uv, uv_tgt, reduction="none") * fg).sum() / (denom * 2)
    return ce + lambda_uv * l1, {"ce": ce.item(), "uv_l1": l1.item()}
