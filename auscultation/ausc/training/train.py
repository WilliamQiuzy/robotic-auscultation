#!/usr/bin/env python3
"""感知模型训练(v0: 冻结编码器 + 训 IUV 头)。
用法:
  python ausc/training/train.py --root data/dataset/synthetic --epochs 20 --bs 12 --out outputs/perc_v0
数据已含增强(生成器烘焙), 这里不再二次增强。
"""
import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "ausc"))
sys.path.insert(0, os.path.join(ROOT, "ausc", "models"))
sys.path.insert(0, os.path.join(ROOT, "ausc", "training"))

from iuv_model import DenseIUVModel, iuv_loss          # noqa: E402
from dataset import AuscIUVDataset, split_manifest, split_manifests, collate, subject_key  # noqa: E402


def audit_config(args, frozen_encoder: bool):
    """Return the persisted training config without deprecated/conflicting flags."""
    cfg = vars(args).copy()
    cfg.pop("frozen", None)
    cfg["frozen_encoder"] = bool(frozen_encoder)
    cfg["trainable_encoder"] = not bool(frozen_encoder)
    return cfg


def lr_record(opt):
    """Name optimizer learning rates by param group for readable metrics history."""
    out = {}
    for idx, group in enumerate(opt.param_groups):
        name = group.get("name") or f"group{idx}"
        out[f"lr_{name}"] = group["lr"]
    return out


def format_lr_record(record):
    return " ".join(f"{k} {v:.2e}" for k, v in record.items())


def run_val(model, loader, device):
    model.eval()
    tot = n = 0.0
    with torch.no_grad():
        for b in loader:
            img = b["image"].to(device); part = b["part"].to(device); uv = b["uv"].to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                pl, puv = model(img)
                loss, _ = iuv_loss(pl, puv, part, uv)
            tot += loss.item() * img.size(0); n += img.size(0)
    model.train()
    return tot / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="单数据集目录(含 manifest.jsonl/img/iuv)")
    ap.add_argument("--roots", default=None, help="多数据集混训, 逗号分隔目录(优先于 --root)")
    ap.add_argument("--unfreeze", type=int, default=0, help="1=解冻编码器微调(差分学习率)")
    ap.add_argument("--enc_lr_mult", type=float, default=0.1, help="解冻时编码器 lr = lr*该值")
    ap.add_argument("--encoder", default="dinov2_vitb14")
    ap.add_argument("--res", type=int, default=518)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val_frac", type=float, default=0.08)
    ap.add_argument("--out", default="outputs/perc_v0")
    ap.add_argument("--frozen", type=int, default=None, help=argparse.SUPPRESS)  # deprecated; use --unfreeze
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    frozen = not bool(args.unfreeze)
    # 审计记录: 配置 + 指标历史(供反思/复核, 不只存结果)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(audit_config(args, frozen_encoder=frozen), f, indent=2)
    metrics_path = os.path.join(args.out, "metrics.jsonl")
    open(metrics_path, "w").close()

    if args.roots:
        roots = [r.strip() for r in args.roots.split(",") if r.strip()]
        tr_rows, va_rows = split_manifests(roots, val_frac=args.val_frac)
        print(f"[data] 混训 {roots}")
    else:
        assert args.root, "需指定 --root 或 --roots"
        tr_rows, va_rows = split_manifest(args.root, val_frac=args.val_frac)
    # 校验无受试者泄漏 + 持久化 val 划分(供 eval 精确复现, 防"同人不同视角"泄漏)
    tr_subs = {subject_key(r["id"]) for r in tr_rows}
    va_subs = {subject_key(r["id"]) for r in va_rows}
    leak = tr_subs & va_subs
    assert not leak, f"train/val 受试者泄漏(必须为空): {list(leak)[:8]}"
    json.dump(va_rows, open(os.path.join(args.out, "val_split.json"), "w"))
    print(f"[data] train={len(tr_rows)} val={len(va_rows)} | 受试者 train {len(tr_subs)}/val {len(va_subs)} 重叠 {len(leak)} (真subject切分, 已存val_split.json)")
    tr = DataLoader(AuscIUVDataset(args.root, tr_rows, args.res), batch_size=args.bs, shuffle=True,
                    num_workers=args.workers, collate_fn=collate, drop_last=True, pin_memory=True)
    va = DataLoader(AuscIUVDataset(args.root, va_rows, args.res), batch_size=args.bs, shuffle=False,
                    num_workers=args.workers, collate_fn=collate, pin_memory=True)

    model = DenseIUVModel(encoder=args.encoder, frozen_encoder=frozen).to(device)
    if args.unfreeze:
        opt = torch.optim.AdamW([
            {"params": model.encoder.parameters(), "lr": args.lr * args.enc_lr_mult, "name": "encoder"},
            {"params": model.head.parameters(), "lr": args.lr, "name": "head"}], weight_decay=1e-4)
    else:
        opt = torch.optim.AdamW([{"params": [p for p in model.parameters() if p.requires_grad],
                                  "lr": args.lr, "name": "head"}], weight_decay=1e-4)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {args.encoder} unfreeze={bool(args.unfreeze)} | 可训练参数 {n_train/1e6:.1f}M")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(tr))
    scaler = torch.cuda.amp.GradScaler()

    best = 1e9
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); run = {}
        for it, b in enumerate(tr):
            img = b["image"].to(device, non_blocking=True)
            part = b["part"].to(device, non_blocking=True)
            uv = b["uv"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                pl, puv = model(img)
                loss, parts = iuv_loss(pl, puv, part, uv)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            for k, v in {"loss": loss.item(), **parts}.items():
                run[k] = run.get(k, 0.0) + v
            if it % 50 == 0:
                print(f"  ep{ep} it{it}/{len(tr)} loss {loss.item():.4f} ce {parts['ce']:.4f} uv {parts['uv_l1']:.4f}", flush=True)
        vloss = run_val(model, va, device)
        sec = time.time() - t0
        msg = " ".join(f"{k} {run[k]/len(tr):.4f}" for k in run)
        lrs = lr_record(opt)
        print(f"[ep{ep}] {msg} | val {vloss:.4f} | {format_lr_record(lrs)} | {sec:.0f}s", flush=True)
        # 解冻微调: 必须存【整模型】(编码器也变了); 冻结: 只存 head 即可
        if args.unfreeze:
            ckpt = {"model": model.state_dict(), "args": vars(args), "epoch": ep, "val": vloss}
        else:
            ckpt = {"head": model.head.state_dict(), "args": vars(args), "epoch": ep, "val": vloss}
        torch.save(ckpt, os.path.join(args.out, "last.pt"))
        if not args.unfreeze:                                        # 全模型 ckpt ~350MB, 解冻时只留 best/last
            torch.save(ckpt, os.path.join(args.out, f"ep{ep:02d}.pt"))   # 冻结: 每 epoch 独立留存(审计)
        with open(metrics_path, "a") as f:                            # 指标历史(供反思)
            rec = {"epoch": ep, "val": vloss, "sec": round(sec, 1)}
            rec.update(lrs)
            rec.update({k: round(run[k] / len(tr), 5) for k in run})
            f.write(json.dumps(rec) + "\n")
        if vloss < best:
            best = vloss; torch.save(ckpt, os.path.join(args.out, "best.pt"))
            print(f"  ↳ saved best (val {best:.4f})", flush=True)
    print(f"[done] best val {best:.4f} → {args.out}")


if __name__ == "__main__":
    main()
