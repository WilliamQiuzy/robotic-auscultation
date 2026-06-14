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

from iuv_model import DenseIUVModel, iuv_loss, apply_lora, landmark_heatmap_loss  # noqa: E402
from dataset import (AuscIUVDataset, split_manifest, split_manifests, train_rows_with_fixed_val,  # noqa: E402
                     collate, subject_key, IMAGENET_MEAN, IMAGENET_STD)
from eval import evaluate as eval_landmarks, landmark_addresses  # noqa: E402
from tasks import N_LANDMARKS, LANDMARK_INDEX  # noqa: E402


def _is_synth(r):
    """合成样本判定: id 以 syn 开头(与目录名无关, 比 _root 子串稳)。"""
    return r.get("id", "").startswith("syn")


def _group(r):
    """采样分组: 合成 / 真实正面 / 真实后背(瓶颈)。"""
    if _is_synth(r):
        return "syn"
    return "realback" if r.get("region") == "back" else "realfront"


def parse_balance_targets(spec):
    """'syn:0.3,realfront:0.25,realback:0.45' → (("syn",0.3),...)。空=默认。"""
    if not spec:
        return (("syn", 0.4), ("realfront", 0.3), ("realback", 0.3))
    return tuple((k, float(v)) for k, v in (kv.split(":") for kv in spec.split(",")))


def make_balanced_sampler(rows, targets=(("syn", 0.4), ("realfront", 0.3), ("realback", 0.3))):
    """WeightedRandomSampler: 按域分组上采样(真实后背稀缺→提权), 每 batch 近目标配比。"""
    import collections
    from torch.utils.data import WeightedRandomSampler
    tgt = dict(targets)
    cnt = collections.Counter(_group(r) for r in rows)
    w = [tgt.get(_group(r), 0.0) / max(1, cnt[_group(r)]) for r in rows]
    return WeightedRandomSampler(w, num_samples=len(rows), replacement=True), dict(cnt)


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


_BACK_NAMES = ["R1", "L1", "R2", "L2", "R3", "L3", "R4", "L4", "R5", "L5"]


@torch.no_grad()
def landmark_head_pck(model, rows, res, hm_res, device, names=_BACK_NAMES, thr=0.10):
    """landmark 头读点(热图 argmax) vs GT points 的 PCK@thr(归一化 thr*res)。
    冻结 IUV 时 IUV-地址 PCK 是常数, 必须用本函数选 best。"""
    import cv2
    import numpy as np
    was = model.training
    model.eval()
    chans = [LANDMARK_INDEX[n] for n in names]
    hits = tot = 0
    for r in rows:
        root = r.get("_root", "")
        try:
            img = cv2.imread(os.path.join(root, r["image"]))[:, :, ::-1]
        except Exception:
            continue
        h0, w0 = img.shape[:2]
        im = cv2.resize(img, (res, res), interpolation=cv2.INTER_AREA)
        x = ((im.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None]
        x = torch.from_numpy(x.copy()).to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            hm = model.predict_landmark_heatmap(x, (hm_res, hm_res))[0].float()   # (K,h,w)
        flat = hm.reshape(hm.shape[0], -1)
        idx = flat.argmax(1)
        py = (idx // hm_res).float() * (res / hm_res)
        px = (idx % hm_res).float() * (res / hm_res)
        sx, sy = res / float(w0), res / float(h0)
        pts = r.get("points") or {}
        for nm, c in zip(names, chans):
            p = pts.get(nm)
            if not p or not p.get("visible"):
                continue
            gx, gy = p["xy"][0] * sx, p["xy"][1] * sy
            d = ((px[c].item() - gx) ** 2 + (py[c].item() - gy) ** 2) ** 0.5
            tot += 1
            hits += int(d < thr * res)
    if was:
        model.train()
    return hits / max(1, tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="单数据集目录(含 manifest.jsonl/img/iuv)")
    ap.add_argument("--roots", default=None, help="多数据集混训, 逗号分隔目录(优先于 --root)")
    ap.add_argument("--unfreeze", type=int, default=0, help="1=解冻编码器微调(差分学习率)")
    ap.add_argument("--enc_lr_mult", type=float, default=0.1, help="解冻时编码器 lr = lr*该值")
    ap.add_argument("--lora", type=int, default=0, help="1=LoRA(冻结基座+低秩适配, 替全解冻; 强正则+小ckpt)")
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--weight_decay", type=float, default=0.05, help="AdamW weight decay(抗过拟合, v1=1e-4偏低)")
    ap.add_argument("--balance", type=int, default=1, help="1=按域分组均衡上采样(合成/真实前/真实后)")
    ap.add_argument("--balance_targets", default="", help="域配比 'syn:0.3,realfront:0.25,realback:0.45'; 空=默认 0.4/0.3/0.3")
    ap.add_argument("--ema", type=float, default=0.0, help=">0 启用权重EMA(如0.999); 每epoch也用EMA权重评测真实后背选best")
    ap.add_argument("--select", default="realback_pck", choices=["realback_pck", "valloss"],
                    help="选 best ckpt 的依据(默认真实后背 PCK@10, 而非混合 val-loss)")
    ap.add_argument("--smpl", default="data/smpl/SMPL_NEUTRAL.pkl")
    ap.add_argument("--uv", default="data/densepose_coco/uv_data/UV_Processed.mat")
    ap.add_argument("--encoder", default="dinov2_vitb14")
    ap.add_argument("--res", type=int, default=518)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val_frac", type=float, default=0.08)
    ap.add_argument("--val_split_in", default=None,
                    help="固定 held-out val_split.json; train rows exclude all val subjects for fair A/B")
    ap.add_argument("--out", default="outputs/perc_v0")
    ap.add_argument("--frozen", type=int, default=None, help=argparse.SUPPRESS)  # deprecated; use --unfreeze
    # ── landmark 头(直接监督听诊点, 绕开有偏 V 场)──
    ap.add_argument("--landmark_head", type=int, default=0, help="1=加直接监督的 landmark 热图头")
    ap.add_argument("--lambda_lm", type=float, default=1.0, help="landmark 热图损失权重")
    ap.add_argument("--hm_res", type=int, default=0, help="landmark 热图分辨率(0=res//4)")
    ap.add_argument("--hm_sigma", type=float, default=2.5, help="高斯热图 sigma(热图像素)")
    ap.add_argument("--freeze_iuv", type=int, default=0,
                    help="1=冻结 encoder LoRA+IUV 头, 只训 landmark 头(配 --init_from 暖启动 → 后背IUV路径不退步)")
    ap.add_argument("--init_from", default=None, help="暖启动: 载入已有 lora ckpt(best.pt) 的 LoRA+IUV 头权重")
    ap.add_argument("--geom_aug", type=int, default=0,
                    help="1=训练在线部署取景增强(上背半身裁切+受试者小远缩放), 修 IUV 场在半身/小远取景下塌缩")
    ap.add_argument("--geom_crop_p", type=float, default=0.18, help="上背半身裁切概率")
    ap.add_argument("--geom_shrink_p", type=float, default=0.18, help="受试者小远缩放概率")
    args = ap.parse_args()
    hm_res = args.hm_res or (args.res // 4)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    frozen = not bool(args.unfreeze)
    # 审计记录: 配置 + 指标历史(供反思/复核, 不只存结果)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(audit_config(args, frozen_encoder=frozen), f, indent=2)
    metrics_path = os.path.join(args.out, "metrics.jsonl")
    open(metrics_path, "w").close()

    if args.val_split_in:
        assert args.roots, "--val_split_in 目前用于多数据集 A/B, 需指定 --roots"
        roots = [r.strip() for r in args.roots.split(",") if r.strip()]
        va_rows = json.load(open(args.val_split_in))
        tr_rows = train_rows_with_fixed_val(roots, va_rows)
        print(f"[data] 混训 {roots} | 固定 val_split={args.val_split_in}")
    elif args.roots:
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
    sampler = None
    if args.balance:
        sampler, gc = make_balanced_sampler(tr_rows, targets=parse_balance_targets(args.balance_targets))
        print(f"[data] 域分组均衡采样: {gc} → 目标 syn0.4/realfront0.3/realback0.3")
    use_lm = bool(args.landmark_head)
    tr = DataLoader(AuscIUVDataset(args.root, tr_rows, args.res,
                                   with_landmarks=use_lm, hm_res=hm_res, hm_sigma=args.hm_sigma,
                                   geom_aug=bool(args.geom_aug), geom_crop_p=args.geom_crop_p,
                                   geom_shrink_p=args.geom_shrink_p),
                    batch_size=args.bs, shuffle=(sampler is None), sampler=sampler,
                    num_workers=args.workers, collate_fn=collate, drop_last=True, pin_memory=True)
    va = DataLoader(AuscIUVDataset(args.root, va_rows, args.res), batch_size=args.bs, shuffle=False,
                    num_workers=args.workers, collate_fn=collate, pin_memory=True)
    # 真实后背 held-out(选 ckpt 用) + landmark 地址
    real_back = [r for r in va_rows if r.get("region") == "back" and not _is_synth(r)]
    sel_addr, _ = landmark_addresses(args.uv, args.smpl) if args.select == "realback_pck" else (None, None)
    print(f"[select] 依据={args.select} | 真实后背 held-out={len(real_back)}")

    n_lm = N_LANDMARKS if use_lm else 0
    model = DenseIUVModel(encoder=args.encoder, frozen_encoder=(frozen and not args.lora),
                          n_landmarks=n_lm).to(device)
    if args.lora:
        apply_lora(model, r=args.lora_rank, alpha=args.lora_alpha)   # 冻结基座, 只训 LoRA A/B + 头
        model.to(device)                                            # 新注入的 A/B 在 CPU, 重新搬到 GPU
        mode_str = f"LoRA(r{args.lora_rank}/a{args.lora_alpha})"
    elif args.unfreeze:
        mode_str = "full-unfreeze"
    else:
        mode_str = "frozen-head"
    # 暖启动: 载入已有 lora ckpt 的 LoRA+IUV 头权重(landmark 头保持新初始化)
    if args.init_from:
        ck = torch.load(args.init_from, map_location=device)
        sd = ck.get("lora_head") or ck.get("model") or ck.get("head") or ck
        miss, unexp = model.load_state_dict(sd, strict=False)
        lm_miss = [k for k in miss if not k.startswith("landmark_head")]
        print(f"[init] 暖启动 {args.init_from} | 载入 {len(sd)} 张量, 非landmark缺失 {len(lm_miss)}, 多余 {len(unexp)}")
    # 冻结 IUV 路径(encoder LoRA + IUV 头), 只训 landmark 头 → 后背 IUV 读出保持=init_from, 不退步
    if args.freeze_iuv:
        assert use_lm, "--freeze_iuv 需配 --landmark_head"
        for nme, p in model.named_parameters():
            p.requires_grad_(nme.startswith("landmark_head."))
        mode_str += " +frozen-IUV(仅训landmark头)"
    elif use_lm:
        mode_str += " +landmark头(联合训)"
    opt = torch.optim.AdamW([{"params": [p for p in model.parameters() if p.requires_grad],
                              "lr": args.lr, "name": "lora_head" if args.lora else "head"}],
                            weight_decay=args.weight_decay)
    if args.unfreeze and not use_lm:
        opt = torch.optim.AdamW([
            {"params": model.encoder.parameters(), "lr": args.lr * args.enc_lr_mult, "name": "encoder"},
            {"params": model.head.parameters(), "lr": args.lr, "name": "head"}], weight_decay=args.weight_decay)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {args.encoder} {mode_str} | 可训练参数 {n_train/1e6:.2f}M")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(tr))
    scaler = torch.cuda.amp.GradScaler()

    ema = None
    if args.ema > 0:                                    # 权重EMA: 平滑可训参数, 评测/存盘用EMA权重
        ema = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        print(f"[ema] 启用 decay={args.ema} (跟踪 {len(ema)} 个可训张量)", flush=True)

    def swap_in_ema():
        """把 EMA 权重换进模型(返回 live 备份以便还原)。"""
        live = {}
        for n, p in model.named_parameters():
            if n in ema:
                live[n] = p.detach().clone(); p.data.copy_(ema[n])
        return live

    def restore_live(live):
        for n, p in model.named_parameters():
            if n in live:
                p.data.copy_(live[n])

    best_score = -1e9; best_ep = -1
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); run = {}
        for it, b in enumerate(tr):
            img = b["image"].to(device, non_blocking=True)
            part = b["part"].to(device, non_blocking=True)
            uv = b["uv"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                if use_lm:
                    pl, puv, hm = model.forward_with_landmarks(img, (hm_res, hm_res))
                    lm_tgt = b["landmark_hm"].to(device, non_blocking=True)
                    lm_mask = b["landmark_mask"].to(device, non_blocking=True)
                    loss_lm, parts = landmark_heatmap_loss(hm, lm_tgt, lm_mask, args.lambda_lm)
                    if args.freeze_iuv:                       # IUV 路径冻结 → 只回传 landmark 损失
                        loss = loss_lm
                    else:                                     # 联合训: IUV + landmark
                        loss_iuv, ip = iuv_loss(pl, puv, part, uv)
                        loss = loss_iuv + loss_lm; parts.update(ip)
                else:
                    pl, puv = model(img)
                    loss, parts = iuv_loss(pl, puv, part, uv)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            if ema is not None:                         # EMA 更新(仅可训参数)
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        if n in ema:
                            ema[n].mul_(args.ema).add_(p.detach(), alpha=1.0 - args.ema)
            for k, v in {"loss": loss.item(), **parts}.items():
                run[k] = run.get(k, 0.0) + v
            if it % 50 == 0:
                pmsg = " ".join(f"{k} {v:.4f}" for k, v in parts.items())
                print(f"  ep{ep} it{it}/{len(tr)} loss {loss.item():.4f} {pmsg}", flush=True)
        live_bak = swap_in_ema() if ema is not None else None   # EMA: 用平滑权重评测+存盘
        vloss = run_val(model, va, device)
        # 真实后背 PCK@10(选 ckpt 依据)。landmark 头模型用【热图读出】PCK(冻结IUV时IUV地址PCK是常数);
        # 否则用 IUV-地址 soft 读出 PCK。两者都 vs V6A(=教科书)目标点。
        rb_pck = None; lm_pck = None
        if real_back:
            if use_lm:
                lm_pck = landmark_head_pck(model, real_back, args.res, hm_res, device)
            if not args.freeze_iuv and args.select == "realback_pck":
                rd = eval_landmarks(model, "", real_back, sel_addr, args.res, device, readout_mode="soft")
                rb_pck = rd.get("back", {}).get("pck@10%")
        if use_lm:                                            # landmark 模型: 用热图 PCK 选 best
            score = lm_pck if lm_pck is not None else -vloss
        else:
            score = rb_pck if (args.select == "realback_pck" and rb_pck is not None) else -vloss
        sec = time.time() - t0
        msg = " ".join(f"{k} {run[k]/len(tr):.4f}" for k in run)
        lrs = lr_record(opt)
        print(f"[ep{ep}] {msg} | val {vloss:.4f} | realback_PCK10 {rb_pck} | landmark_PCK10 {lm_pck} | {format_lr_record(lrs)} | {sec:.0f}s", flush=True)
        meta = {"args": vars(args), "epoch": ep, "val": vloss, "rb_pck": rb_pck, "lm_pck": lm_pck,
                "n_landmarks": n_lm, "hm_res": hm_res}
        if args.lora:                                       # 存 adapter(LoRA A/B)+头, 基座=原DINOv2可重建
            if use_lm:        # landmark 模型: 连同冻结的 LoRA+IUV头(暖启动来的)一并存, 否则推理缺权重
                def _keep(k):
                    return (".A." in k or ".B." in k or k.startswith("head.")
                            or k.startswith("landmark_head."))
            else:
                tk = {n for n, p in model.named_parameters() if p.requires_grad}
                def _keep(k):
                    return k in tk
            ckpt = {"kind": "lora", "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
                    "lora_head": {k: v.detach().cpu() for k, v in model.state_dict().items() if _keep(k)}, **meta}
        elif args.unfreeze:
            ckpt = {"model": model.state_dict(), **meta}
        else:
            ckpt = {"head": model.head.state_dict(), **meta}
        torch.save(ckpt, os.path.join(args.out, "last.pt"))
        if not args.unfreeze:                                        # 全模型 ckpt ~350MB, 解冻时只留 best/last
            torch.save(ckpt, os.path.join(args.out, f"ep{ep:02d}.pt"))
        with open(metrics_path, "a") as f:
            rec = {"epoch": ep, "val": vloss, "realback_pck10": rb_pck, "landmark_pck10": lm_pck, "sec": round(sec, 1)}
            rec.update(lrs)
            rec.update({k: round(run[k] / len(tr), 5) for k in run})
            f.write(json.dumps(rec) + "\n")
        if score > best_score:
            best_score = score; best_ep = ep; torch.save(ckpt, os.path.join(args.out, "best.pt"))
            print(f"  ↳ saved best (ep{ep}, {args.select}={score:.4f})", flush=True)
        if live_bak is not None:                         # 还原 live 权重继续训练
            restore_live(live_bak)
    print(f"[done] best={args.select} {best_score:.4f} @ ep{best_ep} → {args.out}")


if __name__ == "__main__":
    main()
