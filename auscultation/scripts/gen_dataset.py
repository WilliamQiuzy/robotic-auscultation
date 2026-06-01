#!/usr/bin/env python3
"""统一数据集生成 CLI —— 三数据集批量化入口。

用法:
  python scripts/gen_dataset.py --dataset humman    --out data/dataset/humman    [--limit 50] [--n_aug 6]
  python scripts/gen_dataset.py --dataset scoliosis --out data/dataset/scoliosis [--n_aug 14]
  python scripts/gen_dataset.py --dataset synthetic --out data/dataset/synthetic [--n_subjects 200]
  python scripts/gen_dataset.py --dataset all       --out data/dataset           # 三个都跑(各自子目录)

每数据集导出: <out>/{img,iuv}/ + manifest.jsonl(schema 见 docs/datasets_and_processing.md §二.8)。
增强 = 5 种核心(光照/手臂遮挡/水平翻转[仅后背]/小角度对称旋转/尺度裁剪)随机组合, 每原图 ~n_aug 张。
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # auscultation/
sys.path.insert(0, os.path.join(ROOT, "ausc"))
sys.path.insert(0, os.path.join(ROOT, "ausc", "datasets"))


def _p(*parts):
    return os.path.join(ROOT, *parts)


def main():
    ap = argparse.ArgumentParser(description="生成感知训练数据集(HuMMan / Scoliosis / Synthetic)")
    ap.add_argument("--dataset", required=True, choices=["humman", "scoliosis", "synthetic", "all"])
    ap.add_argument("--out", required=True, help="输出根目录")
    ap.add_argument("--n_aug", type=int, default=None, help="每原图随机组合增强张数(默认按数据集)")
    ap.add_argument("--limit", type=int, default=None, help="[humman] 仅处理前 N 个序列(冒烟测试)")
    ap.add_argument("--n_subjects", type=int, default=200, help="[synthetic] 合成受试者数")
    ap.add_argument("--smpl_model", default=_p("data", "smpl", "SMPL_NEUTRAL.pkl"))
    ap.add_argument("--uv", default=_p("data", "densepose_coco", "uv_data", "UV_Processed.mat"))
    ap.add_argument("--humman_root", default=_p("data", "humman"))
    ap.add_argument("--scoliosis_labels", default=_p("data", "scoliosis_back", "labels_hmr2.pkl"))
    ap.add_argument("--scoliosis_imgs", default=_p("data", "scoliosis_back"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bedlam_skin", default=None, help="[synthetic] BEDLAM真皮 SMPL albedo 目录(给定则用真实纹理)")
    ap.add_argument("--smpl_uv_obj", default=_p("data", "smpl_assets", "smpl_uv_full.obj"), help="[synthetic] SMPL UV obj")
    args = ap.parse_args()

    ds = args.dataset

    def run_humman(out):
        import humman
        humman.generate(args.humman_root, args.smpl_model, args.uv, out,
                        limit=args.limit, n_aug=args.n_aug or 6, seed=args.seed)

    def run_scoliosis(out):
        import scoliosis
        scoliosis.generate(args.scoliosis_labels, args.scoliosis_imgs, args.smpl_model, args.uv, out,
                           n_aug=args.n_aug or 12, seed=args.seed)

    def run_synthetic(out):
        import synthetic
        synthetic.generate(args.smpl_model, args.uv, out,
                           n_subjects=args.n_subjects, n_aug=args.n_aug or 6, seed=args.seed,
                           texture_dir=args.bedlam_skin, smpl_uv_obj=args.smpl_uv_obj)

    if ds == "all":
        run_humman(os.path.join(args.out, "humman"))
        run_scoliosis(os.path.join(args.out, "scoliosis"))
        run_synthetic(os.path.join(args.out, "synthetic"))
    else:
        {"humman": run_humman, "scoliosis": run_scoliosis, "synthetic": run_synthetic}[ds](args.out)


if __name__ == "__main__":
    main()
