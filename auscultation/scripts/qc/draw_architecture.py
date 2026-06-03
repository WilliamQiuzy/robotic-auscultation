#!/usr/bin/env python3
"""论文级模型结构图: 感知(单头稠密IUV) → 动作策略 VLA(π0, 设计B接口) → 音频闭环。
与 docs/model_architecture.md 一致。输出 docs/figures/model_architecture.png。
排版: 左→右流水线, 大字号不溢出, 精简箭头。用法: python scripts/qc/draw_architecture.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "docs", "figures", "model_architecture.png")

# 配色(每模块一族色; 不用多余紫色)
C_IN = "#ECEFF3"; C_IN_E = "#8A93A0"          # 输入 灰
C_PER = "#DBE7FF"; C_PER_E = "#2F5FD0"        # 感知 蓝
C_IF = "#FFE7A3"; C_IF_E = "#C2901A"          # 设计B接口 金
C_VLA = "#D5F0DC"; C_VLA_E = "#2E8B57"        # 动作 VLA 绿
C_AUD = "#FBD9C4"; C_AUD_E = "#D9722B"        # 音频闭环 橙
C_ROB = "#DDE3EA"; C_ROB_E = "#5B6675"        # 机器人 中性灰蓝
WHITE = "#FFFFFF"


def box(ax, x, y, w, h, title, body="", fc=WHITE, ec="#333", tfs=12, bfs=10, tcol="#111",
        title_top=False, body_col="#1c1c1c"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=2.0",
                                linewidth=1.8, edgecolor=ec, facecolor=fc, zorder=2))
    top = title_top or bool(body)
    if title:
        ax.text(x + w / 2, (y + h - 2.0) if top else (y + h / 2), title, ha="center",
                va="top" if top else "center", fontsize=tfs, fontweight="bold",
                color=tcol, zorder=3, linespacing=1.2)
    if body:
        ax.text(x + w / 2, y + h / 2 - 1.6, body, ha="center", va="center", fontsize=bfs,
                color=body_col, zorder=3, linespacing=1.4)


def arrow(ax, p0, p1, color="#333", lw=2.0, cs="arc3,rad=0.0", ls="-", z=4):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=18, linewidth=lw,
                                 color=color, connectionstyle=cs, linestyle=ls, zorder=z))


def label(ax, x, y, t, fs=9.5, col="#333"):
    ax.text(x, y, t, ha="center", va="center", fontsize=fs, color=col, zorder=5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))


def main():
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    # ── 输入 ───────────────────────────────────────────────
    box(ax, 2, 20, 16, 68, "Inputs", fc=C_IN, ec=C_IN_E, tfs=13.5, title_top=True)
    inputs = [("Third-person\nRGB-D", 75), ("Wrist RGB", 65), ("Proprioception", 55),
              ("Force / Torque", 45), ("Audio", 35), ("Language", 25)]
    for name, yy in inputs:
        box(ax, 4, yy, 12, 8, name, fc=WHITE, ec=C_IN_E, tfs=10.5)

    # ── ① 感知 ─────────────────────────────────────────────
    box(ax, 23, 57, 42, 31, "①  Perception   (RGB → dense IUV)",
        fc=C_PER, ec=C_PER_E, tfs=14, tcol="#1a3a8a", title_top=True)
    box(ax, 25, 67, 12, 12, "Vision\nEncoder", fc=WHITE, ec=C_PER_E, tfs=11)
    box(ax, 39, 67, 11, 12, "Dense IUV\nHead", fc=WHITE, ec=C_PER_E, tfs=11)
    box(ax, 52, 67, 11, 12, "Dense IUV\n(I, U, V)", fc=WHITE, ec=C_PER_E, tfs=11)
    arrow(ax, (37, 73), (39, 73), C_PER_E); arrow(ax, (50, 73), (52, 73), C_PER_E)
    box(ax, 25, 59, 38, 6.4, "Localize task landmarks  (Heart → cardiac · Lung → pulmonary)",
        fc="#EEF4FF", ec=C_PER_E, tfs=10.5)
    arrow(ax, (57.5, 67), (53, 65.4), C_PER_E)             # IUV -> readout

    # ── 设计B 接口 ──────────────────────────────────────────
    box(ax, 23, 45, 42, 9.5, "Target Hand-off   (Perception → Action)",
        body='"where to listen":  target heatmap  +  3D target point (base frame)',
        fc=C_IF, ec=C_IF_E, tfs=12.5, bfs=10, tcol="#7a5500", body_col="#5a4500", title_top=True)
    arrow(ax, (44, 59), (44, 54.5), C_PER_E)               # readout -> interface

    # ── ② 动作策略 VLA ─────────────────────────────────────
    box(ax, 70, 40, 28, 48, "②  Arm-Control Policy  (VLA, π0)",
        fc=C_VLA, ec=C_VLA_E, tfs=13.5, tcol="#1d5e38", title_top=True)
    box(ax, 72.5, 70, 23, 11, "PaliGemma\nVision-Language Trunk", fc=WHITE, ec=C_VLA_E, tfs=11)
    box(ax, 72.5, 56, 23, 11, "Flow-Matching\nAction Expert", fc=WHITE, ec=C_VLA_E, tfs=11)
    box(ax, 72.5, 43, 23, 10, "Action sequence (next K steps)\n6-DoF Δpose / joint targets",
        fc="#EAF6EE", ec=C_VLA_E, tfs=10)
    arrow(ax, (84, 70), (84, 67), C_VLA_E); arrow(ax, (84, 56), (84, 53), C_VLA_E)

    # ── ③ 音频闭环 ──────────────────────────────────────────
    box(ax, 70, 10, 28, 19, "③  Audio Loop   (post-contact)",
        body="audio quality → search (I,U,V)\nneighborhood → refine target",
        fc=C_AUD, ec=C_AUD_E, tfs=12.5, bfs=10.5, tcol="#8a4416", body_col="#6a3410", title_top=True)

    # ── 机器人 ──────────────────────────────────────────────
    box(ax, 30, 11, 26, 13, "Robot arm\n+ stethoscope", fc=C_ROB, ec=C_ROB_E, tfs=12)

    # ── 连线(精简: 仅主干 + 关键闭环) ───────────────────────
    arrow(ax, (18, 79), (23, 74), C_PER_E)                       # RGB -> 感知
    label(ax, 21, 79, "RGB", col=C_PER_E)
    arrow(ax, (65, 50), (70, 58), C_IF_E, lw=2.4)                # 接口 -> VLA
    label(ax, 68.5, 55, "heatmap\n+ 3D point", col="#a07410", fs=9)
    arrow(ax, (18, 42), (70, 50), C_VLA_E, cs="arc3,rad=-0.06")  # 输入总线 -> VLA
    label(ax, 44, 37.5, "wrist RGB · proprioception · force/torque · audio · language",
          col=C_VLA_E, fs=9.5)
    arrow(ax, (72.5, 44), (56, 19), C_VLA_E, cs="arc3,rad=0.10")  # VLA -> 机器人
    label(ax, 66, 30, "actions", col=C_VLA_E)
    arrow(ax, (56, 15), (70, 16), C_AUD_E)                       # 机器人 -> 音频(接触)
    label(ax, 63, 13, "contact", col=C_AUD_E, fs=9)
    arrow(ax, (80, 29), (60, 45), C_AUD_E, ls=(0, (5, 3)), cs="arc3,rad=0.22")  # 音频 -> 精修(闭环)
    label(ax, 73, 38, "refine (I,U,V)", col=C_AUD_E, fs=9)

    # 标题 + 脚注
    ax.text(50, 96.5, "Robotic Auscultation — Perception + Action Architecture",
            ha="center", va="center", fontsize=18, fontweight="bold", color="#111")
    ax.text(50, 4.2,
            "Modules trained separately, coupled only through the target hand-off interface.   "
            "①  Perception: dense-IUV supervision on body-image data (HuMMan / Scoliosis / Synthetic).   "
            "②  VLA: behavior cloning on teleoperation demos.",
            ha="center", va="center", fontsize=9.5, color="#444", linespacing=1.4)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("saved", OUT)


if __name__ == "__main__":
    main()
