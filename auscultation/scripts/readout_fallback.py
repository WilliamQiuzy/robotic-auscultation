#!/usr/bin/env python3
"""听诊点读出 + 部署兜底机制。soft-argmax 读出基础上加三层安全网, 触发阈值由
86 张真实后背的 IUV 场实测标定(measure_collapse), 健康样本与失败样本完全不重叠:

  (A) 体外夹回: 点落在后背前景(part==1, 腐蚀留边)外 → 移到最近的后背内像素。
      [当前 soft-argmax 数学上必落体内, 0/86 触发, 纯安全网防将来模型/读出退化]
  (B) V 场侧向塌缩重展(85 型): 后背 V(侧向体表坐标)在 mask 内的稳健半幅 V_hw=(p90-p10)/2
      过小(<V_TRIG; 85=0.074 vs 健康 0.27-0.31) = 受试者小/远致左右坐标分辨不出、R/L 点全挤向
      脊柱 → 以观测中线为轴把 (V-中线) 线性放大到目标半幅, 再读出 → R/L 点重新展到两侧。
  (C) 下背不在画面(29/68 半身型): 后背 mask 的 U(纵向)上界 U_p90 过小(<U_TRIG;
      半身=0.42 vs 完整背 0.85-0.95) = 只拍到上背 → U 地址 > U_p90 的点(下肺野)真实位置不可见,
      不臆造, 标记 lowfov(读出仍给可见区底缘, 但提示不可信)。

返回 {name:(x,y,status)} status∈{ok,clamped,respread,lowfov}, flags(诊断), meta(度量)。
不修改入参(pred_uv 复制后重展, 保持不可变)。
"""
import numpy as np, cv2

V_TRIG = 0.15      # V_hw 低于此 = 侧向塌缩 (85=0.074; 健康≥0.265)
V_TARGET = 0.30    # 重展目标半幅 (≈健康样本 V_hw)
V_MAXSCALE = 4.5   # 放大上限, 防极端
U_TRIG = 0.55      # U_p90 低于此 = 下背不在画面 (半身=0.42; 完整背≥0.61)


def _nearest_in_mask(x, y, mask):
    yi, xi = int(round(y)), int(round(x))
    H, W = mask.shape
    if 0 <= yi < H and 0 <= xi < W and mask[yi, xi]:
        return x, y, False
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return x, y, False
    j = int(np.argmin((xs - x) ** 2 + (ys - y) ** 2))
    return float(xs[j]), float(ys[j]), True


def _respread_v(pred_uv, back_m, v_p50, v_hw):
    """以观测中线 v_p50 为轴, 把 mask 内的 V 线性放大到 V_TARGET 半幅(其余像素不动)。"""
    scale = min(V_TARGET / max(v_hw, 1e-3), V_MAXSCALE)
    out = pred_uv.copy()
    v = out[1]
    v_new = np.clip(0.5 + (v - v_p50) * scale, 0.0, 1.0)   # 中线→0.5, 离轴按比例外推
    out[1] = np.where(back_m, v_new, v)
    return out, scale


def _soft_argmax(part_prob, pred_uv, addr, gx, gy, tau):
    part, u, v = addr
    d2 = (pred_uv[0] - u) ** 2 + (pred_uv[1] - v) ** 2
    A = part_prob[int(part)] * np.exp(-d2 / tau)
    s = float(A.sum())
    if s < 1e-8:
        return None, 0.0
    return (float((A * gx).sum() / s), float((A * gy).sum() / s)), s


def readout_with_fallback(part_prob, pred_uv, addr_dict, names, gx, gy, tau=0.01,
                          back_part=1, erode=9):
    """part_prob (P,H,W) softmax; pred_uv (2,H,W); addr_dict {name:(part,U,V)}。"""
    back = (part_prob.argmax(0) == back_part)
    back_m = cv2.erode(back.astype(np.uint8), np.ones((erode, erode), np.uint8)).astype(bool)
    if back_m.sum() < 30:
        back_m = back                                   # 腐蚀过头则用原 mask

    flags, meta = [], {}
    if back_m.sum() < 30:                               # 几乎无后背前景, 直接放弃兜底度量
        meta = {"backfrac": float(back.mean())}
        uv_use = pred_uv
        u_p90 = 1.0
    else:
        Vm = pred_uv[1][back_m]; Um = pred_uv[0][back_m]
        v_p10, v_p50, v_p90 = np.percentile(Vm, [10, 50, 90])
        u_p90 = float(np.percentile(Um, 90))
        v_hw = float((v_p90 - v_p10) / 2)
        meta = {"backfrac": float(back.mean()), "v_hw": v_hw, "u_p90": u_p90}
        # (B) V 塌缩重展
        if v_hw < V_TRIG:
            uv_use, scale = _respread_v(pred_uv, back_m, float(v_p50), v_hw)
            flags.append("v_respread(V_hw%.3f→×%.1f侧向重展)" % (v_hw, scale))
        else:
            uv_use = pred_uv
        # (C) 下背不在画面
        if u_p90 < U_TRIG:
            flags.append("lowfov(下背不在画面 U_p90=%.2f, 下肺点不可信)" % u_p90)

    respread = any("v_respread" in f for f in flags)
    pts = {}
    for nm in names:
        p, _s = _soft_argmax(part_prob, uv_use, addr_dict[nm], gx, gy, tau)
        if p is None:
            continue
        x, y, clamped = _nearest_in_mask(p[0], p[1], back_m)   # (A) 体外夹回
        if clamped:
            status = "clamped"
        elif respread:
            status = "respread"
        else:
            status = "ok"
        # 下背不可见: U 地址超出可见上界的点标 lowfov(优先级最高)
        if u_p90 < U_TRIG and addr_dict[nm][1] > u_p90:
            status = "lowfov"
        pts[nm] = (x, y, status)
    return pts, flags, meta
