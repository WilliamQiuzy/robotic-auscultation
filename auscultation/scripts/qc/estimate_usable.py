#!/usr/bin/env python3
"""估算全数据集可用(前胸/后背)帧数 + 数据体积。只用 smpl_params+cameras(几何),不需视频。"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "ausc"))


def main():
    import smplx, torch
    from frame_filter import front_core_indices, classify_frame
    root = "../data/humman"
    m = smplx.SMPL(model_path="../data/smpl/SMPL_NEUTRAL.pkl", gender="neutral", batch_size=1)
    faces = m.faces; core = front_core_indices(m)
    seqs = sorted(glob.glob(os.path.join(root, "smpl_params", "*")))
    STRIDE = 8                                   # 抽帧近似(连续帧近重复)
    DEDUP = 12                                   # 训练去近重复: 每~12帧保1个独立姿态
    tot_frames = 0; front_ok = 0; back_frame_ok = 0; back_cam_pairs = 0; nseq = 0
    for sq in seqs:
        sid = os.path.basename(sq)
        cj = os.path.join(root, "point_cameras", sid, "cameras.json")
        if not os.path.exists(cj):
            continue
        params = dict(np.load(os.path.join(sq, "smpl_params.npz")))
        cams = json.load(open(cj)); nfp = params["betas"].shape[0]
        Riph = np.asarray(cams["iphone"]["R"]) if "iphone" in cams else None
        kcams = [(k, np.asarray(cams[k]["R"])) for k in cams if k.startswith("kinect_color_")]
        nseq += 1
        for fr in range(0, nfp, STRIDE):
            tot_frames += 1
            with torch.no_grad():
                o = m(betas=torch.tensor(params["betas"][fr:fr+1], dtype=torch.float32),
                      body_pose=torch.tensor(params["body_pose"][fr:fr+1], dtype=torch.float32),
                      global_orient=torch.tensor(params["global_orient"][fr:fr+1], dtype=torch.float32),
                      transl=torch.tensor(params["transl"][fr:fr+1], dtype=torch.float32))
            v = o.vertices[0].numpy(); j = o.joints[0].numpy()
            if Riph is not None and classify_frame(v, faces, Riph, core, joints=j)[0] == "front":
                front_ok += 1
            nb = sum(1 for _, R in kcams if classify_frame(v, faces, R, core, joints=j)[0] == "back")
            if nb:
                back_frame_ok += 1; back_cam_pairs += nb
    sc = STRIDE  # 把抽帧放大回全量
    print(f"序列数: {nseq}  | 总帧(估): {tot_frames*sc:,}")
    print("=== 可用帧 (抽帧×%d放大) ===" % sc)
    print(f"  前胸(iPhone正脸+不弯腰)        : {front_ok*sc:,} 帧")
    print(f"  后背(≥1个Kinect后向+不弯腰)    : {back_frame_ok*sc:,} 帧")
    print(f"  后背(帧×后向相机 多视角对)     : {back_cam_pairs*sc:,} 张")
    print(f"  注: 后背需Kinect视频; 已下part_01(453序列), part_02(~454序列)未下")
    # 去近重复(每DEDUP帧保1)
    f_d = front_ok*sc/DEDUP; b_d = back_frame_ok*sc/DEDUP; bc_d = back_cam_pairs*sc/DEDUP
    print(f"=== 去近重复(每%d帧保1独立姿态) ===" % DEDUP)
    print(f"  前胸 ~{f_d:,.0f} 张 | 后背 ~{b_d:,.0f} 帧 | 后背多视角 ~{bc_d:,.0f} 张")
    # 体积: 每样本 img+iuv. 当前裁剪~380KB; 384²约~120KB
    for res, kb in [("当前裁剪分辨率", 380), ("384² 训练分辨率", 120)]:
        base = (f_d + bc_d)            # 前胸 + 后背多视角(去重)
        for aug, label in [(1, "原图(canonical)"), (6, "+6增强/张")]:
            gb = base * aug * kb / 1024 / 1024
            print(f"  [{res}] {label}: {base*aug:,.0f} 张 ≈ {gb:.1f} GB")


if __name__ == "__main__":
    main()
