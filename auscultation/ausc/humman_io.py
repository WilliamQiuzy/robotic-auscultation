"""HuMMan 视频/IO 工具。"""
import glob, os, re


def find_kinect_videos(root, sid):
    """该序列的 Kinect 彩色视频: {cam_key(kinect_color_NNN): mp4路径}。文件名 kinect_NNN.mp4。"""
    out = {}
    for p in glob.glob(os.path.join(root, "**", sid, "**", "kinect_*.mp4"), recursive=True):
        m = re.search(r"kinect_(\d+)\.mp4", os.path.basename(p))
        if m:
            out[f"kinect_color_{m.group(1)}"] = p
    return out
