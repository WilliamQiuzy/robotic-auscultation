#!/usr/bin/env python3
"""下载 Roboflow 脊柱侧弯裸背数据集(需免费 API key)。
用法: ROBOFLOW_API_KEY=xxx python download_scoliosis.py
或网页 https://universe.roboflow.com/a-nude-back-picture/scoliosis-qywiy-fjjfu-jzarn-oagfj-vuwup 点 Download 导出 zip 解压到 data/scoliosis_back/。"""
import os
from roboflow import Roboflow
key = os.environ.get("ROBOFLOW_API_KEY")
assert key, "需要设置 ROBOFLOW_API_KEY 环境变量(roboflow.com 免费注册→Settings)"
rf = Roboflow(api_key=key)
# workspace/project/version 按 Universe 页面的 download 代码片段填
proj = rf.workspace("a-nude-back-picture").project("scoliosis-qywiy-fjjfu-jzarn-oagfj-vuwup")
proj.version(1).download("coco", location="../data/scoliosis_back")
print("downloaded -> data/scoliosis_back")
