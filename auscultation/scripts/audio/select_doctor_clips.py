#!/usr/bin/env python3
"""给医生标注挑一批肺音 clip(跨数据集 + 跨质量分层, 不要太多), 转成浏览器安全格式。

- 从 S1 弱标 manifest 读 q_proxy, 按 q_proxy 均匀分层抽样(让医生看到好/中/差全谱)。
- 每条转 16kHz mono 16-bit WAV(浏览器都能放; 含 ICBHI 24-bit)。
- 匿名化: clip_0001... 中性 id, 打乱顺序, manifest 不含数据集/质量(防偏见)。
- 私有映射(clip→数据集/原文件/q_proxy)单独存, 【不进公开站点】, 仅供我们事后分析。

用法(pod):
  python scripts/audio/select_doctor_clips.py --s1_dir /workspace/data/audio/_s1 \
    --datasets sprsound,icbhi,hf_lung --n_per 30 --out /workspace/data/audio/_doctor
"""
import argparse, glob, json, os, sys, wave
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "ausc", "audio"))
import proxies as P  # noqa: E402


def write_wav16(src_wav, path):
    xn, srn = P.read_wav(src_wav)
    x16, _ = P.resample(xn, srn, 16000)
    pcm = (np.clip(x16, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(pcm.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1_dir", required=True)
    ap.add_argument("--datasets", default="sprsound,icbhi,hf_lung")
    ap.add_argument("--n_per", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(os.path.join(a.out, "clips"), exist_ok=True)

    clips, mapping, idx = [], {}, 1
    for ds in a.datasets.split(","):
        man = os.path.join(a.s1_dir, f"manifest_{ds}.jsonl")
        rows = [json.loads(l) for l in open(man)] if os.path.exists(man) else []
        rows = [r for r in rows if r.get("q_proxy") is not None and os.path.exists(r["wav"])]
        if not rows:
            print(f"[warn] no rows for {ds}"); continue
        order = np.argsort([r["q_proxy"] for r in rows])          # 按 q_proxy 排序
        pick = np.unique(np.linspace(0, len(rows) - 1, a.n_per).astype(int))  # 均匀分层取
        for i in pick:
            r = rows[order[i]]
            cid = f"clip_{idx:04d}"; idx += 1
            try:
                write_wav16(r["wav"], os.path.join(a.out, "clips", cid + ".wav"))
            except Exception as e:
                print(f"[skip] {r['wav']}: {e}"); continue
            clips.append({"id": cid, "file": f"clips/{cid}.wav"})
            mapping[cid] = {"dataset": ds, "orig": r["wav"], "q_proxy": r["q_proxy"]}
    rng.shuffle(clips)                                            # 打乱显示顺序(数据集交错, 防偏见)
    json.dump(clips, open(os.path.join(a.out, "manifest.json"), "w"), indent=2)
    json.dump(mapping, open(os.path.join(a.out, "_private_mapping.json"), "w"), indent=2)
    print(f"[done] {len(clips)} clips → {a.out}  (datasets: {a.datasets}, n_per={a.n_per})")


if __name__ == "__main__":
    main()
