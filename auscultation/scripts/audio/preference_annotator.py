"""Pairwise preference annotation tool for the audio reward model (Head-Reward).

Doctors compare two lung-sound clips ("which is the more usable recording?"); each choice becomes a
`reward_schema.Preference` row that trains `ausc/audio/reward_model.py`. Data-agnostic: it annotates
ANY clip pool — open data / our own recordings now, robot flight-data segments later — because the
audio reward is DECOUPLED from flight data (quality is a property of the clip, not who held the
stethoscope), so this unblocks the reward model without waiting on teleop hardware.

Pair selection (active learning — matters when doctor time is scarce): if the AST Q head is available
(`--smart`), score every clip and prefer pairs with SIMILAR Q (the fine distinctions the model is
unsure about) plus a few wide-gap "anchor" pairs (sanity checks). Else random pairs.

    python scripts/audio/preference_annotator.py --clips <dir_of_wavs> [--port 8799] [--annotator drX] [--smart]
    # open http://localhost:8799 ; choices append to <clips>/preferences.jsonl

Stdlib only for serving (doctors run it with no heavy deps); --smart needs torch/transformers for Q.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # auscultation/


def build_pair_queue(clips: list[str], smart: bool, seed: int = 0, n_pairs: int = 200) -> list[tuple[int, int]]:
    """Ordered (i,j) index pairs. Smart: mostly similar-Q (informative) + a few wide-gap anchors."""
    import numpy as np
    rng = np.random.RandomState(seed)
    n = len(clips)
    if not smart or n < 4:
        pairs = [(int(a), int(b)) for a, b in rng.randint(0, n, (n_pairs, 2)) if a != b]
        return pairs[:n_pairs]
    sys.path.insert(0, os.path.join(ROOT, "ausc", "audio"))
    import soundfile as sf
    from q_head import make_quality_fn
    q = make_quality_fn()
    scores = []
    for c in clips:
        y, sr = sf.read(c); y = y.mean(1) if getattr(y, "ndim", 1) > 1 else y
        scores.append(q(y, sr))
    scores = np.asarray(scores)
    order = np.argsort(scores)
    pairs = []
    for k in range(len(order) - 1):                         # adjacent in Q-order = similar Q = informative
        pairs.append((int(order[k]), int(order[k + 1])))
    for _ in range(max(4, n_pairs // 20)):                  # a few wide-gap anchors (model-obvious)
        pairs.append((int(order[0]) if rng.rand() < 0.5 else int(order[rng.randint(0, n // 3 + 1)]),
                      int(order[-1]) if rng.rand() < 0.5 else int(order[rng.randint(2 * n // 3, n)])))
    rng.shuffle(pairs)
    return pairs[:n_pairs]


PAGE = """<!doctype html><meta charset=utf-8><title>Auscultation preference annotation</title>
<style>body{font-family:system-ui;max-width:720px;margin:2rem auto;text-align:center}
button{font-size:1.1rem;padding:.7rem 1.4rem;margin:.4rem;border-radius:8px;border:1px solid #ccc;cursor:pointer}
.clip{display:inline-block;margin:1rem;padding:1rem;border:1px solid #ddd;border-radius:10px}
#prog{color:#888}</style>
<h2>Which recording is the more <b>usable</b> lung sound?</h2>
<div id=prog></div>
<div><span class=clip><h3>A</h3><audio id=a controls></audio></span>
<span class=clip><h3>B</h3><audio id=b controls></audio></span></div>
<div><button onclick=choose('a')>A is better</button>
<button onclick=choose('b')>B is better</button>
<button onclick=choose('tie')>About the same</button>
<button onclick=choose('skip')>Can't tell / skip</button></div>
<script>
let cur=null;
async function load(){let r=await fetch('/pair');cur=await r.json();
 if(cur.done){document.body.innerHTML='<h2>All pairs annotated — thank you.</h2>';return;}
 a.src='/audio?i='+cur.a;b.src='/audio?i='+cur.b;
 document.getElementById('prog').textContent='pair '+cur.k+' / '+cur.total+'  ('+cur.done_n+' recorded)';}
async function choose(p){if(!cur)return;await fetch('/choice',{method:'POST',
 headers:{'Content-Type':'application/json'},body:JSON.stringify({pair_id:cur.pair_id,a:cur.a,b:cur.b,preferred:p})});load();}
load();</script>"""


def make_handler(clips, queue, pref_path, annotator):
    state = {"k": 0, "done_n": _count_lines(pref_path)}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _json(self, obj, code=200):
            b = json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                b = PAGE.encode()
                self.send_response(200); self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
            elif u.path == "/pair":
                if state["k"] >= len(queue):
                    return self._json({"done": True})
                i, j = queue[state["k"]]
                self._json({"pair_id": f"p{state['k']:04d}", "a": i, "b": j, "k": state["k"] + 1,
                            "total": len(queue), "done_n": state["done_n"], "done": False})
            elif u.path == "/audio":
                i = int(parse_qs(u.query)["i"][0])
                with open(clips[i], "rb") as f:
                    data = f.read()
                self.send_response(200); self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
            else:
                self.send_response(404); self.end_headers()

        def do_POST(self):
            if urlparse(self.path).path != "/choice":
                self.send_response(404); self.end_headers(); return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["k"] += 1
            if body["preferred"] in ("a", "b", "tie"):
                rec = {"pair_id": body["pair_id"], "segment_a": os.path.basename(clips[body["a"]]),
                       "segment_b": os.path.basename(clips[body["b"]]),
                       "preferred": {"a": "a", "b": "b", "tie": "tie"}[body["preferred"]],
                       "annotator": annotator, "ts_wall": "", "schema_version": "1.0"}
                with open(pref_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                state["done_n"] += 1
            self._json({"ok": True})

    return H


def _count_lines(p):
    return sum(1 for _ in open(p)) if os.path.exists(p) else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True, help="directory of lung-sound wav clips to compare")
    ap.add_argument("--port", type=int, default=8799)
    ap.add_argument("--annotator", default="anon")
    ap.add_argument("--smart", action="store_true", help="Q-informed active-learning pairs (needs torch)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    clips = sorted(glob.glob(os.path.join(a.clips, "*.wav")))
    if len(clips) < 2:
        sys.exit(f"need >=2 wavs in {a.clips}, found {len(clips)}")
    queue = build_pair_queue(clips, a.smart, a.seed)
    pref_path = os.path.join(a.clips, "preferences.jsonl")
    print(f"[annotator] {len(clips)} clips, {len(queue)} pairs ({'Q-informed' if a.smart else 'random'}), "
          f"annotator={a.annotator}\n[annotator] http://localhost:{a.port}  -> {pref_path}")
    HTTPServer(("127.0.0.1", a.port), make_handler(clips, queue, pref_path, a.annotator)).serve_forever()
