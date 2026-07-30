"""Pairwise preference annotation tool for the audio reward model (Head-Reward).

Doctors compare two lung-sound clips ("which is the more usable recording?"); each choice becomes a
`reward_schema.Preference` row that trains `ausc/audio/reward_model.py`. Data-agnostic: it annotates
ANY clip pool — open data / our own recordings now, robot flight-data segments later — because the
audio reward is DECOUPLED from flight data (quality is a property of the clip, not who held the
stethoscope), so this unblocks the reward model without waiting on teleop hardware.

Pair selection (active learning — matters when doctor time is scarce): if the AST Q head is available
(`--smart`), score every clip and prefer pairs with SIMILAR Q (the fine distinctions the model is
unsure about) plus a few wide-gap "anchor" pairs (sanity checks). Else random pairs.

    python scripts/audio/preference_annotator.py --clips <pool-release> [--preferences <persistent.jsonl>]
    # choices are always stored outside the immutable pool release

Stdlib only for serving (doctors run it with no heavy deps); --smart needs torch/transformers for Q.
"""
from __future__ import annotations

import argparse
import fcntl
import glob
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Mapping, Optional, Union
from urllib.parse import urlparse, parse_qs

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # auscultation/


class PreferenceContractError(RuntimeError):
    """The preference output or clip identity is outside the immutable pool contract."""


def load_pool_identity(pool_release: Union[str, Path]) -> dict:
    from build_sprsound_pool import verify_pool_release

    release = Path(pool_release)
    verified = verify_pool_release(release, expected_quotas=None)
    clips = {
        row["destination"]: row["content_sha256"] for row in verified["rows"]
    }
    return {
        "clips": clips,
        "release_path": release.resolve().as_posix(),
        "release_id": verified["release_id"],
        "schema": "wam_audio_preference_pool_identity_v1",
    }


def resolve_preferences_path(
    pool_release: Union[str, Path],
    *,
    explicit_path: Optional[Union[str, Path]] = None,
    preferences_root: Optional[Union[str, Path]] = None,
) -> Path:
    identity = load_pool_identity(pool_release)
    release = Path(identity["release_path"])
    if explicit_path is not None:
        candidate = Path(explicit_path)
    else:
        root = (
            Path(preferences_root)
            if preferences_root is not None
            else Path("/home/ssh-ziyue/wam_ausc/data/preferences")
        )
        candidate = root / identity["release_id"] / "preferences.jsonl"
    if not candidate.is_absolute():
        raise PreferenceContractError("preferences path must be absolute")
    resolved = candidate.resolve(strict=False)
    if resolved == release or release in resolved.parents:
        raise PreferenceContractError("preferences path must not be inside the pool release")
    return candidate.absolute()


def _open_real_directory(path: Path, *, create: bool) -> int:
    target = path.absolute()
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(os.sep, flags)
    try:
        for part in target.parts[1:]:
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
            info = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
            if not stat.S_ISDIR(info.st_mode):
                raise PreferenceContractError(
                    "preferences path contains a symlink or non-directory"
                )
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except PreferenceContractError:
        os.close(descriptor)
        raise
    except (FileNotFoundError, NotADirectoryError, OSError) as exc:
        os.close(descriptor)
        raise PreferenceContractError(
            "preferences path contains a symlink or unsafe directory"
        ) from exc


def append_preference(path: Union[str, Path], row: Mapping[str, object]) -> None:
    """Append one unique row through a locked no-follow descriptor."""
    target = Path(path)
    if not target.is_absolute() or target.name in ("", ".", ".."):
        raise PreferenceContractError("preferences path must be an absolute file path")
    parent = _open_real_directory(target.parent, create=True)
    descriptor = None
    try:
        descriptor = os.open(
            target.name,
            os.O_RDWR | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent,
        )
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise PreferenceContractError("preferences destination is not a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        identity = (
            row.get("pool_release_id"), row.get("annotator"), row.get("pair_id"),
        )
        if not all(isinstance(value, str) and value for value in identity):
            raise PreferenceContractError("preference row identity is invalid")
        os.lseek(descriptor, 0, os.SEEK_SET)
        existing = os.read(descriptor, os.fstat(descriptor).st_size)
        for line in existing.splitlines():
            try:
                value = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise PreferenceContractError("existing preferences file is invalid") from exc
            if (
                value.get("pool_release_id"), value.get("annotator"), value.get("pair_id"),
            ) == identity:
                raise PreferenceContractError("duplicate preference pair identity")
        payload = json.dumps(
            dict(row), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8") + b"\n"
        os.lseek(descriptor, 0, os.SEEK_END)
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    except OSError as exc:
        raise PreferenceContractError("preference append failed closed") from exc
    finally:
        if descriptor is not None:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
        os.close(parent)


def validate_choice(
    value: object, *, a: int, b: int, pair_id: str,
) -> str:
    if not isinstance(value, dict) or set(value) != {"a", "b", "pair_id", "preferred"}:
        raise PreferenceContractError("choice does not match the current pair")
    if (
        type(value["a"]) is not int
        or type(value["b"]) is not int
        or value["a"] != a
        or value["b"] != b
        or value["pair_id"] != pair_id
        or value["preferred"] not in ("a", "b", "tie", "skip")
    ):
        raise PreferenceContractError("choice does not match the current pair")
    return value["preferred"]


def _server_pair_id(pool_identity, annotator: str, index: int, a: int, b: int) -> str:
    payload = "\0".join((
        str(pool_identity["release_id"]), annotator, str(index),
        str(pool_identity["clips"][Path(pool_identity["clip_paths"][a]).name]),
        str(pool_identity["clips"][Path(pool_identity["clip_paths"][b]).name]),
    )).encode("utf-8")
    return "pair-" + hashlib.sha256(payload).hexdigest()[:24]


def _completed_pair_ids(path: Union[str, Path]) -> set[str]:
    target = Path(path)
    if not os.path.lexists(str(target)):
        return set()
    parent = _open_real_directory(target.parent, create=False)
    descriptor = None
    try:
        descriptor = os.open(
            target.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent,
        )
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise PreferenceContractError("preferences destination is not a regular file")
        raw = b""
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            raw += chunk
        result = set()
        for line in raw.splitlines():
            try:
                row = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise PreferenceContractError("existing preferences file is invalid") from exc
            pair_id = row.get("pair_id") if isinstance(row, dict) else None
            if not isinstance(pair_id, str) or not pair_id:
                raise PreferenceContractError("existing preference pair id is invalid")
            result.add(pair_id)
        return result
    except OSError as exc:
        raise PreferenceContractError("preferences file cannot be read safely") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)


def _resume_queue_index(queue, pool_identity, annotator: str, completed: set[str]) -> int:
    index = 0
    while index < len(queue):
        a, b = queue[index]
        if _server_pair_id(pool_identity, annotator, index, a, b) not in completed:
            break
        index += 1
    return index


def preference_record(
    *,
    pool_identity: Mapping[str, object],
    clip_a: Union[str, Path],
    clip_b: Union[str, Path],
    preferred: str,
    annotator: str,
    pair_id: str,
) -> dict:
    if preferred not in ("a", "b", "tie"):
        raise PreferenceContractError("preference choice is invalid")
    if not isinstance(annotator, str) or not annotator or not isinstance(pair_id, str) or not pair_id:
        raise PreferenceContractError("preference annotator or pair id is invalid")
    if (
        not isinstance(pool_identity, Mapping)
        or pool_identity.get("schema") != "wam_audio_preference_pool_identity_v1"
        or not isinstance(pool_identity.get("clips"), dict)
        or not isinstance(pool_identity.get("release_id"), str)
        or not isinstance(pool_identity.get("release_path"), str)
    ):
        raise PreferenceContractError("preference pool identity is invalid")
    release = Path(pool_identity["release_path"])
    clips = pool_identity["clips"]
    paths = [Path(clip_a), Path(clip_b)]
    hashes = []
    for path in paths:
        if path.is_symlink() or not path.is_file() or path.parent.resolve() != release.resolve():
            raise PreferenceContractError("preference clip is outside the pool release")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if clips.get(path.name) != digest:
            raise PreferenceContractError("preference clip hash does not match the pool")
        hashes.append(digest)
    if hashes[0] == hashes[1]:
        raise PreferenceContractError("preference pair must contain distinct clips")
    return {
        "annotator": annotator,
        "pair_id": pair_id,
        "pool_release_id": pool_identity["release_id"],
        "preferred": preferred,
        "schema_version": "wam_audio_preference_v2",
        "segment_a": paths[0].name,
        "segment_a_sha256": hashes[0],
        "segment_b": paths[1].name,
        "segment_b_sha256": hashes[1],
        "ts_wall": "",
    }


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


def make_handler(clips, queue, pref_path, annotator, pool_identity):
    pool_identity = dict(pool_identity)
    pool_identity["clip_paths"] = list(clips)
    completed = _completed_pair_ids(pref_path)
    state = {
        "k": _resume_queue_index(queue, pool_identity, annotator, completed),
        "done_n": len(completed),
    }

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
                pair_id = _server_pair_id(
                    pool_identity, annotator, state["k"], i, j,
                )
                self._json({"pair_id": pair_id, "a": i, "b": j, "k": state["k"] + 1,
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
            if state["k"] >= len(queue):
                return self._json({"error": "no current pair"}, code=409)
            i, j = queue[state["k"]]
            pair_id = _server_pair_id(pool_identity, annotator, state["k"], i, j)
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                preferred = validate_choice(body, a=i, b=j, pair_id=pair_id)
            except (KeyError, ValueError, json.JSONDecodeError, PreferenceContractError) as exc:
                return self._json({"error": str(exc)}, code=409)
            if preferred in ("a", "b", "tie"):
                rec = preference_record(
                    pool_identity=pool_identity,
                    clip_a=clips[i], clip_b=clips[j],
                    preferred=preferred, annotator=annotator,
                    pair_id=pair_id,
                )
                appended = True
                try:
                    append_preference(Path(pref_path), rec)
                except PreferenceContractError as exc:
                    if "duplicate preference pair identity" not in str(exc):
                        return self._json({"error": str(exc)}, code=409)
                    appended = False
                if appended:
                    state["done_n"] += 1
            state["k"] += 1
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
    ap.add_argument("--preferences", help="absolute persistent preferences.jsonl path")
    ap.add_argument("--preferences-root", help="absolute persistent preferences root")
    a = ap.parse_args()
    pool_identity = load_pool_identity(a.clips)
    clips = [
        os.path.join(a.clips, name) for name in sorted(pool_identity["clips"])
    ]
    if len(clips) < 2:
        sys.exit(f"need >=2 wavs in {a.clips}, found {len(clips)}")
    queue = build_pair_queue(clips, a.smart, a.seed)
    pref_path = str(resolve_preferences_path(
        a.clips, explicit_path=a.preferences,
        preferences_root=a.preferences_root,
    ))
    print(f"[annotator] {len(clips)} clips, {len(queue)} pairs ({'Q-informed' if a.smart else 'random'}), "
          f"annotator={a.annotator}\n[annotator] http://localhost:{a.port}  -> {pref_path}")
    HTTPServer(
        ("127.0.0.1", a.port),
        make_handler(clips, queue, pref_path, a.annotator, pool_identity),
    ).serve_forever()
