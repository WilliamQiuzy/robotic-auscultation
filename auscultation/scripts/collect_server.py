"""Web console for point-conditioned auscultation data collection.

One page the doctor + operator watch: the live camera with the current auscultation target overlaid,
a feedback HUD (state / force / quality / cycles / point list), and keyboard controls. The backend
(this server, on linbox) owns the camera + sweep state; the browser is just display + controls, so the
UI is decoupled from the hardware.

Keys:  SPACE = confirm (heard a good sound → next point)   R = redo (keep trying at this point)
       ← / → = previous / next point                       S = start / stop recording   F = fullscreen

  python scripts/collect_server.py --fake                 # synthetic camera + sim sensors (any machine)
  python scripts/collect_server.py --camera-backend orbbec --out ~/wam_data/sessions   # real (linbox)

--fake drives the REAL PointSweep/overlay with a simulated scene + sensors, so the whole console works
without hardware; on linbox swap in the Orbbec camera + live force/Q/kinematics + the teleop launcher.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
import sys
sys.path[:0] = [os.path.join(ROOT, "ausc", "perception"), os.path.join(ROOT, "ausc")]
from ausc_targets import TargetLog, load_targets, overlay          # noqa: E402
from ausc_feedback import FeedbackConfig, PointSweep, State         # noqa: E402


class Collector:
    """Owns the camera + sweep + session; a background thread renders/overlays + advances the state."""

    def __init__(self, targets, fake=True, camera_backend="orbbec", camera_index=0, out_dir=None):
        self.targets = targets
        self.sweep = PointSweep(targets, FeedbackConfig())
        self.fake = fake
        self.camera_backend = camera_backend
        self.camera_index = camera_index
        self.out_dir = out_dir or os.path.join(ROOT, "data", "sessions")
        self.recording = False
        self.session_dir = None
        self.t_start = None
        self.tlog = None
        self.jpeg = self._encode(np.zeros((480, 640, 3), np.uint8))
        self.state = {}
        self._occluded = False
        self._sim = {"tipx": 600, "tipy": 40, "f": 0.0, "q": 0.0, "phase": 0.0, "settle": 0.0}
        self._lock = threading.Lock()
        self._cam = None
        threading.Thread(target=self._loop, daemon=True).start()

    # -- controls (from the browser) --------------------------------------
    def control(self, action):
        with self._lock:
            if action == "confirm":
                self._advance(mark_done=True)       # doctor's ear = arbiter: confirm completes NOW,
                #                                     any time, regardless of cycles/Q/sensors present
            elif action == "redo":
                self.sweep.redo()
            elif action == "next":
                self._advance(mark_done=False)      # skip forward, don't mark this point done
            elif action == "prev":
                self.sweep.idx = (self.sweep.idx - 1) % len(self.targets); self.sweep.fb.reset()
                self._sim.update(tipx=600, tipy=40, f=0, q=0, settle=0)
            elif action == "start":
                self._start()
            elif action == "stop":
                self._stop()
        return {"ok": True}

    def _advance(self, mark_done):
        if mark_done and self.sweep.idx not in self.sweep.completed:
            self.sweep.completed.append(self.sweep.idx)
        self.sweep.idx = (self.sweep.idx + 1) % len(self.targets)
        self.sweep.fb.reset()
        self._sim.update(tipx=600, tipy=40, f=0, q=0, settle=0)   # fake: send arm back for the next point

    def _start(self):
        if self.recording:
            return
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.out_dir, f"teleop_{stamp}")
        os.makedirs(self.session_dir, exist_ok=True)
        self.tlog = TargetLog(self.session_dir)
        self.t_start = time.monotonic()
        self.recording = True
        # NOTE (linbox): also launch record_session.py --session-dir + the teleop here so robot.csv +
        # audio/force/rgb land in the same dir on the same monotonic clock (record_teleop_session.sh).

    def _stop(self):
        if not self.recording:
            return
        self.recording = False
        if self.tlog:
            self.tlog.close(); self.tlog = None

    # -- the render + feedback loop ---------------------------------------
    def _sensors_and_frame(self):
        """Return (bgr_frame, ee_to_target_m, force_n, q_usable, new_cycle). Fake = simulate; else read
        the real camera + force/Q/kinematics (wire on linbox). Frame is BGR (cv2-native) throughout."""
        s = self._sim
        tp = self.targets[self.sweep.idx]
        h, w = 480, 640
        px, py = int(tp.uv[0] * w), int(tp.uv[1] * h)
        if self.fake:
            frame = self._render_back(w, h)
            dx, dy = px - s["tipx"], py - s["tipy"]; d = math.hypot(dx, dy)
            if d > 1:
                step = min(d, 3.0)
                s["tipx"] += dx / d * step; s["tipy"] += dy / d * step
            self._draw_arm(frame, s["tipx"], s["tipy"])
            reached = d <= 26
            prev = s["phase"]; s["phase"] = (s["phase"] + 0.02) % 1.0
            new_cycle = s["phase"] < prev
            s["settle"] = min(1.0, s["settle"] + 0.02) if reached else 0.0
            s["f"] += ((4.0 + 0.7 * math.sin(s["phase"] * 6.28) if reached else 0.0) - s["f"]) * 0.15
            s["q"] += ((0.72 + 0.18 * s["settle"] if reached and 3 <= s["f"] <= 5 else 0.15) - s["q"]) * 0.1
            return frame, d / 500.0, s["f"], s["q"], new_cycle
        # --- real (linbox): camera + live sensors ---
        f = self._cam.latest_frame() if self._cam else None
        if f is not None and getattr(f, "frame", None) is not None:
            frame = np.ascontiguousarray(f.frame[:, :, ::-1])        # camera is RGB → BGR for cv2
        else:
            frame = self._placeholder(f"waiting for {self.camera_backend} camera…")
        # TODO(linbox): ee_to_target from kinematics(tip) vs target xyz; force from ForceSensor;
        # q from the streaming Q head; new_cycle from the Hall respiratory detector. Until then the
        # video + point overlay + manual confirm work; force/Q read 0 (honest — no sensor on this host).
        return frame, 1e9, 0.0, 0.0, False

    @staticmethod
    def _placeholder(text):                                          # honest "no camera yet" frame
        import cv2
        img = np.full((480, 640, 3), (28, 20, 14), np.uint8)
        cv2.putText(img, text, (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 170, 200), 1)
        return img

    def _loop(self):
        if not self.fake:
            try:
                from camera_stream import CameraStream
                self._cam = CameraStream(backend=self.camera_backend, index=self.camera_index, rgb=True).start()
            except Exception as e:  # noqa: BLE001
                print(f"[collect] camera open failed ({e}); falling back to fake"); self.fake = True
        while True:
            with self._lock:
                frame, ee_m, force, q, new_cycle = self._sensors_and_frame()
                st = self.sweep.fb.update(ee_m, force, q, new_cycle)   # drives the HUD hints only
                status = self.sweep.fb.status(ee_m, force, q)
                self._occluded = ee_m * 500 < 30 and st is not State.DONE
                if st is State.DONE:                                   # only fires in autonomous mode
                    self._advance(mark_done=True)                      # (require_confirm=False); confirm-
                #                                                        mode completion goes via control()
                if self.recording and self.tlog:
                    self.tlog.log(time.monotonic(), self.targets[self.sweep.idx], frame.shape[1], frame.shape[0])
                overlay(frame, self.targets, self.sweep.idx, self._occluded, "", self.sweep.completed)
                self.jpeg = self._encode(frame)
                self.state = {
                    "point": self.targets[self.sweep.idx].name, "idx": self.sweep.idx,
                    "n": len(self.targets), "state": st.value, "status": status,
                    "force": round(force, 2), "q": round(q, 2), "cycles": self.sweep.fb.good_cycles,
                    "need": self.sweep.cfg.need_cycles, "occluded": self._occluded,
                    "completed": list(self.sweep.completed), "recording": self.recording,
                    "elapsed": round(time.monotonic() - self.t_start, 1) if self.t_start else 0,
                    "names": [t.name for t in self.targets],
                }
            time.sleep(1 / 30.0)

    @staticmethod
    def _render_back(w, h):                                        # BGR, cv2-native
        import cv2
        img = np.full((h, w, 3), (40, 27, 20), np.uint8)           # dark cool slate ground
        cv2.ellipse(img, (w // 2, h // 2), (150, 210), 0, 0, 360, (104, 74, 56), -1)   # back
        cv2.ellipse(img, (w // 2, h // 2), (150, 210), 0, 0, 360, (150, 110, 90), 2)
        cv2.line(img, (w // 2, h // 2 - 195), (w // 2, h // 2 + 195), (130, 90, 70), 2)  # spine
        return img

    def _draw_arm(self, bgr, tx, ty):                             # draw directly on BGR
        import cv2
        base = (bgr.shape[1] - 14, 14); mid = (int((base[0] + tx) / 2 + 40), int((base[1] + ty) / 2 - 30))
        for wdt, col in ((15, (45, 60, 80)), (9, (70, 95, 120))):
            cv2.line(bgr, base, mid, col, wdt); cv2.line(bgr, mid, (int(tx), int(ty)), col, wdt)
        cv2.circle(bgr, (int(tx), int(ty)), 11, (150, 170, 200), -1)

    @staticmethod
    def _encode(bgr):
        import cv2
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else b""


def make_handler(col: Collector, page: bytes):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/":
                self._send(page, "text/html")
            elif self.path == "/state":
                self._send(json.dumps(col.state).encode(), "application/json")
            elif self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        j = col.jpeg
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(j)}\r\n\r\n".encode())
                        self.wfile.write(j); self.wfile.write(b"\r\n")
                        time.sleep(1 / 25.0)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_response(404); self.end_headers()

        def do_POST(self):
            if self.path == "/control":
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self._send(json.dumps(col.control(body.get("action"))).encode(), "application/json")
            else:
                self.send_response(404); self.end_headers()

        def _send(self, b, ct):
            self.send_response(200); self.send_header("Content-Type", ct)
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    return H


PAGE = ("""<!doctype html><meta charset=utf-8><title>Auscultation collection console</title>""" + r"""
<style>
:root{--bg:#080d16;--panel:#0e1626;--edge:#1b2740;--edge2:#24344f;--ink:#d3ddec;--mut:#7386a0;
 --dim:#4a5a74;--cyan:#38bdf8;--amber:#fbbf24;--good:#34d399;--warn:#f59e0b;--crit:#fb7185;
 --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;--sans:system-ui,-apple-system,sans-serif}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(1200px 600px at 30% -10%,#0d182b,#080d16 55%);
 color:var(--ink);font-family:var(--sans);min-height:100vh;padding:18px;display:flex;justify-content:center}
.wrap{width:100%;max-width:1120px}
.top{display:flex;justify-content:space-between;align-items:baseline;border-bottom:1px solid var(--edge);
 padding-bottom:12px;margin-bottom:16px;flex-wrap:wrap;gap:10px}
.title{font-size:15px;font-weight:650}.title b{color:var(--amber)}
.sub{font-family:var(--mono);font-size:11px;color:var(--mut);letter-spacing:.06em;text-transform:uppercase}
#rec{font-family:var(--mono);font-size:12px;letter-spacing:.05em}
.grid{display:grid;grid-template-columns:1fr 300px;gap:16px}@media(max-width:820px){.grid{grid-template-columns:1fr}}
.feed{position:relative;background:#0a1120;border:1px solid var(--edge);border-radius:12px;overflow:hidden;aspect-ratio:4/3}
.feed img{width:100%;height:100%;object-fit:cover;display:block}
.tag{position:absolute;top:10px;left:12px;font-family:var(--mono);font-size:10px;letter-spacing:.12em;
 color:var(--mut);text-transform:uppercase}
.recdot{position:absolute;top:10px;right:12px;font-family:var(--mono);font-size:11px;letter-spacing:.1em;
 display:flex;align-items:center;gap:6px;color:var(--dim)}.recdot.on{color:var(--crit)}
.recdot i{width:7px;height:7px;border-radius:50%;background:currentColor}.recdot.on i{animation:bl 1.3s infinite}
@keyframes bl{50%{opacity:.25}}
.statusbar{position:absolute;left:0;right:0;bottom:0;background:rgba(10,15,26,.85);backdrop-filter:blur(3px);
 padding:9px 12px;font-size:13px;color:var(--ink);border-top:1px solid var(--edge)}
.side{display:flex;flex-direction:column;gap:12px}
.card{background:var(--panel);border:1px solid var(--edge);border-radius:12px;padding:14px}
.lbl{font-family:var(--mono);font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:var(--dim);margin-bottom:8px}
.tgt{display:flex;justify-content:space-between;align-items:center}
.tname{font-family:var(--mono);font-size:24px;font-weight:700}
.pill{font-family:var(--mono);font-size:11px;font-weight:600;padding:4px 10px;border-radius:999px;border:1px solid currentColor}
.row{display:flex;align-items:center;gap:9px;margin-top:11px}
.rl{font-family:var(--mono);font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:var(--dim);width:48px;flex:none}
.bar{position:relative;flex:1;height:9px;border-radius:6px;background:#0a1120;border:1px solid var(--edge);overflow:hidden}
.bar>span{position:absolute;inset:0 100% 0 0;transition:right .1s}
.band{position:absolute;top:0;bottom:0}
.val{font-family:var(--mono);font-size:12px;width:52px;text-align:right;font-variant-numeric:tabular-nums}
.cyc{display:flex;gap:6px;margin-top:11px;align-items:center}
.dot{width:13px;height:13px;border-radius:50%;border:1.5px solid var(--dim)}.dot.on{background:var(--good);border-color:var(--good)}
.list{display:grid;grid-template-columns:1fr 1fr;gap:5px}
.pt{display:flex;gap:6px;align-items:center;font-family:var(--mono);font-size:11px;color:var(--mut);padding:4px 6px;border-radius:6px}
.pt .m{width:8px;height:8px;border-radius:50%;border:1.5px solid var(--dim)}
.pt.active{color:var(--amber);background:#0c1526}.pt.active .m{background:var(--amber);border-color:var(--amber)}
.pt.done{color:var(--good)}.pt.done .m{background:var(--good);border-color:var(--good)}
.keys{display:grid;grid-template-columns:auto 1fr;gap:6px 10px;font-size:12px;color:var(--mut)}
kbd{font-family:var(--mono);font-size:11px;background:#0d1728;border:1px solid var(--edge2);border-bottom-width:2px;
 border-radius:5px;padding:2px 7px;color:var(--ink)}
.btns{display:flex;gap:8px}
button{flex:1;font-family:var(--mono);font-size:12px;color:var(--ink);background:#0d1728;border:1px solid var(--edge2);
 border-radius:8px;padding:9px;cursor:pointer}button:hover{border-color:var(--cyan);color:var(--cyan)}
button.rec{border-color:var(--crit);color:var(--crit)}button:focus-visible{outline:2px solid var(--cyan)}
</style>
<div class=wrap>
 <div class=top>
  <div><div class=title><b>Auscultation collection</b> · console</div>
   <div class=sub>point-conditioned teleop · doctor confirms by ear</div></div>
  <div id=rec class=sub>● IDLE</div>
 </div>
 <div class=grid>
  <div class=feed>
   <img src="/stream" alt="camera">
   <div class=tag>CAM · target overlay</div>
   <div class="recdot" id=recdot><i></i><span>IDLE</span></div>
   <div class=statusbar id=statusbar>connecting…</div>
  </div>
  <div class=side>
   <div class=card>
    <div class=lbl>Current target · <span id=counter>–</span></div>
    <div class=tgt><div class=tname id=tname>–</div><div class=pill id=pill style="color:var(--cyan)">–</div></div>
    <div class=row><div class=rl>Force</div><div class=bar><div class=band id=fband></div><span id=fbar></span></div><div class=val id=fval>–</div></div>
    <div class=row><div class=rl>Quality</div><div class=bar><div class=band id=qband></div><span id=qbar></span></div><div class=val id=qval>–</div></div>
    <div class=cyc><div class=rl>Cycles</div><div class=dot id=c0></div><div class=dot id=c1></div><div class=val id=cval style="width:auto;color:var(--mut)">0/2</div></div>
   </div>
   <div class=card><div class=lbl>Points · <span id=occ></span></div><div class=list id=plist></div></div>
   <div class=card>
    <div class=lbl>Controls</div>
    <div class=keys>
     <kbd>Space</kbd><span>confirm — heard it → next</span>
     <kbd>R</kbd><span>redo — keep trying here</span>
     <kbd>← →</kbd><span>prev / next point</span>
     <kbd>S</kbd><span>start / stop recording</span>
    </div>
    <div class=btns style=margin-top:12px>
     <button id=btnrec>● Start</button><button onclick="ctl('confirm')">✓ Confirm</button><button onclick="ctl('redo')">↻ Redo</button>
    </div>
   </div>
  </div>
 </div>
</div>
<script>
const $=id=>document.getElementById(id);
async function ctl(a){await fetch('/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:a})});poll();}
function plist(s){const L=$('plist');if(L.childElementCount!==s.n){L.innerHTML='';s.names.forEach((n,i)=>{const d=document.createElement('div');d.className='pt';d.id='p'+i;d.innerHTML='<span class=m></span>'+n;L.appendChild(d);});}
 s.names.forEach((_,i)=>{$('p'+i).className='pt'+(s.completed.includes(i)?' done':(i===s.idx?' active':''));});}
async function poll(){let s;try{s=await(await fetch('/state')).json();}catch(e){return;}if(!s.names)return;
 $('counter').textContent=(s.idx+1)+' / '+s.n;$('tname').textContent=s.point;
 const c={APPROACH:'--cyan',CONTACT:'--warn',LISTENING:'--good',READY:'--amber',DONE:'--good'}[s.state]||'--cyan';
 $('pill').textContent=s.state;$('pill').style.color='var('+c+')';$('statusbar').textContent=s.status;
 $('fband').style.cssText='left:37.5%;right:37.5%;background:rgba(52,211,153,.15);border-left:1px solid var(--good);border-right:1px solid var(--good)';
 $('fbar').style.right=(100-Math.min(1,s.force/8)*100)+'%';$('fbar').style.background=(s.force>=3&&s.force<=5)?'var(--good)':(s.force>5?'var(--crit)':'var(--warn)');$('fval').textContent=s.force.toFixed(1)+'N';
 $('qband').style.cssText='left:60%;right:0;background:rgba(56,189,248,.14);border-left:1px solid var(--cyan)';
 $('qbar').style.right=(100-Math.max(0,Math.min(1,s.q))*100)+'%';$('qbar').style.background=s.q>=0.6?'var(--cyan)':'var(--dim)';$('qval').textContent=s.q.toFixed(2);
 $('c0').className='dot'+(s.cycles>=1?' on':'');$('c1').className='dot'+(s.cycles>=2?' on':'');$('cval').textContent=Math.min(s.cycles,s.need)+'/'+s.need;
 $('occ').textContent=s.occluded?'point OCCLUDED — held':'point visible';
 const on=s.recording;$('rec').textContent=(on?'● REC '+s.elapsed+'s':'● IDLE');$('rec').style.color=on?'var(--crit)':'var(--dim)';
 $('recdot').className='recdot'+(on?' on':'');$('recdot').querySelector('span').textContent=on?('REC '+s.elapsed+'s'):'IDLE';
 $('btnrec').textContent=on?'■ Stop':'● Start';$('btnrec').className=on?'rec':'';
 plist(s);}
$('btnrec').onclick=()=>ctl($('btnrec').textContent[0]==='●'?'start':'stop');
addEventListener('keydown',e=>{const k=e.key;if(k===' '){e.preventDefault();ctl('confirm');}else if(k==='r'||k==='R')ctl('redo');
 else if(k==='ArrowRight')ctl('next');else if(k==='ArrowLeft')ctl('prev');else if(k==='s'||k==='S')ctl($('rec').textContent.includes('REC')?'stop':'start');
 else if(k==='f'||k==='F')document.querySelector('.feed').requestFullscreen?.();});
setInterval(poll,200);poll();
</script>""").encode()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true", help="synthetic camera + simulated sensors (no hardware)")
    ap.add_argument("--camera-backend", default="orbbec", help="orbbec (linbox) | avfoundation (webcam) | fake")
    ap.add_argument("--camera-index", type=int, default=0, help="UVC device index (Mac built-in cam is often 1)")
    ap.add_argument("--targets", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--port", type=int, default=8800)
    a = ap.parse_args()
    col = Collector(load_targets(a.targets), fake=a.fake, camera_backend=a.camera_backend,
                    camera_index=a.camera_index, out_dir=a.out)
    src = "FAKE" if a.fake else f"{a.camera_backend}[{a.camera_index}]"
    print(f"[collect] console at http://localhost:{a.port}  ({src})  keys: SPACE=confirm R=redo ←→=points S=rec")
    print(f"[collect] remote browser: on your laptop run  ssh -L {a.port}:localhost:{a.port} <this-host>  "
          f"then open http://localhost:{a.port}")
    ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(col, PAGE)).serve_forever()
