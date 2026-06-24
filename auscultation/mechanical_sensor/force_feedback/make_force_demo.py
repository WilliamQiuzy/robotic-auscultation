"""Generate a self-contained side-by-side demo webpage for the report:
left = the real-world video of the magnet approaching the Hall sensor,
right = a synced canvas animation of the measured magnetic signal / force.

    python make_force_demo.py            # reads generated/force_feedback/demo_capture.json
                                         # writes  generated/force_feedback/force_demo.html

Drop your phone clip in next to force_demo.html as `real_world.mp4` (or pass
--video NAME) and open force_demo.html in a browser; press the play button to
run both together.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parent / "generated" / "force_feedback" / "demo_capture.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--video", default="real_world.mp4")
    ap.add_argument("--out", default=str(DEFAULT_DATA.parent / "force_demo.html"))
    ap.add_argument("--supply-v", type=float, default=3.3)
    args = ap.parse_args()

    blob = json.loads(Path(args.data).read_text())
    samples = blob["samples"]
    raws = [s["raw"] for s in samples]
    rmin, rmax = min(raws), max(raws)
    dt = blob.get("dt", 0.25)
    # normalize 0 (far / no field) .. 1 (near / strong field)
    series = [
        {"t": round(i * dt, 3), "raw": r, "n": round((r - rmin) / (rmax - rmin), 4),
         "v": round(r / 65535 * args.supply_v, 4)}
        for i, r in enumerate(raws)
    ]
    duration = round((len(series) - 1) * dt, 2)
    data_js = json.dumps(series)

    html = _TEMPLATE.format(
        data=data_js, video=args.video, duration=duration,
        rmin=rmin, rmax=rmax, dt=dt,
    )
    out = Path(args.out)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({len(series)} samples, {duration}s)")
    print(f"  put your clip next to it as '{args.video}', then open in a browser")
    return 0


_TEMPLATE = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>磁力反馈演示 / Magnetic Force Feedback Demo</title>
<style>
  body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:#0f1117;color:#e7e9ee;margin:0;padding:24px}}
  h1{{font-size:20px;font-weight:600;margin:0 0 4px}} .sub{{color:#9aa3b2;font-size:13px;margin-bottom:18px}}
  .row{{display:flex;gap:20px;flex-wrap:wrap;align-items:flex-start}}
  .panel{{background:#171a23;border:1px solid #232838;border-radius:12px;padding:12px;width:560px}}
  .cap{{font-size:13px;color:#aab2c2;margin:8px 2px 0}}
  video{{width:536px;height:360px;background:#000;border-radius:8px;object-fit:contain}}
  canvas{{width:536px;height:360px;background:#0b0d13;border-radius:8px}}
  .bar{{display:flex;align-items:center;gap:14px;margin-top:16px}}
  button{{background:#3b82f6;color:#fff;border:0;border-radius:8px;padding:10px 20px;font-size:15px;cursor:pointer}}
  button:hover{{background:#2f6fd6}}
  .ph{{width:536px;height:360px;display:flex;align-items:center;justify-content:center;color:#5a6478;
       border:1px dashed #2a3146;border-radius:8px;text-align:center;font-size:13px;padding:0 30px;box-sizing:border-box}}
  .meta{{color:#6b7488;font-size:12px;margin-top:14px}}
</style></head>
<body>
  <h1>磁力反馈：磁铁靠近 → Hall 信号 → 接触力</h1>
  <div class="sub">左：现实世界（磁铁靠近 Hall 传感器）　右：实测信号同步可视化　·　数据 {duration}s，raw {rmin}–{rmax}</div>
  <div class="row">
    <div class="panel">
      <video id="vid" controls playsinline></video>
      <div id="vidph" class="ph">把你拍的视频命名为 <b>&nbsp;{video}&nbsp;</b> 放在本网页旁边即可显示<br>（现实世界：磁铁靠近 Hall 传感器）</div>
      <div class="cap">① 现实世界视频</div>
    </div>
    <div class="panel">
      <canvas id="cv" width="536" height="360"></canvas>
      <div class="cap">② 传感器实测：磁场信号 → 力（越靠近，场越强，读数越高）</div>
    </div>
  </div>
  <div class="bar">
    <button id="play">▶ 一起播放</button>
    <span id="status" class="meta">就绪</span>
  </div>
  <div class="meta">机制：磁铁越靠近 Hall，气隙越小、磁场越强、ADC 读数越高 —— 在装配后的探头里，这个读数就对应接触力（标定后映射到牛顿）。</div>

<script>
const DATA = {data};
const DT = {dt}, DUR = {duration};
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const vid = document.getElementById('vid'), vidph = document.getElementById('vidph');
const status = document.getElementById('status');
// show the real video if the file exists
vid.src = "{video}";
vid.addEventListener('loadeddata', () => {{ vid.style.display='block'; vidph.style.display='none'; }});
vid.addEventListener('error', () => {{ vid.style.display='none'; vidph.style.display='flex'; }});
vid.style.display='none';

function sampleAt(t){{
  if (t<=0) return DATA[0];
  const f = t/DT, i = Math.min(DATA.length-1, Math.floor(f));
  const a = DATA[i], b = DATA[Math.min(DATA.length-1,i+1)], u=f-i;
  return {{n:a.n+(b.n-a.n)*u, v:a.v+(b.v-a.v)*u, raw:Math.round(a.raw+(b.raw-a.raw)*u)}};
}}

function draw(t){{
  const W=536,H=360; ctx.clearRect(0,0,W,H);
  const s = sampleAt(t), n=s.n;
  // ---- schematic: magnet (left, moving) -> Hall (right, fixed) ----
  const hallX=430, hallY=92, magY=92;
  const magX = 120 + n*210;                       // closer when n high
  // field glow between magnet and hall
  const g = ctx.createRadialGradient(hallX-14,hallY+18,4, hallX-14,hallY+18,140);
  g.addColorStop(0,`rgba(80,170,255,${{0.10+0.55*n}})`); g.addColorStop(1,'rgba(80,170,255,0)');
  ctx.fillStyle=g; ctx.fillRect(magX-30,10,260,150);
  // field arcs
  ctx.strokeStyle=`rgba(120,190,255,${{0.25+0.6*n}})`; ctx.lineWidth=1.5;
  for(let k=1;k<=4;k++){{ ctx.beginPath();
    ctx.ellipse((magX+58+hallX)/2, hallY+18, (hallX-magX-58)/2*k/4+12, 16*k, 0, -Math.PI/2, Math.PI/2);
    ctx.stroke(); }}
  // magnet (N red / S blue)
  ctx.fillStyle='#e23b3b'; ctx.fillRect(magX,hallY-14,29,30);
  ctx.fillStyle='#3b6fe2'; ctx.fillRect(magX+29,hallY-14,29,30);
  ctx.fillStyle='#fff'; ctx.font='bold 13px sans-serif'; ctx.textAlign='center';
  ctx.fillText('N',magX+14,hallY+6); ctx.fillText('S',magX+43,hallY+6);
  ctx.fillStyle='#9aa3b2'; ctx.font='12px sans-serif'; ctx.fillText('磁铁',magX+29,hallY+34);
  // Hall sensor
  ctx.fillStyle='#2b2f3a'; ctx.strokeStyle='#6a7488'; ctx.lineWidth=2;
  ctx.fillRect(hallX,hallY-22,34,44); ctx.strokeRect(hallX,hallY-22,34,44);
  ctx.fillStyle='#cfd6e4'; ctx.fillText('Hall',hallX+17,hallY+4); ctx.fillText('SS49E',hallX+17,hallY+38);
  // ---- readouts ----
  ctx.textAlign='left'; ctx.fillStyle='#e7e9ee'; ctx.font='bold 26px sans-serif';
  ctx.fillText((n*100).toFixed(0)+'%', 24, 56);
  ctx.font='12px sans-serif'; ctx.fillStyle='#9aa3b2';
  ctx.fillText('磁场/力 信号', 24, 26);
  ctx.fillText('raw '+s.raw+'   '+s.v.toFixed(3)+' V', 24, 76);
  // ---- signal-over-time plot ----
  const px=24,py=200,pw=488,ph=132;
  ctx.strokeStyle='#222838'; ctx.lineWidth=1; ctx.strokeRect(px,py,pw,ph);
  ctx.fillStyle='#8b94a6'; ctx.font='11px sans-serif';
  ctx.fillText('信号随时间（越高=磁铁越近=力越大）',px+4,py-6);
  const X=i=>px+(DATA[i].t/DUR)*pw, Y=v=>py+ph-v*(ph-10)-5;
  ctx.beginPath(); for(let i=0;i<DATA.length;i++){{ const x=X(i),y=Y(DATA[i].n); i?ctx.lineTo(x,y):ctx.moveTo(x,y);}}
  ctx.strokeStyle='#3b82f6'; ctx.lineWidth=2; ctx.stroke();
  // filled area up to t
  const cx = px+(Math.min(t,DUR)/DUR)*pw;
  ctx.save(); ctx.beginPath(); ctx.rect(px,py,cx-px,ph); ctx.clip();
  ctx.beginPath(); ctx.moveTo(px,py+ph);
  for(let i=0;i<DATA.length;i++) ctx.lineTo(X(i),Y(DATA[i].n));
  ctx.lineTo(cx,py+ph); ctx.closePath(); ctx.fillStyle='rgba(59,130,246,0.18)'; ctx.fill(); ctx.restore();
  // playhead
  ctx.beginPath(); ctx.moveTo(cx,py); ctx.lineTo(cx,py+ph); ctx.strokeStyle='#e7e9ee'; ctx.lineWidth=1; ctx.stroke();
  ctx.beginPath(); ctx.arc(cx,Y(n),4,0,7); ctx.fillStyle='#fff'; ctx.fill();
}}

let raf=null, t0=0;
function loop(ts){{
  if(!t0) t0=ts; const t=(ts-t0)/1000;
  draw(t); status.textContent='播放中 '+t.toFixed(1)+' / '+DUR+' s';
  if(t<DUR) raf=requestAnimationFrame(loop);
  else status.textContent='完成';
}}
document.getElementById('play').onclick=()=>{{
  if(raf) cancelAnimationFrame(raf); t0=0;
  if(vid.style.display!=='none'){{ try{{vid.currentTime=0; vid.play();}}catch(e){{}} }}
  raf=requestAnimationFrame(loop);
}};
draw(0);
</script>
</body></html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
