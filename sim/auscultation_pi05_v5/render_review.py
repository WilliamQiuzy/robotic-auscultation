"""Encode recorded RGB camera frames using simulation timestamps, never rerender."""
import argparse
from bisect import bisect_right
import json
import hashlib
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def main():
    p=argparse.ArgumentParser();p.add_argument('--episode',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--ffmpeg',required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('New video path required')
    root=a.episode;rows=[json.loads(x) for x in (root/'sensors/frames.jsonl').read_text().splitlines()]
    trace=[json.loads(x) for x in (root/'expert_trace.jsonl').read_text().splitlines()]
    outcome=json.loads((root/'outcome.json').read_text())
    acceptance=json.loads((root/'collection_result.json').read_text())
    if not rows:raise ValueError('No acquired camera frames')
    stamps=[r['time_s'] for r in rows]
    if np.any(np.diff(stamps)<0):raise ValueError('Camera times out of order')
    rate=20;start=stamps[0];stop=stamps[-1]
    with np.load(root/'sensors'/rows[0]['raw_file']) as z:first=z['rgb']
    h,w=first.shape[:2];bar=96
    command=[a.ffmpeg,'-hide_banner','-loglevel','error','-f','rawvideo','-pixel_format','rgb24',
        '-video_size',f'{w}x{h+bar}','-framerate',str(rate),'-i','-','-an','-c:v','libx264',
        '-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(a.output)]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    cached=None;rgb=None;receipt=[]
    with subprocess.Popen(command,stdin=subprocess.PIPE) as process:
        for i,t in enumerate(np.arange(start,stop+1/(2*rate),1/rate)):
            idx=max(0,bisect_right(stamps,float(t))-1);r=rows[idx]
            path=root/'sensors'/r['raw_file']
            if cached!=path:
                if hashlib.sha256(path.read_bytes()).hexdigest()!=r['raw_sha256']:
                    raise ValueError('Recorded camera file changed')
                with np.load(path,allow_pickle=False) as z:rgb=z['rgb'].copy()
                cached=path
            canvas=Image.new('RGB',(w,h+bar),'#101820');canvas.paste(Image.fromarray(rgb),(0,bar))
            d=ImageDraw.Draw(canvas);physics=trace[r['physics_row']]
            binding=outcome.get('learned_policy_binding',{})
            d.text((8,5),f"pi0.5 | {outcome.get('task_scope','unknown')} scope | checkpoint {binding.get('checkpoint_step','?')} | {'PASS' if acceptance['passed'] else 'FAILED TRIAL'}",font=font,fill='white')
            d.text((8,25),f"SIM {t:.2f}s | {r['target']} | {r['subphase']} | {r['phase']}",font=font,fill='white')
            d.text((8,45),'Recorded RGB-D camera; inference pauses simulation; audio absent',font=font,fill='#aaddff')
            d.text((8,65),f"No expert fallback | hold {physics.get('audit_hold_s',0):.2f}s | recorded physics row {r['physics_row']}",font=font,fill='#aaddff')
            process.stdin.write(np.asarray(canvas).tobytes())
            receipt.append(dict(video_frame=i,simulation_time_s=float(t),source_observation=idx,raw_sha256=r['raw_sha256']))
        process.stdin.close()
        if process.wait():raise RuntimeError('Video encoder failed')
    a.output.with_suffix('.frames.json').write_text(json.dumps(dict(fps=rate,source=str(root),
        resampling='previous acquired frame at uniform simulation timestamps, latest same-time observation',frames=receipt))+'\n')
    print(json.dumps(dict(video=str(a.output),frames=len(receipt),duration_s=len(receipt)/rate)))


if __name__=='__main__':main()
