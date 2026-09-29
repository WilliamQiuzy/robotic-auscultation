"""Encode saved RGB sensor frames at their original simulation cadence."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def main():
    p=argparse.ArgumentParser();p.add_argument('--episode',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('New video output required')
    from imageio_ffmpeg import get_ffmpeg_exe
    frames=[json.loads(s) for s in (args.episode/'sensors/frames.jsonl').open()]
    physics=[json.loads(s) for s in (args.episode/'expert_trace.jsonl').open()]
    times=np.array([r['time_s'] for r in frames])
    if not np.allclose(np.diff(times),.05,atol=1e-7,rtol=0):raise ValueError('Broken 20-Hz frame cadence')
    metadata=json.loads((args.episode/'sensors/camera.json').read_text());w=metadata['width'];h=metadata['height']
    fontpath=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font=ImageFont.truetype(str(fontpath),15) if fontpath.exists() else ImageFont.load_default()
    height=h+88;args.output.parent.mkdir(parents=True,exist_ok=True)
    command=[get_ffmpeg_exe(),'-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{height}',
        '-r','20','-i','pipe:0','-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(args.output)]
    writer=subprocess.Popen(command,stdin=subprocess.PIPE)
    try:
        for i,frame in enumerate(frames):
            path=args.episode/'sensors'/frame['raw_file'];raw=path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=frame['raw_sha256']:raise ValueError('RGB-D source hash changed')
            with np.load(path,allow_pickle=False) as a:rgb=a['rgb']
            canvas=Image.new('RGB',(w,height),(20,25,34));canvas.paste(Image.fromarray(rgb),(0,0));draw=ImageDraw.Draw(canvas)
            row=physics[frame['physics_row']];force=sum(row[k] for k in ['contact_rim_N','lip_N','diaphragm_N'])
            draw.text((12,h+8),'REPAIRED EXPERT | L5 | 1x simulation time | NOT learned VLA',font=font,fill=(239,243,248))
            draw.text((12,h+33),f"t={frame['time_s']:6.2f}s   {frame['phase']}   Hall={frame['state'][20]:.3f}V",font=font,fill=(120,214,219))
            draw.text((12,h+58),f"PhysX front contact={force:.2f}N   Audio absent",font=font,fill=(239,243,248))
            writer.stdin.write(np.asarray(canvas).tobytes())
            if i==len(frames)//2:canvas.save(args.output.with_suffix('.png'))
        writer.stdin.close()
        if writer.wait()!=0:raise RuntimeError('Video encoding failed')
    finally:
        if writer.poll() is None:writer.terminate();writer.wait()
    receipt=dict(frames=len(frames),fps=20,duration_s=len(frames)/20,playback_rate=1,
        source='saved real Isaac RGB camera frames',controller='analytic repaired expert; not learned VLA',
        source_ledger_sha256=hashlib.sha256((args.episode/'sensors/frames.jsonl').read_bytes()).hexdigest(),
        video_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())
    args.output.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
