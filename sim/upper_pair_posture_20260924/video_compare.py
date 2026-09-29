"""Compose original-time upper-pair videos on Linux; never warp or phase-align time."""
import argparse
import json
import math
from pathlib import Path
import subprocess

from .contract import PROJECT, read, require, sha


def clip_length(manifest,point='R2'):
    frames=manifest['frames']
    require(manifest['fps']==15 and manifest['playback_speed']==1.,'only original 15 fps replay')
    hold=[f['frame_index'] for f in frames if f['target']==point and f['phase']=='HOLD']
    require(bool(hold),'comparison requires completed R2 HOLD')
    # Include one following frame, so the full HOLD is retained at CFR boundaries.
    return min(len(frames),max(hold)+2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True,help='render directory')
    parser.add_argument('--alternative',type=Path,required=True,help='render directory')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import sys
    require(sys.platform.startswith('linux'),'video composition is assigned to Linux')
    import cv2
    import numpy as np
    import imageio_ffmpeg
    manifests=[read(p/'review_manifest.json') for p in (args.baseline,args.alternative)]
    directories=[args.baseline,args.alternative]
    sources={}
    captures=[];counts=[]
    for directory,manifest in zip(directories,manifests):
        path=directory/manifest['video']
        require(sha(path)==manifest['video_sha256'],'source video hash mismatch')
        require(manifest['resolution']==[1920,720],'unexpected source resolution')
        counts.append(clip_length(manifest))
        sources[str(path.resolve().relative_to(PROJECT))]=sha(path)
        sources[str((directory/'review_manifest.json').resolve().relative_to(PROJECT))]=sha(directory/'review_manifest.json')
        cap=cv2.VideoCapture(str(path));require(cap.isOpened(),'video decoder unavailable')
        captures.append(cap)
    count=max(counts);fps=15
    require(not args.output.exists(),'comparison output must be new')
    args.output.mkdir(parents=True)
    destination=args.output/'upper_pair_comparison_1x.mp4'
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24',
             '-s','1920x720','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','19',
             '-pix_fmt','yuv420p','-movflags','+faststart',str(destination)]
    encoder=subprocess.Popen(command,stdin=subprocess.PIPE)
    last=[None,None];sampled=set();sample_paths=[]
    try:
        for index in range(count):
            canvas=np.full((720,1920,3),(30,26,22),np.uint8)
            keys=[]
            for panel,(cap,manifest,length) in enumerate(zip(captures,manifests,counts)):
                paused=index>=length
                if not paused:
                    ok,frame=cap.read();require(ok,'source decode failed before clip endpoint');last[panel]=frame
                frame=last[panel];require(frame is not None,'empty panel')
                x=panel*960
                canvas[:540,x:x+960]=cv2.resize(frame[:720,:1280],(960,540),interpolation=cv2.INTER_AREA)
                canvas[540:,x:x+320]=cv2.resize(frame[:360,1280:],(320,180),interpolation=cv2.INTER_AREA)
                metadata=manifest['frames'][min(index,length-1)]
                title='CURRENT B' if panel==0 else 'ALTERNATIVE'
                cv2.rectangle(canvas,(x,0),(x+960,38),(20,20,20),-1)
                cv2.putText(canvas,title+' | ORIGINAL SPEED 1x',(x+18,27),cv2.FONT_HERSHEY_SIMPLEX,.7,(245,245,245),1,cv2.LINE_AA)
                labels=[f"Time {min(index,length-1)/fps:.2f} s | {metadata['target']} {metadata['phase']}",
                        str(metadata['subphase']),
                        'R2 HOLD completed: panel PAUSED' if paused else 'Continuous saved physics poses',
                        'Same approved camera | no audio',
                        'Engineering posture comparison']
                for line,text in enumerate(labels):
                    cv2.putText(canvas,text,(x+338,566+31*line),cv2.FONT_HERSHEY_SIMPLEX,.57,
                                (30,200,255) if paused and line==2 else (235,235,235),1,cv2.LINE_AA)
                keys.append((metadata['target'],metadata['phase'],metadata['subphase'],paused))
            encoder.stdin.write(canvas.tobytes())
            key=tuple(keys)
            if key not in sampled or index==count-1:
                sampled.add(key);path=args.output/f'comparison_{index:05d}.jpg'
                require(cv2.imwrite(str(path),canvas),'snapshot failed');sample_paths.append(path.name)
        encoder.stdin.close();require(encoder.wait()==0,'comparison encode failed')
        encoder=None
    finally:
        for cap in captures:cap.release()
        if encoder is not None:
            if encoder.stdin and not encoder.stdin.closed:encoder.stdin.close()
            encoder.wait()
    # Independent full decode catches truncation, dimensions and frame count.
    cap=cv2.VideoCapture(str(destination));decoded=0
    require(abs(cap.get(cv2.CAP_PROP_FPS)-fps)<1e-8,'output fps changed')
    while True:
        ok,frame=cap.read()
        if not ok:break
        require(frame.shape==(720,1920,3),'bad output frame shape');decoded+=1
    cap.release();require(decoded==count,'output frame count mismatch')
    for rel,digest in sources.items():require(sha(PROJECT/rel)==digest,'render source changed')
    receipt={'schema':'upper_pair_side_by_side_original_time_v1','video':destination.name,
        'sha256':sha(destination),'source_sha256':sources,'fps':fps,'frames':count,'duration_s':count/fps,
        'panel_sources':[{'role':role,'manifest':str((directory/'review_manifest.json').resolve().relative_to(PROJECT)),
                          'video':str((directory/manifest['video']).resolve().relative_to(PROJECT))}
                         for role,directory,manifest in zip(('baseline','alternative'),directories,manifests)],
        'panel_frames':counts,'panel_duration_s':[n/fps for n in counts],
        'frozen_panel_padding_s':[(count-n)/fps for n in counts],'decoded_frames':decoded,
        'review_samples':sample_paths,'scope':'Same global clock from episode start, no phase alignment or time warping. '
        'Each panel includes full R2 HOLD and at most one subsequent video frame; earlier-finished panel explicitly pauses. '
        'Saved actual body poses, original-time RGB replay; no acoustic or clinical acceptance.'}
    (args.output/'comparison_video_manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('source_sha256','review_samples')}),flush=True)


if __name__=='__main__':main()
