"""Encode an actual completed Isaac camera ledger on Linux CPU; never render a scene."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frame_schedule(rows, fps=20.):
    """Sample the latest acquired frame at each video time, preserving simulator speed."""
    if not rows or fps != 20.:
        raise ValueError('Actual nonempty 20 Hz camera ledger required')
    times = np.asarray([row['time_s'] for row in rows],float)
    if not np.isfinite(times).all() or np.any(np.diff(times)<-1e-9):
        raise ValueError('Camera clock is nonfinite or goes backwards')
    if any(row['index'] != i for i,row in enumerate(rows)):
        raise ValueError('Nonconsecutive camera ledger')
    clock = times[0]+np.arange(int(np.floor((times[-1]-times[0])*fps+1e-7))+1)/fps
    indices = np.searchsorted(times,clock+1e-9,side='right')-1
    return clock,indices


def video_identity(owner, case_name, result, outcome):
    if owner.get('completed') is not True or owner.get('gpu_released') is not True:
        raise ValueError('Wait for actual guarded execution completion and GPU release')
    if result.get('execution_completed') is not True:
        raise ValueError('Actual physical execution and cleanup required')
    experiment = owner.get('config',{}).get('experiment')
    if experiment == 'recovery_run' and case_name in ('nominal','joint3-plus','joint3-minus','act609','history20','entry-normal-plus','entry-tangent-plus'):
        if (outcome.get('expert_demonstration') is not True or outcome.get('learned_policy') is not False
                or outcome.get('training_dataset_eligible') is not False):
            raise ValueError('Expert recovery video ownership is ambiguous')
        if case_name in ('act609','history20'):
            moving=outcome.get('moving_recovery',{})
            if (owner['config'].get('mode')!='moving_oracle_pilot'
                    or moving.get('student_inference_executed') is not False
                    or moving.get('source_prefix_replayed') is not True):
                raise ValueError('Moving recovery replay ownership is ambiguous')
            if case_name=='history20':
                from .moving_origin import selected_origin
                index,phase,name=selected_origin(owner['config'])
                if name!=case_name or moving.get('source_action_index')!=index or moving.get('source_phase')!=phase:
                    raise ValueError('Moving history video does not bind the actual source origin')
            return dict(expert_demonstration=True,learned_policy=False,
                scope='Previously acquired ACT commands replayed for an actual moving expert recovery test; no new student inference')
        if case_name.startswith('entry-'):
            capture=outcome.get('capture_component',{})
            if (owner['config'].get('mode')!='capture_component'
                    or capture.get('local_alignment_used') is not False or capture.get('physical_joint_resets')!=0):
                raise ValueError('Direct contact component ownership is ambiguous')
            return dict(expert_demonstration=True,learned_policy=False,
                scope='Actual perturbed direct contact component; approved preparation and analytic L3, no learned policy or local alignment')
        return dict(expert_demonstration=True,learned_policy=False,
            scope='Actual stationary expert recovery pilot camera frames; not a learned policy or qualified dataset')
    if (experiment == 'isaac_run' and case_name == 'case'
            and result.get('closed_loop_evaluated') is True and outcome.get('expert_demonstration') is False):
        return dict(expert_demonstration=False,learned_policy=True,
            scope='Actual ACT D0 Isaac camera frames; no expert substitution')
    raise ValueError('Case name and actual controller ownership do not match')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--case',default='case',choices=('case','nominal','joint3-plus','joint3-minus','act609','history20','entry-normal-plus','entry-tangent-plus'))
    args=parser.parse_args()
    if sys.platform != 'linux':raise RuntimeError('Encode existing sensor pixels on Linux, not Mac')
    run=args.run.resolve();owner=json.loads((run/'result.json').read_text())
    case=run/'output'/args.case;result=json.loads((case/'result.json').read_text())
    identity=video_identity(owner,args.case,result,json.loads((case/'outcome.json').read_text()))
    ffmpeg=Path('/usr/bin/ffmpeg')
    if sha(ffmpeg)!='ed16af623947494a72e284b6eb8ff225f2da22b38b5d5069c2fd4b4ba3384e41':
        raise ValueError('Original verified ffmpeg changed')
    frames=case/'sensors/frames.jsonl'
    rows=[json.loads(line) for line in frames.read_text().splitlines()]
    clock,indices=frame_schedule(rows)
    camera=json.loads((case/'sensors/camera.json').read_text())
    width,height=camera['width'],camera['height']
    stem='review' if args.case=='case' else 'review-'+args.case
    destination=run/'output'/(stem+'.mp4')
    if destination.exists():raise ValueError('Never overwrite experiment media')
    command=[str(ffmpeg),'-hide_banner','-loglevel','error','-n','-f','rawvideo','-pixel_format','rgb24',
        '-video_size',f'{width}x{height}','-framerate','20','-i','pipe:0','-an','-c:v','libx264',
        '-threads','2','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(destination)]
    processed={};previous=None;pixels=None
    with subprocess.Popen(command,stdin=subprocess.PIPE) as process:
        try:
            for index in indices:
                row=rows[int(index)];name=row['raw_file']
                if Path(name).name!=name:raise ValueError('Unsafe raw sensor filename')
                if name!=previous:
                    source=case/'sensors'/name
                    if sha(source)!=row['raw_sha256']:raise ValueError('Raw RGB-D hash mismatch')
                    with np.load(source,allow_pickle=False) as arrays:rgb=arrays['rgb']
                    if rgb.dtype!=np.uint8 or rgb.shape!=(height,width,3):raise ValueError('Unexpected actual RGB pixels')
                    pixels=rgb.tobytes();previous=name;processed[name]=row['raw_sha256']
                process.stdin.write(pixels)
            process.stdin.close();status=process.wait(timeout=60)
            if status!=0:raise RuntimeError('ffmpeg failed')
        except Exception:
            process.terminate();process.wait(timeout=10);raise
    receipt=dict(schema='ausc_actual_camera_video_v1',video_sha256=sha(destination),
        frames_ledger_sha256=sha(frames),source_run=str(run),source_code_commit=owner['code_commit'],
        case_passed=result['case_passed'],m0_passed=False,case_id=args.case,**identity,
        video_frames=len(indices),fps=20,video_duration_s=len(indices)/20,
        simulated_first_frame_s=float(clock[0]),simulated_last_frame_s=float(clock[-1]),
        camera_rows=len(rows),unique_raw_frames=len(processed),ffmpeg_sha256=sha(ffmpeg),
        sampling='latest acquired RGB at each 50 ms video timestamp, same simulated speed; inference pauses omitted',
        audio='absent',realtime_claim=False,scene_rerendered=False)
    (run/'output'/(stem+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
