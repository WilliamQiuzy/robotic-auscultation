"""Compact comparison from completed raw-command replay diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args()
    rows=[]
    for report in sorted(args.root.glob('*/report.json')):
        data=json.loads(report.read_text());m=data['measured']
        rows.append(dict(name=report.parent.name,configuration=data['inputs'],
            forward_acceleration=m['forward']['acceleration_rad_s2']['max'],
            forward_jerk=m['forward']['jerk_rad_s3']['max'],
            centered_acceleration=m['centered']['acceleration_rad_s2']['max'],
            centered_jerk=m['centered']['jerk_rad_s3']['max'],
            stored_velocity_acceleration=m['stored_velocity']['acceleration_rad_s2']['max'],
            force_max_N=data['max_front_N'],forbidden_max_N=data['max_forbidden_N'],
            tracking_max_rad=data['tracking_error_rad'],
            report_sha256=hashlib.sha256(report.read_bytes()).hexdigest()))
        trace=report.parent/'replay_trace.jsonl';arrays=report.parent/'motion_arrays.npz'
        if not arrays.exists():
            raw=[json.loads(line) for line in trace.open()]
            np.savez_compressed(arrays,time_s=[r['time_s'] for r in raw],q=[r['q'] for r in raw],
                dq=[r['dq'] for r in raw],q_command=[r['q_command'] for r in raw],
                compression_m=[r['compression_m'] for r in raw],front_N=[r['contact_rim_N']+r['diaphragm_N']+r['lip_N'] for r in raw],
                phase=np.asarray([r['phase'] for r in raw]),trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest())
    result=dict(scope='identical command replays; not new expert demonstrations or policy execution',
        differentiated_positions='unfiltered 240 Hz samples',rows=rows)
    (args.root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    for row in rows:print(row['name'],round(row['centered_jerk'],3),round(row['force_max_N'],3),flush=True)


if __name__=='__main__':main()
