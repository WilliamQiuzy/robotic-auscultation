"""Finite convergence/sensitivity check. No parameter is accepted by this runner."""
import fcntl
import json
from pathlib import Path
import subprocess


def main():
    root=Path('/home/robot/wam-sim-media/contact-repair-v3')
    source=Path('/home/robot/wam-sim-media/hall-modality-v1/results/expert_rgbd_L3_v1')
    launch=Path('/home/robot/wam-sim-media/hall-modality-v1/project/sim/auscultation_repair_v3/run_probe.sh')
    jobs=[('L3_velocity0',['--velocity-iterations','0']),
          ('L3_armature001_solver16',['--velocity-iterations','1','--armature','0.01','--solver','16']),
          ('L3_velocity0_armature0001',['--velocity-iterations','0','--armature','0.001'])]
    receipt=[]
    for name,options in jobs:
        with Path('/home/robot/wam-sim-media/skill_v4/isaac.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            fcntl.flock(lock,fcntl.LOCK_UN)
        command=['bash',str(launch),'--trace',str(source/'expert_trace.jsonl'),'--scene',str(source/'training_scene.usdc'),
                 '--output',str(root/name),*options]
        with (root/(name+'.log')).open('x') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        receipt.append(dict(name=name,command=command,returncode=result.returncode))
        (root/'solver_sweep.json').write_text(json.dumps(receipt,indent=2)+'\n')
        if result.returncode:raise RuntimeError('Diagnostic failed: '+name)


if __name__=='__main__':main()
