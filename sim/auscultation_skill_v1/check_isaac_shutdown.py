"""Minimal Linux SDK integration probe; no robot or scene is loaded."""
import argparse
import json
import os
from pathlib import Path
import time

from .run import graceful_simulation_shutdown


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--returncode',type=int,choices=(0,1),required=True)
    args=parser.parse_args(argv)
    if args.output.exists(): parser.error('probe output must be new')
    import isaacsim
    started=time.monotonic()
    with graceful_simulation_shutdown(isaacsim) as lifecycle:
        app=isaacsim.SimulationApp({'headless':True})
        for _ in range(3): app.update()
        app.close()
    if lifecycle['apps_created']!=1: raise RuntimeError('missing initialized app')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result=dict(schema='isaac_graceful_shutdown_probe_v1',close_returned=True,
                requested_returncode=args.returncode,elapsed_s=time.monotonic()-started,
                process_exit_policy='explicit os._exit after Kit close and saved receipt',
                scope='SDK close/return integration only; no physics qualification')
    with args.output.open('x') as stream:
        stream.write(json.dumps(result,indent=2)+'\n')
        stream.flush(); os.fsync(stream.fileno())
    print('GRACEFUL_CLOSE_RETURNED',args.returncode,flush=True)
    return args.returncode


if __name__=='__main__':
    from .runtime import exit_after_cleanup
    exit_after_cleanup(main())
