"""Bounded cleanup for a child process owned by this simulation invocation."""
import subprocess


def exit_after_cleanup(returncode):
    """Exit an owned worker after verified Kit close and saved final evidence.

    Isaac's Python finalizers may hang even when SimulationApp.close returned.
    The parent still verifies this actual OS code and the post-close receipt.
    Never call this from an exception path or before worker cleanup returns.
    """
    import os
    import sys
    if type(returncode) is not int or returncode not in (0,1):
        raise ValueError('post-cleanup exit requires an explicit task code 0 or 1')
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(returncode)
    raise RuntimeError('OS worker exit unexpectedly returned')


def stop_process_tree(child):
    import psutil
    try: descendants=psutil.Process(child.pid).children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError): descendants=[]
    for process in reversed(descendants):
        try: process.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError): pass
    if child.poll() is None: child.terminate()
    _,alive=psutil.wait_procs(descendants,timeout=5.)
    for process in alive:
        try: process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError): pass
    try: child.wait(timeout=5.)
    except subprocess.TimeoutExpired: child.kill(); child.wait()
