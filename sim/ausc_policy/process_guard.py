"""Versioned WAM process guard with bounded exit-time GPU identity checks.

Original source SHA256: 48c47cf0d6d8f3d770028931a61d2882c5281684053a6149dacfcbe7b2483ff2.
The immutable diagnostics copy and historic run receipts are not modified.
No flock: the unchanged trainer child owns the shared GPU lock.
Only this guard's direct child and verified group can receive signals.
"""
from datetime import datetime,timezone
import hashlib,json,os,signal,shutil,subprocess,time
from pathlib import Path
GIB=2**30
STAGE_LIMIT_S=12*3600
POLL_S=10
EXIT_GPU_GRACE_S=15.
GUARD=dict(minimum_ram_available_bytes=2*GIB,minimum_disk_free_bytes=2*GIB,minimum_gpu_free_bytes=GIB,
 stage_timeout_s=STAGE_LIMIT_S,sample_period_s=POLL_S,interrupt_grace_s=120,terminate_grace_s=60,
 kill_grace_s=30,process_group_drain_s=60,gpu_release_s=60)
def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*2**20), b''): h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def parse_proc_stat(pid, raw):
    fields = raw.rsplit(')', 1)[1].split()
    return dict(pid=pid, state=fields[0], pgid=int(fields[2]), sid=int(fields[3]), start_ticks=fields[19])


def process_identity(pid):
    return parse_proc_stat(pid, (Path('/proc')/str(pid)/'stat').read_text())


def group_members(identity):
    members = []
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal(): continue
        try: member = process_identity(int(path.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError): continue
        if member['pgid'] == identity['pid'] and member['sid'] == identity['pid'] and member['state'] != 'Z':
            members.append(member)
    return members


def remember_group(identity):
    members = group_members(identity)
    known=identity.setdefault('members', {})
    known.setdefault(str(identity['pid']), identity['start_ticks'])
    for member in members:
        # Keep first-observed identity: replacing it would erase PID reuse.
        known.setdefault(str(member['pid']), member['start_ticks'])
    return members


def signal_owned_group(process, identity, sig):
    """Reject a recycled leader; after its exit require a known live descendant."""
    try: current = process_identity(process.pid)
    except (FileNotFoundError, ProcessLookupError): current = None
    if current is not None and current['start_ticks'] != identity['start_ticks']:
        raise ValueError('Child process identity changed; refusing group signal')
    if current is not None:
        if current.get('pgid') != process.pid or current.get('sid') != process.pid:
            raise ValueError('Child group/session identity changed')
    else:
        members = group_members(identity)
        if not members: return
        if not any(identity.get('members', {}).get(str(p['pid'])) == p['start_ticks'] for p in members):
            raise ValueError('No known child identity remains; refusing group signal')
    try: os.killpg(process.pid, sig)
    except ProcessLookupError: pass


def gpu_pids():
    data = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], text=True, timeout=15)
    return {int(x.strip()) for x in data.splitlines() if x.strip()}


def classify_gpu_ownership(identity, pids, sampled_monotonic):
    """Recheck GPU PIDs after the GPU query, not against an older /proc list.

    Only previously observed identities may receive a bounded disappearance
    grace. Permission errors propagate. Recycled or escaped live PIDs never
    inherit ownership, including zombies; GPU disappearance does not reset
    the original deadline. This is not a cross-project reservation mechanism.
    """
    known=identity.setdefault('members', {})
    deadlines=identity.setdefault('gpu_exit_first_seen_monotonic', {})
    result=dict(verified_owned_pids=[], exiting_owned_pids=[], rejected_pids=[],
                records=[], exit_grace_s=EXIT_GPU_GRACE_S)
    for pid in sorted(pids):
        key=str(pid); expected=known.get(key)
        try:current=process_identity(pid)
        except (FileNotFoundError,ProcessLookupError):current=None
        same_group=(current is not None and current['pgid']==identity['pid']
                    and current['sid']==identity['pid'])
        same_identity=(current is not None and expected==current['start_ticks'])
        if current is not None and same_group and current['state']!='Z' and (expected is None or same_identity):
            known[key]=current['start_ticks']
            bucket='verified_owned_pids';reason='live same-session identity'
            age=None
        elif expected is not None and (current is None or (same_identity and same_group and current['state']=='Z')):
            first=deadlines.setdefault(key, sampled_monotonic)
            age=sampled_monotonic-first
            bucket='exiting_owned_pids' if 0<=age<EXIT_GPU_GRACE_S else 'rejected_pids'
            reason='known exit within bounded grace' if bucket=='exiting_owned_pids' else 'known exit grace expired'
        else:
            bucket='rejected_pids';reason='unknown, recycled or outside original session';age=None
        result[bucket].append(pid)
        result['records'].append(dict(pid=pid, expected_start_ticks=expected,
            observed_identity=current, classification=bucket, reason=reason, exit_age_s=age))
    return result


def sample_resources(output, identity):
    result = dict(sampled_utc=now(), sampling_error=None)
    try:
        members = remember_group(identity)
        result['owned_group_pids'] = [p['pid'] for p in members]
        memory=[]
        for member in members:
            try:
                fields={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in
                        (Path('/proc')/str(member['pid'])/'status').read_text().splitlines() if ':' in line}
                memory.append(dict(pid=member['pid'],start_ticks=member['start_ticks'],
                    rss_bytes=int(fields.get('VmRSS','0 kB').split()[0])*1024,
                    hwm_bytes=int(fields.get('VmHWM','0 kB').split()[0])*1024))
            except FileNotFoundError:pass
        result['owned_process_memory']=memory
        result['owned_group_rss_bytes']=sum(row['rss_bytes'] for row in memory)
        result['gpu_pids'] = sorted(gpu_pids())
        result['gpu_ownership'] = classify_gpu_ownership(identity, result['gpu_pids'], time.monotonic())
        free = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
            '--format=csv,noheader,nounits'], text=True, timeout=15).splitlines()
        if len(free) != 1: raise ValueError('Expected exactly one shared GPU')
        result['gpu_free_bytes'] = int(free[0])*2**20
        result['ram_available_bytes'] = next(int(line.split()[1])*1024 for line in
            Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
        result['disk_free_bytes'] = shutil.disk_usage(Path(output).parent).free
    except Exception as exc:
        result['sampling_error'] = type(exc).__name__+': '+str(exc)
    return result


def guard_violation(sample, elapsed_s, maximum_s):
    if elapsed_s >= maximum_s: return 'Stage wall-clock limit reached'
    if sample.get('sampling_error'): return 'Resource sampling failed: '+sample['sampling_error']
    for field, minimum in [('ram_available_bytes', 2*GIB), ('disk_free_bytes', 2*GIB), ('gpu_free_bytes', GIB)]:
        if sample.get(field, -1) < minimum: return field+' below emergency margin'
    ownership=sample.get('gpu_ownership')
    allowed=(set(ownership.get('verified_owned_pids',[])) | set(ownership.get('exiting_owned_pids',[]))
             if ownership is not None else set(sample.get('owned_group_pids',[])))
    if ((ownership is not None and ownership.get('rejected_pids'))
            or set(sample.get('gpu_pids', []))-allowed):
        return 'GPU compute process outside owned child group'
    return None


def drain(process, identity, seconds):
    deadline = time.monotonic()+seconds
    while True:
        exited = process.poll() is not None
        members = remember_group(identity)
        if exited and not members: return True
        if time.monotonic() >= deadline: return False
        time.sleep(1)


def stop_owned(process, identity, events):
    for sig, seconds in [(signal.SIGINT, 120), (signal.SIGTERM, 60), (signal.SIGKILL, 30)]:
        if process.poll() is not None and not remember_group(identity): return True
        events.append(dict(signal=sig.name, sent_utc=now(), grace_s=seconds))
        signal_owned_group(process, identity, sig)
        try:
            if drain(process, identity, seconds): return True
        except KeyboardInterrupt:
            events.append(dict(interrupt_during_shutdown=True, received_utc=now()))
    return False


def run_stage(name, command, args, state, env=None, pass_fds=()):
    """One new session, continuous guard, bounded graceful shutdown, no retry."""
    result = dict(stage=name, command=command, started_utc=now(), signals=[], deferred_signals=[], errors=[])
    with (args.evidence/(name+'.log')).open('xb') as log, (args.evidence/(name+'.resources.jsonl')).open('x') as resources:
        process = None; identity = None
        prior_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        def deferred(signum, frame):
            result['deferred_signals'].append(dict(signal=signal.Signals(signum).name, received_utc=now()))
        def set_handlers(defer):
            for sig, handler in prior_handlers.items(): signal.signal(sig, deferred if defer else handler)
        def valid_session(value):
            return value.get('pid') == process.pid and value.get('pgid') == process.pid and value.get('sid') == process.pid
        try:
            # Defer parent interrupts through Popen's return/assignment and
            # initial identity capture. Do not mask signals: an exec'd child
            # resets caught handlers and must retain its normal SIGINT handling.
            set_handlers(True)
            process = subprocess.Popen(command, cwd=state['runtime_project'], env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=pass_fds)
            identity = process_identity(process.pid)
            result['process_identity'] = identity
            if not valid_session(identity): raise ValueError('New child session was not established')
            identity['members'] = {str(identity['pid']): identity['start_ticks']}
            set_handlers(False)
            if result['deferred_signals']: raise KeyboardInterrupt('Interrupted during child startup')
            started = time.monotonic()
            while process.poll() is None:
                sample = sample_resources(args.output, identity)
                sample['elapsed_s'] = time.monotonic()-started
                resources.write(json.dumps(sample, allow_nan=False)+'\n'); resources.flush()
                violation = guard_violation(sample, sample['elapsed_s'], STAGE_LIMIT_S)
                if violation:
                    result['errors'].append(violation); break
                try: process.wait(timeout=POLL_S)
                except subprocess.TimeoutExpired: pass
            if not result['errors'] and not drain(process, identity, GUARD['process_group_drain_s']):
                result['errors'].append('Descendants remained after child leader exit')
        except BaseException as exc:
            result['errors'].append(type(exc).__name__+': '+str(exc))
        finally:
            # Repeated parent interrupts cannot bypass owned-child cleanup.
            set_handlers(True)
            try:
                result['process_group_drained'] = process is None
                if process is not None:
                    if identity is None or not valid_session(identity):
                        try:
                            # No poll/wait has run yet on this path: this direct
                            # child is still unreaped and its PID cannot recycle.
                            identity = process_identity(process.pid)
                            result['process_identity'] = identity
                            result['identity_recovered_in_cleanup'] = True
                        except BaseException as exc:
                            result['errors'].append('Identity recovery: '+type(exc).__name__+': '+str(exc))
                    if identity is not None and valid_session(identity):
                        try:
                            result['process_group_drained'] = (stop_owned(process, identity, result['signals'])
                                if result['errors'] else True)
                        except BaseException as exc:
                            result['errors'].append('Shutdown: '+type(exc).__name__+': '+str(exc))
                    else:
                        # Fail closed on group ownership. Popen retains its
                        # unreaped direct child; signal only that child, never
                        # an unverified process group or an arbitrary PID.
                        result['errors'].append('Session identity unavailable; direct-child-only cleanup')
                        for sig, seconds in [(signal.SIGINT, 120), (signal.SIGTERM, 60), (signal.SIGKILL, 30)]:
                            if process.poll() is not None: break
                            result['signals'].append(dict(signal=sig.name, direct_child_only=True, sent_utc=now()))
                            process.send_signal(sig)
                            try: process.wait(timeout=seconds); break
                            except subprocess.TimeoutExpired: pass
                    result['exit_code'] = process.poll()
                else: result['exit_code'] = None
                if result['deferred_signals'] and not result['errors']:
                    result['errors'].append('Parent interrupt received during process lifecycle')
                result['ended_utc'] = now()
                result['passed'] = (result['exit_code'] == 0 and result['process_group_drained'] and not result['errors'])
                write_new(args.evidence/(name+'.process.json'), result)
            finally: set_handlers(False)
        # A signal may arrive while the final JSON is being written, after the
        # earlier cancellation check. Preserve that process receipt and append
        # an explicit cancellation; never dispatch a next qualification stage.
        if result['passed'] and result['deferred_signals']:
            result['passed'] = False
            result['errors'].append('Parent interrupt received during final process receipt write')
            write_new(args.evidence/(name+'.late_interrupt.json'), dict(
                cancelled=True, process_receipt_sha256=sha(args.evidence/(name+'.process.json')), result=result))
    return result


def wait_gpu_release():
    deadline = time.monotonic()+GUARD['gpu_release_s']
    while True:
        if not gpu_pids(): return True
        if time.monotonic() >= deadline: return False
        time.sleep(2)
