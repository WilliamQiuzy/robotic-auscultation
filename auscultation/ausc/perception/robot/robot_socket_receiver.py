"""Credential-bound Unix datagram receiver for exact left OpenArm control.

The OpenArm controller remains the sole CAN owner.  This module only receives
the controller's already-sampled state over a local Unix datagram socket,
validates the kernel-provided sender credentials, and forwards the *original*
v2/v3/v4 envelope bytes to an exact-control sink.  Production sinks require
v4; v2/v3 remain available only to explicit commissioning or diagnostic sinks. Diagnostic v1
robot-state packets cannot satisfy this receiver's constructor or Start
preflight.

Production callers pass the telemetry socket path from the activated pair
contract.  Unless an explicit ``runtime_dir`` is supplied for an isolated
test, the path must be a direct child of ``/run/user/<service uid>/wam``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import errno
import fcntl
import hashlib
import os
from pathlib import Path
import re
import select
import selectors
import socket
import stat
import struct
import sys
import threading
import time
from types import MappingProxyType
from typing import Callable, Deque, Mapping, Optional, Tuple

from ausc.perception.robot.robot_state_stream import (
    CONTROL_ENVELOPE_SCHEMA,
    CONTROL_ENVELOPE_SCHEMA_V3,
    CONTROL_ENVELOPE_SCHEMA_V4,
    MAX_CONTROL_ENVELOPE_BYTES,
    RobotControlEnvelope,
    RobotStateContract,
    parse_control_envelope_datagram,
)


_SO_PASSCRED = getattr(socket, "SO_PASSCRED", None)
_SCM_CREDENTIALS = getattr(socket, "SCM_CREDENTIALS", None)
_SOCK_NONBLOCK = getattr(socket, "SOCK_NONBLOCK", 0)
_SOCK_CLOEXEC = getattr(socket, "SOCK_CLOEXEC", 0)
_MSG_TRUNC = getattr(socket, "MSG_TRUNC", 0)
_MSG_CTRUNC = getattr(socket, "MSG_CTRUNC", 0)

# Linux include/uapi/asm-generic/socket.h.  Python does not expose this on all
# supported distributions, so the numeric ABI value is kept local.
_SO_RXQ_OVFL = getattr(socket, "SO_RXQ_OVFL", 40)
_UCRED_STRUCT = struct.Struct("=3i")
_OVERFLOW_STRUCT = struct.Struct("=I")
_SOCKET_NAME_RE = re.compile(r"[A-Za-z0-9_.-]{1,80}\.sock\Z")
_MAX_UNIX_PATH_BYTES = 107
_MAX_POLL_TIMEOUT_S = 0.25
_MAX_CONTROL_TICK_GAP_NS = 2_000_000
# Cold-start envelope-continuity grace (shared-box scheduling jitter): for the
# first N envelopes, tolerate up to this inter-envelope publisher gap instead of
# latching a terminal fault. Steady state reverts to the contract's hard bound.
_STARTUP_GRACE_ENVELOPES = 20
_STARTUP_GRACE_GAP_NS = 200_000_000


class RobotSocketReceiverError(RuntimeError):
    """The telemetry socket cannot be operated safely."""


class RobotSocketPreflightError(RobotSocketReceiverError):
    """Fresh continuous left-pair telemetry is not ready for capture start."""


def _is_exact_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_process_identity_fields(value: object) -> None:
    for field_name in (
        "pid",
        "uid",
        "proc_start_time_ticks",
        "executable_device",
        "executable_inode",
    ):
        field_value = getattr(value, field_name, None)
        minimum = 1 if field_name != "uid" else 0
        if not _is_exact_int(field_value) or field_value < minimum:
            raise RobotSocketReceiverError(
                f"controller process {field_name} is invalid"
            )
    for field_name in ("executable_sha256", "cgroup_sha256"):
        digest = getattr(value, field_name, None)
        if not isinstance(digest, str) or re.fullmatch(
            r"[0-9a-f]{64}", digest
        ) is None:
            raise RobotSocketReceiverError(
                f"controller process {field_name} is invalid"
            )


@dataclass(frozen=True)
class ControllerProcessSnapshot:
    """One independently inspected Linux process identity snapshot."""

    pid: int
    uid: int
    proc_start_time_ticks: int
    executable_device: int
    executable_inode: int
    executable_sha256: str
    cgroup_sha256: str

    def __post_init__(self) -> None:
        _validate_process_identity_fields(self)


@dataclass(frozen=True)
class ControllerProcessAttestation:
    """Launcher-owned identity pin for exactly one OpenArm controller process.

    The attestation contains no envelope-provided values.  A trusted launcher
    creates it while holding a pidfd; the receiver opens its own pidfd and
    independently repeats the procfs inspection before accepting telemetry.
    """

    pid: int
    uid: int
    proc_start_time_ticks: int
    executable_device: int
    executable_inode: int
    executable_sha256: str
    cgroup_sha256: str

    def __post_init__(self) -> None:
        _validate_process_identity_fields(self)

    @classmethod
    def capture(
        cls,
        *,
        pid: object,
        expected_uid: object,
        inspector: object = None,
    ) -> "ControllerProcessAttestation":
        """Capture a race-guarded process identity for a trusted launcher."""

        if not _is_exact_int(pid) or pid <= 0:
            raise RobotSocketReceiverError(
                "controller process attestation PID must be positive"
            )
        if not _is_exact_int(expected_uid) or expected_uid < 0:
            raise RobotSocketReceiverError(
                "controller process attestation UID must be non-negative"
            )
        process_inspector = _require_process_inspector(inspector)
        guard = process_inspector.open_pidfd(pid)
        try:
            if process_inspector.pidfd_exited(guard):
                raise RobotSocketReceiverError(
                    "controller process pidfd exited during attestation"
                )
            snapshot = process_inspector.inspect(pid)
            if not isinstance(snapshot, ControllerProcessSnapshot):
                raise RobotSocketReceiverError(
                    "controller process inspector returned an invalid snapshot"
                )
            if snapshot.pid != pid:
                raise RobotSocketReceiverError(
                    "controller process inspector returned the wrong PID"
                )
            if snapshot.uid != expected_uid:
                raise RobotSocketReceiverError(
                    "controller process UID does not match trusted launcher UID"
                )
            if process_inspector.pidfd_exited(guard):
                raise RobotSocketReceiverError(
                    "controller process pidfd exited during attestation"
                )
            return cls(
                pid=snapshot.pid,
                uid=snapshot.uid,
                proc_start_time_ticks=snapshot.proc_start_time_ticks,
                executable_device=snapshot.executable_device,
                executable_inode=snapshot.executable_inode,
                executable_sha256=snapshot.executable_sha256,
                cgroup_sha256=snapshot.cgroup_sha256,
            )
        finally:
            process_inspector.close_pidfd(guard)


def _make_syscall_pidfd_open() -> Optional[Callable[[int], int]]:
    """Fallback ``pidfd_open`` via the raw Linux syscall.

    Some CPython builds ship without ``os.pidfd_open`` even though the running
    kernel provides the ``pidfd_open`` syscall (Linux >= 5.3).  The syscall yields
    a genuine pidfd with identical identity/liveness semantics, so controller
    attestation keeps its no-TOCTOU guarantee -- this is not a procfs downgrade.
    Returns ``None`` when a real pidfd cannot be obtained (non-Linux, unknown ABI,
    or no ctypes) so ``open_pidfd`` fails closed to its hard error rather than
    degrading silently.
    """
    if not sys.platform.startswith("linux"):
        return None
    import platform

    # pidfd_open is syscall 434 on every Linux ABI that defines it.
    if platform.machine() not in {"x86_64", "AMD64", "aarch64", "arm64"}:
        return None
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        libc.syscall.restype = ctypes.c_long
        libc.syscall.argtypes = [ctypes.c_long, ctypes.c_long, ctypes.c_ulong]
    except Exception:
        return None

    _SYS_PIDFD_OPEN = 434

    def _pidfd_open(pid: int) -> int:
        fd = libc.syscall(_SYS_PIDFD_OPEN, int(pid), 0)
        if fd < 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))
        return int(fd)

    return _pidfd_open


_SYSCALL_PIDFD_OPEN = _make_syscall_pidfd_open()


class LinuxControllerProcessInspector:
    """Linux procfs + pidfd identity inspector with injectable test hooks."""

    def __init__(
        self,
        *,
        proc_root: object = "/proc",
        pidfd_open: Optional[Callable[[int], object]] = None,
        pidfd_exited: Optional[Callable[[object], bool]] = None,
        pidfd_close: Optional[Callable[[object], None]] = None,
    ) -> None:
        root = Path(os.fspath(proc_root))
        if not root.is_absolute():
            raise RobotSocketReceiverError("procfs root must be absolute")
        self.proc_root = root
        self._pidfd_open_hook = pidfd_open
        self._pidfd_exited_hook = pidfd_exited
        self._pidfd_close_hook = pidfd_close
        self._executable_hash_cache: dict[Tuple[int, int, int, int], str] = {}

    def open_pidfd(self, pid: int) -> object:
        hook = self._pidfd_open_hook
        if hook is None:
            hook = getattr(os, "pidfd_open", None)
            if not sys.platform.startswith("linux") or not callable(hook):
                # CPython built without os.pidfd_open: fall back to the raw syscall,
                # which returns a genuine pidfd (same no-TOCTOU guarantee).
                hook = _SYSCALL_PIDFD_OPEN
                if hook is None:
                    raise RobotSocketReceiverError(
                        "Linux pidfd_open is required for controller process attestation"
                    )
        try:
            guard = hook(pid)
        except OSError as exc:
            raise RobotSocketReceiverError(
                "cannot open controller process pidfd"
            ) from exc
        if isinstance(guard, bool) or not isinstance(guard, int) or guard < 0:
            raise RobotSocketReceiverError(
                "controller process pidfd hook returned an invalid descriptor"
            )
        if self._pidfd_open_hook is None:
            try:
                os.set_inheritable(guard, False)
            except OSError as exc:
                os.close(guard)
                raise RobotSocketReceiverError(
                    "cannot make controller process pidfd close-on-exec"
                ) from exc
        return guard

    def pidfd_exited(self, guard: object) -> bool:
        hook = self._pidfd_exited_hook
        if hook is not None:
            result = hook(guard)
            if not isinstance(result, bool):
                raise RobotSocketReceiverError(
                    "controller pidfd liveness hook returned a non-boolean"
                )
            return result
        if isinstance(guard, bool) or not isinstance(guard, int) or guard < 0:
            raise RobotSocketReceiverError(
                "controller process pidfd descriptor is invalid"
            )
        poller = select.poll()
        poller.register(
            guard,
            select.POLLIN | select.POLLHUP | select.POLLERR | select.POLLNVAL,
        )
        return bool(poller.poll(0))

    def close_pidfd(self, guard: object) -> None:
        hook = self._pidfd_close_hook
        if hook is not None:
            hook(guard)
            return
        if isinstance(guard, int) and not isinstance(guard, bool) and guard >= 0:
            try:
                os.close(guard)
            except OSError:
                pass

    @staticmethod
    def _parse_starttime(payload: bytes) -> int:
        closing_parenthesis = payload.rfind(b")")
        if closing_parenthesis <= 0:
            raise RobotSocketReceiverError(
                "controller /proc stat comm field is malformed"
            )
        fields = payload[closing_parenthesis + 1 :].split()
        if len(fields) <= 19:
            raise RobotSocketReceiverError(
                "controller /proc stat lacks process starttime"
            )
        try:
            starttime = int(fields[19])
        except ValueError as exc:
            raise RobotSocketReceiverError(
                "controller /proc stat starttime is malformed"
            ) from exc
        if starttime <= 0:
            raise RobotSocketReceiverError(
                "controller /proc stat starttime is invalid"
            )
        return starttime

    @staticmethod
    def _parse_uid(payload: bytes) -> int:
        uid_line = next(
            (line for line in payload.splitlines() if line.startswith(b"Uid:")),
            b"",
        )
        fields = uid_line.split()[1:]
        if len(fields) != 4:
            raise RobotSocketReceiverError(
                "controller /proc status lacks four UID identities"
            )
        try:
            uids = tuple(int(field) for field in fields)
        except ValueError as exc:
            raise RobotSocketReceiverError(
                "controller /proc status UID is malformed"
            ) from exc
        if any(uid < 0 for uid in uids) or len(set(uids)) != 1:
            raise RobotSocketReceiverError(
                "controller real/effective/saved/filesystem UID identities differ"
            )
        return uids[0]

    def _hash_open_executable(
        self,
        executable_fd: int,
    ) -> Tuple[os.stat_result, str]:
        before = os.fstat(executable_fd)
        if not stat.S_ISREG(before.st_mode):
            raise RobotSocketReceiverError(
                "controller /proc executable is not a regular file"
            )
        cache_key = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_ctime_ns,
        )
        digest = self._executable_hash_cache.get(cache_key)
        if digest is None:
            hasher = hashlib.sha256()
            os.lseek(executable_fd, 0, os.SEEK_SET)
            while True:
                block = os.read(executable_fd, 1024 * 1024)
                if not block:
                    break
                hasher.update(block)
            digest = hasher.hexdigest()
            after = os.fstat(executable_fd)
            if (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_ctime_ns,
            ) != cache_key:
                raise RobotSocketReceiverError(
                    "controller executable changed while it was hashed"
                )
            self._executable_hash_cache[cache_key] = digest
        return before, digest

    def inspect(self, pid: int) -> ControllerProcessSnapshot:
        if not _is_exact_int(pid) or pid <= 0:
            raise RobotSocketReceiverError(
                "controller process inspection PID must be positive"
            )
        process_root = self.proc_root / str(pid)
        try:
            stat_payload = (process_root / "stat").read_bytes()
            status_payload = (process_root / "status").read_bytes()
            cgroup_payload = (process_root / "cgroup").read_bytes()
            if not cgroup_payload:
                raise RobotSocketReceiverError(
                    "controller /proc cgroup membership is empty"
                )
            open_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            executable_fd = os.open(process_root / "exe", open_flags)
        except RobotSocketReceiverError:
            raise
        except OSError as exc:
            raise RobotSocketReceiverError(
                "cannot inspect controller process through procfs"
            ) from exc
        try:
            executable_stat, executable_sha256 = self._hash_open_executable(
                executable_fd
            )
        finally:
            os.close(executable_fd)
        return ControllerProcessSnapshot(
            pid=pid,
            uid=self._parse_uid(status_payload),
            proc_start_time_ticks=self._parse_starttime(stat_payload),
            executable_device=executable_stat.st_dev,
            executable_inode=executable_stat.st_ino,
            executable_sha256=executable_sha256,
            cgroup_sha256=hashlib.sha256(cgroup_payload).hexdigest(),
        )


def _require_process_inspector(inspector: object) -> object:
    process_inspector = (
        LinuxControllerProcessInspector() if inspector is None else inspector
    )
    for method_name in (
        "open_pidfd",
        "pidfd_exited",
        "inspect",
        "close_pidfd",
    ):
        if not callable(getattr(process_inspector, method_name, None)):
            raise RobotSocketReceiverError(
                "controller process inspector is incomplete"
            )
    return process_inspector


def _process_snapshot_mismatch(
    attestation: ControllerProcessAttestation,
    snapshot: ControllerProcessSnapshot,
) -> str:
    comparisons = (
        ("PID", attestation.pid, snapshot.pid),
        ("UID", attestation.uid, snapshot.uid),
        (
            "starttime",
            attestation.proc_start_time_ticks,
            snapshot.proc_start_time_ticks,
        ),
        (
            "executable device",
            attestation.executable_device,
            snapshot.executable_device,
        ),
        (
            "executable inode",
            attestation.executable_inode,
            snapshot.executable_inode,
        ),
        (
            "executable SHA-256",
            attestation.executable_sha256,
            snapshot.executable_sha256,
        ),
        ("cgroup", attestation.cgroup_sha256, snapshot.cgroup_sha256),
    )
    for label, expected, observed in comparisons:
        if observed != expected:
            return f"controller process {label} no longer matches attestation"
    return ""


@dataclass(frozen=True)
class RobotSocketHealth:
    """Small immutable health surface for Collector state and UI."""

    status: str
    socket_path: str
    last_envelope_seq: Optional[int]
    last_control_tick_seq: Optional[int]
    last_age_ns: Optional[int]
    error: str
    controller_pid: Optional[int]
    pair: str
    envelope_count: int
    control_tick_count: int
    preflight_ready: bool

    @property
    def last_seq(self) -> Optional[int]:
        """Compatibility alias for display code; this is an envelope sequence."""

        return self.last_envelope_seq

    @property
    def packet_count(self) -> int:
        """Compatibility alias for display code; counts exact envelopes."""

        return self.envelope_count


def linux_peer_credentials_available() -> bool:
    """Return whether this kernel/Python can enforce Linux ``SCM_CREDENTIALS``."""

    return (
        os.name == "posix"
        and _SO_PASSCRED is not None
        and _SCM_CREDENTIALS is not None
        and hasattr(socket.socket, "recvmsg")
    )


class RobotSocketReceiver:
    """Background, read-only receiver with fail-closed fault latching."""

    def __init__(
        self,
        *,
        socket_path: object,
        sink: object,
        runtime_dir: object = None,
        controller_process_attestation: object = None,
        deferred_controller_attestation: bool = False,
        process_inspector: object = None,
        abort_callback: Optional[Callable[[str], None]] = None,
        poll_timeout_s: float = 0.05,
        preflight_min_envelopes: int = 11,
        preflight_duration_ns: int = 100_000_000,
    ) -> None:
        self._service_uid = os.geteuid()
        path = Path(os.fspath(socket_path))
        if not path.is_absolute():
            raise RobotSocketReceiverError(
                "activated telemetry socket path must be absolute"
            )
        if "\x00" in os.fspath(path):
            raise RobotSocketReceiverError(
                "activated telemetry socket path contains NUL"
            )
        if _SOCKET_NAME_RE.fullmatch(path.name) is None:
            raise RobotSocketReceiverError(
                "activated telemetry socket must have a safe .sock filename"
            )
        if len(os.fsencode(path)) > _MAX_UNIX_PATH_BYTES:
            raise RobotSocketReceiverError(
                "activated telemetry socket exceeds the Unix path limit"
            )

        if runtime_dir is None:
            expected_runtime_dir = Path(
                f"/run/user/{self._service_uid}/wam"
            )
        else:
            expected_runtime_dir = Path(os.fspath(runtime_dir))
            if not expected_runtime_dir.is_absolute():
                raise RobotSocketReceiverError("runtime directory must be absolute")
        if path.parent != expected_runtime_dir:
            raise RobotSocketReceiverError(
                "activated telemetry socket must be a direct child of the "
                "service runtime wam directory"
            )

        contract = getattr(sink, "contract", None)
        ingest = getattr(sink, "ingest_datagram", None)
        telemetry_schema = getattr(sink, "telemetry_schema", None)
        telemetry_schemas = getattr(
            sink, "telemetry_schemas", frozenset({telemetry_schema})
        )
        supported_schemas = frozenset(
            {
                CONTROL_ENVELOPE_SCHEMA,
                CONTROL_ENVELOPE_SCHEMA_V3,
                CONTROL_ENVELOPE_SCHEMA_V4,
            }
        )
        if (
            not isinstance(contract, RobotStateContract)
            or not callable(ingest)
            or telemetry_schema not in supported_schemas
            or not isinstance(telemetry_schemas, frozenset)
            or telemetry_schema not in telemetry_schemas
            or not telemetry_schemas <= supported_schemas
        ):
            raise RobotSocketReceiverError(
                "sink must expose one supported exact control envelope schema, a "
                "RobotStateContract, and ingest_datagram(bytes)"
            )
        if (
            contract.side != "left"
            or contract.controller_mode != "unilateral"
            or contract.leader_can_interface != "can1"
            or contract.follower_can_interface != "can3"
        ):
            raise RobotSocketReceiverError(
                "production receiver requires left OpenArm can1 leader -> "
                "can3 follower"
            )
        for digest, label in (
            (
                contract.controller_source_manifest_sha256,
                "controller source manifest",
            ),
            (contract.follower_config_sha256, "follower controller config"),
            (contract.openarm_sdk_sha256, "OpenArm SDK"),
        ):
            if (
                not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
                raise RobotSocketReceiverError(
                    f"exact-control receiver requires activated {label} SHA-256"
                )
        if (
            isinstance(poll_timeout_s, bool)
            or not isinstance(poll_timeout_s, (int, float))
            or not 0.0 < float(poll_timeout_s) <= _MAX_POLL_TIMEOUT_S
        ):
            raise RobotSocketReceiverError(
                "poll timeout must be positive and at most 250 ms"
            )
        if (
            not _is_exact_int(preflight_min_envelopes)
            or preflight_min_envelopes < 2
            or not _is_exact_int(preflight_duration_ns)
            or preflight_duration_ns <= 0
            or preflight_duration_ns > contract.buffer_duration_ns
        ):
            raise RobotSocketReceiverError(
                "preflight requires at least two envelopes within the telemetry buffer"
            )
        if abort_callback is not None and not callable(abort_callback):
            raise RobotSocketReceiverError("abort callback must be callable")
        if type(deferred_controller_attestation) is not bool:
            raise RobotSocketReceiverError(
                "deferred controller attestation flag must be boolean"
            )
        if (
            deferred_controller_attestation
            and controller_process_attestation is not None
        ):
            raise RobotSocketReceiverError(
                "deferred startup must begin without a controller attestation"
            )
        if (
            not deferred_controller_attestation
            and not isinstance(
                controller_process_attestation,
                ControllerProcessAttestation,
            )
        ):
            raise RobotSocketReceiverError(
                "production receiver requires an independent controller "
                "process attestation"
            )
        inspector = _require_process_inspector(process_inspector)
        attestation: Optional[ControllerProcessAttestation] = (
            controller_process_attestation
            if isinstance(
                controller_process_attestation,
                ControllerProcessAttestation,
            )
            else None
        )
        guard: Optional[object] = None
        if attestation is not None:
            self._validate_attestation_contract(
                attestation=attestation,
                contract=contract,
                service_uid=self._service_uid,
            )
            guard = inspector.open_pidfd(attestation.pid)
            try:
                self._assert_process_identity(
                    inspector=inspector,
                    guard=guard,
                    attestation=attestation,
                )
            except Exception:
                inspector.close_pidfd(guard)
                raise

        self.socket_path = path
        self.runtime_dir = expected_runtime_dir
        self.contract = contract
        self.required_control_envelope_schema = telemetry_schema
        self._sink = sink
        self._abort_callback = abort_callback
        self._poll_timeout_s = float(poll_timeout_s)
        self._preflight_min_envelopes = preflight_min_envelopes
        self._preflight_duration_ns = preflight_duration_ns
        self._process_attestation = attestation
        self._process_inspector = inspector
        self._process_guard: Optional[object] = guard
        self._deferred_controller_attestation = (
            deferred_controller_attestation
        )

        self._state_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._socket: Optional[socket.socket] = None
        self._selector: Optional[selectors.BaseSelector] = None
        self._runtime_fd: Optional[int] = None
        self._lock_fd: Optional[int] = None
        self._socket_identity: Optional[Tuple[int, int]] = None
        self._prepared = False
        self._ever_started = False
        self._started = False
        self._fault_reason = ""
        self._last_envelope: Optional[RobotControlEnvelope] = None
        self._envelope_count = 0
        self._control_tick_count = 0
        self._recent_envelopes: Deque[RobotControlEnvelope] = deque()
        self._last_overflow_count = 0
        self._socket_flags = MappingProxyType(
            {
                "nonblocking": False,
                "close_on_exec": False,
                "pass_credentials": False,
            }
        )

    @staticmethod
    def _validate_attestation_contract(
        *,
        attestation: ControllerProcessAttestation,
        contract: RobotStateContract,
        service_uid: int,
    ) -> None:
        if attestation.uid != service_uid:
            raise RobotSocketReceiverError(
                "controller process attestation UID does not match service UID"
            )
        if attestation.executable_sha256 != contract.controller_binary_sha256:
            raise RobotSocketReceiverError(
                "attested controller executable SHA-256 does not match the "
                "activated controller binary"
            )

    @staticmethod
    def _assert_process_identity(
        *,
        inspector: object,
        guard: object,
        attestation: ControllerProcessAttestation,
    ) -> None:
        if inspector.pidfd_exited(guard):
            raise RobotSocketReceiverError(
                "attested controller process pidfd has exited"
            )
        try:
            snapshot = inspector.inspect(attestation.pid)
        except RobotSocketReceiverError:
            raise
        except Exception as exc:
            raise RobotSocketReceiverError(
                "controller process inspection failed"
            ) from exc
        if not isinstance(snapshot, ControllerProcessSnapshot):
            raise RobotSocketReceiverError(
                "controller process inspector returned an invalid snapshot"
            )
        mismatch = _process_snapshot_mismatch(attestation, snapshot)
        if mismatch:
            raise RobotSocketReceiverError(mismatch)
        if inspector.pidfd_exited(guard):
            raise RobotSocketReceiverError(
                "attested controller process pidfd has exited"
            )

    def _assert_attested_controller_process(self) -> None:
        guard = self._process_guard
        attestation = self._process_attestation
        if guard is None or attestation is None:
            raise RobotSocketReceiverError(
                "attested controller process is unavailable"
            )
        self._assert_process_identity(
            inspector=self._process_inspector,
            guard=guard,
            attestation=attestation,
        )

    def attach_controller_process(
        self,
        attestation: ControllerProcessAttestation,
    ) -> None:
        """Attach the final exec identity after the telemetry socket is bound.

        This is the second half of the production startup handshake.  The
        trusted launcher must call :meth:`prepare_socket`, start the controller,
        independently capture its attestation, then attach it before
        :meth:`start` enables reception.
        """

        if not isinstance(attestation, ControllerProcessAttestation):
            raise RobotSocketReceiverError(
                "controller process attachment requires an attestation"
            )
        with self._state_lock:
            if not self._deferred_controller_attestation:
                raise RobotSocketReceiverError(
                    "controller process attachment is only valid in deferred mode"
                )
            if not self._prepared:
                raise RobotSocketReceiverError(
                    "attach controller process only after the socket is prepared"
                )
            if self._ever_started or self._started:
                raise RobotSocketReceiverError(
                    "cannot attach a controller after reception has started"
                )
            if (
                self._process_attestation is not None
                or self._process_guard is not None
            ):
                raise RobotSocketReceiverError(
                    "controller process is already attached"
                )
            if self._fault_reason:
                raise RobotSocketReceiverError(
                    f"robot telemetry receiver is faulted: {self._fault_reason}"
                )
        self._validate_attestation_contract(
            attestation=attestation,
            contract=self.contract,
            service_uid=self._service_uid,
        )
        guard = self._process_inspector.open_pidfd(attestation.pid)
        try:
            self._assert_process_identity(
                inspector=self._process_inspector,
                guard=guard,
                attestation=attestation,
            )
            with self._state_lock:
                if (
                    not self._prepared
                    or self._ever_started
                    or self._process_attestation is not None
                    or self._process_guard is not None
                ):
                    raise RobotSocketReceiverError(
                        "controller process attachment ordering changed"
                    )
                self._process_attestation = attestation
                self._process_guard = guard
        except Exception:
            self._process_inspector.close_pidfd(guard)
            raise

    def _latch_process_fault_if_any(self) -> None:
        guard = self._process_guard
        if guard is None:
            return
        try:
            self._assert_attested_controller_process()
        except Exception as exc:  # noqa: BLE001 - identity failures are fatal
            self._latch_fault(str(exc) or type(exc).__name__)

    @property
    def socket_flags(self) -> Mapping[str, bool]:
        return self._socket_flags

    @property
    def faulted(self) -> bool:
        with self._state_lock:
            return bool(self._fault_reason)

    @property
    def fault_reason(self) -> str:
        with self._state_lock:
            return self._fault_reason

    def _open_runtime_directory(self) -> int:
        parent = self.runtime_dir.parent
        try:
            parent_stat = os.lstat(parent)
        except OSError as exc:
            raise RobotSocketReceiverError(
                "service runtime parent directory is unavailable"
            ) from exc
        if not stat.S_ISDIR(parent_stat.st_mode):
            raise RobotSocketReceiverError(
                "service runtime parent must be a real directory"
            )

        try:
            runtime_stat = os.lstat(self.runtime_dir)
        except FileNotFoundError:
            try:
                os.mkdir(self.runtime_dir, mode=0o700)
            except OSError as exc:
                raise RobotSocketReceiverError(
                    "cannot create private telemetry runtime directory"
                ) from exc
            runtime_stat = os.lstat(self.runtime_dir)
        except OSError as exc:
            raise RobotSocketReceiverError(
                "cannot inspect telemetry runtime directory"
            ) from exc

        if (
            not stat.S_ISDIR(runtime_stat.st_mode)
            or stat.S_ISLNK(runtime_stat.st_mode)
            or runtime_stat.st_uid != self._service_uid
        ):
            raise RobotSocketReceiverError(
                "telemetry runtime directory must be a service-owned real directory"
            )

        open_flags = os.O_RDONLY
        open_flags |= getattr(os, "O_DIRECTORY", 0)
        open_flags |= getattr(os, "O_NOFOLLOW", 0)
        open_flags |= getattr(os, "O_CLOEXEC", 0)
        try:
            runtime_fd = os.open(self.runtime_dir, open_flags)
        except OSError as exc:
            raise RobotSocketReceiverError(
                "cannot securely open telemetry runtime directory"
            ) from exc
        try:
            opened_stat = os.fstat(runtime_fd)
            if (
                not stat.S_ISDIR(opened_stat.st_mode)
                or opened_stat.st_uid != self._service_uid
                or (
                    opened_stat.st_dev,
                    opened_stat.st_ino,
                )
                != (runtime_stat.st_dev, runtime_stat.st_ino)
            ):
                raise RobotSocketReceiverError(
                    "telemetry runtime directory changed during startup"
                )
            os.fchmod(runtime_fd, 0o700)
            if stat.S_IMODE(os.fstat(runtime_fd).st_mode) != 0o700:
                raise RobotSocketReceiverError(
                    "telemetry runtime directory mode is not 0700"
                )
            return runtime_fd
        except Exception:
            os.close(runtime_fd)
            raise

    def _acquire_receiver_lock(self, runtime_fd: int) -> int:
        flags = os.O_RDWR | os.O_CREAT
        flags |= getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_CLOEXEC", 0)
        try:
            lock_fd = os.open(
                ".openarm-left-telemetry.receiver.lock",
                flags,
                0o600,
                dir_fd=runtime_fd,
            )
            lock_stat = os.fstat(lock_fd)
            if (
                not stat.S_ISREG(lock_stat.st_mode)
                or lock_stat.st_uid != self._service_uid
                or lock_stat.st_nlink != 1
            ):
                raise RobotSocketReceiverError(
                    "telemetry receiver lock is not a private regular file"
                )
            os.fchmod(lock_fd, 0o600)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RobotSocketReceiverError(
                    "another telemetry receiver owns the activated socket"
                ) from exc
            return lock_fd
        except Exception:
            try:
                os.close(lock_fd)
            except (OSError, UnboundLocalError):
                pass
            raise

    def _remove_stale_socket(self) -> None:
        try:
            existing = os.lstat(self.socket_path)
        except FileNotFoundError:
            return
        except OSError as exc:
            raise RobotSocketReceiverError(
                "cannot inspect activated telemetry socket"
            ) from exc
        if stat.S_ISLNK(existing.st_mode):
            raise RobotSocketReceiverError(
                "activated telemetry socket path must not be a symlink"
            )
        if (
            not stat.S_ISSOCK(existing.st_mode)
            or existing.st_uid != self._service_uid
            or existing.st_nlink != 1
        ):
            raise RobotSocketReceiverError(
                "existing telemetry path is not a private service-owned socket"
            )
        try:
            os.unlink(self.socket_path)
        except OSError as exc:
            raise RobotSocketReceiverError(
                "cannot remove stale activated telemetry socket"
            ) from exc

    def _make_socket(self) -> socket.socket:
        if not linux_peer_credentials_available():
            raise RobotSocketReceiverError(
                "Linux SO_PASSCRED/SCM_CREDENTIALS is required for robot telemetry"
            )
        socket_type = int(socket.SOCK_DGRAM)
        socket_type |= int(_SOCK_NONBLOCK)
        socket_type |= int(_SOCK_CLOEXEC)
        try:
            receiver = socket.socket(socket.AF_UNIX, socket_type)
        except OSError as exc:
            raise RobotSocketReceiverError(
                "cannot create credential-bound Unix datagram socket"
            ) from exc
        try:
            receiver.setblocking(False)
            receiver.set_inheritable(False)
            receiver.setsockopt(socket.SOL_SOCKET, int(_SO_PASSCRED), 1)
            # Anti-starvation headroom on a SHARED box: one envelope can be 128 KiB
            # and the default ~212 KiB rcvbuf holds barely two, so a receiver-thread
            # scheduling stall of a few 10s of ms overflows the queue and the
            # publisher's nonblocking sendto fails -- permanently poisoning its
            # send-failure counter (which must stay zero). 4 MiB (kernel rmem_max)
            # absorbs seconds of stall; the timing bounds still police quality.
            try:
                receiver.setsockopt(
                    socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024
                )
            except OSError:
                pass  # keep the default; the bounds checks still fail closed
            try:
                receiver.setsockopt(socket.SOL_SOCKET, _SO_RXQ_OVFL, 1)
            except OSError as exc:
                if exc.errno not in (
                    errno.ENOPROTOOPT,
                    errno.EINVAL,
                    errno.ENOTSUP,
                ):
                    raise
            receiver.bind(os.fspath(self.socket_path))
            os.chmod(self.socket_path, 0o600, follow_symlinks=False)
            created = os.lstat(self.socket_path)
            if (
                not stat.S_ISSOCK(created.st_mode)
                or stat.S_ISLNK(created.st_mode)
                or created.st_uid != self._service_uid
                or stat.S_IMODE(created.st_mode) != 0o600
                or created.st_nlink != 1
            ):
                raise RobotSocketReceiverError(
                    "bound telemetry socket does not have required 0600 ownership"
                )
            self._socket_identity = (created.st_dev, created.st_ino)
            self._socket_flags = MappingProxyType(
                {
                    "nonblocking": not receiver.getblocking(),
                    "close_on_exec": not receiver.get_inheritable(),
                    "pass_credentials": bool(
                        receiver.getsockopt(
                            socket.SOL_SOCKET, int(_SO_PASSCRED)
                        )
                    ),
                }
            )
            if not all(self._socket_flags.values()):
                raise RobotSocketReceiverError(
                    "telemetry socket safety flags were not activated"
                )
            return receiver
        except Exception:
            receiver.close()
            self._unlink_owned_socket()
            raise

    def prepare_socket(self) -> None:
        """Bind the private socket without accepting controller data yet.

        Deferred startup uses this before launching OpenArm so the publisher's
        first datagram has a real kernel queue.  No receive thread is started
        and no READY state is possible until an independently captured process
        attestation is attached.
        """

        with self._state_lock:
            if self._prepared:
                return
            if self._ever_started or self._started:
                raise RobotSocketReceiverError(
                    "cannot prepare a socket after reception has started"
                )
            if self._fault_reason:
                raise RobotSocketReceiverError(
                    f"robot telemetry receiver is faulted: {self._fault_reason}"
                )
        try:
            runtime_fd = self._open_runtime_directory()
            lock_fd = self._acquire_receiver_lock(runtime_fd)
            self._remove_stale_socket()
            receiver = self._make_socket()
            selector = selectors.DefaultSelector()
            selector.register(receiver, selectors.EVENT_READ)
        except Exception:
            try:
                selector.close()
            except (UnboundLocalError, Exception):
                pass
            try:
                receiver.close()
            except (UnboundLocalError, Exception):
                pass
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
            except (UnboundLocalError, OSError):
                pass
            try:
                os.close(runtime_fd)
            except (UnboundLocalError, OSError):
                pass
            self._unlink_owned_socket()
            raise

        with self._state_lock:
            self._runtime_fd = runtime_fd
            self._lock_fd = lock_fd
            self._socket = receiver
            self._selector = selector
            self._prepared = True

    def start(self) -> None:
        """Enable reception after socket preparation and process attestation."""

        try:
            self._assert_attested_controller_process()
        except Exception as exc:
            self._latch_fault(str(exc) or type(exc).__name__)
            raise
        with self._state_lock:
            if self._ever_started:
                raise RobotSocketReceiverError(
                    "a stopped receiver cannot be restarted; create a new instance"
                )
            if self._fault_reason:
                raise RobotSocketReceiverError(
                    f"robot telemetry receiver is faulted: {self._fault_reason}"
                )
        if not self._prepared:
            self.prepare_socket()
        with self._state_lock:
            if (
                not self._prepared
                or self._socket is None
                or self._selector is None
            ):
                raise RobotSocketReceiverError(
                    "telemetry socket preparation did not complete"
                )
            self._ever_started = True
            self._started = True
            self._thread = threading.Thread(
                target=self._receive_loop,
                name="wam-openarm-telemetry-receiver",
                daemon=True,
            )
            self._thread.start()

    def _decode_ancillary(
        self,
        ancillary: list[tuple[int, int, bytes]],
    ) -> Tuple[int, int, int, Optional[int]]:
        credentials = []
        overflow_count: Optional[int] = None
        for level, kind, data in ancillary:
            if level != socket.SOL_SOCKET:
                continue
            if kind == int(_SCM_CREDENTIALS):
                if len(data) < _UCRED_STRUCT.size:
                    raise RobotSocketReceiverError(
                        "SCM_CREDENTIALS payload is truncated"
                    )
                credentials.append(_UCRED_STRUCT.unpack_from(data))
            elif kind == _SO_RXQ_OVFL:
                if len(data) < _OVERFLOW_STRUCT.size:
                    raise RobotSocketReceiverError(
                        "socket overrun counter is truncated"
                    )
                overflow_count = _OVERFLOW_STRUCT.unpack_from(data)[0]
        if len(credentials) != 1:
            raise RobotSocketReceiverError(
                "exactly one SCM_CREDENTIALS record is required"
            )
        pid, uid, gid = credentials[0]
        return pid, uid, gid, overflow_count

    def _receive_one(self, receiver: socket.socket) -> bool:
        ancillary_size = socket.CMSG_SPACE(
            _UCRED_STRUCT.size
        ) + socket.CMSG_SPACE(_OVERFLOW_STRUCT.size)
        try:
            payload, ancillary, message_flags, _address = receiver.recvmsg(
                MAX_CONTROL_ENVELOPE_BYTES,
                ancillary_size,
            )
        except BlockingIOError:
            return False
        if message_flags & int(_MSG_TRUNC):
            raise RobotSocketReceiverError(
                "robot telemetry socket overrun or oversized datagram"
            )
        if message_flags & int(_MSG_CTRUNC):
            raise RobotSocketReceiverError(
                "robot telemetry credential metadata was truncated"
            )
        peer_pid, peer_uid, _peer_gid, overflow_count = self._decode_ancillary(
            ancillary
        )
        attestation = self._process_attestation
        if attestation is None:
            raise RobotSocketReceiverError(
                "controller datagram arrived before process attestation"
            )
        if peer_uid != self._service_uid:
            raise RobotSocketReceiverError(
                "robot telemetry credential UID does not match service UID"
            )
        if peer_pid != attestation.pid:
            raise RobotSocketReceiverError(
                "robot telemetry credential PID does not match attested "
                "controller PID"
            )
        self._assert_attested_controller_process()
        if overflow_count is not None:
            if overflow_count > self._last_overflow_count:
                self._last_overflow_count = overflow_count
                raise RobotSocketReceiverError(
                    "robot telemetry socket receive queue overrun"
                )
            self._last_overflow_count = overflow_count

        envelope = parse_control_envelope_datagram(payload, self.contract)
        if envelope.schema != self.required_control_envelope_schema:
            required = self.required_control_envelope_schema.rsplit("_", 1)[-1]
            raise RobotSocketReceiverError(
                "production telemetry sink requires exact-control "
                f"{required}; commissioning packet was rejected before ingest"
            )
        if peer_pid != envelope.controller_pid:
            raise RobotSocketReceiverError(
                "robot telemetry credential PID does not match controller_pid"
            )

        with self._state_lock:
            previous = self._last_envelope
        self._validate_envelope_continuity(previous, envelope)

        # Preserve the wire record exactly.  Credential/protocol checks happen
        # before this call so an untrusted peer cannot contaminate the stream.
        ingested = self._sink.ingest_datagram(payload)
        if not isinstance(ingested, RobotControlEnvelope) or ingested != envelope:
            raise RobotSocketReceiverError(
                "exact control sink did not preserve the validated envelope"
            )
        # If the process exited after the procfs snapshot but before the sink
        # returned, the pidfd becomes readable even if Linux has already
        # reused the numeric PID.  Never count that envelope toward READY.
        guard = self._process_guard
        if guard is None or self._process_inspector.pidfd_exited(guard):
            raise RobotSocketReceiverError(
                "attested controller process pidfd has exited"
            )

        with self._state_lock:
            self._last_envelope = envelope
            self._envelope_count += 1
            self._control_tick_count += len(envelope.records)
            self._recent_envelopes.append(envelope)
            cutoff = (
                envelope.records[-1].observation_monotonic_ns
                - self.contract.buffer_duration_ns
            )
            while self._recent_envelopes and (
                self._recent_envelopes[0].records[-1].observation_monotonic_ns
                < cutoff
                or len(self._recent_envelopes) > self.contract.max_packets
            ):
                self._recent_envelopes.popleft()
        return True

    def _validate_envelope_continuity(
        self,
        previous: Optional[RobotControlEnvelope],
        envelope: RobotControlEnvelope,
    ) -> None:
        """Reject loss/jitter before the original bytes reach the sink."""

        records = envelope.records
        for left, right in zip(records, records[1:]):
            gap_ns = (
                right.observation_monotonic_ns
                - left.observation_monotonic_ns
            )
            if gap_ns <= 0 or gap_ns > _MAX_CONTROL_TICK_GAP_NS:
                raise RobotSocketReceiverError(
                    "exact 1 kHz control tick gap exceeds hard 2 ms bound"
                )
        if previous is None:
            return
        if envelope.process_identity != previous.process_identity:
            raise RobotSocketReceiverError(
                "exact control publisher process identity changed"
            )
        if envelope.schema != previous.schema:
            raise RobotSocketReceiverError(
                "exact control envelope schema changed"
            )
        if envelope.envelope_seq != previous.envelope_seq + 1:
            raise RobotSocketReceiverError(
                "control envelope sequence loss or duplication"
            )
        if envelope.publisher_monotonic_ns <= previous.publisher_monotonic_ns:
            raise RobotSocketReceiverError(
                "control envelope publisher clock rollback"
            )
        # Cold-start grace on a shared box: while the controller's RT threads spin
        # up (and noisy neighbors burst), the deliberately low-priority publisher
        # thread can miss a couple of 10 ms periods. For the first envelopes only,
        # tolerate up to 200 ms between envelopes instead of latching a terminal
        # fault; sequence contiguity and clock monotonicity above stay strict, and
        # READY still requires a clean continuous preflight window at the hard
        # bound, so steady-state quality guarantees are unchanged.
        gap_bound_ns = self.contract.max_gap_ns
        if previous.envelope_seq < _STARTUP_GRACE_ENVELOPES:
            gap_bound_ns = max(gap_bound_ns, _STARTUP_GRACE_GAP_NS)
        if (
            envelope.publisher_monotonic_ns - previous.publisher_monotonic_ns
            > gap_bound_ns
        ):
            raise RobotSocketReceiverError(
                "100 Hz control envelope continuity exceeds hard 20 ms bound"
            )
        previous_record = previous.records[-1]
        current_record = records[0]
        if current_record.control_tick_seq != previous_record.control_tick_seq + 1:
            raise RobotSocketReceiverError(
                "exact control tick sequence loss or duplication"
            )
        tick_gap_ns = (
            current_record.observation_monotonic_ns
            - previous_record.observation_monotonic_ns
        )
        if tick_gap_ns <= 0 or tick_gap_ns > _MAX_CONTROL_TICK_GAP_NS:
            raise RobotSocketReceiverError(
                "exact 1 kHz control tick gap exceeds hard 2 ms bound"
            )

    def _receive_loop(self) -> None:
        selector = self._selector
        receiver = self._socket
        assert selector is not None
        assert receiver is not None
        try:
            while not self._stop_event.is_set():
                events = selector.select(self._poll_timeout_s)
                if not events:
                    self._assert_attested_controller_process()
                    continue
                for _key, _mask in events:
                    for _ in range(64):
                        if self._stop_event.is_set():
                            return
                        if not self._receive_one(receiver):
                            break
        except Exception as exc:  # noqa: BLE001 - every receive fault is fatal
            if not self._stop_event.is_set():
                self._latch_fault(str(exc) or type(exc).__name__)

    def _latch_fault(self, reason: str) -> None:
        callback = None
        with self._state_lock:
            if self._fault_reason:
                return
            self._fault_reason = str(reason)
            callback = self._abort_callback
            self._stop_event.set()
        if callback is not None:
            try:
                callback(self._fault_reason)
            except Exception:
                # The receiver's fault is already durable in health.  Capture
                # abort callbacks may intentionally raise after flushing.
                pass

    def _preflight_failure_locked(self, now_monotonic_ns: int) -> str:
        if not self._started:
            return "robot telemetry receiver is not running"
        if self._fault_reason:
            return f"robot telemetry receiver is faulted: {self._fault_reason}"
        if not self._recent_envelopes:
            return "robot start preflight has no exact control telemetry"
        envelopes = tuple(self._recent_envelopes)
        records = tuple(
            record for envelope in envelopes for record in envelope.records
        )
        if any(
            envelope.schema != self.required_control_envelope_schema
            for envelope in envelopes
        ):
            return "robot start preflight envelope schema is not activated"
        if (
            self.required_control_envelope_schema
            in {CONTROL_ENVELOPE_SCHEMA_V3, CONTROL_ENVELOPE_SCHEMA_V4}
            and any(
                not record.action_send_status_verified for record in records
            )
        ):
            return (
                "robot start preflight contains a SocketCAN TX enqueue failure"
            )
        if (
            self.required_control_envelope_schema
            == CONTROL_ENVELOPE_SCHEMA_V4
            and any(not record.feedback_rx_all_fresh for record in records)
        ):
            return (
                "robot start preflight v4 feedback is stale or exceeds the "
                "per-motor 20 ms gap bound"
            )
        latest = records[-1]
        if now_monotonic_ns < latest.observation_monotonic_ns:
            return "robot start preflight query clock precedes telemetry"
        if (
            now_monotonic_ns - latest.observation_monotonic_ns
            > self.contract.max_staleness_ns
        ):
            return "robot start preflight exact control telemetry is stale"
        if len(envelopes) < self._preflight_min_envelopes:
            return "robot start preflight has too few continuous exact envelopes"
        if (
            latest.observation_monotonic_ns
            - records[0].observation_monotonic_ns
            < self._preflight_duration_ns
        ):
            return "robot start preflight continuous duration is too short"
        first = envelopes[0]
        for left, right in zip(envelopes, envelopes[1:]):
            if (
                right.envelope_seq != left.envelope_seq + 1
                or right.publisher_monotonic_ns <= left.publisher_monotonic_ns
                or right.publisher_monotonic_ns - left.publisher_monotonic_ns
                > self.contract.max_gap_ns
                or right.process_identity != first.process_identity
            ):
                return (
                    "robot start preflight is not one continuous 100 Hz envelope stream"
                )
        for left, right in zip(records, records[1:]):
            if (
                right.control_tick_seq != left.control_tick_seq + 1
                or right.observation_monotonic_ns
                <= left.observation_monotonic_ns
                or right.observation_monotonic_ns
                - left.observation_monotonic_ns
                > _MAX_CONTROL_TICK_GAP_NS
                or right.process_identity != records[0].process_identity
            ):
                return (
                    "robot start preflight is not one continuous exact 1 kHz "
                    "control tick stream"
                )
        if any(
            record.side != "left"
            or record.controller_mode != "unilateral"
            or record.leader_can_interface != "can1"
            or record.follower_can_interface != "can3"
            or record.controller_source_manifest_sha256
            != self.contract.controller_source_manifest_sha256
            or record.follower_config_sha256
            != self.contract.follower_config_sha256
            or record.openarm_sdk_sha256 != self.contract.openarm_sdk_sha256
            for record in records
        ):
            return (
                "robot start preflight is not left can1 leader -> can3 follower"
            )
        return ""

    def require_start_preflight(
        self,
        *,
        now_monotonic_ns: Optional[int] = None,
    ) -> RobotSocketHealth:
        """Require fresh continuous left-pair telemetry before capture start."""

        now_ns = time.monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
        if not _is_exact_int(now_ns) or now_ns <= 0:
            raise RobotSocketPreflightError(
                "robot start preflight time must be positive integer ns"
            )
        self._latch_process_fault_if_any()
        with self._state_lock:
            failure = self._preflight_failure_locked(now_ns)
        if failure:
            raise RobotSocketPreflightError(failure)
        return self.health(now_monotonic_ns=now_ns)

    def health(
        self,
        *,
        now_monotonic_ns: Optional[int] = None,
    ) -> RobotSocketHealth:
        now_ns = time.monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
        if not _is_exact_int(now_ns) or now_ns <= 0:
            raise RobotSocketReceiverError(
                "robot socket health time must be positive integer ns"
            )
        self._latch_process_fault_if_any()
        with self._state_lock:
            envelope = self._last_envelope
            latest_record = (
                None if envelope is None else envelope.records[-1]
            )
            age_ns = (
                None
                if latest_record is None
                else max(0, now_ns - latest_record.observation_monotonic_ns)
            )
            preflight_failure = self._preflight_failure_locked(now_ns)
            preflight_ready = not preflight_failure
            if self._fault_reason:
                status = "FAULT"
            elif (
                self._prepared
                and self._process_attestation is None
                and not self._started
            ):
                status = "WAITING_CONTROLLER"
            elif not self._started:
                status = "STOPPED"
            elif envelope is None:
                status = "WAITING"
            elif age_ns is not None and age_ns > self.contract.max_staleness_ns:
                status = "STALE"
            elif preflight_ready:
                status = "READY"
            else:
                status = "WARMING"
            return RobotSocketHealth(
                status=status,
                socket_path=os.fspath(self.socket_path),
                last_envelope_seq=(
                    None if envelope is None else envelope.envelope_seq
                ),
                last_control_tick_seq=(
                    None
                    if latest_record is None
                    else latest_record.control_tick_seq
                ),
                last_age_ns=age_ns,
                error=self._fault_reason,
                controller_pid=(
                    None if envelope is None else envelope.controller_pid
                ),
                pair="left:can1->can3",
                envelope_count=self._envelope_count,
                control_tick_count=self._control_tick_count,
                preflight_ready=preflight_ready,
            )

    def _unlink_owned_socket(self) -> None:
        identity = self._socket_identity
        if identity is None:
            return
        try:
            current = os.lstat(self.socket_path)
        except FileNotFoundError:
            self._socket_identity = None
            return
        except OSError:
            return
        if (
            stat.S_ISSOCK(current.st_mode)
            and not stat.S_ISLNK(current.st_mode)
            and (current.st_dev, current.st_ino) == identity
        ):
            try:
                os.unlink(self.socket_path)
            except OSError:
                return
        self._socket_identity = None

    def stop(self) -> None:
        """Stop bounded polling and unlink only the socket inode we created."""

        with self._state_lock:
            self._stop_event.set()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, self._poll_timeout_s * 4.0))

        selector = self._selector
        self._selector = None
        if selector is not None:
            try:
                selector.close()
            except Exception:
                pass
        receiver = self._socket
        self._socket = None
        if receiver is not None:
            try:
                receiver.close()
            except OSError:
                pass
        self._unlink_owned_socket()

        lock_fd = self._lock_fd
        self._lock_fd = None
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(lock_fd)
            except OSError:
                pass
        runtime_fd = self._runtime_fd
        self._runtime_fd = None
        if runtime_fd is not None:
            try:
                os.close(runtime_fd)
            except OSError:
                pass
        process_guard = self._process_guard
        self._process_guard = None
        if process_guard is not None:
            self._process_inspector.close_pidfd(process_guard)
        with self._state_lock:
            # Stop is terminal even if only the pre-controller socket had been
            # prepared; a fresh object is required for a fresh kernel queue.
            self._ever_started = True
            self._started = False
            self._prepared = False
            self._thread = None

    def __enter__(self) -> "RobotSocketReceiver":
        self.start()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.stop()


__all__ = [
    "ControllerProcessAttestation",
    "ControllerProcessSnapshot",
    "LinuxControllerProcessInspector",
    "RobotSocketHealth",
    "RobotSocketPreflightError",
    "RobotSocketReceiver",
    "RobotSocketReceiverError",
    "linux_peer_credentials_available",
]
