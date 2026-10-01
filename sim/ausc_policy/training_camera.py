"""Guarded, in-memory native RGB-D; no raw recorder or controller integration.

API: ``GuardedRGBDStream(env, camera_bytes=..., camera_sha256=...,
backend_factory=None)`` owns one m2_capture._NativeBackend scope. Native use
retains that backend's owned-Linux gate; CPU tests explicitly inject a backend
with the same snapshot_camera/setup/render/read/detach/destroy_product protocol.
Factories must own no native resources until they return; setup must register
partial resources before an operation can fail, as _NativeBackend does.

``read()`` performs one zero-delta render and returns owned native uint8 RGB
or RGBA, floating metric optical-Z depth in metres, finite-positive validity,
actual camera calibration, and measured episode time (s), q (rad), dq (rad/s),
slider compression (m) and world time (s). It never accepts target ground truth.
Every setup/render/read is surrounded by approved-camera and measured-state
checks, using camera tolerance 1e-7, episode/world/q/dq tolerance 1e-8 and exact
compression equality. Physics may advance between separate read calls.

``warmup(frame_count)`` accepts an explicit integer 1..128, renders exactly that
many frames at a fixed physical state and returns JSON-compatible statistics,
per-render actual camera/clocks, adjacent RGB differences and the final raw
image SHA256. RGB statistics exclude alpha; the final hash includes all native
channels. Only previous/current frames and <=128 scalar summaries are retained.
Warmup and normal acquisition wall times are accounted separately. Warmup does
not test convergence, recommend a count, or change any production default.

``close()`` independently attempts both annotator detaches and product teardown,
returns a bounded error receipt, and is idempotent. Failed setup/read/warmup
automatically closes and latches failure; the original exception is re-raised
with ``training_camera_cleanup`` attached when supported. A cleanup error means
cleanup_complete=False, never a claim of no leaks. No files are written and all
training/physical qualification flags stay false.
"""
from copy import deepcopy
import hashlib
import json
import time

import numpy as np

from . import m2_capture


QUALIFICATIONS = dict(training_eligible=False,physical_qualification=False)


class GuardedRGBDStream:
    def __init__(self, env, *, camera_bytes, camera_sha256, backend_factory=None):
        if (not isinstance(camera_bytes,bytes) or camera_sha256 != m2_capture.APPROVED_CAMERA_SHA256 or
                hashlib.sha256(camera_bytes).hexdigest() != m2_capture.APPROVED_CAMERA_SHA256):
            raise ValueError('Exact approved camera bytes and pinned hash required')
        self._camera = json.loads(camera_bytes)
        self.env = env
        self._backend = None
        self._closed = False
        self._receipt = None
        self._failures = []
        self._stage = 'backend_construction'
        self._read_frames = self._warmup_frames = 0
        self._acquisition_wall_time_s = self._warmup_wall_time_s = 0.
        try:
            # Register ownership before setup can partially attach and fail.
            self._backend = (backend_factory or m2_capture._NativeBackend)(env)
            self._stage = 'setup'
            self._guarded('setup',self._backend.setup)
        except BaseException as error:
            self._fail_and_close(error)
            raise

    def _verify_camera(self):
        # Reuse the raw component's numeric validation without constructing a
        # raw recorder or changing its camera, optics or persistence behavior.
        return m2_capture.M2RGBDCapture._verify_camera(self)

    def _observe(self):
        state = m2_capture._state(self.env.state())
        clock = m2_capture._scalar(self.env.world.current_time,'world clock')
        camera = self._verify_camera()
        m2_capture._same_state(state,m2_capture._state(self.env.state()),'camera snapshot')
        if abs(m2_capture._scalar(self.env.world.current_time,'world clock')-clock) > 1e-8:
            raise ValueError('camera snapshot changed world clock')
        return state,clock,camera

    @staticmethod
    def _same_observation(before,after,label):
        m2_capture._same_state(before[0],after[0],label)
        if abs(before[1]-after[1]) > 1e-8: raise ValueError(label+' changed world clock')
        a,b = before[2],after[2]
        if not np.allclose(a['world_from_camera_usd'],b['world_from_camera_usd'],atol=1e-7,rtol=0):
            raise ValueError(label+' changed camera transform')
        for key,dimension in (('fx','width'),('fy','height'),('cx','width'),('cy','height')):
            if abs(a['intrinsics'][key]/a[dimension]-b['intrinsics'][key]/b[dimension]) > 1e-7:
                raise ValueError(label+' changed camera intrinsics')

    def _guarded(self,label,operation, *, anchor=None):
        self._stage = label
        before = self._observe()
        if anchor is None: anchor = before
        self._same_observation(anchor,before,label+' input')
        result = operation()
        after = self._observe()
        self._same_observation(before,after,label)
        self._same_observation(anchor,after,label+' complete')
        return result,after

    def _check_open(self):
        if self._failures: raise RuntimeError('RGB-D stream already failed; preserve its receipt')
        if self._closed: raise RuntimeError('RGB-D stream is closed')

    def _record_failure(self,stage,error):
        # One initiating failure plus at most three release failures; subsequent
        # operations are rejected before they can grow this receipt.
        try: message = str(error)
        except BaseException: message = '<exception message unavailable>'
        self._failures.append(dict(stage=stage,error=(type(error).__name__+': '+message)[:1024]))

    def _fail_and_close(self,error):
        self._record_failure(self._stage,error)
        receipt = self.close()
        try: error.training_camera_cleanup = receipt
        except BaseException: pass  # Preserve exceptions that disallow attributes.

    def _read_owned(self):
        values = self._backend.read()
        if not isinstance(values,(tuple,list)) or len(values) != 2:
            raise ValueError('Missing native RGB/depth pair')
        # Copy before any later camera snapshot can reuse annotator buffers.
        rgb,depth = (np.asarray(value).copy(order='C') for value in values)
        if rgb.dtype != np.uint8 or rgb.shape not in ((m2_capture.HEIGHT,m2_capture.WIDTH,3),
                                                     (m2_capture.HEIGHT,m2_capture.WIDTH,4)):
            raise ValueError('RGB must be native uint8 (360,640,3/4)')
        if depth.shape != (m2_capture.HEIGHT,m2_capture.WIDTH) or depth.dtype.kind != 'f':
            raise ValueError('Depth must be floating metric optical Z (360,640)')
        return rgb,depth,np.isfinite(depth)&(depth>0)

    @staticmethod
    def _state_dict(state, *, json_compatible=False):
        stamp,q,dq,compression = state
        return dict(time_s=stamp,q=q.tolist() if json_compatible else q.copy(),
                    dq=dq.tolist() if json_compatible else dq.copy(),compression_m=compression)

    def _frame(self):
        started = time.perf_counter()
        self._stage = 'read_anchor'
        anchor = self._observe()
        self._guarded('render',self._backend.render,anchor=anchor)
        values,observed = self._guarded('read',self._read_owned,anchor=anchor)
        rgb,depth,valid = values
        return dict(rgb=rgb,depth_m=depth,depth_valid=valid,actual_camera=observed[2],
            measured_state=self._state_dict(observed[0]),world_time_s=observed[1],
            acquisition_wall_time_s=time.perf_counter()-started),observed

    def read(self):
        """Acquire one owned native frame without advancing measured physics."""
        self._check_open()
        started = time.perf_counter()
        try:
            result,_ = self._frame()
            self._read_frames += 1
            return result
        except BaseException as error:
            self._acquisition_wall_time_s += time.perf_counter()-started
            self._fail_and_close(error)
            raise
        finally:
            if not self._closed: self._acquisition_wall_time_s += time.perf_counter()-started

    @staticmethod
    def _rgb_statistics(rgb,previous):
        # Row-sized temporaries keep full-frame storage to current/previous.
        total = squared = difference = 0.
        largest = 0
        minimum,maximum = 255,0
        for index,row in enumerate(rgb[:,:,:3]):
            numeric = row.astype(np.float64)
            total += float(numeric.sum())
            squared += float(np.square(numeric).sum())
            minimum = min(minimum,int(row.min()));maximum = max(maximum,int(row.max()))
            if previous is not None:
                delta = np.abs(numeric-previous[index,:,:3])
                difference += float(delta.sum())
                largest = max(largest,int(delta.max()))
        size = rgb.shape[0]*rgb.shape[1]*3
        mean = total/size
        return dict(rgb_mean=mean,rgb_std=float(np.sqrt(max(0.,squared/size-mean*mean))),
            rgb_min=minimum,rgb_max=maximum,
            adjacent_rgb_mean_abs_difference=None if previous is None else difference/size,
            adjacent_rgb_max_abs_difference=None if previous is None else largest)

    def warmup(self,frame_count):
        """Return diagnostic summaries for an exact 1..128 zero-delta renders."""
        self._check_open()
        started = time.perf_counter()
        try:
            self._stage = 'warmup_count'
            if type(frame_count) is not int or not 1 <= frame_count <= 128:
                raise ValueError('Warmup frame count must be a predeclared integer 1..128')
            self._stage = 'warmup_start'
            initial = self._observe()
            summaries = []
            previous = None
            for index in range(frame_count):
                frame,observed = self._frame()
                self._same_observation(initial,observed,'warmup')
                row = self._rgb_statistics(frame['rgb'],previous)
                row.update(index=index,time_s=observed[0][0],world_time_s=observed[1],
                    actual_camera=observed[2],depth_valid_fraction=float(frame['depth_valid'].mean()),
                    render_acquisition_wall_time_s=frame['acquisition_wall_time_s'])
                summaries.append(row)
                previous = frame['rgb']
                self._warmup_frames += 1
            self._stage = 'warmup_finish'
            self._same_observation(initial,self._observe(),'warmup finish')
            report = dict(frame_count=frame_count,renders=summaries,
                measured_state=self._state_dict(initial[0],json_compatible=True),
                final_rgb_sha256=hashlib.sha256(previous).hexdigest(),
                warmup_wall_time_s=time.perf_counter()-started,**QUALIFICATIONS)
            self._warmup_wall_time_s += report['warmup_wall_time_s']
            return report
        except BaseException as error:
            self._warmup_wall_time_s += time.perf_counter()-started
            self._fail_and_close(error)
            raise

    def close(self):
        """Attempt every release independently; errors never imply no leaks."""
        if self._receipt is not None: return deepcopy(self._receipt)
        self._closed = True
        releases = {}
        if self._backend is not None:
            operations = [(f'detach:{name}',lambda name=name:self._backend.detach(name))
                          for name in m2_capture.ANNOTATORS]
            operations.append(('destroy_product',self._backend.destroy_product))
            for label,operation in operations:
                try:
                    operation()
                    releases[label] = True
                except BaseException as error:
                    releases[label] = False
                    self._record_failure(label,error)
        self._receipt = dict(completed=not self._failures,cleanup_complete=all(releases.values()),
            release_attempts=releases,leak_check_performed=False,failures=deepcopy(self._failures),
            read_frames=self._read_frames,warmup_frames=self._warmup_frames,
            acquisition_wall_time_s=self._acquisition_wall_time_s,warmup_wall_time_s=self._warmup_wall_time_s,
            scope='in-memory camera component; no episode or settling qualification',**QUALIFICATIONS)
        return deepcopy(self._receipt)
