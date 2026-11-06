"""Read-only single-camera RGB-D and simulated calibrated Hall observations.

Exact depth/masks and simulated packets remain on the native Linux host. Only
a bounded depth summary and current decoded Hall evidence enter the model.
This sensor observer never supplies a force-control command or native force label.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import inspect
import io
import json
from pathlib import Path

import numpy as np

from .astra_supervisor import canonical, need
from .m2_capture import M2CameraSampler, APPROVED_CAMERA_SHA256, CALIBRATION_PATH, _state, _same_state
from .m2_sensor_probe import CALIBRATION_SHA256
from sim.auscultation_vla_v2.hall import HallCalibration, HallSimulator
from .hall_adc_boundary import lower_endpoint_policy, sensor_input

SCHEMA = 'ausc_astra_rgbd_hall_adc_v1'
CAMERA_FILE = Path(__file__).resolve().parents[1]/'camera_framing_20260923/recommended_camera.json'


def source_paths():
    from . import m2_capture,m2_sensor_probe
    return [Path(__file__),Path(inspect.getsourcefile(sensor_input)),Path(m2_capture.__file__),Path(m2_sensor_probe.__file__),CAMERA_FILE,CALIBRATION_PATH,
        Path(inspect.getsourcefile(HallCalibration)),
        Path(inspect.getsourcefile(HallCalibration(CALIBRATION_PATH.read_bytes())._host.__class__))]


def depth_summary(depth, valid, camera, base, reference):
    need(camera.get('meters_per_unit') == 1. and camera.get('projection') == 'perspective',
         'Metric perspective depth calibration required')
    need(depth.shape == valid.shape == (360,640) and depth.dtype.kind == 'f' and valid.dtype == np.bool_,
         'Native depth and validity shapes/types differ')
    need(np.array_equal(valid,np.isfinite(depth)&(depth>0)), 'Depth validity mask differs from raw optical Z')
    values = depth[valid]
    need(values.size > 0, 'No valid depth observations')
    # A 32x18 spatial median grid preserves scene layout without transferring
    # the exact 640x360 numeric map or inventing values at missing pixels.
    grid=[];counts=[]
    for y in range(0,360,20):
        row=[];count=[]
        for x in range(0,640,20):
            cell=depth[y:y+20,x:x+20][valid[y:y+20,x:x+20]]
            row.append(None if not cell.size else float(np.median(cell)));count.append(int(cell.size))
        grid.append(row);counts.append(count)
    K = camera['intrinsics']
    world_optical=np.asarray(camera['world_from_camera_usd'],float)@np.diag([1.,-1.,-1.,1.])
    base_optical=np.linalg.inv(np.asarray(base,float))@world_optical
    target=np.linalg.inv(base_optical)@np.r_[np.asarray(reference,float)[:3,3],1.]
    probe=dict(source='native_depth_at_prescribed_pose_projection', anatomical_target_estimator=False,
        valid=False, prescribed_reference_used_to_choose_pixel=True,
        may_be_occluded=True,patient_surface_verified=False,pixel_validity_verifies_anatomical_surface=False,
        ray_surface_semantics='first visible depth surface; may be robot, tool, or other foreground geometry')
    if target[2] > 0:
        uv=np.array([K['fx']*target[0]/target[2]+K['cx'],K['fy']*target[1]/target[2]+K['cy']])
        probe['projection_uv_px']=uv.tolist()
        if 0 <= uv[0] < 640 and 0 <= uv[1] < 360:
            x,y=np.rint(uv).astype(int)
            if x < 640 and y < 360 and valid[y,x]:
                z=float(depth[y,x]);point=base_optical@np.array([(x-K['cx'])/K['fx']*z,(y-K['cy'])/K['fy']*z,z,1.])
                probe.update(valid=True,pixel_xy=[int(x),int(y)],optical_z_m=z,point_base_m=point[:3].tolist())
    return dict(units='m',semantics='metric_optical_z',annotator='distance_to_image_plane',
        valid_mask_rule='isfinite(depth_m) and depth_m > 0',valid_count=int(values.size),
        invalid_count=int(depth.size-values.size),minimum_m=float(values.min()),maximum_m=float(values.max()),
        quantiles_m=np.quantile(values,[.05,.5,.95]).tolist(),
        grid_shape=[18,32],grid_cell_pixels=[20,20],grid_reduction='valid-pixel median; null means no valid pixels',
        median_grid_m=grid,valid_count_grid=counts,base_from_optical=base_optical.tolist(),
        prescribed_projection_probe=probe,target_source='given_contact_pose; not replaced by depth probe')


class AstraSensors:
    def __init__(self,env,output,*,camera_bytes=None,calibration_bytes=None,backend_factory=None,preload_context=None):
        self.env=env;self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False)
        self.camera=None;self._hall_file=None;self.failures=[];self._closed=False
        self._latest=None;self._last_row=None;self._last_compression=None;self._frame=0
        self._window=[]
        camera_bytes=CAMERA_FILE.read_bytes() if camera_bytes is None else camera_bytes
        calibration_bytes=CALIBRATION_PATH.read_bytes() if calibration_bytes is None else calibration_bytes
        try:
            need(hashlib.sha256(camera_bytes).hexdigest()==APPROVED_CAMERA_SHA256,'Approved camera bytes changed')
            need(hashlib.sha256(calibration_bytes).hexdigest()==CALIBRATION_SHA256,'Pinned Hall calibration bytes changed')
            self.calibration=HallCalibration(calibration_bytes)
            nominal=float(json.loads(calibration_bytes)['force_model']['preload_n'])
            if preload_context is None:
                preload_context=dict(nominal_preload_N=nominal,native_authoring_readback_verified=False,
                    source='pinned calibration only; native authoring context not supplied',sources={})
            need(preload_context.get('nominal_preload_N') == nominal,'Configured/calibrated preload differs')
            host=self.calibration._host
            self.hall_interpretation=dict(quantity='calibrated spring-load estimate including nominal preload',
                nominal_preload_N=nominal,preload_context=deepcopy(preload_context),
                rest_value_indicates_patient_normal_contact=False,decoded_spring_load_is_patient_normal_force=False,
                preload_subtracted=False,WARN_means='outside_target_force_window; can include unloaded rest',
                WARN_alone_is_safety_trip=False,
                status_thresholds_N=dict(target_force_N=host.target_force_n,target_tolerance_N=host.target_tolerance_n,
                    screened_limit_N=host.max_screened_force_n,hard_limit_N=host.hard_limit_force_n),
                status_source='existing ForceCalibration.classify; unchanged screened/hard classifications and local gates')
            self.hall=HallSimulator(self.calibration,sample_period_s=.01,seed=0)
            self.hall_zero_policy=lower_endpoint_policy(self.calibration)
            self._hall_zero_count=0;self._hall_zero_max_delta_m=0.
            helper=Path(inspect.getsourcefile(sensor_input))
            self.hall_zero_helper=dict(file=str(helper),sha256=hashlib.sha256(helper.read_bytes()).hexdigest())
            (self.output/'calibration.json').write_bytes(calibration_bytes)
            (self.output/'approved_camera.json').write_bytes(camera_bytes)
            self._hall_file=(self.output/'hall_samples.jsonl').open('x')
            self.camera=M2CameraSampler(env,camera_bytes=camera_bytes,backend_factory=backend_factory,
                native_image_size=(640,360))
            manifest=dict(schema=SCHEMA,single_camera_path='/World/RobotCam',native_image_size=[640,360],
                camera_sha256=APPROVED_CAMERA_SHA256,calibration_sha256=CALIBRATION_SHA256,
                hall_source='simulated_adc',hall_model='existing HallSimulator RP2040 12-bit eight-read average to u16',
                hall_nominal_hz=100,hall_noise_std_v=0.,hall_bias_v=0.,hall_latency_s=0.,
                bench_verified=self.calibration.bench_verified,force_label_used=False,
                hall_interpretation=deepcopy(self.hall_interpretation),
                hall_zero_conversion=deepcopy(self.hall_zero_policy),
                hall_zero_conversion_helper=deepcopy(self.hall_zero_helper),
                source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths()},
                exact_raw_storage='native Linux only; no depth arrays in HTTP payload',
                local_force_controller='unchanged native axial proxy; this observer supplies no commands')
            (self.output/'manifest.json').write_bytes(canonical(manifest)+b'\n')
            self.manifest_sha256=hashlib.sha256((self.output/'manifest.json').read_bytes()).hexdigest()
            self.observe_hall(env.state(),len(env.rows))
        except BaseException as error:
            self.failures.append(repr(error))
            self.close();raise

    def observe_hall(self,state,physics_row_end_exclusive):
        try:return self._observe_hall(state,physics_row_end_exclusive)
        except BaseException as error:
            self.failures.append('Hall observer: '+repr(error));raise

    def _observe_hall(self,state,physics_row_end_exclusive):
        need(not self._closed,'Astra sensor observer closed')
        _same_state(_state(state),_state(self.env.state()),'Hall source native state')
        stamp=float(state['time_s']);compression=float(state['compression_m'])
        observed_input=sensor_input(self.calibration,compression)
        need(type(physics_row_end_exclusive) is int and physics_row_end_exclusive == len(self.env.rows)
             and abs(stamp-physics_row_end_exclusive/240.) <= 1e-8,'Hall/native physics binding differs')
        if self._last_row == physics_row_end_exclusive:
            need(self._last_compression == compression and self._latest['observed_at_s'] == stamp,
                 'Same-row Hall displacement/time changed')
            return deepcopy(self._latest)
        need(self._last_row is None and physics_row_end_exclusive == 0 or
             self._last_row is not None and physics_row_end_exclusive == self._last_row+1,
             'Hall observer requires consecutive actual native rows')
        if observed_input['sensor_compression_m']!=compression:
            self._hall_zero_count+=1
            self._hall_zero_max_delta_m=max(self._hall_zero_max_delta_m,-compression)
        self.hall.update(stamp,observed_input['sensor_compression_m']);packet=self.hall.read(stamp)
        decoded=self.calibration.decode(packet,now_s=stamp)
        fresh=self._latest is None or self._latest['packet']['sample_time_s'] != packet.sample_time_s
        evidence=dict(packet=asdict(packet),decoded=asdict(decoded),observed_at_s=stamp,
            physics_row_end_exclusive=physics_row_end_exclusive,source_compression_m=compression,
            observation_age_s=stamp-packet.sample_time_s,bench_verified=self.calibration.bench_verified,
            force_label_used=False,actual_adc_measurement=False,calibration_sha256=self.calibration.sha256,
            hall_sensor_input=observed_input)
        self._hall_file.write(canonical(evidence).decode()+'\n');self._hall_file.flush()
        if fresh:self._window.append(asdict(decoded))
        self._latest=evidence;self._last_row=physics_row_end_exclusive;self._last_compression=compression
        return deepcopy(evidence)

    def capture(self,state,physics_row_end_exclusive,reference_pose_base):
        try:
            hall=self.observe_hall(state,physics_row_end_exclusive)
            hall['interpretation']=deepcopy(self.hall_interpretation)
            rgb,depth,valid=self.camera.capture(state)
            raw_file=f'frame_{self._frame:04d}.npz';self._frame+=1
            with (self.output/raw_file).open('xb') as stream:
                np.savez_compressed(stream,rgb=rgb,depth_m=depth,depth_valid=valid)
            summary=depth_summary(depth,valid,self.camera.actual_camera_at_setup,self.env.base,reference_pose_base)
            from PIL import Image
            output=io.BytesIO();Image.fromarray(rgb[:,:,:3]).save(output,format='PNG')
            history=self._window
            hall['since_previous_boundary']=dict(acquisitions=len(history),
                min_voltage_v=None if not history else min(x['hall_voltage_v'] for x in history),
                max_voltage_v=None if not history else max(x['hall_voltage_v'] for x in history),
                min_spring_force_estimate_N=None if not history else min(x['spring_force_estimate_n'] for x in history),
                max_spring_force_estimate_N=None if not history else max(x['spring_force_estimate_n'] for x in history))
            bundle=dict(schema=SCHEMA,single_camera_path='/World/RobotCam',depth=summary,hall=hall,
                camera=deepcopy(self.camera.actual_camera_at_setup),camera_sha256=APPROVED_CAMERA_SHA256,
                sensor_manifest_sha256=self.manifest_sha256,physics_row_end_exclusive=physics_row_end_exclusive,
                observed_at_s=float(state['time_s']),raw_rgbd=dict(file=raw_file,
                    sha256=hashlib.sha256((self.output/raw_file).read_bytes()).hexdigest(),exact_arrays_transferred=False))
            self._window=[]
            return bundle,output.getvalue()
        except BaseException as error:
            self.failures.append(repr(error));raise

    def close(self):
        if not self._closed:
            if self.camera is not None:
                receipt=self.camera.finalize()
                if not receipt['completed']:self.failures.extend(receipt['failures'])
            if self._hall_file is not None:self._hall_file.close()
            self._closed=True
        receipt=dict(completed=not self.failures,failures=list(self.failures),frames=self._frame,schema=SCHEMA)
        if hasattr(self,'hall_zero_policy'):
            receipt.update(hall_zero_conversion=dict(**self.hall_zero_policy,
                normalized_ticks=self._hall_zero_count,max_delta_m=self._hall_zero_max_delta_m),
                hall_zero_conversion_helper=deepcopy(self.hall_zero_helper))
        (self.output/'teardown.json').write_bytes(canonical(receipt)+b'\n')
        return receipt
