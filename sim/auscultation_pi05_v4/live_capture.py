"""Versioned live capture adds measured body transform and same-frame retargeting."""
from dataclasses import asdict
import json
import numpy as np
from sim.auscultation_vla_v2.live_capture import LiveCapture as OriginalCapture
from sim.auscultation_vla_v2.observations import Observation,RobotState,encode
from sim.auscultation_vla_v2.multimodal import INSTRUCTIONS


class CachedTracker:
    def __init__(self,tracker):self.tracker=tracker;self.estimates={}
    def __getattr__(self,key):return getattr(self.tracker,key)
    def initialize(self,*args,**kwargs):
        self.estimates=self.tracker.initialize(*args,**kwargs);return self.estimates
    def update(self,*args,**kwargs):
        self.estimates=self.tracker.update(*args,**kwargs);return self.estimates


class LiveCapture(OriginalCapture):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.tracker=CachedTracker(self.tracker)

    def capture(self,state,phase,target,physics_row,*,eligible,subphase):
        stamp=state['time_s']
        if self.rows and abs(self.rows[-1]['time_s']-stamp)<1e-10:
            # Reuse one acquired RGB-D frame and its cached multi-target fit.
            # Calling BodyTracker.update again would rightly reject its timestamp.
            old=self.rows[-1]
            if not np.array_equal(state['q'],old['q']) or physics_row!=old['physics_row']:
                raise ValueError('Same-time retarget changed physical state')
            with np.load(self.out/old['raw_file'],allow_pickle=False) as a:
                rgb,depth,valid=(a[k] for k in ('rgb','depth_m','depth_valid'))
            estimate=self.tracker.estimates[target];packet=self.hall.read(stamp);wrist=self.kin.fk(state['q'])
            robot=RobotState(state['q'],state['dq'],wrist[:3,3],wrist[:3,:3],packet,stamp)
            obs=Observation(rgb,depth,valid,self.K,self.camera_to_base,stamp,stamp,robot,estimate,INSTRUCTIONS[phase])
            encoded=encode(obs,now_s=stamp,hall_calibration=self.cal)
            target_record=asdict(estimate)
            target_record={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in target_record.items()}
            row=dict(old,index=len(self.rows),target=target,phase=phase,eligible=bool(eligible),subphase=subphase,
                state=encoded['state'].tolist(),target_estimate=target_record,prompt=INSTRUCTIONS[phase],
                reused_camera_frame=True)
            self.rows.append(row)
            with (self.out/'frames.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        else:
            super().capture(state,phase,target,physics_row,eligible=eligible,subphase=subphase)
        T=self.camera_to_base@self.tracker._T@np.linalg.inv(self.camera_to_base)
        self.rows[-1]['body_reference_to_current_base']=T.tolist()
        with (self.out/'body_poses.jsonl').open('a') as f:
            f.write(json.dumps(dict(frame_index=self.rows[-1]['index'],time_s=stamp,
                body_reference_to_current_base=T.tolist(),source='measured RGB-D rigid registration'))+'\n')
