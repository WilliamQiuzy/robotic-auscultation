"""Bounded rigid-torso RGB-D tracking candidate, with explicit initial annotation.

Assumes a fixed calibrated camera and a locally rigid torso over a short motion
window. It cannot track articulated breathing/skin deformation or disambiguate
an identical replacement. Polygon and target pixels are supplied once, before
motion. Hidden target positions are transported only while broad measured
surface support remains geometrically observable and appearance-consistent.
No simulation transforms or semantic IDs are accepted. Thresholds are
engineering gates; real camera validation remains required.
"""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from .tracking import SurfaceTracker,TrackingLost,_fail_closed
from .observations import finite
from .rigid_registration import rgbd_cloud,register


class BodyTracker:
    def __init__(self,intrinsics,camera_to_base):
        # Shared checked camera/frame/local-plane geometry; no image patch matcher.
        self._geometry=SurfaceTracker(intrinsics,camera_to_base,patch_radius=5)
        self._lost=True;self._revision=-1;self._stamp=None;self._now=None;self._rgb_stamp=None
        self._shape=None
        self.loss_reason='explicit torso annotation required'

    @property
    def lost(self):return self._lost

    def _reject(self,reason):
        self._lost=True;self.loss_reason=reason
        raise TrackingLost(reason+'; explicit torso reinitialization required')

    def _frame(self,rgb,depth,valid,rgb_time_s,depth_time_s,now_s,*,reinitialize=False):
        try:
            result=self._geometry._frame(rgb,depth,valid,rgb_time_s,depth_time_s,now_s)
        except TrackingLost as error:self._reject(str(error))
        if self._shape is not None and result[1].shape!=self._shape:
            self._reject('camera image dimensions changed; new calibration required')
        if self._stamp is not None:
            if depth_time_s<=self._stamp or rgb_time_s<=self._rgb_stamp or now_s<self._now:
                self._reject('source or consumer time reversed/repeated')
            if not reinitialize and max(depth_time_s-self._stamp,rgb_time_s-self._rgb_stamp)>.15:
                self._reject('excessive camera frame gap')
        return result

    def _estimates(self,T,stamp,previous=None):
        estimates={}
        for key,(p,n) in self._targets.items():
            position=T[:3,:3]@p+T[:3,3];normal=T[:3,:3]@n
            velocity=np.zeros(3) if previous is None else (position-previous[key])/(stamp-self._stamp)
            if np.linalg.norm(velocity)>.6:self._reject('target speed exceeds tracking scope')
            self._geometry._revision=self._revision
            estimates[key]=self._geometry._estimate(position,normal,velocity,.9,stamp)
        return estimates

    @_fail_closed
    def initialize(self,rgb,depth_m,depth_valid,target_pixels,support_polygon_xy,*,rgb_time_s,depth_time_s,now_s):
        _,depth,valid,_,stamp=self._frame(rgb,depth_m,depth_valid,rgb_time_s,depth_time_s,now_s,reinitialize=True)
        polygon=np.asarray(support_polygon_xy,float)
        if polygon.ndim!=2 or polygon.shape[1]!=2 or len(polygon)<3 or not np.isfinite(polygon).all():
            self._reject('invalid annotated torso polygon')
        h,w=depth.shape
        if np.any(polygon<1) or np.any(polygon[:,0]>=w-1) or np.any(polygon[:,1]>=h-1):
            self._reject('torso annotation outside camera')
        if not isinstance(target_pixels,dict) or not target_pixels:self._reject('missing initial target pixels')
        targets={}
        for key,pixel in target_pixels.items():
            if not isinstance(key,str) or not key:self._reject('invalid target identity')
            pixel=finite(pixel,(2,),'target pixel')
            if cv2.pointPolygonTest(polygon.astype(np.float32),tuple(pixel),False)<0:
                self._reject('target outside annotated torso')
            try:p,n,_,_=self._geometry._surface(depth,valid,pixel)
            except TrackingLost as error:self._reject(str(error))
            targets[key]=(p,n)
        points,normals,colors=rgbd_cloud(rgb,depth,valid,self._geometry._k,polygon)
        register(points,normals,points,normals)  # Reject unobservable initial shape.
        self._targets=targets;self._reference=(points,normals,colors)
        self._polygon=polygon.copy();self._T=np.eye(4);self._revision+=1
        self._shape=depth.shape
        result=self._estimates(self._T,stamp)
        self._stamp=stamp;self._rgb_stamp=rgb_time_s;self._now=now_s;self._lost=False;self.loss_reason=None
        return result

    @_fail_closed
    def update(self,rgb,depth_m,depth_valid,*,rgb_time_s,depth_time_s,now_s):
        if self.lost:self._reject('tracker lost')
        _,depth,valid,_,stamp=self._frame(rgb,depth_m,depth_valid,rgb_time_s,depth_time_s,now_s)
        points,normals,colors=rgbd_cloud(rgb,depth,valid,self._geometry._k,self._polygon)
        p,pn,pc=self._reference
        try:fit=register(p,pn,points,normals,initial=self._T)
        except ValueError as error:self._reject('surface registration rejected: '+str(error))
        before=pc[fit['reference_indices']].astype(float)
        after=colors[fit['current_indices']].astype(float)
        difference=np.mean(np.abs(before-after),axis=1)
        if np.median(difference)>15 or np.quantile(difference,.9)>40:
            self._reject('RGB appearance disagrees with registered torso')
        T=fit['transform'];delta=T[:3,:3]@self._T[:3,:3].T
        if np.linalg.norm(Rotation.from_matrix(delta).as_rotvec())>.12:
            self._reject('body rotation exceeds tracking scope')
        old={key:self._T[:3,:3]@p+self._T[:3,3] for key,(p,_) in self._targets.items()}
        result=self._estimates(T,stamp,old)
        self._T=T;self._stamp=stamp;self._rgb_stamp=rgb_time_s;self._now=now_s
        self.last_quality={k:v for k,v in fit.items() if k not in ('transform','reference_indices','current_indices')}
        return result
