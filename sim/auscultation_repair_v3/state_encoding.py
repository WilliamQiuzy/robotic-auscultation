"""Task-specific physical-domain state tokens; never fit held-out states.

This changes the VLA input representation and requires a new trained checkpoint.
It is not a drop-in change to a frozen V2 model. Geometry uses an invertible
asinh transform to retain fine contact resolution and broad free-space coverage.
"""
import numpy as np


class StateCodec:
    bins=4096
    def __init__(self,joint_low,joint_high):
        lo=np.asarray(joint_low,dtype=float);hi=np.asarray(joint_high,dtype=float)
        if lo.shape!=(7,) or hi.shape!=(7,) or not np.isfinite([lo,hi]).all() or np.any(hi<=lo):
            raise ValueError('Invalid robot joint domain')
        self.low=np.r_[lo,[-2.]*7,[-1.5]*3,[-1.]*3,0.,0.,0.,[-.5]*3,[-1.]*6]
        self.high=np.r_[hi,[2.]*7,[1.5]*3,[1.]*3,3.3,16.,1.,[.5]*3,[1.]*6]
        self.scale=np.ones(32);self.scale[14:17]=.02;self.scale[23:26]=.01
        self.nonlinear=np.zeros(32,bool);self.nonlinear[14:17]=True;self.nonlinear[23:26]=True

    def spec(self):
        return dict(schema='wam_physical_asinh_state_4096_v1',bins=self.bins,fitted_on='none',
                    low=self.low.tolist(),high=self.high.tolist(),asinh_scale=self.scale.tolist(),
                    asinh_dimensions=np.flatnonzero(self.nonlinear).tolist(),outside_domain='reject')

    def _transform(self,x):
        return np.where(self.nonlinear,np.arcsinh(x/self.scale),x)

    def encode(self,state):
        x=np.asarray(state,dtype=np.float64)
        if x.ndim<1 or x.shape[-1]!=32 or not np.isfinite(x).all():raise ValueError('Invalid raw state')
        if np.any(x<self.low) or np.any(x>self.high):raise ValueError('State outside declared physical domain')
        low=self._transform(self.low);high=self._transform(self.high)
        return (2*(self._transform(x)-low)/(high-low)-1).astype(np.float32)

    def discrete(self,state):
        return np.rint((self.encode(state).astype(float)+1)*((self.bins-1)/2)).astype(np.int32)

    def decode(self,indices):
        a=np.asarray(indices)
        if a.ndim<1 or a.shape[-1]!=32 or a.dtype.kind not in 'iu' or np.any(a<0) or np.any(a>=self.bins):
            raise ValueError('Invalid state bin')
        low=self._transform(self.low);high=self._transform(self.high)
        x=low+a/(self.bins-1)*(high-low)
        return np.where(self.nonlinear,np.sinh(x)*self.scale,x)

    def prompt(self,instruction,state):
        if not isinstance(instruction,str) or not instruction.strip():raise ValueError('Missing instruction')
        a=self.discrete(state)
        if a.shape!=(32,):raise ValueError('Prompt requires one state')
        cleaned=instruction.strip().replace('_',' ').replace('\n',' ')
        return f"Task: {cleaned}, State: {' '.join(map(str,a))};\nAction: "

    def tokenize(self,tokenizer,instruction,state,*,max_len=200):
        ids=tokenizer.encode(self.prompt(instruction,state),add_bos=True)
        if not ids or len(ids)>max_len:raise ValueError('Task/state token budget exceeded')
        mask=np.arange(max_len)<len(ids)
        return np.pad(np.asarray(ids,np.int32),(0,max_len-len(ids))),mask
